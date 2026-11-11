"""Ragged shard dataset; targets and provenance never enter observations."""

from bisect import bisect_right
from collections import OrderedDict
from multiprocessing.util import Finalize
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, Sampler


class NativeShard(Dataset):
    """In-memory shard reader, also used by the bounded worker cache."""

    def __init__(self, path, target_columns=None):
        self.path = Path(path)
        self.observations = pq.read_table(self.path / 'observations.parquet')
        self.targets = pq.read_table(self.path / 'targets.parquet',
                                     columns=list(target_columns) if target_columns is not None else None)
        if len(self.observations) != len(self.targets):
            raise ValueError('observations/targets length mismatch')
        # Convert Arrow's contiguous list buffers once per shard. Returning
        # NumPy views avoids rebuilding Python objects for every channel/event.
        self.buffers, self.offsets = {}, {}
        widths = {'xyz_mm': 3, 'bounds_mm': 5, 'xy_mm': 2, 'sigma_xy_mm': 2}
        for name in ['pixel', 'bar', 'tracker_hypothesis']:
            struct = self.observations[name].combine_chunks()
            self.buffers[name] = {}
            for key in struct.type:
                field = struct.field(key.name)
                offsets = field.offsets.to_numpy(zero_copy_only=False)
                if name in self.offsets and not np.array_equal(self.offsets[name], offsets):
                    raise ValueError('ragged attributes have inconsistent channel counts')
                self.offsets[name] = offsets
                flat = field.flatten()
                width = widths.get(key.name)
                dtype = np.float32 if width or key.name == 'edep_proxy_MeV' else np.int64
                if width and len(flat):
                    values = flat.flatten().to_numpy(zero_copy_only=False).astype(dtype, copy=False).reshape(-1, width)
                else:
                    values = flat.to_numpy(zero_copy_only=False).astype(dtype, copy=False)
                    if width:
                        values = values.reshape(-1, width)
                self.buffers[name][key.name] = values
        self.states = {key: np.asarray(self.observations[key].to_pylist())
                       for key in self.observations.column_names if key not in self.buffers}

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, index):
        observation = {name: {key: values[self.offsets[name][index]:self.offsets[name][index+1]]
                              for key, values in fields.items()} for name, fields in self.buffers.items()}
        observation.update({key: values[index] for key, values in self.states.items()})
        return {'observations': observation,
                'targets': self.targets.slice(index, 1).to_pylist()[0]}


class CachedNativeDataset(Dataset):
    """Lazy, process-owned LRU of shards staged from EOS to local scratch.

    Construction reads Parquet footers only. Each worker owns at most
    ``max_shards`` local observation/target pairs and loaded NativeShard objects.
    Event views in the current batch and DataLoader prefetch buffers may outlive
    an eviction; the limit applies to the cache, not total process memory.
    Provenance and ideal truth files are never staged.
    """

    FILES = ('observations.parquet', 'targets.parquet')

    def __init__(self, paths, cache_dir=None, max_shards=1,
                 target_columns=('label', 'label_status')):
        if max_shards < 1:
            raise ValueError('max_shards must be positive')
        self.paths = tuple(Path(p).resolve() for p in paths)
        if not self.paths:
            raise ValueError('at least one shard is required')
        self.max_shards = max_shards
        self.target_columns = tuple(target_columns)
        self.cache_dir = str(cache_dir or os.environ.get('_CONDOR_SCRATCH_DIR') or tempfile.gettempdir())
        if Path(self.cache_dir).resolve().as_posix().startswith(('/eos/', '/afs/')):
            raise ValueError('cache_dir must use local scratch, not EOS or AFS')
        counts = []
        for path in self.paths:
            lengths = [pq.read_metadata(path / name).num_rows for name in self.FILES]
            if lengths[0] != lengths[1]:
                raise ValueError(f'observations/targets length mismatch: {path}')
            counts.append(lengths[0])
        self.boundaries = np.r_[0, np.cumsum(counts)].tolist()
        self._reset_process_state()

    def _reset_process_state(self):
        self._owner_pid = None
        self._root = None
        self._finalizer = None
        self._cache = OrderedDict()
        self._stats = dict(hits=0, misses=0, evictions=0, staged_bytes=0,
                           resident_bytes=0, peak_resident_bytes=0,
                           stage_seconds=0., decode_seconds=0.)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Spawn workers must not receive the parent's arrays or scratch owner.
        for key in ('_owner_pid', '_root', '_finalizer'):
            state[key] = None
        state['_cache'] = OrderedDict()
        state['_stats'] = dict.fromkeys(self._stats, 0)
        return state

    def _ensure_owner(self):
        if self._owner_pid != os.getpid():
            # Under fork the parent retains its own files and finalizer.
            self._reset_process_state()
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            self._root = Path(tempfile.mkdtemp(prefix=f'pinpoint-{os.getpid()}-', dir=self.cache_dir))
            self._owner_pid = os.getpid()
            self._finalizer = Finalize(self, shutil.rmtree, args=(self._root,),
                                       kwargs={'ignore_errors': True}, exitpriority=10)

    def __len__(self):
        return self.boundaries[-1]

    def __getitem__(self, index):
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        self._ensure_owner()
        shard_id = bisect_right(self.boundaries, index) - 1
        if shard_id in self._cache:
            self._stats['hits'] += 1
            self._cache.move_to_end(shard_id)
        else:
            self._stats['misses'] += 1
            if len(self._cache) == self.max_shards:
                _, (old, size) = self._cache.popitem(last=False)
                shutil.rmtree(old.path)
                self._stats['resident_bytes'] -= size
                self._stats['evictions'] += 1
                del old
            local = self._root / str(shard_id)
            local.mkdir()
            started = time.perf_counter()
            size = 0
            try:
                for name in self.FILES:
                    source = self.paths[shard_id] / name
                    size += source.stat().st_size
                    shutil.copyfile(source, local / name)
                self._stats['stage_seconds'] += time.perf_counter() - started
                started = time.perf_counter()
                loaded = NativeShard(local, target_columns=self.target_columns)
            except Exception:
                shutil.rmtree(local)
                raise
            self._stats['decode_seconds'] += time.perf_counter() - started
            self._stats['staged_bytes'] += size
            self._stats['resident_bytes'] += size
            self._stats['peak_resident_bytes'] = max(self._stats['peak_resident_bytes'], self._stats['resident_bytes'])
            self._cache[shard_id] = (loaded, size)
        return self._cache[shard_id][0][index - self.boundaries[shard_id]]

    def cache_stats(self):
        return {**self._stats, 'pid': self._owner_pid,
                'root': str(self._root) if self._root else None, 'resident_shards': len(self._cache)}

    def close(self):
        if self._owner_pid == os.getpid() and self._finalizer is not None:
            self._cache.clear()
            self._finalizer()
            self._root = None
            self._owner_pid = None
            self._stats['resident_bytes'] = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class ShardBatchSampler(Sampler):
    """Shuffle events within bounded shard windows; visit every event once.

    The window should fit the per-worker cache. Pure-class shards require a
    window covering all classes (or mixed-class shards) for training. A one-shard
    window is useful for an I/O benchmark, not a class-mixing training policy.
    """

    def __init__(self, dataset, batch_size, window_shards=1, seed=42):
        if batch_size < 1 or not 1 <= window_shards <= dataset.max_shards:
            raise ValueError('positive batch size and window within cache capacity required')
        self.boundaries = dataset.boundaries
        self.batch_size, self.window_shards, self.seed = batch_size, window_shards, seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _windows(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(len(self.boundaries) - 1)
        for start in range(0, len(order), self.window_shards):
            yield rng, order[start:start + self.window_shards]

    def __iter__(self):
        for rng, shards in self._windows():
            indices = np.concatenate([np.arange(self.boundaries[s], self.boundaries[s+1]) for s in shards])
            rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                yield indices[start:start + self.batch_size].tolist()

    def __len__(self):
        return sum((sum(self.boundaries[s+1] - self.boundaries[s] for s in shards) + self.batch_size - 1)
                   // self.batch_size for _, shards in self._windows())


def collate_native(events):
    result = {}
    for name, field, width in [('pixel', 'xyz_mm', 3), ('bar', 'bounds_mm', 5),
                               ('tracker_hypothesis', 'xy_mm', 2)]:
        arrays = [np.asarray(e['observations'][name][field], dtype='float32').reshape(-1, width) for e in events]
        offsets = np.r_[0, np.cumsum([len(a) for a in arrays])]
        result[name] = {'values': torch.from_numpy(np.concatenate(arrays, axis=0)),
                        'offsets': torch.from_numpy(offsets)}
    result['has_response'] = torch.from_numpy(np.asarray([e['observations']['has_response'] for e in events], dtype=bool))
    for key in ['outside_acceptance', 'fit_attempted', 'momentum_censored']:
        if key in events[0]['observations']:
            result[key] = torch.from_numpy(np.asarray([e['observations'][key] for e in events]))
    # Preserve every observation attribute as ragged tensors, including IDs,
    # intervals, PE and uncertainties; no truncation or hidden support loss.
    for name in ['pixel', 'bar', 'tracker_hypothesis']:
        for key in events[0]['observations'][name]:
            if key == {'pixel': 'xyz_mm', 'bar': 'bounds_mm', 'tracker_hypothesis': 'xy_mm'}[name]:
                continue
            dtype = np.float32 if key in ['edep_proxy_MeV', 'sigma_xy_mm'] else np.int64
            arrays = [np.asarray(e['observations'][name][key], dtype=dtype) for e in events]
            if key == 'sigma_xy_mm':
                arrays = [a.reshape(-1, 2) for a in arrays]
            result[name][key] = torch.as_tensor(np.concatenate(arrays, axis=0))
    targets = {'label': torch.as_tensor([e['targets']['label'] for e in events], dtype=torch.long)}
    if 'label_status' in events[0]['targets']:
        targets['label_status'] = [e['targets']['label_status'] for e in events]
    return {'observations': result, 'targets': targets}
