"""Profile real Train observations, temporary caches and training/validation work.

The stream campaign must run in HTCondor. The 256-event campaign is an
interactive microbenchmark. Profiling uses zero-learning-rate AdamW: it measures
the training operations without selecting a model or using evaluation samples.
"""

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import platform
import tempfile
import threading
import time

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import torch
from torch.utils.data import DataLoader

from analysis.flavour.inventory import write_json
from analysis.flavour.s2.dataset import SpatialBatchSampler
from analysis.flavour.s3.dataset import collate_planes
from analysis.flavour.s3.models_repair import build_model
from analysis.flavour.s4.stage_inputs import stage_shards
from analysis.flavour.v2.data import (
    TIMING_NAMES, PackedPlaneDataset, TimedPlaneDataset, build_packed_cache,
    collate_profile, verify_cache,
)
from analysis.flavour.v2.metadata import representation_metadata


def init_worker(_):
    torch.set_num_threads(1)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)


class MemoryMonitor:
    """Sample simultaneous parent/child RSS; shared mappings can be counted twice."""

    def __enter__(self):
        self.stop = threading.Event()
        self.peak = {'parent_RSS_MiB': 0., 'children_RSS_MiB': 0., 'aggregate_RSS_MiB': 0.}
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def sample(self):
        process = psutil.Process()
        while not self.stop.is_set():
            try:
                parent = process.memory_info().rss / 2**20
                children = 0.
                # lxplus restricts unrelated /proc entries. Walk only this
                # allocation's descendant PIDs instead of scanning /proc/1.
                pending = [process.pid]
                descendants = set()
                while pending:
                    pid = pending.pop()
                    try:
                        child_ids = [int(value) for value in Path(f'/proc/{pid}/task/{pid}/children').read_text().split()]
                    except FileNotFoundError:
                        continue
                    for child_id in child_ids:
                        if child_id not in descendants:
                            descendants.add(child_id)
                            pending.append(child_id)
                for pid in descendants:
                    try:
                        children += psutil.Process(pid).memory_info().rss / 2**20
                    except psutil.NoSuchProcess:
                        continue
                for key, value in zip(self.peak, [parent, children, parent + children]):
                    self.peak[key] = max(self.peak[key], value)
            except psutil.NoSuchProcess:
                pass
            self.stop.wait(.2)

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()


def device_batch(batch, device):
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def shutdown_loader(loader):
    close_fn = getattr(loader, 'close', None)
    if callable(close_fn):
        close_fn()
        return
    iterator = getattr(loader, '_iterator', None)
    if iterator is not None:
        iterator._shutdown_workers()
        loader._iterator = None


def run_case(data, config, workers, batch_size, passes, training, scenario, device, dtype):
    """Wall times use explicit synchronization, with data preparation overlapped."""
    torch.manual_seed(42)
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0., weight_decay=.01)
    autocast = lambda: (torch.autocast('cuda', dtype=dtype) if dtype is not None else nullcontext())
    sampler = SpatialBatchSampler(data, batch_size, 42, shuffle=training)
    options = {'batch_sampler': sampler, 'num_workers': workers, 'collate_fn': collate_profile,
               'pin_memory': True, 'generator': torch.Generator().manual_seed(42)}
    if workers:
        options.update(multiprocessing_context='spawn', worker_init_fn=init_worker,
                       prefetch_factor=1, persistent_workers=True)
    loader = DataLoader(data, **options)
    # First CUDA context/kernels are outside the throughput interval. Worker
    # startup and initial shard decoding remain visible in each cold pass.
    warm_started = time.perf_counter()
    informative = (data.metadata.n_pixel > 0) & (data.metadata.n_bar > 0)
    warm_index = int(np.flatnonzero(informative)[0])
    warm = device_batch(collate_planes([data[warm_index]]), device)
    model.train(training)
    with torch.set_grad_enabled(training), autocast():
        loss = torch.nn.functional.cross_entropy(model(warm), warm['label'])
    if training:
        loss.backward()
        optimizer.step()
    gradient_squared = {name: float(parameter.grad.detach().float().square().sum())
                        for name, parameter in model.named_parameters() if parameter.grad is not None
                        and name in ['input.0.weight', 'local.0.q.weight', 'latent_read.q.weight']}
    optimizer.zero_grad(set_to_none=True)
    del loss, warm
    torch.cuda.synchronize()
    warm_seconds = time.perf_counter()-warm_started
    torch.cuda.reset_peak_memory_stats()
    rows, summaries = [], []
    try:
        with MemoryMonitor() as memory:
            for epoch in range(passes):
                sampler.epoch = epoch
                started = time.perf_counter()
                iterator = iter(loader)
                startup = time.perf_counter() - started
                seen = []
                epoch_rows = []
                for step in range(len(sampler)):
                    started_wait = time.perf_counter()
                    batch = next(iterator)
                    wait = time.perf_counter() - started_wait
                    profile = batch.pop('_profile')
                    indexes = batch['index'].tolist()
                    seen.extend(indexes)
                    started_transfer = time.perf_counter()
                    gpu = device_batch(batch, device)
                    torch.cuda.synchronize()
                    transfer = time.perf_counter() - started_transfer
                    started_compute = time.perf_counter()
                    optimizer.zero_grad(set_to_none=True)
                    with torch.set_grad_enabled(training), autocast():
                        logits = model(gpu)
                        loss = torch.nn.functional.cross_entropy(logits, gpu['label'])
                    if training:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        optimizer.step()
                    torch.cuda.synchronize()
                    compute = time.perf_counter() - started_compute
                    row = {'scenario': scenario, 'pass': epoch, 'batch': step, 'events': len(indexes),
                           'wait_seconds': wait, 'transfer_seconds': transfer, 'compute_seconds': compute,
                           'collate_worker_seconds': profile['collate_seconds'],
                           **{key + '_worker_seconds': value for key, value in zip(TIMING_NAMES, profile['worker_seconds'])},
                           **{key: profile[key] for key in ['represented_channels', 'padded_channel_slots',
                                                           'real_blocks', 'padded_event_blocks']}}
                    epoch_rows.append(row)
                    del gpu, logits, loss, batch
                wall = time.perf_counter() - started
                if sorted(seen) != list(range(len(data))):
                    raise ValueError('profile sampler lost or duplicated real Train events')
                rows.extend(epoch_rows)
                table = pd.DataFrame(epoch_rows)
                # Separate startup from the repeatable blocking wait after the
                # first delivered batch; all-event totals are also reported.
                steady = table.iloc[1:]
                steady_seconds = steady[['wait_seconds', 'transfer_seconds', 'compute_seconds']].to_numpy().sum()
                summary = {'pass': epoch, 'events': len(seen), 'seconds': wall,
                           'events_per_second': len(seen) / wall, 'iterator_startup_seconds': startup,
                           'first_batch_wait_seconds': float(table.wait_seconds.iloc[0]),
                           'data_wait_fraction': float((startup + table.wait_seconds.sum()) / wall),
                           'steady_data_wait_fraction': float(steady.wait_seconds.sum() / steady_seconds),
                           'steady_events_per_second': float(steady.events.sum() / steady_seconds),
                           'seconds_by_stage': {name: float(table[name + '_seconds'].sum())
                                                for name in ['wait', 'transfer', 'compute']},
                           'worker_seconds': {name: float(table[name + '_worker_seconds'].sum())
                                              for name in TIMING_NAMES},
                           'collate_worker_seconds': float(table.collate_worker_seconds.sum()),
                           'channel_padding_utilization': float(table.represented_channels.sum() / table.padded_channel_slots.sum()),
                           'event_block_padding_utilization': float(table.real_blocks.sum() / table.padded_event_blocks.sum()),
                           'exact_event_coverage': True}
                summaries.append(summary)
                print(json.dumps({'scenario': scenario, **summary}), flush=True)
        peak = {**memory.peak, 'GPU_allocated_MiB': torch.cuda.max_memory_allocated() / 2**20,
                'GPU_reserved_MiB': torch.cuda.max_memory_reserved() / 2**20}
        return {'scenario': scenario, 'workers': workers, 'batch_size': batch_size,
                'bucket_size': data.config.get('bucket_size', 0), 'cache_shards_per_worker': data.config.get('cache_shards', 2),
                'training': training, 'passes': summaries, 'peak_memory': peak,
                'warmup_and_compile_seconds': warm_seconds,
                'warmup_gradient_squared_norms': gradient_squared}, rows
    finally:
        shutdown_loader(loader)
        data.native.close()
        del model, optimizer, loader
        gc.collect()
        torch.cuda.empty_cache()


def run(args):
    if args.mode == 'stream' and not os.environ.get('_CONDOR_SCRATCH_DIR'):
        raise ValueError('the 10k stream campaign requires an HTCondor allocation')
    if not torch.cuda.is_available():
        raise RuntimeError('R0 compute profile requires a GPU')
    torch.set_num_threads(1)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    torch.set_float32_matmul_precision('high')
    reference = json.loads(args.candidates.read_text())
    config = next(row['config'] for row in reference['models'] if row['name'] == args.name).copy()
    config.update(compile_local=args.compile_local, cache_shards=2)
    data_root = args.root / ('micro_data' if args.mode == 'micro' else 'profile_data')
    output = args.root / 'profiles'
    output.mkdir(exist_ok=True)
    path = output / f'{args.mode}_{args.attempt}.json'
    if path.exists():
        raise ValueError('profile attempts must be immutable')
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else None
    device = torch.device('cuda:0')
    properties = torch.cuda.get_device_properties(device)
    record = {'status': 'running', 'UTC_started': datetime.now(timezone.utc).isoformat(),
              'mode': args.mode, 'reference_model': args.name, 'config': config,
              'host': platform.node(), 'torch': torch.__version__, 'python': platform.python_version(),
              'GPU': properties.name, 'GPU_memory_MiB': properties.total_memory / 2**20,
              'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'CPU_affinity': sorted(os.sched_getaffinity(0)),
              'precision': 'BF16 autocast with FP32 classifier' if dtype else 'FP32',
              'compilation': 'torch.compile(dynamic=True) local encoder' if args.compile_local else 'eager reference',
              'training_operations': 'forward, CE, backward, clip, AdamW at lr=0; no model selection',
              'validation_scope': 'inference on the same Train stream; not a performance evaluation',
              'timing_scope': 'synchronized GPU operations; worker stage/decode are nested in native_read, overlap main process',
              'memory_scope': 'RSS sampled every 0.2 s; aggregate may double count shared libraries/mappings',
              'Confirm_responses_read': False, 'cases': []}
    started = time.perf_counter()
    write_json(path, record)
    old_staging = os.environ.get('PINPOINT_NATIVE_SHARD_ROOT')
    scratch_parent = os.environ.get('_CONDOR_SCRATCH_DIR', tempfile.gettempdir())
    with tempfile.TemporaryDirectory(prefix='pinpoint-r0-profile-', dir=scratch_parent) as temporary:
        scratch = Path(temporary)
        source = TimedPlaneDataset(data_root, 'train', config, cache_dir=scratch/'native')
        record['events'] = len(source)
        record['staging'] = stage_shards(source.paths, scratch/'shared', workers=2)
        source.native.close()
        os.environ['PINPOINT_NATIVE_SHARD_ROOT'] = str(scratch/'shared')
        source = TimedPlaneDataset(data_root, 'train', config, cache_dir=scratch/'native')
        if args.mode == 'micro':
            canonical = np.arange(len(source))
        else:
            keys = ['run', 'chunk', 'row']
            micro = pd.read_parquet(args.root/'manifests/micro_256.parquet', columns=keys)
            canonical = pd.MultiIndex.from_frame(source.metadata[keys]).get_indexer(pd.MultiIndex.from_frame(micro))
            if (canonical < 0).any():
                raise ValueError('fixed micro events must be present in the 10k Train stream')
        with MemoryMonitor() as memory:
            record['packed_cache_build'] = build_packed_cache(source, scratch/'packed', canonical)
            packed = PackedPlaneDataset(data_root, 'train', config, cache_dir=scratch/'native', packed_root=scratch/'packed')
            began = time.perf_counter()
            record['equivalence'] = verify_cache(source, packed, args.batch_size)
            record['equivalence']['seconds'] = time.perf_counter() - began
            record['equivalence']['cache_build_peak_memory'] = memory.peak
        model = build_model({**config, 'compile_local': False})
        record['representation_metadata'] = representation_metadata(model, source.geometry, config, plane_windows=True)
        # A GPU logits readback complements bitwise collated-tensor checks.
        check_indexes = [int(canonical[0]), int(canonical[-1])]
        model = model.to(device).eval()
        with torch.inference_mode():
            a = model(device_batch(collate_planes([source[i] for i in check_indexes]), device))
            b = model(device_batch(collate_planes([packed[i] for i in check_indexes]), device))
            torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6)
        record['equivalence']['GPU_logits_compared_events'] = len(check_indexes)
        record['equivalence']['GPU_logits_tolerance'] = {'rtol': 3e-5, 'atol': 3e-6}
        record['equivalence']['GPU_logits_allclose'] = True
        record['equivalence']['GPU_logits_bitwise_equal'] = bool(torch.equal(a, b))
        record['equivalence']['GPU_logits_max_absolute_difference'] = float((a-b).abs().max())
        del model, a, b, packed
        source.native.close()
        del source
        all_rows = []
        for variant in args.variants:
            for workers in args.workers:
                case_config = {**config, 'bucket_size': args.batch_size*32 if 'bucket' in variant else 0}
                if variant == 'packed_bucket':
                    data = PackedPlaneDataset(data_root, 'train', case_config, cache_dir=scratch/'native', packed_root=scratch/'packed')
                else:
                    data = TimedPlaneDataset(data_root, 'train', case_config, cache_dir=scratch/'native')
                case, rows = run_case(data, config, workers, args.batch_size, args.passes, True,
                                      f'{variant}_w{workers}', device, dtype)
                record['cases'].append(case)
                all_rows.extend(rows)
                write_json(path, record)
        best = min(record['cases'], key=lambda row: row['passes'][-1]['seconds'])
        case_config = {**config, 'bucket_size': best['bucket_size']}
        data_type = PackedPlaneDataset if best['scenario'].startswith('packed') else TimedPlaneDataset
        extra = {'packed_root': scratch/'packed'} if data_type is PackedPlaneDataset else {}
        data = data_type(data_root, 'train', case_config, cache_dir=scratch/'native', **extra)
        case, rows = run_case(data, config, best['workers'], args.batch_size, 1, False,
                              'validation_' + best['scenario'], device, dtype)
        record['cases'].append(case)
        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_parquet(output/f'{args.mode}_{args.attempt}_batches.parquet', index=False)
        record.update(status='complete', seconds=time.perf_counter()-started,
                      selected_training_scenario=best['scenario'],
                      selected_steady_data_wait_below_20pct=best['passes'][-1]['steady_data_wait_fraction'] < .2,
                      temporary_cache_bytes=record['packed_cache_build']['bytes'] + record['staging']['compressed_bytes'])
    if old_staging is None:
        os.environ.pop('PINPOINT_NATIVE_SHARD_ROOT', None)
    else:
        os.environ['PINPOINT_NATIVE_SHARD_ROOT'] = old_staging
    record['temporary_scratch_removed'] = not scratch.exists()
    record['UTC_finished'] = datetime.now(timezone.utc).isoformat()
    write_json(path, record)
    print(json.dumps({'output': str(path), 'status': record['status'], 'seconds': record['seconds'],
                      'best': record['selected_training_scenario']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--name', default='s4_empty_scratch_n100000_s42')
    parser.add_argument('--mode', choices=['micro', 'stream'], required=True)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--passes', type=int, default=2)
    parser.add_argument('--compile-local', action='store_true')
    parser.add_argument('--variants', nargs='+', choices=['native', 'native_bucket', 'packed_bucket'],
                        default=['native', 'native_bucket', 'packed_bucket'])
    parser.add_argument('--workers', nargs='+', type=int, choices=[2, 4], default=[2, 4])
    run(parser.parse_args())


if __name__ == '__main__':
    main()
