"""Measure native-shard scratch caching and ragged collation, without training."""

import argparse
import gc
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import time

import pyarrow as pa
import torch
from torch.utils.data import DataLoader, get_worker_info

from analysis.flavour.datasets import CachedNativeDataset, ShardBatchSampler, collate_native
from analysis.flavour.inventory import write_json


def init_worker(worker_id):
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    torch.set_num_threads(1)


def collate_profile(events):
    started = time.perf_counter()
    batch = collate_native(events)
    info = get_worker_info()
    batch['_profile'] = {
        'collate_seconds': time.perf_counter() - started,
        'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        'cache': info.dataset.cache_stats() if info else None}
    return batch


def profile_case(paths, args, workers, capacity, window):
    started = time.perf_counter()
    with CachedNativeDataset(paths, cache_dir=args.cache_dir, max_shards=capacity) as dataset:
        metadata_seconds = time.perf_counter() - started
        sampler = ShardBatchSampler(dataset, args.batch_size, window_shards=window, seed=args.seed)
        options = dict(num_workers=workers, collate_fn=collate_profile, batch_sampler=sampler)
        if workers:
            options.update(worker_init_fn=init_worker, multiprocessing_context='spawn',
                           prefetch_factor=1, persistent_workers=True)
        loader = DataLoader(dataset, **options)
        latest, epochs = {}, []
        for epoch in range(args.epochs):
            sampler.set_epoch(epoch)
            counts, tokens, batches = 0, 0, 0
            labels = [0] * 4
            statuses = {}
            mixed_batches, collate_seconds = 0, 0.
            t = time.perf_counter()
            first_batch_seconds = None
            for batch in loader:
                if first_batch_seconds is None:
                    first_batch_seconds = time.perf_counter() - t
                targets = batch['targets']
                counts += len(targets['label'])
                counts_by_class = torch.bincount(targets['label'], minlength=4).tolist()
                labels = [a+b for a, b in zip(labels, counts_by_class)]
                mixed_batches += int(torch.unique(targets['label']).numel() > 1)
                for status in targets['label_status']:
                    statuses[status] = statuses.get(status, 0) + 1
                tokens += sum(len(batch['observations'][d]['values'])
                              for d in ['pixel', 'bar', 'tracker_hypothesis'])
                telemetry = batch.pop('_profile')
                collate_seconds += telemetry['collate_seconds']
                if workers:
                    latest[str(telemetry['cache']['pid'])] = telemetry
                batches += 1
            elapsed = time.perf_counter() - t
            if counts != len(dataset):
                raise ValueError('sampler lost or duplicated events')
            if not workers:
                latest[str(os.getpid())] = {'cache': dataset.cache_stats(),
                    'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
            epochs.append(dict(epoch=epoch, events=counts, tokens=tokens, batches=batches,
                label_counts=labels, label_status_counts=statuses, mixed_class_batches=mixed_batches,
                wall_seconds=elapsed, first_batch_seconds=first_batch_seconds,
                events_per_second=counts / elapsed, collate_worker_seconds=collate_seconds,
                cache_counters={pid: value['cache'].copy() for pid, value in latest.items()}))
        # Explicitly release persistent workers; their process finalizers remove
        # only their own scratch directories. No private DataLoader APIs.
        del batch, loader
        gc.collect()
        roots = [v['cache']['root'] for v in latest.values()]
    cleaned = {p: not Path(p).exists() for p in roots}
    if not all(cleaned.values()):
        raise RuntimeError(f'worker cache cleanup incomplete: {cleaned}')
    return dict(workers=workers, capacity_shards_per_worker=capacity, window_shards=window,
        metadata_seconds=metadata_seconds, batch_size=args.batch_size, seed=args.seed,
        prefetch_factor=1 if workers else None, persistent_workers=bool(workers),
        multiprocessing_context='spawn' if workers else None, epochs=epochs,
        final_worker_telemetry=latest, scratch_cleanup=cleaned)


def transfer_smoke(paths, args):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable for requested transfer check')
    # Four pilot shards, at most 16 events each; no network or optimization.
    with CachedNativeDataset(paths, cache_dir=args.cache_dir) as dataset:
        indices = [i for start, stop in zip(dataset.boundaries[:-1], dataset.boundaries[1:])
                   for i in range(start, min(start + 16, stop))]
        t = time.perf_counter()
        batch = collate_native([dataset[i] for i in indices])
        cpu_seconds = time.perf_counter() - t
        tensors = []

        def gather(value):
            if isinstance(value, torch.Tensor):
                tensors.append(value)
            elif isinstance(value, dict):
                for child in value.values():
                    gather(child)

        gather(batch['observations'])
        torch.cuda.synchronize()
        initial_allocated = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        transferred = [tensor.to('cuda') for tensor in tensors]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t
        exact = all(torch.equal(original, moved.cpu()) for original, moved in zip(tensors, transferred))
        if not exact:
            raise ValueError('native CPU/GPU round trip changed tensors')
        return dict(events=len(indices), scope='native_observation_transfer_only_no_model',
                    cpu_stage_decode_collate_seconds=cpu_seconds,
                    tensor_bytes=sum(t.numel() * t.element_size() for t in tensors),
                    transfer_seconds=elapsed, exact_round_trip=exact,
                    incremental_peak_allocated_mib=(torch.cuda.max_memory_allocated()-initial_allocated) / 2**20,
                    gpu_name=torch.cuda.get_device_name(), cache=dataset.cache_stats())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--transfer-only', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    torch.set_num_threads(1)
    paths = [p for p in sorted(args.input.iterdir()) if (p / 'observations.parquet').exists()]
    if len(paths) != 4:
        parser.error('this pilot profile expects four class shards')
    report = dict(input=str(args.input), scope='S1_development_IO_all_pilot_rows_no_performance_tuning',
                  host=platform.node(), cpu_affinity=sorted(os.sched_getaffinity(0)),
                  code_head_at_launch=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  code_tracked_changes_at_launch=subprocess.check_output(['git', 'diff', '--name-only'], text=True).splitlines(),
                  timing_scope='includes staging/decode/collate/IPC; first epoch includes worker startup; OS/EOS cache state uncontrolled',
                  versions={'python': platform.python_version(), 'pyarrow': pa.__version__, 'torch': torch.__version__})
    if args.transfer_only:
        report['transfer'] = transfer_smoke(paths, args)
    else:
        report['cases'] = [profile_case(paths, args, workers, capacity, window)
                           for workers, capacity, window in [(0, 1, 1), (2, 1, 1), (2, 4, 4)]]
    report['parent_peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
