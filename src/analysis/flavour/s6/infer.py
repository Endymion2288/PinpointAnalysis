"""Observation-only, shared-representation inference and checkpoint replay.

One task writes one wide probability table for every frozen checkpoint. Native
channels are transformed once per representation, with no event/node cap.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from torch.utils.data import DataLoader, Dataset

from analysis.flavour.datasets import CachedNativeDataset
from analysis.flavour.inventory import write_json
from analysis.flavour.s2.train import save_parquet
from analysis.flavour.s3.dataset import plane_tokens, collate_planes
from analysis.flavour.s3.health import TrainingHealth
from analysis.flavour.s3.models_repair import PlaneModuleClassifierV2
from analysis.flavour.s5.model import PhysicsModel
from analysis.flavour.s6.prepare import KEY, require_freeze


REPRESENTATION_KEYS = ('modalities', 'pitch_mm', 'window_pixels', 'signal', 'local_residual', 'region_blocks')


def representations(models):
    groups = {}
    membership = {}
    for model in models:
        config = {k: model['config'].get(k) for k in REPRESENTATION_KEYS if k in model['config']}
        signature = json.dumps(config, sort_keys=True)
        if signature not in groups:
            groups[signature] = config
        membership[model['name']] = list(groups).index(signature)
    return list(groups.values()), membership


class Responses(Dataset):
    """Labels/keys are returned for joining only; transforms read observations."""
    def __init__(self, paths, geometry, configs, selected=None):
        self.paths = paths
        self.geometry, self.configs = geometry, configs
        self.native = CachedNativeDataset(paths, max_shards=1, target_columns=('label',))
        frames = []
        for i, path in enumerate(paths):
            frame = pd.read_parquet(path/'provenance.parquet')
            frame['_native_index'] = self.native.boundaries[i]+np.arange(len(frame))
            frames.append(frame)
        self.metadata = pd.concat(frames, ignore_index=True)
        if selected is not None:
            self.metadata = selected[KEY].merge(self.metadata, on=KEY, how='left', validate='one_to_one', sort=False)
            if self.metadata['_native_index'].isna().any():
                raise ValueError('requested response keys missing')
        self.indexes = self.metadata['_native_index'].to_numpy(dtype='int64')

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        observation = self.native[int(self.indexes[index])]['observations']
        result = []
        for config in self.configs:
            nodes, groups, modules, counts = plane_tokens(observation, self.geometry, config)
            result.append({'nodes': nodes, 'groups': groups, 'modules': modules, 'raw_counts': counts,
                           'region_blocks': config.get('region_blocks', 32), 'label': 0, 'index': index})
        return result


def collate(events):
    return [collate_planes(list(group)) for group in zip(*events)]


def init_worker(_):
    torch.set_num_threads(1)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)


def load_models(candidates, device, compile_local=False):
    loaded = {}
    for item in candidates:
        state = torch.load(item['checkpoint'], map_location='cpu', weights_only=False)
        if state['config'] != item['config']:
            raise ValueError(f"checkpoint config differs: {item['name']}")
        config = {**item['config'], 'compile_local': compile_local}
        if config.get('use_objects'):
            raise ValueError('this campaign contains observation-only candidates')
        model = (PhysicsModel if item['kind'] == 'physics' else PlaneModuleClassifierV2)(config)
        model.load_state_dict(state['model'], strict=True)
        loaded[item['name']] = model.to(device).eval()
    return loaded


def probabilities(model, batch, precision):
    with torch.autocast(device_type=batch['nodes'].device.type, dtype=torch.bfloat16, enabled=precision == 'bf16'):
        output = model(batch)
    if isinstance(output, dict):
        probability = output['probability']
        auxiliary = {k: output[k].float().cpu().numpy() for k in ['vertex', 'kinematics', 'charm']}
        logits = output['logits']
    else:
        logits = output
        probability, auxiliary = logits.float().softmax(-1), {}
    values = probability.float().cpu().numpy()
    if not np.isfinite(values).all() or not np.allclose(values.sum(1), 1, atol=2e-6):
        raise ValueError('invalid probabilities require architecture/runtime review')
    return values, auxiliary, logits


def predict(dataset, candidates, models, membership, args, health=False):
    kwargs = {'prefetch_factor': 2, 'persistent_workers': True} if args.workers else {}
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        collate_fn=collate, worker_init_fn=init_worker, pin_memory=args.device.startswith('cuda'), **kwargs)
    output = {item['name']: np.zeros((len(dataset), 4), dtype='float32') for item in candidates}
    auxiliary = defaultdict(dict)
    samples = {}
    started = time.monotonic()
    model_seconds = defaultdict(float)
    represented = [0 for _ in dataset.configs]
    with torch.inference_mode():
        for step, cpu_batches in enumerate(loader):
            indexes = cpu_batches[0]['index'].numpy()
            for group, cpu in enumerate(cpu_batches):
                represented[group] += int(cpu['local_mask'].sum())
                batch = {k: v.to(args.device, non_blocking=True) for k, v in cpu.items() if k not in ['label', 'index']}
                for item in candidates:
                    name = item['name']
                    if membership[name] != group:
                        continue
                    model = models[name]
                    probe = TrainingHealth(model) if health and step == 0 else None
                    if probe:
                        probe.start(batch)
                    if args.device.startswith('cuda'):
                        torch.cuda.synchronize()
                    tick = time.monotonic()
                    value, extra, logits = probabilities(model, batch, args.precision)
                    if args.device.startswith('cuda'):
                        torch.cuda.synchronize()
                    model_seconds[name] += time.monotonic()-tick
                    output[name][indexes] = value
                    for key, values in extra.items():
                        if key not in auxiliary[name]:
                            auxiliary[name][key] = np.zeros((len(dataset), values.shape[1]), dtype='float32')
                        auxiliary[name][key][indexes] = values
                    if probe:
                        samples[name] = probe.finish(logits, step=0)
                        probe.close()
                del batch
            if step % 100 == 0:
                print(json.dumps({'inference_batch': step, 'events': int(indexes.max()+1), 'total': len(dataset),
                                  'seconds': time.monotonic()-started}), flush=True)
    report = {'seconds': time.monotonic()-started, 'model_seconds': dict(model_seconds), 'events': len(dataset),
        'represented_nodes_by_group': represented, 'health': samples,
        'GPU': torch.cuda.get_device_name() if args.device.startswith('cuda') else None,
        'peak_GPU_MiB': torch.cuda.max_memory_allocated()/2**20 if args.device.startswith('cuda') else None,
        'peak_RSS_MiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'probability_std': {k: p.std(0).tolist() for k, p in output.items()}}
    del loader
    dataset.native.close()
    return output, auxiliary, report


def configure(args):
    torch.set_num_threads(1)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    torch.manual_seed(2026)
    if args.precision == 'bf16' and (not args.device.startswith('cuda') or not torch.cuda.is_bf16_supported(including_emulation=False)):
        raise ValueError('formal BF16 inference requires a supported GPU')
    campaign = json.loads(args.campaign.read_text())
    candidates = [m for m in campaign['models'] if not args.models or m['name'] in args.models]
    if not candidates:
        raise ValueError('empty candidate set')
    configs, membership = representations(candidates)
    models = load_models(candidates, args.device, args.compile_local)
    return campaign, candidates, configs, membership, models


def replay(args):
    campaign, candidates, configs, membership, models = configure(args)
    plan = json.loads((args.source/'plan.json').read_text())
    reference = pd.read_parquet(candidates[0]['validation_predictions'])
    occupancy = pd.read_parquet(args.source/'catalog.parquet', columns=KEY+['pixel_channels'])
    if 'pixel_channels' not in reference:
        reference = reference.merge(occupancy, on=KEY, how='left', validate='one_to_one', sort=False)
    selected = []
    for label, group in reference.groupby('label', sort=True):
        # Deterministic coverage across the existing validation, including its
        # low/high occupancy tails when counts are available.
        if 'pixel_channels' in group:
            group = group.sort_values('pixel_channels', kind='stable')
        selected.append(group.iloc[np.unique(np.linspace(0, len(group)-1, args.events_per_class).astype(int))][KEY])
    selected = pd.concat(selected, ignore_index=True)
    paths = sorted(p for p in (args.source/'shards').iterdir() if (p/'observations.parquet').exists())
    dataset = Responses(paths, json.loads(Path(plan['geometry']).read_text()), configs, selected)
    result, _, report = predict(dataset, candidates, models, membership, args, health=True)
    comparisons = {}
    for item in candidates:
        ref = pd.read_parquet(item['validation_predictions'])
        aligned = selected.merge(ref, on=KEY, how='left', validate='one_to_one', sort=False)
        if aligned['p0'].isna().any():
            raise ValueError('candidate validations have different selected keys')
        delta = np.abs(result[item['name']]-aligned[[f'p{c}' for c in range(4)]].to_numpy())
        comparisons[item['name']] = {'maximum_absolute_difference': float(delta.max()),
            'mean_absolute_difference': float(delta.mean()), 'within_0p003': bool(delta.max() <= .003)}
    report.update(comparisons=comparisons, selected_keys=selected.to_dict('records'),
        precision=args.precision, compile_local=args.compile_local, batch_size=args.batch_size, workers=args.workers,
        source=str(args.source), torch=torch.__version__,
        status='passed' if all(r['within_0p003'] for r in comparisons.values()) and all(r['healthy'] for r in report['health'].values()) else 'review_required')
    write_json(args.root/'audit'/f'{args.name}.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ['selected_keys', 'health']}), flush=True)
    if report['status'] != 'passed':
        raise RuntimeError('checkpoint replay requires review; evidence saved')


def run_task(args, prepared):
    campaign, candidates, configs, membership, models = prepared
    plan = json.loads((args.root/'plan.json').read_text())
    spec = plan['tasks'][args.task]
    require_freeze(args.root, spec['split'])
    path = args.root/'shards'/f"{spec['split']}_{spec['shard']:04d}"
    dataset = Responses([path], json.loads(Path(plan['geometry']).read_text()), configs)
    result, auxiliary, report = predict(dataset, candidates, models, membership, args, health=True)
    columns = {}
    for name, probability in result.items():
        for c in range(4):
            columns[f'{name}__p{c}'] = probability[:, c]
        for key, values in auxiliary[name].items():
            for c in range(values.shape[1]):
                columns[f'{name}__{key}{c}'] = values[:, c]
    frame = pd.concat([dataset.metadata[KEY+['label']].reset_index(drop=True), pd.DataFrame(columns)], axis=1)
    destination = args.root/'predictions'/'nominal'/f"{args.task:04d}"
    destination.mkdir(parents=True, exist_ok=True)
    save_parquet(frame, destination/'probabilities.parquet')
    report.update(task=args.task, split=spec['split'], status='complete',
        candidates=[m['name'] for m in candidates], precision=args.precision, compile_local=args.compile_local,
        batch_size=args.batch_size, workers=args.workers, commit=os.environ.get('PINPOINT_CODE_COMMIT'),
        UTC=datetime.now(timezone.utc).isoformat(), host=socket.gethostname(),
        checkpoint_paths={m['name']: m['checkpoint'] for m in candidates})
    write_json(destination/'summary.json', report)
    print(json.dumps({'task': args.task, 'events': len(frame), 'seconds': report['seconds'], 'status': 'complete'}), flush=True)


def task(args):
    run_task(args, configure(args))


def profile(args):
    prepared = configure(args)
    campaign, candidates, configs, membership, models = prepared
    plan = json.loads((args.root/'plan.json').read_text())
    spec = plan['tasks'][args.task]
    if spec['split'] != 'calibration':
        raise ValueError('execution profiling uses calibration only')
    path = args.root/'shards'/f"calibration_{spec['shard']:04d}"
    selected = pd.read_parquet(path/'provenance.parquet').iloc[:args.limit]
    trials, reference = [], None
    for size in args.batch_sizes:
        args.batch_size = size
        dataset = Responses([path], json.loads(Path(plan['geometry']).read_text()), configs, selected)
        result, _, report = predict(dataset, candidates, models, membership, args, health=False)
        differences = {name: float(np.abs(p-reference[name]).max()) for name, p in result.items()} if reference else {}
        trials.append({'batch_size': size, 'workers': args.workers, **report, 'difference_from_first': differences})
        if reference is None:
            reference = result
    report = {'trials': trials, 'precision': args.precision, 'compile_local': args.compile_local,
              'status': 'passed' if all(d <= .003 for t in trials for d in t['difference_from_first'].values()) else 'review_required'}
    write_json(args.root/'audit'/f'{args.name}.json', report)
    print(json.dumps({'profile': args.name, 'trials': [
        {k: t[k] for k in ['batch_size', 'events', 'seconds', 'peak_GPU_MiB', 'difference_from_first']} for t in trials], 'status': report['status']}), flush=True)
    if report['status'] != 'passed':
        raise RuntimeError('batch-size invariance requires review')


def worker(args):
    policy = json.loads((args.root/'inference_policy.json').read_text())
    for key in ['precision', 'compile_local', 'batch_size']:
        if getattr(args, key) != policy[key]:
            raise ValueError(f'inference policy mismatch: {key}')
    prepared = configure(args)
    plan = json.loads((args.root/'plan.json').read_text())
    claims = args.root/'runtime'/'nominal_claims'
    claims.mkdir(exist_ok=True)
    completed = []
    started = time.monotonic()
    idle_since = None
    while True:
        chosen = None
        incomplete_preparation = False
        for task_id, spec in enumerate(plan['tasks']):
            if spec['split'] != args.split:
                continue
            output = args.root/'predictions'/'nominal'/f'{task_id:04d}'/'summary.json'
            if output.exists():
                previous = json.loads(output.read_text())
                if previous['candidates'] != [m['name'] for m in prepared[1]] or previous['events'] != spec['events']:
                    raise ValueError('completed inference has different coverage/candidates')
                continue
            if not (args.root/'shards'/f"{spec['split']}_{spec['shard']:04d}"/'summary.json').exists():
                incomplete_preparation = True
                continue
            claim = claims/f'{task_id:04d}'
            try:
                claim.mkdir()
            except FileExistsError:
                continue
            chosen = task_id
            write_json(claim/'owner.json', {'host': socket.gethostname(), 'pid': os.getpid(), 'UTC': datetime.now(timezone.utc).isoformat(),
                       'worker': args.name, 'commit': os.environ.get('PINPOINT_CODE_COMMIT')})
            break
        if chosen is None:
            if idle_since is None:
                idle_since = time.monotonic()
            if incomplete_preparation and time.monotonic()-idle_since < 60 and time.monotonic()-started < args.maximum_seconds:
                # Preparation and inference overlap. A short wait permits GPU
                # workers to consume newly finished CPU shards without a queue.
                time.sleep(15)
                continue
            break
        idle_since = None
        args.task = chosen
        try:
            run_task(args, prepared)
            completed.append(chosen)
        finally:
            (claim/'owner.json').unlink()
            claim.rmdir()
        if time.monotonic()-started >= args.maximum_seconds:
            break
    write_json(args.root/'runtime'/f'{args.name}.worker.json', {'status': 'complete', 'tasks': completed,
        'seconds': time.monotonic()-started, 'GPU': torch.cuda.get_device_name(), 'host': socket.gethostname()})


def pool(args):
    """Independent claim-based lanes share one allocation; no duplicate writer."""
    processes = []
    logs = []
    for lane in range(args.lanes):
        name = f'{args.name}_lane{lane}'
        command = [sys.executable, '-m', __name__, 'worker', '--root', str(args.root), '--campaign', str(args.campaign),
                   '--split', args.split, '--name', name, '--batch-size', str(args.batch_size), '--workers', str(args.workers),
                   '--precision', args.precision, '--maximum-seconds', str(args.maximum_seconds)]
        if args.compile_local:
            command += ['--compile-local']
        log = (args.root/'logs'/f'{name}.log').open('w')
        logs.append(log)
        processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
    states = [process.wait() for process in processes]
    for log in logs:
        log.close()
    write_json(args.root/'runtime'/f'{args.name}.pool.json', {'exit_codes': states, 'lanes': args.lanes, 'workers_per_lane': args.workers})
    if any(states):
        raise RuntimeError('an inference lane failed; inspect its EOS log')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['replay', 'task', 'profile', 'worker', 'pool'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--name', default='checkpoint_replay')
    parser.add_argument('--models', nargs='+')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--precision', choices=['fp32', 'bf16'], default='bf16')
    parser.add_argument('--compile-local', action='store_true')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--events-per-class', type=int, default=4)
    parser.add_argument('--task', type=int, default=0)
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[16, 64, 128])
    parser.add_argument('--limit', type=int, default=1024)
    parser.add_argument('--split', choices=['calibration', 'test'], default='calibration')
    parser.add_argument('--lanes', type=int, default=2)
    parser.add_argument('--maximum-seconds', type=int, default=27000)
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == '__main__':
    main()
