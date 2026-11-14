"""Frozen paired digitizer inference, sharing native input and primary models."""
import argparse
from bisect import bisect_right
import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.s3.dataset import plane_tokens
from analysis.flavour.s6.conditions import Condition
from analysis.flavour.s6.infer import Responses, configure, predict
from analysis.flavour.s6.prepare import KEY, require_freeze


class VariantResponses(Responses):
    def __init__(self, paths, geometry, configs, selected, variants, hardware):
        super().__init__(paths, geometry, configs, selected)
        self.base_metadata = self.metadata.copy()
        self.metadata = self.base_metadata.loc[self.base_metadata.index.repeat(len(variants))].reset_index(drop=True)
        self.metadata['variant'] = np.tile([v['name'] for v in variants], len(self.base_metadata))
        self.indexes = np.repeat(self.indexes, len(variants))
        self.transforms = [Condition(geometry, v['parameters'], hardware) for v in variants]
        self.ideal_cache = None

    def __getitem__(self, index):
        native_index = int(self.indexes[index])
        observation = self.native[native_index]['observations']
        shard = bisect_right(self.native.boundaries, native_index)-1
        if self.ideal_cache is None or self.ideal_cache[0] != shard:
            self.ideal_cache = (shard, pq.read_table(self.paths[shard]/'ideal_tracker.parquet').to_pylist())
        ideal = self.ideal_cache[1][native_index-self.native.boundaries[shard]]
        transform = self.transforms[index % len(self.transforms)]
        changed = transform(observation, self.metadata.iloc[index], ideal)
        result = []
        for config in self.configs:
            nodes, groups, modules, counts = plane_tokens(changed, transform.geometry, config)
            result.append({'nodes': nodes, 'groups': groups, 'modules': modules, 'raw_counts': counts,
                'region_blocks': config.get('region_blocks', 32), 'label': 0, 'index': index})
        return result


def run(args, prepared):
    require_freeze(args.root, 'test')
    _, candidates, configs, membership, models = prepared
    config = json.loads((args.root/'conditions.json').read_text())
    plan = json.loads((args.root/'plan.json').read_text())
    geometry = json.loads(Path(plan['geometry']).read_text())
    hardware = json.loads((args.root/'audit/bar_hardware_map.json').read_text())
    manifest = pd.read_parquet(args.root/'variants_manifest.parquet')
    shards = sorted(manifest.shard.unique())[args.task*args.shards_per_task:(args.task+1)*args.shards_per_task]
    selected = manifest[manifest.shard.isin(shards)]
    paths = [args.root/'shards'/f'test_{s:04d}' for s in shards]
    dataset = VariantResponses(paths, geometry, configs, selected, config['variants'], hardware)
    output, _, report = predict(dataset, candidates, models, membership, args, health=True)
    frame = dataset.metadata[KEY+['label', 'variant']].copy().reset_index(drop=True)
    for name, p in output.items():
        for c in range(4):
            frame[f'{name}__p{c}'] = p[:, c]
    destination = args.root/'predictions/variants'/f'{args.task:04d}'
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination/'probabilities.parquet', index=False, compression=COMPRESSION)
    report.update(status='complete', task=args.task, primary_events=len(selected), variants=len(config['variants']),
        candidates=[m['name'] for m in candidates], commit=os.environ.get('PINPOINT_CODE_COMMIT'),
        host=socket.gethostname(), thresholds_retuned=False)
    write_json(destination/'summary.json', report)
    print(json.dumps({'variant_task': args.task, 'events': len(frame), 'seconds': report['seconds']}), flush=True)


def worker(args):
    config = json.loads((args.root/'conditions.json').read_text())
    args.models = config['models']
    prepared = configure(args)
    manifest = pd.read_parquet(args.root/'variants_manifest.parquet', columns=['shard'])
    ntasks = (manifest.shard.nunique()+args.shards_per_task-1)//args.shards_per_task
    claims = args.root/'runtime/variant_claims'
    claims.mkdir(exist_ok=True)
    for task in range(ntasks):
        if (args.root/'predictions/variants'/f'{task:04d}/summary.json').exists():
            continue
        claim = claims/f'{task:04d}'
        try:
            claim.mkdir()
        except FileExistsError:
            continue
        write_json(claim/'owner.json', {'host': socket.gethostname(), 'pid': os.getpid(), 'started': time.time()})
        try:
            args.task = task
            run(args, prepared)
        finally:
            (claim/'owner.json').unlink()
            claim.rmdir()


def run_available(args, prepared):
    """Consume one ready paired task during a gap in nominal CPU preparation.

    Reuse resident primary weights; no new checkpoint loading or compilation
    policy. Nominal inference receives priority whenever a shard is available.
    """
    from analysis.flavour.s6.infer import representations
    config = json.loads((args.root/'conditions.json').read_text())
    campaign, candidates, _, _, models = prepared
    candidates = [m for m in candidates if m['name'] in config['models']]
    if len(candidates) != len(config['models']):
        return False
    configs, membership = representations(candidates)
    manifest = pd.read_parquet(args.root/'variants_manifest.parquet', columns=['shard'])
    shards = sorted(manifest.shard.unique())
    claims = args.root/'runtime/variant_claims'
    claims.mkdir(exist_ok=True)
    for task in range((len(shards)+7)//8):
        if (args.root/'predictions/variants'/f'{task:04d}/summary.json').exists():
            continue
        if not all((args.root/'shards'/f'test_{s:04d}/summary.json').exists() for s in shards[task*8:(task+1)*8]):
            continue
        claim = claims/f'{task:04d}'
        try:
            claim.mkdir()
        except FileExistsError:
            continue
        write_json(claim/'owner.json', {'host': socket.gethostname(), 'pid': os.getpid(), 'started': time.time(),
            'mode': 'resident primary models during nominal preparation gap'})
        local = copy.copy(args)
        local.task, local.shards_per_task = task, 8
        try:
            run(local, (campaign, candidates, configs, membership, models))
        finally:
            (claim/'owner.json').unlink()
            claim.rmdir()
        return True
    return False


def finalize(args):
    config = json.loads((args.root/'conditions.json').read_text())
    expected = pd.read_parquet(args.root/'variants_manifest.parquet')[KEY]
    parts = sorted((args.root/'predictions/variants').glob('*/probabilities.parquet'))
    frame = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    for v in config['variants']:
        actual = frame[frame.variant.eq(v['name'])][KEY]
        if len(actual) != len(expected) or actual.duplicated().any() or not np.array_equal(
            actual.sort_values(KEY).to_numpy(), expected.sort_values(KEY).to_numpy()):
            raise ValueError('paired variation event coverage differs')
    frame.to_parquet(args.root/'predictions/variants_raw.parquet', index=False, compression=COMPRESSION)
    # Replayed nominal must agree with frozen nominal inference independently.
    nominal = frame[frame.variant.eq('nominal')]
    original = pd.read_parquet(args.root/'predictions/test_raw.parquet')
    joined = nominal.merge(original, on=KEY, validate='one_to_one', suffixes=('_replay', '_original'))
    differences = {name: max(float(np.abs(joined[f'{name}__p{c}_replay']-joined[f'{name}__p{c}_original']).max())
        for c in range(4)) for name in config['models']}
    if max(differences.values()) > .003:
        raise ValueError('variant nominal replay exceeds frozen tolerance')
    write_json(args.root/'predictions/variants_completion.json', {'status': 'complete', 'events': len(expected),
        'variants': len(config['variants']), 'nominal_maximum_probability_differences': differences})


def replay_metrics(args):
    from analysis.flavour.s3.metrics import MetricCache
    from analysis.flavour.s6.calibration import extract, transform
    from analysis.flavour.s6.evaluate import FixedCache, METRIC_NAMES, METRIC_INDEX
    from analysis.flavour.s6.bootstrap import weights
    frame = pd.read_parquet(args.root/'predictions/variants_raw.parquet', filters=[('variant', '=', 'nominal')])
    original = pd.read_parquet(args.root/'predictions/test_raw.parquet')
    original = frame[KEY].merge(original, on=KEY, how='left', validate='one_to_one', sort=False)
    if len(original) != 32000 or original.label.isna().any():
        raise ValueError('nominal replay metric event coverage differs')
    freeze = json.loads((args.root/'analysis_freeze.json').read_text())
    names = freeze['conditions']['models']+['mae3_100k_ensemble']
    output = {}
    y = frame.label.to_numpy()
    for name in names:
        cal = freeze['calibrators'][name]
        rows = {}
        for scenario in cal['priors']:
            points = [p for p in cal['thresholds'] if p['scenario'] == scenario['name']]
            prior = scenario['prior']
            a = transform(extract(frame, name, freeze['candidates']), cal['temperature'], prior)
            b = transform(extract(original, name, freeze['candidates']), cal['temperature'], prior)
            w = weights(y, prior)
            fixed, fixed0 = FixedCache(y, a, points), FixedCache(y, b, points)
            metric_names = METRIC_NAMES+fixed.names
            values = np.r_[MetricCache(y, a).compute(w)[0][METRIC_INDEX], fixed.vector(w)]
            before = np.r_[MetricCache(y, b).compute(w)[0][METRIC_INDEX], fixed0.vector(w)]
            rows[scenario['name']] = {'replay_minus_original': dict(zip(metric_names, (values-before).tolist())),
                'argmax_changed_events': int((a.argmax(1) != b.argmax(1)).sum()),
                'tau_cut_membership_changed_events': [int((x[:, 2] != y0[:, 2]).sum()) for x, y0 in zip(fixed.passed, fixed0.passed)]}
        output[name] = rows
    write_json(args.root/'audit/replay_numerical_effects.json', {'events': len(frame), 'models': output,
        'definition': 'identical nominal observations under frozen BF16/compiled/batch64 inference; different batch composition/backend concurrency; fixed cuts unchanged',
        'thresholds_retuned': False})


def pool(args):
    processes, logs = [], []
    for lane in range(args.lanes):
        path = args.root/'logs'/f'{args.name}_lane{lane}.log'
        stream = path.open('w')
        logs.append(stream)
        command = [sys.executable, '-m', 'analysis.flavour.s6.variants', 'worker', '--root', str(args.root),
            '--campaign', str(args.campaign), '--batch-size', str(args.batch_size), '--workers', str(args.workers),
            '--precision', args.precision, '--shards-per-task', str(args.shards_per_task)]
        if args.compile_local:
            command += ['--compile-local']
        processes.append(subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT))
    states = [p.wait() for p in processes]
    for stream in logs:
        stream.close()
    write_json(args.root/'runtime'/f'{args.name}.pool.json', {'lanes': args.lanes, 'exit_codes': states})
    if any(states):
        raise RuntimeError('paired variation inference failed; inspect EOS lane logs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['worker', 'pool', 'finalize', 'replay_metrics'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--models', nargs='+')
    parser.add_argument('--shards-per-task', type=int, default=8)
    parser.add_argument('--task', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--precision', choices=['fp32', 'bf16'], default='bf16')
    parser.add_argument('--compile-local', action='store_true')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lanes', type=int, default=4)
    parser.add_argument('--name', default='variants')
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == '__main__':
    main()
