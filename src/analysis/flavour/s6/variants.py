"""Frozen paired digitizer inference, sharing native input and primary models."""
import argparse
from bisect import bisect_right
import json
import os
from pathlib import Path
import socket
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['worker', 'finalize'])
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
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == '__main__':
    main()
