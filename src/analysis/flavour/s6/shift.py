"""Frozen primary-model predictions on actual transport/generator responses."""
import argparse
import json
import os
from pathlib import Path
import socket
import time

import numpy as np
import pandas as pd

from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.s6.infer import Responses, configure, predict
from analysis.flavour.s6.prepare import KEY, require_freeze


def worker(args):
    require_freeze(args.root, 'test')
    plan = json.loads((args.root/'plan.json').read_text())
    config = json.loads((args.root/'conditions.json').read_text())
    args.models = config['models']
    _, candidates, configs, membership, models = configure(args)
    geometry = json.loads(Path(plan['geometry']).read_text())
    claims = args.root/'runtime'/f'{args.scope}_inference_claims'
    claims.mkdir(exist_ok=True)
    started = time.monotonic()
    while True:
        completed = False
        paths = sorted((args.root/args.scope/'shards').glob('*/*/summary.json'))
        for summary_path in paths:
            path = summary_path.parent
            condition, task = path.parent.name, path.name
            destination = args.root/'predictions'/args.scope/condition/task
            if (destination/'summary.json').exists():
                continue
            claim = claims/(condition+'_'+task)
            try:
                claim.mkdir()
            except FileExistsError:
                continue
            write_json(claim/'owner.json', {'host': socket.gethostname(), 'pid': os.getpid()})
            try:
                dataset = Responses([path], geometry, configs)
                result, _, report = predict(dataset, candidates, models, membership, args, health=True)
                frame = dataset.metadata.copy().drop(columns=['_native_index'])
                for name, value in result.items():
                    for c in range(4):
                        frame[f'{name}__p{c}'] = value[:, c]
                frame['condition'] = condition
                destination.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(destination/'probabilities.parquet', index=False, compression=COMPRESSION)
                report.update(status='complete', scope=args.scope, condition=condition, task=task,
                    candidates=[m['name'] for m in candidates], commit=os.environ.get('PINPOINT_CODE_COMMIT'))
                write_json(destination/'summary.json', report)
                print(json.dumps({'scope': args.scope, 'condition': condition, 'task': task, 'events': len(frame)}), flush=True)
                completed = True
            finally:
                (claim/'owner.json').unlink()
                claim.rmdir()
        if not completed or time.monotonic()-started > args.maximum_seconds:
            break


def finalize(args):
    paths = sorted((args.root/'predictions'/args.scope).glob('*/*/probabilities.parquet'))
    frame = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    if frame.duplicated(KEY+['condition']).any():
        raise ValueError('duplicate shifted inference keys')
    manifest = pd.read_parquet(args.root/args.scope/'manifest.parquet')
    if args.scope == 'generator':
        expected_conditions = ['nominal']
    else:
        from analysis.flavour.s6.transport import transport_variants
        expected_conditions = [v['name'] for v in transport_variants(json.loads((args.root/'conditions.json').read_text()))]
    for condition in expected_conditions:
        expected = manifest
        if condition.startswith('tau_helicity_'):
            expected = manifest[manifest.label.eq(2)]
        actual = frame[frame.condition.eq(condition)]
        if len(actual) != len(expected) or not np.array_equal(actual.sort_values(KEY)[KEY].to_numpy(),
            expected.sort_values(KEY)[KEY].to_numpy()):
            raise ValueError(f'{args.scope} condition {condition} event coverage differs')
    frame.to_parquet(args.root/'predictions'/f'{args.scope}_raw.parquet', index=False, compression=COMPRESSION)
    write_json(args.root/'predictions'/f'{args.scope}_completion.json', {'status': 'complete',
        'events': len(frame), 'condition_counts': frame.groupby('condition').size().to_dict()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['worker', 'finalize'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--scope', choices=['transport', 'generator'], required=True)
    parser.add_argument('--models', nargs='+')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--precision', choices=['fp32', 'bf16'], default='bf16')
    parser.add_argument('--compile-local', action='store_true')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--maximum-seconds', type=int, default=7200)
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == '__main__':
    main()
