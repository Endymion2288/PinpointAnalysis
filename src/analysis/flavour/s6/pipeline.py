"""Dependency-aware CPU finalization; accepted event coverage is never reduced."""
import argparse
import json
from pathlib import Path
import time

from analysis.flavour.inventory import write_json


def calibrate(args):
    from analysis.flavour.s6 import prepare, infer, calibration
    plan = json.loads((args.root/'plan.json').read_text())
    ids = [i for i, t in enumerate(plan['tasks']) if t['split'] == 'calibration']
    started = time.monotonic()
    while True:
        missing = [i for i in ids if not (args.root/'predictions/nominal'/f'{i:04d}/summary.json').exists()]
        if not missing:
            break
        if time.monotonic()-started > 7200:
            raise RuntimeError('calibration finalization timed out awaiting tasks')
        print(json.dumps({'await_calibration': len(missing)}), flush=True)
        time.sleep(30)
    for i in ids:
        summary = json.loads((args.root/'predictions/nominal'/f'{i:04d}/summary.json').read_text())
        if any(not v['healthy'] for v in summary['health'].values()):
            raise ValueError('calibration internal output requires review')
    args.split = 'calibration'
    print(json.dumps({'calibration_finalize': 'all parts ready; merging metadata'}), flush=True)
    prepare.finalize(args)
    print(json.dumps({'calibration_finalize': 'merging probabilities'}), flush=True)
    infer.finalize(args)
    campaign = json.loads(args.campaign.read_text())
    for i, name in enumerate(calibration.model_names(campaign)):
        args.task = i
        calibration.fit(args)
    write_json(args.root/'calibration/completion.json', {'status': 'complete', 'models_and_ensembles': 15,
        'all_calibration_parts_healthy': True, 'events': sum(plan['tasks'][i]['events'] for i in ids),
        'seconds_including_wait': time.monotonic()-started, 'test_opened': False})


def test_finalize(args):
    from analysis.flavour.s6 import prepare, infer, diagnostics
    plan = json.loads((args.root/'plan.json').read_text())
    ids = [i for i, t in enumerate(plan['tasks']) if t['split'] == 'test']
    started = time.monotonic()
    while True:
        missing = {kind: [i for i in ids if not (args.root/path/f'{i:04d}/summary.json').exists()]
                   for kind, path in [('inference', 'predictions/nominal'), ('diagnostics', 'evaluation/diagnostics')]}
        if not any(missing.values()):
            break
        print(json.dumps({'await_test': {k: len(v) for k, v in missing.items()}}), flush=True)
        if time.monotonic()-started > 21600:
            raise RuntimeError('test finalization timed out awaiting task completion')
        time.sleep(30)
    args.split = 'test'
    prepare.finalize(args)
    infer.finalize(args)
    diagnostics.finalize(args)
    from analysis.flavour.s6.evaluate import design
    design(args)
    write_json(args.root/'evaluation/test_ready.json', {'status': 'complete', 'events': sum(plan['tasks'][i]['events'] for i in ids),
        'seconds_including_wait': time.monotonic()-started})


def evaluate_one(args):
    started = time.monotonic()
    while not (args.root/'evaluation/test_ready.json').exists():
        if time.monotonic()-started > 21600:
            raise RuntimeError('awaiting complete frozen test timed out')
        time.sleep(30)
    from analysis.flavour.s6.evaluate import one
    one(args)


def complete_decay(args):
    from analysis.flavour.s6 import transport, shift, decay
    expected = 368 if args.scope == 'transport' else 220
    started = time.monotonic()
    while len(list((args.root/args.scope/'truth_replay').glob('*/*/summary.json'))) < expected:
        if time.monotonic()-started > 10800:
            raise RuntimeError('compact decay recording timed out')
        time.sleep(30)
    transport.finalize_truth(args)
    shift.finalize(args)
    decay.evaluate(args)


def main():
    import pyarrow as pa
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['calibrate', 'test_finalize', 'evaluate_one', 'complete_decay'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--task', type=int, default=0)
    parser.add_argument('--scope', choices=['transport', 'generator'], default='transport')
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == '__main__':
    main()
