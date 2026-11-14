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
    prepare.finalize(args)
    infer.finalize(args)
    campaign = json.loads(args.campaign.read_text())
    for i, name in enumerate(calibration.model_names(campaign)):
        args.task = i
        calibration.fit(args)
    write_json(args.root/'calibration/completion.json', {'status': 'complete', 'models_and_ensembles': 15,
        'all_calibration_parts_healthy': True, 'events': sum(plan['tasks'][i]['events'] for i in ids),
        'seconds_including_wait': time.monotonic()-started, 'test_opened': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['calibrate'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == '__main__':
    main()
