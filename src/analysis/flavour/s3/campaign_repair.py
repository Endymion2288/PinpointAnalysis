"""Run prespecified S3 repair waves in one allocated GPU, then read back fits."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from analysis.flavour.inventory import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--concurrency', type=int, default=8)
    parser.add_argument('--tasks', type=int, nargs='+', required=True)
    args = parser.parse_args()
    configurations = json.loads(args.campaign.read_text())['experiments']
    screen = [t for t in args.tasks if configurations[t]['seed'] == 42]
    confirmation = [t for t in args.tasks if configurations[t]['seed'] != 42]
    record_path = args.root / 'repair_campaign_state.json'
    state = {'job': args.name, 'tasks': args.tasks}
    try:
        for label, tasks in [('screen', screen), ('prespecified_confirmation', confirmation)]:
            if not tasks:
                continue
            state.update(stage=label, status='running', utc=datetime.now(timezone.utc).isoformat())
            write_json(record_path, state)
            subprocess.run([sys.executable, '-m', 'analysis.flavour.s3.parallel', '--root', str(args.root),
                '--campaign', str(args.campaign), '--name', args.name, '--tasks', *map(str, tasks),
                '--concurrency', str(args.concurrency), '--training-module', 'analysis.flavour.s3.train_repair',
                '--append-record','--claim-tasks'], check=True)
            names = [configurations[t]['name'] for t in tasks]
            while True:
                manifests=[args.root/'experiments'/name/'manifest.json' for name in names]
                statuses=[json.loads(p.read_text()).get('status') if p.exists() else None for p in manifests]
                claims=[args.root/'claims'/name/'state.json' for name in names]
                if any(p.exists() and json.loads(p.read_text()).get('status')=='failed' for p in claims):
                    raise RuntimeError('an allocated peer failed; inspect before reassigning its claim')
                if all(value=='complete' for value in statuses):
                    break
                time.sleep(30)
            for module in ['analysis.flavour.s2.readback', 'analysis.flavour.s3.inspect_repair']:
                command = [sys.executable, '-m', module, '--root', str(args.root), '--experiments', *names]
                if module.endswith('readback'):
                    command += ['--campaign', str(args.campaign)]
                subprocess.run(command, check=True)
        state.update(status='all_fits_readbacks_and_health_passed', utc=datetime.now(timezone.utc).isoformat())
        write_json(record_path, state)
    except Exception as error:
        state.update(status='failed', error=str(error), utc=datetime.now(timezone.utc).isoformat())
        write_json(record_path, state)
        raise


if __name__ == '__main__':
    main()
