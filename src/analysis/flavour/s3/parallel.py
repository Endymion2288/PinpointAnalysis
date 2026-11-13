"""Run independent frozen experiments on one GPU; wait for every child."""
import argparse
from datetime import datetime, timezone
import json
import os
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
    parser.add_argument('--tasks', type=int, nargs='+', required=True)
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--training-module', default='analysis.flavour.s3.train')
    parser.add_argument('--append-record', action='store_true')
    args = parser.parse_args()
    if args.concurrency < 1 or len(set(args.tasks)) != len(args.tasks):
        raise ValueError('positive concurrency and distinct task IDs required')
    configs = json.loads(args.campaign.read_text())['experiments']
    names = [configs[t]['name'] for t in args.tasks]
    if len(names) != len(set(names)):
        raise ValueError('experiment names must be distinct')
    directory = args.root / 'kubeflow'
    evidence = {'job': args.name, 'pod': os.environ.get('HOSTNAME'),
                'concurrency_limit': args.concurrency, 'processes': []}
    record = directory / f'parallel_{args.name}.json'
    if args.append_record and record.exists():
        previous = json.loads(record.read_text())
        if any('finished_at' not in process for process in previous['processes']):
            raise ValueError('cannot append while another supervisor has active processes')
        if set(names) & {p['experiment'] for p in previous['processes']}:
            raise ValueError('append wave must have different experiment names')
        evidence['processes'] = previous['processes']
    queue = list(args.tasks)
    active = []
    while queue or active:
        while queue and len(active) < args.concurrency:
            task = queue.pop(0)
            log = directory / f'{args.name}_task{task}.log'
            stream = log.open('w')
            child = subprocess.Popen([sys.executable, '-m', args.training_module,
                '--root', str(args.root), '--campaign', str(args.campaign), '--task', str(task)],
                stdout=stream, stderr=subprocess.STDOUT)
            row = {'task': task, 'experiment': configs[task]['name'], 'pid': child.pid,
                   'started_at': datetime.now(timezone.utc).isoformat(), 'log': str(log)}
            evidence['processes'].append(row)
            active.append((child, stream, row))
            write_json(record, evidence)
        for child, stream, row in active[:]:
            code = child.poll()
            if code is not None:
                stream.close()
                row.update(exit_code=code, finished_at=datetime.now(timezone.utc).isoformat())
                active.remove((child, stream, row))
                write_json(record, evidence)
        if active:
            time.sleep(2)
    evidence['finished_at'] = datetime.now(timezone.utc).isoformat()
    write_json(record, evidence)
    if any(p['exit_code'] != 0 for p in evidence['processes']):
        raise SystemExit('one or more independent training processes failed; see per-task logs')


if __name__ == '__main__':
    main()
