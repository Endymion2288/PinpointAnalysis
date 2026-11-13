"""Run independent frozen experiments on one GPU; wait for every child."""
import argparse
import fcntl
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from analysis.flavour.inventory import write_json


def slots_in_use(directory, allocation_job):
    total = 0
    for path in directory.glob('parallel_*.json'):
        record = json.loads(path.read_text())
        parent = record.get('allocation_job', record.get('job', path.stem.removeprefix('parallel_')))
        if parent == allocation_job:
            total += sum('finished_at' not in process for process in record['processes'])
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--tasks', type=int, nargs='+', required=True)
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--training-module', default='analysis.flavour.s3.train')
    parser.add_argument('--append-record', action='store_true')
    parser.add_argument('--claim-tasks', action='store_true')
    parser.add_argument('--allocation-job', help='parent Job when reusing an already allocated Pod')
    parser.add_argument('--allocation-concurrency', type=int, help='total fits across supervisors sharing this GPU')
    args = parser.parse_args()
    args.claim_tasks = args.claim_tasks or args.training_module == 'analysis.flavour.s3.train_repair'
    if args.concurrency < 1 or len(set(args.tasks)) != len(args.tasks):
        raise ValueError('positive concurrency and distinct task IDs required')
    allocation_job = args.allocation_job or args.name
    allocation_limit = args.allocation_concurrency if args.allocation_concurrency is not None else args.concurrency
    if allocation_limit < 1:
        raise ValueError('positive allocation concurrency required')
    configs = json.loads(args.campaign.read_text())['experiments']
    names = [configs[t]['name'] for t in args.tasks]
    if len(names) != len(set(names)):
        raise ValueError('experiment names must be distinct')
    directory = args.root / 'kubeflow'
    evidence = {'job': args.name, 'pod': os.environ.get('HOSTNAME'),
                'allocation_job': allocation_job, 'allocation_concurrency_limit': allocation_limit,
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
    # A local filesystem lock is shared by supervisors in this worker/Pod.
    # Kernel locks are released on exit, so a stopped supervisor leaves no
    # stale scheduling lock. EOS retains only the small process evidence.
    scratch = Path(os.environ.get('_CONDOR_SCRATCH_DIR', '/tmp'))
    lock_path = scratch / f'pinpoint_slots_{args.root.name}_{allocation_job}.lock'
    while queue or active:
        while queue and len(active) < args.concurrency:
            with lock_path.open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if slots_in_use(directory, allocation_job) >= allocation_limit:
                    break
                task = queue.pop(0)
                claim = args.root/'claims'/configs[task]['name']
                if args.claim_tasks:
                    claim.parent.mkdir(exist_ok=True)
                    try:
                        claim.mkdir()
                    except FileExistsError:
                        print(f'{configs[task]["name"]}: claimed by another allocated worker; no duplicate fit',flush=True)
                        continue
                    write_json(claim/'state.json',{'job':args.name,'status':'running',
                        'host':os.environ.get('HOSTNAME'),'utc':datetime.now(timezone.utc).isoformat()})
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
                if args.claim_tasks:
                    write_json(args.root/'claims'/row['experiment']/'state.json',
                               {'job':args.name,'status':'complete' if code==0 else 'failed',
                                'exit_code':code,'utc':row['finished_at']})
                active.remove((child, stream, row))
                write_json(record, evidence)
                if code and args.claim_tasks:
                    queue.clear()
        if active or queue:
            time.sleep(2)
    evidence['finished_at'] = datetime.now(timezone.utc).isoformat()
    write_json(record, evidence)
    if any(p['exit_code'] != 0 for p in evidence['processes']):
        raise SystemExit('one or more independent training processes failed; see per-task logs')
    if args.claim_tasks:
        while True:
            manifests=[args.root/'experiments'/name/'manifest.json' for name in names]
            statuses=[json.loads(p.read_text()).get('status') if p.exists() else None for p in manifests]
            claims=[args.root/'claims'/name/'state.json' for name in names]
            if any(p.exists() and json.loads(p.read_text()).get('status')=='failed' for p in claims):
                raise SystemExit('a peer fit failed; review its claim and logs before reassignment')
            if all(value=='complete' for value in statuses):
                break
            time.sleep(30)


if __name__ == '__main__':
    main()
