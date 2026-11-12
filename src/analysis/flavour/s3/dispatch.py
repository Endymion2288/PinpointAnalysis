"""Give a runtime-ready Kubeflow slot one still-idle HTCondor task.

No duplicate training is queued on Kubeflow. Its worker waits for an explicit
handoff after the schedd atomically holds a JobStatus==1 task. Running jobs
are excluded from that hold expression. The held copy is removed before
publishing the assignment; no simultaneous writers are possible.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from analysis.flavour.inventory import write_json


def worker(args):
    import torch
    device_name = torch.cuda.get_device_name()
    assert device_name.startswith('NVIDIA H100 NVL'), device_name
    assert torch.__version__ == '2.7.1' and torch.cuda.is_bf16_supported(including_emulation=False)
    memory_bytes = torch.cuda.get_device_properties(0).total_memory
    assert memory_bytes >= 40 * 1024**3, memory_bytes
    x = torch.ones((32, 32), device='cuda'); assert torch.isfinite(x @ x).all()
    directory = args.root / 'kubeflow' / ('dispatch_' + args.name)
    directory.mkdir(exist_ok=True)
    write_json(directory / 'ready.json', {'utc': datetime.now(timezone.utc).isoformat(),
        'GPU': device_name, 'GPU_memory_bytes': memory_bytes, 'torch': torch.__version__, 'BF16': True,
        'pod': os.environ.get('HOSTNAME'), 'pid': os.getpid(), 'worker_wait_seconds': 300})
    # A bounded startup rendezvous, not a persistent idle GPU allocation.
    end = time.monotonic() + 300
    while time.monotonic() < end:
        assignment_path = directory / 'assignment.json'
        if assignment_path.exists():
            assignment = json.loads(assignment_path.read_text())
            assigned_root = os.path.realpath(assignment['root'])
            worker_root = os.path.realpath(str(args.root))
            if assignment['job'] != args.name or assigned_root != worker_root:
                raise ValueError(f'assignment belongs to another worker/root: job={assignment["job"]!r}/{args.name!r}, root={assigned_root!r}/{worker_root!r}')
            write_json(directory / 'accepted.json', {**assignment, 'accepted_utc': datetime.now(timezone.utc).isoformat()})
            os.execv(sys.executable, [sys.executable, '-m', 'analysis.flavour.s3.train', '--root', str(args.root),
                '--campaign', assignment['campaign'], '--task', str(assignment['task'])])
        time.sleep(2)
    write_json(directory / 'unassigned_terminal.json', {'utc': datetime.now(timezone.utc).isoformat(),
        'reason': 'no idle task safely handed off within bounded runtime-ready wait'})


def command(arguments):
    result = subprocess.run(arguments, capture_output=True, text=True, timeout=45)
    if result.returncode:
        raise RuntimeError(f'{arguments[0]}: {result.stderr}')
    return result.stdout


def handoff(args):
    root = args.root.resolve(); directory = root / 'kubeflow' / ('dispatch_' + args.name)
    ready = directory / 'ready.json'
    if not ready.exists():
        print('Kubeflow slot not runtime-ready; existing Condor queue unchanged'); return
    if (directory / 'assignment.json').exists():
        print('assignment already published; no second handoff'); return
    job = json.loads(command(['kubectl', '-n', 'xcheng', 'get', 'job', args.name, '-o', 'json']))
    pods = json.loads(command(['kubectl', '-n', 'xcheng', 'get', 'pods', '-l', 'job-name=' + args.name, '-o', 'json']))['items']
    evidence = json.loads(ready.read_text())
    if not any(p['metadata']['name'] == evidence['pod'] and p['status']['phase'] == 'Running' and
               any(c.get('state', {}).get('running') for c in p['status'].get('containerStatuses', []) if c['name'] == 's3')
               for p in pods) or job.get('status', {}).get('failed'):
        raise ValueError('runtime-ready worker is no longer running')
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(evidence['utc'])).total_seconds()
    if age > 240:
        raise ValueError('worker rendezvous too close to expiry; leave Condor tasks alone')
    candidates = []
    for path in sorted(root.glob('submission_*.json')):
        submission = json.loads(path.read_text())
        match = re.search(r'submitted to cluster (\d+)', submission.get('response', ''))
        if submission['stage'] != 'train' or not match:
            continue
        cluster = int(match[1]); schedd = submission['chosen']
        ads = json.loads(command(['condor_q', '-name', schedd, str(cluster), '-json',
            '-attributes', 'ClusterId,ProcId,JobStatus']) or '[]')
        campaign_line = next(line for line in submission['submit'].splitlines() if line.startswith('arguments ='))
        import shlex
        arguments = shlex.split(campaign_line.split('=', 1)[1])
        campaign = arguments[arguments.index('--campaign') + 1]
        configs = submission['submitted_experiments']
        for ad in ads:
            if ad['JobStatus'] == 1:
                row = configs[ad['ProcId']]
                priority = [5, 14, 6].index(row['task']) if row['task'] in [5, 14, 6] else 3 + row['task']
                candidates.append((priority, schedd, cluster, ad['ProcId'], campaign, row))
    for _, schedd, cluster, process, campaign, row in sorted(candidates):
        identifier = f'{cluster}.{process}'
        expression = f'ClusterId == {cluster} && ProcId == {process} && JobStatus == 1'
        held = command(['condor_hold', '-name', schedd, '-constraint', expression,
                        '-reason', 'S3 runtime-ready Kubeflow handoff; exclude running tasks'])
        ads = json.loads(command(['condor_q', '-name', schedd, identifier, '-json', '-attributes', 'JobStatus']) or '[]')
        if len(ads) != 1 or ads[0]['JobStatus'] != 5:
            continue  # A task that started concurrently remains in Condor.
        removed = command(['condor_rm', '-name', schedd, identifier])
        ads = json.loads(command(['condor_q', '-name', schedd, identifier, '-json', '-attributes', 'JobStatus']) or '[]')
        if any(ad['JobStatus'] not in [3, 4] for ad in ads):
            raise ValueError('old held copy not removed; no assignment published')
        assignment = {'utc': datetime.now(timezone.utc).isoformat(), 'job': args.name, 'root': str(root),
            'task': row['task'], 'campaign': campaign, 'experiment': row['config']['name'],
            'source_schedd': schedd, 'source_job': identifier, 'hold_expression': expression,
            'hold_response': held, 'remove_response': removed}
        record_path = root / 'kubeflow' / f'execution_{args.name}.json'
        record = json.loads(record_path.read_text()); record.update(stage='train', task=row['task'],
            experiment=row['config']['name'], dispatch_handoff=assignment)
        write_json(record_path, record)
        write_json(directory / 'assignment.json', assignment)
        print(json.dumps(assignment)); return
    print('No still-idle Condor task available; worker remains bounded by its 300-second rendezvous')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['worker', 'handoff', 'watch'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--deadline', type=int, default=21600)
    args = parser.parse_args()
    if args.action == 'watch':
        directory = args.root / 'kubeflow' / ('dispatch_' + args.name)
        end = time.monotonic() + args.deadline
        print('Watching actual Kubeflow runtime readiness; Condor queue continues', flush=True)
        while time.monotonic() < end:
            if (directory / 'assignment.json').exists() or (directory / 'unassigned_terminal.json').exists():
                return
            if (directory / 'ready.json').exists():
                handoff(args)
            time.sleep(15)
        return
    (worker if args.action == 'worker' else handoff)(args)


if __name__ == '__main__':
    main()
