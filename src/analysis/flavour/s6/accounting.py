"""Allocation costs from backend terminal records, including failed attempts."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess

from analysis.flavour.inventory import write_json


ATTRIBUTES = ['ClusterId', 'ProcId', 'JobStatus', 'ExitCode', 'ExitBySignal', 'RequestCpus', 'RequestMemory',
    'RequestGpus', 'RemoteWallClockTime', 'RemoteUserCpu', 'RemoteSysCpu', 'MemoryUsage', 'JobStartDate',
    'CompletionDate', 'EnteredCurrentStatus', 'QDate', 'RemoteHost', 'LastRemoteHost', 'HoldReason', 'RemoveReason']


def collect(args):
    groups, submissions = defaultdict(list), []
    for path in sorted((args.root/'runtime').glob('*.submission.json')):
        record = json.loads(path.read_text())
        if not record.get('submitted'):
            continue
        submissions.append(record)
        if record['backend'] == 'condor':
            matches = re.findall(r'cluster (\d+)', record.get('stdout', ''), re.I)
            if len(matches) == 1:
                groups[record['schedd']].append((int(matches[0]), record['name'], len(record.get('tasks') or [0])))
    records, snapshots = [], []
    for schedd, entries in groups.items():
        ids = [i for i, _, _ in entries]
        constraint = ' || '.join(f'ClusterId == {i}' for i in ids)
        commands = [['condor_q', '-name', schedd, '-constraint', constraint, '-json', '-attributes', ','.join(ATTRIBUTES)],
            ['condor_history', '-name', schedd, '-constraint', constraint, '-json', '-attributes', ','.join(ATTRIBUTES),
             '-limit', str(sum(n for _, _, n in entries)), '-scanlimit', '50000']]
        for command in commands:
            try:
                response = subprocess.run(command, capture_output=True, text=True, timeout=55)
                values = json.loads(response.stdout) if response.returncode == 0 else []
                snapshots.append({'schedd': schedd, 'command': command, 'exit': response.returncode,
                    'stderr': response.stderr, 'records': len(values)})
                lookup = {i: n for i, n, _ in entries}
                gpu_lookup = {s['name']: s.get('gpu', False) for s in submissions}
                for row in values:
                    records.append({**row, 'schedd': schedd, 'submission': lookup[row['ClusterId']],
                        'GPU_requested': gpu_lookup[lookup[row['ClusterId']]]})
            except subprocess.TimeoutExpired:
                snapshots.append({'schedd': schedd, 'command': command, 'error': 'timeout at 55 seconds'})
    unique = {(r['schedd'], r['ClusterId'], r['ProcId']): r for r in records}
    records = list(unique.values())
    condor_cpu = sum(float(r.get('RemoteWallClockTime') or 0)*float(r.get('RequestCpus') or 0) for r in records)/3600
    condor_gpu = sum(float(r.get('RemoteWallClockTime') or 0)*int(r['GPU_requested']) for r in records)/3600
    active = [r for r in records if r.get('JobStatus') in [1, 2, 5, 6, 7]]
    pods = {}
    for path in (args.root/'runtime').glob('*.terminal.json'):
        value = json.loads(path.read_text())
        if value.get('kind') == 'Pod' or 'containerStatuses' in value.get('status', {}):
            pods[value['metadata']['uid']] = value
    pod_costs = []
    for value in pods.values():
        status = value['status']
        for container in status.get('containerStatuses', []):
            terminal = container.get('state', {}).get('terminated')
            if not terminal or container['name'] != 's6':
                continue
            parse = lambda v: datetime.fromisoformat(v.replace('Z', '+00:00'))
            seconds = (parse(terminal['finishedAt'])-parse(terminal['startedAt'])).total_seconds()
            resources = next(c['resources'] for c in value['spec']['containers'] if c['name'] == container['name'])
            cpu = str(resources.get('requests', {}).get('cpu', 0))
            cpus = float(cpu.removesuffix('m'))/(1000 if cpu.endswith('m') else 1)
            pod_costs.append({'pod': value['metadata']['name'], 'seconds': seconds, 'device_hours': seconds/3600,
                'resources': resources, 'exit': terminal['exitCode'], 'phase': status['phase'],
                'CPU_allocation_core_hours': seconds*cpus/3600})
    write_json(args.root/'evaluation/resource_accounting.json', {'UTC': datetime.now(timezone.utc).isoformat(),
        'condor_records': records, 'queries': snapshots, 'condor_allocation_core_hours': condor_cpu,
        'condor_GPU_allocation_device_hours': condor_gpu, 'active_jobs': active, 'kubeflow': pod_costs,
        'kubeflow_GPU_allocation_device_hours': sum(p['device_hours'] for p in pod_costs),
        'kubeflow_CPU_allocation_core_hours': sum(p['CPU_allocation_core_hours'] for p in pod_costs),
        'unmeasured': ['monetary price', 'GPU utilization fraction when MIG metrics/RBAC unavailable'],
        'device_convention': 'one requested MIG instance counts as one allocated device, not an entire physical H100; flavors listed separately',
        'scope': 'backend allocation runtime including failed attempts; queue wait excluded from allocation hours, retained in backend timestamps',
        'submitted_condor_factories': [{'schedd': s, 'cluster': i, 'name': n, 'planned_rows': count,
            'recorded_rows': sum(r['ClusterId'] == i and r['schedd'] == s for r in records)}
            for s, entries in groups.items() for i, n, count in entries]})
    print(json.dumps({'condor_records': len(records), 'active_jobs': len(active), 'pods': len(pod_costs),
        'condor_core_hours': condor_cpu, 'condor_device_hours': condor_gpu}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['collect'])
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == '__main__':
    main()
