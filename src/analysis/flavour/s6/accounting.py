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
    'CompletionDate', 'EnteredCurrentStatus', 'QDate', 'RemoteHost', 'LastRemoteHost', 'HoldReason', 'RemoveReason',
    'AssignedGPUs', 'LastMatchTime', 'NumJobStarts', 'MATCH_EXP_Machine', 'ResidentSetSize', 'ResidentSetSize_RAW']


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
                # condor_q may return success and no text when no ads match.
                values = json.loads(response.stdout.strip() or '[]') if response.returncode == 0 else []
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
    for row in records:
        start, queued = row.get('JobStartDate'), row.get('QDate')
        row['initial_queue_seconds'] = max(0, start-queued) if start and queued else None
        row['actual_CPU_seconds'] = float(row.get('RemoteUserCpu') or 0)+float(row.get('RemoteSysCpu') or 0)
    condor_cpu = sum(float(r.get('RemoteWallClockTime') or 0)*float(r.get('RequestCpus') or 0) for r in records)/3600
    condor_gpu = sum(float(r.get('RemoteWallClockTime') or 0)*int(r['GPU_requested']) for r in records)/3600
    active = [r for r in records if r.get('JobStatus') in [1, 2, 5, 6, 7]]
    pods = {}
    for path in (args.root/'runtime').glob('*.terminal.json'):
        value = json.loads(path.read_text())
        for item in value.get('items', [value]):
            if item.get('kind') == 'Pod' or 'containerStatuses' in item.get('status', {}):
                pods[item['metadata']['uid']] = item
    pod_costs = []
    for value in pods.values():
        status = value['status']
        for container in status.get('containerStatuses', []):
            terminal = container.get('state', {}).get('terminated')
            if not terminal or container['name'] != 's6':
                continue
            parse = lambda v: datetime.fromisoformat(v.replace('Z', '+00:00'))
            scheduled = next((c['lastTransitionTime'] for c in status.get('conditions', [])
                              if c['type'] == 'PodScheduled' and c['status'] == 'True'), None)
            start = scheduled or terminal['startedAt']
            seconds = (parse(terminal['finishedAt'])-parse(start)).total_seconds()
            resources = next(c['resources'] for c in value['spec']['containers'] if c['name'] == container['name'])
            cpu = str(resources.get('requests', {}).get('cpu', 0))
            cpus = float(cpu.removesuffix('m'))/(1000 if cpu.endswith('m') else 1)
            pod_costs.append({'pod': value['metadata']['name'], 'seconds': seconds, 'device_hours': seconds/3600,
                'resources': resources, 'exit': terminal['exitCode'], 'phase': status['phase'],
                'CPU_allocation_core_hours': seconds*cpus/3600, 'allocation_started_at': start,
                'allocation_finished_at': terminal['finishedAt'], 'scheduled_start_observed': scheduled is not None,
                'main_container_seconds': (parse(terminal['finishedAt'])-parse(terminal['startedAt'])).total_seconds()})
    observed_jobs = {owner['uid'] for p in pods.values() for owner in p['metadata'].get('ownerReferences', [])}
    for path in (args.root/'runtime').glob('*.job_terminal.json'):
        job = json.loads(path.read_text())
        if job['metadata']['uid'] in observed_jobs:
            continue
        snapshot = path.with_name(path.name.replace('.job_terminal.json', '.last_observed.json'))
        if not snapshot.exists():
            continue
        pod = json.loads(snapshot.read_text())
        ends = [v['lastTransitionTime'] for v in job['status'].get('conditions', [])
                if v['type'] in ['Failed', 'Complete'] and v['status'] == 'True']
        starts = [v['lastTransitionTime'] for v in pod['status'].get('conditions', [])
                  if v['type'] == 'PodScheduled' and v['status'] == 'True']
        if not ends or not starts:
            continue
        parse = lambda v: datetime.fromisoformat(v.replace('Z', '+00:00'))
        seconds = (parse(max(ends))-parse(min(starts))).total_seconds()
        resources = next(c['resources'] for c in pod['spec']['containers'] if c['name'] == 's6')
        cpu = str(resources.get('requests', {}).get('cpu', 0))
        cpus = float(cpu.removesuffix('m'))/(1000 if cpu.endswith('m') else 1)
        pod_costs.append({'pod': pod['metadata']['name'], 'seconds': seconds, 'device_hours': seconds/3600,
            'resources': resources, 'exit': None, 'phase': None, 'observed_job_status': job['status'],
            'CPU_allocation_core_hours': seconds*cpus/3600, 'allocation_started_at': min(starts),
            'allocation_finished_at_upper_bound': max(ends), 'scheduled_start_observed': True,
            'allocation_estimate_kind': 'conservative upper bound to terminal Job time; Pod termination not retained',
            'failure_reason': 'unmeasured; extra exec exit137 alone does not identify the main container kill reason'})
    kube_submissions = []
    for s in submissions:
        if s['backend'] != 'kubeflow':
            continue
        job_terminal = args.root/'runtime'/f'{s["name"]}.job_terminal.json'
        cancellation = args.root/'runtime'/f'{s["name"]}.cancellation.json'
        if s['uid'] in observed_jobs:
            state = 'observed terminal Pod allocation'
        elif job_terminal.exists() and any(p['pod'].startswith(s['name']+'-') for p in pod_costs):
            state = 'terminal Job upper bound; Pod termination unavailable'
        elif cancellation.exists() and json.loads(cancellation.read_text()).get('state_before_removal', '').startswith('Pending; unschedulable'):
            state = 'unscheduled canceled request; zero allocation'
        else:
            state = 'unresolved'
        kube_submissions.append({'job': s['name'], 'uid': s['uid'], 'status': state})
    cancellation_names = {p.name.removesuffix('.cancellation.json') for p in (args.root/'runtime').glob('*.cancellation.json')}
    factories = []
    for schedd, entries in groups.items():
        for cluster, name, count in entries:
            known = [r for r in records if r['ClusterId'] == cluster and r['schedd'] == schedd]
            absent = count-len(known)
            factories.append({'schedd': schedd, 'cluster': cluster, 'name': name, 'planned_rows': count,
                'recorded_rows': len(known), 'unmaterialized_canceled_rows': absent if name in cancellation_names else 0,
                'unresolved_rows': 0 if name in cancellation_names else absent,
                'cancel_evidence': f'runtime/{name}.cancellation.json' if name in cancellation_names else None})
    write_json(args.root/'evaluation/resource_accounting.json', {'UTC': datetime.now(timezone.utc).isoformat(),
        'condor_records': records, 'queries': snapshots, 'condor_allocation_core_hours': condor_cpu,
        'condor_GPU_allocation_device_hours': condor_gpu, 'active_jobs': active, 'kubeflow': pod_costs,
        'condor_actual_CPU_core_hours': sum(r['actual_CPU_seconds'] for r in records)/3600,
        'backend_queries_complete': all(q.get('exit') == 0 for q in snapshots),
        'kubeflow_GPU_allocation_device_hours': sum(p['device_hours'] for p in pod_costs),
        'kubeflow_CPU_allocation_core_hours': sum(p['CPU_allocation_core_hours'] for p in pod_costs),
        'kubeflow_cost_contains_upper_bound': any('allocation_estimate_kind' in p for p in pod_costs),
        'kubeflow_submissions': kube_submissions,
        'unmeasured': ['monetary price', 'GPU utilization fraction when MIG metrics/RBAC unavailable'],
        'device_convention': 'one requested MIG instance counts as one allocated device, not an entire physical H100; flavors listed separately',
        'scope': 'backend allocation runtime including failed attempts; Kubeflow includes time from observed PodScheduled through main termination, including initialization/image pull; requested flavor CPU only; queue wait excluded from allocation hours',
        'factory_count_convention': 'rows absent from complete queue/history queries after factory cancellation are classified as unmaterialized canceled rows; they are not assigned synthetic runtime; canceled factories retain their pre-removal evidence',
        'submitted_condor_factories': factories})
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
