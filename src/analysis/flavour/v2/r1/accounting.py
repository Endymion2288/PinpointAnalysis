"""Collect actual Condor and Kubeflow allocation costs, including failed attempts."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import time

from analysis.flavour.inventory import write_json


def read_json(command):
    value = subprocess.check_output(command, text=True)
    return json.loads(value) if value.strip() else []


def timestamp(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    now = time.time()
    submissions = [json.loads(p.read_text()) for p in sorted((args.root/'runtime').glob('*.submission.json'))]
    observed_devices = {}
    for pattern in ['experiments/*/manifest.json', 'diagnostics/*/manifest.json',
                    'failed_attempts/*/manifest.json']:
        for path in args.root.glob(pattern):
            record = json.loads(path.read_text())
            if record.get('allocation_id') and record.get('GPU'):
                observed_devices[record['allocation_id']] = record['GPU']
    for submission in submissions:
        command = submission.get('command', [])
        if command and command[0].endswith('.r1.profile') and '--family' in command and '--attempt' in command:
            family = command[command.index('--family')+1]
            attempt = command[command.index('--attempt')+1]
            path = args.root/'experiments'/f'profile_{family.lower()}_{attempt}'/'manifest.json'
            if path.exists():
                submission['observed_profile_GPU'] = json.loads(path.read_text()).get('GPU')
        if command and command[0].endswith('.r1.debug') and '--family' in command and '--attempt' in command:
            family = command[command.index('--family')+1]
            attempt = command[command.index('--attempt')+1]
            path = args.root/'audit'/f'diagnose_{family}_{attempt}.json'
            if path.exists():
                cases = json.loads(path.read_text()).get('cases', [])
                devices = {v.get('GPU') for v in cases}
                if len(devices)==1:
                    submission['observed_profile_GPU'] = devices.pop()
    condor = [s for s in submissions if s.get('submitted') and s.get('backend', 'condor') == 'condor']
    groups = defaultdict(list)
    for row in condor:
        match = re.search(r'cluster (\d+)', row.get('stdout', ''))
        if match:
            row['cluster'] = int(match[1])
            groups[row['schedd']].append(row)
    allocations = []
    attributes = ['ClusterId', 'ProcId', 'JobStatus', 'JobStartDate', 'JobCurrentStartDate', 'CompletionDate',
        'RemoteWallClockTime', 'RequestCpus', 'RequestGpus', 'RemoteUserCpu', 'RemoteSysCpu', 'NumJobStarts',
        'ExitCode', 'LastRemoteHost', 'RemoteHost', 'GPUs_DeviceName', 'AssignedGPUs', 'HoldReason', 'MemoryUsage']
    for schedd, rows in groups.items():
        ids = [r['cluster'] for r in rows]
        constraint = ' || '.join(f'ClusterId == {c}' for c in ids)
        common = ['-name', schedd, '-constraint', constraint, '-json', '-attributes', ','.join(attributes)]
        live = read_json(['condor_q', *common])
        history = read_json(['condor_history', *common, '-limit', str(len(ids)*2)])
        lookup = {r['ClusterId']: r for r in history+live}
        for row in rows:
            ad = lookup.get(row['cluster'])
            if ad is None:
                allocations.append({'name': row['name'], 'backend': 'condor', 'cluster': row['cluster'], 'status': 'unresolved'})
                continue
            status = ad['JobStatus']
            seconds = float(ad.get('RemoteWallClockTime', 0.))
            if status == 2 and ad.get('JobCurrentStartDate'):
                seconds += max(0., now-ad['JobCurrentStartDate'])
            allocations.append({'name': row['name'], 'backend': 'condor', 'cluster': row['cluster'],
                'status': status, 'ad': ad, 'allocation_seconds': seconds,
                'device_hours': seconds/3600 if row['gpu'] else 0.,
                'allocated_CPU_core_hours': seconds*ad.get('RequestCpus', row['cpus'])/3600,
                'GPU': observed_devices.get(str(row['cluster'])+'.0') or row.get('observed_profile_GPU') or ad.get('GPUs_DeviceName'),
                'GPU_model_provenance': 'CUDA runtime linked by allocation id, otherwise job ad'})
    kube = [s for s in submissions if s.get('submitted') and s.get('backend')=='kubeflow']
    all_pods = read_json(['kubectl', '--request-timeout=20s', '-n', 'xcheng', 'get', 'pods', '-o', 'json'])['items'] if kube else []
    all_jobs = read_json(['kubectl', '--request-timeout=20s', '-n', 'xcheng', 'get', 'jobs', '-o', 'json'])['items'] if kube else []
    live_pod_names = {p['metadata']['name'] for p in all_pods}
    jobs = {j['metadata']['name']: j for j in all_jobs}
    for row in kube:
        pods = [p for p in all_pods if p['metadata'].get('labels', {}).get('job-name')==row['name']]
        job = jobs.get(row['name'], {})
        stored = args.root/'audit'/f"{row['name']}.pods.json"
        terminal_pod = args.root/'audit'/f"{row['name']}.terminal_pod.json"
        if not pods and terminal_pod.exists():
            pods = [json.loads(terminal_pod.read_text())]
        if not pods and stored.exists():
            pods = json.loads(stored.read_text())['items']
        if not pods and row['name'] == 'pinpoint-r1-profile-b4-k1':
            prior = args.root/'audit/kubeflow_k1_pods.json'
            if prior.exists():
                pods = json.loads(prior.read_text())['items']
        stored_job = args.root/'audit'/f"{row['name']}.job.json"
        if not job and stored_job.exists():
            job = json.loads(stored_job.read_text())
        write_json(args.root/'audit'/f"{row['name']}.pods.json", {'items': pods})
        write_json(args.root/'audit'/f"{row['name']}.job.json", job)
        if not pods and job:
            end = next((c.get('lastTransitionTime') for c in job.get('status', {}).get('conditions', [])
                        if c['type'] in ['Complete', 'Failed'] and c['status'] == 'True'), None)
            seconds = max(0., (timestamp(end) if end else now)-timestamp(job['metadata']['creationTimestamp']))
            allocations.append({'name': row['name'], 'backend': 'kubeflow',
                'status': 'terminal_lifecycle_upper_bound' if end else 'unresolved', 'flavor': row['flavor'],
                'allocation_seconds': seconds, 'device_hours': seconds/3600,
                'allocated_CPU_core_hours': seconds*float(row['admitted_pod']['spec']['containers'][0]['resources']['requests'].get('cpu', '0'))/3600,
                'accounting': 'Pod already removed; Job creation-to-terminal duration is an allocation upper bound'})
        for pod in pods:
            scheduled = next((c.get('lastTransitionTime') for c in pod['status'].get('conditions', [])
                              if c['type'] == 'PodScheduled' and c['status'] == 'True'), None)
            ended = [s['state']['terminated']['finishedAt'] for s in pod['status'].get('containerStatuses', [])
                     if 'terminated' in s.get('state', {})]
            phase = pod['status']['phase']
            inferred_terminal = None
            if phase not in ['Succeeded', 'Failed'] and job and pod['metadata']['name'] not in live_pod_names:
                inferred_terminal = next((c.get('lastTransitionTime') for c in job.get('status', {}).get('conditions', [])
                    if c['type'] in ['Complete', 'Failed'] and c['status']=='True'), None)
            end = max(map(timestamp, ended)) if ended and phase in ['Succeeded', 'Failed'] else now
            if inferred_terminal:
                end = timestamp(inferred_terminal)
                phase = 'terminal_cached_Pod_with_Job_evidence'
            seconds = max(0., end-timestamp(scheduled)) if scheduled else 0.
            cpus = pod['spec']['containers'][0]['resources']['requests'].get('cpu', '0')
            cpu = float(cpus[:-1])/1000 if cpus.endswith('m') else float(cpus)
            allocations.append({'name': row['name'], 'backend': 'kubeflow', 'pod': pod['metadata']['name'],
                'status': phase, 'flavor': row['flavor'], 'node': pod['spec'].get('nodeName'),
                'GPU': observed_devices.get(row['name']),
                'allocation_seconds': seconds, 'device_hours': seconds/3600,
                'allocated_CPU_core_hours': seconds*cpu/3600,
                'accounting': ('cached PodScheduled through terminal Job transition: conservative allocation upper bound'
                    if inferred_terminal else 'from PodScheduled through last app/sidecar termination; includes image/init/failures')})
            if ended and phase in ['Succeeded', 'Failed']:
                response = subprocess.run(['kubectl', '--request-timeout=20s', '-n', 'xcheng',
                    'logs', pod['metadata']['name'], '-c', 'v2'], capture_output=True, text=True)
                if response.returncode == 0:
                    (args.root/'logs'/f"{row['name']}.log").write_text(response.stdout)
    active = [r['name'] for r in allocations if r['status'] in [1, 2, 5, 'Pending', 'Running', 'unresolved']]
    by_device = defaultdict(float)
    for row in allocations:
        if row.get('device_hours'):
            by_device[row.get('GPU') or row.get('flavor') or 'unspecified_failed_or_early_attempt'] += row['device_hours']
    write_json(args.root/'resources.json', {'UTC': datetime.now(timezone.utc).isoformat(), 'allocations': allocations,
        'device_hours': sum(r.get('device_hours', 0.) for r in allocations),
        'allocated_CPU_core_hours': sum(r.get('allocated_CPU_core_hours', 0.) for r in allocations),
        'device_hours_by_model_or_partition': dict(by_device),
        'active_or_unresolved': active, 'interactive_fit': 'recorded separately from fit manifests; launch/setup upper bounds still required',
        'heterogeneous_device_hours_are_not_FLOP_equivalent': True})


if __name__ == '__main__':
    main()
