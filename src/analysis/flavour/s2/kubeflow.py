"""Run frozen S2 tasks on the existing CERN Kubeflow namespace and PVCs."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def kubectl(*args):
    return subprocess.check_output(
        ['kubectl', '--request-timeout=30s', *args], text=True)


def probe_runtime(root):
    import torch
    import pyarrow

    gpu = torch.cuda.is_available()
    result = {
        'utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version, 'torch': torch.__version__,
        'pyarrow': pyarrow.__version__, 'uid': os.getuid(),
        'GPU': torch.cuda.get_device_name() if gpu else None,
        'BF16': torch.cuda.is_bf16_supported(including_emulation=False) if gpu else False,
        'plan_readable': (root / 'h100/plan.json').is_file(),
    }
    # Exercise the mounted accepted input and result filesystem without resampling.
    result['plan_tasks'] = len(json.loads((root / 'h100/plan.json').read_text())['tasks'])
    if gpu:
        x = torch.ones((128, 128), device='cuda')
        result['cuda_matmul_finite'] = bool(torch.isfinite(x @ x).all())
    output = root / 'kubeflow' / ('probe_environment.json' if gpu else 'probe_cpu_environment.json')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


def prepare(args):
    repo = Path(__file__).resolve().parents[4]
    output = args.root / 'kubeflow'
    output.mkdir(exist_ok=True)
    record_path = output / f'execution_{args.name}.json'
    if record_path.exists() and json.loads(record_path.read_text()).get('submitted'):
        raise ValueError('recorded Job name already used; choose a new name to preserve allocation history')
    if args.task is None:
        arguments = ['analysis.flavour.s2.kubeflow', 'probe-runtime', '--root', str(args.root)]
        experiment = 'environment-probe'
    else:
        campaign = json.loads(args.campaign.read_text())
        config = campaign['experiments'][args.task]
        experiment = config['name']
        arguments = ['analysis.flavour.s2.train', '--root', str(args.root / 'h100'),
                     '--campaign', str(args.campaign), '--task', str(args.task)]
    script = '\n'.join([
        'set -o pipefail',
        'which() { command -v "$@"; }',
        'source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh',
        # The complete LCG view probes optional GUI programs absent from this
        # minimal image. The actual Python imports/runtime below are mandatory.
        'set -e',
        f'export PYTHONPATH={shlex.quote(str(repo / "src"))}:${{PYTHONPATH:-}}',
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',
        f'cd {shlex.quote(str(repo))}',
        'exec python -m ' + shlex.join(arguments),
    ])
    if args.task is not None:
        expected_gpu = campaign['hardware']['GPU']
        check = ('import torch; actual=torch.cuda.get_device_name(); '
                 f'assert actual == {expected_gpu!r}, (actual, {expected_gpu!r})')
        script = script.replace('exec python -m ',
                                'python -c ' + shlex.quote(check) + '\nexec python -m ')
    resources = {'cpu': '4' if args.task is not None else '2',
                 'memory': '12Gi' if args.task is not None else '4Gi'}
    if not args.cpu_only:
        resources['nvidia.com/gpu'] = 1
    spec = {
        'restartPolicy': 'Never', 'serviceAccountName': 'default-editor',
        'securityContext': {'runAsUser': os.getuid(), 'runAsGroup': os.getgid()},
        'containers': [{
            'name': 's2', 'image': args.image,
            'command': (['/usr/bin/timeout', '300'] if args.task is None else []) + ['/bin/bash', '-c', script],
            'resources': {'requests': resources, 'limits': resources},
            'volumeMounts': [{'name': 'cvmfs', 'mountPath': '/cvmfs', 'readOnly': True,
                             'mountPropagation': 'HostToContainer'}],
        }],
        'volumes': [{'name': n, 'persistentVolumeClaim': {'claimName': n}}
                    for n in ['cvmfs']],
    }
    if args.gpu_product:
        spec['nodeSelector'] = {'nvidia.com/gpu.product': args.gpu_product}
    job = {
        'apiVersion': 'batch/v1', 'kind': 'Job',
        'metadata': {'name': args.name, 'namespace': args.namespace,
                     'labels': {'app': 'pinpoint-s2'}},
        'spec': {'backoffLimit': 0, 'activeDeadlineSeconds': 86400 if args.task is not None else 3600,
                 'template': {'metadata': {'labels': {'app': 'pinpoint-s2',
                    'mount-eos': 'true', 'mount-kerberos-secret': 'true'}}, 'spec': spec}},
    }
    path = output / f'{args.name}.job.json'
    path.write_text(json.dumps(job, indent=2) + '\n')
    evidence = {
        'utc': datetime.now(timezone.utc).isoformat(), 'backend': 'kubeflow',
        'context': kubectl('config', 'current-context').strip(),
        'namespace': args.namespace, 'job': args.name, 'experiment': experiment,
        'root': str(args.root / 'h100') if args.task is not None else str(args.root),
        'stage': 'train' if args.task is not None else 'probe',
        'task': args.task, 'job_spec': job, 'submitted': False,
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
    }
    if args.apply:
        # create rejects an existing job; it cannot silently replace a live writer.
        created = json.loads(kubectl('create', '-f', str(path), '-o', 'json'))
        evidence.update(submitted=True, uid=created['metadata']['uid'])
    record_path.write_text(json.dumps(evidence, indent=2) + '\n')
    print(path)


def pod_allocation(pod, execution, now):
    """Count reserved GPU time from scheduling, including image pull and init."""
    status = pod['status']
    scheduled = next((c for c in status.get('conditions', [])
                      if c['type'] == 'PodScheduled' and c['status'] == 'True'), None)
    terminal = status.get('phase') in ['Succeeded', 'Failed']
    finished = [s['state']['terminated']['finishedAt']
                for s in status.get('containerStatuses', []) + status.get('initContainerStatuses', [])
                if s.get('state', {}).get('terminated', {}).get('finishedAt')]
    start = datetime.fromisoformat(scheduled['lastTransitionTime'].replace('Z', '+00:00')) if scheduled else None
    main_finished = any(s.get('state', {}).get('terminated', {}).get('finishedAt')
                        for s in status.get('containerStatuses', []))
    if terminal and not main_finished and pod.get('job_terminal_time'):
        finished.append(pod['job_terminal_time'])
    end = datetime.fromisoformat(max(finished).replace('Z', '+00:00')) if terminal and finished else None
    seconds = max(0., ((end or now) - start).total_seconds()) if start else 0.
    # A terminal pod without termination timestamps has an unknown endpoint.
    complete = not (terminal and start and end is None)
    def request(container, resource):
        value = str(container.get('resources', {}).get('requests', {}).get(resource, '0'))
        return float(value.rstrip('m')) / (1000 if value.endswith('m') else 1)

    def reserved(resource):
        sidecars, peak_init = 0., 0.
        for container in pod['spec'].get('initContainers', []):
            amount = request(container, resource)
            if container.get('restartPolicy') == 'Always':
                sidecars += amount
                peak_init = max(peak_init, sidecars)
            else:
                peak_init = max(peak_init, sidecars + amount)
        return max(peak_init, sidecars + sum(request(c, resource) for c in pod['spec']['containers']))

    gpu, cpus = reserved('nvidia.com/gpu'), reserved('cpu')
    exits = [s.get('state', {}).get('terminated', {}).get('exitCode')
             for s in status.get('containerStatuses', []) if s['name'] == 's2']
    return {
        'backend': 'kubeflow', 'namespace': execution['namespace'], 'job': execution['job'],
        'pod': pod['metadata']['name'], 'pod_uid': pod['metadata']['uid'],
        'root': execution['root'], 'experiment': execution['experiment'], 'stage': execution['stage'],
        'status': 4 if terminal else 2 if start else 1,
        'pod_phase': status.get('phase'), 'node': pod['spec'].get('nodeName'),
        'exit_code': exits[0] if exits else None, 'seconds': seconds if complete else None,
        'GPUs': gpu, 'GPU_hours': seconds * gpu / 3600,
        'CPU_core_hours': seconds * cpus / 3600,
        'running_estimate': bool(start and not terminal),
        'started_at': start.isoformat() if start else None,
        'finished_at': end.isoformat() if end else None,
        'allocation_complete': complete,
    }


def allocations(root, now):
    executions = [json.loads(p.read_text()) for p in root.rglob('execution_*.json')]
    executions = [e for e in executions if e.get('backend') == 'kubeflow' and e.get('submitted')]
    if not executions:
        return []
    evidence_dir = root / 'kubeflow'
    cache_path = evidence_dir / 'pod_accounting_snapshots.json'
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    rows = []
    for namespace in sorted({e['namespace'] for e in executions}):
        pods = json.loads(kubectl('get', 'pods', '-n', namespace, '-l', 'app=pinpoint-s2', '-o', 'json'))['items']
        jobs = json.loads(kubectl('get', 'jobs', '-n', namespace, '-l', 'app=pinpoint-s2', '-o', 'json'))['items']
        terminal_times = {j['metadata']['uid']: max(
            [c['lastTransitionTime'] for c in j.get('status', {}).get('conditions', [])
             if c['type'] in ['Complete', 'Failed'] and c['status'] == 'True'], default=None)
                         for j in jobs}
        for pod in pods:
            owners = {o['uid'] for o in pod['metadata'].get('ownerReferences', [])}
            execution = next((e for e in executions if e['uid'] in owners), None)
            if execution:
                # Keep only allocation fields; never persist injected auth env/volumes.
                safe = {'metadata': {k: pod['metadata'][k] for k in ['name', 'uid']},
                        'spec': {'nodeName': pod['spec'].get('nodeName'),
                                 'containers': [{'name': c['name'], 'resources': c.get('resources', {})}
                                                for c in pod['spec']['containers']],
                                 'initContainers': [{k: c[k] for k in ['name', 'resources', 'restartPolicy'] if k in c}
                                                    for c in pod['spec'].get('initContainers', [])]},
                        'status': pod['status'], 'job_terminal_time': terminal_times.get(execution['uid'])}
                row = pod_allocation(safe, execution, now)
                cache[pod['metadata']['uid']] = {'pod': safe, 'allocation': row,
                                                'observed_utc': now.isoformat()}
        current_owners = {o['uid'] for p in pods for o in p['metadata'].get('ownerReferences', [])}
        missing = [e for e in executions if e['namespace'] == namespace and e['uid'] not in current_owners
                   and not any(v['allocation']['job'] == e['job'] and v['allocation']['status'] == 4
                               for v in cache.values())]
        if missing:
            events = json.loads(kubectl('get', 'events', '-n', namespace, '-o', 'json'))['items']
            for execution in missing:
                created = [e for e in events if e.get('involvedObject', {}).get('uid') == execution['uid']
                           and e.get('reason') == 'SuccessfulCreate' and e.get('message', '').startswith('Created pod: ')]
                for event in created:
                    name = event['message'].split('Created pod: ', 1)[1]
                    pevents = [e for e in events if e.get('involvedObject', {}).get('name') == name
                               and e.get('involvedObject', {}).get('kind') == 'Pod']
                    scheduled = next((e for e in pevents if e.get('reason') == 'Scheduled'), None)
                    never_scheduled = any(e.get('reason') == 'FailedScheduling' for e in pevents)
                    end = terminal_times.get(execution['uid'])
                    if not end or (not scheduled and not never_scheduled):
                        continue
                    # Deadline deletion may remove the pod before the first poll.
                    # Scheduler events plus Job terminal time bound that allocation.
                    conditions = []
                    if scheduled:
                        conditions = [{'type': 'PodScheduled', 'status': 'True',
                                       'lastTransitionTime': scheduled.get('firstTimestamp') or scheduled['eventTime']}]
                    job = next(j for j in jobs if j['metadata']['uid'] == execution['uid'])
                    safe = {'metadata': {'name': name, 'uid': pevents[0]['involvedObject']['uid']},
                            'spec': job['spec']['template']['spec'],
                            'status': {'phase': 'Failed', 'conditions': conditions},
                            'job_terminal_time': end}
                    row = pod_allocation(safe, execution, now)
                    row['allocation_basis'] = 'retained_scheduler_events_to_job_terminal_time_conservative_upper_bound'
                    cache[safe['metadata']['uid']] = {'allocation': row, 'observed_utc': now.isoformat()}
    cache_path.write_text(json.dumps(cache, indent=2) + '\n')
    for execution in executions:
        matched = [v['allocation'] for v in cache.values()
                   if v['allocation']['namespace'] == execution['namespace'] and
                   v['allocation']['job'] == execution['job']]
        if not matched:
            rows.append({**{k: execution[k] for k in ['root', 'experiment', 'stage', 'namespace', 'job']},
                         'backend': 'kubeflow', 'status': 'missing_pod_history', 'seconds': None})
        for row in matched:
            # A disappeared active pod cannot silently retain an old zero/partial cost.
            if row['status'] not in [3, 4] and cache[row['pod_uid']]['observed_utc'] != now.isoformat():
                row = {**row, 'seconds': None, 'status': 'missing_terminal_pod_history'}
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'probe-runtime'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--namespace', default='xcheng')
    parser.add_argument('--name', default='pinpoint-s2-env-probe')
    parser.add_argument('--image', default='docker.io/library/almalinux:9')
    parser.add_argument('--campaign', type=Path)
    parser.add_argument('--task', type=int)
    parser.add_argument('--gpu-product')
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if args.action == 'probe-runtime':
        probe_runtime(args.root)
    else:
        if args.task is not None and (args.cpu_only or args.campaign is None or not args.gpu_product):
            parser.error('training requires --campaign and --gpu-product matching its frozen hardware')
        prepare(args)


if __name__ == '__main__':
    main()
