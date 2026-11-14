"""Count all S5 allocations, including failed workers and unscheduled attempts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess


def read_command(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=25)
        return json.loads(result.stdout or '[]') if result.returncode == 0 else None
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        return None


def utc(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def record_condor(arguments):
    root, submission = arguments
    cluster = int(re.search(r'cluster (\d+)', submission['stdout']).group(1))
    expected = len(submission['tasks']) if submission['tasks'] else 1
    cache = root/'runtime/backends'/f'{submission["name"]}.condor.json'
    ads = {(r['ClusterId'], r['ProcId']): r for r in json.loads(cache.read_text())} if cache.exists() else {}
    attributes = 'ClusterId,ProcId,JobStatus,ExitCode,RemoteWallClockTime,RequestCpus,JobCurrentStartDate,JobStartDate,LastRemoteHost,RemoteHost,AssignedGPUs'
    if len(ads) != expected or any(ad.get('JobStatus') not in [3, 4] for ad in ads.values()):
        history = read_command(['condor_history', '-name', submission['schedd'], '-constraint', f'ClusterId == {cluster}',
                                '-match', str(expected), '-attributes', attributes, '-json'])
        queue = read_command(['condor_q', '-name', submission['schedd'], str(cluster), '-attributes', attributes, '-json'])
        for ad in (history or [])+(queue or []):
            ads[(ad['ClusterId'], ad['ProcId'])] = ad
    text_history = root/'runtime'/f'{submission["name"]}.history.txt'
    if len(ads) < expected and text_history.exists():
        for line in text_history.read_text().splitlines():
            c, p, status, code, wall, cpu = line.split()
            ads.setdefault((int(c), int(p)), {'ClusterId': int(c), 'ProcId': int(p), 'JobStatus': int(status),
                'ExitCode': int(code), 'RemoteWallClockTime': float(wall), 'RequestCpus': int(cpu), 'source': str(text_history)})
    cache.write_text(json.dumps(list(ads.values()), indent=2)+'\n')
    jobs = []
    for key, ad in sorted(ads.items()):
        seconds = ad.get('RemoteWallClockTime', 0.)
        if ad.get('JobStatus') == 2:
            seconds = max(seconds, datetime.now(timezone.utc).timestamp()-ad.get('JobCurrentStartDate', ad.get('JobStartDate', datetime.now(timezone.utc).timestamp())))
        jobs.append({'proc': key[1], 'status': ad.get('JobStatus'), 'exit_code': ad.get('ExitCode'),
                     'seconds': seconds, 'cpus': ad.get('RequestCpus', 0), 'remote_host': ad.get('LastRemoteHost', ad.get('RemoteHost'))})
    return {'name': submission['name'], 'backend': 'condor', 'cluster': cluster, 'schedd': submission['schedd'],
            'expected': expected, 'recorded': len(jobs), 'jobs': jobs,
            'terminal': len(jobs) == expected and all(r['status'] in [3, 4] for r in jobs),
            'device_hours': sum(r['seconds'] for r in jobs)/3600 if submission['gpu'] else 0.,
            'CPU_allocation_hours': sum(r['seconds']*r['cpus'] for r in jobs)/3600}


def record_kubeflow(root, submission):
    name = submission['name']
    directory = root/'runtime/backends'
    job = read_command(['kubectl', '--request-timeout=20s', 'get', 'job', name, '-n', 'xcheng', '-o', 'json'])
    job_path = directory/(name+'.job.json')
    if job:
        job_path.write_text(json.dumps(job, indent=2)+'\n')
    elif job_path.exists():
        job = json.loads(job_path.read_text())
    elif (root/'runtime'/f'{name}.terminal.job.json').exists():
        job = json.loads((root/'runtime'/f'{name}.terminal.job.json').read_text())
    pods = read_command(['kubectl', '--request-timeout=20s', 'get', 'pods', '-l', 'job-name='+name, '-n', 'xcheng', '-o', 'json'])
    candidates = (pods or {}).get('items', [])
    for path in directory.glob(name+'-*.terminal.pod.json'):
        candidates.append(json.loads(path.read_text()))
    if not candidates:
        for path in directory.glob(name+'-*.live.pod.json'):
            candidates.append(json.loads(path.read_text()))
    if not candidates and name == 'pinpoint-s5-screen-a':
        candidates = json.loads((root/'runtime/screen-a-pod-live.json').read_text())['items']
    by_name = {p['metadata']['name']: p for p in candidates}
    terminal_conditions = [c for c in (job or {}).get('status', {}).get('conditions', [])
                           if c['type'] in ['Complete', 'Failed'] and c['status'] == 'True']
    terminal_job_time = max((c['lastTransitionTime'] for c in terminal_conditions), default=None)
    rows = []
    for pod in by_name.values():
        status = pod.get('status', {})
        terminated = [c['state']['terminated']['finishedAt'] for c in status.get('containerStatuses', []) if 'terminated' in c.get('state', {})]
        end = max(terminated) if status.get('phase') in ['Succeeded', 'Failed'] and terminated else terminal_job_time
        estimate = 'Pod terminal snapshot' if status.get('phase') in ['Succeeded', 'Failed'] and terminated else 'Job terminal timestamp upper bound after Pod cleanup' if end else 'active allocation'
        start = status.get('startTime')
        seconds = max(0., ((utc(end) if end else datetime.now(timezone.utc))-utc(start)).total_seconds()) if start else 0.
        resources = next(c for c in pod['spec']['containers'] if c['name'] == 's5')['resources']
        cpus = resources['requests'].get('cpu', 0)
        cpus = float(str(cpus).removesuffix('m'))/(1000 if str(cpus).endswith('m') else 1)
        rows.append({'pod': pod['metadata']['name'], 'phase': status.get('phase'), 'start': start, 'end': end,
                     'duration_evidence': estimate, 'seconds': seconds, 'resources': resources, 'cpus': cpus,
                     'container_statuses': status.get('containerStatuses'), 'terminal': bool(end)})
    return {'name': name, 'backend': 'kubeflow', 'flavor': submission['flavor'], 'pods': rows,
            'terminal': bool(rows) and all(r['terminal'] for r in rows), 'job_conditions': terminal_conditions,
            'device_hours': sum(r['seconds'] for r in rows)/3600,
            'CPU_allocation_hours': sum(r['seconds']*r['cpus'] for r in rows)/3600}


def run(root):
    (root/'runtime/backends').mkdir(exist_ok=True)
    submissions = [json.loads(p.read_text()) for p in sorted((root/'runtime').glob('*.submission.json'))]
    condor = [s for s in submissions if s['backend'] == 'condor' and s['submitted']]
    with ThreadPoolExecutor(max_workers=4) as executor:
        records = list(executor.map(record_condor, [(root, s) for s in condor]))
    records += [record_kubeflow(root, s) for s in submissions if s['backend'] == 'kubeflow' and s['submitted']]
    records += [{'name': s['name'], 'backend': s['backend'], 'terminal': True, 'device_hours': 0., 'CPU_allocation_hours': 0.,
                 'reason': 'admission or submission rejected before allocation'} for s in submissions if not s['submitted']]
    pilot_seconds = sum(json.loads(p.read_text())['seconds'] for p in (root/'pilot').glob('gate*.json'))
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'allocations': records, 'all_terminal': all(r['terminal'] for r in records),
              'remote_device_hours': sum(r['device_hours'] for r in records),
              'CPU_allocation_hours': sum(r['CPU_allocation_hours'] for r in records),
              'interactive_pilot_fit_device_seconds': pilot_seconds, 'interactive_scope': 'two measured T4 fit intervals; interactive allocation boundary was not recorded',
              'budget_device_hours': 60, 'scope': 'remote actual allocations including readiness/failure; MIG and full GPU hours are separate device types, not equal throughput; reused supervisors count once'}
    (root/'campaign_accounting.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'allocations'}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)


if __name__ == '__main__':
    main()
