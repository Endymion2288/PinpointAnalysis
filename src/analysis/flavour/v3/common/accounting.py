"""Complete HTCondor and Kubeflow resource accounting for v3 runtime submissions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess

from analysis.flavour.inventory import write_json


def _query_condor_jobs(schedd: str) -> tuple[dict[str, dict], dict[str, dict]]:
    q_map: dict[str, dict] = {}
    h_map: dict[str, dict] = {}
    fields = ['ClusterId', 'ProcId', 'JobStatus', 'ExitCode', 'RemoteWallClockTime', 'RequestCpus', 'RequestGPUs', 'CpusUsage', 'CommittedTime', 'EnteredCurrentStatus', 'QDate', 'JobStartDate', 'LastRemoteHost']

    for cmd, target in (
        (['condor_q', '-name', schedd, '-json', '-attributes', ','.join(fields)], q_map),
        (['condor_history', '-name', schedd, 'xcheng', '-limit', '2000', '-json', '-attributes', ','.join(fields)], h_map),
    ):
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and res.stdout.strip():
            try:
                rows = json.loads(res.stdout)
                for r in rows:
                    cid = f"{r['ClusterId']}.{r['ProcId']}"
                    target[cid] = r
            except json.JSONDecodeError:
                pass
    return q_map, h_map


def _query_single_condor_history(schedd: str, cid: str) -> dict | None:
    fields = ['ClusterId', 'ProcId', 'JobStatus', 'ExitCode', 'RemoteWallClockTime', 'RequestCpus', 'RequestGPUs', 'CpusUsage', 'CommittedTime', 'EnteredCurrentStatus', 'QDate', 'JobStartDate', 'LastRemoteHost']
    res = subprocess.run(
        ['condor_history', '-name', schedd, cid, '-limit', '1', '-json', '-attributes', ','.join(fields)],
        capture_output=True,
        text=True,
    )
    if res.returncode == 0 and res.stdout.strip():
        try:
            rows = json.loads(res.stdout)
            if rows:
                return rows[0]
        except json.JSONDecodeError:
            pass
    return None


def _parse_condor_user_log(log_path: Path) -> dict | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(errors='ignore')
    m_term = re.search(r'Job terminated\.\s+\(\d+\)\s+Normal termination \(return value (\d+)\)', text)
    if not m_term:
        return None
    m_wall = re.search(r'TimeExecute \(s\)\s*:\s*(\d+(?:\.\d+)?)', text)
    return {
        'JobStatus': 4,
        'ExitCode': int(m_term.group(1)),
        'RemoteWallClockTime': float(m_wall.group(1)) if m_wall else 0.0,
        'LastRemoteHost': None,
    }


def collect_v3_resources(root: Path) -> dict:
    root = root.resolve()
    now = datetime.now(timezone.utc)
    records = sorted((root / 'runtime').glob('*.submission.json'))

    condor_schedds = set()
    parsed = []
    for p in records:
        ev = json.loads(p.read_text())
        parsed.append((p, ev))
        if ev.get('backend', 'condor') == 'condor' and ev.get('schedd'):
            condor_schedds.add(ev['schedd'])

    condor_q_all: dict[str, dict] = {}
    condor_h_all: dict[str, dict] = {}
    for schedd in sorted(condor_schedds):
        qm, hm = _query_condor_jobs(schedd)
        condor_q_all.update(qm)
        condor_h_all.update(hm)

    kf_pods: list[dict] = []
    kf_jobs: list[dict] = []
    if any(ev.get('backend') == 'kubeflow' for _, ev in parsed):
        for target, cmd in (
            (kf_pods, ['kubectl', '--request-timeout=5s', 'get', 'pods', '-n', 'xcheng', '-l', 'app=pinpoint-v3', '-o', 'json']),
            (kf_jobs, ['kubectl', '--request-timeout=5s', 'get', 'jobs', '-n', 'xcheng', '-l', 'app=pinpoint-v3', '-o', 'json']),
        ):
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=10)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                res = None
            if res is not None and res.returncode == 0 and res.stdout.strip():
                target.extend(json.loads(res.stdout).get('items', []))

    grp_by_alloc: dict[str, dict] = {}
    for gp in sorted(root.rglob('groups/*.json')):
        try:
            gd = json.loads(gp.read_text())
        except Exception:
            continue
        alloc = gd.get('allocation_id')
        if alloc:
            grp_by_alloc[str(alloc)] = gd

    rows = []
    active_or_unresolved = []
    total_cpu_h = 0.0
    total_gpu_h = 0.0

    for p, ev in parsed:
        backend = ev.get('backend', 'condor')
        name = ev['name']
        stage = ev.get('stage', 'unknown')
        if backend == 'condor':
            m = re.search(r'submitted to cluster (\d+)', ev.get('stdout', ''))
            cid = f"{m.group(1)}.0" if m else None
            info = condor_q_all.get(cid) or condor_h_all.get(cid) if cid else None
            if info is None and cid is not None and ev.get('schedd'):
                info = _query_single_condor_history(ev['schedd'], cid)
                if info is not None:
                    condor_h_all[cid] = info
            if info is None and m is not None:
                info = _parse_condor_user_log(root / 'logs' / f"{name}.{m.group(1)}.log")
            in_queue = cid in condor_q_all if cid else True
            wall_s = float(info.get('RemoteWallClockTime') or info.get('CommittedTime') or 0.0) if info else 0.0
            cpus = int(ev.get('cpus', 1))
            gpus = 1 if ev.get('gpu') else 0
            cpu_h = wall_s * cpus / 3600.0
            gpu_h = wall_s * gpus / 3600.0
            status = int(info.get('JobStatus', -1)) if info else -1
            exit_code = info.get('ExitCode') if info else None
            status_name = {1: 'Idle', 2: 'Running', 3: 'Removed', 4: 'Completed', 5: 'Held'}.get(status, f'Status_{status}')
            if in_queue or info is None:
                active_or_unresolved.append({'name': name, 'backend': 'condor', 'cluster': cid, 'in_queue': in_queue, 'status': status})
            row = {
                'name': name,
                'stage': stage,
                'backend': 'condor',
                'schedd': ev.get('schedd'),
                'cluster_id': cid,
                'cpus': cpus,
                'gpus': gpus,
                'wall_seconds': wall_s,
                'cpu_core_hours': cpu_h,
                'gpu_hours': gpu_h,
                'job_status': status,
                'job_status_name': status_name,
                'exit_code': exit_code,
                'remote_host': info.get('LastRemoteHost') if info else None,
            }
        else:
            uid = ev.get('uid')
            pod = next((pd_item for pd_item in kf_pods if any(o.get('uid') == uid for o in pd_item.get('metadata', {}).get('ownerReferences', []))), None)
            job = next((jb for jb in kf_jobs if jb.get('metadata', {}).get('uid') == uid), None)
            snap_pod_p = root / 'runtime' / f'{name}.pod_snapshot.json'
            snap_job_p = root / 'runtime' / f'{name}.job_snapshot.json'
            from_snapshot = False
            if pod is None and snap_pod_p.exists():
                raw_snap = json.loads(snap_pod_p.read_text())
                pod = (raw_snap.get('items') or [None])[0] if 'items' in raw_snap else raw_snap
                from_snapshot = pod is not None
            if job is None and snap_job_p.exists():
                job = json.loads(snap_job_p.read_text())
            job_conds = job.get('status', {}).get('conditions', []) if job else []
            if any(c.get('type') == 'Complete' and c.get('status') == 'True' for c in job_conds):
                phase = 'Succeeded'
            elif any(c.get('type') == 'Failed' and c.get('status') == 'True' for c in job_conds):
                phase = 'Failed'
            elif from_snapshot:
                phase = 'AbortedSnapshot'
            elif pod:
                phase = pod['status'].get('phase')
            else:
                phase = 'Unknown'
            sched_t = None
            end_t = None
            if pod:
                for c in pod['status'].get('conditions', []):
                    if c.get('type') == 'PodScheduled' and c.get('status') == 'True':
                        sched_t = datetime.fromisoformat(c['lastTransitionTime'].replace('Z', '+00:00'))
                for cs in pod['status'].get('containerStatuses', []):
                    fin = cs.get('state', {}).get('terminated', {}).get('finishedAt')
                    if fin:
                        end_t = datetime.fromisoformat(fin.replace('Z', '+00:00'))
                if from_snapshot and end_t is None and snap_pod_p.exists():
                    end_t = datetime.fromtimestamp(snap_pod_p.stat().st_mtime, tz=timezone.utc)
            if job:
                if sched_t is None and job.get('status', {}).get('startTime'):
                    sched_t = datetime.fromisoformat(job['status']['startTime'].replace('Z', '+00:00'))
                if end_t is None and job.get('status', {}).get('completionTime'):
                    end_t = datetime.fromisoformat(job['status']['completionTime'].replace('Z', '+00:00'))
                if end_t is None:
                    for c in job_conds:
                        if c.get('type') in ('Complete', 'Failed') and c.get('status') == 'True' and c.get('lastTransitionTime'):
                            end_t = datetime.fromisoformat(c['lastTransitionTime'].replace('Z', '+00:00'))
                            break
            wall_s = max(0.0, ((end_t or now) - sched_t).total_seconds()) if sched_t else 0.0
            grp_rec = grp_by_alloc.get(name)
            if phase == 'Unknown' and grp_rec is not None:
                gst = grp_rec.get('status')
                if gst == 'complete':
                    phase = 'Succeeded'
                elif gst == 'failed':
                    phase = 'Failed'
            if wall_s == 0.0 and grp_rec is not None:
                wall_s = float(grp_rec.get('wall_seconds') or 0.0)
            gpu_h = wall_s / 3600.0 if ev.get('gpu') else 0.0
            if phase not in ('Succeeded', 'Failed', 'AbortedSnapshot'):
                active_or_unresolved.append({'name': name, 'backend': 'kubeflow', 'phase': phase})
            row = {
                'name': name,
                'stage': stage,
                'backend': 'kubeflow',
                'flavor': ev.get('flavor'),
                'product': ev.get('product'),
                'resource': ev.get('resource'),
                'uid': uid,
                'pod_phase': phase,
                'node': pod['spec'].get('nodeName') if pod else None,
                'wall_seconds': wall_s,
                'gpu_hours': gpu_h,
                'cpu_core_hours': None,
            }
        rows.append(row)
        total_cpu_h += float(row['cpu_core_hours'] or 0.0)
        total_gpu_h += float(row['gpu_hours'] or 0.0)

    out = {
        'status': 'complete',
        'UTC': now.isoformat(),
        'total_submissions': len(rows),
        'total_condor_cpu_core_hours': total_cpu_h,
        'total_gpu_device_hours': total_gpu_h,
        'active_or_unresolved': active_or_unresolved,
        'allocations': rows,
    }
    write_json(root / 'resources.json', out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15'))
    args = parser.parse_args()
    res = collect_v3_resources(args.root)
    print(json.dumps({k: v for k, v in res.items() if k != 'allocations'}, indent=2))


if __name__ == '__main__':
    main()
