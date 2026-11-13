"""Actual allocation durations, including readiness, failed and empty workers."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import subprocess

from analysis.flavour.inventory import write_json


def utc(value):
    return datetime.fromisoformat(value.replace('Z','+00:00'))


def refresh(root):
    now=datetime.now(timezone.utc)
    records=[]
    snapshots=root/'audit'/'backends';snapshots.mkdir(parents=True,exist_ok=True)
    for record_path in sorted((root/'logs').glob('submission_*.json')):
        submission=json.loads(record_path.read_text())
        if not submission.get('submitted'):continue
        row={'name':submission['name'],'stage':submission['stage'],'backend':submission['backend']}
        gpu=submission['stage'] not in {'data','assemble','query_baseline','compute_profile','budget_replay','matched_plan','validation_job','rel_diagnostic','frontend_profile','functional_probe'}
        if submission['backend']=='kubeflow':
            response=subprocess.run(['kubectl','--request-timeout=30s','get','pods','-n','xcheng','-l',
                'job-name='+submission['name'],'-o','json'],capture_output=True,text=True)
            cache=snapshots/(submission['name']+'.pods.json')
            if response.returncode:
                if not cache.exists():
                    records.append({**row,'status':'snapshot_unavailable','error':response.stderr});continue
                payload=json.loads(cache.read_text());row['cached_snapshot']=True
            else:
                payload=json.loads(response.stdout)
                if payload['items']:write_json(cache,payload)
                elif cache.exists():payload=json.loads(cache.read_text());row['cached_snapshot']=True
            rows=[]
            for pod in payload.get('items',[]):
                status=pod.get('status',{})
                start=status.get('startTime')
                terminated=[]
                gpu_start,gpu_end=None,None
                for container in status.get('containerStatuses',[]):
                    state=container.get('state',{})
                    ended=state.get('terminated',{})
                    running=state.get('running',{})
                    if ended.get('finishedAt'):terminated.append(utc(ended['finishedAt']))
                    if container['name']=='s4':
                        gpu_start=ended.get('startedAt',running.get('startedAt',start))
                        gpu_end=ended.get('finishedAt')
                # The GPU belongs to the Pod during init, dependency setup and
                # sidecar termination, including time outside the main process.
                gpu_start=start or gpu_start
                if status.get('phase') in {'Succeeded','Failed'} and terminated:
                    gpu_end=max(terminated).isoformat()
                stopped=root/'logs'/f'{submission["name"]}_allocation_end.json'
                allocation_kind='observed_pod_duration'
                if stopped.exists():
                    ended=json.loads(stopped.read_text());gpu_end=ended['allocation_end_utc']
                    allocation_kind=ended.get('estimate_kind','conservative_upper_bound')
                duration=(utc(gpu_end)-utc(gpu_start)).total_seconds() if gpu_start and gpu_end else (now-utc(gpu_start or start)).total_seconds() if (gpu_start or start) else 0.
                requested=next(c for c in submission['pod_admission']['spec']['containers'] if c['name']=='s4')['resources']
                cpu=str(requested.get('requests',{}).get('cpu',0))
                cores=float(cpu[:-1])/1000 if cpu.endswith('m') else float(cpu)
                rows.append({'pod':pod['metadata']['name'],'phase':status.get('phase'),
                    'started_at':gpu_start or start,'finished_at':gpu_end,'allocation_seconds':max(0,duration),
                    'flavor':submission['flavor'],'GPU_resources':submission['pod_admission']['spec']['containers'][0]['resources'],
                    'gpu_container_terminal':bool(gpu_end),'allocation_estimate_kind':allocation_kind,
                    'training_container_requested_CPU_cores':cores})
            row.update(pods=rows,allocation_seconds=sum(r['allocation_seconds'] for r in rows),
                status='terminal' if rows and all(r['gpu_container_terminal'] for r in rows) else 'active_or_pending')
        else:
            match=re.search(r'cluster (\d+)',submission.get('response',''))
            if not match:records.append({**row,'status':'missing_cluster_id'});continue
            cluster=match.group(1);schedd=submission['schedd']
            cache=snapshots/(submission['name']+'.condor.json')
            ads={(ad['ClusterId'],ad['ProcId']):ad for ad in json.loads(cache.read_text())} if cache.exists() else {}
            expected=len(submission['tasks']) if submission['stage']=='data' else 1
            if len(ads)==expected and all(ad.get('JobStatus') in {3,4} for ad in ads.values()):
                row['cached_terminal_snapshot']=True
            else:
                queue=subprocess.run(['condor_q','-name',schedd,cluster,'-json'],capture_output=True,text=True)
                try:
                    # History is ordered by completion, not ClusterId. A
                    # lower-cluster stop condition can hide a later finish.
                    history=subprocess.run(['condor_history','-name',schedd,'-constraint','ClusterId == '+cluster,
                        '-match',str(expected),'-json'],capture_output=True,text=True,timeout=20)
                except subprocess.TimeoutExpired:
                    history=None
                if history is not None and history.returncode==0:
                    for ad in json.loads(history.stdout.strip() or '[]'):ads[(ad['ClusterId'],ad['ProcId'])]=ad
                if queue.returncode==0:
                    for ad in json.loads(queue.stdout.strip() or '[]'):ads[(ad['ClusterId'],ad['ProcId'])]=ad
                if history is None:row['history_error']='20-second query timeout; retain available queue/cache and flag missing terminal evidence'
            write_json(cache,list(ads.values()))
            seconds=0.;details=[]
            for (_,proc),ad in ads.items():
                duration=ad.get('RemoteWallClockTime',0.)
                if ad.get('JobStatus')==2:
                    elapsed=now.timestamp()-ad.get('JobCurrentStartDate',ad.get('JobStartDate',now.timestamp()))
                    duration=max(duration,elapsed)
                seconds+=duration
                details.append({'proc':proc,'status':ad.get('JobStatus'),'allocation_seconds':duration,
                    'exit_code':ad.get('ExitCode'),'request_cpus':ad.get('RequestCpus'),
                    'GPU_name':ad.get('GPUs_DeviceName') or ad.get('MATCH_GPUs_DeviceName') or ad.get('MATCH_EXP_GPUsDeviceName'),
                    'assigned_GPUs':ad.get('AssignedGPUs'),
                    'remote_host':ad.get('LastRemoteHost',ad.get('RemoteHost'))})
            row.update(cluster=cluster,schedd=schedd,jobs=details,allocation_seconds=seconds,
                status='terminal' if details and all(r['status'] in {3,4} for r in details) else 'active_or_missing_history')
        row['device_hours']=row.get('allocation_seconds',0.)/3600 if gpu else 0.
        row['CPU_allocation_hours']=(sum(r.get('allocation_seconds',0)*r.get('request_cpus',0)/3600 for r in row.get('jobs',[]))
            if submission['backend']=='condor' else sum(r['allocation_seconds']*r['training_container_requested_CPU_cores']/3600 for r in row.get('pods',[])))
        records.append(row)
    device_hours=sum(r.get('device_hours',0) for r in records)
    result={'utc':now.isoformat(),'allocations':records,'organization_device_hours':device_hours,
        'cap_device_hours':60,'remaining_device_hours':60-device_hours,
        'CPU_allocation_hours':sum(r.get('CPU_allocation_hours',0) for r in records),
        'CPU_scope':'Condor requested CPUs plus Kubeflow training-container CPU requests; platform sidecars excluded',
        'scope':'actual durations by allocation; MIG and full GPUs kept distinct, no equivalent-FLOPs conversion; all readiness/failure time included'}
    write_json(root/'campaign_accounting.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    result=refresh(parser.parse_args().root)
    print(json.dumps({k:v for k,v in result.items() if k!='allocations'}))


if __name__=='__main__':main()
