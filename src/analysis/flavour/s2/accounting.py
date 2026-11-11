"""Account for HTCondor and Kubeflow allocations, including failed attempts."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

from analysis.flavour.inventory import write_json
from analysis.flavour.s2.kubeflow import allocations as kubeflow_allocations


FIELDS=['ClusterId','ProcId','JobStatus','ExitCode','RemoteWallClockTime','RequestGPUs',
        'RequestCpus','NumJobStarts','JobCurrentStartDate','HoldReason']


def submission_jobs(path):
    evidence=json.loads(path.read_text())
    match=re.search(r'submitted to cluster (\d+)',evidence.get('response',''))
    if not match:
        return []
    cluster=int(match[1])
    submit=evidence['submit']
    arguments=shlex.split(next(line.split('=',1)[1] for line in submit.splitlines() if line.startswith('arguments =')))
    stage=evidence['stage']
    root=path.parent
    if stage=='train':
        if evidence.get('submitted_experiments'):
            names=[row['config']['name'] for row in evidence['submitted_experiments']]
        else:
            campaign=Path(arguments[arguments.index('--campaign')+1])
            experiments=json.loads(campaign.read_text())['experiments']
            selected=re.search(r'^queue task in (.+)$',submit,re.M)
            tasks=[int(i) for i in selected[1].split(',')] if selected else list(range(len(experiments)))
            names=[experiments[t]['name'] for t in tasks]
    else:
        queue=re.search(r'^queue (\d+)$',submit,re.M)
        named_queue=re.search(r'^queue \w+ in (.+)$',submit,re.M)
        count=int(queue[1]) if queue else len(named_queue[1].replace(',',' ').split()) if named_queue else 1
        names=[stage]*count
    return [{'schedd':evidence['chosen'],'cluster':cluster,'process':i,'stage':stage,
             'experiment':name,'root':str(root),'submission':str(path)} for i,name in enumerate(names)]


def run(args):
    jobs=[]
    for path in sorted(args.root.rglob('submission_*.json')):
        jobs.extend(submission_jobs(path))
    sources={}
    for job in jobs:
        sources.setdefault(job['schedd'],set()).add(job['cluster'])
    ads={}
    for schedd,clusters in sources.items():
        constraint=' || '.join(f'ClusterId == {c}' for c in sorted(clusters))
        for command in ['condor_history','condor_q']:
            result=subprocess.run([command,'-name',schedd,'-constraint',constraint,
                                   '-json','-attributes',','.join(FIELDS)],capture_output=True,text=True)
            if result.returncode:
                raise RuntimeError(f'{command}: {result.stderr}')
            for raw in json.loads(result.stdout or '[]'):
                ad={k.lower():v for k,v in raw.items()}
                ads[(schedd,ad['clusterid'],ad['procid'])]=ad
    now=time.time()
    allocations=[]
    for job in jobs:
        ad=ads.get((job['schedd'],job['cluster'],job['process']))
        if ad is None:
            allocations.append({**job,'status':'not_in_current_queue_or_retained_history','seconds':None})
            continue
        seconds=float(ad.get('remotewallclocktime',0.))
        running=ad.get('jobstatus')==2
        if running:
            seconds+=max(0.,now-ad.get('jobcurrentstartdate',now))
        gpu_count=float(ad.get('requestgpus',0))
        allocations.append({**job,'status':ad.get('jobstatus'),'exit_code':ad.get('exitcode'),
            'seconds':seconds,'GPUs':gpu_count,'GPU_hours':seconds*gpu_count/3600,
            'CPU_core_hours':seconds*float(ad.get('requestcpus',0))/3600,
            'starts':ad.get('numjobstarts',0),'running_estimate':running,'hold_reason':ad.get('holdreason')})
    allocations.extend(kubeflow_allocations(args.root, datetime.fromtimestamp(now, timezone.utc)))
    utc=datetime.now(timezone.utc).isoformat()
    scopes={args.root:allocations}
    for directory in {Path(row['root']) for row in allocations}:
        if args.root in directory.parents:
            scopes[directory]=[row for row in allocations if Path(row['root'])==directory]
    for directory,rows in scopes.items():
        report=accounting_report(directory,rows,utc)
        report['organization_budget_root']=str(args.root)
        write_json(directory/'campaign_accounting.json',report)


def accounting_report(root, allocations, utc):
    by_experiment={}
    for row in allocations:
        if row['stage']!='train' or row['root']!=str(root):
            continue
        group=by_experiment.setdefault(row['experiment'],{'allocated_seconds':0.,'GPU_hours':0.,
            'jobs':[],'allocation_complete':True,'all_terminal':True})
        group['jobs'].append([row['cluster'],row['process']] if 'cluster' in row else
                             ['kubeflow',row['namespace'],row['job'],row.get('pod')])
        if row['seconds'] is None:
            group['allocation_complete']=False
        else:
            group['allocated_seconds']+=row['seconds']
            group['GPU_hours']+=row['GPU_hours']
            group['all_terminal'] &= row['status'] in [3,4]
    for name,group in by_experiment.items():
        path=root/'experiments'/name/'manifest.json'
        manifest=json.loads(path.read_text()) if path.exists() else {}
        complete=manifest.get('status')=='complete' and group['all_terminal'] and group['allocation_complete']
        group['model_complete']=manifest.get('status')=='complete'
        group['epoch_history_seconds']=manifest.get('seconds')
        group['untracked_allocation_seconds']=max(0.,group['allocated_seconds']-manifest['seconds']) if complete else None
    return {'utc':utc,
        'GPU_hours':sum(row.get('GPU_hours',0.) for row in allocations),
        'CPU_core_hours':sum(row.get('CPU_core_hours',0.) for row in allocations),
        'missing_history_records':sum(row['seconds'] is None for row in allocations),
        'allocations':allocations,'experiments':by_experiment,
        'scope':'actual_remote_allocation_including_failed_attempts_setup_and_resume; Kubeflow includes scheduled image-pull/init time',
        'source':'only recorded S2 HTCondor ClusterIds and Kubeflow Job UIDs; unrelated jobs are excluded'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    run(parser.parse_args())


if __name__=='__main__':
    main()
