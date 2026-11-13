"""Run ready S4 dependency tasks concurrently on a formally allocated GPU."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from analysis.flavour.inventory import write_json


def memory_scope(root=Path('/sys/fs/cgroup'), membership=Path('/proc/self/cgroup')):
    """Find the effective allocation limit under host-mounted cgroup v2."""
    if membership.exists():
        path=next((line.split(':',2)[2] for line in membership.read_text().splitlines()
            if line.startswith('0::')),None)
        if path is not None:
            directory=root/path.lstrip('/')
            for scope in [directory,*directory.parents]:
                if not scope.is_relative_to(root):break
                limit=scope/'memory.max'
                if limit.exists() and limit.read_text().strip().isdigit():return scope
    # Container cgroup namespaces can expose the allocation directly at root.
    return root


def host_memory(directory=None):
    """Read the allocation's cgroup limit, including dataloader subprocesses."""
    directory=directory if directory is not None else memory_scope()
    fields={}
    for name,key in [('memory.current','current_bytes'),('memory.max','limit_bytes'),('memory.peak','peak_bytes')]:
        path=directory/name
        if path.exists():
            value=path.read_text().strip()
            if value.isdigit():fields[key]=int(value)
    return fields


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--name',required=True);parser.add_argument('--tasks',type=int,nargs='+',required=True)
    parser.add_argument('--concurrency',type=int,default=6)
    args=parser.parse_args();configs=json.loads(args.campaign.read_text())['experiments']
    pending=list(args.tasks);active=[];records=[];blocked=[]
    directory=args.root/'kubeflow';directory.mkdir(exist_ok=True)
    record=directory/f'parallel_{args.name}.json'
    claims=args.root/'claims';claims.mkdir(exist_ok=True)
    failed=False
    observed_peak=0
    def status(name):
        manifest=args.root/'experiments'/name/'manifest.json'
        return json.loads(manifest.read_text()).get('status') if manifest.exists() else None
    while pending or active:
        for task in pending[:]:
            config=configs[task];name=config['name']
            if status(name)=='complete' or (claims/name).exists():
                pending.remove(task);continue
            dependencies=config.get('dependencies',[])
            if any(status(d) in {'failed','internal_health_failed','learning_review_required'} for d in dependencies):
                blocked.append({'experiment':name,'dependencies':dependencies,'reason':'upstream failed or requires learning review'})
                failed=True;pending.remove(task);continue
            if not all(status(d)=='complete' for d in dependencies):continue
            readbacks=[args.root/'experiments'/d/'ssl_readback.json' for d in config.get('required_ssl_readbacks',[])]
            if not all(path.exists() for path in readbacks):continue
            if any(not (r['integrity_and_internal_health_pass'] and r['measured_learning_pass'])
                for r in [json.loads(path.read_text()) for path in readbacks]):
                blocked.append({'experiment':name,'reason':'required SSL readback failed'})
                failed=True;pending.remove(task);continue
            if len(active)>=args.concurrency:break
            memory=host_memory()
            if active and memory.get('limit_bytes') and memory.get('current_bytes',0)>.75*memory['limit_bytes']:
                break
            try:(claims/name).mkdir()
            except FileExistsError:
                pending.remove(task);continue
            row={'task':task,'experiment':name,'started_at':datetime.now(timezone.utc).isoformat(),
                'log':str(directory/f'{args.name}_task{task}.log')}
            write_json(claims/name/'state.json',{'job':args.name,'status':'running','utc':row['started_at'],'host':os.environ.get('HOSTNAME')})
            stream=Path(row['log']).open('w')
            process=subprocess.Popen([sys.executable,'-m','analysis.flavour.s4.worker','--root',str(args.root),
                '--campaign',str(args.campaign),'--task',str(task)],stdout=stream,stderr=subprocess.STDOUT)
            row['pid']=process.pid;records.append(row);active.append((process,stream,row));pending.remove(task)
        for process,stream,row in active[:]:
            code=process.poll()
            if code is None:continue
            stream.close();row.update(exit_code=code,finished_at=datetime.now(timezone.utc).isoformat())
            write_json(claims/row['experiment']/'state.json',{'job':args.name,'status':'complete' if code==0 else 'failed',
                'exit_code':code,'utc':row['finished_at']})
            active.remove((process,stream,row))
            if code:failed=True
        memory=host_memory();observed_peak=max(observed_peak,memory.get('peak_bytes',memory.get('current_bytes',0)))
        write_json(record,{'job':args.name,'pod':os.environ.get('HOSTNAME'),'concurrency_limit':args.concurrency,
            'host_memory':memory,'observed_host_peak_bytes':observed_peak,'new_task_memory_start_fraction':.75,
            'processes':records,'pending_tasks':pending,'blocked_tasks':blocked,'failed':failed,'utc':datetime.now(timezone.utc).isoformat()})
        if pending or active:time.sleep(3)
    if failed:raise SystemExit('S4 task failed; dependent starts stopped, see recorded logs')


if __name__=='__main__':main()
