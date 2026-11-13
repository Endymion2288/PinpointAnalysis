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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--name',required=True);parser.add_argument('--tasks',type=int,nargs='+',required=True)
    parser.add_argument('--concurrency',type=int,default=6)
    args=parser.parse_args();configs=json.loads(args.campaign.read_text())['experiments']
    pending=list(args.tasks);active=[];records=[]
    directory=args.root/'kubeflow';directory.mkdir(exist_ok=True)
    record=directory/f'parallel_{args.name}.json'
    claims=args.root/'claims';claims.mkdir(exist_ok=True)
    failed=False
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
                raise RuntimeError('dependency failed: '+name)
            if not all(status(d)=='complete' for d in dependencies):continue
            if len(active)>=args.concurrency:break
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
            if code:failed=True;pending=[]
        write_json(record,{'job':args.name,'pod':os.environ.get('HOSTNAME'),'concurrency_limit':args.concurrency,
            'processes':records,'pending_tasks':pending,'failed':failed,'utc':datetime.now(timezone.utc).isoformat()})
        if pending or active:time.sleep(3)
    if failed:raise SystemExit('S4 task failed; dependent starts stopped, see recorded logs')


if __name__=='__main__':main()
