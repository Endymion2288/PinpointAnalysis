"""Submit S3 GPU work to a currently compatible, least queued schedd."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess


def command(args):
    return subprocess.check_output(args, text=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['train','train_repair','profile','smoke'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--tasks',type=int,nargs='+')
    parser.add_argument('--tag',required=True)
    parser.add_argument('--concurrency',type=int,default=1)
    args=parser.parse_args()
    snapshot=command(['condor_status','-schedd','-af','Name','TotalRunningJobs','TotalIdleJobs','TotalHeldJobs'])
    candidates=[]
    for line in snapshot.splitlines():
        name,running,idle,held=line.split()
        if name.startswith('bigbird'):candidates.append((int(idle),int(held),int(running),name))
    checks=[]
    for *_,name in sorted(candidates):
        policy=subprocess.run(['condor_config_val','-name',name,'-schedd','SUBMIT_REQUIREMENT_NoEos'],capture_output=True,text=True)
        if policy.returncode or policy.stdout.strip().lower()!='true':
            checks.append({'schedd':name,'compatible':False,'EOS_policy':policy.stdout.strip()});continue
        ping=subprocess.run(['condor_ping','-type','SCHEDD','-name',name,'-table','WRITE'],capture_output=True,text=True)
        checks.append({'schedd':name,'WRITE':ping.stdout,'EOS_policy':policy.stdout.strip()})
        if not ping.returncode and 'ALLOW' in ping.stdout:chosen=name;break
    else:raise RuntimeError('no compatible authorized schedd')
    repo=Path(__file__).resolve().parents[4]
    root=args.root.resolve();(root/'logs').mkdir(exist_ok=True)
    directory=Path.home()/'condor/Pinpoint/flavour_s3'/args.tag;directory.mkdir(parents=True,exist_ok=True)
    wrapper=directory/'run.sh'
    wrapper.write_text('#!/bin/bash\nset -o pipefail\nwhich() { command -v "$@"; }\nsource /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh\nset -e\n'+
                       f'export PYTHONPATH={shlex.quote(str(repo/"src"))}:${{PYTHONPATH:-}}\n'+
                       'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1\n'+f'cd {shlex.quote(str(repo))}\nexec python -m "$@"\n')
    wrapper.chmod(0o755)
    campaign=json.loads(args.campaign.read_text())
    configs=campaign['experiments'];tasks=args.tasks if args.tasks is not None else list(range(len(configs)))
    if any(configs[t].get('compile_local') for t in tasks):
        runtime='source '+shlex.quote(str(repo/'src/analysis/flavour/s3/runtime_repair.sh'))+' '+shlex.quote(str(root))+'\n'
        wrapper.write_text(wrapper.read_text().replace('exec python -m',runtime+'exec python -m'))
    assert len(set(tasks))==len(tasks) and all(0<=t<len(configs) for t in tasks)
    if args.concurrency<1:
        parser.error('concurrency must be positive')
    training=args.stage in ['train','train_repair']
    parallel_name='condor-'+args.tag if training and args.concurrency>1 else None
    arguments=['analysis.flavour.s3.'+args.stage,'--root',str(root),'--campaign',str(args.campaign.resolve())]
    queue='queue 1'
    if parallel_name:
        arguments=['analysis.flavour.s3.parallel','--root',str(root),'--campaign',str(args.campaign.resolve()),
                   '--name',parallel_name,'--training-module','analysis.flavour.s3.'+args.stage,
                   '--concurrency',str(args.concurrency),'--tasks',*map(str,tasks)]
    elif training:
        arguments+=['--task','$(task)'];queue='queue task in '+','.join(map(str,tasks))
    elif args.stage=='profile':arguments+=['--tasks',*map(str,tasks),'--events','3','--output',str(root/args.tag)]
    else:arguments+=['--tasks',*map(str,tasks),'--output',str(root/args.tag)]
    arguments=' '.join(shlex.quote(a) if a!='$(task)' else a for a in arguments)
    submit=f'''universe = vanilla
executable = {wrapper}
arguments = {arguments}
output = {root}/logs/{args.tag}.$(ClusterId).$(Process).out
error = {root}/logs/{args.tag}.$(ClusterId).$(Process).err
log = {root}/logs/{args.tag}.$(ClusterId).log
request_cpus = {max(4,args.concurrency*3) if parallel_name else 4}
request_memory = {max(12,args.concurrency*10) if parallel_name else 12}GB
request_disk = 12GB
request_gpus = 1
require_gpus = (DeviceName == "NVIDIA H100 NVL") && (GlobalMemoryMb >= 80000)
+JobFlavour = "{'workday' if training else 'microcentury'}"
requirements = (OpSysAndVer == "AlmaLinux9")
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_output_files = ""
{queue}
'''
    path=directory/'job.sub';path.write_text(submit)
    record=root/f'submission_{args.tag}_{datetime.now(timezone.utc).strftime("%H%M%S")}.json'
    evidence={'utc':datetime.now(timezone.utc).isoformat(),'stage':'train' if training else args.stage,'program':args.stage,'chosen':chosen,
              'schedd_snapshot':snapshot,'checks':checks,'submit':submit,'wrapper':wrapper.read_text(),
              'code_commit':command(['git','rev-parse','HEAD']).strip(),
              'selection_basis':'minimum idle/held/running among direct-EOS compatible WRITE-authorized schedds',
              'gpu_pool_snapshot':command(['condor_status','-constraint','GPUs > 0','-af','Name','State','Activity','GPUs_DeviceName']),
              'parallel_name':parallel_name,
              'parallel_experiments':[{'task':t,'config':configs[t]} for t in tasks] if parallel_name else [],
              'submitted_experiments':([{'config':{'name':parallel_name}}] if parallel_name else
                                       [{'task':t,'config':configs[t]} for t in tasks] if training else [])}
    result=subprocess.run(['condor_submit','-name',chosen,str(path)],capture_output=True,text=True)
    evidence.update(returncode=result.returncode,response=result.stdout,stderr=result.stderr)
    record.write_text(json.dumps(evidence,indent=2)+'\n')
    if result.returncode:raise RuntimeError(result.stderr)
    print(result.stdout,str(record))


if __name__=='__main__':main()
