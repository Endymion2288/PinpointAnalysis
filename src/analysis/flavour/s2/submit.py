"""S2 CPU/GPU Condor submissions with current schedd and resource evidence."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess

from analysis.flavour.inventory import write_json
from analysis.flavour.submit import command


def submit(args):
    snapshot=command(['condor_status','-schedd','-af','Name','TotalRunningJobs','TotalIdleJobs','TotalHeldJobs'])
    candidates=[]
    for line in snapshot.splitlines():
        name,running,idle,held=line.split()
        if name.startswith('bigbird'):
            candidates.append((int(idle),int(held),int(running),name))
    attempts=[]
    for *_,name in sorted(candidates):
        policy=subprocess.run(['condor_config_val','-name',name,'-schedd','SUBMIT_REQUIREMENT_NoEos'],
                              capture_output=True,text=True)
        if policy.returncode or policy.stdout.strip().lower()!='true':
            attempts.append({'name':name,'EOS_policy':policy.stdout.strip(),'compatible':False})
            continue
        ping=subprocess.run(['condor_ping','-type','SCHEDD','-name',name,'-table','WRITE'],capture_output=True,text=True)
        attempts.append({'name':name,'EOS_policy':policy.stdout.strip(),'authorization':ping.stdout})
        if ping.returncode==0 and 'ALLOW' in ping.stdout:
            chosen=name
            break
    else:
        raise RuntimeError('no compatible WRITE-authorized schedd')
    repo=Path(__file__).resolve().parents[4]
    submit_dir=Path.home()/'condor/Pinpoint/flavour_s2'
    submit_dir.mkdir(parents=True,exist_ok=True)
    (args.root/'logs').mkdir(parents=True,exist_ok=True)
    wrapper=submit_dir/'run.sh'
    wrapper.write_text(f'''#!/bin/bash
set -eo pipefail
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH={shlex.quote(str(repo/'src'))}:${{PYTHONPATH}}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd {shlex.quote(str(repo))}
exec python -m "$@"
''')
    wrapper.chmod(0o755)
    cpus,memory,disk,queue,gpu=2,'6GB','2GB','queue 1',''
    baseline=Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s0_s1_2026-11-11')
    if args.stage=='plan':
        arguments=f'analysis.flavour.s2.prepare plan --inventory {baseline}/inventory/events.parquet --geometry {baseline}/shards/10083/geometry.json --output {args.root}'
    elif args.stage=='convert':
        tasks=json.loads((args.root/'plan.json').read_text())['tasks']
        arguments=f'analysis.flavour.s2.prepare convert --root {args.root} --task $(Process) --cpus {cpus}'
        queue=f'queue {len(tasks)}'
    elif args.stage=='train':
        if args.campaign is None:
            raise ValueError('--campaign JSON required')
        configs=json.loads(args.campaign.read_text())['experiments']
        arguments=f'analysis.flavour.s2.train --root {args.root} --campaign {args.campaign} --task $(Process)'
        queue=f'queue {len(configs)}'
        cpus,memory,disk=4,'12GB','12GB'
        gpu='request_gpus = 1\nrequire_gpus = (GlobalMemoryMb >= 14000)\n'
    else:
        raise ValueError(args.stage)
    text=f'''universe = vanilla
executable = {wrapper}
arguments = {arguments}
output = {args.root}/logs/{args.stage}.$(ClusterId).$(Process).out
error = {args.root}/logs/{args.stage}.$(ClusterId).$(Process).err
log = {args.root}/logs/{args.stage}.$(ClusterId).log
request_cpus = {cpus}
request_memory = {memory}
request_disk = {disk}
{gpu}+JobFlavour = "{'workday' if args.stage=='train' else 'microcentury'}"
requirements = (OpSysAndVer == "AlmaLinux9")
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_output_files = ""
{queue}
'''
    filename=submit_dir/f'{args.stage}.sub'
    filename.write_text(text)
    evidence={'utc':datetime.now(timezone.utc).isoformat(),'stage':args.stage,'chosen':chosen,
        'schedd_snapshot':snapshot,'checks':attempts,'submit':text,'wrapper':wrapper.read_text(),
        'code_commit':command(['git','rev-parse','HEAD']).strip(),
        'selection_basis':'compatible direct EOS policy and WRITE; minimum idle/held/running',
        'GPU_requirement':'one CUDA GPU with >=14000 MiB' if gpu else None}
    if gpu:
        evidence['gpu_pool_snapshot']=command(['condor_status','-constraint','GPUs > 0','-af',
                                                'Name','State','Activity','Cpus','Memory','GPUs_GlobalMemoryMb','GPUs_DeviceName'])
    result=subprocess.run(['condor_submit','-name',chosen,str(filename)],text=True,capture_output=True)
    evidence.update(returncode=result.returncode,response=result.stdout,stderr=result.stderr)
    record=args.root/f'submission_{args.stage}_{datetime.now(timezone.utc).strftime("%H%M%S")}.json'
    write_json(record,evidence)
    if result.returncode:
        raise RuntimeError(result.stderr)
    print(result.stdout,str(record))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['plan','convert','train'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--campaign',type=Path)
    submit(parser.parse_args())


if __name__=='__main__':
    main()
