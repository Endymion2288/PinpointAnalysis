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
    platforms={'LCG_108_cuda':'x86_64-el9-gcc13-opt','LCG_107_cuda':'x86_64-el9-gcc11-opt'}
    view=f'/cvmfs/sft.cern.ch/lcg/views/{args.lcg_view}/{platforms[args.lcg_view]}'
    wrapper=submit_dir/f'run_{args.lcg_view}.sh'
    wrapper.write_text(f'''#!/bin/bash
set -eo pipefail
source {view}/setup.sh
export PYTHONPATH={shlex.quote(str(repo/'src'))}:${{PYTHONPATH}}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd {shlex.quote(str(repo))}
exec python -m "$@"
''')
    wrapper.chmod(0o755)
    cpus,memory,disk,queue,gpu=2,'6GB','2GB','queue 1',''
    submitted_experiments=[]
    capability=7.0 if args.lcg_view=='LCG_107_cuda' else 7.5
    GPU_requirement=f'(GlobalMemoryMb >= 14000) && (Capability >= {capability})'
    if args.gpu_device:
        GPU_requirement+=' && (DeviceName == '+json.dumps(args.gpu_device)+')'
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
        arguments=f'analysis.flavour.s2.train --root {args.root} --campaign {args.campaign} --task $(task)'
        tasks=args.tasks if args.tasks is not None else list(range(len(configs)))
        if len(set(tasks))!=len(tasks) or not all(0<=t<len(configs) for t in tasks):
            raise ValueError('training task indices must be unique members of the frozen campaign')
        queue='queue task in '+','.join(map(str,tasks))
        submitted_experiments=[{'task':t,'config':configs[t]} for t in tasks]
        cpus,memory,disk=4,'12GB','12GB'
        gpu=f'request_gpus = 1\nrequire_gpus = {GPU_requirement}\n'
    elif args.stage=='audit':
        arguments=f'analysis.flavour.s2.audit --root {args.root}'
    elif args.stage=='toys':
        arguments=f'analysis.flavour.s2.toys --geometry {baseline}/shards/10083/geometry.json --output {args.root}/toys'
    elif args.stage=='refresh_tau':
        arguments=f'analysis.flavour.s2.prepare refresh_tau --root {args.root}'
    elif args.stage=='profile':
        arguments=f'analysis.flavour.s2.profile --root {args.root} --events {args.profile_events} --output {args.root/args.profile_tag}'
        cpus,memory,disk=2,'8GB','4GB'
        gpu=f'request_gpus = 1\nrequire_gpus = {GPU_requirement}\n'
    elif args.stage=='reference':
        arguments=f'analysis.flavour.s2.reference --root {args.root} --mode $(mode)'
        queue='queue mode in b0_all b0_signal b0_shape truth_energy_vertex'
    elif args.stage=='evaluate':
        arguments=f'analysis.flavour.s2.evaluate --root {args.root} --output {args.root}/{args.evaluation_tag}'
        if args.comparisons:
            arguments+=f' --comparisons {args.comparisons}'
        if args.allow_incomplete:
            arguments+=' --allow-incomplete'
    elif args.stage=='display':
        arguments=f'analysis.flavour.s2.display --root {args.root} --experiment {args.experiment} --evaluation-tag {args.evaluation_tag}'
    elif args.stage=='accounting':
        arguments=f'analysis.flavour.s2.accounting --root {args.root}'
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
{gpu}+JobFlavour = "{args.training_flavour if args.stage=='train' else 'workday' if gpu else 'microcentury'}"
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
        'software_view':view,'GPU_requirement':GPU_requirement if gpu else None,
        'submitted_experiments':submitted_experiments}
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
    parser.add_argument('stage',choices=['plan','convert','train','audit','toys','refresh_tau','profile','reference','evaluate','display','accounting'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--campaign',type=Path)
    parser.add_argument('--tasks',type=int,nargs='+',help='Selected indices from the frozen training campaign')
    parser.add_argument('--comparisons',type=Path)
    parser.add_argument('--allow-incomplete',action='store_true')
    parser.add_argument('--experiment',default='b0_all_s42')
    parser.add_argument('--evaluation-tag',default='evaluation')
    parser.add_argument('--profile-events',type=int,default=24,
                        help='Occupancy-rank events per representation in a resource measurement')
    parser.add_argument('--profile-tag',default='resource_profile_v2',
                        help='Separate output directory for resource measurements on another device')
    parser.add_argument('--lcg-view',choices=['LCG_108_cuda','LCG_107_cuda'],default='LCG_108_cuda')
    parser.add_argument('--gpu-device',help='Optional exact CUDA device name for matched-hardware campaigns')
    parser.add_argument('--training-flavour',choices=['espresso','microcentury','longlunch','workday','tomorrow'],default='tomorrow',
                        help='Training wall-time class; espresso supports small GPU execution checks, scientific runs use tomorrow')
    submit(parser.parse_args())


if __name__=='__main__':
    main()
