"""Formal S4 batch submissions with live backend evidence and shared claims."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess

from analysis.flavour.s2.kubeflow import GPU_FLAVORS, kubectl
from analysis.flavour.inventory import write_json


def command(args):
    return subprocess.check_output(args, text=True)


def choose_schedd():
    snapshot = command(['condor_status', '-schedd', '-af', 'Name', 'TotalRunningJobs', 'TotalIdleJobs', 'TotalHeldJobs'])
    candidates = []
    for line in snapshot.splitlines():
        name, running, idle, held = line.split()
        if name.startswith('bigbird'):
            candidates.append((int(idle), int(held), int(running), name))
    checks = []
    for *_, name in sorted(candidates):
        policy = subprocess.run(['condor_config_val', '-name', name, '-schedd', 'SUBMIT_REQUIREMENT_NoEos'],
            capture_output=True, text=True)
        if policy.returncode or policy.stdout.strip().lower() != 'true':
            checks.append({'schedd': name, 'EOS_policy': policy.stdout.strip(), 'compatible': False})
            continue
        ping = subprocess.run(['condor_ping', '-type', 'SCHEDD', '-name', name, '-table', 'WRITE'],
            capture_output=True, text=True)
        checks.append({'schedd': name, 'EOS_policy': policy.stdout.strip(), 'WRITE': ping.stdout})
        if not ping.returncode and 'ALLOW' in ping.stdout:
            return name, snapshot, checks
    raise RuntimeError('no compatible WRITE-authorized schedd')


def submit(args):
    root = args.root.resolve()
    repo = Path(__file__).resolve().parents[4]
    for name in ['logs', 'kubeflow', 'runtime']:
        (root / name).mkdir(exist_ok=True)
    name = args.name
    if args.stage == 'data':
        arguments = ['analysis.flavour.s4.prepare', 'create', '--root', str(root), '--task', '$(task)', '--cpus', '4']
    else:
        arguments = ['analysis.flavour.s3.parallel', '--root', str(root), '--campaign', str(args.campaign.resolve()),
            '--name', name, '--concurrency', str(args.concurrency), '--training-module', 'analysis.flavour.s4.'+args.stage,
            '--tasks', *map(str, args.tasks), '--claim-tasks']
    script = ['#!/bin/bash', 'set -o pipefail', 'which() { command -v "$@"; }',
        'source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh', 'set -e',
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',
        f'export PYTHONPATH={shlex.quote(str(repo/"src"))}:${{PYTHONPATH:-}}', f'cd {shlex.quote(str(repo))}']
    if args.stage != 'data':
        script.append('source '+shlex.quote(str(repo/'src/analysis/flavour/s3/runtime_repair.sh'))+' '+shlex.quote(str(root)))
    evidence = {'utc': datetime.now(timezone.utc).isoformat(), 'stage': args.stage, 'name': name,
        'backend': args.backend, 'tasks': args.tasks, 'concurrency': args.concurrency,
        'code_commit': command(['git','rev-parse','HEAD']).strip(), 'submitted': False}
    record = root/'logs'/f'submission_{name}.json'
    if record.exists() and json.loads(record.read_text()).get('submitted'):
        raise ValueError('submission name already used')
    if args.backend == 'condor':
        chosen, snapshot, checks = choose_schedd()
        directory = Path.home()/'condor/Pinpoint/flavour_s4'/name
        directory.mkdir(parents=True, exist_ok=True)
        wrapper = directory/'run.sh'
        wrapper.write_text('\n'.join(script+['exec python -m "$@"'])+'\n')
        wrapper.chmod(0o755)
        arguments_text = ' '.join(a if a == '$(task)' else shlex.quote(a) for a in arguments)
        gpu = args.stage != 'data'
        text = f'''universe = vanilla
executable = {wrapper}
arguments = {arguments_text}
output = {root}/logs/{name}.$(ClusterId).$(Process).out
error = {root}/logs/{name}.$(ClusterId).$(Process).err
log = {root}/logs/{name}.$(ClusterId).log
request_cpus = {max(4,args.concurrency*3) if gpu else 4}
request_memory = {max(12,args.concurrency*5) if gpu else 12}GB
request_disk = {max(12,args.concurrency*2) if gpu else 4}GB
requirements = (OpSysAndVer == "AlmaLinux9")
+JobFlavour = "{'workday' if gpu else 'microcentury'}"
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_output_files = ""
'''
        if gpu:
            text += 'request_gpus = 1\nrequire_gpus = ((DeviceName == "NVIDIA H100 NVL") || (DeviceName == "NVIDIA A100-PCIE-40GB")) && (GlobalMemoryMb >= 39000)\nqueue 1\n'
        else:
            text += 'queue task in '+','.join(map(str,args.tasks))+'\n'
        path = directory/'job.sub'
        path.write_text(text)
        evidence.update(schedd=chosen, snapshot=snapshot, checks=checks, submit=text, wrapper=wrapper.read_text(),
            selection_basis='minimum idle/held/running among EOS-compatible WRITE-authorized schedds')
        if args.apply:
            result = subprocess.run(['condor_submit','-name',chosen,str(path)],capture_output=True,text=True)
            evidence.update(response=result.stdout, stderr=result.stderr, submitted=result.returncode==0)
            write_json(record,evidence)
            if result.returncode:
                raise RuntimeError(result.stderr)
    else:
        if args.stage == 'data':
            raise ValueError('bulk data conversion uses HTCondor')
        product, resource = GPU_FLAVORS[args.flavor]
        requests = {resource: 1}
        script.append('exec python -m '+shlex.join(arguments))
        spec = {'restartPolicy':'Never','serviceAccountName':'default-editor',
            'securityContext':{'runAsUser':os.getuid(),'runAsGroup':os.getgid()},
            'nodeSelector':{'nvidia.com/gpu.product':product},
            'volumes':[{'name':'cvmfs','persistentVolumeClaim':{'claimName':'cvmfs'}}],
            'containers':[{'name':'s4','image':'docker.io/library/almalinux:9',
                'command':['/bin/bash','-c','\n'.join(script)],
                'resources':{'requests':requests,'limits':requests},
                'volumeMounts':[{'name':'cvmfs','mountPath':'/cvmfs','readOnly':True,'mountPropagation':'HostToContainer'}]}]}
        labels = {'app':'pinpoint-s4','mount-eos':'true','mount-kerberos-secret':'true'}
        job = {'apiVersion':'batch/v1','kind':'Job','metadata':{'name':name,'namespace':'xcheng','labels':{'app':'pinpoint-s4'}},
            'spec':{'backoffLimit':0,'activeDeadlineSeconds':args.deadline,
                'template':{'metadata':{'labels':labels},'spec':spec}}}
        pod = {'apiVersion':'v1','kind':'Pod','metadata':{'name':name+'-admission','namespace':'xcheng','labels':labels},'spec':spec}
        path = root/'kubeflow'/f'{name}.job.json'
        pod_path = root/'kubeflow'/f'{name}.admission.json'
        write_json(path,job);write_json(pod_path,pod)
        admitted = json.loads(kubectl('create','--dry-run=server','-f',str(pod_path),'-o','json'))
        evidence.update(job_spec=job,pod_admission=admitted,flavor=args.flavor)
        if args.apply:
            result = json.loads(kubectl('create','-f',str(path),'-o','json'))
            evidence.update(submitted=True,uid=result['metadata']['uid'])
    write_json(record,evidence)
    print(json.dumps({'record':str(record),'submitted':evidence['submitted'],'response':evidence.get('response')}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--backend',choices=['condor','kubeflow'],required=True)
    parser.add_argument('--stage',choices=['data','pilot','pretrain','train','probes'],required=True)
    parser.add_argument('--name',required=True)
    parser.add_argument('--campaign',type=Path)
    parser.add_argument('--tasks',type=int,nargs='+',required=True)
    parser.add_argument('--concurrency',type=int,default=6)
    parser.add_argument('--flavor',choices=GPU_FLAVORS,default='h100-47gb')
    parser.add_argument('--deadline',type=int,default=21600)
    parser.add_argument('--apply',action='store_true')
    submit(parser.parse_args())


if __name__=='__main__':
    main()
