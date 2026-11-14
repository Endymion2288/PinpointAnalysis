"""Formal S4 batch submissions with live backend evidence and shared claims."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import io
import tarfile

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
        # NoEos schedds accept a small AFS submit directory and /dev/null for
        # scheduler-managed streams. The worker redirects its real logs to EOS.
        # Exercise their published submission policy; do not disable it.
        routed=(policy.returncode==0 and 'regexp("^/eos", Iwd)' in policy.stdout)
        if policy.returncode or (policy.stdout.strip().lower() != 'true' and not routed):
            checks.append({'schedd': name, 'EOS_policy': policy.stdout.strip(), 'compatible': False})
            continue
        ping = subprocess.run(['condor_ping', '-type', 'SCHEDD', '-name', name, '-table', 'WRITE'],
            capture_output=True, text=True)
        checks.append({'schedd': name, 'EOS_policy': policy.stdout.strip(), 'WRITE': ping.stdout,
            'layout':'AFS_small_wrapper_and_no_scheduler_streams;worker_logs_direct_EOS' if routed else 'direct_EOS_supported'})
        if not ping.returncode and 'ALLOW' in ping.stdout:
            return name, snapshot, checks
    raise RuntimeError('no compatible WRITE-authorized schedd')


def submit(args):
    root = args.root.resolve()
    repo = Path(__file__).resolve().parents[4]
    for name in ['logs', 'kubeflow', 'runtime']:
        (root / name).mkdir(exist_ok=True)
    name = args.name
    prototype_pool=args.gpu_pool=='fp32-prototype'
    compact_pool=args.gpu_pool=='native-bf16-compact'
    small_pool=args.gpu_pool=='native-bf16-small'
    diagnostic_pool=args.gpu_pool=='cuda-fp32-diagnostic'
    if diagnostic_pool:
        if args.backend!='condor' or args.stage!='campaign' or args.concurrency!=1:
            raise ValueError('FP32 diagnostic pool requires a single Condor campaign worker')
        selected=json.loads(args.campaign.read_text())['experiments']
        if not args.tasks or any(selected[task].get('program')!='functional_probe'
            or selected[task].get('device')!='cuda' for task in args.tasks):
            raise ValueError('FP32 diagnostic pool accepts only CUDA functional probes, never BF16 fits')
    if small_pool and (args.backend!='condor' or args.concurrency!=1):
        raise ValueError('12GB native BF16 pool requires one fit per allocation')
    if compact_pool and (args.backend!='condor' or args.concurrency>2):
        raise ValueError('compact native BF16 pool supports at most two concurrent fits')
    if prototype_pool:
        if args.backend!='condor' or args.stage not in {'pilot','campaign'}:
            raise ValueError('FP32 pool is limited to explicit Condor optimization prototypes')
        selected=json.loads(args.campaign.read_text())['experiments']
        for task in args.tasks:
            config=selected[task]
            if (not config.get('fp32_prototype') or config.get('program')!='pilot'
                or config.get('epochs',1)>1 or config.get('steps_per_epoch',0)>1024):
                raise ValueError('FP32 tasks must be labeled one-epoch pilots of at most 1024 steps')
    commit=command(['git','rev-parse','HEAD']).strip()
    if subprocess.run(['git','diff','--quiet','HEAD','--','src/analysis/flavour'],check=False).returncode:
        raise ValueError('commit source changes before freezing a batch snapshot')
    code=root/'code'/commit
    source_archive=root/'code'/f'{commit}.source.tar'
    archive=None
    if not (code/'.ready').exists():
        archive=subprocess.check_output(['git','archive','--format=tar','HEAD','--',
            'src/analysis/__init__.py','src/analysis/flavour'])
        code.mkdir(parents=True,exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            stream.extractall(code,filter='data')
        (code/'.ready').write_text(commit+'\n')
    if not source_archive.exists():
        if archive is None:
            archive=subprocess.check_output(['git','archive','--format=tar','HEAD','--',
                'src/analysis/__init__.py','src/analysis/flavour'])
        from analysis.flavour.s2.train import atomic_output
        atomic_output(source_archive,lambda stream:stream.write(archive))
    if args.campaign:
        frozen=root/'code'/f'{name}.campaign.json'
        payload=args.campaign.read_text()
        if args.stage=='validation_job':
            specification=json.loads(payload)
            inputs=root/'code'/f'{name}.inputs';inputs.mkdir(exist_ok=True)
            for entry in specification['experiments']:
                paths=[]
                for i,path in enumerate(entry['campaigns']):
                    target=inputs/f'campaign_{i}.json';target.write_text(Path(path).read_text());paths.append(str(target))
                comparison=inputs/'comparisons.json';comparison.write_text(Path(entry['comparisons']).read_text())
                entry.update(campaigns=paths,comparisons=str(comparison))
            payload=json.dumps(specification,indent=2)+'\n'
        elif args.stage=='probe_fit':
            specification=json.loads(payload)
            inputs=root/'code'/f'{name}.inputs';inputs.mkdir(exist_ok=True)
            for i,entry in enumerate(specification['experiments']):
                target=inputs/f'readouts_{i}.json'
                target.write_text(Path(entry['readout_campaign']).read_text())
                entry['readout_campaign']=str(target)
            payload=json.dumps(specification,indent=2)+'\n'
        frozen.write_text(payload)
        args.campaign=frozen
    cpu_only = args.stage in {'data','assemble','query_baseline','compute_profile','budget_replay','matched_plan','validation_job','parameter_audit','solver_probe','probe_fit','rel_diagnostic','frontend_profile','functional_probe','staging_profile','cache_profile'}
    if args.stage == 'data':
        arguments = ['analysis.flavour.s4.prepare', 'create', '--root', str(root), '--task', '$(task)', '--cpus', '4']
    elif args.stage == 'assemble':
        arguments = ['analysis.flavour.s4.prepare','finalize','--root',str(root),'--cpus','4']
    elif args.stage in {'query_baseline','compute_profile','budget_replay','matched_plan','validation_job','parameter_audit','solver_probe','probe_fit','rel_diagnostic','frontend_profile','functional_probe','staging_profile','cache_profile'}:
        if len(args.tasks)!=1:raise ValueError('one CPU diagnostic task per allocation')
        arguments=['analysis.flavour.s4.'+args.stage,'--root',str(root),'--campaign',str(args.campaign.resolve()),'--task',str(args.tasks[0])]
    else:
        arguments = ['analysis.flavour.s3.parallel', '--root', str(root), '--campaign', str(args.campaign.resolve()),
            '--name', name, '--concurrency', str(args.concurrency), '--training-module', 'analysis.flavour.s4.'+args.stage,
            '--tasks', *map(str, args.tasks), '--claim-tasks']
        if args.stage=='campaign':
            arguments=['analysis.flavour.s4.campaign','--root',str(root),'--campaign',str(args.campaign.resolve()),
                '--name',name,'--concurrency',str(args.concurrency),'--tasks',*map(str,args.tasks)]
    script = ['#!/bin/bash', 'set -o pipefail', 'which() { command -v "$@"; }',
        'source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh', 'set -e',
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',
        f'PINPOINT_S4_LOCAL_CODE="${{_CONDOR_SCRATCH_DIR:-/tmp}}/pinpoint_s4_code/{commit}"',
        'mkdir -p "$PINPOINT_S4_LOCAL_CODE"',
        'tar -xf '+shlex.quote(str(source_archive))+' -C "$PINPOINT_S4_LOCAL_CODE"',
        'export PYTHONPATH="$PINPOINT_S4_LOCAL_CODE/src:${PYTHONPATH:-}"',
        f'export PINPOINT_CODE_COMMIT={shlex.quote(commit)}',f'cd {shlex.quote(str(repo))}']
    if not cpu_only:
        script.append('source '+shlex.quote(str(repo/'src/analysis/flavour/s3/runtime_repair.sh'))+' '+shlex.quote(str(root)))
    stage_native_inputs=not cpu_only and not prototype_pool
    if stage_native_inputs:
        script.extend(['export PINPOINT_CLASSIFICATION_CACHE_MIB=1024',
            'export PINPOINT_NATIVE_SHARD_ROOT="${_CONDOR_SCRATCH_DIR:-/tmp}/pinpoint_s4_native_inputs"',
            'python -m analysis.flavour.s4.stage_inputs --root '+shlex.quote(str(root))+
            ' --output "$PINPOINT_NATIVE_SHARD_ROOT" --record '+shlex.quote(str(root/'runtime'/f'{name}.native_staging.json'))+
            ' --workers 3'])
    evidence = {'utc': datetime.now(timezone.utc).isoformat(), 'stage': args.stage, 'name': name,
        'backend': args.backend, 'tasks': args.tasks, 'concurrency': args.concurrency,'gpu_pool':args.gpu_pool,
        'code_commit': commit,'source_snapshot':str(code),'source_archive':str(source_archive),
        'worker_source':'local scratch extracted from committed archive',
        'native_input_source':'one shared local compressed copy per allocation, bounded per-reader decoded LRU' if stage_native_inputs else 'direct EOS compressed shards with bounded per-reader LRU',
        'classification_event_cache': '1024 MiB array cap per reader for fixed classification subsets <=10k; release native decode cache when complete; no SSL cache' if stage_native_inputs else 'disabled',
        'frozen_campaign':str(args.campaign) if args.campaign else None,
        'code_diff':command(['git','diff','--','src/analysis/flavour/s4']), 'submitted': False}
    record = root/'logs'/f'submission_{name}.json'
    if record.exists() and json.loads(record.read_text()).get('submitted'):
        raise ValueError('submission name already used')
    if args.backend == 'condor':
        chosen, snapshot, checks = choose_schedd()
        directory = Path.home()/'condor/Pinpoint/flavour_s4'/name
        directory.mkdir(parents=True, exist_ok=True)
        wrapper = directory/'run.sh'
        # All substantive stdout/stderr stays on EOS, while scheduler policy
        # sees only AFS submit metadata and empty /dev/null streams.
        redirect='exec > '+shlex.quote(str(root/'logs'/f'{name}.worker.'))+'"${PINPOINT_JOB_ID:-$$}".log 2>&1'
        wrapper.write_text('\n'.join([script[0],redirect,*script[1:],'exec python -m "$@"'])+'\n')
        wrapper.chmod(0o755)
        arguments_text = ' '.join(a if a == '$(task)' else shlex.quote(a) for a in arguments)
        gpu = not cpu_only
        gpu_cpus=max(4,args.concurrency*(2 if prototype_pool else 3))
        cpu_cpus=1 if args.stage in {'validation_job','parameter_audit'} else 4
        waiting_validation=args.stage in {'validation_job','parameter_audit','probe_fit'} and any(
            c.get('wait_for_completion_seconds',0)>1800 for c in json.loads(args.campaign.read_text())['experiments'])
        text = f'''universe = vanilla
executable = {wrapper}
arguments = {arguments_text}
initialdir = {directory}
environment = "PINPOINT_JOB_ID=$(ClusterId).$(Process)"
output = /dev/null
error = /dev/null
request_cpus = {gpu_cpus if gpu else cpu_cpus}
request_memory = {max(12000,args.concurrency*8000) if gpu else (3000 if args.stage=='validation_job' else 12000)}
request_disk = {max(12,args.concurrency*2) if gpu else 4}GB
requirements = (OpSysAndVer == "AlmaLinux9")
+JobFlavour = "{'workday' if gpu or waiting_validation or args.stage in {'budget_replay','matched_plan','functional_probe'} else 'microcentury'}"
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_output_files = ""
'''
        if gpu:
            if prototype_pool:
                # This LCG108 torch build has sm75+ kernels; V100/sm70 fails
                # even in FP32 before the first optimizer step.
                requirement='(DeviceName == "Tesla T4") && (GlobalMemoryMb >= 15000)'
            elif diagnostic_pool:
                # These probes disable autocast and TF32. T4/sm75 is supported
                # by LCG108; V100/sm70 is not in this torch build. Keep BF16
                # fitting out of this lane through the explicit guard above.
                products=['Tesla T4','NVIDIA A100-PCIE-40GB','NVIDIA H100 NVL','NVIDIA H200',
                    'NVIDIA H100L-1-12C MIG 1g.12gb','NVIDIA H100L-2-24C MIG 2g.24gb']
                requirement='('+' || '.join(f'(DeviceName == "{product}")' for product in products)+') && (GlobalMemoryMb >= 10000)'
            elif small_pool:
                requirement='(DeviceName == "NVIDIA H100L-1-12C MIG 1g.12gb") && (GlobalMemoryMb >= 10000)'
            elif compact_pool:
                requirement='((DeviceName == "NVIDIA H100 NVL") || (DeviceName == "NVIDIA H200") || (DeviceName == "NVIDIA A100-PCIE-40GB") || (DeviceName == "NVIDIA H100L-2-24C MIG 2g.24gb")) && (GlobalMemoryMb >= 20000)'
            else:
                requirement='((DeviceName == "NVIDIA H100 NVL") || (DeviceName == "NVIDIA H200") || (DeviceName == "NVIDIA A100-PCIE-40GB")) && (GlobalMemoryMb >= 39000)'
            text += 'request_gpus = 1\nrequire_gpus = '+requirement+'\nqueue 1\n'
        elif args.stage == 'data':
            text += 'queue task in '+','.join(map(str,args.tasks))+'\n'
        else:
            text += 'queue 1\n'
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
        if cpu_only:
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
        try:
            admitted = json.loads(kubectl('create','--dry-run=server','-f',str(pod_path),'-o','json'))
        except subprocess.CalledProcessError as error:
            evidence.update(flavor=args.flavor,job_spec=job,admission_rejected=True,
                admission_response=error.output,allocated_device_hours=0.)
            write_json(record,evidence)
            raise
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
    parser.add_argument('--stage',choices=['data','assemble','query_baseline','compute_profile','budget_replay','matched_plan','validation_job','parameter_audit','solver_probe','probe_fit','rel_diagnostic','frontend_profile','functional_probe','staging_profile','cache_profile','pilot','pretrain','train','probes','worker','campaign'],required=True)
    parser.add_argument('--name',required=True)
    parser.add_argument('--campaign',type=Path)
    parser.add_argument('--tasks',type=int,nargs='+',required=True)
    parser.add_argument('--concurrency',type=int,default=6)
    parser.add_argument('--flavor',choices=GPU_FLAVORS,default='h100-47gb')
    parser.add_argument('--gpu-pool',choices=['native-bf16','native-bf16-compact','native-bf16-small','fp32-prototype','cuda-fp32-diagnostic'],default='native-bf16')
    parser.add_argument('--deadline',type=int,default=21600)
    parser.add_argument('--apply',action='store_true')
    submit(parser.parse_args())


if __name__=='__main__':
    main()
