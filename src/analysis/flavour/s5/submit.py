"""Small AFS submissions, immutable source snapshots, EOS outputs and live queues."""
import argparse
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile

from analysis.flavour.s2.kubeflow import GPU_FLAVORS, kubectl
from analysis.flavour.s4.submit import choose_schedd


def submit(args):
    root = args.root.resolve()
    repo = Path(__file__).resolve().parents[4]
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    if subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', 'src/analysis/flavour/s5'], cwd=repo).returncode:
        raise ValueError('commit S5 source before freezing a submission')
    archive = root/'code'/f'{commit}.tar'
    if not archive.exists():
        archive.write_bytes(subprocess.check_output(['git', 'archive', '--format=tar', 'HEAD', '--', 'src/analysis/__init__.py', 'src/analysis/flavour'], cwd=repo))
    command = ['analysis.flavour.s5.'+args.module, '--root', str(root)]
    command += args.command
    if args.campaign:
        frozen = root/'code'/f'{args.name}.campaign.json'
        frozen.write_text(args.campaign.read_text())
        command += ['--campaign', str(frozen)]
    script = ['#!/bin/bash', 'set -o pipefail', 'which() { command -v "$@"; }',
              'source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh', 'set -e',
              'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1',
              'export PINPOINT_S5_SCRATCH="${_CONDOR_SCRATCH_DIR:-/tmp}/pinpoint_s5_${PINPOINT_JOB_ID:-$$}"',
              'mkdir -p "$PINPOINT_S5_SCRATCH/code"',
              'tar -xf '+shlex.quote(str(archive))+' -C "$PINPOINT_S5_SCRATCH/code"',
              'export PYTHONPATH="$PINPOINT_S5_SCRATCH/code/src:${PYTHONPATH:-}"',
              'cd "$PINPOINT_S5_SCRATCH"']
    if args.gpu:
        script += ['source "$PINPOINT_S5_SCRATCH/code/src/analysis/flavour/s3/runtime_repair.sh" '+shlex.quote(str(root))]
    record = root/'runtime'/f'{args.name}.submission.json'
    if record.exists():
        raise ValueError('submission name already recorded')
    evidence = {'utc': datetime.now(timezone.utc).isoformat(), 'backend': args.backend, 'name': args.name,
                'commit': commit, 'archive': str(archive), 'tasks': args.tasks, 'gpu': args.gpu,
                'submitted': False, 'command': command}
    if args.backend == 'condor':
        chosen, snapshot, checks = choose_schedd()
        directory = Path.home()/'condor/Pinpoint/flavour_s5'/args.name
        directory.mkdir(parents=True, exist_ok=True)
        redirect = 'exec > '+shlex.quote(str(root/'logs'/f'{args.name}.'))+'"${PINPOINT_JOB_ID}".log 2>&1'
        wrapper = directory/'run.sh'
        wrapper.write_text('\n'.join([script[0], redirect, *script[1:], 'exec python -m "$@"'])+'\n')
        wrapper.chmod(0o755)
        text = '\n'.join(['universe = vanilla', f'executable = {wrapper}',
            'arguments = '+shlex.join(command+(['--task', '$(task)'] if args.tasks else [])),
            f'initialdir = {directory}', 'environment = "PINPOINT_JOB_ID=$(ClusterId).$(Process)"',
            'output = /dev/null', 'error = /dev/null', f'request_cpus = {args.cpus}',
            f'request_memory = {args.memory}', 'request_disk = 12GB',
            'requirements = (OpSysAndVer == "AlmaLinux9")', '+JobFlavour = "workday"',
            'should_transfer_files = YES', 'when_to_transfer_output = ON_EXIT', 'transfer_output_files = ""',
            f'max_materialize = {args.max_materialize}'])+'\n'
        if args.gpu:
            text += 'request_gpus = 1\nrequire_gpus = (GlobalMemoryMb >= 20000) && ((DeviceName == "NVIDIA H100 NVL") || (DeviceName == "NVIDIA H200") || (DeviceName == "NVIDIA A100-PCIE-40GB") || (DeviceName == "NVIDIA H100L-2-24C MIG 2g.24gb"))\n'
        text += 'queue task in '+','.join(map(str, args.tasks))+'\n' if args.tasks else 'queue 1\n'
        path = directory/'job.sub'
        path.write_text(text)
        evidence.update(schedd=chosen, queue_snapshot=snapshot, compatibility=checks,
                        selection='minimum idle/held/running among compatible WRITE-authorized CERN schedds',
                        wrapper=wrapper.read_text(), submit=text, submit_file=str(path))
        if args.apply:
            result = subprocess.run(['condor_submit', '-name', chosen, str(path)], capture_output=True, text=True)
            evidence.update(stdout=result.stdout, stderr=result.stderr, submitted=result.returncode == 0)
    else:
        if not args.gpu or args.tasks:
            raise ValueError('Kubeflow uses one explicit parallel-training command per GPU allocation')
        product, resource = GPU_FLAVORS[args.flavor]
        resources = {resource: 1}
        labels = {'app': 'pinpoint-s5', 'mount-eos': 'true', 'mount-kerberos-secret': 'true'}
        spec = {'restartPolicy': 'Never', 'serviceAccountName': 'default-editor',
                'securityContext': {'runAsUser': os.getuid(), 'runAsGroup': os.getgid()},
                'nodeSelector': {'nvidia.com/gpu.product': product},
                'volumes': [{'name': 'cvmfs', 'persistentVolumeClaim': {'claimName': 'cvmfs'}}],
                'containers': [{'name': 's5', 'image': 'docker.io/library/almalinux:9',
                    'command': ['/bin/bash', '-c', '\n'.join(script+['exec python -m '+shlex.join(command)])],
                    'resources': {'requests': resources, 'limits': resources},
                    'volumeMounts': [{'name': 'cvmfs', 'mountPath': '/cvmfs', 'readOnly': True, 'mountPropagation': 'HostToContainer'}]}]}
        job = {'apiVersion': 'batch/v1', 'kind': 'Job', 'metadata': {'name': args.name, 'namespace': 'xcheng', 'labels': {'app': 'pinpoint-s5'}},
               'spec': {'backoffLimit': 0, 'activeDeadlineSeconds': 28800, 'template': {'metadata': {'labels': labels}, 'spec': spec}}}
        pod = {'apiVersion': 'v1', 'kind': 'Pod', 'metadata': {'name': args.name+'-admission', 'namespace': 'xcheng', 'labels': labels}, 'spec': spec}
        path = root/'kubeflow'/f'{args.name}.json'
        admission = root/'kubeflow'/f'{args.name}.admission.json'
        path.write_text(json.dumps(job, indent=2)+'\n')
        admission.write_text(json.dumps(pod, indent=2)+'\n')
        evidence.update(flavor=args.flavor, job_spec=job)
        try:
            evidence['admitted_pod'] = json.loads(kubectl('create', '--dry-run=server', '-f', str(admission), '-o', 'json'))
            if args.apply:
                response = json.loads(kubectl('create', '-f', str(path), '-o', 'json'))
                evidence.update(submitted=True, uid=response['metadata']['uid'])
        except subprocess.CalledProcessError as error:
            evidence.update(error=str(error), output=error.output)
    record.write_text(json.dumps(evidence, indent=2)+'\n')
    print(json.dumps({'record': str(record), 'submitted': evidence['submitted'], 'response': evidence.get('stdout')}))
    if args.apply and not evidence['submitted']:
        raise RuntimeError('submission failed; evidence retained')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--backend', choices=['condor', 'kubeflow'], default='condor')
    parser.add_argument('--module', required=True)
    parser.add_argument('--campaign', type=Path)
    parser.add_argument('--tasks', nargs='+', type=int)
    parser.add_argument('--cpus', type=int, default=4)
    parser.add_argument('--memory', type=int, default=12000)
    parser.add_argument('--max-materialize', type=int, default=24)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--flavor', choices=GPU_FLAVORS, default='h100-47gb')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ['--']:
        args.command = args.command[1:]
    submit(args)


if __name__ == '__main__':
    main()
