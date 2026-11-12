"""CERN flavor-correct single-GPU S3 jobs and recorded backend snapshots."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess

from analysis.flavour.s2.kubeflow import GPU_FLAVORS, kubectl


def prepare(args):
    root = args.root.resolve()
    repo = Path(__file__).resolve().parents[4]
    output = root / 'kubeflow'
    output.mkdir(parents=True, exist_ok=True)
    record = output / f'execution_{args.name}.json'
    if record.exists() and json.loads(record.read_text()).get('submitted'):
        raise ValueError('Job name already submitted; preserve its allocation history')
    product, resource = GPU_FLAVORS[args.flavor]
    args_list = ['analysis.flavour.s3.probe', '--root', str(root)]
    experiment = 'environment-probe'
    if args.dispatch:
        if args.task is not None or args.flavor not in {'h100-94gb', 'h100-47gb'}:
            raise ValueError('dispatch requires an H100 whole-GPU or 47GB MIG flavor and no fixed task')
        experiment = 'dispatch-unassigned'
        args_list = ['analysis.flavour.s3.dispatch', 'worker', '--root', str(root), '--name', args.name]
    if args.task is not None:
        campaign = json.loads(args.campaign.read_text())
        if product.replace('-', ' ') != campaign['hardware']['GPU'] or resource != 'nvidia.com/gpu':
            raise ValueError('scientific task must use the frozen whole-GPU flavor')
        experiment = campaign['experiments'][args.task]['name']
        args_list = ['analysis.flavour.s3.train', '--root', str(root), '--campaign', str(args.campaign.resolve()), '--task', str(args.task)]
    script = '\n'.join([
        'set -o pipefail', 'which() { command -v "$@"; }',
        'source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh',
        'set -e', f'cd {shlex.quote(str(repo))}',
        f'export PYTHONPATH={shlex.quote(str(repo / "src"))}:${{PYTHONPATH:-}}',
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',
        'exec python -m ' + shlex.join(args_list)])
    resources = {resource: 1}
    job = {'apiVersion': 'batch/v1', 'kind': 'Job',
           'metadata': {'name': args.name, 'namespace': 'xcheng', 'labels': {'app': 'pinpoint-s3'}},
           'spec': {'backoffLimit': 0, 'activeDeadlineSeconds': args.deadline,
                    'template': {'metadata': {'labels': {'app': 'pinpoint-s3', 'mount-eos': 'true', 'mount-kerberos-secret': 'true'}},
                                 'spec': {'restartPolicy': 'Never', 'serviceAccountName': 'default-editor',
                                          'securityContext': {'runAsUser': os.getuid(), 'runAsGroup': os.getgid()},
                                          'nodeSelector': {'nvidia.com/gpu.product': product},
                                          'volumes': [{'name': 'cvmfs', 'persistentVolumeClaim': {'claimName': 'cvmfs'}}],
                                          'containers': [{'name': 's3', 'image': 'docker.io/library/almalinux:9',
                                                          'command': ([] if args.task is not None or args.dispatch else ['/usr/bin/timeout', '300']) + ['/bin/bash', '-c', script],
                                                          'resources': {'requests': resources, 'limits': resources},
                                                          'volumeMounts': [{'name': 'cvmfs', 'mountPath': '/cvmfs', 'readOnly': True, 'mountPropagation': 'HostToContainer'}]}]}}}}
    path = output / f'{args.name}.job.json'
    path.write_text(json.dumps(job, indent=2) + '\n')
    # Admission must exercise the Pod, where CERN injects CPU/RAM and quota.
    pod = {'apiVersion': 'v1', 'kind': 'Pod', 'metadata': {'name': args.name + '-admission', 'namespace': 'xcheng', 'labels': job['spec']['template']['metadata']['labels']}, 'spec': job['spec']['template']['spec']}
    pod_path = output / f'{args.name}.pod-dry-run.json'
    pod_path.write_text(json.dumps(pod, indent=2) + '\n')
    admitted = json.loads(kubectl('create', '--dry-run=server', '-f', str(pod_path), '-o', 'json'))
    evidence = {'utc': datetime.now(timezone.utc).isoformat(), 'backend': 'kubeflow', 'context': kubectl('config', 'current-context').strip(),
                'namespace': 'xcheng', 'job': args.name, 'container_name': 's3',
                'stage': 'train' if args.task is not None else 'dispatch' if args.dispatch else 'probe',
                'experiment': experiment, 'task': args.task, 'root': str(root), 'submitted': False,
                'gpu_flavor': {'name': args.flavor, 'product': product, 'resource': resource, 'count': 1},
                'job_spec': job, 'pod_admission': admitted,
                'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
    if args.apply:
        created = json.loads(kubectl('create', '-f', str(path), '-o', 'json'))
        evidence.update(submitted=True, uid=created['metadata']['uid'])
    record.write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps({'record': str(record), 'submitted': evidence['submitted'], 'resources': admitted['spec']['containers'][0]['resources']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--flavor', choices=GPU_FLAVORS, default='h100-94gb')
    parser.add_argument('--campaign', type=Path)
    parser.add_argument('--task', type=int)
    parser.add_argument('--dispatch', action='store_true', help='offer an H100 slot to a still-idle Condor task after runtime readiness')
    parser.add_argument('--deadline', type=int, default=21600)
    parser.add_argument('--apply', action='store_true')
    prepare(parser.parse_args())


if __name__ == '__main__':
    main()
