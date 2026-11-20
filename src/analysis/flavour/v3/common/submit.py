"""Submit and account for v3 batch CPU/GPU operations on HTCondor and Kubeflow."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile

from analysis.flavour.s2.kubeflow import GPU_FLAVORS, kubectl
from analysis.flavour.v2.submit import choose_schedd, gpu_requirement


def _package_source(repo: Path, archive: Path) -> None:
    import gzip

    archive.parent.mkdir(parents=True, exist_ok=True)
    with archive.open('wb') as raw_fh, gzip.GzipFile(filename='', mode='wb', fileobj=raw_fh, mtime=0) as gz_fh:
        with tarfile.open(fileobj=gz_fh, mode='w') as stream:
            paths = [repo / 'src/analysis/__init__.py']
            paths.extend(sorted((repo / 'src/analysis/utils').glob('*.py')))
            paths.extend(sorted((repo / 'src/analysis/flavour').rglob('*.py')))
            paths.extend(sorted((repo / 'src/analysis/flavour').rglob('*.cpp')))
            s3_repair = repo / 'src/analysis/flavour/s3/runtime_repair.sh'
            if s3_repair.exists():
                paths.append(s3_repair)
            seen = set()
            for path in paths:
                if path in seen or not path.is_file():
                    continue
                seen.add(path)
                ti = stream.gettarinfo(str(path), arcname=str(path.relative_to(repo)))
                ti.mtime = 0
                ti.uid = 0
                ti.gid = 0
                ti.uname = ''
                ti.gname = ''
                with path.open('rb') as src_fh:
                    stream.addfile(ti, src_fh)


def _submit_kubeflow(args: argparse.Namespace, root: Path, script: list[str], archive: Path) -> dict:
    product, resource = GPU_FLAVORS[args.flavor]
    labels = {'app': 'pinpoint-v3', 'mount-eos': 'true', 'mount-kerberos-secret': 'true'}
    resources = {resource: 1}
    spec = {
        'restartPolicy': 'Never',
        'serviceAccountName': 'default-editor',
        'securityContext': {'runAsUser': os.getuid(), 'runAsGroup': os.getgid()},
        'nodeSelector': {'nvidia.com/gpu.product': product},
        'volumes': [{'name': 'cvmfs', 'persistentVolumeClaim': {'claimName': 'cvmfs'}}],
        'containers': [{
            'name': 'v3',
            'image': 'docker.io/library/almalinux:9',
            'env': [{'name': 'PINPOINT_JOB_ID', 'value': args.name}],
            'command': ['/bin/bash', '-c', '\n'.join(script)],
            'resources': {'requests': resources, 'limits': resources},
            'volumeMounts': [{
                'name': 'cvmfs',
                'mountPath': '/cvmfs',
                'readOnly': True,
                'mountPropagation': 'HostToContainer',
            }],
        }],
    }
    if getattr(args, 'exclude_nodes', None):
        spec['affinity'] = {
            'nodeAffinity': {
                'requiredDuringSchedulingIgnoredDuringExecution': {
                    'nodeSelectorTerms': [{
                        'matchExpressions': [{
                            'key': 'kubernetes.io/hostname',
                            'operator': 'NotIn',
                            'values': list(args.exclude_nodes),
                        }],
                    }],
                },
            },
        }
    metadata = {'name': args.name, 'namespace': 'xcheng', 'labels': labels}
    job = {
        'apiVersion': 'batch/v1',
        'kind': 'Job',
        'metadata': metadata,
        'spec': {
            'backoffLimit': 0,
            'activeDeadlineSeconds': args.deadline_seconds,
            'template': {'metadata': {'labels': labels}, 'spec': spec},
        },
    }
    folder = root / 'kubeflow'
    folder.mkdir(parents=True, exist_ok=True)
    job_path = folder / f'{args.name}.json'
    pod_path = folder / f'{args.name}.admission.json'
    job_path.write_text(json.dumps(job, indent=2) + '\n')
    pod_path.write_text(json.dumps({
        'apiVersion': 'v1',
        'kind': 'Pod',
        'metadata': {**metadata, 'name': args.name + '-admission'},
        'spec': spec,
    }, indent=2) + '\n')
    admitted = json.loads(kubectl('create', '--dry-run=server', '-f', str(pod_path), '-o', 'json'))
    created = json.loads(kubectl('create', '-f', str(job_path), '-o', 'json'))
    record_path = root / 'runtime' / f'{args.name}.submission.json'
    evidence = {
        'stage': args.stage,
        'backend': 'kubeflow',
        'name': args.name,
        'UTC': datetime.now(timezone.utc).isoformat(),
        'namespace': 'xcheng',
        'flavor': args.flavor,
        'product': product,
        'resource': resource,
        'uid': created['metadata']['uid'],
        'admitted_pod': admitted,
        'source_archive': str(archive),
        'command': args.command,
        'submitted': True,
        'gpu': True,
        'job_spec': job,
    }
    record_path.write_text(json.dumps(evidence, indent=2) + '\n')
    return {'backend': 'kubeflow', 'name': args.name, 'record': str(record_path), 'submitted': True}


def submit_job(
    root: Path,
    name: str,
    stage: str,
    command: list[str],
    *,
    backend: str = 'condor',
    flavor: str = 'h100-47gb',
    exclude_nodes: list[str] | None = None,
    cpus: int = 4,
    memory: str = '15GB',
    gpu: bool = False,
    gpu_model: str | None = None,
    min_gpu_memory_mb: int = 10000,
    disk: str = '20GB',
    job_flavour: str = 'workday',
    deadline_seconds: int = 28800,
    min_capability: float = 7.5,
    allow_backfill: bool = False,
) -> dict:
    root = root.resolve()
    for folder in ('runtime', 'logs', 'code'):
        (root / folder).mkdir(parents=True, exist_ok=True)
    record_path = root / 'runtime' / f'{name}.submission.json'
    if record_path.exists():
        raise ValueError(f'submission name already exists: {name}')

    repo = Path(__file__).resolve().parents[5]
    archive = root / 'code' / f'{name}.tar.gz'
    _package_source(repo, archive)
    import hashlib as _hl
    _h = _hl.sha256()
    with archive.open('rb') as _af:
        for _ch in iter(lambda: _af.read(1 << 20), b''):
            _h.update(_ch)
    archive_sha256 = _h.hexdigest()

    directory = Path.home() / 'condor/Pinpoint/flavour_v3' / stage / name
    directory.mkdir(parents=True, exist_ok=True)

    script = [
        '#!/bin/bash',
        'set -o pipefail',
        'which() { command -v "$@"; }; export -f which',
        'GPU_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d " ")',
        'if [ "$GPU_CAP" = "7.0" ]; then',
        '  source /cvmfs/sft.cern.ch/lcg/views/LCG_107_cuda/x86_64-el9-gcc11-opt/setup.sh',
        'else',
        '  source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh',
        'fi',
        'set -e',
        'ls /eos/home-x/xcheng/FASER/Pinpoint/results >/dev/null',
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1',
        f'export PINPOINT_CODE_ARCHIVE_SHA256={shlex.quote(archive_sha256)}',
        'export PINPOINT_V3_SCRATCH="${_CONDOR_SCRATCH_DIR:-/tmp}/pinpoint_v3_${PINPOINT_JOB_ID}"',
        'export PINPOINT_V2_SCRATCH="$PINPOINT_V3_SCRATCH"',
        'mkdir -p "$PINPOINT_V3_SCRATCH/code"',
        f'tar -xzf {shlex.quote(str(archive))} -C "$PINPOINT_V3_SCRATCH/code"',
        'export PYTHONPATH="$PINPOINT_V3_SCRATCH/code/src:${PYTHONPATH:-}"',
        'cd "$PINPOINT_V3_SCRATCH"',
        'exec python3 -m ' + shlex.join(command),
    ]

    args_ns = argparse.Namespace(
        stage=stage,
        name=name,
        flavor=flavor,
        exclude_nodes=exclude_nodes,
        deadline_seconds=deadline_seconds,
        command=command,
    )
    if backend == 'kubeflow':
        if not gpu:
            raise ValueError('Kubeflow v3 submission is intended for GPU jobs')
        script[-1] = script[-1].removeprefix('exec ') + ' 2>&1 | tee ' + shlex.quote(str(root / 'logs' / f'{name}.log'))
        return _submit_kubeflow(args_ns, root, script, archive)

    wrapper = directory / 'run.sh'
    wrapper.write_text('\n'.join(script) + '\n')
    wrapper.chmod(0o755)
    chosen, snapshot, checks = choose_schedd(root)
    content = [
        'universe = vanilla',
        f'executable = {wrapper}',
        f'initialdir = {directory}',
        f'output = {root}/logs/{name}.$(ClusterId).out',
        f'error = {root}/logs/{name}.$(ClusterId).err',
        f'log = {root}/logs/{name}.$(ClusterId).log',
        'environment = "PINPOINT_JOB_ID=$(ClusterId).$(Process)"',
        f'request_cpus = {cpus}',
        f'request_memory = {memory}',
        f'request_disk = {disk}',
        'requirements = (OpSysAndVer == "AlmaLinux9")',
        f'+JobFlavour = "{job_flavour}"',
        'should_transfer_files = YES',
        'when_to_transfer_output = ON_EXIT',
        'transfer_output_files = ""',
    ]
    if gpu:
        requirement = gpu_requirement(min_capability, gpu_model, min_gpu_memory_mb)
        content.extend(['request_gpus = 1', 'require_gpus = ' + requirement])
    if allow_backfill:
        content.append('+AllowBackfillSlot = true')
    content.append('queue 1')

    sub_path = directory / 'job.sub'
    sub_path.write_text('\n'.join(content) + '\n')
    response = subprocess.run(['condor_submit', '-name', chosen, str(sub_path)], capture_output=True, text=True)
    evidence = {
        'UTC': datetime.now(timezone.utc).isoformat(),
        'backend': 'condor',
        'stage': stage,
        'name': name,
        'schedd': chosen,
        'queue_snapshot': snapshot,
        'compatibility_checks': checks,
        'selection': 'minimum idle/held/running among compatible WRITE-authorized schedds',
        'source_base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        'source_archive': str(archive),
        'command': command,
        'wrapper': wrapper.read_text(),
        'submit': sub_path.read_text(),
        'cpus': cpus,
        'memory': memory,
        'gpu': gpu,
        'requested_minimum_capability': min_capability,
        'requested_minimum_GPU_memory_MiB': min_gpu_memory_mb,
        'allow_backfill': allow_backfill,
        'stdout': response.stdout,
        'stderr': response.stderr,
        'returncode': response.returncode,
        'submitted': response.returncode == 0,
    }
    record_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')
    if response.returncode:
        raise RuntimeError(response.stderr)
    return {'backend': 'condor', 'schedd': chosen, 'response': response.stdout, 'record': str(record_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--stage', default='V0')
    parser.add_argument('--backend', choices=('condor', 'kubeflow'), default='condor')
    parser.add_argument('--flavor', default='h100-47gb')
    parser.add_argument('--exclude-nodes', nargs='*', default=None)
    parser.add_argument('--cpus', type=int, default=4)
    parser.add_argument('--memory', default='15GB')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--gpu-model')
    parser.add_argument('--min-gpu-memory-mb', type=int, default=10000)
    parser.add_argument('--disk', default='20GB')
    parser.add_argument('--job-flavour', default='workday')
    parser.add_argument('--deadline-seconds', type=int, default=28800)
    parser.add_argument('--min-capability', type=float, default=7.5)
    parser.add_argument('--allow-backfill', action='store_true')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cmd = list(args.command)
    if cmd and cmd[0] == '--':
        cmd = cmd[1:]
    if not cmd:
        parser.error('a python module command is required after arguments')
    res = submit_job(
        args.root,
        args.name,
        args.stage,
        cmd,
        backend=args.backend,
        flavor=args.flavor,
        exclude_nodes=args.exclude_nodes,
        cpus=args.cpus,
        memory=args.memory,
        gpu=args.gpu,
        gpu_model=args.gpu_model,
        min_gpu_memory_mb=args.min_gpu_memory_mb,
        disk=args.disk,
        job_flavour=args.job_flavour,
        deadline_seconds=args.deadline_seconds,
        min_capability=args.min_capability,
        allow_backfill=args.allow_backfill,
    )
    print(json.dumps(res, ensure_ascii=False))


if __name__ == '__main__':
    main()
