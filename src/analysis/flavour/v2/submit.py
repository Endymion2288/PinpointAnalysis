"""Submit one R0 batch operation with an immutable worktree source snapshot."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile

def choose_schedd(root):
    """Match our AFS executable/direct-EOS log layout to live schedd policy."""
    snapshot = subprocess.check_output(['condor_status', '-schedd', '-af', 'Name',
        'TotalRunningJobs', 'TotalIdleJobs', 'TotalHeldJobs'], text=True)
    candidates = []
    checks = []
    for line in snapshot.splitlines():
        name, running, idle, held = line.split()
        if name.startswith('bigbird'):
            candidates.append((int(idle), int(held), int(running), name))
    for *_, name in sorted(candidates):
        policy = subprocess.run(['condor_config_val', '-name', name, '-schedd',
            'SUBMIT_REQUIREMENT_NoEos'], capture_output=True, text=True)
        # The literal regexp("^/eos", Iwd) also occurs in negative NoEos
        # clauses. Its presence does not establish compatibility with EOS logs.
        if policy.returncode or policy.stdout.strip().lower() != 'true':
            checks.append({'schedd': name, 'compatible': False, 'EOS_policy': policy.stdout,
                           'reason': 'AFS executable and direct EOS log paths require the unconditional EOS policy'})
            continue
        ping = subprocess.run(['condor_ping', '-type', 'SCHEDD', '-name', name,
                               '-table', 'WRITE'], capture_output=True, text=True)
        compatible = ping.returncode == 0 and 'ALLOW' in ping.stdout
        checks.append({'schedd': name, 'compatible': compatible, 'EOS_policy': policy.stdout, 'WRITE': ping.stdout})
        if compatible:
            return name, snapshot, checks
    raise RuntimeError('no WRITE-authorized schedd compatible with the current path layout')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--memory", default="16GB")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = args.root.resolve()
    for folder in ["runtime", "logs", "code"]:
        (root / folder).mkdir(parents=True, exist_ok=True)
    record = root / "runtime" / f"{args.name}.submission.json"
    if record.exists():
        raise ValueError("submission names must identify one immutable attempt")
    repo = Path(__file__).resolve().parents[4]
    archive = root / "code" / f"{args.name}.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        paths = [repo / "src/analysis/__init__.py", repo/'src/analysis/flavour/s3/runtime_repair.sh'] + sorted((repo / "src/analysis/flavour").rglob("*.py"))
        for path in paths:
            stream.add(path, arcname=str(path.relative_to(repo)))
    directory = Path.home() / "condor/Pinpoint/flavour_v2/R0" / args.name
    directory.mkdir(parents=True, exist_ok=True)
    script = ["#!/bin/bash", "set -o pipefail", 'which() { command -v "$@"; }',
              "source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh", "set -e",
              "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1",
              'export PINPOINT_R0_SCRATCH="${_CONDOR_SCRATCH_DIR:-/tmp}/pinpoint_r0_${PINPOINT_JOB_ID}"',
              'mkdir -p "$PINPOINT_R0_SCRATCH/code"',
              "tar -xzf " + shlex.quote(str(archive)) + ' -C "$PINPOINT_R0_SCRATCH/code"',
              'export PYTHONPATH="$PINPOINT_R0_SCRATCH/code/src:${PYTHONPATH:-}"',
              'cd "$PINPOINT_R0_SCRATCH"']
    if args.gpu and '--compile-local' in args.command:
        assets = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_repair_2026-11-13/audit')
        (root/'audit').mkdir(exist_ok=True)
        for name in ['compiler_headers.tar.gz', 'compiler_linker_files.tar.gz']:
            if not (root/'audit'/name).exists():
                shutil.copyfile(assets/name, root/'audit'/name)
        script.append('source "$PINPOINT_R0_SCRATCH/code/src/analysis/flavour/s3/runtime_repair.sh" ' + shlex.quote(str(root)))
    script.append("exec python -m " + shlex.join(args.command))
    wrapper = directory / "run.sh"
    wrapper.write_text("\n".join(script) + "\n")
    wrapper.chmod(0o755)
    chosen, snapshot, checks = choose_schedd(root)
    content = ["universe = vanilla", f"executable = {wrapper}", f"initialdir = {directory}",
               f"output = {root}/logs/{args.name}.$(ClusterId).out",
               f"error = {root}/logs/{args.name}.$(ClusterId).err",
               f"log = {root}/logs/{args.name}.$(ClusterId).log",
               'environment = "PINPOINT_JOB_ID=$(ClusterId).$(Process)"',
               f"request_cpus = {args.cpus}", f"request_memory = {args.memory}", "request_disk = 12GB",
               'requirements = (OpSysAndVer == "AlmaLinux9")', '+JobFlavour = "workday"',
               "should_transfer_files = YES", "when_to_transfer_output = ON_EXIT", 'transfer_output_files = ""']
    if args.gpu:
        content += ["request_gpus = 1", "require_gpus = (GlobalMemoryMb >= 10000) && (Capability >= 8.0)"]
    content.append("queue 1")
    path = directory / "job.sub"
    path.write_text("\n".join(content) + "\n")
    response = subprocess.run(["condor_submit", "-name", chosen, str(path)], capture_output=True, text=True)
    evidence = {"UTC": datetime.now(timezone.utc).isoformat(), "stage": "R0", "name": args.name,
                "schedd": chosen, "queue_snapshot": snapshot, "compatibility_checks": checks,
                "selection": "minimum idle/held/running among compatible WRITE-authorized schedds",
                "source_base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                "source_archive": str(archive), "source_snapshot": "exact working source at submission; unrelated worktree changes excluded",
                "command": args.command, "wrapper": wrapper.read_text(), "submit": path.read_text(),
                "cpus": args.cpus, "gpu": args.gpu, "stdout": response.stdout, "stderr": response.stderr,
                "returncode": response.returncode, "submitted": response.returncode == 0}
    record.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"schedd": chosen, "response": response.stdout, "record": str(record)}, ensure_ascii=False))
    if response.returncode:
        raise RuntimeError(response.stderr)


if __name__ == "__main__":
    main()
