"""Write tiny Condor wrappers and submit against a recorded compatible schedd.

Never uploads data or code; CERN workers access CVMFS and EOS directly. Selects
between standard bigbird schedulers after WRITE authorization and queue checks.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess


def command(args):
    result = subprocess.run(args, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f'{args[0]} exit {result.returncode}: {result.stderr.strip()} {result.stdout.strip()}')
    return result.stdout


def submit(args):
    snapshot = command(['condor_status', '-schedd', '-af', 'Name', 'TotalRunningJobs', 'TotalIdleJobs', 'TotalHeldJobs', 'MaxJobsRunning'])
    candidates = []
    for line in snapshot.splitlines():
        fields = line.split()
        if fields[0].startswith('bigbird'):
            candidates.append((int(fields[2]), int(fields[3]), int(fields[1]), fields[0]))
    attempts, chosen = [], None
    for _, _, _, name in sorted(candidates):
        policy = subprocess.run(['condor_config_val', '-name', name, '-schedd',
                                 'SUBMIT_REQUIREMENT_NoEos'], text=True, capture_output=True)
        if policy.returncode or policy.stdout.strip().lower() != 'true':
            attempts.append({'schedd': name, 'compatible': False,
                             'reason': 'direct EOS logs/working directory need unconditional NoEos acceptance',
                             'NoEos': policy.stdout.strip()})
            continue
        ping = subprocess.run(['condor_ping', '-type', 'SCHEDD', '-name', name, '-table', 'WRITE'], text=True, capture_output=True)
        attempts.append({'schedd': name, 'compatible': True, 'NoEos': policy.stdout.strip(),
                         'returncode': ping.returncode, 'output': ping.stdout})
        if ping.returncode == 0 and 'ALLOW' in ping.stdout:
            chosen = name
            break
    if chosen is None:
        raise RuntimeError('no standard bigbird scheduler grants WRITE')
    repo = Path(__file__).resolve().parents[3]
    submit_dir = Path.home() / 'condor' / 'Pinpoint' / 'flavour_s0_s1'
    submit_dir.mkdir(parents=True, exist_ok=True)
    args.root.mkdir(parents=True, exist_ok=True)
    logs = args.root / 'logs'
    logs.mkdir(exist_ok=True)
    wrapper = submit_dir / 'run.sh'
    wrapper.write_text(f'''#!/bin/bash
set -eo pipefail
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
set -u
export PYTHONPATH={repo}/src:${{PYTHONPATH}}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd {repo}
exec python -m "$@"
''')
    wrapper.chmod(0o755)
    queue = 'queue 1'
    cpus, memory, flavour = 2, '4GB', 'microcentury'
    if args.stage in ['pilot', 'inventory']:
        plan = args.root / ('pilot_plan.json' if args.stage == 'pilot' else 'plan.json')
        output = args.root / ('pilot' if args.stage == 'pilot' else 'scan')
        arguments = f'analysis.flavour.inventory scan --plan {plan} --task $(Process) --output {output} --cpus {cpus}'
        number = len(json.loads(plan.read_text())['tasks'])
        if number > 24:
            raise ValueError('plan exceeds 24 jobs; increase --files-per-task and regenerate plan')
        queue = f'queue {number}'
    elif args.stage in ['pilot_merge', 'merge']:
        plan = args.root / ('pilot_plan.json' if args.stage == 'pilot_merge' else 'plan.json')
        input_dir = args.root / ('pilot' if args.stage == 'pilot_merge' else 'scan')
        output = args.root / ('pilot_inventory' if args.stage == 'pilot_merge' else 'inventory')
        cpus = 8 if args.stage == 'merge' else 2
        arguments = f'analysis.flavour.inventory merge --plan {plan} --input {input_dir} --output {output} --cpus {cpus}'
        memory = '24GB' if args.stage == 'merge' else '4GB'
    elif args.stage == 'convert':
        arguments = f'analysis.flavour.create_data --input /eos/project-f/fasersim-bonn/public/pinpoint/data/$(run)/$(run)_000.root --output {args.root}/shards/$(run) --cpus {cpus}'
        queue = 'queue run in 10081 10082 10083 10084'
        memory = '8GB'
    elif args.stage == 'diagnostics':
        arguments = f'analysis.flavour.evaluate --input {args.root}/shards --output {args.root}/diagnostics'
        memory = '8GB'
    elif args.stage == 'loader_profile':
        arguments = f'analysis.flavour.profile_loader --input {args.root}/shards --output {args.root}/loader_profile.json'
        cpus, memory = 3, '6GB'
    elif args.stage == 'source_match':
        arguments = f'analysis.flavour.source_match --source /eos/project-f/fasersim-bonn/public/pinpoint/2026-05-08-genie/$(source) --run $(run) --inventory {args.root}/pilot_inventory/events.parquet --output {args.root}/source_match'
        queue = 'queue run, source from (\n10081 cc-nue.gfaser.root\n10082 cc-num.gfaser.root\n10083 cc-nut.gfaser.root\n10084 nc-nu.gfaser.root\n)'
    else:
        raise ValueError(args.stage)
    text = f'''universe = vanilla
executable = {wrapper}
arguments = {arguments}
output = {logs}/{args.stage}.$(ClusterId).$(Process).out
error = {logs}/{args.stage}.$(ClusterId).$(Process).err
log = {logs}/{args.stage}.$(ClusterId).log
request_cpus = {cpus}
request_memory = {memory}
request_disk = 1GB
+JobFlavour = "{flavour}"
requirements = (OpSysAndVer == "AlmaLinux9")
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_output_files = ""
{queue}
'''
    submit_file = submit_dir / f'{args.stage}.sub'
    submit_file.write_text(text)
    evidence = {'utc': datetime.now(timezone.utc).isoformat(), 'stage': args.stage,
                'schedd_snapshot': snapshot, 'authorization': attempts, 'chosen': chosen,
                'basis': 'direct EOS policy compatible, standard CERN CPU, WRITE authorized; lowest idle then held then running',
                'submit': text, 'wrapper': wrapper.read_text()}
    record = args.root / f"submission_{args.stage}_{datetime.now(timezone.utc).strftime('%H%M%S')}.json"
    record.write_text(json.dumps(evidence, indent=2) + '\n')
    result = subprocess.run(['condor_submit', '-name', chosen, str(submit_file)], text=True, capture_output=True)
    evidence.update(returncode=result.returncode, response=result.stdout, stderr=result.stderr)
    record.write_text(json.dumps(evidence, indent=2) + '\n')
    if result.returncode:
        raise RuntimeError(f'submit exit {result.returncode}: {result.stderr.strip()} (evidence: {record})')
    response = result.stdout
    print(response, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['pilot', 'pilot_merge', 'inventory', 'merge', 'convert', 'diagnostics', 'source_match', 'loader_profile'])
    parser.add_argument('--root', type=Path, required=True)
    submit(parser.parse_args())


if __name__ == '__main__':
    main()
