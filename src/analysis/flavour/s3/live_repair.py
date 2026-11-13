"""Check each finished repair model while the remaining fits still run."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from analysis.flavour.inventory import write_json
from analysis.flavour.s3.inspect_repair import reusable_inspection
from analysis.flavour.s3.diagnose_repair_paths import reusable_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    args = parser.parse_args()
    configs = json.loads(args.campaign.read_text())['experiments']
    state_path = args.root / 'live_diagnostics_state.json'
    logs = args.root / 'audit/live_diagnostics'; logs.mkdir(parents=True, exist_ok=True)
    done = set(); state = {}
    try:
        while len(done) != len(configs):
            for config in configs:
                name = config['name']; directory = args.root / 'experiments' / name
                manifest = directory / 'manifest.json'
                status = json.loads(manifest.read_text())['status'] if manifest.exists() else None
                if status in ['internal_health_failed', 'failed']:
                    raise RuntimeError(f'{name}: architecture or training review required')
                if status != 'complete' or name in done:
                    continue
                state.update(status='checking', experiment=name, models_checked=len(done),
                             utc=datetime.now(timezone.utc).isoformat())
                write_json(state_path, state)
                with (logs / f'{name}.log').open('a') as stream:
                    commands = []
                    readback = directory / 'final_checkpoint_checks.json'
                    if not readback.exists() or json.loads(readback.read_text()).get('status') != 'pass':
                        commands.append(['analysis.flavour.s2.readback', '--root', str(args.root),
                            '--campaign', str(args.campaign), '--experiments', name])
                    if not reusable_inspection(args.root, name):
                        commands.append(['analysis.flavour.s3.inspect_repair', '--root', str(args.root),
                                         '--experiments', name])
                    if config['architecture'] == 'plane_module_v2' and not reusable_paths(args.root, name):
                        commands.append(['analysis.flavour.s3.diagnose_repair_paths', '--root', str(args.root),
                                         '--experiments', name])
                    for command in commands:
                        subprocess.run([sys.executable, '-m', *command], stdout=stream, stderr=subprocess.STDOUT, check=True)
                done.add(name)
                print(json.dumps({'utc':datetime.now(timezone.utc).isoformat(), 'experiment':name,
                                  'completed_model_diagnostics_passed': True, 'models_checked':len(done)}), flush=True)
                state.update(status='waiting_for_next_completed_fit', experiment=None, models_checked=len(done),
                             checked_experiments=sorted(done), utc=datetime.now(timezone.utc).isoformat())
                write_json(state_path, state)
            if len(done) != len(configs):
                time.sleep(15)
        state.update(status='all_completed_models_checked', models_checked=len(done))
        write_json(state_path, state)
    except Exception as error:
        state.update(status='failed', error=str(error), utc=datetime.now(timezone.utc).isoformat())
        write_json(state_path, state)
        raise


if __name__ == '__main__':
    main()
