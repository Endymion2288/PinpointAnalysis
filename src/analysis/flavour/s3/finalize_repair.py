"""Finish revision 2 readback, health, four-class evaluation and output gates."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from analysis.flavour.inventory import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--confirmation', type=Path, required=True)
    parser.add_argument('--comparisons', type=Path, required=True)
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--deadline', type=int, default=86400)
    args = parser.parse_args()
    campaigns = [json.loads(p.read_text()) for p in [args.campaign, args.confirmation]]
    names = [c['name'] for campaign in campaigns for c in campaign['experiments']]
    state_path = args.root / 'repair_finalize_state.json'
    state = {}
    def stage(name, arguments):
        state.update(stage=name, status='running', utc=datetime.now(timezone.utc).isoformat())
        write_json(state_path, state)
        subprocess.run(arguments, check=True)
    def module(name, *arguments):
        stage(name, [sys.executable, '-m', 'analysis.flavour.s3.' + name,
                     '--root', str(args.root), *map(str, arguments)])
    try:
        end = time.monotonic() + args.deadline
        while True:
            manifests = [args.root / 'experiments' / name / 'manifest.json' for name in names]
            statuses = [json.loads(p.read_text())['status'] if p.exists() else 'not_started' for p in manifests]
            if 'internal_health_failed' in statuses:
                raise RuntimeError('failed internal health requires an architecture/training review')
            if all(value == 'complete' for value in statuses):
                break
            if not args.wait or time.monotonic() > end:
                raise RuntimeError('frozen fits not yet complete')
            state.update(stage='waiting_for_frozen_fits', complete=sum(s == 'complete' for s in statuses),
                         total=len(names), utc=datetime.now(timezone.utc).isoformat())
            write_json(state_path, state)
            time.sleep(30)
        for campaign in [args.campaign, args.confirmation]:
            module('monitor', '--campaign', campaign)
            if not json.loads((args.root / 'monitor_state.json').read_text())['all_readbacks_passed']:
                raise RuntimeError('checkpoint readback incomplete')
        module('inspect_repair', '--experiments', *names)
        repaired = [c['name'] for campaign in campaigns for c in campaign['experiments']
                    if c['architecture'] == 'plane_module_v2']
        module('diagnose_repair_paths', '--experiments', *repaired)
        # Give supervisors time to reap their last child; terminal allocation
        # evidence remains mandatory and cannot be replaced with these outputs.
        module('accounting_repair')
        evaluation = args.root / 'evaluation_final'
        module('evaluate', '--campaign', args.campaign, '--confirmation', args.confirmation,
               '--comparisons', args.comparisons, '--output', evaluation, '--bootstrap', 400, '--strict')
        module('conditional', '--comparisons', args.comparisons, '--experiments', *names,
               '--bootstrap', 400, '--output', evaluation / 'conditional')
        module('plot_repair_health', '--experiments', *names, '--output', evaluation / 'internal_training_health.pdf')
        stage('plot_repair_comparisons', [sys.executable,'-m','analysis.flavour.s3.plot_repair_comparisons',
            '--evaluation',str(evaluation/'summary.json'),'--output',str(evaluation)])
        module('token_census', '--campaign', args.campaign, '--confirmation', args.confirmation,
               '--output', evaluation / 'token_census')
        for name in ['s3r_multiscale_native_s42_e4', 's3r_multiscale_tracker_s42_e4']:
            module('display', '--experiment', name, '--evaluation-tag', 'evaluation_final', '--pages', 20)
        module('accept_repair', '--campaign', args.campaign, '--confirmation', args.confirmation,
               '--comparisons', args.comparisons, '--evaluation', evaluation / 'summary.json',
               '--conditional', evaluation / 'conditional/summary.json', '--require-complete')
        state.update(status='strict_gate_passed', utc=datetime.now(timezone.utc).isoformat())
        write_json(state_path, state)
    except Exception as error:
        state.update(status='failed', error=str(error), utc=datetime.now(timezone.utc).isoformat())
        write_json(state_path, state)
        raise


if __name__ == '__main__':
    main()
