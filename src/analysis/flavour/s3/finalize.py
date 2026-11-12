"""Wait for a supervised confirmation Job, then run the complete S3 evaluation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from analysis.flavour.inventory import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--confirmation', type=Path, required=True)
    parser.add_argument('--comparisons', type=Path, required=True)
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    state_path = args.root / 'confirmation_finalize_state.json'
    state = {'job': args.job}

    def execute(stage, arguments):
        state.update(stage=stage, status='running', utc=datetime.now(timezone.utc).isoformat())
        write_json(state_path, state)
        print(json.dumps({'stage': stage, 'command': arguments}), flush=True)
        subprocess.run(arguments, check=True)

    def module(stage, name, *arguments):
        execute(stage, [sys.executable, '-m', 'analysis.flavour.s3.' + name,
                        '--root', str(args.root), *map(str, arguments)])

    try:
        module('confirmation_readback', 'monitor', '--campaign', args.confirmation,
               '--watch', '--poll-seconds', 60, '--deadline', 21600)
        if not json.loads((args.root / 'monitor_state.json').read_text())['all_readbacks_passed']:
            raise RuntimeError('confirmation training/readback incomplete at monitor deadline')
        execute('backend_terminal', ['kubectl', 'wait', '-n', 'xcheng', '--for=condition=complete',
                                     'job/' + args.job, '--timeout=60s'])
        module('allocation_accounting', 'accounting')
        evaluation = args.root / 'evaluation_final'
        module('four_class_evaluation', 'evaluate', '--campaign', args.campaign,
               '--confirmation', args.confirmation, '--comparisons', args.comparisons,
               '--output', evaluation, '--bootstrap', 400, '--strict')
        summary = json.loads((evaluation / 'summary.json').read_text())
        names = [e['name'] for e in summary['experiments']
                 if e['name'].startswith(('s3_', 'part_', 'pn_'))]
        conditional = evaluation / 'conditional'
        module('conditional_diagnostics', 'conditional', '--comparisons', args.comparisons,
               '--experiments', *names, '--bootstrap', 400, '--output', conditional)
        for name in ['s3_f16_context_latent_s42_e4', 's3_f16_context_latent_tracker_s42_e4']:
            module('native_displays_' + name, 'display', '--experiment', name,
                   '--evaluation-tag', 'evaluation_final', '--pages', 20)
        module('strict_acceptance', 'accept', '--campaign', args.campaign,
               '--confirmation', args.confirmation, '--comparisons', args.comparisons,
               '--evaluation', evaluation / 'summary.json',
               '--conditional', conditional / 'summary.json', '--require-complete')
        state.update(status='strict_gate_passed', utc=datetime.now(timezone.utc).isoformat())
        write_json(state_path, state)
    except Exception as error:
        state.update(status='failed', error=str(error), utc=datetime.now(timezone.utc).isoformat())
        write_json(state_path, state)
        raise


if __name__ == '__main__':
    main()
