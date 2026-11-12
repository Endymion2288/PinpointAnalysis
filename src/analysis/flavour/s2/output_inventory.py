"""Summarize existing S2 outputs and backend accounting for a frozen campaign."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from analysis.flavour.inventory import write_json


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def inventory(root, campaign, evaluation):
    accounting = read_json(root / 'campaign_accounting.json')
    summary = read_json(evaluation / 'summary.json')
    evaluated = {item['name']: item for item in summary.get('experiments', [])}
    configurations = []
    for task, config in enumerate(campaign['experiments']):
        name = config['name']
        output = root / 'experiments' / name
        manifest = read_json(output / 'manifest.json')
        checks = read_json(output / 'final_checkpoint_checks.json')
        allocation = accounting.get('experiments', {}).get(name, {})
        records = [item for item in accounting.get('allocations', [])
                   if item.get('experiment') == name]
        result = evaluated.get(name, {})
        item = {
            'task': task, 'name': name, 'role': config.get('role'),
            'status': manifest.get('status', 'not_started'),
            'frozen_config_matches': manifest.get('config') == config,
            'readback_status': checks.get('status'),
            'readback_passed': sum(checks.get('checks', {}).values()),
            'readback_utc': checks.get('utc'),
            'checkpoint_epoch': checks.get('checkpoint_epoch'),
            'GPU': manifest.get('gpu_name'),
            'optimizer_steps': manifest.get('optimizer_steps'),
            'amp_skipped_steps': manifest.get('amp_skipped_steps'),
            'evaluation_present': bool(result),
            'tau_AP': result.get('tau', {}).get('PR_AUC'),
            'all_attempts_terminal': allocation.get('all_terminal', False),
            'allocation_complete': allocation.get('allocation_complete', False),
            'GPU_hours_at_accounting_snapshot': allocation.get('GPU_hours'),
            'backend_records_at_accounting_snapshot': [
                {key: record.get(key) for key in ['schedd', 'cluster', 'process',
                    'backend', 'status', 'exit_code', 'starts', 'GPU_hours']}
                for record in records],
        }
        item['output_complete'] = bool(
            item['status'] == 'complete' and item['frozen_config_matches'] and
            item['readback_status'] == 'pass' and item['readback_passed'] == 14 and
            item['checkpoint_epoch'] == config['epochs'] and
            item['optimizer_steps'] == config['epochs'] * config['steps_per_epoch'] and
            item['amp_skipped_steps'] == 0 and item['evaluation_present'] and
            item['all_attempts_terminal'] and item['allocation_complete'])
        configurations.append(item)
    complete = all(item['output_complete'] for item in configurations)
    strict = bool(complete and summary.get('requested_comparisons_complete') and
                  accounting.get('missing_history_records') == 0)
    return {
        'utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'frozen campaign output/readback/evaluation/accounting inventory; accepted samples and proxy unchanged',
        'accounting_utc': accounting.get('utc'),
        'evaluation_commit': summary.get('code_commit'),
        'configs': configurations,
        'counts': {
            'configurations': len(configurations),
            'manifest_status': dict(Counter(item['status'] for item in configurations)),
            'output_complete': sum(item['output_complete'] for item in configurations),
            'paired_comparisons': len(summary.get('paired_comparisons', [])),
            'seed_mean_comparisons': len(summary.get('seed_mean_comparisons', [])),
            'pending_comparisons': len(summary.get('pending_comparisons', [])),
        },
        'missing_history_records': accounting.get('missing_history_records'),
        'all_outputs_complete': complete,
        'strict_output_gate_passed': strict,
        'interpretation': 'Output gate only; scientific conclusions and stage acceptance require the Chinese stage report. Backend records are from the dated accounting snapshot, not a live queue query.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    report = inventory(args.root, read_json(args.campaign),
                       args.evaluation or args.root / 'evaluation_final')
    write_json(args.root / 'campaign_output_inventory.json', report)
    print(json.dumps({key: report[key] for key in ['utc', 'counts', 'strict_output_gate_passed']}))
    if args.require_complete and not report['strict_output_gate_passed']:
        raise SystemExit('S2 outputs or requested comparisons remain incomplete; see campaign_output_inventory.json')


if __name__ == '__main__':
    main()
