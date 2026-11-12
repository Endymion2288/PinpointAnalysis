"""Read back each newly completed S3 model and retain compact progress.

This observes result publication, not scheduler liveness. Backend accounting
and explicit queue queries remain the authority for job state and GPU cost.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import pandas as pd
import pyarrow as pa
import torch

from analysis.flavour.inventory import write_json
from analysis.flavour.s2.readback import readback


def snapshot(root, campaign, validation):
    rows = []
    for config in campaign['experiments']:
        output = root / 'experiments' / config['name']
        load = lambda filename: json.loads((output / filename).read_text()) if (output / filename).exists() else None
        manifest, checks = load('manifest.json'), load('final_checkpoint_checks.json')
        if manifest and manifest['status'] == 'complete' and not (checks and checks['status'] == 'pass'):
            checks = readback(root, config, campaign['hardware'], validation)
        history = load('history.json') or []
        progress = load('progress.json')
        row = {'experiment': config['name'], 'published_model_status': manifest['status'] if manifest else 'not_published',
               'complete_epochs': len(history), 'readback_pass': bool(checks and checks['status'] == 'pass'),
               'progress_epoch': progress['epoch'] if progress else None,
               'progress_step': progress['step'] if progress else None,
               'amp_skipped_steps': progress['amp_skipped_steps'] if progress else None}
        if history:
            result = history[-1]['validation']
            row['latest_validation'] = {'NLL': result['NLL'], 'macro_F1': result['macro_F1'],
                'balanced_accuracy': result['balanced_accuracy'],
                'macro_AP': sum(result['classwise'][str(c)]['PR_AUC'] for c in range(4)) / 4,
                'class_AP': [result['classwise'][str(c)]['PR_AUC'] for c in range(4)]}
        rows.append(row)
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'experiments': rows,
              'total_experiments': len(rows), 'complete_models': sum(r['published_model_status'] == 'complete' for r in rows),
              'readback_passed': sum(r['readback_pass'] for r in rows),
              'all_readbacks_passed': all(r['readback_pass'] for r in rows),
              'scope': 'model publication and completed-model CPU checks; scheduler liveness and costs separately queried'}
    write_json(root / 'monitor_state.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--poll-seconds', type=int, default=60)
    parser.add_argument('--deadline', type=int, default=21600)
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60:
        parser.error('poll interval must be between 1 and 60 seconds')
    torch.set_num_threads(1); pa.set_cpu_count(1); pa.set_io_thread_count(1)
    campaign = json.loads(args.campaign.read_text())
    catalog = pd.read_parquet(args.root / 'catalog.parquet', columns=['run', 'chunk', 'row', 'label', 'split'])
    validation = catalog[catalog.split.eq('validation')]
    end = time.monotonic() + args.deadline
    previous = None
    while True:
        report = snapshot(args.root, campaign, validation)
        state = [(r['experiment'], r['complete_epochs'], r['published_model_status'], r['readback_pass']) for r in report['experiments']]
        if state != previous:
            print(json.dumps(report), flush=True); previous = state
        if not args.watch or report['all_readbacks_passed'] or time.monotonic() >= end:
            return
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
