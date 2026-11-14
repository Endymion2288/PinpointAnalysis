"""CPU-only HGB screen and seed repeats with the frozen development subset."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from analysis.flavour.inventory import write_json


def execute(root, name, task, seed, lr, threads):
    manifest = root/'experiments'/name/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['status'] == 'complete':
        return json.loads(manifest.read_text())
    subprocess.run([sys.executable, '-m', 'analysis.flavour.v2.r1.baseline',
        '--root', str(root), '--name', name, '--task', task, '--seed', str(seed),
        '--threads', str(threads), '--learning-rate', str(lr)], check=True)
    return json.loads(manifest.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    contract = json.loads((args.root/'campaign.json').read_text())['B0']
    selected = {}
    for task in ['T3', 'T4']:
        trials = []
        for lr in contract['learning_rates']:
            name = f'screen_b0_{task.lower()}_lr{lr:g}_s42'
            manifest = execute(args.root, name, task, 42, lr, args.threads)
            metrics = manifest['screen_metrics']['T3']
            trials.append((metrics['BA'], -metrics['NLL'], lr, name))
        winner = max(trials, key=lambda row: row[:2])
        selected[task] = {'learning_rate': winner[2], 'seed42_model': winner[3], 'models': [winner[3]]}
        for seed in [17, 2026]:
            name = f'final_b0_{task.lower()}_s{seed}'
            execute(args.root, name, task, seed, winner[2], args.threads)
            selected[task]['models'].append(name)
    write_json(args.root/'b0_selected.json', {'status': 'complete', 'selected': selected,
        'selection': 'fixed Tune screen common non-tau BA3 then NLL3; CPU boosting budget separate'})


if __name__ == '__main__':
    main()
