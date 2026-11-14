"""English figures for real optimization checks and later R1 comparisons."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def fitting(root):
    destination = root/'figures'
    destination.mkdir(exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    rows = []
    for axis, task in zip(axes, ['T3', 'T4']):
        axis.axhline(np.log(3 if task == 'T3' else 4), color='black', linestyle=':', label='Constant benchmark')
        for family in ['B1', 'B2', 'B3', 'B4']:
            record = json.loads((root/'fit/interactive_a1'/f'{family}_{task}.json').read_text())
            history = record['history']
            axis.plot([r['step'] for r in history], [r['NLL'] for r in history], marker='o', markersize=3, label=family)
            rows.append({'family': family, 'task': task, 'events': record['events'],
                'updates': record['steps'], 'NLL': record['final']['NLL'], 'accuracy': record['final']['accuracy'],
                'parameters': record['parameters'], 'GPU_peak_MiB': record['GPU_peak_MiB'], 'seconds': record['seconds']})
        axis.set(xlabel='Optimizer updates', ylabel='Negative log-likelihood', yscale='log',
                 title=f'{task}: {24 if task == "T3" else 32} informative Train events')
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    figure.suptitle('Real-event optimization checks (seed 42, FP32)')
    for suffix in ['pdf', 'png']:
        figure.savefig(destination/f'r1_real_fit.{suffix}', dpi=160)
    plt.close(figure)
    pd.DataFrame(rows).to_csv(destination/'r1_real_fit.csv', index=False)


def results(root):
    table = pd.read_csv(root/'evaluation/model_domain_metrics.csv')
    table = table[table.domain.eq('all') & table.seed.eq(42)]
    figure, axis = plt.subplots(figsize=(9, 4), constrained_layout=True)
    for task, marker in [('T3', 'o'), ('T4', 's')]:
        group = table[table.task.eq(task)]
        axis.scatter(group.family, group.BA3, label=task, marker=marker)
    axis.set(ylabel='Balanced accuracy on common non-tau events', xlabel='Model family', title='R1 development Tune comparison')
    axis.grid(alpha=.2)
    axis.legend()
    for suffix in ['pdf', 'png']:
        figure.savefig(root/'figures'/f'r1_common_three.{suffix}', dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--results', action='store_true')
    args = parser.parse_args()
    fitting(args.root)
    if args.results:
        results(args.root)


if __name__ == '__main__':
    main()
