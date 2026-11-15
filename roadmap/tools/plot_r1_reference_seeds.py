"""Show realized seed variation beside intervals conditional on trained models."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--family', default='B4')
    parser.add_argument('--interim', default='reference_seeds')
    args = parser.parse_args()
    family = args.family
    folder = args.root/'interim'/args.interim
    result = json.loads((folder/'evaluation/results.json').read_text())
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    seeds = [17, 42, 2026]
    spread = {(row['family'], row['task']): row for row in result['seed_spread']}
    for task, marker in [('T3','o'), ('T4','s')]:
        axes[0].plot(range(3), np.asarray(spread[(family,task)]['BA3_values'])*100,
                     marker+'-', label=f'{family} {task}')
        axes[0].axhline(spread[('B0',task)]['BA3_mean']*100, linestyle='--',
                        color='tab:blue' if task=='T3' else 'tab:orange', alpha=.5, label=f'HGB {task}')
    axes[0].set(xticks=range(3), xticklabels=[str(seed) for seed in seeds], xlabel='Training seed',
                ylabel='Common non-tau balanced accuracy (%)', title='Realized seed values, same four-pass exposure')
    axes[0].legend(fontsize=8, ncol=2)
    pair = next(row for row in result['mean_seed_pairs']
                if row['a_family_task_kind']==[family,'T3','main'] and row['b_family_task_kind']==[family,'T4','main'])
    metrics = ['BA3','recall_e','recall_mu','recall_NC']
    for i, seed in enumerate(seeds):
        points = [pair['metrics'][key]['per_seed_differences'][i]*100 for key in metrics]
        axes[1].scatter(np.arange(4)+(i-1)*.13, points, label=f'Seed {seed}', s=28)
    for i, metric in enumerate(metrics):
        row = pair['metrics'][metric]
        axes[1].vlines(i, *np.asarray(row['95pct'])*100, color='black', linewidth=1.8)
        axes[1].scatter(i, row['difference_a_minus_b']*100, color='black', marker='_', s=150,
                        label='Mean and event interval' if i==0 else None)
    axes[1].axhline(0., color='black', linestyle=':', linewidth=.8)
    axes[1].set(xticks=range(4), xticklabels=['Balanced accuracy','Electron recall','Muon recall','NC recall'],
                ylabel='T3 minus conditional T4 (percentage points)',
                title='Seed differences and mean-seed 95% event intervals')
    axes[1].legend(fontsize=8, ncol=2)
    for axis in axes:
        axis.grid(axis='y', alpha=.2)
    figure.suptitle(f'{family}: development Tune 40k; seeds 17, 42, 2026\nIntervals condition on these trained models; colored dots include seed and runtime variation')
    output = folder/'figures'
    output.mkdir(exist_ok=True)
    for ext in ['pdf','png']:
        figure.savefig(output/f'r1_reference_seeds.{ext}', dpi=170)
    plt.close(figure)


if __name__=='__main__':
    main()
