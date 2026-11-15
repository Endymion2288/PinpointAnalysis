"""Plot completed R1 condition comparisons on frozen Tune events."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='R1 results root')
    parser.add_argument('--interim', default='stress_seed42', help='Completed folder under R1/interim')
    args = parser.parse_args()
    root = args.root/'interim'/args.interim
    reports = json.loads((root/'evaluation/stress_results.json').read_text())['models']
    entries = json.loads((root/'evaluation_config.json').read_text())['models']
    colors = {'B0': 'tab:gray', 'B1': 'tab:red', 'B2': 'tab:orange',
              'B4': 'tab:blue', 'B4-MAE': 'tab:purple'}
    labels = [f"{e['family']} / {e['task']}" for e in entries]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for row, column, condition, title in [
        (0, 0, 'noise', 'Sparse-noise proxy'),
        (0, 1, 'combined', 'Combined condition proxy'),
        (1, 0, 'remove_pixel', 'Frozen weights: remove pixel'),
        (1, 1, 'remove_tracker', 'Frozen weights: remove tracker'),
    ]:
        axis = axes[row, column]
        for i, entry in enumerate(entries):
            metric = reports[entry['name']]['conditions'][condition]['paired_descriptive_differences']['BA3']
            point = metric['difference_a_minus_b']*100
            limits = np.asarray(metric['95pct'])*100
            color = colors[entry['family']]
            axis.hlines(i, *limits, color=color, linewidth=1.5)
            axis.scatter(point, i, color=color, marker='o' if entry['task']=='T3' else 's', s=28)
        axis.axvline(0., color='black', linestyle=':', linewidth=.8)
        axis.set(title=title, yticks=range(len(entries)), yticklabels=labels,
                 ylim=(len(entries)-.5, -.5), xlabel='Change in common three-class BA (pp)')
        axis.grid(axis='x', alpha=.2)
    t4_entries = [e for e in entries if e['task']=='T4']
    for row, metric, title in [(0, 'efficiency', 'Tau efficiency at frozen nominal cut'),
                                (1, 'actual_FPR', 'Non-tau acceptance at frozen nominal cut')]:
        axis = axes[row, 2]
        for i, entry in enumerate(t4_entries):
            conditions = reports[entry['name']]['conditions']
            points, limits = [], []
            for condition in ['noise', 'combined']:
                result = next(r for r in conditions[condition]['frozen_points'] if r['name']=='tau_FPR_0.01')
                points.append(result[f'{metric}_difference_from_nominal']*100)
                limits.append(np.asarray(result[f'{metric}_difference_95pct'])*100)
            x = np.arange(2)+(i-(len(t4_entries)-1)/2)*.12
            limits = np.asarray(limits)
            axis.vlines(x, limits[:, 0], limits[:, 1], color=colors[entry['family']], linewidth=1.5)
            axis.scatter(x, points, color=colors[entry['family']], s=28, label=entry['family'])
        axis.axhline(0., color='black', linestyle=':', linewidth=.8)
        axis.set(title=title, xticks=[0, 1], xticklabels=['Noise', 'Combined'],
                 ylabel='Change from paired nominal (pp)')
        axis.grid(axis='y', alpha=.2)
        axis.legend(fontsize=8)
    figure.suptitle('Interim R1: 1,600 paired development Tune events, seed 42\n'
        '400 physical-block resamples; intervals conditional on trained models; tau cut from full nominal Tune (1% FPR)',
        fontsize=12)
    output = root/'figures'
    output.mkdir(exist_ok=True)
    for extension in ['pdf', 'png']:
        figure.savefig(output/f'r1_interim_stress.{extension}', dpi=170, bbox_inches='tight')
    plt.close(figure)


if __name__ == '__main__':
    main()
