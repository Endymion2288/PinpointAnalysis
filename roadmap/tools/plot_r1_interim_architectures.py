"""Plot completed seed42 T4 contrasts and the observed muon-domain tradeoff."""

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
    args = parser.parse_args()
    root = args.root/'interim/architectures_seed42'
    result = json.loads((root/'evaluation/results.json').read_text())
    pairs = [p for p in result['pairs'] if p['domain']=='all']
    labels = ['ParticleNet', 'Geometry ParT', 'Scratch', 'MAE init.', 'HGB']
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.7), constrained_layout=True)
    for axis, metric, title in zip(axes, ['BA3','recall_mu','tau_eff_1pct'],
            ['Common non-tau balanced accuracy', 'Common non-tau muon recall', 'Tau efficiency at 1% FPR']):
        for i, pair in enumerate(pairs):
            row = pair['metrics'][metric]
            axis.hlines(i, *np.asarray(row['95pct'])*100, color='tab:blue')
            axis.scatter(row['difference_a_minus_b']*100, i, color='tab:blue', s=26)
        axis.axvline(0., color='black', linestyle=':', linewidth=1.)
        axis.set(yticks=range(len(labels)), yticklabels=labels, xlabel='ResNet minus reference (percentage points)',
                 title=title, ylim=(len(labels)-.5, -.5))
        axis.grid(axis='x', alpha=.2)
    figure.suptitle('Completed T4 models, seed 42: 400 paired physical-block intervals\nDevelopment Tune 40k; actual heterogeneous runtime recipes; descriptive comparisons')
    output = root/'figures'
    output.mkdir(exist_ok=True)
    for ext in ['pdf','png']:
        figure.savefig(output/f'r1_interim_architectures.{ext}', dpi=170)
    plt.close(figure)

    a, b = [result['models'][f'final_{family}_t4_s42'] for family in ['b1','b4']]
    domains = ['all','large_scintillator','tracker_active','tracker_empty','occupancy_0_100']
    labels = ['All', 'Large scintillator', 'Tracker active', 'Tracker empty', 'Occupancy < 100']
    values = [[report[d]['T3']['per_class'][1]['recall']*100 for d in domains] for report in [a,b]]
    support = [a[d]['raw_counts_original_labels'][1] for d in domains]
    for i, report in enumerate([a,b]):
        full, large = [report[d]['T3_observed_mixture'] for d in ['all','large_scintillator']]
        denominator = full['raw_counts'][1]-large['raw_counts'][1]
        numerator = full['confusion_counts'][1][1]-large['confusion_counts'][1][1]
        values[i].append(100*numerator/denominator)
        if i==0:
            support.append(denominator)
    labels.append('Outside large scintillator')
    figure, axis = plt.subplots(figsize=(11, 4), constrained_layout=True)
    x = np.arange(len(labels))
    for i, name in enumerate(['ResNet','Scratch']):
        axis.bar(x+(i-.5)*.32, values[i], width=.32, label=name)
    axis.set(xticks=x, xticklabels=[f'{label}\n(n_mu={n:,})' for label,n in zip(labels,support)],
        ylabel='Conditional three-class muon recall (%)', ylim=(0,100),
        title='Observed domain tradeoff: fixed seed42 T4 models\nPoint estimates, overlapping domains; no domain confidence intervals')
    axis.tick_params(axis='x', labelsize=8)
    axis.legend()
    axis.grid(axis='y', alpha=.2)
    for ext in ['pdf','png']:
        figure.savefig(output/f'r1_interim_muon_domains.{ext}', dpi=170)
    plt.close(figure)


if __name__=='__main__':
    main()
