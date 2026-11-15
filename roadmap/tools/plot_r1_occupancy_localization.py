"""Plot paired recall differences and their additive occupancy contributions."""

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
    parser.add_argument('--label-a', default='B1 T3')
    parser.add_argument('--label-b', default='B4 T3')
    args = parser.parse_args()
    result = json.loads((args.root/'evaluation/occupancy_localization.json').read_text())
    rows = result['strata']
    labels = ['0-99', '100-999', '1,000-4,999', '5,000+']
    colors = {'e': 'tab:blue', 'mu': 'tab:orange', 'NC': 'tab:green'}
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for key, color in colors.items():
        for j, row in enumerate(rows):
            value = row['classes'][key]['recall_difference']
            x = j+(['e', 'mu', 'NC'].index(key)-1)*.14
            axes[0].vlines(x, *np.asarray(value['95pct'])*100, color=color)
            axes[0].scatter(x, value['difference_a_minus_b']*100, color=color,
                            label=key if j==0 else None, s=25)
    axes[0].set(ylabel='Within-stratum recall difference (pp)', title='Class tradeoffs by accepted occupancy')
    axes[0].legend(title='True class', loc='upper center')
    for j, row in enumerate(rows):
        value = row['classes']['mu']['contribution_to_all_class_recall']
        axes[1].vlines(j, *np.asarray(value['95pct'])*100, color=colors['mu'], linewidth=1.5)
        axes[1].scatter(j, value['difference_a_minus_b']*100, color=colors['mu'], s=30)
    axes[1].set(ylabel='Contribution to all-domain muon recall (pp)',
                title=f"Contributions sum to {result['all_class_recall_differences']['mu']*100:+.2f} pp")
    for name, shift, label, color in [('a_only_correct', -.18, args.label_a+' only correct', 'tab:red'),
                                     ('b_only_correct', .18, args.label_b+' only correct', 'tab:blue')]:
        axes[2].bar(np.arange(4)+shift, [r['classes']['mu'][name] for r in rows],
                    width=.36, label=label, color=color)
    axes[2].set(ylabel='Muon event count', title='Paired disagreements, same muon events')
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.set(xticks=range(4), xticklabels=labels, xlabel='Accepted channel count before any removal')
        axis.grid(axis='y', alpha=.2)
    for axis in axes[:2]:
        axis.axhline(0., color='black', linestyle=':', linewidth=.8)
    figure.suptitle(f'{args.label_a} minus {args.label_b}, seed 42: post-hoc development Tune localization\n'
                   f"{result['replicates']} paired physical-block draws; occupancy correlates with geometry and acceptance", fontsize=12)
    output = args.root/'figures'
    output.mkdir(exist_ok=True)
    for extension in ['pdf', 'png']:
        figure.savefig(output/f'r1_occupancy_localization.{extension}', dpi=170)
    plt.close(figure)


if __name__=='__main__':
    main()
