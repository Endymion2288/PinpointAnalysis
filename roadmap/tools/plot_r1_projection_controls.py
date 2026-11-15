"""Plot the completed R1 projection controls without recomputing statistics."""

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
    folder = args.root/'interim/projection_controls'
    result = json.loads((folder/'evaluation/results.json').read_text())
    domains = ['all', 'pixel_centered', 'large_scintillator']
    full = 'final_b1_t3_s42'
    b4 = 'final_b4_t3_s42'
    hgb = 'screen_b0_t3_lr0.05_s42'
    without = 'final_b1_t3_s42_without_tracker_fp32'
    cnn = 'control_b1_cnn_t3_s42'

    def pair(a, b, domain='all'):
        return next(row['metrics'] for row in result['pairs']
                    if row['a']==a and row['b']==b and row['domain']==domain)

    def interval(axis, row, y, color, label=None):
        point = row['difference_a_minus_b']*100
        low, high = np.asarray(row['95pct'])*100
        axis.hlines(y, low, high, color=color, linewidth=1.5)
        axis.plot(point, y, 'o', color=color, markersize=4, label=label)

    figure, axes = plt.subplots(2, 2, figsize=(13.8, 9.2), constrained_layout=True)
    for name, label in [(full, 'B1 ResNet'), (hgb, 'HGB summary'),
                        (b4, 'B4 native'), (without, 'B1 without tracker'),
                        (cnn, 'B1 plain CNN')]:
        values = [result['models'][name][domain]['T3']['BA']*100 for domain in domains]
        axes[0, 0].plot(range(3), values, 'o-', label=label)
    axes[0, 0].set(xticks=range(3),
                   xticklabels=['All\n30,000 non-tau', 'Pixel-centered\n3,008 non-tau',
                                'Large\n13,866 non-tau'],
                   ylabel='Common non-tau balanced accuracy (%)',
                   title='T3 training: performance depends on reporting domain')
    axes[0, 0].legend(fontsize=8, ncol=2)

    comparisons = [(full, hgb, 'B1 T3 - HGB T3'), (full, b4, 'B1 T3 - B4 T3'),
                   (full, 'final_b1_t4_s42', 'B1 T3 - B1 conditional T4'),
                   (full, without, 'B1 full - without tracker'),
                   (cnn, full, 'CNN T3 - ResNet T3'),
                   ('control_b1_cnn_t4_s42', 'final_b1_t4_s42', 'CNN T4 - ResNet T4')]
    for i, (a, b, _) in enumerate(comparisons):
        interval(axes[0, 1], pair(a, b)['BA3'], i, 'tab:blue')
    axes[0, 1].set(yticks=range(len(comparisons)),
                   yticklabels=[row[2] for row in comparisons],
                   xlabel='Balanced accuracy difference (percentage points)',
                   title='All-domain paired differences and 95% intervals')
    axes[0, 1].invert_yaxis()

    contrasts = [(hgb, 'B1 - HGB', 'tab:blue'), (b4, 'B1 - B4', 'tab:orange'),
                 (without, 'Full - without tracker', 'tab:green')]
    for j, (other, label, color) in enumerate(contrasts):
        for i, metric in enumerate(['recall_e', 'recall_mu', 'recall_NC']):
            interval(axes[1, 0], pair(full, other)[metric], i+(j-1)*.18,
                     color, label if i==0 else None)
    axes[1, 0].set(yticks=range(3), yticklabels=['Electron', 'Muon', 'NC'],
                   xlabel='Recall difference (percentage points)',
                   title='All-domain class tradeoffs')
    axes[1, 0].invert_yaxis()
    axes[1, 0].legend(fontsize=8, loc='upper right')

    for j, (other, label, color) in enumerate(contrasts):
        for i, domain in enumerate(domains):
            interval(axes[1, 1], pair(full, other, domain)['recall_mu'],
                     i+(j-1)*.18, color, label if i==0 else None)
    axes[1, 1].set(yticks=range(3), yticklabels=['All', 'Pixel-centered', 'Large'],
                   xlabel='Muon recall difference (percentage points)',
                   title='Muon tradeoffs change across reporting domains')
    axes[1, 1].invert_yaxis()
    axes[1, 1].legend(fontsize=8, loc='upper right')

    axes[0, 0].grid(axis='y', alpha=.2)
    for axis in [axes[0, 1], axes[1, 0], axes[1, 1]]:
        axis.axvline(0., color='black', linestyle=':', linewidth=.8)
        axis.grid(axis='x', alpha=.2)
    figure.suptitle('Projection controls: four-pass development Tune; seed 42\n'
                   'Intervals condition on trained models; heterogeneous devices and precision; '
                   'no inference on retraining variance', fontsize=12)
    output = folder/'figures'
    output.mkdir(exist_ok=True)
    for ext in ['pdf', 'png']:
        figure.savefig(output/f'r1_projection_controls.{ext}', dpi=170)
    plt.close(figure)


if __name__=='__main__':
    main()
