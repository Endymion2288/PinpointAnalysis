"""Plot the measured seed-42 B4 contrasts without waiting for the full R1 ranking."""

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
    args = parser.parse_args()
    root = args.root/'interim/b4_seed42'
    results = json.loads((root/'evaluation/results.json').read_text())
    domains = ['all', 'pixel_centered', 'large_scintillator']
    titles = ['T3 training minus conditional T4', 'Tracker incremental value (retrained)', 'B4 minus summary HGB']
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    for panel, axis in enumerate(axes):
        for index, domain in enumerate(domains):
            for pair in results['pairs']:
                if pair['domain'] != domain:
                    continue
                if panel==0 and pair['reason']!='same non-tau unique rows and four exposures':
                    continue
                if panel==1 and pair['reason']!='retrained tracker incremental value':
                    continue
                if panel==2 and pair['reason']!='observation-summary baseline':
                    continue
                t4 = '_t4_' in pair['a']
                color = 'tab:orange' if panel==2 and t4 else 'tab:blue'
                y = index+(.10 if t4 else -.10) if panel==2 else index
                metric = pair['metrics']['BA3']
                point = metric['difference_a_minus_b']*100
                limits = np.asarray(metric['95pct'])*100
                axis.hlines(y, *limits, color=color, linewidth=1.6)
                axis.scatter(point, y, s=25, color=color, marker='s' if t4 else 'o',
                    label=('T4' if t4 else 'T3') if panel==2 and index==0 else None)
        axis.axvline(0., color='black', linestyle=':', linewidth=1.)
        axis.set(title=titles[panel], yticks=range(3),
            yticklabels=['All accepted', 'Pixel-centered', 'Large scintillator'],
            ylim=(2.5, -.5))
        axis.grid(axis='x', alpha=.2)
    axes[2].legend(title='Training task', fontsize=8)
    figure.supxlabel('Change in common three-class balanced accuracy (percentage points)', fontsize=11)
    figure.suptitle('Interim B4: seed 42, four passes, development Tune\n400 paired physical-block resamples; intervals conditional on this trained seed')
    output = root/'figures'
    output.mkdir(exist_ok=True)
    for suffix in ['pdf', 'png']:
        figure.savefig(output/f'r1_interim_b4.{suffix}', dpi=170, bbox_inches='tight')
    plt.close(figure)


if __name__ == '__main__':
    main()
