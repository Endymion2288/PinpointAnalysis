"""Compact four-class seed spread and paired physical-event interval figures."""
import argparse
import json
from pathlib import Path

import numpy as np


def label(name):
    return (name.replace('_s42_e4', '').replace('s3r_', '').replace('_', ' ')
            .replace(' minus fresh ParT three seed', ' vs fresh ParT')
            .replace(' three seed', ''))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.evaluation.read_text())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    groups = [g for g in data['training_seed_spread'] if g['three_seed_confirmation_complete']]
    by_name = {e['name']: e for e in data['experiments']}
    metric_labels = {'macro_AP': 'Macro AP', 'NLL': 'NLL', 'balanced_accuracy': 'Balanced accuracy'}
    with PdfPages(args.output / 'repaired_model_overview.pdf') as pdf:
        fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
        for axis, metric in zip(axes, ['macro_AP', 'NLL', 'balanced_accuracy']):
            for i, group in enumerate(groups):
                values = [by_name[name]['four_class'][metric] for name in group['experiments']]
                axis.scatter(values, np.full(len(values), i), s=20, color=f'C{i}')
                axis.errorbar(np.mean(values), i, xerr=np.std(values, ddof=1), fmt='k|', capsize=4)
            axis.set(yticks=range(len(groups)), yticklabels=[label(g['experiments'][0].replace('_s17_', '_s42_')) for g in groups],
                     xlabel=metric_labels[metric], title=metric_labels[metric])
            axis.grid(axis='x', alpha=.25)
        fig.suptitle('Repaired S3: complete validation sample, three separately trained seeds\n'
                     'Dots are seed fits; bars are one sample SD, not physical-event confidence intervals', fontsize=11)
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
    comparisons = data['seed_mean_comparisons']
    from analysis.flavour.s3.metrics import CLASSES
    metric_groups = [
        ('repaired_paired_intervals.pdf', ['macro_AP', 'NLL', 'macro_F1', 'balanced_accuracy'],
         ['Macro AP', 'NLL', 'Macro F1', 'Balanced accuracy']),
        ('repaired_class_intervals.pdf', [f'class_{c}_AP' for c in range(4)],
         [f'{name} AP' for name in CLASSES]),
    ]
    for filename, metrics, titles in metric_groups:
        with PdfPages(args.output / filename) as pdf:
            for scope in ['raw', 'common_support']:
                fig, axes = plt.subplots(1, 4, figsize=(18, 6), constrained_layout=True)
                y = np.arange(len(comparisons))
                for axis, metric, title in zip(axes, metrics, titles):
                    for i, row in enumerate(comparisons):
                        measured = row['metrics'][scope][metric]
                        delta = measured['delta']; low, high = measured['CI95']
                        axis.plot([low, high], [i, i], color=f'C{i%10}')
                        axis.scatter([delta], [i], color=f'C{i%10}', s=20)
                    axis.axvline(0, color='gray', linestyle='--')
                    axis.set(yticks=y, yticklabels=[label(c['name']) for c in comparisons] if axis is axes[0] else [],
                             xlabel=f'Delta {title} (B - A)', title=title)
                    axis.grid(axis='x', alpha=.25)
                fig.suptitle(f'Repaired S3, {scope}: prespecified three-seed mean contrasts\n'
                             '95% paired physical-block intervals, 400 replicates; conditional on fitted seeds', fontsize=11)
                pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


if __name__ == '__main__':
    main()
