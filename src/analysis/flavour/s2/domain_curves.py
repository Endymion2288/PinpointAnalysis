"""Read conditional learning curves from completed S2 epoch probabilities."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.flavour.inventory import write_json
from analysis.flavour.s2.train import metrics


def summarize(root, names):
    experiments = []
    for name in names:
        path = root / 'experiments' / name
        manifest = json.loads((path / 'manifest.json').read_text())
        checks = json.loads((path / 'final_checkpoint_checks.json').read_text())
        if manifest['status'] != 'complete' or checks['status'] != 'pass':
            raise ValueError(f'{name}: completed output readback required')
        frame = pd.read_parquet(path / 'validation_predictions.parquet')
        epochs = pd.read_parquet(path / 'epoch_predictions.parquet')
        history = json.loads((path / 'history.json').read_text())
        rows = []
        masks = {'all': np.ones(len(frame), dtype=bool),
                 'pixel_centered': frame.pixel_centered.to_numpy(dtype=bool),
                 'large_scintillator': frame.large_scintillator.to_numpy(dtype=bool)}
        for epoch, predictions in epochs.groupby('epoch', sort=True):
            predictions = predictions.sort_values('row')
            if not np.array_equal(predictions.row, np.arange(len(frame))):
                raise ValueError(f'{name}: epoch {epoch} validation indices differ')
            probabilities = predictions[[f'p{c}' for c in range(4)]].to_numpy()
            for domain, mask in masks.items():
                labels = frame.loc[mask, 'label'].to_numpy()
                measured = metrics(labels, probabilities[mask])
                if domain == 'all':
                    recorded = history[int(epoch) - 1]['validation']
                    if not (np.isclose(measured['NLL'], recorded['NLL'], rtol=0, atol=1e-8)
                            and np.isclose(measured['classwise']['2']['PR_AUC'],
                                recorded['classwise']['2']['PR_AUC'], rtol=0, atol=1e-10)):
                        raise ValueError(f'{name}: full-domain epoch metrics differ from history')
                rows.append({'epoch': int(epoch), 'domain': domain,
                             'class_counts': np.bincount(labels, minlength=4).tolist(),
                             'NLL': measured['NLL'],
                             'tau_AP': measured['classwise']['2']['PR_AUC'],
                             'balanced_accuracy': measured['balanced_accuracy']})
        preferred = {domain: min((r for r in rows if r['domain'] == domain),
                                key=lambda r: r['NLL'])['epoch'] for domain in masks}
        if preferred['all'] != manifest['best_epoch']:
            raise ValueError(f'{name}: frozen full-domain checkpoint selection differs')
        experiments.append({'name': name, 'train_domain': manifest['config']['train_domain'],
                            'train_events': manifest['train_events'],
                            'train_class_counts': manifest['train_class_counts'],
                            'primary_epoch': manifest['best_epoch'],
                            'posthoc_domain_minimum_NLL_epochs': preferred, 'curves': rows})
    return {
        'utc': datetime.now(timezone.utc).isoformat(),
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'experiments': experiments,
        'scope': 'posthoc conditional validation learning curves; no new model fit',
        'primary_selection': 'unchanged full-validation NLL-selected checkpoint',
        'domain_minimum_epochs': 'diagnostic only; not substituted into main or paired comparisons',
        'uncertainty': 'epoch/domain point estimates; no intervals or across-domain significance claim',
        'calibration_test_used': False,
    }


def plot_curves(report, output):
    experiments = report['experiments']
    fig, axes = plt.subplots(len(experiments), 2, squeeze=False,
                             figsize=(10, 3.2 * len(experiments)), constrained_layout=True)
    labels = {'all': 'Full', 'pixel_centered': 'Pixel-centered',
              'large_scintillator': 'Large scintillator'}
    for row, experiment in zip(axes, experiments):
        for axis, key in zip(row, ['NLL', 'tau_AP']):
            for domain, label in labels.items():
                points = [p for p in experiment['curves'] if p['domain'] == domain]
                axis.plot([p['epoch'] for p in points], [p[key] for p in points],
                          marker='o', markersize=3, label=label)
            axis.axvline(experiment['primary_epoch'], color='grey', linestyle=':')
            axis.set(xlabel='Epoch', ylabel='Validation NLL' if key == 'NLL' else 'Tau AP',
                     title=experiment['name'], xticks=sorted({p['epoch'] for p in experiment['curves']}))
            axis.legend(fontsize=7)
            axis.grid(alpha=.2)
    fig.suptitle('S2 conditional learning curves; sample domain priors\n'
                 'Dotted line: frozen full-validation NLL selection; point estimates', fontsize=10)
    for suffix in ['png', 'pdf']:
        fig.savefig(output.with_suffix(f'.{suffix}'), dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--experiments', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.root, args.experiments)
    write_json(args.output, report)
    plot_curves(report, args.output)
    print(json.dumps([{'name': item['name'], 'primary_epoch': item['primary_epoch'],
                      'posthoc_domain_minimum_NLL_epochs': item['posthoc_domain_minimum_NLL_epochs']}
                     for item in report['experiments']]))


if __name__ == '__main__':
    main()
