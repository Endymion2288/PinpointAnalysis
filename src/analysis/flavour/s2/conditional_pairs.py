"""Compare completed S2 models on identical conditional validation events."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from analysis.flavour.inventory import write_json
from analysis.flavour.s2.evaluate import (
    historical_three_class,
    paired_interval,
    tau_metrics,
)
from analysis.flavour.s2.train import metrics


def compare(root, a, b, domains, replicates):
    """Use existing fiducial flags and predictions; fit no new model or weights."""
    keys = ['run', 'chunk', 'row']
    frames = []
    configurations = []
    for name in [a, b]:
        path = root / 'experiments' / name
        manifest = json.loads((path / 'manifest.json').read_text())
        if manifest['status'] != 'complete':
            raise ValueError(f'{name}: training remains {manifest["status"]}')
        checks = json.loads((path / 'final_checkpoint_checks.json').read_text())
        if checks['status'] != 'pass':
            raise ValueError(f'{name}: completed output readback has not passed')
        frames.append(pd.read_parquet(path / 'validation_predictions.parquet')
                      .sort_values(keys).reset_index(drop=True))
        configurations.append(manifest['config'])
    if not frames[0][keys + ['label']].equals(frames[1][keys + ['label']]):
        raise ValueError('conditional comparison has different validation events')
    results = []
    for domain in domains:
        masks = [frame[domain].to_numpy(dtype=bool) for frame in frames]
        if not (masks[0] == masks[1]).all():
            raise ValueError(f'{domain}: fiducial flags differ between models')
        selected = [frame.loc[mask].reset_index(drop=True)
                    for frame, mask in zip(frames, masks)]
        models = []
        for name, frame in zip([a, b], selected):
            models.append({
                'name': name,
                'class_counts': [int(frame.label.eq(c).sum()) for c in range(4)],
                'tau': tau_metrics(frame),
                'four_class': metrics(frame.label.to_numpy(),
                                      frame[[f'p{c}' for c in range(4)]].to_numpy()),
                'historical_three_class': historical_three_class(frame),
            })
        results.append({
            'domain': domain, 'models': models,
            'paired': paired_interval(*selected, replicates=replicates),
        })
    return {
        'utc': datetime.now(timezone.utc).isoformat(),
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                              text=True).strip(),
        'a': a, 'b': b, 'configurations': configurations, 'results': results,
        'scope': 'posthoc conditional validation comparison on existing fiducial flags; B minus A',
        'selection': 'same full-validation NLL-selected checkpoints as the main S2 evaluation',
        'uncertainty': 'pointwise accepted-family/chunk bootstrap conditional on fitted models; no E/vertex reweighting or simultaneous coverage',
        'working_points': 'each conditional domain uses its own empirical background threshold; not the inclusive threshold',
        'historical_uncertainty': 'three-class conditional point estimates only; no difference intervals',
        'sample_prior': 'observed conditional-domain prior; absolute AP across domains is not a paired contrast',
        'calibration_test_used': False,
    }


def plot_comparison(report, output):
    labels = {'pixel_centered': 'Pixel-centered', 'large_scintillator': 'Large scintillator'}
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(11, 3.8), constrained_layout=True)
    for axis, metric, title in zip(axes, ['tau_AP', 'tau_ROC_AUC', 'tau_eff_at_0.01_FPR'],
                                   ['Tau AP', 'Tau ROC AUC', 'Tau efficiency at 1% FPR']):
        for i, row in enumerate(report['results']):
            effect = row['paired']['other_paired_metrics'][metric]
            low, high = effect['CI95']
            axis.hlines(i, low, high, color='tab:blue')
            axis.plot([low, high], [i, i], '|', color='tab:blue')
            axis.plot(effect['delta'], i, 'o', color='tab:blue')
        axis.axvline(0, color='grey', linewidth=.8)
        if metric == 'tau_AP':
            axis.axvline(.01, color='grey', linestyle=':', linewidth=.8)
        axis.set(title=title, xlabel='Difference (B - A)',
                 yticks=range(len(report['results'])),
                 yticklabels=[labels[r['domain']] for r in report['results']],
                 ylim=(len(report['results']) - .5, -.5))
        axis.grid(axis='x', alpha=.2)
    fig.suptitle(f"S2 conditional validation; A: {report['a']}; B: {report['b']}\n"
                 'Pointwise 95% event intervals; observed domain priors; no calibration or test',
                 fontsize=9)
    for suffix in ['png', 'pdf']:
        fig.savefig(output.with_suffix(f'.{suffix}'), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--a', required=True)
    parser.add_argument('--b', required=True)
    parser.add_argument('--domains', nargs='+', choices=['pixel_centered', 'large_scintillator'],
                        default=['pixel_centered', 'large_scintillator'])
    parser.add_argument('--bootstrap', type=int, default=400)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.root, args.a, args.b, args.domains, args.bootstrap)
    write_json(args.output, report)
    plot_comparison(report, args.output)
    print(json.dumps({'a': args.a, 'b': args.b, 'domains': [
        {'domain': row['domain'], 'events': row['paired']['events'],
         'delta': row['paired']['delta'], 'CI95': row['paired']['CI95']}
        for row in report['results']]}))


if __name__ == '__main__':
    main()
