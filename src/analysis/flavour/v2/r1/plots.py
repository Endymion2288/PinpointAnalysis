"""English figures for real optimization checks and later R1 comparisons."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from analysis.flavour.v2.r1.evaluate import probability_columns
from analysis.flavour.v2.r1.metrics import three_from_four, three_targets


LABELS = ['Electron CC', 'Muon CC', 'Tau CC', 'NC']
COLORS = dict(zip(['B0', 'B1', 'B2', 'B3', 'B4', 'B4-MAE'], plt.get_cmap('tab10').colors[:6]))


def save(figure, root, name):
    (root/'figures').mkdir(exist_ok=True)
    for suffix in ['pdf', 'png']:
        figure.savefig(root/'figures'/f'{name}.{suffix}', dpi=170)
    plt.close(figure)


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
    table = table[table.kind.eq('main')]
    config = json.loads((root/'evaluation_config.json').read_text())
    entries = [e for e in config['models'] if e.get('kind', 'main') == 'main' and e['seed'] == 42]
    frames = {e['name']: pd.read_parquet(root/e['predictions']) for e in entries}
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    families = list(COLORS)
    for axis, domain, title in zip(axes, ['all', 'pixel_centered', 'large_scintillator'],
                                   ['All accepted events', 'Pixel-centered domain', 'Large scintillator domain']):
        for task, marker, shift in [('T3', 'o', -.12), ('T4', 's', .12)]:
            for family in families:
                group = table[table.domain.eq(domain) & table.task.eq(task) & table.family.eq(family)]
                if group.empty:
                    continue
                axis.errorbar(families.index(family)+shift, group.BA3.mean(),
                    yerr=group.BA3.std(ddof=1) if len(group)>1 else 0., fmt=marker,
                    color=COLORS[family], capsize=3, label=task if family == 'B0' else None)
        axis.set(xticks=range(len(families)), xticklabels=families, ylabel='Common non-tau balanced accuracy',
                 title=title)
        axis.tick_params(axis='x', rotation=35)
        axis.grid(alpha=.2)
    axes[0].legend(title='Training task')
    figure.suptitle('Development Tune; seed mean and sample SD (single seed: no error bar)')
    save(figure, root, 'r1_common_three')

    for task in ['T3', 'T4']:
        figure, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
        for entry in [e for e in entries if e['task'] == task]:
            frame = frames[entry['name']]
            non_tau = frame.label.ne(2).to_numpy()
            p = frame[probability_columns(frame)].to_numpy()[non_tau]
            if p.shape[1] == 4:
                p = three_from_four(p)
            y = three_targets(frame.label.to_numpy()[non_tau])
            for c, label in enumerate([LABELS[0], LABELS[1], LABELS[3]]):
                fpr, tpr, _ = roc_curve(y==c, p[:, c])
                precision, recall, _ = precision_recall_curve(y==c, p[:, c])
                color = COLORS[entry['family']]
                axes[0, c].plot(fpr, tpr, color=color, label=entry['family'])
                axes[1, c].plot(recall, precision, color=color)
                axes[0, c].set(xscale='log', xlim=(1e-4, 1), ylim=(0, 1),
                    xlabel='Background acceptance', ylabel='Signal efficiency', title=label)
                axes[1, c].set(xlim=(0, 1), ylim=(0, 1), xlabel='Recall', ylabel='Benchmark precision')
                axes[0, c].grid(alpha=.2)
                axes[1, c].grid(alpha=.2)
        axes[0, 0].legend(fontsize=8)
        figure.suptitle(f'{task} training: common non-tau ROC and PR, seed 42, uniform three-class prior')
        save(figure, root, f'r1_{task.lower()}_roc_pr')

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for entry in [e for e in entries if e['task'] == 'T4']:
        frame = frames[entry['name']]
        score = frame.p2.to_numpy()
        fpr, tpr, _ = roc_curve(frame.label.eq(2), score)
        precision, recall, _ = precision_recall_curve(frame.label.eq(2), score)
        axes[0].plot(fpr, tpr, label=entry['family'], color=COLORS[entry['family']])
        axes[1].plot(recall, precision, color=COLORS[entry['family']])
    axes[0].set(xscale='log', xlim=(1e-4, 1), ylim=(0, 1), xlabel='Non-tau background acceptance',
                ylabel='Inclusive tau efficiency', title='Tau ROC')
    for fpr in [.001, .01]:
        axes[0].axvline(fpr, color='gray', linestyle=':', linewidth=.8)
    axes[1].set(xlim=(0, 1), ylim=(0, 1), xlabel='Inclusive tau recall', ylabel='Benchmark precision', title='Tau PR')
    axes[0].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=.2)
    figure.suptitle('Development Tune, seed 42, uniform four-class prior')
    save(figure, root, 'r1_tau_roc_pr')

    ranks = json.loads((root/'architecture_ranking.json').read_text())
    best = ranks['ranking'][0]['family'].lower()
    figure, axes = plt.subplots(1, 4, figsize=(14, 3.5), constrained_layout=True)
    frame = frames[f'final_{best}_t3_s42']
    for axis, c in zip(axes[:3], [0, 1, 3]):
        for label in range(4):
            values = frame.loc[frame.label.eq(label), f'p{c}']
            axis.hist(values, bins=35, range=(0, 1), density=True, histtype='step', label=LABELS[label])
        axis.set(title=f'T3 score: {LABELS[c]}', xlabel='Raw conditional probability', ylabel='Density', yscale='log')
    frame = frames[f'final_{best}_t4_s42']
    for label in range(4):
        axes[3].hist(frame.loc[frame.label.eq(label), 'p2'], bins=35, range=(0, 1), density=True,
                     histtype='step', label=LABELS[label])
    axes[3].set(title='T4 score: Tau CC', xlabel='Raw tau probability', ylabel='Density', yscale='log')
    axes[0].legend(fontsize=8)
    figure.suptitle(f'{best.upper()}: uncalibrated score distributions by true process, seed 42')
    save(figure, root, 'r1_raw_scores')

    costs = []
    figure, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for entry in entries:
        manifest = json.loads((root/'experiments'/entry['name']/'manifest.json').read_text())
        wall = manifest['wall_seconds']
        ba = manifest['metrics']['T3']['BA']
        device = manifest.get('GPU', manifest.get('device', 'unknown'))
        axis.scatter(wall, ba, marker='o' if entry['task']=='T3' else 's', color=COLORS[entry['family']],
            label=f"{entry['family']} {entry['task']}: {device}")
        costs.append({'name': entry['name'], 'device': device, 'wall_seconds': wall, 'BA3': ba,
            'parameters': manifest.get('parameters'), 'peak_MiB': manifest.get('GPU_peak_MiB'),
            'unique_train_events': manifest['unique_train_events']})
    axis.set(xscale='log', xlabel='Operation wall time (s; heterogeneous devices)',
        ylabel='Common non-tau balanced accuracy',
        title='Supervised operation cost versus quality, seed 42\nHistorical MAE pretraining cost is reported separately')
    axis.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=7)
    axis.grid(alpha=.2)
    save(figure, root, 'r1_resource_pareto')
    pd.DataFrame(costs).to_csv(root/'figures/r1_resource_pareto.csv', index=False)

    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for entry in [e for e in entries if e['family'] not in ['B0', 'B4-MAE']]:
        record = json.loads((root/'experiments'/entry['name']/'manifest.json').read_text())
        column = int(entry['task'] == 'T4')
        history = record['history']
        x = [sum(r['exposure_by_class'][c] for c in [0, 1, 3]) for r in history]
        axes[0, column].plot(x, [r['train_NLL'] for r in history], 'o-', markersize=3,
            label=entry['family'], color=COLORS[entry['family']])
        axes[1, column].plot(x, [r['Tune_screen']['T3']['BA'] for r in history], 'o-', markersize=3,
            color=COLORS[entry['family']])
    for column, task in enumerate(['T3', 'T4']):
        axes[0, column].set(title=f'{task} training', ylabel=f'{task} training NLL')
        axes[1, column].set(xlabel='Cumulative non-tau Train exposures', ylabel='Fixed Tune screen BA3')
        axes[0, column].legend(fontsize=8)
        for axis in axes[:, column]:
            axis.grid(alpha=.2)
    figure.suptitle('Four-pass development learning curves, seed 42')
    save(figure, root, 'r1_learning_curves')

    stress = json.loads((root/'evaluation/stress_results.json').read_text())['models']
    figure, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for entry in entries:
        conditions = stress[entry['name']]['conditions']
        column = int(entry['task']=='T4')
        for axis, names in [(axes[0, column], ['noise', 'combined']),
                            (axes[1, column], ['remove_pixel', 'remove_tracker'])]:
            values = [conditions[n]['paired_descriptive_differences']['BA3'] for n in names]
            point = np.array([v['difference_a_minus_b'] for v in values])
            intervals = np.array([v['95pct'] for v in values])
            offset = (list(COLORS).index(entry['family'])-2.5)*.07
            # Percentile intervals can exclude their point in small samples;
            # draw interval segments directly instead of forcing positive yerr.
            x = np.arange(len(names))+offset
            axis.vlines(x, intervals[:, 0], intervals[:, 1], color=COLORS[entry['family']], linewidth=1.)
            axis.scatter(x, point, color=COLORS[entry['family']], s=15, label=entry['family'])
            axis.set(xticks=range(len(names)), xticklabels=[n.replace('_', ' ') for n in names],
                ylabel='Change in common non-tau BA3', title=f"{entry['task']}: {'condition shift' if axis is axes[0, column] else 'input removal'}")
        if entry['task']=='T4':
            for row, metric in [(0, 'efficiency'), (1, 'actual_FPR')]:
                values = [next(p for p in conditions[n]['frozen_points'] if p['name']=='tau_FPR_0.01')
                          for n in ['noise', 'combined']]
                points = np.array([v[f'{metric}_difference_from_nominal'] for v in values])
                limits = np.array([v[f'{metric}_difference_95pct'] for v in values])
                x = np.arange(2)+(list(COLORS).index(entry['family'])-2.5)*.07
                axes[row, 2].vlines(x, limits[:, 0], limits[:, 1], color=COLORS[entry['family']])
                axes[row, 2].scatter(x, points, color=COLORS[entry['family']], s=15)
                axes[row, 2].set(xticks=[0, 1], xticklabels=['noise', 'combined'],
                    ylabel='Change in tau efficiency' if row==0 else 'Change in actual background acceptance',
                    title='Frozen nominal tau cut (1% full Tune FPR)')
    axes[0, 0].legend(fontsize=7, ncol=2)
    for axis in axes.flat:
        axis.axhline(0., color='black', linestyle=':', linewidth=.8)
        axis.grid(alpha=.15)
    figure.suptitle('1,600 paired Tune events; 95% physical-block intervals; accepted condition proxies')
    save(figure, root, 'r1_stress')


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
