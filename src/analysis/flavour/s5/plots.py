"""Standalone English figures, native-response examples and selection denominators."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from analysis.flavour.datasets import CachedNativeDataset


def save(fig, output, name):
    fig.savefig(output/(name+'.pdf'), bbox_inches='tight')
    fig.savefig(output/(name+'.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)


def native_examples(root, output, entry):
    name, frame, manifest = entry
    score = frame[[f'p{c}' for c in range(4)]].to_numpy()
    prediction = score.argmax(1)
    correct = prediction == frame.label.to_numpy()
    rows = []
    for visible in [False, True]:
        for is_correct in [False, True]:
            mask = frame.label.eq(2).to_numpy() & frame.tau_visible_any.eq(visible).to_numpy() & (correct == is_correct)
            ids = np.flatnonzero(mask)
            if len(ids):
                index = ids[np.argmax(score[ids].max(1))]
                rows.append((index, f'Tau: visible={visible}, correct={is_correct}', len(ids)))
    for label in [0, 1]:
        ids = np.flatnonzero(frame.label.eq(label).to_numpy() & frame.charm_presence.to_numpy())
        if len(ids):
            rows.append((ids[np.argmax(frame.p2.to_numpy()[ids])], 'Charm background, highest tau score', len(ids)))
    for index in np.random.default_rng(20261114).choice(len(frame), 4, replace=False):
        rows.append((index, 'Fixed random reference', len(frame)))
    paths = sorted(p for p in (root/'shards').iterdir() if (p/'observations.parquet').exists())
    native = CachedNativeDataset(paths, max_shards=2)
    metadata = []
    for i, path in enumerate(paths):
        part = pd.read_parquet(path/'provenance.parquet', columns=['run', 'chunk', 'row'])
        part['native_index'] = np.arange(len(part))+native.boundaries[i]
        metadata.append(part)
    locations = pd.concat(metadata).set_index(['run', 'chunk', 'row']).native_index
    selected = []
    with PdfPages(output/'native_event_examples.pdf') as pdf:
        for index, group, denominator in rows:
            row = frame.iloc[index]
            native_index = int(locations.loc[tuple(row[k] for k in ['run', 'chunk', 'row'])])
            observation = native[native_index]['observations']
            xyz = np.asarray(observation['pixel']['xyz_mm']).reshape(-1, 3)
            bars = np.asarray(observation['bar']['bounds_mm']).reshape(-1, 5)
            tracker = observation['tracker_hypothesis']
            xy = np.asarray(tracker['xy_mm']).reshape(-1, 2)
            fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
            for coordinate in range(2):
                axis = axes[coordinate, 0]
                if len(xyz):
                    axis.scatter(xyz[:, 2], xyz[:, coordinate], s=.6, alpha=.5, rasterized=True, label='Binary pixel')
                truth_vertex = np.asarray(row.vertex_mm)
                axis.scatter(truth_vertex[2], truth_vertex[coordinate], marker='*', s=65, label='Truth primary')
                if row.tau_decay_valid:
                    tau_vertex = np.asarray(row.tau_decay_vertex_mm)
                    axis.scatter(tau_vertex[2], tau_vertex[coordinate], marker='D', s=30, label='Truth tau decay')
                if 'vertex' in manifest['config'].get('auxiliaries', []):
                    v = row[[f'pred_vertex_{c}' for c in range(3)]].to_numpy(dtype=float)*[210, 210, 3000]
                    axis.scatter(v[2], v[coordinate], marker='x', s=40, label='Predicted primary')
                axis.set(xlabel='z [mm]', ylabel='xy'[coordinate]+' [mm]', title='Native pixel response; separate truth/prediction markers')
                axis.legend(fontsize=6)
                axis = axes[coordinate, 1]
                if len(bars):
                    segments = np.stack([np.column_stack([bars[:, 4], bars[:, 2*coordinate]]), np.column_stack([bars[:, 4], bars[:, 2*coordinate+1]])], 1)
                    axis.add_collection(LineCollection(segments, array=np.log1p(observation['bar']['pe_proxy']), cmap='viridis', linewidths=.7, rasterized=True))
                    axis.autoscale()
                axis.set(xlabel='z [mm]', ylabel='xy'[coordinate]+' interval [mm]', title='Scintillator readout extents; color log(1+PE)')
            if len(xy):
                for station in np.unique(tracker['station']):
                    mask = tracker['station'] == station
                    axes[0, 2].scatter(xy[mask, 0], xy[mask, 1], s=6, label=f'Station {station}')
                axes[0, 2].legend(fontsize=7)
            axes[0, 2].set(xlabel='x [mm]', ylabel='y [mm]', title='Accepted tracker proxy')
            axes[1, 2].bar(['e', 'mu', 'tau', 'NC'], score[index])
            axes[1, 2].set(ylim=(0, 1), ylabel='Predicted probability', title='Four-way scores')
            fig.suptitle(f'{name}: {group}; group N={denominator}\nRun {row.run}, chunk {row.chunk}, row {row.row}; '
                         f'true={int(row.label)}, predicted={prediction[index]}; tau branch={row.tau_branch}; charm={bool(row.charm_presence)}', fontsize=10)
            pdf.savefig(fig, dpi=160)
            plt.close(fig)
            selected.append({'run': int(row.run), 'chunk': int(row.chunk), 'row': int(row.row), 'group': group, 'group_events': denominator})
    pd.DataFrame(selected).to_csv(output/'native_examples.csv', index=False)
    native.close()


def plot_results(output, entries, report, root):
    confirmed = [r for r in report['seed_means'] if r['cohort'] == 'confirmation8']
    if confirmed:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        for axis, metric in zip(axes, ['NLL', 'macro_AP', 'class_2_AP']):
            for index, row in enumerate(confirmed):
                item = row['metrics']['raw'][metric]
                axis.plot(item['CI95'], [index, index], color='C0', linewidth=2)
                axis.plot(item['delta'], index, 'o', color='C0')
                deltas = item['per_seed_deltas']
                axis.scatter(deltas, index+np.linspace(-.12, .12, len(deltas)),
                             color='gray', s=15, alpha=.8)
            axis.axvline(0., color='black', linestyle='--', linewidth=.8)
            axis.set(yticks=range(len(confirmed)),
                     yticklabels=[r['variant'].replace('_', ' ') for r in confirmed] if metric == 'NLL' else [],
                     xlabel='Variant minus direct four-class control', title=metric.replace('_', ' '))
            axis.invert_yaxis()
        fig.suptitle('Eight-pass confirmation: mean of three seeds; event-block CI95; gray points are individual seeds')
        save(fig, output, 'three_seed_comparisons')
    passes = sorted({e[2]['config']['epochs'] for e in entries})
    for epochs in passes:
        fig, axis = plt.subplots(figsize=(7, 5))
        for name, frame, manifest in entries:
            config = manifest['config']
            variant = config.get('variant')
            if config.get('seed') != 42 or config['epochs'] != epochs:
                continue
            if epochs == 4 and variant not in ['direct4', 'six', 'combined', 'relative', 'objects', 'object_only', 'moments']:
                continue
            precision, recall, _ = precision_recall_curve(frame.label.eq(2), frame.p2)
            axis.plot(recall, precision, label=variant.replace('_', ' '))
        axis.set(xlabel='Inclusive tau efficiency', ylabel='Tau purity; balanced four-class prior',
                 title=f'S5 validation; neural models: {epochs} passes\n2,500 tau + 7,500 backgrounds; HGB references fitted separately')
        axis.legend(fontsize=8)
        save(fig, output, f'tau_efficiency_purity_passes{epochs}')
    frame = entries[0][1]
    tau = frame[frame.label.eq(2)]
    groups = tau.groupby('tau_branch').agg(events=('label', 'size'), visible=('tau_visible_any', 'sum'), tracker=('tau_visible_tracker', 'sum'))
    fig, axis = plt.subplots(figsize=(7, 4))
    x = np.arange(len(groups))
    axis.bar(x-.2, groups.visible/groups.events, width=.4, label='Any daughter response')
    axis.bar(x+.2, groups.tracker/groups.events, width=.4, label='Direct charged daughter in tracker')
    axis.set(xticks=x, xticklabels=[f'{k}\nN={n}' for k, n in zip(groups.index, groups.events)], ylim=(0, 1), ylabel='Fraction of all tau events in branch', title='Tau visibility; response presence is not resolved decay structure')
    axis.legend(fontsize=8)
    save(fig, output, 'tau_decay_visibility')
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    baselines = [e for e in entries if e[2]['config'].get('variant') == 'direct4' and e[2]['config'].get('seed') == 42]
    baseline = max(baselines, key=lambda e: e[2]['config']['epochs']) if baselines else entries[0]
    f = baseline[1]
    for axis, column, label in zip(axes, ['pixel_channels', 'tracker_channels', 'pixel_tracks'], ['Active pixel channels', 'Tracker measurements', 'Pixel segment candidates']):
        axis.hexbin(np.log1p(f[column]), f.p2, gridsize=35, mincnt=1, bins='log', rasterized=True)
        axis.set(xlabel='log(1 + '+label+')', ylabel='Predicted tau score', title='All 10,000 validation events')
    save(fig, output, 'tau_score_observables')
    fig, axis = plt.subplots(figsize=(7, 4))
    rows = [r for r in report['object_resolution_coverage'] if r['class'] == 'all' and r['selected'] > 0]
    if rows:
        axis.plot([r['coverage'] for r in rows], [r['primary_resolution_mm']['absolute_quantiles_50_68_95'][1][2] for r in rows], marker='o')
    axis.set(xlabel='Vertex candidate coverage of all validation events', ylabel='68% absolute primary z residual [mm]', title='Measured-object vertex quality scan')
    save(fig, output, 'object_vertex_resolution_coverage')
    for epochs in passes:
        candidates = [e for e in entries if e[2]['config']['epochs'] == epochs and e[2]['config'].get('seed') == 42]
        selected = min(candidates, key=lambda e: next(r['metrics']['NLL'] for r in report['experiments'] if r['name'] == e[0]))
        matrix = np.zeros((4, 4), int)
        labels = selected[1].label.to_numpy()
        predicted = selected[1][[f'p{i}' for i in range(4)]].to_numpy().argmax(1)
        np.add.at(matrix, (labels, predicted), 1)
        fig, axis = plt.subplots(figsize=(6, 5))
        axis.imshow(matrix/np.maximum(matrix.sum(1, keepdims=True), 1), vmin=0, vmax=1, cmap='Blues')
        for i in range(4):
            for j in range(4):
                axis.text(j, i, str(matrix[i, j]), ha='center', va='center')
        axis.set(xticks=range(4), yticks=range(4), xticklabels=['e', 'mu', 'tau', 'NC'], yticklabels=['e', 'mu', 'tau', 'NC'],
                 xlabel='Predicted class', ylabel='True class', title=f'Minimum validation NLL at seed 42; {epochs} passes\n{selected[2]["config"]["variant"]}; counts per true class = 2,500')
        save(fig, output, f'class_confusion_passes{epochs}')
    kinematic_variants = ['kinematics_absolute', 'kinematics_relative',
                          'kinematics_standard_linear', 'kinematics_standard_mlp']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for experiment in report['experiments']:
        if experiment['variant'] not in kinematic_variants or experiment['seed'] != 42:
            continue
        rows = [r for r in experiment['kinematics']['groups'] if r['class'] == 'all']
        for axis, component in zip(axes, ['pT_relative', 'delta_theta_rad', 'wrapped_delta_phi_rad']):
            values = [r['resolutions'][component]['absolute_quantiles_50_68_95'] for r in rows]
            axis.plot([r['coverage'] for r in rows], [v[1] if v is not None else np.nan for v in values], marker='o', label=experiment['variant'].replace('_', ' '))
            axis.set(xlabel='Coverage after classification-confidence cut', ylabel='68% absolute residual', title=component.replace('_', ' '))
    axes[0].legend(fontsize=7)
    save(fig, output, 'kinematic_resolution_coverage')
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    for experiment in report['experiments']:
        if experiment['variant'] not in kinematic_variants or experiment['seed'] != 42:
            continue
        for column, field in enumerate(['initE', 'initZ']):
            strata = [r for r in experiment['kinematics'].get('strata', []) if r['field'] == field]
            for axis, component in zip(axes[:, column], ['pT_relative', 'delta_theta_rad', 'wrapped_delta_phi_rad']):
                values = [r['resolution'][component]['absolute_quantiles_50_68_95'] for r in strata]
                axis.plot(range(len(strata)), [v[1] if v is not None else np.nan for v in values], marker='o',
                          label=experiment['variant'].replace('_', ' '))
                labels = (['<100', '100-500', '500-1000', '>=1000'] if field == 'initE' else
                          ['<-2700', '-2700:-2300', '-2300:-1900', '>=-1900'])
                axis.set(xticks=range(len(strata)), xticklabels=[f'{label}\nN={r["resolution"][component]["events"]}' for label, r in zip(labels, strata)],
                         xlabel='True neutrino energy [GeV]' if field == 'initE' else 'True primary z [mm]',
                         ylabel='68% absolute residual', title=component.replace('_', ' '))
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('Simulation-conditional component errors; all valid targets, no confidence cut')
    save(fig, output, 'kinematic_energy_position')
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
    component_rows = []
    for experiment in report['experiments']:
        if experiment['variant'] not in ['vertex_kinematics', *kinematic_variants] or experiment['seed'] != 42:
            continue
        config = next(e[2]['config'] for e in entries if e[0] == experiment['name'])
        if config['epochs'] != 4:
            continue
        quality = experiment['auxiliary_quality']['kinematics']
        for i, component in enumerate(['px', 'py', 'pz', 'E']):
            interval = quality['component_MSE_minus_constant'][i]
            reference = quality['constant_train_mean_MSE'][i]
            estimate = interval['estimate']/reference
            low, high = np.asarray(interval['block_CI95'])/reference
            component_rows.append({'name': experiment['name'], 'component': component,
                'events': quality['events'], 'MSE': quality['component_MSE'][i], 'constant_MSE': reference,
                'fractional_delta': estimate, 'block_CI95_low': low, 'block_CI95_high': high})
    for axis, component in zip(axes, ['px', 'py', 'pz', 'E']):
        rows = [r for r in component_rows if r['component'] == component]
        for index, row in enumerate(rows):
            # Draw the interval directly: a percentile interval need not contain its point estimate.
            axis.plot([row['block_CI95_low'], row['block_CI95_high']], [index, index], color='C0')
            axis.plot(row['fractional_delta'], index, 'o', color='C0')
        labels = [next(e['variant'].replace('_', ' ') for e in report['experiments'] if e['name'] == r['name']) for r in rows]
        axis.axvline(0., color='black', linestyle='--', linewidth=.8)
        axis.set(yticks=range(len(rows)), yticklabels=labels if component == 'px' else [],
                 xlabel='(MSE - constant MSE) / constant MSE', title=component+'; encoded target')
        axis.invert_yaxis()
    fig.suptitle('Four-pass regression: paired event-block CI95; selected train-mean reference')
    save(fig, output, 'kinematic_component_vs_constant')
    pd.DataFrame(component_rows).to_csv(output/'kinematic_components.csv', index=False)
    diagnostic_path = root/'regression_diagnostic/summary.json'
    if diagnostic_path.exists():
        import json
        diagnostic = json.loads(diagnostic_path.read_text())
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        cases = [('checkpoint_train', 'MLP: train'), ('checkpoint_validation', 'MLP: validation'),
                 ('ridge_train', 'Frozen ridge: train'), ('ridge_validation', 'Frozen ridge: validation')]
        checkpoint = next((e for e in report['experiments'] if e['name'] == diagnostic['source']), None)
        for i, axis in enumerate(axes):
            for index, (key, label) in enumerate(cases):
                row = diagnostic[key]
                baseline_mse = row['constant_MSE'][i]
                delta = row['MSE'][i]/baseline_mse-1
                axis.plot(delta, index, 'o', color='C0')
                if key.endswith('validation') and (key.startswith('ridge') or checkpoint is not None):
                    interval = (diagnostic['ridge_component_MSE_minus_constant'][i]['block_CI95']
                        if key.startswith('ridge') else checkpoint['auxiliary_quality']['kinematics']['component_MSE_minus_constant'][i]['block_CI95'])
                    axis.plot(np.asarray(interval)/baseline_mse, [index, index], color='C0')
            axis.axvline(0., color='black', linestyle='--', linewidth=.8)
            axis.set(yticks=range(len(cases)), yticklabels=[label for key, label in cases] if i == 0 else [],
                     xlabel='(MSE - constant MSE) / constant MSE', title=['px', 'py'][i]+'; encoded target')
            axis.invert_yaxis()
        fig.suptitle('Fixed encoder: train error and regularized readout; validation intervals use event blocks')
        save(fig, output, 'transverse_train_and_readout')
    display = next((e for e in entries if e[2]['config'].get('variant') == 'vertex_kinematics' and e[2]['config'].get('seed') == 42), baseline)
    native_examples(root, output, display)
