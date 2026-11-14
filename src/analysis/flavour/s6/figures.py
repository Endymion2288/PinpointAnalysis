"""English final figures and score/energy response matrices from frozen test."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.s6.calibration import extract, transform
from analysis.flavour.s6.prepare import KEY


CLASSES = ['CC $\\nu_e$', 'CC $\\nu_\\mu$', 'CC $\\nu_\\tau$', 'NC']


def save(fig, root, name):
    destination = root/'evaluation/figures'
    destination.mkdir(exist_ok=True)
    fig.savefig(destination/(name+'.pdf'), bbox_inches='tight')
    fig.savefig(destination/(name+'.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)


def render(args):
    root = args.root
    primary = 'mae3_100k_ensemble'
    result = json.loads((root/'evaluation/models'/primary/'balanced/result.json').read_text())
    matrix = np.asarray(result['confusion_raw_counts'])
    fig, ax = plt.subplots(figsize=(6, 5))
    normalized = matrix/matrix.sum(1, keepdims=True)
    im = ax.imshow(normalized, vmin=0, vmax=1, cmap='Blues')
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{matrix[i,j]:,}\n{normalized[i,j]:.3f}', ha='center', va='center',
                color='white' if normalized[i,j] > .55 else 'black', fontsize=9)
    ax.set(xticks=range(4), yticks=range(4), xticklabels=CLASSES, yticklabels=CLASSES,
        xlabel='Predicted class', ylabel='True class', title='Independent test: calibrated MAE ensemble')
    fig.colorbar(im, ax=ax, label='Row fraction')
    save(fig, root, 'test_confusion')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for c in range(4):
        directory = root/'evaluation/models'/primary/'balanced'
        roc, pr = pd.read_parquet(directory/f'roc_class{c}.parquet'), pd.read_parquet(directory/f'pr_class{c}.parquet')
        axes[0].plot(roc.FPR, roc.TPR, label=CLASSES[c])
        axes[1].plot(pr.recall, pr.precision, label=CLASSES[c])
    axes[0].set(xscale='log', xlim=(1e-5, 1), ylim=(0, 1), xlabel='False-positive rate', ylabel='Efficiency', title='Descriptive ROC curves')
    axes[1].set(xlim=(0, 1), ylim=(0, 1), xlabel='Recall', ylabel='Precision', title='Precision-recall, balanced prior')
    for ax in axes:
        ax.grid(alpha=.25); ax.legend()
    save(fig, root, 'test_roc_pr')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for key, label in [('reliability_before_temperature', 'Before temperature'), ('reliability', 'After temperature')]:
        row = result[key]['top_label']
        flag = np.asarray(row['counts']) > 0
        axes[0].plot(np.asarray(row['confidence'])[flag], np.asarray(row['accuracy'])[flag], 'o-', label=label)
    axes[0].plot([0, 1], [0, 1], '--', color='gray')
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel='Weighted mean confidence', ylabel='Weighted accuracy', title='Reliability, balanced prior')
    axes[0].legend()
    row = result['reliability']['top_label']
    axes[1].bar(np.linspace(.025, .975, 20), row['counts'], width=.045)
    axes[1].set(yscale='log', xlabel='Confidence bin', ylabel='Raw event count', title='Reliability bin statistics')
    save(fig, root, 'test_reliability')
    table = pd.read_csv(root/'evaluation/model_comparison.csv')
    balanced = table[table.scenario.eq('balanced')]
    chosen = balanced[balanced.name.isin([primary, 'scratch_100k_ensemble', 's3r_multiscale_1mm_s42_e4',
        's3r_multiscale_native_s42_e4', 's3r_multiscale_tracker_s42_e4', 's5_confirm8_vertex_kinematics_n10000_s42'])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = ['MAE ensemble', 'Scratch ensemble', 'S5 vertex/kinematics', 'S3 1 mm', 'S3 native', 'S3 + tracker']
    mapping = dict(zip([primary, 'scratch_100k_ensemble', 's5_confirm8_vertex_kinematics_n10000_s42',
        's3r_multiscale_1mm_s42_e4', 's3r_multiscale_native_s42_e4', 's3r_multiscale_tracker_s42_e4'], labels))
    axes[0].barh([mapping[n] for n in chosen.name], chosen.balanced_accuracy)
    axes[1].barh([mapping[n] for n in chosen.name], chosen['class_2_efficiency@0.01'])
    axes[0].set(xlabel='Balanced accuracy', title='Predeclared controls (different training budgets)')
    axes[1].set(xlabel='Tau efficiency', title='Frozen validation 1% background working point')
    save(fig, root, 'test_controls')
    fig, ax = plt.subplots(figsize=(8, 5))
    paths = sorted((root/'evaluation/conditions/variants').glob('*.json'))
    names, deltas, lower, upper = [], [], [], []
    for path in paths:
        data = json.loads(path.read_text())
        if data['condition'] == 'nominal':
            continue
        value = data['scenarios']['balanced']['metrics']['class_2_efficiency@0.01']
        names.append(data['condition']); deltas.append(value['delta'])
        lower.append(value['CI95'][0]); upper.append(value['CI95'][1])
    if names:
        ax.hlines(np.arange(len(names)), lower, upper)
        ax.plot(deltas, np.arange(len(names)), 'o')
        ax.set(yticks=np.arange(len(names)), yticklabels=names)
    ax.axvline(0, color='gray', ls='--')
    ax.set(xlabel='Tau-efficiency change (absolute)', title='Paired readout proxies, fixed nominal calibration/cuts')
    save(fig, root, 'readout_conditions')


def migration(args):
    root = args.root
    freeze = json.loads((root/'analysis_freeze.json').read_text())
    campaign, config = freeze['candidates'], freeze['evaluation']
    raw = pd.read_parquet(root/'predictions/test_raw.parquet')
    catalog = pd.read_parquet(root/'test_catalog.parquet')
    truth = pd.read_parquet(root/'test_truth.parquet')
    frame = raw[KEY+['label']].merge(catalog[KEY+['initE']], on=KEY, validate='one_to_one', sort=False)
    energy_edges = np.asarray(config['energy_edges_GeV'])
    score_edges = np.asarray(config['score_migration_edges'])
    ebin = np.clip(np.digitize(frame.initE*.001, energy_edges)-1, 0, len(energy_edges)-2)
    destination = root/'evaluation/migration'
    destination.mkdir(exist_ok=True)
    output = []
    for name in ['mae3_100k_ensemble', 'scratch_100k_ensemble']:
        p_raw = extract(raw, name, campaign)
        for scenario in freeze['calibrators'][name]['priors']:
            p = transform(p_raw, freeze['calibrators'][name]['temperature'], scenario['prior'])
            data = {}
            for c in range(4):
                take = frame.label.eq(c).to_numpy()
                score_matrix = np.histogram2d((frame.initE*.001).to_numpy()[take], p[take, c], bins=[energy_edges, score_edges])[0]
                data[f'class_{c}_energy_score'] = score_matrix
            data['class_energy_prediction'] = np.bincount((frame.label.to_numpy()*len(energy_edges[:-1])+ebin)*4+p.argmax(1),
                minlength=4*len(energy_edges[:-1])*4).reshape(4, len(energy_edges[:-1]), 4)
            data['raw_to_calibrated_argmax'] = np.bincount(p_raw.argmax(1)*4+p.argmax(1), minlength=16).reshape(4, 4)
            np.savez_compressed(destination/f'{name}_{scenario["name"]}.npz', energy_edges_GeV=energy_edges,
                score_edges=score_edges, **data)
            output.append({'model': name, 'scenario': scenario['name'], 'events': len(frame), 'physical_flux_weights_used': False})
    model = 's5_confirm8_vertex_kinematics_n10000_s42'
    aux = raw[KEY].copy()
    for k in ['vertex', 'kinematics']:
        for c in range(3 if k == 'vertex' else 4):
            aux[f'{k}{c}'] = raw[f'{model}__{k}{c}']
    aux = aux.merge(truth[KEY+['vertex_mm', 'vertex_supervision_valid', 'visible_E_GeV', 'visible_target_valid']],
        on=KEY, validate='one_to_one', sort=False)
    encoded = aux.kinematics3.to_numpy()*7.
    finite = np.isfinite(encoded) & (encoded >= 0) & (encoded < 50)
    predicted = np.full(len(aux), np.nan)
    predicted[finite] = np.expm1(encoded[finite])
    valid = aux.visible_target_valid.to_numpy() & finite
    m = np.histogram2d(aux.visible_E_GeV.to_numpy()[valid], predicted[valid], bins=[energy_edges, energy_edges])[0]
    residual = aux[[f'vertex{c}' for c in range(3)]].to_numpy()*[210, 210, 3000]-np.stack(aux.vertex_mm)
    flag = aux.vertex_supervision_valid.to_numpy()
    np.savez_compressed(destination/'s5_auxiliary_energy.npz', true_visible_energy_edges_GeV=energy_edges,
        reconstructed_energy_edges_GeV=energy_edges, counts=m)
    diagnostic = {'model': model, 'events': len(aux), 'energy_target_valid': int(aux.visible_target_valid.sum()),
        'energy_nonphysical_or_nonfinite_outputs': int((~finite).sum()), 'usable_energy_pairs': int(valid.sum()),
        'energy_outside_matrix_range': int(valid.sum()-m.sum()),
        'vertex_absolute_residual_quantiles_mm': np.quantile(np.abs(residual[flag]), [.5, .9, .99], axis=0).tolist(),
        'energy_scope': 'S5 visible-primary proxy target, not incoming neutrino energy and not calibrated calorimetric energy',
        'forward_folding_or_unfolding_validated': False, 'physical_exposure': 'unmeasured'}
    write_json(destination/'summary.json', {'score_energy_matrices': output, 'auxiliary': diagnostic,
        'empty_response_included': True, 'definition': 'rows condition on fixed true-energy bins; raw counts, no extrapolation or response inversion'})
    energy(args)


def energy(args):
    """Plot the compact frozen response matrices without rereading events."""
    root = args.root
    p = np.load(root/'evaluation/migration/mae3_100k_ensemble_balanced.npz')
    energy_edges = p['energy_edges_GeV']
    labels = [f'{low:g}–{high:g}' if high < 1e9 else f'≥{low:g}'
              for low, high in zip(energy_edges[:-1], energy_edges[1:])]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), layout='constrained')
    cmap = plt.colormaps['viridis'].copy()
    cmap.set_bad('lightgray')
    for c, ax in enumerate(axes):
        values = p[f'class_{c}_energy_score']
        counts = values.sum(1)
        fractions = np.divide(values, counts[:, None], out=np.full_like(values, np.nan), where=counts[:, None] > 0)
        im = ax.imshow(fractions, aspect='auto', origin='lower', vmin=0, vmax=1, cmap=cmap,
            extent=[p['score_edges'][0], p['score_edges'][-1], -.5, len(counts)-.5])
        ax.set(title=CLASSES[c], xlabel='True-class probability', yticks=range(len(counts)),
            yticklabels=labels if c == 0 else [], xticks=[0, .5, 1],
            ylabel='Incoming energy [GeV]' if c == 0 else None)
        for i in np.flatnonzero(counts == 0):
            ax.text(.5, i, 'No events', ha='center', va='center')
    fig.colorbar(im, ax=axes, label='Fraction in energy row', shrink=.8)
    fig.suptitle('Independent test: score response by incoming energy')
    save(fig, root, 'energy_score_response')


def physical(args):
    root = args.root
    primary = root/'evaluation/models/mae3_100k_ensemble'
    results = [json.loads((p/'result.json').read_text()) for p in primary.iterdir() if (p/'result.json').exists()]
    results.sort(key=lambda v: v['prior'][2])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for target, label in [('0.01', 'Validation 1% background cut'), ('0.001', 'Validation 0.1% background cut')]:
        for ax, metric, ylabel in zip(axes, ['efficiency', 'purity'], ['Tau efficiency', 'Tau purity']):
            names = f'class_2_{metric}@{target}'
            x = [v['prior'][2] for v in results]
            y = [v['fixed'][names] for v in results]
            lo, hi = np.array([v['intervals'][names]['CI95'] for v in results]).T
            ax.vlines(x, lo, hi)
            ax.plot(x, y, 'o-', label=label)
            ax.set(xscale='log', xlabel='Assumed tau prior', ylabel=ylabel, ylim=(0, 1))
            ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    efficiency = [v['fixed'][f'class_2_efficiency@{target}'] for v in results for target in ['0.01', '0.001']]
    axes[0].set_ylim(0, max(efficiency)*1.3)
    axes[1].set(yscale='log', ylim=(1e-4, 1))
    fig.suptitle('Frozen independent test: fixed prior scenarios and block intervals')
    save(fig, root, 'tau_prior_scenarios')
    balanced = json.loads((primary/'balanced/result.json').read_text())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, prefix, title in zip(axes, ['energy_GeV_', 'vertex_z_mm_', 'pixel_occupancy_'],
        ['Incoming energy [GeV]', 'Interaction vertex z [mm]', 'Pixel channel occupancy']):
        rows = [v for v in balanced['conditional'] if v['stratum'].startswith(prefix)]
        y = [v['fixed']['class_2_efficiency@0.01'] for v in rows]
        intervals = [v['fixed_block_intervals']['class_2_efficiency@0.01']['CI95'] for v in rows]
        lo, hi = np.array(intervals, dtype=float).T
        ax.vlines(np.arange(len(rows)), lo, hi)
        ax.plot(np.arange(len(rows)), y, 'o')
        labels = [v['stratum'][len(prefix):].replace('1000000000', '1e9')+f'\nTau events: {v["class_counts"][2]:,}' for v in rows]
        finite = np.r_[lo, hi]
        upper = max(.01, float(np.nanmax(finite))*1.3) if np.isfinite(finite).any() else .01
        ax.set(xticks=np.arange(len(rows)), xticklabels=labels, ylim=(0, min(1, upper)), xlabel=title,
            ylabel='Tau efficiency' if ax is axes[0] else None)
        ax.tick_params(axis='x', labelrotation=40, labelsize=8)
        ax.grid(alpha=.25)
    fig.suptitle('Fixed validation 1% background cut; conditional block intervals')
    save(fig, root, 'tau_physical_conditions')
    paths = sorted((root/'evaluation/conditions/transport').glob('*.json'))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    rows = [json.loads(p.read_text()) for p in paths if p.stem != 'nominal']
    for ax, name, label in zip(axes, ['balanced_accuracy', 'class_2_efficiency@0.01'],
        ['Balanced-accuracy change', 'Tau-efficiency change']):
        values = [r['scenarios']['balanced']['metrics'][name] for r in rows]
        if values:
            ax.hlines(np.arange(len(rows)), [v['CI95'][0] for v in values], [v['CI95'][1] for v in values])
            ax.plot([v['delta'] for v in values], np.arange(len(rows)), 'o')
        ax.set(yticks=np.arange(len(rows)), yticklabels=[r['condition'] for r in rows], xlabel=label)
        ax.axvline(0, color='gray', ls='--')
    fig.suptitle('Actual G4 transport proxies; nominal calibration and cuts fixed')
    save(fig, root, 'transport_conditions')
    generator = json.loads((root/'evaluation/conditions/generator_comparison.json').read_text())
    delta = generator['scenarios']['balanced']['NUX_minus_Standard']
    names = ['balanced_accuracy', 'NLL', 'class_2_efficiency@0.01']
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hlines(range(3), [delta[n]['CI95'][0] for n in names], [delta[n]['CI95'][1] for n in names])
    ax.plot([delta[n]['delta'] for n in names], range(3), 'o')
    ax.axvline(0, color='gray', ls='--')
    ax.set(yticks=range(3), yticklabels=['Balanced accuracy', 'NLL', 'Tau efficiency at frozen 1% cut'],
        xlabel='NUX minus Standard (absolute)', title='Matched benchmark, independent generator source-file intervals')
    save(fig, root, 'generator_comparison')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['render', 'migration', 'physical', 'energy'])
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == '__main__':
    main()
