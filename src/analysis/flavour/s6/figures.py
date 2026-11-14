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
    p = np.load(destination/'mae3_100k_ensemble_balanced.npz')
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for c, ax in enumerate(axes):
        values = p[f'class_{c}_energy_score']
        values = np.divide(values, values.sum(1, keepdims=True), out=np.zeros_like(values), where=values.sum(1, keepdims=True) > 0)
        ax.imshow(values, aspect='auto', origin='lower', vmin=0, vmax=1, cmap='viridis')
        ax.set(title=CLASSES[c], xlabel='True-class score bin', ylabel='True energy bin' if c == 0 else None)
    fig.suptitle('Score response by incoming energy; independent test, row fractions')
    save(fig, root, 'energy_score_response')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['render', 'migration'])
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == '__main__':
    main()
