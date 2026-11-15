"""Paired small Tune conditions at both descriptive and frozen nominal cuts."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.flavour.inventory import write_json
from analysis.flavour.s3.metrics import bootstrap_weights
from analysis.flavour.v2.partitions import KEY, equal_identity_rows
from analysis.flavour.v2.r1.evaluate import domain_report, probability_columns, scalar_summary, interval
from analysis.flavour.v2.r1.metrics import METRICS_REVISION, three_from_four


def frozen_points(frame, reference):
    y = frame.label.to_numpy()
    p = frame[probability_columns(frame)].to_numpy()
    p3 = p if p.shape[1] == 3 else three_from_four(p)
    specifications = []
    for c, label in enumerate([0, 1, 3]):
        for fpr in [.01, .001]:
            specifications.append((f'T3_{label}_FPR_{fpr}', y==label, (y!=label)&(y!=2),
                p3[:, c], reference['T3']['per_class'][c][f'threshold_FPR_{fpr}']))
    if p.shape[1] == 4:
        for fpr in [.01, .001]:
            specifications.append((f'tau_FPR_{fpr}', y==2, y!=2, p[:, 2],
                reference['T4']['per_class'][2][f'threshold_FPR_{fpr}']))
    rows = []
    selections = []
    for name, positive, negative, score, cut in specifications:
        selected = score>=cut if cut is not None else np.zeros(len(y), dtype=bool)
        rows.append({'name': name, 'frozen_full_Tune_nominal_threshold': cut,
            'signal_events': int(positive.sum()), 'background_events': int(negative.sum()),
            'selected_signal': int((positive&selected).sum()),
            'selected_background': int((negative&selected).sum()),
            'selected_tau': int(((y==2)&selected).sum())})
        selections.append((positive, negative, selected))
    def compute(weight=None):
        w = np.ones(len(y)) if weight is None else np.asarray(weight, dtype=float)
        support = np.bincount(y, weights=w, minlength=4)
        w = w/np.maximum(support[y], 1e-30)
        return np.array([float(w[mask&selected].sum()/max(w[mask].sum(), 1e-30))
            for positive, negative, selected in selections for mask in [positive, negative]])
    values = compute()
    for i, row in enumerate(rows):
        row.update(efficiency=values[2*i], actual_FPR=values[2*i+1])
    return rows, compute


def model_condition_contrasts(left, right):
    """Retain the covariance of models, conditions and their common events."""
    nominal_point = left['nominal'][0]-right['nominal'][0]
    nominal_samples = left['nominal'][1]-right['nominal'][1]
    result = {}
    for condition in left:
        point = left[condition][0]-right[condition][0]
        samples = left[condition][1]-right[condition][1]
        result[condition] = {'a_minus_b': interval(point, samples),
            'condition_shift_a_minus_b': interval(point-nominal_point, samples-nominal_samples)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--replicates', type=int, default=400)
    args = parser.parse_args()
    root = args.root
    config = json.loads((root/'evaluation_config.json').read_text())
    models = [e for e in config['models'] if e.get('kind', 'main') == 'main' and e['seed'] == 42]
    reports, summary, descriptive = {}, [], {}
    common_nominal = None
    for entry in models:
        name = entry['name']
        nominal = pd.read_parquet(root/'diagnostics'/f'{name}_nominal/predictions.parquet').sort_values(KEY).reset_index(drop=True)
        if common_nominal is None:
            common_nominal = nominal
            weights, design = bootstrap_weights(nominal, args.replicates, 20261117)
        elif not equal_identity_rows(nominal, common_nominal) or not np.array_equal(nominal.label, common_nominal.label):
            raise ValueError('model stress cohorts or labels differ')
        training_manifest = json.loads((root/'experiments'/name/'manifest.json').read_text())
        if training_manifest.get('status') != 'complete':
            raise ValueError(f'R1 stress evaluation requires a completed model: {name}')
        reference = training_manifest['metrics']
        nominal_fixed, nominal_compute = frozen_points(nominal, reference)
        nominal_samples = np.array([nominal_compute(w) for w in weights])
        nominal_scalar = scalar_summary(nominal)
        nominal_scalar_point = nominal_scalar()
        nominal_scalar_samples = np.asarray([nominal_scalar(w) for w in weights])
        descriptive[name] = {}
        full = pd.read_parquet(root/entry['predictions']).set_index(KEY).loc[pd.MultiIndex.from_frame(nominal[KEY])].reset_index()
        if not equal_identity_rows(full, nominal) or not np.array_equal(full.label, nominal.label):
            raise ValueError('nominal checkpoint readback identities/labels differ')
        nominal_p = nominal[probability_columns(nominal)].to_numpy()
        full_p = full[probability_columns(full)].to_numpy()
        delta_p = nominal_p-full_p
        nominal_three = nominal_p if nominal_p.shape[1]==3 else nominal_p[:, [0, 1, 3]]
        full_three = full_p if full_p.shape[1]==3 else full_p[:, [0, 1, 3]]
        tau_cut_changes = {}
        if nominal_p.shape[1]==4:
            for fpr in [.01, .001]:
                cut = reference['T4']['per_class'][2][f'threshold_FPR_{fpr}']
                tau_cut_changes[str(fpr)] = int(np.count_nonzero((nominal_p[:, 2]>=cut)!=(full_p[:, 2]>=cut))) if cut is not None else 0
        inference_manifest = json.loads((root/'diagnostics'/f'{name}_nominal/manifest.json').read_text())
        reports[name] = {'nominal_probability_readback_max_absolute': float(np.abs(delta_p).max()),
            'nominal_probability_readback_mean_absolute': float(np.abs(delta_p).mean()),
            'nominal_argmax_changes': int(np.count_nonzero(nominal_p.argmax(1)!=full_p.argmax(1))),
            'nominal_conditional_three_argmax_changes': int(np.count_nonzero(nominal_three.argmax(1)!=full_three.argmax(1))),
            'nominal_tau_cut_selection_changes': tau_cut_changes,
            'training_GPU': training_manifest.get('GPU', 'CPU'),
            'diagnostic_GPU': inference_manifest['GPU'],
            'bootstrap_design': design, 'conditions': {}}
        for condition in ['nominal', 'noise', 'combined', 'remove_pixel', 'remove_tracker']:
            frame = pd.read_parquet(root/'diagnostics'/f'{name}_{condition}/predictions.parquet').sort_values(KEY).reset_index(drop=True)
            if not equal_identity_rows(frame, nominal) or not np.array_equal(frame.label, nominal.label):
                raise ValueError('condition pair identities/labels differ')
            report = domain_report(frame, np.ones(len(frame), dtype=bool))
            if condition=='nominal':
                scalar_point, scalar_samples = nominal_scalar_point, nominal_scalar_samples
            else:
                compute_scalar = scalar_summary(frame)
                scalar_point = compute_scalar()
                scalar_samples = np.asarray([compute_scalar(w) for w in weights])
            descriptive[name][condition] = (scalar_point, scalar_samples)
            fixed, compute = frozen_points(frame, reference)
            changes = np.array([compute(w) for w in weights])-nominal_samples
            point = compute()-nominal_compute()
            for i, row in enumerate(fixed):
                for j, metric in enumerate(['efficiency', 'actual_FPR']):
                    row[f'{metric}_difference_from_nominal'] = float(point[2*i+j])
                    row[f'{metric}_difference_95pct'] = np.quantile(changes[:, 2*i+j], [.025, .975]).tolist()
            reports[name]['conditions'][condition] = {'metrics': report, 'frozen_points': fixed,
                'paired_descriptive_differences': interval(scalar_point-nominal_scalar_point,
                    scalar_samples-nominal_scalar_samples) if condition!='nominal' else None}
            summary.append({'name': name, 'family': entry['family'], 'task': entry['task'], 'condition': condition,
                'BA3': report['T3']['BA'], 'NLL3': report['T3']['NLL'],
                'tau_eff_nominal_1pct_cut': next((r['efficiency'] for r in fixed if r['name']=='tau_FPR_0.01'), None),
                'tau_actual_FPR_nominal_1pct_cut': next((r['actual_FPR'] for r in fixed if r['name']=='tau_FPR_0.01'), None)})
    output = root/'evaluation'
    comparisons = [{**pair, 'conditions': model_condition_contrasts(descriptive[pair['a']], descriptive[pair['b']])}
                   for pair in config.get('pairs', []) if pair.get('domain', 'all')=='all'
                   and pair['a'] in descriptive and pair['b'] in descriptive]
    pd.DataFrame(summary).to_csv(output/'stress_metrics.csv', index=False)
    write_json(output/'stress_results.json', {'status': 'complete', 'models': reports, 'replicates': args.replicates,
        'metrics_revision': METRICS_REVISION,
        'paired_model_comparisons': comparisons,
        'model_comparison_scope': 'descriptive contrasts on common physical-block draws; condition-shift difference retains model/condition covariance; training seeds fixed',
        'scope': 'fixed small development Tune; accepted proxy amplitudes; paired family-connected physical blocks',
        'frozen_cut_source': 'full nominal development Tune, descriptive; no calibration/cut/confirmation payload',
        'input_removal_interpretation': 'reliance only; retrained no-tracker controls are evaluated separately',
        'Cal_A_Cut_B_Confirm_payload_read': False})


if __name__ == '__main__':
    main()
