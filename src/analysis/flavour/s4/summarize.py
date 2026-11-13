"""Export compact S4 measurements without changing selection or statistics."""
import argparse
import json
from pathlib import Path
import statistics

from analysis.flavour.inventory import write_json


METRICS = ['macro_AP', 'NLL', 'macro_F1', 'balanced_accuracy', 'class_2_AP', 'class_2_F1',
    'class_2_recall', 'class_2_eff_at_0.01_FPR', 'class_2_eff_at_0.001_FPR']


def compact_comparison(row):
    return {**{k: v for k, v in row.items() if k not in ['metrics', 'tau_branches']},
        'metrics': {domain: {k: values[k] for k in METRICS} for domain, values in row['metrics'].items()},
        'tau_branches': {domain: {k: values[k] for k in ['AP', 'eff_at_0.01_FPR', 'eff_at_0.001_FPR']}
            for domain, values in row.get('tau_branches', {}).items()}}


def summarize(evaluation, health):
    groups = {}
    points = []
    for row in evaluation['experiments']:
        config = row['config']
        point = {'name': row['name'], 'program': row['program'], 'initialization': config.get('initialization', 'scratch'),
            'representation': config.get('probe_representation', 'full'), 'label_budget': config['label_budget'],
            'seed': config['seed'], 'raw': {k: row['four_class'][k] for k in METRICS},
            'common_support': {k: row['common_support_four_class'][k] for k in METRICS}}
        points.append(point)
        identity = tuple(point[k] for k in ['program', 'initialization', 'representation', 'label_budget'])
        groups.setdefault(identity, []).append(point)
    label_statistics = []
    for identity, rows in groups.items():
        measured = {}
        for domain in ['raw', 'common_support']:
            measured[domain] = {k: {'mean': statistics.mean(r[domain][k] for r in rows),
                'training_seed_sample_SD': statistics.stdev(r[domain][k] for r in rows) if len(rows) > 1 else None}
                for k in METRICS}
        label_statistics.append({**dict(zip(['program', 'initialization', 'representation', 'label_budget'], identity)),
            'seeds': [r['seed'] for r in rows], 'measurements': measured})
    sampled = health['models']
    silu = [row['activations']['head.2'] for row in sampled]
    return {'utc': evaluation['utc'], 'scope': evaluation['scope'],
        'requested_complete': evaluation['requested_complete'], 'evaluated_predictions': len(points),
        'pending_items': len(evaluation['pending']), 'paired_comparisons': len(evaluation['paired_comparisons']),
        'seed_mean_comparisons_completed': len(evaluation['seed_mean_comparisons']),
        'checkpoint_health': {'verified_train_models': len(sampled), 'checkpoint_checks_per_model': 14,
            'health_samples': health['samples'], 'all_sampled_healthy': all(row['all_healthy'] for row in sampled),
            'SiLU_RMS_range': [min(row['RMS_range'][0] for row in silu), max(row['RMS_range'][1] for row in silu)],
            'SiLU_maximum_exact_zero_fraction': max(row['maximum_zero_fraction'] for row in silu),
            'minimum_informative_input_gradient_squared_norm': min(row['minimum_input_gradient_squared_norm'] for row in sampled)},
        'points': points, 'label_statistics': label_statistics,
        'contrasts': [compact_comparison(row) for row in evaluation['paired_comparisons']],
        'seed_mean_contrasts': [compact_comparison(row) for row in evaluation['seed_mean_comparisons']],
        'classwise_operating_points': {row['name']: {kind: {rate: {
                **{k: point[k] for k in ['threshold', 'background_acceptance', 'background_selected', 'signal_selected',
                    'signal_count_interval_95', 'background_count_interval_95', 'signal_events', 'background_events']},
                'signal_efficiency': point['tau_efficiency']}
                for rate, point in rates.items()} for kind, rates in row['classwise_operating_points'].items()}
            for row in evaluation['experiments']},
        'operating_point_scope': 'validation-selected thresholds; descriptive Clopper-Pearson count intervals, no independent operating-point coverage claim',
        'common_support': evaluation['common_support'], 'resampling_blocks': evaluation['resampling_blocks'],
        'bootstrap_replicates': evaluation['bootstrap_replicates'], 'interval_scope': evaluation['interval_scope'],
        'tau_branch_comparison_scope': evaluation['tau_branch_comparison_scope'], 'calibration_test_used': False,
        'selection_scope': 'fixed validation NLL checkpoints; seed mean metrics are not an ensemble; seed SD is separate from conditional physical-event intervals'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--health', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = summarize(json.loads(args.evaluation.read_text()), json.loads(args.health.read_text()))
    report.update(source=str(args.evaluation), health_source=str(args.health))
    write_json(args.output, report)
    print(json.dumps({k: report[k] for k in ['evaluated_predictions', 'pending_items', 'checkpoint_health']}))


if __name__ == '__main__':
    main()
