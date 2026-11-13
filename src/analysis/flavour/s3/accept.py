"""Final S3 output gates and four-class method promotion decisions."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from analysis.flavour.inventory import write_json
from analysis.flavour.s3.metrics import NAMES


def comparison_inventory(evaluation, specification):
    """Require every frozen contrast, domain, metric and bootstrap replicate."""
    expected_pairs = {(p['a'], p['b']) for p in specification['pairs']}
    actual_pairs = [(p['a'], p['b']) for p in evaluation['paired_comparisons']]
    expected_means = {g['name']: [(p['a'], p['b']) for p in g['pairs']]
                      for g in specification['seed_mean_comparisons']}
    actual_means = {g['name']: [(p['a'], p['b']) for p in g['pairs']]
                    for g in evaluation['seed_mean_comparisons']}
    reports = [*evaluation['paired_comparisons'], *evaluation['seed_mean_comparisons'],
               *evaluation['factorial_interactions']]
    metrics_complete = True
    for report in reports:
        if report.get('replicates') != 400:
            metrics_complete = False
        for scope in ['raw', 'common_support']:
            details = report.get('metrics', {}).get(scope, {})
            if set(details) != set(NAMES):
                metrics_complete = False
            for value in details.values():
                numbers = [value.get('delta'), *value.get('CI95', [])]
                if (len(numbers) != 3 or any(v is None or not np.isfinite(v) for v in numbers)
                        or value.get('valid_replicates') != 400):
                    metrics_complete = False
    return {
        'all_frozen_single_seed_pairs': len(actual_pairs) == len(expected_pairs) and set(actual_pairs) == expected_pairs,
        'all_frozen_three_seed_pairs': actual_means == expected_means and len(actual_means) == len(evaluation['seed_mean_comparisons'])
            and all(len(pairs) == 3 for pairs in expected_means.values()),
        'all_four_class_contrast_metrics': metrics_complete,
    }


def confirmation_contract(screen, confirmation):
    """Confirmation must contain both extra seeds for all selected structures."""
    selection = confirmation.get('screen_selection', {})
    selected = selection.get('selected_tasks', [])
    tasks = sorted(set(selected + [5, 6]))
    if len(selected) != 2 or len(set(selected)) != 2 or selection.get('confirmed_tasks') != tasks:
        return False
    expected = []
    for task in tasks:
        if task not in [*range(10), 14, 16]:
            return False
        for seed in [17, 2026]:
            original = screen['experiments'][task]
            expected.append({**original, 'seed': seed, 'name': original['name'].replace('_s42_', f'_s{seed}_'),
                             'role': 's3_frozen_confirmation'})
    return confirmation['experiments'] == expected


def conditional_inventory(conditional, evaluation, specification):
    """Audit every requested model/domain contrast, retaining undefined metrics."""
    expected_models = {e['name'] for e in evaluation['experiments']
                       if e['name'].startswith(('s3_', 's3r_', 'part_', 'pn_'))}
    actual_models = conditional.get('experiments', [])
    model_complete = (set(actual_models) == expected_models and
                      len(actual_models) == len(expected_models))
    expected_domains = {'pixel_centered', 'large_scintillator', 'detector_active',
                        'all_empty', 'pixel_ge_1000', 'tracker_active', 'tracker_empty'}
    rows = conditional.get('paired_domains', [])
    domain_complete = ({row['domain'] for row in rows} == expected_domains and
                       len(rows) == len(expected_domains))
    expected_pairs = {(p['a'], p['b']) for p in specification['pairs']
                      if p['a'] in expected_models and p['b'] in expected_models}
    expected_means = {g['name']: [(p['a'], p['b']) for p in g['pairs']]
                      for g in specification['seed_mean_comparisons']
                      if all(p['a'] in expected_models and p['b'] in expected_models for p in g['pairs'])}
    contrasts_complete = True
    metrics_complete = True
    for row in rows:
        pairs = row['paired_comparisons']; means = row['seed_mean_comparisons']
        contrasts_complete &= (len(pairs) == len(expected_pairs) and
            {(p['a'], p['b']) for p in pairs} == expected_pairs and
            len(means) == len(expected_means) and
            {g['name']: [(p['a'], p['b']) for p in g['pairs']] for g in means} == expected_means)
        for report in [*pairs, *means]:
            metrics_complete &= report.get('replicates') == 400
            for scope in ['raw', 'common_support']:
                details = report.get('metrics', {}).get(scope, {})
                metrics_complete &= set(details) == set(NAMES)
                for value in details.values():
                    valid = value.get('valid_replicates')
                    ci = value.get('CI95', [])
                    metrics_complete &= (isinstance(valid, int) and 0 <= valid <= 400 and
                        len(ci) == 2 and (ci == [None, None] if valid == 0 else
                                         all(v is not None and np.isfinite(v) for v in ci)))
                    delta = value.get('delta')
                    metrics_complete &= delta is None or np.isfinite(delta)
    return {'conditional_model_inventory_complete': model_complete,
            'conditional_domain_inventory_complete': domain_complete,
            'conditional_frozen_contrasts_complete': bool(contrasts_complete),
            'conditional_four_class_metrics_complete': bool(metrics_complete)}


def promotion(comparison, experiments):
    """Research completion and model promotion are separate decisions."""
    raw = comparison['metrics']['raw']
    quality_paths = [key for key in ['macro_AP', 'macro_F1', 'balanced_accuracy']
                     if raw[key]['delta'] >= .01 and raw[key]['CI95'][0] > 0]
    nll_safe = raw['NLL']['CI95'][1] <= .01
    class_degradation = [c for c in range(4) if raw[f'class_{c}_AP']['delta'] < -.01]
    a_cost, b_cost = [], []
    for pair in comparison['pairs']:
        a_cost.append(experiments[pair['a']]['allocated_gpu_hours'])
        b_cost.append(experiments[pair['b']]['allocated_gpu_hours'])
    costs_complete = all(v is not None for v in a_cost + b_cost)
    hardware_comparable = all(experiments[p['a']].get('GPU') is not None and
        experiments[p['a']]['GPU'] == experiments[p['b']].get('GPU') for p in comparison['pairs'])
    ratio = float(np.mean(b_cost) / np.mean(a_cost)) if costs_complete else None
    # Equivalence requires the event interval to respect every preregistered
    # noninferiority margin, rather than accepting an inconclusive point estimate.
    near_equivalent = raw['macro_AP']['CI95'][0] >= -.005 and nll_safe and all(
        raw[f'class_{c}_AP']['CI95'][0] >= -.01 for c in range(4))
    quality = bool(quality_paths and nll_safe and not class_degradation)
    efficiency = bool(costs_complete and hardware_comparable and ratio <= .5 and near_equivalent)
    return {'name': comparison['name'], 'quality_path': quality, 'improved_macro_metrics': quality_paths,
        'NLL_noninferiority_interval_pass': nll_safe, 'class_AP_point_degradation_over_0.01': class_degradation,
        'efficiency_path': efficiency, 'quality_near_equivalence_interval_pass': bool(near_equivalent),
        'matched_GPU_type_for_efficiency': hardware_comparable,
        'mean_seed_allocated_cost_ratio_B_over_A': ratio, 'fully_promoted': quality or efficiency,
        'training_seed_spread': {key: raw[key].get('training_seed_sample_SD') for key in
                                ['macro_AP', 'NLL', 'macro_F1', 'balanced_accuracy']},
        'common_support_results': comparison['metrics']['common_support'],
        'scope': 'four-class preregistered validation comparison; tau-only benefit cannot grant promotion'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--confirmation', type=Path, required=True)
    parser.add_argument('--comparisons', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--conditional', type=Path, required=True)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    campaigns = [json.loads(p.read_text()) for p in [args.campaign, args.confirmation]]
    evaluation = json.loads(args.evaluation.read_text())
    accounting = json.loads((args.root / 'campaign_accounting.json').read_text())
    conditional = json.loads(args.conditional.read_text())
    comparisons = json.loads(args.comparisons.read_text())
    by_name = {e['name']: e for e in evaluation['experiments']}
    records = []
    for config in [c for campaign in campaigns for c in campaign['experiments']]:
        path = args.root / 'experiments' / config['name']
        manifest_path = path / 'manifest.json'; checks_path = path / 'final_checkpoint_checks.json'
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        checks = json.loads(checks_path.read_text()) if checks_path.exists() else {}
        records.append({'name': config['name'], 'complete': manifest.get('status') == 'complete',
            'frozen_configuration': manifest.get('config') == config,
            'readback': checks.get('status') == 'pass' and len(checks.get('checks', {})) == 14 and all(checks.get('checks', {}).values()),
            'evaluated': config['name'] in by_name,
            'allocation_complete': by_name.get(config['name'], {}).get('allocation_complete', False),
            'calibration_test_unused': manifest.get('calibration_test_used') is False})
    checks = {
        'frozen_confirmation_matrix_complete': confirmation_contract(*campaigns),
        'all_frozen_experiments_complete': all(all(v for k, v in row.items() if k != 'name') for row in records),
        'all_evaluation_contrasts_complete': not evaluation['pending'],
        'three_seed_confirmation_present': len(evaluation['seed_mean_comparisons']) >= 5,
        'four_class_factorial_present': len(evaluation['factorial_interactions']) == 1,
        'expected_physical_bootstrap': evaluation['bootstrap_replicates'] == 400 and evaluation['resampling_blocks']['resampling_blocks'] == 128,
        'conditional_diagnostics_complete': conditional.get('requested_complete') is True and
            set(row['name'] for row in records).issubset(conditional.get('experiments', [])) and
            conditional.get('bootstrap_replicates') == 400 and not conditional.get('pending'),
        'conditional_curves_present': (args.conditional.parent / 'four_class_domain_curves.pdf').is_file() and
            (args.conditional.parent / 'sampled_curves.parquet').is_file(),
        'terminal_backend_history_complete': accounting['missing_history_records'] == 0 and
            all(a['status'] in [3, 4] and a.get('allocation_complete', True) for a in accounting['allocations']),
        'organization_budget_respected': accounting['organization_GPU_hours_upper_bound'] <= 120,
        'diagnostic_plots_present': all((args.evaluation.parent / p).is_file() for p in
                                      ['four_class_diagnostics.pdf', 'four_class_resource_tradeoff.pdf', 'error_groups.parquet']),
        'native_error_displays_present': all((args.evaluation.parent / 'event_displays' / name / 'native_error_groups.pdf').is_file()
            for name in ['s3_f16_context_latent_s42_e4', 's3_f16_context_latent_tracker_s42_e4']),
    }
    checks.update(comparison_inventory(evaluation, comparisons))
    checks.update(conditional_inventory(conditional, evaluation, comparisons))
    decisions = [promotion(c, by_name) for c in evaluation['seed_mean_comparisons']
                 if c['name'] != 'primary_native_minus_1mm_three_seed'] if checks['all_frozen_experiments_complete'] else []
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'stage': 'S3', 'checks': checks,
        'strict_output_gate_passed': all(checks.values()), 'experiments': records, 'promotion_decisions': decisions,
        'organization_GPU_hours_upper_bound': accounting['organization_GPU_hours_upper_bound'],
        'research_completion_does_not_require_new_model_promotion': True, 'S4_S6_executed': False}
    write_json(args.root / 'final_acceptance.json', report)
    print(json.dumps({k: report[k] for k in ['utc', 'strict_output_gate_passed', 'checks']}))
    if args.require_complete and not report['strict_output_gate_passed']:
        raise SystemExit('S3 outputs remain incomplete; see final_acceptance.json')


if __name__ == '__main__':
    main()
