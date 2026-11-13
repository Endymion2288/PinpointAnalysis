"""S3 revision 2 requires healthy learning as well as complete four-class outputs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from analysis.flavour.inventory import write_json
from analysis.flavour.s3.accept import comparison_inventory, conditional_inventory, promotion
from analysis.flavour.s3.evaluate import factorial_specifications


def confirmation_contract(screen, confirmation):
    tasks = [0, 2, 3, 4, 6, 7]
    expected = []
    for task in tasks:
        for seed in [17, 2026]:
            config = screen['experiments'][task]
            expected.append({**config, 'seed': seed,
                             'name': config['name'].replace('_s42_', f'_s{seed}_'),
                             'role': 's3_revision2_frozen_confirmation'})
    return len(screen['experiments']) == 8 and confirmation['experiments'] == expected


def experiment_record(root, config, evaluation):
    directory = root / 'experiments' / config['name']
    def load(filename):
        path = directory / filename
        return json.loads(path.read_text()) if path.exists() else {}
    manifest, readback, health = [load(p) for p in ['manifest.json', 'final_checkpoint_checks.json', 'internal_health_checks.json']]
    path = load('path_diagnostics.json')
    return {'name': config['name'], 'complete': manifest.get('status') == 'complete',
            'frozen_configuration': manifest.get('config') == config,
            'readback': readback.get('status') == 'pass' and len(readback.get('checks', {})) == 14
                        and all(readback.get('checks', {}).values()),
            'internal_health_and_effective_learning': health.get('internal_health_and_learning_passed') is True
                        and bool(health.get('checks')) and all(health['checks'].values()),
            'fixed_checkpoint_path_diagnostic': config.get('architecture') != 'plane_module_v2' or (
                path.get('cached_head_matches_original') is True and path.get('presence_preserved') is True
                and path.get('probe_events') == 64 and path.get('permuted_fraction', 0) > 0),
            'evaluated': config['name'] in evaluation,
            'allocation_complete': evaluation.get(config['name'], {}).get('allocation_complete', False),
            'calibration_test_unused': manifest.get('calibration_test_used') is False}


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
    screen, confirmation, specification, evaluation, conditional = [json.loads(p.read_text()) for p in
        [args.campaign, args.confirmation, args.comparisons, args.evaluation, args.conditional]]
    accounting = json.loads((args.root / 'campaign_accounting.json').read_text())
    by_name = {e['name']: e for e in evaluation['experiments']}
    configurations = screen['experiments'] + confirmation['experiments']
    records = [experiment_record(args.root, config, by_name) for config in configurations]
    checks = {
        'prespecified_confirmation_matrix_complete': confirmation_contract(screen, confirmation),
        'all_frozen_fits_readbacks_and_internal_health_pass': all(
            all(value for key, value in row.items() if key != 'name') for row in records),
        'only_revision2_models_used_for_conclusions': set(by_name) == {c['name'] for c in configurations}
            and all(name.startswith('s3r_') for name in by_name),
        'all_frozen_contrasts_complete': not evaluation['pending'],
        'factorial_inventory_matches_frozen_specification': len(evaluation['factorial_interactions']) == len(factorial_specifications(specification)),
        'expected_physical_bootstrap': evaluation['bootstrap_replicates'] == 400
            and evaluation['resampling_blocks']['resampling_blocks'] == 128,
        'conditional_complete': conditional.get('requested_complete') is True
            and conditional.get('bootstrap_replicates') == 400 and not conditional.get('pending'),
        'terminal_allocation_evidence_complete': accounting['missing_history_records'] == 0
            and all(row['status'] in [3, 4] and row.get('allocation_complete', True) for row in accounting['allocations']),
        'cumulative_S3_budget_respected': accounting.get('prior_S3_device_hours_upper_bound', 0) > 0
            and accounting['organization_GPU_hours_upper_bound'] <= 120,
        'plots_and_error_tables_present': all((args.evaluation.parent / p).is_file() for p in
            ['four_class_diagnostics.pdf', 'four_class_resource_tradeoff.pdf', 'error_groups.parquet',
             'internal_training_health.pdf', 'token_census/counts.parquet',
             'repaired_model_overview.pdf','repaired_paired_intervals.pdf'])
            and all((args.conditional.parent / p).is_file() for p in ['four_class_domain_curves.pdf', 'sampled_curves.parquet']),
        'native_error_displays_present': all((args.evaluation.parent / 'event_displays' / name / 'native_error_groups.pdf').is_file()
            for name in ['s3r_multiscale_native_s42_e4', 's3r_multiscale_tracker_s42_e4']),
    }
    checks.update(comparison_inventory(evaluation, specification))
    checks.update(conditional_inventory(conditional, evaluation, specification))
    decisions = []
    if checks['all_frozen_fits_readbacks_and_internal_health_pass']:
        for comparison in evaluation['seed_mean_comparisons']:
            if 'minus_fresh_ParT' in comparison['name']:
                decision = promotion(comparison, by_name)
                # Shared operational charges include backend/setup/concurrency.
                # They do not establish an intrinsic model throughput advantage.
                decision.update(efficiency_path=False, fully_promoted=decision['quality_path'],
                                efficiency_scope='not claimed from shared mixed-backend allocation charges')
                decisions.append(decision)
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'stage': 'S3_revision2',
              'strict_output_gate_passed': all(checks.values()), 'checks': checks,
              'experiments': records, 'promotion_decisions': decisions,
              'legacy_S3_method_conclusions_withdrawn': True,
              'organization_GPU_hours_upper_bound': accounting['organization_GPU_hours_upper_bound'],
              'research_completion_does_not_require_new_model_promotion': True,
              'calibration_test_used': False, 'S4_S6_executed': False}
    write_json(args.root / 'final_acceptance.json', report)
    print(json.dumps({k: report[k] for k in ['utc', 'strict_output_gate_passed', 'checks']}))
    if args.require_complete and not report['strict_output_gate_passed']:
        raise SystemExit('S3 repair remains incomplete; inspect final_acceptance.json')


if __name__ == '__main__':
    main()
