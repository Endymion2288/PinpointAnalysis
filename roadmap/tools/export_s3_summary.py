"""Export versioned S3 statistics after the complete output gate passes.

Reads published reports only. Manual figure review and the scientific
completion decision remain recorded in the workbook, separately from this gate.
"""

import argparse
import json
from pathlib import Path


DOMAIN_METRICS = ('macro_AP', 'macro_F1', 'balanced_accuracy', 'NLL',
                  'class_0_AP', 'class_1_AP', 'class_2_AP', 'class_3_AP')


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    if not (root / 'final_acceptance.json').is_file():
        raise SystemExit('Complete S3 output gate required; no summary written')
    acceptance = load(root / 'final_acceptance.json')
    finalize = load(root / 'confirmation_finalize_state.json')
    if (not acceptance['strict_output_gate_passed'] or not all(acceptance['checks'].values())
            or finalize['status'] != 'strict_gate_passed'):
        raise SystemExit('Complete S3 output gate required; no summary written')
    evaluation = load(root / 'evaluation_final/summary.json')
    conditional = load(root / 'evaluation_final/conditional/summary.json')
    accounting = load(root / 'campaign_accounting.json')
    if evaluation['pending'] or conditional['pending'] or accounting['missing_history_records']:
        raise SystemExit('Complete S3 output gate required; no summary written')

    experiments = [{k: e[k] for k in
        ('name', 'config', 'four_class', 'common_support_four_class',
         'prediction_diagnostics', 'parameters', 'GPU', 'peak_gpu_mib',
         'allocated_gpu_hours', 'allocation_complete')}
        for e in evaluation['experiments']]
    domain_means = []
    for domain in conditional['paired_domains']:
        comparisons = [{
            'name': group['name'],
            'metrics': {scope: {k: group['metrics'][scope][k] for k in DOMAIN_METRICS}
                        for scope in ('raw', 'common_support')},
        } for group in domain['seed_mean_comparisons']]
        domain_means.append({'domain': domain['domain'], 'events': domain['events'],
                             'seed_mean_comparisons': comparisons})
    names = {e['name'] for e in experiments}
    figures = ['four_class_diagnostics.pdf', 'four_class_resource_tradeoff.pdf',
               'conditional/four_class_domain_curves.pdf']
    for name in ('s3_f16_context_latent_s42_e4', 's3_f16_context_latent_tracker_s42_e4'):
        figures.append(f'event_displays/{name}/native_error_groups.pdf')
    summary = {
        'status': 'strict_scientific_outputs_complete',
        'scope': evaluation['scope'],
        'evaluation_utc': evaluation['utc'],
        'evaluation_code_commit': evaluation['code_commit'],
        'results_root': str(root),
        'classes': ['CC nu e', 'CC nu mu', 'CC nu tau', 'NC'],
        'counts': {'screen': 17, 'additional_confirmation': 8,
                   'scientific_models': len(acceptance['experiments']),
                   'readback_checks': 14 * len(acceptance['experiments']),
                   'all_evaluated_models': len(experiments),
                   'single_seed_pairs': len(evaluation['paired_comparisons']),
                   'three_seed_comparisons': len(evaluation['seed_mean_comparisons']),
                   'factorial_interactions': len(evaluation['factorial_interactions']),
                   'conditional_domains': len(domain_means)},
        'strict_output_gate_passed': acceptance['strict_output_gate_passed'],
        'checks': acceptance['checks'],
        'metric_definitions': evaluation['metric_definitions'],
        'bootstrap_replicates': evaluation['bootstrap_replicates'],
        'resampling_blocks': evaluation['resampling_blocks'],
        'common_support': evaluation['common_support'],
        'experiments': experiments,
        'training_seed_spread': evaluation['training_seed_spread'],
        'paired_comparisons': evaluation['paired_comparisons'],
        'seed_mean_comparisons': evaluation['seed_mean_comparisons'],
        'factorial_interactions': evaluation['factorial_interactions'],
        'conditional_seed_mean_comparisons': domain_means,
        'domain_denominators': load(root / 'validation_domain_denominators.json'),
        'promotion_decisions': acceptance['promotion_decisions'],
        'accounting': {k: accounting[k] for k in
            ('GPU_hours', 'organization_GPU_hours_upper_bound',
             'organization_budget_GPU_hours', 'remaining_GPU_hours_lower_bound',
             'missing_history_records', 'bounded_allocation_rows', 'cost_interpretation')},
        'calibration_test_used': False,
        'S4_S6_executed': False,
        'source_artifacts': {
            'full_evaluation': str(root / 'evaluation_final/summary.json'),
            'full_conditional_metrics_curves_counts': str(root / 'evaluation_final/conditional/summary.json'),
            'backend_and_shared_allocation_evidence': str(root / 'campaign_accounting.json'),
            'output_acceptance': str(root / 'final_acceptance.json'),
            'figures': [str(root / 'evaluation_final' / name) for name in figures],
        },
        'manual_review': 'scientific interpretation and visual review documented separately in workbook stage77',
    }
    required = {e['name'] for e in acceptance['experiments']}
    if len(required) != 25 or not required.issubset(names):
        raise SystemExit('Unexpected scientific model inventory; no summary written')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(args.output)
    print(json.dumps({'output': str(args.output), 'counts': summary['counts']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
