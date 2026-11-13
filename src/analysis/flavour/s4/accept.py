"""Audit the frozen S4 matrix, measured learning and terminal resource evidence."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from analysis.flavour.inventory import write_json


def load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def program(config):
    return config.get('program', 'train' if config.get('objective') == 'classification' else 'pretrain')


def same_configuration(expected, actual):
    return {**expected, 'program': program(expected)} == {**actual, 'program': program(actual)}


def training_health(records, total_steps, every, initial_steps=0, maximum_preclip_norm=10000.):
    informative = [r for r in records if r.get('informative_nodes', 0) > 0]
    activations = [v for r in records for rows in r.get('activations', {}).values() for v in rows]
    silu = [v for r in records for v in r.get('activations', {}).get('head.2', [])]
    return {
        'scheduled_samples_present': total_steps > 0 and [r['step'] for r in records] == sorted(
            set(range(1, min(initial_steps, total_steps) + 1)) | set(range(1, total_steps + 1, every))),
        'health_decisions_pass': bool(records) and all(r.get('healthy') is True for r in records),
        'finite_sampled_activations': bool(activations) and all(r['all_finite'] == 1 for r in activations),
        'informative_input_gradients': bool(informative) and all(
            math.isfinite(r['gradient_squared_norms'].get('input', 0.))
            and r['gradient_squared_norms'].get('input', 0.) > 0 for r in informative),
        'SiLU_never_entirely_zero': len(silu) >= len(records) > 0 and all(v['zero_fraction'] < 1. for v in silu),
        'preclip_gradients_within_review_limit': bool(records) and all(
            math.isfinite(sum(r.get('gradient_squared_norms', {}).values()))
            and sum(r.get('gradient_squared_norms', {}).values()) <= maximum_preclip_norm ** 2 for r in records),
    }


def fit_record(root, config, evaluated):
    directory = root / 'experiments' / config['name']
    manifest = load(directory / 'manifest.json')
    checks = {'complete': manifest.get('status') == 'complete',
        'frozen_configuration': same_configuration(config, manifest.get('config', {})),
        'calibration_test_unused': manifest.get('calibration_test_used') is False,
        'evaluated': config['name'] in evaluated}
    if program(config) == 'train':
        readback = load(directory / 'final_checkpoint_checks.json')
        checks['checkpoint_readback'] = (readback.get('status') == 'pass'
            and len(readback.get('checks', {})) == 14 and all(readback['checks'].values())
            and readback.get('best_epoch') == manifest.get('best_epoch'))
        best_health = load(directory / 'best_checkpoint_health.json')
        checks['independent_best_FP32_backward'] = (best_health.get('version') == 1
            and best_health.get('passed') is True and bool(best_health.get('checks'))
            and all(best_health['checks'].values()) and best_health.get('config') == manifest.get('config')
            and best_health.get('best_epoch') == manifest.get('best_epoch')
            and best_health.get('selected_NLL') == manifest.get('best_NLL'))
        records = load(directory / 'training_health.json').get('records', [])
        checks.update(training_health(records, manifest.get('optimizer_steps', 0), config.get('health_every_steps', 100),
            config.get('health_initial_steps', 0), config.get('maximum_preclip_gradient_norm', 10000.)))
        if config.get('empty_response_bypass'):
            every_step=manifest.get('gradient_health',{})
            norm=every_step.get('maximum_preclip_gradient_norm',float('inf'))
            checks['every_update_gradient_review']=(every_step.get('checked_updates')==manifest.get('optimizer_steps',0)
                and math.isfinite(norm) and norm<=config.get('maximum_preclip_gradient_norm',10000.))
        samples = len(records)
    else:
        statistics = load(directory / 'feature_statistics.json')
        checks['frozen_fit_and_validation_counts'] = (manifest.get('frozen_encoder') is True
            and manifest.get('fit_events') == config['label_budget'] and manifest.get('probe_events') == 10000)
        checks['finite_full_and_global_features'] = all(statistics.get(branch, {}).get(split, {}).get('all_finite') is True
            for branch in ['full', 'global'] for split in ['fit', 'probe'])
        checks['global_probe_evaluated'] = config['name'] + '__global' in evaluated
        samples = 0
    return {'name': config['name'], 'program': program(config), 'checks': checks, 'health_samples': samples,
        'passed': all(checks.values())}


def terminal_allocations(accounting):
    rows = accounting.get('allocations', [])
    return bool(rows) and all(row.get('status') == 'terminal' for row in rows)


def inventory_checks(configurations, specification, evaluation):
    fits = [c for c in configurations.values() if program(c) in {'train', 'probes'}
        and c.get('probe_scope') != 'train_only_branches']
    expected = {c['name'] for c in fits}
    if specification.get('include_global_frozen'):
        expected.update(c['name'] + '__global' for c in fits if program(c) == 'probes')
    if specification.get('include_supervised_selection_envelope'):
        expected.update(c['name'] + '__selection' for c in fits if c.get('supervised_initial_checkpoint'))
    actual = [e['name'] for e in evaluation.get('experiments', [])]
    pairs = [(p['a'], p['b']) for p in evaluation.get('paired_comparisons', [])]
    expected_pairs = [(p['a'], p['b']) for p in specification.get('pairs', [])]
    means = [p['name'] for p in evaluation.get('seed_mean_comparisons', [])]
    expected_means = [p['name'] for p in specification.get('seed_mean_comparisons', [])]
    return {
        'requested_matrix_complete': bool(expected) and len(actual) == len(expected) and set(actual) == expected,
        'all_frozen_pairs_complete': bool(expected_pairs) and len(pairs) == len(expected_pairs) and set(pairs) == set(expected_pairs),
        'all_frozen_seed_means_complete': bool(expected_means) and len(means) == len(expected_means) and set(means) == set(expected_means),
        'strict_validation_has_no_pending': evaluation.get('requested_complete') is True and evaluation.get('pending') == [],
    }


def audit(root, campaigns, comparisons, evaluation_path, paths_path):
    configurations = {c['name']: c for p in campaigns for c in load(p)['experiments']}
    specification = load(comparisons)
    evaluation, functional = load(evaluation_path), load(paths_path)
    accounting = load(root / 'campaign_accounting.json')
    evaluated = {e['name']: e for e in evaluation.get('experiments', [])}
    records = [fit_record(root, c, evaluated) for c in configurations.values()
        if program(c) in {'train', 'probes'} and c.get('probe_scope') != 'train_only_branches']
    pretrained = []
    for c in configurations.values():
        if program(c) != 'pretrain':
            continue
        directory = root / 'experiments' / c['name']
        manifest, readback = load(directory / 'manifest.json'), load(directory / 'ssl_readback.json')
        pretrained.append({'name': c['name'], 'passed': manifest.get('status') == 'complete'
            and same_configuration(c, manifest.get('config', {}))
            and readback.get('integrity_and_internal_health_pass') is True and bool(readback.get('checks'))
            and all(readback['checks'].values()) and readback.get('measured_learning_pass') is True
            and bool(readback.get('learning_checks')) and all(readback['learning_checks'].values())})
    expected_paths = {f's4_{kind}_n{budget}_s42' for kind in ['scratch', 'mae3'] for budget in [1000, 10000]}
    path_rows = functional.get('experiments', [])
    checks = inventory_checks(configurations, specification, evaluation)
    checks.update({
        'all_fits_configs_checkpoints_and_internal_health': bool(records) and all(r['passed'] for r in records),
        'all_full_pretraining_measured_learning': bool(pretrained) and all(r['passed'] for r in pretrained),
        'complete_10k_balanced_validation': bool(evaluated) and all(e['four_class']['events'] == 10000
            and e['four_class']['class_weight_sums'] == [2500., 2500., 2500., 2500.] for e in evaluated.values()),
        'predictions_nonconstant': bool(evaluated) and all(e['prediction_diagnostics']['all_events_same_probability_vector'] is False
            for e in evaluated.values()),
        'physical_block_bootstrap': evaluation.get('bootstrap_replicates') == 400
            and evaluation.get('resampling_blocks', {}).get('resampling_blocks') == 128
            and evaluation.get('resampling_blocks', {}).get('accepted_family_links_used') is True,
        'raw_common_support_and_tau_contrasts': all(set(p.get('metrics', {})) == {'raw', 'common_support'}
            and set(p.get('tau_branches', {})) == {b + '__' + d for b in ['electron', 'muon', 'hadronic'] for d in ['raw', 'common_support']}
            for p in evaluation.get('paired_comparisons', []) + evaluation.get('seed_mean_comparisons', [])),
        'classwise_finite_MC_counts': bool(evaluated) and all(len(e.get('classwise_operating_points', {})) == 4
            and all(set(points) == {'FPR_0.01', 'FPR_0.001'} and all(p['signal_events'] == 2500
                and p['background_events'] == 7500 and 'background_count_interval_95' in p for p in points.values())
                for points in e['classwise_operating_points'].values()) for e in evaluated.values()),
        'complete_fixed_weight_path_evidence': len(path_rows) == len(expected_paths)
            and {p['experiment'] for p in path_rows} == expected_paths
            and functional.get('calibration_test_used') is False and functional.get('fitting_or_selection') is False
            and all(p['events'] == 10000 and p['cached_head_matches_original'] is True
                and p['cached_head_logit_maximum_error'] <= 2e-6 and p['presence_preserved'] is True
                and p.get('head_replay_batch_shapes_and_recipient_order_preserved') is True
                and p['permutation_seeds'] == [2026, 17, 42] for p in path_rows),
        'terminal_allocation_evidence': terminal_allocations(accounting),
        'organization_budget': 0 < accounting.get('organization_device_hours', 0.) <= accounting.get('cap_device_hours', 0.) == 60,
        'plots_present': all((evaluation_path.parent / name).is_file() for name in
            ['label_efficiency.pdf', 'label_efficiency_train_full.png', 'label_efficiency_probes_full.png',
             'label_efficiency_probes_global.png', 'four_class_diagnostics.pdf']),
        'training_and_health_plots_present': all((evaluation_path.parent / name).is_file() for name in
            ['classification_training_curves.pdf','classification_internal_health.pdf','classification_health_ranges.csv']),
    })
    return {'utc': datetime.now(timezone.utc).isoformat(), 'stage': 'S4', 'checks': checks,
        'strict_output_gate_passed': all(checks.values()), 'fits': records, 'pretraining': pretrained,
        'health_samples': sum(r['health_samples'] for r in records),
        'organization_device_hours': accounting.get('organization_device_hours'),
        'calibration_test_used': False,
        'scope': 'frozen S4 execution and measured-learning evidence; completion does not require promotion or improved NLL at every label budget',
        'evaluation_source': str(evaluation_path), 'paths_source': str(paths_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, nargs='+', required=True)
    parser.add_argument('--comparisons', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    report = audit(args.root, args.campaign, args.comparisons, args.evaluation, args.paths)
    write_json(args.output or args.root / 'final_acceptance.json', report)
    print(json.dumps({k: report[k] for k in ['utc', 'strict_output_gate_passed', 'checks', 'health_samples']}))
    if args.require_complete and not report['strict_output_gate_passed']:
        raise SystemExit('S4 evidence is incomplete; inspect final_acceptance.json')


if __name__ == '__main__':
    main()
