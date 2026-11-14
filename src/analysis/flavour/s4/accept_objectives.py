"""Audit the valid S4 objective cohort and stationary readouts from saved evidence."""
import argparse
from datetime import datetime, timezone
import math
from pathlib import Path

from analysis.flavour.inventory import write_json
from analysis.flavour.s4.accept import fit_record, inventory_checks, load, program, same_configuration, terminal_allocations


def probe_record(root, config, replay_summary):
    directory = root / 'experiments' / config['name']
    optimization = load(directory / 'probe_optimization.json')
    replay = replay_summary.get('artifact_prediction_replay') or {}
    branches = config.get('probe_branches', ['full', 'global'])
    checks = {
        'complete_optimization_inventory': set(optimization) == {b + '_linear' for b in branches},
        'replay_matches_saved_optimization': replay_summary.get('branches') == optimization,
        'complete_replay_inventory': set(replay) == set(branches),
        'fit_counts': replay_summary.get('fit_events') == config['label_budget']
            and replay_summary.get('validation_events') == 10000,
    }
    for branch in branches:
        row = optimization.get(branch + '_linear', {})
        gradient = row.get('gradient_infinity_norm', float('inf'))
        checks[branch + '_stationary_same_objective'] = (
            row.get('passed') is True and row.get('stationarity_passed') is True
            and row.get('stationarity_required') is True and row.get('finite_coefficients') is True
            and row.get('convergence_warnings') == [] and row.get('solver') == config['probe_linear_solver']
            and row.get('C') == 1. and row.get('fit_events') == config['label_budget']
            and math.isfinite(gradient) and 0 <= gradient <= config['probe_linear_tolerance'] + 1e-10)
        saved = replay.get(branch, {})
        error = saved.get('maximum_probability_difference', float('inf'))
        checks[branch + '_saved_probability_replay'] = (saved.get('passed') is True
            and saved.get('events') == 10000 and saved.get('absolute_tolerance') == 1e-10
            and math.isfinite(error) and 0 <= error <= 1e-10)
    return {'name': config['name'], 'checks': checks, 'passed': all(checks.values())}


def audit(root, campaign, comparisons, evaluation_path, validation_name):
    configurations = {c['name']: c for c in load(campaign)['experiments']}
    specification, evaluation = load(comparisons), load(evaluation_path)
    evaluated = {e['name']: e for e in evaluation.get('experiments', [])}
    validation = root / 'experiments' / validation_name
    manifest = load(validation / 'manifest.json')
    summaries = load(validation / 'probe_optimization_summary.json').get('models', [])
    by_name = {r['name']: r for r in summaries}
    probes = [probe_record(root, c, by_name.get(c['name'], {})) for c in configurations.values()
        if program(c) == 'probes']
    acceleration_gates = []
    for name in sorted({c['required_acceleration_gate'] for c in configurations.values() if c.get('required_acceleration_gate')}):
        gate = load(root / 'experiments' / name / 'solver_probe.json')
        cases = gate.get('cases', [])
        acceleration_gates.append({'name': name, 'passed': gate.get('passed') is True and len(cases) == 8
            and all(r.get('passed') is True and r.get('absolute_objective_difference', float('inf')) <= 1e-5
                and r.get('maximum_fit_probability_difference', float('inf')) <= .005
                and r.get('CPU_reference', {}).get('gradient_infinity_norm', float('inf')) <= 1e-6 + 1e-10
                and r.get('candidate', {}).get('gradient_infinity_norm', float('inf')) <= 1e-6 + 1e-10 for r in cases)})
    fits = [fit_record(root, c, evaluated) for c in configurations.values()]
    pretrained = []
    for name in sorted({name for c in configurations.values() for name in c.get('required_ssl_readbacks', [])}):
        directory = root / 'experiments' / name
        readback = load(directory / 'ssl_readback.json')
        pretrained.append({'name': name, 'passed': load(directory / 'manifest.json').get('status') == 'complete'
            and readback.get('integrity_and_internal_health_pass') is True
            and bool(readback.get('checks')) and all(readback['checks'].values())
            and readback.get('measured_learning_pass') is True
            and bool(readback.get('learning_checks')) and all(readback['learning_checks'].values())})
    updates = []
    for c in configurations.values():
        if program(c) != 'train' or not c.get('transfer_recipe'):
            continue
        row = load(root / 'experiments' / c['name'] / 'parameter_update_audit.json')
        updates.append({'name': c['name'], 'passed': row.get('passed') is True
            and same_configuration(c, row.get('config', {}))
            and len(row.get('checks', {})) == 6 and all(row['checks'].values())})
    accounting = load(root / 'campaign_accounting.json')
    checks = inventory_checks(configurations, specification, evaluation)
    expected_probes = {c['name'] for c in configurations.values() if program(c) == 'probes'}
    frozen = manifest.get('config', {})
    frozen_campaigns = {c['name']: c for path in frozen.get('campaigns', []) for c in load(Path(path)).get('experiments', [])}
    solver_gates = []
    for name in sorted({c['required_solver_probe'] for c in configurations.values() if program(c) == 'probes'}):
        gate = load(root / 'experiments' / name / 'solver_probe.json')
        cases = gate.get('cases', [])
        solver_gates.append({'name': name, 'passed': gate.get('passed') is True and len(cases) == 4
            and all(r.get('passed') is True and r.get('absolute_objective_difference', float('inf')) <= 1e-5
                and r.get('maximum_fit_probability_difference', float('inf')) <= .005
                and all(r.get(solver, {}).get('passed') is True
                    and r[solver].get('gradient_infinity_norm', float('inf')) <= tolerance
                    for solver, tolerance in [('LBFGS', 1e-8), ('Newton_CG', 1e-6)]) for r in cases)})
    contrasts = evaluation.get('paired_comparisons', []) + evaluation.get('seed_mean_comparisons', [])
    checks.update({
        'frozen_evaluation_inputs': frozen_campaigns == configurations and bool(frozen.get('comparisons'))
            and load(Path(frozen['comparisons'])) == specification and frozen.get('strict') is True,
        'validation_worker_complete': manifest.get('status') == 'complete'
            and manifest.get('pending_train') == [] and manifest.get('pending_fits') == []
            and set(manifest.get('verified_train', [])) == {c['name'] for c in configurations.values() if program(c) == 'train'},
        'all_fits_configs_checkpoints_and_internal_health': bool(fits) and all(r['passed'] for r in fits),
        'actual_encoder_parameter_updates': bool(updates) and all(r['passed'] for r in updates),
        'all_required_pretraining_measured_learning': bool(pretrained) and all(r['passed'] for r in pretrained),
        'all_stationary_readouts_replayed': bool(probes) and len(summaries) == len(expected_probes)
            and set(by_name) == expected_probes and all(r['passed'] for r in probes),
        'same_objective_solver_regressions': bool(solver_gates) and all(r['passed'] for r in solver_gates),
        'device_acceleration_regressions': all(r['passed'] for r in acceleration_gates),
        'complete_10k_balanced_validation': bool(evaluated) and all(e['four_class']['events'] == 10000
            and e['four_class']['class_weight_sums'] == [2500.] * 4 for e in evaluated.values()),
        'predictions_nonconstant': bool(evaluated) and all(
            e.get('prediction_diagnostics', {}).get('all_events_same_probability_vector') is False for e in evaluated.values()),
        'physical_block_bootstrap': evaluation.get('bootstrap_replicates') == 400
            and evaluation.get('resampling_blocks', {}).get('resampling_blocks') == 128
            and evaluation.get('resampling_blocks', {}).get('accepted_family_links_used') is True,
        'raw_common_support_and_tau_contrasts': bool(contrasts) and all(
            set(p.get('metrics', {})) == {'raw', 'common_support'}
            and set(p.get('tau_branches', {})) == {b + '__' + d for b in ['electron', 'muon', 'hadronic']
                for d in ['raw', 'common_support']} for p in contrasts),
        'classwise_finite_MC_counts': bool(evaluated) and all(len(e.get('classwise_operating_points', {})) == 4
            and all(set(points) == {'FPR_0.01', 'FPR_0.001'} and all(p['signal_events'] == 2500
                and p['background_events'] == 7500 and 'background_count_interval_95' in p
                for p in points.values()) for points in e['classwise_operating_points'].values()) for e in evaluated.values()),
        'terminal_allocation_evidence': terminal_allocations(accounting),
        'organization_budget': 0 < accounting.get('organization_device_hours', 0.) <= accounting.get('cap_device_hours', 0.) == 60,
        'plots_present': all((evaluation_path.parent / name).is_file() for name in
            ['label_efficiency.pdf', 'label_efficiency_train_full.png', 'label_efficiency_probes_full.png',
             'label_efficiency_probes_global.png', 'four_class_diagnostics.pdf',
             'classification_training_curves.pdf', 'classification_internal_health.pdf', 'classification_health_ranges.csv']),
    })
    return {'utc': datetime.now(timezone.utc).isoformat(), 'checks': checks,
        'strict_output_gate_passed': all(checks.values()), 'fits': fits, 'probes': probes,
        'pretraining': pretrained, 'parameter_updates': updates, 'solver_regressions': solver_gates,
        'acceleration_regressions': acceleration_gates,
        'health_samples': sum(r['health_samples'] for r in fits),
        'organization_device_hours': accounting.get('organization_device_hours'),
        'CPU_allocation_hours': accounting.get('CPU_allocation_hours'),
        'calibration_test_used': False,
        'scope': 'completed frozen objective cohort, numerical and artifact evidence; '
            'technical completion does not establish probability generalization or method promotion',
        'sources': {'campaign': str(campaign), 'comparisons': str(comparisons),
            'evaluation': str(evaluation_path), 'validation': str(validation),
            'accounting': str(root / 'campaign_accounting.json')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--comparisons', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--validation-name', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    report = audit(args.root, args.campaign, args.comparisons, args.evaluation, args.validation_name)
    write_json(args.output, report)
    print({k: report[k] for k in ['strict_output_gate_passed', 'checks', 'health_samples']})
    if args.require_complete and not report['strict_output_gate_passed']:
        raise SystemExit('Objective cohort evidence is incomplete; inspect the saved audit')


if __name__ == '__main__':
    main()
