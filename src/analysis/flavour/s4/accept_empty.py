"""Audit the separately frozen S4 empty-response repair and its paired evidence."""
import argparse
from datetime import datetime, timezone
import math
from pathlib import Path

import pandas as pd

from analysis.flavour.inventory import write_json
from analysis.flavour.s4.accept import fit_record, inventory_checks, load, terminal_allocations


def audit(root, campaign, comparisons, evaluation_path):
    specification = load(campaign)
    configurations = {c['name']: c for c in specification['experiments']}
    comparison = load(comparisons);evaluation = load(evaluation_path)
    evaluated = {r['name']: r for r in evaluation.get('experiments', [])}
    fits = [fit_record(root, c, evaluated) for c in configurations.values()]
    expected = {(kind, budget, seed) for kind in ['scratch', 'mae3']
        for budget in [1000, 10000, 100000] for seed in [17, 42, 2026]}
    actual = {(c['initialization'], c['label_budget'], c['seed']) for c in configurations.values()}
    pretrained = []
    for path in sorted({c['pretrained_checkpoint'] for c in configurations.values() if c.get('pretrained_checkpoint')}):
        directory = Path(path).parent
        manifest = load(directory / 'manifest.json');readback = load(directory / 'ssl_readback.json')
        pretrained.append({'name': directory.name, 'passed': manifest.get('status') == 'complete'
            and readback.get('integrity_and_internal_health_pass') is True
            and readback.get('measured_learning_pass') is True
            and bool(readback.get('checks')) and all(readback['checks'].values())
            and bool(readback.get('learning_checks')) and all(readback['learning_checks'].values())})
    accounting = load(root / 'campaign_accounting.json')
    pilot_name = next(iter(configurations.values())).get('required_pilot')
    pilot = load(root / 'pilot' / pilot_name / 'gate.json')
    stress_path = root / 'stress' / pilot_name
    stress_summary = load(stress_path / 'summary.json')
    stress = pd.read_parquet(stress_path / 'measurements.parquet') if (stress_path / 'measurements.parquet').exists() else pd.DataFrame()
    catalog = pd.read_parquet(root / 'catalog.parquet', columns=['split', 'pixel_channels', 'bar_channels', 'tracker_channels'])
    maximum_occupancy = int(catalog.loc[catalog.split.eq('train'), ['pixel_channels', 'bar_channels', 'tracker_channels']].sum(axis=1).max())
    checks = inventory_checks(configurations, comparison, evaluation)
    checks.update({
        'exact_18_fit_paired_matrix': len(configurations) == 18 and actual == expected,
        'all_use_repaired_route_and_initial_health': all(c.get('empty_response_bypass') is True
            and c.get('architecture_revision') == 3 and c.get('health_initial_steps') == 32 for c in configurations.values()),
        'real_train_memorization_gate': pilot.get('gate_passed') is True
            and pilot.get('config', {}).get('empty_response_bypass') is True,
        'BF16_empty_and_maximum_occupancy_stress': stress_summary.get('precision') == 'BF16'
            and len(stress) == 6 and bool(stress.finite.all())
            and set(stress['mode']) == {'inference', 'backward'}
            and int(stress.raw_channels.max()) == maximum_occupancy and int(stress.raw_channels.min()) == 0
            and bool(stress.raw_channels.eq(stress.represented_nodes).where(stress.raw_channels.gt(0), True).all())
            and bool((stress.loc[stress['mode'].eq('backward') & stress.raw_channels.gt(0), 'input_gradient_squared_norm'] > 0).all())
            and bool((stress.loc[stress['mode'].eq('backward') & stress.raw_channels.eq(0), 'input_gradient_squared_norm'] == 0).all()),
        'all_fits_checkpoints_and_internal_health': bool(fits) and all(r['passed'] for r in fits),
        'three_measured_pretraining_sources': len(pretrained) == 3 and all(r['passed'] for r in pretrained),
        'complete_balanced_10k_validation': len(evaluated) == 18 and all(r['four_class']['events'] == 10000
            and r['four_class']['class_weight_sums'] == [2500.] * 4 for r in evaluated.values()),
        'complete_validation_predictions_vary': len(evaluated) == 18 and all(
            r['prediction_diagnostics']['all_events_same_probability_vector'] is False for r in evaluated.values()),
        'physical_block_bootstrap': evaluation.get('bootstrap_replicates') == 400
            and evaluation.get('resampling_blocks', {}).get('resampling_blocks') == 128
            and evaluation.get('resampling_blocks', {}).get('accepted_family_links_used') is True,
        'tau_branches_and_common_support': len(evaluation.get('paired_comparisons', [])) == 9
            and all(set(p.get('metrics', {})) == {'raw', 'common_support'}
                and set(p.get('tau_branches', {})) == {b + '__' + d for b in ['electron', 'muon', 'hadronic']
                    for d in ['raw', 'common_support']} for p in evaluation.get('paired_comparisons', [])),
        'training_health_figures_present': all((evaluation_path.parent / filename).is_file() for filename in
            ['classification_training_curves.pdf', 'classification_internal_health.pdf', 'classification_health_ranges.csv']),
        'all_allocations_terminal': terminal_allocations(accounting),
        'within_original_60_device_hours': 0 < accounting.get('organization_device_hours', math.inf) <= 60,
        'calibration_test_unused': specification.get('calibration_test_used') is False,
    })
    return {'utc': datetime.now(timezone.utc).isoformat(), 'stage': 'S4_empty_response_repair', 'checks': checks,
        'technical_execution_passed': all(checks.values()), 'fits': fits, 'pretraining_sources': pretrained,
        'organization_device_hours': accounting.get('organization_device_hours'),
        'probability_quality': {r['name']: {'validation_NLL': r['four_class']['NLL'],
            'NLL_minus_balanced_constant_prior': r['four_class']['NLL'] - math.log(4)} for r in evaluated.values()},
        'scope': 'technical completion and numerical health; scientific improvement is decided separately from paired AP, NLL, tau metrics and their intervals; original S4 fits are excluded',
        'designed_after_observing_original_S4_validation': True, 'calibration_test_used': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--comparisons', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.root, args.campaign, args.comparisons, args.evaluation)
    write_json(args.output, report)
    print(report['checks'])
    if not report['technical_execution_passed']:
        raise SystemExit('empty-response repair has outstanding execution or evidence checks')


if __name__ == '__main__':
    main()
