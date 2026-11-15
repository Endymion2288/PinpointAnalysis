"""Check the measured R1 deliverables, separately from scientific promotion."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.flavour.inventory import write_json
from analysis.flavour.v2.partitions import KEY, equal_identity_rows
from analysis.flavour.v2.r1.metrics import METRICS_REVISION
from analysis.flavour.v2.r1.deliver import evaluation_config


DOMAINS = {'all', 'pixel_centered', 'large_scintillator', 'active', 'empty',
           'tracker_active', 'tracker_empty', 'occupancy_0_100', 'occupancy_100_1000',
           'occupancy_1000_5000', 'occupancy_5000_inf'}


def delivery_contract_checks(config, expected, evaluation):
    """Require the frozen matrix, even when a smaller result is internally valid."""
    checks = []
    def check(name, passed, **details):
        checks.append({'name': name, 'passed': bool(passed), **details})
    check('configuration_preserves_frozen_model_and_pair_matrix',
          config['models'] == expected['models'] and config['pairs'] == expected['pairs'],
          expected_models=len(expected['models']), expected_pairs=len(expected['pairs']))
    expected_names = {e['name'] for e in expected['models']}
    check('evaluation_contains_exactly_all_frozen_models', set(evaluation['models']) == expected_names)
    def pair_key(row):
        return tuple(row.get(key) for key in ['a', 'b', 'domain', 'reason'])
    check('evaluation_contains_all_frozen_pairs_once',
          Counter(map(pair_key, evaluation['pairs'])) == Counter(map(pair_key, expected['pairs'])))
    check('all_frozen_models_have_all_eleven_domains',
          all(set(evaluation['models'].get(name, {})) == DOMAINS for name in expected_names))
    entries = {row['name']: row for row in expected['models']}
    groups = defaultdict(list)
    for pair in expected['pairs']:
        a, b = entries[pair['a']], entries[pair['b']]
        if a['seed'] == b['seed']:
            group = tuple((row['family'], row['task'], row.get('kind', 'main')) for row in [a, b])
            groups[(*group, pair['domain'], pair['reason'])].append(a['seed'])
    required = Counter(key for key, seeds in groups.items() if sorted(seeds) == [17, 42, 2026])
    actual = evaluation.get('mean_seed_pairs', [])
    keys = Counter((tuple(row['a_family_task_kind']), tuple(row['b_family_task_kind']),
                    row['domain'], row['reason']) for row in actual)
    check('all_three_seed_pairs_present_once_with_exact_seeds', keys == required and all(
        sorted(row['seeds']) == [17, 42, 2026] for row in actual), expected_mean_seed_pairs=sum(required.values()))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    checks = []
    def check(name, value, **details):
        checks.append({'name': name, 'passed': bool(value), **details})
    preparation = json.loads((root/'preparation.json').read_text())
    check('complete_frozen_data', preparation['status']=='complete' and preparation['events']==140000
          and preparation['all_event_family_keys_exact'] and preparation['PE_conserved'])
    health = json.loads((root/'fit/all_health_k10/summary.json').read_text())
    check('all_eight_real_learning_gates', health['status']=='complete' and len(health['cases'])==8)
    cnn_health = json.loads((root/'fit/cnn_cost_control/summary.json').read_text())
    check('plain_CNN_real_learning_gates', cnn_health['status']=='complete' and len(cnn_health['cases'])==2)
    contract = json.loads((root/'campaign.json').read_text())
    for family in contract['families']:
        for task in contract['tasks']:
            for lr in contract['learning_rates'][family]:
                name = f'screen_{family.lower()}_{task.lower()}_lr{lr:g}_s42'
                record = json.loads((root/'experiments'/name/'manifest.json').read_text())
                seconds = record['counters']['train_seconds']
                check(f'{name}_complete', record['status']=='complete' and seconds>=300., train_seconds=seconds)
    config = json.loads((root/'evaluation_config.json').read_text())
    check('plain_CNN_T3_T4_cost_references', {e['task'] for e in config['models']
          if e.get('kind')=='cnn_cost_control'}=={'T3', 'T4'})
    catalog = pd.read_parquet(root/'data/catalog.parquet')
    tune = catalog[catalog.split.eq('validation')]
    exposure_rows = []
    for entry in config['models']:
        folder = root/'experiments'/entry['name']
        record = json.loads((folder/'manifest.json').read_text())
        predictions = pd.read_parquet(folder/'predictions.parquet')
        check(entry['name']+'_Tune_identities', equal_identity_rows(predictions, tune))
        check(entry['name']+'_complete', record['status']=='complete' and not record['Cal_A_Cut_B_Confirm_payload_read'])
        p = predictions[[f'p{c}' for c in ([0,1,2,3] if entry['task']=='T4' else [0,1,3])]].to_numpy()
        check(entry['name']+'_probabilities', np.isfinite(p).all() and np.allclose(p.sum(1), 1., atol=2e-6))
        if entry['family']=='B0':
            check(entry['name']+'_actual_T3_training', record['unique_train_events']==(75000 if entry['task']=='T3' else 100000))
            continue
        coverage = pd.read_parquet(folder/'train_exposures.parquet')
        expected = catalog[catalog.split.eq('train') & (catalog.label.ne(2) if entry['task']=='T3' else True)]
        check(entry['name']+'_Train_identities', equal_identity_rows(coverage, expected))
        check(entry['name']+'_exposure_sum', int(coverage.exposures.sum())==record['counters']['training_exposures'])
        if entry.get('kind', 'main')!='matched_compute':
            check(entry['name']+'_four_complete_passes', coverage.exposures.eq(4).all())
        else:
            target = record['config']['total_compute_seconds']
            difference = record['wall_seconds']/target-1.
            check(entry['name']+'_compute_alignment', abs(difference)<=.05, relative_difference=difference)
            reference_name = record['config']['compute_reference']
            reference = json.loads((root/'experiments'/reference_name/'manifest.json').read_text())
            check(entry['name']+'_same_allocation', bool(record.get('allocation_id'))
                  and record['allocation_id']==reference.get('allocation_id')
                  and record['GPU']==reference['GPU'],
                  allocation_id=record.get('allocation_id'), reference=reference_name)
            check(entry['name']+'_reference_budget', np.isclose(target, reference['wall_seconds']))
        check(entry['name']+'_finite_activations', all(v['finite'] for h in record['health'] for v in h['activations'].values()))
        groups = record['health'][0]['gradient_squared_norms']
        active = [g for g in groups if not (g=='tracker' and entry.get('kind')=='without_tracker')]
        check(entry['name']+'_branch_gradients_and_updates', all(
            any(np.isfinite(h['gradient_squared_norms'][g]) and h['gradient_squared_norms'][g]>0 for h in record['health'])
            and any(h['parameter_update_RMS'][g]>0 for h in record['health']) for g in active))
        check(entry['name']+'_nonconstant_predictions', record['probability_diagnostics']['distinct_vectors']>=100)
        if entry.get('kind')=='without_tracker':
            reference = json.loads((root/'experiments'/f"final_{entry['family'].lower()}_t3_s42"/'manifest.json').read_text())
            check(entry['name']+'_matched_training_precision', record['config'].get('bf16', True)==
                  reference['config'].get('bf16', True))
        exposure_rows.append({'name': entry['name'], 'family': entry['family'], 'task': entry['task'],
            'seed': entry['seed'], 'kind': entry.get('kind', 'main'),
            'unique_rows': record['unique_train_events'], 'input_rows': len(coverage),
            'minimum_exposures': int(coverage.exposures.min()), 'maximum_exposures': int(coverage.exposures.max()),
            'total_exposures': int(coverage.exposures.sum()), 'updates': record['counters']['updates'],
            'GPU': record['GPU'], 'precision': record['config'].get('precision'),
            'wall_seconds': record['wall_seconds'], 'training_seconds': record['counters']['train_seconds'],
            'validation_seconds': record['counters']['validation_seconds']})
    evaluation = json.loads((root/'evaluation/results.json').read_text())
    expected_config = evaluation_config(root, write=False)
    checks.extend(delivery_contract_checks(config, expected_config, evaluation))
    review = json.loads((root/'evaluation/independent_review.json').read_text())
    check('independent_full_probability_count_exposure_review',
          review['status']=='complete' and review['require_full'] and not review['failed'])
    interval_review = json.loads((root/'evaluation/independent_core_intervals.json').read_text())
    check('independent_core_interval_and_seed_SD_review', interval_review['status']=='complete'
          and not interval_review['failed'] and interval_review['replicates']==400
          and interval_review['pairs']==len(expected_config['pairs'])
          and interval_review['mean_seed_pairs']==21)
    stress_review = json.loads((root/'evaluation/independent_stress_review.json').read_text())
    check('independent_stress_point_and_fixed_cut_review', stress_review['status']=='complete'
          and stress_review['model_conditions']==55 and not stress_review['failed'])
    check('400_paired_physical_block_bootstraps', evaluation['status']=='complete' and evaluation['replicates']==400)
    check('roundoff_aware_operating_points', evaluation.get('metrics_revision')==METRICS_REVISION)
    for entry in [e for e in config['models'] if e['task']=='T4']:
        scenarios = evaluation['models'][entry['name']]['all'].get('T4_tau_prior_scenarios', {})
        check(entry['name']+'_fixed_score_tau_prior_scenarios', set(scenarios)=={'0.01', '0.001'}
              and all(set(row['prior_scenarios'])=={'0.25', '0.01', '0.001'} for row in scenarios.values()))
    ranking = json.loads((root/'architecture_ranking.json').read_text())
    check('Scratch_ranking_and_expansion', set(ranking['expanded_families'])==
          {r['family'] for r in ranking['ranking'][:2]}|{'B4'})
    model_keys = {(e['family'], e['task'], e['seed'], e.get('kind', 'main')) for e in config['models']}
    for family in contract['families']:
        for seed in ([17,42,2026] if family in ranking['expanded_families'] else [42]):
            check(f'{family}_s{seed}_complete_tasks', all((family, task, seed, kind) in model_keys
                  for task, kind in [('T3', 'main'), ('T4', 'main'), ('T3', 'matched_compute')]))
    spread = {(row['family'], row['task']): row for row in evaluation['seed_spread']}
    for family in ranking['expanded_families']+['B0']:
        for task in ['T3', 'T4']:
            row = spread.get((family, task), {})
            check(f'{family}_{task}_three_seeds', set(row.get('seeds', []))=={17,42,2026})
    stress = json.loads((root/'evaluation/stress_results.json').read_text())
    check('400_paired_stress_bootstraps', stress['status']=='complete' and stress['replicates']==400)
    check('roundoff_aware_stress_operating_points', stress.get('metrics_revision')==METRICS_REVISION)
    stress_pairs = [p for p in config['pairs'] if p.get('domain', 'all')=='all'
                    and p['a'] in stress['models'] and p['b'] in stress['models']]
    check('paired_stress_model_condition_contrasts', len(stress.get('paired_model_comparisons', []))==len(stress_pairs)
          and all(set(p['conditions'])=={'nominal', 'noise', 'combined', 'remove_pixel', 'remove_tracker'}
                  for p in stress.get('paired_model_comparisons', [])))
    for entry in [e for e in config['models'] if e.get('kind', 'main')=='main' and e['seed']==42]:
        report = stress['models'].get(entry['name'], {})
        check(entry['name']+'_all_stress_conditions', set(report.get('conditions', {}))==
              {'nominal', 'noise', 'combined', 'remove_pixel', 'remove_tracker'})
    figures = ['r1_common_three', 'r1_t3_roc_pr', 'r1_t4_roc_pr', 'r1_tau_roc_pr',
               'r1_four_class_roc_pr', 'r1_confusion',
               'r1_raw_scores', 'r1_resource_pareto', 'r1_learning_curves', 'r1_stress', 'r1_matched_compute']
    for name in figures:
        check(name+'_figures', all((root/'figures'/f'{name}.{ext}').is_file() for ext in ['pdf', 'png']))
    resources = json.loads((root/'resources.json').read_text())
    check('all_allocations_terminal_and_accounted', not resources['active_or_unresolved'],
          active_or_unresolved=resources['active_or_unresolved'])
    check('heterogeneous_device_costs_explicit', resources['heterogeneous_device_hours_are_not_FLOP_equivalent'])
    pd.DataFrame(exposure_rows).to_csv(root/'evaluation/exposure_and_compute.csv', index=False)
    failed = [c['name'] for c in checks if not c['passed']]
    write_json(root/'acceptance.json', {'status': 'complete' if not failed else 'failed',
        'checks': checks, 'passed': len(checks)-len(failed), 'failed': failed,
        'scope': 'R1 baseline comparison delivery; scientific promotion and global optimality are separate questions',
        'Cal_A_Cut_B_Confirm_payload_read': False})
    print(json.dumps({'passed': len(checks)-len(failed), 'failed': failed}), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
