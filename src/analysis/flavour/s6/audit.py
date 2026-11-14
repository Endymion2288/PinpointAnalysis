"""Map the original S6 contract to direct measured artifacts and open work."""
import argparse
import json
from pathlib import Path

import numpy as np

from analysis.flavour.inventory import write_json
from analysis.flavour.s6.calibration import model_names


def run(args):
    root = args.root
    freeze = json.loads((root/'analysis_freeze.json').read_text())
    requirements = []

    def record(name, status, evidence, facts):
        requirements.append({'requirement': name, 'status': status, 'evidence': evidence, 'facts': facts})

    def read(path):
        candidate = root/path
        return json.loads(candidate.read_text()) if candidate.exists() else None

    calibration = read('calibration/completion.json')
    record('all accepted calibration, temperature and validation work points',
        'complete' if calibration and calibration['events'] == 722272 and len(freeze['calibrators']) == 15 else 'open',
        ['analysis_freeze.json', 'calibration/completion.json'], calibration)
    test = read('predictions/test_completion.json')
    record('all accepted independent test scores with empty responses retained',
        'complete' if test and test['events'] == 724441 else 'open', ['predictions/test_completion.json'], test)
    expected = [(n, s['name']) for n in model_names(freeze['candidates']) for s in freeze['candidates']['prior_scenarios']]
    outputs = []
    missing = []
    for name, scenario in expected:
        path = Path('evaluation/models')/name/scenario/'result.json'
        result = read(path)
        if result is None:
            missing.append(str(path))
        else:
            outputs.append(result)
    valid = len(outputs) == 60 and all(v['events'] == 724441 and not v['thresholds_selected_on_test'] for v in outputs)
    record('four-class confusion, ROC/PR, calibrated priors, fixed selections, exact tail counts and cutflows',
        'complete' if valid else 'open', ['evaluation/model_comparison.csv', 'evaluation/models/*/*/result.json'],
        {'complete_results': len(outputs), 'expected_results': 60, 'missing': missing})
    interval_valid = valid and all(np.load(root/'evaluation/models'/n/s/'bootstrap.npz')['values'].shape[0] == 400
                                  for n, s in expected)
    record('400 physical-block intervals, paired controls and separate training-seed spread',
        'complete' if interval_valid and (root/'evaluation/paired_comparisons.json').exists() else 'open',
        ['evaluation/bootstrap_design.json', 'evaluation/paired_comparisons.json'],
        {'replicates': 400, 'all_model_replicates_present': interval_valid,
         'undefined_metrics': 'valid replicate counts are reported; a zero selected denominator leaves purity undefined'})
    quality = root/'evaluation/quality.parquet'
    conditional_valid = valid and all(any(s['stratum'] == 'tracker_stations=3' for s in v['conditional']) and
        any(s['stratum'].startswith('tau_branch=') for s in v['conditional']) and
        any(s['stratum'].startswith('charm_presence=') for s in v['conditional']) for v in outputs)
    record('energy, vertex, occupancy, tau/charm, sign, containment proxies and reconstructability',
        'complete' if quality.exists() and conditional_valid else 'open',
        ['evaluation/quality.parquet', 'test_truth.parquet', 'evaluation/models/*/*/result.json'],
        {'observable_leakage_indicators': True, 'truth_energy_containment_measured': False,
         'quality_and_subsets_present': conditional_valid})
    support = read('evaluation/exact_target_overlap.json')
    record('joint energy/vertex/exact-target common support diagnostic', 'complete' if support else 'open',
        ['evaluation/exact_target_overlap.json'], support['support'] if support else None)
    variants = read('predictions/variants_completion.json')
    variation_results = list((root/'evaluation/conditions/variants').glob('*.json'))
    record('14 paired readout conditions including heldout pattern, fixed nominal cuts',
        'complete' if variants and variants['events'] == 32000 and variants['variants'] == 14 and len(variation_results) == 14 else 'open',
        ['predictions/variants_completion.json', 'evaluation/conditions/variants/*.json'],
        {'coverage': variants, 'result_files': len(variation_results), 'amplitudes': 'accepted detector-proxy stress values'})
    transport = read('predictions/transport_completion.json')
    transports = list((root/'evaluation/conditions/transport').glob('*.json'))
    record('actual fixed-primary/vertex field, W-density and tau-helicity transport',
        'complete' if transport and len(transport['condition_counts']) == 8 and len(transports) == 8 else 'open',
        ['g4/build.json', 'transport/inputs.json', 'predictions/transport_completion.json', 'evaluation/conditions/transport/*.json'],
        {'coverage': transport, 'result_files': len(transports), 'physical_systematic_calibration': 'not claimed'})
    generator = read('evaluation/conditions/generator_comparison.json')
    record('matched compatible hadronization-generator comparison with flavour/charm support',
        'complete' if generator and len(generator['class_counts']) == 8 else 'open',
        ['generator/inventory.json', 'generator/matching_diagnostics.json', 'evaluation/conditions/generator_comparison.json'],
        {'measured': 'GENIE3.04 Pythia6 Standard/NUX on W, 3512 independent hard scatters',
         'NuWro': 'unmeasured; no readable compatible input found', 'comparison_present': generator is not None})
    decay = {scope: read(f'{scope}/truth_completion.json') for scope in ['transport', 'generator']}
    decay_valid = all(decay[s] and decay[s]['all_observations_exactly_equal'] and
        decay[s]['rows'] == n and read(f'evaluation/conditions/{s}_decay.json') for s, n in [('transport', 5888), ('generator', 3512)])
    record('resimulated tau/charm truth and polarization/secondary implementation evidence',
        'complete' if decay_valid else 'open',
        ['transport/truth_completion.json', 'generator/truth_completion.json', 'evaluation/conditions/*_decay.json'],
        {'replays': decay, 'charm_children_are_not_branching_fractions': True})
    migration = read('evaluation/migration/summary.json')
    record('score/energy migration and auxiliary-output validity', 'complete' if migration else 'open',
        ['evaluation/migration/summary.json', 'evaluation/migration/*.npz'], migration)
    cost = read('evaluation/resource_accounting.json')
    record('resource allocation evidence, backend closure and compact outputs',
        'complete' if cost and cost.get('backend_queries_complete') and not cost['active_jobs'] and
        not any(f['unresolved_rows'] for f in cost['submitted_condor_factories']) and
        (root/'evaluation/cleanup.json').exists() and (root/'evaluation/backend_closure.json').exists() else 'open',
        ['evaluation/resource_accounting.json', 'evaluation/cleanup.json', 'evaluation/backend_closure.json'],
        {'accounting_present': cost is not None, 'active_jobs': len(cost['active_jobs']) if cost else None,
         'MIG_utilization_fraction': 'unmeasured where metrics unavailable'})
    record('timing/window and physical exposure scope', 'documented_input_limit', ['conditions.json', 'analysis_freeze.json'],
        {'time_window': 'unmeasured; no timing observation', 'luminosity_yields_and_significance': 'unmeasured; no physical exposure',
         'prior_efficiency_purity_scenarios': ['balanced', 'production_sampling', 'tau_1_percent', 'tau_0p1_percent']})
    result = {'stage': 'S6', 'requirements': requirements, 'complete': all(v['status'] in ['complete', 'documented_input_limit'] for v in requirements),
        'open_requirements': [v['requirement'] for v in requirements if v['status'] == 'open'],
        'sample_and_proxy_authorization': 'use accepted original samples/splits and proxy values; no repeated certification'}
    write_json(root/'evaluation/scope_completion_audit.json', result)
    print(json.dumps({'complete': result['complete'], 'open': result['open_requirements']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
