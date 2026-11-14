"""Compact measured S6 tables, preserving the frozen analysis scope."""
import argparse
import json
from pathlib import Path

from analysis.flavour.inventory import write_json


METRICS = ['balanced_accuracy', 'NLL', 'Brier', 'macro_AP', 'macro_AUROC', 'macro_F1']


def collect(args):
    root = args.root
    read = lambda path: json.loads((root/path).read_text())
    freeze = read('analysis_freeze.json')
    tables = []
    primary = {}
    for path in sorted((root/'evaluation/models').glob('*/*/result.json')):
        result = json.loads(path.read_text())
        selected = [m for m in result['fixed'] if m.startswith('class_2_')]
        values = {m: {'value': result['metrics'][m], **result['intervals'][m]} for m in METRICS}
        values.update({m: {'value': result['fixed'][m], **result['intervals'][m]} for m in selected})
        tables.append({'name': result['name'], 'scenario': result['scenario'], 'prior': result['prior'],
            'Neff': result['Neff'], 'metrics': values})
        if result['name'] == 'mae3_100k_ensemble':
            primary[result['scenario']] = {'metrics': values, 'confusion_raw_counts': result['confusion_raw_counts'],
                'fixed_raw_counts': result['fixed_raw_counts'], 'cutflow': result['cutflow'],
                'conditional': result['conditional'],
                'reliability_ECE': result['reliability']['top_label']['ECE'],
                'reliability_ECE_before_temperature': result['reliability_before_temperature']['top_label']['ECE']}
    if len(tables) != 60:
        raise ValueError('incomplete final model table')
    conditions = {}
    for scope, expected in [('variants', 14), ('transport', 8)]:
        rows = {}
        for path in sorted((root/'evaluation/conditions'/scope).glob('*.json')):
            result = json.loads(path.read_text())
            rows[result['condition']] = {'events': result['events'], 'class_counts': result['class_counts'],
                'scenarios': {scenario: {m: value['metrics'][m] for m in
                    ['balanced_accuracy', 'NLL', 'class_2_efficiency@0.01', 'class_2_purity@0.01']}
                    for scenario, value in result['scenarios'].items()}}
        if len(rows) != expected:
            raise ValueError(f'incomplete condition {scope}')
        conditions[scope] = rows
    write_json(root/'evaluation/final_summary.json', {'stage': 'S6', 'analysis_frozen_UTC': freeze['UTC'],
        'calibration': read('calibration/completion.json'), 'test': read('predictions/test_completion.json'),
        'models_and_priors': tables, 'primary': primary, 'conditions': conditions,
        'paired_controls_and_seed_spread': read('evaluation/paired_comparisons.json'),
        'common_support': read('evaluation/exact_target_overlap.json'),
        'generator': read('evaluation/conditions/generator_comparison.json'),
        'transport_decay': read('evaluation/conditions/transport_decay.json'),
        'generator_decay': read('evaluation/conditions/generator_decay.json'),
        'migration': read('evaluation/migration/summary.json'),
        'resources': {k: v for k, v in read('evaluation/resource_accounting.json').items()
            if k not in ['condor_records', 'queries']},
        'scientific_scope': {'test_threshold_retuning': False, 'old_faulty_S3_used_for_conclusions': False,
            'physical_flux_or_exposure': 'unmeasured', 'hardware_amplitudes': 'accepted detector-proxy values',
            'new_hypotheses_require_new_validation_contract': True}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    collect(parser.parse_args())


if __name__ == '__main__':
    main()
