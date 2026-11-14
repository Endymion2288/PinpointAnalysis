"""Export compact S5 measurements without model files or repeated diagnostics."""
import argparse
import json
from pathlib import Path


def run(root, output):
    acceptance = json.loads((root/'acceptance.json').read_text())
    if not acceptance['stage_complete']:
        raise ValueError('complete scientific and allocation closure required')
    report = json.loads((root/'evaluation/final/summary.json').read_text())
    scope = json.loads((root/'evaluation/scope/summary.json').read_text())
    accounting = json.loads((root/'campaign_accounting.json').read_text())
    cleanup = json.loads((root/'cleanup.json').read_text())
    evidence = {row['name']: row for row in acceptance['models']}
    columns = ['NLL', 'macro_AP', 'macro_AUROC', 'macro_F1', 'balanced_accuracy',
               'class_0_AP', 'class_1_AP', 'class_2_AP', 'class_3_AP',
               'class_2_eff_at_0.01_FPR', 'class_2_eff_at_0.001_FPR']
    measurements = []
    for entry in report['experiments']:
        manifest = json.loads((root/'experiments'/entry['name']/'manifest.json').read_text())
        config = manifest['config']
        hgb = entry['name'].startswith('s5_hgb_')
        measurements.append({'name': entry['name'], 'variant': entry['variant'], 'seed': entry['seed'],
            'cohort': 'HGB_reference' if hgb else config.get('analysis_cohort', f"passes{config['epochs']}_labels{config['label_budget']}"),
            'training_kind': 'HGB reference fitted separately' if hgb else 'neural complete-pass training',
            'epochs': None if hgb else config['epochs'], 'label_budget': config['label_budget'],
            'metrics': {key: entry['metrics'][key] for key in columns},
            'auxiliary_quality': entry['auxiliary_quality'], 'six_way': entry.get('six_way'),
            'learning': evidence.get(entry['name'], {}).get('learning'),
            'elapsed_seconds': manifest.get('elapsed_seconds', manifest.get('seconds'))})
    result = {'stage': 'S5', 'complete': True, 'root': str(root),
        'scope': acceptance['scope'], 'calibration_test_used': False,
        'counts': {'neural_models': acceptance['neural_models'], 'HGB_models': acceptance['HGB_models'],
                   'validation_events_per_model': acceptance['validation_events_per_model'], 'pairs': acceptance['pairs']},
        'sample_strata': report['sample_strata'], 'bootstrap': report['bootstrap'],
        'selection': json.loads((root/'selection.json').read_text()),
        'measurements': measurements, 'three_seed_comparisons': report['seed_means'],
        'direction_diagnostic': json.loads((root/'regression_diagnostic/summary.json').read_text()),
        'scope_diagnostics': {key: scope[key] for key in ['target_common_support', 'three_seed_comparisons', 'tau_geometry', 'bootstrap']},
        'geometry_crosscheck': json.loads((root/'evaluation/scope/geometry_crosscheck.json').read_text()),
        'resources': {key: accounting[key] for key in ['all_terminal', 'remote_device_hours', 'CPU_allocation_hours',
            'interactive_pilot_fit_device_seconds', 'interactive_pilot_fit_intervals']},
        'cleanup': {key: cleanup[key] for key in ['complete', 'AFS_cleanup_complete', 'bytes_released', 'best_checkpoints_retained']},
        'figures': acceptance['figures'], 'supplementary_figures': acceptance['supplementary_figures'],
        'next_stage': acceptance['next_stage']}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'output': str(output), 'models': len(measurements), 'pairs': acceptance['pairs']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)


if __name__ == '__main__':
    main()
