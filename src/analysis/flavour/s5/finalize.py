"""Publish S5 evidence only after the full declared validation matrix is complete."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def audit(root, campaign, tag, require_closure=False):
    configs = json.loads(campaign.read_text())['experiments']
    report = json.loads((root/'evaluation'/tag/'summary.json').read_text())
    entries = {e['name']: e for e in report['experiments']}
    hgb = sorted((root/'experiments').glob('s5_hgb_*'))
    expected = {c['name'] for c in configs}|{p.name for p in hgb}
    if set(entries) != expected or len(entries) != len(report['experiments']) or len(expected) != len(configs)+3:
        raise ValueError('final report differs from the full declared matrix')
    models = []
    for config in configs:
        name = config['name']
        directory = root/'experiments'/name
        manifest = json.loads((directory/'manifest.json').read_text())
        replay = json.loads((directory/'readback.json').read_text())
        if manifest['status'] != 'complete' or manifest['config'] != config or manifest['calibration_test_used']:
            raise ValueError('incomplete or inconsistent configuration: '+name)
        if not all([manifest['all_internal_health_pass'], replay['checkpoint_replay_pass'], replay['encoder_update_pass']]):
            raise ValueError('learning/replay evidence missing: '+name)
        if not (directory/'best.pt').exists() or entries[name]['metrics']['events'] != 10000:
            raise ValueError('checkpoint or full validation missing: '+name)
        samples = json.loads((directory/'training_health.json').read_text())['samples']
        if not samples or not all(row['healthy'] for row in samples):
            raise ValueError('saved internal-health measurements failed: '+name)
        head_statistics = [item for row in samples if row['informative_nodes'] > 0
                           for item in row['activations'].get('head.2', [])]
        if not head_statistics or max(item['zero_fraction'] for item in head_statistics) == 1.:
            raise ValueError('head activation learning evidence is missing: '+name)
        models.append({'name': name, 'variant': config['variant'], 'seed': config['seed'], 'epochs': config['epochs'],
                       'parameters': manifest['parameters'], 'GPU': manifest['gpu'], 'best_epoch': manifest['best_epoch'],
                       'replay_maximum_probability_difference': replay['maximum_probability_difference'],
                       'parameter_delta_norms': replay['parameter_delta_norms'],
                       'learning': {'optimizer_steps': manifest['optimizer_steps'], 'internal_health_samples': len(samples),
                                    'maximum_SiLU_zero_fraction': max(item['zero_fraction'] for item in head_statistics),
                                    'selected_checkpoint_probability_std': replay['validation_probability_std']},
                       'metrics': entries[name]['metrics']})
    for directory in hgb:
        manifest = json.loads((directory/'manifest.json').read_text())
        if manifest['status'] != 'complete' or manifest['maximum_replay_difference'] != 0:
            raise ValueError('HGB replay incomplete: '+directory.name)
    index = {c['name']: c for c in configs}
    for pair in report['comparisons']:
        if pair['a'] in index and pair['b'] in index:
            a, b = index[pair['a']], index[pair['b']]
            if any(a[k] != b[k] for k in ['epochs', 'label_budget', 'seed']):
                raise ValueError('unmatched comparison budgets: '+str(pair))
    confirmed = {r['variant'] for r in report['seed_means'] if r['cohort'] == 'confirmation8'}
    if confirmed != {'six', 'vertex_kinematics'}:
        raise ValueError('three-seed confirmation incomplete')
    if report['bootstrap']['replicates'] < 400 or report['calibration_test_used']:
        raise ValueError('declared validation statistics or split boundary differ')
    gate = json.loads((root/'pilot/gate.json').read_text())
    if not gate['gate_passed'] or gate.get('final_six_accuracy') != 1.:
        raise ValueError('balanced real six-class fit evidence missing')
    direction = json.loads((root/'regression_diagnostic/summary.json').read_text())
    if direction['status'] != 'complete' or direction['selected_checkpoint_component_replay_difference'] > .003:
        raise ValueError('transverse train/readout review is incomplete')
    figures = sorted(p.name for p in (root/'evaluation'/tag).glob('*.pdf'))
    if len(figures) < 9:
        raise ValueError('declared physics figures are incomplete')
    accounting = json.loads((root/'campaign_accounting.json').read_text())
    cleanup = json.loads((root/'cleanup.json').read_text()) if (root/'cleanup.json').exists() else {}
    closed = accounting['all_terminal'] and cleanup.get('complete', False) and cleanup.get('AFS_cleanup_complete', False)
    if require_closure and not closed:
        raise ValueError('allocation terminal states or output cleanup are pending')
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'scientific_complete': True, 'stage_complete': closed,
        'matrix': str(campaign), 'neural_models': len(configs), 'HGB_models': len(hgb), 'validation_events_per_model': 10000,
        'models': models, 'pairs': len(report['comparisons']), 'three_seed_comparisons': sorted(confirmed),
        'bootstrap': report['bootstrap'], 'sample_strata': report['sample_strata'], 'figures': figures,
        'calibration_test_used': False, 'next_stage': 'S6 calibration, frozen final test and systematic variants',
        'scope': 'complete S5 validation study; shared validation used for candidate selection, no independent test claim',
        'transverse_regression_review': str(root/'regression_diagnostic/summary.json'),
        'remote_device_hours': accounting['remote_device_hours'], 'CPU_allocation_hours': accounting['CPU_allocation_hours'],
        'interactive_pilot_fit_device_seconds': accounting['interactive_pilot_fit_device_seconds']}
    (root/'acceptance.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['models', 'sample_strata', 'bootstrap', 'figures']}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--tag', default='final')
    parser.add_argument('--require-closure', action='store_true')
    args = parser.parse_args()
    audit(args.root, args.campaign, args.tag, args.require_closure)


if __name__ == '__main__':
    main()
