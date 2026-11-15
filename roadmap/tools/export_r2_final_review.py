"""Export the accepted R2 evidence without reading sealed event payloads."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


DOMAINS = ['all', 'pixel_centered', 'large_scintillator', 'occupancy_0_100', 'empty']
PAIR_METRICS = [
    'BA3', 'NLL3', 'macro_AP3', 'macro_AUROC3',
    'recall_e', 'recall_mu', 'recall_NC', 'BA4', 'NLL4',
    'tau_eff_1pct', 'tau_eff_0p1pct', 'mu_eff_FPR_0.01',
]


def read(path):
    return json.loads(path.read_text())


def task_summary(row):
    return {
        **{key: row[key] for key in ['BA', 'NLL', 'macro_AP', 'macro_AUROC', 'raw_counts']},
        'per_class': [{key: value.get(key) for key in [
            'recall', 'AP', 'AUROC', 'eff_FPR_0.01', 'eff_FPR_0.001',
        ]} for value in row['per_class']],
    }


def export(root):
    acceptance = read(root/'acceptance.json')
    if acceptance['status'] != 'complete' or acceptance['failed']:
        raise ValueError('final evidence requires completed R2 acceptance')
    evaluation = read(root/'evaluation/results.json')
    readout = read(root/'readout/results.json')
    campaign = read(root/'campaign.json')
    resources = read(root/'resources.json')
    decision = read(root/'decision.json')
    models = {}
    for name in campaign['initial_models'] + campaign.get('extension_models', []):
        record = read(root/'experiments'/name/'manifest.json')
        diagnostic = readout['models'][name]
        views = {}
        for kind in ['endpoint', 'best_screen']:
            views[kind] = {}
            for domain in DOMAINS:
                row = evaluation['models'][name+'_'+kind][domain]
                views[kind][domain] = {
                    'raw_counts_original_labels': row['raw_counts_original_labels'],
                    **{task: task_summary(row[task]) for task in ['T3', 'T4'] if row.get(task)},
                }
        models[name] = {
            'recipe': {key: record['config'][key] for key in [
                'design', 'task', 'seed', 'passes', 'batch_size', 'learning_rate', 'weight_decay',
                'cnn_width', 'precision',
            ]},
            'counters': record['counters'],
            'GPU': record['GPU'],
            'allocation_id': record['allocation_id'],
            'best_screen': record['best_screen'],
            'views': views,
            'convergence': diagnostic['convergence'],
            'readout': {
                domain: {
                    variant: {key: row[key] for key in ['BA', 'NLL']}
                    for variant, row in diagnostic['domains'][domain].items()
                } for domain in ['all', 'occupancy_0_100']
            },
        }
    pairs = [
        {**{key: pair[key] for key in ['a', 'b', 'domain', 'reason']},
         'metrics': {key: pair['metrics'][key] for key in PAIR_METRICS}}
        for pair in evaluation['pairs']
        if pair['a'].endswith('_endpoint') and pair['domain'] in ['all', 'occupancy_0_100']
    ]
    result = {
        'status': 'complete',
        'UTC': datetime.now(timezone.utc).isoformat(),
        'source_root': str(root),
        'scope': 'single seed42 development Tune; endpoint primary; event intervals do not estimate training variation',
        'matrix': {key: acceptance[key] for key in ['models', 'checkpoint_views', 'pairs']},
        'bootstrap': evaluation['bootstrap'],
        'models': models,
        'selected_endpoint_pairs': pairs,
        'decision': decision,
        'acceptance': {key: value for key, value in acceptance.items() if key != 'checks'},
        'resources': {
            **{key: resources[key] for key in [
                'UTC', 'device_hours', 'allocated_CPU_core_hours',
                'device_hours_by_model_or_partition', 'active_or_unresolved',
                'interactive_operation_lower_bound_device_hours',
                'heterogeneous_device_hours_are_not_FLOP_equivalent',
            ]},
            'allocation_records': len(resources['allocations']),
            'rejected_before_allocation_count': len(resources['rejected_before_allocation']),
            'terminal_cost': not resources['active_or_unresolved'],
            'full_allocation_ledger': str(root/'resources.json'),
        },
        'Cal_A_Cut_B_Confirm_payload_read': False,
    }
    if (root/'sampling/acceptance.json').exists():
        result['conditional_sampling_replay'] = {
            'contract': read(root/'sampling/contract.json'),
            'acceptance': read(root/'sampling/acceptance.json'),
            'reproduction': read(root/'sampling/reproduction.json'),
            'models': {name: {domain: {
                task: task_summary(row[task]) for task in ['T3', 'T4'] if row.get(task)
            } for domain, row in values.items() if domain in ['all', 'occupancy_0_100']}
                for name, values in read(root/'sampling/evaluation/results.json')['models'].items()},
            'pairs': [{**{key: pair[key] for key in ['a', 'b', 'domain', 'reason']},
                'metrics': {key: pair['metrics'][key] for key in PAIR_METRICS}}
                for pair in read(root/'sampling/evaluation/results.json')['pairs']],
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = export(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': result['status'], 'matrix': result['matrix'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
