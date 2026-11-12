"""Freeze the preregistered four-class/resource Pareto confirmation campaign."""
import argparse
import copy
import json
from pathlib import Path

from analysis.flavour.inventory import write_json


CANDIDATES = [*range(10), 14, 16]


def select_candidates(rows):
    """Maximize macro AP, minimize NLL and actual allocated GPU hours."""
    by_task = {row['task']: row for row in rows}
    if set(by_task) != set(CANDIDATES):
        raise ValueError('incomplete structural candidate set')
    for row in rows:
        if any(row[k] is None for k in ['macro_AP', 'NLL', 'allocated_gpu_hours']):
            raise ValueError('candidate has incomplete quality/resource evidence')
    def objective(row):
        return [-row['macro_AP'], row['NLL'], row['allocated_gpu_hours']]
    frontier = []
    for row in rows:
        value = objective(row)
        dominated = any(all(a <= b for a, b in zip(objective(other), value)) and
                        any(a < b for a, b in zip(objective(other), value)) for other in rows)
        if not dominated:
            frontier.append(row)
    first = min(frontier, key=lambda r: (-r['macro_AP'], r['NLL'], r['allocated_gpu_hours'], r['task']))
    remaining = [r for r in frontier if r['task'] != first['task']]
    if remaining:
        second = min(remaining, key=lambda r: (r['NLL'], r['allocated_gpu_hours'], r['task']))
        reason = 'lowest NLL among remaining Pareto candidates'
    else:
        remaining = [r for r in rows if r['task'] != first['task'] and r['macro_AP'] >= first['macro_AP'] - .005]
        if remaining:
            second = min(remaining, key=lambda r: (r['allocated_gpu_hours'], r['NLL'], r['task']))
            reason = 'single Pareto point; least allocation among remaining candidates within 0.005 macro AP'
        else:
            second = min([r for r in rows if r['task'] != first['task']], key=lambda r: (r['NLL'], r['allocated_gpu_hours'], r['task']))
            reason = 'single Pareto point with no near-quality remaining candidate; lowest remaining NLL'
    return [first['task'], second['task']], {'frontier': [r['task'] for r in frontier], 'rows': rows,
        'first_rule': 'highest macro AP on quality/resource Pareto frontier', 'second_rule': reason}


def freeze_json(path, content):
    if path.exists() and json.loads(path.read_text()) != content:
        raise ValueError(f'refusing to overwrite frozen confirmation file {path}')
    if not path.exists():
        write_json(path, content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--output-campaign', type=Path, required=True)
    parser.add_argument('--output-pairs', type=Path, required=True)
    args = parser.parse_args()
    campaign = json.loads(args.campaign.read_text())
    evaluation = json.loads(args.evaluation.read_text())
    by_name = {e['name']: e for e in evaluation['experiments']}
    rows = []
    for task, config in enumerate(campaign['experiments']):
        name = config['name']
        checks = json.loads((args.root / 'experiments' / name / 'final_checkpoint_checks.json').read_text())
        if checks['status'] != 'pass' or not all(checks['checks'].values()):
            raise ValueError(f'{name}: final readback failed')
        e = by_name[name]
        if not e['allocation_complete']:
            raise ValueError(f'{name}: allocation not complete')
        if task in CANDIDATES:
            rows.append({'task': task, 'name': name, 'macro_AP': e['four_class']['macro_AP'],
                         'NLL': e['four_class']['NLL'], 'allocated_gpu_hours': e['allocated_gpu_hours']})
    if not evaluation['requested_screen_and_pairs_complete']:
        raise ValueError('screen or preregistered comparisons incomplete')
    selected, evidence = select_candidates(rows)
    confirm_tasks = sorted(set(selected + [5, 6]))
    configs = []
    for task in confirm_tasks:
        for seed in [17, 2026]:
            config = copy.deepcopy(campaign['experiments'][task]); config['seed'] = seed
            config['name'] = config['name'].replace('_s42_', f'_s{seed}_')
            config['role'] = 's3_frozen_confirmation'
            configs.append(config)
    confirmation = {**{k: v for k, v in campaign.items() if k != 'experiments'},
        'screen_selection': {'selected_tasks': selected, 'confirmed_tasks': confirm_tasks, **evidence}, 'experiments': configs}
    pairs = json.loads(args.pairs.read_text())
    means = []
    for task in confirm_tasks:
        name = campaign['experiments'][task]['name']
        for reference in ['part_native_joint_s42_e4_v2', 'pn_native_joint_s42_e4_v2']:
            means.append({'name': name + '_vs_' + reference, 'pairs': [
                {'a': reference.replace('_s42_', f'_s{seed}_'), 'b': name.replace('_s42_', f'_s{seed}_')}
                for seed in [17, 42, 2026]]})
    native, coarse = [campaign['experiments'][i]['name'] for i in [5, 6]]
    means.append({'name': 'primary_native_minus_1mm_three_seed', 'pairs': [
        {'a': coarse.replace('_s42_', f'_s{seed}_'), 'b': native.replace('_s42_', f'_s{seed}_')}
        for seed in [17, 42, 2026]]})
    pairs['seed_mean_comparisons'] = means
    freeze_json(args.output_campaign, confirmation); freeze_json(args.output_pairs, pairs)
    print(json.dumps({'selected_tasks': selected, 'confirmed_tasks': confirm_tasks, 'additional_runs': len(configs)}))


if __name__ == '__main__':
    main()
