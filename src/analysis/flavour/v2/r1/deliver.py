"""Rank completed Scratch baselines and enumerate the frozen R1 deliverables."""

import argparse
import json
from pathlib import Path

import numpy as np

from analysis.flavour.inventory import write_json
from analysis.flavour.v2.r1.campaign import FAMILIES


def load(root, name):
    record = json.loads((root/'experiments'/name/'manifest.json').read_text())
    if record['status'] != 'complete':
        raise ValueError(f'{name} is incomplete')
    return record


def ranking(root):
    rows = []
    for family in FAMILIES:
        values = [load(root, f'final_{family.lower()}_{task.lower()}_s42') for task in ['T3', 'T4']]
        rows.append({'family': family,
            'mean_common_BA3': float(np.mean([v['metrics']['T3']['BA'] for v in values])),
            'mean_common_NLL3': float(np.mean([v['metrics']['T3']['NLL'] for v in values])),
            'per_task': {v['config']['task']: v['metrics'] for v in values}})
    rows.sort(key=lambda r: (-r['mean_common_BA3'], r['mean_common_NLL3']))
    expanded = list(dict.fromkeys([r['family'] for r in rows[:2]]+['B4']))
    write_json(root/'architecture_ranking.json', {'status': 'complete', 'ranking': rows,
        'expanded_families': expanded, 'rule': 'frozen Scratch mean T3/T4 conditional BA3 then NLL3',
        'scope': 'development Tune at four passes; convergence is studied in R2'})


def evaluation_config(root):
    ranks = json.loads((root/'architecture_ranking.json').read_text())
    expanded = ranks['expanded_families']
    models, pairs = [], []
    def model(name, family, task, seed, kind='main'):
        load(root, name)
        models.append({'name': name, 'family': family, 'task': task, 'seed': seed,
                       'kind': kind, 'predictions': f'experiments/{name}/predictions.parquet'})
    def pair(a, b, reason, geometry=False):
        if a == b:
            return
        for domain in (['all', 'pixel_centered', 'large_scintillator'] if geometry else ['all']):
            item = {'a': a, 'b': b, 'domain': domain, 'reason': reason}
            if item not in pairs:
                pairs.append(item)
    for family in FAMILIES:
        seeds = [17, 42, 2026] if family in expanded else [42]
        for seed in seeds:
            prefix = f'final_{family.lower()}'
            t3, t4 = f'{prefix}_t3_s{seed}', f'{prefix}_t4_s{seed}'
            model(t3, family, 'T3', seed)
            model(t4, family, 'T4', seed)
            compute = f'compute_{family.lower()}_t3_s{seed}'
            model(compute, family, 'T3', seed, 'matched_compute')
            pair(t3, t4, 'same non-tau unique rows and four exposures', geometry=seed==42)
            pair(compute, t4, 'same-allocation total operation wall budget', geometry=seed==42)
            pair(compute, t3, 'extra non-tau exposures under T4 resource budget')
            if family != 'B4':
                for task in ['t3', 't4']:
                    pair(f'{prefix}_{task}_s{seed}', f'final_b4_{task}_s{seed}',
                         'Scratch architecture comparison', geometry=seed==42)
    mae = 'final_b4-mae_t4_s42'
    model(mae, 'B4-MAE', 'T4', 42)
    pair(mae, 'final_b4_t4_s42', 'MAE initialization control; SSL cost separate', geometry=True)
    b0 = json.loads((root/'b0_selected.json').read_text())['selected']
    b0_names = {}
    for task, entry in b0.items():
        for name in entry['models']:
            record = load(root, name)
            seed = record['config']['seed']
            b0_names[(task, seed)] = name
            model(name, 'B0', task, seed)
            pair(name, f'final_b4_{task.lower()}_s{seed}', 'observation-summary baseline', geometry=seed==42)
    for entry in list(models):
        if entry['family'] in ['B1', 'B2', 'B3'] and entry['kind']=='main':
            pair(entry['name'], b0_names[(entry['task'], entry['seed'])],
                 'Scratch versus observation-summary baseline', geometry=entry['seed']==42)
    for family in ['B1', 'B4']:
        name = f'final_{family.lower()}_t3_s42_without_tracker'
        full = load(root, f'final_{family.lower()}_t3_s42')
        without = load(root, name)
        if full['config'].get('bf16', True)!=without['config'].get('bf16', True):
            name += '_bf16' if full['config'].get('bf16', True) else '_fp32'
        model(name, family, 'T3', 42, 'without_tracker')
        pair(name, f'final_{family.lower()}_t3_s42', 'retrained tracker incremental value', geometry=True)
    for task in ['t3', 't4']:
        name = f'control_b1_cnn_{task}_s42'
        model(name, 'B1-CNN', task.upper(), 42, 'cnn_cost_control')
        pair(name, f'final_b1_{task}_s42', 'plain CNN versus ResNet with the B1-selected recipe', geometry=True)
    best = next(row['family'].lower() for row in ranks['ranking'] if row['family'] in ['B2', 'B3', 'B4'])
    model_names = {e['name'] for e in models}
    for seed in [17, 42, 2026]:
        for task in ['t3', 't4']:
            native, projection = f'final_{best}_{task}_s{seed}', f'final_b1_{task}_s{seed}'
            if native in model_names and projection in model_names:
                pair(native, projection, 'native versus projection homologous baseline', geometry=seed==42)
    write_json(root/'evaluation_config.json', {'models': models, 'pairs': pairs,
        'Cal_A_Cut_B_Confirm_payload_read': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['ranking', 'evaluation_config'])
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    globals()[args.command](args.root)


if __name__ == '__main__':
    main()
