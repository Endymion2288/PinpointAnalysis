"""Compare each learned component to constants using the actual training loss."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.flavour.inventory import write_json
from analysis.flavour.s5.dataset import PhysicsDataset


def huber(residual, beta=.1):
    absolute = np.abs(residual)
    return np.where(absolute < beta, residual**2/(2*beta), absolute-beta/2)


def run(root, campaign):
    configs = json.loads(campaign.read_text())['experiments']
    train = PhysicsDataset(root, 'train', {**configs[0], 'use_objects': False})
    validation = PhysicsDataset(root, 'validation', {**configs[0], 'use_objects': False})
    target_train = train.targets['kinematics'][train.targets['kinematics_mask']]
    target = validation.targets['kinematics'][validation.targets['kinematics_mask']]
    order = np.lexsort(tuple(validation.metadata[k] for k in ['row', 'chunk', 'run']))
    valid_order = validation.targets['kinematics_mask'][order]
    target = validation.targets['kinematics'][order][valid_order]
    mean, median = target_train.mean(0), np.median(target_train, axis=0)
    constants = {'mean': mean, 'median': median}
    rows = []
    for config in configs:
        if 'kinematics' not in config.get('auxiliaries', []):
            continue
        frame = pd.read_parquet(root/'experiments'/config['name']/'validation_predictions.parquet').sort_values(['run', 'chunk', 'row'])
        prediction = frame[[f'pred_kinematics_{i}' for i in range(4)]].to_numpy()[valid_order]
        residual = prediction-target
        row = {'name': config['name'], 'events': len(target), 'encoded_component_bias': residual.mean(0).tolist(),
               'component_MSE': (residual**2).mean(0).tolist(), 'component_Huber': huber(residual).mean(0).tolist(),
               'negative_encoded_pz_E_counts': (prediction[:, 2:] < 0).sum(0).tolist(),
               'component_correlation': [float(np.corrcoef(prediction[:, i], target[:, i])[0, 1]) for i in range(4)],
               'constant_references': {name: {'MSE': ((value-target)**2).mean(0).tolist(), 'Huber': huber(value-target).mean(0).tolist()} for name, value in constants.items()}}
        rows.append(row)
    write_json(root/'auxiliary_objective_audit.json', {'models': rows, 'units': 'dataset asinh/log encoded momentum divided by 7',
        'fit_reference': 'selected 10k train labels only', 'calibration_test_used': False})
    train.native.close(); validation.native.close()
    print(json.dumps(rows), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.campaign)


if __name__ == '__main__':
    main()
