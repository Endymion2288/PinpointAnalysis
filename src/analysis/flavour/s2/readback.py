"""Read completed training outputs on CPU; use the accepted catalog unchanged."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from sklearn.metrics import log_loss

from analysis.flavour.inventory import write_json


def finite_tensors(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tensors(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tensors(item) for item in value)
    return True


def readback(root, config, hardware, validation):
    name = config['name']
    output = root / 'experiments' / name
    manifest = json.loads((output / 'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError(f'{name}: training is {manifest["status"]}')
    last = torch.load(output / 'last.pt', map_location='cpu', weights_only=False)
    best = torch.load(output / 'best.pt', map_location='cpu', weights_only=False)
    history = json.loads((output / 'history.json').read_text())
    predictions = pd.read_parquet(output / 'validation_predictions.parquet')
    count = manifest['validation_events']
    n_epochs = config['epochs']
    steps = n_epochs * config['steps_per_epoch']
    selected = min(history, key=lambda item: item['validation']['NLL'])
    probabilities = predictions[[f'p{c}' for c in range(4)]].to_numpy()
    keys = ['run', 'chunk', 'row', 'label']
    ordered = lambda frame: frame[keys].sort_values(keys).reset_index(drop=True)
    same_weights = best['model'].keys() == last['best_model'].keys() and all(
        torch.equal(value, last['best_model'][key]) for key, value in best['model'].items())
    epoch_error = None
    try:
        epochs = pd.read_parquet(output / 'epoch_predictions.parquet')
        valid_epochs = len(epochs) == count * n_epochs and not epochs.duplicated(['epoch', 'row']).any()
        valid_epochs &= set(epochs.epoch) == set(range(1, n_epochs + 1))
        valid_epochs &= all(np.array_equal(group.row.sort_values(), np.arange(count))
                            for _, group in epochs.groupby('epoch'))
        epoch_probabilities = epochs[[f'p{c}' for c in range(4)]].to_numpy()
        selected_probabilities = epochs[epochs.epoch.eq(selected['epoch'])].sort_values('row')
        selected_probabilities = selected_probabilities[[f'p{c}' for c in range(4)]].to_numpy()
        valid_epochs &= bool(np.isfinite(epoch_probabilities).all() and
            ((epoch_probabilities >= 0) & (epoch_probabilities <= 1)).all() and
            np.allclose(epoch_probabilities.sum(1), 1, rtol=0, atol=1e-6) and
            np.array_equal(probabilities, selected_probabilities))
    except (pa.ArrowInvalid, OSError) as error:
        # Keep independent checkpoint/best-prediction evidence when an EOS
        # epoch-table write is incomplete; the overall readback still fails.
        valid_epochs = False
        epoch_error = str(error)
    checks = {
        'complete_epochs': last['epoch'] == n_epochs and
            [item['epoch'] for item in history] == list(range(1, n_epochs + 1)) and last['history'] == history,
        'frozen_configuration': manifest['config'] == last['config'] == best['config'] == config,
        'execution_environment': manifest['gpu_name'] == hardware['GPU'] and
            manifest['torch'] == hardware['torch'] and manifest['mixed_precision'] == 'torch.bfloat16',
        'all_optimizer_steps_applied': last['optimizer_steps'] == manifest['optimizer_steps'] == steps and
            last['amp_skipped_steps'] == manifest['amp_skipped_steps'] == 0 and
            history[-1]['attempted_optimizer_steps_cumulative'] == steps,
        'NLL_selection': best['epoch'] == last['best_epoch'] == manifest['best_epoch'] == selected['epoch'] and
            np.isclose(manifest['best_NLL'], selected['validation']['NLL'], rtol=0, atol=1e-10) and
            np.isclose(log_loss(predictions.label, probabilities, labels=[0, 1, 2, 3]),
                       selected['validation']['NLL'], rtol=0, atol=1e-8),
        'checkpoint_parameters_finite': finite_tensors(last['model']) and finite_tensors(best['model']),
        'optimizer_tensors_finite': finite_tensors(last['optimizer']),
        'resume_fields_present': all(key in last for key in [
            'optimizer', 'scheduler', 'scaler', 'random_state', 'numpy_state', 'torch_state', 'cuda_states']),
        'validation_keys_match_accepted_catalog': len(predictions) == count == len(validation) and
            not predictions.duplicated(keys[:3]).any() and ordered(predictions).equals(ordered(validation)),
        'probabilities_finite_and_normalized': bool(np.isfinite(probabilities).all() and
            ((probabilities >= 0) & (probabilities <= 1)).all() and
            np.allclose(probabilities.sum(1), 1, rtol=0, atol=1e-6)),
        'selected_probabilities_match_checkpoint': same_weights and
            np.array_equal(probabilities, last['best_probabilities']),
        'all_epoch_probabilities_present': bool(valid_epochs),
        'exposures_recorded': all(0 < item['train_exposures'] <= config['steps_per_epoch'] * config['batch_size']
                                  for item in history),
    }
    report = {
        'utc': datetime.now(timezone.utc).isoformat(), 'experiment': name,
        'status': 'pass' if all(checks.values()) else 'fail',
        'checks': {key: bool(value) for key, value in checks.items()},
        'checkpoint_epoch': last['epoch'], 'best_epoch': best['epoch'],
        'training_exposures': sum(item['train_exposures'] for item in history),
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'scope': 'completed-model CPU checkpoint, optimizer and prediction readback; accepted catalog unchanged',
        'epoch_predictions_error': epoch_error,
    }
    write_json(output / 'final_checkpoint_checks.json', report)
    print(json.dumps(report), flush=True)
    if report['status'] != 'pass':
        raise ValueError(f'{name}: failed {[key for key, value in checks.items() if not value]}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--experiments', nargs='+', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    campaign = json.loads(args.campaign.read_text())
    configs = {config['name']: config for config in campaign['experiments']}
    catalog = pd.read_parquet(args.root / 'catalog.parquet', columns=['run', 'chunk', 'row', 'label', 'split'])
    validation = catalog[catalog.split.eq('validation')]
    for name in args.experiments:
        readback(args.root, configs[name], campaign['hardware'], validation)


if __name__ == '__main__':
    main()
