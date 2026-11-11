"""Apply the prespecified H100 execution gate without quality-based selection."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path


def execution_decision(rule, activation, record):
    """A complete first epoch and every execution check are required."""
    if record.get('status') == 'pending':
        return {**record, 'status': 'pending'}
    checks = dict(record['checks'])
    seconds = record['first_epoch_wall_seconds']
    attempted = record['attempted_steps']
    applied = record['applied_steps']
    fraction = applied / attempted if attempted else 0.
    checks.update(
        first_epoch_wall_finite=math.isfinite(seconds),
        first_epoch_wall_within_frozen_limit=(
            0 < seconds <= rule['H100_gate_maximum_epoch_wall_seconds']),
        applied_fraction_within_frozen_limit=(
            attempted > 0 and activation['minimum_applied_over_attempted_steps'] <= fraction <= 1),
    )
    return {**record, 'checks': checks,
            'first_epoch_H100_over_V100_wall_ratio': seconds / rule['V100_first_epoch_wall_seconds'],
            'applied_over_attempted_steps': fraction,
            'status': 'pass' if all(checks.values()) else 'fail'}


def overall_status(records):
    statuses = [r['status'] for r in records]
    if 'fail' in statuses:
        return 'fail'
    return 'pass' if len(statuses) == 2 and all(s == 'pass' for s in statuses) else 'pending'


def read_execution(root, config, expected):
    directory = root / 'experiments' / config['name']
    if not all((directory / name).exists() for name in ['manifest.json', 'history.json', 'last.pt']):
        return {'experiment': config['name'], 'status': 'pending', 'reason': 'no_committed_first_epoch'}
    history = json.loads((directory / 'history.json').read_text())
    if not history:
        return {'experiment': config['name'], 'status': 'pending', 'reason': 'no_committed_first_epoch'}
    import numpy as np
    import pandas as pd
    import torch

    manifest = json.loads((directory / 'manifest.json').read_text())
    checkpoint = torch.load(directory / 'last.pt', map_location='cpu', weights_only=False)
    first = checkpoint['history'][0]
    probabilities = pd.read_parquet(directory / 'epoch_predictions.parquet',
                                    columns=['row', 'epoch', 'p0', 'p1', 'p2', 'p3'])
    probabilities = probabilities[probabilities.epoch.eq(1)]
    scores = probabilities[['p0', 'p1', 'p2', 'p3']].to_numpy()
    optimizer_tensors = [value for state in checkpoint['optimizer']['state'].values()
                         for value in state.values() if isinstance(value, torch.Tensor)]
    attempted = first['attempted_optimizer_steps_cumulative']
    applied = first['optimizer_steps_cumulative']
    checks = {
        'first_epoch_record': first['epoch'] == 1,
        'frozen_configuration': manifest['config'] == checkpoint['config'] == config,
        'expected_GPU': manifest['gpu_name'] == expected['GPU'],
        'expected_torch': manifest['torch'] == expected['torch'],
        'expected_precision': expected['precision'] == 'BF16' and manifest['mixed_precision'] == 'torch.bfloat16',
        'expected_python': manifest['python'].startswith('3.12.'),
        'expected_subset_size': manifest['train_events'] == 80000 and manifest['validation_events'] == 10000,
        'expected_class_counts': manifest['train_class_counts'] == [20000] * 4 and
                                 manifest['validation_class_counts'] == [2500] * 4,
        'calibration_test_unused': manifest['calibration_test_used'] is False,
        'expected_attempted_steps': attempted == config['steps_per_epoch'],
        'first_epoch_train_loss_finite': math.isfinite(first['train_loss']),
        'first_epoch_validation_rows': len(scores) == 10000 and probabilities.row.nunique() == 10000,
        'first_epoch_probabilities_finite': bool(np.isfinite(scores).all()),
        'first_epoch_probability_range': bool(((scores >= 0) & (scores <= 1)).all()),
        'first_epoch_probability_sums': bool(np.allclose(scores.sum(axis=1), 1., atol=2e-6, rtol=0)),
        'checkpoint_parameters_finite': all(bool(torch.isfinite(v).all()) for v in checkpoint['model'].values()),
        'optimizer_tensors_finite': bool(optimizer_tensors) and all(bool(torch.isfinite(v).all()) for v in optimizer_tensors),
        'checkpoint_counters_consistent': checkpoint['optimizer_steps'] + checkpoint['amp_skipped_steps'] ==
                                          checkpoint['epoch'] * config['steps_per_epoch'],
        'first_epoch_counters_consistent': applied + first['amp_skipped_steps_cumulative'] == attempted,
        'resume_fields_present': all(k in checkpoint for k in ['scheduler', 'scaler', 'random_state',
                                                             'numpy_state', 'torch_state', 'cuda_states']),
    }
    return {'experiment': config['name'], 'status': 'measured', 'checks': checks,
            'observed_utc': datetime.now(timezone.utc).isoformat(),
            'first_epoch_wall_seconds': first['wall_seconds'], 'attempted_steps': attempted,
            'applied_steps': applied, 'AMP_skipped_steps': first['amp_skipped_steps_cumulative'],
            'checkpoint_observed_epoch': checkpoint['epoch'], 'code_commit': manifest['code_commit'],
            'GPU': manifest['gpu_name'], 'torch': manifest['torch'], 'precision': manifest['mixed_precision']}


def run(args):
    contract = json.loads(args.campaign.read_text())
    activation = contract['activation_gate']
    records = []
    for rule in activation['reference_epochs']:
        config = contract['experiments'][rule['task']]
        cache = args.root / 'experiments' / config['name'] / 'first_epoch_execution_gate.json'
        if cache.exists():
            existing = json.loads(cache.read_text())
            if (existing['frozen_rule'] != rule or existing['frozen_config'] != config or
                    existing['expected_environment'] != contract['hardware'] or
                    existing['frozen_activation_gate'] != activation):
                raise ValueError('cached execution evidence belongs to a different frozen contract')
            record = existing['record']
        else:
            record = execution_decision(rule, activation, read_execution(args.root, config, contract['hardware']))
            if record['status'] != 'pending':
                cache.write_text(json.dumps({'frozen_rule': rule, 'frozen_config': config,
                                             'expected_environment': contract['hardware'],
                                             'frozen_activation_gate': activation,
                                             'record': record}, indent=2) + '\n')
        records.append(record)
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'campaign': str(args.campaign),
              'status': overall_status(records), 'records': records,
              'classification_quality_used_for_hardware_selection': False,
              'scope': 'both full first epochs; program wall includes preparation/training/validation; allocation charged separately',
              'scientific_S2_complete': False}
    (args.root / 'execution_gate.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
