"""Summarize all training health samples and independently probe the best model."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import torch

from analysis.flavour.inventory import write_json
from analysis.flavour.s3.dataset import PlaneDataset, collate_planes
from analysis.flavour.s3.health import TrainingHealth
from analysis.flavour.s3.models_repair import build_model


ENCODER = ['input', 'local', 'local_pool']


def summarize_samples(records, total_steps, every):
    expected = list(range(1, total_steps + 1, every))
    informative = [r for r in records if r.get('informative_nodes') != 0]
    gradients = [sum(r['gradient_squared_norms'].get(k, 0.) for k in ENCODER) for r in informative]
    activations = {}
    for record in records:
        for name, rows in record['activations'].items():
            activations.setdefault(name, []).extend(rows)
    ranges = {name: {'samples': len(rows), 'RMS_min': min(r['RMS'] for r in rows),
                     'RMS_max': max(r['RMS'] for r in rows), 'min': min(r['min'] for r in rows),
                     'max': max(r['max'] for r in rows),
                     'largest_zero_fraction': max(r['zero_fraction'] for r in rows),
                     'largest_fraction_below_minus5': max(r['fraction_below_minus5'] for r in rows)}
              for name, rows in activations.items()}
    checks = {'every_scheduled_health_sample_present': [r['step'] for r in records] == expected,
              'all_sampled_health_decisions_pass': bool(records) and all(r['healthy'] for r in records),
              'all_sampled_activations_finite': bool(activations) and all(
                  r['all_finite'] == 1 for rows in activations.values() for r in rows),
              'informative_encoder_gradients_nonzero': bool(gradients) and all(v > 0 and np.isfinite(v) for v in gradients)}
    return {'checks': checks, 'sample_count': len(records), 'informative_sample_count': len(informative),
            'encoder_gradient_squared_norm_min': min(gradients) if gradients else None,
            'encoder_gradient_squared_norm_median': float(np.median(gradients)) if gradients else None,
            'encoder_gradient_squared_norm_max': max(gradients) if gradients else None,
            'activation_ranges': ranges}


def inspect(root, name):
    output = root / 'experiments' / name
    manifest = json.loads((output / 'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('completed fit required for independent best-checkpoint health')
    config = manifest['config']
    records = json.loads((output / 'training_health.json').read_text())['records']
    report = summarize_samples(records, config['epochs'] * config['steps_per_epoch'], config['health_every_steps'])
    dataset = PlaneDataset(root, 'validation', config)
    key = ['run', 'chunk', 'row']
    counts = pd.read_parquet(root / 'catalog.parquet', columns=key + ['pixel_channels', 'bar_channels', 'tracker_channels'])
    metadata = dataset.metadata.merge(counts, on=key, validate='one_to_one', how='left')
    occupancy = metadata[[d + '_channels' for d in config['modalities']]].sum(axis=1)
    ids = np.concatenate([np.flatnonzero(metadata.label.eq(c) & occupancy.between(1, 3000))[:2] for c in range(4)])
    batch = collate_planes([dataset[int(i)] for i in ids])
    labels = batch.pop('label'); batch.pop('index')
    state = torch.load(output / 'best.pt', map_location='cpu', weights_only=False)
    model = build_model(config).eval(); model.load_state_dict(state['model'])
    health = TrainingHealth(model); health.start(batch)
    logits = model(batch)
    loss = torch.nn.functional.cross_entropy(logits.float(), labels)
    loss.backward()
    probe = health.finish(logits, 0)
    health.close(); dataset.native.close()
    encoder_gradient = sum(probe['gradient_squared_norms'].get(k, 0.) for k in ENCODER)
    report['checks'].update(best_checkpoint_probe_finite=probe['healthy'] and bool(torch.isfinite(loss)),
                            best_checkpoint_encoder_connected=encoder_gradient > 0,
                            best_checkpoint_scores_vary=probe['unique_batch_probability_vectors'] > 1)
    history = json.loads((output / 'history.json').read_text())
    selected = history[state['epoch'] - 1]['validation']
    predictions = pd.read_parquet(output / 'validation_predictions.parquet')
    from analysis.flavour.s3.metrics import prediction_diagnostics
    diagnostics = prediction_diagnostics(predictions[[f'p{c}' for c in range(4)]].to_numpy())
    # Learning gate: outperform the constant class-prior NLL. This is separate
    # from improvement over another architecture or a promotion threshold.
    prior = np.asarray(manifest['validation_class_counts'], dtype=float)
    prior /= prior.sum()
    constant_nll = float(-(prior[prior > 0] * np.log(prior[prior > 0])).sum())
    report['checks'].update(validation_NLL_better_than_constant_prior=selected['NLL'] < constant_nll,
                            full_validation_scores_nonconstant=not diagnostics['all_events_same_probability_vector'])
    report.update(utc=datetime.now(timezone.utc).isoformat(), experiment=name,
                  best_epoch=state['epoch'], constant_prior_NLL=constant_nll,
                  selected_validation_NLL=selected['NLL'], prediction_diagnostics=diagnostics,
                  independent_best_FP32_probe=probe,
                  probe_events=metadata.iloc[ids][key + ['label']].to_dict('records'),
                  scope='all scheduled BF16 training health samples plus independent CPU FP32 backward on eight nonempty validation events; no fit or threshold selection')
    report['internal_health_and_learning_passed'] = all(report['checks'].values())
    write_json(output / 'internal_health_checks.json', report)
    print(json.dumps({'experiment': name, 'internal_health_and_learning_passed': report['internal_health_and_learning_passed'],
                      'checks': report['checks']}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--experiments', nargs='+', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1); pa.set_cpu_count(1); pa.set_io_thread_count(1)
    for name in args.experiments:
        result = inspect(args.root, name)
        if not result['internal_health_and_learning_passed']:
            raise SystemExit(f'{name}: inspect architecture and training before drawing a conclusion')


if __name__ == '__main__':
    main()
