"""Measure train error and converged transverse readouts on one fixed encoder."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from torch.utils.data import DataLoader

from analysis.flavour.inventory import write_json
from analysis.flavour.s2.dataset import SpatialBatchSampler
from analysis.flavour.s2.train import init_worker
from analysis.flavour.s3.metrics import bootstrap_weights
from analysis.flavour.s4.low_label import inner_folds
from analysis.flavour.s4.stage_inputs import stage_shards
from analysis.flavour.s5.dataset import PhysicsDataset, collate_physics
from analysis.flavour.s5.model import build_model
from analysis.flavour.s5.train import device_batch


def ridge_fit(features, targets, penalties):
    """Minimize mean squared residual plus lambda times squared weight norm."""
    features, targets = features.astype('float64'), targets.astype('float64')
    center = features.mean(0)
    scale = np.maximum(features.std(0), .001)
    target_mean = targets.mean(0)
    x = (features-center)/scale
    gram = x.T@x/len(x)
    values, vectors = np.linalg.eigh(gram)
    rhs = vectors.T@(x.T@(targets-target_mean)/len(x))
    coefficients = vectors@(rhs/(np.maximum(values, 0)[:, None]+np.asarray(penalties)))
    return center, scale, target_mean, coefficients


def ridge_predict(fit, features):
    center, scale, target_mean, coefficients = fit
    return (features-center)/scale@coefficients+target_mean


def extract(model, dataset, device):
    loader = DataLoader(dataset, batch_sampler=SpatialBatchSampler(dataset, 16, 42, shuffle=False),
        collate_fn=collate_physics, num_workers=1, multiprocessing_context='spawn',
        worker_init_fn=init_worker, prefetch_factor=1, pin_memory=True)
    collected = {k: [] for k in ['index', 'features', 'prediction']}
    model.eval()
    with torch.no_grad():
        for cpu in loader:
            batch = device_batch(cpu, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == 'cuda' and torch.cuda.is_bf16_supported(including_emulation=False)):
                output = model(batch['observation'], batch['objects'])
            collected['index'].append(cpu['observation']['index'].numpy())
            collected['features'].append(output['features'].float().cpu().numpy())
            collected['prediction'].append(output['kinematics'].float().cpu().numpy())
    arrays = {k: np.concatenate(v) for k, v in collected.items()}
    order = np.argsort(arrays['index'])
    if not np.array_equal(arrays['index'][order], np.arange(len(dataset))):
        raise ValueError('feature extraction did not visit every selected event once')
    return {k: v[order] for k, v in arrays.items()}


def component_report(target, prediction, reference):
    return {'events': len(target), 'MSE': ((prediction-target)**2).mean(0).tolist(),
            'constant_MSE': ((reference-target)**2).mean(0).tolist(),
            'bias': (prediction-target).mean(0).tolist(),
            'prediction_std': prediction.std(0).tolist(), 'target_std': target.std(0).tolist(),
            'correlation': [float(np.corrcoef(prediction[:, i], target[:, i])[0, 1])
                            if prediction[:, i].std() > 0 else None for i in range(target.shape[1])]}


def run(args):
    config = json.loads(args.campaign.read_text())['experiments'][args.task]
    source = args.root/'experiments'/config['source']
    manifest = json.loads((source/'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('complete source model required')
    output = args.root/'regression_diagnostic'
    output.mkdir(exist_ok=True)
    started = time.monotonic()
    torch.set_num_threads(1); pa.set_cpu_count(1); pa.set_io_thread_count(1)
    scratch = Path(os.environ.get('PINPOINT_S5_SCRATCH', os.environ.get('_CONDOR_SCRATCH_DIR', '/tmp')))
    local = scratch/'native_inputs'
    paths = sorted(p for p in (args.root/'shards').iterdir() if (p/'observations.parquet').exists())
    staging = stage_shards(paths, local, workers=3)
    os.environ['PINPOINT_NATIVE_SHARD_ROOT'] = str(local)
    os.environ['PINPOINT_CLASSIFICATION_CACHE_MIB'] = '256'
    device = torch.device(args.device)
    checkpoint = torch.load(source/'best.pt', map_location='cpu', weights_only=False)
    model = build_model(manifest['config']).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    arrays, datasets = {}, {}
    for split in ['train', 'validation']:
        dataset = PhysicsDataset(args.root, split, manifest['config'])
        dataset.object_mean = checkpoint['object_mean']
        dataset.object_scale = checkpoint['object_scale']
        arrays[split] = extract(model, dataset, device)
        datasets[split] = dataset
        print(json.dumps({'split': split, 'events': len(dataset), 'seconds': time.monotonic()-started}), flush=True)
    train, validation = datasets['train'], datasets['validation']
    valid_train, valid_validation = train.targets['kinematics_mask'], validation.targets['kinematics_mask']
    target_train, target_validation = train.targets['kinematics'][valid_train], validation.targets['kinematics'][valid_validation]
    mean = target_train.mean(0)
    x, xv = arrays['train']['features'][valid_train], arrays['validation']['features'][valid_validation]
    fold = inner_folds(train.metadata, folds=3)[valid_train]
    fit, held = fold != 0, fold == 0
    penalties = config['ridge_penalties']
    scores = []
    for penalty in penalties:
        fitted = ridge_fit(x[fit], target_train[fit, :2], [penalty, penalty])
        scores.append(((ridge_predict(fitted, x[held])-target_train[held, :2])**2).mean(0))
    selected = np.asarray(penalties)[np.argmin(scores, axis=0)]
    fitted = ridge_fit(x, target_train[:, :2], selected)
    predicted = ridge_predict(fitted, xv)
    full_prediction = np.zeros((len(validation), 2))
    full_prediction[valid_validation] = predicted
    key = ['run', 'chunk', 'row']
    frame = validation.metadata[key+['family_hi', 'family_lo', 'label']].copy()
    frame['index'] = np.arange(len(frame))
    frame = frame.sort_values(key)
    weights, blocks = bootstrap_weights(frame, 400)
    order = frame['index'].to_numpy()
    mask = valid_validation[order]
    residual = full_prediction[order][mask]-validation.targets['kinematics'][order][mask, :2]
    constant = mean[:2]-validation.targets['kinematics'][order][mask, :2]
    difference = residual**2-constant**2
    eligible_weights = weights[:, mask]
    samples = eligible_weights@difference/eligible_weights.sum(1, keepdims=True)
    original = pd.read_parquet(source/'validation_predictions.parquet').sort_values(key)
    replay_difference = float(np.max(np.abs(arrays['validation']['prediction'][order]-
        original[[f'pred_kinematics_{i}' for i in range(4)]].to_numpy())))
    if replay_difference > .003:
        raise ValueError('component readback differs from selected checkpoint')
    result = {'source': config['source'], 'status': 'complete', 'config': config,
        'GPU': torch.cuda.get_device_name() if device.type == 'cuda' else None,
        'seconds': time.monotonic()-started, 'staging': staging, 'feature_dimensions': x.shape[1],
        'selected_checkpoint_component_replay_difference': replay_difference,
        'checkpoint_train': component_report(target_train, arrays['train']['prediction'][valid_train], mean),
        'checkpoint_validation': component_report(target_validation, arrays['validation']['prediction'][valid_validation], mean),
        'ridge_train': component_report(target_train[:, :2], ridge_predict(fitted, x), mean[:2]),
        'ridge_validation': component_report(target_validation[:, :2], predicted, mean[:2]),
        'ridge_penalties': selected.tolist(), 'train_inner_scores': np.asarray(scores).tolist(),
        'ridge_component_MSE_minus_constant': [{'estimate': float(delta),
            'block_CI95': np.quantile(samples[:, i], [.025, .975]).tolist()}
            for i, delta in enumerate(difference.mean(0))],
        'bootstrap': {'replicates': 400, 'blocks': blocks}, 'calibration_test_used': False,
        'scope': 'diagnostic frozen representation, encoder fitted on all selected train labels; ridge regularization selected within train, not an independent encoder cross-validation; signed encoded px/py only'}
    write_json(output/'summary.json', result)
    np.savez_compressed(output/'ridge_readout.npz', center=fitted[0], scale=fitted[1],
        target_mean=fitted[2], coefficients=fitted[3], selected_penalties=selected)
    for i in range(2):
        frame[f'ridge_encoded_p{i}'] = full_prediction[order, i]
    frame['kinematics_valid'] = valid_validation[order]
    frame.drop(columns='index').to_parquet(output/'validation_predictions.parquet', index=False)
    for dataset in datasets.values():
        dataset.native.close()
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--task', type=int, required=True)
    parser.add_argument('--device', default='cuda')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
