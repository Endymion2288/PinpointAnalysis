"""Fixed-weight branch interventions on the complete S4 validation partition."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from torch.utils.data import DataLoader

from analysis.flavour.inventory import write_json
from analysis.flavour.s2.dataset import SpatialBatchSampler
from analysis.flavour.s2.evaluate import common_support_weights
from analysis.flavour.s2.train import init_worker, save_parquet
from analysis.flavour.s3.dataset import collate_planes
from analysis.flavour.s3.diagnose_repair_paths import branch_slices, head_logits, presence_permutation
from analysis.flavour.s3.evaluate import align_frame
from analysis.flavour.s3.metrics import MetricCache, NAMES, bootstrap_weights
from analysis.flavour.s3.models_repair import PlaneModuleClassifierV2
from analysis.flavour.s4.dataset import LabelDataset


def permute_context(context, permutation, intervals):
    """Replace only specified pre-head-LN branches with the same event donor."""
    result = context.clone()
    for interval in intervals:
        result[:, interval] = context[permutation, interval]
    return result


def contrast_report(original, changes, raw_weights, common_weights, multiplicities):
    """Average fixed mappings; resample recipient physical blocks in pairs."""
    result = {}
    for domain, weight in [('raw', raw_weights), ('common_support', common_weights)]:
        before = original.compute(weight)[0]
        per_mapping = np.stack([cache.compute(weight)[0] - before for cache in changes])
        samples = []
        for bw in multiplicities:
            w = bw * weight
            baseline = original.compute(w)[0]
            samples.append(np.mean([cache.compute(w)[0] - baseline for cache in changes], axis=0))
        samples = np.stack(samples)
        result[domain] = {}
        for i, name in enumerate(NAMES):
            finite = np.isfinite(samples[:, i])
            result[domain][name] = {
                'delta': float(per_mapping[:, i].mean()),
                'per_mapping_deltas': per_mapping[:, i].tolist(),
                'CI95': np.quantile(samples[finite, i], [.025, .975]).tolist() if finite.any() else [None, None],
                'valid_replicates': int(finite.sum())}
    result['probability_change_RMS'] = [float(np.mean((cache.p - original.p)**2)**.5) for cache in changes]
    result['argmax_change_fraction'] = [float(np.mean(cache.prediction != original.prediction)) for cache in changes]
    return result


def inspect(root, name, config, output):
    directory = root / 'experiments' / name
    manifest = json.loads((directory / 'manifest.json').read_text())
    readback = json.loads((directory / 'final_checkpoint_checks.json').read_text())
    if manifest['status'] != 'complete' or readback['status'] != 'pass':
        raise ValueError('completed, read-back classification checkpoint required: ' + name)
    settings = {**manifest['config'], 'compile_local': False}
    model = PlaneModuleClassifierV2(settings).eval().requires_grad_(False)
    state = torch.load(directory / 'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'], strict=True)
    if state['epoch'] != manifest['best_epoch']:
        raise ValueError('checkpoint epoch differs from completed manifest')
    data = LabelDataset(root, 'validation', settings)
    if len(data) != config['probe_events']:
        raise ValueError('complete fixed validation partition required')
    sampler = SpatialBatchSampler(data, config.get('batch_size', 16), 42, shuffle=False)
    loader = DataLoader(data, batch_sampler=sampler, num_workers=2, collate_fn=collate_planes,
        multiprocessing_context='spawn', worker_init_fn=init_worker, prefetch_factor=1)
    contexts, logits, presences, indices = [], [], [], []
    handle = model.head_input_norm.register_forward_pre_hook(
        lambda module, inputs: contexts.append(inputs[0].detach().clone()))
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for batch in loader:
                indices.append(batch['index'].numpy())
                logits.append(model(batch))
                presence = torch.zeros((len(batch['index']), 3), dtype=torch.bool)
                for modality in range(3):
                    active = batch['block_detector'] == modality
                    presence[batch['event_index'][active], modality] = True
                presences.append(presence)
    finally:
        handle.remove()
        data.native.close()
    order = np.argsort(np.concatenate(indices))
    if not np.array_equal(np.concatenate(indices)[order], np.arange(len(data))):
        raise ValueError('functional probe must cover every validation event exactly once')
    context, original = torch.cat(contexts)[order], torch.cat(logits)[order]
    presence = torch.cat(presences)[order].numpy()
    reference = data.metadata.copy().reset_index(drop=True)
    event_order = reference.sort_values(['run', 'chunk', 'row']).index.to_numpy()
    reference = reference.iloc[event_order].reset_index(drop=True)
    context, original, presence = context[event_order], original[event_order], presence[event_order]
    with torch.inference_mode():
        replay = head_logits(model, context)
        torch.testing.assert_close(replay, original, rtol=1e-5, atol=2e-6)
    saved = align_frame(pd.read_parquet(directory / 'validation_predictions.parquet'), reference)
    for c in range(4):
        reference[f'p{c}'] = original.softmax(-1)[:, c].numpy()
    baseline = MetricCache(reference.label.to_numpy(), reference[[f'p{c}' for c in range(4)]].to_numpy())
    published = MetricCache(saved.label.to_numpy(), saved[[f'p{c}' for c in range(4)]].to_numpy())
    common, support = common_support_weights(reference)
    multiplicities, blocks = bootstrap_weights(reference, config.get('bootstrap', 400))
    mappings = [presence_permutation(presence, seed) for seed in config['permutation_seeds']]
    if not all(np.array_equal(presence, presence[mapping]) for mapping in mappings):
        raise ValueError('modality presence changed under permutation')
    slices = branch_slices(model.dim, model.direct)
    groups = {'global_latent': ['global_latent'],
        'all_regional': [k for k in slices if k.startswith('regional_')],
        'all_local': [k for k in slices if k.startswith('local_')],
        'response_summary': ['response_summary'],
        'all_spatial': [k for k in slices if k != 'response_summary']}
    changes = {}
    with torch.inference_mode():
        for group, members in groups.items():
            caches = []
            for mapping in mappings:
                changed = head_logits(model, permute_context(context, mapping, [slices[k] for k in members]))
                caches.append(MetricCache(baseline.y, changed.softmax(-1).numpy()))
            changes[group] = contrast_report(baseline, caches, np.ones(len(data)), common, multiplicities)
    prediction = reference.copy()
    for i, mapping in enumerate(mappings):
        prediction[f'donor_position_{i}'] = mapping
    save_parquet(prediction, output / (name + '.baseline.parquet'))
    report = {'experiment': name, 'checkpoint_epoch': state['epoch'], 'events': len(data),
        'seconds': time.perf_counter() - started, 'checkpoint_config': manifest['config'],
        'training_code_commit': manifest.get('code_commit'),
        'precision': 'CPU FP32; same complete checkpoint, compilation disabled, evaluation mode',
        'baseline': {'raw': baseline.report(), 'common_support': baseline.report(common)},
        'saved_BF16_comparison': {
            'probability_change_RMS': float(np.mean((baseline.p - published.p)**2)**.5),
            'probability_change_maximum': float(np.abs(baseline.p - published.p).max()),
            'argmax_change_fraction': float(np.mean(baseline.prediction != published.prediction)),
            'NLL_delta': baseline.report()['NLL'] - published.report()['NLL'],
            'macro_AP_delta': baseline.report()['macro_AP'] - published.report()['macro_AP']},
        'cached_head_matches_original': True,
        'cached_head_logit_maximum_error': float((replay - original).abs().max()),
        'presence_preserved': True,
        'presence_strata': [{'presence': pattern.tolist(), 'events': int(np.all(presence == pattern, axis=1).sum())}
            for pattern in np.unique(presence, axis=0)],
        'permuted_fraction': [float(np.mean(mapping != np.arange(len(data)))) for mapping in mappings],
        'permutation_seeds': config['permutation_seeds'], 'branch_permutation': changes,
        'common_support': support, 'resampling_blocks': blocks}
    write_json(output / (name + '.json'), report)
    print(json.dumps({'experiment': name, 'events': len(data),
        'branch_delta_NLL': {k: v['raw']['NLL']['delta'] for k, v in changes.items()},
        'seconds': report['seconds']}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--task', type=int, required=True)
    args = parser.parse_args()
    config = json.loads(args.campaign.read_text())['experiments'][args.task]
    output = args.root / 'experiments' / config['name']
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1); pa.set_cpu_count(1); pa.set_io_thread_count(1)
    started = time.perf_counter()
    manifest = {'utc': datetime.now(timezone.utc).isoformat(), 'status': 'running', 'config': config,
        'GPU': None, 'code_commit': os.environ.get('PINPOINT_CODE_COMMIT') or
            subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
    write_json(output / 'manifest.json', manifest)
    rows = [inspect(args.root, name, config, output) for name in config['checkpoints']]
    write_json(output / 'functional_paths.json', {'experiments': rows,
        'calibration_test_used': False, 'fitting_or_selection': False,
        'scope': 'fixed-weight fitted-path reliance on complete 10k validation; correlations between branches are broken; not retrained ablations or causal physical importance',
        'contrast_sign': 'changed minus original, using the FP32 replay baseline for both',
        'interval_scope': 'pointwise paired connected-chunk bootstrap of recipient events, conditional on checkpoint and three fixed presence-preserving donor mappings; mapping spread is not training-seed uncertainty',
        'saved_prediction_scope': 'BF16 training predictions are compared descriptively; FP32 diagnostic baseline does not replace their validation metrics'})
    manifest.update(status='complete', seconds=time.perf_counter() - started)
    write_json(output / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
