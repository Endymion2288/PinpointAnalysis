"""Fixed-checkpoint path use on a deterministic, occupancy-spanning val probe.

Permute cached branches between events with the same modality presence. This
describes reliance of the fitted classifier, not a retrained ablation or a
population performance estimate. All encoder passes use CPU FP32.
"""
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
from analysis.flavour.s3.models_repair import build_model


def presence_permutation(presence, seed=2026):
    """A deterministic derangement inside each observed-presence stratum."""
    rng = np.random.default_rng(seed)
    permutation = np.arange(len(presence))
    for group in np.unique(presence, axis=0):
        ids = np.flatnonzero(np.all(presence == group, axis=1))
        if len(ids) > 1:
            shuffled = rng.permutation(ids)
            permutation[shuffled] = np.roll(shuffled, 1)
    return permutation


def branch_slices(dim, direct):
    names = ['regional_pixel', 'regional_bar', 'regional_tracker', 'global_latent']
    if direct:
        names += ['local_pixel', 'local_bar', 'local_tracker']
    result = {name: slice(2 * dim * i, 2 * dim * (i + 1)) for i, name in enumerate(names)}
    result['response_summary'] = slice(2 * dim * len(names), (2 * len(names) + 1) * dim)
    return result


def effective_rank(tokens):
    """Entropy rank of the uncentered squared singular spectrum, in FP32."""
    values = tokens.detach().float()
    singular = torch.linalg.svdvals(values)
    energy = singular.square()
    probability = energy / energy.sum(-1, keepdim=True).clamp_min(1e-30)
    rank = (-(probability * probability.clamp_min(1e-30).log()).sum(-1)).exp()
    rank = torch.where(energy.sum(-1) > 0, rank, torch.zeros_like(rank))
    return rank


def head_logits(model, context, nonlinear=True, linear=True):
    value = model.head_input_norm(context)
    logits = value.new_zeros((len(value), 4))
    if nonlinear:
        logits = logits + model.head(value)
    if linear:
        logits = logits + model.linear_readout(value)
    return logits


def contrast(original, changed, labels):
    p, q = original.softmax(-1), changed.softmax(-1)
    before = torch.nn.functional.cross_entropy(original, labels)
    after = torch.nn.functional.cross_entropy(changed, labels)
    return {'probe_NLL': float(after), 'probe_delta_NLL': float(after - before),
            'probability_change_RMS': float((p - q).square().mean().sqrt()),
            'argmax_change_fraction': float((p.argmax(-1) != q.argmax(-1)).float().mean())}


def diagnose(root, name, per_class=16):
    directory = root / 'experiments' / name
    manifest = json.loads((directory / 'manifest.json').read_text())
    config = manifest['config']
    if manifest['status'] != 'complete' or config['architecture'] != 'plane_module_v2':
        raise ValueError('completed revision 2 model required')
    dataset = PlaneDataset(root, 'validation', config)
    key = ['run', 'chunk', 'row']
    catalog = pd.read_parquet(root / 'catalog.parquet', columns=key + [
        d + '_channels' for d in config['modalities']])
    metadata = dataset.metadata.merge(catalog, on=key, validate='one_to_one', how='left')
    occupancy = metadata[[d + '_channels' for d in config['modalities']]].sum(axis=1).to_numpy()
    # Include the complete nonempty occupancy range, with no node cap. Labels
    # balance this diagnostic only; no model/threshold selection uses it.
    selections = []
    for label in range(4):
        candidates = np.flatnonzero(metadata.label.eq(label).to_numpy() & (occupancy > 0))
        ordered = candidates[np.argsort(occupancy[candidates], kind='stable')]
        selections.append(ordered[np.linspace(0, len(ordered) - 1, per_class).astype(int)])
    indexes = np.stack(selections, axis=1).reshape(-1)
    model = build_model(config).eval()
    state = torch.load(directory / 'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'])
    contexts, outputs, presences, spectral = [], [], [], {}
    handles = [model.head_input_norm.register_forward_pre_hook(
        lambda module, inputs: contexts.append(inputs[0].detach().clone()))]

    def record(name):
        def hook(module, inputs, output):
            # Explicit sampling scope for the variable number of regions.
            value = output.detach().float()[:128]
            spectral.setdefault(name, []).append({
                'groups': len(value), 'queries': value.shape[1],
                'mean_effective_rank': float(effective_rank(value).mean()),
                'activation_RMS': float(value.square().mean().sqrt()),
                'within_group_query_variation_RMS': float(value.std(1, unbiased=False).square().mean().sqrt())})
        return hook

    for field in ['module_read', 'latent_read', 'global_norm']:
        handles.append(getattr(model, field).register_forward_hook(record(field)))
    try:
        with torch.inference_mode():
            for start in range(0, len(indexes), 8):
                batch = collate_planes([dataset[int(i)] for i in indexes[start:start + 8]])
                outputs.append(model(batch))
                presence = torch.zeros((len(batch['index']), 3), dtype=torch.bool)
                for modality in range(3):
                    active = batch['block_detector'] == modality
                    presence[batch['event_index'][active], modality] = True
                presences.append(presence)
    finally:
        for handle in handles:
            handle.remove()
        dataset.native.close()
    context, original = torch.cat(contexts), torch.cat(outputs)
    presence = torch.cat(presences).numpy()
    labels = torch.tensor(metadata.iloc[indexes].label.to_numpy(), dtype=torch.long)
    slices = branch_slices(model.dim, model.direct)
    permutation = presence_permutation(presence)
    groups = {name: [name] for name in slices}
    groups['all_regional'] = [name for name in slices if name.startswith('regional_')]
    groups['all_spatial'] = [name for name in slices if name != 'response_summary']
    if model.direct:
        groups['all_local'] = [name for name in slices if name.startswith('local_')]
    with torch.inference_mode():
        reconstructed = head_logits(model, context)
        torch.testing.assert_close(reconstructed, original, rtol=1e-5, atol=2e-6)
        changed = {}
        for group, members in groups.items():
            swapped = context.clone()
            for member in members:
                interval = slices[member]
                swapped[:, interval] = context[permutation, interval]
            changed[group] = contrast(original, head_logits(model, swapped), labels)
        head_controls = {
            'nonlinear_head_only': contrast(original, head_logits(model, context, linear=False), labels),
            'linear_readout_only': contrast(original, head_logits(model, context, nonlinear=False), labels)}
    report = {
        'utc': datetime.now(timezone.utc).isoformat(), 'experiment': name, 'best_epoch': state['epoch'],
        'scope': 'CPU FP32 fixed-checkpoint diagnostic; occupancy-quantile balanced nonempty validation probe; no fit or model selection; not population or held-out test metrics',
        'events': metadata.iloc[indexes][key + ['label']].to_dict('records'),
        'occupancy_range_nodes': [int(occupancy[indexes].min()), int(occupancy[indexes].max())],
        'probe_events': len(indexes), 'probe_NLL': float(torch.nn.functional.cross_entropy(original, labels)),
        'permutation_seed': 2026, 'donor_probe_positions': permutation.tolist(),
        'permuted_fraction': float(np.mean(permutation != np.arange(len(permutation)))),
        'presence_preserved': bool(np.array_equal(presence, presence[permutation])),
        'cached_head_matches_original': True, 'branch_permutation': changed, 'head_controls': head_controls,
        'branch_between_event_variation_RMS': {name: float(context[:, interval].std(0, unbiased=False).square().mean().sqrt())
                                                for name, interval in slices.items()},
        'token_spectrum': spectral,
        'token_spectrum_scope': 'first at most 128 groups of each 8-event batch; uncentered squared-singular-spectrum entropy rank; no collapse threshold',
        'interpretation': 'permutation measures fitted path reliance with event-wise correlations broken; redundancy and interactions can hide useful information; head-only removals are interventions without retraining; low query rank alone does not establish training failure'}
    write_json(directory / 'path_diagnostics.json', report)
    print(json.dumps({'experiment': name, 'probe_NLL': report['probe_NLL'],
                      'permuted_fraction': report['permuted_fraction'],
                      'branch_delta_NLL': {k: v['probe_delta_NLL'] for k, v in changed.items()}}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--experiments', nargs='+', required=True)
    parser.add_argument('--per-class', type=int, default=16)
    args = parser.parse_args()
    if args.per_class < 2:
        parser.error('--per-class must be at least 2')
    torch.set_num_threads(1); pa.set_cpu_count(1); pa.set_io_thread_count(1)
    for name in args.experiments:
        diagnose(args.root, name, args.per_class)


if __name__ == '__main__':
    main()
