"""Temporary frozen features: one GPU pass, nested CPU readouts, explicit cleanup."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from analysis.flavour.inventory import write_json
from analysis.flavour.s2.train import atomic_output, save_parquet
from analysis.flavour.s4.dataset import LabelDataset
from analysis.flavour.s4.train import build_model, validate_pretraining


FEATURE_KEYS = ('architecture', 'architecture_revision', 'dim', 'compression', 'latents',
    'region_queries', 'region_blocks', 'readout_skip', 'local_chunk_blocks', 'local_execution_version',
    'empty_response_bypass', 'residual_scale', 'modalities', 'pitch_mm', 'window_pixels',
    'signal', 'local_residual', 'pretrained_checkpoint', 'supervised_checkpoint', 'seed',
    'response_summary_version', 'linear_probe_passes')


def select_features(bank_metadata, desired_metadata, features):
    keys = ['run', 'chunk', 'row']
    source = bank_metadata[keys + ['label']].assign(feature_row=np.arange(len(bank_metadata)))
    joined = desired_metadata[keys + ['label']].merge(source, on=keys, how='left',
        suffixes=('', '_bank'), sort=False, validate='one_to_one')
    if joined.feature_row.isna().any() or not joined.label.equals(joined.label_bank):
        raise ValueError('feature bank must cover the exact requested labeled events')
    ids = joined.feature_row.to_numpy(dtype=np.int64)
    if np.array_equal(ids, np.arange(len(bank_metadata))):
        return features
    return {key: value[ids] for key, value in features.items()}


def load(root, config):
    source = root / 'experiments' / config['probe_feature_source']
    manifest = json.loads((source / 'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('completed frozen feature bank required')
    for key in FEATURE_KEYS:
        if manifest['config'].get(key) != config.get(key):
            raise ValueError('frozen feature representation mismatch: ' + key)
    bank = root / 'temporary_features' / config['probe_feature_source']
    result = []
    for split in ['train', 'validation']:
        metadata = pd.read_parquet(bank / (split + '_metadata.parquet'))
        arrays = {branch: np.load(bank / (split + '_' + branch + '.npy'), mmap_mode='r')
            for branch in config.get('probe_branches', ['full', 'global'])}
        desired = LabelDataset(root, split, config)
        selected = select_features(metadata, desired.metadata, arrays)
        result.append((selected, desired.metadata.copy()))
        desired.native.close()
    return result


def main():
    from analysis.flavour.s4.probes import extract
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--task', type=int, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    config = json.loads(args.campaign.read_text())['experiments'][args.task]
    output = args.root / 'experiments' / config['name']; output.mkdir(parents=True, exist_ok=True)
    bank = args.root / 'temporary_features' / config['name']; bank.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter(); torch.set_num_threads(1)
    torch.manual_seed(config.get('seed', 42)); np.random.seed(config.get('seed', 42))
    validate_pretraining(config)
    if not json.loads((args.root / 'experiments' / config['required_solver_probe'] / 'solver_probe.json').read_text())['passed']:
        raise RuntimeError('passing solver regression required before feature-bank recovery')
    write_json(output / 'manifest.json', {'status': 'running', 'config': config,
        'utc': datetime.now(timezone.utc).isoformat()})
    model = build_model(config).to(args.device).eval().requires_grad_(False)
    if config.get('supervised_checkpoint'):
        model.load_state_dict(torch.load(config['supervised_checkpoint'], map_location=args.device,
            weights_only=False)['model'], strict=True)
    rows = {}
    for split in ['train', 'validation']:
        dataset = LabelDataset(args.root, split, config)
        features = extract(model, dataset, {**config, 'extract_branches': ['full', 'global']},
            torch.device(args.device), output / 'progress.json', split)
        save_parquet(dataset.metadata, bank / (split + '_metadata.parquet'))
        rows[split] = {'events': len(dataset), 'arrays': {}}
        for branch, array in features.items():
            path = bank / (split + '_' + branch + '.npy')
            atomic_output(path, lambda stream, value=array: np.save(stream, value, allow_pickle=False))
            rows[split]['arrays'][branch] = {'shape': list(array.shape), 'dtype': str(array.dtype),
                'bytes': path.stat().st_size}
        dataset.native.close(); del features, dataset
    write_json(output / 'manifest.json', {'status': 'complete', 'config': config, 'splits': rows,
        'seconds': time.perf_counter() - started, 'GPU': torch.cuda.get_device_name() if args.device == 'cuda' else None,
        'utc': datetime.now(timezone.utc).isoformat(), 'code_commit': os.environ.get('PINPOINT_CODE_COMMIT'),
        'scope': 'temporary train-100k and validation-10k frozen representations; '
            'nested label readouts select exact train rows and fit their own scaler; remove arrays after CPU acceptance'})


if __name__ == '__main__':
    main()
