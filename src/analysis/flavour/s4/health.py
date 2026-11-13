"""Independent FP32 backward on each selected S4 classification checkpoint."""
from datetime import datetime, timezone
import json
import os

import numpy as np
import torch

from analysis.flavour.inventory import write_json
from analysis.flavour.s3.dataset import collate_planes
from analysis.flavour.s3.health import TrainingHealth
from analysis.flavour.s3.models_repair import PlaneModuleClassifierV2
from analysis.flavour.s4.dataset import LabelDataset


def inspect(root, name, catalog):
    directory = root / 'experiments' / name
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('complete selected checkpoint required')
    config = manifest['config']
    path = directory / 'best_checkpoint_health.json'
    if path.exists():
        cached = json.loads(path.read_text())
        if (cached.get('version') == 1 and cached.get('config') == config
            and cached.get('best_epoch') == manifest['best_epoch']
            and cached.get('selected_NLL') == manifest['best_NLL'] and cached.get('passed') is True):
            return cached
    data = LabelDataset(root, 'validation', config)
    counts = catalog[['run', 'chunk', 'row', 'pixel_channels', 'bar_channels', 'tracker_channels']]
    keys = ['run', 'chunk', 'row']
    metadata = data.metadata[keys + ['label']].merge(counts, on=keys, how='left', validate='one_to_one', sort=False)
    occupancy = metadata[[kind + '_channels' for kind in config['modalities']]].sum(axis=1)
    ids = np.concatenate([np.flatnonzero(metadata.label.eq(c) & occupancy.between(1, 3000))[:2] for c in range(4)])
    if len(ids) != 8:
        raise ValueError('eight nonempty diagnostic events required')
    batch = collate_planes([data[int(i)] for i in ids])
    labels = batch.pop('label');batch.pop('index')
    checkpoint = torch.load(directory / 'best.pt', map_location='cpu', weights_only=False)
    model = PlaneModuleClassifierV2({**config, 'compile_local': False}).eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    health = TrainingHealth(model);health.start(batch)
    logits = model(batch)
    loss = torch.nn.functional.cross_entropy(logits.float(), labels)
    loss.backward()
    measured = health.finish(logits, 0)
    health.close();data.native.close()
    input_gradient = measured['gradient_squared_norms'].get('input', 0.)
    silu = measured['activations'].get('head.2', [])
    checks = {'selected_epoch': checkpoint['epoch'] == manifest['best_epoch'],
        'finite_probe': measured['healthy'] and bool(torch.isfinite(loss)),
        'nonempty_observation': measured['informative_nodes'] > 0,
        'input_gradient_connected': bool(np.isfinite(input_gradient)) and input_gradient > 0,
        'scores_vary': measured['unique_batch_probability_vectors'] > 1,
        'SiLU_not_entirely_zero': bool(silu) and all(row['zero_fraction'] < 1. for row in silu)}
    report = {'version': 1, 'utc': datetime.now(timezone.utc).isoformat(), 'config': config,
        'best_epoch': manifest['best_epoch'], 'selected_NLL': manifest['best_NLL'], 'checks': checks,
        'passed': all(checks.values()), 'probe_cross_entropy': float(loss.detach()), 'health': measured,
        'events': metadata.iloc[ids][keys + ['label']].to_dict('records'),
        'precision': 'CPU FP32, eval mode, compilation disabled, fixed selected checkpoint',
        'scope': 'deterministic eight nonempty validation events, two per class; no fitting or threshold selection; complements all scheduled BF16 training samples',
        'code_commit': os.environ.get('PINPOINT_CODE_COMMIT'), 'calibration_test_used': False}
    write_json(path, report)
    return report
