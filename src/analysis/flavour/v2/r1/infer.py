"""Paired nominal/stress and masked-input diagnostics for frozen R1 weights."""

import argparse
import json
import os
from pathlib import Path
import tempfile
import time

import joblib
import numpy as np
import pandas as pd
import torch

from analysis.flavour.inventory import write_json
from analysis.flavour.v2.r1.data import R1Dataset, SUMMARY_NAMES
from analysis.flavour.v2.r1.metrics import quick_report
from analysis.flavour.v2.r1.models import build_model
from analysis.flavour.v2.r1.train import loader_for, predict, save_predictions
from analysis.flavour.v2.profile import shutdown_loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--drop-modality', choices=['pixel', 'bar', 'tracker'])
    parser.add_argument('--checkpoint', default='exposure.pt')
    parser.add_argument('--compile-local', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.perf_counter()
    output = args.root/'diagnostics'/args.name
    output.mkdir(parents=True, exist_ok=False)
    model_dir = args.root/'experiments'/args.model
    manifest = json.loads((model_dir/'manifest.json').read_text())
    config = manifest['config'].copy()
    neural = config['family'] != 'B0'
    capable = neural and torch.cuda.get_device_capability()[0]>=8
    config.update(compile_local=bool(config.get('compile_local') and args.compile_local and capable),
        workers=2, cache_shards=2, drop_modality=args.drop_modality,
        bf16=bool(config.get('bf16', True) and capable))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if config['family'] == 'B0':
        config.update(task=config['task'], batch_size=32)
    with tempfile.TemporaryDirectory(prefix='r1-infer-') as scratch:
        data = R1Dataset(args.input_root, 'validation', config, cache_dir=scratch)
        if config['family'] == 'B0':
            model = joblib.load(model_dir/'model.joblib')
            summaries = np.stack([data[i]['summary'] for i in range(len(data))])
            p = model.predict_proba(pd.DataFrame(summaries, columns=SUMMARY_NAMES))
            indices = np.arange(len(data))
            logits = np.log(p.clip(1e-15))
            inference_seconds = time.perf_counter()-started
        else:
            model = build_model(config).cuda()
            state = torch.load(model_dir/args.checkpoint, map_location='cpu', weights_only=False)
            model.load_state_dict(state['model'], strict=True)
            loader, _ = loader_for(data, config, False, config['seed'])
            indices, p, logits, inference_seconds = predict(model, loader, config['task'], config['bf16'])
            shutdown_loader(loader)
        save_predictions(output/'predictions.parquet', data, indices, p, logits)
        write_json(output/'manifest.json', {'status': 'complete', 'model': args.model, 'input_root': str(args.input_root),
            'drop_modality': args.drop_modality, 'checkpoint': args.checkpoint,
            'GPU': torch.cuda.get_device_name() if neural else 'CPU', 'actual_inference_config': config,
            'allocation_id': os.environ.get('PINPOINT_JOB_ID'),
            'metrics': quick_report(data.metadata.iloc[indices].label.to_numpy(), p),
            'events': len(indices), 'inference_seconds': inference_seconds, 'wall_seconds': time.perf_counter()-started,
            'scope': 'masked-input measures reliance; retrained controls measure incremental modality value',
            'Cal_A_Cut_B_Confirm_payload_read': False})
        data.native.close()


if __name__ == '__main__':
    main()
