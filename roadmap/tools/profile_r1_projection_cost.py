"""Compare CNN/ResNet step costs on the same GPU and 32 fixed real Train events."""

import argparse
import gc
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd
import torch

from analysis.flavour.inventory import write_json
from analysis.flavour.v2.fit import fit_event_positions
from analysis.flavour.v2.profile import device_batch
from analysis.flavour.v2.r1.config import recipe
from analysis.flavour.v2.r1.data import R1Dataset, collate
from analysis.flavour.v2.r1.models import build_model
from analysis.flavour.v2.r1.tasks import task_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--warmup', type=int, default=16)
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    r0 = Path(json.loads((args.root/'campaign.json').read_text())['r0'])
    config = recipe(r0, 'B1', 'T4')
    bf16 = torch.cuda.get_device_capability()[0]>=8
    keys = pd.read_parquet(r0/'fit/fit32_a2_events.parquet')
    with tempfile.TemporaryDirectory(prefix='r1-projection-cost-') as scratch:
        data = R1Dataset(r0/'micro_data', 'train', config, cache_dir=scratch)
        positions = fit_event_positions(keys, data.metadata).reshape(-1, 8).T.reshape(-1)
        events = [data[int(i)] for i in positions]
        data.native.close()
        batches = [device_batch(collate(events[i:i+8]), 'cuda:0') for i in range(0, len(events), 8)]
    measurements = []
    for variant in ['resnet', 'base']:
        torch.manual_seed(42)
        model = build_model({**config, 'cnn_variant': variant}).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0., weight_decay=0.)
        model.train()
        samples = []
        torch.cuda.reset_peak_memory_stats()
        for step in range(args.warmup+args.steps):
            batch = batches[step % len(batches)]
            torch.cuda.synchronize()
            tick = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
                loss = task_loss(model(batch).float(), batch['label'], 'T4')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            if step>=args.warmup:
                samples.append(time.perf_counter()-tick)
        measurements.append({'variant': variant, 'parameters': sum(p.numel() for p in model.parameters()),
            'batch_size': 8, 'measured_steps': args.steps, 'warmup_steps': args.warmup,
            'median_step_seconds': float(np.median(samples)), 'mean_step_seconds': float(np.mean(samples)),
            'step_seconds_10pct_90pct': np.quantile(samples, [.1,.9]).tolist(),
            'events_per_measured_second': 8*len(samples)/sum(samples),
            'GPU_peak_MiB': torch.cuda.max_memory_allocated()/2**20})
        del model, optimizer, loss
        gc.collect()
        torch.cuda.empty_cache()
    write_json(args.root/'audit/projection_cost_profile.json', {'status': 'complete',
        'GPU': torch.cuda.get_device_name(), 'capability': list(torch.cuda.get_device_capability()),
        'precision': 'BF16' if bf16 else 'FP32', 'task': 'T4', 'unique_Train_events': len(keys),
        'keys': keys[['run','chunk','row','label']].to_dict('records'), 'measurements': measurements,
        'operation_seconds': time.perf_counter()-started,
        'scope': 'same-device fixed 32 informative Train events; preloaded batches; forward/backward/AdamW with lr=0; excludes EOS, loading and validation; no generalization or full-campaign throughput claim',
        'Cal_A_Cut_B_Confirm_payload_read': False})
    print(json.dumps(measurements), flush=True)


if __name__ == '__main__':
    main()
