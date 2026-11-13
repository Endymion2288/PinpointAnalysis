"""Measure local compute batching without changing detector windows or tokens."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import torch

from analysis.flavour.inventory import write_json
from analysis.flavour.s3.dataset import PlaneDataset, collate_planes
from analysis.flavour.s3.models_repair import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--task', type=int, default=2)
    parser.add_argument('--chunks', type=int, nargs='+', default=[256, 2048, 4096])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compile', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1); pa.set_cpu_count(1); pa.set_io_thread_count(1)
    torch.manual_seed(42)
    config = json.loads(args.campaign.read_text())['experiments'][args.task]
    dataset = PlaneDataset(args.root, 'train', config)
    catalog = pd.read_parquet(args.root / 'catalog.parquet')
    key = ['run', 'chunk', 'row']
    joined = dataset.metadata[key].merge(catalog, on=key, validate='one_to_one', how='left')
    counts = joined[['pixel_channels', 'bar_channels', 'tracker_channels']].sum(axis=1).to_numpy()
    ranks = (np.asarray([.1, .25, .5, .5, .75, .9, .99, 1]) * (len(dataset)-1)).astype(int)
    ids = np.argsort(counts, kind='stable')[ranks]
    batch = collate_planes([dataset[int(i)] for i in ids])
    labels = batch.pop('label').cuda(); batch.pop('index')
    batch = {k: v.cuda() for k, v in batch.items()}
    model = build_model({**config, 'compile_local': False}).cuda().train()
    rows = []; reference = None
    for chunk in args.chunks:
        if args.compile and chunk != 0:
            if hasattr(model, 'compiled_features'):
                model.compiled_features = torch.compile(model.encode_features, dynamic=True)
            else:
                model.compiled_local = torch.compile(model.encode_local, dynamic=True)
        else:
            if hasattr(model, 'compiled_features'):
                model.compiled_features = None
            else:
                model.compiled_local = None
        model.local_chunk_blocks = chunk or 256
        for repeat in range(4):
            model.zero_grad(set_to_none=True); torch.manual_seed(42)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(batch)
                loss = torch.nn.functional.cross_entropy(logits.float(), labels)
            loss.backward(); torch.cuda.synchronize()
            seconds = time.perf_counter()-start
            gradient = model.input[0].weight.grad.detach().float().clone()
            if reference is None:
                reference = logits.detach().clone(), gradient
            rows.append({'chunk_blocks': chunk or 256, 'compiled': args.compile and chunk != 0,
                         'repeat': repeat, 'seconds': seconds,
                         'peak_mib': torch.cuda.max_memory_allocated()/2**20,
                         'logits_max_abs_difference': float((logits.detach()-reference[0]).abs().max()),
                         'input_gradient_relative_L2_difference': float((gradient-reference[1]).norm()/reference[1].norm()),
                         'finite': bool(torch.isfinite(logits).all() and torch.isfinite(gradient).all())})
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'config': config, 'rows': rows, 'compiled': args.compile,
              'events': joined.iloc[ids][key].to_dict('records'), 'raw_nodes_by_event': counts[ids].tolist(),
              'local_blocks': len(batch['nodes']), 'GPU': torch.cuda.get_device_name(),
              'scope': 'same eight cached train events and identical weights; BF16 encoder/FP32 head, forward+backward; concurrent production processes remain active; repeat0 warmup per chunk, no optimizer step'}
    write_json(args.output, report); dataset.native.close()
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
