"""B0 detector-proxy MLP pilot. Only train/validation partitions are opened.

No calibration or formal test campaign is performed at S1. Source signatures
are provisional and must be replaced/verified before physics performance claims.
"""

import argparse
import json
from pathlib import Path
import resource
import subprocess
import time

import numpy as np
import pandas as pd
import pyarrow as pa
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

from analysis.flavour.features import FEATURE_NAMES
from analysis.flavour.inventory import write_json, COMPRESSION


def train(args):
    torch.set_num_threads(args.cpus)
    pa.set_cpu_count(args.cpus)
    pa.set_io_thread_count(args.cpus)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    key = ['run', 'chunk', 'row']
    shard_keys = []
    for shard in sorted(args.input.iterdir()):
        if (shard / 'provenance.parquet').exists():
            pairs = pd.read_parquet(shard / 'provenance.parquet', columns=['run', 'chunk']).drop_duplicates()
            shard_keys.extend([(int(r.run), int(r.chunk)) for r in pairs.itertuples()])
    filters = [[('run', '==', run), ('chunk', '==', chunk)] for run, chunk in shard_keys]
    manifest = pd.read_parquet(args.inventory,
                               columns=key + ['split', 'label_status', 'family_hi', 'family_lo'], filters=filters)
    selection = manifest[key + ['split', 'label_status', 'family_hi', 'family_lo']]
    features, labels, provenance = [], [], []
    for shard in sorted(args.input.iterdir()):
        if not (shard / 'b0_features.parquet').exists():
            continue
        x = pd.read_parquet(shard / 'b0_features.parquet')
        p = pd.read_parquet(shard / 'provenance.parquet')
        target = pd.read_parquet(shard / 'targets.parquet')
        joined = p[key].merge(selection, on=key, how='left', validate='one_to_one', sort=False)
        if joined.split.isna().any() or len(x) != len(joined):
            raise ValueError('shard missing from frozen inventory')
        keep = joined.split.isin(['train', 'validation']) & (joined.label_status == 'consistent') & p.usable_join
        features.append(x.loc[keep, FEATURE_NAMES])
        labels.append(target.loc[keep, 'label'])
        provenance.append(joined.loc[keep])
    x = pd.concat(features, ignore_index=True).to_numpy(dtype='float32')
    y = pd.concat(labels, ignore_index=True).to_numpy(dtype='int64')
    provenance = pd.concat(provenance, ignore_index=True)
    is_train = (provenance.split == 'train').to_numpy()
    is_val = ~is_train
    if provenance.groupby(['family_hi', 'family_lo']).split.nunique().max() != 1:
        raise ValueError('source-signature family overlaps partitions')
    scaler = StandardScaler().fit(x[is_train])
    x = scaler.transform(x).astype('float32')
    load_seconds = time.perf_counter() - started
    transfer_start = time.perf_counter()
    xt, yt = torch.tensor(x[is_train], device=device), torch.tensor(y[is_train], device=device)
    xv = torch.tensor(x[is_val], device=device)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    transfer_seconds = time.perf_counter() - transfer_start
    model = nn.Sequential(nn.Linear(len(FEATURE_NAMES), 48), nn.ReLU(), nn.Linear(48, 24), nn.ReLU(), nn.Linear(24, 4)).to(device)
    counts = np.bincount(y[is_train], minlength=4)
    if np.any(counts == 0):
        raise ValueError('missing training class')
    class_weights = torch.as_tensor(counts.sum() / (4 * counts), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    training_start = time.perf_counter()
    best, best_state, history = float('inf'), None, []
    forward_seconds, backward_seconds = 0., 0.
    for epoch in range(args.epochs):
        model.train()
        for indices in torch.randperm(len(xt), device=device).split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t = time.perf_counter()
            loss = loss_fn(model(xt[indices]), yt[indices])
            if device.type == 'cuda':
                torch.cuda.synchronize()
            forward_seconds += time.perf_counter() - t
            t = time.perf_counter()
            loss.backward()
            optimizer.step()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            backward_seconds += time.perf_counter() - t
        model.eval()
        with torch.no_grad():
            p = model(xv).softmax(-1).cpu().numpy()
        val_loss = log_loss(y[is_val], p, labels=[0, 1, 2, 3])
        history.append({'epoch': epoch + 1, 'validation_NLL': val_loss})
        if val_loss < best:
            best = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    with torch.no_grad():
        p = model(xv).softmax(-1).cpu().numpy()
    target = y[is_val]
    class_metrics = {}
    for c in range(4):
        truth = target == c
        class_metrics[str(c)] = {'ROC_AUC': float(roc_auc_score(truth, p[:, c])),
                                 'PR_AUC': float(average_precision_score(truth, p[:, c])),
                                 'events': int(truth.sum())}
    metrics = {'scope': 'S1_validation_smoke_unweighted_sample_priors_no_test',
               'split_status': 'provisional_signature_grouped', 'features': FEATURE_NAMES,
               'class_counts_train': counts.tolist(), 'validation_events': len(target),
               'balanced_accuracy': float(balanced_accuracy_score(target, p.argmax(1))),
               'macro_F1': float(f1_score(target, p.argmax(1), average='macro')),
               'NLL': float(log_loss(target, p)),
               'Brier_multiclass': float(np.mean(np.sum((p - np.eye(4)[target])**2, axis=1))),
               'classwise': class_metrics, 'confusion_counts': confusion_matrix(target, p.argmax(1), labels=[0, 1, 2, 3]).tolist(),
               'load_seconds': load_seconds, 'training_seconds': time.perf_counter() - training_start,
               'device_transfer_seconds': transfer_seconds,
               'forward_seconds': forward_seconds, 'backward_seconds': backward_seconds,
               'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
               'gpu_peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20 if device.type == 'cuda' else None,
               'device': str(device), 'gpu_name': torch.cuda.get_device_name() if device.type == 'cuda' else None,
               'torch_version': torch.__version__, 'seed': args.seed, 'epochs': args.epochs,
               'tracker_input': False, 'normalization_fit': 'train_only', 'history': history}
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': best_state, 'mean': torch.tensor(scaler.mean_), 'scale': torch.tensor(scaler.scale_),
                'features': FEATURE_NAMES, 'seed': args.seed}, args.output / 'b0.pt')
    predictions = provenance.loc[is_val, key].copy()
    predictions['label'] = target
    for c in range(4):
        predictions[f'p{c}'] = p[:, c]
    predictions.to_parquet(args.output / 'validation_predictions.parquet', index=False, compression=COMPRESSION)
    write_json(args.output / 'metrics.json', metrics)
    print(json.dumps({k:v for k,v in metrics.items() if k != 'history'}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--cpus', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    monitor = None
    if args.device.startswith('cuda'):
        monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,name,utilization.gpu,memory.used,power.draw',
                                    '--format=csv,noheader,nounits', '--loop-ms=100',
                                    '--filename=' + str(args.output / 'gpu_usage.csv')],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        train(args)
    finally:
        if monitor is not None:
            monitor.terminate()
            monitor.wait(timeout=5)


if __name__ == '__main__':
    main()
