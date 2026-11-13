"""Balanced real-train memorization and internal-health gate before S3 revision 2."""
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
from analysis.flavour.s3.health import TrainingHealth
from analysis.flavour.s3.models_repair import build_model


def optimizer_for(model, config):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (decay if parameter.ndim >= 2 and 'embedding' not in name and 'queries' not in name
         and 'latents' not in name else no_decay).append(parameter)
    return torch.optim.AdamW([{'params': decay, 'weight_decay': config.get('weight_decay', .01)},
                              {'params': no_decay, 'weight_decay': 0.}],
                             lr=config.get('learning_rate', .0003), betas=(.9, .95), eps=1e-8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--task', type=int, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(1); pa.set_cpu_count(1); pa.set_io_thread_count(1)
    config = json.loads(args.campaign.read_text())['experiments'][args.task]
    torch.manual_seed(config['seed']); rng = np.random.default_rng(config['seed'])
    dataset = PlaneDataset(args.root, 'train', config)
    catalog = pd.read_parquet(args.root / 'catalog.parquet')
    meta = dataset.metadata[['run', 'chunk', 'row', 'label']].merge(catalog.drop(columns=['label']),
                  on=['run', 'chunk', 'row'], how='left', validate='one_to_one')
    active = meta.pixel_channels + meta.bar_channels
    ids = np.concatenate([np.flatnonzero((meta.label.to_numpy() == c) & active.between(20, 3000))[:8]
                          for c in range(4)])
    assert len(ids) == 32
    items = [dataset[int(i)] for i in ids]
    device = torch.device(args.device)
    model = build_model(config).to(device)
    optimizer = optimizer_for(model, config)
    health = TrainingHealth(model)
    output = args.root / 'pilot' / config['name']
    output.mkdir(parents=True, exist_ok=True)
    records, checks = [], []
    started = time.perf_counter()
    bf16 = device.type == 'cuda' and torch.cuda.is_bf16_supported(including_emulation=False)
    steps = config.get('pilot_steps', 768)
    for step in range(steps + 1):
        if step % 64 == 0 or step == steps:
            model.eval(); probabilities = []
            for start in range(0, len(items), 4):
                batch = collate_planes(items[start:start + 4]); batch.pop('label'); batch.pop('index')
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
                    probabilities.append(model(batch).float().softmax(-1).cpu())
            p = torch.cat(probabilities); labels = torch.tensor([e['label'] for e in items])
            row = {'step': step, 'train_probe_NLL': float(-p[torch.arange(32), labels].clamp_min(1e-20).log().mean()),
                   'train_probe_accuracy': float((p.argmax(-1) == labels).float().mean()),
                   'unique_vectors': len(torch.unique(p, dim=0)), 'wall_seconds': time.perf_counter() - started}
            records.append(row); print(json.dumps(row), flush=True)
            write_json(output / 'progress.json', {'config': config, 'records': records, 'health': checks})
        if step == steps:
            break
        model.train(); selected = np.concatenate([rng.choice(np.arange(c * 8, (c + 1) * 8), 2, replace=False) for c in range(4)])
        batch = collate_planes([items[int(i)] for i in selected]); label = batch.pop('label').to(device); batch.pop('index')
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        diagnostic = step % 64 == 0 or step == steps - 1
        if diagnostic: health.start(batch)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            logits = model(batch); loss = torch.nn.functional.cross_entropy(logits.float(), label)
        loss.backward()
        if diagnostic:
            checks.append(health.finish(logits, step + 1))
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        if not torch.isfinite(loss) or (checks and not checks[-1]['healthy']):
            write_json(output / 'failure.json', {'step': step + 1, 'health': checks, 'loss': float(loss.detach())})
            raise RuntimeError('repair pilot internal health failed')
        for group in optimizer.param_groups:
            group['lr'] = config.get('learning_rate', .0003) * min(1., (step + 1) / 32)
        optimizer.step()
    final = records[-1]
    gate = {'memorization_accuracy_at_least_0_95': final['train_probe_accuracy'] >= .95,
            'memorization_NLL_below_0_25': final['train_probe_NLL'] < .25,
            'nonconstant_real_train_scores': final['unique_vectors'] >= 24,
            'all_internal_health_samples_pass': all(c['healthy'] for c in checks),
            'last_encoder_gradient_nonzero': sum(checks[-1]['gradient_squared_norms'].get(k, 0.)
                                               for k in ['input', 'local', 'local_pool']) > 0}
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'config': config, 'checks': gate,
              'gate_passed': all(gate.values()), 'records': records, 'health': checks,
              'train_events': meta.iloc[ids][['run', 'chunk', 'row', 'label']].to_dict('records'),
              'parameters': sum(p.numel() for p in model.parameters()),
              'GPU': torch.cuda.get_device_name() if device.type == 'cuda' else None,
              'precision': 'BF16 encoder/FP32 head' if bf16 else 'FP32',
              'peak_gpu_mib': torch.cuda.max_memory_allocated() / 2**20 if device.type == 'cuda' else None,
              'wall_seconds': time.perf_counter() - started,
              'scope': '32 nonempty TRAIN events only; memorization/internal health, no validation performance selection'}
    write_json(output / 'gate.json', report)
    health.close(); dataset.native.close()
    if not report['gate_passed']:
        raise SystemExit('repair pilot did not pass; do not launch scientific training')


if __name__ == '__main__':
    main()
