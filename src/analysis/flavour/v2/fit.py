"""Real balanced 32-event optimization and internal-health gate, Train only."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import torch

from analysis.flavour.inventory import write_json
from analysis.flavour.s3.dataset import collate_planes
from analysis.flavour.s3.health import TrainingHealth
from analysis.flavour.s3.models_repair import build_model
from analysis.flavour.v2.data import TimedPlaneDataset
from analysis.flavour.v2.profile import MemoryMonitor, device_batch
from analysis.flavour.v2.partitions import KEY


def select_fit_events(metadata):
    counts = metadata[['pixel_channels', 'bar_channels', 'tracker_channels']]
    eligible = metadata[(counts.pixel_channels > 0) & (counts.bar_channels > 0) & (counts.sum(axis=1) < 5000)].copy()
    eligible['workload'] = counts.sum(axis=1)
    rows = []
    for label in range(4):
        group = eligible[eligible.label.eq(label)].sort_values(['workload', 'micro_index'])
        if len(group) < 8:
            raise ValueError('32-event fitting needs eight distinct informative Train events per class')
        rows.append(group.iloc[np.linspace(0, len(group)-1, 8).astype(int)])
    return pd.concat(rows, ignore_index=True)


def evaluate_fit(model, batches, device):
    model.eval()
    logits, labels = [], []
    with torch.inference_mode():
        for batch in batches:
            gpu = device_batch(batch, device)
            logits.append(model(gpu))
            labels.append(gpu['label'])
    value, truth = torch.cat(logits), torch.cat(labels)
    probability = value.softmax(-1)
    return {'NLL': float(torch.nn.functional.cross_entropy(value, truth)),
            'accuracy': float((value.argmax(-1) == truth).float().mean()),
            'probability_std': probability.std(0, unbiased=False).cpu().tolist(),
            'probability_vectors': len(torch.unique(probability, dim=0)),
            'finite': bool(torch.isfinite(value).all())}


def fit_event_positions(selected, metadata):
    """Resolve physical keys; sample-list ordinals can differ from shard order."""
    positions = pd.MultiIndex.from_frame(metadata[KEY]).get_indexer(pd.MultiIndex.from_frame(selected[KEY]))
    if (positions < 0).any() or len(np.unique(positions)) != len(selected):
        raise ValueError('fit event identities are missing or duplicated')
    if not np.array_equal(metadata.iloc[positions].label.to_numpy(), selected.label.to_numpy()):
        raise ValueError('fit event labels differ from selected physical keys')
    return positions


def run(args):
    torch.set_num_threads(1)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    torch.manual_seed(42)
    config = next(row['config'] for row in json.loads(args.candidates.read_text())['models']
                  if row['name'] == args.name).copy()
    config.update(compile_local=False, cache_shards=2, local_chunk_blocks=256)
    output = args.root/'fit'
    output.mkdir(exist_ok=True)
    path = output/f'fit32_{args.attempt}.json'
    if path.exists():
        raise ValueError('fit attempts must be immutable')
    selected = select_fit_events(pd.read_parquet(args.root/'micro_data/catalog.parquet'))
    started = time.perf_counter()
    device = torch.device('cuda:0')
    record = {'status': 'running', 'UTC_started': datetime.now(timezone.utc).isoformat(),
              'seed': 42, 'events': len(selected),
              'class_counts': selected.groupby('label').size().to_dict(), 'config': config,
              'GPU': torch.cuda.get_device_name(device), 'precision': 'FP32',
              'selection': 'eight real Train events/class at pixel+bar occupancy quantiles below 5000 native channels',
              'learning_rate': args.learning_rate, 'weight_decay': 0., 'batch_size': 8,
              'scope': 'optimization/health only; informative small-sample memorization, no generalization estimate',
              'Confirm_responses_read': False, 'history': [], 'health': []}
    with tempfile.TemporaryDirectory(prefix='pinpoint-r0-fit-') as scratch, MemoryMonitor() as memory:
        data = TimedPlaneDataset(args.root/'micro_data', 'train', config, cache_dir=scratch)
        # Round-robin classes gives every optimization batch two/class.
        selected['dataset_index'] = fit_event_positions(selected, data.metadata)
        selected.to_parquet(output/f'fit32_{args.attempt}_events.parquet', index=False)
        indexes = selected.dataset_index.to_numpy().reshape(4, 8).T.reshape(-1)
        items = [data[int(i)] for i in indexes]
        if np.bincount([item['label'] for item in items], minlength=4).tolist() != [8]*4:
            raise ValueError('actual loaded fit labels are not balanced')
        record['loaded_event_keys_and_labels_match_selection'] = True
        data.native.close()
        batches = [collate_planes(items[i:i+8]) for i in range(0, len(items), 8)]
        model = build_model(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.)
        health = TrainingHealth(model)
        initial = evaluate_fit(model, batches, device)
        record['constant_NLL'] = float(np.log(4.))
        record['history'].append({'step': 0, **initial})
        watch = ['input.0.weight', 'local.0.q.weight', 'module_queries', 'latent_read.q.weight',
                 'global_blocks.0.q.weight', 'head.0.weight', 'linear_readout.weight']
        parameters = dict(model.named_parameters())
        success_count = 0
        torch.cuda.reset_peak_memory_stats()
        try:
            for step in range(1, args.max_steps+1):
                model.train()
                gpu = device_batch(batches[(step-1) % len(batches)], device)
                optimizer.zero_grad(set_to_none=True)
                sampled = step in [1, 4, 16] or step % 32 == 0
                before = {name: parameters[name].detach().clone() for name in watch} if sampled else {}
                if sampled:
                    health.start(gpu)
                logits = model(gpu)
                loss = torch.nn.functional.cross_entropy(logits, gpu['label'])
                loss.backward()
                if sampled:
                    report = health.finish(logits, step)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                if sampled:
                    report['parameter_update_RMS'] = {name: float((parameters[name]-value).detach().square().mean().sqrt())
                                                      for name, value in before.items()}
                    record['health'].append(report)
                    if not report['healthy']:
                        raise RuntimeError('real 32-event encoder health failed')
                if step % 32 == 0:
                    evaluation = evaluate_fit(model, batches, device)
                    row = {'step': step, **evaluation}
                    record['history'].append(row)
                    print(json.dumps(row), flush=True)
                    success = evaluation['accuracy'] >= .9375 and evaluation['NLL'] < .25 and evaluation['finite']
                    success_count = success_count+1 if success else 0
                    write_json(path, record)
                    if success_count >= 2 and step >= 64:
                        break
            torch.cuda.synchronize()
            final = evaluate_fit(model, batches, device)
            gradients = [row['gradient_squared_norms'] for row in record['health']]
            checks = {'fit_accuracy_at_least_30_of_32': final['accuracy'] >= .9375,
                      'fit_NLL_below_025': final['NLL'] < .25,
                      'loss_below_quarter_constant': final['NLL'] < np.log(4.)/4,
                      'nonconstant_probabilities': final['probability_vectors'] >= 4 and max(final['probability_std']) > .05,
                      'all_health_samples_pass': all(row['healthy'] for row in record['health']),
                      'local_gradient_positive': all(row.get('input', 0.) + row.get('local', 0.) > 0 for row in gradients),
                      'global_gradient_positive': all(row.get('latent_read', 0.) > 0 for row in gradients),
                      'observed_parameter_updates': all(all(value > 0 for value in row['parameter_update_RMS'].values())
                                                        for row in record['health'])}
            checks = {key: bool(value) for key, value in checks.items()}
            record.update(status='complete' if all(checks.values()) else 'failed_learning_gate',
                          steps=step, training_event_exposures=step*8, final=final, checks=checks,
                          seconds=time.perf_counter()-started, peak_memory={**memory.peak,
                          'GPU_allocated_MiB': torch.cuda.max_memory_allocated()/2**20})
        finally:
            health.close()
    record['temporary_scratch_removed'] = not Path(scratch).exists()
    record['UTC_finished'] = datetime.now(timezone.utc).isoformat()
    write_json(path, record)
    if record['status'] != 'complete':
        raise RuntimeError('real 32-event learning gate failed; inspect internal evidence before any scientific inference')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--name', default='s4_empty_scratch_n100000_s42')
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--max-steps', type=int, default=512)
    parser.add_argument('--learning-rate', type=float, default=.0005)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
