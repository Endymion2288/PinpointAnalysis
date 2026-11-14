"""R1 training with measured exposures, a fixed compute checkpoint and raw scores."""

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from torch.utils.data import DataLoader

from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.s2.dataset import SpatialBatchSampler
from analysis.flavour.s4.stage_inputs import stage_shards
from analysis.flavour.v2.profile import device_batch, init_worker, shutdown_loader, MemoryMonitor
from analysis.flavour.v2.r1.data import R1Dataset, collate
from analysis.flavour.v2.r1.fit import gradients
from analysis.flavour.v2.r1.metrics import quick_report
from analysis.flavour.v2.r1.models import build_model, branch_groups
from analysis.flavour.v2.r1.tasks import probabilities, task_loss
from analysis.flavour.v2.r1.health import Activations


def loader_for(data, config, shuffle, seed, steps=None):
    sampler = SpatialBatchSampler(data, config['batch_size'], seed, shuffle=shuffle, steps=steps)
    workers = config.get('workers', 2)
    loader = DataLoader(data, batch_sampler=sampler, collate_fn=collate, num_workers=workers,
        pin_memory=True, persistent_workers=workers>0, worker_init_fn=init_worker,
        **({'multiprocessing_context': 'spawn', 'prefetch_factor': 2} if workers else {}))
    return loader, sampler


def checkpoint(path, model, optimizer, config, counters):
    temporary = path.with_suffix('.tmp')
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'config': config, 'counters': counters}, temporary)
    os.replace(temporary, path)


def predict(model, loader, task, precision=True):
    started = time.perf_counter()
    model.eval()
    rows, values, logits = [], [], []
    with torch.inference_mode():
        for batch in loader:
            gpu = device_batch(batch, 'cuda:0')
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision):
                value = model(gpu).float()
            logits.append(value.cpu().numpy())
            values.append(probabilities(value, task).cpu().numpy())
            rows.extend(batch['index'].tolist())
    order = np.argsort(rows)
    return np.asarray(rows)[order], np.concatenate(values)[order], np.concatenate(logits)[order], time.perf_counter()-started


def save_predictions(path, dataset, indices, p, logits):
    frame = dataset.metadata.iloc[indices].reset_index(drop=True).copy()
    classes = [0, 1, 3] if p.shape[1] == 3 else [0, 1, 2, 3]
    for c, values in zip(classes, p.T):
        frame[f'p{c}'] = values
    for c, values in enumerate(logits.T):
        frame[f'logit{c}'] = values
    frame.to_parquet(path, index=False, compression=COMPRESSION)


def run(args):
    started = time.perf_counter()
    config = json.loads(args.config.read_text())
    if bool(config.get('compile_local')) != args.compile_local:
        raise ValueError('compile-local flag and frozen recipe must agree')
    root = args.root
    output = root/'experiments'/config['name']
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'config.json', config)
    torch.set_num_threads(config.get('main_threads', 1))
    pa.set_cpu_count(1)
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    # Reuse idle older cards without asking them to execute unsupported BF16
    # kernels. The submitted recipe and actual runtime mode are both recorded.
    submitted_config = config.copy()
    if torch.cuda.get_device_capability()[0] < 8:
        config.update(bf16=False, compile_local=False, precision='FP32')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    data_root = Path(config['data_root'])
    scratch_parent = os.environ.get('PINPOINT_V2_SCRATCH', os.environ.get('_CONDOR_SCRATCH_DIR', '/tmp'))
    record = {'status': 'running', 'config': config, 'GPU': torch.cuda.get_device_name(),
        'allocation_id': os.environ.get('PINPOINT_JOB_ID'),
        'submitted_config': submitted_config, 'capability': list(torch.cuda.get_device_capability()),
        'history': [], 'health': [], 'Cal_A_Cut_B_Confirm_payload_read': False}
    with tempfile.TemporaryDirectory(prefix='r1-train-', dir=scratch_parent) as scratch, MemoryMonitor() as memory:
        paths = sorted(p for p in (data_root/'shards').iterdir() if (p/'observations.parquet').exists())
        record['staging'] = stage_shards(paths, Path(scratch)/'native', workers=3)
        os.environ['PINPOINT_NATIVE_SHARD_ROOT'] = str(Path(scratch)/'native')
        train = R1Dataset(data_root, 'train', config, cache_dir=scratch)
        tune = R1Dataset(data_root, 'validation', config, cache_dir=scratch)
        if config.get('screen_only'):
            selected = pd.read_parquet(root/'data/tune_screen.parquet')
            lookup = pd.MultiIndex.from_frame(tune.metadata[['run', 'chunk', 'row']])
            keep = lookup.get_indexer(pd.MultiIndex.from_frame(selected[['run', 'chunk', 'row']]))
            if (keep < 0).any():
                raise ValueError('screen Tune event identities are absent')
            tune.indexes = tune.indexes[keep]
            tune.metadata = tune.metadata.iloc[keep].reset_index(drop=True)
            if tune.workload is not None:
                tune.workload = tune.workload[keep]
        # Same frozen 1k/class for every intermediate development checkpoint.
        if not config.get('profile') and not config.get('screen_only'):
            subset = pd.read_parquet(root/'data/tune_screen.parquet')
            lookup = pd.MultiIndex.from_frame(tune.metadata[['run', 'chunk', 'row']])
            keep = lookup.get_indexer(pd.MultiIndex.from_frame(subset[['run', 'chunk', 'row']]))
            if (keep < 0).any():
                raise ValueError('fixed Tune screen rows absent')
            small = R1Dataset(data_root, 'validation', config, cache_dir=scratch)
            small.indexes = small.indexes[keep]
            small.metadata = small.metadata.iloc[keep].reset_index(drop=True)
            if small.workload is not None:
                small.workload = small.workload[keep]
            validation, _ = loader_for(small, config, False, config['seed'])
        else:
            small, validation = None, None
        train_loader, sampler = loader_for(train, config, True, config['seed'], config.get('profile_steps'))
        full_loader, _ = loader_for(tune, config, False, config['seed']) if not config.get('profile') else (None, None)
        model = build_model(config).cuda()
        parameters = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        groups = branch_groups(model)
        activations = Activations(model)
        watch = {name: values[0][1] for name, values in groups.items()}
        counts = np.zeros(len(train), dtype='uint16')
        exposure = np.zeros(4, dtype='int64')
        counters = {'updates': 0, 'train_seconds': 0., 'training_exposures': 0,
                    'data_wait_seconds': 0., 'validation_seconds': 0., 'exposure_by_class': [0]*4}
        record.update(counters=counters, parameters=parameters)
        write_json(output/'manifest.json', record)
        total_steps = len(sampler)*config.get('passes', 4)
        budget = config.get('screen_seconds')
        wall_budget = config.get('total_compute_seconds')
        final_inference_estimate = config.get('final_inference_estimate_seconds', 0.)
        compute_budget = config.get('compute_checkpoint_seconds')
        compute_saved = False
        train_start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        epochs = config.get('passes', 4) if budget is None and wall_budget is None else 100
        stopped_for_budget = False
        best = -float('inf')
        for epoch in range(epochs):
            sampler.epoch = epoch
            model.train()
            window_loss, window_events = 0., 0
            iterator = iter(train_loader)
            tick = time.perf_counter()
            for batch in iterator:
                counters['data_wait_seconds'] += time.perf_counter()-tick
                step = counters['updates']
                if wall_budget is not None:
                    progress = min(1., (time.perf_counter()-started)/max(1., wall_budget-final_inference_estimate))
                elif budget is None:
                    progress = min(1., step/max(1, total_steps-1))
                else:
                    progress = min(1., counters['train_seconds']/budget)
                warmup = config.get('warmup_fraction', .05)
                factor = progress/warmup if progress < warmup else .05+.95*.5*(1+math.cos(math.pi*(progress-warmup)/(1-warmup)))
                lr = config['learning_rate']*max(.05, factor)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                gpu = device_batch(batch, 'cuda:0')
                sampled = step in [0, 15] or (step+1) % 1000 == 0
                before = {name: p.detach().clone() for name, p in watch.items()} if sampled else {}
                if sampled:
                    activations.start()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=config.get('bf16', True)):
                    logits = model(gpu).float()
                    loss = task_loss(logits, gpu['label'], config['task'])
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite loss')
                loss.backward()
                if sampled:
                    grad = gradients(groups)
                try:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_norm'], error_if_nonfinite=True)
                except RuntimeError:
                    write_json(output/'nonfinite_gradient.json', {'step': counters['updates'],
                        'GPU': record['GPU'], 'config': config,
                        'keys': train.metadata.iloc[batch['index'].numpy()][['run', 'chunk', 'row', 'label']].to_dict('records'),
                        'nonfinite_parameters': [name for name,p in model.named_parameters()
                            if p.grad is not None and not torch.isfinite(p.grad).all()]})
                    raise
                optimizer.step()
                torch.cuda.synchronize()
                counters['updates'] += 1
                counters['training_exposures'] += len(batch['label'])
                np.add.at(counts, batch['index'].numpy(), 1)
                exposure += np.bincount(batch['label'].numpy(), minlength=4)
                counters['exposure_by_class'] = exposure.tolist()
                window_loss += float(loss.detach())*len(batch['label'])
                window_events += len(batch['label'])
                counters['train_seconds'] = time.perf_counter()-train_start-counters['validation_seconds']
                if sampled:
                    record['health'].append({'step': counters['updates'], 'gradient_squared_norms': grad,
                        'activations': activations.finish(),
                        'parameter_update_RMS': {n: float((p-before[n]).detach().float().square().mean().sqrt()) for n,p in watch.items()},
                        'probability_std': probabilities(logits, config['task']).detach().std(0, unbiased=False).cpu().tolist()})
                if counters['updates'] % 250 == 0:
                    print(json.dumps({'epoch': epoch+1, **counters, 'train_NLL_window': window_loss/window_events, 'lr': lr}), flush=True)
                if compute_budget is not None and not compute_saved and counters['train_seconds'] >= compute_budget:
                    checkpoint(output/'compute.pt', model, optimizer, config, counters)
                    write_json(output/'compute_counters.json', dict(counters))
                    compute_saved = True
                if ((budget is not None and counters['train_seconds'] >= budget)
                        or (wall_budget is not None and time.perf_counter()-started+final_inference_estimate >= wall_budget)):
                    stopped_for_budget = True
                    break
                tick = time.perf_counter()
            row = {'pass': epoch+1, **counters, 'train_NLL': window_loss/window_events,
                   'unique_events': int((counts>0).sum()), 'lr': lr}
            if epoch+1 == config.get('exposure_checkpoint_passes'):
                checkpoint(output/'matched_exposure.pt', model, optimizer, config, counters)
                write_json(output/'matched_exposure_counters.json', dict(counters))
                np.save(output/'matched_exposure_counts.npy', counts)
            if not config.get('profile') and not config.get('screen_only') and not (wall_budget is not None and stopped_for_budget):
                indices, p, _, seconds = predict(model, validation, config['task'], config.get('bf16', True))
                counters['validation_seconds'] += seconds
                report = quick_report(small.metadata.label.to_numpy()[indices], p)
                row['Tune_screen'] = report
                row['validation_seconds'] = counters['validation_seconds']
                if report['T3']['BA'] > best:
                    best = report['T3']['BA']
                    checkpoint(output/'best.pt', model, optimizer, config, counters)
                model.train()
            record['history'].append(row)
            record.update(counters=counters, parameters=parameters)
            write_json(output/'manifest.json', record)
            print(json.dumps(row), flush=True)
            if stopped_for_budget:
                break
        checkpoint(output/'exposure.pt', model, optimizer, config, counters)
        if not config.get('profile'):
            indices, p, logits, seconds = predict(model, full_loader, config['task'], config.get('bf16', True))
            record['full_Tune_inference_seconds'] = seconds
            save_predictions(output/'predictions.parquet', tune, indices, p, logits)
            record['metrics'] = quick_report(tune.metadata.label.to_numpy()[indices], p)
            record['probability_diagnostics'] = {'mean': p.mean(0).tolist(), 'std': p.std(0).tolist(),
                'minimum': p.min(0).tolist(), 'maximum': p.max(0).tolist(),
                'distinct_vectors': len(np.unique(p, axis=0))}
            counters['validation_seconds'] += seconds
            if compute_saved:
                state = torch.load(output/'compute.pt', map_location='cpu', weights_only=False)
                model.load_state_dict(state['model'])
                indices, p, logits, seconds = predict(model, full_loader, config['task'], config.get('bf16', True))
                save_predictions(output/'compute_predictions.parquet', tune, indices, p, logits)
                record['compute_metrics'] = quick_report(tune.metadata.label.to_numpy()[indices], p)
                counters['validation_seconds'] += seconds
            if (output/'matched_exposure.pt').exists():
                state = torch.load(output/'matched_exposure.pt', map_location='cpu', weights_only=False)
                model.load_state_dict(state['model'])
                indices, p, logits, seconds = predict(model, full_loader, config['task'], config.get('bf16', True))
                save_predictions(output/'matched_exposure_predictions.parquet', tune, indices, p, logits)
                record['matched_exposure_metrics'] = quick_report(tune.metadata.label.to_numpy()[indices], p)
                counters['validation_seconds'] += seconds
        coverage = train.metadata[['run', 'chunk', 'row', 'label', 'family_hi', 'family_lo']].copy()
        coverage['exposures'] = counts
        coverage.to_parquet(output/'train_exposures.parquet', index=False, compression=COMPRESSION)
        for loader in [train_loader, full_loader, None if config.get('profile') else validation]:
            if loader is not None:
                shutdown_loader(loader)
        for data in [train, tune, small]:
            if data is not None:
                data.native.close()
        record.update(status='complete', counters=counters, unique_train_events=int((counts>0).sum()),
            input_train_events=len(train), parameters=parameters, wall_seconds=time.perf_counter()-started,
            GPU_peak_MiB=torch.cuda.max_memory_allocated()/2**20, memory=memory.peak,
            events_per_train_second=counters['training_exposures']/counters['train_seconds'])
        if wall_budget is not None:
            record['compute_alignment'] = {'target_wall_seconds': wall_budget,
                'actual_wall_seconds': record['wall_seconds'],
                'relative_difference': record['wall_seconds']/wall_budget-1.,
                'within_five_percent': abs(record['wall_seconds']/wall_budget-1.) <= .05,
                'reserved_final_inference_seconds': final_inference_estimate,
                'scope': 'same-allocation operation wall time, including loader, training and validation'}
        del os.environ['PINPOINT_NATIVE_SHARD_ROOT']
    record['temporary_scratch_removed'] = not Path(scratch).exists()
    write_json(output/'manifest.json', record)
    print(json.dumps({'name': config['name'], 'status': 'complete', 'counters': counters}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--compile-local', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
