"""Real Train-only fitting and branch-gradient/update gates for all R1 tasks."""

import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import torch

from analysis.flavour.inventory import write_json
from analysis.flavour.v2.fit import fit_event_positions
from analysis.flavour.v2.profile import device_batch
from analysis.flavour.v2.r1.config import recipe
from analysis.flavour.v2.r1.data import R1Dataset, collate
from analysis.flavour.v2.r1.models import build_model, branch_groups
from analysis.flavour.v2.r1.tasks import task_loss, probabilities, three_labels
from analysis.flavour.v2.r1.health import Activations


def measure(model, batches, task, bf16=False):
    model.eval()
    logits, labels = [], []
    with torch.inference_mode():
        for batch in batches:
            gpu = device_batch(batch, 'cuda:0')
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
                logits.append(model(gpu).float())
            labels.append(gpu['label'])
    value, truth = torch.cat(logits), torch.cat(labels)
    p = probabilities(value, task)
    target = three_labels(truth) if task == 'T3' else truth
    return {'NLL': float(task_loss(value, truth, task)), 'accuracy': float((p.argmax(-1) == target).float().mean()),
            'probability_std': p.std(0, unbiased=False).cpu().tolist(), 'distinct_vectors': len(torch.unique(p, dim=0)),
            'finite': bool(torch.isfinite(value).all())}


def gradients(groups):
    return {name: sum(float(p.grad.detach().float().square().sum()) for _, p in params if p.grad is not None)
            for name, params in groups.items()}


def run(args):
    torch.set_num_threads(1)
    pa.set_cpu_count(1)
    output = args.root/'fit'/args.attempt
    output.mkdir(parents=True, exist_ok=False)
    selected = pd.read_parquet(args.r0/'fit/fit32_a2_events.parquet')
    records = []
    for family in args.families:
        for task in args.tasks:
            torch.manual_seed(42)
            np.random.seed(42)
            config = recipe(args.r0, family, task, compile_local=args.compile_local)
            if family == 'B1':
                config['cnn_variant'] = args.cnn_variant
            config['precision'] = 'BF16_encoder_FP32_head' if args.bf16 else 'FP32'
            with tempfile.TemporaryDirectory(prefix='pinpoint-r1-fit-') as scratch:
                data = R1Dataset(args.r0/'micro_data', 'train', config, cache_dir=scratch)
                rows = selected if task != 'T3' else selected[selected.label.ne(2)]
                positions = fit_event_positions(rows, data.metadata)
                # Interleave labels so every fit batch contains each class.
                positions = positions.reshape(-1, 8).T.reshape(-1)
                items = [data[int(i)] for i in positions]
                batches = [collate(items[i:i+args.batch_size]) for i in range(0, len(items), args.batch_size)]
                data.native.close()
                started = time.perf_counter()
                model = build_model(config).cuda()
                groups = branch_groups(model)
                activations = Activations(model)
                watch = {name: params[0][1] for name, params in groups.items()}
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.)
                record = {'family': family, 'task': task, 'config': config, 'status': 'running',
                    'GPU': torch.cuda.get_device_name(), 'events': len(rows), 'seed': 42,
                    'keys': rows[['run', 'chunk', 'row', 'label']].to_dict('records'),
                    'parameters': sum(p.numel() for p in model.parameters()), 'history': [], 'health': [],
                    'scope': 'Train-only informative small-sample fit; no generalization estimate'}
                initial = measure(model, batches, task, args.bf16)
                record['history'].append({'step': 0, **initial})
                torch.cuda.reset_peak_memory_stats()
                success_count = 0
                for step in range(1, args.max_steps+1):
                    model.train()
                    gpu = device_batch(batches[(step-1) % len(batches)], 'cuda:0')
                    sampled = step in [1, 4, 16] or step % 32 == 0
                    before = {name: p.detach().clone() for name, p in watch.items()} if sampled else {}
                    if sampled:
                        activations.start()
                        gpu['tracker'].requires_grad_(True)
                        if 'nodes' in gpu:
                            gpu['nodes'].requires_grad_(True)
                        if 'projection_values' in gpu:
                            gpu['projection_values'].requires_grad_(True)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.bf16):
                        logits = model(gpu).float()
                        loss = task_loss(logits, gpu['label'], task)
                    loss.backward()
                    if sampled:
                        grad = gradients(groups)
                        sensitivity = {}
                        for name in ['nodes', 'tracker', 'projection_values']:
                            if name in gpu and gpu[name].grad is not None:
                                sensitivity[name] = float(gpu[name].grad.float().square().sum())
                        if 'nodes' in gpu and gpu['nodes'].grad is not None:
                            for d, name in enumerate(['pixel', 'bar', 'tracker_native']):
                                mask = gpu['nodes'][..., 12+d] > .5
                                sensitivity[name] = float(gpu['nodes'].grad[mask].float().square().sum())
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    if sampled:
                        updates = {name: float((p-before[name]).detach().float().square().mean().sqrt()) for name, p in watch.items()}
                        record['health'].append({'step': step, 'gradient_squared_norms': grad,
                            'parameter_update_RMS': updates, 'input_gradient_squared_norms': sensitivity,
                            'activations': activations.finish(),
                            'logit_std': logits.detach().float().std(0, unbiased=False).cpu().tolist()})
                    if step % 32 == 0:
                        final = measure(model, batches, task, args.bf16)
                        record['history'].append({'step': step, **final})
                        print(json.dumps({'family': family, 'task': task, 'step': step, **final}), flush=True)
                        passed = final['accuracy'] >= .9375 and final['NLL'] < .25 and final['finite']
                        success_count = success_count+1 if passed else 0
                        write_json(output/f'{family}_{task}.json', record)
                        if step >= 64 and success_count >= 2:
                            break
                final = measure(model, batches, task, args.bf16)
                active_groups = [name for name in groups if family in ['B4', 'B4-MAE'] or name != 'native']
                checks = {'fit_accuracy': final['accuracy'] >= .9375, 'fit_NLL': final['NLL'] < .25,
                    'nonconstant_probability': final['distinct_vectors'] >= 4 and max(final['probability_std']) > .05,
                    'branch_gradients': all(any(h['gradient_squared_norms'][name] > 0 for h in record['health']) for name in active_groups),
                    'branch_updates': all(any(h['parameter_update_RMS'][name] > 0 for h in record['health']) for name in active_groups)}
                checks['finite_branch_activations'] = all(value['finite'] for h in record['health'] for value in h['activations'].values())
                record.update(status='complete' if all(checks.values()) else 'failed_learning_gate',
                    final=final, steps=step, checks={k: bool(v) for k,v in checks.items()},
                    training_event_exposures=sum(len(batches[(i-1) % len(batches)]['label']) for i in range(1, step+1)),
                    seconds=time.perf_counter()-started, GPU_peak_MiB=torch.cuda.max_memory_allocated()/2**20)
                write_json(output/f'{family}_{task}.json', record)
                records.append(record)
                del model, optimizer
                torch.cuda.empty_cache()
                if not all(checks.values()):
                    raise RuntimeError(f'{family}/{task} failed real learning; repair before campaign')
    write_json(output/'summary.json', {'status': 'complete', 'cases': [{k:r[k] for k in
        ['family', 'task', 'status', 'parameters', 'steps', 'seconds', 'GPU_peak_MiB', 'final']} for r in records]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--r0', type=Path, required=True)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--families', nargs='+', default=['B1', 'B2', 'B3', 'B4'])
    parser.add_argument('--tasks', nargs='+', default=['T3', 'T4'])
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=.0005)
    parser.add_argument('--max-steps', type=int, default=256)
    parser.add_argument('--compile-local', action='store_true')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--cnn-variant', choices=['resnet', 'base'], default='resnet')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
