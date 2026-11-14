"""Masked physics auxiliaries, normalized fusion and measured-learning readback."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import DataLoader, Subset

from analysis.flavour.inventory import write_json
from analysis.flavour.s2.dataset import SpatialBatchSampler
from analysis.flavour.s2.train import init_worker, metrics, save_checkpoint, save_parquet
from analysis.flavour.s3.health import TrainingHealth
from analysis.flavour.s4.low_label import CompletePassSampler
from analysis.flavour.s5.dataset import PhysicsDataset, collate_physics
from analysis.flavour.s5.model import build_model, losses


def device_batch(batch, device):
    return {k: device_batch(v, device) if isinstance(v, dict) else v.to(device, non_blocking=True) for k, v in batch.items()}


def predict(model, loader, device, bf16):
    model.eval()
    collected = {}
    with torch.no_grad():
        for cpu in loader:
            batch = device_batch(cpu, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
                output = model(batch['observation'], batch['objects'])
            values = {k: output[k].float().cpu().numpy() for k in ['probability', 'logits', 'vertex', 'kinematics', 'semantics', 'charm']}
            values['index'] = cpu['observation']['index'].numpy()
            for key, value in values.items():
                collected.setdefault(key, []).append(value)
    result = {k: np.concatenate(v) for k, v in collected.items()}
    order = np.argsort(result['index'])
    return {k: v[order] for k, v in result.items()}


def gradient_relations(values, output):
    reference = torch.autograd.grad(values['classification'], output['raw_features'], retain_graph=True, allow_unused=True)[0]
    result = {}
    if reference is None:
        return {'classification': 'disconnected'}
    for task, value in values.items():
        if task == 'classification':
            continue
        gradient = torch.autograd.grad(value, output['raw_features'], retain_graph=True, allow_unused=True)[0]
        if gradient is not None:
            cosine = torch.nn.functional.cosine_similarity(reference.flatten()[None], gradient.flatten()[None], dim=-1)
            result[task] = {'cosine': float(cosine), 'norm': float(gradient.norm()), 'main_norm': float(reference.norm())}
    return result


def run(args):
    config = json.loads(args.campaign.read_text())['experiments'][args.task]
    gate = json.loads((args.root/'pilot/gate.json').read_text())
    if not gate['gate_passed']:
        raise RuntimeError('S5 real-event fit gate required')
    if config.get('coordinate_revision', 1) > gate['config'].get('coordinate_revision', 1):
        raise RuntimeError('coordinate revision requires a matching real-event fit gate')
    destination = args.root/'experiments'/config['name']
    destination.mkdir(parents=True, exist_ok=True)
    if (destination/'manifest.json').exists():
        previous = json.loads((destination/'manifest.json').read_text())
        if previous.get('status') == 'complete':
            if previous['config'] != config:
                raise ValueError('completed name reused with different settings')
            print(config['name']+' already complete', flush=True)
            return
        if (destination/'last.pt').exists():
            raise ValueError('explicit recovery configuration required for interrupted S5 training')
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    device = torch.device(args.device)
    bf16 = device.type == 'cuda' and torch.cuda.is_bf16_supported(including_emulation=False)
    train = PhysicsDataset(args.root, 'train', config)
    validation = PhysicsDataset(args.root, 'validation', config)
    object_mean, object_scale = train.objects.mean(0), np.maximum(train.objects.std(0), .001)
    train.object_mean = validation.object_mean = object_mean
    train.object_scale = validation.object_scale = object_scale
    sampler = CompletePassSampler(train, config['batch_size'], seed, steps=config.get('steps_per_epoch'))
    validation_sampler = SpatialBatchSampler(validation, config['batch_size'], seed, shuffle=False)
    options = {'num_workers': config.get('workers', 1), 'collate_fn': collate_physics, 'pin_memory': device.type == 'cuda',
               'generator': torch.Generator().manual_seed(seed)}
    if options['num_workers']:
        options.update(multiprocessing_context='spawn', worker_init_fn=init_worker, prefetch_factor=1, persistent_workers=True)
    loader = DataLoader(train, batch_sampler=sampler, **options)
    val_loader = DataLoader(validation, batch_sampler=validation_sampler, **options)
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config.get('weight_decay', .01))
    total_steps = len(sampler)*config['epochs']
    warmup = config.get('warmup_steps', 100)
    minimum = config.get('minimum_learning_rate', 1e-5)/config['learning_rate']
    def lr_factor(step):
        if step < warmup:
            return (step+1)/max(warmup, 1)
        return minimum+(1-minimum)*.5*(1+math.cos(math.pi*min(1., (step-warmup)/max(total_steps-warmup, 1))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    health = TrainingHealth(model)
    records, history = [], []
    manifest = {'config': config, 'status': 'running', 'started': datetime.now(timezone.utc).isoformat(),
                'code_commit': os.environ.get('PINPOINT_CODE_COMMIT', 'interactive'), 'stage': 'S5',
                'train_events': len(train), 'validation_events': len(validation), 'calibration_test_used': False,
                'parameters': sum(p.numel() for p in model.parameters()), 'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None,
                'torch': torch.__version__, 'bf16': bf16, 'label_masks': {k: int(v.sum()) for k, v in train.targets.items() if k.endswith('_mask')},
                'object_standardizer': {'mean': object_mean.tolist(), 'scale': object_scale.tolist(), 'fit': 'selected train labels only'},
                'input': 'native hardware observations; optional measured objects; no truth variables',
                'selection': 'validation NLL' if not config.get('fixed_final') else 'fixed final epoch',
                'gradient_cosine_scope': 'shared raw head-input representation, not all encoder parameters'}
    write_json(destination/'manifest.json', manifest)
    initial_parameters = {k: v.detach().cpu().clone() for k, v in model.named_parameters() if k in ['input.0.weight', 'local.0.q.weight', 'head.0.weight']}
    started = time.monotonic()
    step, best = 0, float('inf')
    initial_loss = None
    best_prediction = None
    maximum_gradient = 0.
    for epoch in range(config['epochs']):
        sampler.epoch = epoch
        model.train()
        loss_sum, count = {}, 0
        for cpu in loader:
            batch = device_batch(cpu, device)
            optimizer.zero_grad(set_to_none=True)
            sample = step < config.get('health_initial_steps', 8) or step % config.get('health_every_steps', 100) == 0
            if sample:
                health.start(batch['observation'])
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
                output = model(batch['observation'], batch['objects'])
                total, values, masks = losses(model, output, batch['targets'], batch['observation']['label'])
            relations = gradient_relations(values, output) if sample else None
            total.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('clip_grad_norm', 1.), error_if_nonfinite=True)
            maximum_gradient = max(maximum_gradient, float(norm))
            if sample:
                record = health.finish(output['logits'], step)
                record.update(gradient_cosines=relations, auxiliary_valid_counts=masks,
                              preclip_gradient_norm_all_parameters=float(norm), losses={k: float(v.detach()) for k, v in values.items()})
                records.append(record)
                write_json(destination/'training_health.json', {'samples': records, 'all_healthy': all(r['healthy'] for r in records)})
                if not record['healthy']:
                    raise RuntimeError('S5 internal-health failure: '+str(record['failure_reasons']))
            optimizer.step()
            scheduler.step()
            if initial_loss is None:
                initial_loss = float(values['classification'].detach())
            for key, value in values.items():
                loss_sum[key] = loss_sum.get(key, 0.)+float(value.detach())*len(batch['objects'])
            count += len(batch['objects'])
            step += 1
            if step % config.get('progress_every_steps', 100) == 0:
                write_json(destination/'progress.json', {'epoch': epoch+1, 'step': step, 'seconds': time.monotonic()-started,
                           'classification_loss': float(values['classification'].detach()), 'maximum_preclip_gradient': maximum_gradient,
                           'GPU_peak_allocated_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
                           'GPU_reserved_bytes': torch.cuda.memory_reserved() if device.type == 'cuda' else None})
        prediction = predict(model, val_loader, device, bf16)
        measured = metrics(validation.metadata.label.to_numpy(), prediction['probability'])
        row = {'epoch': epoch+1, 'train_losses': {k: v/count for k, v in loss_sum.items()}, 'validation': measured,
               'seconds': time.monotonic()-started}
        history.append(row)
        score = measured['NLL'] if not config.get('fixed_final') else -float(epoch)
        if score < best:
            best = score
            best_prediction = prediction
            save_checkpoint(destination/'best.pt', {'model': model.state_dict(), 'config': config, 'epoch': epoch+1,
                            'object_mean': object_mean, 'object_scale': object_scale})
        save_checkpoint(destination/'last.pt', {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                        'config': config, 'epoch': epoch+1, 'step': step})
        write_json(destination/'history.json', history)
        print(json.dumps({'experiment': config['name'], **row}), flush=True)
    checkpoint = torch.load(destination/'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    indices = np.unique(np.linspace(0, len(validation)-1, min(128, len(validation))).astype(int))
    replay_loader = DataLoader(Subset(validation, indices), batch_size=min(config['batch_size'], 8), collate_fn=collate_physics)
    replay = predict(model, replay_loader, device, bf16)
    difference = float(np.max(np.abs(replay['probability']-best_prediction['probability'][indices])))
    deltas = {k: float((checkpoint['model'][k].cpu()-v).double().norm()) for k, v in initial_parameters.items()}
    if difference > .003 or not all(v > 0 for v in deltas.values()) or best_prediction['probability'].std(0).max() < .001:
        raise RuntimeError('selected checkpoint replay or actual encoder update failed')
    frame = validation.metadata.copy()
    for c in range(4):
        frame[f'p{c}'] = best_prediction['probability'][:, c]
    if config.get('six_way'):
        q = torch.from_numpy(best_prediction['logits']).softmax(-1).numpy()
        for c in range(6):
            frame[f'q{c}'] = q[:, c]
    for k in ['vertex', 'kinematics', 'semantics', 'charm']:
        values = best_prediction[k]
        for c in range(values.shape[1]):
            frame[f'pred_{k}_{c}'] = values[:, c]
    save_parquet(frame, destination/'validation_predictions.parquet')
    final_metrics = metrics(frame.label.to_numpy(), best_prediction['probability'])
    write_json(destination/'metrics.json', final_metrics)
    readback = {'events': len(indices), 'maximum_probability_difference': difference, 'parameter_delta_norms': deltas,
                'initial_first_batch_classification_loss': initial_loss, 'final_train_losses': history[-1]['train_losses'],
                'validation_probability_std': best_prediction['probability'].std(0).tolist(),
                'all_health_samples_pass': all(r['healthy'] for r in records), 'maximum_preclip_gradient': maximum_gradient,
                'checkpoint_replay_pass': True, 'encoder_update_pass': True}
    write_json(destination/'readback.json', readback)
    manifest.update(status='complete', finished=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started,
                    best_epoch=checkpoint['epoch'], optimizer_steps=step, all_internal_health_pass=all(r['healthy'] for r in records),
                    peak_GPU_memory_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else None)
    write_json(destination/'manifest.json', manifest)
    health.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--task', type=int, required=True)
    parser.add_argument('--device', default='cuda')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
