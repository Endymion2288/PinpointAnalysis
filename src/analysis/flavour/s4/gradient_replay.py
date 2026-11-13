"""Replay an initialization batch to isolate empty-response gradient amplification."""
import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from analysis.flavour.inventory import write_json
from analysis.flavour.s2.dataset import SpatialBatchSampler
from analysis.flavour.s3.dataset import collate_planes
from analysis.flavour.s3.health import TrainingHealth
from analysis.flavour.s3.models_repair import PlaneModuleClassifierV2
from analysis.flavour.s3.pilot_repair import optimizer_for
from analysis.flavour.s4.dataset import LabelDataset


def norms(model):
    result = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            group = name.split('.')[0]
            result[group] = result.get(group, 0.) + float(parameter.grad.detach().double().square().sum())
    return result


def replay(root, name):
    original = json.loads((root / 'experiments' / name / 'manifest.json').read_text())['config']
    config = {**original, 'compile_local': False, 'maximum_preclip_gradient_norm': 10000.}
    data = LabelDataset(root, 'train', config)
    ids = next(iter(SpatialBatchSampler(data, config['batch_size'], config['seed'],
        steps=config.get('steps_per_epoch'))))
    items = [data[int(i)] for i in ids]
    template = collate_planes(items)
    empty = torch.tensor([int((item['nodes'][:, 15] > 0).sum()) == 0 for item in items])
    records = []
    for safe in [False, True]:
        random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
        model = PlaneModuleClassifierV2({**config, 'empty_response_bypass': safe}).eval()
        variance = {}
        handles = []
        def record_variance(name):
            def hook(module, inputs):
                x = inputs[0].detach().float()
                variance[name] = {'minimum_feature_variance': float(x.var(-1, unbiased=False).min()),
                    'zero_variance_rows': int((x.var(-1, unbiased=False) == 0).sum()),
                    'eps': module.eps}
            return hook
        for key, module in model.named_modules():
            if isinstance(module, torch.nn.LayerNorm):
                handles.append(module.register_forward_pre_hook(record_variance(key)))
        paths = {}
        for label, selected in [('empty_events', empty), ('observed_events', ~empty),
                                ('complete_batch', torch.ones_like(empty))]:
            if not selected.any():
                continue
            model.zero_grad(set_to_none=True)
            health = TrainingHealth(model);health.start(template)
            logits = model(template)
            losses = torch.nn.functional.cross_entropy(logits, template['label'], reduction='none')
            # Preserve the full batch denominator when decomposing its loss.
            loss = losses[selected].sum() / len(losses)
            loss.backward()
            measured = health.finish(logits, 1);health.close()
            gradients = norms(model)
            before = {key: parameter.detach().clone() for key, parameter in model.named_parameters()}
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('clip_grad_norm', 1.),
                error_if_nonfinite=True))
            clipped = norms(model)
            update = {}
            if label == 'complete_batch':
                optimizer = optimizer_for(model, config)
                for group in optimizer.param_groups:
                    group['lr'] = config['learning_rate'] / config['warmup_steps']
                optimizer.step()
                for key, parameter in model.named_parameters():
                    group = key.split('.')[0]
                    update[group] = update.get(group, 0.) + float((parameter.detach() - before[key]).double().square().sum())
            paths[label] = {'events': int(selected.sum()), 'loss_contribution': float(loss.detach()),
                'preclip_gradient_norm': norm, 'clip_scale': min(1., config.get('clip_grad_norm', 1.) / max(norm, 1e-30)),
                'gradient_squared_norms': gradients, 'clipped_gradient_squared_norms': clipped,
                'first_AdamW_parameter_delta_squared_norms': update, 'health': measured,
                'layernorm_inputs': dict(variance)}
        for handle in handles:
            handle.remove()
        records.append({'empty_response_bypass': safe, 'paths': paths})
    report = {'name': name, 'config': original,
        'events': data.metadata.iloc[ids][['run', 'chunk', 'row', 'label']].to_dict('records'),
        'occupancy': [int((item['nodes'][:, 15] > 0).sum()) for item in items], 'variants': records}
    data.native.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--names', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args();torch.set_num_threads(1)
    report = {'precision': 'CPU FP32, original initialization and first sampler batch, eval mode, no compilation',
        'scope': 'train events only; empty and observed losses use the complete batch denominator; each variant starts from the same RNG seed',
        'models': [replay(args.root, name) for name in args.names], 'calibration_test_used': False}
    write_json(args.output, report)
    for model in report['models']:
        for variant in model['variants']:
            print(json.dumps({'name': model['name'], 'safe': variant['empty_response_bypass'],
                'paths': {key: {field: value[field] for field in ['preclip_gradient_norm', 'clip_scale']}
                    for key, value in variant['paths'].items()}}))


if __name__ == '__main__':
    main()
