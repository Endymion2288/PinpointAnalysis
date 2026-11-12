"""Read-only FP32 diagnostics for a published S3 training checkpoint."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch

from analysis.flavour.inventory import write_json
from analysis.flavour.s3.dataset import PlaneDataset, collate_planes
from analysis.flavour.s3.models import PlaneModuleClassifier


def tensor_summary(value):
    value = value.detach().float()
    return {'shape': list(value.shape), 'finite': bool(torch.isfinite(value).all()),
            'min': float(value.min()), 'max': float(value.max()),
            'mean': float(value.mean()), 'RMS': float(value.square().mean().sqrt()),
            'zero_fraction': float((value == 0).float().mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--checkpoint', choices=['last', 'best'], default='last')
    parser.add_argument('--examples-per-class', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.examples_per_class <= 4:
        parser.error('use 1..4 examples per class for this small CPU diagnostic')
    torch.set_num_threads(1)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    checkpoint_path = args.root / 'experiments' / args.experiment / (args.checkpoint + '.pt')
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = state['config']
    dataset = PlaneDataset(args.root, 'validation', config)
    ids = np.concatenate([np.flatnonzero(dataset.metadata.label.to_numpy() == c)[:args.examples_per_class]
                          for c in range(4)])
    batch = collate_planes([dataset[int(i)] for i in ids])
    target = batch.pop('label')
    batch.pop('index')
    model = PlaneModuleClassifier(config)
    model.load_state_dict(state['model'])
    model.eval()
    activations = {}

    def hook(name):
        def record(module, inputs, output):
            activations.setdefault(name, []).append(tensor_summary(output))
        return record

    names = ['local_pool', 'module_read', *[f'context.{i}' for i in range(len(model.context))],
             *(['latent_read'] if model.fusion == 'latent' else []),
             *[f'global_blocks.{i}' for i in range(len(model.global_blocks))],
             'head.0', 'head.1', 'head.3']
    handles = [model.get_submodule(name).register_forward_hook(hook(name)) for name in names]
    logits = model(batch)
    loss = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()
    for handle in handles:
        handle.remove()
    gradients = {}
    for name, parameter in model.named_parameters():
        group = name.split('.')[0]
        record = gradients.setdefault(group, {'parameters': 0, 'with_gradient': 0,
                                             'gradient_squared_norm': 0., 'nonzero_gradient_elements': 0,
                                             'finite': True})
        record['parameters'] += parameter.numel()
        if parameter.grad is not None:
            grad = parameter.grad.detach().float()
            record['with_gradient'] += grad.numel()
            record['gradient_squared_norm'] += float(grad.square().sum())
            record['nonzero_gradient_elements'] += int(torch.count_nonzero(grad))
            record['finite'] &= bool(torch.isfinite(grad).all())
    probability = logits.detach().softmax(-1).numpy().astype('float64')
    active_nodes = torch.zeros(len(ids))
    active_nodes.index_add_(0, batch['event_index'], batch['nodes'][..., 15].sum(1))
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'experiment': args.experiment,
              'checkpoint_path': str(checkpoint_path), 'checkpoint_epoch': state['epoch'],
              'optimizer_steps': state.get('optimizer_steps'), 'config': config,
              'execution': 'CPU FP32 eval mode with one backward pass; no optimizer step or checkpoint write',
              'scope': 'small diagnostic of optimization and representation; not replacement for frozen H100 BF16 evaluation',
              'events': dataset.metadata.iloc[ids][['run', 'chunk', 'row', 'label']].to_dict('records'),
              'input_active_nodes_by_event': active_nodes.tolist(),
              'activations': activations, 'gradients': gradients, 'loss': float(loss.detach()),
              'probabilities': probability.tolist(),
              'unique_score_vectors': int(len(np.unique(probability, axis=0))),
              'score_std': probability.std(0).tolist(),
              'all_parameters_finite': all(bool(torch.isfinite(p).all()) for p in model.parameters())}
    dataset.native.close()
    write_json(args.output, report)
    print(json.dumps({key: report[key] for key in ['checkpoint_epoch', 'optimizer_steps', 'unique_score_vectors', 'score_std']}))


if __name__ == '__main__':
    main()
