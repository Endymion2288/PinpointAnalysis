"""Diagnose the first real profile batch without accessing Tune scores."""

import argparse
import json
from pathlib import Path
import tempfile

import torch

from analysis.flavour.inventory import write_json
from analysis.flavour.s2.dataset import SpatialBatchSampler
from analysis.flavour.v2.profile import device_batch
from analysis.flavour.v2.r1.config import recipe
from analysis.flavour.v2.r1.data import R1Dataset, collate
from analysis.flavour.v2.r1.models import build_model
from analysis.flavour.v2.r1.tasks import task_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--r0', type=Path, required=True)
    parser.add_argument('--family', default='B2')
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--compile-local', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    records = []
    for bf16 in [False, True]:
        for compiled in ([False, True] if args.compile_local else [False]):
            torch.manual_seed(42)
            config = recipe(args.r0, args.family, 'T4', compile_local=compiled)
            config['compile_local'] = compiled  # Explicitly reproduce the unsafe diagnostic mode.
            with tempfile.TemporaryDirectory(prefix='r1-diagnose-') as scratch:
                data = R1Dataset(args.r0/'profile_data', 'train', config, cache_dir=scratch)
                indices = next(iter(SpatialBatchSampler(data, 8, 42)))
                batch = collate([data[int(i)] for i in indices])
                keys = data.metadata.iloc[indices][['run', 'chunk', 'row', 'label']].to_dict('records')
                data.native.close()
                model = build_model(config).cuda()
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.)
                row = {'bf16': bf16, 'compiled': compiled, 'GPU': torch.cuda.get_device_name(),
                    'keys': keys, 'steps': [], 'shapes': {k: list(v.shape) for k,v in batch.items()}}
                for step in range(2):
                    gpu = device_batch(batch, 'cuda:0')
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
                        logits = model(gpu).float()
                        loss = task_loss(logits, gpu['label'], 'T4')
                    loss.backward()
                    bad = [name for name,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    row['steps'].append({'step': step+1, 'finite_loss': bool(torch.isfinite(loss)),
                        'loss': float(loss.detach()), 'grad_norm': float(grad_norm), 'nonfinite_gradient_parameters': bad,
                        'maximum_absolute_gradient': max(float(p.grad.abs().max()) for p in model.parameters() if p.grad is not None),
                        'nonfinite_parameters_after_step': [name for name,p in model.named_parameters() if not torch.isfinite(p).all()]})
                records.append(row)
                print(json.dumps(row), flush=True)
                del model, optimizer
                torch.cuda.empty_cache()
    write_json(args.root/'audit'/f'diagnose_{args.family}_{args.attempt}.json', {'cases': records})


if __name__ == '__main__':
    main()
