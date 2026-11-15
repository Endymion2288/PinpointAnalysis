"""R3 adapters around the accepted recoverable, fixed-exposure runner."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from analysis.flavour.s2.dataset import SpatialBatchSampler
from analysis.flavour.v2.profile import init_worker
from analysis.flavour.v2.r2.train import OffsetSampler, run
from analysis.flavour.v2.r3.data import R3Dataset, collate
from analysis.flavour.v2.r3.sampling import GlobalBatchSampler
from analysis.flavour.v2.r3.models import build_model, branch_groups


def make_loader(dataset, config, shuffle):
    if shuffle and config.get('event_mixing') == 'global':
        sampler = GlobalBatchSampler(dataset, config['batch_size'], config['seed'])
    else:
        sampler = SpatialBatchSampler(dataset, config['batch_size'], config['seed'], shuffle=shuffle)
    offset = OffsetSampler(sampler)
    workers = 0 if dataset.items is not None else config.get('workers', 2)
    generator = torch.Generator().manual_seed(config['seed']+7819)
    loader = DataLoader(dataset, batch_sampler=offset, collate_fn=collate,
        num_workers=workers, pin_memory=True, persistent_workers=workers>0,
        worker_init_fn=init_worker, generator=generator,
        **({'multiprocessing_context':'spawn', 'prefetch_factor':2} if workers else {}))
    return loader, sampler, offset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-updates', type=int)
    args = parser.parse_args()
    run(args, dataset_factory=R3Dataset, loader_factory=make_loader,
        model_factory=build_model, group_factory=branch_groups)


if __name__ == '__main__':
    main()
