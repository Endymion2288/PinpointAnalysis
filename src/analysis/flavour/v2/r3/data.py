"""Allocation-local transformed inputs and exact-cache evidence."""

import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from analysis.flavour.v2.profile import init_worker
from analysis.flavour.v2.r1.data import R1Dataset, collate


def keep_events(events):
    return events


class R3Dataset(R1Dataset):
    def __init__(self, root, split, config, cache_dir=None):
        super().__init__(root, split, config, cache_dir)
        self.items = None
        if split == 'train' and config.get('cache_training_inputs', False):
            self.cache_inputs()

    def __getitem__(self, index):
        if self.items is not None:
            return self.items[index]
        return super().__getitem__(index)

    def cache_inputs(self):
        started = time.perf_counter()
        generator = torch.Generator().manual_seed(self.config['seed']+7819)
        loader = DataLoader(self, batch_size=128, shuffle=False, collate_fn=keep_events,
            num_workers=2, worker_init_fn=init_worker, generator=generator,
            multiprocessing_context='spawn', prefetch_factor=2)
        items = [item for group in loader for item in group]
        if [item['index'] for item in items] != list(range(len(self))):
            raise ValueError('transformed cache changed frozen event coverage')
        for index in np.linspace(0, len(self)-1, min(32, len(self)), dtype=int):
            a, b = collate([self[int(index)]]), collate([items[int(index)]])
            if any(not torch.equal(a[key], b[key]) for key in a if torch.is_tensor(a[key])):
                raise ValueError('transformed cache changed actual model inputs')
        self.cache_audit = {'events':len(items),
            'array_bytes':sum(v.nbytes for item in items for v in item.values() if isinstance(v, np.ndarray)),
            'seconds':time.perf_counter()-started, 'scattered_exact_input_checks':min(32, len(self)),
            'storage':'allocation-local transformed arrays; no persistent EOS tensor cache'}
        self.items = items
        self.native.close()
