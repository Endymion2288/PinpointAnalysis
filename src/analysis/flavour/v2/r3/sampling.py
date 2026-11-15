"""Global reshuffling with the exact original per-update batch sizes."""

import numpy as np

from analysis.flavour.s2.dataset import SpatialBatchSampler


class GlobalBatchSampler:
    def __init__(self, dataset, batch_size, seed):
        self.spatial = SpatialBatchSampler(dataset, batch_size, seed, shuffle=True)
        self.seed = dataset.config.get('order_seed', 20261119+(seed-42)*1009)

    @property
    def epoch(self):
        return self.spatial.epoch

    @epoch.setter
    def epoch(self, value):
        self.spatial.epoch = value

    def __len__(self):
        return len(self.spatial)

    def __iter__(self):
        original = list(self.spatial)
        sizes = np.asarray([len(batch) for batch in original])
        events = np.random.default_rng(self.seed+self.epoch).permutation(np.concatenate(original))
        yield from (part.tolist() for part in np.split(events, np.cumsum(sizes)[:-1]))

    def _windows(self):
        """Composition diagnostics describe actual 512-row global windows."""
        order = np.concatenate(list(self))
        yield from (order[start:start+512] for start in range(0, len(order), 512))
