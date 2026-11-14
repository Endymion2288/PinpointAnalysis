"""Keep observations, measurement-derived objects and supervised targets separate."""
import numpy as np
import pandas as pd
import torch

from analysis.flavour.s3.dataset import collate_planes
from analysis.flavour.s4.dataset import LabelDataset
from analysis.flavour.s4.dataset import restrict_dataset


TRUTH_COLUMNS = ['run', 'chunk', 'row', 'six_label', 'vertex_mm', 'vertex_supervision_valid',
                 'bar_semantic_fractions', 'bar_semantic_valid', 'charm_presence', 'charm_label_valid',
                 'visible_p3_GeV', 'visible_E_GeV', 'visible_target_valid', 'tau_direct_prongs', 'tau_prong_valid']


class PhysicsDataset(LabelDataset):
    def __init__(self, root, split, config, cache_dir=None):
        super().__init__(root, split, config, cache_dir)
        if config.get('s5_smoke_events'):
            restrict_dataset(self, np.arange(min(len(self), config['s5_smoke_events'])))
        key = ['run', 'chunk', 'row']
        truth = pd.read_parquet(self.root/'truth.parquet', columns=TRUTH_COLUMNS)
        frame = self.metadata[key].merge(truth, on=key, how='left', validate='one_to_one', sort=False)
        if frame.six_label.isna().any():
            raise ValueError('selected event lacks S5 truth sidecar')
        self.targets = {'six': frame.six_label.to_numpy(dtype='int64'),
                        'vertex': np.stack(frame.vertex_mm)/[210, 210, 3000],
                        'vertex_mask': frame.vertex_supervision_valid.to_numpy(dtype=bool),
                        'semantics': np.stack(frame.bar_semantic_fractions),
                        'semantics_mask': frame.bar_semantic_valid.to_numpy(dtype=bool),
                        'charm': frame.charm_presence.to_numpy(dtype='float32')[:, None],
                        'charm_mask': frame.charm_label_valid.to_numpy(dtype=bool)}
        momentum = np.stack(frame.visible_p3_GeV)
        energy = frame.visible_E_GeV.to_numpy()
        self.targets['kinematics'] = np.column_stack([np.arcsinh(momentum[:, 0]), np.arcsinh(momentum[:, 1]),
                                                     np.log1p(np.maximum(momentum[:, 2], 0)), np.log1p(np.maximum(energy, 0))])/7.
        self.targets['kinematics_mask'] = frame.visible_target_valid.to_numpy(dtype=bool) & (momentum[:, 2] >= 0)
        self.objects = np.zeros((len(self), 31), dtype='float32')
        if config.get('use_objects'):
            objects = pd.read_parquet(self.root/'objects.parquet', columns=key+['object_features'])
            joined = self.metadata[key].merge(objects, on=key, how='left', validate='one_to_one', sort=False)
            if joined.object_features.isna().any():
                raise ValueError('selected event lacks observation-only objects')
            self.objects = np.stack(joined.object_features).astype('float32')
        self.object_mean = np.zeros(self.objects.shape[1], dtype='float32')
        self.object_scale = np.ones(self.objects.shape[1], dtype='float32')

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['supervision'] = {k: v[index] for k, v in self.targets.items()}
        item['objects'] = (self.objects[index]-self.object_mean)/self.object_scale
        return item


def collate_physics(items):
    batch = {'observation': collate_planes(items)}
    batch['objects'] = torch.from_numpy(np.stack([i['objects'] for i in items])).float()
    batch['targets'] = {k: torch.as_tensor(np.stack([i['supervision'][k] for i in items])) for k in items[0]['supervision']}
    return batch
