"""Ragged shard dataset; targets and provenance never enter observations."""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


class NativeShard(Dataset):
    """Small-shard reader; use one cached shard per worker for production."""

    def __init__(self, path):
        self.path = Path(path)
        self.observations = pq.read_table(self.path / 'observations.parquet')
        self.targets = pq.read_table(self.path / 'targets.parquet')
        if len(self.observations) != len(self.targets):
            raise ValueError('observations/targets length mismatch')
        # Convert Arrow's contiguous list buffers once per shard. Returning
        # NumPy views avoids rebuilding Python objects for every channel/event.
        self.buffers, self.offsets = {}, {}
        widths = {'xyz_mm': 3, 'bounds_mm': 5, 'xy_mm': 2, 'sigma_xy_mm': 2}
        for name in ['pixel', 'bar', 'tracker_hypothesis']:
            struct = self.observations[name].combine_chunks()
            self.buffers[name] = {}
            for key in struct.type:
                field = struct.field(key.name)
                offsets = field.offsets.to_numpy(zero_copy_only=False)
                if name in self.offsets and not np.array_equal(self.offsets[name], offsets):
                    raise ValueError('ragged attributes have inconsistent channel counts')
                self.offsets[name] = offsets
                flat = field.flatten()
                width = widths.get(key.name)
                dtype = np.float32 if width or key.name == 'edep_proxy_MeV' else np.int64
                if width and len(flat):
                    values = flat.flatten().to_numpy(zero_copy_only=False).astype(dtype, copy=False).reshape(-1, width)
                else:
                    values = flat.to_numpy(zero_copy_only=False).astype(dtype, copy=False)
                    if width:
                        values = values.reshape(-1, width)
                self.buffers[name][key.name] = values
        self.states = {key: np.asarray(self.observations[key].to_pylist())
                       for key in self.observations.column_names if key not in self.buffers}

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, index):
        observation = {name: {key: values[self.offsets[name][index]:self.offsets[name][index+1]]
                              for key, values in fields.items()} for name, fields in self.buffers.items()}
        observation.update({key: values[index] for key, values in self.states.items()})
        return {'observations': observation,
                'targets': self.targets.slice(index, 1).to_pylist()[0]}


def collate_native(events):
    result = {}
    for name, field, width in [('pixel', 'xyz_mm', 3), ('bar', 'bounds_mm', 5),
                               ('tracker_hypothesis', 'xy_mm', 2)]:
        arrays = [np.asarray(e['observations'][name][field], dtype='float32').reshape(-1, width) for e in events]
        offsets = np.r_[0, np.cumsum([len(a) for a in arrays])]
        result[name] = {'values': torch.from_numpy(np.concatenate(arrays, axis=0)),
                        'offsets': torch.from_numpy(offsets)}
    result['has_response'] = torch.from_numpy(np.asarray([e['observations']['has_response'] for e in events], dtype=bool))
    for key in ['outside_acceptance', 'fit_attempted', 'momentum_censored']:
        if key in events[0]['observations']:
            result[key] = torch.from_numpy(np.asarray([e['observations'][key] for e in events]))
    # Preserve every observation attribute as ragged tensors, including IDs,
    # intervals, PE and uncertainties; no truncation or hidden support loss.
    for name in ['pixel', 'bar', 'tracker_hypothesis']:
        for key in events[0]['observations'][name]:
            if key == {'pixel': 'xyz_mm', 'bar': 'bounds_mm', 'tracker_hypothesis': 'xy_mm'}[name]:
                continue
            dtype = np.float32 if key in ['edep_proxy_MeV', 'sigma_xy_mm'] else np.int64
            arrays = [np.asarray(e['observations'][name][key], dtype=dtype) for e in events]
            if key == 'sigma_xy_mm':
                arrays = [a.reshape(-1, 2) for a in arrays]
            result[name][key] = torch.as_tensor(np.concatenate(arrays, axis=0))
    return {'observations': result,
            'targets': {'label': torch.as_tensor([e['targets']['label'] for e in events], dtype=torch.long)}}
