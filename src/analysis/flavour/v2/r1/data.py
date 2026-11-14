"""Shared observation inputs, physical 1 mm projections and tracker sets."""

import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from analysis.flavour.features import FEATURE_NAMES, event_features
from analysis.flavour.s2.dataset import SpatialDataset, collate_spatial
from analysis.flavour.s2.representation import token_inputs
from analysis.flavour.s3.dataset import plane_tokens, collate_planes
from analysis.flavour.v2.r1.tasks import NON_TAU


TRACKER_FEATURE_NAMES = (['tracker_log_channels', 'tracker_stations', 'tracker_present']
    + [f'tracker_{stat}_{axis}' for stat in ['mean', 'std'] for axis in ['x', 'y']]
    + [f'tracker_station_{s}_log_channels' for s in range(4)])
SUMMARY_NAMES = FEATURE_NAMES+TRACKER_FEATURE_NAMES


def tracker_features(tracker):
    xy = np.asarray(tracker['xy_mm'], dtype='float32').reshape(-1, 2)
    station = np.asarray(tracker['station'], dtype='int64')
    sigma = np.asarray(tracker['sigma_xy_mm'], dtype='float32').reshape(-1, 2)
    features = np.column_stack([station/4., xy/420., sigma/np.array([.58, .017])]).astype('float32')
    shape = np.r_[xy.mean(0), xy.std(0)] if len(xy) else np.zeros(4)
    summary = np.r_[np.log1p(len(xy)), len(np.unique(station)), float(len(xy)>0),
                    shape, [np.log1p(np.count_nonzero(station == s)) for s in range(4)]]
    return features, summary.astype('float32')


def image_geometry(geometry):
    """Union of active pixel centers and complete bar intervals, in mm."""
    x = np.asarray(geometry['scint_bar_Xcenter'])
    y = np.asarray(geometry['scint_bar_Ycenter'])
    left = min(min(geometry['pixel_Xpos']), x.min()-geometry['scint_bar_width']/2)
    right = max(max(geometry['pixel_Xpos']), x.max()+geometry['scint_bar_width']/2)
    bottom = min(min(geometry['pixel_Ypos']), y.min()-geometry['scint_bar_height']/2)
    top = max(max(geometry['pixel_Ypos']), y.max()+geometry['scint_bar_height']/2)
    z = np.r_[geometry['pixel_Zpos'], geometry['scint_Zpos']]
    origin = np.floor([left, bottom, z.min()])
    height = int(np.floor(z.max()-origin[2]))+1
    width = int(max(np.floor(right-origin[0]), np.floor(top-origin[1])))+1
    return {'origin_mm': origin.tolist(), 'height': height, 'width': width,
            'pitch_mm': 1., 'channels_per_view': ['pixel_count', 'bar_count', 'bar_edep_MeV', 'bar_PE'],
            'bar_projection': 'vertical bars in XZ; horizontal bars in YZ; unmeasured coordinate never localized'}


def projection(observation, geometry, image=None):
    image = image or image_geometry(geometry)
    x0, y0, z0 = image['origin_mm']
    h, w = image['height'], image['width']
    pixels, bars = observation['pixel'], observation['bar']
    xyz = np.asarray(pixels['xyz_mm']).reshape(-1, 3)
    indices, values = [], []
    for view, axis, offset in [(0, 0, x0), (1, 1, y0)]:
        zz = np.floor(xyz[:, 2]-z0).astype('int64')
        xx = np.floor(xyz[:, axis]-offset).astype('int64')
        indices.extend(((view*4*h+zz)*w+xx).tolist())
        values.extend(np.ones(len(xyz)).tolist())
        bounds = np.asarray(bars['bounds_mm']).reshape(-1, 5)
        horizontal = np.asarray(bars['horizontal'], dtype=bool)
        take = horizontal if view else ~horizontal
        measured = (bounds[take, 2*axis]+bounds[take, 2*axis+1])/2
        bz = np.floor(bounds[take, 4]-z0).astype('int64')
        bx = np.floor(measured-offset).astype('int64')
        for channel, weight in [(1, np.ones(take.sum())),
                (2, np.asarray(bars['edep_proxy_MeV'])[take]), (3, np.asarray(bars['pe_proxy'])[take])]:
            indices.extend((((view*4+channel)*h+bz)*w+bx).tolist())
            values.extend(weight.tolist())
    return np.asarray(indices, dtype='int64'), np.asarray(values, dtype='float32')


class R1Dataset(SpatialDataset):
    def __init__(self, root, split, config, cache_dir=None):
        super().__init__(root, split, config, cache_dir)
        catalog_path = self.root/'catalog.parquet'
        diagnostics = ['pixel_channels', 'bar_channels', 'tracker_channels', 'all_empty']
        missing = [name for name in diagnostics if name not in self.metadata]
        if missing and catalog_path.exists():
            missing = [name for name in missing if name in pq.read_schema(catalog_path).names]
            table = pd.read_parquet(catalog_path, columns=['run', 'chunk', 'row']+missing)
            self.metadata = self.metadata.merge(table, on=['run', 'chunk', 'row'], how='left',
                                               sort=False, validate='one_to_one')
        if 'all_empty' not in self.metadata and all(name in self.metadata for name in diagnostics[:3]):
            self.metadata['all_empty'] = self.metadata[diagnostics[:3]].sum(axis=1).eq(0)
        if split == 'train' and config.get('task') == 'T3':
            keep = self.metadata.label.isin(NON_TAU).to_numpy()
            self.indexes = self.indexes[keep]
            self.metadata = self.metadata.loc[keep].reset_index(drop=True)
            if self.workload is not None:
                self.workload = self.workload[keep]
        self.condition = None
        self.ideals = None

    def __getitem__(self, index):
        start = time.perf_counter()
        observation = self.native[int(self.indexes[index])]['observations']
        if self.condition is not None:
            meta = self.metadata.iloc[index].to_dict()
            ideal = None if self.ideals is None else self.ideals[index]
            observation = self.condition(observation, meta, ideal)
        if self.config.get('drop_modality'):
            observation = dict(observation)
            key = {'tracker': 'tracker_hypothesis'}.get(self.config['drop_modality'], self.config['drop_modality'])
            observation[key] = {name: values[:0] for name, values in observation[key].items()}
        tracker, summary = tracker_features(observation['tracker_hypothesis'])
        item = {'index': index, 'label': int(self.metadata.iloc[index].label),
                'tracker': tracker, 'presence': np.array([len(observation[n][k])>0 for n, k in
                    [('pixel', 'channel_id'), ('bar', 'channel_id'), ('tracker_hypothesis', 'station')]], dtype='float32'),
                'family': self.config['family']}
        if self.config['family'] == 'B1':
            item['projection_indices'], item['projection_values'] = projection(observation, self.geometry, self.config['image'])
        elif self.config['family'] in ['B2', 'B3']:
            item['nodes'], item['groups'], _ = token_inputs(observation, self.geometry, modalities=['pixel', 'bar'])
        elif self.config['family'] in ['B4', 'B4-MAE']:
            item['nodes'], item['groups'], item['modules'], _ = plane_tokens(observation, self.geometry, self.config, seed=index+42)
            item['region_blocks'] = self.config.get('region_blocks', 32)
        else:
            item['summary'] = np.r_[event_features(observation['pixel'], observation['bar']), summary]
        item['transform_seconds'] = time.perf_counter()-start
        return item


def collate(events):
    family = events[0]['family']
    if family in ['B4', 'B4-MAE']:
        batch = collate_planes(events)
    elif family in ['B2', 'B3']:
        batch = collate_spatial(events)
    else:
        batch = {'index': torch.tensor([e['index'] for e in events]), 'label': torch.tensor([e['label'] for e in events])}
    width = max(1, max(len(e['tracker']) for e in events))
    tracker = np.zeros((len(events), width, 5), dtype='float32')
    mask = np.zeros((len(events), width), dtype=bool)
    for i, event in enumerate(events):
        tracker[i, :len(event['tracker'])] = event['tracker']
        mask[i, :len(event['tracker'])] = True
    batch['tracker'] = torch.from_numpy(tracker)
    batch['tracker_mask'] = torch.from_numpy(mask)
    batch['presence'] = torch.from_numpy(np.stack([e['presence'] for e in events]))
    if family == 'B1':
        batch['projection_indices'] = torch.from_numpy(np.concatenate([e['projection_indices'] for e in events]))
        batch['projection_values'] = torch.from_numpy(np.concatenate([e['projection_values'] for e in events]))
        batch['projection_event'] = torch.repeat_interleave(torch.arange(len(events)), torch.tensor([len(e['projection_indices']) for e in events]))
    if family == 'B0':
        batch['summary'] = torch.from_numpy(np.stack([e['summary'] for e in events]))
    return batch
