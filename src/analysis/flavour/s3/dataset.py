"""Observation-only plane windows, native residuals and physical module labels."""

import numpy as np
import torch

from analysis.flavour.s2.dataset import SpatialDataset
from analysis.flavour.s2.representation import token_inputs


def plane_tokens(observations, geometry, config, seed=0, *, build_input_blocks=False):
    """Preserve every represented channel; never use native residuals after rebin.

    Pixel module = ordinal true entry of layer_is_pixel; bar module = stored
    hardware layer. Tracker modules follow the target layers, using station ID.
    Groups contain one module/type and at most 32 channels. Pixel cells are
    physical pitch multiples, with absolute-origin windows as in S2.
    """
    features, groups, modules = [], [], []
    counts = []
    offset = 0
    layout = np.flatnonzero(geometry['layer_is_pixel'])
    n_layers = len(geometry['layer_is_pixel'])
    for detector in config.get('modalities', ['pixel', 'bar']):
        nodes, _, raw = token_inputs(observations, geometry,
            pitch_mm=config.get('pitch_mm', 0.), modalities=[detector], seed=seed,
            build_blocks=build_input_blocks)
        counts.extend(raw)
        nodes = nodes[nodes[:, 15] > 0].copy()
        if not len(nodes):
            continue
        xyz = nodes[:, :3].astype('float64') * [210, 210, 3000]
        if detector == 'pixel':
            plane = np.abs(xyz[:, 2, None] - np.asarray(geometry['pixel_Zpos'])).argmin(1)
            module = layout[plane]
            size = config.get('window_pixels', 16)
            width = (np.array([4., 4.]) if size == 0 else size * np.array([
                np.median(np.diff(geometry['pixel_Xpos'])), np.median(np.diff(geometry['pixel_Ypos']))]))
            cell = np.floor(xyz[:, :2] / width)
            nodes[:, 16:18] = ((xyz[:, :2] / width - cell) - .5) * nodes[:, 3:5]
            keys = np.column_stack([module, plane, cell])
        elif detector == 'bar':
            bars = observations['bar']
            z = np.asarray(bars['bounds_mm']).reshape(-1, 5)[:, 4]
            # Hardware layer is explicitly stored in accepted observations.
            mapping = {}
            for value, layer in zip(z, bars['layer']):
                key = round(float(value), 3)
                if key in mapping and mapping[key] != int(layer):
                    raise ValueError('one bar z belongs to multiple hardware layers')
                mapping[key] = int(layer)
            module = np.asarray([mapping[round(float(v), 3)] for v in xyz[:, 2]])
            keys = np.column_stack([module, np.round(xyz[:, 2], 3)])
        else:
            module = n_layers + np.rint(xyz[:, 2] / 1000).astype('int64')
            keys = np.column_stack([module, np.floor(xyz[:, :2] / 16)])
        if not config.get('signal', True):
            nodes[:, 6:8] = 0
        if not config.get('local_residual', True):
            nodes[:, 16:18] = 0
        _, inverse = np.unique(keys, axis=0, return_inverse=True)
        order = np.lexsort((nodes[:, 1], nodes[:, 0], inverse))
        ordered_group = inverse[order]
        boundaries = np.r_[0, np.flatnonzero(np.diff(ordered_group)) + 1, len(order)]
        ids, block_ids = [], []
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            for start in range(left, right, 32):
                take = order[start:min(start + 32, right)]
                ids.extend(take)
                block_ids.extend([offset] * len(take))
                offset += 1
        ids = np.asarray(ids)
        features.append(nodes[ids])
        groups.extend(block_ids)
        modules.extend(module[ids])
    if not features:
        return np.zeros((1, 18), dtype='float32'), np.array([0]), np.array([0]), counts
    return np.concatenate(features), np.asarray(groups), np.asarray(modules), counts


class PlaneDataset(SpatialDataset):
    def __getitem__(self, index):
        event = self.native[int(self.indexes[index])]
        nodes, groups, modules, counts = plane_tokens(event['observations'], self.geometry, self.config, seed=index + 42)
        return {'nodes': nodes, 'groups': groups, 'modules': modules,
                'label': event['targets']['label'], 'index': index, 'raw_counts': counts,
                'region_blocks': self.config.get('region_blocks', 32)}


def collate_planes(events):
    counts_by_event = [np.bincount(event['groups']) for event in events]
    total_blocks = sum(len(counts) for counts in counts_by_event)
    nodes = np.zeros((total_blocks, 32, 18), dtype='float32')
    masks = np.zeros((total_blocks, 32), dtype=bool)
    event_mask = np.zeros((len(events), max(map(len, counts_by_event))), dtype=bool)
    keys, geometry, event_ids, block_ids = [], [], [], []
    offset = 0
    for event_id, (event, counts) in enumerate(zip(events, counts_by_event)):
        groups = event['groups']
        if np.any(np.diff(groups) < 0) or counts.max() > 32 or counts.min() < 1:
            raise ValueError('S3 requires canonical contiguous blocks of 1..32 nodes')
        starts = np.r_[0, np.cumsum(counts)[:-1]]
        within = np.arange(len(groups)) - starts[groups]
        nodes[offset + groups, within] = event['nodes']
        masks[offset + groups, within] = True
        first = event['nodes'][starts]
        module = event['modules'][starts]
        detector = first[:, 12:15].argmax(1)
        detector[first[:, 15] == 0] = 3
        keys.extend(zip(np.full(len(counts), event_id), module, detector))
        geo = np.zeros((len(counts), 18), dtype='float32')
        geo[:, 2] = np.bincount(groups, weights=event['nodes'][:, 2]) / counts
        geo[:, 12:16] = first[:, 12:16]
        geometry.extend(geo)
        event_mask[event_id, :len(counts)] = True
        event_ids.extend([event_id] * len(counts))
        block_ids.extend(range(len(counts)))
        offset += len(counts)
    batch = {'nodes': torch.from_numpy(nodes), 'local_mask': torch.from_numpy(masks),
             'event_mask': torch.from_numpy(event_mask), 'event_index': torch.tensor(event_ids),
             'block_index': torch.tensor(block_ids), 'label': torch.tensor([e['label'] for e in events]),
             'index': torch.tensor([e['index'] for e in events])}
    unique, inverse = np.unique(np.asarray(keys), axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    positions = np.empty(len(inverse), dtype='int64')
    order = np.argsort(inverse, kind='stable')
    positions[order] = np.arange(len(inverse)) - np.repeat(np.r_[0, np.cumsum(counts)[:-1]], counts)
    batch['module_group'] = torch.from_numpy(inverse)
    batch['module_position'] = torch.from_numpy(positions)
    batch['module_mask'] = torch.from_numpy(np.arange(counts.max())[None] < counts[:, None])
    batch['module_event'] = torch.from_numpy(unique[:, 0])
    batch['module_id'] = torch.from_numpy(unique[:, 1])
    batch['module_detector'] = torch.from_numpy(unique[:, 2])
    geo = np.zeros((len(unique), 18), dtype='float32')
    np.add.at(geo, inverse, np.asarray(geometry))
    geo /= counts[:, None]
    batch['module_geometry'] = torch.from_numpy(geo)
    # Revision 2 partitions each physical module into an uncapped number of
    # canonical spatial regions. Every original block belongs to one region.
    region_sizes = np.asarray([events[i].get('region_blocks', 32) for i in unique[inverse, 0]])
    region_keys, region_inverse = np.unique(
        np.column_stack([inverse, positions // region_sizes]), axis=0, return_inverse=True)
    region_counts = np.bincount(region_inverse)
    region_positions = np.empty(len(inverse), dtype='int64')
    region_order = np.argsort(region_inverse, kind='stable')
    region_positions[region_order] = np.arange(len(inverse)) - np.repeat(
        np.r_[0, np.cumsum(region_counts)[:-1]], region_counts)
    batch['region_group'] = torch.from_numpy(region_inverse)
    batch['region_position'] = torch.from_numpy(region_positions)
    batch['region_mask'] = torch.from_numpy(np.arange(region_counts.max())[None] < region_counts[:, None])
    for name, column in [('event', 0), ('id', 1), ('detector', 2)]:
        batch['region_' + name] = torch.from_numpy(unique[region_keys[:, 0], column])
    region_geo = np.zeros((len(region_keys), 18), dtype='float32')
    np.add.at(region_geo, region_inverse, np.asarray(geometry))
    region_geo /= region_counts[:, None]
    batch['region_geometry'] = torch.from_numpy(region_geo)
    batch['block_detector'] = torch.from_numpy(unique[inverse, 2])
    return batch
