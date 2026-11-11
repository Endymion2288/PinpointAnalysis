"""Small B0 summaries derived exclusively from observation whitelists."""

import numpy as np


def _shape(values):
    if len(values) == 0:
        return np.zeros(6, dtype='float64')
    return np.r_[values.mean(axis=0), values.std(axis=0)]


FEATURE_NAMES = ([f'pixel_{s}_{a}' for s in ['mean', 'std'] for a in ['x', 'y', 'z']]
                 + ['log_pixel_channels', 'pixel_planes']
                 + [f'bar_{s}_{a}' for s in ['mean', 'std'] for a in ['x', 'y', 'z']]
                 + ['log_bar_channels', 'log_bar_edep_proxy', 'log_bar_pe_proxy', 'bar_h_fraction']
                 + ['pixel_present', 'bar_present'])


def event_features(pixel, bars):
    xyz = pixel['xyz_mm'].astype('float64')
    bounds = bars['bounds_mm'].astype('float64')
    centers = np.column_stack([(bounds[:, 0] + bounds[:, 1]) / 2,
                               (bounds[:, 2] + bounds[:, 3]) / 2, bounds[:, 4]])
    values = np.r_[_shape(xyz), np.log1p(len(xyz)), len(np.unique(xyz[:, 2])),
                   _shape(centers), np.log1p(len(bounds)),
                   np.log1p(np.sum(bars['edep_proxy_MeV'], dtype='float64')),
                   np.log1p(np.sum(bars['pe_proxy'], dtype='float64')),
                   float(bars['horizontal'].mean()) if len(bounds) else 0,
                   float(len(xyz) > 0), float(len(bounds) > 0)]
    if len(values) != len(FEATURE_NAMES) or not np.isfinite(values).all():
        raise ValueError('invalid B0 observations')
    return values.astype('float32')
