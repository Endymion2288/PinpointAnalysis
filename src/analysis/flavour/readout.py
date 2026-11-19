"""Geometry-checked channel reducers. Positions mm; simulated deposits MeV.

Pixel observations are binary. Bar edep/PE are linear detector proxies; no
electronics window, timing or calibrated ADC is claimed. Tracker digitization
implements the explicitly unvalidated L3 resolution/efficiency hypothesis.
"""

import numpy as np


def flat(record, name, dtype=None):
    return np.asarray(record[name], dtype=dtype)


def require_range(values, size, name):
    if np.any((values < 0) | (values >= size)):
        raise ValueError(f'{name} outside geometry [0, {size})')


def pixel_channels(record, geometry):
    col, row, plane = [flat(record, k, np.int64) for k in ['colID', 'rowID', 'layerID']]
    nx, ny, nz = [len(geometry[k]) for k in ['pixel_Xpos', 'pixel_Ypos', 'pixel_Zpos']]
    for values, size, name in [(col, nx, 'pixel col'), (row, ny, 'pixel row'), (plane, nz, 'pixel plane')]:
        require_range(values, size, name)
    ids = np.unique((plane * ny + row) * nx + col)
    xyz = pixel_positions(ids, geometry)
    return {'channel_id': ids.astype('uint64'), 'xyz_mm': xyz.astype('float32')}


def pixel_positions(ids, geometry):
    nx, ny = len(geometry['pixel_Xpos']), len(geometry['pixel_Ypos'])
    ids = np.asarray(ids, dtype=np.int64)
    col, row, plane = ids % nx, (ids // nx) % ny, ids // (nx * ny)
    require_range(plane, len(geometry['pixel_Zpos']), 'pixel plane')
    return np.column_stack([np.asarray(geometry[k])[index] for k, index in
                            [('pixel_Xpos', col), ('pixel_Ypos', row), ('pixel_Zpos', plane)]])


def bar_channels(record, geometry, *, include_truth_edep=True):
    layer, panel, horizontal, row, col = [flat(record, k, np.int64) for k in
                                         ['layerID', 'panelID', 'isHorizontal', 'rowID', 'colID']]
    nbar = len(geometry['scint_bar_Xcenter'])
    if nbar != len(geometry['scint_bar_Ycenter']):
        raise ValueError('different H/V channel counts require a new key codec')
    require_range(horizontal, 2, 'bar orientation')
    require_range(panel, len(geometry['scint_Zpos']), 'bar panel')
    require_range(layer, len(geometry['layer_is_pixel']), 'bar layer')
    bar = np.where(horizontal == 1, row, col)
    unseen = np.where(horizontal == 1, col, row)
    require_range(bar, nbar, 'bar index')
    if np.any(unseen != -1):
        raise ValueError('unmeasured bar coordinate is not the expected -1')
    keys = (layer * len(geometry['scint_Zpos']) + panel) * (2 * nbar) + horizontal * nbar + bar
    unique, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    pe = flat(record, 'nPE', np.int64)
    if np.any(pe < 0):
        raise ValueError('negative simulated bar PE response')
    summed_pe = np.zeros(len(unique), dtype=np.int64)
    np.add.at(summed_pe, inverse, pe)
    if include_truth_edep:
        edep = flat(record, 'edep', np.float64)
        if np.any(~np.isfinite(edep)) or np.any(edep < 0):
            raise ValueError('negative/nonfinite simulated bar response')
        energy = np.bincount(inverse, weights=edep, minlength=len(unique))
        if not np.isclose(energy.astype('float32').sum(dtype='float64'), edep.sum(), rtol=2e-6, atol=1e-6):
            raise ValueError('float32 channel storage exceeds energy tolerance')
    else:
        edep = None
        energy = None
    hit_time_ns = None
    if 'hitTime' in record:
        ht = flat(record, 'hitTime', np.float64)
        if len(ht) == len(keys):
            min_t = np.full(len(unique), np.inf, dtype=np.float64)
            valid_t = np.isfinite(ht) & (ht >= 0.0)
            if np.any(valid_t):
                np.minimum.at(min_t, inverse[valid_t], ht[valid_t])
            min_t[~np.isfinite(min_t)] = -1.0
            hit_time_ns = min_t.astype('float32')
    h, b, p = horizontal[first], bar[first], panel[first]
    center_x, center_y = np.mean(geometry['scint_bar_Xcenter']), np.mean(geometry['scint_bar_Ycenter'])
    x = np.where(h == 1, center_x, np.asarray(geometry['scint_bar_Xcenter'])[b])
    y = np.where(h == 1, np.asarray(geometry['scint_bar_Ycenter'])[b], center_y)
    hx = np.where(h == 1, geometry['scint_detector_width'] / 2, geometry['scint_bar_width'] / 2)
    hy = np.where(h == 1, geometry['scint_bar_height'] / 2, geometry['scint_detector_height'] / 2)
    bounds = np.column_stack([x - hx, x + hx, y - hy, y + hy,
                              np.asarray(geometry['scint_Zpos'])[p]])
    out = {
        'channel_id': unique.astype('uint64'),
        'layer': layer[first].astype('int16'),
        'panel': p.astype('int16'),
        'horizontal': h.astype('int8'),
        'bar': b.astype('int16'),
        'bounds_mm': bounds.astype('float32'),
        'pe_proxy': summed_pe,
    }
    if include_truth_edep and energy is not None:
        out['edep_proxy_MeV'] = energy.astype('float32')
    if hit_time_ns is not None:
        out['hit_time_ns'] = hit_time_ns
    audit = {
        'input_pe': int(pe.sum()),
        'output_pe': int(summed_pe.sum()),
        'records': len(keys),
        'channels': len(unique),
    }
    if include_truth_edep and edep is not None and energy is not None:
        audit['input_edep_MeV'] = float(edep.sum())
        audit['output_edep_MeV'] = float(energy.sum())
    return out, audit


def tracker_measurements(record, seed, merge_pitch_mm=.1, efficiency=.99,
                         sigma_x_mm=.58, sigma_y_mm=.017):
    station = flat(record, 'trackerID', np.int64)
    require_range(station, 3, 'tracker station')
    xyz = np.column_stack([flat(record, k, np.float64) for k in ['x', 'y', 'z']])
    if not np.isfinite(xyz).all():
        raise ValueError('nonfinite tracker crossing')
    # Grid merge is truth-free and permutation invariant, unlike a true-track
    # highest-energy crossing selection. Cell centers avoid step-density bias.
    cells = np.column_stack([station, np.floor(xyz[:, :2] / merge_pitch_mm).astype('int64')])
    unique = np.unique(cells, axis=0)
    rng = np.random.default_rng(seed)
    keep = rng.random(len(unique)) < efficiency
    unique = unique[keep]
    s = unique[:, 0].astype('int8')
    coordinates = (unique[:, 1:].astype('float64') + .5) * merge_pitch_mm
    coordinates += rng.normal(size=coordinates.shape) * [sigma_x_mm, sigma_y_mm]
    # No station z, field, momentum or charge is inferred from event truth.
    ideal = {'station': station.astype('int8'), 'xyz_mm': xyz.astype('float32')}
    digitized = {'station': s, 'xy_mm': coordinates.astype('float32'),
                 'sigma_xy_mm': np.tile([sigma_x_mm, sigma_y_mm], (len(s), 1)).astype('float32')}
    return ideal, digitized


def modality_state(pixel, bars, tracker):
    # Measured response state only. Missing crossings cannot determine whether
    # a particle was outside acceptance; that status is deliberately unknown.
    return {'has_response': np.asarray([len(pixel['channel_id']) > 0,
                                        len(bars['channel_id']) > 0,
                                        len(tracker['station']) > 0], dtype=bool),
            'outside_acceptance': np.full(3, -1, dtype='int8'),
            'fit_attempted': False, 'momentum_censored': np.int8(-1)}
