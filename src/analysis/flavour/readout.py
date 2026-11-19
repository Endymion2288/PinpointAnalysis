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
            # Only records with actual photoelectrons (pe > 0) may determine aggregated channel timing (Stage 322-12).
            valid_t = np.isfinite(ht) & (ht >= 0.0) & (pe > 0)
            if np.any(valid_t):
                np.minimum.at(min_t, inverse[valid_t], ht[valid_t])
            min_t[(~np.isfinite(min_t)) | (summed_pe <= 0)] = -1.0
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
                         sigma_x_mm=.58, sigma_y_mm=.017,
                         dead_fraction=0.0, noise_rate_per_station=0.0):
    station = flat(record, 'trackerID', np.int64)
    require_range(station, 3, 'tracker station')
    xyz = np.column_stack([flat(record, k, np.float64) for k in ['x', 'y', 'z']])
    if not np.isfinite(xyz).all():
        raise ValueError('nonfinite tracker crossing')
    ideal = {'station': station.astype('int8'), 'xyz_mm': xyz.astype('float32')}
    rng = np.random.default_rng(seed)

    if len(station) > 0:
        # Canonical ordering guarantees strict permutation invariance even in float64 summation.
        order = np.lexsort((xyz[:, 1], xyz[:, 0], station))
        st_s = station[order]
        xy_s = xyz[order, :2]
        # Two-hit cell merging uses exact crossing centroids instead of coarse cell centers
        # so 100 um cell quantization does not inflate the calibrated 17 um Y spacepoint RMS (Stage 322-07).
        cells = np.column_stack([st_s, np.floor(xy_s / merge_pitch_mm).astype('int64')])
        unique, inverse = np.unique(cells, axis=0, return_inverse=True)
        counts = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
        cx = np.bincount(inverse, weights=xy_s[:, 0], minlength=len(unique)) / counts
        cy = np.bincount(inverse, weights=xy_s[:, 1], minlength=len(unique)) / counts
        eff_keep = float(efficiency) * max(0.0, 1.0 - float(dead_fraction))
        keep = rng.random(len(unique)) < eff_keep
        s = unique[keep, 0].astype('int8')
        coordinates = np.column_stack([cx[keep], cy[keep]])
        if len(s):
            coordinates += rng.normal(size=coordinates.shape) * [sigma_x_mm, sigma_y_mm]
    else:
        s = np.empty(0, dtype='int8')
        coordinates = np.empty((0, 2), dtype='float64')

    if noise_rate_per_station > 0.0:
        ns_list, nc_list = [], []
        for st_id in range(3):
            n_noise = int(rng.poisson(float(noise_rate_per_station)))
            if n_noise > 0:
                ns_list.append(np.full(n_noise, st_id, dtype='int8'))
                nc_list.append(rng.uniform(-125.0, 125.0, size=(n_noise, 2)))
        if ns_list:
            s = np.concatenate([s, *ns_list]) if len(s) else np.concatenate(ns_list)
            coordinates = np.vstack([coordinates, *nc_list]) if len(coordinates) else np.vstack(nc_list)

    # No station z, field, momentum or charge is inferred from event truth.
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
