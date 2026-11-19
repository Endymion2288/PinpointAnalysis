"""Zero-hole compact sensor projections, z-plane layout, and exact Y-mirror augmentation."""

from __future__ import annotations

import numpy as np
import torch

from analysis.flavour.v3.common.domains import (
    calibrated_bar_channels,
    cumulative_radiation_length,
    detector_bounds,
    validate_finite_observation,
)
from analysis.flavour.v3.v1_b1pp.spectrometer import (
    CAND_Y_FLIP_IDX,
    HIT_Y_FLIP_IDX,
    extract_spectrometer_kinematics,
)


FINE_HALF_MM = 128.0
FINE_BINS = 512
WIDE_HALF_MM = 140.0
WIDE_BINS = 256


def static_plane_layout(geometry: dict, bounds: dict | None = None) -> dict:
    """Build deterministic z-sorted 111-plane physical layout (15 pix + 54 Xbar + 42 Ybar)."""
    if bounds is None:
        bounds = detector_bounds(geometry)
    pz = bounds['pixel_Zpos']
    sz = bounds['scint_Zpos']
    vp = bounds['vert_panels']
    hp = bounds['horiz_panels']
    z_mid = float(bounds['z_mid'])

    entries = []
    for i, z in enumerate(pz):
        entries.append((float(z), 0, i))
    for i, p in enumerate(vp):
        entries.append((float(sz[p]), 1, i))
    for i, p in enumerate(hp):
        entries.append((float(sz[p]), 2, i))
    # Sort strictly ascending by physical z_mm (tie-break by modality)
    entries.sort(key=lambda t: (t[0], t[1], t[2]))

    n_seq = len(entries)
    seq_z_mm = np.array([e[0] for e in entries], dtype=np.float32)
    seq_mod = np.array([e[1] for e in entries], dtype=np.int64)
    seq_src = np.array([e[2] for e in entries], dtype=np.int64)
    seq_x0 = np.array(
        [cumulative_radiation_length(float(z), geometry) for z in seq_z_mm],
        dtype=np.float32,
    )

    dz_prev = np.r_[0.0, np.maximum(0.0, np.diff(seq_z_mm))].astype(np.float32)
    dz_next = np.r_[np.maximum(0.0, np.diff(seq_z_mm)), 0.0].astype(np.float32)

    pix_to_seq = np.zeros(len(pz), dtype=np.int64)
    xbar_to_seq = np.zeros(len(vp), dtype=np.int64)
    ybar_to_seq = np.zeros(len(hp), dtype=np.int64)
    for seq_idx, (_, mod, src_idx) in enumerate(entries):
        if mod == 0:
            pix_to_seq[src_idx] = seq_idx
        elif mod == 1:
            xbar_to_seq[src_idx] = seq_idx
        else:
            ybar_to_seq[src_idx] = seq_idx

    # Plane continuous features: [z_norm, log1p(dz_prev)/5, log1p(dz_next)/5, cum_X0/50]
    plane_geom = np.column_stack([
        (seq_z_mm - z_mid) / 1000.0,
        np.log1p(dz_prev) / 5.0,
        np.log1p(dz_next) / 5.0,
        seq_x0 / 50.0,
    ]).astype(np.float32)

    return {
        'n_seq': n_seq,
        'n_pix': len(pz),
        'n_xbar': len(vp),
        'n_ybar': len(hp),
        'n_bars': int(bounds['n_bars']),
        'seq_z_mm': seq_z_mm,
        'seq_x0': seq_x0,
        'seq_mod': seq_mod,
        'seq_src': seq_src,
        'pix_to_seq': pix_to_seq,
        'xbar_to_seq': xbar_to_seq,
        'ybar_to_seq': ybar_to_seq,
        'plane_geom': plane_geom,
        'pix_z_norm': plane_geom[pix_to_seq, 0],
        'pix_x0_norm': plane_geom[pix_to_seq, 3],
        'xbar_z_norm': plane_geom[xbar_to_seq, 0],
        'xbar_x0_norm': plane_geom[xbar_to_seq, 3],
        'ybar_z_norm': plane_geom[ybar_to_seq, 0],
        'ybar_x0_norm': plane_geom[ybar_to_seq, 3],
    }


def _project_pixel_window(
    planes: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    u_center: float,
    half_width_mm: float,
    n_bins: int,
    n_planes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin pixel hits into compact occupied (plane, bin) sparse coordinates + 3 features."""
    if len(planes) == 0:
        return np.empty(0, dtype=np.int16), np.empty((0, 3), dtype=np.float16)

    rel_u = u - u_center
    keep = (rel_u >= -half_width_mm) & (rel_u < half_width_mm)
    if not keep.any():
        return np.empty(0, dtype=np.int16), np.empty((0, 3), dtype=np.float16)

    p = planes[keep].astype(np.int64)
    uk = rel_u[keep]
    vk = v[keep]
    pitch = (2.0 * half_width_mm) / n_bins
    bins = np.clip(np.floor((uk + half_width_mm) / pitch).astype(np.int64), 0, n_bins - 1)
    cell = p * n_bins + bins

    # Sort by (cell, uk) to compute counts, max contiguous u-cluster span, and v-spread
    order = np.lexsort((uk, cell))
    c_s = cell[order]
    u_s = uk[order]
    v_s = vk[order]

    uniq_mask = np.r_[True, c_s[1:] != c_s[:-1]]
    starts = np.flatnonzero(uniq_mask)
    ends = np.r_[starts[1:], len(c_s)]
    counts = ends - starts
    uniq_cells = c_s[starts].astype(np.int16)

    # Orthogonal span per bin (invariant under reflection)
    v_max = np.maximum.reduceat(v_s, starts)
    v_min = np.minimum.reduceat(v_s, starts)
    ortho_spread = np.minimum(2.0, (v_max - v_min) / 10.0)

    # Max contiguous u-subcluster span (gap <= 0.06 mm) inside each bin
    if len(c_s) == 1:
        max_span = np.zeros(1, dtype=np.float64)
    else:
        same_cell = c_s[1:] == c_s[:-1]
        close_u = (u_s[1:] - u_s[:-1]) <= 0.06
        break_idx = np.flatnonzero(~(same_cell & close_u)) + 1
        sub_starts = np.r_[0, break_idx]
        sub_ends = np.r_[break_idx, len(c_s)] - 1
        sub_spans = u_s[sub_ends] - u_s[sub_starts]
        # Map each subcluster back to its cell index in 0..len(starts)-1
        sub_cell_pos = np.searchsorted(starts, sub_starts, side='right') - 1
        max_span = np.zeros(len(starts), dtype=np.float64)
        np.maximum.at(max_span, sub_cell_pos, sub_spans)
        max_span = np.minimum(2.0, max_span)

    vals = np.column_stack([
        np.log1p(counts) / 6.0,
        max_span,
        ortho_spread,
    ]).astype(np.float16)
    return uniq_cells, vals


def extract_b1pp_item(
    index: int,
    observation: dict,
    label: int | None,
    dom: dict,
    geometry: dict,
    bounds: dict,
    config: dict,
) -> dict:
    """Transform one raw event observation into compact sparse B1++ arrays (~1.6 KB)."""
    pe_only = bool((config or {}).get('pe_only', False))
    validate_finite_observation(observation, check_edep=not pe_only)

    pixel = observation['pixel']
    bar = observation['bar']
    tracker = observation['tracker_hypothesis']

    n_pix_planes = bounds['nz']
    nx, ny = bounds['nx'], bounds['ny']
    n_bars = bounds['n_bars']

    pcid = np.asarray(pixel['channel_id'], dtype=np.int64)
    pxyz = np.asarray(pixel['xyz_mm'], dtype=np.float64).reshape(-1, 3)
    p_planes = pcid // (nx * ny)

    fx_idx, fx_val = _project_pixel_window(
        p_planes, pxyz[:, 0], pxyz[:, 1], bounds['x_pix_center'], FINE_HALF_MM, FINE_BINS, n_pix_planes
    )
    fy_idx, fy_val = _project_pixel_window(
        p_planes, pxyz[:, 1], pxyz[:, 0], bounds['y_scint_center'], FINE_HALF_MM, FINE_BINS, n_pix_planes
    )
    wx_idx, wx_val = _project_pixel_window(
        p_planes, pxyz[:, 0], pxyz[:, 1], bounds['x_pix_center'], WIDE_HALF_MM, WIDE_BINS, n_pix_planes
    )
    wy_idx, wy_val = _project_pixel_window(
        p_planes, pxyz[:, 1], pxyz[:, 0], bounds['y_scint_center'], WIDE_HALF_MM, WIDE_BINS, n_pix_planes
    )

    keep_bar, be, bpe = calibrated_bar_channels(bar, config)
    bp = np.asarray(bar['panel'], dtype=np.int64)[keep_bar]
    bh = np.asarray(bar['horizontal'], dtype=np.int64)[keep_bar]
    bi = np.asarray(bar['bar'], dtype=np.int64)[keep_bar]

    vmask = bh == 0
    hmask = bh == 1
    if vmask.any():
        vx_p = bounds['panel_to_vert'][bp[vmask]]
        bar_x_idx = (vx_p * n_bars + bi[vmask]).astype(np.int16)
        bar_x_val = np.column_stack([
            np.log1p(be[vmask]) / 10.0,
            np.log1p(bpe[vmask]) / 15.0,
        ]).astype(np.float16)
    else:
        bar_x_idx = np.empty(0, dtype=np.int16)
        bar_x_val = np.empty((0, 2), dtype=np.float16)

    if hmask.any():
        hy_p = bounds['panel_to_horiz'][bp[hmask]]
        bar_y_idx = (hy_p * n_bars + bi[hmask]).astype(np.int16)
        bar_y_val = np.column_stack([
            np.log1p(be[hmask]) / 10.0,
            np.log1p(bpe[hmask]) / 15.0,
        ]).astype(np.float16)
    else:
        bar_y_idx = np.empty(0, dtype=np.int16)
        bar_y_val = np.empty((0, 2), dtype=np.float16)

    spec = extract_spectrometer_kinematics(tracker, dom['cal_start'], bounds)

    out = {
        'index': int(index),
        'obs_domain_id': int(dom['obs_domain_id']),
        'c_obs': dom['c_obs'].astype(np.float32),
        'fx_idx': fx_idx,
        'fx_val': fx_val,
        'fy_idx': fy_idx,
        'fy_val': fy_val,
        'wx_idx': wx_idx,
        'wx_val': wx_val,
        'wy_idx': wy_idx,
        'wy_val': wy_val,
        'bx_idx': bar_x_idx,
        'bx_val': bar_x_val,
        'by_idx': bar_y_idx,
        'by_val': bar_y_val,
        'trk_cand': spec['cand'],
        'trk_cand_mask': spec['cand_mask'],
        'trk_hits': spec['hits'],
    }
    if label is not None:
        out['label'] = int(label)
    return out


def apply_y_mirror_to_item(item: dict, n_bars: int = 40) -> dict:
    """Apply exact detector-centered Y-reflection (y -> 2*y_c - y) to one transformed item."""
    out = dict(item)
    if len(item['by_idx']):
        idx = item['by_idx'].astype(np.int64)
        out['by_idx'] = ((idx // n_bars) * n_bars + (n_bars - 1 - (idx % n_bars))).astype(np.int16)
    if len(item['fy_idx']):
        idx = item['fy_idx'].astype(np.int64)
        out['fy_idx'] = ((idx // FINE_BINS) * FINE_BINS + (FINE_BINS - 1 - (idx % FINE_BINS))).astype(np.int16)
    if len(item['wy_idx']):
        idx = item['wy_idx'].astype(np.int64)
        out['wy_idx'] = ((idx // WIDE_BINS) * WIDE_BINS + (WIDE_BINS - 1 - (idx % WIDE_BINS))).astype(np.int16)

    c_obs = item['c_obs'].copy()
    c_obs[10] = -c_obs[10]
    out['c_obs'] = c_obs

    cand = item['trk_cand'].copy()
    cand[:, CAND_Y_FLIP_IDX] *= -1.0
    out['trk_cand'] = cand

    hits = item['trk_hits'].copy()
    if len(hits):
        hits[:, HIT_Y_FLIP_IDX] *= -1.0
    out['trk_hits'] = hits
    return out


def collate_b1pp(events: list[dict], mirror_rng: np.random.Generator | None = None) -> dict:
    """Collate compact sparse B1++ events with optional exact Y-mirror augmentation."""
    if mirror_rng is not None:
        do_flip = mirror_rng.random(len(events)) < 0.5
        events = [apply_y_mirror_to_item(e) if f else e for e, f in zip(events, do_flip)]

    b = len(events)
    max_hits = max(1, max(len(e['trk_hits']) for e in events))
    trk_hits = np.zeros((b, max_hits, 6), dtype=np.float32)
    trk_hit_mask = np.zeros((b, max_hits), dtype=bool)
    for i, e in enumerate(events):
        nh = len(e['trk_hits'])
        if nh:
            trk_hits[i, :nh] = e['trk_hits']
            trk_hit_mask[i, :nh] = True

    batch = {
        'index': torch.tensor([e['index'] for e in events], dtype=torch.long),
        'obs_domain_id': torch.tensor([e['obs_domain_id'] for e in events], dtype=torch.long),
        'c_obs': torch.from_numpy(np.stack([e['c_obs'] for e in events])),
        'trk_cand': torch.from_numpy(np.stack([e['trk_cand'] for e in events])),
        'trk_cand_mask': torch.from_numpy(np.stack([e['trk_cand_mask'] for e in events])),
        'trk_hits': torch.from_numpy(trk_hits),
        'trk_hit_mask': torch.from_numpy(trk_hit_mask),
    }
    if 'label' in events[0]:
        batch['label'] = torch.tensor([e['label'] for e in events], dtype=torch.long)

    for key in ('fx', 'fy', 'wx', 'wy', 'bx', 'by'):
        idxs = [e[f'{key}_idx'] for e in events]
        vals = [e[f'{key}_val'] for e in events]
        counts = np.array([len(x) for x in idxs], dtype=np.int64)
        if counts.sum() > 0:
            ev_ix = np.repeat(np.arange(b, dtype=np.int64), counts)
            flat_idx = np.concatenate(idxs).astype(np.int64)
            flat_val = np.concatenate(vals, axis=0).astype(np.float32)
        else:
            ev_ix = np.empty(0, dtype=np.int64)
            flat_idx = np.empty(0, dtype=np.int64)
            n_ch = 2 if key.startswith('b') else 3
            flat_val = np.empty((0, n_ch), dtype=np.float32)
        batch[f'{key}_ev'] = torch.from_numpy(ev_ix)
        batch[f'{key}_idx'] = torch.from_numpy(flat_idx)
        batch[f'{key}_val'] = torch.from_numpy(flat_val)

    if 'sample_weight' in events[0]:
        batch['sample_weight'] = torch.tensor([e['sample_weight'] for e in events], dtype=torch.float32)
    return batch
