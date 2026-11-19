"""Sub-mm connected pixel cluster tokenizer, 32x32 micro-patches, and PF collation."""

from __future__ import annotations

import math
import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn
from torch.nn import functional as F

from analysis.flavour.v3.common.domains import (
    calibrated_bar_channels,
    compute_observable_containment,
    cumulative_radiation_length,
    validate_finite_observation,
)
from analysis.flavour.v3.common.fast_kernels import (
    MICRO_VTX_FEAT_DIM,
    plane_neighborhoods_fast,
    rasterize_patches_fast,
    reconstruct_micron_vertex,
)
from analysis.flavour.v3.common.pixel_clusters import cluster_pixel_hits
from analysis.flavour.v3.v1_b1pp.spectrometer import (
    CAND_Y_FLIP_IDX,
    HIT_Y_FLIP_IDX,
    extract_spectrometer_kinematics,
)


N_MAX_CLUSTERS = 1024
N_MAX_TARGET_CLUSTERS = 832
N_MAX_IPT_CLUSTERS = 192
N_MAX_PATCHES = 64
N_MAX_TARGET_PATCHES = 48
N_MAX_IPT_PATCHES = 16
PATCH_SIZE = 32
PIX_FEAT_DIM = 22
PIX_PLANE_SUMMARY_DIM = 12
PATCH_TRIGGER_MIN_HITS = 3
PATCH_TRIGGER_MIN_SPAN_MM = 0.04
PIX_Y_FLIP_IDX = (1, 11, 17)
PLANE_SUMMARY_Y_FLIP_IDX = (4, 6)


def resolve_sensor_pitches(
    bounds: dict,
    config: dict | None = None,
) -> dict[str, float | None]:
    """Resolve independent target-silicon and IPT-silicon pitch & clustering scales (Stage 322-03)."""
    cfg = config or {}
    q_tgt_raw = cfg.get('quantize_target_xy_mm')
    q_ipt_raw = cfg.get('quantize_ipt_xy_mm')
    q_all_raw = cfg.get('quantize_xy_mm')
    q_tgt_only = bool(cfg.get('quantize_target_only', False))

    q_tgt = float(q_tgt_raw) if (q_tgt_raw is not None and float(q_tgt_raw) > 0.0) else None
    q_ipt = float(q_ipt_raw) if (q_ipt_raw is not None and float(q_ipt_raw) > 0.0) else None

    if q_tgt is None and q_ipt is None and q_all_raw is not None and float(q_all_raw) > 0.0:
        q_tgt = float(q_all_raw)
        q_ipt = None if q_tgt_only else float(q_all_raw)

    dx_nat = float(bounds['dx_mm'])
    dy_nat = float(bounds['dy_mm'])

    px_tgt = q_tgt if q_tgt is not None else dx_nat
    py_tgt = q_tgt if q_tgt is not None else dy_nat
    r_tgt = max(0.06, 1.45 * q_tgt) if q_tgt is not None else 0.06

    px_ipt = q_ipt if q_ipt is not None else dx_nat
    py_ipt = q_ipt if q_ipt is not None else dy_nat
    r_ipt = max(0.06, 1.45 * q_ipt) if q_ipt is not None else 0.06

    return {
        'q_tgt': q_tgt,
        'q_ipt': q_ipt,
        'px_tgt': px_tgt,
        'py_tgt': py_tgt,
        'r_tgt': r_tgt,
        'px_ipt': px_ipt,
        'py_ipt': py_ipt,
        'r_ipt': r_ipt,
    }


def quantize_pixel_hits(
    pcid: np.ndarray,
    pxyz: np.ndarray,
    bounds: dict,
    quantize_xy_mm: float | None = None,
    *,
    target_planes_only: bool = False,
    n_target_planes: int = 12,
    quantize_target_xy_mm: float | None = None,
    quantize_ipt_xy_mm: float | None = None,
) -> tuple[np.ndarray, np.ndarray, int, int, float, float, np.ndarray]:
    """Snap hits per sensor group (target vs IPT) independently to symmetric q-mm binary grids.

    Target planes (`plane < n_target_planes`) and IPT planes (`plane >= n_target_planes`) use
    strictly decoupled quantization grids so varying target pitch never alters IPT hits or IDs,
    and varying IPT pitch never alters target hits or IDs (Stage 322-03 fix).
    """
    q_tgt = float(quantize_target_xy_mm) if (quantize_target_xy_mm is not None and float(quantize_target_xy_mm) > 0.0) else None
    q_ipt = float(quantize_ipt_xy_mm) if (quantize_ipt_xy_mm is not None and float(quantize_ipt_xy_mm) > 0.0) else None
    if q_tgt is None and q_ipt is None and quantize_xy_mm is not None and float(quantize_xy_mm) > 0.0:
        q_tgt = float(quantize_xy_mm)
        q_ipt = None if target_planes_only else float(quantize_xy_mm)

    x_c = float(bounds['x_pix_center'])
    y_c = float(bounds['y_scint_center'])
    nx_raw = int(bounds['nx'])
    ny_raw = int(bounds['ny'])
    planes = pcid // (nx_raw * ny_raw)

    if q_tgt is None and q_ipt is None:
        uniq = np.arange(len(pcid), dtype=np.int64)
        return pcid.astype(np.int64, copy=True), pxyz.astype(np.float64, copy=True), nx_raw, ny_raw, float(bounds['dx_mm']), float(bounds['dy_mm']), uniq

    def _grid_dims(q_val: float) -> tuple[int, int]:
        half_x = max(abs(bounds['px_max'] - x_c), abs(x_c - bounds['px_min'])) + q_val
        half_y = max(abs(bounds['py_max'] - y_c), abs(y_c - bounds['py_min'])) + q_val
        return int(math.ceil(2.0 * half_x / q_val)) | 1, int(math.ceil(2.0 * half_y / q_val)) | 1

    # Compute common safe stride so target & IPT channel IDs never collide even on tiny synthetic grids
    nx_stride, ny_stride = nx_raw, ny_raw
    if q_tgt is not None:
        nxg, nyg = _grid_dims(q_tgt)
        nx_stride = max(nx_stride, nxg)
        ny_stride = max(ny_stride, nyg)
    if q_ipt is not None:
        nxg, nyg = _grid_dims(q_ipt)
        nx_stride = max(nx_stride, nxg)
        ny_stride = max(ny_stride, nyg)

    xyz_out = pxyz.astype(np.float64, copy=True)
    if nx_stride == nx_raw and ny_stride == ny_raw:
        cid_out = pcid.astype(np.int64, copy=True)
    else:
        col_raw = pcid % nx_raw
        row_raw = (pcid // nx_raw) % ny_raw
        cid_out = (planes * ny_stride + row_raw) * nx_stride + col_raw

    is_tgt = planes < int(n_target_planes)
    for mask, q_val in ((is_tgt, q_tgt), (~is_tgt, q_ipt)):
        if q_val is None or not mask.any():
            continue
        nx_q, ny_q = _grid_dims(q_val)
        hx = nx_q // 2
        hy = ny_q // 2
        col_q = np.clip(np.rint((pxyz[mask, 0] - x_c) / q_val).astype(np.int64) + hx, 0, nx_q - 1)
        row_q = np.clip(np.rint((pxyz[mask, 1] - y_c) / q_val).astype(np.int64) + hy, 0, ny_q - 1)
        xyz_out[mask, 0] = x_c + (col_q - hx) * q_val
        xyz_out[mask, 1] = y_c + (row_q - hy) * q_val
        cid_out[mask] = (planes[mask] * ny_stride + row_q) * nx_stride + col_q

    _, uniq = np.unique(cid_out, return_index=True)
    uniq = np.sort(uniq)
    rep_q = q_tgt if q_tgt is not None else float(q_ipt)
    return cid_out[uniq], xyz_out[uniq], nx_stride, ny_stride, rep_q, rep_q, uniq


def _compute_plane_neighborhoods(
    cl_plane: np.ndarray,
    x_mean: np.ndarray,
    y_mean: np.ndarray,
    n_hits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-cluster same-plane cKDTree counts in r <= 0.15, 0.40, 1.0 and 5.0 mm."""
    n_cl = len(cl_plane)
    if n_cl <= 1:
        z = np.zeros(n_cl, dtype=np.float64)
        return z, z.copy(), z.copy(), z.copy(), z.copy(), z.copy(), z.copy()
    fast_nb = plane_neighborhoods_fast(cl_plane, x_mean, y_mean, n_hits)
    if fast_nb is not None:
        return fast_nb
    cnt_r015 = np.zeros(n_cl, dtype=np.float64)
    hits_r015 = np.zeros(n_cl, dtype=np.float64)
    cnt_r040 = np.zeros(n_cl, dtype=np.float64)
    hits_r040 = np.zeros(n_cl, dtype=np.float64)
    cnt_r1 = np.zeros(n_cl, dtype=np.float64)
    cnt_r5 = np.zeros(n_cl, dtype=np.float64)
    hits_r5 = np.zeros(n_cl, dtype=np.float64)

    coords = np.column_stack([x_mean, y_mean])
    for p in np.unique(cl_plane):
        idx = np.flatnonzero(cl_plane == p)
        if len(idx) <= 1:
            continue
        tree = cKDTree(coords[idx])
        coo = tree.sparse_distance_matrix(tree, max_distance=5.0, output_type='coo_matrix')
        offdiag = coo.row != coo.col
        if not offdiag.any():
            continue
        r = coo.row[offdiag]
        c = coo.col[offdiag]
        d = coo.data[offdiag]
        m_p = len(idx)
        m015 = d <= 0.15
        if m015.any():
            cnt_r015[idx] = np.bincount(r[m015], minlength=m_p).astype(np.float64)
            hits_r015[idx] = np.bincount(r[m015], weights=n_hits[idx[c[m015]]], minlength=m_p)
        m040 = d <= 0.40
        if m040.any():
            cnt_r040[idx] = np.bincount(r[m040], minlength=m_p).astype(np.float64)
            hits_r040[idx] = np.bincount(r[m040], weights=n_hits[idx[c[m040]]], minlength=m_p)
        r1 = d <= 1.0
        if r1.any():
            cnt_r1[idx] = np.bincount(r[r1], minlength=m_p).astype(np.float64)
        cnt_r5[idx] = np.bincount(r, minlength=m_p).astype(np.float64)
        hits_r5[idx] = np.bincount(r, weights=n_hits[idx[c]], minlength=m_p)
    return cnt_r015, hits_r015, cnt_r040, hits_r040, cnt_r1, cnt_r5, hits_r5


def _rasterize_micro_patches(
    order: np.ndarray,
    offsets: np.ndarray,
    n_hits: np.ndarray,
    pxyz: np.ndarray,
    x_mean: np.ndarray,
    y_mean: np.ndarray,
    sel_local: np.ndarray,
    keep_idx: np.ndarray,
    pitch_x: float,
    pitch_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize up to N_MAX_PATCHES bilinear 32x32 pixel patches + symmetric boundary channel."""
    n_p = len(sel_local)
    if n_p == 0:
        return np.empty((0, 2, PATCH_SIZE, PATCH_SIZE), dtype=np.float16), np.empty(0, dtype=np.int16)
    fast_p = rasterize_patches_fast(
        order, offsets, pxyz, x_mean, y_mean, sel_local, keep_idx, PATCH_SIZE, pitch_x, pitch_y
    )
    if fast_p is not None:
        return fast_p

    patches = np.zeros((n_p, 2, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    half_center = 0.5 * (PATCH_SIZE - 1)  # 15.5 ensures exact Y-mirror symmetry
    hx_ord = pxyz[order, 0].astype(np.float64)
    hy_ord = pxyz[order, 1].astype(np.float64)

    for p_i, loc_i in enumerate(sel_local):
        cl_id = int(keep_idx[loc_i])
        s = int(offsets[cl_id])
        e = int(offsets[cl_id + 1])
        u = half_center + (hx_ord[s:e] - x_mean[cl_id]) / pitch_x
        v = half_center + (hy_ord[s:e] - y_mean[cl_id]) / pitch_y

        inside = (u >= -0.5) & (u <= (PATCH_SIZE - 0.5)) & (v >= -0.5) & (v <= (PATCH_SIZE - 0.5))
        n_out = int((~inside).sum())
        if inside.any():
            ui = np.clip(u[inside], 0.0, PATCH_SIZE - 1.0 - 1e-5)
            vi = np.clip(v[inside], 0.0, PATCH_SIZE - 1.0 - 1e-5)
            u0 = np.floor(ui).astype(np.int64)
            v0 = np.floor(vi).astype(np.int64)
            du = ui - u0
            dv = vi - v0
            u1 = np.minimum(u0 + 1, PATCH_SIZE - 1)
            v1 = np.minimum(v0 + 1, PATCH_SIZE - 1)
            np.add.at(patches[p_i, 0], (v0, u0), (1.0 - dv) * (1.0 - du))
            np.add.at(patches[p_i, 0], (v0, u1), (1.0 - dv) * du)
            np.add.at(patches[p_i, 0], (v1, u0), dv * (1.0 - du))
            np.add.at(patches[p_i, 0], (v1, u1), dv * du)
        if n_out > 0:
            edge_val = math.log1p(n_out) / 5.0
            patches[p_i, 1, :2, :] = edge_val
            patches[p_i, 1, -2:, :] = edge_val
            patches[p_i, 1, :, :2] = edge_val
            patches[p_i, 1, :, -2:] = edge_val

    return patches.astype(np.float16), sel_local.astype(np.int16)


def _cluster_sensor_groups(
    cid_cl: np.ndarray,
    xyz_cl: np.ndarray,
    nx_cl: int,
    ny_cl: int,
    n_target_planes: int,
    r_tgt: float,
    r_ipt: float,
) -> dict:
    """Cluster target planes (< n_target_planes) and IPT planes (>= n_target_planes) independently."""
    planes = cid_cl // (int(nx_cl) * int(ny_cl))
    tgt_idx = np.flatnonzero(planes < int(n_target_planes))
    ipt_idx = np.flatnonzero(planes >= int(n_target_planes))

    if len(ipt_idx) == 0:
        return cluster_pixel_hits(cid_cl, xyz_cl, nx_cl, ny_cl, radius_mm=r_tgt)
    if len(tgt_idx) == 0:
        return cluster_pixel_hits(cid_cl, xyz_cl, nx_cl, ny_cl, radius_mm=r_ipt)

    cl_t = cluster_pixel_hits(cid_cl[tgt_idx], xyz_cl[tgt_idx], nx_cl, ny_cl, radius_mm=r_tgt)
    cl_i = cluster_pixel_hits(cid_cl[ipt_idx], xyz_cl[ipt_idx], nx_cl, ny_cl, radius_mm=r_ipt)

    order_t = tgt_idx[cl_t['hit_order']].astype(np.int32)
    order_i = ipt_idx[cl_i['hit_order']].astype(np.int32)
    offsets = np.concatenate([
        cl_t['cluster_offsets'],
        cl_i['cluster_offsets'][1:] + np.int32(len(order_t)),
    ]).astype(np.int32)
    return {
        'n_clusters': int(cl_t['n_clusters']) + int(cl_i['n_clusters']),
        'hit_order': np.concatenate([order_t, order_i]),
        'cluster_offsets': offsets,
        'cluster_plane': np.concatenate([cl_t['cluster_plane'], cl_i['cluster_plane']]),
        'cluster_first_channel_id': np.concatenate([
            cl_t['cluster_first_channel_id'],
            cl_i['cluster_first_channel_id'],
        ]),
    }


def _select_group_clusters(
    group_idx: np.ndarray,
    cl_plane: np.ndarray,
    n_hits: np.ndarray,
    hits_r040: np.ndarray,
    cnt_r5: np.ndarray,
    max_cap: int,
) -> np.ndarray:
    """Plane-stratified cluster retention strictly within one sensor group (target or IPT)."""
    if len(group_idx) <= max_cap:
        return group_idx
    g_plane = cl_plane[group_idx]
    g_hits = n_hits[group_idx]
    g_score = g_hits + 0.5 * hits_r040[group_idx]
    g_cnt5 = cnt_r5[group_idx]
    uniq_p = np.unique(g_plane)
    p_hits = np.array([g_hits[g_plane == pv].sum() for pv in uniq_p], dtype=np.float64)
    cum_p = np.cumsum(p_hits)
    sp_idx = int(np.searchsorted(cum_p, 0.05 * max(1.0, cum_p[-1]), side='left'))
    start_plane = int(uniq_p[min(sp_idx, len(uniq_p) - 1)])

    plane_rank = np.zeros(len(group_idx), dtype=np.int64)
    for p_val in uniq_p:
        loc = np.flatnonzero(g_plane == p_val)
        ord_p = np.lexsort((g_cnt5[loc], -g_score[loc]))
        plane_rank[loc[ord_p]] = np.arange(len(loc), dtype=np.int64)

    tier1 = np.flatnonzero(plane_rank < 64)
    if len(tier1) >= max_cap:
        ord_t1 = np.lexsort((-g_score[tier1], np.abs(g_plane[tier1] - start_plane), plane_rank[tier1]))
        chosen_loc = np.sort(tier1[ord_t1[:max_cap]])
    else:
        rest = np.flatnonzero(plane_rank >= 64)
        is_far = np.abs(g_plane[rest] - start_plane) > 5
        ord_rest = np.lexsort((g_cnt5[rest], -g_score[rest], is_far))
        need = max_cap - len(tier1)
        chosen_loc = np.sort(np.concatenate([tier1, rest[ord_rest[:need]]]))
    return group_idx[chosen_loc]


def extract_pf_item(
    index: int,
    observation: dict,
    label: int | None,
    dom: dict,
    geometry: dict,
    bounds: dict,
    config: dict,
) -> dict:
    """Extract micron-exact connected pixel clusters, patches, active bars, and spectrometer inputs."""
    cfg = config or {}
    pe_only = bool(cfg.get('pe_only', False))
    validate_finite_observation(observation, check_edep=not pe_only)

    pixel = observation['pixel']
    bar = observation['bar']
    tracker = observation['tracker_hypothesis']

    pcid = np.asarray(pixel['channel_id'], dtype=np.int64)
    pxyz = np.asarray(pixel['xyz_mm'], dtype=np.float64).reshape(-1, 3)

    n_pix_planes = int(bounds['nz'])
    nx_raw, ny_raw = int(bounds['nx']), int(bounds['ny'])
    recompute_dom = False

    active_planes = cfg.get('active_pixel_planes')
    if active_planes is not None and len(pcid) > 0:
        pl_raw = pcid // (nx_raw * ny_raw)
        keep_pl = np.isin(pl_raw, np.asarray(active_planes, dtype=np.int64))
        pcid = pcid[keep_pl]
        pxyz = pxyz[keep_pl]
        recompute_dom = True

    misalign_mm = cfg.get('misalign_sigma_xy_mm')
    if misalign_mm is not None and float(misalign_mm) > 0.0 and len(pcid) > 0:
        pl_raw = pcid // (nx_raw * ny_raw)
        m_rng = np.random.default_rng(int(cfg.get('misalign_seed', 20261118)))
        plane_shifts = m_rng.normal(0.0, float(misalign_mm), size=(n_pix_planes, 2))
        pxyz = pxyz.copy()
        pxyz[:, :2] += plane_shifts[pl_raw]
        recompute_dom = True

    if recompute_dom:
        obs_mod = dict(observation)
        obs_mod['pixel'] = {'channel_id': pcid, 'xyz_mm': pxyz.astype(np.float32)}
        dom = compute_observable_containment(obs_mod, geometry, bounds, config=cfg)

    raw_pixel_hits = len(pcid)

    x_c = float(bounds['x_pix_center'])
    y_c = float(bounds['y_scint_center'])
    z_mid = float(bounds['z_mid'])
    pix_zpos = np.asarray(bounds['pixel_Zpos'], dtype=np.float64)
    pix_x0 = np.array(
        [cumulative_radiation_length(float(z), geometry) for z in pix_zpos],
        dtype=np.float32,
    )

    n_target_planes = int(geometry['nPinpointBlocks']) + max(0, int(geometry['nFortuneBlocks']) - 1)
    sp = resolve_sensor_pitches(bounds, cfg)
    q_tgt = sp['q_tgt']
    q_ipt = sp['q_ipt']
    px_tgt, py_tgt, r_tgt = float(sp['px_tgt']), float(sp['py_tgt']), float(sp['r_tgt'])
    px_ipt, py_ipt, r_ipt = float(sp['px_ipt']), float(sp['py_ipt']), float(sp['r_ipt'])

    if (q_tgt is not None or q_ipt is not None) and raw_pixel_hits > 0:
        cid_cl, xyz_cl, nx_cl, ny_cl, _, _, uniq = quantize_pixel_hits(
            pcid,
            pxyz,
            bounds,
            n_target_planes=n_target_planes,
            quantize_target_xy_mm=q_tgt,
            quantize_ipt_xy_mm=q_ipt,
        )
        obs_q = dict(observation)
        obs_q['pixel'] = {'channel_id': pcid[uniq], 'xyz_mm': xyz_cl.astype(np.float32)}
        dom = compute_observable_containment(obs_q, geometry, bounds, config=cfg)
        eff_pixel_hits = len(cid_cl)
    else:
        cid_cl, xyz_cl = pcid, pxyz
        nx_cl, ny_cl = int(bounds['nx']), int(bounds['ny'])
        eff_pixel_hits = raw_pixel_hits

    plane_ids = np.arange(n_pix_planes, dtype=np.int64)
    px_plane = np.where(plane_ids < n_target_planes, px_tgt, px_ipt).astype(np.float64)
    py_plane = np.where(plane_ids < n_target_planes, py_tgt, py_ipt).astype(np.float64)

    plane_summary = np.zeros((n_pix_planes, PIX_PLANE_SUMMARY_DIM), dtype=np.float32)
    plane_summary[:, 5] = pix_x0 / 50.0

    if eff_pixel_hits > 0:
        cl = _cluster_sensor_groups(cid_cl, xyz_cl, nx_cl, ny_cl, n_target_planes, r_tgt, r_ipt)
        n_cl = int(cl['n_clusters'])
        order = cl['hit_order']
        offsets = cl['cluster_offsets']
        starts = offsets[:-1]
        n_hits = np.diff(offsets).astype(np.float64)
        cl_plane = cl['cluster_plane'].astype(np.int64)
        cl_fcid = cl['cluster_first_channel_id'].astype(np.uint64)

        px_cl = px_plane[cl_plane]
        py_cl = py_plane[cl_plane]

        hx = xyz_cl[order, 0].astype(np.float64)
        hy = xyz_cl[order, 1].astype(np.float64)
        x_mean = np.add.reduceat(hx, starts) / n_hits
        y_mean = np.add.reduceat(hy, starts) / n_hits
        z_cl = pix_zpos[cl_plane]

        (
            cnt_r015,
            hits_r015,
            cnt_r040,
            hits_r040,
            cnt_r1,
            cnt_r5,
            hits_r5,
        ) = _compute_plane_neighborhoods(cl_plane, x_mean, y_mean, n_hits)

        # Whole-event per-plane summary BEFORE cluster cap (per-sensor pitch preserves IPT bit-exactness)
        p_hit_cnt = np.bincount(cl_plane, weights=n_hits, minlength=n_pix_planes)
        p_cl_cnt = np.bincount(cl_plane, minlength=n_pix_planes)
        p_xsum = np.bincount(cl_plane, weights=x_mean * n_hits, minlength=n_pix_planes)
        p_ysum = np.bincount(cl_plane, weights=y_mean * n_hits, minlength=n_pix_planes)
        p_act = p_hit_cnt > 0

        w_cl = np.square(n_hits + hits_r040)
        p_wsum = np.bincount(cl_plane, weights=w_cl, minlength=n_pix_planes)
        plane_xc = np.where(
            p_act,
            np.bincount(cl_plane, weights=w_cl * x_mean, minlength=n_pix_planes) / np.maximum(1e-12, p_wsum),
            x_c,
        )
        plane_yc = np.where(
            p_act,
            np.bincount(cl_plane, weights=w_cl * y_mean, minlength=n_pix_planes) / np.maximum(1e-12, p_wsum),
            y_c,
        )

        n_rep = n_hits.astype(np.int64)
        hit_pl = np.repeat(cl_plane, n_rep)
        dx_hp = hx - plane_xc[hit_pl]
        dy_hp = hy - plane_yc[hit_pl]
        dr_hp = np.hypot(dx_hp, dy_hp)
        p_h015 = np.bincount(hit_pl, weights=(dr_hp <= 0.15).astype(np.float64), minlength=n_pix_planes)
        p_h040 = np.bincount(hit_pl, weights=(dr_hp <= 0.40).astype(np.float64), minlength=n_pix_planes)
        p_h200 = np.bincount(hit_pl, weights=(dr_hp <= 2.00).astype(np.float64), minlength=n_pix_planes)

        core25 = dr_hp <= 2.50
        h_c25 = np.bincount(hit_pl[core25], minlength=n_pix_planes).astype(np.float64)
        p_cmx = np.bincount(hit_pl[core25], weights=dx_hp[core25], minlength=n_pix_planes) / np.maximum(1.0, h_c25)
        p_cmy = np.bincount(hit_pl[core25], weights=dy_hp[core25], minlength=n_pix_planes) / np.maximum(1.0, h_c25)
        p_cxx = np.bincount(hit_pl[core25], weights=dx_hp[core25] * dx_hp[core25], minlength=n_pix_planes)
        p_cyy = np.bincount(hit_pl[core25], weights=dy_hp[core25] * dy_hp[core25], minlength=n_pix_planes)
        p_cxy = np.bincount(hit_pl[core25], weights=dx_hp[core25] * dy_hp[core25], minlength=n_pix_planes)
        cvxx = np.maximum(0.0, p_cxx / np.maximum(1.0, h_c25) - p_cmx * p_cmx) + (px_plane ** 2) / 12.0
        cvyy = np.maximum(0.0, p_cyy / np.maximum(1.0, h_c25) - p_cmy * p_cmy) + (py_plane ** 2) / 12.0
        cvxy = p_cxy / np.maximum(1.0, h_c25) - p_cmx * p_cmy
        c_tr = cvxx + cvyy
        c_df = cvxx - cvyy
        c_disc = np.sqrt(np.maximum(0.0, c_df * c_df + 4.0 * cvxy * cvxy))
        c_l1 = np.maximum(1e-12, 0.5 * (c_tr + c_disc))
        c_l2 = np.maximum(1e-12, 0.5 * (c_tr - c_disc))
        core_ecc = np.where(h_c25 >= 2.0, np.sqrt(np.clip(1.0 - c_l2 / c_l1, 0.0, 1.0)), 0.0)

        plane_summary[p_act, 0] = 1.0
        plane_summary[p_act, 1] = np.log1p(p_hit_cnt[p_act]) / 8.0
        plane_summary[p_act, 2] = np.log1p(p_cl_cnt[p_act]) / 6.0
        plane_summary[p_act, 3] = (p_xsum[p_act] / p_hit_cnt[p_act] - x_c) / 135.0
        plane_summary[p_act, 4] = (p_ysum[p_act] / p_hit_cnt[p_act] - y_c) / 100.0
        plane_summary[p_act, 6] = (plane_yc[p_act] - y_c) / 100.0
        plane_summary[p_act, 7] = (plane_xc[p_act] - x_c) / 135.0
        plane_summary[p_act, 8] = np.log1p(p_h015[p_act]) / 6.0
        plane_summary[p_act, 9] = np.log1p(p_h040[p_act]) / 7.0
        plane_summary[p_act, 10] = p_h040[p_act] / np.maximum(1.0, p_h200[p_act])
        plane_summary[p_act, 11] = core_ecc[p_act]

        dx_raw = np.maximum.reduceat(hx, starts) - np.minimum.reduceat(hx, starts)
        dy_raw = np.maximum.reduceat(hy, starts) - np.minimum.reduceat(hy, starts)
        dx_box = dx_raw + px_cl
        dy_box = dy_raw + py_cl
        fill_factor = np.clip(n_hits / np.maximum(1.0, (dx_box / px_cl) * (dy_box / py_cl)), 0.0, 1.0)

        dhx = hx - np.repeat(x_mean, n_rep)
        dhy = hy - np.repeat(y_mean, n_rep)
        var_x = np.add.reduceat(dhx * dhx, starts) / n_hits + (px_cl ** 2) / 12.0
        var_y = np.add.reduceat(dhy * dhy, starts) / n_hits + (py_cl ** 2) / 12.0
        cov_xy = np.add.reduceat(dhx * dhy, starts) / n_hits

        trace = var_x + var_y
        diff_v = var_x - var_y
        disc = np.sqrt(np.maximum(0.0, diff_v * diff_v + 4.0 * cov_xy * cov_xy))
        lam1 = np.maximum(1e-12, 0.5 * (trace + disc))
        lam2 = np.maximum(1e-12, 0.5 * (trace - disc))
        ecc = np.sqrt(np.clip(1.0 - lam2 / lam1, 0.0, 1.0))
        cos2a = np.where(disc > 1e-12, diff_v / np.maximum(disc, 1e-12), 0.0)
        sin2a = np.where(disc > 1e-12, (2.0 * cov_xy) / np.maximum(disc, 1e-12), 0.0)

        skew_x = (np.add.reduceat(dhx * dhx * dhx, starts) / n_hits) / np.power(var_x, 1.5)
        skew_y = (np.add.reduceat(dhy * dhy * dhy, starts) / n_hits) / np.power(var_y, 1.5)
        skew_norm = np.clip(np.hypot(skew_x, skew_y), 0.0, 5.0) / 5.0

        # Strictly independent target vs IPT token retention budgets (832 target + 192 IPT = 1024)
        # so altering target pitch never adds or drops any IPT token (Stage 322-03 fix).
        tgt_cl_all = np.flatnonzero(cl_plane < n_target_planes)
        ipt_cl_all = np.flatnonzero(cl_plane >= n_target_planes)
        keep_tgt = _select_group_clusters(tgt_cl_all, cl_plane, n_hits, hits_r040, cnt_r5, N_MAX_TARGET_CLUSTERS)
        keep_ipt = _select_group_clusters(ipt_cl_all, cl_plane, n_hits, hits_r040, cnt_r5, N_MAX_IPT_CLUSTERS)
        keep_idx = np.concatenate([keep_tgt, keep_ipt])

        pix_xyz = np.column_stack([
            x_mean[keep_idx],
            y_mean[keep_idx],
            z_cl[keep_idx],
        ]).astype(np.float32)

        dx_core = x_mean[keep_idx] - plane_xc[cl_plane[keep_idx]]
        dy_core = y_mean[keep_idx] - plane_yc[cl_plane[keep_idx]]
        r_core = np.hypot(dx_core, dy_core)

        pix_feat = np.column_stack([
            (x_mean[keep_idx] - x_c) / 135.0,
            (y_mean[keep_idx] - y_c) / 100.0,
            (z_cl[keep_idx] - z_mid) / 1000.0,
            np.log1p(n_hits[keep_idx]) / 6.0,
            np.arcsinh(dx_box[keep_idx] / 0.1) / 3.0,
            np.arcsinh(dy_box[keep_idx] / 0.1) / 3.0,
            fill_factor[keep_idx],
            np.arcsinh(np.sqrt(lam1[keep_idx]) / 0.05) / 3.0,
            np.arcsinh(np.sqrt(lam2[keep_idx]) / 0.05) / 3.0,
            ecc[keep_idx],
            cos2a[keep_idx],
            sin2a[keep_idx],
            skew_norm[keep_idx],
            np.log1p(cnt_r1[keep_idx]) / 4.0,
            np.log1p(cnt_r5[keep_idx]) / 5.0,
            np.log1p(hits_r5[keep_idx]) / 7.0,
            np.arcsinh(dx_core / 0.10) / 4.0,
            np.arcsinh(dy_core / 0.10) / 4.0,
            np.arcsinh(r_core / 0.08) / 4.0,
            np.log1p(cnt_r015[keep_idx]) / 3.0,
            np.log1p(hits_r015[keep_idx]) / 5.0,
            np.log1p(cnt_r040[keep_idx]) / 4.0,
        ]).astype(np.float32)
        pix_plane = cl_plane[keep_idx].astype(np.int8)
        pix_first_cid = cl_fcid[keep_idx]

        # Independent target vs IPT micro-patch selection & per-sensor-pitch rasterization (48 + 16 = 64)
        trig = (
            (n_hits[keep_idx] >= PATCH_TRIGGER_MIN_HITS)
            & (np.maximum(dx_raw[keep_idx], dy_raw[keep_idx]) > PATCH_TRIGGER_MIN_SPAN_MM)
        )
        trig_tgt = np.flatnonzero(trig & (pix_plane < n_target_planes))
        if len(trig_tgt) > N_MAX_TARGET_PATCHES:
            rank_t = np.argsort(-n_hits[keep_idx[trig_tgt]], kind='mergesort')
            trig_tgt = np.sort(trig_tgt[rank_t[:N_MAX_TARGET_PATCHES]])
        trig_ipt = np.flatnonzero(trig & (pix_plane >= n_target_planes))
        if len(trig_ipt) > N_MAX_IPT_PATCHES:
            rank_i = np.argsort(-n_hits[keep_idx[trig_ipt]], kind='mergesort')
            trig_ipt = np.sort(trig_ipt[rank_i[:N_MAX_IPT_PATCHES]])

        p_t, idx_t = _rasterize_micro_patches(
            order, offsets, n_hits, xyz_cl, x_mean, y_mean, trig_tgt, keep_idx, px_tgt, py_tgt
        )
        p_i, idx_i = _rasterize_micro_patches(
            order, offsets, n_hits, xyz_cl, x_mean, y_mean, trig_ipt, keep_idx, px_ipt, py_ipt
        )
        if len(idx_t) and len(idx_i):
            pix_patches = np.concatenate([p_t, p_i], axis=0)
            patch_cl_idx = np.concatenate([idx_t, idx_i], axis=0)
        elif len(idx_t):
            pix_patches, patch_cl_idx = p_t, idx_t
        else:
            pix_patches, patch_cl_idx = p_i, idx_i

        mv_x = x_mean
        mv_y = y_mean
        if bool(cfg.get('shuffle_micro_vertex', False)) and len(cl_plane) >= 2:
            # Stage 320/322 Exp-1 falsification control: content-derived seed ensures invariance to dataset row order
            base_seed = int(cfg.get('shuffle_seed', 2026321))
            content_sig = int(cl_fcid[0]) ^ (int(len(cl_plane)) * 2654435761)
            sh_rng = np.random.default_rng([base_seed & 0xFFFFFFFF, content_sig & 0xFFFFFFFF, (content_sig >> 32) & 0xFFFFFFFF])
            mv_x = x_mean.copy()
            mv_y = y_mean.copy()
            for p_val in np.unique(cl_plane):
                pm = np.flatnonzero(cl_plane == p_val)
                if len(pm) >= 2:
                    ang = float(sh_rng.uniform(0.5, 2.0 * math.pi - 0.5))
                    ca, sa = math.cos(ang), math.sin(ang)
                    cx_p, cy_p = float(plane_xc[int(p_val)]), float(plane_yc[int(p_val)])
                    dx_p = mv_x[pm] - cx_p
                    dy_p = mv_y[pm] - cy_p
                    mv_x[pm] = cx_p + ca * dx_p - sa * dy_p
                    mv_y[pm] = cy_p + sa * dx_p + ca * dy_p
        has_tgt_cl = len(tgt_cl_all) > 0
        mv_px = px_tgt if has_tgt_cl else px_ipt
        mv_py = py_tgt if has_tgt_cl else py_ipt
        avr_xyz, avr_ok, micro_vtx_feat = reconstruct_micron_vertex(
            cl_plane, mv_x, mv_y, z_cl, n_hits, lam2,
            n_pix_planes, mv_px, mv_py, dom['cal_start'], (x_c, y_c, z_mid),
        )
    else:
        pix_xyz = np.empty((0, 3), dtype=np.float32)
        pix_feat = np.empty((0, PIX_FEAT_DIM), dtype=np.float32)
        pix_plane = np.empty(0, dtype=np.int8)
        pix_first_cid = np.empty(0, dtype=np.uint64)
        pix_patches = np.empty((0, 2, PATCH_SIZE, PATCH_SIZE), dtype=np.float16)
        patch_cl_idx = np.empty(0, dtype=np.int16)
        sz0, sx0, sy0, sv0 = dom['cal_start']
        avr_xyz = np.array([sx0, sy0, sz0] if sv0 else [x_c, y_c, z_mid], dtype=np.float32)
        avr_ok = False
        micro_vtx_feat = np.zeros(MICRO_VTX_FEAT_DIM, dtype=np.float32)

    # Calibrated bar extraction: when config['pe_only']=True, edep_proxy_MeV is never read (Audit P1-1)
    keep_bar, be, bpe = calibrated_bar_channels(bar, cfg)
    bp = np.asarray(bar['panel'], dtype=np.int16)[keep_bar]
    bh = np.asarray(bar['horizontal'], dtype=np.int8)[keep_bar]
    bi = np.asarray(bar['bar'], dtype=np.int16)[keep_bar]
    if len(bp) > 0:
        bar_val = np.column_stack([
            np.log1p(np.maximum(be, 0.0)) / 10.0,
            np.log1p(np.maximum(bpe, 0.0)) / 15.0,
        ]).astype(np.float16)
    else:
        bar_val = np.empty((0, 2), dtype=np.float16)

    spec = extract_spectrometer_kinematics(tracker, dom['cal_start'], bounds)
    sz, sx, sy, svalid = dom['cal_start']
    avr_pv = np.array([avr_xyz[0], avr_xyz[1], avr_xyz[2], 1.0 if avr_ok else 0.0], dtype=np.float32)

    out_item = {
        'index': int(index),
        'obs_domain_id': int(dom['obs_domain_id']),
        'c_obs': dom['c_obs'].astype(np.float32),
        'cal_start': np.array([sz, sx, sy, 1.0 if svalid else 0.0], dtype=np.float32),
        'avr_pv_mm': avr_pv,
        'micro_vtx_feat': micro_vtx_feat,
        'raw_pixel_hits': int(raw_pixel_hits),
        'pix_xyz_mm': pix_xyz,
        'pix_feat': pix_feat,
        'pix_plane': pix_plane,
        'pix_first_cid': pix_first_cid,
        'pix_plane_summary': plane_summary,
        'pix_patches': pix_patches,
        'patch_cluster_idx': patch_cl_idx,
        'bar_panel': bp,
        'bar_horiz': bh,
        'bar_idx': bi,
        'bar_val': bar_val,
        'trk_cand': spec['cand'],
        'trk_cand_mask': spec['cand_mask'],
        'trk_hits': spec['hits'],
        'bar_keep_mask': keep_bar,
    }
    if label is not None:
        out_item['label'] = int(label)
    return out_item


def apply_y_mirror_pf_item(
    item: dict,
    y_scint_center: float = -12.0,
    n_bars: int = 40,
) -> dict:
    """Apply exact detector-centered Y-reflection (y -> 2*y_c - y) to one PinpointFormer item."""
    out = dict(item)
    yc = float(y_scint_center)

    if len(item['pix_xyz_mm']):
        xyz = item['pix_xyz_mm'].copy()
        xyz[:, 1] = np.float32(2.0 * yc) - xyz[:, 1]
        out['pix_xyz_mm'] = xyz

        pf = item['pix_feat'].astype(np.float32, copy=True)
        pf[:, PIX_Y_FLIP_IDX] *= np.float32(-1.0)
        out['pix_feat'] = pf

    ps = item['pix_plane_summary'].copy()
    ps[:, list(PLANE_SUMMARY_Y_FLIP_IDX)] *= np.float32(-1.0)
    out['pix_plane_summary'] = ps

    if len(item['pix_patches']):
        out['pix_patches'] = item['pix_patches'][:, :, ::-1, :].copy()

    cs = item['cal_start'].copy()
    if cs[3] > 0.5:
        cs[2] = np.float32(2.0 * yc) - cs[2]
    out['cal_start'] = cs

    if 'avr_pv_mm' in item:
        apv = item['avr_pv_mm'].copy()
        apv[1] = np.float32(2.0 * yc) - apv[1]
        out['avr_pv_mm'] = apv

    if len(item['bar_idx']):
        bi = item['bar_idx'].copy()
        hmask = item['bar_horiz'] == 1
        if hmask.any():
            bi[hmask] = np.int16(n_bars - 1) - bi[hmask]
        out['bar_idx'] = bi

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

    if 'vertex_xyz_mm' in item:
        vtx = item['vertex_xyz_mm'].copy()
        if int(item.get('vertex_valid', 0)) > 0:
            vtx[1] = np.float32(2.0 * yc) - vtx[1]
        out['vertex_xyz_mm'] = vtx
    if 'label_7' in item:
        l7 = int(item['label_7'])
        if l7 == 1:
            out['label_7'] = 2
        elif l7 == 2:
            out['label_7'] = 1
    if 'vis_kinematics' in item:
        vk = item['vis_kinematics'].copy()
        if int(item.get('kin_valid', 0)) > 0:
            vk[3] *= np.float32(-1.0)
        out['vis_kinematics'] = vk
    return out


def collate_pf(
    events: list[dict],
    mirror_rng: np.random.Generator | None = None,
    y_scint_center: float = -12.0,
    n_bars: int = 40,
) -> dict:
    """Collate variable-length PinpointFormer events into padded/sparse GPU tensors."""
    if mirror_rng is not None:
        do_flip = mirror_rng.random(len(events)) < 0.5
        events = [
            apply_y_mirror_pf_item(e, y_scint_center=y_scint_center, n_bars=n_bars) if f else e
            for e, f in zip(events, do_flip)
        ]

    b = len(events)
    max_cl = max(1, max(len(e['pix_xyz_mm']) for e in events))
    pix_xyz = np.zeros((b, max_cl, 3), dtype=np.float32)
    pix_feat = np.zeros((b, max_cl, PIX_FEAT_DIM), dtype=np.float32)
    pix_plane = np.zeros((b, max_cl), dtype=np.int64)
    pix_mask = np.zeros((b, max_cl), dtype=bool)

    max_hits = max(1, max(len(e['trk_hits']) for e in events))
    trk_hits = np.zeros((b, max_hits, 6), dtype=np.float32)
    trk_hit_mask = np.zeros((b, max_hits), dtype=bool)

    patch_list: list[np.ndarray] = []
    patch_ev_list: list[np.ndarray] = []
    patch_cl_list: list[np.ndarray] = []

    bar_ev_list: list[np.ndarray] = []
    bar_panel_list: list[np.ndarray] = []
    bar_horiz_list: list[np.ndarray] = []
    bar_idx_list: list[np.ndarray] = []
    bar_val_list: list[np.ndarray] = []

    for i, e in enumerate(events):
        nc = len(e['pix_xyz_mm'])
        if nc:
            pix_xyz[i, :nc] = e['pix_xyz_mm']
            pix_feat[i, :nc] = e['pix_feat'].astype(np.float32)
            pix_plane[i, :nc] = e['pix_plane']
            pix_mask[i, :nc] = True

        np_i = len(e['pix_patches'])
        if np_i:
            patch_list.append(e['pix_patches'].astype(np.float32))
            patch_ev_list.append(np.full(np_i, i, dtype=np.int64))
            patch_cl_list.append(e['patch_cluster_idx'].astype(np.int64))

        nh = len(e['trk_hits'])
        if nh:
            trk_hits[i, :nh] = e['trk_hits']
            trk_hit_mask[i, :nh] = True

        nb = len(e['bar_panel'])
        if nb:
            bar_ev_list.append(np.full(nb, i, dtype=np.int64))
            bar_panel_list.append(e['bar_panel'].astype(np.int64))
            bar_horiz_list.append(e['bar_horiz'].astype(np.int64))
            bar_idx_list.append(e['bar_idx'].astype(np.int64))
            bar_val_list.append(e['bar_val'].astype(np.float32))

    if patch_list:
        patches_t = torch.from_numpy(np.concatenate(patch_list, axis=0))
        patch_ev_t = torch.from_numpy(np.concatenate(patch_ev_list, axis=0))
        patch_cl_t = torch.from_numpy(np.concatenate(patch_cl_list, axis=0))
    else:
        patches_t = torch.empty((0, 2, PATCH_SIZE, PATCH_SIZE), dtype=torch.float32)
        patch_ev_t = torch.empty(0, dtype=torch.long)
        patch_cl_t = torch.empty(0, dtype=torch.long)

    if bar_ev_list:
        bar_ev_t = torch.from_numpy(np.concatenate(bar_ev_list, axis=0))
        bar_panel_t = torch.from_numpy(np.concatenate(bar_panel_list, axis=0))
        bar_horiz_t = torch.from_numpy(np.concatenate(bar_horiz_list, axis=0))
        bar_idx_t = torch.from_numpy(np.concatenate(bar_idx_list, axis=0))
        bar_val_t = torch.from_numpy(np.concatenate(bar_val_list, axis=0))
    else:
        bar_ev_t = torch.empty(0, dtype=torch.long)
        bar_panel_t = torch.empty(0, dtype=torch.long)
        bar_horiz_t = torch.empty(0, dtype=torch.long)
        bar_idx_t = torch.empty(0, dtype=torch.long)
        bar_val_t = torch.empty((0, 2), dtype=torch.float32)

    batch = {
        'index': torch.tensor([e['index'] for e in events], dtype=torch.long),
        'obs_domain_id': torch.tensor([e['obs_domain_id'] for e in events], dtype=torch.long),
        'c_obs': torch.from_numpy(np.stack([e['c_obs'] for e in events])),
        'cal_start': torch.from_numpy(np.stack([e['cal_start'] for e in events])),
        'raw_pixel_hits': torch.tensor([e['raw_pixel_hits'] for e in events], dtype=torch.long),
        'pix_xyz_mm': torch.from_numpy(pix_xyz),
        'pix_feat': torch.from_numpy(pix_feat),
        'pix_plane': torch.from_numpy(pix_plane),
        'pix_mask': torch.from_numpy(pix_mask),
        'pix_plane_summary': torch.from_numpy(np.stack([e['pix_plane_summary'] for e in events])),
        'pix_patches': patches_t,
        'patch_ev': patch_ev_t,
        'patch_cl': patch_cl_t,
        'bar_ev': bar_ev_t,
        'bar_panel': bar_panel_t,
        'bar_horiz': bar_horiz_t,
        'bar_idx': bar_idx_t,
        'bar_val': bar_val_t,
        'trk_cand': torch.from_numpy(np.stack([e['trk_cand'] for e in events])),
        'trk_cand_mask': torch.from_numpy(np.stack([e['trk_cand_mask'] for e in events])),
        'trk_hits': torch.from_numpy(trk_hits),
        'trk_hit_mask': torch.from_numpy(trk_hit_mask),
    }
    if 'label' in events[0]:
        batch['label'] = torch.tensor([e['label'] for e in events], dtype=torch.long)

    if 'avr_pv_mm' in events[0]:
        batch['avr_pv_mm'] = torch.from_numpy(np.stack([e['avr_pv_mm'] for e in events]).astype(np.float32))
    if 'micro_vtx_feat' in events[0]:
        batch['micro_vtx_feat'] = torch.from_numpy(np.stack([e['micro_vtx_feat'] for e in events]).astype(np.float32))

    for key, dtype in (
        ('sample_weight', torch.float32),
        ('label_7', torch.long),
        ('tau_branch_valid', torch.long),
        ('charm_flag', torch.long),
        ('charm_valid', torch.long),
        ('vertex_valid', torch.long),
        ('kin_valid', torch.long),
    ):
        if key in events[0]:
            batch[key] = torch.tensor([e[key] for e in events], dtype=dtype)
    for key in ('vertex_xyz_mm', 'vis_kinematics'):
        if key in events[0]:
            batch[key] = torch.from_numpy(np.stack([e[key] for e in events]).astype(np.float32))

    # Optional V3 cluster/bar ragged sidecar targets aligned to pix_first_cid and bar order
    if 'sc_first_cid' in events[0]:
        orig_arr = np.full((b, max_cl), -1, dtype=np.int64)
        anc_arr = np.full((b, max_cl, 2), -1, dtype=np.int64)
        pix_rel_arr = np.zeros((b, max_cl, 3), dtype=np.float32)
        pix_rel_valid = np.zeros((b, max_cl), dtype=bool)
        bar_rel_chunks: list[np.ndarray] = []
        for i, e in enumerate(events):
            nc = len(e['pix_first_cid'])
            sc_cid = np.asarray(e['sc_first_cid'], dtype=np.uint64)
            if nc and len(sc_cid):
                pos = np.searchsorted(sc_cid, e['pix_first_cid'])
                valid = (pos < len(sc_cid))
                safe_pos = np.minimum(pos, max(0, len(sc_cid) - 1))
                valid = valid & (sc_cid[safe_pos] == e['pix_first_cid'])
                if valid.any():
                    v_loc = np.flatnonzero(valid)
                    m_pos = safe_pos[v_loc]
                    orig_arr[i, v_loc] = np.asarray(e['sc_origin'][m_pos], dtype=np.int64)
                    anc_arr[i, v_loc] = np.asarray(e['sc_ancestry'][m_pos], dtype=np.int64)
                    pix_rel_arr[i, v_loc] = np.asarray(e['sc_pix_rel'][m_pos], dtype=np.float32)
                    pix_rel_valid[i, v_loc] = True
            if len(e['sc_bar_rel']):
                sc_br = np.asarray(e['sc_bar_rel'], dtype=np.float32)
                if 'bar_keep_mask' in e:
                    sc_br = sc_br[np.asarray(e['bar_keep_mask'], dtype=bool)]
                if len(sc_br):
                    bar_rel_chunks.append(sc_br)
        batch['pix_origin_target'] = torch.from_numpy(orig_arr)
        batch['pix_ancestry_target'] = torch.from_numpy(anc_arr)
        batch['pix_rel_target'] = torch.from_numpy(pix_rel_arr)
        batch['pix_rel_valid'] = torch.from_numpy(pix_rel_valid)
        batch['bar_rel_target'] = (
            torch.from_numpy(np.concatenate(bar_rel_chunks, axis=0))
            if bar_rel_chunks
            else torch.empty((0, 3), dtype=torch.float32)
        )

    return batch


class MicroPatchConv2D(nn.Module):
    """Lightweight 3-layer 2D ConvNet (~35k params) encoding 32x32 pixel cluster morphology."""

    def __init__(self, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if len(patches) == 0:
            return patches.new_empty((0, self.head[0].out_features))
        feat = self.net(patches)
        avg_p = feat.mean(dim=(2, 3))
        max_p = feat.amax(dim=(2, 3))
        return self.head(torch.cat([avg_p, max_p], dim=-1))
