"""Zero/low-parameter inverse physics frontend for Pinpoint-Gen4:
  1. Pre-truncation `DarkShield` hot-pixel/dark-noise filter + 32-dim cluster features
     (22-dim `PIX_FEAT_DIM` + signed principal skewness gamma_{1,||}, 4th-order excess
     kurtosis kappa_||, bimodal split B_split, neck ratio eta_neck, persistence g_persist,
     rotated 2x2 measurement covariance [Sigma_xx, Sigma_yy, Sigma_xy], and Track-B slots)
     plus bilinear 32x32 micro-patches (`uint8` quantized for compact cache).
  2. 54-station inverse digitization unlocking <2.5 PE halo bars & `bar.hit_time_ns`,
     Renyi 3rd-PE threshold time-walk subtraction, Cauchy-Moliere flank desaturation,
     and sub-mm continuous bar cluster centroid estimation (10-channel `scint_grid` contract).
  3. Four-modal differentiable broken-line Kalman muon spectrometer `UnifiedMuonKalman`
     (M1: IPT+M0+3-station bore; M2: IPT+M0+Trk1/2 221.1 GeV*mm soft-muon arm;
      M3: Sm2Co17 shell-scatter |p| fit; M4: Stopping CSDA range + side-exiting Highland MCS)
     exporting `(4, 16)` `muon_kalman_tracks` & `(4,)` `muon_kalman_mask`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn

from analysis.flavour.gen4.geometry_contract import (
    BC408_ATTENUATION_MM,
    BC408_TAU_SCINT_NS,
    COMMON_Y_AXIS_MM,
    IPT_Z_REF_MM,
    KAPPA_B_CONST_MEV_MM,
    MAGNET0_BEND_CONST_MEV_MM,
    MAGNET0_BEND_INTEGRAL_MM2,
    MAGNET0_LENGTH_MM,
    MAGNET0_Z_CENTER_MM,
    MAGNET0_Z_MAX_MM,
    MAGNET0_Z_MIN_MM,
    MAGNET1_Z_MAX_MM,
    SIPM_ETA0_PE_PER_MEV,
    SIPM_NPIX,
    SPEED_OF_LIGHT_MM_PER_NS,
    WLS_GROUP_VELOCITY_MM_PER_NS,
    Gen4GeometryContract,
)
from analysis.flavour.nextgen.dual_regime_modalnet import extract_nonexclusive_spec_candidates
from analysis.flavour.v3.common.domains import (
    compute_observable_containment,
    validate_finite_observation,
)
from analysis.flavour.v3.common.fast_kernels import (
    cluster_pixel_hits_fast,
    plane_neighborhoods_fast,
    rasterize_patches_fast,
)
from analysis.flavour.v3.common.pixel_clusters import cluster_pixel_hits
from analysis.flavour.v3.v1_b1pp.spectrometer import (
    ENTRY_SLOPE_SAGITTA_COEFF,
    MAGNET_BORE_FID_R_MM,
    MAGNET_BORE_X_MM,
    MAGNET_BORE_Y_MM,
    SAGITTA_CONST_MEV_MM,
    SAGITTA_INTEGRAL_MM2,
    SIGMA_DELTA_X_MM,
    SIGMA_SAGITTA_Y_MM,
    SIGMA_X_MM,
    SIGMA_Y_MM,
    TRACKER_Z_MM,
    back_extrapolate_y_through_magnet0,
    check_magnet12_bore_traversal,
)


N_MAX_CLUSTERS_GEN4 = 2304
N_MAX_TARGET_CLUSTERS_GEN4 = 2048
N_MAX_IPT_CLUSTERS_GEN4 = 256
N_MAX_PATCHES_GEN4 = 64
N_MAX_TARGET_PATCHES_GEN4 = 48
N_MAX_IPT_PATCHES_GEN4 = 16
PATCH_SIZE = 32
PATCH_TRIGGER_MIN_HITS = 3
PATCH_TRIGGER_MIN_SPAN_MM = 0.04
PIX_FEAT_GEN4_DIM = 36
PIX_PLANE_SUMMARY_DIM = 12
SCINT_GRID_CHANNELS = 10
MUON_TRACK_SLOTS = 4
MUON_TRACK_DIM = 16
CAL_TIMING_DIM = 6
RM_EFF_BC408_W_MM = 13.5  # Effective Moliere width (W 9.3mm + 5mm BC408)
CSDA_MEV_PER_X0_W = 6.5   # ~1.85 MeV/mm * 3.504 mm

# 10085-10088 silicon pixel & scintillator digitization constants ([gen4-contract-v2])
PIXEL_MIP_KEV = 3.09
PIXEL_EV_PER_ELECTRON = 3.6e-6  # MeV per e-h pair
PIXEL_ENC_MEV = 10.0 * PIXEL_EV_PER_ELECTRON   # 36 eV = 3.6e-5 MeV
PIXEL_THR_MEV = 100.0 * PIXEL_EV_PER_ELECTRON  # 0.36 keV = 3.6e-4 MeV
TRACKER_MERGE_DIST_MM = 0.1
TRACKER_HIT_EFF = 0.99
TRACKER_X_MIN_MM = -125.0
TRACKER_X_MAX_MM = 125.0
TRACKER_Y_MIN_MM = -110.0
TRACKER_Y_MAX_MM = 140.0


def _sigmoid_np(x: np.ndarray | float) -> np.ndarray | float:
    x_c = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x_c))


def _highland_sigma_rad(dx0: float, p_mev: float) -> float:
    x0 = max(float(dx0), 1e-4)
    p = max(float(p_mev), 120.0)
    return (13.6 / p) * math.sqrt(x0) * (1.0 + 0.038 * math.log(max(x0, 1.0)))


# =============================================================================
# 0. Leak-Free 10085-10088 Raw ROOT Digitization -> Canonical Observation Dict
# =============================================================================

def digitize_raw_pixel_hits(
    row_id: np.ndarray,
    col_id: np.ndarray,
    layer_id: np.ndarray,
    edep_mev: np.ndarray,
    contract: Gen4GeometryContract,
    *,
    event_seed: int,
    apply_noise_thr: bool = True,
    dark_rate: float = 0.0,
) -> dict[str, np.ndarray]:
    """Digitize raw 10085-10088 pixelHits with 10 e- ENC noise & 100 e- threshold (zero truth)."""
    r = np.asarray(row_id, dtype=np.int64)
    c = np.asarray(col_id, dtype=np.int64)
    l = np.asarray(layer_id, dtype=np.int64)
    e = np.asarray(edep_mev, dtype=np.float64)
    rng = np.random.default_rng(int(event_seed))

    if len(r) > 0:
        valid = (l >= 0) & (l < contract.nz) & (r >= 0) & (r < contract.ny) & (c >= 0) & (c < contract.nx)
        r, c, l, e = r[valid], c[valid], l[valid], e[valid]

    if len(r) > 0 and apply_noise_thr:
        e_noisy = e + rng.normal(0.0, PIXEL_ENC_MEV, size=len(e))
        keep = e_noisy >= PIXEL_THR_MEV
        r, c, l, e = r[keep], c[keep], l[keep], e_noisy[keep]

    if dark_rate > 0.0:
        mean_per_layer = float(dark_rate) * float(contract.nx * contract.ny)
        dark_r, dark_c, dark_l, dark_e = [], [], [], []
        for lay in range(contract.nz):
            n_f = int(rng.poisson(mean_per_layer))
            if n_f > 0:
                dark_r.append(rng.integers(0, contract.ny, size=n_f, dtype=np.int64))
                dark_c.append(rng.integers(0, contract.nx, size=n_f, dtype=np.int64))
                dark_l.append(np.full(n_f, lay, dtype=np.int64))
                dark_e.append(np.full(n_f, PIXEL_THR_MEV, dtype=np.float64))
        if dark_r:
            dr = np.concatenate(dark_r)
            dc = np.concatenate(dark_c)
            dl = np.concatenate(dark_l)
            de = np.concatenate(dark_e)
            r = np.concatenate([r, dr]) if len(r) else dr
            c = np.concatenate([c, dc]) if len(c) else dc
            l = np.concatenate([l, dl]) if len(l) else dl
            e = np.concatenate([e, de]) if len(e) else de

    if len(r) == 0:
        return {
            'channel_id': np.zeros(0, dtype=np.uint64),
            'xyz_mm': np.zeros((0, 3), dtype=np.float32),
            'edep_mev': np.zeros(0, dtype=np.float32),
        }

    cid = ((l * contract.ny + r) * contract.nx + c).astype(np.uint64)
    if dark_rate > 0.0:
        # Deduplicate injected dark pixels against real signal hits
        cid, uniq_idx = np.unique(cid, return_index=True)
        r, c, l, e = r[uniq_idx], c[uniq_idx], l[uniq_idx], e[uniq_idx]

    px = np.asarray(contract.raw_geometry['pixel_Xpos'], dtype=np.float64)
    py = np.asarray(contract.raw_geometry['pixel_Ypos'], dtype=np.float64)
    pz = contract.pixel_Zpos
    xyz = np.column_stack([px[c], py[r], pz[l]]).astype(np.float32)
    return {
        'channel_id': cid,
        'xyz_mm': xyz,
        'edep_mev': e.astype(np.float32),
    }


def digitize_raw_scint_hits(
    panel_id: np.ndarray,
    is_horizontal: np.ndarray,
    row_id: np.ndarray,
    col_id: np.ndarray,
    n_pe: np.ndarray,
    hit_time_ns: np.ndarray,
    contract: Gen4GeometryContract,
    *,
    pe_threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    """Aggregate 10085-10088 track-level scintHits into physical bar channels and apply 14,400 SiPM saturation.

    Strictly forbids reading `scintHits.edep`! Applies exact:
      N_PE,obs = round(14400 * (1 - exp(-N_raw / 14400)))
      t_obs    = min_{k: nPE_k > 0} hitTime_k  if N_PE,obs >= 3  else -1.0
    """
    if len(n_pe) == 0:
        return {
            'channel_id': np.zeros(0, dtype=np.uint64),
            'layer': np.zeros(0, dtype=np.int16),
            'panel': np.zeros(0, dtype=np.int16),
            'horizontal': np.zeros(0, dtype=np.int8),
            'bar': np.zeros(0, dtype=np.int16),
            'pe_proxy': np.zeros(0, dtype=np.int64),
            'hit_time_ns': np.zeros(0, dtype=np.float32),
            'bounds_mm': np.zeros((0, 5), dtype=np.float32),
        }

    pan = np.asarray(panel_id, dtype=np.int64)
    ish = np.asarray(is_horizontal, dtype=np.int64)
    bar_idx = np.where(ish == 1, np.asarray(row_id, dtype=np.int64), np.asarray(col_id, dtype=np.int64))
    npe_raw = np.maximum(0.0, np.asarray(n_pe, dtype=np.float64))
    ht_raw = np.asarray(hit_time_ns, dtype=np.float64)

    n_scint = len(contract.raw_geometry['scint_Zpos'])
    n_bars = contract.n_bars
    valid = (pan >= 0) & (pan < n_scint) & (bar_idx >= 0) & (bar_idx < n_bars)
    if not valid.any():
        return digitize_raw_scint_hits(
            np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0), contract
        )
    pan, ish, bar_idx, npe_raw, ht_raw = pan[valid], ish[valid], bar_idx[valid], npe_raw[valid], ht_raw[valid]

    st = contract.panel_to_station[pan]
    raw_cid = ((st * n_scint + pan) * (2 * n_bars) + ish * n_bars + bar_idx).astype(np.uint64)
    uniq_cid, first_idx, inv = np.unique(raw_cid, return_index=True, return_inverse=True)

    sum_npe = np.bincount(inv, weights=npe_raw, minlength=len(uniq_cid))
    npe_obs_f = np.rint(SIPM_NPIX * (1.0 - np.exp(-sum_npe / SIPM_NPIX)))
    keep_ch = npe_obs_f >= float(pe_threshold)
    if not keep_ch.any():
        return digitize_raw_scint_hits(
            np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0), contract
        )

    # Earliest optical arrival time among positive-PE deposits when N_PE,obs >= 3
    t_min = np.full(len(uniq_cid), 1e18, dtype=np.float64)
    pos_t = (npe_raw > 0.0) & np.isfinite(ht_raw) & (ht_raw >= 0.0)
    if pos_t.any():
        np.minimum.at(t_min, inv[pos_t], ht_raw[pos_t])

    u_pan = pan[first_idx][keep_ch]
    u_ish = ish[first_idx][keep_ch]
    u_bar = bar_idx[first_idx][keep_ch]
    u_st = st[first_idx][keep_ch]
    u_pe = np.clip(npe_obs_f[keep_ch], 0, int(SIPM_NPIX)).astype(np.int64)
    u_t = t_min[keep_ch]
    u_ht = np.where((u_pe >= 3) & (u_t < 1e17), u_t, -1.0).astype(np.float32)

    x_c = np.where(u_ish == 1, contract.x_scint_center, contract.bar_x_centers_mm[u_bar])
    y_c = np.where(u_ish == 1, contract.bar_y_centers_mm[u_bar], contract.y_scint_center)
    hx = np.where(u_ish == 1, 0.5 * float(contract.raw_geometry['scint_detector_width']), 0.5 * contract.bar_width_mm)
    hy = np.where(u_ish == 1, 0.5 * contract.bar_width_mm, 0.5 * float(contract.raw_geometry['scint_detector_height']))
    z_c = np.asarray(contract.raw_geometry['scint_Zpos'], dtype=np.float64)[u_pan]
    b_bounds = np.column_stack([x_c - hx, x_c + hx, y_c - hy, y_c + hy, z_c]).astype(np.float32)

    return {
        'channel_id': uniq_cid[keep_ch],
        'layer': u_st.astype(np.int16),
        'panel': u_pan.astype(np.int16),
        'horizontal': u_ish.astype(np.int8),
        'bar': u_bar.astype(np.int16),
        'pe_proxy': u_pe,
        'hit_time_ns': u_ht,
        'bounds_mm': b_bounds,
    }


def digitize_raw_faser_hits(
    tracker_id: np.ndarray,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    z_mm: np.ndarray,
    *,
    event_seed: int,
    apply_eff: bool = True,
    dead_prob: float = 0.0,
    noise_per_station: int = 0,
) -> dict[str, np.ndarray]:
    """Digitize 10085-10088 faserHits strictly from (trackerID, x, y, z) (zero momentum/PDG leakage)."""
    rng = np.random.default_rng(int(event_seed) + 7919)
    tid = np.asarray(tracker_id, dtype=np.int64)
    hx = np.asarray(x_mm, dtype=np.float64)
    hy = np.asarray(y_mm, dtype=np.float64)
    hz = np.asarray(z_mm, dtype=np.float64)

    out_st: list[int] = []
    out_xy: list[list[float]] = []
    out_sig: list[list[float]] = []

    eff = (1.0 - float(dead_prob)) * (TRACKER_HIT_EFF if apply_eff else 1.0)
    for s_id in (0, 1, 2):
        ms = tid == s_id
        if ms.any():
            pts = np.column_stack([hx[ms], hy[ms]])
            # Merge hits within 0.1 mm on the same tracker plane
            used = np.zeros(len(pts), dtype=bool)
            for idx_i in range(len(pts)):
                if used[idx_i]:
                    continue
                d = np.hypot(pts[:, 0] - pts[idx_i, 0], pts[:, 1] - pts[idx_i, 1])
                grp = (~used) & (d <= TRACKER_MERGE_DIST_MM)
                used |= grp
                if apply_eff and rng.random() > eff:
                    continue
                xm = float(np.mean(pts[grp, 0])) + float(rng.normal(0.0, SIGMA_X_MM))
                ym = float(np.mean(pts[grp, 1])) + float(rng.normal(0.0, SIGMA_Y_MM))
                if TRACKER_X_MIN_MM <= xm <= TRACKER_X_MAX_MM and TRACKER_Y_MIN_MM <= ym <= TRACKER_Y_MAX_MM:
                    out_st.append(s_id)
                    out_xy.append([xm, ym])
                    out_sig.append([SIGMA_X_MM, SIGMA_Y_MM])

        if noise_per_station > 0:
            for _ in range(int(noise_per_station)):
                out_st.append(s_id)
                out_xy.append([
                    float(rng.uniform(TRACKER_X_MIN_MM, TRACKER_X_MAX_MM)),
                    float(rng.uniform(TRACKER_Y_MIN_MM, TRACKER_Y_MAX_MM)),
                ])
                out_sig.append([SIGMA_X_MM, SIGMA_Y_MM])

    return {
        'station': np.asarray(out_st, dtype=np.int8),
        'xy_mm': np.asarray(out_xy, dtype=np.float32).reshape(-1, 2),
        'sigma_xy_mm': np.asarray(out_sig, dtype=np.float32).reshape(-1, 2),
    }


def build_observation_from_raw_root_event(
    pix_row: np.ndarray,
    pix_col: np.ndarray,
    pix_layer: np.ndarray,
    pix_edep: np.ndarray,
    scint_panel: np.ndarray,
    scint_ish: np.ndarray,
    scint_row: np.ndarray,
    scint_col: np.ndarray,
    scint_npe: np.ndarray,
    scint_ht: np.ndarray,
    faser_tid: np.ndarray,
    faser_x: np.ndarray,
    faser_y: np.ndarray,
    faser_z: np.ndarray,
    contract: Gen4GeometryContract,
    *,
    event_seed: int,
    dark_rate: float = 0.0,
    tracker_dead_prob: float = 0.0,
    tracker_noise: int = 0,
) -> dict[str, Any]:
    """Build canonical leak-free Gen4 observation dictionary from raw 10085-10088 ROOT branches."""
    return {
        'pixel': digitize_raw_pixel_hits(
            pix_row, pix_col, pix_layer, pix_edep, contract,
            event_seed=event_seed, apply_noise_thr=True, dark_rate=dark_rate,
        ),
        'bar': digitize_raw_scint_hits(
            scint_panel, scint_ish, scint_row, scint_col, scint_npe, scint_ht, contract,
        ),
        'tracker_hypothesis': digitize_raw_faser_hits(
            faser_tid, faser_x, faser_y, faser_z,
            event_seed=event_seed, apply_eff=True, dead_prob=tracker_dead_prob, noise_per_station=tracker_noise,
        ),
    }


# =============================================================================
# 1. Pre-Truncation DarkShield + 32-dim Kurtosis/Bimodal Cluster Tokenizer
# =============================================================================

def _cluster_sensor_groups_gen4(
    cid: np.ndarray,
    xyz: np.ndarray,
    nx: int,
    ny: int,
    n_target_planes: int,
    r_tgt: float = 0.06,
    r_ipt: float = 0.06,
) -> dict[str, Any]:
    """Cluster raw binary silicon hits per sensor group (target 0..11 vs IPT 12..14)."""
    if len(cid) == 0:
        return {
            'n_clusters': 0,
            'hit_order': np.empty(0, dtype=np.int32),
            'cluster_offsets': np.zeros(1, dtype=np.int32),
            'cluster_plane': np.empty(0, dtype=np.int16),
            'cluster_first_channel_id': np.empty(0, dtype=np.uint64),
        }
    stride = int(nx) * int(ny)
    planes = cid // stride
    tgt_idx = np.flatnonzero(planes < n_target_planes)
    ipt_idx = np.flatnonzero(planes >= n_target_planes)

    def _run_one(sub_cid: np.ndarray, sub_xyz: np.ndarray, rad: float) -> dict[str, Any]:
        res = cluster_pixel_hits_fast(sub_cid, sub_xyz, nx, ny, radius_mm=rad)
        if res is None:
            res = cluster_pixel_hits(sub_cid, sub_xyz, nx, ny, radius_mm=rad)
        return res

    if len(ipt_idx) == 0:
        return _run_one(cid, xyz, r_tgt)
    if len(tgt_idx) == 0:
        return _run_one(cid, xyz, r_ipt)

    cl_t = _run_one(cid[tgt_idx], xyz[tgt_idx], r_tgt)
    cl_i = _run_one(cid[ipt_idx], xyz[ipt_idx], r_ipt)
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


def _compute_neighborhoods_safe(
    cl_plane: np.ndarray,
    x_mean: np.ndarray,
    y_mean: np.ndarray,
    n_hits: np.ndarray,
) -> tuple[np.ndarray, ...]:
    n_cl = len(cl_plane)
    if n_cl <= 1:
        z = np.zeros(n_cl, dtype=np.float64)
        return z, z.copy(), z.copy(), z.copy(), z.copy(), z.copy(), z.copy()
    nb = plane_neighborhoods_fast(cl_plane, x_mean, y_mean, n_hits)
    if nb is not None:
        return nb
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
        off = coo.row != coo.col
        if not off.any():
            continue
        r, c, d = coo.row[off], coo.col[off], coo.data[off]
        m_p = len(idx)
        for msk, c_arr, h_arr in (
            (d <= 0.15, cnt_r015, hits_r015),
            (d <= 0.40, cnt_r040, hits_r040),
            (d <= 5.0, cnt_r5, hits_r5),
        ):
            if msk.any():
                c_arr[idx] = np.bincount(r[msk], minlength=m_p).astype(np.float64)
                h_arr[idx] = np.bincount(r[msk], weights=n_hits[idx[c[msk]]], minlength=m_p)
        m1 = d <= 1.0
        if m1.any():
            cnt_r1[idx] = np.bincount(r[m1], minlength=m_p).astype(np.float64)
    return cnt_r015, hits_r015, cnt_r040, hits_r040, cnt_r1, cnt_r5, hits_r5


def _extract_scint_active_x_by_block(
    bar: dict[str, Any],
    contract: Gen4GeometryContract,
    *,
    min_pe: float = 0.5,
) -> dict[int, np.ndarray]:
    """Return active vertical-bar X centers (including <2.5 PE MIP bars!) per PinpointBlock [0..11]."""
    out: dict[int, np.ndarray] = {}
    if len(bar.get('panel', ())) == 0:
        return out
    bp = np.asarray(bar['panel'], dtype=np.int64)
    bh = np.asarray(bar['horizontal'], dtype=np.int64)
    bi = np.asarray(bar['bar'], dtype=np.int64)
    bpe = np.asarray(bar['pe_proxy'], dtype=np.float64)
    st = contract.panel_to_station[bp]
    keep = (bh == 0) & (bpe >= min_pe)
    if not keep.any():
        return out
    st_k = st[keep]
    x_k = contract.bar_x_centers_mm[bi[keep]]
    for pb_idx, scint_st in enumerate(contract.pinpoint_scint_stations):
        m = st_k == int(scint_st)
        if m.any():
            out[pb_idx] = x_k[m]
    return out


def _extract_quiet_doublet_xy_by_station(
    bar: dict[str, Any],
    contract: Gen4GeometryContract,
    *,
    min_pe: float = 0.5,
    max_bars_per_view: int = 3,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Return active (x_bars, y_bars) ONLY for quiet/MIP-like doublet stations (<= 3 bars/view)."""
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if len(bar.get('panel', ())) == 0:
        return out
    bp = np.asarray(bar['panel'], dtype=np.int64)
    bh = np.asarray(bar['horizontal'], dtype=np.int64)
    bi = np.asarray(bar['bar'], dtype=np.int64)
    bpe = np.asarray(bar['pe_proxy'], dtype=np.float64)
    st = contract.panel_to_station[bp]
    keep = bpe >= min_pe
    if not keep.any():
        return out
    st_k, bh_k, bi_k = st[keep], bh[keep], bi[keep]
    for s_dbl in contract.s_stereo:
        s_i = int(s_dbl)
        ms = st_k == s_i
        if ms.any():
            xs = contract.bar_x_centers_mm[bi_k[ms & (bh_k == 0)]]
            ys = contract.bar_y_centers_mm[bi_k[ms & (bh_k == 1)]]
            if 1 <= len(xs) <= max_bars_per_view and 1 <= len(ys) <= max_bars_per_view:
                out[s_i] = (xs, ys)
    return out


def _rasterize_gen4_patches(
    order: np.ndarray,
    offsets: np.ndarray,
    pxyz: np.ndarray,
    pedep_mev: np.ndarray | None,
    x_mean: np.ndarray,
    y_mean: np.ndarray,
    sel_local: np.ndarray,
    keep_idx: np.ndarray,
    pitch_x: float,
    pitch_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize up to 64 (48 target + 16 IPT) 32x32 micro-patches stored as compact uint8 [0..255].

    When `pedep_mev` is provided (`10085-10088` Track-B), channel 0 is binary footprint
    and channel 1 is continuous pixel charge `clip(ln(1 + q_{r,c}^{MIP}) / 2.5, 0, 1)`.
    """
    n_p = len(sel_local)
    if n_p == 0:
        return np.empty((0, 2, PATCH_SIZE, PATCH_SIZE), dtype=np.uint8), np.empty(0, dtype=np.int64)
    if pedep_mev is None:
        fast_p = rasterize_patches_fast(
            order, offsets, pxyz, x_mean, y_mean, sel_local, keep_idx, PATCH_SIZE, pitch_x, pitch_y
        )
        if fast_p is not None:
            patches_f16, _ = fast_p
            patches_u8 = np.rint(np.clip(patches_f16.astype(np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)
            return patches_u8, sel_local.astype(np.int64)

    patches = np.zeros((n_p, 2, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    half_center = 0.5 * (PATCH_SIZE - 1)
    hx_ord = pxyz[order, 0].astype(np.float64)
    hy_ord = pxyz[order, 1].astype(np.float64)
    hq_ord = (
        np.clip(np.log1p(np.maximum(0.0, pedep_mev[order].astype(np.float64) * 1000.0 / PIXEL_MIP_KEV)) / 2.5, 0.0, 1.0)
        if pedep_mev is not None
        else None
    )
    for p_i, loc_i in enumerate(sel_local):
        cl_id = int(keep_idx[loc_i])
        s, e = int(offsets[cl_id]), int(offsets[cl_id + 1])
        u = half_center + (hx_ord[s:e] - x_mean[cl_id]) / pitch_x
        v = half_center + (hy_ord[s:e] - y_mean[cl_id]) / pitch_y
        inside = (u >= -0.5) & (u <= (PATCH_SIZE - 0.5)) & (v >= -0.5) & (v <= (PATCH_SIZE - 0.5))
        n_out = int((~inside).sum())
        if inside.any():
            ui = np.clip(u[inside], 0.0, PATCH_SIZE - 1.0 - 1e-5)
            vi = np.clip(v[inside], 0.0, PATCH_SIZE - 1.0 - 1e-5)
            u0, v0 = np.floor(ui).astype(np.int64), np.floor(vi).astype(np.int64)
            du, dv = ui - u0, vi - v0
            u1, v1 = np.minimum(u0 + 1, PATCH_SIZE - 1), np.minimum(v0 + 1, PATCH_SIZE - 1)
            w00 = (1.0 - dv) * (1.0 - du)
            w01 = (1.0 - dv) * du
            w10 = dv * (1.0 - du)
            w11 = dv * du
            np.add.at(patches[p_i, 0], (v0, u0), w00)
            np.add.at(patches[p_i, 0], (v0, u1), w01)
            np.add.at(patches[p_i, 0], (v1, u0), w10)
            np.add.at(patches[p_i, 0], (v1, u1), w11)
            if hq_ord is not None:
                q_in = hq_ord[s:e][inside]
                np.add.at(patches[p_i, 1], (v0, u0), w00 * q_in)
                np.add.at(patches[p_i, 1], (v0, u1), w01 * q_in)
                np.add.at(patches[p_i, 1], (v1, u0), w10 * q_in)
                np.add.at(patches[p_i, 1], (v1, u1), w11 * q_in)
        if hq_ord is None and n_out > 0:
            edge_val = math.log1p(n_out) / 5.0
            patches[p_i, 1, :2, :] = edge_val
            patches[p_i, 1, -2:, :] = edge_val
            patches[p_i, 1, :, :2] = edge_val
            patches[p_i, 1, :, -2:] = edge_val
    patches_u8 = np.rint(np.clip(patches, 0.0, 1.0) * 255.0).astype(np.uint8)
    return patches_u8, sel_local.astype(np.int64)


def dark_shield_filter_and_features(
    observation: dict[str, Any],
    contract: Gen4GeometryContract,
    *,
    max_target_clusters: int = N_MAX_TARGET_CLUSTERS_GEN4,
    max_ipt_clusters: int = N_MAX_IPT_CLUSTERS_GEN4,
    scint_min_pe: float = 0.5,
    enable_dark_filter: str | bool = 'auto',
    use_edep: bool = True,
) -> dict[str, Any]:
    """Pre-truncation DarkShield filter + 36-dim cluster features & 32x32 micro-patches.

    - Supports continuous single-pixel `edep_mev` (`10085-10088` Track-B -> `PIX_FEAT_GEN4_DIM=36`).
    - When `enable_dark_filter='auto'`, activates hard topological pruning when uniform dark-noise
      floor is present (e.g. `pixel_dark_1e-5`: `median(isolated_1pix_per_plane) >= 150`), achieving
      `>99.8%` dark rejection while preserving `100%` of zero-noise `10085-10088` single-pixel MIPs.
      Pass `enable_dark_filter=True` to force pruning unconditionally (e.g. DarkShield unit tests).
    """
    pixel = observation['pixel']
    bar = observation['bar']
    pcid = np.asarray(pixel['channel_id'], dtype=np.uint64)
    pxyz = np.asarray(pixel['xyz_mm'], dtype=np.float64).reshape(-1, 3)
    raw_edep = pixel.get('edep_mev', None)
    has_edep = bool(use_edep and (raw_edep is not None))
    pedep = np.asarray(raw_edep, dtype=np.float64) if has_edep else None

    n_planes = contract.nz
    n_tgt_planes = contract.n_target_planes

    if len(pcid) == 0:
        obs_clean = dict(observation)
        dom_clean = compute_observable_containment(
            obs_clean, contract.raw_geometry, contract.bounds_dict, config={'pe_only': True, 'bar_threshold_pe': 0.5}
        )
        ps = np.zeros((n_planes, PIX_PLANE_SUMMARY_DIM), dtype=np.float32)
        ps[:, 5] = (contract.pixel_X0 / 50.0).astype(np.float32)
        return {
            'n_raw_hits': 0,
            'n_raw_clusters': 0,
            'n_kept_clusters': 0,
            'n_truncated_clusters': 0,
            'has_edep': bool(has_edep),
            'dark_noise_detected': False,
            'pix_xyz_mm': np.zeros((0, 3), dtype=np.float32),
            'pix_plane': np.zeros(0, dtype=np.int16),
            'pix_feat': np.zeros((0, PIX_FEAT_GEN4_DIM), dtype=np.float32),
            'pix_cov_2x2': np.zeros((0, 3), dtype=np.float32),
            'pix_persist_prob': np.zeros(0, dtype=np.float32),
            'pix_first_cid': np.zeros(0, dtype=np.uint64),
            'pix_plane_summary': ps,
            'pix_patches_u8': np.zeros((0, 2, PATCH_SIZE, PATCH_SIZE), dtype=np.uint8),
            'patch_cluster_idx': np.zeros(0, dtype=np.int64),
            'c_obs_clean': dom_clean['c_obs'],
            'obs_domain_id_clean': int(dom_clean['obs_domain_id']),
            'cal_start_clean': dom_clean['cal_start'],
            'raw_keep_mask_clusters': np.zeros(0, dtype=bool),
            'raw_keep_mask_hits': np.zeros(0, dtype=bool),
            'raw_cluster_plane': np.zeros(0, dtype=np.int16),
            'raw_cluster_nhits': np.zeros(0, dtype=np.float64),
            'raw_cluster_xyz': np.zeros((0, 3), dtype=np.float64),
            'hit_order': np.zeros(0, dtype=np.int32),
            'cluster_offsets': np.zeros(1, dtype=np.int32),
            'sel_raw_clusters': np.zeros(0, dtype=np.int64),
        }

    cl = _cluster_sensor_groups_gen4(pcid, pxyz, contract.nx, contract.ny, n_tgt_planes, 0.06, 0.06)
    n_cl = int(cl['n_clusters'])
    order = cl['hit_order']
    offsets = cl['cluster_offsets']
    starts = offsets[:-1]
    n_hits = np.diff(offsets).astype(np.float64)
    cl_plane = cl['cluster_plane'].astype(np.int64)
    cl_fcid = cl['cluster_first_channel_id'].astype(np.uint64)

    hx = pxyz[order, 0].astype(np.float64)
    hy = pxyz[order, 1].astype(np.float64)
    x_geom_mean = np.add.reduceat(hx, starts) / n_hits
    y_geom_mean = np.add.reduceat(hy, starts) / n_hits
    if pedep is not None:
        he_mev = np.maximum(0.0, pedep[order].astype(np.float64))
        cl_edep_mev = np.add.reduceat(he_mev, starts)
        w_sum = np.maximum(cl_edep_mev, 1e-12)
        x_mean = np.where(cl_edep_mev > 0.0, np.add.reduceat(hx * he_mev, starts) / w_sum, x_geom_mean)
        y_mean = np.where(cl_edep_mev > 0.0, np.add.reduceat(hy * he_mev, starts) / w_sum, y_geom_mean)
        he_kev = he_mev * 1000.0
        cl_edep_kev = cl_edep_mev * 1000.0
        q_max_kev = np.maximum.reduceat(he_kev, starts)
    else:
        x_mean = x_geom_mean
        y_mean = y_geom_mean
        cl_edep_kev = np.zeros(n_cl, dtype=np.float64)
        q_max_kev = np.zeros(n_cl, dtype=np.float64)
    z_cl = contract.pixel_Zpos[cl_plane]

    # Pre-truncation DarkShield classification on ALL raw clusters
    multi_mask = n_hits >= 2.0
    near_multi_same = np.zeros(n_cl, dtype=bool)
    cross_multi_ok = np.zeros(n_cl, dtype=bool)
    cross_scint_1hit_ok = np.zeros(n_cl, dtype=bool)
    collinear_3_ok = np.zeros(n_cl, dtype=bool)

    cl_per_plane_raw = np.bincount(cl_plane, minlength=n_planes)
    need_full_darkshield = (enable_dark_filter is True) or (
        enable_dark_filter == 'auto' and float(np.median(cl_per_plane_raw)) >= 150.0
    )

    if need_full_darkshield:
        scint_x_by_blk = _extract_scint_active_x_by_block(bar, contract, min_pe=scint_min_pe)
        quiet_dbl_by_st = _extract_quiet_doublet_xy_by_station(bar, contract, min_pe=scint_min_pe, max_bars_per_view=3)

        plane_trees_all: dict[int, tuple[np.ndarray, cKDTree]] = {}
        plane_trees_multi: dict[int, tuple[np.ndarray, cKDTree]] = {}
        for p in np.unique(cl_plane):
            p_i = int(p)
            idx_p = np.flatnonzero(cl_plane == p_i)
            plane_trees_all[p_i] = (idx_p, cKDTree(np.column_stack([x_mean[idx_p], y_mean[idx_p]])))
            idx_pm = idx_p[multi_mask[idx_p]]
            if len(idx_pm) > 0:
                tree_m = cKDTree(np.column_stack([x_mean[idx_pm], y_mean[idx_pm]]))
                plane_trees_multi[p_i] = (idx_pm, tree_m)
                idx_p1 = idx_p[~multi_mask[idx_p]]
                if len(idx_p1) > 0:
                    dists, _ = tree_m.query(np.column_stack([x_mean[idx_p1], y_mean[idx_p1]]), k=1)
                    near_multi_same[idx_p1] = dists <= 0.25

        single_idx = np.flatnonzero(~multi_mask & ~near_multi_same)
    else:
        single_idx = np.empty(0, dtype=np.int64)
    if len(single_idx) > 0:
        for p_i in np.unique(cl_plane[single_idx]):
            p_int = int(p_i)
            loc = single_idx[cl_plane[single_idx] == p_int]
            pts = np.column_stack([x_mean[loc], y_mean[loc]])
            z_p = float(contract.pixel_Zpos[p_int])

            if p_int <= 5:
                adj_list = [q for q in (p_int - 1, p_int + 1, p_int - 2, p_int + 2) if 0 <= q <= 5]
            elif p_int < n_tgt_planes:
                adj_list = [q for q in (p_int - 1, p_int + 1) if 5 <= q < n_tgt_planes]
            else:
                adj_list = [q for q in (p_int - 1, p_int + 1, p_int - 2, p_int + 2) if n_tgt_planes <= q < n_planes]

            # 1) Tight projective corridor to multi-pixel clusters in adjacent plane
            for q in adj_list:
                if q not in plane_trees_multi:
                    continue
                dz = abs(float(contract.pixel_Zpos[q]) - z_p)
                r_tol = (0.015 * dz + 0.12) if dz <= 65.0 else (0.012 * dz + 0.15)
                _, tree_m = plane_trees_multi[q]
                d_q, _ = tree_m.query(pts, k=1)
                cross_multi_ok[loc] |= (d_q <= r_tol)

            # 2) Same-block Scint_V[p] (-13.505 mm) & Front30 downstream Scint_V[p+1] (+16.505 mm) coincidence
            if p_int < n_tgt_planes:
                sc_xs_up = scint_x_by_blk.get(p_int)
                sc_xs_dn = scint_x_by_blk.get(p_int + 1) if (contract.is_mc8588_block_order and p_int < 5) else None
                if sc_xs_up is not None and sc_xs_dn is not None:
                    sc_xs = np.concatenate([sc_xs_up, sc_xs_dn])
                else:
                    sc_xs = sc_xs_up if sc_xs_up is not None else sc_xs_dn
                has_scint_match = (
                    np.min(np.abs(pts[:, 0:1] - sc_xs[None, :]), axis=1) <= 5.5
                    if (sc_xs is not None and len(sc_xs) > 0)
                    else np.zeros(len(loc), dtype=bool)
                )
                if has_scint_match.any():
                    if p_int <= 5:
                        # Front30: require Scint[p] match AND tight neighbor in {p-1, p+1} matching Scint[q]
                        for q in (p_int - 1, p_int + 1):
                            if 0 <= q <= 5 and q in plane_trees_all and q in scint_x_by_blk:
                                dz = abs(float(contract.pixel_Zpos[q]) - z_p)
                                idx_q, tree_q = plane_trees_all[q]
                                d_q, pos_q = tree_q.query(pts, k=1)
                                x_q = x_mean[idx_q[pos_q]]
                                sc_xq = scint_x_by_blk[q]
                                q_sc = np.min(np.abs(x_q[:, None] - sc_xq[None, :]), axis=1) <= 5.5
                                cross_scint_1hit_ok[loc] |= (has_scint_match & q_sc & (d_q <= 0.012 * dz + 0.10))
                    else:
                        # Interm150: quiet MIP-like adjacent Fortune doublet station (<= 3 bars/view) + upstream/downstream Si match
                        st_sc = int(contract.pinpoint_scint_stations[p_int])
                        for s_adj in (st_sc - 1, st_sc + 1):
                            if s_adj in quiet_dbl_by_st:
                                xs_d, ys_d = quiet_dbl_by_st[s_adj]
                                dx_d = np.min(np.abs(pts[:, 0:1] - xs_d[None, :]), axis=1)
                                dy_d = np.min(np.abs(pts[:, 1:2] - ys_d[None, :]), axis=1)
                                cand_sc = has_scint_match & (dx_d <= 5.5) & (dy_d <= 5.5)
                                if cand_sc.any():
                                    # Also require neighbor in adjacent Si plane within 3.5 mm to kill ~99% random coincidences
                                    has_adj_si = np.zeros(len(loc), dtype=bool)
                                    for q in (p_int - 1, p_int + 1):
                                        if 5 <= q < n_tgt_planes and q in plane_trees_all:
                                            _, tree_q = plane_trees_all[q]
                                            d_q, _ = tree_q.query(pts, k=1)
                                            has_adj_si |= (d_q <= 3.5)
                                    cross_scint_1hit_ok[loc] |= (cand_sc & has_adj_si)
            else:
                has_scint_match = np.zeros(len(loc), dtype=bool)

            # 3) Exact 3-plane collinear triplet search via query_ball_point (recovers 1-pixel MIPs cleanly!)
            triplets: list[tuple[int, int]] = []
            if p_int <= 5:
                for a, b in ((p_int - 1, p_int + 1), (p_int - 2, p_int - 1), (p_int + 1, p_int + 2)):
                    if 0 <= a <= 5 and 0 <= b <= 5:
                        triplets.append((a, b))
            elif p_int >= n_tgt_planes:
                if p_int == n_tgt_planes:
                    triplets.append((p_int + 1, p_int + 2))
                elif p_int == n_tgt_planes + 1:
                    triplets.append((p_int - 1, p_int + 1))
                elif p_int == n_tgt_planes + 2:
                    triplets.append((p_int - 2, p_int - 1))

            is_ipt_plane = p_int >= n_tgt_planes
            max_slope_3 = 0.16 if is_ipt_plane else 0.18
            db_tol = 0.055 if is_ipt_plane else 0.12

            for pa, pb in triplets:
                if pa not in plane_trees_all or pb not in plane_trees_all:
                    continue
                idx_a, tree_a = plane_trees_all[pa]
                idx_b, tree_b = plane_trees_all[pb]
                za = float(contract.pixel_Zpos[pa])
                zb = float(contract.pixel_Zpos[pb])
                if abs(zb - za) < 1.0:
                    continue
                max_ra = max_slope_3 * abs(z_p - za) + 0.10
                nbrs_a = tree_a.query_ball_point(pts, r=max_ra)
                t_fac = (zb - za) / (z_p - za)

                # Vectorize candidate pairs across all single-pixel points in this plane
                pair_loc_i: list[int] = []
                pair_idx_a: list[int] = []
                pair_pred_b: list[np.ndarray] = []
                for li_k, a_list in enumerate(nbrs_a):
                    if not a_list:
                        continue
                    # For Front30 without Scint match, only check if >=1 endpoint can be multi-pixel or ultra-tight 0.05mm
                    px_c, py_c = pts[li_k, 0], pts[li_k, 1]
                    for ak in a_list:
                        ia_glob = int(idx_a[ak])
                        ax, ay = x_mean[ia_glob], y_mean[ia_glob]
                        pair_loc_i.append(li_k)
                        pair_idx_a.append(ia_glob)
                        pair_pred_b.append([ax + t_fac * (px_c - ax), ay + t_fac * (py_c - ay)])

                if pair_pred_b:
                    pred_arr = np.asarray(pair_pred_b, dtype=np.float64)
                    db, pos_b = tree_b.query(pred_arr, k=1)
                    if is_ipt_plane:
                        hit_m = db <= db_tol
                    else:
                        # Front30: either scintillator match on this plane, or one of (a, b) is multi-pixel, or ultra-tight 0.05mm
                        a_multi = multi_mask[np.asarray(pair_idx_a, dtype=np.int64)]
                        b_multi = multi_mask[idx_b[pos_b]]
                        sc_loc = has_scint_match[np.asarray(pair_loc_i, dtype=np.int64)]
                        hit_m = (db <= db_tol) & (sc_loc | a_multi | b_multi | (db <= 0.050))
                    if hit_m.any():
                        for li_ok in np.asarray(pair_loc_i, dtype=np.int64)[hit_m]:
                            collinear_3_ok[loc[li_ok]] = True

    topo_keep = multi_mask | near_multi_same | cross_multi_ok | cross_scint_1hit_ok | collinear_3_ok

    q_mip_all = cl_edep_kev / PIXEL_MIP_KEV
    mip_charge_bonus = (
        np.where(q_mip_all >= 0.45, 0.9, np.where(q_mip_all <= 0.16, -0.8, 0.0))
        if has_edep
        else np.zeros(n_cl, dtype=np.float64)
    )
    persist_logit = (
        4.0 * multi_mask.astype(np.float64)
        + 2.5 * near_multi_same.astype(np.float64)
        + 2.2 * cross_multi_ok.astype(np.float64)
        + 2.2 * cross_scint_1hit_ok.astype(np.float64)
        + 2.8 * collinear_3_ok.astype(np.float64)
        + mip_charge_bonus
        - 1.5
    )
    persist_prob = _sigmoid_np(persist_logit).astype(np.float32)

    if enable_dark_filter is True:
        apply_hard_prune = True
        dark_detected = True
    elif enable_dark_filter is False:
        apply_hard_prune = False
        dark_detected = False
    else:
        iso_per_plane = np.bincount(cl_plane, weights=(~topo_keep).astype(np.float64), minlength=n_planes)
        dark_detected = bool(np.median(iso_per_plane) >= 150.0)
        apply_hard_prune = dark_detected

    keep_cluster = topo_keep if apply_hard_prune else np.ones(n_cl, dtype=bool)

    # Map kept clusters back to kept raw hits so `c_obs_clean` is immune to 1e-5 dark noise!
    n_rep = n_hits.astype(np.int64)
    hit_cl_sorted = np.repeat(np.arange(n_cl, dtype=np.int64), n_rep)
    keep_hit_sorted = keep_cluster[hit_cl_sorted]
    keep_hit_mask = np.zeros(len(pcid), dtype=bool)
    keep_hit_mask[order[keep_hit_sorted]] = True

    obs_clean = dict(observation)
    obs_clean['pixel'] = {
        'channel_id': pcid[keep_hit_mask],
        'xyz_mm': pxyz[keep_hit_mask].astype(np.float32),
    }
    dom_clean = compute_observable_containment(
        obs_clean, contract.raw_geometry, contract.bounds_dict, config={'pe_only': True, 'bar_threshold_pe': 0.5}
    )

    kept_cl_ids = np.flatnonzero(keep_cluster)
    if len(kept_cl_ids) == 0:
        ps = np.zeros((n_planes, PIX_PLANE_SUMMARY_DIM), dtype=np.float32)
        ps[:, 5] = (contract.pixel_X0 / 50.0).astype(np.float32)
        return {
            'n_raw_hits': int(len(pcid)),
            'n_raw_clusters': int(n_cl),
            'n_kept_clusters': 0,
            'n_truncated_clusters': 0,
            'has_edep': bool(has_edep),
            'dark_noise_detected': bool(dark_detected),
            'pix_xyz_mm': np.zeros((0, 3), dtype=np.float32),
            'pix_plane': np.zeros(0, dtype=np.int16),
            'pix_feat': np.zeros((0, PIX_FEAT_GEN4_DIM), dtype=np.float32),
            'pix_cov_2x2': np.zeros((0, 3), dtype=np.float32),
            'pix_persist_prob': np.zeros(0, dtype=np.float32),
            'pix_first_cid': np.zeros(0, dtype=np.uint64),
            'pix_plane_summary': ps,
            'pix_patches_u8': np.zeros((0, 2, PATCH_SIZE, PATCH_SIZE), dtype=np.uint8),
            'patch_cluster_idx': np.zeros(0, dtype=np.int64),
            'c_obs_clean': dom_clean['c_obs'],
            'obs_domain_id_clean': int(dom_clean['obs_domain_id']),
            'cal_start_clean': dom_clean['cal_start'],
            'raw_keep_mask_clusters': keep_cluster,
            'raw_keep_mask_hits': keep_hit_mask,
            'raw_cluster_plane': cl_plane.astype(np.int16),
            'raw_cluster_nhits': n_hits,
            'raw_cluster_xyz': np.column_stack([x_mean, y_mean, z_cl]),
            'hit_order': order,
            'cluster_offsets': offsets,
            'sel_raw_clusters': np.zeros(0, dtype=np.int64),
        }

    # Recompute neighborhoods & noise-immune plane summary strictly on kept clusters BEFORE Top-K truncation
    k_plane = cl_plane[kept_cl_ids]
    k_nhits = n_hits[kept_cl_ids]
    k_xm = x_mean[kept_cl_ids]
    k_ym = y_mean[kept_cl_ids]
    k_zm = z_cl[kept_cl_ids]
    k_persist = persist_prob[kept_cl_ids]
    k_fcid = cl_fcid[kept_cl_ids]

    k_c015, k_h015, k_c040, k_h040, k_c1, k_c5, k_h5 = _compute_neighborhoods_safe(k_plane, k_xm, k_ym, k_nhits)

    x_c, y_c = contract.x_pix_center, contract.y_pix_center
    z_mid = float(contract.bounds_dict['z_mid'])
    plane_summary = np.zeros((n_planes, PIX_PLANE_SUMMARY_DIM), dtype=np.float32)
    plane_summary[:, 5] = (contract.pixel_X0 / 50.0).astype(np.float32)

    p_hit_cnt = np.bincount(k_plane, weights=k_nhits, minlength=n_planes)
    p_cl_cnt = np.bincount(k_plane, minlength=n_planes)
    p_xsum = np.bincount(k_plane, weights=k_xm * k_nhits, minlength=n_planes)
    p_ysum = np.bincount(k_plane, weights=k_ym * k_nhits, minlength=n_planes)
    p_act = p_hit_cnt > 0

    w_rob = np.square(k_nhits + 0.5 * k_h040)
    p_wsum = np.bincount(k_plane, weights=w_rob, minlength=n_planes)
    plane_xc = np.where(
        p_act,
        np.bincount(k_plane, weights=w_rob * k_xm, minlength=n_planes) / np.maximum(1e-12, p_wsum),
        x_c,
    )
    plane_yc = np.where(
        p_act,
        np.bincount(k_plane, weights=w_rob * k_ym, minlength=n_planes) / np.maximum(1e-12, p_wsum),
        y_c,
    )
    dr_cp = np.hypot(k_xm - plane_xc[k_plane], k_ym - plane_yc[k_plane])
    p_h015 = np.bincount(k_plane, weights=np.where(dr_cp <= 0.15, k_nhits, 0.0), minlength=n_planes)
    p_h040 = np.bincount(k_plane, weights=np.where(dr_cp <= 0.40, k_nhits, 0.0), minlength=n_planes)
    p_h200 = np.bincount(k_plane, weights=np.where(dr_cp <= 2.00, k_nhits, 0.0), minlength=n_planes)

    # Core eccentricity (within r <= 2.50 mm of robust centroid) matching pixel_tokenizer.py slot 11
    core25 = dr_cp <= 2.50
    h_c25 = np.bincount(k_plane[core25], weights=k_nhits[core25], minlength=n_planes)
    dx_cp = k_xm - plane_xc[k_plane]
    dy_cp = k_ym - plane_yc[k_plane]
    p_cmx = np.bincount(k_plane[core25], weights=k_nhits[core25] * dx_cp[core25], minlength=n_planes) / np.maximum(1.0, h_c25)
    p_cmy = np.bincount(k_plane[core25], weights=k_nhits[core25] * dy_cp[core25], minlength=n_planes) / np.maximum(1.0, h_c25)
    p_cxx = np.bincount(k_plane[core25], weights=k_nhits[core25] * dx_cp[core25] ** 2, minlength=n_planes) / np.maximum(1.0, h_c25)
    p_cyy = np.bincount(k_plane[core25], weights=k_nhits[core25] * dy_cp[core25] ** 2, minlength=n_planes) / np.maximum(1.0, h_c25)
    p_cxy = np.bincount(k_plane[core25], weights=k_nhits[core25] * dx_cp[core25] * dy_cp[core25], minlength=n_planes) / np.maximum(1.0, h_c25)
    cvxx = np.maximum(0.0, p_cxx - p_cmx ** 2) + (contract.dx_mm ** 2) / 12.0
    cvyy = np.maximum(0.0, p_cyy - p_cmy ** 2) + (contract.dy_mm ** 2) / 12.0
    cvxy = p_cxy - p_cmx * p_cmy
    c_tr = cvxx + cvyy
    c_df = cvxx - cvyy
    c_disc = np.sqrt(np.maximum(0.0, c_df ** 2 + 4.0 * cvxy ** 2))
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

    # Top-K stratified selection (2304 total: 2048 Target + 256 IPT), preserving 100% of first-4-plane core R<=15mm
    def _select_cap(local_mask: np.ndarray, cap: int) -> np.ndarray:
        g_idx = np.flatnonzero(local_mask)
        if len(g_idx) <= cap:
            return g_idx
        gp = k_plane[g_idx]
        uniq_p = np.unique(gp)
        first4 = uniq_p[:4]
        core_flag = np.isin(gp, first4) & (np.hypot(k_xm[g_idx] - plane_xc[gp], k_ym[g_idx] - plane_yc[gp]) <= 15.0)
        core_sel = g_idx[core_flag]
        if len(core_sel) >= cap:
            ord_c = np.argsort(-(k_nhits[core_sel] + 0.5 * k_h040[core_sel]), kind='mergesort')
            return np.sort(core_sel[ord_c[:cap]])
        rem_sel = g_idx[~core_flag]
        score_r = k_nhits[rem_sel] + 0.5 * k_h040[rem_sel] - 0.1 * k_c5[rem_sel]
        ord_r = np.argsort(-score_r, kind='mergesort')
        need = cap - len(core_sel)
        return np.sort(np.concatenate([core_sel, rem_sel[ord_r[:need]]]))

    sel_tgt = _select_cap(k_plane < n_tgt_planes, max_target_clusters)
    sel_ipt = _select_cap(k_plane >= n_tgt_planes, max_ipt_clusters)
    sel_kept = np.concatenate([sel_tgt, sel_ipt])
    sel_raw = kept_cl_ids[sel_kept]
    n_sel = len(sel_raw)

    # Extract 32-dim `pix_feat` matching Gen4EventBatch contract:
    # [0:22] = PIX_FEAT_DIM=22 (CP_y flip indices (1, 11, 17)),
    # [22] = tanh(gamma_{1,||}/2), [23] = tanh(kappa_||/3), [24] = B_split, [25] = eta_neck,
    # [26] = g_persist, [27:30] = [Sigma_xx, Sigma_yy, Sigma_xy] (mm^2), [30:32] = 0.0 (Track-B)
    px2 = contract.dx_mm ** 2
    py2 = contract.dy_mm ** 2
    feat = np.zeros((n_sel, PIX_FEAT_GEN4_DIM), dtype=np.float32)
    cov_2x2 = np.zeros((n_sel, 3), dtype=np.float32)
    dx_raw_arr = np.zeros(n_sel, dtype=np.float64)
    dy_raw_arr = np.zeros(n_sel, dtype=np.float64)

    # Vectorized fast-path for all 1-pixel clusters (~91% of 10085-10088 clusters)
    sel_nh_i = (offsets[sel_raw + 1] - offsets[sel_raw]).astype(np.int64)
    m1 = sel_nh_i == 1
    if m1.any():
        rk1 = sel_raw[m1]
        kp1 = sel_kept[m1]
        pl1 = cl_plane[rk1]
        xm1 = x_mean[rk1]
        ym1 = y_mean[rk1]
        zm1 = z_cl[rk1]
        rxc1 = xm1 - plane_xc[pl1]
        ryc1 = ym1 - plane_yc[pl1]
        drc1 = np.hypot(rxc1, ryc1)
        vx1 = px2 / 12.0
        vy1 = py2 / 12.0
        lam1_1 = max(vx1, vy1)
        lam2_1 = min(vx1, vy1)
        ecc1 = math.sqrt(max(0.0, 1.0 - lam2_1 / lam1_1))
        df1 = vx1 - vy1
        disc1 = abs(df1)
        cos2a1 = (df1 / disc1) if disc1 > 1e-12 else 0.0
        sxx1 = 1.12 * vx1 if vx1 >= vy1 else 1.20 * vx1
        syy1 = 1.20 * vy1 if vx1 >= vy1 else 1.12 * vy1
        if vx1 < vy1:
            sxx1, syy1 = 1.12 * vx1, 1.20 * vy1
        else:
            sxx1, syy1 = 1.20 * vx1, 1.12 * vy1

        feat[m1, 0] = (xm1 - x_c) / 135.0
        feat[m1, 1] = (ym1 - y_c) / 100.0
        feat[m1, 2] = (zm1 - z_mid) / 1000.0
        feat[m1, 3] = math.log1p(1.0) / 6.0
        feat[m1, 4] = math.asinh(contract.dx_mm / 0.1) / 3.0
        feat[m1, 5] = math.asinh(contract.dy_mm / 0.1) / 3.0
        feat[m1, 6] = 1.0
        feat[m1, 7] = math.asinh(math.sqrt(lam1_1) / 0.05) / 3.0
        feat[m1, 8] = math.asinh(math.sqrt(lam2_1) / 0.05) / 3.0
        feat[m1, 9] = ecc1
        feat[m1, 10] = cos2a1
        feat[m1, 11] = 0.0
        feat[m1, 12] = 0.0
        feat[m1, 13] = np.log1p(k_c1[kp1]) / 4.0
        feat[m1, 14] = np.log1p(k_c5[kp1]) / 5.0
        feat[m1, 15] = np.log1p(k_h5[kp1]) / 7.0
        feat[m1, 16] = np.arcsinh(rxc1 / 0.10) / 4.0
        feat[m1, 17] = np.arcsinh(ryc1 / 0.10) / 4.0
        feat[m1, 18] = np.arcsinh(drc1 / 0.08) / 4.0
        feat[m1, 19] = np.log1p(k_c015[kp1]) / 3.0
        feat[m1, 20] = np.log1p(k_h015[kp1]) / 5.0
        feat[m1, 21] = np.log1p(k_c040[kp1]) / 4.0
        feat[m1, 26] = k_persist[kp1]
        feat[m1, 27] = sxx1
        feat[m1, 28] = syy1
        cov_2x2[m1, 0] = sxx1
        cov_2x2[m1, 1] = syy1
        if has_edep:
            q1 = np.maximum(0.0, cl_edep_kev[rk1] / PIXEL_MIP_KEV)
            feat[m1, 30] = np.log1p(q1)
            feat[m1, 31] = _sigmoid_np((q1 - 1.65) / 0.25)
            feat[m1, 32] = 1.0
            feat[m1, 35] = np.log1p(q1 / math.sqrt(lam1_1 / px2 + 1.0))

    multi_out_indices = np.flatnonzero(~m1)
    for out_i in multi_out_indices:
        rk = int(sel_raw[out_i])
        kp = int(sel_kept[out_i])
        s_h = int(offsets[rk])
        e_h = int(offsets[rk + 1])
        m_h = e_h - s_h
        pl_i = int(cl_plane[rk])
        xm_i = x_mean[rk]
        ym_i = y_mean[rk]
        zm_i = z_cl[rk]

        dx_h = hx[s_h:e_h] - xm_i
        dy_h = hy[s_h:e_h] - ym_i
        vx = float(np.mean(dx_h * dx_h)) + px2 / 12.0
        vy = float(np.mean(dy_h * dy_h)) + py2 / 12.0
        cxy = float(np.mean(dx_h * dy_h))

        tr = vx + vy
        df = vx - vy
        disc = math.sqrt(max(0.0, df * df + 4.0 * cxy * cxy))
        lam1 = max(1e-12, 0.5 * (tr + disc))
        lam2 = max(1e-12, 0.5 * (tr - disc))
        ecc = math.sqrt(max(0.0, min(1.0, 1.0 - lam2 / lam1)))

        if disc > 1e-12:
            cos2a = df / disc
            sin2a = (2.0 * cxy) / disc
            alpha = 0.5 * math.atan2(2.0 * cxy, df)
        else:
            cos2a, sin2a, alpha = 0.0, 0.0, 0.0

        ca, sa = math.cos(alpha), math.sin(alpha)
        rx_c = xm_i - plane_xc[pl_i]
        ry_c = ym_i - plane_yc[pl_i]
        if ca * rx_c + sa * ry_c < 0.0:
            ca, sa = -ca, -sa

        sig_par2 = (px2 * ca * ca + py2 * sa * sa) / (12.0 * max(1.0, float(m_h))) + 0.20 * lam1
        sig_prp2 = (px2 * sa * sa + py2 * ca * ca) / (12.0 * max(1.0, float(m_h))) + 0.12 * lam2
        sxx = sig_par2 * ca * ca + sig_prp2 * sa * sa
        syy = sig_par2 * sa * sa + sig_prp2 * ca * ca
        sxy = (sig_par2 - sig_prp2) * ca * sa
        cov_2x2[out_i] = (sxx, syy, sxy)

        skew_x = (float(np.mean(dx_h ** 3)) / (vx ** 1.5)) if m_h >= 2 else 0.0
        skew_y = (float(np.mean(dy_h ** 3)) / (vy ** 1.5)) if m_h >= 2 else 0.0
        skew_norm = min(5.0, math.hypot(skew_x, skew_y)) / 5.0

        if m_h >= 2:
            u = dx_h * ca + dy_h * sa
            gamma1_par = float(np.mean(u ** 3)) / (lam1 ** 1.5)
            kappa_par = float(np.mean(u ** 4)) / (lam1 ** 2) - 3.0
            b_split = float(_sigmoid_np((-0.75 - kappa_par) / 0.30))
        else:
            gamma1_par = 0.0
            kappa_par = 0.0
            b_split = 0.0

        dx_raw = float(dx_h.max() - dx_h.min())
        dy_raw = float(dy_h.max() - dy_h.min())
        dx_raw_arr[out_i] = dx_raw
        dy_raw_arr[out_i] = dy_raw
        dx_box = dx_raw + contract.dx_mm
        dy_box = dy_raw + contract.dy_mm
        fill_f = min(1.0, float(m_h) / max(1.0, (dx_box / contract.dx_mm) * (dy_box / contract.dy_mm)))
        eta_neck = max(0.0, min(1.0, 1.0 - fill_f)) if m_h >= 3 else 0.0
        dr_c = math.hypot(rx_c, ry_c)

        # [0:22] exact PIX_FEAT_DIM=22 alignment
        feat[out_i, 0] = (xm_i - x_c) / 135.0
        feat[out_i, 1] = (ym_i - y_c) / 100.0
        feat[out_i, 2] = (zm_i - z_mid) / 1000.0
        feat[out_i, 3] = math.log1p(m_h) / 6.0
        feat[out_i, 4] = math.asinh(dx_box / 0.1) / 3.0
        feat[out_i, 5] = math.asinh(dy_box / 0.1) / 3.0
        feat[out_i, 6] = fill_f
        feat[out_i, 7] = math.asinh(math.sqrt(lam1) / 0.05) / 3.0
        feat[out_i, 8] = math.asinh(math.sqrt(lam2) / 0.05) / 3.0
        feat[out_i, 9] = ecc
        feat[out_i, 10] = cos2a
        feat[out_i, 11] = sin2a
        feat[out_i, 12] = skew_norm
        feat[out_i, 13] = math.log1p(k_c1[kp]) / 4.0
        feat[out_i, 14] = math.log1p(k_c5[kp]) / 5.0
        feat[out_i, 15] = math.log1p(k_h5[kp]) / 7.0
        feat[out_i, 16] = math.asinh(rx_c / 0.10) / 4.0
        feat[out_i, 17] = math.asinh(ry_c / 0.10) / 4.0
        feat[out_i, 18] = math.asinh(dr_c / 0.08) / 4.0
        feat[out_i, 19] = math.log1p(k_c015[kp]) / 3.0
        feat[out_i, 20] = math.log1p(k_h015[kp]) / 5.0
        feat[out_i, 21] = math.log1p(k_c040[kp]) / 4.0

        # [22:30] Gen4 high-order morphology, persistence & rotated 2x2 covariance
        feat[out_i, 22] = math.tanh(gamma1_par / 2.0)
        feat[out_i, 23] = math.tanh(kappa_par / 3.0)
        feat[out_i, 24] = b_split
        feat[out_i, 25] = eta_neck
        feat[out_i, 26] = float(k_persist[kp])
        feat[out_i, 27] = sxx
        feat[out_i, 28] = syy
        feat[out_i, 29] = sxy

        # [30:36] Track-B 10085-10088 continuous dE/dx & double-MIP conversion features
        if has_edep:
            q_tot_mip = cl_edep_kev[rk] / PIXEL_MIP_KEV
            feat[out_i, 30] = math.log1p(max(0.0, q_tot_mip))
            conv_arg = (q_tot_mip / max(1.0, 0.70 * float(m_h)) - 1.65) / 0.25
            feat[out_i, 31] = float(_sigmoid_np(conv_arg))
            feat[out_i, 32] = q_max_kev[rk] / max(1e-6, cl_edep_kev[rk])
            feat[out_i, 33] = (xm_i - x_geom_mean[rk]) / contract.dx_mm
            feat[out_i, 34] = (ym_i - y_geom_mean[rk]) / contract.dy_mm
            feat[out_i, 35] = math.log1p(max(0.0, q_tot_mip) / math.sqrt(lam1 / px2 + 1.0))

    # 32x32 micro-patch rasterization (up to 48 target + 16 IPT) stored as compact uint8
    sel_plane = cl_plane[sel_raw]
    sel_nhits = n_hits[sel_raw]
    trig = (sel_nhits >= PATCH_TRIGGER_MIN_HITS) & (np.maximum(dx_raw_arr, dy_raw_arr) > PATCH_TRIGGER_MIN_SPAN_MM)
    trig_tgt = np.flatnonzero(trig & (sel_plane < n_tgt_planes))
    if len(trig_tgt) > N_MAX_TARGET_PATCHES_GEN4:
        rank_t = np.argsort(-sel_nhits[trig_tgt], kind='mergesort')
        trig_tgt = np.sort(trig_tgt[rank_t[:N_MAX_TARGET_PATCHES_GEN4]])
    trig_ipt = np.flatnonzero(trig & (sel_plane >= n_tgt_planes))
    if len(trig_ipt) > N_MAX_IPT_PATCHES_GEN4:
        rank_i = np.argsort(-sel_nhits[trig_ipt], kind='mergesort')
        trig_ipt = np.sort(trig_ipt[rank_i[:N_MAX_IPT_PATCHES_GEN4]])

    p_t, idx_t = _rasterize_gen4_patches(
        order, offsets, pxyz, pedep, x_mean, y_mean, trig_tgt, sel_raw, contract.dx_mm, contract.dy_mm
    )
    p_i, idx_i = _rasterize_gen4_patches(
        order, offsets, pxyz, pedep, x_mean, y_mean, trig_ipt, sel_raw, contract.dx_mm, contract.dy_mm
    )
    if len(idx_t) and len(idx_i):
        patches_u8 = np.concatenate([p_t, p_i], axis=0)
        patch_cl_idx = np.concatenate([idx_t, idx_i], axis=0)
    elif len(idx_t):
        patches_u8, patch_cl_idx = p_t, idx_t
    else:
        patches_u8, patch_cl_idx = p_i, idx_i

    return {
        'n_raw_hits': int(len(pcid)),
        'n_raw_clusters': int(n_cl),
        'n_kept_clusters': int(len(kept_cl_ids)),
        'n_truncated_clusters': int(n_sel),
        'has_edep': bool(has_edep),
        'dark_noise_detected': bool(dark_detected),
        'pix_xyz_mm': np.column_stack([x_mean[sel_raw], y_mean[sel_raw], z_cl[sel_raw]]).astype(np.float32),
        'pix_plane': sel_plane.astype(np.int16),
        'pix_feat': feat,
        'pix_cov_2x2': cov_2x2,
        'pix_persist_prob': persist_prob[sel_raw],
        'pix_first_cid': cl_fcid[sel_raw],
        'pix_plane_summary': plane_summary,
        'pix_patches_u8': patches_u8,
        'patch_cluster_idx': patch_cl_idx.astype(np.int64),
        'c_obs_clean': dom_clean['c_obs'],
        'obs_domain_id_clean': int(dom_clean['obs_domain_id']),
        'cal_start_clean': dom_clean['cal_start'],
        'raw_keep_mask_clusters': keep_cluster,
        'raw_keep_mask_hits': keep_hit_mask,
        'raw_cluster_plane': cl_plane.astype(np.int16),
        'raw_cluster_nhits': n_hits,
        'raw_cluster_xyz': np.column_stack([x_mean, y_mean, z_cl]),
        'hit_order': order,
        'cluster_offsets': offsets,
        'sel_raw_clusters': sel_raw.astype(np.int64),
    }


# =============================================================================
# 2. 54-Station Inverse Digitization, Renyi Time-Walk & Cauchy-Moliere Desaturation
# =============================================================================

def _fit_cauchy_moliere_panel(
    u_centers: np.ndarray,
    pe_lin: np.ndarray,
    is_sat: np.ndarray,
    bar_pitch: float,
    r_m: float = RM_EFF_BC408_W_MM,
) -> tuple[float, np.ndarray]:
    """Fit sub-mm centroid u_c and desaturated linear PE across a single 40-bar panel."""
    if len(u_centers) == 0:
        return 0.0, pe_lin.copy()
    tot_raw = float(np.sum(pe_lin))
    if tot_raw <= 0.0:
        return float(np.mean(u_centers)), pe_lin.copy()

    u_init = float(np.dot(u_centers, pe_lin) / tot_raw)
    unsat = ~is_sat
    half_b = 0.5 * bar_pitch
    u_grid = np.linspace(u_init - 1.5 * bar_pitch, u_init + 1.5 * bar_pitch, 13)
    best_u = u_init
    best_pe_desat = pe_lin.copy()

    if unsat.any():
        u_u = u_centers[unsat]
        y_u = pe_lin[unsat]
        phi_u = (
            np.arctan((u_u[None, :] + half_b - u_grid[:, None]) / r_m)
            - np.arctan((u_u[None, :] - half_b - u_grid[:, None]) / r_m)
        ) / math.pi
        phi_u = np.maximum(phi_u, 1e-6)
        w_u = np.maximum(y_u, 0.5)
        amp_u = np.sum(w_u[None, :] * phi_u * y_u[None, :], axis=1) / np.maximum(
            np.sum(w_u[None, :] * phi_u * phi_u, axis=1), 1e-9
        )
        if is_sat.any():
            u_s = u_centers[is_sat]
            y_s = pe_lin[is_sat]
            phi_s = (
                np.arctan((u_s[None, :] + half_b - u_grid[:, None]) / r_m)
                - np.arctan((u_s[None, :] - half_b - u_grid[:, None]) / r_m)
            ) / math.pi
            amp_floor = np.max(y_s[None, :] / np.maximum(phi_s, 1e-4), axis=1)
            amp = np.maximum(amp_u, amp_floor)
        else:
            amp = amp_u

        pred_u = amp[:, None] * phi_u
        loss = np.sum((np.log1p(pred_u) - np.log1p(y_u[None, :])) ** 2 * np.sqrt(w_u[None, :]), axis=1)
        idx_best = int(np.argmin(loss))
        best_u = float(u_grid[idx_best])
        best_amp = float(amp[idx_best])

        if is_sat.any():
            phi_all = (
                np.arctan((u_centers + half_b - best_u) / r_m)
                - np.arctan((u_centers - half_b - best_u) / r_m)
            ) / math.pi
            best_pe_desat = np.where(is_sat, np.maximum(pe_lin, best_amp * phi_all), pe_lin)
    else:
        best_pe_desat = pe_lin * 1.35

    return best_u, best_pe_desat


def reconstruct_calorimeter_54_stations(
    observation: dict[str, Any],
    contract: Gen4GeometryContract,
    v_ref_mm: np.ndarray | None = None,
    *,
    has_subthr_readout: bool | None = None,
    s_mir: int = 1,
) -> dict[str, Any]:
    """54-station inverse digitization, Cauchy-Moliere desaturation & Renyi time-walk correction.

    Unlocks `<2.5 PE` halo bars and `bar.hit_time_ns` directly from `observations['bar']`,
    and computes sparse per-bar 10-channel attributes matching `scint_grid: (B, 54, 2, 40, 10)`.
    """
    bar = observation['bar']
    n_st = contract.n_stations
    n_dbl = contract.n_doublets
    n_bars = contract.n_bars

    grid_x_e = np.zeros((n_st, n_bars), dtype=np.float32)
    grid_y_e = np.zeros((n_dbl, n_bars), dtype=np.float32)
    grid_x_prompt = np.zeros((n_st, n_bars), dtype=np.float32)
    grid_y_prompt = np.zeros((n_dbl, n_bars), dtype=np.float32)

    st_x_cent = np.full(n_st, contract.x_scint_center, dtype=np.float32)
    st_y_cent = np.full(n_st, contract.y_scint_center, dtype=np.float32)
    st_has_x = np.zeros(n_st, dtype=bool)
    st_has_y = np.zeros(n_st, dtype=bool)

    bpe_raw = np.asarray(bar.get('pe_proxy', ()), dtype=np.float64)
    if len(bpe_raw) == 0 or not np.any(bpe_raw > 0.0):
        return {
            'has_subthr_readout': bool(has_subthr_readout if has_subthr_readout is not None else False),
            'n_active_bars': 0,
            'n_subthr_bars': 0,
            'n_saturated_bars': 0,
            't0_ns': 0.0,
            'total_desat_mev': 0.0,
            'total_prompt_mev': 0.0,
            'total_delayed_mev': 0.0,
            'bar_station': np.zeros(0, dtype=np.int8),
            'bar_view': np.zeros(0, dtype=np.int8),
            'bar_idx': np.zeros(0, dtype=np.int8),
            'bar_feat_10': np.zeros((0, SCINT_GRID_CHANNELS), dtype=np.float32),
            'grid_x_e_mev': grid_x_e,
            'grid_y_e_mev': grid_y_e,
            'grid_x_prompt_mev': grid_x_prompt,
            'grid_y_prompt_mev': grid_y_prompt,
            'st_x_cent_mm': st_x_cent,
            'st_y_cent_mm': st_y_cent,
            'st_has_x': st_has_x,
            'st_has_y': st_has_y,
        }

    keep = bpe_raw > 0.0
    bpe = bpe_raw[keep]
    bp = np.asarray(bar['panel'], dtype=np.int64)[keep]
    bh = np.asarray(bar['horizontal'], dtype=np.int64)[keep]
    bi = np.asarray(bar['bar'], dtype=np.int64)[keep]
    bht = (
        np.asarray(bar['hit_time_ns'], dtype=np.float64)[keep]
        if 'hit_time_ns' in bar and len(bar['hit_time_ns']) == len(bpe_raw)
        else np.full(len(bpe), -1.0, dtype=np.float64)
    )
    n_b = len(bpe)
    n_subthr = int(np.count_nonzero(bpe < 2.5))
    subthr_flag = bool(has_subthr_readout if has_subthr_readout is not None else (n_subthr > 0))

    # Three-regime inverse digitization
    is_sub = bpe < 2.5
    is_sat = bpe >= 13800.0
    frac = np.clip(bpe / SIPM_NPIX, 0.0, 14399.0 / SIPM_NPIX)
    n_raw = np.where(is_sub, bpe, -SIPM_NPIX * np.log(1.0 - frac))
    one_m_f = np.maximum(1.0 - frac, 1.0 / SIPM_NPIX)
    sigma_lne = np.where(
        is_sub,
        1.0 / np.sqrt(np.maximum(bpe, 0.5)),
        np.sqrt(
            1.0 / (np.maximum(n_raw, 1.0) * (one_m_f ** 2))
            + (1.0 / 12.0) / ((np.maximum(n_raw, 1.0) * one_m_f) ** 2)
        ),
    )
    sigma_lne = np.where(is_sat, np.maximum(sigma_lne, 0.35), sigma_lne)

    st_arr = contract.panel_to_station[bp]
    x_slot = contract.panel_to_xslot[bp]
    y_slot = contract.panel_to_yslot[bp]
    n_desat = n_raw.copy()
    subbar_offset = np.zeros(n_b, dtype=np.float64)
    half_pitch = 0.5 * contract.bar_pitch_mm  # 5.25 mm

    for p_id in np.unique(bp):
        pm = bp == p_id
        is_h = int(bh[pm][0])
        u_c = contract.bar_x_centers_mm[bi[pm]] if is_h == 0 else contract.bar_y_centers_mm[bi[pm]]
        u_hat, pe_desat_p = _fit_cauchy_moliere_panel(
            u_c, n_raw[pm], is_sat[pm], contract.bar_pitch_mm, r_m=RM_EFF_BC408_W_MM
        )
        n_desat[pm] = pe_desat_p
        subbar_offset[pm] = np.clip(u_hat - u_c, -half_pitch, half_pitch)
        s_idx = int(contract.panel_to_station[p_id])
        if is_h == 0:
            st_x_cent[s_idx] = float(u_hat)
            st_has_x[s_idx] = True
        else:
            st_y_cent[s_idx] = float(u_hat)
            st_has_y[s_idx] = True

    if st_has_y.any() and (~st_has_y).any():
        act_y_idx = np.flatnonzero(st_has_y)
        z_y = contract.station_z_mm[act_y_idx]
        for s_miss in np.flatnonzero(~st_has_y):
            nearest = act_y_idx[int(np.argmin(np.abs(z_y - contract.station_z_mm[s_miss])))]
            st_y_cent[s_miss] = st_y_cent[nearest]

    bx_est = np.where(bh == 0, contract.bar_x_centers_mm[bi], st_x_cent[st_arr].astype(np.float64))
    by_est = np.where(bh == 1, contract.bar_y_centers_mm[bi], st_y_cent[st_arr].astype(np.float64))
    bz_est = contract.station_z_mm[st_arr]
    d_sipm = contract.optical_distance_to_sipm(bh, bx_est, by_est, s_mir=s_mir)

    att_corr = np.exp(d_sipm / BC408_ATTENUATION_MM)
    e_vis_mev = (n_desat * att_corr) / SIPM_ETA0_PE_PER_MEV

    if v_ref_mm is not None:
        v0 = np.asarray(v_ref_mm, dtype=np.float64).reshape(3)
    else:
        act_s = np.flatnonzero(st_has_x | st_has_y)
        s0 = int(act_s[0]) if len(act_s) else 0
        v0 = np.array([float(st_x_cent[s0]), float(st_y_cent[s0]), float(contract.station_z_mm[s0])], dtype=np.float64)

    n_safe = np.maximum(n_raw, 3.5)
    t_walk = BC408_TAU_SCINT_NS * (1.0 / n_safe + 1.0 / (n_safe - 1.0) + 1.0 / (n_safe - 2.0))
    tof_dist = np.sqrt((bx_est - v0[0]) ** 2 + (by_est - v0[1]) ** 2 + (bz_est - v0[2]) ** 2)
    t_corr = bht - (tof_dist / SPEED_OF_LIGHT_MM_PER_NS) - (d_sipm / WLS_GROUP_VELOCITY_MM_PER_NS) - t_walk

    valid_t = bht >= 0.0
    if valid_t.any():
        core_t = valid_t & (n_raw >= 15.0)
        t_pool = t_corr[core_t] if core_t.any() else t_corr[valid_t]
        t0_ns = float(np.quantile(t_pool, 0.05))
    else:
        t0_ns = 0.0

    dt_res = np.where(valid_t, t_corr - t0_ns, 0.0)
    delayed_frac = np.where(valid_t, _sigmoid_np((dt_res - 2.5) / 0.8), 0.0)
    is_delayed = valid_t & (dt_res > 2.5)
    is_prompt = (~valid_t) | (dt_res <= 2.5)

    # Assemble exact 10-channel sparse bar feature matrix matching Gen4EventBatch `scint_grid`
    bar_feat_10 = np.column_stack([
        e_vis_mev,                                  # [0] e_desat_mev
        sigma_lne,                                  # [1] sigma_log_e
        bpe,                                        # [2] pe_obs (incl <2.5 PE halo)
        np.where(valid_t, bht, -1.0),               # [3] t_raw_ns
        dt_res,                                     # [4] dt_res_ns
        valid_t.astype(np.float64),                 # [5] t_valid
        is_sat.astype(np.float64),                  # [6] is_saturated
        is_sub.astype(np.float64),                  # [7] is_subthr_halo
        subbar_offset,                              # [8] subbar_offset_mm in [-5.25, +5.25]
        delayed_frac,                               # [9] delayed_tail_frac
    ]).astype(np.float32)

    xm = bh == 0
    if xm.any():
        np.add.at(grid_x_e, (x_slot[xm], bi[xm]), e_vis_mev[xm])
        np.add.at(grid_x_prompt, (x_slot[xm], bi[xm]), np.where(is_prompt[xm], e_vis_mev[xm], 0.0))

    ym = bh == 1
    if ym.any():
        np.add.at(grid_y_e, (y_slot[ym], bi[ym]), e_vis_mev[ym])
        np.add.at(grid_y_prompt, (y_slot[ym], bi[ym]), np.where(is_prompt[ym], e_vis_mev[ym], 0.0))

    # Station-averaged energy totals
    e_st_x = grid_x_e.sum(axis=-1)
    e_st_y = np.zeros(n_st, dtype=np.float32)
    e_st_y[contract.s_stereo] = grid_y_e.sum(axis=-1)
    e_tot = float(np.where(contract.station_has_y, 0.5 * (e_st_x + e_st_y), e_st_x).sum())

    ep_x = grid_x_prompt.sum(axis=-1)
    ep_y = np.zeros(n_st, dtype=np.float32)
    ep_y[contract.s_stereo] = grid_y_prompt.sum(axis=-1)
    e_pr = float(np.where(contract.station_has_y, 0.5 * (ep_x + ep_y), ep_x).sum())
    e_dl = max(0.0, e_tot - e_pr)

    return {
        'has_subthr_readout': subthr_flag,
        'n_active_bars': int(n_b),
        'n_subthr_bars': int(n_subthr),
        'n_saturated_bars': int(np.count_nonzero(is_sat)),
        't0_ns': float(t0_ns),
        'total_desat_mev': e_tot,
        'total_prompt_mev': e_pr,
        'total_delayed_mev': e_dl,
        'bar_station': st_arr.astype(np.int8),
        'bar_view': bh.astype(np.int8),
        'bar_idx': bi.astype(np.int8),
        'bar_feat_10': bar_feat_10,
        'grid_x_e_mev': grid_x_e,
        'grid_y_e_mev': grid_y_e,
        'grid_x_prompt_mev': grid_x_prompt,
        'grid_y_prompt_mev': grid_y_prompt,
        'st_x_cent_mm': st_x_cent,
        'st_y_cent_mm': st_y_cent,
        'st_has_x': st_has_x,
        'st_has_y': st_has_y,
    }


# =============================================================================
# 3. Four-Modal UnifiedMuonKalman (M1 - M4)
# =============================================================================

MUON_MODE_NONE = 0
MUON_MODE_M1_BORE3 = 1
MUON_MODE_M2_IPT_M0_TRK1 = 2
MUON_MODE_M3_SHELL = 3
MUON_MODE_M4_RANGE_MCS = 4


def _extract_ipt_rays(
    pix_xyz_mm: np.ndarray,
    pix_plane: np.ndarray,
    contract: Gen4GeometryContract,
    *,
    max_rays: int = 16,
    max_slope: float = 0.25,
) -> list[dict[str, float]]:
    """Build precision 3-plane & 2-plane rays across IPT silicon planes 12, 13, 14 (0.14 mrad)."""
    if len(pix_xyz_mm) == 0:
        return []
    ipt_m = pix_plane >= contract.n_target_planes
    if not ipt_m.any():
        return []
    xyz = pix_xyz_mm[ipt_m].astype(np.float64)
    pl = pix_plane[ipt_m].astype(np.int64) - contract.n_target_planes
    idx_by_p = [np.flatnonzero(pl == k) for k in range(3)]
    rays: list[dict[str, float]] = []

    z12, z13, z14 = float(contract.pixel_Zpos[12]), float(contract.pixel_Zpos[13]), float(contract.pixel_Zpos[14])
    sig_x2 = (contract.dx_mm ** 2) / 12.0 + 1e-6
    sig_y2 = (contract.dy_mm ** 2) / 12.0 + 1e-6

    if all(len(idx_by_p[k]) > 0 for k in (0, 1, 2)):
        for i0 in idx_by_p[0][:24]:
            x0, y0 = xyz[i0, 0], xyz[i0, 1]
            for i2 in idx_by_p[2][:24]:
                tx = (xyz[i2, 0] - x0) / (z14 - z12)
                ty = (xyz[i2, 1] - y0) / (z14 - z12)
                if math.hypot(tx, ty) > max_slope:
                    continue
                xm = x0 + tx * (z13 - z12)
                ym = y0 + ty * (z13 - z12)
                dx_m = xyz[idx_by_p[1], 0] - xm
                dy_m = xyz[idx_by_p[1], 1] - ym
                chi2_m = (dx_m * dx_m) / (1.5 * sig_x2 + 1e-4) + (dy_m * dy_m) / (1.5 * sig_y2 + 1e-4)
                best_k = int(np.argmin(chi2_m))
                if chi2_m[best_k] <= 25.0:
                    i1 = idx_by_p[1][best_k]
                    x_ref = (x0 + xyz[i1, 0] + xyz[i2, 0]) / 3.0
                    y_ref = (y0 + xyz[i1, 1] + xyz[i2, 1]) / 3.0
                    rays.append({
                        'n_hits': 3.0,
                        'z_ref': z13,
                        'x_ref': x_ref,
                        'y_ref': y_ref,
                        'tx': tx,
                        'ty': ty,
                        'sigma_y_mm': math.sqrt(sig_y2 / 3.0),
                        'sigma_ty': math.sqrt(2.0 * sig_y2) / (z14 - z12),
                        'sigma_x_mm': math.sqrt(sig_x2 / 3.0),
                        'sigma_tx': math.sqrt(2.0 * sig_x2) / (z14 - z12),
                        'chi2': float(chi2_m[best_k]),
                    })

    if len(rays) < max_rays:
        for pa, pb in ((0, 2), (0, 1), (1, 2)):
            if len(idx_by_p[pa]) == 0 or len(idx_by_p[pb]) == 0:
                continue
            za = float(contract.pixel_Zpos[contract.n_target_planes + pa])
            zb = float(contract.pixel_Zpos[contract.n_target_planes + pb])
            dz = zb - za
            for ia in idx_by_p[pa][:16]:
                for ib in idx_by_p[pb][:16]:
                    tx = (xyz[ib, 0] - xyz[ia, 0]) / dz
                    ty = (xyz[ib, 1] - xyz[ia, 1]) / dz
                    if math.hypot(tx, ty) > max_slope:
                        continue
                    x_ref = xyz[ia, 0] + tx * (z13 - za)
                    y_ref = xyz[ia, 1] + ty * (z13 - za)
                    rays.append({
                        'n_hits': 2.0,
                        'z_ref': z13,
                        'x_ref': x_ref,
                        'y_ref': y_ref,
                        'tx': tx,
                        'ty': ty,
                        'sigma_y_mm': math.sqrt(sig_y2),
                        'sigma_ty': math.sqrt(2.0 * sig_y2) / dz,
                        'sigma_x_mm': math.sqrt(sig_x2),
                        'sigma_tx': math.sqrt(2.0 * sig_x2) / dz,
                        'chi2': 2.0,
                    })

    rays.sort(key=lambda r: (-r['n_hits'], r['chi2'], r['tx'] ** 2 + r['ty'] ** 2))
    return rays[:max_rays]


def _compute_cal_iso_gev(
    x_ref: float,
    y_ref: float,
    z_ref: float,
    tx: float,
    ty: float,
    cal_54: dict[str, Any],
    contract: Gen4GeometryContract,
    *,
    cone_mm: float = 25.0,
) -> float:
    """Calorimeter isolation energy (GeV) around extrapolated muon corridor in last 12 stations."""
    gx = cal_54['grid_x_prompt_mev']
    total_mev = 0.0
    for s in range(max(0, contract.n_stations - 12), contract.n_stations):
        zs = float(contract.station_z_mm[s])
        xs = x_ref + tx * (zs - z_ref)
        act = np.flatnonzero(gx[s] > 0.0)
        if len(act) == 0:
            continue
        dx = np.abs(contract.bar_x_centers_mm[act] - xs)
        in_cone = (dx > 12.0) & (dx <= cone_mm)
        if in_cone.any():
            total_mev += float(gx[s, act[in_cone]].sum())
    return total_mev / 1000.0


def _fit_m1_m2_m3_spectrometer(
    tracker_obs: dict[str, Any],
    ipt_rays: list[dict[str, float]],
    cal_54: dict[str, Any],
    contract: Gen4GeometryContract,
    *,
    s_mir: int = 1,
) -> list[dict[str, Any]]:
    """Unified M1 (3-station bore), M2 (IPT+M0+Trk1/2 soft muon), and M3 (shell scatter) solver."""
    xy = np.asarray(tracker_obs.get('xy_mm', ()), dtype=np.float64).reshape(-1, 2)
    st = np.asarray(tracker_obs.get('station', ()), dtype=np.int64)
    if len(st) == 0:
        return []

    bore_y = COMMON_Y_AXIS_MM + (MAGNET_BORE_Y_MM - COMMON_Y_AXIS_MM) * float(s_mir)
    idx_s = [np.flatnonzero(st == k) for k in (0, 1, 2)]
    cands: list[dict[str, Any]] = []

    def _m0_bore_check(x_ipt: float, y_ipt: float, tx: float, ty_ipt: float, kappa: float) -> tuple[bool, float]:
        max_r = 0.0
        for z_s in (MAGNET0_Z_MIN_MM, MAGNET0_Z_CENTER_MM, MAGNET0_Z_MAX_MM):
            xs = x_ipt + tx * (z_s - IPT_Z_REF_MM)
            u = max(0.0, z_s - MAGNET0_Z_MIN_MM)
            ys = y_ipt + ty_ipt * (z_s - IPT_Z_REF_MM) + 0.5 * kappa * (u * u)
            max_r = max(max_r, math.hypot(xs - MAGNET_BORE_X_MM, ys - bore_y))
        return max_r <= MAGNET_BORE_FID_R_MM, max_r

    used_s0, used_s1, used_s2 = set(), set(), set()
    if all(len(idx_s[k]) > 0 for k in (0, 1, 2)):
        g0, g1, g2 = np.meshgrid(idx_s[0][:14], idx_s[1][:14], idx_s[2][:14], indexing='ij')
        g0, g1, g2 = g0.ravel(), g1.ravel(), g2.ravel()
        x0, y0 = xy[g0, 0], xy[g0, 1]
        x1, y1 = xy[g1, 0], xy[g1, 1]
        x2, y2 = xy[g2, 0], xy[g2, 1]
        dz12 = TRACKER_Z_MM[1] - TRACKER_Z_MM[0]
        s_y = y1 - 0.5 * (y0 + y2)
        kappa = -s_y / SAGITTA_INTEGRAL_MM2
        dx = x1 - 0.5 * (x0 + x2)
        chi2_x = (dx / SIGMA_DELTA_X_MM) ** 2
        tx = (x2 - x0) / (2.0 * dz12)
        ty_entry = (y1 - y0) / dz12 + ENTRY_SLOPE_SAGITTA_COEFF * (s_y / dz12)
        y0_b = y0 - (bore_y - MAGNET_BORE_Y_MM)
        y1_b = y1 - (bore_y - MAGNET_BORE_Y_MM)
        in_bore12, _ = check_magnet12_bore_traversal(x0, y0_b, y1_b, tx, ty_entry, kappa)
        q_score = chi2_x + np.where(in_bore12, 0.0, 25.0) + (np.abs(s_y) / 25.0) ** 2
        order = np.argsort(q_score, kind='mergesort')

        for k in order:
            a, b, c = int(g0[k]), int(g1[k]), int(g2[k])
            if a in used_s0 or b in used_s1 or c in used_s2:
                continue
            if chi2_x[k] > 36.0 and in_bore12[k]:
                continue
            used_s0.add(a)
            used_s1.add(b)
            used_s2.add(c)

            best_ipt = None
            best_ipt_d = 1e9
            y_at_ipt_pred, _ = back_extrapolate_y_through_magnet0(
                np.array([y0[k]]), np.array([ty_entry[k]]), np.array([kappa[k] if in_bore12[k] else 0.0]), IPT_Z_REF_MM
            )
            x_at_ipt_pred = float(x0[k] + tx[k] * (IPT_Z_REF_MM - TRACKER_Z_MM[0]))
            for ray in ipt_rays:
                d_ipt = math.hypot((ray['x_ref'] - x_at_ipt_pred) / 3.0, (ray['y_ref'] - float(y_at_ipt_pred[0])) / 1.5)
                if d_ipt < best_ipt_d and d_ipt < 15.0:
                    best_ipt_d = d_ipt
                    best_ipt = ray

            if in_bore12[k]:
                q_over_p_3st = float(kappa[k]) * 1000.0 / KAPPA_B_CONST_MEV_MM
                sig_qp_3st = (SIGMA_SAGITTA_Y_MM * 1000.0) / SAGITTA_CONST_MEV_MM
                if best_ipt is not None:
                    dy_m0 = float(y0[k]) - (best_ipt['y_ref'] + best_ipt['ty'] * (TRACKER_Z_MM[0] - best_ipt['z_ref']))
                    q_over_p_m0 = (dy_m0 * 1000.0) / MAGNET0_BEND_CONST_MEV_MM
                    sig_y_m0 = math.sqrt(
                        SIGMA_Y_MM ** 2
                        + best_ipt['sigma_y_mm'] ** 2
                        + ((TRACKER_Z_MM[0] - best_ipt['z_ref']) * best_ipt['sigma_ty']) ** 2
                    )
                    sig_qp_m0 = (sig_y_m0 * 1000.0) / MAGNET0_BEND_CONST_MEV_MM
                    w1 = 1.0 / (sig_qp_3st ** 2)
                    w2 = 1.0 / (sig_qp_m0 ** 2)
                    q_over_p = (w1 * q_over_p_3st + w2 * q_over_p_m0) / (w1 + w2)
                    sig_qp = 1.0 / math.sqrt(w1 + w2)
                else:
                    q_over_p = q_over_p_3st
                    sig_qp = sig_qp_3st
                p_gev = min(5000.0, 1.0 / max(abs(q_over_p), 1e-4))
                sig_p = p_gev * p_gev * sig_qp
                q_sign = 1 if q_over_p > 0.0 else (-1 if q_over_p < 0.0 else 0)
                iso_gev = _compute_cal_iso_gev(float(x0[k]), float(y0[k]), TRACKER_Z_MM[0], float(tx[k]), float(ty_entry[k]), cal_54, contract)
                cands.append({
                    'mode': MUON_MODE_M1_BORE3,
                    'q_over_p': float(q_over_p),
                    'sigma_q_over_p': float(sig_qp),
                    'p_gev': float(p_gev),
                    'sigma_p_gev': float(min(sig_p, 5000.0)),
                    'q_mu': int(q_sign),
                    'q_valid_prob': 1.0,
                    'in_bore': True,
                    'm0_in_bore': True,
                    'chi2_ndf': float(chi2_x[k]),
                    'x_ref': float(x0[k]),
                    'y_ref': float(y0[k]),
                    'tx': float(tx[k]),
                    'ty': float(ty_entry[k]),
                    'iso_energy_gev': float(iso_gev),
                })
            else:
                if best_ipt is not None:
                    dy_m0 = float(y0[k]) - (best_ipt['y_ref'] + best_ipt['ty'] * (TRACKER_Z_MM[0] - best_ipt['z_ref']))
                    q_over_p_m0 = (dy_m0 * 1000.0) / MAGNET0_BEND_CONST_MEV_MM
                    m0_ok, _ = _m0_bore_check(best_ipt['x_ref'], best_ipt['y_ref'], best_ipt['tx'], best_ipt['ty'], q_over_p_m0 * KAPPA_B_CONST_MEV_MM / 1000.0)
                    sig_y_m0 = math.sqrt(
                        SIGMA_Y_MM ** 2
                        + best_ipt['sigma_y_mm'] ** 2
                        + ((TRACKER_Z_MM[0] - best_ipt['z_ref']) * best_ipt['sigma_ty']) ** 2
                    )
                    sig_qp = (sig_y_m0 * 1000.0) / MAGNET0_BEND_CONST_MEV_MM
                    p_gev = min(5000.0, 1.0 / max(abs(q_over_p_m0), 1e-4))
                    q_sign = (1 if q_over_p_m0 > 0 else -1) if m0_ok else 0
                    q_over_p = q_over_p_m0 if m0_ok else 0.0
                    q_vp = 0.95 if m0_ok else 0.50
                else:
                    p_gev = min(2000.0, max(2.0, SAGITTA_CONST_MEV_MM / max(abs(float(s_y[k])) * 1000.0, 0.05)))
                    q_over_p = 0.0
                    sig_qp = 0.25 / max(p_gev, 1.0)
                    q_sign = 0
                    m0_ok = False
                    q_vp = 0.50
                sig_p = p_gev * p_gev * sig_qp
                iso_gev = _compute_cal_iso_gev(float(x0[k]), float(y0[k]), TRACKER_Z_MM[0], float(tx[k]), float(ty_entry[k]), cal_54, contract)
                cands.append({
                    'mode': MUON_MODE_M3_SHELL,
                    'q_over_p': float(q_over_p),
                    'sigma_q_over_p': float(sig_qp),
                    'p_gev': float(p_gev),
                    'sigma_p_gev': float(min(sig_p, 5000.0)),
                    'q_mu': int(q_sign),
                    'q_valid_prob': float(q_vp),
                    'in_bore': False,
                    'm0_in_bore': bool(m0_ok),
                    'chi2_ndf': float(chi2_x[k]),
                    'x_ref': float(x0[k]),
                    'y_ref': float(y0[k]),
                    'tx': float(tx[k]),
                    'ty': float(ty_entry[k]),
                    'iso_energy_gev': float(iso_gev),
                })
            if len(cands) >= MUON_TRACK_SLOTS:
                break

    # --- M2: IPT + Magnet-0 + Tracker-1 / (1,2) Soft-Muon 221.1 GeV*mm Bending Arm ---
    rem_s0 = [int(i) for i in idx_s[0] if int(i) not in used_s0]
    if ipt_rays and rem_s0 and len(cands) < MUON_TRACK_SLOTS:
        dz_ipt_t1 = TRACKER_Z_MM[0] - IPT_Z_REF_MM  # 1752.265 mm
        m2_pairs: list[tuple[float, dict[str, float], int, float, float, float]] = []
        for ray in ipt_rays:
            x_pred = ray['x_ref'] + ray['tx'] * dz_ipt_t1
            y_lin_pred = ray['y_ref'] + ray['ty'] * dz_ipt_t1
            sig_dx = math.sqrt(SIGMA_X_MM ** 2 + ray['sigma_x_mm'] ** 2 + (dz_ipt_t1 * ray['sigma_tx']) ** 2 + 0.5 ** 2)
            for i0 in rem_s0:
                dx = float(xy[i0, 0]) - x_pred
                chi2_x_m2 = (dx / sig_dx) ** 2
                if chi2_x_m2 > 16.0:
                    continue
                dy_m0 = float(xy[i0, 1]) - y_lin_pred
                kappa_m2 = dy_m0 / MAGNET0_BEND_INTEGRAL_MM2
                m0_ok, _ = _m0_bore_check(ray['x_ref'], ray['y_ref'], ray['tx'], ray['ty'], kappa_m2)
                if not m0_ok:
                    continue
                q_over_p = (dy_m0 * 1000.0) / MAGNET0_BEND_CONST_MEV_MM
                p_est = 1.0 / max(abs(q_over_p), 1e-4)
                if p_est < 1.5 or p_est > 500.0:
                    continue
                th_ms_ipt = _highland_sigma_rad(0.07, p_est * 1000.0)
                sig_y_tot = math.sqrt(
                    SIGMA_Y_MM ** 2
                    + ray['sigma_y_mm'] ** 2
                    + (dz_ipt_t1 * math.hypot(ray['sigma_ty'], th_ms_ipt)) ** 2
                )
                sig_qp = (sig_y_tot * 1000.0) / MAGNET0_BEND_CONST_MEV_MM
                score = chi2_x_m2 + 0.02 * abs(dy_m0)
                m2_pairs.append((score, ray, i0, q_over_p, sig_qp, dy_m0))

        m2_pairs.sort(key=lambda r: r[0])
        for score, ray, i0, q_over_p, sig_qp, dy_m0 in m2_pairs:
            if i0 in used_s0:
                continue
            used_s0.add(i0)
            p_gev = min(5000.0, 1.0 / max(abs(q_over_p), 1e-4))
            sig_p = min(5000.0, p_gev * p_gev * sig_qp)
            q_sign = 1 if q_over_p > 0.0 else (-1 if q_over_p < 0.0 else 0)
            kappa_m2 = dy_m0 / MAGNET0_BEND_INTEGRAL_MM2
            ty_exit = ray['ty'] + kappa_m2 * MAGNET0_LENGTH_MM
            iso_gev = _compute_cal_iso_gev(float(xy[i0, 0]), float(xy[i0, 1]), TRACKER_Z_MM[0], ray['tx'], ty_exit, cal_54, contract)
            cands.append({
                'mode': MUON_MODE_M2_IPT_M0_TRK1,
                'q_over_p': float(q_over_p),
                'sigma_q_over_p': float(sig_qp),
                'p_gev': float(p_gev),
                'sigma_p_gev': float(sig_p),
                'q_mu': int(q_sign),
                'q_valid_prob': 1.0,
                'in_bore': True,
                'm0_in_bore': True,
                'chi2_ndf': float(score),
                'x_ref': float(xy[i0, 0]),
                'y_ref': float(xy[i0, 1]),
                'tx': float(ray['tx']),
                'ty': float(ty_exit),
                'iso_energy_gev': float(iso_gev),
            })
            if len(cands) >= MUON_TRACK_SLOTS:
                break

    # M2 fallback for Tracker 1 (station index 1 at z=1237.4 mm) when Tracker 0 is dead/missed
    rem_s1 = [int(i) for i in idx_s[1] if int(i) not in used_s1]
    if ipt_rays and rem_s1 and len(cands) < MUON_TRACK_SLOTS:
        dz_ipt_t2 = TRACKER_Z_MM[1] - IPT_Z_REF_MM
        m01_int_mm2 = (
            0.5 * (MAGNET0_LENGTH_MM ** 2)
            + MAGNET0_LENGTH_MM * (TRACKER_Z_MM[1] - MAGNET0_Z_MAX_MM)
            + 0.5 * (1000.0 ** 2)
            + 1000.0 * (TRACKER_Z_MM[1] - MAGNET1_Z_MAX_MM)
        )  # 3,679,050.0 mm^2
        m01_const_mev_mm = m01_int_mm2 * KAPPA_B_CONST_MEV_MM
        for ray in ipt_rays:
            x_pred = ray['x_ref'] + ray['tx'] * dz_ipt_t2
            y_lin_pred = ray['y_ref'] + ray['ty'] * dz_ipt_t2
            sig_dx = math.sqrt(SIGMA_X_MM ** 2 + ray['sigma_x_mm'] ** 2 + (dz_ipt_t2 * ray['sigma_tx']) ** 2 + 1.0 ** 2)
            for i1 in rem_s1:
                if i1 in used_s1:
                    continue
                dx = float(xy[i1, 0]) - x_pred
                chi2_x_m2 = (dx / sig_dx) ** 2
                if chi2_x_m2 > 16.0:
                    continue
                dy_m01 = float(xy[i1, 1]) - y_lin_pred
                kappa_m2 = dy_m01 / m01_int_mm2
                m0_ok, _ = _m0_bore_check(ray['x_ref'], ray['y_ref'], ray['tx'], ray['ty'], kappa_m2)
                if not m0_ok:
                    continue
                q_over_p = (dy_m01 * 1000.0) / m01_const_mev_mm
                p_est = 1.0 / max(abs(q_over_p), 1e-4)
                if p_est < 1.5 or p_est > 500.0:
                    continue
                used_s1.add(i1)
                th_ms_ipt = _highland_sigma_rad(0.07, p_est * 1000.0)
                sig_y_tot = math.sqrt(SIGMA_Y_MM ** 2 + ray['sigma_y_mm'] ** 2 + (dz_ipt_t2 * math.hypot(ray['sigma_ty'], th_ms_ipt)) ** 2)
                sig_qp = (sig_y_tot * 1000.0) / m01_const_mev_mm
                p_gev = min(5000.0, p_est)
                ty_exit = ray['ty'] + kappa_m2 * (MAGNET0_LENGTH_MM + 1000.0)
                iso_gev = _compute_cal_iso_gev(float(xy[i1, 0]), float(xy[i1, 1]), TRACKER_Z_MM[1], ray['tx'], ty_exit, cal_54, contract)
                cands.append({
                    'mode': MUON_MODE_M2_IPT_M0_TRK1,
                    'q_over_p': float(q_over_p),
                    'sigma_q_over_p': float(sig_qp),
                    'p_gev': float(p_gev),
                    'sigma_p_gev': float(min(5000.0, p_gev * p_gev * sig_qp)),
                    'q_mu': 1 if q_over_p > 0.0 else -1,
                    'q_valid_prob': 0.95,
                    'in_bore': True,
                    'm0_in_bore': True,
                    'chi2_ndf': float(chi2_x_m2 + 0.02 * abs(dy_m01)),
                    'x_ref': float(xy[i1, 0]),
                    'y_ref': float(xy[i1, 1]),
                    'tx': float(ray['tx']),
                    'ty': float(ty_exit),
                    'iso_energy_gev': float(iso_gev),
                })
                if len(cands) >= MUON_TRACK_SLOTS:
                    break
            if len(cands) >= MUON_TRACK_SLOTS:
                break

    return cands


def _fit_m4_calorimeter_range_mcs(
    cal_54: dict[str, Any],
    v_pv_mm: np.ndarray,
    contract: Gen4GeometryContract,
) -> list[dict[str, Any]]:
    """M4: Stopping CSDA Range + Side-Exiting Station-by-Station Highland MCS momentum meter."""
    gx = cal_54['grid_x_prompt_mev']
    gy = cal_54['grid_y_prompt_mev']
    n_st = contract.n_stations
    st_to_dbl = contract.station_to_doublet

    mip_x_bars: dict[int, list[int]] = {}
    mip_y_bars: dict[int, list[int]] = {}
    for s in range(n_st):
        act_x = np.flatnonzero((gx[s] >= 0.35) & (gx[s] <= 6.5))
        if 1 <= len(act_x) <= 2 and (len(act_x) == 1 or abs(int(act_x[1]) - int(act_x[0])) <= 1):
            if gx[s].sum() <= 25.0 or len(np.flatnonzero(gx[s] >= 0.35)) <= 3:
                mip_x_bars[s] = act_x.tolist()
        d_i = int(st_to_dbl[s])
        if d_i >= 0:
            act_y = np.flatnonzero((gy[d_i] >= 0.35) & (gy[d_i] <= 6.5))
            if 1 <= len(act_y) <= 2 and (len(act_y) == 1 or abs(int(act_y[1]) - int(act_y[0])) <= 1):
                if gy[d_i].sum() <= 25.0 or len(np.flatnonzero(gy[d_i] >= 0.35)) <= 3:
                    mip_y_bars[s] = act_y.tolist()

    if len(mip_x_bars) < 5:
        return []

    sorted_s = sorted(mip_x_bars.keys())
    best_chain: list[tuple[int, float, float | None]] = []
    for i_start in range(len(sorted_s) - 4):
        s0 = sorted_s[i_start]
        x0 = float(np.mean(contract.bar_x_centers_mm[mip_x_bars[s0]]))
        z0 = float(contract.station_z_mm[s0])
        for i_next in range(i_start + 1, min(i_start + 4, len(sorted_s))):
            s1 = sorted_s[i_next]
            x1 = float(np.mean(contract.bar_x_centers_mm[mip_x_bars[s1]]))
            z1 = float(contract.station_z_mm[s1])
            tx = (x1 - x0) / max(z1 - z0, 1.0)
            if abs(tx) > 0.35:
                continue
            chain: list[tuple[int, float, float | None]] = []
            for s_k in sorted_s[i_start:]:
                zk = float(contract.station_z_mm[s_k])
                xp = x0 + tx * (zk - z0)
                bx_cand = mip_x_bars[s_k]
                x_vals = contract.bar_x_centers_mm[bx_cand]
                j_best = int(np.argmin(np.abs(x_vals - xp)))
                if abs(float(x_vals[j_best]) - xp) <= 18.0:
                    yk = float(np.mean(contract.bar_y_centers_mm[mip_y_bars[s_k]])) if s_k in mip_y_bars else None
                    chain.append((s_k, float(x_vals[j_best]), yk))
            if len(chain) > len(best_chain):
                best_chain = chain

    if len(best_chain) < 5:
        return []

    zs = np.array([float(contract.station_z_mm[c[0]]) for c in best_chain], dtype=np.float64)
    xs = np.array([c[1] for c in best_chain], dtype=np.float64)
    s_last = best_chain[-1][0]
    z_start_track = min(float(v_pv_mm[2]), float(zs[0]))
    dx0_traversed = max(1.0, float(contract.interp_x0(zs[-1]) - contract.interp_x0(z_start_track)))

    last_bar_x = int(np.argmin(np.abs(contract.bar_x_centers_mm - xs[-1])))
    y_valid = [(float(contract.station_z_mm[c[0]]), c[2]) for c in best_chain if c[2] is not None]
    last_bar_y = int(np.argmin(np.abs(contract.bar_y_centers_mm - y_valid[-1][1]))) if y_valid else 20

    exits_rear = s_last >= (n_st - 3)
    exits_side = (last_bar_x <= 1) or (last_bar_x >= contract.n_bars - 2) or (last_bar_y <= 1) or (last_bar_y >= contract.n_bars - 2)
    is_stopping = (not exits_rear) and (not exits_side)
    p_csda_gev = max(0.35, (dx0_traversed * CSDA_MEV_PER_X0_W) / 1000.0)

    if is_stopping:
        p_est_gev = p_csda_gev
        sig_qp = 0.12 / max(p_est_gev, 0.35)
    else:
        def _mcs_kinks(z_arr: np.ndarray, u_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            if len(z_arr) < 3:
                return np.empty(0), np.empty((0, 2))
            dz1 = z_arr[1:-1] - z_arr[:-2]
            dz2 = z_arr[2:] - z_arr[1:-1]
            s1 = (u_arr[1:-1] - u_arr[:-2]) / np.maximum(dz1, 1.0)
            s2 = (u_arr[2:] - u_arr[1:-1]) / np.maximum(dz2, 1.0)
            dth = s2 - s1
            dx0_seg = np.maximum(0.5, contract.interp_x0(z_arr[2:]) - contract.interp_x0(z_arr[:-2])) * 0.5
            sig_bar2 = (contract.bar_width_mm ** 2) / 12.0
            var_meas = sig_bar2 * (1.0 / (dz1 ** 2) + (1.0 / dz1 + 1.0 / dz2) ** 2 + 1.0 / (dz2 ** 2))
            return dth, np.stack([dx0_seg, var_meas], axis=-1)

        dth_x, meta_x = _mcs_kinks(zs, xs)
        if len(y_valid) >= 3:
            zy = np.array([v[0] for v in y_valid], dtype=np.float64)
            yy = np.array([v[1] for v in y_valid], dtype=np.float64)
            dth_y, meta_y = _mcs_kinks(zy, yy)
            dth_all = np.concatenate([dth_x, dth_y])
            meta_all = np.vstack([meta_x, meta_y])
        else:
            dth_all, meta_all = dth_x, meta_x

        if len(dth_all) > 0:
            p_grid = np.geomspace(max(p_csda_gev, 0.5), 120.0, 25)
            dx0_s, var_m = meta_all[:, 0], meta_all[:, 1]
            best_nll, best_p = 1e18, max(p_csda_gev, 5.0)
            for p_cand in p_grid:
                th_ms = (13.6 / (p_cand * 1000.0)) * np.sqrt(dx0_s) * (1.0 + 0.038 * np.log(np.maximum(dx0_s, 1.0)))
                var_tot = var_m + th_ms * th_ms
                nll = float(np.sum(np.log(var_tot) + (dth_all * dth_all) / var_tot))
                if nll < best_nll:
                    best_nll = nll
                    best_p = float(p_cand)
            p_est_gev = max(p_csda_gev, best_p)
            sig_qp = 0.22 / max(p_est_gev, 0.5)
        else:
            p_est_gev = max(p_csda_gev, 3.0)
            sig_qp = 0.30 / max(p_est_gev, 0.5)

    tx_avg = float((xs[-1] - xs[0]) / max(zs[-1] - zs[0], 1.0))
    ty_avg = float((y_valid[-1][1] - y_valid[0][1]) / max(y_valid[-1][0] - y_valid[0][0], 1.0)) if len(y_valid) >= 2 else 0.0
    x_ref = float(xs[0])
    y_ref = float(y_valid[0][1]) if y_valid else float(v_pv_mm[1])
    iso_gev = _compute_cal_iso_gev(x_ref, y_ref, float(zs[0]), tx_avg, ty_avg, cal_54, contract)
    sig_p = min(5000.0, p_est_gev * p_est_gev * sig_qp)
    return [{
        'mode': MUON_MODE_M4_RANGE_MCS,
        'q_over_p': 0.0,
        'sigma_q_over_p': float(sig_qp),
        'p_gev': float(p_est_gev),
        'sigma_p_gev': float(sig_p),
        'q_mu': 0,
        'q_valid_prob': 0.5,
        'in_bore': False,
        'm0_in_bore': False,
        'chi2_ndf': float(len(best_chain)),
        'x_ref': x_ref,
        'y_ref': y_ref,
        'tx': tx_avg,
        'ty': ty_avg,
        'iso_energy_gev': float(iso_gev),
    }]


def _pack_muon_kalman_tracks(cands: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Pack up to 4 candidates into Gen4EventBatch `(4, 16)` float32 & `(4,)` bool."""
    tracks = np.zeros((MUON_TRACK_SLOTS, MUON_TRACK_DIM), dtype=np.float32)
    mask = np.zeros(MUON_TRACK_SLOTS, dtype=bool)
    for k, c in enumerate(cands[:MUON_TRACK_SLOTS]):
        m_id = int(c['mode'])
        tracks[k, 0] = float(c['q_over_p'])
        tracks[k, 1] = float(c['sigma_q_over_p'])
        tracks[k, 2] = float(c['p_gev'])
        tracks[k, 3] = float(c.get('sigma_p_gev', c['p_gev'] ** 2 * c['sigma_q_over_p']))
        tracks[k, 4] = float(c['q_mu'])
        tracks[k, 5] = float(c.get('q_valid_prob', 1.0 if c['q_mu'] != 0 else 0.5))
        if 1 <= m_id <= 4:
            tracks[k, 5 + m_id] = 1.0  # [6:10] mode_onehot_M1_M2_M3_M4
        tracks[k, 10] = float(c['x_ref'])
        tracks[k, 11] = float(c['y_ref'])
        tracks[k, 12] = float(c['tx'])
        tracks[k, 13] = float(c['ty'])
        tracks[k, 14] = float(c['chi2_ndf'])
        tracks[k, 15] = float(c.get('iso_energy_gev', 0.0))
        mask[k] = True
    return tracks, mask


def unified_muon_kalman_event(
    observation: dict[str, Any],
    pix_data: dict[str, Any],
    cal_54: dict[str, Any],
    contract: Gen4GeometryContract,
    *,
    v_pv_mm: np.ndarray | None = None,
    s_mir: int = 1,
) -> dict[str, Any]:
    """Run four-modal M1-M4 UnifiedMuonKalman reconstruction on a single event."""
    if v_pv_mm is None:
        cs = pix_data['cal_start_clean']
        v_pv = np.array([cs[1], cs[2], cs[0]], dtype=np.float64) if cs[3] else np.array([0.0, -12.0, -2900.0])
    else:
        v_pv = np.asarray(v_pv_mm, dtype=np.float64).reshape(3)

    ipt_rays = _extract_ipt_rays(pix_data['pix_xyz_mm'], pix_data['pix_plane'], contract)
    spec_cands = _fit_m1_m2_m3_spectrometer(
        observation['tracker_hypothesis'], ipt_rays, cal_54, contract, s_mir=s_mir
    )
    m4_cands = _fit_m4_calorimeter_range_mcs(cal_54, v_pv, contract)

    all_cands = spec_cands + (m4_cands if not spec_cands else [])
    mode_rank = {
        MUON_MODE_M1_BORE3: 0,
        MUON_MODE_M2_IPT_M0_TRK1: 1,
        MUON_MODE_M3_SHELL: 2,
        MUON_MODE_M4_RANGE_MCS: 3,
    }
    all_cands.sort(key=lambda c: (mode_rank[c['mode']], c['chi2_ndf']))
    tracks, trk_mask = _pack_muon_kalman_tracks(all_cands)

    if not all_cands:
        return {
            'has_muon': False,
            'mode': MUON_MODE_NONE,
            'q_over_p': 0.0,
            'sigma_q_over_p': 1.0,
            'p_gev': 0.0,
            'q_mu': 0,
            'in_bore': False,
            'm0_in_bore': False,
            'n_mu_cands': 0,
            'has_opposite_sign_dimuon': False,
            'n_ipt_rays': int(len(ipt_rays)),
            'muon_kalman_tracks': tracks,
            'muon_kalman_mask': trk_mask,
            'candidates': [],
        }

    best = all_cands[0]
    qs = [c['q_mu'] for c in spec_cands if c['q_mu'] != 0]
    has_os_dimuon = (1 in qs) and (-1 in qs)

    return {
        'has_muon': True,
        'mode': int(best['mode']),
        'q_over_p': float(best['q_over_p']),
        'sigma_q_over_p': float(best['sigma_q_over_p']),
        'p_gev': float(best['p_gev']),
        'q_mu': int(best['q_mu']),
        'in_bore': bool(best['in_bore']),
        'm0_in_bore': bool(best['m0_in_bore']),
        'n_mu_cands': int(len(all_cands)),
        'has_opposite_sign_dimuon': bool(has_os_dimuon),
        'n_ipt_rays': int(len(ipt_rays)),
        'muon_kalman_tracks': tracks,
        'muon_kalman_mask': trk_mask,
        'candidates': all_cands,
    }


class UnifiedMuonKalman(nn.Module):
    """Differentiable PyTorch 4-modal broken-line Kalman muon spectrometer (M1-M4)."""

    def __init__(self, contract: Gen4GeometryContract | None = None) -> None:
        super().__init__()
        self.contract = contract
        self.log_mcs_scale = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.m4_range_scale = nn.Parameter(torch.ones(1, dtype=torch.float32))

    def forward_m2_batch(
        self,
        y_ipt: torch.Tensor,
        ty_ipt: torch.Tensor,
        z_ipt: torch.Tensor,
        y_trk1: torch.Tensor,
        sigma_y_ipt: torch.Tensor,
        sigma_ty_ipt: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Differentiable Magnet-0 soft-muon bending solver (221.1 GeV*mm arm)."""
        with torch.amp.autocast(device_type=y_ipt.device.type, enabled=False):
            dz = float(TRACKER_Z_MM[0]) - z_ipt.double()
            dy_m0 = y_trk1.double() - (y_ipt.double() + ty_ipt.double() * dz)
            q_over_p = (dy_m0 * 1000.0) / MAGNET0_BEND_CONST_MEV_MM
            mcs_fac = torch.exp(self.log_mcs_scale.double())
            sig_y = torch.sqrt(
                (SIGMA_Y_MM ** 2)
                + sigma_y_ipt.double().square()
                + (dz * sigma_ty_ipt.double() * mcs_fac).square()
                + 1e-8
            )
            sig_qp = (sig_y * 1000.0) / MAGNET0_BEND_CONST_MEV_MM
            p_gev = 1.0 / q_over_p.abs().clamp_min(1e-4)
            return {
                'q_over_p': q_over_p.float(),
                'sigma_q_over_p': sig_qp.float(),
                'p_gev': p_gev.clamp(max=5000.0).float(),
                'dy_m0_mm': dy_m0.float(),
            }

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Refine batched `muon_kalman_tracks` `(B, 4, 16)` in-place differentiable form."""
        tracks = batch['muon_kalman_tracks'].float()
        mask = batch['muon_kalman_mask'].bool()
        is_m4 = tracks[..., 9] > 0.5
        p_hat = tracks[..., 2]
        p_refined = torch.where(is_m4, p_hat * self.m4_range_scale.float().clamp(0.5, 2.0), p_hat)
        sig_qp_refined = tracks[..., 1] * torch.exp(self.log_mcs_scale.float())
        out_tracks = tracks.clone()
        out_tracks[..., 1] = sig_qp_refined
        out_tracks[..., 2] = p_refined
        return {
            'muon_kalman_tracks': out_tracks,
            'muon_kalman_mask': mask,
            'has_muon': mask.any(dim=-1),
            'primary_q_over_p': out_tracks[:, 0, 0],
            'primary_sigma_q_over_p': out_tracks[:, 0, 1],
            'primary_p_gev': out_tracks[:, 0, 2],
        }


# =============================================================================
# 4. Complete Single-Event Gen4 Frontend Extractor (Gen4EventBatch Item)
# =============================================================================

def extract_gen4_frontend_event(
    observation: dict[str, Any],
    contract: Gen4GeometryContract,
    *,
    has_subthr_readout: bool | None = True,
    s_mir: int = 1,
    enable_dark_filter: str | bool = 'auto',
    use_edep: bool = True,
    return_cluster_map: bool = False,
) -> dict[str, Any]:
    """Run complete zero-truth observation frontend matching `Gen4EventBatch` pure-obs schema."""
    validate_finite_observation(observation, check_edep=False)
    ds = dark_shield_filter_and_features(
        observation, contract, enable_dark_filter=enable_dark_filter, use_edep=use_edep
    )
    cs = ds['cal_start_clean']
    v_ref = np.array([cs[1], cs[2], cs[0]], dtype=np.float64) if cs[3] else None
    cal54 = reconstruct_calorimeter_54_stations(
        observation, contract, v_ref_mm=v_ref, has_subthr_readout=has_subthr_readout, s_mir=s_mir
    )
    muon = unified_muon_kalman_event(
        observation, ds, cal54, contract, v_pv_mm=v_ref, s_mir=s_mir
    )
    trk_cand_14, trk_cand_mask, trk_station_counts = extract_nonexclusive_spec_candidates(
        observation['tracker_hypothesis'], cs, contract.bounds_dict, max_candidates=8
    )
    trk_cand_16 = np.zeros((8, 16), dtype=np.float32)
    trk_cand_16[:, :14] = trk_cand_14

    out = {
        'c_obs': ds['c_obs_clean'].astype(np.float32),
        'obs_domain_id': int(ds['obs_domain_id_clean']),
        'cal_start_mm': np.array([cs[1], cs[2], cs[0]], dtype=np.float32) if cs[3] else np.zeros(3, dtype=np.float32),
        'has_subthr_readout': 1.0 if cal54['has_subthr_readout'] else 0.0,
        'has_edep': 1.0 if ds['has_edep'] else 0.0,
        's_mir': float(s_mir),
        'pix_xyz_mm': ds['pix_xyz_mm'],
        'pix_plane': ds['pix_plane'].astype(np.int64),
        'pix_feat': ds['pix_feat'],
        'pix_plane_summary': ds['pix_plane_summary'],
        'pix_patches_u8': ds['pix_patches_u8'],
        'patch_cluster_idx': ds['patch_cluster_idx'],
        'bar_station': cal54['bar_station'],
        'bar_view': cal54['bar_view'],
        'bar_idx': cal54['bar_idx'],
        'bar_feat_10': cal54['bar_feat_10'],
        'muon_kalman_tracks': muon['muon_kalman_tracks'],
        'muon_kalman_mask': muon['muon_kalman_mask'],
        'trk_cand': trk_cand_16,
        'trk_cand_mask': trk_cand_mask,
        'trk_station_counts': trk_station_counts,
        # Diagnostic metadata (purely observable)
        'muon_mode': muon['mode'],
        'total_desat_mev': float(cal54['total_desat_mev']),
        'n_raw_clusters': int(ds['n_raw_clusters']),
        'n_kept_clusters': int(ds['n_kept_clusters']),
    }
    if return_cluster_map:
        out['_cluster_map'] = (ds['hit_order'], ds['cluster_offsets'], ds['sel_raw_clusters'])
    return out
