"""Experiment L1: TrackletIPFormer-GN2 with ATLAS GN2 dense training-time supervision
and inference-time predicted-origin / purity / co-vertex topological conditioning.

Preserves `TrackletIPFormer` (6.23M base) pure-observable forward contract:
  - Strictly zero truth or cluster_sidecar access in `forward()` / `forward_NextGen()`
    (39-key + cluster_sidecar tamper invariance `max_abs_diff == 0.0`).
  - Dense training-time supervision via `derive_tracklet_gn2_targets`:
      1. `tracklet_origin_head (B, 64, 5)`: {0: PV_primary, 1: tau_decay, 2: charm_decay,
         3: photon_conv_ee, 4: had_sec_or_ghost}
      2. `tracklet_purity_head (B, 64, 1)`: true single-particle tracklet (1) vs combinatoric ghost (0)
      3. `pairwise_covertex_head (B, 64, 64, 1)`: symmetric co-vertex affinity Y_{ab}^{covtx}
  - Inference-time topological conditioning (`enable_readout_gate=True`):
      Feeds detached predicted `hat_w_real`, `hat_p_disp_hf`, and `hat_C_ab` into the
      differentiable Student-t(nu=3) PV/SV layer, post-PV relational attention, tracklet pool gate,
      and Perceiver cross-modal readout to suppress the ~97% ghost tracklets and ~99.46%
      gamma->e+e- conversion pairs without touching any truth labels.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from analysis.flavour.nextgen.protocols import (
    NextGenModelOutput,
    marginal_four_from_seven,
    student_t_vertex_loss,
)
from analysis.flavour.nextgen.tracklet_ipformer import (
    CONE_FEAT_DIM,
    MAX_TRACKLETS,
    MICRO_VTX_LEGACY_DIM,
    TrackletIPFormer,
    collate_tracklet_pf,
    extract_tracklet_pf_item,
)
from analysis.flavour.nextgen.tracklet_ops import (
    DifferentiableStudentTVertexLayer,
    PAIR_TRK_FEAT_DIM,
    TRK_KIN_DIM,
    TRK_MORPH_DIM,
    _safe_hypot,
    _solve_sym_3x3,
    extract_soft_tracklets_batch,
    highland_scattering_rad,
    pairwise_skew_dca_features,
)
from analysis.flavour.v3.common.domains import cumulative_radiation_length
from analysis.flavour.v3.v3_multitask.losses import (
    focal_charm_bce,
    marginal_tau_branch_ce,
)


GN2_ORIGIN_CLASSES = 5
GN2_ORIGIN_NAMES: tuple[str, ...] = (
    'PV_primary',
    'tau_decay',
    'charm_decay',
    'photon_conv_ee',
    'had_sec_or_ghost',
)
# Mild inverse-sqrt-frequency weights so rare signal tracklets (tau, charm, PV)
# are not drowned out by combinatoric ghosts (class 4) or EM conversions (class 3).
GN2_DEFAULT_ORIGIN_WEIGHTS: tuple[float, ...] = (2.0, 6.0, 6.0, 0.65, 0.35)
GN2_TOPO_SUMMARY_DIM = 12

GN2_SIDECAR_BATCH_KEYS: tuple[str, ...] = (
    'cl_origin_label',
    'cl_secondary_origin_label',
    'cl_sv_group_id',
    'cl_dominant_track_id',
    'cl_second_track_id',
    'cl_third_track_id',
    'trk_cl_idx',
    'trk_origin_target',
    'trk_purity_target',
    'trk_gn2_valid',
    'trk_covertex_target',
    'trk_covertex_valid',
)


def extract_soft_tracklets_with_indices(
    pix_xyz_mm: np.ndarray,
    pix_plane: np.ndarray,
    pix_feat: np.ndarray | None,
    plane_x0: np.ndarray,
    *,
    max_tracklets: int = MAX_TRACKLETS,
    max_per_plane: int = 36,
    pitch_xy_mm: tuple[float, float] = (0.0208, 0.0228),
    max_slope: float = 0.24,
) -> dict[str, np.ndarray]:
    """Exact mirror of `extract_soft_tracklets_single` that also returns constituent cluster indices `(M, 3)`."""
    kin = np.zeros((max_tracklets, TRK_KIN_DIM), dtype=np.float32)
    morph = np.zeros((max_tracklets, TRK_MORPH_DIM), dtype=np.float32)
    mask = np.zeros(max_tracklets, dtype=bool)
    cl_idx = np.full((max_tracklets, 3), -1, dtype=np.int64)

    n_cl = len(pix_xyz_mm)
    if n_cl < 2:
        return {'trk_kin': kin, 'trk_morph': morph, 'trk_mask': mask, 'trk_cl_idx': cl_idx}

    x = pix_xyz_mm[:, 0].astype(np.float64)
    y = pix_xyz_mm[:, 1].astype(np.float64)
    z = pix_xyz_mm[:, 2].astype(np.float64)
    pl = pix_plane.astype(np.int64)
    n_planes = int(len(plane_x0))

    if pix_feat is not None and len(pix_feat) == n_cl and pix_feat.shape[1] >= 10:
        log_nhits = pix_feat[:, 3].astype(np.float64) * 6.0
        sqrt_lam1 = 0.05 * np.sinh(np.clip(pix_feat[:, 7].astype(np.float64) * 3.0, 0.0, 6.0))
        sqrt_lam2 = 0.05 * np.sinh(np.clip(pix_feat[:, 8].astype(np.float64) * 3.0, 0.0, 6.0))
        ecc = pix_feat[:, 9].astype(np.float64)
    else:
        log_nhits = np.full(n_cl, math.log1p(1.0), dtype=np.float64)
        sqrt_lam1 = np.full(n_cl, pitch_xy_mm[0] / math.sqrt(12.0), dtype=np.float64)
        sqrt_lam2 = np.full(n_cl, pitch_xy_mm[1] / math.sqrt(12.0), dtype=np.float64)
        ecc = np.zeros(n_cl, dtype=np.float64)

    sig_px2 = (pitch_xy_mm[0] ** 2) / 12.0 + 1e-6
    sig_py2 = (pitch_xy_mm[1] ** 2) / 12.0 + 1e-6

    plane_groups: list[np.ndarray] = []
    for p in range(n_planes):
        idx = np.flatnonzero(pl == p)
        if len(idx) > 0:
            xc_p = float(np.mean(x[idx]))
            yc_p = float(np.mean(y[idx]))
            r2 = (x[idx] - xc_p) ** 2 + (y[idx] - yc_p) ** 2
            ord_p = np.lexsort((np.abs(x[idx] - xc_p), -log_nhits[idx], r2))
            idx = idx[ord_p[:max_per_plane]]
        plane_groups.append(idx)

    active_planes = [p for p in range(n_planes) if len(plane_groups[p]) > 0]
    if len(active_planes) < 2:
        return {'trk_kin': kin, 'trk_morph': morph, 'trk_mask': mask, 'trk_cl_idx': cl_idx}
    first_act_p = active_planes[0]

    records: list[tuple[float, int, np.ndarray, np.ndarray, int, tuple[int, int, int]]] = []

    for pa in range(n_planes - 1):
        ia_arr = plane_groups[pa]
        if len(ia_arr) == 0:
            continue
        is_pa_ipt = pa >= 12
        for pb in range(pa + 1, min(n_planes, pa + 3)):
            if is_pa_ipt != (pb >= 12):
                continue
            ib_arr = plane_groups[pb]
            if len(ib_arr) == 0:
                continue
            dz_ab = float(z[ib_arr[0]] - z[ia_arr[0]])
            if dz_ab < 5.0 or dz_ab > 185.0:
                continue
            if pb > pa + 1 and dz_ab > 70.0:
                continue
            dx0_ab = max(0.05, float(plane_x0[pb] - plane_x0[pa]))
            th_ms_ab = float(highland_scattering_rad(dx0_ab, p_ref_mev=8000.0))

            for ia in ia_arr:
                xa, ya, za = x[ia], y[ia], z[ia]
                sxa2 = sig_px2 + 0.15 * (sqrt_lam2[ia] ** 2)
                sya2 = sig_py2 + 0.15 * (sqrt_lam2[ia] ** 2)
                for ib in ib_arr:
                    tx_ab = (x[ib] - xa) / dz_ab
                    ty_ab = (y[ib] - ya) / dz_ab
                    sl_ab = math.hypot(tx_ab, ty_ab)
                    if sl_ab > max_slope:
                        continue

                    found_3 = False
                    for pc in range(pb + 1, min(n_planes, pb + 3)):
                        if is_pa_ipt != (pc >= 12):
                            continue
                        ic_arr = plane_groups[pc]
                        if len(ic_arr) == 0:
                            continue
                        dz_bc = float(z[ic_arr[0]] - z[ib])
                        dz_ac = float(z[ic_arr[0]] - za)
                        if dz_bc < 5.0 or dz_bc > 185.0 or dz_ac > 340.0:
                            continue
                        if pc > pb + 1 and dz_bc > 70.0:
                            continue
                        dx0_ac = max(0.10, float(plane_x0[pc] - plane_x0[pa]))
                        lever = (dz_ab * dz_bc) / dz_ac
                        th_mid = float(
                            highland_scattering_rad(
                                min(dx0_ab, max(0.05, float(plane_x0[pc] - plane_x0[pb]))),
                                8000.0,
                            )
                        )
                        sig_mid2 = (
                            1.5 * (sig_px2 + sig_py2)
                            + 0.15 * (sqrt_lam2[ib] ** 2)
                            + (2.0 * th_mid * lever) ** 2
                        )

                        xc_v = x[ic_arr]
                        yc_v = y[ic_arr]
                        tx_v = (xc_v - xa) / dz_ac
                        ty_v = (yc_v - ya) / dz_ac
                        sl_ok = np.hypot(tx_v, ty_v) <= max_slope
                        if not sl_ok.any():
                            continue
                        rx = x[ib] - (xa + tx_v * dz_ab)
                        ry = y[ib] - (ya + ty_v * dz_ab)
                        chi2_v = (rx * rx + ry * ry) / sig_mid2
                        valid_c = np.flatnonzero(sl_ok & (chi2_v <= 18.0))
                        for k_c in valid_c:
                            ic = int(ic_arr[k_c])
                            chi2_val = float(chi2_v[k_c])
                            zm = (za + z[ib] + z[ic]) / 3.0
                            xm = (xa + x[ib] + x[ic]) / 3.0
                            ym = (ya + y[ib] + y[ic]) / 3.0
                            szz = (za - zm) ** 2 + (z[ib] - zm) ** 2 + (z[ic] - zm) ** 2
                            tx_f = (
                                (za - zm) * (xa - xm)
                                + (z[ib] - zm) * (x[ib] - xm)
                                + (z[ic] - zm) * (x[ic] - xm)
                            ) / szz
                            ty_f = (
                                (za - zm) * (ya - ym)
                                + (z[ib] - zm) * (y[ib] - ym)
                                + (z[ic] - zm) * (y[ic] - ym)
                            ) / szz
                            x0_f = xm + tx_f * (za - zm)
                            y0_f = ym + ty_f * (za - zm)

                            half_g = min(40.0, 0.5 * dz_ab)
                            ms_pos2 = (0.8 * th_ms_ab * half_g) ** 2
                            sig_tx2 = (sxa2 + sig_px2) / (dz_ac * dz_ac) + (th_ms_ab ** 2)
                            sig_ty2 = (sya2 + sig_py2) / (dz_ac * dz_ac) + (th_ms_ab ** 2)

                            has_stub = 0.0
                            gap_flag = 0.0
                            if pa > first_act_p and not is_pa_ipt:
                                min_d_up = 1e9
                                for pu in range(first_act_p, pa):
                                    iu_arr = plane_groups[pu]
                                    if len(iu_arr) == 0:
                                        continue
                                    zu = z[iu_arr[0]]
                                    if za - zu > 165.0:
                                        continue
                                    xp = x0_f + tx_f * (zu - za)
                                    yp = y0_f + ty_f * (zu - za)
                                    du = float(np.min(np.hypot(x[iu_arr] - xp, y[iu_arr] - yp)))
                                    if du < min_d_up:
                                        min_d_up = du
                                if min_d_up <= 0.35:
                                    has_stub = 1.0
                                elif min_d_up > 1.20:
                                    gap_flag = 1.0

                            k_vec = np.array([
                                x0_f, y0_f, za, tx_f, ty_f,
                                sxa2 + ms_pos2, sya2 + ms_pos2,
                                sig_tx2, sig_ty2,
                                chi2_val, dx0_ac, dz_ac,
                            ], dtype=np.float32)
                            m_vec = np.array([
                                1.0,
                                1.0 if is_pa_ipt else 0.0,
                                float(log_nhits[ia]) / 5.0,
                                math.asinh(sqrt_lam1[ia] / 0.03) / 3.0,
                                math.asinh(sqrt_lam2[ia] / 0.03) / 3.0,
                                float(ecc[ia]),
                                has_stub,
                                gap_flag,
                            ], dtype=np.float32)
                            sort_score = chi2_val + 25.0 * (tx_f * tx_f + ty_f * ty_f)
                            records.append((sort_score, 3, k_vec, m_vec, int(ia), (int(ia), int(ib), int(ic))))
                            found_3 = True

                    if not found_3 and pb == pa + 1 and dz_ab <= 165.0 and sl_ab <= 0.16:
                        half_g = min(45.0, 0.5 * dz_ab)
                        ms_pos2 = (1.5 * th_ms_ab * half_g) ** 2
                        sig_tx2 = 2.0 * (sxa2 + sig_px2) / (dz_ab * dz_ab) + 1.5 * (th_ms_ab ** 2)
                        sig_ty2 = 2.0 * (sya2 + sig_py2) / (dz_ab * dz_ab) + 1.5 * (th_ms_ab ** 2)
                        p_chi2 = 4.0 + 180.0 * (tx_ab * tx_ab + ty_ab * ty_ab)
                        k_vec = np.array([
                            xa, ya, za, tx_ab, ty_ab,
                            2.0 * sxa2 + ms_pos2, 2.0 * sya2 + ms_pos2,
                            sig_tx2, sig_ty2,
                            p_chi2, dx0_ab, dz_ab,
                        ], dtype=np.float32)
                        m_vec = np.array([
                            0.0,
                            1.0 if is_pa_ipt else 0.0,
                            float(log_nhits[ia]) / 5.0,
                            math.asinh(sqrt_lam1[ia] / 0.03) / 3.0,
                            math.asinh(sqrt_lam2[ia] / 0.03) / 3.0,
                            float(ecc[ia]),
                            0.0,
                            0.0,
                        ], dtype=np.float32)
                        records.append((p_chi2 + 20.0, 2, k_vec, m_vec, int(ia), (int(ia), int(ib), -1)))

    if not records:
        return {'trk_kin': kin, 'trk_morph': morph, 'trk_mask': mask, 'trk_cl_idx': cl_idx}

    records.sort(key=lambda r: (-r[1], round(r[0], 7), round(float(r[2][2]), 4), round(abs(float(r[2][3])), 7)))

    root_counts: dict[int, int] = {}
    selected: list[tuple[np.ndarray, np.ndarray, int, tuple[int, int, int]]] = []
    for _, _, kv, mv, root_ia, idx_trip in records:
        c_cnt = root_counts.get(root_ia, 0)
        if c_cnt >= 3:
            continue
        root_counts[root_ia] = c_cnt + 1
        selected.append((kv, mv, root_ia, idx_trip))
        if len(selected) >= max_tracklets:
            break

    for idx_out, (kv, mv, root_ia, idx_trip) in enumerate(selected):
        kin[idx_out] = kv
        mv_out = mv.copy()
        if root_counts[root_ia] >= 2:
            mv_out[0] = mv_out[0] + 0.5
        morph[idx_out] = mv_out
        mask[idx_out] = True
        cl_idx[idx_out] = idx_trip

    return {'trk_kin': kin, 'trk_morph': morph, 'trk_mask': mask, 'trk_cl_idx': cl_idx}


def extract_tracklet_gn2_item(
    index: int,
    observation: dict,
    label: int | None,
    dom: dict,
    geometry: dict,
    bounds: dict,
    config: dict,
) -> dict:
    """Extract pure-observable TrackletIPFormer item plus deterministic constituent cluster index map.

    Strictly pure-observation: `trk_cl_idx` maps tracklets back to `pix_xyz_mm` row indices ONLY
    for training-time sidecar target derivation and offline post-inference diagnostics; `forward()`
    never reads `trk_cl_idx` or any sidecar tensors.
    """
    item = extract_tracklet_pf_item(index, observation, label, dom, geometry, bounds, config)
    pix_zpos = np.asarray(bounds['pixel_Zpos'], dtype=np.float64)
    plane_x0 = np.array(
        [cumulative_radiation_length(float(z), geometry) for z in pix_zpos],
        dtype=np.float32,
    )
    qc = config.get('quantize_cluster_coords_only_mm') if config else None
    if qc is not None and float(qc) > 0.0:
        pitch_xy = (float(qc), float(qc))
    else:
        pitch_xy = (float(bounds['dx_mm']), float(bounds['dy_mm']))
    trk = extract_soft_tracklets_with_indices(
        item['pix_xyz_mm'],
        item['pix_plane'],
        item['pix_feat'],
        plane_x0,
        max_tracklets=int((config or {}).get('max_tracklets', MAX_TRACKLETS)),
        pitch_xy_mm=pitch_xy,
    )
    item['trk_cl_idx'] = trk['trk_cl_idx']
    return item


def collate_tracklet_gn2(
    events: list[dict],
    mirror_rng: np.random.Generator | None = None,
    y_scint_center: float = -12.0,
    n_bars: int = 40,
) -> dict:
    """Collate TrackletIPFormerGN2 events, optionally packing cluster sidecar training targets."""
    batch = collate_tracklet_pf(events, mirror_rng=mirror_rng, y_scint_center=y_scint_center, n_bars=n_bars)
    if 'trk_cl_idx' in events[0]:
        batch['trk_cl_idx'] = torch.from_numpy(
            np.stack([e['trk_cl_idx'] for e in events]).astype(np.int64)
        )
    if 'cl_origin_label' in events[0]:
        b = len(events)
        max_cl = int(batch['pix_xyz_mm'].shape[1])
        orig = np.full((b, max_cl), -1, dtype=np.int64)
        sec_orig = np.full((b, max_cl), 255, dtype=np.int64)
        sv_gid = np.full((b, max_cl), -1, dtype=np.int64)
        tid = np.full((b, max_cl), -1, dtype=np.int64)
        tid2 = np.full((b, max_cl), -1, dtype=np.int64)
        tid3 = np.full((b, max_cl), -1, dtype=np.int64)
        for i, e in enumerate(events):
            nc = min(max_cl, len(e['cl_origin_label']))
            if nc > 0:
                orig[i, :nc] = np.asarray(e['cl_origin_label'][:nc], dtype=np.int64)
                sv_gid[i, :nc] = np.asarray(e['cl_sv_group_id'][:nc], dtype=np.int64)
                tid[i, :nc] = np.asarray(e['cl_dominant_track_id'][:nc], dtype=np.int64)
                if 'cl_second_track_id' in e:
                    tid2[i, :nc] = np.asarray(e['cl_second_track_id'][:nc], dtype=np.int64)
                if 'cl_third_track_id' in e:
                    tid3[i, :nc] = np.asarray(e['cl_third_track_id'][:nc], dtype=np.int64)
                if 'cl_secondary_origin_label' in e:
                    sec_orig[i, :nc] = np.asarray(e['cl_secondary_origin_label'][:nc], dtype=np.int64)
        batch['cl_origin_label'] = torch.from_numpy(orig)
        batch['cl_secondary_origin_label'] = torch.from_numpy(sec_orig)
        batch['cl_sv_group_id'] = torch.from_numpy(sv_gid)
        batch['cl_dominant_track_id'] = torch.from_numpy(tid)
        batch['cl_second_track_id'] = torch.from_numpy(tid2)
        batch['cl_third_track_id'] = torch.from_numpy(tid3)
    return batch


@torch.no_grad()
def match_tracklets_to_clusters_batch(
    trk_kin: torch.Tensor,
    trk_morph: torch.Tensor,
    trk_mask: torch.Tensor,
    pix_xyz_mm: torch.Tensor,
    pix_mask: torch.Tensor,
) -> torch.Tensor:
    """Vectorized fallback mapping tracklets `(B, M, 12)` to cluster indices `(B, M, 3)` when `trk_cl_idx` is absent."""
    b, m, _ = trk_kin.shape
    n_cl = int(pix_xyz_mm.shape[1])
    device = trk_kin.device
    if n_cl == 0:
        return torch.full((b, m, 3), -1, dtype=torch.long, device=device)

    kf = trk_kin.float()
    xyz = pix_xyz_mm.float()
    x0, y0, z0 = kf[..., 0], kf[..., 1], kf[..., 2]
    tx, ty = kf[..., 3], kf[..., 4]
    span = kf[..., 11]
    is_3hit = trk_morph[..., 0] >= 0.75

    cx = xyz[:, None, :, 0]
    cy = xyz[:, None, :, 1]
    cz = xyz[:, None, :, 2]
    cm = pix_mask[:, None, :]

    # Endpoint a: plane at z0
    same_za = (torch.abs(cz - z0[:, :, None]) < 1.0) & cm
    da2 = (cx - x0[:, :, None]).square() + (cy - y0[:, :, None]).square()
    da2_m = da2.masked_fill(~same_za, 1e12)
    ia = da2_m.argmin(dim=-1)
    has_a = same_za.any(dim=-1) & trk_mask

    # Endpoint c (3-hit) or b (2-hit): plane at z0 + span
    zend = z0 + span
    xend = x0 + tx * span
    yend = y0 + ty * span
    same_ze = (torch.abs(cz - zend[:, :, None]) < 1.0) & cm
    de2 = (cx - xend[:, :, None]).square() + (cy - yend[:, :, None]).square()
    de2_m = de2.masked_fill(~same_ze, 1e12)
    ie = de2_m.argmin(dim=-1)
    has_e = same_ze.any(dim=-1) & trk_mask

    # Intermediate b (for 3-hit): z strictly between z0 and zend
    mid_m = (cz > (z0[:, :, None] + 1.0)) & (cz < (zend[:, :, None] - 1.0)) & cm
    dz_c = cz - z0[:, :, None]
    dm2 = (cx - (x0[:, :, None] + tx[:, :, None] * dz_c)).square() + (cy - (y0[:, :, None] + ty[:, :, None] * dz_c)).square()
    dm2_m = dm2.masked_fill(~mid_m, 1e12)
    ib_3 = dm2_m.argmin(dim=-1)
    has_b3 = mid_m.any(dim=-1) & trk_mask

    ia_out = torch.where(has_a, ia, torch.full_like(ia, -1))
    ib_out = torch.where(is_3hit, torch.where(has_b3, ib_3, torch.full_like(ib_3, -1)), torch.where(has_e, ie, torch.full_like(ie, -1)))
    ic_out = torch.where(is_3hit & has_e, ie, torch.full_like(ie, -1))
    return torch.stack([ia_out, ib_out, ic_out], dim=-1)


@torch.no_grad()
def derive_tracklet_gn2_targets(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor] | None:
    """Derive 5-class tracklet origin, binary single-track purity, and 64x64 co-vertex affinity targets.

    Strictly called ONLY inside the training loss computation or offline post-inference diagnostics,
    NEVER inside `forward()` or `forward_NextGen()`.

    Returns `None` if `cl_origin_label` is absent from `batch`.
    """
    if 'cl_origin_label' not in batch:
        return None

    trk_kin = batch['trk_kin']
    trk_morph = batch['trk_morph']
    trk_mask = batch['trk_mask'].bool()
    b, m, _ = trk_kin.shape
    device = trk_kin.device

    cl_orig = batch['cl_origin_label'].long()          # (B, N_cl), 0..4 or -1
    cl_sv = batch['cl_sv_group_id'].long()             # (B, N_cl), >=0 or -1
    cl_tid = batch['cl_dominant_track_id'].long()      # (B, N_cl), >0 or <=0
    n_cl = int(cl_orig.shape[1])

    if 'trk_cl_idx' in batch:
        cl_idx = batch['trk_cl_idx'].long()
    else:
        cl_idx = match_tracklets_to_clusters_batch(
            trk_kin, trk_morph, trk_mask, batch['pix_xyz_mm'], batch['pix_mask'].bool()
        )

    is_3hit = trk_morph[..., 0] >= 0.75
    is_target_trk = trk_mask & (trk_morph[..., 1] < 0.5)

    if n_cl == 0:
        return {
            'trk_origin_target': torch.full((b, m), -1, dtype=torch.long, device=device),
            'trk_purity_target': torch.zeros((b, m), dtype=torch.float32, device=device),
            'trk_gn2_valid': torch.zeros((b, m), dtype=torch.bool, device=device),
            'trk_sv_group_id': torch.full((b, m), -1, dtype=torch.long, device=device),
            'trk_covertex_target': torch.zeros((b, m, m), dtype=torch.float32, device=device),
            'trk_covertex_valid': torch.zeros((b, m, m), dtype=torch.bool, device=device),
            'trk_covertex_is_disp_sv': torch.zeros((b, m, m), dtype=torch.bool, device=device),
        }

    idx_a = cl_idx[..., 0]
    idx_b = cl_idx[..., 1]
    idx_c = cl_idx[..., 2]

    has_a = (idx_a >= 0) & (idx_a < n_cl)
    has_b = (idx_b >= 0) & (idx_b < n_cl)
    has_c = (idx_c >= 0) & (idx_c < n_cl)
    has_all = is_target_trk & has_a & has_b & ((~is_3hit) | has_c)

    sa = idx_a.clamp(0, n_cl - 1)
    sb = idx_b.clamp(0, n_cl - 1)
    sc = idx_c.clamp(0, n_cl - 1)

    oa = torch.gather(cl_orig, 1, sa)
    ob = torch.gather(cl_orig, 1, sb)
    oc = torch.gather(cl_orig, 1, sc)

    cl_tid2 = batch.get('cl_second_track_id', torch.full_like(cl_tid, -1)).long()
    cl_tid3 = batch.get('cl_third_track_id', torch.full_like(cl_tid, -1)).long()
    cl_sec_orig = batch.get('cl_secondary_origin_label', torch.full_like(cl_orig, 255)).long()

    ta_set = torch.stack([torch.gather(cl_tid, 1, sa), torch.gather(cl_tid2, 1, sa), torch.gather(cl_tid3, 1, sa)], dim=-1)
    tb_set = torch.stack([torch.gather(cl_tid, 1, sb), torch.gather(cl_tid2, 1, sb), torch.gather(cl_tid3, 1, sb)], dim=-1)
    tc_set = torch.stack([torch.gather(cl_tid, 1, sc), torch.gather(cl_tid2, 1, sc), torch.gather(cl_tid3, 1, sc)], dim=-1)
    sec_ob = torch.gather(cl_sec_orig, 1, sb)

    sva = torch.gather(cl_sv, 1, sa)
    svb = torch.gather(cl_sv, 1, sb)
    svc = torch.gather(cl_sv, 1, sc)

    orig_ok = has_all & (oa >= 0) & (oa < GN2_ORIGIN_CLASSES) & (ob >= 0) & (ob < GN2_ORIGIN_CLASSES) & (
        (~is_3hit) | ((oc >= 0) & (oc < GN2_ORIGIN_CLASSES))
    )

    # Single-particle tracklet purity (supports shared-root top-3 trackIDs so merged 3-prong taus are not ghosts!)
    tb_pos = tb_set > 0
    in_ab = ((ta_set[..., None, :] == tb_set[..., :, None]) & (ta_set[..., None, :] > 0)).any(dim=-1)
    in_bc = ((tc_set[..., None, :] == tb_set[..., :, None]) & (tc_set[..., None, :] > 0)).any(dim=-1)
    hit_b = tb_pos & in_ab & ((~is_3hit[..., None]) | in_bc)
    is_pure = orig_ok & hit_b.any(dim=-1)

    # Pick matched track t*_m from hit_b (prefer heavy-flavour secondary origin if matched to slot 1)
    prefer_sec = hit_b[..., 1] & ((sec_ob == 1) | (sec_ob == 2)) & (~((ob == 1) | (ob == 2)))
    first_slot = hit_b.float().argmax(dim=-1)
    chosen_slot = torch.where(prefer_sec, torch.ones_like(first_slot), first_slot)
    t_star = torch.gather(tb_set, -1, chosen_slot.unsqueeze(-1)).squeeze(-1)
    t_star = torch.where(is_pure, t_star, torch.full_like(t_star, -1))

    # 5-class origin mapping:
    #   - pure tracklet (`is_pure`) -> matched downstream origin (with heavy-flavour priority from vote/sec_ob)
    #   - non-pure EM conversion/shower tracklet (`oa==3 & ob==3 & (~is_3hit | oc==3)`) -> class 3 (`photon_conv_ee`)
    #   - remaining combinatoric ghosts (`~is_pure & ~is_em_conv`) -> class 4 (`had_sec_or_ghost`)
    ob_eff = torch.where(prefer_sec, sec_ob, ob)
    vote = torch.zeros((b, m, GN2_ORIGIN_CLASSES), dtype=torch.float32, device=device)
    for k_cls in range(GN2_ORIGIN_CLASSES):
        cnt = (oa == k_cls).float() + 1.1 * (ob_eff == k_cls).float() + (is_3hit.float() * (oc == k_cls).float())
        vote[..., k_cls] = cnt
    # Tiny deterministic tie-break bias: 1 > 2 > 3 > 0 > 4
    tie = torch.tensor([0.02, 0.05, 0.04, 0.03, 0.01], dtype=torch.float32, device=device)[None, None, :]
    pure_origin = (vote + tie).argmax(dim=-1)
    is_em_conv = (~is_pure) & (oa == 3) & (ob == 3) & ((~is_3hit) | (oc == 3))
    non_pure_origin = torch.where(is_em_conv, torch.full_like(pure_origin, 3), torch.full_like(pure_origin, 4))
    y_origin = torch.where(is_pure, pure_origin, non_pure_origin)
    y_origin = torch.where(orig_ok, y_origin, torch.full_like(y_origin, -1))

    # Tracklet SV group ID (for co-vertex affinity): only valid for pure tracklets
    sv_a_ok = sva >= 0
    sv_b_ok = svb >= 0
    sv_c_ok = is_3hit & (svc >= 0)
    trk_sv = torch.where(
        sv_b_ok,
        svb,
        torch.where(sv_a_ok, sva, torch.where(sv_c_ok, svc, torch.full_like(sva, -1))),
    )
    trk_sv = torch.where(is_pure, trk_sv, torch.full_like(trk_sv, -1))

    # Pairwise co-vertex affinity Y_{ab}^{covtx}: distinct pure sister tracks sharing same vertex
    eye = torch.eye(m, dtype=torch.bool, device=device)[None, :, :]
    cov_valid = orig_ok[:, :, None] & orig_ok[:, None, :] & (~eye)
    both_pure = is_pure[:, :, None] & is_pure[:, None, :]
    diff_trk = (t_star[:, :, None] > 0) & (t_star[:, None, :] > 0) & (t_star[:, :, None] != t_star[:, None, :])
    both_pv = (y_origin[:, :, None] == 0) & (y_origin[:, None, :] == 0)
    same_sv_id = (trk_sv[:, :, None] >= 0) & (trk_sv[:, :, None] == trk_sv[:, None, :])
    same_vtx = both_pure & diff_trk & (same_sv_id | (both_pv & (trk_sv[:, :, None] < 0) & (trk_sv[:, None, :] < 0)))
    y_covtx = (same_vtx & cov_valid).float()
    is_disp_sv_pair = same_vtx & cov_valid & (trk_sv[:, :, None] >= 1)

    return {
        'trk_origin_target': y_origin,
        'trk_purity_target': (is_pure & orig_ok).float(),
        'trk_gn2_valid': orig_ok,
        'trk_sv_group_id': trk_sv,
        'trk_covertex_target': y_covtx,
        'trk_covertex_valid': cov_valid,
        'trk_covertex_is_disp_sv': is_disp_sv_pair,
    }


class GN2ConditionedStudentTVertexLayer(DifferentiableStudentTVertexLayer):
    """Differentiable Student-t(nu=3) PV/SV layer conditioned by predicted GN2 purity/origin weights."""

    def forward(
        self,
        trk_kin: torch.Tensor,
        trk_emb: torch.Tensor,
        trk_mask: torch.Tensor,
        v_cal_prior_mm: torch.Tensor,
        cal_valid: torch.Tensor | None = None,
        *,
        w_pv_cond: torch.Tensor | None = None,
        w_sv_cond: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        with torch.amp.autocast(device_type=trk_kin.device.type, enabled=False):
            k = trk_kin.float()
            emb = trk_emb.float()
            prior = v_cal_prior_mm.float()
            b, m, _ = k.shape

            x0 = k[..., 0]
            y0 = k[..., 1]
            z0 = k[..., 2]
            tx = k[..., 3]
            ty = k[..., 4]
            sxx = k[..., 5].clamp_min(4e-5)
            syy = k[..., 6].clamp_min(4e-5)
            stx = k[..., 7].clamp_min(1e-7)
            sty = k[..., 8].clamp_min(1e-7)

            pi_pv = torch.sigmoid(self.pv_weight_head(emb).squeeze(-1)) * trk_mask.float()
            if w_pv_cond is not None:
                pi_pv = pi_pv * w_pv_cond.float().clamp(0.02, 1.0)

            base_prec = torch.exp(self.cal_log_prec.clamp(-9.0, 2.0))[None, :]
            if cal_valid is not None:
                cv = cal_valid.float().view(b, 1)
                p_prior = cv * base_prec + (1.0 - cv) * 1e-5
            else:
                p_prior = base_prec.expand(b, 3)

            v_cur = prior.clone()
            nu = self.df

            for _ in range(self.n_iters):
                dz = (v_cur[:, None, 2] - z0).clamp(-350.0, 350.0)
                var_x = sxx + dz.square() * stx
                var_y = syy + dz.square() * sty
                rx = (x0 + tx * dz) - v_cur[:, None, 0]
                ry = (y0 + ty * dz) - v_cur[:, None, 1]
                chi2 = (rx.square() / var_x + ry.square() / var_y).clamp_max(1e4)

                w_rob = pi_pv / torch.pow(1.0 + chi2 / nu, 0.5 * (nu + 1.0))
                wx = w_rob / var_x
                wy = w_rob / var_y

                cx = x0 - tx * z0
                cy = y0 - ty * z0

                m00 = p_prior[:, 0] + wx.sum(dim=1)
                m11 = p_prior[:, 1] + wy.sum(dim=1)
                m02 = -(wx * tx).sum(dim=1)
                m12 = -(wy * ty).sum(dim=1)
                m22 = p_prior[:, 2] + (wx * tx.square() + wy * ty.square()).sum(dim=1)
                zeros = torch.zeros_like(m00)

                mat = torch.stack([
                    torch.stack([m00, zeros, m02], dim=-1),
                    torch.stack([zeros, m11, m12], dim=-1),
                    torch.stack([m02, m12, m22], dim=-1),
                ], dim=-2)

                rhs = torch.stack([
                    p_prior[:, 0] * prior[:, 0] + (wx * cx).sum(dim=1),
                    p_prior[:, 1] * prior[:, 1] + (wy * cy).sum(dim=1),
                    p_prior[:, 2] * prior[:, 2] - (wx * tx * cx + wy * ty * cy).sum(dim=1),
                ], dim=-1)

                v_cur, inv_diag = _solve_sym_3x3(mat, rhs)

            v_pv = v_cur
            scale_pv = (torch.sqrt(inv_diag) * torch.exp(self.scale_cal.clamp(-2.0, 3.0))[None, :]).clamp(0.02, 150.0)

            dz_f = (v_pv[:, None, 2] - z0).clamp(-350.0, 350.0)
            var_xf = sxx + dz_f.square() * stx + inv_diag[:, None, 0]
            var_yf = syy + dz_f.square() * sty + inv_diag[:, None, 1]
            dx_pv = (x0 + tx * dz_f) - v_pv[:, None, 0]
            dy_pv = (y0 + ty * dz_f) - v_pv[:, None, 1]
            d0 = _safe_hypot(dx_pv, dy_pv, eps=1.0e-12) * trk_mask.float()
            chi2_pv = (dx_pv.square() / var_xf + dy_pv.square() / var_yf) * trk_mask.float()
            s_d0 = torch.sqrt(chi2_pv + 1.0e-10) * trk_mask.float()

            w_thr = trk_mask.float()
            ux = (tx * w_thr).sum(dim=1, keepdim=True)
            uy = (ty * w_thr).sum(dim=1, keepdim=True)
            u_n = torch.sqrt(ux.square() + uy.square() + 1.0e-10).clamp_min(1e-5)
            signed_d0 = (-(dx_pv * (ux / u_n) + dy_pv * (uy / u_n))) * trk_mask.float()

            ip_feat = torch.stack([
                torch.asinh(d0 / 0.05) / 3.0,
                torch.log1p(s_d0) / 3.0,
                torch.asinh(signed_d0 / 0.05) / 3.0,
                pi_pv,
            ], dim=-1) * trk_mask.unsqueeze(-1).float()

            sv_in = torch.cat([emb, ip_feat], dim=-1)
            pi_sv = torch.sigmoid(self.sv_weight_head(sv_in).squeeze(-1)) * torch.sigmoid(s_d0 - 2.0) * trk_mask.float()
            if w_sv_cond is not None:
                pi_sv = pi_sv * w_sv_cond.float().clamp(0.01, 1.0)

            sv_prior = v_pv + v_pv.new_tensor([0.0, 0.0, 2.5])[None, :]
            p_sv0 = v_pv.new_tensor([1e-2, 1e-2, 5e-3])[None, :].expand(b, 3)
            v_sv = sv_prior.clone()
            for _ in range(2):
                dz_s = (v_sv[:, None, 2] - z0).clamp(-350.0, 350.0)
                vxs = sxx + dz_s.square() * stx
                vys = syy + dz_s.square() * sty
                wxs = pi_sv / vxs
                wys = pi_sv / vys
                cxs = x0 - tx * z0
                cys = y0 - ty * z0
                ms00 = p_sv0[:, 0] + wxs.sum(dim=1)
                ms11 = p_sv0[:, 1] + wys.sum(dim=1)
                ms02 = -(wxs * tx).sum(dim=1)
                ms12 = -(wys * ty).sum(dim=1)
                ms22 = p_sv0[:, 2] + (wxs * tx.square() + wys * ty.square()).sum(dim=1)
                zs = torch.zeros_like(ms00)
                mat_s = torch.stack([
                    torch.stack([ms00, zs, ms02], dim=-1),
                    torch.stack([zs, ms11, ms12], dim=-1),
                    torch.stack([ms02, ms12, ms22], dim=-1),
                ], dim=-2)
                rhs_s = torch.stack([
                    p_sv0[:, 0] * sv_prior[:, 0] + (wxs * cxs).sum(dim=1),
                    p_sv0[:, 1] * sv_prior[:, 1] + (wys * cys).sum(dim=1),
                    p_sv0[:, 2] * sv_prior[:, 2] - (wxs * tx * cxs + wys * ty * cys).sum(dim=1),
                ], dim=-1)
                v_sv, _ = _solve_sym_3x3(mat_s, rhs_s)

            sv_disp = v_sv - v_pv
            dz_sv_f = (v_sv[:, None, 2] - z0).clamp(-350.0, 350.0)
            dx_sv = (x0 + tx * dz_sv_f) - v_sv[:, None, 0]
            dy_sv = (y0 + ty * dz_sv_f) - v_sv[:, None, 1]
            chi2_sv = (dx_sv.square() / (sxx + dz_sv_f.square() * stx) + dy_sv.square() / (syy + dz_sv_f.square() * sty))
            delta_chi2_2v = (pi_sv * (chi2_pv - chi2_sv).clamp(min=0.0, max=50.0)).sum(dim=1, keepdim=True)

            return {
                'v_pv_mm': v_pv,
                'scale_pv_mm': scale_pv,
                'v_sv_mm': v_sv,
                'sv_disp_mm': sv_disp,
                'delta_chi2_2v': delta_chi2_2v,
                'trk_ip_feat': ip_feat,
                'pi_pv': pi_pv,
                'pi_sv': pi_sv,
            }


class TrackletIPFormerGN2(TrackletIPFormer):
    """TrackletIPFormer augmented with ATLAS GN2 dense supervision & topological conditioning."""

    def __init__(self, geometry: dict, config: dict | None = None) -> None:
        cfg = dict(config or {})
        cfg['use_legacy_16d_only'] = False
        cfg['use_hard_avr_pv'] = False
        super().__init__(geometry, cfg)

        dim = self.dim
        heads = self.heads

        # Single-factor ablation switches
        self.enable_origin_purity = bool(self.config.get('enable_origin_purity', True))
        self.enable_covertex = bool(self.config.get('enable_covertex', True))
        self.enable_readout_gate = bool(self.config.get('enable_readout_gate', True))

        # Upgrade vertex_layer in-place with GN2-conditioned Student-t PV/SV layer (identical param names)
        self.vertex_layer = GN2ConditionedStudentTVertexLayer(hidden_dim=dim, df=3.0, n_iters=4)

        if self.enable_origin_purity:
            self.tracklet_origin_head: nn.Module | None = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, 128),
                nn.SiLU(),
                nn.Dropout(0.10),
                nn.Linear(128, GN2_ORIGIN_CLASSES),
            )
            self.tracklet_purity_head: nn.Module | None = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, 96),
                nn.SiLU(),
                nn.Dropout(0.10),
                nn.Linear(96, 1),
            )
        else:
            self.tracklet_origin_head = None
            self.tracklet_purity_head = None

        if self.enable_covertex:
            cov_dim = 48
            self.covtx_node_proj: nn.Module | None = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, cov_dim),
                nn.SiLU(),
            )
            self.covtx_sum_proj: nn.Module | None = nn.Linear(cov_dim, 64)
            self.covtx_prod_proj: nn.Module | None = nn.Linear(cov_dim, 64)
            self.covtx_pair_r_proj: nn.Module | None = nn.Linear(PAIR_TRK_FEAT_DIM, 64, bias=False)
            self.covtx_out_proj: nn.Module | None = nn.Linear(64, 1)
        else:
            self.covtx_node_proj = None
            self.covtx_sum_proj = None
            self.covtx_prod_proj = None
            self.covtx_pair_r_proj = None
            self.covtx_out_proj = None

        if self.enable_readout_gate and (self.enable_origin_purity or self.enable_covertex):
            if self.enable_covertex:
                self.covtx_attn_alpha: nn.Parameter | None = nn.Parameter(torch.full((heads,), 0.5, dtype=torch.float32))
            else:
                self.covtx_attn_alpha = None
            self.gn2_pool_alpha: nn.Parameter | None = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
            self.gn2_perceiver_beta: nn.Parameter | None = nn.Parameter(torch.full((1, heads, 1, 1), 0.35, dtype=torch.float32))
            self.gn2_topo_proj: nn.Module | None = nn.Sequential(
                nn.Linear(GN2_TOPO_SUMMARY_DIM, dim // 2),
                nn.LayerNorm(dim // 2),
                nn.SiLU(),
            )
            fused_in_dim = 7 * dim  # 6*dim + (dim//2 vtx) + (dim//2 gn2_topo)
            self.global_fusion = nn.Sequential(
                nn.LayerNorm(fused_in_dim),
                nn.Linear(fused_in_dim, 3 * dim),
                nn.SiLU(),
            )
        else:
            self.covtx_attn_alpha = None
            self.gn2_pool_alpha = None
            self.gn2_perceiver_beta = None
            self.gn2_topo_proj = None

    def branch_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        groups: dict[str, list[nn.Parameter]] = {}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith('patch_'):
                br = 'pixel_patch_conv'
            elif name.startswith(('pix_in.', 'pix_plane_emb.')):
                br = 'pixel_clusters'
            elif name.startswith('stereo_bars.'):
                br = 'stereo_bars'
            elif name.startswith('spectrometer.'):
                br = 'spectrometer'
            elif name.startswith(('trk_in.', 'pair_bias_mlp.', 'rel_attn_', 'ip_reinject.', 'ip_gate', 'trk_pool_gate.')):
                br = 'tracklet_relational'
            elif name.startswith(('vertex_layer.', 'pv_anchor.', 'pv_scale_cal.', 'vtx_summary_proj.')):
                br = 'vertex_layer'
            elif name.startswith(('tracklet_origin_head.', 'tracklet_purity_head.', 'covtx_', 'gn2_')):
                br = 'gn2_heads'
            elif name.startswith(('z_pos_proj.', 'z_profile_encoder.', 'z_pool_gate.', 'latent_queries', 'cobs_query.', 'global_blocks.', 'pool_gate.', 'global_fusion.')):
                br = 'global_fusion'
            else:
                br = 'heads'
            groups.setdefault(br, []).append(param)
        return groups

    def forward_NextGen(self, batch: dict[str, torch.Tensor]) -> NextGenModelOutput:
        b = int(batch['c_obs'].shape[0])
        device = batch['c_obs'].device

        # 1. Pixel cluster tokens + 32x32 micro-patch CNN (strictly pure observables)
        pix_feat = batch['pix_feat']
        pix_mask = batch['pix_mask']
        u_pix = self.pix_in(pix_feat) * pix_mask.unsqueeze(-1).to(pix_feat.dtype)
        if len(batch['pix_patches']):
            patch_emb = self.patch_proj(self.patch_conv(batch['pix_patches']))
            delta_patch = torch.zeros_like(u_pix, dtype=patch_emb.dtype)
            delta_patch.index_put_((batch['patch_ev'], batch['patch_cl']), patch_emb, accumulate=True)
            u_pix = u_pix + delta_patch * pix_mask.unsqueeze(-1).to(u_pix.dtype)
        pix_plane_tok = self.pix_plane_emb(batch['pix_plane_summary'])
        pix_plane_act = batch['pix_plane_summary'][..., 0] > 0.5

        # 2. Stereo calorimeter (with ToF)
        bar_out = self.stereo_bars(batch)
        cal_tok = bar_out['cal_tokens']
        cal_mask = bar_out['cal_mask']
        cal_global = bar_out['cal_global']

        # 3. Soft non-exclusive tracklets + Magnet-0 IPT-Spectrometer interlock
        if 'trk_kin' in batch and 'trk_morph' in batch and 'trk_mask' in batch:
            trk_kin = batch['trk_kin']
            trk_morph = batch['trk_morph']
            trk_mask = batch['trk_mask']
        else:
            ext = extract_soft_tracklets_batch(
                batch['pix_xyz_mm'],
                batch['pix_plane'],
                pix_mask,
                self.pix_plane_x0,
                batch['pix_feat'],
                max_tracklets=self.max_tracklets,
            )
            trk_kin = ext['trk_kin']
            trk_morph = ext['trk_morph']
            trk_mask = ext['trk_mask']

        is_ipt_stub = (trk_morph[..., 1] > 0.5) & trk_mask
        spec_glob, spec_cand_tok = self.spectrometer(
            batch['trk_cand'],
            batch['trk_cand_mask'],
            batch['trk_hits'],
            batch['trk_hit_mask'],
            trk_kin,
            is_ipt_stub,
        )

        # 4. Longitudinal 69-plane profile encoder
        z_in = torch.cat([pix_plane_tok, cal_tok[:, :self.n_st]], dim=1)[:, self.z_order_idx]
        z_act = torch.cat([pix_plane_act, cal_mask[:, :self.n_st]], dim=1)[:, self.z_order_idx]
        z_in = (z_in + self.z_pos_proj(self.z_plane_geom).unsqueeze(0).to(z_in.dtype)) * z_act.unsqueeze(-1).to(z_in.dtype)
        z_seq = self.z_profile_encoder(z_in, z_act)

        v_cal_prior, cal_valid = self._get_v_cal_prior(batch, b, device)
        with torch.amp.autocast(device_type=device.type, enabled=False):
            any_z = z_act.any(dim=1, keepdim=True)
            safe_za = z_act.clone()
            if (~any_z[:, 0]).any():
                safe_za[~any_z[:, 0], 0] = True
            z_logits = self.z_pool_gate(z_seq.float()).squeeze(-1).masked_fill(~safe_za, -1e4)
            z_w = torch.softmax(z_logits, dim=1) * z_act.float()
            z_prof_glob = (z_w.unsqueeze(-1) * z_seq.float()).sum(dim=1)

        # 5. Tracklet Relational Pre-Encoder + GN2 Dense Predictions
        assert self.cone_projector is not None and self.trk_in is not None
        assert self.rel_attn_pre is not None and self.rel_attn_post is not None
        assert self.ip_reinject is not None and self.ip_gate is not None and self.trk_pool_gate is not None
        assert self.vertex_layer is not None and self.pv_anchor is not None and self.pv_scale_cal is not None

        cone_feat = self.cone_projector(
            trk_kin,
            trk_mask,
            bar_out['nex_xyz_mm'],
            bar_out['nex_e'],
            bar_out['nex_tof'],
            bar_out['nex_mask'],
        )
        trk_norm = self._normalize_trk_kin(trk_kin)
        trk_in_cat = torch.cat([trk_norm, trk_morph.float(), cone_feat], dim=-1)
        h_trk = self.trk_in(trk_in_cat.to(u_pix.dtype)) * trk_mask.unsqueeze(-1).to(u_pix.dtype)

        r_ab = pairwise_skew_dca_features(trk_kin, trk_mask)
        if self.pair_bias_mlp is not None:
            pair_bias: torch.Tensor | None = self.pair_bias_mlp(r_ab.to(u_pix.dtype)).permute(0, 3, 1, 2)
        else:
            pair_bias = None

        h_trk = self.rel_attn_pre(h_trk, trk_mask, pair_bias=pair_bias)
        h_trk = h_trk * trk_mask.unsqueeze(-1).to(h_trk.dtype)

        # GN2 dense heads (strictly pure-observable predictions in FP32)
        m_trk = int(trk_kin.shape[1])
        tm_f = trk_mask.float()
        eye = torch.eye(m_trk, dtype=torch.bool, device=device)[None, :, :]
        off_diag = (trk_mask[:, :, None] & trk_mask[:, None, :]) & (~eye)

        origin_logits: torch.Tensor | None = None
        purity_logit: torch.Tensor | None = None
        covertex_logits: torch.Tensor | None = None
        hat_C_ab: torch.Tensor | None = None

        with torch.amp.autocast(device_type=device.type, enabled=False):
            ht_f = h_trk.float()
            if self.enable_origin_purity and self.tracklet_origin_head is not None and self.tracklet_purity_head is not None:
                origin_logits = self.tracklet_origin_head(ht_f)
                purity_logit = self.tracklet_purity_head(ht_f)
                p_orig = torch.softmax(origin_logits, dim=-1)
                hat_w_real = torch.sigmoid(purity_logit.squeeze(-1)) * tm_f
                hat_p_pv = p_orig[..., 0] * tm_f
                hat_p_tau = p_orig[..., 1] * tm_f
                hat_p_charm = p_orig[..., 2] * tm_f
                hat_p_conv = p_orig[..., 3] * tm_f
                hat_p_disp_hf = (hat_p_tau + hat_p_charm) * tm_f
            else:
                hat_w_real = tm_f
                hat_p_pv = trk_kin.new_zeros((b, m_trk), dtype=torch.float32)
                hat_p_tau = trk_kin.new_zeros((b, m_trk), dtype=torch.float32)
                hat_p_charm = trk_kin.new_zeros((b, m_trk), dtype=torch.float32)
                hat_p_conv = trk_kin.new_zeros((b, m_trk), dtype=torch.float32)
                hat_p_disp_hf = trk_kin.new_zeros((b, m_trk), dtype=torch.float32)

            if self.enable_covertex and self.covtx_node_proj is not None:
                assert self.covtx_sum_proj is not None and self.covtx_prod_proj is not None
                assert self.covtx_pair_r_proj is not None and self.covtx_out_proj is not None
                e_cov = self.covtx_node_proj(ht_f)
                s_cov = self.covtx_sum_proj(e_cov)
                p_cov = self.covtx_prod_proj(e_cov)
                r_sym = 0.5 * (r_ab.float() + r_ab.float().transpose(1, 2))
                pair_h = F.silu(
                    s_cov[:, :, None, :]
                    + s_cov[:, None, :, :]
                    + (p_cov[:, :, None, :] * p_cov[:, None, :, :])
                    + self.covtx_pair_r_proj(r_sym)
                )
                covertex_logits = self.covtx_out_proj(pair_h)
                hat_C_ab = torch.sigmoid(covertex_logits.squeeze(-1)) * off_diag.float()
                c_det = hat_C_ab.detach()
                top2_c = torch.topk(c_det, k=min(2, m_trk), dim=-1).values.mean(dim=-1) * tm_f
            else:
                top2_c = trk_kin.new_zeros((b, m_trk), dtype=torch.float32)

            # Detached predicted conditioning weights for Student-t PV/SV & readout
            w_pv_cond: torch.Tensor | None = None
            w_sv_cond: torch.Tensor | None = None
            if self.enable_readout_gate and (self.enable_origin_purity or self.enable_covertex):
                w_real_det = hat_w_real.detach()
                p_pv_det = hat_p_pv.detach()
                p_disp_det = hat_p_disp_hf.detach()
                p_conv_det = hat_p_conv.detach()
                if self.enable_origin_purity:
                    w_pv_cond = (0.15 + 0.85 * w_real_det) * (0.20 + 0.80 * (1.0 - p_conv_det)) * (0.40 + 0.60 * p_pv_det)
                    w_sv_cond = (0.10 + 0.90 * w_real_det) * (0.15 + 0.85 * p_disp_det) * (0.15 + 0.85 * (1.0 - p_conv_det))
                else:
                    w_pv_cond = 0.25 + 0.75 * top2_c
                    w_sv_cond = 0.25 + 0.75 * top2_c
                if self.enable_covertex:
                    w_sv_cond = w_sv_cond * (0.35 + 0.65 * top2_c)

        # 6. Differentiable Student-t(nu=3) PV/SV Layer
        v_nn_mm, scale_nn_mm, h_vtx_ctx = self.pv_anchor(
            z_seq, z_act, v_cal_prior, u_pix, batch['pix_xyz_mm'], pix_mask, batch['pix_plane']
        )
        vtx_out = self.vertex_layer(
            trk_kin,
            h_trk,
            trk_mask,
            v_cal_prior,
            cal_valid=cal_valid,
            w_pv_cond=w_pv_cond,
            w_sv_cond=w_sv_cond,
        )
        with torch.amp.autocast(device_type=device.type, enabled=False):
            prec_nn = 1.0 / (scale_nn_mm.square() + 0.04)
            prec_bil = 1.0 / (vtx_out['scale_pv_mm'].square() + 0.04)
            prec_tot = prec_nn + prec_bil
            v_pv_mm = (prec_nn * v_nn_mm + prec_bil * vtx_out['v_pv_mm']) / prec_tot
            sigma_ivw = torch.rsqrt(prec_tot)
            trk_ip_feat = self._compute_ip_features(
                trk_kin, trk_mask, v_pv_mm, pi_pv=vtx_out['trk_ip_feat'][..., 3]
            )
            s_d0_raw = torch.expm1((3.0 * trk_ip_feat[..., 1]).clamp(0.0, 12.0)) * tm_f

        sv_disp_mm = vtx_out['sv_disp_mm']
        delta_chi2_2v = vtx_out['delta_chi2_2v']
        pi_sv = vtx_out['pi_sv']

        # Post-PV relational attention (optionally biased by detached predicted co-vertex affinity hat_C_ab)
        ip_in = torch.cat([trk_ip_feat, pi_sv.unsqueeze(-1)], dim=-1)
        h_trk = h_trk + self.ip_gate * self.ip_reinject(ip_in.to(h_trk.dtype)) * trk_mask.unsqueeze(-1).to(h_trk.dtype)
        if self.enable_readout_gate and self.enable_covertex and self.covtx_attn_alpha is not None and hat_C_ab is not None:
            cov_bias = self.covtx_attn_alpha.float().view(1, self.heads, 1, 1) * (hat_C_ab.detach()[:, None, :, :] - 0.5) * off_diag[:, None, :, :].float()
            post_pair_bias = (pair_bias.float() + cov_bias) if pair_bias is not None else cov_bias
        else:
            post_pair_bias = pair_bias
        h_trk = self.rel_attn_post(h_trk, trk_mask, pair_bias=post_pair_bias)
        h_trk = h_trk * trk_mask.unsqueeze(-1).to(h_trk.dtype)

        # Tracklet readout conditioning weights & pooling
        with torch.amp.autocast(device_type=device.type, enabled=False):
            any_trk = trk_mask.any(dim=1, keepdim=True)
            safe_tm = trk_mask.clone()
            if (~any_trk[:, 0]).any():
                safe_tm[~any_trk[:, 0], 0] = True

            if self.enable_readout_gate and (self.enable_origin_purity or self.enable_covertex):
                assert self.gn2_pool_alpha is not None and self.gn2_perceiver_beta is not None
                if self.enable_origin_purity:
                    tok_rho_trk = (0.12 + 0.88 * hat_w_real.detach()) * (0.15 + 0.85 * (1.0 - hat_p_conv.detach()))
                else:
                    tok_rho_trk = 0.25 + 0.75 * top2_c
                if self.enable_covertex:
                    tok_rho_trk = tok_rho_trk * (0.70 + 0.30 * top2_c)
                tok_rho_trk = tok_rho_trk.clamp(0.05, 1.0)
                t_logits = self.trk_pool_gate(h_trk.float()).squeeze(-1) + self.gn2_pool_alpha.float() * torch.log(tok_rho_trk)
            else:
                tok_rho_trk = None
                t_logits = self.trk_pool_gate(h_trk.float()).squeeze(-1)

            t_logits = t_logits.masked_fill(~safe_tm, -1e4)
            t_w = torch.softmax(t_logits, dim=1) * tm_f
            trk_glob = (t_w.unsqueeze(-1) * h_trk.float()).sum(dim=1)

            pv_ctx = torch.cat([h_vtx_ctx, z_prof_glob, cal_global.float(), trk_glob, batch['c_obs'].float()], dim=-1)
            scale_pv_mm = (sigma_ivw * torch.exp(self.pv_scale_cal(pv_ctx).clamp(-2.0, 2.0))).clamp(0.02, 150.0)

        # 7. Cross-Modal Perceiver Fusion (with optional GN2 topological attention conditioning)
        source = torch.cat([h_trk, u_pix, pix_plane_tok, cal_tok, spec_cand_tok], dim=1)
        source_mask = torch.cat([trk_mask, pix_mask, pix_plane_act, cal_mask, batch['trk_cand_mask']], dim=1)
        cobs_q = self.cobs_query(batch['c_obs']).unsqueeze(1)
        latents = torch.cat([self.latent_queries.expand(b, -1, -1), cobs_q], dim=1)
        latent_mask = torch.ones((b, self.n_queries + 1), dtype=torch.bool, device=device)

        if self.enable_readout_gate and tok_rho_trk is not None and self.gn2_perceiver_beta is not None:
            n_src = int(source.shape[1])
            perc_bias = torch.zeros((b, self.heads, 1, n_src), dtype=torch.float32, device=device)
            perc_bias[:, :, :, :m_trk] = self.gn2_perceiver_beta.float() * torch.log(tok_rho_trk)[:, None, None, :]
            for blk in self.global_blocks:
                latents = blk.cross(latents, source, source_mask, pair_bias=perc_bias, query_mask=latent_mask)
                latents = blk.self_attn(latents, latents, latent_mask)
        else:
            for blk in self.global_blocks:
                latents = blk(latents, source, source_mask, latent_mask)

        q_lat = latents[:, :self.n_queries]
        q_cobs = latents[:, self.n_queries]
        with torch.amp.autocast(device_type=device.type, enabled=False):
            w_pool = torch.softmax(self.pool_gate(q_lat.float()), dim=1)
            pool_attn = (w_pool * q_lat.float()).sum(dim=1)

            sv_norm = torch.sqrt(sv_disp_mm.float().square().sum(dim=-1, keepdim=True) + 1e-8)
            vtx_summary = torch.cat([
                torch.log1p(delta_chi2_2v.float().clamp_min(0.0)) / 4.0,
                torch.asinh(sv_norm / 0.2) / 4.0,
                torch.asinh(sv_disp_mm.float()[:, 2:3] / 1.0) / 4.0,
                torch.log(scale_pv_mm.float().clamp(min=0.02, max=150.0)) / 4.0,
                (v_pv_mm.float()[:, 2:3] - self.z_mid) / 1000.0,
                (v_pv_mm.float()[:, 0:1] - self.x_pix_center) / 135.0,
                torch.asinh((v_pv_mm.float()[:, 1:2] - self.y_scint_center) / 100.0),
            ], dim=-1)
            vtx_emb = self.vtx_summary_proj(vtx_summary)

            fuse_list = [
                pool_attn,
                z_prof_glob,
                cal_global.float(),
                trk_glob,
                spec_glob.float(),
                q_cobs.float(),
                vtx_emb,
            ]
            if self.enable_readout_gate and self.gn2_topo_proj is not None:
                n_act_t = tm_f.sum(dim=1).clamp_min(1.0)
                has_t = trk_mask.any(dim=1).float()
                if self.enable_origin_purity:
                    wr_d = hat_w_real.detach()
                    ppv_d = hat_p_pv.detach()
                    ptau_d = hat_p_tau.detach()
                    pchm_d = hat_p_charm.detach()
                    pcnv_d = hat_p_conv.detach()
                    pdisp_d = hat_p_disp_hf.detach()

                    w_tau_r = wr_d * ptau_d
                    w_chm_r = wr_d * pchm_d
                    w_disp_r = wr_d * pdisp_d * (1.0 - pcnv_d)
                    disp_gate = torch.sigmoid(s_d0_raw - 2.0) * tm_f

                    mean_wr = (wr_d.sum(dim=1) / n_act_t) * has_t
                    max_wr = wr_d.max(dim=1).values * has_t
                    cnt_wr = torch.log1p(wr_d.sum(dim=1)) / 4.0
                    sum_pv = torch.log1p((wr_d * ppv_d).sum(dim=1)) / 4.0
                    max_tau = w_tau_r.max(dim=1).values * has_t
                    top3_tau = torch.topk(w_tau_r, k=min(3, m_trk), dim=-1).values.mean(dim=-1) * has_t
                    sum_tau = torch.log1p(w_tau_r.sum(dim=1)) / 3.0
                    max_chm = w_chm_r.max(dim=1).values * has_t
                    max_cnv = pcnv_d.max(dim=1).values * has_t
                    disp_cnv = ((pcnv_d * disp_gate).sum(dim=1) / disp_gate.sum(dim=1).clamp_min(1.0)) * (disp_gate.sum(dim=1) > 0.1).float()
                else:
                    mean_wr = trk_kin.new_zeros((b,))
                    max_wr = trk_kin.new_zeros((b,))
                    cnt_wr = trk_kin.new_zeros((b,))
                    sum_pv = trk_kin.new_zeros((b,))
                    max_tau = trk_kin.new_zeros((b,))
                    top3_tau = trk_kin.new_zeros((b,))
                    sum_tau = trk_kin.new_zeros((b,))
                    max_chm = trk_kin.new_zeros((b,))
                    max_cnv = trk_kin.new_zeros((b,))
                    disp_cnv = trk_kin.new_zeros((b,))
                    w_disp_r = tm_f
                    disp_gate = torch.sigmoid(s_d0_raw - 2.0) * tm_f

                if self.enable_covertex and hat_C_ab is not None:
                    c_d = hat_C_ab.detach()
                    w_d = disp_gate * (w_disp_r if self.enable_origin_purity else tm_f)
                    pair_disp = c_d * w_d[:, :, None] * w_d[:, None, :]
                    cov_disp_sum = torch.log1p(pair_disp.sum(dim=(1, 2)) * 0.5) / 3.0
                    cov_disp_max = pair_disp.amax(dim=(1, 2)) * has_t
                else:
                    cov_disp_sum = trk_kin.new_zeros((b,))
                    cov_disp_max = trk_kin.new_zeros((b,))

                topo_vec = torch.stack([
                    mean_wr, max_wr, cnt_wr, sum_pv,
                    max_tau, top3_tau, sum_tau, max_chm,
                    max_cnv, disp_cnv, cov_disp_sum, cov_disp_max,
                ], dim=-1)
                fuse_list.append(self.gn2_topo_proj(topo_vec))

            fused = self.global_fusion(torch.cat(fuse_list, dim=-1))
            logits7 = self.event_head_7(fused)
            probs4 = marginal_four_from_seven(logits7, is_probs=False)
            charm_logit = self.charm_head(fused).squeeze(-1)
            kin_pred = self.kin_head(fused)

        aux_out: dict[str, Any] = {
            'delta_chi2_2v': delta_chi2_2v.float(),
            'p_sv_trklet': pi_sv.float(),
            'pi_pv': vtx_out['pi_pv'].float(),
            'pi_sv': pi_sv.float(),
            'n_tracklets': trk_mask.sum(dim=1).float(),
            'trk_mask': trk_mask,
            'trk_kin': trk_kin.float(),
            'trk_morph': trk_morph.float(),
            'v_nn_mm': v_nn_mm.float(),
            'scale_nn_mm': scale_nn_mm.float(),
            'hat_w_real': hat_w_real,
            'hat_p_disp_hf': hat_p_disp_hf,
            'hat_p_conv': hat_p_conv,
            'hat_p_pv': hat_p_pv,
        }
        if origin_logits is not None:
            aux_out['tracklet_origin_logits'] = origin_logits
        if purity_logit is not None:
            aux_out['tracklet_purity_logit'] = purity_logit
        if covertex_logits is not None and hat_C_ab is not None:
            aux_out['pairwise_covertex_logits'] = covertex_logits
            aux_out['hat_C_ab'] = hat_C_ab

        return NextGenModelOutput(
            logits7=logits7,
            probs4=probs4,
            charm_logit=charm_logit,
            v_pv_mm=v_pv_mm.float(),
            scale_pv_mm=scale_pv_mm.float(),
            sv_disp_mm=sv_disp_mm.float(),
            kin_pred=kin_pred.float(),
            aux=aux_out,
        )


def compute_gn2_window_denominators(
    micro_batches: list[dict[str, Any]],
    *,
    pos_weight: float = 3.0,
    origin_weights: tuple[float, ...] = GN2_DEFAULT_ORIGIN_WEIGHTS,
) -> dict[str, float]:
    """Compute exact full-window denominators across microbatches for base + GN2 dense tasks."""
    d_proc = 0.0
    d_charm = 0.0
    d_vtx = 0.0
    d_kin = 0.0
    d_pur_pos = 0.0
    d_pur_neg = 0.0
    d_orig_w = 0.0
    d_cov_pos = 0.0
    d_cov_neg = 0.0
    ow = torch.tensor(origin_weights, dtype=torch.float32)

    for mb in micro_batches:
        sw = mb.get('sample_weight')
        sw_f = sw.float().clamp_min(0.0) if sw is not None else None
        lbl4 = mb['label'].long()
        if 'label_7' in mb:
            valid_p = (mb['label_7'] >= 0) | ((mb['label_7'] < 0) & (lbl4 == 2))
        else:
            valid_p = torch.ones_like(lbl4, dtype=torch.bool)
        if valid_p.any():
            d_proc += float((sw_f[valid_p].sum() if sw_f is not None else valid_p.float().sum()).item())

        if 'charm_valid' in mb and 'charm_flag' in mb:
            vc = mb['charm_valid'] > 0
            if vc.any():
                yc = mb['charm_flag'][vc].float().view(-1)
                d_charm += float(torch.where(yc > 0.5, float(pos_weight), 1.0).sum().item())

        if 'vertex_valid' in mb:
            vv = mb['vertex_valid'] > 0
            if vv.any():
                d_vtx += float(vv.float().sum().item())

        if 'kin_valid' in mb:
            kv = mb['kin_valid'] > 0
            if kv.any():
                d_kin += float(kv.float().sum().item())

        gn2_t = derive_tracklet_gn2_targets(mb)
        if gn2_t is not None:
            gv = gn2_t['trk_gn2_valid']
            if gv.any():
                yp = gn2_t['trk_purity_target'][gv] > 0.5
                d_pur_pos += float(yp.float().sum().item())
                d_pur_neg += float((~yp).float().sum().item())
                yo = gn2_t['trk_origin_target'][gv]
                d_orig_w += float(ow.to(yo.device)[yo].sum().item())
            cv = gn2_t['trk_covertex_valid']
            if cv.any():
                yc_p = gn2_t['trk_covertex_target'][cv] > 0.5
                d_cov_pos += float(yc_p.float().sum().item())
                d_cov_neg += float((~yc_p).float().sum().item())

    return {
        'proc': max(d_proc, 1e-8),
        'charm': max(d_charm, 1e-8),
        'vtx': max(d_vtx, 1e-8),
        'kin': max(d_kin, 1e-8),
        'gn2_pur_pos': d_pur_pos,
        'gn2_pur_neg': d_pur_neg,
        'gn2_orig_w': max(d_orig_w, 1e-8),
        'gn2_cov_pos': d_cov_pos,
        'gn2_cov_neg': d_cov_neg,
    }


def gn2_tracklet_ipformer_loss(
    out: NextGenModelOutput,
    batch: dict[str, torch.Tensor],
    *,
    fixed_denoms: dict[str, float] | None = None,
    w_proc: float = 1.0,
    w_charm: float = 0.25,
    w_vtx: float = 0.20,
    w_kin: float = 0.15,
    w_origin: float = 0.10,
    w_purity: float = 0.15,
    w_covertex: float = 0.15,
    origin_weights: tuple[float, ...] = GN2_DEFAULT_ORIGIN_WEIGHTS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute 4-head base multitask loss + ATLAS GN2 dense auxiliary supervision in FP32."""
    fd = fixed_denoms or {}
    sw = batch.get('sample_weight')

    if 'label_7' in batch:
        l_proc = marginal_tau_branch_ce(
            out.logits7,
            batch['label_7'],
            batch['label'],
            sample_weight=sw,
            denom_override=fd.get('proc'),
        )
    else:
        per_p = F.nll_loss(torch.log(out.probs4.float().clamp_min(1e-7)), batch['label'].long(), reduction='none')
        denom_p = (
            torch.as_tensor(fd['proc'], dtype=torch.float32, device=per_p.device).clamp_min(1e-8)
            if 'proc' in fd
            else torch.tensor(float(len(per_p)), dtype=torch.float32, device=per_p.device).clamp_min(1e-8)
        )
        l_proc = per_p.sum() / denom_p

    if 'charm_flag' in batch and 'charm_valid' in batch:
        l_charm = focal_charm_bce(
            out.charm_logit,
            batch['charm_flag'],
            batch['charm_valid'],
            gamma=1.5,
            pos_weight=3.0,
            sample_weight=None,
            denom_override=fd.get('charm'),
        )
    else:
        l_charm = out.charm_logit.sum() * 0.0

    if 'vertex_xyz_mm' in batch and 'vertex_valid' in batch:
        l_vtx = student_t_vertex_loss(
            out.v_pv_mm,
            out.scale_pv_mm,
            batch['vertex_xyz_mm'],
            batch['vertex_valid'],
            df=3.0,
            sample_weight=None,
            denom_override=fd.get('vtx'),
        )
        if 'v_nn_mm' in out.aux and 'scale_nn_mm' in out.aux:
            l_vtx = l_vtx + 0.15 * student_t_vertex_loss(
                out.aux['v_nn_mm'],
                out.aux['scale_nn_mm'],
                batch['vertex_xyz_mm'],
                batch['vertex_valid'],
                df=3.0,
                sample_weight=None,
                denom_override=fd.get('vtx'),
            )
    else:
        l_vtx = (out.v_pv_mm.sum() + out.scale_pv_mm.sum()) * 0.0

    if out.kin_pred is not None and 'vis_kinematics' in batch and 'kin_valid' in batch:
        kv = batch['kin_valid'] > 0
        if kv.any():
            with torch.amp.autocast(device_type=out.kin_pred.device.type, enabled=False):
                ke = F.smooth_l1_loss(
                    out.kin_pred[kv].float(),
                    batch['vis_kinematics'][kv].float(),
                    reduction='none',
                ).mean(dim=-1)
                denom_k = (
                    torch.as_tensor(fd['kin'], dtype=torch.float32, device=ke.device).clamp_min(1e-8)
                    if 'kin' in fd
                    else torch.tensor(float(kv.sum().item()), dtype=torch.float32, device=ke.device).clamp_min(1e-8)
                )
                l_kin = ke.sum() / denom_k
        else:
            l_kin = out.kin_pred.sum() * 0.0
    else:
        l_kin = out.logits7.sum() * 0.0

    # GN2 Dense Auxiliary Losses (targets derived strictly inside training loss!)
    l_origin = out.logits7.sum() * 0.0
    l_purity = out.logits7.sum() * 0.0
    l_covertex = out.logits7.sum() * 0.0

    gn2_t = (
        derive_tracklet_gn2_targets(batch)
        if ('tracklet_origin_logits' in out.aux or 'pairwise_covertex_logits' in out.aux)
        else None
    )
    if gn2_t is not None:
        with torch.amp.autocast(device_type=out.logits7.device.type, enabled=False):
            gv = gn2_t['trk_gn2_valid']
            if 'tracklet_origin_logits' in out.aux and gv.any():
                og_lg = out.aux['tracklet_origin_logits'][gv].float()
                og_y = gn2_t['trk_origin_target'][gv].long()
                ow = torch.tensor(origin_weights, dtype=torch.float32, device=og_lg.device)
                ce_per = F.cross_entropy(og_lg, og_y, reduction='none')
                w_tok = ow[og_y]
                denom_o = (
                    torch.as_tensor(fd['gn2_orig_w'], dtype=torch.float32, device=og_lg.device).clamp_min(1e-8)
                    if 'gn2_orig_w' in fd
                    else w_tok.sum().clamp_min(1e-8)
                )
                l_origin = (ce_per * w_tok).sum() / denom_o
            elif 'tracklet_origin_logits' in out.aux:
                l_origin = out.aux['tracklet_origin_logits'].sum() * 0.0

            if 'tracklet_purity_logit' in out.aux and gv.any():
                pur_lg = out.aux['tracklet_purity_logit'].squeeze(-1)[gv].float()
                pur_y = gn2_t['trk_purity_target'][gv].float()
                bce_p = F.binary_cross_entropy_with_logits(pur_lg, pur_y, reduction='none')
                pos_p = pur_y > 0.5
                neg_p = ~pos_p
                d_pos = fd['gn2_pur_pos'] if 'gn2_pur_pos' in fd else float(pos_p.sum().item())
                d_neg = fd['gn2_pur_neg'] if 'gn2_pur_neg' in fd else float(neg_p.sum().item())
                if d_pos > 0.0 and d_neg > 0.0:
                    term_pos = (bce_p[pos_p].sum() / d_pos) if pos_p.any() else pur_lg.sum() * 0.0
                    term_neg = (bce_p[neg_p].sum() / d_neg) if neg_p.any() else pur_lg.sum() * 0.0
                    l_purity = 0.5 * (term_pos + term_neg)
                elif d_pos > 0.0:
                    l_purity = bce_p[pos_p].sum() / d_pos if pos_p.any() else pur_lg.sum() * 0.0
                elif d_neg > 0.0:
                    l_purity = bce_p[neg_p].sum() / d_neg if neg_p.any() else pur_lg.sum() * 0.0
            elif 'tracklet_purity_logit' in out.aux:
                l_purity = out.aux['tracklet_purity_logit'].sum() * 0.0

            cv = gn2_t['trk_covertex_valid']
            if 'pairwise_covertex_logits' in out.aux and cv.any():
                cov_lg = out.aux['pairwise_covertex_logits'].squeeze(-1)[cv].float()
                cov_y = gn2_t['trk_covertex_target'][cv].float()
                bce_c = F.binary_cross_entropy_with_logits(cov_lg, cov_y, reduction='none')
                disp_sv = gn2_t.get('trk_covertex_is_disp_sv')
                if disp_sv is not None:
                    sv_m = disp_sv[cv]
                    bce_c = torch.where(sv_m, 3.0 * bce_c, bce_c)
                pos_c = cov_y > 0.5
                neg_c = ~pos_c
                dc_pos = fd['gn2_cov_pos'] if 'gn2_cov_pos' in fd else float(pos_c.sum().item())
                dc_neg = fd['gn2_cov_neg'] if 'gn2_cov_neg' in fd else float(neg_c.sum().item())
                if dc_pos > 0.0 and dc_neg > 0.0:
                    t_pos = (bce_c[pos_c].sum() / dc_pos) if pos_c.any() else cov_lg.sum() * 0.0
                    t_neg = (bce_c[neg_c].sum() / dc_neg) if neg_c.any() else cov_lg.sum() * 0.0
                    l_covertex = 0.5 * (t_pos + t_neg)
                elif dc_pos > 0.0:
                    l_covertex = bce_c[pos_c].sum() / dc_pos if pos_c.any() else cov_lg.sum() * 0.0
                elif dc_neg > 0.0:
                    l_covertex = bce_c[neg_c].sum() / dc_neg if neg_c.any() else cov_lg.sum() * 0.0
            elif 'pairwise_covertex_logits' in out.aux:
                l_covertex = out.aux['pairwise_covertex_logits'].sum() * 0.0

    total = (
        w_proc * l_proc.float()
        + w_charm * l_charm.float()
        + w_vtx * l_vtx.float()
        + w_kin * l_kin.float()
        + w_origin * l_origin.float()
        + w_purity * l_purity.float()
        + w_covertex * l_covertex.float()
    )
    return total, out.probs4
