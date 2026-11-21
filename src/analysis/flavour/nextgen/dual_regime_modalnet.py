"""Proposal D: DualRegimeModalNet — Detector-Regime-Factorized Experts, Tomographic Sinkhorn + ToF,
and Cross-Magnet-0 4D IPT-Spectrometer Interlock with Precision-Gated Multi-Modal Fusion.

Strictly observation-only during inference; satisfies NextGenFlavourModel / Gate A-F contracts.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from analysis.flavour.nextgen.protocols import (
    NextGenFlavourModel,
    NextGenModelOutput,
    marginal_four_from_seven,
)
from analysis.flavour.nextgen.tracklet_ops import (
    DifferentiableStudentTVertexLayer,
    TRK_KIN_DIM,
    TRK_MORPH_DIM,
    compute_ipt_spectrometer_interlock,
    compute_scint_tof_residual_ns,
    extract_soft_tracklets_batch,
    extract_soft_tracklets_single,
)
from analysis.flavour.v3.common.domains import (
    calibrated_bar_channels,
    cumulative_radiation_length,
    detector_bounds,
)
from analysis.flavour.v3.common.layers import ScaledQKNormAttention, split_adamw_param_groups
from analysis.flavour.v3.v1_b1pp.models import TransverseLayerEncoder
from analysis.flavour.v3.v1_b1pp.spectrometer import (
    CAND_Y_FLIP_IDX,
    MAGNET0_LENGTH_MM,
    MAGNET0_Z_CENTER_MM,
    MAGNET0_Z_MAX_MM,
    MAGNET0_Z_MIN_MM,
    SAGITTA_CONST_MEV_MM,
    SAGITTA_INTEGRAL_MM2,
    SIGMA_DELTA_X_MM,
    SIGMA_SAGITTA_Y_MM,
    TRACKER_Z_MM,
    back_extrapolate_y_through_magnet0,
    check_magnet12_bore_traversal,
    DELTA_Z_MM,
    ENTRY_SLOPE_SAGITTA_COEFF,
)
from analysis.flavour.v3.v2_pinpointformer.attention import AttentionStack, PerceiverCrossBlock
from analysis.flavour.v3.v2_pinpointformer.pixel_tokenizer import (
    MicroPatchConv2D,
    PIX_FEAT_DIM,
    PIX_PLANE_SUMMARY_DIM,
    apply_y_mirror_pf_item,
    collate_pf,
    extract_pf_item,
)
from analysis.flavour.v3.v2_pinpointformer.stereo_bars import static_scint_stations


N_SPEC_CAND_MAX = 8
SINKHORN_K_TOP = 12
SINKHORN_N_NEXUS = 96
K_FRONT_MAX = 160
K_MID_MAX = 128
K_IPT_MAX = 64
M_TRK_MAX = 64


def extract_nonexclusive_spec_candidates(
    tracker_obs: dict,
    cal_start: tuple[float, float, float, bool],
    bounds: dict,
    max_candidates: int = N_SPEC_CAND_MAX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract up to `max_candidates` non-exclusive 3-station + 2-station spectrometer triplets.

    Unlike greedy `extract_spectrometer_kinematics`, allows shared hits across candidate triplets
    so a real muon track is never suppressed by a random noise hit on one station (`TrkD2%+N1`).
    Returns:
      cand: (max_candidates, 14) float32
      cand_mask: (max_candidates,) bool
      station_counts: (3,) float32 [N_s0, N_s1, N_s2]
    """
    xy = np.asarray(tracker_obs['xy_mm'], dtype=np.float64).reshape(-1, 2)
    station = np.asarray(tracker_obs['station'], dtype=np.int64)
    n_hits = len(station)

    cand = np.zeros((max_candidates, 14), dtype=np.float32)
    cand_mask = np.zeros(max_candidates, dtype=bool)
    station_counts = np.array(
        [float((station == s).sum()) for s in (0, 1, 2)], dtype=np.float32
    )
    if n_hits == 0:
        return cand, cand_mask, station_counts

    x_c = float(bounds['x_pix_center'])
    y_c = float(bounds['y_scint_center'])
    z_start, x_start, y_start, has_start = cal_start

    idx_s = [np.flatnonzero(station == s) for s in (0, 1, 2)]
    max_per_station = 12
    for s in (0, 1, 2):
        ix = idx_s[s]
        if len(ix) > max_per_station:
            ref_x = x_start if has_start else x_c
            ref_y = y_start if has_start else y_c
            dist2 = (xy[ix, 0] - ref_x) ** 2 + (xy[ix, 1] - ref_y) ** 2
            idx_s[s] = ix[np.argsort(dist2, kind='mergesort')[:max_per_station]]

    slot = 0
    if all(len(idx_s[s]) > 0 for s in (0, 1, 2)):
        i0, i1, i2 = idx_s
        g0, g1, g2 = np.meshgrid(i0, i1, i2, indexing='ij')
        g0, g1, g2 = g0.ravel(), g1.ravel(), g2.ravel()

        x0, y0 = xy[g0, 0], xy[g0, 1]
        x1, y1 = xy[g1, 0], xy[g1, 1]
        x2, y2 = xy[g2, 0], xy[g2, 1]

        s_y = y1 - 0.5 * (y0 + y2)
        kappa = -s_y / SAGITTA_INTEGRAL_MM2
        delta_x = x1 - 0.5 * (x0 + x2)
        chi2_x = (delta_x / SIGMA_DELTA_X_MM) ** 2
        theta_x = (x2 - x0) / (2.0 * DELTA_Z_MM)
        theta_y_entry = (y1 - y0) / DELTA_Z_MM + ENTRY_SLOPE_SAGITTA_COEFF * (s_y / DELTA_Z_MM)
        in_bore, _ = check_magnet12_bore_traversal(x0, y0, y1, theta_x, theta_y_entry, kappa)
        kappa_eff = np.where(in_bore, kappa, 0.0)

        if has_start:
            dz_ext = z_start - TRACKER_Z_MM[0]
            dx_ext = (x0 + theta_x * dz_ext) - x_start
            y_ext, _ = back_extrapolate_y_through_magnet0(y0, theta_y_entry, kappa_eff, float(z_start))
            dy_ext = y_ext - y_start
            d_ext = np.hypot(dx_ext / 20.0, dy_ext / 20.0)
        else:
            dx_ext = np.zeros_like(s_y)
            dy_ext = np.zeros_like(s_y)
            d_ext = np.zeros_like(s_y)

        bore_pen = np.where(in_bore, 0.0, 25.0)
        q_score = chi2_x + bore_pen + (0.5 * (d_ext ** 2) if has_start else 0.0) + (np.abs(s_y) / 20.0) ** 2
        # Deterministic Y-mirror-symmetric tie break
        order = np.lexsort((np.abs(y0 - y_c), x0, q_score))

        # Allow up to 2 uses per hit so genuine track survives one shared noise hit
        hit_uses: dict[tuple[int, int], int] = {}
        for k in order:
            a, b, c = int(g0[k]), int(g1[k]), int(g2[k])
            if hit_uses.get((0, a), 0) >= 2 or hit_uses.get((1, b), 0) >= 2 or hit_uses.get((2, c), 0) >= 2:
                continue
            hit_uses[(0, a)] = hit_uses.get((0, a), 0) + 1
            hit_uses[(1, b)] = hit_uses.get((1, b), 0) + 1
            hit_uses[(2, c)] = hit_uses.get((2, c), 0) + 1
            ib_k = bool(in_bore[k])
            sy_k = float(s_y[k]) if ib_k else 0.0
            p_proxy = (1.0 / (abs(sy_k) / 0.5 + 0.01)) if ib_k else 0.0
            cand[slot] = np.array([
                1.0 if ib_k else 0.70,
                1.0 / (1.0 + 0.1 * float(q_score[k])),
                (float(x0[k]) - x_c) / 150.0,
                (float(y0[k]) - y_c) / 150.0,
                float(theta_x[k]) * 10.0,
                float(theta_y_entry[k]) * 10.0,
                (math.asinh(sy_k / SIGMA_SAGITTA_Y_MM) / 5.0) if ib_k else 0.0,
                (float(np.clip(sy_k, -25.0, 25.0)) / 5.0) if ib_k else 0.0,
                (math.log1p(p_proxy) / 5.0) if ib_k else 0.0,
                math.log1p(float(chi2_x[k])) / 5.0,
                float(np.clip(dx_ext[k], -300.0, 300.0)) / 50.0,
                float(np.clip(dy_ext[k], -300.0, 300.0)) / 50.0,
                math.log1p(float(d_ext[k])) / 3.0,
                (math.log1p(abs(sy_k) / 0.05) / 5.0) if ib_k else 0.0,
            ], dtype=np.float32)
            cand_mask[slot] = True
            slot += 1
            if slot >= max_candidates:
                break

    if slot < max_candidates:
        pair_records = []
        for sa, sb in ((0, 2), (0, 1), (1, 2)):
            rem_a, rem_b = idx_s[sa], idx_s[sb]
            if len(rem_a) == 0 or len(rem_b) == 0:
                continue
            ga, gb = np.meshgrid(rem_a, rem_b, indexing='ij')
            ga, gb = ga.ravel(), gb.ravel()
            xa, ya = xy[ga, 0], xy[ga, 1]
            xb, yb = xy[gb, 0], xy[gb, 1]
            dz_ab = TRACKER_Z_MM[sb] - TRACKER_Z_MM[sa]
            tx = (xb - xa) / dz_ab
            ty = (yb - ya) / dz_ab
            if has_start:
                dz_ext = z_start - TRACKER_Z_MM[sa]
                dx_ext = (xa + tx * dz_ext) - x_start
                dy_ext = (ya + ty * dz_ext) - y_start
                d_ext = np.hypot(dx_ext / 20.0, dy_ext / 20.0)
            else:
                dx_ext = np.zeros_like(tx)
                dy_ext = np.zeros_like(tx)
                d_ext = np.zeros_like(tx)
            q2 = 10.0 + (0.5 * (d_ext ** 2) if has_start else 0.0) + (tx * 10.0) ** 2 + (ty * 10.0) ** 2
            for k in range(len(ga)):
                pair_records.append((
                    round(float(q2[k]), 6),
                    round(float(xa[k]), 4),
                    round(abs(float(ya[k] - y_c)), 4),
                    sa, sb, int(ga[k]), int(gb[k]),
                    float(xa[k]), float(ya[k]), float(tx[k]), float(ty[k]),
                    float(dx_ext[k]), float(dy_ext[k]), float(d_ext[k]),
                ))
        pair_records.sort()
        seen_p: set[tuple[int, int, int, int]] = set()
        for rec in pair_records:
            q2_k, _, _, sa, sb, a, b, xa_k, ya_k, tx_k, ty_k, dxe_k, dye_k, de_k = rec
            if (sa, a, sb, b) in seen_p:
                continue
            seen_p.add((sa, a, sb, b))
            cand[slot] = np.array([
                0.5,
                1.0 / (1.0 + 0.1 * q2_k),
                (xa_k - x_c) / 150.0,
                (ya_k - y_c) / 150.0,
                tx_k * 10.0,
                ty_k * 10.0,
                0.0, 0.0, 0.0,
                float(sb - sa) / 2.0,
                float(np.clip(dxe_k, -300.0, 300.0)) / 50.0,
                float(np.clip(dye_k, -300.0, 300.0)) / 50.0,
                math.log1p(de_k) / 3.0,
                0.0,
            ], dtype=np.float32)
            cand_mask[slot] = True
            slot += 1
            if slot >= max_candidates:
                break

    return cand, cand_mask, station_counts


def extract_dual_regime_item(
    index: int,
    observation: dict,
    label: int | None,
    dom: dict,
    geometry: dict,
    bounds: dict,
    config: dict | None = None,
) -> dict:
    """Pure-observation item extractor extending `extract_pf_item` with bar ToF, 8-cand spec, and soft tracklets."""
    cfg = config or {}
    item = extract_pf_item(index, observation, label, dom, geometry, bounds, cfg)

    bar = observation['bar']
    keep_bar = np.asarray(item['bar_keep_mask'], dtype=bool)
    if len(keep_bar) and 'hit_time_ns' in bar:
        t_raw = np.asarray(bar['hit_time_ns'], dtype=np.float32)
        bar_time = t_raw[keep_bar].astype(np.float32)
    else:
        bar_time = np.full(len(item['bar_panel']), -1.0, dtype=np.float32)
    item['bar_time_ns'] = bar_time

    cand8, cand8_mask, st_counts = extract_nonexclusive_spec_candidates(
        observation['tracker_hypothesis'], dom['cal_start'], bounds, max_candidates=N_SPEC_CAND_MAX
    )
    item['trk_cand'] = cand8
    item['trk_cand_mask'] = cand8_mask
    item['trk_station_counts'] = st_counts

    plane_x0 = bounds.get('_plane_x0')
    if plane_x0 is None:
        plane_x0 = np.array(
            [cumulative_radiation_length(float(z), geometry) for z in bounds['pixel_Zpos']],
            dtype=np.float32,
        )
    pitch_xy = (float(bounds['dx_mm']), float(bounds['dy_mm']))
    trk_res = extract_soft_tracklets_single(
        item['pix_xyz_mm'],
        item['pix_plane'],
        item['pix_feat'],
        plane_x0,
        max_tracklets=M_TRK_MAX,
        pitch_xy_mm=pitch_xy,
    )
    item['trk_kin'] = trk_res['trk_kin']
    item['trk_morph'] = trk_res['trk_morph']
    item['trk_mask'] = trk_res['trk_mask']
    return item


def apply_y_mirror_dual_regime_item(
    item: dict,
    y_scint_center: float = -12.0,
    n_bars: int = 40,
) -> dict:
    """Apply exact detector-centered Y-reflection (y -> 2*y_c - y) to DualRegime item."""
    out = apply_y_mirror_pf_item(item, y_scint_center=y_scint_center, n_bars=n_bars)
    yc = float(y_scint_center)
    if 'bar_time_ns' in item:
        out['bar_time_ns'] = item['bar_time_ns'].copy()
    if 'trk_station_counts' in item:
        out['trk_station_counts'] = item['trk_station_counts'].copy()
    if 'trk_kin' in item:
        tk = item['trk_kin'].copy()
        tm = item['trk_mask']
        if tm.any():
            tk[tm, 1] = np.float32(2.0 * yc) - tk[tm, 1]
            tk[tm, 4] = -tk[tm, 4]
        out['trk_kin'] = tk
        out['trk_morph'] = item['trk_morph'].copy()
        out['trk_mask'] = tm.copy()
    return out


def collate_dual_regime(
    events: list[dict],
    mirror_rng: np.random.Generator | None = None,
    y_scint_center: float = -12.0,
    n_bars: int = 40,
) -> dict:
    """Collate DualRegimeModalNet items (compatible with `collate_pf` batches as a strict superset)."""
    if mirror_rng is not None:
        do_flip = mirror_rng.random(len(events)) < 0.5
        events = [
            apply_y_mirror_dual_regime_item(e, y_scint_center=y_scint_center, n_bars=n_bars) if f else e
            for e, f in zip(events, do_flip)
        ]
    batch = collate_pf(events, mirror_rng=None, y_scint_center=y_scint_center, n_bars=n_bars)
    if 'bar_time_ns' in events[0]:
        t_list = [e['bar_time_ns'].astype(np.float32) for e in events if len(e['bar_time_ns']) > 0]
        batch['bar_time_ns'] = (
            torch.from_numpy(np.concatenate(t_list, axis=0))
            if t_list
            else torch.empty(0, dtype=torch.float32)
        )
    if 'trk_station_counts' in events[0]:
        batch['trk_station_counts'] = torch.from_numpy(
            np.stack([e['trk_station_counts'] for e in events]).astype(np.float32)
        )
    if 'trk_kin' in events[0]:
        batch['trk_kin'] = torch.from_numpy(np.stack([e['trk_kin'] for e in events]).astype(np.float32))
        batch['trk_morph'] = torch.from_numpy(np.stack([e['trk_morph'] for e in events]).astype(np.float32))
        batch['trk_mask'] = torch.from_numpy(np.stack([e['trk_mask'] for e in events]).astype(bool))
    return batch


def _select_masked_tokens(
    scores: torch.Tensor,
    mask: torch.Tensor,
    k_max: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select up to `k_max` valid tokens deterministically by score, returning (idx, sel_mask)."""
    b, n = scores.shape
    k_eff = min(int(k_max), n)
    s_masked = torch.where(mask, scores, torch.full_like(scores, -1e9))
    top_s, top_idx = torch.topk(s_masked, k=k_eff, dim=1)
    sel_mask = top_s > -1e8
    safe_idx = torch.where(sel_mask, top_idx, torch.zeros_like(top_idx))
    return safe_idx, sel_mask


def _solve_weighted_pv_with_prior(
    trk_kin: torch.Tensor,
    w_rob: torch.Tensor,
    v_anchor: torch.Tensor,
    p_anchor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytic 3D weighted tracklet + anchor fusion returning (v_pv_mm, diag_precision_mm_inv2)."""
    from analysis.flavour.nextgen.tracklet_ops import _solve_sym_3x3

    x0 = trk_kin[..., 0]
    y0 = trk_kin[..., 1]
    z0 = trk_kin[..., 2]
    tx = trk_kin[..., 3]
    ty = trk_kin[..., 4]
    sxx = trk_kin[..., 5].clamp_min(4e-5)
    syy = trk_kin[..., 6].clamp_min(4e-5)
    stx = trk_kin[..., 7].clamp_min(1e-7)
    sty = trk_kin[..., 8].clamp_min(1e-7)

    dz = (v_anchor[:, None, 2] - z0).clamp(-350.0, 350.0)
    var_x = sxx + dz.square() * stx
    var_y = syy + dz.square() * sty
    wx = w_rob / var_x
    wy = w_rob / var_y
    cx = x0 - tx * z0
    cy = y0 - ty * z0

    m00 = p_anchor[:, 0] + wx.sum(dim=1)
    m11 = p_anchor[:, 1] + wy.sum(dim=1)
    m02 = -(wx * tx).sum(dim=1)
    m12 = -(wy * ty).sum(dim=1)
    m22 = p_anchor[:, 2] + (wx * tx.square() + wy * ty.square()).sum(dim=1)
    zeros = torch.zeros_like(m00)
    mat = torch.stack([
        torch.stack([m00, zeros, m02], dim=-1),
        torch.stack([zeros, m11, m12], dim=-1),
        torch.stack([m02, m12, m22], dim=-1),
    ], dim=-2)
    rhs = torch.stack([
        p_anchor[:, 0] * v_anchor[:, 0] + (wx * cx).sum(dim=1),
        p_anchor[:, 1] * v_anchor[:, 1] + (wy * cy).sum(dim=1),
        p_anchor[:, 2] * v_anchor[:, 2] - (wx * tx * cx + wy * ty * cy).sum(dim=1),
    ], dim=-1)
    v_sol, inv_diag = _solve_sym_3x3(mat, rhs)
    p_diag = torch.stack([m00, m11, m22], dim=-1).clamp_min(1e-6)
    return v_sol, p_diag


class DualRegimeModalNet(NextGenFlavourModel):
    """Proposal D: DualRegimeModalNet (~6.20M params).

    Four physics-factorized regime experts:
      1. FrontMicroTrackerExpert (Planes 0..5, 30.01 mm): PixDark multi-plane persistence gate,
         causal cross-plane micro-tracking attention, front Student-t PV precision estimator,
         and 2 boundary bridge tokens to Regime 2.
      2. MidTomographicSinkhornExpert (Fortune 0..6 + Planes 6..11, 150.01 mm): Per-event relativistic
         ToF residuals (`hit_time_ns`), 2D X/Y bar convs, 3-step entropic Sinkhorn-Knopp X-Y stereo
         de-ghosting with `(sum_E_X > 0) & (sum_E_Y > 0)` & `clamp_min(1e-8)` guards, mid-silicon
         tomographic slice attention, and mid-regime PV + precision estimator.
      3. IPTSpectrometerInterlockExpert (Planes 12..14 + Trackers 0..2): Cross-Magnet-0 4D trajectory
         interlock (`compute_ipt_spectrometer_interlock`), multi-candidate reliability gating `rho_spec`,
         and soft reliability-weighted muon kinematics.
      4. PrecisionGatedCrossModalFusion: Inverse-covariance PV fusion across regimes, uncertainty-gated
         Perceiver cross-attention, 69-plane longitudinal profile stack, and 7-class/charm/PV/kin heads.
    """

    def __init__(self, geometry: dict, config: dict | None = None) -> None:
        super().__init__()
        cfg = dict(config or {})
        self.config = cfg
        self.dim = int(cfg.get('dim', 160))
        self.heads = int(cfg.get('heads', 8))
        drop_path = float(cfg.get('drop_path', 0.10))
        head_drop = float(cfg.get('head_dropout', 0.15))

        # Causal ablation toggles (all True in full model)
        self.use_ipt_spec_interlock = bool(cfg.get('use_ipt_spec_interlock', True))
        self.use_sinkhorn_tof = bool(cfg.get('use_sinkhorn_tof', True))
        self.enable_pixdark_noise_aug = bool(cfg.get('enable_pixdark_noise_aug', True))
        self.homogeneous_regime = bool(cfg.get('homogeneous_regime', False))
        self.aug_prob = float(cfg.get('degradation_aug_prob', 0.20))

        bounds = detector_bounds(geometry)
        stations = static_scint_stations(geometry, bounds)
        self.n_st = int(stations['n_stations'])
        self.n_dbl = int(stations['n_doublets'])
        self.n_bars = int(stations['n_bars'])
        self.x_pix_c = float(bounds['x_pix_center'])
        self.x_scint_c = float(stations['x_scint_center'])
        self.y_c = float(stations['y_scint_center'])
        self.z_mid = float(stations['z_mid'])
        self.nx = int(bounds['nx'])
        self.ny = int(bounds['ny'])

        pix_z = np.asarray(bounds['pixel_Zpos'], dtype=np.float32)
        pix_x0 = np.array(
            [cumulative_radiation_length(float(z), geometry) for z in pix_z],
            dtype=np.float32,
        )
        self.register_buffer('pix_plane_z', torch.from_numpy(pix_z))
        self.register_buffer('pix_plane_x0', torch.from_numpy(pix_x0))
        self.register_buffer(
            'default_center_mm',
            torch.tensor([self.x_pix_c, self.y_c, self.z_mid], dtype=torch.float32),
        )

        # Static scintillator station buffers
        self.register_buffer('station_z_mm', torch.from_numpy(stations['station_z_mm']))
        self.register_buffer('station_x0', torch.from_numpy(stations['station_x0']))
        self.register_buffer('doublet_stations', torch.from_numpy(stations['doublet_stations']))
        self.register_buffer('panel_to_station', torch.from_numpy(stations['panel_to_station']))
        self.register_buffer('panel_to_xslot', torch.from_numpy(stations['panel_to_xslot']))
        self.register_buffer('panel_to_yslot', torch.from_numpy(stations['panel_to_yslot']))
        self.register_buffer('bar_x_mm', torch.from_numpy(stations['bar_x_centers_mm']))
        self.register_buffer('bar_y_mm', torch.from_numpy(stations['bar_y_centers_mm']))
        self.register_buffer('u_bars', torch.linspace(-1.0, 1.0, self.n_bars))

        z_norm = (self.station_z_mm - self.z_mid) / 1000.0
        x0_norm = self.station_x0 / 50.0
        self.register_buffer('xbar_geom', torch.stack([z_norm, x0_norm], dim=0))
        self.register_buffer('ybar_geom', torch.stack([
            z_norm[self.doublet_stations],
            x0_norm[self.doublet_stations],
        ], dim=0))

        # ------------------------------------------------------------------
        # Shared Silicon Micro-Morphology Front-End
        # ------------------------------------------------------------------
        dim = self.dim
        heads = self.heads
        self.patch_conv = MicroPatchConv2D(out_dim=64)
        self.patch_proj = nn.Sequential(nn.Linear(64, dim), nn.LayerNorm(dim))
        self.pix_in = nn.Sequential(
            nn.Linear(PIX_FEAT_DIM, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.pix_plane_emb = nn.Sequential(
            nn.Linear(PIX_PLANE_SUMMARY_DIM, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
        )

        # ------------------------------------------------------------------
        # Regime 1: Front 30mm Micro-Tracker (Planes 0..5) + PixDark Persistence Gate
        # ------------------------------------------------------------------
        self.pixdark_gate_mlp = nn.Sequential(
            nn.Linear(12, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        self.front_pair_mlp = nn.Sequential(
            nn.Linear(6, 32),
            nn.SiLU(),
            nn.Linear(32, heads),
        )
        self.front_encoder = AttentionStack(
            dim=dim, n_layers=3, heads=heads, init_scale=12.0, max_scale=50.0, drop_path=drop_path
        )
        self.front_bridge_queries = nn.Parameter(torch.randn(1, 2, dim) * 0.02)
        self.front_bridge_attn = ScaledQKNormAttention(
            dim=dim, heads=heads, init_scale=10.0, drop_path=drop_path
        )
        self.front_pool_gate = nn.Linear(dim, 1)

        # Soft tracklet encoder + Student-t PV fitters
        self.trk_in = nn.Sequential(
            nn.Linear(TRK_KIN_DIM + TRK_MORPH_DIM, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.trk_front_cross = ScaledQKNormAttention(
            dim=dim, heads=heads, init_scale=10.0, drop_path=drop_path
        )
        self.front_vtx_layer = DifferentiableStudentTVertexLayer(hidden_dim=dim, df=3.0, n_iters=4)
        self.front_vtx_refine = nn.Sequential(
            nn.Linear(dim, 96),
            nn.SiLU(),
            nn.Linear(96, 6),  # [dx, dy, dz, log_px, log_py, log_pz]
        )

        # ------------------------------------------------------------------
        # Regime 2: Mid 150mm Tomographer (Sinkhorn X-Y + ToF + Mid Si Slices 6..11)
        # ------------------------------------------------------------------
        self.xbar_enc = TransverseLayerEncoder(7, (32, 48, 64), stem_k=3, stem_s=1, out_dim=dim)
        self.ybar_enc = TransverseLayerEncoder(7, (32, 48, 64), stem_k=3, stem_s=1, out_dim=dim)
        self.stereo_2d_proj = nn.Sequential(
            nn.Linear(256, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
        )
        self.station_fuse = nn.Sequential(
            nn.Linear(2 * dim + 7, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
        )
        self.sinkhorn_cost_t = nn.Parameter(torch.tensor(0.8))
        self.sinkhorn_cost_e = nn.Parameter(torch.tensor(1.0))
        self.nexus_in = nn.Sequential(
            nn.Linear(9, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
        )
        self.cal_pair_mlp = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, heads))
        self.bipartite_attn = nn.ModuleList([
            ScaledQKNormAttention(dim, heads=heads, init_scale=12.0, drop_path=drop_path)
            for _ in range(2)
        ])

        self.mid_pix_cross = ScaledQKNormAttention(
            dim=dim, heads=heads, init_scale=10.0, drop_path=drop_path
        )
        self.mid_pix_self = ScaledQKNormAttention(
            dim=dim, heads=heads, init_scale=10.0, drop_path=drop_path
        )
        self.mid_pool_gate = nn.Linear(dim, 1)
        self.mid_vtx_layer = DifferentiableStudentTVertexLayer(hidden_dim=dim, df=3.0, n_iters=4)
        self.mid_vtx_head = nn.Sequential(
            nn.LayerNorm(2 * dim + 4),
            nn.Linear(2 * dim + 4, 128),
            nn.SiLU(),
            nn.Linear(128, 6),  # [v_mid_offset_xyz, log_prec_mid_xyz]
        )
        self.pv_prior_log_prec = nn.Parameter(torch.tensor([-5.0, -5.0, -5.8], dtype=torch.float32))
        self.pv_scale_cal = nn.Parameter(torch.zeros(3, dtype=torch.float32))

        # ------------------------------------------------------------------
        # Regime 3 & 4: IPT (Planes 12..14) <-> Spectrometer Cross-Magnet-0 Interlock
        # ------------------------------------------------------------------
        self.ipt_self = ScaledQKNormAttention(
            dim=dim, heads=heads, init_scale=10.0, drop_path=drop_path
        )
        self.ipt_pool_gate = nn.Linear(dim, 1)

        spec_dim = 128
        self.spec_cand_in = nn.Sequential(
            nn.Linear(14 + 4 + 3, spec_dim),
            nn.LayerNorm(spec_dim),
            nn.SiLU(),
            nn.Linear(spec_dim, spec_dim),
            nn.LayerNorm(spec_dim),
        )
        self.spec_cand_attn1 = ScaledQKNormAttention(spec_dim, heads=4, init_scale=8.0, drop_path=0.05)
        self.spec_cand_attn2 = ScaledQKNormAttention(spec_dim, heads=4, init_scale=8.0, drop_path=0.05)
        self.spec_rel_head = nn.Sequential(
            nn.LayerNorm(spec_dim),
            nn.Linear(spec_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        self.spec_cand_to_dim = nn.Sequential(
            nn.Linear(spec_dim + 4, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
        )
        self.spec_hit_in = nn.Sequential(
            nn.Linear(6, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.spec_hit_score = nn.Linear(64, 1)
        self.spec_glob_fusion = nn.Sequential(
            nn.Linear(spec_dim + 128 + dim + 5, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
        )

        # ------------------------------------------------------------------
        # Longitudinal 69-Plane Profile + Uncertainty-Gated Cross-Modal Fusion
        # ------------------------------------------------------------------
        n_pix_p = len(bounds['pixel_Zpos'])
        all_z = np.concatenate([
            np.asarray(bounds['pixel_Zpos'], dtype=np.float64),
            np.asarray(stations['station_z_mm'], dtype=np.float64),
        ])
        all_x0 = np.concatenate([
            pix_x0.astype(np.float64),
            np.asarray(stations['station_x0'], dtype=np.float64),
        ])
        is_pix = np.r_[np.ones(n_pix_p, dtype=np.float32), np.zeros(self.n_st, dtype=np.float32)]
        z_order = np.argsort(all_z, kind='mergesort')
        z_geom = np.column_stack([
            (all_z[z_order] - self.z_mid) / 1000.0,
            all_x0[z_order] / 50.0,
            is_pix[z_order],
            1.0 - is_pix[z_order],
        ]).astype(np.float32)
        self.register_buffer('z_order_idx', torch.from_numpy(z_order).long())
        self.register_buffer('z_plane_geom', torch.from_numpy(z_geom))
        self.z_pos_proj = nn.Linear(4, dim)
        self.z_profile_encoder = AttentionStack(
            dim=dim, n_layers=2, heads=heads, init_scale=10.0, max_scale=50.0, drop_path=drop_path
        )
        self.z_pool_gate = nn.Linear(dim, 1)

        self.n_queries = int(cfg.get('n_queries', 16))
        self.latent_queries = nn.Parameter(torch.randn(1, self.n_queries, dim) * 0.02)
        self.cobs_query = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(16, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
        )
        self.ugca_beta = nn.Parameter(torch.full((1, heads, 1, 1), 0.35))
        self.global_blocks = nn.ModuleList([
            PerceiverCrossBlock(dim=dim, heads=heads, init_scale=12.0, max_scale=50.0, drop_path=drop_path)
            for _ in range(2)
        ])
        self.pool_gate = nn.Linear(dim, 1)

        self.global_fusion = nn.Sequential(
            nn.LayerNorm(6 * dim),
            nn.Linear(6 * dim, 3 * dim),
            nn.SiLU(),
        )

        # Output Heads (7-class, Charm, Kinematics)
        in_global = 3 * dim
        self.event_head_7 = nn.Sequential(
            nn.LayerNorm(in_global),
            nn.Linear(in_global, 256),
            nn.SiLU(),
            nn.Dropout(head_drop),
            nn.Linear(256, 7),
        )
        self.charm_head = nn.Sequential(
            nn.LayerNorm(in_global),
            nn.Linear(in_global, 192),
            nn.SiLU(),
            nn.Dropout(head_drop),
            nn.Linear(192, 1),
        )
        self.kin_head = nn.Sequential(
            nn.LayerNorm(in_global),
            nn.Linear(in_global, 192),
            nn.SiLU(),
            nn.Linear(192, 4),
        )

    def build_param_groups(self, base_lr: float, weight_decay: float) -> list[dict]:
        return split_adamw_param_groups(self, base_lr, weight_decay)

    def branch_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """Physical branch grouping for Gate-B gradient health audits."""
        groups: dict[str, list[nn.Parameter]] = {}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith(('patch_', 'pix_in.', 'pix_plane_emb.')):
                br = 'pixel_frontend'
            elif name.startswith(('pixdark_', 'front_', 'trk_')):
                br = 'regime1_front_micro'
            elif name.startswith(('xbar_enc.', 'ybar_enc.', 'stereo_2d_', 'station_fuse.', 'sinkhorn_', 'nexus_in.', 'cal_pair_', 'bipartite_', 'mid_', 'pv_')):
                br = 'regime2_mid_sinkhorn'
            elif name.startswith(('ipt_', 'spec_')):
                br = 'regime34_ipt_spec'
            elif name.startswith(('z_', 'latent_queries', 'cobs_query.', 'ugca_', 'global_')):
                br = 'precision_global_fusion'
            else:
                br = 'output_heads'
            groups.setdefault(br, []).append(param)
        return groups

    def _apply_training_degradation_aug(
        self,
        pix_xyz: torch.Tensor,
        pix_feat: torch.Tensor,
        pix_plane: torch.Tensor,
        pix_mask: torch.Tensor,
        trk_cand: torch.Tensor,
        trk_cand_mask: torch.Tensor,
        trk_hits: torch.Tensor,
        trk_hit_mask: torch.Tensor,
        station_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Online stochastic PixDark + tracker noise/dropout curriculum (strictly training-only)."""
        b, n_cl, _ = pix_xyz.shape
        device = pix_xyz.device
        apply_ev = (torch.rand((b,), device=device) < self.aug_prob)
        if not apply_ev.any():
            return apply_ev, pix_xyz, pix_feat, pix_plane, pix_mask, trk_cand, trk_cand_mask, trk_hits, trk_hit_mask, station_counts

        pix_xyz = pix_xyz.clone()
        pix_feat = pix_feat.clone()
        pix_plane = pix_plane.clone()
        pix_mask = pix_mask.clone()
        trk_cand = trk_cand.clone()
        trk_cand_mask = trk_cand_mask.clone()
        trk_hits = trk_hits.clone()
        trk_hit_mask = trk_hit_mask.clone()
        station_counts = station_counts.clone()

        # 1. Mild 2% pixel cluster dropout + 1..4 isolated PixDark clusters in padded slots
        ev_mask_pix = apply_ev.unsqueeze(1) & pix_mask
        drop_pix = ev_mask_pix & (torch.rand((b, n_cl), device=device) < 0.025)
        # Keep at least 1 pixel cluster if event originally had any
        orig_cnt = pix_mask.sum(dim=1, keepdim=True)
        would_empty = (pix_mask & (~drop_pix)).sum(dim=1, keepdim=True) == 0
        drop_pix = drop_pix & (~(would_empty & (orig_cnt > 0)))
        pix_mask = pix_mask & (~drop_pix)

        # Inject 2 isolated PixDark single-pixel clusters into available padded slots
        for _ in range(2):
            has_free = apply_ev & (~pix_mask).any(dim=1)
            if not has_free.any():
                break
            free_pos = (~pix_mask).long().argmax(dim=1)
            idx_b = torch.nonzero(has_free, as_tuple=False).squeeze(-1)
            pos_b = free_pos[idx_b]
            pl_rand = torch.randint(0, 15, (len(idx_b),), device=device)
            xr = torch.empty((len(idx_b),), device=device).uniform_(-125.0, 125.0)
            yr = self.y_c + torch.empty((len(idx_b),), device=device).uniform_(-90.0, 90.0)
            zr = self.pix_plane_z[pl_rand]
            pix_xyz[idx_b, pos_b] = torch.stack([xr, yr, zr], dim=-1)
            pix_plane[idx_b, pos_b] = pl_rand
            dark_feat = torch.zeros((len(idx_b), PIX_FEAT_DIM), device=device, dtype=pix_feat.dtype)
            dark_feat[:, 0] = (xr - self.x_pix_c) / 150.0
            dark_feat[:, 1] = (yr - self.y_c) / 150.0
            dark_feat[:, 2] = (zr - self.z_mid) / 1000.0
            dark_feat[:, 3] = math.log1p(1.0) / 6.0
            dark_feat[:, 6] = 1.0
            dark_feat[:, 7] = math.asinh(0.006 / 0.05) / 3.0
            dark_feat[:, 8] = math.asinh(0.006 / 0.05) / 3.0
            pix_feat[idx_b, pos_b] = dark_feat
            pix_mask[idx_b, pos_b] = True

        # 2. Vectorized tracker noise hit + spurious combinatorial candidate injection (zero CPU sync!)
        for st in (0, 1, 2):
            station_counts[:, st] = station_counts[:, st] + apply_ev.float()
            has_h_free = apply_ev & (~trk_hit_mask).any(dim=1)
            if has_h_free.any():
                h_pos = (~trk_hit_mask).long().argmax(dim=1)
                idx_b = torch.nonzero(has_h_free, as_tuple=False).squeeze(-1)
                pos_b = h_pos[idx_b]
                hx = torch.empty((len(idx_b),), device=device).uniform_(-120.0, 120.0)
                hy = torch.empty((len(idx_b),), device=device).uniform_(-110.0, 140.0)
                noise_hit = torch.zeros((len(idx_b), 6), device=device, dtype=trk_hits.dtype)
                noise_hit[:, 0] = float(st) / 2.0
                noise_hit[:, 1] = (hx - self.x_pix_c) / 150.0
                noise_hit[:, 2] = (hy - self.y_c) / 150.0
                trk_hits[idx_b, pos_b] = noise_hit
                trk_hit_mask[idx_b, pos_b] = True

        has_c_free = apply_ev & (~trk_cand_mask).any(dim=1)
        if has_c_free.any():
            c_pos = (~trk_cand_mask).long().argmax(dim=1)
            idx_b = torch.nonzero(has_c_free, as_tuple=False).squeeze(-1)
            pos_b = c_pos[idx_b]
            n_c = len(idx_b)
            cand_noise = torch.zeros((n_c, 14), device=device, dtype=trk_cand.dtype)
            cand_noise[:, 0] = 1.0
            cand_noise[:, 1] = 0.15
            cand_noise[:, 2] = torch.empty((n_c,), device=device).uniform_(-0.7, 0.7)
            cand_noise[:, 3] = torch.empty((n_c,), device=device).uniform_(-0.7, 0.7)
            cand_noise[:, 4] = torch.empty((n_c,), device=device).uniform_(-0.5, 0.5)
            cand_noise[:, 5] = torch.empty((n_c,), device=device).uniform_(-0.5, 0.5)
            cand_noise[:, 6] = 0.4
            cand_noise[:, 7] = 0.2
            cand_noise[:, 8] = 0.3
            cand_noise[:, 9] = 1.2
            cand_noise[:, 10] = 1.8
            cand_noise[:, 11] = -1.5
            cand_noise[:, 12] = 1.4
            cand_noise[:, 13] = 0.4
            trk_cand[idx_b, pos_b] = cand_noise
            trk_cand_mask[idx_b, pos_b] = True

        return apply_ev, pix_xyz, pix_feat, pix_plane, pix_mask, trk_cand, trk_cand_mask, trk_hits, trk_hit_mask, station_counts

    def _compute_front_persistence_gate(
        self,
        f_xyz: torch.Tensor,
        f_feat: torch.Tensor,
        f_plane: torch.Tensor,
        f_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-plane persistence gate g_persist in (0, 1) suppressing isolated PixDark."""
        b, k, _ = f_xyz.shape
        if not self.enable_pixdark_noise_aug or k == 0:
            return f_mask.float()

        with torch.amp.autocast(device_type=f_xyz.device.type, enabled=False):
            xyz = f_xyz.float()
            dp = torch.abs(f_plane[:, :, None] - f_plane[:, None, :])
            dz = torch.abs(xyz[:, :, None, 2] - xyz[:, None, :, 2]).clamp_min(1.0)
            dx = xyz[:, :, None, 0] - xyz[:, None, :, 0]
            dy = xyz[:, :, None, 1] - xyz[:, None, :, 1]
            d_perp_sq = dx.square() + dy.square()
            pair_ok = f_mask[:, :, None] & f_mask[:, None, :]

            sig1_sq = (0.14 * dz + 0.15).square()
            sig2_sq = (0.18 * dz + 0.25).square()
            w1 = torch.exp(-0.5 * (d_perp_sq / sig1_sq).clamp_max(30.0)) * ((dp == 1) & pair_ok).float()
            w2 = torch.exp(-0.5 * (d_perp_sq / sig2_sq).clamp_max(30.0)) * ((dp == 2) & pair_ok).float()
            cone1 = torch.log1p(w1.sum(dim=-1)) / 2.0
            cone2 = torch.log1p(w2.sum(dim=-1)) / 2.0

            ff = f_feat.float()
            sel_cols = ff[..., [3, 7, 8, 9, 13, 14, 15, 19, 20, 21]]
            gate_in = torch.cat([cone1.unsqueeze(-1), cone2.unsqueeze(-1), sel_cols], dim=-1)
            g_persist = torch.sigmoid(self.pixdark_gate_mlp(gate_in).squeeze(-1)) * f_mask.float()
        return g_persist

    def _encode_mid_sinkhorn_tomography(
        self,
        batch: dict[str, torch.Tensor],
        pix_xyz: torch.Tensor,
        pix_plane: torch.Tensor,
        pix_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Regime-2 2D bar convs + relativistic ToF + 3-step Sinkhorn X-Y stereo de-ghosting."""
        b = int(batch['c_obs'].shape[0])
        device = batch['c_obs'].device
        bar_ev = batch['bar_ev']
        bar_panel = batch['bar_panel']
        bar_horiz = batch['bar_horiz']
        bar_idx = batch['bar_idx']
        bar_val = batch['bar_val'].float()

        if self.use_sinkhorn_tof and 'bar_time_ns' in batch and len(bar_ev) > 0:
            bar_z = self.station_z_mm[self.panel_to_station[bar_panel]]
            bt = batch['bar_time_ns'].float()
            bar_tof = compute_scint_tof_residual_ns(bt, bar_z, valid_mask=(bt >= 0.0), event_idx=bar_ev)
        else:
            bar_tof = bar_val.new_zeros((len(bar_ev),))

        grid_x = torch.zeros((b, 7, self.n_st, self.n_bars), device=device, dtype=torch.float32)
        grid_y = torch.zeros((b, 7, self.n_dbl, self.n_bars), device=device, dtype=torch.float32)

        x_mask_b = (bar_horiz == 0)
        y_mask_b = (bar_horiz == 1)
        if x_mask_b.any():
            ev_x = bar_ev[x_mask_b]
            xs = self.panel_to_xslot[bar_panel[x_mask_b]]
            bx = bar_idx[x_mask_b]
            vx = bar_val[x_mask_b]
            grid_x[ev_x, 0, xs, bx] = vx[:, 0]
            grid_x[ev_x, 1, xs, bx] = vx[:, 1]
            grid_x[ev_x, 4, xs, bx] = 1.0
            grid_x[ev_x, 6, xs, bx] = bar_tof[x_mask_b]
        if y_mask_b.any():
            ev_y = bar_ev[y_mask_b]
            ys = self.panel_to_yslot[bar_panel[y_mask_b]]
            by = bar_idx[y_mask_b]
            vy = bar_val[y_mask_b]
            grid_y[ev_y, 0, ys, by] = vy[:, 0]
            grid_y[ev_y, 1, ys, by] = vy[:, 1]
            grid_y[ev_y, 4, ys, by] = 1.0
            grid_y[ev_y, 6, ys, by] = bar_tof[y_mask_b]

        grid_x[:, 2:4] = self.xbar_geom[None, :, :, None]
        grid_x[:, 5] = grid_x[:, 4] * self.u_bars[None, None, :]
        grid_y[:, 2:4] = self.ybar_geom[None, :, :, None]
        grid_y[:, 5] = grid_y[:, 4] * self.u_bars[None, None, :]

        x_act = grid_x[:, 4] > 0.5
        y_act = grid_y[:, 4] > 0.5
        x_st_act = x_act.any(dim=-1)  # (B, 54)
        y_st_act = y_act.any(dim=-1)  # (B, 42)
        x_tok, x_glob = self.xbar_enc(grid_x, x_st_act)
        y_tok, y_glob = self.ybar_enc(grid_y, y_st_act)
        cal_glob = self.stereo_2d_proj(torch.cat([x_glob, y_glob], dim=-1))

        y_full = torch.zeros_like(x_tok)
        y_st_act_full = torch.zeros_like(x_st_act)
        y_ex = grid_y[:, 0]
        y_has = grid_y[:, 4]
        y_tof = grid_y[:, 6]
        y_ex_full = torch.zeros((b, self.n_st, self.n_bars), device=device)
        y_has_full = torch.zeros((b, self.n_st, self.n_bars), device=device)
        y_tof_full = torch.zeros((b, self.n_st, self.n_bars), device=device)
        y_full[:, self.doublet_stations] = y_tok
        y_st_act_full[:, self.doublet_stations] = y_st_act
        y_ex_full[:, self.doublet_stations] = y_ex
        y_has_full[:, self.doublet_stations] = y_has
        y_tof_full[:, self.doublet_stations] = y_tof

        e_x_st = grid_x[:, 0].sum(dim=-1)
        e_y_st = y_ex_full.sum(dim=-1)
        w_x = grid_x[:, 0] + 1e-4 * grid_x[:, 4]
        w_y = y_ex_full + 1e-4 * y_has_full
        x_cent = (w_x * self.bar_x_mm[None, None, :]).sum(dim=-1) / w_x.sum(dim=-1).clamp_min(1e-6)
        y_cent = (w_y * self.bar_y_mm[None, None, :]).sum(dim=-1) / w_y.sum(dim=-1).clamp_min(1e-6)
        x_cent = torch.where(x_st_act, x_cent, torch.full_like(x_cent, self.x_scint_c))
        y_cent = torch.where(y_st_act_full, y_cent, torch.full_like(y_cent, self.y_c))

        tof_x_st = (w_x * grid_x[:, 6]).sum(dim=-1) / w_x.sum(dim=-1).clamp_min(1e-6) * x_st_act.float()
        tof_y_st = (w_y * y_tof_full).sum(dim=-1) / w_y.sum(dim=-1).clamp_min(1e-6) * y_st_act_full.float()
        asym = (e_x_st - e_y_st) / (e_x_st + e_y_st + 1e-4) * (x_st_act & y_st_act_full).float()
        st_aux = torch.stack([
            e_x_st, e_y_st, asym, x_st_act.float(), y_st_act_full.float(), tof_x_st, tof_y_st
        ], dim=-1)
        st_tok = self.station_fuse(torch.cat([x_tok, y_full, st_aux], dim=-1))
        st_mask = x_st_act | y_st_act_full
        st_tok = st_tok * st_mask.unsqueeze(-1).to(st_tok.dtype)
        st_xyz = torch.stack([x_cent, y_cent, self.station_z_mm[None, :].expand(b, -1)], dim=-1).float()

        # Select Top-K_top bars per side symmetrically
        ex_dbl = grid_x[:, 0, self.doublet_stations]
        act_x_dbl = grid_x[:, 4, self.doublet_stations] > 0.5
        tx_dbl = grid_x[:, 6, self.doublet_stations]
        ey_dbl = y_ex
        act_y_dbl = y_has > 0.5
        ty_dbl = y_tof

        sym_tie = 1e-5 * (1.0 - self.u_bars.abs())[None, None, :]
        score_x = torch.where(act_x_dbl, ex_dbl + 1e-3 + sym_tie, torch.full_like(ex_dbl, -1e6))
        score_y = torch.where(act_y_dbl, ey_dbl + 1e-3 + sym_tie, torch.full_like(ey_dbl, -1e6))
        _, top_ix = torch.topk(score_x, k=SINKHORN_K_TOP, dim=-1)
        _, top_iy = torch.topk(score_y, k=SINKHORN_K_TOP, dim=-1)
        top_vx = torch.gather(act_x_dbl, dim=-1, index=top_ix)
        top_vy = torch.gather(act_y_dbl, dim=-1, index=top_iy)
        top_ex = torch.gather(ex_dbl, dim=-1, index=top_ix) * top_vx.float()
        top_ey = torch.gather(ey_dbl, dim=-1, index=top_iy) * top_vy.float()
        top_tx = torch.gather(tx_dbl, dim=-1, index=top_ix)
        top_ty = torch.gather(ty_dbl, dim=-1, index=top_iy)

        with torch.amp.autocast(device_type=device.type, enabled=False):
            ex_f = top_ex.float()
            ey_f = top_ey.float()
            tx_f = top_tx.float()
            ty_f = top_ty.float()
            pair_valid = top_vx[:, :, :, None] & top_vy[:, :, None, :]

            if self.use_sinkhorn_tof:
                # Dual-active station guard (Round-2 F5 fix): (sum_E_X > 0) & (sum_E_Y > 0) + clamp_min(1e-8)
                sum_ex = ex_f.sum(dim=-1, keepdim=True)
                sum_ey = ey_f.sum(dim=-1, keepdim=True)
                m_dual = ((sum_ex > 0.0) & (sum_ey > 0.0)).squeeze(-1)  # (B, 42)
                p_s = ex_f / sum_ex.clamp_min(1e-8)
                q_s = ey_f / sum_ey.clamp_min(1e-8)

                # Silicon-slice tomographic prior from mid silicon clusters (Planes 6..11)
                x_cand = self.bar_x_mm[top_ix]  # (B, 42, K)
                y_cand = self.bar_y_mm[top_iy]  # (B, 42, K)
                z_dbl = self.station_z_mm[self.doublet_stations]  # (42,)
                mid_si_m = pix_mask & (pix_plane >= 6) & (pix_plane <= 11)
                if mid_si_m.any():
                    px_f = pix_xyz[:, :, 0].float()
                    py_f = pix_xyz[:, :, 1].float()
                    pz_f = pix_xyz[:, :, 2].float()
                    # Only use silicon planes near each station (|dz| <= 180 mm)
                    dz_ps = torch.abs(z_dbl[None, :, None] - pz_f[:, None, :]) <= 180.0
                    w_si = (mid_si_m[:, None, :] & dz_ps).float()  # (B, 42, N_cl)
                    dx_p = x_cand[:, :, :, None] - px_f[:, None, None, :]  # (B, 42, K, N_cl)
                    dy_p = y_cand[:, :, :, None] - py_f[:, None, None, :]
                    gx_p = (torch.exp(-0.5 * (dx_p / 35.0).square().clamp_max(20.0)) * w_si[:, :, None, :]).sum(dim=-1)
                    gy_p = (torch.exp(-0.5 * (dy_p / 35.0).square().clamp_max(20.0)) * w_si[:, :, None, :]).sum(dim=-1)
                    has_mid_near = w_si.sum(dim=-1) > 0.5  # (B, 42)
                    pi_prior = torch.where(
                        has_mid_near[:, :, None, None],
                        gx_p[:, :, :, None] * gy_p[:, :, None, :] + 1e-3,
                        torch.ones((b, self.n_dbl, SINKHORN_K_TOP, SINKHORN_K_TOP), device=device),
                    )
                else:
                    pi_prior = torch.ones((b, self.n_dbl, SINKHORN_K_TOP, SINKHORN_K_TOP), device=device)

                dt_xy = torch.abs(tx_f[:, :, :, None] - ty_f[:, :, None, :])
                de_xy = torch.abs(ex_f[:, :, :, None] - ey_f[:, :, None, :])
                ct = F.softplus(self.sinkhorn_cost_t.float())
                ce = F.softplus(self.sinkhorn_cost_e.float())
                cost = (-torch.log(pi_prior.clamp_min(1e-3)) + ct * dt_xy + ce * de_xy).clamp(0.0, 20.0)
                cost = torch.where(pair_valid, cost, torch.full_like(cost, 40.0))
                cost_shift = cost - cost.amin(dim=(-2, -1), keepdim=True)
                K_mat = torch.exp((-cost_shift / 0.5).clamp_min(-40.0)) * pair_valid.float()

                u_vec = p_s.clone()
                v_vec = q_s.clone()
                for _ in range(3):
                    kv = torch.matmul(K_mat, v_vec.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8)
                    u_vec = p_s / kv
                    ktu = torch.matmul(K_mat.transpose(-2, -1), u_vec.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8)
                    v_vec = q_s / ktu

                gamma = m_dual[:, :, None, None].float() * (
                    u_vec[:, :, :, None] * K_mat * v_vec[:, :, None, :]
                )
                pair_score = torch.where(
                    pair_valid & m_dual[:, :, None, None],
                    gamma * (0.5 * (sum_ex[:, :, :, None] + sum_ey[:, :, None, :])) + 1e-4,
                    torch.full_like(gamma, -1e6),
                ).view(b, -1)
                gamma_flat = gamma.view(b, -1)
                prior_flat = pi_prior.clamp_min(1e-3).log().view(b, -1)
            else:
                # Ablation: raw Cartesian crossing score without Sinkhorn or ToF
                ex_p = ex_f[:, :, :, None].expand(-1, -1, -1, SINKHORN_K_TOP)
                ey_p = ey_f[:, :, None, :].expand(-1, -1, SINKHORN_K_TOP, -1)
                pair_score = torch.where(
                    pair_valid,
                    2.0 * torch.minimum(ex_p, ey_p) + 0.1 * (ex_p + ey_p),
                    torch.full_like(ex_p, -1e6),
                ).view(b, -1)
                gamma_flat = torch.zeros_like(pair_score)
                prior_flat = torch.zeros_like(pair_score)

            best_scores, best_flat = torch.topk(pair_score, k=SINKHORN_N_NEXUS, dim=-1)
            nex_mask = best_scores > -1e5
            safe_flat = torch.where(nex_mask, best_flat, torch.zeros_like(best_flat))
            k2 = SINKHORN_K_TOP * SINKHORN_K_TOP
            d_idx = safe_flat // k2
            rem = safe_flat % k2
            xi_slot = rem // SINKHORN_K_TOP
            yi_slot = rem % SINKHORN_K_TOP

            bx_id = torch.gather(top_ix.view(b, -1), 1, d_idx * SINKHORN_K_TOP + xi_slot)
            by_id = torch.gather(top_iy.view(b, -1), 1, d_idx * SINKHORN_K_TOP + yi_slot)
            ex_val = torch.gather(ex_f.view(b, -1), 1, d_idx * SINKHORN_K_TOP + xi_slot)
            ey_val = torch.gather(ey_f.view(b, -1), 1, d_idx * SINKHORN_K_TOP + yi_slot)
            tx_val = torch.gather(tx_f.view(b, -1), 1, d_idx * SINKHORN_K_TOP + xi_slot)
            ty_val = torch.gather(ty_f.view(b, -1), 1, d_idx * SINKHORN_K_TOP + yi_slot)
            gam_val = torch.gather(gamma_flat, 1, safe_flat)
            pri_val = torch.gather(prior_flat, 1, safe_flat)

            st_idx_nex = self.doublet_stations[d_idx]
            nex_x = self.bar_x_mm[bx_id]
            nex_y = self.bar_y_mm[by_id]
            nex_z = self.station_z_mm[st_idx_nex]
            nex_x0 = self.station_x0[st_idx_nex]
            mean_tof = 0.5 * (tx_val + ty_val) if self.use_sinkhorn_tof else torch.zeros_like(tx_val)
            abs_dt = torch.abs(tx_val - ty_val) if self.use_sinkhorn_tof else torch.zeros_like(tx_val)

            nex_feat = torch.stack([
                (nex_x - self.x_scint_c) / 200.0,
                (nex_y - self.y_c) / 200.0,
                (nex_z - self.z_mid) / 1000.0,
                ex_val,
                ey_val,
                gam_val * 5.0,
                mean_tof,
                abs_dt,
                pri_val / 5.0,
            ], dim=-1)

        nex_tok = self.nexus_in(nex_feat) * nex_mask.unsqueeze(-1).float()
        nex_xyz = torch.stack([nex_x, nex_y, nex_z], dim=-1).float()

        cal_tok = torch.cat([st_tok, nex_tok], dim=1)
        cal_xyz = torch.cat([st_xyz, nex_xyz], dim=1)
        cal_mask = torch.cat([st_mask, nex_mask], dim=1)
        cal_z = cal_xyz[..., 2]
        cal_x0 = torch.cat([self.station_x0[None, :].expand(b, -1), nex_x0], dim=1)

        pair_geom = torch.stack([
            torch.abs(cal_z[:, :, None] - cal_z[:, None, :]) / 500.0,
            torch.abs(cal_x0[:, :, None] - cal_x0[:, None, :]) / 20.0,
        ], dim=-1)
        pair_bias = self.cal_pair_mlp(pair_geom).permute(0, 3, 1, 2)
        for attn in self.bipartite_attn:
            cal_tok = attn(cal_tok, cal_tok, cal_mask, pair_bias=pair_bias)
        cal_tok = cal_tok * cal_mask.unsqueeze(-1).to(cal_tok.dtype)

        return {
            'cal_tokens': cal_tok,
            'cal_xyz_mm': cal_xyz,
            'cal_mask': cal_mask,
            'cal_global': cal_glob,
            'st_tok': st_tok,
            'st_mask': st_mask,
            'n_dual_active': (x_st_act[:, self.doublet_stations] & y_st_act).sum(dim=-1),
        }

    def forward_NextGen(self, batch: dict[str, torch.Tensor]) -> NextGenModelOutput:
        b = int(batch['c_obs'].shape[0])
        device = batch['c_obs'].device

        pix_xyz = batch['pix_xyz_mm'].float()
        pix_feat = batch['pix_feat'].float()
        pix_plane = batch['pix_plane'].long()
        pix_mask = batch['pix_mask'].bool()

        trk_cand = batch['trk_cand'].float()
        trk_cand_mask = batch['trk_cand_mask'].bool()
        trk_hits = batch['trk_hits'].float()
        trk_hit_mask = batch['trk_hit_mask'].bool()
        if 'trk_station_counts' in batch:
            station_counts = batch['trk_station_counts'].float()
        else:
            st_ids = torch.round(trk_hits[..., 0] * 2.0).long()
            station_counts = torch.stack([
                ((st_ids == s) & trk_hit_mask).sum(dim=1).float() for s in (0, 1, 2)
            ], dim=-1)

        # Training-only stochastic PixDark & tracker noise/dropout curriculum
        aug_ev = torch.zeros((b,), dtype=torch.bool, device=device)
        if self.training and self.enable_pixdark_noise_aug and self.aug_prob > 0.0:
            (
                aug_ev, pix_xyz, pix_feat, pix_plane, pix_mask,
                trk_cand, trk_cand_mask, trk_hits, trk_hit_mask, station_counts,
            ) = self._apply_training_degradation_aug(
                pix_xyz, pix_feat, pix_plane, pix_mask,
                trk_cand, trk_cand_mask, trk_hits, trk_hit_mask, station_counts,
            )

        # Shared micro-cluster + micro-patch embeddings
        u_pix = self.pix_in(pix_feat) * pix_mask.unsqueeze(-1).to(pix_feat.dtype)
        if 'pix_patches' in batch and len(batch['pix_patches']) > 0:
            patch_emb = self.patch_proj(self.patch_conv(batch['pix_patches'].float()))
            delta_patch = torch.zeros_like(u_pix, dtype=patch_emb.dtype)
            delta_patch.index_put_((batch['patch_ev'], batch['patch_cl']), patch_emb, accumulate=True)
            u_pix = u_pix + delta_patch * pix_mask.unsqueeze(-1).to(u_pix.dtype)

        pix_plane_tok = self.pix_plane_emb(batch['pix_plane_summary'].float())
        pix_plane_act = batch['pix_plane_summary'][..., 0] > 0.5

        # Soft non-exclusive tracklets (pure-GPU path when precomputed; vectorized dropout on aug_ev)
        if 'trk_kin' in batch and 'trk_morph' in batch and 'trk_mask' in batch:
            trk_kin = batch['trk_kin'].float()
            trk_morph = batch['trk_morph'].float()
            trk_mask = batch['trk_mask'].bool()
            if aug_ev.any():
                drop_trk = aug_ev.unsqueeze(1) & trk_mask & (torch.rand(trk_mask.shape, device=device) < 0.03)
                orig_t = trk_mask.sum(dim=1, keepdim=True)
                empty_t = (trk_mask & (~drop_trk)).sum(dim=1, keepdim=True) == 0
                trk_mask = trk_mask & (~(drop_trk & (~(empty_t & (orig_t > 0)))))
        else:
            trk_dict = extract_soft_tracklets_batch(
                pix_xyz, pix_plane, pix_mask, self.pix_plane_x0, pix_feat=pix_feat, max_tracklets=M_TRK_MAX
            )
            trk_kin = trk_dict['trk_kin']
            trk_morph = trk_dict['trk_morph']
            trk_mask = trk_dict['trk_mask']

        # Calibrated calorimeter anchor prior
        cal_start = batch['cal_start'].float()
        cal_valid = (cal_start[:, 3] > 0.5)
        cal_xyz_prior = torch.stack([cal_start[:, 1], cal_start[:, 2], cal_start[:, 0]], dim=-1)
        default_c = self.default_center_mm[None, :].expand(b, -1)
        v_cal_prior = torch.where(cal_valid[:, None], cal_xyz_prior, default_c)
        # Center z around v_cal_prior[:, 2] before FP32 3x3 Student-t normal-equation solve
        # to eliminate 3000 mm catastrophic cancellation in `x0 - tx * z0`!
        z_ref_pv = v_cal_prior[:, 2:3]
        trk_kin_c = trk_kin.clone()
        trk_kin_c[..., 2] = trk_kin_c[..., 2] - z_ref_pv
        v_cal_prior_c = torch.cat([v_cal_prior[:, :2], torch.zeros_like(z_ref_pv)], dim=-1)

        # Normalize tracklet kinematics for neural encoding
        trk_kin_norm = torch.stack([
            (trk_kin[..., 0] - self.x_pix_c) / 150.0,
            (trk_kin[..., 1] - self.y_c) / 150.0,
            (trk_kin[..., 2] - self.z_mid) / 1000.0,
            trk_kin[..., 3] * 10.0,
            trk_kin[..., 4] * 10.0,
            torch.log1p(trk_kin[..., 5]) / 2.0,
            torch.log1p(trk_kin[..., 6]) / 2.0,
            torch.log1p(trk_kin[..., 7] * 1e4) / 5.0,
            torch.log1p(trk_kin[..., 8] * 1e4) / 5.0,
            torch.log1p(trk_kin[..., 9]) / 4.0,
            trk_kin[..., 10] / 20.0,
            trk_kin[..., 11] / 200.0,
        ], dim=-1)
        trk_emb_all = self.trk_in(torch.cat([trk_kin_norm, trk_morph], dim=-1)) * trk_mask.unsqueeze(-1).float()

        # ==================================================================
        # Regime 2: Mid Tomographic Sinkhorn + ToF Calorimeter
        # ==================================================================
        cal_out = self._encode_mid_sinkhorn_tomography(batch, pix_xyz, pix_plane, pix_mask)
        cal_tok = cal_out['cal_tokens']
        cal_mask = cal_out['cal_mask']

        # ==================================================================
        # Regime 1 & Regime 2 (Target Silicon: Front 0..5 vs Mid 6..11)
        # ==================================================================
        # Symmetric token ranking score for bounded selection
        tok_rank = (
            pix_feat[..., 3]
            + 0.25 * pix_feat[..., 13]
            + 1e-4 * (1.0 - torch.abs(pix_feat[..., 0]).clamp_max(1.0))
        )

        if not self.homogeneous_regime:
            # --- Regime 1: Front 30mm Micro-Tracker (Planes 0..5) ---
            front_raw_m = pix_mask & (pix_plane <= 5)
            f_idx, f_mask = _select_masked_tokens(tok_rank, front_raw_m, K_FRONT_MAX)
            f_xyz = torch.gather(pix_xyz, 1, f_idx.unsqueeze(-1).expand(-1, -1, 3))
            f_feat = torch.gather(pix_feat, 1, f_idx.unsqueeze(-1).expand(-1, -1, PIX_FEAT_DIM))
            f_pl = torch.gather(pix_plane, 1, f_idx)
            f_x0 = self.pix_plane_x0[f_pl]
            z_front = torch.gather(u_pix, 1, f_idx.unsqueeze(-1).expand(-1, -1, self.dim))

            g_persist = self._compute_front_persistence_gate(f_xyz, f_feat, f_pl, f_mask)
            z_front = z_front * (0.25 + 0.75 * g_persist).unsqueeze(-1)

            with torch.amp.autocast(device_type=device.type, enabled=False):
                dp_f = torch.abs(f_pl[:, :, None] - f_pl[:, None, :])
                dz_f = torch.abs(f_xyz[:, :, None, 2] - f_xyz[:, None, :, 2])
                dx0_f = torch.abs(f_x0[:, :, None] - f_x0[:, None, :])
                dx_f = f_xyz[:, :, None, 0] - f_xyz[:, None, :, 0]
                dy_f = f_xyz[:, :, None, 1] - f_xyz[:, None, :, 1]
                d_perp_f = torch.sqrt(dx_f.square() + dy_f.square() + 1e-10)
                tan_f = d_perp_f / dz_f.clamp_min(1.0)
                pair_f = torch.stack([
                    dz_f / 30.0,
                    dx0_f / 3.0,
                    torch.log1p(d_perp_f / 0.1) / 3.0,
                    torch.log1p(tan_f / 0.05) / 3.0,
                    (dp_f == 0).float(),
                    ((dp_f == 1) | (dp_f == 2)).float(),
                ], dim=-1)
                f_bias = self.front_pair_mlp(pair_f).permute(0, 3, 1, 2)
                if self.enable_pixdark_noise_aug:
                    f_bias = f_bias + 0.5 * torch.log(g_persist[:, None, None, :].clamp_min(1e-3))

            z_front = self.front_encoder(z_front, f_mask, pair_bias=f_bias)
            has_front = f_mask.any(dim=1)
            f_safe_m = f_mask.clone()
            if (~has_front).any():
                f_safe_m[~has_front, 0] = True
            fw_log = self.front_pool_gate(z_front).squeeze(-1).float().masked_fill(~f_safe_m, float('-inf'))
            fw = torch.softmax(fw_log, dim=-1) * f_mask.float()
            front_glob = (fw.unsqueeze(-1) * z_front.float()).sum(dim=1)

            # 2 Boundary Bridge tokens (Front <-> Mid)
            bridge_q = self.front_bridge_queries.expand(b, -1, -1)
            z_bridge = self.front_bridge_attn(bridge_q, z_front, f_mask, query_mask=torch.ones((b, 2), dtype=torch.bool, device=device))
            bridge_mask = has_front.unsqueeze(1).expand(b, 2)

            # Front Student-t PV & precision (solved in z-centered frame, then shifted back)
            is_front_trk = trk_mask & (trk_morph[..., 1] < 0.5) & (trk_kin[..., 2] < -2750.0)
            tf_emb = self.trk_front_cross(trk_emb_all, z_front, f_mask, query_mask=is_front_trk)
            fv_out = self.front_vtx_layer(trk_kin_c, tf_emb, is_front_trk, v_cal_prior_c, cal_valid=cal_valid)
            ref_f = self.front_vtx_refine(front_glob)
            v_front_base = torch.cat([fv_out['v_pv_mm'][:, :2], fv_out['v_pv_mm'][:, 2:3] + z_ref_pv], dim=-1)
            v_front = v_front_base + 5.0 * torch.tanh(ref_f[:, :3]) * has_front[:, None].float()
            has_front_trk = is_front_trk.any(dim=1, keepdim=True).float()
            # Explicit zero precision when no front hits exist (per Section 6.3.3)
            p_front_base = torch.where(
                has_front_trk > 0.5,
                1.0 / fv_out['scale_pv_mm'].square().clamp_min(2.5e-3),
                torch.exp(ref_f[:, 3:6].clamp(-6.0, 1.0)) * has_front[:, None].float(),
            )
            p_front = p_front_base * has_front[:, None].float()

            # --- Regime 2 Mid Silicon Slices (Planes 6..11) ---
            mid_raw_m = pix_mask & (pix_plane >= 6) & (pix_plane <= 11)
            m_idx, m_mask = _select_masked_tokens(tok_rank, mid_raw_m, K_MID_MAX)
            z_mid = torch.gather(u_pix, 1, m_idx.unsqueeze(-1).expand(-1, -1, self.dim))
            z_mid = self.mid_pix_cross(z_mid, cal_tok, cal_mask, query_mask=m_mask)
            z_mid = self.mid_pix_self(z_mid, z_mid, m_mask) * m_mask.unsqueeze(-1).to(z_mid.dtype)

            has_mid_si = m_mask.any(dim=1)
            m_safe_m = m_mask.clone()
            if (~has_mid_si).any():
                m_safe_m[~has_mid_si, 0] = True
            mw_log = self.mid_pool_gate(z_mid).squeeze(-1).float().masked_fill(~m_safe_m, float('-inf'))
            mw = torch.softmax(mw_log, dim=-1) * m_mask.float()
            mid_si_glob = (mw.unsqueeze(-1) * z_mid.float()).sum(dim=1)

            is_mid_trk = trk_mask & (trk_morph[..., 1] < 0.5) & (trk_kin[..., 2] >= -2750.0)
            mv_trk_out = self.mid_vtx_layer(trk_kin_c, trk_emb_all, is_mid_trk, v_cal_prior_c, cal_valid=cal_valid)
            v_mid_trk_abs = torch.cat([mv_trk_out['v_pv_mm'][:, :2], mv_trk_out['v_pv_mm'][:, 2:3] + z_ref_pv], dim=-1)
            mid_head_in = torch.cat([mid_si_glob, cal_out['cal_global'].float(), cal_start], dim=-1)
            mid_ref = self.mid_vtx_head(mid_head_in)
            v_mid_tomo = v_cal_prior + 150.0 * torch.tanh(mid_ref[:, :3] / 2.0)
            p_mid_tomo = torch.exp(mid_ref[:, 3:6].clamp(-7.0, 2.0))
            p_mid_trk = torch.where(
                is_mid_trk.any(dim=1, keepdim=True),
                0.25 / mv_trk_out['scale_pv_mm'].square().clamp_min(1e-3),
                torch.zeros_like(p_mid_tomo),
            )
            p_mid = p_mid_tomo + p_mid_trk
            v_mid = (p_mid_tomo * v_mid_tomo + p_mid_trk * v_mid_trk_abs) / p_mid.clamp_min(1e-6)

            # Inverse-covariance cross-regime PV fusion (Section 6.3.3)
            p_0 = torch.where(
                cal_valid[:, None],
                torch.exp(self.pv_prior_log_prec.clamp(-9.0, 1.0))[None, :].expand(b, 3),
                torch.full((b, 3), 1e-5, device=device),
            )
            p_tot = (p_front + p_mid + p_0).clamp_min(1e-6)
            v_pv_mm = (p_front * v_front + p_mid * v_mid + p_0 * v_cal_prior) / p_tot
            scale_pv_mm = (torch.rsqrt(p_tot) * torch.exp(self.pv_scale_cal.clamp(-2.0, 3.0))[None, :]).clamp(0.05, 150.0)
            sv_disp_mm = fv_out['sv_disp_mm']
        else:
            # Ablation `abl_homogeneous_regime`: collapse Front30 & Mid150 into single shared encoder
            tgt_raw_m = pix_mask & (pix_plane <= 11)
            t_idx, t_mask = _select_masked_tokens(tok_rank, tgt_raw_m, K_FRONT_MAX + K_MID_MAX)
            t_xyz = torch.gather(pix_xyz, 1, t_idx.unsqueeze(-1).expand(-1, -1, 3))
            t_feat = torch.gather(pix_feat, 1, t_idx.unsqueeze(-1).expand(-1, -1, PIX_FEAT_DIM))
            t_pl = torch.gather(pix_plane, 1, t_idx)
            t_x0 = self.pix_plane_x0[t_pl]
            z_tgt = torch.gather(u_pix, 1, t_idx.unsqueeze(-1).expand(-1, -1, self.dim))

            g_persist = self._compute_front_persistence_gate(t_xyz, t_feat, t_pl, t_mask)
            z_tgt = z_tgt * (0.25 + 0.75 * g_persist).unsqueeze(-1)
            with torch.amp.autocast(device_type=device.type, enabled=False):
                dp_t = torch.abs(t_pl[:, :, None] - t_pl[:, None, :])
                dz_t = torch.abs(t_xyz[:, :, None, 2] - t_xyz[:, None, :, 2])
                dx0_t = torch.abs(t_x0[:, :, None] - t_x0[:, None, :])
                dx_t = t_xyz[:, :, None, 0] - t_xyz[:, None, :, 0]
                dy_t = t_xyz[:, :, None, 1] - t_xyz[:, None, :, 1]
                d_perp_t = torch.sqrt(dx_t.square() + dy_t.square() + 1e-10)
                tan_t = d_perp_t / dz_t.clamp_min(1.0)
                pair_t = torch.stack([
                    dz_t / 60.0,
                    dx0_t / 10.0,
                    torch.log1p(d_perp_t / 0.1) / 3.0,
                    torch.log1p(tan_t / 0.05) / 3.0,
                    (dp_t == 0).float(),
                    ((dp_t == 1) | (dp_t == 2)).float(),
                ], dim=-1)
                t_bias = self.front_pair_mlp(pair_t).permute(0, 3, 1, 2)
                if self.enable_pixdark_noise_aug:
                    t_bias = t_bias + 0.5 * torch.log(g_persist[:, None, None, :].clamp_min(1e-3))

            z_tgt = self.front_encoder(z_tgt, t_mask, pair_bias=t_bias)
            z_tgt = self.mid_pix_cross(z_tgt, cal_tok, cal_mask, query_mask=t_mask)
            z_tgt = self.mid_pix_self(z_tgt, z_tgt, t_mask) * t_mask.unsqueeze(-1).to(z_tgt.dtype)

            has_tgt = t_mask.any(dim=1)
            t_safe_m = t_mask.clone()
            if (~has_tgt).any():
                t_safe_m[~has_tgt, 0] = True
            tw_log = self.front_pool_gate(z_tgt).squeeze(-1).float().masked_fill(~t_safe_m, float('-inf'))
            tw = torch.softmax(tw_log, dim=-1) * t_mask.float()
            front_glob = (tw.unsqueeze(-1) * z_tgt.float()).sum(dim=1)

            is_tgt_trk = trk_mask & (trk_morph[..., 1] < 0.5)
            tv_out = self.front_vtx_layer(trk_kin_c, trk_emb_all, is_tgt_trk, v_cal_prior_c, cal_valid=cal_valid)
            mid_head_in = torch.cat([front_glob, cal_out['cal_global'].float(), cal_start], dim=-1)
            mid_ref = self.mid_vtx_head(mid_head_in)
            v_pv_base = torch.cat([tv_out['v_pv_mm'][:, :2], tv_out['v_pv_mm'][:, 2:3] + z_ref_pv], dim=-1)
            v_pv_mm = v_pv_base + 5.0 * torch.tanh(mid_ref[:, :3])
            scale_pv_mm = tv_out['scale_pv_mm']
            sv_disp_mm = tv_out['sv_disp_mm']
            p_front = 1.0 / scale_pv_mm.square().clamp_min(1e-4)
            p_mid = torch.zeros_like(p_front)
            v_front = v_pv_mm
            v_mid = v_pv_mm

            z_front = z_tgt
            f_mask = t_mask
            f_g_persist = g_persist
            z_mid = z_tgt[:, :0]
            m_mask = t_mask[:, :0]
            z_bridge = z_tgt[:, :0]
            bridge_mask = t_mask[:, :0]
            # keep front_vtx_refine & mid_vtx_layer & mid_pool_gate gradients active in Gate B if called
            _unused_reg = (
                0.0 * self.front_vtx_refine(front_glob).sum()
                + 0.0 * self.mid_pool_gate(front_glob).sum()
                + 0.0 * self.front_bridge_attn(self.front_bridge_queries.expand(b, -1, -1), z_tgt, t_mask).sum()
                + 0.0 * self.mid_vtx_layer(trk_kin, trk_emb_all, is_tgt_trk, v_cal_prior, cal_valid=cal_valid)['v_pv_mm'].sum()
            )
            v_pv_mm = v_pv_mm + _unused_reg
            g_persist = f_g_persist

        # ==================================================================
        # Regime 3 & 4: IPT (Planes 12..14) + Cross-Magnet-0 Spec Interlock
        # ==================================================================
        ipt_raw_m = pix_mask & (pix_plane >= 12)
        i_idx, i_mask = _select_masked_tokens(tok_rank, ipt_raw_m, K_IPT_MAX)
        z_ipt = torch.gather(u_pix, 1, i_idx.unsqueeze(-1).expand(-1, -1, self.dim))
        z_ipt = self.ipt_self(z_ipt, z_ipt, i_mask) * i_mask.unsqueeze(-1).to(z_ipt.dtype)
        has_ipt = i_mask.any(dim=1)
        i_safe_m = i_mask.clone()
        if (~has_ipt).any():
            i_safe_m[~has_ipt, 0] = True
        iw_log = self.ipt_pool_gate(z_ipt).squeeze(-1).float().masked_fill(~i_safe_m, float('-inf'))
        iw = torch.softmax(iw_log, dim=-1) * i_mask.float()
        ipt_glob = (iw.unsqueeze(-1) * z_ipt.float()).sum(dim=1)

        is_ipt_trk = trk_mask & (trk_morph[..., 1] > 0.5)
        ipt_kin = trk_kin
        if self.use_ipt_spec_interlock:
            interlock = compute_ipt_spectrometer_interlock(
                ipt_kin, is_ipt_trk, trk_cand, trk_cand_mask,
                x_pix_center=self.x_pix_c, y_scint_center=self.y_c,
            )  # (B, K_cand, 4)
        else:
            interlock = trk_cand.new_zeros((b, trk_cand.shape[1], 4))

        st_occ_feat = (torch.log1p(station_counts.clamp_min(0.0)) / 3.0)[:, None, :].expand(b, trk_cand.shape[1], 3)
        cand_in = torch.cat([trk_cand, interlock, st_occ_feat], dim=-1)
        hc = self.spec_cand_in(cand_in) * trk_cand_mask.unsqueeze(-1).to(cand_in.dtype)
        hc = self.spec_cand_attn1(hc, hc, trk_cand_mask)
        hc = self.spec_cand_attn2(hc, hc, trk_cand_mask)

        if self.use_ipt_spec_interlock:
            rho_spec = torch.sigmoid(self.spec_rel_head(hc).squeeze(-1)) * trk_cand_mask.float()
        else:
            rho_spec = trk_cand_mask.float()

        z_spec_cands = self.spec_cand_to_dim(torch.cat([hc, interlock], dim=-1)) * trk_cand_mask.unsqueeze(-1).to(hc.dtype)

        # Soft reliability-weighted candidate pool + raw station hit pool
        with torch.amp.autocast(device_type=device.type, enabled=False):
            cm_f = trk_cand_mask.float()
            w_cand = rho_spec.float() if self.use_ipt_spec_interlock else cm_f
            w_cand = w_cand / w_cand.sum(dim=1, keepdim=True).clamp_min(1e-6)
            cand_pool = (w_cand.unsqueeze(-1) * hc.float()).sum(dim=1) * trk_cand_mask.any(dim=1, keepdim=True).float()

            # Soft momentum proxy from weighted candidates
            is_3st = (trk_cand[..., 0] > 0.75) & trk_cand_mask
            sy_signed = SIGMA_SAGITTA_Y_MM * torch.sinh((5.0 * trk_cand[..., 6]).clamp(-9.0, 9.0))
            p_est = (SAGITTA_CONST_MEV_MM / (sy_signed.abs() + 0.015)).clamp(min=100.0, max=5.0e6)
            p_log_w = (w_cand * torch.log1p(p_est / 1000.0) * is_3st.float()).sum(dim=1)
            q_charge_w = (w_cand * torch.sign(sy_signed) * is_3st.float()).sum(dim=1)
            best_ilock = (interlock[..., 0] * cm_f).max(dim=1).values
            max_rho = rho_spec.max(dim=1).values
            mean_rho = (rho_spec.sum(dim=1) / cm_f.sum(dim=1).clamp_min(1.0))

        has_hit = trk_hit_mask.any(dim=1)
        hh = self.spec_hit_in(trk_hits)
        hm = trk_hit_mask.unsqueeze(-1).to(hh.dtype)
        h_mean = (hh * hm).sum(dim=1) / trk_hit_mask.sum(dim=1, keepdim=True).clamp(min=1).to(hh.dtype)
        safe_hm = trk_hit_mask.clone()
        if (~has_hit).any():
            safe_hm[~has_hit, 0] = True
        h_w = torch.softmax(self.spec_hit_score(hh).squeeze(-1).float().masked_fill(~safe_hm, float('-inf')), dim=-1).to(hh.dtype)
        h_attn = (hh * h_w.unsqueeze(-1)).sum(dim=1)
        hit_pool = torch.cat([h_mean, h_attn], dim=-1) * has_hit.unsqueeze(-1).to(hh.dtype)

        spec_summary_vec = torch.stack([p_log_w, q_charge_w, best_ilock, max_rho, mean_rho], dim=-1)
        spec_glob = self.spec_glob_fusion(torch.cat([cand_pool, hit_pool, ipt_glob, spec_summary_vec], dim=-1))

        # ==================================================================
        # Longitudinal 69-Plane Profile + Uncertainty-Gated Cross-Modal Fusion
        # ==================================================================
        z_in = torch.cat([pix_plane_tok, cal_out['st_tok']], dim=1)[:, self.z_order_idx]
        z_act = torch.cat([pix_plane_act, cal_out['st_mask']], dim=1)[:, self.z_order_idx]
        z_in = (z_in + self.z_pos_proj(self.z_plane_geom).unsqueeze(0).to(z_in.dtype)) * z_act.unsqueeze(-1).to(z_in.dtype)
        z_seq = self.z_profile_encoder(z_in, z_act)

        with torch.amp.autocast(device_type=device.type, enabled=False):
            z_safe = z_act.clone()
            no_z = ~z_safe.any(dim=1)
            if no_z.any():
                z_safe[no_z, 0] = True
            z_logits = self.z_pool_gate(z_seq.float()).squeeze(-1).masked_fill(~z_safe, float('-inf'))
            z_w = torch.softmax(z_logits, dim=1) * z_act.float()
            z_prof_glob = (z_w.unsqueeze(-1) * z_seq.float()).sum(dim=1)

        source = torch.cat([
            z_front, z_bridge, z_mid, cal_tok, z_ipt, z_spec_cands, spec_glob.unsqueeze(1)
        ], dim=1)
        source_mask = torch.cat([
            f_mask, bridge_mask, m_mask, cal_mask, i_mask, trk_cand_mask, has_hit.unsqueeze(1)
        ], dim=1)

        cobs_q = self.cobs_query(batch['c_obs'].float()).unsqueeze(1)
        latents = torch.cat([self.latent_queries.expand(b, -1, -1), cobs_q], dim=1)
        latent_mask = torch.ones((b, self.n_queries + 1), dtype=torch.bool, device=device)

        if self.use_ipt_spec_interlock:
            # Token reliability bias in cross-attention logits (Section 6.2.2)
            ones_br = torch.ones((b, z_bridge.shape[1]), device=device)
            ones_m = torch.ones((b, z_mid.shape[1]), device=device)
            ones_c = torch.ones((b, cal_tok.shape[1]), device=device)
            ones_i = torch.ones((b, z_ipt.shape[1]), device=device)
            tok_rho = torch.cat([
                g_persist.clamp_min(1e-3),
                ones_br,
                ones_m,
                ones_c,
                ones_i,
                rho_spec.clamp_min(1e-3),
                max_rho.unsqueeze(1).clamp_min(1e-3),
            ], dim=1)
            ugca_bias = self.ugca_beta.float() * torch.log(tok_rho)[:, None, None, :]
        else:
            ugca_bias = None

        for blk in self.global_blocks:
            latents = blk.cross(latents, source, source_mask, pair_bias=ugca_bias, query_mask=latent_mask)
            latents = blk.self_attn(latents, latents, latent_mask)

        q_lat = latents[:, :self.n_queries]
        q_cobs = latents[:, self.n_queries]
        with torch.amp.autocast(device_type=device.type, enabled=False):
            w_lat = torch.softmax(self.pool_gate(q_lat.float()), dim=1)
            pool_attn = (w_lat * q_lat.float()).sum(dim=1)
            fused_global = self.global_fusion(torch.cat([
                pool_attn,
                front_glob.float(),
                cal_out['cal_global'].float(),
                z_prof_glob,
                spec_glob.float(),
                q_cobs.float(),
            ], dim=-1))

            logits7 = self.event_head_7(fused_global)
            probs4 = marginal_four_from_seven(logits7, is_probs=False)
            charm_logit = self.charm_head(fused_global).squeeze(-1)
            kin_pred = self.kin_head(fused_global)

        return NextGenModelOutput(
            logits7=logits7,
            probs4=probs4,
            charm_logit=charm_logit,
            v_pv_mm=v_pv_mm.float(),
            scale_pv_mm=scale_pv_mm.float(),
            sv_disp_mm=sv_disp_mm.float(),
            kin_pred=kin_pred,
            aux={
                'rho_spec': rho_spec.detach(),
                'v_front_mm': v_front,
                'P_front': p_front,
                'v_mid_mm': v_mid,
                'P_mid': p_mid,
                'g_persist_mean': g_persist.mean(dim=-1).detach(),
                'n_dual_active': cal_out['n_dual_active'].detach(),
            },
        )


def dual_regime_branch_groups(model: DualRegimeModalNet) -> dict[str, list[tuple[str, nn.Parameter]]]:
    """Group DualRegimeModalNet parameters into physical subsystem branches for Gate-A/B audits."""
    groups: dict[str, list[tuple[str, nn.Parameter]]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(('patch_', 'pix_in.', 'pix_plane_emb.')):
            br = 'pixel_frontend'
        elif name.startswith(('pixdark_', 'front_', 'trk_')):
            br = 'regime1_front_micro'
        elif name.startswith(('xbar_enc.', 'ybar_enc.', 'stereo_2d_', 'station_fuse.', 'sinkhorn_', 'nexus_in.', 'cal_pair_', 'bipartite_', 'mid_', 'pv_')):
            br = 'regime2_mid_sinkhorn'
        elif name.startswith(('ipt_', 'spec_')):
            br = 'regime34_ipt_spec'
        elif name.startswith(('z_', 'latent_queries', 'cobs_query.', 'ugca_', 'global_')):
            br = 'precision_global_fusion'
        else:
            br = 'output_heads'
        groups.setdefault(br, []).append((name, param))
    return groups
