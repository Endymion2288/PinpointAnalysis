"""Proposal B: TrackletIPFormer (~5.85M params) and causal physics ablation switches.

Implements:
  - Non-exclusive shared-root soft 3D micro-tracklet tokenization (`extract_soft_tracklets_single`)
  - Tracklet forward X0 calorimeter cone continuation (`TrackletCalorimeterConeProjector`)
  - Vertex-free translation-invariant 3D skew-DCA relational attention bias (`pairwise_skew_dca_features`)
  - Differentiable 4-step Student-t (nu=3) Billoir/Kalman PV + inverse-covariance `cal_start` fusion
    with outlier-gated secondary vertex fit & covariance-normalized IP re-injection
  - Relativistic scintillator ToF residual (`compute_scint_tof_residual_ns`)
  - Cross-Magnet-0 IPT-Spectrometer 4D interlock (`compute_ipt_spectrometer_interlock`)
  - Strict pure-observation inference (`NextGenFlavourModel` / `NextGenModelOutput`)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from analysis.flavour.v3.common.domains import cumulative_radiation_length, detector_bounds
from analysis.flavour.v3.common.layers import ScaledQKNormAttention, split_adamw_param_groups
from analysis.flavour.v3.v1_b1pp.models import TransverseLayerEncoder
from analysis.flavour.v3.v2_pinpointformer.attention import (
    AttentionStack,
    PerceiverCrossBlock,
    collect_attention_scales,
    collect_effective_attention_sharpness,
)
from analysis.flavour.v3.v2_pinpointformer.pixel_tokenizer import (
    MicroPatchConv2D,
    PIX_FEAT_DIM,
    PIX_PLANE_SUMMARY_DIM,
    collate_pf,
    extract_pf_item,
)
from analysis.flavour.v3.v2_pinpointformer.stereo_bars import (
    MAX_CROSS_PER_SIDE,
    N_MAX_NEXUS,
    static_scint_stations,
)
from analysis.flavour.v3.v2_pinpointformer.vertex_anchor import (
    VERTEX_AXIS_SCALES_MM,
    select_near_vertex_clusters,
)
from analysis.flavour.v3.v3_multitask.losses import (
    focal_charm_bce,
    marginal_tau_branch_ce,
)
from analysis.flavour.nextgen.protocols import (
    NextGenFlavourModel,
    NextGenModelOutput,
    marginal_four_from_seven,
    student_t_vertex_loss,
)
from analysis.flavour.nextgen.tracklet_ops import (
    DifferentiableStudentTVertexLayer,
    PAIR_TRK_FEAT_DIM,
    TRK_KIN_DIM,
    TRK_MORPH_DIM,
    _safe_hypot,
    compute_ipt_spectrometer_interlock,
    compute_scint_tof_residual_ns,
    extract_soft_tracklets_batch,
    extract_soft_tracklets_single,
    pairwise_skew_dca_features,
)


MAX_TRACKLETS = 64
CONE_FEAT_DIM = 6
MICRO_VTX_LEGACY_DIM = 16


def extract_tracklet_pf_item(
    index: int,
    observation: dict,
    label: int | None,
    dom: dict,
    geometry: dict,
    bounds: dict,
    config: dict,
) -> dict:
    """Extract standard PF clusters/patches/bars/spectrometer PLUS non-exclusive soft tracklets & bar ToF.

    Strictly pure-observation: never reads `targets.parquet`, `trackID`, or truth sidecars.
    """
    item = extract_pf_item(index, observation, label, dom, geometry, bounds, config)
    bar = observation['bar']
    keep_bar = item['bar_keep_mask']
    if 'hit_time_ns' in bar and len(bar['hit_time_ns']) == len(keep_bar):
        ht = np.asarray(bar['hit_time_ns'], dtype=np.float32)[keep_bar]
    else:
        ht = np.full(len(item['bar_panel']), -1.0, dtype=np.float32)
    item['bar_hit_time_ns'] = ht

    # Pre-extract non-exclusive soft tracklets on exact (possibly 1.0 mm quantized) pix_xyz_mm
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

    trk = extract_soft_tracklets_single(
        item['pix_xyz_mm'],
        item['pix_plane'],
        item['pix_feat'],
        plane_x0,
        max_tracklets=int((config or {}).get('max_tracklets', MAX_TRACKLETS)),
        pitch_xy_mm=pitch_xy,
    )
    item['trk_kin'] = trk['trk_kin']
    item['trk_morph'] = trk['trk_morph']
    item['trk_mask'] = trk['trk_mask']
    return item


def collate_tracklet_pf(
    events: list[dict],
    mirror_rng: np.random.Generator | None = None,
    y_scint_center: float = -12.0,
    n_bars: int = 40,
) -> dict:
    """Collate variable-length TrackletIPFormer events into batched tensors."""
    batch = collate_pf(events, mirror_rng=None, y_scint_center=y_scint_center, n_bars=n_bars)
    if 'trk_kin' in events[0]:
        batch['trk_kin'] = torch.from_numpy(np.stack([e['trk_kin'] for e in events]).astype(np.float32))
        batch['trk_morph'] = torch.from_numpy(np.stack([e['trk_morph'] for e in events]).astype(np.float32))
        batch['trk_mask'] = torch.from_numpy(np.stack([e['trk_mask'] for e in events]).astype(bool))
    ht_list = [
        e['bar_hit_time_ns'].astype(np.float32)
        for e in events
        if 'bar_hit_time_ns' in e and len(e['bar_hit_time_ns']) > 0
    ]
    if ht_list:
        batch['bar_hit_time_ns'] = torch.from_numpy(np.concatenate(ht_list, axis=0))
    else:
        batch['bar_hit_time_ns'] = torch.full((len(batch['bar_ev']),), -1.0, dtype=torch.float32)
    return batch


class NextGenStereoBarNexus(nn.Module):
    """Stereo X/Y bar convs + 3D crossings + relativistic ToF residual channel."""

    def __init__(self, stations: dict, dim: int = 176, heads: int = 8, drop_path: float = 0.10) -> None:
        super().__init__()
        self.dim = dim
        self.n_st = int(stations['n_stations'])
        self.n_dbl = int(stations['n_doublets'])
        self.n_bars = int(stations['n_bars'])
        self.x_c = float(stations['x_scint_center'])
        self.y_c = float(stations['y_scint_center'])
        self.z_mid = float(stations['z_mid'])

        self.register_buffer('station_z_mm', torch.from_numpy(stations['station_z_mm']))
        self.register_buffer('station_x0', torch.from_numpy(stations['station_x0']))
        self.register_buffer('station_has_y', torch.from_numpy(stations['station_has_y']))
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
        self.register_buffer(
            'ybar_geom',
            torch.stack([z_norm[self.doublet_stations], x0_norm[self.doublet_stations]], dim=0),
        )

        # 7 channels: [E, PE, z_norm, x0_norm, active, active*u, active*tof_resid]
        self.xbar_enc = TransverseLayerEncoder(7, (32, 48, 64), stem_k=3, stem_s=1, out_dim=dim)
        self.ybar_enc = TransverseLayerEncoder(7, (32, 48, 64), stem_k=3, stem_s=1, out_dim=dim)
        self.stereo_2d_proj = nn.Sequential(
            nn.Linear(256, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
        )
        self.station_fuse = nn.Sequential(
            nn.Linear(2 * dim + 6, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
        )
        self.nexus_in = nn.Sequential(
            nn.Linear(9, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
        )
        self.pair_mlp = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, heads))
        self.bipartite_attn = nn.ModuleList([
            ScaledQKNormAttention(dim, heads=heads, init_scale=12.0, drop_path=drop_path)
            for _ in range(2)
        ])

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        b = int(batch['c_obs'].shape[0])
        device = batch['c_obs'].device
        bar_ev = batch['bar_ev']
        bar_panel = batch['bar_panel']
        bar_horiz = batch['bar_horiz']
        bar_idx = batch['bar_idx']
        bar_val = batch['bar_val']

        grid_x = torch.zeros((b, 7, self.n_st, self.n_bars), device=device, dtype=torch.float32)
        grid_y = torch.zeros((b, 7, self.n_dbl, self.n_bars), device=device, dtype=torch.float32)

        if len(bar_ev) > 0:
            bar_st = self.panel_to_station[bar_panel]
            bar_z = self.station_z_mm[bar_st]
            if 'bar_hit_time_ns' in batch and len(batch['bar_hit_time_ns']) == len(bar_ev):
                ht = batch['bar_hit_time_ns']
                bar_tof = compute_scint_tof_residual_ns(ht, bar_z, event_idx=bar_ev)
            else:
                bar_tof = torch.zeros(len(bar_ev), device=device, dtype=torch.float32)
        else:
            bar_tof = torch.empty(0, device=device, dtype=torch.float32)

        x_mask_b = bar_horiz == 0
        y_mask_b = bar_horiz == 1
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

        x_act = grid_x[:, 4].any(dim=-1)
        y_act = grid_y[:, 4].any(dim=-1)
        x_tok, x_glob = self.xbar_enc(grid_x, x_act)
        y_tok, y_glob = self.ybar_enc(grid_y, y_act)
        cal_glob = self.stereo_2d_proj(torch.cat([x_glob, y_glob], dim=-1))

        y_full = torch.zeros_like(x_tok)
        y_act_full = torch.zeros_like(x_act)
        y_ex = grid_y[:, 0]
        y_has = grid_y[:, 4]
        y_tof = grid_y[:, 6]
        y_ex_full = torch.zeros((b, self.n_st, self.n_bars), device=device)
        y_has_full = torch.zeros((b, self.n_st, self.n_bars), device=device)
        y_full[:, self.doublet_stations] = y_tok
        y_act_full[:, self.doublet_stations] = y_act
        y_ex_full[:, self.doublet_stations] = y_ex
        y_has_full[:, self.doublet_stations] = y_has

        e_x = grid_x[:, 0].sum(dim=-1)
        e_y = y_ex_full.sum(dim=-1)
        w_x = grid_x[:, 0] + 1e-4 * grid_x[:, 4]
        w_y = y_ex_full + 1e-4 * y_has_full
        x_cent = (w_x * self.bar_x_mm[None, None, :]).sum(dim=-1) / w_x.sum(dim=-1).clamp_min(1e-6)
        y_cent = (w_y * self.bar_y_mm[None, None, :]).sum(dim=-1) / w_y.sum(dim=-1).clamp_min(1e-6)
        x_cent = torch.where(x_act, x_cent, torch.full_like(x_cent, self.x_c))
        y_cent = torch.where(y_act_full, y_cent, torch.full_like(y_cent, self.y_c))

        # Station energy-weighted mean ToF
        tof_x_st = (w_x * grid_x[:, 6]).sum(dim=-1) / w_x.sum(dim=-1).clamp_min(1e-6)
        tof_x_st = torch.where(x_act, tof_x_st, torch.zeros_like(tof_x_st))

        asym = (e_x - e_y) / (e_x + e_y + 1e-4) * (x_act & y_act_full).float()
        st_aux = torch.stack([e_x, e_y, asym, x_act.float(), y_act_full.float(), tof_x_st], dim=-1)
        st_tok = self.station_fuse(torch.cat([x_tok, y_full, st_aux], dim=-1))
        st_mask = x_act | y_act_full
        st_tok = st_tok * st_mask.unsqueeze(-1).to(st_tok.dtype)
        st_xyz = torch.stack([x_cent, y_cent, self.station_z_mm[None, :].expand(b, -1)], dim=-1).float()
        st_e = torch.maximum(e_x, e_y)

        # Top-N_MAX_NEXUS 3D stereo crossings
        ex_dbl = grid_x[:, 0, self.doublet_stations]
        act_x_dbl = grid_x[:, 4, self.doublet_stations] > 0.5
        tof_x_dbl = grid_x[:, 6, self.doublet_stations]
        ey_dbl = y_ex
        act_y_dbl = y_has > 0.5
        sym_tie = 1e-5 * (1.0 - self.u_bars.abs())[None, None, :]
        score_x = torch.where(act_x_dbl, ex_dbl + 1e-3 + sym_tie, torch.full_like(ex_dbl, -1e6))
        score_y = torch.where(act_y_dbl, ey_dbl + 1e-3 + sym_tie, torch.full_like(ey_dbl, -1e6))
        top_ex, top_ix = torch.topk(score_x, k=MAX_CROSS_PER_SIDE, dim=-1)
        top_ey, top_iy = torch.topk(score_y, k=MAX_CROSS_PER_SIDE, dim=-1)
        top_vx = torch.gather(act_x_dbl, dim=-1, index=top_ix)
        top_vy = torch.gather(act_y_dbl, dim=-1, index=top_iy)
        top_ex_clean = torch.gather(ex_dbl, dim=-1, index=top_ix)
        top_ey_clean = torch.gather(ey_dbl, dim=-1, index=top_iy)
        top_tx_clean = torch.gather(tof_x_dbl, dim=-1, index=top_ix)
        top_ty_clean = torch.gather(y_tof, dim=-1, index=top_iy)

        pair_valid = top_vx[:, :, :, None] & top_vy[:, :, None, :]
        ex_p = top_ex_clean[:, :, :, None].expand(-1, -1, -1, MAX_CROSS_PER_SIDE)
        ey_p = top_ey_clean[:, :, None, :].expand(-1, -1, MAX_CROSS_PER_SIDE, -1)
        pair_score = torch.where(
            pair_valid,
            2.0 * torch.minimum(ex_p, ey_p) + 0.1 * (ex_p + ey_p),
            torch.full_like(ex_p, -1e6),
        ).view(b, -1)

        best_scores, best_flat = torch.topk(pair_score, k=N_MAX_NEXUS, dim=-1)
        nex_mask = best_scores > -1e5
        safe_flat = torch.where(nex_mask, best_flat, torch.zeros_like(best_flat))
        d_idx = safe_flat // (MAX_CROSS_PER_SIDE * MAX_CROSS_PER_SIDE)
        rem = safe_flat % (MAX_CROSS_PER_SIDE * MAX_CROSS_PER_SIDE)
        xi_slot = rem // MAX_CROSS_PER_SIDE
        yi_slot = rem % MAX_CROSS_PER_SIDE

        bx_id = torch.gather(top_ix.view(b, -1), 1, d_idx * MAX_CROSS_PER_SIDE + xi_slot)
        by_id = torch.gather(top_iy.view(b, -1), 1, d_idx * MAX_CROSS_PER_SIDE + yi_slot)
        ex_val = torch.gather(top_ex_clean.view(b, -1), 1, d_idx * MAX_CROSS_PER_SIDE + xi_slot)
        ey_val = torch.gather(top_ey_clean.view(b, -1), 1, d_idx * MAX_CROSS_PER_SIDE + yi_slot)
        tx_val = torch.gather(top_tx_clean.view(b, -1), 1, d_idx * MAX_CROSS_PER_SIDE + xi_slot)
        ty_val = torch.gather(top_ty_clean.view(b, -1), 1, d_idx * MAX_CROSS_PER_SIDE + yi_slot)

        st_idx_nex = self.doublet_stations[d_idx]
        nex_x = self.bar_x_mm[bx_id]
        nex_y = self.bar_y_mm[by_id]
        nex_z = self.station_z_mm[st_idx_nex]
        nex_x0 = self.station_x0[st_idx_nex]

        nex_feat = torch.stack([
            (nex_x - self.x_c) / 200.0,
            (nex_y - self.y_c) / 200.0,
            (nex_z - self.z_mid) / 1000.0,
            ex_val,
            ey_val,
            torch.abs(ex_val - ey_val),
            (2.0 * torch.minimum(ex_val, ey_val)) / (ex_val + ey_val + 1e-4),
            tx_val,
            ty_val,
        ], dim=-1)
        nex_tok = self.nexus_in(nex_feat) * nex_mask.unsqueeze(-1).float()
        nex_xyz = torch.stack([nex_x, nex_y, nex_z], dim=-1).float()
        nex_e = 0.5 * (ex_val + ey_val)
        nex_tof = 0.5 * (tx_val + ty_val)

        cal_tok = torch.cat([st_tok, nex_tok], dim=1)
        cal_xyz = torch.cat([st_xyz, nex_xyz], dim=1)
        cal_mask = torch.cat([st_mask, nex_mask], dim=1)
        cal_e = torch.cat([st_e, nex_e], dim=1)
        cal_tof = torch.cat([tof_x_st, nex_tof], dim=1)
        cal_z = cal_xyz[..., 2]
        cal_x0 = torch.cat([self.station_x0[None, :].expand(b, -1), nex_x0], dim=1)

        pair_geom = torch.stack([
            torch.abs(cal_z[:, :, None] - cal_z[:, None, :]) / 500.0,
            torch.abs(cal_x0[:, :, None] - cal_x0[:, None, :]) / 20.0,
        ], dim=-1)
        pair_bias = self.pair_mlp(pair_geom).permute(0, 3, 1, 2)
        for attn in self.bipartite_attn:
            cal_tok = attn(cal_tok, cal_tok, cal_mask, pair_bias=pair_bias)
        cal_tok = cal_tok * cal_mask.unsqueeze(-1).to(cal_tok.dtype)

        return {
            'cal_tokens': cal_tok,
            'cal_xyz_mm': cal_xyz,
            'cal_x0': cal_x0,
            'cal_e': cal_e,
            'cal_tof': cal_tof,
            'cal_mask': cal_mask,
            'cal_global': cal_glob,
            'nex_xyz_mm': nex_xyz,
            'nex_x0': nex_x0,
            'nex_e': nex_e,
            'nex_tof': nex_tof,
            'nex_mask': nex_mask,
        }


class NextGenSpectrometerEncoder(nn.Module):
    """Bore-verified spectrometer encoder augmented with cross-Magnet-0 IPT 4D interlock."""

    def __init__(self, out_dim: int = 176, x_pix_center: float = -0.0048, y_scint_center: float = -12.0) -> None:
        super().__init__()
        self.x_pix_center = float(x_pix_center)
        self.y_scint_center = float(y_scint_center)
        self.cand_in = nn.Sequential(
            nn.Linear(18, 96),
            nn.LayerNorm(96),
            nn.SiLU(),
            nn.Linear(96, 96),
            nn.SiLU(),
        )
        self.cand_attn1 = ScaledQKNormAttention(96, heads=4, init_scale=8.0, drop_path=0.05)
        self.cand_attn2 = ScaledQKNormAttention(96, heads=4, init_scale=8.0, drop_path=0.05)
        self.cand_to_dim = nn.Sequential(
            nn.Linear(96, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.hit_in = nn.Sequential(
            nn.Linear(6, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.hit_score = nn.Linear(64, 1)
        self.fusion = nn.Sequential(
            nn.Linear(192 + 128, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        trk_cand: torch.Tensor,
        trk_cand_mask: torch.Tensor,
        trk_hits: torch.Tensor,
        trk_hit_mask: torch.Tensor,
        ipt_kin: torch.Tensor,
        ipt_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ilock = compute_ipt_spectrometer_interlock(
            ipt_kin,
            ipt_mask,
            trk_cand,
            trk_cand_mask,
            x_pix_center=self.x_pix_center,
            y_scint_center=self.y_scint_center,
        )
        cand_in = torch.cat([trk_cand.float(), ilock], dim=-1)
        has_cand = trk_cand_mask.any(dim=1)
        hc = self.cand_in(cand_in)
        hc = self.cand_attn1(hc, hc, trk_cand_mask)
        hc = self.cand_attn2(hc, hc, trk_cand_mask)
        cand_tok = self.cand_to_dim(hc) * trk_cand_mask.unsqueeze(-1).to(hc.dtype)

        cm = trk_cand_mask.unsqueeze(-1).to(hc.dtype)
        c_mean = (hc * cm).sum(dim=1) / trk_cand_mask.sum(dim=1, keepdim=True).clamp(min=1).to(hc.dtype)
        c_max = hc.masked_fill(~trk_cand_mask.unsqueeze(-1), -1e4).amax(dim=1)
        cand_pool = torch.cat([c_mean, c_max], dim=-1) * has_cand.unsqueeze(-1).to(hc.dtype)

        has_hit = trk_hit_mask.any(dim=1)
        hh = self.hit_in(trk_hits)
        hm = trk_hit_mask.unsqueeze(-1).to(hh.dtype)
        h_mean = (hh * hm).sum(dim=1) / trk_hit_mask.sum(dim=1, keepdim=True).clamp(min=1).to(hh.dtype)
        safe_hm = trk_hit_mask.clone()
        if (~has_hit).any():
            safe_hm[~has_hit, 0] = True
        h_w = torch.softmax(self.hit_score(hh).squeeze(-1).float().masked_fill(~safe_hm, float('-inf')), dim=-1).to(hh.dtype)
        h_attn = (hh * h_w.unsqueeze(-1)).sum(dim=1)
        hit_pool = torch.cat([h_mean, h_attn], dim=-1) * has_hit.unsqueeze(-1).to(hh.dtype)

        spec_glob = self.fusion(torch.cat([cand_pool, hit_pool], dim=-1)) * has_hit.unsqueeze(-1).to(hh.dtype)
        return spec_glob, cand_tok


class TrackletCalorimeterConeProjector(nn.Module):
    """Forward X0-cone continuation from micro-tracklets onto 3D stereo calorimeter crossings.

    Strictly translation-invariant and Y-mirror-invariant: distinguishes prompt gamma->ee EM
    conversions (early narrow energy burst in 1-4 X0) from penetrating tau/hadron daughters.
    """

    def forward(
        self,
        trk_kin: torch.Tensor,
        trk_mask: torch.Tensor,
        cal_xyz_mm: torch.Tensor,
        cal_e: torch.Tensor,
        cal_tof: torch.Tensor,
        cal_mask: torch.Tensor,
    ) -> torch.Tensor:
        with torch.amp.autocast(device_type=trk_kin.device.type, enabled=False):
            k = trk_kin.float()
            b, m, _ = k.shape
            if cal_xyz_mm.shape[1] == 0:
                return k.new_zeros((b, m, CONE_FEAT_DIM))
            x0, y0, z0 = k[..., 0:1], k[..., 1:2], k[..., 2:3]
            tx, ty = k[..., 3:4], k[..., 4:5]

            cx = cal_xyz_mm[:, None, :, 0].float()
            cy = cal_xyz_mm[:, None, :, 1].float()
            cz = cal_xyz_mm[:, None, :, 2].float()
            ce = cal_e[:, None, :].float().clamp_min(0.0)
            ct = cal_tof[:, None, :].float()
            cm = (trk_mask[:, :, None] & cal_mask[:, None, :]).float()

            dz = cz - z0  # (B, M, N_cal)
            downstream = (dz >= -10.0).float() * cm
            dz_c = dz.clamp(min=0.0, max=800.0)
            dx = (x0 + tx * dz_c) - cx
            dy = (y0 + ty * dz_c) - cy
            dr = _safe_hypot(dx, dy, eps=1.0e-8)

            sig_r = 22.0 + 0.04 * dz_c
            w_wide = torch.exp(-0.5 * (dr / (2.0 * sig_r)).clamp_max(8.0).square()) * downstream
            w_narrow = torch.exp(-0.5 * (dr / 20.0).clamp_max(8.0).square()) * downstream

            # Near window (~1-4 X0 downstream, dz in [-10, 90] mm) vs Deep window (5-20 X0, dz in [90, 520] mm)
            m_near = ((dz >= -10.0) & (dz < 90.0)).float()
            m_deep = ((dz >= 90.0) & (dz <= 520.0)).float()

            e_near_wide = (ce * w_wide * m_near).sum(dim=-1)
            e_near_nar = (ce * w_narrow * m_near).sum(dim=-1)
            e_deep_wide = (ce * w_wide * m_deep).sum(dim=-1)
            e_tot_cone = (ce * w_wide).sum(dim=-1)

            nar_ratio = e_near_nar / (e_near_wide + 1e-4)
            deep_frac = e_deep_wide / (e_near_wide + e_deep_wide + 1e-4)
            w_all = ce * w_wide + 1e-6 * w_wide
            w_sum = w_all.sum(dim=-1).clamp_min(1e-5)
            mean_tof = (w_all * ct).sum(dim=-1) / w_sum
            mean_r = (w_all * dr).sum(dim=-1) / w_sum

            feat = torch.stack([
                torch.log1p(e_near_wide) / 4.0,
                torch.log1p(e_deep_wide) / 4.0,
                nar_ratio.clamp(0.0, 1.0),
                deep_frac.clamp(0.0, 1.0),
                mean_tof * (e_tot_cone > 1e-4).float(),
                (mean_r / 50.0).clamp(0.0, 4.0) * (e_tot_cone > 1e-4).float(),
            ], dim=-1)
            return feat * trk_mask.unsqueeze(-1).float()


class PixelCalVertexAnchor(nn.Module):
    """Two-stage coarse longitudinal-plane + fine near-vertex cluster PV anchor (FP32)."""

    def __init__(self, dim: int, heads: int = 8) -> None:
        super().__init__()
        self.coarse_queries = nn.Parameter(torch.randn(1, 4, dim) * 0.02)
        self.coarse_cross = ScaledQKNormAttention(dim=dim, heads=heads, init_scale=10.0, drop_path=0.05)
        self.coarse_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.SiLU())
        self.delta_coarse = nn.Linear(dim, 3)
        nn.init.normal_(self.delta_coarse.weight, std=1e-3)
        nn.init.zeros_(self.delta_coarse.bias)

        self.fine_rel_proj = nn.Sequential(nn.Linear(6, dim), nn.LayerNorm(dim), nn.SiLU())
        self.fine_queries = nn.Parameter(torch.randn(1, 4, dim) * 0.02)
        self.fine_cross = ScaledQKNormAttention(dim=dim, heads=heads, init_scale=10.0, drop_path=0.05)
        self.fine_head = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim), nn.SiLU())
        self.delta_fine = nn.Linear(dim, 3)
        self.log_var = nn.Linear(dim, 3)
        nn.init.normal_(self.delta_fine.weight, std=1e-3)
        nn.init.zeros_(self.delta_fine.bias)
        nn.init.normal_(self.log_var.weight, std=1e-3)
        nn.init.constant_(self.log_var.bias, 1.0)

    def forward(
        self,
        z_seq: torch.Tensor,
        z_act: torch.Tensor,
        v_cal_prior: torch.Tensor,
        pix_tokens: torch.Tensor,
        pix_xyz_mm: torch.Tensor,
        pix_mask: torch.Tensor,
        pix_plane: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = int(z_seq.shape[0])
        any_z = z_act.any(dim=1, keepdim=True)
        safe_z = z_act.clone()
        if (~any_z[:, 0]).any():
            safe_z[~any_z[:, 0], 0] = True
        q_c = self.coarse_queries.expand(b, -1, -1)
        h_coarse = self.coarse_head(self.coarse_cross(q_c, z_seq, safe_z).mean(dim=1))
        with torch.amp.autocast(device_type=z_seq.device.type, enabled=False):
            hf = h_coarse.float()
            prior = v_cal_prior.float()
            v_coarse = prior + torch.tanh(self.delta_coarse(hf)) * prior.new_tensor([150.0, 150.0, 350.0])

        if pix_xyz_mm.shape[1] > 0:
            k_sel = min(64, pix_xyz_mm.shape[1])
            c_idx, c_mask = select_near_vertex_clusters(
                pix_xyz_mm, pix_mask, v_coarse.detach(), k_vtx=k_sel, pix_plane=pix_plane
            )
            c_tok = torch.gather(pix_tokens, 1, c_idx.unsqueeze(-1).expand(-1, -1, pix_tokens.shape[-1]))
            c_xyz = torch.gather(pix_xyz_mm.float(), 1, c_idx.unsqueeze(-1).expand(-1, -1, 3))
            rel = c_xyz - v_coarse[:, None, :]
            rel_feat = torch.stack([
                rel[..., 0] / 15.0,
                rel[..., 1] / 15.0,
                rel[..., 2] / 40.0,
                torch.asinh(rel[..., 0] / 2.0) / 3.0,
                torch.asinh(rel[..., 1] / 2.0) / 3.0,
                torch.asinh(rel[..., 2] / 10.0) / 3.0,
            ], dim=-1)
            c_in = c_tok + self.fine_rel_proj(rel_feat.to(c_tok.dtype)) * c_mask.unsqueeze(-1).to(c_tok.dtype)
            h_fine_attn = self.fine_cross(self.fine_queries.expand(b, -1, -1), c_in, c_mask).mean(dim=1)
            any_pix = c_mask.any(dim=1, keepdim=True)
            h_fine_safe = torch.where(any_pix, h_fine_attn, h_coarse)
        else:
            h_fine_safe = h_coarse
        h_comb = self.fine_head(torch.cat([h_coarse, h_fine_safe], dim=-1))
        with torch.amp.autocast(device_type=h_comb.device.type, enabled=False):
            hcf = h_comb.float()
            v_nn = v_coarse + torch.tanh(self.delta_fine(hcf)) * v_coarse.new_tensor([25.0, 25.0, 80.0])
            lv = torch.clamp(self.log_var(hcf), min=-6.0, max=5.0)
            ax_sc = v_coarse.new_tensor(VERTEX_AXIS_SCALES_MM, dtype=torch.float32)
            scale_nn = (torch.exp(0.5 * lv) * ax_sc[None, :]).clamp(0.02, 150.0)
        return v_nn, scale_nn, h_comb.float()


class TrackletIPFormer(NextGenFlavourModel):
    """Soft Multi-Tracklet & Translation-Invariant 3D Skew-DCA Relational Transformer."""

    def __init__(self, geometry: dict, config: dict | None = None) -> None:
        super().__init__()
        self.config = dict(config or {})
        bounds = detector_bounds(geometry)
        stations = static_scint_stations(geometry, bounds)

        dim = int(self.config.get('dim', 176))
        heads = int(self.config.get('heads', 8))
        drop_path = float(self.config.get('drop_path', 0.10))
        head_drop = float(self.config.get('head_dropout', 0.15))
        n_rel_pre = int(self.config.get('n_rel_pre', 2))
        n_rel_post = int(self.config.get('n_rel_post', 2))
        n_global_layers = int(self.config.get('n_global_layers', 2))
        self.dim = dim
        self.heads = heads
        self.max_tracklets = int(self.config.get('max_tracklets', MAX_TRACKLETS))
        self.n_queries = int(self.config.get('n_queries', 16))
        self.x_pix_center = float(bounds['x_pix_center'])
        self.y_scint_center = float(bounds['y_scint_center'])
        self.z_mid = float(bounds['z_mid'])

        # Causal ablation flags
        self.disable_skew_dca = bool(self.config.get('disable_skew_dca', False))
        self.use_hard_avr_pv = bool(self.config.get('use_hard_avr_pv', False))
        self.use_legacy_16d_only = bool(self.config.get('use_legacy_16d_only', False))

        pix_x0 = torch.tensor(
            [cumulative_radiation_length(float(z), geometry) for z in bounds['pixel_Zpos']],
            dtype=torch.float32,
        )
        self.register_buffer('pix_plane_x0', pix_x0)
        self.register_buffer(
            'default_center_mm',
            torch.tensor([self.x_pix_center, self.y_scint_center, self.z_mid], dtype=torch.float32),
        )

        # 1. Pixel Cluster & 32x32 Micro-Patch Encoders
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

        # 2. Stereo Calorimeter (with Relativistic ToF) & IPT-Interlocked Spectrometer
        self.stereo_bars = NextGenStereoBarNexus(stations, dim=dim, heads=heads, drop_path=drop_path)
        self.spectrometer = NextGenSpectrometerEncoder(
            out_dim=dim, x_pix_center=self.x_pix_center, y_scint_center=self.y_scint_center
        )

        # 3. Longitudinal 69-plane profile encoder
        n_pix_p = len(bounds['pixel_Zpos'])
        n_st = int(stations['n_stations'])
        all_z = np.concatenate([
            np.asarray(bounds['pixel_Zpos'], dtype=np.float64),
            np.asarray(stations['station_z_mm'], dtype=np.float64),
        ])
        all_x0 = np.concatenate([
            pix_x0.numpy().astype(np.float64),
            np.asarray(stations['station_x0'], dtype=np.float64),
        ])
        is_pix = np.r_[np.ones(n_pix_p, dtype=np.float32), np.zeros(n_st, dtype=np.float32)]
        z_order = np.argsort(all_z, kind='mergesort')
        z_geom = np.column_stack([
            (all_z[z_order] - self.z_mid) / 1000.0,
            all_x0[z_order] / 50.0,
            is_pix[z_order],
            1.0 - is_pix[z_order],
        ]).astype(np.float32)
        self.n_st = n_st
        self.register_buffer('z_order_idx', torch.from_numpy(z_order).long())
        self.register_buffer('z_plane_geom', torch.from_numpy(z_geom))
        self.z_pos_proj = nn.Linear(4, dim)
        self.z_profile_encoder = AttentionStack(
            dim=dim, n_layers=1, heads=heads, init_scale=10.0, max_scale=50.0, drop_path=drop_path
        )
        self.z_pool_gate = nn.Linear(dim, 1)

        # 4. Soft Multi-Tracklet Branch OR Legacy 16-d Scalar Ablation Branch
        if self.use_legacy_16d_only:
            self.legacy_16d_token = nn.Sequential(
                nn.LayerNorm(MICRO_VTX_LEGACY_DIM),
                nn.Linear(MICRO_VTX_LEGACY_DIM, dim),
                nn.LayerNorm(dim),
                nn.SiLU(),
            )
            self.legacy_pv_mlp = nn.Sequential(
                nn.LayerNorm(dim + 16),
                nn.Linear(dim + 16, 128),
                nn.SiLU(),
                nn.Linear(128, 9),  # [dx, dy, dz, log_sig_x, log_sig_y, log_sig_z, sv_dx, sv_dy, sv_dz]
            )
            self.cone_projector = None
            self.trk_in = None
            self.pair_bias_mlp = None
            self.rel_attn_pre = None
            self.vertex_layer = None
            self.pv_anchor = None
            self.pv_scale_cal = None
            self.hard_pv_scale_mlp = None
            self.ip_reinject = None
            self.ip_gate = None
            self.rel_attn_post = None
            self.trk_pool_gate = None
        else:
            self.legacy_16d_token = None
            self.legacy_pv_mlp = None
            self.cone_projector = TrackletCalorimeterConeProjector()
            self.trk_in = nn.Sequential(
                nn.Linear(TRK_KIN_DIM + TRK_MORPH_DIM + CONE_FEAT_DIM, dim),
                nn.LayerNorm(dim),
                nn.SiLU(),
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
            )
            if not self.disable_skew_dca:
                self.pair_bias_mlp: nn.Module | None = nn.Sequential(
                    nn.Linear(PAIR_TRK_FEAT_DIM, 32),
                    nn.SiLU(),
                    nn.Linear(32, heads),
                )
            else:
                self.pair_bias_mlp = None

            self.rel_attn_pre = AttentionStack(
                dim=dim, n_layers=n_rel_pre, heads=heads, init_scale=12.0, max_scale=50.0, drop_path=drop_path
            )
            if self.use_hard_avr_pv:
                self.vertex_layer = None
                self.pv_anchor = None
                self.pv_scale_cal = None
                self.hard_pv_scale_mlp = nn.Sequential(
                    nn.LayerNorm(dim + 16),
                    nn.Linear(dim + 16, 96),
                    nn.SiLU(),
                    nn.Linear(96, 6),  # [log_sx, log_sy, log_sz, sv_dx, sv_dy, sv_dz]
                )
            else:
                self.pv_anchor = PixelCalVertexAnchor(dim=dim, heads=heads)
                self.vertex_layer = DifferentiableStudentTVertexLayer(hidden_dim=dim, df=3.0, n_iters=4)
                self.pv_scale_cal = nn.Sequential(
                    nn.LayerNorm(4 * dim + 16),
                    nn.Linear(4 * dim + 16, 128),
                    nn.SiLU(),
                    nn.Linear(128, 3),
                )
                nn.init.zeros_(self.pv_scale_cal[-1].bias)
                self.hard_pv_scale_mlp = None

            self.ip_reinject = nn.Sequential(
                nn.Linear(5, dim),
                nn.LayerNorm(dim),
                nn.SiLU(),
            )
            self.ip_gate = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))
            self.rel_attn_post = AttentionStack(
                dim=dim, n_layers=n_rel_post, heads=heads, init_scale=12.0, max_scale=50.0, drop_path=drop_path
            )
            self.trk_pool_gate = nn.Linear(dim, 1)

        # 5. Cross-Modal Perceiver Fusion & Multi-Task Heads
        self.latent_queries = nn.Parameter(torch.randn(1, self.n_queries, dim) * 0.02)
        self.cobs_query = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(16, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
        )
        self.global_blocks = nn.ModuleList([
            PerceiverCrossBlock(dim=dim, heads=heads, init_scale=12.0, max_scale=50.0, drop_path=drop_path)
            for _ in range(n_global_layers)
        ])
        self.pool_gate = nn.Linear(dim, 1)
        self.vtx_summary_proj = nn.Sequential(
            nn.Linear(9, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.SiLU(),
        )
        fused_in_dim = 6 * dim + (dim // 2)
        self.global_fusion = nn.Sequential(
            nn.LayerNorm(fused_in_dim),
            nn.Linear(fused_in_dim, 3 * dim),
            nn.SiLU(),
        )
        head_in = 3 * dim
        self.event_head_7 = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, 256),
            nn.SiLU(),
            nn.Dropout(head_drop),
            nn.Linear(256, 7),
        )
        self.charm_head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, 160),
            nn.SiLU(),
            nn.Dropout(head_drop),
            nn.Linear(160, 1),
        )
        self.kin_head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, 160),
            nn.SiLU(),
            nn.Linear(160, 4),
        )

    def build_param_groups(self, base_lr: float, weight_decay: float) -> list[dict]:
        return split_adamw_param_groups(self, base_lr, weight_decay)

    def learned_attention_scales(self) -> dict[str, list[float]]:
        return collect_attention_scales(self)

    def effective_attention_sharpness(self) -> dict[str, list[dict[str, float]]]:
        return collect_effective_attention_sharpness(self)

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
            elif name.startswith(('trk_in.', 'pair_bias_mlp.', 'rel_attn_', 'ip_reinject.', 'ip_gate', 'trk_pool_gate.', 'legacy_')):
                br = 'tracklet_relational'
            elif name.startswith(('vertex_layer.', 'pv_anchor.', 'pv_scale_cal.', 'hard_pv_scale_mlp.', 'vtx_summary_proj.')):
                br = 'vertex_layer'
            elif name.startswith(('z_pos_proj.', 'z_profile_encoder.', 'z_pool_gate.', 'latent_queries', 'cobs_query.', 'global_blocks.', 'pool_gate.', 'global_fusion.')):
                br = 'global_fusion'
            else:
                br = 'heads'
            groups.setdefault(br, []).append(param)
        return groups

    def _get_v_cal_prior(self, batch: dict, b: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        center = self.default_center_mm[None, :].expand(b, -1).to(device)
        cal_start = batch['cal_start'].float()
        cal_valid = cal_start[:, 3] > 0.5
        cal_xyz = torch.stack([cal_start[:, 1], cal_start[:, 2], cal_start[:, 0]], dim=-1)
        prior = torch.where(cal_valid[:, None], cal_xyz, center)
        return prior, cal_valid

    def _normalize_trk_kin(self, trk_kin: torch.Tensor) -> torch.Tensor:
        k = trk_kin.float()
        return torch.stack([
            (k[..., 0] - self.x_pix_center) / 135.0,
            (k[..., 1] - self.y_scint_center) / 100.0,
            (k[..., 2] - self.z_mid) / 1000.0,
            k[..., 3] / 0.25,
            k[..., 4] / 0.25,
            torch.log1p(k[..., 5].clamp_min(0.0) * 1e3) / 5.0,
            torch.log1p(k[..., 6].clamp_min(0.0) * 1e3) / 5.0,
            torch.log1p(k[..., 7].clamp_min(0.0) * 1e5) / 5.0,
            torch.log1p(k[..., 8].clamp_min(0.0) * 1e5) / 5.0,
            k[..., 9].clamp(0.0, 25.0) / 18.0,
            k[..., 10].clamp(0.0, 40.0) / 20.0,
            k[..., 11].clamp(0.0, 400.0) / 200.0,
        ], dim=-1)

    def _hard_avr_pv(self, batch: dict, v_cal_prior: torch.Tensor) -> torch.Tensor:
        cal_start = batch['cal_start'].float()
        cal_valid = cal_start[:, 3:4] > 0.5
        prior = v_cal_prior
        if 'avr_pv_mm' in batch:
            avr = batch['avr_pv_mm'].float()
            dz_cal = avr[:, 2:3] - cal_start[:, 0:1]
            z_ok = (~cal_valid) | ((dz_cal >= -50.0) & (dz_cal <= 150.0))
            avr_valid = (avr[:, 3:4] > 0.5) & z_ok
            prior = torch.where(avr_valid, avr[:, :3], prior)
        return prior

    @staticmethod
    def _compute_ip_features(
        trk_kin: torch.Tensor,
        trk_mask: torch.Tensor,
        v_pv_mm: torch.Tensor,
        pi_pv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with torch.amp.autocast(device_type=trk_kin.device.type, enabled=False):
            kf = trk_kin.float()
            vp = v_pv_mm.float()
            dz_f = (vp[:, None, 2] - kf[..., 2]).clamp(-350.0, 350.0)
            dx_pv = (kf[..., 0] + kf[..., 3] * dz_f) - vp[:, None, 0]
            dy_pv = (kf[..., 1] + kf[..., 4] * dz_f) - vp[:, None, 1]
            d0 = _safe_hypot(dx_pv, dy_pv, eps=1.0e-12) * trk_mask.float()
            var_xf = kf[..., 5].clamp_min(4e-5) + dz_f.square() * kf[..., 7].clamp_min(1e-7)
            var_yf = kf[..., 6].clamp_min(4e-5) + dz_f.square() * kf[..., 8].clamp_min(1e-7)
            s_d0 = torch.sqrt((dx_pv.square() / var_xf + dy_pv.square() / var_yf) * trk_mask.float() + 1e-10) * trk_mask.float()
            ux = (kf[..., 3] * trk_mask.float()).sum(dim=1, keepdim=True)
            uy = (kf[..., 4] * trk_mask.float()).sum(dim=1, keepdim=True)
            u_n = torch.sqrt(ux.square() + uy.square() + 1.0e-10).clamp_min(1e-5)
            signed_d0 = (-(dx_pv * (ux / u_n) + dy_pv * (uy / u_n))) * trk_mask.float()
            weight_col = pi_pv.float() if pi_pv is not None else trk_mask.float()
            return torch.stack([
                torch.asinh(d0 / 0.05) / 3.0,
                torch.log1p(s_d0) / 3.0,
                torch.asinh(signed_d0 / 0.05) / 3.0,
                weight_col,
            ], dim=-1) * trk_mask.unsqueeze(-1).float()

    def forward_NextGen(self, batch: dict[str, torch.Tensor]) -> NextGenModelOutput:
        b = int(batch['c_obs'].shape[0])
        device = batch['c_obs'].device

        # 1. Pixel cluster tokens + 32x32 micro-patch CNN
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

        # 3. Extract or read pre-extracted soft non-exclusive tracklets
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

        # Split IPT stubs (trk_morph[..., 1] > 0.5) for Magnet-0 spectrometer interlock
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

        # 5. Tracklet Relational Geometry & Differentiable Student-t Vertexing (or Causal Ablations)
        pi_sv = trk_kin.new_zeros((b, trk_kin.shape[1]))
        v_nn_mm: torch.Tensor | None = None
        scale_nn_mm: torch.Tensor | None = None
        h_vtx_ctx: torch.Tensor | None = None
        sigma_ivw: torch.Tensor | None = None
        if self.use_legacy_16d_only:
            assert self.legacy_16d_token is not None and self.legacy_pv_mlp is not None
            mv_feat = batch.get('micro_vtx_feat', torch.zeros((b, MICRO_VTX_LEGACY_DIM), device=device, dtype=torch.float32)).float()
            mv_tok = self.legacy_16d_token(mv_feat).unsqueeze(1).to(u_pix.dtype)
            mv_mask = mv_feat[:, 0:1] > -1.0  # always active token so gradient flows cleanly
            trk_tokens = mv_tok
            trk_tok_mask = mv_mask
            trk_glob = mv_tok.squeeze(1).float()
            with torch.amp.autocast(device_type=device.type, enabled=False):
                pv_base = self._hard_avr_pv(batch, v_cal_prior)
                pv_raw = self.legacy_pv_mlp(torch.cat([cal_global.float(), batch['c_obs'].float()], dim=-1))
                v_pv_mm = pv_base + torch.tanh(pv_raw[:, 0:3]) * pv_base.new_tensor([20.0, 20.0, 60.0])
                scale_pv_mm = torch.exp(pv_raw[:, 3:6].clamp(-3.0, 4.5)).clamp(0.02, 150.0)
                sv_disp_mm = pv_raw[:, 6:9]
                delta_chi2_2v = trk_kin.new_zeros((b, 1))
        else:
            assert self.cone_projector is not None and self.trk_in is not None
            assert self.rel_attn_pre is not None and self.rel_attn_post is not None
            assert self.ip_reinject is not None and self.ip_gate is not None and self.trk_pool_gate is not None
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

            if self.pair_bias_mlp is not None:
                r_ab = pairwise_skew_dca_features(trk_kin, trk_mask)
                pair_bias = self.pair_bias_mlp(r_ab.to(u_pix.dtype)).permute(0, 3, 1, 2)
            else:
                pair_bias = None

            h_trk = self.rel_attn_pre(h_trk, trk_mask, pair_bias=pair_bias)

            if self.use_hard_avr_pv:
                assert self.hard_pv_scale_mlp is not None
                with torch.amp.autocast(device_type=device.type, enabled=False):
                    v_pv_mm = self._hard_avr_pv(batch, v_cal_prior)
                    pv_raw = self.hard_pv_scale_mlp(torch.cat([cal_global.float(), batch['c_obs'].float()], dim=-1))
                    scale_pv_mm = torch.exp(pv_raw[:, 0:3].clamp(-3.0, 4.5)).clamp(0.02, 150.0)
                    sv_disp_mm = pv_raw[:, 3:6]
                    delta_chi2_2v = trk_kin.new_zeros((b, 1))
                    trk_ip_feat = self._compute_ip_features(trk_kin, trk_mask, v_pv_mm)
            else:
                assert self.vertex_layer is not None and self.pv_anchor is not None and self.pv_scale_cal is not None
                v_nn_mm, scale_nn_mm, h_vtx_ctx = self.pv_anchor(
                    z_seq, z_act, v_cal_prior, u_pix, batch['pix_xyz_mm'], pix_mask, batch['pix_plane']
                )
                vtx_out = self.vertex_layer(
                    trk_kin, h_trk, trk_mask, v_cal_prior, cal_valid=cal_valid
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
                sv_disp_mm = vtx_out['sv_disp_mm']
                delta_chi2_2v = vtx_out['delta_chi2_2v']
                pi_sv = vtx_out['pi_sv']

            ip_in = torch.cat([trk_ip_feat, pi_sv.unsqueeze(-1)], dim=-1)
            h_trk = h_trk + self.ip_gate * self.ip_reinject(ip_in.to(h_trk.dtype)) * trk_mask.unsqueeze(-1).to(h_trk.dtype)
            h_trk = self.rel_attn_post(h_trk, trk_mask, pair_bias=pair_bias)
            h_trk = h_trk * trk_mask.unsqueeze(-1).to(h_trk.dtype)

            trk_tokens = h_trk
            trk_tok_mask = trk_mask
            with torch.amp.autocast(device_type=device.type, enabled=False):
                any_trk = trk_mask.any(dim=1, keepdim=True)
                safe_tm = trk_mask.clone()
                if (~any_trk[:, 0]).any():
                    safe_tm[~any_trk[:, 0], 0] = True
                t_logits = self.trk_pool_gate(h_trk.float()).squeeze(-1).masked_fill(~safe_tm, -1e4)
                t_w = torch.softmax(t_logits, dim=1) * trk_mask.float()
                trk_glob = (t_w.unsqueeze(-1) * h_trk.float()).sum(dim=1)
                if self.pv_scale_cal is not None:
                    assert sigma_ivw is not None and h_vtx_ctx is not None
                    pv_ctx = torch.cat([h_vtx_ctx, z_prof_glob, cal_global.float(), trk_glob, batch['c_obs'].float()], dim=-1)
                    scale_pv_mm = (sigma_ivw * torch.exp(self.pv_scale_cal(pv_ctx).clamp(-2.0, 2.0))).clamp(0.02, 150.0)

        # 6. Cross-Modal Perceiver Fusion
        source = torch.cat([trk_tokens, u_pix, pix_plane_tok, cal_tok, spec_cand_tok], dim=1)
        source_mask = torch.cat([trk_tok_mask, pix_mask, pix_plane_act, cal_mask, batch['trk_cand_mask']], dim=1)
        cobs_q = self.cobs_query(batch['c_obs']).unsqueeze(1)
        latents = torch.cat([self.latent_queries.expand(b, -1, -1), cobs_q], dim=1)
        latent_mask = torch.ones((b, self.n_queries + 1), dtype=torch.bool, device=device)

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

            fused = self.global_fusion(
                torch.cat([
                    pool_attn,
                    z_prof_glob,
                    cal_global.float(),
                    trk_glob,
                    spec_glob.float(),
                    q_cobs.float(),
                    vtx_emb,
                ], dim=-1)
            )
            logits7 = self.event_head_7(fused)
            probs4 = marginal_four_from_seven(logits7, is_probs=False)
            charm_logit = self.charm_head(fused).squeeze(-1)
            kin_pred = self.kin_head(fused)

        aux_out: dict[str, torch.Tensor] = {
            'delta_chi2_2v': delta_chi2_2v.float(),
            'p_sv_trklet': pi_sv.float(),
            'n_tracklets': trk_mask.sum(dim=1).float(),
        }
        if v_nn_mm is not None and scale_nn_mm is not None:
            aux_out['v_nn_mm'] = v_nn_mm.float()
            aux_out['scale_nn_mm'] = scale_nn_mm.float()

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


def tracklet_ipformer_loss(
    out: NextGenModelOutput | dict[str, Any],
    batch: dict[str, torch.Tensor],
    *,
    fixed_denoms: dict[str, float] | None = None,
    w_proc: float = 1.0,
    w_charm: float = 0.25,
    w_vtx: float = 0.20,
    w_kin: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute 4-head Clean-v3 compatible multitask loss with Student-t(df=3) PV NLL."""
    if isinstance(out, NextGenModelOutput):
        logits7 = out.logits7
        probs4 = out.probs4
        charm_logit = out.charm_logit
        v_pv = out.v_pv_mm
        scale_pv = out.scale_pv_mm
        kin_pred = out.kin_pred
    else:
        logits7 = out['logits']
        probs4 = out['probs4']
        charm_logit = out['charm_logit']
        v_pv = out['v_hat']
        scale_pv = out['scale_pv_mm']
        kin_pred = out['kin_pred']

    fd = fixed_denoms or {}
    sw = batch.get('sample_weight')

    if 'label_7' in batch:
        l_proc = marginal_tau_branch_ce(
            logits7,
            batch['label_7'],
            batch['label'],
            sample_weight=sw,
            denom_override=fd.get('proc'),
        )
    else:
        per_p = F.nll_loss(torch.log(probs4.float().clamp_min(1e-7)), batch['label'].long(), reduction='none')
        denom_p = (
            torch.as_tensor(fd['proc'], dtype=torch.float32, device=per_p.device).clamp_min(1e-8)
            if 'proc' in fd
            else torch.tensor(float(len(per_p)), dtype=torch.float32, device=per_p.device).clamp_min(1e-8)
        )
        l_proc = per_p.sum() / denom_p

    if 'charm_flag' in batch and 'charm_valid' in batch:
        l_charm = focal_charm_bce(
            charm_logit,
            batch['charm_flag'],
            batch['charm_valid'],
            gamma=1.5,
            pos_weight=3.0,
            sample_weight=None,
            denom_override=fd.get('charm'),
        )
    else:
        l_charm = charm_logit.sum() * 0.0

    if 'vertex_xyz_mm' in batch and 'vertex_valid' in batch:
        l_vtx = student_t_vertex_loss(
            v_pv,
            scale_pv,
            batch['vertex_xyz_mm'],
            batch['vertex_valid'],
            df=3.0,
            sample_weight=None,
            denom_override=fd.get('vtx'),
        )
        aux_d = out.aux if isinstance(out, NextGenModelOutput) else out
        if 'v_nn_mm' in aux_d and 'scale_nn_mm' in aux_d:
            l_vtx = l_vtx + 0.15 * student_t_vertex_loss(
                aux_d['v_nn_mm'],
                aux_d['scale_nn_mm'],
                batch['vertex_xyz_mm'],
                batch['vertex_valid'],
                df=3.0,
                sample_weight=None,
                denom_override=fd.get('vtx'),
            )
    else:
        l_vtx = (v_pv.sum() + scale_pv.sum()) * 0.0

    if kin_pred is not None and 'vis_kinematics' in batch and 'kin_valid' in batch:
        kv = batch['kin_valid'] > 0
        if kv.any():
            with torch.amp.autocast(device_type=kin_pred.device.type, enabled=False):
                ke = F.smooth_l1_loss(
                    kin_pred[kv].float(),
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
            l_kin = kin_pred.sum() * 0.0
    else:
        l_kin = logits7.sum() * 0.0

    total = w_proc * l_proc.float() + w_charm * l_charm.float() + w_vtx * l_vtx.float() + w_kin * l_kin.float()
    return total, probs4
