"""Proposal E: PinpointSSL-JEPA — Whole-Plane Silicon Drop JEPA + 8-GMM 2D Cluster Density,
Stereo Scintillator X/Y + ToF Cross-Prediction, Cross-Magnet-0 Full Spectrometer Drop
Pre-Calibration, VICReg Anti-Collapse, and Layer-Decayed Relational Tracklet Fine-Tuning.

Strictly follows the NOMAE/Sonata anti-shortcut rule during SSL pretraining:
  - Masked silicon planes have 100% of their clusters, 32x32 patches, and plane summaries
    removed prior to student encoding; the predictor receives ONLY longitudinal depth
    `(z_p, X0_p)` as query tokens, guaranteeing exact `0.0` sensitivity to masked-plane `(x, y)`.
  - Zero truth/sidecar inputs during SSL pretraining and during inference (`NextGenFlavourModel`).
"""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from analysis.flavour.nextgen.dual_regime_modalnet import (
    M_TRK_MAX,
    N_SPEC_CAND_MAX,
    extract_nonexclusive_spec_candidates,
)
from analysis.flavour.nextgen.protocols import (
    NextGenFlavourModel,
    NextGenModelOutput,
    marginal_four_from_seven,
    student_t_vertex_loss,
)
from analysis.flavour.nextgen.tau_kinema_diffnet import _focal_bce_from_prob
from analysis.flavour.nextgen.tracklet_ipformer import (
    CONE_FEAT_DIM,
    PixelCalVertexAnchor,
    TrackletCalorimeterConeProjector,
    TrackletIPFormer,
)
from analysis.flavour.nextgen.tracklet_ops import (
    DifferentiableStudentTVertexLayer,
    PAIR_TRK_FEAT_DIM,
    TRK_KIN_DIM,
    TRK_MORPH_DIM,
    compute_ipt_spectrometer_interlock,
    compute_scint_tof_residual_ns,
    extract_soft_tracklets_batch,
    extract_soft_tracklets_single,
    pairwise_skew_dca_features,
)
from analysis.flavour.v3.common.domains import cumulative_radiation_length, detector_bounds
from analysis.flavour.v3.common.layers import (
    DropPath,
    ScaledQKNormAttention,
    is_weight_decay_exempt,
)
from analysis.flavour.v3.v1_b1pp.models import TransverseLayerEncoder
from analysis.flavour.v3.v1_b1pp.spectrometer import SIGMA_SAGITTA_Y_MM
from analysis.flavour.v3.v2_pinpointformer.attention import (
    AttentionStack,
    PerceiverCrossBlock,
    collect_attention_scales,
)
from analysis.flavour.v3.v2_pinpointformer.pixel_tokenizer import (
    MicroPatchConv2D,
    PIX_FEAT_DIM,
    PIX_PLANE_SUMMARY_DIM,
    extract_pf_item,
)
from analysis.flavour.v3.v2_pinpointformer.stereo_bars import (
    MAX_CROSS_PER_SIDE,
    N_MAX_NEXUS,
    static_scint_stations,
)
from analysis.flavour.v3.v3_multitask.losses import marginal_tau_branch_ce


N_PIX_PLANES = 15
GMM_COMPONENTS = 8


def extract_jepa_item(
    index: int,
    observation: dict,
    label: int | None,
    dom: dict,
    geometry: dict,
    bounds: dict,
    config: dict | None = None,
) -> dict:
    """Pure-observation item extractor for PinpointSSL-JEPA (zero truth/sidecar access)."""
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


class _MaskedCrossAttention(nn.Module):
    """FP32-exact ScaledQKNorm cross-attention supporting per-(query, key) boolean mask (B, Nq, Nk)."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        init_scale: float = 12.0,
        max_scale: float = 50.0,
        residual_init: float = 0.5,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.max_scale = max_scale

        self.q_norm = nn.LayerNorm(dim)
        self.k_norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

        self.log_scale = nn.Parameter(torch.full((1, heads, 1, 1), math.log(init_scale)))
        self.len_beta = nn.Parameter(torch.full((1, heads, 1, 1), -2.0))
        self.gamma_attn = nn.Parameter(torch.full((dim,), residual_init))
        self.drop_path_attn = DropPath(drop_path)

        self.ff_norm = nn.LayerNorm(dim)
        self.ff_up = nn.Linear(dim, 4 * dim)
        self.ff_down = nn.Linear(2 * dim, dim)
        self.gamma_ff = nn.Parameter(torch.full((dim,), residual_init))
        self.drop_path_ff = DropPath(drop_path)

    def forward(
        self,
        q_in: torch.Tensor,
        source: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        """q_in: (B, Nq, D), source: (B, Nk, D), pair_mask: (B, Nq, Nk) bool."""
        b, nq, _ = q_in.shape
        nk = source.shape[1]
        if nk == 0:
            g, val = self.ff_up(self.ff_norm(q_in)).chunk(2, dim=-1)
            return q_in + self.drop_path_ff(self.gamma_ff * self.ff_down(F.silu(g) * val))

        q = self.q_proj(self.q_norm(q_in)).view(b, nq, self.heads, self.head_dim).transpose(1, 2)
        k_in = self.k_norm(source)
        k = self.k_proj(k_in).view(b, nk, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(k_in).view(b, nk, self.heads, self.head_dim).transpose(1, 2)

        q_hat = F.normalize(q.float(), p=2, dim=-1, eps=1e-6)
        k_hat = F.normalize(k.float(), p=2, dim=-1, eps=1e-6)

        valid_cnt = pair_mask.sum(dim=-1).clamp(min=2).float()  # (B, Nq)
        scale = torch.clamp(self.log_scale.exp(), min=1.0, max=self.max_scale)
        len_factor = 1.0 + F.softplus(self.len_beta) * torch.log(valid_cnt)[:, None, :, None]

        logits = torch.matmul(q_hat, k_hat.transpose(-2, -1)) * (scale * len_factor)  # (B, H, Nq, Nk)
        empty_row = ~pair_mask.any(dim=-1)  # (B, Nq)
        safe_mask = pair_mask.clone()
        if empty_row.any():
            safe_mask[..., 0] = safe_mask[..., 0] | empty_row
        logits = logits.masked_fill(~safe_mask[:, None, :, :], float('-inf'))

        probs = torch.softmax(logits, dim=-1)
        if empty_row.any():
            probs = probs * (~empty_row)[:, None, :, None].to(probs.dtype)

        attended = torch.matmul(probs.to(v.dtype), v)
        attended = attended.transpose(1, 2).contiguous().view(b, nq, self.dim)
        h = q_in + self.drop_path_attn(self.gamma_attn * self.out_proj(attended))

        g, val = self.ff_up(self.ff_norm(h)).chunk(2, dim=-1)
        return h + self.drop_path_ff(self.gamma_ff * self.ff_down(F.silu(g) * val))


class StereoCalorimeterSSLModule(nn.Module):
    """42-station X/Y stereo scintillator + relativistic ToF encoder with view-masking support."""

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

        self.xbar_enc = TransverseLayerEncoder(7, (32, 48, 64), stem_k=3, stem_s=1, out_dim=dim)
        self.ybar_enc = TransverseLayerEncoder(7, (32, 48, 64), stem_k=3, stem_s=1, out_dim=dim)
        self.stereo_2d_proj = nn.Sequential(nn.Linear(256, dim), nn.LayerNorm(dim), nn.SiLU())
        self.station_fuse = nn.Sequential(nn.Linear(2 * dim + 6, dim), nn.LayerNorm(dim), nn.SiLU())
        self.nexus_in = nn.Sequential(nn.Linear(9, dim), nn.LayerNorm(dim), nn.SiLU())
        self.pair_mlp = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, heads))
        self.bipartite_attn = nn.ModuleList([
            ScaledQKNormAttention(dim, heads=heads, init_scale=12.0, drop_path=drop_path)
            for _ in range(2)
        ])

    def build_grids_and_targets(
        self,
        batch: dict[str, torch.Tensor],
        stereo_panel_drop_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Build 42-station X and 35-doublet Y bar grids, respecting optional view drop masks."""
        b = int(batch['c_obs'].shape[0])
        device = batch['c_obs'].device
        bar_ev = batch['bar_ev']
        bar_panel = batch['bar_panel']
        bar_horiz = batch['bar_horiz']
        bar_idx = batch['bar_idx']
        bar_val = batch['bar_val'].float()

        bt_key = 'bar_hit_time_ns' if 'bar_hit_time_ns' in batch else ('bar_time_ns' if 'bar_time_ns' in batch else None)
        n_hits = len(bar_ev)

        if n_hits > 0:
            bar_st = self.panel_to_station[bar_panel]
            if stereo_panel_drop_mask is not None:
                is_x = bar_horiz == 0
                drop_x = stereo_panel_drop_mask[bar_ev, bar_st, 0]
                drop_y = stereo_panel_drop_mask[bar_ev, bar_st, 1]
                keep = ~(torch.where(is_x, drop_x, drop_y))
            else:
                keep = torch.ones(n_hits, dtype=torch.bool, device=device)

            bar_z = self.station_z_mm[bar_st]
            if bt_key is not None and len(batch[bt_key]) == n_hits:
                ht = batch[bt_key].float()
                bar_tof = compute_scint_tof_residual_ns(
                    ht, bar_z, valid_mask=(keep & (ht >= 0.0)), event_idx=bar_ev
                )
            else:
                bar_tof = torch.zeros(n_hits, device=device, dtype=torch.float32)
        else:
            keep = torch.empty(0, dtype=torch.bool, device=device)
            bar_tof = torch.empty(0, device=device, dtype=torch.float32)

        grid_x = torch.zeros((b, 7, self.n_st, self.n_bars), device=device, dtype=torch.float32)
        grid_y = torch.zeros((b, 7, self.n_dbl, self.n_bars), device=device, dtype=torch.float32)

        x_mask_b = keep & (bar_horiz == 0)
        y_mask_b = keep & (bar_horiz == 1)
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
        return {'grid_x': grid_x, 'grid_y': grid_y}

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        stereo_panel_drop_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        b = int(batch['c_obs'].shape[0])
        device = batch['c_obs'].device
        grids = self.build_grids_and_targets(batch, stereo_panel_drop_mask=stereo_panel_drop_mask)
        grid_x = grids['grid_x']
        grid_y = grids['grid_y']

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
        y_tof_full = torch.zeros((b, self.n_st, self.n_bars), device=device)
        y_full[:, self.doublet_stations] = y_tok
        y_act_full[:, self.doublet_stations] = y_act
        y_ex_full[:, self.doublet_stations] = y_ex
        y_has_full[:, self.doublet_stations] = y_has
        y_tof_full[:, self.doublet_stations] = y_tof

        e_x = grid_x[:, 0].sum(dim=-1)
        e_y = y_ex_full.sum(dim=-1)
        w_x = grid_x[:, 0] + 1e-4 * grid_x[:, 4]
        w_y = y_ex_full + 1e-4 * y_has_full
        x_cent = (w_x * self.bar_x_mm[None, None, :]).sum(dim=-1) / w_x.sum(dim=-1).clamp_min(1e-6)
        y_cent = (w_y * self.bar_y_mm[None, None, :]).sum(dim=-1) / w_y.sum(dim=-1).clamp_min(1e-6)
        x_cent = torch.where(x_act, x_cent, torch.full_like(x_cent, self.x_c))
        y_cent = torch.where(y_act_full, y_cent, torch.full_like(y_cent, self.y_c))

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
        _, top_ix = torch.topk(score_x, k=MAX_CROSS_PER_SIDE, dim=-1)
        _, top_iy = torch.topk(score_y, k=MAX_CROSS_PER_SIDE, dim=-1)
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
            'st_tokens': cal_tok[:, :self.n_st],
            'st_mask': st_mask,
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


class PinpointSSLBackbone(nn.Module):
    """Shared multi-modal PINPOINT backbone trained via 3-task JEPA SSL and fine-tuned."""

    def __init__(self, geometry: dict, bounds: dict, stations: dict, cfg: dict) -> None:
        super().__init__()
        dim = int(cfg.get('dim', 176))
        heads = int(cfg.get('heads', 8))
        drop_path = float(cfg.get('drop_path', 0.10))
        n_z_layers = int(cfg.get('n_z_layers', 2))
        n_global_layers = int(cfg.get('n_global_layers', 2))

        self.dim = dim
        self.heads = heads
        self.n_queries = int(cfg.get('n_queries', 16))
        self.n_st = int(stations['n_stations'])
        self.x_pix_center = float(bounds['x_pix_center'])
        self.y_scint_center = float(bounds['y_scint_center'])
        self.z_mid = float(stations['z_mid'])

        pix_z = np.asarray(bounds['pixel_Zpos'], dtype=np.float64)
        pix_x0 = np.array(
            [cumulative_radiation_length(float(z), geometry) for z in pix_z],
            dtype=np.float32,
        )
        self.register_buffer('pix_plane_z', torch.from_numpy(pix_z.astype(np.float32)))
        self.register_buffer('pix_plane_x0', torch.from_numpy(pix_x0))
        pix_geom = np.column_stack([
            (pix_z - self.z_mid) / 1000.0,
            pix_x0.astype(np.float64) / 50.0,
        ]).astype(np.float32)
        self.register_buffer('pix_plane_geom', torch.from_numpy(pix_geom))

        # Level 0: Sensor Front-End Embeddings
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
        self.pix_plane_queries = nn.Parameter(torch.randn(1, N_PIX_PLANES, dim) * 0.02)
        self.pix_plane_mask_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        self.stereo_bars = StereoCalorimeterSSLModule(stations, dim=dim, heads=heads, drop_path=drop_path)

        # Spectrometer front-end & Level 1 encoders
        self.spec_cand_in = nn.Sequential(
            nn.Linear(14 + 4 + 3, 96),
            nn.LayerNorm(96),
            nn.SiLU(),
            nn.Linear(96, 96),
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

        # Level 1: Intra-Modality Encoders
        self.pix_plane_cross = _MaskedCrossAttention(dim=dim, heads=heads, init_scale=12.0, drop_path=drop_path)
        self.pix_plane_self = ScaledQKNormAttention(dim=dim, heads=heads, init_scale=12.0, drop_path=drop_path)
        self.ipt_self = ScaledQKNormAttention(dim=dim, heads=heads, init_scale=10.0, drop_path=drop_path)
        self.ipt_pool_gate = nn.Linear(dim, 1)

        self.spec_cand_attn1 = ScaledQKNormAttention(96, heads=4, init_scale=8.0, drop_path=0.05)
        self.spec_cand_attn2 = ScaledQKNormAttention(96, heads=4, init_scale=8.0, drop_path=0.05)
        self.spec_cand_to_dim = nn.Sequential(nn.Linear(96, dim), nn.LayerNorm(dim))
        self.spec_glob_fusion = nn.Sequential(nn.Linear(192 + 128 + dim, dim), nn.LayerNorm(dim), nn.SiLU())

        # Level 2: Longitudinal 69-plane transport encoder
        all_z = np.concatenate([pix_z, np.asarray(stations['station_z_mm'], dtype=np.float64)])
        all_x0 = np.concatenate([pix_x0.astype(np.float64), np.asarray(stations['station_x0'], dtype=np.float64)])
        is_pix = np.r_[np.ones(N_PIX_PLANES, dtype=np.float32), np.zeros(self.n_st, dtype=np.float32)]
        z_order = np.argsort(all_z, kind='mergesort')
        inv_z_order = np.argsort(z_order)
        z_geom = np.column_stack([
            (all_z[z_order] - self.z_mid) / 1000.0,
            all_x0[z_order] / 50.0,
            is_pix[z_order],
            1.0 - is_pix[z_order],
        ]).astype(np.float32)
        self.register_buffer('z_order_idx', torch.from_numpy(z_order).long())
        self.register_buffer('pix_slots_in_z', torch.from_numpy(inv_z_order[:N_PIX_PLANES]).long())
        self.register_buffer('cal_slots_in_z', torch.from_numpy(inv_z_order[N_PIX_PLANES:]).long())
        self.register_buffer('z_plane_geom', torch.from_numpy(z_geom))

        self.z_pos_proj = nn.Linear(4, dim)
        self.z_profile_encoder = AttentionStack(
            dim=dim, n_layers=n_z_layers, heads=heads, init_scale=10.0, max_scale=50.0, drop_path=drop_path
        )
        self.z_pool_gate = nn.Linear(dim, 1)

        # Level 3: Cross-Modal Perceiver Fusion
        self.latent_queries = nn.Parameter(torch.randn(1, self.n_queries, dim) * 0.02)
        self.ssl_query_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
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

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        pix_plane_drop_mask: torch.Tensor | None = None,
        pix_cluster_drop_mask: torch.Tensor | None = None,
        stereo_panel_drop_mask: torch.Tensor | None = None,
        spec_drop_mask: torch.Tensor | None = None,
        ssl_mode: bool = False,
        ipt_kin_override: torch.Tensor | None = None,
        ipt_mask_override: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run shared multi-modal encoding.

        When `pix_plane_drop_mask` is supplied, ALL cluster features, 32x32 patches, and
        plane summaries on masked planes are completely zeroed/removed prior to encoding,
        leaving ONLY `(z_p, X0_p)` longitudinal geometry for masked slots.
        """
        b = int(batch['c_obs'].shape[0])
        device = batch['c_obs'].device

        raw_pix_mask = batch['pix_mask'].bool()
        raw_pix_plane = batch['pix_plane'].long().clamp(0, N_PIX_PLANES - 1)
        raw_pix_feat = batch['pix_feat']
        raw_pix_summary = batch['pix_plane_summary']

        if pix_plane_drop_mask is not None:
            drop_cl = torch.gather(pix_plane_drop_mask.bool(), 1, raw_pix_plane)
            eff_pix_mask = raw_pix_mask & (~drop_cl)
            eff_plane_summary = raw_pix_summary * (~pix_plane_drop_mask.bool()).unsqueeze(-1).to(raw_pix_summary.dtype)
            plane_unmasked = ~pix_plane_drop_mask.bool()
        else:
            eff_pix_mask = raw_pix_mask
            eff_plane_summary = raw_pix_summary
            plane_unmasked = torch.ones((b, N_PIX_PLANES), dtype=torch.bool, device=device)

        if pix_cluster_drop_mask is not None:
            eff_pix_mask = eff_pix_mask & (~pix_cluster_drop_mask.bool())

        eff_pix_feat = raw_pix_feat * eff_pix_mask.unsqueeze(-1).to(raw_pix_feat.dtype)
        u_pix = self.pix_in(eff_pix_feat) * eff_pix_mask.unsqueeze(-1).to(eff_pix_feat.dtype)

        if len(batch['pix_patches']):
            patch_ev = batch['patch_ev']
            patch_cl = batch['patch_cl']
            patch_keep = eff_pix_mask[patch_ev, patch_cl]
            if patch_keep.any():
                kept_patches = batch['pix_patches'][patch_keep]
                patch_emb = self.patch_proj(self.patch_conv(kept_patches))
                delta_patch = torch.zeros_like(u_pix, dtype=patch_emb.dtype)
                delta_patch.index_put_(
                    (patch_ev[patch_keep], patch_cl[patch_keep]),
                    patch_emb,
                    accumulate=True,
                )
                u_pix = u_pix + delta_patch * eff_pix_mask.unsqueeze(-1).to(u_pix.dtype)

        # Plane-level tokens strictly from unmasked clusters + unmasked plane summaries
        pix_plane_sum_tok = self.pix_plane_emb(eff_plane_summary) * plane_unmasked.unsqueeze(-1).to(u_pix.dtype)
        plane_ids = torch.arange(N_PIX_PLANES, device=device, dtype=torch.long)[None, :, None]
        same_plane = (raw_pix_plane[:, None, :] == plane_ids) & eff_pix_mask[:, None, :] & plane_unmasked[:, :, None]
        pix_plane_act_real = same_plane.any(dim=-1) | ((eff_plane_summary[..., 0] > 0.5) & plane_unmasked)

        q_planes = self.pix_plane_queries.expand(b, -1, -1) + pix_plane_sum_tok
        z_pix_planes = self.pix_plane_cross(q_planes, u_pix, same_plane)
        if pix_plane_drop_mask is not None:
            mask_tok_exp = self.pix_plane_mask_token.expand(b, N_PIX_PLANES, -1).to(z_pix_planes.dtype)
            z_pix_planes = torch.where(pix_plane_drop_mask[:, :, None], mask_tok_exp, z_pix_planes)
            pix_plane_seq_act = pix_plane_act_real | pix_plane_drop_mask.bool()
        else:
            pix_plane_seq_act = pix_plane_act_real

        z_pix_planes = self.pix_plane_self(z_pix_planes, z_pix_planes, pix_plane_seq_act)

        # IPT global representation strictly from unmasked clusters in planes 12..14
        ipt_cl_mask = eff_pix_mask & (raw_pix_plane >= 12)
        has_ipt = ipt_cl_mask.any(dim=1)
        ipt_safe = ipt_cl_mask.clone()
        if (~has_ipt).any():
            ipt_safe[~has_ipt, 0] = True
        u_ipt = self.ipt_self(u_pix, u_pix, ipt_safe) * ipt_cl_mask.unsqueeze(-1).to(u_pix.dtype)
        iw_log = self.ipt_pool_gate(u_ipt).squeeze(-1).float().masked_fill(~ipt_safe, float('-inf'))
        iw = torch.softmax(iw_log, dim=-1) * ipt_cl_mask.float()
        ipt_glob = (iw.unsqueeze(-1) * u_ipt.float()).sum(dim=1)

        # Stereo calorimeter (supports X/Y view drop)
        bar_out = self.stereo_bars(batch, stereo_panel_drop_mask=stereo_panel_drop_mask)
        cal_tok = bar_out['cal_tokens']
        cal_mask = bar_out['cal_mask']
        cal_global = bar_out['cal_global']
        st_mask = bar_out['st_mask']

        # Spectrometer (supports full spectrometer drop)
        raw_cand = batch['trk_cand']
        raw_cand_mask = batch['trk_cand_mask'].bool()
        raw_hits = batch['trk_hits']
        raw_hit_mask = batch['trk_hit_mask'].bool()
        st_counts = batch.get(
            'trk_station_counts',
            torch.zeros((b, 3), dtype=torch.float32, device=device),
        )
        if spec_drop_mask is not None:
            keep_spec = (~spec_drop_mask.bool())
            eff_cand_mask = raw_cand_mask & keep_spec[:, None]
            eff_cand = raw_cand * eff_cand_mask.unsqueeze(-1).to(raw_cand.dtype)
            eff_hit_mask = raw_hit_mask & keep_spec[:, None]
            eff_hits = raw_hits * eff_hit_mask.unsqueeze(-1).to(raw_hits.dtype)
            eff_st_counts = st_counts * keep_spec[:, None].to(st_counts.dtype)
        else:
            eff_cand_mask = raw_cand_mask
            eff_cand = raw_cand
            eff_hit_mask = raw_hit_mask
            eff_hits = raw_hits
            eff_st_counts = st_counts

        if ipt_kin_override is not None and ipt_mask_override is not None:
            ilock = compute_ipt_spectrometer_interlock(
                ipt_kin_override,
                ipt_mask_override,
                eff_cand,
                eff_cand_mask,
                x_pix_center=self.x_pix_center,
                y_scint_center=self.y_scint_center,
            )
        else:
            ilock = eff_cand.new_zeros((b, eff_cand.shape[1], 4))

        st_occ_feat = (torch.log1p(eff_st_counts.clamp_min(0.0)) / 3.0)[:, None, :].expand(b, eff_cand.shape[1], 3)
        cand_in = torch.cat([eff_cand.float(), ilock.float(), st_occ_feat.float()], dim=-1)
        hc = self.spec_cand_in(cand_in.to(u_pix.dtype)) * eff_cand_mask.unsqueeze(-1).to(u_pix.dtype)
        hc = self.spec_cand_attn1(hc, hc, eff_cand_mask)
        hc = self.spec_cand_attn2(hc, hc, eff_cand_mask)
        spec_cand_tok = self.spec_cand_to_dim(hc) * eff_cand_mask.unsqueeze(-1).to(hc.dtype)

        has_cand = eff_cand_mask.any(dim=1)
        cm = eff_cand_mask.unsqueeze(-1).to(hc.dtype)
        c_mean = (hc * cm).sum(dim=1) / eff_cand_mask.sum(dim=1, keepdim=True).clamp(min=1).to(hc.dtype)
        c_max = hc.masked_fill(~eff_cand_mask.unsqueeze(-1), -1e4).amax(dim=1)
        cand_pool = torch.cat([c_mean, c_max], dim=-1) * has_cand.unsqueeze(-1).to(hc.dtype)

        has_hit = eff_hit_mask.any(dim=1)
        hh = self.spec_hit_in(eff_hits)
        hm = eff_hit_mask.unsqueeze(-1).to(hh.dtype)
        h_mean = (hh * hm).sum(dim=1) / eff_hit_mask.sum(dim=1, keepdim=True).clamp(min=1).to(hh.dtype)
        safe_hm = eff_hit_mask.clone()
        if (~has_hit).any():
            safe_hm[~has_hit, 0] = True
        h_w = torch.softmax(self.spec_hit_score(hh).squeeze(-1).float().masked_fill(~safe_hm, float('-inf')), dim=-1).to(hh.dtype)
        h_attn = (hh * h_w.unsqueeze(-1)).sum(dim=1)
        hit_pool = torch.cat([h_mean, h_attn], dim=-1) * has_hit.unsqueeze(-1).to(hh.dtype)
        spec_glob = self.spec_glob_fusion(torch.cat([cand_pool, hit_pool, ipt_glob.to(hc.dtype)], dim=-1)) * (
            (has_hit | has_cand | has_ipt).unsqueeze(-1).to(hc.dtype)
        )

        # Longitudinal 69-plane profile encoder (masked silicon slots carry ONLY (z_p, X0_p) + mask token!)
        z_in = torch.cat([z_pix_planes, cal_tok[:, :self.n_st]], dim=1)[:, self.z_order_idx]
        z_act = torch.cat([pix_plane_seq_act, st_mask], dim=1)[:, self.z_order_idx]
        z_in = (z_in + self.z_pos_proj(self.z_plane_geom).unsqueeze(0).to(z_in.dtype)) * z_act.unsqueeze(-1).to(z_in.dtype)
        z_seq = self.z_profile_encoder(z_in, z_act)

        with torch.amp.autocast(device_type=device.type, enabled=False):
            any_z = z_act.any(dim=1, keepdim=True)
            safe_za = z_act.clone()
            if (~any_z[:, 0]).any():
                safe_za[~any_z[:, 0], 0] = True
            z_logits = self.z_pool_gate(z_seq.float()).squeeze(-1).masked_fill(~safe_za, float('-inf'))
            z_w = torch.softmax(z_logits, dim=1) * z_act.float()
            z_prof_glob = (z_w.unsqueeze(-1) * z_seq.float()).sum(dim=1)

        # Cross-modal Perceiver fusion over UNMASKED observations only
        source = torch.cat([u_pix, z_pix_planes, cal_tok, spec_cand_tok], dim=1)
        source_mask = torch.cat([eff_pix_mask, pix_plane_act_real, cal_mask, eff_cand_mask], dim=1)
        if ssl_mode:
            q_ctx = self.ssl_query_token.expand(b, 1, -1).to(u_pix.dtype)
        else:
            q_ctx = self.cobs_query(batch['c_obs']).unsqueeze(1).to(u_pix.dtype)
        latents = torch.cat([self.latent_queries.expand(b, -1, -1).to(u_pix.dtype), q_ctx], dim=1)
        latent_mask = torch.ones((b, self.n_queries + 1), dtype=torch.bool, device=device)

        for blk in self.global_blocks:
            latents = blk(latents, source, source_mask, latent_mask)

        q_lat = latents[:, :self.n_queries]
        q_cobs = latents[:, self.n_queries]
        with torch.amp.autocast(device_type=device.type, enabled=False):
            w_pool = torch.softmax(self.pool_gate(q_lat.float()), dim=1)
            pool_attn = (w_pool * q_lat.float()).sum(dim=1)

        z_seq_pix = z_seq[:, self.pix_slots_in_z]  # (B, 15, dim)
        z_seq_cal = z_seq[:, self.cal_slots_in_z]  # (B, 42, dim)
        return {
            'u_pix': u_pix,
            'eff_pix_mask': eff_pix_mask,
            'z_pix_planes': z_pix_planes,
            'pix_plane_act_real': pix_plane_act_real,
            'z_seq': z_seq,
            'z_act': z_act,
            'z_seq_pix': z_seq_pix,
            'z_seq_cal': z_seq_cal,
            'z_prof_glob': z_prof_glob,
            'cal_out': bar_out,
            'ipt_glob': ipt_glob,
            'spec_glob': spec_glob.float(),
            'spec_cand_tok': spec_cand_tok,
            'eff_cand_mask': eff_cand_mask,
            'latents': latents,
            'pool_attn': pool_attn,
            'q_cobs': q_cobs.float(),
        }


class PinpointSSLPredictors(nn.Module):
    """Pretraining-only SSL predictors (discarded at fine-tuning & inference)."""

    def __init__(self, dim: int = 176, heads: int = 8) -> None:
        super().__init__()
        self.dim = dim
        # Task 1: Whole-Plane Silicon Drop JEPA + 8-GMM 2D Cluster Density
        self.pix_depth_proj = nn.Sequential(nn.Linear(2, dim), nn.LayerNorm(dim), nn.SiLU())
        self.pix_pred_cross = ScaledQKNormAttention(dim=dim, heads=heads, init_scale=10.0, drop_path=0.0)
        self.pix_pred_self = ScaledQKNormAttention(dim=dim, heads=heads, init_scale=10.0, drop_path=0.0)
        self.jepa_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.gmm_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, GMM_COMPONENTS * 5),  # [logit_pi, mu_x, mu_y, raw_sx, raw_sy]
        )

        # Task 2: Stereo Scintillator X/Y Panel & Relativistic ToF Cross-Prediction
        self.stereo_view_emb = nn.Parameter(torch.randn(1, 1, 2, dim) * 0.02)
        self.stereo_bar_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 40),
        )
        self.stereo_tof_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

        # Task 3: Cross-Magnet-0 Full Spectrometer Drop Pre-Calibration
        self.spec_head = nn.Sequential(
            nn.LayerNorm(4 * dim),
            nn.Linear(4 * dim, 2 * dim),
            nn.SiLU(),
            nn.Linear(2 * dim, 6),  # [mu_logit, x0_norm, y0_norm, tx_norm, ty_norm, ln_abs_sy]
        )


class PinpointSSLJEPA(NextGenFlavourModel):
    """PinpointSSL-JEPA: 3-task physics self-supervised architecture + layer-decayed fine-tune head."""

    def __init__(self, geometry: dict, config: dict | None = None) -> None:
        super().__init__()
        cfg = dict(config or {})
        self.config = cfg
        bounds = detector_bounds(geometry)
        stations = static_scint_stations(geometry, bounds)

        dim = int(cfg.get('dim', 176))
        heads = int(cfg.get('heads', 8))
        drop_path = float(cfg.get('drop_path', 0.10))
        head_drop = float(cfg.get('head_dropout', 0.15))
        n_rel_pre = int(cfg.get('n_rel_pre', 2))
        n_rel_post = int(cfg.get('n_rel_post', 2))

        self.dim = dim
        self.heads = heads
        self.max_tracklets = int(cfg.get('max_tracklets', M_TRK_MAX))
        self.x_pix_center = float(bounds['x_pix_center'])
        self.y_scint_center = float(bounds['y_scint_center'])
        self.z_mid = float(stations['z_mid'])
        self.register_buffer(
            'default_center_mm',
            torch.tensor([self.x_pix_center, self.y_scint_center, self.z_mid], dtype=torch.float32),
        )

        # Shared Pretrainable Backbone (~5.7M params)
        self.backbone = PinpointSSLBackbone(geometry, bounds, stations, cfg)

        # Level 4: Fine-Tuning Soft 3D Tracklet Relational R_ab + Student-t Billoir PV/SV + Heads (~2.0M params)
        self.cone_projector = TrackletCalorimeterConeProjector()
        self.trk_in = nn.Sequential(
            nn.Linear(TRK_KIN_DIM + TRK_MORPH_DIM + CONE_FEAT_DIM, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.pair_bias_mlp = nn.Sequential(
            nn.Linear(PAIR_TRK_FEAT_DIM, 32),
            nn.SiLU(),
            nn.Linear(32, heads),
        )
        self.rel_attn_pre = AttentionStack(
            dim=dim, n_layers=n_rel_pre, heads=heads, init_scale=12.0, max_scale=50.0, drop_path=drop_path
        )
        self.pv_anchor = PixelCalVertexAnchor(dim=dim, heads=heads)
        self.vertex_layer = DifferentiableStudentTVertexLayer(hidden_dim=dim, df=3.0, n_iters=4)
        self.pv_scale_cal = nn.Sequential(
            nn.LayerNorm(4 * dim + 16),
            nn.Linear(4 * dim + 16, 128),
            nn.SiLU(),
            nn.Linear(128, 3),
        )
        nn.init.zeros_(self.pv_scale_cal[-1].bias)

        self.ip_reinject = nn.Sequential(nn.Linear(5, dim), nn.LayerNorm(dim), nn.SiLU())
        self.ip_gate = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))
        self.rel_attn_post = AttentionStack(
            dim=dim, n_layers=n_rel_post, heads=heads, init_scale=12.0, max_scale=50.0, drop_path=drop_path
        )
        self.trk_pool_gate = nn.Linear(dim, 1)

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
        self.charm_bkg_head = nn.Sequential(
            nn.LayerNorm(dim + (dim // 2)),
            nn.Linear(dim + (dim // 2), 160),
            nn.SiLU(),
            nn.Dropout(head_drop),
            nn.Linear(160, 1),
        )
        self.tau_charm_head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )
        self.kin_head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, 160),
            nn.SiLU(),
            nn.Linear(160, 4),
        )

        # Pretraining-only modules (not counted in online+finetune parameter budget)
        self.include_ssl_heads = bool(cfg.get('include_ssl_heads', True))
        if self.include_ssl_heads:
            self.ssl_predictors: PinpointSSLPredictors | None = PinpointSSLPredictors(dim=dim, heads=heads)
        else:
            self.ssl_predictors = None

    def online_finetune_parameters(self) -> list[tuple[str, nn.Parameter]]:
        """Return `(name, param)` pairs for the online encoder + supervised fine-tuning head only."""
        return [
            (n, p)
            for n, p in self.named_parameters()
            if p.requires_grad and not n.startswith(('ssl_predictors.', 'ema_backbone.'))
        ]

    def online_finetune_param_count(self) -> int:
        return int(sum(p.numel() for _, p in self.online_finetune_parameters()))

    def _layer_level_for_param(self, name: str) -> int:
        """Map parameter name to depth level 0..4 for layer-wise LR decay (xi=0.78)."""
        if name.startswith((
            'backbone.patch_',
            'backbone.pix_in.',
            'backbone.pix_plane_emb.',
            'backbone.pix_plane_queries',
            'backbone.pix_plane_mask_token',
            'backbone.stereo_bars.xbar_enc.',
            'backbone.stereo_bars.ybar_enc.',
            'backbone.stereo_bars.stereo_2d_proj.',
            'backbone.stereo_bars.station_fuse.',
            'backbone.stereo_bars.nexus_in.',
            'backbone.stereo_bars.pair_mlp.',
            'backbone.spec_cand_in.',
            'backbone.spec_hit_',
        )):
            return 0
        if name.startswith((
            'backbone.pix_plane_cross.',
            'backbone.pix_plane_self.',
            'backbone.ipt_',
            'backbone.stereo_bars.bipartite_attn.',
            'backbone.spec_cand_attn',
            'backbone.spec_cand_to_dim.',
            'backbone.spec_glob_fusion.',
        )):
            return 1
        if name.startswith((
            'backbone.z_pos_proj.',
            'backbone.z_profile_encoder.',
            'backbone.z_pool_gate.',
        )):
            return 2
        if name.startswith((
            'backbone.latent_queries',
            'backbone.ssl_query_token',
            'backbone.cobs_query.',
            'backbone.global_blocks.',
            'backbone.pool_gate.',
        )):
            return 3
        return 4

    def build_param_groups(
        self,
        base_lr: float,
        weight_decay: float,
        *,
        xi: float = 0.78,
        include_ssl_heads: bool = False,
    ) -> list[dict]:
        """Build AdamW parameter groups with layer-wise LR decay `eta_l = base_lr * xi^(L - l)`."""
        max_level = 4
        buckets: dict[tuple[int, float], list[nn.Parameter]] = {}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if not include_ssl_heads and name.startswith('ssl_predictors.'):
                continue
            lvl = max_level if name.startswith('ssl_predictors.') else self._layer_level_for_param(name)
            lr_l = float(base_lr) * (float(xi) ** (max_level - lvl))
            wd_l = 0.0 if is_weight_decay_exempt(name, param) else float(weight_decay)
            buckets.setdefault((lvl, wd_l), []).append(param)

        groups: list[dict] = []
        for (lvl, wd_l), params in sorted(buckets.items()):
            lr_l = float(base_lr) * (float(xi) ** (max_level - lvl))
            groups.append({'params': params, 'lr': lr_l, 'weight_decay': wd_l})
        return groups

    def learned_attention_scales(self) -> dict[str, list[float]]:
        return collect_attention_scales(self)

    def branch_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        groups: dict[str, list[nn.Parameter]] = {}
        for name, param in self.online_finetune_parameters():
            if name.startswith(('backbone.patch_', 'backbone.pix_')):
                br = 'pixel_frontend'
            elif name.startswith('backbone.stereo_bars.'):
                br = 'stereo_bars'
            elif name.startswith(('backbone.spec_', 'backbone.ipt_')):
                br = 'spectrometer_ipt'
            elif name.startswith(('backbone.z_', 'backbone.latent_', 'backbone.cobs_', 'backbone.global_blocks', 'backbone.pool_gate')):
                br = 'longitudinal_perceiver'
            elif name.startswith(('cone_', 'trk_', 'pair_bias_mlp.', 'rel_attn_', 'ip_reinject.', 'ip_gate')):
                br = 'tracklet_relational'
            elif name.startswith(('pv_anchor.', 'vertex_layer.', 'pv_scale_cal.', 'vtx_summary_proj.')):
                br = 'vertex_layer'
            else:
                br = 'readout_heads'
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

    # ------------------------------------------------------------------
    # SSL Forward (NOMAE/Sonata-Strict Whole-Plane Masking + 3 Tasks)
    # ------------------------------------------------------------------
    def forward_ssl_student(
        self,
        batch: dict[str, torch.Tensor],
        *,
        pix_plane_drop_mask: torch.Tensor,
        stereo_panel_drop_mask: torch.Tensor,
        spec_drop_mask: torch.Tensor,
        pix_cluster_drop_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run SSL student encoder & predictors ONLY on masked inputs.

        Strictly obeys NOMAE/Sonata anti-shortcut rule: when `pix_cluster_drop_mask is None`,
        corrupting masked-plane `(x, y)` coordinates leaves all outputs bit-identical (`0.0`).
        """
        if self.ssl_predictors is None:
            raise RuntimeError('SSL predictors disabled (`include_ssl_heads=False`)')

        stu = self.backbone(
            batch,
            pix_plane_drop_mask=pix_plane_drop_mask,
            pix_cluster_drop_mask=pix_cluster_drop_mask,
            stereo_panel_drop_mask=stereo_panel_drop_mask,
            spec_drop_mask=spec_drop_mask,
            ssl_mode=True,
        )
        pred_mod = self.ssl_predictors

        # Task 1 predictor: query ONLY with longitudinal depth (z_p, X0_p) + student context
        depth_emb = pred_mod.pix_depth_proj(self.backbone.pix_plane_geom).unsqueeze(0).to(stu['z_seq_pix'].dtype)
        q_pix = stu['z_seq_pix'] + depth_emb
        ctx_tokens = torch.cat([stu['z_seq'], stu['latents']], dim=1)
        ctx_mask = torch.cat([
            stu['z_act'],
            torch.ones((q_pix.shape[0], stu['latents'].shape[1]), dtype=torch.bool, device=q_pix.device),
        ], dim=1)
        h_pix = pred_mod.pix_pred_cross(q_pix, ctx_tokens, ctx_mask)
        h_pix = pred_mod.pix_pred_self(h_pix, h_pix, torch.ones_like(pix_plane_drop_mask, dtype=torch.bool))
        z_pix_pred = pred_mod.jepa_head(h_pix)
        gmm_raw = pred_mod.gmm_head(h_pix).view(h_pix.shape[0], N_PIX_PLANES, GMM_COMPONENTS, 5)

        # Task 2 predictor: stereo scintillator X/Y bar profile + ToF
        st_ctx = stu['z_seq_cal'][:, :, None, :] + pred_mod.stereo_view_emb.to(stu['z_seq_cal'].dtype)
        bar_pred = pred_mod.stereo_bar_head(st_ctx)    # (B, 42, 2, 40)
        tof_pred = pred_mod.stereo_tof_head(st_ctx).squeeze(-1)  # (B, 42, 2)

        # Task 3 predictor: cross-Magnet-0 spectrometer presence + kinematics
        spec_in = torch.cat([
            stu['z_prof_glob'],
            stu['pool_attn'],
            stu['cal_out']['cal_global'].float(),
            stu['ipt_glob'],
        ], dim=-1)
        spec_pred = pred_mod.spec_head(spec_in.to(next(pred_mod.spec_head.parameters()).dtype))

        return {
            'z_stu_glob': stu['pool_attn'].float(),
            'z_pix_pred': z_pix_pred.float(),
            'gmm_raw': gmm_raw.float(),
            'bar_pred': bar_pred.float(),
            'tof_pred': tof_pred.float(),
            'spec_pred': spec_pred.float(),
        }

    # ------------------------------------------------------------------
    # Supervised Fine-Tuning & Pure-Observation Inference
    # ------------------------------------------------------------------
    def forward_NextGen(self, batch: dict[str, torch.Tensor]) -> NextGenModelOutput:
        b = int(batch['c_obs'].shape[0])
        device = batch['c_obs'].device

        if 'trk_kin' in batch and 'trk_morph' in batch and 'trk_mask' in batch:
            trk_kin = batch['trk_kin']
            trk_morph = batch['trk_morph']
            trk_mask = batch['trk_mask']
        else:
            ext = extract_soft_tracklets_batch(
                batch['pix_xyz_mm'],
                batch['pix_plane'],
                batch['pix_mask'],
                self.backbone.pix_plane_x0,
                batch['pix_feat'],
                max_tracklets=self.max_tracklets,
            )
            trk_kin = ext['trk_kin']
            trk_morph = ext['trk_morph']
            trk_mask = ext['trk_mask']

        is_ipt_stub = (trk_morph[..., 1] > 0.5) & trk_mask
        bb = self.backbone(
            batch,
            ssl_mode=False,
            ipt_kin_override=trk_kin,
            ipt_mask_override=is_ipt_stub,
        )

        cal_out = bb['cal_out']
        v_cal_prior, cal_valid = self._get_v_cal_prior(batch, b, device)

        cone_feat = self.cone_projector(
            trk_kin,
            trk_mask,
            cal_out['nex_xyz_mm'],
            cal_out['nex_e'],
            cal_out['nex_tof'],
            cal_out['nex_mask'],
        )
        trk_norm = self._normalize_trk_kin(trk_kin)
        trk_in_cat = torch.cat([trk_norm, trk_morph.float(), cone_feat], dim=-1)
        h_trk = self.trk_in(trk_in_cat.to(bb['u_pix'].dtype)) * trk_mask.unsqueeze(-1).to(bb['u_pix'].dtype)

        r_ab = pairwise_skew_dca_features(trk_kin, trk_mask)
        pair_bias = self.pair_bias_mlp(r_ab.to(bb['u_pix'].dtype)).permute(0, 3, 1, 2)
        h_trk = self.rel_attn_pre(h_trk, trk_mask, pair_bias=pair_bias)

        v_nn_mm, scale_nn_mm, h_vtx_ctx = self.pv_anchor(
            bb['z_seq'], bb['z_act'], v_cal_prior, bb['u_pix'],
            batch['pix_xyz_mm'], batch['pix_mask'], batch['pix_plane'],
        )
        vtx_out = self.vertex_layer(trk_kin, h_trk, trk_mask, v_cal_prior, cal_valid=cal_valid)
        with torch.amp.autocast(device_type=device.type, enabled=False):
            prec_nn = 1.0 / (scale_nn_mm.square() + 0.04)
            prec_bil = 1.0 / (vtx_out['scale_pv_mm'].square() + 0.04)
            prec_tot = prec_nn + prec_bil
            v_pv_mm = (prec_nn * v_nn_mm + prec_bil * vtx_out['v_pv_mm']) / prec_tot
            sigma_ivw = torch.rsqrt(prec_tot)
            trk_ip_feat = TrackletIPFormer._compute_ip_features(
                trk_kin, trk_mask, v_pv_mm, pi_pv=vtx_out['trk_ip_feat'][..., 3]
            )
        sv_disp_mm = vtx_out['sv_disp_mm']
        delta_chi2_2v = vtx_out['delta_chi2_2v']
        pi_sv = vtx_out['pi_sv']

        ip_in = torch.cat([trk_ip_feat, pi_sv.unsqueeze(-1)], dim=-1)
        h_trk = h_trk + self.ip_gate * self.ip_reinject(ip_in.to(h_trk.dtype)) * trk_mask.unsqueeze(-1).to(h_trk.dtype)
        h_trk = self.rel_attn_post(h_trk, trk_mask, pair_bias=pair_bias)
        h_trk = h_trk * trk_mask.unsqueeze(-1).to(h_trk.dtype)

        with torch.amp.autocast(device_type=device.type, enabled=False):
            any_trk = trk_mask.any(dim=1, keepdim=True)
            safe_tm = trk_mask.clone()
            if (~any_trk[:, 0]).any():
                safe_tm[~any_trk[:, 0], 0] = True
            t_logits = self.trk_pool_gate(h_trk.float()).squeeze(-1).masked_fill(~safe_tm, float('-inf'))
            t_w = torch.softmax(t_logits, dim=1) * trk_mask.float()
            trk_glob = (t_w.unsqueeze(-1) * h_trk.float()).sum(dim=1)

            pv_ctx = torch.cat([
                h_vtx_ctx,
                bb['z_prof_glob'],
                cal_out['cal_global'].float(),
                trk_glob,
                batch['c_obs'].float(),
            ], dim=-1)
            scale_pv_mm = (sigma_ivw * torch.exp(self.pv_scale_cal(pv_ctx).clamp(-2.0, 2.0))).clamp(0.02, 150.0)

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

            fused_global = self.global_fusion(
                torch.cat([
                    bb['pool_attn'],
                    bb['z_prof_glob'],
                    cal_out['cal_global'].float(),
                    trk_glob,
                    bb['spec_glob'],
                    bb['q_cobs'],
                    vtx_emb,
                ], dim=-1)
            )
            logits7 = self.event_head_7(fused_global)
            probs4 = marginal_four_from_seven(logits7, is_probs=False)

            z_bkg_c = self.charm_bkg_head(torch.cat([trk_glob, vtx_emb], dim=-1)).squeeze(-1)
            q_bkg_charm = torch.sigmoid(z_bkg_c).clamp(1e-6, 1.0 - 1e-6)
            z_tau_c = self.tau_charm_head(fused_global).squeeze(-1)
            p_tau_c = torch.sigmoid(z_tau_c)
            p_tau = probs4[:, 2].clamp(0.0, 1.0 - 1e-6)
            p_all_charm = ((1.0 - p_tau) * q_bkg_charm + p_tau * p_tau_c).clamp(1e-6, 1.0 - 1e-6)
            charm_logit = torch.log(p_all_charm) - torch.log1p(-p_all_charm)
            kin_pred = self.kin_head(fused_global)

        return NextGenModelOutput(
            logits7=logits7,
            probs4=probs4,
            charm_logit=charm_logit,
            v_pv_mm=v_pv_mm.float(),
            scale_pv_mm=scale_pv_mm.float(),
            sv_disp_mm=sv_disp_mm.float(),
            kin_pred=kin_pred.float(),
            aux={
                'fused_global': fused_global.detach(),
                'q_bkg_charm': q_bkg_charm,
                'p_all_charm': p_all_charm,
                'z_tau_c': z_tau_c,
                'delta_chi2_2v': delta_chi2_2v.float(),
                'p_sv_trklet': pi_sv.float(),
                'n_tracklets': trk_mask.sum(dim=1).float(),
            },
        )


# ===========================================================================
# SSL Masking, VICReg, 8-GMM & 3-Task Self-Supervised Loss
# ===========================================================================
def sample_ssl_masks(
    batch: dict[str, torch.Tensor],
    *,
    generator: torch.Generator | None = None,
    n_drop_planes_range: tuple[int, int] = (2, 3),
    stereo_drop_prob: float = 0.30,
    spec_drop_prob: float = 0.50,
    shortcut_point_mask: bool = False,
    n_st: int = 42,
    n_pix_planes: int = N_PIX_PLANES,
) -> dict[str, torch.Tensor | None]:
    """Sample structured physics masks for all 3 SSL tasks.

    By default (`shortcut_point_mask=False`), enforces whole-plane silicon masking
    (`pix_plane_drop_mask`). When `shortcut_point_mask=True` (negative-control ablation),
    masks random clusters within planes instead of whole planes.
    """
    device = batch['pix_mask'].device
    b = int(batch['pix_mask'].shape[0])
    n_cl = int(batch['pix_mask'].shape[1])
    pix_mask = batch['pix_mask'].bool()
    pix_plane = batch['pix_plane'].long().clamp(0, n_pix_planes - 1)

    pix_plane_drop = torch.zeros((b, n_pix_planes), dtype=torch.bool, device=device)
    pix_cluster_drop = torch.zeros((b, n_cl), dtype=torch.bool, device=device) if shortcut_point_mask else None

    if not shortcut_point_mask:
        for i in range(b):
            if not pix_mask[i].any():
                continue
            act_p = torch.unique(pix_plane[i][pix_mask[i]])
            n_act = int(act_p.numel())
            if n_act < 2:
                continue
            k_max = min(int(n_drop_planes_range[1]), n_act - 1)
            k_min = min(int(n_drop_planes_range[0]), k_max)
            k_drop = int(torch.randint(k_min, k_max + 1, (1,), generator=generator, device=device).item()) if k_max > k_min else k_min
            # Prefer interior planes (neither first nor last active) when enough active planes exist
            cand_p = act_p[1:-1] if (n_act - 2) >= k_drop else act_p
            perm = torch.randperm(int(cand_p.numel()), generator=generator, device=device)[:k_drop]
            pix_plane_drop[i, cand_p[perm]] = True
    else:
        assert pix_cluster_drop is not None
        r_cl = torch.rand((b, n_cl), generator=generator, device=device)
        pix_cluster_drop = pix_mask & (r_cl < 0.45)

    # Task 2: Stereo X/Y panel drop (mask X OR Y on ~30% of stations, never both on same station)
    r_st = torch.rand((b, int(n_st)), generator=generator, device=device)
    drop_st = r_st < float(stereo_drop_prob)
    choose_y = torch.rand((b, int(n_st)), generator=generator, device=device) < 0.5
    stereo_panel_drop = torch.zeros((b, int(n_st), 2), dtype=torch.bool, device=device)
    stereo_panel_drop[..., 0] = drop_st & (~choose_y)
    stereo_panel_drop[..., 1] = drop_st & choose_y

    # Task 3: Cross-Magnet-0 whole spectrometer drop on `spec_drop_prob` of events
    spec_drop = torch.rand((b,), generator=generator, device=device) < float(spec_drop_prob)
    if b > 0 and not spec_drop.any():
        spec_drop[0] = True

    return {
        'pix_plane_drop_mask': pix_plane_drop,
        'pix_cluster_drop_mask': pix_cluster_drop,
        'stereo_panel_drop_mask': stereo_panel_drop,
        'spec_drop_mask': spec_drop,
    }


def vicreg_variance_covariance_loss(
    z: torch.Tensor,
    *,
    gamma: float = 1.0,
    eps: float = 1e-4,
    var_weight: float = 25.0,
    cov_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """VICReg variance + off-diagonal covariance anti-collapse regularizer in FP32."""
    with torch.amp.autocast(device_type=z.device.type, enabled=False):
        zf = z.float()
        n, d = zf.shape
        if n < 2 or d < 2:
            zero = zf.sum() * 0.0
            return zero, {'std_mean': 0.0, 'var_loss': 0.0, 'cov_loss': 0.0}
        zc = zf - zf.mean(dim=0, keepdim=True)
        var = zc.var(dim=0, unbiased=False)
        std = torch.sqrt(var + eps)
        l_var = F.relu(float(gamma) - std).mean()
        cov = (zc.T @ zc) / float(n - 1)
        off_sq = cov.square().sum() - cov.diagonal().square().sum()
        l_cov = off_sq / float(d)
        loss = float(var_weight) * l_var + float(cov_weight) * l_cov
        return loss, {
            'std_mean': float(std.detach().mean().item()),
            'var_loss': float(l_var.detach().item()),
            'cov_loss': float(l_cov.detach().item()),
        }


def compute_ssl_pretrain_loss(
    student: PinpointSSLJEPA,
    teacher_backbone: PinpointSSLBackbone,
    batch: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor | None],
    *,
    w_jepa: float = 1.0,
    w_gmm: float = 0.25,
    w_stereo: float = 0.50,
    w_spec: float = 0.50,
    w_vicreg: float = 0.15,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute joint 3-task PinpointSSL-JEPA + VICReg self-supervised loss in FP32."""
    device = batch['c_obs'].device
    pix_plane_drop = masks['pix_plane_drop_mask']
    pix_cluster_drop = masks['pix_cluster_drop_mask']
    stereo_drop = masks['stereo_panel_drop_mask']
    spec_drop = masks['spec_drop_mask']
    assert pix_plane_drop is not None and stereo_drop is not None and spec_drop is not None

    with torch.no_grad():
        tea = teacher_backbone(batch, ssl_mode=True)
        tea_grids = teacher_backbone.stereo_bars.build_grids_and_targets(batch, stereo_panel_drop_mask=None)

    stu_out = student.forward_ssl_student(
        batch,
        pix_plane_drop_mask=pix_plane_drop,
        pix_cluster_drop_mask=pix_cluster_drop,
        stereo_panel_drop_mask=stereo_drop,
        spec_drop_mask=spec_drop,
    )

    with torch.amp.autocast(device_type=device.type, enabled=False):
        # ------------------------------------------------------------------
        # Task 1a: Whole-Plane Silicon Drop JEPA Latent Loss
        # ------------------------------------------------------------------
        z_pred = F.normalize(stu_out['z_pix_pred'].float(), p=2, dim=-1, eps=1e-6)
        z_tea = F.normalize(tea['z_seq_pix'].detach().float(), p=2, dim=-1, eps=1e-6)
        jepa_plane_err = (z_pred - z_tea).square().sum(dim=-1)  # (B, 15)

        if pix_cluster_drop is not None:
            # Negative control ablation: supervise planes that had dropped clusters
            pix_plane_all = batch['pix_plane'].long().clamp(0, N_PIX_PLANES - 1)
            sup_p_mask = torch.zeros_like(pix_plane_drop)
            for p in range(N_PIX_PLANES):
                sup_p_mask[:, p] = (pix_cluster_drop & (pix_plane_all == p)).any(dim=1)
        else:
            sup_p_mask = pix_plane_drop.bool() & batch['pix_mask'].any(dim=1, keepdim=True)

        if sup_p_mask.any():
            l_jepa = jepa_plane_err[sup_p_mask].mean()
        else:
            l_jepa = z_pred.sum() * 0.0

        # ------------------------------------------------------------------
        # Task 1b: 8-Component 2D Gaussian Mixture NLL over Masked Clusters (x, y)
        # ------------------------------------------------------------------
        gmm = stu_out['gmm_raw'].float()  # (B, 15, 8, 5)
        log_pi = F.log_softmax(gmm[..., 0], dim=-1)  # (B, 15, 8)
        mu_xy = 1.2 * torch.tanh(gmm[..., 1:3])      # (B, 15, 8, 2)
        sig_xy = F.softplus(gmm[..., 3:5]).clamp(min=0.005, max=1.5)  # (B, 15, 8, 2)

        cl_mask = batch['pix_mask'].bool()
        cl_plane = batch['pix_plane'].long().clamp(0, N_PIX_PLANES - 1)
        if pix_cluster_drop is not None:
            cl_sup = pix_cluster_drop.bool()
        else:
            cl_sup = cl_mask & torch.gather(pix_plane_drop.bool(), 1, cl_plane)

        if cl_sup.any():
            b_idx, c_idx = torch.nonzero(cl_sup, as_tuple=True)
            p_idx = cl_plane[b_idx, c_idx]
            xy_tgt = torch.stack([
                (batch['pix_xyz_mm'][b_idx, c_idx, 0].float() - student.x_pix_center) / 135.0,
                (batch['pix_xyz_mm'][b_idx, c_idx, 1].float() - student.y_scint_center) / 100.0,
            ], dim=-1).clamp(-1.8, 1.8)  # (N_c, 2)

            lp = log_pi[b_idx, p_idx]          # (N_c, 8)
            mu = mu_xy[b_idx, p_idx]           # (N_c, 8, 2)
            sg = sig_xy[b_idx, p_idx]          # (N_c, 8, 2)
            diff = (xy_tgt[:, None, :] - mu) / sg
            log_gauss = -0.5 * (diff.square().sum(dim=-1)) - torch.log(sg).sum(dim=-1) - math.log(2.0 * math.pi)
            nll_cl = -torch.logsumexp(lp + log_gauss, dim=-1)  # (N_c,)

            # Plane-balanced mean so dense EM showers don't drown single-prong tracks
            flat_pid = b_idx * N_PIX_PLANES + p_idx
            uniq_p, inv_p = torch.unique(flat_pid, return_inverse=True)
            p_sum = torch.zeros(uniq_p.shape[0], dtype=torch.float32, device=device)
            p_cnt = torch.zeros(uniq_p.shape[0], dtype=torch.float32, device=device)
            p_sum.index_add_(0, inv_p, nll_cl)
            p_cnt.index_add_(0, inv_p, torch.ones_like(nll_cl))
            l_gmm = (p_sum / p_cnt.clamp_min(1.0)).mean()
        else:
            l_gmm = gmm.sum() * 0.0

        # ------------------------------------------------------------------
        # Task 2: Stereo Scintillator X/Y Panel & Relativistic ToF Cross-Prediction
        # ------------------------------------------------------------------
        gx = tea_grids['grid_x'].float()
        gy = tea_grids['grid_y'].float()
        b_sz = gx.shape[0]
        n_st = gx.shape[2]
        dbl_st = student.backbone.stereo_bars.doublet_stations

        tgt_e = torch.zeros((b_sz, n_st, 2, 40), dtype=torch.float32, device=device)
        tgt_act = torch.zeros((b_sz, n_st, 2, 40), dtype=torch.float32, device=device)
        tgt_tof = torch.zeros((b_sz, n_st, 2), dtype=torch.float32, device=device)
        tgt_view_act = torch.zeros((b_sz, n_st, 2), dtype=torch.bool, device=device)

        tgt_e[:, :, 0] = gx[:, 0]
        tgt_act[:, :, 0] = gx[:, 4]
        wx_t = gx[:, 0] + 1e-4 * gx[:, 4]
        tgt_view_act[:, :, 0] = gx[:, 4].any(dim=-1)
        tgt_tof[:, :, 0] = torch.where(
            tgt_view_act[:, :, 0],
            (wx_t * gx[:, 6]).sum(dim=-1) / wx_t.sum(dim=-1).clamp_min(1e-6),
            torch.zeros((b_sz, n_st), device=device),
        )

        tgt_e[:, dbl_st, 1] = gy[:, 0]
        tgt_act[:, dbl_st, 1] = gy[:, 4]
        wy_t = gy[:, 0] + 1e-4 * gy[:, 4]
        y_act_dbl = gy[:, 4].any(dim=-1)
        tgt_view_act[:, dbl_st, 1] = y_act_dbl
        tgt_tof[:, dbl_st, 1] = torch.where(
            y_act_dbl,
            (wy_t * gy[:, 6]).sum(dim=-1) / wy_t.sum(dim=-1).clamp_min(1e-6),
            torch.zeros((b_sz, len(dbl_st)), device=device),
        )

        valid_dbl_mask = torch.zeros((b_sz, n_st, 2), dtype=torch.bool, device=device)
        valid_dbl_mask[:, :, 0] = True
        valid_dbl_mask[:, dbl_st, 1] = True
        st_drop_sup = stereo_drop.bool() & valid_dbl_mask & tgt_view_act

        if st_drop_sup.any():
            pred_bp = stu_out['bar_pred'][st_drop_sup]
            tgt_bp = tgt_e[st_drop_sup]
            act_bp = tgt_act[st_drop_sup] > 0.5
            err_bp = F.smooth_l1_loss(pred_bp, tgt_bp, beta=0.1, reduction='none')
            l_bar_all = err_bp.mean()
            l_bar_act = err_bp[act_bp].mean() if act_bp.any() else l_bar_all
            l_tof = F.smooth_l1_loss(stu_out['tof_pred'][st_drop_sup], tgt_tof[st_drop_sup], beta=0.25)
            l_stereo = 0.5 * l_bar_all + 0.5 * l_bar_act + 0.5 * l_tof
        else:
            l_stereo = stu_out['bar_pred'].sum() * 0.0 + stu_out['tof_pred'].sum() * 0.0

        # ------------------------------------------------------------------
        # Task 3: Cross-Magnet-0 Full Spectrometer Drop Pre-Calibration
        # ------------------------------------------------------------------
        sp_pred = stu_out['spec_pred']  # (B, 6)
        tc = batch['trk_cand'].float()
        tcm = batch['trk_cand_mask'].bool()
        has_3st_bore = tcm[:, 0] & (tc[:, 0, 0] > 0.75)

        sd_mask = spec_drop.bool()
        if w_spec > 0.0 and sd_mask.any():
            y_mu = has_3st_bore.float()
            l_mu_bce = F.binary_cross_entropy_with_logits(sp_pred[sd_mask, 0], y_mu[sd_mask])
            pos_sp = sd_mask & has_3st_bore
            if pos_sp.any():
                sy_mm = SIGMA_SAGITTA_Y_MM * torch.sinh((5.0 * tc[pos_sp, 0, 6]).clamp(-9.0, 9.0))
                ln_abs_sy = torch.log(sy_mm.abs() + 0.05)
                tgt_kin = torch.stack([
                    tc[pos_sp, 0, 2],
                    tc[pos_sp, 0, 3],
                    tc[pos_sp, 0, 4],
                    tc[pos_sp, 0, 5],
                    ln_abs_sy,
                ], dim=-1)
                l_mu_kin = F.smooth_l1_loss(sp_pred[pos_sp, 1:6], tgt_kin, beta=0.5)
            else:
                l_mu_kin = sp_pred.sum() * 0.0
            l_spec = l_mu_bce + 0.5 * l_mu_kin
        else:
            l_spec = sp_pred.sum() * 0.0

        # ------------------------------------------------------------------
        # Anti-Collapse: VICReg Variance + Covariance Regularizer
        # ------------------------------------------------------------------
        l_vic_glob, vic_g_stats = vicreg_variance_covariance_loss(stu_out['z_stu_glob'])
        if sup_p_mask.any() and int(sup_p_mask.sum().item()) >= 4:
            l_vic_pix, vic_p_stats = vicreg_variance_covariance_loss(stu_out['z_pix_pred'][sup_p_mask])
            l_vic = 0.5 * (l_vic_glob + l_vic_pix)
        else:
            l_vic = l_vic_glob
            vic_p_stats = {'std_mean': 0.0}

        total = (
            float(w_jepa) * l_jepa
            + float(w_gmm) * l_gmm
            + float(w_stereo) * l_stereo
            + float(w_spec) * l_spec
            + float(w_vicreg) * l_vic
        )
        metrics = {
            'ssl_total': float(total.detach().item()),
            'ssl_jepa': float(l_jepa.detach().item()),
            'ssl_gmm': float(l_gmm.detach().item()),
            'ssl_stereo': float(l_stereo.detach().item()),
            'ssl_spec': float(l_spec.detach().item()),
            'ssl_vicreg': float(l_vic.detach().item()),
            'vic_glob_std': vic_g_stats['std_mean'],
            'vic_pix_std': vic_p_stats['std_mean'],
        }
        return total, metrics


def compute_jepa_finetune_window_denominators(
    micro_batches: list[dict],
    pos_weight: float = 3.0,
) -> dict[str, float]:
    """Exact full-window per-task denominators across microbatches for grad_accum equivalence."""
    d_proc = 0.0
    d_charm_nontau = 0.0
    d_charm_all = 0.0
    d_vtx = 0.0
    d_kin = 0.0
    for mb in micro_batches:
        lbl4 = mb['label'].long()
        valid_p = (mb['label_7'] >= 0) | ((mb['label_7'] < 0) & (lbl4 == 2)) if 'label_7' in mb else torch.ones_like(lbl4, dtype=torch.bool)
        d_proc += float(valid_p.float().sum().item())

        if 'charm_valid' in mb and 'charm_flag' in mb:
            vc = mb['charm_valid'].long() > 0
            vc_all = vc & (lbl4 <= 2)
            if vc_all.any():
                yc = mb['charm_flag'][vc_all].float()
                d_charm_all += float(torch.where(yc > 0.5, float(pos_weight), 1.0).sum().item())
            vc_nt = vc & (lbl4 < 2)
            if vc_nt.any():
                yc_nt = mb['charm_flag'][vc_nt].float()
                d_charm_nontau += float(torch.where(yc_nt > 0.5, float(pos_weight), 1.0).sum().item())
        if 'vertex_valid' in mb:
            d_vtx += float((mb['vertex_valid'] > 0).float().sum().item())
        if 'kin_valid' in mb:
            d_kin += float((mb['kin_valid'] > 0).float().sum().item())
    return {
        'proc': max(d_proc, 1e-8),
        'charm_nontau': max(d_charm_nontau, 1e-8),
        'charm_all': max(d_charm_all, 1e-8),
        'vtx': max(d_vtx, 1e-8),
        'kin': max(d_kin, 1e-8),
    }


def compute_jepa_finetune_loss(
    model: PinpointSSLJEPA,
    batch: dict[str, torch.Tensor],
    *,
    fixed_denoms: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Supervised fine-tuning loss: 7->4 marginal CE + conflict-free charm + Student-t(nu=3) PV + kin."""
    fd = fixed_denoms or {}
    out = model.forward_NextGen(batch)

    if 'label_7' in batch:
        l_proc = marginal_tau_branch_ce(
            out.logits7,
            batch['label_7'],
            batch['label'],
            denom_override=fd.get('proc'),
        )
    else:
        per_p = F.nll_loss(torch.log(out.probs4.float().clamp_min(1e-7)), batch['label'].long(), reduction='none')
        denom_p = torch.as_tensor(fd.get('proc', float(len(per_p))), dtype=torch.float32, device=per_p.device).clamp_min(1e-8)
        l_proc = per_p.sum() / denom_p

    if 'charm_valid' in batch and 'charm_flag' in batch:
        cv = batch['charm_valid'].long() > 0
        cf = batch['charm_flag'].float()
        lbl4 = batch['label'].long()
        l_c_nt = _focal_bce_from_prob(out.aux['q_bkg_charm'], cf, cv & (lbl4 < 2), denom_override=fd.get('charm_nontau'))
        l_c_all = _focal_bce_from_prob(out.aux['p_all_charm'], cf, cv & (lbl4 <= 2), denom_override=fd.get('charm_all'))
        l_charm = 0.5 * l_c_nt + 0.5 * l_c_all
    else:
        l_charm = out.logits7.sum() * 0.0

    if 'vertex_valid' in batch and 'vertex_xyz_mm' in batch:
        l_vtx = student_t_vertex_loss(
            out.v_pv_mm,
            out.scale_pv_mm,
            batch['vertex_xyz_mm'],
            batch['vertex_valid'],
            df=3.0,
            denom_override=fd.get('vtx'),
        )
    else:
        l_vtx = out.v_pv_mm.sum() * 0.0

    if 'kin_valid' in batch and 'vis_kinematics' in batch and out.kin_pred is not None:
        kv = batch['kin_valid'] > 0
        if kv.any():
            with torch.amp.autocast(device_type=out.logits7.device.type, enabled=False):
                k_ev = F.smooth_l1_loss(
                    out.kin_pred[kv].float(),
                    batch['vis_kinematics'][kv].float(),
                    reduction='none',
                ).mean(dim=-1)
                dk = torch.as_tensor(fd.get('kin', float(kv.sum())), dtype=torch.float32, device=k_ev.device).clamp_min(1e-8)
                l_kin = k_ev.sum() / dk
        else:
            l_kin = out.logits7.sum() * 0.0
    else:
        l_kin = out.logits7.sum() * 0.0

    total = l_proc.float() + 0.25 * l_charm.float() + 0.25 * l_vtx.float() + 0.15 * l_kin.float()
    return total, out.probs4
