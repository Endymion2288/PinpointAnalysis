"""PinpointUnifiedNet: Single-Backbone Unified Next-Generation PINPOINT Architecture (~7.4M params).

Unites the orthogonal validated mechanisms from Stage 326:
  1. Non-exclusive soft 3D micro-tracklets (Planes 0..11, M_T <= 64) with 4x4 Highland MCS
     covariance, shared-root flag, upstream-stub vs neutral-gap flag, forward X0 cone continuation
     onto de-ghosted calorimeter crossings, and FP64->FP32 vertex-free skew-line DCA bias R_ab;
  2. Four-regime detector factorization:
     - Regime 1 (Front 30mm, Planes 0..5): cross-plane persistence gate suppressing PixDark
       + causal Front30 Student-t PV/SV + symmetric two-stage cluster refiner;
     - Regime 2 (Mid 150mm, Fortune 0..6 + Planes 6..11): 42-station dual-active Sinkhorn X-Y
       de-ghosting + per-event relativistic bar ToF + 69-layer longitudinal X0 profile +
       Mid150 Student-t PV + symmetric two-stage refiner;
     - Regime 3 & 4 (IPT Planes 12..14 <-> Spectrometer): Magnet-0 4D trajectory interlock
       chi2_IPT_Spec + `spec_rel_head` trained with 20% online bore-checked (`R_fid=96 mm`,
       `x_pix_center=-0.0048 mm`, `y_scint_center=-12.0 mm`) combinatorial fake-triplet &
       PixDark curriculum;
  3. Regime-gated (`front_gate`) inverse-covariance PV fusion decoupling 30mm vs 150mm priors,
     followed by covariance-normalized signed IP re-injection & 2-layer post-PV relational attn;
  4. Displacement-anchored recoil hemisphere C_hem = -cos(dphi_disp_had) & cone isolation;
  5. Precision-gated Perceiver cross-modal fusion (`ScaledQKNorm + ln(1+M)`), `_safe_hypot`,
     7-class -> 4-class exact marginalization, and conflict-free charm decomposition.

Strictly observation-only during inference; satisfies NextGenFlavourModel / Gate A-F contracts.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from analysis.flavour.nextgen.dual_regime_modalnet import (
    K_FRONT_MAX,
    K_IPT_MAX,
    K_MID_MAX,
    M_TRK_MAX,
    N_SPEC_CAND_MAX,
    SINKHORN_K_TOP,
    SINKHORN_N_NEXUS,
    _SymmetricTwoStageVertexRefiner,
    _select_masked_tokens,
    extract_nonexclusive_spec_candidates,
)
from analysis.flavour.nextgen.protocols import (
    NextGenFlavourModel,
    NextGenModelOutput,
    marginal_four_from_seven,
    student_t_vertex_loss,
)
from analysis.flavour.nextgen.tau_kinema_diffnet import (
    DisplacementAnchoredHemisphereModule,
    _focal_bce_from_prob,
)
from analysis.flavour.nextgen.tracklet_ipformer import (
    CONE_FEAT_DIM,
    TrackletCalorimeterConeProjector,
    TrackletIPFormer,
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
from analysis.flavour.v3.common.domains import cumulative_radiation_length, detector_bounds
from analysis.flavour.v3.common.layers import ScaledQKNormAttention, split_adamw_param_groups
from analysis.flavour.v3.v1_b1pp.models import TransverseLayerEncoder
from analysis.flavour.v3.v1_b1pp.spectrometer import (
    MAGNET0_LENGTH_MM,
    MAGNET0_Z_CENTER_MM,
    MAGNET0_Z_MAX_MM,
    MAGNET0_Z_MIN_MM,
    MAGNET1_Z_MAX_MM,
    MAGNET1_Z_MIN_MM,
    MAGNET2_Z_MAX_MM,
    MAGNET2_Z_MIN_MM,
    MAGNET_BORE_FID_R_MM,
    MAGNET_BORE_X_MM,
    MAGNET_BORE_Y_MM,
    SAGITTA_CONST_MEV_MM,
    SAGITTA_INTEGRAL_MM2,
    SIGMA_DELTA_X_MM,
    SIGMA_SAGITTA_Y_MM,
    TRACKER_Z_MM,
    DELTA_Z_MM,
    ENTRY_SLOPE_SAGITTA_COEFF,
)
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
    extract_pf_item,
)
from analysis.flavour.v3.v2_pinpointformer.stereo_bars import static_scint_stations
from analysis.flavour.v3.v3_multitask.losses import marginal_tau_branch_ce


HEM_FEAT_DIM = 8
LAW2_STUB_DIM = 3


def extract_unified_item(
    index: int,
    observation: dict,
    label: int | None,
    dom: dict,
    geometry: dict,
    bounds: dict,
    config: dict | None = None,
) -> dict:
    """Pure-observation item extractor with pre-tracklet monopixel dark-hit filtering."""
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

    pxyz = item['pix_xyz_mm']
    ppl = item['pix_plane']
    pf = item['pix_feat']
    n_cl = len(pxyz)
    if n_cl > 0:
        keep_trk_cl = (
            (pf[:, 3] >= 0.15)
            | (pf[:, 21] >= 0.05)
            | (pf[:, 13] >= 0.05)
        )
        # Check tight dp in {1, 2} MIP partner coincidence for residual isolated monopixels
        need_chk = np.flatnonzero(~keep_trk_cl)
        if len(need_chk) > 0:
            x_all = pxyz[:, 0].astype(np.float64)
            y_all = pxyz[:, 1].astype(np.float64)
            z_all = pxyz[:, 2].astype(np.float64)
            pl_all = ppl.astype(np.int64)
            is_ipt = pl_all >= 12
            for idx_i in need_chk:
                pl_i = pl_all[idx_i]
                dp = np.abs(pl_all - pl_i)
                cand_j = ((dp == 1) | (dp == 2)) & (is_ipt == is_ipt[idx_i])
                if not cand_j.any():
                    continue
                dz = np.abs(z_all[cand_j] - z_all[idx_i])
                r_perp = np.hypot(x_all[cand_j] - x_all[idx_i], y_all[cand_j] - y_all[idx_i])
                r_tol = 0.025 * dz + 0.08
                if (r_perp <= r_tol).any():
                    keep_trk_cl[idx_i] = True

        trk_res = extract_soft_tracklets_single(
            pxyz[keep_trk_cl],
            ppl[keep_trk_cl],
            pf[keep_trk_cl],
            plane_x0,
            max_tracklets=M_TRK_MAX,
            pitch_xy_mm=pitch_xy,
        )
    else:
        trk_res = extract_soft_tracklets_single(
            pxyz, ppl, pf, plane_x0, max_tracklets=M_TRK_MAX, pitch_xy_mm=pitch_xy,
        )

    item['trk_kin'] = trk_res['trk_kin']
    item['trk_morph'] = trk_res['trk_morph']
    item['trk_mask'] = trk_res['trk_mask']
    return item


class PinpointUnifiedNet(NextGenFlavourModel):
    """Single-backbone unified PINPOINT neutrino flavour & vertex network (~7.4M params)."""

    def __init__(self, geometry: dict, config: dict | None = None) -> None:
        super().__init__()
        cfg = dict(config or {})
        self.config = cfg

        dim = int(cfg.get('dim', 160))
        heads = int(cfg.get('heads', 8))
        drop_path = float(cfg.get('drop_path', 0.12))
        head_drop = float(cfg.get('head_dropout', 0.18))
        n_front_layers = int(cfg.get('n_front_layers', 2))
        n_z_layers = int(cfg.get('n_z_layers', 2))
        n_rel_pre = int(cfg.get('n_rel_pre', 2))
        n_rel_post = int(cfg.get('n_rel_post', 2))
        n_global_layers = int(cfg.get('n_global_layers', 2))

        self.dim = dim
        self.heads = heads
        self.n_queries = int(cfg.get('n_queries', 16))
        self.max_tracklets = int(cfg.get('max_tracklets', M_TRK_MAX))

        # Single-factor causal ablation toggles (all True in full PinpointUnifiedNet)
        self.use_recoil_hem = bool(cfg.get('use_recoil_hem', True))
        self.enable_online_spec_curriculum = bool(cfg.get('enable_online_spec_curriculum', True))
        self.use_tracklet_rab = bool(cfg.get('use_tracklet_rab', True))
        self.aug_prob = float(cfg.get('degradation_aug_prob', 0.20 if self.enable_online_spec_curriculum else 0.0))

        bounds = detector_bounds(geometry)
        stations = static_scint_stations(geometry, bounds)
        self.n_st = int(stations['n_stations'])
        self.n_dbl = int(stations['n_doublets'])
        self.n_bars = int(stations['n_bars'])
        self.x_pix_c = float(bounds['x_pix_center'])
        self.x_scint_c = float(stations['x_scint_center'])
        self.y_c = float(stations['y_scint_center'])
        self.z_mid = float(stations['z_mid'])
        self.z_target_min = float(bounds['z_target_min'])
        self.l_target = float(bounds['L_target'])
        z_w0 = float(bounds.get('z_w0', self.z_target_min + 2.5))
        self.scint_p0_upstream = float(bounds['scint_Zpos'][0]) < z_w0
        self.pix_p0_upstream = float(bounds['pixel_Zpos'][0]) < z_w0

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
        self.register_buffer(
            'ybar_geom',
            torch.stack([z_norm[self.doublet_stations], x0_norm[self.doublet_stations]], dim=0),
        )

        # ------------------------------------------------------------------
        # 1. Shared Silicon Micro-Morphology Front-End
        # ------------------------------------------------------------------
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
        # 2. Regime 2: 42-Station Dual-Active Sinkhorn + Bar Convs + Relativistic ToF
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

        # ------------------------------------------------------------------
        # 3. Longitudinal 69-Plane Profile Encoder
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
            dim=dim, n_layers=n_z_layers, heads=heads, init_scale=10.0, max_scale=50.0, drop_path=drop_path
        )
        self.z_pool_gate = nn.Linear(dim, 1)

        # ------------------------------------------------------------------
        # 4. Regime 1 (Front 30mm) & Regime 2 Mid Silicon (Planes 6..11)
        # ------------------------------------------------------------------
        self.pixdark_gate_mlp = nn.Sequential(
            nn.Linear(12, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        # Prior: isolated 1-hit monopixels without cone coincidence start suppressed (< 0.45) at t=0.
        nn.init.constant_(self.pixdark_gate_mlp[-1].bias, -0.8)
        self.front_pair_mlp = nn.Sequential(
            nn.Linear(6, 32),
            nn.SiLU(),
            nn.Linear(32, heads),
        )
        self.front_encoder = AttentionStack(
            dim=dim, n_layers=n_front_layers, heads=heads, init_scale=12.0, max_scale=50.0, drop_path=drop_path
        )
        self.front_bridge_queries = nn.Parameter(torch.randn(1, 2, dim) * 0.02)
        self.front_bridge_attn = ScaledQKNormAttention(
            dim=dim, heads=heads, init_scale=10.0, drop_path=drop_path
        )
        self.front_pool_gate = nn.Linear(dim, 1)

        self.mid_pix_cross = ScaledQKNormAttention(
            dim=dim, heads=heads, init_scale=10.0, drop_path=drop_path
        )
        self.mid_pix_self = ScaledQKNormAttention(
            dim=dim, heads=heads, init_scale=10.0, drop_path=drop_path
        )
        self.mid_pool_gate = nn.Linear(dim, 1)

        # ------------------------------------------------------------------
        # 5. Soft Multi-Tracklets + X0 Cone + Skew-DCA R_ab + Regime-Gated Student-t PV/SV
        # ------------------------------------------------------------------
        self.cone_projector = TrackletCalorimeterConeProjector()
        self.trk_in = nn.Sequential(
            nn.Linear(TRK_KIN_DIM + TRK_MORPH_DIM + CONE_FEAT_DIM, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        if self.use_tracklet_rab:
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

        self.trk_front_cross = ScaledQKNormAttention(
            dim=dim, heads=heads, init_scale=10.0, drop_path=drop_path
        )
        self.front_vtx_layer = DifferentiableStudentTVertexLayer(hidden_dim=dim, df=3.0, n_iters=4)
        self.front_vtx_refine = _SymmetricTwoStageVertexRefiner(
            tok_dim=dim, vtx_dim=80, heads=4, coarse_xz_mm=(80.0, 140.0), fine_xz_mm=(20.0, 35.0)
        )
        self.mid_vtx_layer = DifferentiableStudentTVertexLayer(hidden_dim=dim, df=3.0, n_iters=4)
        self.mid_vtx_head = _SymmetricTwoStageVertexRefiner(
            tok_dim=dim, vtx_dim=80, heads=4, coarse_xz_mm=(150.0, 320.0), fine_xz_mm=(25.0, 65.0)
        )
        self.register_buffer('pv_regime_scale_corr', torch.ones((3, 3, 3), dtype=torch.float32))

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

        # ------------------------------------------------------------------
        # 6. Regime 3 & 4: IPT (Planes 12..14) <-> Magnet-0 Spectrometer Interlock
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
        # 7. Displacement-Anchored Recoil Hemisphere & Vertex Summary
        # ------------------------------------------------------------------
        self.vtx_summary_proj = nn.Sequential(
            nn.Linear(9, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.SiLU(),
        )
        if self.use_recoil_hem:
            self.hemisphere_mod: DisplacementAnchoredHemisphereModule | None = DisplacementAnchoredHemisphereModule(
                geometry, bounds, stations
            )
            self.hem_proj: nn.Module | None = nn.Sequential(
                nn.Linear(HEM_FEAT_DIM + LAW2_STUB_DIM, dim // 2),
                nn.LayerNorm(dim // 2),
                nn.SiLU(),
            )
            hem_emb_dim = dim // 2
        else:
            self.hemisphere_mod = None
            self.hem_proj = None
            hem_emb_dim = 0

        # ------------------------------------------------------------------
        # 8. Precision-Gated Cross-Modal Perceiver Fusion & Conflict-Free Heads
        # ------------------------------------------------------------------
        self.latent_queries = nn.Parameter(torch.randn(1, self.n_queries, dim) * 0.02)
        self.cobs_query = nn.Sequential(
            nn.Dropout(0.30),
            nn.Linear(16, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
        )
        self.ugca_beta = nn.Parameter(torch.full((1, heads, 1, 1), 0.35))
        self.global_blocks = nn.ModuleList([
            PerceiverCrossBlock(dim=dim, heads=heads, init_scale=12.0, max_scale=50.0, drop_path=drop_path)
            for _ in range(n_global_layers)
        ])
        self.pool_gate = nn.Linear(dim, 1)

        fused_in = 9 * dim + (dim // 2) + hem_emb_dim
        self.global_fusion = nn.Sequential(
            nn.LayerNorm(fused_in),
            nn.Linear(fused_in, 3 * dim),
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
        # Conflict-free lepton-blind CC charm decomposition: q_bkg_charm on e_CC/mu_CC + p_tau_c on tau
        charm_bkg_in = 2 * dim
        self.charm_bkg_head = nn.Sequential(
            nn.LayerNorm(charm_bkg_in),
            nn.Linear(charm_bkg_in, 160),
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
            if name.startswith(('patch_', 'pix_in.', 'pix_plane_emb.')):
                br = 'pixel_frontend'
            elif name.startswith(('pixdark_', 'front_')):
                br = 'regime1_front_micro'
            elif name.startswith((
                'xbar_enc.', 'ybar_enc.', 'stereo_2d_', 'station_fuse.', 'sinkhorn_',
                'nexus_in.', 'cal_pair_', 'bipartite_', 'mid_', 'pv_',
            )):
                br = 'regime2_mid_sinkhorn'
            elif name.startswith(('trk_', 'pair_bias_mlp.', 'rel_attn_', 'ip_reinject.', 'ip_gate')):
                br = 'tracklet_relational'
            elif name.startswith(('ipt_', 'spec_')):
                br = 'regime34_ipt_spec'
            elif name.startswith(('hemisphere_mod.', 'hem_proj.', 'vtx_summary_proj.')):
                br = 'recoil_hemisphere'
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
        cal_start: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """20% online bore-checked fake-triplet/pair & LogUniform(4,96) PixDark curriculum (training-only)."""
        b, n_cl, _ = pix_xyz.shape
        device = pix_xyz.device
        cand_injected_noise = torch.zeros_like(trk_cand_mask, dtype=torch.bool)
        apply_ev = torch.rand((b,), device=device) < self.aug_prob
        if not apply_ev.any():
            pix_injected_dark = torch.zeros_like(pix_mask, dtype=torch.bool)
            return (
                apply_ev, pix_xyz, pix_feat, pix_plane, pix_mask, pix_injected_dark,
                trk_cand, trk_cand_mask, trk_hits, trk_hit_mask, station_counts, cand_injected_noise,
            )

        # Pad pixel tensors by +96 slots so LogUniform(4, 96) dark hits never collide with real clusters
        max_extra = 96
        pix_xyz = F.pad(pix_xyz.clone(), (0, 0, 0, max_extra), value=0.0)
        pix_feat = F.pad(pix_feat.clone(), (0, 0, 0, max_extra), value=0.0)
        pix_plane = F.pad(pix_plane.clone(), (0, max_extra), value=0)
        pix_mask = F.pad(pix_mask.clone(), (0, max_extra), value=False)
        pix_injected_dark = torch.zeros_like(pix_mask, dtype=torch.bool)

        trk_cand = trk_cand.clone()
        trk_cand_mask = trk_cand_mask.clone()
        trk_hits = trk_hits.clone()
        trk_hit_mask = trk_hit_mask.clone()
        station_counts = station_counts.clone()

        # 1. Mild 2.5% pixel cluster dropout + LogUniform(4, 96) isolated PixDark monopixels
        ev_mask_pix = apply_ev.unsqueeze(1) & pix_mask
        drop_pix = ev_mask_pix & (torch.rand_like(pix_mask.float()) < 0.025)
        orig_cnt = pix_mask.sum(dim=1, keepdim=True)
        would_empty = (pix_mask & (~drop_pix)).sum(dim=1, keepdim=True) == 0
        drop_pix = drop_pix & (~(would_empty & (orig_cnt > 0)))
        pix_mask = pix_mask & (~drop_pix)

        k_dark = max(4, min(max_extra, int(torch.exp(torch.empty((), device=device).uniform_(math.log(4.0), math.log(96.0))).item())))
        idx_aug = torch.nonzero(apply_ev, as_tuple=False).squeeze(-1)
        n_aug = len(idx_aug)
        pl_rand = torch.randint(0, 15, (n_aug, k_dark), device=device)
        xr = torch.empty((n_aug, k_dark), device=device).uniform_(-125.0, 125.0)
        yr = self.y_c + torch.empty((n_aug, k_dark), device=device).uniform_(-90.0, 90.0)
        zr = self.pix_plane_z[pl_rand]
        sl_end = n_cl + k_dark
        pix_xyz[idx_aug, n_cl:sl_end] = torch.stack([xr, yr, zr], dim=-1)
        pix_plane[idx_aug, n_cl:sl_end] = pl_rand
        dark_feat = torch.zeros((n_aug, k_dark, PIX_FEAT_DIM), device=device, dtype=pix_feat.dtype)
        dark_feat[..., 0] = (xr - self.x_pix_c) / 135.0
        dark_feat[..., 1] = (yr - self.y_c) / 100.0
        dark_feat[..., 2] = (zr - self.z_mid) / 1000.0
        dark_feat[..., 3] = math.log1p(1.0) / 6.0
        dark_feat[..., 6] = 1.0
        dark_feat[..., 7] = math.asinh(0.006 / 0.05) / 3.0
        dark_feat[..., 8] = math.asinh(0.006 / 0.05) / 3.0
        pix_feat[idx_aug, n_cl:sl_end] = dark_feat
        pix_mask[idx_aug, n_cl:sl_end] = True
        pix_injected_dark[idx_aug, n_cl:sl_end] = True

        # 2. Mild 2.5% real tracker hit dropout + 1 random noise hit per station (matching spectrometer.py:140-145)
        orig_hm = trk_hit_mask.clone()
        drop_h = apply_ev.unsqueeze(1) & trk_hit_mask & (torch.rand(trk_hit_mask.shape, device=device) < 0.025)
        orig_hcnt = trk_hit_mask.sum(dim=1, keepdim=True)
        empty_h = (trk_hit_mask & (~drop_h)).sum(dim=1, keepdim=True) == 0
        trk_hit_mask = trk_hit_mask & (~(drop_h & (~(empty_h & (orig_hcnt > 0)))))

        noise_xy = torch.zeros((b, 3, 2), device=device, dtype=torch.float32)
        for st in (0, 1, 2):
            station_counts[:, st] = station_counts[:, st] + apply_ev.float()
            hx_all = torch.empty((b,), device=device).uniform_(-120.0, 120.0)
            hy_all = torch.empty((b,), device=device).uniform_(-110.0, 140.0)
            noise_xy[:, st, 0] = hx_all
            noise_xy[:, st, 1] = hy_all
            has_h_free = apply_ev & (~trk_hit_mask).any(dim=1)
            if has_h_free.any():
                h_pos = (~trk_hit_mask).long().argmax(dim=1)
                idx_b = torch.nonzero(has_h_free, as_tuple=False).squeeze(-1)
                pos_b = h_pos[idx_b]
                z_trk_st = float(TRACKER_Z_MM[st])
                cs_h = cal_start[idx_b].float()
                has_st_h = cs_h[:, 3] > 0.5
                dz_h = (z_trk_st - cs_h[:, 0]).clamp_min(100.0)
                noise_hit = torch.zeros((len(idx_b), 6), device=device, dtype=trk_hits.dtype)
                noise_hit[:, 0] = float(st) / 2.0
                noise_hit[:, 1] = (hx_all[idx_b] - self.x_pix_c) / 150.0
                noise_hit[:, 2] = (hy_all[idx_b] - self.y_c) / 150.0
                noise_hit[:, 3] = torch.where(
                    has_st_h, ((hx_all[idx_b] - cs_h[:, 1]) / dz_h).clamp(-2.0, 2.0) * 10.0, torch.zeros_like(dz_h)
                )
                noise_hit[:, 4] = torch.where(
                    has_st_h, ((hy_all[idx_b] - cs_h[:, 2]) / dz_h).clamp(-2.0, 2.0) * 10.0, torch.zeros_like(dz_h)
                )
                noise_hit[:, 5] = has_st_h.to(trk_hits.dtype)
                trk_hits[idx_b, pos_b] = noise_hit
                trk_hit_mask[idx_b, pos_b] = True

        # 3. Real candidate dropout (p=0.05) + fake bore-checked 3-station triplet + fake 2-station pair + re-sort
        drop_c = apply_ev.unsqueeze(1) & trk_cand_mask & (torch.rand(trk_cand_mask.shape, device=device) < 0.05)
        trk_cand_mask = trk_cand_mask & (~drop_c)
        trk_cand = trk_cand * trk_cand_mask.unsqueeze(-1).to(trk_cand.dtype)

        has_c_free = apply_ev & (~trk_cand_mask).any(dim=1)
        if has_c_free.any():
            c_pos = (~trk_cand_mask).long().argmax(dim=1)
            idx_b = torch.nonzero(has_c_free, as_tuple=False).squeeze(-1)
            pos_b = c_pos[idx_b]
            n_c = len(idx_b)

            st_orig = torch.round(trk_hits[idx_b, :, 0].float() * 2.0).long()
            m_orig = orig_hm[idx_b]
            trip_x = noise_xy[idx_b, :, 0].clone()
            trip_y = noise_xy[idx_b, :, 1].clone()
            forced_noise_st = torch.randint(0, 3, (n_c,), device=device)
            for st in (0, 1, 2):
                m_st = m_orig & (st_orig == st)
                use_real = m_st.any(dim=1) & (forced_noise_st != st) & (torch.rand((n_c,), device=device) < 0.65)
                if use_real.any():
                    first_r = m_st.long().argmax(dim=1)
                    rx = trk_hits[idx_b, first_r, 1].float() * 150.0 + self.x_pix_c
                    ry = trk_hits[idx_b, first_r, 2].float() * 150.0 + self.y_c
                    trip_x[:, st] = torch.where(use_real, rx, trip_x[:, st])
                    trip_y[:, st] = torch.where(use_real, ry, trip_y[:, st])

            x0, x1, x2 = trip_x[:, 0], trip_x[:, 1], trip_x[:, 2]
            y0, y1, y2 = trip_y[:, 0], trip_y[:, 1], trip_y[:, 2]
            s_y = y1 - 0.5 * (y0 + y2)
            kappa = -s_y / float(SAGITTA_INTEGRAL_MM2)
            delta_x = x1 - 0.5 * (x0 + x2)
            chi2_x = (delta_x / float(SIGMA_DELTA_X_MM)).square()
            theta_x = (x2 - x0) / (2.0 * float(DELTA_Z_MM))
            theta_y_entry = (y1 - y0) / float(DELTA_Z_MM) + float(ENTRY_SLOPE_SAGITTA_COEFF) * (s_y / float(DELTA_Z_MM))

            z0_s, z1_s = float(TRACKER_Z_MM[0]), float(TRACKER_Z_MM[1])
            theta_y1 = theta_y_entry + kappa * 1000.0
            max_r = torch.zeros_like(x0)
            for zs, ys in (
                (float(MAGNET1_Z_MIN_MM), y0 + theta_y_entry * (float(MAGNET1_Z_MIN_MM) - z0_s)),
                (637.4, y0 + theta_y_entry * (637.4 - z0_s) + 0.5 * kappa * (500.0 ** 2)),
                (float(MAGNET1_Z_MAX_MM), y0 + theta_y_entry * (float(MAGNET1_Z_MAX_MM) - z0_s) + 0.5 * kappa * (1000.0 ** 2)),
                (float(MAGNET2_Z_MIN_MM), y1 + theta_y1 * (float(MAGNET2_Z_MIN_MM) - z1_s)),
                (1837.4, y1 + theta_y1 * (1837.4 - z1_s) + 0.5 * kappa * (500.0 ** 2)),
                (float(MAGNET2_Z_MAX_MM), y1 + theta_y1 * (float(MAGNET2_Z_MAX_MM) - z1_s) + 0.5 * kappa * (1000.0 ** 2)),
            ):
                xs = x0 + theta_x * (zs - z0_s)
                rs = _safe_hypot(xs - float(MAGNET_BORE_X_MM), ys - float(MAGNET_BORE_Y_MM), eps=1e-10)
                max_r = torch.maximum(max_r, rs)
            in_bore = max_r <= float(MAGNET_BORE_FID_R_MM)
            kappa_eff = torch.where(in_bore, kappa, torch.zeros_like(kappa))

            cs_sub = cal_start[idx_b].float()
            z_st, x_st, y_st = cs_sub[:, 0], cs_sub[:, 1], cs_sub[:, 2]
            has_st = cs_sub[:, 3] > 0.5
            dz_ext = z_st - z0_s
            dx_ext = torch.where(has_st, (x0 + theta_x * dz_ext) - x_st, torch.zeros_like(x0))
            y_lin = y0 + theta_y_entry * dz_ext
            y_up = y_lin - kappa_eff * float(MAGNET0_LENGTH_MM) * (z_st - float(MAGNET0_Z_CENTER_MM))
            u_m0 = (float(MAGNET0_Z_MAX_MM) - z_st).clamp_min(0.0)
            y_mid = y_lin + 0.5 * kappa_eff * u_m0.square()
            y_ext = torch.where(
                z_st >= float(MAGNET0_Z_MAX_MM),
                y_lin,
                torch.where(z_st <= float(MAGNET0_Z_MIN_MM), y_up, y_mid),
            )
            dy_ext = torch.where(has_st, y_ext - y_st, torch.zeros_like(y0))
            d_ext = torch.where(has_st, _safe_hypot(dx_ext / 20.0, dy_ext / 20.0, eps=1e-10), torch.zeros_like(x0))

            bore_pen = torch.where(in_bore, torch.zeros_like(x0), torch.full_like(x0, 25.0))
            q_score = chi2_x + bore_pen + torch.where(has_st, 0.5 * d_ext.square(), torch.zeros_like(x0)) + (s_y.abs() / 20.0).square()
            sy_ib = torch.where(in_bore, s_y, torch.zeros_like(s_y))
            p_proxy = torch.where(in_bore, 1.0 / (sy_ib.abs() / 0.5 + 0.01), torch.zeros_like(s_y))

            cand_noise = torch.stack([
                torch.where(in_bore, torch.ones_like(x0), torch.full_like(x0, 0.70)),
                1.0 / (1.0 + 0.1 * q_score),
                (x0 - self.x_pix_c) / 150.0,
                (y0 - self.y_c) / 150.0,
                theta_x * 10.0,
                theta_y_entry * 10.0,
                torch.where(in_bore, torch.asinh(sy_ib / float(SIGMA_SAGITTA_Y_MM)) / 5.0, torch.zeros_like(x0)),
                torch.where(in_bore, sy_ib.clamp(-25.0, 25.0) / 5.0, torch.zeros_like(x0)),
                torch.where(in_bore, torch.log1p(p_proxy) / 5.0, torch.zeros_like(x0)),
                torch.log1p(chi2_x) / 5.0,
                dx_ext.clamp(-300.0, 300.0) / 50.0,
                dy_ext.clamp(-300.0, 300.0) / 50.0,
                torch.log1p(d_ext) / 3.0,
                torch.where(in_bore, torch.log1p(sy_ib.abs() / 0.05) / 5.0, torch.zeros_like(x0)),
            ], dim=-1).to(trk_cand.dtype)
            trk_cand[idx_b, pos_b] = cand_noise
            trk_cand_mask[idx_b, pos_b] = True
            cand_injected_noise[idx_b, pos_b] = True

            # Inject fake 2-station pair (sa=0, sb=2) when a second free slot exists
            has_c2_free = apply_ev & (~trk_cand_mask).any(dim=1)
            if has_c2_free.any():
                c2_pos = (~trk_cand_mask).long().argmax(dim=1)
                idx_2 = torch.nonzero(has_c2_free, as_tuple=False).squeeze(-1)
                pos_2 = c2_pos[idx_2]
                x0_2, y0_2 = noise_xy[idx_2, 0, 0], noise_xy[idx_2, 0, 1]
                x2_2, y2_2 = noise_xy[idx_2, 2, 0], noise_xy[idx_2, 2, 1]
                dz_02 = float(TRACKER_Z_MM[2] - TRACKER_Z_MM[0])
                tx_2 = (x2_2 - x0_2) / dz_02
                ty_2 = (y2_2 - y0_2) / dz_02
                cs_2 = cal_start[idx_2].float()
                has_st2 = cs_2[:, 3] > 0.5
                dz_ext2 = cs_2[:, 0] - z0_s
                dxe_2 = torch.where(has_st2, (x0_2 + tx_2 * dz_ext2) - cs_2[:, 1], torch.zeros_like(x0_2))
                dye_2 = torch.where(has_st2, (y0_2 + ty_2 * dz_ext2) - cs_2[:, 2], torch.zeros_like(y0_2))
                de_2 = torch.where(has_st2, _safe_hypot(dxe_2 / 20.0, dye_2 / 20.0, eps=1e-10), torch.zeros_like(x0_2))
                q2 = 10.0 + torch.where(has_st2, 0.5 * de_2.square(), torch.zeros_like(x0_2)) + (tx_2 * 10.0).square() + (ty_2 * 10.0).square()
                cand_p2 = torch.stack([
                    torch.full_like(x0_2, 0.5),
                    1.0 / (1.0 + 0.1 * q2),
                    (x0_2 - self.x_pix_c) / 150.0,
                    (y0_2 - self.y_c) / 150.0,
                    tx_2 * 10.0,
                    ty_2 * 10.0,
                    torch.zeros_like(x0_2),
                    torch.zeros_like(x0_2),
                    torch.zeros_like(x0_2),
                    torch.ones_like(x0_2),
                    dxe_2.clamp(-300.0, 300.0) / 50.0,
                    dye_2.clamp(-300.0, 300.0) / 50.0,
                    torch.log1p(de_2) / 3.0,
                    torch.zeros_like(x0_2),
                ], dim=-1).to(trk_cand.dtype)
                trk_cand[idx_2, pos_2] = cand_p2
                trk_cand_mask[idx_2, pos_2] = True
                cand_injected_noise[idx_2, pos_2] = True

        # Re-sort candidates descending by (2*(col0>0.6) + col1) so fake triplets can hijack slot 0
        sort_key = trk_cand_mask.float() * (2.0 * (trk_cand[..., 0] > 0.6).float() + trk_cand[..., 1].clamp_min(0.0))
        sort_idx = torch.argsort(sort_key, dim=1, descending=True)
        trk_cand = torch.gather(trk_cand, 1, sort_idx.unsqueeze(-1).expand(-1, -1, trk_cand.shape[-1]))
        trk_cand_mask = torch.gather(trk_cand_mask, 1, sort_idx)
        cand_injected_noise = torch.gather(cand_injected_noise, 1, sort_idx)

        return (
            apply_ev, pix_xyz, pix_feat, pix_plane, pix_mask, pix_injected_dark,
            trk_cand, trk_cand_mask, trk_hits, trk_hit_mask, station_counts, cand_injected_noise,
        )

    def _compute_persistence_gate(
        self,
        p_xyz: torch.Tensor,
        p_feat: torch.Tensor,
        p_plane: torch.Tensor,
        p_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute all-plane tight-cone MIP persistence gate g_persist in (0, 1) and logits."""
        if p_xyz.shape[1] == 0:
            z0 = p_mask.float()
            return z0, torch.zeros_like(z0)
        with torch.amp.autocast(device_type=p_xyz.device.type, enabled=False):
            xyz = p_xyz.float()
            dp = torch.abs(p_plane[:, :, None] - p_plane[:, None, :])
            dz = torch.abs(xyz[:, :, None, 2] - xyz[:, None, :, 2]).clamp_min(1.0)
            dx = xyz[:, :, None, 0] - xyz[:, None, :, 0]
            dy = xyz[:, :, None, 1] - xyz[:, None, :, 1]
            d_perp_sq = dx.square() + dy.square()
            is_ipt = p_plane >= 12
            same_block = (is_ipt[:, :, None] == is_ipt[:, None, :])
            pair_ok = p_mask[:, :, None] & p_mask[:, None, :] & same_block

            sig1_sq = (0.035 * dz + 0.10).square()
            sig2_sq = (0.045 * dz + 0.15).square()
            w1 = torch.exp(-0.5 * (d_perp_sq / sig1_sq).clamp_max(30.0)) * ((dp == 1) & pair_ok).float()
            w2 = torch.exp(-0.5 * (d_perp_sq / sig2_sq).clamp_max(30.0)) * ((dp == 2) & pair_ok).float()
            cone1 = torch.log1p(w1.sum(dim=-1)) / 2.0
            cone2 = torch.log1p(w2.sum(dim=-1)) / 2.0

            ff = p_feat.float()
            sel_cols = ff[..., [3, 7, 8, 9, 13, 14, 15, 19, 20, 21]]
            gate_in = torch.cat([cone1.unsqueeze(-1), cone2.unsqueeze(-1), sel_cols], dim=-1)
            logits = self.pixdark_gate_mlp(gate_in).squeeze(-1)
            g_persist = torch.sigmoid(logits) * p_mask.float()
        return g_persist, logits

    def _encode_mid_sinkhorn_tomography(
        self,
        batch: dict[str, torch.Tensor],
        pix_xyz: torch.Tensor,
        pix_plane: torch.Tensor,
        pix_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """42-station dual-active Sinkhorn X-Y de-ghosting + per-event relativistic ToF."""
        b = int(batch['c_obs'].shape[0])
        device = batch['c_obs'].device
        bar_ev = batch['bar_ev']
        bar_panel = batch['bar_panel']
        bar_horiz = batch['bar_horiz']
        bar_idx = batch['bar_idx']
        bar_val = batch['bar_val'].float()

        bt_key = 'bar_time_ns' if 'bar_time_ns' in batch else ('bar_hit_time_ns' if 'bar_hit_time_ns' in batch else None)
        if bt_key is not None and len(bar_ev) > 0:
            bar_z = self.station_z_mm[self.panel_to_station[bar_panel]]
            bt = batch[bt_key].float()
            bar_tof = compute_scint_tof_residual_ns(bt, bar_z, valid_mask=(bt >= 0.0), event_idx=bar_ev)
        else:
            bar_tof = bar_val.new_zeros((len(bar_ev),))

        grid_x = torch.zeros((b, 7, self.n_st, self.n_bars), device=device, dtype=torch.float32)
        grid_y = torch.zeros((b, 7, self.n_dbl, self.n_bars), device=device, dtype=torch.float32)

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

        grid_x_e = torch.expm1((10.0 * grid_x[:, 0]).clamp(0.0, 14.0)) * grid_x[:, 4]
        grid_y_e = torch.expm1((10.0 * grid_y[:, 0]).clamp(0.0, 14.0)) * grid_y[:, 4]

        x_act = grid_x[:, 4] > 0.5
        y_act = grid_y[:, 4] > 0.5
        x_st_act = x_act.any(dim=-1)
        y_st_act = y_act.any(dim=-1)
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
            e_x_st, e_y_st, asym, x_st_act.float(), y_st_act_full.float(), tof_x_st, tof_y_st,
        ], dim=-1)
        st_tok = self.station_fuse(torch.cat([x_tok, y_full, st_aux], dim=-1))
        st_mask = x_st_act | y_st_act_full
        st_tok = st_tok * st_mask.unsqueeze(-1).to(st_tok.dtype)
        st_xyz = torch.stack([x_cent, y_cent, self.station_z_mm[None, :].expand(b, -1)], dim=-1).float()

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

            # Dual-active station guard: (sum_E_X > 0) & (sum_E_Y > 0) + clamp_min(1e-8)
            sum_ex = ex_f.sum(dim=-1, keepdim=True)
            sum_ey = ey_f.sum(dim=-1, keepdim=True)
            m_dual = ((sum_ex > 0.0) & (sum_ey > 0.0)).squeeze(-1)
            p_s = ex_f / sum_ex.clamp_min(1e-8)
            q_s = ey_f / sum_ey.clamp_min(1e-8)

            x_cand = self.bar_x_mm[top_ix]
            y_cand = self.bar_y_mm[top_iy]
            z_dbl = self.station_z_mm[self.doublet_stations]
            mid_si_m = pix_mask & (pix_plane >= 6) & (pix_plane <= 11)
            if mid_si_m.any():
                px_f = pix_xyz[:, :, 0].float()
                py_f = pix_xyz[:, :, 1].float()
                pz_f = pix_xyz[:, :, 2].float()
                dz_ps = torch.abs(z_dbl[None, :, None] - pz_f[:, None, :]) <= 180.0
                w_si = (mid_si_m[:, None, :] & dz_ps).float()
                dx_p = x_cand[:, :, :, None] - px_f[:, None, None, :]
                dy_p = y_cand[:, :, :, None] - py_f[:, None, None, :]
                gx_p = (torch.exp(-0.5 * (dx_p / 35.0).square().clamp_max(20.0)) * w_si[:, :, None, :]).sum(dim=-1)
                gy_p = (torch.exp(-0.5 * (dy_p / 35.0).square().clamp_max(20.0)) * w_si[:, :, None, :]).sum(dim=-1)
                has_mid_near = w_si.sum(dim=-1) > 0.5
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
            k_mat = torch.exp((-cost_shift / 0.5).clamp_min(-40.0)) * pair_valid.float()

            u_vec = p_s.clone()
            v_vec = q_s.clone()
            for _ in range(3):
                kv = torch.matmul(k_mat, v_vec.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8)
                u_vec = p_s / kv
                ktu = torch.matmul(k_mat.transpose(-2, -1), u_vec.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8)
                v_vec = q_s / ktu

            gamma = m_dual[:, :, None, None].float() * (u_vec[:, :, :, None] * k_mat * v_vec[:, :, None, :])
            pair_score = torch.where(
                pair_valid & m_dual[:, :, None, None],
                gamma * (0.5 * (sum_ex[:, :, :, None] + sum_ey[:, :, None, :])) + 1e-4,
                torch.full_like(gamma, -1e6),
            ).view(b, -1)
            gamma_flat = gamma.view(b, -1)
            prior_flat = pi_prior.clamp_min(1e-3).log().view(b, -1)

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
            mean_tof = 0.5 * (tx_val + ty_val)
            abs_dt = torch.abs(tx_val - ty_val)

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
        nex_e = 0.5 * (ex_val + ey_val)

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
            'nex_xyz_mm': nex_xyz,
            'nex_e': nex_e,
            'nex_tof': mean_tof,
            'nex_mask': nex_mask,
            'grid_x_e': grid_x_e,
            'grid_y_e': grid_y_e,
            'n_dual_active': (x_st_act[:, self.doublet_stations] & y_st_act).sum(dim=-1),
        }

    def _normalize_trk_kin(self, trk_kin: torch.Tensor) -> torch.Tensor:
        k = trk_kin.float()
        return torch.stack([
            (k[..., 0] - self.x_pix_c) / 135.0,
            (k[..., 1] - self.y_c) / 100.0,
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
        cal_start = batch['cal_start'].float()

        # Training-only 20% online bore-checked fake-triplet/pair & LogUniform(4,96) PixDark curriculum
        aug_ev = torch.zeros((b,), dtype=torch.bool, device=device)
        pix_injected_dark = torch.zeros_like(pix_mask, dtype=torch.bool)
        cand_injected_noise = torch.zeros_like(trk_cand_mask, dtype=torch.bool)
        if self.training and self.enable_online_spec_curriculum and self.aug_prob > 0.0:
            (
                aug_ev, pix_xyz, pix_feat, pix_plane, pix_mask, pix_injected_dark,
                trk_cand, trk_cand_mask, trk_hits, trk_hit_mask, station_counts, cand_injected_noise,
            ) = self._apply_training_degradation_aug(
                pix_xyz, pix_feat, pix_plane, pix_mask,
                trk_cand, trk_cand_mask, trk_hits, trk_hit_mask, station_counts, cal_start,
            )

        # All-plane tight-cone MIP persistence gate (Fix R2-2)
        g_persist_all, pixdark_logit_all = self._compute_persistence_gate(pix_xyz, pix_feat, pix_plane, pix_mask)
        pm_ver = pix_mask & ((g_persist_all.detach() > 0.45) | (pix_feat[..., 3] >= 0.15))
        is_mono_iso = pix_mask & (pix_feat[..., 3] < 0.15) & (pix_feat[..., 21] < 0.05) & (g_persist_all.detach() < 0.45)
        mono_plane_cnt = (F.one_hot(pix_plane.clamp(0, 14), 15) * is_mono_iso.unsqueeze(-1).long()).sum(dim=1)
        is_flood = (mono_plane_cnt >= 3).sum(dim=1) >= 8

        # 1. Shared Silicon Micro-Cluster + 32x32 Micro-Patch Embeddings
        u_pix = self.pix_in(pix_feat) * pix_mask.unsqueeze(-1).to(pix_feat.dtype)
        if 'pix_patches' in batch and len(batch['pix_patches']) > 0:
            patch_emb = self.patch_proj(self.patch_conv(batch['pix_patches'].float()))
            delta_patch = torch.zeros_like(u_pix, dtype=patch_emb.dtype)
            delta_patch.index_put_((batch['patch_ev'], batch['patch_cl']), patch_emb, accumulate=True)
            u_pix = u_pix + delta_patch * pix_mask.unsqueeze(-1).to(u_pix.dtype)

        pps_raw = batch['pix_plane_summary'].float()
        if is_flood.any():
            with torch.amp.autocast(device_type=device.type, enabled=False):
                oh_p = F.one_hot(pix_plane.clamp(0, 14), 15).float() * pm_ver.unsqueeze(-1).float()
                n_hits_cl = torch.expm1(6.0 * pix_feat[..., 3].float().clamp_min(0.0)).clamp_min(1.0)
                cnt_r040_cl = torch.expm1(4.0 * pix_feat[..., 21].float().clamp_min(0.0)).clamp_min(0.0)
                p_cl_cnt = oh_p.sum(dim=1)
                p_act_c = p_cl_cnt > 0.5
                p_hit_cnt = (oh_p * n_hits_cl.unsqueeze(-1)).sum(dim=1)
                w_hx = oh_p * (n_hits_cl * pix_xyz[..., 0].float()).unsqueeze(-1)
                w_hy = oh_p * (n_hits_cl * pix_xyz[..., 1].float()).unsqueeze(-1)
                p_x_mean = w_hx.sum(dim=1) / p_hit_cnt.clamp_min(1.0)
                p_y_mean = w_hy.sum(dim=1) / p_hit_cnt.clamp_min(1.0)
                w_core = (n_hits_cl + cnt_r040_cl).square()
                oh_w = oh_p * w_core.unsqueeze(-1)
                p_wsum = oh_w.sum(dim=1).clamp_min(1e-12)
                plane_xc = (oh_w * pix_xyz[..., 0].float().unsqueeze(-1)).sum(dim=1) / p_wsum
                plane_yc = (oh_w * pix_xyz[..., 1].float().unsqueeze(-1)).sum(dim=1) / p_wsum
                pps_clean = torch.zeros_like(pps_raw)
                pps_clean[:, :, 0] = p_act_c.float()
                pps_clean[:, :, 1] = (torch.log1p(p_hit_cnt) / 8.0) * p_act_c.float()
                pps_clean[:, :, 2] = (torch.log1p(p_cl_cnt) / 6.0) * p_act_c.float()
                pps_clean[:, :, 3] = ((p_x_mean - self.x_pix_c) / 135.0) * p_act_c.float()
                pps_clean[:, :, 4] = ((p_y_mean - self.y_c) / 100.0) * p_act_c.float()
                pps_clean[:, :, 6] = ((plane_yc - self.y_c) / 100.0) * p_act_c.float()
                pps_clean[:, :, 7] = ((plane_xc - self.x_pix_c) / 135.0) * p_act_c.float()
                pps_eff = torch.where(is_flood[:, None, None], pps_clean, pps_raw)
        else:
            pps_eff = pps_raw
        pix_plane_tok = self.pix_plane_emb(pps_eff)
        pix_plane_act = pps_eff[..., 0] > 0.5

        # Soft non-exclusive tracklets (M_T <= 64)
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
                pix_xyz, pix_plane, pix_mask, self.pix_plane_x0, pix_feat=pix_feat, max_tracklets=self.max_tracklets
            )
            trk_kin = trk_dict['trk_kin']
            trk_morph = trk_dict['trk_morph']
            trk_mask = trk_dict['trk_mask']

        # Calibrated calorimeter & AVR priors
        cal_valid = cal_start[:, 3] > 0.5
        cal_xyz_prior = torch.stack([cal_start[:, 1], cal_start[:, 2], cal_start[:, 0]], dim=-1)
        default_c = self.default_center_mm[None, :].expand(b, -1)
        v_cal_prior = torch.where(cal_valid[:, None], cal_xyz_prior, default_c)
        avr_pv = (
            batch['avr_pv_mm'].float()
            if 'avr_pv_mm' in batch
            else torch.zeros((b, 4), device=device, dtype=torch.float32)
        )
        avr_has = avr_pv[:, 3] > 0.5

        # 2. Regime 2: 42-Station Dual-Active Sinkhorn + ToF + 69-Layer Longitudinal X0 Profile
        cal_out = self._encode_mid_sinkhorn_tomography(batch, pix_xyz, pix_plane, pix_mask)
        cal_tok = cal_out['cal_tokens']
        cal_mask = cal_out['cal_mask']

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

        # 3. Tracklet Forward X0 Cone Continuation & Vertex-Free FP64->FP32 Skew-Line DCA Bias R_ab
        cone_feat = self.cone_projector(
            trk_kin,
            trk_mask,
            cal_out['nex_xyz_mm'],
            cal_out['nex_e'],
            cal_out['nex_tof'],
            cal_out['nex_mask'],
        )
        trk_norm = self._normalize_trk_kin(trk_kin)
        h_trk = self.trk_in(torch.cat([trk_norm, trk_morph, cone_feat], dim=-1).to(u_pix.dtype))
        h_trk = h_trk * trk_mask.unsqueeze(-1).to(h_trk.dtype)

        if self.pair_bias_mlp is not None:
            r_ab = pairwise_skew_dca_features(trk_kin, trk_mask)
            pair_bias = self.pair_bias_mlp(r_ab.to(h_trk.dtype)).permute(0, 3, 1, 2)
        else:
            pair_bias = None
        h_trk = self.rel_attn_pre(h_trk, trk_mask, pair_bias=pair_bias)

        # 4. Regime 1 (Front 30mm, Planes 0..5) & Regime 2 Mid Silicon (Planes 6..11)
        tok_rank = (
            pix_feat[..., 3]
            + 0.25 * pix_feat[..., 13]
            + 2.0 * torch.log(g_persist_all.detach().clamp_min(1e-3))
            + 1e-4 * (1.0 - torch.abs(pix_feat[..., 0]).clamp_max(1.0))
        )

        # --- Regime 1: Front 30mm Micro-Tracker ---
        front_raw_m = pix_mask & (pix_plane <= 5)
        f_idx, f_mask = _select_masked_tokens(tok_rank, front_raw_m, K_FRONT_MAX)
        f_xyz = torch.gather(pix_xyz, 1, f_idx.unsqueeze(-1).expand(-1, -1, 3))
        f_feat = torch.gather(pix_feat, 1, f_idx.unsqueeze(-1).expand(-1, -1, PIX_FEAT_DIM))
        f_pl = torch.gather(pix_plane, 1, f_idx)
        f_x0 = self.pix_plane_x0[f_pl]
        z_front = torch.gather(u_pix, 1, f_idx.unsqueeze(-1).expand(-1, -1, self.dim))

        g_persist = torch.gather(g_persist_all, 1, f_idx) * f_mask.float()
        z_front = z_front * (0.25 + 0.75 * g_persist).unsqueeze(-1)

        with torch.amp.autocast(device_type=device.type, enabled=False):
            dp_f = torch.abs(f_pl[:, :, None] - f_pl[:, None, :])
            dz_f = torch.abs(f_xyz[:, :, None, 2] - f_xyz[:, None, :, 2])
            dx0_f = torch.abs(f_x0[:, :, None] - f_x0[:, None, :])
            dx_f = f_xyz[:, :, None, 0] - f_xyz[:, None, :, 0]
            dy_f = f_xyz[:, :, None, 1] - f_xyz[:, None, :, 1]
            d_perp_f = _safe_hypot(dx_f, dy_f, eps=1e-10)
            tan_f = d_perp_f / dz_f.clamp_min(1.0)
            pair_f = torch.stack([
                dz_f / 30.0,
                dx0_f / 3.0,
                torch.log1p(d_perp_f / 0.1) / 3.0,
                torch.log1p(tan_f / 0.05) / 3.0,
                (dp_f == 0).float(),
                ((dp_f == 1) | (dp_f == 2)).float(),
            ], dim=-1)
            f_bias = self.front_pair_mlp(pair_f).permute(0, 3, 1, 2) + 0.5 * torch.log(
                g_persist[:, None, None, :].clamp_min(1e-3)
            )

        z_front = self.front_encoder(z_front, f_mask, pair_bias=f_bias)
        has_front = f_mask.any(dim=1)
        f_mask_clean = f_mask & ((g_persist > 0.45) | (f_feat[..., 3] >= 0.15))
        has_front_clean = f_mask_clean.any(dim=1)
        f_safe_m = f_mask.clone()
        if (~has_front).any():
            f_safe_m[~has_front, 0] = True
        fw_log = self.front_pool_gate(z_front).squeeze(-1).float().masked_fill(~f_safe_m, float('-inf'))
        fw = torch.softmax(fw_log, dim=-1) * f_mask.float()
        front_glob = (fw.unsqueeze(-1) * z_front.float()).sum(dim=1)

        bridge_q = self.front_bridge_queries.expand(b, -1, -1)
        z_bridge = self.front_bridge_attn(
            bridge_q, z_front, f_mask, query_mask=torch.ones((b, 2), dtype=torch.bool, device=device)
        )
        bridge_mask = has_front.unsqueeze(1).expand(b, 2)

        # Front 30mm Student-t Billoir PV/SV + Two-Stage Symmetric Refiner (Fix R2-2 clean anchor)
        with torch.amp.autocast(device_type=device.type, enabled=False):
            p_min_f = f_pl.masked_fill(~f_mask_clean, 99).min(dim=1).values
            p_min_fc = p_min_f.clamp(0, 14)
            z_first_f = self.pix_plane_z[p_min_fc].float()
            z_w_f = torch.where(p_min_fc == 0, z_first_f + 7.5, z_first_f - 14.0)
            first_f_m = f_mask_clean & (f_pl == p_min_f[:, None])
            ff_safe = first_f_m.clone()
            if (~has_front_clean).any():
                ff_safe[~has_front_clean, 0] = True
            w_ff = torch.softmax(f_feat[..., 3].float().masked_fill(~ff_safe, float('-inf')), dim=1) * first_f_m.float()
            xy_ff = (w_ff.unsqueeze(-1) * f_xyz[..., :2].float()).sum(dim=1)
            v_si_front = torch.stack([xy_ff[:, 0], xy_ff[:, 1], z_w_f], dim=-1)
            avr_ok_f = has_front_clean & avr_has & ((avr_pv[:, 2] - z_w_f).abs() < 45.0)
            v_init_f = torch.where(
                avr_ok_f[:, None], avr_pv[:, :3], torch.where(has_front_clean[:, None], v_si_front, v_cal_prior)
            )

        is_front_trk = trk_mask & (trk_morph[..., 1] < 0.5) & (trk_kin[..., 2] < -2750.0)
        tf_emb = self.trk_front_cross(h_trk, z_front, f_mask, query_mask=is_front_trk)
        z_ref_f = v_init_f[:, 2:3]
        trk_kin_cf = trk_kin.clone()
        trk_kin_cf[..., 2] = trk_kin_cf[..., 2] - z_ref_f
        v_init_f_c = torch.cat([v_init_f[:, :2], torch.zeros_like(z_ref_f)], dim=-1)
        fv_out = self.front_vtx_layer(trk_kin_cf, tf_emb, is_front_trk, v_init_f_c, cal_valid=has_front_clean)
        n_f_trk = is_front_trk.sum(dim=1, keepdim=True).float()
        w_trk_f = (n_f_trk >= 2.0).float() * 0.65
        v_front_prior = torch.cat([
            w_trk_f * fv_out['v_pv_mm'][:, :2] + (1.0 - w_trk_f) * v_init_f[:, :2],
            v_init_f[:, 2:3],
        ], dim=-1)
        front_meta_feat = torch.stack([
            (v_front_prior[:, 2] - self.z_mid) / 1000.0,
            ((z_first_f - v_front_prior[:, 2]) / 30.0).clamp(-4.0, 4.0) * has_front_clean.float(),
            ((v_cal_prior[:, 2] - v_front_prior[:, 2]) / 60.0).clamp(-5.0, 5.0),
            ((avr_pv[:, 2] - v_front_prior[:, 2]) / 30.0).clamp(-4.0, 4.0) * avr_has.float(),
            avr_ok_f.float(),
            torch.log1p(f_mask_clean.sum(dim=1).float()) / 5.0,
        ], dim=-1)
        f_prof_tok = torch.cat([z_seq, front_glob.to(z_seq.dtype).unsqueeze(1)], dim=1)
        f_prof_m = torch.cat([z_act, has_front_clean.unsqueeze(1)], dim=1)
        v_front, scale_front_raw = self.front_vtx_refine(
            f_prof_tok, f_prof_m, front_meta_feat, v_front_prior,
            pix_tokens=z_front, pix_xyz_mm=f_xyz, pix_mask=f_mask, pix_plane=f_pl,
        )

        # --- Regime 2 Mid Silicon Slices (Planes 6..11) ---
        mid_raw_m = pix_mask & (pix_plane >= 6) & (pix_plane <= 11)
        m_idx, m_mask = _select_masked_tokens(tok_rank, mid_raw_m, K_MID_MAX)
        m_xyz = torch.gather(pix_xyz, 1, m_idx.unsqueeze(-1).expand(-1, -1, 3))
        m_feat = torch.gather(pix_feat, 1, m_idx.unsqueeze(-1).expand(-1, -1, PIX_FEAT_DIM))
        m_pl = torch.gather(pix_plane, 1, m_idx)
        g_mid = torch.gather(g_persist_all, 1, m_idx) * m_mask.float()
        z_mid = torch.gather(u_pix, 1, m_idx.unsqueeze(-1).expand(-1, -1, self.dim))
        z_mid = self.mid_pix_cross(z_mid, cal_tok, cal_mask, query_mask=m_mask)
        z_mid = self.mid_pix_self(z_mid, z_mid, m_mask) * m_mask.unsqueeze(-1).to(z_mid.dtype)

        has_mid_si = m_mask.any(dim=1)
        m_mask_clean = m_mask & ((g_mid > 0.45) | (m_feat[..., 3] >= 0.15))
        has_mid_clean = m_mask_clean.any(dim=1)
        m_safe_m = m_mask.clone()
        if (~has_mid_si).any():
            m_safe_m[~has_mid_si, 0] = True
        mw_log = self.mid_pool_gate(z_mid).squeeze(-1).float().masked_fill(~m_safe_m, float('-inf'))
        mw = torch.softmax(mw_log, dim=-1) * m_mask.float()
        mid_si_glob = (mw.unsqueeze(-1) * z_mid.float()).sum(dim=1)

        with torch.amp.autocast(device_type=device.type, enabled=False):
            p_min_m = m_pl.masked_fill(~m_mask_clean, 99).min(dim=1).values
            p_min_mc = p_min_m.clamp(0, 14)
            z_first_m = self.pix_plane_z[p_min_mc].float()
            first_m_m = m_mask_clean & (m_pl == p_min_m[:, None])
            fm_safe = first_m_m.clone()
            if (~has_mid_clean).any():
                fm_safe[~has_mid_clean, 0] = True
            w_fm = torch.softmax(m_feat[..., 3].float().masked_fill(~fm_safe, float('-inf')), dim=1) * first_m_m.float()
            xy_fm = (w_fm.unsqueeze(-1) * m_xyz[..., :2].float()).sum(dim=1)
            z_est_m = torch.where(
                has_mid_clean & cal_valid,
                torch.minimum(v_cal_prior[:, 2], z_first_m - 12.5),
                torch.where(has_mid_clean, z_first_m - 30.0, v_cal_prior[:, 2]),
            )
            xy_est_m = torch.where(has_mid_clean[:, None], xy_fm, v_cal_prior[:, :2])
            v_si_mid = torch.cat([xy_est_m, z_est_m[:, None]], dim=-1)
            avr_ok_m = (~has_front_clean) & avr_has & (avr_pv[:, 2] >= -2780.0) & ((avr_pv[:, 2] - z_est_m).abs() < 90.0)
            v_init_m = torch.where(avr_ok_m[:, None], avr_pv[:, :3], v_si_mid)

        is_mid_trk = trk_mask & (trk_morph[..., 1] < 0.5) & (trk_kin[..., 2] >= -2750.0)
        z_ref_m = v_init_m[:, 2:3]
        trk_kin_cm = trk_kin.clone()
        trk_kin_cm[..., 2] = trk_kin_cm[..., 2] - z_ref_m
        v_init_m_c = torch.cat([v_init_m[:, :2], torch.zeros_like(z_ref_m)], dim=-1)
        mv_trk_out = self.mid_vtx_layer(trk_kin_cm, h_trk, is_mid_trk, v_init_m_c, cal_valid=(cal_valid | has_mid_clean))
        n_m_trk = is_mid_trk.sum(dim=1, keepdim=True).float()
        w_trk_m = (n_m_trk >= 2.0).float() * 0.50
        v_mid_prior = torch.cat([
            w_trk_m * mv_trk_out['v_pv_mm'][:, :2] + (1.0 - w_trk_m) * v_init_m[:, :2],
            v_init_m[:, 2:3],
        ], dim=-1)
        mid_meta_feat = torch.stack([
            (v_mid_prior[:, 2] - self.z_mid) / 1000.0,
            ((z_first_m - v_mid_prior[:, 2]) / 60.0).clamp(-5.0, 5.0) * has_mid_clean.float(),
            ((v_cal_prior[:, 2] - v_mid_prior[:, 2]) / 60.0).clamp(-5.0, 5.0),
            ((avr_pv[:, 2] - v_mid_prior[:, 2]) / 60.0).clamp(-5.0, 5.0) * avr_has.float(),
            has_mid_clean.float(),
            cal_valid.float(),
        ], dim=-1)
        m_prof_tok = torch.cat([z_seq, cal_tok.to(z_seq.dtype), mid_si_glob.to(z_seq.dtype).unsqueeze(1)], dim=1)
        m_prof_m = torch.cat([z_act, cal_mask, has_mid_clean.unsqueeze(1)], dim=1)
        v_mid, scale_mid_raw = self.mid_vtx_head(
            m_prof_tok, m_prof_m, mid_meta_feat, v_mid_prior,
            pix_tokens=z_mid, pix_xyz_mm=m_xyz, pix_mask=m_mask, pix_plane=m_pl,
        )

        # Regime-gated (`front_gate`) inverse-covariance PV fusion decoupling 30mm vs 150mm priors (9-cell corr)
        with torch.amp.autocast(device_type=device.type, enabled=False):
            front_persist_sum = F.relu(g_persist.float() - 0.45).sum(dim=1, keepdim=True) * f_mask.any(dim=1, keepdim=True).float()
            front_gate = has_front_clean[:, None].float() * torch.sigmoid(2.5 * (front_persist_sum - 0.55))
            p_f_raw = 1.0 / scale_front_raw.square().clamp_min(1e-4) + (n_f_trk >= 2.0).float() * (
                0.10 / fv_out['scale_pv_mm'].float().square().clamp_min(1e-2)
            )
            p_m_raw = 1.0 / scale_mid_raw.square().clamp_min(1e-4) + (n_m_trk >= 2.0).float() * (
                0.10 / mv_trk_out['scale_pv_mm'].float().square().clamp_min(1e-2)
            )
            dom_idx = (
                batch['obs_domain_id'].long().clamp(1, 3) - 1
                if 'obs_domain_id' in batch
                else torch.zeros((b,), dtype=torch.long, device=device)
            )
            corr = self.pv_regime_scale_corr.float()
            scale_front = (torch.rsqrt(p_f_raw) * corr[0, dom_idx, :]).clamp(0.03, 200.0)
            dz_to_mid_si = z_first_m - v_mid[:, 2].detach()
            is_mid_near_vtx = has_mid_clean & (dz_to_mid_si >= -15.0) & (dz_to_mid_si <= 35.0)
            mid_reg_idx = torch.where(is_mid_near_vtx, torch.ones_like(dom_idx), torch.full_like(dom_idx, 2))
            corr_mid_row = corr[mid_reg_idx, dom_idx, :]
            scale_mid = (torch.rsqrt(p_m_raw) * corr_mid_row).clamp(0.05, 220.0)

            p_front = (1.0 / scale_front.square().clamp_min(1e-4)) * front_gate
            p_mid = (1.0 / scale_mid.square().clamp_min(1e-4)) * (1.0 - 0.995 * front_gate)
            p_tot = (p_front + p_mid).clamp_min(1e-6)
            v_pv_mm = (p_front * v_front + p_mid * v_mid) / p_tot
            scale_pv_mm = torch.rsqrt(p_tot).clamp(0.03, 220.0)

            pi_pv_comb = front_gate * fv_out['pi_pv'] + (1.0 - front_gate) * mv_trk_out['pi_pv']
            pi_sv = front_gate * fv_out['pi_sv'] + (1.0 - front_gate) * mv_trk_out['pi_sv']
            sv_disp_mm = front_gate * fv_out['sv_disp_mm'] + (1.0 - front_gate) * mv_trk_out['sv_disp_mm']
            delta_chi2_2v = front_gate * fv_out['delta_chi2_2v'] + (1.0 - front_gate) * mv_trk_out['delta_chi2_2v']

            trk_ip_feat = TrackletIPFormer._compute_ip_features(trk_kin, trk_mask, v_pv_mm, pi_pv=pi_pv_comb)
            s_d0_raw = torch.expm1((3.0 * trk_ip_feat[..., 1]).clamp(0.0, 12.0))

        # Covariance-normalized signed IP re-injection + post-PV relational attention
        ip_in = torch.cat([trk_ip_feat, pi_sv.unsqueeze(-1)], dim=-1)
        h_trk = h_trk + self.ip_gate * self.ip_reinject(ip_in.to(h_trk.dtype)) * trk_mask.unsqueeze(-1).to(h_trk.dtype)
        h_trk = self.rel_attn_post(h_trk, trk_mask, pair_bias=pair_bias)
        h_trk = h_trk * trk_mask.unsqueeze(-1).to(h_trk.dtype)

        with torch.amp.autocast(device_type=device.type, enabled=False):
            any_trk = trk_mask.any(dim=1, keepdim=True)
            safe_tm = trk_mask.clone()
            if (~any_trk[:, 0]).any():
                safe_tm[~any_trk[:, 0], 0] = True
            is_mono_trk = trk_mask & (trk_morph[..., 2] < 0.16)
            t_logits = self.trk_pool_gate(h_trk.float()).squeeze(-1)
            t_logits = t_logits - 2.0 * (is_flood[:, None] & is_mono_trk).float()
            t_logits = t_logits.masked_fill(~safe_tm, float('-inf'))
            t_w = torch.softmax(t_logits, dim=1) * trk_mask.float()
            trk_glob = (t_w.unsqueeze(-1) * h_trk.float()).sum(dim=1)

        # 5. Regime 3 & 4: IPT (Planes 12..14) <-> Magnet-0 Spectrometer Interlock
        ipt_raw_m = pix_mask & (pix_plane >= 12)
        i_idx, i_mask = _select_masked_tokens(tok_rank, ipt_raw_m, K_IPT_MAX)
        g_ipt = torch.gather(g_persist_all, 1, i_idx) * i_mask.float()
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
        interlock = compute_ipt_spectrometer_interlock(
            trk_kin, is_ipt_trk, trk_cand, trk_cand_mask,
            x_pix_center=self.x_pix_c, y_scint_center=self.y_c,
        )
        st_occ_feat = (torch.log1p(station_counts.clamp_min(0.0)) / 3.0)[:, None, :].expand(b, trk_cand.shape[1], 3)
        cand_in = torch.cat([trk_cand, interlock, st_occ_feat], dim=-1)
        hc = self.spec_cand_in(cand_in) * trk_cand_mask.unsqueeze(-1).to(cand_in.dtype)
        hc = self.spec_cand_attn1(hc, hc, trk_cand_mask)
        hc = self.spec_cand_attn2(hc, hc, trk_cand_mask)

        spec_rel_logit = self.spec_rel_head(hc).squeeze(-1)
        rho_spec = torch.sigmoid(spec_rel_logit) * trk_cand_mask.float()
        z_spec_cands = self.spec_cand_to_dim(torch.cat([hc, interlock], dim=-1)) * trk_cand_mask.unsqueeze(-1).to(hc.dtype)

        with torch.amp.autocast(device_type=device.type, enabled=False):
            cm_f = trk_cand_mask.float()
            max_rho = rho_spec.max(dim=1).values
            w_cand = (rho_spec.float() / rho_spec.float().sum(dim=1, keepdim=True).clamp_min(1e-6)) * max_rho.float().unsqueeze(1)
            cand_pool = (w_cand.unsqueeze(-1) * hc.float()).sum(dim=1) * trk_cand_mask.any(dim=1, keepdim=True).float()

            is_3st = (trk_cand[..., 0] > 0.75) & trk_cand_mask
            sy_signed = SIGMA_SAGITTA_Y_MM * torch.sinh((5.0 * trk_cand[..., 6]).clamp(-9.0, 9.0))
            p_est = (SAGITTA_CONST_MEV_MM / (sy_signed.abs() + 0.015)).clamp(min=100.0, max=5.0e6)
            p_log_w = (w_cand * torch.log1p(p_est / 1000.0) * is_3st.float()).sum(dim=1)
            q_charge_w = (w_cand * torch.sign(sy_signed) * is_3st.float()).sum(dim=1)
            best_ilock = (interlock[..., 0] * cm_f).max(dim=1).values
            mean_rho = rho_spec.sum(dim=1) / cm_f.sum(dim=1).clamp_min(1.0)

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
        hit_pool = hit_pool * (0.05 + 0.95 * max_rho.unsqueeze(-1)).to(hit_pool.dtype)

        spec_summary_vec = torch.stack([p_log_w, q_charge_w, best_ilock, max_rho, mean_rho], dim=-1)
        spec_glob = self.spec_glob_fusion(torch.cat([cand_pool, hit_pool, ipt_glob, spec_summary_vec], dim=-1))

        # 6. Precision-Gated Cross-Modal Perceiver Fusion & Sanitized c_obs_eff (Fix R2-1 & R2-2)
        source = torch.cat([
            h_trk, z_front, z_bridge, z_mid, cal_tok, z_ipt, z_spec_cands, spec_glob.unsqueeze(1),
        ], dim=1)
        source_mask = torch.cat([
            trk_mask, f_mask, bridge_mask, m_mask, cal_mask, i_mask, trk_cand_mask, has_hit.unsqueeze(1),
        ], dim=1)

        with torch.amp.autocast(device_type=device.type, enabled=False):
            c_obs_eff = batch['c_obs'].float().clone()
            c_obs_eff[:, 2] = max_rho.detach()
            c_obs_eff[:, 3] = (rho_spec.detach() * is_3st.float()).max(dim=1).values
            if self.pix_p0_upstream and not self.scint_p0_upstream:
                has_p0_raw = (pix_mask & (pix_plane == 0)).any(dim=1)
                has_p0_ver = (pm_ver & (pix_plane == 0)).any(dim=1)
                c_obs_eff[:, 6] = torch.where(has_p0_raw, has_p0_ver.float(), c_obs_eff[:, 6])
            if is_flood.any():
                has_pm = pm_ver.any(dim=1)
                z_p_first = torch.where(
                    has_pm,
                    pix_xyz[..., 2].float().masked_fill(~pm_ver, 1e9).min(dim=1).values,
                    torch.full((b,), 1e9, device=device),
                )
                has_bar = c_obs_eff[:, 1] > 0.5
                st_m_cal = cal_out['st_mask']
                z_bar_first = torch.where(
                    st_m_cal, self.station_z_mm[None, :].float(), torch.full((b, self.n_st), 1e9, device=device)
                ).min(dim=1).values
                z_first_clean = torch.minimum(z_p_first, z_bar_first)
                has_any_clean = has_pm | (has_bar & st_m_cal.any(dim=1))
                z_first_norm = torch.where(
                    has_any_clean,
                    (z_first_clean - self.z_target_min) / max(self.l_target, 1.0),
                    torch.full((b,), 1.5, device=device),
                )
                n_hits_cl_eff = torch.expm1(6.0 * pix_feat[..., 3].float().clamp_min(0.0)).clamp_min(1.0)
                n_pix_clean = (n_hits_cl_eff * pm_ver.float()).sum(dim=1)
                c_obs_eff[:, 0] = torch.where(is_flood, has_pm.float(), c_obs_eff[:, 0])
                c_obs_eff[:, 4] = torch.where(is_flood, z_first_norm, c_obs_eff[:, 4])
                c_obs_eff[:, 13] = torch.where(is_flood, torch.log1p(n_pix_clean) / 10.0, c_obs_eff[:, 13])

        cobs_q = self.cobs_query(c_obs_eff).unsqueeze(1)
        latents = torch.cat([self.latent_queries.expand(b, -1, -1), cobs_q], dim=1)
        latent_mask = torch.ones((b, self.n_queries + 1), dtype=torch.bool, device=device)

        has_any_cand = trk_cand_mask.any(dim=1, keepdim=True)
        spec_glob_rho = torch.where(
            has_any_cand | has_hit.unsqueeze(1), max_rho.unsqueeze(1).clamp_min(1e-3), torch.ones((b, 1), device=device),
        )
        trk_rho = torch.where(
            is_flood[:, None] & (trk_morph[..., 2] < 0.16),
            torch.full_like(trk_morph[..., 2], 0.15),
            torch.ones_like(trk_morph[..., 2]),
        )
        tok_rho = torch.cat([
            trk_rho,
            g_persist.clamp_min(1e-3),
            torch.ones((b, z_bridge.shape[1]), device=device),
            g_mid.clamp_min(1e-3),
            torch.ones((b, cal_tok.shape[1]), device=device),
            g_ipt.clamp_min(1e-3),
            rho_spec.clamp_min(1e-3),
            spec_glob_rho,
        ], dim=1)
        ugca_bias = self.ugca_beta.float() * torch.log(tok_rho)[:, None, None, :]

        for blk in self.global_blocks:
            latents = blk.cross(latents, source, source_mask, pair_bias=ugca_bias, query_mask=latent_mask)
            latents = blk.self_attn(latents, latents, latent_mask)

        q_lat = latents[:, :self.n_queries]
        q_cobs = latents[:, self.n_queries]

        with torch.amp.autocast(device_type=device.type, enabled=False):
            w_lat = torch.softmax(self.pool_gate(q_lat.float()), dim=1)
            pool_attn = (w_lat * q_lat.float()).sum(dim=1)

            # Fix R2-3: strictly detach PV/SV outputs fed into classification/recoil fusion
            sv_det = sv_disp_mm.detach().float()
            dc_det = delta_chi2_2v.detach().float()
            sc_det = scale_pv_mm.detach().float()
            vp_det = v_pv_mm.detach().float()
            sv_norm = torch.sqrt(sv_det.square().sum(dim=-1, keepdim=True) + 1e-8)
            vtx_summary = torch.cat([
                torch.log1p(dc_det.clamp_min(0.0)) / 4.0,
                torch.asinh(sv_norm / 0.2) / 4.0,
                torch.asinh(sv_det[:, 2:3] / 1.0) / 4.0,
                torch.log(sc_det.clamp(min=0.02, max=150.0)) / 4.0,
                (vp_det[:, 2:3] - self.z_mid) / 1000.0,
                (vp_det[:, 0:1] - self.x_pix_c) / 135.0,
                torch.asinh((vp_det[:, 1:2] - self.y_c) / 100.0),
            ], dim=-1)
            vtx_emb = self.vtx_summary_proj(vtx_summary)

            fuse_parts = [
                pool_attn,
                trk_glob,
                front_glob.float(),
                mid_si_glob.float(),
                ipt_glob.float(),
                cal_out['cal_global'].float(),
                z_prof_glob,
                spec_glob.float(),
                q_cobs.float(),
                vtx_emb,
            ]
            if self.hemisphere_mod is not None and self.hem_proj is not None:
                cand_hem = trk_cand.clone()
                cand_hem[..., 1] = trk_cand[..., 1] * rho_spec.detach()
                mask_hem = trk_cand_mask & (rho_spec.detach() > 0.20)
                hem_out = self.hemisphere_mod(
                    cal_out['grid_x_e'],
                    cal_out['grid_y_e'],
                    vp_det,
                    trk_kin,
                    trk_morph,
                    trk_mask,
                    pi_sv.detach(),
                    s_d0_raw.detach(),
                    cand_hem,
                    mask_hem,
                    interlock,
                )
                hem_feat = hem_out['law3_feat']
                is_tgt_trk = trk_mask & (trk_morph[..., 1] < 0.5)
                has_tgt_trk = is_tgt_trk.any(dim=1)
                has_stub_d = (trk_morph[..., 6].float() * is_tgt_trk.float()).max(dim=1).values * has_tgt_trk.float()
                gap_d = (trk_morph[..., 7].float() * is_tgt_trk.float()).max(dim=1).values * has_tgt_trk.float()
                shared_root_3p = (trk_morph[..., 0].float() * is_tgt_trk.float()).max(dim=1).values * has_tgt_trk.float()
                hem_in = torch.cat([hem_feat, torch.stack([has_stub_d, gap_d, shared_root_3p], dim=-1)], dim=-1)
                fuse_parts.append(self.hem_proj(hem_in))
                even_mask = hem_in.new_ones((1, hem_in.shape[1]))
                even_mask[0, 7] = 0.0
                hem_emb_even = self.hem_proj(hem_in * even_mask)
            else:
                hem_feat = v_pv_mm.new_zeros((b, HEM_FEAT_DIM))
                hem_out = None
                hem_emb_even = trk_glob.new_zeros((b, self.dim // 2))

            fused_global = self.global_fusion(torch.cat(fuse_parts, dim=-1))

            # 7. Calibrated Multi-Task Readout + Conflict-Free CC Charm + Zero-Shot NC_cc
            z_bkg_c = self.charm_bkg_head(torch.cat([trk_glob, vtx_emb, hem_emb_even], dim=-1)).squeeze(-1)
            q_bkg_charm = torch.sigmoid(z_bkg_c).clamp(1e-6, 1.0 - 1e-6)
            z_tau_c = self.tau_charm_head(fused_global).squeeze(-1)
            p_tau_c = torch.sigmoid(z_tau_c)

            logits7_raw = self.event_head_7(fused_global)
            if self.hemisphere_mod is not None and hem_out is not None:
                scales_h = F.softplus(self.hemisphere_mod.law_norm_scale.float()).clamp(0.2, 3.0)
                hem_raw = hem_feat.float() / scales_h[None, :]
                c_hem_r, i_iso_r, acop_r, mu_conf_r = hem_raw[:, 0], hem_raw[:, 1], hem_raw[:, 2], hem_raw[:, 4]
                sv_s = torch.asinh(sv_det.norm(dim=-1) / 0.2)
                back_to_back = F.relu(-c_hem_r - 0.12)
                non_iso_jet = torch.sigmoid((0.38 - i_iso_r) / 0.08) * torch.sigmoid((0.62 - acop_r) / 0.10)
                no_mu_gate = torch.sigmoid((0.10 - mu_conf_r) / 0.03) * torch.sigmoid((0.25 - max_rho.detach()) / 0.05)
                sv_gate = torch.sigmoid((sv_s - 3.5) / 1.0)
                delta_nccc = 1.35 * back_to_back * non_iso_jet * no_mu_gate * sv_gate * (0.50 + 0.60 * q_bkg_charm.detach())
                ell_nccc = torch.maximum(logits7_raw[:, 6], logits7_raw[:, 5] - 1.45) - 1.15 + delta_nccc
                active_nccc = (back_to_back > 0.0) & (mu_conf_r < 0.15) & (hem_out['has_disp'] > 0.5)
                logits7 = logits7_raw.clone()
                logits7[:, 6] = torch.where(active_nccc, torch.logaddexp(logits7_raw[:, 6], ell_nccc), logits7_raw[:, 6])
                nccc_boost = torch.where(active_nccc, delta_nccc - 1.15, torch.zeros_like(delta_nccc))
            else:
                logits7 = logits7_raw
                active_nccc = torch.zeros((b,), dtype=torch.bool, device=device)
                nccc_boost = torch.zeros((b,), dtype=torch.float32, device=device)

            probs4 = marginal_four_from_seven(logits7, is_probs=False)
            p_tau = probs4[:, 2].clamp(0.0, 1.0 - 1e-6)
            p_all_charm = ((1.0 - p_tau) * q_bkg_charm + p_tau * p_tau_c).clamp(1e-6, 1.0 - 1e-6)
            charm_logit = torch.log(p_all_charm) - torch.log1p(-p_all_charm)
            kin_pred = self.kin_head(fused_global)

        cand_clean_pos = (~cand_injected_noise) & trk_cand_mask & (
            ((trk_cand[..., 0] > 0.75) & ((interlock[..., 0] > 0.25) | (trk_cand[..., 9] < 0.6)))
            | ((trk_cand[..., 0] > 0.40) & (interlock[..., 0] > 0.45))
        )
        pixdark_pos_m = (~pix_injected_dark) & pix_mask & (pix_feat[..., 3] >= 0.15)
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
                'nccc_active': active_nccc.detach(),
                'nccc_boost': nccc_boost.detach(),
                'rho_spec': rho_spec.detach(),
                'spec_rel_logit': spec_rel_logit,
                'trk_cand_eff': trk_cand.detach(),
                'trk_cand_mask_eff': trk_cand_mask.detach(),
                'cand_injected_noise': cand_injected_noise,
                'cand_clean_pos': cand_clean_pos,
                'pixdark_logit_all': pixdark_logit_all,
                'pixdark_pos_m': pixdark_pos_m,
                'pix_injected_dark': pix_injected_dark,
                'front_gate': front_gate.squeeze(-1).detach(),
                'has_mid_si': is_mid_near_vtx.detach(),
                'v_front_mm': v_front,
                'scale_front_mm': scale_front,
                'P_front': p_front,
                'v_mid_mm': v_mid,
                'scale_mid_mm': scale_mid,
                'P_mid': p_mid,
                'delta_chi2_2v': delta_chi2_2v.float(),
                'p_sv_trklet': pi_sv.float(),
                'g_persist_mean': g_persist.mean(dim=-1).detach(),
                'n_dual_active': cal_out['n_dual_active'].detach(),
                'c_hem_opp': hem_feat[:, 0].detach(),
            },
        )


def compute_unified_window_denominators(micro_batches: list[dict], pos_weight: float = 3.0) -> dict[str, float]:
    """Exact full-window per-task denominators across microbatches guaranteeing grad_accum equivalence."""
    d_proc = 0.0
    d_charm_nontau = 0.0
    d_charm_all = 0.0
    d_vtx = 0.0
    d_vtx_front = 0.0
    d_vtx_mid = 0.0
    d_kin = 0.0
    for mb in micro_batches:
        lbl4 = mb['label'].long()
        valid_p = (mb['label_7'] >= 0) | ((mb['label_7'] < 0) & (lbl4 == 2))
        d_proc += float(valid_p.float().sum().item())

        valid_c = mb['charm_valid'] > 0
        valid_c_all = valid_c & (lbl4 <= 2)
        if valid_c_all.any():
            yc = mb['charm_flag'][valid_c_all].float()
            d_charm_all += float(torch.where(yc > 0.5, float(pos_weight), 1.0).sum().item())
        valid_c_nt = valid_c & (lbl4 < 2)
        if valid_c_nt.any():
            yc_nt = mb['charm_flag'][valid_c_nt].float()
            d_charm_nontau += float(torch.where(yc_nt > 0.5, float(pos_weight), 1.0).sum().item())

        valid_v = mb['vertex_valid'] > 0
        d_vtx += float(valid_v.float().sum().item())
        if valid_v.any():
            vz = mb['vertex_xyz_mm'][:, 2]
            d_vtx_front += float((valid_v & (vz < -2750.0)).float().sum().item())
            d_vtx_mid += float((valid_v & (vz >= -2750.0)).float().sum().item())
        d_kin += float((mb['kin_valid'] > 0).float().sum().item())
    return {
        'proc': max(d_proc, 1e-8),
        'charm_nontau': max(d_charm_nontau, 1e-8),
        'charm_all': max(d_charm_all, 1e-8),
        'vtx': max(d_vtx, 1e-8),
        'vtx_front': max(d_vtx_front, 1e-8),
        'vtx_mid': max(d_vtx_mid, 1e-8),
        'kin': max(d_kin, 1e-8),
    }


def compute_unified_loss(
    model: PinpointUnifiedNet,
    batch: dict[str, torch.Tensor],
    fixed_denoms: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Multitask loss: 7->4 marginal CE + CC-only Charm + Student-t PV + spec_rel + pixdark_bce + kinematics."""
    fd = fixed_denoms or {}
    out = model.forward_NextGen(batch)

    l_proc = marginal_tau_branch_ce(
        out.logits7,
        batch['label_7'],
        batch['label'],
        denom_override=fd.get('proc'),
    )

    if 'charm_valid' in batch and 'charm_flag' in batch:
        cv = batch['charm_valid'].long() > 0
        cf = batch['charm_flag'].float()
        lbl4 = batch['label'].long()
        cv_nt = cv & (lbl4 < 2)
        cv_all = cv & (lbl4 <= 2)
        l_c_nt = _focal_bce_from_prob(
            out.aux['q_bkg_charm'],
            cf,
            cv_nt,
            denom_override=fd.get('charm_nontau'),
        )
        l_c_all = _focal_bce_from_prob(
            out.aux['p_all_charm'],
            cf,
            cv_all,
            denom_override=fd.get('charm_all'),
        )
        l_charm = 0.5 * l_c_nt + 0.5 * l_c_all
    else:
        l_charm = out.logits7.sum() * 0.0

    l_vtx = student_t_vertex_loss(
        out.v_pv_mm,
        out.scale_pv_mm,
        batch['vertex_xyz_mm'],
        batch['vertex_valid'],
        df=3.0,
        denom_override=fd.get('vtx'),
    )

    l_vtx_reg = l_proc.new_zeros(())
    if 'v_front_mm' in out.aux and 'v_mid_mm' in out.aux:
        vv = batch['vertex_valid'] > 0
        vz = batch['vertex_xyz_mm'][:, 2]
        m_front = vv & (vz < -2750.0)
        m_mid = vv & (vz >= -2750.0)
        if m_front.any():
            l_vtx_reg = l_vtx_reg + 0.12 * student_t_vertex_loss(
                out.aux['v_front_mm'],
                out.aux['scale_front_mm'],
                batch['vertex_xyz_mm'],
                m_front.long(),
                df=3.0,
                denom_override=fd.get('vtx_front'),
            )
        if m_mid.any():
            l_vtx_reg = l_vtx_reg + 0.12 * student_t_vertex_loss(
                out.aux['v_mid_mm'],
                out.aux['scale_mid_mm'],
                batch['vertex_xyz_mm'],
                m_mid.long(),
                df=3.0,
                denom_override=fd.get('vtx_mid'),
            )

    l_spec_rel = l_proc.new_zeros(())
    if 'spec_rel_logit' in out.aux:
        with torch.amp.autocast(device_type=out.logits7.device.type, enabled=False):
            s_logit = out.aux['spec_rel_logit'].float()
            pos_m = out.aux['cand_clean_pos'].bool()
            neg_inj = out.aux['cand_injected_noise'].bool()
            tc_eff = out.aux['trk_cand_eff']
            tcm_eff = out.aux['trk_cand_mask_eff'].bool()
            neg_chi2 = (~neg_inj) & (~pos_m) & tcm_eff & (tc_eff[..., 9].float() > 0.85)
            neg_m = neg_inj | neg_chi2
            sup_m = pos_m | neg_m
            if sup_m.any():
                tgt_rel = pos_m.float()
                bce_all = F.binary_cross_entropy_with_logits(s_logit, tgt_rel, reduction='none') * sup_m.float()
                ev_sup_cnt = sup_m.float().sum(dim=1)
                ev_bce = torch.where(ev_sup_cnt > 0, bce_all.sum(dim=1) / ev_sup_cnt.clamp_min(1.0), torch.zeros_like(ev_sup_cnt))
                dp = torch.as_tensor(fd.get('proc', float(len(batch['label']))), dtype=torch.float32, device=s_logit.device).clamp_min(1e-8)
                l_spec_rel = ev_bce.sum() / dp

    l_pixdark_bce = l_proc.new_zeros(())
    if 'pixdark_logit_all' in out.aux:
        with torch.amp.autocast(device_type=out.logits7.device.type, enabled=False):
            d_logit = out.aux['pixdark_logit_all'].float()
            pos_d = out.aux['pixdark_pos_m'].bool()
            neg_d = out.aux['pix_injected_dark'].bool()
            sup_d = pos_d | neg_d
            if sup_d.any():
                tgt_d = pos_d.float()
                bce_d = F.binary_cross_entropy_with_logits(d_logit, tgt_d, reduction='none') * sup_d.float()
                ev_sup_d = sup_d.float().sum(dim=1)
                ev_bce_d = torch.where(ev_sup_d > 0, bce_d.sum(dim=1) / ev_sup_d.clamp_min(1.0), torch.zeros_like(ev_sup_d))
                dp = torch.as_tensor(fd.get('proc', float(len(batch['label']))), dtype=torch.float32, device=d_logit.device).clamp_min(1e-8)
                l_pixdark_bce = ev_bce_d.sum() / dp

    kin_valid = batch['kin_valid'] > 0
    if kin_valid.any() and out.kin_pred is not None:
        with torch.amp.autocast(device_type=out.logits7.device.type, enabled=False):
            kin_ev = F.smooth_l1_loss(
                out.kin_pred[kin_valid].float(),
                batch['vis_kinematics'][kin_valid].float(),
                reduction='none',
            ).mean(dim=-1)
            dk = torch.as_tensor(fd.get('kin', float(kin_valid.sum())), dtype=torch.float32, device=kin_ev.device).clamp_min(1e-8)
            l_kin = kin_ev.sum() / dk
    else:
        l_kin = l_proc.new_zeros(())

    total = (
        l_proc.float()
        + 0.25 * l_charm.float()
        + 0.30 * l_vtx.float()
        + l_vtx_reg.float()
        + 0.05 * l_spec_rel.float()
        + 0.04 * l_pixdark_bce.float()
        + 0.15 * l_kin.float()
    )
    return total, out.probs4
