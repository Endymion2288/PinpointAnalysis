"""Proposal F: TauKinemaDiffNet — PINPOINT-Native Differentiable 12-Hypothesis Physics-Law Network.

Strictly isolated under `src/analysis/flavour/nextgen/`, consuming pure observables only
(zero truth inputs at inference). Evaluates explicit physical interaction hypotheses:
    [0: nu_e_p, 1: nu_e_c, 2: nu_mu-_p, 3: nu_mu+_p, 4: nu_mu-_c, 5: nu_mu+_c,
     6: tau_e, 7: tau_mu, 8: tau_1h, 9: tau_3h, 10: NC_light, 11: NC_cc]
via three differentiable PINPOINT-native physics law modules + bounded neural residual:
  1. TungstenCascadeProfileFitter: 1.427 X0/plate Longo-Sestili Gamma EM vs lambda_I MIP profile;
  2. Parent-tau collinear stub vs gamma->e+e- neutral-gap Bayes factor B_stub/gap;
  3. DisplacementAnchoredHemisphereModule: C_hem = -cos(dphi(u_T_disp, Q_T_had)) + bore/IPT muon.
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
    pairwise_skew_dca_features,
)
from analysis.flavour.v3.common.domains import cumulative_radiation_length, detector_bounds
from analysis.flavour.v3.common.layers import ScaledQKNormAttention, is_weight_decay_exempt
from analysis.flavour.v3.v1_b1pp.spectrometer import (
    MAGNET0_LENGTH_MM,
    MAGNET0_Z_CENTER_MM,
    MAGNET0_Z_MAX_MM,
    MAGNET0_Z_MIN_MM,
    SAGITTA_CONST_MEV_MM,
    SAGITTA_INTEGRAL_MM2,
    SIGMA_SAGITTA_Y_MM,
    TRACKER_Z_MM,
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
)
from analysis.flavour.v3.v2_pinpointformer.stereo_bars import StereoBarNexus, static_scint_stations
from analysis.flavour.v3.v2_pinpointformer.vertex_anchor import (
    VertexAnchorPredictor,
    select_near_vertex_clusters,
)


PHI_LAW_DIM = 18
N_HYPOTHESES = 12
HYPOTHESIS_NAMES = (
    'nu_e_p',
    'nu_e_c',
    'nu_mu_minus_p',
    'nu_mu_plus_p',
    'nu_mu_minus_c',
    'nu_mu_plus_c',
    'tau_e',
    'tau_mu',
    'tau_1h',
    'tau_3h',
    'NC_light',
    'NC_cc',
)
PHI_LAW_NAMES = (
    'd_kl_em',
    'l_mip_pen',
    'moliere_pen',
    'f_early_em',
    'f_tail_pen',
    's_disp',
    'dchi2_2v',
    'sv_flight',
    'b_stub_gap',
    'shared_root_3p',
    'c_hem_opp',
    'i_iso_prong',
    'acoplanarity_disp',
    'pt_miss_frac',
    'mu_bore_ipt_conf',
    'log_pmu_ehad',
    'mu_disp_opp',
    'mu_charge_signed',
)

# Tungsten physical constants matching Pinpoint_G4 (G4_W)
W_CRITICAL_ENERGY_MEV = 7.97
W_MOLIERE_RADIUS_MM = 9.30
W_X0_OVER_LAMBDA_I = 3.504 / 99.5

TAU_KINEMA_NO_WD_KEYWORDS = (
    'A_phys',
    'b_phys',
    'alpha_res',
    'cal_log_prec',
    'scale_cal',
    'b_em_param',
    'w_stub',
    'w_gap',
    'law_norm_scale',
)


def _build_initial_physics_matrix() -> tuple[torch.Tensor, torch.Tensor]:
    """Construct signed prior physics-law matrix A_init (12, 18) and prior intercept b_init (12,).

    Rows: [0:e_p, 1:e_c, 2:mu-_p, 3:mu+_p, 4:mu-_c, 5:mu+_c,
           6:tau_e, 7:tau_mu, 8:tau_1h, 9:tau_3h, 10:NC_light, 11:NC_cc]
    Cols: 18 dimensionless PINPOINT physics-law sufficient statistics `phi_law`.
    """
    a = torch.zeros((N_HYPOTHESES, PHI_LAW_DIM), dtype=torch.float32)

    # Cols 0..4: Law 1 Tungsten 1.427 X0 EM vs MIP profile
    # 0: d_kl_em (low for EM, high for hadronic/MIP)
    a[[0, 1, 6], 0] = torch.tensor([-1.8, -1.0, -1.4])
    a[[2, 3, 4, 5, 7, 8, 9, 10, 11], 0] = torch.tensor([0.5, 0.5, 0.4, 0.4, 0.4, 0.9, 0.9, 1.2, 1.0])
    # 1: l_mip_pen (low for prompt e, high for penetrating pi/mu)
    a[[0, 1, 6], 1] = torch.tensor([-1.5, -0.8, -1.2])
    a[[2, 3, 7, 8, 9], 1] = torch.tensor([1.0, 1.0, 1.0, 1.4, 0.8])
    # 2: moliere_pen (early wide R_perp > 1.8 R_M)
    a[[0, 6], 2] = torch.tensor([-1.2, -1.0])
    a[[8, 9, 10, 11], 2] = torch.tensor([0.5, 0.7, 1.1, 1.0])
    # 3: f_early_em (energy in <= 6 X0)
    a[[0, 1, 6], 3] = torch.tensor([1.6, 1.0, 1.3])
    a[[2, 3, 7, 10], 3] = torch.tensor([-0.6, -0.6, -0.5, -0.8])
    # 4: f_tail_pen (deep tail >= 35 X0)
    a[[2, 3, 4, 5, 7], 4] = torch.tensor([1.2, 1.2, 1.0, 1.0, 1.1])
    a[[0, 6, 10], 4] = torch.tensor([-1.0, -0.8, -0.6])

    # Cols 5..9: Law 2 Displaced topology & parent-stub vs photon-gap Bayes factor
    # 5: s_disp (IP significance)
    a[[0, 2, 3, 10], 5] = torch.tensor([-1.2, -1.2, -1.2, -1.2])
    a[[1, 4, 5, 6, 7, 8, 9, 11], 5] = torch.tensor([1.3, 1.3, 1.3, 1.2, 1.2, 1.8, 1.9, 1.7])
    # 6: dchi2_2v (2-vertex improvement)
    a[[0, 2, 3, 10], 6] = torch.tensor([-0.8, -0.8, -0.8, -0.8])
    a[[1, 4, 5, 9, 11], 6] = torch.tensor([1.2, 1.2, 1.2, 1.7, 1.5])
    # 7: sv_flight (SV displacement scale)
    a[[0, 2, 3, 10], 7] = torch.tensor([-0.7, -0.7, -0.7, -0.7])
    a[[1, 4, 5, 6, 7, 8, 9, 11], 7] = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.4, 1.5, 1.3])
    # 8: b_stub_gap (+ for parent tau stub, - for gamma->e+e- neutral gap)
    a[[0, 1, 10], 8] = torch.tensor([-1.5, -0.8, -0.6])
    a[[6, 7, 8, 9], 8] = torch.tensor([0.7, 0.7, 2.0, 1.4])
    # 9: shared_root_3p (3-prong shared merged root)
    a[[0, 2, 3, 8, 10], 9] = torch.tensor([-0.4, -0.4, -0.4, -0.3, -0.4])
    a[[1, 4, 5, 9, 11], 9] = torch.tensor([0.9, 0.9, 0.9, 1.8, 1.2])

    # Cols 10..17: Law 3 Displacement-anchored azimuthal recoil hemisphere & spectrometer
    # 10: c_hem_opp = -cos(dphi(u_T_disp, Q_T_had)): + for tau CC, - for NC_cc & CC_charm
    a[[1, 4, 5, 11], 10] = torch.tensor([-1.5, -1.5, -1.5, -2.4])
    a[[6, 7, 8, 9], 10] = torch.tensor([1.6, 1.4, 2.4, 2.4])
    # 11: i_iso_prong (prong isolation: + for tau, - for charm in jet & NC)
    a[[1, 4, 5, 10, 11], 11] = torch.tensor([-1.0, -1.0, -1.0, -1.2, -2.0])
    a[[6, 7, 8, 9], 11] = torch.tensor([1.3, 1.2, 2.0, 1.9])
    # 12: acoplanarity_disp (escaping neutrino non-collinearity)
    a[[0, 2, 3], 12] = torch.tensor([-0.6, -0.6, -0.6])
    a[[6, 7, 8, 9, 10, 11], 12] = torch.tensor([0.9, 0.9, 1.1, 1.1, 0.8, 0.7])
    # 13: pt_miss_frac (missing transverse momentum fraction)
    a[[0, 2, 3], 13] = torch.tensor([-0.8, -0.9, -0.9])
    a[[6, 7, 8, 9, 10, 11], 13] = torch.tensor([1.1, 1.1, 1.2, 1.2, 1.3, 1.0])
    # 14: mu_bore_ipt_conf (bore- & IPT-verified 3-station muon)
    a[[0, 1, 6, 8, 9, 10, 11], 14] = torch.tensor([-2.8, -2.6, -2.6, -2.8, -2.8, -2.8, -2.8])
    a[[2, 3, 4, 5, 7], 14] = torch.tensor([3.0, 3.0, 2.8, 2.8, 2.6])
    # 15: log_pmu_ehad (V-A softened muon momentum ratio: + for prompt nu_mu, - for tau_mu)
    a[[2, 3, 4, 5], 15] = torch.tensor([1.2, 1.2, 0.9, 0.9])
    a[7, 15] = -1.6
    # 16: mu_disp_opp (verified muon opposite displaced charm vertex -> nu_mu_charm)
    a[[2, 3, 7], 16] = torch.tensor([-0.6, -0.6, -0.4])
    a[[4, 5], 16] = torch.tensor([1.9, 1.9])
    # 17: mu_charge_signed (exact Y-mirror-odd curvature sign: + for mu-, - for mu+)
    a[[2, 4], 17] = torch.tensor([1.4, 1.4])
    a[[3, 5], 17] = torch.tensor([-1.4, -1.4])

    # Gentle scaling (0.20x) for standardized phi_norm avoids step-0 logit saturation.
    a = 0.20 * a

    # Balanced 4-class macro priors (probs4 ~ [0.25, 0.25, 0.25, 0.25], NLL ~ 1.39),
    # Y-mirror mu-/mu+ symmetry, and ~88:12 prompt/charm branching priors.
    b = torch.tensor(
        [-0.128, -2.120, -0.821, -0.821, -2.813, -2.813, -1.715, -1.772, -0.693, -1.897, -0.128, -2.120],
        dtype=torch.float32,
    )
    return a, b


class TungstenCascadeProfileFitter(nn.Module):
    """Law 1: Differentiable 1.427 X0/plate Longo-Sestili EM vs lambda_I MIP profile fitter."""

    def __init__(self, geometry: dict, stations: dict) -> None:
        super().__init__()
        st_z = np.asarray(stations['station_z_mm'], dtype=np.float32)
        st_x0 = np.asarray(stations['station_x0'], dtype=np.float32)
        all_z = np.concatenate([
            np.asarray(geometry['pixel_Zpos'], dtype=np.float32),
            st_z,
        ])
        all_x0 = np.array([cumulative_radiation_length(float(z), geometry) for z in all_z], dtype=np.float32)
        ord_z = np.argsort(all_z, kind='mergesort')
        self.register_buffer('ref_z_mm', torch.from_numpy(all_z[ord_z]))
        self.register_buffer('ref_x0', torch.from_numpy(all_x0[ord_z]))
        self.register_buffer('st_z_mm', torch.from_numpy(st_z))
        self.register_buffer('st_x0', torch.from_numpy(st_x0))
        self.register_buffer('st_has_y', torch.from_numpy(stations['station_has_y']))
        self.register_buffer('dbl_idx', torch.from_numpy(stations['doublet_stations']))
        self.register_buffer('bar_x_mm', torch.from_numpy(stations['bar_x_centers_mm']))
        self.register_buffer('bar_y_mm', torch.from_numpy(stations['bar_y_centers_mm']))

        self.b_em_param = nn.Parameter(torch.tensor(0.50, dtype=torch.float32))
        self.law_norm_scale = nn.Parameter(torch.ones(5, dtype=torch.float32))

    def _interp_x0(self, z_mm: torch.Tensor) -> torch.Tensor:
        """Piecewise linear differentiable interpolation of cumulative X0(z)."""
        rz = self.ref_z_mm
        rx = self.ref_x0
        z_c = z_mm.clamp(min=float(rz[0].item()), max=float(rz[-1].item()))
        idx_hi = torch.searchsorted(rz, z_c.contiguous()).clamp(min=1, max=len(rz) - 1)
        idx_lo = idx_hi - 1
        z0 = rz[idx_lo]
        z1 = rz[idx_hi]
        x0 = rx[idx_lo]
        x1 = rx[idx_hi]
        frac = (z_c - z0) / (z1 - z0).clamp_min(1e-4)
        return x0 + frac * (x1 - x0)

    def forward(
        self,
        grid_x_e: torch.Tensor,
        grid_y_e: torch.Tensor,
        v_pv_mm: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute 5-dim Law-1 features (B, 5) & per-station normalized energy profile (B, 54)."""
        with torch.amp.autocast(device_type=grid_x_e.device.type, enabled=False):
            ex = grid_x_e.float()  # (B, 54, 40) MeV
            ey = grid_y_e.float()  # (B, 42, 40) MeV
            b = ex.shape[0]

            ey_full = ex.new_zeros((b, ex.shape[1], ex.shape[2]))
            ey_full[:, self.dbl_idx] = ey
            e_x_st = ex.sum(dim=-1)  # (B, 54)
            e_y_st = ey_full.sum(dim=-1)  # (B, 54)
            has_y = self.st_has_y[None, :].expand(b, -1)
            e_st = torch.where(has_y & (e_y_st > 0.0), 0.5 * (e_x_st + e_y_st), e_x_st)  # (B, 54)

            # Transverse RMS radius per station (in mm, Y-mirror invariant)
            bx = self.bar_x_mm[None, None, :]
            by = self.bar_y_mm[None, None, :]
            wx_sum = e_x_st.clamp_min(1e-6)
            x_mean = (ex * bx).sum(dim=-1) / wx_sum
            var_x = (ex * (bx - x_mean.unsqueeze(-1)).square()).sum(dim=-1) / wx_sum
            wy_sum = e_y_st.clamp_min(1e-6)
            y_mean = (ey_full * by).sum(dim=-1) / wy_sum
            var_y = (ey_full * (by - y_mean.unsqueeze(-1)).square()).sum(dim=-1) / wy_sum
            var_tot = torch.where(has_y & (e_y_st > 0.0), var_x + var_y, 2.0 * var_x)
            r_perp = torch.sqrt(var_tot.clamp_min(0.0) + 1e-4)

            x0_pv = self._interp_x0(v_pv_mm[:, 2].float())  # (B,)
            dz_st = self.st_z_mm[None, :] - v_pv_mm[:, 2:3].float()
            gate_ds = torch.sigmoid((dz_st + 15.0) / 10.0)
            t_l = (self.st_x0[None, :] - x0_pv[:, None]).clamp_min(0.10)  # (B, 54)

            e_ds = e_st * gate_ds
            e_vis = e_ds.sum(dim=-1)  # (B,)
            has_cal = (e_vis > 1.0).float()
            f_l = e_ds / e_vis.unsqueeze(-1).clamp_min(1e-6)  # (B, 54)

            # Longo-Sestili Gamma EM shower profile
            b_em = self.b_em_param.float().clamp(0.20, 1.20)
            a_em = 1.0 + b_em * torch.log((e_vis / W_CRITICAL_ENERGY_MEV).clamp_min(2.0))
            log_q_raw = (a_em[:, None] - 1.0) * torch.log(t_l) - b_em * t_l + torch.log(gate_ds.clamp_min(1e-5))
            log_q_raw = log_q_raw - log_q_raw.amax(dim=-1, keepdim=True)
            q_em = torch.softmax(log_q_raw, dim=-1)

            d_kl_em = (f_l * (torch.log(f_l + 1e-6) - torch.log(q_em + 1e-6))).sum(dim=-1).clamp(0.0, 15.0)
            early_moliere_mask = torch.sigmoid((8.0 - t_l) / 2.0)
            excess_r = F.relu((r_perp - 1.8 * W_MOLIERE_RADIUS_MM) / W_MOLIERE_RADIUS_MM)
            moliere_pen = (f_l * early_moliere_mask * excess_r.square()).sum(dim=-1).clamp(0.0, 20.0)

            # Cumulative shower-free MIP penetration depth t_10 (in X0)
            dt_step = torch.empty_like(t_l)
            dt_step[:, 0] = t_l[:, 0].clamp(0.1, 6.0)
            dt_step[:, 1:] = (t_l[:, 1:] - t_l[:, :-1]).clamp(0.1, 6.0)
            f_cum = torch.cumsum(f_l, dim=-1)
            t_10 = (dt_step * torch.sigmoid((0.10 - f_cum) / 0.03) * gate_ds).sum(dim=-1)
            l_mip_pen = torch.log1p(t_10.clamp_min(0.0))

            f_early_em = (f_l * torch.sigmoid((6.0 - t_l) / 1.5)).sum(dim=-1)
            f_tail_pen = (f_l * torch.sigmoid((t_l - 35.0) / 3.0)).sum(dim=-1)

            raw_5 = torch.stack([
                torch.log1p(d_kl_em) / 2.5,
                l_mip_pen / 2.5,
                torch.log1p(moliere_pen) / 2.0,
                f_early_em,
                f_tail_pen,
            ], dim=-1) * has_cal.unsqueeze(-1)
            scales = F.softplus(self.law_norm_scale.float()).clamp(0.2, 3.0)
            return {
                'law1_feat': raw_5 * scales[None, :],
                'f_l': f_l,
                't_l': t_l,
                'e_vis_mev': e_vis,
                'has_cal': has_cal,
                'st_x0': self.st_x0,
            }


class DisplacementAnchoredHemisphereModule(nn.Module):
    """Law 3: Dynamic displacement-anchored lepton/recoil decomposition & spectrometer consistency."""

    def __init__(self, geometry: dict, bounds: dict, stations: dict) -> None:
        super().__init__()
        self.n_dbl = int(stations['n_doublets'])
        self.n_bars = int(stations['n_bars'])
        self.x_pix_center = float(bounds['x_pix_center'])
        self.x_scint_center = float(stations['x_scint_center'])
        self.y_scint_center = float(stations['y_scint_center'])
        self.z_trk0 = float(TRACKER_Z_MM[0])

        dbl = np.asarray(stations['doublet_stations'], dtype=np.int64)
        st_z = np.asarray(stations['station_z_mm'], dtype=np.float32)
        self.register_buffer('dbl_idx', torch.from_numpy(dbl))
        self.register_buffer('dbl_z_mm', torch.from_numpy(st_z[dbl]))
        self.register_buffer('bar_x_mm', torch.from_numpy(stations['bar_x_centers_mm']))
        self.register_buffer('bar_y_mm', torch.from_numpy(stations['bar_y_centers_mm']))

        self.cone_radius_disp = nn.Parameter(torch.tensor(35.0, dtype=torch.float32))
        self.law_norm_scale = nn.Parameter(torch.ones(8, dtype=torch.float32))

    def forward(
        self,
        grid_x_e: torch.Tensor,
        grid_y_e: torch.Tensor,
        v_pv_mm: torch.Tensor,
        trk_kin: torch.Tensor,
        trk_morph: torch.Tensor,
        trk_mask: torch.Tensor,
        pi_sv: torch.Tensor,
        s_d0: torch.Tensor,
        trk_cand: torch.Tensor,
        trk_cand_mask: torch.Tensor,
        ipt_interlock: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute 8-dim Law-3 features (B, 8) purely from observables + soft tracklet/PV state."""
        with torch.amp.autocast(device_type=grid_x_e.device.type, enabled=False):
            b = grid_x_e.shape[0]
            ex_d = grid_x_e[:, self.dbl_idx, :].float()  # (B, 42, 40) MeV
            ey_d = grid_y_e.float()  # (B, 42, 40) MeV
            ex_st = ex_d.sum(dim=-1)
            ey_st = ey_d.sum(dim=-1)
            e_st = 0.5 * (ex_st + ey_st)
            w3d = (ex_d[:, :, :, None] * ey_d[:, :, None, :]) / e_st.clamp_min(1e-6)[:, :, None, None]  # (B, 42, 40, 40)
            e_stereo = w3d.sum(dim=(1, 2, 3))
            has_stereo = (e_stereo > 0.5).float()

            x_grid = self.bar_x_mm[None, None, :, None]
            y_grid = self.bar_y_mm[None, None, None, :]
            z_dbl = self.dbl_z_mm[None, :]  # (1, 42)
            vp = v_pv_mm.float()

            # Lab-frame pseudo-transverse momentum q_T per 3D voxel relative to calibrated v_pv_mm
            dx_v = x_grid - vp[:, 0, None, None, None]
            dy_v = y_grid - vp[:, 1, None, None, None]
            dz_v = (z_dbl - vp[:, 2, None]).clamp_min(20.0)[:, :, None, None]
            inv_r = torch.rsqrt(dx_v.square() + dy_v.square() + dz_v.square() + 1e-4)
            qx_vox = w3d * dx_v * inv_r
            qy_vox = w3d * dy_v * inv_r

            # 1. Dominant displaced target tracklet bundle direction u_T_disp
            is_tgt = trk_mask & (trk_morph[..., 1] < 0.5)
            w_disp_raw = (pi_sv.float() * torch.sigmoid(s_d0.float() - 1.2) + 0.05 * s_d0.float()) * is_tgt.float()
            w_sum = w_disp_raw.sum(dim=1, keepdim=True)
            has_disp = (w_sum.squeeze(-1) > 0.05).float()
            w_disp = w_disp_raw / w_sum.clamp_min(1e-6)  # (B, M)

            k = trk_kin.float()
            dz_pv = (vp[:, 2:3] - k[..., 2]).clamp(-350.0, 350.0)
            x_at_pv = (w_disp * (k[..., 0] + k[..., 3] * dz_pv)).sum(dim=1)  # (B,)
            y_at_pv = (w_disp * (k[..., 1] + k[..., 4] * dz_pv)).sum(dim=1)  # (B,)
            tx_disp = (w_disp * k[..., 3]).sum(dim=1)
            ty_disp = (w_disp * k[..., 4]).sum(dim=1)

            t_disp_norm = torch.sqrt(tx_disp.square() + ty_disp.square() + 1e-8)
            ux_disp = tx_disp / t_disp_norm
            uy_disp = ty_disp / t_disp_norm

            dz_ray = (z_dbl - vp[:, 2:3]).clamp_min(0.0)
            x_ray_disp = x_at_pv[:, None] + tx_disp[:, None] * dz_ray
            y_ray_disp = y_at_pv[:, None] + ty_disp[:, None] * dz_ray
            dr_disp = torch.sqrt(
                (x_grid - x_ray_disp[:, :, None, None]).square()
                + (y_grid - y_ray_disp[:, :, None, None]).square()
                + 1e-4
            )
            r_cone = self.cone_radius_disp.float().clamp(20.0, 60.0)
            m_cone = torch.sigmoid((r_cone - dr_disp) / 5.0)
            m_core15 = torch.sigmoid((15.0 - dr_disp) / 4.0)
            m_cone60 = torch.sigmoid((60.0 - dr_disp) / 5.0)

            # Dig out displaced prong corridor -> residual hadronic recoil Q_T_had
            qx_cand = (qx_vox * m_cone).sum(dim=(1, 2, 3))
            qy_cand = (qy_vox * m_cone).sum(dim=(1, 2, 3))
            qx_had = (qx_vox * (1.0 - m_cone)).sum(dim=(1, 2, 3))
            qy_had = (qy_vox * (1.0 - m_cone)).sum(dim=(1, 2, 3))
            q_had_mag = _safe_hypot(qx_had, qy_had, eps=1e-10)
            q_cand_mag = _safe_hypot(qx_cand, qy_cand, eps=1e-10)

            # Recoil opposition C_hem = -cos(dphi(u_T_disp, Q_T_had))
            valid_hem = (has_disp > 0.5) & (has_stereo > 0.5) & (q_had_mag > 0.5) & (t_disp_norm > 1e-3)
            cos_disp_had = (ux_disp * qx_had + uy_disp * qy_had) / q_had_mag.clamp_min(1e-4)
            c_hem_opp = torch.where(valid_hem, (-cos_disp_had).clamp(-1.0, 1.0), torch.zeros_like(q_had_mag))
            sin_disp_had = torch.abs(ux_disp * qy_had - uy_disp * qx_had) / q_had_mag.clamp_min(1e-4)
            acoplanarity_disp = torch.where(valid_hem, sin_disp_had.clamp(0.0, 1.0), torch.zeros_like(q_had_mag))

            # Prong isolation I_iso = 1 - E_annulus(15..60 mm) / E_cone(<60 mm)
            e_c15 = (w3d * m_core15).sum(dim=(1, 2, 3))
            e_c60 = (w3d * m_cone60).sum(dim=(1, 2, 3))
            i_iso_prong = torch.where(
                (has_disp > 0.5) & (e_c60 > 0.5),
                (e_c15 / e_c60.clamp_min(1e-4)).clamp(0.0, 1.0),
                torch.zeros_like(e_c15),
            )

            # Missing p_T fraction (using total calibrated cal vectors or displaced + recoil)
            q_tot_mag = _safe_hypot(qx_cand + qx_had, qy_cand + qy_had, eps=1e-10)
            pt_miss_frac = torch.where(
                has_stereo > 0.5,
                (q_tot_mag / (q_cand_mag + q_had_mag + 1.0)).clamp(0.0, 1.0),
                torch.zeros_like(q_tot_mag),
            )

            # 2. Bore- & IPT-verified Spectrometer Muon kinematics
            cand = trk_cand.float()  # (B, K, 14)
            cm = trk_cand_mask  # (B, K)
            is_bore_3st = (cand[..., 0] > 0.75) & cm
            ipt_match = ipt_interlock[..., 0].float()  # exp(-0.5*min_chi2/9)
            any_ipt = ipt_interlock[..., 3].float()
            in_ipt_ap = ipt_interlock[..., 2].float()
            # If track points inside IPT aperture and IPT has stubs, require IPT match; else fall back to bore quality
            ipt_gate = torch.where(
                (any_ipt > 0.5) & (in_ipt_ap > 0.5),
                0.15 + 0.85 * ipt_match,
                torch.ones_like(ipt_match),
            )
            cand_score = torch.where(is_bore_3st, cand[..., 1].clamp(0.0, 1.0) * ipt_gate, torch.zeros_like(ipt_match))
            mu_bore_ipt_conf = cand_score.max(dim=1).values.clamp(0.0, 1.0)
            has_bore_mu = is_bore_3st.any(dim=1).float()

            best_k = cand_score.argmax(dim=1)  # (B,)
            b_idx = torch.arange(b, device=cand.device)
            c_best = cand[b_idx, best_k]  # (B, 14)

            x_trk0 = self.x_pix_center + 150.0 * c_best[:, 2]
            y_trk0 = self.y_scint_center + 150.0 * c_best[:, 3]
            tx_mu = 0.1 * c_best[:, 4]
            ty_entry = 0.1 * c_best[:, 5]
            sy_signed_mm = SIGMA_SAGITTA_Y_MM * torch.sinh((5.0 * c_best[:, 6]).clamp(-9.0, 9.0))
            sy_abs_mm = 0.05 * torch.expm1((5.0 * c_best[:, 13]).clamp(0.0, 10.0))
            p_mu_mev = (SAGITTA_CONST_MEV_MM / (sy_abs_mm + 0.015)).clamp(100.0, 5.0e6) * has_bore_mu
            kappa = torch.where(has_bore_mu > 0.5, -sy_signed_mm / SAGITTA_INTEGRAL_MM2, torch.zeros_like(sy_signed_mm))
            ty_mu = ty_entry - kappa * MAGNET0_LENGTH_MM

            # Back-extrapolate muon through Magnet 0 to dig out muon corridor for E_had
            dz_spec = z_dbl - self.z_trk0
            x_ray_mu = x_trk0[:, None] + tx_mu[:, None] * dz_spec
            y_ray_lin = y_trk0[:, None] + ty_entry[:, None] * dz_spec
            u_m0 = (MAGNET0_Z_MAX_MM - z_dbl).clamp(min=0.0, max=MAGNET0_LENGTH_MM)
            y_bend = torch.where(
                z_dbl <= MAGNET0_Z_MIN_MM,
                -kappa[:, None] * MAGNET0_LENGTH_MM * (z_dbl - MAGNET0_Z_CENTER_MM),
                0.5 * kappa[:, None] * (u_m0 * u_m0),
            )
            y_ray_mu = y_ray_lin + y_bend
            dr_mu = torch.sqrt(
                (x_grid - x_ray_mu[:, :, None, None]).square()
                + (y_grid - y_ray_mu[:, :, None, None]).square()
                + 1e-4
            )
            m_mu_corridor = torch.sigmoid((45.0 - dr_mu) / 5.0)
            # Calibrated hadron recoil energy excluding muon ionization corridor (12.5 MeV scale ~ EM/had MIP calibration)
            e_had_mev = 12.5 * (w3d * (1.0 - m_mu_corridor)).sum(dim=(1, 2, 3))
            log_pmu_ehad = torch.where(
                has_bore_mu > 0.5,
                torch.asinh(torch.log((p_mu_mev + 200.0) / (e_had_mev + 200.0))) / 3.0,
                torch.zeros_like(p_mu_mev),
            )

            # Muon vs displaced-tracklet azimuthal opposition (high for nu_mu_charm!)
            t_mu_norm = torch.sqrt(tx_mu.square() + ty_mu.square() + 1e-8)
            ux_mu = tx_mu / t_mu_norm
            uy_mu = ty_mu / t_mu_norm
            valid_mu_disp = (has_bore_mu > 0.5) & (has_disp > 0.5) & (t_mu_norm > 1e-4) & (t_disp_norm > 1e-3)
            cos_mu_disp = (ux_mu * ux_disp + uy_mu * uy_disp).clamp(-1.0, 1.0)
            mu_disp_opp = torch.where(valid_mu_disp, -cos_mu_disp, torch.zeros_like(cos_mu_disp))

            # Exact Y-mirror-odd signed curvature indicator for mu- vs mu+
            mu_charge_signed = torch.tanh(sy_signed_mm / 0.8) * has_bore_mu

            raw_8 = torch.stack([
                c_hem_opp,
                i_iso_prong,
                acoplanarity_disp,
                pt_miss_frac,
                mu_bore_ipt_conf,
                log_pmu_ehad,
                mu_disp_opp,
                mu_charge_signed,
            ], dim=-1)
            scales = F.softplus(self.law_norm_scale.float()).clamp(0.2, 3.0)
            return {
                'law3_feat': raw_8 * scales[None, :],
                'has_disp': has_disp,
                'has_bore_mu': has_bore_mu,
            }


class Explicit12HypothesisEngine(nn.Module):
    """Differentiable 12-hypothesis physical log-likelihood layer + bounded neural residual."""

    def __init__(self, in_dim: int, head_dropout: float = 0.15, abl_unconstrained: bool = False) -> None:
        super().__init__()
        self.abl_unconstrained = bool(abl_unconstrained)
        a_init, b_init = _build_initial_physics_matrix()
        self.register_buffer('A_init', a_init)
        self.register_buffer('b_init', b_init)
        # Fixed pure-observation reference scales (mean[16]=0, mean[17]=0 for exact Y-mirror symmetry)
        self.register_buffer(
            'phi_ref_mean',
            torch.tensor(
                [0.80, 0.80, 0.40, 0.20, 0.45, 0.70, 0.80, 2.50, 0.08, 0.55, -0.10, 0.30, 0.55, 1.10, 0.11, 0.04, 0.0, 0.0],
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            'phi_ref_std',
            torch.tensor(
                [0.35, 0.40, 0.35, 0.22, 0.35, 0.55, 1.00, 0.90, 0.36, 0.45, 0.85, 0.26, 0.50, 0.25, 0.35, 0.30, 0.32, 0.38],
                dtype=torch.float32,
            ),
        )

        # Residual mask: m_res[NC_cc] = 0.0 (slot 11 strictly anchored to physics laws!)
        m_res = torch.ones(N_HYPOTHESES, dtype=torch.float32)
        m_res[11] = 0.0
        self.register_buffer('m_res', m_res)

        if self.abl_unconstrained:
            self.A_phys = None
            self.b_phys = None
            self.alpha_res = None
            self.res_mlp = None
            self.unconstrained_head4 = nn.Sequential(
                nn.LayerNorm(in_dim + PHI_LAW_DIM),
                nn.Linear(in_dim + PHI_LAW_DIM, 256),
                nn.SiLU(),
                nn.Dropout(head_dropout),
                nn.Linear(256, 4),
            )
            self.unconstrained_charm_head = nn.Sequential(
                nn.LayerNorm(in_dim + PHI_LAW_DIM),
                nn.Linear(in_dim + PHI_LAW_DIM, 128),
                nn.SiLU(),
                nn.Linear(128, 1),
            )
            self.tau_charm_head = None
        else:
            self.A_phys = nn.Parameter(a_init.clone())
            self.b_phys = nn.Parameter(b_init.clone())
            self.alpha_res = nn.Parameter(torch.tensor(0.5413, dtype=torch.float32))
            self.res_mlp = nn.Sequential(
                nn.LayerNorm(in_dim + PHI_LAW_DIM),
                nn.Linear(in_dim + PHI_LAW_DIM, 256),
                nn.SiLU(),
                nn.Dropout(head_dropout),
                nn.Linear(256, N_HYPOTHESES),
            )
            nn.init.normal_(self.res_mlp[-1].weight, std=1.0e-2)
            nn.init.zeros_(self.res_mlp[-1].bias)
            self.unconstrained_head4 = None
            self.unconstrained_charm_head = None
            # Conditional tau-associated charm branch P(charm | tau, x) = sigma(z_tau_c)
            self.tau_charm_head = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, 128),
                nn.SiLU(),
                nn.Linear(128, 1),
            )

    def physics_prior_penalty(self) -> torch.Tensor:
        if self.abl_unconstrained:
            return self.A_init.sum() * 0.0
        assert self.A_phys is not None and self.b_phys is not None
        return 0.05 * ((self.A_phys.float() - self.A_init).square().mean() + (self.b_phys.float() - self.b_init).square().mean())

    def forward(self, z_backbone: torch.Tensor, phi_law: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.amp.autocast(device_type=z_backbone.device.type, enabled=False):
            zf = z_backbone.float()
            phi = phi_law.float()
            phi_norm = ((phi - self.phi_ref_mean[None, :]) / self.phi_ref_std[None, :]).clamp(-5.0, 5.0)

            if self.abl_unconstrained:
                assert self.unconstrained_head4 is not None and self.unconstrained_charm_head is not None
                joint = torch.cat([zf, phi_norm], dim=-1)
                logits4 = self.unconstrained_head4(joint)
                probs4 = F.softmax(logits4, dim=-1)
                # Map 4-class logits back to 7-class contract symmetrically for uniform output interface
                log_half = math.log(0.5)
                log_third = math.log(1.0 / 3.0)
                logits7 = torch.stack([
                    logits4[:, 0],
                    logits4[:, 1] + log_half,
                    logits4[:, 1] + log_half,
                    logits4[:, 2] + log_third,
                    logits4[:, 2] + log_third,
                    logits4[:, 2] + log_third,
                    logits4[:, 3],
                ], dim=-1)
                charm_logit = self.unconstrained_charm_head(joint).squeeze(-1)
                p_all_charm = torch.sigmoid(charm_logit)
                return {
                    'logits7': logits7,
                    'probs4': probs4,
                    'probs12': zf.new_zeros((zf.shape[0], N_HYPOTHESES)),
                    'l_phys12': zf.new_zeros((zf.shape[0], N_HYPOTHESES)),
                    'r_res12': zf.new_zeros((zf.shape[0], N_HYPOTHESES)),
                    'charm_logit': charm_logit,
                    'q_bkg_charm': p_all_charm,
                    'p_all_charm': p_all_charm,
                    'z_tau_c': torch.zeros_like(charm_logit),
                }

            assert (
                self.A_phys is not None
                and self.b_phys is not None
                and self.alpha_res is not None
                and self.res_mlp is not None
                and self.tau_charm_head is not None
            )
            l_phys12 = F.linear(phi_norm, self.A_phys.float(), self.b_phys.float())  # (B, 12)
            joint = torch.cat([zf, phi_norm], dim=-1)
            r_res12 = self.res_mlp(joint)  # (B, 12)
            gamma_res = F.softplus(self.alpha_res.float())
            ell12 = l_phys12 + gamma_res * self.m_res[None, :] * r_res12

            # Exact log-space marginalization from 12 sub-hypotheses -> 7 canonical classes
            logits7 = torch.stack([
                torch.logsumexp(ell12[:, [0, 1]], dim=-1),          # e_CC = e_p + e_c
                torch.logsumexp(ell12[:, [2, 4]], dim=-1),          # mu-_CC = mu-_p + mu-_c
                torch.logsumexp(ell12[:, [3, 5]], dim=-1),          # mu+_CC = mu+_p + mu+_c
                ell12[:, 6],                                        # tau_e
                ell12[:, 7],                                        # tau_mu
                torch.logsumexp(ell12[:, [8, 9]], dim=-1),          # tau_had = tau_1h + tau_3h
                torch.logsumexp(ell12[:, [10, 11]], dim=-1),        # NC = NC_light + NC_cc
            ], dim=-1)
            probs12 = F.softmax(ell12, dim=-1)
            probs4 = marginal_four_from_seven(logits7, is_probs=False)

            # Conflict-free charm decomposition:
            # P_bkg_charm = P(e_c) + P(mu-_c) + P(mu+_c) + P(NC_cc)
            p_bkg_charm = probs12[:, 1] + probs12[:, 4] + probs12[:, 5] + probs12[:, 11]
            p_tau = probs4[:, 2].clamp(0.0, 1.0 - 1e-6)
            q_bkg_charm = (p_bkg_charm / (1.0 - p_tau + 1e-6)).clamp(1e-6, 1.0 - 1e-6)
            z_tau_c = self.tau_charm_head(zf).squeeze(-1)
            p_all_charm = (p_bkg_charm + probs4[:, 2] * torch.sigmoid(z_tau_c)).clamp(1e-6, 1.0 - 1e-6)
            charm_logit = torch.log(p_all_charm) - torch.log1p(-p_all_charm)

            return {
                'logits7': logits7,
                'probs4': probs4,
                'probs12': probs12,
                'l_phys12': l_phys12,
                'r_res12': r_res12,
                'charm_logit': charm_logit,
                'q_bkg_charm': q_bkg_charm,
                'p_all_charm': p_all_charm,
                'z_tau_c': z_tau_c,
            }


class _RegimeAwareSymmetricVertexRefiner(nn.Module):
    """Regime-aware Y-mirror-equivariant two-stage (69+54 longitudinal coarse + near-vertex fine) PV refiner."""

    def __init__(self, tok_dim: int, vtx_dim: int = 96, heads: int = 4) -> None:
        super().__init__()
        self.prof_proj = nn.Linear(tok_dim, vtx_dim)
        self.cal_proj = nn.Linear(tok_dim, vtx_dim)
        self.rel_z_proj = nn.Linear(3, vtx_dim)
        self.coarse_q = nn.Parameter(torch.randn(1, 4, vtx_dim) * 0.02)
        self.coarse_cross = ScaledQKNormAttention(dim=vtx_dim, heads=heads, init_scale=8.0, drop_path=0.05)
        self.coarse_head = nn.Sequential(
            nn.LayerNorm(vtx_dim + 8),
            nn.Linear(vtx_dim + 8, vtx_dim),
            nn.SiLU(),
        )
        self.delta_coarse_xz = nn.Linear(vtx_dim, 2)

        self.pix_proj = nn.Linear(tok_dim, vtx_dim)
        self.fine_rel_proj = nn.Sequential(
            nn.Linear(6, vtx_dim),
            nn.LayerNorm(vtx_dim),
            nn.SiLU(),
        )
        self.fine_q = nn.Parameter(torch.randn(1, 4, vtx_dim) * 0.02)
        self.fine_cross = ScaledQKNormAttention(dim=vtx_dim, heads=heads, init_scale=10.0, drop_path=0.05)
        self.y_cluster_gate = nn.Linear(vtx_dim, 1)

        self.fine_head = nn.Sequential(
            nn.LayerNorm(2 * vtx_dim),
            nn.Linear(2 * vtx_dim, vtx_dim),
            nn.SiLU(),
        )
        self.delta_fine_xz = nn.Linear(vtx_dim, 2)
        self.y_blend_gate = nn.Linear(vtx_dim, 1)
        self.log_scale_xyz = nn.Linear(vtx_dim, 3)

        nn.init.normal_(self.delta_coarse_xz.weight, std=1e-3)
        nn.init.zeros_(self.delta_coarse_xz.bias)
        nn.init.normal_(self.delta_fine_xz.weight, std=1e-3)
        nn.init.zeros_(self.delta_fine_xz.bias)
        nn.init.constant_(self.y_blend_gate.bias, -1.0)
        nn.init.normal_(self.log_scale_xyz.weight, std=1e-3)
        nn.init.zeros_(self.log_scale_xyz.bias)

    def forward(
        self,
        z_seq: torch.Tensor,
        z_act: torch.Tensor,
        layer_z_69_mm: torch.Tensor,
        cal_st_tok: torch.Tensor,
        cal_st_mask: torch.Tensor,
        cal_st_z_mm: torch.Tensor,
        anchor_meta_feat: torch.Tensor,
        v_prior_mm: torch.Tensor,
        is_front_p: torch.Tensor,
        u_pix: torch.Tensor | None = None,
        pix_xyz_mm: torch.Tensor | None = None,
        pix_mask: torch.Tensor | None = None,
        pix_plane: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b = int(z_seq.shape[0])
        prior = v_prior_mm.float()

        dz_69 = layer_z_69_mm[None, :] - prior[:, 2:3]
        rel_69 = torch.stack([
            dz_69 / 40.0,
            torch.asinh(dz_69 / 10.0) / 3.0,
            ((dz_69 >= -25.0) & (dz_69 <= 165.0)).float(),
        ], dim=-1)
        p_69 = (self.prof_proj(z_seq) + self.rel_z_proj(rel_69.to(z_seq.dtype))) * z_act.unsqueeze(-1).to(z_seq.dtype)

        dz_54 = cal_st_z_mm[None, :] - prior[:, 2:3]
        rel_54 = torch.stack([
            dz_54 / 40.0,
            torch.asinh(dz_54 / 10.0) / 3.0,
            ((dz_54 >= -25.0) & (dz_54 <= 165.0)).float(),
        ], dim=-1)
        p_54 = (self.cal_proj(cal_st_tok) + self.rel_z_proj(rel_54.to(cal_st_tok.dtype))) * cal_st_mask.unsqueeze(-1).to(cal_st_tok.dtype)

        prof_tok = torch.cat([p_69, p_54], dim=1)
        prof_m = torch.cat([z_act, cal_st_mask], dim=1)
        any_prof = prof_m.any(dim=1, keepdim=True)
        safe_pm = prof_m.clone()
        if (~any_prof.squeeze(-1)).any():
            safe_pm[~any_prof.squeeze(-1), 0] = True
        h_c_attn = self.coarse_cross(self.coarse_q.expand(b, -1, -1), prof_tok, safe_pm).mean(dim=1)
        h_c_attn = h_c_attn * any_prof.to(h_c_attn.dtype)
        h_c = self.coarse_head(torch.cat([h_c_attn, anchor_meta_feat.to(h_c_attn.dtype)], dim=-1))

        with torch.amp.autocast(device_type=h_c.device.type, enabled=False):
            c_scale = torch.where(
                is_front_p[:, None],
                prior.new_tensor([70.0, 90.0]),
                prior.new_tensor([140.0, 260.0]),
            )
            d_xz_c = torch.tanh(self.delta_coarse_xz(h_c.float())) * c_scale
            v_coarse = torch.stack([
                prior[:, 0] + d_xz_c[:, 0],
                prior[:, 1],
                prior[:, 2] + d_xz_c[:, 1],
            ], dim=-1)

        if u_pix is not None and pix_xyz_mm is not None and pix_mask is not None and pix_xyz_mm.shape[1] > 0:
            k_sel = min(64, pix_xyz_mm.shape[1])
            c_idx, c_mask = select_near_vertex_clusters(
                pix_xyz_mm, pix_mask, v_coarse.detach(), k_vtx=k_sel, pix_plane=pix_plane
            )
            c_tok = self.pix_proj(
                torch.gather(u_pix, 1, c_idx.unsqueeze(-1).expand(-1, -1, u_pix.shape[-1]))
            )
            c_xyz = torch.gather(pix_xyz_mm.float(), 1, c_idx.unsqueeze(-1).expand(-1, -1, 3))
            rel = c_xyz - v_coarse[:, None, :]
            dx = rel[..., 0]
            dy = rel[..., 1]
            dz = rel[..., 2]
            abs_dy = dy.abs()
            rel_inv = torch.stack([
                dx / 15.0,
                abs_dy / 15.0,
                dz / 35.0,
                torch.asinh(dx / 2.0) / 3.0,
                torch.asinh(abs_dy / 2.0) / 3.0,
                torch.asinh(dz / 8.0) / 3.0,
            ], dim=-1)
            c_in = (c_tok + self.fine_rel_proj(rel_inv.to(c_tok.dtype))) * c_mask.unsqueeze(-1).to(c_tok.dtype)
            any_c = c_mask.any(dim=1, keepdim=True)
            safe_cm = c_mask.clone()
            if (~any_c.squeeze(-1)).any():
                safe_cm[~any_c.squeeze(-1), 0] = True
            h_f_attn = self.fine_cross(self.fine_q.expand(b, -1, -1), c_in, safe_cm).mean(dim=1)
            h_f_safe = torch.where(any_c, h_f_attn, h_c_attn)

            w_y = torch.softmax(
                self.y_cluster_gate(c_in).squeeze(-1).float().masked_fill(~safe_cm, float('-inf')),
                dim=1,
            ) * c_mask.float()
            dy_cent = (w_y * dy.clamp(-30.0, 30.0)).sum(dim=1)
            has_fine_y = any_c.squeeze(-1).float()
        else:
            h_f_safe = h_c_attn
            dy_cent = v_coarse.new_zeros((b,))
            has_fine_y = v_coarse.new_zeros((b,))

        h_comb = self.fine_head(torch.cat([h_c, h_f_safe], dim=-1))
        with torch.amp.autocast(device_type=h_comb.device.type, enabled=False):
            hcf = h_comb.float()
            f_scale = torch.where(
                is_front_p[:, None],
                v_coarse.new_tensor([18.0, 30.0]),
                v_coarse.new_tensor([25.0, 60.0]),
            )
            d_xz_f = torch.tanh(self.delta_fine_xz(hcf)) * f_scale
            dy_f = torch.sigmoid(self.y_blend_gate(hcf).squeeze(-1)) * dy_cent * has_fine_y
            v_anchor = torch.stack([
                v_coarse[:, 0] + d_xz_f[:, 0],
                v_coarse[:, 1] + dy_f,
                v_coarse[:, 2] + d_xz_f[:, 1],
            ], dim=-1)
            base_sc = torch.where(
                is_front_p[:, None],
                v_coarse.new_tensor([0.35, 0.35, 3.5]),
                v_coarse.new_tensor([1.20, 1.20, 8.5]),
            )
            sigma_anc = (torch.exp(self.log_scale_xyz(hcf).clamp(-4.5, 4.0)) * base_sc).clamp(0.03, 160.0)
        return v_anchor, sigma_anc


class TauKinemaDiffNet(NextGenFlavourModel):
    """PINPOINT-Native Differentiable Physics-Hypothesis Likelihood Network (~5.35M params)."""

    VALID_ABLATIONS = (
        'full',
        'no_hemisphere_opposition',
        'no_gamma_em_profile',
        'no_stub_conversion_law',
        'unconstrained_mlp_head',
    )

    def __init__(self, geometry: dict, config: dict | None = None) -> None:
        super().__init__()
        cfg = dict(config or {})
        self.config = cfg
        self.ablation = str(cfg.get('ablation', 'full'))
        if self.ablation not in self.VALID_ABLATIONS:
            raise ValueError(f'Unknown ablation {self.ablation!r}; expected one of {self.VALID_ABLATIONS}')
        self.n2_vtx_mode = str(cfg.get('n2_vtx_mode', 'v1'))
        if self.n2_vtx_mode not in ('v1', 'v2'):
            raise ValueError(f'Unknown n2_vtx_mode {self.n2_vtx_mode!r}; expected v1 or v2')

        bounds = detector_bounds(geometry)
        stations = static_scint_stations(geometry, bounds)
        dim = int(cfg.get('dim', 168))
        heads = int(cfg.get('heads', 8))
        drop_path = float(cfg.get('drop_path', 0.10))
        head_dropout = float(cfg.get('head_dropout', 0.15))
        n_global_layers = int(cfg.get('n_global_layers', 2))
        n_queries = int(cfg.get('n_queries', 12))

        self.dim = dim
        self.heads = heads
        self.n_st = int(stations['n_stations'])
        self.n_dbl = int(stations['n_doublets'])
        self.n_bars = int(stations['n_bars'])
        self.n_queries = n_queries
        self.max_tracklets = int(cfg.get('max_tracklets', 64))

        pix_x0 = torch.tensor(
            [cumulative_radiation_length(float(z), geometry) for z in bounds['pixel_Zpos']],
            dtype=torch.float32,
        )
        self.register_buffer('pix_plane_x0', pix_x0)
        self.register_buffer(
            'pix_plane_z_mm',
            torch.tensor(np.asarray(bounds['pixel_Zpos'], dtype=np.float32), dtype=torch.float32),
        )
        self.register_buffer(
            'default_center_mm',
            torch.tensor([bounds['x_pix_center'], bounds['y_scint_center'], bounds['z_mid']], dtype=torch.float32),
        )
        self.register_buffer('panel_to_xslot', torch.from_numpy(stations['panel_to_xslot']))
        self.register_buffer('panel_to_yslot', torch.from_numpy(stations['panel_to_yslot']))
        self.register_buffer('panel_to_station', torch.from_numpy(stations['panel_to_station']))
        self.register_buffer('station_z_mm', torch.from_numpy(stations['station_z_mm']))

        # 1. Pixel Clusters + Micro-Patches + Plane Summary
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

        # 2. Soft Multi-Tracklets + Translation-Invariant DCA + Student-t PV/SV
        self.trk_in = nn.Sequential(
            nn.Linear(TRK_KIN_DIM + TRK_MORPH_DIM, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.trk_pair_mlp = nn.Sequential(nn.Linear(PAIR_TRK_FEAT_DIM, 32), nn.SiLU(), nn.Linear(32, heads))
        self.trk_attn = AttentionStack(
            dim=dim, n_layers=2, heads=heads, init_scale=12.0, max_scale=50.0, drop_path=drop_path
        )
        if self.n2_vtx_mode == 'v2':
            self.vtx_anchor = None
            self.vtx_refiner_v2 = _RegimeAwareSymmetricVertexRefiner(tok_dim=dim, vtx_dim=96, heads=4)
        else:
            self.vtx_anchor = VertexAnchorPredictor(dim=dim, heads=heads)
            self.vtx_refiner_v2 = None
        self.vtx_layer = DifferentiableStudentTVertexLayer(hidden_dim=dim, df=3.0, n_iters=4)
        self.pv_regime_scale = nn.Sequential(
            nn.Linear(3 * dim + 16 + 6, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
        )
        nn.init.zeros_(self.pv_regime_scale[-1].weight)
        nn.init.zeros_(self.pv_regime_scale[-1].bias)
        self.trk_ip_proj = nn.Sequential(nn.Linear(4, dim), nn.LayerNorm(dim), nn.SiLU())
        self.trk_post_attn = AttentionStack(
            dim=dim, n_layers=1, heads=heads, init_scale=12.0, max_scale=50.0, drop_path=drop_path
        )
        self.trk_pool_gate = nn.Linear(dim, 1)

        # 3. Stereo Calorimeter Nexus
        self.stereo_bars = StereoBarNexus(stations, dim=dim, heads=heads, drop_path=drop_path)

        # 4. IPT-Interlocked Spectrometer Encoder
        self.spec_cand_in = nn.Sequential(
            nn.Linear(14 + 4, 96),
            nn.LayerNorm(96),
            nn.SiLU(),
            nn.Linear(96, 96),
            nn.SiLU(),
        )
        self.spec_cand_attn = AttentionStack(
            dim=96, n_layers=2, heads=4, init_scale=8.0, max_scale=40.0, drop_path=drop_path * 0.5
        )
        self.spec_hit_in = nn.Sequential(
            nn.Linear(6, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.spec_hit_score = nn.Linear(64, 1)
        self.spec_fusion = nn.Sequential(
            nn.Linear(192 + 128, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
        )

        # 5. Longitudinal 69-Plane Profile Encoder
        n_pix_p = len(bounds['pixel_Zpos'])
        all_z = np.concatenate([
            np.asarray(bounds['pixel_Zpos'], dtype=np.float64),
            np.asarray(stations['station_z_mm'], dtype=np.float64),
        ])
        all_x0 = np.concatenate([
            pix_x0.numpy().astype(np.float64),
            np.asarray(stations['station_x0'], dtype=np.float64),
        ])
        is_pix = np.r_[np.ones(n_pix_p, dtype=np.float32), np.zeros(self.n_st, dtype=np.float32)]
        z_order = np.argsort(all_z, kind='mergesort')
        z_geom = np.column_stack([
            (all_z[z_order] - float(bounds['z_mid'])) / 1000.0,
            all_x0[z_order] / 50.0,
            is_pix[z_order],
            1.0 - is_pix[z_order],
        ]).astype(np.float32)
        self.register_buffer('z_order_idx', torch.from_numpy(z_order).long())
        self.register_buffer('z_plane_geom', torch.from_numpy(z_geom))
        if self.n2_vtx_mode == 'v2':
            self.register_buffer('layer_z_69_mm', torch.from_numpy(all_z[z_order].astype(np.float32)))
            self.register_buffer('cal_st_z_mm', torch.from_numpy(np.asarray(stations['station_z_mm'], dtype=np.float32)))
        self.z_pos_proj = nn.Linear(4, dim)
        self.z_profile_encoder = AttentionStack(
            dim=dim, n_layers=2, heads=heads, init_scale=10.0, max_scale=50.0, drop_path=drop_path
        )
        self.z_pool_gate = nn.Linear(dim, 1)

        # 6. Global Perceiver Cross-Modal Fusion
        self.latent_queries = nn.Parameter(torch.randn(1, n_queries, dim) * 0.02)
        self.cobs_query = nn.Sequential(
            nn.Dropout(0.30),
            nn.Linear(16, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
        )
        self.global_blocks = nn.ModuleList([
            PerceiverCrossBlock(dim=dim, heads=heads, init_scale=12.0, max_scale=50.0, drop_path=drop_path)
            for _ in range(n_global_layers)
        ])
        self.pool_gate = nn.Linear(dim, 1)
        self.vtx_stat_proj = nn.Sequential(
            nn.Linear(8, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
        )
        self.global_fusion = nn.Sequential(
            nn.LayerNorm(6 * dim),
            nn.Linear(6 * dim, 3 * dim),
            nn.SiLU(),
        )

        # 7. Three Differentiable PINPOINT Physics Law Modules
        self.cascade_fitter = TungstenCascadeProfileFitter(geometry, stations)
        self.hemisphere_mod = DisplacementAnchoredHemisphereModule(geometry, bounds, stations)
        self.w_stub = nn.Parameter(torch.tensor(1.5, dtype=torch.float32))
        self.w_gap = nn.Parameter(torch.tensor(1.8, dtype=torch.float32))
        self.law2_scale = nn.Parameter(torch.ones(5, dtype=torch.float32))

        # 8. 12-Hypothesis Physics Likelihood Engine + Kinematics Head
        self.hyp_engine = Explicit12HypothesisEngine(
            in_dim=3 * dim,
            head_dropout=head_dropout,
            abl_unconstrained=(self.ablation == 'unconstrained_mlp_head'),
        )
        self.kin_head = nn.Sequential(
            nn.LayerNorm(3 * dim),
            nn.Linear(3 * dim, 160),
            nn.SiLU(),
            nn.Linear(160, 4),
        )

    def build_param_groups(self, base_lr: float, weight_decay: float) -> list[dict]:
        decay_params: list[nn.Parameter] = []
        no_decay_params: list[nn.Parameter] = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if is_weight_decay_exempt(name, param) or any(k in name for k in TAU_KINEMA_NO_WD_KEYWORDS):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        groups: list[dict] = []
        if decay_params:
            groups.append({'params': decay_params, 'lr': float(base_lr), 'weight_decay': float(weight_decay)})
        if no_decay_params:
            groups.append({'params': no_decay_params, 'lr': float(base_lr), 'weight_decay': 0.0})
        return groups

    def learned_attention_scales(self) -> dict[str, list[float]]:
        return collect_attention_scales(self)

    def effective_attention_sharpness(self) -> dict[str, list[dict[str, float]]]:
        return collect_effective_attention_sharpness(self)

    def branch_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        groups: dict[str, list[nn.Parameter]] = {}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith(('patch_',)):
                br = 'pixel_patch_conv'
            elif name.startswith(('pix_in.', 'pix_plane_emb.')):
                br = 'pixel_clusters'
            elif name.startswith('stereo_bars.'):
                br = 'stereo_bars'
            elif name.startswith('spec_'):
                br = 'spectrometer'
            elif name.startswith(('trk_', 'vtx_layer.', 'vtx_anchor.', 'vtx_refiner_v2.', 'vtx_stat_proj.', 'pv_regime_scale.')):
                br = 'tracklet_vertex'
            elif name.startswith(('cascade_fitter.', 'hemisphere_mod.', 'w_stub', 'w_gap', 'law2_scale')):
                br = 'physics_laws'
            elif name.startswith(('latent_queries', 'cobs_query.', 'z_', 'global_blocks.', 'pool_gate.', 'global_fusion.')):
                br = 'global_perceiver'
            else:
                br = 'hypothesis_head'
            groups.setdefault(br, []).append(param)
        return groups

    def _compute_cal_prior_v2(
        self,
        batch: dict[str, torch.Tensor],
        b: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Backscatter-safe regime prior + 8-d anchor metadata feature vector for n2v2."""
        center = self.default_center_mm[None, :].expand(b, -1).to(device)
        cal_start = batch['cal_start'].float()
        cal_valid = cal_start[:, 3] > 0.5
        v_cal_prior = torch.where(
            cal_valid.unsqueeze(-1),
            torch.stack([cal_start[:, 1], cal_start[:, 2], cal_start[:, 0]], dim=-1),
            center,
        )

        ps = batch['pix_plane_summary'].float()  # (B, 15, 12)
        tgt_act_raw = ps[:, :12, 0] > 0.5
        tgt_multi = ps[:, :12, 1] >= (math.log1p(2.0) / 8.0 - 1e-4)
        near_cal = (~cal_valid[:, None]) | ((cal_start[:, 0:1] - self.pix_plane_z_mm[None, :12]) < 95.0)
        tgt_act_clean = tgt_act_raw & (tgt_multi | near_cal)
        use_clean = tgt_act_clean.any(dim=1, keepdim=True)
        tgt_act = torch.where(use_clean, tgt_act_clean, tgt_act_raw)

        has_tgt = tgt_act.any(dim=1)
        p_grid = torch.arange(12, device=device, dtype=torch.long)[None, :].expand(b, 12)
        first_tgt_p = torch.where(tgt_act, p_grid, torch.full_like(p_grid, 99)).min(dim=1).values.clamp(max=11)
        is_front_p = has_tgt & (first_tgt_p <= 5)
        is_mid_p = has_tgt & (first_tgt_p >= 6)

        b_idx = torch.arange(b, device=device)
        ps_first = ps[b_idx, first_tgt_p]
        x_si = self.default_center_mm[0] + 135.0 * ps_first[:, 7]
        y_si = self.default_center_mm[1] + 100.0 * ps_first[:, 6]
        z_si_p = self.pix_plane_z_mm[first_tgt_p]

        z_w_f = torch.where(first_tgt_p == 0, z_si_p + 7.5, z_si_p - 14.0)
        v_si_front = torch.stack([x_si, y_si, z_w_f], dim=-1)
        avr = batch['avr_pv_mm'].float() if 'avr_pv_mm' in batch else torch.zeros((b, 4), device=device)
        avr_has = avr[:, 3] > 0.5
        avr_ok = is_front_p & avr_has & ((avr[:, 2] - z_w_f).abs() < 45.0)
        v_prior_front = torch.where(avr_ok.unsqueeze(-1), avr[:, :3], v_si_front)

        si_precedes_cal = is_mid_p & cal_valid & (z_si_p <= cal_start[:, 0] + 10.0) & ((cal_start[:, 0] - z_si_p) < 85.0)
        z_est_m = torch.where(
            si_precedes_cal,
            torch.minimum(cal_start[:, 0], z_si_p - 12.5),
            torch.where(cal_valid, cal_start[:, 0], torch.where(is_mid_p, z_si_p - 30.0, center[:, 2])),
        )
        use_si_xy_m = is_mid_p & ((~cal_valid) | ((z_si_p - z_est_m).abs() <= 40.0))
        xy_est_m = torch.where(use_si_xy_m.unsqueeze(-1), torch.stack([x_si, y_si], dim=-1), v_cal_prior[:, :2])
        v_prior_mid = torch.cat([xy_est_m, z_est_m.unsqueeze(-1)], dim=-1)

        v_phys_prior = torch.where(is_front_p.unsqueeze(-1), v_prior_front, v_prior_mid)
        any_valid = cal_valid | has_tgt | avr_ok

        anchor_meta_feat = torch.stack([
            (v_phys_prior[:, 2] - self.default_center_mm[2]) / 1000.0,
            ((z_si_p - v_phys_prior[:, 2]) / 45.0).clamp(-5.0, 5.0) * has_tgt.float(),
            ((v_cal_prior[:, 2] - v_phys_prior[:, 2]) / 45.0).clamp(-5.0, 5.0) * cal_valid.float(),
            ((avr[:, 2] - v_phys_prior[:, 2]) / 30.0).clamp(-4.0, 4.0) * avr_has.float(),
            is_front_p.float(),
            is_mid_p.float(),
            avr_ok.float(),
            cal_valid.float(),
        ], dim=-1)
        return v_phys_prior, any_valid, is_front_p, is_mid_p, anchor_meta_feat

    def _compute_cal_prior(
        self,
        batch: dict[str, torch.Tensor],
        b: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute causal first-silicon + calorimeter physical prior (mm) and regime flags."""
        center = self.default_center_mm[None, :].expand(b, -1).to(device)
        cal_start = batch['cal_start'].float()
        cal_valid = cal_start[:, 3] > 0.5
        cal_xyz = torch.stack([cal_start[:, 1], cal_start[:, 2], cal_start[:, 0]], dim=-1)
        prior = torch.where(cal_valid.unsqueeze(-1), cal_xyz, center)

        ps = batch['pix_plane_summary'].float()  # (B, 15, 12)
        tgt_act = ps[:, :12, 0] > 0.5  # Target planes 0..11
        has_tgt = tgt_act.any(dim=1)
        p_grid = torch.arange(12, device=device, dtype=torch.long)[None, :].expand(b, 12)
        first_tgt_p = torch.where(tgt_act, p_grid, torch.full_like(p_grid, 99)).min(dim=1).values.clamp(max=11)
        is_front_p = has_tgt & (first_tgt_p <= 5)
        is_mid_p = has_tgt & (first_tgt_p >= 6)

        b_idx = torch.arange(b, device=device)
        ps_first = ps[b_idx, first_tgt_p]
        x_si = self.default_center_mm[0] + 135.0 * ps_first[:, 7]
        y_si = self.default_center_mm[1] + 100.0 * ps_first[:, 6]
        z_si_p = self.pix_plane_z_mm[first_tgt_p]
        # Plane 0 sits upstream of W0 (+7.5mm); Planes 1..5 sit downstream of 2 W plates (-14.0mm)
        z_front = torch.where(first_tgt_p == 0, z_si_p + 7.5, z_si_p - 14.0)
        z_mid = torch.where(cal_valid, torch.minimum(cal_start[:, 0], z_si_p - 12.5), z_si_p - 12.5)
        z_si = torch.where(first_tgt_p <= 5, z_front, z_mid)
        si_xyz = torch.stack([x_si, y_si, z_si], dim=-1)

        prior = torch.where(has_tgt.unsqueeze(-1), si_xyz, prior)
        any_valid = cal_valid | has_tgt
        if 'avr_pv_mm' in batch:
            avr = batch['avr_pv_mm'].float()
            dz_cal = avr[:, 2] - cal_start[:, 0]
            # N2 fix: restrict coarse C++ AVR prior strictly to front 30mm module (first_tgt_p <= 5)
            # so coarse 150mm C++ AVR never overrides the neural/calorimeter anchor in Interm150.
            avr_ok = is_front_p & (avr[:, 3] > 0.5) & ((~cal_valid) | ((dz_cal >= -50.0) & (dz_cal <= 150.0)))
            prior = torch.where(avr_ok.unsqueeze(-1), avr[:, :3], prior)
            any_valid = any_valid | avr_ok
        return prior, any_valid, is_front_p, is_mid_p

    def _build_bar_energy_grids(self, batch: dict[str, torch.Tensor], b: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct MeV-calibrated (B, 54, 40) X-bar and (B, 42, 40) Y-bar energy grids."""
        grid_x_e = torch.zeros((b, self.n_st, self.n_bars), device=device, dtype=torch.float32)
        grid_y_e = torch.zeros((b, self.n_dbl, self.n_bars), device=device, dtype=torch.float32)
        bar_ev = batch['bar_ev']
        if len(bar_ev) == 0:
            return grid_x_e, grid_y_e
        bar_panel = batch['bar_panel']
        bar_horiz = batch['bar_horiz']
        bar_idx = batch['bar_idx']
        bar_val = batch['bar_val'].float()
        e_mev = torch.expm1((10.0 * bar_val[:, 0]).clamp(0.0, 14.0))

        xm = bar_horiz == 0
        ym = bar_horiz == 1
        if xm.any():
            xs = self.panel_to_xslot[bar_panel[xm]]
            grid_x_e.index_put_((bar_ev[xm], xs, bar_idx[xm]), e_mev[xm], accumulate=True)
        if ym.any():
            ys = self.panel_to_yslot[bar_panel[ym]]
            grid_y_e.index_put_((bar_ev[ym], ys, bar_idx[ym]), e_mev[ym], accumulate=True)
        return grid_x_e, grid_y_e

    def forward_NextGen(
        self,
        batch: dict[str, torch.Tensor],
        return_entropy: bool = False,
    ) -> NextGenModelOutput:
        b = int(batch['c_obs'].shape[0])
        device = batch['c_obs'].device

        # 1. Pixel clusters & 32x32 patches
        pix_feat = batch['pix_feat']
        pix_mask = batch['pix_mask']
        pix_xyz = batch['pix_xyz_mm'].float()
        u_pix = self.pix_in(pix_feat) * pix_mask.unsqueeze(-1).to(pix_feat.dtype)
        if len(batch['pix_patches']):
            patch_emb = self.patch_proj(self.patch_conv(batch['pix_patches']))
            delta_patch = torch.zeros_like(u_pix, dtype=patch_emb.dtype)
            delta_patch.index_put_((batch['patch_ev'], batch['patch_cl']), patch_emb, accumulate=True)
            u_pix = u_pix + delta_patch * pix_mask.unsqueeze(-1).to(u_pix.dtype)
        pix_plane_tok = self.pix_plane_emb(batch['pix_plane_summary'])
        pix_plane_act = batch['pix_plane_summary'][..., 0] > 0.5

        # 2. Stereo Calorimeter Nexus
        bar_out = self.stereo_bars(batch)
        cal_tok = bar_out['cal_tokens']
        cal_mask = bar_out['cal_mask']

        # 3. Longitudinal 69-Plane Profile Encoder (must run BEFORE vtx_anchor to provide z_mm/X0/lambda_I!)
        z_in = torch.cat([pix_plane_tok, cal_tok[:, :self.n_st]], dim=1)[:, self.z_order_idx]
        z_act = torch.cat([pix_plane_act, cal_mask[:, :self.n_st]], dim=1)[:, self.z_order_idx]
        z_in = (z_in + self.z_pos_proj(self.z_plane_geom).unsqueeze(0).to(z_in.dtype)) * z_act.unsqueeze(-1).to(z_in.dtype)
        z_seq = self.z_profile_encoder(z_in, z_act)
        with torch.amp.autocast(device_type=device.type, enabled=False):
            z_logits = self.z_pool_gate(z_seq.float()).squeeze(-1).masked_fill(~z_act, -1e4)
            z_w = torch.softmax(z_logits, dim=1) * z_act.float()
            z_w = z_w / z_w.sum(dim=1, keepdim=True).clamp_min(1e-6)
            z_prof_glob = (z_w.unsqueeze(-1) * z_seq.float()).sum(dim=1)

        # 4. Two-Stage Neural Vertex Anchor (causal silicon+calorimeter+AVR prior -> coarse+fine PV)
        if self.n2_vtx_mode == 'v2':
            v_phys_prior, any_valid_prior, is_front_p, is_mid_p, anchor_meta_feat = self._compute_cal_prior_v2(batch, b, device)
            v_anchor, sigma_anc_v2 = self.vtx_refiner_v2(
                z_seq.to(pix_plane_tok.dtype),
                z_act,
                self.layer_z_69_mm,
                cal_tok[:, :self.n_st].to(pix_plane_tok.dtype),
                cal_mask[:, :self.n_st],
                self.cal_st_z_mm,
                anchor_meta_feat,
                v_phys_prior,
                is_front_p,
                u_pix=u_pix,
                pix_xyz_mm=pix_xyz,
                pix_mask=pix_mask,
                pix_plane=batch['pix_plane'],
            )
        else:
            v_phys_prior, any_valid_prior, is_front_p, is_mid_p = self._compute_cal_prior(batch, b, device)
            v_anchor, log_var_anchor = self.vtx_anchor(
                z_seq.to(pix_plane_tok.dtype),
                z_act,
                v_phys_prior,
                pix_tokens=u_pix,
                pix_xyz_mm=pix_xyz,
                pix_mask=pix_mask,
                pix_plane=batch['pix_plane'],
            )

        # 4. Soft non-exclusive multi-tracklets (reuse precomputed if present, else extract on-the-fly)
        if 'trk_kin' in batch and 'trk_morph' in batch and 'trk_mask' in batch:
            trk_kin = batch['trk_kin']
            trk_morph = batch['trk_morph']
            trk_mask = batch['trk_mask'].bool()
        else:
            trk_dict = extract_soft_tracklets_batch(
                pix_xyz,
                batch['pix_plane'],
                pix_mask,
                self.pix_plane_x0,
                pix_feat=pix_feat,
                max_tracklets=self.max_tracklets,
            )
            trk_kin = trk_dict['trk_kin']
            trk_morph = trk_dict['trk_morph']
            trk_mask = trk_dict['trk_mask']

        if self.ablation == 'no_stub_conversion_law':
            trk_morph_in = trk_morph.clone()
            trk_morph_in[..., 6:8] = 0.0
        else:
            trk_morph_in = trk_morph

        trk_kin_norm = torch.cat([
            (trk_kin[..., 0:2] - self.default_center_mm[:2]) / 135.0,
            (trk_kin[..., 2:3] - self.default_center_mm[2:3]) / 1000.0,
            trk_kin[..., 3:5] * 10.0,
            torch.log1p(trk_kin[..., 5:9].clamp_min(0.0)),
            torch.log1p(trk_kin[..., 9:10].clamp_min(0.0)) / 3.0,
            trk_kin[..., 10:11] / 20.0,
            trk_kin[..., 11:12] / 200.0,
        ], dim=-1)
        trk_emb = self.trk_in(torch.cat([trk_kin_norm, trk_morph_in], dim=-1)) * trk_mask.unsqueeze(-1).float()
        r_ab = pairwise_skew_dca_features(trk_kin, trk_mask)
        trk_pair_bias = self.trk_pair_mlp(r_ab).permute(0, 3, 1, 2)
        trk_emb = self.trk_attn(trk_emb, trk_mask, pair_bias=trk_pair_bias)

        # Differentiable Student-t (df=3) PV + SV + IP re-injection, z-centered around v_anchor[:, 2]
        z_ref = (v_anchor[:, 2:3].detach() if self.n2_vtx_mode == 'v2' else v_anchor[:, 2:3]).float()
        trk_kin_c = trk_kin.float().clone()
        trk_kin_c[..., 2] = trk_kin_c[..., 2] - z_ref
        prior_c = v_anchor.float().clone()
        prior_c[:, 2] = 0.0
        vtx_out = self.vtx_layer(trk_kin_c, trk_emb, trk_mask, prior_c, cal_valid=any_valid_prior)
        v_pv_trk = torch.cat([vtx_out['v_pv_mm'][:, :2], vtx_out['v_pv_mm'][:, 2:3] + z_ref], dim=-1)
        vtx_out['v_pv_mm'] = v_pv_trk
        vtx_out['v_sv_mm'] = torch.cat([vtx_out['v_sv_mm'][:, :2], vtx_out['v_sv_mm'][:, 2:3] + z_ref], dim=-1)
        trk_emb = trk_emb + self.trk_ip_proj(vtx_out['trk_ip_feat']) * trk_mask.unsqueeze(-1).to(trk_emb.dtype)
        trk_emb = self.trk_post_attn(trk_emb, trk_mask, pair_bias=trk_pair_bias)

        # Inverse-variance precision fusion of two-stage neural anchor & Student-t tracklet vertex
        with torch.amp.autocast(device_type=device.type, enabled=False):
            t_logits = self.trk_pool_gate(trk_emb.float()).squeeze(-1).masked_fill(~trk_mask, -1e4)
            t_w = torch.softmax(t_logits, dim=1) * trk_mask.float()
            t_w = t_w / t_w.sum(dim=1, keepdim=True).clamp_min(1e-6)
            trk_glob = (t_w.unsqueeze(-1) * trk_emb.float()).sum(dim=1)

            if self.n2_vtx_mode == 'v2':
                tgt_trk_m = trk_mask & (trk_morph[..., 1] < 0.5)
                has_tgt_trk = tgt_trk_m.any(dim=1)
                has_2trk = (tgt_trk_m.sum(dim=1) >= 2).float()
                sigma_anc = sigma_anc_v2.float().clamp(0.03, 160.0)
                prec_anc = 1.0 / sigma_anc.square().clamp_min(9e-4)
                sigma_trk = vtx_out['scale_pv_mm'].float().clamp(0.02, 150.0)
                trk_axis_gate = torch.stack([
                    has_2trk,
                    has_2trk,
                    0.25 * has_2trk * is_front_p.float(),
                ], dim=-1)
                prec_trk = (1.0 / sigma_trk.square().clamp_min(9e-4)) * trk_axis_gate
            else:
                has_tgt_trk = (trk_mask & (trk_morph[..., 1] < 0.5)).any(dim=1)
                anc_scale_base = v_anchor.new_tensor([0.20, 0.20, 2.50])[None, :]
                sigma_anc = (anc_scale_base * torch.exp(0.5 * log_var_anchor.float().clamp(-6.0, 6.0))).clamp(0.02, 150.0)
                prec_anc = 1.0 / sigma_anc.square()
                sigma_trk = vtx_out['scale_pv_mm'].float().clamp(0.02, 150.0)
                prec_trk = (1.0 / sigma_trk.square()) * has_tgt_trk.unsqueeze(-1).float()
            prec_tot = (prec_anc + prec_trk).clamp_min(1e-6)
            v_pv_fused = (prec_anc * v_anchor.float() + prec_trk * v_pv_trk.float()) / prec_tot
            n_trk_norm = torch.log1p((trk_mask & (trk_morph[..., 1] < 0.5)).float().sum(dim=1)) / 4.0
            regime_ctx = torch.stack([
                is_front_p.float(),
                is_mid_p.float(),
                has_tgt_trk.float(),
                n_trk_norm,
                (cal_mask[:, :self.n_st].float().sum(dim=1) / 54.0),
                (v_pv_fused[:, 2] - self.default_center_mm[2]) / 500.0,
            ], dim=-1)
            pv_scale_ctx = torch.cat([
                z_prof_glob,
                bar_out['cal_global'].float(),
                trk_glob,
                batch['c_obs'].float(),
                regime_ctx,
            ], dim=-1)
            scale_pv_fused = (
                torch.rsqrt(prec_tot) * torch.exp(self.pv_regime_scale(pv_scale_ctx).float().clamp(-1.5, 1.5))
            ).clamp(0.02, 150.0)

        # 4. IPT-Interlocked Spectrometer Encoder
        ipt_mask = trk_mask & (trk_morph[..., 1] > 0.5)
        ipt_interlock = compute_ipt_spectrometer_interlock(
            trk_kin, ipt_mask, batch['trk_cand'], batch['trk_cand_mask']
        )
        cand_in = torch.cat([batch['trk_cand'].float(), ipt_interlock], dim=-1)
        cm = batch['trk_cand_mask']
        has_cand = cm.any(dim=1)
        hc = self.spec_cand_in(cand_in) * cm.unsqueeze(-1).float()
        hc = self.spec_cand_attn(hc, cm)
        c_mean = (hc * cm.unsqueeze(-1).to(hc.dtype)).sum(dim=1) / cm.sum(dim=1, keepdim=True).clamp_min(1).to(hc.dtype)
        c_max = hc.masked_fill(~cm.unsqueeze(-1), -1e4).amax(dim=1)
        cand_pool = torch.cat([c_mean, c_max], dim=-1) * has_cand.unsqueeze(-1).to(hc.dtype)

        hm_bool = batch['trk_hit_mask']
        has_hit = hm_bool.any(dim=1)
        hh = self.spec_hit_in(batch['trk_hits'])
        h_mean = (hh * hm_bool.unsqueeze(-1).to(hh.dtype)).sum(dim=1) / hm_bool.sum(dim=1, keepdim=True).clamp_min(1).to(hh.dtype)
        safe_hm = hm_bool.clone()
        if (~has_hit).any():
            safe_hm[~has_hit, 0] = True
        h_w = torch.softmax(self.spec_hit_score(hh).squeeze(-1).float().masked_fill(~safe_hm, float('-inf')), dim=-1).to(hh.dtype)
        h_attn = (hh * h_w.unsqueeze(-1)).sum(dim=1)
        hit_pool = torch.cat([h_mean, h_attn], dim=-1) * has_hit.unsqueeze(-1).to(hh.dtype)
        spec_tok = self.spec_fusion(torch.cat([cand_pool, hit_pool], dim=-1)) * (has_hit | has_cand).unsqueeze(-1).to(hh.dtype)

        # 6. Global Heterogeneous Perceiver Cross-Modal Fusion
        source = torch.cat([u_pix, pix_plane_tok, cal_tok, trk_emb, spec_tok.unsqueeze(1)], dim=1)
        source_mask = torch.cat([pix_mask, pix_plane_act, cal_mask, trk_mask, (has_hit | has_cand).unsqueeze(1)], dim=1)
        cobs_q = self.cobs_query(batch['c_obs']).unsqueeze(1)
        latents = torch.cat([self.latent_queries.expand(b, -1, -1), cobs_q], dim=1)
        latent_mask = torch.ones((b, self.n_queries + 1), dtype=torch.bool, device=device)
        entropies: list[torch.Tensor] = []
        for blk in self.global_blocks:
            if return_entropy:
                latents, ent = blk(latents, source, source_mask, latent_mask, return_entropy=True)
                entropies.append(ent)
            else:
                latents = blk(latents, source, source_mask, latent_mask)

        q_lat = latents[:, :self.n_queries]
        q_cobs = latents[:, self.n_queries]

        with torch.amp.autocast(device_type=device.type, enabled=False):
            w_lat = torch.softmax(self.pool_gate(q_lat.float()), dim=1)
            pool_attn = (w_lat * q_lat.float()).sum(dim=1)

            v_pv = v_pv_fused
            scale_pv = scale_pv_fused
            sv_disp = vtx_out['sv_disp_mm'].float()
            dchi2 = vtx_out['delta_chi2_2v'].float()
            sv_mag = torch.sqrt(sv_disp.square().sum(dim=-1, keepdim=True) + 1e-8)
            if self.n2_vtx_mode == 'v2':
                sv_disp_det_s = sv_disp.detach()
                sv_mag_det_s = sv_mag.detach()
                vtx_stats = torch.cat([
                    torch.log1p(dchi2.detach().clamp_min(0.0)) / 3.0,
                    torch.asinh(sv_mag_det_s / 0.2) / 3.0,
                    torch.asinh(sv_disp_det_s[:, 2:3] / 0.5) / 3.0,
                    torch.log(scale_pv.detach().clamp_min(0.01)),
                    (v_pv.detach()[:, 2:3] - self.default_center_mm[2]) / 500.0,
                    vtx_out['pi_sv'].detach().float().amax(dim=1, keepdim=True),
                ], dim=-1)
            else:
                vtx_stats = torch.cat([
                    torch.log1p(dchi2.clamp_min(0.0)) / 3.0,
                    torch.asinh(sv_mag / 0.2) / 3.0,
                    torch.asinh(sv_disp[:, 2:3] / 0.5) / 3.0,
                    torch.log(scale_pv.clamp_min(0.01)),
                    (v_pv[:, 2:3] - self.default_center_mm[2]) / 500.0,
                    vtx_out['pi_sv'].float().amax(dim=1, keepdim=True),
                ], dim=-1)
            vtx_stat_tok = self.vtx_stat_proj(vtx_stats)

            z_backbone = self.global_fusion(torch.cat([
                pool_attn,
                z_prof_glob,
                bar_out['cal_global'].float(),
                trk_glob + vtx_stat_tok,
                spec_tok.float(),
                q_cobs.float(),
            ], dim=-1))

            # 7. Three Differentiable PINPOINT Physics Laws -> phi_law (B, 18)
            # Detach PV/SV geometry inputs so l_phys12 cannot hijack v_pv or crush pi_sv
            v_pv_det = v_pv.detach()
            pi_sv_det = vtx_out['pi_sv'].detach().float()
            dchi2_det = dchi2.detach()
            sv_mag_det = sv_mag.detach()

            grid_x_e, grid_y_e = self._build_bar_energy_grids(batch, b, device)
            law1_out = self.cascade_fitter(grid_x_e, grid_y_e, v_pv_det)

            # Law 2: Displaced Topology & Parent-Stub vs Photon-Gap Bayes factor
            s_d0_norm = vtx_out['trk_ip_feat'][..., 1].detach().float()  # log1p(s_d0) / 3
            s_d0_raw = torch.expm1((3.0 * s_d0_norm).clamp(0.0, 12.0))
            is_tgt = trk_mask & (trk_morph[..., 1] < 0.5)
            w_disp_raw = (pi_sv_det * torch.sigmoid(s_d0_raw - 1.2) + 0.05 * s_d0_raw) * is_tgt.float()
            has_disp = (w_disp_raw.sum(dim=1) > 0.05).float()
            w_disp = w_disp_raw / w_disp_raw.sum(dim=1, keepdim=True).clamp_min(1e-6)

            s_disp_agg = (w_disp * s_d0_raw).sum(dim=1)
            sqrt_lam2 = 0.03 * torch.sinh((3.0 * trk_morph[..., 4].float()).clamp(0.0, 6.0))
            wide_pair = torch.sigmoid((sqrt_lam2 - 0.012) / 0.005)
            has_stub_d = (w_disp * trk_morph[..., 6].float()).sum(dim=1)
            gap_d = (w_disp * trk_morph[..., 7].float() * wide_pair).sum(dim=1)

            # Early 3 X0 EM burst fraction downstream of displaced tracklet start z0_d
            z0_disp = (w_disp * trk_kin[..., 2].float()).sum(dim=1)
            x0_disp = self.cascade_fitter._interp_x0(z0_disp)
            dx0_disp = self.cascade_fitter.st_x0[None, :] - x0_disp[:, None]
            em3_mask = torch.sigmoid((dx0_disp + 0.2) / 0.4) * torch.sigmoid((3.5 - dx0_disp) / 0.8)
            i_em_3x0 = (law1_out['f_l'] * em3_mask).sum(dim=-1).clamp(0.0, 1.0)

            w_s = F.softplus(self.w_stub.float())
            w_g = F.softplus(self.w_gap.float())
            b_stub_gap = (w_s * has_stub_d - w_g * gap_d * (0.2 + 0.8 * i_em_3x0)) * has_disp

            # Shared-root 3-prong indicator (frac part of trk_morph[..., 0] > 0.25)
            is_shared_root = ((trk_morph[..., 0].float() - torch.floor(trk_morph[..., 0].float())) > 0.25).float()
            shared_root_3p = (w_disp * is_shared_root).sum(dim=1) * has_disp

            law2_raw = torch.stack([
                torch.log1p(s_disp_agg.clamp_min(0.0)) / 2.5,
                torch.log1p(dchi2_det.squeeze(-1).clamp_min(0.0)) / 3.0,
                torch.asinh(sv_mag_det.squeeze(-1) / 0.2) / 3.0,
                b_stub_gap,
                shared_root_3p,
            ], dim=-1) * F.softplus(self.law2_scale.float()).clamp(0.2, 3.0)[None, :]

            law3_out = self.hemisphere_mod(
                grid_x_e,
                grid_y_e,
                v_pv_det,
                trk_kin,
                trk_morph,
                trk_mask,
                pi_sv_det,
                s_d0_raw,
                batch['trk_cand'],
                batch['trk_cand_mask'],
                ipt_interlock,
            )

            law1_feat = law1_out['law1_feat']
            law2_feat = law2_raw
            law3_feat = law3_out['law3_feat']

            # Apply exact causal physics ablations
            if self.ablation == 'no_gamma_em_profile':
                law1_feat = law1_feat * 0.0
            elif self.ablation == 'no_stub_conversion_law':
                law2_feat = torch.cat([law2_feat[:, :3], law2_feat[:, 3:4] * 0.0, law2_feat[:, 4:5]], dim=-1)
            elif self.ablation == 'no_hemisphere_opposition':
                # Zero out C_hem, I_iso, acoplanarity_disp, mu_disp_opp
                law3_mask = law3_feat.new_tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0])[None, :]
                law3_feat = law3_feat * law3_mask

            phi_law = torch.cat([law1_feat, law2_feat, law3_feat], dim=-1)  # (B, 18)

            # 8. 12-Hypothesis Physics Likelihood + Bounded Neural Residual
            hyp_out = self.hyp_engine(z_backbone, phi_law)
            kin_pred = self.kin_head(z_backbone)

        aux: dict[str, Any] = {
            'phi_law': phi_law,
            'probs12': hyp_out['probs12'],
            'l_phys12': hyp_out['l_phys12'],
            'r_res12': hyp_out['r_res12'],
            'q_bkg_charm': hyp_out['q_bkg_charm'],
            'p_all_charm': hyp_out['p_all_charm'],
            'z_tau_c': hyp_out['z_tau_c'],
            'delta_chi2_2v': dchi2.squeeze(-1),
            'v_anchor': v_anchor.float(),
            'sigma_anc': sigma_anc,
            'entropies': entropies,
        }
        return NextGenModelOutput(
            logits7=hyp_out['logits7'],
            probs4=hyp_out['probs4'],
            charm_logit=hyp_out['charm_logit'],
            v_pv_mm=v_pv,
            scale_pv_mm=scale_pv,
            sv_disp_mm=sv_disp,
            kin_pred=kin_pred,
            aux=aux,
        )


def _focal_bce_from_prob(
    prob: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    gamma: float = 1.5,
    pos_weight: float = 3.0,
    denom_override: float | None = None,
) -> torch.Tensor:
    """Focal binary cross-entropy from probability in FP32 with exact microbatch denominator support."""
    if not valid.any():
        return prob.sum() * 0.0
    p = prob[valid].float().clamp(1.0e-6, 1.0 - 1.0e-6)
    y = target[valid].float()
    bce = -(y * torch.log(p) + (1.0 - y) * torch.log1p(-p))
    pt = torch.where(y > 0.5, p, 1.0 - p)
    focal = ((1.0 - pt) ** gamma) * bce
    w = torch.where(y > 0.5, float(pos_weight), 1.0)
    denom = float(denom_override) if denom_override is not None else float(w.sum().clamp_min(1.0e-8).item())
    return (focal * w).sum() / max(denom, 1.0e-8)


def compute_tau_kinema_window_denoms(
    micro_batches: list[dict[str, Any]],
    *,
    pos_weight: float = 3.0,
) -> dict[str, float]:
    """Compute exact full-window per-task denominators across microbatches for F5 supervision."""
    d_proc7 = 0.0
    d_proc4 = 0.0
    d_charm_nontau = 0.0
    d_charm_all = 0.0
    d_vtx = 0.0
    d_kin = 0.0

    for mb in micro_batches:
        lbl4 = mb['label'].long()
        d_proc4 += float(len(lbl4))

        if 'label_7' in mb:
            l7 = mb['label_7'].long()
            tbv = mb['tau_branch_valid'].long() if 'tau_branch_valid' in mb else (l7 >= 0).long()
            v7 = (l7 >= 0) & (tbv > 0)
            d_proc7 += float(v7.sum().item())

        if 'charm_valid' in mb and 'charm_flag' in mb:
            cv = mb['charm_valid'].long() > 0
            if cv.any():
                cf = mb['charm_flag'].float()
                w_all = torch.where(cf[cv] > 0.5, float(pos_weight), 1.0)
                d_charm_all += float(w_all.sum().item())
            cv_nt = cv & (lbl4 != 2)
            if cv_nt.any():
                cf = mb['charm_flag'].float()
                w_nt = torch.where(cf[cv_nt] > 0.5, float(pos_weight), 1.0)
                d_charm_nontau += float(w_nt.sum().item())

        if 'vertex_valid' in mb:
            d_vtx += float((mb['vertex_valid'].long() > 0).sum().item())

        if 'kin_valid' in mb:
            d_kin += float((mb['kin_valid'].long() > 0).sum().item())

    return {
        'proc7': max(d_proc7, 1.0e-8),
        'proc4': max(d_proc4, 1.0e-8),
        'charm_nontau': max(d_charm_nontau, 1.0e-8),
        'charm_all': max(d_charm_all, 1.0e-8),
        'vtx': max(d_vtx, 1.0e-8),
        'kin': max(d_kin, 1.0e-8),
    }


def compute_tau_kinema_loss(
    model: TauKinemaDiffNet,
    out: NextGenModelOutput | dict[str, Any],
    batch: dict[str, Any],
    *,
    fixed_denoms: dict[str, float] | None = None,
    w_proc: float = 1.0,
    w_charm: float = 0.25,
    w_vtx: float | None = None,
    w_kin: float = 0.15,
    w_res: float = 1.0e-3,
    w_sv_disp: float = 0.0,  # Strictly 0.0 on Clean v3 (zero tau_decay_L3d_mm on train)
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute exact F5 supervision loss for TauKinemaDiffNet (strictly internal_train keys only)."""
    del w_sv_disp  # Explicitly unused; kept in signature to document F5 closure
    is_n2v2 = getattr(model, 'n2_vtx_mode', 'v1') == 'v2'
    eff_w_vtx = (0.30 if is_n2v2 else 0.20) if w_vtx is None else float(w_vtx)
    if isinstance(out, NextGenModelOutput):
        logits7 = out.logits7
        probs4 = out.probs4
        v_pv_mm = out.v_pv_mm
        scale_pv_mm = out.scale_pv_mm
        kin_pred = out.kin_pred
        aux = out.aux
    else:
        logits7 = out['logits']
        probs4 = out['probs4']
        v_pv_mm = out['v_hat']
        scale_pv_mm = out['scale_pv_mm']
        kin_pred = out.get('kin_pred')
        aux = out

    device = logits7.device
    fd = fixed_denoms or {}
    lbl4 = batch['label'].long()
    b = int(len(lbl4))
    d_proc4 = float(fd.get('proc4', max(b, 1)))

    with torch.amp.autocast(device_type=device.type, enabled=False):
        p4 = probs4.float().clamp(1.0e-7, 1.0)
        l_proc4 = (-torch.log(p4[torch.arange(b, device=device), lbl4])).sum() / d_proc4

        has_l7 = 'label_7' in batch and not model.hyp_engine.abl_unconstrained
        if has_l7:
            l7 = batch['label_7'].long()
            tbv = batch['tau_branch_valid'].long() if 'tau_branch_valid' in batch else (l7 >= 0).long()
            v7 = (l7 >= 0) & (tbv > 0)
            if v7.any():
                d_proc7 = float(fd.get('proc7', float(v7.sum().item())))
                l_proc7 = F.nll_loss(
                    F.log_softmax(logits7[v7].float(), dim=-1),
                    l7[v7],
                    reduction='sum',
                ) / max(d_proc7, 1.0e-8)
                l_proc = 0.5 * l_proc7 + 0.5 * l_proc4
            else:
                l_proc = l_proc4
        else:
            l_proc = l_proc4

        # Conflict-free F5 charm supervision
        if 'charm_valid' in batch and 'charm_flag' in batch:
            cv = batch['charm_valid'].long() > 0
            cf = batch['charm_flag'].float()
            if model.hyp_engine.abl_unconstrained:
                l_charm = _focal_bce_from_prob(
                    aux['p_all_charm'],
                    cf,
                    cv,
                    denom_override=fd.get('charm_all'),
                )
            else:
                cv_nt = cv & (lbl4 != 2)
                l_c_nt = _focal_bce_from_prob(
                    aux['q_bkg_charm'],
                    cf,
                    cv_nt,
                    denom_override=fd.get('charm_nontau'),
                )
                l_c_all = _focal_bce_from_prob(
                    aux['p_all_charm'],
                    cf,
                    cv,
                    denom_override=fd.get('charm_all'),
                )
                l_charm = 0.5 * l_c_nt + 0.5 * l_c_all
        else:
            l_charm = logits7.sum() * 0.0

        # Heavy-tailed Student-t (df=3) PV loss (+ anchor auxiliary supervision)
        if 'vertex_valid' in batch and 'vertex_xyz_mm' in batch:
            l_vtx = student_t_vertex_loss(
                v_pv_mm,
                scale_pv_mm,
                batch['vertex_xyz_mm'],
                batch['vertex_valid'],
                df=3.0,
                denom_override=fd.get('vtx'),
            )
            if 'v_anchor' in aux and 'sigma_anc' in aux:
                anc_w = 0.25 if is_n2v2 else 0.15
                l_vtx = l_vtx + anc_w * student_t_vertex_loss(
                    aux['v_anchor'],
                    aux['sigma_anc'],
                    batch['vertex_xyz_mm'],
                    batch['vertex_valid'],
                    df=3.0,
                    denom_override=fd.get('vtx'),
                )
        else:
            l_vtx = logits7.sum() * 0.0

        # Visible kinematics Smooth L1
        if kin_pred is not None and 'kin_valid' in batch and 'vis_kinematics' in batch:
            kv = batch['kin_valid'].long() > 0
            if kv.any():
                d_kin = float(fd.get('kin', float(kv.sum().item())))
                k_ev = F.smooth_l1_loss(
                    kin_pred[kv].float(),
                    batch['vis_kinematics'][kv].float(),
                    reduction='none',
                ).mean(dim=-1)
                l_kin = k_ev.sum() / max(d_kin, 1.0e-8)
            else:
                l_kin = logits7.sum() * 0.0
        else:
            l_kin = logits7.sum() * 0.0

        # F5 physics-law anchor + bounded residual L2 (scaled by b / d_proc4 for exact microbatch accum)
        l_prior = model.hyp_engine.physics_prior_penalty() * (float(b) / max(d_proc4, 1.0e-8))
        r_res12 = aux.get('r_res12')
        if r_res12 is not None and not model.hyp_engine.abl_unconstrained:
            l_res = w_res * (r_res12.float().square().mean(dim=-1).sum() / max(d_proc4, 1.0e-8))
        else:
            l_res = logits7.sum() * 0.0

        total = (
            w_proc * l_proc
            + w_charm * l_charm
            + eff_w_vtx * l_vtx
            + w_kin * l_kin
            + l_prior
            + l_res
        )

    parts = {
        'total': float(total.detach().item()),
        'l_proc': float(l_proc.detach().item()),
        'l_charm': float(l_charm.detach().item()),
        'l_vtx': float(l_vtx.detach().item()),
        'l_kin': float(l_kin.detach().item()),
        'l_prior': float(l_prior.detach().item()),
        'l_res': float(l_res.detach().item()),
    }
    return total, parts
