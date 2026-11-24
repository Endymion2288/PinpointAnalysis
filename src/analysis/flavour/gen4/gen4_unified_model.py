"""Unified PinpointGen4Net Architecture, Loss Engine, and Pure-Observable NSBI Extractor.

Implements the exact 5-group top-level module layout from `[gen4-contract]`:
  - `self.si_vtx`:    MicroConePairmixerNDIVE (~1.74M)
  - `self.cal_pflow`: PinpointPFlowFMPE       (~3.45M)
  - `self.spec_ipt`:  UnifiedMuonSpecEncoder  (~0.35M)
  - `self.fusion`:    CrossModalSynergyFusion (~0.60M)
  - `self.head`:      PinpointUParT13Head     (pluggable; unbiased 13-subchannel likelihood ratio)

Strictly enforces Gate-A zero truth invariance (`forward()` reads only pure observables;
all `GEN4_TRUTH_BATCH_KEYS` are consumed exclusively inside `compute_gen4_supervised_loss`).
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from analysis.flavour.gen4.geometry_contract import Gen4GeometryContract, load_geometry_contract
from analysis.flavour.gen4.micro_ndive import (
    MicroConePairmixerNDIVE,
    MicroVtxOutput,
    PHI_VTX_DIM,
    _solve_spd_3x3,
)
from analysis.flavour.gen4.pflow_fmpe import (
    FMPE_LATENT_DIM,
    PFlowFMPEOutput,
    PSI_NOMAD_DIM,
    PinpointPFlowFMPE,
    SpecIPTOutput,
    UnifiedMuonSpecEncoder,
)
from analysis.flavour.gen4.upart13_tpauc import (
    PinpointUParT13Head,
    compute_upart13_lort_loss,
)
from analysis.flavour.nextgen.gn2_tracklet_ipformer import GN2_SIDECAR_BATCH_KEYS
from analysis.flavour.nextgen.protocols import (
    NextGenFlavourModel,
    NextGenModelOutput,
    student_t_vertex_loss,
)
from analysis.flavour.nextgen.seal_guard import TRUTH_BATCH_KEYS
from analysis.flavour.v3.v3_multitask.losses import focal_charm_bce


GEN4_TRUTH_BATCH_KEYS: tuple[str, ...] = tuple(
    sorted(
        set(TRUTH_BATCH_KEYS)
        | set(GN2_SIDECAR_BATCH_KEYS)
        | {
            'cl_sidecar_valid',
            'cl_dominant_edep_frac',
            'true_sv_xyz_mm',
            'has_true_sv',
            'tau_n_charged_prongs_si',
            'tier1b_genie_valid',
            'true_tau_submode',
            'true_n_pi0',
            'true_nu_energy_gev',
            'true_branch_p4_gev',
            'cl_is_real_target',
        }
    )
)


class CrossModalSynergyFusion(nn.Module):
    """Cross-modal gated synergy fusion block (`self.fusion`, ~0.60M params).

    Signature strictly matches `ssl_sonata_v2.py` & `conflict_free_optim.py`:
        `fusion(z_si_vtx, z_cal_pflow, z_spec_ipt) -> z_fuse (B, 256)`.
    """

    def __init__(self, d_in: int = 256, d_out: int = 256, dropout: float = 0.10) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        self.norm_si = nn.LayerNorm(d_in)
        self.norm_cal = nn.LayerNorm(d_in)
        self.norm_spec = nn.LayerNorm(d_in)

        self.bilinear_si_cal = nn.Linear(d_in, d_in, bias=False)
        self.gate_proj = nn.Sequential(
            nn.Linear(3 * d_in, d_in),
            nn.SiLU(),
            nn.Linear(d_in, 3),
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(4 * d_in),
            nn.Linear(4 * d_in, 2 * d_out),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_out, d_out),
            nn.LayerNorm(d_out),
        )

    def forward(
        self,
        z_si_vtx: torch.Tensor,
        z_cal_pflow: torch.Tensor,
        z_spec_ipt: torch.Tensor,
    ) -> torch.Tensor:
        zs = self.norm_si(z_si_vtx.float())
        zc = self.norm_cal(z_cal_pflow.float())
        zp = self.norm_spec(z_spec_ipt.float())

        # Cross-modal bilinear synergy term (zero when silicon is absent on Fortune_ScintOnly)
        si_active = (z_si_vtx.abs().sum(dim=-1, keepdim=True) > 1e-6).float()
        z_syn = self.bilinear_si_cal(zs) * zc * si_active

        cat3 = torch.cat([zs, zc, zp], dim=-1)
        gates = torch.softmax(self.gate_proj(cat3), dim=-1)
        z_mix = gates[:, 0:1] * zs + gates[:, 1:2] * zc + gates[:, 2:3] * zp
        out = self.mlp(torch.cat([cat3, z_syn], dim=-1))
        return z_mix + out


class PinpointGen4Net(NextGenFlavourModel):
    """Unified Pinpoint-Gen4 dual-pillar architecture with 5 contracted top-level groups."""

    def __init__(
        self,
        geometry: Gen4GeometryContract | dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if geometry is None:
            self.geom = load_geometry_contract()
        elif isinstance(geometry, Gen4GeometryContract):
            self.geom = geometry
        else:
            self.geom = load_geometry_contract(geometry)

        self.cfg = dict(config or {})
        dropout = float(self.cfg.get('head_dropout', 0.12))
        lambda_charm = float(self.cfg.get('lambda_charm', 1.5))
        lambda_pi0 = float(self.cfg.get('lambda_pi0', 0.25))

        # Exact 5 contracted top-level nn.Module attributes:
        self.si_vtx = MicroConePairmixerNDIVE(self.geom, self.cfg)
        self.cal_pflow = PinpointPFlowFMPE(self.geom, self.cfg)
        self.spec_ipt = UnifiedMuonSpecEncoder(
            self.geom,
            d_tok=192,
            enable_m2_kalman=bool(self.cfg.get('enable_m2_kalman', True)),
        )
        self.fusion = CrossModalSynergyFusion(d_in=256, d_out=256, dropout=dropout)
        self.head = PinpointUParT13Head(
            fuse_dim=256,
            phi_vtx_dim=PHI_VTX_DIM,
            psi_nomad_dim=PSI_NOMAD_DIM,
            hidden_dim=256,
            dropout=dropout,
            lambda_charm=lambda_charm,
            lambda_pi0=lambda_pi0,
        )

    def forward_features(self, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        """Run full PinpointGen4Net hierarchy without head (for PinpointSSL-v2 & probing)."""
        micro_out: MicroVtxOutput = self.si_vtx(dict(batch))
        spec_out: SpecIPTOutput = self.spec_ipt(dict(batch), micro_out)
        pflow_out: PFlowFMPEOutput = self.cal_pflow(dict(batch), micro_out, spec_out)
        z_fuse = self.fusion(micro_out.z_si_vtx, pflow_out.z_cal_pflow, spec_out.z_spec_ipt)
        return {
            'micro_out': micro_out,
            'spec_out': spec_out,
            'pflow_out': pflow_out,
            'u_micro': micro_out.u_micro,
            's_si_meso': micro_out.s_si_meso,
            's_cal_meso': pflow_out.s_cal_meso,
            'z_si_vtx': micro_out.z_si_vtx,
            'z_cal_pflow': pflow_out.z_cal_pflow,
            'z_spec_ipt': spec_out.z_spec_ipt,
            'z_fuse': z_fuse,
            'seg_mask': micro_out.seg_mask,
            'phi_vtx': micro_out.phi_vtx,
            'psi_nomad': pflow_out.psi_nomad,
            'e_vis_hat_gev': pflow_out.e_vis_hat_gev,
        }

    def forward_NextGen(self, batch: dict[str, torch.Tensor]) -> NextGenModelOutput:
        feats = self.forward_features(batch)
        micro_out: MicroVtxOutput = feats['micro_out']
        spec_out: SpecIPTOutput = feats['spec_out']
        pflow_out: PFlowFMPEOutput = feats['pflow_out']
        z_fuse: torch.Tensor = feats['z_fuse']

        head_out = self.head(
            z_fuse,
            micro_out.phi_vtx,
            pflow_out.psi_nomad,
            batch['obs_domain_id'],
            pflow_out.e_vis_hat_gev,
        )

        aux: dict[str, Any] = {
            'logits13': head_out['logits13'],
            'tau_llr_score': head_out['tau_llr_score'],
            'z_si_vtx': micro_out.z_si_vtx,
            'z_cal_pflow': pflow_out.z_cal_pflow,
            'z_spec_ipt': spec_out.z_spec_ipt,
            'z_fuse': z_fuse,
            'phi_vtx': micro_out.phi_vtx,
            'psi_nomad': pflow_out.psi_nomad,
            'e_vis_hat_gev': pflow_out.e_vis_hat_gev,
            'cov_pv_mm2': micro_out.cov_pv_mm2,
            'v_slots_mm': micro_out.v_slots_mm,
            'cov_slots_mm2': micro_out.cov_slots_mm2,
            'seg_state': micro_out.seg_state,
            'seg_mask': micro_out.seg_mask,
            'seg_slot_probs': micro_out.seg_slot_probs,
            'pflow_slot_p4_gev': pflow_out.pflow_slot_p4_gev,
            'pflow_slot_embed': pflow_out.pflow_slot_embed,
            'fmpe_samples_intrinsic': pflow_out.fmpe_samples_intrinsic,
            'u_micro': micro_out.u_micro,
            's_si_meso': micro_out.s_si_meso,
            's_cal_meso': pflow_out.s_cal_meso,
            'refined_qp': spec_out.refined_qp,
            'muon_at_pv': spec_out.muon_at_pv,
        }
        aux.update(micro_out.aux)
        aux.update(pflow_out.aux)

        return NextGenModelOutput(
            logits7=head_out['logits7'].float(),
            probs4=head_out['probs4'].float(),
            charm_logit=head_out['charm_logit'].float(),
            v_pv_mm=micro_out.v_pv_mm.float(),
            scale_pv_mm=micro_out.scale_pv_mm.float(),
            sv_disp_mm=micro_out.sv_disp_mm.float(),
            kin_pred=pflow_out.kin_pred.float(),
            aux=aux,
        )


def multivariate_student_t_vertex_loss(
    v_pred_mm: torch.Tensor,
    cov_pred_mm2: torch.Tensor,
    v_true_mm: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    df: float = 3.0,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Full 3x3 inverse-covariance multivariate Student-t(nu=3) vertex NLL in FP32."""
    valid = valid_mask.bool()
    if not valid.any():
        return (v_pred_mm.sum() + cov_pred_mm2.sum()) * 0.0
    with torch.amp.autocast(device_type=v_pred_mm.device.type, enabled=False):
        vp = v_pred_mm[valid].float()
        vt = v_true_mm[valid].float()
        cov = cov_pred_mm2[valid].float()
        diff = vp - vt
        _, inv_cov = _solve_spd_3x3(cov, diff)
        mahal2 = torch.einsum('ni,nij,nj->n', diff, inv_cov, diff).clamp(0.0, 1.0e6)
        _, logdet = torch.linalg.slogdet(0.5 * (cov + cov.transpose(-1, -2)) + 1e-6 * torch.eye(3, device=cov.device))
        nu = float(max(1.5, df))
        nll = 0.5 * (nu + 3.0) * torch.log1p(mahal2 / nu) + 0.5 * logdet + 6.0
        if sample_weight is not None:
            w = sample_weight[valid].float().clamp_min(0.0)
            return (nll * w).sum() / w.sum().clamp_min(1e-8)
        return nll.mean()


def compute_gen4_supervised_loss(
    model: PinpointGen4Net,
    out: NextGenModelOutput,
    batch: Mapping[str, torch.Tensor],
    *,
    delta_lort: float = 0.08,
    w_charm: float = 0.25,
    w_vtx_pv: float = 0.25,
    w_vtx_sv: float = 0.20,
    w_cl_origin: float = 0.15,
    w_slot_mask: float = 0.15,
    w_kin: float = 0.20,
    w_pflow_weak: float = 0.10,
    w_dark: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], dict[str, float]]:
    """Compute complete multi-level Gen4 supervised loss (strictly inside training only).

    Returns:
      (total_loss, primary_loss, aux_losses, breakdown_dict)
    so it integrates cleanly with both standard AdamW and `ConflictFreeStepEngine`.
    """
    dev = out.logits7.device
    b = int(out.logits7.shape[0])
    sw = batch.get('sample_weight')

    with torch.amp.autocast(device_type=dev.type, enabled=False):
        # 1. Primary 13-subchannel LORT loss + auxiliary Charm loss
        l_proc, proc_info = compute_upart13_lort_loss(
            out.aux['logits13'],
            batch,
            delta_lort=delta_lort,
            lambda_mag=0.01,
            sample_weight=sw,
        )
        if 'charm_flag' in batch and 'charm_valid' in batch:
            l_charm = focal_charm_bce(
                out.charm_logit,
                batch['charm_flag'],
                batch['charm_valid'],
                gamma=1.5,
                pos_weight=3.0,
                sample_weight=sw,
            )
        else:
            l_charm = out.charm_logit.sum() * 0.0
        primary_loss = l_proc + w_charm * l_charm

        # 2. Aux 1: Full 3x3 covariance Student-t PV & SV losses
        if 'vertex_xyz_mm' in batch and 'vertex_valid' in batch:
            vv = batch['vertex_valid'] > 0
            l_pv = multivariate_student_t_vertex_loss(
                out.v_pv_mm, out.aux['cov_pv_mm2'], batch['vertex_xyz_mm'], vv, df=3.0, sample_weight=sw
            )
            # Also calibrate axis scale consistency
            l_pv = l_pv + 0.25 * student_t_vertex_loss(
                out.v_pv_mm, out.scale_pv_mm, batch['vertex_xyz_mm'], batch['vertex_valid'], df=3.0, sample_weight=sw
            )
        else:
            l_pv = out.v_pv_mm.sum() * 0.0

        # Secondary vertex supervision from l1_cluster_sidecar / 10085-10088 SV truth
        if (
            'true_sv_xyz_mm' in batch
            and 'has_true_sv' in batch
            and 'cl_sidecar_valid' in batch
        ):
            sv_ok = batch['has_true_sv'].bool() & batch['cl_sidecar_valid'].bool()
            y4 = batch['label'].long()
            cf = batch['charm_flag'].long() if 'charm_flag' in batch else torch.zeros_like(y4)
            tau_sv_m = sv_ok & (y4 == 2)
            chm_sv_m = sv_ok & (y4 != 2) & (cf == 1)
            v_slots = out.aux['v_slots_mm']
            cov_slots = out.aux['cov_slots_mm2']
            l_sv_t = multivariate_student_t_vertex_loss(
                v_slots[:, 1], cov_slots[:, 1], batch['true_sv_xyz_mm'], tau_sv_m, df=3.0, sample_weight=sw
            )
            l_sv_c = multivariate_student_t_vertex_loss(
                v_slots[:, 2], cov_slots[:, 2], batch['true_sv_xyz_mm'], chm_sv_m, df=3.0, sample_weight=sw
            )
            l_sv = l_sv_t + l_sv_c
        else:
            l_sv = out.aux['v_slots_mm'].sum() * 0.0

        # 3. Aux 2: Cluster soft-origin KL & Competitive 8-slot / Co-vertex routing
        l_cl = out.aux['cl_origin_logits'].sum() * 0.0
        l_slot = out.aux['seg_slot_probs'].sum() * 0.0
        if 'cl_origin_label' in batch and 'cl_sidecar_valid' in batch:
            sc_v = batch['cl_sidecar_valid'].bool()
            cl_o = batch['cl_origin_label'].long()
            cl_m = (cl_o >= 0) & sc_v[:, None] & batch['pix_valid'].bool()
            if cl_m.any():
                lg_cl = out.aux['cl_origin_logits'][cl_m].float()
                y1 = cl_o[cl_m]
                if 'cl_secondary_origin_label' in batch and 'cl_dominant_edep_frac' in batch:
                    y2 = batch['cl_secondary_origin_label'][cl_m].long()
                    f1 = batch['cl_dominant_edep_frac'][cl_m].float().clamp(0.0, 1.0)
                    y2_safe = torch.where(y2 >= 0, y2, y1)
                    soft_tgt = f1.unsqueeze(-1) * F.one_hot(y1, 5).float() + (1.0 - f1).unsqueeze(-1) * F.one_hot(y2_safe, 5).float()
                else:
                    soft_tgt = F.one_hot(y1, 5).float()
                cw = torch.tensor([2.0, 8.0, 8.0, 0.5, 0.4], dtype=torch.float32, device=dev)
                w_cl = cw[y1]
                log_p_cl = F.log_softmax(lg_cl, dim=-1)
                kl_cl = -(soft_tgt * log_p_cl).sum(dim=-1)
                l_cl = (kl_cl * w_cl).sum() / w_cl.sum().clamp_min(1e-8)

            # Map cluster origin -> competitive slot targets for constituent segments
            seg_cl = out.aux['seg_cl_idx']  # (B, 128, 3)
            seg_m = out.aux['seg_mask'] & sc_v[:, None]
            if seg_m.any() and cl_o.shape[1] > 0:
                ca = seg_cl[:, :, 0].clamp(0, cl_o.shape[1] - 1)
                oa = torch.where(seg_cl[:, :, 0] >= 0, torch.gather(cl_o, 1, ca), -1)
                # Slot mapping: 0->PV(0), 1->tau(1), 2->charm(2), 3->gammaV0(3), 4/ghost->ghost(7)
                slot_tgt = torch.full_like(oa, 7)
                slot_tgt = torch.where(oa == 0, 0, slot_tgt)
                slot_tgt = torch.where(oa == 1, 1, slot_tgt)
                slot_tgt = torch.where(oa == 2, 2, slot_tgt)
                slot_tgt = torch.where(oa == 3, 3, slot_tgt)

                # Verify trackID consistency for 3-hit/2-hit purity if available
                if 'cl_dominant_track_id' in batch:
                    tid_cl = batch['cl_dominant_track_id'].long()
                    ta = torch.where(seg_cl[:, :, 0] >= 0, torch.gather(tid_cl, 1, seg_cl[:, :, 0].clamp(0, cl_o.shape[1] - 1)), -1)
                    tb = torch.where(seg_cl[:, :, 1] >= 0, torch.gather(tid_cl, 1, seg_cl[:, :, 1].clamp(0, cl_o.shape[1] - 1)), -2)
                    is_2p = seg_cl[:, :, 1] >= 0
                    ghost_m = is_2p & ((ta <= 0) | (ta != tb)) & (oa != 3)
                    slot_tgt = torch.where(ghost_m, 7, slot_tgt)

                p_ik = out.aux['seg_slot_probs'].clamp_min(1e-6)
                valid_s = seg_m & (oa >= 0)
                if valid_s.any():
                    # Collapse V0 slots 3..6 into combined V0 mass for clean supervision without permutation ambiguity
                    p_v0 = p_ik[:, :, 3:7].sum(dim=-1, keepdim=True)
                    p_5 = torch.cat([p_ik[:, :, 0:3], p_v0, p_ik[:, :, 7:8]], dim=-1).clamp_min(1e-6)
                    tgt_5 = torch.where(slot_tgt == 7, 4, slot_tgt).clamp(0, 4)
                    nll_slot = -torch.log(torch.gather(p_5[valid_s], -1, tgt_5[valid_s].unsqueeze(-1)).squeeze(-1))
                    sw_5 = torch.tensor([1.5, 6.0, 6.0, 0.5, 0.4], dtype=torch.float32, device=dev)
                    w_s = sw_5[tgt_5[valid_s]]
                    l_slot = (nll_slot * w_s).sum() / w_s.sum().clamp_min(1e-8)

        # DarkShield real-vs-dark pixel loss if target exists
        if 'cl_is_real_target' in batch:
            dm = batch['pix_valid'].bool() & (batch['cl_is_real_target'] >= 0)
            if dm.any():
                l_dark = F.binary_cross_entropy_with_logits(
                    out.aux['dark_logits'][dm].float(),
                    batch['cl_is_real_target'][dm].float(),
                )
            else:
                l_dark = out.aux['dark_logits'].sum() * 0.0
        else:
            l_dark = out.aux['dark_logits'].sum() * 0.0

        # 4. Aux 3: Visible kinematics & Macro-HGPflow Tier-1A / Tier-1B 4-momentum closure
        if out.kin_pred is not None and 'vis_kinematics' in batch and 'kin_valid' in batch:
            kv = batch['kin_valid'] > 0
            if kv.any():
                l_kin = F.smooth_l1_loss(out.kin_pred[kv].float(), batch['vis_kinematics'][kv].float())
            else:
                l_kin = out.kin_pred.sum() * 0.0
        else:
            l_kin = out.logits7.sum() * 0.0

        # Tier-1A physics weak supervision on Macro-HGPflow slots:
        p4_slots = out.aux['pflow_slot_p4_gev']  # (B, 8, 4)
        l_pf = p4_slots.sum() * 0.0
        y4 = batch['label'].long()
        cf = batch['charm_flag'].long() if 'charm_flag' in batch else torch.zeros_like(y4)
        # (a) Total visible 4-momentum closure (slots 0..6) vs vis_kinematics
        if 'vis_kinematics' in batch and 'kin_valid' in batch:
            kv = batch['kin_valid'] > 0
            if kv.any():
                p_tot = p4_slots[kv, :7, :3].sum(dim=1)
                pz_l1p = torch.log1p(p_tot[:, 2].clamp_min(0.0))
                pt_mag = torch.sqrt(p_tot[:, 0].square() + p_tot[:, 1].square() + 1e-6)
                pt_l1p = torch.log1p(pt_mag)
                vk = batch['vis_kinematics'][kv].float()
                l_pf = l_pf + 0.5 * (F.smooth_l1_loss(pz_l1p, vk[:, 0]) + F.smooth_l1_loss(pt_l1p, vk[:, 1]))
        # (b) Sparsity: light NC (y4==3 & cf==0) should not put large energy in charged lepton/muon slots 0, 2, 3
        nc_light_m = (y4 == 3) & (cf == 0)
        if nc_light_m.any():
            e_lep_nc = (p4_slots[nc_light_m, 0, 3] + p4_slots[nc_light_m, 2, 3] + p4_slots[nc_light_m, 3, 3])
            e_tot_nc = p4_slots[nc_light_m, :7, 3].sum(dim=-1).clamp_min(1.0)
            l_pf = l_pf + 0.25 * (e_lep_nc / e_tot_nc).mean()
        # (c) Tier-1B exact branch 4-momentum supervision when GENIE truth attached
        if 'tier1b_genie_valid' in batch and 'true_branch_p4_gev' in batch:
            t1b = batch['tier1b_genie_valid'].bool()
            if t1b.any():
                tp4 = batch['true_branch_p4_gev'][t1b].float()  # (N, 4, 4): [lep, had, miss, sec_pi0]
                pred_lep = p4_slots[t1b, 0] + p4_slots[t1b, 2]
                pred_had = p4_slots[t1b, 1]
                pred_sec = p4_slots[t1b, 4:7].sum(dim=1)
                l_pf = l_pf + 0.25 * (
                    F.smooth_l1_loss(torch.asinh(pred_lep / 5.0), torch.asinh(tp4[:, 0] / 5.0))
                    + F.smooth_l1_loss(torch.asinh(pred_had / 5.0), torch.asinh(tp4[:, 1] / 5.0))
                    + F.smooth_l1_loss(torch.asinh(pred_sec / 5.0), torch.asinh(tp4[:, 3] / 5.0))
                )

        aux_vtx_loss = w_vtx_pv * l_pv + w_vtx_sv * l_sv + w_cl_origin * l_cl + w_slot_mask * l_slot + w_dark * l_dark
        aux_kin_loss = w_kin * l_kin + w_pflow_weak * l_pf
        total_loss = primary_loss + aux_vtx_loss + aux_kin_loss

        breakdown = {
            'total': float(total_loss.detach().item()),
            'primary': float(primary_loss.detach().item()),
            'lort_ce': float(proc_info['lort_ce']),
            'charm': float(l_charm.detach().item()),
            'pv': float(l_pv.detach().item()),
            'sv': float(l_sv.detach().item()),
            'cl_origin': float(l_cl.detach().item()),
            'slot_mask': float(l_slot.detach().item()),
            'kin': float(l_kin.detach().item()),
            'pflow': float(l_pf.detach().item()),
            'dark': float(l_dark.detach().item()),
        }
        return total_loss, primary_loss, [aux_vtx_loss, aux_kin_loss], breakdown


@torch.no_grad()
def extract_nsbi_observables_from_output(
    batch: Mapping[str, torch.Tensor],
    out: NextGenModelOutput,
) -> dict[str, np.ndarray]:
    """Extract the 13 strictly pure-observable NSBI routing arrays matching `[gen4-contract]` Sec. V.2."""
    p_ik = out.aux['seg_slot_probs'].float()   # (B, 128, 8)
    seg_m = out.aux['seg_mask'].bool()         # (B, 128)
    s_l = out.aux['s_l_slots'].float()         # (B, 6) -> idx 0 is tau-SV
    m_corr = out.aux['m_corr_slots'].float()   # (B, 6) -> idx 0 is tau-SV
    psi = out.aux['psi_nomad'].float()         # (B, 16)
    e_vis = out.aux['e_vis_hat_gev'].float()   # (B,)
    mk = batch['muon_kalman_tracks'].float()   # (B, 4, 16)
    mkm = batch['muon_kalman_mask'].bool()     # (B, 4)

    # Effective number of prongs routed to tau-SV slot 1
    p_tau_sv = p_ik[:, :, 1] * seg_m.float()
    n_sv_prongs = (p_tau_sv > 0.35).sum(dim=1).cpu().numpy().astype(np.float32)

    sl_xy = s_l[:, 0].cpu().numpy().astype(np.float32)
    m_sv_corr_gev = m_corr[:, 0].cpu().numpy().astype(np.float32)

    n_mu_veto = mkm.sum(dim=1).cpu().numpy().astype(np.float32)
    q_mu1 = torch.where(mkm[:, 0], mk[:, 0, 4], torch.zeros_like(mk[:, 0, 4])).cpu().numpy().astype(np.float32)
    q_mu2 = torch.where(mkm[:, 1], mk[:, 1, 4], torch.zeros_like(mk[:, 1, 4])).cpu().numpy().astype(np.float32)

    lambda_kink = out.aux.get('pt_kink_max', torch.zeros_like(e_vis)).cpu().numpy().astype(np.float32)
    dphi_mu_ht = out.aux.get('dphi_mu_ht', torch.zeros_like(e_vis)).cpu().numpy().astype(np.float32)
    ip_sig_max = out.aux.get('s_d0_mu_max', torch.zeros_like(e_vis)).cpu().numpy().astype(np.float32)
    m_vis_rho_gev = out.aux.get('m_vis_rho_gev', torch.zeros_like(e_vis)).cpu().numpy().astype(np.float32)
    pt_miss_hat_gev = out.aux.get('pt_miss_hat_gev', torch.zeros_like(e_vis)).cpu().numpy().astype(np.float32)
    m_gg_pi0_gev = out.aux.get('m_gg_pi0_gev', torch.zeros_like(e_vis)).cpu().numpy().astype(np.float32)

    _ = psi
    return {
        'n_sv_prongs': n_sv_prongs,
        'sl_xy': sl_xy,
        'm_sv_corr_gev': m_sv_corr_gev,
        'n_mu_veto': n_mu_veto,
        'q_mu1': q_mu1,
        'q_mu2': q_mu2,
        'lambda_kink': lambda_kink,
        'dphi_mu_ht': dphi_mu_ht,
        'ip_sig_max': ip_sig_max,
        'm_vis_rho_gev': m_vis_rho_gev,
        'e_vis_hat_gev': e_vis.cpu().numpy().astype(np.float32),
        'pt_miss_hat_gev': pt_miss_hat_gev,
        'm_gg_pi0_gev': m_gg_pi0_gev,
    }
