"""Pinpoint-UParT-13 Explicit Sub-Channel Likelihood Head, LORT Re-scaling,
Full-Dataset 256 KB State Table, and Unfrozen Live Hard-Negative NSWC-TPAUC Tail Fine-Tuning.

Strictly implements the `[gen4-contract]` v1.0 & Stage-329 Pillar-III blueprint:
  - 13 explicit physics sub-channels:
      0: tau_e
      1: tau_mu
      2: tau_1h
      3: tau_3h
      4: nue_CC_light
      5: nue_CC_charm
      6: numu_minus_CC_light
      7: numu_plus_CC_light
      8: numu_minus_CC_charm
      9: numu_plus_CC_charm
     10: NC_had_light
     11: NC_pi0_em
     12: NC_ccbar  (zero-free-parameter factorized nuclear-charm slot!)
  - Unbiased macro-class prior initialization (initial probs4 == [0.25, 0.25, 0.25, 0.25])
    and macro-class balanced LORT label over-smoothing (delta_LORT / 4 per macro-class).
  - Exact additive marginalization to `logits7`, `probs4`, `charm_logit`, and analytical
    forward-flux Bayesian likelihood ratio `tau_llr_score = T(x; d, E_vis_hat)`.
  - Unfrozen top-65% backbone Stage-3 fine-tuning with full Train32k 256 KB state table
    (`s_ema`, `u_ema`), live Top-2.5% hard-negative + short-L_tau/tau_mu re-forwarding,
    variance-reduced NSWC-TPAUC, Quantile-Pivot + UParT RNG invariance, and 4x4 INFERNO Fisher proxy.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from analysis.flavour.nextgen.protocols import marginal_four_from_seven


UPART13_NAMES: tuple[str, ...] = (
    'tau_e',
    'tau_mu',
    'tau_1h',
    'tau_3h',
    'nue_CC_light',
    'nue_CC_charm',
    'numu_minus_CC_light',
    'numu_plus_CC_light',
    'numu_minus_CC_charm',
    'numu_plus_CC_charm',
    'NC_had_light',
    'NC_pi0_em',
    'NC_ccbar',
)

# Macro-class membership of the 13 sub-channels:
# 0: nu_e CC ({4, 5}), 1: nu_mu CC ({6, 7, 8, 9}), 2: nu_tau CC ({0, 1, 2, 3}), 3: NC ({10, 11, 12})
MACRO_TO_SUB13: tuple[tuple[int, ...], ...] = (
    (4, 5),
    (6, 7, 8, 9),
    (0, 1, 2, 3),
    (10, 11, 12),
)

# Conditional within-macro-class prior pi_{sub | C(h)} summing to 1.0 within each macro class
# Ensures exact unbiased uniform macro initialization: sum_{h in C} exp(b_h) = 1.0 -> probs4 = [0.25]*4
SUB_CONDITIONAL_PRIORS: tuple[float, ...] = (
    0.175, 0.172, 0.435, 0.218,           # nu_tau: tau_e, tau_mu, tau_1h, tau_3h
    0.945, 0.055,                         # nu_e:   light, charm
    0.735, 0.195, 0.055, 0.015,           # nu_mu:  mu- light, mu+ light, mu- charm, mu+ charm
    0.665, 0.295, 0.040,                  # NC:     had_light, pi0_em, nc_ccbar
)

# Realistic FASER/PINPOINT forward-flux macro proportions (nu_e : nu_mu : nu_tau : NC)
# Background shares match FIDUCIAL_2T_BKG_SHARES ~ (17 : 58 : 25) and f_tau = 0.003
PHYS_MACRO_PRIORS: tuple[float, float, float, float] = (
    0.17 * 0.997,
    0.58 * 0.997,
    0.003,
    0.25 * 0.997,
)

# Uniform balanced training macro priors
TRAIN_MACRO_PRIORS: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)


def compute_sub13_conditional_priors_tensor(device: torch.device | None = None) -> torch.Tensor:
    """Return length-13 tensor of conditional priors pi_{sub | C(h)} normalized per macro class."""
    p = torch.tensor(SUB_CONDITIONAL_PRIORS, dtype=torch.float32, device=device)
    for idxs in MACRO_TO_SUB13:
        idx_t = torch.tensor(idxs, dtype=torch.long, device=device)
        p[idx_t] = p[idx_t] / p[idx_t].sum().clamp_min(1e-8)
    return p


def logits7_from_logits13(logits13: torch.Tensor) -> torch.Tensor:
    """Exact log-sum-exp marginalization from 13 sub-channels -> 7 canonical NextGen classes in FP32."""
    lg = logits13.float()
    return torch.stack([
        torch.logsumexp(lg[:, (4, 5)], dim=-1),            # 0: e_CC
        torch.logsumexp(lg[:, (6, 8)], dim=-1),            # 1: mu-_CC
        torch.logsumexp(lg[:, (7, 9)], dim=-1),            # 2: mu+_CC
        lg[:, 0],                                          # 3: tau_e
        lg[:, 1],                                          # 4: tau_mu
        torch.logsumexp(lg[:, (2, 3)], dim=-1),            # 5: tau_had
        torch.logsumexp(lg[:, (10, 11, 12)], dim=-1),      # 6: NC
    ], dim=-1)


def charm_logit_from_logits13(logits13: torch.Tensor) -> torch.Tensor:
    """Exact charm log-odds from the 4 charm vs 9 non-charm sub-channels in FP32."""
    lg = logits13.float()
    l_ch = torch.logsumexp(lg[:, (5, 8, 9, 12)], dim=-1)
    l_no = torch.logsumexp(lg[:, (0, 1, 2, 3, 4, 6, 7, 10, 11)], dim=-1)
    return l_ch - l_no


def bayesian_tau_llr_from_logits13(
    logits13: torch.Tensor,
    obs_domain_id: torch.Tensor,
    e_vis_hat_gev: torch.Tensor,
    *,
    lambda_charm: float = 1.5,
    lambda_pi0: float = 0.25,
) -> torch.Tensor:
    """Compute analytical forward-flux Bayesian log-likelihood ratio T(x; d, E_vis_hat) in FP32.

    Uses exact prior-odds correction `pi_phys(h | d, E_vis) / pi_train(h | d, E_vis)` plus
    explicit Charm / pi0 risk weighting `w_risk(b)`:
      T(x; d, E) = log( sum_{h in 0..3} r_h(d, E) e^{l_h} / sum_{b in 4..12} w_risk(b) r_b(d, E) e^{l_b} ).
    Consumes ONLY pure observables `(logits13, obs_domain_id, e_vis_hat_gev)`.
    """
    with torch.amp.autocast(device_type=logits13.device.type, enabled=False):
        lg = logits13.float()
        dev = lg.device
        cond_sub = compute_sub13_conditional_priors_tensor(dev)

        # Energy-dependent modulation of forward flux ratios & charm cross-section threshold
        e_gev = e_vis_hat_gev.float().clamp(min=5.0, max=5000.0)
        log_e_rel = torch.log(e_gev / 400.0)  # centered around 400 GeV

        # Domain-dependent acceptance prior adjustment (D1 silicon favors shorter/harder interactions;
        # D2 scint-only favors slightly higher NC / high-E fraction)
        dom = obs_domain_id.long()
        is_d1 = (dom == 1).float()
        is_d2 = (dom == 2).float()

        # Macro-class physical vs training log-prior ratio log(pi_phys / pi_train)
        # Mild energy slope: nu_tau flux peaks higher in energy (+0.22 * log_e_rel)
        lr_macro = torch.zeros((lg.shape[0], 4), dtype=torch.float32, device=dev)
        lr_macro[:, 0] = math.log(PHYS_MACRO_PRIORS[0] / TRAIN_MACRO_PRIORS[0]) + 0.05 * log_e_rel
        lr_macro[:, 1] = math.log(PHYS_MACRO_PRIORS[1] / TRAIN_MACRO_PRIORS[1]) - 0.04 * log_e_rel + 0.06 * is_d2
        lr_macro[:, 2] = math.log(PHYS_MACRO_PRIORS[2] / TRAIN_MACRO_PRIORS[2]) + 0.22 * log_e_rel + 0.08 * is_d1
        lr_macro[:, 3] = math.log(PHYS_MACRO_PRIORS[3] / TRAIN_MACRO_PRIORS[3]) - 0.05 * log_e_rel + 0.04 * is_d2

        # Expand macro log-ratios to 13 sub-channels + charm onset log-factor
        log_r13 = torch.zeros_like(lg)
        for m_idx, sub_ids in enumerate(MACRO_TO_SUB13):
            for sid in sub_ids:
                log_r13[:, sid] = lr_macro[:, m_idx]

        # Threshold turn-on for charm slots (5, 8, 9, 12) above E_vis ~ 80 GeV
        charm_e_mod = 0.35 * torch.tanh(0.6 * log_e_rel)
        for ch_sid in (5, 8, 9, 12):
            log_r13[:, ch_sid] = log_r13[:, ch_sid] + charm_e_mod

        # Risk weighting w_risk(b) for background channels
        log_w_risk = torch.zeros(9, dtype=torch.float32, device=dev)
        w_ch = math.log(max(1.0 + float(lambda_charm), 1.0))
        w_p0 = math.log(max(1.0 + float(lambda_pi0), 1.0))
        # Indices relative to 4..12:
        # 0:4(nue_l), 1:5(nue_c), 2:6(numu-_l), 3:7(numu+_l), 4:8(numu-_c), 5:9(numu+_c), 6:10(NC_l), 7:11(NC_pi0), 8:12(NC_cc)
        log_w_risk[1] = w_ch
        log_w_risk[4] = w_ch
        log_w_risk[5] = w_ch
        log_w_risk[8] = w_ch
        log_w_risk[7] = w_p0

        # Add log(cond_sub) normalization consistency (already baked into bias init, kept exact)
        _ = cond_sub
        log_num = torch.logsumexp(lg[:, 0:4] + log_r13[:, 0:4], dim=-1)
        log_den = torch.logsumexp(lg[:, 4:13] + log_r13[:, 4:13] + log_w_risk[None, :], dim=-1)
        return log_num - log_den


class PinpointUParT13Head(nn.Module):
    """Explicit 13-subchannel likelihood-ratio head with zero-parameter factorized NC_{ccbar} slot.

    Consumes purely observable tensors:
      `z_fuse: (B, 256)`, `phi_vtx: (B, 36)`, `psi_nomad: (B, 16)`,
      `obs_domain_id: (B,)`, `e_vis_hat_gev: (B,)`.
    """

    def __init__(
        self,
        fuse_dim: int = 256,
        phi_vtx_dim: int = 36,
        psi_nomad_dim: int = 16,
        *,
        hidden_dim: int = 256,
        dropout: float = 0.12,
        lambda_charm: float = 1.5,
        lambda_pi0: float = 0.25,
    ) -> None:
        super().__init__()
        self.fuse_dim = int(fuse_dim)
        self.phi_vtx_dim = int(phi_vtx_dim)
        self.psi_nomad_dim = int(psi_nomad_dim)
        self.lambda_charm = float(lambda_charm)
        self.lambda_pi0 = float(lambda_pi0)

        in_dim = self.fuse_dim + self.phi_vtx_dim + self.psi_nomad_dim + 4
        self.stem = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        # Only 12 trainable sub-channel projections (slot 12 NC_ccbar has ZERO free parameters!)
        self.proj_12 = nn.Linear(hidden_dim, 12)
        self.register_buffer('cond_priors_13', compute_sub13_conditional_priors_tensor())
        self.reset_unbiased_prior_bias()

    def reset_unbiased_prior_bias(self) -> None:
        """Initialize biases b_h = ln(pi_{sub|C(h)}) so initial probs4 == [0.25, 0.25, 0.25, 0.25]."""
        with torch.no_grad():
            self.proj_12.bias.copy_(torch.log(self.cond_priors_13[:12].clamp_min(1e-6)))
            nn.init.normal_(self.proj_12.weight, mean=0.0, std=0.02)

    def forward(
        self,
        z_fuse: torch.Tensor,
        phi_vtx: torch.Tensor,
        psi_nomad: torch.Tensor,
        obs_domain_id: torch.Tensor,
        e_vis_hat_gev: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        with torch.amp.autocast(device_type=z_fuse.device.type, enabled=False):
            zf = z_fuse.float()
            pv = phi_vtx.float()
            pn = psi_nomad.float()
            dom = obs_domain_id.long().clamp(0, 3)
            dom_oh = F.one_hot(dom, num_classes=4).float()

            feat = torch.cat([zf, pv, pn, dom_oh], dim=-1)
            h = self.stem(feat)
            logits12 = self.proj_12(h)  # (B, 12)

            # Slot 12: Zero-free-parameter factorized nuclear-charm slot:
            # l_12 = l_10 + sg(1/3 * [(l5 - l4) + (l8 - l6) + (l9 - l7)]) + psi_in_jet(x) + log pi_cc(E_vis)
            charm_shift = (
                (logits12[:, 5] - logits12[:, 4])
                + (logits12[:, 8] - logits12[:, 6])
                + (logits12[:, 9] - logits12[:, 7])
            ) / 3.0
            charm_shift = charm_shift.detach().clamp(-4.0, 4.0)

            # Analytical in-jet charm-SV indicator from purely observable vertex & muon-isolation invariants:
            # phi_vtx[:, 2] = S_{L,c}, phi_vtx[:, 3] = M_{corr,c} proxy, psi_nomad[:, 11] = muon iso
            si_active = (pv[:, :4].abs().sum(dim=-1) > 1e-6).float()
            s_lc = pv[:, 2].clamp(min=0.0, max=12.0)
            m_corr_c = pv[:, 3].clamp(min=0.0, max=6.0)
            psi_in_jet = si_active * (0.25 * torch.tanh(0.4 * (s_lc - 1.5)) + 0.15 * torch.tanh(0.5 * (m_corr_c - 1.2)))

            e_gev = e_vis_hat_gev.float().clamp(min=5.0, max=5000.0)
            # Nuclear charm production fraction increases gently with E_vis (~2% @ 100 GeV -> ~6% @ 1 TeV)
            pi_cc = (0.015 + 0.045 * torch.sigmoid(0.8 * torch.log(e_gev / 350.0))).clamp(1e-4, 0.25)
            log_pi_cc = torch.log(pi_cc)

            l12 = logits12[:, 10] + charm_shift + psi_in_jet + log_pi_cc
            logits13 = torch.cat([logits12, l12.unsqueeze(-1)], dim=-1)  # (B, 13)

            logits7 = logits7_from_logits13(logits13)
            probs4 = marginal_four_from_seven(logits7, is_probs=False)
            charm_logit = charm_logit_from_logits13(logits13)
            tau_llr = bayesian_tau_llr_from_logits13(
                logits13,
                obs_domain_id,
                e_vis_hat_gev,
                lambda_charm=self.lambda_charm,
                lambda_pi0=self.lambda_pi0,
            )
            return {
                'logits13': logits13,
                'logits7': logits7,
                'probs4': probs4,
                'charm_logit': charm_logit,
                'tau_llr_score': tau_llr,
            }


def compute_upart13_lort_loss(
    logits13: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    *,
    delta_lort: float = 0.08,
    lambda_mag: float = 0.01,
    sample_weight: torch.Tensor | None = None,
    denom_override: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Hierarchical 13-subchannel marginal CE with macro-class balanced LORT & magnitude penalty.

    Implements the exact macro-balanced LORT smoothing from `[gen4-contract]`:
      - Total smoothing mass `delta_lort` is split equally across the 4 macro-classes (`delta_lort / 4` each),
        then distributed across sub-channels proportional to `pi_{sub|C(h)}` to avoid class-count bias.
      - Supervised hierarchically using `label`, `label_7`, `charm_flag`, `tau_n_charged_prongs_si`,
        and `true_tau_submode` (when Tier-1B is present; marginalizes `LSE(l2, l3)` cleanly when absent).
    """
    with torch.amp.autocast(device_type=logits13.device.type, enabled=False):
        lg = logits13.float()
        bsz = int(lg.shape[0])
        dev = lg.device
        cond_sub = compute_sub13_conditional_priors_tensor(dev)

        y4 = batch['label'].long()
        l7 = batch['label_7'].long() if 'label_7' in batch else torch.full((bsz,), -1, dtype=torch.long, device=dev)
        cf = batch['charm_flag'].long() if 'charm_flag' in batch else torch.zeros((bsz,), dtype=torch.long, device=dev)
        cv = (batch['charm_valid'].long() > 0) if 'charm_valid' in batch else torch.zeros((bsz,), dtype=torch.bool, device=dev)
        submode = (
            batch['true_tau_submode'].long()
            if 'true_tau_submode' in batch
            else torch.full((bsz,), -1, dtype=torch.long, device=dev)
        )
        t1b_valid = (
            batch['tier1b_genie_valid'].bool()
            if 'tier1b_genie_valid' in batch
            else torch.zeros((bsz,), dtype=torch.bool, device=dev)
        )
        prongs = (
            batch['tau_n_charged_prongs_si'].long()
            if 'tau_n_charged_prongs_si' in batch
            else torch.full((bsz,), -1, dtype=torch.long, device=dev)
        )

        # Build compatibility mask (B, 13) over allowed fine sub-channels for each event
        compat = torch.zeros((bsz, 13), dtype=torch.bool, device=dev)

        # Macro fallbacks from label_4
        m_e = y4 == 0
        m_mu = y4 == 1
        m_tau = y4 == 2
        m_nc = y4 == 3

        # nu_e CC -> {4: light, 5: charm}
        compat[m_e, 4] = (~cv[m_e]) | (cf[m_e] == 0)
        compat[m_e, 5] = (~cv[m_e]) | (cf[m_e] == 1)

        # nu_mu CC -> {6: mu- light, 7: mu+ light, 8: mu- charm, 9: mu+ charm}
        is_mu_minus = m_mu & (l7 == 1)
        is_mu_plus = m_mu & (l7 == 2)
        is_mu_unk = m_mu & (~(is_mu_minus | is_mu_plus))
        compat[is_mu_minus, 6] = (~cv[is_mu_minus]) | (cf[is_mu_minus] == 0)
        compat[is_mu_minus, 8] = (~cv[is_mu_minus]) | (cf[is_mu_minus] == 1)
        compat[is_mu_plus, 7] = (~cv[is_mu_plus]) | (cf[is_mu_plus] == 0)
        compat[is_mu_plus, 9] = (~cv[is_mu_plus]) | (cf[is_mu_plus] == 1)
        compat[is_mu_unk, 6] = (~cv[is_mu_unk]) | (cf[is_mu_unk] == 0)
        compat[is_mu_unk, 7] = (~cv[is_mu_unk]) | (cf[is_mu_unk] == 0)
        compat[is_mu_unk, 8] = (~cv[is_mu_unk]) | (cf[is_mu_unk] == 1)
        compat[is_mu_unk, 9] = (~cv[is_mu_unk]) | (cf[is_mu_unk] == 1)

        # nu_tau CC -> {0: tau_e, 1: tau_mu, 2: tau_1h, 3: tau_3h}
        is_te = m_tau & ((l7 == 3) | (t1b_valid & (submode == 0)))
        is_tmu = m_tau & ((l7 == 4) | (t1b_valid & (submode == 1)))
        is_thad = m_tau & ((l7 == 5) | (t1b_valid & ((submode == 2) | (submode == 3))))
        is_tunk = m_tau & (~(is_te | is_tmu | is_thad))
        compat[is_te, 0] = True
        compat[is_tmu, 1] = True
        # Distinguish tau_1h vs tau_3h via Tier-1B submode when valid, else marginalize {2, 3}
        is_t1h = is_thad & (t1b_valid & (submode == 2))
        is_t3h = is_thad & (t1b_valid & (submode == 3))
        is_thad_marg = is_thad & (~(is_t1h | is_t3h))
        compat[is_t1h, 2] = True
        compat[is_t3h, 3] = True
        compat[is_thad_marg, 2] = True
        compat[is_thad_marg, 3] = True
        compat[is_tunk, 0:4] = True

        # NC -> {10: NC_had_light, 11: NC_pi0_em, 12: NC_ccbar}
        compat[m_nc, 10] = (~cv[m_nc]) | (cf[m_nc] == 0)
        compat[m_nc, 11] = (~cv[m_nc]) | (cf[m_nc] == 0)
        compat[m_nc, 12] = (~cv[m_nc]) | (cf[m_nc] == 1)

        # Marginal log-likelihood over compatible target set: log(sum_{h in compat} softmax(lg)_h)
        log_z = torch.logsumexp(lg, dim=-1)  # (B,)
        masked_lg = lg.masked_fill(~compat, -1e9)
        log_pos = torch.logsumexp(masked_lg, dim=-1)
        nll_hard = log_z - log_pos  # (B,)

        # Optional soft prior within tau_had from `tau_n_charged_prongs_si` (1 prong -> tau_1h, >=3 -> tau_3h)
        soft_had_loss = lg.sum() * 0.0
        has_prong_hint = is_thad_marg & (prongs > 0)
        if has_prong_hint.any():
            p_had = F.softmax(lg[has_prong_hint][:, 2:4], dim=-1).clamp_min(1e-7)
            tgt_1h = torch.where(prongs[has_prong_hint] >= 3, 0.15, 0.85)
            soft_had_loss = -(tgt_1h * torch.log(p_had[:, 0]) + (1.0 - tgt_1h) * torch.log(p_had[:, 1])).mean() * 0.15

        # Macro-class balanced LORT smoothing target (delta_lort / 4 per macro-class!)
        log_p13 = lg - log_z.unsqueeze(-1)
        smooth_prior = torch.zeros(13, dtype=torch.float32, device=dev)
        for idxs in MACRO_TO_SUB13:
            idx_t = torch.tensor(idxs, dtype=torch.long, device=dev)
            smooth_prior[idx_t] = 0.25 * cond_sub[idx_t]
        kl_smooth = -(smooth_prior.unsqueeze(0) * log_p13).sum(dim=-1)

        dl = float(max(0.0, min(0.5, delta_lort)))
        per_ev = (1.0 - dl) * nll_hard + dl * kl_smooth

        # Logit magnitude & inter-class variance regularization (LORT)
        lg_centered = lg - lg.mean(dim=-1, keepdim=True)
        l_mag = float(lambda_mag) * lg_centered.square().mean()

        if sample_weight is not None:
            w = sample_weight.float().clamp_min(0.0)
        elif 'sample_weight' in batch and batch['sample_weight'] is not None:
            w = batch['sample_weight'].float().clamp_min(0.0)
        else:
            w = torch.ones_like(per_ev)
        denom = (
            torch.as_tensor(denom_override, dtype=torch.float32, device=dev).clamp_min(1e-8)
            if denom_override is not None
            else w.sum().clamp_min(1e-8)
        )
        l_ce = (per_ev * w).sum() / denom
        total = l_ce + soft_had_loss + l_mag
        return total, {
            'lort_ce': float(l_ce.detach().item()),
            'lort_mag': float(l_mag.detach().item()),
            'nll_hard': float((nll_hard * w).sum().detach().item() / max(float(w.sum().item()), 1e-8)),
        }


# ===========================================================================
# Stage-3 Unfrozen Backbone Tail Optimization: 256 KB Table + Live Hard-Neg NSWC-TPAUC
# ===========================================================================

class FullDatasetScoreTable:
    """Exact full-dataset 256 KB sliding state table `(s_ema[N], u_ema[N])` for Train32k.

    At N=32,000, two float32 vectors occupy exactly 256,000 bytes (~250 KB), enabling
    zero-approximation live mining of per-domain Top-2.5% hard negatives (50% Charm)
    and boundary short-L_tau / tau_mu positives every batch.
    """

    def __init__(
        self,
        n_events: int,
        labels4: np.ndarray,
        obs_domain_id: np.ndarray,
        *,
        charm_flag: np.ndarray | None = None,
        label_7: np.ndarray | None = None,
        tau_decay_L3d_mm: np.ndarray | None = None,
        ema_momentum: float = 0.85,
    ) -> None:
        self.n_events = int(n_events)
        self.ema_momentum = float(ema_momentum)
        self.s_ema = np.zeros(self.n_events, dtype=np.float32)
        self.u_ema = np.zeros(self.n_events, dtype=np.float32)
        self.initialized = np.zeros(self.n_events, dtype=bool)

        self.y4 = np.asarray(labels4, dtype=np.int64)
        self.dom = np.asarray(obs_domain_id, dtype=np.int64)
        self.is_tau = self.y4 == 2
        self.is_bg = ~self.is_tau
        self.is_charm_bg = (
            self.is_bg & (np.asarray(charm_flag, dtype=np.int64) > 0)
            if charm_flag is not None
            else np.zeros(self.n_events, dtype=bool)
        )
        # Boundary positives: tau_mu (label_7==4) or short-L_tau (< 30 mm)
        is_tmu = (np.asarray(label_7, dtype=np.int64) == 4) if label_7 is not None else np.zeros(self.n_events, dtype=bool)
        is_short = (
            self.is_tau & (np.asarray(tau_decay_L3d_mm, dtype=np.float32) > 0.0) & (np.asarray(tau_decay_L3d_mm, dtype=np.float32) < 30.0)
            if tau_decay_L3d_mm is not None
            else np.zeros(self.n_events, dtype=bool)
        )
        self.is_boundary_tau = self.is_tau & (is_tmu | is_short)
        if not self.is_boundary_tau.any():
            self.is_boundary_tau = self.is_tau.copy()

    @property
    def nbytes(self) -> int:
        return int(self.s_ema.nbytes + self.u_ema.nbytes)

    def update(self, indices: np.ndarray | torch.Tensor, scores: np.ndarray | torch.Tensor) -> None:
        idx = np.asarray(indices.detach().cpu().numpy() if torch.is_tensor(indices) else indices, dtype=np.int64)
        sc = np.asarray(scores.detach().float().cpu().numpy() if torch.is_tensor(scores) else scores, dtype=np.float32)
        if len(idx) == 0:
            return
        m = self.ema_momentum
        first = ~self.initialized[idx]
        if first.any():
            i_f = idx[first]
            self.s_ema[i_f] = sc[first]
            self.u_ema[i_f] = np.exp(np.clip(sc[first], -6.0, 6.0))
            self.initialized[i_f] = True
        upd = ~first
        if upd.any():
            i_u = idx[upd]
            self.s_ema[i_u] = m * self.s_ema[i_u] + (1.0 - m) * sc[upd]
            self.u_ema[i_u] = m * self.u_ema[i_u] + (1.0 - m) * np.exp(np.clip(sc[upd], -6.0, 6.0))

    def sample_hard_batch_indices(
        self,
        rng: np.random.Generator,
        *,
        n_random: int = 80,
        n_hard_neg: int = 32,
        n_boundary_pos: int = 16,
        top_frac: float = 0.025,
    ) -> np.ndarray:
        """Sample composite Stage-3 batch: 80 random + 32 live Top-2.5% hard negs (half Charm) + 16 boundary tau."""
        idx_rand = rng.integers(0, self.n_events, size=n_random, endpoint=False)

        # Per-domain Top-2.5% hard negatives, targeting 50% Charm hard negatives
        hard_pool_charm: list[np.ndarray] = []
        hard_pool_other: list[np.ndarray] = []
        for d in (1, 2, 3):
            bg_d = np.flatnonzero(self.is_bg & (self.dom == d))
            if len(bg_d) == 0:
                continue
            k_top = max(8, int(math.ceil(top_frac * len(bg_d))))
            k_top = min(k_top, len(bg_d))
            sc_d = self.s_ema[bg_d]
            top_local = np.argpartition(sc_d, -k_top)[-k_top:]
            top_ids = bg_d[top_local]
            ch_m = self.is_charm_bg[top_ids]
            if ch_m.any():
                hard_pool_charm.append(top_ids[ch_m])
            if (~ch_m).any():
                hard_pool_other.append(top_ids[~ch_m])

        n_ch_target = n_hard_neg // 2
        charm_cand = np.concatenate(hard_pool_charm) if hard_pool_charm else np.flatnonzero(self.is_charm_bg)
        other_cand = np.concatenate(hard_pool_other) if hard_pool_other else np.flatnonzero(self.is_bg & (~self.is_charm_bg))
        if len(charm_cand) == 0:
            charm_cand = np.flatnonzero(self.is_bg)
        if len(other_cand) == 0:
            other_cand = np.flatnonzero(self.is_bg)

        idx_ch = rng.choice(charm_cand, size=n_ch_target, replace=(len(charm_cand) < n_ch_target))
        idx_ot = rng.choice(other_cand, size=n_hard_neg - n_ch_target, replace=(len(other_cand) < (n_hard_neg - n_ch_target)))

        pos_cand = np.flatnonzero(self.is_boundary_tau)
        idx_pos = rng.choice(pos_cand, size=n_boundary_pos, replace=(len(pos_cand) < n_boundary_pos))

        return np.concatenate([idx_rand, idx_ch, idx_ot, idx_pos]).astype(np.int64)


@dataclass
class Stage3TailConfig:
    """Hyperparameters for Stage-3 unfrozen top-65% NSWC-TPAUC + Quantile-Pivot + INFERNO."""

    unfreeze_top_frac: float = 0.65
    fpr_target: float = 0.01
    tpr_floor: float = 0.25
    margin: float = 0.55
    score_bound: float = 6.0
    lambda_tpauc: float = 0.65
    lambda_nll: float = 0.60
    lambda_kl: float = 0.20
    lambda_pivot: float = 0.12
    lambda_inferno: float = 0.08
    lambda_charm: float = 1.5
    rng_eps: float = 0.03
    lr_head: float = 3.0e-5
    weight_decay: float = 1.0e-3
    passes: int = 4
    n_random: int = 80
    n_hard_neg: int = 32
    n_boundary_pos: int = 16


def configure_unfrozen_top_backbone(
    model: nn.Module,
    *,
    top_frac: float = 0.65,
    lr_head: float = 3.0e-5,
    weight_decay: float = 1.0e-3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Freeze bottom `(1 - top_frac)` parameters of each branch and unfreeze top `top_frac` + head.

    Maintains deterministic parameter ordering and returns AdamW parameter groups with
    gentle layerwise decay `xi = 0.65` across unfrozen backbone blocks.
    """
    frac = float(min(max(top_frac, 0.05), 1.0))
    head_params: list[nn.Parameter] = []
    backbone_unfrozen: list[nn.Parameter] = []
    n_total = 0
    n_trainable = 0

    by_branch: dict[str, list[tuple[str, nn.Parameter]]] = {}
    for name, p in model.named_parameters():
        n_total += p.numel()
        prefix = name.split('.', 1)[0]
        by_branch.setdefault(prefix, []).append((name, p))

    for branch, items in by_branch.items():
        if branch == 'head':
            for _, p in items:
                p.requires_grad_(True)
                head_params.append(p)
                n_trainable += p.numel()
            continue
        # Freeze earliest (1 - frac) fraction of parameters in this branch, unfreeze top frac
        branch_tot = sum(p.numel() for _, p in items)
        cutoff = int(math.floor((1.0 - frac) * branch_tot))
        cum = 0
        for _, p in items:
            if cum + p.numel() <= cutoff:
                p.requires_grad_(False)
            else:
                p.requires_grad_(True)
                backbone_unfrozen.append(p)
                n_trainable += p.numel()
            cum += p.numel()

    def _split_wd(plist: list[nn.Parameter], lr: float) -> list[dict[str, Any]]:
        dec, nodec = [], []
        for p in plist:
            if p.ndim <= 1:
                nodec.append(p)
            else:
                dec.append(p)
        out = []
        if dec:
            out.append({'params': dec, 'lr': lr, 'weight_decay': weight_decay})
        if nodec:
            out.append({'params': nodec, 'lr': lr, 'weight_decay': 0.0})
        return out

    pgs = _split_wd(head_params, lr_head) + _split_wd(backbone_unfrozen, lr_head * 0.65)
    info = {
        'total_params': int(n_total),
        'trainable_params': int(n_trainable),
        'trainable_frac': float(n_trainable / max(n_total, 1)),
    }
    return pgs, info


def compute_nswc_tpauc_loss(
    scores: torch.Tensor,
    is_tau: torch.Tensor,
    obs_domain_id: torch.Tensor,
    state_table: FullDatasetScoreTable | None = None,
    *,
    fpr_target: float = 0.01,
    margin: float = 0.55,
    score_bound: float = 6.0,
    is_charm_bg: torch.Tensor | None = None,
    charm_boost: float = 1.6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Variance-reduced per-domain NSWC-TPAUC surrogate loss (NeurIPS 2023 / TMLR 2025) in FP32.

    Computes two-way partial-AUC / quantile-anchored margin loss over live re-forwarded
    hard negatives (with explicit Charm weight `charm_boost`) and live boundary positives,
    anchored against the global domain quantile thresholds `eta_d^-` from the 256 KB state table.
    """
    with torch.amp.autocast(device_type=scores.device.type, enabled=False):
        b = float(max(1.0, score_bound))
        sc = b * torch.tanh(scores.float() / b)
        pos_m = is_tau.bool()
        neg_m = ~pos_m
        dom = obs_domain_id.long()

        if not pos_m.any() or not neg_m.any():
            return sc.sum() * 0.0, {'tpauc_loss': 0.0}

        loss_terms: list[torch.Tensor] = []
        q_tails: dict[str, float] = {}
        for d in (1, 2, 3):
            dm_pos = pos_m & (dom == d)
            dm_neg = neg_m & (dom == d)
            if not dm_pos.any() or not dm_neg.any():
                continue
            sp = sc[dm_pos]
            sn = sc[dm_neg]

            # Obtain domain tail threshold eta_d^- from full 256 KB table when available
            if state_table is not None:
                bg_d = state_table.s_ema[state_table.is_bg & (state_table.dom == d)]
                if len(bg_d) >= 50:
                    q_raw = float(np.quantile(bg_d, 1.0 - float(fpr_target)))
                    eta_neg = float(b * math.tanh(q_raw / b))
                else:
                    eta_neg = float(torch.quantile(sn.detach(), max(0.5, 1.0 - float(fpr_target))).item())
            else:
                eta_neg = float(torch.quantile(sn.detach(), max(0.5, 1.0 - float(fpr_target))).item())
            q_tails[f'eta_d{d}'] = eta_neg

            # Hard-negative weights: highlight tail negatives near/above eta_neg and Charm background
            with torch.no_grad():
                w_neg = torch.sigmoid((sn.detach() - (eta_neg - 0.8)) / 0.5)
                if is_charm_bg is not None:
                    ch_d = is_charm_bg[dm_neg].bool()
                    w_neg = torch.where(ch_d, w_neg * float(charm_boost), w_neg)
                w_neg = w_neg / w_neg.sum().clamp_min(1e-8)

                # Two-way focus on near-boundary / salvageable positives
                w_pos = 0.20 + 0.80 * torch.sigmoid((sp.detach() - (eta_neg - 2.0)) / 0.8)
                w_pos = w_pos / w_pos.sum().clamp_min(1e-8)

            diff = sn.unsqueeze(0) - sp.unsqueeze(1) + float(margin)  # (Np, Nn)
            pair = F.softplus(diff)
            w_pair = w_pos.unsqueeze(1) * w_neg.unsqueeze(0)
            l_dom = (w_pair * pair).sum()

            # Direct tail-exceedance CVaR hinge for negatives above eta_neg
            l_cvar = F.relu(sn - eta_neg + 0.15).square()
            l_cvar_w = (w_neg * l_cvar).sum() * 1.5
            loss_terms.append(l_dom + 0.35 * l_cvar_w)

        if not loss_terms:
            # Global fallback
            sp = sc[pos_m]
            sn = sc[neg_m]
            l_glob = F.softplus(sn.unsqueeze(0) - sp.unsqueeze(1) + float(margin)).mean()
            return l_glob, {'tpauc_loss': float(l_glob.detach().item())}

        total = torch.stack(loss_terms).mean()
        return total, {'tpauc_loss': float(total.detach().item()), **q_tails}


def compute_quantile_pivot_and_rng_loss(
    scores_nom: torch.Tensor,
    scores_pert: torch.Tensor | None,
    is_tau: torch.Tensor,
    e_vis_hat_gev: torch.Tensor,
) -> torch.Tensor:
    """Zero-OOD-leakage Quantile-Pivot + UParT RNG invariance on high-quantile background tail.

    Penalizes:
      1. Background tail sensitivity to in-distribution delayed-neutron / dark-noise / embedding
         perturbations (`scores_pert` vs `scores_nom`) focused on the top-10% background tail.
      2. Energy correlation of top-10% background tail scores (Pivot decorrelation against `log(E_vis)`).
    """
    with torch.amp.autocast(device_type=scores_nom.device.type, enabled=False):
        sn = scores_nom.float()
        neg = ~is_tau.bool()
        if int(neg.sum().item()) < 8:
            return sn.sum() * 0.0

        sn_bg = sn[neg]
        q90 = torch.quantile(sn_bg.detach(), 0.90)
        tail_w = torch.sigmoid((sn_bg.detach() - q90) / 0.5)
        tail_w = tail_w / tail_w.sum().clamp_min(1e-8)

        loss = sn.sum() * 0.0
        if scores_pert is not None:
            sp_bg = scores_pert.float()[neg]
            loss = loss + (tail_w * (sp_bg - sn_bg).square()).sum()

        # Pivot decorrelation between background tail scores and log(E_vis_hat)
        le = torch.log(e_vis_hat_gev.float()[neg].clamp(min=5.0, max=5000.0))
        le_c = le - (tail_w * le).sum()
        sc_c = sn_bg - (tail_w * sn_bg).sum()
        cov = (tail_w * le_c * sc_c).sum()
        var_e = (tail_w * le_c.square()).sum().clamp_min(1e-4)
        var_s = (tail_w * sc_c.square()).sum().clamp_min(1e-4)
        corr_sq = (cov * cov) / (var_e * var_s)
        return loss + 0.25 * corr_sq


def compute_inferno_fisher_proxy(
    tau_llr: torch.Tensor,
    is_tau: torch.Tensor,
    is_charm_bg: torch.Tensor,
    *,
    n_soft_bins: int = 4,
    delta_sys: float = 0.0159,
    sigma_charm: float = 0.30,
) -> torch.Tensor:
    """Differentiable 4-bin profiled Fisher-information proxy maximizing discovery significance Z0."""
    with torch.amp.autocast(device_type=tau_llr.device.type, enabled=False):
        sc = tau_llr.float()
        pos = is_tau.bool()
        neg = ~pos
        if not pos.any() or int(neg.sum().item()) < 10:
            return sc.sum() * 0.0

        # Smooth Gaussian-kernel soft binning over the upper tail [q75 .. max]
        lo = float(torch.quantile(sc.detach(), 0.75).item())
        hi = float(sc.detach().max().item())
        if hi <= lo + 0.1:
            hi = lo + 1.0
        centers = torch.linspace(lo, hi, n_soft_bins, device=sc.device, dtype=torch.float32)
        bw = max((hi - lo) / float(n_soft_bins), 0.25)
        probs_bin = F.softmax(-0.5 * ((sc.unsqueeze(-1) - centers.unsqueeze(0)) / bw).square(), dim=-1)  # (B, K)

        # Scale batch counts to HL-LHC relative flux (f_tau = 0.003)
        s_vec = probs_bin[pos].mean(dim=0) * 540.0
        b_vec = probs_bin[neg].mean(dim=0) * 179460.0 + 1.0
        b_ch = (
            probs_bin[neg & is_charm_bg.bool()].mean(dim=0) * 8000.0
            if (neg & is_charm_bg.bool()).any()
            else torch.zeros_like(b_vec)
        )

        # Effective profiled Fisher significance sum_k S_k^2 / (B_k + (delta_sys*B_k)^2 + (sigma_charm*B_ch_k)^2)
        var_k = b_vec + (float(delta_sys) * b_vec).square() + (float(sigma_charm) * b_ch).square()
        q0_approx = (s_vec.square() / var_k.clamp_min(1.0)).sum()
        # Maximize log(1 + q0)
        return -torch.log1p(q0_approx)
