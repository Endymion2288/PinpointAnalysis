"""Base interfaces, calibrated Student-t vertex losses, Collaboration T3 projection & Gate-B checks."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class NextGenModelOutput:
    """Standardized FP32 output contract across all PINPOINT next-generation architectures."""

    logits7: torch.Tensor                  # (B, 7): [e, mu-, mu+, tau_e, tau_mu, tau_had, NC]
    probs4: torch.Tensor                   # (B, 4): [e_CC, mu_CC, tau_CC, NC]
    charm_logit: torch.Tensor              # (B,)
    v_pv_mm: torch.Tensor                  # (B, 3) primary vertex in world mm
    scale_pv_mm: torch.Tensor              # (B, 3) Student-t scale sigma in mm (positive)
    sv_disp_mm: torch.Tensor | None = None # (B, 3) optional secondary vertex displacement (v_SV - v_PV) in mm
    kin_pred: torch.Tensor | None = None   # (B, 4) [log1p(pz), log1p(pt), cos_dphi, sin_dphi]
    aux: dict[str, Any] = field(default_factory=dict)

    def tau_score(self, charm_penalty: float = 0.0) -> torch.Tensor:
        """Compute Neyman-Pearson log-likelihood ratio score S_tau(x) in FP32."""
        p = self.probs4.float()
        p_tau = p[:, 2].clamp(min=1e-8, max=1.0 - 1e-7)
        p_bg = (p[:, 0] + p[:, 1] + p[:, 3]).clamp_min(1e-8)
        if charm_penalty > 0.0:
            p_bg = p_bg + float(charm_penalty) * torch.sigmoid(self.charm_logit.float())
        return torch.log(p_tau) - torch.log(p_bg)


def marginal_four_from_seven(logits_or_probs: torch.Tensor, *, is_probs: bool = False) -> torch.Tensor:
    """Exact marginalization from 7 fine physics classes -> 4 canonical flavour classes in FP32."""
    x = logits_or_probs.float()
    p7 = x if is_probs else F.softmax(x, dim=-1)
    p_e = p7[:, 0]
    p_mu = p7[:, 1] + p7[:, 2]
    p_tau = p7[:, 3] + p7[:, 4] + p7[:, 5]
    p_nc = p7[:, 6]
    p4 = torch.stack([p_e, p_mu, p_tau, p_nc], dim=-1)
    return p4 / p4.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def conditional_three_from_four(probs4: torch.Tensor) -> torch.Tensor:
    """Collaboration matched-slice projection: T4 -> Conditional T3 (nu_e, nu_mu, NC | non-tau)."""
    p = probs4.float()
    p3 = torch.stack([p[:, 0], p[:, 1], p[:, 3]], dim=-1).clamp_min(1e-8)
    return p3 / p3.sum(dim=-1, keepdim=True)


def student_t_vertex_loss(
    v_pred_mm: torch.Tensor,
    scale_mm: torch.Tensor,
    v_true_mm: torch.Tensor,
    vertex_valid: torch.Tensor,
    *,
    df: float = 3.0,
    sample_weight: torch.Tensor | None = None,
    denom_override: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Heavy-tailed Student-t negative log-likelihood in FP32 (resolving Stage 325 324-13 pull blowup).

    For r_a = (v_pred_a - v_true_a) / sigma_a, Student-t(df) NLL per axis up to an additive constant is:
        0.5 * (df + 1) * log(1 + r_a^2 / df) + log(sigma_a).
    At stationary point E[(df + 1) * r_a^2 / (df + r_a^2)] = 1, yielding calibrated 68% pull coverage
    even in the presence of non-Gaussian hadronic/wide-gap shower outliers.
    """
    valid = vertex_valid > 0
    if not valid.any():
        return (v_pred_mm.sum() + scale_mm.sum()) * 0.0
    with torch.amp.autocast(device_type=v_pred_mm.device.type, enabled=False):
        vp = v_pred_mm[valid].float()
        vt = v_true_mm[valid].float()
        sig = scale_mm[valid].float().clamp(min=0.015, max=250.0)
        nu = float(max(1.5, df))
        z2 = ((vp - vt) / sig).square()
        nll_axis = 0.5 * (nu + 1.0) * torch.log1p(z2 / nu) + torch.log(sig) + 4.5
        nll_ev = nll_axis.mean(dim=-1)
        w = (
            sample_weight[valid].float().clamp_min(0.0)
            if sample_weight is not None
            else torch.ones_like(nll_ev)
        )
        denom = (
            torch.as_tensor(denom_override, dtype=torch.float32, device=nll_ev.device).clamp_min(1e-8)
            if denom_override is not None
            else w.sum().clamp_min(1e-8)
        )
        return (nll_ev * w).sum() / denom


def compute_student_t_pulls(
    v_pred_mm: torch.Tensor,
    scale_mm: torch.Tensor,
    v_true_mm: torch.Tensor,
    vertex_valid: torch.Tensor,
    *,
    df: float = 3.0,
) -> dict[str, list[float]]:
    """Compute per-axis pull p68(|r|) normalized to unit Gaussian 68.27% quantile for Student-t(df)."""
    valid = vertex_valid > 0
    if not valid.any():
        return {'p68_abs_pull': [0.0, 0.0, 0.0], 'median_sigma_mm': [0.0, 0.0, 0.0]}
    # For Student-t(nu=3), the 68.27% central quantile is q68(t_3) = 1.1969 (vs 1.0000 for N(0,1))
    t_q68 = 1.1969 if abs(df - 3.0) < 0.1 else math.sqrt(df * (math.pow(0.3173, -2.0 / df) - 1.0) * 0.35 + 1.0)
    with torch.no_grad():
        raw = ((v_pred_mm[valid].float() - v_true_mm[valid].float()) / scale_mm[valid].float().clamp_min(1e-4)).abs()
        norm_pull = raw / t_q68
        p68 = torch.quantile(norm_pull, 0.6827, dim=0).cpu().tolist()
        med_s = torch.quantile(scale_mm[valid].float(), 0.50, dim=0).cpu().tolist()
    return {'p68_abs_pull': [float(x) for x in p68], 'median_sigma_mm': [float(x) for x in med_s]}


class NextGenFlavourModel(nn.Module):
    """Abstract base class for PINPOINT next-generation architectures enforcing Gate A/B/C contracts."""

    def forward_NextGen(self, batch: dict[str, torch.Tensor]) -> NextGenModelOutput:
        raise NotImplementedError

    def forward(self, batch: dict[str, torch.Tensor], return_dataclass: bool = False) -> dict[str, Any] | NextGenModelOutput:
        out = self.forward_NextGen(batch)
        if return_dataclass:
            return out
        res: dict[str, Any] = {
            'logits': out.logits7,
            'probs4': out.probs4,
            'charm_logit': out.charm_logit,
            'charm_prob': torch.sigmoid(out.charm_logit.float()),
            'v_hat': out.v_pv_mm,
            'scale_pv_mm': out.scale_pv_mm,
        }
        if out.sv_disp_mm is not None:
            res['sv_disp_mm'] = out.sv_disp_mm
        if out.kin_pred is not None:
            res['kin_pred'] = out.kin_pred
        res.update(out.aux)
        return res

    def predict_logits4(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.forward_NextGen(batch)
        return torch.log(out.probs4.float().clamp_min(1e-8))

    def branch_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """Group trainable parameters by top-level submodule name for Gate-B gradient health checks."""
        groups: dict[str, list[nn.Parameter]] = {}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            prefix = name.split('.', 1)[0]
            groups.setdefault(prefix, []).append(param)
        return groups


def check_branch_gradient_health(
    model: nn.Module,
    loss: torch.Tensor,
    *,
    min_grad_l2: float = 1.0e-7,
) -> dict[str, Any]:
    """Gate-B helper: run backward and verify finite, non-zero gradients in every parameter branch."""
    if not torch.isfinite(loss):
        raise FloatingPointError(f'Non-finite loss in Gate-B check: {loss.item()}')
    model.zero_grad(set_to_none=True)
    loss.backward()

    if hasattr(model, 'branch_parameter_groups'):
        groups = model.branch_parameter_groups()
    else:
        groups = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                groups.setdefault(name.split('.', 1)[0], []).append(p)

    norms: dict[str, float] = {}
    dead_branches: list[str] = []
    nonfinite_branches: list[str] = []
    for br, plist in groups.items():
        sq = 0.0
        bad = False
        for p in plist:
            if p.grad is None:
                continue
            g = p.grad.detach().float()
            if not torch.isfinite(g).all():
                bad = True
                break
            sq += float(g.square().sum().item())
        if bad:
            nonfinite_branches.append(br)
            norms[br] = float('nan')
        else:
            l2 = math.sqrt(sq)
            norms[br] = l2
            if l2 <= min_grad_l2:
                dead_branches.append(br)

    return {
        'passed': (len(dead_branches) == 0 and len(nonfinite_branches) == 0),
        'branch_grad_l2': norms,
        'dead_branches': dead_branches,
        'nonfinite_branches': nonfinite_branches,
    }
