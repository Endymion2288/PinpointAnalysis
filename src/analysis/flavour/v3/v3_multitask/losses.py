"""Event-level 4-head homoscedastic uncertainty loss and 7-way marginalization (Phase V3)."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from analysis.flavour.v3.v2_pinpointformer.vertex_anchor import heteroscedastic_vertex_loss


LABEL_7_NAMES = (
    'e_CC',
    'mu_minus_CC',
    'mu_plus_CC',
    'tau_e',
    'tau_mu',
    'tau_had',
    'NC',
)


def marginal_four_from_seven(x: torch.Tensor, is_probs: bool = False) -> torch.Tensor:
    """Collapse 7 fine process probabilities/logits to exact 4-class probabilities."""
    p7 = x.float() if is_probs else torch.softmax(x.float(), dim=-1)
    return torch.stack([
        p7[:, 0],
        p7[:, 1] + p7[:, 2],
        p7[:, 3] + p7[:, 4] + p7[:, 5],
        p7[:, 6],
    ], dim=-1)


def marginal_tau_branch_ce(
    logits7: torch.Tensor,
    label_7: torch.Tensor,
    label_4: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    denom_override: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """7-class cross-entropy with exact -log(p3+p4+p5) marginal loss for unknown tau decay rows."""
    with torch.autocast(device_type=logits7.device.type, enabled=False):
        lg = logits7.float()
        known = label_7 >= 0
        per = torch.zeros(len(lg), dtype=torch.float32, device=lg.device)
        if known.any():
            per[known] = F.cross_entropy(lg[known], label_7[known], reduction='none')
        unk_tau = (~known) & (label_4 == 2)
        if unk_tau.any():
            log_z = torch.logsumexp(lg[unk_tau], dim=-1)
            log_tau = torch.logsumexp(lg[unk_tau, 3:6], dim=-1)
            per[unk_tau] = log_z - log_tau
        valid = known | unk_tau
        if not valid.any():
            return lg.sum() * 0.0
        if sample_weight is not None:
            w = sample_weight.float().clamp_min(0.0) * valid.float()
        else:
            w = valid.float()
        if denom_override is not None:
            denom = torch.as_tensor(denom_override, dtype=torch.float32, device=lg.device).clamp_min(1e-8)
        else:
            denom = w.sum().clamp_min(1e-8)
        return (per * w).sum() / denom


def focal_charm_bce(
    charm_logit: torch.Tensor,
    charm_flag: torch.Tensor,
    charm_valid: torch.Tensor,
    gamma: float = 1.5,
    pos_weight: float = 3.0,
    sample_weight: torch.Tensor | None = None,
    denom_override: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Imbalance-aware focal BCE loss for concurrent Charm production (approx 7.4% pos)."""
    valid = charm_valid > 0
    if not valid.any():
        return charm_logit.sum() * 0.0
    with torch.autocast(device_type=charm_logit.device.type, enabled=False):
        lg = charm_logit[valid].float().view(-1)
        y = charm_flag[valid].float().view(-1)
        p = torch.sigmoid(lg)
        pt = torch.where(y > 0.5, p, 1.0 - p).clamp(min=1e-6, max=1.0 - 1e-6)
        bce = F.binary_cross_entropy_with_logits(lg, y, reduction='none')
        focal = ((1.0 - pt) ** gamma) * bce
        w = torch.where(y > 0.5, float(pos_weight), 1.0)
        if sample_weight is not None:
            w = w * sample_weight[valid].float().view(-1).clamp_min(0.0)
        if denom_override is not None:
            denom = torch.as_tensor(denom_override, dtype=torch.float32, device=lg.device).clamp_min(1e-8)
        else:
            denom = w.sum().clamp_min(1e-8)
        return (focal * w).sum() / denom


def compute_window_task_denominators(
    micro_batches: list[dict],
    *,
    pos_weight: float = 3.0,
    weight_all_tasks: bool = False,
) -> dict[str, float]:
    """Compute exact full-window per-task denominators across microbatches (Stage 324-02).

    Guarantees sum_j (S_j / D_full) == (sum_j S_j) / D_full for masked & weighted losses.
    """
    d_proc = 0.0
    d_charm = 0.0
    d_vtx = 0.0
    d_kin = 0.0
    for mb in micro_batches:
        sw = mb.get('sample_weight')
        sw_f = sw.float().clamp_min(0.0) if sw is not None else None
        sw_aux = sw_f if (weight_all_tasks and sw_f is not None) else None

        valid_p = (mb['label_7'] >= 0) | ((mb['label_7'] < 0) & (mb['label'] == 2))
        if valid_p.any():
            d_proc += float((sw_f[valid_p].sum() if sw_f is not None else valid_p.float().sum()).item())

        valid_c = mb['charm_valid'] > 0
        if valid_c.any():
            yc = mb['charm_flag'][valid_c].float().view(-1)
            wc = torch.where(yc > 0.5, float(pos_weight), 1.0)
            if sw_aux is not None:
                wc = wc * sw_aux[valid_c].view(-1)
            d_charm += float(wc.sum().item())

        valid_v = mb['vertex_valid'] > 0
        if valid_v.any():
            d_vtx += float((sw_aux[valid_v].sum() if sw_aux is not None else valid_v.float().sum()).item())

        valid_k = mb['kin_valid'] > 0
        if valid_k.any():
            d_kin += float((sw_aux[valid_k].sum() if sw_aux is not None else valid_k.float().sum()).item())

    return {
        'proc': max(d_proc, 1e-8),
        'charm': max(d_charm, 1e-8),
        'vtx': max(d_vtx, 1e-8),
        'kin': max(d_kin, 1e-8),
    }


class HomoscedasticEventLoss(nn.Module):
    """4-head uncertainty-weighted event loss over 7-way process, Charm, 3D vertex, and kinematics."""

    def __init__(self, log_var_min: float = -4.0, log_var_max: float = 4.0) -> None:
        super().__init__()
        self.log_var_min = float(log_var_min)
        self.log_var_max = float(log_var_max)
        # Order: [proc, charm, vtx, kin]
        self.log_vars = nn.Parameter(torch.zeros(4, dtype=torch.float32))

    def forward(
        self,
        logits7: torch.Tensor,
        charm_logit: torch.Tensor,
        v_hat: torch.Tensor,
        log_var_v: torch.Tensor,
        kin_pred: torch.Tensor,
        batch: dict,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        l_proc = marginal_tau_branch_ce(
            logits7,
            batch['label_7'],
            batch['label'],
            sample_weight=batch.get('sample_weight'),
        )
        l_charm = focal_charm_bce(charm_logit, batch['charm_flag'], batch['charm_valid']) + 0.5
        l_vtx = heteroscedastic_vertex_loss(
            v_hat,
            log_var_v,
            batch['vertex_xyz_mm'],
            batch['vertex_valid'],
            legacy_biased=True,
        )

        kin_valid = batch['kin_valid'] > 0
        if kin_valid.any():
            with torch.autocast(device_type=kin_pred.device.type, enabled=False):
                l_kin = (
                    F.smooth_l1_loss(
                        kin_pred[kin_valid].float(),
                        batch['vis_kinematics'][kin_valid].float(),
                        reduction='mean',
                    )
                    + 0.5
                )
        else:
            l_kin = kin_pred.sum() * 0.0

        with torch.autocast(device_type=logits7.device.type, enabled=False):
            u = torch.clamp(self.log_vars.float(), min=self.log_var_min, max=self.log_var_max)
            losses = torch.stack([l_proc.float(), l_charm.float(), l_vtx.float(), l_kin.float()])
            total = (0.5 * torch.exp(-u) * losses + 0.5 * u).sum()

        parts = {
            'l_proc': l_proc.detach(),
            'l_charm': (l_charm - 0.5).detach(),
            'l_vtx': l_vtx.detach(),
            'l_kin': (l_kin - 0.5).detach(),
            'l_event': total.detach(),
            'u_proc': u[0].detach(),
            'u_charm': u[1].detach(),
            'u_vtx': u[2].detach(),
            'u_kin': u[3].detach(),
        }
        return total, parts


class FixedWeightEventLoss(nn.Module):
    """Unbiased fixed-weight 4-head event loss with exact microbatch denominator support (324-02/03)."""

    def __init__(
        self,
        w_proc: float = 1.0,
        w_charm: float = 0.25,
        w_vtx: float = 0.20,
        w_kin: float = 0.15,
        weight_all_tasks: bool = False,
    ) -> None:
        super().__init__()
        self.w_proc = float(w_proc)
        self.w_charm = float(w_charm)
        self.w_vtx = float(w_vtx)
        self.w_kin = float(w_kin)
        self.weight_all_tasks = bool(weight_all_tasks)

    def forward(
        self,
        logits7: torch.Tensor,
        charm_logit: torch.Tensor,
        v_hat: torch.Tensor,
        log_var_v: torch.Tensor,
        kin_pred: torch.Tensor,
        batch: dict,
        fixed_denoms: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        sw = batch.get('sample_weight')
        sw_aux = sw if self.weight_all_tasks else None
        fd = fixed_denoms or {}

        l_proc = marginal_tau_branch_ce(
            logits7,
            batch['label_7'],
            batch['label'],
            sample_weight=sw,
            denom_override=fd.get('proc'),
        )
        l_charm = focal_charm_bce(
            charm_logit,
            batch['charm_flag'],
            batch['charm_valid'],
            sample_weight=sw_aux,
            denom_override=fd.get('charm'),
        )
        l_vtx = heteroscedastic_vertex_loss(
            v_hat,
            log_var_v,
            batch['vertex_xyz_mm'],
            batch['vertex_valid'],
            legacy_biased=False,
            sample_weight=sw_aux,
            denom_override=fd.get('vtx'),
        )

        kin_valid = batch['kin_valid'] > 0
        if kin_valid.any():
            with torch.autocast(device_type=kin_pred.device.type, enabled=False):
                kin_ev = F.smooth_l1_loss(
                    kin_pred[kin_valid].float(),
                    batch['vis_kinematics'][kin_valid].float(),
                    reduction='none',
                ).mean(dim=-1)
                if sw_aux is not None:
                    wk = sw_aux[kin_valid].float().clamp_min(0.0)
                else:
                    wk = torch.ones_like(kin_ev)
                if 'kin' in fd:
                    dk = torch.as_tensor(fd['kin'], dtype=torch.float32, device=kin_ev.device).clamp_min(1e-8)
                else:
                    dk = wk.sum().clamp_min(1e-8)
                l_kin = (kin_ev * wk).sum() / dk
        else:
            l_kin = kin_pred.sum() * 0.0

        with torch.autocast(device_type=logits7.device.type, enabled=False):
            total = (
                self.w_proc * l_proc.float()
                + self.w_charm * l_charm.float()
                + self.w_vtx * l_vtx.float()
                + self.w_kin * l_kin.float()
            )

        parts = {
            'l_proc': l_proc.detach(),
            'l_charm': l_charm.detach(),
            'l_vtx': l_vtx.detach(),
            'l_kin': l_kin.detach(),
            'l_event': total.detach(),
        }
        return total, parts
