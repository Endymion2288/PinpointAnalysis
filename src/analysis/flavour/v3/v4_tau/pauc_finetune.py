"""One-way SOPA-s KL-DRO + boundary-gated CVaR partial-AUC@1% FPR tail loss (Zhu et al., ICML 2022)."""

from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


class DROPartialAUCLoss(nn.Module):
    """Per-positive KL-DRO (SOPA-s) on bounded P(tau) plus top-beta FP CVaR on bounded log-odds."""

    def __init__(
        self,
        beta: float = 0.01,
        margin: float = 0.40,
        margin_p: float = 0.15,
        kl_temp: float = 0.04,
        init_tau_q: float = 0.5,
    ) -> None:
        super().__init__()
        if not (0.0 < beta <= 1.0):
            raise ValueError(f'beta must lie in (0, 1], got {beta}')
        if kl_temp <= 0.0:
            raise ValueError(f'kl_temp must be positive, got {kl_temp}')
        self.beta = float(beta)
        self.margin = float(margin)
        self.margin_p = float(margin_p)
        self.kl_temp = float(kl_temp)
        self.tau_q = nn.Parameter(torch.tensor(float(init_tau_q), dtype=torch.float32))
        self.register_buffer('tau_q_initialized', torch.tensor(False, dtype=torch.bool))
        self.register_buffer('q_beta_ema', torch.tensor(0.0, dtype=torch.float32))

    def forward(
        self,
        probs4: torch.Tensor,
        labels: torch.Tensor,
        obs_domain_id: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        with torch.amp.autocast(device_type=probs4.device.type, enabled=False):
            valid = obs_domain_id != 0
            pos_mask = valid & (labels == 2)
            neg_mask = valid & (labels != 2)
            if not pos_mask.any() or not neg_mask.any():
                zero = 0.0 * probs4[:, 2].float().sum() + 0.0 * self.tau_q
                return zero, {
                    'pauc_loss': 0.0,
                    'tau_q': float(self.tau_q.detach()),
                    'active_frac': 0.0,
                }

            p_tau = probs4[:, 2].float().clamp(1e-6, 1.0 - 1e-6)
            s = torch.log(p_tau / (1.0 - p_tau))
            s_tilde = 3.0 * torch.tanh(s / 3.0)

            p_pos, p_neg = p_tau[pos_mask], p_tau[neg_mask]
            s_pos, s_neg = s[pos_mask], s[neg_mask]
            n_neg = int(p_neg.numel())

            q_batch = torch.quantile(s_neg.detach(), max(0.0, 1.0 - self.beta))
            if self.training:
                if not bool(self.tau_q_initialized.item()):
                    self.q_beta_ema.copy_(q_batch)
                else:
                    self.q_beta_ema.mul_(0.95).add_(q_batch, alpha=0.05)
            q_ref = self.q_beta_ema if (bool(self.tau_q_initialized.item()) or self.training) else q_batch

            # Boundary-discoverability gate focuses updates near the 1% FPR operating threshold
            g_pos = 0.05 + 0.95 * torch.sigmoid((s_pos.detach() - (q_ref - 1.8)) / 0.8)
            if sample_weight is not None:
                sw = sample_weight.float().clamp_min(0.0)
                g_pos = g_pos * sw[pos_mask].clamp_min(1e-4)
                w_neg = sw[neg_mask] / sw[neg_mask].sum().clamp_min(1e-8)
            else:
                w_neg = None
            g_norm = g_pos / g_pos.sum().clamp_min(1e-8)

            # Term 1: SOPA-s per-positive KL-DRO over bounded P(tau) in [0, 1]
            m_p = F.relu(p_neg[:, None] - p_pos[None, :] + self.margin_p).square()  # (N-, N+)
            if w_neg is None:
                l_sopa_i = self.kl_temp * (torch.logsumexp(m_p / self.kl_temp, dim=0) - math.log(max(1, n_neg)))
            else:
                l_sopa_i = self.kl_temp * torch.logsumexp(
                    m_p / self.kl_temp + torch.log(w_neg.clamp_min(1e-8))[:, None],
                    dim=0,
                )
            loss_sopa = (g_norm * l_sopa_i).sum()

            # Term 2: Top-beta negative tail CVaR against boundary-weighted bounded log-odds
            s_pos_bar = (g_norm * s_tilde[pos_mask]).sum()
            delta_neg = s_tilde[neg_mask] - s_pos_bar + self.margin  # (N-,)
            beta_eff = max(self.beta, 1.5 / max(1, n_neg))
            if self.training and not bool(self.tau_q_initialized.item()):
                q_cvar = torch.quantile(delta_neg.detach(), max(0.0, 1.0 - beta_eff)) - 0.03
                self.tau_q.data.copy_(q_cvar.clamp(-6.0, 6.0))
                self.tau_q_initialized.fill_(True)

            excess = F.relu(delta_neg - self.tau_q)
            hinge_neg = (w_neg * excess.square()).sum() if w_neg is not None else excess.square().mean()
            loss_cvar = self.tau_q + (1.0 / beta_eff) * hinge_neg

            loss = 0.70 * loss_sopa + 0.30 * loss_cvar
            return loss, {
                'pauc_loss': float(loss.detach()),
                'tau_q': float(self.tau_q.detach()),
                'active_frac': float((excess > 0.0).float().mean().detach()),
            }
