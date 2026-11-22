"""Frozen-backbone Score-Bank DRO Partial-AUC tail fine-tuning for PINPOINT nextgen models.

Implements:
1. 100% backbone freezing & exact bit-identity verification (only `event_head_7` trains).
2. Deterministic IEEE-FP32 (`allow_tf32=False`) penultimate-representation caching via
   forward pre-hook with exact `max|logits_cached - logits_full_model| <= 1e-7` verification.
3. Circular FIFO `ScoreBank` (default capacity 8,192) storing historical tau log-odds scores.
4. One-way CVaR / KL-DRO Partial-AUC surrogate loss targeting FPR <= 1% (Zhu et al., ICML 2022)
   jointly anchored with `marginal_tau_branch_ce` NLL and KL-to-pretrained-logits regularization.
5. Tail, calibration (ECE), and normalized pAUC@FPR evaluation utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from analysis.flavour.nextgen.protocols import marginal_four_from_seven
from analysis.flavour.v3.v3_multitask.losses import marginal_tau_branch_ce


def freeze_backbone_for_tail_finetuning(
    model: nn.Module,
    *,
    head_attr: str = 'event_head_7',
) -> nn.Module:
    """Freeze 100% of backbone parameters; keep only `head_attr` trainable."""
    if not hasattr(model, head_attr):
        raise AttributeError(f'Model {type(model).__name__} has no head attribute {head_attr!r}')
    model.eval()
    head = getattr(model, head_attr)
    head_prefix = f'{head_attr}.'
    n_trainable = 0
    for name, param in model.named_parameters():
        is_head = name.startswith(head_prefix) or name == head_attr
        param.requires_grad_(is_head)
        if is_head:
            n_trainable += param.numel()
    if n_trainable == 0:
        raise RuntimeError(f'No trainable parameters found under {head_attr!r}')
    return head


def snapshot_backbone_weights(
    model: nn.Module,
    *,
    head_attr: str = 'event_head_7',
) -> dict[str, torch.Tensor]:
    """Capture CPU bit-exact clones of all frozen backbone parameters & buffers."""
    head_prefix = f'{head_attr}.'
    snap: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if not (name.startswith(head_prefix) or name == head_attr):
            snap[f'param:{name}'] = param.detach().cpu().clone()
    for name, buf in model.named_buffers():
        snap[f'buf:{name}'] = buf.detach().cpu().clone()
    return snap


def verify_backbone_bit_identity(
    model: nn.Module,
    snapshot: dict[str, torch.Tensor],
    *,
    head_attr: str = 'event_head_7',
) -> dict[str, Any]:
    """Verify every frozen backbone parameter has requires_grad=False and bit-identical values."""
    head_prefix = f'{head_attr}.'
    checked_params = 0
    checked_buffers = 0
    mismatches: list[str] = []
    grad_violations: list[str] = []

    for name, param in model.named_parameters():
        if name.startswith(head_prefix) or name == head_attr:
            continue
        if param.requires_grad:
            grad_violations.append(name)
        key = f'param:{name}'
        if key not in snapshot or not torch.equal(param.detach().cpu(), snapshot[key]):
            mismatches.append(name)
        checked_params += 1

    for name, buf in model.named_buffers():
        key = f'buf:{name}'
        if key not in snapshot or not torch.equal(buf.detach().cpu(), snapshot[key]):
            mismatches.append(f'buffer:{name}')
        checked_buffers += 1

    if grad_violations or mismatches:
        raise AssertionError(
            f'Backbone freeze violation: grad_violations={grad_violations}, mismatches={mismatches}'
        )
    return {
        'passed': True,
        'checked_params': checked_params,
        'checked_buffers': checked_buffers,
        'mismatches': [],
        'grad_violations': [],
    }


def tau_log_odds_from_logits7(logits7: torch.Tensor) -> torch.Tensor:
    """Compute exact Neyman-Pearson log-odds log(P_tau / P_bg) directly from 7-class logits in FP32."""
    lg = logits7.float()
    log_tau = torch.logsumexp(lg[:, 3:6], dim=-1)
    log_bg = torch.logsumexp(lg[:, [0, 1, 2, 6]], dim=-1)
    return log_tau - log_bg


class ScoreBank(nn.Module):
    """Fixed-capacity circular FIFO queue storing historical tau log-odds & class indicators."""

    def __init__(self, capacity: int = 8192) -> None:
        super().__init__()
        if capacity <= 0:
            raise ValueError(f'ScoreBank capacity must be positive, got {capacity}')
        self.capacity = int(capacity)
        self.register_buffer('scores', torch.zeros(self.capacity, dtype=torch.float32))
        self.register_buffer('is_tau', torch.zeros(self.capacity, dtype=torch.bool))
        self.register_buffer('valid', torch.zeros(self.capacity, dtype=torch.bool))
        self.register_buffer('ptr', torch.zeros((), dtype=torch.long))
        self.register_buffer('count', torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def enqueue(self, scores_batch: torch.Tensor, is_tau_batch: torch.Tensor) -> None:
        """Append detached batch scores and signal flags into the circular FIFO buffer."""
        s = scores_batch.detach().float().reshape(-1).to(self.scores.device)
        t = is_tau_batch.detach().bool().reshape(-1).to(self.is_tau.device)
        n = int(s.numel())
        if n == 0:
            return
        if n >= self.capacity:
            s = s[-self.capacity:]
            t = t[-self.capacity:]
            self.scores.copy_(s)
            self.is_tau.copy_(t)
            self.valid.fill_(True)
            self.ptr.zero_()
            self.count.fill_(self.capacity)
            return

        p = int(self.ptr.item())
        first = min(n, self.capacity - p)
        self.scores[p:p + first].copy_(s[:first])
        self.is_tau[p:p + first].copy_(t[:first])
        self.valid[p:p + first].fill_(True)
        rem = n - first
        if rem > 0:
            self.scores[:rem].copy_(s[first:])
            self.is_tau[:rem].copy_(t[first:])
            self.valid[:rem].fill_(True)
        self.ptr.fill_((p + n) % self.capacity)
        self.count.fill_(min(self.capacity, int(self.count.item()) + n))

    def get_valid(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(neg_scores, pos_scores)` of all valid entries currently in the bank."""
        n_val = int(self.count.item())
        if n_val == 0:
            empty = self.scores[:0]
            return empty, empty
        v_mask = self.valid
        s_val = self.scores[v_mask]
        t_val = self.is_tau[v_mask]
        return s_val[~t_val], s_val[t_val]


@dataclass
class TailDROConfig:
    """Hyperparameters for anchored Score-Bank DRO Partial-AUC tail fine-tuning."""

    fpr_target: float = 0.01
    dro_mode: str = 'sopa_s_hybrid' # 'sopa_s_hybrid' | 'cvar' | 'kl_dro'
    surrogate: str = 'softplus'     # 'softplus' | 'squared_hinge' (for cvar / kl_dro)
    margin: float = 0.35
    kl_dro_temp: float = 0.05
    score_bound: float = 4.0
    cvar_weight: float = 0.25
    eta_lr: float = 5.0e-4
    lambda_dro: float = 0.40
    lambda_nll: float = 1.0
    lambda_kl: float = 0.25
    bank_capacity: int = 8192
    use_bank: bool = True
    lr: float = 5.0e-5
    weight_decay: float = 1.0e-4
    passes: int = 3
    batch_size: int = 128
    grad_clip: float = 2.0
    seed: int = 42


def bounded_tau_score(scores: torch.Tensor, bound: float = 4.0) -> torch.Tensor:
    """Soft tanh bounding preserving unit slope around FPR band while preventing tail blowup."""
    b = float(max(0.5, bound))
    return b * torch.tanh(scores.float() / b)


def _pairwise_surrogate(diff: torch.Tensor, *, surrogate: str, margin: float) -> torch.Tensor:
    """Smooth convex surrogate on `margin + s_neg - s_pos` computed in FP32."""
    u = float(margin) + diff
    if surrogate == 'softplus':
        return F.softplus(u)
    if surrogate == 'squared_hinge':
        return F.relu(u).square()
    raise ValueError(f'Unknown surrogate {surrogate!r}; expected softplus or squared_hinge')


def dro_pauc_surrogate_loss(
    scores: torch.Tensor,
    is_tau: torch.Tensor,
    bank: ScoreBank | None = None,
    *,
    fpr_target: float = 0.01,
    dro_mode: str = 'sopa_s_hybrid',
    surrogate: str = 'softplus',
    margin: float = 0.35,
    kl_dro_temp: float = 0.05,
    score_bound: float = 4.0,
    cvar_weight: float = 0.25,
    eta_param: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute one-way DRO / CVaR Partial-AUC surrogate loss (Zhu et al., ICML 2022) in FP32.

    Supports:
      - `'sopa_s_hybrid'`: bounded score `s_tilde = B * tanh(s / B)`, discoverability-weighted
        SOPA-s KL-DRO over `(live_neg U bank_neg)` + top-beta CVaR hinge (`0.75 KL-DRO + 0.25 CVaR`).
      - `'cvar'`: top-beta hardest negatives in `(live_neg U bank_neg)` + symmetric bank-pos push.
      - `'kl_dro'`: exponential-tilted KL-DRO over top-5*beta pool.
    """
    with torch.amp.autocast(device_type=scores.device.type, enabled=False):
        sc = bounded_tau_score(scores.float(), bound=score_bound) if dro_mode == 'sopa_s_hybrid' else scores.float()
        pos_mask = is_tau.bool()
        neg_mask = ~pos_mask
        s_pos_b = sc[pos_mask]
        s_neg_b = sc[neg_mask]

        if bank is not None:
            s_neg_bank, s_pos_bank = bank.get_valid()
            s_neg_bank = s_neg_bank.to(device=sc.device, dtype=torch.float32)
            s_pos_bank = s_pos_bank.to(device=sc.device, dtype=torch.float32)
            if dro_mode == 'sopa_s_hybrid':
                s_neg_bank = bounded_tau_score(s_neg_bank, bound=score_bound)
                s_pos_bank = bounded_tau_score(s_pos_bank, bound=score_bound)
        else:
            s_neg_bank = sc[:0]
            s_pos_bank = sc[:0]

        s_neg_all = torch.cat([s_neg_b, s_neg_bank], dim=0)
        n_neg_all = int(s_neg_all.numel())
        n_pos_b = int(s_pos_b.numel())
        n_neg_b = int(s_neg_b.numel())

        if n_neg_all == 0 or (n_pos_b == 0 and int(s_pos_bank.numel()) == 0):
            return sc.sum() * 0.0, {'n_neg_pool': float(n_neg_all), 'q_tail': 0.0, 'n_tail_neg': 0.0}

        beta = float(max(1e-4, min(1.0, fpr_target)))
        k_tail = max(1, int(math.ceil(beta * n_neg_all)))
        top_vals, top_idx = torch.topk(s_neg_all.detach(), k=k_tail, largest=True, sorted=True)
        q_tail = float(top_vals[-1].item())

        if dro_mode == 'cvar_balanced':
            # Near-boundary balanced CVaR-DRO: focus positive and negative gradients symmetrically
            # on the top-1%..3% tail band so global 4-class priors & NLL stay tightly anchored.
            k_band = min(n_neg_all, max(k_tail, int(math.ceil(min(1.0, 3.0 * beta) * n_neg_all))))
            band_vals, band_idx = torch.topk(s_neg_all.detach(), k=k_band, largest=True, sorted=True)
            q_band = float(band_vals[-1].item())

            l_pos = sc.sum() * 0.0
            if n_pos_b > 0:
                with torch.no_grad():
                    g_pos = 0.10 + 0.90 * torch.sigmoid((s_pos_b.detach() - (q_tail - 1.5)) / 0.7)
                    g_pos = g_pos / g_pos.sum().clamp_min(1e-8)
                s_neg_tail = s_neg_all[top_idx]
                diff_pos = s_neg_tail.unsqueeze(0) - s_pos_b.unsqueeze(1)
                per_pos = _pairwise_surrogate(diff_pos, surrogate=surrogate, margin=margin).mean(dim=1)
                l_pos = (g_pos * per_pos).sum()

            l_neg = sc.sum() * 0.0
            if n_neg_b > 0:
                pos_pool = torch.cat([s_pos_b.detach(), s_pos_bank.detach()], dim=0)
                if int(pos_pool.numel()) > 0:
                    # Select batch negatives in the top-3*beta tail band and focus against near-boundary positives
                    near_pos_mask = pos_pool >= (q_tail - 2.0)
                    pos_ref = pos_pool[near_pos_mask] if near_pos_mask.any() else pos_pool
                    if int(pos_ref.numel()) > 512:
                        pos_ref = pos_ref[:512]
                    tail_b_mask = s_neg_b.detach() >= q_band
                    if tail_b_mask.any():
                        s_neg_bt = s_neg_b[tail_b_mask]
                        diff_neg = s_neg_bt.unsqueeze(1) - pos_ref.unsqueeze(0)
                        per_neg = _pairwise_surrogate(diff_neg, surrogate=surrogate, margin=margin).mean(dim=1)
                        l_neg = per_neg.sum() / max(1.0, float(n_neg_b) * (3.0 * beta))

            loss_total = 0.5 * l_pos + 0.5 * l_neg
            return loss_total, {'n_neg_pool': float(n_neg_all), 'n_tail_neg': float(k_tail), 'q_tail': q_tail}

        if dro_mode == 'sopa_s_hybrid':
            l_kl_dro = sc.sum() * 0.0
            if n_pos_b > 0:
                with torch.no_grad():
                    g_raw = 0.10 + 0.90 * torch.sigmoid((s_pos_b.detach() - (q_tail - 1.5)) / 0.7)
                    g_w = g_raw / g_raw.sum().clamp_min(1e-8)
                phi = F.relu(s_neg_all.unsqueeze(0) - s_pos_b.unsqueeze(1) + float(margin)).square()
                t_kl = float(max(1e-3, kl_dro_temp))
                l_per_pos = t_kl * (torch.logsumexp(phi / t_kl, dim=1) - math.log(float(n_neg_all)))
                l_kl_dro = (g_w * l_per_pos).sum()

            eta = eta_param if eta_param is not None else sc.new_tensor(q_tail - 0.03)
            denom_cvar = max(beta, 1.0 / float(n_neg_all)) * float(n_neg_all)
            l_cvar = eta + F.relu(s_neg_all - eta).square().sum() / denom_cvar

            cw = float(max(0.0, min(1.0, cvar_weight)))
            loss_total = (1.0 - cw) * l_kl_dro + cw * l_cvar
            return loss_total, {'n_neg_pool': float(n_neg_all), 'n_tail_neg': float(k_tail), 'q_tail': q_tail}

        loss_total = sc.sum() * 0.0
        n_terms = 0

        if n_pos_b > 0:
            if dro_mode == 'cvar':
                s_neg_tail = s_neg_all[top_idx]
                diff_pos = s_neg_tail.unsqueeze(0) - s_pos_b.unsqueeze(1)
                loss_pos = _pairwise_surrogate(diff_pos, surrogate=surrogate, margin=margin).mean()
            elif dro_mode == 'kl_dro':
                k_pool = min(n_neg_all, max(k_tail, int(math.ceil(min(1.0, 5.0 * beta) * n_neg_all))))
                _, pool_idx = torch.topk(s_neg_all.detach(), k=k_pool, largest=True, sorted=False)
                s_neg_pool = s_neg_all[pool_idx]
                diff_pos = s_neg_pool.unsqueeze(0) - s_pos_b.unsqueeze(1)
                pair_loss = _pairwise_surrogate(diff_pos, surrogate=surrogate, margin=margin)
                gamma = float(max(1e-2, kl_dro_temp))
                w_kl = torch.softmax(pair_loss.detach() / gamma, dim=1)
                loss_pos = (w_kl * pair_loss).sum(dim=1).mean()
            else:
                raise ValueError(f'Unknown dro_mode {dro_mode!r}')
            loss_total = loss_total + loss_pos
            n_terms += 1

        if n_neg_b > 0 and int(s_pos_bank.numel()) > 0:
            tail_b_mask = s_neg_b.detach() >= q_tail
            if tail_b_mask.any():
                s_neg_bt = s_neg_b[tail_b_mask]
                diff_neg = s_neg_bt.unsqueeze(1) - s_pos_bank.unsqueeze(0)
                per_neg = _pairwise_surrogate(diff_neg, surrogate=surrogate, margin=margin).mean(dim=1)
                denom_b = max(1.0, float(n_neg_b) * beta)
                loss_neg = per_neg.sum() / denom_b
                loss_total = loss_total + 0.5 * loss_neg
                n_terms += 1

        if n_terms > 0:
            loss_total = loss_total / float(max(1, n_terms))

        stats = {
            'n_neg_pool': float(n_neg_all),
            'n_tail_neg': float(k_tail),
            'q_tail': q_tail,
        }
        return loss_total, stats


def compute_anchored_tail_loss(
    logits7: torch.Tensor,
    logits7_ref: torch.Tensor,
    label_4: torch.Tensor,
    *,
    label_7: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
    bank: ScoreBank | None = None,
    cfg: TailDROConfig | None = None,
    eta_param: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute joint NLL + KL-anchor + Score-Bank DRO Partial-AUC loss in FP32."""
    c = cfg or TailDROConfig()
    with torch.amp.autocast(device_type=logits7.device.type, enabled=False):
        lg = logits7.float()
        lg_ref = logits7_ref.float().detach()
        y4 = label_4.long()

        if c.lambda_nll > 0.0:
            if label_7 is not None:
                l_nll = marginal_tau_branch_ce(lg, label_7.long(), y4, sample_weight=sample_weight)
            else:
                p4 = marginal_four_from_seven(lg, is_probs=False).clamp_min(1e-8)
                per = F.nll_loss(torch.log(p4), y4, reduction='none')
                if sample_weight is not None:
                    w = sample_weight.float().clamp_min(0.0)
                    l_nll = (per * w).sum() / w.sum().clamp_min(1e-8)
                else:
                    l_nll = per.mean()
        else:
            l_nll = lg.sum() * 0.0

        if c.lambda_kl > 0.0:
            log_p_ref = F.log_softmax(lg_ref, dim=-1)
            p_ref = torch.exp(log_p_ref)
            log_p = F.log_softmax(lg, dim=-1)
            l_kl = (p_ref * (log_p_ref - log_p)).sum(dim=-1).mean()
        else:
            l_kl = lg.sum() * 0.0

        scores = tau_log_odds_from_logits7(lg)
        is_tau = y4 == 2
        if c.lambda_dro > 0.0:
            l_dro, dro_stats = dro_pauc_surrogate_loss(
                scores,
                is_tau,
                bank=bank if c.use_bank else None,
                fpr_target=c.fpr_target,
                dro_mode=c.dro_mode,
                surrogate=c.surrogate,
                margin=c.margin,
                kl_dro_temp=c.kl_dro_temp,
                score_bound=c.score_bound,
                cvar_weight=c.cvar_weight,
                eta_param=eta_param,
            )
        else:
            l_dro = lg.sum() * 0.0
            dro_stats = {'n_neg_pool': 0.0, 'n_tail_neg': 0.0, 'q_tail': 0.0}

        total = float(c.lambda_nll) * l_nll + float(c.lambda_kl) * l_kl + float(c.lambda_dro) * l_dro
        metrics = {
            'loss_total': float(total.detach().item()),
            'loss_nll': float(l_nll.detach().item()),
            'loss_kl': float(l_kl.detach().item()),
            'loss_dro': float(l_dro.detach().item()),
            **dro_stats,
        }
        return total, metrics


def set_ieee_fp32_mode(enable: bool = True) -> tuple[bool, bool] | None:
    """Force deterministic IEEE-FP32 (disable TF32) and return previous `(matmul, cudnn)` flags."""
    if not torch.cuda.is_available():
        return None
    prev = (bool(torch.backends.cuda.matmul.allow_tf32), bool(torch.backends.cudnn.allow_tf32))
    torch.backends.cuda.matmul.allow_tf32 = not enable
    torch.backends.cudnn.allow_tf32 = not enable
    return prev


def restore_tf32_mode(prev: tuple[bool, bool] | None) -> None:
    if prev is not None and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = prev[0]
        torch.backends.cudnn.allow_tf32 = prev[1]


@torch.no_grad()
def extract_penultimate_cache(
    model: nn.Module,
    batches: Iterable[dict[str, Any]],
    device: torch.device,
    *,
    head_attr: str = 'event_head_7',
    verify_atol: float = 1.0e-7,
) -> dict[str, Any]:
    """Run one deterministic IEEE-FP32 pass to cache penultimate latents & verify exact equivalence."""
    if not hasattr(model, head_attr):
        raise AttributeError(f'Model {type(model).__name__} missing {head_attr!r}')
    head = getattr(model, head_attr)
    was_training = model.training
    model.eval()
    prev_tf32 = set_ieee_fp32_mode(True)

    captured: dict[str, torch.Tensor] = {}

    def _pre_hook(_mod: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        captured['penultimate'] = inputs[0].detach()

    handle = head.register_forward_pre_hook(_pre_hook)

    penult_list: list[torch.Tensor] = []
    logits7_list: list[torch.Tensor] = []
    probs4_list: list[torch.Tensor] = []
    charm_list: list[torch.Tensor] = []
    pv_list: list[torch.Tensor] = []
    scale_list: list[torch.Tensor] = []
    opt_tensors: dict[str, list[torch.Tensor]] = {
        k: [] for k in ('label', 'label_7', 'tau_branch_valid', 'sample_weight', 'obs_domain_id')
    }
    max_verify_diff = 0.0

    try:
        for raw_batch in batches:
            batch = {
                k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in raw_batch.items()
            }
            captured.clear()
            with torch.inference_mode(), torch.amp.autocast(device_type=device.type, enabled=False):
                out = model.forward_NextGen(batch)
                if 'penultimate' not in captured:
                    raise RuntimeError(f'Forward pre-hook on {head_attr!r} did not fire')
                pen = captured['penultimate'].float()
                logits_check = head(pen).float()
                diff = float((logits_check - out.logits7.float()).abs().max().item())
                if diff > max_verify_diff:
                    max_verify_diff = diff
                if diff > verify_atol:
                    raise AssertionError(
                        f'Penultimate cache equivalence check failed: {diff:.3e} > {verify_atol:.3e}'
                    )

            penult_list.append(pen.cpu())
            logits7_list.append(out.logits7.detach().float().cpu())
            probs4_list.append(out.probs4.detach().float().cpu())
            charm_list.append(out.charm_logit.detach().float().cpu())
            pv_list.append(out.v_pv_mm.detach().float().cpu())
            scale_list.append(out.scale_pv_mm.detach().float().cpu())

            for ok in opt_tensors:
                if ok in batch and torch.is_tensor(batch[ok]):
                    opt_tensors[ok].append(batch[ok].detach().cpu())
    finally:
        handle.remove()
        restore_tf32_mode(prev_tf32)
        model.train(was_training)

    cache: dict[str, Any] = {
        'penultimate': torch.cat(penult_list, dim=0),
        'logits7_ref': torch.cat(logits7_list, dim=0),
        'probs4_ref': torch.cat(probs4_list, dim=0),
        'charm_logit_ref': torch.cat(charm_list, dim=0),
        'v_pv_mm': torch.cat(pv_list, dim=0),
        'scale_pv_mm': torch.cat(scale_list, dim=0),
        'max_verify_diff': float(max_verify_diff),
    }
    for ok, lst in opt_tensors.items():
        if lst and len(lst) == len(penult_list):
            cache[ok] = torch.cat(lst, dim=0)
    return cache


def finetune_head_on_cached_latents(
    head: nn.Module,
    train_cache: dict[str, torch.Tensor],
    cfg: TailDROConfig,
    device: torch.device,
    *,
    eval_head_in_eval_mode: bool = True,
) -> tuple[nn.Module, list[dict[str, float]]]:
    """Fine-tune only `head` for `cfg.passes` passes over cached penultimate latents on GPU."""
    torch.manual_seed(cfg.seed)
    prev_tf32 = set_ieee_fp32_mode(True)
    head = head.to(device)

    pen = train_cache['penultimate'].to(device=device, dtype=torch.float32)
    lg_ref = train_cache['logits7_ref'].to(device=device, dtype=torch.float32)
    y4 = train_cache['label'].to(device=device, dtype=torch.long)
    y7 = train_cache['label_7'].to(device=device, dtype=torch.long) if 'label_7' in train_cache else None
    sw = train_cache['sample_weight'].to(device=device, dtype=torch.float32) if 'sample_weight' in train_cache else None

    n_ev = int(pen.shape[0])
    bank = ScoreBank(capacity=cfg.bank_capacity).to(device) if cfg.use_bank else None

    # Pre-populate ScoreBank with one deterministic pass of pretrained scores so step 1
    # already sees an accurate top-1% background quantile boundary (eliminates cold-start noise).
    eta_param: nn.Parameter | None = None
    q_init = 0.0
    if bank is not None:
        head.eval()
        with torch.no_grad():
            init_n = min(n_ev, cfg.bank_capacity)
            g_init = torch.Generator(device='cpu')
            g_init.manual_seed(cfg.seed + 13)
            init_idx = torch.randperm(n_ev, generator=g_init)[:init_n].to(device)
            for st in range(0, init_n, 512):
                sl = init_idx[st:st + 512]
                s0 = tau_log_odds_from_logits7(head(pen[sl]))
                bank.enqueue(s0, y4[sl] == 2)
            neg_b0, _ = bank.get_valid()
            if int(neg_b0.numel()) > 0:
                s_neg0 = bounded_tau_score(neg_b0, cfg.score_bound) if cfg.dro_mode == 'sopa_s_hybrid' else neg_b0
                q_init = float(torch.quantile(s_neg0, 1.0 - cfg.fpr_target).item())

    param_groups: list[dict[str, Any]] = [
        {'params': [p for p in head.parameters() if p.requires_grad], 'lr': cfg.lr, 'weight_decay': cfg.weight_decay}
    ]
    if cfg.dro_mode == 'sopa_s_hybrid':
        eta_param = nn.Parameter(torch.tensor(q_init - 0.03, dtype=torch.float32, device=device))
        param_groups.append({'params': [eta_param], 'lr': cfg.eta_lr, 'weight_decay': 0.0})

    opt = torch.optim.AdamW(param_groups, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.98))
    steps_per_pass = max(1, math.ceil(n_ev / cfg.batch_size))
    total_steps = max(1, cfg.passes * steps_per_pass)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=cfg.lr * 0.1)

    history: list[dict[str, float]] = []
    try:
        for p_idx in range(1, cfg.passes + 1):
            head.train()
            g_cpu = torch.Generator(device='cpu')
            g_cpu.manual_seed(cfg.seed + p_idx * 101)
            perm = torch.randperm(n_ev, generator=g_cpu).to(device)

            sums = {'loss_total': 0.0, 'loss_nll': 0.0, 'loss_kl': 0.0, 'loss_dro': 0.0}
            seen = 0
            for st in range(0, n_ev, cfg.batch_size):
                idx = perm[st:st + cfg.batch_size]
                opt.zero_grad(set_to_none=True)
                lg7 = head(pen[idx])
                loss, m_step = compute_anchored_tail_loss(
                    lg7,
                    lg_ref[idx],
                    y4[idx],
                    label_7=(y7[idx] if y7 is not None else None),
                    sample_weight=(sw[idx] if sw is not None else None),
                    bank=bank,
                    cfg=cfg,
                    eta_param=eta_param,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Non-finite tail DRO loss at pass={p_idx}: {m_step}')
                loss.backward()
                for p in head.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        raise FloatingPointError('Non-finite gradient in fine-tuned head')
                torch.nn.utils.clip_grad_norm_(head.parameters(), cfg.grad_clip)
                opt.step()
                scheduler.step()

                if bank is not None:
                    with torch.no_grad():
                        if eval_head_in_eval_mode:
                            head.eval()
                            s_new = tau_log_odds_from_logits7(head(pen[idx]))
                            head.train()
                        else:
                            s_new = tau_log_odds_from_logits7(lg7.detach())
                        bank.enqueue(s_new, y4[idx] == 2)

                bs = int(idx.numel())
                for k in sums:
                    sums[k] += m_step[k] * bs
                seen += bs

            row = {'pass': float(p_idx)}
            for k, v in sums.items():
                row[k] = v / max(1, seen)
            history.append(row)
    finally:
        head.eval()
        restore_tf32_mode(prev_tf32)
    return head, history


@torch.no_grad()
def predict_from_cached_latents(
    head: nn.Module,
    penultimate: torch.Tensor,
    device: torch.device,
    *,
    batch_size: int = 1024,
) -> dict[str, np.ndarray]:
    """Evaluate `head` on cached penultimate latents in deterministic IEEE-FP32."""
    prev_tf32 = set_ieee_fp32_mode(True)
    was_training = head.training
    head.eval()
    lg7_l: list[torch.Tensor] = []
    p4_l: list[torch.Tensor] = []
    sc_l: list[torch.Tensor] = []
    try:
        n = int(penultimate.shape[0])
        for st in range(0, n, batch_size):
            x = penultimate[st:st + batch_size].to(device=device, dtype=torch.float32)
            with torch.amp.autocast(device_type=device.type, enabled=False):
                lg7 = head(x).float()
                p4 = marginal_four_from_seven(lg7, is_probs=False)
                sc = tau_log_odds_from_logits7(lg7)
            lg7_l.append(lg7.cpu())
            p4_l.append(p4.cpu())
            sc_l.append(sc.cpu())
    finally:
        head.train(was_training)
        restore_tf32_mode(prev_tf32)
    return {
        'logits7': torch.cat(lg7_l, dim=0).numpy(),
        'probs4': torch.cat(p4_l, dim=0).numpy(),
        'tau_score': torch.cat(sc_l, dim=0).numpy(),
    }


def compute_multiclass_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Standard 15-bin top-label Expected Calibration Error (ECE)."""
    p = np.asarray(probs, dtype=np.float64)
    p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    y = np.asarray(labels, dtype=np.int64)
    conf = p.max(axis=1)
    acc = (p.argmax(axis=1) == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = max(1, len(y))
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf >= lo) & (conf <= hi if i == n_bins - 1 else conf < hi)
        if m.any():
            ece += (float(m.sum()) / n) * abs(float(acc[m].mean()) - float(conf[m].mean()))
    return float(ece)


def compute_tau_binary_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """15-bin binary Expected Calibration Error on `P_tau` vs `label == 2`."""
    p = np.asarray(probs, dtype=np.float64)
    p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    p_tau = p[:, 2]
    y_tau = (np.asarray(labels, dtype=np.int64) == 2).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = max(1, len(y_tau))
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p_tau >= lo) & (p_tau <= hi if i == n_bins - 1 else p_tau < hi)
        if m.any():
            ece += (float(m.sum()) / n) * abs(float(y_tau[m].mean()) - float(p_tau[m].mean()))
    return float(ece)


def compute_normalized_pauc(scores: np.ndarray, labels: np.ndarray, fpr_max: float = 0.01) -> float:
    """Compute exact normalized one-way Partial AUC on FPR in [0, fpr_max] using trapezoidal ROC."""
    s = np.asarray(scores, dtype=np.float64)
    y = (np.asarray(labels, dtype=np.int64) == 2)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0
    order = np.argsort(-s, kind='mergesort')
    y_ord = y[order].astype(np.float64)
    tp = np.cumsum(y_ord)
    fp = np.cumsum(1.0 - y_ord)
    tpr = np.r_[0.0, tp / n_pos]
    fpr = np.r_[0.0, fp / n_neg]

    fmax = float(max(1e-6, min(1.0, fpr_max)))
    mask = fpr <= fmax
    fpr_sub = fpr[mask]
    tpr_sub = tpr[mask]
    if fpr_sub[-1] < fmax:
        idx_hi = int(np.searchsorted(fpr, fmax, side='right'))
        if idx_hi < len(fpr):
            f0, f1 = fpr[idx_hi - 1], fpr[idx_hi]
            t0, t1 = tpr[idx_hi - 1], tpr[idx_hi]
            t_interp = t0 + (t1 - t0) * ((fmax - f0) / max(1e-12, f1 - f0))
            fpr_sub = np.r_[fpr_sub, fmax]
            tpr_sub = np.r_[tpr_sub, t_interp]
    area = float(np.trapezoid(tpr_sub, fpr_sub)) if hasattr(np, 'trapezoid') else float(np.trapz(tpr_sub, fpr_sub))
    return float(area / fmax)
