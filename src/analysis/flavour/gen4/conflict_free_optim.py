"""Conflict-free multimodal & multitask optimization for Pinpoint-Gen4.

Implements the Pillar-III optimization suite matching `[gen4-contract]`:
  1. `DomainCGGM`: Domain-active Conditional Gradient Modulation (Guo et al., NeurIPS 2024)
     gated strictly by physical modality presence (`a_si`, `a_cal`, `a_spec`) so empty-silicon
     Fortune_ScintOnly events never amplify dead silicon gradients, plus fusion-layer
     direction alignment with norm-normalized weak-modality probes.
  2. `MMPareto`: Norm-preserving 4x4 convex quadratic programming gradient projection
     (Wei & Hu, ICML 2024) over primary classification + auxiliary vertex / origin / pflow
     gradients with primary floor `gamma_0 >= 0.45` and strict norm preservation.
  3. `MCRSynergyRegularizer`: Cross-modal permutation synergy mutual-information regularizer
     (Kontras et al., NeurIPS 2025) over silicon-active batches preventing calorimeter dominance
     and unimodal overconfidence shortcuts.
  4. `ConflictFreeStepEngine`: Unified trainer helper integrating `DomainCGGM`, `MMPareto`, and
     `MCR` cleanly over the 5 contracted top-level module groups (`si_vtx`, `cal_pflow`,
     `spec_ipt`, `fusion`, `head`).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from analysis.flavour.nextgen.protocols import marginal_four_from_seven


GEN4_MODALITY_GROUPS: tuple[str, ...] = ('si_vtx', 'cal_pflow', 'spec_ipt')
GEN4_SHARED_GROUPS: tuple[str, ...] = ('si_vtx', 'cal_pflow', 'spec_ipt', 'fusion')
GEN4_ALL_GROUPS: tuple[str, ...] = ('si_vtx', 'cal_pflow', 'spec_ipt', 'fusion', 'head')


def collect_gen4_param_groups(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    """Collect trainable parameters partitioned by the 5 contracted Gen4 top-level groups."""
    groups: dict[str, list[nn.Parameter]] = {k: [] for k in GEN4_ALL_GROUPS}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        prefix = name.split('.', 1)[0]
        if prefix in groups:
            groups[prefix].append(p)
        else:
            groups.setdefault(prefix, []).append(p)
    return groups


def compute_modality_presence_masks(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Derive strictly observable modality presence masks `(a_si, a_cal, a_spec)` in bool (B,)."""
    c_obs = batch['c_obs']
    a_si = c_obs[:, 0] > 0.0
    a_cal = c_obs[:, 1] > 0.0
    a_spec = c_obs[:, 2] > 0.0
    if 'muon_kalman_mask' in batch and batch['muon_kalman_mask'] is not None:
        a_spec = a_spec | batch['muon_kalman_mask'].bool().any(dim=-1)
    return {'si_vtx': a_si, 'cal_pflow': a_cal, 'spec_ipt': a_spec}


def solve_mmpareto_weights(
    gram: torch.Tensor,
    *,
    gamma0_min: float = 0.45,
    active_mask: torch.Tensor | None = None,
    n_iters: int = 40,
) -> torch.Tensor:
    """Solve KxK convex QP: min_{gamma in Delta^{K-1}, gamma_0 >= gamma0_min} gamma^T G gamma.

    Uses projected Nesterov/FISTA iterations onto the constrained simplex
    {gamma >= 0, sum(gamma) == 1, gamma_0 >= gamma0_min, gamma_k = 0 if inactive}.
    """
    k_dim = int(gram.shape[0])
    if k_dim == 0:
        return gram.new_empty((0,))
    if k_dim == 1:
        return gram.new_ones((1,))

    active = (
        torch.ones(k_dim, dtype=torch.bool, device=gram.device)
        if active_mask is None
        else active_mask.bool().clone()
    )
    active[0] = True  # Primary classification objective is always active
    if int(active.sum().item()) <= 1:
        out = gram.new_zeros((k_dim,))
        out[0] = 1.0
        return out

    g_mat = 0.5 * (gram.float() + gram.float().T)
    # Spectral upper bound for Lipschitz step size
    lip = float(torch.linalg.eigvalsh(g_mat).max().clamp_min(1e-4).item())
    step = 0.5 / lip
    g0_floor = float(min(max(gamma0_min, 0.0), 1.0))

    def _project_constrained_simplex(v: torch.Tensor) -> torch.Tensor:
        w = v.clone()
        w[~active] = -1e9
        act_idx = torch.nonzero(active, as_tuple=True)[0]
        if len(act_idx) == 1:
            res = torch.zeros_like(v)
            res[act_idx[0]] = 1.0
            return res
        # Enforce gamma_0 >= g0_floor and project remaining (1 - gamma_0) onto simplex over act_idx[1:]
        rem_idx = act_idx[1:]
        # Coordinate-exact piecewise projection: test candidate gamma_0 = max(g0_floor, ...)
        # Since K <= 8, 1D bisection over Lagrange multiplier lambda gives exact Euclidean projection.
        v_act = v[act_idx]
        lo = float((v_act - 1.0).min().item()) - 1.0
        hi = float(v_act.max().item()) + 1.0
        for _ in range(36):
            mid = 0.5 * (lo + hi)
            g0 = max(g0_floor, float(v_act[0].item()) - mid)
            g_rem = torch.clamp(v_act[1:] - mid, min=0.0)
            s = g0 + float(g_rem.sum().item())
            if s > 1.0:
                lo = mid
            else:
                hi = mid
        lam = 0.5 * (lo + hi)
        proj = torch.zeros_like(v)
        proj[act_idx[0]] = max(g0_floor, float(v_act[0].item()) - lam)
        proj[rem_idx] = torch.clamp(v_act[1:] - lam, min=0.0)
        tot = proj.sum().clamp_min(1e-8)
        if abs(float(tot.item()) - 1.0) > 1e-5:
            # Renormalize non-primary coordinates while preserving g0_floor
            proj[act_idx[0]] = proj[act_idx[0]].clamp(min=g0_floor, max=1.0)
            rem_sum = proj[rem_idx].sum()
            target_rem = max(0.0, 1.0 - float(proj[act_idx[0]].item()))
            if rem_sum > 1e-8:
                proj[rem_idx] = proj[rem_idx] * (target_rem / rem_sum)
            else:
                proj[rem_idx] = target_rem / float(len(rem_idx))
        return proj

    gamma = gram.new_zeros((k_dim,), dtype=torch.float32)
    act_idx = torch.nonzero(active, as_tuple=True)[0]
    gamma[0] = max(g0_floor, 1.0 / float(len(act_idx)))
    if len(act_idx) > 1:
        gamma[act_idx[1:]] = (1.0 - gamma[0]) / float(len(act_idx) - 1)

    y = gamma.clone()
    t_mom = 1.0
    for _ in range(n_iters):
        grad_y = 2.0 * (g_mat @ y)
        gamma_next = _project_constrained_simplex(y - step * grad_y)
        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t_mom * t_mom))
        y = gamma_next + ((t_mom - 1.0) / t_next) * (gamma_next - gamma)
        gamma = gamma_next
        t_mom = t_next
    return gamma.detach()


def mmpareto_combine_gradients(
    task_grads: Sequence[Sequence[torch.Tensor | None]],
    *,
    gamma0_min: float = 0.45,
) -> tuple[list[torch.Tensor | None], dict[str, Any]]:
    """Combine K task gradient lists via norm-preserving MMPareto projection.

    Args:
        task_grads: Sequence of length K (`task 0` is primary classification loss;
            `tasks 1..K-1` are auxiliary losses), each a list of per-parameter grads (or None).
        gamma0_min: Lower bound on primary task weight ensuring positive primary inner product.

    Returns:
        `(combined_grads, diagnostics)` where `combined_grads` has exact L2 norm `||g_0||_2`
        (when primary grad is non-zero) and `<g_mmpareto, g_0> > 0`.
    """
    k_tasks = len(task_grads)
    if k_tasks == 0:
        return [], {'gammas': [], 'cos_with_primary': 1.0, 'g0_norm': 0.0}
    n_params = len(task_grads[0])
    if n_params == 0:
        return [], {'gammas': [1.0] + [0.0] * (k_tasks - 1), 'cos_with_primary': 1.0, 'g0_norm': 0.0}

    ref_device = None
    for g_list in task_grads:
        for g in g_list:
            if g is not None:
                ref_device = g.device
                break
        if ref_device is not None:
            break
    if ref_device is None:
        return [None] * n_params, {'gammas': [1.0] + [0.0] * (k_tasks - 1), 'cos_with_primary': 1.0, 'g0_norm': 0.0}

    norms = torch.zeros(k_tasks, dtype=torch.float32, device=ref_device)
    active = torch.zeros(k_tasks, dtype=torch.bool, device=ref_device)
    for k in range(k_tasks):
        sq = 0.0
        for g in task_grads[k]:
            if g is not None:
                sq += float(g.detach().float().square().sum().item())
        nrm = math.sqrt(sq)
        norms[k] = nrm
        if nrm > 1e-10:
            active[k] = True

    if float(norms[0].item()) <= 1e-10:
        # Degenerate zero primary grad: average active auxiliary grads cleanly
        out_deg: list[torch.Tensor | None] = []
        n_act = max(1, int(active.sum().item()))
        for p_idx in range(n_params):
            acc = None
            for k in range(k_tasks):
                g = task_grads[k][p_idx]
                if g is not None and active[k]:
                    gf = g.float() / float(n_act)
                    acc = gf if acc is None else (acc + gf)
            out_deg.append(acc)
        return out_deg, {'gammas': [0.0] * k_tasks, 'cos_with_primary': 1.0, 'g0_norm': 0.0}

    # Build KxK normalized Gram matrix G_ij = <g_hat_i, g_hat_j>
    gram = torch.zeros((k_tasks, k_tasks), dtype=torch.float32, device=ref_device)
    for i in range(k_tasks):
        if not active[i]:
            continue
        for j in range(i, k_tasks):
            if not active[j]:
                continue
            dot = 0.0
            for p_idx in range(n_params):
                gi = task_grads[i][p_idx]
                gj = task_grads[j][p_idx]
                if gi is not None and gj is not None:
                    dot += float((gi.detach().float() * gj.detach().float()).sum().item())
            val = dot / (float(norms[i].item()) * float(norms[j].item()) + 1e-12)
            gram[i, j] = val
            gram[j, i] = val

    gammas = solve_mmpareto_weights(gram, gamma0_min=gamma0_min, active_mask=active)

    # Combine normalized gradients and compute exact norm for rescaling to ||g_0||_2
    comb_sq = 0.0
    dot_p0 = 0.0
    combined: list[torch.Tensor | None] = []
    for p_idx in range(n_params):
        acc = None
        for k in range(k_tasks):
            if not active[k] or float(gammas[k].item()) <= 0.0:
                continue
            g = task_grads[k][p_idx]
            if g is not None:
                contrib = (float(gammas[k].item()) / float(norms[k].item())) * g.detach().float()
                acc = contrib if acc is None else (acc + contrib)
        if acc is not None:
            comb_sq += float(acc.square().sum().item())
            g0 = task_grads[0][p_idx]
            if g0 is not None:
                dot_p0 += float((acc * g0.detach().float()).sum().item())
        combined.append(acc)

    comb_norm = math.sqrt(max(comb_sq, 1e-16))
    g0_norm = float(norms[0].item())
    scale = g0_norm / max(comb_norm, 1e-8)

    final_grads: list[torch.Tensor | None] = []
    for p_idx, acc in enumerate(combined):
        if acc is None:
            final_grads.append(None)
        else:
            ref_g = next((task_grads[k][p_idx] for k in range(k_tasks) if task_grads[k][p_idx] is not None), None)
            dt = ref_g.dtype if ref_g is not None else torch.float32
            final_grads.append((acc * scale).to(dtype=dt))

    cos_0 = dot_p0 / (comb_norm * g0_norm + 1e-12)
    return final_grads, {
        'gammas': [float(x) for x in gammas.cpu().tolist()],
        'cos_with_primary': float(cos_0),
        'g0_norm': g0_norm,
        'task_norms': [float(x) for x in norms.cpu().tolist()],
    }


@dataclass
class DomainCGGMConfig:
    """Hyperparameters for Domain-active CGGM + MMPareto + MCR conflict-free optimization."""

    ema_decay: float = 0.90
    kappa_max: float = 2.5
    kappa_gain: float = 0.8
    fusion_align_weight: float = 0.30
    mmpareto_gamma0_min: float = 0.45
    mcr_weight: float = 0.12
    mcr_unimodal_weight: float = 0.15
    probe_loss_weight: float = 0.25


class DomainCGGMProbes(nn.Module):
    """Training-only unimodal linear probes `h_m(z_m)` (zero overhead at inference)."""

    def __init__(self, embed_dim: int = 256, num_classes: int = 4) -> None:
        super().__init__()
        self.probes = nn.ModuleDict({
            'si_vtx': nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes)),
            'cal_pflow': nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes)),
            'spec_ipt': nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes)),
        })

    def forward(
        self,
        latents: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for mod, head in self.probes.items():
            if mod in latents and latents[mod] is not None:
                out[mod] = head(latents[mod].float())
        return out


class DomainCGGMTracker:
    """Tracks EMA marginal log-likelihood utilities `u_m^(t)` on active subsets `B_m`."""

    def __init__(self, cfg: DomainCGGMConfig | None = None) -> None:
        self.cfg = cfg or DomainCGGMConfig()
        self.ema_util: dict[str, float] = {m: 0.0 for m in GEN4_MODALITY_GROUPS}
        self.initialized: dict[str, bool] = {m: False for m in GEN4_MODALITY_GROUPS}

    @torch.no_grad()
    def update_and_get_kappas(
        self,
        probe_logits: Mapping[str, torch.Tensor],
        labels4: torch.Tensor,
        presence_masks: Mapping[str, torch.Tensor],
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Update per-modality active log-likelihood EMA and return gradient scalers `kappa_m`."""
        y4 = labels4.long()
        cur_u: dict[str, float] = {}
        for m in GEN4_MODALITY_GROUPS:
            if m not in probe_logits or m not in presence_masks:
                continue
            act = presence_masks[m].bool()
            if not act.any():
                continue
            lg = probe_logits[m][act].float()
            yt = y4[act]
            # Marginal accuracy / log-likelihood utility (higher = stronger modality)
            logp = F.log_softmax(lg, dim=-1)
            ll = float(logp[torch.arange(len(yt), device=lg.device), yt].mean().item())
            # Convert negative NLL into positive utility exp(ll) in (0, 1]
            util = math.exp(max(-6.0, min(0.0, ll)))
            cur_u[m] = util
            if not self.initialized[m]:
                self.ema_util[m] = util
                self.initialized[m] = True
            else:
                d = float(self.cfg.ema_decay)
                self.ema_util[m] = d * self.ema_util[m] + (1.0 - d) * util

        active_utils = [self.ema_util[m] for m in cur_u if self.initialized[m]]
        if not active_utils:
            return {m: 1.0 for m in GEN4_MODALITY_GROUPS}, {m: 1.0 for m in GEN4_MODALITY_GROUPS}

        max_u = max(active_utils)
        kappas: dict[str, float] = {}
        ratios: dict[str, float] = {}
        for m in GEN4_MODALITY_GROUPS:
            if m not in cur_u:
                # Modality absent in this batch (e.g. Fortune_ScintOnly has no Si): keep 1.0 (never boost!)
                kappas[m] = 1.0
                ratios[m] = 1.0
                continue
            rho = max_u / max(self.ema_util[m], 1e-4)
            ratios[m] = float(rho)
            raw_k = 1.0 + float(self.cfg.kappa_gain) * math.tanh(max(0.0, rho - 1.0))
            kappas[m] = float(min(max(raw_k, 1.0), float(self.cfg.kappa_max)))
        return kappas, ratios


def compute_mcr_synergy_loss(
    fusion_module: nn.Module,
    head_fn: Any,
    z_si: torch.Tensor,
    z_cal: torch.Tensor,
    z_spec: torch.Tensor,
    labels4: torch.Tensor,
    presence_masks: Mapping[str, torch.Tensor],
    *,
    extra_obs: Mapping[str, torch.Tensor] | None = None,
    margin_clamp: float = 1.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute Cross-Modal Synergy Regularization `R_syn` (MCR, NeurIPS 2025) in FP32.

    Evaluated strictly on samples where BOTH silicon and calorimeter are active (`a_si & a_cal`)
    so empty-silicon Fortune events are never corrupted by fake cross-modal swaps.
    """
    both_active = presence_masks['si_vtx'].bool() & presence_masks['cal_pflow'].bool()
    n_both = int(both_active.sum().item())
    if n_both < 4:
        zero = z_si.sum() * 0.0
        return zero, {'mcr_syn_gain': 0.0, 'mcr_n_active': float(n_both)}

    idx = torch.nonzero(both_active, as_tuple=True)[0]
    # Deterministic derangement shift within the active sub-batch
    perm = torch.roll(idx, shifts=1, dims=0)

    def _eval_logp(zs: torch.Tensor, zc: torch.Tensor, zp: torch.Tensor) -> torch.Tensor:
        zf = fusion_module(zs, zc, zp)
        obs = {k: v[idx] for k, v in (extra_obs or {}).items()}
        out = head_fn(zf, **obs) if obs else head_fn(zf)
        if isinstance(out, dict):
            if 'probs4' in out:
                logp = torch.log(out['probs4'].float().clamp_min(1e-7))
            elif 'logits7' in out:
                logp = torch.log(marginal_four_from_seven(out['logits7']).clamp_min(1e-7))
            else:
                logp = F.log_softmax(out['logits4'].float(), dim=-1)
        else:
            logp = F.log_softmax(out.float(), dim=-1)
        yt = labels4[idx].long()
        return logp[torch.arange(n_both, device=logp.device), yt]

    with torch.amp.autocast(device_type=z_si.device.type, enabled=False):
        zs_a = z_si[idx].float()
        zc_a = z_cal[idx].float()
        zp_a = z_spec[idx].float()
        zs_p = z_si[perm].detach().float()
        zc_p = z_cal[perm].detach().float()

        ll_joint = _eval_logp(zs_a, zc_a, zp_a)
        with torch.no_grad():
            ll_swap_cal = _eval_logp(zs_a, zc_p, zp_a)
            ll_swap_si = _eval_logp(zs_p, zc_a, zp_a)
        # Synergy gain: positive when joint modalities outperform permuted pairs
        syn = ll_joint - 0.5 * (ll_swap_cal + ll_swap_si)
        # Soft-clamped maximization of synergy gain (avoid destabilizing joint NLL)
        loss_mcr = -torch.clamp(syn, min=-float(margin_clamp), max=float(margin_clamp)).mean()
        return loss_mcr, {
            'mcr_syn_gain': float(syn.detach().mean().item()),
            'mcr_n_active': float(n_both),
        }


def compute_probe_loss(
    probe_logits: Mapping[str, torch.Tensor],
    labels4: torch.Tensor,
    presence_masks: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Cross-entropy of unimodal training-only probes strictly on modality-active subsets."""
    y4 = labels4.long()
    device = y4.device
    loss_sum = torch.zeros((), dtype=torch.float32, device=device)
    n_active_mods = 0
    stats: dict[str, float] = {}
    for m, lg in probe_logits.items():
        act = presence_masks.get(m)
        if act is None or not act.any():
            continue
        ce = F.cross_entropy(lg[act].float(), y4[act])
        loss_sum = loss_sum + ce
        n_active_mods += 1
        stats[f'probe_ce_{m}'] = float(ce.detach().item())
    if n_active_mods == 0:
        # Touch probe parameters with 0 weight so DDP/grad checks remain well-defined
        for lg in probe_logits.values():
            loss_sum = loss_sum + lg.sum() * 0.0
    else:
        loss_sum = loss_sum / float(n_active_mods)
    return loss_sum, stats


def apply_conflict_free_backward(
    model: nn.Module,
    primary_loss: torch.Tensor,
    aux_losses: Sequence[torch.Tensor],
    *,
    presence_masks: Mapping[str, torch.Tensor],
    kappas: Mapping[str, float],
    probe_loss: torch.Tensor | None = None,
    weak_probe_loss: torch.Tensor | None = None,
    cfg: DomainCGGMConfig | None = None,
    mcr_loss: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Perform conflict-free backward combining MMPareto, Domain-CGGM, and MCR.

    Guarantees:
      1. `si_vtx`, `cal_pflow`, `spec_ipt`, and `fusion` receive norm-preserving MMPareto
         gradients over `[primary_loss, *aux_losses]` so auxiliary tasks never shrink or
         oppose primary classification (`<g_mmpareto, g_0> > 0`).
      2. Active weak modality encoders (`kappa_m > 1.0`) have their backward gradient
         scaled by `kappa_m` (`Domain-CGGM`).
      3. `fusion` receives `fusion_align_weight` (0.30) norm-normalized weak-modality probe
         direction alignment when `weak_probe_loss` is provided.
      4. `head` parameters receive clean `primary_loss + mcr_loss` gradients; unimodal probes
         receive `probe_loss` gradients (isolated from encoders if desired or joint).
    """
    c = cfg or DomainCGGMConfig()
    groups = collect_gen4_param_groups(model)
    shared_params: list[nn.Parameter] = []
    shared_group_slices: dict[str, slice] = {}
    for gname in GEN4_SHARED_GROUPS:
        plist = groups.get(gname, [])
        start = len(shared_params)
        shared_params.extend(plist)
        shared_group_slices[gname] = slice(start, len(shared_params))

    head_params = groups.get('head', [])
    all_model_params = shared_params + head_params

    # 1. Compute per-task gradients for MMPareto on shared parameters
    # Combine primary_loss + mcr_loss for head_params, and compute task_grads on shared_params
    prim_for_shared = primary_loss + (float(c.mcr_weight) * mcr_loss if mcr_loss is not None else 0.0)
    tasks_to_eval = [prim_for_shared] + [al for al in aux_losses if al is not None]

    task_grads_shared: list[list[torch.Tensor | None]] = []
    if shared_params:
        for t_idx, t_loss in enumerate(tasks_to_eval):
            if not t_loss.requires_grad:
                task_grads_shared.append([None] * len(shared_params))
                continue
            retain = True
            grads = torch.autograd.grad(
                t_loss,
                shared_params,
                retain_graph=retain,
                allow_unused=True,
            )
            task_grads_shared.append(list(grads))

    # 2. Compute weak-modality probe direction on fusion params (Domain-CGGM fusion alignment)
    fusion_sl = shared_group_slices.get('fusion', slice(0, 0))
    fusion_params = shared_params[fusion_sl]
    weak_fusion_grads: list[torch.Tensor | None] | None = None
    if weak_probe_loss is not None and weak_probe_loss.requires_grad and fusion_params:
        weak_fusion_grads = list(
            torch.autograd.grad(
                weak_probe_loss,
                fusion_params,
                retain_graph=True,
                allow_unused=True,
            )
        )

    # 3. Compute head gradients cleanly (+ probe parameters if attached separately)
    head_grads: list[torch.Tensor | None] = []
    if head_params:
        head_loss = prim_for_shared
        for al in aux_losses:
            if al is not None and al.requires_grad:
                head_loss = head_loss + 0.0 * al
        if head_loss.requires_grad:
            head_grads = list(
                torch.autograd.grad(
                    head_loss,
                    head_params,
                    retain_graph=(probe_loss is not None and probe_loss.requires_grad),
                    allow_unused=True,
                )
            )
        else:
            head_grads = [None] * len(head_params)

    # 4. Probe loss backward (updates probe heads and provides unimodal encoder supervision)
    probe_enc_grads: list[torch.Tensor | None] = [None] * len(shared_params)
    if probe_loss is not None and probe_loss.requires_grad:
        probe_params = [p for n, p in model.named_parameters() if p.requires_grad and 'probe' in n]
        target_p = shared_params + probe_params
        p_grads = torch.autograd.grad(
            float(c.probe_loss_weight) * probe_loss,
            target_p,
            retain_graph=False,
            allow_unused=True,
        )
        probe_enc_grads = list(p_grads[: len(shared_params)])
        for p, gp in zip(probe_params, p_grads[len(shared_params) :]):
            if gp is not None:
                p.grad = gp.detach().clone() if p.grad is None else (p.grad + gp.detach())

    # 5. Apply MMPareto combination + Domain-CGGM modality scaling & fusion alignment
    mmp_diag: dict[str, Any] = {'gammas': [1.0], 'cos_with_primary': 1.0, 'g0_norm': 0.0}
    if shared_params:
        comb_shared, mmp_diag = mmpareto_combine_gradients(
            task_grads_shared,
            gamma0_min=float(c.mmpareto_gamma0_min),
        )
        # Add unimodal probe encoder gradients cleanly
        for i in range(len(shared_params)):
            pg = probe_enc_grads[i]
            if pg is not None:
                comb_shared[i] = pg.detach() if comb_shared[i] is None else (comb_shared[i] + pg.detach())

        # Domain-CGGM: scale active modality encoders by kappa_m
        for mod in GEN4_MODALITY_GROUPS:
            kap = float(kappas.get(mod, 1.0))
            if abs(kap - 1.0) > 1e-5 and bool(presence_masks[mod].any().item()):
                sl = shared_group_slices[mod]
                for i in range(sl.start, sl.stop):
                    if comb_shared[i] is not None:
                        comb_shared[i] = comb_shared[i] * kap

        # Domain-CGGM: align fusion layer gradient toward norm-normalized weak-modality direction
        if weak_fusion_grads is not None and float(c.fusion_align_weight) > 0.0:
            f_start, f_stop = fusion_sl.start, fusion_sl.stop
            g_fuse_sq = sum(
                float(comb_shared[i].float().square().sum().item())
                for i in range(f_start, f_stop)
                if comb_shared[i] is not None
            )
            g_weak_sq = sum(
                float(wg.float().square().sum().item())
                for wg in weak_fusion_grads
                if wg is not None
            )
            if g_fuse_sq > 1e-16 and g_weak_sq > 1e-16:
                scale_w = float(c.fusion_align_weight) * math.sqrt(g_fuse_sq / g_weak_sq)
                for offset, wg in enumerate(weak_fusion_grads):
                    if wg is not None:
                        idx_p = f_start + offset
                        contrib = (wg.detach() * scale_w).to(dtype=shared_params[idx_p].dtype)
                        comb_shared[idx_p] = contrib if comb_shared[idx_p] is None else (comb_shared[idx_p] + contrib)

        for p, g in zip(shared_params, comb_shared):
            if g is not None:
                gd = g.detach()
                p.grad = gd if p.grad is None else (p.grad + gd)

    for p, g in zip(head_params, head_grads):
        if g is not None:
            gd = g.detach()
            p.grad = gd if p.grad is None else (p.grad + gd)

    return {
        'mmpareto': mmp_diag,
        'kappas': {k: float(v) for k, v in kappas.items()},
    }
