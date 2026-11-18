"""Shared neural building blocks for v3 models (FP32-exact ScaledQKNormAttention & DropPath)."""

from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


class DropPath(nn.Module):
    """Stochastic depth per sample for residual branches."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep)
        return x * mask / keep


class ScaledQKNormAttention(nn.Module):
    """L2-normalized QK attention with per-head learnable scale and log(N) sharpness factor."""

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
        if dim % heads != 0:
            raise ValueError(f'dim ({dim}) must be divisible by heads ({heads})')
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
        x: torch.Tensor,
        source: torch.Tensor,
        source_mask: torch.Tensor,
        pair_bias: torch.Tensor | None = None,
        return_entropy: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        b, nq, _ = x.shape
        nk = source.shape[1]
        q = self.q_proj(self.q_norm(x)).view(b, nq, self.heads, self.head_dim).transpose(1, 2)
        k_in = self.k_norm(source)
        k = self.k_proj(k_in).view(b, nk, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(k_in).view(b, nk, self.heads, self.head_dim).transpose(1, 2)

        q_hat = F.normalize(q.float(), p=2, dim=-1, eps=1e-6)
        k_hat = F.normalize(k.float(), p=2, dim=-1, eps=1e-6)

        valid_count = source_mask.sum(dim=-1, keepdim=True).clamp(min=2).float()
        scale = torch.clamp(self.log_scale.exp(), min=1.0, max=self.max_scale)
        len_factor = 1.0 + F.softplus(self.len_beta) * torch.log(valid_count)[:, None, None, :]

        # Compute explicitly in FP32 so high-scale softmax stays exact under BF16 autocast
        logits = torch.matmul(q_hat, k_hat.transpose(-2, -1)) * (scale * len_factor)
        if pair_bias is not None:
            logits = logits + pair_bias.float()

        empty_row = ~source_mask.any(dim=-1)
        attn_mask = source_mask.clone()
        if empty_row.any():
            attn_mask[empty_row, 0] = True
        logits = logits.masked_fill(~attn_mask[:, None, None, :], float('-inf'))

        probs = torch.softmax(logits, dim=-1)
        if empty_row.any():
            probs = probs * (~empty_row)[:, None, None, None].to(probs.dtype)

        attended = torch.matmul(probs.to(v.dtype), v)
        attended = attended.transpose(1, 2).contiguous().view(b, nq, self.dim)
        h = x + self.drop_path_attn(self.gamma_attn * self.out_proj(attended))

        g, val = self.ff_up(self.ff_norm(h)).chunk(2, dim=-1)
        out = h + self.drop_path_ff(self.gamma_ff * self.ff_down(F.silu(g) * val))

        if not return_entropy:
            return out
        p_safe = probs.clamp(min=1e-12)
        ent = -(probs * torch.log(p_safe)).sum(dim=-1)
        denom = torch.log(valid_count)[:, None, None, :].clamp(min=1e-6)
        norm_ent = (ent / denom).mean(dim=-1)
        return out, norm_ent


NO_WD_KEYWORDS = (
    '.bias',
    'norm.',
    '.log_scale',
    '.len_beta',
    '.gamma_attn',
    '.gamma_ff',
    'polar_gate',
    'local_scatter_scale',
    'queries',
    'nomad_scale',
    'log_vars',
    'tau_q',
)


def is_weight_decay_exempt(name: str, param: nn.Parameter) -> bool:
    """Return True for 1D/scalar/norm/bias/temperature/gate/query parameters (Audit P2-3)."""
    if param.ndim < 2 or param.numel() <= 64:
        # Matrix weights in Linear/Conv always have ndim >= 2; 1D/scalar/head scales are exempt
        if param.ndim < 2 or any(k in name for k in NO_WD_KEYWORDS):
            return True
    return any(k in name for k in NO_WD_KEYWORDS)


def split_adamw_param_groups(
    module: nn.Module,
    base_lr: float,
    weight_decay: float,
) -> list[dict]:
    """Build standard AdamW parameter groups exempting all 1D/scalar/norm/gate/query tensors from WD."""
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if is_weight_decay_exempt(name, param):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    groups: list[dict] = []
    if decay_params:
        groups.append({'params': decay_params, 'lr': float(base_lr), 'weight_decay': float(weight_decay)})
    if no_decay_params:
        groups.append({'params': no_decay_params, 'lr': float(base_lr), 'weight_decay': 0.0})
    return groups
