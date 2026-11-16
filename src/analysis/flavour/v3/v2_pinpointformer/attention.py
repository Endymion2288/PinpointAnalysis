"""Perceiver cross/self-attention blocks wrapping FP32-exact ScaledQKNormAttention."""

from __future__ import annotations

import torch
from torch import nn

from analysis.flavour.v3.common.layers import ScaledQKNormAttention


def collect_attention_scales(module: nn.Module) -> dict[str, list[float]]:
    """Return clamped per-head inverse temperatures gamma_h across all ScaledQKNormAttention layers."""
    out: dict[str, list[float]] = {}
    for name, sub in module.named_modules():
        if isinstance(sub, ScaledQKNormAttention):
            scales = torch.clamp(sub.log_scale.exp(), min=1.0, max=sub.max_scale)
            out[name] = [float(x) for x in scales.detach().cpu().view(-1).tolist()]
    return out


class PerceiverCrossBlock(nn.Module):
    """Latent cross-attention over physical detector tokens followed by latent self-attention."""

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
        self.cross = ScaledQKNormAttention(
            dim=dim,
            heads=heads,
            init_scale=init_scale,
            max_scale=max_scale,
            residual_init=residual_init,
            drop_path=drop_path,
        )
        self.self_attn = ScaledQKNormAttention(
            dim=dim,
            heads=heads,
            init_scale=init_scale,
            max_scale=max_scale,
            residual_init=residual_init,
            drop_path=drop_path,
        )

    def learned_scales(self) -> dict[str, list[float]]:
        return collect_attention_scales(self)

    def forward(
        self,
        latents: torch.Tensor,
        source: torch.Tensor,
        source_mask: torch.Tensor,
        latent_mask: torch.Tensor | None = None,
        return_entropy: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if latent_mask is None:
            latent_mask = torch.ones(
                (latents.shape[0], latents.shape[1]), dtype=torch.bool, device=latents.device
            )
        if return_entropy:
            latents, ent = self.cross(latents, source, source_mask, return_entropy=True)
            latents = self.self_attn(latents, latents, latent_mask)
            return latents, ent
        latents = self.cross(latents, source, source_mask)
        return self.self_attn(latents, latents, latent_mask)


class AttentionStack(nn.Module):
    """Stack of self-attention ScaledQKNormAttention layers with optional pair bias."""

    def __init__(
        self,
        dim: int,
        n_layers: int,
        heads: int = 8,
        init_scale: float = 12.0,
        max_scale: float = 50.0,
        residual_init: float = 0.5,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            ScaledQKNormAttention(
                dim=dim,
                heads=heads,
                init_scale=init_scale,
                max_scale=max_scale,
                residual_init=residual_init,
                drop_path=drop_path,
            )
            for _ in range(n_layers)
        ])

    def learned_scales(self) -> dict[str, list[float]]:
        return collect_attention_scales(self)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pair_bias: torch.Tensor | None = None,
        return_entropy: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        ents: list[torch.Tensor] = []
        for layer in self.layers:
            if return_entropy:
                x, ent = layer(x, x, mask, pair_bias=pair_bias, return_entropy=True)
                ents.append(ent)
            else:
                x = layer(x, x, mask, pair_bias=pair_bias)
        if return_entropy:
            return x, ents
        return x
