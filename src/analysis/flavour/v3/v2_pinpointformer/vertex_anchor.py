"""Differentiable primary vertex anchor predictor and FP32 Stage-2 polar/IP geometry."""

from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from analysis.flavour.v3.common.layers import ScaledQKNormAttention


K_VTX_MAX = 128
PAIR_FEAT_DIM = 10
VERTEX_AXIS_SCALES_MM = (0.20, 0.20, 2.50)


def heteroscedastic_vertex_loss(
    v_hat: torch.Tensor,
    log_var_v: torch.Tensor,
    v_true: torch.Tensor,
    vertex_valid: torch.Tensor,
    *,
    legacy_biased: bool = False,
) -> torch.Tensor:
    """Unbiased heteroscedastic SmoothL1+NLL vertex loss in FP32 (resolving Audit P2-2).

    At the stationary point d/d(lv) [0.5 * (exp(-lv) * L + lv) + C] = 0, exp(lv) = L without
    any floor distortion. Placing a constant +4.0 OUTSIDE exp(-lv)*L keeps logged values > 0
    while permitting sub-100-um transverse and mm-level longitudinal uncertainty calibration.
    """
    valid = vertex_valid > 0
    if not valid.any():
        return v_hat.sum() * 0.0
    with torch.autocast(device_type=v_hat.device.type, enabled=False):
        if legacy_biased:
            scale = v_hat.new_tensor((1.0, 1.0, 10.0), dtype=torch.float32)
            lv = torch.clamp(log_var_v[valid].float(), min=-4.0, max=4.0)
            err = (v_hat[valid].float() - v_true[valid].float()) / scale
            l_axis = F.smooth_l1_loss(err, torch.zeros_like(err), reduction='none') + 2.0
            return (0.5 * torch.exp(-lv) * l_axis + 0.5 * lv).mean()

        scale = v_hat.new_tensor(VERTEX_AXIS_SCALES_MM, dtype=torch.float32)
        lv = torch.clamp(log_var_v[valid].float(), min=-6.0, max=6.0)
        err = (v_hat[valid].float() - v_true[valid].float()) / scale
        l_axis = F.smooth_l1_loss(err, torch.zeros_like(err), beta=1.0, reduction='none') + 1e-4
        nll = 0.5 * (torch.exp(-lv) * l_axis + lv) + 4.0
        return nll.mean()


def select_near_vertex_clusters(
    pix_xyz_mm: torch.Tensor,
    pix_mask: torch.Tensor,
    v_0_mm: torch.Tensor,
    k_vtx: int = K_VTX_MAX,
    pix_plane: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select up to k_vtx plane-stratified clusters closest to [-15, +250] mm vertex window."""
    b, m, _ = pix_xyz_mm.shape
    k = min(k_vtx, m)
    with torch.autocast(device_type=pix_xyz_mm.device.type, enabled=False):
        xyz = pix_xyz_mm.float()
        v0 = v_0_mm.float()
        dx = xyz[..., 0] - v0[:, None, 0]
        dy = xyz[..., 1] - v0[:, None, 1]
        dz = xyz[..., 2] - v0[:, None, 2]
        r_perp = torch.sqrt((dx * dx + dy * dy).clamp_min(1e-8))
        in_win = pix_mask & (dz >= -15.0) & (dz <= 250.0)
        score = torch.where(
            in_win,
            torch.abs(dz) + 0.5 * r_perp,
            torch.where(pix_mask, 1e4 + torch.abs(dz) + 0.5 * r_perp, 1e9),
        )
        if pix_plane is not None and m > 1:
            pl = torch.where(pix_mask, pix_plane.long(), torch.full_like(pix_plane.long(), 99))
            plane_key = pl.float() * 1e5 + torch.where(
                pix_mask, r_perp.clamp_max(9e4), torch.full_like(r_perp, 9e4)
            )
            sort_ord = torch.argsort(plane_key, dim=1, stable=True)
            pl_sorted = torch.gather(pl, 1, sort_ord)
            is_start = torch.ones_like(pl_sorted, dtype=torch.bool)
            is_start[:, 1:] = pl_sorted[:, 1:] != pl_sorted[:, :-1]
            pos_grid = torch.arange(m, device=pix_xyz_mm.device).unsqueeze(0).expand(b, m)
            start_pos, _ = torch.cummax(
                torch.where(is_start, pos_grid, torch.zeros_like(pos_grid)), dim=1
            )
            rank_sorted = pos_grid - start_pos
            rank_in_plane = torch.zeros_like(rank_sorted).scatter(1, sort_ord, rank_sorted)
            score = torch.where(
                in_win,
                score + 250.0 * torch.clamp(rank_in_plane - 16, min=0).float(),
                score,
            )
        _, idx = torch.topk(score, k=k, dim=1, largest=False)
        vtx_mask = torch.gather(pix_mask, 1, idx)
    return idx, vtx_mask


class VertexAnchorPredictor(nn.Module):
    """Two-stage (plane-coarse + cluster-fine) primary vertex v_hat (mm) & log-variance."""

    def __init__(self, dim: int = 192, heads: int = 8) -> None:
        super().__init__()
        self.front_queries = nn.Parameter(torch.randn(1, 4, dim) * 0.02)
        self.cross = ScaledQKNormAttention(dim=dim, heads=heads, init_scale=8.0, drop_path=0.05)
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
        )
        self.delta_coarse = nn.Linear(dim, 3)

        self.fine_rel_proj = nn.Sequential(
            nn.Linear(6, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
        )
        self.fine_queries = nn.Parameter(torch.randn(1, 4, dim) * 0.02)
        self.fine_cross = ScaledQKNormAttention(dim=dim, heads=heads, init_scale=10.0, drop_path=0.05)
        self.fine_head = nn.Sequential(
            nn.LayerNorm(2 * dim),
            nn.Linear(2 * dim, dim),
            nn.SiLU(),
        )
        self.delta_fine = nn.Linear(dim, 3)
        self.log_var = nn.Linear(dim, 3)
        nn.init.normal_(self.delta_fine.weight, std=1e-3)
        nn.init.zeros_(self.delta_fine.bias)

    def forward(
        self,
        front_tokens: torch.Tensor,
        front_mask: torch.Tensor,
        v_prior_mm: torch.Tensor,
        pix_tokens: torch.Tensor | None = None,
        pix_xyz_mm: torch.Tensor | None = None,
        pix_mask: torch.Tensor | None = None,
        pix_plane: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b = len(front_tokens)
        q = self.front_queries.expand(b, -1, -1)
        h_coarse = self.head(self.cross(q, front_tokens, front_mask).mean(dim=1))

        with torch.autocast(device_type=h_coarse.device.type, enabled=False):
            hf = h_coarse.float()
            prior = v_prior_mm.float()
            coarse_scale = prior.new_tensor([150.0, 150.0, 350.0])
            v_coarse = prior + torch.tanh(self.delta_coarse(hf)) * coarse_scale

        if pix_tokens is not None and pix_xyz_mm is not None and pix_mask is not None:
            k_sel = min(64, pix_xyz_mm.shape[1])
            c_idx, c_mask = select_near_vertex_clusters(
                pix_xyz_mm, pix_mask, v_coarse.detach(), k_vtx=k_sel, pix_plane=pix_plane
            )
            c_tok = torch.gather(pix_tokens, 1, c_idx.unsqueeze(-1).expand(-1, -1, pix_tokens.shape[-1]))
            c_xyz = torch.gather(pix_xyz_mm.float(), 1, c_idx.unsqueeze(-1).expand(-1, -1, 3))
            rel = c_xyz - v_coarse[:, None, :]
            rel_feat = torch.stack([
                rel[..., 0] / 15.0,
                rel[..., 1] / 15.0,
                rel[..., 2] / 40.0,
                torch.asinh(rel[..., 0] / 2.0) / 3.0,
                torch.asinh(rel[..., 1] / 2.0) / 3.0,
                torch.asinh(rel[..., 2] / 10.0) / 3.0,
            ], dim=-1)
            c_in = c_tok + self.fine_rel_proj(rel_feat.to(c_tok.dtype)) * c_mask.unsqueeze(-1).to(c_tok.dtype)
            h_fine_attn = self.fine_cross(self.fine_queries.expand(b, -1, -1), c_in, c_mask).mean(dim=1)
            any_pix = c_mask.any(dim=1, keepdim=True)
            h_fine_safe = torch.where(any_pix, h_fine_attn, h_coarse)
        else:
            h_fine_safe = h_coarse

        h_comb = self.fine_head(torch.cat([h_coarse, h_fine_safe], dim=-1))
        with torch.autocast(device_type=h_comb.device.type, enabled=False):
            hcf = h_comb.float()
            fine_scale = v_coarse.new_tensor([25.0, 25.0, 80.0])
            v_hat = v_coarse + torch.tanh(self.delta_fine(hcf)) * fine_scale
            log_var_v = torch.clamp(self.log_var(hcf), min=-6.0, max=6.0)
        return v_hat, log_var_v


def compute_stage2_polar_features(
    xyz_mm: torch.Tensor,
    measured_xy: torch.Tensor,
    v_0_mm: torch.Tensor,
    log_var_v: torch.Tensor | None = None,
) -> torch.Tensor:
    """Transform 3D coordinates into smooth uncertainty-regularized vertex polar features in FP32.

    Every Y-mirror flips ONLY indices 1 (dy) and 5 (theta_yz); unmeasured axes are masked to zero.
    """
    with torch.autocast(device_type=xyz_mm.device.type, enabled=False):
        xyz = xyz_mm.float()
        mxy = measured_xy.float()
        v0 = v_0_mm.float()

        mx = mxy[..., 0]
        my = mxy[..., 1]
        both = mx * my
        any_xy = torch.clamp(mx + my, max=1.0)

        dx = (xyz[..., 0] - v0[:, None, 0]) * mx
        dy = (xyz[..., 1] - v0[:, None, 1]) * my
        dz = xyz[..., 2] - v0[:, None, 2]
        r_perp = torch.sqrt((dx * dx + dy * dy).clamp_min(1e-8)) * any_xy

        sig_z_sq = (
            torch.exp(torch.clamp(log_var_v[:, 2:3].float(), -4.0, 6.0)) * 100.0
            if log_var_v is not None
            else 0.0
        )
        # Smooth odd denominator (no discontinuity at dz=0; shrinks angular blowup when v0_z uncertain)
        inv_dz = dz / (dz * dz + sig_z_sq + 100.0)
        theta_xz = torch.clamp(dx * inv_dz * 10.0, -5.0, 5.0) * mx
        theta_yz = torch.clamp(dy * inv_dz * 10.0, -5.0, 5.0) * my
        theta_rad = torch.sqrt((theta_xz * theta_xz + theta_yz * theta_yz).clamp_min(1e-8)) * any_xy

        return torch.stack([
            torch.asinh(dx / 2.0) / 5.0,
            torch.asinh(dy / 2.0) / 5.0,
            torch.asinh(dz / 20.0) / 5.0,
            torch.asinh(r_perp / 2.0) / 5.0,
            theta_xz,
            theta_yz,
            torch.asinh(theta_rad * 20.0) / 4.0,
            both,
        ], dim=-1)


def compute_near_vertex_pair_features(
    vtx_xyz_mm: torch.Tensor,
    vtx_x0: torch.Tensor,
    vtx_mask: torch.Tensor,
    v_0_mm: torch.Tensor,
    log_var_v: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute 10-dim mirror-invariant pairwise IP_perp, sub-mm sep, and kink features (FP32)."""
    with torch.autocast(device_type=vtx_xyz_mm.device.type, enabled=False):
        xyz = vtx_xyz_mm.float()
        x0 = vtx_x0.float()
        v0 = v_0_mm.float()

        x = xyz[..., 0]
        y = xyz[..., 1]
        z = xyz[..., 2]

        dz_v = z - v0[:, None, 2]
        dz_v_safe = torch.where(dz_v >= 0.0, dz_v.clamp_min(1.0), dz_v.clamp_max(-1.0))
        tx = torch.clamp((x - v0[:, None, 0]) / dz_v_safe, -5.0, 5.0)
        ty = torch.clamp((y - v0[:, None, 1]) / dz_v_safe, -5.0, 5.0)

        dx_ij = x[:, None, :] - x[:, :, None]  # (B, K, K): j - i
        dy_ij = y[:, None, :] - y[:, :, None]
        dz_ij = z[:, None, :] - z[:, :, None]
        abs_dz = torch.abs(dz_ij)
        pair_valid = vtx_mask[:, :, None] & vtx_mask[:, None, :]
        cross_layer = (abs_dz >= 0.5) & pair_valid
        same_plane = (abs_dz < 0.5) & pair_valid
        r_xy_ij = torch.sqrt((dx_ij * dx_ij + dy_ij * dy_ij).clamp_min(1e-10)) * pair_valid.float()

        denom_z = torch.where(dz_ij >= 0.0, dz_ij.clamp_min(1.0), dz_ij.clamp_max(-1.0))
        sx_ij = torch.where(cross_layer, torch.clamp(dx_ij / denom_z, -10.0, 10.0), 0.0)
        sy_ij = torch.where(cross_layer, torch.clamp(dy_ij / denom_z, -10.0, 10.0), 0.0)

        # Back-project (i, j) secant line to vertex z = v0_z -> transverse IP_perp
        vz_rel_i = v0[:, None, None, 2] - z[:, :, None]
        ip_x = (x[:, :, None] + sx_ij * vz_rel_i) - v0[:, None, None, 0]
        ip_y = (y[:, :, None] + sy_ij * vz_rel_i) - v0[:, None, None, 1]
        ip_perp = torch.sqrt((ip_x * ip_x + ip_y * ip_y).clamp_min(1e-10)) * cross_layer.float()

        sig_xy_sq = (
            torch.exp(torch.clamp(log_var_v[:, :2].float(), -4.0, 4.0)).mean(dim=-1)[:, None, None]
            if log_var_v is not None
            else xyz.new_zeros((xyz.shape[0], 1, 1))
        )
        sig_xy = torch.sqrt(0.0025 + sig_xy_sq)

        # Angle between vertex ray u_i = norm(x_i - v0) and forward chord vector from i to j
        w_vec = xyz - v0[:, None, :]
        w_norm = torch.linalg.norm(w_vec, dim=-1, keepdim=True).clamp_min(1e-3)
        u_vec = w_vec / w_norm

        sgn_z = torch.where(dz_ij >= 0.0, 1.0, -1.0)
        d_vec = (xyz[:, None, :, :] - xyz[:, :, None, :]) * sgn_z.unsqueeze(-1)
        d_norm = torch.linalg.norm(d_vec, dim=-1, keepdim=True).clamp_min(1e-3)
        d_hat = d_vec / d_norm

        cos_kink = torch.clamp((u_vec[:, :, None, :] * d_hat).sum(dim=-1), -1.0 + 1e-6, 1.0 - 1e-6)
        sin_kink = torch.sqrt((1.0 - cos_kink * cos_kink).clamp_min(0.0))

        d_theta = torch.sqrt(
            (tx[:, None, :] - tx[:, :, None]).square()
            + (ty[:, None, :] - ty[:, :, None]).square()
            + 1e-10
        )
        dx0_ij = torch.abs(x0[:, None, :] - x0[:, :, None])
        causal_sign = dz_ij / (abs_dz + 1.0)

        return torch.stack([
            torch.log1p(abs_dz) / 5.0,
            dx0_ij / 10.0,
            torch.log1p(100.0 * d_theta) / 5.0,
            torch.log1p(ip_perp / 0.050) / 5.0,
            cos_kink * cross_layer.float(),
            torch.log1p(100.0 * sin_kink) / 5.0 * cross_layer.float(),
            causal_sign,
            torch.log1p(r_xy_ij / 0.050) / 5.0,
            same_plane.float(),
            torch.log1p(ip_perp / sig_xy) / 5.0 * cross_layer.float(),
        ], dim=-1)


def compute_near_vertex_pair_bias(
    pair_mlp: nn.Module,
    vtx_xyz_mm: torch.Tensor,
    vtx_x0: torch.Tensor,
    vtx_mask: torch.Tensor,
    v_0_mm: torch.Tensor,
    log_var_v: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project 10-dim pairwise IP/kink geometry to (B, heads, K_vtx, K_vtx) attention bias."""
    feat = compute_near_vertex_pair_features(vtx_xyz_mm, vtx_x0, vtx_mask, v_0_mm, log_var_v=log_var_v)
    return pair_mlp(feat).permute(0, 3, 1, 2).contiguous()
