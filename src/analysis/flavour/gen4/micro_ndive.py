"""Pillar I: MicroCone-Pairmixer-NDIVE (~1.74M params) for sub-mm vertexing & zero-loss silicon tracking.

Architecture components:
  1. DualTrackClusterEncoder: Track-A binary 4th-order kurtosis / dumbbell splitting + Track-B
     zero-offset gated edep residual + 32x32 micro-patch CNN + DarkShield persistence gate.
  2. AnisoPTv3: Dual-order (intra-plane 2D Hilbert + cross-plane projective slope Hilbert)
     windowed patch attention (W=64, N_cl <= 2304) with cluster origin & dark-noise heads.
  3. BackConeSlabPrior: Sandwich Scint[p0-1] disambiguation of W_back vs W_front (sigma_z = 1.44 mm).
  4. Vectorized GBL (Type-I 3-hit, Type-II 2-hit) + Type-III 1-Hit Parent-Tau Kink Stubs (M=128),
     fixing the IPT 1e-4 X0 air-gap floor and downstream MCS slope contamination.
  5. PV-free GTA Local Ray-Frame Attention + Pairmixer Triangle Multiplicative Closure +
     6 GN3 background registers -> 8-slot Competitive Slot Router (sum_k P_ik == 1).
  6. Comp-NDIVE: 4 no_grad IRLS steps + 1 exact implicit-differentiation Hessian step with
     material-adaptive X0(z) covariance, outputting full 3x3 covariances, S_L,k, M_corr,k,
     and gated 36-dim phi_vtx.

Zero-leakage contract:
  forward() consumes strictly pure observables; all cluster_sidecar / truth targets live
  exclusively inside loss computation so truth/sidecar tamper diff is strictly == 0.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from analysis.flavour.gen4.geometry_contract import (
    AIR_CONV_SUPPRESSION,
    Gen4GeometryContract,
    MAGNET0_LENGTH_MM,
    MAGNET0_Z_CENTER_MM,
    MAGNET0_Z_MAX_MM,
    MAGNET0_Z_MIN_MM,
    KAPPA_B_CONST_MEV_MM,
    TRACKER_Z_MM,
    load_geometry_contract,
)


IPT_X0_FLOOR = 1.0e-4
MAX_TARGET_CLUSTERS = 2048
MAX_IPT_CLUSTERS = 256
MAX_TOTAL_CLUSTERS = MAX_TARGET_CLUSTERS + MAX_IPT_CLUSTERS
N_SEG_TARGET = 96
N_SEG_IPT = 32
N_SEG_TOTAL = N_SEG_TARGET + N_SEG_IPT  # 128
N_COMP_SLOTS = 8  # {0:PV, 1:tau-SV, 2:Charm-SV, 3..6:gamma-V0_1..4, 7:ghost}
PHI_VTX_DIM = 36
SEG_STATE_DIM = 16
PAIR_R_DIM = 20


@dataclass
class MicroVtxOutput:
    """Standardized Pillar-I output contract matching [gen4-contract] Section III."""

    v_pv_mm: torch.Tensor              # (B, 3)
    cov_pv_mm2: torch.Tensor           # (B, 3, 3) SPD
    scale_pv_mm: torch.Tensor          # (B, 3) sqrt(diag(cov_pv_mm2))
    v_slots_mm: torch.Tensor           # (B, 7, 3)
    cov_slots_mm2: torch.Tensor        # (B, 7, 3, 3) SPD
    sv_disp_mm: torch.Tensor           # (B, 3) v_slots_mm[:, 1] - v_pv_mm
    seg_state: torch.Tensor            # (B, 128, 16)
    seg_mask: torch.Tensor             # (B, 128) bool
    seg_slot_probs: torch.Tensor       # (B, 128, 8) sum_k == 1
    phi_vtx: torch.Tensor              # (B, 36) zero-gated when c_obs[:, 0] == 0
    z_si_vtx: torch.Tensor             # (B, 256)
    u_micro: torch.Tensor              # (B, 128, 128)
    s_si_meso: torch.Tensor            # (B, 15, 192)
    aux: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        if key in self.aux:
            return self.aux[key]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        return self.aux.get(key, default)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and (hasattr(self, key) or key in self.aux)


def _safe_hypot(a: torch.Tensor, b: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    out = torch.sqrt(a.float().square() + b.float().square() + eps) - math.sqrt(eps)
    return out.to(a.dtype)


def _highland_theta_torch(dx0: torch.Tensor, p_gev: torch.Tensor) -> torch.Tensor:
    """Highland multiple Coulomb scattering RMS angle (rad) in FP32."""
    x0 = dx0.float().clamp_min(IPT_X0_FLOOR)
    p_mev = (p_gev.float() * 1000.0).clamp(min=150.0, max=5.0e6)
    return (13.6 / p_mev) * torch.sqrt(x0) * (1.0 + 0.038 * torch.log(x0.clamp_min(1.0)))


def _hilbert_2d_code(ix: torch.Tensor, iy: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Vectorized 2D Hilbert curve index for integer grid coords in [0, 2^bits - 1]."""
    n = 1 << bits
    x = ix.long().clamp(0, n - 1)
    y = iy.long().clamp(0, n - 1)
    d = torch.zeros_like(x)
    s = n >> 1
    while s > 0:
        rx = (x & s) > 0
        ry = (y & s) > 0
        rx_i = rx.long()
        ry_i = ry.long()
        d = d + (s * s) * ((3 * rx_i) ^ ry_i)
        # Rotate/flip quadrant where ~ry
        need_rot = ~ry
        need_flip = need_rot & rx
        x = torch.where(need_flip, (n - 1) - x, x)
        y = torch.where(need_flip, (n - 1) - y, y)
        x_new = torch.where(need_rot, y, x)
        y_new = torch.where(need_rot, x, y)
        x, y = x_new, y_new
        s >>= 1
    return d


def _solve_spd_3x3(mat: torch.Tensor, rhs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve 3x3 SPD system in FP32 with analytical fallback & full inverse 3x3 covariance.

    mat: (..., 3, 3), rhs: (..., 3).
    Returns (sol (..., 3), inv (..., 3, 3)).
    """
    orig_shape = rhs.shape[:-1]
    m_flat = mat.float().reshape(-1, 3, 3)
    r_flat = rhs.float().reshape(-1, 3)
    m_sym = 0.5 * (m_flat + m_flat.transpose(-1, -2))
    eye = torch.eye(3, dtype=torch.float32, device=m_sym.device).unsqueeze(0)
    m_reg = m_sym + 1.0e-5 * eye

    # Analytical adjugate & determinant (completely avoids cuSOLVER kernel overhead on small batch)
    a00, a01, a02 = m_reg[:, 0, 0], m_reg[:, 0, 1], m_reg[:, 0, 2]
    a11, a12, a22 = m_reg[:, 1, 1], m_reg[:, 1, 2], m_reg[:, 2, 2]

    c00 = a11 * a22 - a12 * a12
    c01 = -(a01 * a22 - a12 * a02)
    c02 = a01 * a12 - a11 * a02
    c11 = a00 * a22 - a02 * a02
    c12 = -(a00 * a12 - a01 * a02)
    c22 = a00 * a11 - a01 * a01

    det = (a00 * c00 + a01 * c01 + a02 * c02).clamp_min(1.0e-12)
    inv_det = 1.0 / det

    inv = torch.stack([
        torch.stack([c00, c01, c02], dim=-1),
        torch.stack([c01, c11, c12], dim=-1),
        torch.stack([c02, c12, c22], dim=-1),
    ], dim=-2) * inv_det[:, None, None]
    inv = 0.5 * (inv + inv.transpose(-1, -2))

    sol = torch.bmm(inv, r_flat.unsqueeze(-1)).squeeze(-1)
    return sol.reshape(*orig_shape, 3), inv.reshape(*orig_shape, 3, 3)


def _clamp_spd_cov3(cov: torch.Tensor, min_std_mm: float = 0.015, max_std_mm: float = 250.0) -> torch.Tensor:
    """Ensure 3x3 covariance tensor is strictly SPD with eigenvalues >= min_std^2 without eigh.

    Avoids PyTorch `linalg.eigh` backward 0/0 = NaN on degenerate diagonal covariances
    (e.g., empty-silicon Fortune_ScintOnly events in mixed batches).
    """
    cov_f = 0.5 * (cov.float() + cov.float().transpose(-1, -2))
    min_var = float(min_std_mm) ** 2
    max_var = float(max_std_mm) ** 2
    diag_max = torch.diagonal(cov_f, dim1=-2, dim2=-1).amax(dim=-1, keepdim=True).clamp_min(max_var)
    scale = (max_var / diag_max).unsqueeze(-1)
    eye = torch.eye(3, dtype=torch.float32, device=cov_f.device).view((1,) * (cov_f.ndim - 2) + (3, 3))
    out = cov_f * scale + min_var * eye
    return 0.5 * (out + out.transpose(-1, -2))


class DualTrackClusterEncoder(nn.Module):
    """Dual-track (binary kurtosis + zero-offset gated edep) cluster & 32x32 micro-patch encoder."""

    def __init__(self, d_model: int = 128, d_meso: int = 192) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_meso = d_meso

        self.mlp_geom = nn.Sequential(
            nn.Linear(30, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        # Small-scale init gated by m_edep so Track-A (has_edep==0) has exact zero shift
        # while Track-B (has_edep==1, 10085-10088 6-dim pixel edep) receives healthy step-0 gradients.
        edep_last = nn.Linear(64, d_model)
        nn.init.normal_(edep_last.weight, mean=0.0, std=0.02)
        nn.init.zeros_(edep_last.bias)
        self.mlp_edep = nn.Sequential(
            nn.Linear(6, 64),
            nn.SiLU(),
            edep_last,
        )

        self.patch_cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, d_model),
        )
        self.patch_norm = nn.LayerNorm(d_model)

        self.plane_summary_proj = nn.Sequential(
            nn.Linear(12, d_meso),
            nn.LayerNorm(d_meso),
            nn.SiLU(),
        )
        self.plane_id_emb = nn.Embedding(15, d_meso)
        self.cl_to_meso = nn.Linear(d_model, d_meso)
        self.meso_out = nn.Sequential(
            nn.LayerNorm(d_meso),
            nn.Linear(d_meso, d_meso),
            nn.SiLU(),
        )

    def forward(
        self,
        pix_feat: torch.Tensor,
        pix_valid: torch.Tensor,
        pix_plane: torch.Tensor,
        pix_plane_summary: torch.Tensor,
        pix_patches: torch.Tensor,
        patch_cluster_idx: torch.Tensor,
        has_edep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, n_cl, feat_dim = pix_feat.shape
        h_geom = self.mlp_geom(pix_feat[..., :30].float())
        m_edep = has_edep.float().view(b, 1, 1)
        if feat_dim >= 36:
            edep_in = pix_feat[..., 30:36].float()
        elif feat_dim > 30:
            edep_in = F.pad(pix_feat[..., 30:].float(), (0, 36 - feat_dim))
        else:
            edep_in = torch.zeros((b, n_cl, 6), dtype=torch.float32, device=pix_feat.device)
        h_edep = self.mlp_edep(edep_in) * m_edep
        h_cl = (h_geom + h_edep) * pix_valid.unsqueeze(-1).float()

        # Deterministic micro-patch CNN aggregation onto constituent clusters (zero atomicAdd)
        if pix_patches.shape[1] > 0 and n_cl > 0:
            p_valid = patch_cluster_idx >= 0
            if p_valid.any():
                k_p = pix_patches.shape[1]
                p_flat = pix_patches.reshape(b * k_p, pix_patches.shape[2], pix_patches.shape[3], pix_patches.shape[4]).float()
                p_emb = self.patch_norm(self.patch_cnn(p_flat)).view(b, k_p, -1)
                p_emb = p_emb * p_valid.unsqueeze(-1).float()
                cl_range = torch.arange(n_cl, device=pix_feat.device, dtype=torch.long)[None, :, None]
                assign_mat = (patch_cluster_idx[:, None, :] == cl_range).float()  # (B, N_cl, K_p)
                delta = torch.bmm(assign_mat, p_emb)
                h_cl = (h_cl + delta) * pix_valid.unsqueeze(-1).float()

        # Plane-level meso tokens (B, 15, d_meso)
        p_ids = torch.arange(15, device=pix_feat.device, dtype=torch.long)
        s_meso = self.plane_summary_proj(pix_plane_summary.float()) + self.plane_id_emb(p_ids).unsqueeze(0)
        if n_cl > 0:
            cl_m = self.cl_to_meso(h_cl) * pix_valid.unsqueeze(-1).float()
            pl_safe = pix_plane.clamp(0, 14)
            pl_one_hot = F.one_hot(pl_safe, num_classes=15).float() * pix_valid.unsqueeze(-1).float()
            pl_sum = torch.einsum('bnp,bnd->bpd', pl_one_hot, cl_m)
            pl_cnt = pl_one_hot.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
            s_meso = s_meso + pl_sum / pl_cnt
        s_meso = self.meso_out(s_meso)
        return h_cl, s_meso


class _HilbertWindowBlock(nn.Module):
    """Windowed Patch Attention (W=64) along serialized Hilbert order with relative coord bias."""

    def __init__(self, d_model: int = 128, heads: int = 4, window_size: int = 64) -> None:
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads
        self.window_size = window_size

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.rel_bias = nn.Linear(4, heads, bias=False)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        coords_norm: torch.Tensor,
        valid: torch.Tensor,
        order_idx: torch.Tensor,
    ) -> torch.Tensor:
        b, n, d = x.shape
        w = self.window_size
        # Pad to multiple of window_size
        rem = n % w
        pad_len = (w - rem) % w

        x_ord = torch.gather(x, 1, order_idx.unsqueeze(-1).expand(-1, -1, d))
        c_ord = torch.gather(coords_norm, 1, order_idx.unsqueeze(-1).expand(-1, -1, 4))
        v_ord = torch.gather(valid, 1, order_idx)

        if pad_len > 0:
            x_ord = F.pad(x_ord, (0, 0, 0, pad_len))
            c_ord = F.pad(c_ord, (0, 0, 0, pad_len))
            v_ord = F.pad(v_ord, (0, pad_len), value=False)

        n_win = x_ord.shape[1] // w
        xw = self.norm1(x_ord).view(b, n_win, w, d)
        cw = c_ord.view(b, n_win, w, 4)
        vw = v_ord.view(b, n_win, w)

        qkv = self.qkv(xw).view(b, n_win, w, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=3)
        q = q.permute(0, 1, 3, 2, 4)  # (B, n_win, H, w, hd)
        k = k.permute(0, 1, 3, 2, 4)
        v = v.permute(0, 1, 3, 2, 4)

        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        dc = cw[:, :, :, None, :] - cw[:, :, None, :, :]
        attn = attn + self.rel_bias(dc).permute(0, 1, 4, 2, 3)

        # Prevent all-masked window NaN
        has_any = vw.any(dim=-1, keepdim=True)
        safe_vw = torch.where(has_any, vw, torch.ones_like(vw))
        attn = attn.masked_fill(~safe_vw[:, :, None, None, :], -1.0e4)
        probs = torch.softmax(attn, dim=-1) * vw[:, :, None, None, :].float()

        out = torch.matmul(probs, v).permute(0, 1, 3, 2, 4).reshape(b, n_win * w, d)
        x_ord = x_ord + self.proj(out)
        x_ord = x_ord + self.ffn(self.norm2(x_ord))
        x_ord = x_ord * v_ord.unsqueeze(-1).float()

        if pad_len > 0:
            x_ord = x_ord[:, :n]

        # Scatter back to original cluster order
        x_out = torch.zeros_like(x)
        x_out.scatter_(1, order_idx.unsqueeze(-1).expand(-1, -1, d), x_ord)
        return x_out * valid.unsqueeze(-1).float()


class AnisoPTv3(nn.Module):
    """Dual-order (intra-plane + projective-slope Hilbert) sparse point transformer (W=64)."""

    def __init__(self, d_model: int = 128, heads: int = 4, n_layers: int = 3) -> None:
        super().__init__()
        self.d_model = d_model
        self.blocks = nn.ModuleList([
            _HilbertWindowBlock(d_model=d_model, heads=heads, window_size=64)
            for _ in range(n_layers)
        ])
        self.cl_origin_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.SiLU(),
            nn.Linear(64, 5),
        )
        self.dark_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    @torch.no_grad()
    def _compute_dual_orders(
        self,
        pix_xyz_mm: torch.Tensor,
        pix_plane: torch.Tensor,
        pix_valid: torch.Tensor,
        apex_mm: torch.Tensor,
        x0_planes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, n, _ = pix_xyz_mm.shape
        device = pix_xyz_mm.device
        if n == 0:
            empty_i = torch.zeros((b, 0), dtype=torch.long, device=device)
            empty_c = torch.zeros((b, 0, 4), dtype=torch.float32, device=device)
            return empty_i, empty_i, empty_c

        dx = pix_xyz_mm[..., 0] - apex_mm[:, 0:1]
        dy = pix_xyz_mm[..., 1] - apex_mm[:, 1:2]
        dz = pix_xyz_mm[..., 2] - apex_mm[:, 2:3]
        pl_safe = pix_plane.clamp(0, 14)
        x0_val = x0_planes[pl_safe]

        # Normalized 4D relative coordinates for attention bias
        dz_pos = dz.clamp_min(15.0)
        coords_norm = torch.stack([
            torch.asinh(dx / 5.0) / 3.0,
            torch.asinh(dy / 5.0) / 3.0,
            dx / dz_pos,
            x0_val / 25.0,
        ], dim=-1) * pix_valid.unsqueeze(-1).float()

        # 1. Intra-plane 2D Hilbert order
        ix_intra = torch.round((dx + 12.8) / 0.10).long().clamp(0, 255)
        iy_intra = torch.round((dy + 12.8) / 0.10).long().clamp(0, 255)
        h_intra = _hilbert_2d_code(ix_intra, iy_intra, bits=8)
        key_intra = torch.where(pix_valid, pl_safe * 65536 + h_intra, torch.full_like(h_intra, 1 << 30))
        order_intra = torch.argsort(key_intra, dim=1, stable=True)

        # 2. Cross-plane projective cone slope Hilbert order (separating Target 0..11 vs IPT 12..14)
        slope_x = dx / dz_pos
        slope_y = dy / dz_pos
        ix_proj = torch.round((slope_x + 0.35) / 0.70 * 255.0).long().clamp(0, 255)
        iy_proj = torch.round((slope_y + 0.35) / 0.70 * 255.0).long().clamp(0, 255)
        h_proj = _hilbert_2d_code(ix_proj, iy_proj, bits=8)
        is_ipt = (pl_safe >= 12).long()
        x0_bin = (x0_val / 3.0).long().clamp(0, 31)
        key_proj = torch.where(
            pix_valid,
            is_ipt * (1 << 24) + h_proj * 32 + x0_bin,
            torch.full_like(h_proj, 1 << 30),
        )
        order_proj = torch.argsort(key_proj, dim=1, stable=True)

        return order_intra, order_proj, coords_norm

    def forward(
        self,
        h_cl: torch.Tensor,
        pix_xyz_mm: torch.Tensor,
        pix_plane: torch.Tensor,
        pix_valid: torch.Tensor,
        pix_feat: torch.Tensor,
        apex_mm: torch.Tensor,
        x0_planes: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        order_intra, order_proj, coords_norm = self._compute_dual_orders(
            pix_xyz_mm, pix_plane, pix_valid, apex_mm, x0_planes
        )
        x = h_cl
        for i, blk in enumerate(self.blocks):
            ord_i = order_intra if (i % 2 == 0) else order_proj
            x = blk(x, coords_norm, pix_valid, ord_i)

        cl_origin_logits = self.cl_origin_head(x)
        dark_logits = self.dark_head(x).squeeze(-1)
        p_orig = torch.softmax(cl_origin_logits.float(), dim=-1)
        g_raw = pix_feat[..., 26].float().clamp(0.05, 1.0)
        g_persist = (0.3 + 0.7 * torch.sigmoid(dark_logits.float())) * g_raw * pix_valid.float()
        s_tilde = g_persist * (1.0 - 0.90 * p_orig[..., 3]) * pix_valid.float()

        return {
            'h_cl': x,
            'cl_origin_logits': cl_origin_logits,
            'dark_logits': dark_logits,
            'g_persist': g_persist,
            's_tilde': s_tilde,
        }


class BackConeSlabPrior(nn.Module):
    """Sandwich scintillator Scint[p0-1] disambiguation of W_back vs W_front (sigma_z = 1.44 mm),
    plus heteroscedastic first-3-station calorimeter prior for Fortune_ScintOnly (D2_obs).
    """

    def __init__(self, geom: Gen4GeometryContract) -> None:
        super().__init__()
        self.register_buffer('pixel_z', torch.as_tensor(geom.pixel_Zpos, dtype=torch.float32))
        self.register_buffer('station_z', torch.as_tensor(geom.station_z_mm, dtype=torch.float32))
        self.register_buffer('station_has_y', torch.as_tensor(geom.station_has_y, dtype=torch.bool))
        self.register_buffer('bar_x', torch.as_tensor(geom.bar_x_centers_mm, dtype=torch.float32))
        self.register_buffer('bar_y', torch.as_tensor(geom.bar_y_centers_mm, dtype=torch.float32))
        self.register_buffer('pin_scint_st', torch.as_tensor(geom.pinpoint_scint_stations, dtype=torch.long))
        self.register_buffer('w_back_z', torch.as_tensor(geom.pinpoint_w_back_z, dtype=torch.float32))
        self.register_buffer('w_front_z', torch.as_tensor(geom.pinpoint_w_front_z, dtype=torch.float32))
        # Use contracted 10085-10088 arrays from Gen4GeometryContract when present
        self.is_mc8588_layout = bool(
            getattr(geom, 'is_mc8588_block_order', bool(geom.pinpoint_w_back_z[0] < geom.pixel_Zpos[0]))
        )
        sand_arr = getattr(geom, 'pinpoint_sand_scint_station', None)
        if sand_arr is not None:
            self.register_buffer('pin_sand_scint', torch.as_tensor(sand_arr, dtype=torch.long))
        else:
            fallback_sand = np.array([-1] + [p for p in range(1, 6)] + [int(geom.pinpoint_scint_stations[p]) for p in range(6, 12)], dtype=np.int64)
            self.register_buffer('pin_sand_scint', torch.as_tensor(fallback_sand, dtype=torch.long))
        up_wf_arr = getattr(geom, 'pinpoint_upstream_w_front_z', None)
        if up_wf_arr is not None:
            up_wf_np = np.asarray(up_wf_arr, dtype=np.float64)
            wb_np = np.asarray(geom.pinpoint_w_back_z, dtype=np.float64)
            up_wf_safe = np.where(np.isfinite(up_wf_np), up_wf_np, wb_np)
            self.register_buffer('pin_up_w_front_z', torch.as_tensor(up_wf_safe, dtype=torch.float32))
        else:
            fallback_up = np.concatenate([[geom.pinpoint_w_back_z[0]], geom.pinpoint_w_front_z[:-1]]).astype(np.float64)
            self.register_buffer('pin_up_w_front_z', torch.as_tensor(fallback_up, dtype=torch.float32))
        fb_id_arr = getattr(geom, 'pinpoint_fortune_block_id', np.array([-1] * 6 + list(range(6)), dtype=np.int64))
        self.register_buffer('pin_fortune_id', torch.as_tensor(fb_id_arr, dtype=torch.long))
        f_mask_arr = getattr(geom, 'fortune_station_mask', None)
        self.has_fortune_mask = f_mask_arr is not None
        if f_mask_arr is not None:
            self.register_buffer('fortune_station_mask', torch.as_tensor(f_mask_arr, dtype=torch.bool))
        self.enable_sandwich_slab = True

        # Learnable threshold logit around 0.65 MeV for sandwich scintillator firing
        self.sand_thr_mev = nn.Parameter(torch.tensor(0.65, dtype=torch.float32))
        self.sand_slope = nn.Parameter(torch.tensor(2.0, dtype=torch.float32))

        # Heteroscedastic calorimeter-only PV scale predictor (centered at ~14 mm XY, ~45 mm Z)
        self.cal_prior_mlp = nn.Sequential(
            nn.Linear(6, 32),
            nn.SiLU(),
            nn.Linear(32, 2),
        )
        nn.init.zeros_(self.cal_prior_mlp[-1].weight)
        nn.init.zeros_(self.cal_prior_mlp[-1].bias)

    def forward(
        self,
        pix_xyz_mm: torch.Tensor,
        pix_plane: torch.Tensor,
        pix_valid: torch.Tensor,
        g_persist: torch.Tensor,
        scint_grid: torch.Tensor,
        cal_start_mm: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        b = pix_xyz_mm.shape[0]
        device = pix_xyz_mm.device

        # Compute persistent cluster count & centroid per target plane 0..11
        tgt_mask = pix_valid & (pix_plane >= 0) & (pix_plane < 12)
        w_cl = g_persist.float() * tgt_mask.float()
        pl_safe = pix_plane.clamp(0, 11)
        pl_oh = F.one_hot(pl_safe, num_classes=12).float() * tgt_mask.unsqueeze(-1).float()
        pl_w = torch.einsum('bnp,bn->bp', pl_oh, w_cl)  # (B, 12)
        pl_x = torch.einsum('bnp,bn->bp', pl_oh, w_cl * pix_xyz_mm[..., 0]) / pl_w.clamp_min(1e-6)
        pl_y = torch.einsum('bnp,bn->bp', pl_oh, w_cl * pix_xyz_mm[..., 1]) / pl_w.clamp_min(1e-6)

        has_plane = pl_w > 0.35
        has_any_si = has_plane.any(dim=1)
        p_grid = torch.arange(12, device=device, dtype=torch.long)[None, :]
        p0 = torch.where(has_plane, p_grid, torch.full_like(p_grid, 12)).min(dim=1).values.clamp(0, 11)

        # First-3-active-station calorimeter centroid refinement for ScintOnly / D2_obs
        ex_bars = scint_grid[:, :, 0, :, 0].float()
        ey_bars = scint_grid[:, :, 1, :, 0].float()
        ex_st = ex_bars.sum(dim=-1)
        ey_st = ey_bars.sum(dim=-1)
        has_y = self.station_has_y[None, :] & (ey_st > 0.1)
        e_st_all = torch.where(has_y, 0.5 * (ex_st + ey_st), ex_st)
        act_st = e_st_all > 0.5
        has_any_cal = act_st.any(dim=1)
        s_idx = torch.arange(e_st_all.shape[1], device=device)[None, :]
        s0_cal = torch.where(act_st, s_idx, torch.full_like(s_idx, 99)).min(dim=1).values.clamp(0, 53)
        win3 = (s_idx >= s0_cal[:, None]) & (s_idx <= s0_cal[:, None] + 2) & act_st
        w3_x = (ex_bars * win3.unsqueeze(-1).float()).sum(dim=1)  # (B, 40)
        w3_y = (ey_bars * (win3 & has_y).unsqueeze(-1).float()).sum(dim=1)
        sx3 = w3_x.sum(dim=-1).clamp_min(1e-5)
        sy3 = w3_y.sum(dim=-1).clamp_min(1e-5)
        x_cal_c = torch.where(has_any_cal & (w3_x.sum(dim=-1) > 0.1), (w3_x * self.bar_x[None, :]).sum(dim=-1) / sx3, cal_start_mm[:, 0].float())
        y_cal_c = torch.where(has_any_cal & (w3_y.sum(dim=-1) > 0.1), (w3_y * self.bar_y[None, :]).sum(dim=-1) / sy3, cal_start_mm[:, 1].float())
        w3_z = e_st_all * win3.float()
        z_cal_c = torch.where(has_any_cal & (w3_z.sum(dim=-1) > 0.1), (w3_z * self.station_z[None, :]).sum(dim=-1) / w3_z.sum(dim=-1).clamp_min(1e-5) - 4.0, cal_start_mm[:, 2].float())

        cal_ctx = torch.stack([
            torch.log1p(e_st_all.sum(dim=-1)) / 8.0,
            torch.log1p(w3_z.sum(dim=-1)) / 6.0,
            win3.float().sum(dim=-1) / 3.0,
            (has_y & win3).float().sum(dim=-1) / 3.0,
            (s0_cal.float() / 53.0),
            has_any_cal.float(),
        ], dim=-1)
        cal_log_sig = self.cal_prior_mlp(cal_ctx).clamp(-2.0, 2.0)
        sig_xy_cal = torch.where(has_any_cal, 14.0 * torch.exp(cal_log_sig[:, 0]), torch.full_like(x_cal_c, 65.0))
        sig_z_cal = torch.where(has_any_cal, 45.0 * torch.exp(cal_log_sig[:, 1]), torch.full_like(x_cal_c, 150.0))

        # Transverse apex from first active silicon plane (or refined 3-station cal centroid)
        idx_b = torch.arange(b, device=device)
        x_p0 = torch.where(has_any_si, pl_x[idx_b, p0], x_cal_c)
        y_p0 = torch.where(has_any_si, pl_y[idx_b, p0], y_cal_c)
        z_p0 = self.pixel_z[p0]

        is_p0_zero = has_any_si & (p0 == 0)
        is_front30 = has_any_si & (p0 >= 1) & (p0 <= 5)
        is_interm = has_any_si & (p0 >= 6)

        if self.is_mc8588_layout:
            # 10085-10088 layout: Scint_V[p0] (-13.505mm) lies between W_front[p0-1] (-22.505mm) and W_back[p0] (-8.505mm)
            s_sand = p0.clamp(0, 11)
            st_sand = self.pin_sand_scint[s_sand].clamp(0, 53)
            z_upstream_plate = self.pin_up_w_front_z[s_sand]
            z_downstream_plate = self.w_back_z[s_sand]
        else:
            # Legacy layout: Scint_V[p0-1] lies between W_back[p0-1] and W_front[p0-1]
            s_sand = (p0 - 1).clamp(0, 11)
            st_sand = self.pin_scint_st[s_sand]
            z_upstream_plate = self.w_back_z[s_sand]
            z_downstream_plate = self.w_front_z[s_sand]

        e_sand_bars = scint_grid[idx_b, st_sand, 0, :, 0].float()  # (B, 40) MeV
        pe_sand_bars = scint_grid[idx_b, st_sand, 0, :, 2].float()  # (B, 40) PE
        bar_near = (torch.abs(self.bar_x[None, :] - x_p0[:, None]) <= 45.0).float()
        e_sand = (e_sand_bars * bar_near).sum(dim=-1)
        pe_sand = (pe_sand_bars * bar_near).max(dim=-1).values

        thr = self.sand_thr_mev.clamp(0.15, 2.5)
        slp = self.sand_slope.clamp(0.5, 8.0)
        p_upstream_plate = torch.sigmoid(slp * (e_sand - thr + 0.25 * (pe_sand >= 2.5).float()))
        if not self.enable_sandwich_slab:
            p_upstream_plate = torch.full_like(p_upstream_plate, 0.5)

        z_f30 = p_upstream_plate * z_upstream_plate + (1.0 - p_upstream_plate) * z_downstream_plate
        sig_z_f30 = (1.44 + 4.0 * p_upstream_plate * (1.0 - p_upstream_plate)) if self.enable_sandwich_slab else torch.full_like(z_f30, 10.0)

        # Interm150 (6 <= p0 <= 11): check FortuneBlock[p0-6] & sandwich Scint_V[p0]
        z_prev = self.pixel_z[(p0 - 1).clamp(0, 11)]
        if self.has_fortune_mask and self.is_mc8588_layout:
            fb_id = self.pin_fortune_id[p0].clamp_min(0)
            in_gap = self.fortune_station_mask[fb_id] & (self.station_z[None, :] < z_p0[:, None])
        else:
            in_gap = (self.station_z[None, :] > z_prev[:, None]) & (self.station_z[None, :] < z_p0[:, None])
        act_gap = in_gap & (e_st_all > 0.6)
        has_gap_act = act_gap.any(dim=1)
        s_first = torch.where(act_gap, s_idx, torch.full_like(s_idx, 99)).min(dim=1).values.clamp(0, 53)
        if self.is_mc8588_layout and self.enable_sandwich_slab:
            # If both FortuneBlock[p0-6] and Scint_V[p0] are silent while Si[p0] fired -> lock to W_back[p0] (-8.505 mm)
            silent_interm_sand = (e_sand <= 0.65) & (~has_gap_act)
            z_interm = torch.where(
                silent_interm_sand,
                z_downstream_plate,
                torch.where(has_gap_act, self.station_z[s_first] - 4.0, 0.5 * (z_prev + z_p0)),
            )
            sig_z_interm = torch.where(
                silent_interm_sand,
                torch.full_like(z_interm, 1.44),
                torch.where(has_gap_act, torch.full_like(z_interm, 6.0), torch.full_like(z_interm, 35.0)),
            )
        else:
            z_interm = torch.where(has_gap_act, self.station_z[s_first] - 4.0, 0.5 * (z_prev + z_p0))
            sig_z_interm = torch.where(has_gap_act, torch.full_like(z_interm, 6.0), torch.full_like(z_interm, 35.0))

        # Assemble initial PV prior (p0==0 in 10085-10088 sits directly downstream of W_back[0] at -8.505mm)
        z_p0_zero = self.w_back_z[0] if self.is_mc8588_layout else (z_p0 - 3.5)
        sig_z_p0_zero = 1.44 if (self.is_mc8588_layout and self.enable_sandwich_slab) else 6.0
        z_init = torch.where(
            is_front30,
            z_f30,
            torch.where(is_interm, z_interm, torch.where(is_p0_zero, z_p0_zero, z_cal_c)),
        )
        sig_z = torch.where(
            is_front30,
            sig_z_f30,
            torch.where(is_interm, sig_z_interm, torch.where(is_p0_zero, torch.full_like(z_init, sig_z_p0_zero), sig_z_cal)),
        )

        v_pv_init = torch.stack([x_p0, y_p0, z_init], dim=-1)
        sig_xy = torch.where(has_any_si, torch.full_like(z_init, 1.5), sig_xy_cal)
        p_pv_prior = torch.diag_embed(torch.stack([
            1.0 / sig_xy.square().clamp_min(1e-4),
            1.0 / sig_xy.square().clamp_min(1e-4),
            1.0 / sig_z.square().clamp_min(1e-4),
        ], dim=-1))

        slab_probs = torch.stack([
            is_front30.float() * p_upstream_plate,
            is_front30.float() * (1.0 - p_upstream_plate),
            is_interm.float(),
            (~(is_front30 | is_interm)).float(),
        ], dim=-1)
        regime_weights = torch.stack([
            (is_front30 | is_p0_zero).float(),
            is_interm.float(),
            (~has_any_si).float(),
        ], dim=-1)

        return {
            'v_pv_init': v_pv_init,
            'p_pv_prior': p_pv_prior,
            'slab_probs': slab_probs,
            'regime_weights': regime_weights,
            'p0': p0,
            'has_any_si': has_any_si,
        }


def _select_plane_cluster_pool(
    pix_xyz_mm: torch.Tensor,
    pix_plane: torch.Tensor,
    pix_valid: torch.Tensor,
    s_tilde: torch.Tensor,
    apex_mm: torch.Tensor,
    k_per_plane: int = 40,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select top K_p clusters per plane (0..14) using anti-EM priority + micro-cone protection."""
    b, n, _ = pix_xyz_mm.shape
    device = pix_xyz_mm.device
    if n == 0:
        return (
            torch.full((b, 15, k_per_plane), -1, dtype=torch.long, device=device),
            torch.zeros((b, 15, k_per_plane), dtype=torch.bool, device=device),
        )

    r_perp = _safe_hypot(pix_xyz_mm[..., 0] - apex_mm[:, 0:1], pix_xyz_mm[..., 1] - apex_mm[:, 1:2])
    # Micro-cone R_perp <= 8 mm gets +10 priority boost so core prongs are never truncated
    priority = s_tilde.float() + torch.where(r_perp <= 8.0, 10.0, 0.0) - 1e-3 * r_perp

    pool_idx = torch.full((b, 15, k_per_plane), -1, dtype=torch.long, device=device)
    pool_mask = torch.zeros((b, 15, k_per_plane), dtype=torch.bool, device=device)
    k_take = min(k_per_plane, n)

    for p in range(15):
        pm = pix_valid & (pix_plane == p)
        score_p = torch.where(pm, priority, torch.full_like(priority, -1.0e9))
        vals, idx = torch.topk(score_p, k=k_take, dim=1, sorted=True)
        ok = vals > -1.0e8
        pool_idx[:, p, :k_take] = torch.where(ok, idx, -1)
        pool_mask[:, p, :k_take] = ok

    return pool_idx, pool_mask


@torch.no_grad()
def build_gbl_and_kink_segments(
    pix_xyz_mm: torch.Tensor,
    pix_plane: torch.Tensor,
    pix_valid: torch.Tensor,
    pix_feat: torch.Tensor,
    s_tilde: torch.Tensor,
    v_pv_init: torch.Tensor,
    muon_kalman_tracks: torch.Tensor,
    muon_kalman_mask: torch.Tensor,
    plane_x0: torch.Tensor,
    has_edep: torch.Tensor | None = None,
    enable_kink_stubs: bool = True,
    k_per_plane: int = 36,
    max_slope: float = 0.24,
) -> dict[str, torch.Tensor]:
    """Vectorized GPU/CPU builder for Type-I GBL (3-hit), Type-II (2-hit), and Type-III 1-Hit Parent-Tau Stubs."""
    b, n, _ = pix_xyz_mm.shape
    device = pix_xyz_mm.device

    seg_state = torch.zeros((b, N_SEG_TOTAL, SEG_STATE_DIM), dtype=torch.float32, device=device)
    seg_mask = torch.zeros((b, N_SEG_TOTAL), dtype=torch.bool, device=device)
    seg_cl_idx = torch.full((b, N_SEG_TOTAL, 3), -1, dtype=torch.long, device=device)
    seg_type = torch.zeros((b, N_SEG_TOTAL), dtype=torch.long, device=device)
    kink_pair_matrix = torch.zeros((b, N_SEG_TOTAL, N_SEG_TOTAL), dtype=torch.bool, device=device)

    if n == 0:
        return {
            'seg_state': seg_state,
            'seg_mask': seg_mask,
            'seg_cl_idx': seg_cl_idx,
            'seg_type': seg_type,
            'kink_pair_matrix': kink_pair_matrix,
        }

    pool_idx, pool_mask = _select_plane_cluster_pool(
        pix_xyz_mm, pix_plane, pix_valid, s_tilde, v_pv_init, k_per_plane=k_per_plane
    )
    safe_idx = pool_idx.clamp_min(0)
    # Gather pool coordinates in FP64 for exact micro-slope / GBL arithmetic
    pxyz_f64 = pix_xyz_mm.double()
    p_x = torch.gather(pxyz_f64[..., 0], 1, safe_idx.view(b, -1)).view(b, 15, k_per_plane)
    p_y = torch.gather(pxyz_f64[..., 1], 1, safe_idx.view(b, -1)).view(b, 15, k_per_plane)
    p_z = torch.gather(pxyz_f64[..., 2], 1, safe_idx.view(b, -1)).view(b, 15, k_per_plane)
    p_s = torch.gather(s_tilde.double(), 1, safe_idx.view(b, -1)).view(b, 15, k_per_plane)
    p_sigxx = torch.gather(pix_feat[..., 27].double().clamp_min(2e-5), 1, safe_idx.view(b, -1)).view(b, 15, k_per_plane)
    p_sigyy = torch.gather(pix_feat[..., 28].double().clamp_min(2e-5), 1, safe_idx.view(b, -1)).view(b, 15, k_per_plane)
    p_nhits = torch.gather(torch.expm1(pix_feat[..., 3].double() * 6.0), 1, safe_idx.view(b, -1)).view(b, 15, k_per_plane)
    if pix_feat.shape[-1] > 30:
        p_qmip = torch.gather(torch.expm1(pix_feat[..., 30].double().clamp(0.0, 6.0)), 1, safe_idx.view(b, -1)).view(b, 15, k_per_plane)
    else:
        p_qmip = torch.ones_like(p_nhits)

    # Pre-collect candidates per regime (Target: 0..11, IPT: 12..14)
    tgt_states: list[torch.Tensor] = []
    tgt_scores: list[torch.Tensor] = []
    tgt_cidx: list[torch.Tensor] = []
    tgt_types: list[torch.Tensor] = []

    ipt_states: list[torch.Tensor] = []
    ipt_scores: list[torch.Tensor] = []
    ipt_cidx: list[torch.Tensor] = []
    ipt_types: list[torch.Tensor] = []

    p_x0_64 = plane_x0.double()

    for pa in range(14):
        is_ipt = pa >= 12
        p_end = 15 if is_ipt else 12
        ma = pool_mask[:, pa]
        if not ma.any():
            continue
        xa, ya, za = p_x[:, pa], p_y[:, pa], p_z[:, pa]
        sa_s, vxa, vya = p_s[:, pa], p_sigxx[:, pa], p_sigyy[:, pa]

        for pb in range(pa + 1, min(p_end, pa + 3)):
            mb = pool_mask[:, pb]
            if not mb.any():
                continue
            xb, yb, zb = p_x[:, pb], p_y[:, pb], p_z[:, pb]
            sb_s, vxb, vyb = p_s[:, pb], p_sigxx[:, pb], p_sigyy[:, pb]
            dz_ab = (zb[:, 0] - za[:, 0]).clamp_min(1e-3)  # (B,)
            dx0_ab = (p_x0_64[pb] - p_x0_64[pa]).clamp_min(IPT_X0_FLOOR)

            # Pairwise slopes (B, K, K)
            tx_ab = (xb[:, None, :] - xa[:, :, None]) / dz_ab[:, None, None]
            ty_ab = (yb[:, None, :] - ya[:, :, None]) / dz_ab[:, None, None]
            sl_ab = torch.hypot(tx_ab, ty_ab)
            valid_ab = ma[:, :, None] & mb[:, None, :] & (sl_ab <= max_slope)
            if pb > pa + 1:
                valid_ab = valid_ab & (dz_ab[:, None, None] <= 70.0)
            if not valid_ab.any():
                continue

            # Keep top K_ab = 96 pairs per batch for 3-hit search
            ab_score = torch.where(
                valid_ab,
                (sa_s[:, :, None] + sb_s[:, None, :]) - 5.0 * (tx_ab.square() + ty_ab.square()),
                torch.full_like(sl_ab, -1.0e9),
            ).view(b, -1)
            k_ab = min(96, ab_score.shape[1])
            ab_val, ab_flat = torch.topk(ab_score, k=k_ab, dim=1)
            ab_ok = ab_val > -1.0e8
            ia_idx = ab_flat // k_per_plane
            ib_idx = ab_flat % k_per_plane

            xa_p = torch.gather(xa, 1, ia_idx)
            ya_p = torch.gather(ya, 1, ia_idx)
            za_p = torch.gather(za, 1, ia_idx)
            vxa_p = torch.gather(vxa, 1, ia_idx)
            vya_p = torch.gather(vya, 1, ia_idx)
            sa_p = torch.gather(sa_s, 1, ia_idx)
            ca_p = torch.gather(pool_idx[:, pa], 1, ia_idx)

            xb_p = torch.gather(xb, 1, ib_idx)
            yb_p = torch.gather(yb, 1, ib_idx)
            zb_p = torch.gather(zb, 1, ib_idx)
            vxb_p = torch.gather(vxb, 1, ib_idx)
            vyb_p = torch.gather(vyb, 1, ib_idx)
            sb_p = torch.gather(sb_s, 1, ib_idx)
            cb_p = torch.gather(pool_idx[:, pb], 1, ib_idx)

            # Type-I: 3-hit GBL
            found_3_for_ab = torch.zeros_like(ab_ok)
            for pc in range(pb + 1, min(p_end, pb + 3)):
                mc = pool_mask[:, pc]
                if not mc.any():
                    continue
                xc, yc, zc = p_x[:, pc], p_y[:, pc], p_z[:, pc]
                sc_s, vxc, vyc = p_s[:, pc], p_sigxx[:, pc], p_sigyy[:, pc]
                dz_bc = (zc[:, 0] - zb[:, 0]).clamp_min(1e-3)
                dz_ac = (zc[:, 0] - za[:, 0]).clamp_min(1e-3)
                if pc > pb + 1 and (dz_bc > 70.0).all():
                    continue

                dx0_bc = (p_x0_64[pc] - p_x0_64[pb]).clamp_min(IPT_X0_FLOOR)
                dx0_ac = (p_x0_64[pc] - p_x0_64[pa]).clamp_min(IPT_X0_FLOOR)

                # Extrapolate straight line a->b to plane c
                tx_ab_p = (xb_p - xa_p) / dz_ab[:, None]
                ty_ab_p = (yb_p - ya_p) / dz_ab[:, None]
                xp_c = xa_p + tx_ab_p * dz_ac[:, None]
                yp_c = ya_p + ty_ab_p * dz_ac[:, None]

                rx_c = xc[:, None, :] - xp_c[:, :, None]  # (B, K_ab, K_p)
                ry_c = yc[:, None, :] - yp_c[:, :, None]
                # Scattering angle at plane b relative to a->b
                dtx = rx_c / dz_bc[:, None, None]
                dty = ry_c / dz_bc[:, None, None]
                th_kink = torch.hypot(dtx, dty)

                # Reference 8 GeV Highland tolerance window
                th_ms_b_ref = (13.6 / 8000.0) * torch.sqrt(dx0_ab.clamp_min(IPT_X0_FLOOR)) * (1.0 + 0.038 * torch.log(dx0_ab.clamp_min(1.0)))
                sig_c2 = (vxa_p[:, :, None] + vxb_p[:, :, None] + vxc[:, None, :]) * 2.0 + (2.5 * th_ms_b_ref * dz_bc[:, None, None]).square()
                chi2_c = (rx_c.square() + ry_c.square()) / sig_c2.clamp_min(1e-5)

                valid_abc = ab_ok[:, :, None] & mc[:, None, :] & (chi2_c <= 22.0) & (th_kink <= 0.18)
                if pc > pb + 1:
                    valid_abc = valid_abc & (dz_bc[:, None, None] <= 70.0)
                if not valid_abc.any():
                    continue

                # Select top K_3 = 48 3-hit triplets per (pa, pb, pc)
                abc_score = torch.where(
                    valid_abc,
                    (sa_p[:, :, None] + sb_p[:, :, None] + sc_s[:, None, :]) - 0.15 * chi2_c,
                    torch.full_like(chi2_c, -1.0e9),
                ).view(b, -1)
                k_3 = min(48, abc_score.shape[1])
                t_val, t_flat = torch.topk(abc_score, k=k_3, dim=1)
                t_ok = t_val > -1.0e8
                if not t_ok.any():
                    continue

                ab_i = t_flat // k_per_plane
                ic_i = t_flat % k_per_plane

                xa_t = torch.gather(xa_p, 1, ab_i)
                ya_t = torch.gather(ya_p, 1, ab_i)
                za_t = torch.gather(za_p, 1, ab_i)
                vxa_t = torch.gather(vxa_p, 1, ab_i)
                vya_t = torch.gather(vya_p, 1, ab_i)
                ca_t = torch.gather(ca_p, 1, ab_i)
                sa_t = torch.gather(sa_p, 1, ab_i)

                xb_t = torch.gather(xb_p, 1, ab_i)
                yb_t = torch.gather(yb_p, 1, ab_i)
                zb_t = torch.gather(zb_p, 1, ab_i)
                vxb_t = torch.gather(vxb_p, 1, ab_i)
                vyb_t = torch.gather(vyb_p, 1, ab_i)
                cb_t = torch.gather(cb_p, 1, ab_i)
                sb_t = torch.gather(sb_p, 1, ab_i)

                xc_t = torch.gather(xc, 1, ic_i)
                yc_t = torch.gather(yc, 1, ic_i)
                zc_t = torch.gather(zc, 1, ic_i)
                vxc_t = torch.gather(vxc, 1, ic_i)
                vyc_t = torch.gather(vyc, 1, ic_i)
                cc_t = torch.gather(pool_idx[:, pc], 1, ic_i)
                sc_t = torch.gather(sc_s, 1, ic_i)

                # Mark (ab) as having a 3-hit match (race-free integer scatter_add_)
                hit_cnt = torch.zeros_like(found_3_for_ab, dtype=torch.int32)
                hit_cnt.scatter_add_(1, ab_i, t_ok.to(torch.int32))
                found_3_for_ab = found_3_for_ab | (hit_cnt > 0)

                # Adaptive momentum proxy from Highland scattering residual
                tx1 = (xb_t - xa_t) / (zb_t - za_t).clamp_min(1e-3)
                ty1 = (yb_t - ya_t) / (zb_t - za_t).clamp_min(1e-3)
                tx2 = (xc_t - xb_t) / (zc_t - zb_t).clamp_min(1e-3)
                ty2 = (yc_t - yb_t) / (zc_t - zb_t).clamp_min(1e-3)
                th_k = torch.hypot(tx2 - tx1, ty2 - ty1)
                p_est_gev = ((0.0136 * torch.sqrt(dx0_ab.clamp_min(IPT_X0_FLOOR))) / th_k.clamp_min(2.5e-4)).clamp(0.25, 250.0)
                if is_ipt and muon_kalman_mask.any():
                    # Match IPT segments to UnifiedMuonKalman momentum via Magnet-0 back-extrapolation
                    mk_p = muon_kalman_tracks[:, :, 2].double().clamp(0.5, 500.0)
                    mk_qp = muon_kalman_tracks[:, :, 0].double()
                    mk_kappa = mk_qp * (KAPPA_B_CONST_MEV_MM / 1000.0)
                    mk_x = muon_kalman_tracks[:, :, 10].double()
                    mk_y = muon_kalman_tracks[:, :, 11].double()
                    mk_tx = muon_kalman_tracks[:, :, 12].double()
                    mk_ty = muon_kalman_tracks[:, :, 13].double()
                    dz_mk = za_t[:, :, None] - float(TRACKER_Z_MM[0])
                    dx_mk = (mk_x[:, None, :] + mk_tx[:, None, :] * dz_mk) - xa_t[:, :, None]
                    y_ext_mk = (
                        mk_y[:, None, :]
                        + mk_ty[:, None, :] * dz_mk
                        - mk_kappa[:, None, :] * float(MAGNET0_LENGTH_MM) * (za_t[:, :, None] - float(MAGNET0_Z_CENTER_MM))
                    )
                    dy_mk = y_ext_mk - ya_t[:, :, None]
                    d_mk = torch.hypot(dx_mk, dy_mk) + (1.0 - muon_kalman_mask[:, None, :].double()) * 1e6
                    min_d, min_j = d_mk.min(dim=-1)
                    mk_p_best = torch.gather(mk_p, 1, min_j)
                    p_est_gev = torch.where(min_d < 8.0, mk_p_best, p_est_gev)

                # Two-scattering-point Generalized Broken Line (GBL) weights at z_a
                th_ab_p = _highland_theta_torch(dx0_ab, p_est_gev).double()
                th_bc_p = _highland_theta_torch(dx0_bc, p_est_gev).double()
                ms_b_at_c = (th_ab_p * (zc_t - zb_t)).square()
                ms_a_at_b = (th_ab_p * (zb_t - za_t)).square()

                wx_a = 1.0 / vxa_t
                wy_a = 1.0 / vya_t
                wx_b = 1.0 / (vxb_t + ms_a_at_b)
                wy_b = 1.0 / (vyb_t + ms_a_at_b)
                wx_c = 1.0 / (vxc_t + ms_a_at_b + ms_b_at_c + (th_bc_p * (zc_t - zb_t)).square())
                wy_c = 1.0 / (vyc_t + ms_a_at_b + ms_b_at_c + (th_bc_p * (zc_t - zb_t)).square())

                dz_b = zb_t - za_t
                dz_c = zc_t - za_t

                # Weighted 2x2 linear fit for (x0, tx) at z_a
                sx0 = wx_a + wx_b + wx_c
                sx1 = wx_b * dz_b + wx_c * dz_c
                sx2 = wx_b * dz_b.square() + wx_c * dz_c.square()
                bx0 = wx_a * xa_t + wx_b * xb_t + wx_c * xc_t
                bx1 = wx_b * dz_b * xb_t + wx_c * dz_c * xc_t
                detx = (sx0 * sx2 - sx1 * sx1).clamp_min(1e-9)
                x0_f = (sx2 * bx0 - sx1 * bx1) / detx
                tx_f = (-sx1 * bx0 + sx0 * bx1) / detx

                sy0 = wy_a + wy_b + wy_c
                sy1 = wy_b * dz_b + wy_c * dz_c
                sy2 = wy_b * dz_b.square() + wy_c * dz_c.square()
                by0 = wy_a * ya_t + wy_b * yb_t + wy_c * yc_t
                by1 = wy_b * dz_b * yb_t + wy_c * dz_c * yc_t
                dety = (sy0 * sy2 - sy1 * sy1).clamp_min(1e-9)
                y0_f = (sy2 * by0 - sy1 * by1) / dety
                ty_f = (-sy1 * by0 + sy0 * by1) / dety

                # Covariances at z_a
                sxx_f = (sx2 / detx).clamp_min(4e-5)
                syy_f = (sy2 / dety).clamp_min(4e-5)
                stx_f = (sx0 / detx).clamp_min(1e-7) + th_ab_p.square() * 0.25
                sty_f = (sy0 / dety).clamp_min(1e-7) + th_ab_p.square() * 0.25

                rx_all = wx_a * (xa_t - x0_f).square() + wx_b * (xb_t - (x0_f + tx_f * dz_b)).square() + wx_c * (xc_t - (x0_f + tx_f * dz_c)).square()
                ry_all = wy_a * (ya_t - y0_f).square() + wy_b * (yb_t - (y0_f + ty_f * dz_b)).square() + wy_c * (yc_t - (y0_f + ty_f * dz_c)).square()
                chi2_f = (rx_all + ry_all).clamp(0.0, 50.0)

                s_mean = (sa_t + sb_t + sc_t) / 3.0
                st_t = torch.stack([
                    x0_f, y0_f, za_t, tx_f, ty_f,
                    sxx_f, syy_f, stx_f, sty_f,
                    chi2_f, dx0_ac.expand_as(chi2_f), dz_c,
                    p_est_gev,
                    torch.full_like(chi2_f, 1.0 if is_ipt else 0.0),
                    torch.full_like(chi2_f, 1.0 / 3.0),
                    s_mean,
                ], dim=-1).float()
                sc_f = torch.where(t_ok, s_mean.float() + 0.5 - 0.05 * chi2_f.float(), torch.full((b, k_3), -1e9, device=device))
                ci_t = torch.stack([ca_t, cb_t, cc_t], dim=-1)
                tp_t = torch.where(t_ok, torch.ones_like(ca_t), torch.zeros_like(ca_t))

                if is_ipt:
                    ipt_states.append(st_t); ipt_scores.append(sc_f); ipt_cidx.append(ci_t); ipt_types.append(tp_t)
                else:
                    tgt_states.append(st_t); tgt_scores.append(sc_f); tgt_cidx.append(ci_t); tgt_types.append(tp_t)

            # Type-II: 2-hit fallback for unmatched consecutive pairs
            if pb == pa + 1:
                unm_2 = ab_ok & (~found_3_for_ab) & (dz_ab[:, None] <= 165.0)
                if unm_2.any():
                    tx_2 = (xb_p - xa_p) / dz_ab[:, None]
                    ty_2 = (yb_p - ya_p) / dz_ab[:, None]
                    unm_2 = unm_2 & (torch.hypot(tx_2, ty_2) <= 0.18)
                    if unm_2.any():
                        k_2 = min(24, k_ab)
                        s2_raw = torch.where(unm_2, 0.5 * (sa_p + sb_p) - 10.0 * (tx_2.square() + ty_2.square()), torch.full_like(tx_2, -1e9))
                        v2, i2 = torch.topk(s2_raw, k=k_2, dim=1)
                        ok2 = v2 > -1e8
                        xa2, ya2, za2 = torch.gather(xa_p, 1, i2), torch.gather(ya_p, 1, i2), torch.gather(za_p, 1, i2)
                        tx2, ty2 = torch.gather(tx_2, 1, i2), torch.gather(ty_2, 1, i2)
                        vxa2, vya2 = torch.gather(vxa_p, 1, i2), torch.gather(vya_p, 1, i2)
                        vxb2, vyb2 = torch.gather(vxb_p, 1, i2), torch.gather(vyb_p, 1, i2)
                        dz2 = (torch.gather(zb_p, 1, i2) - za2).clamp_min(1e-3)
                        th2 = _highland_theta_torch(dx0_ab, torch.full_like(dz2, 8.0)).double()
                        sxx2 = vxa2.clamp_min(4e-5)
                        syy2 = vya2.clamp_min(4e-5)
                        stx2 = (vxa2 + vxb2) / dz2.square() + th2.square()
                        sty2 = (vya2 + vyb2) / dz2.square() + th2.square()
                        s_m2 = 0.5 * (torch.gather(sa_p, 1, i2) + torch.gather(sb_p, 1, i2))
                        st2 = torch.stack([
                            xa2, ya2, za2, tx2, ty2,
                            sxx2, syy2, stx2, sty2,
                            torch.full_like(xa2, 2.5), dx0_ab.expand_as(xa2), dz2,
                            torch.full_like(xa2, 8.0),
                            torch.full_like(xa2, 1.0 if is_ipt else 0.0),
                            torch.full_like(xa2, 2.0 / 3.0),
                            s_m2,
                        ], dim=-1).float()
                        sc2 = torch.where(ok2, s_m2.float() - 0.1, torch.full((b, k_2), -1e9, device=device))
                        ci2 = torch.stack([torch.gather(ca_p, 1, i2), torch.gather(cb_p, 1, i2), torch.full_like(i2, -1)], dim=-1)
                        tp2 = torch.where(ok2, torch.full_like(i2, 2), torch.zeros_like(i2))
                        if is_ipt:
                            ipt_states.append(st2); ipt_scores.append(sc2); ipt_cidx.append(ci2); ipt_types.append(tp2)
                        else:
                            tgt_states.append(st2); tgt_scores.append(sc2); tgt_cidx.append(ci2); tgt_types.append(tp2)

    # Pack top Target segments (leaving room for up to 16 Type-III 1-Hit Parent Stubs!)
    n_tgt_main_cap = N_SEG_TARGET - 16
    if tgt_states:
        all_ts = torch.cat(tgt_states, dim=1)
        all_tsc = torch.cat(tgt_scores, dim=1)
        all_tci = torch.cat(tgt_cidx, dim=1)
        all_ttp = torch.cat(tgt_types, dim=1)
        kt = min(n_tgt_main_cap, all_ts.shape[1])
        vt, it = torch.topk(all_tsc, k=kt, dim=1)
        okt = vt > -1e8
        seg_state[:, :kt] = torch.gather(all_ts, 1, it.unsqueeze(-1).expand(-1, -1, SEG_STATE_DIM))
        seg_mask[:, :kt] = okt
        seg_cl_idx[:, :kt] = torch.gather(all_tci, 1, it.unsqueeze(-1).expand(-1, -1, 3))
        seg_type[:, :kt] = torch.gather(all_ttp, 1, it)

    # Type-III: 1-Hit Parent-Tau Kink Stubs (anchor PV_init -> 1-MIP cluster on p_u -> daughter segment on p_u + 1)
    # Search target planes 0..10; enforce 0.55 <= Q_MIP <= 1.65 when Mode-B pixel edep is active
    if enable_kink_stubs and seg_mask[:, :n_tgt_main_cap].any():
        dau_st = seg_state[:, :n_tgt_main_cap].double()
        dau_m = seg_mask[:, :n_tgt_main_cap]
        dau_cl_a = seg_cl_idx[:, :n_tgt_main_cap, 0].clamp_min(0)
        dau_pa = torch.gather(pix_plane, 1, dau_cl_a)  # (B, n_tgt_main)

        stub_list_s: list[torch.Tensor] = []
        stub_list_sc: list[torch.Tensor] = []
        stub_list_ci: list[torch.Tensor] = []
        stub_list_dau: list[torch.Tensor] = []

        use_edep_gate = (has_edep.view(b, 1) > 0.5) if has_edep is not None else torch.zeros((b, 1), dtype=torch.bool, device=device)
        pv_d = v_pv_init.double()
        for pu in range(11):
            q_ok_u = (~use_edep_gate) | ((p_qmip[:, pu] >= 0.55) & (p_qmip[:, pu] <= 1.65))
            mp_u = pool_mask[:, pu] & (p_nhits[:, pu] <= 5.5) & (p_s[:, pu] > 0.25) & q_ok_u
            if not mp_u.any():
                continue
            xu, yu, zu = p_x[:, pu], p_y[:, pu], p_z[:, pu]
            # Daughter segments starting at pu + 1
            is_dau = dau_m & (dau_pa == (pu + 1))
            if not is_dau.any():
                continue
            # Extrapolate daughter segment backward to zu
            dz_ext = zu[:, 0][:, None] - dau_st[:, :, 2]  # (B, D)
            xd_u = dau_st[:, :, 0] + dau_st[:, :, 3] * dz_ext
            yd_u = dau_st[:, :, 1] + dau_st[:, :, 4] * dz_ext

            r_kink = torch.hypot(xu[:, :, None] - xd_u[:, None, :], yu[:, :, None] - yd_u[:, None, :])  # (B, K_p, D)
            dz_pv_u = (zu - pv_d[:, 2:3]).clamp_min(3.0)
            tx_par = (xu - pv_d[:, 0:1]) / dz_pv_u
            ty_par = (yu - pv_d[:, 1:2]) / dz_pv_u
            sl_par = torch.hypot(tx_par, ty_par)
            th_kink = torch.hypot(tx_par[:, :, None] - dau_st[:, None, :, 3], ty_par[:, :, None] - dau_st[:, None, :, 4])

            valid_k = mp_u[:, :, None] & is_dau[:, None, :] & (r_kink <= 0.45) & (sl_par[:, :, None] <= 0.22) & (th_kink >= 0.002) & (th_kink <= 0.16)
            if not valid_k.any():
                continue
            # Best daughter per cluster u
            sc_k = torch.where(valid_k, p_s[:, pu][:, :, None] + th_kink * 20.0 - r_kink * 5.0, torch.full_like(r_kink, -1e9))
            best_sc, best_d = sc_k.max(dim=-1)  # (B, K_p)
            ok_u = best_sc > -1e8
            dx0_u = (p_x0_64[pu] - p_x0_64[max(0, pu - 1)]).clamp_min(0.5)
            vxu, vyu = p_sigxx[:, pu].clamp_min(4e-5), p_sigyy[:, pu].clamp_min(4e-5)
            st_u = torch.stack([
                xu, yu, zu, tx_par, ty_par,
                vxu, vyu, vxu / dz_pv_u.square() + 1e-5, vyu / dz_pv_u.square() + 1e-5,
                torch.full_like(xu, 1.5), dx0_u.expand_as(xu), dz_pv_u,
                torch.full_like(xu, 15.0),
                torch.zeros_like(xu),
                torch.ones_like(xu),  # seg_type_norm = 1.0 (Type-III 1-hit stub)
                p_s[:, pu],
            ], dim=-1).float()
            ci_u = torch.stack([pool_idx[:, pu], torch.full_like(pool_idx[:, pu], -1), torch.full_like(pool_idx[:, pu], -1)], dim=-1)
            stub_list_s.append(st_u)
            stub_list_sc.append(torch.where(ok_u, best_sc.float(), torch.full((b, k_per_plane), -1e9, device=device)))
            stub_list_ci.append(ci_u)
            stub_list_dau.append(best_d)

        if stub_list_s:
            all_ss = torch.cat(stub_list_s, dim=1)
            all_ssc = torch.cat(stub_list_sc, dim=1)
            all_sci = torch.cat(stub_list_ci, dim=1)
            all_sdau = torch.cat(stub_list_dau, dim=1)
            ks = min(16, all_ss.shape[1])
            vs, is_idx = torch.topk(all_ssc, k=ks, dim=1)
            oks = vs > -1e8
            s_slot = n_tgt_main_cap
            seg_state[:, s_slot : s_slot + ks] = torch.gather(all_ss, 1, is_idx.unsqueeze(-1).expand(-1, -1, SEG_STATE_DIM))
            seg_mask[:, s_slot : s_slot + ks] = oks
            seg_cl_idx[:, s_slot : s_slot + ks] = torch.gather(all_sci, 1, is_idx.unsqueeze(-1).expand(-1, -1, 3))
            seg_type[:, s_slot : s_slot + ks] = torch.where(oks, 3, 0)
            # Record parent-daughter kink pairs
            dau_matched = torch.gather(all_sdau, 1, is_idx)  # (B, ks)
            b_idx = torch.arange(b, device=device)[:, None].expand(b, ks)
            par_slots = torch.arange(s_slot, s_slot + ks, device=device)[None, :].expand(b, ks)
            kink_pair_matrix[b_idx[oks], par_slots[oks], dau_matched[oks]] = True
            kink_pair_matrix[b_idx[oks], dau_matched[oks], par_slots[oks]] = True

    # Pack top IPT segments into [N_SEG_TARGET : N_SEG_TOTAL]
    if ipt_states:
        all_is = torch.cat(ipt_states, dim=1)
        all_isc = torch.cat(ipt_scores, dim=1)
        all_ici = torch.cat(ipt_cidx, dim=1)
        all_itp = torch.cat(ipt_types, dim=1)
        ki = min(N_SEG_IPT, all_is.shape[1])
        vi, ii = torch.topk(all_isc, k=ki, dim=1)
        oki = vi > -1e8
        seg_state[:, N_SEG_TARGET : N_SEG_TARGET + ki] = torch.gather(all_is, 1, ii.unsqueeze(-1).expand(-1, -1, SEG_STATE_DIM))
        seg_mask[:, N_SEG_TARGET : N_SEG_TARGET + ki] = oki
        seg_cl_idx[:, N_SEG_TARGET : N_SEG_TARGET + ki] = torch.gather(all_ici, 1, ii.unsqueeze(-1).expand(-1, -1, 3))
        seg_type[:, N_SEG_TARGET : N_SEG_TARGET + ki] = torch.gather(all_itp, 1, ii)

    return {
        'seg_state': seg_state,
        'seg_mask': seg_mask,
        'seg_cl_idx': seg_cl_idx,
        'seg_type': seg_type,
        'kink_pair_matrix': kink_pair_matrix,
    }


@torch.no_grad()
def compute_pairwise_ray_geometry(
    seg_state: torch.Tensor,
    seg_mask: torch.Tensor,
    kink_pair_matrix: torch.Tensor,
    geom: Gen4GeometryContract,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FP64 PV-free skew-DCA, 6 mm air-gap material prior, p_T_kink & GTA SO(3) relative pose."""
    b, m, _ = seg_state.shape
    device = seg_state.device
    if m == 0:
        return (
            torch.zeros((b, 0, 0, PAIR_R_DIM), dtype=torch.float32, device=device),
            torch.zeros((b, 0, 3, 3), dtype=torch.float32, device=device),
            torch.zeros((b, 0, 0, 3), dtype=torch.float32, device=device),
        )

    st = seg_state.double()
    x0, y0, z0 = st[..., 0], st[..., 1], st[..., 2]
    tx, ty = st[..., 3], st[..., 4]
    sxx, syy = st[..., 5].clamp_min(2e-5), st[..., 6].clamp_min(2e-5)
    stx, sty = st[..., 7].clamp_min(1e-7), st[..., 8].clamp_min(1e-7)
    p_hat = st[..., 12].clamp(0.25, 300.0)

    # Local ray frame E_i = [e1, e2, u] in SO(3) (strictly PV-free, Y-mirror equivariant)
    norm_u = torch.sqrt(1.0 + tx.square() + ty.square())
    ux, uy, uz = tx / norm_u, ty / norm_u, 1.0 / norm_u
    u_vec = torch.stack([ux, uy, uz], dim=-1)  # (B, M, 3)
    # e1 orthogonal to u along lab X: [1 - ux^2, -ux*uy, -ux*uz] normalized
    e1_raw = torch.stack([1.0 - ux * ux, -ux * uy, -ux * uz], dim=-1)
    e1 = e1_raw / e1_raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    e2 = torch.cross(u_vec, e1, dim=-1)
    E_frame = torch.stack([e1, e2, u_vec], dim=-1)  # (B, M, 3, 3)

    # Relative frame transforms: R_ij = E_i^T E_j, dr_ij = E_i^T (r_j - r_i)
    r0 = torch.stack([x0, y0, z0], dim=-1)
    dr = r0[:, None, :, :] - r0[:, :, None, :]  # (B, M, M, 3)
    rel_pos = torch.einsum('bikp,bijk->bijp', E_frame, dr).float()  # E_i^T (r_j - r_i)
    rel_rot = torch.einsum('bikp,bjkq->bijpq', E_frame, E_frame)  # (B, M, M, 3, 3)

    # FP64 closest approach between rays i and j
    dx = x0[:, None, :] - x0[:, :, None]
    dy = y0[:, None, :] - y0[:, :, None]
    dz = z0[:, None, :] - z0[:, :, None]

    # Ray directions v_i = (tx_i, ty_i, 1)
    # Minimize ||(r_i + s_i v_i) - (r_j + s_j v_j)||^2
    vi_vj = tx[:, :, None] * tx[:, None, :] + ty[:, :, None] * ty[:, None, :] + 1.0
    vi_2 = tx.square() + ty.square() + 1.0
    vj_2 = vi_2
    denom = (vi_2[:, :, None] * vj_2[:, None, :] - vi_vj.square()).clamp_min(1.0e-10)

    vi_dr = tx[:, :, None] * dx + ty[:, :, None] * dy + dz
    vj_dr = tx[:, None, :] * dx + ty[:, None, :] * dy + dz

    s_i = ((vi_dr * vj_2[:, None, :] - vj_dr * vi_vj) / denom).clamp(-350.0, 350.0)
    s_j = ((vi_dr * vi_vj - vj_dr * vi_2[:, :, None]) / denom).clamp(-350.0, 350.0)

    pi_x = x0[:, :, None] + tx[:, :, None] * s_i
    pi_y = y0[:, :, None] + ty[:, :, None] * s_i
    pi_z = z0[:, :, None] + s_i
    pj_x = x0[:, None, :] + tx[:, None, :] * s_j
    pj_y = y0[:, None, :] + ty[:, None, :] * s_j
    pj_z = z0[:, None, :] + s_j

    d_doca = torch.sqrt((pi_x - pj_x).square() + (pi_y - pj_y).square() + (pi_z - pj_z).square() + 1e-12)
    z_ca = 0.5 * (pi_z + pj_z)
    var_i = sxx[:, :, None] + s_i.square() * stx[:, :, None] + syy[:, :, None] + s_i.square() * sty[:, :, None]
    var_j = sxx[:, None, :] + s_j.square() * stx[:, None, :] + syy[:, None, :] + s_j.square() * sty[:, None, :]
    s_doca = d_doca / torch.sqrt((var_i + var_j).clamp_min(1e-5))

    # Opening angle & momentum-invariant p_T_kink
    cos_th = (vi_vj / torch.sqrt(vi_2[:, :, None] * vj_2[:, None, :])).clamp(-1.0, 1.0)
    sin_th = torch.sqrt((1.0 - cos_th.square()).clamp_min(1e-12))
    th_open = torch.acos(cos_th.clamp(-0.9999999, 0.9999999))
    p_min = torch.minimum(p_hat[:, :, None], p_hat[:, None, :])
    p_t_kink = p_min * sin_th  # GeV (MCS floor is ~0.0234 GeV in Front30; tau kink is 0.25-0.89 GeV!)
    m_ee = torch.sqrt((2.0 * p_hat[:, :, None] * p_hat[:, None, :] * (1.0 - cos_th) + 2.0 * 0.000511 ** 2).clamp_min(0.0))
    m_pipi = torch.sqrt((2.0 * 0.13957 ** 2 + 2.0 * p_hat[:, :, None] * p_hat[:, None, :] * (1.0 - cos_th)).clamp_min(0.0))

    # Material conversion prior at z_ca: 1/87000 inside 6mm air gaps vs 1.0 inside tungsten!
    z_ca_np = z_ca.cpu().numpy()
    p_conv_mat = torch.from_numpy(geom.material_conversion_prior(z_ca_np)).to(device=device, dtype=torch.float64)
    in_air_gap = (p_conv_mat <= 2.0 * AIR_CONV_SUPPRESSION).double()

    pair_valid = (seg_mask[:, :, None] & seg_mask[:, None, :]).double()
    r_ij = torch.stack([
        torch.asinh(d_doca / 0.1) / 4.0,
        torch.log1p(s_doca) / 4.0,
        torch.asinh(s_i / 30.0) / 3.0,
        torch.asinh(s_j / 30.0) / 3.0,
        torch.log1p(th_open * 100.0) / 4.0,
        torch.asinh(p_t_kink / 0.05) / 3.0,
        torch.log1p(m_ee / 0.02) / 4.0,
        torch.log1p(m_pipi / 0.5) / 3.0,
        torch.log10(p_conv_mat.clamp_min(1e-6)) / 5.0,
        in_air_gap,
        (z_ca - z0[:, :, None]) / 100.0,
        (z_ca - z0[:, None, :]) / 100.0,
        kink_pair_matrix.double(),
        rel_pos[..., 0].double() / 20.0,
        rel_pos[..., 1].double() / 20.0,
        rel_pos[..., 2].double() / 100.0,
        rel_rot[..., 0, 0],
        rel_rot[..., 1, 1],
        rel_rot[..., 2, 2],
        (rel_rot[..., 0, 1] - rel_rot[..., 1, 0]) * 0.5,  # parity-odd twist
    ], dim=-1).float() * pair_valid.unsqueeze(-1).float()

    return r_ij, E_frame.float(), rel_pos


class _GTALayer(nn.Module):
    """PV-free Geometric Transformer Attention (GTA) with pair-edge bias & background registers."""

    def __init__(self, d_node: int = 160, d_pair: int = 48, heads: int = 8) -> None:
        super().__init__()
        self.d_node = d_node
        self.heads = heads
        self.head_dim = d_node // heads

        self.norm1 = nn.LayerNorm(d_node)
        self.qkv = nn.Linear(d_node, 3 * d_node, bias=False)
        self.pair_to_bias = nn.Linear(d_pair, heads, bias=False)
        self.proj = nn.Linear(d_node, d_node)

        self.norm2 = nn.LayerNorm(d_node)
        self.ffn = nn.Sequential(
            nn.Linear(d_node, 2 * d_node),
            nn.SiLU(),
            nn.Linear(2 * d_node, d_node),
        )

    def forward(
        self,
        s_all: torch.Tensor,
        mask_all: torch.Tensor,
        z_pair: torch.Tensor,
        rel_pos: torch.Tensor,
    ) -> torch.Tensor:
        b, n_tot, d = s_all.shape
        m_seg = z_pair.shape[1]
        x = self.norm1(s_all)
        qkv = self.qkv(x).view(b, n_tot, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.permute(0, 2, 1, 3)  # (B, H, N, hd)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        # Inject Pairmixer geometric closure bias onto the segment-segment block
        p_bias = self.pair_to_bias(z_pair).permute(0, 3, 1, 2)  # (B, H, M, M)
        attn[:, :, :m_seg, :m_seg] = attn[:, :, :m_seg, :m_seg] + p_bias

        attn = attn.masked_fill(~mask_all[:, None, None, :], -1.0e4)
        probs = torch.softmax(attn, dim=-1) * mask_all[:, None, None, :].float()
        out = torch.matmul(probs, v).permute(0, 2, 1, 3).reshape(b, n_tot, d)

        s_all = s_all + self.proj(out)
        s_all = s_all + self.ffn(self.norm2(s_all))
        return s_all * mask_all.unsqueeze(-1).float()


class _PairmixerLayer(nn.Module):
    """Attention-free gated triangle multiplicative update on pairwise tensor Z (B, M, M, d_z)."""

    def __init__(self, d_pair: int = 48, d_node: int = 160) -> None:
        super().__init__()
        self.norm_z = nn.LayerNorm(d_pair)
        self.phi1 = nn.Linear(d_pair, d_pair)
        self.g1 = nn.Linear(d_pair, d_pair)
        self.phi2 = nn.Linear(d_pair, d_pair)
        self.g2 = nn.Linear(d_pair, d_pair)
        self.norm_tri = nn.LayerNorm(d_pair)
        self.out_proj = nn.Linear(d_pair, d_pair)
        self.node_inject = nn.Linear(2 * d_node, d_pair)

    def forward(
        self,
        z_pair: torch.Tensor,
        s_seg: torch.Tensor,
        seg_mask: torch.Tensor,
    ) -> torch.Tensor:
        z_n = self.norm_z(z_pair)
        a = torch.sigmoid(self.g1(z_n)) * self.phi1(z_n)
        b_t = torch.sigmoid(self.g2(z_n)) * self.phi2(z_n)

        m_f = seg_mask.float()
        a = a * (m_f[:, :, None, None] * m_f[:, None, :, None])
        b_t = b_t * (m_f[:, :, None, None] * m_f[:, None, :, None])
        n_act = m_f.sum(dim=1).clamp_min(1.0).view(-1, 1, 1, 1)

        # Outgoing triangle multiplicative closure: sum_k a_ik * b_jk
        tri = torch.einsum('bikd,bjkd->bijd', a, b_t) / torch.sqrt(n_act)
        # Symmetrize & add node context
        tri = 0.5 * (tri + tri.transpose(1, 2))
        node_ctx = self.node_inject(torch.cat([
            s_seg[:, :, None, :] + s_seg[:, None, :, :],
            s_seg[:, :, None, :] * s_seg[:, None, :, :],
        ], dim=-1))
        z_out = z_pair + self.out_proj(self.norm_tri(tri)) + 0.1 * node_ctx
        return z_out * (m_f[:, :, None, None] * m_f[:, None, :, None])


class CompetitiveSlotNDIVE(nn.Module):
    """8-slot competitive implicit-differentiation vertexing layer (Comp-NDIVE)."""

    def __init__(
        self,
        geom: Gen4GeometryContract,
        d_node: int = 160,
        d_pair: int = 48,
        enable_comp_ndive: bool = True,
    ) -> None:
        super().__init__()
        self.geom = geom
        self.nu = 3.0
        self.n_no_grad_iters = 4
        self.enable_comp_ndive = bool(enable_comp_ndive)

        self.slot_queries = nn.Parameter(torch.randn(1, N_COMP_SLOTS, d_node) * 0.02)
        self.q_norm = nn.LayerNorm(d_node)
        self.k_norm = nn.LayerNorm(d_node)
        self.v_norm = nn.LayerNorm(d_node)
        self.q_proj = nn.Linear(d_node, d_node)
        self.k_proj = nn.Linear(d_node, d_node)
        self.v_proj = nn.Linear(d_node, d_node)
        self.slot_ffn = nn.Sequential(
            nn.LayerNorm(d_node),
            nn.Linear(d_node, 2 * d_node),
            nn.SiLU(),
            nn.Linear(2 * d_node, d_node),
        )
        self.router_head = nn.Linear(d_node, d_node)
        self.seg_router_proj = nn.Linear(d_node, d_node)
        self.pair_v0_bias = nn.Linear(d_pair, 1)

        # Per-domain calibrated PV scales {0: Front30, 1: Interm150, 2: ScintOnly} x {x, y, z}
        self.pv_scale_cal = nn.Parameter(torch.zeros((3, 3), dtype=torch.float32))
        self.sv_dz_prior = nn.Parameter(torch.tensor([5.0, 5.0], dtype=torch.float32))

    def _evaluate_slot_energy_and_normal_eq(
        self,
        v_cur: torch.Tensor,        # (B, 7, 3)
        mu_prior: torch.Tensor,     # (B, 7, 3)
        p_prior: torch.Tensor,      # (B, 7, 3, 3)
        x0: torch.Tensor,           # (B, M)
        y0: torch.Tensor,           # (B, M)
        z0: torch.Tensor,           # (B, M)
        tx: torch.Tensor,           # (B, M)
        ty: torch.Tensor,           # (B, M)
        sxx: torch.Tensor,          # (B, M)
        syy: torch.Tensor,          # (B, M)
        stx: torch.Tensor,          # (B, M)
        sty: torch.Tensor,          # (B, M)
        p_hat: torch.Tensor,        # (B, M)
        is_stub: torch.Tensor,      # (B, M) bool
        p_ik: torch.Tensor,         # (B, M, 7)
        x0_z0: torch.Tensor,        # (B, M)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute exact Hessian H_k (B, 7, 3, 3) and gradient g_k (B, 7, 3) at v_cur."""
        b, m = x0.shape
        vz = v_cur[:, :, 2:3]  # (B, 7, 1) -> broadcast with (B, 1, M) -> (B, 7, M)
        dz = (vz - z0[:, None, :]).clamp(-350.0, 350.0)

        # Material-adaptive cumulative X0 between segment anchor z0 and vertex candidate z_k
        x0_vk = self.geom.interp_x0_torch(v_cur[:, :, 2]).unsqueeze(-1)  # (B, 7, 1)
        dx0_ik = torch.abs(x0_z0[:, None, :] - x0_vk).clamp_min(IPT_X0_FLOOR)
        th_ms = _highland_theta_torch(dx0_ik, p_hat[:, None, :])
        ms_pos2 = (1.0 / 3.0) * dz.square() * th_ms.square()

        var_x = (sxx[:, None, :] + dz.square() * stx[:, None, :] + ms_pos2).clamp_min(4e-5)
        var_y = (syy[:, None, :] + dz.square() * sty[:, None, :] + ms_pos2).clamp_min(4e-5)

        rx = (x0[:, None, :] + tx[:, None, :] * dz) - v_cur[:, :, 0:1]
        ry = (y0[:, None, :] + ty[:, None, :] * dz) - v_cur[:, :, 1:2]
        chi2 = (rx.square() / var_x + ry.square() / var_y).clamp_max(1.0e4)

        # Causal gate: SV slots (k=1,2) prefer segments starting downstream unless Type-III 1-hit parent stub
        caus = torch.ones_like(chi2)
        sv_caus = torch.where(
            is_stub[:, None, :],
            torch.ones_like(dz[:, 1:3, :]),
            torch.sigmoid((z0[:, None, :] - vz[:, 1:3, :] + 1.5) / 1.5),
        )
        caus[:, 1:3, :] = sv_caus

        p_slot = p_ik.permute(0, 2, 1)  # (B, 7, M)
        w_rob = p_slot * caus * (self.nu + 2.0) / (self.nu + chi2)
        wx = w_rob / var_x
        wy = w_rob / var_y

        # Gradient g_k = P_{k,0}(v_k - mu_k) - sum_i w_i J_i^T r_i
        # Here r_x = x0 + tx*(z_k - z0) - x_k; d(r_x)/d(x_k) = -1, d(r_x)/d(z_k) = tx
        g_prior = torch.einsum('bkij,bkj->bki', p_prior, v_cur - mu_prior)
        gx = g_prior[..., 0] - (wx * rx).sum(dim=-1)
        gy = g_prior[..., 1] - (wy * ry).sum(dim=-1)
        gz = g_prior[..., 2] + (wx * tx[:, None, :] * rx + wy * ty[:, None, :] * ry).sum(dim=-1)
        grad = torch.stack([gx, gy, gz], dim=-1)

        # Gauss-Newton / Fisher Hessian H_k + minimal (250 mm)^-2 LM floor (never overrides D2_obs prior)
        lm_floor = 1.6e-5
        h00 = p_prior[..., 0, 0] + wx.sum(dim=-1) + lm_floor
        h11 = p_prior[..., 1, 1] + wy.sum(dim=-1) + lm_floor
        h02 = p_prior[..., 0, 2] - (wx * tx[:, None, :]).sum(dim=-1)
        h12 = p_prior[..., 1, 2] - (wy * ty[:, None, :]).sum(dim=-1)
        h22 = p_prior[..., 2, 2] + (wx * tx[:, None, :].square() + wy * ty[:, None, :].square()).sum(dim=-1) + lm_floor
        h01 = p_prior[..., 0, 1]

        hess = torch.stack([
            torch.stack([h00, h01, h02], dim=-1),
            torch.stack([h01, h11, h12], dim=-1),
            torch.stack([h02, h12, h22], dim=-1),
        ], dim=-2)
        return hess, grad

    def forward(
        self,
        s_seg: torch.Tensor,
        seg_state: torch.Tensor,
        seg_mask: torch.Tensor,
        seg_type: torch.Tensor,
        z_pair: torch.Tensor,
        r_ij: torch.Tensor,
        v_pv_init: torch.Tensor,
        p_pv_prior: torch.Tensor,
        regime_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        b, m, d = s_seg.shape
        device = s_seg.device
        has_any_seg = seg_mask.any(dim=1)
        safe_seg_mask = torch.where(
            has_any_seg.unsqueeze(-1),
            seg_mask,
            F.one_hot(torch.zeros((b,), dtype=torch.long, device=device), num_classes=m).bool(),
        )

        # 1. Competitive 8-slot router
        q = self.slot_queries.expand(b, -1, -1)
        qn = self.q_norm(q)
        kn = self.k_norm(s_seg)
        vn = self.v_norm(s_seg)
        attn_q = torch.einsum('bkd,bmd->bkm', self.q_proj(qn), self.k_proj(kn)) / math.sqrt(d)
        attn_q = attn_q.masked_fill(~safe_seg_mask[:, None, :], -1.0e4)
        w_q = torch.softmax(attn_q, dim=-1) * seg_mask[:, None, :].float()
        q = q + torch.einsum('bkm,bmd->bkd', w_q, self.v_proj(vn))
        q = q + self.slot_ffn(q)

        # Dot-product routing logits (B, M, 8) + physical slot biases
        seg_r = self.seg_router_proj(s_seg)
        q_r = self.router_head(q)
        slot_logits = torch.einsum('bmd,bkd->bmk', seg_r, q_r) / math.sqrt(d)

        # Physics prior biases: V0 slots get bias toward collinear/conversion-rich pairs
        v0_pair_affinity = self.pair_v0_bias(z_pair).squeeze(-1).mean(dim=-1)  # (B, M)
        slot_logits[:, :, 3:7] = slot_logits[:, :, 3:7] + 0.25 * v0_pair_affinity.unsqueeze(-1)
        # Type-III 1-hit stub gets slight bias away from V0/ghost toward PV/tau-SV
        is_stub = seg_type == 3
        slot_logits[:, :, 1] = slot_logits[:, :, 1] + 0.5 * is_stub.float()

        # Strictly normalized competitive assignment sum_{k=0}^7 P_{i,k} == 1
        seg_slot_probs = torch.softmax(slot_logits.float(), dim=-1)
        p_fit = torch.where(
            seg_mask.unsqueeze(-1),
            seg_slot_probs[:, :, :7],
            torch.zeros((b, m, 7), dtype=torch.float32, device=device),
        )  # slots 0..6

        # 2. Comp-NDIVE Implicit Differential Fit in FP32
        with torch.amp.autocast(device_type=device.type, enabled=False):
            st = seg_state.float()
            x0, y0, z0 = st[..., 0], st[..., 1], st[..., 2]
            tx, ty = st[..., 3], st[..., 4]
            sxx = st[..., 5].clamp_min(4e-5)
            syy = st[..., 6].clamp_min(4e-5)
            stx = st[..., 7].clamp_min(1e-7)
            sty = st[..., 8].clamp_min(1e-7)
            p_hat = st[..., 12].clamp(0.25, 300.0)
            x0_z0 = self.geom.interp_x0_torch(z0)

            # Build 7-slot prior means & precisions
            mu_prior = torch.zeros((b, 7, 3), dtype=torch.float32, device=device)
            p_prior = torch.zeros((b, 7, 3, 3), dtype=torch.float32, device=device)

            mu_prior[:, 0] = v_pv_init.float()
            p_prior[:, 0] = p_pv_prior.float()

            dz_sv = self.sv_dz_prior.clamp(1.0, 40.0)
            mu_prior[:, 1] = v_pv_init.float() + torch.tensor([0.0, 0.0, float(dz_sv[0].item())], device=device)
            mu_prior[:, 2] = v_pv_init.float() + torch.tensor([0.0, 0.0, float(dz_sv[1].item())], device=device)
            p_prior[:, 1] = torch.diag(torch.tensor([0.02, 0.02, 0.004], device=device))
            p_prior[:, 2] = torch.diag(torch.tensor([0.02, 0.02, 0.004], device=device))

            # V0 slots initialized at weighted anchor z0 + weak transverse prior
            for kv0 in range(3, 7):
                w_k = p_fit[:, :, kv0] + 1e-4 * seg_mask.float()
                w_sum = w_k.sum(dim=1, keepdim=True).clamp_min(1e-4)
                z_k = (w_k * z0).sum(dim=1, keepdim=True) / w_sum
                x_k = (w_k * x0).sum(dim=1, keepdim=True) / w_sum
                y_k = (w_k * y0).sum(dim=1, keepdim=True) / w_sum
                mu_prior[:, kv0] = torch.cat([x_k, y_k, z_k], dim=-1)
                p_prior[:, kv0] = torch.diag(torch.tensor([0.01, 0.01, 0.002], device=device))

            # 4 no_grad IRLS steps + 1 implicit-diff step (or prior fallback when enable_comp_ndive=False)
            eye3 = torch.eye(3, dtype=torch.float32, device=device)[None, None, :, :]
            _, cov_prior = _solve_spd_3x3(p_prior + 1.6e-5 * eye3, mu_prior)
            if self.enable_comp_ndive:
                with torch.no_grad():
                    v_cur = mu_prior.clone()
                    for _ in range(self.n_no_grad_iters):
                        hess_ng, grad_ng = self._evaluate_slot_energy_and_normal_eq(
                            v_cur, mu_prior, p_prior,
                            x0, y0, z0, tx, ty, sxx, syy, stx, sty, p_hat, is_stub, p_fit, x0_z0,
                        )
                        delta_v, _ = _solve_spd_3x3(hess_ng, grad_ng)
                        v_cur = v_cur - delta_v.clamp(-30.0, 30.0)

                # 1 exact implicit-differentiation Hessian step WITH autograd
                hess_f, grad_f = self._evaluate_slot_energy_and_normal_eq(
                    v_cur, mu_prior, p_prior,
                    x0, y0, z0, tx, ty, sxx, syy, stx, sty, p_hat, is_stub, p_fit, x0_z0,
                )
                delta_f, cov_raw = _solve_spd_3x3(hess_f, grad_f)
                v_slots_fit = v_cur - delta_f.clamp(-30.0, 30.0)

                has_seg_3 = has_any_seg.view(b, 1, 1)
                has_seg_4 = has_any_seg.view(b, 1, 1, 1)
                v_slots = torch.where(has_seg_3, v_slots_fit, mu_prior)
                cov_slots = torch.where(has_seg_4, cov_raw, cov_prior)
            else:
                v_slots = mu_prior.clone()
                cov_slots = cov_prior.clone()

            # Per-domain calibrated full 3x3 covariance & SPD clamping
            if regime_weights is None:
                rw = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32, device=device).expand(b, 3)
            else:
                rw = regime_weights.float()
            cal_sc = torch.exp((rw @ self.pv_scale_cal).clamp(-2.0, 2.5))  # (B, 3)
            cal_mat = cal_sc[:, :, None] * cal_sc[:, None, :]
            cov_pv_scaled = cov_slots[:, 0] * cal_mat
            cov_slots = torch.cat([cov_pv_scaled.unsqueeze(1), cov_slots[:, 1:]], dim=1)
            cov_slots = _clamp_spd_cov3(cov_slots, min_std_mm=0.015, max_std_mm=250.0)

            v_pv = v_slots[:, 0]
            cov_pv = cov_slots[:, 0]
            scale_pv = torch.sqrt(torch.diagonal(cov_pv, dim1=-2, dim2=-1)).clamp(0.015, 250.0)
            sv_disp = v_slots[:, 1] - v_pv

            # Full 3x3 covariance displacement significance S_{L,k} for k = 1..6
            diff_k = v_slots[:, 1:] - v_pv[:, None, :]  # (B, 6, 3)
            cov_sum_k = cov_slots[:, 1:] + cov_pv[:, None, :, :]
            _, inv_sum_k = _solve_spd_3x3(cov_sum_k, diff_k)
            s_l2_k = torch.einsum('bki,bkij,bkj->bk', diff_k, inv_sum_k, diff_k).clamp_min(0.0)
            s_l_k = torch.sqrt(s_l2_k + 1e-8)  # (B, 6)

            # Flight-direction p_T-corrected invariant mass M_{corr, k} for k = 1..6
            diff_n = torch.sqrt(diff_k.square().sum(dim=-1, keepdim=True) + 1.0e-8)
            u_flight = diff_k / diff_n.clamp_min(1e-4)  # (B, 6, 3)
            norm_tr = torch.sqrt(1.0 + tx.square() + ty.square()).clamp_min(1.0)
            px_i = p_hat * tx / norm_tr
            py_i = p_hat * ty / norm_tr
            pz_i = p_hat / norm_tr
            p_vec_i = torch.stack([px_i, py_i, pz_i], dim=-1)  # (B, M, 3)
            e_vec_i = torch.sqrt(p_hat.square() + 0.13957 ** 2)

            p_fit_sv = p_fit[:, :, 1:].permute(0, 2, 1)  # (B, 6, M)
            p_sum_k = torch.einsum('bkm,bmi->bki', p_fit_sv, p_vec_i)
            e_sum_k = torch.einsum('bkm,bm->bk', p_fit_sv, e_vec_i)
            m_vis_k = torch.sqrt((e_sum_k.square() - p_sum_k.square().sum(dim=-1)).clamp_min(1.0e-8))
            cross_pf = torch.cross(p_sum_k, u_flight, dim=-1)
            pt_flight_k = torch.sqrt(cross_pf.square().sum(dim=-1) + 1.0e-8)
            m_corr_k = torch.sqrt(m_vis_k.square() + pt_flight_k.square() + 1.0e-8) + pt_flight_k

            # Air-gap prior P_air(z_SV) & 2-vertex chi2 improvement
            # Compare PV chi2 vs min(PV, SV1) chi2
            dz_0 = (v_pv[:, 2:3] - z0).clamp(-350.0, 350.0)
            r0_2 = ((x0 + tx * dz_0 - v_pv[:, 0:1]).square() / (sxx + dz_0.square() * stx)
                    + (y0 + ty * dz_0 - v_pv[:, 1:2]).square() / (syy + dz_0.square() * sty))
            dz_1 = (v_slots[:, 1, 2:3] - z0).clamp(-350.0, 350.0)
            r1_2 = ((x0 + tx * dz_1 - v_slots[:, 1, 0:1]).square() / (sxx + dz_1.square() * stx)
                    + (y0 + ty * dz_1 - v_slots[:, 1, 1:2]).square() / (syy + dz_1.square() * sty))
            delta_chi2_2v = (p_fit[:, :, 1] * (r0_2 - r1_2).clamp(0.0, 50.0)).sum(dim=-1)

        return {
            'v_pv_mm': v_pv,
            'cov_pv_mm2': cov_pv,
            'scale_pv_mm': scale_pv,
            'v_slots_mm': v_slots,
            'cov_slots_mm2': cov_slots,
            'sv_disp_mm': sv_disp,
            'seg_slot_probs': seg_slot_probs,
            's_l_slots': s_l_k,
            'm_corr_slots': m_corr_k,
            'delta_chi2_2v': delta_chi2_2v,
            'slot_embeds': q,
        }


class MicroConePairmixerNDIVE(nn.Module):
    """Complete Pillar-I backbone (`self.si_vtx`, ~1.74M parameters)."""

    def __init__(self, geom: Gen4GeometryContract | None = None, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.geom = geom or load_geometry_contract()
        self.cfg = dict(config or {})
        self.register_buffer('plane_x0', torch.as_tensor(self.geom.pixel_X0, dtype=torch.float32))

        d_cl = 128
        d_node = 160
        d_pair = 48
        d_meso = 192

        self.cluster_enc = DualTrackClusterEncoder(d_model=d_cl, d_meso=d_meso)
        self.aniso_ptv3 = AnisoPTv3(d_model=d_cl, heads=4, n_layers=3)
        self.slab_prior = BackConeSlabPrior(self.geom)
        self.slab_prior.enable_sandwich_slab = bool(self.cfg.get('enable_sandwich_slab', True))
        self.enable_kink_stubs = bool(self.cfg.get('enable_kink_stubs', True))
        self.enable_m2_kalman = bool(self.cfg.get('enable_m2_kalman', True))
        self.force_binary_no_edep = bool(self.cfg.get('force_binary_no_edep', False))

        # Segment embedding from 16 kinematics + 128 anchor cluster + 16 mean cluster
        self.seg_in = nn.Sequential(
            nn.Linear(SEG_STATE_DIM + d_cl + 8, d_node),
            nn.LayerNorm(d_node),
            nn.SiLU(),
        )
        self.pair_in = nn.Sequential(
            nn.Linear(PAIR_R_DIM, d_pair),
            nn.LayerNorm(d_pair),
            nn.SiLU(),
        )
        self.pairmixers = nn.ModuleList([_PairmixerLayer(d_pair=d_pair, d_node=d_node) for _ in range(2)])
        self.gta_layers = nn.ModuleList([_GTALayer(d_node=d_node, d_pair=d_pair, heads=8) for _ in range(2)])
        self.bg_registers = nn.Parameter(torch.randn(1, 6, d_node) * 0.02)

        self.comp_ndive = CompetitiveSlotNDIVE(
            self.geom,
            d_node=d_node,
            d_pair=d_pair,
            enable_comp_ndive=bool(self.cfg.get('enable_comp_ndive', True)),
        )

        # Pairwise co-vertex logit head (for dense GN2/sidecar supervision)
        self.pair_covertex_head = nn.Linear(d_pair, 1)

        # 36-dim explicit phi_vtx calibrator & 256-dim z_si_vtx pooler
        self.seg_pool_gate = nn.Linear(d_node, 1)
        self.u_micro_proj = nn.Linear(d_node, 128)
        self.z_si_vtx_proj = nn.Sequential(
            nn.LayerNorm(d_node + d_node + PHI_VTX_DIM),
            nn.Linear(d_node + d_node + PHI_VTX_DIM, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
        )

    def _compute_coarse_apex(
        self,
        pix_xyz_mm: torch.Tensor,
        pix_plane: torch.Tensor,
        pix_valid: torch.Tensor,
        pix_feat: torch.Tensor,
        cal_start_mm: torch.Tensor,
    ) -> torch.Tensor:
        w = pix_feat[..., 26].float().clamp_min(0.0) * (pix_plane < 12).float() * pix_valid.float()
        w_sum = w.sum(dim=1, keepdim=True)
        has_w = (w_sum[:, 0] > 0.5)
        cen = (w.unsqueeze(-1) * pix_xyz_mm.float()).sum(dim=1) / w_sum.clamp_min(1e-6)
        return torch.where(has_w[:, None], cen, cal_start_mm.float())

    def forward(self, batch: dict[str, torch.Tensor]) -> MicroVtxOutput:
        pix_xyz = batch['pix_xyz_mm']
        pix_plane = batch['pix_plane']
        pix_valid = batch['pix_valid']
        pix_feat = batch['pix_feat']
        pix_ps = batch['pix_plane_summary']
        pix_patches = batch['pix_patches']
        patch_cl_idx = batch['patch_cluster_idx']
        has_edep = batch['has_edep']
        if self.force_binary_no_edep:
            has_edep = torch.zeros_like(has_edep)
            if pix_patches.shape[2] >= 2:
                pix_patches = torch.cat([pix_patches[:, :, :1], torch.zeros_like(pix_patches[:, :, :1])], dim=2)
        cal_start = batch['cal_start_mm']
        scint_grid = batch['scint_grid']
        c_obs = batch['c_obs']
        muon_k = batch['muon_kalman_tracks']
        muon_km = batch['muon_kalman_mask'].bool()
        if not self.enable_m2_kalman:
            muon_km = muon_km & (~(muon_k[..., 7] > 0.5))
        b = int(c_obs.shape[0])
        device = c_obs.device

        # 1. Dual-track cluster encoding & AnisoPTv3
        h_cl, s_si_meso = self.cluster_enc(
            pix_feat, pix_valid, pix_plane, pix_ps, pix_patches, patch_cl_idx, has_edep
        )
        apex_mm = self._compute_coarse_apex(pix_xyz, pix_plane, pix_valid, pix_feat, cal_start)
        aniso_out = self.aniso_ptv3(
            h_cl, pix_xyz, pix_plane, pix_valid, pix_feat, apex_mm, self.plane_x0
        )
        h_cl = aniso_out['h_cl']
        g_persist = aniso_out['g_persist']
        s_tilde = aniso_out['s_tilde']

        # 2. Sandwich scintillator single-plate prior
        slab_out = self.slab_prior(
            pix_xyz, pix_plane, pix_valid, g_persist, scint_grid, cal_start
        )
        v_pv_init = slab_out['v_pv_init']
        p_pv_prior = slab_out['p_pv_prior']

        # 3. Vectorized GBL (Type-I/II) + Type-III 1-Hit Parent-Tau Kink Stubs (M=128)
        seg_out = build_gbl_and_kink_segments(
            pix_xyz, pix_plane, pix_valid, pix_feat, s_tilde,
            v_pv_init, muon_k, muon_km, self.plane_x0,
            has_edep=has_edep, enable_kink_stubs=self.enable_kink_stubs,
        )
        seg_state = seg_out['seg_state']
        seg_mask = seg_out['seg_mask']
        seg_cl_idx = seg_out['seg_cl_idx']
        seg_type = seg_out['seg_type']
        kink_mat = seg_out['kink_pair_matrix']

        # Gather anchor cluster embeddings for each segment
        n_cl = pix_xyz.shape[1]
        if n_cl > 0:
            ca_safe = seg_cl_idx[:, :, 0].clamp(0, n_cl - 1)
            cl_emb_seg = torch.gather(h_cl, 1, ca_safe.unsqueeze(-1).expand(-1, -1, h_cl.shape[-1]))
            cl_emb_seg = cl_emb_seg * (seg_cl_idx[:, :, 0] >= 0).unsqueeze(-1).float()
        else:
            cl_emb_seg = torch.zeros((b, N_SEG_TOTAL, 128), dtype=torch.float32, device=device)

        # Normalize segment kinematics for neural embedding
        st_f = seg_state.float()
        seg_norm = torch.stack([
            (st_f[..., 0] - self.geom.x_pix_center) / 135.0,
            (st_f[..., 1] - self.geom.y_scint_center) / 110.0,
            (st_f[..., 2] + 2400.0) / 800.0,
            st_f[..., 3] * 10.0,
            st_f[..., 4] * 10.0,
            torch.log1p(st_f[..., 5] / 1e-4) / 5.0,
            torch.log1p(st_f[..., 6] / 1e-4) / 5.0,
            torch.log1p(st_f[..., 7] / 1e-5) / 5.0,
            torch.log1p(st_f[..., 8] / 1e-5) / 5.0,
            torch.log1p(st_f[..., 9].clamp_min(0.0)) / 4.0,
            torch.log1p(st_f[..., 10].clamp_min(0.0)) / 4.0,
            st_f[..., 11] / 150.0,
            torch.log1p(st_f[..., 12].clamp_min(0.0)) / 5.0,
            st_f[..., 13],
            st_f[..., 14],
            st_f[..., 15],
        ], dim=-1)
        seg_ctx = torch.cat([
            seg_norm,
            cl_emb_seg,
            slab_out['slab_probs'].unsqueeze(1).expand(-1, N_SEG_TOTAL, -1),
            kink_mat.any(dim=-1).unsqueeze(-1).float(),
            torch.zeros((b, N_SEG_TOTAL, 3), dtype=torch.float32, device=device),
        ], dim=-1)
        s_seg = self.seg_in(seg_ctx) * seg_mask.unsqueeze(-1).float()

        # 4. PV-free GTA + Pairmixer + 6 GN3 background registers
        r_ij, _, rel_pos = compute_pairwise_ray_geometry(seg_state, seg_mask, kink_mat, self.geom)
        z_pair = self.pair_in(r_ij) * (seg_mask[:, :, None, None] & seg_mask[:, None, :, None]).float()

        regs = self.bg_registers.expand(b, -1, -1)
        reg_mask = torch.ones((b, 6), dtype=torch.bool, device=device)

        for pmx, gta in zip(self.pairmixers, self.gta_layers):
            z_pair = pmx(z_pair, s_seg, seg_mask)
            s_all = torch.cat([s_seg, regs], dim=1)
            m_all = torch.cat([seg_mask, reg_mask], dim=1)
            s_all = gta(s_all, m_all, z_pair, rel_pos)
            s_seg = s_all[:, :N_SEG_TOTAL]
            regs = s_all[:, N_SEG_TOTAL:]

        pair_covtx_logits = self.pair_covertex_head(z_pair).squeeze(-1)

        # 5. Competitive 8-slot Comp-NDIVE Implicit Differentiation
        nd_out = self.comp_ndive(
            s_seg, seg_state, seg_mask, seg_type, z_pair, r_ij, v_pv_init, p_pv_prior,
            regime_weights=slab_out['regime_weights'],
        )

        # 6. Build 36-dim phi_vtx strictly zero-gated when c_obs[:, 0] == 0
        has_si = (c_obs[:, 0] > 0.5).float().view(b, 1)
        p_ik = nd_out['seg_slot_probs']  # (B, 128, 8)
        slot_counts = p_ik.sum(dim=1)    # (B, 8)
        s_l = nd_out['s_l_slots']        # (B, 6)
        m_corr = nd_out['m_corr_slots']  # (B, 6)
        v_slots = nd_out['v_slots_mm']
        sv_disp = nd_out['sv_disp_mm']

        # Max kink p_T across Type-III pairs
        pt_kink_mat = r_ij[..., 5]  # asinh(p_T_kink / 0.05) / 3.0
        pt_kink_max = (pt_kink_mat * kink_mat.float()).amax(dim=(1, 2))
        # Air-gap prior at tau-SV and Charm-SV
        z_sv_np = v_slots[:, 1:3, 2].detach().cpu().numpy()
        p_air_sv = torch.from_numpy(
            (self.geom.material_conversion_prior(z_sv_np) <= 2.0 * AIR_CONV_SUPPRESSION).astype(np.float32)
        ).to(device=device)

        # Muon track max IP significance relative to calibrated PV
        v_pv = nd_out['v_pv_mm']
        cov_pv = nd_out['cov_pv_mm2']
        if muon_km.any():
            mk_x, mk_y = muon_k[..., 10].float(), muon_k[..., 11].float()
            mk_tx, mk_ty = muon_k[..., 12].float(), muon_k[..., 13].float()
            mk_kappa = muon_k[..., 0].float() * float(KAPPA_B_CONST_MEV_MM / 1000.0)
            dz_mk = v_pv[:, 2:3] - float(TRACKER_Z_MM[0])
            dx_mk = (mk_x + mk_tx * dz_mk) - v_pv[:, 0:1]
            y_ext_mk = (
                mk_y
                + mk_ty * dz_mk
                - mk_kappa * float(MAGNET0_LENGTH_MM) * (v_pv[:, 2:3] - float(MAGNET0_Z_CENTER_MM))
            )
            dy_mk = y_ext_mk - v_pv[:, 1:2]
            d0_mk = _safe_hypot(dx_mk, dy_mk) * muon_km.float()
            sig_d0_mk = torch.sqrt(cov_pv[:, 0, 0:1] + cov_pv[:, 1, 1:2] + (0.05 + 0.002 * dz_mk.abs() * muon_k[..., 2].float().clamp_min(1.0).reciprocal()).square())
            s_d0_mu_max = (d0_mk / sig_d0_mk.clamp_min(0.02)).amax(dim=1)
        else:
            s_d0_mu_max = torch.zeros((b,), dtype=torch.float32, device=device)

        s_tilde_mean = aniso_out['s_tilde'].sum(dim=1, keepdim=True) / pix_valid.sum(dim=1, keepdim=True).clamp_min(1).float()
        v0_frac = (p_ik[:, :, 3:7].sum(dim=(1, 2)) / seg_mask.sum(dim=1).clamp_min(1).float()).unsqueeze(-1)
        phi_vtx_raw = torch.cat([
            torch.log1p(s_l) / 3.0,                                       # 6: S_L for tau, charm, 4xV0
            torch.asinh(m_corr / 1.5) / 3.0,                              # 6: M_corr for tau, charm, 4xV0
            torch.log1p(slot_counts) / 4.0,                               # 8: effective multiplicity per slot
            torch.log1p(nd_out['delta_chi2_2v'].clamp_min(0.0)).unsqueeze(-1) / 4.0,  # 1
            torch.asinh(torch.sqrt(sv_disp.square().sum(dim=-1, keepdim=True) + 1e-8) / 2.0) / 3.0,  # 1: |sv_disp|
            torch.asinh(sv_disp[:, 2:3] / 5.0) / 3.0,                     # 1: dz_sv
            pt_kink_max.unsqueeze(-1),                                    # 1: max kink pT
            torch.log1p(s_d0_mu_max).unsqueeze(-1) / 3.0,                 # 1: S_d0^mu
            p_air_sv,                                                     # 2: air gap flag tau/charm
            torch.log(nd_out['scale_pv_mm'].clamp(0.015, 250.0)) / 4.0,   # 3: PV scale log
            slab_out['slab_probs'],                                       # 4: slab prior probs
            v0_frac,                                                      # 1: V0 fraction
            torch.tanh(s_tilde_mean),                                     # 1: non-conv cluster density
        ], dim=-1)
        assert phi_vtx_raw.shape[-1] == PHI_VTX_DIM
        phi_vtx = phi_vtx_raw * has_si

        # 7. Pool segment tokens -> z_si_vtx (B, 256) and u_micro (B, 128, 128)
        any_seg = seg_mask.any(dim=1, keepdim=True)
        safe_sm = torch.where(any_seg, seg_mask, F.one_hot(torch.zeros((b,), dtype=torch.long, device=device), num_classes=N_SEG_TOTAL).bool())
        pool_logits = self.seg_pool_gate(s_seg).squeeze(-1).masked_fill(~safe_sm, -1.0e4)
        pool_w = torch.softmax(pool_logits, dim=1) * seg_mask.float()
        seg_glob = (pool_w.unsqueeze(-1) * s_seg).sum(dim=1)
        reg_glob = regs.mean(dim=1)
        z_si_vtx = self.z_si_vtx_proj(torch.cat([seg_glob, reg_glob, phi_vtx], dim=-1)) * has_si
        u_micro = self.u_micro_proj(s_seg) * seg_mask.unsqueeze(-1).float()

        return MicroVtxOutput(
            v_pv_mm=v_pv,
            cov_pv_mm2=cov_pv,
            scale_pv_mm=nd_out['scale_pv_mm'],
            v_slots_mm=v_slots,
            cov_slots_mm2=nd_out['cov_slots_mm2'],
            sv_disp_mm=sv_disp,
            seg_state=seg_state,
            seg_mask=seg_mask,
            seg_slot_probs=p_ik,
            phi_vtx=phi_vtx,
            z_si_vtx=z_si_vtx,
            u_micro=u_micro,
            s_si_meso=s_si_meso,
            aux={
                'cl_origin_logits': aniso_out['cl_origin_logits'],
                'dark_logits': aniso_out['dark_logits'],
                'g_persist': g_persist,
                'pair_covertex_logits': pair_covtx_logits,
                'seg_cl_idx': seg_cl_idx,
                'seg_type': seg_type,
                'kink_pair_matrix': kink_mat,
                'slab_probs': slab_out['slab_probs'],
                'v_pv_init': v_pv_init,
                's_l_slots': s_l,
                'm_corr_slots': m_corr,
                'delta_chi2_2v': nd_out['delta_chi2_2v'],
                'pt_kink_max': pt_kink_max,
                's_d0_mu_max': s_d0_mu_max,
            },
        )
