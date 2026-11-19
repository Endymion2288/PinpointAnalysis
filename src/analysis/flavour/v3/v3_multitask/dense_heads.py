"""GN2 + FASERCal dense particle-rel, 5-class cluster origin, and track/vertex compatibility heads."""

from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


CLUSTER_ORIGIN_NAMES = (
    'Primary_lepton',
    'Primary_hadron',
    'Tau_daughter',
    'Charm_daughter',
    'Secondary_shower',
)
TOPO_SUMMARY_DIM = 8
K_PAIR_MAX = 64

# Mild square-root-inverse frequency weights N_c^-0.4 clipped to [0.5, 5.0]
# From observed cluster counts {0: 616658, 1: 18837, 2: 73303, 3: 46823, 4: 1244379}
DEFAULT_ORIGIN_WEIGHTS = (0.85, 3.40, 2.00, 2.38, 0.50)


def _soft_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked 3-class (EM, mu, had) soft cross-entropy computed in FP32."""
    if len(logits) == 0 or not mask.any():
        return logits.sum() * 0.0
    with torch.autocast(device_type=logits.device.type, enabled=False):
        lg = logits[mask].float()
        tgt = target_probs[mask].float()
        tgt = tgt / tgt.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        log_p = F.log_softmax(lg, dim=-1)
        return -(tgt * log_p).sum(dim=-1).mean()


def _balanced_pair_bce(logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Pos/neg balanced binary cross-entropy across valid off-diagonal cluster pairs."""
    if not valid_mask.any():
        return logits.sum() * 0.0
    with torch.autocast(device_type=logits.device.type, enabled=False):
        lg = logits[valid_mask].float()
        yt = targets[valid_mask].float()
        bce = F.binary_cross_entropy_with_logits(lg, yt, reduction='none')
        pos = yt > 0.5
        neg = ~pos
        if pos.any() and neg.any():
            return 0.5 * (bce[pos].mean() + bce[neg].mean())
        return bce.mean()


class DenseGN2Heads(nn.Module):
    """Dense Rel, Cluster Origin, Track/Vertex Compatibility heads + 8-dim zero-truth summary."""

    def __init__(
        self,
        dim: int = 192,
        bar_dim: int = 128,
        aff_dim: int = 64,
        k_pair: int = K_PAIR_MAX,
        origin_weights: tuple[float, ...] = DEFAULT_ORIGIN_WEIGHTS,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.k_pair = k_pair
        self.register_buffer('origin_weights', torch.tensor(origin_weights, dtype=torch.float32))

        self.pix_rel_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 96),
            nn.SiLU(),
            nn.Linear(96, 3),
        )
        self.bar_rel_head = nn.Sequential(
            nn.LayerNorm(bar_dim),
            nn.Linear(bar_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
        )
        self.origin_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 128),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 5),
        )
        self.trk_aff_proj = nn.Linear(dim, aff_dim)
        self.vtx_aff_proj = nn.Linear(dim, aff_dim)
        self.log_tau_trk = nn.Parameter(torch.tensor(math.log(8.0)))
        self.log_tau_vtx = nn.Parameter(torch.tensor(math.log(8.0)))

    def forward(self, aux: dict, enable_dense: bool = True) -> dict[str, torch.Tensor]:
        """Compute dense predictions and differentiable 8-dim self-predicted topological summary."""
        u_pix = aux['u_pix']
        u_vtx = aux['u_vtx']
        vtx_xyz = aux['vtx_xyz_mm'].float()
        vtx_mask = aux['vtx_mask']
        v_hat = aux['v_hat'].float()
        active_bar_emb = aux['active_bar_emb']

        pix_rel_logits = self.pix_rel_head(u_pix)
        bar_rel_logits = self.bar_rel_head(active_bar_emb)
        origin_logits = self.origin_head(u_vtx)

        k_p = min(self.k_pair, u_vtx.shape[1])
        u_pair = u_vtx[:, :k_p]
        mask_p = vtx_mask[:, :k_p]
        xyz_p = vtx_xyz[:, :k_p]

        e_trk = F.normalize(self.trk_aff_proj(u_pair).float(), p=2, dim=-1, eps=1e-6)
        e_vtx = F.normalize(self.vtx_aff_proj(u_pair).float(), p=2, dim=-1, eps=1e-6)
        tau_trk = torch.clamp(self.log_tau_trk.exp().float(), min=1.0, max=30.0)
        tau_vtx = torch.clamp(self.log_tau_vtx.exp().float(), min=1.0, max=30.0)
        m_trk = torch.matmul(e_trk, e_trk.transpose(-2, -1)) * tau_trk
        m_vtx = torch.matmul(e_vtx, e_vtx.transpose(-2, -1)) * tau_vtx

        # Pure-forward zero-truth 8-dim Self-Predicted Topological Summary (computed in FP32)
        if enable_dense:
            with torch.autocast(device_type=origin_logits.device.type, enabled=False):
                q_or = torch.softmax(origin_logits.float(), dim=-1)
                vm = vtx_mask.float()
                denom_v = vm.sum(dim=1, keepdim=True).clamp_min(1.0)

                q_tau = q_or[..., 2] * vm
                q_lep = q_or[..., 0] * vm
                q_chm = q_or[..., 3] * vm

                max_tau = q_tau.max(dim=1).values
                top3_tau = torch.topk(q_tau, k=min(3, q_tau.shape[1]), dim=1).values.mean(dim=1)
                max_lep = q_lep.max(dim=1).values
                max_chm = q_chm.max(dim=1).values
                mean_chm = q_chm.sum(dim=1) / denom_v.squeeze(1)

                dxyz = vtx_xyz - v_hat[:, None, :]
                r_perp = torch.sqrt((dxyz[..., 0].square() + dxyz[..., 1].square()).clamp_min(1e-8))
                dz = dxyz[..., 2]
                dz_safe = torch.where(dz >= 0.0, dz.clamp_min(1.0), dz.clamp_max(-1.0))
                tx = torch.clamp(dxyz[..., 0] / dz_safe, -5.0, 5.0)
                ty = torch.clamp(dxyz[..., 1] / dz_safe, -5.0, 5.0)

                w_tau = q_tau + 1e-4 * vm
                w_sum = w_tau.sum(dim=1, keepdim=True).clamp_min(1e-6)
                tau_r_mean = (w_tau * torch.log1p(r_perp / 0.05) / 5.0).sum(dim=1) / w_sum.squeeze(1)
                mean_tx = (w_tau * tx).sum(dim=1, keepdim=True) / w_sum
                mean_ty = (w_tau * ty).sum(dim=1, keepdim=True) / w_sum
                polar_disp = torch.sqrt(
                    ((w_tau * ((tx - mean_tx).square() + (ty - mean_ty).square())).sum(dim=1) / w_sum.squeeze(1))
                    .clamp_min(1e-8)
                )

                # Soft count of high-probability off-diagonal cross-plane secondary vertex pairs
                z_p = xyz_p[..., 2]
                upper_cross = (torch.triu(torch.ones((k_p, k_p), device=z_p.device, dtype=torch.bool), diagonal=1)[None, :, :])
                upper_cross = upper_cross & mask_p[:, :, None] & mask_p[:, None, :] & (torch.abs(z_p[:, None, :] - z_p[:, :, None]) >= 0.5)
                prob_sec = torch.sigmoid(m_vtx) * upper_cross.float()
                soft_sec_cnt = torch.clamp(prob_sec.sum(dim=(1, 2)) / 10.0, max=5.0)

                topo_summary = torch.stack([
                    max_tau,
                    top3_tau,
                    max_lep,
                    max_chm,
                    mean_chm,
                    tau_r_mean,
                    torch.clamp(polar_disp * 20.0, max=5.0),
                    soft_sec_cnt,
                ], dim=-1)
        else:
            topo_summary = origin_logits.new_zeros((len(origin_logits), TOPO_SUMMARY_DIM), dtype=torch.float32)

        return {
            'pix_rel_logits': pix_rel_logits,
            'bar_rel_logits': bar_rel_logits,
            'origin_logits': origin_logits,
            'm_trk': m_trk,
            'm_vtx': m_vtx,
            'topo_summary': topo_summary,
        }

    def compute_losses(self, dense_out: dict[str, torch.Tensor], aux: dict, batch: dict) -> dict[str, torch.Tensor]:
        """Compute supervised L_rel, L_origin, and L_compat from training sidecar targets."""
        l_pix_rel = _soft_cross_entropy(
            dense_out['pix_rel_logits'],
            batch['pix_rel_target'],
            batch['pix_rel_valid'],
        )
        bar_rel_t = batch['bar_rel_target']
        if len(dense_out['bar_rel_logits']) and len(bar_rel_t) == len(dense_out['bar_rel_logits']):
            bar_mask = bar_rel_t.sum(dim=-1) > 0.01
            l_bar_rel = _soft_cross_entropy(dense_out['bar_rel_logits'], bar_rel_t, bar_mask)
        else:
            l_bar_rel = dense_out['pix_rel_logits'].sum() * 0.0
        l_rel = 0.65 * l_pix_rel + 0.35 * l_bar_rel

        vtx_idx = aux['vtx_idx']
        vtx_mask = aux['vtx_mask']
        or_tgt = torch.gather(batch['pix_origin_target'], 1, vtx_idx)
        or_valid = vtx_mask & (or_tgt >= 0) & (or_tgt < 5)
        if or_valid.any():
            with torch.autocast(device_type=vtx_idx.device.type, enabled=False):
                l_origin = F.cross_entropy(
                    dense_out['origin_logits'][or_valid].float(),
                    or_tgt[or_valid],
                    weight=self.origin_weights.float(),
                )
        else:
            l_origin = dense_out['origin_logits'].sum() * 0.0

        # L_compat on K_pair <= 64 cross-plane clusters
        k_p = dense_out['m_trk'].shape[1]
        pair_idx = vtx_idx[:, :k_p]
        pair_mask = vtx_mask[:, :k_p]
        pair_xyz = aux['vtx_xyz_mm'][:, :k_p].float()
        pair_or = torch.gather(batch['pix_origin_target'], 1, pair_idx)
        pair_anc = torch.gather(batch['pix_ancestry_target'], 1, pair_idx.unsqueeze(-1).expand(-1, -1, 2))

        anc0 = pair_anc[..., 0]
        anc1 = pair_anc[..., 1]
        z_p = pair_xyz[..., 2]
        upper = torch.triu(torch.ones((k_p, k_p), device=z_p.device, dtype=torch.bool), diagonal=1)[None, :, :]
        cross = upper & pair_mask[:, :, None] & pair_mask[:, None, :] & (torch.abs(z_p[:, None, :] - z_p[:, :, None]) >= 0.5)

        is_sig_i = (pair_or >= 0) & (pair_or <= 3)
        is_sig_j = is_sig_i
        trk_valid = cross & (anc0[:, :, None] > 0) & (anc0[:, None, :] > 0) & (is_sig_i[:, :, None] | is_sig_j[:, None, :])
        trk_tgt = anc0[:, :, None] == anc0[:, None, :]
        l_trk_cmp = _balanced_pair_bce(dense_out['m_trk'], trk_tgt, trk_valid)

        sec_i = (pair_or == 2) | (pair_or == 3)
        sec_j = sec_i
        # Require resolved vertex ID (anc1 >= 0) so unresolved multi-charm shower descendants are masked out
        vtx_cmp_valid = (
            cross
            & (anc0[:, :, None] > 0)
            & (anc0[:, None, :] > 0)
            & (anc1[:, :, None] >= 0)
            & (anc1[:, None, :] >= 0)
            & (pair_or[:, :, None] >= 0)
            & (pair_or[:, None, :] >= 0)
        )
        same_sec = sec_i[:, :, None] & sec_j[:, None, :] & (pair_or[:, :, None] == pair_or[:, None, :]) & (anc1[:, :, None] == anc1[:, None, :])
        l_vtx_cmp = _balanced_pair_bce(dense_out['m_vtx'], same_sec, vtx_cmp_valid)

        l_compat = 0.60 * l_trk_cmp + 0.40 * l_vtx_cmp
        return {
            'l_rel': l_rel,
            'l_origin': l_origin,
            'l_compat': l_compat,
        }
