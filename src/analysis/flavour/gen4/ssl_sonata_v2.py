"""PinpointSSL-v2: Physics-Relational Sonata-MS + O(N) LeJEPA SIGReg/KoLeo +
15 um Ray Interpolation + Pre-Truncation Poisson Dark-Noise Consistency (HW-Inv).

Operates directly on the full PinpointGen4Net hierarchy (`si_vtx`, `cal_pflow`, `spec_ipt`, `fusion`),
pretraining >96% of all downstream supervised parameters on the unlabeled `10085-10088`
`SSL_320k` corpus (`r in {14, 15}`, `has_subthr_readout = 1.0`, `has_edep = 1.0`, `pix_feat[..., 30:36]`)
with zero truth/sidecar access.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from analysis.flavour.v3.v1_b1pp.spectrometer import SIGMA_SAGITTA_Y_MM


N_PIX_PLANES = 15
N_CAL_STATIONS = 54
SONATA_NUM_PROTOTYPES = 512


@dataclass
class SonataSSLConfig:
    """Hyperparameters for PinpointSSL-v2 pretraining."""

    dim_micro: int = 128
    dim_meso: int = 192
    dim_macro: int = 256
    num_prototypes: int = SONATA_NUM_PROTOTYPES
    proj_dim: int = 128
    teacher_temp: float = 0.06
    student_temp: float = 0.10
    sinkhorn_iters: int = 3
    sigreg_num_proj: int = 64
    sigreg_gh_nodes: int = 8
    hilbert_mask_frac: float = 0.45
    plane_drop_min: int = 2
    plane_drop_max: int = 3
    coord_jitter_um: float = 60.0
    stereo_drop_prob: float = 0.30
    spec_drop_prob: float = 0.50
    dark_noise_prob: float = 0.35
    hw_deg_prob: float = 0.30
    subthr_mask_prob: float = 0.15
    w_sonata: float = 1.0
    w_sigreg: float = 0.25
    w_koleo: float = 0.10
    w_phys_ray: float = 0.60
    w_phys_cal: float = 0.45
    w_phys_spec: float = 0.45
    w_hw_inv: float = 0.50


# ---------------------------------------------------------------------------
# Sinkhorn-Knopp Centered Prototypes & Sonata-MS Self-Distillation
# ---------------------------------------------------------------------------
@torch.no_grad()
def sinkhorn_knopp_teacher(
    logits: torch.Tensor,
    *,
    temp: float = 0.06,
    n_iters: int = 3,
) -> torch.Tensor:
    """Balanced Sinkhorn-Knopp prototype assignment Q in FP32 avoiding prototype collapse."""
    if logits.numel() == 0:
        return logits.new_empty(logits.shape)
    q = torch.exp((logits.float() / max(float(temp), 1e-3)).clamp(-30.0, 30.0)).t()  # (K, N)
    q = q / q.sum().clamp_min(1e-12)
    k_dim, n_tok = q.shape
    for _ in range(int(max(1, n_iters))):
        q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-12) / float(k_dim)
        q = q / q.sum(dim=0, keepdim=True).clamp_min(1e-12) / float(n_tok)
    q = q * float(n_tok)
    return q.t().contiguous()


class SonataPrototypeHead(nn.Module):
    """L2-normalized prototype projection head for one Sonata scale."""

    def __init__(self, in_dim: int, proj_dim: int = 128, num_prototypes: int = SONATA_NUM_PROTOTYPES) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, proj_dim),
        )
        self.prototypes = nn.Linear(proj_dim, num_prototypes, bias=False)
        nn.init.normal_(self.prototypes.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = F.normalize(self.mlp(x.float()), p=2, dim=-1, eps=1e-6)
        w_norm = F.normalize(self.prototypes.weight.float(), p=2, dim=-1, eps=1e-6)
        logits = torch.matmul(z, w_norm.t())
        return z, logits


# ---------------------------------------------------------------------------
# O(N) LeJEPA SIGReg (Epps-Pulley Gauss-Hermite) + Intra-Event KoLeo
# ---------------------------------------------------------------------------
def _gauss_hermite_nodes_weights(q: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    nodes, weights = np.polynomial.hermite.hermgauss(q)
    # Standardize for N(0, 1): x = sqrt(2)*t_herm, w_norm = w / sqrt(pi)
    t_q = torch.tensor(nodes * math.sqrt(2.0), dtype=torch.float32)
    w_q = torch.tensor(weights / math.sqrt(math.pi), dtype=torch.float32)
    return t_q, w_q


class LeJEPASIGReg(nn.Module):
    """O(N * M_proj * Q) isotropic Gaussian characteristic-function regularizer (Balestriero & LeCun, 2025)."""

    def __init__(self, num_proj: int = 64, gh_nodes: int = 8) -> None:
        super().__init__()
        self.num_proj = int(num_proj)
        t_q, w_q = _gauss_hermite_nodes_weights(gh_nodes)
        self.register_buffer('t_q', t_q)
        self.register_buffer('w_q', w_q)
        self.register_buffer('phi_gauss', torch.exp(-0.5 * t_q.square()))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (N, D) representations -> scalar Epps-Pulley discrepancy against N(0, I_D)."""
        with torch.amp.autocast(device_type=z.device.type, enabled=False):
            zf = z.float()
            n, d = zf.shape
            if n < 4 or d < 2:
                return zf.sum() * 0.0
            # Standardize per feature across batch for scale-free isotropic Gaussian check
            zn = (zf - zf.mean(dim=0, keepdim=True)) / zf.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
            u = torch.randn((d, self.num_proj), device=zn.device, dtype=torch.float32)
            u = F.normalize(u, p=2, dim=0, eps=1e-6)
            proj = zn @ u  # (N, M)
            # (N, M, Q)
            ang = proj.unsqueeze(-1) * self.t_q.view(1, 1, -1)
            ecf_re = torch.cos(ang).mean(dim=0)  # (M, Q)
            ecf_im = torch.sin(ang).mean(dim=0)  # (M, Q)
            err = (ecf_re - self.phi_gauss.view(1, -1)).square() + ecf_im.square()
            return (err * self.w_q.view(1, -1)).sum(dim=-1).mean()


def intra_event_koleo_loss(
    u_tokens: torch.Tensor,
    token_mask: torch.Tensor,
    *,
    max_k: int = 64,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Intra-event Kozachenko-Leonenko differential entropy repulsion over top-K tokens per event."""
    with torch.amp.autocast(device_type=u_tokens.device.type, enabled=False):
        b, n, d = u_tokens.shape
        k_use = min(int(max_k), n)
        u_hat = F.normalize(u_tokens[:, :k_use].float(), p=2, dim=-1, eps=1e-6)
        m = token_mask[:, :k_use].bool()
        losses: list[torch.Tensor] = []
        for bi in range(b):
            cnt = int(m[bi].sum().item())
            if cnt < 2:
                continue
            ub = u_hat[bi, m[bi]]  # (C, D)
            dist = torch.cdist(ub, ub, p=2)
            eye = torch.eye(cnt, device=ub.device, dtype=torch.bool)
            dist = dist.masked_fill(eye, 1e6)
            nn_dist = dist.min(dim=-1).values
            losses.append(-torch.log(nn_dist + float(eps)).mean())
        if not losses:
            return u_tokens.sum() * 0.0
        return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Anisotropic Hilbert Serialization & Anti-Shortcut Student Masking
# ---------------------------------------------------------------------------
def _anisotropic_hilbert_order_2d3d(
    x_mm: torch.Tensor,
    y_mm: torch.Tensor,
    x0: torch.Tensor,
) -> torch.Tensor:
    """Compute vectorized 3D Z-order / Morton proxy on `(x/0.05mm, y/0.05mm, X0/0.5)`."""
    qx = ((x_mm + 140.0) / 0.05).long().clamp(0, 1023)
    qy = ((y_mm + 120.0) / 0.05).long().clamp(0, 1023)
    qz = (x0 / 0.5).long().clamp(0, 1023)

    def _part1by2(v: torch.Tensor) -> torch.Tensor:
        v = (v | (v << 16)) & 0x030000FF
        v = (v | (v << 8)) & 0x0300F00F
        v = (v | (v << 4)) & 0x030C30C3
        v = (v | (v << 2)) & 0x09249249
        return v

    return _part1by2(qx) | (_part1by2(qy) << 1) | (_part1by2(qz) << 2)


def _recompute_c_obs_from_clusters_and_bars(
    c_obs_orig: torch.Tensor,
    pix_xyz: torch.Tensor,
    pix_plane: torch.Tensor,
    pix_valid: torch.Tensor,
    pix_persist: torch.Tensor,
    spec_drop_mask: torch.Tensor,
) -> torch.Tensor:
    """Recompute noise-immune `c_obs` columns affected by plane masking or spectrometer drop."""
    c_new = c_obs_orig.clone()
    b = c_new.shape[0]
    for bi in range(b):
        vm = pix_valid[bi]
        if vm.any():
            w = pix_persist[bi, vm].clamp_min(0.0)
            has_tgt = bool(((pix_plane[bi, vm] < 12) & (w > 0.5)).any().item())
            c_new[bi, 0] = 1.0 if has_tgt else 0.0
            c_new[bi, 13] = float(torch.log1p(w.sum()).item())
        else:
            c_new[bi, 0] = 0.0
            c_new[bi, 13] = 0.0
    if spec_drop_mask.any():
        c_new[spec_drop_mask, 2] = 0.0
        c_new[spec_drop_mask, 3] = 0.0
    return c_new


def inject_online_dark_noise_and_degradation(
    batch: Mapping[str, torch.Tensor],
    *,
    dark_prob: float = 0.35,
    deg_prob: float = 0.30,
    generator: torch.Generator | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Pre-truncation Poisson dark noise (35..2195 hits/plane) & dead-channel degradation.

    Returns:
      `(degraded_batch, orig_valid_target_mask, apply_hw_inv_mask)`
      where `orig_valid_target_mask` is `(B, N_cl)` indicating true signal clusters (1.0)
      vs injected single-pixel dark noise clusters (0.0) for `BCE(g_persist, is_orig)`.
    """
    dev = batch['pix_xyz_mm'].device
    bsz, n_cl, _ = batch['pix_xyz_mm'].shape
    out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}

    is_orig = batch['pix_valid'].bool().clone()
    r_dark = torch.rand((bsz,), generator=generator, device=dev) < float(dark_prob)
    r_deg = torch.rand((bsz,), generator=generator, device=dev) < float(deg_prob)

    # Inject isolated 1-pixel dark noise into unused padding slots + measure persistence degradation
    # For dense tensors already padded to N_cl, replace up to 25% of empty slots (or random low-budget
    # slots when full) with isolated single-pixel dark noise across 15 planes spanning 35..2195/plane!
    if r_dark.any():
        px = out['pix_xyz_mm']
        pp = out['pix_plane']
        pf = out['pix_feat']
        pv = out['pix_valid']
        for bi in torch.nonzero(r_dark, as_tuple=True)[0].tolist():
            # Sample log-uniform dark rate 10^[-6.5, -4.7] -> mean 35..2195 hits/plane
            log_rho = float(torch.empty((), device=dev).uniform_(-6.5, -4.7, generator=generator).item())
            rho = 10.0 ** log_rho
            mean_per_plane = rho * 1.099e8  # 12788 * 8596
            n_dark_total = int(min(max(16, int(mean_per_plane * 0.08)), max(16, n_cl // 3)))

            empty_slots = torch.nonzero(~pv[bi], as_tuple=True)[0]
            if len(empty_slots) >= 8:
                n_ins = min(len(empty_slots), n_dark_total)
                ins_idx = empty_slots[:n_ins]
            else:
                # Simulate truncation flooding by overwriting tail non-core slots
                n_ins = min(n_dark_total, max(8, n_cl // 5))
                ins_idx = torch.arange(n_cl - n_ins, n_cl, device=dev)
                is_orig[bi, ins_idx] = False

            pl_rand = torch.randint(0, N_PIX_PLANES, (n_ins,), generator=generator, device=dev)
            x_rand = torch.empty((n_ins,), device=dev).uniform_(-132.0, 132.0, generator=generator)
            y_rand = torch.empty((n_ins,), device=dev).uniform_(-109.0, 85.0, generator=generator)
            z_vals = px[bi, pv[bi], 2]
            z_mean = float(z_vals.mean().item()) if len(z_vals) > 0 else -2900.0

            px[bi, ins_idx, 0] = x_rand
            px[bi, ins_idx, 1] = y_rand
            px[bi, ins_idx, 2] = z_mean + (pl_rand.float() - 7.0) * 25.0
            pp[bi, ins_idx] = pl_rand
            pv[bi, ins_idx] = True

            # Single-pixel dark hit morphology: n_hits=1, zero kurtosis/skewness, flooded plane log(N_1pix)
            pf[bi, ins_idx] = 0.0
            pf[bi, ins_idx, 0] = x_rand / 135.0
            pf[bi, ins_idx, 1] = (y_rand + 12.0) / 100.0
            pf[bi, ins_idx, 2] = (px[bi, ins_idx, 2] - z_mean) / 1000.0
            pf[bi, ins_idx, 3] = math.log1p(1.0) / 6.0
            # Encode dark flood density cue in persistence/neighborhood slots (high 1-pix count, 0 adj match)
            pf[bi, ins_idx, 26] = 0.05
            if pf.shape[-1] >= 36:
                pf[bi, ins_idx, 30] = math.log1p(0.35)
                pf[bi, ins_idx, 32] = 1.0

            # Mark original vs dark in `is_orig`
            is_orig[bi, ins_idx] = False

    if r_deg.any():
        sg = out['scint_grid']
        for bi in torch.nonzero(r_deg, as_tuple=True)[0].tolist():
            # 0-4% dead scintillator channels
            p_dead = float(torch.empty((), device=dev).uniform_(0.0, 0.04, generator=generator).item())
            if p_dead > 0.0:
                alive = torch.rand((N_CAL_STATIONS, 2, 40), generator=generator, device=dev) >= p_dead
                sg[bi] = sg[bi] * alive.unsqueeze(-1).float()
            # 5% SiPM gain jitter on energy channels [0, 2]
            gain = 1.0 + 0.05 * torch.randn((N_CAL_STATIONS, 2, 40, 1), generator=generator, device=dev)
            sg[bi, ..., 0] = (sg[bi, ..., 0] * gain.squeeze(-1)).clamp_min(0.0)
            sg[bi, ..., 2] = (sg[bi, ..., 2] * gain.squeeze(-1)).clamp_min(0.0)

            # 0-4% dead tracker candidates + Poisson(1) false tracker noise
            if 'muon_kalman_mask' in out and out['muon_kalman_mask'] is not None:
                mkm = out['muon_kalman_mask'][bi]
                if mkm.any():
                    drop_m = torch.rand(mkm.shape, generator=generator, device=dev) < p_dead
                    out['muon_kalman_mask'][bi] = mkm & (~drop_m)
                    out['muon_kalman_tracks'][bi, drop_m] = 0.0

    # Preserve `has_subthr_readout` and `has_edep` from input batch (compatible with both old & 10085-10088 MC)
    return out, is_orig, (r_dark | r_deg)


def build_sonata_student_view(
    batch: Mapping[str, torch.Tensor],
    *,
    cfg: SonataSSLConfig | None = None,
    generator: torch.Generator | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Construct zero-leakage Sonata-MS student batch + ground-truth SSL mask targets."""
    c = cfg or SonataSSLConfig()
    dev = batch['pix_xyz_mm'].device
    bsz, n_cl, _ = batch['pix_xyz_mm'].shape
    stu = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    if 'has_subthr_readout' in stu and 'scint_grid' in stu:
        # Randomly mask <2.5 PE subthreshold halo bars on p=subthr_mask_prob so backbone generalizes across thresholds
        drop_sub = torch.rand((bsz,), generator=generator, device=dev) < float(c.subthr_mask_prob)
        if drop_sub.any():
            is_sub = stu['scint_grid'][..., 7] > 0.5
            stu['scint_grid'][drop_sub] = torch.where(
                is_sub[drop_sub].unsqueeze(-1),
                torch.zeros_like(stu['scint_grid'][drop_sub]),
                stu['scint_grid'][drop_sub],
            )
            stu['has_subthr_readout'] = stu['has_subthr_readout'].float() * (~drop_sub).float()

    px_xyz = stu['pix_xyz_mm']
    px_plane = stu['pix_plane']
    px_valid = stu['pix_valid'].bool()
    px_feat = stu['pix_feat']
    px_ps = stu['pix_plane_summary']

    # 1. Whole-silicon-plane masking (2-3 active inner silicon planes, strictly 0.0 coord leakage!)
    plane_active = torch.zeros((bsz, N_PIX_PLANES), dtype=torch.bool, device=dev)
    for p in range(N_PIX_PLANES):
        plane_active[:, p] = (px_valid & (px_plane == p)).any(dim=1)

    pix_plane_drop = torch.zeros((bsz, N_PIX_PLANES), dtype=torch.bool, device=dev)
    for bi in range(bsz):
        act_inner = torch.nonzero(plane_active[bi, 1:11], as_tuple=True)[0] + 1  # inner planes 1..10
        if len(act_inner) == 0:
            act_inner = torch.nonzero(plane_active[bi], as_tuple=True)[0]
        if len(act_inner) >= 2:
            n_drop = min(len(act_inner) - 1, int(torch.randint(c.plane_drop_min, c.plane_drop_max + 1, (1,), generator=generator, device=dev).item()))
            perm = torch.randperm(len(act_inner), generator=generator, device=dev)[:n_drop]
            pix_plane_drop[bi, act_inner[perm]] = True
        elif len(act_inner) == 1:
            pix_plane_drop[bi, act_inner[0]] = True

    cl_in_dropped_plane = px_valid & torch.gather(pix_plane_drop, 1, px_plane.clamp(0, N_PIX_PLANES - 1))

    # 2. Anisotropic Hilbert local patch masking (mask 45% contiguous block on surviving clusters)
    cl_hilbert_drop = torch.zeros((bsz, n_cl), dtype=torch.bool, device=dev)
    for bi in range(bsz):
        surv = torch.nonzero(px_valid[bi] & (~cl_in_dropped_plane[bi]), as_tuple=True)[0]
        if len(surv) >= 4:
            x0_proxy = px_plane[bi, surv].float() * 0.5
            hkey = _anisotropic_hilbert_order_2d3d(px_xyz[bi, surv, 0], px_xyz[bi, surv, 1], x0_proxy)
            order = torch.argsort(hkey)
            n_mask = max(1, int(round(c.hilbert_mask_frac * len(surv))))
            start = int(torch.randint(0, max(1, len(surv) - n_mask + 1), (1,), generator=generator, device=dev).item())
            cl_hilbert_drop[bi, surv[order[start : start + n_mask]]] = True

    cl_dropped = cl_in_dropped_plane | cl_hilbert_drop

    # Zero out ALL coordinates, features, plane indices, plane summaries, and patches for dropped clusters!
    px_valid_new = px_valid & (~cl_dropped)
    px_xyz[cl_dropped] = 0.0
    px_feat[cl_dropped] = 0.0
    px_plane[cl_dropped] = -1
    stu['pix_valid'] = px_valid_new

    for p in range(N_PIX_PLANES):
        pm = pix_plane_drop[:, p]
        if pm.any():
            px_ps[pm, p] = 0.0

    if 'patch_cluster_idx' in stu and 'pix_patches' in stu:
        pci = stu['patch_cluster_idx']
        pat_valid = pci >= 0
        pat_drop = torch.zeros_like(pat_valid)
        if pat_valid.any():
            cl_idx_safe = pci.clamp(0, max(0, n_cl - 1))
            pat_drop = pat_valid & torch.gather(cl_dropped, 1, cl_idx_safe)
            stu['pix_patches'][pat_drop] = 0.0
            pci[pat_drop] = -1

    # 3. Cross-plane global coordinate perturbation (+-60 um = +-0.060 mm) on surviving clusters
    jitter_mm = float(c.coord_jitter_um) * 1e-3
    eps_xy = (torch.rand((bsz, 1, 2), generator=generator, device=dev) * 2.0 - 1.0) * jitter_mm
    surv_mask = stu['pix_valid'].unsqueeze(-1).float()
    px_xyz[:, :, :2] = px_xyz[:, :, :2] + eps_xy * surv_mask
    px_feat[:, :, 0] = px_feat[:, :, 0] + (eps_xy[:, :, 0] / 135.0) * surv_mask.squeeze(-1)
    px_feat[:, :, 1] = px_feat[:, :, 1] + (eps_xy[:, :, 1] / 100.0) * surv_mask.squeeze(-1)

    # 4. 54-station stereo panel drop (drop X or Y on 30% of doublet stations)
    st_drop = torch.zeros((bsz, N_CAL_STATIONS, 2), dtype=torch.bool, device=dev)
    r_st = torch.rand((bsz, N_CAL_STATIONS), generator=generator, device=dev) < float(c.stereo_drop_prob)
    choose_y = torch.rand((bsz, N_CAL_STATIONS), generator=generator, device=dev) < 0.5
    # Only drop one side on doublet stations (stations with both X and Y active in scint_grid)
    sg = stu['scint_grid']
    has_y_st = (sg[:, :, 1, :, 0] > 0.0).any(dim=-1)
    st_drop[:, :, 0] = r_st & has_y_st & (~choose_y)
    st_drop[:, :, 1] = r_st & has_y_st & choose_y
    sg[st_drop] = 0.0

    # 5. Cross-Magnet-0 whole spectrometer drop on 50% of events
    spec_drop = torch.rand((bsz,), generator=generator, device=dev) < float(c.spec_drop_prob)
    if bsz > 0 and not spec_drop.any():
        spec_drop[0] = True
    if 'muon_kalman_tracks' in stu and stu['muon_kalman_tracks'] is not None:
        stu['muon_kalman_tracks'][spec_drop] = 0.0
    if 'muon_kalman_mask' in stu and stu['muon_kalman_mask'] is not None:
        stu['muon_kalman_mask'][spec_drop] = False
    if 'trk_cand' in stu and stu['trk_cand'] is not None:
        stu['trk_cand'][spec_drop] = 0.0
    if 'trk_cand_mask' in stu and stu['trk_cand_mask'] is not None:
        stu['trk_cand_mask'][spec_drop] = False
    if 'trk_station_counts' in stu and stu['trk_station_counts'] is not None:
        stu['trk_station_counts'][spec_drop] = 0.0

    stu['c_obs'] = _recompute_c_obs_from_clusters_and_bars(
        stu['c_obs'], px_xyz, px_plane, stu['pix_valid'], px_feat[:, :, 26], spec_drop
    )

    masks = {
        'pix_plane_drop_mask': pix_plane_drop,
        'cl_dropped_plane': cl_in_dropped_plane,
        'cl_hilbert_drop': cl_hilbert_drop,
        'cl_dropped_any': cl_dropped,
        'stereo_panel_drop_mask': st_drop,
        'spec_drop_mask': spec_drop,
    }
    return stu, masks


# ---------------------------------------------------------------------------
# PinpointSSLv2 Pretrainer Wrapper (Direct Full-Hierarchy Backbone + SSL Heads)
# ---------------------------------------------------------------------------
class PinpointSSLv2Model(nn.Module):
    """Wraps a `PinpointGen4Net` backbone with decoder-free Sonata-MS, SIGReg,
    15 um cross-plane ray interpolation, 54-station calorimeter, cross-M0 curvature,
    and pre-truncation persistence gate predictors.
    """

    def __init__(
        self,
        backbone: nn.Module,
        cfg: SonataSSLConfig | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg or SonataSSLConfig()

        # Sonata-MS 3-scale prototype heads
        self.proto_micro = SonataPrototypeHead(self.cfg.dim_micro, self.cfg.proj_dim, self.cfg.num_prototypes)
        self.proto_meso = SonataPrototypeHead(self.cfg.dim_meso, self.cfg.proj_dim, self.cfg.num_prototypes)
        self.proto_macro = SonataPrototypeHead(self.cfg.dim_macro, self.cfg.proj_dim, self.cfg.num_prototypes)

        self.sigreg = LeJEPASIGReg(self.cfg.sigreg_num_proj, self.cfg.sigreg_gh_nodes)

        # Task (c1): 15 um Cross-Plane Ray Penetration + Heteroscedastic Student-t(nu=3) + 4D Patch Morph
        self.ray_query_enc = nn.Sequential(
            nn.Linear(8, self.cfg.dim_meso),
            nn.LayerNorm(self.cfg.dim_meso),
            nn.GELU(),
            nn.Linear(self.cfg.dim_meso, self.cfg.dim_meso),
        )
        self.ray_cross_attn = nn.MultiheadAttention(self.cfg.dim_meso, num_heads=6, batch_first=True)
        self.ray_head = nn.Sequential(
            nn.LayerNorm(self.cfg.dim_meso),
            nn.Linear(self.cfg.dim_meso, self.cfg.dim_meso),
            nn.GELU(),
            nn.Linear(self.cfg.dim_meso, 15),  # [cont_logit, dx, dy, log_sx, log_sy, morph_4, edep_6]
        )

        # Task (c2): 54-Station Calorimeter Energy Conservation + Bar Profile + Delayed Time Tail
        self.cal_pred_head = nn.Sequential(
            nn.LayerNorm(self.cfg.dim_meso),
            nn.Linear(self.cfg.dim_meso, self.cfg.dim_meso),
            nn.GELU(),
            nn.Linear(self.cfg.dim_meso, 42),  # 40 log-PE bars + 1 total log-PE + 1 delayed_tail_logit
        )

        # Task (c3): Cross-Magnet-0 Signed Curvature kappa_y 2-Component GMM Extrapolation
        # Predicts bore penetration logit + 2 components * (pi_logit, 5 means [x0, y0, tx, ty, kappa_y], 5 log_sigs)
        self.spec_pred_head = nn.Sequential(
            nn.LayerNorm(self.cfg.dim_macro),
            nn.Linear(self.cfg.dim_macro, self.cfg.dim_macro),
            nn.GELU(),
            nn.Linear(self.cfg.dim_macro, 1 + 2 * 11),
        )

        # Task (d): Differentiable Pre-Truncation Persistence Gate g_i^persist
        self.persist_gate = nn.Sequential(
            nn.Linear(8, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def encode_hierarchy(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run full PinpointGen4Net hierarchy (`si_vtx`, `spec_ipt`, `cal_pflow`, `fusion`) without head."""
        if hasattr(self.backbone, 'forward_features'):
            return self.backbone.forward_features(batch)
        micro_out = self.backbone.si_vtx(batch)
        spec_out = self.backbone.spec_ipt(batch, micro_out)
        pflow_out = self.backbone.cal_pflow(batch, micro_out, spec_out)
        z_fuse = self.backbone.fusion(micro_out['z_si_vtx'], pflow_out['z_cal_pflow'], spec_out['z_spec_ipt'])
        return {
            'micro_out': micro_out,
            'spec_out': spec_out,
            'pflow_out': pflow_out,
            'u_micro': micro_out['u_micro'],
            's_si_meso': micro_out['s_si_meso'],
            's_cal_meso': pflow_out['s_cal_meso'],
            'z_fuse': z_fuse,
            'seg_mask': micro_out['seg_mask'],
        }

    def export_backbone_checkpoint_dict(self) -> dict[str, torch.Tensor]:
        """Export contracted `backbone_state_dict` keys (`si_vtx.`, `cal_pflow.`, `spec_ipt.`, `fusion.`)."""
        prefixes = ('si_vtx.', 'cal_pflow.', 'spec_ipt.', 'fusion.')
        return {
            k: v.detach().cpu().clone()
            for k, v in self.backbone.state_dict().items()
            if k.startswith(prefixes)
        }


@torch.no_grad()
def _build_15um_interpolated_rays(
    batch_ref: Mapping[str, torch.Tensor],
    pix_plane_drop: torch.Tensor,
    *,
    max_rays_per_event: int = 32,
) -> dict[str, torch.Tensor]:
    """Construct pure-observable cross-plane ray interpolation targets on whole-plane dropped planes."""
    px = batch_ref['pix_xyz_mm'].float()
    pp = batch_ref['pix_plane'].long()
    pv = batch_ref['pix_valid'].bool()
    pf = batch_ref['pix_feat'].float()
    dev = px.device
    bsz = px.shape[0]

    q_feats = torch.zeros((bsz, max_rays_per_event, 8), dtype=torch.float32, device=dev)
    q_mask = torch.zeros((bsz, max_rays_per_event), dtype=torch.bool, device=dev)
    tgt_cont = torch.zeros((bsz, max_rays_per_event), dtype=torch.float32, device=dev)
    tgt_dxy = torch.zeros((bsz, max_rays_per_event, 2), dtype=torch.float32, device=dev)
    tgt_morph = torch.zeros((bsz, max_rays_per_event, 4), dtype=torch.float32, device=dev)
    tgt_edep = torch.zeros((bsz, max_rays_per_event, 6), dtype=torch.float32, device=dev)

    for bi in range(bsz):
        dropped_p = torch.nonzero(pix_plane_drop[bi], as_tuple=True)[0].tolist()
        if not dropped_p:
            continue
        cnt = 0
        for p in dropped_p:
            if p <= 0 or p >= N_PIX_PLANES - 1 or cnt >= max_rays_per_event:
                continue
            m_prev = pv[bi] & (pp[bi] == p - 1)
            m_next = pv[bi] & (pp[bi] == p + 1)
            m_mid = pv[bi] & (pp[bi] == p)
            if not m_prev.any() or not m_next.any():
                continue

            xyz_a = px[bi, m_prev][:12]
            xyz_b = px[bi, m_next][:12]
            z_mid_val = float(px[bi, m_mid, 2].mean().item()) if m_mid.any() else 0.5 * (float(xyz_a[0, 2].item()) + float(xyz_b[0, 2].item()))
            dz_ab = (xyz_b[0, 2] - xyz_a[0, 2]).clamp_min(1.0)
            t_frac = (z_mid_val - xyz_a[0, 2]) / dz_ab

            # Pairwise slopes within physical cone (slope < 0.16)
            dx = xyz_b[None, :, 0] - xyz_a[:, None, 0]
            dy = xyz_b[None, :, 1] - xyz_a[:, None, 1]
            slope = torch.hypot(dx, dy) / dz_ab
            cand = torch.nonzero(slope < 0.16, as_tuple=False)
            if len(cand) == 0:
                continue
            cand = cand[: max_rays_per_event - cnt]

            ia, ib = cand[:, 0], cand[:, 1]
            x_proj = xyz_a[ia, 0] + t_frac * (xyz_b[ib, 0] - xyz_a[ia, 0])
            y_proj = xyz_a[ia, 1] + t_frac * (xyz_b[ib, 1] - xyz_a[ia, 1])
            tx_r = (xyz_b[ib, 0] - xyz_a[ia, 0]) / dz_ab
            ty_r = (xyz_b[ib, 1] - xyz_a[ia, 1]) / dz_ab

            n_add = len(cand)
            q_feats[bi, cnt : cnt + n_add, 0] = x_proj / 135.0
            q_feats[bi, cnt : cnt + n_add, 1] = (y_proj + 12.0) / 100.0
            q_feats[bi, cnt : cnt + n_add, 2] = z_mid_val / 1000.0
            q_feats[bi, cnt : cnt + n_add, 3] = float(p) / 15.0
            q_feats[bi, cnt : cnt + n_add, 4] = tx_r * 5.0
            q_feats[bi, cnt : cnt + n_add, 5] = ty_r * 5.0
            q_feats[bi, cnt : cnt + n_add, 6] = dz_ab / 200.0
            q_feats[bi, cnt : cnt + n_add, 7] = 1.0
            q_mask[bi, cnt : cnt + n_add] = True

            if m_mid.any():
                xyz_c = px[bi, m_mid]
                pf_c = pf[bi, m_mid]
                dist_c = torch.hypot(
                    xyz_c[None, :, 0] - x_proj[:, None],
                    xyz_c[None, :, 1] - y_proj[:, None],
                )
                min_d, arg_c = dist_c.min(dim=-1)
                # 3*sigma_MCS tolerance (~0.25 mm for inner Target)
                cont = min_d <= 0.25
                tgt_cont[bi, cnt : cnt + n_add] = cont.float()
                tgt_dxy[bi, cnt : cnt + n_add, 0] = (xyz_c[arg_c, 0] - x_proj).clamp(-0.5, 0.5)
                tgt_dxy[bi, cnt : cnt + n_add, 1] = (xyz_c[arg_c, 1] - y_proj).clamp(-0.5, 0.5)
                tgt_morph[bi, cnt : cnt + n_add] = pf_c[arg_c, 22:26]
                if pf_c.shape[-1] >= 36:
                    tgt_edep[bi, cnt : cnt + n_add] = pf_c[arg_c, 30:36]
                elif pf_c.shape[-1] >= 32:
                    tgt_edep[bi, cnt : cnt + n_add, :2] = pf_c[arg_c, 30:32]
            cnt += n_add

    return {
        'q_feats': q_feats,
        'q_mask': q_mask,
        'tgt_cont': tgt_cont,
        'tgt_dxy': tgt_dxy,
        'tgt_morph': tgt_morph,
        'tgt_edep': tgt_edep,
    }


def compute_sonata_v2_ssl_loss(
    student: PinpointSSLv2Model,
    teacher_backbone: nn.Module,
    teacher_proto_heads: Mapping[str, SonataPrototypeHead],
    batch_ref: Mapping[str, torch.Tensor],
    *,
    cfg: SonataSSLConfig | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute complete PinpointSSL-v2 self-supervised objective in FP32."""
    c = cfg or student.cfg
    dev = batch_ref['c_obs'].device

    # Preserve batch's `has_subthr_readout` and `has_edep` directly for MC 10085-10088 compatibility
    clean_ref = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch_ref.items()}

    # 1. Clean Teacher Forward (no grad)
    with torch.no_grad():
        if hasattr(teacher_backbone, 'forward_features'):
            tea_f = teacher_backbone.forward_features(clean_ref)
        else:
            m_t = teacher_backbone.si_vtx(clean_ref)
            s_t = teacher_backbone.spec_ipt(clean_ref, m_t)
            p_t = teacher_backbone.cal_pflow(clean_ref, m_t, s_t)
            zf_t = teacher_backbone.fusion(m_t['z_si_vtx'], p_t['z_cal_pflow'], s_t['z_spec_ipt'])
            tea_f = {
                'u_micro': m_t['u_micro'],
                's_si_meso': m_t['s_si_meso'],
                's_cal_meso': p_t['s_cal_meso'],
                'z_fuse': zf_t,
                'seg_mask': m_t['seg_mask'],
            }
        tea_meso = torch.cat([tea_f['s_si_meso'], tea_f['s_cal_meso']], dim=1)
        _, tea_lg_micro = teacher_proto_heads['micro'](tea_f['u_micro'])
        _, tea_lg_meso = teacher_proto_heads['meso'](tea_meso)
        _, tea_lg_macro = teacher_proto_heads['macro'](tea_f['z_fuse'])

    # 2. Student Masked View + Student Forward
    stu_view, masks = build_sonata_student_view(clean_ref, cfg=c, generator=generator)
    stu_f = student.encode_hierarchy(stu_view)
    stu_meso = torch.cat([stu_f['s_si_meso'], stu_f['s_cal_meso']], dim=1)

    with torch.amp.autocast(device_type=dev.type, enabled=False):
        # ------------------------------------------------------------------
        # (a) Sonata-MS 3-Scale Sinkhorn Prototype Self-Distillation
        # ------------------------------------------------------------------
        z_stu_micro, stu_lg_micro = student.proto_micro(stu_f['u_micro'])
        z_stu_meso, stu_lg_meso = student.proto_meso(stu_meso)
        z_stu_macro, stu_lg_macro = student.proto_macro(stu_f['z_fuse'])

        q_micro = sinkhorn_knopp_teacher(tea_lg_micro.reshape(-1, c.num_prototypes), temp=c.teacher_temp, n_iters=c.sinkhorn_iters).view_as(tea_lg_micro)
        q_meso = sinkhorn_knopp_teacher(tea_lg_meso.reshape(-1, c.num_prototypes), temp=c.teacher_temp, n_iters=c.sinkhorn_iters).view_as(tea_lg_meso)
        q_macro = sinkhorn_knopp_teacher(tea_lg_macro, temp=c.teacher_temp, n_iters=c.sinkhorn_iters)

        # Micro loss over valid segments
        sm_u = stu_f['seg_mask'].bool() & tea_f['seg_mask'].bool()
        if sm_u.any():
            lp_u = F.log_softmax(stu_lg_micro[sm_u] / float(c.student_temp), dim=-1)
            l_son_micro = -(q_micro[sm_u] * lp_u).sum(dim=-1).mean()
        else:
            l_son_micro = stu_lg_micro.sum() * 0.0

        # Meso loss weighted toward masked planes/stations
        meso_m = torch.cat([masks['pix_plane_drop_mask'], masks['stereo_panel_drop_mask'].any(dim=-1)], dim=1)
        if meso_m.any():
            lp_m = F.log_softmax(stu_lg_meso[meso_m] / float(c.student_temp), dim=-1)
            l_son_meso = -(q_meso[meso_m] * lp_m).sum(dim=-1).mean()
        else:
            lp_m = F.log_softmax(stu_lg_meso / float(c.student_temp), dim=-1)
            l_son_meso = -(q_meso * lp_m).sum(dim=-1).mean()

        lp_g = F.log_softmax(stu_lg_macro / float(c.student_temp), dim=-1)
        l_son_macro = -(q_macro * lp_g).sum(dim=-1).mean()

        l_sonata = 0.45 * l_son_micro + 0.30 * l_son_meso + 0.25 * l_son_macro

        # ------------------------------------------------------------------
        # (b) O(N) LeJEPA SIGReg + Intra-Event KoLeo Repulsion
        # ------------------------------------------------------------------
        l_sig_macro = student.sigreg(stu_f['z_fuse'])
        l_sig_cal = student.sigreg(stu_f['s_cal_meso'].reshape(-1, c.dim_meso))
        if sm_u.any():
            l_sig_micro = student.sigreg(stu_f['u_micro'][sm_u])
        else:
            l_sig_micro = stu_f['u_micro'].sum() * 0.0
        l_sigreg = (l_sig_macro + l_sig_cal + l_sig_micro) / 3.0

        l_koleo = intra_event_koleo_loss(stu_f['u_micro'], stu_f['seg_mask'], max_k=64)

        # ------------------------------------------------------------------
        # (c1) 15 um Cross-Plane Ray Penetration + Student-t + 4D Morph
        # ------------------------------------------------------------------
        ray_tgt = _build_15um_interpolated_rays(clean_ref, masks['pix_plane_drop_mask'])
        qm = ray_tgt['q_mask']
        if qm.any():
            q_emb = student.ray_query_enc(ray_tgt['q_feats'])
            ctx, _ = student.ray_cross_attn(q_emb, stu_meso, stu_meso)
            rp = student.ray_head(q_emb + ctx)  # (B, R, 9)
            rp_v = rp[qm]
            cont_tgt = ray_tgt['tgt_cont'][qm]
            l_cont = F.binary_cross_entropy_with_logits(rp_v[:, 0], cont_tgt)

            pos_r = cont_tgt > 0.5
            if pos_r.any():
                dxy_tgt = ray_tgt['tgt_dxy'][qm][pos_r]
                dxy_pred = 0.5 * torch.tanh(rp_v[pos_r, 1:3])
                # 15 um = 0.015 mm lower bound! (45x sharper than old 675 um 8-GMM)
                sig_xy = F.softplus(rp_v[pos_r, 3:5]).clamp(min=0.015, max=2.0)
                nu = 3.0
                z2 = ((dxy_pred - dxy_tgt) / sig_xy).square()
                l_ray_t = (0.5 * (nu + 1.0) * torch.log1p(z2 / nu) + torch.log(sig_xy)).mean()
                l_morph = F.smooth_l1_loss(rp_v[pos_r, 5:9], ray_tgt['tgt_morph'][qm][pos_r], beta=0.2)
                has_edep_b = (
                    clean_ref['has_edep'].bool().unsqueeze(-1).expand_as(qm)
                    if 'has_edep' in clean_ref
                    else torch.zeros_like(qm)
                )
                pos_edep = has_edep_b[qm][pos_r]
                edep_dim = 6 if clean_ref['pix_feat'].shape[-1] >= 36 else (2 if clean_ref['pix_feat'].shape[-1] >= 32 else 0)
                if pos_edep.any() and edep_dim > 0:
                    l_edep_ssl = F.smooth_l1_loss(
                        rp_v[pos_r, 9 : 9 + edep_dim][pos_edep],
                        ray_tgt['tgt_edep'][qm][pos_r][:, :edep_dim][pos_edep],
                        beta=0.2,
                    )
                else:
                    l_edep_ssl = rp.sum() * 0.0
            else:
                l_ray_t = rp.sum() * 0.0
                l_morph = rp.sum() * 0.0
                l_edep_ssl = rp.sum() * 0.0
            l_phys_ray = l_cont + 0.5 * l_ray_t + 0.35 * l_morph + 0.35 * l_edep_ssl
        else:
            l_phys_ray = stu_meso.sum() * 0.0

        # ------------------------------------------------------------------
        # (c2) 54-Station Calorimeter Conservation + Delayed Neutron Time Tail
        # ------------------------------------------------------------------
        st_drop_any = masks['stereo_panel_drop_mask'].any(dim=-1)
        sg_ref = clean_ref['scint_grid'].float()  # (B, 54, 2, 40, 10)
        # Predict dropped panel side log-energy, station total energy, and delayed tail (>2.5 ns)
        cal_pred = student.cal_pred_head(stu_f['s_cal_meso'])  # (B, 54, 42)
        if st_drop_any.any():
            drop_x = masks['stereo_panel_drop_mask'][:, :, 0]
            drop_y = masks['stereo_panel_drop_mask'][:, :, 1]
            # Target panel is the dropped side (0 if drop_x else 1)
            tgt_side = torch.where(drop_x[:, :, None, None], sg_ref[:, :, 0, :, :], sg_ref[:, :, 1, :, :])  # (B, 54, 40, 10)
            tgt_log_e = torch.log1p(tgt_side[..., 0].clamp_min(0.0))  # (B, 54, 40)
            tgt_tot_log_e = torch.log1p(tgt_side[..., 0].sum(dim=-1).clamp_min(0.0))  # (B, 54)
            tgt_delayed = (tgt_side[..., 9] > 0.5).any(dim=-1).float()  # (B, 54)

            m_st = st_drop_any & ((tgt_side[..., 0] > 0.0).any(dim=-1))
            if m_st.any():
                cp_m = cal_pred[m_st]
                l_bar = F.smooth_l1_loss(cp_m[:, :40], tgt_log_e[m_st], beta=0.25)
                l_cons = F.smooth_l1_loss(cp_m[:, 40], tgt_tot_log_e[m_st], beta=0.25)
                l_del = F.binary_cross_entropy_with_logits(cp_m[:, 41], tgt_delayed[m_st])
                l_phys_cal = l_bar + 0.5 * l_cons + 0.35 * l_del
            else:
                l_phys_cal = cal_pred.sum() * 0.0
        else:
            l_phys_cal = cal_pred.sum() * 0.0

        # ------------------------------------------------------------------
        # (c3) Cross-Magnet-0 Signed Curvature kappa_y Extrapolation (2-GMM)
        # ------------------------------------------------------------------
        sd_mask = masks['spec_drop_mask']
        sp_out = student.spec_pred_head(stu_f['z_fuse'])  # (B, 23)
        tc = clean_ref['trk_cand'].float()
        tcm = clean_ref['trk_cand_mask'].bool()
        has_3st_bore = tcm[:, 0] & (tc[:, 0, 0] > 0.75)

        if sd_mask.any():
            l_sp_bce = F.binary_cross_entropy_with_logits(sp_out[sd_mask, 0], has_3st_bore[sd_mask].float())
            pos_sp = sd_mask & has_3st_bore
            if pos_sp.any():
                # Signed curvature kappa_y = -s_y / 590000.0 (preserved sign!)
                sy_mm = SIGMA_SAGITTA_Y_MM * torch.sinh((5.0 * tc[pos_sp, 0, 6]).clamp(-9.0, 9.0))
                kappa_y_scaled = (-sy_mm / 590000.0) * 1e4  # O(1) scaled mm^-1 * 1e4
                tgt_k = torch.stack([
                    tc[pos_sp, 0, 2],
                    tc[pos_sp, 0, 3],
                    tc[pos_sp, 0, 4],
                    tc[pos_sp, 0, 5],
                    kappa_y_scaled.clamp(-10.0, 10.0),
                ], dim=-1)  # (Np, 5)
                gmm_raw = sp_out[pos_sp, 1:].view(-1, 2, 11)
                log_pi = F.log_softmax(gmm_raw[..., 0], dim=-1)  # (Np, 2)
                mu_k = gmm_raw[..., 1:6]                         # (Np, 2, 5)
                sg_k = F.softplus(gmm_raw[..., 6:11]).clamp(min=0.05, max=5.0)
                diff_k = (tgt_k.unsqueeze(1) - mu_k) / sg_k
                log_g = -0.5 * diff_k.square().sum(dim=-1) - torch.log(sg_k).sum(dim=-1) - 2.5 * math.log(2.0 * math.pi)
                l_sp_gmm = -torch.logsumexp(log_pi + log_g, dim=-1).mean()
            else:
                l_sp_gmm = sp_out.sum() * 0.0
            l_phys_spec = l_sp_bce + 0.5 * l_sp_gmm
        else:
            l_phys_spec = sp_out.sum() * 0.0

        # ------------------------------------------------------------------
        # (d) Pre-Truncation Dark-Noise & Dead-Channel Consistency L_HW-Inv
        # ------------------------------------------------------------------
        deg_batch, is_orig_cl, hw_active = inject_online_dark_noise_and_degradation(
            clean_ref, dark_prob=c.dark_noise_prob, deg_prob=c.hw_deg_prob, generator=generator
        )
        if hw_active.any():
            deg_f = student.encode_hierarchy(deg_batch)
            # Persistence gate features from purely observable local cluster geometry & per-plane 1-pix density
            pf_d = deg_batch['pix_feat'].float()
            pp_d = deg_batch['pix_plane'].long().clamp(0, N_PIX_PLANES - 1)
            pv_d = deg_batch['pix_valid'].bool()
            is_1pix = pv_d & (pf_d[..., 3] <= (math.log1p(1.1) / 6.0))
            n1_plane = torch.zeros((pf_d.shape[0], N_PIX_PLANES), dtype=torch.float32, device=dev)
            for p_i in range(N_PIX_PLANES):
                n1_plane[:, p_i] = (is_1pix & (pp_d == p_i)).float().sum(dim=-1)
            log_n1_cl = torch.log1p(torch.gather(n1_plane, 1, pp_d)) / 6.0
            gate_in = torch.cat([
                pf_d[..., 0:4],
                pf_d[..., 18:20],
                is_1pix.float().unsqueeze(-1),
                log_n1_cl.unsqueeze(-1),
            ], dim=-1)  # (B, N_cl, 8)
            g_logit = student.persist_gate(gate_in).squeeze(-1)  # (B, N_cl)

            m_hw_cl = pv_d & hw_active.unsqueeze(-1)
            if m_hw_cl.any():
                l_gate = F.binary_cross_entropy_with_logits(g_logit[m_hw_cl], is_orig_cl[m_hw_cl].float())
            else:
                l_gate = g_logit.sum() * 0.0

            z_d = F.normalize(deg_f['z_fuse'].float(), p=2, dim=-1, eps=1e-6)
            z_r = F.normalize(tea_f['z_fuse'].detach().float(), p=2, dim=-1, eps=1e-6)
            l_inv_glob = (z_d[hw_active] - z_r[hw_active]).square().sum(dim=-1).mean()

            sm_hw = deg_f['seg_mask'].bool() & tea_f['seg_mask'].bool() & hw_active.unsqueeze(-1)
            if sm_hw.any():
                u_d = F.normalize(deg_f['u_micro'][sm_hw].float(), p=2, dim=-1, eps=1e-6)
                u_r = F.normalize(tea_f['u_micro'][sm_hw].detach().float(), p=2, dim=-1, eps=1e-6)
                l_inv_loc = (u_d - u_r).square().sum(dim=-1).mean()
            else:
                l_inv_loc = z_d.sum() * 0.0
            l_hw_inv = l_gate + l_inv_glob + 0.5 * l_inv_loc
        else:
            l_hw_inv = stu_f['z_fuse'].sum() * 0.0

        total = (
            float(c.w_sonata) * l_sonata
            + float(c.w_sigreg) * l_sigreg
            + float(c.w_koleo) * l_koleo
            + float(c.w_phys_ray) * l_phys_ray
            + float(c.w_phys_cal) * l_phys_cal
            + float(c.w_phys_spec) * l_phys_spec
            + float(c.w_hw_inv) * l_hw_inv
        )
        metrics = {
            'ssl_total': float(total.detach().item()),
            'ssl_sonata': float(l_sonata.detach().item()),
            'ssl_sigreg': float(l_sigreg.detach().item()),
            'ssl_koleo': float(l_koleo.detach().item()),
            'ssl_phys_ray': float(l_phys_ray.detach().item()),
            'ssl_phys_cal': float(l_phys_cal.detach().item()),
            'ssl_phys_spec': float(l_phys_spec.detach().item()),
            'ssl_hw_inv': float(l_hw_inv.detach().item()),
        }
        return total, metrics
