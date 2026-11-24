"""Stage-I Causal Ablation & Stage-II 3-Seed Supervised Runner for PinpointGen4Net.

Strictly targets the new official MC (`10085-10088`) cache once `[gen4-contract-v2]` and
`data_cache.py` are ready (never reads deprecated `v6_clean_v3` / `10081-10084` / `l2_ssl_300k`).
Enforces `SealAuditGuard` zero-access on sealed shards/chunks and `verify_inference_truth_invariance`.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import pickle
import re
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from analysis.flavour.gen4.gen4_unified_model import (
    GEN4_TRUTH_BATCH_KEYS,
    PinpointGen4Net,
    compute_gen4_supervised_loss,
    extract_nsbi_observables_from_output,
)
from analysis.flavour.gen4.geometry_contract import Gen4GeometryContract, load_geometry_contract
from analysis.flavour.inventory import write_json
from analysis.flavour.nextgen.protocols import (
    check_branch_gradient_health,
    compute_student_t_pulls,
)
from analysis.flavour.nextgen.seal_guard import (
    SealAuditGuard,
    SealViolationError,
    assert_unsealed_path,
    assert_unsealed_split,
    verify_inference_truth_invariance,
)
from analysis.flavour.v2.partitions import FAMILY, KEY


GEN4_RESULTS_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage329_gen4_mc8588')
GEN4_CACHE_DIR = GEN4_RESULTS_ROOT / 'cache'
EXP_SUP_DIR = GEN4_RESULTS_ROOT / 'exp_gen4_supervised'
CONDOR_SUP_DIR = Path.home() / 'condor/Pinpoint/stage329_gen4_sup'
STAGE2_SEEDS: tuple[int, ...] = (17, 42, 2026)

# Causal ablation configs on Stage-I (Val, seed=42)
ABLATION_CONFIGS: dict[str, dict[str, Any]] = {
    'gen4_full': {
        'enable_sandwich_slab': True,
        'enable_comp_ndive': True,
        'enable_kink_stubs': True,
        'enable_m2_kalman': True,
        'enable_sinkhorn_fmpe': True,
        'force_binary_no_edep': False,
    },
    'abl_binary_no_edep': {
        'enable_sandwich_slab': True,
        'enable_comp_ndive': True,
        'enable_kink_stubs': True,
        'enable_m2_kalman': True,
        'enable_sinkhorn_fmpe': True,
        'force_binary_no_edep': True,
    },
    'abl_no_sandwich_no_ndive': {
        'enable_sandwich_slab': False,
        'enable_comp_ndive': False,
        'enable_kink_stubs': True,
        'enable_m2_kalman': True,
        'enable_sinkhorn_fmpe': True,
        'force_binary_no_edep': False,
    },
    'abl_no_m2_no_kink_stub': {
        'enable_sandwich_slab': True,
        'enable_comp_ndive': True,
        'enable_kink_stubs': False,
        'enable_m2_kalman': False,
        'enable_sinkhorn_fmpe': True,
        'force_binary_no_edep': False,
    },
    'abl_no_desat_sinkhorn_fmpe': {
        'enable_sandwich_slab': True,
        'enable_comp_ndive': True,
        'enable_kink_stubs': True,
        'enable_m2_kalman': True,
        'enable_sinkhorn_fmpe': False,
        'force_binary_no_edep': False,
    },
}

_STAGE1_FORBIDDEN_RE = re.compile(
    r'(?:^|/)shards/0(?:8\d|9\d)(?:/|$)|validation_meta|tune_screen|tune_truth_meta|tune8k|tune_eval',
    re.IGNORECASE,
)
_STAGE1_HOOK_INSTALLED = False


def install_stage1_only_hook() -> None:
    """Block Tune8k / Tune_Eval filesystem syscalls during Stage-I exploration."""
    global _STAGE1_HOOK_INSTALLED
    if _STAGE1_HOOK_INSTALLED:
        return

    def _hook(event: str, args: tuple[Any, ...]) -> None:
        if event not in ('open', 'os.stat', 'os.scandir', 'os.listdir') or not args:
            return
        target = args[0]
        if isinstance(target, int):
            return
        try:
            p = str(target)
        except Exception:
            return
        if _STAGE1_FORBIDDEN_RE.search(p.replace('\\', '/')):
            raise SealViolationError(f'Stage-I forbids accessing Tune/Eval path: {p}')

    sys.addaudithook(_hook)
    _STAGE1_HOOK_INSTALLED = True


def _assert_not_deprecated_mc(path: Path | str) -> None:
    low = str(path).lower()
    for bad in ('v6_clean_v3', 'l2_ssl_300k', 'l1_cluster_sidecar'):
        if bad in low:
            raise ValueError(f'Deprecated pre-10085-10088 dataset/cache forbidden by directive #44: {path}')


def make_synthetic_gen4_batch(
    batch_size: int = 4,
    n_cl: int = 128,
    *,
    has_edep: float = 1.0,
    fortune_only: bool = False,
    empty_event: bool = False,
    device: torch.device | str = 'cpu',
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Build a physically consistent synthetic Gen4EventBatch for Gate-A/B & overfit health checks."""
    rng = np.random.default_rng(seed)
    dev = torch.device(device)
    b = int(batch_size)

    c_obs = torch.zeros((b, 16), dtype=torch.float32, device=dev)
    if empty_event:
        fortune_only = True
        obs_dom = torch.zeros((b,), dtype=torch.long, device=dev)
    else:
        c_obs[:, 0] = 0.0 if fortune_only else 1.0
        c_obs[:, 1] = 1.0
        c_obs[:, 2] = 1.0
        obs_dom = torch.full((b,), 2 if fortune_only else 1, dtype=torch.long, device=dev)

    if fortune_only:
        n_cl = 0
        pix_xyz = torch.zeros((b, 0, 3), dtype=torch.float32, device=dev)
        pix_pl = torch.zeros((b, 0), dtype=torch.long, device=dev)
        pix_val = torch.zeros((b, 0), dtype=torch.bool, device=dev)
        pix_feat = torch.zeros((b, 0, 36), dtype=torch.float32, device=dev)
        cl_orig = torch.zeros((b, 0), dtype=torch.long, device=dev)
        cl_sec = torch.zeros((b, 0), dtype=torch.long, device=dev)
        cl_frac = torch.zeros((b, 0), dtype=torch.float32, device=dev)
        cl_tid = torch.zeros((b, 0), dtype=torch.long, device=dev)
    else:
        geom = load_geometry_contract()
        z_p = torch.as_tensor(geom.pixel_Zpos, dtype=torch.float32, device=dev)
        pl = torch.randint(0, 15, (b, n_cl), device=dev)
        # Concentrate clusters near realistic shower tracks
        xy = torch.randn((b, n_cl, 2), device=dev) * 8.0
        xy[..., 1] -= 12.0
        pix_xyz = torch.stack([xy[..., 0], xy[..., 1], z_p[pl]], dim=-1)
        pix_pl = pl
        pix_val = torch.ones((b, n_cl), dtype=torch.bool, device=dev)
        pix_feat = torch.randn((b, n_cl, 36), device=dev) * 0.25
        pix_feat[..., 3] = 0.25  # ~log(nhits)/6
        pix_feat[..., 26] = 0.90  # DarkShield persist prob
        pix_feat[..., 27] = 1.0e-4
        pix_feat[..., 28] = 1.0e-4
        pix_feat[..., 29] = 0.0
        if has_edep == 0.0:
            pix_feat[..., 30:36] = 0.0
        else:
            pix_feat[..., 30] = torch.log1p(torch.rand((b, n_cl), device=dev) * 1.2 + 0.5)
            pix_feat[..., 31:36] = torch.rand((b, n_cl, 5), device=dev)
        cl_orig = torch.randint(0, 5, (b, n_cl), device=dev)
        cl_sec = torch.full((b, n_cl), -1, dtype=torch.long, device=dev)
        cl_frac = torch.ones((b, n_cl), dtype=torch.float32, device=dev)
        cl_tid = torch.randint(1, 8, (b, n_cl), device=dev)

    scint_grid = torch.zeros((b, 54, 2, 40, 10), dtype=torch.float32, device=dev)
    if not empty_event:
        # Activate realistic shower bars across stations 6..30
        for st in range(6, 30):
            scint_grid[:, st, :, 12:20, 0] = torch.rand((b, 2, 8), device=dev) * 35.0 + 1.0
            scint_grid[:, st, :, 12:20, 2] = scint_grid[:, st, :, 12:20, 0] * 350.0
            scint_grid[:, st, :, 12:20, 3] = 4.0 + torch.rand((b, 2, 8), device=dev) * 0.5
            scint_grid[:, st, :, 12:20, 5] = 1.0
            scint_grid[:, st, :, 12:20, 9] = torch.rand((b, 2, 8), device=dev) * 0.2

    mu_k = torch.zeros((b, 4, 16), dtype=torch.float32, device=dev)
    mu_km = torch.zeros((b, 4), dtype=torch.bool, device=dev)
    if not empty_event:
        mu_k[:, 0, 0] = -0.04  # q/p = -1 / 25 GeV^-1
        mu_k[:, 0, 1] = 0.005
        mu_k[:, 0, 2] = 25.0
        mu_k[:, 0, 3] = 2.5
        mu_k[:, 0, 4] = -1.0
        mu_k[:, 0, 5] = 1.0
        mu_k[:, 0, 6] = 1.0
        mu_k[:, 0, 11] = -12.0
        mu_km[:, 0] = True

    y4 = torch.tensor(rng.integers(0, 4, size=b), dtype=torch.long, device=dev)
    l7_map = torch.tensor([0, 1, 5, 6], dtype=torch.long, device=dev)

    return {
        'c_obs': c_obs,
        'obs_domain_id': obs_dom,
        'cal_start_mm': torch.tensor([[0.0, -12.0, -2850.0]], device=dev).expand(b, 3).clone(),
        'has_subthr_readout': torch.ones((b,), dtype=torch.float32, device=dev),
        'has_edep': torch.full((b,), float(has_edep), dtype=torch.float32, device=dev),
        's_mir': torch.ones((b,), dtype=torch.float32, device=dev),
        'pix_xyz_mm': pix_xyz,
        'pix_plane': pix_pl,
        'pix_valid': pix_val,
        'pix_feat': pix_feat,
        'pix_plane_summary': torch.randn((b, 15, 12), device=dev) * 0.1,
        'pix_patches': torch.zeros((b, 64, 2, 32, 32), dtype=torch.float32, device=dev),
        'patch_cluster_idx': torch.full((b, 64), -1, dtype=torch.long, device=dev),
        'scint_grid': scint_grid,
        'muon_kalman_tracks': mu_k,
        'muon_kalman_mask': mu_km,
        'trk_cand': torch.zeros((b, 8, 16), dtype=torch.float32, device=dev),
        'trk_cand_mask': torch.zeros((b, 8), dtype=torch.bool, device=dev),
        'trk_station_counts': torch.tensor([[1.0, 1.0, 1.0]], device=dev).expand(b, 3).clone(),
        # Truth & sidecar keys (used strictly in loss, never forward!)
        'label': y4,
        'label_7': l7_map[y4],
        'charm_flag': torch.zeros((b,), dtype=torch.long, device=dev),
        'charm_valid': torch.ones((b,), dtype=torch.long, device=dev),
        'vertex_xyz_mm': torch.tensor([[0.0, -12.0, -2950.0]], device=dev).expand(b, 3).clone(),
        'vertex_valid': torch.ones((b,), dtype=torch.long, device=dev),
        'vis_kinematics': torch.tensor([[5.5, 1.2, 0.8, 0.6]], device=dev).expand(b, 4).clone(),
        'kin_valid': torch.ones((b,), dtype=torch.long, device=dev),
        'cl_sidecar_valid': torch.full((b,), not fortune_only, dtype=torch.bool, device=dev),
        'cl_origin_label': cl_orig,
        'cl_secondary_origin_label': cl_sec,
        'cl_dominant_edep_frac': cl_frac,
        'cl_dominant_track_id': cl_tid,
        'true_sv_xyz_mm': torch.tensor([[0.2, -11.9, -2942.0]], device=dev).expand(b, 3).clone(),
        'has_true_sv': (y4 == 2),
        'sample_weight': torch.ones((b,), dtype=torch.float32, device=dev),
    }


def run_gate_ab_and_overfit_health(
    device: torch.device | str = 'cpu',
    steps: int = 40,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Run Gate-A truth/sidecar invariance (atol=0.0), Gate-B gradient health, and small-sample fit."""
    dev = torch.device(device)
    model = PinpointGen4Net().to(dev)
    n_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    groups = model.branch_parameter_groups()
    group_counts = {k: int(sum(p.numel() for p in v)) for k, v in groups.items()}

    # Gate-A: exact truth & sidecar tamper invariance (max_abs_diff == 0.0)
    batch_a = make_synthetic_gen4_batch(batch_size=batch_size, n_cl=64, has_edep=1.0, device=dev, seed=101)
    gate_a_base = verify_inference_truth_invariance(model, batch_a, atol=0.0)

    # Extra tamper check on Gen4-specific sidecar keys
    model.eval()
    with torch.no_grad():
        clean_b = {k: v for k, v in batch_a.items() if k not in GEN4_TRUTH_BATCH_KEYS}
        ref_out = model.forward_NextGen(clean_b)
        mut_b = dict(batch_a)
        mut_b['cl_origin_label'] = torch.full_like(batch_a['cl_origin_label'], 3)
        mut_b['true_sv_xyz_mm'] = torch.full_like(batch_a['true_sv_xyz_mm'], 999.0)
        mut_b['cl_sidecar_valid'] = torch.zeros_like(batch_a['cl_sidecar_valid'])
        mut_b['has_true_sv'] = torch.ones_like(batch_a['has_true_sv'])
        mut_out = model.forward_NextGen(mut_b)
        sidecar_diff = max(
            float((ref_out.probs4 - mut_out.probs4).abs().max().item()),
            float((ref_out.v_pv_mm - mut_out.v_pv_mm).abs().max().item()),
            float((ref_out.aux['tau_llr_score'] - mut_out.aux['tau_llr_score']).abs().max().item()),
        )
    if sidecar_diff > 0.0:
        raise AssertionError(f'Gen4 sidecar leakage detected: max_abs_diff={sidecar_diff}')

    # Gate-B: gradient health across all 5 top-level groups
    model.train()
    out_train = model.forward_NextGen(batch_a)
    total_loss, _, _, _ = compute_gen4_supervised_loss(model, out_train, batch_a)
    gate_b = check_branch_gradient_health(model, total_loss, min_grad_l2=1.0e-6)
    if not gate_b['passed']:
        raise RuntimeError(f'Gate-B gradient check failed: {gate_b}')

    if dev.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(dev)
    # Small-sample overfit check
    opt = torch.optim.AdamW(model.parameters(), lr=5.0e-4, weight_decay=1.0e-4)
    losses: list[float] = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        out_s = model.forward_NextGen(batch_a)
        loss_s, _, _, _ = compute_gen4_supervised_loss(model, out_s, batch_a)
        loss_s.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        losses.append(float(loss_s.detach().item()))

    if not math.isfinite(losses[-1]) or losses[-1] >= losses[0]:
        raise RuntimeError(f'Small-sample overfit failed to reduce loss: {losses[0]:.4f} -> {losses[-1]:.4f}')

    peak_vram_mb = float(torch.cuda.max_memory_allocated(dev) / (1024.0 ** 2)) if dev.type == 'cuda' else 0.0
    return {
        'n_params_total': n_params,
        'group_param_counts': group_counts,
        'gate_a_base': gate_a_base,
        'sidecar_max_abs_diff': sidecar_diff,
        'gate_b': gate_b,
        'overfit_loss_start': losses[0],
        'overfit_loss_end': losses[-1],
        'steps': steps,
        'peak_vram_mb': peak_vram_mb,
    }


def _collate_cached_items(
    items: list[dict[str, Any]],
    geom: Gen4GeometryContract,
    device: torch.device,
    *,
    aug_cfg: dict[str, Any] | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Use worker-foundation `data_cache.collate_gen4_batch` when available."""
    try:
        from analysis.flavour.gen4.data_cache import collate_gen4_batch
    except ImportError as exc:
        raise RuntimeError(
            'data_cache.py (`collate_gen4_batch`) from @worker-foundation is required for real-cache training.'
        ) from exc
    batch = collate_gen4_batch(items, geom, aug_cfg=aug_cfg, rng=rng)
    return {
        k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


def _annotate_eval_meta(meta_df: pd.DataFrame, geom: Gen4GeometryContract) -> pd.DataFrame:
    """Derive exact `Front30`, `Interm150`, and `pixel_centered` slice flags from `vertex_xyz_mm`."""
    df = meta_df.copy()
    if 'obs_domain_id' in df.columns and 'obs_dom' not in df.columns:
        df['obs_dom'] = df['obs_domain_id'].astype(np.int64)
    if 'vertex_xyz_mm' in df.columns:
        v = np.stack([np.asarray(x, dtype=np.float64) for x in df['vertex_xyz_mm'].tolist()], axis=0)
        vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
        in_pix_xy = (vx >= geom.px_min) & (vx <= geom.px_max) & (vy >= geom.py_min) & (vy <= geom.py_max)
        # Front30: first 6 PinpointBlocks (copy 0..5); Interm150: copy 6..11
        z_f_lo = float(min(geom.pinpoint_w_back_z[0], geom.pixel_Zpos[0])) - 6.0
        z_f_hi = float(max(geom.pinpoint_w_front_z[5], geom.pixel_Zpos[5])) + 6.0
        in_front = in_pix_xy & (vz >= z_f_lo) & (vz <= z_f_hi)
        in_interm = np.zeros(len(df), dtype=bool)
        for p_idx in range(6, min(12, len(geom.pixel_Zpos))):
            z_lo = float(min(geom.pinpoint_w_back_z[p_idx], geom.pixel_Zpos[p_idx])) - 6.0
            z_hi = float(max(geom.pinpoint_w_front_z[p_idx], geom.pixel_Zpos[p_idx])) + 6.0
            in_interm |= in_pix_xy & (vz >= z_lo) & (vz <= z_hi)
        df['front_pinpoint_pixel_centered'] = in_front
        df['interm_pinpoint_pixel_centered'] = in_interm
        df['pixel_centered'] = in_front | in_interm
        df['v_true_x_mm'] = vx
        df['v_true_y_mm'] = vy
        df['v_true_z_mm'] = vz
    return df


def _auroc_binary(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(s, kind='mergesort')
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    sum_pos = float(ranks[y].sum())
    return float((sum_pos - 0.5 * n_pos * (n_pos + 1)) / (n_pos * n_neg))


def _evaluate_predictions(
    probs4: np.ndarray,
    tau_llr: np.ndarray,
    labels4: np.ndarray,
    meta: pd.DataFrame,
    v_pv_mm: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute standardized Gen4 evaluation slices matching `[gen4-contract-v2]`."""
    y = np.asarray(labels4, dtype=np.int64)
    p = np.asarray(probs4, dtype=np.float64)
    p = p / np.maximum(p.sum(axis=-1, keepdims=True), 1e-12)
    pred = p.argmax(axis=1)
    recalls = [float(np.mean(pred[y == c] == c)) for c in range(4) if (y == c).any()]
    ba4 = float(np.mean(recalls)) * 100.0 if recalls else 0.0
    nll = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-8, 1.0)).mean())

    def _eff_at_fpr(scores: np.ndarray, mask_sig: np.ndarray, mask_bg: np.ndarray, fpr_target: float) -> float:
        s = scores
        bg = np.sort(s[mask_bg])[::-1]
        sg = s[mask_sig]
        if len(bg) == 0 or len(sg) == 0:
            return 0.0
        k_allow = int(math.floor(fpr_target * len(bg)))
        thr = float(bg[min(k_allow, len(bg) - 1)])
        return float(np.mean(sg > thr) * 100.0)

    all_m = np.ones(len(y), dtype=bool)
    res: dict[str, Any] = {
        'n_events': int(len(y)),
        'BA4_pct': ba4,
        'NLL': nll,
        'tau_1pct_fpr_probs4_pct': _eff_at_fpr(np.log(np.clip(p[:, 2], 1e-8, 1.0)), y == 2, y != 2, 0.01),
        'tau_1pct_fpr_llr_pct': _eff_at_fpr(tau_llr, y == 2, y != 2, 0.01),
        'tau_0p1pct_fpr_llr_pct': _eff_at_fpr(tau_llr, y == 2, y != 2, 0.001),
    }

    if 'obs_dom' in meta.columns:
        od = meta['obs_dom'].to_numpy(dtype=np.int64)
        for d_val, d_name in ((1, 'D1_obs'), (2, 'D2_obs'), (3, 'D3_obs')):
            dm = od == d_val
            if dm.any():
                rec_d = [float(np.mean(pred[dm & (y == c)] == c)) for c in range(4) if (dm & (y == c)).any()]
                res[f'{d_name}_BA4_pct'] = float(np.mean(rec_d)) * 100.0 if rec_d else 0.0
                res[f'{d_name}_n'] = int(dm.sum())

    if 'pixel_centered' in meta.columns:
        pc = meta['pixel_centered'].to_numpy(dtype=bool)
        if 'front_pinpoint_pixel_centered' in meta.columns:
            f30 = meta['front_pinpoint_pixel_centered'].to_numpy(dtype=bool)
            i150 = meta['interm_pinpoint_pixel_centered'].to_numpy(dtype=bool) if 'interm_pinpoint_pixel_centered' in meta.columns else (pc & (~f30))
            for m_sl, name in ((f30, 'Front30'), (i150, 'Interm150')):
                if m_sl.any():
                    rc = [float(np.mean(pred[m_sl & (y == c)] == c)) for c in range(4) if (m_sl & (y == c)).any()]
                    res[f'{name}_BA4_pct'] = float(np.mean(rc)) * 100.0 if rc else 0.0
                    res[f'{name}_n'] = int(m_sl.sum())
                    if v_pv_mm is not None and 'v_true_x_mm' in meta.columns:
                        dx = v_pv_mm[m_sl, 0] - meta['v_true_x_mm'].to_numpy(dtype=np.float64)[m_sl]
                        dy = v_pv_mm[m_sl, 1] - meta['v_true_y_mm'].to_numpy(dtype=np.float64)[m_sl]
                        dz = v_pv_mm[m_sl, 2] - meta['v_true_z_mm'].to_numpy(dtype=np.float64)[m_sl]
                        res[f'{name}_pv_transverse_median_um'] = float(np.median(np.hypot(dx, dy)) * 1000.0)
                        res[f'{name}_pv_z_median_um'] = float(np.median(np.abs(dz)) * 1000.0)
        if 'obs_dom' in meta.columns:
            fort = (~pc) & (meta['obs_dom'].to_numpy(dtype=np.int64) == 2)
            if fort.any():
                rc = [float(np.mean(pred[fort & (y == c)] == c)) for c in range(4) if (fort & (y == c)).any()]
                res['Fortune_ScintOnly_BA4_pct'] = float(np.mean(rc)) * 100.0 if rc else 0.0
                res['Fortune_ScintOnly_n'] = int(fort.sum())

    if v_pv_mm is not None and 'v_true_x_mm' in meta.columns:
        dx_a = v_pv_mm[:, 0] - meta['v_true_x_mm'].to_numpy(dtype=np.float64)
        dy_a = v_pv_mm[:, 1] - meta['v_true_y_mm'].to_numpy(dtype=np.float64)
        res['All_pv_transverse_median_um'] = float(np.median(np.hypot(dx_a, dy_a)) * 1000.0)

    # Tau decay submode & short decay-length efficiency at global 1% non-tau FPR
    if 'true_tau_submode' in meta.columns:
        tsm = meta['true_tau_submode'].to_numpy(dtype=np.int64)
        for sm_id, sm_name in ((0, 'tau_e'), (1, 'tau_mu'), (2, 'tau_1h'), (3, 'tau_3h')):
            sig_m = (y == 2) & (tsm == sm_id)
            if sig_m.any():
                res[f'{sm_name}_1pct_fpr_pct'] = _eff_at_fpr(tau_llr, sig_m, y != 2, 0.01)
        sig_had = (y == 2) & ((tsm == 2) | (tsm == 3))
        if sig_had.any():
            res['tau_had_1pct_fpr_pct'] = _eff_at_fpr(tau_llr, sig_had, y != 2, 0.01)

    if 'tau_decay_L3d_mm' in meta.columns:
        l3d = meta['tau_decay_L3d_mm'].to_numpy(dtype=np.float64)
        sig_short = (y == 2) & (l3d > 0.0) & (l3d < 30.0)
        if sig_short.any():
            res['tau_L_lt_30mm_1pct_fpr_pct'] = _eff_at_fpr(tau_llr, sig_short, y != 2, 0.01)

    if 'charm_flag' in meta.columns:
        chm = meta['charm_flag'].to_numpy(dtype=np.int64) > 0
        tgt_w = (meta['tgt_Z'].to_numpy(dtype=np.int64) == 74) if 'tgt_Z' in meta.columns else all_m
        non_tau_w = (y != 2) & tgt_w
        if (non_tau_w & chm).any() and (non_tau_w & (~chm)).any():
            res['W184_Charm_AUROC'] = _auroc_binary(chm[non_tau_w], tau_llr[non_tau_w])
            # FPR of W184 non-tau charm at global 1% non-tau tau threshold
            bg_all = np.sort(tau_llr[y != 2])[::-1]
            if len(bg_all):
                thr_1 = float(bg_all[min(int(math.floor(0.01 * len(bg_all))), len(bg_all) - 1)])
                res['W184_Charm_FPR_at_1pct_tau_pct'] = float(np.mean(tau_llr[non_tau_w & chm] > thr_1) * 100.0)

    return res


def run_supervised_training(
    *,
    cache_dir: Path,
    out_dir: Path,
    seed: int = 42,
    stage: int = 1,
    variant: str = 'gen4_full',
    epochs: int = 12,
    batch_size: int = 64,
    lr: float = 6.0e-4,
    device_str: str = 'cuda:0',
) -> dict[str, Any]:
    """Run Stage-I or Stage-II supervised training on the authorized `10085-10088` cache."""
    _assert_not_deprecated_mc(cache_dir)
    assert_unsealed_path(cache_dir)
    if stage == 1:
        install_stage1_only_hook()
        assert_unsealed_split('internal_train', stage=1)
        assert_unsealed_split('internal_val', stage=1)
    else:
        assert_unsealed_split('train', stage=2)
        assert_unsealed_split('tune6k', stage=2)

    if not cache_dir.is_dir():
        raise FileNotFoundError(f'Gen4 10085-10088 cache directory missing: {cache_dir}')

    with SealAuditGuard() as guard:
        device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(seed)
        np.random.seed(seed)

        geom = load_geometry_contract()
        cfg = dict(ABLATION_CONFIGS.get(variant, ABLATION_CONFIGS['gen4_full']))
        model = PinpointGen4Net(geom, cfg).to(device)

        # Load cached train/eval shards built by worker-foundation on 10085-10088
        train_files = sorted(cache_dir.glob('train*_part_*.pkl'))
        if not train_files:
            raise FileNotFoundError(f'No train*_part_*.pkl found in {cache_dir}')
        train_items: list[dict[str, Any]] = []
        for tf in train_files:
            assert_unsealed_path(tf)
            with tf.open('rb') as fh:
                train_items.extend(pickle.load(fh))

        if stage == 1:
            tr_sub = [it for it in train_items if it.get('internal_split') == 'internal_train']
            ev_sub = [it for it in train_items if it.get('internal_split') == 'internal_val_eval']
            if not tr_sub or not ev_sub:
                n_split = int(0.8 * len(train_items))
                tr_sub, ev_sub = train_items[:n_split], train_items[n_split:]
            del train_items
            ev_meta_raw = pd.DataFrame([
                {
                    'run': int(it['run']),
                    'chunk': int(it['chunk']),
                    'evt_id': int(it['evt_id']),
                    'obs_domain_id': int(it.get('obs_domain_id', 0)),
                    'label': int(it['label']),
                    'label_7': int(it.get('label_7', -1)),
                    'charm_flag': int(it.get('charm_flag', 0)),
                    'vertex_xyz_mm': it.get('vertex_xyz_mm', [0.0, 0.0, 0.0]),
                    'true_tau_submode': int(it.get('true_tau_submode', -1)),
                    'tgt_Z': int(it.get('tgt_Z', 74)),
                    'tau_decay_L3d_mm': float(it.get('tau_decay_L3d_mm', 0.0)),
                }
                for it in ev_sub
            ])
        else:
            tr_sub = train_items
            del train_items
            ev_sub = []
            ev_meta_raw = pd.DataFrame()

        train_aug_cfg = {
            'p_cp_y': 0.5,
            'p_px_si': 0.5,
            'p_subthr_drop': 0.15,
            'drop_edep': bool(cfg.get('force_binary_no_edep', False)),
        }

        out_dir.mkdir(parents=True, exist_ok=True)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        history: list[dict[str, float]] = []

        for ep in range(1, epochs + 1):
            model.train()
            order = rng.permutation(len(tr_sub))
            ep_loss = 0.0
            n_steps = 0
            for start in range(0, len(order), batch_size):
                idx = order[start : start + batch_size]
                batch = _collate_cached_items(
                    [tr_sub[int(i)] for i in idx], geom, device, aug_cfg=train_aug_cfg, rng=rng
                )
                opt.zero_grad(set_to_none=True)
                out = model.forward_NextGen(batch)
                loss, _, _, _ = compute_gen4_supervised_loss(model, out, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
                ep_loss += float(loss.detach().item())
                n_steps += 1
            sched.step()
            history.append({'epoch': ep, 'train_loss': ep_loss / max(n_steps, 1)})
            print(f'[Gen4-Sup s={stage} {variant} seed={seed}] ep={ep:02d}/{epochs} loss={history[-1]["train_loss"]:.4f}', flush=True)

        # Free training items in Stage-II before loading Tune6k to keep peak host RAM < 14 GB
        if stage == 2:
            import gc
            del tr_sub
            gc.collect()
            val_files = sorted(cache_dir.glob('tune*_part_*.pkl'))
            if not val_files:
                raise FileNotFoundError(f'No tune*_part_*.pkl found in {cache_dir}')
            tune_all: list[dict[str, Any]] = []
            for vf in val_files:
                assert_unsealed_path(vf)
                with vf.open('rb') as fh:
                    tune_all.extend(pickle.load(fh))
            ev_sub = [it for it in tune_all if not bool(it.get('in_screen', False))] or tune_all
            del tune_all
            meta_pq = cache_dir.parent / 'validation_meta/tune_truth_meta.parquet'
            if not meta_pq.is_file():
                raise FileNotFoundError(f'Missing validation_meta parquet: {meta_pq}')
            meta_full = pd.read_parquet(meta_pq)
            key_to_row = {
                (int(r), int(c), int(e)): idx
                for idx, (r, c, e) in enumerate(zip(meta_full['run'], meta_full['chunk'], meta_full['evt_id']))
            }
            ev_indices = [key_to_row[(int(it['run']), int(it['chunk']), int(it['evt_id']))] for it in ev_sub]
            ev_meta_raw = meta_full.iloc[ev_indices].reset_index(drop=True)
        ev_meta = _annotate_eval_meta(ev_meta_raw, geom)

        # Pure-observable evaluation (strip truth keys before forward!)
        model.eval()
        probs_list: list[np.ndarray] = []
        llr_list: list[np.ndarray] = []
        vpv_list: list[np.ndarray] = []
        spv_list: list[np.ndarray] = []
        nsbi_accum: dict[str, list[np.ndarray]] = {}

        with torch.no_grad():
            for start in range(0, len(ev_sub), batch_size):
                chunk = ev_sub[start : start + batch_size]
                batch_full = _collate_cached_items(
                    chunk,
                    geom,
                    device,
                    aug_cfg={'drop_edep': bool(cfg.get('force_binary_no_edep', False))} if cfg.get('force_binary_no_edep', False) else None,
                )
                batch_obs = {k: v for k, v in batch_full.items() if k not in GEN4_TRUTH_BATCH_KEYS}
                out = model.forward_NextGen(batch_obs)
                probs_list.append(out.probs4.cpu().numpy())
                llr_list.append(out.aux['tau_llr_score'].cpu().numpy())
                vpv_list.append(out.v_pv_mm.cpu().numpy())
                spv_list.append(out.scale_pv_mm.cpu().numpy())
                nsbi_b = extract_nsbi_observables_from_output(batch_obs, out)
                for nk, nv in nsbi_b.items():
                    nsbi_accum.setdefault(nk, []).append(nv)

        probs4_ev = np.concatenate(probs_list, axis=0)
        tau_llr_ev = np.concatenate(llr_list, axis=0)
        v_pv_ev = np.concatenate(vpv_list, axis=0)
        scale_pv_ev = np.concatenate(spv_list, axis=0)
        labels_ev = ev_meta['label'].to_numpy(dtype=np.int64)
        nsbi_ev = {k: np.concatenate(v, axis=0) for k, v in nsbi_accum.items()}

        metrics = _evaluate_predictions(probs4_ev, tau_llr_ev, labels_ev, ev_meta, v_pv_mm=v_pv_ev)
        metrics['seal_violations'] = len(guard.violations)
        metrics['wall_seconds'] = float(time.perf_counter() - t0)
        metrics['seed'] = seed
        metrics['variant'] = variant
        metrics['history'] = history

        # Save contracted Stage-2 prediction artifacts
        if stage == 2:
            # Evaluate Stress400 & Diag_OOD only when stress_diag_pack.pkl is at least as fresh as tune_truth_meta.parquet
            stress_pack_p = cache_dir / 'stress_diag_pack.pkl'
            diag_meta_p = cache_dir.parent / 'validation_meta/diag_truth_meta.pkl'
            tune_meta_p = cache_dir.parent / 'validation_meta/tune_truth_meta.parquet'
            is_fresh_pack = (
                stress_pack_p.is_file()
                and diag_meta_p.is_file()
                and tune_meta_p.is_file()
                and (stress_pack_p.stat().st_mtime >= tune_meta_p.stat().st_mtime - 60.0)
            )
            if is_fresh_pack:
                with stress_pack_p.open('rb') as fh:
                    spack = pickle.load(fh)
                with diag_meta_p.open('rb') as fh:
                    dmeta = pickle.load(fh)
                stress_y = np.asarray(dmeta['stress_y'], dtype=np.int64)
                stress_res: dict[str, dict[str, float]] = {}
                for st_name, st_items in spack.get('stress_items', {}).items():
                    st_probs, st_llr = [], []
                    with torch.no_grad():
                        for s_i in range(0, len(st_items), batch_size):
                            b_st = _collate_cached_items(st_items[s_i : s_i + batch_size], geom, device)
                            b_st_obs = {k: v for k, v in b_st.items() if k not in GEN4_TRUTH_BATCH_KEYS}
                            o_st = model.forward_NextGen(b_st_obs)
                            st_probs.append(o_st.probs4.cpu().numpy())
                            st_llr.append(o_st.aux['tau_llr_score'].cpu().numpy())
                    sp4 = np.concatenate(st_probs, axis=0)
                    sllr = np.concatenate(st_llr, axis=0)
                    sm = _evaluate_predictions(sp4, sllr, stress_y, pd.DataFrame())
                    stress_res[st_name] = {'BA4_pct': sm['BA4_pct'], 'tau_1pct_fpr_llr_pct': sm['tau_1pct_fpr_llr_pct']}
                metrics['stress400'] = stress_res

                diag_res: dict[str, Any] = {}
                for dk, d_items in spack.get('diag_packs', {}).items():
                    if not d_items or dk not in dmeta.get('diag_meta', {}):
                        continue
                    dp_l, dl_l, dv_l = [], [], []
                    with torch.no_grad():
                        for d_i in range(0, len(d_items), batch_size):
                            b_dg = _collate_cached_items(d_items[d_i : d_i + batch_size], geom, device)
                            b_dg_obs = {k: v for k, v in b_dg.items() if k not in GEN4_TRUTH_BATCH_KEYS}
                            o_dg = model.forward_NextGen(b_dg_obs)
                            dp_l.append(o_dg.probs4.cpu().numpy())
                            dl_l.append(o_dg.aux['tau_llr_score'].cpu().numpy())
                            dv_l.append(o_dg.v_pv_mm.cpu().numpy())
                    d_meta_df = _annotate_eval_meta(dmeta['diag_meta'][dk], geom)
                    diag_res[dk] = _evaluate_predictions(
                        np.concatenate(dp_l, axis=0),
                        np.concatenate(dl_l, axis=0),
                        d_meta_df['label'].to_numpy(dtype=np.int64),
                        d_meta_df,
                        v_pv_mm=np.concatenate(dv_l, axis=0),
                    )
                metrics['diag_ood'] = diag_res

            seed_dir = out_dir / f'stage2/seed_{seed}'
            seed_dir.mkdir(parents=True, exist_ok=True)
            np.save(seed_dir / f'probs4_tune8k_seed{seed}.npy', probs4_ev)
            np.save(seed_dir / f'tau_llr_tune8k_seed{seed}.npy', tau_llr_ev)
            np.save(seed_dir / f'v_pv_tune8k_seed{seed}.npy', v_pv_ev)
            np.save(seed_dir / f'scale_pv_tune8k_seed{seed}.npy', scale_pv_ev)
            np.savez(seed_dir / f'nsbi_obs_tune8k_seed{seed}.npz', **nsbi_ev)
            torch.save({'model_state_dict': model.state_dict(), 'config': cfg}, seed_dir / 'checkpoint.pt')
            write_json(seed_dir / 'metrics.json', metrics)
        else:
            ab_dir = out_dir / f'stage1/{variant}'
            ab_dir.mkdir(parents=True, exist_ok=True)
            write_json(ab_dir / 'metrics.json', {'metrics': metrics, 'history': history})

        return metrics


def aggregate_stage2_results(
    *,
    out_dir: Path = EXP_SUP_DIR,
    seeds: tuple[int, ...] = STAGE2_SEEDS,
) -> dict[str, Any]:
    """Aggregate 3-seed Stage-II supervised results (mean +/- std) and check Ref-L1-GN2."""
    per_seed: dict[str, dict[str, Any]] = {}
    for s in seeds:
        mp = out_dir / f'stage2/seed_{s}/metrics.json'
        if not mp.is_file():
            raise FileNotFoundError(f'Missing Stage-II metrics for seed={s}: {mp}')
        per_seed[str(s)] = json.loads(mp.read_text())

    numeric_keys = sorted(
        {
            k
            for d in per_seed.values()
            for k, v in d.items()
            if isinstance(v, (int, float)) and k not in ('seed', 'seal_violations')
        }
    )
    summary_stats: dict[str, dict[str, float]] = {}
    for k in numeric_keys:
        vals = [float(per_seed[str(s)][k]) for s in seeds if k in per_seed[str(s)]]
        if len(vals) == len(seeds):
            summary_stats[k] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            }

    out = {
        'UTC_aggregated': datetime.now(timezone.utc).isoformat(),
        'seeds': list(seeds),
        'summary': summary_stats,
        'per_seed': per_seed,
    }
    # Optional Ref-L1-GN2 3-seed summary
    ref_seeds: dict[str, dict[str, Any]] = {}
    for s in seeds:
        rp = out_dir / f'ref_l1_gn2/seed_{s}/metrics.json'
        if rp.is_file():
            ref_seeds[str(s)] = json.loads(rp.read_text())
    if len(ref_seeds) == len(seeds):
        ref_stats: dict[str, dict[str, float]] = {}
        for k in sorted({kk for d in ref_seeds.values() for kk, vv in d.items() if isinstance(vv, (int, float)) and kk not in ('seed', 'seal_violations')}):
            vals = [float(ref_seeds[str(s)][k]) for s in seeds if k in ref_seeds[str(s)]]
            if len(vals) == len(seeds):
                ref_stats[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}
        out['ref_l1_gn2_summary'] = ref_stats
        out['ref_l1_gn2_per_seed'] = ref_seeds

    # Optional Stage-I ablation summary
    ab_map: dict[str, Any] = {}
    for vname in ABLATION_CONFIGS:
        ap = out_dir / f'stage1/{vname}/metrics.json'
        if ap.is_file():
            d = json.loads(ap.read_text())
            ab_map[vname] = d.get('metrics', d)
    if ab_map:
        out['stage1_ablations'] = ab_map

    write_json(out_dir / 'stage2/stage2_summary.json', out)
    return out


def evaluate_stage2_stress_diag(
    *,
    cache_dir: Path = GEN4_CACHE_DIR,
    out_dir: Path = EXP_SUP_DIR,
    seeds: tuple[int, ...] = STAGE2_SEEDS,
    batch_size: int = 64,
    device_str: str = 'cuda:0',
) -> dict[str, Any]:
    """Evaluate trained Stage-II checkpoints on freshly built Stress400 & Diag_OOD."""
    _assert_not_deprecated_mc(cache_dir)
    assert_unsealed_path(cache_dir)
    stress_pack_p = cache_dir / 'stress_diag_pack.pkl'
    diag_meta_p = cache_dir.parent / 'validation_meta/diag_truth_meta.pkl'
    if not (stress_pack_p.is_file() and diag_meta_p.is_file()):
        raise FileNotFoundError(f'Missing {stress_pack_p} or {diag_meta_p}')

    with SealAuditGuard():
        device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
        geom = load_geometry_contract()
        with stress_pack_p.open('rb') as fh:
            spack = pickle.load(fh)
        with diag_meta_p.open('rb') as fh:
            dmeta = pickle.load(fh)
        stress_y = np.asarray(dmeta['stress_y'], dtype=np.int64)

        results: dict[str, Any] = {}
        for s in seeds:
            ckpt_p = out_dir / f'stage2/seed_{s}/checkpoint.pt'
            mp = out_dir / f'stage2/seed_{s}/metrics.json'
            if not ckpt_p.is_file():
                continue
            ck = torch.load(ckpt_p, map_location=device, weights_only=False)
            model = PinpointGen4Net(geom, ck.get('config', {})).to(device)
            model.load_state_dict(ck['model_state_dict'])
            model.eval()

            stress_res: dict[str, dict[str, float]] = {}
            for st_name, st_items in spack.get('stress_items', {}).items():
                st_probs, st_llr = [], []
                with torch.no_grad():
                    for s_i in range(0, len(st_items), batch_size):
                        b_st = _collate_cached_items(st_items[s_i : s_i + batch_size], geom, device)
                        b_st_obs = {k: v for k, v in b_st.items() if k not in GEN4_TRUTH_BATCH_KEYS}
                        o_st = model.forward_NextGen(b_st_obs)
                        st_probs.append(o_st.probs4.cpu().numpy())
                        st_llr.append(o_st.aux['tau_llr_score'].cpu().numpy())
                sm = _evaluate_predictions(np.concatenate(st_probs, axis=0), np.concatenate(st_llr, axis=0), stress_y, pd.DataFrame())
                stress_res[st_name] = {'BA4_pct': sm['BA4_pct'], 'tau_1pct_fpr_llr_pct': sm['tau_1pct_fpr_llr_pct']}

            diag_res: dict[str, Any] = {}
            for dk, d_items in spack.get('diag_packs', {}).items():
                if not d_items or dk not in dmeta.get('diag_meta', {}):
                    continue
                dp_l, dl_l, dv_l = [], [], []
                with torch.no_grad():
                    for d_i in range(0, len(d_items), batch_size):
                        b_dg = _collate_cached_items(d_items[d_i : d_i + batch_size], geom, device)
                        b_dg_obs = {k: v for k, v in b_dg.items() if k not in GEN4_TRUTH_BATCH_KEYS}
                        o_dg = model.forward_NextGen(b_dg_obs)
                        dp_l.append(o_dg.probs4.cpu().numpy())
                        dl_l.append(o_dg.aux['tau_llr_score'].cpu().numpy())
                        dv_l.append(o_dg.v_pv_mm.cpu().numpy())
                d_meta_df = _annotate_eval_meta(dmeta['diag_meta'][dk], geom)
                diag_res[dk] = _evaluate_predictions(
                    np.concatenate(dp_l, axis=0),
                    np.concatenate(dl_l, axis=0),
                    d_meta_df['label'].to_numpy(dtype=np.int64),
                    d_meta_df,
                    v_pv_mm=np.concatenate(dv_l, axis=0),
                )

            if mp.is_file():
                m_cur = json.loads(mp.read_text())
                m_cur['stress400'] = stress_res
                m_cur['diag_ood'] = diag_res
                write_json(mp, m_cur)
            results[str(s)] = {'stress400': stress_res, 'diag_ood': diag_res}
        return results


def write_condor_submit_files(
    *,
    cache_dir: Path = GEN4_CACHE_DIR,
    schedd: str | None = None,
) -> dict[str, str]:
    """Generate HTCondor GPU job scripts in ~/condor/Pinpoint/stage329_gen4_sup/."""
    _assert_not_deprecated_mc(cache_dir)
    if schedd is None:
        from analysis.flavour.gen4.run_gen4_ssl_and_tail import choose_schedd
        schedd = choose_schedd(EXP_SUP_DIR)[0]

    CONDOR_SUP_DIR.mkdir(parents=True, exist_ok=True)
    logs_dir = CONDOR_SUP_DIR / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)

    run_sh = CONDOR_SUP_DIR / 'run_sup.sh'
    run_sh.write_text(
        '#!/bin/bash\n'
        'set -e\n'
        'command -v which >/dev/null 2>&1 || { which() { command -v "$@"; }; export -f which; }\n'
        'SAVED_ARGS=("$@"); set --\n'
        'source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh >/dev/null 2>&1\n'
        'set -- "${SAVED_ARGS[@]}"\n'
        'export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4\n'
        'cd /eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis\n'
        'export PYTHONPATH="src:${PYTHONPATH:-}"\n'
        f'exec /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/bin/python -u -m analysis.flavour.gen4.run_gen4_supervised "$@" --cache-dir {cache_dir} --out-dir {EXP_SUP_DIR}\n'
    )
    run_sh.chmod(0o755)

    init_dir = str(CONDOR_SUP_DIR.resolve())
    req_gpu = '((Capability >= 7.5 && GlobalMemoryMb >= 14500) || ((Capability =?= UNDEFINED) && regexp("^NVIDIA H100.* MIG [2347]g", DeviceName)))'
    ab_sub = CONDOR_SUP_DIR / 'stage1_ablation.sub'
    ab_variants = ' '.join(ABLATION_CONFIGS.keys())
    ab_sub.write_text(
        f'universe = vanilla\n'
        f'initialdir = {init_dir}\n'
        f'executable = {run_sh}\n'
        f'arguments = --mode stage1-ablation --variant $(VARIANT) --seed 42 --epochs 12 --batch-size 64\n'
        f'output = {logs_dir}/stage1_$(VARIANT).out\n'
        f'error = {logs_dir}/stage1_$(VARIANT).err\n'
        f'log = {logs_dir}/stage1_$(VARIANT).log\n'
        f'request_gpus = 1\n'
        f'require_gpus = {req_gpu}\n'
        f'request_cpus = 6\n'
        f'request_memory = 18000\n'
        f'+JobFlavour = "workday"\n'
        f'queue VARIANT in ({ab_variants})\n'
    )

    sub_f = CONDOR_SUP_DIR / 'stage2_3seed.sub'
    sub_f.write_text(
        f'universe = vanilla\n'
        f'initialdir = {init_dir}\n'
        f'executable = {run_sh}\n'
        f'arguments = --mode stage2-train --seed $(SEED) --epochs 16 --batch-size 64\n'
        f'output = {logs_dir}/stage2_$(SEED).out\n'
        f'error = {logs_dir}/stage2_$(SEED).err\n'
        f'log = {logs_dir}/stage2_$(SEED).log\n'
        f'request_gpus = 1\n'
        f'require_gpus = {req_gpu}\n'
        f'request_cpus = 6\n'
        f'request_memory = 18000\n'
        f'+JobFlavour = "workday"\n'
        f'queue SEED in (17, 42, 2026)\n'
    )

    ref_sub = CONDOR_SUP_DIR / 'ref_l1_gn2.sub'
    ref_sub.write_text(
        f'universe = vanilla\n'
        f'initialdir = {init_dir}\n'
        f'executable = {run_sh}\n'
        f'arguments = --mode ref-l1-gn2 --seed $(SEED) --epochs 16 --batch-size 64\n'
        f'output = {logs_dir}/ref_l1_gn2_$(SEED).out\n'
        f'error = {logs_dir}/ref_l1_gn2_$(SEED).err\n'
        f'log = {logs_dir}/ref_l1_gn2_$(SEED).log\n'
        f'request_gpus = 1\n'
        f'require_gpus = {req_gpu}\n'
        f'request_cpus = 6\n'
        f'request_memory = 36000\n'
        f'+JobFlavour = "workday"\n'
        f'queue SEED in (17, 42, 2026)\n'
    )
    return {
        'run_sh': str(run_sh),
        'stage1_ablation_sub': str(ab_sub),
        'stage2_sub': str(sub_f),
        'ref_l1_gn2_sub': str(ref_sub),
        'schedd': schedd,
    }


_GN2_WORKER_STATE: dict[str, Any] = {}


def _init_gn2_worker(
    geom: Gen4GeometryContract,
    raw_geom: dict[str, Any],
    bounds: dict[str, Any],
    plane_x0: np.ndarray,
    legacy_obs_cfg: dict[str, Any],
) -> None:
    _GN2_WORKER_STATE['geom'] = geom
    _GN2_WORKER_STATE['raw_geom'] = raw_geom
    _GN2_WORKER_STATE['bounds'] = bounds
    _GN2_WORKER_STATE['plane_x0'] = plane_x0
    _GN2_WORKER_STATE['legacy_obs_cfg'] = legacy_obs_cfg


def _align_gn2_cluster_sidecars(it: dict[str, Any], gn2_it: dict[str, Any]) -> None:
    """Align Gen4 cached cluster sidecars onto `gn2_it` clusters without ever omitting keys."""
    g_plane = np.asarray(gn2_it.get('pix_plane', []), dtype=np.int16)
    n_cl = int(len(g_plane))
    if 'cl_origin_label' in it and len(it['cl_origin_label']) == n_cl:
        gn2_it['cl_origin_label'] = np.asarray(it['cl_origin_label'], dtype=np.int16)
        gn2_it['cl_secondary_origin_label'] = np.asarray(it['cl_secondary_origin_label'], dtype=np.uint8)
        gn2_it['cl_sv_group_id'] = np.asarray(it['cl_sv_group_id'], dtype=np.int16)
        gn2_it['cl_dominant_track_id'] = np.asarray(it['cl_dominant_track_id'], dtype=np.int32)
        return

    orig = np.full(n_cl, -1, dtype=np.int16)
    sec_orig = np.full(n_cl, 255, dtype=np.uint8)
    sv_gid = np.full(n_cl, -1, dtype=np.int16)
    tid = np.full(n_cl, -1, dtype=np.int32)
    if (
        n_cl > 0
        and 'cl_origin_label' in it
        and 'pix_xyz_mm' in it
        and 'pix_plane' in it
        and len(it['pix_plane']) > 0
    ):
        g_xyz = np.asarray(gn2_it['pix_xyz_mm'], dtype=np.float32)
        c_xyz = np.asarray(it['pix_xyz_mm'], dtype=np.float32)
        c_plane = np.asarray(it['pix_plane'], dtype=np.int16)
        c_orig = np.asarray(it['cl_origin_label'], dtype=np.int16)
        c_sec = np.asarray(it['cl_secondary_origin_label'], dtype=np.uint8)
        c_sv = np.asarray(it['cl_sv_group_id'], dtype=np.int16)
        c_tid = np.asarray(it['cl_dominant_track_id'], dtype=np.int32)
        for pl in np.unique(g_plane):
            g_idx = np.flatnonzero(g_plane == pl)
            c_idx = np.flatnonzero(c_plane == pl)
            if len(c_idx) == 0:
                continue
            d2 = ((g_xyz[g_idx, :2][:, None, :] - c_xyz[c_idx, :2][None, :, :]) ** 2).sum(axis=-1)
            best = np.argmin(d2, axis=1)
            ok = d2[np.arange(len(g_idx)), best] <= (0.25 ** 2)
            mg = g_idx[ok]
            mc = c_idx[best[ok]]
            orig[mg] = c_orig[mc]
            sec_orig[mg] = c_sec[mc]
            sv_gid[mg] = c_sv[mc]
            tid[mg] = c_tid[mc]

    gn2_it['cl_origin_label'] = orig
    gn2_it['cl_secondary_origin_label'] = sec_orig
    gn2_it['cl_sv_group_id'] = sv_gid
    gn2_it['cl_dominant_track_id'] = tid


def _process_gn2_chunk_task(
    task: tuple[str, int, int, bool, list[tuple[int, dict[str, Any]]]],
) -> list[tuple[int, dict[str, Any]]]:
    import uproot
    from analysis.flavour.gen4.physics_frontend import build_observation_from_raw_root_event
    from analysis.flavour.nextgen.gn2_tracklet_ipformer import extract_soft_tracklets_with_indices
    from analysis.flavour.v3.common.domains import compute_observable_containment
    from analysis.flavour.v3.v2_pinpointformer.pixel_tokenizer import extract_pf_item

    rp_str, chunk_i, seed_base, is_train, entries = task
    geom = _GN2_WORKER_STATE['geom']
    raw_geom = _GN2_WORKER_STATE['raw_geom']
    bounds = _GN2_WORKER_STATE['bounds']
    plane_x0 = _GN2_WORKER_STATE['plane_x0']
    legacy_obs_cfg = _GN2_WORKER_STATE['legacy_obs_cfg']
    pitch_xy = (float(bounds['dx_mm']), float(bounds['dy_mm']))

    with uproot.open(rp_str) as f:
        ph = f['Hits/pixelHits'].arrays(['rowID', 'colID', 'layerID', 'edep'], library='np')
        sh = f['Hits/scintHits'].arrays(['panelID', 'isHorizontal', 'rowID', 'colID', 'nPE', 'hitTime'], library='np')
        fh = f['Hits/faserHits'].arrays(['trackerID', 'x', 'y', 'z'], library='np')

    out: list[tuple[int, dict[str, Any]]] = []
    for pos_i, it in entries:
        eid_i = int(it['evt_id'])
        ev_seed = int(seed_base) + chunk_i * 2000 + eid_i
        obs = build_observation_from_raw_root_event(
            ph['rowID'][eid_i], ph['colID'][eid_i], ph['layerID'][eid_i], ph['edep'][eid_i],
            sh['panelID'][eid_i], sh['isHorizontal'][eid_i], sh['rowID'][eid_i],
            sh['colID'][eid_i], sh['nPE'][eid_i], sh['hitTime'][eid_i],
            fh['trackerID'][eid_i], fh['x'][eid_i], fh['y'][eid_i], fh['z'][eid_i],
            geom, event_seed=ev_seed,
        )
        dom = compute_observable_containment(obs, raw_geom, bounds, config=legacy_obs_cfg)
        lbl_int = int(it['label']) if is_train else None
        gn2_it = extract_pf_item(eid_i, obs, lbl_int, dom, raw_geom, bounds, legacy_obs_cfg)
        keep_bar = gn2_it.pop('bar_keep_mask', None)
        gn2_it.pop('pix_first_cid', None)
        bar = obs['bar']
        if keep_bar is not None and 'hit_time_ns' in bar and len(bar['hit_time_ns']) == len(keep_bar):
            ht = np.asarray(bar['hit_time_ns'], dtype=np.float32)[keep_bar]
        else:
            ht = np.full(len(gn2_it['bar_panel']), -1.0, dtype=np.float32)
        gn2_it['bar_hit_time_ns'] = ht
        trk = extract_soft_tracklets_with_indices(
            gn2_it['pix_xyz_mm'], gn2_it['pix_plane'], gn2_it['pix_feat'],
            plane_x0, max_tracklets=64, pitch_xy_mm=pitch_xy,
        )
        gn2_it.update(trk)
        if is_train:
            gn2_it['label_7'] = int(it.get('label_7', -1))
            gn2_it['charm_flag'] = int(it.get('charm_flag', 0))
            gn2_it['charm_valid'] = int(it.get('charm_valid', 1))
            gn2_it['vertex_xyz_mm'] = np.asarray(it['vertex_xyz_mm'], dtype=np.float32)
            gn2_it['vertex_valid'] = int(it.get('vertex_valid', 1))
            gn2_it['vis_kinematics'] = np.asarray(it['vis_kinematics'], dtype=np.float32)
            gn2_it['kin_valid'] = int(it.get('kin_valid', 1))
            gn2_it['sample_weight'] = float(it.get('sample_weight', 1.0))
            _align_gn2_cluster_sidecars(it, gn2_it)
        out.append((pos_i, gn2_it))
    return out


def _load_gn2_items_from_mc8588_cache(
    cache_dir: Path,
    raw_geom: dict[str, Any],
    bounds: dict[str, Any],
    geom: Gen4GeometryContract,
    *,
    is_train: bool,
    workers: int = 1,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Reconstruct exact TrackletIPFormerGN2 items on the exact 10085-10088 cached events."""
    from analysis.flavour.gen4.data_cache import (
        MC8588_DATASETS,
        assert_unsealed_mc8588_path,
        partition_mc8588_chunks,
    )

    import concurrent.futures
    import gc
    import multiprocessing as mp

    pattern = 'train*_part_*.pkl' if is_train else 'tune*_part_*.pkl'
    keep_keys = (
        'run', 'chunk', 'evt_id', 'c_obs', 'in_screen',
        'label', 'label_7', 'charm_flag', 'charm_valid',
        'vertex_xyz_mm', 'vertex_valid', 'vis_kinematics', 'kin_valid', 'sample_weight',
        'pix_xyz_mm', 'pix_plane',
        'cl_origin_label', 'cl_secondary_origin_label', 'cl_sv_group_id', 'cl_dominant_track_id',
    )
    items: list[dict[str, Any]] = []
    for pf in sorted(cache_dir.glob(pattern)):
        assert_unsealed_path(pf)
        with pf.open('rb') as fh:
            part_items = pickle.load(fh)
        for it in part_items:
            if not is_train and bool(it.get('in_screen', False)):
                continue
            items.append({k: it[k] for k in keep_keys if k in it})
        del part_items
        gc.collect()

    splits = partition_mc8588_chunks()
    chunk_paths: dict[tuple[int, int], Path] = {}
    for bucket in splits.values():
        for ds_str, pairs in bucket.items():
            for cid, p in pairs:
                chunk_paths[(int(ds_str), int(cid))] = p

    by_rc: dict[tuple[int, int], list[tuple[int, dict[str, Any]]]] = {}
    for pos_i, it in enumerate(items):
        by_rc.setdefault((int(it['run']), int(it['chunk'])), []).append((pos_i, it))

    legacy_obs_cfg = {
        'pe_only': True,
        'bar_threshold_pe': 2.5,
        'sipm_npix': 14400.0,
        'apply_sipm_saturation': False,
        'invert_sipm_saturation': True,
    }
    plane_x0 = np.asarray(bounds['_plane_x0'], dtype=np.float32)
    ds_keys = list(MC8588_DATASETS.keys())
    tasks: list[tuple[str, int, int, bool, list[tuple[int, dict[str, Any]]]]] = []
    for (run_i, chunk_i), entries in sorted(by_rc.items()):
        rp = assert_unsealed_mc8588_path(chunk_paths[(run_i, chunk_i)])
        seed_base = (10_000_000 if is_train else 30_000_000) * (ds_keys.index(str(run_i)) + 1)
        tasks.append((str(rp), chunk_i, seed_base, is_train, entries))

    gn2_items: list[dict[str, Any] | None] = [None] * len(items)
    t_load0 = time.perf_counter()
    if int(workers) > 1:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=int(workers),
            mp_context=mp.get_context('spawn'),
            initializer=_init_gn2_worker,
            initargs=(geom, raw_geom, bounds, plane_x0, legacy_obs_cfg),
        ) as ex:
            for ci, res_list in enumerate(ex.map(_process_gn2_chunk_task, tasks), start=1):
                for pos_i, gn2_it in res_list:
                    gn2_items[pos_i] = gn2_it
                    items[pos_i].pop('pix_xyz_mm', None)
                    items[pos_i].pop('pix_plane', None)
                    items[pos_i].pop('cl_origin_label', None)
                    items[pos_i].pop('cl_secondary_origin_label', None)
                    items[pos_i].pop('cl_sv_group_id', None)
                    items[pos_i].pop('cl_dominant_track_id', None)
                if ci % 40 == 0 or ci == len(tasks):
                    print(f'[GN2-Load train={is_train}] chunks={ci}/{len(tasks)} wall={time.perf_counter()-t_load0:.1f}s', flush=True)
    else:
        _init_gn2_worker(geom, raw_geom, bounds, plane_x0, legacy_obs_cfg)
        for ci, task in enumerate(tasks, start=1):
            for pos_i, gn2_it in _process_gn2_chunk_task(task):
                gn2_items[pos_i] = gn2_it
                items[pos_i].pop('pix_xyz_mm', None)
                items[pos_i].pop('pix_plane', None)
                items[pos_i].pop('cl_origin_label', None)
                items[pos_i].pop('cl_secondary_origin_label', None)
                items[pos_i].pop('cl_sv_group_id', None)
                items[pos_i].pop('cl_dominant_track_id', None)
            if ci % 40 == 0 or ci == len(tasks):
                print(f'[GN2-Load train={is_train}] chunks={ci}/{len(tasks)} wall={time.perf_counter()-t_load0:.1f}s', flush=True)

    if is_train:
        meta_df = pd.DataFrame()
    else:
        meta_pq = cache_dir.parent / 'validation_meta/tune_truth_meta.parquet'
        meta_full = pd.read_parquet(meta_pq)
        kmap = {
            (int(r), int(c), int(e)): idx
            for idx, (r, c, e) in enumerate(zip(meta_full['run'], meta_full['chunk'], meta_full['evt_id']))
        }
        ev_idx = [kmap[(int(it['run']), int(it['chunk']), int(it['evt_id']))] for it in items]
        meta_df = _annotate_eval_meta(meta_full.iloc[ev_idx].reset_index(drop=True), geom)

    del items, tasks, by_rc
    gc.collect()
    return [x for x in gn2_items if x is not None], meta_df


def run_ref_l1_gn2_baseline(
    *,
    cache_dir: Path,
    out_dir: Path,
    seed: int = 42,
    epochs: int = 16,
    batch_size: int = 64,
    lr: float = 8.0e-4,
    device_str: str = 'cuda:0',
    workers: int = 1,
) -> dict[str, Any]:
    """Run controlled prior-gen reference `Ref-L1-GN2` (`TrackletIPFormerGN2`) on the 10085-10088 split."""
    from analysis.flavour.nextgen.gn2_tracklet_ipformer import (
        TrackletIPFormerGN2,
        collate_tracklet_gn2,
        extract_tracklet_gn2_item,
        gn2_tracklet_ipformer_loss,
    )
    from analysis.flavour.v3.common.domains import cumulative_radiation_length, detector_bounds

    _assert_not_deprecated_mc(cache_dir)
    assert_unsealed_path(cache_dir)
    assert_unsealed_split('train', stage=2)
    assert_unsealed_split('tune6k', stage=2)

    with SealAuditGuard() as guard:
        device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(seed)
        np.random.seed(seed)

        geom = load_geometry_contract()
        raw_geom = geom.raw_geometry
        bounds = detector_bounds(raw_geom)
        bounds['_plane_x0'] = np.array(
            [cumulative_radiation_length(float(z), raw_geom) for z in bounds['pixel_Zpos']], dtype=np.float32
        )

        raw_tr_file = cache_dir / 'raw_obs_train32k.pkl'
        raw_va_file = cache_dir / 'raw_obs_tune8k.pkl'
        if raw_tr_file.is_file() and raw_va_file.is_file():
            with raw_tr_file.open('rb') as fh:
                raw_tr = pickle.load(fh)
            with raw_va_file.open('rb') as fh:
                raw_va = pickle.load(fh)
            legacy_obs_cfg = {
                'pe_only': True,
                'bar_threshold_pe': 2.5,
                'sipm_npix': 14400.0,
                'apply_sipm_saturation': False,
                'invert_sipm_saturation': True,
            }
            tr_items = [
                extract_tracklet_gn2_item(i, it['obs'], it.get('truth'), it['c_obs'], raw_geom, bounds, legacy_obs_cfg, it.get('sidecar'))
                for i, it in enumerate(raw_tr)
            ]
            ev_items = [
                extract_tracklet_gn2_item(i, it['obs'], None, it['c_obs'], raw_geom, bounds, legacy_obs_cfg, None)
                for i, it in enumerate(raw_va)
            ]
            ev_meta = pd.DataFrame([it.get('meta', {}) for it in raw_va])
            labels_ev = np.array([int(it['label']) for it in raw_va], dtype=np.int64)
            use_raw_va = True
        else:
            tr_items, _ = _load_gn2_items_from_mc8588_cache(
                cache_dir, raw_geom, bounds, geom, is_train=True, workers=workers
            )
            ev_items = []
            ev_meta = pd.DataFrame()
            labels_ev = np.empty(0, dtype=np.int64)
            use_raw_va = False

        from analysis.flavour.nextgen.tail_dro_pauc import (
            TailDROConfig,
            extract_penultimate_cache,
            finetune_head_on_cached_latents,
            freeze_backbone_for_tail_finetuning,
            snapshot_backbone_weights,
            tau_log_odds_from_logits7,
            verify_backbone_bit_identity,
        )

        yc = float(bounds['y_scint_center'])
        nb = int(bounds['n_bars'])
        model = TrackletIPFormerGN2(raw_geom, {'dim': 160, 'heads': 8}).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.03)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()

        for ep in range(1, epochs + 1):
            model.train()
            order = rng.permutation(len(tr_items))
            ep_loss = 0.0
            n_st = 0
            for start in range(0, len(order), batch_size):
                mb = collate_tracklet_gn2(
                    [tr_items[int(i)] for i in order[start : start + batch_size]],
                    mirror_rng=None, y_scint_center=yc, n_bars=nb,
                )
                mb_dev = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in mb.items()}
                opt.zero_grad(set_to_none=True)
                out = model.forward_NextGen(mb_dev)
                loss, _ = gn2_tracklet_ipformer_loss(out, mb_dev)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
                ep_loss += float(loss.detach().item())
                n_st += 1
            sched.step()
            print(f'[Ref-L1-GN2 seed={seed}] ep={ep:02d}/{epochs} loss={ep_loss/max(n_st,1):.4f}', flush=True)

        # Frozen-backbone Score-Bank KL-DRO Partial-AUC tail fine-tuning (+KL-DRO)
        dro_cfg = TailDROConfig(
            fpr_target=0.01,
            dro_mode='cvar_balanced',
            surrogate='softplus',
            margin=0.5,
            kl_dro_temp=0.05,
            score_bound=4.0,
            cvar_weight=0.25,
            eta_lr=5.0e-4,
            lambda_dro=0.08,
            lambda_nll=0.5,
            lambda_kl=1.5,
            use_bank=True,
            bank_capacity=8192,
            lr=5.0e-5,
            weight_decay=1.0e-4,
            passes=3,
            batch_size=128,
            grad_clip=2.0,
            seed=seed,
        )
        tr_batches = (
            collate_tracklet_gn2(tr_items[st : st + batch_size], mirror_rng=None, y_scint_center=yc, n_bars=nb)
            for st in range(0, len(tr_items), batch_size)
        )
        snap = snapshot_backbone_weights(model, head_attr='event_head_7')
        head = freeze_backbone_for_tail_finetuning(model, head_attr='event_head_7')
        tr_cache = extract_penultimate_cache(model, tr_batches, device, head_attr='event_head_7', verify_atol=1e-6)
        del tr_batches
        finetune_head_on_cached_latents(head, tr_cache, dro_cfg, device)
        verify_backbone_bit_identity(model, snap, head_attr='event_head_7')
        import gc
        del tr_items, tr_cache
        gc.collect()
        if not use_raw_va:
            ev_items, ev_meta = _load_gn2_items_from_mc8588_cache(
                cache_dir, raw_geom, bounds, geom, is_train=False, workers=workers
            )
            labels_ev = ev_meta['label'].to_numpy(dtype=np.int64)

        model.eval()
        probs_list: list[np.ndarray] = []
        llr_list: list[np.ndarray] = []
        vpv_list: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(ev_items), batch_size):
                mb = collate_tracklet_gn2(ev_items[start : start + batch_size], mirror_rng=None, y_scint_center=yc, n_bars=nb)
                mb_obs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in mb.items() if k not in GEN4_TRUTH_BATCH_KEYS}
                out = model.forward_NextGen(mb_obs)
                probs_list.append(out.probs4.cpu().numpy())
                llr_list.append(tau_log_odds_from_logits7(out.logits7).cpu().numpy())
                vpv_list.append(out.v_pv_mm.cpu().numpy())

        probs4_ev = np.concatenate(probs_list, axis=0)
        tau_llr = np.concatenate(llr_list, axis=0)
        v_pv_ev = np.concatenate(vpv_list, axis=0)
        metrics = _evaluate_predictions(probs4_ev, tau_llr, labels_ev, ev_meta, v_pv_mm=v_pv_ev)
        metrics['seal_violations'] = len(guard.violations)
        metrics['wall_seconds'] = float(time.perf_counter() - t0)
        metrics['seed'] = seed

        ref_dir = out_dir / f'ref_l1_gn2/seed_{seed}'
        ref_dir.mkdir(parents=True, exist_ok=True)
        np.save(ref_dir / f'probs4_tune8k_seed{seed}.npy', probs4_ev)
        np.save(ref_dir / f'tau_llr_tune8k_seed{seed}.npy', tau_llr)
        np.save(ref_dir / f'v_pv_tune8k_seed{seed}.npy', v_pv_ev)
        torch.save({'model_state_dict': model.state_dict()}, ref_dir / 'checkpoint_dro.pt')
        write_json(ref_dir / 'metrics.json', metrics)
        return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Pinpoint-Gen4 Supervised Runner')
    parser.add_argument(
        '--mode',
        choices=('gate-ab', 'stage1-ablation', 'stage2-train', 'stage2-aggregate', 'eval-stress-diag', 'ref-l1-gn2', 'prepare-condor'),
        default='gate-ab',
    )
    parser.add_argument('--cache-dir', type=Path, default=GEN4_CACHE_DIR)
    parser.add_argument('--out-dir', type=Path, default=EXP_SUP_DIR)
    parser.add_argument('--variant', type=str, default='gen4_full')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=6.0e-4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--schedd', type=str, default=None)
    args = parser.parse_args(argv)

    if args.mode == 'gate-ab':
        dev = 'cpu' if args.device == 'cuda:0' and not torch.cuda.is_available() else args.device
        res = run_gate_ab_and_overfit_health(device=dev)
        print(json.dumps(res, indent=2))
        return

    if args.mode == 'prepare-condor':
        info = write_condor_submit_files(cache_dir=args.cache_dir, schedd=args.schedd)
        print(json.dumps(info, indent=2))
        return

    if args.mode == 'stage2-aggregate':
        res = aggregate_stage2_results(out_dir=args.out_dir)
        print(json.dumps(res, indent=2))
        return

    if args.mode == 'eval-stress-diag':
        res = evaluate_stage2_stress_diag(
            cache_dir=args.cache_dir, out_dir=args.out_dir, batch_size=args.batch_size, device_str=args.device
        )
        print(json.dumps(res, indent=2))
        return

    if args.mode == 'ref-l1-gn2':
        res = run_ref_l1_gn2_baseline(
            cache_dir=args.cache_dir,
            out_dir=args.out_dir,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device_str=args.device,
            workers=args.workers,
        )
        print(json.dumps(res, indent=2))
        return

    stage = 1 if args.mode == 'stage1-ablation' else 2
    res = run_supervised_training(
        cache_dir=args.cache_dir,
        out_dir=args.out_dir,
        seed=args.seed,
        stage=stage,
        variant=args.variant,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_str=args.device,
    )
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
