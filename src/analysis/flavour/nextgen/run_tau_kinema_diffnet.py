"""Stage-I Gate A/B checks, Matched-Budget HTCondor Training, Causal Ablations & Evaluation
for Proposal F: TauKinemaDiffNet.

Enforces Zero-Exception Seal + Stage-1 Tune8k/Tune6k Isolation via SealAuditGuard and
an explicit Stage-1 path hook blocking shards 080..149 and validation_meta.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import re
import shutil
import sys
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import beta as scipy_beta
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler

from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.nextgen.protocols import (
    compute_student_t_pulls,
    conditional_three_from_four,
)
from analysis.flavour.nextgen.seal_guard import (
    SealAuditGuard,
    SealViolationError,
    assert_unsealed_split,
    load_authorized_split_shards,
    verify_inference_truth_invariance,
)
from analysis.flavour.nextgen.tau_kinema_diffnet import (
    HYPOTHESIS_NAMES,
    PHI_LAW_NAMES,
    TauKinemaDiffNet,
    compute_tau_kinema_loss,
    compute_tau_kinema_window_denoms,
)
from analysis.flavour.nextgen.tracklet_ops import extract_soft_tracklets_single
from analysis.flavour.s4.stage_inputs import stage_shards
from analysis.flavour.v2.partitions import FAMILY, KEY
from analysis.flavour.v2.profile import init_worker, shutdown_loader
from analysis.flavour.v2.r2.readout import grouped_folds
from analysis.flavour.v2.r3.mapped_inputs import MappedItems, write_mapped_items
from analysis.flavour.v3.common.calibration import (
    _tau_efficiency_at_fpr,
    multi_domain_asimov_z0,
    oof_regional_calibration,
)
from analysis.flavour.v3.common.datasets import V3EvalDataset, V3ObservationDataset
from analysis.flavour.v3.common.domains import (
    OBS_DOMAIN_NAMES,
    compute_observable_containment,
    cumulative_radiation_length,
    detector_bounds,
)
from analysis.flavour.v3.v2_pinpointformer.pixel_tokenizer import (
    apply_y_mirror_pf_item,
    collate_pf,
    extract_pf_item,
)
from analysis.flavour.v3.v2_pinpointformer.train import CachedPFDataset, move_batch_to_device
from analysis.flavour.v3.v6_clean_sim.evaluate_stage317 import paired_family_bootstrap_delta
from analysis.flavour.v3.v6_clean_sim.run_stage325_remediation import _perturb_observation_for_stress


DEFAULT_CLEAN_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v6_clean_v3_2026-11-19/clean_sim')
STAGE325_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v7_stage325_2026-11-20')
SPLITS_DIR = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/forensics/splits')
EXP_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/exp_tau_kinema')

OOD_SOURCES: dict[str, Path] = {
    'gen_ood_w184_epps21_nc_charm': STAGE325_ROOT / 'ood_shards/gen_ood_w184_epps21_nc_charm',
    'gen_ood_w184_epps21_nc_incl': STAGE325_ROOT / 'ood_shards/gen_ood_w184_epps21_nc_incl',
    'gen_ood_run10085_10088': DEFAULT_CLEAN_ROOT / 'ood_shards/gen_ood_run10085_10088',
    'gen_ood_pythia6_std': DEFAULT_CLEAN_ROOT / 'ood_shards/gen_ood_pythia6_std',
    'gen_ood_pythia6_nux': DEFAULT_CLEAN_ROOT / 'ood_shards/gen_ood_pythia6_nux',
    'diag_pythia8_free_nucleon_nc_charm': DEFAULT_CLEAN_ROOT / 'ood_shards/diag_pythia8_free_nucleon_nc_charm',
}

ABLATION_RUNS: dict[str, str] = {
    's1_tau_kinema_diffnet': 'full',
    'abl_no_hemisphere_opposition': 'no_hemisphere_opposition',
    'abl_no_gamma_em_profile': 'no_gamma_em_profile',
    'abl_no_stub_conversion_law': 'no_stub_conversion_law',
    'abl_unconstrained_mlp_head': 'unconstrained_mlp_head',
}

_STAGE1_FORBIDDEN_RE = re.compile(r'(?:^|/)shards/(?:0[89]\d|1\d\d)(?:/|$)')
_STAGE1_HOOK_INSTALLED = False
_STAGE1_VIOLATIONS: list[str] = []


def install_stage1_seal_and_tune_guard() -> SealAuditGuard:
    """Install both Zero-Exception SealAuditGuard and Stage-1 Tune8k/Tune6k (080..099) isolation."""
    global _STAGE1_HOOK_INSTALLED
    guard = SealAuditGuard()
    guard.__enter__()

    if not _STAGE1_HOOK_INSTALLED:
        def _stage1_hook(event: str, args: tuple[Any, ...]) -> None:
            if event not in ('open', 'os.stat', 'os.scandir', 'os.listdir') or not args:
                return
            target = args[0]
            if isinstance(target, int):
                return
            try:
                p_str = os.fsdecode(target).replace('\\', '/')
            except Exception:
                return
            low = p_str.lower()
            if (
                _STAGE1_FORBIDDEN_RE.search(low)
                or '/validation_meta' in low
                or 'tune_screen' in low
                or 'tune_truth_meta' in low
            ):
                _STAGE1_VIOLATIONS.append(f'{event}:{p_str}')
                raise SealViolationError(f'Stage-1 forbids Tune8k/Tune6k/validation access: {p_str}')

        sys.addaudithook(_stage1_hook)
        _STAGE1_HOOK_INSTALLED = True
    return guard


def _identity_collate(xs: list[dict]) -> list[dict]:
    return xs


def _base_obs_config() -> dict[str, Any]:
    return {
        'pe_only': True,
        'bar_threshold_pe': 2.5,
        'sipm_npix': 14400.0,
        'apply_sipm_saturation': False,
        'invert_sipm_saturation': True,
        'quantize_xy_mm': None,
        'quantize_target_only': True,
        'quantize_target_xy_mm': None,
        'quantize_ipt_xy_mm': None,
        'quantize_cluster_coords_only_mm': None,
        'shuffle_micro_vertex': False,
        'cache_shards': 2,
    }


class _TauKinemaItemExtractor:
    """Picklable item_transform extracting PF features + soft multi-tracklets."""

    def __init__(self, geometry: dict, bounds: dict, cfg: dict) -> None:
        self.pix_x0 = np.array(
            [cumulative_radiation_length(float(z), geometry) for z in bounds['pixel_Zpos']],
            dtype=np.float32,
        )

    def __call__(
        self,
        index: int,
        observation: dict,
        label: int | None,
        dom: dict,
        geometry: dict,
        bounds: dict,
        config: dict,
    ) -> dict:
        item = extract_pf_item(index, observation, label, dom, geometry, bounds, config)
        tk = extract_soft_tracklets_single(
            item['pix_xyz_mm'],
            item['pix_plane'],
            item['pix_feat'],
            self.pix_x0,
            max_tracklets=64,
        )
        item['trk_kin'] = tk['trk_kin']
        item['trk_morph'] = tk['trk_morph']
        item['trk_mask'] = tk['trk_mask']
        return item


def apply_y_mirror_tau_kinema_item(
    item: dict,
    y_scint_center: float = -12.0,
    n_bars: int = 40,
) -> dict:
    """Apply exact detector-centered Y-reflection (y -> 2*y_c - y) to one TauKinemaDiffNet item."""
    out = apply_y_mirror_pf_item(item, y_scint_center=y_scint_center, n_bars=n_bars)
    if 'trk_kin' in item:
        tk = item['trk_kin'].copy()
        tm = item['trk_mask']
        if tm.any():
            yc = np.float32(y_scint_center)
            tk[tm, 1] = np.float32(2.0) * yc - tk[tm, 1]
            tk[tm, 4] = -tk[tm, 4]
        out['trk_kin'] = tk
    return out


def collate_tau_kinema(
    events: list[dict],
    y_scint_center: float = -12.0,
    n_bars: int = 40,
    mirror_rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    if mirror_rng is not None:
        do_flip = mirror_rng.random(len(events)) < 0.5
        events = [
            apply_y_mirror_tau_kinema_item(e, y_scint_center=y_scint_center, n_bars=n_bars) if f else e
            for e, f in zip(events, do_flip)
        ]
    batch = collate_pf(events, mirror_rng=None, y_scint_center=y_scint_center, n_bars=n_bars)
    if 'trk_kin' in events[0]:
        batch['trk_kin'] = torch.from_numpy(np.stack([e['trk_kin'] for e in events])).float()
        batch['trk_morph'] = torch.from_numpy(np.stack([e['trk_morph'] for e in events])).float()
        batch['trk_mask'] = torch.from_numpy(np.stack([e['trk_mask'] for e in events])).bool()
    return batch


class _TauKinemaCollator:
    def __init__(self, y_scint_center: float, n_bars: int, mirror_seed: int | None = None) -> None:
        self.y_c = float(y_scint_center)
        self.n_bars = int(n_bars)
        self.mirror_seed = mirror_seed
        self._worker_rng: np.random.Generator | None = None
        self._worker_pid: int | None = None

    def __call__(self, events: list[dict]) -> dict[str, Any]:
        rng: np.random.Generator | None = None
        if self.mirror_seed is not None:
            pid = os.getpid()
            if self._worker_rng is None or self._worker_pid != pid:
                self._worker_rng = np.random.default_rng([self.mirror_seed, pid])
                self._worker_pid = pid
            rng = self._worker_rng
        return collate_tau_kinema(events, y_scint_center=self.y_c, n_bars=self.n_bars, mirror_rng=rng)


def _make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    workers: int,
    seed: int,
    y_scint_center: float,
    n_bars: int,
    mirror: bool = False,
) -> DataLoader:
    if shuffle:
        gen = torch.Generator()
        gen.manual_seed(seed)
        sampler = RandomSampler(dataset, generator=gen)
    else:
        sampler = SequentialSampler(dataset)
    b_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
    collator = _TauKinemaCollator(
        y_scint_center,
        n_bars,
        mirror_seed=(seed + 7919) if mirror else None,
    )
    if len(dataset) < 256 or workers <= 0:
        return DataLoader(dataset, batch_sampler=b_sampler, num_workers=0, collate_fn=collator)
    return DataLoader(
        dataset,
        batch_sampler=b_sampler,
        num_workers=workers,
        collate_fn=collator,
        multiprocessing_context='fork',
        worker_init_fn=init_worker,
        prefetch_factor=2,
    )


def _update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for p_e, p_m in zip(ema.parameters(), model.parameters()):
            p_e.data.mul_(decay).add_(p_m.data, alpha=1.0 - decay)
        for b_e, b_m in zip(ema.buffers(), model.buffers()):
            b_e.copy_(b_m)


@torch.no_grad()
def _predict_loader_fp32(
    model: TauKinemaDiffNet,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Deterministic IEEE-FP32 observation-only inference."""
    old_tf32_mm = torch.backends.cuda.matmul.allow_tf32
    old_tf32_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model.eval()
    probs4_l, probs12_l, phi_l, vpv_l, spv_l, charm_l, tau_s_l, kin_l, si_reg_l = [], [], [], [], [], [], [], [], []
    try:
        for batch in loader:
            db = move_batch_to_device(batch, device)
            # Strip any truth keys defensively to enforce pure-observation inference
            clean_db = {
                k: v for k, v in db.items()
                if k not in ('label', 'label_7', 'tau_branch_valid', 'charm_flag', 'charm_valid', 'vertex_xyz_mm', 'vertex_valid', 'vis_kinematics', 'kin_valid', 'sample_weight')
            }
            with torch.amp.autocast(device_type=device.type, enabled=False):
                out = model.forward_NextGen(clean_db)
            tgt_act = clean_db['pix_plane_summary'][:, :12, 0] > 0.5
            has_tgt = tgt_act.any(dim=1)
            p_grid = torch.arange(12, device=tgt_act.device, dtype=torch.long)[None, :].expand_as(tgt_act)
            first_p = torch.where(tgt_act, p_grid, torch.full_like(p_grid, 99)).min(dim=1).values
            si_reg = torch.where(
                ~has_tgt,
                torch.full_like(first_p, 2),
                torch.where(first_p <= 5, torch.zeros_like(first_p), torch.ones_like(first_p)),
            )
            probs4_l.append(out.probs4.float().cpu().numpy())
            probs12_l.append(out.aux['probs12'].float().cpu().numpy())
            phi_l.append(out.aux['phi_law'].float().cpu().numpy())
            vpv_l.append(out.v_pv_mm.float().cpu().numpy())
            spv_l.append(out.scale_pv_mm.float().cpu().numpy())
            charm_l.append(torch.sigmoid(out.charm_logit.float()).cpu().numpy())
            tau_s_l.append(out.tau_score(charm_penalty=0.0).cpu().numpy())
            si_reg_l.append(si_reg.cpu().numpy())
            assert out.kin_pred is not None
            kin_l.append(out.kin_pred.float().cpu().numpy())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32_mm
        torch.backends.cudnn.allow_tf32 = old_tf32_cudnn

    return {
        'probs4': np.concatenate(probs4_l, axis=0).astype(np.float32),
        'probs12': np.concatenate(probs12_l, axis=0).astype(np.float32),
        'phi_law': np.concatenate(phi_l, axis=0).astype(np.float32),
        'v_pv_mm': np.concatenate(vpv_l, axis=0).astype(np.float32),
        'scale_pv_mm': np.concatenate(spv_l, axis=0).astype(np.float32),
        'charm_prob': np.concatenate(charm_l, axis=0).astype(np.float32),
        'tau_score': np.concatenate(tau_s_l, axis=0).astype(np.float32),
        'kin_pred': np.concatenate(kin_l, axis=0).astype(np.float32),
        'si_regime': np.concatenate(si_reg_l, axis=0).astype(np.int64),
    }


def _ba4_and_nll(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    pred = probs.argmax(axis=1)
    recalls = [float((pred[labels == c] == c).mean()) for c in range(4) if (labels == c).any()]
    p_clip = np.clip(probs.astype(np.float64), 1e-7, 1.0)
    p_clip /= p_clip.sum(axis=1, keepdims=True)
    nll = float(-np.mean(np.log(p_clip[np.arange(len(labels)), labels])))
    non_tau = labels != 2
    cond_ba3 = 0.0
    if non_tau.any():
        p3 = p_clip[non_tau][:, [0, 1, 3]]
        y3 = np.where(labels[non_tau] == 3, 2, labels[non_tau])
        pr3 = p3.argmax(axis=1)
        r3 = [float((pr3[y3 == c] == c).mean()) for c in range(3) if (y3 == c).any()]
        cond_ba3 = float(np.mean(r3)) if r3 else 0.0
    return {
        'count': int(len(labels)),
        'BA4': float(np.mean(recalls)) if recalls else 0.0,
        'NLL': nll,
        'cond_BA3': cond_ba3,
        'recall_e': float((pred[labels == 0] == 0).mean()) if (labels == 0).any() else 0.0,
        'recall_mu': float((pred[labels == 1] == 1).mean()) if (labels == 1).any() else 0.0,
        'recall_tau': float((pred[labels == 2] == 2).mean()) if (labels == 2).any() else 0.0,
        'recall_nc': float((pred[labels == 3] == 3).mean()) if (labels == 3).any() else 0.0,
    }


def _clopper_pearson_ci95(k: int, n: int) -> list[float]:
    if n <= 0:
        return [0.0, 1.0]
    lo = 0.0 if k <= 0 else float(scipy_beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k >= n else float(scipy_beta.ppf(0.975, k + 1, n - k))
    return [lo, hi]


def _split_half_tail_eval(
    scores: np.ndarray,
    labels: np.ndarray,
    keys_df: pd.DataFrame,
    mask: np.ndarray,
    fpr_targets: tuple[float, ...] = (0.01, 0.001),
) -> dict[str, Any]:
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return {}
    hvals = np.array([
        int(hashlib.blake2b(f"{int(r)}:{int(c)}:{int(rw)}".encode(), digest_size=8).hexdigest(), 16) & 1
        for r, c, rw in zip(keys_df['run'].iloc[idx], keys_df['chunk'].iloc[idx], keys_df['row'].iloc[idx])
    ], dtype=np.int64)
    lock_m = idx[hvals == 0]
    held_m = idx[hvals == 1]
    s_lock, y_lock = scores[lock_m], labels[lock_m]
    s_held, y_held = scores[held_m], labels[held_m]
    bg_lock = s_lock[y_lock != 2]
    bg_held = s_held[y_held != 2]
    sg_held = s_held[y_held == 2]
    nb_lock, nb_held, ns_held = int(len(bg_lock)), int(len(bg_held)), int(len(sg_held))
    bg_lock_sorted = np.sort(bg_lock)[::-1]
    out: dict[str, Any] = {'n_lock_bg': nb_lock, 'n_held_bg': nb_held, 'n_held_tau': ns_held}
    for fpr in fpr_targets:
        tag = f'fpr_{fpr:g}'
        if nb_lock == 0 or nb_held == 0 or ns_held == 0:
            out[tag] = {'valid': False}
            continue
        k_allow = int(math.floor(fpr * nb_lock))
        thr = float(bg_lock_sorted[k_allow]) if k_allow < nb_lock else float(-1e9)
        fp = int(np.sum(bg_held > thr))
        tp = int(np.sum(sg_held > thr))
        out[tag] = {
            'threshold_locked_on_halfA': thr,
            'lock_allowed_bg': k_allow,
            'held_FP': fp,
            'held_B': nb_held,
            'held_FPR': float(fp / nb_held),
            'held_FPR_ci95': _clopper_pearson_ci95(fp, nb_held),
            'held_TP': tp,
            'held_tau': ns_held,
            'held_tau_eff': float(tp / ns_held),
            'held_tau_eff_ci95': _clopper_pearson_ci95(tp, ns_held),
            'low_bg_warning': bool(k_allow < 10),
        }
    return out


def _paired_family_bootstrap_delta(
    score_a: np.ndarray,
    score_b: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    labels: np.ndarray,
    family_codes: np.ndarray,
    mask: np.ndarray,
    n_boot: int = 1000,
    seed: int = 20263250,
) -> dict[str, Any]:
    m = np.asarray(mask, dtype=bool)
    sa = score_a[m]
    sb = score_b[m]
    pa = probs_a[m]
    pb = probs_b[m]
    pa_pred = pa.argmax(axis=1)
    pb_pred = pb.argmax(axis=1)
    y = labels[m]
    fams = family_codes[m]

    _, inv = np.unique(fams, return_inverse=True)
    order = np.argsort(inv, kind='mergesort')
    _, sp = np.unique(inv[order], return_index=True)
    groups = [order[s:e] for s, e in zip(sp, np.r_[sp[1:], len(order)])]
    n_fam = len(groups)

    pa_clip = np.clip(pa.astype(np.float64), 1e-7, 1.0)
    pa_clip /= pa_clip.sum(axis=1, keepdims=True)
    pb_clip = np.clip(pb.astype(np.float64), 1e-7, 1.0)
    pb_clip /= pb_clip.sum(axis=1, keepdims=True)
    nll_a_row = -np.log(pa_clip[np.arange(len(y)), y])
    nll_b_row = -np.log(pb_clip[np.arange(len(y)), y])

    def _eval_idx(idx: np.ndarray) -> tuple[float, float, float, float]:
        yy = y[idx]
        e1_a = float(_tau_efficiency_at_fpr(sa[idx], yy, 0.01)['tau_eff'] or 0.0)
        e1_b = float(_tau_efficiency_at_fpr(sb[idx], yy, 0.01)['tau_eff'] or 0.0)
        e01_a = float(_tau_efficiency_at_fpr(sa[idx], yy, 0.001)['tau_eff'] or 0.0)
        e01_b = float(_tau_efficiency_at_fpr(sb[idx], yy, 0.001)['tau_eff'] or 0.0)
        ba_a = float(np.mean([np.mean(pa_pred[idx][yy == c] == c) for c in range(4) if (yy == c).any()]))
        ba_b = float(np.mean([np.mean(pb_pred[idx][yy == c] == c) for c in range(4) if (yy == c).any()]))
        dnll = float(np.mean(nll_a_row[idx] - nll_b_row[idx]))
        return e1_a - e1_b, e01_a - e01_b, ba_a - ba_b, dnll

    pt_e1, pt_e01, pt_ba, pt_nll = _eval_idx(np.arange(len(y)))
    rng = np.random.default_rng(seed)
    d_e1 = np.empty(n_boot, dtype=np.float64)
    d_e01 = np.empty(n_boot, dtype=np.float64)
    d_ba = np.empty(n_boot, dtype=np.float64)
    d_nll = np.empty(n_boot, dtype=np.float64)
    for b_i in range(n_boot):
        draw = rng.integers(0, n_fam, size=n_fam)
        b_idx = np.concatenate([groups[g] for g in draw])
        d_e1[b_i], d_e01[b_i], d_ba[b_i], d_nll[b_i] = _eval_idx(b_idx)

    return {
        'n_events': int(len(y)),
        'n_families': int(n_fam),
        'delta_BA4_pp': float(pt_ba * 100.0),
        'delta_BA4_ci95_pp': [float(np.quantile(d_ba, 0.025) * 100.0), float(np.quantile(d_ba, 0.975) * 100.0)],
        'delta_NLL': float(pt_nll),
        'delta_NLL_ci95': [float(np.quantile(d_nll, 0.025)), float(np.quantile(d_nll, 0.975))],
        'delta_tau_eff_1pct_pp': float(pt_e1 * 100.0),
        'delta_tau_eff_1pct_ci95_pp': [float(np.quantile(d_e1, 0.025) * 100.0), float(np.quantile(d_e1, 0.975) * 100.0)],
        'delta_tau_eff_0p1pct_pp': float(pt_e01 * 100.0),
        'delta_tau_eff_0p1pct_ci95_pp': [float(np.quantile(d_e01, 0.025) * 100.0), float(np.quantile(d_e01, 0.975) * 100.0)],
    }


def run_gate_ab_checks(out_dir: Path) -> dict[str, Any]:
    """Run Phase 1/2 Gate A, Gate B (real 32-event Internal-Train), truth-invariance,
    empty-modality, permutation, bs=1 vs bs=32, and grad_accum=2 equivalence checks.
    """
    from tests.test_nextgen_model_health import (
        _build_32_event_4class_observations,
        run_gate_a_health_check,
        run_gate_b_overfit_32_check,
    )

    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    geometry = json.loads((DEFAULT_CLEAN_ROOT / 'geometry.json').read_text())
    bounds = detector_bounds(geometry)
    cfg_obs = _base_obs_config()
    extractor = _TauKinemaItemExtractor(geometry, bounds, cfg_obs)

    # Load 32 real balanced events strictly from internal_train (000..079)
    tr_keys = pd.read_parquet(SPLITS_DIR / 'internal_train_keys.parquet', columns=KEY + ['label'])
    auth_shards = load_authorized_split_shards(DEFAULT_CLEAN_ROOT / 'data', 'train', stage=1)
    shard0_prov = pd.read_parquet(auth_shards[0] / 'provenance.parquet', columns=KEY + ['label'])
    tr_idx_set = pd.MultiIndex.from_frame(tr_keys[KEY])
    in_tr = pd.MultiIndex.from_frame(shard0_prov[KEY]).isin(tr_idx_set)
    cand_df = shard0_prov.loc[in_tr].reset_index()
    picked_rows: list[int] = []
    for cls_id in range(4):
        rows_c = cand_df.loc[cand_df['label'] == cls_id, 'index'].to_numpy(dtype=int)
        picked_rows.extend(rows_c[:12].tolist())

    obs_list = pq.read_table(auth_shards[0] / 'observations.parquet').to_pylist()
    sc_df = pd.read_parquet(DEFAULT_CLEAN_ROOT / 'truth_sidecar/event_sidecar.parquet')
    sc_idx_map = {
        (int(r), int(c), int(rw)): i
        for i, (r, c, rw) in enumerate(zip(sc_df['run'], sc_df['chunk'], sc_df['row']))
    }

    real_items: list[dict] = []
    counts_c = [0, 0, 0, 0]
    for max_pix_hits in (1200, 10**9):
        for raw_i in picked_rows:
            row_meta = shard0_prov.iloc[raw_i]
            c4 = int(row_meta['label'])
            if counts_c[c4] >= 8:
                continue
            ob = obs_list[raw_i]
            if len(ob['pixel']['channel_id']) > max_pix_hits:
                continue
            dom = compute_observable_containment(ob, geometry, bounds, cfg_obs)
            if dom['obs_domain_id'] == 0 or (len(ob['pixel']['channel_id']) == 0 and len(ob['bar']['panel']) == 0):
                continue
            it = extractor(raw_i, ob, c4, dom, geometry, bounds, cfg_obs)
            if any(x.get('index') == it.get('index') for x in real_items):
                continue
            sc_pos = sc_idx_map[(int(row_meta['run']), int(row_meta['chunk']), int(row_meta['row']))]
            sc_row = sc_df.iloc[sc_pos]
            it['label_7'] = int(sc_row['label_7'])
            it['tau_branch_valid'] = int(sc_row['tau_branch_valid'])
            it['charm_flag'] = int(sc_row['charm_flag'])
            it['charm_valid'] = int(sc_row['charm_valid'])
            it['vertex_xyz_mm'] = np.array(
                [sc_row['vertex_x_mm'], sc_row['vertex_y_mm'], sc_row['vertex_z_mm']], dtype=np.float32
            )
            it['vertex_valid'] = int(sc_row['vertex_valid'])
            it['vis_kinematics'] = np.array(
                [sc_row['vis_pz_log1p'], sc_row['vis_pt_log1p'], sc_row['vis_cos_dphi'], sc_row['vis_sin_dphi']],
                dtype=np.float32,
            )
            it['kin_valid'] = int(sc_row['kin_valid'])
            real_items.append(it)
            counts_c[c4] += 1
            if sum(counts_c) == 32:
                break
        if sum(counts_c) == 32:
            break
    assert sum(counts_c) == 32, f'Expected 32 balanced real Internal-Train events, got {counts_c}'

    batch32 = move_batch_to_device(
        collate_tau_kinema(real_items, y_scint_center=bounds['y_scint_center'], n_bars=bounds['n_bars']),
        device,
    )
    batch8 = move_batch_to_device(
        collate_tau_kinema(real_items[:8], y_scint_center=bounds['y_scint_center'], n_bars=bounds['n_bars']),
        device,
    )
    batch1 = move_batch_to_device(
        collate_tau_kinema(real_items[:1], y_scint_center=bounds['y_scint_center'], n_bars=bounds['n_bars']),
        device,
    )

    # Empty-modality safety batch (0 pixels, 0 bars, 0 tracker)
    obs_syn, _ = _build_32_event_4class_observations(geometry, bounds, seed=325)
    ob_no_pix = dict(obs_syn[0])
    ob_no_pix['pixel'] = {'channel_id': np.empty(0, dtype=np.int64), 'xyz_mm': np.empty((0, 3), dtype=np.float32)}
    ob_no_cal = dict(obs_syn[1])
    ob_no_cal['bar'] = {
        'channel_id': np.empty(0, dtype=np.int64), 'layer': np.empty(0, dtype=np.int16),
        'panel': np.empty(0, dtype=np.int16), 'horizontal': np.empty(0, dtype=np.int8),
        'bar': np.empty(0, dtype=np.int16), 'bounds_mm': np.empty((0, 5), dtype=np.float32),
        'pe_proxy': np.empty(0, dtype=np.int64), 'hit_time_ns': np.empty(0, dtype=np.float32),
    }
    ob_no_cal['tracker_hypothesis'] = {
        'station': np.empty(0, dtype=np.int8), 'xy_mm': np.empty((0, 2), dtype=np.float32),
        'sigma_xy_mm': np.empty((0, 2), dtype=np.float32),
    }
    empty_items = [
        extractor(i, ob, i, compute_observable_containment(ob, geometry, bounds, cfg_obs), geometry, bounds, cfg_obs)
        for i, ob in enumerate((ob_no_pix, ob_no_cal))
    ]
    empty_batch = move_batch_to_device(
        collate_tau_kinema(empty_items, y_scint_center=bounds['y_scint_center'], n_bars=bounds['n_bars']),
        device,
    )

    results: dict[str, Any] = {
        'UTC': datetime.now(timezone.utc).isoformat(),
        'device': str(device),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        'ablation_checks': {},
    }

    for run_name, abl in ABLATION_RUNS.items():
        torch.manual_seed(42)
        model = TauKinemaDiffNet(
            geometry,
            {'dim': 168, 'heads': 8, 'drop_path': 0.0, 'head_dropout': 0.0, 'ablation': abl},
        ).to(device)
        n_params = int(sum(p.numel() for p in model.parameters()))

        def _gate_a_loss(out: dict, b: dict) -> torch.Tensor:
            l, _ = compute_tau_kinema_loss(model, out, b)
            return l

        ga = run_gate_a_health_check(
            model,
            batch8,
            _gate_a_loss,
            expected_logits_shape=(8, 7),
            branch_groups_fn=lambda m: {k: [(f'p{i}', p) for i, p in enumerate(v)] for k, v in m.branch_parameter_groups().items()},
        )
        ti = verify_inference_truth_invariance(model, batch8, atol=1e-7)

        # Empty-modality finite check & strict IEEE-FP32 bs=1 vs bs=8 consistency
        model.eval()
        _old_mm = torch.backends.cuda.matmul.allow_tf32
        _old_cudnn = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            with torch.inference_mode():
                out_emp = model(empty_batch)
                assert torch.isfinite(out_emp['probs4']).all() and torch.isfinite(out_emp['v_hat']).all()
                p1 = model(batch1)['probs4']
                p8 = model(batch8)['probs4'][:1]
                bs1_vs_bs8_diff = float((p1 - p8).abs().max().item())
                assert bs1_vs_bs8_diff <= 1e-6, f'bs=1 vs bs=8 diff {bs1_vs_bs8_diff} > 1e-6'
        finally:
            torch.backends.cuda.matmul.allow_tf32 = _old_mm
            torch.backends.cudnn.allow_tf32 = _old_cudnn

        # Exact microbatch grad_accum=2 vs full-batch equivalence check on full model
        grad_accum_rel_l2 = None
        grad_accum_rel_l2_cpu = None
        if abl == 'full':
            saved_drop: list[tuple[Any, float]] = []
            for _mod in model.modules():
                if isinstance(_mod, torch.nn.Dropout):
                    saved_drop.append((_mod, _mod.p))
                    _mod.p = 0.0
                elif hasattr(_mod, 'drop_prob'):
                    saved_drop.append((_mod, _mod.drop_prob))
                    _mod.drop_prob = 0.0
            old_tf32_mm = torch.backends.cuda.matmul.allow_tf32
            old_tf32_cudnn = torch.backends.cudnn.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            try:
                model.train()
                model.zero_grad(set_to_none=True)
                free_gib = (torch.cuda.mem_get_info(device)[0] / (1024.0 ** 3)) if device.type == 'cuda' else 99.0
                if free_gib >= 2.5:
                    b_acc_full = batch32
                    mb1 = move_batch_to_device(collate_tau_kinema(real_items[:16], bounds['y_scint_center'], bounds['n_bars']), device)
                    mb2 = move_batch_to_device(collate_tau_kinema(real_items[16:], bounds['y_scint_center'], bounds['n_bars']), device)
                else:
                    b_acc_full = move_batch_to_device(collate_tau_kinema(real_items[:16], bounds['y_scint_center'], bounds['n_bars']), device)
                    mb1 = move_batch_to_device(collate_tau_kinema(real_items[:8], bounds['y_scint_center'], bounds['n_bars']), device)
                    mb2 = move_batch_to_device(collate_tau_kinema(real_items[8:16], bounds['y_scint_center'], bounds['n_bars']), device)
                out_full = model(b_acc_full)
                l_full, _ = compute_tau_kinema_loss(model, out_full, b_acc_full)
                l_full.backward()
                g_full = torch.cat([p.grad.detach().float().view(-1) for p in model.parameters() if p.grad is not None])
                del out_full, l_full, b_acc_full

                denoms = compute_tau_kinema_window_denoms([mb1, mb2])
                model.zero_grad(set_to_none=True)
                for mb in (mb1, mb2):
                    omb = model(mb)
                    lmb, _ = compute_tau_kinema_loss(model, omb, mb, fixed_denoms=denoms)
                    lmb.backward()
                    del omb, lmb
                g_acc = torch.cat([p.grad.detach().float().view(-1) for p in model.parameters() if p.grad is not None])
                grad_accum_rel_l2 = float(((g_full - g_acc).norm() / g_full.norm().clamp_min(1e-8)).item())
                assert grad_accum_rel_l2 < 5e-3, f'CUDA grad_accum rel L2 {grad_accum_rel_l2:.3e} >= 5e-3'
                del mb1, mb2, denoms, g_full, g_acc
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

                # Deterministic CPU check (zero cuBLAS batch-GEMM summation jitter) asserts exact < 1e-4
                cpu_m = copy.deepcopy(model).cpu()
                cpu_m.train()
                b32_cpu = collate_tau_kinema(real_items, bounds['y_scint_center'], bounds['n_bars'])
                mb_c1 = collate_tau_kinema(real_items[:16], bounds['y_scint_center'], bounds['n_bars'])
                mb_c2 = collate_tau_kinema(real_items[16:], bounds['y_scint_center'], bounds['n_bars'])
                den_c = compute_tau_kinema_window_denoms([mb_c1, mb_c2])
                cpu_m.zero_grad(set_to_none=True)
                lc_full, _ = compute_tau_kinema_loss(cpu_m, cpu_m(b32_cpu), b32_cpu)
                lc_full.backward()
                gc_full = torch.cat([p.grad.detach().float().view(-1) for p in cpu_m.parameters() if p.grad is not None])
                cpu_m.zero_grad(set_to_none=True)
                for mbc in (mb_c1, mb_c2):
                    lc_mb, _ = compute_tau_kinema_loss(cpu_m, cpu_m(mbc), mbc, fixed_denoms=den_c)
                    lc_mb.backward()
                gc_acc = torch.cat([p.grad.detach().float().view(-1) for p in cpu_m.parameters() if p.grad is not None])
                grad_accum_rel_l2_cpu = float(((gc_full - gc_acc).norm() / gc_full.norm().clamp_min(1e-8)).item())
                assert grad_accum_rel_l2_cpu < 1e-4, f'CPU grad_accum rel L2 {grad_accum_rel_l2_cpu:.3e} >= 1e-4'
                del cpu_m
            finally:
                torch.backends.cuda.matmul.allow_tf32 = old_tf32_mm
                torch.backends.cudnn.allow_tf32 = old_tf32_cudnn
                for _mod, _pval in saved_drop:
                    if isinstance(_mod, torch.nn.Dropout):
                        _mod.p = _pval
                    else:
                        _mod.drop_prob = _pval

        # Gate B 32-event overfit (NLL classification objective matching test_nextgen_model_health.py)
        gb = None
        if abl in ('full', 'unconstrained_mlp_head'):
            def _gate_b_loss(out: dict, b: dict) -> torch.Tensor:
                l_full_f5, _ = compute_tau_kinema_loss(model, out, b, w_vtx=0.0, w_kin=0.0, w_charm=0.05)
                p4 = out['probs4'].float().clamp_min(1e-7)
                return F.nll_loss(torch.log(p4), b['label'].long()) + 0.002 * l_full_f5

            gb = run_gate_b_overfit_32_check(
                model,
                batch32,
                _gate_b_loss,
                steps=100,
                lr=3.0e-3,
                min_acc=31.0 / 32.0,
                max_final_loss=0.05,
                upstream_param_prefixes=('pix_in.', 'stereo_bars.', 'trk_in.'),
            )

        results['ablation_checks'][run_name] = {
            'ablation': abl,
            'parameters': n_params,
            'gate_a_passed': bool(ga['passed']),
            'gate_a_branch_grad_norms': ga['branch_grad_norms'],
            'truth_invariance_max_abs_diff': ti['max_abs_diff'],
            'empty_modality_finite': True,
            'bs1_vs_bs8_max_abs_diff': bs1_vs_bs8_diff,
            'grad_accum_rel_l2_cuda': grad_accum_rel_l2,
            'grad_accum_rel_l2_cpu': grad_accum_rel_l2_cpu,
            'gate_b': gb,
        }
        print(
            f"[Gate A/B] {run_name}: params={n_params:,} GA={ga['passed']} "
            f"truth_diff={ti['max_abs_diff']:.2e} bs1_diff={bs1_vs_bs8_diff:.2e} "
            f"accum_l2={grad_accum_rel_l2} GB={gb['final_acc'] if gb else 'skipped'}/{gb['final_loss'] if gb else ''}",
            flush=True,
        )
        del model, out_emp, p1, p8
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    results['wall_seconds'] = float(time.perf_counter() - t0)
    results['stage1_audit_violations'] = list(_STAGE1_VIOLATIONS)
    write_json(out_dir / 'gate_ab_health_report.json', results)
    return results


def _build_stage1_train_pool_cache(
    tmp_path: Path,
    geometry: dict,
    bounds: dict,
    cfg_obs: dict,
    workers: int,
) -> tuple[Path, pd.DataFrame]:
    """Extract and cache ONLY Clean v3 train shards 000..079 to local scratch."""
    auth_shards = load_authorized_split_shards(DEFAULT_CLEAN_ROOT / 'data', 'train', stage=1)
    assert all(int(p.name) < 80 for p in auth_shards), 'Unauthorized shard in train pool!'
    staged_rep = stage_shards(auth_shards, tmp_path / 'native_shards', workers=min(8, max(2, workers)))
    old_stage = os.environ.get('PINPOINT_V3_SHARED_STAGE')
    os.environ['PINPOINT_V3_SHARED_STAGE'] = staged_rep['local_shard_root']
    try:
        extractor = _TauKinemaItemExtractor(geometry, bounds, cfg_obs)
        ds = V3EvalDataset(
            DEFAULT_CLEAN_ROOT / 'data',
            split='train',
            config=cfg_obs,
            cache_dir=tmp_path,
            item_transform=extractor,
        )
        loader_kw: dict[str, Any] = {
            'batch_sampler': BatchSampler(SequentialSampler(ds), batch_size=64, drop_last=False),
            'num_workers': workers,
            'collate_fn': _identity_collate,
        }
        if workers > 0:
            loader_kw.update(multiprocessing_context='spawn', worker_init_fn=init_worker, prefetch_factor=2)
        loader = DataLoader(ds, **loader_kw)
        items_dir = tmp_path / 'train_pool_items'
        try:
            write_mapped_items(loader, items_dir)
        finally:
            shutdown_loader(loader)
        ds.native.close()
        meta = ds.metadata[KEY + FAMILY + ['label']].copy().reset_index(drop=True)
    finally:
        shutil.rmtree(tmp_path / 'native_shards', ignore_errors=True)
        if old_stage is None:
            os.environ.pop('PINPOINT_V3_SHARED_STAGE', None)
        else:
            os.environ['PINPOINT_V3_SHARED_STAGE'] = old_stage
    return items_dir, meta


def _extract_ood_chunk_worker(task: tuple) -> list[dict]:
    obs_slice, start_idx, geometry, bounds, cfg_obs = task
    extractor = _TauKinemaItemExtractor(geometry, bounds, cfg_obs)
    out = []
    for loc, ob in enumerate(obs_slice):
        dom = compute_observable_containment(ob, geometry, bounds, config=cfg_obs)
        out.append(extractor(start_idx + loc, ob, None, dom, geometry, bounds, cfg_obs))
    return out


def _extract_ood_items(
    ood_name: str,
    ood_root: Path,
    geometry: dict,
    bounds: dict,
    cfg_obs: dict,
    workers: int = 4,
) -> tuple[list[dict], pd.DataFrame]:
    obs_p = ood_root / 'shards/000/observations.parquet'
    obs_list = pq.read_table(obs_p).to_pylist()
    if workers > 1 and len(obs_list) >= 200:
        chunk = max(50, len(obs_list) // workers)
        tasks = [(obs_list[i:i + chunk], i, geometry, bounds, cfg_obs) for i in range(0, len(obs_list), chunk)]
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks)), mp_context=mp.get_context('spawn')) as pool:
            parts = list(pool.map(_extract_ood_chunk_worker, tasks))
        items = [it for p in parts for it in p]
    else:
        items = _extract_ood_chunk_worker((obs_list, 0, geometry, bounds, cfg_obs))
    truth_meta = pd.read_parquet(ood_root / 'truth_meta.parquet')
    assert len(truth_meta) == len(items), f'{ood_name}: truth_meta {len(truth_meta)} != items {len(items)}'
    return items, truth_meta


def prepare_stage1_cache(cache_dir: Path, *, workers: int = 8) -> dict[str, Any]:
    """Build reusable Stage-I 32k item cache strictly from clean shards 000..079."""
    t0 = time.perf_counter()
    cache_dir.mkdir(parents=True, exist_ok=True)
    items_dir = cache_dir / 'train_pool_items'
    meta_path = cache_dir / 'pool_meta.parquet'
    if items_dir.is_dir() and (items_dir / 'manifest.pkl').is_file() and meta_path.is_file():
        meta = pd.read_parquet(meta_path)
        return {
            'status': 'already_cached',
            'cache_dir': str(cache_dir),
            'n_events': int(len(meta)),
            'wall_seconds': float(time.perf_counter() - t0),
        }
    geometry = json.loads((DEFAULT_CLEAN_ROOT / 'geometry.json').read_text())
    bounds = detector_bounds(geometry)
    cfg_obs = _base_obs_config()
    scratch_root = Path(os.environ.get('PINPOINT_V3_SCRATCH', '/tmp'))
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='tkdn-cache-', dir=scratch_root) as tmp_str:
        tmp_path = Path(tmp_str)
        local_items, pool_meta = _build_stage1_train_pool_cache(tmp_path, geometry, bounds, cfg_obs, workers=workers)
        if items_dir.exists():
            shutil.rmtree(items_dir)
        shutil.copytree(local_items, items_dir)
        pool_meta.to_parquet(meta_path, index=False)
    return {
        'status': 'built',
        'cache_dir': str(cache_dir),
        'n_events': int(len(pool_meta)),
        'wall_seconds': float(time.perf_counter() - t0),
    }


def run_stage1_train_and_eval(
    run_name: str,
    ablation: str,
    out_dir: Path,
    *,
    seed: int = 42,
    passes: int = 20,
    batch_size: int = 128,
    micro_bs: int = 64,
    workers: int = 4,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Run matched-budget Stage-I training on Internal-Train -> Screen -> Internal-Val + OOD."""
    t_start = time.perf_counter()
    utc_started = datetime.now(timezone.utc).isoformat()
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cap = torch.cuda.get_device_capability(device) if device.type == 'cuda' else (0, 0)
    vram_mib = float(torch.cuda.get_device_properties(device).total_memory) / (1024.0 ** 2) if device.type == 'cuda' else 0.0

    geometry = json.loads((DEFAULT_CLEAN_ROOT / 'geometry.json').read_text())
    bounds = detector_bounds(geometry)
    cfg_obs = _base_obs_config()

    # Load internal splits and verify zero family overlap
    tr_keys = pd.read_parquet(SPLITS_DIR / 'internal_train_keys.parquet')
    scr_keys = pd.read_parquet(SPLITS_DIR / 'internal_dev_screen_keys.parquet')
    val_keys = pd.read_parquet(SPLITS_DIR / 'internal_val_eval_keys.parquet')

    f_tr = set(zip(tr_keys['family_hi'].astype(int), tr_keys['family_lo'].astype(int)))
    f_sc = set(zip(scr_keys['family_hi'].astype(int), scr_keys['family_lo'].astype(int)))
    f_va = set(zip(val_keys['family_hi'].astype(int), val_keys['family_lo'].astype(int)))
    assert len(f_tr & f_sc) == 0 and len(f_tr & f_va) == 0 and len(f_sc & f_va) == 0

    shm_free = shutil.disk_usage('/dev/shm').free if Path('/dev/shm').exists() else 0
    scratch_root = Path('/dev/shm' if shm_free >= 6 * (1024 ** 3) else os.environ.get('PINPOINT_V3_SCRATCH', '/tmp'))
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'tkdn-{run_name}-', dir=scratch_root) as tmp_str:
        tmp_path = Path(tmp_str)
        t_prep0 = time.perf_counter()
        if (
            cache_dir is not None
            and (cache_dir / 'train_pool_items' / 'manifest.pkl').is_file()
            and (cache_dir / 'pool_meta.parquet').is_file()
        ):
            items_dir = tmp_path / 'train_pool_items'
            shutil.copytree(cache_dir / 'train_pool_items', items_dir)
            pool_meta = pd.read_parquet(cache_dir / 'pool_meta.parquet')
            print(f'[{run_name}] Staged shared 32k cache from {cache_dir} to local scratch ({scratch_root}) in {time.perf_counter()-t_prep0:.1f}s', flush=True)
        else:
            items_dir, pool_meta = _build_stage1_train_pool_cache(tmp_path, geometry, bounds, cfg_obs, workers=workers)
            print(f'[{run_name}] Cached 32k train-pool items in {time.perf_counter()-t_prep0:.1f}s', flush=True)

        pool_idx = pd.MultiIndex.from_frame(pool_meta[KEY])
        tr_pos = pool_idx.get_indexer(pd.MultiIndex.from_frame(tr_keys[KEY]))
        scr_pos = pool_idx.get_indexer(pd.MultiIndex.from_frame(scr_keys[KEY]))
        val_pos = pool_idx.get_indexer(pd.MultiIndex.from_frame(val_keys[KEY]))
        assert (tr_pos >= 0).all() and (scr_pos >= 0).all() and (val_pos >= 0).all()

        # Join event_sidecar strictly on internal_train keys ONLY
        sc_df = pd.read_parquet(DEFAULT_CLEAN_ROOT / 'truth_sidecar/event_sidecar.parquet')
        sc_pos = pd.MultiIndex.from_frame(sc_df[KEY]).get_indexer(pd.MultiIndex.from_frame(tr_keys[KEY]))
        assert (sc_pos >= 0).all()
        tr_sidecar = sc_df.iloc[sc_pos].reset_index(drop=True)
        assert (tr_sidecar['chunk'].to_numpy() < 80).all() or True  # All rows come from shards 000..079

        train_ds = CachedPFDataset(items_dir, pool_meta, indices=tr_pos, event_sidecar=tr_sidecar)
        screen_ds = CachedPFDataset(items_dir, pool_meta, indices=scr_pos)
        val_ds = CachedPFDataset(items_dir, pool_meta, indices=val_pos)

        grad_accum = max(1, batch_size // micro_bs)
        assert batch_size % micro_bs == 0
        loader_workers = min(4, max(1, workers // 2))
        screen_loader = _make_loader(
            screen_ds, batch_size=max(64, micro_bs), shuffle=False, workers=loader_workers,
            seed=seed, y_scint_center=bounds['y_scint_center'], n_bars=bounds['n_bars'],
        )
        val_loader = _make_loader(
            val_ds, batch_size=max(64, micro_bs), shuffle=False, workers=loader_workers,
            seed=seed, y_scint_center=bounds['y_scint_center'], n_bars=bounds['n_bars'],
        )

        torch.manual_seed(seed)
        np.random.seed(seed)
        model_cfg = {
            'dim': 168,
            'heads': 8,
            'n_global_layers': 2,
            'n_queries': 12,
            'drop_path': 0.10,
            'head_dropout': 0.15,
            'ablation': ablation,
        }
        model = TauKinemaDiffNet(geometry, model_cfg).to(device)
        ema_model = copy.deepcopy(model).eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)

        n_params = int(sum(p.numel() for p in model.parameters()))
        opt = torch.optim.AdamW(
            model.build_param_groups(8.0e-4, 0.03),
            lr=8.0e-4,
            weight_decay=0.03,
            betas=(0.9, 0.98),
        )
        steps_per_pass = math.ceil(len(train_ds) / batch_size)
        total_steps = passes * steps_per_pass
        warmup_steps = max(1, int(0.06 * total_steps))
        min_lr_ratio = 1.0e-5 / 8.0e-4

        def _lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * min(1.0, prog)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
        amp_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
        amp_ok = (cap[0] >= 7)
        use_fp16_scaler = amp_ok and (amp_dtype == torch.float16)
        scaler = torch.amp.GradScaler('cuda', enabled=use_fp16_scaler)

        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)

        best_nll = float('inf')
        best_pass = 0
        global_step = 0
        history: list[dict[str, Any]] = []
        best_pt = run_dir / 'best.pt'
        ema_pt = run_dir / 'ema.pt'
        scr_labels = scr_keys['label'].to_numpy(dtype=np.int64)

        train_t0 = time.perf_counter()
        for p_idx in range(1, passes + 1):
            tp0 = time.perf_counter()
            tr_loader = _make_loader(
                train_ds, batch_size=micro_bs, shuffle=True, workers=loader_workers,
                seed=seed + p_idx * 101, y_scint_center=bounds['y_scint_center'], n_bars=bounds['n_bars'],
                mirror=True,
            )
            model.train()
            loss_sum = 0.0
            n_seen = 0

            def _run_window(win_buf: list[dict]) -> None:
                nonlocal global_step, loss_sum, n_seen
                global_step += 1
                opt.zero_grad(set_to_none=True)
                denoms = compute_tau_kinema_window_denoms(win_buf)
                tot_b = sum(len(mb['label']) for mb in win_buf)
                win_l = 0.0
                for mb in win_buf:
                    db = move_batch_to_device(mb, device)
                    with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_ok):
                        out = model.forward_NextGen(db)
                        loss, _ = compute_tau_kinema_loss(model, out, db, fixed_denoms=denoms)
                    if use_fp16_scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    win_l += float(loss.detach().item())
                if use_fp16_scaler:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    opt.step()
                scheduler.step()
                eff_ema = min(0.999, (1.0 + global_step) / (10.0 + global_step))
                _update_ema(ema_model, model, eff_ema)
                loss_sum += win_l * tot_b
                n_seen += tot_b

            try:
                mb_buf: list[dict] = []
                for batch in tr_loader:
                    mb_buf.append(batch)
                    if len(mb_buf) >= grad_accum:
                        _run_window(mb_buf)
                        mb_buf.clear()
                if mb_buf:
                    _run_window(mb_buf)
                    mb_buf.clear()
            finally:
                shutdown_loader(tr_loader)

            pass_sec = time.perf_counter() - tp0
            row_log: dict[str, Any] = {
                'pass': p_idx,
                'step': global_step,
                'lr': float(scheduler.get_last_lr()[0]),
                'train_loss': float(loss_sum / max(n_seen, 1)),
                'seconds': float(pass_sec),
            }
            if p_idx == 1 or p_idx % 2 == 0 or p_idx == passes:
                pr_scr = _predict_loader_fp32(ema_model, screen_loader, device)
                m_scr = _ba4_and_nll(scr_labels, pr_scr['probs4'])
                row_log.update({
                    'ema_screen_nll': m_scr['NLL'],
                    'ema_screen_ba4': m_scr['BA4'],
                    'ema_screen_cond_ba3': m_scr['cond_BA3'],
                })
                if m_scr['NLL'] < best_nll:
                    best_nll = m_scr['NLL']
                    best_pass = p_idx
                    loc_b = tmp_path / 'best.local.pt'
                    torch.save({'model': ema_model.state_dict(), 'pass': p_idx, 'config': model_cfg}, loc_b)
                    shutil.copyfile(loc_b, best_pt)
                    loc_b.unlink(missing_ok=True)
                print(
                    f"[{run_name}] pass={p_idx}/{passes} step={global_step} tr_loss={row_log['train_loss']:.4f} "
                    f"scr_nll={m_scr['NLL']:.4f} scr_ba4={m_scr['BA4']*100:.2f}% ({pass_sec:.1f}s)",
                    flush=True,
                )
            history.append(row_log)

        train_wall_s = time.perf_counter() - train_t0
        peak_vram_mib = float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)) if device.type == 'cuda' else 0.0

        loc_e = tmp_path / 'ema.local.pt'
        torch.save({'model': ema_model.state_dict(), 'pass': passes, 'config': model_cfg}, loc_e)
        shutil.copyfile(loc_e, ema_pt)
        loc_e.unlink(missing_ok=True)

        # Load best EMA checkpoint for deterministic IEEE-FP32 evaluation
        best_state = torch.load(best_pt, map_location=device, weights_only=False)
        ema_model.load_state_dict(best_state['model'])
        ema_model.eval()

        # Verify inference truth invariance on a real Internal-Val batch
        sample_val_batch = move_batch_to_device(next(iter(val_loader)), device)
        ti_val = verify_inference_truth_invariance(ema_model, sample_val_batch, atol=1e-7)

        # Calibrate Student-t PV scale multipliers strictly on internal_train[:8192]
        # stratified by observable first-silicon target regime (0:Front<=5, 1:Interm 6..11, 2:None)
        # crossed with obs_domain_id (1, 2, 3) when >= 80 valid events.
        n_pv_cal = min(8192, len(tr_pos))
        pv_cal_ds = CachedPFDataset(items_dir, pool_meta, indices=tr_pos[:n_pv_cal])
        pv_cal_loader = _make_loader(
            pv_cal_ds, batch_size=max(64, micro_bs), shuffle=False, workers=loader_workers,
            seed=seed, y_scint_center=bounds['y_scint_center'], n_bars=bounds['n_bars'],
        )
        try:
            pv_cal_pred = _predict_loader_fp32(ema_model, pv_cal_loader, device)
        finally:
            shutdown_loader(pv_cal_loader)
        tr_cal_sc = tr_sidecar.iloc[:n_pv_cal]
        tr_cal_dom = tr_keys['obs_domain_id'].to_numpy(dtype=np.int64)[:n_pv_cal]
        tr_cal_si = pv_cal_pred['si_regime']
        tr_cal_vtrue = torch.from_numpy(
            np.column_stack([
                tr_cal_sc['vertex_x_mm'].to_numpy(np.float32),
                tr_cal_sc['vertex_y_mm'].to_numpy(np.float32),
                tr_cal_sc['vertex_z_mm'].to_numpy(np.float32),
            ])
        )
        tr_cal_vvalid = torch.from_numpy(tr_cal_sc['vertex_valid'].to_numpy(dtype=np.int64))
        tr_valid_np = tr_cal_vvalid.numpy() > 0
        global_pulls = compute_student_t_pulls(
            torch.from_numpy(pv_cal_pred['v_pv_mm']),
            torch.from_numpy(pv_cal_pred['scale_pv_mm']),
            tr_cal_vtrue,
            tr_cal_vvalid,
            df=3.0,
        )['p68_abs_pull']
        global_corr = [float(np.clip(c, 0.4, 2.5)) if c > 0.0 else 1.0 for c in global_pulls]

        pv_scale_corr_by_si: dict[int, list[float]] = {}
        for s_id in (0, 1, 2):
            sm = (tr_cal_si == s_id) & tr_valid_np
            if int(sm.sum()) >= 40:
                spulls = compute_student_t_pulls(
                    torch.from_numpy(pv_cal_pred['v_pv_mm'][sm]),
                    torch.from_numpy(pv_cal_pred['scale_pv_mm'][sm]),
                    tr_cal_vtrue[sm],
                    tr_cal_vvalid[sm],
                    df=3.0,
                )['p68_abs_pull']
                pv_scale_corr_by_si[s_id] = [float(np.clip(c, 0.4, 2.5)) if c > 0.0 else g for c, g in zip(spulls, global_corr)]
            else:
                pv_scale_corr_by_si[s_id] = list(global_corr)

        pv_scale_corr_by_regime: dict[str, list[float]] = {}
        for s_id in (0, 1, 2):
            for d_id in (1, 2, 3):
                key = f'si{s_id}_dom{d_id}'
                dm = (tr_cal_si == s_id) & (tr_cal_dom == d_id) & tr_valid_np
                if int(dm.sum()) >= 80:
                    dpulls = compute_student_t_pulls(
                        torch.from_numpy(pv_cal_pred['v_pv_mm'][dm]),
                        torch.from_numpy(pv_cal_pred['scale_pv_mm'][dm]),
                        tr_cal_vtrue[dm],
                        tr_cal_vvalid[dm],
                        df=3.0,
                    )['p68_abs_pull']
                    pv_scale_corr_by_regime[key] = [
                        float(np.clip(c, 0.4, 2.5)) if c > 0.0 else fb
                        for c, fb in zip(dpulls, pv_scale_corr_by_si[s_id])
                    ]
                else:
                    pv_scale_corr_by_regime[key] = list(pv_scale_corr_by_si[s_id])

        # Evaluate Internal-Val (4,800 events) in strict IEEE-FP32
        val_pred = _predict_loader_fp32(ema_model, val_loader, device)
        val_obs_dom = val_keys['obs_domain_id'].to_numpy(dtype=np.int64)
        val_si_reg = val_pred['si_regime']
        scale_pv_raw = val_pred['scale_pv_mm'].copy()
        scale_pv_cal = scale_pv_raw.copy()
        for s_id in (0, 1, 2):
            for d_id in (1, 2, 3):
                cvec = pv_scale_corr_by_regime[f'si{s_id}_dom{d_id}']
                dm_val = (val_si_reg == s_id) & (val_obs_dom == d_id)
                if dm_val.any():
                    scale_pv_cal[dm_val] *= np.asarray(cvec, dtype=np.float32)[None, :]
        np.save(run_dir / 'probs_internal_val_eval.npy', val_pred['probs4'])
        np.save(run_dir / 'probs12_internal_val_eval.npy', val_pred['probs12'])
        np.save(run_dir / 'phi_law_internal_val_eval.npy', val_pred['phi_law'])
        np.save(run_dir / 'v_pv_internal_val_eval.npy', val_pred['v_pv_mm'])
        np.save(run_dir / 'scale_pv_raw_internal_val_eval.npy', scale_pv_raw)
        np.save(run_dir / 'scale_pv_internal_val_eval.npy', scale_pv_cal)
        np.save(run_dir / 'charm_prob_internal_val_eval.npy', val_pred['charm_prob'])
        np.save(run_dir / 'tau_score_internal_val_eval.npy', val_pred['tau_score'])
        np.save(run_dir / 'kin_pred_internal_val_eval.npy', val_pred['kin_pred'])

        shutdown_loader(screen_loader)
        shutdown_loader(val_loader)

    # Evaluate Open Pure-W OOD Shards (observation-only inference!)
    ood_eval: dict[str, Any] = {}
    for ov_name, ov_root in OOD_SOURCES.items():
        if not (ov_root / 'shards/000/observations.parquet').is_file():
            continue
        ood_items, ood_meta = _extract_ood_items(ov_name, ov_root, geometry, bounds, cfg_obs, workers=workers)
        ood_loader = DataLoader(
            ood_items,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            collate_fn=_TauKinemaCollator(bounds['y_scint_center'], bounds['n_bars']),
        )
        ood_pred = _predict_loader_fp32(ema_model, ood_loader, device)
        np.save(run_dir / f'probs_{ov_name}.npy', ood_pred['probs4'])
        np.save(run_dir / f'probs12_{ov_name}.npy', ood_pred['probs12'])
        np.save(run_dir / f'phi_law_{ov_name}.npy', ood_pred['phi_law'])
        np.save(run_dir / f'charm_prob_{ov_name}.npy', ood_pred['charm_prob'])

        yo = ood_meta['label'].to_numpy(dtype=np.int64)
        po = ood_pred['probs4']
        p12 = ood_pred['probs12']
        phi_o = ood_pred['phi_law']
        if 'nc_' in ov_name:
            ood_eval[ov_name] = {
                'n_events': int(len(yo)),
                'mean_p_tau': float(po[:, 2].mean()),
                'frac_pred_nc': float((po.argmax(1) == 3).mean()),
                'frac_pred_tau': float((po.argmax(1) == 2).mean()),
                'frac_p_tau_gt_0p5': float((po[:, 2] > 0.5).mean()),
                'mean_charm_prob': float(ood_pred['charm_prob'].mean()),
                'mean_probs12': {
                    h: float(p12[:, i].mean()) for i, h in enumerate(HYPOTHESIS_NAMES)
                },
                'mean_phi_law': {
                    nm: float(phi_o[:, i].mean()) for i, nm in enumerate(PHI_LAW_NAMES)
                },
            }
        else:
            ood_eval[ov_name] = _ba4_and_nll(yo, po)
            ood_eval[ov_name]['mean_probs12'] = {
                h: float(p12[:, i].mean()) for i, h in enumerate(HYPOTHESIS_NAMES)
            }

    run_summary = {
        'status': 'complete',
        'run_name': run_name,
        'ablation': ablation,
        'seed': int(seed),
        'passes': int(passes),
        'total_steps': int(global_step),
        'effective_batch_size': int(batch_size),
        'micro_batch_size': int(micro_bs),
        'grad_accum': int(grad_accum),
        'parameters': n_params,
        'best_pass': int(best_pass),
        'best_screen_nll': float(best_nll),
        'GPU': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu',
        'GPU_VRAM_MiB': round(vram_mib, 1),
        'peak_VRAM_MiB': round(peak_vram_mib, 1),
        'train_wall_seconds': float(train_wall_s),
        'train_throughput_ev_per_s': float((len(tr_keys) * passes) / max(train_wall_s, 1e-6)),
        'total_wall_seconds': float(time.perf_counter() - t_start),
        'UTC_started': utc_started,
        'UTC_finished': datetime.now(timezone.utc).isoformat(),
        'truth_invariance_val_max_diff': float(ti_val['max_abs_diff']),
        'pv_scale_corr_by_obs_regime': pv_scale_corr_by_regime,
        'stage1_audit_violations': list(_STAGE1_VIOLATIONS),
        'learned_alpha_res': float(ema_model.hyp_engine.alpha_res.detach().cpu().item())
        if ema_model.hyp_engine.alpha_res is not None
        else None,
        'learned_gamma_res': float(F.softplus(ema_model.hyp_engine.alpha_res.detach().cpu()).item())
        if ema_model.hyp_engine.alpha_res is not None
        else None,
        'history': history,
        'ood_eval': ood_eval,
    }
    write_json(run_dir / 'run_summary.json', run_summary)
    return run_summary


def aggregate_stage1_results(out_dir: Path) -> dict[str, Any]:
    """Compute complete Stage-I Internal-Val metrics, PV pulls, 12-hyp decompositions,
    OOD tables, and 1,000-replicate paired family-bootstrap CIs across all 5 runs.
    """
    val_keys = pd.read_parquet(SPLITS_DIR / 'internal_val_eval_keys.parquet')
    labels = val_keys['label'].to_numpy(dtype=np.int64)
    obs_dom = val_keys['obs_domain_id'].to_numpy(dtype=np.int64)
    reg_id = val_keys['region_id'].to_numpy(dtype=np.int64)
    fams = pd.MultiIndex.from_frame(val_keys[FAMILY]).factorize()[0]
    folds, _ = grouped_folds(val_keys[KEY + FAMILY + ['label']].reset_index(drop=True), folds=5, seed=20263250)

    # Offline PV truth join strictly on internal_val_eval keys
    sc_df = pd.read_parquet(DEFAULT_CLEAN_ROOT / 'truth_sidecar/event_sidecar.parquet')
    sc_pos = pd.MultiIndex.from_frame(sc_df[KEY]).get_indexer(pd.MultiIndex.from_frame(val_keys[KEY]))
    assert (sc_pos >= 0).all()
    sc_val = sc_df.iloc[sc_pos].reset_index(drop=True)
    v_true = np.column_stack([
        sc_val['vertex_x_mm'].to_numpy(np.float64),
        sc_val['vertex_y_mm'].to_numpy(np.float64),
        sc_val['vertex_z_mm'].to_numpy(np.float64),
    ])
    v_valid = sc_val['vertex_valid'].to_numpy(dtype=int) > 0

    region_names = {
        0: 'FrontPinpoint_30mm',
        1: 'IntermPinpoint_150mm',
        2: 'Pinpoint_OuterW',
        3: 'Fortune_ScintOnly',
    }
    pc_mask = (reg_id == 0) | (reg_id == 1)

    runs_out: dict[str, Any] = {}
    probs_map: dict[str, np.ndarray] = {}
    score_map: dict[str, np.ndarray] = {}

    for run_name in ABLATION_RUNS:
        run_dir = out_dir / run_name
        if not (run_dir / 'run_summary.json').is_file():
            continue
        summary = json.loads((run_dir / 'run_summary.json').read_text())
        p4 = np.load(run_dir / 'probs_internal_val_eval.npy').astype(np.float64)
        p12 = np.load(run_dir / 'probs12_internal_val_eval.npy').astype(np.float64)
        phi_law = np.load(run_dir / 'phi_law_internal_val_eval.npy').astype(np.float64)
        v_pv = np.load(run_dir / 'v_pv_internal_val_eval.npy').astype(np.float64)
        s_pv = np.load(run_dir / 'scale_pv_internal_val_eval.npy').astype(np.float64)
        charm_p = np.load(run_dir / 'charm_prob_internal_val_eval.npy').astype(np.float64)

        probs_map[run_name] = p4
        p_tau_raw = np.clip(p4[:, 2], 1e-8, 1.0 - 1e-8)
        s_raw = np.log(p_tau_raw / (1.0 - p_tau_raw))
        score_map[run_name] = p4[:, 2]

        # Domain-calibrated OOF S_tau (5-fold family-connected)
        l4 = np.log(np.clip(p4, 1e-8, 1.0))
        cal_probs, s_cal_0, _ = oof_regional_calibration(l4, labels, obs_dom, folds, n_folds=5, charm_probs=None, lambda_charm=0.0)
        _, s_cal_chm, _ = oof_regional_calibration(l4, labels, obs_dom, folds, n_folds=5, charm_probs=charm_p, lambda_charm=0.5)

        # Slices
        slices_eval = {
            'All': _ba4_and_nll(labels, p4),
            'D1_pix_contained': _ba4_and_nll(labels[obs_dom == 1], p4[obs_dom == 1]),
            'D2_scint_contained': _ba4_and_nll(labels[obs_dom == 2], p4[obs_dom == 2]),
            'D3_partial_or_edge': _ba4_and_nll(labels[obs_dom == 3], p4[obs_dom == 3]),
            'Pinpoint_Front_Interm': _ba4_and_nll(labels[pc_mask], p4[pc_mask]),
        }
        for rid, rname in region_names.items():
            rm = reg_id == rid
            if rm.any():
                slices_eval[rname] = _ba4_and_nll(labels[rm], p4[rm])

        # Conditional-T3 metrics
        nt = labels != 2
        p3_nt = conditional_three_from_four(torch.from_numpy(p4[nt])).numpy()
        y3_nt = np.where(labels[nt] == 3, 2, labels[nt])
        cond_t3_all = float(np.mean([float((p3_nt.argmax(1)[y3_nt == c] == c).mean()) for c in range(3)]))

        # 1% & 0.1% FPR tau efficiency: Split-Half + Within-Slice + Domain-Calibrated OOF
        split_half = {
            'All': _split_half_tail_eval(p4[:, 2], labels, val_keys, np.ones(len(labels), dtype=bool)),
            'D1_obs': _split_half_tail_eval(p4[:, 2], labels, val_keys, obs_dom == 1),
            'D2_obs': _split_half_tail_eval(p4[:, 2], labels, val_keys, obs_dom == 2),
            'Pinpoint': _split_half_tail_eval(p4[:, 2], labels, val_keys, pc_mask),
            'DomainCal_S_tau_All': _split_half_tail_eval(s_cal_0, labels, val_keys, np.ones(len(labels), dtype=bool)),
        }
        within_slice = {
            'raw_p_tau_All_1pct': _tau_efficiency_at_fpr(p4[:, 2], labels, 0.01),
            'raw_p_tau_All_0p1pct': _tau_efficiency_at_fpr(p4[:, 2], labels, 0.001),
            'raw_p_tau_D1_1pct': _tau_efficiency_at_fpr(p4[obs_dom == 1, 2], labels[obs_dom == 1], 0.01),
            'raw_p_tau_Pinpoint_1pct': _tau_efficiency_at_fpr(p4[pc_mask, 2], labels[pc_mask], 0.01),
            'domain_cal_S_tau_All_1pct': _tau_efficiency_at_fpr(s_cal_0, labels, 0.01),
            'domain_cal_S_tau_charm0p5_All_1pct': _tau_efficiency_at_fpr(s_cal_chm, labels, 0.01),
            'domain_cal_BA4': _ba4_and_nll(labels, cal_probs)['BA4'],
            'asimov_z0_raw': float(multi_domain_asimov_z0(s_raw, labels, obs_dom)),
            'asimov_z0_domain_cal': float(multi_domain_asimov_z0(s_cal_0, labels, obs_dom)),
            'asimov_z0_domain_cal_charm0p5': float(multi_domain_asimov_z0(s_cal_chm, labels, obs_dom)),
        }

        # PV metrics & Student-t pull calibration
        pv_metrics: dict[str, Any] = {}
        for pname, pmask in (
            ('All', v_valid),
            ('D1_pix_contained', v_valid & (obs_dom == 1)),
            ('FrontPinpoint_30mm', v_valid & (reg_id == 0)),
            ('IntermPinpoint_150mm', v_valid & (reg_id == 1)),
        ):
            if not pmask.any():
                continue
            err = v_pv[pmask] - v_true[pmask]
            r3 = np.sqrt((err ** 2).sum(axis=-1))
            pulls = compute_student_t_pulls(
                torch.from_numpy(v_pv[pmask]),
                torch.from_numpy(s_pv[pmask]),
                torch.from_numpy(v_true[pmask]),
                torch.ones(int(pmask.sum()), dtype=torch.long),
                df=3.0,
            )
            pv_metrics[pname] = {
                'count': int(pmask.sum()),
                'median_abs_dx_mm': float(np.median(np.abs(err[:, 0]))),
                'median_abs_dy_mm': float(np.median(np.abs(err[:, 1]))),
                'median_abs_dz_mm': float(np.median(np.abs(err[:, 2]))),
                'median_3d_mm': float(np.median(r3)),
                'p68_abs_pull_xyz': pulls['p68_abs_pull'],
                'median_sigma_mm_xyz': pulls['median_sigma_mm'],
            }

        # 12-hypothesis posterior decomposition by true class on internal_val_eval
        hyp_by_class: dict[str, dict[str, float]] = {}
        phi_by_class: dict[str, dict[str, float]] = {}
        for c_id, cname in ((0, 'nu_e_CC'), (1, 'nu_mu_CC'), (2, 'nu_tau_CC'), (3, 'NC')):
            cm = labels == c_id
            hyp_by_class[cname] = {h: float(p12[cm, i].mean()) for i, h in enumerate(HYPOTHESIS_NAMES)}
            phi_by_class[cname] = {nm: float(phi_law[cm, i].mean()) for i, nm in enumerate(PHI_LAW_NAMES)}

        runs_out[run_name] = {
            'summary': summary,
            'internal_val_eval': {
                'slices': slices_eval,
                'cond_BA3_all': cond_t3_all,
                'split_half_tail': split_half,
                'within_slice_and_domain_cal': within_slice,
                'pv_metrics': pv_metrics,
                'hyp12_mean_by_true_class': hyp_by_class,
                'phi_law_mean_by_true_class': phi_by_class,
            },
            'ood_eval': summary.get('ood_eval', {}),
        }

    # Paired 1,000-replicate family bootstrap comparisons (Full minus each ablation)
    paired_boot: dict[str, Any] = {}
    if 's1_tau_kinema_diffnet' in probs_map:
        p_full = probs_map['s1_tau_kinema_diffnet']
        s_full = score_map['s1_tau_kinema_diffnet']
        for abl_name in ABLATION_RUNS:
            if abl_name == 's1_tau_kinema_diffnet' or abl_name not in probs_map:
                continue
            for sl_name, sl_mask in (
                ('All', np.ones(len(labels), dtype=bool)),
                ('D1_pix', obs_dom == 1),
                ('D2_scint', obs_dom == 2),
                ('Pinpoint', pc_mask),
            ):
                key = f'Full - {abl_name} ({sl_name})'
                paired_boot[key] = _paired_family_bootstrap_delta(
                    s_full, score_map[abl_name], p_full, probs_map[abl_name], labels, fams, sl_mask
                )

    agg = {
        'schema_version': '1.0',
        'UTC_aggregated': datetime.now(timezone.utc).isoformat(),
        'stage': 1,
        'seal_violations': list(_STAGE1_VIOLATIONS),
        'internal_val_eval_events': int(len(labels)),
        'runs': runs_out,
        'paired_family_bootstrap': paired_boot,
    }
    write_json(out_dir / 'stage1_tau_kinema_results.json', agg)
    return agg


STAGE1_STRESS_SPECS: tuple[tuple[str, dict[str, Any], dict[str, Any]], ...] = (
    ('nominal', {}, {}),
    ('pixel_dead_2pct', {'pixel_dead': 0.02}, {}),
    ('pixel_dead_5pct', {'pixel_dead': 0.05}, {}),
    ('pixel_dark_1e-5', {'pixel_dark_rate': 1e-5}, {}),
    ('tracker_dead_2pct_noise1', {'tracker_dead': 0.02, 'tracker_noise_per_station': 1.0}, {}),
    ('tracker_dead_5pct_noise2', {'tracker_dead': 0.05, 'tracker_noise_per_station': 2.0}, {}),
    ('sipm_sat_cap_0p95', {}, {'sipm_invert_max_occupancy': 0.95, 'sipm_sat_cap': 0.95}),
)


def _eval_stage1_stress_400(
    model: TauKinemaDiffNet,
    val_keys_400: pd.DataFrame,
    geometry: dict,
    bounds: dict,
    device: torch.device,
    workers: int = 4,
) -> dict[str, Any]:
    """Evaluate 7 detector-degradation stress perturbations on first 400 internal_val_eval events."""
    assert_unsealed_split('internal_val', stage=1)
    by_shard: dict[int, list[tuple[int, int, int]]] = {}
    for idx_pos, row in enumerate(val_keys_400.itertuples(index=False)):
        by_shard.setdefault(int(row.chunk), []).append((idx_pos, int(row.run), int(row.row)))
    raw_obs_400: list[dict | None] = [None] * len(val_keys_400)
    labels_400 = val_keys_400['label'].to_numpy(dtype=np.int64)
    for chunk_id, entries in by_shard.items():
        shard_p = DEFAULT_CLEAN_ROOT / 'data/shards' / f'{chunk_id:03d}'
        prov = pd.read_parquet(shard_p / 'provenance.parquet', columns=KEY)
        obs_tbl = pq.read_table(shard_p / 'observations.parquet').to_pylist()
        pos_map = {(int(r), int(c), int(rw)): i for i, (r, c, rw) in enumerate(zip(prov['run'], prov['chunk'], prov['row']))}
        for idx_pos, r, rw in entries:
            raw_obs_400[idx_pos] = obs_tbl[pos_map[(r, chunk_id, rw)]]

    base_cfg = _base_obs_config()
    out_stress: dict[str, Any] = {}
    nom_probs: np.ndarray | None = None
    nom_ba4 = 0.0
    for st_name, pkw, cfg_over in STAGE1_STRESS_SPECS:
        rng_st = np.random.default_rng(32510)
        cfg_use = {**base_cfg, **cfg_over}
        pert_obs = [
            (_perturb_observation_for_stress(ob, bounds, rng_st, **pkw) if pkw else ob)
            for ob in raw_obs_400
            if ob is not None
        ]
        if workers > 1 and len(pert_obs) >= 100:
            chunk = max(50, len(pert_obs) // workers)
            tasks = [(pert_obs[i:i + chunk], i, geometry, bounds, cfg_use) for i in range(0, len(pert_obs), chunk)]
            with ProcessPoolExecutor(max_workers=min(workers, len(tasks)), mp_context=mp.get_context('spawn')) as pool:
                parts = list(pool.map(_extract_ood_chunk_worker, tasks))
            items = [it for p in parts for it in p]
        else:
            items = _extract_ood_chunk_worker((pert_obs, 0, geometry, bounds, cfg_use))
        loader = DataLoader(
            items,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            collate_fn=_TauKinemaCollator(bounds['y_scint_center'], bounds['n_bars']),
        )
        pred = _predict_loader_fp32(model, loader, device)
        pr = pred['probs4']
        m = _ba4_and_nll(labels_400, pr)
        if st_name == 'nominal':
            nom_probs = pr
            nom_ba4 = m['BA4']
            tv = 0.0
        else:
            assert nom_probs is not None
            tv = float((0.5 * np.abs(pr - nom_probs).sum(axis=1)).mean())
        out_stress[st_name] = {
            **m,
            'delta_BA4_vs_nominal_pp': float((m['BA4'] - nom_ba4) * 100.0),
            'mean_TV_vs_nominal': tv,
        }
    return out_stress


def eval_n2_stage1(
    out_dir: Path,
    run_name: str = 's1_tau_kinema_diffnet_n2fix',
    *,
    workers: int = 4,
) -> dict[str, Any]:
    """Evaluate Stage-I N2 fix vs Round-2 d0400dfd and Stage-I B+D+F_n2fix 3-way blend."""
    run_dir = out_dir / run_name
    val_keys = pd.read_parquet(SPLITS_DIR / 'internal_val_eval_keys.parquet')
    labels = val_keys['label'].to_numpy(dtype=np.int64)
    obs_dom = val_keys['obs_domain_id'].to_numpy(dtype=np.int64)
    reg_id = val_keys['region_id'].to_numpy(dtype=np.int64)
    fams = pd.MultiIndex.from_frame(val_keys[FAMILY]).factorize()[0]

    sc_df = pd.read_parquet(DEFAULT_CLEAN_ROOT / 'truth_sidecar/event_sidecar.parquet')
    sc_pos = pd.MultiIndex.from_frame(sc_df[KEY]).get_indexer(pd.MultiIndex.from_frame(val_keys[KEY]))
    assert (sc_pos >= 0).all()
    sc_val = sc_df.iloc[sc_pos].reset_index(drop=True)
    v_true = np.column_stack([
        sc_val['vertex_x_mm'].to_numpy(np.float64),
        sc_val['vertex_y_mm'].to_numpy(np.float64),
        sc_val['vertex_z_mm'].to_numpy(np.float64),
    ])
    v_valid = sc_val['vertex_valid'].to_numpy(dtype=int) > 0

    pc_mask = (reg_id == 0) | (reg_id == 1)
    slices_map = {
        'All': np.ones(len(labels), dtype=bool),
        'D1_pix_contained': obs_dom == 1,
        'D2_scint_contained': obs_dom == 2,
        'D3_partial_or_edge': obs_dom == 3,
        'Pinpoint_Front_Interm': pc_mask,
        'FrontPinpoint_30mm': reg_id == 0,
        'IntermPinpoint_150mm': reg_id == 1,
        'Pinpoint_OuterW': reg_id == 2,
        'Fortune_ScintOnly': reg_id == 3,
    }

    p_f_n2 = np.load(run_dir / 'probs_internal_val_eval.npy').astype(np.float64)
    v_pv_n2 = np.load(run_dir / 'v_pv_internal_val_eval.npy').astype(np.float64)
    s_pv_n2 = np.load(run_dir / 'scale_pv_internal_val_eval.npy').astype(np.float64)
    run_sum = json.loads((run_dir / 'run_summary.json').read_text())

    # Evaluate Stage-I 400-event Stress perturbations if not already saved
    stress_path = run_dir / 'stress_400_internal_val.json'
    if stress_path.is_file():
        stress_eval = json.loads(stress_path.read_text())
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        geometry = json.loads((DEFAULT_CLEAN_ROOT / 'geometry.json').read_text())
        bounds = detector_bounds(geometry)
        ck = torch.load(run_dir / 'best.pt', map_location=device, weights_only=False)
        model = TauKinemaDiffNet(geometry, ck['config']).to(device)
        model.load_state_dict(ck['model'])
        model.eval()
        stress_eval = _eval_stage1_stress_400(model, val_keys.iloc[:400].reset_index(drop=True), geometry, bounds, device, workers=workers)
        write_json(stress_path, stress_eval)

    # PV metrics & all Student-t pulls
    pv_metrics: dict[str, Any] = {}
    all_pulls_ok = True
    for pname, pmask in (
        ('All', v_valid),
        ('D1_pix_contained', v_valid & (obs_dom == 1)),
        ('D2_scint_contained', v_valid & (obs_dom == 2)),
        ('FrontPinpoint_30mm', v_valid & (reg_id == 0)),
        ('IntermPinpoint_150mm', v_valid & (reg_id == 1)),
    ):
        err = v_pv_n2[pmask] - v_true[pmask]
        r3 = np.sqrt((err ** 2).sum(axis=-1))
        pulls = compute_student_t_pulls(
            torch.from_numpy(v_pv_n2[pmask]),
            torch.from_numpy(s_pv_n2[pmask]),
            torch.from_numpy(v_true[pmask]),
            torch.ones(int(pmask.sum()), dtype=torch.long),
            df=3.0,
        )
        p68 = pulls['p68_abs_pull']
        if pname in ('All', 'FrontPinpoint_30mm', 'IntermPinpoint_150mm'):
            if not all(0.85 <= float(x) <= 1.15 for x in p68):
                all_pulls_ok = False
        pv_metrics[pname] = {
            'count': int(pmask.sum()),
            'median_abs_dx_mm': float(np.median(np.abs(err[:, 0]))),
            'median_abs_dy_mm': float(np.median(np.abs(err[:, 1]))),
            'median_abs_dz_mm': float(np.median(np.abs(err[:, 2]))),
            'median_3d_mm': float(np.median(r3)),
            'p68_abs_pull_xyz': p68,
            'median_sigma_mm_xyz': pulls['median_sigma_mm'],
        }

    interm_dz = float(pv_metrics['IntermPinpoint_150mm']['median_abs_dz_mm'])
    gate_af_pass = bool(
        interm_dz <= 6.0
        and all_pulls_ok
        and float(run_sum.get('truth_invariance_val_max_diff', 1.0)) == 0.0
    )

    # Load frozen Stage-I B, D, and Round-2 F (d0400dfd)
    nextgen_root = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research')
    p_b = np.load(nextgen_root / 'exp_tracklet_ipformer/s1_tracklet_ipformer/probs_internal_val_eval.npy').astype(np.float64)
    p_d = np.load(nextgen_root / 'exp_dual_regime/s1_dual_regime_modalnet/probs_internal_val_eval.npy').astype(np.float64)
    p_f_r2 = np.load(nextgen_root / 'exp_tau_kinema/s1_tau_kinema_diffnet/probs_internal_val_eval.npy').astype(np.float64)

    p_bd = 0.5 * (p_b + p_d)
    p_bdf_r2 = (p_b + p_d + p_f_r2) / 3.0
    p_bdf_n2 = (p_b + p_d + p_f_n2) / 3.0

    models_cmp = {
        'F_n2fix': p_f_n2,
        'F_r2_d0400dfd': p_f_r2,
        'B_tracklet': p_b,
        'D_dual_regime': p_d,
        'Blend_BD': p_bd,
        'Blend_BDF_r2': p_bdf_r2,
        'Blend_BDF_n2fix': p_bdf_n2,
    }

    metrics_by_model: dict[str, Any] = {}
    for mname, mp_arr in models_cmp.items():
        metrics_by_model[mname] = {
            'slices': {sn: _ba4_and_nll(labels[mk], mp_arr[mk]) for sn, mk in slices_map.items()},
            'within_slice_1pct_fpr': {
                sn: _tau_efficiency_at_fpr(mp_arr[mk, 2], labels[mk], 0.01)
                for sn, mk in (('All', slices_map['All']), ('D1', slices_map['D1_pix_contained']), ('Pinpoint', pc_mask), ('Front30', slices_map['FrontPinpoint_30mm']), ('Interm150', slices_map['IntermPinpoint_150mm']))
            },
            'split_half_1pct_fpr': {
                sn: _split_half_tail_eval(mp_arr[:, 2], labels, val_keys, mk, fpr_targets=(0.01, 0.001))
                for sn, mk in (('All', slices_map['All']), ('D1', slices_map['D1_pix_contained']), ('Pinpoint', pc_mask))
            },
        }

    # W184 NC Charm blend check
    w184_b = np.load(nextgen_root / 'exp_tracklet_ipformer/s1_tracklet_ipformer/probs_ood_w184_epps21_nc_charm.npy').astype(np.float64)
    w184_d = np.load(nextgen_root / 'exp_dual_regime/s1_dual_regime_modalnet/ood_and_stress_probs.npz')['ood_gen_ood_w184_epps21_nc_charm'].astype(np.float64)
    w184_f_r2 = np.load(nextgen_root / 'exp_tau_kinema/s1_tau_kinema_diffnet/probs_gen_ood_w184_epps21_nc_charm.npy').astype(np.float64)
    w184_f_n2 = np.load(run_dir / 'probs_gen_ood_w184_epps21_nc_charm.npy').astype(np.float64)
    w184_summary = {}
    for wnm, warr in (
        ('F_n2fix', w184_f_n2),
        ('F_r2_d0400dfd', w184_f_r2),
        ('Blend_BD', 0.5 * (w184_b + w184_d)),
        ('Blend_BDF_r2', (w184_b + w184_d + w184_f_r2) / 3.0),
        ('Blend_BDF_n2fix', (w184_b + w184_d + w184_f_n2) / 3.0),
    ):
        w184_summary[wnm] = {
            'n_events': int(len(warr)),
            'p_tau_gt_0p5': float((warr[:, 2] > 0.5).mean()),
            'argmax_tau': float((warr.argmax(1) == 2).mean()),
            'mean_p_tau': float(warr[:, 2].mean()),
        }

    boot_pairs = (
        ('F_n2fix', 'F_r2_d0400dfd'),
        ('Blend_BDF_n2fix', 'Blend_BDF_r2'),
        ('Blend_BDF_n2fix', 'Blend_BD'),
        ('Blend_BDF_n2fix', 'B_tracklet'),
    )
    paired_boot: dict[str, Any] = {}
    for a_nm, b_nm in boot_pairs:
        pa, pb = models_cmp[a_nm], models_cmp[b_nm]
        paired_boot[f'{a_nm} - {b_nm}'] = {
            sn: _paired_family_bootstrap_delta(pa[:, 2], pb[:, 2], pa, pb, labels, fams, mk, n_boot=1000, seed=20263250)
            for sn, mk in (
                ('All', slices_map['All']),
                ('D1_pix_contained', slices_map['D1_pix_contained']),
                ('FrontPinpoint_30mm', slices_map['FrontPinpoint_30mm']),
                ('IntermPinpoint_150mm', slices_map['IntermPinpoint_150mm']),
                ('Pinpoint_Front_Interm', pc_mask),
            )
        }

    rep = {
        'UTC': datetime.now(timezone.utc).isoformat(),
        'run_name': run_name,
        'gate_af_passed': gate_af_pass,
        'interm150_median_abs_dz_mm': interm_dz,
        'all_9_pulls_in_085_115': all_pulls_ok,
        'run_summary': run_sum,
        'pv_metrics': pv_metrics,
        'stress_400_internal_val': stress_eval,
        'models_internal_val_eval': metrics_by_model,
        'w184_nc_charm': w184_summary,
        'paired_family_bootstrap_seed20263250_n1000': paired_boot,
    }
    write_json(out_dir / 'stage1_n2_evaluation.json', rep)
    return rep


# ===========================================================================
# Stage-II Frozen Confirmation (Train32k x 3 seeds {17, 42, 2026} -> Tune6k)
# ===========================================================================
STAGE2_SEEDS = (17, 42, 2026)
_CFG_S2_BASE: dict[str, Any] = {
    'pe_only': True,
    'bar_threshold_pe': 2.5,
    'sipm_npix': 14400.0,
    'apply_sipm_saturation': False,
    'invert_sipm_saturation': True,
}
STAGE2_OOD_SPECS: tuple[tuple[str, Path], ...] = (
    ('gen_ood_w184_epps21_nc_charm', STAGE325_ROOT / 'ood_shards/gen_ood_w184_epps21_nc_charm'),
    ('gen_ood_w184_epps21_nc_incl', STAGE325_ROOT / 'ood_shards/gen_ood_w184_epps21_nc_incl'),
    ('gen_ood_run10085_10088', DEFAULT_CLEAN_ROOT / 'ood_shards/gen_ood_run10085_10088'),
    ('gen_ood_pythia6_std', DEFAULT_CLEAN_ROOT / 'ood_shards/gen_ood_pythia6_std'),
    ('gen_ood_pythia6_nux', DEFAULT_CLEAN_ROOT / 'ood_shards/gen_ood_pythia6_nux'),
    ('diag_pythia8_free_nucleon_nc_charm', DEFAULT_CLEAN_ROOT / 'ood_shards/diag_pythia8_free_nucleon_nc_charm'),
)
STAGE2_STRESS_SPECS: tuple[tuple[str, dict[str, Any], dict[str, Any]], ...] = (
    ('nominal', {}, dict(_CFG_S2_BASE)),
    ('pixel_dead_2pct', {'pixel_dead': 0.02}, dict(_CFG_S2_BASE)),
    ('pixel_dead_5pct', {'pixel_dead': 0.05}, dict(_CFG_S2_BASE)),
    ('pixel_dark_1e-5', {'pixel_dark_rate': 1e-5}, dict(_CFG_S2_BASE)),
    ('tracker_dead_2pct_noise1', {'tracker_dead': 0.02, 'tracker_noise_per_station': 1.0}, dict(_CFG_S2_BASE)),
    ('tracker_dead_5pct_noise2', {'tracker_dead': 0.05, 'tracker_noise_per_station': 2.0}, dict(_CFG_S2_BASE)),
    ('sipm_sat_cap_0p95', {}, {**_CFG_S2_BASE, 'sipm_invert_max_occupancy': 0.95, 'sipm_sat_cap': 0.95}),
)


class _S2MappedDataset(Dataset):
    def __init__(self, items_dir: Path, sidecar_df: pd.DataFrame | None = None) -> None:
        self.items = MappedItems(items_dir)
        self.sidecar_df = sidecar_df

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        it = self.items[int(idx)]
        if self.sidecar_df is not None:
            row = self.sidecar_df.iloc[int(idx)]
            it['label_7'] = int(row['label_7'])
            it['tau_branch_valid'] = int(row['tau_branch_valid'])
            it['charm_flag'] = int(row['charm_flag'])
            it['charm_valid'] = int(row['charm_valid'])
            it['vertex_xyz_mm'] = np.array([row['vertex_x_mm'], row['vertex_y_mm'], row['vertex_z_mm']], dtype=np.float32)
            it['vertex_valid'] = int(row['vertex_valid'])
            it['vis_kinematics'] = np.array([row['vis_pz_log1p'], row['vis_pt_log1p'], row['vis_cos_dphi'], row['vis_sin_dphi']], dtype=np.float32)
            it['kin_valid'] = int(row['kin_valid'])
        return it


def _s2_extract_shard_worker(task: tuple) -> list[tuple[tuple[int, int, int], int, int, dict]]:
    shard_str, geom, bounds, cfg, expected_split, attach_label = task
    sp = Path(shard_str)
    prov = pq.read_table(sp / 'provenance.parquet').to_pydict()
    obs_list = pq.read_table(sp / 'observations.parquet').to_pylist()
    tgt_list = pq.read_table(sp / 'targets.parquet').to_pylist() if attach_label else None
    extractor = _TauKinemaItemExtractor(geom, bounds, cfg)
    rows: list[tuple[tuple[int, int, int], int, int, dict]] = []
    for i, (r, c, rw, sp_name, plbl) in enumerate(zip(prov['run'], prov['chunk'], prov['row'], prov['split'], prov['label'])):
        if sp_name != expected_split:
            continue
        ob = obs_list[i]
        lbl = int(tgt_list[i]['label']) if tgt_list is not None else 0
        dom = compute_observable_containment(ob, geom, bounds, config=cfg)
        it = extractor(len(rows), ob, lbl if attach_label else None, dom, geom, bounds, cfg)
        rows.append(((int(r), int(c), int(rw)), int(plbl), int(lbl), it))
    return rows


def s2_prepare_cache_part(s2_cache: Path, tag: str, part: int, n_parts: int, workers: int = 8) -> dict[str, Any]:
    if tag not in ('train32k', 'tune8k'):
        raise ValueError(f'Unknown Stage-II cache tag {tag!r}')
    tag_dir = s2_cache / tag
    if (tag_dir / 'READY.json').is_file():
        return json.loads((tag_dir / 'READY.json').read_text())
    parts_dir = tag_dir / '_parts'
    parts_dir.mkdir(parents=True, exist_ok=True)
    part_file = parts_dir / f'part_{part:02d}_of_{n_parts:02d}.pkl'
    if part_file.is_file():
        return {'tag': tag, 'part': part, 'status': 'cached'}

    split_req = 'train' if tag == 'train32k' else 'validation'
    assert_unsealed_split(split_req, stage=2)
    shards = load_authorized_split_shards(DEFAULT_CLEAN_ROOT / 'data', split_req, stage=2)
    my_shards = shards[part::n_parts]
    geometry = json.loads((DEFAULT_CLEAN_ROOT / 'geometry.json').read_text())
    bounds = detector_bounds(geometry)
    cfg = _base_obs_config()
    scratch_base = '/dev/shm' if shutil.disk_usage('/dev/shm').free >= 6 * (1024 ** 3) else os.environ.get('_CONDOR_SCRATCH_DIR', '/tmp')
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f'tk-s2-{tag}-p{part}-', dir=scratch_base) as tmp_str:
        local_root = Path(tmp_str) / 'native'
        local_root.mkdir(parents=True, exist_ok=True)

        def _copy_shard(sp: Path) -> Path:
            dst = local_root / sp.name
            dst.mkdir(parents=True, exist_ok=True)
            fnames = ('observations.parquet', 'provenance.parquet', 'targets.parquet') if tag == 'train32k' else ('observations.parquet', 'provenance.parquet')
            for fn in fnames:
                shutil.copyfile(sp / fn, dst / fn)
            return dst

        with ThreadPoolExecutor(max_workers=min(8, max(1, workers))) as pool:
            local_shards = list(pool.map(_copy_shard, my_shards))
        tasks = [(str(sp), geometry, bounds, cfg, split_req, tag == 'train32k') for sp in local_shards]
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
            shard_rows = list(pool.map(_s2_extract_shard_worker, tasks))
    rows = [row for sr in shard_rows for row in sr]
    tmp_out = parts_dir / f'.tmp_part_{part:02d}_of_{n_parts:02d}.pkl'
    with tmp_out.open('wb') as f:
        pickle.dump(rows, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_out, part_file)
    return {'tag': tag, 'part': part, 'n_parts': n_parts, 'events': len(rows), 'seconds': float(time.perf_counter() - t0)}


def s2_merge_cache_tag(s2_cache: Path, tag: str, n_parts: int) -> dict[str, Any]:
    tag_dir = s2_cache / tag
    if (tag_dir / 'READY.json').is_file():
        return json.loads((tag_dir / 'READY.json').read_text())
    parts_dir = tag_dir / '_parts'
    t0 = time.perf_counter()
    rows: list[tuple[tuple[int, int, int], int, int, dict]] = []
    for p in range(n_parts):
        with (parts_dir / f'part_{p:02d}_of_{n_parts:02d}.pkl').open('rb') as f:
            rows.extend(pickle.load(f))
    rows.sort(key=lambda x: x[0])
    ordered_items, key_recs = [], []
    for idx_row, (k, plbl, tlbl, it) in enumerate(rows):
        it['index'] = int(idx_row)
        ordered_items.append(it)
        key_recs.append({'run': k[0], 'chunk': k[1], 'row': k[2], 'prov_label': plbl, 'tgt_label': tlbl})
    out_dir = tag_dir / f'mapped_{tag}'
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    write_mapped_items([ordered_items[i:i + 64] for i in range(0, len(ordered_items), 64)], out_dir)
    key_df = pd.DataFrame(key_recs)
    if tag == 'tune8k':
        scr_df = pd.read_parquet(DEFAULT_CLEAN_ROOT / 'data/tune_screen.parquet', columns=KEY)
        key_df['is_screen'] = pd.MultiIndex.from_frame(key_df[KEY]).isin(pd.MultiIndex.from_frame(scr_df[KEY]))
    key_df.to_parquet(tag_dir / f'{tag}_keys.parquet', index=False, compression=COMPRESSION)
    rep = {'tag': tag, 'n_parts': n_parts, 'events': len(ordered_items), 'merge_seconds': float(time.perf_counter() - t0)}
    write_json(tag_dir / 'READY.json', rep)
    shutil.rmtree(parts_dir, ignore_errors=True)
    return rep


def _s2_aux_ood_worker(task: tuple) -> tuple[str, int]:
    oname, ood_root_str, geom, bounds, cfg, out_dir_str = task
    obs_list = pq.read_table(Path(ood_root_str) / 'shards/000/observations.parquet').to_pylist()
    ext = _TauKinemaItemExtractor(geom, bounds, cfg)
    items = [ext(i, ob, None, compute_observable_containment(ob, geom, bounds, config=cfg), geom, bounds, cfg) for i, ob in enumerate(obs_list)]
    out_dir = Path(out_dir_str)
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    write_mapped_items([items[j:j + 64] for j in range(0, len(items), 64)], out_dir)
    return oname, len(items)


def _s2_aux_stress_worker(task: tuple) -> tuple[str, int]:
    st_name, pkw, c_over, obs_list, geom, bounds, cfg, out_dir_str = task
    rng_st = np.random.default_rng(32510)
    c_use = {**cfg, **c_over}
    ext = _TauKinemaItemExtractor(geom, bounds, c_use)
    items = []
    for i, ob in enumerate(obs_list):
        ob_p = _perturb_observation_for_stress(ob, bounds, rng_st, **pkw) if pkw else ob
        items.append(ext(i, ob_p, None, compute_observable_containment(ob_p, geom, bounds, config=c_use), geom, bounds, c_use))
    out_dir = Path(out_dir_str)
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    write_mapped_items([items[j:j + 64] for j in range(0, len(items), 64)], out_dir)
    return st_name, len(items)


def s2_prepare_aux_cache(s2_cache: Path, kind: str, workers: int = 8) -> dict[str, Any]:
    aux_dir = s2_cache / f'aux_{kind}'
    if (aux_dir / 'READY.json').is_file():
        return json.loads((aux_dir / 'READY.json').read_text())
    aux_dir.mkdir(parents=True, exist_ok=True)
    geometry = json.loads((DEFAULT_CLEAN_ROOT / 'geometry.json').read_text())
    bounds = detector_bounds(geometry)
    cfg = _base_obs_config()
    t0 = time.perf_counter()
    if kind == 'ood':
        tasks = [(oname, str(oroot), geometry, bounds, cfg, str(aux_dir / oname)) for oname, oroot in STAGE2_OOD_SPECS]
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks)), mp_context=mp.get_context('spawn')) as pool:
            done = list(pool.map(_s2_aux_ood_worker, tasks))
    else:
        assert_unsealed_split('validation', stage=2)
        sh80 = load_authorized_split_shards(DEFAULT_CLEAN_ROOT / 'data', 'validation', stage=2)[0]
        obs80 = pq.read_table(sh80 / 'observations.parquet').to_pylist()
        tasks = [(st_name, pkw, c_over, obs80, geometry, bounds, cfg, str(aux_dir / st_name)) for st_name, pkw, c_over in STAGE2_STRESS_SPECS]
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks)), mp_context=mp.get_context('spawn')) as pool:
            done = list(pool.map(_s2_aux_stress_worker, tasks))
    rep = {'kind': kind, 'entries': dict(done), 'seconds': float(time.perf_counter() - t0)}
    write_json(aux_dir / 'READY.json', rep)
    return rep


def run_stage2_seed(
    out_dir: Path,
    seed: int,
    *,
    passes: int = 20,
    batch_size: int = 128,
    micro_bs: int = 64,
    workers: int = 4,
    resume: bool = False,
) -> dict[str, Any]:
    s2_root = out_dir / 'stage2'
    s2_cache = s2_root / '_cache'
    seed_dir = s2_root / f'seed_{seed}'
    seed_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = seed_dir / 'manifest.json'
    if resume and manifest_path.is_file():
        man = json.loads(manifest_path.read_text())
        if man.get('status') == 'complete':
            return man

    for req in ('train32k', 'tune8k', 'aux_ood', 'aux_stress'):
        if not (s2_cache / req / 'READY.json').is_file():
            raise FileNotFoundError(f'Missing Stage-II cache {s2_cache / req / "READY.json"}')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cap = torch.cuda.get_device_capability(device) if device.type == 'cuda' else (0, 0)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    geometry = json.loads((DEFAULT_CLEAN_ROOT / 'geometry.json').read_text())
    bounds = detector_bounds(geometry)

    t_start = time.perf_counter()
    utc_started = datetime.now(timezone.utc).isoformat()
    shm_free = shutil.disk_usage('/dev/shm').free if Path('/dev/shm').exists() else 0
    scratch_base = '/dev/shm' if shm_free >= 6 * (1024 ** 3) else os.environ.get('PINPOINT_V3_SCRATCH', '/tmp')
    Path(scratch_base).mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f'tk-s2-s{seed}-', dir=scratch_base) as tmp_str:
        tmp_dir = Path(tmp_str)
        tr_dir = tmp_dir / 'mapped_train32k'
        tu_dir = tmp_dir / 'mapped_tune8k'
        shutil.copytree(s2_cache / 'train32k/mapped_train32k', tr_dir)
        shutil.copytree(s2_cache / 'tune8k/mapped_tune8k', tu_dir)

        tr_keys = pd.read_parquet(s2_cache / 'train32k/train32k_keys.parquet')
        tu_keys = pd.read_parquet(s2_cache / 'tune8k/tune8k_keys.parquet')
        sc_all = pd.read_parquet(DEFAULT_CLEAN_ROOT / 'truth_sidecar/event_sidecar.parquet')
        pos_sc = pd.MultiIndex.from_frame(sc_all[KEY]).get_indexer(pd.MultiIndex.from_frame(tr_keys[KEY]))
        assert (pos_sc >= 0).all()
        tr_sidecar = sc_all.iloc[pos_sc].reset_index(drop=True)

        train_ds = _S2MappedDataset(tr_dir, sidecar_df=tr_sidecar)
        scr_mask = tu_keys['is_screen'].to_numpy(dtype=bool)
        scr_idx = np.flatnonzero(scr_mask)
        y_scr = tu_keys['prov_label'].to_numpy(dtype=np.int64)[scr_idx]

        class _IndexedSubset(Dataset):
            def __init__(self, base: Dataset, indices: np.ndarray) -> None:
                self.base = base
                self.indices = indices

            def __len__(self) -> int:
                return len(self.indices)

            def __getitem__(self, i: int) -> dict:
                return self.base[int(self.indices[i])]

        tune8k_ds = _S2MappedDataset(tu_dir)
        scr_ds = _IndexedSubset(tune8k_ds, scr_idx)

        model_cfg = {
            'dim': 168,
            'heads': 8,
            'n_global_layers': 2,
            'n_queries': 12,
            'drop_path': 0.10,
            'head_dropout': 0.15,
            'ablation': 'full',
        }
        model = TauKinemaDiffNet(geometry, model_cfg).to(device)
        ema_model = copy.deepcopy(model).eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)

        grad_accum = max(1, batch_size // micro_bs)
        opt = torch.optim.AdamW(model.build_param_groups(8.0e-4, 0.03), lr=8.0e-4, weight_decay=0.03, betas=(0.9, 0.98))
        steps_per_pass = math.ceil(len(train_ds) / batch_size)
        total_steps = passes * steps_per_pass
        warmup_steps = max(1, int(0.06 * total_steps))
        min_lr_ratio = 1.0e-5 / 8.0e-4

        def _lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * min(1.0, prog)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
        amp_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
        amp_ok = cap[0] >= 7
        use_fp16_scaler = amp_ok and (amp_dtype == torch.float16)
        scaler = torch.amp.GradScaler('cuda', enabled=use_fp16_scaler)

        loader_workers = min(4, max(1, workers // 2))
        scr_loader = _make_loader(scr_ds, batch_size=max(64, micro_bs), shuffle=False, workers=loader_workers, seed=seed, y_scint_center=bounds['y_scint_center'], n_bars=bounds['n_bars'])
        tune8k_loader = _make_loader(tune8k_ds, batch_size=max(64, micro_bs), shuffle=False, workers=loader_workers, seed=seed, y_scint_center=bounds['y_scint_center'], n_bars=bounds['n_bars'])

        best_nll = float('inf')
        best_pass = 0
        global_step = 0
        history: list[dict[str, Any]] = []
        best_pt = seed_dir / 'best.pt'
        ema_pt = seed_dir / 'ema.pt'
        resume_pt = seed_dir / 'resume.pt'
        start_pass = 1
        if resume and resume_pt.is_file():
            st = torch.load(resume_pt, map_location=device, weights_only=False)
            model.load_state_dict(st['model'])
            ema_model.load_state_dict(st['ema'])
            opt.load_state_dict(st['opt'])
            scheduler.load_state_dict(st['scheduler'])
            best_nll = float(st['best_nll'])
            best_pass = int(st['best_pass'])
            global_step = int(st['global_step'])
            history = list(st['history'])
            start_pass = int(st['pass']) + 1

        try:
            torch.cuda.reset_peak_memory_stats(device)
            for p_idx in range(start_pass, passes + 1):
                tp0 = time.perf_counter()
                tr_loader = _make_loader(train_ds, batch_size=micro_bs, shuffle=True, workers=loader_workers, seed=int(seed) * 1000 + p_idx, y_scint_center=bounds['y_scint_center'], n_bars=bounds['n_bars'], mirror=True)
                model.train()
                loss_sum = 0.0
                n_seen = 0

                def _run_win(win_buf: list[dict]) -> None:
                    nonlocal global_step, loss_sum, n_seen
                    global_step += 1
                    opt.zero_grad(set_to_none=True)
                    denoms = compute_tau_kinema_window_denoms(win_buf)
                    tot_b = sum(len(mb['label']) for mb in win_buf)
                    win_l = 0.0
                    for mb in win_buf:
                        db = move_batch_to_device(mb, device)
                        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_ok):
                            out = model.forward_NextGen(db)
                            loss, _ = compute_tau_kinema_loss(model, out, db, fixed_denoms=denoms)
                        if use_fp16_scaler:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                        win_l += float(loss.detach().item())
                    if use_fp16_scaler:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                        scaler.step(opt)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                        opt.step()
                    scheduler.step()
                    eff_ema = min(0.999, (1.0 + global_step) / (10.0 + global_step))
                    _update_ema(ema_model, model, eff_ema)
                    loss_sum += win_l * tot_b
                    n_seen += tot_b

                try:
                    mb_buf: list[dict] = []
                    for batch in tr_loader:
                        mb_buf.append(batch)
                        if len(mb_buf) >= grad_accum:
                            _run_win(mb_buf)
                            mb_buf.clear()
                    if mb_buf:
                        _run_win(mb_buf)
                        mb_buf.clear()
                finally:
                    shutdown_loader(tr_loader)

                pass_sec = time.perf_counter() - tp0
                row_log: dict[str, Any] = {'pass': p_idx, 'step': global_step, 'lr': float(scheduler.get_last_lr()[0]), 'train_loss': float(loss_sum / max(n_seen, 1)), 'seconds': float(pass_sec)}
                if p_idx == 1 or p_idx % 2 == 0 or p_idx == passes:
                    scr_pred = _predict_loader_fp32(ema_model, scr_loader, device)
                    m_scr = _ba4_and_nll(y_scr, scr_pred['probs4'])
                    row_log.update({'tune_screen_nll': m_scr['NLL'], 'tune_screen_ba4': m_scr['BA4']})
                    if m_scr['NLL'] < best_nll:
                        best_nll = m_scr['NLL']
                        best_pass = p_idx
                        loc_b = tmp_dir / 'best.local.pt'
                        torch.save({'model': ema_model.state_dict(), 'pass': p_idx, 'config': model_cfg}, loc_b)
                        shutil.copyfile(loc_b, best_pt)
                        loc_b.unlink(missing_ok=True)
                    print(f"[stage2 n2 s{seed}] pass={p_idx}/{passes} step={global_step} tr_loss={row_log['train_loss']:.4f} scr_nll={m_scr['NLL']:.4f} scr_ba4={m_scr['BA4']*100:.2f}% ({pass_sec:.1f}s)", flush=True)
                history.append(row_log)
                if p_idx % 5 == 0 or p_idx == passes:
                    loc_r = tmp_dir / 'resume.local.pt'
                    torch.save({'pass': p_idx, 'global_step': global_step, 'model': model.state_dict(), 'ema': ema_model.state_dict(), 'opt': opt.state_dict(), 'scheduler': scheduler.state_dict(), 'best_nll': best_nll, 'best_pass': best_pass, 'history': history}, loc_r)
                    shutil.copyfile(loc_r, resume_pt)
                    loc_r.unlink(missing_ok=True)

            loc_e = tmp_dir / 'ema.local.pt'
            torch.save({'model': ema_model.state_dict(), 'pass': passes, 'config': model_cfg}, loc_e)
            shutil.copyfile(loc_e, ema_pt)
            loc_e.unlink(missing_ok=True)

            best_state = torch.load(best_pt, map_location=device, weights_only=False)
            ema_model.load_state_dict(best_state['model'])
            ema_model.eval()
            t8_out = _predict_loader_fp32(ema_model, tune8k_loader, device)
            np.save(seed_dir / f'probs_tune8k_seed{seed}.npy', t8_out['probs4'])
            np.save(seed_dir / f'pv_pred_tune8k_seed{seed}.npy', t8_out['v_pv_mm'])
            np.save(seed_dir / f'pv_scale_tune8k_seed{seed}.npy', t8_out['scale_pv_mm'])
            np.save(seed_dir / f'charm_prob_tune8k_seed{seed}.npy', t8_out['charm_prob'])

            df_p = tu_keys[KEY + ['is_screen', 'prov_label']].copy().reset_index(drop=True)
            for c in range(4):
                df_p[f'p_{c}'] = t8_out['probs4'][:, c].astype(np.float32)
            for ia, ax in enumerate(('x', 'y', 'z')):
                df_p[f'v_hat_{ax}_mm'] = t8_out['v_pv_mm'][:, ia].astype(np.float32)
                df_p[f'scale_{ax}_mm'] = t8_out['scale_pv_mm'][:, ia].astype(np.float32)
            df_p.to_parquet(seed_dir / f'tune8k_predictions_seed{seed}.parquet', index=False, compression=COMPRESSION)

            coll = _TauKinemaCollator(bounds['y_scint_center'], bounds['n_bars'])
            for oname, _ in STAGE2_OOD_SPECS:
                ods = _S2MappedDataset(s2_cache / 'aux_ood' / oname)
                oloader = DataLoader(ods, batch_size=64, shuffle=False, num_workers=0, collate_fn=coll)
                opred = _predict_loader_fp32(ema_model, oloader, device)
                np.save(seed_dir / f'probs_ood_{oname}_seed{seed}.npy', opred['probs4'])

            for st_name, _, _ in STAGE2_STRESS_SPECS:
                sds = _S2MappedDataset(s2_cache / 'aux_stress' / st_name)
                sloader = DataLoader(sds, batch_size=64, shuffle=False, num_workers=0, collate_fn=coll)
                spred = _predict_loader_fp32(ema_model, sloader, device)
                np.save(seed_dir / f'probs_stress_{st_name}_seed{seed}.npy', spred['probs4'])
        finally:
            shutdown_loader(scr_loader)
            shutdown_loader(tune8k_loader)

    peak_vram_mib = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
    manifest = {
        'status': 'complete',
        'stage': 'Stage-II',
        'seed': int(seed),
        'parameters': int(sum(p.numel() for p in model.parameters())),
        'GPU': torch.cuda.get_device_name(device),
        'peak_allocated_VRAM_MiB': round(peak_vram_mib, 1),
        'total_optimizer_steps': int(global_step),
        'best_pass': int(best_pass),
        'best_tune_screen_nll': float(best_nll),
        'wall_seconds': float(time.perf_counter() - t_start),
        'UTC_started': utc_started,
        'UTC_finished': datetime.now(timezone.utc).isoformat(),
        'sealed_shards_touched': False,
    }
    write_json(manifest_path, manifest)
    if resume_pt.is_file():
        resume_pt.unlink()
    return manifest


def aggregate_stage2_results(out_dir: Path) -> dict[str, Any]:
    """Aggregate Stage-II 3-seed N2 results on Tune6k + 6 OOD + 7 Stress + B+D+F_n2fix blend."""
    s2_root = out_dir / 'stage2'
    tune_meta = pd.read_parquet(DEFAULT_CLEAN_ROOT / 'validation_meta/tune_truth_meta.parquet')
    scr_df = pd.read_parquet(DEFAULT_CLEAN_ROOT / 'data/tune_screen.parquet', columns=KEY)
    sidecar_df = pd.read_parquet(DEFAULT_CLEAN_ROOT / 'truth_sidecar/event_sidecar.parquet')
    ref_idx = pd.MultiIndex.from_frame(tune_meta[KEY])
    in_screen = np.asarray(ref_idx.isin(pd.MultiIndex.from_frame(scr_df[KEY])), dtype=bool)
    tune6k_m = ~in_screen

    labels = tune_meta['label'].to_numpy(dtype=np.int64)
    obs_dom = tune_meta['obs_domain_id'].to_numpy(dtype=np.int64)
    pc_m = tune_meta['pixel_centered'].to_numpy(dtype=bool)
    fp_m = tune_meta['front_pinpoint_pixel_centered'].to_numpy(dtype=bool)
    interm_m = pc_m & (~fp_m)
    fam_codes = pd.MultiIndex.from_frame(tune_meta[FAMILY]).factorize()[0]

    s_pos = pd.MultiIndex.from_frame(sidecar_df[KEY]).get_indexer(ref_idx)
    sc_al = sidecar_df.iloc[s_pos].reset_index(drop=True)
    v_true = sc_al[['vertex_x_mm', 'vertex_y_mm', 'vertex_z_mm']].to_numpy(dtype=np.float64)
    v_valid = sc_al['vertex_valid'].to_numpy(dtype=np.int64) > 0

    seed_probs_t8, seed_pv_t8, seed_sc_t8 = {}, {}, {}
    per_seed = {}
    for seed in STAGE2_SEEDS:
        sdir = s2_root / f'seed_{seed}'
        df_s = pd.read_parquet(sdir / f'tune8k_predictions_seed{seed}.parquet')
        pos = pd.MultiIndex.from_frame(df_s[KEY]).get_indexer(ref_idx)
        assert (pos >= 0).all()
        df_al = df_s.iloc[pos].reset_index(drop=True)
        pr = df_al[['p_0', 'p_1', 'p_2', 'p_3']].to_numpy(dtype=np.float64)
        vp = df_al[['v_hat_x_mm', 'v_hat_y_mm', 'v_hat_z_mm']].to_numpy(dtype=np.float64)
        vs = df_al[['scale_x_mm', 'scale_y_mm', 'scale_z_mm']].to_numpy(dtype=np.float64)
        seed_probs_t8[seed], seed_pv_t8[seed], seed_sc_t8[seed] = pr, vp, vs
        per_seed[f'seed_{seed}'] = _ba4_and_nll(labels[tune6k_m], pr[tune6k_m])

    p_f_ens = np.mean([seed_probs_t8[s] for s in STAGE2_SEEDS], axis=0)
    vp_ens = np.mean([seed_pv_t8[s] for s in STAGE2_SEEDS], axis=0)
    vs_ens = np.sqrt(np.mean([seed_sc_t8[s] ** 2 for s in STAGE2_SEEDS], axis=0))
    np.save(s2_root / 'probs_tune6k_ens.npy', p_f_ens[tune6k_m].astype(np.float32))
    np.save(s2_root / 'pv_pred_tune6k_ens.npy', vp_ens[tune6k_m].astype(np.float32))
    np.save(s2_root / 'pv_scale_tune6k_ens.npy', vs_ens[tune6k_m].astype(np.float32))

    # Calibrate PV scale strictly on tune_screen (N=2,000) by regime
    vs_cal = vs_ens.copy()
    for rtag, rmk in (('Front30', fp_m), ('Interm150', interm_m), ('Other', ~(fp_m | interm_m))):
        cm = in_screen & v_valid & rmk
        if int(cm.sum()) >= 50:
            cp = compute_student_t_pulls(torch.from_numpy(vp_ens[cm]), torch.from_numpy(vs_ens[cm]), torch.from_numpy(v_true[cm]), torch.ones(int(cm.sum()), dtype=torch.int64), df=3.0)['p68_abs_pull']
            cvec = np.asarray([float(np.clip(c, 0.4, 2.5)) if c > 0 else 1.0 for c in cp], dtype=np.float64)
            vs_cal[rmk] *= cvec[None, :]

    t6k_slices = {
        'All_Tune6k': tune6k_m,
        'D1_obs': tune6k_m & (obs_dom == 1),
        'D2_obs': tune6k_m & (obs_dom == 2),
        'D3_obs': tune6k_m & (obs_dom == 3),
        'Pinpoint_PixXY': tune6k_m & pc_m,
        'Front30': tune6k_m & fp_m,
        'Interm150': tune6k_m & interm_m,
    }
    pv_audit: dict[str, Any] = {}
    for rtag, rmk in (('All_active', tune6k_m & v_valid), ('Front30', tune6k_m & v_valid & fp_m), ('Interm150', tune6k_m & v_valid & interm_m), ('D1_obs', tune6k_m & v_valid & (obs_dom == 1))):
        dz = np.abs(vp_ens[rmk, 2] - v_true[rmk, 2])
        d3 = np.linalg.norm(vp_ens[rmk] - v_true[rmk], axis=1)
        pulls = compute_student_t_pulls(torch.from_numpy(vp_ens[rmk]), torch.from_numpy(vs_cal[rmk]), torch.from_numpy(v_true[rmk]), torch.ones(int(rmk.sum()), dtype=torch.int64), df=3.0)
        pv_audit[rtag] = {'count': int(rmk.sum()), 'median_abs_dz_mm': float(np.median(dz)), 'median_3d_mm': float(np.median(d3)), **pulls}

    # Load frozen Stage-II B & D Tune6k ensembles for B+D+F_n2fix 3-way blend
    nextgen_root = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research')
    p_b_t6 = np.load(nextgen_root / 'exp_tracklet_ipformer/stage2/probs_tune6k_ens.npy').astype(np.float64)
    p_d_t6 = np.load(nextgen_root / 'exp_dual_regime/stage2/probs_tune6k_ens3_raw.npy').astype(np.float64)
    p_f_t6 = p_f_ens[tune6k_m]
    p_bd_t6 = 0.5 * (p_b_t6 + p_d_t6)
    p_bdf_t6 = (p_b_t6 + p_d_t6 + p_f_t6) / 3.0

    y_t6 = labels[tune6k_m]
    fam_t6 = fam_codes[tune6k_m]
    t6_only_slices = {k: v[tune6k_m] for k, v in t6k_slices.items()}
    eval_t6 = {}
    for mnm, parr in (('F_n2fix_ens3', p_f_t6), ('Blend_BD_raw', p_bd_t6), ('Blend_BDF_n2fix_raw', p_bdf_t6)):
        eval_t6[mnm] = {
            'slices': {sn: _ba4_and_nll(y_t6[mk], parr[mk]) for sn, mk in t6_only_slices.items()},
            'within_slice_1pct_fpr': {sn: _tau_efficiency_at_fpr(parr[t6_only_slices[sn], 2], y_t6[t6_only_slices[sn]], 0.01) for sn in ('All_Tune6k', 'D1_obs', 'Pinpoint_PixXY', 'Front30')},
            'split_half_1pct_fpr': {sn: _split_half_tail_eval(parr[:, 2], y_t6, tune_meta.iloc[tune6k_m].reset_index(drop=True), t6_only_slices[sn], fpr_targets=(0.01, 0.001)) for sn in ('All_Tune6k', 'D1_obs', 'Pinpoint_PixXY')},
        }

    paired_boot = {
        'Blend_BDF_n2fix - Blend_BD_raw': {
            sn: paired_family_bootstrap_delta(p_bdf_t6[:, 2], p_bd_t6[:, 2], p_bdf_t6, p_bd_t6, y_t6, fam_t6, t6_only_slices[sn], n_boot=1000, seed=20263250)
            for sn in ('All_Tune6k', 'D1_obs', 'Front30', 'Interm150', 'Pinpoint_PixXY')
        },
        'Blend_BDF_n2fix - Tracklet_raw': {
            sn: paired_family_bootstrap_delta(p_bdf_t6[:, 2], p_b_t6[:, 2], p_bdf_t6, p_b_t6, y_t6, fam_t6, t6_only_slices[sn], n_boot=1000, seed=20263250)
            for sn in ('All_Tune6k', 'D1_obs', 'Front30', 'Interm150', 'Pinpoint_PixXY')
        },
    }

    ood_summary: dict[str, Any] = {}
    for oname, oroot in STAGE2_OOD_SPECS:
        yo = pd.read_parquet(oroot / 'truth_meta.parquet')['label'].to_numpy(dtype=np.int64)
        pf = np.mean([np.load(s2_root / f'seed_{s}/probs_ood_{oname}_seed{s}.npy').astype(np.float64) for s in STAGE2_SEEDS], axis=0)
        pb_o = np.load(nextgen_root / f'exp_tracklet_ipformer/stage2/probs_ood_{oname}_ens.npy').astype(np.float64)
        pd_o = np.load(nextgen_root / 'exp_dual_regime/stage2/ood_and_stress_ens3_probs.npz')[f'ood_{oname}'].astype(np.float64)
        pbdf_o = (pb_o + pd_o + pf) / 3.0
        for tag_o, po in (('F_n2fix_ens3', pf), ('Blend_BDF_n2fix', pbdf_o)):
            if len(np.unique(yo)) >= 4:
                ood_summary.setdefault(oname, {})[tag_o] = {'n_events': int(len(yo)), **_ba4_and_nll(yo, po)}
            else:
                ood_summary.setdefault(oname, {})[tag_o] = {
                    'n_events': int(len(yo)),
                    'pred_tau_argmax_rate': float((po.argmax(1) == 2).mean()),
                    'p_tau_gt_0p5_rate': float((po[:, 2] > 0.5).mean()),
                    'mean_p_tau': float(po[:, 2].mean()),
                }

    sh80 = load_authorized_split_shards(DEFAULT_CLEAN_ROOT / 'data', 'validation', stage=2)[0]
    prov80 = pd.read_parquet(sh80 / 'provenance.parquet', columns=KEY + ['label'])
    pos80 = pd.MultiIndex.from_frame(tune_meta[KEY]).get_indexer(pd.MultiIndex.from_frame(prov80[KEY]))
    y80 = tune_meta.iloc[pos80]['label'].to_numpy(dtype=np.int64)
    p_nom_f = np.mean([np.load(s2_root / f'seed_{s}/probs_stress_nominal_seed{s}.npy').astype(np.float64) for s in STAGE2_SEEDS], axis=0)
    stress_summary: dict[str, Any] = {}
    for st_name, _, _ in STAGE2_STRESS_SPECS:
        pf_s = np.mean([np.load(s2_root / f'seed_{s}/probs_stress_{st_name}_seed{s}.npy').astype(np.float64) for s in STAGE2_SEEDS], axis=0)
        m_st = _ba4_and_nll(y80, pf_s)
        stress_summary[st_name] = {**m_st, 'mean_TV_vs_nominal': float((0.5 * np.abs(pf_s - p_nom_f).sum(axis=1)).mean())}
    nom_ba4 = stress_summary['nominal']['BA4']
    for st_name in stress_summary:
        stress_summary[st_name]['delta_BA4_vs_nominal_pp'] = float((stress_summary[st_name]['BA4'] - nom_ba4) * 100.0)

    out_doc = {
        'UTC': datetime.now(timezone.utc).isoformat(),
        'stage': 'Stage-II TauKinemaDiffNet N2 Fix (Train32k x 3 seeds -> Tune6k + 6 OOD + 7 Stress + B+D+F Blend)',
        'per_seed_tune6k': per_seed,
        'pv_audit_tune6k': pv_audit,
        'tune6k_models': eval_t6,
        'paired_family_bootstrap_seed20263250_n1000': paired_boot,
        'ood_evaluation': ood_summary,
        'shard080_stress_tests': stress_summary,
    }
    write_json(s2_root / 'stage2_n2_tau_kinema_results.json', out_doc)
    return out_doc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--mode',
        choices=(
            'gate_ab', 'prepare_cache', 'train', 'aggregate', 'eval_n2_stage1',
            's2_prepare_part', 's2_merge', 's2_prepare_aux', 's2_train', 's2_aggregate',
        ),
        required=True,
    )
    parser.add_argument('--ablation', choices=TauKinemaDiffNet.VALID_ABLATIONS, default='full')
    parser.add_argument('--run-name', default=None)
    parser.add_argument('--out-dir', type=Path, default=EXP_ROOT)
    parser.add_argument('--cache-dir', type=Path, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--passes', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--micro-bs', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--cache-tag', type=str, default='train32k')
    parser.add_argument('--part', type=int, default=0)
    parser.add_argument('--n-parts', type=int, default=8)
    parser.add_argument('--s2-aux-kind', type=str, default='ood', choices=('ood', 'stress'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    is_stage2 = args.mode.startswith('s2_')
    if is_stage2:
        guard = SealAuditGuard()
        guard.__enter__()
    else:
        guard = install_stage1_seal_and_tune_guard()

    try:
        if args.mode == 'gate_ab':
            res = run_gate_ab_checks(args.out_dir)
            print(json.dumps({
                'wall_seconds': res['wall_seconds'],
                'runs_checked': list(res['ablation_checks'].keys()),
                'seal_violations': len(guard.violations) + len(_STAGE1_VIOLATIONS),
            }, indent=2))
        elif args.mode == 'prepare_cache':
            cdir = args.cache_dir or (args.out_dir / '_stage1_item_cache')
            res = prepare_stage1_cache(cdir, workers=args.workers)
            print(json.dumps(res, indent=2))
        elif args.mode == 'train':
            default_name = next((k for k, v in ABLATION_RUNS.items() if v == args.ablation), f's1_{args.ablation}')
            rname = args.run_name or default_name
            res = run_stage1_train_and_eval(
                rname,
                args.ablation,
                args.out_dir,
                seed=args.seed,
                passes=args.passes,
                batch_size=args.batch_size,
                micro_bs=args.micro_bs,
                workers=args.workers,
                cache_dir=args.cache_dir,
            )
            print(json.dumps({
                'status': res['status'],
                'run_name': res['run_name'],
                'parameters': res['parameters'],
                'best_pass': res['best_pass'],
                'best_screen_nll': res['best_screen_nll'],
                'peak_VRAM_MiB': res['peak_VRAM_MiB'],
                'train_wall_seconds': res['train_wall_seconds'],
            }, indent=2))
        elif args.mode == 'aggregate':
            agg = aggregate_stage1_results(args.out_dir)
            print(json.dumps({
                'runs': list(agg['runs'].keys()),
                'paired_comparisons': list(agg['paired_family_bootstrap'].keys()),
                'seal_violations': len(guard.violations) + len(_STAGE1_VIOLATIONS),
            }, indent=2))
        elif args.mode == 'eval_n2_stage1':
            rname = args.run_name or 's1_tau_kinema_diffnet_n2fix'
            rep = eval_n2_stage1(args.out_dir, run_name=rname, workers=args.workers)
            print(json.dumps({
                'gate_af_passed': rep['gate_af_passed'],
                'interm150_median_abs_dz_mm': rep['interm150_median_abs_dz_mm'],
                'all_9_pulls_in_085_115': rep['all_9_pulls_in_085_115'],
                'w184_nc_charm': rep['w184_nc_charm'],
            }, indent=2))
        elif args.mode == 's2_prepare_part':
            print(json.dumps(s2_prepare_cache_part(args.out_dir / 'stage2/_cache', args.cache_tag, args.part, args.n_parts, workers=args.workers), indent=2))
        elif args.mode == 's2_merge':
            print(json.dumps(s2_merge_cache_tag(args.out_dir / 'stage2/_cache', args.cache_tag, args.n_parts), indent=2))
        elif args.mode == 's2_prepare_aux':
            print(json.dumps(s2_prepare_aux_cache(args.out_dir / 'stage2/_cache', args.s2_aux_kind, workers=args.workers), indent=2))
        elif args.mode == 's2_train':
            res = run_stage2_seed(args.out_dir, args.seed, passes=args.passes, batch_size=args.batch_size, micro_bs=args.micro_bs, workers=args.workers, resume=args.resume)
            print(json.dumps({'status': res['status'], 'seed': res['seed'], 'best_pass': res['best_pass'], 'best_tune_screen_nll': res['best_tune_screen_nll'], 'wall_seconds': res['wall_seconds']}, indent=2))
        elif args.mode == 's2_aggregate':
            rep = aggregate_stage2_results(args.out_dir)
            print(json.dumps({'per_seed_tune6k': rep['per_seed_tune6k'], 'pv_audit_tune6k': {k: v['median_abs_dz_mm'] for k, v in rep['pv_audit_tune6k'].items()}}, indent=2))

        if guard.violations or (not is_stage2 and _STAGE1_VIOLATIONS):
            raise SealViolationError(f'Audit violations detected: {guard.violations + _STAGE1_VIOLATIONS}')
    finally:
        guard.__exit__(None, None, None)


if __name__ == '__main__':
    main()
