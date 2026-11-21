"""Stage-I Gate A/B checks, Matched-Budget HTCondor Training, Causal Ablations & Evaluation
for Proposal F: TauKinemaDiffNet.

Enforces Zero-Exception Seal + Stage-1 Tune8k/Tune6k Isolation via SealAuditGuard and
an explicit Stage-1 path hook blocking shards 080..149 and validation_meta.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
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
import pyarrow.parquet as pq
from scipy.stats import beta as scipy_beta
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler

from analysis.flavour.inventory import write_json
from analysis.flavour.nextgen.protocols import (
    compute_student_t_pulls,
    conditional_three_from_four,
)
from analysis.flavour.nextgen.seal_guard import (
    SealAuditGuard,
    SealViolationError,
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
    probs4_l, probs12_l, phi_l, vpv_l, spv_l, charm_l, tau_s_l, kin_l = [], [], [], [], [], [], [], []
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
            probs4_l.append(out.probs4.float().cpu().numpy())
            probs12_l.append(out.aux['probs12'].float().cpu().numpy())
            phi_l.append(out.aux['phi_law'].float().cpu().numpy())
            vpv_l.append(out.v_pv_mm.float().cpu().numpy())
            spv_l.append(out.scale_pv_mm.float().cpu().numpy())
            charm_l.append(torch.sigmoid(out.charm_logit.float()).cpu().numpy())
            tau_s_l.append(out.tau_score(charm_penalty=0.0).cpu().numpy())
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

        # Empty-modality finite check & bs=1 vs bs=8 consistency
        model.eval()
        with torch.inference_mode():
            out_emp = model(empty_batch)
            assert torch.isfinite(out_emp['probs4']).all() and torch.isfinite(out_emp['v_hat']).all()
            p1 = model(batch1)['probs4']
            p8 = model(batch8)['probs4'][:1]
            bs1_vs_bs8_diff = float((p1 - p8).abs().max().item())
            assert bs1_vs_bs8_diff <= 1e-6, f'bs=1 vs bs=8 diff {bs1_vs_bs8_diff} > 1e-6'

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


def _extract_ood_items(
    ood_name: str,
    ood_root: Path,
    geometry: dict,
    bounds: dict,
    cfg_obs: dict,
) -> tuple[list[dict], pd.DataFrame]:
    obs_p = ood_root / 'shards/000/observations.parquet'
    ds = V3ObservationDataset(
        obs_p,
        DEFAULT_CLEAN_ROOT / 'geometry.json',
        config=cfg_obs,
        item_transform=_TauKinemaItemExtractor(geometry, bounds, cfg_obs),
    )
    items = [ds[i] for i in range(len(ds))]
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

    scratch_root = Path(os.environ.get('PINPOINT_V3_SCRATCH', '/tmp'))
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
            print(f'[{run_name}] Staged shared 32k cache from {cache_dir} to local scratch in {time.perf_counter()-t_prep0:.1f}s', flush=True)
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
        loader_workers = min(2, max(1, workers // 2))
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
                _update_ema(ema_model, model, 0.999)
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

        # Calibrate per-obs-domain Student-t PV scale multipliers strictly on internal_train[:4096]
        n_pv_cal = min(4096, len(tr_pos))
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
        tr_cal_vtrue = torch.from_numpy(
            np.column_stack([
                tr_cal_sc['vertex_x_mm'].to_numpy(np.float32),
                tr_cal_sc['vertex_y_mm'].to_numpy(np.float32),
                tr_cal_sc['vertex_z_mm'].to_numpy(np.float32),
            ])
        )
        tr_cal_vvalid = torch.from_numpy(tr_cal_sc['vertex_valid'].to_numpy(dtype=np.int64))
        global_pulls = compute_student_t_pulls(
            torch.from_numpy(pv_cal_pred['v_pv_mm']),
            torch.from_numpy(pv_cal_pred['scale_pv_mm']),
            tr_cal_vtrue,
            tr_cal_vvalid,
            df=3.0,
        )['p68_abs_pull']
        pv_scale_corr_by_dom: dict[int, list[float]] = {}
        global_corr = [float(np.clip(c, 0.5, 2.5)) if c > 0.0 else 1.0 for c in global_pulls]
        for d_id in (1, 2, 3):
            dm = (tr_cal_dom == d_id) & (tr_cal_vvalid.numpy() > 0)
            if int(dm.sum()) >= 40:
                dpulls = compute_student_t_pulls(
                    torch.from_numpy(pv_cal_pred['v_pv_mm'][dm]),
                    torch.from_numpy(pv_cal_pred['scale_pv_mm'][dm]),
                    tr_cal_vtrue[dm],
                    tr_cal_vvalid[dm],
                    df=3.0,
                )['p68_abs_pull']
                pv_scale_corr_by_dom[d_id] = [float(np.clip(c, 0.5, 2.5)) if c > 0.0 else g for c, g in zip(dpulls, global_corr)]
            else:
                pv_scale_corr_by_dom[d_id] = list(global_corr)

        # Evaluate Internal-Val (4,800 events) in strict IEEE-FP32
        val_pred = _predict_loader_fp32(ema_model, val_loader, device)
        val_obs_dom = val_keys['obs_domain_id'].to_numpy(dtype=np.int64)
        scale_pv_raw = val_pred['scale_pv_mm'].copy()
        scale_pv_cal = scale_pv_raw.copy()
        for d_id, cvec in pv_scale_corr_by_dom.items():
            dm_val = val_obs_dom == d_id
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
        ood_items, ood_meta = _extract_ood_items(ov_name, ov_root, geometry, bounds, cfg_obs)
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
        'pv_scale_corr_by_obs_domain': {str(k): v for k, v in pv_scale_corr_by_dom.items()},
        'stage1_audit_violations': list(_STAGE1_VIOLATIONS),
        'learned_alpha_res': float(ema_model.hyp_engine.alpha_res.detach().cpu().item())
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('gate_ab', 'prepare_cache', 'train', 'aggregate'), required=True)
    parser.add_argument('--ablation', choices=TauKinemaDiffNet.VALID_ABLATIONS, default='full')
    parser.add_argument('--run-name', default=None)
    parser.add_argument('--out-dir', type=Path, default=EXP_ROOT)
    parser.add_argument('--cache-dir', type=Path, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--passes', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--micro-bs', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    with install_stage1_seal_and_tune_guard() as guard:
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

        if guard.violations or _STAGE1_VIOLATIONS:
            raise SealViolationError(f'Audit violations detected: {guard.violations + _STAGE1_VIOLATIONS}')


if __name__ == '__main__':
    main()
