"""Stage-I & Stage-II runner for Frozen-Backbone Score-Bank DRO Partial-AUC Tail Fine-Tuning.

Enforces zero-exception seal & Stage-I anti-leakage guards (`SealAuditGuard`),
multi-threaded CPU shard extraction to `/dev/shm`, single-pass deterministic IEEE-FP32
penultimate latent caching (`max|logits_cached - logits_full_model| <= 1e-7`),
Stage-I ablation & `internal_dev_screen`-only hyperparameter locking, and Stage-II
3-seed frozen confirmation on `Tune6k` + 6 OOD + 7 Shard-080 Stress with 1,000x paired
family block bootstrap (`seed=20263250`).
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

from analysis.flavour.inventory import write_json
from analysis.flavour.nextgen.dual_regime_modalnet import (
    DualRegimeModalNet,
    collate_dual_regime,
    extract_dual_regime_item,
)
from analysis.flavour.nextgen.seal_guard import (
    SealAuditGuard,
    SealViolationError,
    assert_unsealed_path,
    assert_unsealed_split,
    load_authorized_split_shards,
    verify_inference_truth_invariance,
)
from analysis.flavour.nextgen.tail_dro_pauc import (
    TailDROConfig,
    compute_multiclass_ece,
    compute_normalized_pauc,
    compute_tau_binary_ece,
    extract_penultimate_cache,
    finetune_head_on_cached_latents,
    freeze_backbone_for_tail_finetuning,
    predict_from_cached_latents,
    restore_tf32_mode,
    set_ieee_fp32_mode,
    snapshot_backbone_weights,
    verify_backbone_bit_identity,
)
from analysis.flavour.nextgen.tracklet_ipformer import (
    TrackletIPFormer,
    collate_tracklet_pf,
    extract_tracklet_pf_item,
)
from analysis.flavour.v2.partitions import FAMILY, KEY
from analysis.flavour.v3.common.calibration import _tau_efficiency_at_fpr
from analysis.flavour.v3.common.domains import (
    compute_observable_containment,
    cumulative_radiation_length,
    detector_bounds,
)
from analysis.flavour.v3.v4_tau.ensemble import _apply_domain_convex_blend, _fit_domain_convex_blend
from analysis.flavour.v3.v6_clean_sim.run_stage325_remediation import _perturb_observation_for_stress


CLEAN_V3_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v6_clean_v3_2026-11-19/clean_sim')
STAGE325_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v7_stage325_2026-11-20')
NEXTGEN_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research')
SPLITS_DIR = NEXTGEN_ROOT / 'forensics/splits'
OUT_ROOT = NEXTGEN_ROOT / 'exp_tail_dro'

STAGE2_SEEDS = (17, 42, 2026)

BASE_OBS_CFG: dict[str, Any] = {
    'pe_only': True,
    'bar_threshold_pe': 2.5,
    'sipm_npix': 14400.0,
    'apply_sipm_saturation': False,
    'invert_sipm_saturation': True,
}

OOD_SPECS: tuple[tuple[str, Path], ...] = (
    ('gen_ood_run10085_10088', CLEAN_V3_ROOT / 'ood_shards/gen_ood_run10085_10088'),
    ('gen_ood_pythia6_nux', CLEAN_V3_ROOT / 'ood_shards/gen_ood_pythia6_nux'),
    ('gen_ood_pythia6_std', CLEAN_V3_ROOT / 'ood_shards/gen_ood_pythia6_std'),
    ('gen_ood_w184_epps21_nc_charm', STAGE325_ROOT / 'ood_shards/gen_ood_w184_epps21_nc_charm'),
    ('gen_ood_w184_epps21_nc_incl', STAGE325_ROOT / 'ood_shards/gen_ood_w184_epps21_nc_incl'),
    ('diag_pythia8_free_nucleon_nc_charm', CLEAN_V3_ROOT / 'ood_shards/diag_pythia8_free_nucleon_nc_charm'),
)

STRESS_SPECS: tuple[tuple[str, dict[str, Any], dict[str, Any]], ...] = (
    ('nominal', {}, {}),
    ('pixel_dead_2pct', {'pixel_dead': 0.02}, {}),
    ('pixel_dead_5pct', {'pixel_dead': 0.05}, {}),
    ('pixel_dark_1e-5', {'pixel_dark_rate': 1e-5}, {}),
    ('tracker_dead_2pct_noise1', {'tracker_dead': 0.02, 'tracker_noise_per_station': 1.0}, {}),
    ('tracker_dead_5pct_noise2', {'tracker_dead': 0.05, 'tracker_noise_per_station': 2.0}, {}),
    ('sipm_sat_cap_0p95', {}, {'sipm_invert_max_occupancy': 0.95, 'sipm_sat_cap': 0.95}),
)

STAGE1_FORBIDDEN_RE = re.compile(
    r'(?:^|/)shards/0(?:8\d|9\d)(?:/|$)|validation_meta|tune_screen\.parquet|tune_truth_meta\.parquet',
    re.IGNORECASE,
)
_STAGE1_HOOK_INSTALLED = False


def install_stage1_seal_guard() -> SealAuditGuard:
    """Install both SealAuditGuard (100..149/sealed) and Stage-I Tune8k/080..099 guard."""
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
                ps = os.fspath(target) if hasattr(target, '__fspath__') else str(target)
            except Exception:
                return
            if isinstance(ps, bytes):
                ps = ps.decode('utf-8', errors='ignore')
            norm = ps.replace('\\', '/')
            if STAGE1_FORBIDDEN_RE.search(norm):
                raise SealViolationError(f'Stage-I forbids accessing Tune8k/validation path: {norm}')

        sys.addaudithook(_stage1_hook)
        _STAGE1_HOOK_INSTALLED = True
    return guard


def _configure_runtime() -> None:
    torch.set_num_threads(1)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)


def _shm_scratch_root() -> Path:
    if Path('/dev/shm').is_dir() and os.access('/dev/shm', os.W_OK):
        return Path('/dev/shm')
    return Path(os.environ.get('_CONDOR_SCRATCH_DIR', tempfile.gettempdir()))


# ---------------------------------------------------------------------------
# Checkpoint Loading (TrackletIPFormer, DualRegimeModalNet, PinpointUnifiedNet)
# ---------------------------------------------------------------------------
def _load_model_checkpoint(
    family: str,
    ckpt_path: Path,
    geometry: dict,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    ckpt = torch.load(assert_unsealed_path(ckpt_path), map_location='cpu', weights_only=False)
    cfg = dict(ckpt.get('config', {}))
    if family == 'tracklet_ipformer':
        model = TrackletIPFormer(geometry, cfg)
    elif family == 'dual_regime_modalnet':
        model_cfg = {
            'dim': 160, 'heads': 8, 'drop_path': 0.10, 'head_dropout': 0.15,
            'use_ipt_spec_interlock': True, 'use_sinkhorn_tof': True,
            'enable_pixdark_noise_aug': True, 'homogeneous_regime': False,
            **cfg,
        }
        model = DualRegimeModalNet(geometry, model_cfg)
        cfg = model_cfg
    elif family == 'unified_pinpoint_net':
        from analysis.flavour.nextgen.unified_pinpoint_net import PinpointUnifiedNet  # type: ignore
        model = PinpointUnifiedNet(geometry, cfg)
    else:
        raise ValueError(f'Unsupported family {family!r}')
    model.load_state_dict(ckpt['model'], strict=True)
    model = model.to(device)
    model.eval()
    return model, cfg


def _get_item_and_collate_fns(family: str, bounds: dict, cfg: dict[str, Any]):
    yc, nb = float(bounds['y_scint_center']), int(bounds['n_bars'])
    if family == 'tracklet_ipformer':
        item_fn = extract_tracklet_pf_item
        collate_fn = lambda evs: collate_tracklet_pf(evs, mirror_rng=None, y_scint_center=yc, n_bars=nb)
    elif family == 'dual_regime_modalnet':
        item_fn = extract_dual_regime_item
        collate_fn = lambda evs: collate_dual_regime(evs, mirror_rng=None, y_scint_center=yc, n_bars=nb)
    elif family == 'unified_pinpoint_net':
        from analysis.flavour.nextgen.unified_pinpoint_net import (  # type: ignore
            collate_unified_pinpoint,
            extract_unified_pinpoint_item,
        )
        item_fn = extract_unified_pinpoint_item
        collate_fn = lambda evs: collate_unified_pinpoint(evs, mirror_rng=None, y_scint_center=yc, n_bars=nb)
    else:
        raise ValueError(f'Unsupported family {family!r}')
    return item_fn, collate_fn


# ---------------------------------------------------------------------------
# Parallel Shard Extraction Helpers
# ---------------------------------------------------------------------------
def _extract_shard_worker(task: tuple) -> list[tuple[tuple[int, int, int], int, dict[str, dict]]]:
    shard_str, expected_split, attach_label, geom, bounds, cfg, families = task
    _configure_runtime()
    sp = Path(shard_str)
    prov = pq.read_table(sp / 'provenance.parquet').to_pydict()
    obs_list = pq.read_table(sp / 'observations.parquet').to_pylist()
    tgt_list = pq.read_table(sp / 'targets.parquet').to_pylist() if attach_label else None
    item_fns = {fam: _get_item_and_collate_fns(fam, bounds, cfg)[0] for fam in families}
    rows: list[tuple[tuple[int, int, int], int, dict[str, dict]]] = []
    for i, (r, c, rw, sp_name, plbl) in enumerate(
        zip(prov['run'], prov['chunk'], prov['row'], prov['split'], prov['label'])
    ):
        if sp_name != expected_split:
            continue
        ob = obs_list[i]
        lbl = int(tgt_list[i]['label']) if tgt_list is not None else int(plbl)
        dom = compute_observable_containment(ob, geom, bounds, config=cfg)
        fam_items: dict[str, dict] = {}
        for fam, fn in item_fns.items():
            it = fn(len(rows), ob, lbl if attach_label else None, dom, geom, bounds, cfg)
            it['obs_domain_id'] = int(dom['obs_domain_id'])
            fam_items[fam] = it
        rows.append(((int(r), int(c), int(rw)), int(lbl), fam_items))
    return rows


def _extract_obs_list_for_families(
    obs_list: list[dict],
    geom: dict,
    bounds: dict,
    cfg: dict[str, Any],
    families: tuple[str, ...],
) -> dict[str, list[dict]]:
    item_fns = {fam: _get_item_and_collate_fns(fam, bounds, cfg)[0] for fam in families}
    out: dict[str, list[dict]] = {fam: [] for fam in families}
    for i, ob in enumerate(obs_list):
        dom = compute_observable_containment(ob, geom, bounds, config=cfg)
        for fam, fn in item_fns.items():
            it = fn(i, ob, None, dom, geom, bounds, cfg)
            it['obs_domain_id'] = int(dom['obs_domain_id'])
            out[fam].append(it)
    return out


def _extract_ood_chunk_worker(task: tuple) -> tuple[str, int, dict[str, list[dict]]]:
    oname, c_idx, obs_chunk, geom, bounds, cfg, families = task
    _configure_runtime()
    return oname, c_idx, _extract_obs_list_for_families(obs_chunk, geom, bounds, cfg, families)


def _extract_ood_parallel(
    ood_specs: list[tuple[str, Path]],
    geom: dict,
    bounds: dict,
    cfg: dict[str, Any],
    families: tuple[str, ...],
    ppool: ProcessPoolExecutor,
    chunk_size: int = 200,
) -> list[tuple[str, dict[str, list[dict]], np.ndarray]]:
    tasks = []
    yo_map: dict[str, np.ndarray] = {}
    for oname, oroot in ood_specs:
        obs_list = pq.read_table(assert_unsealed_path(oroot / 'shards/000/observations.parquet')).to_pylist()
        ometa = pd.read_parquet(assert_unsealed_path(oroot / 'truth_meta.parquet'))
        yo_map[oname] = ometa['label'].to_numpy(np.int64)
        for ci, st in enumerate(range(0, len(obs_list), chunk_size)):
            tasks.append((oname, ci, obs_list[st:st + chunk_size], geom, bounds, cfg, families))
    chunks = list(ppool.map(_extract_ood_chunk_worker, tasks))
    grouped: dict[str, list[tuple[int, dict[str, list[dict]]]]] = {oname: [] for oname, _ in ood_specs}
    for oname, ci, fdict in chunks:
        grouped[oname].append((ci, fdict))
    out = []
    for oname, _ in ood_specs:
        parts = sorted(grouped[oname], key=lambda x: x[0])
        merged: dict[str, list[dict]] = {fam: [] for fam in families}
        for _, fdict in parts:
            for fam in families:
                merged[fam].extend(fdict[fam])
        out.append((oname, merged, yo_map[oname]))
    return out


def _extract_ood_worker(task: tuple) -> tuple[str, dict[str, list[dict]], np.ndarray]:
    oname, oroot_str, geom, bounds, cfg, families = task
    _configure_runtime()
    oroot = Path(oroot_str)
    obs_list = pq.read_table(assert_unsealed_path(oroot / 'shards/000/observations.parquet')).to_pylist()
    ometa = pd.read_parquet(assert_unsealed_path(oroot / 'truth_meta.parquet'))
    yo = ometa['label'].to_numpy(np.int64)
    return oname, _extract_obs_list_for_families(obs_list, geom, bounds, cfg, families), yo


def _extract_stress_worker(task: tuple) -> tuple[str, dict[str, list[dict]]]:
    st_name, pkw, c_over, obs_list, geom, bounds, families = task
    _configure_runtime()
    rng_st = np.random.default_rng(32510)
    cfg_use = {**BASE_OBS_CFG, **c_over}
    pert_obs = [
        (_perturb_observation_for_stress(ob, bounds, rng_st, **pkw) if pkw else ob)
        for ob in obs_list
    ]
    return st_name, _extract_obs_list_for_families(pert_obs, geom, bounds, cfg_use, families)


def _batches_from_items(
    items: list[dict],
    collate_fn,
    *,
    batch_size: int = 64,
    sidecar_df: pd.DataFrame | None = None,
    labels_override: np.ndarray | None = None,
):
    for st in range(0, len(items), batch_size):
        chunk = items[st:st + batch_size]
        if sidecar_df is not None:
            aug_chunk = []
            for j, it in enumerate(chunk):
                row = sidecar_df.iloc[st + j]
                it2 = dict(it)
                it2['label_7'] = int(row['label_7'])
                it2['tau_branch_valid'] = int(row['tau_branch_valid'])
                if labels_override is not None:
                    it2['label'] = int(labels_override[st + j])
                aug_chunk.append(it2)
            chunk = aug_chunk
        elif labels_override is not None:
            chunk = [{**it, 'label': int(labels_override[st + j])} for j, it in enumerate(chunk)]

        batch = collate_fn(chunk)
        if sidecar_df is not None:
            sl = sidecar_df.iloc[st:st + len(chunk)]
            batch['label_7'] = torch.tensor(sl['label_7'].to_numpy(np.int64), dtype=torch.long)
            batch['tau_branch_valid'] = torch.tensor(sl['tau_branch_valid'].to_numpy(np.int64), dtype=torch.long)
        if labels_override is not None:
            batch['label'] = torch.tensor(labels_override[st:st + len(chunk)], dtype=torch.long)
        if 'obs_domain_id' in chunk[0]:
            batch['obs_domain_id'] = torch.tensor([int(c['obs_domain_id']) for c in chunk], dtype=torch.long)
        yield batch


# ---------------------------------------------------------------------------
# Metrics, Tail Evaluators, Calibration & Paired Bootstrap (seed=20263250)
# ---------------------------------------------------------------------------
def _tau_scores_np(probs: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    p /= np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    pt = np.clip(p[:, 2], 1e-8, 1.0 - 1e-8)
    return np.log(pt) - np.log(1.0 - pt)


def _ba4_nll_ece(labels: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int64)
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-8, 1.0)
    p /= np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    pred = p.argmax(axis=1)
    recalls = [float((pred[y == c] == c).mean()) for c in range(4) if (y == c).any()]
    ba4 = float(np.mean(recalls)) * 100.0 if recalls else 0.0
    nll = float(-np.mean(np.log(p[np.arange(len(y)), y])))
    ece_top1 = compute_multiclass_ece(p, y, n_bins=15)
    ece_tau = compute_tau_binary_ece(p, y, n_bins=15)
    return {
        'n_events': int(len(y)),
        'BA4_pct': ba4,
        'NLL': nll,
        'ECE_top1': ece_top1,
        'ECE_tau': ece_tau,
    }


def _clopper_pearson_ci95_pct(k: int, n: int) -> list[float]:
    if n <= 0:
        return [0.0, 0.0]
    lo = 0.0 if k == 0 else float(scipy_beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(scipy_beta.ppf(0.975, k + 1, n - k))
    return [lo * 100.0, hi * 100.0]


def _within_slice_tail_eval(scores: np.ndarray, labels: np.ndarray, mask: np.ndarray, fpr_target: float) -> dict[str, Any]:
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return {'valid': False}
    res = _tau_efficiency_at_fpr(scores[m], labels[m], fpr_target)
    if res['tau_eff'] is None:
        return {'valid': False}
    return {
        'valid': True,
        'tau_eff_pct': float(res['tau_eff'] * 100.0),
        'tau_selected': int(res['tau_selected']),
        'tau_total': int(res['tau_total']),
        'bkg_selected': int(res['bkg_selected']),
        'bkg_total': int(res['bkg_total']),
        'actual_fpr_pct': float(res['actual_fpr'] * 100.0),
    }


def _split_half_tail_eval(
    scores: np.ndarray,
    labels: np.ndarray,
    keys_df: pd.DataFrame,
    mask: np.ndarray,
    fpr_target: float,
) -> dict[str, Any]:
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return {'valid': False}
    hvals = np.array([
        int(hashlib.blake2b(f'{int(r)}:{int(c)}:{int(rw)}'.encode(), digest_size=8).hexdigest(), 16) & 1
        for r, c, rw in zip(keys_df['run'].iloc[idx], keys_df['chunk'].iloc[idx], keys_df['row'].iloc[idx])
    ], dtype=np.int64)
    lock_m = idx[hvals == 0]
    held_m = idx[hvals == 1]
    bg_lock = scores[lock_m][labels[lock_m] != 2]
    bg_held = scores[held_m][labels[held_m] != 2]
    sg_held = scores[held_m][labels[held_m] == 2]
    nb_lock, nb_held, ns_held = len(bg_lock), len(bg_held), len(sg_held)
    if nb_lock == 0 or nb_held == 0 or ns_held == 0:
        return {'valid': False}
    bg_sorted = np.sort(bg_lock)[::-1]
    k_allow = int(math.floor(fpr_target * nb_lock))
    thr = float(bg_sorted[k_allow]) if k_allow < nb_lock else float(-1e9)
    fp = int(np.sum(bg_held > thr))
    tp = int(np.sum(sg_held > thr))
    return {
        'valid': True,
        'n_lock_bg': int(nb_lock),
        'lock_allowed_bg': int(k_allow),
        'held_TP': tp,
        'held_tau': int(ns_held),
        'held_FP': fp,
        'held_B': int(nb_held),
        'held_tau_eff_pct': float(tp / ns_held * 100.0),
        'held_tau_eff_ci95_pct': _clopper_pearson_ci95_pct(tp, ns_held),
        'held_FPR_pct': float(fp / nb_held * 100.0),
        'low_bg_warning': bool(k_allow < 10),
    }


def _paired_bootstrap(
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    labels: np.ndarray,
    family_codes: np.ndarray,
    mask: np.ndarray,
    *,
    scores_a: np.ndarray | None = None,
    scores_b: np.ndarray | None = None,
    n_boot: int = 1000,
    seed: int = 20263250,
) -> dict[str, Any]:
    m = np.asarray(mask, dtype=bool)
    pa = np.clip(probs_a[m].astype(np.float64), 1e-8, 1.0)
    pa /= np.maximum(pa.sum(axis=1, keepdims=True), 1e-12)
    pb = np.clip(probs_b[m].astype(np.float64), 1e-8, 1.0)
    pb /= np.maximum(pb.sum(axis=1, keepdims=True), 1e-12)
    sa = _tau_scores_np(pa) if scores_a is None else np.asarray(scores_a[m], dtype=np.float64)
    sb = _tau_scores_np(pb) if scores_b is None else np.asarray(scores_b[m], dtype=np.float64)
    ya = pa.argmax(axis=1)
    yb = pb.argmax(axis=1)
    y = labels[m].astype(np.int64)
    n_ev = len(y)
    nll_diff = (-np.log(pa[np.arange(n_ev), y])) - (-np.log(pb[np.arange(n_ev), y]))

    fams = family_codes[m]
    _, inv = np.unique(fams, return_inverse=True)
    n_fam = int(inv.max()) + 1 if n_ev > 0 else 0
    one_to_one = n_fam == n_ev

    ord_a = np.argsort(sa)[::-1]
    ord_b = np.argsort(sb)[::-1]
    is_tau_a = y[ord_a] == 2
    is_bg_a = ~is_tau_a
    is_tau_b = y[ord_b] == 2
    is_bg_b = ~is_tau_b

    hit_a = (ya == y).astype(np.int64)
    hit_b = (yb == y).astype(np.int64)

    def _eval_batch_weights(w_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        tot_y = w_mat @ np.column_stack([(y == c).astype(np.float64) for c in range(4)])
        hit_cnt_a = w_mat @ np.column_stack([((y == c) & (hit_a == 1)).astype(np.float64) for c in range(4)])
        hit_cnt_b = w_mat @ np.column_stack([((y == c) & (hit_b == 1)).astype(np.float64) for c in range(4)])
        present = tot_y[0] > 0
        ba_a = np.mean(hit_cnt_a[:, present] / np.maximum(tot_y[:, present], 1e-12), axis=1)
        ba_b = np.mean(hit_cnt_b[:, present] / np.maximum(tot_y[:, present], 1e-12), axis=1)
        dnll_w = (w_mat @ nll_diff) / np.maximum(w_mat.sum(axis=1), 1e-12)

        wa = w_mat[:, ord_a]
        wb = w_mat[:, ord_b]
        cbg_a = np.cumsum(wa * is_bg_a[None, :], axis=1)
        csg_a = np.cumsum(wa * is_tau_a[None, :], axis=1)
        cbg_b = np.cumsum(wb * is_bg_b[None, :], axis=1)
        csg_b = np.cumsum(wb * is_tau_b[None, :], axis=1)

        def _eff_batch(cbg: np.ndarray, csg: np.ndarray, nbg: np.ndarray, nsg: np.ndarray, fpr: float) -> np.ndarray:
            k_allow = np.floor(fpr * nbg).astype(np.int64)
            mask_gt = cbg > k_allow[:, None]
            has_gt = mask_gt.any(axis=1)
            first_gt = np.argmax(mask_gt, axis=1)
            pos = np.where(has_gt, first_gt - 1, n_ev - 1)
            return np.where((k_allow > 0) & (pos >= 0), csg[np.arange(len(nbg)), np.maximum(pos, 0)] / nsg, 0.0)

        e1a = _eff_batch(cbg_a, csg_a, cbg_a[:, -1], np.maximum(csg_a[:, -1], 1e-12), 0.01)
        e1b = _eff_batch(cbg_b, csg_b, cbg_b[:, -1], np.maximum(csg_b[:, -1], 1e-12), 0.01)
        e01a = _eff_batch(cbg_a, csg_a, cbg_a[:, -1], np.maximum(csg_a[:, -1], 1e-12), 0.001)
        e01b = _eff_batch(cbg_b, csg_b, cbg_b[:, -1], np.maximum(csg_b[:, -1], 1e-12), 0.001)
        return (ba_a - ba_b) * 100.0, dnll_w, (e1a - e1b) * 100.0, (e01a - e01b) * 100.0

    pt_vals = _eval_batch_weights(np.ones((1, n_ev), dtype=np.float64))
    pt_ba, pt_nll, pt_e1, pt_e01 = (float(x[0]) for x in pt_vals)

    rng = np.random.default_rng(seed)
    dba = np.empty(n_boot, dtype=np.float64)
    dnll = np.empty(n_boot, dtype=np.float64)
    de1 = np.empty(n_boot, dtype=np.float64)
    de01 = np.empty(n_boot, dtype=np.float64)
    chunk = 100
    for start in range(0, n_boot, chunk):
        bs = min(chunk, n_boot - start)
        draws = rng.integers(0, n_fam, size=(bs, n_fam))
        flat_idx = (draws + np.arange(bs, dtype=np.int64)[:, None] * n_fam).ravel()
        f_w = np.bincount(flat_idx, minlength=bs * n_fam).reshape(bs, n_fam).astype(np.float64)
        w_mat = f_w if one_to_one else f_w[:, inv]
        cba, cnll, ce1, ce01 = _eval_batch_weights(w_mat)
        dba[start:start + bs] = cba
        dnll[start:start + bs] = cnll
        de1[start:start + bs] = ce1
        de01[start:start + bs] = ce01

    return {
        'delta_BA4_pp': pt_ba,
        'delta_BA4_ci95_pp': [float(np.quantile(dba, 0.025)), float(np.quantile(dba, 0.975))],
        'delta_NLL': pt_nll,
        'delta_NLL_ci95': [float(np.quantile(dnll, 0.025)), float(np.quantile(dnll, 0.975))],
        'delta_tau1pct_pp': pt_e1,
        'delta_tau1pct_ci95_pp': [float(np.quantile(de1, 0.025)), float(np.quantile(de1, 0.975))],
        'delta_tau0p1pct_pp': pt_e01,
        'delta_tau0p1pct_ci95_pp': [float(np.quantile(de01, 0.025)), float(np.quantile(de01, 0.975))],
    }


def _evaluate_predictions_suite(
    probs4: np.ndarray,
    labels: np.ndarray,
    keys_df: pd.DataFrame,
    slices: dict[str, np.ndarray],
    *,
    scores: np.ndarray | None = None,
) -> dict[str, Any]:
    sc = _tau_scores_np(probs4) if scores is None else np.asarray(scores, dtype=np.float64)
    out: dict[str, Any] = {
        'slices': {sn: _ba4_nll_ece(labels[mk], probs4[mk]) for sn, mk in slices.items()},
        'pAUC_1pct': {
            sn: compute_normalized_pauc(sc[mk], labels[mk], 0.01)
            for sn, mk in slices.items()
        },
        'pAUC_0p1pct': {
            sn: compute_normalized_pauc(sc[mk], labels[mk], 0.001)
            for sn, mk in slices.items()
        },
        'within_slice_1pct_fpr': {
            sn: _within_slice_tail_eval(sc, labels, mk, 0.01)
            for sn, mk in slices.items()
        },
        'within_slice_0p1pct_fpr': {
            sn: _within_slice_tail_eval(sc, labels, mk, 0.001)
            for sn, mk in slices.items()
        },
        'split_half_1pct_fpr': {
            sn: _split_half_tail_eval(sc, labels, keys_df, mk, 0.01)
            for sn, mk in slices.items()
        },
        'split_half_0p1pct_fpr': {
            sn: _split_half_tail_eval(sc, labels, keys_df, mk, 0.001)
            for sn, mk in slices.items()
        },
    }
    return out


def _screen_domain_calibrate(
    seed_probs_t8: dict[int, np.ndarray],
    labels_t8: np.ndarray,
    in_screen: np.ndarray,
    obs_dom_t8: np.ndarray,
) -> np.ndarray:
    seeds = sorted(seed_probs_t8.keys())
    lp_stack = np.stack([
        np.log(np.clip(seed_probs_t8[s], 1e-8, 1.0))
        - np.mean(np.log(np.clip(seed_probs_t8[s], 1e-8, 1.0)), axis=1, keepdims=True)
        for s in seeds
    ], axis=0)
    gfit = _fit_domain_convex_blend(lp_stack[:, in_screen, :], labels_t8[in_screen])
    out = _apply_domain_convex_blend(lp_stack, gfit)
    for did in (1, 2, 3):
        m_fit = in_screen & (obs_dom_t8 == did)
        m_app = obs_dom_t8 == did
        if int(m_fit.sum()) >= 40:
            dfit = _fit_domain_convex_blend(lp_stack[:, m_fit, :], labels_t8[m_fit])
            out[m_app] = _apply_domain_convex_blend(lp_stack[:, m_app, :], dfit)
    return out


# ---------------------------------------------------------------------------
# Stage-I Controlled Exploration & Hyperparameter Selection
# ---------------------------------------------------------------------------
def _stage1_ablation_candidates() -> list[tuple[str, TailDROConfig]]:
    """Controlled Stage-I grid (all evaluated strictly on internal_dev_screen 1.6k)."""
    return [
        ('pure_ce_3p', TailDROConfig(
            dro_mode='cvar_balanced', lambda_dro=0.0, lambda_nll=1.0, lambda_kl=0.0,
            use_bank=False, lr=5.0e-5, passes=3, batch_size=128, seed=42,
        )),
        ('pure_dro_no_anchor', TailDROConfig(
            dro_mode='cvar_balanced', margin=0.5, surrogate='softplus', lambda_dro=1.0, lambda_nll=0.0, lambda_kl=0.0,
            use_bank=True, bank_capacity=8192, lr=5.0e-5, passes=3, batch_size=128, seed=42,
        )),
        ('dro_anchored_nobank', TailDROConfig(
            dro_mode='cvar_balanced', margin=0.5, surrogate='softplus', lambda_dro=0.08, lambda_nll=0.5, lambda_kl=1.5,
            use_bank=False, lr=5.0e-5, passes=3, batch_size=128, seed=42,
        )),
        ('dro_bal_l003_kl15_bank8192', TailDROConfig(
            dro_mode='cvar_balanced', margin=0.5, surrogate='softplus', lambda_dro=0.03, lambda_nll=0.5, lambda_kl=1.5,
            use_bank=True, bank_capacity=8192, lr=5.0e-5, passes=3, batch_size=128, seed=42,
        )),
        ('dro_bal_l008_kl15_bank8192', TailDROConfig(
            dro_mode='cvar_balanced', margin=0.5, surrogate='softplus', lambda_dro=0.08, lambda_nll=0.5, lambda_kl=1.5,
            use_bank=True, bank_capacity=8192, lr=5.0e-5, passes=3, batch_size=128, seed=42,
        )),
        ('dro_bal_l015_kl15_bank8192', TailDROConfig(
            dro_mode='cvar_balanced', margin=0.5, surrogate='softplus', lambda_dro=0.15, lambda_nll=0.5, lambda_kl=1.5,
            use_bank=True, bank_capacity=8192, lr=5.0e-5, passes=3, batch_size=128, seed=42,
        )),
        ('sopa_l005_kl20_bank8192', TailDROConfig(
            dro_mode='sopa_s_hybrid', margin=0.35, lambda_dro=0.05, lambda_nll=0.5, lambda_kl=2.0,
            use_bank=True, bank_capacity=8192, lr=5.0e-5, passes=3, batch_size=128, seed=42,
        )),
    ]


def _unified_net_available() -> bool:
    import importlib.util
    return importlib.util.find_spec('analysis.flavour.nextgen.unified_pinpoint_net') is not None


def _default_stage1_checkpoints() -> dict[str, tuple[str, Path]]:
    specs: dict[str, tuple[str, Path]] = {
        'tracklet_ipformer': (
            'tracklet_ipformer',
            NEXTGEN_ROOT / 'exp_tracklet_ipformer/s1_tracklet_ipformer/best.pt',
        ),
        'dual_regime_modalnet': (
            'dual_regime_modalnet',
            NEXTGEN_ROOT / 'exp_dual_regime/s1_dual_regime_modalnet/best_ema.pt',
        ),
    }
    if _unified_net_available():
        uni_ckpt = NEXTGEN_ROOT / 'exp_unified_net/s1_unified_pinpoint_net/best.pt'
        if not uni_ckpt.is_file():
            uni_ckpt = NEXTGEN_ROOT / 'exp_unified_net/s1_unified_pinpoint_net/best_ema.pt'
        if uni_ckpt.is_file():
            specs['unified_pinpoint_net'] = ('unified_pinpoint_net', uni_ckpt)
    return specs


def _default_stage2_checkpoints() -> dict[str, tuple[str, dict[int, Path]]]:
    specs: dict[str, tuple[str, dict[int, Path]]] = {
        'tracklet_ipformer': (
            'tracklet_ipformer',
            {s: NEXTGEN_ROOT / f'exp_tracklet_ipformer/stage2/seed_{s}/best.pt' for s in STAGE2_SEEDS},
        ),
        'dual_regime_modalnet': (
            'dual_regime_modalnet',
            {s: NEXTGEN_ROOT / f'exp_dual_regime/stage2/seed_{s}/best_ema.pt' for s in STAGE2_SEEDS},
        ),
    }
    if _unified_net_available():
        uni_seeds: dict[int, Path] = {}
        for s in STAGE2_SEEDS:
            for cand in (
                NEXTGEN_ROOT / f'exp_unified_net/stage2/seed_{s}/best.pt',
                NEXTGEN_ROOT / f'exp_unified_net/stage2/seed_{s}/best_ema.pt',
            ):
                if cand.is_file():
                    uni_seeds[s] = cand
                    break
        if len(uni_seeds) == 3:
            specs['unified_pinpoint_net'] = ('unified_pinpoint_net', uni_seeds)
    return specs


def _stage1_cache_dir() -> Path:
    return OUT_ROOT / '_tmp_item_cache_s1'


def _stage2_cache_dir() -> Path:
    return OUT_ROOT / '_tmp_item_cache_s2'


def preextract_stage1_part(part: int, n_parts: int = 16, workers: int = 8) -> dict[str, Any]:
    """Sharded Stage-I item extraction (`part=0..n_parts-1` train shards 000..079, `part=n_parts` OOD)."""
    import pickle
    t0 = time.perf_counter()
    with install_stage1_seal_guard() as guard:
        assert_unsealed_split('internal_train', stage=1)
        geom = json.loads(assert_unsealed_path(CLEAN_V3_ROOT / 'geometry.json').read_text())
        bounds = detector_bounds(geom)
        bounds['_plane_x0'] = np.array([cumulative_radiation_length(float(z), geom) for z in bounds['pixel_Zpos']], dtype=np.float32)
        families = ('tracklet_ipformer', 'dual_regime_modalnet')
        cdir = _stage1_cache_dir()
        cdir.mkdir(parents=True, exist_ok=True)

        if part == n_parts:
            out_f = cdir / 'ood_s1.pkl'
            if out_f.is_file():
                return {'part': part, 'status': 'cached'}
            s1_oods = [
                (oname, oroot)
                for oname, oroot in OOD_SPECS
                if oname in ('gen_ood_w184_epps21_nc_charm', 'gen_ood_run10085_10088')
            ]
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as ppool:
                ood_res = _extract_ood_parallel(s1_oods, geom, bounds, BASE_OBS_CFG, families, ppool, chunk_size=200)
            tmp_f = cdir / '.tmp_ood_s1.pkl'
            with tmp_f.open('wb') as fh:
                pickle.dump(ood_res, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_f, out_f)
            return {'part': part, 'ood_sets': len(ood_res), 'seconds': float(time.perf_counter() - t0), 'violations': len(guard.violations)}

        out_f = cdir / f'train_part_{part:02d}_of_{n_parts:02d}.pkl'
        if out_f.is_file():
            return {'part': part, 'status': 'cached'}
        shards = load_authorized_split_shards(CLEAN_V3_ROOT / 'data', 'train', stage=1)
        my_shards = shards[part::n_parts]
        shm_base = _shm_scratch_root()
        with tempfile.TemporaryDirectory(prefix=f'tdro-s1-p{part}-', dir=shm_base) as tmp_d:
            tmp_p = Path(tmp_d) / 'shards'
            tmp_p.mkdir(parents=True, exist_ok=True)

            def _copy_shard(sp: Path) -> Path:
                dst = tmp_p / sp.name
                dst.mkdir(parents=True, exist_ok=True)
                for fname in ('observations.parquet', 'targets.parquet', 'provenance.parquet'):
                    shutil.copyfile(sp / fname, dst / fname)
                return dst

            with ThreadPoolExecutor(max_workers=min(8, workers)) as tpool:
                local_shards = list(tpool.map(_copy_shard, my_shards))
            tasks = [(str(sp), 'train', True, geom, bounds, BASE_OBS_CFG, families) for sp in local_shards]
            with ProcessPoolExecutor(max_workers=min(workers, len(tasks)), mp_context=mp.get_context('spawn')) as ppool:
                part_rows = list(ppool.map(_extract_shard_worker, tasks))
        rows = [r for pr in part_rows for r in pr]
        tmp_f = cdir / f'.tmp_train_part_{part:02d}_of_{n_parts:02d}.pkl'
        with tmp_f.open('wb') as fh:
            pickle.dump(rows, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_f, out_f)
        return {'part': part, 'n_events': len(rows), 'seconds': float(time.perf_counter() - t0), 'violations': len(guard.violations)}


def preextract_stage2_part(part: int, n_parts: int = 4, workers: int = 8) -> dict[str, Any]:
    """Sharded Stage-II item extraction (`part=0..n_parts-1` Tune8k 080..099, `part=n_parts` 6 OOD + 7 Stress)."""
    import pickle
    t0 = time.perf_counter()
    with SealAuditGuard() as guard:
        for sp in ('train', 'validation', 'tune_screen', 'tune6k', 'ood'):
            assert_unsealed_split(sp, stage=2)
        geom = json.loads(assert_unsealed_path(CLEAN_V3_ROOT / 'geometry.json').read_text())
        bounds = detector_bounds(geom)
        bounds['_plane_x0'] = np.array([cumulative_radiation_length(float(z), geom) for z in bounds['pixel_Zpos']], dtype=np.float32)
        families = ('tracklet_ipformer', 'dual_regime_modalnet')
        cdir = _stage2_cache_dir()
        cdir.mkdir(parents=True, exist_ok=True)

        if part == n_parts:
            out_f = cdir / 'ood_stress_s2.pkl'
            if out_f.is_file():
                return {'part': part, 'status': 'cached'}
            va_shards = load_authorized_split_shards(CLEAN_V3_ROOT / 'data', 'validation', stage=2)
            sh80 = va_shards[0]
            obs_080 = pq.read_table(assert_unsealed_path(sh80 / 'observations.parquet')).to_pylist()
            prov_080 = pd.read_parquet(assert_unsealed_path(sh80 / 'provenance.parquet'), columns=KEY + FAMILY + ['label'])
            y_080 = prov_080['label'].to_numpy(np.int64)
            st_tasks = [(st_name, pkw, c_over, obs_080, geom, bounds, families) for st_name, pkw, c_over in STRESS_SPECS]
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as ppool:
                ood_res = _extract_ood_parallel(list(OOD_SPECS), geom, bounds, BASE_OBS_CFG, families, ppool, chunk_size=200)
                st_res = list(ppool.map(_extract_stress_worker, st_tasks))
            tmp_f = cdir / '.tmp_ood_stress_s2.pkl'
            with tmp_f.open('wb') as fh:
                pickle.dump({'ood_res': ood_res, 'st_res': st_res, 'y_080': y_080}, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_f, out_f)
            return {'part': part, 'ood_sets': len(ood_res), 'stress_sets': len(st_res), 'seconds': float(time.perf_counter() - t0), 'violations': len(guard.violations)}

        out_f = cdir / f'val_part_{part:02d}_of_{n_parts:02d}.pkl'
        if out_f.is_file():
            return {'part': part, 'status': 'cached'}
        va_shards = load_authorized_split_shards(CLEAN_V3_ROOT / 'data', 'validation', stage=2)
        my_shards = va_shards[part::n_parts]
        shm_base = _shm_scratch_root()
        with tempfile.TemporaryDirectory(prefix=f'tdro-s2-p{part}-', dir=shm_base) as tmp_d:
            tmp_p = Path(tmp_d) / 'shards'
            tmp_p.mkdir(parents=True, exist_ok=True)

            def _copy_va(sp: Path) -> Path:
                dst = tmp_p / sp.name
                dst.mkdir(parents=True, exist_ok=True)
                for fname in ('observations.parquet', 'provenance.parquet'):
                    shutil.copyfile(sp / fname, dst / fname)
                return dst

            with ThreadPoolExecutor(max_workers=min(8, workers)) as tpool:
                local_shards = list(tpool.map(_copy_va, my_shards))
            tasks = [(str(sp), 'validation', False, geom, bounds, BASE_OBS_CFG, families) for sp in local_shards]
            with ProcessPoolExecutor(max_workers=min(workers, len(tasks)), mp_context=mp.get_context('spawn')) as ppool:
                va_parts = list(ppool.map(_extract_shard_worker, tasks))
        rows = [r for pr in va_parts for r in pr]
        tmp_f = cdir / f'.tmp_val_part_{part:02d}_of_{n_parts:02d}.pkl'
        with tmp_f.open('wb') as fh:
            pickle.dump(rows, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_f, out_f)
        return {'part': part, 'n_events': len(rows), 'seconds': float(time.perf_counter() - t0), 'violations': len(guard.violations)}


def _extract_stage1_caches(
    model_specs: dict[str, tuple[str, Path]],
    geom: dict,
    bounds: dict,
    device: torch.device,
) -> dict[str, Any]:
    """Extract Stage-I authorized items & penultimate latents under strict Stage-I seal guard."""
    import pickle
    tr_keys = pd.read_parquet(assert_unsealed_path(SPLITS_DIR / 'internal_train_keys.parquet'))
    dev_keys = pd.read_parquet(assert_unsealed_path(SPLITS_DIR / 'internal_dev_screen_keys.parquet'))
    val_keys = pd.read_parquet(assert_unsealed_path(SPLITS_DIR / 'internal_val_eval_keys.parquet'))

    sc_df = pd.read_parquet(assert_unsealed_path(CLEAN_V3_ROOT / 'truth_sidecar/event_sidecar.parquet'))
    pos_sc = pd.MultiIndex.from_frame(sc_df[KEY]).get_indexer(pd.MultiIndex.from_frame(tr_keys[KEY]))
    if (pos_sc < 0).any():
        raise ValueError('internal_train keys missing from event_sidecar.parquet')
    tr_sidecar = sc_df.iloc[pos_sc].reset_index(drop=True)
    del sc_df

    families = tuple(sorted({fam for fam, _ in model_specs.values()}))
    cdir = _stage1_cache_dir()
    part_files = sorted(cdir.glob('train_part_*_of_*.pkl'))
    ood_file = cdir / 'ood_s1.pkl'
    t0 = time.perf_counter()

    pen_dir = OUT_ROOT / '_penult_cache_s1'
    pen_dir.mkdir(parents=True, exist_ok=True)
    if all((pen_dir / f'{mname}.pt').is_file() for mname in model_specs):
        caches_fast: dict[str, Any] = {'models': {}, 'tr_keys': tr_keys, 'dev_keys': dev_keys, 'val_keys': val_keys, 'ood_labels': {}}
        for mname, (fam, ckpt_path) in model_specs.items():
            blob = torch.load(pen_dir / f'{mname}.pt', map_location='cpu', weights_only=False)
            model, _ = _load_model_checkpoint(fam, ckpt_path, geom, device)
            caches_fast['ood_labels'].update(blob['ood_labels'])
            caches_fast['models'][mname] = {
                'family': fam,
                'ckpt_path': str(ckpt_path),
                'model': model,
                'train': blob['train'],
                'dev_screen': blob['dev_screen'],
                'val_eval': blob['val_eval'],
                'ood': blob['ood'],
                'truth_invariance': blob['truth_invariance'],
                'max_cache_equiv_diff': blob['max_cache_equiv_diff'],
            }
        print(f'[Stage-I Cache] Loaded saved penultimate caches from {pen_dir} in {time.perf_counter() - t0:.1f}s', flush=True)
        return caches_fast

    if part_files and ood_file.is_file() and all(f in ('tracklet_ipformer', 'dual_regime_modalnet') for f in families):
        with ood_file.open('rb') as fh:
            ood_res = _pickle_load(fh)
        caches: dict[str, Any] = {'models': {}, 'tr_keys': tr_keys, 'dev_keys': dev_keys, 'val_keys': val_keys, 'ood_labels': {}}
        for oname, _, yo in ood_res:
            caches['ood_labels'][oname] = yo

        tr_key_arr = tr_keys[KEY].to_numpy()
        dev_key_arr = dev_keys[KEY].to_numpy()
        val_key_arr = val_keys[KEY].to_numpy()

        for mname, (fam, ckpt_path) in model_specs.items():
            key_to_item: dict[tuple[int, int, int], dict] = {}
            for pf in part_files:
                with pf.open('rb') as fh:
                    rows = _pickle_load(fh)
                for k, _lbl, fdict in rows:
                    key_to_item[k] = fdict[fam]
                del rows
            tr_it = [key_to_item[(int(r), int(c), int(rw))] for r, c, rw in tr_key_arr]
            dev_it = [key_to_item[(int(r), int(c), int(rw))] for r, c, rw in dev_key_arr]
            val_it = [key_to_item[(int(r), int(c), int(rw))] for r, c, rw in val_key_arr]
            del key_to_item

            model, cfg = _load_model_checkpoint(fam, ckpt_path, geom, device)
            _, collate_fn = _get_item_and_collate_fns(fam, bounds, cfg)

            tr_c = extract_penultimate_cache(
                model,
                _batches_from_items(tr_it, collate_fn, batch_size=64, sidecar_df=tr_sidecar, labels_override=tr_keys['label'].to_numpy(np.int64)),
                device,
            )
            dev_c = extract_penultimate_cache(
                model,
                _batches_from_items(dev_it, collate_fn, batch_size=64, labels_override=dev_keys['label'].to_numpy(np.int64)),
                device,
            )
            val_c = extract_penultimate_cache(
                model,
                _batches_from_items(val_it, collate_fn, batch_size=64, labels_override=val_keys['label'].to_numpy(np.int64)),
                device,
            )
            ood_c = {}
            for oname, fam_dict, yo in ood_res:
                ood_c[oname] = extract_penultimate_cache(
                    model,
                    _batches_from_items(fam_dict[fam], collate_fn, batch_size=64, labels_override=yo),
                    device,
                )

            sample_b = next(_batches_from_items(val_it[:8], collate_fn, batch_size=8))
            sample_b_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in sample_b.items()}
            inv_res = verify_inference_truth_invariance(model, sample_b_dev, atol=1e-7)
            del tr_it, dev_it, val_it

            caches['models'][mname] = {
                'family': fam,
                'ckpt_path': str(ckpt_path),
                'model': model,
                'train': tr_c,
                'dev_screen': dev_c,
                'val_eval': val_c,
                'ood': ood_c,
                'truth_invariance': inv_res,
                'max_cache_equiv_diff': float(max(tr_c['max_verify_diff'], dev_c['max_verify_diff'], val_c['max_verify_diff'])),
            }
            torch.save({
                'train': tr_c,
                'dev_screen': dev_c,
                'val_eval': val_c,
                'ood': ood_c,
                'ood_labels': caches['ood_labels'],
                'truth_invariance': inv_res,
                'max_cache_equiv_diff': caches['models'][mname]['max_cache_equiv_diff'],
            }, pen_dir / f'{mname}.pt')
            print(
                f'[Stage-I Cache] {mname}: extracted & saved in {time.perf_counter() - t0:.1f}s '
                f'(max|cached-full|={caches["models"][mname]["max_cache_equiv_diff"]:.2e})',
                flush=True,
            )
        return caches
    else:
        shards = load_authorized_split_shards(CLEAN_V3_ROOT / 'data', 'train', stage=1)
        n_workers = min(8, max(1, os.cpu_count() or 1))
        shm_base = _shm_scratch_root()
        with tempfile.TemporaryDirectory(prefix='tdro-s1-raw-', dir=shm_base) as tmp_d:
            tmp_p = Path(tmp_d) / 'shards'
            tmp_p.mkdir(parents=True, exist_ok=True)

            def _copy_shard(sp: Path) -> Path:
                dst = tmp_p / sp.name
                dst.mkdir(parents=True, exist_ok=True)
                for fname in ('observations.parquet', 'targets.parquet', 'provenance.parquet'):
                    shutil.copyfile(sp / fname, dst / fname)
                return dst

            with ThreadPoolExecutor(max_workers=min(8, n_workers)) as tpool:
                local_shards = list(tpool.map(_copy_shard, shards))

            tasks = [(str(sp), 'train', True, geom, bounds, BASE_OBS_CFG, families) for sp in local_shards]
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context('spawn')) as ppool:
                part_rows = list(ppool.map(_extract_shard_worker, tasks))
                ood_tasks = [
                    (oname, str(oroot), geom, bounds, BASE_OBS_CFG, families)
                    for oname, oroot in OOD_SPECS
                    if oname in ('gen_ood_w184_epps21_nc_charm', 'gen_ood_run10085_10088')
                ]
                ood_res = list(ppool.map(_extract_ood_worker, ood_tasks))

    # Index extracted rows by (run, chunk, row)
    key_to_items: dict[tuple[int, int, int], dict[str, dict]] = {}
    for rows in part_rows:
        for k, _lbl, fdict in rows:
            key_to_items[k] = fdict
    del part_rows

    def _align_fam(keys_df: pd.DataFrame, fam: str) -> list[dict]:
        return [key_to_items[(int(r), int(c), int(rw))][fam] for r, c, rw in keys_df[KEY].to_numpy()]

    caches: dict[str, Any] = {'models': {}, 'tr_keys': tr_keys, 'dev_keys': dev_keys, 'val_keys': val_keys, 'ood_labels': {}}
    for oname, _, yo in ood_res:
        caches['ood_labels'][oname] = yo

    for mname, (fam, ckpt_path) in model_specs.items():
        model, cfg = _load_model_checkpoint(fam, ckpt_path, geom, device)
        _, collate_fn = _get_item_and_collate_fns(fam, bounds, cfg)

        tr_it = _align_fam(tr_keys, fam)
        dev_it = _align_fam(dev_keys, fam)
        val_it = _align_fam(val_keys, fam)

        tr_c = extract_penultimate_cache(
            model,
            _batches_from_items(tr_it, collate_fn, batch_size=64, sidecar_df=tr_sidecar, labels_override=tr_keys['label'].to_numpy(np.int64)),
            device,
        )
        dev_c = extract_penultimate_cache(
            model,
            _batches_from_items(dev_it, collate_fn, batch_size=64, labels_override=dev_keys['label'].to_numpy(np.int64)),
            device,
        )
        val_c = extract_penultimate_cache(
            model,
            _batches_from_items(val_it, collate_fn, batch_size=64, labels_override=val_keys['label'].to_numpy(np.int64)),
            device,
        )
        ood_c = {}
        for oname, fam_dict, yo in ood_res:
            ood_c[oname] = extract_penultimate_cache(
                model,
                _batches_from_items(fam_dict[fam], collate_fn, batch_size=64, labels_override=yo),
                device,
            )

        # Verify zero truth-leakage on this model before fine-tuning
        sample_b = next(_batches_from_items(val_it[:8], collate_fn, batch_size=8))
        sample_b_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in sample_b.items()}
        inv_res = verify_inference_truth_invariance(model, sample_b_dev, atol=1e-7)

        caches['models'][mname] = {
            'family': fam,
            'ckpt_path': str(ckpt_path),
            'model': model,
            'train': tr_c,
            'dev_screen': dev_c,
            'val_eval': val_c,
            'ood': ood_c,
            'truth_invariance': inv_res,
            'max_cache_equiv_diff': float(max(tr_c['max_verify_diff'], dev_c['max_verify_diff'], val_c['max_verify_diff'])),
        }
        print(
            f'[Stage-I Cache] {mname}: extracted {len(tr_it)}+{len(dev_it)}+{len(val_it)} + OOD in '
            f'{time.perf_counter() - t0:.1f}s (max|cached-full|={caches["models"][mname]["max_cache_equiv_diff"]:.2e})',
            flush=True,
        )

    return caches


def run_stage1(models_filter: list[str] | None = None) -> dict[str, Any]:
    """Run Stage-I DRO ablations, dev-screen hyperparameter lock, and val_eval + W184 evaluation."""
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    device = torch.device('cuda:0')
    t0 = time.perf_counter()

    with install_stage1_seal_guard() as guard:
        assert_unsealed_split('internal_train', stage=1)
        assert_unsealed_split('internal_screen', stage=1)
        assert_unsealed_split('internal_val', stage=1)

        geom = json.loads(assert_unsealed_path(CLEAN_V3_ROOT / 'geometry.json').read_text())
        bounds = detector_bounds(geom)
        bounds['_plane_x0'] = np.array([cumulative_radiation_length(float(z), geom) for z in bounds['pixel_Zpos']], dtype=np.float32)

        all_specs = _default_stage1_checkpoints()
        if models_filter:
            all_specs = {k: v for k, v in all_specs.items() if k in models_filter}

        caches = _extract_stage1_caches(all_specs, geom, bounds, device)

        dev_keys = caches['dev_keys']
        val_keys = caches['val_keys']
        y_dev = dev_keys['label'].to_numpy(np.int64)
        y_val = val_keys['label'].to_numpy(np.int64)
        obs_dev = dev_keys['obs_domain_id'].to_numpy(np.int64)
        obs_val = val_keys['obs_domain_id'].to_numpy(np.int64)
        reg_val = val_keys['region_id'].to_numpy(np.int64)
        fam_val = pd.MultiIndex.from_frame(val_keys[FAMILY]).factorize()[0]

        dev_slices = {
            'All': np.ones(len(y_dev), dtype=bool),
            'D1_obs': obs_dev == 1,
        }
        val_slices = {
            'All': np.ones(len(y_val), dtype=bool),
            'D1_obs': obs_val == 1,
            'D2_obs': obs_val == 2,
            'D3_obs': obs_val == 3,
            'Pinpoint_PixXY': (reg_val == 0) | (reg_val == 1),
            'Front30': reg_val == 0,
            'Interm150': reg_val == 1,
            'Fortune_ScintOnly': reg_val == 3,
        }

        s1_dir = OUT_ROOT / 'stage1'
        s1_dir.mkdir(parents=True, exist_ok=True)
        candidates = _stage1_ablation_candidates()

        results_models: dict[str, Any] = {}
        dev_selection_scores: dict[str, list[dict[str, float]]] = {cname: [] for cname, _ in candidates}

        for mname, mdata in caches['models'].items():
            model = mdata['model']
            snap = snapshot_backbone_weights(model, head_attr='event_head_7')
            base_head_state = copy.deepcopy(model.event_head_7.state_dict())

            # Baseline (0 passes)
            pre_dev_p = mdata['dev_screen']['probs4_ref'].numpy().astype(np.float64)
            pre_val_p = mdata['val_eval']['probs4_ref'].numpy().astype(np.float64)
            pre_dev_m = _evaluate_predictions_suite(pre_dev_p, y_dev, dev_keys, dev_slices)
            pre_val_m = _evaluate_predictions_suite(pre_val_p, y_val, val_keys, val_slices)

            pre_ood = {}
            for oname, oc in mdata['ood'].items():
                op = oc['probs4_ref'].numpy().astype(np.float64)
                yo = caches['ood_labels'][oname]
                if 'nc_' in oname:
                    pre_ood[oname] = {
                        'n_events': int(len(yo)),
                        'mean_p_tau_pct': float(op[:, 2].mean() * 100.0),
                        'frac_pred_tau_pct': float((op.argmax(1) == 2).mean() * 100.0),
                        'frac_p_tau_gt_0p5_pct': float((op[:, 2] > 0.5).mean() * 100.0),
                    }
                else:
                    pre_ood[oname] = _ba4_nll_ece(yo, op)

            m_runs: dict[str, Any] = {
                'pretrained_baseline': {
                    'dev_screen': pre_dev_m,
                    'val_eval': pre_val_m,
                    'ood': pre_ood,
                }
            }
            val_probs_by_cand: dict[str, np.ndarray] = {'pretrained_baseline': pre_val_p}
            dev_probs_by_cand: dict[str, np.ndarray] = {'pretrained_baseline': pre_dev_p}

            for cname, cfg in candidates:
                model.event_head_7.load_state_dict(copy.deepcopy(base_head_state))
                head = freeze_backbone_for_tail_finetuning(model, head_attr='event_head_7')
                t_ft0 = time.perf_counter()
                _, hist = finetune_head_on_cached_latents(head, mdata['train'], cfg, device)
                ft_sec = float(time.perf_counter() - t_ft0)
                id_res = verify_backbone_bit_identity(model, snap, head_attr='event_head_7')

                dev_pred = predict_from_cached_latents(head, mdata['dev_screen']['penultimate'], device)
                val_pred = predict_from_cached_latents(head, mdata['val_eval']['penultimate'], device)
                dev_m = _evaluate_predictions_suite(dev_pred['probs4'], y_dev, dev_keys, dev_slices, scores=dev_pred['tau_score'])
                val_m = _evaluate_predictions_suite(val_pred['probs4'], y_val, val_keys, val_slices, scores=val_pred['tau_score'])

                ood_m = {}
                for oname, oc in mdata['ood'].items():
                    op = predict_from_cached_latents(head, oc['penultimate'], device)['probs4']
                    yo = caches['ood_labels'][oname]
                    if 'nc_' in oname:
                        ood_m[oname] = {
                            'n_events': int(len(yo)),
                            'mean_p_tau_pct': float(op[:, 2].mean() * 100.0),
                            'frac_pred_tau_pct': float((op.argmax(1) == 2).mean() * 100.0),
                            'frac_p_tau_gt_0p5_pct': float((op[:, 2] > 0.5).mean() * 100.0),
                        }
                    else:
                        ood_m[oname] = _ba4_nll_ece(yo, op)

                dnll_dev = dev_m['slices']['All']['NLL'] - pre_dev_m['slices']['All']['NLL']
                ece_ratio_dev = dev_m['slices']['All']['ECE_top1'] / max(pre_dev_m['slices']['All']['ECE_top1'], 1e-6)
                ece_tau_ratio_dev = dev_m['slices']['All']['ECE_tau'] / max(pre_dev_m['slices']['All']['ECE_tau'], 1e-6)
                fallback_dev = bool(dnll_dev > 0.01 or ece_ratio_dev >= 2.0 or ece_tau_ratio_dev >= 2.0)

                dnll_val = val_m['slices']['All']['NLL'] - pre_val_m['slices']['All']['NLL']
                ece_ratio_val = val_m['slices']['All']['ECE_top1'] / max(pre_val_m['slices']['All']['ECE_top1'], 1e-6)
                ece_tau_ratio_val = val_m['slices']['All']['ECE_tau'] / max(pre_val_m['slices']['All']['ECE_tau'], 1e-6)
                fallback_val = bool(dnll_val > 0.01 or ece_ratio_val >= 2.0 or ece_tau_ratio_val >= 2.0)

                dev_selection_scores[cname].append({
                    'dnll_dev': dnll_dev,
                    'ece_ratio_dev': ece_ratio_dev,
                    'ece_tau_ratio_dev': ece_tau_ratio_dev,
                    'ws1_all': dev_m['within_slice_1pct_fpr']['All']['tau_eff_pct'],
                    'pauc1_all': dev_m['pAUC_1pct']['All'],
                    'fallback_dev': float(fallback_dev),
                    'passes_sec10_2': float(dnll_dev <= 0.005 and ece_ratio_dev < 1.5 and ece_tau_ratio_dev < 1.5),
                })

                val_probs_by_cand[cname] = val_pred['probs4']
                dev_probs_by_cand[cname] = dev_pred['probs4']
                np.save(s1_dir / f'{mname}_{cname}_val_probs4.npy', val_pred['probs4'].astype(np.float32))

                m_runs[cname] = {
                    'config': cfg.__dict__,
                    'finetune_seconds': ft_sec,
                    'history': hist,
                    'backbone_bit_identity': id_res['passed'],
                    'dev_screen': dev_m,
                    'dev_delta_vs_pretrained': {
                        'delta_NLL': dnll_dev,
                        'ECE_top1_ratio': ece_ratio_dev,
                        'ECE_tau_ratio': ece_tau_ratio_dev,
                        'fallback_triggered': fallback_dev,
                    },
                    'val_eval': val_m,
                    'val_delta_vs_pretrained': {
                        'delta_NLL': dnll_val,
                        'ECE_top1_ratio': ece_ratio_val,
                        'ECE_tau_ratio': ece_tau_ratio_val,
                        'fallback_triggered': fallback_val,
                        'delta_BA4_pp': val_m['slices']['All']['BA4_pct'] - pre_val_m['slices']['All']['BA4_pct'],
                        'delta_ws1_All_pp': val_m['within_slice_1pct_fpr']['All']['tau_eff_pct'] - pre_val_m['within_slice_1pct_fpr']['All']['tau_eff_pct'],
                        'delta_ws1_D1_pp': val_m['within_slice_1pct_fpr']['D1_obs']['tau_eff_pct'] - pre_val_m['within_slice_1pct_fpr']['D1_obs']['tau_eff_pct'],
                        'delta_ws1_Pinpoint_pp': val_m['within_slice_1pct_fpr']['Pinpoint_PixXY']['tau_eff_pct'] - pre_val_m['within_slice_1pct_fpr']['Pinpoint_PixXY']['tau_eff_pct'],
                    },
                    'ood': ood_m,
                }
                print(
                    f'[Stage-I {mname} | {cname}] ({ft_sec:.2f}s) '
                    f'dev_dNLL={dnll_dev:+.4f} val_BA4={val_m["slices"]["All"]["BA4_pct"]:.2f}% '
                    f'val_NLL={val_m["slices"]["All"]["NLL"]:.4f} '
                    f'val_ws1_All={val_m["within_slice_1pct_fpr"]["All"]["tau_eff_pct"]:.2f}% '
                    f'val_ws1_D1={val_m["within_slice_1pct_fpr"]["D1_obs"]["tau_eff_pct"]:.2f}% '
                    f'W184={ood_m["gen_ood_w184_epps21_nc_charm"]["frac_p_tau_gt_0p5_pct"]:.2f}%',
                    flush=True,
                )

            # Restore baseline head before next model
            model.event_head_7.load_state_dict(base_head_state)
            results_models[mname] = {
                'family': mdata['family'],
                'ckpt_path': mdata['ckpt_path'],
                'max_cache_equiv_diff': mdata['max_cache_equiv_diff'],
                'truth_invariance': mdata['truth_invariance'],
                'runs': m_runs,
                'val_probs': val_probs_by_cand,
                'dev_probs': dev_probs_by_cand,
            }

        # Select locked Stage-I DRO recipe strictly on internal_dev_screen (anchored Score-Bank=8192 only)
        anchored_bank_cands = [cn for cn, cfg in candidates if cfg.lambda_dro > 0 and cfg.lambda_nll > 0 and cfg.use_bank]
        best_cand = None
        best_score = -1e9
        for cn in anchored_bank_cands:
            rows = dev_selection_scores[cn]
            ok_all = all(r['passes_sec10_2'] > 0.5 for r in rows)
            score_mean = float(np.mean([r['ws1_all'] + 10.0 * r['pauc1_all'] for r in rows]))
            if ok_all and score_mean > best_score:
                best_score = score_mean
                best_cand = cn
        if best_cand is None:
            # Fallback to lowest dev dNLL among anchored Score-Bank candidates and flag honestly
            best_cand = min(anchored_bank_cands, key=lambda cn: float(np.mean([r['dnll_dev'] for r in dev_selection_scores[cn]])))
            stage1_fallback_flag = True
        else:
            stage1_fallback_flag = False

        # Stage-I 50/50 Blend comparison & 1,000x paired family bootstrap (locked_dro vs pretrained)
        boot_s1: dict[str, Any] = {}
        for mname, mres in results_models.items():
            vp = mres['val_probs']
            boot_s1[mname] = {
                f'{best_cand}_vs_pretrained': {
                    sn: _paired_bootstrap(vp[best_cand], vp['pretrained_baseline'], y_val, fam_val, val_slices[sn])
                    for sn in ('All', 'D1_obs', 'Pinpoint_PixXY')
                },
                f'{best_cand}_vs_pure_ce_3p': {
                    sn: _paired_bootstrap(vp[best_cand], vp['pure_ce_3p'], y_val, fam_val, val_slices[sn])
                    for sn in ('All', 'D1_obs', 'Pinpoint_PixXY')
                },
            }

        blend_s1 = {}
        if 'tracklet_ipformer' in results_models and 'dual_regime_modalnet' in results_models:
            p_pre_b = 0.5 * results_models['tracklet_ipformer']['val_probs']['pretrained_baseline'] + 0.5 * results_models['dual_regime_modalnet']['val_probs']['pretrained_baseline']
            p_dro_b = 0.5 * results_models['tracklet_ipformer']['val_probs'][best_cand] + 0.5 * results_models['dual_regime_modalnet']['val_probs'][best_cand]
            blend_s1 = {
                'pretrained_50_50': _evaluate_predictions_suite(p_pre_b, y_val, val_keys, val_slices),
                f'{best_cand}_50_50': _evaluate_predictions_suite(p_dro_b, y_val, val_keys, val_slices),
                'bootstrap_dro_vs_pretrained': {
                    sn: _paired_bootstrap(p_dro_b, p_pre_b, y_val, fam_val, val_slices[sn])
                    for sn in ('All', 'D1_obs', 'Pinpoint_PixXY')
                },
            }

        locked_cfg = {cn: cfg for cn, cfg in candidates}[best_cand].__dict__
        for mres in results_models.values():
            mres.pop('val_probs', None)
            mres.pop('dev_probs', None)

        summary = {
            'stage': 'Stage-I',
            'UTC_finished': datetime.now(timezone.utc).isoformat(),
            'wall_seconds': float(time.perf_counter() - t0),
            'seal_audit_violations': list(guard.violations),
            'tune8k_or_sealed_touched': False,
            'selection_split': 'internal_dev_screen (N=1,600)',
            'eval_split': 'internal_val_eval (N=4,800)',
            'locked_candidate_name': best_cand,
            'locked_config': locked_cfg,
            'stage1_calibration_fallback_triggered': stage1_fallback_flag,
            'dev_selection_table': dev_selection_scores,
            'models': results_models,
            'blend_50_50': blend_s1,
            'paired_family_bootstrap_seed20263250': boot_s1,
        }
        write_json(s1_dir / 'stage1_tail_dro_summary.json', summary)
        print(f'[Stage-I COMPLETE] Locked={best_cand} fallback={stage1_fallback_flag} in {summary["wall_seconds"]:.1f}s -> {s1_dir / "stage1_tail_dro_summary.json"}', flush=True)
        return summary


# ---------------------------------------------------------------------------
# Stage-II Frozen Confirmation (Train32k -> Tune6k + 6 OOD + 7 Shard-080 Stress)
# ---------------------------------------------------------------------------
def _extract_stage2_items(
    geom: dict,
    bounds: dict,
    families: tuple[str, ...],
) -> dict[str, Any]:
    """Extract Train32k, Tune8k, 6 OOD, and 7 Shard-080 Stress items (reusing sharded caches if present)."""
    import pickle
    tune_meta = pd.read_parquet(assert_unsealed_path(CLEAN_V3_ROOT / 'validation_meta/tune_truth_meta.parquet'))
    scr_df = pd.read_parquet(assert_unsealed_path(CLEAN_V3_ROOT / 'data/tune_screen.parquet'), columns=KEY)
    ref_idx = pd.MultiIndex.from_frame(tune_meta[KEY])
    in_screen = np.asarray(ref_idx.isin(pd.MultiIndex.from_frame(scr_df[KEY])), dtype=bool)
    tune6k_m = ~in_screen

    t0 = time.perf_counter()
    cdir1 = _stage1_cache_dir()
    cdir2 = _stage2_cache_dir()
    tr_part_files = sorted(cdir1.glob('train_part_*_of_*.pkl'))
    va_part_files = sorted(cdir2.glob('val_part_*_of_*.pkl'))
    os_file = cdir2 / 'ood_stress_s2.pkl'
    can_reuse = (
        len(tr_part_files) > 0 and len(va_part_files) > 0 and os_file.is_file()
        and all(f in ('tracklet_ipformer', 'dual_regime_modalnet') for f in families)
    )

    if can_reuse:
        tr_parts = []
        for pf in tr_part_files:
            with pf.open('rb') as fh:
                tr_parts.append(_pickle_load(fh))
        va_parts = []
        for pf in va_part_files:
            with pf.open('rb') as fh:
                va_parts.append(_pickle_load(fh))
        with os_file.open('rb') as fh:
            os_pack = _pickle_load(fh)
        ood_res = os_pack['ood_res']
        st_res = os_pack['st_res']
        y_080 = os_pack['y_080']
    else:
        tr_shards = load_authorized_split_shards(CLEAN_V3_ROOT / 'data', 'train', stage=2)
        va_shards = load_authorized_split_shards(CLEAN_V3_ROOT / 'data', 'validation', stage=2)
        sh80 = va_shards[0]
        obs_080 = pq.read_table(assert_unsealed_path(sh80 / 'observations.parquet')).to_pylist()
        prov_080 = pd.read_parquet(assert_unsealed_path(sh80 / 'provenance.parquet'), columns=KEY + FAMILY + ['label'])
        y_080 = prov_080['label'].to_numpy(np.int64)

        n_workers = min(8, max(1, os.cpu_count() or 1))
        shm_base = _shm_scratch_root()

        with tempfile.TemporaryDirectory(prefix='tdro-s2-raw-', dir=shm_base) as tmp_d:
            tmp_p = Path(tmp_d) / 'shards'
            tmp_p.mkdir(parents=True, exist_ok=True)

            def _copy_tr(sp: Path) -> Path:
                dst = tmp_p / 'tr' / sp.name
                dst.mkdir(parents=True, exist_ok=True)
                for fname in ('observations.parquet', 'targets.parquet', 'provenance.parquet'):
                    shutil.copyfile(sp / fname, dst / fname)
                return dst

            def _copy_va(sp: Path) -> Path:
                dst = tmp_p / 'va' / sp.name
                dst.mkdir(parents=True, exist_ok=True)
                for fname in ('observations.parquet', 'provenance.parquet'):
                    shutil.copyfile(sp / fname, dst / fname)
                return dst

            with ThreadPoolExecutor(max_workers=min(8, n_workers)) as tpool:
                loc_tr = list(tpool.map(_copy_tr, tr_shards))
                loc_va = list(tpool.map(_copy_va, va_shards))

            tr_tasks = [(str(sp), 'train', True, geom, bounds, BASE_OBS_CFG, families) for sp in loc_tr]
            va_tasks = [(str(sp), 'validation', False, geom, bounds, BASE_OBS_CFG, families) for sp in loc_va]
            ood_tasks = [(oname, str(oroot), geom, bounds, BASE_OBS_CFG, families) for oname, oroot in OOD_SPECS]
            st_tasks = [(st_name, pkw, c_over, obs_080, geom, bounds, families) for st_name, pkw, c_over in STRESS_SPECS]

            with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context('spawn')) as ppool:
                tr_parts = list(ppool.map(_extract_shard_worker, tr_tasks))
                va_parts = list(ppool.map(_extract_shard_worker, va_tasks))
                ood_res = list(ppool.map(_extract_ood_worker, ood_tasks))
                st_res = list(ppool.map(_extract_stress_worker, st_tasks))

    # Deterministically sort Train32k by (run, chunk, row) & join event_sidecar
    tr_rows = [r for part in tr_parts for r in part]
    tr_rows.sort(key=lambda x: x[0])
    tr_keys = pd.DataFrame({'run': [r[0][0] for r in tr_rows], 'chunk': [r[0][1] for r in tr_rows], 'row': [r[0][2] for r in tr_rows]})
    y_tr = np.array([r[1] for r in tr_rows], dtype=np.int64)

    sc_df = pd.read_parquet(assert_unsealed_path(CLEAN_V3_ROOT / 'truth_sidecar/event_sidecar.parquet'))
    pos_sc = pd.MultiIndex.from_frame(sc_df[KEY]).get_indexer(pd.MultiIndex.from_frame(tr_keys[KEY]))
    if (pos_sc < 0).any():
        raise ValueError('Train32k keys missing from event_sidecar.parquet')
    tr_sidecar = sc_df.iloc[pos_sc].reset_index(drop=True)
    del sc_df

    va_map: dict[tuple[int, int, int], dict[str, dict]] = {}
    for part in va_parts:
        for k, _lbl, fdict in part:
            va_map[k] = fdict

    tr_by_fam = {fam: [r[2][fam] for r in tr_rows] for fam in families}
    t8_by_fam = {fam: [va_map[(int(r), int(c), int(rw))][fam] for r, c, rw in tune_meta[KEY].to_numpy()] for fam in families}
    ood_by_name = {oname: (fdict, yo) for oname, fdict, yo in ood_res}
    stress_by_name = {st_name: fdict for st_name, fdict in st_res}

    print(f'[Stage-II Extract] Extracted Train32k + Tune8k + 6 OOD + 7 Stress in {time.perf_counter() - t0:.1f}s', flush=True)
    return {
        'tr_by_fam': tr_by_fam,
        'y_tr': y_tr,
        'tr_sidecar': tr_sidecar,
        't8_by_fam': t8_by_fam,
        'tune_meta': tune_meta,
        'in_screen': in_screen,
        'tune6k_m': tune6k_m,
        'ood_by_name': ood_by_name,
        'stress_by_name': stress_by_name,
        'y_080': y_080,
    }


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Unpickler mapping NumPy 2.x `numpy._core` module paths onto NumPy 1.x `numpy.core` when needed."""

    def find_class(self, module: str, name: str) -> Any:
        if module.startswith('numpy._core'):
            try:
                return super().find_class(module, name)
            except ModuleNotFoundError:
                module = 'numpy.core' + module[len('numpy._core'):]
        return super().find_class(module, name)


def _pickle_load(fh: Any) -> Any:
    return _NumpyCompatUnpickler(fh).load()


def _load_stage2_items_for_family(
    fam: str,
    geom: dict,
    bounds: dict,
) -> dict[str, Any]:
    """Load Train32k, Tune8k, 6 OOD, and 7 Stress items for ONE family (`< 4.5 GB` RAM)."""
    import pickle
    tune_meta = pd.read_parquet(assert_unsealed_path(CLEAN_V3_ROOT / 'validation_meta/tune_truth_meta.parquet'))
    scr_df = pd.read_parquet(assert_unsealed_path(CLEAN_V3_ROOT / 'data/tune_screen.parquet'), columns=KEY)
    ref_idx = pd.MultiIndex.from_frame(tune_meta[KEY])
    in_screen = np.asarray(ref_idx.isin(pd.MultiIndex.from_frame(scr_df[KEY])), dtype=bool)
    tune6k_m = ~in_screen

    cdir1 = _stage1_cache_dir()
    cdir2 = _stage2_cache_dir()
    tr_part_files = sorted(cdir1.glob('train_part_*_of_*.pkl'))
    va_part_files = sorted(cdir2.glob('val_part_*_of_*.pkl'))
    os_file = cdir2 / 'ood_stress_s2.pkl'
    if not (tr_part_files and va_part_files and os_file.is_file()):
        return _extract_stage2_items(geom, bounds, (fam,))

    tr_rows: list[tuple[tuple[int, int, int], int, dict]] = []
    for pf in tr_part_files:
        with pf.open('rb') as fh:
            rows = _pickle_load(fh)
        for k, lbl, fdict in rows:
            tr_rows.append((k, lbl, fdict[fam]))
        del rows
    tr_rows.sort(key=lambda x: x[0])
    tr_keys = pd.DataFrame({'run': [r[0][0] for r in tr_rows], 'chunk': [r[0][1] for r in tr_rows], 'row': [r[0][2] for r in tr_rows]})
    y_tr = np.array([r[1] for r in tr_rows], dtype=np.int64)
    tr_items = [r[2] for r in tr_rows]
    del tr_rows

    sc_df = pd.read_parquet(assert_unsealed_path(CLEAN_V3_ROOT / 'truth_sidecar/event_sidecar.parquet'))
    pos_sc = pd.MultiIndex.from_frame(sc_df[KEY]).get_indexer(pd.MultiIndex.from_frame(tr_keys[KEY]))
    if (pos_sc < 0).any():
        raise ValueError('Train32k keys missing from event_sidecar.parquet')
    tr_sidecar = sc_df.iloc[pos_sc].reset_index(drop=True)
    del sc_df

    va_map: dict[tuple[int, int, int], dict] = {}
    for pf in va_part_files:
        with pf.open('rb') as fh:
            rows = _pickle_load(fh)
        for k, _lbl, fdict in rows:
            va_map[k] = fdict[fam]
        del rows
    t8_items = [va_map[(int(r), int(c), int(rw))] for r, c, rw in tune_meta[KEY].to_numpy()]
    del va_map

    with os_file.open('rb') as fh:
        os_pack = _pickle_load(fh)
    ood_by_name = {oname: ({fam: fdict[fam]}, yo) for oname, fdict, yo in os_pack['ood_res']}
    stress_by_name = {st_name: {fam: fdict[fam]} for st_name, fdict in os_pack['st_res']}
    y_080 = os_pack['y_080']
    del os_pack

    return {
        'tr_by_fam': {fam: tr_items},
        'y_tr': y_tr,
        'tr_sidecar': tr_sidecar,
        't8_by_fam': {fam: t8_items},
        'tune_meta': tune_meta,
        'in_screen': in_screen,
        'tune6k_m': tune6k_m,
        'ood_by_name': ood_by_name,
        'stress_by_name': stress_by_name,
        'y_080': y_080,
    }


def run_stage2(locked_cfg_path: Path | None = None, models_filter: list[str] | None = None) -> dict[str, Any]:
    """Run Stage-II frozen confirmation with locked Stage-I DRO hyperparameters (zero tuning)."""
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    device = torch.device('cuda:0')
    t0 = time.perf_counter()

    s1_summary_p = locked_cfg_path or (OUT_ROOT / 'stage1/stage1_tail_dro_summary.json')
    if not s1_summary_p.is_file():
        raise FileNotFoundError(f'Missing Stage-I summary at {s1_summary_p}; run --mode stage1-run first')
    s1_doc = json.loads(s1_summary_p.read_text())
    locked_cfg = TailDROConfig(**s1_doc['locked_config'])
    locked_name = str(s1_doc['locked_candidate_name'])

    with SealAuditGuard() as guard:
        for sp in ('train', 'validation', 'tune_screen', 'tune6k', 'ood'):
            assert_unsealed_split(sp, stage=2)

        geom = json.loads(assert_unsealed_path(CLEAN_V3_ROOT / 'geometry.json').read_text())
        bounds = detector_bounds(geom)
        bounds['_plane_x0'] = np.array([cumulative_radiation_length(float(z), geom) for z in bounds['pixel_Zpos']], dtype=np.float32)

        all_specs = _default_stage2_checkpoints()
        if models_filter:
            all_specs = {k: v for k, v in all_specs.items() if k in models_filter}

        tune_meta = pd.read_parquet(assert_unsealed_path(CLEAN_V3_ROOT / 'validation_meta/tune_truth_meta.parquet'))
        scr_df = pd.read_parquet(assert_unsealed_path(CLEAN_V3_ROOT / 'data/tune_screen.parquet'), columns=KEY)
        ref_idx = pd.MultiIndex.from_frame(tune_meta[KEY])
        in_screen = np.asarray(ref_idx.isin(pd.MultiIndex.from_frame(scr_df[KEY])), dtype=bool)
        tune6k_m = ~in_screen
        y_t8 = tune_meta['label'].to_numpy(np.int64)
        obs_t8 = tune_meta['obs_domain_id'].to_numpy(np.int64)
        pc_t8 = tune_meta['pixel_centered'].to_numpy(bool)
        fp_t8 = tune_meta['front_pinpoint_pixel_centered'].to_numpy(bool)
        interm_t8 = pc_t8 & (~fp_t8)
        fort_t8 = tune_meta['fortune_module'].to_numpy(bool)
        fam_t8 = pd.MultiIndex.from_frame(tune_meta[FAMILY]).factorize()[0]

        t6k_slices = {
            'All_Tune6k': tune6k_m,
            'D1_obs': tune6k_m & (obs_t8 == 1),
            'D2_obs': tune6k_m & (obs_t8 == 2),
            'D3_obs': tune6k_m & (obs_t8 == 3),
            'Pinpoint_PixXY': tune6k_m & pc_t8,
            'Front30': tune6k_m & fp_t8,
            'Interm150': tune6k_m & interm_t8,
            'Fortune_ScintOnly': tune6k_m & fort_t8,
        }

        s2_dir = OUT_ROOT / 'stage2'
        s2_dir.mkdir(parents=True, exist_ok=True)

        models_out: dict[str, Any] = {}
        ens_store: dict[str, dict[str, Any]] = {}
        ood_labels_map: dict[str, np.ndarray] = {}
        y_080_ref: np.ndarray | None = None

        for mname, (fam, seed_ckpts) in all_specs.items():
            data = _load_stage2_items_for_family(fam, geom, bounds)
            y_080_ref = data['y_080']
            for oname in data['ood_by_name']:
                ood_labels_map[oname] = data['ood_by_name'][oname][1]
            pre_seed_t8: dict[int, np.ndarray] = {}
            dro_seed_t8: dict[int, np.ndarray] = {}
            pre_ood_s: dict[str, list[np.ndarray]] = {oname: [] for oname, _ in OOD_SPECS}
            dro_ood_s: dict[str, list[np.ndarray]] = {oname: [] for oname, _ in OOD_SPECS}
            pre_st_s: dict[str, list[np.ndarray]] = {st_name: [] for st_name, _, _ in STRESS_SPECS}
            dro_st_s: dict[str, list[np.ndarray]] = {st_name: [] for st_name, _, _ in STRESS_SPECS}
            seed_audit: dict[int, Any] = {}

            for seed in STAGE2_SEEDS:
                ckpt_p = seed_ckpts[seed]
                model, cfg = _load_model_checkpoint(fam, ckpt_p, geom, device)
                _, collate_fn = _get_item_and_collate_fns(fam, bounds, cfg)

                tr_c = extract_penultimate_cache(
                    model,
                    _batches_from_items(data['tr_by_fam'][fam], collate_fn, batch_size=64, sidecar_df=data['tr_sidecar'], labels_override=data['y_tr']),
                    device,
                )
                t8_c = extract_penultimate_cache(
                    model,
                    _batches_from_items(data['t8_by_fam'][fam], collate_fn, batch_size=64, labels_override=y_t8),
                    device,
                )
                ood_c = {
                    oname: extract_penultimate_cache(
                        model,
                        _batches_from_items(data['ood_by_name'][oname][0][fam], collate_fn, batch_size=64),
                        device,
                    )
                    for oname, _ in OOD_SPECS
                }
                st_c = {
                    st_name: extract_penultimate_cache(
                        model,
                        _batches_from_items(data['stress_by_name'][st_name][fam], collate_fn, batch_size=64),
                        device,
                    )
                    for st_name, _, _ in STRESS_SPECS
                }

                pre_seed_t8[seed] = t8_c['probs4_ref'].numpy().astype(np.float64)
                for oname in ood_c:
                    pre_ood_s[oname].append(ood_c[oname]['probs4_ref'].numpy().astype(np.float64))
                for st_name in st_c:
                    pre_st_s[st_name].append(st_c[st_name]['probs4_ref'].numpy().astype(np.float64))

                # Freeze backbone & fine-tune ONLY event_head_7 with locked Stage-I config (seed-matched)
                snap = snapshot_backbone_weights(model, head_attr='event_head_7')
                head = freeze_backbone_for_tail_finetuning(model, head_attr='event_head_7')
                seed_cfg = TailDROConfig(**{**locked_cfg.__dict__, 'seed': int(seed)})
                t_ft0 = time.perf_counter()
                _, hist = finetune_head_on_cached_latents(head, tr_c, seed_cfg, device)
                ft_sec = float(time.perf_counter() - t_ft0)
                id_res = verify_backbone_bit_identity(model, snap, head_attr='event_head_7')

                # 39-key truth invariance on fine-tuned model
                sb = next(_batches_from_items(data['t8_by_fam'][fam][:8], collate_fn, batch_size=8))
                sb_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in sb.items()}
                inv_res = verify_inference_truth_invariance(model, sb_dev, atol=1e-7)

                dro_seed_t8[seed] = predict_from_cached_latents(head, t8_c['penultimate'], device)['probs4'].astype(np.float64)
                for oname in ood_c:
                    dro_ood_s[oname].append(predict_from_cached_latents(head, ood_c[oname]['penultimate'], device)['probs4'].astype(np.float64))
                for st_name in st_c:
                    dro_st_s[st_name].append(predict_from_cached_latents(head, st_c[st_name]['penultimate'], device)['probs4'].astype(np.float64))

                max_eq = float(max(
                    tr_c['max_verify_diff'], t8_c['max_verify_diff'],
                    max(c['max_verify_diff'] for c in ood_c.values()),
                    max(c['max_verify_diff'] for c in st_c.values()),
                ))
                seed_audit[seed] = {
                    'ckpt_path': str(ckpt_p),
                    'finetune_seconds': ft_sec,
                    'history': hist,
                    'backbone_bit_identity': id_res['passed'],
                    'truth_invariance': inv_res,
                    'max_cache_equiv_diff': max_eq,
                }
                print(
                    f'[Stage-II {mname} s{seed}] ft={ft_sec:.2f}s max_eq={max_eq:.2e} '
                    f'pre_BA4={_ba4_nll_ece(y_t8[tune6k_m], pre_seed_t8[seed][tune6k_m])["BA4_pct"]:.2f}% '
                    f'dro_BA4={_ba4_nll_ece(y_t8[tune6k_m], dro_seed_t8[seed][tune6k_m])["BA4_pct"]:.2f}%',
                    flush=True,
                )

            pre_ens_raw = np.mean([pre_seed_t8[s] for s in STAGE2_SEEDS], axis=0)
            dro_ens_raw = np.mean([dro_seed_t8[s] for s in STAGE2_SEEDS], axis=0)
            pre_ens_cal = _screen_domain_calibrate(pre_seed_t8, y_t8, in_screen, obs_t8)
            dro_ens_cal = _screen_domain_calibrate(dro_seed_t8, y_t8, in_screen, obs_t8)

            np.save(s2_dir / f'{mname}_pre_tune8k_ens3_raw.npy', pre_ens_raw.astype(np.float32))
            np.save(s2_dir / f'{mname}_dro_tune8k_ens3_raw.npy', dro_ens_raw.astype(np.float32))
            np.save(s2_dir / f'{mname}_dro_tune8k_ens3_screen_cal.npy', dro_ens_cal.astype(np.float32))

            ens_store[mname] = {
                'pre_raw': pre_ens_raw, 'dro_raw': dro_ens_raw,
                'pre_cal': pre_ens_cal, 'dro_cal': dro_ens_cal,
                'pre_ood': {oname: np.mean(pre_ood_s[oname], axis=0) for oname, _ in OOD_SPECS},
                'dro_ood': {oname: np.mean(dro_ood_s[oname], axis=0) for oname, _ in OOD_SPECS},
                'pre_stress': {sk: np.mean(pre_st_s[sk], axis=0) for sk, _, _ in STRESS_SPECS},
                'dro_stress': {sk: np.mean(dro_st_s[sk], axis=0) for sk, _, _ in STRESS_SPECS},
            }

            pre_raw_m = _evaluate_predictions_suite(pre_ens_raw, y_t8, tune_meta, t6k_slices)
            dro_raw_m = _evaluate_predictions_suite(dro_ens_raw, y_t8, tune_meta, t6k_slices)
            pre_cal_m = _evaluate_predictions_suite(pre_ens_cal, y_t8, tune_meta, t6k_slices)
            dro_cal_m = _evaluate_predictions_suite(dro_ens_cal, y_t8, tune_meta, t6k_slices)

            # OOD & Shard-080 Stress metrics
            ood_eval = {}
            for oname, _ in OOD_SPECS:
                yo = ood_labels_map[oname]
                po_pre = ens_store[mname]['pre_ood'][oname]
                po_dro = ens_store[mname]['dro_ood'][oname]
                if len(np.unique(yo)) >= 4:
                    ood_eval[oname] = {'pretrained': _ba4_nll_ece(yo, po_pre), 'dro': _ba4_nll_ece(yo, po_dro)}
                else:
                    ood_eval[oname] = {
                        'pretrained': {
                            'n_events': int(len(yo)),
                            'p_tau_gt_0p5_pct': float((po_pre[:, 2] > 0.5).mean() * 100.0),
                            'argmax_tau_pct': float((po_pre.argmax(1) == 2).mean() * 100.0),
                        },
                        'dro': {
                            'n_events': int(len(yo)),
                            'p_tau_gt_0p5_pct': float((po_dro[:, 2] > 0.5).mean() * 100.0),
                            'argmax_tau_pct': float((po_dro.argmax(1) == 2).mean() * 100.0),
                        },
                    }

            assert y_080_ref is not None
            stress_eval = {}
            nom_pre = _ba4_nll_ece(y_080_ref, ens_store[mname]['pre_stress']['nominal'])['BA4_pct']
            nom_dro = _ba4_nll_ece(y_080_ref, ens_store[mname]['dro_stress']['nominal'])['BA4_pct']
            for st_name, _, _ in STRESS_SPECS:
                mp_pre = _ba4_nll_ece(y_080_ref, ens_store[mname]['pre_stress'][st_name])
                mp_dro = _ba4_nll_ece(y_080_ref, ens_store[mname]['dro_stress'][st_name])
                stress_eval[st_name] = {
                    'pretrained': {**mp_pre, 'delta_BA4_vs_nom_pp': mp_pre['BA4_pct'] - nom_pre},
                    'dro': {**mp_dro, 'delta_BA4_vs_nom_pp': mp_dro['BA4_pct'] - nom_dro},
                }

            del data

            boot_raw = {
                sn: _paired_bootstrap(dro_ens_raw, pre_ens_raw, y_t8, fam_t8, t6k_slices[sn])
                for sn in ('All_Tune6k', 'D1_obs', 'Pinpoint_PixXY', 'Front30')
            }
            boot_cal = {
                sn: _paired_bootstrap(dro_ens_cal, pre_ens_cal, y_t8, fam_t8, t6k_slices[sn])
                for sn in ('All_Tune6k', 'D1_obs', 'Pinpoint_PixXY')
            }

            dnll_raw = dro_raw_m['slices']['All_Tune6k']['NLL'] - pre_raw_m['slices']['All_Tune6k']['NLL']
            ece_r_raw = dro_raw_m['slices']['All_Tune6k']['ECE_top1'] / max(pre_raw_m['slices']['All_Tune6k']['ECE_top1'], 1e-6)
            ece_t_raw = dro_raw_m['slices']['All_Tune6k']['ECE_tau'] / max(pre_raw_m['slices']['All_Tune6k']['ECE_tau'], 1e-6)

            models_out[mname] = {
                'seed_audit': seed_audit,
                'tune6k_raw': {'pretrained': pre_raw_m, 'dro': dro_raw_m},
                'tune6k_screen_cal': {'pretrained': pre_cal_m, 'dro': dro_cal_m},
                'calibration_check_raw': {
                    'delta_NLL': dnll_raw,
                    'ECE_top1_ratio': ece_r_raw,
                    'ECE_tau_ratio': ece_t_raw,
                    'fallback_triggered': bool(dnll_raw > 0.01 or ece_r_raw >= 2.0 or ece_t_raw >= 2.0),
                },
                'bootstrap_dro_vs_pretrained_raw': boot_raw,
                'bootstrap_dro_vs_pretrained_screen_cal': boot_cal,
                'ood_6shards': ood_eval,
                'stress_shard080': stress_eval,
            }

        # 50/50 Blend (`TrackletIPFormer + DualRegimeModalNet`)
        blend_out = {}
        if 'tracklet_ipformer' in ens_store and 'dual_regime_modalnet' in ens_store:
            t_st, d_st = ens_store['tracklet_ipformer'], ens_store['dual_regime_modalnet']
            b_pre_raw = 0.5 * t_st['pre_raw'] + 0.5 * d_st['pre_raw']
            b_dro_raw = 0.5 * t_st['dro_raw'] + 0.5 * d_st['dro_raw']
            b_pre_cal = 0.5 * t_st['pre_cal'] + 0.5 * d_st['pre_cal']
            b_dro_cal = 0.5 * t_st['dro_cal'] + 0.5 * d_st['dro_cal']

            np.save(s2_dir / 'blend_50_50_dro_tune8k_raw.npy', b_dro_raw.astype(np.float32))
            np.save(s2_dir / 'blend_50_50_dro_tune8k_screen_cal.npy', b_dro_cal.astype(np.float32))

            blend_ood = {}
            for oname, _ in OOD_SPECS:
                yo = ood_labels_map[oname]
                po_pre = 0.5 * t_st['pre_ood'][oname] + 0.5 * d_st['pre_ood'][oname]
                po_dro = 0.5 * t_st['dro_ood'][oname] + 0.5 * d_st['dro_ood'][oname]
                if len(np.unique(yo)) >= 4:
                    blend_ood[oname] = {'pretrained': _ba4_nll_ece(yo, po_pre), 'dro': _ba4_nll_ece(yo, po_dro)}
                else:
                    blend_ood[oname] = {
                        'pretrained': {'p_tau_gt_0p5_pct': float((po_pre[:, 2] > 0.5).mean() * 100.0), 'argmax_tau_pct': float((po_pre.argmax(1) == 2).mean() * 100.0)},
                        'dro': {'p_tau_gt_0p5_pct': float((po_dro[:, 2] > 0.5).mean() * 100.0), 'argmax_tau_pct': float((po_dro.argmax(1) == 2).mean() * 100.0)},
                    }

            blend_out = {
                'tune6k_raw': {
                    'pretrained': _evaluate_predictions_suite(b_pre_raw, y_t8, tune_meta, t6k_slices),
                    'dro': _evaluate_predictions_suite(b_dro_raw, y_t8, tune_meta, t6k_slices),
                },
                'tune6k_screen_cal': {
                    'pretrained': _evaluate_predictions_suite(b_pre_cal, y_t8, tune_meta, t6k_slices),
                    'dro': _evaluate_predictions_suite(b_dro_cal, y_t8, tune_meta, t6k_slices),
                },
                'bootstrap_raw': {
                    sn: _paired_bootstrap(b_dro_raw, b_pre_raw, y_t8, fam_t8, t6k_slices[sn])
                    for sn in ('All_Tune6k', 'D1_obs', 'Pinpoint_PixXY', 'Front30')
                },
                'bootstrap_screen_cal': {
                    sn: _paired_bootstrap(b_dro_cal, b_pre_cal, y_t8, fam_t8, t6k_slices[sn])
                    for sn in ('All_Tune6k', 'D1_obs', 'Pinpoint_PixXY')
                },
                'ood_6shards': blend_ood,
            }

        summary = {
            'stage': 'Stage-II',
            'UTC_finished': datetime.now(timezone.utc).isoformat(),
            'wall_seconds': float(time.perf_counter() - t0),
            'seal_audit_violations': list(guard.violations),
            'sealed_shards_accessed': False,
            'locked_candidate_name': locked_name,
            'locked_config': locked_cfg.__dict__,
            'models': models_out,
            'blend_50_50': blend_out,
        }
        write_json(s2_dir / 'stage2_tail_dro_summary.json', summary)
        print(f'[Stage-II COMPLETE] in {summary["wall_seconds"]:.1f}s -> {s2_dir / "stage2_tail_dro_summary.json"}', flush=True)
        return summary


class _UnifiedHeadWithNCCC(nn.Module):
    """Differentiable Stage-I wrapper around `PinpointUnifiedNet.event_head_7` preserving frozen `nccc_boost`."""

    def __init__(self, event_head_7: nn.Module) -> None:
        super().__init__()
        self.event_head_7 = event_head_7

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        fused_global = packed[:, :480]
        logits7_raw = self.event_head_7(fused_global)
        if packed.shape[1] >= 482:
            nccc_active = packed[:, 480] > 0.5
            if nccc_active.any():
                nccc_boost = packed[:, 481]
                ell_nccc = torch.maximum(logits7_raw[:, 6], logits7_raw[:, 5] - 1.45) + nccc_boost
                logits7 = logits7_raw.clone()
                logits7[:, 6] = torch.where(nccc_active, torch.logaddexp(logits7_raw[:, 6], ell_nccc), logits7_raw[:, 6])
                return logits7
        return logits7_raw


def run_stage1_unified_cache(locked_cfg_path: Path | None = None) -> dict[str, Any]:
    """Apply locked Stage-I Tail-DRO (`dro_bal_l008_kl15_bank8192`) to PinpointUnifiedNet NPZ latent caches."""
    t0 = time.perf_counter()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    s1_summary_p = locked_cfg_path or (OUT_ROOT / 'stage1/stage1_tail_dro_summary.json')
    if not s1_summary_p.is_file():
        raise FileNotFoundError(f'Missing Stage-I summary at {s1_summary_p}')
    s1_doc = json.loads(assert_unsealed_path(s1_summary_p).read_text())
    locked_cfg = TailDROConfig(**s1_doc['locked_config'])
    locked_name = str(s1_doc['locked_candidate_name'])

    unified_runs = [
        ('s1_unified_pinpoint_net_r1', NEXTGEN_ROOT / 'exp_unified_net/round1_archive/s1_unified_pinpoint_net'),
        ('s1_unified_pinpoint_net_r2', NEXTGEN_ROOT / 'exp_unified_net/s1_unified_pinpoint_net'),
    ]

    with install_stage1_seal_guard() as guard:
        for sp in ('internal_train', 'internal_screen', 'internal_val'):
            assert_unsealed_split(sp, stage=1)

        out_models: dict[str, Any] = {}
        for rname, rdir in unified_runs:
            ckpt_p = assert_unsealed_path(rdir / 'best_ema.pt')
            tr_npz = np.load(assert_unsealed_path(rdir / 'latent_cache_internal_train.npz'))
            dev_npz = np.load(assert_unsealed_path(rdir / 'latent_cache_internal_dev_screen.npz'))
            val_npz = np.load(assert_unsealed_path(rdir / 'latent_cache_internal_val_eval.npz'))

            def _pack_npz(npz_obj: Any, include_train_targets: bool = False) -> dict[str, torch.Tensor]:
                fg = torch.from_numpy(np.asarray(npz_obj['fused_global'], dtype=np.float32))
                n = fg.shape[0]
                if 'nccc_active' in npz_obj and 'nccc_boost' in npz_obj:
                    act = torch.from_numpy(np.asarray(npz_obj['nccc_active'], dtype=np.float32)).unsqueeze(1)
                    bst = torch.from_numpy(np.asarray(npz_obj['nccc_boost'], dtype=np.float32)).unsqueeze(1)
                else:
                    act = torch.zeros((n, 1), dtype=torch.float32)
                    bst = torch.zeros((n, 1), dtype=torch.float32)
                out: dict[str, torch.Tensor] = {
                    'penultimate': torch.cat([fg, act, bst], dim=1),
                    'logits7_ref': torch.from_numpy(np.asarray(npz_obj['logits7'], dtype=np.float32)),
                    'probs4_ref': torch.from_numpy(np.asarray(npz_obj['probs4'], dtype=np.float32)),
                    'label': torch.from_numpy(np.asarray(npz_obj['label'], dtype=np.int64)),
                }
                if include_train_targets and 'label_7' in npz_obj:
                    out['label_7'] = torch.from_numpy(np.asarray(npz_obj['label_7'], dtype=np.int64))
                return out

            tr_cache = _pack_npz(tr_npz, include_train_targets=True)
            dev_cache = _pack_npz(dev_npz)
            val_cache = _pack_npz(val_npz)

            ckpt = torch.load(ckpt_p, map_location='cpu', weights_only=False)
            sd = ckpt['model']
            non_head_snap = {k: v.detach().clone() for k, v in sd.items() if not k.startswith('event_head_7.')}
            event_head_7 = nn.Sequential(nn.LayerNorm(480), nn.Linear(480, 256), nn.SiLU(), nn.Dropout(0.1), nn.Linear(256, 7))
            event_head_7.load_state_dict({k.split('event_head_7.', 1)[1]: v for k, v in sd.items() if k.startswith('event_head_7.')})
            wrapped_head = _UnifiedHeadWithNCCC(event_head_7).to(device)

            # Check head equivalence vs NPZ logits7_ref
            dev_pre = predict_from_cached_latents(wrapped_head, dev_cache['penultimate'], device)
            val_pre = predict_from_cached_latents(wrapped_head, val_cache['penultimate'], device)
            max_eq_dev = float(np.max(np.abs(dev_pre['logits7'] - dev_cache['logits7_ref'].numpy())))
            max_eq_val = float(np.max(np.abs(val_pre['logits7'] - val_cache['logits7_ref'].numpy())))

            # Snapshot initial head weights, then fine-tune 3 passes with locked Score-Bank DRO
            base_head_bytes = {k: v.detach().cpu().clone() for k, v in wrapped_head.state_dict().items()}
            t_ft0 = time.perf_counter()
            _, hist = finetune_head_on_cached_latents(wrapped_head, tr_cache, locked_cfg, device)
            ft_sec = float(time.perf_counter() - t_ft0)

            # Verify non-head checkpoint state is strictly untouched and head weights updated cleanly
            backbone_bit_id = all(torch.equal(non_head_snap[k], sd[k]) for k in non_head_snap)
            head_updated = any(not torch.equal(base_head_bytes[k], wrapped_head.state_dict()[k].cpu()) for k in base_head_bytes)

            dev_dro = predict_from_cached_latents(wrapped_head, dev_cache['penultimate'], device)
            val_dro = predict_from_cached_latents(wrapped_head, val_cache['penultimate'], device)

            def _keys_from_npz(npz_obj: Any) -> pd.DataFrame:
                return pd.DataFrame({
                    'run': np.asarray(npz_obj['run'], dtype=np.int64),
                    'chunk': np.asarray(npz_obj['chunk'], dtype=np.int64),
                    'row': np.asarray(npz_obj['row'], dtype=np.int64),
                    'label': np.asarray(npz_obj['label'], dtype=np.int64),
                    'obs_domain_id': np.asarray(npz_obj['obs_domain_id'], dtype=np.int64),
                    'region_id': np.asarray(npz_obj['region_id'], dtype=np.int64),
                    'family_hi': np.asarray(npz_obj['family_hi'], dtype=np.int64),
                    'family_lo': np.asarray(npz_obj['family_lo'], dtype=np.int64),
                })

            dev_keys = _keys_from_npz(dev_npz)
            val_keys = _keys_from_npz(val_npz)
            y_dev = dev_keys['label'].to_numpy(np.int64)
            y_val = val_keys['label'].to_numpy(np.int64)
            obs_dev = dev_keys['obs_domain_id'].to_numpy(np.int64)
            obs_val = val_keys['obs_domain_id'].to_numpy(np.int64)
            reg_val = val_keys['region_id'].to_numpy(np.int64)
            fam_val = pd.MultiIndex.from_frame(val_keys[FAMILY]).factorize()[0]

            dev_slices = {'All': np.ones(len(y_dev), dtype=bool), 'D1_obs': obs_dev == 1}
            val_slices = {
                'All': np.ones(len(y_val), dtype=bool),
                'D1_obs': obs_val == 1,
                'D2_obs': obs_val == 2,
                'D3_obs': obs_val == 3,
                'Pinpoint_PixXY': (reg_val == 0) | (reg_val == 1),
                'Front30': reg_val == 0,
                'Interm150': reg_val == 1,
                'Fortune_ScintOnly': reg_val == 3,
            }

            pre_dev_p = dev_cache['probs4_ref'].numpy().astype(np.float64)
            pre_val_p = val_cache['probs4_ref'].numpy().astype(np.float64)
            pre_dev_m = _evaluate_predictions_suite(pre_dev_p, y_dev, dev_keys, dev_slices)
            pre_val_m = _evaluate_predictions_suite(pre_val_p, y_val, val_keys, val_slices)
            dro_dev_m = _evaluate_predictions_suite(dev_dro['probs4'], y_dev, dev_keys, dev_slices, scores=dev_dro['tau_score'])
            dro_val_m = _evaluate_predictions_suite(val_dro['probs4'], y_val, val_keys, val_slices, scores=val_dro['tau_score'])

            dnll_val = dro_val_m['slices']['All']['NLL'] - pre_val_m['slices']['All']['NLL']
            ece_top1_ratio = dro_val_m['slices']['All']['ECE_top1'] / max(pre_val_m['slices']['All']['ECE_top1'], 1e-6)
            ece_tau_ratio = dro_val_m['slices']['All']['ECE_tau'] / max(pre_val_m['slices']['All']['ECE_tau'], 1e-6)
            fallback = bool(dnll_val > 0.01 or ece_top1_ratio >= 2.0 or ece_tau_ratio >= 2.0)

            boot_val = {
                sn: _paired_bootstrap(dro_val_m_p := dro_val_m and val_dro['probs4'], pre_val_p, y_val, fam_val, val_slices[sn])
                for sn in ('All', 'D1_obs', 'Pinpoint_PixXY', 'Front30')
            }

            ood_pre_ref = {}
            ood_npz_p = rdir / 'ood_and_stress_probs.npz'
            if ood_npz_p.is_file():
                ood_npz = np.load(assert_unsealed_path(ood_npz_p))
                if 'ood_gen_ood_w184_epps21_nc_charm' in ood_npz:
                    op_w = np.asarray(ood_npz['ood_gen_ood_w184_epps21_nc_charm'], dtype=np.float64)
                    ood_pre_ref['gen_ood_w184_epps21_nc_charm'] = {
                        'n_events': int(len(op_w)),
                        'mean_p_tau_pct': float(op_w[:, 2].mean() * 100.0),
                        'frac_pred_tau_pct': float((op_w.argmax(1) == 2).mean() * 100.0),
                        'frac_p_tau_gt_0p5_pct': float((op_w[:, 2] > 0.5).mean() * 100.0),
                    }

            out_models[rname] = {
                'source_dir': str(rdir),
                'device': str(device),
                'train_events': int(tr_cache['penultimate'].shape[0]),
                'finetune_seconds': ft_sec,
                'history': hist,
                'backbone_bit_identity': bool(backbone_bit_id),
                'head_weights_updated': bool(head_updated),
                'max_head_equiv_diff_dev': max_eq_dev,
                'max_head_equiv_diff_val': max_eq_val,
                'ood_latent_cache_available': False,
                'pretrained_ood_probs_ref': ood_pre_ref,
                'pretrained': {'dev_screen': pre_dev_m, 'val_eval': pre_val_m},
                'dro_locked': {
                    'dev_screen': dro_dev_m,
                    'val_eval': dro_val_m,
                    'val_delta_vs_pretrained': {
                        'delta_BA4_pp': float(dro_val_m['slices']['All']['BA4_pct'] - pre_val_m['slices']['All']['BA4_pct']),
                        'delta_NLL': float(dnll_val),
                        'ECE_top1_ratio': float(ece_top1_ratio),
                        'ECE_tau_ratio': float(ece_tau_ratio),
                        'delta_ws1_All_pp': float(dro_val_m['within_slice_1pct_fpr']['All']['tau_eff_pct'] - pre_val_m['within_slice_1pct_fpr']['All']['tau_eff_pct']),
                        'delta_ws1_D1_pp': float(dro_val_m['within_slice_1pct_fpr']['D1_obs']['tau_eff_pct'] - pre_val_m['within_slice_1pct_fpr']['D1_obs']['tau_eff_pct']),
                        'delta_sh1_All_pp': float(dro_val_m['split_half_1pct_fpr']['All']['held_tau_eff_pct'] - pre_val_m['split_half_1pct_fpr']['All']['held_tau_eff_pct']),
                        'delta_pAUC1_All': float(dro_val_m['pAUC_1pct']['All'] - pre_val_m['pAUC_1pct']['All']),
                        'fallback_triggered': fallback,
                    },
                    'bootstrap_val_eval_vs_pretrained': boot_val,
                },
            }
            print(
                f'[Stage-I Unified {rname}] ft={ft_sec:.2f}s eq_val={max_eq_val:.2e} '
                f'BA4 {pre_val_m["slices"]["All"]["BA4_pct"]:.2f}%->{dro_val_m["slices"]["All"]["BA4_pct"]:.2f}% '
                f'dNLL={dnll_val:+.4f} WS1_All {pre_val_m["within_slice_1pct_fpr"]["All"]["tau_eff_pct"]:.2f}%->{dro_val_m["within_slice_1pct_fpr"]["All"]["tau_eff_pct"]:.2f}% '
                f'D1 {pre_val_m["within_slice_1pct_fpr"]["D1_obs"]["tau_eff_pct"]:.2f}%->{dro_val_m["within_slice_1pct_fpr"]["D1_obs"]["tau_eff_pct"]:.2f}%',
                flush=True,
            )

        summary = {
            'stage': 'Stage-I Unified Cache Tail-DRO',
            'UTC_finished': datetime.now(timezone.utc).isoformat(),
            'wall_seconds': float(time.perf_counter() - t0),
            'seal_audit_violations': list(guard.violations),
            'sealed_shards_accessed': False,
            'locked_candidate_name': locked_name,
            'locked_config': locked_cfg.__dict__,
            'models': out_models,
        }
        out_p = OUT_ROOT / 'stage1/stage1_unified_tail_dro_summary.json'
        write_json(out_p, summary)
        print(f'[Stage-I Unified Tail-DRO COMPLETE] -> {out_p}', flush=True)
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Stage 327 Frozen-Backbone Score-Bank DRO Partial-AUC Fine-Tuning')
    parser.add_argument(
        '--mode',
        required=True,
        choices=('stage1-preextract-part', 'stage1-run', 'stage1-unified-cache', 'stage2-preextract-part', 'stage2-eval'),
    )
    parser.add_argument('--part', type=int, default=0)
    parser.add_argument('--n-parts', type=int, default=16)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--models', nargs='*', default=None, help='Optional subset of model names')
    parser.add_argument('--locked-cfg', type=Path, default=None, help='Path to Stage-I summary JSON for Stage-II')
    args = parser.parse_args()

    if args.mode == 'stage1-preextract-part':
        print(json.dumps(preextract_stage1_part(args.part, n_parts=args.n_parts, workers=args.workers), indent=2))
    elif args.mode == 'stage1-run':
        run_stage1(models_filter=args.models)
    elif args.mode == 'stage1-unified-cache':
        run_stage1_unified_cache(locked_cfg_path=args.locked_cfg)
    elif args.mode == 'stage2-preextract-part':
        print(json.dumps(preextract_stage2_part(args.part, n_parts=args.n_parts, workers=args.workers), indent=2))
    elif args.mode == 'stage2-eval':
        run_stage2(locked_cfg_path=args.locked_cfg, models_filter=args.models)


if __name__ == '__main__':
    main()
