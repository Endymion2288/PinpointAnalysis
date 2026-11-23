"""Stage-I & Stage-II Runner for Experiment L1: TrackletIPFormer-GN2.

Features:
  - Zero-exception SealAuditGuard(stage=1/2) isolation (`Cal-A/Cut-B/Confirm` 100..149 = 0 accesses;
    Stage-I additionally blocks Tune8k/080..099).
  - `/dev/shm` fast cache staging (auto-cleaned on exit) + sharded HTCondor CPU pre-extraction.
  - Alignment of `l1_cluster_sidecar/` onto `internal_train` (for training-time GN2 dense
    supervision) and `internal_val_eval` (strictly post-inference for offline microscopic probe AUCs).
  - Stage-I (`25.6k x 20 passes = 4,000` steps @ `eff_bs=128`, `seed=42`, IEEE-FP32 eval on
    `internal_val_eval` 4.8k + 6 OOD + 7 Stress) + 3 single-factor ablations:
      `gn2_full`, `abl_origin_purity_only`, `abl_covertex_only`, `abl_aux_loss_only_no_readout_gate`.
  - Stage-II (`Train32k x 3 seeds {17, 42, 2026}`, `5,000` steps @ `eff_bs=128`) single evaluation on
    `Tune6k` (6k) + 6 OOD + 7 `Shard 080` Stress + Sec 12 `BA3` + 50/50 Blend with
    `DualRegimeModalNet` + Stage 327 `Score-Bank KL-DRO` (`dro_bal_l008_kl15_bank8192`) +
    1,000x paired family bootstrap (`seed=20263250`).
"""

from __future__ import annotations

import argparse
import atexit
from concurrent.futures import ProcessPoolExecutor
import copy
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler

from analysis.flavour.inventory import write_json
from analysis.flavour.nextgen.gn2_tracklet_ipformer import (
    GN2_ORIGIN_CLASSES,
    GN2_SIDECAR_BATCH_KEYS,
    TrackletIPFormerGN2,
    collate_tracklet_gn2,
    compute_gn2_window_denominators,
    derive_tracklet_gn2_targets,
    extract_tracklet_gn2_item,
    gn2_tracklet_ipformer_loss,
)
from analysis.flavour.nextgen.protocols import compute_student_t_pulls, marginal_four_from_seven
from analysis.flavour.nextgen.run_dual_regime_modalnet import (
    ALL_STAGE2_OOD,
    BASE_OBS_CFG,
    CLEAN_ROOT,
    SPLITS_DIR,
    STAGE2_SEEDS,
    STAGE325_ROOT,
    STRESS_SPECS,
    _align_items_to_keys,
    _apply_scalar_temperature,
    _ba4_and_nll,
    _collaboration_section12_slice,
    _collaboration_t3_eval,
    _fit_scalar_temperature_on_screen,
    _load_ood_observations,
    _paired_family_bootstrap_full,
    _perturb_observation_for_stress,
    _set_ieee_fp32,
    _split_half_tail_eval,
    _weighted_ba4_pct,
    _within_slice_tail_eval,
)
from analysis.flavour.nextgen.seal_guard import (
    SealAuditGuard,
    SealViolationError,
    TRUTH_BATCH_KEYS,
    assert_unsealed_path,
    assert_unsealed_split,
    load_authorized_split_shards,
    verify_inference_truth_invariance,
)
from analysis.flavour.nextgen.tail_dro_pauc import (
    TailDROConfig,
    extract_penultimate_cache,
    finetune_head_on_cached_latents,
    freeze_backbone_for_tail_finetuning,
    predict_from_cached_latents,
    snapshot_backbone_weights,
    verify_backbone_bit_identity,
)
from analysis.flavour.v2.partitions import FAMILY, KEY
from analysis.flavour.v2.submit import choose_schedd, gpu_requirement
from analysis.flavour.v3.common.domains import (
    OBS_DOMAIN_NAMES,
    compute_observable_containment,
    cumulative_radiation_length,
    detector_bounds,
)
from analysis.flavour.v3.common.submit import _package_source
from analysis.flavour.v3.v2_pinpointformer.train import move_batch_to_device
from analysis.flavour.v3.v4_tau.ensemble import _apply_domain_convex_blend, _fit_domain_convex_blend
from analysis.flavour.v3.v6_clean_sim.evaluate_stage317 import _select_seed_prediction_df


NEXTGEN_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research')
EXP_ROOT = NEXTGEN_ROOT / 'exp_l1_gn2'
STAGE1_CACHE_DIR = EXP_ROOT / '_stage1_item_cache'
STAGE2_DIR = EXP_ROOT / 'stage2'
STAGE2_CACHE_DIR = STAGE2_DIR / '_stage2_item_cache'
SIDECAR_DIR = NEXTGEN_ROOT / 'l1_cluster_sidecar'
CONDOR_DIR = Path.home() / 'condor/Pinpoint/stage328_l1_gn2'

STAGE1_S326_TRACKLET_DIR = NEXTGEN_ROOT / 'exp_tracklet_ipformer'
STAGE1_S326_DUAL_DIR = NEXTGEN_ROOT / 'exp_dual_regime'
STAGE2_S326_DUAL_DIR = STAGE1_S326_DUAL_DIR / 'stage2'
STAGE2_S326_TRACKLET_DIR = STAGE1_S326_TRACKLET_DIR / 'stage2'
STAGE2_S327_DRO_DIR = NEXTGEN_ROOT / 'exp_tail_dro/stage2'

VARIANT_CONFIGS: dict[str, dict[str, Any]] = {
    'gn2_full': {
        'enable_origin_purity': True,
        'enable_covertex': True,
        'enable_readout_gate': True,
    },
    'l1_ctrl_no_gn2': {
        'enable_origin_purity': False,
        'enable_covertex': False,
        'enable_readout_gate': False,
    },
    'abl_origin_purity_only': {
        'enable_origin_purity': True,
        'enable_covertex': False,
        'enable_readout_gate': True,
    },
    'abl_covertex_only': {
        'enable_origin_purity': False,
        'enable_covertex': True,
        'enable_readout_gate': True,
    },
    'abl_aux_loss_only_no_readout_gate': {
        'enable_origin_purity': True,
        'enable_covertex': True,
        'enable_readout_gate': False,
    },
}

LOCKED_DRO_CFG = TailDROConfig(
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
    seed=42,
)

STAGE1_FORBIDDEN_RE = re.compile(
    r'(?:^|/)shards/0(?:8\d|9\d)(?:/|$)|validation_meta|tune_screen\.parquet|tune_truth_meta\.parquet',
    re.IGNORECASE,
)
_STAGE1_HOOK_INSTALLED = False


def install_stage1_only_hook() -> None:
    """Block Tune8k/080..099 filesystem syscalls strictly during Stage-I execution."""
    global _STAGE1_HOOK_INSTALLED
    if _STAGE1_HOOK_INSTALLED:
        return

    def _stage1_hook(event: str, args: tuple[Any, ...]) -> None:
        if event not in ('open', 'os.stat', 'os.scandir', 'os.listdir') or not args:
            return
        target = args[0]
        if isinstance(target, int):
            return
        try:
            path_str = os.fspath(target) if hasattr(target, '__fspath__') else str(target)
        except Exception:
            return
        if isinstance(path_str, bytes):
            path_str = path_str.decode('utf-8', errors='ignore')
        norm = path_str.replace('\\', '/')
        if STAGE1_FORBIDDEN_RE.search(norm):
            raise SealViolationError(f'Stage-I forbids accessing Tune8k/validation path: {norm}')

    sys.addaudithook(_stage1_hook)
    _STAGE1_HOOK_INSTALLED = True


def _resolve_fast_cache_root(min_free_gb: float = 8.0) -> Path:
    shm = Path('/dev/shm')
    if shm.is_dir() and os.access(shm, os.W_OK):
        try:
            free_gb = shutil.disk_usage(shm).free / (1024 ** 3)
            if free_gb >= min_free_gb:
                return shm
        except OSError:
            pass
    for env_k in ('PINPOINT_V3_SCRATCH', '_CONDOR_SCRATCH_DIR'):
        val = os.environ.get(env_k)
        if val and Path(val).is_dir():
            return Path(val)
    return Path('/tmp')


def _stage_pkl_dir_to_shm(src_dir: Path, filenames: list[str], prefix: str) -> tuple[Path, bool]:
    fast_root = _resolve_fast_cache_root(min_free_gb=12.0)
    if fast_root != Path('/dev/shm'):
        scratch = os.environ.get('PINPOINT_V3_SCRATCH') or os.environ.get('_CONDOR_SCRATCH_DIR')
        if scratch and Path(scratch).is_dir():
            fast_root = Path(scratch)
        else:
            return src_dir, False
    dst_dir = Path(tempfile.mkdtemp(prefix=f'{prefix}_', dir=str(fast_root)))
    atexit.register(lambda: shutil.rmtree(dst_dir, ignore_errors=True))
    for fn in filenames:
        src_f = src_dir / fn
        if src_f.is_file():
            shutil.copy2(src_f, dst_dir / fn)
    return dst_dir, True


def _resolve_l1_data_root() -> Path:
    """Resolve L1_Train32k root (`SIDECAR_DIR`), ensuring Option A paired dataset exists."""
    assert_unsealed_path(SIDECAR_DIR)
    if (SIDECAR_DIR / 'data/shards/000').is_dir():
        return SIDECAR_DIR
    raise FileNotFoundError(
        f'L1_Train32k dataset not found under {SIDECAR_DIR}; wait for @prod-l1 sidecar production.'
    )


def _load_l1_train_shards(l1_root: Path, *, stage: int) -> list[Path]:
    """Load shards 000..079 from L1_Train32k cleanly through SealAuditGuard."""
    data_dir = assert_unsealed_path(l1_root / 'data')
    if (data_dir / 'split_shards.json').is_file():
        return load_authorized_split_shards(data_dir, 'train', stage=stage)
    shards = [assert_unsealed_path(data_dir / 'shards' / f'{i:03d}') for i in range(80)]
    missing = [str(p) for p in shards if not p.is_dir()]
    if missing:
        raise FileNotFoundError(f'Missing L1_Train32k shards: {missing[:4]} ...')
    return shards


def _load_l1_internal_splits(l1_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load L1_internal_{train,dev_screen,val_eval} key DataFrames from L1_Train32k."""
    assert_unsealed_split('internal_train', stage=1)
    assert_unsealed_split('internal_screen', stage=1)
    assert_unsealed_split('internal_val', stage=1)
    for base_d, pref in (
        (l1_root, 'l1_internal_'),
        (l1_root / 'splits', 'l1_internal_'),
        (l1_root / 'splits', 'internal_'),
    ):
        tr_f = base_d / f'{pref}train_keys.parquet'
        dv_f = base_d / f'{pref}dev_screen_keys.parquet'
        va_f = base_d / f'{pref}val_eval_keys.parquet'
        if tr_f.is_file() and dv_f.is_file() and va_f.is_file():
            return (
                pd.read_parquet(assert_unsealed_path(tr_f)),
                pd.read_parquet(assert_unsealed_path(dv_f)),
                pd.read_parquet(assert_unsealed_path(va_f)),
            )

    shards = _load_l1_train_shards(l1_root, stage=1)
    prov_all = pd.concat(
        [pd.read_parquet(assert_unsealed_path(sp / 'provenance.parquet')) for sp in shards],
        ignore_index=True,
    )
    if 'internal_split' in prov_all.columns:
        tr_df = prov_all[prov_all['internal_split'] == 'internal_train'].reset_index(drop=True)
        dv_df = prov_all[prov_all['internal_split'] == 'internal_dev_screen'].reset_index(drop=True)
        va_df = prov_all[prov_all['internal_split'] == 'internal_val_eval'].reset_index(drop=True)
        return tr_df, dv_df, va_df

    spec_p = assert_unsealed_path(l1_root / 'l1_internal_split_spec.json')
    if not spec_p.is_file():
        raise FileNotFoundError(f'Missing {spec_p} and l1_internal_*_keys.parquet in {l1_root}')
    spec = json.loads(spec_p.read_text())
    splits_node = spec.get('splits', spec)

    def _slice_split(names: tuple[str, ...]) -> pd.DataFrame:
        entry = next((splits_node[k] for k in names if k in splits_node), None)
        if entry is None:
            raise KeyError(f'Could not find split {names} in {spec_p}')
        if isinstance(entry, dict) and 'keys' in entry:
            entry = entry['keys']
        if isinstance(entry, list) and entry and isinstance(entry[0], dict):
            edf = pd.DataFrame(entry)
            if all(k in edf.columns for k in KEY):
                idx_m = pd.MultiIndex.from_frame(prov_all[KEY]).isin(pd.MultiIndex.from_frame(edf[KEY]))
                return prov_all[idx_m].reset_index(drop=True)
            if all(k in edf.columns for k in FAMILY):
                idx_m = pd.MultiIndex.from_frame(prov_all[FAMILY]).isin(pd.MultiIndex.from_frame(edf[FAMILY]))
                return prov_all[idx_m].reset_index(drop=True)
        raise ValueError(f'Unsupported l1_internal_split_spec format for {names}: type={type(entry)}')

    return (
        _slice_split(('internal_train', 'L1_internal_train', 'train')),
        _slice_split(('internal_dev_screen', 'L1_internal_dev_screen', 'dev_screen')),
        _slice_split(('internal_val_eval', 'L1_internal_val_eval', 'val_eval')),
    )


def _load_cluster_sidecar_table(sidecar_dir: Path = SIDECAR_DIR) -> pd.DataFrame:
    """Load `l1_cluster_sidecar` Parquet shard(s) covering L1_Train32k (`000..079`) strictly unsealed."""
    assert_unsealed_path(sidecar_dir)
    if not sidecar_dir.is_dir():
        raise FileNotFoundError(f'Missing l1_cluster_sidecar directory at {sidecar_dir}')
    sub_cs = sidecar_dir / 'cluster_sidecar'
    if sub_cs.is_dir():
        pq_files = sorted(p for p in sub_cs.glob('*.parquet') if 'tmp' not in p.name.lower())
    else:
        pq_files = sorted(
            p for p in sidecar_dir.rglob('*cluster_sidecar*.parquet')
            if 'tmp' not in p.name.lower()
        )
    if not pq_files:
        raise FileNotFoundError(f'No cluster_sidecar parquet files found under {sidecar_dir}')
    frames = [pd.read_parquet(assert_unsealed_path(p)) for p in pq_files]
    return pd.concat(frames, ignore_index=True)


def _attach_cluster_sidecar_to_items(
    items: list[dict],
    keys_df: pd.DataFrame,
    sidecar_dir: Path = SIDECAR_DIR,
    *,
    match_tol_mm: float = 0.050,
) -> dict[str, Any]:
    """Attach `cl_origin_label`, `cl_sv_group_id`, top-3 trackIDs, & secondary origin onto items in-place.

    Supports both nested per-event list columns (`blueprint_l1_l2.md` Sec 1.2) and flat per-cluster tables.
    """
    sc_df = _load_cluster_sidecar_table(sidecar_dir)
    cols = set(sc_df.columns)

    # Resolve column names flexibly across producer conventions
    orig_col = next((c for c in ('origin_label', 'cl_origin_label', 'cluster_origin') if c in cols), None)
    sec_orig_col = next((c for c in ('secondary_origin_label', 'cl_secondary_origin_label', 'alt_origin_label') if c in cols), None)
    sv_col = next((c for c in ('sv_group_id', 'cl_sv_group_id') if c in cols), None)
    tid_col = next((c for c in ('dominant_track_id', 'cl_dominant_track_id', 'track_id') if c in cols), None)
    tid2_col = next((c for c in ('second_track_id', 'cl_second_track_id') if c in cols), None)
    tid3_col = next((c for c in ('third_track_id', 'cl_third_track_id') if c in cols), None)
    pl_col = next((c for c in ('cl_plane', 'plane', 'pix_plane', 'cluster_plane') if c in cols), None)
    x_col = next((c for c in ('cl_x_mm', 'x_mm', 'cluster_x_mm', 'x') if c in cols), None)
    y_col = next((c for c in ('cl_y_mm', 'y_mm', 'cluster_y_mm', 'y') if c in cols), None)
    cid_col = next((c for c in ('cluster_first_cid', 'first_channel_id', 'pix_first_cid') if c in cols), None)
    if orig_col is None or sv_col is None or tid_col is None:
        raise KeyError(f'l1_cluster_sidecar missing required columns; found={sorted(cols)}')

    is_nested = (len(sc_df) <= 40000) and isinstance(sc_df[orig_col].iloc[0], (list, np.ndarray))
    tol2 = float(match_tol_mm) ** 2

    ev_map: dict[tuple[int, int, int], Any] = {}
    if is_nested:
        for row in sc_df.itertuples(index=False):
            k = (int(row.run), int(row.chunk), int(row.row))
            ev_map[k] = row
    else:
        sc_df = sc_df.sort_values(['run', 'chunk', 'row'], kind='mergesort').reset_index(drop=True)
        r_arr = sc_df['run'].to_numpy(dtype=np.int64)
        c_arr = sc_df['chunk'].to_numpy(dtype=np.int64)
        w_arr = sc_df['row'].to_numpy(dtype=np.int64)
        chg = np.flatnonzero(
            (r_arr[1:] != r_arr[:-1]) | (c_arr[1:] != c_arr[:-1]) | (w_arr[1:] != w_arr[:-1])
        ) + 1
        starts = np.r_[0, chg]
        ends = np.r_[chg, len(sc_df)]
        orig_all = sc_df[orig_col].to_numpy(dtype=np.int64)
        sec_orig_all = sc_df[sec_orig_col].to_numpy(dtype=np.int64) if sec_orig_col else None
        sv_all = sc_df[sv_col].to_numpy(dtype=np.int64)
        tid_all = sc_df[tid_col].to_numpy(dtype=np.int64)
        tid2_all = sc_df[tid2_col].to_numpy(dtype=np.int64) if tid2_col else None
        tid3_all = sc_df[tid3_col].to_numpy(dtype=np.int64) if tid3_col else None
        pl_all = sc_df[pl_col].to_numpy(dtype=np.int64) if pl_col else None
        x_all = sc_df[x_col].to_numpy(dtype=np.float32) if x_col else None
        y_all = sc_df[y_col].to_numpy(dtype=np.float32) if y_col else None
        cid_all = sc_df[cid_col].to_numpy(dtype=np.uint64) if cid_col else None
        for s_i, e_i in zip(starts, ends):
            k = (int(r_arr[s_i]), int(c_arr[s_i]), int(w_arr[s_i]))
            ev_map[k] = (
                orig_all[s_i:e_i],
                sv_all[s_i:e_i],
                tid_all[s_i:e_i],
                pl_all[s_i:e_i] if pl_all is not None else None,
                x_all[s_i:e_i] if x_all is not None else None,
                y_all[s_i:e_i] if y_all is not None else None,
                cid_all[s_i:e_i] if cid_all is not None else None,
                tid2_all[s_i:e_i] if tid2_all is not None else None,
                tid3_all[s_i:e_i] if tid3_all is not None else None,
                sec_orig_all[s_i:e_i] if sec_orig_all is not None else None,
            )

    n_ev_matched = 0
    n_tgt_cl_tot = 0
    n_tgt_cl_matched = 0

    for idx, row in enumerate(keys_df.itertuples(index=False)):
        it = items[idx]
        k = (int(row.run), int(row.chunk), int(row.row))
        n_cl = len(it['pix_xyz_mm'])
        cl_o = np.full(n_cl, -1, dtype=np.int64)
        cl_so = np.full(n_cl, 255, dtype=np.int64)
        cl_s = np.full(n_cl, -1, dtype=np.int64)
        cl_t = np.full(n_cl, -1, dtype=np.int64)
        cl_t2 = np.full(n_cl, -1, dtype=np.int64)
        cl_t3 = np.full(n_cl, -1, dtype=np.int64)

        rec = ev_map.get(k)
        if rec is not None and n_cl > 0:
            n_ev_matched += 1
            if is_nested:
                s_orig = np.asarray(getattr(rec, orig_col), dtype=np.int64)
                s_so = np.asarray(getattr(rec, sec_orig_col), dtype=np.int64) if sec_orig_col else None
                s_sv = np.asarray(getattr(rec, sv_col), dtype=np.int64)
                s_tid = np.asarray(getattr(rec, tid_col), dtype=np.int64)
                s_t2 = np.asarray(getattr(rec, tid2_col), dtype=np.int64) if tid2_col else None
                s_t3 = np.asarray(getattr(rec, tid3_col), dtype=np.int64) if tid3_col else None
                s_pl = np.asarray(getattr(rec, pl_col), dtype=np.int64) if pl_col else None
                s_x = np.asarray(getattr(rec, x_col), dtype=np.float32) if x_col else None
                s_y = np.asarray(getattr(rec, y_col), dtype=np.float32) if y_col else None
                s_cid = np.asarray(getattr(rec, cid_col), dtype=np.uint64) if cid_col else None
            else:
                s_orig, s_sv, s_tid, s_pl, s_x, s_y, s_cid, s_t2, s_t3, s_so = rec

            def _copy_src(dst_idx: Any, src_idx: Any) -> None:
                cl_o[dst_idx] = s_orig[src_idx]
                cl_s[dst_idx] = s_sv[src_idx]
                cl_t[dst_idx] = s_tid[src_idx]
                if s_t2 is not None:
                    cl_t2[dst_idx] = s_t2[src_idx]
                if s_t3 is not None:
                    cl_t3[dst_idx] = s_t3[src_idx]
                if s_so is not None:
                    cl_so[dst_idx] = s_so[src_idx]

            # 0. Fast O(1) exact 1-to-1 alignment check when lengths & coordinates match
            matched = np.zeros(n_cl, dtype=bool)
            if (
                len(s_orig) == n_cl
                and s_pl is not None
                and s_x is not None
                and s_y is not None
                and np.array_equal(s_pl, it['pix_plane'])
                and float(np.max(np.abs(s_x - it['pix_xyz_mm'][:, 0]))) <= match_tol_mm
                and float(np.max(np.abs(s_y - it['pix_xyz_mm'][:, 1]))) <= match_tol_mm
            ):
                _copy_src(slice(None), slice(None))
                matched[:] = True

            # 1. Exact first-channel-ID match if available
            if not matched.all() and s_cid is not None and 'pix_first_cid' in it and len(s_cid) > 0:
                ord_c = np.argsort(s_cid, kind='mergesort')
                cid_sorted = s_cid[ord_c]
                pos = np.searchsorted(cid_sorted, it['pix_first_cid'])
                ok = pos < len(cid_sorted)
                safe_pos = np.minimum(pos, len(cid_sorted) - 1)
                ok = ok & (cid_sorted[safe_pos] == it['pix_first_cid'])
                if ok.any():
                    m_loc = np.flatnonzero(ok)
                    _copy_src(m_loc, ord_c[safe_pos[m_loc]])
                    matched[m_loc] = True

            # 2. Spatial (plane, x_mm, y_mm) < 50 um match for remaining clusters
            rem_idx = np.flatnonzero(~matched)
            if len(rem_idx) > 0 and s_pl is not None and s_x is not None and s_y is not None and len(s_pl) > 0:
                p_xyz = it['pix_xyz_mm']
                p_pl = it['pix_plane']
                for pl_val in np.unique(p_pl[rem_idx]):
                    q_loc = rem_idx[p_pl[rem_idx] == pl_val]
                    s_loc = np.flatnonzero(s_pl == pl_val)
                    if len(s_loc) == 0:
                        continue
                    dx = p_xyz[q_loc, 0:1].astype(np.float64) - s_x[s_loc][None, :].astype(np.float64)
                    dy = p_xyz[q_loc, 1:2].astype(np.float64) - s_y[s_loc][None, :].astype(np.float64)
                    d2 = dx * dx + dy * dy
                    best_j = d2.argmin(axis=1)
                    best_d2 = d2[np.arange(len(q_loc)), best_j]
                    hit = best_d2 <= tol2
                    if hit.any():
                        _copy_src(q_loc[hit], s_loc[best_j[hit]])
                        matched[q_loc[hit]] = True

        tgt_m = (it['pix_plane'] < 12) if n_cl > 0 else np.zeros(0, dtype=bool)
        n_tgt_cl_tot += int(tgt_m.sum())
        n_tgt_cl_matched += int((cl_o[tgt_m] >= 0).sum()) if tgt_m.any() else 0

        it['cl_origin_label'] = cl_o.astype(np.int16)
        it['cl_secondary_origin_label'] = cl_so.astype(np.int16)
        it['cl_sv_group_id'] = cl_s.astype(np.int32)
        it['cl_dominant_track_id'] = cl_t.astype(np.int32)
        it['cl_second_track_id'] = cl_t2.astype(np.int32)
        it['cl_third_track_id'] = cl_t3.astype(np.int32)

    return {
        'events_total': int(len(items)),
        'events_matched': int(n_ev_matched),
        'target_clusters_total': int(n_tgt_cl_tot),
        'target_clusters_matched': int(n_tgt_cl_matched),
        'target_cluster_match_rate': float(n_tgt_cl_matched / max(1, n_tgt_cl_tot)),
    }


class InMemoryGN2Dataset(Dataset):
    """In-memory dataset with event_sidecar & cluster_sidecar attachment."""

    def __init__(
        self,
        items: list[dict],
        event_sidecar: pd.DataFrame | None = None,
    ) -> None:
        self.items = items
        if event_sidecar is not None:
            self._sc_label7 = event_sidecar['label_7'].to_numpy(dtype=np.int64, copy=True)
            self._sc_tb = event_sidecar['tau_branch_valid'].to_numpy(dtype=np.int64, copy=True)
            self._sc_cf = event_sidecar['charm_flag'].to_numpy(dtype=np.int64, copy=True)
            self._sc_cv = event_sidecar['charm_valid'].to_numpy(dtype=np.int64, copy=True)
            self._sc_vtx = np.column_stack([
                event_sidecar['vertex_x_mm'].to_numpy(dtype=np.float32),
                event_sidecar['vertex_y_mm'].to_numpy(dtype=np.float32),
                event_sidecar['vertex_z_mm'].to_numpy(dtype=np.float32),
            ]).astype(np.float32, copy=True)
            self._sc_vv = event_sidecar['vertex_valid'].to_numpy(dtype=np.int64, copy=True)
            self._sc_kin = np.column_stack([
                event_sidecar['vis_pz_log1p'].to_numpy(dtype=np.float32),
                event_sidecar['vis_pt_log1p'].to_numpy(dtype=np.float32),
                event_sidecar['vis_cos_dphi'].to_numpy(dtype=np.float32),
                event_sidecar['vis_sin_dphi'].to_numpy(dtype=np.float32),
            ]).astype(np.float32, copy=True)
            self._sc_kv = event_sidecar['kin_valid'].to_numpy(dtype=np.int64, copy=True)
        else:
            self._sc_label7 = None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        it = dict(self.items[idx])
        it['index'] = int(idx)
        if self._sc_label7 is not None:
            it['label_7'] = int(self._sc_label7[idx])
            it['tau_branch_valid'] = int(self._sc_tb[idx])
            it['charm_flag'] = int(self._sc_cf[idx])
            it['charm_valid'] = int(self._sc_cv[idx])
            it['vertex_xyz_mm'] = self._sc_vtx[idx]
            it['vertex_valid'] = int(self._sc_vv[idx])
            it['vis_kinematics'] = self._sc_kin[idx]
            it['kin_valid'] = int(self._sc_kv[idx])
        return it


class _GN2Collator:
    def __init__(self, y_scint_center: float, n_bars: int) -> None:
        self.y_scint_center = float(y_scint_center)
        self.n_bars = int(n_bars)

    def __call__(self, xs: list[dict]) -> dict:
        return collate_tracklet_gn2(xs, mirror_rng=None, y_scint_center=self.y_scint_center, n_bars=self.n_bars)


def _update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for pe, pm in zip(ema.parameters(), model.parameters()):
            pe.data.mul_(decay).add_(pm.data, alpha=1.0 - decay)
        for be, bm in zip(ema.buffers(), model.buffers()):
            be.copy_(bm)


def _extract_gn2_shard_worker(task: tuple) -> list[dict]:
    shard_str, geom, bounds, cfg, expected_split = task
    shard_p = Path(assert_unsealed_path(shard_str))
    prov = pd.read_parquet(shard_p / 'provenance.parquet', columns=KEY + FAMILY + ['split', 'label'])
    obs_list = pq.read_table(shard_p / 'observations.parquet').to_pylist()
    allowed_splits = (
        {'train', 'internal_train', 'internal_dev_screen', 'internal_val_eval'}
        if expected_split == 'train'
        else {expected_split}
    )
    items = []
    for li, row in enumerate(prov.itertuples(index=False)):
        if row.split not in allowed_splits:
            continue
        ob = obs_list[li]
        dm = compute_observable_containment(ob, geom, bounds, config=cfg)
        it = extract_tracklet_gn2_item(li, ob, int(row.label), dm, geom, bounds, cfg)
        it['run'] = int(row.run)
        it['chunk'] = int(row.chunk)
        it['row'] = int(row.row)
        it['family_hi'] = int(row.family_hi)
        it['family_lo'] = int(row.family_lo)
        items.append(it)
    return items


def _extract_gn2_obs_chunk_worker(task: tuple) -> list[dict]:
    raw_obs, labels, start_idx, geom, bounds, cfg = task
    items = []
    for k, ob in enumerate(raw_obs):
        lbl = None if labels is None else int(labels[k])
        dm = compute_observable_containment(ob, geom, bounds, config=cfg)
        items.append(extract_tracklet_gn2_item(start_idx + k, ob, lbl, dm, geom, bounds, cfg))
    return items


def _parallel_extract_gn2_obs_list(
    raw_obs: list[dict],
    labels: np.ndarray | None,
    geom: dict,
    bounds: dict,
    cfg: dict,
    pool: ProcessPoolExecutor | None = None,
    chunk_size: int = 100,
) -> list[dict]:
    tasks = []
    for st in range(0, len(raw_obs), chunk_size):
        sub_o = raw_obs[st : st + chunk_size]
        sub_l = None if labels is None else labels[st : st + chunk_size]
        tasks.append((sub_o, sub_l, st, geom, bounds, cfg))
    if pool is not None:
        parts = list(pool.map(_extract_gn2_obs_chunk_worker, tasks))
    else:
        parts = [_extract_gn2_obs_chunk_worker(t) for t in tasks]
    return [it for part in parts for it in part]


def _build_gn2_stress_items_from_internal_val(
    val_keys_400: pd.DataFrame,
    geom: dict,
    bounds: dict,
    shards_root: Path,
    pool: ProcessPoolExecutor | None = None,
) -> tuple[dict[str, list[dict]], np.ndarray]:
    assert_unsealed_split('internal_val', stage=1)
    by_shard: dict[int, list[tuple[int, int, int]]] = {}
    for idx_pos, row in enumerate(val_keys_400.itertuples(index=False)):
        by_shard.setdefault(int(row.chunk), []).append((idx_pos, int(row.run), int(row.row)))
    raw_obs_400: list[dict | None] = [None] * len(val_keys_400)
    labels_400 = val_keys_400['label'].to_numpy(dtype=np.int64)
    for chunk_id, entries in by_shard.items():
        shard_p = assert_unsealed_path(shards_root / f'{chunk_id:03d}')
        prov = pd.read_parquet(shard_p / 'provenance.parquet', columns=KEY)
        obs_tbl = pq.read_table(shard_p / 'observations.parquet').to_pylist()
        pos_map = {(int(r), int(c), int(rw)): i for i, (r, c, rw) in enumerate(zip(prov['run'], prov['chunk'], prov['row']))}
        for idx_pos, r, rw in entries:
            raw_obs_400[idx_pos] = obs_tbl[pos_map[(r, chunk_id, rw)]]
    stress_items: dict[str, list[dict]] = {}
    for st_name, pkw, cfg_over in STRESS_SPECS:
        rng_st = np.random.default_rng(32510)
        cfg_use = {**BASE_OBS_CFG, **cfg_over}
        pert_obs = [
            (_perturb_observation_for_stress(ob, bounds, rng_st, **pkw) if pkw else ob)
            for ob in raw_obs_400
            if ob is not None
        ]
        stress_items[st_name] = _parallel_extract_gn2_obs_list(pert_obs, labels_400, geom, bounds, cfg_use, pool=pool)
    return stress_items, labels_400


def _build_gn2_shard080_stress_items(
    geom: dict,
    bounds: dict,
    pool: ProcessPoolExecutor | None = None,
) -> tuple[dict[str, list[dict]], np.ndarray, pd.DataFrame]:
    assert_unsealed_split('validation', stage=2)
    shard080 = assert_unsealed_path(CLEAN_ROOT / 'data/shards/080')
    obs_080 = pq.read_table(shard080 / 'observations.parquet').to_pylist()
    prov_080 = pd.read_parquet(shard080 / 'provenance.parquet', columns=KEY + FAMILY + ['split', 'label'])
    y_080 = prov_080['label'].to_numpy(dtype=np.int64)
    stress_items: dict[str, list[dict]] = {}
    for st_name, pkw, cfg_over in STRESS_SPECS:
        rng_st = np.random.default_rng(32510)
        cfg_use = {**BASE_OBS_CFG, **cfg_over}
        pert_obs = [
            (_perturb_observation_for_stress(ob, bounds, rng_st, **pkw) if pkw else ob)
            for ob in obs_080
        ]
        stress_items[st_name] = _parallel_extract_gn2_obs_list(pert_obs, None, geom, bounds, cfg_use, pool=pool)
    return stress_items, y_080, prov_080


@torch.no_grad()
def predict_gn2_items_fp32(
    model: TrackletIPFormerGN2,
    items: list[dict],
    bounds: dict,
    device: torch.device,
    batch_size: int = 64,
    *,
    return_gn2_preds: bool = False,
) -> dict[str, Any]:
    """Deterministic IEEE-FP32 pure-observation inference (sidecar/truth stripped)."""
    _set_ieee_fp32(True)
    model.eval()
    yc, nb = float(bounds['y_scint_center']), int(bounds['n_bars'])
    strip_keys = set(TRUTH_BATCH_KEYS) | set(GN2_SIDECAR_BATCH_KEYS)

    probs_l, vp_l, sp_l, sv_l, ch_l = [], [], [], [], []
    wreal_l, pdisp_l, pconv_l, ppv_l, ptau_l, cab_l, tmask_l = [], [], [], [], [], [], []
    for st in range(0, len(items), batch_size):
        sub = items[st : st + batch_size]
        raw_b = collate_tracklet_gn2(sub, mirror_rng=None, y_scint_center=yc, n_bars=nb)
        clean_b = move_batch_to_device({k: v for k, v in raw_b.items() if k not in strip_keys}, device)
        with torch.amp.autocast('cuda', enabled=False):
            out = model.forward_NextGen(clean_b)
        probs_l.append(out.probs4.float().cpu().numpy())
        vp_l.append(out.v_pv_mm.float().cpu().numpy())
        sp_l.append(out.scale_pv_mm.float().cpu().numpy())
        sv_l.append(out.sv_disp_mm.float().cpu().numpy() if out.sv_disp_mm is not None else np.zeros((len(sub), 3), dtype=np.float32))
        ch_l.append(torch.sigmoid(out.charm_logit.float()).cpu().numpy())
        if return_gn2_preds:
            tmask_l.append(out.aux['trk_mask'].bool().cpu().numpy())
            wreal_l.append(out.aux['hat_w_real'].float().cpu().numpy())
            pdisp_l.append(out.aux['hat_p_disp_hf'].float().cpu().numpy())
            pconv_l.append(out.aux['hat_p_conv'].float().cpu().numpy())
            ppv_l.append(out.aux['hat_p_pv'].float().cpu().numpy())
            if 'tracklet_origin_logits' in out.aux:
                po = torch.softmax(out.aux['tracklet_origin_logits'].float(), dim=-1)
                ptau_l.append((po[..., 1] * out.aux['hat_w_real'].float()).cpu().numpy())
            if 'hat_C_ab' in out.aux:
                cab_l.append(out.aux['hat_C_ab'].float().cpu().numpy())

    res: dict[str, Any] = {
        'probs': np.concatenate(probs_l, axis=0).astype(np.float32),
        'v_pv_mm': np.concatenate(vp_l, axis=0).astype(np.float32),
        'scale_pv_mm': np.concatenate(sp_l, axis=0).astype(np.float32),
        'sv_disp_mm': np.concatenate(sv_l, axis=0).astype(np.float32),
        'charm_prob': np.concatenate(ch_l, axis=0).astype(np.float32),
    }
    if return_gn2_preds:
        res['trk_mask'] = np.concatenate(tmask_l, axis=0)
        res['hat_w_real'] = np.concatenate(wreal_l, axis=0)
        res['hat_p_disp_hf'] = np.concatenate(pdisp_l, axis=0)
        res['hat_p_conv'] = np.concatenate(pconv_l, axis=0)
        res['hat_p_pv'] = np.concatenate(ppv_l, axis=0)
        if ptau_l:
            res['hat_w_tau'] = np.concatenate(ptau_l, axis=0)
        if cab_l:
            res['hat_C_ab'] = np.concatenate(cab_l, axis=0)
    return res


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y = np.asarray(y_true, dtype=np.int64)
    if y.size == 0 or y.min() == y.max():
        return None
    return float(roc_auc_score(y, np.asarray(y_score, dtype=np.float64)))


@torch.no_grad()
def evaluate_microscopic_gn2_probes(
    gn2_preds: dict[str, Any],
    val_items_with_sidecar: list[dict],
    bounds: dict,
) -> dict[str, Any]:
    """Strictly post-inference offline microscopic tracklet AUCs on held-out `internal_val_eval` sidecar."""
    yc, nb = float(bounds['y_scint_center']), int(bounds['n_bars'])
    gv_l, pur_l, orig_l, cov_v_l, cov_y_l = [], [], [], [], []
    for st in range(0, len(val_items_with_sidecar), 64):
        b_cpu = collate_tracklet_gn2(val_items_with_sidecar[st : st + 64], mirror_rng=None, y_scint_center=yc, n_bars=nb)
        t_dict = derive_tracklet_gn2_targets(b_cpu)
        if t_dict is None:
            return {'available': False}
        gv_l.append(t_dict['trk_gn2_valid'].numpy())
        pur_l.append(t_dict['trk_purity_target'].numpy())
        orig_l.append(t_dict['trk_origin_target'].numpy())
        cov_v_l.append(t_dict['trk_covertex_valid'].numpy())
        cov_y_l.append(t_dict['trk_covertex_target'].numpy())

    gv = np.concatenate(gv_l, axis=0)
    y_pur = np.concatenate(pur_l, axis=0)
    y_orig = np.concatenate(orig_l, axis=0)
    cov_v = np.concatenate(cov_v_l, axis=0)
    y_cov = np.concatenate(cov_y_l, axis=0)

    n_valid = int(gv.sum())
    if n_valid == 0:
        return {'available': False}

    yp_v = (y_pur[gv] > 0.5).astype(np.int64)
    yo_v = y_orig[gv]
    out: dict[str, Any] = {
        'available': True,
        'n_valid_target_tracklets': n_valid,
        'true_purity_rate': float(yp_v.mean()),
        'origin_counts': {str(k): int((yo_v == k).sum()) for k in range(GN2_ORIGIN_CLASSES)},
    }
    if 'hat_w_real' in gn2_preds:
        w_real_v = gn2_preds['hat_w_real'][gv]
        out['purity_auroc'] = _safe_auroc(yp_v, w_real_v)
    if 'hat_p_conv' in gn2_preds:
        out['conversion_auroc'] = _safe_auroc((yo_v == 3).astype(np.int64), gn2_preds['hat_p_conv'][gv])
        # Among displaced-like / non-PV tracklets, conversion vs heavy-flavour
        hf_or_conv = (yo_v == 1) | (yo_v == 2) | (yo_v == 3)
        if hf_or_conv.any():
            out['hf_vs_conversion_auroc'] = _safe_auroc(
                ((yo_v[hf_or_conv] == 1) | (yo_v[hf_or_conv] == 2)).astype(np.int64),
                gn2_preds['hat_p_disp_hf'][gv][hf_or_conv],
            )
    if 'hat_w_tau' in gn2_preds:
        out['tau_prong_auroc'] = _safe_auroc((yo_v == 1).astype(np.int64), gn2_preds['hat_w_tau'][gv])
        out['hf_disp_auroc'] = _safe_auroc(
            ((yo_v == 1) | (yo_v == 2)).astype(np.int64),
            (gn2_preds['hat_p_disp_hf'] * gn2_preds['hat_w_real'])[gv],
        )
    if 'hat_C_ab' in gn2_preds and cov_v.any():
        out['covertex_auroc'] = _safe_auroc((y_cov[cov_v] > 0.5).astype(np.int64), gn2_preds['hat_C_ab'][cov_v])
        out['n_valid_covertex_pairs'] = int(cov_v.sum())
        out['covertex_pos_rate'] = float((y_cov[cov_v] > 0.5).mean())
    return out


# ===========================================================================
# Stage-I Pre-extraction & Training
# ===========================================================================
def run_preextract_ood6() -> dict[str, Any]:
    """Pre-extract the 6 OOD evaluation packs (`ALL_STAGE2_OOD`, strictly unsealed) into STAGE1_CACHE_DIR."""
    assert_unsealed_split('ood', stage=1)
    t0 = time.perf_counter()
    n_cpus = min(8, max(1, os.cpu_count() or 1))
    geom = json.loads(assert_unsealed_path(CLEAN_ROOT / 'geometry.json').read_text())
    bounds = detector_bounds(geom)
    STAGE1_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_f = STAGE1_CACHE_DIR / 'ood_6_pack.pkl'
    if out_f.is_file():
        return {'cached': True, 'path': str(out_f)}
    with ProcessPoolExecutor(max_workers=n_cpus, mp_context=mp.get_context('spawn')) as pool:
        ood_pre = {}
        for ov in ALL_STAGE2_OOD:
            raw_o, y_o = _load_ood_observations(ov)
            ood_pre[ov] = (_parallel_extract_gn2_obs_list(raw_o, None, geom, bounds, BASE_OBS_CFG, pool=pool), y_o)
    tmp_f = STAGE1_CACHE_DIR / 'ood_6_pack.pkl.tmp'
    with tmp_f.open('wb') as fh:
        pickle.dump(ood_pre, fh, protocol=5)
    tmp_f.replace(out_f)
    counts = {ov: int(len(v[0])) for ov, v in ood_pre.items()}
    return {'cached': False, 'path': str(out_f), 'counts': counts, 'seconds': float(time.perf_counter() - t0)}


def run_stage1_preextract(part: int = -1) -> dict[str, Any]:
    """Sharded parallel Stage-I pure-observable extraction (`part=0..7` train, `part=8` OOD+stress)."""
    assert_unsealed_split('internal_train', stage=1)
    assert_unsealed_split('internal_screen', stage=1)
    assert_unsealed_split('internal_val', stage=1)

    t0 = time.perf_counter()
    n_cpus = min(8, max(1, os.cpu_count() or 1))
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)

    l1_root = _resolve_l1_data_root()
    geom_p = l1_root / 'geometry.json'
    geom = json.loads(assert_unsealed_path(geom_p if geom_p.is_file() else (CLEAN_ROOT / 'geometry.json')).read_text())
    bounds = detector_bounds(geom)
    STAGE1_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    scratch_base = str(_resolve_fast_cache_root(min_free_gb=6.0))

    parts_to_run = list(range(9)) if part < 0 else [part]
    reports = {}
    for p_cur in parts_to_run:
        if 0 <= p_cur <= 7:
            out_f = STAGE1_CACHE_DIR / f'train_part_{p_cur}.pkl'
            if out_f.is_file():
                reports[str(p_cur)] = {'part': p_cur, 'cached': True}
                continue
            all_auth = _load_l1_train_shards(l1_root, stage=1)
            sub_shards = all_auth[p_cur * 10 : (p_cur + 1) * 10]
            with tempfile.TemporaryDirectory(prefix=f'gn2-pre-{p_cur}-', dir=scratch_base) as tmp_dir:
                tmp_s = Path(tmp_dir) / 'shards'
                tmp_s.mkdir(parents=True, exist_ok=True)
                local_dirs = []
                for sp in sub_shards:
                    assert_unsealed_path(sp)
                    dst_sp = tmp_s / sp.name
                    shutil.copytree(sp, dst_sp)
                    local_dirs.append(dst_sp)
                tasks = [(str(sd), geom, bounds, BASE_OBS_CFG, 'train') for sd in local_dirs]
                with ProcessPoolExecutor(max_workers=n_cpus, mp_context=mp.get_context('spawn')) as pool:
                    parts = list(pool.map(_extract_gn2_shard_worker, tasks))
                part_items = [it for p in parts for it in p]
            tmp_f = STAGE1_CACHE_DIR / f'train_part_{p_cur}.pkl.tmp'
            with tmp_f.open('wb') as fh:
                pickle.dump(part_items, fh, protocol=5)
            tmp_f.replace(out_f)
            sec = float(time.perf_counter() - t0)
            print(f'[stage1-preextract part={p_cur}] Wrote {len(part_items)} items in {sec:.1f}s', flush=True)
            reports[str(p_cur)] = {'part': p_cur, 'n_items': len(part_items), 'seconds': sec}
        elif p_cur == 8:
            out_f = STAGE1_CACHE_DIR / 'ood_stress_pack.pkl'
            if out_f.is_file():
                reports['8'] = {'part': 8, 'cached': True}
                continue
            _, _, val_keys = _load_l1_internal_splits(l1_root)
            val_400 = val_keys.iloc[:400].reset_index(drop=True)
            ood_cache_f = STAGE1_CACHE_DIR / 'ood_6_pack.pkl'
            with ProcessPoolExecutor(max_workers=n_cpus, mp_context=mp.get_context('spawn')) as pool:
                if ood_cache_f.is_file():
                    with ood_cache_f.open('rb') as fh:
                        ood_pre = pickle.load(fh)
                else:
                    ood_pre = {}
                    for ov in ALL_STAGE2_OOD:
                        raw_o, y_o = _load_ood_observations(ov)
                        ood_pre[ov] = (_parallel_extract_gn2_obs_list(raw_o, None, geom, bounds, BASE_OBS_CFG, pool=pool), y_o)
                stress_items, y_400 = _build_gn2_stress_items_from_internal_val(
                    val_400, geom, bounds, l1_root / 'data/shards', pool=pool
                )
            tmp_f = STAGE1_CACHE_DIR / 'ood_stress_pack.pkl.tmp'
            with tmp_f.open('wb') as fh:
                pickle.dump({'ood_pre': ood_pre, 'stress_items': stress_items, 'y_400': y_400}, fh, protocol=5)
            tmp_f.replace(out_f)
            sec = float(time.perf_counter() - t0)
            print(f'[stage1-preextract part=8] Wrote 6-OOD + stress pack in {sec:.1f}s', flush=True)
            reports['8'] = {'part': 8, 'seconds': sec}
        else:
            raise ValueError(f'Expected --part in -1..8, got {p_cur}')
    return reports if part < 0 else reports[str(part)]


def _train_single_stage1_variant(
    variant: str,
    *,
    geom: dict,
    bounds: dict,
    tr_items: list[dict],
    dev_items: list[dict],
    val_items_pure: list[dict],
    val_items_probe: list[dict],
    tr_sidecar: pd.DataFrame,
    tr_keys: pd.DataFrame,
    dev_keys: pd.DataFrame,
    val_keys: pd.DataFrame,
    ood_pre: dict[str, tuple[list[dict], np.ndarray]],
    stress_items: dict[str, list[dict]],
    y_400: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    utc_started = datetime.now(timezone.utc).isoformat()
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    yc, nb = float(bounds['y_scint_center']), int(bounds['n_bars'])
    out_dir = EXP_ROOT / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = InMemoryGN2Dataset(tr_items, event_sidecar=tr_sidecar)
    model_cfg = {
        'dim': 176,
        'heads': 8,
        'n_rel_pre': 2,
        'n_rel_post': 2,
        'n_global_layers': 2,
        'drop_path': 0.10,
        'head_dropout': 0.15,
        **VARIANT_CONFIGS[variant],
    }
    model = TrackletIPFormerGN2(geom, model_cfg).to(device)
    ema_model = copy.deepcopy(model).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    n_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    base_lr = 8.0e-4
    min_lr = 1.0e-5
    wd = 0.03
    passes = 20
    batch_size = 128
    micro_bs = 64
    grad_accum = 2
    ema_decay = 0.999

    opt = torch.optim.AdamW(model.build_param_groups(base_lr, wd), lr=base_lr, weight_decay=wd, betas=(0.9, 0.98))
    steps_per_pass = math.ceil(len(train_ds) / batch_size)
    total_steps = passes * steps_per_pass
    warmup_steps = max(1, int(0.06 * total_steps))
    min_lr_ratio = min_lr / base_lr

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * min(1.0, prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
    cap = torch.cuda.get_device_capability(device)
    amp_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
    amp_ok = cap[0] >= 7
    scaler = torch.amp.GradScaler('cuda', enabled=(amp_ok and amp_dtype == torch.float16))

    y_dev = dev_keys['label'].to_numpy(dtype=np.int64)
    best_nll = float('inf')
    best_pass = 0
    history: list[dict[str, Any]] = []
    best_ema_path = out_dir / 'best_ema.pt'

    n_dl_workers = 0
    if n_dl_workers > 0:
        torch.multiprocessing.set_sharing_strategy('file_system')
    collator = _GN2Collator(yc, nb)

    torch.cuda.reset_peak_memory_stats(device)
    global_step = 0
    for p_idx in range(1, passes + 1):
        tp0 = time.perf_counter()
        gen = torch.Generator()
        gen.manual_seed(seed + p_idx * 101)
        sampler = BatchSampler(RandomSampler(train_ds, generator=gen), batch_size=micro_bs, drop_last=False)
        dl_kwargs: dict[str, Any] = {
            'batch_sampler': sampler,
            'num_workers': n_dl_workers,
            'collate_fn': collator,
            'pin_memory': True,
        }
        if n_dl_workers > 0:
            dl_kwargs['prefetch_factor'] = 2
        loader = DataLoader(train_ds, **dl_kwargs)

        model.train()
        _set_ieee_fp32(False)
        loss_sum = 0.0
        n_seen = 0

        def _step_win(win_buf: list[dict]) -> None:
            nonlocal global_step, loss_sum, n_seen
            global_step += 1
            opt.zero_grad(set_to_none=True)
            fd = compute_gn2_window_denominators(win_buf)
            win_loss = 0.0
            win_n = 0
            for sub_mb in win_buf:
                db = move_batch_to_device(sub_mb, device)
                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_ok):
                    out_dc = model.forward_NextGen(db)
                    loss, _ = gn2_tracklet_ipformer_loss(out_dc, db, fixed_denoms=fd)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                win_loss += float(loss.detach().item())
                win_n += len(db['label'])
            if scaler.is_enabled():
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(opt)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
            scheduler.step()
            _update_ema(ema_model, model, ema_decay)
            loss_sum += win_loss * win_n
            n_seen += win_n

        win_buf: list[dict] = []
        for mb in loader:
            win_buf.append(mb)
            if len(win_buf) >= grad_accum:
                _step_win(win_buf)
                win_buf.clear()
        if win_buf:
            _step_win(win_buf)
            win_buf.clear()

        train_loss = loss_sum / max(n_seen, 1)
        row_hist: dict[str, Any] = {'pass': p_idx, 'train_loss': train_loss, 'pass_sec': float(time.perf_counter() - tp0)}

        if p_idx == 1 or p_idx >= 10 or p_idx % 2 == 0 or p_idx == passes:
            dev_pred = predict_gn2_items_fp32(ema_model, dev_items, bounds, device, batch_size=64)
            dev_m = _ba4_and_nll(y_dev, dev_pred['probs'])
            row_hist['dev_screen_NLL'] = dev_m['NLL']
            row_hist['dev_screen_BA4_pct'] = dev_m['BA4_pct']
            if dev_m['NLL'] < best_nll:
                best_nll = dev_m['NLL']
                best_pass = p_idx
                torch.save({'model': ema_model.state_dict(), 'config': model_cfg, 'pass': p_idx}, best_ema_path)
            print(
                f"[{variant}] pass={p_idx:02d}/{passes} train_loss={train_loss:.4f} "
                f"dev_NLL={dev_m['NLL']:.4f} dev_BA4={dev_m['BA4_pct']:.2f}% ({row_hist['pass_sec']:.1f}s)",
                flush=True,
            )
        history.append(row_hist)

    peak_vram_mib = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
    _set_ieee_fp32(True)
    ckpt = torch.load(best_ema_path, map_location=device, weights_only=False)
    ema_model.load_state_dict(ckpt['model'])
    ema_model.eval()

    dev_pred = predict_gn2_items_fp32(ema_model, dev_items, bounds, device, batch_size=64)
    val_pred = predict_gn2_items_fp32(ema_model, val_items_pure, bounds, device, batch_size=64, return_gn2_preds=True)

    np.save(out_dir / 'probs_dev_screen.npy', dev_pred['probs'])
    np.save(out_dir / 'probs_internal_val_eval.npy', val_pred['probs'])
    np.save(out_dir / 'v_pv_internal_val_eval.npy', val_pred['v_pv_mm'])
    np.save(out_dir / 'scale_pv_internal_val_eval.npy', val_pred['scale_pv_mm'])
    np.save(out_dir / 'sv_disp_internal_val_eval.npy', val_pred['sv_disp_mm'])

    # Strictly post-inference offline microscopic probe AUCs on held-out internal_val_eval sidecar
    micro_probes = evaluate_microscopic_gn2_probes(val_pred, val_items_probe, bounds)

    # 6 OOD & 7 Stress evaluations in IEEE-FP32
    ood_probs: dict[str, np.ndarray] = {}
    ood_metrics: dict[str, Any] = {}
    for ov, (o_items, yo) in ood_pre.items():
        op = predict_gn2_items_fp32(ema_model, o_items, bounds, device, batch_size=64)['probs']
        ood_probs[f'ood_{ov}'] = op
        if 'nc_' in ov:
            ood_metrics[ov] = {
                'n_events': int(len(yo)),
                'mean_p_tau': float(op[:, 2].mean()),
                'frac_pred_nc_pct': float((op.argmax(1) == 3).mean() * 100.0),
                'frac_pred_tau_pct': float((op.argmax(1) == 2).mean() * 100.0),
                'frac_p_tau_gt_0p5_pct': float((op[:, 2] > 0.5).mean() * 100.0),
            }
        else:
            ood_metrics[ov] = _ba4_and_nll(yo, op)

    stress_probs: dict[str, np.ndarray] = {}
    stress_metrics: dict[str, Any] = {}
    p_nom = None
    ba4_nom = 0.0
    for st_name, _, _ in STRESS_SPECS:
        sp = predict_gn2_items_fp32(ema_model, stress_items[st_name], bounds, device, batch_size=64)['probs']
        stress_probs[f'stress_{st_name}'] = sp
        m_st = _ba4_and_nll(y_400, sp)
        if st_name == 'nominal':
            p_nom = sp
            ba4_nom = m_st['BA4_pct']
        tv = float(0.5 * np.abs(sp - p_nom).sum(axis=1).mean()) if p_nom is not None else 0.0
        stress_metrics[st_name] = {
            'BA4_pct': m_st['BA4_pct'],
            'delta_BA4_vs_nominal_pp': float(m_st['BA4_pct'] - ba4_nom),
            'NLL': m_st['NLL'],
            'mean_TV_vs_nominal': tv,
        }
    np.savez_compressed(out_dir / 'ood_and_stress_probs.npz', **ood_probs, **stress_probs)

    # Gate-A 39-key + cluster_sidecar tamper check on live trained checkpoint
    b_chk = move_batch_to_device(
        collate_tracklet_gn2(val_items_probe[:8], mirror_rng=None, y_scint_center=yc, n_bars=nb),
        device,
    )
    truth_inv = verify_inference_truth_invariance(ema_model, b_chk, atol=0.0)
    # Also verify cluster_sidecar keys tamper == 0.0
    clean_b = {k: v for k, v in b_chk.items() if k not in (set(TRUTH_BATCH_KEYS) | set(GN2_SIDECAR_BATCH_KEYS))}
    mut_b = dict(b_chk)
    mut_b['cl_origin_label'] = torch.ones_like(b_chk['cl_origin_label'])
    mut_b['cl_sv_group_id'] = torch.full_like(b_chk['cl_sv_group_id'], 77)
    mut_b['cl_dominant_track_id'] = torch.full_like(b_chk['cl_dominant_track_id'], 999)
    with torch.inference_mode():
        o_c = ema_model.forward_NextGen(clean_b)
        o_m = ema_model.forward_NextGen(mut_b)
    sidecar_tamper_diff = max(
        float((getattr(o_c, fn) - getattr(o_m, fn)).abs().max().item())
        for fn in ('logits7', 'probs4', 'charm_logit', 'v_pv_mm', 'scale_pv_mm', 'sv_disp_mm', 'kin_pred')
    )
    if sidecar_tamper_diff > 0.0:
        raise AssertionError(f'cluster_sidecar tamper diff {sidecar_tamper_diff} != 0.0')

    y_val = val_keys['label'].to_numpy(dtype=np.int64)
    val_m = _ba4_and_nll(y_val, val_pred['probs'])
    summary = {
        'variant': variant,
        'UTC_started': utc_started,
        'UTC_finished': datetime.now(timezone.utc).isoformat(),
        'wall_seconds': float(time.perf_counter() - t0),
        'gpu_name': torch.cuda.get_device_name(device),
        'n_params': n_params,
        'seed': seed,
        'passes': passes,
        'effective_batch_size': batch_size,
        'micro_bs': micro_bs,
        'grad_accum': grad_accum,
        'total_optimizer_steps': global_step,
        'peak_vram_MiB': peak_vram_mib,
        'best_pass': best_pass,
        'best_dev_screen_NLL': best_nll,
        'truth_invariance_max_diff': truth_inv['max_abs_diff'],
        'sidecar_tamper_max_diff': sidecar_tamper_diff,
        'internal_val_eval_all': val_m,
        'microscopic_probe_aucs_val_eval': micro_probes,
        'ood_metrics': ood_metrics,
        'stress_400_internal_val': stress_metrics,
        'history': history,
    }
    write_json(out_dir / 'run_summary.json', summary)
    print(
        f"[{variant}] DONE in {summary['wall_seconds']:.1f}s | peak_VRAM={peak_vram_mib:.0f} MiB | "
        f"val_BA4={val_m['BA4_pct']:.2f}% val_NLL={val_m['NLL']:.4f} | probes={micro_probes}",
        flush=True,
    )
    return summary


def run_stage1_train(variants: list[str]) -> dict[str, Any]:
    """Train one or more Stage-I variants sequentially on one GPU sharing `/dev/shm` cache."""
    for v in variants:
        if v not in VARIANT_CONFIGS:
            raise ValueError(f'Unknown variant {v!r}; expected one of {sorted(VARIANT_CONFIGS)}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for Stage-I training')

    assert_unsealed_split('internal_train', stage=1)
    assert_unsealed_split('internal_screen', stage=1)
    assert_unsealed_split('internal_val', stage=1)
    device = torch.device('cuda:0')

    l1_root = _resolve_l1_data_root()
    geom_p = l1_root / 'geometry.json'
    geom = json.loads(assert_unsealed_path(geom_p if geom_p.is_file() else (CLEAN_ROOT / 'geometry.json')).read_text())
    bounds = detector_bounds(geom)
    tr_keys, dev_keys, val_keys = _load_l1_internal_splits(l1_root)

    sc_df = pd.read_parquet(assert_unsealed_path(l1_root / 'truth_sidecar/event_sidecar.parquet'))
    pos_sc = pd.MultiIndex.from_frame(sc_df[KEY]).get_indexer(pd.MultiIndex.from_frame(tr_keys[KEY]))
    if (pos_sc < 0).any():
        raise ValueError('L1 internal_train keys missing from event_sidecar.parquet')
    tr_sidecar = sc_df.iloc[pos_sc].reset_index(drop=True)
    del sc_df

    part_names = [f'train_part_{p}.pkl' for p in range(8)] + ['ood_stress_pack.pkl']
    if not all((STAGE1_CACHE_DIR / fn).is_file() for fn in part_names):
        raise FileNotFoundError(f'Stage-I cache missing in {STAGE1_CACHE_DIR}; run --s1-preextract first')

    t_load = time.perf_counter()
    staged_dir, used_shm = _stage_pkl_dir_to_shm(STAGE1_CACHE_DIR, part_names, prefix='gn2_s1')
    try:
        all_items: list[dict] = []
        for p in range(8):
            pf = staged_dir / f'train_part_{p}.pkl'
            with pf.open('rb') as fh:
                all_items.extend(pickle.load(fh))
            if used_shm:
                pf.unlink(missing_ok=True)
        osf = staged_dir / 'ood_stress_pack.pkl'
        with osf.open('rb') as fh:
            os_pack = pickle.load(fh)
        if used_shm:
            osf.unlink(missing_ok=True)
    finally:
        if used_shm:
            shutil.rmtree(staged_dir, ignore_errors=True)

    ood_pre = os_pack['ood_pre']
    stress_items = os_pack['stress_items']
    y_400 = os_pack['y_400']
    del os_pack

    tr_items = _align_items_to_keys(all_items, tr_keys)
    dev_items = _align_items_to_keys(all_items, dev_keys)
    val_items_pure = _align_items_to_keys(all_items, val_keys)
    del all_items
    gc.collect()

    # Keep a shallow-copied val list for strictly post-inference microscopic probe AUCs;
    # val_items_pure remains 100% free of sidecar keys during inference!
    val_items_probe = [dict(it) for it in val_items_pure]
    tr_match = _attach_cluster_sidecar_to_items(tr_items, tr_keys, SIDECAR_DIR)
    val_match = _attach_cluster_sidecar_to_items(val_items_probe, val_keys, SIDECAR_DIR)
    print(
        f'[Stage-I] Loaded cache (shm={used_shm}) in {time.perf_counter()-t_load:.1f}s | '
        f'sidecar match train={tr_match["target_cluster_match_rate"]:.4f} '
        f'val={val_match["target_cluster_match_rate"]:.4f}',
        flush=True,
    )
    write_json(EXP_ROOT / 'stage1_sidecar_alignment.json', {'train': tr_match, 'val_eval': val_match})

    results = {}
    for v in variants:
        results[v] = _train_single_stage1_variant(
            v,
            geom=geom,
            bounds=bounds,
            tr_items=tr_items,
            dev_items=dev_items,
            val_items_pure=val_items_pure,
            val_items_probe=val_items_probe,
            tr_sidecar=tr_sidecar,
            tr_keys=tr_keys,
            dev_keys=dev_keys,
            val_keys=val_keys,
            ood_pre=ood_pre,
            stress_items=stress_items,
            y_400=y_400,
            device=device,
        )
        gc.collect()
        torch.cuda.empty_cache()
    return results


def run_stage1_aggregate() -> dict[str, Any]:
    """Aggregate Stage-I runs, check gates vs `l1_ctrl_no_gn2` & `s1_tracklet_ipformer`, and 1,000x bootstrap."""
    assert_unsealed_split('internal_train', stage=1)
    assert_unsealed_split('internal_val', stage=1)
    l1_root = _resolve_l1_data_root()
    _, _, val_keys = _load_l1_internal_splits(l1_root)
    sc_df = pd.read_parquet(assert_unsealed_path(l1_root / 'truth_sidecar/event_sidecar.parquet'))

    y_val = val_keys['label'].to_numpy(dtype=np.int64)
    dom_val = val_keys['obs_domain_id'].to_numpy(dtype=np.int64)
    reg_val = val_keys['region_id'].to_numpy(dtype=np.int64)
    fam_codes = pd.MultiIndex.from_frame(val_keys[FAMILY]).factorize()[0]

    pos_sc = pd.MultiIndex.from_frame(sc_df[KEY]).get_indexer(pd.MultiIndex.from_frame(val_keys[KEY]))
    sc_val = sc_df.iloc[pos_sc].reset_index(drop=True)
    v_true = np.column_stack([
        sc_val['vertex_x_mm'].to_numpy(np.float64),
        sc_val['vertex_y_mm'].to_numpy(np.float64),
        sc_val['vertex_z_mm'].to_numpy(np.float64),
    ])
    v_valid = sc_val['vertex_valid'].to_numpy(dtype=np.int64) > 0

    probs_map: dict[str, np.ndarray] = {}
    models_report: dict[str, Any] = {}
    reg_names = {0: 'FrontPinpoint_30mm_PixXY', 1: 'IntermPinpoint_150mm_PixXY', 2: 'Pinpoint_OuterW', 3: 'Fortune_ScintOnly'}

    for name in VARIANT_CONFIGS:
        d_run = EXP_ROOT / name
        if not (d_run / 'probs_internal_val_eval.npy').is_file():
            continue
        probs = np.load(d_run / 'probs_internal_val_eval.npy').astype(np.float64)
        probs_map[name] = probs
        run_sum = json.loads((d_run / 'run_summary.json').read_text())

        dom_res = {
            OBS_DOMAIN_NAMES[did]: _ba4_and_nll(y_val[dom_val == did], probs[dom_val == did])
            for did in (1, 2, 3)
            if (dom_val == did).any()
        }
        reg_res = {
            rname: _ba4_and_nll(y_val[reg_val == rid], probs[reg_val == rid])
            for rid, rname in reg_names.items()
            if (reg_val == rid).any()
        }
        s_tau = np.log(np.clip(probs[:, 2], 1e-8, 1.0)) - np.log(np.clip(1.0 - probs[:, 2], 1e-8, 1.0))
        m_all = np.ones(len(y_val), dtype=bool)
        m_d1 = dom_val == 1
        m_pin = (reg_val == 0) | (reg_val == 1)
        m_f30 = reg_val == 0
        tail = {
            'split_half_1pct_All': _split_half_tail_eval(s_tau, y_val, val_keys, m_all, 0.01),
            'split_half_1pct_D1_obs': _split_half_tail_eval(s_tau, y_val, val_keys, m_d1, 0.01),
            'split_half_1pct_Pinpoint': _split_half_tail_eval(s_tau, y_val, val_keys, m_pin, 0.01),
            'split_half_0p1pct_All': _split_half_tail_eval(s_tau, y_val, val_keys, m_all, 0.001),
            'within_slice_1pct_All': _within_slice_tail_eval(s_tau, y_val, m_all, 0.01),
            'within_slice_1pct_D1_obs': _within_slice_tail_eval(s_tau, y_val, m_d1, 0.01),
            'within_slice_1pct_Pinpoint': _within_slice_tail_eval(s_tau, y_val, m_pin, 0.01),
            'within_slice_1pct_Front30': _within_slice_tail_eval(s_tau, y_val, m_f30, 0.01),
            'within_slice_0p1pct_All': _within_slice_tail_eval(s_tau, y_val, m_all, 0.001),
        }
        collab_t3 = {
            'All_non_tau': _collaboration_t3_eval(probs, y_val, m_all),
            'D1_obs_non_tau': _collaboration_t3_eval(probs, y_val, m_d1),
            'D2_obs_non_tau': _collaboration_t3_eval(probs, y_val, dom_val == 2),
        }

        vh = np.load(d_run / 'v_pv_internal_val_eval.npy').astype(np.float64)
        sh = np.load(d_run / 'scale_pv_internal_val_eval.npy').astype(np.float64)
        pv_audit = {}
        for tag, mk in (
            ('All_valid', v_valid),
            ('Pinpoint_PixXY', v_valid & m_pin),
            ('FrontPinpoint_30mm', v_valid & (reg_val == 0)),
            ('IntermPinpoint_150mm', v_valid & (reg_val == 1)),
        ):
            if mk.any():
                dz = np.abs(vh[mk, 2] - v_true[mk, 2])
                r3 = np.linalg.norm(vh[mk] - v_true[mk], axis=-1)
                pulls = compute_student_t_pulls(
                    torch.from_numpy(vh), torch.from_numpy(sh),
                    torch.from_numpy(v_true), torch.from_numpy(mk.astype(np.int64)), df=3.0,
                )
                pv_audit[tag] = {
                    'n_events': int(mk.sum()),
                    'median_abs_dz_mm': float(np.median(dz)),
                    'median_3d_mm': float(np.median(r3)),
                    **pulls,
                }

        models_report[name] = {
            'n_params': run_sum.get('n_params'),
            'best_pass': run_sum.get('best_pass'),
            'wall_seconds': run_sum.get('wall_seconds'),
            'peak_vram_MiB': run_sum.get('peak_vram_MiB'),
            'internal_val_eval_all': _ba4_and_nll(y_val, probs),
            'by_obs_domain': dom_res,
            'by_truth_region': reg_res,
            'tau_tail': tail,
            'collaboration_t3': collab_t3,
            'pv_audit': pv_audit,
            'microscopic_probe_aucs_val_eval': run_sum.get('microscopic_probe_aucs_val_eval'),
            'ood_metrics': run_sum.get('ood_metrics'),
            'stress_400_internal_val': run_sum.get('stress_400_internal_val'),
        }

    s326_refs: dict[str, np.ndarray] = {}
    s326_common_mask = np.zeros(len(y_val), dtype=bool)
    p_tip_f = STAGE1_S326_TRACKLET_DIR / 's1_tracklet_ipformer/val_eval_predictions.parquet'
    if p_tip_f.is_file():
        df_tip = pd.read_parquet(p_tip_f)
        # Align strictly on (family_hi, family_lo) so the 3,600 shared (e, tau, NC) events
        # match 1-to-1 and the 1,200 disjoint 10086 vs 10082 mu_CC slots are excluded.
        if all(c in df_tip.columns for c in FAMILY):
            pos_t = pd.MultiIndex.from_frame(df_tip[FAMILY]).get_indexer(pd.MultiIndex.from_frame(val_keys[FAMILY]))
            m_ok = pos_t >= 0
            if m_ok.any():
                arr = np.zeros((len(y_val), 4), dtype=np.float64)
                arr[m_ok] = df_tip.iloc[pos_t[m_ok]][['p_0', 'p_1', 'p_2', 'p_3']].to_numpy(np.float64)
                s326_refs['s1_tracklet_ipformer'] = arr
                s326_common_mask = m_ok
    p_dr_f = STAGE1_S326_DUAL_DIR / 's1_dual_regime_modalnet/probs_internal_val_eval.npy'
    p_clean_vk = SPLITS_DIR / 'internal_val_eval_keys.parquet'
    if p_dr_f.is_file() and p_clean_vk.is_file() and s326_common_mask.any():
        df_cvk = pd.read_parquet(assert_unsealed_path(p_clean_vk))
        dr_p = np.load(p_dr_f).astype(np.float64)
        pos_d = pd.MultiIndex.from_frame(df_cvk[FAMILY]).get_indexer(pd.MultiIndex.from_frame(val_keys[FAMILY]))
        m_d_ok = (pos_d >= 0) & s326_common_mask
        if m_d_ok.any():
            arr_d = np.zeros((len(y_val), 4), dtype=np.float64)
            arr_d[m_d_ok] = dr_p[pos_d[m_d_ok]]
            s326_refs['s1_dual_regime_modalnet'] = arr_d
    if 's1_tracklet_ipformer' in s326_refs and 's1_dual_regime_modalnet' in s326_refs:
        s326_refs['s1_blend_50_50'] = 0.5 * (s326_refs['s1_tracklet_ipformer'] + s326_refs['s1_dual_regime_modalnet'])

    paired_boot: dict[str, Any] = {}
    m_all = np.ones(len(y_val), dtype=bool)
    m_d1 = dom_val == 1
    m_pin = (reg_val == 0) | (reg_val == 1)
    m_f30 = reg_val == 0
    for anchor in ('gn2_full', 'abl_origin_purity_only'):
        if anchor not in probs_map:
            continue
        pa = probs_map[anchor]
        for other_name, pb in probs_map.items():
            if other_name == anchor:
                continue
            for stag, mk in (('All', m_all), ('D1_obs', m_d1), ('Pinpoint', m_pin), ('Front30', m_f30)):
                paired_boot[f'{anchor} - {other_name} ({stag})'] = _paired_family_bootstrap_full(
                    pa, pb, y_val, fam_codes, mk, n_boot=1000, seed=20263250
                )
        for other_name, pb in s326_refs.items():
            for stag, mk in (('All_common3600', m_all), ('D1_obs_common', m_d1), ('Pinpoint_common', m_pin), ('Front30_common', m_f30)):
                mk_c = mk & s326_common_mask
                if mk_c.any():
                    paired_boot[f'{anchor} - {other_name} ({stag})'] = _paired_family_bootstrap_full(
                        pa, pb, y_val, fam_codes, mk_c, n_boot=1000, seed=20263250
                    )

    d1_key = OBS_DOMAIN_NAMES[1]
    ctrl_ref: dict[str, Any] = {}
    if 'l1_ctrl_no_gn2' in models_report:
        cr = models_report['l1_ctrl_no_gn2']
        ctrl_ref = {
            'All_BA4': cr['internal_val_eval_all']['BA4_pct'],
            'All_NLL': cr['internal_val_eval_all']['NLL'],
            'D1_BA4': cr['by_obs_domain'][d1_key]['BA4_pct'],
            'Front30_BA4': cr['by_truth_region']['FrontPinpoint_30mm_PixXY']['BA4_pct'],
            'W184_p_tau_gt_0p5_pct': float((cr.get('ood_metrics') or {}).get('gen_ood_w184_epps21_nc_charm', {}).get('frac_p_tau_gt_0p5_pct', 999.0)),
        }

    def _eval_gate_for_variant(vname: str) -> dict[str, Any]:
        rep = models_report[vname]
        ba4 = rep['internal_val_eval_all']['BA4_pct']
        nll = rep['internal_val_eval_all']['NLL']
        d1_ba4 = rep['by_obs_domain'][d1_key]['BA4_pct']
        f30_ba4 = rep['by_truth_region']['FrontPinpoint_30mm_PixXY']['BA4_pct']
        w184_pt = float((rep.get('ood_metrics') or {}).get('gen_ood_w184_epps21_nc_charm', {}).get('frac_p_tau_gt_0p5_pct', 999.0))
        all_pulls = [
            float(x)
            for reg in ('All_valid', 'FrontPinpoint_30mm', 'IntermPinpoint_150mm')
            for x in rep['pv_audit'][reg]['p68_abs_pull']
        ]
        pulls_ok = all(0.85 <= p <= 1.15 for p in all_pulls)
        w184_ok = w184_pt <= 2.2
        if ctrl_ref:
            beats_ctrl = bool(
                (ba4 > ctrl_ref['All_BA4'])
                or (nll < ctrl_ref['All_NLL'] - 0.004)
                or (d1_ba4 > ctrl_ref['D1_BA4'] + 0.5)
                or (f30_ba4 > ctrl_ref['Front30_BA4'] + 0.5)
            )
        else:
            beats_ctrl = bool((ba4 > 57.81) or (nll < 0.9659) or (d1_ba4 > 64.24) or (f30_ba4 > 63.26))
        return {
            'variant': vname,
            'All_BA4_pct': ba4,
            'All_NLL': nll,
            'D1_BA4_pct': d1_ba4,
            'Front30_BA4_pct': f30_ba4,
            'W184_p_tau_gt_0p5_pct': w184_pt,
            'gate_beats_ctrl': beats_ctrl,
            'all_9_pulls': all_pulls,
            'gate_all_9_pulls_in_0p85_1p15': bool(pulls_ok),
            'gate_w184_le_2p2_pct': bool(w184_ok),
            'eligible': bool(beats_ctrl and pulls_ok and w184_ok),
        }

    per_variant_gates = {
        vn: _eval_gate_for_variant(vn)
        for vn in ('gn2_full', 'abl_origin_purity_only', 'abl_covertex_only', 'abl_aux_loss_only_no_readout_gate')
        if vn in models_report
    }
    eligible_variants = [vn for vn, g in per_variant_gates.items() if g['eligible']]
    promoted_variant = None
    if eligible_variants:
        promoted_variant = max(eligible_variants, key=lambda vn: per_variant_gates[vn]['All_BA4_pct'])

    gates: dict[str, Any] = {}
    if 'gn2_full' in per_variant_gates:
        g_full = per_variant_gates['gn2_full']
        gates = {
            'gn2_full_All_BA4_pct': g_full['All_BA4_pct'],
            'gn2_full_All_NLL': g_full['All_NLL'],
            'gn2_full_D1_BA4_pct': g_full['D1_BA4_pct'],
            'gn2_full_Front30_BA4_pct': g_full['Front30_BA4_pct'],
            'gn2_full_W184_p_tau_gt_0p5_pct': g_full['W184_p_tau_gt_0p5_pct'],
            'ref_l1_ctrl_no_gn2': ctrl_ref,
            'ref_s326_s1_tracklet_ipformer': {'All_BA4': 57.81, 'All_NLL': 0.9659, 'D1_BA4': 64.24, 'Front30_BA4': 63.26},
            'per_variant_gates': per_variant_gates,
            'promoted_variant': promoted_variant,
            'stage2_promotion_eligible': bool(promoted_variant is not None),
        }

    out_doc = {
        'UTC': datetime.now(timezone.utc).isoformat(),
        'split_manifest': str(l1_root),
        'internal_val_eval_events': int(len(y_val)),
        'models': models_report,
        'stage1_gates': gates,
        's326_references_common3600': {k: _ba4_and_nll(y_val[s326_common_mask], v[s326_common_mask]) for k, v in s326_refs.items()} if s326_common_mask.any() else {},
        'paired_family_bootstrap_1000_seed20263250': paired_boot,
    }
    write_json(EXP_ROOT / 'stage1_l1_gn2_results.json', out_doc)
    print(json.dumps({'stage1_gates': gates}, indent=2), flush=True)
    return out_doc


# ===========================================================================
# Stage-II Pre-extraction, Training & Evaluation (ONLY if Stage-I passes!)
# ===========================================================================
def run_stage2_preextract(part: int = -1) -> dict[str, Any]:
    for sp in ('train', 'validation', 'tune_screen', 'tune6k', 'ood'):
        assert_unsealed_split(sp, stage=2)

    t0 = time.perf_counter()
    n_cpus = min(8, max(1, os.cpu_count() or 1))
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)

    l1_root = _resolve_l1_data_root()
    geom_p = l1_root / 'geometry.json'
    geom = json.loads(assert_unsealed_path(geom_p if geom_p.is_file() else (CLEAN_ROOT / 'geometry.json')).read_text())
    bounds = detector_bounds(geom)
    STAGE2_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    scratch_base = str(_resolve_fast_cache_root(min_free_gb=6.0))

    parts_to_run = list(range(11)) if part < 0 else [part]
    reports = {}
    for p_cur in parts_to_run:
        if 0 <= p_cur <= 7:
            out_f = STAGE2_CACHE_DIR / f'train_part_{p_cur}.pkl'
            if out_f.is_file():
                reports[str(p_cur)] = {'part': p_cur, 'cached': True}
                continue
            s1_tr = STAGE1_CACHE_DIR / f'train_part_{p_cur}.pkl'
            if s1_tr.is_file():
                shutil.copy2(s1_tr, out_f)
                reports[str(p_cur)] = {'part': p_cur, 'copied_from_stage1': True}
                continue
            all_tr = _load_l1_train_shards(l1_root, stage=2)
            sub_shards = all_tr[p_cur * 10 : (p_cur + 1) * 10]
            with tempfile.TemporaryDirectory(prefix=f'gn2-s2-tr-{p_cur}-', dir=scratch_base) as tmp_dir:
                tmp_s = Path(tmp_dir) / 'shards'
                tmp_s.mkdir(parents=True, exist_ok=True)
                local_dirs = []
                for sp in sub_shards:
                    assert_unsealed_path(sp)
                    dst_sp = tmp_s / sp.name
                    shutil.copytree(sp, dst_sp)
                    local_dirs.append(dst_sp)
                tasks = [(str(sd), geom, bounds, BASE_OBS_CFG, 'train') for sd in local_dirs]
                with ProcessPoolExecutor(max_workers=n_cpus, mp_context=mp.get_context('spawn')) as pool:
                    parts = list(pool.map(_extract_gn2_shard_worker, tasks))
                part_items = [it for p in parts for it in p]
            tmp_f = STAGE2_CACHE_DIR / f'train_part_{p_cur}.pkl.tmp'
            with tmp_f.open('wb') as fh:
                pickle.dump(part_items, fh, protocol=5)
            tmp_f.replace(out_f)
            reports[str(p_cur)] = {'part': p_cur, 'n_items': len(part_items), 'seconds': float(time.perf_counter() - t0)}
        elif 8 <= p_cur <= 9:
            v_idx = p_cur - 8
            out_f = STAGE2_CACHE_DIR / f'val_part_{v_idx}.pkl'
            if out_f.is_file():
                reports[str(p_cur)] = {'part': p_cur, 'cached': True}
                continue
            all_va = load_authorized_split_shards(CLEAN_ROOT / 'data', 'validation', stage=2)
            sub_shards = all_va[v_idx * 10 : (v_idx + 1) * 10]
            with tempfile.TemporaryDirectory(prefix=f'gn2-s2-va-{v_idx}-', dir=scratch_base) as tmp_dir:
                tmp_s = Path(tmp_dir) / 'shards'
                tmp_s.mkdir(parents=True, exist_ok=True)
                local_dirs = []
                for sp in sub_shards:
                    assert_unsealed_path(sp)
                    dst_sp = tmp_s / sp.name
                    shutil.copytree(sp, dst_sp)
                    local_dirs.append(dst_sp)
                tasks = [(str(sd), geom, bounds, BASE_OBS_CFG, 'validation') for sd in local_dirs]
                with ProcessPoolExecutor(max_workers=n_cpus, mp_context=mp.get_context('spawn')) as pool:
                    parts = list(pool.map(_extract_gn2_shard_worker, tasks))
                part_items = [it for p in parts for it in p]
            tmp_f = STAGE2_CACHE_DIR / f'val_part_{v_idx}.pkl.tmp'
            with tmp_f.open('wb') as fh:
                pickle.dump(part_items, fh, protocol=5)
            tmp_f.replace(out_f)
            reports[str(p_cur)] = {'part': p_cur, 'n_items': len(part_items), 'seconds': float(time.perf_counter() - t0)}
        elif p_cur == 10:
            out_f = STAGE2_CACHE_DIR / 'ood_stress_stage2_pack.pkl'
            if out_f.is_file():
                reports['10'] = {'part': 10, 'cached': True}
                continue
            ood_cache_f = STAGE1_CACHE_DIR / 'ood_6_pack.pkl'
            with ProcessPoolExecutor(max_workers=n_cpus, mp_context=mp.get_context('spawn')) as pool:
                if ood_cache_f.is_file():
                    with ood_cache_f.open('rb') as fh:
                        ood_pre = pickle.load(fh)
                else:
                    ood_pre = {}
                    for ov in ALL_STAGE2_OOD:
                        raw_o, y_o = _load_ood_observations(ov)
                        ood_pre[ov] = (_parallel_extract_gn2_obs_list(raw_o, None, geom, bounds, BASE_OBS_CFG, pool=pool), y_o)
                stress_items, y_080, prov_080 = _build_gn2_shard080_stress_items(geom, bounds, pool=pool)
            tmp_f = STAGE2_CACHE_DIR / 'ood_stress_stage2_pack.pkl.tmp'
            with tmp_f.open('wb') as fh:
                pickle.dump({'ood_pre': ood_pre, 'stress_items': stress_items, 'y_080': y_080, 'prov_080': prov_080}, fh, protocol=5)
            tmp_f.replace(out_f)
            reports['10'] = {'part': 10, 'seconds': float(time.perf_counter() - t0)}
        else:
            raise ValueError(f'Expected --part in -1..10, got {p_cur}')
    return reports if part < 0 else reports[str(part)]


def run_stage2_train(seeds: list[int], variant: str = 'gn2_full') -> dict[str, Any]:
    """Train frozen `variant` (`gn2_full` or `l1_ctrl_no_gn2`) on L1_Train32k across `seeds` on 1 H200/H100 GPU."""
    if variant not in VARIANT_CONFIGS:
        raise ValueError(f'Unknown Stage-2 variant {variant!r}; expected one of {sorted(VARIANT_CONFIGS)}')
    for s in seeds:
        if s not in STAGE2_SEEDS:
            raise ValueError(f'Expected Stage-2 seed in {STAGE2_SEEDS}, got {s}')
    for sp in ('train', 'validation', 'tune_screen', 'tune6k', 'ood'):
        assert_unsealed_split(sp, stage=2)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for Stage-2 training')

    # Verify Stage-I gates passed before executing Stage-II
    s1_res_p = EXP_ROOT / 'stage1_l1_gn2_results.json'
    if s1_res_p.is_file():
        s1_doc = json.loads(s1_res_p.read_text())
        if not s1_doc.get('stage1_gates', {}).get('stage2_promotion_eligible', False):
            raise RuntimeError('Stage-I gates did NOT pass; refusing Stage-II execution!')

    l1_root = _resolve_l1_data_root()
    device = torch.device('cuda:0')
    geom_p = l1_root / 'geometry.json'
    geom = json.loads(assert_unsealed_path(geom_p if geom_p.is_file() else (CLEAN_ROOT / 'geometry.json')).read_text())
    bounds = detector_bounds(geom)
    yc, nb = float(bounds['y_scint_center']), int(bounds['n_bars'])

    tune_meta = pd.read_parquet(assert_unsealed_path(CLEAN_ROOT / 'validation_meta/tune_truth_meta.parquet'))
    scr_df = pd.read_parquet(assert_unsealed_path(CLEAN_ROOT / 'data/tune_screen.parquet'), columns=KEY)
    scr_mask = pd.MultiIndex.from_frame(tune_meta[KEY]).isin(pd.MultiIndex.from_frame(scr_df[KEY]))
    scr_idx = np.flatnonzero(scr_mask)
    y_scr = tune_meta['label'].to_numpy(dtype=np.int64)[scr_idx]

    sc_all = pd.read_parquet(assert_unsealed_path(l1_root / 'truth_sidecar/event_sidecar.parquet'))
    tr_df_list = []
    for sp in _load_l1_train_shards(l1_root, stage=2):
        prov = pd.read_parquet(sp / 'provenance.parquet', columns=KEY + ['split', 'label'])
        tr_df_list.append(prov[prov['split'].isin(['train', 'internal_train', 'internal_dev_screen', 'internal_val_eval'])])
    tr_keys = pd.concat(tr_df_list, ignore_index=True).sort_values(KEY, kind='mergesort').reset_index(drop=True)
    pos_sc = pd.MultiIndex.from_frame(sc_all[KEY]).get_indexer(pd.MultiIndex.from_frame(tr_keys[KEY]))
    if (pos_sc < 0).any():
        raise ValueError('L1_Train32k keys missing from event_sidecar.parquet')
    tr_sidecar = sc_all.iloc[pos_sc].reset_index(drop=True)
    del sc_all

    req_files = [f'train_part_{p}.pkl' for p in range(8)] + ['val_part_0.pkl', 'val_part_1.pkl', 'ood_stress_stage2_pack.pkl']
    if not all((STAGE2_CACHE_DIR / fn).is_file() for fn in req_files):
        raise FileNotFoundError(f'Stage-II cache missing in {STAGE2_CACHE_DIR}; run --s2-preextract first')

    staged_dir, used_shm = _stage_pkl_dir_to_shm(STAGE2_CACHE_DIR, req_files, prefix='gn2_s2')
    try:
        tr_raw, va_raw = [], []
        for p in range(8):
            with (staged_dir / f'train_part_{p}.pkl').open('rb') as fh:
                tr_raw.extend(pickle.load(fh))
        for vp in range(2):
            with (staged_dir / f'val_part_{vp}.pkl').open('rb') as fh:
                va_raw.extend(pickle.load(fh))
        with (staged_dir / 'ood_stress_stage2_pack.pkl').open('rb') as fh:
            os_pack = pickle.load(fh)
    finally:
        if used_shm:
            shutil.rmtree(staged_dir, ignore_errors=True)

    tr_items = _align_items_to_keys(tr_raw, tr_keys)
    tune8k_items = _align_items_to_keys(va_raw, tune_meta)
    scr_items = [tune8k_items[int(i)] for i in scr_idx]
    del tr_raw, va_raw
    gc.collect()

    s2_match = _attach_cluster_sidecar_to_items(tr_items, tr_keys, SIDECAR_DIR)
    print(f'[Stage-II] Train32k sidecar match_rate={s2_match["target_cluster_match_rate"]:.4f}', flush=True)

    ood_pre = os_pack['ood_pre']
    stress_items = os_pack['stress_items']
    del os_pack

    out_all: dict[str, Any] = {}
    for seed in seeds:
        t0 = time.perf_counter()
        utc_started = datetime.now(timezone.utc).isoformat()
        torch.manual_seed(seed)
        np.random.seed(seed)

        out_s2_base = STAGE2_DIR if variant == 'gn2_full' else (STAGE2_DIR / variant)
        seed_dir = out_s2_base / f'seed_{seed}'
        seed_dir.mkdir(parents=True, exist_ok=True)
        train_ds = InMemoryGN2Dataset(tr_items, event_sidecar=tr_sidecar)
        model_cfg = {
            'dim': 176,
            'heads': 8,
            'n_rel_pre': 2,
            'n_rel_post': 2,
            'n_global_layers': 2,
            'drop_path': 0.10,
            'head_dropout': 0.15,
            **VARIANT_CONFIGS[variant],
        }
        model = TrackletIPFormerGN2(geom, model_cfg).to(device)
        ema_model = copy.deepcopy(model).eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)
        n_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

        base_lr, min_lr, wd = 8.0e-4, 1.0e-5, 0.03
        passes, batch_size, micro_bs, grad_accum, ema_decay = 20, 128, 64, 2, 0.999
        opt = torch.optim.AdamW(model.build_param_groups(base_lr, wd), lr=base_lr, weight_decay=wd, betas=(0.9, 0.98))
        steps_per_pass = math.ceil(len(train_ds) / batch_size)
        total_steps = passes * steps_per_pass
        warmup_steps = max(1, int(0.06 * total_steps))
        min_lr_ratio = min_lr / base_lr

        def _lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * min(1.0, prog)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
        cap = torch.cuda.get_device_capability(device)
        amp_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
        amp_ok = cap[0] >= 7
        scaler = torch.amp.GradScaler('cuda', enabled=(amp_ok and amp_dtype == torch.float16))
        collator = _GN2Collator(yc, nb)
        n_dl_workers = 0

        best_nll = float('inf')
        best_pass = 0
        best_pt = seed_dir / 'best_ema.pt'
        history = []
        global_step = 0
        torch.cuda.reset_peak_memory_stats(device)

        for p_idx in range(1, passes + 1):
            tp0 = time.perf_counter()
            gen = torch.Generator()
            gen.manual_seed(seed * 1000 + p_idx)
            sampler = BatchSampler(RandomSampler(train_ds, generator=gen), batch_size=micro_bs, drop_last=False)
            dl_kwargs: dict[str, Any] = {'batch_sampler': sampler, 'num_workers': n_dl_workers, 'collate_fn': collator, 'pin_memory': True}
            if n_dl_workers > 0:
                dl_kwargs['prefetch_factor'] = 2
            loader = DataLoader(train_ds, **dl_kwargs)
            model.train()
            _set_ieee_fp32(False)
            loss_sum, n_seen = 0.0, 0
            win_buf: list[dict] = []

            def _step_s2(wb: list[dict]) -> None:
                nonlocal global_step, loss_sum, n_seen
                global_step += 1
                opt.zero_grad(set_to_none=True)
                fd = compute_gn2_window_denominators(wb)
                wl, wn = 0.0, 0
                for sub_mb in wb:
                    db = move_batch_to_device(sub_mb, device)
                    with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_ok):
                        loss, _ = gn2_tracklet_ipformer_loss(model.forward_NextGen(db), db, fixed_denoms=fd)
                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    wl += float(loss.detach().item())
                    wn += len(db['label'])
                if scaler.is_enabled():
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    opt.step()
                scheduler.step()
                _update_ema(ema_model, model, ema_decay)
                loss_sum += wl * wn
                n_seen += wn

            for mb in loader:
                win_buf.append(mb)
                if len(win_buf) >= grad_accum:
                    _step_s2(win_buf)
                    win_buf.clear()
            if win_buf:
                _step_s2(win_buf)
                win_buf.clear()

            row_h: dict[str, Any] = {'pass': p_idx, 'train_loss': loss_sum / max(n_seen, 1), 'seconds': float(time.perf_counter() - tp0)}
            if p_idx == 1 or p_idx >= 10 or p_idx % 2 == 0 or p_idx == passes:
                sp_pred = predict_gn2_items_fp32(ema_model, scr_items, bounds, device, batch_size=64)
                m_scr = _ba4_and_nll(y_scr, sp_pred['probs'])
                row_h.update({'tune_screen_NLL': m_scr['NLL'], 'tune_screen_BA4_pct': m_scr['BA4_pct']})
                if m_scr['NLL'] < best_nll:
                    best_nll = m_scr['NLL']
                    best_pass = p_idx
                    torch.save({'model': ema_model.state_dict(), 'config': model_cfg, 'pass': p_idx}, best_pt)
                print(
                    f"[stage2 seed={seed}] pass={p_idx:02d}/{passes} scr_NLL={m_scr['NLL']:.4f} scr_BA4={m_scr['BA4_pct']:.2f}%",
                    flush=True,
                )
            history.append(row_h)

        peak_vram = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        _set_ieee_fp32(True)
        ckpt = torch.load(best_pt, map_location=device, weights_only=False)
        ema_model.load_state_dict(ckpt['model'])
        ema_model.eval()

        t8_pred = predict_gn2_items_fp32(ema_model, tune8k_items, bounds, device, batch_size=64)
        np.save(seed_dir / f'probs_tune8k_seed{seed}.npy', t8_pred['probs'])
        np.save(seed_dir / f'v_pv_tune8k_seed{seed}.npy', t8_pred['v_pv_mm'])
        np.save(seed_dir / f'scale_pv_tune8k_seed{seed}.npy', t8_pred['scale_pv_mm'])
        np.save(seed_dir / f'sv_disp_tune8k_seed{seed}.npy', t8_pred['sv_disp_mm'])

        ood_p = {f'ood_{ov}': predict_gn2_items_fp32(ema_model, o_items, bounds, device, batch_size=64)['probs'] for ov, (o_items, _) in ood_pre.items()}
        st_p = {f'stress_{sn}': predict_gn2_items_fp32(ema_model, stress_items[sn], bounds, device, batch_size=64)['probs'] for sn, _, _ in STRESS_SPECS}
        np.savez_compressed(seed_dir / f'ood_and_stress_seed{seed}.npz', **ood_p, **st_p)

        # Stage-II Score-Bank KL-DRO tail fine-tuning on frozen Train32k penultimate latents
        dro_cfg = TailDROConfig(**{**LOCKED_DRO_CFG.__dict__, 'seed': seed})
        tr_batches = []
        strip_for_cache = set()
        for st in range(0, len(tr_items), 64):
            chunk_ds = [train_ds[j] for j in range(st, min(st + 64, len(tr_items)))]
            tr_batches.append(collate_tracklet_gn2(chunk_ds, mirror_rng=None, y_scint_center=yc, n_bars=nb))
        snap = snapshot_backbone_weights(ema_model, head_attr='event_head_7')
        head = freeze_backbone_for_tail_finetuning(ema_model, head_attr='event_head_7')
        tr_cache = extract_penultimate_cache(ema_model, tr_batches, device, head_attr='event_head_7', verify_atol=1e-6)
        del tr_batches
        finetune_head_on_cached_latents(head, tr_cache, dro_cfg, device)
        verify_backbone_bit_identity(ema_model, snap, head_attr='event_head_7')
        torch.save({'model': ema_model.state_dict(), 'config': model_cfg, 'pass': best_pass, 'dro': True}, seed_dir / 'best_ema_dro.pt')

        t8_dro = predict_gn2_items_fp32(ema_model, tune8k_items, bounds, device, batch_size=64)['probs']
        np.save(seed_dir / f'probs_tune8k_dro_seed{seed}.npy', t8_dro)
        ood_dro = {f'ood_{ov}': predict_gn2_items_fp32(ema_model, o_items, bounds, device, batch_size=64)['probs'] for ov, (o_items, _) in ood_pre.items()}
        st_dro = {f'stress_{sn}': predict_gn2_items_fp32(ema_model, stress_items[sn], bounds, device, batch_size=64)['probs'] for sn, _, _ in STRESS_SPECS}
        np.savez_compressed(seed_dir / f'ood_and_stress_dro_seed{seed}.npz', **ood_dro, **st_dro)

        # Observation-only 64-event replay & truth invariance check
        sh80 = load_authorized_split_shards(CLEAN_ROOT / 'data', 'validation', stage=2)[0]
        obs80 = pq.read_table(sh80 / 'observations.parquet').to_pylist()[:64]
        obs_items_64 = [
            extract_tracklet_gn2_item(i, ob, None, compute_observable_containment(ob, geom, bounds, BASE_OBS_CFG), geom, bounds, BASE_OBS_CFG)
            for i, ob in enumerate(obs80)
        ]
        p_rep = predict_gn2_items_fp32(ema_model, obs_items_64, bounds, device, batch_size=64)['probs']
        rep_diff = float(np.max(np.abs(p_rep - st_dro['stress_nominal'][:64])))
        if rep_diff > 1e-6:
            raise AssertionError(f'Stage-II replay diff {rep_diff:.3e} > 1e-6')
        b_chk = move_batch_to_device(collate_tracklet_gn2(obs_items_64[:8], mirror_rng=None, y_scint_center=yc, n_bars=nb), device)
        inv = verify_inference_truth_invariance(ema_model, b_chk, atol=0.0)

        man = {
            'status': 'complete',
            'seed': int(seed),
            'n_params': n_params,
            'best_pass': best_pass,
            'best_tune_screen_NLL': best_nll,
            'peak_vram_MiB': peak_vram,
            'total_optimizer_steps': global_step,
            'max_cache_equiv_diff': tr_cache['max_verify_diff'],
            'obs_only_replay_max_abs_diff': rep_diff,
            'truth_invariance_max_abs_diff': inv['max_abs_diff'],
            'wall_seconds': float(time.perf_counter() - t0),
            'UTC_started': utc_started,
            'UTC_finished': datetime.now(timezone.utc).isoformat(),
            'history': history,
        }
        write_json(seed_dir / 'manifest.json', man)
        out_all[f'seed_{seed}'] = man
        del model, ema_model, tr_cache
        gc.collect()
        torch.cuda.empty_cache()

    return out_all


def run_stage2_aggregate(variant: str | None = None) -> dict[str, Any]:
    """Aggregate Stage-II 3-seed promoted GN2 variant + `50/50 Blend` with DualRegime + KL-DRO."""
    for sp in ('train', 'validation', 'tune_screen', 'tune6k', 'ood'):
        assert_unsealed_split(sp, stage=2)

    if variant is None:
        s1_res_p = EXP_ROOT / 'stage1_l1_gn2_results.json'
        if s1_res_p.is_file():
            variant = str(json.loads(s1_res_p.read_text()).get('stage1_gates', {}).get('promoted_variant') or 'gn2_full')
        else:
            variant = 'gn2_full'
    gn2_s2_dir = STAGE2_DIR if variant == 'gn2_full' else (STAGE2_DIR / variant)

    tune_meta = pd.read_parquet(assert_unsealed_path(CLEAN_ROOT / 'validation_meta/tune_truth_meta.parquet'))
    scr_df = pd.read_parquet(assert_unsealed_path(CLEAN_ROOT / 'data/tune_screen.parquet'), columns=KEY)
    sidecar_df = pd.read_parquet(assert_unsealed_path(CLEAN_ROOT / 'truth_sidecar/event_sidecar.parquet'))

    ref_idx = pd.MultiIndex.from_frame(tune_meta[KEY])
    in_screen = np.asarray(ref_idx.isin(pd.MultiIndex.from_frame(scr_df[KEY])), dtype=bool)
    tune6k_m = ~in_screen

    labels = tune_meta['label'].to_numpy(dtype=np.int64)
    obs_dom = tune_meta['obs_domain_id'].to_numpy(dtype=np.int64)
    pc_m = tune_meta['pixel_centered'].to_numpy(dtype=bool)
    fp_m = tune_meta['front_pinpoint_pixel_centered'].to_numpy(dtype=bool)
    interm_m = pc_m & (~fp_m)
    geom_w = tune_meta['geom_weight'].to_numpy(dtype=np.float64)
    ev_tune = tune_meta['energy_nu_GeV'].to_numpy(dtype=np.float64)
    common_e_m = (ev_tune >= 50.0) & (ev_tune <= 2312.0)
    fam_codes = pd.MultiIndex.from_frame(tune_meta[FAMILY]).factorize()[0]

    s_pos = pd.MultiIndex.from_frame(sidecar_df[KEY]).get_indexer(ref_idx)
    sc_al = sidecar_df.iloc[s_pos].reset_index(drop=True)
    v_true = sc_al[['vertex_x_mm', 'vertex_y_mm', 'vertex_z_mm']].to_numpy(dtype=np.float64)
    v_valid = sc_al['vertex_valid'].to_numpy(dtype=np.int64) > 0

    seed_p_raw = [np.load(gn2_s2_dir / f'seed_{s}/probs_tune8k_seed{s}.npy').astype(np.float64) for s in STAGE2_SEEDS]
    seed_p_dro = [np.load(gn2_s2_dir / f'seed_{s}/probs_tune8k_dro_seed{s}.npy').astype(np.float64) for s in STAGE2_SEEDS]
    seed_vp = [np.load(gn2_s2_dir / f'seed_{s}/v_pv_tune8k_seed{s}.npy').astype(np.float64) for s in STAGE2_SEEDS]
    seed_sc = [np.load(gn2_s2_dir / f'seed_{s}/scale_pv_tune8k_seed{s}.npy').astype(np.float64) for s in STAGE2_SEEDS]

    p_ens_raw = np.mean(seed_p_raw, axis=0)
    p_ens_dro = np.mean(seed_p_dro, axis=0)
    vp_ens = np.mean(seed_vp, axis=0)
    vs_ens = np.sqrt(np.mean([s ** 2 for s in seed_sc], axis=0))

    seed_ba4s = [_ba4_and_nll(labels[tune6k_m], p[tune6k_m])['BA4_pct'] for p in seed_p_raw]
    seed_nlls = [_ba4_and_nll(labels[tune6k_m], p[tune6k_m])['NLL'] for p in seed_p_raw]

    # Load DualRegimeModalNet Stage-II raw & DRO Tune8k probs for 50/50 Blend
    dual_raw = np.load(STAGE2_S326_DUAL_DIR / 'probs_tune8k_ens3_raw.npy').astype(np.float64)
    dual_dro = np.load(STAGE2_S327_DRO_DIR / 'dual_regime_modalnet_dro_tune8k_ens3_raw.npy').astype(np.float64)
    blend_gn2_dual_raw = 0.5 * (p_ens_raw + dual_raw)
    blend_gn2_dual_dro = 0.5 * (p_ens_dro + dual_dro)

    # Fit tune_screen scalar temperature & calibrated probs cleanly on tune_screen (N=2,000) only
    t_raw = _fit_scalar_temperature_on_screen(p_ens_raw[in_screen], labels[in_screen])
    p_cal_raw = _apply_scalar_temperature(p_ens_raw, t_raw['inv_T'])
    t_dro = _fit_scalar_temperature_on_screen(p_ens_dro[in_screen], labels[in_screen])
    p_cal_dro = _apply_scalar_temperature(p_ens_dro, t_dro['inv_T'])
    t_b_dro = _fit_scalar_temperature_on_screen(blend_gn2_dual_dro[in_screen], labels[in_screen])
    p_cal_blend_dro = _apply_scalar_temperature(blend_gn2_dual_dro, t_b_dro['inv_T'])

    t6k_slices = {
        'All_Tune6k': tune6k_m,
        'D1_obs': tune6k_m & (obs_dom == 1),
        'D2_obs': tune6k_m & (obs_dom == 2),
        'D3_obs': tune6k_m & (obs_dom == 3),
        'Pinpoint_PixXY': tune6k_m & pc_m,
        'Front30': tune6k_m & fp_m,
        'Interm150': tune6k_m & interm_m,
        'Common_E_50_2312': tune6k_m & common_e_m,
    }
    vx = tune_meta['vertex_x_mm'].to_numpy(dtype=np.float64)
    vy = tune_meta['vertex_y_mm'].to_numpy(dtype=np.float64)
    vz = tune_meta['vertex_z_mm'].to_numpy(dtype=np.float64)
    pc_box = tune6k_m & (vx >= -113.0) & (vx <= 112.99) & (vy >= -90.0) & (vy <= 65.99) & (vz >= -2986.99) & (vz <= -2486.99)
    lg_box = tune6k_m & (vx >= -118.0) & (vx <= 272.0) & (vy >= -207.0) & (vy <= 183.0) & (vz >= -2986.99) & (vz <= -2004.88)

    def _eval_t6k(p_t8: np.ndarray) -> dict[str, Any]:
        s_tau = np.log(np.clip(p_t8[:, 2], 1e-8, 1.0)) - np.log(np.clip(1.0 - p_t8[:, 2], 1e-8, 1.0))
        return {
            'slices': {sn: _ba4_and_nll(labels[mk], p_t8[mk]) for sn, mk in t6k_slices.items()},
            'mass_weighted_2t_BA4_pct': _weighted_ba4_pct(labels[tune6k_m], p_t8[tune6k_m], geom_w[tune6k_m]),
            'within_slice_1pct_fpr': {
                sn: _within_slice_tail_eval(s_tau, labels, t6k_slices[sn], 0.01)
                for sn in ('All_Tune6k', 'D1_obs', 'Pinpoint_PixXY', 'Front30')
            },
            'within_slice_0p1pct_fpr': {
                sn: _within_slice_tail_eval(s_tau, labels, t6k_slices[sn], 0.001)
                for sn in ('All_Tune6k', 'D1_obs')
            },
            'split_half_1pct_fpr': {
                sn: _split_half_tail_eval(s_tau, labels, tune_meta, t6k_slices[sn], 0.01)
                for sn in ('All_Tune6k', 'D1_obs', 'Pinpoint_PixXY')
            },
            'collaboration_matched_slices_sec12': {
                'All_non_tau': _collaboration_section12_slice(p_t8, labels, tune6k_m),
                'Truth_Mahlstedt_PC': _collaboration_section12_slice(
                    p_t8, labels, pc_box, purity_target=0.90, natural_prior=(0.1145, 0.6397, 0.2459)
                ),
                'Truth_Mahlstedt_LG': _collaboration_section12_slice(
                    p_t8, labels, lg_box, purity_target=0.80, natural_prior=(0.1195, 0.6399, 0.2406)
                ),
                'D1_obs_non_tau': _collaboration_section12_slice(
                    p_t8, labels, tune6k_m & (obs_dom == 1), purity_target=0.90, natural_prior=(0.1145, 0.6397, 0.2459)
                ),
            },
        }

    pv_audit: dict[str, Any] = {}
    for rtag, rmk in (
        ('All_active', tune6k_m & v_valid),
        ('Pinpoint_PixXY', tune6k_m & v_valid & pc_m),
        ('Front30', tune6k_m & v_valid & fp_m),
        ('Interm150', tune6k_m & v_valid & interm_m),
        ('D1_obs', tune6k_m & v_valid & (obs_dom == 1)),
        ('D2_obs', tune6k_m & v_valid & (obs_dom == 2)),
    ):
        dz = np.abs(vp_ens[rmk, 2] - v_true[rmk, 2])
        d3 = np.linalg.norm(vp_ens[rmk] - v_true[rmk], axis=1)
        pulls = compute_student_t_pulls(
            torch.from_numpy(vp_ens[rmk]), torch.from_numpy(vs_ens[rmk]),
            torch.from_numpy(v_true[rmk]), torch.ones(int(rmk.sum()), dtype=torch.int64), df=3.0,
        )
        pv_audit[rtag] = {'count': int(rmk.sum()), 'median_abs_dz_mm': float(np.median(dz)), 'median_3d_mm': float(np.median(d3)), **pulls}

    # 6 OOD & 7 Stress
    ood_summary: dict[str, Any] = {}
    for ov in ALL_STAGE2_OOD:
        _, yo = _load_ood_observations(ov)
        p_o_raw = np.mean([np.load(gn2_s2_dir / f'seed_{s}/ood_and_stress_seed{s}.npz')[f'ood_{ov}'].astype(np.float64) for s in STAGE2_SEEDS], axis=0)
        p_o_dro = np.mean([np.load(gn2_s2_dir / f'seed_{s}/ood_and_stress_dro_seed{s}.npz')[f'ood_{ov}'].astype(np.float64) for s in STAGE2_SEEDS], axis=0)
        rec = {}
        for tag_m, po in (('gn2_raw', p_o_raw), ('gn2_dro', p_o_dro)):
            if len(np.unique(yo)) >= 4:
                rec[tag_m] = _ba4_and_nll(yo, po)
            else:
                rec[tag_m] = {
                    'n_events': int(len(yo)),
                    'pred_tau_argmax_pct': float((po.argmax(1) == 2).mean() * 100.0),
                    'p_tau_gt_0p5_pct': float((po[:, 2] > 0.5).mean() * 100.0),
                    'mean_p_tau': float(po[:, 2].mean()),
                }
        ood_summary[ov] = rec

    sh80 = load_authorized_split_shards(CLEAN_ROOT / 'data', 'validation', stage=2)[0]
    prov80 = pd.read_parquet(sh80 / 'provenance.parquet', columns=KEY + ['label'])
    pos80 = pd.MultiIndex.from_frame(tune_meta[KEY]).get_indexer(pd.MultiIndex.from_frame(prov80[KEY]))
    y80 = tune_meta.iloc[pos80]['label'].to_numpy(dtype=np.int64)
    p_nom_st = np.mean([np.load(gn2_s2_dir / f'seed_{s}/ood_and_stress_seed{s}.npz')['stress_nominal'].astype(np.float64) for s in STAGE2_SEEDS], axis=0)
    stress_summary: dict[str, Any] = {}
    nom_ba4 = _ba4_and_nll(y80, p_nom_st)['BA4_pct']
    for sn, _, _ in STRESS_SPECS:
        ps = np.mean([np.load(gn2_s2_dir / f'seed_{s}/ood_and_stress_seed{s}.npz')[f'stress_{sn}'].astype(np.float64) for s in STAGE2_SEEDS], axis=0)
        ms = _ba4_and_nll(y80, ps)
        ms['delta_BA4_vs_nominal_pp'] = float(ms['BA4_pct'] - nom_ba4)
        ms['mean_TV_vs_nominal'] = float(0.5 * np.abs(ps - p_nom_st).sum(axis=1).mean())
        stress_summary[sn] = ms

    # Paired family-bootstrap vs Stage-326 TrackletIPFormer, Stage-327 50/50 Blend + KL-DRO,
    # and L1 zero-confounding 3-seed control (`l1_ctrl_no_gn2`) if present.
    s326_trk_t6k = np.load(STAGE2_S326_TRACKLET_DIR / 'probs_tune6k_ens.npy').astype(np.float64)
    s327_blend_dro_t8 = np.load(STAGE2_S327_DRO_DIR / 'blend_50_50_dro_tune8k_raw.npy').astype(np.float64)
    s326_blend_raw_t8 = 0.5 * (
        np.load(STAGE2_S327_DRO_DIR / 'tracklet_ipformer_pre_tune8k_ens3_raw.npy').astype(np.float64) + dual_raw
    )
    ctrl_dir = STAGE2_DIR / 'l1_ctrl_no_gn2'
    has_ctrl_s2 = all((ctrl_dir / f'seed_{s}/probs_tune8k_seed{s}.npy').is_file() for s in STAGE2_SEEDS)
    ctrl_ens_raw = (
        np.mean([np.load(ctrl_dir / f'seed_{s}/probs_tune8k_seed{s}.npy').astype(np.float64) for s in STAGE2_SEEDS], axis=0)
        if has_ctrl_s2 else None
    )
    ctrl_ens_dro = (
        np.mean([np.load(ctrl_dir / f'seed_{s}/probs_tune8k_dro_seed{s}.npy').astype(np.float64) for s in STAGE2_SEEDS], axis=0)
        if has_ctrl_s2 else None
    )
    boot = {}
    comp_models: list[tuple[str, np.ndarray]] = [
        ('s326_blend_raw', s326_blend_raw_t8),
        ('s327_blend_dro', s327_blend_dro_t8),
    ]
    if ctrl_ens_raw is not None:
        comp_models.append(('l1_ctrl_no_gn2_ens_raw', ctrl_ens_raw))
    if ctrl_ens_dro is not None:
        comp_models.append(('l1_ctrl_no_gn2_ens_dro', ctrl_ens_dro))
    for bname, pb in comp_models:
        for sn in ('All_Tune6k', 'D1_obs', 'Pinpoint_PixXY', 'Front30'):
            boot[f'gn2_blend_dro - {bname} ({sn})'] = _paired_family_bootstrap_full(
                blend_gn2_dual_dro, pb, labels, fam_codes, t6k_slices[sn], n_boot=1000, seed=20263250
            )
            boot[f'gn2_ens_raw - {bname} ({sn})'] = _paired_family_bootstrap_full(
                p_ens_raw, pb, labels, fam_codes, t6k_slices[sn], n_boot=1000, seed=20263250
            )
            boot[f'gn2_ens_dro - {bname} ({sn})'] = _paired_family_bootstrap_full(
                p_ens_dro, pb, labels, fam_codes, t6k_slices[sn], n_boot=1000, seed=20263250
            )
    # Direct gn2_ens_raw vs s326 TrackletIPFormer on Tune6k
    boot['gn2_ens_raw - s326_TrackletIPFormer (All_Tune6k)'] = _paired_family_bootstrap_full(
        p_ens_raw[tune6k_m], s326_trk_t6k, labels[tune6k_m], fam_codes[tune6k_m],
        np.ones(int(tune6k_m.sum()), dtype=bool), n_boot=1000, seed=20263250
    )

    ctrl_ood_summary: dict[str, Any] = {}
    if has_ctrl_s2:
        for ov in ALL_STAGE2_OOD:
            _, yo = _load_ood_observations(ov)
            po_r = np.mean([np.load(ctrl_dir / f'seed_{s}/ood_and_stress_seed{s}.npz')[f'ood_{ov}'].astype(np.float64) for s in STAGE2_SEEDS], axis=0)
            po_d = np.mean([np.load(ctrl_dir / f'seed_{s}/ood_and_stress_dro_seed{s}.npz')[f'ood_{ov}'].astype(np.float64) for s in STAGE2_SEEDS], axis=0)
            rec_c = {}
            for tm_c, po_c in (('ctrl_raw', po_r), ('ctrl_dro', po_d)):
                if len(np.unique(yo)) >= 4:
                    rec_c[tm_c] = _ba4_and_nll(yo, po_c)
                else:
                    rec_c[tm_c] = {
                        'n_events': int(len(yo)),
                        'pred_tau_argmax_pct': float((po_c.argmax(1) == 2).mean() * 100.0),
                        'p_tau_gt_0p5_pct': float((po_c[:, 2] > 0.5).mean() * 100.0),
                        'mean_p_tau': float(po_c[:, 2].mean()),
                    }
            ctrl_ood_summary[ov] = rec_c

    np.save(STAGE2_DIR / 'probs_tune8k_gn2_ens3_raw.npy', p_ens_raw.astype(np.float32))
    np.save(STAGE2_DIR / 'probs_tune8k_gn2_ens3_dro.npy', p_ens_dro.astype(np.float32))
    np.save(STAGE2_DIR / 'probs_tune8k_gn2_blend_dual_raw.npy', blend_gn2_dual_raw.astype(np.float32))
    np.save(STAGE2_DIR / 'probs_tune8k_gn2_blend_dual_dro.npy', blend_gn2_dual_dro.astype(np.float32))

    out_doc = {
        'UTC': datetime.now(timezone.utc).isoformat(),
        'variant': variant,
        'stage': f'Stage-II {variant} (L1_Train32k x 3 seeds {{17,42,2026}}; Tune6k N=6,000)',
        'cross_seed_summary': {
            'tune6k_BA4_mean_pct': float(np.mean(seed_ba4s)),
            'tune6k_BA4_std_pp': float(np.std(seed_ba4s, ddof=0)),
            'tune6k_NLL_mean': float(np.mean(seed_nlls)),
            'tune6k_NLL_std': float(np.std(seed_nlls, ddof=0)),
        },
        'temperatures': {'raw': t_raw, 'dro': t_dro, 'blend_dro': t_b_dro},
        'gn2_ens3_raw_tune6k': _eval_t6k(p_ens_raw),
        'gn2_ens3_cal_tune6k': _eval_t6k(p_cal_raw),
        'gn2_ens3_dro_tune6k': _eval_t6k(p_ens_dro),
        'gn2_ens3_dro_cal_tune6k': _eval_t6k(p_cal_dro),
        'blend_gn2_dual_raw_tune6k': _eval_t6k(blend_gn2_dual_raw),
        'blend_gn2_dual_dro_tune6k': _eval_t6k(blend_gn2_dual_dro),
        'blend_gn2_dual_dro_cal_tune6k': _eval_t6k(p_cal_blend_dro),
        'l1_ctrl_no_gn2_ens3_raw_tune6k': _eval_t6k(ctrl_ens_raw) if ctrl_ens_raw is not None else None,
        'l1_ctrl_no_gn2_ens3_dro_tune6k': _eval_t6k(ctrl_ens_dro) if ctrl_ens_dro is not None else None,
        'l1_ctrl_no_gn2_ood_evaluation': ctrl_ood_summary if has_ctrl_s2 else None,
        'pv_audit_tune6k': pv_audit,
        'ood_evaluation': ood_summary,
        'shard080_stress_tests': stress_summary,
        'paired_family_bootstrap_1000_seed20263250': boot,
    }
    write_json(STAGE2_DIR / 'stage2_l1_gn2_results.json', out_doc)
    return out_doc


# ===========================================================================
# HTCondor Submission & EOS Hygiene
# ===========================================================================
def _submit_condor_job(
    jname: str,
    cmd_args: str,
    *,
    cpus: int,
    mem: str,
    disk: str,
    flavour: str,
    use_gpu: bool,
    min_gpu_mem_mb: int = 10000,
    min_cap: float = 8.0,
) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[4]
    EXP_ROOT.mkdir(parents=True, exist_ok=True)
    (EXP_ROOT / 'logs').mkdir(parents=True, exist_ok=True)
    (EXP_ROOT / 'code').mkdir(parents=True, exist_ok=True)
    (EXP_ROOT / 'runtime').mkdir(parents=True, exist_ok=True)
    CONDOR_DIR.mkdir(parents=True, exist_ok=True)

    archive = EXP_ROOT / 'code' / f'nextgen_l1_gn2_{jname}_{int(time.time())}.tar.gz'
    _package_source(repo, archive)
    chosen, _, _ = choose_schedd(EXP_ROOT)

    job_dir = CONDOR_DIR / jname
    job_dir.mkdir(parents=True, exist_ok=True)
    script = ['#!/bin/bash', 'set -o pipefail', 'which() { command -v "$@"; }; export -f which']
    if use_gpu:
        script.extend([
            'GPU_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d " ")',
            'if [ "$GPU_CAP" = "7.0" ]; then',
            '  source /cvmfs/sft.cern.ch/lcg/views/LCG_107_cuda/x86_64-el9-gcc11-opt/setup.sh',
            'else',
            '  source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh',
            'fi',
        ])
    else:
        script.append('source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh')
    script.extend([
        'set -e',
        'ls /eos/home-x/xcheng/FASER/Pinpoint/results >/dev/null',
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1',
        'export PINPOINT_V3_SCRATCH="${_CONDOR_SCRATCH_DIR:-/tmp}/ng_gn2_${PINPOINT_JOB_ID}"',
        'mkdir -p "$PINPOINT_V3_SCRATCH/code"',
        f'tar -xzf {shlex.quote(str(archive))} -C "$PINPOINT_V3_SCRATCH/code"',
        'export PYTHONPATH="$PINPOINT_V3_SCRATCH/code/src:${PYTHONPATH:-}"',
        'cd "$PINPOINT_V3_SCRATCH"',
        f'exec python3 -u -m analysis.flavour.nextgen.run_l1_gn2_tracklet {cmd_args}',
    ])
    wrapper = job_dir / 'run.sh'
    wrapper.write_text('\n'.join(script) + '\n')
    wrapper.chmod(0o755)

    sub_lines = [
        'universe = vanilla',
        f'executable = {wrapper}',
        f'initialdir = {job_dir}',
        f'output = {EXP_ROOT}/logs/{jname}.$(ClusterId).out',
        f'error = {EXP_ROOT}/logs/{jname}.$(ClusterId).err',
        f'log = {EXP_ROOT}/logs/{jname}.$(ClusterId).log',
        'environment = "PINPOINT_JOB_ID=$(ClusterId).$(Process)"',
        f'request_cpus = {cpus}',
        f'request_memory = {mem}',
        f'request_disk = {disk}',
        'requirements = (OpSysAndVer == "AlmaLinux9")',
        f'+JobFlavour = "{flavour}"',
        'should_transfer_files = YES',
        'when_to_transfer_output = ON_EXIT',
        'transfer_output_files = ""',
    ]
    if use_gpu:
        cap_clause = f'((Capability >= {min_cap:.1f}) || ((Capability =?= UNDEFINED) && regexp("^NVIDIA H100.* MIG ", DeviceName)))'
        req_g = f'(GlobalMemoryMb >= {int(min_gpu_mem_mb)}) && {cap_clause}' if min_gpu_mem_mb > 0 else cap_clause
        sub_lines.extend([
            'request_gpus = 1',
            f'require_gpus = {req_g}',
        ])
    sub_lines.append('queue 1')
    sub_path = job_dir / 'job.sub'
    sub_path.write_text('\n'.join(sub_lines) + '\n')
    resp = subprocess.run(['condor_submit', '-name', chosen, str(sub_path)], capture_output=True, text=True)
    rec = {'job': jname, 'schedd': chosen, 'stdout': resp.stdout.strip(), 'stderr': resp.stderr.strip(), 'returncode': resp.returncode}
    write_json(EXP_ROOT / 'runtime' / f'{jname}.submission.json', rec)
    if resp.returncode != 0:
        raise RuntimeError(f'condor_submit failed for {jname}: {resp.stderr}')
    return rec


def cleanup_temporary_caches() -> dict[str, bool]:
    """Remove temporary EOS item caches after training/aggregation completes."""
    removed = {}
    for d in (STAGE1_CACHE_DIR, STAGE2_CACHE_DIR):
        existed = d.exists()
        if existed:
            shutil.rmtree(d, ignore_errors=True)
        removed[str(d)] = existed
    return removed


def main() -> None:
    guard = SealAuditGuard()
    guard.__enter__()
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--s1-preextract-ood', action='store_true')
        parser.add_argument('--s1-preextract', action='store_true')
        parser.add_argument('--s1-submit-preextract', action='store_true')
        parser.add_argument('--part', type=int, default=-1)
        parser.add_argument('--s1-train', action='store_true')
        parser.add_argument('--variants', nargs='+', default=['gn2_full'])
        parser.add_argument('--s1-submit', action='store_true')
        parser.add_argument('--s1-pack', action='store_true', help='Pack specified variants onto 1 GPU job')
        parser.add_argument('--min-cap', type=float, default=8.0)
        parser.add_argument('--min-gpu-mem-mb', type=int, default=0)
        parser.add_argument('--s1-aggregate', action='store_true')
        parser.add_argument('--s2-preextract', action='store_true')
        parser.add_argument('--s2-submit-preextract', action='store_true')
        parser.add_argument('--s2-train', action='store_true')
        parser.add_argument('--s2-variant', type=str, default='gn2_full')
        parser.add_argument('--seeds', nargs='+', type=int, default=[17, 42, 2026])
        parser.add_argument('--s2-submit', action='store_true')
        parser.add_argument('--s2-aggregate', action='store_true')
        parser.add_argument('--cleanup-caches', action='store_true')
        args = parser.parse_args()

        is_stage2 = any([args.s2_preextract, args.s2_submit_preextract, args.s2_train, args.s2_submit, args.s2_aggregate])
        if not is_stage2:
            install_stage1_only_hook()

        if args.s1_preextract_ood:
            print(json.dumps(run_preextract_ood6(), indent=2))
        elif args.s1_preextract:
            print(json.dumps(run_stage1_preextract(part=args.part), indent=2))
        elif args.s1_submit_preextract:
            parts = list(range(9)) if args.part < 0 else [args.part]
            subs = {
                f's1_pre_p{p}': _submit_condor_job(
                    f's1_pre_p{p}', f'--s1-preextract --part {p}',
                    cpus=8, mem='16GB', disk='12GB', flavour='microcentury', use_gpu=False,
                )
                for p in parts
            }
            print(json.dumps(subs, indent=2))
        elif args.s1_train:
            res = run_stage1_train(args.variants)
            print(json.dumps({k: v['internal_val_eval_all'] for k, v in res.items()}, indent=2))
        elif args.s1_submit:
            if args.s1_pack:
                v_arg = ' '.join(shlex.quote(v) for v in args.variants)
                sub = _submit_condor_job(
                    's1_pack_' + '_'.join(args.variants)[:28],
                    f'--s1-train --variants {v_arg}',
                    cpus=6, mem='24GB', disk='24GB', flavour='workday', use_gpu=True,
                    min_gpu_mem_mb=args.min_gpu_mem_mb, min_cap=args.min_cap,
                )
                print(json.dumps(sub, indent=2))
            else:
                subs = {
                    v: _submit_condor_job(
                        f's1_{v}', f'--s1-train --variants {shlex.quote(v)}',
                        cpus=6, mem='24GB', disk='20GB', flavour='longlunch', use_gpu=True,
                        min_gpu_mem_mb=args.min_gpu_mem_mb, min_cap=args.min_cap,
                    )
                    for v in args.variants
                }
                print(json.dumps(subs, indent=2))
        elif args.s1_aggregate:
            doc = run_stage1_aggregate()
            print(json.dumps({'stage1_gates': doc['stage1_gates'], 'out': str(EXP_ROOT / 'stage1_l1_gn2_results.json')}, indent=2))
        elif args.s2_preextract:
            print(json.dumps(run_stage2_preextract(part=args.part), indent=2))
        elif args.s2_submit_preextract:
            parts = list(range(11)) if args.part < 0 else [args.part]
            subs = {
                f's2_pre_p{p}': _submit_condor_job(
                    f's2_pre_p{p}', f'--s2-preextract --part {p}',
                    cpus=8, mem='16GB', disk='12GB', flavour='microcentury', use_gpu=False,
                )
                for p in parts
            }
            print(json.dumps(subs, indent=2))
        elif args.s2_train:
            res = run_stage2_train(args.seeds, variant=args.s2_variant)
            print(json.dumps({k: v['best_tune_screen_NLL'] for k, v in res.items()}, indent=2))
        elif args.s2_submit:
            s_arg = ' '.join(str(s) for s in args.seeds)
            j_s2 = f's2_{args.s2_variant}_s' + '_'.join(str(s) for s in args.seeds)
            sub = _submit_condor_job(
                j_s2, f'--s2-train --s2-variant {shlex.quote(args.s2_variant)} --seeds {s_arg}',
                cpus=6, mem='24GB', disk='24GB', flavour='workday', use_gpu=True,
                min_gpu_mem_mb=args.min_gpu_mem_mb, min_cap=args.min_cap,
            )
            print(json.dumps(sub, indent=2))
        elif args.s2_aggregate:
            v_agg = None if args.s2_variant == 'gn2_full' else args.s2_variant
            doc = run_stage2_aggregate(variant=v_agg)
            print(json.dumps({'cross_seed_summary': doc['cross_seed_summary'], 'out': str(STAGE2_DIR / 'stage2_l1_gn2_results.json')}, indent=2))
        elif args.cleanup_caches:
            print(json.dumps(cleanup_temporary_caches(), indent=2))
        else:
            parser.error('Specify an action flag')
    finally:
        guard.__exit__(None, None, None)


if __name__ == '__main__':
    main()
