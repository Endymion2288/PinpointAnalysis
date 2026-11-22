"""Stage-I & Stage-II Runner for PinpointUnifiedNet (Experiment N1).

Features:
  - Strict SealAuditGuard(stage=1/2) isolation (zero access to Cal-A/Cut-B/Confirm/GENIE 016..028;
    Stage-I additionally blocks Tune8k/080..099).
  - `/dev/shm` fast cache staging (cleaned on exit) + multi-worker deterministic Y-mirror DataLoader.
  - Stage-I (`internal_train` 25.6k x 20 passes = 4,000 steps @ eff_bs=128, seed=42) + 3 single-factor
    ablations (`abl_no_recoil_hem`, `abl_no_online_spec_curriculum`, `abl_no_tracklet_rab`).
  - Stage-I acceptance gate verification vs Stage-I 50/50 Blend (58.54% / 0.9593) & Sec 10.4 fallback.
  - Frozen Stage-II (`Train32k` x 3 seeds {17, 42, 2026}, 5,000 steps @ eff_bs=128), `tune_screen`
    calibration, single-shot `Tune6k` + 6 OOD + 7 Shard-080 Stress + Sec 12 BA3 + 1000x paired bootstrap.
  - Exports latent/logit caches (`fused_global`, `logits7`, `probs4`) for downstream DRO tail optimization.
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
import tarfile
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler

from analysis.flavour.inventory import write_json
from analysis.flavour.nextgen.dual_regime_modalnet import (
    apply_y_mirror_dual_regime_item,
    collate_dual_regime,
)
from analysis.flavour.nextgen.protocols import compute_student_t_pulls
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
    _clopper_pearson_ci95,
    _collaboration_section12_slice,
    _collaboration_t3_eval,
    _fit_scalar_temperature_on_screen,
    _load_ood_observations,
    _paired_family_bootstrap_full,
    _perturb_observation_for_stress,
    _set_ieee_fp32,
    _split_half_tail_eval,
    _split_half_tail_eval_multi,
    _weighted_ba4_pct,
    _within_slice_tail_eval,
    stage_authorized_train_shards,
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
from analysis.flavour.nextgen.unified_pinpoint_net import (
    PinpointUnifiedNet,
    compute_unified_loss,
    compute_unified_window_denominators,
    extract_unified_item,
)
from analysis.flavour.v2.partitions import FAMILY, KEY
from analysis.flavour.v2.submit import choose_schedd, gpu_requirement
from analysis.flavour.v3.common.domains import (
    OBS_DOMAIN_NAMES,
    compute_observable_containment,
    cumulative_radiation_length,
    detector_bounds,
)
from analysis.flavour.v3.v2_pinpointformer.train import move_batch_to_device
from analysis.flavour.v3.v4_tau.ensemble import _apply_domain_convex_blend, _fit_domain_convex_blend
from analysis.flavour.v3.v6_clean_sim.evaluate_stage317 import _select_seed_prediction_df


EXP_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/exp_unified_net')
STAGE1_CACHE_DIR = EXP_ROOT / '_stage1_item_cache'
STAGE2_DIR = EXP_ROOT / 'stage2'
STAGE2_CACHE_DIR = STAGE2_DIR / '_stage2_item_cache'
CONDOR_DIR = Path.home() / 'condor/Pinpoint/stage327_n1'

STAGE1_S326_TRACKLET_DIR = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/exp_tracklet_ipformer')
STAGE1_S326_DUAL_DIR = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/exp_dual_regime')

VARIANT_CONFIGS: dict[str, dict[str, Any]] = {
    's1_unified_pinpoint_net': {
        'use_recoil_hem': True,
        'enable_online_spec_curriculum': True,
        'use_tracklet_rab': True,
    },
    'abl_no_recoil_hem': {
        'use_recoil_hem': False,
        'enable_online_spec_curriculum': True,
        'use_tracklet_rab': True,
    },
    'abl_no_online_spec_curriculum': {
        'use_recoil_hem': True,
        'enable_online_spec_curriculum': False,
        'use_tracklet_rab': True,
    },
    'abl_no_tracklet_rab': {
        'use_recoil_hem': True,
        'enable_online_spec_curriculum': True,
        'use_tracklet_rab': False,
    },
}

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
    """Prefer `/dev/shm` if >= `min_free_gb` free, else `_CONDOR_SCRATCH_DIR` or `/tmp`."""
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
    """Copy sharded `.pkl` cache files to local `/dev/shm` (if >=14GB free) or condor scratch for fast reads."""
    fast_root = _resolve_fast_cache_root(min_free_gb=14.0)
    if fast_root != Path('/dev/shm'):
        scratch = os.environ.get('PINPOINT_V3_SCRATCH') or os.environ.get('_CONDOR_SCRATCH_DIR')
        if scratch and Path(scratch).is_dir():
            fast_root = Path(scratch)
        else:
            return src_dir, False
    dst_dir = Path(tempfile.mkdtemp(prefix=f'{prefix}_', dir=str(fast_root)))

    def _cleanup() -> None:
        shutil.rmtree(dst_dir, ignore_errors=True)

    atexit.register(_cleanup)
    for fn in filenames:
        src_f = src_dir / fn
        if src_f.is_file():
            shutil.copy2(src_f, dst_dir / fn)
    return dst_dir, True


def _deterministic_mirror_bit(seed: int, pass_idx: int, idx: int) -> bool:
    digest = hashlib.blake2b(f'{seed}:{pass_idx}:{idx}'.encode(), digest_size=8).digest()
    return (int.from_bytes(digest, 'little') & 1) == 1


class InMemoryUnifiedDataset(Dataset):
    """In-memory dataset with sidecar attachment and worker-count-invariant 50% Y-mirror aug."""

    def __init__(
        self,
        items: list[dict],
        event_sidecar: pd.DataFrame | None = None,
        *,
        mirror_seed: int | None = None,
        y_scint_center: float = -12.0,
        n_bars: int = 40,
    ) -> None:
        self.items = items
        self.mirror_seed = mirror_seed
        self.pass_idx = 0
        self.y_scint_center = float(y_scint_center)
        self.n_bars = int(n_bars)
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

    def set_pass(self, pass_idx: int) -> None:
        self.pass_idx = int(pass_idx)

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
        if self.mirror_seed is not None and _deterministic_mirror_bit(self.mirror_seed, self.pass_idx, idx):
            it = apply_y_mirror_dual_regime_item(it, y_scint_center=self.y_scint_center, n_bars=self.n_bars)
        return it


class _UnifiedCollator:
    """Picklable collator for multi-worker DataLoader."""

    def __init__(self, y_scint_center: float, n_bars: int) -> None:
        self.y_scint_center = float(y_scint_center)
        self.n_bars = int(n_bars)

    def __call__(self, xs: list[dict]) -> dict:
        return collate_dual_regime(xs, mirror_rng=None, y_scint_center=self.y_scint_center, n_bars=self.n_bars)


def _update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for pe, pm in zip(ema.parameters(), model.parameters()):
            pe.data.mul_(decay).add_(pm.data, alpha=1.0 - decay)
        for be, bm in zip(ema.buffers(), model.buffers()):
            be.copy_(bm)


def _extract_unified_shard_worker(task: tuple) -> list[dict]:
    if len(task) == 6:
        shard_str, geom, bounds, cfg, keys_filter_set, expected_split = task
    else:
        shard_str, geom, bounds, cfg, keys_filter_set = task
        expected_split = 'train'
    shard_p = Path(assert_unsealed_path(shard_str))
    prov = pd.read_parquet(shard_p / 'provenance.parquet', columns=KEY + FAMILY + ['split', 'label'])
    obs_list = pq.read_table(shard_p / 'observations.parquet').to_pylist()
    items = []
    for li, row in enumerate(prov.itertuples(index=False)):
        k = (int(row.run), int(row.chunk), int(row.row))
        if row.split != expected_split:
            continue
        if keys_filter_set is not None and k not in keys_filter_set:
            continue
        ob = obs_list[li]
        dm = compute_observable_containment(ob, geom, bounds, config=cfg)
        it = extract_unified_item(li, ob, int(row.label), dm, geom, bounds, cfg)
        it['run'] = int(row.run)
        it['chunk'] = int(row.chunk)
        it['row'] = int(row.row)
        it['family_hi'] = int(row.family_hi)
        it['family_lo'] = int(row.family_lo)
        items.append(it)
    return items


def _extract_unified_obs_chunk_worker(task: tuple) -> list[dict]:
    raw_obs, labels, start_idx, geom, bounds, cfg = task
    items = []
    for k, ob in enumerate(raw_obs):
        lbl = None if labels is None else int(labels[k])
        dm = compute_observable_containment(ob, geom, bounds, config=cfg)
        items.append(extract_unified_item(start_idx + k, ob, lbl, dm, geom, bounds, cfg))
    return items


def _parallel_extract_unified_obs_list(
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
        parts = list(pool.map(_extract_unified_obs_chunk_worker, tasks))
    else:
        parts = [_extract_unified_obs_chunk_worker(t) for t in tasks]
    return [it for part in parts for it in part]


def _build_unified_ood_items(
    ov: str,
    geom: dict,
    bounds: dict,
    pool: ProcessPoolExecutor | None = None,
) -> tuple[list[dict], np.ndarray]:
    raw_o, labels = _load_ood_observations(ov)
    items = _parallel_extract_unified_obs_list(raw_o, None, geom, bounds, BASE_OBS_CFG, pool=pool)
    return items, labels


def _build_unified_stress_items_from_internal_val(
    val_keys_400: pd.DataFrame,
    geom: dict,
    bounds: dict,
    pool: ProcessPoolExecutor | None = None,
) -> tuple[dict[str, list[dict]], np.ndarray]:
    """Build 7 stress perturbations on the first 400-event slice of `internal_val_eval`."""
    assert_unsealed_split('internal_val', stage=1)
    by_shard: dict[int, list[tuple[int, int, int]]] = {}
    for idx_pos, row in enumerate(val_keys_400.itertuples(index=False)):
        by_shard.setdefault(int(row.chunk), []).append((idx_pos, int(row.run), int(row.row)))

    raw_obs_400: list[dict | None] = [None] * len(val_keys_400)
    labels_400 = val_keys_400['label'].to_numpy(dtype=np.int64)
    for chunk_id, entries in by_shard.items():
        shard_p = assert_unsealed_path(CLEAN_ROOT / 'data/shards' / f'{chunk_id:03d}')
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
        stress_items[st_name] = _parallel_extract_unified_obs_list(pert_obs, labels_400, geom, bounds, cfg_use, pool=pool)
    return stress_items, labels_400


def _build_unified_shard080_stress_items(
    geom: dict,
    bounds: dict,
    pool: ProcessPoolExecutor | None = None,
) -> tuple[dict[str, list[dict]], np.ndarray, pd.DataFrame]:
    """Exact Stage-325 Shard-080 7-condition stress suite with monopixel-clean extract_unified_item."""
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
        stress_items[st_name] = _parallel_extract_unified_obs_list(pert_obs, None, geom, bounds, cfg_use, pool=pool)
    return stress_items, y_080, prov_080


@torch.no_grad()
def predict_unified_items_fp32(
    model: PinpointUnifiedNet,
    items: list[dict],
    bounds: dict,
    device: torch.device,
    batch_size: int = 64,
    *,
    return_latents: bool = False,
) -> dict[str, np.ndarray]:
    """Deterministic IEEE-FP32 inference returning probs4, logits7, v_pv_mm, scale_pv_mm, charm_prob (+ optional latents)."""
    _set_ieee_fp32(True)
    model.eval()
    probs_l, l7_l, v_l, s_l, c_l, qc_l, ztc_l, fg_l, na_l, nb_l = [], [], [], [], [], [], [], [], [], []
    yc, nb = float(bounds['y_scint_center']), int(bounds['n_bars'])
    for st in range(0, len(items), batch_size):
        sub = items[st : st + batch_size]
        batch = move_batch_to_device(collate_dual_regime(sub, mirror_rng=None, y_scint_center=yc, n_bars=nb), device)
        with torch.amp.autocast('cuda', enabled=False):
            out = model.forward_NextGen(batch)
        probs_l.append(out.probs4.float().cpu().numpy())
        v_l.append(out.v_pv_mm.float().cpu().numpy())
        s_l.append(out.scale_pv_mm.float().cpu().numpy())
        c_l.append(torch.sigmoid(out.charm_logit.float()).cpu().numpy())
        if return_latents:
            l7_l.append(out.logits7.float().cpu().numpy())
            qc_l.append(out.aux['q_bkg_charm'].float().cpu().numpy())
            ztc_l.append(out.aux['z_tau_c'].float().cpu().numpy())
            fg_l.append(out.aux['fused_global'].float().cpu().numpy())
            na_l.append(out.aux['nccc_active'].bool().cpu().numpy())
            nb_l.append(out.aux['nccc_boost'].float().cpu().numpy())

    res: dict[str, np.ndarray] = {
        'probs': np.concatenate(probs_l, axis=0).astype(np.float32),
        'v_pv_mm': np.concatenate(v_l, axis=0).astype(np.float32),
        'scale_pv_mm': np.concatenate(s_l, axis=0).astype(np.float32),
        'charm_prob': np.concatenate(c_l, axis=0).astype(np.float32),
    }
    if return_latents:
        res['logits7'] = np.concatenate(l7_l, axis=0).astype(np.float32)
        res['q_bkg_charm'] = np.concatenate(qc_l, axis=0).astype(np.float32)
        res['z_tau_c'] = np.concatenate(ztc_l, axis=0).astype(np.float32)
        res['fused_global'] = np.concatenate(fg_l, axis=0).astype(np.float32)
        res['nccc_active'] = np.concatenate(na_l, axis=0).astype(bool)
        res['nccc_boost'] = np.concatenate(nb_l, axis=0).astype(np.float32)
    return res


@torch.no_grad()
def _calibrate_pv_regime_scales_on_train(
    ema_model: PinpointUnifiedNet,
    tr_items: list[dict],
    tr_sidecar: pd.DataFrame,
    bounds: dict,
    device: torch.device,
    max_events: int = 8192,
) -> np.ndarray:
    """Fit 9-cell (3 si_regime x 3 obs_domain) Student-t(nu=3) p68(|pull|)=1.0 scale multipliers strictly on train."""
    _set_ieee_fp32(True)
    ema_model.eval()
    ema_model.pv_regime_scale_corr.fill_(1.0)

    n_sub = min(max_events, len(tr_items))
    sub_items = tr_items[:n_sub]
    sub_sc = tr_sidecar.iloc[:n_sub]
    v_true = np.column_stack([
        sub_sc['vertex_x_mm'].to_numpy(np.float64),
        sub_sc['vertex_y_mm'].to_numpy(np.float64),
        sub_sc['vertex_z_mm'].to_numpy(np.float64),
    ])
    v_valid = sub_sc['vertex_valid'].to_numpy(dtype=np.int64) > 0

    yc, nb = float(bounds['y_scint_center']), int(bounds['n_bars'])
    vh_l, sh_l, fg_l, hm_l, dom_l = [], [], [], [], []
    for st in range(0, n_sub, 32):
        chunk = sub_items[st : st + 32]
        batch = move_batch_to_device(collate_dual_regime(chunk, mirror_rng=None, y_scint_center=yc, n_bars=nb), device)
        with torch.amp.autocast('cuda', enabled=False):
            out = ema_model.forward_NextGen(batch)
        vh_l.append(out.v_pv_mm.float().cpu().numpy())
        sh_l.append(out.scale_pv_mm.float().cpu().numpy())
        fg_l.append(out.aux['front_gate'].float().cpu().numpy())
        hm_l.append(out.aux['has_mid_si'].bool().cpu().numpy())
        dom_l.append(batch['obs_domain_id'].long().cpu().numpy())

    vh = np.concatenate(vh_l, axis=0).astype(np.float64)
    sh = np.concatenate(sh_l, axis=0).astype(np.float64)
    fg = np.concatenate(fg_l, axis=0).astype(np.float64)
    hm = np.concatenate(hm_l, axis=0)
    dom = np.clip(np.concatenate(dom_l, axis=0), 1, 3) - 1

    norm_pull = np.abs(vh - v_true) / (np.maximum(sh, 1e-4) * 1.1969)
    corr_np = np.ones((3, 3, 3), dtype=np.float32)
    reg_masks = (
        v_valid & (fg > 0.5),
        v_valid & (fg <= 0.5) & hm,
        v_valid & (fg <= 0.5) & (~hm),
    )
    for r_idx, mk_r in enumerate(reg_masks):
        row_fallback = np.ones(3, dtype=np.float32)
        if int(mk_r.sum()) >= 30:
            q68_r = np.sqrt(np.quantile(norm_pull[mk_r], 0.6827, axis=0))
            row_fallback = np.clip(q68_r, 0.50, 2.0).astype(np.float32)
        for d_idx in range(3):
            mk = mk_r & (dom == d_idx)
            if int(mk.sum()) >= 30:
                q68 = np.sqrt(np.quantile(norm_pull[mk], 0.6827, axis=0))
                corr_np[r_idx, d_idx] = np.clip(q68, 0.50, 2.0).astype(np.float32)
            else:
                corr_np[r_idx, d_idx] = row_fallback
    ema_model.pv_regime_scale_corr.copy_(torch.from_numpy(corr_np).to(device))
    return corr_np


def _export_latent_cache(
    path: Path,
    pred: dict[str, np.ndarray],
    keys_df: pd.DataFrame,
    sidecar_df: pd.DataFrame | None = None,
) -> None:
    payload: dict[str, np.ndarray] = {
        'fused_global': pred['fused_global'],
        'z_penult': pred['fused_global'],
        'logits7': pred['logits7'],
        'probs4': pred['probs'],
        'charm_prob': pred['charm_prob'],
        'q_bkg_charm': pred['q_bkg_charm'],
        'z_tau_c': pred['z_tau_c'],
        'nccc_active': pred['nccc_active'],
        'nccc_boost': pred['nccc_boost'],
        'run': keys_df['run'].to_numpy(dtype=np.int64),
        'chunk': keys_df['chunk'].to_numpy(dtype=np.int64),
        'row': keys_df['row'].to_numpy(dtype=np.int64),
        'label': keys_df['label'].to_numpy(dtype=np.int64),
    }
    for col in ('obs_domain_id', 'region_id', 'family_hi', 'family_lo'):
        if col in keys_df.columns:
            payload[col] = keys_df[col].to_numpy(dtype=np.int64)
    if sidecar_df is not None:
        payload['label_7'] = sidecar_df['label_7'].to_numpy(dtype=np.int64)
        payload['tau_branch_valid'] = sidecar_df['tau_branch_valid'].to_numpy(dtype=np.int64)
    tmp_p = path.with_suffix('.tmp.npz')
    np.savez(tmp_p, **payload)
    tmp_p.replace(path)


# ===========================================================================
# Stage-I Pre-extraction & Training
# ===========================================================================
def run_stage1_preextract(part: int = -1) -> dict[str, Any]:
    """Sharded parallel Stage-I extraction (`part=0..7` train shards, `part=8` OOD+stress)."""
    assert_unsealed_split('internal_train', stage=1)
    assert_unsealed_split('internal_screen', stage=1)
    assert_unsealed_split('internal_val', stage=1)

    t0 = time.perf_counter()
    n_cpus = min(8, max(1, os.cpu_count() or 1))
    pa.set_cpu_count(n_cpus)
    pa.set_io_thread_count(n_cpus)

    geom = json.loads(assert_unsealed_path(CLEAN_ROOT / 'geometry.json').read_text())
    bounds = detector_bounds(geom)
    bounds['_plane_x0'] = np.array([cumulative_radiation_length(float(z), geom) for z in bounds['pixel_Zpos']], dtype=np.float32)

    STAGE1_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    scratch_base = str(_resolve_fast_cache_root(min_free_gb=6.0))

    if 0 <= part <= 7:
        out_f = STAGE1_CACHE_DIR / f'train_part_{part}.pkl'
        if out_f.is_file():
            return {'part': part, 'cached': True}
        all_auth = load_authorized_split_shards(CLEAN_ROOT / 'data', 'train', stage=1)
        sub_shards = all_auth[part * 10 : (part + 1) * 10]
        with tempfile.TemporaryDirectory(prefix=f'uni-pre-{part}-', dir=scratch_base) as tmp_dir:
            tmp_s = Path(tmp_dir) / 'shards'
            tmp_s.mkdir(parents=True, exist_ok=True)
            local_dirs = []
            for sp in sub_shards:
                assert_unsealed_path(sp)
                dst_sp = tmp_s / sp.name
                shutil.copytree(sp, dst_sp)
                local_dirs.append(dst_sp)
            tasks = [(str(sd), geom, bounds, BASE_OBS_CFG, None) for sd in local_dirs]
            with ProcessPoolExecutor(max_workers=n_cpus, mp_context=mp.get_context('fork')) as pool:
                parts = list(pool.map(_extract_unified_shard_worker, tasks))
            part_items = [it for p in parts for it in p]
        tmp_f = STAGE1_CACHE_DIR / f'train_part_{part}.pkl.tmp'
        with tmp_f.open('wb') as fh:
            pickle.dump(part_items, fh, protocol=5)
        tmp_f.replace(out_f)
        sec = float(time.perf_counter() - t0)
        print(f'[stage1-preextract part={part}] Wrote {len(part_items)} items in {sec:.1f}s', flush=True)
        return {'part': part, 'n_items': len(part_items), 'seconds': sec}

    if part == 8:
        out_f = STAGE1_CACHE_DIR / 'ood_stress_pack.pkl'
        if out_f.is_file():
            return {'part': 8, 'cached': True}
        val_keys = pd.read_parquet(assert_unsealed_path(SPLITS_DIR / 'internal_val_eval_keys.parquet'))
        val_400 = val_keys.iloc[:400].reset_index(drop=True)
        with ProcessPoolExecutor(max_workers=n_cpus, mp_context=mp.get_context('fork')) as pool:
            ood_pre = {
                ov: _build_unified_ood_items(ov, geom, bounds, pool=pool)
                for ov in ('gen_ood_w184_epps21_nc_charm', 'gen_ood_run10085_10088')
            }
            stress_items, y_400 = _build_unified_stress_items_from_internal_val(val_400, geom, bounds, pool=pool)
        tmp_f = STAGE1_CACHE_DIR / 'ood_stress_pack.pkl.tmp'
        with tmp_f.open('wb') as fh:
            pickle.dump({'ood_pre': ood_pre, 'stress_items': stress_items, 'y_400': y_400}, fh, protocol=5)
        tmp_f.replace(out_f)
        sec = float(time.perf_counter() - t0)
        print(f'[stage1-preextract part=8] Wrote OOD + stress pack in {sec:.1f}s', flush=True)
        return {'part': 8, 'seconds': sec}

    raise ValueError(f'Expected --part in 0..8, got {part}')


def run_stage1_train(variant: str) -> dict[str, Any]:
    if variant not in VARIANT_CONFIGS:
        raise ValueError(f'Unknown variant {variant!r}; expected one of {sorted(VARIANT_CONFIGS)}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for neural training')

    assert_unsealed_split('internal_train', stage=1)
    assert_unsealed_split('internal_screen', stage=1)
    assert_unsealed_split('internal_val', stage=1)

    t0 = time.perf_counter()
    utc_started = datetime.now(timezone.utc).isoformat()
    device = torch.device('cuda:0')
    seed = 42
    np.random.seed(seed)

    out_dir = EXP_ROOT / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    geom = json.loads(assert_unsealed_path(CLEAN_ROOT / 'geometry.json').read_text())
    bounds = detector_bounds(geom)
    bounds['_plane_x0'] = np.array([cumulative_radiation_length(float(z), geom) for z in bounds['pixel_Zpos']], dtype=np.float32)
    yc, nb = float(bounds['y_scint_center']), int(bounds['n_bars'])

    tr_keys = pd.read_parquet(assert_unsealed_path(SPLITS_DIR / 'internal_train_keys.parquet'))
    dev_keys = pd.read_parquet(assert_unsealed_path(SPLITS_DIR / 'internal_dev_screen_keys.parquet'))
    val_keys = pd.read_parquet(assert_unsealed_path(SPLITS_DIR / 'internal_val_eval_keys.parquet'))

    sc_df = pd.read_parquet(assert_unsealed_path(CLEAN_ROOT / 'truth_sidecar/event_sidecar.parquet'))
    pos_sc = pd.MultiIndex.from_frame(sc_df[KEY]).get_indexer(pd.MultiIndex.from_frame(tr_keys[KEY]))
    if (pos_sc < 0).any():
        raise ValueError('internal_train keys missing from event_sidecar.parquet')
    tr_sidecar = sc_df.iloc[pos_sc].reset_index(drop=True)
    del sc_df

    part_names = [f'train_part_{p}.pkl' for p in range(8)] + ['ood_stress_pack.pkl']
    if not all((STAGE1_CACHE_DIR / fn).is_file() for fn in part_names):
        raise FileNotFoundError(f'Stage-I cache missing in {STAGE1_CACHE_DIR}; run stage1-preextract first')

    staged_dir, used_shm = _stage_pkl_dir_to_shm(STAGE1_CACHE_DIR, part_names, prefix=f'uni_s1_{variant}')
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
    val_items = _align_items_to_keys(all_items, val_keys)
    del all_items
    gc.collect()
    print(f'[{variant}] Loaded Stage-1 cache (shm={used_shm}) in {time.perf_counter()-t0:.1f}s', flush=True)

    train_ds = InMemoryUnifiedDataset(
        tr_items,
        event_sidecar=tr_sidecar,
        mirror_seed=seed,
        y_scint_center=yc,
        n_bars=nb,
    )
    cap = torch.cuda.get_device_capability(0)
    torch.manual_seed(seed)

    model_cfg = {
        'dim': 160,
        'heads': 8,
        'drop_path': 0.12,
        'head_dropout': 0.18,
        'n_front_layers': 2,
        'n_z_layers': 2,
        'n_rel_pre': 2,
        'n_rel_post': 2,
        'n_global_layers': 2,
        **VARIANT_CONFIGS[variant],
    }
    model = PinpointUnifiedNet(geom, model_cfg).to(device)
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
    ema_decay_max = 0.9975

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
    amp_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
    amp_ok = cap[0] >= 7
    scaler = torch.amp.GradScaler('cuda', enabled=(amp_ok and amp_dtype == torch.float16))

    y_dev = dev_keys['label'].to_numpy(dtype=np.int64)
    best_nll = float('inf')
    best_pass = 0
    history: list[dict[str, Any]] = []
    best_ema_path = out_dir / 'best_ema.pt'

    n_dl_workers = min(2, max(0, (os.cpu_count() or 2) - 1))
    if n_dl_workers > 0:
        torch.multiprocessing.set_sharing_strategy('file_system')
    collator = _UnifiedCollator(yc, nb)

    torch.cuda.reset_peak_memory_stats(device)
    global_step = 0
    for p_idx in range(1, passes + 1):
        tp0 = time.perf_counter()
        train_ds.set_pass(p_idx)
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
        win_buf: list[dict] = []
        for mb in loader:
            win_buf.append(mb)
            if len(win_buf) < grad_accum:
                continue
            global_step += 1
            opt.zero_grad(set_to_none=True)
            fd = compute_unified_window_denominators(win_buf)
            win_loss = 0.0
            win_n = 0
            for sub_mb in win_buf:
                db = move_batch_to_device(sub_mb, device)
                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_ok):
                    loss, _ = compute_unified_loss(model, db, fixed_denoms=fd)
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
            ema_d = min(ema_decay_max, (1.0 + global_step) / (10.0 + global_step))
            _update_ema(ema_model, model, ema_d)
            loss_sum += win_loss * win_n
            n_seen += win_n
            win_buf.clear()

        if win_buf:
            global_step += 1
            opt.zero_grad(set_to_none=True)
            fd = compute_unified_window_denominators(win_buf)
            tail_loss = 0.0
            tail_n = 0
            for sub_mb in win_buf:
                db = move_batch_to_device(sub_mb, device)
                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_ok):
                    loss, _ = compute_unified_loss(model, db, fixed_denoms=fd)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                tail_loss += float(loss.detach().item())
                tail_n += len(db['label'])
            loss_sum += tail_loss * tail_n
            n_seen += tail_n
            if scaler.is_enabled():
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(opt)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
            scheduler.step()
            ema_d = min(ema_decay_max, (1.0 + global_step) / (10.0 + global_step))
            _update_ema(ema_model, model, ema_d)

        train_loss = loss_sum / max(n_seen, 1)
        row_hist: dict[str, Any] = {'pass': p_idx, 'train_loss': train_loss, 'pass_sec': float(time.perf_counter() - tp0)}

        if p_idx >= 10 or p_idx % 2 == 0 or p_idx == passes:
            dev_pred = predict_unified_items_fp32(ema_model, dev_items, bounds, device, batch_size=64)
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

    # Load best EMA, calibrate PV scales strictly on internal_train, evaluate in IEEE-FP32
    _set_ieee_fp32(True)
    ckpt = torch.load(best_ema_path, map_location=device, weights_only=False)
    ema_model.load_state_dict(ckpt['model'])
    ema_model.eval()
    pv_corr = _calibrate_pv_regime_scales_on_train(ema_model, tr_items, tr_sidecar, bounds, device)
    torch.save(
        {'model': ema_model.state_dict(), 'config': model_cfg, 'pass': best_pass, 'pv_regime_scale_corr': pv_corr.tolist()},
        best_ema_path,
    )

    need_latents = (variant == 's1_unified_pinpoint_net')
    dev_pred = predict_unified_items_fp32(ema_model, dev_items, bounds, device, batch_size=64, return_latents=need_latents)
    val_pred = predict_unified_items_fp32(ema_model, val_items, bounds, device, batch_size=64, return_latents=need_latents)
    np.save(out_dir / 'probs_dev_screen.npy', dev_pred['probs'])
    np.save(out_dir / 'probs_internal_val_eval.npy', val_pred['probs'])
    np.save(out_dir / 'v_pv_internal_val_eval.npy', val_pred['v_pv_mm'])
    np.save(out_dir / 'scale_pv_internal_val_eval.npy', val_pred['scale_pv_mm'])

    if need_latents:
        tr_pred = predict_unified_items_fp32(ema_model, tr_items, bounds, device, batch_size=64, return_latents=True)
        _export_latent_cache(out_dir / 'latent_cache_internal_train.npz', tr_pred, tr_keys, sidecar_df=tr_sidecar)
        _export_latent_cache(out_dir / 'latent_cache_internal_dev_screen.npz', dev_pred, dev_keys)
        _export_latent_cache(out_dir / 'latent_cache_internal_val_eval.npz', val_pred, val_keys)

    # Open pure-W OOD evaluation
    ood_probs: dict[str, np.ndarray] = {}
    ood_metrics: dict[str, Any] = {}
    for ov, (o_items, yo) in ood_pre.items():
        op = predict_unified_items_fp32(ema_model, o_items, bounds, device, batch_size=64)['probs']
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

    # 7 Stress perturbations on first 400 events of internal_val_eval
    stress_probs: dict[str, np.ndarray] = {}
    stress_metrics: dict[str, Any] = {}
    p_nom = None
    ba4_nom = 0.0
    for st_name, _, _ in STRESS_SPECS:
        sp = predict_unified_items_fp32(ema_model, stress_items[st_name], bounds, device, batch_size=64)['probs']
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

    # Gate-A 39-key truth tamper check on trained checkpoint
    b_chk = move_batch_to_device(collate_dual_regime(val_items[:8], mirror_rng=None, y_scint_center=yc, n_bars=nb), device)
    truth_inv = verify_inference_truth_invariance(ema_model, b_chk, atol=1e-7)

    y_val = val_keys['label'].to_numpy(dtype=np.int64)
    val_m = _ba4_and_nll(y_val, val_pred['probs'])
    summary = {
        'variant': variant,
        'UTC_started': utc_started,
        'UTC_finished': datetime.now(timezone.utc).isoformat(),
        'wall_seconds': float(time.perf_counter() - t0),
        'gpu_name': torch.cuda.get_device_name(0),
        'n_params': n_params,
        'seed': seed,
        'passes': passes,
        'effective_batch_size': batch_size,
        'micro_bs': micro_bs,
        'grad_accum': grad_accum,
        'peak_vram_MiB': peak_vram_mib,
        'best_pass': best_pass,
        'best_dev_screen_NLL': best_nll,
        'pv_regime_scale_corr': pv_corr.tolist(),
        'truth_invariance_max_diff': truth_inv['max_abs_diff'],
        'internal_val_eval_all': val_m,
        'ood_metrics': ood_metrics,
        'stress_400_internal_val': stress_metrics,
        'history': history,
    }
    write_json(out_dir / 'run_summary.json', summary)
    print(
        f"[{variant}] DONE in {summary['wall_seconds']:.1f}s | peak_VRAM={peak_vram_mib:.0f} MiB | "
        f"val_BA4={val_m['BA4_pct']:.2f}% val_NLL={val_m['NLL']:.4f}",
        flush=True,
    )
    return summary


def run_stage1_aggregate() -> dict[str, Any]:
    """Aggregate Stage-I runs, check acceptance gates vs 50/50 Blend & run 1,000x paired bootstrap."""
    assert_unsealed_split('internal_train', stage=1)
    assert_unsealed_split('internal_val', stage=1)
    val_keys = pd.read_parquet(assert_unsealed_path(SPLITS_DIR / 'internal_val_eval_keys.parquet'))
    sc_df = pd.read_parquet(assert_unsealed_path(CLEAN_ROOT / 'truth_sidecar/event_sidecar.parquet'))

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
        tail = {
            'split_half_1pct_All': _split_half_tail_eval(s_tau, y_val, val_keys, m_all, 0.01),
            'split_half_1pct_D1_obs': _split_half_tail_eval(s_tau, y_val, val_keys, m_d1, 0.01),
            'split_half_1pct_Pinpoint': _split_half_tail_eval(s_tau, y_val, val_keys, m_pin, 0.01),
            'split_half_0p1pct_All': _split_half_tail_eval(s_tau, y_val, val_keys, m_all, 0.001),
            'within_slice_1pct_All': _within_slice_tail_eval(s_tau, y_val, m_all, 0.01),
            'within_slice_1pct_D1_obs': _within_slice_tail_eval(s_tau, y_val, m_d1, 0.01),
            'within_slice_1pct_Pinpoint': _within_slice_tail_eval(s_tau, y_val, m_pin, 0.01),
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
            'ood_metrics': run_sum.get('ood_metrics'),
            'stress_400_internal_val': run_sum.get('stress_400_internal_val'),
        }

    # Load Stage-326 Stage-I TrackletIPFormer & DualRegimeModalNet for exact paired bootstrap
    s326_refs: dict[str, np.ndarray] = {}
    p_tip_f = STAGE1_S326_TRACKLET_DIR / 's1_tracklet_ipformer/val_eval_predictions.parquet'
    if p_tip_f.is_file():
        df_tip = pd.read_parquet(p_tip_f)
        pos_t = pd.MultiIndex.from_frame(df_tip[KEY]).get_indexer(pd.MultiIndex.from_frame(val_keys[KEY]))
        if not (pos_t < 0).any():
            s326_refs['s1_tracklet_ipformer'] = df_tip.iloc[pos_t][['p_0', 'p_1', 'p_2', 'p_3']].to_numpy(np.float64)
    p_dr_f = STAGE1_S326_DUAL_DIR / 's1_dual_regime_modalnet/probs_internal_val_eval.npy'
    if p_dr_f.is_file():
        s326_refs['s1_dual_regime_modalnet'] = np.load(p_dr_f).astype(np.float64)
    if 's1_tracklet_ipformer' in s326_refs and 's1_dual_regime_modalnet' in s326_refs:
        s326_refs['s1_blend_50_50'] = 0.5 * (s326_refs['s1_tracklet_ipformer'] + s326_refs['s1_dual_regime_modalnet'])

    paired_boot: dict[str, Any] = {}
    if 's1_unified_pinpoint_net' in probs_map:
        pa = probs_map['s1_unified_pinpoint_net']
        m_all = np.ones(len(y_val), dtype=bool)
        m_d1 = dom_val == 1
        m_pin = (reg_val == 0) | (reg_val == 1)
        for other_name, pb in {**{k: v for k, v in probs_map.items() if k != 's1_unified_pinpoint_net'}, **s326_refs}.items():
            for stag, mk in (('All', m_all), ('D1_obs', m_d1), ('Pinpoint', m_pin)):
                paired_boot[f'PinpointUnifiedNet - {other_name} ({stag})'] = _paired_family_bootstrap_full(
                    pa, pb, y_val, fam_codes, mk, n_boot=1000, seed=20263250
                )

    # Evaluate Stage-I Acceptance Gates
    gates: dict[str, Any] = {}
    if 's1_unified_pinpoint_net' in models_report:
        rep = models_report['s1_unified_pinpoint_net']
        ba4 = rep['internal_val_eval_all']['BA4_pct']
        nll = rep['internal_val_eval_all']['NLL']
        w184_gt05 = rep['ood_metrics']['gen_ood_w184_epps21_nc_charm']['frac_p_tau_gt_0p5_pct']
        trk_drop = -rep['stress_400_internal_val']['tracker_dead_2pct_noise1']['delta_BA4_vs_nominal_pp']
        all_pulls = [
            float(x)
            for reg in ('All_valid', 'FrontPinpoint_30mm', 'IntermPinpoint_150mm')
            for x in rep['pv_audit'][reg]['p68_abs_pull']
        ]
        pulls_ok = all(0.85 <= p <= 1.15 for p in all_pulls)
        blend_ba4 = 58.54
        blend_nll = 0.9593
        if 's1_blend_50_50' in s326_refs:
            bm = _ba4_and_nll(y_val, s326_refs['s1_blend_50_50'])
            blend_ba4 = bm['BA4_pct']
            blend_nll = bm['NLL']

        gates = {
            'All_BA4_pct': ba4,
            'All_NLL': nll,
            'gate_BA4_ge_58p3': bool(ba4 >= 58.3),
            'blend_ref_BA4_pct': blend_ba4,
            'blend_ref_NLL': blend_nll,
            'gate_within_0p3pp_of_blend': bool((blend_ba4 - ba4) <= 0.3),
            'gate_within_0p005_NLL_of_blend': bool((nll - blend_nll) <= 0.005),
            'w184_p_tau_gt_0p5_pct': w184_gt05,
            'gate_w184_le_1p8pct': bool(w184_gt05 <= 1.8),
            'trk_dead_2pct_n1_drop_pp': trk_drop,
            'gate_trk_drop_lt_3p0pp': bool(trk_drop < 3.0),
            'all_9_pulls': all_pulls,
            'gate_all_9_pulls_in_0p85_1p15': bool(pulls_ok),
        }
        gates['all_stage1_gates_passed'] = bool(
            gates['gate_BA4_ge_58p3']
            and gates['gate_within_0p3pp_of_blend']
            and gates['gate_within_0p005_NLL_of_blend']
            and gates['gate_w184_le_1p8pct']
            and gates['gate_trk_drop_lt_3p0pp']
            and gates['gate_all_9_pulls_in_0p85_1p15']
        )

    out_doc = {
        'UTC': datetime.now(timezone.utc).isoformat(),
        'split_manifest': str(SPLITS_DIR),
        'internal_val_eval_events': int(len(y_val)),
        'models': models_report,
        'stage1_gates': gates,
        's326_references': {k: _ba4_and_nll(y_val, v) for k, v in s326_refs.items()},
        'paired_family_bootstrap_1000_seed20263250': paired_boot,
    }
    write_json(EXP_ROOT / 'stage1_unified_results.json', out_doc)
    print(json.dumps({'stage1_gates': gates}, indent=2), flush=True)
    return out_doc


# ===========================================================================
# Stage-II Pre-extraction, Training & Aggregation
# ===========================================================================
def run_stage2_preextract(part: int = -1) -> dict[str, Any]:
    """Sharded Stage-II item extraction (`part=0..7` Train32k, `part=8..9` Tune8k, `part=10` 6xOOD + Shard-080)."""
    for sp in ('train', 'validation', 'tune_screen', 'tune6k', 'ood'):
        assert_unsealed_split(sp, stage=2)

    t0 = time.perf_counter()
    n_cpus = min(8, max(1, os.cpu_count() or 1))
    pa.set_cpu_count(n_cpus)
    pa.set_io_thread_count(n_cpus)

    geom = json.loads(assert_unsealed_path(CLEAN_ROOT / 'geometry.json').read_text())
    bounds = detector_bounds(geom)
    bounds['_plane_x0'] = np.array([cumulative_radiation_length(float(z), geom) for z in bounds['pixel_Zpos']], dtype=np.float32)

    STAGE2_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    scratch_base = str(_resolve_fast_cache_root(min_free_gb=6.0))

    if 0 <= part <= 7:
        out_f = STAGE2_CACHE_DIR / f'train_part_{part}.pkl'
        if out_f.is_file():
            return {'part': part, 'split': 'train', 'cached': True}
        all_tr = load_authorized_split_shards(CLEAN_ROOT / 'data', 'train', stage=2)
        sub_shards = all_tr[part * 10 : (part + 1) * 10]
        with tempfile.TemporaryDirectory(prefix=f'uni-s2-tr-{part}-', dir=scratch_base) as tmp_dir:
            tmp_s = Path(tmp_dir) / 'shards'
            tmp_s.mkdir(parents=True, exist_ok=True)
            local_dirs = []
            for sp in sub_shards:
                assert_unsealed_path(sp)
                dst_sp = tmp_s / sp.name
                shutil.copytree(sp, dst_sp)
                local_dirs.append(dst_sp)
            tasks = [(str(sd), geom, bounds, BASE_OBS_CFG, None, 'train') for sd in local_dirs]
            with ProcessPoolExecutor(max_workers=n_cpus, mp_context=mp.get_context('fork')) as pool:
                parts = list(pool.map(_extract_unified_shard_worker, tasks))
            part_items = [it for p in parts for it in p]
        tmp_f = STAGE2_CACHE_DIR / f'train_part_{part}.pkl.tmp'
        with tmp_f.open('wb') as fh:
            pickle.dump(part_items, fh, protocol=5)
        tmp_f.replace(out_f)
        return {'part': part, 'split': 'train', 'n_items': len(part_items), 'seconds': float(time.perf_counter() - t0)}

    if 8 <= part <= 9:
        v_idx = part - 8
        out_f = STAGE2_CACHE_DIR / f'val_part_{v_idx}.pkl'
        if out_f.is_file():
            return {'part': part, 'split': 'validation', 'cached': True}
        all_va = load_authorized_split_shards(CLEAN_ROOT / 'data', 'validation', stage=2)
        sub_shards = all_va[v_idx * 10 : (v_idx + 1) * 10]
        with tempfile.TemporaryDirectory(prefix=f'uni-s2-va-{v_idx}-', dir=scratch_base) as tmp_dir:
            tmp_s = Path(tmp_dir) / 'shards'
            tmp_s.mkdir(parents=True, exist_ok=True)
            local_dirs = []
            for sp in sub_shards:
                assert_unsealed_path(sp)
                dst_sp = tmp_s / sp.name
                shutil.copytree(sp, dst_sp)
                local_dirs.append(dst_sp)
            tasks = [(str(sd), geom, bounds, BASE_OBS_CFG, None, 'validation') for sd in local_dirs]
            with ProcessPoolExecutor(max_workers=n_cpus, mp_context=mp.get_context('fork')) as pool:
                parts = list(pool.map(_extract_unified_shard_worker, tasks))
            part_items = [it for p in parts for it in p]
        tmp_f = STAGE2_CACHE_DIR / f'val_part_{v_idx}.pkl.tmp'
        with tmp_f.open('wb') as fh:
            pickle.dump(part_items, fh, protocol=5)
        tmp_f.replace(out_f)
        return {'part': part, 'split': 'validation', 'n_items': len(part_items), 'seconds': float(time.perf_counter() - t0)}

    if part == 10:
        out_f = STAGE2_CACHE_DIR / 'ood_stress_stage2_pack.pkl'
        if out_f.is_file():
            return {'part': 10, 'cached': True}
        with ProcessPoolExecutor(max_workers=n_cpus, mp_context=mp.get_context('fork')) as pool:
            ood_pre = {ov: _build_unified_ood_items(ov, geom, bounds, pool=pool) for ov in ALL_STAGE2_OOD}
            stress_items, y_080, prov_080 = _build_unified_shard080_stress_items(geom, bounds, pool=pool)
        tmp_f = STAGE2_CACHE_DIR / 'ood_stress_stage2_pack.pkl.tmp'
        with tmp_f.open('wb') as fh:
            pickle.dump(
                {'ood_pre': ood_pre, 'stress_items': stress_items, 'y_080': y_080, 'prov_080': prov_080},
                fh,
                protocol=5,
            )
        tmp_f.replace(out_f)
        return {'part': 10, 'n_ood_sets': len(ood_pre), 'seconds': float(time.perf_counter() - t0)}

    raise ValueError(f'Expected --part in 0..10, got {part}')


@torch.no_grad()
def _verify_stage2_obs_only_and_truth_invariance(
    ema_model: PinpointUnifiedNet,
    geom: dict,
    bounds: dict,
    tune_meta: pd.DataFrame,
    saved_probs_tune8k: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    _set_ieee_fp32(True)
    ema_model.eval()
    shard080 = assert_unsealed_path(CLEAN_ROOT / 'data/shards/080')
    prov64 = pd.read_parquet(shard080 / 'provenance.parquet', columns=KEY).iloc[:64].reset_index(drop=True)
    raw64 = pq.read_table(shard080 / 'observations.parquet').slice(0, 64)

    with tempfile.TemporaryDirectory(prefix='uni-s2-obsonly-') as tmp_d:
        tmp_p = Path(tmp_d)
        pq.write_table(raw64, tmp_p / 'observations.parquet')
        obs_list = pq.read_table(tmp_p / 'observations.parquet').to_pylist()
        obs_items = [
            extract_unified_item(
                i, ob, None, compute_observable_containment(ob, geom, bounds, BASE_OBS_CFG),
                geom, bounds, BASE_OBS_CFG,
            )
            for i, ob in enumerate(obs_list)
        ]
        rep_pred = predict_unified_items_fp32(ema_model, obs_items, bounds, device, batch_size=64)
        p_obs = rep_pred['probs'].astype(np.float64)

    pos64 = pd.MultiIndex.from_frame(tune_meta[KEY]).get_indexer(pd.MultiIndex.from_frame(prov64[KEY]))
    diff_vs_saved = float(np.max(np.abs(p_obs - saved_probs_tune8k[pos64].astype(np.float64))))

    yc, nb = float(bounds['y_scint_center']), int(bounds['n_bars'])
    b_clean = move_batch_to_device(collate_dual_regime(obs_items[:16], mirror_rng=None, y_scint_center=yc, n_bars=nb), device)
    b_clean = {k: v for k, v in b_clean.items() if k not in TRUTH_BATCH_KEYS}
    out_ref = ema_model.forward_NextGen(b_clean)

    b_mut = dict(b_clean)
    bsz = 16
    for tk in TRUTH_BATCH_KEYS:
        if tk == 'tau_branch':
            b_mut[tk] = ['had'] * bsz
        elif tk in ('label', 'label_4', 'label_7', 'tau_branch_valid', 'charm_flag', 'charm_valid', 'vertex_valid', 'kin_valid', 'family_hi', 'family_lo'):
            b_mut[tk] = torch.ones((bsz,), dtype=torch.long, device=device)
        elif tk == 'has_charm':
            b_mut[tk] = torch.ones((bsz,), dtype=torch.bool, device=device)
        elif tk == 'vertex_xyz_mm':
            b_mut[tk] = torch.full((bsz, 3), 999.0, dtype=torch.float32, device=device)
        elif tk == 'vis_kinematics':
            b_mut[tk] = torch.full((bsz, 4), 42.0, dtype=torch.float32, device=device)
        elif tk == 'trackID':
            b_mut[tk] = torch.full((bsz, 64), 777, dtype=torch.long, device=device)
        else:
            b_mut[tk] = torch.ones((bsz,), dtype=torch.float32, device=device)
    out_mut = ema_model.forward_NextGen(b_mut)

    out_diffs = {
        fname: float((getattr(out_ref, fname).detach().float() - getattr(out_mut, fname).detach().float()).abs().max().item())
        for fname in ('logits7', 'probs4', 'charm_logit', 'v_pv_mm', 'scale_pv_mm', 'sv_disp_mm', 'kin_pred')
    }
    max_truth_diff = max(out_diffs.values())
    if diff_vs_saved > 1e-5:
        raise RuntimeError(f'Stage-II label-free replay mismatch: {diff_vs_saved}')
    if max_truth_diff > 1e-7:
        raise RuntimeError(f'Stage-II truth-invariance violation: {out_diffs}')
    return {
        'obs_only_64_max_abs_diff': diff_vs_saved,
        'truth_invariance_max_abs_diff': max_truth_diff,
        'passed': bool(diff_vs_saved <= 1e-6 and max_truth_diff <= 1e-7),
    }


def run_stage2_train(seed: int) -> dict[str, Any]:
    if seed not in STAGE2_SEEDS:
        raise ValueError(f'Expected Stage-2 seed in {STAGE2_SEEDS}, got {seed}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for Stage-2 training')

    for sp in ('train', 'validation', 'tune_screen', 'tune6k', 'ood'):
        assert_unsealed_split(sp, stage=2)

    t0 = time.perf_counter()
    utc_started = datetime.now(timezone.utc).isoformat()
    device = torch.device('cuda:0')
    np.random.seed(seed)

    out_dir = STAGE2_DIR / f'seed_{seed}'
    out_dir.mkdir(parents=True, exist_ok=True)

    geom = json.loads(assert_unsealed_path(CLEAN_ROOT / 'geometry.json').read_text())
    bounds = detector_bounds(geom)
    bounds['_plane_x0'] = np.array([cumulative_radiation_length(float(z), geom) for z in bounds['pixel_Zpos']], dtype=np.float32)
    yc, nb = float(bounds['y_scint_center']), int(bounds['n_bars'])

    tr_part_names = [f'train_part_{p}.pkl' for p in range(8)]
    va_part_names = [f'val_part_{p}.pkl' for p in range(2)]
    all_names = tr_part_names + va_part_names + ['ood_stress_stage2_pack.pkl']
    if not all((STAGE2_CACHE_DIR / fn).is_file() for fn in tr_part_names + va_part_names):
        raise FileNotFoundError(f'Stage-2 Train32k/Tune8k cache missing in {STAGE2_CACHE_DIR}')

    staged_dir, used_shm = _stage_pkl_dir_to_shm(STAGE2_CACHE_DIR, all_names, prefix=f'uni_s2_s{seed}')
    try:
        tr_items_raw = []
        for fn in tr_part_names:
            pf = staged_dir / fn
            with pf.open('rb') as fh:
                tr_items_raw.extend(pickle.load(fh))
            if used_shm:
                pf.unlink(missing_ok=True)
        va_items_raw = []
        for fn in va_part_names:
            pf = staged_dir / fn
            with pf.open('rb') as fh:
                va_items_raw.extend(pickle.load(fh))
            if used_shm:
                pf.unlink(missing_ok=True)
        os_file = staged_dir / 'ood_stress_stage2_pack.pkl'
        os_pack = None
        if os_file.is_file():
            with os_file.open('rb') as fh:
                os_pack = pickle.load(fh)
            if used_shm:
                os_file.unlink(missing_ok=True)
    finally:
        if used_shm:
            shutil.rmtree(staged_dir, ignore_errors=True)
    gc.collect()

    tr_items_raw.sort(key=lambda it: (int(it['run']), int(it['chunk']), int(it['row'])))
    tr_keys = pd.DataFrame({
        'run': [int(it['run']) for it in tr_items_raw],
        'chunk': [int(it['chunk']) for it in tr_items_raw],
        'row': [int(it['row']) for it in tr_items_raw],
        'label': [int(it['label']) for it in tr_items_raw],
        'obs_domain_id': [int(it['obs_domain_id']) for it in tr_items_raw],
        'family_hi': [int(it['family_hi']) for it in tr_items_raw],
        'family_lo': [int(it['family_lo']) for it in tr_items_raw],
    })
    sc_df = pd.read_parquet(assert_unsealed_path(CLEAN_ROOT / 'truth_sidecar/event_sidecar.parquet'))
    pos_sc = pd.MultiIndex.from_frame(sc_df[KEY]).get_indexer(pd.MultiIndex.from_frame(tr_keys[KEY]))
    tr_sidecar = sc_df.iloc[pos_sc].reset_index(drop=True)
    del sc_df

    tune_meta = pd.read_parquet(assert_unsealed_path(CLEAN_ROOT / 'validation_meta/tune_truth_meta.parquet'))
    scr_df = pd.read_parquet(assert_unsealed_path(CLEAN_ROOT / 'data/tune_screen.parquet'), columns=KEY + ['label'])
    tune8k_items = _align_items_to_keys(va_items_raw, tune_meta)
    del va_items_raw

    in_screen = np.asarray(pd.MultiIndex.from_frame(tune_meta[KEY]).isin(pd.MultiIndex.from_frame(scr_df[KEY])), dtype=bool)
    tune6k_m = ~in_screen
    screen_items = [tune8k_items[i] for i in np.flatnonzero(in_screen)]
    y_screen = tune_meta['label'].to_numpy(dtype=np.int64)[in_screen]

    train_ds = InMemoryUnifiedDataset(
        tr_items_raw,
        event_sidecar=tr_sidecar,
        mirror_seed=seed,
        y_scint_center=yc,
        n_bars=nb,
    )
    cap = torch.cuda.get_device_capability(0)
    torch.manual_seed(seed)

    model_cfg = {
        'dim': 160,
        'heads': 8,
        'drop_path': 0.12,
        'head_dropout': 0.18,
        'n_front_layers': 2,
        'n_z_layers': 2,
        'n_rel_pre': 2,
        'n_rel_post': 2,
        'n_global_layers': 2,
        **VARIANT_CONFIGS['s1_unified_pinpoint_net'],
    }
    model = PinpointUnifiedNet(geom, model_cfg).to(device)
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
    ema_decay_max = 0.9975

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
    amp_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
    amp_ok = cap[0] >= 7
    scaler = torch.amp.GradScaler('cuda', enabled=(amp_ok and amp_dtype == torch.float16))

    best_nll = float('inf')
    best_pass = 0
    history: list[dict[str, Any]] = []
    best_ema_path = out_dir / 'best_ema.pt'

    n_dl_workers = min(2, max(0, (os.cpu_count() or 2) - 1))
    if n_dl_workers > 0:
        torch.multiprocessing.set_sharing_strategy('file_system')
    collator = _UnifiedCollator(yc, nb)

    torch.cuda.reset_peak_memory_stats(device)
    global_step = 0
    for p_idx in range(1, passes + 1):
        tp0 = time.perf_counter()
        train_ds.set_pass(p_idx)
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
        win_buf: list[dict] = []
        for mb in loader:
            win_buf.append(mb)
            if len(win_buf) < grad_accum:
                continue
            global_step += 1
            opt.zero_grad(set_to_none=True)
            fd = compute_unified_window_denominators(win_buf)
            win_loss = 0.0
            win_n = 0
            for sub_mb in win_buf:
                db = move_batch_to_device(sub_mb, device)
                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_ok):
                    loss, _ = compute_unified_loss(model, db, fixed_denoms=fd)
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
            ema_d = min(ema_decay_max, (1.0 + global_step) / (10.0 + global_step))
            _update_ema(ema_model, model, ema_d)
            loss_sum += win_loss * win_n
            n_seen += win_n
            win_buf.clear()

        if win_buf:
            global_step += 1
            opt.zero_grad(set_to_none=True)
            fd = compute_unified_window_denominators(win_buf)
            tail_loss = 0.0
            tail_n = 0
            for sub_mb in win_buf:
                db = move_batch_to_device(sub_mb, device)
                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_ok):
                    loss, _ = compute_unified_loss(model, db, fixed_denoms=fd)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                tail_loss += float(loss.detach().item())
                tail_n += len(db['label'])
            loss_sum += tail_loss * tail_n
            n_seen += tail_n
            if scaler.is_enabled():
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(opt)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
            scheduler.step()
            ema_d = min(ema_decay_max, (1.0 + global_step) / (10.0 + global_step))
            _update_ema(ema_model, model, ema_d)

        train_loss = loss_sum / max(n_seen, 1)
        row_hist: dict[str, Any] = {'pass': p_idx, 'step': global_step, 'train_loss': train_loss, 'pass_sec': float(time.perf_counter() - tp0)}
        if p_idx == 1 or p_idx >= 10 or p_idx % 2 == 0 or p_idx == passes:
            scr_pred = predict_unified_items_fp32(ema_model, screen_items, bounds, device, batch_size=64)
            scr_m = _ba4_and_nll(y_screen, scr_pred['probs'])
            row_hist['tune_screen_NLL'] = scr_m['NLL']
            row_hist['tune_screen_BA4_pct'] = scr_m['BA4_pct']
            if scr_m['NLL'] < best_nll:
                best_nll = scr_m['NLL']
                best_pass = p_idx
                torch.save({'model': ema_model.state_dict(), 'config': model_cfg, 'pass': p_idx, 'seed': seed}, best_ema_path)
            print(
                f"[stage2 seed={seed}] pass={p_idx:02d}/{passes} step={global_step} "
                f"tune_screen_NLL={scr_m['NLL']:.4f} BA4={scr_m['BA4_pct']:.2f}% ({row_hist['pass_sec']:.1f}s)",
                flush=True,
            )
        history.append(row_hist)

    peak_vram_mib = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))

    _set_ieee_fp32(True)
    ckpt = torch.load(best_ema_path, map_location=device, weights_only=False)
    ema_model.load_state_dict(ckpt['model'])
    ema_model.eval()
    pv_corr = _calibrate_pv_regime_scales_on_train(ema_model, tr_items_raw, tr_sidecar, bounds, device)
    torch.save(
        {'model': ema_model.state_dict(), 'config': model_cfg, 'pass': best_pass, 'seed': seed, 'pv_regime_scale_corr': pv_corr.tolist()},
        best_ema_path,
    )

    tr_pred = predict_unified_items_fp32(ema_model, tr_items_raw, bounds, device, batch_size=64, return_latents=True)
    tune8k_pred = predict_unified_items_fp32(ema_model, tune8k_items, bounds, device, batch_size=64, return_latents=True)
    _export_latent_cache(out_dir / 'latent_cache_train32k.npz', tr_pred, tr_keys, sidecar_df=tr_sidecar)
    _export_latent_cache(out_dir / 'latent_cache_tune8k.npz', tune8k_pred, tune_meta)

    p_tune8k = tune8k_pred['probs']
    np.save(out_dir / 'probs_tune8k.npy', p_tune8k)
    np.save(out_dir / 'probs_tune_screen.npy', p_tune8k[in_screen])
    np.save(out_dir / 'probs_tune6k.npy', p_tune8k[tune6k_m])
    np.save(out_dir / 'v_pv_tune8k.npy', tune8k_pred['v_pv_mm'])
    np.save(out_dir / 'scale_pv_tune8k.npy', tune8k_pred['scale_pv_mm'])
    np.save(out_dir / 'v_pv_tune6k.npy', tune8k_pred['v_pv_mm'][tune6k_m])
    np.save(out_dir / 'scale_pv_tune6k.npy', tune8k_pred['scale_pv_mm'][tune6k_m])
    np.save(out_dir / 'charm_prob_tune8k.npy', tune8k_pred['charm_prob'])

    if os_pack is None:
        os_eos = STAGE2_CACHE_DIR / 'ood_stress_stage2_pack.pkl'
        for _ in range(60):
            if os_eos.is_file():
                break
            time.sleep(15.0)
        with os_eos.open('rb') as fh:
            os_pack = pickle.load(fh)
    ood_pre = os_pack['ood_pre']
    stress_items = os_pack['stress_items']
    y_080 = os_pack['y_080']
    del os_pack

    ood_probs: dict[str, np.ndarray] = {}
    ood_metrics: dict[str, Any] = {}
    for ov, (o_items, o_lbls) in ood_pre.items():
        op = predict_unified_items_fp32(ema_model, o_items, bounds, device, batch_size=64)['probs']
        ood_probs[f'ood_{ov}'] = op
        if 'nc_' in ov:
            ood_metrics[ov] = {
                'n_events': int(len(o_lbls)),
                'mean_p_tau': float(op[:, 2].mean()),
                'frac_pred_nc_pct': float((op.argmax(1) == 3).mean() * 100.0),
                'frac_pred_tau_pct': float((op.argmax(1) == 2).mean() * 100.0),
                'frac_p_tau_gt_0p5_pct': float((op[:, 2] > 0.5).mean() * 100.0),
            }
        else:
            ood_metrics[ov] = _ba4_and_nll(o_lbls, op)

    stress_probs: dict[str, np.ndarray] = {}
    stress_metrics: dict[str, Any] = {}
    p_nom = None
    ba4_nom = 0.0
    for st_name, _, _ in STRESS_SPECS:
        ps = predict_unified_items_fp32(ema_model, stress_items[st_name], bounds, device, batch_size=64)['probs']
        stress_probs[f'stress_{st_name}'] = ps
        m_st = _ba4_and_nll(y_080, ps)
        if st_name == 'nominal':
            p_nom = ps
            ba4_nom = m_st['BA4_pct']
        tv = float(0.5 * np.abs(ps - p_nom).sum(axis=1).mean()) if p_nom is not None else 0.0
        stress_metrics[st_name] = {
            'BA4_pct': m_st['BA4_pct'],
            'delta_BA4_vs_nominal_pp': float(m_st['BA4_pct'] - ba4_nom),
            'NLL': m_st['NLL'],
            'mean_TV_vs_nominal': tv,
        }
    np.savez_compressed(out_dir / 'ood_and_stress_probs.npz', **ood_probs, **stress_probs)

    closure = _verify_stage2_obs_only_and_truth_invariance(ema_model, geom, bounds, tune_meta, p_tune8k, device)
    y_tune8k = tune_meta['label'].to_numpy(dtype=np.int64)
    t6k_m = _ba4_and_nll(y_tune8k[tune6k_m], p_tune8k[tune6k_m])
    summary = {
        'variant': 's2_unified_pinpoint_net',
        'seed': int(seed),
        'UTC_started': utc_started,
        'UTC_finished': datetime.now(timezone.utc).isoformat(),
        'wall_seconds': float(time.perf_counter() - t0),
        'gpu_name': torch.cuda.get_device_name(0),
        'n_params': n_params,
        'train_events': len(train_ds),
        'passes': passes,
        'total_steps': global_step,
        'peak_vram_MiB': peak_vram_mib,
        'best_pass': best_pass,
        'best_tune_screen_NLL': best_nll,
        'pv_regime_scale_corr': pv_corr.tolist(),
        'tune6k_all': t6k_m,
        'ood_metrics': ood_metrics,
        'stress_shard080': stress_metrics,
        'gate_g_closure': closure,
        'history': history,
    }
    write_json(out_dir / 'run_summary.json', summary)
    print(f"[stage2 seed={seed}] DONE in {summary['wall_seconds']:.1f}s | Tune6k_BA4={t6k_m['BA4_pct']:.2f}% NLL={t6k_m['NLL']:.4f}", flush=True)
    return summary


def run_stage2_aggregate() -> dict[str, Any]:
    """Aggregate 3-seed Stage-II confirmation on Tune6k, Sec 12, 6xOOD, Shard-080 Stress & 1000x Bootstrap."""
    for sp in ('train', 'validation', 'tune_screen', 'tune6k', 'ood'):
        assert_unsealed_split(sp, stage=2)

    STAGE2_DIR.mkdir(parents=True, exist_ok=True)
    tune_meta = pd.read_parquet(assert_unsealed_path(CLEAN_ROOT / 'validation_meta/tune_truth_meta.parquet'))
    scr_df = pd.read_parquet(assert_unsealed_path(CLEAN_ROOT / 'data/tune_screen.parquet'), columns=KEY)
    sc_df = pd.read_parquet(assert_unsealed_path(CLEAN_ROOT / 'truth_sidecar/event_sidecar.parquet'))
    prov_080 = pd.read_parquet(assert_unsealed_path(CLEAN_ROOT / 'data/shards/080/provenance.parquet'), columns=KEY + ['label'])
    y_080 = prov_080['label'].to_numpy(dtype=np.int64)

    in_screen = np.asarray(pd.MultiIndex.from_frame(tune_meta[KEY]).isin(pd.MultiIndex.from_frame(scr_df[KEY])), dtype=bool)
    tune6k_m = ~in_screen
    ref_idx = pd.MultiIndex.from_frame(tune_meta[KEY])
    fam_codes = pd.MultiIndex.from_frame(tune_meta[FAMILY]).factorize()[0]

    labels = tune_meta['label'].to_numpy(dtype=np.int64)
    obs_dom = tune_meta['obs_domain_id'].to_numpy(dtype=np.int64)
    geom_w = tune_meta['geom_weight'].to_numpy(dtype=np.float64)
    ev_tune = tune_meta['energy_nu_GeV'].to_numpy(dtype=np.float64)
    pc_m = tune_meta['pixel_centered'].to_numpy(dtype=bool)
    fp_m = tune_meta['front_pinpoint_pixel_centered'].to_numpy(dtype=bool)
    interm_m = pc_m & (~fp_m)
    fort_m = tune_meta['fortune_module'].to_numpy(dtype=bool)

    common_e_m = (ev_tune >= 50.0) & (ev_tune <= 2312.0)
    nu_sign = (tune_meta['init_pdg'].to_numpy(dtype=np.int64) > 0).astype(np.int64)
    e_edges = np.geomspace(50.0, 2312.0, 11)
    eb_t = np.clip(np.digitize(ev_tune, e_edges) - 1, 0, 9)
    cell_t = eb_t * 2 + nu_sign
    p_cell = np.zeros((4, 20), dtype=np.float64)
    for c in range(4):
        mc = (labels == c) & common_e_m & tune6k_m
        cnt = np.bincount(cell_t[mc], minlength=20).astype(np.float64)
        p_cell[c] = (cnt + 0.5) / max(1.0, cnt.sum() + 10.0)
    p_cell_tgt = np.power(np.prod(p_cell, axis=0), 0.25)
    p_cell_tgt /= max(float(p_cell_tgt.sum()), 1e-12)
    matched_val_w = np.zeros(len(tune_meta), dtype=np.float64)
    for c in range(4):
        mc = (labels == c) & common_e_m & tune6k_m
        rw = np.clip(p_cell_tgt[cell_t[mc]] / np.maximum(p_cell[c, cell_t[mc]], 1e-9), 0.1, 10.0)
        rw /= max(float(rw.sum()), 1e-12)
        matched_val_w[mc] = rw

    pos_sc = pd.MultiIndex.from_frame(sc_df[KEY]).get_indexer(ref_idx)
    sc_tune = sc_df.iloc[pos_sc].reset_index(drop=True)
    v_true = sc_tune[['vertex_x_mm', 'vertex_y_mm', 'vertex_z_mm']].to_numpy(np.float64)
    v_valid = sc_tune['vertex_valid'].to_numpy(dtype=np.int64) > 0
    pos_080_in_tune = ref_idx.get_indexer(pd.MultiIndex.from_frame(prov_080[KEY]))

    vx_m, vy_m, vz_m = tune_meta['vertex_x_mm'].to_numpy(np.float64), tune_meta['vertex_y_mm'].to_numpy(np.float64), tune_meta['vertex_z_mm'].to_numpy(np.float64)
    mahlstedt_pc_m = tune6k_m & (vx_m >= -113.00) & (vx_m <= 112.99) & (vy_m >= -90.00) & (vy_m <= 65.99) & (vz_m >= -2986.99) & (vz_m <= -2486.99)
    mahlstedt_lg_m = tune6k_m & (vx_m >= -118.00) & (vx_m <= 272.00) & (vy_m >= -207.00) & (vy_m <= 183.00) & (vz_m >= -2986.99) & (vz_m <= -2004.88)

    seed_probs_t8k, seed_vpv_t8k, seed_spv_t8k, seed_ood_npz, seed_summaries = {}, {}, {}, {}, {}
    for s in STAGE2_SEEDS:
        sdir = STAGE2_DIR / f'seed_{s}'
        p8 = np.load(sdir / 'probs_tune8k.npy').astype(np.float64)
        p8 /= np.maximum(p8.sum(axis=1, keepdims=True), 1e-12)
        seed_probs_t8k[s] = p8
        seed_vpv_t8k[s] = np.load(sdir / 'v_pv_tune8k.npy').astype(np.float64)
        seed_spv_t8k[s] = np.load(sdir / 'scale_pv_tune8k.npy').astype(np.float64)
        seed_ood_npz[s] = np.load(sdir / 'ood_and_stress_probs.npz')
        seed_summaries[f'seed_{s}'] = json.loads((sdir / 'run_summary.json').read_text())

    p_ens_raw = np.mean([seed_probs_t8k[s] for s in STAGE2_SEEDS], axis=0)
    np.save(STAGE2_DIR / 'probs_tune8k_ens3_raw.npy', p_ens_raw.astype(np.float32))
    np.save(STAGE2_DIR / 'probs_tune6k_ens3_raw.npy', p_ens_raw[tune6k_m].astype(np.float32))

    temp_fit = _fit_scalar_temperature_on_screen(p_ens_raw[in_screen], labels[in_screen])
    p_ens_screen_temp = _apply_scalar_temperature(p_ens_raw, temp_fit['inv_T'])
    np.save(STAGE2_DIR / 'probs_tune8k_ens3_screen_temp.npy', p_ens_screen_temp.astype(np.float32))
    np.save(STAGE2_DIR / 'probs_tune6k_ens3_screen_temp.npy', p_ens_screen_temp[tune6k_m].astype(np.float32))

    lp_stack = np.stack([
        np.log(np.clip(seed_probs_t8k[s], 1e-8, 1.0)) - np.mean(np.log(np.clip(seed_probs_t8k[s], 1e-8, 1.0)), axis=1, keepdims=True)
        for s in STAGE2_SEEDS
    ], axis=0)
    global_blend_fit = _fit_domain_convex_blend(lp_stack[:, in_screen, :], labels[in_screen])
    p_ens_screen_dom = _apply_domain_convex_blend(lp_stack, global_blend_fit)
    for did in (1, 2, 3):
        m_fit = in_screen & (obs_dom == did)
        m_app = obs_dom == did
        if int(m_fit.sum()) >= 40:
            dfit = _fit_domain_convex_blend(lp_stack[:, m_fit, :], labels[m_fit])
            p_ens_screen_dom[m_app] = _apply_domain_convex_blend(lp_stack[:, m_app, :], dfit)
    np.save(STAGE2_DIR / 'probs_tune8k_ens3_screen_cal.npy', p_ens_screen_dom.astype(np.float32))
    np.save(STAGE2_DIR / 'probs_tune6k_ens3_screen_cal.npy', p_ens_screen_dom[tune6k_m].astype(np.float32))

    prec_list = [1.0 / np.maximum(seed_spv_t8k[s], 0.03) ** 2 for s in STAGE2_SEEDS]
    v_ens_t8k = np.sum([w * seed_vpv_t8k[s] for w, s in zip(prec_list, STAGE2_SEEDS)], axis=0) / np.maximum(np.sum(prec_list, axis=0), 1e-8)
    s_ens_t8k = 1.0 / np.sqrt(np.maximum(np.mean(prec_list, axis=0), 1e-8))

    def _eval_tune_prob_array(p_arr: np.ndarray) -> dict[str, Any]:
        s_tau = np.log(np.clip(p_arr[:, 2], 1e-8, 1.0)) - np.log(np.clip(1.0 - p_arr[:, 2], 1e-8, 1.0))
        dom_res = {OBS_DOMAIN_NAMES[did]: _ba4_and_nll(labels[tune6k_m & (obs_dom == did)], p_arr[tune6k_m & (obs_dom == did)]) for did in (1, 2, 3)}
        reg_res = {
            'Pinpoint_PixXY': _ba4_and_nll(labels[tune6k_m & pc_m], p_arr[tune6k_m & pc_m]),
            'FrontPinpoint_30mm': _ba4_and_nll(labels[tune6k_m & fp_m], p_arr[tune6k_m & fp_m]),
            'IntermPinpoint_150mm': _ba4_and_nll(labels[tune6k_m & interm_m], p_arr[tune6k_m & interm_m]),
            'Fortune_ScintOnly': _ba4_and_nll(labels[tune6k_m & fort_m], p_arr[tune6k_m & fort_m]),
        }
        return {
            'tune6k_screen_excl': _ba4_and_nll(labels[tune6k_m], p_arr[tune6k_m]),
            'tune6k_common_E_nu_matched_BA4_pct': _weighted_ba4_pct(labels, p_arr, matched_val_w),
            'tune6k_mass_weighted_2t_BA4_pct': _weighted_ba4_pct(labels[tune6k_m], p_arr[tune6k_m], geom_w[tune6k_m]),
            'tune6k_by_obs_domain': dom_res,
            'tune6k_by_truth_region': reg_res,
            'tune6k_tau_tail': {
                'split_half_All': _split_half_tail_eval_multi(s_tau, labels, tune_meta, tune6k_m),
                'split_half_Pinpoint': _split_half_tail_eval_multi(s_tau, labels, tune_meta, tune6k_m & pc_m),
                'split_half_D1_obs': _split_half_tail_eval_multi(s_tau, labels, tune_meta, tune6k_m & (obs_dom == 1)),
                'within_slice_1pct_All': _within_slice_tail_eval(s_tau, labels, tune6k_m, 0.01),
                'within_slice_1pct_D1_obs': _within_slice_tail_eval(s_tau, labels, tune6k_m & (obs_dom == 1), 0.01),
                'within_slice_0p1pct_All': _within_slice_tail_eval(s_tau, labels, tune6k_m, 0.001),
            },
            'section12_collaboration_slices': {
                'All_non_tau': _collaboration_section12_slice(p_arr, labels, tune6k_m),
                'Truth_Mahlstedt_PC': _collaboration_section12_slice(p_arr, labels, mahlstedt_pc_m, purity_target=0.90, natural_prior=(0.1145, 0.6397, 0.2459)),
                'Truth_Mahlstedt_LG': _collaboration_section12_slice(p_arr, labels, mahlstedt_lg_m, purity_target=0.80, natural_prior=(0.1195, 0.6399, 0.2406)),
                'D1_obs_non_tau': _collaboration_section12_slice(p_arr, labels, tune6k_m & (obs_dom == 1), purity_target=0.90, natural_prior=(0.1145, 0.6397, 0.2459)),
                'D2_obs_non_tau': _collaboration_section12_slice(p_arr, labels, tune6k_m & (obs_dom == 2), purity_target=0.80, natural_prior=(0.1195, 0.6399, 0.2406)),
            },
        }

    def _eval_pv(vh_arr: np.ndarray, sh_arr: np.ndarray) -> dict[str, Any]:
        out_pv = {}
        for tag, mk in (
            ('Tune6k_All_valid', tune6k_m & v_valid),
            ('Tune6k_FrontPinpoint_30mm', tune6k_m & fp_m & v_valid),
            ('Tune6k_IntermPinpoint_150mm', tune6k_m & interm_m & v_valid),
        ):
            dz = np.abs(vh_arr[mk, 2] - v_true[mk, 2])
            r3 = np.linalg.norm(vh_arr[mk] - v_true[mk], axis=-1)
            pulls = compute_student_t_pulls(
                torch.from_numpy(vh_arr), torch.from_numpy(sh_arr),
                torch.from_numpy(v_true), torch.from_numpy(mk.astype(np.int64)), df=3.0,
            )
            out_pv[tag] = {'n_events': int(mk.sum()), 'median_abs_dz_mm': float(np.median(dz)), 'median_3d_mm': float(np.median(r3)), **pulls}
        return out_pv

    per_seed = {
        f'seed_{s}': {
            'metrics': _eval_tune_prob_array(seed_probs_t8k[s]),
            'pv_audit': _eval_pv(seed_vpv_t8k[s], seed_spv_t8k[s]),
            'best_pass': seed_summaries[f'seed_{s}']['best_pass'],
            'wall_seconds': seed_summaries[f'seed_{s}']['wall_seconds'],
            'peak_vram_MiB': seed_summaries[f'seed_{s}']['peak_vram_MiB'],
        }
        for s in STAGE2_SEEDS
    }
    ba4_seeds = [per_seed[f'seed_{s}']['metrics']['tune6k_screen_excl']['BA4_pct'] for s in STAGE2_SEEDS]

    # OOD & Shard-080 Stress ensembles
    ood_ens = {}
    for ov in ALL_STAGE2_OOD:
        ps_list = [seed_ood_npz[s][f'ood_{ov}'].astype(np.float64) for s in STAGE2_SEEDS]
        p_eo = np.mean(ps_list, axis=0)
        if 'nc_' in ov:
            ood_ens[ov] = {
                'ens3_frac_pred_tau_pct': float((p_eo.argmax(1) == 2).mean() * 100.0),
                'ens3_frac_p_tau_gt_0p5_pct': float((p_eo[:, 2] > 0.5).mean() * 100.0),
                'ens3_mean_p_tau': float(p_eo[:, 2].mean()),
            }
        else:
            oroot = (STAGE325_ROOT / 'ood_shards' / ov) if 'w184' in ov else (CLEAN_ROOT / 'ood_shards' / ov)
            yo = pd.read_parquet(assert_unsealed_path(oroot / 'truth_meta.parquet'))['label'].to_numpy(np.int64)
            ood_ens[ov] = _ba4_and_nll(yo, p_eo)

    stress_ens = {}
    ba4_nom_ens = 0.0
    p_nom_ens = None
    for st_name, _, _ in STRESS_SPECS:
        p_es = np.mean([seed_ood_npz[s][f'stress_{st_name}'].astype(np.float64) for s in STAGE2_SEEDS], axis=0)
        m_es = _ba4_and_nll(y_080, p_es)
        if st_name == 'nominal':
            p_nom_ens = p_es
            ba4_nom_ens = m_es['BA4_pct']
        tv = float(0.5 * np.abs(p_es - p_nom_ens).sum(axis=1).mean()) if p_nom_ens is not None else 0.0
        stress_ens[st_name] = {'ens3_BA4_pct': m_es['BA4_pct'], 'ens3_delta_BA4_vs_nominal_pp': float(m_es['BA4_pct'] - ba4_nom_ens), 'ens3_NLL': m_es['NLL'], 'ens3_mean_TV_vs_nominal': tv}

    # Paired bootstrap vs Stage-325 Hybrid+NOMAD & Stage-326 50/50 Blend
    ref_probs: dict[str, np.ndarray] = {}
    s325_hyb = []
    for s in STAGE2_SEEDS:
        df_s, _ = _select_seed_prediction_df(assert_unsealed_path(STAGE325_ROOT / 'experiments' / f'v325_pf_hybrid_v2_nomad_s{s}'))
        p_cols = ['p_0', 'p_1', 'p_2', 'p_3'] if 'p_0' in df_s.columns else ['p0', 'p1', 'p2', 'p3']
        pos = pd.MultiIndex.from_frame(df_s[KEY]).get_indexer(ref_idx)
        s325_hyb.append(df_s.iloc[pos][p_cols].to_numpy(np.float64))
    ref_probs['Hybrid+NOMAD_ens3'] = np.mean(s325_hyb, axis=0)

    tip_s2 = STAGE1_S326_TRACKLET_DIR / 'stage2'
    dr_s2 = STAGE1_S326_DUAL_DIR / 'stage2'
    if (tip_s2 / 'seed_17/tune8k_predictions_seed17.parquet').is_file() and (dr_s2 / 'probs_tune8k_ens3_raw.npy').is_file():
        tip_seeds = []
        for s in STAGE2_SEEDS:
            df_t = pd.read_parquet(tip_s2 / f'seed_{s}/tune8k_predictions_seed{s}.parquet')
            pos_t = pd.MultiIndex.from_frame(df_t[KEY]).get_indexer(ref_idx)
            tip_seeds.append(df_t.iloc[pos_t][['p_0', 'p_1', 'p_2', 'p_3']].to_numpy(np.float64))
        p_tip_ens = np.mean(tip_seeds, axis=0)
        p_dr_ens = np.load(dr_s2 / 'probs_tune8k_ens3_raw.npy').astype(np.float64)
        ref_probs['S326_TrackletIPFormer_ens3'] = p_tip_ens
        ref_probs['S326_DualRegime_ens3'] = p_dr_ens
        ref_probs['S326_50_50_Blend_raw'] = 0.5 * (p_tip_ens + p_dr_ens)

    paired_boot = {}
    for bname, pb in ref_probs.items():
        for stag, mk in (('All_Tune6k', tune6k_m), ('D1_obs', tune6k_m & (obs_dom == 1)), ('Pinpoint', tune6k_m & pc_m)):
            paired_boot[f'UnifiedNet_ens3 - {bname} ({stag})'] = _paired_family_bootstrap_full(
                p_ens_raw, pb, labels, fam_codes, mk, n_boot=1000, seed=20263250
            )

    doc = {
        'UTC': datetime.now(timezone.utc).isoformat(),
        'stage': 'Stage-II Frozen PinpointUnifiedNet (Train32k x 3 seeds {17,42,2026})',
        'seed_stability': {
            'tune6k_BA4_per_seed_pct': ba4_seeds,
            'tune6k_BA4_mean_pct': float(np.mean(ba4_seeds)),
            'tune6k_BA4_sample_std_pct': float(np.std(ba4_seeds, ddof=1)),
        },
        'per_seed': per_seed,
        'raw_ens3': _eval_tune_prob_array(p_ens_raw),
        'screen_temp_ens3': _eval_tune_prob_array(p_ens_screen_temp),
        'screen_cal_ens3': _eval_tune_prob_array(p_ens_screen_dom),
        'pv_audit_ens3': _eval_pv(v_ens_t8k, s_ens_t8k),
        'ood_ens3': ood_ens,
        'stress_shard080_ens3': stress_ens,
        'paired_family_bootstrap_1000_seed20263250': paired_boot,
    }
    write_json(STAGE2_DIR / 'stage2_unified_results.json', doc)
    return doc


# ===========================================================================
# HTCondor Submission Utilities (bigbird20.cern.ch, --min-capability 8.0)
# ===========================================================================
def _package_source(repo: Path, archive: Path) -> None:
    import gzip

    archive.parent.mkdir(parents=True, exist_ok=True)
    with archive.open('wb') as raw_fh, gzip.GzipFile(filename='', mode='wb', fileobj=raw_fh, mtime=0) as gz_fh:
        with tarfile.open(fileobj=gz_fh, mode='w') as stream:
            paths = [repo / 'src/analysis/__init__.py']
            paths.extend(sorted((repo / 'src/analysis/utils').glob('*.py')))
            paths.extend(sorted((repo / 'src/analysis/flavour').rglob('*.py')))
            paths.extend(sorted((repo / 'src/analysis/flavour').rglob('*.cpp')))
            seen = set()
            for p in paths:
                if p in seen or not p.is_file():
                    continue
                seen.add(p)
                ti = stream.gettarinfo(str(p), arcname=str(p.relative_to(repo)))
                ti.mtime = 0
                ti.uid = 0
                ti.gid = 0
                ti.uname = ''
                ti.gname = ''
                with p.open('rb') as src_fh:
                    stream.addfile(ti, src_fh)


def _submit_condor_job(
    jname: str,
    cmd_args: str,
    archive: Path,
    chosen_schedd: str,
    *,
    cpus: int,
    mem: str,
    disk: str,
    flavour: str,
    use_gpu: bool,
) -> dict[str, Any]:
    job_dir = CONDOR_DIR / jname
    job_dir.mkdir(parents=True, exist_ok=True)
    (EXP_ROOT / 'logs').mkdir(parents=True, exist_ok=True)

    script = [
        '#!/bin/bash',
        'set -o pipefail',
        'which() { command -v "$@"; }; export -f which',
        'source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh',
        'set -e',
        'ls /eos/home-x/xcheng/FASER/Pinpoint/results >/dev/null',
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1',
        'export PINPOINT_V3_SCRATCH="${_CONDOR_SCRATCH_DIR:-/tmp}/pinpoint_uni_${PINPOINT_JOB_ID}"',
        'mkdir -p "$PINPOINT_V3_SCRATCH/code"',
        f'tar -xzf {shlex.quote(str(archive))} -C "$PINPOINT_V3_SCRATCH/code"',
        'export PYTHONPATH="$PINPOINT_V3_SCRATCH/code/src:${PYTHONPATH:-}"',
        'cd "$PINPOINT_V3_SCRATCH"',
        f'exec python3 -u -m analysis.flavour.nextgen.run_unified_pinpoint_net {cmd_args}',
    ]
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
        sub_lines.extend([
            '+AllowBackfillSlot = true',
            'request_gpus = 1',
            'require_gpus = ' + gpu_requirement(8.0, minimum_memory_mb=10000),
        ])
    sub_lines.append('queue 1')
    sub_path = job_dir / 'job.sub'
    sub_path.write_text('\n'.join(sub_lines) + '\n')

    resp = subprocess.run(['condor_submit', '-name', chosen_schedd, str(sub_path)], capture_output=True, text=True)
    if resp.returncode != 0:
        raise RuntimeError(f'condor_submit failed for {jname}: {resp.stderr}')
    return {'schedd': chosen_schedd, 'stdout': resp.stdout.strip()}


def submit_jobs(action: str, variants: list[str] | None = None) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[4]
    EXP_ROOT.mkdir(parents=True, exist_ok=True)
    archive = EXP_ROOT / 'code' / f'unified_src_{int(time.time())}.tar.gz'
    _package_source(repo, archive)
    chosen, _, _ = choose_schedd(EXP_ROOT)
    subs: dict[str, Any] = {}

    if action == 'submit-stage1-preextract':
        for part in range(9):
            subs[f's1_pre_{part}'] = _submit_condor_job(
                f's1_pre_{part}', f'--mode stage1-preextract --part {part}',
                archive, chosen, cpus=8, mem='16GB', disk='12GB', flavour='microcentury', use_gpu=False,
            )
    elif action == 'submit-stage1-train':
        vlist = variants or list(VARIANT_CONFIGS.keys())
        for v in vlist:
            subs[v] = _submit_condor_job(
                v, f'--mode stage1-train --variant {v}',
                archive, chosen, cpus=4, mem='16000MB', disk='20GB', flavour='workday', use_gpu=True,
            )
    elif action == 'submit-stage2-preextract':
        for part in range(11):
            subs[f's2_pre_{part}'] = _submit_condor_job(
                f's2_pre_{part}', f'--mode stage2-preextract --part {part}',
                archive, chosen, cpus=8, mem='16GB', disk='12GB', flavour='microcentury', use_gpu=False,
            )
    elif action == 'submit-stage2-train':
        for s in STAGE2_SEEDS:
            subs[f's2_seed_{s}'] = _submit_condor_job(
                f's2_seed_{s}', f'--mode stage2-train --seed {s}',
                archive, chosen, cpus=4, mem='16500MB', disk='20GB', flavour='workday', use_gpu=True,
            )
    else:
        raise ValueError(f'Unknown submit action {action!r}')
    write_json(EXP_ROOT / f'submissions_{action}.json', subs)
    return subs


def main() -> None:
    guard = SealAuditGuard()
    guard.__enter__()
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            '--mode',
            required=True,
            choices=(
                'stage1-preextract',
                'stage1-train',
                'stage1-aggregate',
                'stage2-preextract',
                'stage2-train',
                'stage2-aggregate',
                'submit-stage1-preextract',
                'submit-stage1-train',
                'submit-stage2-preextract',
                'submit-stage2-train',
            ),
        )
        parser.add_argument('--part', type=int, default=0)
        parser.add_argument('--variant', type=str, default='s1_unified_pinpoint_net')
        parser.add_argument('--variants', nargs='*', default=None)
        parser.add_argument('--seed', type=int, default=42)
        args = parser.parse_args()

        is_stage2 = 'stage2' in args.mode
        if not is_stage2:
            install_stage1_only_hook()

        if args.mode == 'stage1-preextract':
            print(json.dumps(run_stage1_preextract(args.part), indent=2))
        elif args.mode == 'stage1-train':
            run_stage1_train(args.variant)
        elif args.mode == 'stage1-aggregate':
            run_stage1_aggregate()
        elif args.mode == 'stage2-preextract':
            print(json.dumps(run_stage2_preextract(args.part), indent=2))
        elif args.mode == 'stage2-train':
            run_stage2_train(args.seed)
        elif args.mode == 'stage2-aggregate':
            print(json.dumps(run_stage2_aggregate()['seed_stability'], indent=2))
        else:
            print(json.dumps(submit_jobs(args.mode, variants=args.variants), indent=2))
    finally:
        guard.__exit__(None, None, None)


if __name__ == '__main__':
    main()
