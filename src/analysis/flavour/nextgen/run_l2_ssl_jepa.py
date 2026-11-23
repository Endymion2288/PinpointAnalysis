"""Runner for Experiment L2: PinpointSSL-JEPA Pretraining, Pre-Calibration, Equal-Compute
Stage-I Benchmark Matrix, and (if Stage-I gates pass) Frozen Stage-II 3-Seed Evaluation.

Enforces strict SealAuditGuard isolation (`Cal-A/Cut-B/Confirm/GENIE 016..028` = 0 accesses;
Stage-I additionally blocks `Tune8k/080..099` = 0 accesses).
"""

from __future__ import annotations

import argparse
import atexit
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import copy
from datetime import datetime, timezone
import gc
import hashlib
import io
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
from scipy.stats import pearsonr, spearmanr, t as scipy_t
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler
import zstandard as zstd

from analysis.flavour.inventory import write_json
from analysis.flavour.nextgen.dual_regime_modalnet import (
    apply_y_mirror_dual_regime_item,
    collate_dual_regime,
)
from analysis.flavour.nextgen.pinpoint_ssl_jepa import (
    PinpointSSLJEPA,
    compute_jepa_finetune_loss,
    compute_jepa_finetune_window_denominators,
    compute_ssl_pretrain_loss,
    extract_jepa_item,
    sample_ssl_masks,
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
)
from analysis.flavour.nextgen.run_unified_pinpoint_net import (
    _calibrate_pv_regime_scales_on_train,
    _deterministic_mirror_bit,
    _export_latent_cache,
    _package_source,
    _resolve_fast_cache_root,
    _stage_pkl_dir_to_shm,
    _update_ema,
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
from analysis.flavour.v2.partitions import FAMILY, KEY
from analysis.flavour.v2.submit import choose_schedd, gpu_requirement
from analysis.flavour.v3.common.domains import (
    OBS_DOMAIN_NAMES,
    PE_PER_MEV_CENTER_CAL,
    compute_observable_containment,
    cumulative_radiation_length,
    detector_bounds,
)
from analysis.flavour.v3.v1_b1pp.spectrometer import SIGMA_SAGITTA_Y_MM
from analysis.flavour.v3.v2_pinpointformer.pixel_tokenizer import _compute_plane_neighborhoods
from analysis.flavour.v3.v2_pinpointformer.train import move_batch_to_device
from analysis.flavour.v3.v4_tau.ensemble import _apply_domain_convex_blend, _fit_domain_convex_blend


EXP_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/exp_l2_jepa')
SSL_DATA_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/l2_ssl_300k')
STAGE1_CACHE_DIR = EXP_ROOT / '_stage1_item_cache'
STAGE2_DIR = EXP_ROOT / 'stage2'
STAGE2_CACHE_DIR = STAGE2_DIR / '_stage2_item_cache'
CONDOR_DIR = Path.home() / 'condor/Pinpoint/stage328_l2_jepa'

STAGE1_S326_TRACKLET_DIR = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/exp_tracklet_ipformer')
STAGE1_S326_DUAL_DIR = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/exp_dual_regime')
L1_EXP_DIR = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/exp_l1_gn2')

STAGE1_FORBIDDEN_RE = re.compile(
    r'(?:^|/)shards/0(?:8\d|9\d)(?:/|$)|validation_meta|tune_screen\.parquet|tune_truth_meta\.parquet',
    re.IGNORECASE,
)
_STAGE1_HOOK_INSTALLED = False


def install_stage1_only_hook() -> None:
    """Block Tune8k/080..099 filesystem syscalls strictly during SSL pretraining & Stage-I."""
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


# ---------------------------------------------------------------------------
# Stage-I Controlled Benchmark Matrix Configuration
# ---------------------------------------------------------------------------
SSL_VARIANTS: dict[str, dict[str, Any]] = {
    'ssl_full': {
        'w_jepa': 1.0,
        'w_gmm': 0.25,
        'w_stereo': 0.50,
        'w_spec': 0.50,
        'w_vicreg': 0.15,
        'shortcut_point_mask': False,
    },
    'abl_ssl_no_spec_precal': {
        'w_jepa': 1.0,
        'w_gmm': 0.25,
        'w_stereo': 0.50,
        'w_spec': 0.0,
        'w_vicreg': 0.15,
        'shortcut_point_mask': False,
    },
    'abl_ssl_random_point_mask_shortcut': {
        'w_jepa': 1.0,
        'w_gmm': 0.25,
        'w_stereo': 0.50,
        'w_spec': 0.50,
        'w_vicreg': 0.15,
        'shortcut_point_mask': True,
    },
}

STAGE1_MATRIX: dict[str, dict[str, Any]] = {
    'ssl300k_jepa_ft_100pct': {
        'ssl_variant': 'ssl_full',
        'train_frac': 1.00,
        'passes': 20,
        'lr': 6.0e-4,
        'xi': 0.78,
        'drop_path': 0.10,
        'head_dropout': 0.15,
        'y_mirror': True,
    },
    'sup_scratch_20p': {
        'ssl_variant': None,
        'train_frac': 1.00,
        'passes': 20,
        'lr': 8.0e-4,
        'xi': 1.00,
        'drop_path': 0.10,
        'head_dropout': 0.15,
        'y_mirror': True,
    },
    'sup_scratch_equal_compute': {
        'ssl_variant': None,
        'train_frac': 1.00,
        'passes': 255,  # Dynamically refined via math.ceil from measured SSL/Sup FLOP ratio!
        'lr': 8.0e-4,
        'xi': 1.00,
        'drop_path': 0.18,
        'head_dropout': 0.20,
        'weight_decay': 0.05,
        'batch_size': 256,
        'micro_bs': 128,
        'grad_accum': 2,
        'y_mirror': True,
        'match_equal_compute': True,
    },
    'ssl300k_jepa_ft_25pct': {
        'ssl_variant': 'ssl_full',
        'train_frac': 0.25,
        'passes': 30,
        'lr': 6.0e-4,
        'xi': 0.78,
        'drop_path': 0.10,
        'head_dropout': 0.15,
        'y_mirror': True,
    },
    'sup_scratch_25pct': {
        'ssl_variant': None,
        'train_frac': 0.25,
        'passes': 30,
        'lr': 8.0e-4,
        'xi': 1.00,
        'drop_path': 0.12,
        'head_dropout': 0.18,
        'y_mirror': True,
    },
    'ssl300k_jepa_ft_10pct': {
        'ssl_variant': 'ssl_full',
        'train_frac': 0.10,
        'passes': 40,
        'lr': 6.0e-4,
        'xi': 0.78,
        'drop_path': 0.10,
        'head_dropout': 0.15,
        'y_mirror': True,
    },
    'sup_scratch_10pct': {
        'ssl_variant': None,
        'train_frac': 0.10,
        'passes': 40,
        'lr': 8.0e-4,
        'xi': 1.00,
        'drop_path': 0.12,
        'head_dropout': 0.18,
        'y_mirror': True,
    },
    'abl_ssl_no_spec_precal_ft_100pct': {
        'ssl_variant': 'abl_ssl_no_spec_precal',
        'train_frac': 1.00,
        'passes': 20,
        'lr': 6.0e-4,
        'xi': 0.78,
        'drop_path': 0.10,
        'head_dropout': 0.15,
        'y_mirror': True,
    },
    'abl_ssl_random_point_mask_shortcut_ft_100pct': {
        'ssl_variant': 'abl_ssl_random_point_mask_shortcut',
        'train_frac': 1.00,
        'passes': 20,
        'lr': 6.0e-4,
        'xi': 0.78,
        'drop_path': 0.10,
        'head_dropout': 0.15,
        'y_mirror': True,
    },
}


# ---------------------------------------------------------------------------
# Zstd CSR `.npz.zst` Loader for `l2_ssl_300k`
# ---------------------------------------------------------------------------
_SSL_DROP_KEYS = frozenset({'family_hi', 'family_lo', 'avr_pv_mm', 'trk_kin', 'trk_morph', 'trk_mask'})


def _load_npz_zst(path: Path, keep_keys: set[str] | frozenset[str] | None = None) -> dict[str, np.ndarray]:
    assert_unsealed_path(path)
    dctx = zstd.ZstdDecompressor()
    raw = dctx.decompress(path.read_bytes())
    with np.load(io.BytesIO(raw), allow_pickle=False) as npz:
        if keep_keys is not None:
            return {k: npz[k] for k in npz.files if k in keep_keys}
        return {k: npz[k] for k in npz.files if k not in _SSL_DROP_KEYS}


def _reconstruct_pix_feat_22(
    pix_xyz: np.ndarray,
    pix_plane: np.ndarray,
    pix_moments: np.ndarray,
    pix_plane_summary: np.ndarray,
    x_c: float,
    y_c: float,
    z_mid: float,
) -> np.ndarray:
    """Reconstruct exact 22-d `pix_feat` from CSR shard fields (supports 10d or 19d `pix_moments`)."""
    n_cl = len(pix_xyz)
    if n_cl == 0:
        return np.empty((0, 22), dtype=np.float32)

    mom = np.asarray(pix_moments, dtype=np.float32)
    coords = np.column_stack([
        (pix_xyz[:, 0].astype(np.float32) - np.float32(x_c)) / np.float32(135.0),
        (pix_xyz[:, 1].astype(np.float32) - np.float32(y_c)) / np.float32(100.0),
        (pix_xyz[:, 2].astype(np.float32) - np.float32(z_mid)) / np.float32(1000.0),
    ]).astype(np.float32)
    if mom.shape[1] >= 19:
        return np.concatenate([coords, mom[:, :19]], axis=1).astype(np.float32)

    # 10d `pix_moments` = cols 3..12; reconstruct cols 13..21 cleanly
    pf = np.zeros((n_cl, 22), dtype=np.float32)
    pf[:, 0:3] = coords
    pf[:, 3:13] = mom[:, :10]

    pl = pix_plane.astype(np.int64)
    n_hits = np.expm1(np.clip(mom[:, 0].astype(np.float64), 0.0, 4.0) * 6.0).clip(min=1.0)
    (
        cnt_r015, hits_r015, cnt_r040, _, cnt_r1, cnt_r5, hits_r5,
    ) = _compute_plane_neighborhoods(pl, pix_xyz[:, 0].astype(np.float64), pix_xyz[:, 1].astype(np.float64), n_hits)

    plane_xc = pix_plane_summary[:, 7].astype(np.float64) * 135.0 + x_c
    plane_yc = pix_plane_summary[:, 6].astype(np.float64) * 100.0 + y_c
    dx_core = pix_xyz[:, 0].astype(np.float64) - plane_xc[pl]
    dy_core = pix_xyz[:, 1].astype(np.float64) - plane_yc[pl]
    r_core = np.hypot(dx_core, dy_core)

    pf[:, 13] = np.log1p(cnt_r1) / 4.0
    pf[:, 14] = np.log1p(cnt_r5) / 5.0
    pf[:, 15] = np.log1p(hits_r5) / 7.0
    pf[:, 16] = np.arcsinh(dx_core / 0.10) / 4.0
    pf[:, 17] = np.arcsinh(dy_core / 0.10) / 4.0
    pf[:, 18] = np.arcsinh(r_core / 0.08) / 4.0
    pf[:, 19] = np.log1p(cnt_r015) / 3.0
    pf[:, 20] = np.log1p(hits_r015) / 5.0
    pf[:, 21] = np.log1p(cnt_r040) / 4.0
    return pf


def _bar_pe_to_val(bpe_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw `bar_pe` (i32) to `(keep_mask, bar_val [f32, 2])` matching `BASE_OBS_CFG`."""
    bpe = np.maximum(0.0, np.asarray(bpe_raw, dtype=np.float64))
    keep = bpe >= 2.5
    bpe_k = bpe[keep]
    if len(bpe_k) == 0:
        return keep, np.empty((0, 2), dtype=np.float32)
    frac = np.clip(bpe_k / 14400.0, 0.0, 0.98)
    pe_lin = -14400.0 * np.log(1.0 - frac)
    e_cal = pe_lin / PE_PER_MEV_CENTER_CAL
    bval = np.column_stack([
        np.log1p(np.maximum(e_cal, 0.0)) / 10.0,
        np.log1p(np.maximum(bpe_k, 0.0)) / 15.0,
    ]).astype(np.float32)
    return keep, bval


def _csr_shard_event_to_item(
    sh: dict[str, np.ndarray],
    ev_i: int,
    x_c: float,
    y_c: float,
    z_mid: float,
) -> dict[str, Any]:
    """Unpack one event from an in-memory CSR `.npz.zst` shard dict into a collatable item."""
    p0, p1 = int(sh['pix_ptr'][ev_i]), int(sh['pix_ptr'][ev_i + 1])
    pxyz = np.asarray(sh['pix_xyz_mm'][p0:p1], dtype=np.float32)
    ppl = np.asarray(sh['pix_plane'][p0:p1], dtype=np.int8)
    pps = np.asarray(sh['pix_plane_summary'][ev_i], dtype=np.float32)
    if 'pix_feat' in sh:
        raw_pf = np.asarray(sh['pix_feat'][p0:p1], dtype=np.float32)
        if raw_pf.shape[1] >= 22:
            pfeat = raw_pf[:, :22].astype(np.float32)
        else:
            pfeat = _reconstruct_pix_feat_22(pxyz, ppl, raw_pf, pps, x_c, y_c, z_mid)
    elif 'pix_feat_16' in sh:
        f16 = np.asarray(sh['pix_feat_16'][p0:p1], dtype=np.float32)
        n_cl = len(pxyz)
        if n_cl == 0:
            pfeat = np.empty((0, 22), dtype=np.float32)
        else:
            pfeat = np.zeros((n_cl, 22), dtype=np.float32)
            pfeat[:, 0] = (pxyz[:, 0] - np.float32(x_c)) / np.float32(135.0)
            pfeat[:, 1] = (pxyz[:, 1] - np.float32(y_c)) / np.float32(100.0)
            pfeat[:, 2] = (pxyz[:, 2] - np.float32(z_mid)) / np.float32(1000.0)
            pfeat[:, 3:16] = f16[:, :13]
            pl = ppl.astype(np.int64)
            plane_xc = pps[:, 7].astype(np.float64) * 135.0 + x_c
            plane_yc = pps[:, 6].astype(np.float64) * 100.0 + y_c
            dx_core = pxyz[:, 0].astype(np.float64) - plane_xc[pl]
            dy_core = pxyz[:, 1].astype(np.float64) - plane_yc[pl]
            r_core = np.hypot(dx_core, dy_core)
            pfeat[:, 16] = (np.arcsinh(dx_core / 0.10) / 4.0).astype(np.float32)
            pfeat[:, 17] = (np.arcsinh(dy_core / 0.10) / 4.0).astype(np.float32)
            pfeat[:, 18] = (np.arcsinh(r_core / 0.08) / 4.0).astype(np.float32)
            pfeat[:, 19:22] = f16[:, 13:16]
    else:
        pfeat = _reconstruct_pix_feat_22(pxyz, ppl, sh['pix_moments'][p0:p1], pps, x_c, y_c, z_mid)

    t0, t1 = int(sh['pat_ptr'][ev_i]), int(sh['pat_ptr'][ev_i + 1])
    patches = np.asarray(sh['pix_patches'][t0:t1], dtype=np.float16)
    pat_cl = np.asarray(sh['patch_cluster_idx'][t0:t1], dtype=np.int16)

    b0, b1 = int(sh['bar_ptr'][ev_i]), int(sh['bar_ptr'][ev_i + 1])
    keep_b, bval = _bar_pe_to_val(sh['bar_pe'][b0:b1])
    bp = np.asarray(sh['bar_panel'][b0:b1], dtype=np.int16)[keep_b]
    bh = np.asarray(sh['bar_horiz'][b0:b1], dtype=np.int8)[keep_b]
    bi = np.asarray(sh['bar_idx'][b0:b1], dtype=np.int16)[keep_b]
    btof = np.asarray(sh['bar_hit_time_ns'][b0:b1], dtype=np.float32)[keep_b]

    h0, h1 = int(sh['hit_ptr'][ev_i]), int(sh['hit_ptr'][ev_i + 1])
    thits = np.asarray(sh['trk_hits'][h0:h1], dtype=np.float32)

    return {
        'index': int(ev_i),
        'obs_domain_id': int(sh['obs_domain_id'][ev_i]),
        'c_obs': np.asarray(sh['c_obs'][ev_i], dtype=np.float32),
        'cal_start': np.asarray(sh['cal_start'][ev_i], dtype=np.float32),
        'raw_pixel_hits': int(len(pxyz)),
        'pix_xyz_mm': pxyz,
        'pix_feat': pfeat,
        'pix_plane': ppl,
        'pix_plane_summary': pps,
        'pix_patches': patches,
        'patch_cluster_idx': pat_cl,
        'bar_panel': bp,
        'bar_horiz': bh,
        'bar_idx': bi,
        'bar_val': bval,
        'bar_time_ns': btof,
        'trk_cand': np.asarray(sh['trk_cand'][ev_i], dtype=np.float32),
        'trk_cand_mask': np.asarray(sh['trk_cand_mask'][ev_i], dtype=bool),
        'trk_station_counts': np.asarray(sh['trk_station_counts'][ev_i], dtype=np.float32),
        'trk_hits': thits,
    }


def _stage_ssl_shards_to_shm(src_dir: Path, prefix: str) -> tuple[list[Path], Path | None]:
    """Copy `.npz.zst` shards to `/dev/shm` if space permits for zero-EOS-latency reads."""
    shard_paths = sorted(assert_unsealed_path(p) for p in src_dir.glob('*.npz.zst'))
    if not shard_paths:
        raise FileNotFoundError(f'No .npz.zst SSL shards found under {src_dir}')
    total_bytes = sum(p.stat().st_size for p in shard_paths)
    fast_root = _resolve_fast_cache_root(min_free_gb=max(4.0, total_bytes / (1024 ** 3) + 2.0))
    if fast_root != Path('/dev/shm'):
        return shard_paths, None
    dst_dir = Path(tempfile.mkdtemp(prefix=f'{prefix}_', dir=str(fast_root)))

    def _cleanup() -> None:
        shutil.rmtree(dst_dir, ignore_errors=True)

    atexit.register(_cleanup)
    with ThreadPoolExecutor(max_workers=8) as pool:
        local_paths = list(pool.map(lambda p: Path(shutil.copy2(p, dst_dir / p.name)), shard_paths))
    return local_paths, dst_dir


class InMemorySSLShardsDataset(Dataset):
    """In-memory CSR shard dataset holding >=300k unlabeled D1∪D2 events cleanly in RAM."""

    def __init__(
        self,
        shards: list[dict[str, np.ndarray]],
        bounds: dict[str, Any],
    ) -> None:
        self.shards = shards
        self.x_c = float(bounds['x_pix_center'])
        self.y_c = float(bounds['y_scint_center'])
        self.z_mid = float(bounds['z_mid'])
        self.shard_sizes = [int(len(sh['obs_domain_id'])) for sh in shards]
        self.cum_sizes = np.cumsum([0] + self.shard_sizes, dtype=np.int64)
        self.total_len = int(self.cum_sizes[-1])

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s_idx = int(np.searchsorted(self.cum_sizes[1:], idx, side='right'))
        local_i = int(idx - self.cum_sizes[s_idx])
        return _csr_shard_event_to_item(self.shards[s_idx], local_i, self.x_c, self.y_c, self.z_mid)


class _SSLCollator:
    def __init__(self, y_scint_center: float, n_bars: int) -> None:
        self.y_scint_center = float(y_scint_center)
        self.n_bars = int(n_bars)

    def __call__(self, xs: list[dict]) -> dict[str, torch.Tensor]:
        return collate_dual_regime(xs, mirror_rng=None, y_scint_center=self.y_scint_center, n_bars=self.n_bars)


def _available_ram_gb() -> float:
    avail = 16.0
    try:
        with open('/proc/meminfo') as fh:
            for line in fh:
                if line.startswith('MemAvailable:'):
                    avail = float(line.split()[1]) / (1024.0 * 1024.0)
                    break
    except OSError:
        pass
    for max_p, cur_p in (
        ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory.current'),
        ('/sys/fs/cgroup/memory/memory.limit_in_bytes', '/sys/fs/cgroup/memory/memory.usage_in_bytes'),
    ):
        try:
            mx_s = Path(max_p).read_text().strip()
            if mx_s.isdigit():
                mx = int(mx_s)
                if 0 < mx < (1 << 50):
                    cur = int(Path(cur_p).read_text().strip())
                    avail = min(avail, max(0.0, (mx - cur) / (1024.0 ** 3)))
        except OSError:
            pass
    return avail


def _load_ssl_shards_parallel(paths: list[Path], max_workers: int = 8, keep_keys: set[str] | frozenset[str] | None = None) -> list[dict[str, np.ndarray]]:
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(lambda p: _load_npz_zst(p, keep_keys=keep_keys), paths))


# ---------------------------------------------------------------------------
# Unlabeled Penetrating-Muon Pre-Calibration Evaluation (Task 3)
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_unlabeled_muon_precalibration(
    model: PinpointSSLJEPA,
    items: list[dict],
    bounds: dict,
    device: torch.device,
    batch_size: int = 128,
) -> dict[str, Any]:
    """Evaluate unlabeled Task-3 cross-Magnet-0 spectrometer prediction with full spectrometer masked."""
    _set_ieee_fp32(True)
    model.eval()
    yc, nb = float(bounds['y_scint_center']), int(bounds['n_bars'])
    n_st = model.backbone.n_st

    y_mu_l, p_mu_l = [], []
    tgt_kin_l, pred_kin_l = [], []

    for st in range(0, len(items), batch_size):
        sub = items[st : st + batch_size]
        batch = move_batch_to_device(collate_dual_regime(sub, mirror_rng=None, y_scint_center=yc, n_bars=nb), device)
        bsz = int(batch['c_obs'].shape[0])
        zero_pix_drop = torch.zeros((bsz, 15), dtype=torch.bool, device=device)
        zero_st_drop = torch.zeros((bsz, n_st, 2), dtype=torch.bool, device=device)
        full_spec_drop = torch.ones((bsz,), dtype=torch.bool, device=device)

        with torch.amp.autocast('cuda', enabled=False):
            out = model.forward_ssl_student(
                batch,
                pix_plane_drop_mask=zero_pix_drop,
                stereo_panel_drop_mask=zero_st_drop,
                spec_drop_mask=full_spec_drop,
            )
            sp = out['spec_pred'].float()
            tc = batch['trk_cand'].float()
            tcm = batch['trk_cand_mask'].bool()
            has_3st = tcm[:, 0] & (tc[:, 0, 0] > 0.75)

            y_mu_l.append(has_3st.cpu().numpy())
            p_mu_l.append(torch.sigmoid(sp[:, 0]).cpu().numpy())
            if has_3st.any():
                sy_mm = SIGMA_SAGITTA_Y_MM * torch.sinh((5.0 * tc[has_3st, 0, 6]).clamp(-9.0, 9.0))
                ln_abs_sy = torch.log(sy_mm.abs() + 0.05)
                tgt = torch.stack([
                    tc[has_3st, 0, 2],
                    tc[has_3st, 0, 3],
                    tc[has_3st, 0, 4],
                    tc[has_3st, 0, 5],
                    ln_abs_sy,
                ], dim=-1)
                tgt_kin_l.append(tgt.cpu().numpy())
                pred_kin_l.append(sp[has_3st, 1:6].cpu().numpy())

    y_mu = np.concatenate(y_mu_l, axis=0).astype(np.int64)
    p_mu = np.concatenate(p_mu_l, axis=0).astype(np.float64)
    res: dict[str, Any] = {
        'n_events': int(len(y_mu)),
        'n_penetrating_mu': int(y_mu.sum()),
        'mu_fraction_pct': float(y_mu.mean() * 100.0),
    }
    if 0 < int(y_mu.sum()) < len(y_mu):
        res['mu_presence_AUROC'] = float(roc_auc_score(y_mu, p_mu))
        res['mu_presence_AUPRC'] = float(average_precision_score(y_mu, p_mu))

    if tgt_kin_l:
        tk = np.concatenate(tgt_kin_l, axis=0).astype(np.float64)
        pk = np.concatenate(pred_kin_l, axis=0).astype(np.float64)
        if len(tk) >= 10:
            r_sy, p_sy = pearsonr(tk[:, 4], pk[:, 4])
            rho_sy, _ = spearmanr(tk[:, 4], pk[:, 4])
            r_tx, _ = pearsonr(tk[:, 2], pk[:, 2])
            r_ty, _ = pearsonr(tk[:, 3], pk[:, 3])
            r_x0, _ = pearsonr(tk[:, 0], pk[:, 0])
            r_y0, _ = pearsonr(tk[:, 1], pk[:, 1])
            res['ln_abs_sy_pearson_r'] = float(r_sy)
            res['ln_abs_sy_pearson_p'] = float(p_sy)
            res['ln_abs_sy_spearman_rho'] = float(rho_sy)
            res['ln_abs_sy_MAE'] = float(np.abs(tk[:, 4] - pk[:, 4]).mean())
            res['exit_slope_tx_pearson_r'] = float(r_tx)
            res['exit_slope_ty_pearson_r'] = float(r_ty)
            res['exit_slope_RMSE_rad'] = float(np.sqrt(np.mean(((tk[:, 2:4] - pk[:, 2:4]) * 0.1) ** 2)))
            res['entry_x0_pearson_r'] = float(r_x0)
            res['entry_y0_pearson_r'] = float(r_y0)
    return res


def _measure_step_flops(
    model: PinpointSSLJEPA,
    teacher_bb: Any,
    batch_ssl: dict[str, torch.Tensor],
    batch_sup: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Measure exact forward+backward FLOPs per sample for SSL vs Supervised using FlopCounterMode."""
    from torch.utils.flop_counter import FlopCounterMode

    bs_ssl = int(batch_ssl['c_obs'].shape[0])
    bs_sup = int(batch_sup['c_obs'].shape[0])
    model.eval()
    masks = sample_ssl_masks(batch_ssl, n_st=model.backbone.n_st)

    with FlopCounterMode(display=False) as fc_ssl:
        loss_ssl, _ = compute_ssl_pretrain_loss(model, teacher_bb, batch_ssl, masks)
        loss_ssl.backward()
    ssl_flops_per_sample = float(fc_ssl.get_total_flops()) / max(1, bs_ssl)
    model.zero_grad(set_to_none=True)

    with FlopCounterMode(display=False) as fc_sup:
        loss_sup, _ = compute_jepa_finetune_loss(model, batch_sup)
        loss_sup.backward()
    sup_flops_per_sample = float(fc_sup.get_total_flops()) / max(1, bs_sup)
    model.zero_grad(set_to_none=True)

    return {
        'ssl_flops_per_sample': ssl_flops_per_sample,
        'sup_flops_per_sample': sup_flops_per_sample,
        'ssl_to_sup_ratio': ssl_flops_per_sample / max(sup_flops_per_sample, 1.0),
    }


# ---------------------------------------------------------------------------
# Phase 1: SSL Pretraining (`l2_ssl_300k`) & Unlabeled Pre-Calibration
# ---------------------------------------------------------------------------
def run_ssl_pretrain(
    ssl_variant: str = 'ssl_full',
    epochs: int = 15,
    batch_size: int = 512,
    lr: float = 1.0e-3,
    seed: int = 42,
    *,
    micro_bs: int = 64,
    out_dir_name: str | None = None,
) -> dict[str, Any]:
    if ssl_variant not in SSL_VARIANTS:
        raise ValueError(f'Unknown ssl_variant {ssl_variant!r}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for PinpointSSL-JEPA pretraining')
    assert_unsealed_split('internal_train', stage=1)
    assert_unsealed_split('internal_val', stage=1)

    t0 = time.perf_counter()
    utc_started = datetime.now(timezone.utc).isoformat()
    device = torch.device('cuda:0')
    torch.manual_seed(seed)
    np.random.seed(seed)

    out_dir = EXP_ROOT / (out_dir_name or ssl_variant)
    out_dir.mkdir(parents=True, exist_ok=True)
    geom = json.loads(assert_unsealed_path(CLEAN_ROOT / 'geometry.json').read_text())
    bounds = detector_bounds(geom)
    bounds['_plane_x0'] = np.array(
        [cumulative_radiation_length(float(z), geom) for z in bounds['pixel_Zpos']],
        dtype=np.float32,
    )
    yc, nb = float(bounds['y_scint_center']), int(bounds['n_bars'])

    train_dir = assert_unsealed_path(SSL_DATA_ROOT / 'train_shards')
    val_dir = assert_unsealed_path(SSL_DATA_ROOT / 'ssl_val_calib')

    local_tr_paths, shm_tr_dir = _stage_ssl_shards_to_shm(train_dir, prefix='l2_ssl_shm')
    val_paths = sorted(assert_unsealed_path(p) for p in val_dir.glob('*.npz.zst'))
    val_shards = _load_ssl_shards_parallel(val_paths, max_workers=min(8, max(1, len(val_paths))))
    val_calib_items = [
        _csr_shard_event_to_item(sh, i, float(bounds['x_pix_center']), yc, float(bounds['z_mid']))
        for sh in val_shards
        for i in range(len(sh['obs_domain_id']))
    ]
    del val_shards

    avail_gb = _available_ram_gb()
    print(f'[{ssl_variant}] Found {len(local_tr_paths)} train shards, {len(val_calib_items)} ssl_val_calib events; MemAvailable={avail_gb:.1f} GB', flush=True)

    cap = torch.cuda.get_device_capability(0)
    amp_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
    amp_ok = cap[0] >= 7

    model_cfg = {'dim': 176, 'heads': 8, 'drop_path': 0.10, 'head_dropout': 0.15}
    student = PinpointSSLJEPA(geom, model_cfg).to(device)
    teacher = copy.deepcopy(student.backbone).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    vcfg = SSL_VARIANTS[ssl_variant]
    opt = torch.optim.AdamW(
        student.build_param_groups(lr, weight_decay=0.04, xi=1.0, include_ssl_heads=True),
        lr=lr,
        weight_decay=0.04,
        betas=(0.9, 0.95),
    )

    # Load all training shards into RAM if >= 42 GB free (H100/Kubeflow), else stream in chunks
    if avail_gb >= 42.0:
        tr_shards = _load_ssl_shards_parallel(local_tr_paths, max_workers=min(12, os.cpu_count() or 8))
        shard_groups = [tr_shards]
        n_total_events = sum(len(sh['obs_domain_id']) for sh in tr_shards)
        if shm_tr_dir is not None:
            shutil.rmtree(shm_tr_dir, ignore_errors=True)
            shm_tr_dir = None
    else:
        tr_shards = []
        shard_groups = None
        cnt_sh = _load_ssl_shards_parallel(local_tr_paths, max_workers=8, keep_keys={'obs_domain_id'})
        n_total_events = sum(len(sh['obs_domain_id']) for sh in cnt_sh)
        del cnt_sh

    steps_per_epoch = math.ceil(n_total_events / batch_size)
    total_steps = epochs * steps_per_epoch
    warmup_steps = max(1, int(0.05 * total_steps))
    min_lr_ratio = 1.0e-5 / lr
    micro_bs = min(batch_size, max(16, int(micro_bs)))
    grad_accum = max(1, math.ceil(batch_size / micro_bs))

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * min(1.0, prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
    scaler = torch.amp.GradScaler('cuda', enabled=(amp_ok and amp_dtype == torch.float16))
    collator = _SSLCollator(yc, nb)
    n_dl_workers = min(2, max(0, (os.cpu_count() or 2) - 2))
    if n_dl_workers > 0:
        torch.multiprocessing.set_sharing_strategy('file_system')

    # Measure exact FLOPs per SSL vs Supervised step on a 16-event slice
    sample_b = move_batch_to_device(collator(val_calib_items[:16]), device)
    sample_b['label'] = torch.zeros((16,), dtype=torch.long, device=device)
    sample_b['label_7'] = torch.zeros((16,), dtype=torch.long, device=device)
    sample_b['charm_valid'] = torch.ones((16,), dtype=torch.long, device=device)
    sample_b['charm_flag'] = torch.zeros((16,), dtype=torch.long, device=device)
    sample_b['vertex_valid'] = torch.ones((16,), dtype=torch.long, device=device)
    sample_b['vertex_xyz_mm'] = torch.zeros((16, 3), dtype=torch.float32, device=device)
    sample_b['kin_valid'] = torch.ones((16,), dtype=torch.long, device=device)
    sample_b['vis_kinematics'] = torch.zeros((16, 4), dtype=torch.float32, device=device)
    flop_info = _measure_step_flops(student, teacher, sample_b, sample_b)
    print(f'[{ssl_variant}] Measured FLOPs/sample: SSL={flop_info["ssl_flops_per_sample"]/1e9:.3f}G, Sup={flop_info["sup_flops_per_sample"]/1e9:.3f}G, ratio={flop_info["ssl_to_sup_ratio"]:.3f}x', flush=True)

    torch.cuda.reset_peak_memory_stats(device)
    global_step = 0
    best_precal_auroc = -1.0
    history: list[dict[str, Any]] = []
    ckpt_path = out_dir / 'best_ssl.pt'
    last_ckpt_path = out_dir / 'last_ssl.pt'

    try:
        for ep in range(1, epochs + 1):
            te0 = time.perf_counter()
            student.train()
            _set_ieee_fp32(False)
            ep_sums: dict[str, float] = {}
            ep_n = 0

            if shard_groups is not None:
                ds_list = [InMemorySSLShardsDataset(shard_groups[0], bounds)]
            else:
                # Stream 40 shards at a time
                rng_sh = np.random.default_rng(seed + ep)
                perm_p = [local_tr_paths[i] for i in rng_sh.permutation(len(local_tr_paths))]
                ds_list = [perm_p[k:k + 40] for k in range(0, len(perm_p), 40)]

            for chunk_item in ds_list:
                if isinstance(chunk_item, InMemorySSLShardsDataset):
                    ds = chunk_item
                else:
                    shs = _load_ssl_shards_parallel(chunk_item, max_workers=8)
                    ds = InMemorySSLShardsDataset(shs, bounds)

                gen = torch.Generator()
                gen.manual_seed(seed + ep * 1009)
                sampler = BatchSampler(RandomSampler(ds, generator=gen), batch_size=micro_bs, drop_last=False)
                dl_kwargs: dict[str, Any] = {
                    'batch_sampler': sampler,
                    'num_workers': n_dl_workers,
                    'collate_fn': collator,
                    'pin_memory': True,
                }
                if n_dl_workers > 0:
                    dl_kwargs['prefetch_factor'] = 2
                loader = DataLoader(ds, **dl_kwargs)

                mask_gen = torch.Generator(device=device).manual_seed(seed + ep * 97 + global_step)
                acc_count = 0
                opt.zero_grad(set_to_none=True)
                for batch in loader:
                    db = move_batch_to_device(batch, device)
                    masks = sample_ssl_masks(
                        db,
                        generator=mask_gen,
                        shortcut_point_mask=bool(vcfg['shortcut_point_mask']),
                        n_st=student.backbone.n_st,
                    )
                    with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_ok):
                        loss, m_step = compute_ssl_pretrain_loss(
                            student,
                            teacher,
                            db,
                            masks,
                            w_jepa=float(vcfg['w_jepa']),
                            w_gmm=float(vcfg['w_gmm']),
                            w_stereo=float(vcfg['w_stereo']),
                            w_spec=float(vcfg['w_spec']),
                            w_vicreg=float(vcfg['w_vicreg']),
                        )
                        scaled_loss = loss / float(grad_accum)
                    if scaler.is_enabled():
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

                    bs_now = int(db['c_obs'].shape[0])
                    ep_n += bs_now
                    for mk, mv in m_step.items():
                        ep_sums[mk] = ep_sums.get(mk, 0.0) + mv * bs_now
                    acc_count += 1
                    if acc_count < grad_accum:
                        continue

                    global_step += 1
                    if scaler.is_enabled():
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(student.parameters(), 2.0)
                        scaler.step(opt)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(student.parameters(), 2.0)
                        opt.step()
                    opt.zero_grad(set_to_none=True)
                    scheduler.step()

                    prog = min(1.0, global_step / max(1, total_steps))
                    ema_m = 0.9995 - 0.5 * (0.9995 - 0.9960) * (1.0 + math.cos(math.pi * prog))
                    _update_ema(teacher, student.backbone, ema_m)
                    acc_count = 0

                if acc_count > 0:
                    global_step += 1
                    if scaler.is_enabled():
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(student.parameters(), 2.0)
                        scaler.step(opt)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(student.parameters(), 2.0)
                        opt.step()
                    opt.zero_grad(set_to_none=True)
                    scheduler.step()
                    prog = min(1.0, global_step / max(1, total_steps))
                    ema_m = 0.9995 - 0.5 * (0.9995 - 0.9960) * (1.0 + math.cos(math.pi * prog))
                    _update_ema(teacher, student.backbone, ema_m)

                if shard_groups is None:
                    del ds
                    gc.collect()

            ep_avg = {k: v / max(1, ep_n) for k, v in ep_sums.items()}
            ep_sec = float(time.perf_counter() - te0)
            row: dict[str, Any] = {'epoch': ep, 'step': global_step, 'ep_sec': ep_sec, **ep_avg}
            torch.save({
                'backbone': student.backbone.state_dict(),
                'teacher_backbone': teacher.state_dict(),
                'ssl_predictors': student.ssl_predictors.state_dict() if student.ssl_predictors is not None else None,
                'config': model_cfg,
                'epoch': ep,
                'ssl_variant': ssl_variant,
            }, last_ckpt_path)

            if ep % 2 == 0 or ep == epochs:
                precal = evaluate_unlabeled_muon_precalibration(student, val_calib_items, bounds, device)
                row['ssl_val_calib'] = precal
                score = float(precal.get('mu_presence_AUROC', 0.0)) + 0.1 * float(precal.get('ln_abs_sy_pearson_r', 0.0))
                if score > best_precal_auroc or not ckpt_path.is_file():
                    best_precal_auroc = score
                    torch.save({
                        'backbone': student.backbone.state_dict(),
                        'teacher_backbone': teacher.state_dict(),
                        'ssl_predictors': student.ssl_predictors.state_dict() if student.ssl_predictors is not None else None,
                        'config': model_cfg,
                        'epoch': ep,
                        'ssl_variant': ssl_variant,
                    }, ckpt_path)
                print(
                    f"[{ssl_variant}] ep={ep:02d}/{epochs} loss={ep_avg['ssl_total']:.4f} "
                    f"(jepa={ep_avg['ssl_jepa']:.3f} gmm={ep_avg['ssl_gmm']:.3f} st={ep_avg['ssl_stereo']:.3f} "
                    f"sp={ep_avg['ssl_spec']:.3f} vic_std={ep_avg['vic_glob_std']:.2f}) | "
                    f"mu_AUROC={precal.get('mu_presence_AUROC', float('nan')):.4f} "
                    f"r(ln|sy|)={precal.get('ln_abs_sy_pearson_r', float('nan')):.4f} ({ep_sec:.1f}s)",
                    flush=True,
                )
            history.append(row)

        torch.save({
            'backbone': student.backbone.state_dict(),
            'teacher_backbone': teacher.state_dict(),
            'ssl_predictors': student.ssl_predictors.state_dict() if student.ssl_predictors is not None else None,
            'config': model_cfg,
            'epoch': epochs,
            'ssl_variant': ssl_variant,
        }, last_ckpt_path)

        peak_vram = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        wall_sec = float(time.perf_counter() - t0)
        summary = {
            'ssl_variant': ssl_variant,
            'UTC_started': utc_started,
            'UTC_finished': datetime.now(timezone.utc).isoformat(),
            'wall_seconds': wall_sec,
            'gpu_name': torch.cuda.get_device_name(0),
            'peak_vram_MiB': peak_vram,
            'n_train_events': n_total_events,
            'epochs': epochs,
            'total_views_seen': n_total_events * epochs,
            'batch_size': batch_size,
            'online_finetune_params': student.online_finetune_param_count(),
            'flop_info': flop_info,
            'total_ssl_flops': flop_info['ssl_flops_per_sample'] * n_total_events * epochs,
            'history': history,
        }
        write_json(out_dir / 'ssl_pretrain_summary.json', summary)
        return summary
    finally:
        if shm_tr_dir is not None:
            shutil.rmtree(shm_tr_dir, ignore_errors=True)


def run_ssl_eval_precal(ssl_variant: str = 'ssl_full') -> dict[str, Any]:
    """Evaluate unlabeled penetrating-muon pre-calibration on `ssl_val_calib` and `internal_val_eval`."""
    assert_unsealed_split('internal_val', stage=1)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    geom = json.loads(assert_unsealed_path(CLEAN_ROOT / 'geometry.json').read_text())
    bounds = detector_bounds(geom)
    bounds['_plane_x0'] = np.array([cumulative_radiation_length(float(z), geom) for z in bounds['pixel_Zpos']], dtype=np.float32)
    yc, z_mid = float(bounds['y_scint_center']), float(bounds['z_mid'])
    xc = float(bounds['x_pix_center'])

    ckpt_p = EXP_ROOT / ssl_variant / 'last_ssl.pt'
    if not ckpt_p.is_file():
        ckpt_p = EXP_ROOT / ssl_variant / 'best_ssl.pt'
    ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
    model = PinpointSSLJEPA(geom, ckpt['config']).to(device).eval()
    model.backbone.load_state_dict(ckpt['backbone'], strict=True)
    if ckpt.get('ssl_predictors') is not None and model.ssl_predictors is not None:
        model.ssl_predictors.load_state_dict(ckpt['ssl_predictors'], strict=True)

    val_paths = sorted(assert_unsealed_path(p) for p in (SSL_DATA_ROOT / 'ssl_val_calib').glob('*.npz.zst'))
    val_shards = _load_ssl_shards_parallel(val_paths, max_workers=min(8, max(1, len(val_paths))))
    calib_items = [
        _csr_shard_event_to_item(sh, i, xc, yc, z_mid)
        for sh in val_shards
        for i in range(len(sh['obs_domain_id']))
    ]
    res_calib = evaluate_unlabeled_muon_precalibration(model, calib_items, bounds, device)

    # Also evaluate on internal_val_eval (pure observables only!)
    val_keys = pd.read_parquet(assert_unsealed_path(SPLITS_DIR / 'internal_val_eval_keys.parquet'))
    part_names = [f'train_part_{p}.pkl' for p in range(8)]
    all_items: list[dict] = []
    for fn in part_names:
        with (STAGE1_CACHE_DIR / fn).open('rb') as fh:
            all_items.extend(pickle.load(fh))
    val_items = _align_items_to_keys(all_items, val_keys)
    res_intval = evaluate_unlabeled_muon_precalibration(model, val_items, bounds, device)

    out = {
        'ssl_variant': ssl_variant,
        'ckpt': str(ckpt_p),
        'ssl_val_calib': res_calib,
        'internal_val_eval_observables': res_intval,
    }
    write_json(EXP_ROOT / ssl_variant / 'precalibration_eval.json', out)
    print(json.dumps(out, indent=2), flush=True)
    return out


# ---------------------------------------------------------------------------
# Supervised Dataset & Stage-I Pre-extraction / Training / Matrix
# ---------------------------------------------------------------------------
class InMemoryJEPADataset(Dataset):
    """In-memory supervised dataset with sidecar attachment & worker-invariant Y-mirror aug."""

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


class _JEPACollator:
    def __init__(self, y_scint_center: float, n_bars: int) -> None:
        self.y_scint_center = float(y_scint_center)
        self.n_bars = int(n_bars)

    def __call__(self, xs: list[dict]) -> dict[str, torch.Tensor]:
        return collate_dual_regime(xs, mirror_rng=None, y_scint_center=self.y_scint_center, n_bars=self.n_bars)


def _extract_jepa_shard_worker(task: tuple) -> list[dict]:
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
        it = extract_jepa_item(li, ob, int(row.label), dm, geom, bounds, cfg)
        it['run'] = int(row.run)
        it['chunk'] = int(row.chunk)
        it['row'] = int(row.row)
        it['family_hi'] = int(row.family_hi)
        it['family_lo'] = int(row.family_lo)
        items.append(it)
    return items


def _extract_jepa_obs_chunk_worker(task: tuple) -> list[dict]:
    raw_obs, labels, start_idx, geom, bounds, cfg = task
    items = []
    for k, ob in enumerate(raw_obs):
        lbl = None if labels is None else int(labels[k])
        dm = compute_observable_containment(ob, geom, bounds, config=cfg)
        items.append(extract_jepa_item(start_idx + k, ob, lbl, dm, geom, bounds, cfg))
    return items


def _parallel_extract_jepa_obs_list(
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
        parts = list(pool.map(_extract_jepa_obs_chunk_worker, tasks))
    else:
        parts = [_extract_jepa_obs_chunk_worker(t) for t in tasks]
    return [it for part in parts for it in part]


def run_stage1_preextract(part: int = -1) -> dict[str, Any]:
    """Sharded parallel Stage-I extraction (`part=0..7` train shards, `part=8` 6xOOD + 7 stress)."""
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
        with tempfile.TemporaryDirectory(prefix=f'jepa-pre-{part}-', dir=scratch_base) as tmp_dir:
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
                parts = list(pool.map(_extract_jepa_shard_worker, tasks))
            part_items = [it for p in parts for it in p]
        tmp_f = STAGE1_CACHE_DIR / f'train_part_{part}.pkl.tmp'
        with tmp_f.open('wb') as fh:
            pickle.dump(part_items, fh, protocol=5)
        tmp_f.replace(out_f)
        return {'part': part, 'n_items': len(part_items), 'seconds': float(time.perf_counter() - t0)}

    if part == 8:
        out_f = STAGE1_CACHE_DIR / 'ood_stress_pack.pkl'
        if out_f.is_file():
            return {'part': 8, 'cached': True}
        val_keys = pd.read_parquet(assert_unsealed_path(SPLITS_DIR / 'internal_val_eval_keys.parquet'))
        val_400 = val_keys.iloc[:400].reset_index(drop=True)

        by_shard: dict[int, list[tuple[int, int, int]]] = {}
        for idx_pos, row in enumerate(val_400.itertuples(index=False)):
            by_shard.setdefault(int(row.chunk), []).append((idx_pos, int(row.run), int(row.row)))
        raw_obs_400: list[dict | None] = [None] * len(val_400)
        y_400 = val_400['label'].to_numpy(dtype=np.int64)
        for chunk_id, entries in by_shard.items():
            shard_p = assert_unsealed_path(CLEAN_ROOT / 'data/shards' / f'{chunk_id:03d}')
            prov = pd.read_parquet(shard_p / 'provenance.parquet', columns=KEY)
            obs_tbl = pq.read_table(shard_p / 'observations.parquet').to_pylist()
            pos_map = {(int(r), int(c), int(rw)): i for i, (r, c, rw) in enumerate(zip(prov['run'], prov['chunk'], prov['row']))}
            for idx_pos, r, rw in entries:
                raw_obs_400[idx_pos] = obs_tbl[pos_map[(r, chunk_id, rw)]]

        with ProcessPoolExecutor(max_workers=n_cpus, mp_context=mp.get_context('fork')) as pool:
            ood_pre = {}
            for ov in ALL_STAGE2_OOD:
                raw_o, yo = _load_ood_observations(ov)
                ood_pre[ov] = (_parallel_extract_jepa_obs_list(raw_o, None, geom, bounds, BASE_OBS_CFG, pool=pool), yo)
            stress_items = {}
            for st_name, pkw, cfg_over in STRESS_SPECS:
                rng_st = np.random.default_rng(32510)
                cfg_use = {**BASE_OBS_CFG, **cfg_over}
                pert_obs = [
                    (_perturb_observation_for_stress(ob, bounds, rng_st, **pkw) if pkw else ob)
                    for ob in raw_obs_400
                    if ob is not None
                ]
                stress_items[st_name] = _parallel_extract_jepa_obs_list(pert_obs, y_400, geom, bounds, cfg_use, pool=pool)
        tmp_f = STAGE1_CACHE_DIR / 'ood_stress_pack.pkl.tmp'
        with tmp_f.open('wb') as fh:
            pickle.dump({'ood_pre': ood_pre, 'stress_items': stress_items, 'y_400': y_400}, fh, protocol=5)
        tmp_f.replace(out_f)
        return {'part': 8, 'seconds': float(time.perf_counter() - t0)}

    raise ValueError(f'Expected --part in 0..8, got {part}')


def _stratified_subset_indices(df: pd.DataFrame, frac: float, seed: int = 42) -> np.ndarray:
    """Deterministically sample `frac` of `internal_train` stratified by `(label, region_id, obs_domain_id)`."""
    if frac >= 0.999:
        return np.arange(len(df), dtype=np.int64)
    rng = np.random.default_rng(seed)
    target_n = int(round(len(df) * frac))
    strata = df.groupby(['label', 'region_id', 'obs_domain_id'], sort=True).indices
    picked: list[int] = []
    for _, idxs in sorted(strata.items()):
        arr = np.asarray(idxs, dtype=np.int64)
        k = max(1, int(round(len(arr) * frac)))
        k = min(k, len(arr))
        chosen = rng.choice(arr, size=k, replace=False)
        picked.extend(chosen.tolist())
    picked_arr = np.sort(np.asarray(picked, dtype=np.int64))
    if len(picked_arr) > target_n:
        drop = rng.choice(len(picked_arr), size=len(picked_arr) - target_n, replace=False)
        picked_arr = np.delete(picked_arr, drop)
    elif len(picked_arr) < target_n:
        avail = np.setdiff1d(np.arange(len(df), dtype=np.int64), picked_arr)
        add = rng.choice(avail, size=min(len(avail), target_n - len(picked_arr)), replace=False)
        picked_arr = np.sort(np.concatenate([picked_arr, add]))
    return picked_arr


@torch.no_grad()
def predict_jepa_items_fp32(
    model: PinpointSSLJEPA,
    items: list[dict],
    bounds: dict,
    device: torch.device,
    batch_size: int = 64,
    *,
    return_latents: bool = False,
) -> dict[str, np.ndarray]:
    """Deterministic IEEE-FP32 inference returning probs4, v_pv_mm, scale_pv_mm, charm_prob (+ latents)."""
    _set_ieee_fp32(True)
    model.eval()
    yc, nb = float(bounds['y_scint_center']), int(bounds['n_bars'])
    probs_l, v_l, s_l, c_l, l7_l, fg_l, qc_l, ztc_l = [], [], [], [], [], [], [], []
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
            fg_l.append(out.aux['fused_global'].float().cpu().numpy())
            qc_l.append(out.aux['q_bkg_charm'].float().cpu().numpy())
            ztc_l.append(out.aux['z_tau_c'].float().cpu().numpy())

    res: dict[str, np.ndarray] = {
        'probs': np.concatenate(probs_l, axis=0).astype(np.float32),
        'v_pv_mm': np.concatenate(v_l, axis=0).astype(np.float32),
        'scale_pv_mm': np.concatenate(s_l, axis=0).astype(np.float32),
        'charm_prob': np.concatenate(c_l, axis=0).astype(np.float32),
    }
    if return_latents:
        res['logits7'] = np.concatenate(l7_l, axis=0).astype(np.float32)
        res['fused_global'] = np.concatenate(fg_l, axis=0).astype(np.float32)
        res['q_bkg_charm'] = np.concatenate(qc_l, axis=0).astype(np.float32)
        res['z_tau_c'] = np.concatenate(ztc_l, axis=0).astype(np.float32)
        res['nccc_active'] = np.zeros((len(res['probs']),), dtype=bool)
        res['nccc_boost'] = np.zeros((len(res['probs']),), dtype=np.float32)
    return res


def _calibrate_jepa_pv_scales_on_train(
    ema_model: PinpointSSLJEPA,
    tr_items: list[dict],
    tr_sidecar: pd.DataFrame,
    bounds: dict,
    device: torch.device,
    max_events: int = 8192,
) -> np.ndarray:
    """Fit 3-axis Student-t(nu=3) p68(|pull|)=1.0 scale multiplier strictly on train."""
    _set_ieee_fp32(True)
    ema_model.eval()
    n_sub = min(max_events, len(tr_items))
    pred = predict_jepa_items_fp32(ema_model, tr_items[:n_sub], bounds, device, batch_size=64)
    sub_sc = tr_sidecar.iloc[:n_sub]
    v_true = np.column_stack([
        sub_sc['vertex_x_mm'].to_numpy(np.float64),
        sub_sc['vertex_y_mm'].to_numpy(np.float64),
        sub_sc['vertex_z_mm'].to_numpy(np.float64),
    ])
    v_valid = sub_sc['vertex_valid'].to_numpy(dtype=np.int64) > 0
    norm_pull = np.abs(pred['v_pv_mm'][v_valid].astype(np.float64) - v_true[v_valid]) / (
        np.maximum(pred['scale_pv_mm'][v_valid].astype(np.float64), 1e-4) * 1.1969
    )
    q68 = np.quantile(norm_pull, 0.6827, axis=0)
    mult = np.clip(q68, 0.65, 1.80).astype(np.float32)
    return mult


def run_stage1_train(variant: str) -> dict[str, Any]:
    if variant not in STAGE1_MATRIX:
        raise ValueError(f'Unknown variant {variant!r}; expected one of {sorted(STAGE1_MATRIX)}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    assert_unsealed_split('internal_train', stage=1)
    assert_unsealed_split('internal_screen', stage=1)
    assert_unsealed_split('internal_val', stage=1)

    t0 = time.perf_counter()
    utc_started = datetime.now(timezone.utc).isoformat()
    device = torch.device('cuda:0')
    seed = 42
    np.random.seed(seed)

    vcfg = dict(STAGE1_MATRIX[variant])
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
    tr_sidecar = sc_df.iloc[pos_sc].reset_index(drop=True)
    del sc_df

    part_names = [f'train_part_{p}.pkl' for p in range(8)] + ['ood_stress_pack.pkl']
    if not all((STAGE1_CACHE_DIR / fn).is_file() for fn in part_names):
        raise FileNotFoundError(f'Stage-I cache missing in {STAGE1_CACHE_DIR}; run stage1-preextract first')

    staged_dir, used_shm = _stage_pkl_dir_to_shm(STAGE1_CACHE_DIR, part_names, prefix='jepa_s1_shm')
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

    tr_items_full = _align_items_to_keys(all_items, tr_keys)
    dev_items = _align_items_to_keys(all_items, dev_keys)
    val_items = _align_items_to_keys(all_items, val_keys)
    del all_items
    gc.collect()

    sub_idx = _stratified_subset_indices(tr_keys, float(vcfg['train_frac']), seed=seed)
    tr_items = [tr_items_full[i] for i in sub_idx]
    tr_sidecar_sub = tr_sidecar.iloc[sub_idx].reset_index(drop=True)
    tr_keys_sub = tr_keys.iloc[sub_idx].reset_index(drop=True)
    del tr_items_full

    passes = int(vcfg['passes'])
    flops_parity: dict[str, float] | None = None
    if vcfg.get('match_equal_compute', False):
        ssl_sum_p = EXP_ROOT / 'ssl_full/ssl_pretrain_summary.json'
        if ssl_sum_p.is_file():
            ssl_sum = json.loads(ssl_sum_p.read_text())
            total_ssl_flops = float(ssl_sum['total_ssl_flops'])
            sup_flops_per_s = float(ssl_sum['flop_info']['sup_flops_per_sample'])
        else:
            # Measure exact F_SSL and F_sup directly via FlopCounterMode (assume conservative 325k x 15 SSL views)
            m_probe = PinpointSSLJEPA(geom, {'dim': 176, 'heads': 8, 'drop_path': 0.10, 'head_dropout': 0.15, 'include_ssl_heads': True}).to(device)
            t_probe = copy.deepcopy(m_probe.backbone).eval()
            for pp in t_probe.parameters():
                pp.requires_grad_(False)
            sb_probe = move_batch_to_device(collate_dual_regime(tr_items[:16], mirror_rng=None, y_scint_center=yc, n_bars=nb), device)
            sl_p = tr_sidecar_sub.iloc[:16]
            sb_probe['label'] = torch.tensor([int(x['label']) for x in tr_items[:16]], dtype=torch.long, device=device)
            sb_probe['label_7'] = torch.tensor(sl_p['label_7'].to_numpy(np.int64), dtype=torch.long, device=device)
            sb_probe['charm_valid'] = torch.tensor(sl_p['charm_valid'].to_numpy(np.int64), dtype=torch.long, device=device)
            sb_probe['charm_flag'] = torch.tensor(sl_p['charm_flag'].to_numpy(np.int64), dtype=torch.long, device=device)
            sb_probe['vertex_valid'] = torch.tensor(sl_p['vertex_valid'].to_numpy(np.int64), dtype=torch.long, device=device)
            sb_probe['vertex_xyz_mm'] = torch.tensor(np.column_stack([sl_p['vertex_x_mm'], sl_p['vertex_y_mm'], sl_p['vertex_z_mm']]), dtype=torch.float32, device=device)
            sb_probe['kin_valid'] = torch.tensor(sl_p['kin_valid'].to_numpy(np.int64), dtype=torch.long, device=device)
            sb_probe['vis_kinematics'] = torch.tensor(np.column_stack([sl_p['vis_pz_log1p'], sl_p['vis_pt_log1p'], sl_p['vis_cos_dphi'], sl_p['vis_sin_dphi']]), dtype=torch.float32, device=device)
            f_probe = _measure_step_flops(m_probe, t_probe, sb_probe, sb_probe)
            sup_flops_per_s = float(f_probe['sup_flops_per_sample'])
            total_ssl_flops = 330000.0 * 15.0 * float(f_probe['ssl_flops_per_sample'])
            del m_probe, t_probe, sb_probe
            torch.cuda.empty_cache()
        ft_flops = 25600.0 * 20.0 * sup_flops_per_s
        c_target = total_ssl_flops + ft_flops
        eq_samples = c_target / max(sup_flops_per_s, 1.0)
        passes = max(40, int(math.ceil(eq_samples / max(1, len(tr_items)))))
        c_sup_eq = float(passes * len(tr_items) * sup_flops_per_s)
        flops_parity = {
            'total_ssl_plus_ft_flops': c_target,
            'total_sup_eq_flops': c_sup_eq,
            'FLOPs_ratio': c_sup_eq / max(c_target, 1.0),
            'passes': int(passes),
            'sup_flops_per_sample': sup_flops_per_s,
            'assumed_ssl_events': (None if ssl_sum_p.is_file() else 330000),
        }
        print(f"[{variant}] Matched equal-compute passes = {passes} (FLOPs_ratio={flops_parity['FLOPs_ratio']:.4f})", flush=True)

    train_ds = InMemoryJEPADataset(
        tr_items,
        event_sidecar=tr_sidecar_sub,
        mirror_seed=(seed if vcfg['y_mirror'] else None),
        y_scint_center=yc,
        n_bars=nb,
    )

    cap = torch.cuda.get_device_capability(0)
    torch.manual_seed(seed)
    model_cfg = {
        'dim': 176,
        'heads': 8,
        'drop_path': float(vcfg['drop_path']),
        'head_dropout': float(vcfg['head_dropout']),
        'include_ssl_heads': False,
    }
    model = PinpointSSLJEPA(geom, model_cfg).to(device)

    ssl_var = vcfg.get('ssl_variant')
    if ssl_var is not None:
        ssl_ckpt_p = EXP_ROOT / ssl_var / 'last_ssl.pt'
        if not ssl_ckpt_p.is_file():
            ssl_ckpt_p = EXP_ROOT / ssl_var / 'best_ssl.pt'
        ssl_ckpt = torch.load(assert_unsealed_path(ssl_ckpt_p), map_location=device, weights_only=False)
        model.backbone.load_state_dict(ssl_ckpt['backbone'], strict=True)
        print(f'[{variant}] Loaded pretrained SSL backbone from {ssl_ckpt_p}', flush=True)

    ema_model = copy.deepcopy(model).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    n_params = model.online_finetune_param_count()

    base_lr = float(vcfg['lr'])
    min_lr = 1.0e-5
    wd = float(vcfg.get('weight_decay', 0.03))
    batch_size = int(vcfg.get('batch_size', 128))
    micro_bs = int(vcfg.get('micro_bs', 64))
    if torch.cuda.get_device_properties(0).total_memory < 16 * (1024 ** 3) and micro_bs > 32:
        micro_bs = 32
    grad_accum = max(1, batch_size // micro_bs)
    ema_decay_max = 0.9975

    opt = torch.optim.AdamW(
        model.build_param_groups(base_lr, wd, xi=float(vcfg['xi']), include_ssl_heads=False),
        lr=base_lr,
        weight_decay=wd,
        betas=(0.9, 0.98),
    )
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
    collator = _JEPACollator(yc, nb)

    torch.cuda.reset_peak_memory_stats(device)
    global_step = 0
    eval_every = 1 if passes <= 25 else max(2, passes // 20)

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
            fd = compute_jepa_finetune_window_denominators(win_buf)
            win_loss = 0.0
            win_n = 0
            for sub_mb in win_buf:
                db = move_batch_to_device(sub_mb, device)
                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_ok):
                    loss, _ = compute_jepa_finetune_loss(model, db, fixed_denoms=fd)
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
            fd = compute_jepa_finetune_window_denominators(win_buf)
            tail_loss = 0.0
            tail_n = 0
            for sub_mb in win_buf:
                db = move_batch_to_device(sub_mb, device)
                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_ok):
                    loss, _ = compute_jepa_finetune_loss(model, db, fixed_denoms=fd)
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

        if p_idx % eval_every == 0 or p_idx >= passes - 4 or p_idx == passes:
            dev_pred = predict_jepa_items_fp32(ema_model, dev_items, bounds, device, batch_size=64)
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
    pv_mult = _calibrate_jepa_pv_scales_on_train(ema_model, tr_items, tr_sidecar_sub, bounds, device)

    need_latents = (variant in ('ssl300k_jepa_ft_100pct', 'sup_scratch_equal_compute'))
    dev_pred = predict_jepa_items_fp32(ema_model, dev_items, bounds, device, batch_size=64, return_latents=need_latents)
    val_pred = predict_jepa_items_fp32(ema_model, val_items, bounds, device, batch_size=64, return_latents=need_latents)
    val_scale = (val_pred['scale_pv_mm'] * pv_mult[None, :]).astype(np.float32)

    np.save(out_dir / 'probs_dev_screen.npy', dev_pred['probs'])
    np.save(out_dir / 'probs_internal_val_eval.npy', val_pred['probs'])
    np.save(out_dir / 'v_pv_internal_val_eval.npy', val_pred['v_pv_mm'])
    np.save(out_dir / 'scale_pv_internal_val_eval.npy', val_scale)

    if need_latents:
        tr_pred = predict_jepa_items_fp32(ema_model, tr_items, bounds, device, batch_size=64, return_latents=True)
        _export_latent_cache(out_dir / 'latent_cache_internal_train.npz', tr_pred, tr_keys_sub, sidecar_df=tr_sidecar_sub)
        _export_latent_cache(out_dir / 'latent_cache_internal_dev_screen.npz', dev_pred, dev_keys)
        _export_latent_cache(out_dir / 'latent_cache_internal_val_eval.npz', val_pred, val_keys)

    # 6 OOD evaluation
    ood_probs: dict[str, np.ndarray] = {}
    ood_metrics: dict[str, Any] = {}
    for ov, (o_items, yo) in ood_pre.items():
        op = predict_jepa_items_fp32(ema_model, o_items, bounds, device, batch_size=64)['probs']
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

    # 7 Stress perturbations
    stress_probs: dict[str, np.ndarray] = {}
    stress_metrics: dict[str, Any] = {}
    p_nom = None
    ba4_nom = 0.0
    for st_name, _, _ in STRESS_SPECS:
        sp = predict_jepa_items_fp32(ema_model, stress_items[st_name], bounds, device, batch_size=64)['probs']
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

    b_chk = move_batch_to_device(collate_dual_regime(val_items[:8], mirror_rng=None, y_scint_center=yc, n_bars=nb), device)
    truth_inv = verify_inference_truth_invariance(ema_model, b_chk, atol=0.0)

    y_val = val_keys['label'].to_numpy(dtype=np.int64)
    val_m = _ba4_and_nll(y_val, val_pred['probs'])
    summary = {
        'variant': variant,
        'config': vcfg,
        'UTC_started': utc_started,
        'UTC_finished': datetime.now(timezone.utc).isoformat(),
        'wall_seconds': float(time.perf_counter() - t0),
        'gpu_name': torch.cuda.get_device_name(0),
        'n_params': n_params,
        'n_train_events': len(tr_items),
        'seed': seed,
        'passes': passes,
        'flops_parity': flops_parity,
        'total_steps': global_step,
        'peak_vram_MiB': peak_vram_mib,
        'best_pass': best_pass,
        'best_dev_screen_NLL': best_nll,
        'pv_scale_mult': pv_mult.tolist(),
        'truth_invariance_max_diff': truth_inv['max_abs_diff'],
        'internal_val_eval_all': val_m,
        'ood_metrics': ood_metrics,
        'stress_400_internal_val': stress_metrics,
        'history': history,
    }
    write_json(out_dir / 'run_summary.json', summary)
    print(
        f"[{variant}] DONE in {summary['wall_seconds']:.1f}s | passes={passes} | "
        f"val_BA4={val_m['BA4_pct']:.2f}% val_NLL={val_m['NLL']:.4f}",
        flush=True,
    )
    return summary


def run_stage1_aggregate() -> dict[str, Any]:
    """Aggregate Stage-I controlled benchmark matrix, Pull audit, equal-compute parity & 1,000x bootstrap."""
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
    v_true = sc_val[['vertex_x_mm', 'vertex_y_mm', 'vertex_z_mm']].to_numpy(np.float64)
    v_valid = sc_val['vertex_valid'].to_numpy(dtype=np.int64) > 0

    probs_map: dict[str, np.ndarray] = {}
    models_report: dict[str, Any] = {}
    reg_names = {0: 'FrontPinpoint_30mm_PixXY', 1: 'IntermPinpoint_150mm_PixXY', 2: 'Pinpoint_OuterW', 3: 'Fortune_ScintOnly'}

    for name in STAGE1_MATRIX:
        d_run = EXP_ROOT / name
        if not (d_run / 'probs_internal_val_eval.npy').is_file():
            continue
        probs = np.load(d_run / 'probs_internal_val_eval.npy').astype(np.float64)
        probs_map[name] = probs
        run_sum = json.loads((d_run / 'run_summary.json').read_text())

        dom_res = {
            OBS_DOMAIN_NAMES[did]: _ba4_and_nll(y_val[dom_val == did], probs[dom_val == did])
            for did in (1, 2, 3) if (dom_val == did).any()
        }
        reg_res = {
            rname: _ba4_and_nll(y_val[reg_val == rid], probs[reg_val == rid])
            for rid, rname in reg_names.items() if (reg_val == rid).any()
        }
        s_tau = np.log(np.clip(probs[:, 2], 1e-8, 1.0)) - np.log(np.clip(1.0 - probs[:, 2], 1e-8, 1.0))
        m_all = np.ones(len(y_val), dtype=bool)
        m_d1 = dom_val == 1
        m_pin = (reg_val == 0) | (reg_val == 1)
        tail = {
            'split_half_1pct_All': _split_half_tail_eval(s_tau, y_val, val_keys, m_all, 0.01),
            'split_half_1pct_D1_obs': _split_half_tail_eval(s_tau, y_val, val_keys, m_d1, 0.01),
            'within_slice_1pct_All': _within_slice_tail_eval(s_tau, y_val, m_all, 0.01),
            'within_slice_1pct_D1_obs': _within_slice_tail_eval(s_tau, y_val, m_d1, 0.01),
            'within_slice_0p1pct_All': _within_slice_tail_eval(s_tau, y_val, m_all, 0.001),
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
                pv_audit[tag] = {'n_events': int(mk.sum()), 'median_abs_dz_mm': float(np.median(dz)), 'median_3d_mm': float(np.median(r3)), **pulls}

        models_report[name] = {
            'n_params': run_sum.get('n_params'),
            'n_train_events': run_sum.get('n_train_events'),
            'passes': run_sum.get('passes'),
            'best_pass': run_sum.get('best_pass'),
            'wall_seconds': run_sum.get('wall_seconds'),
            'peak_vram_MiB': run_sum.get('peak_vram_MiB'),
            'internal_val_eval_all': _ba4_and_nll(y_val, probs),
            'by_obs_domain': dom_res,
            'by_truth_region': reg_res,
            'tau_tail': tail,
            'pv_audit': pv_audit,
            'ood_metrics': run_sum.get('ood_metrics'),
            'stress_400_internal_val': run_sum.get('stress_400_internal_val'),
        }

    paired_boot: dict[str, Any] = {}
    m_all = np.ones(len(y_val), dtype=bool)
    m_d1 = dom_val == 1
    m_pin = (reg_val == 0) | (reg_val == 1)
    pairs = [
        ('ssl300k_jepa_ft_100pct', 'sup_scratch_20p'),
        ('ssl300k_jepa_ft_100pct', 'sup_scratch_equal_compute'),
        ('ssl300k_jepa_ft_25pct', 'sup_scratch_25pct'),
        ('ssl300k_jepa_ft_10pct', 'sup_scratch_10pct'),
        ('ssl300k_jepa_ft_100pct', 'abl_ssl_no_spec_precal_ft_100pct'),
        ('ssl300k_jepa_ft_100pct', 'abl_ssl_random_point_mask_shortcut_ft_100pct'),
    ]
    for ma, mb in pairs:
        if ma in probs_map and mb in probs_map:
            for stag, mk in (('All', m_all), ('D1_obs', m_d1), ('Pinpoint', m_pin)):
                paired_boot[f'{ma} - {mb} ({stag})'] = _paired_family_bootstrap_full(
                    probs_map[ma], probs_map[mb], y_val, fam_codes, mk, n_boot=1000, seed=20263250
                )

    # Check Stage-I gates for promotion to Stage-II
    gates: dict[str, Any] = {}
    if 'ssl300k_jepa_ft_100pct' in models_report and 'sup_scratch_equal_compute' in models_report:
        rep_ssl = models_report['ssl300k_jepa_ft_100pct']
        rep_eq = models_report['sup_scratch_equal_compute']
        ba4_ssl = rep_ssl['internal_val_eval_all']['BA4_pct']
        nll_ssl = rep_ssl['internal_val_eval_all']['NLL']
        ba4_eq = rep_eq['internal_val_eval_all']['BA4_pct']
        nll_eq = rep_eq['internal_val_eval_all']['NLL']
        all_pulls = [
            float(x)
            for reg in ('All_valid', 'FrontPinpoint_30mm', 'IntermPinpoint_150mm')
            for x in rep_ssl['pv_audit'][reg]['p68_abs_pull']
        ]
        pulls_ok = all(0.85 <= p <= 1.15 for p in all_pulls)
        beats_eq = (ba4_ssl > ba4_eq) or (nll_ssl < nll_eq - 0.004)
        gates = {
            'ssl_BA4_pct': ba4_ssl,
            'ssl_NLL': nll_ssl,
            'eq_compute_BA4_pct': ba4_eq,
            'eq_compute_NLL': nll_eq,
            'delta_BA4_vs_eq_pp': ba4_ssl - ba4_eq,
            'delta_NLL_vs_eq': nll_ssl - nll_eq,
            'flops_parity': rep_eq.get('flops_parity'),
            'gate_beats_equal_compute': bool(beats_eq),
            'all_9_pulls': all_pulls,
            'gate_pulls_in_0p85_1p15': bool(pulls_ok),
            'promote_to_stage2': bool(beats_eq and pulls_ok),
        }

    ssl_precal = {}
    for sv in SSL_VARIANTS:
        p_pc = EXP_ROOT / sv / 'precalibration_eval.json'
        p_sm = EXP_ROOT / sv / 'ssl_pretrain_summary.json'
        if p_pc.is_file() or p_sm.is_file():
            ssl_precal[sv] = {
                'summary': json.loads(p_sm.read_text()) if p_sm.is_file() else None,
                'precalibration': json.loads(p_pc.read_text()) if p_pc.is_file() else None,
            }

    out_doc = {
        'UTC': datetime.now(timezone.utc).isoformat(),
        'internal_val_eval_events': int(len(y_val)),
        'ssl_pretraining': ssl_precal,
        'models': models_report,
        'stage1_gates': gates,
        'paired_family_bootstrap_1000_seed20263250': paired_boot,
    }
    write_json(EXP_ROOT / 'stage1_l2_jepa_results.json', out_doc)
    print(json.dumps({'stage1_gates': gates}, indent=2), flush=True)
    return out_doc


# ---------------------------------------------------------------------------
# Stage-II (Train32k x 3 seeds -> Tune6k) — ONLY run if Stage-I gates pass!
# ---------------------------------------------------------------------------
def run_stage2_preextract(part: int = -1) -> dict[str, Any]:
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
            return {'part': part, 'cached': True}
        all_tr = load_authorized_split_shards(CLEAN_ROOT / 'data', 'train', stage=2)
        sub_shards = all_tr[part * 10 : (part + 1) * 10]
        with tempfile.TemporaryDirectory(prefix=f'jepa-s2-tr-{part}-', dir=scratch_base) as tmp_dir:
            tmp_s = Path(tmp_dir) / 'shards'
            tmp_s.mkdir(parents=True, exist_ok=True)
            local_dirs = []
            for sp in sub_shards:
                dst_sp = tmp_s / sp.name
                shutil.copytree(sp, dst_sp)
                local_dirs.append(dst_sp)
            tasks = [(str(sd), geom, bounds, BASE_OBS_CFG, None, 'train') for sd in local_dirs]
            with ProcessPoolExecutor(max_workers=n_cpus, mp_context=mp.get_context('fork')) as pool:
                parts = list(pool.map(_extract_jepa_shard_worker, tasks))
            part_items = [it for p in parts for it in p]
        tmp_f = out_f.with_suffix('.pkl.tmp')
        with tmp_f.open('wb') as fh:
            pickle.dump(part_items, fh, protocol=5)
        tmp_f.replace(out_f)
        return {'part': part, 'n_items': len(part_items), 'seconds': float(time.perf_counter() - t0)}

    if 8 <= part <= 9:
        v_idx = part - 8
        out_f = STAGE2_CACHE_DIR / f'val_part_{v_idx}.pkl'
        if out_f.is_file():
            return {'part': part, 'cached': True}
        all_va = load_authorized_split_shards(CLEAN_ROOT / 'data', 'validation', stage=2)
        sub_shards = all_va[v_idx * 10 : (v_idx + 1) * 10]
        with tempfile.TemporaryDirectory(prefix=f'jepa-s2-va-{v_idx}-', dir=scratch_base) as tmp_dir:
            tmp_s = Path(tmp_dir) / 'shards'
            tmp_s.mkdir(parents=True, exist_ok=True)
            local_dirs = []
            for sp in sub_shards:
                dst_sp = tmp_s / sp.name
                shutil.copytree(sp, dst_sp)
                local_dirs.append(dst_sp)
            tasks = [(str(sd), geom, bounds, BASE_OBS_CFG, None, 'validation') for sd in local_dirs]
            with ProcessPoolExecutor(max_workers=n_cpus, mp_context=mp.get_context('fork')) as pool:
                parts = list(pool.map(_extract_jepa_shard_worker, tasks))
            part_items = [it for p in parts for it in p]
        tmp_f = out_f.with_suffix('.pkl.tmp')
        with tmp_f.open('wb') as fh:
            pickle.dump(part_items, fh, protocol=5)
        tmp_f.replace(out_f)
        return {'part': part, 'n_items': len(part_items), 'seconds': float(time.perf_counter() - t0)}

    if part == 10:
        out_f = STAGE2_CACHE_DIR / 'ood_stress_stage2_pack.pkl'
        if out_f.is_file():
            return {'part': 10, 'cached': True}
        shard080 = assert_unsealed_path(CLEAN_ROOT / 'data/shards/080')
        obs_080 = pq.read_table(shard080 / 'observations.parquet').to_pylist()
        prov_080 = pd.read_parquet(shard080 / 'provenance.parquet', columns=KEY + FAMILY + ['split', 'label'])
        y_080 = prov_080['label'].to_numpy(dtype=np.int64)
        with ProcessPoolExecutor(max_workers=n_cpus, mp_context=mp.get_context('fork')) as pool:
            ood_pre = {}
            for ov in ALL_STAGE2_OOD:
                raw_o, yo = _load_ood_observations(ov)
                ood_pre[ov] = (_parallel_extract_jepa_obs_list(raw_o, None, geom, bounds, BASE_OBS_CFG, pool=pool), yo)
            stress_items = {}
            for st_name, pkw, cfg_over in STRESS_SPECS:
                rng_st = np.random.default_rng(32510)
                cfg_use = {**BASE_OBS_CFG, **cfg_over}
                pert_obs = [(_perturb_observation_for_stress(ob, bounds, rng_st, **pkw) if pkw else ob) for ob in obs_080]
                stress_items[st_name] = _parallel_extract_jepa_obs_list(pert_obs, None, geom, bounds, cfg_use, pool=pool)
        tmp_f = out_f.with_suffix('.pkl.tmp')
        with tmp_f.open('wb') as fh:
            pickle.dump({'ood_pre': ood_pre, 'stress_items': stress_items, 'y_080': y_080, 'prov_080': prov_080}, fh, protocol=5)
        tmp_f.replace(out_f)
        return {'part': 10, 'seconds': float(time.perf_counter() - t0)}

    raise ValueError(f'Expected --part in 0..10, got {part}')


def run_stage2_train(seed: int) -> dict[str, Any]:
    s1_res = EXP_ROOT / 'stage1_l2_jepa_results.json'
    if not s1_res.is_file():
        raise FileNotFoundError('Stage-I results missing; must pass Stage-I gates before Stage-II')
    s1_doc = json.loads(s1_res.read_text())
    if not s1_doc.get('stage1_gates', {}).get('promote_to_stage2', False):
        raise RuntimeError('Stage-I gates did NOT pass; Stage-II is blocked by protocol!')
    if seed not in STAGE2_SEEDS:
        raise ValueError(f'Expected seed in {STAGE2_SEEDS}, got {seed}')

    for sp in ('train', 'validation', 'tune_screen', 'tune6k', 'ood'):
        assert_unsealed_split(sp, stage=2)

    t0 = time.perf_counter()
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
    staged_dir, used_shm = _stage_pkl_dir_to_shm(STAGE2_CACHE_DIR, all_names, prefix=f'jepa_s2_s{seed}')
    try:
        tr_items_raw, va_items_raw = [], []
        for fn in tr_part_names:
            with (staged_dir / fn).open('rb') as fh:
                tr_items_raw.extend(pickle.load(fh))
        for fn in va_part_names:
            with (staged_dir / fn).open('rb') as fh:
                va_items_raw.extend(pickle.load(fh))
        with (staged_dir / 'ood_stress_stage2_pack.pkl').open('rb') as fh:
            os_pack = pickle.load(fh)
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

    train_ds = InMemoryJEPADataset(tr_items_raw, event_sidecar=tr_sidecar, mirror_seed=seed, y_scint_center=yc, n_bars=nb)
    cap = torch.cuda.get_device_capability(0)
    torch.manual_seed(seed)
    model_cfg = {'dim': 176, 'heads': 8, 'drop_path': 0.10, 'head_dropout': 0.15, 'include_ssl_heads': False}
    model = PinpointSSLJEPA(geom, model_cfg).to(device)
    ssl_ckpt_p = EXP_ROOT / 'ssl_full/last_ssl.pt'
    if not ssl_ckpt_p.is_file():
        ssl_ckpt_p = EXP_ROOT / 'ssl_full/best_ssl.pt'
    ssl_ckpt = torch.load(assert_unsealed_path(ssl_ckpt_p), map_location=device, weights_only=False)
    model.backbone.load_state_dict(ssl_ckpt['backbone'], strict=True)

    ema_model = copy.deepcopy(model).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    base_lr, min_lr, wd = 6.0e-4, 1.0e-5, 0.03
    passes, batch_size = 20, 128
    micro_bs = 32 if torch.cuda.get_device_properties(0).total_memory < 16 * (1024 ** 3) else 64
    grad_accum = max(1, batch_size // micro_bs)
    opt = torch.optim.AdamW(model.build_param_groups(base_lr, wd, xi=0.78, include_ssl_heads=False), lr=base_lr, weight_decay=wd, betas=(0.9, 0.98))
    total_steps = passes * math.ceil(len(train_ds) / batch_size)
    warmup_steps = max(1, int(0.06 * total_steps))
    min_lr_ratio = min_lr / base_lr
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (float(s + 1) / warmup_steps) if s < warmup_steps else min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * min(1.0, (s - warmup_steps) / max(1, total_steps - warmup_steps)))),
    )
    amp_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
    amp_ok = cap[0] >= 7
    scaler = torch.amp.GradScaler('cuda', enabled=(amp_ok and amp_dtype == torch.float16))
    collator = _JEPACollator(yc, nb)
    n_dl_workers = min(2, max(0, (os.cpu_count() or 2) - 1))

    best_nll, best_pass = float('inf'), 0
    best_ema_path = out_dir / 'best_ema.pt'
    global_step = 0
    for p_idx in range(1, passes + 1):
        train_ds.set_pass(p_idx)
        gen = torch.Generator().manual_seed(seed + p_idx * 101)
        loader = DataLoader(train_ds, batch_sampler=BatchSampler(RandomSampler(train_ds, generator=gen), batch_size=micro_bs, drop_last=False), num_workers=n_dl_workers, collate_fn=collator, pin_memory=True)
        model.train()
        _set_ieee_fp32(False)
        win_buf: list[dict] = []
        for mb in loader:
            win_buf.append(mb)
            if len(win_buf) < grad_accum:
                continue
            global_step += 1
            opt.zero_grad(set_to_none=True)
            fd = compute_jepa_finetune_window_denominators(win_buf)
            for sub_mb in win_buf:
                db = move_batch_to_device(sub_mb, device)
                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_ok):
                    loss, _ = compute_jepa_finetune_loss(model, db, fixed_denoms=fd)
                (scaler.scale(loss) if scaler.is_enabled() else loss).backward()
            if scaler.is_enabled():
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(opt)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
            scheduler.step()
            _update_ema(ema_model, model, min(0.9975, (1.0 + global_step) / (10.0 + global_step)))
            win_buf.clear()

        if win_buf:
            global_step += 1
            opt.zero_grad(set_to_none=True)
            fd = compute_jepa_finetune_window_denominators(win_buf)
            for sub_mb in win_buf:
                db = move_batch_to_device(sub_mb, device)
                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_ok):
                    loss, _ = compute_jepa_finetune_loss(model, db, fixed_denoms=fd)
                (scaler.scale(loss) if scaler.is_enabled() else loss).backward()
            if scaler.is_enabled():
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(opt)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
            scheduler.step()
            _update_ema(ema_model, model, min(0.9975, (1.0 + global_step) / (10.0 + global_step)))

        if p_idx >= 10 or p_idx % 2 == 0 or p_idx == passes:
            scr_m = _ba4_and_nll(y_screen, predict_jepa_items_fp32(ema_model, screen_items, bounds, device, 64)['probs'])
            if scr_m['NLL'] < best_nll:
                best_nll, best_pass = scr_m['NLL'], p_idx
                torch.save({'model': ema_model.state_dict(), 'config': model_cfg, 'pass': p_idx, 'seed': seed}, best_ema_path)

    _set_ieee_fp32(True)
    ckpt = torch.load(best_ema_path, map_location=device, weights_only=False)
    ema_model.load_state_dict(ckpt['model'])
    ema_model.eval()
    pv_mult = _calibrate_jepa_pv_scales_on_train(ema_model, tr_items_raw, tr_sidecar, bounds, device)

    tune8k_pred = predict_jepa_items_fp32(ema_model, tune8k_items, bounds, device, 64, return_latents=True)
    tr_pred = predict_jepa_items_fp32(ema_model, tr_items_raw, bounds, device, 64, return_latents=True)
    _export_latent_cache(out_dir / 'latent_cache_train32k.npz', tr_pred, tr_keys, sidecar_df=tr_sidecar)
    _export_latent_cache(out_dir / 'latent_cache_tune8k.npz', tune8k_pred, tune_meta)

    p_tune8k = tune8k_pred['probs']
    s_tune8k = (tune8k_pred['scale_pv_mm'] * pv_mult[None, :]).astype(np.float32)
    np.save(out_dir / 'probs_tune8k.npy', p_tune8k)
    np.save(out_dir / 'probs_tune6k.npy', p_tune8k[tune6k_m])
    np.save(out_dir / 'v_pv_tune8k.npy', tune8k_pred['v_pv_mm'])
    np.save(out_dir / 'scale_pv_tune8k.npy', s_tune8k)

    ood_probs, stress_probs = {}, {}
    for ov, (o_items, _) in os_pack['ood_pre'].items():
        ood_probs[f'ood_{ov}'] = predict_jepa_items_fp32(ema_model, o_items, bounds, device, 64)['probs']
    for st_name, _, _ in STRESS_SPECS:
        stress_probs[f'stress_{st_name}'] = predict_jepa_items_fp32(ema_model, os_pack['stress_items'][st_name], bounds, device, 64)['probs']
    np.savez_compressed(out_dir / 'ood_and_stress_probs.npz', **ood_probs, **stress_probs)

    y_t8k = tune_meta['label'].to_numpy(dtype=np.int64)
    t6k_m = _ba4_and_nll(y_t8k[tune6k_m], p_tune8k[tune6k_m])
    summary = {
        'variant': 's2_l2_pinpoint_ssl_jepa',
        'seed': int(seed),
        'wall_seconds': float(time.perf_counter() - t0),
        'best_pass': best_pass,
        'best_tune_screen_NLL': best_nll,
        'pv_scale_mult': pv_mult.tolist(),
        'tune6k_all': t6k_m,
    }
    write_json(out_dir / 'run_summary.json', summary)
    return summary


def run_stage2_dro() -> dict[str, Any]:
    """Run Score-Bank KL-DRO fine-tuning on frozen Stage-II `latent_cache_*` (only if Stage-I passed)."""
    from analysis.flavour.nextgen.tail_dro_pauc import (
        TailDROConfig,
        finetune_head_on_cached_latents,
        marginal_four_from_seven,
        predict_from_cached_latents,
    )

    s1_res = EXP_ROOT / 'stage1_l2_jepa_results.json'
    if not s1_res.is_file() or not json.loads(s1_res.read_text()).get('stage1_gates', {}).get('promote_to_stage2', False):
        raise RuntimeError('Stage-I gates did NOT pass; Stage-II DRO is blocked by protocol!')
    for sp in ('train', 'validation', 'tune_screen', 'tune6k', 'ood'):
        assert_unsealed_split(sp, stage=2)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tune_meta = pd.read_parquet(assert_unsealed_path(CLEAN_ROOT / 'validation_meta/tune_truth_meta.parquet'))
    scr_df = pd.read_parquet(assert_unsealed_path(CLEAN_ROOT / 'data/tune_screen.parquet'), columns=KEY)
    in_screen = np.asarray(pd.MultiIndex.from_frame(tune_meta[KEY]).isin(pd.MultiIndex.from_frame(scr_df[KEY])), dtype=bool)
    tune6k_m = ~in_screen
    y_t8k = tune_meta['label'].to_numpy(dtype=np.int64)
    geom = json.loads(assert_unsealed_path(CLEAN_ROOT / 'geometry.json').read_text())

    dro_cands = [
        ('sopa_l005_kl20_bank8192', TailDROConfig(
            dro_mode='sopa_s_hybrid', margin=0.35, lambda_dro=0.05, lambda_nll=0.5, lambda_kl=2.0,
            use_bank=True, bank_capacity=8192, lr=5.0e-5, passes=3, batch_size=128, seed=42,
        )),
        ('dro_bal_l008_kl15_bank8192', TailDROConfig(
            dro_mode='cvar_balanced', margin=0.5, surrogate='softplus', lambda_dro=0.08, lambda_nll=0.5, lambda_kl=1.5,
            use_bank=True, bank_capacity=8192, lr=5.0e-5, passes=3, batch_size=128, seed=42,
        )),
    ]

    cand_reports: dict[str, Any] = {}
    best_cname, best_scr_pauc, best_probs_t8k = '', -1.0, None
    for cname, dcfg in dro_cands:
        seed_probs_t8k: list[np.ndarray] = []
        for s in STAGE2_SEEDS:
            sdir = STAGE2_DIR / f'seed_{s}'
            tr_npz = np.load(sdir / 'latent_cache_train32k.npz')
            t8_npz = np.load(sdir / 'latent_cache_tune8k.npz')
            ckpt = torch.load(sdir / 'best_ema.pt', map_location=device, weights_only=False)
            model = PinpointSSLJEPA(geom, ckpt['config']).to(device)
            model.load_state_dict(ckpt['model'])
            head = copy.deepcopy(model.event_head_7).to(device)
            tr_cache = {
                'penultimate': torch.from_numpy(tr_npz['fused_global']),
                'logits7_ref': torch.from_numpy(tr_npz['logits7']),
                'label': torch.from_numpy(tr_npz['label']),
                'label_7': torch.from_numpy(tr_npz['label_7']),
            }
            cfg_s = copy.deepcopy(dcfg)
            cfg_s.seed = int(s)
            head_ft, _ = finetune_head_on_cached_latents(head, tr_cache, cfg_s, device)
            pred_t8 = predict_from_cached_latents(head_ft, torch.from_numpy(t8_npz['fused_global']), device)
            p_t8 = pred_t8['probs4'].astype(np.float64)
            np.save(sdir / f'probs_tune8k_dro_{cname}.npy', p_t8.astype(np.float32))
            seed_probs_t8k.append(p_t8)
        p_ens_t8 = np.mean(seed_probs_t8k, axis=0)
        s_tau = np.log(np.clip(p_ens_t8[:, 2], 1e-8, 1.0)) - np.log(np.clip(1.0 - p_ens_t8[:, 2], 1e-8, 1.0))
        scr_tail = _within_slice_tail_eval(s_tau, y_t8k, in_screen, 0.01)
        scr_score = float(scr_tail.get('tau_eff_pct', 0.0))
        t6_tail = _within_slice_tail_eval(s_tau, y_t8k, tune6k_m, 0.01)
        t6_tail_01 = _within_slice_tail_eval(s_tau, y_t8k, tune6k_m, 0.001)
        cand_reports[cname] = {
            'tune_screen_all': _ba4_and_nll(y_t8k[in_screen], p_ens_t8[in_screen]),
            'tune_screen_tau1pct': scr_tail,
            'tune6k_all': _ba4_and_nll(y_t8k[tune6k_m], p_ens_t8[tune6k_m]),
            'tune6k_tau1pct': t6_tail,
            'tune6k_tau0p1pct': t6_tail_01,
        }
        if scr_score > best_scr_pauc:
            best_scr_pauc, best_cname, best_probs_t8k = scr_score, cname, p_ens_t8

    assert best_probs_t8k is not None
    np.save(STAGE2_DIR / 'probs_tune8k_ens3_dro.npy', best_probs_t8k.astype(np.float32))
    np.save(STAGE2_DIR / 'probs_tune6k_ens3_dro.npy', best_probs_t8k[tune6k_m].astype(np.float32))
    out_doc = {
        'UTC': datetime.now(timezone.utc).isoformat(),
        'selected_candidate_by_tune_screen': best_cname,
        'candidates': cand_reports,
    }
    write_json(STAGE2_DIR / 'stage2_dro_results.json', out_doc)
    return out_doc


def run_stage2_aggregate() -> dict[str, Any]:
    for sp in ('train', 'validation', 'tune_screen', 'tune6k', 'ood'):
        assert_unsealed_split(sp, stage=2)
    tune_meta = pd.read_parquet(assert_unsealed_path(CLEAN_ROOT / 'validation_meta/tune_truth_meta.parquet'))
    scr_df = pd.read_parquet(assert_unsealed_path(CLEAN_ROOT / 'data/tune_screen.parquet'), columns=KEY)
    in_screen = np.asarray(pd.MultiIndex.from_frame(tune_meta[KEY]).isin(pd.MultiIndex.from_frame(scr_df[KEY])), dtype=bool)
    tune6k_m = ~in_screen
    t6_keys = tune_meta.iloc[tune6k_m].reset_index(drop=True)
    labels = tune_meta['label'].to_numpy(dtype=np.int64)
    y_t6 = labels[tune6k_m]
    obs_dom = tune_meta['obs_domain_id'].to_numpy(dtype=np.int64)[tune6k_m]
    if 'region_id' in tune_meta.columns:
        reg_all = tune_meta['region_id'].to_numpy(dtype=np.int64)
    else:
        pc_m = tune_meta['pixel_centered'].to_numpy(dtype=bool)
        fp_m = tune_meta['front_pinpoint_pixel_centered'].to_numpy(dtype=bool)
        fort_m = tune_meta['fortune_module'].to_numpy(dtype=bool)
        reg_all = np.full(len(tune_meta), 2, dtype=np.int64)
        reg_all[pc_m & (~fp_m)] = 1
        reg_all[fp_m] = 0
        reg_all[fort_m] = 3
    reg_t6 = reg_all[tune6k_m]
    fam_t6 = pd.MultiIndex.from_frame(t6_keys[FAMILY]).factorize()[0]

    sc_df = pd.read_parquet(assert_unsealed_path(CLEAN_ROOT / 'truth_sidecar/event_sidecar.parquet'))
    pos_sc = pd.MultiIndex.from_frame(sc_df[KEY]).get_indexer(pd.MultiIndex.from_frame(t6_keys[KEY]))
    sc_t6 = sc_df.iloc[pos_sc].reset_index(drop=True)
    v_true = sc_t6[['vertex_x_mm', 'vertex_y_mm', 'vertex_z_mm']].to_numpy(np.float64)
    v_valid = sc_t6['vertex_valid'].to_numpy(dtype=np.int64) > 0

    seed_p = {s: np.load(STAGE2_DIR / f'seed_{s}/probs_tune8k.npy').astype(np.float64) for s in STAGE2_SEEDS}
    p_ens_raw = np.mean([seed_p[s] for s in STAGE2_SEEDS], axis=0)
    np.save(STAGE2_DIR / 'probs_tune8k_ens3_raw.npy', p_ens_raw.astype(np.float32))
    p_t6 = p_ens_raw[tune6k_m]
    np.save(STAGE2_DIR / 'probs_tune6k_ens3_raw.npy', p_t6.astype(np.float32))

    # Domain, truth-region, and tau-tail breakdown on Tune6k
    reg_names = {0: 'FrontPinpoint_30mm_PixXY', 1: 'IntermPinpoint_150mm_PixXY', 2: 'Pinpoint_OuterW', 3: 'Fortune_ScintOnly'}
    dom_res = {OBS_DOMAIN_NAMES[did]: _ba4_and_nll(y_t6[obs_dom == did], p_t6[obs_dom == did]) for did in (1, 2, 3) if (obs_dom == did).any()}
    reg_res = {rname: _ba4_and_nll(y_t6[reg_t6 == rid], p_t6[reg_t6 == rid]) for rid, rname in reg_names.items() if (reg_t6 == rid).any()}
    s_tau = np.log(np.clip(p_t6[:, 2], 1e-8, 1.0)) - np.log(np.clip(1.0 - p_t6[:, 2], 1e-8, 1.0))
    m_all = np.ones(len(y_t6), dtype=bool)
    m_d1 = obs_dom == 1
    m_pin = (reg_t6 == 0) | (reg_t6 == 1)
    tau_tail = {
        'split_half_1pct_All': _split_half_tail_eval(s_tau, y_t6, t6_keys, m_all, 0.01),
        'split_half_1pct_D1_obs': _split_half_tail_eval(s_tau, y_t6, t6_keys, m_d1, 0.01),
        'within_slice_1pct_All': _within_slice_tail_eval(s_tau, y_t6, m_all, 0.01),
        'within_slice_1pct_D1_obs': _within_slice_tail_eval(s_tau, y_t6, m_d1, 0.01),
        'within_slice_0p1pct_All': _within_slice_tail_eval(s_tau, y_t6, m_all, 0.001),
    }

    # Ensemble PV Pull audit on Tune6k (raw + calibrated strictly on tune_screen N=2,000 per Stage-II protocol)
    v_ens_t8 = np.mean([np.load(STAGE2_DIR / f'seed_{s}/v_pv_tune8k.npy').astype(np.float64) for s in STAGE2_SEEDS], axis=0)
    s_ens_t8 = np.mean([np.load(STAGE2_DIR / f'seed_{s}/scale_pv_tune8k.npy').astype(np.float64) for s in STAGE2_SEEDS], axis=0)

    pos_t8_sc = pd.MultiIndex.from_frame(sc_df[KEY]).get_indexer(pd.MultiIndex.from_frame(tune_meta[KEY]))
    sc_t8 = sc_df.iloc[pos_t8_sc].reset_index(drop=True)
    v_true_t8 = sc_t8[['vertex_x_mm', 'vertex_y_mm', 'vertex_z_mm']].to_numpy(np.float64)
    v_valid_t8 = sc_t8['vertex_valid'].to_numpy(dtype=np.int64) > 0
    m_scr_pv = in_screen & v_valid_t8
    q68_t3 = float(scipy_t.ppf(0.84, df=3.0))
    ens_mult = np.ones((3,), dtype=np.float64)
    if m_scr_pv.any():
        for ax in range(3):
            z_scr = np.abs(v_ens_t8[m_scr_pv, ax] - v_true_t8[m_scr_pv, ax]) / np.maximum(s_ens_t8[m_scr_pv, ax], 1e-4)
            ens_mult[ax] = float(np.clip(np.quantile(z_scr, 0.68) / q68_t3, 0.65, 1.50))
    s_ens_t8_cal = (s_ens_t8 * ens_mult[None, :]).astype(np.float32)
    np.save(STAGE2_DIR / 'scale_pv_tune8k_ens3_cal.npy', s_ens_t8_cal)

    v_ens = v_ens_t8[tune6k_m]
    s_ens_uncal = s_ens_t8[tune6k_m]
    s_ens = s_ens_t8_cal[tune6k_m].astype(np.float64)
    pv_audit, pv_audit_uncal = {}, {}
    for tag, mk in (('All_valid', v_valid), ('FrontPinpoint_30mm', v_valid & (reg_t6 == 0)), ('IntermPinpoint_150mm', v_valid & (reg_t6 == 1))):
        if mk.any():
            dz = np.abs(v_ens[mk, 2] - v_true[mk, 2])
            r3 = np.linalg.norm(v_ens[mk] - v_true[mk], axis=-1)
            pulls = compute_student_t_pulls(torch.from_numpy(v_ens), torch.from_numpy(s_ens), torch.from_numpy(v_true), torch.from_numpy(mk.astype(np.int64)), df=3.0)
            pulls_u = compute_student_t_pulls(torch.from_numpy(v_ens), torch.from_numpy(s_ens_uncal), torch.from_numpy(v_true), torch.from_numpy(mk.astype(np.int64)), df=3.0)
            pv_audit[tag] = {'n_events': int(mk.sum()), 'median_abs_dz_mm': float(np.median(dz)), 'median_3d_mm': float(np.median(r3)), **pulls}
            pv_audit_uncal[tag] = {'n_events': int(mk.sum()), 'median_abs_dz_mm': float(np.median(dz)), 'median_3d_mm': float(np.median(r3)), **pulls_u}

    # Ensemble OOD & Shard-080 Stress (averaging per-seed outputs)
    ood_ens3, stress_ens3 = {}, {}
    npz_paths = [STAGE2_DIR / f'seed_{s}/ood_and_stress_probs.npz' for s in STAGE2_SEEDS]
    if all(p.is_file() for p in npz_paths):
        npzs = [np.load(p) for p in npz_paths]
        for ov in ALL_STAGE2_OOD:
            k = f'ood_{ov}'
            if all(k in z for z in npzs):
                op = np.mean([z[k].astype(np.float64) for z in npzs], axis=0)
                _, yo = _load_ood_observations(ov)
                if 'nc_' in ov:
                    ood_ens3[ov] = {
                        'n_events': int(len(yo)),
                        'mean_p_tau_pct': float(op[:, 2].mean() * 100.0),
                        'frac_pred_tau_pct': float((op.argmax(axis=1) == 2).mean() * 100.0),
                    }
                else:
                    ood_ens3[ov] = _ba4_and_nll(yo, op)
        shard080 = assert_unsealed_path(CLEAN_ROOT / 'data/shards/080')
        y_080 = pd.read_parquet(shard080 / 'provenance.parquet', columns=['label'])['label'].to_numpy(dtype=np.int64)
        for st_name, _, _ in STRESS_SPECS:
            k = f'stress_{st_name}'
            if all(k in z for z in npzs):
                sp = np.mean([z[k].astype(np.float64) for z in npzs], axis=0)
                stress_ens3[st_name] = _ba4_and_nll(y_080, sp)

    # Synergy blends & 1,000x paired family bootstrap (seed=20263250) on Tune6k
    blends: dict[str, Any] = {}
    paired_boot: dict[str, Any] = {}
    dr_p = STAGE1_S326_DUAL_DIR / 'stage2/probs_tune8k_ens3_raw.npy'
    l1_p = L1_EXP_DIR / 'stage2/probs_tune8k_gn2_ens3_raw.npy'
    if not l1_p.is_file():
        l1_p = L1_EXP_DIR / 'stage2/probs_tune8k_ens3_raw.npy'
    p_dr_t6 = np.load(dr_p).astype(np.float64)[tune6k_m] if dr_p.is_file() else None
    p_l1_t6 = np.load(l1_p).astype(np.float64)[tune6k_m] if l1_p.is_file() else None
    if p_dr_t6 is not None:
        b_jd = 0.5 * (p_t6 + p_dr_t6)
        blends['blend_50_50_jepa_dual'] = _ba4_and_nll(y_t6, b_jd)
        for stag, mk in (('All', m_all), ('D1_obs', m_d1), ('Pinpoint', m_pin)):
            paired_boot[f'PinpointSSL_JEPA_ens3 - DualRegime_ens3 ({stag})'] = _paired_family_bootstrap_full(p_t6, p_dr_t6, y_t6, fam_t6, mk, n_boot=1000, seed=20263250)
            paired_boot[f'blend_50_50_jepa_dual - DualRegime_ens3 ({stag})'] = _paired_family_bootstrap_full(b_jd, p_dr_t6, y_t6, fam_t6, mk, n_boot=1000, seed=20263250)
    if p_l1_t6 is not None:
        b_jl = 0.5 * (p_t6 + p_l1_t6)
        blends['blend_50_50_jepa_l1'] = _ba4_and_nll(y_t6, b_jl)
        for stag, mk in (('All', m_all), ('D1_obs', m_d1), ('Pinpoint', m_pin)):
            paired_boot[f'PinpointSSL_JEPA_ens3 - L1_GN2_ens3 ({stag})'] = _paired_family_bootstrap_full(p_t6, p_l1_t6, y_t6, fam_t6, mk, n_boot=1000, seed=20263250)
    if p_dr_t6 is not None and p_l1_t6 is not None:
        b_3w = (p_t6 + p_dr_t6 + p_l1_t6) / 3.0
        blends['blend_3way_l1_dual_jepa'] = _ba4_and_nll(y_t6, b_3w)
        for stag, mk in (('All', m_all), ('D1_obs', m_d1), ('Pinpoint', m_pin)):
            paired_boot[f'blend_3way - DualRegime_ens3 ({stag})'] = _paired_family_bootstrap_full(b_3w, p_dr_t6, y_t6, fam_t6, mk, n_boot=1000, seed=20263250)
            paired_boot[f'blend_3way - L1_GN2_ens3 ({stag})'] = _paired_family_bootstrap_full(b_3w, p_l1_t6, y_t6, fam_t6, mk, n_boot=1000, seed=20263250)

    dro_p = STAGE2_DIR / 'stage2_dro_results.json'
    dro_doc = json.loads(dro_p.read_text()) if dro_p.is_file() else None

    doc = {
        'UTC': datetime.now(timezone.utc).isoformat(),
        'tune6k_events': int(len(y_t6)),
        'tune6k_ens3_raw': _ba4_and_nll(y_t6, p_t6),
        'per_seed': {f'seed_{s}': _ba4_and_nll(y_t6, seed_p[s][tune6k_m]) for s in STAGE2_SEEDS},
        'by_obs_domain': dom_res,
        'by_truth_region': reg_res,
        'tau_tail': tau_tail,
        'pv_audit': pv_audit,
        'pv_audit_uncal': pv_audit_uncal,
        'pv_scale_ens3_mult': ens_mult.tolist(),
        'ood_ens3': ood_ens3,
        'stress_shard080_ens3': stress_ens3,
        'score_bank_kl_dro': dro_doc,
        'blends': blends,
        'paired_family_bootstrap_1000_seed20263250': paired_boot,
    }
    write_json(STAGE2_DIR / 'stage2_l2_jepa_results.json', doc)
    return doc


# ---------------------------------------------------------------------------
# HTCondor (`bigbird20.cern.ch` / `choose_schedd`) & Kubeflow Submission
# ---------------------------------------------------------------------------
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
    min_gpu_mb: int = 38000,
    min_cap: float = 8.0,
    require_non_mig: bool = False,
) -> dict[str, Any]:
    job_dir = CONDOR_DIR / jname
    job_dir.mkdir(parents=True, exist_ok=True)
    (EXP_ROOT / 'logs').mkdir(parents=True, exist_ok=True)

    script = [
        '#!/bin/bash',
        'set -o pipefail',
        'command -v which >/dev/null 2>&1 || { which() { command -v "$@"; }; export -f which; }',
        'SAVED_ARGS=("$@"); set --',
        'source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh >/dev/null 2>&1',
        'set -- "${SAVED_ARGS[@]}"',
        'set -e',
        'ls /eos/home-x/xcheng/FASER/Pinpoint/results >/dev/null',
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1',
        'export PINPOINT_V3_SCRATCH="${_CONDOR_SCRATCH_DIR:-/tmp}/pinpoint_l2_${PINPOINT_JOB_ID}"',
        'trap \'rm -rf "$PINPOINT_V3_SCRATCH"\' EXIT INT TERM',
        'mkdir -p "$PINPOINT_V3_SCRATCH/code"',
        f'tar -xzf {shlex.quote(str(archive))} -C "$PINPOINT_V3_SCRATCH/code"',
        'export PYTHONPATH="$PINPOINT_V3_SCRATCH/code/src:${PYTHONPATH:-}"',
        'cd "$PINPOINT_V3_SCRATCH"',
        f'exec /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/bin/python -u -m analysis.flavour.nextgen.run_l2_ssl_jepa {cmd_args}',
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
        req_g = gpu_requirement(min_cap, minimum_memory_mb=min_gpu_mb)
        if require_non_mig:
            req_g = f'({req_g}) && !regexp("MIG", DeviceName)'
        sub_lines.extend([
            '+AllowBackfillSlot = true',
            'request_gpus = 1',
            'require_gpus = ' + req_g,
        ])
    sub_lines.append('queue 1')
    sub_path = job_dir / 'job.sub'
    sub_path.write_text('\n'.join(sub_lines) + '\n')

    resp = subprocess.run(['condor_submit', '-name', chosen_schedd, str(sub_path)], capture_output=True, text=True)
    if resp.returncode != 0:
        raise RuntimeError(f'condor_submit failed for {jname}: {resp.stderr}')
    return {'schedd': chosen_schedd, 'stdout': resp.stdout.strip()}


def submit_jobs(action: str, variants: list[str] | None = None, schedd: str | None = None) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[4]
    EXP_ROOT.mkdir(parents=True, exist_ok=True)
    archive = EXP_ROOT / 'code' / f'l2_jepa_src_{int(time.time())}.tar.gz'
    _package_source(repo, archive)
    chosen = schedd if schedd else choose_schedd(EXP_ROOT)[0]
    subs: dict[str, Any] = {}

    if action == 'submit-stage1-preextract':
        for part in range(9):
            subs[f's1_pre_{part}'] = _submit_condor_job(
                f's1_pre_{part}', f'--mode stage1-preextract --part {part}',
                archive, chosen, cpus=8, mem='16GB', disk='12GB', flavour='microcentury', use_gpu=False,
            )
    elif action == 'submit-ssl-pretrain':
        vlist = variants or list(SSL_VARIANTS.keys())
        for sv in vlist:
            subs[sv] = _submit_condor_job(
                sv, f'--mode ssl-pretrain --ssl-variant {sv} --epochs 14 --batch-size 256',
                archive, chosen, cpus=8, mem='28GB', disk='20GB', flavour='tomorrow', use_gpu=True, min_gpu_mb=20000,
            )
    elif action == 'submit-ssl-fast':
        subs['ssl_full_fast'] = _submit_condor_job(
            'ssl_full_fast',
            '--mode ssl-pretrain --ssl-variant ssl_full --out-dir-name ssl_full_fast --epochs 14 --batch-size 256 --micro-bs 128',
            archive, chosen, cpus=8, mem='32GB', disk='20GB', flavour='tomorrow', use_gpu=True, min_gpu_mb=38000, require_non_mig=True,
        )
    elif action == 'submit-ssl-h100':
        subs['ssl_full_h100'] = _submit_condor_job(
            'ssl_full_h100',
            '--mode ssl-pretrain --ssl-variant ssl_full --out-dir-name ssl_full_h100 --epochs 14 --batch-size 256 --micro-bs 128',
            archive, chosen, cpus=10, mem='52000MB', disk='20GB', flavour='tomorrow', use_gpu=True, min_gpu_mb=20000, min_cap=9.0, require_non_mig=True,
        )
    elif action == 'submit-stage1-train':
        vlist = variants or list(STAGE1_MATRIX.keys())
        for v in vlist:
            is_eq = (v == 'sup_scratch_equal_compute')
            subs[v] = _submit_condor_job(
                v, f'--mode stage1-train --variant {v}',
                archive, chosen,
                cpus=(6 if is_eq else 4),
                mem=('32GB' if is_eq else '15GB'),
                disk='20GB',
                flavour='workday',
                use_gpu=True,
                min_gpu_mb=(20000 if is_eq else 10000),
            )
    elif action == 'submit-ssl-precal':
        vlist = variants or list(SSL_VARIANTS.keys())
        for sv in vlist:
            subs[f'probe_{sv}'] = _submit_condor_job(
                f'probe_{sv}', f'--mode ssl-eval-precal --ssl-variant {sv}',
                archive, chosen, cpus=4, mem='12GB', disk='8GB', flavour='microcentury', use_gpu=True, min_gpu_mb=8000,
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
                archive, chosen, cpus=4, mem='15GB', disk='20GB', flavour='workday', use_gpu=True, min_gpu_mb=10000,
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
                'ssl-pretrain',
                'ssl-eval-precal',
                'stage1-preextract',
                'stage1-train',
                'stage1-aggregate',
                'stage2-preextract',
                'stage2-train',
                'stage2-dro',
                'stage2-aggregate',
                'submit-stage1-preextract',
                'submit-ssl-pretrain',
                'submit-ssl-fast',
                'submit-ssl-h100',
                'submit-ssl-precal',
                'submit-stage1-train',
                'submit-stage2-preextract',
                'submit-stage2-train',
            ),
        )
        parser.add_argument('--ssl-variant', type=str, default='ssl_full')
        parser.add_argument('--out-dir-name', type=str, default=None)
        parser.add_argument('--epochs', type=int, default=15)
        parser.add_argument('--batch-size', type=int, default=512)
        parser.add_argument('--micro-bs', type=int, default=64)
        parser.add_argument('--lr', type=float, default=1.0e-3)
        parser.add_argument('--part', type=int, default=0)
        parser.add_argument('--variant', type=str, default='ssl300k_jepa_ft_100pct')
        parser.add_argument('--variants', nargs='*', default=None)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--schedd', type=str, default=None)
        args = parser.parse_args()

        is_stage2 = 'stage2' in args.mode
        if not is_stage2:
            install_stage1_only_hook()

        if args.mode == 'ssl-pretrain':
            run_ssl_pretrain(
                args.ssl_variant,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed,
                micro_bs=args.micro_bs,
                out_dir_name=args.out_dir_name,
            )
        elif args.mode == 'ssl-eval-precal':
            run_ssl_eval_precal(args.ssl_variant)
        elif args.mode == 'stage1-preextract':
            print(json.dumps(run_stage1_preextract(args.part), indent=2))
        elif args.mode == 'stage1-train':
            run_stage1_train(args.variant)
        elif args.mode == 'stage1-aggregate':
            run_stage1_aggregate()
        elif args.mode == 'stage2-preextract':
            print(json.dumps(run_stage2_preextract(args.part), indent=2))
        elif args.mode == 'stage2-train':
            run_stage2_train(args.seed)
        elif args.mode == 'stage2-dro':
            print(json.dumps(run_stage2_dro(), indent=2))
        elif args.mode == 'stage2-aggregate':
            print(json.dumps(run_stage2_aggregate(), indent=2))
        else:
            print(json.dumps(submit_jobs(args.mode, variants=args.variants, schedd=args.schedd), indent=2))
    finally:
        guard.__exit__(None, None, None)


if __name__ == '__main__':
    main()
