"""Q13 Near-Term Prototype: Hardware-Equivalent Full 1.00 mm Sensor Control (`abl_full_1mm_sensor`).

Reuses frozen `TrackletIPFormer` without modifying `tracklet_ipformer.py`. Implements the exact
paired Section 2.2 hardware-equivalent 1.00 mm sensor specification:
  - Locks native 20 um connected-component clustering & token retention (`N_c`, `keep_idx`, `c_obs`,
    `obs_domain_id`, `bar_*`, `trk_cand`, and ALL IPT `plane >= 12` tokens/patches bit-identical).
  - For Target planes (`plane < 12`), snaps raw hits inside each locked cluster onto the symmetric
    1.00 mm binary voxel grid around `(x_c, y_c)`, deduplicates 1-bit voxels, quantizes centroids
    `(x_1mm, y_1mm)`, recomputes target `pix_feat`, `plane_summary[:12]`, `32x32` target `pix_patches`
    at `1.00 mm` pitch, `reconstruct_micron_vertex(..., 1.00, 1.00)`, and soft tracklets at `pitch=(1.00, 1.00)`.

Enforces Zero-Exception Seal + Stage-I `080..099` isolation via `SealAuditGuard` + audit hook.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import copy
from datetime import datetime, timezone
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
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler

from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.nextgen.protocols import compute_student_t_pulls
from analysis.flavour.nextgen.run_tau_kinema_diffnet import _paired_family_bootstrap_delta
from analysis.flavour.nextgen.seal_guard import (
    SealAuditGuard,
    SealViolationError,
    assert_unsealed_split,
    load_authorized_split_shards,
    verify_inference_truth_invariance,
)
from analysis.flavour.nextgen.tracklet_ipformer import (
    MAX_TRACKLETS,
    TrackletIPFormer,
    collate_tracklet_pf,
    tracklet_ipformer_loss,
)
from analysis.flavour.nextgen.tracklet_ops import extract_soft_tracklets_single
from analysis.flavour.v2.partitions import FAMILY, KEY
from analysis.flavour.v2.profile import init_worker, shutdown_loader
from analysis.flavour.v2.r3.mapped_inputs import MappedItems, write_mapped_items
from analysis.flavour.v3.common.calibration import _tau_efficiency_at_fpr
from analysis.flavour.v3.common.domains import (
    calibrated_bar_channels,
    compute_observable_containment,
    cumulative_radiation_length,
    detector_bounds,
    validate_finite_observation,
)
from analysis.flavour.v3.common.fast_kernels import (
    MICRO_VTX_FEAT_DIM,
    reconstruct_micron_vertex,
)
from analysis.flavour.v3.v1_b1pp.spectrometer import extract_spectrometer_kinematics
from analysis.flavour.v3.v2_pinpointformer.pixel_tokenizer import (
    N_MAX_IPT_CLUSTERS,
    N_MAX_IPT_PATCHES,
    N_MAX_TARGET_CLUSTERS,
    N_MAX_TARGET_PATCHES,
    PATCH_SIZE,
    PATCH_TRIGGER_MIN_HITS,
    PATCH_TRIGGER_MIN_SPAN_MM,
    PIX_FEAT_DIM,
    PIX_PLANE_SUMMARY_DIM,
    _cluster_sensor_groups,
    _compute_plane_neighborhoods,
    _rasterize_micro_patches,
    _select_group_clusters,
)
from analysis.flavour.v3.v3_multitask.losses import compute_window_task_denominators
from analysis.flavour.v3.v6_clean_sim.evaluate_stage317 import paired_family_bootstrap_delta
from analysis.flavour.v3.v6_clean_sim.run_stage325_remediation import (
    _ba4_and_nll,
    _split_half_tail_eval,
)


CLEAN_V3_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v6_clean_v3_2026-11-19/clean_sim')
STAGE325_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v7_stage325_2026-11-20')
SPLITS_DIR = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/forensics/splits')
TIP_EXP_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/exp_tracklet_ipformer')
DEFAULT_OUT_DIR = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/exp_n2_tau_kinema/q13_full_1mm_sensor')

OOD_W184_NC_CHARM = STAGE325_ROOT / 'ood_shards/gen_ood_w184_epps21_nc_charm'
OOD_RUN10085_10088 = CLEAN_V3_ROOT / 'ood_shards/gen_ood_run10085_10088'

STAGE1_FORBIDDEN_RE = re.compile(
    r'(?:^|/)shards/0(?:8\d|9\d)(?:/|$)|validation_meta|tune_screen\.parquet|tune_truth_meta\.parquet',
    re.IGNORECASE,
)
_STAGE1_HOOK_INSTALLED = False
_STAGE1_VIOLATIONS: list[str] = []


def install_stage1_seal_guard() -> SealAuditGuard:
    """Install both SealAuditGuard (100..149/sealed) and Stage-I Tune8k/080..099 hook."""
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
                path_str = os.fspath(target) if hasattr(target, '__fspath__') else str(target)
            except Exception:
                return
            if isinstance(path_str, bytes):
                path_str = path_str.decode('utf-8', errors='ignore')
            norm = path_str.replace('\\', '/')
            if STAGE1_FORBIDDEN_RE.search(norm):
                _STAGE1_VIOLATIONS.append(f'{event}:{norm}')
                raise SealViolationError(f'Stage-1 forbids Tune8k/validation access: {norm}')

        sys.addaudithook(_stage1_hook)
        _STAGE1_HOOK_INSTALLED = True
    return guard


def get_q13_config() -> dict[str, Any]:
    """Exact Stage-I matched-budget TrackletIPFormer config with full 1.00 mm target sensor."""
    return {
        'seed': 42,
        'task': 'T4',
        'passes': 20,
        'batch_size': 128,
        'micro_bs': 64,
        'grad_accum': 2,
        'min_lr': 1.0e-5,
        'weight_decay': 0.03,
        'warmup_fraction': 0.06,
        'ema_decay': 0.999,
        'pe_only': True,
        'bar_threshold_pe': 2.5,
        'sipm_npix': 14400.0,
        'apply_sipm_saturation': False,
        'invert_sipm_saturation': True,
        'y_mirror': False,
        'quantize_cluster_coords_only_mm': 1.00,
        'q13_full_1mm_sensor': True,
        'family': 'tracklet_ipformer',
        'dim': 176,
        'heads': 8,
        'n_rel_pre': 2,
        'n_rel_post': 2,
        'n_global_layers': 2,
        'drop_path': 0.10,
        'head_dropout': 0.15,
        'learning_rate': 8.0e-4,
        'grad_clip': 2.0,
    }


def extract_full_1mm_sensor_item(
    index: int,
    observation: dict,
    label: int | None,
    dom: dict,
    geometry: dict,
    bounds: dict,
    config: dict,
) -> dict[str, Any]:
    """Extract fixed-20um-cluster item with 1-bit 1.00 mm snapped target hits, centroids, moments & patches."""
    cfg = dict(config or {})
    pe_only = bool(cfg.get('pe_only', False))
    validate_finite_observation(observation, check_edep=not pe_only)

    pixel = observation['pixel']
    bar = observation['bar']
    tracker = observation['tracker_hypothesis']

    pcid = np.asarray(pixel['channel_id'], dtype=np.int64)
    pxyz = np.asarray(pixel['xyz_mm'], dtype=np.float64).reshape(-1, 3)
    raw_pixel_hits = len(pcid)

    n_pix_planes = int(bounds['nz'])
    nx_raw, ny_raw = int(bounds['nx']), int(bounds['ny'])
    x_c = float(bounds['x_pix_center'])
    y_c = float(bounds['y_scint_center'])
    z_mid = float(bounds['z_mid'])
    pix_zpos = np.asarray(bounds['pixel_Zpos'], dtype=np.float64)
    pix_x0 = np.array([cumulative_radiation_length(float(z), geometry) for z in pix_zpos], dtype=np.float32)

    n_target_planes = int(geometry['nPinpointBlocks']) + max(0, int(geometry['nFortuneBlocks']) - 1)
    dx_nat, dy_nat = float(bounds['dx_mm']), float(bounds['dy_mm'])
    q_mm = 1.00

    plane_summary = np.zeros((n_pix_planes, PIX_PLANE_SUMMARY_DIM), dtype=np.float32)
    plane_summary[:, 5] = pix_x0 / 50.0

    if raw_pixel_hits > 0:
        # Step 1: Lock exact native 20 um connected-component clustering & keep_idx
        cl = _cluster_sensor_groups(pcid, pxyz, nx_raw, ny_raw, n_target_planes, 0.06, 0.06)
        n_cl = int(cl['n_clusters'])
        order_nat = cl['hit_order']
        offsets_nat = cl['cluster_offsets']
        starts_nat = offsets_nat[:-1]
        n_hits_nat = np.diff(offsets_nat).astype(np.float64)
        cl_plane = cl['cluster_plane'].astype(np.int64)
        cl_fcid = cl['cluster_first_channel_id'].astype(np.uint64)

        hx_nat = pxyz[order_nat, 0].astype(np.float64)
        hy_nat = pxyz[order_nat, 1].astype(np.float64)
        x_mean_nat = np.add.reduceat(hx_nat, starts_nat) / n_hits_nat
        y_mean_nat = np.add.reduceat(hy_nat, starts_nat) / n_hits_nat
        z_cl = pix_zpos[cl_plane]

        _, _, _, hits_r040_nat, _, cnt_r5_nat, _ = _compute_plane_neighborhoods(
            cl_plane, x_mean_nat, y_mean_nat, n_hits_nat
        )
        tgt_cl_all = np.flatnonzero(cl_plane < n_target_planes)
        ipt_cl_all = np.flatnonzero(cl_plane >= n_target_planes)
        keep_tgt = _select_group_clusters(tgt_cl_all, cl_plane, n_hits_nat, hits_r040_nat, cnt_r5_nat, N_MAX_TARGET_CLUSTERS)
        keep_ipt = _select_group_clusters(ipt_cl_all, cl_plane, n_hits_nat, hits_r040_nat, cnt_r5_nat, N_MAX_IPT_CLUSTERS)
        keep_idx = np.concatenate([keep_tgt, keep_ipt])

        # Lock native patch trigger indices before coarse resampling
        dx_raw_nat = np.maximum.reduceat(hx_nat, starts_nat) - np.minimum.reduceat(hx_nat, starts_nat)
        dy_raw_nat = np.maximum.reduceat(hy_nat, starts_nat) - np.minimum.reduceat(hy_nat, starts_nat)
        trig_nat = (
            (n_hits_nat[keep_idx] >= PATCH_TRIGGER_MIN_HITS)
            & (np.maximum(dx_raw_nat[keep_idx], dy_raw_nat[keep_idx]) > PATCH_TRIGGER_MIN_SPAN_MM)
        )
        trig_tgt = np.flatnonzero(trig_nat & (cl_plane[keep_idx] < n_target_planes))
        if len(trig_tgt) > N_MAX_TARGET_PATCHES:
            rank_t = np.argsort(-n_hits_nat[keep_idx[trig_tgt]], kind='mergesort')
            trig_tgt = np.sort(trig_tgt[rank_t[:N_MAX_TARGET_PATCHES]])
        trig_ipt = np.flatnonzero(trig_nat & (cl_plane[keep_idx] >= n_target_planes))
        if len(trig_ipt) > N_MAX_IPT_PATCHES:
            rank_i = np.argsort(-n_hits_nat[keep_idx[trig_ipt]], kind='mergesort')
            trig_ipt = np.sort(trig_ipt[rank_i[:N_MAX_IPT_PATCHES]])

        # Step 2: Snap Target cluster hits onto 1-bit 1.00 mm voxel grid; keep IPT bit-exact!
        x_out = x_mean_nat.copy()
        y_out = y_mean_nat.copy()
        n_hits_q = n_hits_nat.copy()
        hx_chunks: list[np.ndarray] = []
        hy_chunks: list[np.ndarray] = []
        offsets_q = np.zeros(n_cl + 1, dtype=np.int32)
        cur_pos = 0
        for cid_i in range(n_cl):
            s_i = int(offsets_nat[cid_i])
            e_i = int(offsets_nat[cid_i + 1])
            if cl_plane[cid_i] < n_target_planes:
                cx_i = np.rint((hx_nat[s_i:e_i] - x_c) / q_mm).astype(np.int64)
                cy_i = np.rint((hy_nat[s_i:e_i] - y_c) / q_mm).astype(np.int64)
                uniq_c = np.unique(np.column_stack([cx_i, cy_i]), axis=0)
                vx_i = x_c + q_mm * uniq_c[:, 0].astype(np.float64)
                vy_i = y_c + q_mm * uniq_c[:, 1].astype(np.float64)
                hx_chunks.append(vx_i)
                hy_chunks.append(vy_i)
                n_hits_q[cid_i] = float(len(vx_i))
                x_out[cid_i] = x_c + round((float(vx_i.mean()) - x_c) / q_mm) * q_mm
                y_out[cid_i] = y_c + round((float(vy_i.mean()) - y_c) / q_mm) * q_mm
                cur_pos += len(vx_i)
            else:
                hx_chunks.append(hx_nat[s_i:e_i])
                hy_chunks.append(hy_nat[s_i:e_i])
                cur_pos += (e_i - s_i)
            offsets_q[cid_i + 1] = cur_pos

        hx_q = np.concatenate(hx_chunks) if hx_chunks else np.empty(0, dtype=np.float64)
        hy_q = np.concatenate(hy_chunks) if hy_chunks else np.empty(0, dtype=np.float64)
        starts_q = offsets_q[:-1]
        order_q = np.arange(len(hx_q), dtype=np.int32)
        xyz_q = np.column_stack([hx_q, hy_q, np.zeros_like(hx_q)])

        px_plane = np.where(np.arange(n_pix_planes) < n_target_planes, q_mm, dx_nat).astype(np.float64)
        py_plane = np.where(np.arange(n_pix_planes) < n_target_planes, q_mm, dy_nat).astype(np.float64)
        px_cl = px_plane[cl_plane]
        py_cl = py_plane[cl_plane]

        # Step 3: Recompute plane neighborhoods, plane_summary[:12], pix_feat, patches & tracklets
        (
            cnt_r015,
            hits_r015,
            cnt_r040,
            hits_r040,
            cnt_r1,
            cnt_r5,
            hits_r5,
        ) = _compute_plane_neighborhoods(cl_plane, x_out, y_out, n_hits_q)

        p_hit_cnt = np.bincount(cl_plane, weights=n_hits_q, minlength=n_pix_planes)
        p_cl_cnt = np.bincount(cl_plane, minlength=n_pix_planes)
        p_xsum = np.bincount(cl_plane, weights=x_out * n_hits_q, minlength=n_pix_planes)
        p_ysum = np.bincount(cl_plane, weights=y_out * n_hits_q, minlength=n_pix_planes)
        p_act = p_hit_cnt > 0

        w_cl = np.square(n_hits_q + hits_r040)
        p_wsum = np.bincount(cl_plane, weights=w_cl, minlength=n_pix_planes)
        plane_xc = np.where(
            p_act,
            np.bincount(cl_plane, weights=w_cl * x_out, minlength=n_pix_planes) / np.maximum(1e-12, p_wsum),
            x_c,
        )
        plane_yc = np.where(
            p_act,
            np.bincount(cl_plane, weights=w_cl * y_out, minlength=n_pix_planes) / np.maximum(1e-12, p_wsum),
            y_c,
        )

        n_rep_q = n_hits_q.astype(np.int64)
        hit_pl = np.repeat(cl_plane, n_rep_q)
        dx_hp = hx_q - plane_xc[hit_pl]
        dy_hp = hy_q - plane_yc[hit_pl]
        dr_hp = np.hypot(dx_hp, dy_hp)
        p_h015 = np.bincount(hit_pl, weights=(dr_hp <= 0.15).astype(np.float64), minlength=n_pix_planes)
        p_h040 = np.bincount(hit_pl, weights=(dr_hp <= 0.40).astype(np.float64), minlength=n_pix_planes)
        p_h200 = np.bincount(hit_pl, weights=(dr_hp <= 2.00).astype(np.float64), minlength=n_pix_planes)

        core25 = dr_hp <= 2.50
        h_c25 = np.bincount(hit_pl[core25], minlength=n_pix_planes).astype(np.float64)
        p_cmx = np.bincount(hit_pl[core25], weights=dx_hp[core25], minlength=n_pix_planes) / np.maximum(1.0, h_c25)
        p_cmy = np.bincount(hit_pl[core25], weights=dy_hp[core25], minlength=n_pix_planes) / np.maximum(1.0, h_c25)
        p_cxx = np.bincount(hit_pl[core25], weights=dx_hp[core25] * dx_hp[core25], minlength=n_pix_planes)
        p_cyy = np.bincount(hit_pl[core25], weights=dy_hp[core25] * dy_hp[core25], minlength=n_pix_planes)
        p_cxy = np.bincount(hit_pl[core25], weights=dx_hp[core25] * dy_hp[core25], minlength=n_pix_planes)
        cvxx = np.maximum(0.0, p_cxx / np.maximum(1.0, h_c25) - p_cmx * p_cmx) + (px_plane ** 2) / 12.0
        cvyy = np.maximum(0.0, p_cyy / np.maximum(1.0, h_c25) - p_cmy * p_cmy) + (py_plane ** 2) / 12.0
        cvxy = p_cxy / np.maximum(1.0, h_c25) - p_cmx * p_cmy
        c_tr = cvxx + cvyy
        c_df = cvxx - cvyy
        c_disc = np.sqrt(np.maximum(0.0, c_df * c_df + 4.0 * cvxy * cvxy))
        c_l1 = np.maximum(1e-12, 0.5 * (c_tr + c_disc))
        c_l2 = np.maximum(1e-12, 0.5 * (c_tr - c_disc))
        core_ecc = np.where(h_c25 >= 2.0, np.sqrt(np.clip(1.0 - c_l2 / c_l1, 0.0, 1.0)), 0.0)

        plane_summary[p_act, 0] = 1.0
        plane_summary[p_act, 1] = np.log1p(p_hit_cnt[p_act]) / 8.0
        plane_summary[p_act, 2] = np.log1p(p_cl_cnt[p_act]) / 6.0
        plane_summary[p_act, 3] = (p_xsum[p_act] / p_hit_cnt[p_act] - x_c) / 135.0
        plane_summary[p_act, 4] = (p_ysum[p_act] / p_hit_cnt[p_act] - y_c) / 100.0
        plane_summary[p_act, 6] = (plane_yc[p_act] - y_c) / 100.0
        plane_summary[p_act, 7] = (plane_xc[p_act] - x_c) / 135.0
        plane_summary[p_act, 8] = np.log1p(p_h015[p_act]) / 6.0
        plane_summary[p_act, 9] = np.log1p(p_h040[p_act]) / 7.0
        plane_summary[p_act, 10] = p_h040[p_act] / np.maximum(1.0, p_h200[p_act])
        plane_summary[p_act, 11] = core_ecc[p_act]

        dx_raw = np.maximum.reduceat(hx_q, starts_q) - np.minimum.reduceat(hx_q, starts_q)
        dy_raw = np.maximum.reduceat(hy_q, starts_q) - np.minimum.reduceat(hy_q, starts_q)
        dx_box = dx_raw + px_cl
        dy_box = dy_raw + py_cl
        fill_factor = np.clip(n_hits_q / np.maximum(1.0, (dx_box / px_cl) * (dy_box / py_cl)), 0.0, 1.0)

        # Moments around cluster hit-mean (note: for target 1-bit voxels, mean of vx_i is used for central moments)
        hx_mean_q = np.add.reduceat(hx_q, starts_q) / n_hits_q
        hy_mean_q = np.add.reduceat(hy_q, starts_q) / n_hits_q
        dhx = hx_q - np.repeat(hx_mean_q, n_rep_q)
        dhy = hy_q - np.repeat(hy_mean_q, n_rep_q)
        var_x = np.add.reduceat(dhx * dhx, starts_q) / n_hits_q + (px_cl ** 2) / 12.0
        var_y = np.add.reduceat(dhy * dhy, starts_q) / n_hits_q + (py_cl ** 2) / 12.0
        cov_xy = np.add.reduceat(dhx * dhy, starts_q) / n_hits_q

        trace = var_x + var_y
        diff_v = var_x - var_y
        disc = np.sqrt(np.maximum(0.0, diff_v * diff_v + 4.0 * cov_xy * cov_xy))
        lam1 = np.maximum(1e-12, 0.5 * (trace + disc))
        lam2 = np.maximum(1e-12, 0.5 * (trace - disc))
        ecc = np.sqrt(np.clip(1.0 - lam2 / lam1, 0.0, 1.0))
        cos2a = np.where(disc > 1e-12, diff_v / np.maximum(disc, 1e-12), 0.0)
        sin2a = np.where(disc > 1e-12, (2.0 * cov_xy) / np.maximum(disc, 1e-12), 0.0)
        skew_x = (np.add.reduceat(dhx * dhx * dhx, starts_q) / n_hits_q) / np.power(var_x, 1.5)
        skew_y = (np.add.reduceat(dhy * dhy * dhy, starts_q) / n_hits_q) / np.power(var_y, 1.5)
        skew_norm = np.clip(np.hypot(skew_x, skew_y), 0.0, 5.0) / 5.0

        pix_xyz = np.column_stack([x_out[keep_idx], y_out[keep_idx], z_cl[keep_idx]]).astype(np.float32)
        dx_core = x_out[keep_idx] - plane_xc[cl_plane[keep_idx]]
        dy_core = y_out[keep_idx] - plane_yc[cl_plane[keep_idx]]
        r_core = np.hypot(dx_core, dy_core)

        pix_feat = np.column_stack([
            (x_out[keep_idx] - x_c) / 135.0,
            (y_out[keep_idx] - y_c) / 100.0,
            (z_cl[keep_idx] - z_mid) / 1000.0,
            np.log1p(n_hits_q[keep_idx]) / 6.0,
            np.arcsinh(dx_box[keep_idx] / 0.1) / 3.0,
            np.arcsinh(dy_box[keep_idx] / 0.1) / 3.0,
            fill_factor[keep_idx],
            np.arcsinh(np.sqrt(lam1[keep_idx]) / 0.05) / 3.0,
            np.arcsinh(np.sqrt(lam2[keep_idx]) / 0.05) / 3.0,
            ecc[keep_idx],
            cos2a[keep_idx],
            sin2a[keep_idx],
            skew_norm[keep_idx],
            np.log1p(cnt_r1[keep_idx]) / 4.0,
            np.log1p(cnt_r5[keep_idx]) / 5.0,
            np.log1p(hits_r5[keep_idx]) / 7.0,
            np.arcsinh(dx_core / 0.10) / 4.0,
            np.arcsinh(dy_core / 0.10) / 4.0,
            np.arcsinh(r_core / 0.08) / 4.0,
            np.log1p(cnt_r015[keep_idx]) / 3.0,
            np.log1p(hits_r015[keep_idx]) / 5.0,
            np.log1p(cnt_r040[keep_idx]) / 4.0,
        ]).astype(np.float32)
        pix_plane = cl_plane[keep_idx].astype(np.int8)
        pix_first_cid = cl_fcid[keep_idx]

        # Target patches rasterized at 1.00 mm pitch on deduplicated 1.00 mm voxels centered at (x_1mm, y_1mm);
        # IPT patches rasterized bit-identical to native 20 um!
        p_t, idx_t = _rasterize_micro_patches(
            order_q, offsets_q, n_hits_q, xyz_q, x_out, y_out, trig_tgt, keep_idx, q_mm, q_mm
        )
        p_i, idx_i = _rasterize_micro_patches(
            order_nat, offsets_nat, n_hits_nat, pxyz, x_mean_nat, y_mean_nat, trig_ipt, keep_idx, dx_nat, dy_nat
        )
        if len(idx_t) and len(idx_i):
            pix_patches = np.concatenate([p_t, p_i], axis=0)
            patch_cl_idx = np.concatenate([idx_t, idx_i], axis=0)
        elif len(idx_t):
            pix_patches, patch_cl_idx = p_t, idx_t
        else:
            pix_patches, patch_cl_idx = p_i, idx_i

        has_tgt_cl = len(tgt_cl_all) > 0
        mv_px = q_mm if has_tgt_cl else dx_nat
        mv_py = q_mm if has_tgt_cl else dy_nat
        avr_xyz, avr_ok, micro_vtx_feat = reconstruct_micron_vertex(
            cl_plane, x_out, y_out, z_cl, n_hits_q, lam2,
            n_pix_planes, mv_px, mv_py, dom['cal_start'], (x_c, y_c, z_mid),
        )
    else:
        pix_xyz = np.empty((0, 3), dtype=np.float32)
        pix_feat = np.empty((0, PIX_FEAT_DIM), dtype=np.float32)
        pix_plane = np.empty(0, dtype=np.int8)
        pix_first_cid = np.empty(0, dtype=np.uint64)
        pix_patches = np.empty((0, 2, PATCH_SIZE, PATCH_SIZE), dtype=np.float16)
        patch_cl_idx = np.empty(0, dtype=np.int16)
        sz0, sx0, sy0, sv0 = dom['cal_start']
        avr_xyz = np.array([sx0, sy0, sz0] if sv0 else [x_c, y_c, z_mid], dtype=np.float32)
        avr_ok = False
        micro_vtx_feat = np.zeros(MICRO_VTX_FEAT_DIM, dtype=np.float32)

    keep_bar, be, bpe = calibrated_bar_channels(bar, cfg)
    bp = np.asarray(bar['panel'], dtype=np.int16)[keep_bar]
    bh = np.asarray(bar['horizontal'], dtype=np.int8)[keep_bar]
    bi = np.asarray(bar['bar'], dtype=np.int16)[keep_bar]
    if len(bp) > 0:
        bar_val = np.column_stack([
            np.log1p(np.maximum(be, 0.0)) / 10.0,
            np.log1p(np.maximum(bpe, 0.0)) / 15.0,
        ]).astype(np.float16)
    else:
        bar_val = np.empty((0, 2), dtype=np.float16)

    if 'hit_time_ns' in bar and len(bar['hit_time_ns']) == len(keep_bar):
        bar_ht = np.asarray(bar['hit_time_ns'], dtype=np.float32)[keep_bar]
    else:
        bar_ht = np.full(len(bp), -1.0, dtype=np.float32)

    spec = extract_spectrometer_kinematics(tracker, dom['cal_start'], bounds)
    sz, sx, sy, svalid = dom['cal_start']
    avr_pv = np.array([avr_xyz[0], avr_xyz[1], avr_xyz[2], 1.0 if avr_ok else 0.0], dtype=np.float32)

    item: dict[str, Any] = {
        'index': int(index),
        'obs_domain_id': int(dom['obs_domain_id']),
        'c_obs': dom['c_obs'].astype(np.float32),
        'cal_start': np.array([sz, sx, sy, 1.0 if svalid else 0.0], dtype=np.float32),
        'avr_pv_mm': avr_pv,
        'micro_vtx_feat': micro_vtx_feat,
        'raw_pixel_hits': int(raw_pixel_hits),
        'pix_xyz_mm': pix_xyz,
        'pix_feat': pix_feat,
        'pix_plane': pix_plane,
        'pix_first_cid': pix_first_cid,
        'pix_plane_summary': plane_summary,
        'pix_patches': pix_patches,
        'patch_cluster_idx': patch_cl_idx,
        'bar_panel': bp,
        'bar_horiz': bh,
        'bar_idx': bi,
        'bar_val': bar_val,
        'bar_hit_time_ns': bar_ht,
        'trk_cand': spec['cand'],
        'trk_cand_mask': spec['cand_mask'],
        'trk_hits': spec['hits'],
        'bar_keep_mask': keep_bar,
    }
    if label is not None:
        item['label'] = int(label)

    trk = extract_soft_tracklets_single(
        item['pix_xyz_mm'],
        item['pix_plane'],
        item['pix_feat'],
        pix_x0,
        max_tracklets=int(cfg.get('max_tracklets', MAX_TRACKLETS)),
        pitch_xy_mm=(q_mm, q_mm),
    )
    item['trk_kin'] = trk['trk_kin']
    item['trk_morph'] = trk['trk_morph']
    item['trk_mask'] = trk['trk_mask']
    return item


def _configure_runtime() -> None:
    torch.set_num_threads(1)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    torch.multiprocessing.set_sharing_strategy('file_system')


def _to_device_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _load_stage1_split_keys() -> dict[str, pd.DataFrame]:
    assert_unsealed_split('internal_train', stage=1)
    assert_unsealed_split('internal_screen', stage=1)
    assert_unsealed_split('internal_val', stage=1)
    splits = {
        'train': pd.read_parquet(SPLITS_DIR / 'internal_train_keys.parquet'),
        'dev_screen': pd.read_parquet(SPLITS_DIR / 'internal_dev_screen_keys.parquet'),
        'val_eval': pd.read_parquet(SPLITS_DIR / 'internal_val_eval_keys.parquet'),
    }
    fams = {k: set(map(tuple, df[FAMILY].to_numpy().tolist())) for k, df in splits.items()}
    if fams['train'] & fams['dev_screen'] or fams['train'] & fams['val_eval'] or fams['dev_screen'] & fams['val_eval']:
        raise AssertionError('Family leakage across Stage-I split manifests!')
    return splits


class _Stage1MappedDataset(Dataset):
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
            it['vertex_xyz_mm'] = np.array(
                [row['vertex_x_mm'], row['vertex_y_mm'], row['vertex_z_mm']], dtype=np.float32
            )
            it['vertex_valid'] = int(row['vertex_valid'])
            it['vis_kinematics'] = np.array(
                [row['vis_pz_log1p'], row['vis_pt_log1p'], row['vis_cos_dphi'], row['vis_sin_dphi']],
                dtype=np.float32,
            )
            it['kin_valid'] = int(row['kin_valid'])
        return it


def _extract_shard_worker(task: tuple) -> dict[str, list[tuple[tuple[int, int, int], dict]]]:
    shard_path_str, key_split_map, geom, bounds, cfg = task
    install_stage1_seal_guard()
    shard_path = Path(shard_path_str)
    prov = pd.read_parquet(shard_path / 'provenance.parquet', columns=KEY + ['split', 'label'])
    obs_list = pq.read_table(shard_path / 'observations.parquet').to_pylist()
    tgt_list = pq.read_table(shard_path / 'targets.parquet').to_pylist()
    out: dict[str, list[tuple[tuple[int, int, int], dict]]] = {'train': [], 'dev_screen': [], 'val_eval': []}
    for i, (r, c, rw, sp_name, lbl) in enumerate(zip(prov['run'], prov['chunk'], prov['row'], prov['split'], prov['label'])):
        if sp_name != 'train':
            continue
        k = (int(r), int(c), int(rw))
        sp_dst = key_split_map.get(k)
        if sp_dst is None:
            continue
        ob = obs_list[i]
        t_lbl = int(tgt_list[i]['label'])
        assert t_lbl == int(lbl)
        dom = compute_observable_containment(ob, geom, bounds, config=cfg)
        it = extract_full_1mm_sensor_item(len(out[sp_dst]), ob, t_lbl, dom, geom, bounds, cfg)
        out[sp_dst].append((k, it))
    return out


def prepare_cache_part(cache_root: Path, part: int, n_parts: int, workers: int = 8) -> dict[str, Any]:
    _configure_runtime()
    if (cache_root / 'READY.json').is_file():
        return json.loads((cache_root / 'READY.json').read_text())
    parts_dir = cache_root / '_parts'
    parts_dir.mkdir(parents=True, exist_ok=True)
    part_file = parts_dir / f'part_{part:02d}_of_{n_parts:02d}.pkl'
    if part_file.is_file():
        return {'part': part, 'n_parts': n_parts, 'status': 'cached'}

    cfg = get_q13_config()
    geometry = json.loads((CLEAN_V3_ROOT / 'geometry.json').read_text())
    bounds = detector_bounds(geometry)
    splits = _load_stage1_split_keys()
    auth_shards = load_authorized_split_shards(CLEAN_V3_ROOT / 'data', 'internal_train', stage=1)
    my_shards = auth_shards[part::n_parts]
    scratch_base = '/dev/shm' if shutil.disk_usage('/dev/shm').free > 6 * (1024 ** 3) else os.environ.get('_CONDOR_SCRATCH_DIR', '/tmp')
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f'q13-p{part}-', dir=scratch_base) as tmp_str:
        local_root = Path(tmp_str) / 'native'
        local_root.mkdir(parents=True, exist_ok=True)

        def _copy_shard(sp: Path) -> Path:
            dst = local_root / sp.name
            dst.mkdir(parents=True, exist_ok=True)
            for fname in ('observations.parquet', 'targets.parquet', 'provenance.parquet'):
                shutil.copyfile(sp / fname, dst / fname)
            return dst

        with ThreadPoolExecutor(max_workers=min(8, max(1, workers))) as pool:
            local_shards = list(pool.map(_copy_shard, my_shards))

        key_split_map: dict[tuple[int, int, int], str] = {}
        for sp_name, df in splits.items():
            for r, c, rw in df[KEY].to_numpy():
                key_split_map[(int(r), int(c), int(rw))] = sp_name

        tasks = [(str(sp), key_split_map, geometry, bounds, cfg) for sp in local_shards]
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
            shard_parts = list(pool.map(_extract_shard_worker, tasks))

    collected: dict[str, list[tuple[tuple[int, int, int], dict]]] = {'train': [], 'dev_screen': [], 'val_eval': []}
    for sp_res in shard_parts:
        for sp_name in ('train', 'dev_screen', 'val_eval'):
            collected[sp_name].extend(sp_res[sp_name])
    tmp_out = parts_dir / f'.tmp_part_{part:02d}_of_{n_parts:02d}.pkl'
    with tmp_out.open('wb') as f:
        pickle.dump(collected, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_out, part_file)
    return {
        'part': part,
        'n_parts': n_parts,
        'shards': len(my_shards),
        'events': sum(len(v) for v in collected.values()),
        'seconds': float(time.perf_counter() - t0),
    }


def merge_cache(cache_root: Path, n_parts: int) -> dict[str, Any]:
    _configure_runtime()
    if (cache_root / 'READY.json').is_file():
        return json.loads((cache_root / 'READY.json').read_text())
    parts_dir = cache_root / '_parts'
    splits = _load_stage1_split_keys()
    t0 = time.perf_counter()
    collected: dict[str, dict[tuple[int, int, int], dict]] = {'train': {}, 'dev_screen': {}, 'val_eval': {}}
    for part in range(n_parts):
        pfile = parts_dir / f'part_{part:02d}_of_{n_parts:02d}.pkl'
        if not pfile.is_file():
            raise FileNotFoundError(f'Missing cache part {pfile}')
        with pfile.open('rb') as f:
            part_data = pickle.load(f)
        for sp_name in ('train', 'dev_screen', 'val_eval'):
            for k, it in part_data[sp_name]:
                collected[sp_name][k] = it

    for sp_name, df in splits.items():
        ordered = []
        for idx_row, (r, c, rw) in enumerate(df[KEY].to_numpy()):
            it = collected[sp_name][(int(r), int(c), int(rw))]
            it['index'] = int(idx_row)
            ordered.append(it)
        out_dir = cache_root / f'mapped_{sp_name}'
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
        write_mapped_items([ordered[i:i + 64] for i in range(0, len(ordered), 64)], out_dir)

    meta_rep = {'n_parts': n_parts, 'merge_seconds': float(time.perf_counter() - t0)}
    write_json(cache_root / 'READY.json', meta_rep)
    shutil.rmtree(parts_dir, ignore_errors=True)
    return meta_rep


def prepare_stage1_cache(cache_root: Path, workers: int = 8) -> dict[str, Any]:
    if (cache_root / 'READY.json').is_file():
        return json.loads((cache_root / 'READY.json').read_text())
    t0 = time.perf_counter()
    n_parts = 1
    prepare_cache_part(cache_root, 0, n_parts, workers=workers)
    rep = merge_cache(cache_root, n_parts)
    rep['total_seconds'] = float(time.perf_counter() - t0)
    return rep


def _make_loader(ds: Dataset, batch_size: int, shuffle: bool, seed: int, collate_fn: Any, workers: int) -> DataLoader:
    if shuffle:
        gen = torch.Generator()
        gen.manual_seed(seed)
        sampler: Any = RandomSampler(ds, generator=gen)
    else:
        sampler = SequentialSampler(ds)
    bsamp = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
    eff_w = 0 if len(ds) < 512 else max(0, min(workers, 6))
    kw: dict[str, Any] = {'batch_sampler': bsamp, 'num_workers': eff_w, 'collate_fn': collate_fn, 'pin_memory': True}
    if eff_w > 0:
        kw.update(multiprocessing_context='fork', worker_init_fn=init_worker, prefetch_factor=2)
    return DataLoader(ds, **kw)


@torch.no_grad()
def _eval_ieee_fp32(model: TrackletIPFormer, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    prev_mm = torch.backends.cuda.matmul.allow_tf32
    prev_cd = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model.eval()
    probs_l, pv_l, sc_l, sv_l = [], [], [], []
    try:
        for batch in loader:
            db = _to_device_batch(batch, device)
            clean_db = {
                k: v for k, v in db.items()
                if k not in ('label', 'label_7', 'tau_branch_valid', 'charm_flag', 'charm_valid', 'vertex_xyz_mm', 'vertex_valid', 'vis_kinematics', 'kin_valid', 'sample_weight')
            }
            with torch.amp.autocast('cuda', enabled=False):
                out = model(clean_db, return_dataclass=True)
            probs_l.append(out.probs4.detach().float().cpu().numpy())
            pv_l.append(out.v_pv_mm.detach().float().cpu().numpy())
            sc_l.append(out.scale_pv_mm.detach().float().cpu().numpy())
            sv_l.append((out.sv_disp_mm if out.sv_disp_mm is not None else out.v_pv_mm.new_zeros((len(out.probs4), 3))).detach().float().cpu().numpy())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_mm
        torch.backends.cudnn.allow_tf32 = prev_cd
    return {
        'probs': np.concatenate(probs_l, axis=0).astype(np.float32),
        'pv_pred': np.concatenate(pv_l, axis=0).astype(np.float32),
        'pv_scale': np.concatenate(sc_l, axis=0).astype(np.float32),
        'sv_disp': np.concatenate(sv_l, axis=0).astype(np.float32),
    }


def _extract_ood_worker(task: tuple) -> list[dict]:
    obs_slice, start_idx, geom, bounds, cfg = task
    out = []
    for loc, ob in enumerate(obs_slice):
        dom = compute_observable_containment(ob, geom, bounds, config=cfg)
        out.append(extract_full_1mm_sensor_item(start_idx + loc, ob, None, dom, geom, bounds, cfg))
    return out


def _eval_ood_fp32(
    model: TrackletIPFormer,
    ood_root: Path,
    geometry: dict,
    bounds: dict,
    cfg: dict[str, Any],
    collate_fn: Any,
    device: torch.device,
    workers: int = 4,
) -> np.ndarray:
    obs_p = ood_root / 'shards/000/observations.parquet'
    obs_list = pq.read_table(obs_p).to_pylist()
    if workers > 1 and len(obs_list) >= 200:
        chunk = max(50, len(obs_list) // workers)
        tasks = [(obs_list[i:i + chunk], i, geometry, bounds, cfg) for i in range(0, len(obs_list), chunk)]
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks)), mp_context=mp.get_context('spawn')) as pool:
            parts = list(pool.map(_extract_ood_worker, tasks))
        items = [it for p in parts for it in p]
    else:
        items = _extract_ood_worker((obs_list, 0, geometry, bounds, cfg))
    loader = DataLoader(items, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate_fn)
    return _eval_ieee_fp32(model, loader, device)['probs']


def train_q13_full_1mm(
    out_dir: Path,
    cache_dir: Path,
    *,
    workers: int = 4,
    resume: bool = False,
) -> dict[str, Any]:
    _configure_runtime()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for abl_full_1mm_sensor training')
    device = torch.device('cuda:0')
    cap = torch.cuda.get_device_capability(0)
    cfg = get_q13_config()

    exp_dir = out_dir / 'abl_full_1mm_sensor'
    exp_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = exp_dir / 'manifest.json'
    if resume and manifest_path.is_file():
        man = json.loads(manifest_path.read_text())
        if man.get('status') == 'complete':
            return man

    if not (cache_dir / 'READY.json').is_file():
        prepare_stage1_cache(cache_dir, workers=max(4, workers))

    seed = int(cfg['seed'])
    torch.manual_seed(seed)
    np.random.seed(seed)

    geometry = json.loads((CLEAN_V3_ROOT / 'geometry.json').read_text())
    bounds = detector_bounds(geometry)
    splits = _load_stage1_split_keys()
    collate_fn = lambda evs: collate_tracklet_pf(evs, mirror_rng=None, y_scint_center=bounds['y_scint_center'], n_bars=bounds['n_bars'])

    shm_free = shutil.disk_usage('/dev/shm').free if Path('/dev/shm').exists() else 0
    scratch_base = '/dev/shm' if shm_free >= 6 * (1024 ** 3) else os.environ.get('PINPOINT_V3_SCRATCH', '/tmp')
    Path(scratch_base).mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    utc_started = datetime.now(timezone.utc).isoformat()
    with tempfile.TemporaryDirectory(prefix='q13-1mm-', dir=scratch_base) as tmp_str:
        tmp_dir = Path(tmp_str)
        t_prep0 = time.perf_counter()
        for sp_name in ('train', 'dev_screen', 'val_eval'):
            shutil.copytree(cache_dir / f'mapped_{sp_name}', tmp_dir / f'mapped_{sp_name}')
        sc_all = pd.read_parquet(CLEAN_V3_ROOT / 'truth_sidecar/event_sidecar.parquet')
        tr_pos = pd.MultiIndex.from_frame(sc_all[KEY]).get_indexer(pd.MultiIndex.from_frame(splits['train'][KEY]))
        assert (tr_pos >= 0).all()
        tr_sidecar = sc_all.iloc[tr_pos].reset_index(drop=True)
        prep_seconds = float(time.perf_counter() - t_prep0)

        train_ds = _Stage1MappedDataset(tmp_dir / 'mapped_train', sidecar_df=tr_sidecar)
        dev_ds = _Stage1MappedDataset(tmp_dir / 'mapped_dev_screen')
        val_ds = _Stage1MappedDataset(tmp_dir / 'mapped_val_eval')

        model = TrackletIPFormer(geometry, cfg).to(device)
        ema_model = copy.deepcopy(model).eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)

        micro_bs = int(cfg['micro_bs'])
        grad_accum = int(cfg['grad_accum'])
        eff_bs = int(cfg['batch_size'])
        passes = int(cfg['passes'])
        base_lr = float(cfg['learning_rate'])
        wd = float(cfg['weight_decay'])
        grad_clip = float(cfg['grad_clip'])
        ema_decay = float(cfg['ema_decay'])

        opt = torch.optim.AdamW(model.build_param_groups(base_lr, wd), lr=base_lr, weight_decay=wd, betas=(0.9, 0.98))
        steps_per_pass = math.ceil(len(train_ds) / eff_bs)
        total_steps = passes * steps_per_pass
        warmup_steps = max(1, int(float(cfg['warmup_fraction']) * total_steps))
        min_lr_ratio = float(cfg['min_lr']) / base_lr

        def _lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * min(1.0, prog)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
        dev_loader = _make_loader(dev_ds, batch_size=64, shuffle=False, seed=seed, collate_fn=collate_fn, workers=min(2, workers))
        val_loader = _make_loader(val_ds, batch_size=64, shuffle=False, seed=seed, collate_fn=collate_fn, workers=min(2, workers))
        y_dev = splits['dev_screen']['label'].to_numpy(dtype=np.int64)

        amp_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
        amp_ok = cap[0] >= 7
        scaler = torch.amp.GradScaler('cuda', enabled=(amp_ok and amp_dtype == torch.float16))

        best_nll = float('inf')
        best_pass = 0
        global_step = 0
        history: list[dict[str, Any]] = []
        best_pt = exp_dir / 'best.pt'
        ema_pt = exp_dir / 'ema.pt'
        resume_pt = exp_dir / 'resume.pt'
        start_pass = 1

        if resume and resume_pt.is_file():
            ck = torch.load(resume_pt, map_location=device, weights_only=False)
            model.load_state_dict(ck['model'])
            ema_model.load_state_dict(ck['ema'])
            opt.load_state_dict(ck['opt'])
            scheduler.load_state_dict(ck['scheduler'])
            start_pass = int(ck['pass']) + 1
            global_step = int(ck['global_step'])
            best_nll = float(ck['best_nll'])
            best_pass = int(ck['best_pass'])
            history = list(ck['history'])

        torch.cuda.reset_peak_memory_stats(device)
        try:
            for p_idx in range(start_pass, passes + 1):
                tp0 = time.perf_counter()
                tr_loader = _make_loader(train_ds, batch_size=micro_bs, shuffle=True, seed=seed + p_idx * 101, collate_fn=collate_fn, workers=workers)
                model.train()
                loss_sum = 0.0
                n_seen = 0

                def _step_window(win_buf: list[dict]) -> None:
                    nonlocal global_step, loss_sum, n_seen
                    global_step += 1
                    opt.zero_grad(set_to_none=True)
                    tot_b = sum(len(mb['label']) for mb in win_buf)
                    fd = compute_window_task_denominators(win_buf, pos_weight=3.0, weight_all_tasks=False)
                    win_loss = 0.0
                    for mb in win_buf:
                        db = _to_device_batch(mb, device)
                        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_ok):
                            out = model(db, return_dataclass=True)
                            loss, _ = tracklet_ipformer_loss(out, db, fixed_denoms=fd)
                        if scaler.is_enabled():
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                        win_loss += float(loss.detach())
                    if scaler.is_enabled():
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        scaler.step(opt)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        opt.step()
                    scheduler.step()
                    with torch.no_grad():
                        for pe, pm in zip(ema_model.parameters(), model.parameters()):
                            pe.data.mul_(ema_decay).add_(pm.data, alpha=1.0 - ema_decay)
                        for be, bm in zip(ema_model.buffers(), model.buffers()):
                            be.copy_(bm)
                    loss_sum += win_loss * tot_b
                    n_seen += tot_b

                try:
                    micro_buf: list[dict] = []
                    for batch in tr_loader:
                        micro_buf.append(batch)
                        if len(micro_buf) >= grad_accum:
                            _step_window(micro_buf)
                            micro_buf.clear()
                    if micro_buf:
                        _step_window(micro_buf)
                        micro_buf.clear()
                finally:
                    shutdown_loader(tr_loader)

                tr_loss = loss_sum / max(n_seen, 1)
                row_log: dict[str, Any] = {
                    'pass': p_idx,
                    'step': global_step,
                    'lr': float(scheduler.get_last_lr()[0]),
                    'train_loss': float(tr_loss),
                    'seconds': float(time.perf_counter() - tp0),
                }
                if p_idx == 1 or (p_idx % 2 == 0) or p_idx == passes:
                    dev_pred = _eval_ieee_fp32(ema_model, dev_loader, device)
                    m_dev = _ba4_and_nll(y_dev, dev_pred['probs'])
                    row_log.update({'dev_nll': m_dev['NLL'], 'dev_ba4': m_dev['BA4']})
                    if m_dev['NLL'] < best_nll:
                        best_nll = m_dev['NLL']
                        best_pass = p_idx
                        loc_b = tmp_dir / 'best.local.pt'
                        torch.save({'model': ema_model.state_dict(), 'pass': p_idx, 'config': cfg}, loc_b)
                        shutil.copyfile(loc_b, best_pt)
                        loc_b.unlink(missing_ok=True)
                    print(
                        f"[abl_full_1mm_sensor] pass={p_idx}/{passes} step={global_step} train_loss={tr_loss:.4f} "
                        f"dev_nll={m_dev['NLL']:.4f} dev_ba4={m_dev['BA4']:.4f} ({row_log['seconds']:.1f}s)",
                        flush=True,
                    )
                history.append(row_log)
                if p_idx % 5 == 0 or p_idx == passes:
                    loc_r = tmp_dir / 'resume.local.pt'
                    torch.save({
                        'pass': p_idx, 'global_step': global_step,
                        'model': model.state_dict(), 'ema': ema_model.state_dict(),
                        'opt': opt.state_dict(), 'scheduler': scheduler.state_dict(),
                        'best_nll': best_nll, 'best_pass': best_pass, 'history': history,
                    }, loc_r)
                    shutil.copyfile(loc_r, resume_pt)
                    loc_r.unlink(missing_ok=True)

            loc_e = tmp_dir / 'ema.local.pt'
            torch.save({'model': ema_model.state_dict(), 'pass': passes, 'config': cfg}, loc_e)
            shutil.copyfile(loc_e, ema_pt)
            loc_e.unlink(missing_ok=True)

            best_state = torch.load(best_pt, map_location=device, weights_only=False)
            ema_model.load_state_dict(best_state['model'])
            sample_val_b = _to_device_batch(next(iter(val_loader)), device)
            ti_val = verify_inference_truth_invariance(ema_model, sample_val_b, atol=1e-7)

            dev_out = _eval_ieee_fp32(ema_model, dev_loader, device)
            val_out = _eval_ieee_fp32(ema_model, val_loader, device)
            np.save(exp_dir / 'probs_dev_screen.npy', dev_out['probs'])
            np.save(exp_dir / 'probs_internal_val_eval.npy', val_out['probs'])
            np.save(exp_dir / 'pv_pred_internal_val_eval.npy', val_out['pv_pred'])
            np.save(exp_dir / 'pv_scale_internal_val_eval.npy', val_out['pv_scale'])
            np.save(exp_dir / 'sv_disp_internal_val_eval.npy', val_out['sv_disp'])

            ood_w184 = _eval_ood_fp32(ema_model, OOD_W184_NC_CHARM, geometry, bounds, cfg, collate_fn, device, workers=workers)
            ood_r10085 = _eval_ood_fp32(ema_model, OOD_RUN10085_10088, geometry, bounds, cfg, collate_fn, device, workers=workers)
            np.save(exp_dir / 'probs_ood_w184_epps21_nc_charm.npy', ood_w184)
            np.save(exp_dir / 'probs_ood_run10085_10088.npy', ood_r10085)

            for sp_name, df_meta, out_dict, fname in (
                ('dev_screen', splits['dev_screen'], dev_out, 'dev_screen_predictions.parquet'),
                ('val_eval', splits['val_eval'], val_out, 'val_eval_predictions.parquet'),
            ):
                df_p = df_meta[KEY + FAMILY + ['label', 'obs_domain_id', 'region_id']].copy().reset_index(drop=True)
                pr = out_dict['probs']
                for c in range(4):
                    df_p[f'p_{c}'] = pr[:, c].astype(np.float32)
                for ia, ax in enumerate(('x', 'y', 'z')):
                    df_p[f'v_hat_{ax}_mm'] = out_dict['pv_pred'][:, ia].astype(np.float32)
                    df_p[f'scale_{ax}_mm'] = out_dict['pv_scale'][:, ia].astype(np.float32)
                df_p.to_parquet(exp_dir / fname, index=False, compression=COMPRESSION)
        finally:
            shutdown_loader(dev_loader)
            shutdown_loader(val_loader)

    peak_vram_mib = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
    vram_total_mib = float(torch.cuda.get_device_properties(device).total_memory / (1024 ** 2))
    manifest = {
        'status': 'complete',
        'variant': 'abl_full_1mm_sensor',
        'family': 'tracklet_ipformer',
        'UTC_started': utc_started,
        'UTC_finished': datetime.now(timezone.utc).isoformat(),
        'GPU': torch.cuda.get_device_name(device),
        'GPU_VRAM_MiB': round(vram_total_mib, 1),
        'peak_allocated_VRAM_MiB': round(peak_vram_mib, 1),
        'parameters': int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        'train_events': len(splits['train']),
        'dev_screen_events': len(splits['dev_screen']),
        'val_eval_events': len(splits['val_eval']),
        'effective_batch_size': eff_bs,
        'micro_batch_size': micro_bs,
        'grad_accum': grad_accum,
        'total_optimizer_steps': global_step,
        'best_pass': best_pass,
        'best_dev_screen_nll': best_nll,
        'truth_invariance_val_max_diff': float(ti_val['max_abs_diff']),
        'eval_precision': 'ieee_fp32',
        'prep_seconds': prep_seconds,
        'scratch_base': scratch_base,
        'wall_seconds': float(time.perf_counter() - t_start),
        'config': cfg,
        'history': history,
        'tune8k_or_sealed_touched': False,
    }
    write_json(manifest_path, manifest)
    if resume_pt.is_file():
        resume_pt.unlink()
    return manifest


def aggregate_q13_comparison(out_dir: Path) -> dict[str, Any]:
    """3-way Q13 comparison: native 20um vs centroid-only 1mm vs full centroid+patch+moment 1mm."""
    splits = _load_stage1_split_keys()
    val_df = splits['val_eval']
    labels = val_df['label'].to_numpy(dtype=np.int64)
    obs_dom = val_df['obs_domain_id'].to_numpy(dtype=np.int64)
    reg_id = val_df['region_id'].to_numpy(dtype=np.int64)
    fam_codes = pd.MultiIndex.from_frame(val_df[FAMILY]).factorize()[0]

    sc_all = pd.read_parquet(CLEAN_V3_ROOT / 'truth_sidecar/event_sidecar.parquet')
    pos = pd.MultiIndex.from_frame(sc_all[KEY]).get_indexer(pd.MultiIndex.from_frame(val_df[KEY]))
    sc_val = sc_all.iloc[pos].reset_index(drop=True)
    v_true = sc_val[['vertex_x_mm', 'vertex_y_mm', 'vertex_z_mm']].to_numpy(dtype=np.float64)
    v_valid = sc_val['vertex_valid'].to_numpy(dtype=np.int64) > 0

    slices = {
        'All': np.ones(len(val_df), dtype=bool),
        'D1_pix_contained': obs_dom == 1,
        'D2_scint_contained': obs_dom == 2,
        'Pinpoint_PixXY': (reg_id == 0) | (reg_id == 1),
        'Front30': reg_id == 0,
        'Interm150': reg_id == 1,
    }

    ood_r10085_y = pd.read_parquet(OOD_RUN10085_10088 / 'truth_meta.parquet')['label'].to_numpy(dtype=np.int64)
    sources = {
        'native_20um_s1_tracklet_ipformer': TIP_EXP_ROOT / 's1_tracklet_ipformer',
        'centroid_only_1mm_abl_1mm_quant': TIP_EXP_ROOT / 'abl_1mm_quant',
        'full_1mm_abl_full_1mm_sensor': out_dir / 'abl_full_1mm_sensor',
    }

    runs: dict[str, Any] = {}
    probs_map: dict[str, np.ndarray] = {}
    for label_name, rdir in sources.items():
        if not (rdir / 'probs_internal_val_eval.npy').is_file():
            continue
        p_val = np.load(rdir / 'probs_internal_val_eval.npy').astype(np.float64)
        probs_map[label_name] = p_val
        slice_metrics = {sn: _ba4_and_nll(labels[mk], p_val[mk]) for sn, mk in slices.items()}

        tau_scores = np.log(np.clip(p_val[:, 2], 1e-8, 1.0)) - np.log(np.clip(1.0 - p_val[:, 2], 1e-8, 1.0))
        within_1pct = {}
        for sn in ('All', 'D1_pix_contained', 'Pinpoint_PixXY', 'Front30', 'Interm150'):
            mk = slices[sn]
            wp = _tau_efficiency_at_fpr(tau_scores[mk], labels[mk], 0.01)
            within_1pct[sn] = {
                'tau_eff': float(wp['tau_eff']),
                'actual_fpr': float(wp['actual_fpr']),
                'tau_selected': int(wp['tau_selected']),
                'tau_total': int(wp['tau_total']),
                'bkg_selected': int(wp['bkg_selected']),
                'bkg_total': int(wp['bkg_total']),
            }
        split_half = {
            sn: _split_half_tail_eval(tau_scores, labels, val_df, slices[sn], fpr_targets=(0.01,))
            for sn in ('All', 'D1_pix_contained', 'Pinpoint_PixXY', 'Front30')
        }

        pv_audit: dict[str, Any] = {}
        if (rdir / 'pv_pred_internal_val_eval.npy').is_file():
            vp = np.load(rdir / 'pv_pred_internal_val_eval.npy').astype(np.float64)
            vsc = np.load(rdir / 'pv_scale_internal_val_eval.npy').astype(np.float64)
            for sn in ('All', 'Front30', 'Interm150'):
                rmk = v_valid & slices[sn]
                dz = np.abs(vp[rmk, 2] - v_true[rmk, 2])
                d3 = np.linalg.norm(vp[rmk] - v_true[rmk], axis=1)
                pulls = compute_student_t_pulls(
                    torch.from_numpy(vp[rmk]),
                    torch.from_numpy(vsc[rmk]),
                    torch.from_numpy(v_true[rmk]),
                    torch.ones(int(rmk.sum()), dtype=torch.int64),
                    df=3.0,
                )
                pv_audit[sn] = {
                    'count': int(rmk.sum()),
                    'median_abs_dz_mm': float(np.median(dz)),
                    'median_3d_mm': float(np.median(d3)),
                    'p68_abs_pull': pulls['p68_abs_pull'],
                }

        ood_metrics: dict[str, Any] = {}
        w184_p = rdir / 'probs_ood_w184_epps21_nc_charm.npy'
        if w184_p.is_file():
            pw = np.load(w184_p).astype(np.float64)
            ood_metrics['gen_ood_w184_epps21_nc_charm'] = {
                'n_events': int(len(pw)),
                'pred_tau_argmax_rate': float((pw.argmax(axis=1) == 2).mean()),
                'p_tau_gt_0p5_rate': float((pw[:, 2] > 0.5).mean()),
                'mean_p_tau': float(pw[:, 2].mean()),
            }
        r85_p = rdir / 'probs_ood_run10085_10088.npy'
        if r85_p.is_file():
            pr85 = np.load(r85_p).astype(np.float64)
            ood_metrics['gen_ood_run10085_10088'] = {'n_events': int(len(pr85)), **_ba4_and_nll(ood_r10085_y, pr85)}

        man = json.loads((rdir / 'manifest.json').read_text()) if (rdir / 'manifest.json').is_file() else {}
        runs[label_name] = {
            'best_pass': man.get('best_pass'),
            'best_dev_screen_nll': man.get('best_dev_screen_nll'),
            'slices': slice_metrics,
            'within_slice_1pct_fpr': within_1pct,
            'split_half_1pct_fpr': split_half,
            'pv_audit': pv_audit,
            'ood': ood_metrics,
        }

    paired_boot: dict[str, Any] = {}
    pairs = (
        ('native_20um_s1_tracklet_ipformer', 'centroid_only_1mm_abl_1mm_quant'),
        ('native_20um_s1_tracklet_ipformer', 'full_1mm_abl_full_1mm_sensor'),
        ('centroid_only_1mm_abl_1mm_quant', 'full_1mm_abl_full_1mm_sensor'),
    )
    for a_name, b_name in pairs:
        if a_name not in probs_map or b_name not in probs_map:
            continue
        pa, pb = probs_map[a_name], probs_map[b_name]
        paired_boot[f'{a_name} - {b_name}'] = {
            sn: _paired_family_bootstrap_delta(
                pa[:, 2], pb[:, 2], pa, pb, labels, fam_codes, slices[sn], n_boot=1000, seed=20263250
            )
            for sn in ('All', 'D1_pix_contained', 'Pinpoint_PixXY', 'Front30', 'Interm150')
        }

    rep = {
        'UTC': datetime.now(timezone.utc).isoformat(),
        'stage': 'Q13 Stage-I Internal-Val 3-Way Sensor Comparison (native 20um vs centroid-only 1mm vs full 1mm)',
        'val_eval_events': int(len(val_df)),
        'seal_violations': list(_STAGE1_VIOLATIONS),
        'runs': runs,
        'paired_family_bootstrap_seed20263250_n1000': paired_boot,
    }
    write_json(out_dir / 'q13_full_1mm_comparison.json', rep)
    return rep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('prepare_cache', 'prepare_cache_part', 'merge_cache', 'train', 'aggregate'), required=True)
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument('--cache-dir', type=Path, default=None)
    parser.add_argument('--part', type=int, default=0)
    parser.add_argument('--n-parts', type=int, default=8)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    with install_stage1_seal_guard() as guard:
        cdir = args.cache_dir or (args.out_dir / '_stage1_item_cache')
        if args.mode == 'prepare_cache':
            res = prepare_stage1_cache(cdir, workers=args.workers)
        elif args.mode == 'prepare_cache_part':
            res = prepare_cache_part(cdir, args.part, args.n_parts, workers=args.workers)
        elif args.mode == 'merge_cache':
            res = merge_cache(cdir, args.n_parts)
        elif args.mode == 'train':
            res = train_q13_full_1mm(args.out_dir, cdir, workers=args.workers, resume=args.resume)
        elif args.mode == 'aggregate':
            res = aggregate_q13_comparison(args.out_dir)
        else:
            raise ValueError(args.mode)

        if guard.violations or _STAGE1_VIOLATIONS:
            raise SealViolationError(f'Audit violations detected: {guard.violations + _STAGE1_VIOLATIONS}')
        print(json.dumps({k: v for k, v in res.items() if k not in ('history', 'runs', 'paired_family_bootstrap_seed20263250_n1000')}, indent=2))


if __name__ == '__main__':
    main()
