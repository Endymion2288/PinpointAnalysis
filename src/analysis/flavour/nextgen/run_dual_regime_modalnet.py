"""Phase 2-4 Runner for Proposal D (DualRegimeModalNet):
  - gate-ab: Local Gate A/B, truth-invariance, symmetry, empty-modality & grad_accum equivalence.
  - train-neural: Matched-budget Stage-I training (Internal-Train 25.6k -> DevScreen 1.6k -> ValEval 4.8k)
    + Open W OOD + 7 stress perturbations on 400-event Internal-Val slice (ZERO Tune6k/sealed access).
  - train-sec13: Section 13 Upstream-Reco-vs-Classifier controls (s1_bdt_70d, s1_improved_reco_bdt, s1_improved_reco_mlp).
  - aggregate: Paired family-bootstrap CIs (seed=20263250), T4->T3 Collaboration slice, PV pull & JSON summary.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import shutil
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import beta as scipy_beta
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler

from analysis.flavour.inventory import write_json
from analysis.flavour.nextgen.dual_regime_modalnet import (
    DualRegimeModalNet,
    N_SPEC_CAND_MAX,
    apply_y_mirror_dual_regime_item,
    collate_dual_regime,
    dual_regime_branch_groups,
    extract_dual_regime_item,
    extract_nonexclusive_spec_candidates,
)
from analysis.flavour.nextgen.protocols import (
    compute_student_t_pulls,
    conditional_three_from_four,
    student_t_vertex_loss,
)
from analysis.flavour.nextgen.seal_guard import (
    SealAuditGuard,
    assert_unsealed_path,
    assert_unsealed_split,
    load_authorized_split_shards,
    verify_inference_truth_invariance,
)
from analysis.flavour.nextgen.tracklet_ops import (
    compute_ipt_spectrometer_interlock,
    compute_scint_tof_residual_ns,
    pairwise_skew_dca_features,
    _solve_sym_3x3,
)
from analysis.flavour.v2.partitions import FAMILY, KEY
from analysis.flavour.v2.r1.metrics import Metrics, three_from_four, three_targets
from analysis.flavour.v2.r3.mapped_inputs import MappedItems, write_mapped_items
from analysis.flavour.v3.common.calibration import _tau_efficiency_at_fpr
from analysis.flavour.v3.common.datasets import V3EvalDataset, V3TrainDataset
from analysis.flavour.v3.common.domains import (
    OBS_DOMAIN_NAMES,
    calibrated_bar_channels,
    compute_observable_containment,
    cumulative_radiation_length,
    detector_bounds,
)
from analysis.flavour.v3.v2_pinpointformer.pixel_tokenizer import extract_pf_item
from analysis.flavour.v3.v2_pinpointformer.stereo_bars import static_scint_stations
from analysis.flavour.v3.v2_pinpointformer.train import move_batch_to_device
from analysis.flavour.v3.v3_multitask.losses import focal_charm_bce, marginal_tau_branch_ce
from analysis.flavour.v3.v4_tau.nomad_kinematics import NomadTransverseExtractor
from analysis.flavour.v3.v6_clean_sim.evaluate_stage317 import paired_family_bootstrap_delta


CLEAN_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v6_clean_v3_2026-11-19/clean_sim')
STAGE325_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v7_stage325_2026-11-20')
SPLITS_DIR = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/forensics/splits')
EXP_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/exp_dual_regime')

VARIANT_CONFIGS: dict[str, dict[str, Any]] = {
    's1_dual_regime_modalnet': {
        'use_ipt_spec_interlock': True,
        'use_sinkhorn_tof': True,
        'enable_pixdark_noise_aug': True,
        'homogeneous_regime': False,
    },
    'abl_no_ipt_spec_interlock': {
        'use_ipt_spec_interlock': False,
        'use_sinkhorn_tof': True,
        'enable_pixdark_noise_aug': True,
        'homogeneous_regime': False,
    },
    'abl_no_sinkhorn_tof': {
        'use_ipt_spec_interlock': True,
        'use_sinkhorn_tof': False,
        'enable_pixdark_noise_aug': True,
        'homogeneous_regime': False,
    },
    'abl_no_pixdark_noise_aug': {
        'use_ipt_spec_interlock': True,
        'use_sinkhorn_tof': True,
        'enable_pixdark_noise_aug': False,
        'homogeneous_regime': False,
    },
    'abl_homogeneous_regime': {
        'use_ipt_spec_interlock': True,
        'use_sinkhorn_tof': True,
        'enable_pixdark_noise_aug': True,
        'homogeneous_regime': True,
    },
}

STRESS_SPECS: list[tuple[str, dict[str, float], dict[str, Any]]] = [
    ('nominal', {}, {}),
    ('pixel_dead_2pct', {'pixel_dead': 0.02}, {}),
    ('pixel_dead_5pct', {'pixel_dead': 0.05}, {}),
    ('pixel_dark_1e-5', {'pixel_dark_rate': 1e-5}, {}),
    ('tracker_dead_2pct_noise1', {'tracker_dead': 0.02, 'tracker_noise_per_station': 1.0}, {}),
    ('tracker_dead_5pct_noise2', {'tracker_dead': 0.05, 'tracker_noise_per_station': 2.0}, {}),
    ('sipm_sat_cap_0p95', {}, {'sipm_invert_max_occupancy': 0.95, 'sipm_sat_cap': 0.95}),
]

BASE_OBS_CFG: dict[str, Any] = {
    'pe_only': True,
    'bar_threshold_pe': 2.5,
    'sipm_npix': 14400.0,
    'apply_sipm_saturation': False,
    'invert_sipm_saturation': True,
}


def _set_ieee_fp32(enable: bool = True) -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = not enable
        torch.backends.cudnn.allow_tf32 = not enable


def _identity_collate(xs: list[dict]) -> list[dict]:
    return xs


def _clopper_pearson_ci95(k: int, n: int) -> list[float]:
    if n <= 0:
        return [0.0, 0.0]
    lo = 0.0 if k == 0 else float(scipy_beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(scipy_beta.ppf(0.975, k + 1, n - k))
    return [lo, hi]


def _ba4_and_nll(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probs, dtype=np.float64)
    p = p / np.maximum(p.sum(axis=-1, keepdims=True), 1e-12)
    pred = p.argmax(axis=1)
    recalls = [float(np.mean(pred[y == c] == c)) for c in range(4) if (y == c).any()]
    ba4 = float(np.mean(recalls)) * 100.0 if recalls else 0.0
    nll = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-8, 1.0)).mean())
    return {'n_events': int(len(y)), 'BA4_pct': ba4, 'NLL': nll}


def _split_half_tail_eval(
    scores: np.ndarray,
    labels: np.ndarray,
    keys_df: pd.DataFrame,
    mask: np.ndarray,
    fpr_target: float = 0.01,
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
    s_lock, y_lock = scores[lock_m], labels[lock_m]
    s_held, y_held = scores[held_m], labels[held_m]
    bg_lock = s_lock[y_lock != 2]
    bg_held = s_held[y_held != 2]
    sg_held = s_held[y_held == 2]
    nb_lock, nb_held, ns_held = int(len(bg_lock)), int(len(bg_held)), int(len(sg_held))
    if nb_lock == 0 or nb_held == 0 or ns_held == 0:
        return {'valid': False}
    bg_sorted = np.sort(bg_lock)[::-1]
    k_allow = int(math.floor(fpr_target * nb_lock))
    thr = float(bg_sorted[k_allow]) if k_allow < nb_lock else float(-1e9)
    fp = int(np.sum(bg_held > thr))
    tp = int(np.sum(sg_held > thr))
    return {
        'valid': True,
        'n_lock_bg': nb_lock,
        'lock_allowed_bg': k_allow,
        'held_TP': tp,
        'held_tau': ns_held,
        'held_FP': fp,
        'held_B': nb_held,
        'held_tau_eff_pct': float(tp / ns_held * 100.0),
        'held_tau_eff_ci95_pct': [100.0 * x for x in _clopper_pearson_ci95(tp, ns_held)],
        'held_FPR_pct': float(fp / nb_held * 100.0),
        'low_bg_warning': bool(k_allow < 10),
    }


def _within_slice_tail_eval(scores: np.ndarray, labels: np.ndarray, mask: np.ndarray, fpr_target: float = 0.01) -> dict[str, Any]:
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


def _collaboration_t3_eval(probs4: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    m = np.asarray(mask, dtype=bool) & (labels != 2)
    if not m.any():
        return {'n_events': 0}
    y3 = three_targets(labels[m])
    p3 = three_from_four(probs4[m])
    ba3 = float(Metrics(y3, p3).report()['BA']) * 100.0
    aucs = []
    for c in range(3):
        if (y3 == c).any() and (~(y3 == c)).any():
            aucs.append(float(roc_auc_score((y3 == c).astype(int), p3[:, c])))
        else:
            aucs.append(float('nan'))
    return {'n_events': int(m.sum()), 'BA3_pct': ba3, 'ovr_auc_e_mu_nc': aucs}


def _perturb_observation_for_stress(
    obs: dict,
    bounds: dict,
    rng: np.random.Generator,
    *,
    pixel_dead: float = 0.0,
    pixel_dark_rate: float = 0.0,
    tracker_dead: float = 0.0,
    tracker_noise_per_station: float = 0.0,
) -> dict:
    ob = {
        'pixel': {k: list(v) for k, v in obs['pixel'].items()},
        'bar': {k: list(v) for k, v in obs['bar'].items()},
        'tracker_hypothesis': {k: list(v) for k, v in obs['tracker_hypothesis'].items()},
    }
    if pixel_dead > 0.0 and len(ob['pixel']['channel_id']) > 0:
        keep = rng.random(len(ob['pixel']['channel_id'])) >= pixel_dead
        ob['pixel']['channel_id'] = [x for x, m in zip(ob['pixel']['channel_id'], keep) if m]
        ob['pixel']['xyz_mm'] = [x for x, m in zip(ob['pixel']['xyz_mm'], keep) if m]
    if pixel_dark_rate > 0.0:
        nx, ny, nz = int(bounds['nx']), int(bounds['ny']), int(bounds['nz'])
        mean_per_plane = pixel_dark_rate * float(nx * ny)
        px_min, py_min = float(bounds['px_min']), float(bounds['py_min'])
        dx, dy = float(bounds['dx_mm']), float(bounds['dy_mm'])
        pzs = bounds['pixel_Zpos']
        cids = ob['pixel']['channel_id']
        xyzs = ob['pixel']['xyz_mm']
        for p in range(nz):
            nf = int(rng.poisson(mean_per_plane))
            if nf <= 0:
                continue
            cs = rng.integers(0, nx, size=nf)
            rs = rng.integers(0, ny, size=nf)
            for c, r in zip(cs, rs):
                cids.append(int((p * ny + int(r)) * nx + int(c)))
                xyzs.append([px_min + (int(c) + 0.5) * dx, py_min + (int(r) + 0.5) * dy, float(pzs[p])])
    if tracker_dead > 0.0 and len(ob['tracker_hypothesis']['station']) > 0:
        keep = rng.random(len(ob['tracker_hypothesis']['station'])) >= tracker_dead
        for k in ('station', 'xy_mm', 'sigma_xy_mm'):
            if k in ob['tracker_hypothesis']:
                ob['tracker_hypothesis'][k] = [x for x, m in zip(ob['tracker_hypothesis'][k], keep) if m]
    if tracker_noise_per_station > 0.0:
        for st in (0, 1, 2):
            nf = int(rng.poisson(tracker_noise_per_station))
            for _ in range(nf):
                ob['tracker_hypothesis']['station'].append(st)
                ob['tracker_hypothesis']['xy_mm'].append([
                    float(rng.uniform(-125.0, 125.0)),
                    float(rng.uniform(-110.0, 140.0)),
                ])
                if 'sigma_xy_mm' in ob['tracker_hypothesis']:
                    ob['tracker_hypothesis']['sigma_xy_mm'].append([0.58, 0.017])
    return ob


def stage_authorized_train_shards(data_root: Path, dst_root: Path) -> Path:
    """Copy ONLY authorized Stage-1 `train` shards (`000..079`) to fast local scratch."""
    assert_unsealed_split('train', stage=1)
    shard_paths = load_authorized_split_shards(data_root, 'train', stage=1)
    dst_shards = dst_root / 'shards'
    dst_shards.mkdir(parents=True, exist_ok=True)
    for sp in shard_paths:
        assert_unsealed_path(sp)
        out_sp = dst_shards / sp.name
        if not out_sp.exists():
            shutil.copytree(sp, out_sp)
    for fname in ('plan.json', 'split_shards.json'):
        src_f = assert_unsealed_path(data_root / fname)
        if src_f.is_file() and not (dst_root / fname).exists():
            shutil.copy2(src_f, dst_root / fname)
    return dst_root


def _extract_shard_worker(task: tuple) -> list[dict]:
    shard_str, geom, bounds, cfg, keys_filter_set = task
    shard_p = Path(assert_unsealed_path(shard_str))
    prov = pd.read_parquet(shard_p / 'provenance.parquet', columns=KEY + FAMILY + ['split', 'label'])
    obs_list = pq.read_table(shard_p / 'observations.parquet').to_pylist()
    items = []
    for li, row in enumerate(prov.itertuples(index=False)):
        k = (int(row.run), int(row.chunk), int(row.row))
        if row.split != 'train':
            continue
        if keys_filter_set is not None and k not in keys_filter_set:
            continue
        ob = obs_list[li]
        dm = compute_observable_containment(ob, geom, bounds, config=cfg)
        it = extract_dual_regime_item(li, ob, int(row.label), dm, geom, bounds, cfg)
        it['run'] = int(row.run)
        it['chunk'] = int(row.chunk)
        it['row'] = int(row.row)
        it['family_hi'] = int(row.family_hi)
        it['family_lo'] = int(row.family_lo)
        items.append(it)
    return items


class InMemoryDualDataset(Dataset):
    def __init__(self, items: list[dict], event_sidecar: pd.DataFrame | None = None) -> None:
        self.items = items
        self.event_sidecar = event_sidecar
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


def _align_items_to_keys(items: list[dict], keys_df: pd.DataFrame) -> list[dict]:
    lookup = {(int(it['run']), int(it['chunk']), int(it['row'])): it for it in items}
    aligned = []
    for r, c, rw in zip(keys_df['run'], keys_df['chunk'], keys_df['row']):
        k = (int(r), int(c), int(rw))
        if k not in lookup:
            raise KeyError(f'Missing split key {k}')
        aligned.append(lookup[k])
    return aligned


def compute_dual_window_denominators(micro_batches: list[dict], pos_weight: float = 3.0) -> dict[str, float]:
    """Exact full-window denominators across microbatches ensuring grad_accum equivalence (< 1e-4)."""
    d_proc = 0.0
    d_charm = 0.0
    d_vtx = 0.0
    d_vtx_front = 0.0
    d_vtx_mid = 0.0
    d_kin = 0.0
    for mb in micro_batches:
        valid_p = (mb['label_7'] >= 0) | ((mb['label_7'] < 0) & (mb['label'] == 2))
        d_proc += float(valid_p.float().sum().item())
        valid_c = mb['charm_valid'] > 0
        if valid_c.any():
            yc = mb['charm_flag'][valid_c].float()
            d_charm += float(torch.where(yc > 0.5, float(pos_weight), 1.0).sum().item())
        valid_v = mb['vertex_valid'] > 0
        d_vtx += float(valid_v.float().sum().item())
        if valid_v.any():
            vz = mb['vertex_xyz_mm'][:, 2]
            d_vtx_front += float((valid_v & (vz < -2750.0)).float().sum().item())
            d_vtx_mid += float((valid_v & (vz >= -2750.0) & (vz <= -1800.0)).float().sum().item())
        d_kin += float((mb['kin_valid'] > 0).float().sum().item())
    return {
        'proc': max(d_proc, 1e-8),
        'charm': max(d_charm, 1e-8),
        'vtx': max(d_vtx, 1e-8),
        'vtx_front': max(d_vtx_front, 1e-8),
        'vtx_mid': max(d_vtx_mid, 1e-8),
        'kin': max(d_kin, 1e-8),
    }


def compute_dual_regime_loss(
    model: DualRegimeModalNet,
    batch: dict[str, torch.Tensor],
    fixed_denoms: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Multitask Stage-1 loss: 7-class marginal CE + focal Charm + Student-t PV + smooth-L1 kinematics."""
    fd = fixed_denoms or {}
    out = model.forward_NextGen(batch)

    l_proc = marginal_tau_branch_ce(
        out.logits7, batch['label_7'], batch['label'],
        denom_override=fd.get('proc'),
    )
    l_charm = focal_charm_bce(
        out.charm_logit, batch['charm_flag'], batch['charm_valid'],
        denom_override=fd.get('charm'),
    )
    l_vtx = student_t_vertex_loss(
        out.v_pv_mm, out.scale_pv_mm, batch['vertex_xyz_mm'], batch['vertex_valid'],
        df=3.0, denom_override=fd.get('vtx'),
    )

    # Regime-specific auxiliary PV supervision (Front vs Mid) when non-homogeneous
    l_vtx_reg = l_proc.new_zeros(())
    if not model.homogeneous_regime and 'v_front_mm' in out.aux and 'v_mid_mm' in out.aux:
        vv = batch['vertex_valid'] > 0
        vz = batch['vertex_xyz_mm'][:, 2]
        m_front = vv & (vz < -2750.0)
        m_mid = vv & (vz >= -2750.0) & (vz <= -1800.0)
        if m_front.any():
            vf = out.aux['v_front_mm']
            pf = out.aux['P_front'].clamp_min(1e-4)
            sf = torch.rsqrt(pf).clamp(0.02, 150.0)
            l_vtx_reg = l_vtx_reg + 0.05 * student_t_vertex_loss(
                vf, sf, batch['vertex_xyz_mm'], m_front.long(),
                df=3.0, denom_override=fd.get('vtx_front'),
            )
        if m_mid.any():
            vm = out.aux['v_mid_mm']
            pm = out.aux['P_mid'].clamp_min(1e-4)
            sm = torch.rsqrt(pm).clamp(0.02, 150.0)
            l_vtx_reg = l_vtx_reg + 0.05 * student_t_vertex_loss(
                vm, sm, batch['vertex_xyz_mm'], m_mid.long(),
                df=3.0, denom_override=fd.get('vtx_mid'),
            )

    kin_valid = batch['kin_valid'] > 0
    if kin_valid.any() and out.kin_pred is not None:
        with torch.amp.autocast(device_type=out.logits7.device.type, enabled=False):
            kin_ev = F.smooth_l1_loss(
                out.kin_pred[kin_valid].float(),
                batch['vis_kinematics'][kin_valid].float(),
                reduction='none',
            ).mean(dim=-1)
            dk = torch.as_tensor(fd.get('kin', float(kin_valid.sum())), dtype=torch.float32, device=kin_ev.device).clamp_min(1e-8)
            l_kin = kin_ev.sum() / dk
    else:
        l_kin = l_proc.new_zeros(())

    total = l_proc.float() + 0.25 * l_charm.float() + 0.20 * l_vtx.float() + l_vtx_reg.float() + 0.15 * l_kin.float()
    return total, out.probs4


@torch.no_grad()
def predict_dual_items_fp32(
    model: DualRegimeModalNet,
    items: list[dict],
    bounds: dict,
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, np.ndarray]:
    """Deterministic IEEE-FP32 inference returning probs4, v_pv_mm, scale_pv_mm, charm_prob."""
    _set_ieee_fp32(True)
    model.eval()
    probs_l, v_l, s_l, c_l = [], [], [], []
    yc = float(bounds['y_scint_center'])
    nb = int(bounds['n_bars'])
    for st in range(0, len(items), batch_size):
        sub = items[st : st + batch_size]
        batch = move_batch_to_device(collate_dual_regime(sub, mirror_rng=None, y_scint_center=yc, n_bars=nb), device)
        out = model.forward_NextGen(batch)
        probs_l.append(out.probs4.float().cpu().numpy())
        v_l.append(out.v_pv_mm.float().cpu().numpy())
        s_l.append(out.scale_pv_mm.float().cpu().numpy())
        c_l.append(torch.sigmoid(out.charm_logit.float()).cpu().numpy())
    return {
        'probs': np.concatenate(probs_l, axis=0).astype(np.float32),
        'v_pv_mm': np.concatenate(v_l, axis=0).astype(np.float32),
        'scale_pv_mm': np.concatenate(s_l, axis=0).astype(np.float32),
        'charm_prob': np.concatenate(c_l, axis=0).astype(np.float32),
    }


def _load_ood_observations(ov: str) -> tuple[list[dict], np.ndarray]:
    if 'w184' in ov:
        oroot = STAGE325_ROOT / 'ood_shards' / ov
    else:
        oroot = CLEAN_ROOT / 'ood_shards' / ov
    obs_p = assert_unsealed_path(oroot / 'shards/000/observations.parquet')
    meta_p = assert_unsealed_path(oroot / 'truth_meta.parquet')
    raw_o = pq.read_table(obs_p).to_pylist()
    ometa = pd.read_parquet(meta_p)
    return raw_o, ometa['label'].to_numpy(dtype=np.int64)


def _extract_obs_chunk_worker(task: tuple) -> list[dict]:
    obs_slice, labels_slice, start_idx, geom, bounds, cfg = task
    out = []
    for loc, ob in enumerate(obs_slice):
        lbl = None if labels_slice is None else int(labels_slice[loc])
        dm = compute_observable_containment(ob, geom, bounds, config=cfg)
        out.append(extract_dual_regime_item(start_idx + loc, ob, lbl, dm, geom, bounds, cfg))
    return out


def _parallel_extract_obs_list(
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
        parts = list(pool.map(_extract_obs_chunk_worker, tasks))
    else:
        parts = [_extract_obs_chunk_worker(t) for t in tasks]
    return [it for part in parts for it in part]


def _build_ood_items(
    ov: str,
    geom: dict,
    bounds: dict,
    pool: ProcessPoolExecutor | None = None,
) -> tuple[list[dict], np.ndarray]:
    raw_o, labels = _load_ood_observations(ov)
    items = _parallel_extract_obs_list(raw_o, None, geom, bounds, BASE_OBS_CFG, pool=pool)
    return items, labels


def _build_stress_items_from_internal_val(
    val_keys_400: pd.DataFrame,
    geom: dict,
    bounds: dict,
    pool: ProcessPoolExecutor | None = None,
) -> tuple[dict[str, list[dict]], np.ndarray]:
    """Build 7 stress perturbations on the first 400-event slice of `internal_val_eval` (ZERO Tune6k access!)."""
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
        assert len(pert_obs) == len(raw_obs_400)
        stress_items[st_name] = _parallel_extract_obs_list(pert_obs, labels_400, geom, bounds, cfg_use, pool=pool)
    return stress_items, labels_400


# ===========================================================================
# Mode 1: Gate A/B & Invariance Verification
# ===========================================================================
def run_gate_ab() -> dict[str, Any]:
    from tests.test_nextgen_model_health import _build_32_event_4class_observations, run_gate_a_health_check, run_gate_b_overfit_32_check
    from tests.test_nextgen_seal_and_splits import build_minimal_geometry

    _set_ieee_fp32(True)
    EXP_ROOT.mkdir(parents=True, exist_ok=True)
    geom = build_minimal_geometry()
    bounds = detector_bounds(geom)
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    yc, nb = float(bounds['y_scint_center']), int(bounds['n_bars'])

    obs32, lbl32 = _build_32_event_4class_observations(geom, bounds, seed=325)
    items = [
        extract_dual_regime_item(i, obs32[i], lbl32[i], compute_observable_containment(obs32[i], geom, bounds, BASE_OBS_CFG), geom, bounds, BASE_OBS_CFG)
        for i in range(32)
    ]
    b8 = move_batch_to_device(collate_dual_regime(items[:8], y_scint_center=yc, n_bars=nb), dev)
    b32 = move_batch_to_device(collate_dual_regime(items, y_scint_center=yc, n_bars=nb), dev)

    torch.manual_seed(42)
    model = DualRegimeModalNet(geom, {'dim': 160, 'heads': 8}).to(dev)
    n_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    def _health_loss(out: dict, b: dict) -> torch.Tensor:
        p4 = out['probs4'].clamp(min=1e-6)
        loss = F.nll_loss(torch.log(p4), b['label'])
        v_dummy = torch.zeros_like(out['v_hat'])
        v_valid = torch.ones(len(b['label']), dtype=torch.long, device=dev)
        loss = loss + 0.1 * student_t_vertex_loss(out['v_hat'], out['scale_pv_mm'], v_dummy, v_valid)
        loss = loss + 0.01 * out['charm_logit'].pow(2).mean() + 0.01 * out['kin_pred'].pow(2).mean()
        return loss

    res_a = run_gate_a_health_check(model, b8, _health_loss, expected_logits_shape=(8, 7), branch_groups_fn=dual_regime_branch_groups)
    res_inv = verify_inference_truth_invariance(model, b8)

    model.eval()
    with torch.no_grad():
        p_b8 = model(b8)['probs4']
        b1 = move_batch_to_device(collate_dual_regime(items[:1], y_scint_center=yc, n_bars=nb), dev)
        p_b1 = model(b1)['probs4']
        bs_diff = float((p_b8[:1] - p_b1).abs().max().item())

        # Empty modality check (0 pixel, 0 bar, 0 tracker)
        emp_obs = [dict(obs32[0]), dict(obs32[1])]
        emp_obs[0]['pixel'] = {'channel_id': np.empty(0, dtype=np.int64), 'xyz_mm': np.empty((0, 3), dtype=np.float32)}
        emp_obs[1]['bar'] = {
            'channel_id': np.empty(0, dtype=np.int64), 'layer': np.empty(0, dtype=np.int16),
            'panel': np.empty(0, dtype=np.int16), 'horizontal': np.empty(0, dtype=np.int8),
            'bar': np.empty(0, dtype=np.int16), 'bounds_mm': np.empty((0, 5), dtype=np.float32),
            'pe_proxy': np.empty(0, dtype=np.int64), 'hit_time_ns': np.empty(0, dtype=np.float32),
        }
        emp_obs[1]['tracker_hypothesis'] = {
            'station': np.empty(0, dtype=np.int8), 'xy_mm': np.empty((0, 2), dtype=np.float32),
            'sigma_xy_mm': np.empty((0, 2), dtype=np.float32),
        }
        emp_items = [extract_dual_regime_item(i, ob, i, compute_observable_containment(ob, geom, bounds, BASE_OBS_CFG), geom, bounds, BASE_OBS_CFG) for i, ob in enumerate(emp_obs)]
        emp_b = move_batch_to_device(collate_dual_regime(emp_items, y_scint_center=yc, n_bars=nb), dev)
        emp_out = model(emp_b)
        emp_finite = bool(torch.isfinite(emp_out['logits']).all() and torch.isfinite(emp_out['probs4']).all() and torch.isfinite(emp_out['v_hat']).all())

        # Permutation invariance (pixel hits, scintillator bars, and tracker hits)
        rng_p = np.random.default_rng(7)
        p_idx = rng_p.permutation(len(obs32[0]['pixel']['channel_id']))
        b_idx = rng_p.permutation(len(obs32[0]['bar']['panel']))
        ob_perm = {
            'pixel': {
                'channel_id': np.asarray(obs32[0]['pixel']['channel_id'])[p_idx],
                'xyz_mm': np.asarray(obs32[0]['pixel']['xyz_mm'])[p_idx],
            },
            'bar': {k: np.asarray(v)[b_idx] for k, v in obs32[0]['bar'].items()},
            'tracker_hypothesis': obs32[0]['tracker_hypothesis'],
        }
        if len(obs32[1]['tracker_hypothesis']['station']):
            t_idx = rng_p.permutation(len(obs32[1]['tracker_hypothesis']['station']))
            ob_perm1 = {
                'pixel': obs32[1]['pixel'],
                'bar': obs32[1]['bar'],
                'tracker_hypothesis': {k: np.asarray(v)[t_idx] for k, v in obs32[1]['tracker_hypothesis'].items()},
            }
        else:
            ob_perm1 = obs32[1]
        it_p0 = extract_dual_regime_item(0, ob_perm, 0, compute_observable_containment(ob_perm, geom, bounds, BASE_OBS_CFG), geom, bounds, BASE_OBS_CFG)
        it_p1 = extract_dual_regime_item(1, ob_perm1, 1, compute_observable_containment(ob_perm1, geom, bounds, BASE_OBS_CFG), geom, bounds, BASE_OBS_CFG)
        bp = move_batch_to_device(collate_dual_regime([it_p0, it_p1], y_scint_center=yc, n_bars=nb), dev)
        p_perm = model(bp)['probs4']
        p_orig = model(move_batch_to_device(collate_dual_regime(items[:2], y_scint_center=yc, n_bars=nb), dev))['probs4']
        perm_diff = float((p_perm - p_orig).abs().max().item())

        # Y-mirror symmetry of physical operators (soft tracklets + 4D IPT-Spec interlock + ToF)
        mir_items = [apply_y_mirror_dual_regime_item(it, y_scint_center=yc, n_bars=nb) for it in items[:4]]
        bm = move_batch_to_device(collate_dual_regime(mir_items, y_scint_center=yc, n_bars=nb), dev)
        b4 = move_batch_to_device(collate_dual_regime(items[:4], y_scint_center=yc, n_bars=nb), dev)
        is_ipt_4 = b4['trk_mask'] & (b4['trk_morph'][..., 1] > 0.5)
        is_ipt_m = bm['trk_mask'] & (bm['trk_morph'][..., 1] > 0.5)
        il_orig = compute_ipt_spectrometer_interlock(b4['trk_kin'], is_ipt_4, b4['trk_cand'], b4['trk_cand_mask'])
        il_mir = compute_ipt_spectrometer_interlock(bm['trk_kin'], is_ipt_m, bm['trk_cand'], bm['trk_cand_mask'])
        ymir_diff = float((il_orig - il_mir).abs().max().item())

    # Grad accumulation equivalence check (bs=8 vs 4x2)
    def _attach_dummy_targets(sub_items: list[dict]) -> dict:
        b_d = collate_dual_regime(sub_items, y_scint_center=yc, n_bars=nb)
        n_e = len(sub_items)
        b_d['label_7'] = b_d['label'].clone()
        b_d['charm_flag'] = torch.zeros(n_e, dtype=torch.long)
        b_d['charm_valid'] = torch.ones(n_e, dtype=torch.long)
        cs = b_d['cal_start']
        b_d['vertex_xyz_mm'] = torch.stack([cs[:, 1], cs[:, 2], cs[:, 0]], dim=-1)
        b_d['vertex_valid'] = torch.ones(n_e, dtype=torch.long)
        b_d['vis_kinematics'] = torch.zeros((n_e, 4), dtype=torch.float32)
        b_d['kin_valid'] = torch.ones(n_e, dtype=torch.long)
        return b_d

    b8_t = _attach_dummy_targets(items[:8])
    mb0_t = _attach_dummy_targets(items[:4])
    mb1_t = _attach_dummy_targets(items[4:8])
    for mb in (mb0_t, mb1_t):
        for k, v in list(mb.items()):
            if torch.is_tensor(v) and v.ndim >= 2 and v.shape[1] < b8_t[k].shape[1]:
                ps = list(v.shape)
                ps[1] = b8_t[k].shape[1] - v.shape[1]
                mb[k] = torch.cat([v, torch.zeros(ps, dtype=v.dtype)], dim=1)

    # Grad accumulation equivalence check (bs=8 vs 4x2; verify exact CPU FP32 < 1e-4 & record CUDA FP32)
    torch.manual_seed(99)
    m_full_cpu = DualRegimeModalNet(geom, {'dim': 160, 'heads': 8, 'drop_path': 0.0, 'head_dropout': 0.0, 'degradation_aug_prob': 0.0}).cpu()
    torch.manual_seed(99)
    m_acc_cpu = DualRegimeModalNet(geom, {'dim': 160, 'heads': 8, 'drop_path': 0.0, 'head_dropout': 0.0, 'degradation_aug_prob': 0.0}).cpu()
    for _mdl in (m_full_cpu, m_acc_cpu):
        _mdl.train()
        for _mod in _mdl.modules():
            if isinstance(_mod, torch.nn.Dropout):
                _mod.p = 0.0
            if hasattr(_mod, 'drop_prob'):
                _mod.drop_prob = 0.0
    m_full_cpu.zero_grad(set_to_none=True)
    lf_cpu, _ = compute_dual_regime_loss(m_full_cpu, b8_t)
    lf_cpu.backward()
    g_full_cpu = {n: p.grad.detach().float() for n, p in m_full_cpu.named_parameters() if p.grad is not None}

    m_acc_cpu.zero_grad(set_to_none=True)
    fd = compute_dual_window_denominators([mb0_t, mb1_t])
    la_cpu_sum = 0.0
    for mb in (mb0_t, mb1_t):
        la_c, _ = compute_dual_regime_loss(m_acc_cpu, mb, fixed_denoms=fd)
        la_c.backward()
        la_cpu_sum += float(la_c.item())
    g_acc_cpu = {n: p.grad.detach().float() for n, p in m_acc_cpu.named_parameters() if p.grad is not None}
    gn_full_c = math.sqrt(sum(float(v.square().sum()) for v in g_full_cpu.values()))
    gn_diff_c = math.sqrt(sum(float((g_full_cpu[k] - g_acc_cpu[k]).square().sum()) for k in g_full_cpu))
    grad_rel_diff = float(gn_diff_c / max(gn_full_c, 1e-8))
    lf = lf_cpu
    la_sum = la_cpu_sum
    del m_full_cpu, m_acc_cpu

    # Gate B: 32-event overfit convergence
    torch.manual_seed(42)
    t0 = time.perf_counter()
    model_b = DualRegimeModalNet(geom, {'dim': 160, 'heads': 8}).to(dev)
    res_b = run_gate_b_overfit_32_check(
        model_b, b32,
        lambda out, b: F.nll_loss(torch.log(out['probs4'].clamp(min=1e-6)), b['label']),
        steps=75, lr=2.5e-3, min_acc=31.0 / 32.0, max_final_loss=0.05,
    )
    gate_b_sec = float(time.perf_counter() - t0)

    report = {
        'UTC': datetime.now(timezone.utc).isoformat(),
        'device': str(dev),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        'n_params': n_params,
        'gate_a': res_a,
        'truth_invariance': res_inv,
        'bs1_vs_bs8_max_diff': bs_diff,
        'empty_modality_finite': emp_finite,
        'permutation_max_diff': perm_diff,
        'y_mirror_max_diff': ymir_diff,
        'grad_accum_rel_l2_diff': grad_rel_diff,
        'grad_accum_loss_abs_diff': abs(float(lf.item()) - la_sum),
        'gate_b': {**res_b, 'seconds': gate_b_sec},
        'all_passed': bool(
            res_a['passed']
            and res_inv['passed']
            and bs_diff <= 1e-6
            and emp_finite
            and perm_diff <= 1e-6
            and ymir_diff <= 1e-5
            and grad_rel_diff < 1e-4
            and res_b['passed']
        ),
    }
    write_json(EXP_ROOT / 'gate_ab_results.json', report)
    print(json.dumps(report, indent=2))
    if not report['all_passed']:
        raise RuntimeError('Gate A/B check failed')
    return report


# ===========================================================================
# Mode 2: Stage-I Neural Training & Evaluation
# ===========================================================================
def _update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for pe, pm in zip(ema.parameters(), model.parameters()):
            pe.data.mul_(decay).add_(pm.data, alpha=1.0 - decay)
        for be, bm in zip(ema.buffers(), model.buffers()):
            be.copy_(bm)


def run_train_neural(variant: str) -> dict[str, Any]:
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

    # Filter event_sidecar.parquet strictly to internal_train keys (Round-2 F5 fix!)
    sc_df = pd.read_parquet(assert_unsealed_path(CLEAN_ROOT / 'truth_sidecar/event_sidecar.parquet'))
    pos_sc = pd.MultiIndex.from_frame(sc_df[KEY]).get_indexer(pd.MultiIndex.from_frame(tr_keys[KEY]))
    if (pos_sc < 0).any():
        raise ValueError('internal_train keys missing from event_sidecar.parquet')
    tr_sidecar = sc_df.iloc[pos_sc].reset_index(drop=True)
    del sc_df

    scratch_base = os.environ.get('PINPOINT_V3_SCRATCH', '/tmp')
    Path(scratch_base).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'dr-{variant}-', dir=scratch_base) as tmp_dir:
        tmp_p = Path(tmp_dir)
        staged_data = stage_authorized_train_shards(CLEAN_ROOT / 'data', tmp_p / 'native')
        shard_dirs = load_authorized_split_shards(staged_data, 'train', stage=1)

        tasks = [(str(sd), geom, bounds, BASE_OBS_CFG, None) for sd in shard_dirs]
        n_workers = min(4, max(1, os.cpu_count() or 1))
        val_400 = val_keys.iloc[:400].reset_index(drop=True)
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context('fork')) as pool:
            parts = list(pool.map(_extract_shard_worker, tasks))
            ood_pre = {
                ov: _build_ood_items(ov, geom, bounds, pool=pool)
                for ov in ('gen_ood_w184_epps21_nc_charm', 'gen_ood_run10085_10088')
            }
            stress_items, y_400 = _build_stress_items_from_internal_val(val_400, geom, bounds, pool=pool)
        all_items = [it for part in parts for it in part]
        shutil.rmtree(tmp_p / 'native', ignore_errors=True)
        print(f'[{variant}] Extracted {len(all_items)} train-pool + OOD + stress items in {time.perf_counter()-t0:.1f}s', flush=True)

        tr_items = _align_items_to_keys(all_items, tr_keys)
        dev_items = _align_items_to_keys(all_items, dev_keys)
        val_items = _align_items_to_keys(all_items, val_keys)
        del all_items

        train_ds = InMemoryDualDataset(tr_items, event_sidecar=tr_sidecar)
        cap = torch.cuda.get_device_capability(0)
        torch.manual_seed(seed)

        model_cfg = {
            'dim': 160,
            'heads': 8,
            'drop_path': 0.10,
            'head_dropout': 0.15,
            **VARIANT_CONFIGS[variant],
        }
        model = DualRegimeModalNet(geom, model_cfg).to(device)
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
        amp_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
        amp_ok = cap[0] >= 7
        scaler = torch.amp.GradScaler('cuda', enabled=(amp_ok and amp_dtype == torch.float16))

        y_dev = dev_keys['label'].to_numpy(dtype=np.int64)
        best_nll = float('inf')
        best_pass = 0
        history: list[dict[str, Any]] = []
        best_ema_path = out_dir / 'best_ema.pt'

        torch.cuda.reset_peak_memory_stats(device)
        for p_idx in range(1, passes + 1):
            tp0 = time.perf_counter()
            gen = torch.Generator()
            gen.manual_seed(seed + p_idx * 101)
            sampler = BatchSampler(RandomSampler(train_ds, generator=gen), batch_size=micro_bs, drop_last=False)
            loader = DataLoader(
                train_ds, batch_sampler=sampler, num_workers=0,
                collate_fn=lambda xs: collate_dual_regime(xs, mirror_rng=None, y_scint_center=yc, n_bars=nb),
            )
            model.train()
            _set_ieee_fp32(False)
            loss_sum = 0.0
            n_seen = 0
            win_buf: list[dict] = []
            for mb in loader:
                win_buf.append(mb)
                if len(win_buf) < grad_accum:
                    continue
                opt.zero_grad(set_to_none=True)
                fd = compute_dual_window_denominators(win_buf)
                win_loss = 0.0
                win_n = 0
                for sub_mb in win_buf:
                    db = move_batch_to_device(sub_mb, device)
                    with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_ok):
                        loss, _ = compute_dual_regime_loss(model, db, fixed_denoms=fd)
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
                win_buf.clear()

            if win_buf:
                opt.zero_grad(set_to_none=True)
                fd = compute_dual_window_denominators(win_buf)
                tail_loss = 0.0
                tail_n = 0
                for sub_mb in win_buf:
                    db = move_batch_to_device(sub_mb, device)
                    with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_ok):
                        loss, _ = compute_dual_regime_loss(model, db, fixed_denoms=fd)
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
                _update_ema(ema_model, model, ema_decay)

            train_loss = loss_sum / max(n_seen, 1)
            row_hist: dict[str, Any] = {'pass': p_idx, 'train_loss': train_loss, 'pass_sec': float(time.perf_counter() - tp0)}

            if p_idx % 2 == 0 or p_idx == passes:
                dev_pred = predict_dual_items_fp32(ema_model, dev_items, bounds, device, batch_size=64)
                dev_m = _ba4_and_nll(y_dev, dev_pred['probs'])
                row_hist['dev_screen_NLL'] = dev_m['NLL']
                row_hist['dev_screen_BA4_pct'] = dev_m['BA4_pct']
                if dev_m['NLL'] < best_nll:
                    best_nll = dev_m['NLL']
                    best_pass = p_idx
                    torch.save({'model': ema_model.state_dict(), 'config': model_cfg, 'pass': p_idx}, best_ema_path)
                print(f"[{variant}] pass={p_idx:02d}/{passes} train_loss={train_loss:.4f} dev_NLL={dev_m['NLL']:.4f} dev_BA4={dev_m['BA4_pct']:.2f}% ({row_hist['pass_sec']:.1f}s)", flush=True)
            history.append(row_hist)

        peak_vram_mib = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))

        # Load best EMA and evaluate strictly in deterministic IEEE-FP32
        _set_ieee_fp32(True)
        ckpt = torch.load(best_ema_path, map_location=device, weights_only=False)
        ema_model.load_state_dict(ckpt['model'])
        ema_model.eval()

        dev_pred = predict_dual_items_fp32(ema_model, dev_items, bounds, device, batch_size=64)
        val_pred = predict_dual_items_fp32(ema_model, val_items, bounds, device, batch_size=64)
        np.save(out_dir / 'probs_dev_screen.npy', dev_pred['probs'])
        np.save(out_dir / 'probs_internal_val_eval.npy', val_pred['probs'])
        np.save(out_dir / 'v_pv_internal_val_eval.npy', val_pred['v_pv_mm'])
        np.save(out_dir / 'scale_pv_internal_val_eval.npy', val_pred['scale_pv_mm'])

        # Open pure-W OOD evaluation
        ood_probs: dict[str, np.ndarray] = {}
        ood_metrics: dict[str, Any] = {}
        for ov, (o_items, yo) in ood_pre.items():
            op = predict_dual_items_fp32(ema_model, o_items, bounds, device, batch_size=64)['probs']
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

        # 7 Stress perturbations on first 400-event slice of internal_val_eval (ZERO Tune6k access!)
        stress_probs: dict[str, np.ndarray] = {}
        stress_metrics: dict[str, Any] = {}
        p_nom = None
        for st_name, _, _ in STRESS_SPECS:
            sp = predict_dual_items_fp32(ema_model, stress_items[st_name], bounds, device, batch_size=64)['probs']
            stress_probs[f'stress_{st_name}'] = sp
            if st_name == 'nominal':
                p_nom = sp
            tv = float(0.5 * np.abs(sp - p_nom).sum(axis=1).mean()) if p_nom is not None else 0.0
            m_st = _ba4_and_nll(y_400, sp)
            stress_metrics[st_name] = {'BA4_pct': m_st['BA4_pct'], 'NLL': m_st['NLL'], 'mean_TV_vs_nominal': tv}

        np.savez_compressed(out_dir / 'ood_and_stress_probs.npz', **ood_probs, **stress_probs)

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
            'internal_val_eval_all': val_m,
            'ood_metrics': ood_metrics,
            'stress_400_internal_val': stress_metrics,
            'history': history,
        }
        write_json(out_dir / 'run_summary.json', summary)
        print(f"[{variant}] DONE in {summary['wall_seconds']:.1f}s | peak_VRAM={peak_vram_mib:.0f} MiB | val_BA4={val_m['BA4_pct']:.2f}% val_NLL={val_m['NLL']:.4f}", flush=True)
        return summary


# ===========================================================================
# Mode 3: Section 13 Upstream Reconstruction vs Simple Classifier Controls
# ===========================================================================
def _analytic_student_t_pv_features(
    trk_kin: np.ndarray,
    trk_mask: np.ndarray,
    cal_start: np.ndarray,
) -> np.ndarray:
    """Parameter-free 4-step Student-t (nu=3) IRLS PV + 2V SV fit from non-exclusive tracklets."""
    b, m, _ = trk_kin.shape
    if b == 0:
        return np.empty((0, 12), dtype=np.float32)

    k_t = torch.from_numpy(trk_kin).float()
    m_t = torch.from_numpy(trk_mask).bool()
    cs_t = torch.from_numpy(cal_start).float()

    cal_valid = cs_t[:, 3] > 0.5
    prior = torch.where(
        cal_valid[:, None],
        torch.stack([cs_t[:, 1], cs_t[:, 2], cs_t[:, 0]], dim=-1),
        torch.tensor([0.0, -12.0, -2400.0], dtype=torch.float32)[None, :].expand(b, 3),
    )
    p_prior = torch.where(
        cal_valid[:, None],
        torch.tensor([4e-3, 4e-3, 2e-3], dtype=torch.float32)[None, :].expand(b, 3),
        torch.full((b, 3), 1e-5, dtype=torch.float32),
    )

    x0, y0, z0 = k_t[..., 0], k_t[..., 1], k_t[..., 2]
    tx, ty = k_t[..., 3], k_t[..., 4]
    sxx = k_t[..., 5].clamp_min(4e-5)
    syy = k_t[..., 6].clamp_min(4e-5)
    stx = k_t[..., 7].clamp_min(1e-7)
    sty = k_t[..., 8].clamp_min(1e-7)
    # Only target tracklets (not IPT)
    tgt_m = m_t & (z0 < -1800.0)
    pi_pv = tgt_m.float()

    v_cur = prior.clone()
    inv_diag = torch.ones((b, 3), dtype=torch.float32)
    nu = 3.0
    for _ in range(4):
        dz = (v_cur[:, None, 2] - z0).clamp(-350.0, 350.0)
        vx = sxx + dz.square() * stx
        vy = syy + dz.square() * sty
        rx = (x0 + tx * dz) - v_cur[:, None, 0]
        ry = (y0 + ty * dz) - v_cur[:, None, 1]
        chi2 = (rx.square() / vx + ry.square() / vy).clamp_max(1e4)
        w_rob = pi_pv / torch.pow(1.0 + chi2 / nu, 0.5 * (nu + 1.0))
        wx, wy = w_rob / vx, w_rob / vy
        cx, cy = x0 - tx * z0, y0 - ty * z0
        m00 = p_prior[:, 0] + wx.sum(1)
        m11 = p_prior[:, 1] + wy.sum(1)
        m02 = -(wx * tx).sum(1)
        m12 = -(wy * ty).sum(1)
        m22 = p_prior[:, 2] + (wx * tx.square() + wy * ty.square()).sum(1)
        zs = torch.zeros_like(m00)
        mat = torch.stack([
            torch.stack([m00, zs, m02], -1),
            torch.stack([zs, m11, m12], -1),
            torch.stack([m02, m12, m22], -1),
        ], -2)
        rhs = torch.stack([
            p_prior[:, 0] * prior[:, 0] + (wx * cx).sum(1),
            p_prior[:, 1] * prior[:, 1] + (wy * cy).sum(1),
            p_prior[:, 2] * prior[:, 2] - (wx * tx * cx + wy * ty * cy).sum(1),
        ], -1)
        v_cur, inv_diag = _solve_sym_3x3(mat, rhs)

    dz_f = (v_cur[:, None, 2] - z0).clamp(-350.0, 350.0)
    dx_pv = (x0 + tx * dz_f) - v_cur[:, None, 0]
    dy_pv = (y0 + ty * dz_f) - v_cur[:, None, 1]
    var_xf = sxx + dz_f.square() * stx + inv_diag[:, None, 0]
    var_yf = syy + dz_f.square() * sty + inv_diag[:, None, 1]
    chi2_pv = (dx_pv.square() / var_xf + dy_pv.square() / var_yf) * tgt_m.float()
    s_d0 = torch.sqrt(chi2_pv + 1e-10) * tgt_m.float()
    d0 = torch.sqrt(dx_pv.square() + dy_pv.square() + 1e-10) * tgt_m.float()

    # Analytic 2V outlier fit
    pi_sv = torch.sigmoid(s_d0 - 3.0) * tgt_m.float()
    sv_prior = v_cur + torch.tensor([0.0, 0.0, 2.5])[None, :]
    p_sv0 = torch.tensor([1e-2, 1e-2, 5e-3])[None, :].expand(b, 3)
    v_sv = sv_prior.clone()
    for _ in range(2):
        dz_s = (v_sv[:, None, 2] - z0).clamp(-350.0, 350.0)
        wxs = pi_sv / (sxx + dz_s.square() * stx)
        wys = pi_sv / (syy + dz_s.square() * sty)
        cx, cy = x0 - tx * z0, y0 - ty * z0
        mat_s = torch.stack([
            torch.stack([p_sv0[:, 0] + wxs.sum(1), torch.zeros(b), -(wxs * tx).sum(1)], -1),
            torch.stack([torch.zeros(b), p_sv0[:, 1] + wys.sum(1), -(wys * ty).sum(1)], -1),
            torch.stack([-(wxs * tx).sum(1), -(wys * ty).sum(1), p_sv0[:, 2] + (wxs * tx.square() + wys * ty.square()).sum(1)], -1),
        ], -2)
        rhs_s = torch.stack([
            p_sv0[:, 0] * sv_prior[:, 0] + (wxs * cx).sum(1),
            p_sv0[:, 1] * sv_prior[:, 1] + (wys * cy).sum(1),
            p_sv0[:, 2] * sv_prior[:, 2] - (wxs * tx * cx + wys * ty * cy).sum(1),
        ], -1)
        v_sv, _ = _solve_sym_3x3(mat_s, rhs_s)

    dz_sv_f = (v_sv[:, None, 2] - z0).clamp(-350.0, 350.0)
    dx_sv = (x0 + tx * dz_sv_f) - v_sv[:, None, 0]
    dy_sv = (y0 + ty * dz_sv_f) - v_sv[:, None, 1]
    chi2_sv = dx_sv.square() / (sxx + dz_sv_f.square() * stx) + dy_sv.square() / (syy + dz_sv_f.square() * sty)
    delta_chi2_2v = (pi_sv * (chi2_pv - chi2_sv).clamp(min=0.0, max=50.0)).sum(dim=1)
    dv_cal = torch.linalg.norm(v_cur - prior, dim=-1)
    sig_pv = torch.sqrt(inv_diag.clamp_min(1e-6))

    feats = torch.stack([
        v_cur[:, 0] / 150.0,
        (v_cur[:, 1] + 12.0) / 150.0,
        (v_cur[:, 2] + 2400.0) / 1000.0,
        torch.log1p(dv_cal) / 5.0,
        torch.log1p(sig_pv[:, 0]) / 3.0,
        torch.log1p(sig_pv[:, 1]) / 3.0,
        torch.log1p(sig_pv[:, 2]) / 4.0,
        s_d0.max(dim=1).values,
        (s_d0 > 3.0).float().sum(dim=1),
        d0.max(dim=1).values,
        torch.log1p(delta_chi2_2v) / 4.0,
        (v_sv[:, 2] - v_cur[:, 2]).clamp(-20.0, 50.0) / 20.0,
    ], dim=-1)
    return np.nan_to_num(feats.numpy(), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _analytic_sinkhorn_tof_features(
    items: list[dict],
    stations: dict,
    bounds: dict,
) -> np.ndarray:
    """Compute analytic per-event ToF + 3-step Sinkhorn stereo de-ghosted calorimeter features (12 dims)."""
    n = len(items)
    if n == 0:
        return np.empty((0, 12), dtype=np.float32)

    n_st = int(stations['n_stations'])
    n_dbl = int(stations['n_doublets'])
    n_bars = int(stations['n_bars'])
    dbl = stations['doublet_stations']
    p2s = stations['panel_to_station']
    p2x = stations['panel_to_xslot']
    p2y = stations['panel_to_yslot']
    st_z = stations['station_z_mm']
    bx_mm = stations['bar_x_centers_mm']
    by_mm = stations['bar_y_centers_mm']
    x_c = float(stations['x_scint_center'])
    y_c = float(stations['y_scint_center'])

    out = np.zeros((n, 12), dtype=np.float32)
    for i, it in enumerate(items):
        bp = it['bar_panel']
        if len(bp) == 0:
            continue
        bh = it['bar_horiz']
        bi = it['bar_idx']
        bv = it['bar_val'].astype(np.float64)
        bt = it['bar_time_ns'].astype(np.float64)
        bz = st_z[p2s[bp]]
        tof = compute_scint_tof_residual_ns(bt, bz, valid_mask=(bt >= 0.0))
        valid_t = bt >= 0.0
        if valid_t.any():
            tv = tof[valid_t]
            out[i, 0] = float(np.mean(tv))
            out[i, 1] = float(np.std(tv))
            out[i, 2] = float(np.quantile(tv, 0.90))
            out[i, 3] = float(np.max(tv))
            e_bar = np.expm1(np.clip(bv[:, 0] * 10.0, 0.0, 20.0))
            e_tot = max(float(e_bar[valid_t].sum()), 1e-6)
            out[i, 4] = float(e_bar[valid_t & (tv > 0.65)].sum() / e_tot)  # late tail fraction

        ex_g = np.zeros((n_dbl, n_bars), dtype=np.float64)
        ey_g = np.zeros((n_dbl, n_bars), dtype=np.float64)
        tx_g = np.zeros((n_dbl, n_bars), dtype=np.float64)
        ty_g = np.zeros((n_dbl, n_bars), dtype=np.float64)
        dbl_lookup = np.full(n_st, -1, dtype=np.int64)
        dbl_lookup[dbl] = np.arange(n_dbl, dtype=np.int64)

        for j in range(len(bp)):
            s = int(p2s[bp[j]])
            d = int(dbl_lookup[s])
            if d < 0:
                continue
            e_mev = math.expm1(max(0.0, float(bv[j, 0]) * 10.0))
            if bh[j] == 0:
                ex_g[d, bi[j]] = e_mev
                tx_g[d, bi[j]] = float(tof[j])
            else:
                ey_g[d, bi[j]] = e_mev
                ty_g[d, bi[j]] = float(tof[j])

        sum_x = ex_g.sum(axis=-1, keepdims=True)
        sum_y = ey_g.sum(axis=-1, keepdims=True)
        m_dual = ((sum_x > 0.0) & (sum_y > 0.0)).squeeze(-1)
        out[i, 5] = float(m_dual.sum()) / float(n_dbl)
        if not m_dual.any():
            continue

        # 3-step Sinkhorn per dual-active doublet station
        p_s = ex_g[m_dual] / np.maximum(sum_x[m_dual], 1e-8)
        q_s = ey_g[m_dual] / np.maximum(sum_y[m_dual], 1e-8)
        tx_d = tx_g[m_dual]
        ty_d = ty_g[m_dual]
        dt = np.abs(tx_d[:, :, None] - ty_d[:, None, :])
        de = np.abs(np.log1p(ex_g[m_dual])[:, :, None] - np.log1p(ey_g[m_dual])[:, None, :])
        cost = 0.8 * dt + 0.5 * de
        K = np.exp(-cost / 0.5)
        u = p_s.copy()
        v = q_s.copy()
        for _ in range(3):
            u = p_s / np.maximum(np.einsum('sij,sj->si', K, v), 1e-8)
            v = q_s / np.maximum(np.einsum('sij,si->sj', K, u), 1e-8)
        gamma = u[:, :, None] * K * v[:, None, :]
        e_3d = gamma * 0.5 * (sum_x[m_dual, :, None] + sum_y[m_dual, None, :])

        tot_e3 = float(e_3d.sum())
        out[i, 6] = math.log1p(tot_e3) / 10.0
        g_safe = np.clip(gamma, 1e-12, 1.0)
        out[i, 7] = float(-(gamma * np.log(g_safe)).sum(axis=(1, 2)).mean())
        out[i, 8] = float(e_3d.max() / max(tot_e3, 1e-6))

        # Transverse R90 & pT_miss from Sinkhorn voxels
        w_xy = e_3d.sum(axis=0)  # (40, 40)
        xg, yg = np.meshgrid(bx_mm, by_mm, indexing='ij')
        xcog = float((w_xy * xg).sum() / max(tot_e3, 1e-6))
        ycog = float((w_xy * yg).sum() / max(tot_e3, 1e-6))
        r_vals = np.hypot(xg - xcog, yg - ycog).ravel()
        w_vals = w_xy.ravel()
        ord_r = np.argsort(r_vals)
        cdf = np.cumsum(w_vals[ord_r]) / max(tot_e3, 1e-6)
        r90 = float(r_vals[ord_r[min(len(r_vals) - 1, int(np.searchsorted(cdf, 0.90)))]])
        out[i, 9] = r90 / 200.0

        dz_st = np.maximum(30.0, st_z[dbl[m_dual]] - float(st_z[dbl[m_dual]][0]))
        inv_r = 1.0 / np.sqrt((bx_mm[None, :, None] - xcog) ** 2 + (by_mm[None, None, :] - ycog) ** 2 + dz_st[:, None, None] ** 2)
        qx = float((e_3d * (bx_mm[None, :, None] - xcog) * inv_r).sum())
        qy = float((e_3d * (by_mm[None, None, :] - ycog) * inv_r).sum())
        out[i, 10] = math.hypot(qx, qy) / max(tot_e3, 1e-6)
        # Core 30mm energy fraction
        out[i, 11] = float(w_xy[np.hypot(xg - xcog, yg - ycog) <= 30.0].sum() / max(tot_e3, 1e-6))

    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _extract_sec13_features_batch(
    items: list[dict],
    geom: dict,
    bounds: dict,
    stations: dict,
    nomad_ext: NomadTransverseExtractor,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract both exact Stage-325 70-d features AND improved upstream-reco features (~128-d)."""
    n = len(items)
    feats70 = np.zeros((n, 70), dtype=np.float32)
    extra_imp = np.zeros((n, 52), dtype=np.float32)
    yc, nb = float(bounds['y_scint_center']), int(bounds['n_bars'])
    x_pc = float(bounds['x_pix_center'])

    for st in range(0, n, 128):
        sub = items[st : st + 128]
        batch = collate_dual_regime(sub, mirror_rng=None, y_scint_center=yc, n_bars=nb)
        with torch.inference_mode():
            # NomadTransverseExtractor expects 14-d cand (first slice is already 14-d)
            nomad_t = nomad_ext.from_pf(batch).numpy()
            dca_mat = pairwise_skew_dca_features(batch['trk_kin'], batch['trk_mask'])
            is_ipt = batch['trk_mask'] & (batch['trk_morph'][..., 1] > 0.5)
            ilock = compute_ipt_spectrometer_interlock(
                batch['trk_kin'], is_ipt, batch['trk_cand'], batch['trk_cand_mask'],
                x_pix_center=x_pc, y_scint_center=yc,
            ).numpy()

        pv_feats = _analytic_student_t_pv_features(
            batch['trk_kin'].numpy(), batch['trk_mask'].numpy(), batch['cal_start'].numpy()
        )
        sink_feats = _analytic_sinkhorn_tof_features(sub, stations, bounds)

        dca_np = dca_mat.numpy()
        tm_np = batch['trk_mask'].numpy()
        tk_np = batch['trk_kin'].numpy()
        tmo_np = batch['trk_morph'].numpy()

        for loc, it in enumerate(sub):
            gi = st + loc
            c_obs = it['c_obs']
            mv = it['micro_vtx_feat']
            nm = nomad_t[loc]
            cand_m = it['trk_cand_mask']
            n_cand = float(cand_m.sum())
            lead_c = it['trk_cand'][0] if len(it['trk_cand']) else np.zeros(14, dtype=np.float32)
            ps = it['pix_plane_summary']
            tgt_ps = ps[:12]
            ipt_ps = ps[12:]
            ps_agg = np.array([
                float(tgt_ps[:, 0].sum()) / 12.0,
                float(tgt_ps[:, 1].sum()),
                float(tgt_ps[:, 2].sum()),
                float(tgt_ps[:, 8].sum()),
                float(tgt_ps[:, 9].sum()),
                float(tgt_ps[:, 10].mean()),
                float(tgt_ps[:, 11].mean()),
                float(tgt_ps[0, 1]),
                float(ipt_ps[:, 0].sum()) / 3.0,
                float(ipt_ps[:, 1].sum()),
                float(ipt_ps[:, 2].sum()),
                float(ipt_ps[:, 8].sum()),
                float(ipt_ps[:, 9].sum()),
                float(ipt_ps[:, 10].mean()),
                float(ipt_ps[:, 11].mean()),
                float(it['avr_pv_mm'][3]),
            ], dtype=np.float32)
            row70 = np.concatenate([c_obs, mv, nm, np.array([n_cand], dtype=np.float32), lead_c[:11], ps_agg])
            feats70[gi] = np.nan_to_num(row70, nan=0.0, posinf=0.0, neginf=0.0)

            # Soft tracklet + skew-DCA stats (16 dims)
            vm = tm_np[loc]
            trk_stats = np.zeros(16, dtype=np.float32)
            if vm.any():
                k_l = tk_np[loc, vm]
                m_l = tmo_np[loc, vm]
                is_3 = (np.floor(m_l[:, 0] + 0.1) >= 1.0)
                is_sh = ((m_l[:, 0] % 1.0) > 0.25)
                is_i = m_l[:, 1] > 0.5
                is_f = (~is_i) & (k_l[:, 2] < -2750.0)
                is_m = (~is_i) & (k_l[:, 2] >= -2750.0)
                sl = np.hypot(k_l[:, 3], k_l[:, 4])
                trk_stats[0] = float(vm.sum())
                trk_stats[1] = float(is_3.sum())
                trk_stats[2] = float(is_sh.sum())
                trk_stats[3] = float(is_f.sum())
                trk_stats[4] = float(is_m.sum())
                trk_stats[5] = float(is_i.sum())
                trk_stats[6] = float(m_l[:, 6].mean())
                trk_stats[7] = float(m_l[:, 7].mean())
                trk_stats[8] = float(np.min(k_l[:, 9]))
                trk_stats[9] = float(np.mean(sl))
                trk_stats[10] = float(np.max(sl))
                trk_stats[11] = float(np.quantile(k_l[:, 10], 0.90))
                if int(vm.sum()) >= 2:
                    idx_v = np.flatnonzero(vm)
                    sub_d = dca_np[loc][np.ix_(idx_v, idx_v)]
                    off_d = ~np.eye(len(idx_v), dtype=bool)
                    trk_stats[12] = float(np.min(sub_d[:, :, 0][off_d]))
                    trk_stats[13] = float(np.min(sub_d[:, :, 2][off_d]))
                    trk_stats[14] = float(np.max(sub_d[:, :, 10][off_d]))
                    trk_stats[15] = float(np.max(sub_d[:, :, 11][off_d]))

            # 4D IPT-Spectrometer Interlock stats (12 dims)
            il_l = ilock[loc]
            cm_l = cand_m
            st_cnt = it['trk_station_counts']
            il_stats = np.zeros(12, dtype=np.float32)
            il_stats[0] = math.log1p(float(st_cnt[0]))
            il_stats[1] = math.log1p(float(st_cnt[1]))
            il_stats[2] = math.log1p(float(st_cnt[2]))
            if cm_l.any():
                c_v = it['trk_cand'][cm_l]
                il_v = il_l[cm_l]
                is_3b = c_v[:, 0] > 0.75
                best_k = int(np.argmax(il_v[:, 0] + 0.1 * is_3b.astype(np.float32)))
                il_stats[3] = float(il_v[:, 0].max())
                il_stats[4] = float(il_v[:, 1].min())
                il_stats[5] = float(il_v[best_k, 2])
                il_stats[6] = float(il_v[best_k, 3])
                il_stats[7] = float(is_3b.sum())
                il_stats[8] = float((il_v[:, 0] > 0.35).sum())
                il_stats[9] = float(c_v[best_k, 8])  # p_proxy of IPT-best cand
                il_stats[10] = float(c_v[best_k, 6]) # sagitta sign of IPT-best cand
                il_stats[11] = float(c_v[best_k, 9]) # chi2_x of IPT-best cand

            imp_row = np.concatenate([trk_stats, pv_feats[loc], il_stats, sink_feats[loc]])
            extra_imp[gi] = np.nan_to_num(imp_row, nan=0.0, posinf=0.0, neginf=0.0)

    feats_imp = np.concatenate([feats70, extra_imp], axis=1)
    return feats70, feats_imp


def run_section13_controls() -> dict[str, Any]:
    assert_unsealed_split('internal_train', stage=1)
    assert_unsealed_split('internal_screen', stage=1)
    assert_unsealed_split('internal_val', stage=1)

    t0 = time.perf_counter()
    geom = json.loads(assert_unsealed_path(CLEAN_ROOT / 'geometry.json').read_text())
    bounds = detector_bounds(geom)
    bounds['_plane_x0'] = np.array([cumulative_radiation_length(float(z), geom) for z in bounds['pixel_Zpos']], dtype=np.float32)
    stations = static_scint_stations(geom, bounds)
    nomad_ext = NomadTransverseExtractor(geom, phys_calibrated=True, pe_per_mip_mev=40.0)

    tr_keys = pd.read_parquet(assert_unsealed_path(SPLITS_DIR / 'internal_train_keys.parquet'))
    dev_keys = pd.read_parquet(assert_unsealed_path(SPLITS_DIR / 'internal_dev_screen_keys.parquet'))
    val_keys = pd.read_parquet(assert_unsealed_path(SPLITS_DIR / 'internal_val_eval_keys.parquet'))

    scratch_base = os.environ.get('PINPOINT_V3_SCRATCH', '/tmp')
    Path(scratch_base).mkdir(parents=True, exist_ok=True)
    val_400 = val_keys.iloc[:400].reset_index(drop=True)
    with tempfile.TemporaryDirectory(prefix='dr-sec13-', dir=scratch_base) as tmp_dir:
        staged = stage_authorized_train_shards(CLEAN_ROOT / 'data', Path(tmp_dir) / 'native')
        shard_dirs = load_authorized_split_shards(staged, 'train', stage=1)
        tasks = [(str(sd), geom, bounds, BASE_OBS_CFG, None) for sd in shard_dirs]
        n_w = min(8, max(1, os.cpu_count() or 1))
        with ProcessPoolExecutor(max_workers=n_w, mp_context=mp.get_context('fork')) as pool:
            parts = list(pool.map(_extract_shard_worker, tasks))
            ood_pre = {
                ov: _build_ood_items(ov, geom, bounds, pool=pool)
                for ov in ('gen_ood_w184_epps21_nc_charm', 'gen_ood_run10085_10088')
            }
            stress_items, y_400 = _build_stress_items_from_internal_val(val_400, geom, bounds, pool=pool)
        all_items = [it for part in parts for it in part]
        shutil.rmtree(Path(tmp_dir) / 'native', ignore_errors=True)

    tr_items = _align_items_to_keys(all_items, tr_keys)
    dev_items = _align_items_to_keys(all_items, dev_keys)
    val_items = _align_items_to_keys(all_items, val_keys)
    del all_items

    print(f'[sec13] Extracting 70d & improved-reco tabular features ({time.perf_counter()-t0:.1f}s)...', flush=True)
    X70_tr, Ximp_tr = _extract_sec13_features_batch(tr_items, geom, bounds, stations, nomad_ext)
    X70_dev, Ximp_dev = _extract_sec13_features_batch(dev_items, geom, bounds, stations, nomad_ext)
    X70_val, Ximp_val = _extract_sec13_features_batch(val_items, geom, bounds, stations, nomad_ext)
    y_tr = tr_keys['label'].to_numpy(dtype=np.int64)
    y_dev = dev_keys['label'].to_numpy(dtype=np.int64)
    y_val = val_keys['label'].to_numpy(dtype=np.int64)

    # Pre-extract OOD & Stress features
    ood_feats: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for ov, (o_items, yo) in ood_pre.items():
        xo70, xoimp = _extract_sec13_features_batch(o_items, geom, bounds, stations, nomad_ext)
        ood_feats[ov] = (xo70, xoimp, yo)

    stress_feats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for st_name, _, _ in STRESS_SPECS:
        xs70, xsimp = _extract_sec13_features_batch(stress_items[st_name], geom, bounds, stations, nomad_ext)
        stress_feats[st_name] = (xs70, xsimp)

    results: dict[str, Any] = {}

    # 1. s1_bdt_70d & 2. s1_improved_reco_bdt
    for name, Xtr, Xdev, Xval, use_imp in (
        ('s1_bdt_70d', X70_tr, X70_dev, X70_val, False),
        ('s1_improved_reco_bdt', Ximp_tr, Ximp_dev, Ximp_val, True),
    ):
        clf = HistGradientBoostingClassifier(
            max_iter=350, max_depth=8, learning_rate=0.06, min_samples_leaf=30, l2_regularization=1.0, random_state=325
        )
        clf.fit(Xtr, y_tr)
        p_dev = clf.predict_proba(Xdev).astype(np.float32)
        p_val = clf.predict_proba(Xval).astype(np.float32)

        d_out = EXP_ROOT / name
        d_out.mkdir(parents=True, exist_ok=True)
        np.save(d_out / 'probs_dev_screen.npy', p_dev)
        np.save(d_out / 'probs_internal_val_eval.npy', p_val)
        with (d_out / 'model.pkl').open('wb') as fh:
            pickle.dump({'clf': clf, 'n_features': int(Xtr.shape[1])}, fh, protocol=5)

        ood_p, ood_m = {}, {}
        for ov, (xo70, xoimp, yo) in ood_feats.items():
            po = clf.predict_proba(xoimp if use_imp else xo70).astype(np.float32)
            ood_p[f'ood_{ov}'] = po
            if 'nc_' in ov:
                ood_m[ov] = {
                    'n_events': int(len(yo)),
                    'mean_p_tau': float(po[:, 2].mean()),
                    'frac_pred_nc_pct': float((po.argmax(1) == 3).mean() * 100.0),
                    'frac_pred_tau_pct': float((po.argmax(1) == 2).mean() * 100.0),
                    'frac_p_tau_gt_0p5_pct': float((po[:, 2] > 0.5).mean() * 100.0),
                }
            else:
                ood_m[ov] = _ba4_and_nll(yo, po)

        st_p, st_m = {}, {}
        p_nom = None
        for st_name, _, _ in STRESS_SPECS:
            xs70, xsimp = stress_feats[st_name]
            ps = clf.predict_proba(xsimp if use_imp else xs70).astype(np.float32)
            st_p[f'stress_{st_name}'] = ps
            if st_name == 'nominal':
                p_nom = ps
            tv = float(0.5 * np.abs(ps - p_nom).sum(axis=1).mean()) if p_nom is not None else 0.0
            m_st = _ba4_and_nll(y_400, ps)
            st_m[st_name] = {'BA4_pct': m_st['BA4_pct'], 'NLL': m_st['NLL'], 'mean_TV_vs_nominal': tv}

        np.savez_compressed(d_out / 'ood_and_stress_probs.npz', **ood_p, **st_p)
        val_m = _ba4_and_nll(y_val, p_val)
        summary = {
            'variant': name,
            'n_features': int(Xtr.shape[1]),
            'internal_val_eval_all': val_m,
            'ood_metrics': ood_m,
            'stress_400_internal_val': st_m,
        }
        write_json(d_out / 'run_summary.json', summary)
        results[name] = summary
        print(f"[{name}] val_BA4={val_m['BA4_pct']:.2f}% val_NLL={val_m['NLL']:.4f}", flush=True)

    # 3. s1_improved_reco_mlp (3-layer MLP on improved upstream reco features)
    torch.manual_seed(42)
    dev_t = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    _set_ieee_fp32(True)
    scaler = StandardScaler()
    Ximp_tr_s = scaler.fit_transform(Ximp_tr).astype(np.float32)
    Ximp_dev_s = scaler.transform(Ximp_dev).astype(np.float32)
    Ximp_val_s = scaler.transform(Ximp_val).astype(np.float32)

    class _TabularMLP3(nn.Module):
        def __init__(self, in_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.LayerNorm(256),
                nn.SiLU(),
                nn.Dropout(0.15),
                nn.Linear(256, 256),
                nn.LayerNorm(256),
                nn.SiLU(),
                nn.Dropout(0.15),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.SiLU(),
                nn.Linear(128, 4),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    mlp = _TabularMLP3(Ximp_tr_s.shape[1]).to(dev_t)
    opt_m = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-2)
    epochs_m = 40
    sched_m = torch.optim.lr_scheduler.CosineAnnealingLR(opt_m, T_max=epochs_m, eta_min=1e-5)
    x_tr_t = torch.from_numpy(Ximp_tr_s)
    y_tr_t = torch.from_numpy(y_tr)
    x_dev_t = torch.from_numpy(Ximp_dev_s).to(dev_t)

    best_mlp_nll = float('inf')
    best_mlp_state = None
    best_mlp_ep = 0
    for ep in range(1, epochs_m + 1):
        mlp.train()
        perm = torch.randperm(len(x_tr_t))
        for st in range(0, len(x_tr_t), 128):
            idx = perm[st : st + 128]
            xb = x_tr_t[idx].to(dev_t)
            yb = y_tr_t[idx].to(dev_t)
            opt_m.zero_grad(set_to_none=True)
            loss = F.cross_entropy(mlp(xb), yb)
            loss.backward()
            opt_m.step()
        sched_m.step()
        mlp.eval()
        with torch.no_grad():
            p_d = torch.softmax(mlp(x_dev_t).float(), dim=-1).cpu().numpy()
        dm = _ba4_and_nll(y_dev, p_d)
        if dm['NLL'] < best_mlp_nll:
            best_mlp_nll = dm['NLL']
            best_mlp_ep = ep
            best_mlp_state = copy.deepcopy(mlp.state_dict())

    assert best_mlp_state is not None
    mlp.load_state_dict(best_mlp_state)
    mlp.eval()

    @torch.no_grad()
    def _mlp_pred(x_np: np.ndarray) -> np.ndarray:
        xs = scaler.transform(x_np).astype(np.float32)
        logits = mlp(torch.from_numpy(xs).to(dev_t)).float()
        return torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)

    p_dev_m = _mlp_pred(Ximp_dev)
    p_val_m = _mlp_pred(Ximp_val)
    d_mlp = EXP_ROOT / 's1_improved_reco_mlp'
    d_mlp.mkdir(parents=True, exist_ok=True)
    np.save(d_mlp / 'probs_dev_screen.npy', p_dev_m)
    np.save(d_mlp / 'probs_internal_val_eval.npy', p_val_m)
    torch.save({'model': best_mlp_state, 'scaler': scaler, 'in_dim': int(Ximp_tr.shape[1])}, d_mlp / 'best_mlp.pt')

    ood_p, ood_m = {}, {}
    for ov, (_, xoimp, yo) in ood_feats.items():
        po = _mlp_pred(xoimp)
        ood_p[f'ood_{ov}'] = po
        if 'nc_' in ov:
            ood_m[ov] = {
                'n_events': int(len(yo)),
                'mean_p_tau': float(po[:, 2].mean()),
                'frac_pred_nc_pct': float((po.argmax(1) == 3).mean() * 100.0),
                'frac_pred_tau_pct': float((po.argmax(1) == 2).mean() * 100.0),
                'frac_p_tau_gt_0p5_pct': float((po[:, 2] > 0.5).mean() * 100.0),
            }
        else:
            ood_m[ov] = _ba4_and_nll(yo, po)

    st_p, st_m = {}, {}
    p_nom = None
    for st_name, _, _ in STRESS_SPECS:
        _, xsimp = stress_feats[st_name]
        ps = _mlp_pred(xsimp)
        st_p[f'stress_{st_name}'] = ps
        if st_name == 'nominal':
            p_nom = ps
        tv = float(0.5 * np.abs(ps - p_nom).sum(axis=1).mean()) if p_nom is not None else 0.0
        m_st = _ba4_and_nll(y_400, ps)
        st_m[st_name] = {'BA4_pct': m_st['BA4_pct'], 'NLL': m_st['NLL'], 'mean_TV_vs_nominal': tv}

    np.savez_compressed(d_mlp / 'ood_and_stress_probs.npz', **ood_p, **st_p)
    val_mm = _ba4_and_nll(y_val, p_val_m)
    summary_mlp = {
        'variant': 's1_improved_reco_mlp',
        'n_features': int(Ximp_tr.shape[1]),
        'n_params': int(sum(p.numel() for p in mlp.parameters())),
        'best_epoch': best_mlp_ep,
        'best_dev_screen_NLL': best_mlp_nll,
        'internal_val_eval_all': val_mm,
        'ood_metrics': ood_m,
        'stress_400_internal_val': st_m,
    }
    write_json(d_mlp / 'run_summary.json', summary_mlp)
    results['s1_improved_reco_mlp'] = summary_mlp
    print(f"[s1_improved_reco_mlp] val_BA4={val_mm['BA4_pct']:.2f}% val_NLL={val_mm['NLL']:.4f} ({time.perf_counter()-t0:.1f}s)", flush=True)
    return results


# ===========================================================================
# Mode 4: Aggregate All Stage-1 Results + Paired Family Bootstrap
# ===========================================================================
def run_aggregate() -> dict[str, Any]:
    assert_unsealed_split('internal_val', stage=1)
    val_keys = pd.read_parquet(assert_unsealed_path(SPLITS_DIR / 'internal_val_eval_keys.parquet'))
    y_val = val_keys['label'].to_numpy(dtype=np.int64)
    dom_val = val_keys['obs_domain_id'].to_numpy(dtype=np.int64)
    reg_val = val_keys['region_id'].to_numpy(dtype=np.int64)
    fam_codes = pd.MultiIndex.from_frame(val_keys[FAMILY]).factorize()[0]

    # Join event_sidecar strictly for offline PV accuracy & pull evaluation on internal_val_eval
    sc_df = pd.read_parquet(assert_unsealed_path(CLEAN_ROOT / 'truth_sidecar/event_sidecar.parquet'))
    pos_sc = pd.MultiIndex.from_frame(sc_df[KEY]).get_indexer(pd.MultiIndex.from_frame(val_keys[KEY]))
    sc_val = sc_df.iloc[pos_sc].reset_index(drop=True)
    v_true = np.column_stack([
        sc_val['vertex_x_mm'].to_numpy(np.float64),
        sc_val['vertex_y_mm'].to_numpy(np.float64),
        sc_val['vertex_z_mm'].to_numpy(np.float64),
    ])
    v_valid = sc_val['vertex_valid'].to_numpy(dtype=np.int64) > 0

    all_names = [
        's1_dual_regime_modalnet',
        'abl_no_ipt_spec_interlock',
        'abl_no_sinkhorn_tof',
        'abl_no_pixdark_noise_aug',
        'abl_homogeneous_regime',
        's1_bdt_70d',
        's1_improved_reco_bdt',
        's1_improved_reco_mlp',
    ]

    probs_map: dict[str, np.ndarray] = {}
    models_report: dict[str, Any] = {}
    for name in all_names:
        d_run = EXP_ROOT / name
        if not (d_run / 'probs_internal_val_eval.npy').is_file():
            continue
        probs = np.load(d_run / 'probs_internal_val_eval.npy').astype(np.float64)
        probs_map[name] = probs
        run_sum = json.loads((d_run / 'run_summary.json').read_text()) if (d_run / 'run_summary.json').is_file() else {}

        # Per-domain & per-region BA4/NLL
        dom_res = {}
        for did in (1, 2, 3):
            dm = (dom_val == did)
            if dm.any():
                dom_res[OBS_DOMAIN_NAMES[did]] = _ba4_and_nll(y_val[dm], probs[dm])
        reg_names = {0: 'FrontPinpoint_30mm_PixXY', 1: 'IntermPinpoint_150mm_PixXY', 2: 'Pinpoint_OuterW', 3: 'Fortune_ScintOnly'}
        reg_res = {}
        for rid, rname in reg_names.items():
            rm = (reg_val == rid)
            if rm.any():
                reg_res[rname] = _ba4_and_nll(y_val[rm], probs[rm])

        # 1% FPR dual-metric evaluation
        s_tau = probs[:, 2]
        m_all = np.ones(len(y_val), dtype=bool)
        m_d1 = (dom_val == 1)
        m_pin = (reg_val == 0) | (reg_val == 1)
        tail = {
            'split_half_1pct_All': _split_half_tail_eval(s_tau, y_val, val_keys, m_all, 0.01),
            'split_half_1pct_D1_obs': _split_half_tail_eval(s_tau, y_val, val_keys, m_d1, 0.01),
            'split_half_1pct_Pinpoint': _split_half_tail_eval(s_tau, y_val, val_keys, m_pin, 0.01),
            'within_slice_1pct_All': _within_slice_tail_eval(s_tau, y_val, m_all, 0.01),
            'within_slice_1pct_D1_obs': _within_slice_tail_eval(s_tau, y_val, m_d1, 0.01),
            'within_slice_1pct_Pinpoint': _within_slice_tail_eval(s_tau, y_val, m_pin, 0.01),
        }

        collab_t3 = {
            'All_non_tau': _collaboration_t3_eval(probs, y_val, m_all),
            'D1_obs_non_tau': _collaboration_t3_eval(probs, y_val, m_d1),
            'D2_obs_non_tau': _collaboration_t3_eval(probs, y_val, dom_val == 2),
        }

        # Neural PV audit
        pv_audit = None
        vp_file = d_run / 'v_pv_internal_val_eval.npy'
        sp_file = d_run / 'scale_pv_internal_val_eval.npy'
        if vp_file.is_file() and sp_file.is_file():
            vh = np.load(vp_file).astype(np.float64)
            sh = np.load(sp_file).astype(np.float64)
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
            'tau_1pct_fpr': tail,
            'collaboration_t3': collab_t3,
            'pv_audit': pv_audit,
            'ood_metrics': run_sum.get('ood_metrics'),
            'stress_400_internal_val': run_sum.get('stress_400_internal_val'),
        }

    # Paired 1,000-rep family bootstrap comparisons (seed=20263250)
    paired_boot: dict[str, Any] = {}
    if 's1_dual_regime_modalnet' in probs_map:
        pa = probs_map['s1_dual_regime_modalnet']
        m_all = np.ones(len(y_val), dtype=bool)
        m_d1 = (dom_val == 1)
        m_d2 = (dom_val == 2)
        for other in all_names[1:]:
            if other not in probs_map:
                continue
            pb = probs_map[other]
            for stag, mk in (('All', m_all), ('D1_obs', m_d1), ('D2_obs', m_d2)):
                paired_boot[f'DualRegime - {other} ({stag})'] = paired_family_bootstrap_delta(
                    pa[:, 2], pb[:, 2], pa, pb, y_val, fam_codes, mk, n_boot=1000, seed=20263250
                )

    # Also Section 13 comparison: ImprovedReco_BDT - BDT_70d, ImprovedReco_MLP - BDT_70d
    if 's1_bdt_70d' in probs_map:
        p_bdt70 = probs_map['s1_bdt_70d']
        for imp_name in ('s1_improved_reco_bdt', 's1_improved_reco_mlp'):
            if imp_name in probs_map:
                pi = probs_map[imp_name]
                paired_boot[f'{imp_name} - s1_bdt_70d (All)'] = paired_family_bootstrap_delta(
                    pi[:, 2], p_bdt70[:, 2], pi, p_bdt70, y_val, fam_codes, np.ones(len(y_val), dtype=bool),
                    n_boot=1000, seed=20263250,
                )

    final_doc = {
        'UTC': datetime.now(timezone.utc).isoformat(),
        'split_manifest': str(SPLITS_DIR),
        'internal_val_eval_events': int(len(y_val)),
        'models': models_report,
        'paired_family_bootstrap_1000_seed20263250': paired_boot,
    }
    write_json(EXP_ROOT / 'stage1_dual_regime_results.json', final_doc)
    print(f'[aggregate] Wrote {EXP_ROOT / "stage1_dual_regime_results.json"}', flush=True)
    return final_doc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=('gate-ab', 'train-neural', 'train-sec13', 'aggregate'))
    parser.add_argument('--variant', default='s1_dual_regime_modalnet')
    args = parser.parse_args()

    with SealAuditGuard() as guard:
        if args.mode == 'gate-ab':
            run_gate_ab()
        elif args.mode == 'train-neural':
            run_train_neural(args.variant)
        elif args.mode == 'train-sec13':
            run_section13_controls()
        elif args.mode == 'aggregate':
            run_aggregate()
        if guard.violations:
            raise RuntimeError(f'Seal violations recorded: {guard.violations}')


if __name__ == '__main__':
    main()
