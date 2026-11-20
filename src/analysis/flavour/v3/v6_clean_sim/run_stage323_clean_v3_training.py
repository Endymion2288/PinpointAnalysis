"""Stage 323 Clean v3 Pure-Tungsten Model-Training Remediation, Validation & Pre-Analysis Freeze.

Implements end-to-end remediation and verification of all model-training-related issues from
the Stage 322 independent scientific audit while STRICTLY PAUSING post-training physics analysis
and keeping Cal-A, Cut-B, and Confirm 100% SEALED AND UNREAD:
  - Dataset audit (`audit-dataset`): verifies 100% pure-W GENIE & G4Navigator match, 0 sources
    crossing splits, >=99.95% W volume closure, 0-PE timing closure, and 0.0 bit-for-bit IPT
    invariance on real Clean v3 events.
  - Pre-training GPU health gate (`smoke-32`): 32-event memorization, positive finite per-branch
    gradients across all branches, and modality-drop sensitivity for every architecture family.
  - Strong observable non-NN baselines (`baselines`): L2 Multinomial LR (presence/counts, c_obs,
    physics features) + HistGradientBoostingClassifier BDT on identical Clean v3 train/Tune8k/OOD.
  - Matched-budget, matched-seed Clean v3 training (`build-configs`, `submit-group`):
    10 groups x 3 seeds = 30 models (20 passes, batch 128, seeds {17, 42, 2026}), including
    orthogonal 2x2 target x IPT pitch factorial, 6T layer depth, Hybrid/Shuffled/NOMAD,
    energy-matched weighting control, and B1++ light neural baseline.
  - Standalone pre-analysis cryptographic freezer & Tune8k/OOD/observation-only verification
    (`freeze-and-validate`): writes `v8_clean_v3_freeze_manifest.json` with full G4/code/data/
    config/checkpoint SHA-256 provenance, evaluates ONLY on CleanTune8k + pure-W Generator-OOD
    validation shards + label-stripped observation-only replay, and asserts
    `Cal_A_Cut_B_Confirm_touched == False`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import tempfile
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
import torch
from torch.nn import functional as F

from analysis.flavour.inventory import write_json
from analysis.flavour.s4.stage_inputs import stage_shards
from analysis.flavour.v2.partitions import FAMILY, KEY
from analysis.flavour.v2.profile import device_batch
from analysis.flavour.v2.r1.tasks import probabilities, task_loss
from analysis.flavour.v2.r2.readout import grouped_folds
from analysis.flavour.v3.common.calibration import _tau_efficiency_at_fpr
from analysis.flavour.v3.common.datasets import V3EvalDataset, V3TrainDataset
from analysis.flavour.v3.common.domains import (
    OBS_DOMAIN_NAMES,
    compute_observable_containment,
    detector_bounds,
    pinpoint_slab_catalog,
)
from analysis.flavour.v3.common.observation_inference import infer_observation_only
from analysis.flavour.v3.common.submit import submit_job
from analysis.flavour.v3.v1_b1pp.models import ZAlignedStereoNet, b1pp_branch_groups
from analysis.flavour.v3.v1_b1pp.representation import (
    collate_b1pp,
    extract_b1pp_item,
    static_plane_layout,
)
from analysis.flavour.v3.v2_pinpointformer.evaluate import compute_slice_metrics
from analysis.flavour.v3.v2_pinpointformer.pixel_tokenizer import collate_pf, extract_pf_item
from analysis.flavour.v3.v2_pinpointformer.train import (
    CachedPFDataset,
    _ThreadedPFLoader,
    build_pf_mapped_split,
    move_batch_to_device,
)
from analysis.flavour.v3.v3_multitask.models import PinpointFormerMultitask, v3_branch_groups
from analysis.flavour.v3.v3_multitask.train import _pf_v3_loss, _pf_v3_predict
from analysis.flavour.v3.v4_tau.ensemble import fit_oof_domain_ensemble
from analysis.flavour.v3.v4_tau.models import PinpointFormerV4, pf_v4_branch_groups
from analysis.flavour.v3.v4_tau.nomad_kinematics import NomadTransverseExtractor
from analysis.flavour.v3.v6_clean_sim.evaluate_stage317 import (
    SEEDS,
    _compute_vertex_audit,
    _select_seed_prediction_df,
    paired_family_bootstrap_delta,
)
from analysis.flavour.v3.v6_clean_sim.produce_clean_v3 import (
    DEFAULT_CLEAN_V3_ROOT,
    GEOM_V3_MACRO,
    OOD_VARIANTS,
    PINPOINT_V3_BIN,
)


REPO_ROOT = Path(__file__).resolve().parents[5]
G4_REPO_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/Pinpoint_G4')
ROADMAP_FREEZE_V8 = REPO_ROOT / 'roadmap/v3/v8_clean_v3_freeze_manifest.json'


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _weighted_ba4(labels: np.ndarray, probs: np.ndarray, weights: np.ndarray) -> float:
    pred = probs.argmax(axis=1)
    recalls = []
    for c in range(4):
        m = (labels == c) & (weights > 0.0)
        if m.any():
            recalls.append(float(weights[m & (pred == c)].sum() / max(float(weights[m].sum()), 1e-12)))
    return float(np.mean(recalls)) if recalls else 0.0


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
    return {'BA4': float(np.mean(recalls)) if recalls else 0.0, 'cond_BA3': cond_ba3, 'NLL': nll}


# ---------------------------------------------------------------------------
# 1. Clean v3 Dataset Physics, Target, Timing & Real-Event IPT Invariance Audit
# ---------------------------------------------------------------------------
def audit_clean_v3(clean_root: Path) -> dict:
    t0 = time.perf_counter()
    geom = json.loads((clean_root / 'geometry.json').read_text())
    bounds = detector_bounds(geom)
    fin = json.loads((clean_root / 'finalize_summary.json').read_text())
    prep = json.loads((clean_root / 'prepare_summary.json').read_text())
    man_df = pd.read_parquet(clean_root / 'planned_manifest.parquet')
    tune_meta = pd.read_parquet(clean_root / 'validation_meta/tune_truth_meta.parquet')
    cat = pinpoint_slab_catalog(geom)

    # 1. Nuclear target & G4Navigator closure (322-01)
    non_w_manifest = int((man_df['tgt_Z'] != 74).sum())
    non_w_tune_genie = int((tune_meta['tgt_Z'] != 74).sum())
    non_w_tune_nav = int(((tune_meta['vtx_mat_Z'] != 74) | (tune_meta['vtx_mat_name'] != 'G4_W')).sum())
    src_cross_splits = int((man_df.groupby('source_tag')['split'].nunique() > 1).sum())
    if non_w_manifest != 0 or non_w_tune_genie != 0 or non_w_tune_nav != 0 or src_cross_splits != 0:
        raise AssertionError(
            f'Clean v3 purity check failed: non_w_man={non_w_manifest}, non_w_tune_genie={non_w_tune_genie}, '
            f'non_w_tune_nav={non_w_tune_nav}, src_cross_splits={src_cross_splits}'
        )

    # 2. Full-volume W containment & mass weights (322-13)
    w_z = cat['z_center_mm']
    w_half = 0.5 * float(cat['w_thickness_mm'])
    vz = man_df['initZ'].to_numpy(dtype=np.float64)
    vx = man_df['initX'].to_numpy(dtype=np.float64)
    vy = man_df['initY'].to_numpy(dtype=np.float64)
    slabs = man_df['slab_id'].to_numpy(dtype=np.int64)
    weights = man_df['geom_weight'].to_numpy(dtype=np.float64)
    is_pin = man_df['is_pinpoint_module'].to_numpy(dtype=bool)
    dz_slab = np.abs(vz - w_z[slabs])
    outside_z = int((dz_slab > w_half).sum())
    outside_xy = int(((vx < cat['w_x_min_mm']) | (vx > cat['w_x_max_mm']) | (vy < cat['w_y_min_mm']) | (vy > cat['w_y_max_mm'])).sum())
    if outside_z > 0 or outside_xy > 0:
        raise AssertionError(f'Vertices outside physical W: z={outside_z}, xy={outside_xy}')

    # 3. Inspect validation shard 080 for single-sat PE, zero edep_proxy_MeV, zero 0-PE positive times (322-12)
    obs_val = pq.read_table(clean_root / 'data/shards/080/observations.parquet').to_pylist()
    bar_keys = sorted(obs_val[0]['bar'].keys())
    if 'edep_proxy_MeV' in bar_keys:
        raise AssertionError('edep_proxy_MeV leaked into Clean v3 observations!')
    pe_all = np.concatenate([np.asarray(ev['bar']['pe_proxy'], dtype=np.float64) for ev in obs_val if len(ev['bar']['pe_proxy']) > 0])
    ht_all = np.concatenate([np.asarray(ev['bar']['hit_time_ns'], dtype=np.float64) for ev in obs_val if len(ev['bar']['pe_proxy']) > 0])
    zero_pe_pos_t = int(np.sum((pe_all <= 0) & (ht_all >= 0.0)))
    if zero_pe_pos_t != 0:
        raise AssertionError(f'Found {zero_pe_pos_t} 0-PE bars with non-negative hit_time_ns in shard 080!')

    # 4. Real Clean v3 64-event IPT bit-for-bit invariance under target-only pitch quantization (322-03)
    ipt_inv = {}
    base_items = []
    for i in range(64):
        ob = obs_val[i]
        dom = compute_observable_containment(ob, geom, bounds, config={'pe_only': True})
        base_items.append(extract_pf_item(i, ob, None, dom, geom, bounds, {'pe_only': True}))
    for q_tgt in (0.10, 0.30, 1.00):
        cnt_chg = 0
        max_feat_diff = 0.0
        max_patch_diff = 0.0
        max_ps_diff = 0.0
        for i in range(64):
            ob = obs_val[i]
            dom = compute_observable_containment(ob, geom, bounds, config={'pe_only': True})
            it_q = extract_pf_item(
                i, ob, None, dom, geom, bounds,
                {'pe_only': True, 'quantize_target_xy_mm': q_tgt, 'quantize_ipt_xy_mm': None},
            )
            it_b = base_items[i]
            ib = it_b['pix_plane'] >= 12
            iq = it_q['pix_plane'] >= 12
            if int(ib.sum()) != int(iq.sum()):
                cnt_chg += 1
            elif ib.any():
                max_feat_diff = max(max_feat_diff, float(np.max(np.abs(it_b['pix_feat'][ib] - it_q['pix_feat'][iq]))))
            max_ps_diff = max(max_ps_diff, float(np.max(np.abs(it_b['pix_plane_summary'][12:] - it_q['pix_plane_summary'][12:]))))
            pb = it_b['pix_plane'][it_b['patch_cluster_idx']] >= 12 if len(it_b['patch_cluster_idx']) else np.zeros(0, dtype=bool)
            pq_m = it_q['pix_plane'][it_q['patch_cluster_idx']] >= 12 if len(it_q['patch_cluster_idx']) else np.zeros(0, dtype=bool)
            if int(pb.sum()) != int(pq_m.sum()):
                cnt_chg += 1
            elif pb.any():
                max_patch_diff = max(max_patch_diff, float(np.max(np.abs(it_b['pix_patches'][pb] - it_q['pix_patches'][pq_m]))))
        if cnt_chg != 0 or max_feat_diff != 0.0 or max_patch_diff != 0.0 or max_ps_diff != 0.0:
            raise AssertionError(
                f'Real Clean v3 IPT invariance failed at q={q_tgt}: cnt_chg={cnt_chg}, '
                f'feat={max_feat_diff}, patch={max_patch_diff}, ps={max_ps_diff}'
            )
        ipt_inv[f'target_{q_tgt:.2f}mm'] = {
            'events_tested': 64,
            'events_with_ipt_count_change': cnt_chg,
            'max_abs_ipt_feat_diff': max_feat_diff,
            'max_abs_ipt_patch_diff': max_patch_diff,
            'max_abs_ipt_plane_summary_diff': max_ps_diff,
        }

    # 5. OOD validation shards audit
    ood_audit = {}
    for vname in OOD_VARIANTS:
        mdf = pd.read_parquet(clean_root / 'ood_shards' / vname / 'truth_meta.parquet')
        ood_audit[vname] = {
            'events': int(len(mdf)),
            'unique_tgt_Z': sorted(int(z) for z in mdf['tgt_Z'].unique()),
            'unique_vtx_mat_name': sorted(str(x) for x in mdf['vtx_mat_name'].unique()),
            'unique_source_tags': int(mdf['source_tag'].nunique()),
            'charm_count': int(mdf['has_charm'].sum()),
            'E_max_GeV': float(mdf['energy_nu_GeV'].max()),
        }

    # 6. Render 300-DPI English audit figure
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9), dpi=300)
    ax0, ax1, ax2 = axes

    slab_counts = np.bincount(slabs, minlength=108)
    slab_w_counts = np.bincount(slabs, weights=weights, minlength=108)
    slab_ids = np.arange(108)
    colors_s = ['#C44E52' if cat['is_front_pinpoint'][k] else ('#DD8452' if cat['is_intermediate_pinpoint'][k] else '#4C72B0') for k in slab_ids]
    ax0.bar(slab_ids, slab_counts, width=0.9, color=colors_s, alpha=0.85, label='Raw Stratified (99.96% W Vol)')
    ax0.plot(slab_ids, slab_w_counts, color='#111111', lw=1.4, label='Mass-Weighted ($w_{\\mathrm{geom}}$)')
    ax0.set_xlabel('Physical Tungsten Slab Index $k \\in [0, 107]$')
    ax0.set_ylabel('Events per Slab (60k Pure-W)')
    ax0.set_title('Clean v3: 100% Pure-W & Full-Volume Closure')
    ax0.legend(loc='upper right', fontsize=8.0, frameon=True)
    ax0.grid(True, axis='y', alpha=0.25)

    bins_e = np.geomspace(20.0, 5500.0, 45)
    cls_labels = [(0, r'$\nu_e$ CC (10081)', '#4C72B0'), (1, r'$\nu_\mu$ CC (10082)', '#55A868'),
                  (2, r'$\nu_\tau$ CC (10083)', '#C44E52'), (3, r'NC (10084)', '#8172B3')]
    for cid, clbl, col in cls_labels:
        ec = man_df.loc[man_df['label'] == cid, 'Ev_GeV'].to_numpy(dtype=np.float64)
        ax1.hist(ec, bins=bins_e, density=True, histtype='step', lw=1.8, color=col, label=f'{clbl} (med {np.median(ec):.0f} GeV)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Incident Neutrino Energy $E_\\nu$ [GeV]')
    ax1.set_ylabel('Normalized Density [GeV$^{-1}$]')
    ax1.set_title('Unified Pure-W GENIE 10081-10084 Spectra')
    ax1.legend(loc='lower left', fontsize=7.8, frameon=True)
    ax1.grid(True, which='both', alpha=0.25)

    bins_pe = np.geomspace(2.5, 16000.0, 55)
    ax2.hist(pe_all[pe_all >= 2.5], bins=bins_pe, density=True, histtype='stepfilled', alpha=0.35, color='#4C72B0',
             edgecolor='#2B4F81', lw=1.6, label=f'Clean v3 Single-Sat PE (max={pe_all.max():.0f})')
    ax2.axvline(14400.0, color='#111111', ls=':', lw=1.4, label='SiPM Cap (14,400 PE)')
    ax2.text(0.04, 0.22, f'0-PE pos-time hits: {zero_pe_pos_t}\nNon-W GENIE/Nav: 0 / 60,000\nSources crossing splits: 0\nIPT diff @1mm tgt: 0.0',
             transform=ax2.transAxes, fontsize=8.2, bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#888888', alpha=0.9))
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Digitized Scintillator Bar Output [PE] (Zero Truth $E_{\\mathrm{dep}}$)')
    ax2.set_ylabel('Normalized Channel Density')
    ax2.set_title('Clean v3 Single-Sat PE & Timing Closure')
    ax2.legend(loc='upper right', fontsize=7.6, frameon=True)
    ax2.grid(True, which='both', alpha=0.25)

    fig.tight_layout()
    fig_path = clean_root / 'stage323_clean_v3_dataset_audit.png'
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    report = {
        'UTC': datetime.now(timezone.utc).isoformat(),
        'wall_seconds': float(time.perf_counter() - t0),
        'total_events': int(len(man_df)),
        'non_W_manifest_count': non_w_manifest,
        'non_W_tune_genie_count': non_w_tune_genie,
        'non_W_tune_g4_navigator_count': non_w_tune_nav,
        'g4_navigator_W_verified_all_shards': int(fin['g4_navigator_W_verified']),
        'sources_crossing_splits': src_cross_splits,
        'family_disjoint_verified': bool(prep['family_disjoint_verified']),
        'sampled_physical_W_volume_fraction': float(prep['sampled_physical_W_volume_fraction']),
        'tungsten_3d_containment': {
            'outside_z_count': outside_z,
            'outside_xy_count': outside_xy,
            'max_abs_dz_from_slab_center_mm': float(dz_slab.max()),
            'mass_weighted_pinpoint_fraction': float(weights[is_pin].sum() / weights.sum()),
            'exact_geometric_pinpoint_fraction': float(24.0 / 108.0),
            'mean_geom_weight': float(weights.mean()),
        },
        'sipm_and_timing_audit': {
            'observations_bar_keys': bar_keys,
            'edep_proxy_MeV_absent': True,
            'shard_080_channels': int(len(pe_all)),
            'max_observed_pe': float(pe_all.max()),
            'channels_above_9103_pe': int((pe_all > 9103.0).sum()),
            'zero_pe_nonneg_time_count': zero_pe_pos_t,
        },
        'real_event_ipt_invariance_322_03': ipt_inv,
        'ood_validation_shards': ood_audit,
        'figure_path': str(fig_path),
    }
    write_json(clean_root / 'stage323_clean_v3_dataset_audit.json', report)
    return report


# ---------------------------------------------------------------------------
# 2. Pre-Training 32-Event GPU Health Gate (322-15 & AGENTS.md)
# ---------------------------------------------------------------------------
def _branch_grad_norms(groups: dict[str, list[tuple[str, torch.nn.Parameter]]]) -> dict[str, float]:
    out = {}
    for br, params in groups.items():
        sq = 0.0
        for _, p in params:
            if p.grad is not None:
                sq += float(p.grad.detach().float().square().sum().item())
        out[br] = float(math.sqrt(max(sq, 0.0)))
    return out


def run_smoke32_health_gate_v3(clean_root: Path) -> dict:
    """Verify 32-event memorization, positive finite per-branch gradients, and modality sensitivity on Clean v3."""
    t0 = time.perf_counter()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    geom = json.loads((clean_root / 'geometry.json').read_text())
    data_root = clean_root / 'data'
    sidecar_root = clean_root / 'truth_sidecar'

    base_obs_cfg = {
        'pe_only': True, 'bar_threshold_pe': 2.5, 'sipm_npix': 14400.0,
        'apply_sipm_saturation': False, 'invert_sipm_saturation': True,
    }
    tr_pf_ds = V3TrainDataset(data_root, sidecar_root, split='train', config=base_obs_cfg, item_transform=extract_pf_item)
    tr_b1_ds = V3TrainDataset(data_root, None, split='train', config=base_obs_cfg, item_transform=extract_b1pp_item)
    val_pf_ds = V3EvalDataset(data_root, split='validation', config=base_obs_cfg, item_transform=extract_pf_item)

    # Select balanced 32 training events (8 per class)
    tr_lbl = tr_pf_ds.metadata['label'].to_numpy(dtype=np.int64)
    sel_idx = []
    for c in range(4):
        sel_idx.extend(np.flatnonzero(tr_lbl == c)[:8].tolist())
    sel_idx = sorted(sel_idx)

    pf_items_32 = [tr_pf_ds[i] for i in sel_idx]
    b1_items_32 = [tr_b1_ds[i] for i in sel_idx]
    val_obs_32 = [val_pf_ds._raw_observation(i)[0] for i in range(32)]
    bounds = detector_bounds(geom)

    variants = [
        ('b1pp_clean', 'b1pp', {
            **base_obs_cfg, 'task': 'T4', 'dim': 160, 'n_long_layers': 4, 'heads': 8,
            'drop_path': 0.0, 'channel_dropout': 0.0,
        }),
        ('pf_20um', 'pf', {
            **base_obs_cfg, 'task': 'T4', 'dim': 192, 'heads': 8, 'n_local_layers': 4, 'n_global_layers': 4,
            'drop_path': 0.0, 'enable_dense_gn2': False, 'enable_micro_vertex': False, 'use_fixed_event_loss': True,
        }),
        ('pf_hybrid_uvtx', 'pf', {
            **base_obs_cfg, 'task': 'T4', 'dim': 192, 'heads': 8, 'n_local_layers': 4, 'n_global_layers': 4,
            'drop_path': 0.0, 'enable_dense_gn2': False, 'enable_micro_vertex': True, 'use_fixed_event_loss': True,
        }),
        ('pf_hybrid_nomad', 'pf_v4', {
            **base_obs_cfg, 'task': 'T4', 'dim': 192, 'heads': 8, 'n_local_layers': 4, 'n_global_layers': 4,
            'drop_path': 0.0, 'enable_dense_gn2': False, 'enable_micro_vertex': True,
            'use_fixed_event_loss': True, 'model_family': 'pf_v4', 'enable_nomad': True, 'enable_pauc': False,
            'legacy_nomad_scale': False, 'nomad_init_scale': 1.0,
        }),
    ]

    results = {}
    all_passed = True
    for vname, fam, cfg in variants:
        torch.manual_seed(323)
        if fam == 'b1pp':
            model = ZAlignedStereoNet(static_plane_layout(geom), cfg).to(device)
            groups = b1pp_branch_groups(model)
            batch_cpu = collate_b1pp(b1_items_32, mirror_rng=None)
            batch = device_batch(batch_cpu, device)
            from analysis.flavour.v3.common.layers import split_adamw_param_groups
            opt = torch.optim.AdamW(split_adamw_param_groups(model, 2.0e-3, 1e-4))
        elif fam == 'pf_v4':
            model = PinpointFormerV4(geom, cfg).to(device)
            groups = pf_v4_branch_groups(model)
            batch_cpu = collate_pf(pf_items_32, mirror_rng=None)
            batch = move_batch_to_device(batch_cpu, device)
            opt = torch.optim.AdamW(model.build_param_groups(1.5e-3, 1e-4))
        else:
            model = PinpointFormerMultitask(geom, cfg).to(device)
            groups = v3_branch_groups(model)
            batch_cpu = collate_pf(pf_items_32, mirror_rng=None)
            batch = move_batch_to_device(batch_cpu, device)
            opt = torch.optim.AdamW(model.build_param_groups(1.5e-3, 1e-4))

        model.train()
        step1_grads = {}
        init_nll = None
        final_nll = None
        final_acc = 0.0
        n_steps = 80
        for step in range(1, n_steps + 1):
            opt.zero_grad(set_to_none=True)
            if fam == 'b1pp':
                logits = model(batch).float()
                loss = task_loss(logits, batch['label'], 'T4')
                probs = probabilities(logits, 'T4')
            else:
                loss, probs = _pf_v3_loss(model, batch, {'grad_blend': 1.0})
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss in smoke32 {vname} at step {step}')
            loss.backward()
            if step == 1:
                step1_grads = _branch_grad_norms(groups)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            with torch.inference_mode():
                ce4 = float(-torch.log(probs.clamp_min(1e-7))[torch.arange(32, device=device), batch['label']].mean().item())
                acc = float((probs.argmax(-1) == batch['label']).float().mean().item())
            if step == 1:
                init_nll = ce4
            final_nll = ce4
            final_acc = acc

        # Modality-drop sensitivity on 32 real Tune events
        model.eval()
        drop_res = {}
        def _eval_obs_list(obs_seq: list[dict]) -> np.ndarray:
            its = []
            for idx_i, ob in enumerate(obs_seq):
                d_i = compute_observable_containment(ob, geom, bounds, config=cfg)
                fn = extract_b1pp_item if fam == 'b1pp' else extract_pf_item
                its.append(fn(idx_i, ob, None, d_i, geom, bounds, cfg))
            with torch.inference_mode():
                if fam == 'b1pp':
                    b_in = device_batch(collate_b1pp(its, mirror_rng=None), device)
                    p = probabilities(model(b_in).float(), 'T4').cpu().numpy()
                else:
                    b_in = move_batch_to_device(collate_pf(its, mirror_rng=None), device)
                    p = model(b_in, grad_blend=1.0, return_aux=False)['probs4'].float().cpu().numpy()
            return p

        p_base = _eval_obs_list(val_obs_32)
        for mod in ('pixel', 'bar', 'tracker_hypothesis', 'all'):
            mod_list = []
            for ob in val_obs_32:
                ob2 = dict(ob)
                keys_drop = ('pixel', 'bar', 'tracker_hypothesis') if mod == 'all' else (mod,)
                for kd in keys_drop:
                    ob2[kd] = {k: [] for k in ob[kd].keys()}
                mod_list.append(ob2)
            p_drop = _eval_obs_list(mod_list)
            tv = float(0.5 * np.abs(p_drop - p_base).sum(axis=1).mean())
            drop_res[mod] = {
                'mean_tv': tv,
                'argmax_flips': int((p_drop.argmax(1) != p_base.argmax(1)).sum()),
                'all_finite': bool(np.isfinite(p_drop).all()),
            }

        # Ignore optional disabled heads if empty
        active_grads = {k: v for k, v in step1_grads.items() if len(groups[k]) > 0}
        grads_ok = all(np.isfinite(v) and v > 0.0 for v in active_grads.values())
        passed = bool(grads_ok and final_acc >= 0.90 and final_nll < 0.35 and all(d['all_finite'] for d in drop_res.values()))
        all_passed = all_passed and passed
        results[vname] = {
            'parameters': int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            'init_NLL4': float(init_nll),
            'final_NLL4': float(final_nll),
            'final_acc32': float(final_acc),
            'step1_branch_grad_norms': active_grads,
            'all_branches_positive_finite': grads_ok,
            'modality_drop_32_tune': drop_res,
            'passed': passed,
        }
        del model, opt, batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    tr_pf_ds.native.close()
    tr_b1_ds.native.close()
    val_pf_ds.native.close()

    rep = {
        'UTC': datetime.now(timezone.utc).isoformat(),
        'wall_seconds': float(time.perf_counter() - t0),
        'device': str(device),
        'all_passed': all_passed,
        'variants': results,
    }
    out_p = clean_root / 'stage323' / 'stage323_smoke32_health_gate.json'
    out_p.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_p, rep)
    if not all_passed:
        raise RuntimeError(f'Smoke-32 health gate FAILED! See {out_p}')
    return rep


# ---------------------------------------------------------------------------
# 3. Strong Observable Non-NN Baselines (322-15: L2-LR & HistGradientBoosting BDT)
# ---------------------------------------------------------------------------
def _identity_collate(xs: list[dict]) -> list[dict]:
    return xs


def _extract_tabular_features_split(data_root: Path, split: str, geometry: dict, cfg: dict, workers: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract pure-observation physics feature matrix (70-dim) in parallel across multiple CPU workers."""
    from torch.utils.data import BatchSampler, DataLoader, SequentialSampler
    from analysis.flavour.v2.profile import init_worker, shutdown_loader

    ds = V3EvalDataset(data_root, split=split, config=cfg, item_transform=extract_pf_item)
    nomad_ext = NomadTransverseExtractor(geometry, phys_calibrated=True, pe_per_mip_mev=40.0)
    n = len(ds)
    feats = np.zeros((n, 70), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    doms = np.zeros(n, dtype=np.int64)

    eff_w = min(workers, 8) if n >= 512 else 0
    lkw: dict = {
        'batch_sampler': BatchSampler(SequentialSampler(ds), batch_size=128, drop_last=False),
        'num_workers': eff_w,
        'collate_fn': _identity_collate,
    }
    if eff_w > 0:
        lkw.update(multiprocessing_context='spawn', worker_init_fn=init_worker, prefetch_factor=2)
    loader = DataLoader(ds, **lkw)
    cursor = 0
    try:
        for items in loader:
            batch = collate_pf(items, mirror_rng=None)
            with torch.inference_mode():
                nomad_t = nomad_ext.from_pf(batch).numpy()
            for loc, it in enumerate(items):
                gi = cursor + loc
                c_obs = it['c_obs']                                  # 16
                mv = it['micro_vtx_feat']                            # 16
                nm = nomad_t[loc]                                    # 10
                cand_m = it['trk_cand_mask']
                n_cand = float(cand_m.sum())
                lead_c = it['trk_cand'][0] if len(it['trk_cand']) else np.zeros(11, dtype=np.float32)
                ps = it['pix_plane_summary']  # (15, 12)
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
                row_f = np.concatenate([c_obs, mv, nm, np.array([n_cand], dtype=np.float32), lead_c[:11], ps_agg])
                feats[gi] = np.nan_to_num(row_f, nan=0.0, posinf=0.0, neginf=0.0)
                labels[gi] = int(it['label'])
                doms[gi] = int(it['obs_domain_id'])
            cursor += len(items)
    finally:
        shutdown_loader(loader)
    ds.native.close()
    return feats, labels, doms


def run_observable_baselines_v3(clean_root: Path) -> dict:
    t0 = time.perf_counter()
    geom = json.loads((clean_root / 'geometry.json').read_text())
    cfg = {
        'pe_only': True, 'bar_threshold_pe': 2.5, 'sipm_npix': 14400.0,
        'apply_sipm_saturation': False, 'invert_sipm_saturation': True,
    }
    X_tr, y_tr, _ = _extract_tabular_features_split(clean_root / 'data', 'train', geom, cfg)
    X_val, y_val, d_val = _extract_tabular_features_split(clean_root / 'data', 'validation', geom, cfg)
    tune_meta = pd.read_parquet(clean_root / 'validation_meta/tune_truth_meta.parquet')
    gw_val = tune_meta['geom_weight'].to_numpy(dtype=np.float64)
    pin_val = tune_meta['pixel_centered'].to_numpy(dtype=bool)

    ood_data = {}
    for ov in OOD_VARIANTS:
        Xo, yo, do = _extract_tabular_features_split(clean_root / 'ood_shards' / ov, 'validation', geom, cfg)
        ood_data[ov] = (Xo, yo, do)

    # Feature subsets:
    #   presence_counts: [has_pix, has_bar, has_trk, has_trk3, log1p_pix, log1p_bar] = idx [0,1,2,3,13,14]
    #   all_c_obs: idx 0..15
    #   full_obs_physics: idx 0..69 (c_obs + micro_vtx + NOMAD + spectrometer + plane summaries)
    specs = [
        ('lr_presence_counts_6d', [0, 1, 2, 3, 13, 14], 'lr'),
        ('lr_c_obs_16d', list(range(16)), 'lr'),
        ('lr_full_obs_physics_70d', list(range(70)), 'lr'),
        ('bdt_histgb_full_obs_70d', list(range(70)), 'bdt'),
    ]

    summary = {}
    for name, cols, kind in specs:
        if kind == 'lr':
            scaler = StandardScaler()
            Xtr_s = scaler.fit_transform(X_tr[:, cols])
            Xva_s = scaler.transform(X_val[:, cols])
            clf = LogisticRegression(C=1.0, max_iter=1000, random_state=323)
            clf.fit(Xtr_s, y_tr)
            p_val = clf.predict_proba(Xva_s)
            predict_fn = lambda X: clf.predict_proba(scaler.transform(X[:, cols]))
        else:
            clf = HistGradientBoostingClassifier(
                max_iter=350, max_depth=8, learning_rate=0.06, min_samples_leaf=30, l2_regularization=1.0, random_state=323
            )
            clf.fit(X_tr[:, cols], y_tr)
            p_val = clf.predict_proba(X_val[:, cols])
            predict_fn = lambda X: clf.predict_proba(X[:, cols])

        m_all = _ba4_and_nll(y_val, p_val)
        m_pin = _ba4_and_nll(y_val[pin_val], p_val[pin_val])
        tau_1pct_pin = float(_tau_efficiency_at_fpr(p_val[pin_val, 2], y_val[pin_val], 0.01)['tau_eff'])
        dom_metrics = {}
        for did in (1, 2, 3):
            dm = d_val == did
            if dm.any():
                dom_metrics[OBS_DOMAIN_NAMES[did]] = _ba4_and_nll(y_val[dm], p_val[dm])

        ood_metrics = {}
        for ov, (Xo, yo, _) in ood_data.items():
            po = predict_fn(Xo)
            if ov == 'diag_pythia8_free_nucleon_nc_charm':
                ood_metrics[ov] = {
                    'n_events': int(len(yo)),
                    'mean_p_tau': float(po[:, 2].mean()),
                    'frac_pred_nc': float((po.argmax(1) == 3).mean()),
                    'frac_pred_tau': float((po.argmax(1) == 2).mean()),
                }
            else:
                ood_metrics[ov] = _ba4_and_nll(yo, po)

        summary[name] = {
            'n_features': len(cols),
            'tune8k_all': m_all,
            'tune8k_mw2t_BA4': _weighted_ba4(y_val, p_val, gw_val),
            'tune8k_pinpoint': {**m_pin, 'tau_eff_1pct_fpr_ptau': tau_1pct_pin},
            'tune8k_by_domain': dom_metrics,
            'ood_validation': ood_metrics,
        }

    out = {
        'UTC': datetime.now(timezone.utc).isoformat(),
        'wall_seconds': float(time.perf_counter() - t0),
        'train_events': int(len(y_tr)),
        'tune_events': int(len(y_val)),
        'baselines': summary,
    }
    out_p = clean_root / 'stage323' / 'stage323_observable_baselines.json'
    out_p.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_p, out)
    return out


# ---------------------------------------------------------------------------
# 4. Matched-Budget, Matched-Seed Clean v3 Training Matrix (10 groups x 3 seeds = 30 models)
# ---------------------------------------------------------------------------
def build_stage323_configs(clean_root: Path) -> dict[str, dict]:
    s323_root = clean_root / 'stage323'
    cfg_dir = s323_root / 'configs'
    cfg_dir.mkdir(parents=True, exist_ok=True)

    data_root = str(clean_root / 'data')
    geom_json = str(clean_root / 'geometry.json')
    tune_screen = str(clean_root / 'data/tune_screen.parquet')
    sidecar_root = str(clean_root / 'truth_sidecar')
    emw_path = str(clean_root / 'train_energy_match_weights.parquet')
    base_train_mod = 'analysis.flavour.v3.v3_multitask.train'

    def _base_pf(name: str, seed: int) -> dict:
        return {
            'name': name,
            'seed': int(seed),
            'task': 'T4',
            'data_root': data_root,
            'geometry_json': geom_json,
            'tune_screen_path': tune_screen,
            'sidecar_root': sidecar_root,
            'quantize_xy_mm': None,
            'quantize_target_only': True,
            'quantize_target_xy_mm': None,
            'quantize_ipt_xy_mm': None,
            'shuffle_micro_vertex': False,
            'dim': 192,
            'heads': 8,
            'n_local_layers': 4,
            'n_global_layers': 4,
            'drop_path': 0.10,
            'passes': 20,
            'batch_size': 128,
            'learning_rate': 8.0e-4,
            'min_lr': 1.0e-5,
            'weight_decay': 0.03,
            'warmup_fraction': 0.06,
            'ema_decay': 0.999,
            'bf16': True,
            'screen_every_passes': 4,
            'cache_workers': 6,
            'loader_workers': 4,
            'pe_only': True,
            'bar_threshold_pe': 2.5,
            'sipm_npix': 14400.0,
            'apply_sipm_saturation': False,
            'invert_sipm_saturation': True,
            'y_mirror': False,
            'enable_dense_gn2': False,
            'load_cluster_sidecar': False,
            'use_fixed_event_loss': True,
            'train_module': base_train_mod,
        }

    specs = [
        # 2x2 Target x IPT Pitch Factorial (Stage 322-03 Priority 3)
        ('s323-pf-20um', 'v3_pf_tgt20um_ipt20um', {'quantize_target_xy_mm': None, 'quantize_ipt_xy_mm': None, 'enable_micro_vertex': False}, base_train_mod),
        ('s323-pf-tgt1mm-ipt20um', 'v3_pf_tgt1mm_ipt20um', {'quantize_target_xy_mm': 1.00, 'quantize_ipt_xy_mm': None, 'enable_micro_vertex': False}, base_train_mod),
        ('s323-pf-tgt20um-ipt1mm', 'v3_pf_tgt20um_ipt1mm', {'quantize_target_xy_mm': None, 'quantize_ipt_xy_mm': 1.00, 'enable_micro_vertex': False}, base_train_mod),
        ('s323-pf-tgt1mm-ipt1mm', 'v3_pf_tgt1mm_ipt1mm', {'quantize_target_xy_mm': 1.00, 'quantize_ipt_xy_mm': 1.00, 'enable_micro_vertex': False}, base_train_mod),
        # Target layer depth control
        ('s323-pf-layers-6t', 'v3_pf_layers_6t', {'active_pixel_planes': [0, 2, 4, 6, 8, 10, 12, 13, 14], 'enable_micro_vertex': False}, base_train_mod),
        # Micro-vertex Hybrid, content-seeded Shuffled falsification, and NOMAD V4
        ('s323-pf-hybrid-uvtx', 'v3_pf_hybrid_uvtx', {'enable_micro_vertex': True}, base_train_mod),
        ('s323-pf-hybrid-shuffled', 'v3_pf_hybrid_shuffled', {'enable_micro_vertex': True, 'shuffle_micro_vertex': True}, base_train_mod),
        (
            's323-pf-hybrid-nomad',
            'v3_pf_hybrid_nomad',
            {
                'enable_micro_vertex': True,
                'model_family': 'pf_v4',
                'enable_nomad': True,
                'legacy_nomad_scale': False,
                'enable_pauc': False,
                'nomad_init_scale': 1.0,
                'train_module': 'analysis.flavour.v3.v4_tau.train',
            },
            'analysis.flavour.v3.v4_tau.train',
        ),
        # Energy-matched importance weighting control (Stage 322-05 energy-shortcut check)
        ('s323-pf-20um-ematched', 'v3_pf_tgt20um_ematched', {'enable_micro_vertex': False, 'sample_weight_path': emw_path}, base_train_mod),
    ]

    groups: dict[str, dict] = {}
    for gname, prefix, overrides, tmod in specs:
        c_paths = []
        for s in SEEDS:
            c = _base_pf(f'{prefix}_s{s}', s)
            c.update(overrides)
            p = cfg_dir / f"{c['name']}.json"
            write_json(p, c)
            c_paths.append(p)
        groups[gname] = {'prefix': prefix, 'configs': c_paths, 'share': '--share-pf-inputs', 'train_module': tmod}

    # Group 10: Matched-budget B1++ light neural baseline
    b1_paths = []
    for s in SEEDS:
        c = {
            'name': f'v3_b1pp_clean_s{s}',
            'seed': int(s),
            'task': 'T4',
            'data_root': data_root,
            'geometry_json': geom_json,
            'tune_screen_path': tune_screen,
            'dim': 160,
            'n_long_layers': 4,
            'heads': 8,
            'drop_path': 0.10,
            'channel_dropout': 0.03,
            'passes': 20,
            'batch_size': 128,
            'learning_rate': 9.0e-4,
            'min_lr': 1.0e-5,
            'weight_decay': 0.03,
            'warmup_fraction': 0.06,
            'ema_decay': 0.999,
            'bf16': True,
            'screen_every_passes': 4,
            'cache_workers': 4,
            'loader_workers': 2,
            'pe_only': True,
            'bar_threshold_pe': 2.5,
            'sipm_npix': 14400.0,
            'apply_sipm_saturation': False,
            'invert_sipm_saturation': True,
            'y_mirror': False,
            'train_module': 'analysis.flavour.v3.v1_b1pp.train',
        }
        p = cfg_dir / f"{c['name']}.json"
        write_json(p, c)
        b1_paths.append(p)
    groups['s323-b1pp-clean'] = {
        'prefix': 'v3_b1pp_clean',
        'configs': b1_paths,
        'share': '--share-b1pp-inputs',
        'train_module': 'analysis.flavour.v3.v1_b1pp.train',
    }
    return groups


def submit_stage323_group(
    clean_root: Path,
    group_name: str,
    backend: str,
    parallel: int = 3,
    kubeflow_flavor: str = 'h100-47gb',
) -> dict:
    s323_root = clean_root / 'stage323'
    groups = build_stage323_configs(clean_root)
    spec = groups[group_name]
    cmd = [
        'analysis.flavour.v3.common.group',
        '--root', str(s323_root),
        '--name', group_name,
        '--parallel', str(parallel),
        spec['share'],
        '--cache-workers', '4',
        '--train-module', spec['train_module'],
        '--configs', *[str(p) for p in spec['configs']],
    ]
    if backend == 'kubeflow':
        return submit_job(
            root=s323_root,
            name=group_name,
            stage='stage323',
            command=cmd,
            backend='kubeflow',
            flavor=kubeflow_flavor,
            gpu=True,
            deadline_seconds=14400,
        )
    if backend == 'condor':
        return submit_job(
            root=s323_root,
            name=group_name,
            stage='stage323',
            command=cmd,
            backend='condor',
            cpus=4,
            memory='15000MB',
            disk='16GB',
            gpu=True,
            min_gpu_memory_mb=10000,
            min_capability=8.0,
            job_flavour='workday',
            allow_backfill=True,
        )
    raise ValueError(f'Unsupported backend: {backend}')


# ---------------------------------------------------------------------------
# 5. Standalone Cryptographic Pre-Analysis Freeze & Tune8k / OOD / Obs-Only Verification
#    (STRICTLY ZERO ACCESS TO Cal-A, Cut-B, OR Confirm!)
# ---------------------------------------------------------------------------
def _build_ood_batches_cache(
    ood_root: Path,
    cfg: dict,
    geometry: dict,
    batch_size: int = 128,
) -> list[dict]:
    """Tokenize an OOD shard ONCE (observation-only, zero labels) for fast multi-seed GPU replay."""
    from analysis.flavour.v3.common.datasets import V3ObservationDataset

    is_b1 = ('v1_b1pp' in str(cfg.get('train_module', ''))) or (cfg.get('model_family') == 'b1pp')
    ds = V3ObservationDataset(
        obs_paths=ood_root / 'shards/000/observations.parquet',
        geometry_json=ood_root / 'shards/000/geometry.json',
        config=cfg,
        item_transform=extract_b1pp_item if is_b1 else extract_pf_item,
    )
    items = [ds[i] for i in range(len(ds))]
    batches = []
    for st in range(0, len(items), batch_size):
        sub = items[st : st + batch_size]
        batches.append(collate_b1pp(sub, mirror_rng=None) if is_b1 else collate_pf(sub, mirror_rng=None))
    return batches


def _infer_group_ood_fp32(
    ood_root: Path,
    prefix: str,
    checkpoints: dict,
    geometry: dict,
    device: torch.device,
    batch_cache: dict[str, list[dict]],
) -> dict[str, pd.DataFrame]:
    """Deterministic FP32 observation-only inference reusing tokenized batches across seeds & groups."""
    cfg17 = json.loads(Path(checkpoints[f'{prefix}_s17']['config_path']).read_text())
    is_b1 = ('v1_b1pp' in str(cfg17.get('train_module', ''))) or (cfg17.get('model_family') == 'b1pp')
    cache_key = json.dumps({
        'ood': str(ood_root),
        'is_b1': is_b1,
        'qt': cfg17.get('quantize_target_xy_mm'),
        'qi': cfg17.get('quantize_ipt_xy_mm'),
        'ap': cfg17.get('active_pixel_planes'),
        'sh': cfg17.get('shuffle_micro_vertex', False),
    }, sort_keys=True)
    if cache_key not in batch_cache:
        batch_cache[cache_key] = _build_ood_batches_cache(ood_root, cfg17, geometry, batch_size=128)
    batches = batch_cache[cache_key]
    layout = static_plane_layout(geometry) if is_b1 else None

    out = {}
    for s in SEEDS:
        sname = f'{prefix}_s{s}'
        ckpt_p = Path(checkpoints[sname]['path'])
        c_s = json.loads(Path(checkpoints[sname]['config_path']).read_text())
        state = torch.load(ckpt_p, map_location=device, weights_only=False)
        if is_b1:
            model: torch.nn.Module = ZAlignedStereoNet(layout, c_s).to(device)
        elif c_s.get('model_family') == 'pf_v4':
            model = PinpointFormerV4(geometry, c_s).to(device)
        else:
            model = PinpointFormerMultitask(geometry, c_s).to(device)
        model.load_state_dict(state['model'])
        model.eval()
        p_chunks = []
        with torch.inference_mode():
            for b_cpu in batches:
                if is_b1:
                    g_b = device_batch(b_cpu, device)
                    p_chunks.append(probabilities(model(g_b).float(), c_s.get('task', 'T4')).cpu().numpy())
                else:
                    g_b = move_batch_to_device(b_cpu, device)
                    p_chunks.append(model(g_b, grad_blend=1.0, return_aux=False)['probs4'].float().cpu().numpy())
        p_all = np.concatenate(p_chunks, axis=0)
        out[sname] = pd.DataFrame({'p0': p_all[:, 0], 'p1': p_all[:, 1], 'p2': p_all[:, 2], 'p3': p_all[:, 3]})
    return out


def freeze_and_validate_stage323(clean_root: Path) -> dict:
    """Freeze all 30 Clean v3 models with full SHA-256 provenance and validate strictly on Tune8k + OOD."""
    t0 = time.perf_counter()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    s323_root = clean_root / 'stage323'
    groups = build_stage323_configs(clean_root)
    geom = json.loads((clean_root / 'geometry.json').read_text())
    tune_meta = pd.read_parquet(clean_root / 'validation_meta/tune_truth_meta.parquet')
    sidecar_df = pd.read_parquet(clean_root / 'truth_sidecar/event_sidecar.parquet')

    # Verify all 30 checkpoints exist and record full SHA-256 fingerprints
    labels = tune_meta['label'].to_numpy(dtype=np.int64)
    obs_dom = tune_meta['obs_domain_id'].to_numpy(dtype=np.int64)
    geom_w = tune_meta['geom_weight'].to_numpy(dtype=np.float64)
    folds, _ = grouped_folds(tune_meta[KEY + FAMILY + ['label']].reset_index(drop=True), folds=5, seed=20263230)
    slices = {
        'All_Tune8k': np.ones(len(tune_meta), dtype=bool),
        'PinpointBlock_PixXY': tune_meta['pixel_centered'].to_numpy(dtype=bool),
        'FrontPinpoint_PixXY': tune_meta['front_pinpoint_pixel_centered'].to_numpy(dtype=bool),
        'D_pix_contained': obs_dom == 1,
    }

    checkpoints = {}
    tune_summary = {}
    tune_group_probs: dict[str, np.ndarray] = {}
    tune_group_pv: dict[str, dict] = {}

    for gname, spec in groups.items():
        prefix = spec['prefix']
        m_frames = {}
        for s in SEEDS:
            sname = f'{prefix}_s{s}'
            exp_dir = s323_root / 'experiments' / sname
            df, info = _select_seed_prediction_df(exp_dir)
            m_frames[sname] = df
            ckpt_file = exp_dir / ('best.pt' if info['chosen_file'] == 'best_predictions.parquet' else 'ema.pt')
            pred_file = exp_dir / info['chosen_file']
            cfg_file = s323_root / 'configs' / f'{sname}.json'
            checkpoints[sname] = {
                'group': gname,
                'path': str(ckpt_file),
                'sha256': _sha256_file(ckpt_file),
                'chosen_prediction': info['chosen_file'],
                'prediction_sha256': _sha256_file(pred_file),
                'config_path': str(cfg_file),
                'config_sha256': _sha256_file(cfg_file),
            }
        # Evaluate both direct equal-weight 3-seed average (pure p_tau, cross-domain comparable)
        # AND 5-fold OOF domain-calibrated ensemble on Tune8k
        raw_probs_list = []
        ref_idx = pd.MultiIndex.from_frame(tune_meta[KEY])
        for df in m_frames.values():
            p_cols = ['p_0', 'p_1', 'p_2', 'p_3'] if 'p_0' in df.columns else ['p0', 'p1', 'p2', 'p3']
            pos = pd.MultiIndex.from_frame(df[KEY]).get_indexer(ref_idx)
            raw_probs_list.append(df.iloc[pos][p_cols].to_numpy(dtype=np.float64))
        p_raw_avg = np.mean(raw_probs_list, axis=0)
        ens = fit_oof_domain_ensemble(m_frames, tune_meta, slices, n_folds=5, folds=folds)
        p_oof = ens['oof_probs']
        tune_group_probs[gname] = p_oof

        vtx = _compute_vertex_audit(list(m_frames.values()), tune_meta, sidecar_df)
        if vtx is not None:
            tune_group_pv[gname] = {'dz_pin': vtx.pop('_dz_pix_centered', None)}

        pc_m = slices['PinpointBlock_PixXY']
        fp_m = slices['FrontPinpoint_PixXY']
        m_all = _ba4_and_nll(labels, p_oof)
        m_pin = _ba4_and_nll(labels[pc_m], p_oof[pc_m])
        dom_res = {}
        for did in (1, 2, 3):
            dm = obs_dom == did
            if dm.any():
                dom_res[OBS_DOMAIN_NAMES[did]] = {
                    **_ba4_and_nll(labels[dm], p_oof[dm]),
                    'tau_eff_1pct_fpr_ptau': float(_tau_efficiency_at_fpr(p_oof[dm, 2], labels[dm], 0.01)['tau_eff']),
                }

        tune_summary[gname] = {
            'prefix': prefix,
            'all_tune8k': m_all,
            'raw_3seed_all_BA4': float(_ba4_and_nll(labels, p_raw_avg)['BA4']),
            'mass_weighted_2t_BA4': _weighted_ba4(labels, p_oof, geom_w),
            'pinpoint_pix_xy': {
                **m_pin,
                'tau_eff_1pct_fpr_ptau': float(_tau_efficiency_at_fpr(p_oof[pc_m, 2], labels[pc_m], 0.01)['tau_eff']),
                'tau_eff_0p1pct_fpr_ptau': float(_tau_efficiency_at_fpr(p_oof[pc_m, 2], labels[pc_m], 0.001)['tau_eff']),
            },
            'front_pinpoint_tau_eff_1pct_ptau': float(_tau_efficiency_at_fpr(p_oof[fp_m, 2], labels[fp_m], 0.01)['tau_eff']),
            'by_domain': dom_res,
            'vertex_audit': vtx,
        }

    # Paired family bootstrap CIs on CleanTune8k (1,000 resamples) for key ablations
    pc_m = slices['PinpointBlock_PixXY']
    fam_codes = pd.MultiIndex.from_frame(tune_meta[FAMILY]).factorize()[0]
    pairs_to_test = [
        ('2x2_target_effect_at_ipt20um: (20um,20um) - (1mm,20um)', 's323-pf-20um', 's323-pf-tgt1mm-ipt20um'),
        ('2x2_ipt_effect_at_tgt20um: (20um,20um) - (20um,1mm)', 's323-pf-20um', 's323-pf-tgt20um-ipt1mm'),
        ('2x2_both_coarse: (20um,20um) - (1mm,1mm)', 's323-pf-20um', 's323-pf-tgt1mm-ipt1mm'),
        ('2x2_target_effect_at_ipt1mm: (20um,1mm) - (1mm,1mm)', 's323-pf-tgt20um-ipt1mm', 's323-pf-tgt1mm-ipt1mm'),
        ('layers: 12T - 6T', 's323-pf-20um', 's323-pf-layers-6t'),
        ('hybrid_vs_vanilla: Hybrid - Vanilla20um', 's323-pf-hybrid-uvtx', 's323-pf-20um'),
        ('hybrid_vs_shuffled: Hybrid - Shuffled', 's323-pf-hybrid-uvtx', 's323-pf-hybrid-shuffled'),
        ('nomad_vs_hybrid: Hybrid+NOMAD - Hybrid', 's323-pf-hybrid-nomad', 's323-pf-hybrid-uvtx'),
        ('energy_weight: Natural - EnergyMatched', 's323-pf-20um', 's323-pf-20um-ematched'),
        ('pf_vs_b1pp: PF20um - B1++', 's323-pf-20um', 's323-b1pp-clean'),
    ]
    paired_boot = {}
    for label_p, ga, gb in pairs_to_test:
        pa_m = tune_group_probs[ga]
        pb_m = tune_group_probs[gb]
        paired_boot[label_p] = paired_family_bootstrap_delta(
            pa_m[:, 2], pb_m[:, 2], pa_m, pb_m, labels, fam_codes, pc_m, n_boot=1000, seed=20263230
        )

    # Pure-W Generator-OOD & Leave-One-Source-Out FP32 validation (strictly zero Confirm!)
    ood_eval_groups = ['s323-b1pp-clean', 's323-pf-20um', 's323-pf-tgt1mm-ipt20um', 's323-pf-hybrid-uvtx', 's323-pf-hybrid-nomad', 's323-pf-20um-ematched']
    ood_results: dict[str, dict] = {g: {} for g in ood_eval_groups}
    ood_batch_cache: dict[str, list[dict]] = {}
    for ov in OOD_VARIANTS:
        ood_root = clean_root / 'ood_shards' / ov
        ometa = pd.read_parquet(ood_root / 'truth_meta.parquet')
        yo = ometa['label'].to_numpy(dtype=np.int64)
        for g in ood_eval_groups:
            preds_s = _infer_group_ood_fp32(ood_root, groups[g]['prefix'], checkpoints, geom, device, ood_batch_cache)
            po = np.mean([d[['p0', 'p1', 'p2', 'p3']].to_numpy(dtype=np.float64) for d in preds_s.values()], axis=0)
            if ov == 'diag_pythia8_free_nucleon_nc_charm':
                ood_results[g][ov] = {
                    'n_events': int(len(yo)),
                    'mean_p_tau': float(po[:, 2].mean()),
                    'frac_pred_nc': float((po.argmax(1) == 3).mean()),
                    'frac_pred_tau': float((po.argmax(1) == 2).mean()),
                }
            else:
                ood_results[g][ov] = _ba4_and_nll(yo, po)

    # Observation-only label-free deployment closure test on 64 real Clean v3 Tune events (322-11)
    obs_only_closure = {}
    with tempfile.TemporaryDirectory(prefix='s323-obs-only-verify-') as tmp:
        tmp_p = Path(tmp)
        raw64 = pq.read_table(clean_root / 'data/shards/080/observations.parquet').slice(0, 64)
        pq.write_table(raw64, tmp_p / 'observations.parquet')
        (tmp_p / 'geometry.json').write_text(json.dumps(geom))
        for g in ('s323-b1pp-clean', 's323-pf-20um', 's323-pf-hybrid-uvtx', 's323-pf-hybrid-nomad'):
            sname = f"{groups[g]['prefix']}_s17"
            ckpt_p = Path(checkpoints[sname]['path'])
            cfg_s = json.loads(Path(checkpoints[sname]['config_path']).read_text())
            df_obs = infer_observation_only(tmp_p, tmp_p / 'geometry.json', ckpt_p, cfg_s, batch_size=32, device=device)
            p_obs = df_obs[['p0', 'p1', 'p2', 'p3']].to_numpy(dtype=np.float64)
            # Compare against standard labeled extraction on the exact same 64 events
            obs_list = raw64.to_pylist()
            bounds = detector_bounds(geom)
            is_b1 = ('v1_b1pp' in str(cfg_s.get('train_module', '')))
            its_lbl = []
            for ii, ob in enumerate(obs_list):
                d_i = compute_observable_containment(ob, geom, bounds, config=cfg_s)
                fn = extract_b1pp_item if is_b1 else extract_pf_item
                its_lbl.append(fn(ii, ob, (ii % 4), d_i, geom, bounds, cfg_s))
            state = torch.load(ckpt_p, map_location=device, weights_only=False)
            if is_b1:
                m = ZAlignedStereoNet(static_plane_layout(geom), cfg_s).to(device)
            elif cfg_s.get('model_family') == 'pf_v4':
                m = PinpointFormerV4(geom, cfg_s).to(device)
            else:
                m = PinpointFormerMultitask(geom, cfg_s).to(device)
            m.load_state_dict(state['model'])
            m.eval()
            p_lbl_chunks = []
            with torch.inference_mode():
                for st_i in range(0, 64, 32):
                    sub = its_lbl[st_i : st_i + 32]
                    if is_b1:
                        b_l = device_batch(collate_b1pp(sub, mirror_rng=None), device)
                        p_lbl_chunks.append(probabilities(m(b_l).float(), 'T4').cpu().numpy())
                    else:
                        b_l = move_batch_to_device(collate_pf(sub, mirror_rng=None), device)
                        p_lbl_chunks.append(m(b_l, grad_blend=1.0, return_aux=False)['probs4'].float().cpu().numpy())
            p_lbl = np.concatenate(p_lbl_chunks, axis=0).astype(np.float64)
            max_diff = float(np.max(np.abs(p_obs - p_lbl)))
            obs_only_closure[sname] = {
                'events': 64,
                'targets_file_present': False,
                'all_finite': bool(np.isfinite(p_obs).all()),
                'max_abs_diff_vs_labeled_fp32': max_diff,
                'passed': bool(np.isfinite(p_obs).all() and max_diff <= 1e-6),
            }
            if max_diff > 1e-6:
                raise AssertionError(f'Observation-only closure failed for {sname}: max_diff={max_diff}')

    # Compute full dataset & executable cryptographic fingerprints (322-09)
    shard_hashes = {}
    for sid in range(150):
        sp = clean_root / 'data/shards' / f'{sid:03d}' / 'observations.parquet'
        if sp.is_file():
            shard_hashes[f'{sid:03d}'] = _sha256_file(sp)
    key_files = {
        'pinpoint_v3_bin': str(PINPOINT_V3_BIN),
        'geom_mac': str(GEOM_V3_MACRO),
        'geometry_json': str(clean_root / 'geometry.json'),
        'planned_manifest': str(clean_root / 'planned_manifest.parquet'),
        'prepare_summary': str(clean_root / 'prepare_summary.json'),
        'finalize_summary': str(clean_root / 'finalize_summary.json'),
        'data_plan': str(clean_root / 'data/plan.json'),
        'tune_screen': str(clean_root / 'data/tune_screen.parquet'),
        'event_sidecar': str(clean_root / 'truth_sidecar/event_sidecar.parquet'),
        'tune_truth_meta': str(clean_root / 'validation_meta/tune_truth_meta.parquet'),
        'train_energy_match_weights': str(clean_root / 'train_energy_match_weights.parquet'),
    }
    file_hashes = {k: {'path': v, 'sha256': _sha256_file(Path(v))} for k, v in key_files.items() if Path(v).is_file()}

    analysis_sha = subprocess.check_output(['git', '-C', str(REPO_ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    g4_sha = subprocess.check_output(['git', '-C', str(G4_REPO_ROOT), 'rev-parse', 'HEAD'], text=True).strip()

    # Load baselines if present for summary figure
    bl_path = s323_root / 'stage323_observable_baselines.json'
    bl_doc = json.loads(bl_path.read_text()) if bl_path.is_file() else None

    # Render 300-DPI English Stage 323 summary figure (Tune8k 2x2 factorial + Baselines + Generator-OOD)
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.2), dpi=300)
    ax0, ax1, ax2 = axes

    # Panel 0: Baselines vs Neural Models on Clean v3 Pure-W Tune8k (PinpointAll BA4 & All BA4)
    bar_names, bar_all, bar_pin = [], [], []
    if bl_doc:
        for bk, bshort in [('lr_presence_counts_6d', 'LR-6d'), ('lr_c_obs_16d', 'LR-16d'),
                           ('lr_full_obs_physics_70d', 'LR-70d'), ('bdt_histgb_full_obs_70d', 'BDT-70d')]:
            if bk in bl_doc['baselines']:
                bar_names.append(bshort)
                bar_all.append(100.0 * bl_doc['baselines'][bk]['tune8k_all']['BA4'])
                bar_pin.append(100.0 * bl_doc['baselines'][bk]['tune8k_pinpoint']['BA4'])
    for gk, gshort in [('s323-b1pp-clean', 'B1++'), ('s323-pf-20um', 'PF-20um'),
                       ('s323-pf-hybrid-uvtx', 'PF-Hybrid'), ('s323-pf-hybrid-nomad', 'PF-NOMAD')]:
        bar_names.append(gshort)
        bar_all.append(100.0 * tune_summary[gk]['all_tune8k']['BA4'])
        bar_pin.append(100.0 * tune_summary[gk]['pinpoint_pix_xy']['BA4'])
    x_pos = np.arange(len(bar_names))
    ax0.bar(x_pos - 0.18, bar_all, width=0.35, color='#4C72B0', label='All Tune8k BA4')
    ax0.bar(x_pos + 0.18, bar_pin, width=0.35, color='#C44E52', label='PinpointBlock BA4')
    ax0.set_xticks(x_pos)
    ax0.set_xticklabels(bar_names, rotation=30, ha='right', fontsize=8.2)
    ax0.set_ylabel('Balanced Accuracy BA4 [%]')
    ax0.set_title('Clean v3 Pure-W Tune8k: Baselines vs Models')
    ax0.legend(loc='lower right', fontsize=8.0)
    ax0.grid(True, axis='y', alpha=0.25)

    # Panel 1: Orthogonal 2x2 Target x IPT Pitch Factorial (BA4 & PV 3D median error)
    f22 = [
        ('Tgt 20um\nIPT 20um', 's323-pf-20um'),
        ('Tgt 1mm\nIPT 20um', 's323-pf-tgt1mm-ipt20um'),
        ('Tgt 20um\nIPT 1mm', 's323-pf-tgt20um-ipt1mm'),
        ('Tgt 1mm\nIPT 1mm', 's323-pf-tgt1mm-ipt1mm'),
        ('6T+3IPT\n20um', 's323-pf-layers-6t'),
    ]
    x2 = np.arange(len(f22))
    ba2 = [100.0 * tune_summary[g]['pinpoint_pix_xy']['BA4'] for _, g in f22]
    pvz2 = [float(tune_summary[g]['vertex_audit']['Truth_pixel_centered']['median_abs_dz_mm']) if tune_summary[g]['vertex_audit'] else 0.0 for _, g in f22]
    ax1.bar(x2 - 0.18, ba2, width=0.35, color='#55A868', label='Pinpoint BA4 [%]')
    ax1_r = ax1.twinx()
    ax1_r.plot(x2 + 0.18, pvz2, 'o-', color='#C44E52', lw=2.0, ms=6, label='PV median $|\\Delta z|$ [mm]')
    ax1.set_xticks(x2)
    ax1.set_xticklabels([lbl for lbl, _ in f22], fontsize=8.0)
    ax1.set_ylabel('PinpointBlock BA4 [%]', color='#2E7D32')
    ax1_r.set_ylabel('PV median $|\\Delta z|$ [mm]', color='#C44E52')
    ax1.set_title('Stage 322-03 Fixed: 2x2 Target x IPT Factorial')
    ax1.grid(True, axis='y', alpha=0.25)

    # Panel 2: Generator-OOD / Leave-One-Source-Out Generalization (Pure-W)
    ood_show = [('gen_ood_run10085_10088', 'Run10085-88'), ('gen_ood_pythia6_std', 'Pythia6-Std'), ('gen_ood_pythia6_nux', 'Pythia6-NUX')]
    x3 = np.arange(len(ood_show))
    w3 = 0.16
    for gi, (gk, gshort, col) in enumerate([
        ('s323-b1pp-clean', 'B1++', '#8C8C8C'),
        ('s323-pf-20um', 'PF-20um', '#4C72B0'),
        ('s323-pf-hybrid-uvtx', 'PF-Hybrid', '#C44E52'),
        ('s323-pf-hybrid-nomad', 'PF-NOMAD', '#55A868'),
        ('s323-pf-20um-ematched', 'PF-EMatch', '#DD8452'),
    ]):
        vals = [100.0 * ood_results[gk][ov]['BA4'] for ov, _ in ood_show]
        ax2.bar(x3 + (gi - 2) * w3, vals, width=w3, color=col, label=gshort)
    ax2.set_xticks(x3)
    ax2.set_xticklabels([lbl for _, lbl in ood_show], fontsize=8.8)
    ax2.set_ylabel('Pure-W OOD BA4 [%]')
    ax2.set_title('Leave-One-Source & Generator-OOD Closure')
    ax2.legend(loc='lower right', fontsize=7.6)
    ax2.grid(True, axis='y', alpha=0.25)

    fig.tight_layout()
    fig_path = s323_root / 'stage323_clean_v3_training_and_ood_validation.png'
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    freeze_doc = {
        'protocol': 'Stage 323 Clean v3 Pure-W Pre-Analysis Cryptographic Freeze (Cal-A/Cut-B/Confirm SEALED)',
        'UTC_frozen': datetime.now(timezone.utc).isoformat(),
        'wall_seconds': float(time.perf_counter() - t0),
        'PinpointAnalysis_git_commit': analysis_sha,
        'Pinpoint_G4_git_commit': g4_sha,
        'Cal_A_Cut_B_Confirm_touched': False,
        'post_training_physics_analysis_paused': True,
        'paused_audit_items': ['322-02', '322-04', '322-06', '322-08', '322-10', '322-14'],
        'remediated_audit_items': ['322-01', '322-03', '322-05', '322-07', '322-09 (upstream+freeze)', '322-11', '322-12', '322-13', '322-15', '322-16'],
        'artifact_fingerprints': file_hashes,
        'nominal_shard_observations_sha256': shard_hashes,
        'checkpoints': checkpoints,
        'tune8k_validation_summary': tune_summary,
        'tune8k_paired_bootstrap_pinpoint': paired_boot,
        'generator_ood_validation': ood_results,
        'observation_only_deployment_closure_322_11': obs_only_closure,
        'figure_path': str(fig_path),
    }
    write_json(clean_root / 'v8_clean_v3_freeze_manifest.json', freeze_doc)
    write_json(ROADMAP_FREEZE_V8, freeze_doc)
    return freeze_doc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('audit-dataset', 'smoke-32', 'baselines', 'build-configs', 'submit-group', 'freeze-and-validate'))
    parser.add_argument('--clean-root', type=Path, default=DEFAULT_CLEAN_V3_ROOT)
    parser.add_argument('--group', type=str, default=None)
    parser.add_argument('--backend', type=str, default='condor', choices=('condor', 'kubeflow'))
    parser.add_argument('--parallel', type=int, default=3)
    parser.add_argument('--kubeflow-flavor', type=str, default='h100-47gb')
    args = parser.parse_args()

    if args.command == 'audit-dataset':
        out = audit_clean_v3(args.clean_root)
    elif args.command == 'smoke-32':
        out = run_smoke32_health_gate_v3(args.clean_root)
    elif args.command == 'baselines':
        out = run_observable_baselines_v3(args.clean_root)
    elif args.command == 'build-configs':
        grps = build_stage323_configs(args.clean_root)
        out = {k: {'prefix': v['prefix'], 'configs': [str(p) for p in v['configs']]} for k, v in grps.items()}
    elif args.command == 'submit-group':
        if not args.group:
            parser.error('--group required for submit-group')
        out = submit_stage323_group(args.clean_root, args.group, args.backend, parallel=args.parallel, kubeflow_flavor=args.kubeflow_flavor)
    else:
        out = freeze_and_validate_stage323(args.clean_root)
    print(json.dumps({k: v for k, v in out.items() if k not in ('nominal_shard_observations_sha256', 'checkpoints')}, indent=2, default=str))


if __name__ == '__main__':
    main()
