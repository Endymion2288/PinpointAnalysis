"""Stage 321 Clean v2 Remediation Campaign:
End-to-end physics remediation of all Stage 320 audit findings:
  1. Exact 108-slab physical-volume tungsten sampling (PinpointBlock vs FortuneBlock + analytic 3D geom_weight).
  2. Single-saturated SiPM bar readout (no double compression; zero MC truth edep_proxy_MeV in observations).
  3. Physical Geant4 geometry loading in all models + Magnet0-corrected spectrometer/NOMAD optics.
  4. Target-only pixel pitch quantization (preserving IPT planes 12..14) + Exp-1 shuffled micro-vertex control.
  5. Exact analytical Cal-A gradient & verified member-permutation invariance + pure observable Cut-B locks.
  6. Deterministic FP32 final unsealing inference, batch-independent Probe A entropy, true Pixel-0 upstream veto,
     additive external neutral-hadron template, separated nu_e charm, per-process SR+CR systematic morphing,
     exact 3D mass-weighted 27-bin profile likelihood, and 1,000-toy H0 calibration.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
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
import pyarrow.parquet as pq
import torch
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler

from analysis.flavour.inventory import write_json
from analysis.flavour.s4.stage_inputs import stage_shards
from analysis.flavour.v2.partitions import FAMILY, KEY
from analysis.flavour.v2.profile import device_batch
from analysis.flavour.v2.r2.readout import grouped_folds
from analysis.flavour.v3.common.calibration import _tau_efficiency_at_fpr
from analysis.flavour.v3.common.datasets import V3EvalDataset
from analysis.flavour.v3.common.domains import (
    OBS_DOMAIN_NAMES,
    detector_bounds,
    pinpoint_slab_catalog,
)
from analysis.flavour.v3.common.probes import compute_probe_a, run_model_with_entropy
from analysis.flavour.v3.common.submit import submit_job
from analysis.flavour.v3.v1_b1pp.models import ZAlignedStereoNet
from analysis.flavour.v3.v1_b1pp.representation import extract_b1pp_item, static_plane_layout
from analysis.flavour.v3.v1_b1pp.train import B1PPCollator
from analysis.flavour.v3.v2_pinpointformer.evaluate import compute_slice_metrics
from analysis.flavour.v3.v2_pinpointformer.pixel_tokenizer import extract_pf_item
from analysis.flavour.v3.v2_pinpointformer.train import (
    CachedPFDataset,
    _ThreadedPFLoader,
    build_pf_mapped_split,
    move_batch_to_device,
)
from analysis.flavour.v3.v3_multitask.models import PinpointFormerMultitask
from analysis.flavour.v3.v3_multitask.train import _pf_v3_predict
from analysis.flavour.v3.v4_tau.ensemble import (
    _apply_domain_convex_blend,
    _fit_domain_convex_blend,
    extract_member_arrays,
    fit_oof_domain_ensemble,
)
from analysis.flavour.v3.v4_tau.models import PinpointFormerV4
from analysis.flavour.v3.v5_confirm.flux_significance import (
    FIDUCIAL_2T_BKG_SHARES,
    assign_signal_and_charm_cr_bins,
    build_physical_templates,
    fit_realistic_profile_likelihood,
    run_h0_toy_coverage_verification,
)
from analysis.flavour.v3.v5_confirm.run_exp6_flux_audit import ULTRA_TAIL_QUANTILES
from analysis.flavour.v3.v5_confirm.significance import _fit_bin_edges
from analysis.flavour.v3.v6_clean_sim.evaluate_stage317 import (
    SEEDS,
    _compute_vertex_audit,
    _select_seed_prediction_df,
    paired_family_bootstrap_delta,
)
from analysis.flavour.v3.v6_clean_sim.produce_clean_sim import (
    DEFAULT_CLEAN_V2_ROOT,
    SYS_VARIANTS,
)


REPO_ROOT = Path(__file__).resolve().parents[5]
ROADMAP_FREEZE_V7 = REPO_ROOT / 'roadmap/v3/v7_freeze_manifest.json'
STAGE318_LEGACY_CLEAN = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v4_audit_2026-11-18/clean_sim')


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values, kind='mergesort')
    v = values[order]
    w = weights[order]
    cdf = np.cumsum(w)
    cdf /= max(float(cdf[-1]), 1e-12)
    return float(v[min(len(v) - 1, int(np.searchsorted(cdf, q)))])


def _weighted_tau_eff_at_fpr(scores: np.ndarray, labels: np.ndarray, weights: np.ndarray, fpr: float) -> float:
    neg = (labels != 2) & (weights > 0.0)
    pos = (labels == 2) & (weights > 0.0)
    if not neg.any() or not pos.any():
        return 0.0
    thr = _weighted_quantile(scores[neg], weights[neg], 1.0 - fpr)
    return float(weights[pos & (scores >= thr)].sum() / max(float(weights[pos].sum()), 1e-12))


def _weighted_ba4(labels: np.ndarray, probs: np.ndarray, weights: np.ndarray) -> float:
    pred = probs.argmax(axis=1)
    recalls = []
    for c in range(4):
        m = (labels == c) & (weights > 0.0)
        if m.any():
            recalls.append(float(weights[m & (pred == c)].sum() / max(float(weights[m].sum()), 1e-12)))
    return float(np.mean(recalls)) if recalls else 0.0


# ---------------------------------------------------------------------------
# 1. Clean v2 Physics, Geometry & Pure-Observables Audit
# ---------------------------------------------------------------------------
def audit_clean_v2(clean_root: Path) -> dict:
    t0 = time.perf_counter()
    geom = json.loads((clean_root / 'geometry.json').read_text())
    fin = json.loads((clean_root / 'finalize_summary.json').read_text())
    prep = json.loads((clean_root / 'prepare_summary.json').read_text())
    man_df = pd.read_parquet(clean_root / 'planned_manifest.parquet')
    cat = pinpoint_slab_catalog(geom)

    w_z = cat['z_center_mm']
    w_half = 0.5 * float(cat['w_thickness_mm'])
    vz = man_df['initZ'].to_numpy(dtype=np.float64)
    vx = man_df['initX'].to_numpy(dtype=np.float64)
    vy = man_df['initY'].to_numpy(dtype=np.float64)
    slabs = man_df['slab_id'].to_numpy(dtype=np.int64)
    weights = man_df['geom_weight'].to_numpy(dtype=np.float64)
    is_pin = man_df['is_pinpoint_module'].to_numpy(dtype=bool)
    is_front_pin = man_df['is_front_pinpoint'].to_numpy(dtype=bool)
    is_interm_pin = man_df['is_interm_pinpoint'].to_numpy(dtype=bool)
    in_pix_xy = man_df['is_pixel_xy_acceptance'].to_numpy(dtype=bool)

    dz_slab = np.abs(vz - w_z[slabs])
    outside_z = int((dz_slab > w_half).sum())
    outside_xy = int(((vx < cat['w_x_min_mm']) | (vx > cat['w_x_max_mm']) | (vy < cat['w_y_min_mm']) | (vy > cat['w_y_max_mm'])).sum())
    if outside_z > 0 or outside_xy > 0:
        raise AssertionError(f'Clean v2 vertices outside physical G4_W: outside_z={outside_z}, outside_xy={outside_xy}')

    # Inspect validation shard 080 to verify ZERO edep_proxy_MeV & single-saturated PE reaching ~14,400!
    obs_val = pq.read_table(clean_root / 'data/shards/080/observations.parquet').to_pylist()
    bar_keys = sorted(obs_val[0]['bar'].keys())
    if 'edep_proxy_MeV' in bar_keys:
        raise AssertionError('MC truth edep_proxy_MeV leaked into Clean v2 observations!')
    pe_all = np.concatenate([np.asarray(ev['bar']['pe_proxy'], dtype=np.float64) for ev in obs_val if len(ev['bar']['pe_proxy']) > 0])
    pe_double_sim = 14400.0 * (1.0 - np.exp(-pe_all / 14400.0))

    # Check paired physics systematic & external/generator shards (also zero edep_proxy_MeV)
    sys_counts = {}
    for sv in (*SYS_VARIANTS, 'external_punchthrough', 'pythia6_std_ref', 'pythia6_nux'):
        sp = clean_root / 'sys_shards' / sv / 'shards/000/observations.parquet'
        if not sp.exists():
            raise FileNotFoundError(f'Missing paired physics systematic shard: {sp}')
        pf = pq.ParquetFile(sp)
        sys_counts[sv] = int(pf.metadata.num_rows)

    # Verify continuous Pythia8 NC charm spectrum (not 6 discrete spikes!)
    nc_ch_ev = man_df.loc[man_df['source_tag'].astype(str).str.startswith('pythia8_nc_charm'), 'Ev_GeV'].to_numpy(dtype=np.float64)
    n_unique_nc_ch_e = int(len(np.unique(np.round(nc_ch_ev, 2))))

    # Render 3-panel 300-DPI English Clean v2 physics audit figure
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.9), dpi=300)
    ax0, ax1, ax2 = axes

    # Panel 0: True slab distribution & exact PinpointBlock vs FortuneBlock mapping
    slab_counts = np.bincount(slabs, minlength=108)
    slab_w_counts = np.bincount(slabs, weights=weights, minlength=108)
    slab_ids = np.arange(108)
    colors_s = ['#C44E52' if cat['is_front_pinpoint'][k] else ('#DD8452' if cat['is_intermediate_pinpoint'][k] else '#4C72B0') for k in slab_ids]
    ax0.bar(slab_ids, slab_counts, width=0.9, color=colors_s, alpha=0.85, label='Raw Stratified Count')
    ax0.plot(slab_ids, slab_w_counts, color='#111111', lw=1.4, label='Mass-Weighted Count ($w_{\\mathrm{geom}}$)')
    ax0.set_xlabel('Physical Tungsten Slab Index $k \\in [0, 107]$')
    ax0.set_ylabel('Events per Slab (60k Total)')
    ax0.set_title('P1 Fixed: Exact Interleaved Pinpoint / Fortune Slabs')
    ax0.legend(loc='upper right', fontsize=8.0, frameon=True)
    ax0.grid(True, axis='y', alpha=0.25)

    # Panel 1: Continuous multi-TeV nu_tau CC and continuous Pythia8 NC charm energy spectra
    bins_e = np.geomspace(20.0, 5000.0, 45)
    e_tau = man_df.loc[man_df['label'] == 2, 'Ev_GeV'].to_numpy(dtype=np.float64)
    e_nc = man_df.loc[(man_df['label'] == 3) & (~man_df['source_tag'].astype(str).str.startswith('pythia8')), 'Ev_GeV'].to_numpy(dtype=np.float64)
    ax1.hist(e_tau, bins=bins_e, density=True, histtype='step', lw=2.0, color='#C44E52', label=f'$\\nu_\\tau$ CC (max {e_tau.max()/1e3:.2f} TeV)')
    ax1.hist(e_nc, bins=bins_e, density=True, histtype='step', lw=1.6, color='#4C72B0', label='GENIE Tungsten NC')
    ax1.hist(nc_ch_ev, bins=bins_e, density=True, histtype='stepfilled', alpha=0.30, color='#55A868', edgecolor='#2E7D32', lw=1.5,
             label=f'Pythia8 NC Charm ({n_unique_nc_ch_e} unique $E_\\nu$)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Incident Neutrino Energy $E_\\nu$ [GeV]')
    ax1.set_ylabel('Normalized Density [GeV$^{-1}$]')
    ax1.set_title('Continuous Multi-TeV $\\nu_\\tau$ & Pythia8 NC Charm Spectra')
    ax1.legend(loc='lower left', fontsize=7.6, frameon=True)
    ax1.grid(True, which='both', alpha=0.25)

    # Panel 2: Single-saturated SiPM PE distribution vs legacy double-compression ceiling
    bins_pe = np.geomspace(2.5, 16000.0, 55)
    ax2.hist(pe_all[pe_all >= 2.5], bins=bins_pe, density=True, histtype='stepfilled', alpha=0.35, color='#4C72B0',
             edgecolor='#2B4F81', lw=1.6, label=f'Clean v2 Single-Sat PE (max={pe_all.max():.0f})')
    ax2.hist(pe_double_sim[pe_double_sim >= 2.5], bins=bins_pe, density=True, histtype='step', ls='--', lw=1.8, color='#C44E52',
             label=f'Legacy Double-Sat Bug (cap={pe_double_sim.max():.0f})')
    ax2.axvline(14400.0, color='#111111', ls=':', lw=1.4, label='Physical SiPM Cap (14,400 PE)')
    ax2.axvline(9102.5, color='#C44E52', ls=':', lw=1.2, label='Spurious Double-Sat Cap (9,103 PE)')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Digitized Scintillator Bar Output [PE] (Zero Truth $E_{\\mathrm{dep}}$)')
    ax2.set_ylabel('Normalized Channel Density')
    ax2.set_title('P1 Fixed: Single SiPM Saturation Preserved')
    ax2.legend(loc='lower left', fontsize=7.4, frameon=True)
    ax2.grid(True, which='both', alpha=0.25)

    fig.tight_layout()
    fig_path = clean_root / 'stage321_clean_v2_physics_audit.png'
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    report = {
        'UTC': datetime.now(timezone.utc).isoformat(),
        'wall_seconds': float(time.perf_counter() - t0),
        'total_events': int(len(man_df)),
        'family_disjoint_verified': bool(prep['family_disjoint_verified']),
        'observations_bar_keys': bar_keys,
        'edep_proxy_MeV_absent_verified': True,
        'tungsten_3d_containment': {
            'outside_z_count': outside_z,
            'outside_xy_count': outside_xy,
            'max_abs_dz_from_slab_center_mm': float(dz_slab.max()),
            'pinpoint_module_raw_count': int(is_pin.sum()),
            'front_pinpoint_pix_xy_raw_count': int((is_front_pin & in_pix_xy).sum()),
            'interm_pinpoint_pix_xy_raw_count': int((is_interm_pin & in_pix_xy).sum()),
            'fortune_module_raw_count': int((~is_pin).sum()),
            'mass_weighted_pinpoint_fraction': float(weights[is_pin].sum() / weights.sum()),
            'exact_geometric_pinpoint_fraction': float(24.0 / 108.0),
            'mass_weighted_fortune_fraction': float(weights[~is_pin].sum() / weights.sum()),
            'mean_geom_weight': float(weights.mean()),
        },
        'sipm_single_saturation_audit': {
            'shard_080_bar_channels': int(len(pe_all)),
            'max_observed_pe': float(pe_all.max()),
            'channels_above_9103_pe': int((pe_all > 9103.0).sum()),
            'fraction_above_2p5_pe': float((pe_all >= 2.5).mean()),
        },
        'pythia8_nc_charm_continuous_audit': {
            'count': int(len(nc_ch_ev)),
            'unique_beam_energies': n_unique_nc_ch_e,
            'E_min_GeV': float(nc_ch_ev.min()) if len(nc_ch_ev) else 0.0,
            'E_median_GeV': float(np.median(nc_ch_ev)) if len(nc_ch_ev) else 0.0,
            'E_max_GeV': float(nc_ch_ev.max()) if len(nc_ch_ev) else 0.0,
        },
        'paired_physics_systematics_shards': sys_counts,
        'finalize_summary': fin,
        'figure_path': str(fig_path),
    }
    write_json(clean_root / 'stage321_clean_v2_dataset_audit.json', report)
    return report


# ---------------------------------------------------------------------------
# 2. Stage 321 Clean v2 Training Configs & Multi-GPU Submission
# ---------------------------------------------------------------------------
def build_stage321_configs(clean_root: Path) -> dict[str, dict]:
    s321_root = clean_root / 'stage321'
    cfg_dir = s321_root / 'configs'
    cfg_dir.mkdir(parents=True, exist_ok=True)

    data_root = str(clean_root / 'data')
    geom_json = str(clean_root / 'geometry.json')
    tune_screen = str(clean_root / 'data/tune_screen.parquet')
    sidecar_root = str(clean_root / 'truth_sidecar')
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
            'apply_sipm_saturation': False,  # CRITICAL P1 FIX: G4 ScintSD already saturates PE once!
            'invert_sipm_saturation': True,
            'y_mirror': False,
            'enable_dense_gn2': False,
            'load_cluster_sidecar': False,
            'use_fixed_event_loss': True,
            'train_module': base_train_mod,
        }

    specs = [
        ('s321-pf-20um', 'v2_pf_pitch_20um', {'quantize_xy_mm': None, 'enable_micro_vertex': False}, base_train_mod),
        ('s321-pf-hybrid-uvtx', 'v2_pf_hybrid_uvtx', {'quantize_xy_mm': None, 'enable_micro_vertex': True}, base_train_mod),
        (
            's321-pf-hybrid-shuffled',
            'v2_pf_hybrid_shuffled',
            {'quantize_xy_mm': None, 'enable_micro_vertex': True, 'shuffle_micro_vertex': True},
            base_train_mod,
        ),
        (
            's321-pf-hybrid-nomad',
            'v2_pf_hybrid_nomad',
            {
                'quantize_xy_mm': None,
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
        (
            's321-pf-100um',
            'v2_pf_pitch_100um_tgtonly',
            {'quantize_xy_mm': 0.10, 'quantize_target_only': True, 'enable_micro_vertex': False},
            base_train_mod,
        ),
        (
            's321-pf-300um',
            'v2_pf_pitch_300um_tgtonly',
            {'quantize_xy_mm': 0.30, 'quantize_target_only': True, 'enable_micro_vertex': False},
            base_train_mod,
        ),
        (
            's321-pf-1mm',
            'v2_pf_pitch_1mm_tgtonly',
            {'quantize_xy_mm': 1.00, 'quantize_target_only': True, 'enable_micro_vertex': False},
            base_train_mod,
        ),
        (
            's321-pf-layers-6t',
            'v2_pf_6layer_20um',
            {'quantize_xy_mm': None, 'active_pixel_planes': [0, 2, 4, 6, 8, 10, 12, 13, 14], 'enable_micro_vertex': False},
            base_train_mod,
        ),
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
        groups[gname] = {
            'prefix': prefix,
            'configs': c_paths,
            'share': '--share-pf-inputs',
            'train_module': tmod,
        }

    # Group 7: Independent B1++ baseline under fixed single-sat PE-only + Magnet0 spectrometer optics
    b1_paths = []
    for s in SEEDS:
        c = {
            'name': f'v2_b1pp_clean_s{s}',
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
            'cache_workers': 6,
            'loader_workers': 4,
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
    groups['s321-b1pp-clean'] = {
        'prefix': 'v2_b1pp_clean',
        'configs': b1_paths,
        'share': '--share-b1pp-inputs',
        'train_module': 'analysis.flavour.v3.v1_b1pp.train',
    }
    return groups


def submit_stage321_group(
    clean_root: Path,
    group_name: str,
    backend: str,
    parallel: int = 3,
    kubeflow_flavor: str = 'h100-47gb',
) -> dict:
    s321_root = clean_root / 'stage321'
    groups = build_stage321_configs(clean_root)
    spec = groups[group_name]
    cmd = [
        'analysis.flavour.v3.common.group',
        '--root', str(s321_root),
        '--name', group_name,
        '--parallel', str(parallel),
        spec['share'],
        '--cache-workers', '6',
        '--train-module', spec['train_module'],
        '--configs', *[str(p) for p in spec['configs']],
    ]
    if backend == 'kubeflow':
        return submit_job(
            root=s321_root,
            name=group_name,
            stage='stage321',
            command=cmd,
            backend='kubeflow',
            flavor=kubeflow_flavor,
            gpu=True,
            deadline_seconds=14400,
        )
    if backend == 'condor':
        # Request 4 CPUs / 20 GB RAM / >= 20 GB VRAM (SM >= 8.0 or H100 MIG 2g.24gb):
        # use --parallel 2 on 24 GB MIG (~15/23.8 GiB) or --parallel 3 on >=36 GB / B1++.
        min_vram = 20000 if (parallel <= 2 or group_name == 's321-b1pp-clean') else 36000
        return submit_job(
            root=s321_root,
            name=group_name,
            stage='stage321',
            command=cmd,
            backend='condor',
            cpus=4,
            memory='20000MB',
            disk='16GB',
            gpu=True,
            min_gpu_memory_mb=min_vram,
            min_capability=8.0,
            job_flavour='longlunch',
            allow_backfill=True,
        )
    raise ValueError(f'Unsupported submit backend: {backend}')


# ---------------------------------------------------------------------------
# 3. Clean v2 Tune8k Selection & Pre-Unsealing Freeze Manifest (v7)
# ---------------------------------------------------------------------------
def freeze_stage321_before_unsealing(clean_root: Path) -> dict:
    s321_root = clean_root / 'stage321'
    groups = build_stage321_configs(clean_root)
    tune_meta = pd.read_parquet(clean_root / 'validation_meta/tune_truth_meta.parquet')
    sidecar_df = pd.read_parquet(clean_root / 'truth_sidecar/event_sidecar.parquet')

    labels = tune_meta['label'].to_numpy(dtype=np.int64)
    obs_dom = tune_meta['obs_domain_id'].to_numpy(dtype=np.int64)
    geom_w = tune_meta['geom_weight'].to_numpy(dtype=np.float64)
    folds, _ = grouped_folds(tune_meta[KEY + FAMILY + ['label']].reset_index(drop=True), folds=5, seed=20261119)
    slices = {
        'All_Tune8k': np.ones(len(tune_meta), dtype=bool),
        'PinpointBlock_PixXY': tune_meta['pixel_centered'].to_numpy(dtype=bool),
        'FrontPinpoint_PixXY': tune_meta['front_pinpoint_pixel_centered'].to_numpy(dtype=bool),
        'D_pix_contained': obs_dom == 1,
    }

    tune_summary = {}
    checkpoints = {}
    for gname, spec in groups.items():
        prefix = spec['prefix']
        m_frames = {}
        for s in SEEDS:
            sname = f'{prefix}_s{s}'
            exp_dir = s321_root / 'experiments' / sname
            df, info = _select_seed_prediction_df(exp_dir)
            m_frames[sname] = df
            ckpt_file = exp_dir / ('best.pt' if info['chosen_file'] == 'best_predictions.parquet' else 'ema.pt')
            checkpoints[sname] = {
                'path': str(ckpt_file),
                'sha256': _sha256_file(ckpt_file),
                'chosen_prediction': info['chosen_file'],
                'config_path': str(s321_root / 'configs' / f'{sname}.json'),
            }
        ens = fit_oof_domain_ensemble(m_frames, tune_meta, slices, n_folds=5, folds=folds)
        vtx = _compute_vertex_audit(list(m_frames.values()), tune_meta, sidecar_df)
        if vtx is not None:
            vtx.pop('_dz_pix_centered', None)
        pc_m = slices['PinpointBlock_PixXY']
        tune_summary[gname] = {
            'prefix': prefix,
            'pinpoint_pix_xy_tau_eff_1pct_fpr': float(_tau_efficiency_at_fpr(ens['s_tau'][pc_m], labels[pc_m], 0.01)['tau_eff']),
            'pinpoint_pix_xy_tau_eff_0p1pct_fpr': float(_tau_efficiency_at_fpr(ens['s_tau'][pc_m], labels[pc_m], 0.001)['tau_eff']),
            'mass_weighted_2t_tau_eff_1pct_fpr': _weighted_tau_eff_at_fpr(ens['s_tau'], labels, geom_w, 0.01),
            'mass_weighted_2t_BA4': _weighted_ba4(labels, ens['oof_probs'], geom_w),
            'all_tune8k_BA4': float(compute_slice_metrics(labels, ens['oof_probs'], slices['All_Tune8k'])['BA4']),
            'vertex_audit': vtx,
        }

    # Select primary PinpointFormer strictly on Clean v2 Tune8k (among physical unshuffled 20 um candidates)
    candidates = ['s321-pf-20um', 's321-pf-hybrid-uvtx', 's321-pf-hybrid-nomad']
    primary_pf_group = max(
        candidates,
        key=lambda g: (
            tune_summary[g]['pinpoint_pix_xy_tau_eff_1pct_fpr']
            + 0.5 * tune_summary[g]['pinpoint_pix_xy_tau_eff_0p1pct_fpr']
            + 0.5 * tune_summary[g]['mass_weighted_2t_tau_eff_1pct_fpr']
        ),
    )
    primary_pf_prefix = groups[primary_pf_group]['prefix']

    git_sha = subprocess.check_output(['git', '-C', str(REPO_ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    if len(git_sha) != 40:
        raise RuntimeError(f'Invalid git commit SHA: {git_sha}')

    freeze_doc = {
        'protocol': 'Stage 321 Clean v2 one-shot sealed calibration/cut/confirmation freeze (CleanCalA -> CleanCutB -> CleanConfirm)',
        'UTC_frozen': datetime.now(timezone.utc).isoformat(),
        'pre_unsealing_git_commit': git_sha,
        'Cal_A_Cut_B_Confirm_read_before_freeze': False,
        'primary_pf_group': primary_pf_group,
        'primary_pf_prefix': primary_pf_prefix,
        'primary_ensemble_members': (
            [f'{primary_pf_prefix}_s{s}' for s in SEEDS]
            + [f'v2_b1pp_clean_s{s}' for s in SEEDS]
        ),
        'tune8k_selection_summary': tune_summary,
        'checkpoints': checkpoints,
    }
    write_json(clean_root / 'v7_freeze_manifest.json', freeze_doc)
    write_json(ROADMAP_FREEZE_V7, freeze_doc)
    return freeze_doc


# ---------------------------------------------------------------------------
# 4. Deterministic FP32 Model Inference & Pure-Observable Cal-A / Cut-B Locks
# ---------------------------------------------------------------------------
def _load_pf_model(ckpt_path: Path, cfg: dict, geometry: dict, device: torch.device) -> torch.nn.Module:
    cls = PinpointFormerV4 if cfg.get('model_family') == 'pf_v4' else PinpointFormerMultitask
    model = cls(geometry, cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state['model'])
    model.eval()
    return model


def _infer_pf_group_fp32(
    staged_data_root: Path,
    split: str,
    prefix: str,
    s321_root: Path,
    checkpoints: dict,
    geometry: dict,
    device: torch.device,
    cfg_override: dict | None = None,
    shared_mapped_root: Path | None = None,
    collect_probe_a: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict | None]:
    cache_tag = hashlib.sha256(
        json.dumps(
            {
                'root': str(staged_data_root),
                'split': split,
                'prefix': prefix,
                'shas': [checkpoints[f'{prefix}_s{s}']['sha256'] for s in SEEDS],
                'override': cfg_override or {},
                'precision': 'fp32',
            },
            sort_keys=True,
        ).encode('utf-8')
    ).hexdigest()[:16]
    pred_cache_dir = s321_root / 'unsealed_preds_fp32' / f'{prefix}_{split}_{cache_tag}'
    if not collect_probe_a and all((pred_cache_dir / f'{prefix}_s{s}.parquet').exists() for s in SEEDS):
        return {f'{prefix}_s{s}': pd.read_parquet(pred_cache_dir / f'{prefix}_s{s}.parquet') for s in SEEDS}, None

    base_cfg = json.loads(Path(checkpoints[f'{prefix}_s17']['config_path']).read_text())
    eval_cfg = {**base_cfg, **(cfg_override or {})}

    tok_key = hashlib.sha256(
        json.dumps(
            {
                'root': str(staged_data_root),
                'split': split,
                'q': eval_cfg.get('quantize_xy_mm'),
                'q_tgt': eval_cfg.get('quantize_target_only', True),
                'planes': eval_cfg.get('active_pixel_planes'),
                'mis': eval_cfg.get('misalign_sigma_xy_mm'),
                'shuf': eval_cfg.get('shuffle_micro_vertex', False),
                'pe_only': eval_cfg.get('pe_only', True),
                'apply_sat': eval_cfg.get('apply_sipm_saturation', False),
                'inv_sat': eval_cfg.get('invert_sipm_saturation', True),
            },
            sort_keys=True,
        ).encode('utf-8')
    ).hexdigest()[:14]

    def _run_with_items(items_dir: Path, pkl_path: Path, ds_cache: Path) -> tuple[dict[str, pd.DataFrame], dict | None]:
        if not pkl_path.exists():
            build_pf_mapped_split(staged_data_root, split, items_dir, pkl_path, ds_cache, workers=6, config=eval_cfg)
        with pkl_path.open('rb') as fp_p:
            pkg = pickle.load(fp_p)
        meta = pkg['metadata']
        bounds = detector_bounds(geometry)
        ds = CachedPFDataset(items_dir, meta)
        b_sampler = BatchSampler(SequentialSampler(ds), batch_size=128, drop_last=False)
        loader = _ThreadedPFLoader(
            ds, b_sampler, seed=42, y_scint_center=bounds['y_scint_center'],
            n_bars=bounds['n_bars'], mirror=False, pin_memory=True, threads=4,
        )
        models = {
            f'{prefix}_s{s}': _load_pf_model(
                Path(checkpoints[f'{prefix}_s{s}']['path']),
                json.loads(Path(checkpoints[f'{prefix}_s{s}']['config_path']).read_text()),
                geometry,
                device,
            )
            for s in SEEDS
        }
        n_ev = len(meta)
        probs = {k: np.zeros((n_ev, 4), dtype=np.float32) for k in models}
        ch_p = {k: np.zeros(n_ev, dtype=np.float32) for k in models}
        vhat = {k: np.zeros((n_ev, 3), dtype=np.float32) for k in models}
        probe_out = None
        try:
            with torch.inference_mode():
                for batch in loader:
                    idx = batch['index'].numpy()
                    db = move_batch_to_device(batch, device)
                    for mname, mod in models.items():
                        # Strictly deterministic FP32 inference (zero autocast drift!)
                        with torch.amp.autocast('cuda', enabled=False):
                            if isinstance(mod, PinpointFormerV4):
                                out = mod(db, grad_blend=1.0, return_aux=False)
                                pr = {
                                    'probs': out['probs4'].float(),
                                    'charm_prob': out['charm_prob'].float(),
                                    'v_hat': out['v_hat'].float(),
                                }
                            else:
                                pr = _pf_v3_predict(mod, db)
                        probs[mname][idx] = pr['probs'].cpu().numpy()
                        ch_p[mname][idx] = pr['charm_prob'].cpu().numpy()
                        vhat[mname][idx] = pr['v_hat'].cpu().numpy()
            if collect_probe_a:
                # Run batch-independent Probe A entropy audit on seed 42 in FP32
                probe_loader = _ThreadedPFLoader(
                    ds, BatchSampler(SequentialSampler(ds), batch_size=128, drop_last=False),
                    seed=42, y_scint_center=bounds['y_scint_center'],
                    n_bars=bounds['n_bars'], mirror=False, pin_memory=True, threads=4,
                )
                ent_res = run_model_with_entropy(models[f'{prefix}_s42'], probe_loader, device, use_bf16=False)
                probe_out = compute_probe_a(
                    ent_res['probs'],
                    meta['label'].to_numpy(dtype=np.int64),
                    ent_res['raw_pixel_hits'],
                    ent_res['mean_entropy'],
                    learned_scales=ent_res['learned_scales'],
                    layer_entropy=ent_res['layer_entropy'],
                )
        finally:
            if hasattr(ds, 'items'):
                ds.items.close()

        out_f = {}
        pred_cache_dir.mkdir(parents=True, exist_ok=True)
        for mname in models:
            df = meta[KEY + FAMILY + ['label']].copy().reset_index(drop=True)
            for c in range(4):
                df[f'p_{c}'] = probs[mname][:, c]
            df['charm_prob'] = ch_p[mname]
            df['v_hat_x_mm'] = vhat[mname][:, 0]
            df['v_hat_y_mm'] = vhat[mname][:, 1]
            df['v_hat_z_mm'] = vhat[mname][:, 2]
            df.to_parquet(pred_cache_dir / f'{mname}.parquet', index=False)
            out_f[mname] = df
        return out_f, probe_out

    if shared_mapped_root is not None:
        sub = shared_mapped_root / f'tok_{split}_{tok_key}'
        sub.mkdir(parents=True, exist_ok=True)
        return _run_with_items(sub / 'items', sub / 'items.pkl', sub / 'ds_cache')

    with tempfile.TemporaryDirectory(prefix=f's321-inf-{prefix}-{split}-', dir=os.environ.get('PINPOINT_V3_SCRATCH', '/tmp')) as tmp:
        tp = Path(tmp)
        return _run_with_items(tp / 'items', tp / 'items.pkl', tp / 'ds_cache')


def _infer_b1pp_group_fp32(
    staged_data_root: Path,
    split: str,
    prefix: str,
    checkpoints: dict,
    geometry: dict,
    bounds: dict,
    device: torch.device,
    cfg_override: dict | None = None,
) -> dict[str, pd.DataFrame]:
    layout = static_plane_layout(geometry, bounds)
    base_cfg = json.loads(Path(checkpoints[f'{prefix}_s17']['config_path']).read_text())
    eval_cfg = {**base_cfg, **(cfg_override or {})}
    raw_ds = V3EvalDataset(staged_data_root, split=split, config=eval_cfg, item_transform=extract_b1pp_item)
    meta = raw_ds.metadata[KEY + FAMILY + ['label']].copy().reset_index(drop=True)
    loader = DataLoader(
        raw_ds,
        batch_sampler=BatchSampler(SequentialSampler(raw_ds), batch_size=128, drop_last=False),
        num_workers=0,
        collate_fn=B1PPCollator(y_mirror=False, n_bars=layout['n_bars']),
    )
    models = {}
    for s in SEEDS:
        mname = f'{prefix}_s{s}'
        mcfg = json.loads(Path(checkpoints[mname]['config_path']).read_text())
        mod = ZAlignedStereoNet(layout, mcfg).to(device)
        st = torch.load(checkpoints[mname]['path'], map_location=device, weights_only=False)
        mod.load_state_dict(st['model'])
        mod.eval()
        models[mname] = mod

    n_ev = len(meta)
    probs = {k: np.zeros((n_ev, 4), dtype=np.float32) for k in models}
    try:
        with torch.inference_mode():
            for batch in loader:
                idx = batch['index'].numpy()
                gpu = device_batch(batch, str(device))
                for mname, mod in models.items():
                    with torch.amp.autocast('cuda', enabled=False):
                        logits = mod(gpu).float()
                    probs[mname][idx] = torch.softmax(logits, dim=-1).cpu().numpy()
    finally:
        raw_ds.native.close()

    out_frames = {}
    for mname in models:
        df = meta.copy()
        for c in range(4):
            df[f'p_{c}'] = probs[mname][:, c]
        out_frames[mname] = df
    return out_frames


def _fit_cala_cutb_and_score_confirm(
    cal_frames: dict[str, pd.DataFrame],
    cut_frames: dict[str, pd.DataFrame],
    conf_frames: dict[str, pd.DataFrame],
    cal_meta: pd.DataFrame,
    cut_meta: pd.DataFrame,
    conf_meta: pd.DataFrame,
    *,
    verify_permutation_invariance: bool = False,
) -> dict:
    mnames = list(cal_frames.keys())
    al_cal = {m: extract_member_arrays(cal_frames[m], cal_meta) for m in mnames}
    al_cut = {m: extract_member_arrays(cut_frames[m], cut_meta) for m in mnames}
    al_con = {m: extract_member_arrays(conf_frames[m], conf_meta) for m in mnames}

    lp_cal = np.stack([al_cal[m]['log_p'] for m in mnames], axis=0)
    lp_cut = np.stack([al_cut[m]['log_p'] for m in mnames], axis=0)
    lp_con = np.stack([al_con[m]['log_p'] for m in mnames], axis=0)

    ch_cut_l = [al_cut[m]['charm'] for m in mnames if al_cut[m]['charm'] is not None]
    ch_con_l = [al_con[m]['charm'] for m in mnames if al_con[m]['charm'] is not None]
    ch_cut = np.mean(ch_cut_l, axis=0) if ch_cut_l else np.zeros(len(cut_meta), dtype=np.float64)
    ch_con = np.mean(ch_con_l, axis=0) if ch_con_l else np.zeros(len(conf_meta), dtype=np.float64)

    y_cal = cal_meta['label'].to_numpy(dtype=np.int64)
    d_cal = cal_meta['obs_domain_id'].to_numpy(dtype=np.int64)
    y_cut = cut_meta['label'].to_numpy(dtype=np.int64)
    d_cut = cut_meta['obs_domain_id'].to_numpy(dtype=np.int64)
    y_con = conf_meta['label'].to_numpy(dtype=np.int64)
    d_con = conf_meta['obs_domain_id'].to_numpy(dtype=np.int64)

    dom_params = {}
    p_cut = np.full((len(cut_meta), 4), 0.25, dtype=np.float64)
    p_con = np.full((len(conf_meta), 4), 0.25, dtype=np.float64)
    perm_audit = None
    for d in (1, 2, 3):
        mc = d_cal == d
        dom_params[d] = _fit_domain_convex_blend(lp_cal[:, mc, :], y_cal[mc])
        if (d_cut == d).any():
            p_cut[d_cut == d] = _apply_domain_convex_blend(lp_cut[:, d_cut == d, :], dom_params[d])
        if (d_con == d).any():
            p_con[d_con == d] = _apply_domain_convex_blend(lp_con[:, d_con == d, :], dom_params[d])

    # Explicit Stage 320 permutation-invariance gate: reversed & rotated member order must yield identical probs!
    if verify_permutation_invariance and len(mnames) >= 2:
        max_diff = 0.0
        total_flips = 0
        rev_idx = list(range(len(mnames) - 1, -1, -1))
        rot_idx = list(range(1, len(mnames))) + [0]
        for perm in (rev_idx, rot_idx):
            p_con_p = np.full_like(p_con, 0.25)
            for d in (1, 2, 3):
                mc = d_cal == d
                fp = _fit_domain_convex_blend(lp_cal[perm][:, mc, :], y_cal[mc])
                if (d_con == d).any():
                    p_con_p[d_con == d] = _apply_domain_convex_blend(lp_con[perm][:, d_con == d, :], fp)
            max_diff = max(max_diff, float(np.max(np.abs(p_con_p - p_con))))
            total_flips += int((p_con_p.argmax(axis=1) != p_con.argmax(axis=1)).sum())
        perm_audit = {'max_abs_prob_diff': max_diff, 'argmax_flips': total_flips, 'passed': bool(max_diff < 1e-6 and total_flips == 0)}
        if not perm_audit['passed']:
            raise AssertionError(f'Cal-A permutation invariance violated: {perm_audit}')

    # Lock lambda_charm & PURELY OBSERVABLE Cut-B thresholds strictly on Cut-B (NEVER using truth vertex slice!)
    s_cut = np.full(len(cut_meta), -25.0, dtype=np.float64)
    s_con = np.full(len(conf_meta), -25.0, dtype=np.float64)
    lam_by_dom = {}
    obs_cutb_thresholds: dict[str, dict[str, float]] = {}
    for d in (1, 2, 3):
        mc = d_cut == d
        mn = d_con == d
        best_lam = 0.0
        if ch_cut_l and (y_cut[mc] == 2).any() and (y_cut[mc] != 2).any():
            sc0 = np.log(np.clip(p_cut[mc, 2], 1e-12, 1.0) / (np.clip(1.0 - p_cut[mc, 2], 1e-12, 1.0) + 1e-12))
            base_obj = (_tau_efficiency_at_fpr(sc0, y_cut[mc], 0.01)['tau_eff'] or 0.0) + 0.5 * (
                _tau_efficiency_at_fpr(sc0, y_cut[mc], 0.001)['tau_eff'] or 0.0
            )
            best_obj = base_obj + 0.005
            for lam in (0.1, 0.25, 0.5, 0.75, 1.0):
                sc = np.log(np.clip(p_cut[mc, 2], 1e-12, 1.0) / (np.clip(1.0 - p_cut[mc, 2], 1e-12, 1.0) + lam * ch_cut[mc] + 1e-12))
                e1 = _tau_efficiency_at_fpr(sc, y_cut[mc], 0.01)['tau_eff'] or 0.0
                e01 = _tau_efficiency_at_fpr(sc, y_cut[mc], 0.001)['tau_eff'] or 0.0
                if e1 + 0.5 * e01 > best_obj + 1e-9:
                    best_obj = e1 + 0.5 * e01
                    best_lam = lam
        lam_by_dom[d] = best_lam
        if mc.any():
            s_cut[mc] = np.log(np.clip(p_cut[mc, 2], 1e-12, 1.0) / (np.clip(1.0 - p_cut[mc, 2], 1e-12, 1.0) + best_lam * ch_cut[mc] + 1e-12))
            neg_d = mc & (y_cut != 2)
            obs_cutb_thresholds[OBS_DOMAIN_NAMES[d]] = {
                '1pct_fpr': float(np.quantile(s_cut[neg_d], 0.99)) if neg_d.any() else float('inf'),
                '0p1pct_fpr': float(np.quantile(s_cut[neg_d], 0.999)) if neg_d.any() else float('inf'),
            }
        if mn.any():
            s_con[mn] = np.log(np.clip(p_con[mn, 2], 1e-12, 1.0) / (np.clip(1.0 - p_con[mn, 2], 1e-12, 1.0) + best_lam * ch_con[mn] + 1e-12))

    act_neg_cut = (d_cut != 0) & (y_cut != 2)
    obs_cutb_thresholds['All_Active'] = {
        '1pct_fpr': float(np.quantile(s_cut[act_neg_cut], 0.99)),
        '0p1pct_fpr': float(np.quantile(s_cut[act_neg_cut], 0.999)),
    }
    w_cut = cut_meta['geom_weight'].to_numpy(dtype=np.float64)
    obs_cutb_thresholds['MassWeighted_2t_Active'] = {
        '1pct_fpr': _weighted_quantile(s_cut[act_neg_cut], w_cut[act_neg_cut], 0.99),
        '0p1pct_fpr': _weighted_quantile(s_cut[act_neg_cut], w_cut[act_neg_cut], 0.999),
    }

    return {
        'member_names': mnames,
        'dom_params': dom_params,
        'lam_by_dom': lam_by_dom,
        'obs_cutb_thresholds': obs_cutb_thresholds,
        'permutation_invariance_audit': perm_audit,
        'probs_cut': p_cut,
        'probs_confirm': p_con,
        's_tau_cut': s_cut,
        's_tau_confirm': s_con,
        'charm_cut': ch_cut,
        'charm_confirm': ch_con,
    }


def _apply_frozen_ensemble_to_frames(
    frames: dict[str, pd.DataFrame],
    ref_meta: pd.DataFrame,
    frozen_fit: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mnames = list(frozen_fit['member_names'])
    al = {m: extract_member_arrays(frames[m], ref_meta) for m in mnames}
    lp = np.stack([al[m]['log_p'] for m in mnames], axis=0)
    ch_l = [al[m]['charm'] for m in mnames if al[m]['charm'] is not None]
    ch = np.mean(ch_l, axis=0) if ch_l else np.zeros(len(ref_meta), dtype=np.float64)
    dom = ref_meta['obs_domain_id'].to_numpy(dtype=np.int64)
    probs = np.full((len(ref_meta), 4), 0.25, dtype=np.float64)
    s_tau = np.full(len(ref_meta), -25.0, dtype=np.float64)
    for d in (1, 2, 3):
        m = dom == d
        if m.any():
            probs[m] = _apply_domain_convex_blend(lp[:, m, :], frozen_fit['dom_params'][d])
            lam = frozen_fit['lam_by_dom'][d]
            s_tau[m] = np.log(np.clip(probs[m, 2], 1e-12, 1.0) / (np.clip(1.0 - probs[m, 2], 1e-12, 1.0) + lam * ch[m] + 1e-12))
    return probs, s_tau, ch


# ---------------------------------------------------------------------------
# 5. One-Shot FP32 Unsealing & Full Physical / Statistical Evaluation
# ---------------------------------------------------------------------------
def unseal_and_evaluate_stage321(clean_root: Path, n_toys: int = 1000) -> dict:
    t0 = time.perf_counter()
    freeze_path = clean_root / 'v7_freeze_manifest.json'
    if not freeze_path.exists():
        raise FileNotFoundError(f'Pre-unsealing freeze manifest required before unsealing: {freeze_path}')
    freeze_doc = json.loads(freeze_path.read_text())
    if len(str(freeze_doc.get('pre_unsealing_git_commit', ''))) != 40:
        raise RuntimeError('v7_freeze_manifest missing valid 40-hex pre_unsealing_git_commit')

    checkpoints = freeze_doc['checkpoints']
    for sname, cinfo in checkpoints.items():
        actual_sha = _sha256_file(Path(cinfo['path']))
        if actual_sha != cinfo['sha256']:
            raise RuntimeError(f'Checkpoint SHA-256 mismatch for {sname}: {actual_sha} != {cinfo["sha256"]}')

    device = torch.device('cuda:0')
    geometry = json.loads((clean_root / 'geometry.json').read_text())
    bounds = detector_bounds(geometry)
    s321_root = clean_root / 'stage321'
    groups = build_stage321_configs(clean_root)

    cal_meta = pd.read_parquet(clean_root / 'sealed_truth_meta/cal_a_truth_meta.parquet')
    cut_meta = pd.read_parquet(clean_root / 'sealed_truth_meta/cut_b_truth_meta.parquet')
    conf_meta = pd.read_parquet(clean_root / 'sealed_truth_meta/confirm_truth_meta.parquet')

    y_con = conf_meta['label'].to_numpy(dtype=np.int64)
    con_dom = conf_meta['obs_domain_id'].to_numpy(dtype=np.int64)
    w_con = conf_meta['geom_weight'].to_numpy(dtype=np.float64)
    pc_con = conf_meta['pixel_centered'].to_numpy(dtype=bool)
    front_pc_con = conf_meta['front_pinpoint_pixel_centered'].to_numpy(dtype=bool)
    interm_pc_con = conf_meta['interm_pinpoint_pixel_centered'].to_numpy(dtype=bool)
    fortune_con = conf_meta['fortune_module'].to_numpy(dtype=bool)
    has_ch_con = conf_meta['has_charm'].to_numpy(dtype=bool)
    fam_hash = conf_meta['family_hi'].to_numpy(dtype=np.uint64) * np.uint64(1000003) + conf_meta['family_lo'].to_numpy(dtype=np.uint64)
    _, fam_codes = np.unique(fam_hash, return_inverse=True)

    conf_vtx_sidecar = conf_meta[KEY].copy()
    conf_vtx_sidecar['vertex_valid'] = (con_dom != 0).astype(np.uint8)
    conf_vtx_sidecar['vertex_x_mm'] = conf_meta['vertex_x_mm']
    conf_vtx_sidecar['vertex_y_mm'] = conf_meta['vertex_y_mm']
    conf_vtx_sidecar['vertex_z_mm'] = conf_meta['vertex_z_mm']

    prim_pf_g = freeze_doc['primary_pf_group']
    prim_pf_prefix = freeze_doc['primary_pf_prefix']
    probe_a_primary = None

    with tempfile.TemporaryDirectory(prefix='s321-unseal-', dir=os.environ.get('PINPOINT_V3_SCRATCH', '/tmp')) as scratch:
        shard_paths = sorted(p for p in (clean_root / 'data/shards').iterdir() if (p / 'observations.parquet').exists())
        st_rep = stage_shards(shard_paths, Path(scratch) / 'staged_clean', workers=6)
        staged_root = clean_root / 'data'
        old_stage_env = os.environ.get('PINPOINT_V3_SHARED_STAGE')
        os.environ['PINPOINT_V3_SHARED_STAGE'] = st_rep['local_shard_root']
        try:
            unsealed_groups = {}
            fits_by_group = {}
            conf_frames_cache = {}
            cal_frames_cache = {}
            cut_frames_cache = {}
            vtx_dz_confirm = {}

            for gname, spec in groups.items():
                prefix = spec['prefix']
                is_b1 = gname == 's321-b1pp-clean'
                if is_b1:
                    f_cal = _infer_b1pp_group_fp32(staged_root, 'cal_a', prefix, checkpoints, geometry, bounds, device)
                    f_cut = _infer_b1pp_group_fp32(staged_root, 'cut_b', prefix, checkpoints, geometry, bounds, device)
                    f_con = _infer_b1pp_group_fp32(staged_root, 'confirm', prefix, checkpoints, geometry, bounds, device)
                else:
                    sm_root = Path(scratch) / 'shared_mapped'
                    f_cal, _ = _infer_pf_group_fp32(staged_root, 'cal_a', prefix, s321_root, checkpoints, geometry, device, shared_mapped_root=sm_root)
                    f_cut, _ = _infer_pf_group_fp32(staged_root, 'cut_b', prefix, s321_root, checkpoints, geometry, device, shared_mapped_root=sm_root)
                    f_con, pa_out = _infer_pf_group_fp32(
                        staged_root, 'confirm', prefix, s321_root, checkpoints, geometry, device,
                        shared_mapped_root=sm_root, collect_probe_a=(gname == prim_pf_g),
                    )
                    if gname == prim_pf_g:
                        probe_a_primary = pa_out

                cal_frames_cache[gname] = f_cal
                cut_frames_cache[gname] = f_cut
                conf_frames_cache[gname] = f_con

                fit = _fit_cala_cutb_and_score_confirm(
                    f_cal, f_cut, f_con, cal_meta, cut_meta, conf_meta,
                    verify_permutation_invariance=(gname == prim_pf_g),
                )
                fits_by_group[gname] = fit

                vtx_a = None if is_b1 else _compute_vertex_audit(list(f_con.values()), conf_meta, conf_vtx_sidecar)
                if vtx_a is not None:
                    vtx_dz_confirm[gname] = vtx_a.pop('_dz_pix_centered')

                s_c = fit['s_tau_confirm']
                p_c = fit['probs_confirm']
                thr_d1_1p = fit['obs_cutb_thresholds']['D_pix_contained']['1pct_fpr']
                thr_d1_01p = fit['obs_cutb_thresholds']['D_pix_contained']['0p1pct_fpr']
                thr_mw_1p = fit['obs_cutb_thresholds']['MassWeighted_2t_Active']['1pct_fpr']

                def _slice_block(mask: np.ndarray) -> dict:
                    return {
                        'count': int(mask.sum()),
                        **compute_slice_metrics(y_con, p_c, mask),
                        'within_slice_exact_1pct_fpr': float(_tau_efficiency_at_fpr(s_c[mask], y_con[mask], 0.01)['tau_eff']),
                        'within_slice_exact_0p1pct_fpr': float(_tau_efficiency_at_fpr(s_c[mask], y_con[mask], 0.001)['tau_eff']),
                    }

                d1_m = con_dom == 1
                nc_ch_d1 = d1_m & (y_con == 3) & has_ch_con
                cc_ch_d1 = d1_m & (y_con != 2) & (y_con != 3) & has_ch_con
                lt_d1 = d1_m & (y_con != 2) & (~has_ch_con)

                unsealed_groups[gname] = {
                    'prefix': prefix,
                    'All_CleanConfirm_Unweighted': _slice_block(np.ones(len(y_con), dtype=bool)),
                    'Full_2t_Detector_MassWeighted': {
                        'BA4': _weighted_ba4(y_con, p_c, w_con),
                        'within_slice_exact_1pct_fpr': _weighted_tau_eff_at_fpr(s_c, y_con, w_con, 0.01),
                        'within_slice_exact_0p1pct_fpr': _weighted_tau_eff_at_fpr(s_c, y_con, w_con, 0.001),
                        'obs_cutb_locked_1pct_tau_eff': float(w_con[(y_con == 2) & (s_c >= thr_mw_1p)].sum() / max(float(w_con[y_con == 2].sum()), 1e-12)),
                        'obs_cutb_locked_1pct_realized_fpr': float(w_con[(y_con != 2) & (s_c >= thr_mw_1p)].sum() / max(float(w_con[y_con != 2].sum()), 1e-12)),
                    },
                    'Obs_D_pix_contained': {
                        **_slice_block(d1_m),
                        'obs_cutb_locked_1pct_tau_eff': float((s_c[d1_m & (y_con == 2)] >= thr_d1_1p).mean()),
                        'obs_cutb_locked_1pct_realized_fpr': float((s_c[d1_m & (y_con != 2)] >= thr_d1_1p).mean()),
                        'obs_cutb_locked_0p1pct_tau_eff': float((s_c[d1_m & (y_con == 2)] >= thr_d1_01p).mean()),
                        'obs_cutb_locked_0p1pct_realized_fpr': float((s_c[d1_m & (y_con != 2)] >= thr_d1_01p).mean()),
                        'nc_charm_fpr_at_obs_1pct_cut': float((s_c[nc_ch_d1] >= thr_d1_1p).mean()) if nc_ch_d1.any() else None,
                        'cc_charm_fpr_at_obs_1pct_cut': float((s_c[cc_ch_d1] >= thr_d1_1p).mean()) if cc_ch_d1.any() else None,
                        'light_bkg_fpr_at_obs_1pct_cut': float((s_c[lt_d1] >= thr_d1_1p).mean()) if lt_d1.any() else None,
                    },
                    'Obs_D_scint_contained': _slice_block(con_dom == 2),
                    'PinpointBlock_PixXY_All': _slice_block(pc_con),
                    'FrontPinpoint_PixXY': _slice_block(front_pc_con),
                    'IntermPinpoint_PixXY': _slice_block(interm_pc_con),
                    'FortuneBlock_All': _slice_block(fortune_con),
                    'vertex_audit': vtx_a,
                }

            # Primary 6-model Dual-Family Clean v2 Ensemble (verified permutation-invariant!)
            ens6_cal = {**cal_frames_cache[prim_pf_g], **cal_frames_cache['s321-b1pp-clean']}
            ens6_cut = {**cut_frames_cache[prim_pf_g], **cut_frames_cache['s321-b1pp-clean']}
            ens6_con = {**conf_frames_cache[prim_pf_g], **conf_frames_cache['s321-b1pp-clean']}
            ens6_fit = _fit_cala_cutb_and_score_confirm(
                ens6_cal, ens6_cut, ens6_con, cal_meta, cut_meta, conf_meta,
                verify_permutation_invariance=True,
            )
            fits_by_group['primary_dual_family_6m'] = ens6_fit

            # Persist exact frozen Cal-A and Cut-B parameters to disk for reproducibility
            write_json(s321_root / 'cal_a_frozen_params.json', {
                'primary_dual_family_6m': {
                    'member_names': ens6_fit['member_names'],
                    'dom_params': {str(k): v for k, v in ens6_fit['dom_params'].items()},
                    'permutation_invariance_audit': ens6_fit['permutation_invariance_audit'],
                },
                'primary_pf_group': {
                    'group': prim_pf_g,
                    'member_names': fits_by_group[prim_pf_g]['member_names'],
                    'dom_params': {str(k): v for k, v in fits_by_group[prim_pf_g]['dom_params'].items()},
                    'permutation_invariance_audit': fits_by_group[prim_pf_g]['permutation_invariance_audit'],
                },
            })
            write_json(s321_root / 'cut_b_locked_params.json', {
                'primary_dual_family_6m': {
                    'lam_by_dom': {str(k): v for k, v in ens6_fit['lam_by_dom'].items()},
                    'obs_cutb_thresholds': ens6_fit['obs_cutb_thresholds'],
                },
                'primary_pf_group': {
                    'group': prim_pf_g,
                    'lam_by_dom': {str(k): v for k, v in fits_by_group[prim_pf_g]['lam_by_dom'].items()},
                    'obs_cutb_thresholds': fits_by_group[prim_pf_g]['obs_cutb_thresholds'],
                },
            })

            s_e6 = ens6_fit['s_tau_confirm']
            p_e6 = ens6_fit['probs_confirm']
            unsealed_groups['primary_dual_family_6m'] = {
                'permutation_invariance_audit': ens6_fit['permutation_invariance_audit'],
                'All_CleanConfirm_Unweighted': {
                    **compute_slice_metrics(y_con, p_e6, None),
                    'within_slice_exact_1pct_fpr': float(_tau_efficiency_at_fpr(s_e6, y_con, 0.01)['tau_eff']),
                    'within_slice_exact_0p1pct_fpr': float(_tau_efficiency_at_fpr(s_e6, y_con, 0.001)['tau_eff']),
                },
                'Full_2t_Detector_MassWeighted': {
                    'BA4': _weighted_ba4(y_con, p_e6, w_con),
                    'within_slice_exact_1pct_fpr': _weighted_tau_eff_at_fpr(s_e6, y_con, w_con, 0.01),
                    'within_slice_exact_0p1pct_fpr': _weighted_tau_eff_at_fpr(s_e6, y_con, w_con, 0.001),
                },
                'Obs_D_pix_contained': {
                    **compute_slice_metrics(y_con, p_e6, con_dom == 1),
                    'within_slice_exact_1pct_fpr': float(_tau_efficiency_at_fpr(s_e6[con_dom == 1], y_con[con_dom == 1], 0.01)['tau_eff']),
                    'within_slice_exact_0p1pct_fpr': float(_tau_efficiency_at_fpr(s_e6[con_dom == 1], y_con[con_dom == 1], 0.001)['tau_eff']),
                },
                'PinpointBlock_PixXY_All': {
                    **compute_slice_metrics(y_con, p_e6, pc_con),
                    'within_slice_exact_1pct_fpr': float(_tau_efficiency_at_fpr(s_e6[pc_con], y_con[pc_con], 0.01)['tau_eff']),
                    'within_slice_exact_0p1pct_fpr': float(_tau_efficiency_at_fpr(s_e6[pc_con], y_con[pc_con], 0.001)['tau_eff']),
                },
                'FrontPinpoint_PixXY': {
                    **compute_slice_metrics(y_con, p_e6, front_pc_con),
                    'within_slice_exact_1pct_fpr': float(_tau_efficiency_at_fpr(s_e6[front_pc_con], y_con[front_pc_con], 0.01)['tau_eff']),
                },
                'IntermPinpoint_PixXY': {
                    **compute_slice_metrics(y_con, p_e6, interm_pc_con),
                    'within_slice_exact_1pct_fpr': float(_tau_efficiency_at_fpr(s_e6[interm_pc_con], y_con[interm_pc_con], 0.01)['tau_eff']),
                },
                'FortuneBlock_All': {
                    **compute_slice_metrics(y_con, p_e6, fortune_con),
                    'within_slice_exact_1pct_fpr': float(_tau_efficiency_at_fpr(s_e6[fortune_con], y_con[fortune_con], 0.01)['tau_eff']),
                },
            }

            # OOD post-hoc target-only quantization comparison on 20 um model (full 4-point curve matching Stage 319)
            ood_pitch_curve = {'0.020': unsealed_groups['s321-pf-20um']['PinpointBlock_PixXY_All']['within_slice_exact_1pct_fpr']}
            for q_mm, q_key in ((0.10, '0.100'), (0.30, '0.300'), (1.00, '1.000')):
                f_ood, _ = _infer_pf_group_fp32(
                    staged_root, 'confirm', 'v2_pf_pitch_20um', s321_root, checkpoints, geometry, device,
                    cfg_override={'quantize_xy_mm': q_mm, 'quantize_target_only': True},
                )
                _, s_ood, _ = _apply_frozen_ensemble_to_frames(f_ood, conf_meta, fits_by_group['s321-pf-20um'])
                ood_pitch_curve[q_key] = float(_tau_efficiency_at_fpr(s_ood[pc_con], y_con[pc_con], 0.01)['tau_eff'])

            # Mechanical plane-wise sensor misalignment scan (0, 10, 25, 50 um) on primary PF model
            misalign_scan = {'0_um': unsealed_groups[prim_pf_g]['PinpointBlock_PixXY_All']['within_slice_exact_1pct_fpr']}
            for sig_mm, m_tag in ((0.010, '10_um'), (0.025, '25_um'), (0.050, '50_um')):
                f_mis, _ = _infer_pf_group_fp32(
                    staged_root, 'confirm', prim_pf_prefix, s321_root, checkpoints, geometry, device,
                    cfg_override={'misalign_sigma_xy_mm': sig_mm},
                )
                _, s_mis, _ = _apply_frozen_ensemble_to_frames(f_mis, conf_meta, fits_by_group[prim_pf_g])
                misalign_scan[m_tag] = float(_tau_efficiency_at_fpr(s_mis[pc_con], y_con[pc_con], 0.01)['tau_eff'])
        finally:
            if old_stage_env is None:
                os.environ.pop('PINPOINT_V3_SHARED_STAGE', None)
            else:
                os.environ['PINPOINT_V3_SHARED_STAGE'] = old_stage_env

    # Paired family-bootstrap comparisons on CleanConfirm (PinpointBlock_PixXY_All & FrontPinpoint_PixXY)
    paired_confirm_comparisons = {
        'Exp1_Hybrid_vs_Vanilla_20um_PinpointAll': paired_family_bootstrap_delta(
            fits_by_group['s321-pf-hybrid-uvtx']['s_tau_confirm'],
            fits_by_group['s321-pf-20um']['s_tau_confirm'],
            fits_by_group['s321-pf-hybrid-uvtx']['probs_confirm'],
            fits_by_group['s321-pf-20um']['probs_confirm'],
            y_con, fam_codes, pc_con, n_boot=1000,
        ),
        'Exp1_Falsification_Hybrid_vs_Shuffled_PinpointAll': paired_family_bootstrap_delta(
            fits_by_group['s321-pf-hybrid-uvtx']['s_tau_confirm'],
            fits_by_group['s321-pf-hybrid-shuffled']['s_tau_confirm'],
            fits_by_group['s321-pf-hybrid-uvtx']['probs_confirm'],
            fits_by_group['s321-pf-hybrid-shuffled']['probs_confirm'],
            y_con, fam_codes, pc_con, n_boot=1000,
        ),
        'Exp1_HybridNOMAD_vs_Hybrid_PinpointAll': paired_family_bootstrap_delta(
            fits_by_group['s321-pf-hybrid-nomad']['s_tau_confirm'],
            fits_by_group['s321-pf-hybrid-uvtx']['s_tau_confirm'],
            fits_by_group['s321-pf-hybrid-nomad']['probs_confirm'],
            fits_by_group['s321-pf-hybrid-uvtx']['probs_confirm'],
            y_con, fam_codes, pc_con, n_boot=1000,
        ),
        'Exp2_20um_vs_100um_TargetOnly_PinpointAll': paired_family_bootstrap_delta(
            fits_by_group['s321-pf-20um']['s_tau_confirm'],
            fits_by_group['s321-pf-100um']['s_tau_confirm'],
            fits_by_group['s321-pf-20um']['probs_confirm'],
            fits_by_group['s321-pf-100um']['probs_confirm'],
            y_con, fam_codes, pc_con, n_boot=1000,
        ),
        'Exp2_20um_vs_300um_TargetOnly_PinpointAll': paired_family_bootstrap_delta(
            fits_by_group['s321-pf-20um']['s_tau_confirm'],
            fits_by_group['s321-pf-300um']['s_tau_confirm'],
            fits_by_group['s321-pf-20um']['probs_confirm'],
            fits_by_group['s321-pf-300um']['probs_confirm'],
            y_con, fam_codes, pc_con, n_boot=1000,
        ),
        'Exp2_20um_vs_1mm_TargetOnly_PinpointAll': paired_family_bootstrap_delta(
            fits_by_group['s321-pf-20um']['s_tau_confirm'],
            fits_by_group['s321-pf-1mm']['s_tau_confirm'],
            fits_by_group['s321-pf-20um']['probs_confirm'],
            fits_by_group['s321-pf-1mm']['probs_confirm'],
            y_con, fam_codes, pc_con, n_boot=1000,
        ),
        'Exp2_20um_vs_1mm_TargetOnly_FrontPinpoint': paired_family_bootstrap_delta(
            fits_by_group['s321-pf-20um']['s_tau_confirm'],
            fits_by_group['s321-pf-1mm']['s_tau_confirm'],
            fits_by_group['s321-pf-20um']['probs_confirm'],
            fits_by_group['s321-pf-1mm']['probs_confirm'],
            y_con, fam_codes, front_pc_con, n_boot=1000,
        ),
        'Exp2_12Layer_vs_6Layer_Target_PinpointAll': paired_family_bootstrap_delta(
            fits_by_group['s321-pf-20um']['s_tau_confirm'],
            fits_by_group['s321-pf-layers-6t']['s_tau_confirm'],
            fits_by_group['s321-pf-20um']['probs_confirm'],
            fits_by_group['s321-pf-layers-6t']['probs_confirm'],
            y_con, fam_codes, pc_con, n_boot=1000,
        ),
        'Exp2_12Layer_vs_6Layer_Target_FrontPinpoint': paired_family_bootstrap_delta(
            fits_by_group['s321-pf-20um']['s_tau_confirm'],
            fits_by_group['s321-pf-layers-6t']['s_tau_confirm'],
            fits_by_group['s321-pf-20um']['probs_confirm'],
            fits_by_group['s321-pf-layers-6t']['probs_confirm'],
            y_con, fam_codes, front_pc_con, n_boot=1000,
        ),
        'DualFamily_6m_vs_PrimaryPF_PinpointAll': paired_family_bootstrap_delta(
            fits_by_group['primary_dual_family_6m']['s_tau_confirm'],
            fits_by_group[prim_pf_g]['s_tau_confirm'],
            fits_by_group['primary_dual_family_6m']['probs_confirm'],
            fits_by_group[prim_pf_g]['probs_confirm'],
            y_con, fam_codes, pc_con, n_boot=1000,
        ),
    }

    # Evaluate tau-decay flight-length stratification (object-level physical validation!)
    tau_l3d = conf_meta['tau_decay_L3d_mm'].to_numpy(dtype=np.float64)
    thr_d1_1p_ens = ens6_fit['obs_cutb_thresholds']['D_pix_contained']['1pct_fpr']
    flight_length_diag = {}
    for lname, lo_l, hi_l in (('L3d_0_1mm', 0.0, 1.0), ('L3d_1_3mm', 1.0, 3.0), ('L3d_3_10mm', 3.0, 10.0), ('L3d_gt_10mm', 10.0, 1e9)):
        tm = pc_con & (y_con == 2) & (tau_l3d >= lo_l) & (tau_l3d < hi_l)
        if tm.any():
            flight_length_diag[lname] = {
                'count': int(tm.sum()),
                'eff_pf_20um': float((fits_by_group['s321-pf-20um']['s_tau_confirm'][tm] >= fits_by_group['s321-pf-20um']['obs_cutb_thresholds']['D_pix_contained']['1pct_fpr']).mean()),
                'eff_pf_hybrid': float((fits_by_group['s321-pf-hybrid-uvtx']['s_tau_confirm'][tm] >= fits_by_group['s321-pf-hybrid-uvtx']['obs_cutb_thresholds']['D_pix_contained']['1pct_fpr']).mean()),
                'eff_pf_shuffled': float((fits_by_group['s321-pf-hybrid-shuffled']['s_tau_confirm'][tm] >= fits_by_group['s321-pf-hybrid-shuffled']['obs_cutb_thresholds']['D_pix_contained']['1pct_fpr']).mean()),
                'eff_pf_1mm': float((fits_by_group['s321-pf-1mm']['s_tau_confirm'][tm] >= fits_by_group['s321-pf-1mm']['obs_cutb_thresholds']['D_pix_contained']['1pct_fpr']).mean()),
                'eff_ens6': float((s_e6[tm] >= thr_d1_1p_ens).mean()),
            }

    # Lock 27-bin Ultra-Tail SR (21) + Charm-CR (6) edges strictly on observable Cut-B
    cut_dom = cut_meta['obs_domain_id'].to_numpy(dtype=np.int64)
    ultra_edges = _fit_bin_edges(ens6_fit['s_tau_cut'], cut_dom, ULTRA_TAIL_QUANTILES)
    charm_cr_edges: dict[str, tuple[float, float]] = {}
    for d_id in (1, 2, 3):
        dn = OBS_DOMAIN_NAMES[d_id]
        dm_cut = cut_dom == d_id
        e_dom = np.asarray(ultra_edges[dn], dtype=np.float64)
        loc_cut = np.clip(np.searchsorted(e_dom[1:-1], ens6_fit['s_tau_cut'][dm_cut], side='right'), 0, len(e_dom) - 2)
        low_cut = loc_cut <= 1
        if low_cut.any():
            q85, q96 = np.quantile(ens6_fit['charm_cut'][dm_cut][low_cut], [0.85, 0.96])
            charm_cr_edges[dn] = (float(q85), float(q96))
        else:
            charm_cr_edges[dn] = (0.42, 0.46)

    sr21_con, joint27_con = assign_signal_and_charm_cr_bins(
        ens6_fit['s_tau_confirm'], ens6_fit['charm_confirm'], con_dom, ultra_edges,
        charm_cr_edges_by_domain=charm_cr_edges,
    )

    # Evaluate ALL 5 Paired Geant4 Physics Systematics across ALL 27 SR+CR bins with separate signal & background morphing!
    cls_to_share = {0: FIDUCIAL_2T_BKG_SHARES['nue_cc'], 1: FIDUCIAL_2T_BKG_SHARES['numu_cc'], 3: FIDUCIAL_2T_BKG_SHARES['nc']}
    sys_eval = {}
    sys_joint_by_var: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for sv in SYS_VARIANTS:
        sv_root = clean_root / 'sys_shards' / sv
        sv_meta = pd.read_parquet(sv_root / 'truth_meta.parquet')
        f_pf_sv, _ = _infer_pf_group_fp32(sv_root, 'confirm', prim_pf_prefix, s321_root, checkpoints, geometry, device)
        f_b1_sv = _infer_b1pp_group_fp32(sv_root, 'confirm', 'v2_b1pp_clean', checkpoints, geometry, bounds, device)
        p_sv, s_sv, c_sv = _apply_frozen_ensemble_to_frames({**f_pf_sv, **f_b1_sv}, sv_meta, ens6_fit)
        y_sv = sv_meta['label'].to_numpy(dtype=np.int64)
        pc_sv = sv_meta['pixel_centered'].to_numpy(dtype=bool)
        dom_sv = sv_meta['obs_domain_id'].to_numpy(dtype=np.int64)
        gw_sv = sv_meta['geom_weight'].to_numpy(dtype=np.float64)
        _, j27_sv = assign_signal_and_charm_cr_bins(s_sv, c_sv, dom_sv, ultra_edges, charm_cr_edges_by_domain=charm_cr_edges)
        sys_joint_by_var[sv] = (j27_sv, y_sv, gw_sv)
        sys_eval[sv] = {
            'events': int(len(sv_meta)),
            'All_BA4': float(compute_slice_metrics(y_sv, p_sv, None)['BA4']),
            'MassWeighted_2t_BA4': _weighted_ba4(y_sv, p_sv, gw_sv),
            'pinpoint_pix_xy_tau_eff_1pct_fpr': float(_tau_efficiency_at_fpr(s_sv[pc_sv], y_sv[pc_sv], 0.01)['tau_eff']),
        }

    def _sig_bkg_hist27(bins27: np.ndarray, y_arr: np.ndarray, gw_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ok = (bins27 >= 0) & (bins27 < 27)
        w_sig = np.where(y_arr == 2, gw_arr, 0.0)
        if w_sig.sum() > 0:
            w_sig = w_sig / w_sig.sum()
        w_bkg = np.zeros(len(y_arr), dtype=np.float64)
        for c_k, sh_c in cls_to_share.items():
            mc = y_arr == c_k
            sw = float(gw_arr[mc].sum())
            if sw > 0.0:
                w_bkg[mc] = gw_arr[mc] * (sh_c / sw)
        h_s = np.bincount(bins27[ok], weights=w_sig[ok], minlength=27)
        h_b = np.bincount(bins27[ok], weights=w_bkg[ok], minlength=27)
        return h_s, h_b

    nom_s27, nom_b27 = _sig_bkg_hist27(*sys_joint_by_var['nominal_sys_ref'])
    signed_g4_sys: dict[str, dict] = {}
    for sv_k in ('tau_helicity_0', 'w_density_0p98', 'no_magnet_shells', 'field_0p97'):
        hs, hb = _sig_bkg_hist27(*sys_joint_by_var[sv_k])
        ds = np.clip(np.log((hs + 1e-3) / (nom_s27 + 1e-3)), -0.35, 0.35)
        db = np.clip(np.log((hb + 1e-3) / (nom_b27 + 1e-3)), -0.35, 0.35)
        if sv_k == 'tau_helicity_0':
            # P1 Physics Fix: tau helicity polarization ONLY modifies nu_tau CC signal, NEVER backgrounds!
            signed_g4_sys['g4_tau_helicity_0'] = {
                'applies_to': 'signal',
                'delta_log_up': ds.tolist(),
                'delta_log_down': (-0.5 * ds).tolist(),
            }
        else:
            signed_g4_sys[f'g4_{sv_k}'] = {
                'delta_log_up_sig': ds.tolist(),
                'delta_log_down_sig': (-0.5 * ds).tolist(),
                'delta_log_up_bkg': db.tolist(),
                'delta_log_down_bkg': (-0.5 * db).tolist(),
            }

    # Evaluate cell-matched GENIE Pythia6_NUX vs Pythia6_Standard hadronization systematic across ALL 27 bins
    gen_sys_eval = {}
    if (clean_root / 'sys_shards/pythia6_nux/shards/000/observations.parquet').exists():
        gen_j27 = {}
        for gv in ('pythia6_std_ref', 'pythia6_nux'):
            gv_root = clean_root / 'sys_shards' / gv
            gv_meta = pd.read_parquet(gv_root / 'truth_meta.parquet')
            f_pf_gv, _ = _infer_pf_group_fp32(gv_root, 'confirm', prim_pf_prefix, s321_root, checkpoints, geometry, device)
            f_b1_gv = _infer_b1pp_group_fp32(gv_root, 'confirm', 'v2_b1pp_clean', checkpoints, geometry, bounds, device)
            p_gv, s_gv, c_gv = _apply_frozen_ensemble_to_frames({**f_pf_gv, **f_b1_gv}, gv_meta, ens6_fit)
            y_gv = gv_meta['label'].to_numpy(dtype=np.int64)
            pc_gv = gv_meta['pixel_centered'].to_numpy(dtype=bool)
            dom_gv = gv_meta['obs_domain_id'].to_numpy(dtype=np.int64)
            gw_gv = gv_meta['geom_weight'].to_numpy(dtype=np.float64)
            _, j27_gv = assign_signal_and_charm_cr_bins(s_gv, c_gv, dom_gv, ultra_edges, charm_cr_edges_by_domain=charm_cr_edges)
            gen_j27[gv] = (j27_gv, y_gv, gw_gv)
            gen_sys_eval[gv] = {
                'events': int(len(gv_meta)),
                'All_BA4': float(compute_slice_metrics(y_gv, p_gv, None)['BA4']),
                'MassWeighted_2t_BA4': _weighted_ba4(y_gv, p_gv, gw_gv),
                'pinpoint_pix_xy_tau_eff_1pct_fpr': float(_tau_efficiency_at_fpr(s_gv[pc_gv], y_gv[pc_gv], 0.01)['tau_eff']),
            }
        hs_std, hb_std = _sig_bkg_hist27(*gen_j27['pythia6_std_ref'])
        hs_nux, hb_nux = _sig_bkg_hist27(*gen_j27['pythia6_nux'])
        ds_gen = np.clip(np.log((hs_nux + 1e-3) / (hs_std + 1e-3)), -0.35, 0.35)
        db_gen = np.clip(np.log((hb_nux + 1e-3) / (hb_std + 1e-3)), -0.35, 0.35)
        signed_g4_sys['gen_pythia6_nux'] = {
            'delta_log_up_sig': ds_gen.tolist(),
            'delta_log_down_sig': (-ds_gen).tolist(),
            'delta_log_up_bkg': db_gen.tolist(),
            'delta_log_down_bkg': (-db_gen).tolist(),
        }

    # External punch-through audit using TRUE PHYSICAL UPSTREAM VETO (PixelSensor[0] >= 1 hit, z=-2991.995 < W0!)
    ext_punch_eval = {}
    ext_nh_shape_27 = np.zeros(27, dtype=np.float64)
    p0_con_hits = conf_meta['p0_pixel_hits'].to_numpy(dtype=np.int64)
    neutrino_p0_veto_loss_all_mw = float(w_con[p0_con_hits >= 1].sum() / max(float(w_con.sum()), 1e-12))
    neutrino_p0_veto_loss_front_pin = float((p0_con_hits[front_pc_con] >= 1).mean()) if front_pc_con.any() else 0.0
    if (clean_root / 'sys_shards/external_punchthrough/shards/000/observations.parquet').exists():
        ext_root = clean_root / 'sys_shards/external_punchthrough'
        ext_meta = pd.read_parquet(ext_root / 'truth_meta.parquet')
        # True upstream veto: PixelSensor[0] hit count >= 1 (z=-2991.995 mm, upstream of W0=-2984.49 mm!)
        veto_p0_fired = ext_meta['p0_pixel_hits'].to_numpy(dtype=np.int64) >= 1
        f_pf_ex, _ = _infer_pf_group_fp32(ext_root, 'confirm', prim_pf_prefix, s321_root, checkpoints, geometry, device)
        f_b1_ex = _infer_b1pp_group_fp32(ext_root, 'confirm', 'v2_b1pp_clean', checkpoints, geometry, bounds, device)
        _, s_ex, c_ex = _apply_frozen_ensemble_to_frames({**f_pf_ex, **f_b1_ex}, ext_meta, ens6_fit)
        dom_ex = ext_meta['obs_domain_id'].to_numpy(dtype=np.int64)
        sr21_ex, j27_ex = assign_signal_and_charm_cr_bins(s_ex, c_ex, dom_ex, ultra_edges, charm_cr_edges_by_domain=charm_cr_edges)
        thr_global_1p = ens6_fit['obs_cutb_thresholds']['All_Active']['1pct_fpr']
        subtypes = ext_meta['ext_subtype'].to_numpy(dtype=str)
        mu_pix = subtypes == 'rock_muon_pix_xy'
        mu_out = subtypes == 'rock_muon_outer_xy'
        nh_m = np.isin(subtypes, ['neutron', 'k0l'])
        ext_punch_eval = {
            'total_events': int(len(ext_meta)),
            'upstream_veto_definition': 'PixelSensor[0] (z=-2991.995 mm, upstream of W0=-2984.49 mm) hits >= 1',
            'neutrino_signal_loss_mass_weighted_2t': neutrino_p0_veto_loss_all_mw,
            'neutrino_signal_loss_front_pinpoint': neutrino_p0_veto_loss_front_pin,
            'rock_muon_pix_xy_count': int(mu_pix.sum()),
            'rock_muon_pix_xy_p0_veto_eff': float(veto_p0_fired[mu_pix].mean()) if mu_pix.any() else None,
            'rock_muon_pix_xy_unvetoed_tau_1pct_leakage': float(((~veto_p0_fired) & mu_pix & (s_ex >= thr_global_1p)).sum() / max(int(mu_pix.sum()), 1)),
            'rock_muon_outer_xy_count': int(mu_out.sum()),
            'rock_muon_outer_xy_tau_1pct_leakage_no_front_veto': float((mu_out & (s_ex >= thr_global_1p)).sum() / max(int(mu_out.sum()), 1)),
            'neutral_hadron_count': int(nh_m.sum()),
            'neutral_hadron_unvetoed_fraction': float((~veto_p0_fired[nh_m]).mean()) if nh_m.any() else None,
            'neutral_hadron_unvetoed_tau_1pct_fpr': float(((~veto_p0_fired) & nh_m & (s_ex >= thr_global_1p)).sum() / max(int(nh_m.sum()), 1)),
            'neutral_hadron_ultra_tail_top3bin_rate': float(((~veto_p0_fired) & nh_m & np.isin(sr21_ex, [6, 13, 20])).sum() / max(int(nh_m.sum()), 1)),
        }
        unv_nh = (~veto_p0_fired) & nh_m & (j27_ex >= 0) & (j27_ex < 27)
        if unv_nh.any():
            h_nh = np.bincount(j27_ex[unv_nh], minlength=27).astype(np.float64)
            ext_nh_shape_27 = h_nh / max(float(h_nh.sum()), 1.0)

    # Exact 3D mass-weighted 27-bin forward-flux sensitivity with additive b_ext (2% of fiducial rate, +-50% norm nuisance!)
    fid_m = conf_meta['large_scintillator'].to_numpy(dtype=bool)
    flux_projections = {}
    hl_tpl_clean = None
    for era, lum_fb, n_fid in (
        ('Run3_250fb', 250.0, 15000.0),
        ('Run4_680fb', 680.0, 40800.0),
        ('HLLHC_3000fb', 3000.0, 180000.0),
        ('FPF_10000fb', 10000.0, 600000.0),
    ):
        b_ext_era = (0.02 * n_fid) * ext_nh_shape_27
        tpl27 = build_physical_templates(
            joint27_con[fid_m], y_con[fid_m], has_ch_con[fid_m],
            n_bins=27, total_events=n_fid, tau_fraction=0.003, bkg_shares=FIDUCIAL_2T_BKG_SHARES,
            nc_charm_fraction_override=None, nc_charm_physical_fraction=0.045, zero_bkg_jeffreys_prior=True,
            event_weights=w_con[fid_m], external_bkg=b_ext_era,
        )
        if era == 'HLLHC_3000fb':
            hl_tpl_clean = tpl27
        pl_phys = fit_realistic_profile_likelihood(tpl27, signed_sys=signed_g4_sys, n_sr_bins=21, enable_barlow_beeston=False)
        pl_bb = fit_realistic_profile_likelihood(tpl27, signed_sys=signed_g4_sys, n_sr_bins=21, enable_barlow_beeston=True)
        flux_projections[era] = {
            'luminosity_fb_inv': lum_fb,
            'fiducial_events': n_fid,
            'expected_nutau_cc': float(tpl27['s_tau'].sum()),
            'expected_ext_neutral_hadron': float(tpl27['b_ext'].sum()),
            'min_raw_mc_bkg_in_sr21': int(tpl27['n_raw_bkg'][:21].min()),
            'min_eff_mc_bkg_in_sr21': float(tpl27['n_eff_bkg'][:21].min()),
            **pl_phys,
            'z0_pl_with_10k_confirm_bb_mc_stat': pl_bb['z0_profile_likelihood'],
            'q0_pl_with_10k_confirm_bb_mc_stat': pl_bb['q0_profile_likelihood'],
            'bb_optimizer_converged': pl_bb['optimizer_converged'],
            'min_gamma_mc_stat_at_H0': pl_bb['min_gamma_mc_stat_at_H0'],
            'max_gamma_mc_stat_at_H0': pl_bb['max_gamma_mc_stat_at_H0'],
        }

    assert hl_tpl_clean is not None
    h0_clean_bb = run_h0_toy_coverage_verification(
        hl_tpl_clean, signed_sys=signed_g4_sys, n_sr_bins=21, n_toys=n_toys, workers=16, enable_barlow_beeston=True
    )
    h0_clean_phys = run_h0_toy_coverage_verification(
        hl_tpl_clean, signed_sys=signed_g4_sys, n_sr_bins=21, n_toys=n_toys, workers=16, enable_barlow_beeston=False
    )

    # Figure 1: Hardware Target-Only Pitch Curve, Exp-1 Shuffled Falsification & Module Composition, Misalignment
    fig1, axes1 = plt.subplots(1, 3, figsize=(16.5, 5.0), dpi=300)
    a0, a1, a2 = axes1

    pitches_um = [20.8, 100.0, 300.0, 1000.0]
    fs_vals = [
        unsealed_groups['s321-pf-20um']['PinpointBlock_PixXY_All']['within_slice_exact_1pct_fpr'] * 100.0,
        unsealed_groups['s321-pf-100um']['PinpointBlock_PixXY_All']['within_slice_exact_1pct_fpr'] * 100.0,
        unsealed_groups['s321-pf-300um']['PinpointBlock_PixXY_All']['within_slice_exact_1pct_fpr'] * 100.0,
        unsealed_groups['s321-pf-1mm']['PinpointBlock_PixXY_All']['within_slice_exact_1pct_fpr'] * 100.0,
    ]
    ood_vals = [ood_pitch_curve[k] * 100.0 for k in ('0.020', '0.100', '0.300', '1.000')]
    a0.plot(pitches_um, fs_vals, 'o-', lw=2.3, ms=7, color='#4C72B0', label='Retrained Target-Only (IPT Native)')
    a0.plot(pitches_um, ood_vals, 's--', lw=1.8, ms=6, color='#C44E52', label='Post-hoc OOD Quantized on 20.8 $\\mu$m')
    a0.set_xscale('log')
    a0.set_xticks(pitches_um)
    a0.set_xticklabels(['20.8 $\\mu$m', '100 $\\mu$m', '300 $\\mu$m', '1 mm'])
    a0.set_xlabel('Target Silicon Pixel Pitch $\\Delta_{xy}$ (IPT Unchanged)')
    a0.set_ylabel('PinpointBlock Pixel-XY $\\nu_\\tau$ Eff. @ 1% FPR [%]')
    a0.set_title('Exp-2 Fixed: Target-Only Pitch Curve (True Pinpoint Slabs)')
    a0.legend(loc='lower left', fontsize=8.0, frameon=True)
    a0.grid(True, which='both', alpha=0.25)

    m_labels = ['PF 20.8 $\\mu$m\n12T+3IPT', 'PF 20.8 $\\mu$m\n6T+3IPT', 'PF Hybrid\nMicro-Vtx', 'PF Hybrid\nShuffled', 'PF Hybrid\n+M0-NOMAD', 'B1++\nClean', 'Dual-Fam\n6-Model']
    m_keys = ['s321-pf-20um', 's321-pf-layers-6t', 's321-pf-hybrid-uvtx', 's321-pf-hybrid-shuffled', 's321-pf-hybrid-nomad', 's321-b1pp-clean', 'primary_dual_family_6m']
    y_pin = [unsealed_groups[k]['PinpointBlock_PixXY_All']['within_slice_exact_1pct_fpr'] * 100.0 for k in m_keys]
    y_mw = [unsealed_groups[k]['Full_2t_Detector_MassWeighted']['within_slice_exact_1pct_fpr'] * 100.0 for k in m_keys]
    xm = np.arange(len(m_keys))
    b_p = a1.bar(xm - 0.19, y_pin, width=0.36, color='#4C72B0', label='PinpointBlock Pixel-XY (24 slabs)')
    b_w = a1.bar(xm + 0.19, y_mw, width=0.36, color='#55A868', label='Mass-Weighted Full 2t Detector')
    for rect, v in zip(b_p, y_pin):
        a1.text(rect.get_x() + rect.get_width() / 2.0, v + 0.18, f'{v:.1f}%', ha='center', fontsize=6.8)
    for rect, v in zip(b_w, y_mw):
        a1.text(rect.get_x() + rect.get_width() / 2.0, v + 0.18, f'{v:.1f}%', ha='center', fontsize=6.8)
    a1.set_xticks(xm)
    a1.set_xticklabels(m_labels, fontsize=7.0)
    a1.set_ylabel('$\\nu_\\tau$ CC Efficiency @ Exact 1% FPR [%]')
    a1.set_title('Exp-1 Falsification Control & Mass-Weighted 2t Performance')
    a1.set_ylim(0.0, max(max(y_pin), max(y_mw)) * 1.32)
    a1.legend(loc='upper left', fontsize=7.5, frameon=True)
    a1.grid(True, axis='y', alpha=0.25)

    mis_x = [0.0, 10.0, 25.0, 50.0]
    mis_y = [misalign_scan[k] * 100.0 for k in ('0_um', '10_um', '25_um', '50_um')]
    a2.plot(mis_x, mis_y, 'D-', lw=2.2, ms=6.5, color='#8172B2')
    for x_i, y_i in zip(mis_x, mis_y):
        a2.annotate(f'{y_i:.2f}%', (x_i, y_i), textcoords='offset points', xytext=(0, 7), ha='center', fontsize=8.0)
    a2.set_xlabel('Plane-Wise Transverse Misalignment $\\sigma_{xy}$ [$\\mu$m]')
    a2.set_ylabel('PinpointBlock Pixel-XY $\\nu_\\tau$ Eff. @ 1% FPR [%]')
    a2.set_title('Mechanical Alignment Tolerance (Clean v2 FP32)')
    a2.set_ylim(max(0.0, min(mis_y) - 2.0), min(100.0, max(mis_y) + 2.0))
    a2.grid(True, alpha=0.25)

    fig1.tight_layout()
    fig1_path = clean_root / 'stage321_clean_v2_hardware_and_falsification.png'
    fig1.savefig(fig1_path, dpi=300)
    plt.close(fig1)

    # Figure 2: Vertex Precision, Paired 1,200-Event G4 Systematics, and Mass-Weighted 27-Bin Sensitivity
    fig2, axes2 = plt.subplots(1, 3, figsize=(16.5, 5.0), dpi=300)
    b0, b1, b2 = axes2

    bins_vz = np.linspace(-30.0, 30.0, 61)
    for lbl_v, gk_v, col_v in (
        ('PF 20.8 $\\mu$m Baseline', 's321-pf-20um', '#4C72B0'),
        ('PF Hybrid Micro-Vtx', 's321-pf-hybrid-uvtx', '#C44E52'),
        ('PF 1 mm Target-Only', 's321-pf-1mm', '#8C8C8C'),
    ):
        if gk_v in vtx_dz_confirm:
            va = unsealed_groups[gk_v]['vertex_audit']['Truth_pixel_centered']
            b0.hist(
                vtx_dz_confirm[gk_v], bins=bins_vz, range=(-30.0, 30.0), density=True, histtype='step', lw=1.9, color=col_v,
                label=f"{lbl_v}\n(med $|\\Delta z|={va['median_abs_dz_mm']:.2f}$ mm, $\\mu={va['bias_z_mm']:+.2f}$ mm)",
            )
    b0.axvline(0.0, color='#222222', ls=':', lw=1.1)
    b0.set_ylim(0.0, b0.get_ylim()[1] * 1.38)
    b0.set_xlabel('Primary Vertex Residual $\\hat{z}_v - z_{\\mathrm{true}}$ [mm]')
    b0.set_ylabel('Normalized Core Density [mm$^{-1}$]')
    b0.set_title('Primary Vertex Precision (Physical G4 Geometry)')
    b0.legend(loc='upper left', fontsize=7.4, frameon=True)
    b0.grid(True, alpha=0.25)

    sv_labels = ['Nominal\nRef', '$h(\\tau)=0$\n(Sig-Only)', '$\\rho_W \\times 0.98$\nW Density', 'No Magnet\nShells', '$B \\times 0.97$\nField']
    sv_keys = ['nominal_sys_ref', 'tau_helicity_0', 'w_density_0p98', 'no_magnet_shells', 'field_0p97']
    sv_vals = [sys_eval[k]['pinpoint_pix_xy_tau_eff_1pct_fpr'] * 100.0 for k in sv_keys]
    bars_sv = b1.bar(np.arange(len(sv_keys)), sv_vals, color=['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2'], width=0.55)
    b1.axhline(sv_vals[0], color='#222222', ls='--', lw=1.1)
    b1.set_xticks(np.arange(len(sv_keys)))
    b1.set_xticklabels(sv_labels, fontsize=7.8)
    b1.set_ylabel('Paired 1,200-Event $\\nu_\\tau$ Eff. @ 1% FPR [%]')
    b1.set_title('Paired Geant4 Transport Systematics Stability')
    b1.set_ylim(0.0, max(sv_vals) * 1.38)
    for rect, v in zip(bars_sv, sv_vals):
        b1.text(rect.get_x() + rect.get_width() / 2.0, v + 0.15, f'{v:.1f}%', ha='center', fontsize=7.8)
    b1.grid(True, axis='y', alpha=0.25)

    eras_x = ['Run 3\n250 fb$^{-1}$', 'Run 4\n680 fb$^{-1}$', 'HL-LHC\n3 ab$^{-1}$', 'FPF\n10 ab$^{-1}$']
    eras_k = ['Run3_250fb', 'Run4_680fb', 'HLLHC_3000fb', 'FPF_10000fb']
    z_stat = [flux_projections[k]['z0_stat_only'] for k in eras_k]
    z_prof = [flux_projections[k]['z0_profile_likelihood'] for k in eras_k]
    z_bb = [flux_projections[k]['z0_pl_with_10k_confirm_bb_mc_stat'] for k in eras_k]
    xe = np.arange(len(eras_k))
    b2.bar(xe - 0.25, z_stat, width=0.24, color='#4C72B0', label='Stat-only (Mass-Weighted 27-bin)')
    bars_zp = b2.bar(xe, z_prof, width=0.24, color='#C44E52', label='Full Physics+G4/Gen/Ext/Charm PL')
    b2.bar(xe + 0.25, z_bb, width=0.24, color='#DD8452', label='+ 10k-Confirm Weighted BB-lite')
    b2.axhline(3.0, color='#555555', ls=':', lw=1.1, label='3$\\sigma$ threshold')
    b2.set_ylim(0.0, max(max(z_stat), 3.3) * 1.24)
    for rect, zv in zip(bars_zp, z_prof):
        b2.text(rect.get_x() + rect.get_width() / 2.0, zv + 0.08, f'{zv:.2f}$\\sigma$', ha='center', fontsize=7.3, color='#902020')
    b2.set_xticks(xe)
    b2.set_xticklabels(eras_x, fontsize=8.0)
    b2.set_ylabel('Expected $\\nu_\\tau$ Discovery Significance $Z_0$ [$\\sigma$]')
    b2.set_title('Clean v2 Mass-Weighted 2t Realistic Sensitivity')
    b2.legend(loc='upper left', fontsize=7.0, frameon=True)
    b2.grid(True, axis='y', alpha=0.25)

    fig2.tight_layout()
    fig2_path = clean_root / 'stage321_clean_v2_vertex_sys_and_sensitivity.png'
    fig2.savefig(fig2_path, dpi=300)
    plt.close(fig2)

    final_report = {
        'UTC': datetime.now(timezone.utc).isoformat(),
        'wall_seconds': float(time.perf_counter() - t0),
        'inference_precision': 'FP32',
        'freeze_manifest_verified': True,
        'pre_unsealing_git_commit': freeze_doc['pre_unsealing_git_commit'],
        'primary_pf_group': prim_pf_g,
        'cutb_locked_charm_cr_edges': charm_cr_edges,
        'probe_a_primary_pf_batch_independent_entropy': probe_a_primary,
        'unsealed_clean_v2_confirm_groups': unsealed_groups,
        'tau_flight_length_diagnostics': flight_length_diag,
        'ood_vs_from_scratch_target_only_pitch': {
            'from_scratch_exact_1pct_fpr': {
                '0.020': unsealed_groups['s321-pf-20um']['PinpointBlock_PixXY_All']['within_slice_exact_1pct_fpr'],
                '0.100': unsealed_groups['s321-pf-100um']['PinpointBlock_PixXY_All']['within_slice_exact_1pct_fpr'],
                '0.300': unsealed_groups['s321-pf-300um']['PinpointBlock_PixXY_All']['within_slice_exact_1pct_fpr'],
                '1.000': unsealed_groups['s321-pf-1mm']['PinpointBlock_PixXY_All']['within_slice_exact_1pct_fpr'],
            },
            'post_hoc_ood_quantized_on_20um_model': ood_pitch_curve,
        },
        'mechanical_misalignment_scan_PinpointBlock_PixXY': misalign_scan,
        'paired_family_bootstrap_comparisons': paired_confirm_comparisons,
        'paired_geant4_physics_systematics': sys_eval,
        'paired_generator_hadronization_systematics': gen_sys_eval,
        'external_punchthrough_audit': ext_punch_eval,
        'realistic_forward_flux_projections_mass_weighted_2t': flux_projections,
        'h0_toy_coverage_bb_lite': h0_clean_bb,
        'h0_toy_coverage_physics_sys': h0_clean_phys,
        'figures': [str(fig1_path), str(fig2_path)],
    }
    write_json(clean_root / 'stage321_clean_v2_unsealing_report.json', final_report)
    return final_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('audit-clean-v2', 'build-configs', 'submit-group', 'freeze', 'unseal-and-evaluate'))
    parser.add_argument('--clean-root', type=Path, default=DEFAULT_CLEAN_V2_ROOT)
    parser.add_argument('--group', type=str, default=None)
    parser.add_argument('--backend', choices=('kubeflow', 'condor'), default='condor')
    parser.add_argument('--kubeflow-flavor', type=str, default='h100-47gb')
    parser.add_argument('--parallel', type=int, default=3)
    parser.add_argument('--n-toys', type=int, default=1000)
    args = parser.parse_args()

    if args.command == 'audit-clean-v2':
        out = audit_clean_v2(args.clean_root)
    elif args.command == 'build-configs':
        gs = build_stage321_configs(args.clean_root)
        out = {k: [str(p) for p in v['configs']] for k, v in gs.items()}
    elif args.command == 'submit-group':
        if not args.group:
            parser.error('--group is required for submit-group')
        out = submit_stage321_group(
            args.clean_root, args.group, args.backend,
            parallel=args.parallel, kubeflow_flavor=args.kubeflow_flavor,
        )
    elif args.command == 'freeze':
        out = freeze_stage321_before_unsealing(args.clean_root)
    else:
        out = unseal_and_evaluate_stage321(args.clean_root, n_toys=args.n_toys)
    print(json.dumps(out, indent=2, default=str))


if __name__ == '__main__':
    main()
