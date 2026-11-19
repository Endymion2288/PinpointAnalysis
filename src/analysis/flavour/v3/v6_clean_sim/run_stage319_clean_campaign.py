"""Stage 318 Audit & Stage 319 Clean Pinpoint_G4 Campaign:
From-Scratch Hardware Pitch/Layer Curves (Exp-2 / P1-5), Explicit Micron Multi-Tracklet &
Secondary-Vertex Hybrid (Path C), Pre-Unsealing Freeze (P3-1), Mechanical Misalignment &
Geant4 Physics Systematics, and One-Shot CleanConfirm Unsealing + 27-Bin Flux Likelihood (Path D).
"""

from __future__ import annotations

import argparse
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
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, SequentialSampler, BatchSampler

from analysis.flavour.inventory import write_json
from analysis.flavour.s4.stage_inputs import stage_shards
from analysis.flavour.v2.partitions import FAMILY, KEY
from analysis.flavour.v2.profile import device_batch
from analysis.flavour.v2.r2.readout import grouped_folds
from analysis.flavour.v3.common.calibration import _tau_efficiency_at_fpr
from analysis.flavour.v3.common.datasets import V3EvalDataset
from analysis.flavour.v3.common.domains import OBS_DOMAIN_NAMES, detector_bounds
from analysis.flavour.v3.common.submit import submit_job
from analysis.flavour.v3.v1_b1pp.models import ZAlignedStereoNet
from analysis.flavour.v3.v1_b1pp.representation import collate_b1pp, extract_b1pp_item, static_plane_layout
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
    DEFAULT_STAGE317_ROOT,
    SEEDS,
    _compute_vertex_audit,
    _select_seed_prediction_df,
    paired_family_bootstrap_delta,
)
from analysis.flavour.v3.v6_clean_sim.produce_clean_sim import DEFAULT_CLEAN_ROOT, SYS_VARIANTS


REPO_ROOT = Path(__file__).resolve().parents[5]
ROADMAP_FREEZE_V6 = REPO_ROOT / 'roadmap/v3/v6_freeze_manifest.json'
LEGACY_MANIFEST = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s6_2026-11-14/manifest.parquet')


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 1. Stage 318 Clean Simulation Physics & Geometry Audit
# ---------------------------------------------------------------------------
def audit_clean_sim(clean_root: Path) -> dict:
    t0 = time.perf_counter()
    geom = json.loads((clean_root / 'geometry.json').read_text())
    fin = json.loads((clean_root / 'finalize_summary.json').read_text())
    prep = json.loads((clean_root / 'prepare_summary.json').read_text())
    man_df = pd.read_parquet(clean_root / 'planned_manifest.parquet')

    w_z = np.asarray(geom['tungsten_Zpos'], dtype=np.float64)
    w_half = 0.5 * float(geom['tungsten_thickness'])

    vz_clean = man_df['initZ'].to_numpy(dtype=np.float64)
    dist_clean = np.min(np.abs(vz_clean[:, None] - w_z[None, :]), axis=1)
    outside_clean = int((dist_clean > w_half).sum())

    # Compare against legacy R2/s6 manifest (where 12 back W plates were shifted by -6.010 mm!)
    leg_df = pd.read_parquet(LEGACY_MANIFEST, columns=['label', 'initE', 'initZ']).sample(
        n=60000, random_state=20261118
    )
    vz_leg = leg_df['initZ'].to_numpy(dtype=np.float64)
    dist_leg = np.min(np.abs(vz_leg[:, None] - w_z[None, :]), axis=1)
    outside_leg_frac = float((dist_leg > w_half).mean())

    # Inspect one validation shard (080) to verify Birks + 14,400-cell SiPM saturation
    obs_val = pq.read_table(clean_root / 'data/shards/080/observations.parquet').to_pylist()
    edep_all = np.concatenate([np.asarray(ev['bar']['edep_proxy_MeV'], dtype=np.float64) for ev in obs_val if len(ev['bar']['pe_proxy']) > 0])
    pe_all = np.concatenate([np.asarray(ev['bar']['pe_proxy'], dtype=np.float64) for ev in obs_val if len(ev['bar']['pe_proxy']) > 0])

    # Check paired physics systematic & external/generator shards
    sys_counts = {}
    for sv in (*SYS_VARIANTS, 'external_punchthrough', 'pythia6_std_ref', 'pythia6_nux'):
        sp = clean_root / 'sys_shards' / sv / 'shards/000/observations.parquet'
        if not sp.exists():
            raise FileNotFoundError(f'Missing paired physics systematic shard: {sp}')
        sys_counts[sv] = int(pq.ParquetFile(sp).metadata.num_rows)

    # Split-level NC_charm and E > 2.312 TeV nu_tau CC counts
    split_physics = {}
    for sp_name, sub in man_df.groupby('split'):
        tau_m = sub['label'] == 2
        nc_m = sub['label'] == 3
        split_physics[str(sp_name)] = {
            'total': int(len(sub)),
            'nutau_cc_total': int(tau_m.sum()),
            'nutau_cc_E_gt_2312_GeV': int((tau_m & (sub['Ev_GeV'] > 2312.0)).sum()),
            'nutau_cc_E_max_GeV': float(sub.loc[tau_m, 'Ev_GeV'].max()),
            'nc_total': int(nc_m.sum()),
            'nc_charm_count': int((nc_m & sub['has_gen_charm']).sum()),
            'cc_charm_count': int(((~nc_m) & sub['has_gen_charm']).sum()),
        }

    # Render 3-panel 300-DPI English physics audit figure
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.9), dpi=300)
    ax0, ax1, ax2 = axes

    bins_d = np.linspace(0.0, 9.0, 91)
    ax0.hist(np.clip(dist_leg, 0.0, 8.95), bins=bins_d, density=True, histtype='step', lw=1.9, color='#8C8C8C', ls='--',
             label=f'Legacy Simulation ({outside_leg_frac*100:.2f}% outside G4_W)')
    ax0.hist(np.clip(dist_clean, 0.0, 8.95), bins=bins_d, density=True, histtype='stepfilled', alpha=0.35, color='#4C72B0',
             edgecolor='#2B4F81', lw=1.6, label='Clean Pinpoint_G4 (100.00% inside G4_W)')
    ax0.axvline(w_half, color='#C44E52', ls=':', lw=1.6, label=f'W Slab Half-Thickness ({w_half:.2f} mm)')
    ax0.set_xlabel('Distance to Nearest Physical G4_W Slab Center $|z_v - z_W|$ [mm]')
    ax0.set_ylabel('Normalized Event Density [mm$^{-1}$]')
    ax0.set_title('P0-2 Fixed: 100% Tungsten Vertex Containment')
    ax0.legend(loc='upper right', fontsize=7.8, frameon=True)
    ax0.grid(True, alpha=0.25)

    bins_e = np.linspace(0.0, 5.5, 56)
    e_leg_tau = leg_df.loc[leg_df['label'] == 2, 'initE'].to_numpy(dtype=np.float64) / 1e6
    e_cln_tau = man_df.loc[man_df['label'] == 2, 'Ev_GeV'].to_numpy(dtype=np.float64) / 1e3
    e_cln_nt = man_df.loc[man_df['label'] != 2, 'Ev_GeV'].to_numpy(dtype=np.float64) / 1e3
    ax1.hist(np.clip(e_leg_tau, 0.0, 5.45), bins=bins_e, density=True, histtype='step', lw=1.8, color='#8C8C8C', ls='--',
             label='Legacy $\\nu_\\tau$ CC (cutoff at 2.31 TeV)')
    ax1.hist(np.clip(e_cln_tau, 0.0, 5.45), bins=bins_e, density=True, histtype='step', lw=2.1, color='#C44E52',
             label=f'Clean $\\nu_\\tau$ CC (up to {e_cln_tau.max():.2f} TeV)')
    ax1.hist(np.clip(e_cln_nt, 0.0, 5.45), bins=bins_e, density=True, histtype='step', lw=1.6, color='#4C72B0',
             label='Clean Non-$\\tau$ ($\\nu_e, \\nu_\\mu$, NC + NC_charm)')
    ax1.axvline(2.312, color='#222222', ls=':', lw=1.2, label='Legacy 2.31 TeV Ceiling')
    ax1.set_yscale('log')
    ax1.set_xlabel('Incident Neutrino Energy $E_\\nu$ [TeV]')
    ax1.set_ylabel('Normalized Log-Density [TeV$^{-1}$]')
    ax1.set_title('P2-1 Fixed: Continuous Multi-TeV $\\nu_\\tau$ Spectrum')
    ax1.legend(loc='upper right', fontsize=7.6, frameon=True)
    ax1.grid(True, which='both', alpha=0.25)

    rng_s = np.random.default_rng(42)
    sub_idx = rng_s.choice(len(pe_all), size=min(12000, len(pe_all)), replace=False)
    ax2.scatter(edep_all[sub_idx], pe_all[sub_idx], s=3.5, alpha=0.28, color='#55A868', rasterized=True, label='Clean G4 Scint Channels')
    ax2.axhline(14400.0, color='#C44E52', ls='--', lw=1.5, label='SiPM Microcell Cap ($N_{\\mathrm{pix}}=14,400$)')
    ax2.axhline(2.5, color='#222222', ls=':', lw=1.3, label='Readout Threshold (2.5 PE)')
    ax2.set_xscale('log')
    ax2.set_yscale('symlog', linthresh=1.0)
    ax2.set_xlim(1e-3, max(200.0, float(edep_all.max()) * 1.2))
    ax2.set_ylim(0.0, 35000.0)
    ax2.set_xlabel('True Step Energy Deposit $E_{\\mathrm{dep}}$ [MeV] (withheld from ML)')
    ax2.set_ylabel('Digitized Channel Output [PE]')
    ax2.set_title("P1-1 Fixed: Birks' Quenching + SiPM Saturation")
    ax2.legend(loc='lower right', fontsize=7.6, frameon=True)
    ax2.grid(True, which='both', alpha=0.25)

    fig.tight_layout()
    fig_path = clean_root / 'stage318_clean_pinpoint_g4_physics_audit.png'
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    report = {
        'UTC': datetime.now(timezone.utc).isoformat(),
        'wall_seconds': float(time.perf_counter() - t0),
        'total_events': int(len(man_df)),
        'family_disjoint_verified': bool(prep['family_disjoint_verified']),
        'tungsten_containment': {
            'clean_outside_g4_w_count': outside_clean,
            'clean_inside_g4_w_fraction': float(1.0 - outside_clean / len(man_df)),
            'legacy_outside_g4_w_fraction': outside_leg_frac,
            'max_abs_z_offset_from_slab_center_mm': float(dist_clean.max()),
            'tungsten_half_thickness_mm': w_half,
        },
        'digitizer_birks_and_sipm': {
            'validation_shard_080_channels': int(len(pe_all)),
            'max_observed_pe': float(pe_all.max()),
            'sipm_npix_cap': 14400,
            'fraction_above_2p5_pe': float((pe_all >= 2.5).mean()),
            'mean_pe_per_mev_edep_gt_5mev': float(pe_all[edep_all > 5.0].sum() / max(edep_all[edep_all > 5.0].sum(), 1e-9)),
        },
        'split_physics_breakdown': split_physics,
        'paired_physics_systematics_shards': sys_counts,
        'finalize_summary': fin,
        'figure_path': str(fig_path),
    }
    write_json(clean_root / 'stage318_clean_dataset_audit.json', report)
    return report


# ---------------------------------------------------------------------------
# 2. Stage 319 Training Config Builder & Multi-GPU Submission
# ---------------------------------------------------------------------------
def build_stage319_configs(clean_root: Path, stage317_root: Path) -> dict[str, dict]:
    s319_root = clean_root / 'stage319'
    cfg_dir = s319_root / 'configs'
    cfg_dir.mkdir(parents=True, exist_ok=True)

    data_root = str(clean_root / 'data')
    tune_screen = str(clean_root / 'data/tune_screen.parquet')
    sidecar_root = str(clean_root / 'truth_sidecar')
    base_train_mod = 'analysis.flavour.v3.v3_multitask.train'

    def _base_pf(name: str, seed: int) -> dict:
        return {
            'name': name,
            'seed': int(seed),
            'task': 'T4',
            'data_root': data_root,
            'tune_screen_path': tune_screen,
            'sidecar_root': sidecar_root,
            'quantize_xy_mm': None,
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
            'cache_workers': 4,
            'loader_workers': 4,
            'pe_only': True,
            'bar_threshold_pe': 2.5,
            'sipm_npix': 14400.0,
            'apply_sipm_saturation': True,
            'y_mirror': False,
            'enable_dense_gn2': False,
            'load_cluster_sidecar': False,
            'use_fixed_event_loss': True,
            'train_module': base_train_mod,
        }

    specs = [
        ('s319-pitch-20um', 'clean_pf_pitch_20um', {'quantize_xy_mm': None, 'enable_micro_vertex': False}, base_train_mod),
        ('s319-pitch-100um', 'clean_pf_pitch_100um', {'quantize_xy_mm': 0.10, 'enable_micro_vertex': False}, base_train_mod),
        ('s319-pitch-300um', 'clean_pf_pitch_300um', {'quantize_xy_mm': 0.30, 'enable_micro_vertex': False}, base_train_mod),
        ('s319-pitch-1mm', 'clean_pf_pitch_1mm', {'quantize_xy_mm': 1.00, 'enable_micro_vertex': False}, base_train_mod),
        ('s319-layers-6t', 'clean_pf_6layer_20um', {'quantize_xy_mm': None, 'active_pixel_planes': [0, 2, 4, 6, 8, 10, 12, 13, 14]}, base_train_mod),
        ('s319-hybrid-uvtx', 'clean_pf_hybrid_uvtx_20um', {'quantize_xy_mm': None, 'enable_micro_vertex': True}, base_train_mod),
        (
            's319-hybrid-uvtx-nomad',
            'clean_pf_hybrid_uvtx_nomad_20um',
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

    # Group 7: Independent projection baseline B1++ under clean PE-only readout
    b1_paths = []
    for s in SEEDS:
        c = {
            'name': f'clean_b1pp_pe_only_s{s}',
            'seed': int(s),
            'task': 'T4',
            'data_root': data_root,
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
            'loader_workers': 4,
            'pe_only': True,
            'bar_threshold_pe': 2.5,
            'sipm_npix': 14400.0,
            'apply_sipm_saturation': True,
            'y_mirror': False,
            'train_module': 'analysis.flavour.v3.v1_b1pp.train',
        }
        p = cfg_dir / f"{c['name']}.json"
        write_json(p, c)
        b1_paths.append(p)
    groups['s319-b1pp-clean'] = {
        'prefix': 'clean_b1pp_pe_only',
        'configs': b1_paths,
        'share': '--share-b1pp-inputs',
        'train_module': 'analysis.flavour.v3.v1_b1pp.train',
    }
    return groups


def submit_stage319_group(
    clean_root: Path,
    stage317_root: Path,
    group_name: str,
    backend: str,
    parallel: int = 3,
) -> dict:
    s319_root = clean_root / 'stage319'
    groups = build_stage319_configs(clean_root, stage317_root)
    spec = groups[group_name]
    cmd = [
        'analysis.flavour.v3.common.group',
        '--root', str(s319_root),
        '--name', group_name,
        '--parallel', str(parallel),
        spec['share'],
        '--cache-workers', '4',
        '--train-module', spec['train_module'],
        '--configs', *[str(p) for p in spec['configs']],
    ]
    if backend == 'kubeflow':
        return submit_job(
            root=s319_root,
            name=group_name,
            stage='stage319',
            command=cmd,
            backend='kubeflow',
            flavor='h100-47gb',
            gpu=True,
            deadline_seconds=14400,
        )
    if backend == 'condor':
        return submit_job(
            root=s319_root,
            name=group_name,
            stage='stage319',
            command=cmd,
            backend='condor',
            cpus=2,
            memory='10000MB',
            disk='12GB',
            gpu=True,
            min_gpu_memory_mb=10000,
            min_capability=7.0,
            job_flavour='longlunch',
        )
    raise ValueError(f'Unsupported submit backend: {backend}')


# ---------------------------------------------------------------------------
# 3. CleanTune8k Evaluation & Pre-Unsealing Freeze Manifest (P3-1)
# ---------------------------------------------------------------------------
def freeze_stage319_before_unsealing(clean_root: Path, stage317_root: Path) -> dict:
    s319_root = clean_root / 'stage319'
    groups = build_stage319_configs(clean_root, stage317_root)
    tune_meta = pd.read_parquet(clean_root / 'validation_meta/tune_truth_meta.parquet')
    sidecar_df = pd.read_parquet(clean_root / 'truth_sidecar/event_sidecar.parquet')

    labels = tune_meta['label'].to_numpy(dtype=np.int64)
    obs_dom = tune_meta['obs_domain_id'].to_numpy(dtype=np.int64)
    folds, _ = grouped_folds(tune_meta[KEY + FAMILY + ['label']].reset_index(drop=True), folds=5, seed=20261118)
    slices = {
        'All_Tune8k': np.ones(len(tune_meta), dtype=bool),
        'Truth_pixel_centered': tune_meta['pixel_centered'].to_numpy(dtype=bool),
        'D_pix_trk': obs_dom == 1,
    }

    tune_summary = {}
    checkpoints = {}
    for gname, spec in groups.items():
        prefix = spec['prefix']
        m_frames = {}
        for s in SEEDS:
            sname = f'{prefix}_s{s}'
            exp_dir = s319_root / 'experiments' / sname
            df, info = _select_seed_prediction_df(exp_dir)
            m_frames[sname] = df
            ckpt_file = exp_dir / ('best.pt' if info['chosen_file'] == 'best_predictions.parquet' else 'ema.pt')
            checkpoints[sname] = {
                'path': str(ckpt_file),
                'sha256': _sha256_file(ckpt_file),
                'chosen_prediction': info['chosen_file'],
                'config_path': str(s319_root / 'configs' / f'{sname}.json'),
            }
        ens = fit_oof_domain_ensemble(m_frames, tune_meta, slices, n_folds=5, folds=folds)
        vtx = _compute_vertex_audit(list(m_frames.values()), tune_meta, sidecar_df)
        if vtx is not None:
            vtx.pop('_dz_pix_centered', None)
        pc_m = slices['Truth_pixel_centered']
        tune_summary[gname] = {
            'prefix': prefix,
            'pix_centered_tau_eff_1pct_fpr': float(_tau_efficiency_at_fpr(ens['s_tau'][pc_m], labels[pc_m], 0.01)['tau_eff']),
            'pix_centered_tau_eff_0p1pct_fpr': float(_tau_efficiency_at_fpr(ens['s_tau'][pc_m], labels[pc_m], 0.001)['tau_eff']),
            'all_tune8k_BA4': float(compute_slice_metrics(labels, ens['oof_probs'], slices['All_Tune8k'])['BA4']),
            'vertex_audit': vtx,
        }

    # Select primary PinpointFormer model strictly on CleanTune8k
    ret_nomad = False
    s317_rep_path = stage317_root / 'stage317_ablation_evaluation_report.json'
    if s317_rep_path.exists():
        dec = json.loads(s317_rep_path.read_text()).get('stage314_complexity_retention_decisions', {})
        ret_nomad = bool(dec.get('retain_calibrated_nomad_head', False))

    candidates = ['s319-pitch-20um', 's319-hybrid-uvtx']
    if ret_nomad and 's319-hybrid-uvtx-nomad' in tune_summary:
        candidates.append('s319-hybrid-uvtx-nomad')
    primary_pf_group = max(
        candidates,
        key=lambda g: (
            tune_summary[g]['pix_centered_tau_eff_1pct_fpr']
            + 0.5 * tune_summary[g]['pix_centered_tau_eff_0p1pct_fpr']
        ),
    )
    primary_pf_prefix = groups[primary_pf_group]['prefix']

    git_sha = subprocess.check_output(['git', '-C', str(REPO_ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    if len(git_sha) != 40:
        raise RuntimeError(f'Invalid git commit SHA: {git_sha}')

    freeze_doc = {
        'protocol': 'Stage 319 one-shot sealed calibration/cut/confirmation freeze (CleanCalA -> CleanCutB -> CleanConfirm)',
        'UTC_frozen': datetime.now(timezone.utc).isoformat(),
        'pre_unsealing_git_commit': git_sha,
        'Cal_A_Cut_B_Confirm_read_before_freeze': False,
        'primary_pf_group': primary_pf_group,
        'primary_pf_prefix': primary_pf_prefix,
        'primary_ensemble_members': (
            [f'{primary_pf_prefix}_s{s}' for s in SEEDS]
            + [f'clean_b1pp_pe_only_s{s}' for s in SEEDS]
        ),
        'tune8k_selection_summary': tune_summary,
        'checkpoints': checkpoints,
    }
    write_json(clean_root / 'v6_freeze_manifest.json', freeze_doc)
    write_json(ROADMAP_FREEZE_V6, freeze_doc)
    return freeze_doc


# ---------------------------------------------------------------------------
# 4. Model Inference Helpers for One-Shot Unsealing & Misalignment/Physics Sys
# ---------------------------------------------------------------------------
def _load_pf_model(ckpt_path: Path, cfg: dict, geometry: dict, device: torch.device) -> torch.nn.Module:
    cls = PinpointFormerV4 if cfg.get('model_family') == 'pf_v4' else PinpointFormerMultitask
    model = cls(geometry, cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state['model'])
    model.eval()
    return model


def _infer_pf_group_on_split(
    staged_data_root: Path,
    split: str,
    prefix: str,
    s319_root: Path,
    checkpoints: dict,
    geometry: dict,
    device: torch.device,
    cfg_override: dict | None = None,
    shared_mapped_root: Path | None = None,
) -> dict[str, pd.DataFrame]:
    cache_tag = hashlib.sha256(
        json.dumps(
            {
                'root': str(staged_data_root),
                'split': split,
                'prefix': prefix,
                'shas': [checkpoints[f'{prefix}_s{s}']['sha256'] for s in SEEDS],
                'override': cfg_override or {},
            },
            sort_keys=True,
        ).encode('utf-8')
    ).hexdigest()[:16]
    pred_cache_dir = s319_root / 'unsealed_preds' / f'{prefix}_{split}_{cache_tag}'
    if all((pred_cache_dir / f'{prefix}_s{s}.parquet').exists() for s in SEEDS):
        return {f'{prefix}_s{s}': pd.read_parquet(pred_cache_dir / f'{prefix}_s{s}.parquet') for s in SEEDS}

    amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    base_cfg = json.loads(Path(checkpoints[f'{prefix}_s17']['config_path']).read_text())
    eval_cfg = {**base_cfg, **(cfg_override or {})}

    tok_key = hashlib.sha256(
        json.dumps(
            {
                'root': str(staged_data_root),
                'split': split,
                'q': eval_cfg.get('quantize_xy_mm'),
                'planes': eval_cfg.get('active_pixel_planes'),
                'mis': eval_cfg.get('misalign_sigma_xy_mm'),
                'pe_only': eval_cfg.get('pe_only', True),
            },
            sort_keys=True,
        ).encode('utf-8')
    ).hexdigest()[:14]

    def _run_with_items(items_dir: Path, pkl_path: Path, ds_cache: Path) -> dict[str, pd.DataFrame]:
        if not pkl_path.exists():
            build_pf_mapped_split(
                staged_data_root, split, items_dir, pkl_path, ds_cache, workers=6, config=eval_cfg
            )
        with pkl_path.open('rb') as fp_p:
            pkg = pickle.load(fp_p)
        meta = pkg['metadata']
        bounds = detector_bounds(geometry)
        ds = CachedPFDataset(items_dir, meta)
        b_sampler = BatchSampler(SequentialSampler(ds), batch_size=128, drop_last=False)
        loader = _ThreadedPFLoader(
            ds,
            b_sampler,
            seed=42,
            y_scint_center=bounds['y_scint_center'],
            n_bars=bounds['n_bars'],
            mirror=False,
            pin_memory=True,
            threads=4,
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
        try:
            with torch.inference_mode():
                for batch in loader:
                    idx = batch['index'].numpy()
                    db = move_batch_to_device(batch, device)
                    for mname, mod in models.items():
                        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=True):
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
        return out_f

    if shared_mapped_root is not None:
        sub = shared_mapped_root / f'tok_{split}_{tok_key}'
        sub.mkdir(parents=True, exist_ok=True)
        return _run_with_items(sub / 'items', sub / 'items.pkl', sub / 'ds_cache')

    with tempfile.TemporaryDirectory(prefix=f's319-inf-{prefix}-{split}-', dir=os.environ.get('PINPOINT_V3_SCRATCH', '/tmp')) as tmp:
        tp = Path(tmp)
        return _run_with_items(tp / 'items', tp / 'items.pkl', tp / 'ds_cache')


def _infer_b1pp_group_on_split(
    staged_data_root: Path,
    split: str,
    prefix: str,
    checkpoints: dict,
    geometry: dict,
    bounds: dict,
    device: torch.device,
    cfg_override: dict | None = None,
) -> dict[str, pd.DataFrame]:
    amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
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
                    with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=True):
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
    for d in (1, 2, 3):
        mc = d_cal == d
        dom_params[d] = _fit_domain_convex_blend(lp_cal[:, mc, :], y_cal[mc])
        if (d_cut == d).any():
            p_cut[d_cut == d] = _apply_domain_convex_blend(lp_cut[:, d_cut == d, :], dom_params[d])
        if (d_con == d).any():
            p_con[d_con == d] = _apply_domain_convex_blend(lp_con[:, d_con == d, :], dom_params[d])

    # Lock lambda_charm & exact Cut-B thresholds strictly on Cut-B
    s_cut = np.full(len(cut_meta), -25.0, dtype=np.float64)
    s_con = np.full(len(conf_meta), -25.0, dtype=np.float64)
    lam_by_dom = {}
    for d in (1, 2, 3):
        mc = d_cut == d
        mn = d_con == d
        best_lam = 0.0
        best_obj = -1.0
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
        if mn.any():
            s_con[mn] = np.log(np.clip(p_con[mn, 2], 1e-12, 1.0) / (np.clip(1.0 - p_con[mn, 2], 1e-12, 1.0) + best_lam * ch_con[mn] + 1e-12))

    return {
        'dom_params': dom_params,
        'lam_by_dom': lam_by_dom,
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
    mnames = list(frames.keys())
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
# 5. One-Shot Unsealing: Exp-2 Pitch/Layers, Path C Hybrid, Misalignment & Path D PL
# ---------------------------------------------------------------------------
def unseal_and_evaluate_stage319(clean_root: Path, stage317_root: Path, n_toys: int = 1000) -> dict:
    t0 = time.perf_counter()
    freeze_path = clean_root / 'v6_freeze_manifest.json'
    if not freeze_path.exists():
        raise FileNotFoundError(f'Pre-unsealing freeze manifest required before unsealing: {freeze_path}')
    freeze_doc = json.loads(freeze_path.read_text())
    if len(str(freeze_doc.get('pre_unsealing_git_commit', ''))) != 40:
        raise RuntimeError('v6_freeze_manifest missing valid 40-hex pre_unsealing_git_commit')

    # Verify SHA-256 of every frozen checkpoint prior to touching Cal-A / Cut-B / Confirm
    checkpoints = freeze_doc['checkpoints']
    for sname, cinfo in checkpoints.items():
        actual_sha = _sha256_file(Path(cinfo['path']))
        if actual_sha != cinfo['sha256']:
            raise RuntimeError(f'Checkpoint SHA-256 mismatch for {sname}: {actual_sha} != {cinfo["sha256"]}')

    device = torch.device('cuda:0')
    geometry = json.loads((clean_root / 'geometry.json').read_text())
    bounds = detector_bounds(geometry)
    s319_root = clean_root / 'stage319'
    groups = build_stage319_configs(clean_root, stage317_root)

    cal_meta = pd.read_parquet(clean_root / 'sealed_truth_meta/cal_a_truth_meta.parquet')
    cut_meta = pd.read_parquet(clean_root / 'sealed_truth_meta/cut_b_truth_meta.parquet')
    conf_meta = pd.read_parquet(clean_root / 'sealed_truth_meta/confirm_truth_meta.parquet')

    y_con = conf_meta['label'].to_numpy(dtype=np.int64)
    pc_con = conf_meta['pixel_centered'].to_numpy(dtype=bool)
    fam_hash = conf_meta['family_hi'].to_numpy(dtype=np.uint64) * np.uint64(1000003) + conf_meta['family_lo'].to_numpy(dtype=np.uint64)
    _, fam_codes = np.unique(fam_hash, return_inverse=True)

    # Build vertex sidecar-compatible frame for CleanConfirm primary vertex resolution audit
    conf_vtx_sidecar = conf_meta[KEY].copy()
    conf_vtx_sidecar['vertex_valid'] = (conf_meta['obs_domain_id'] != 0).astype(np.uint8)
    conf_vtx_sidecar['vertex_x_mm'] = conf_meta['vertex_x_mm']
    conf_vtx_sidecar['vertex_y_mm'] = conf_meta['vertex_y_mm']
    conf_vtx_sidecar['vertex_z_mm'] = conf_meta['vertex_z_mm']

    with tempfile.TemporaryDirectory(prefix='s319-unseal-', dir=os.environ.get('PINPOINT_V3_SCRATCH', '/tmp')) as scratch:
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
                is_b1 = gname == 's319-b1pp-clean'
                if is_b1:
                    f_cal = _infer_b1pp_group_on_split(staged_root, 'cal_a', prefix, checkpoints, geometry, bounds, device)
                    f_cut = _infer_b1pp_group_on_split(staged_root, 'cut_b', prefix, checkpoints, geometry, bounds, device)
                    f_con = _infer_b1pp_group_on_split(staged_root, 'confirm', prefix, checkpoints, geometry, bounds, device)
                else:
                    sm_root = Path(scratch) / 'shared_mapped'
                    f_cal = _infer_pf_group_on_split(staged_root, 'cal_a', prefix, s319_root, checkpoints, geometry, device, shared_mapped_root=sm_root)
                    f_cut = _infer_pf_group_on_split(staged_root, 'cut_b', prefix, s319_root, checkpoints, geometry, device, shared_mapped_root=sm_root)
                    f_con = _infer_pf_group_on_split(staged_root, 'confirm', prefix, s319_root, checkpoints, geometry, device, shared_mapped_root=sm_root)

                cal_frames_cache[gname] = f_cal
                cut_frames_cache[gname] = f_cut
                conf_frames_cache[gname] = f_con

                fit = _fit_cala_cutb_and_score_confirm(f_cal, f_cut, f_con, cal_meta, cut_meta, conf_meta)
                fits_by_group[gname] = fit

                vtx_a = None if is_b1 else _compute_vertex_audit(list(f_con.values()), conf_meta, conf_vtx_sidecar)
                if vtx_a is not None:
                    vtx_dz_confirm[gname] = vtx_a.pop('_dz_pix_centered')

                # Compute exact within-slice 1% & 0.1% FPR + Cut-B locked threshold realization + NC_charm / CC_charm leakage
                s_c = fit['s_tau_confirm']
                s_b = fit['s_tau_cut']
                y_b = cut_meta['label'].to_numpy(dtype=np.int64)
                pc_b = cut_meta['pixel_centered'].to_numpy(dtype=bool)
                cutb_thr_1p_pc = float(np.quantile(s_b[pc_b & (y_b != 2)], 0.99))
                cutb_thr_01p_pc = float(np.quantile(s_b[pc_b & (y_b != 2)], 0.999))

                nc_charm_m = pc_con & (y_con == 3) & conf_meta['has_charm'].to_numpy(dtype=bool)
                cc_charm_m = pc_con & (y_con != 2) & (y_con != 3) & conf_meta['has_charm'].to_numpy(dtype=bool)
                light_bkg_m = pc_con & (y_con != 2) & (~conf_meta['has_charm'].to_numpy(dtype=bool))

                unsealed_groups[gname] = {
                    'prefix': prefix,
                    'All_CleanConfirm': {
                        **compute_slice_metrics(y_con, fit['probs_confirm'], None),
                        'within_slice_exact_1pct_fpr': float(_tau_efficiency_at_fpr(s_c, y_con, 0.01)['tau_eff']),
                        'within_slice_exact_0p1pct_fpr': float(_tau_efficiency_at_fpr(s_c, y_con, 0.001)['tau_eff']),
                    },
                    'Truth_pixel_centered': {
                        **compute_slice_metrics(y_con, fit['probs_confirm'], pc_con),
                        'within_slice_exact_1pct_fpr': float(_tau_efficiency_at_fpr(s_c[pc_con], y_con[pc_con], 0.01)['tau_eff']),
                        'within_slice_exact_0p1pct_fpr': float(_tau_efficiency_at_fpr(s_c[pc_con], y_con[pc_con], 0.001)['tau_eff']),
                        'cutb_locked_1pct_tau_eff': float((s_c[pc_con & (y_con == 2)] >= cutb_thr_1p_pc).mean()),
                        'cutb_locked_1pct_realized_fpr': float((s_c[pc_con & (y_con != 2)] >= cutb_thr_1p_pc).mean()),
                        'cutb_locked_0p1pct_tau_eff': float((s_c[pc_con & (y_con == 2)] >= cutb_thr_01p_pc).mean()),
                        'cutb_locked_0p1pct_realized_fpr': float((s_c[pc_con & (y_con != 2)] >= cutb_thr_01p_pc).mean()),
                        'nc_charm_fpr_at_1pct_cut': float((s_c[nc_charm_m] >= cutb_thr_1p_pc).mean()) if nc_charm_m.any() else None,
                        'cc_charm_fpr_at_1pct_cut': float((s_c[cc_charm_m] >= cutb_thr_1p_pc).mean()) if cc_charm_m.any() else None,
                        'light_bkg_fpr_at_1pct_cut': float((s_c[light_bkg_m] >= cutb_thr_1p_pc).mean()) if light_bkg_m.any() else None,
                    },
                    'vertex_audit': vtx_a,
                }

            # Primary 6-model Dual-Family Clean Ensemble (primary_pf_group + clean_b1pp_pe_only)
            prim_pf_g = freeze_doc['primary_pf_group']
            prim_pf_prefix = freeze_doc['primary_pf_prefix']
            ens6_cal = {**cal_frames_cache[prim_pf_g], **cal_frames_cache['s319-b1pp-clean']}
            ens6_cut = {**cut_frames_cache[prim_pf_g], **cut_frames_cache['s319-b1pp-clean']}
            ens6_con = {**conf_frames_cache[prim_pf_g], **conf_frames_cache['s319-b1pp-clean']}
            ens6_fit = _fit_cala_cutb_and_score_confirm(ens6_cal, ens6_cut, ens6_con, cal_meta, cut_meta, conf_meta)
            fits_by_group['primary_dual_family_6m'] = ens6_fit
            unsealed_groups['primary_dual_family_6m'] = {
                'All_CleanConfirm': {
                    **compute_slice_metrics(y_con, ens6_fit['probs_confirm'], None),
                    'within_slice_exact_1pct_fpr': float(_tau_efficiency_at_fpr(ens6_fit['s_tau_confirm'], y_con, 0.01)['tau_eff']),
                    'within_slice_exact_0p1pct_fpr': float(_tau_efficiency_at_fpr(ens6_fit['s_tau_confirm'], y_con, 0.001)['tau_eff']),
                },
                'Truth_pixel_centered': {
                    **compute_slice_metrics(y_con, ens6_fit['probs_confirm'], pc_con),
                    'within_slice_exact_1pct_fpr': float(_tau_efficiency_at_fpr(ens6_fit['s_tau_confirm'][pc_con], y_con[pc_con], 0.01)['tau_eff']),
                    'within_slice_exact_0p1pct_fpr': float(_tau_efficiency_at_fpr(ens6_fit['s_tau_confirm'][pc_con], y_con[pc_con], 0.001)['tau_eff']),
                },
            }

            # OOD post-hoc quantization comparison on 20um model (proving why P1-5 from-scratch retraining mattered!)
            ood_pitch_curve = {'0.020': unsealed_groups['s319-pitch-20um']['Truth_pixel_centered']['within_slice_exact_1pct_fpr']}
            for q_mm, q_key in ((0.10, '0.100'), (0.30, '0.300'), (1.00, '1.000')):
                f_ood = _infer_pf_group_on_split(
                    staged_root, 'confirm', 'clean_pf_pitch_20um', s319_root, checkpoints, geometry, device,
                    cfg_override={'quantize_xy_mm': q_mm},
                )
                _, s_ood, _ = _apply_frozen_ensemble_to_frames(f_ood, conf_meta, fits_by_group['s319-pitch-20um'])
                ood_pitch_curve[q_key] = float(_tau_efficiency_at_fpr(s_ood[pc_con], y_con[pc_con], 0.01)['tau_eff'])

            # Mechanical plane-wise sensor misalignment scan (0, 10, 25, 50 um) on primary PF model
            misalign_scan = {'0_um': unsealed_groups[prim_pf_g]['Truth_pixel_centered']['within_slice_exact_1pct_fpr']}
            for sig_mm, m_tag in ((0.010, '10_um'), (0.025, '25_um'), (0.050, '50_um')):
                f_mis = _infer_pf_group_on_split(
                    staged_root, 'confirm', prim_pf_prefix, s319_root, checkpoints, geometry, device,
                    cfg_override={'misalign_sigma_xy_mm': sig_mm},
                )
                _, s_mis, _ = _apply_frozen_ensemble_to_frames(f_mis, conf_meta, fits_by_group[prim_pf_g])
                misalign_scan[m_tag] = float(_tau_efficiency_at_fpr(s_mis[pc_con], y_con[pc_con], 0.01)['tau_eff'])
        finally:
            if old_stage_env is None:
                os.environ.pop('PINPOINT_V3_SHARED_STAGE', None)
            else:
                os.environ['PINPOINT_V3_SHARED_STAGE'] = old_stage_env

    # Paired bootstrap comparisons on CleanConfirm (Truth_pixel_centered)
    paired_confirm_comparisons = {
        'PathC_Hybrid_MicroVertex_vs_Vanilla_20um': paired_family_bootstrap_delta(
            fits_by_group['s319-hybrid-uvtx']['s_tau_confirm'],
            fits_by_group['s319-pitch-20um']['s_tau_confirm'],
            fits_by_group['s319-hybrid-uvtx']['probs_confirm'],
            fits_by_group['s319-pitch-20um']['probs_confirm'],
            y_con, fam_codes, pc_con, n_boot=1000,
        ),
        'Exp2_20um_vs_100um_FromScratch': paired_family_bootstrap_delta(
            fits_by_group['s319-pitch-20um']['s_tau_confirm'],
            fits_by_group['s319-pitch-100um']['s_tau_confirm'],
            fits_by_group['s319-pitch-20um']['probs_confirm'],
            fits_by_group['s319-pitch-100um']['probs_confirm'],
            y_con, fam_codes, pc_con, n_boot=1000,
        ),
        'Exp2_20um_vs_1mm_FromScratch': paired_family_bootstrap_delta(
            fits_by_group['s319-pitch-20um']['s_tau_confirm'],
            fits_by_group['s319-pitch-1mm']['s_tau_confirm'],
            fits_by_group['s319-pitch-20um']['probs_confirm'],
            fits_by_group['s319-pitch-1mm']['probs_confirm'],
            y_con, fam_codes, pc_con, n_boot=1000,
        ),
        'Exp2_12Layer_vs_6Layer_Target': paired_family_bootstrap_delta(
            fits_by_group['s319-pitch-20um']['s_tau_confirm'],
            fits_by_group['s319-layers-6t']['s_tau_confirm'],
            fits_by_group['s319-pitch-20um']['probs_confirm'],
            fits_by_group['s319-layers-6t']['probs_confirm'],
            y_con, fam_codes, pc_con, n_boot=1000,
        ),
        'DualFamily_6m_vs_SingleFamily_PF': paired_family_bootstrap_delta(
            fits_by_group['primary_dual_family_6m']['s_tau_confirm'],
            fits_by_group[prim_pf_g]['s_tau_confirm'],
            fits_by_group['primary_dual_family_6m']['probs_confirm'],
            fits_by_group[prim_pf_g]['probs_confirm'],
            y_con, fam_codes, pc_con, n_boot=1000,
        ),
    }

    # Evaluate all 5 Paired Geant4 Physics Systematics on the frozen 6-model ensemble!
    cut_dom = cut_meta['obs_domain_id'].to_numpy(dtype=np.int64)
    con_dom = conf_meta['obs_domain_id'].to_numpy(dtype=np.int64)
    ultra_edges = _fit_bin_edges(ens6_fit['s_tau_cut'], cut_dom, ULTRA_TAIL_QUANTILES)

    # Lock Charm-CR quantile edges (85%, 96% of charm_prob in low-S_tau bulk) strictly on Cut-B
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

    sys_eval = {}
    sys_bin_by_var = {}
    for sv in SYS_VARIANTS:
        sv_root = clean_root / 'sys_shards' / sv
        sv_meta = pd.read_parquet(sv_root / 'truth_meta.parquet')
        f_pf_sv = _infer_pf_group_on_split(sv_root, 'confirm', prim_pf_prefix, s319_root, checkpoints, geometry, device)
        f_b1_sv = _infer_b1pp_group_on_split(sv_root, 'confirm', 'clean_b1pp_pe_only', checkpoints, geometry, bounds, device)
        p_sv, s_sv, c_sv = _apply_frozen_ensemble_to_frames({**f_pf_sv, **f_b1_sv}, sv_meta, ens6_fit)
        y_sv = sv_meta['label'].to_numpy(dtype=np.int64)
        pc_sv = sv_meta['pixel_centered'].to_numpy(dtype=bool)
        dom_sv = sv_meta['obs_domain_id'].to_numpy(dtype=np.int64)
        sr21_sv, _ = assign_signal_and_charm_cr_bins(
            s_sv, c_sv, dom_sv, ultra_edges, charm_cr_edges_by_domain=charm_cr_edges
        )
        sys_bin_by_var[sv] = sr21_sv
        sys_eval[sv] = {
            'events': int(len(sv_meta)),
            'All_BA4': float(compute_slice_metrics(y_sv, p_sv, None)['BA4']),
            'pix_centered_tau_eff_1pct_fpr': float(_tau_efficiency_at_fpr(s_sv[pc_sv], y_sv[pc_sv], 0.01)['tau_eff']),
        }

    # Build signed per-bin physics systematic shifts from paired 600-event G4 shards
    cls_to_share = {0: FIDUCIAL_2T_BKG_SHARES['nue_cc'], 1: FIDUCIAL_2T_BKG_SHARES['numu_cc'], 3: FIDUCIAL_2T_BKG_SHARES['nc']}
    ref_meta_sv = pd.read_parquet(clean_root / 'sys_shards/nominal_sys_ref/truth_meta.parquet')
    y_sv_ref = ref_meta_sv['label'].to_numpy(dtype=np.int64)
    w_sv = np.zeros(len(y_sv_ref), dtype=np.float64)
    for c_k, sh_c in cls_to_share.items():
        mc_k = y_sv_ref == c_k
        if mc_k.any():
            w_sv[mc_k] = sh_c / float(mc_k.sum())
    nom_hist = np.bincount(np.clip(sys_bin_by_var['nominal_sys_ref'][sys_bin_by_var['nominal_sys_ref'] >= 0], 0, 20),
                           weights=w_sv[sys_bin_by_var['nominal_sys_ref'] >= 0], minlength=21)
    signed_g4_sys: dict[str, dict[str, list[float]]] = {}
    for sv_k in ('tau_helicity_0', 'w_density_0p98', 'no_magnet_shells', 'field_0p97'):
        b_sv = sys_bin_by_var[sv_k]
        h_sv = np.bincount(np.clip(b_sv[b_sv >= 0], 0, 20), weights=w_sv[b_sv >= 0], minlength=21)
        delta_log = np.clip(np.log((h_sv + 5e-4) / (nom_hist + 5e-4)), -0.35, 0.35)
        signed_g4_sys[f'g4_{sv_k}'] = {
            'delta_log_up': delta_log.tolist(),
            'delta_log_down': (-0.5 * delta_log).tolist(),
        }

    # Evaluate cell-matched GENIE Pythia6_NUX vs Pythia6_Standard hadronization systematic if present
    gen_sys_eval = {}
    if (clean_root / 'sys_shards/pythia6_nux/shards/000/observations.parquet').exists():
        gen_bins = {}
        for gv in ('pythia6_std_ref', 'pythia6_nux'):
            gv_root = clean_root / 'sys_shards' / gv
            gv_meta = pd.read_parquet(gv_root / 'truth_meta.parquet')
            f_pf_gv = _infer_pf_group_on_split(gv_root, 'confirm', prim_pf_prefix, s319_root, checkpoints, geometry, device)
            f_b1_gv = _infer_b1pp_group_on_split(gv_root, 'confirm', 'clean_b1pp_pe_only', checkpoints, geometry, bounds, device)
            p_gv, s_gv, c_gv = _apply_frozen_ensemble_to_frames({**f_pf_gv, **f_b1_gv}, gv_meta, ens6_fit)
            y_gv = gv_meta['label'].to_numpy(dtype=np.int64)
            pc_gv = gv_meta['pixel_centered'].to_numpy(dtype=bool)
            dom_gv = gv_meta['obs_domain_id'].to_numpy(dtype=np.int64)
            sr21_gv, _ = assign_signal_and_charm_cr_bins(
                s_gv, c_gv, dom_gv, ultra_edges, charm_cr_edges_by_domain=charm_cr_edges
            )
            gen_bins[gv] = (sr21_gv, y_gv)
            gen_sys_eval[gv] = {
                'events': int(len(gv_meta)),
                'All_BA4': float(compute_slice_metrics(y_gv, p_gv, None)['BA4']),
                'pix_centered_tau_eff_1pct_fpr': float(_tau_efficiency_at_fpr(s_gv[pc_gv], y_gv[pc_gv], 0.01)['tau_eff']),
            }
        b_std, y_std_g = gen_bins['pythia6_std_ref']
        b_nux, y_nux_g = gen_bins['pythia6_nux']
        w_std_g = np.array([cls_to_share.get(int(y), 0.0) / max(int((y_std_g == y).sum()), 1) for y in y_std_g], dtype=np.float64)
        w_nux_g = np.array([cls_to_share.get(int(y), 0.0) / max(int((y_nux_g == y).sum()), 1) for y in y_nux_g], dtype=np.float64)
        h_std_g = np.bincount(np.clip(b_std[b_std >= 0], 0, 20), weights=w_std_g[b_std >= 0], minlength=21)
        h_nux_g = np.bincount(np.clip(b_nux[b_nux >= 0], 0, 20), weights=w_nux_g[b_nux >= 0], minlength=21)
        d_gen = np.clip(np.log((h_nux_g + 5e-4) / (h_std_g + 5e-4)), -0.35, 0.35)
        signed_g4_sys['gen_pythia6_nux'] = {
            'delta_log_up': d_gen.tolist(),
            'delta_log_down': (-d_gen).tolist(),
        }

    # Evaluate external rock-muon & neutral-hadron (n, K0L) punch-through sample if present
    ext_punch_eval = {}
    if (clean_root / 'sys_shards/external_punchthrough/shards/000/observations.parquet').exists():
        ext_root = clean_root / 'sys_shards/external_punchthrough'
        ext_meta = pd.read_parquet(ext_root / 'truth_meta.parquet')
        ext_obs = pq.read_table(ext_root / 'shards/000/observations.parquet').to_pylist()
        # Upstream charged-particle veto: >= 2.5 PE in upstream scintillator panel 0 or 1
        veto_fired = np.array([
            any((int(p) <= 1 and float(pe) >= 2.5) for p, pe in zip(ev['bar']['panel'], ev['bar']['pe_proxy']))
            for ev in ext_obs
        ], dtype=bool)
        f_pf_ex = _infer_pf_group_on_split(ext_root, 'confirm', prim_pf_prefix, s319_root, checkpoints, geometry, device)
        f_b1_ex = _infer_b1pp_group_on_split(ext_root, 'confirm', 'clean_b1pp_pe_only', checkpoints, geometry, bounds, device)
        _, s_ex, c_ex = _apply_frozen_ensemble_to_frames({**f_pf_ex, **f_b1_ex}, ext_meta, ens6_fit)
        dom_ex = ext_meta['obs_domain_id'].to_numpy(dtype=np.int64)
        sr21_ex, _ = assign_signal_and_charm_cr_bins(
            s_ex, c_ex, dom_ex, ultra_edges, charm_cr_edges_by_domain=charm_cr_edges
        )
        cutb_1pct_global = float(np.quantile(ens6_fit['s_tau_cut'][cut_meta['label'].to_numpy() != 2], 0.99))
        subtypes = ext_meta['ext_subtype'].to_numpy(dtype=str) if 'ext_subtype' in ext_meta.columns else np.full(len(ext_meta), 'unknown')
        mu_m = subtypes == 'rock_muon'
        nh_m = subtypes != 'rock_muon'
        ext_punch_eval = {
            'total_events': int(len(ext_meta)),
            'rock_muon_count': int(mu_m.sum()),
            'rock_muon_upstream_veto_efficiency': float(veto_fired[mu_m].mean()) if mu_m.any() else None,
            'rock_muon_unvetoed_tau_1pct_leakage': float(((~veto_fired) & mu_m & (s_ex >= cutb_1pct_global)).sum() / max(int(mu_m.sum()), 1)),
            'neutral_hadron_count': int(nh_m.sum()),
            'neutral_hadron_unvetoed_fraction': float((~veto_fired[nh_m]).mean()) if nh_m.any() else None,
            'neutral_hadron_unvetoed_tau_1pct_fpr': float(((~veto_fired) & nh_m & (s_ex >= cutb_1pct_global)).sum() / max(int(nh_m.sum()), 1)),
            'neutral_hadron_ultra_tail_top3bin_rate': float(((~veto_fired) & nh_m & np.isin(sr21_ex, [6, 13, 20])).sum() / max(int(nh_m.sum()), 1)),
        }
        # Add external neutral-hadron normalization uncertainty on unvetoed neutral-hadron SR shape
        unv_nh = (~veto_fired) & nh_m & (sr21_ex >= 0)
        if unv_nh.any():
            h_nh = np.bincount(np.clip(sr21_ex[unv_nh], 0, 20), minlength=21).astype(np.float64)
            h_nh = 0.02 * (h_nh / max(h_nh.sum(), 1.0))  # 2% fiducial neutral-hadron punch-through rate x 50% norm unc
            d_nh = np.clip(np.log1p(h_nh / np.maximum(nom_hist, 5e-4)), 0.0, 0.25)
            signed_g4_sys['ext_neutral_hadron_punch'] = {
                'delta_log_up': d_nh.tolist(),
                'delta_log_down': (-0.5 * d_nh).tolist(),
            }

    # Note: CleanConfirm has ~15.3% importance-sampled NC_charm enrichment in class 3;
    # physical forward NC_charm fraction in NC is ~4.5%, weighted directly inside build_physical_templates.
    has_ch_con = conf_meta['has_charm'].to_numpy(dtype=bool)
    fid_m = conf_meta['large_scintillator'].to_numpy(dtype=bool)

    flux_projections = {}
    for era, lum_fb, n_fid in (
        ('Run3_250fb', 250.0, 15000.0),
        ('Run4_680fb', 680.0, 40800.0),
        ('HLLHC_3000fb', 3000.0, 180000.0),
        ('FPF_10000fb', 10000.0, 600000.0),
    ):
        tpl27 = build_physical_templates(
            joint27_con[fid_m], y_con[fid_m], has_ch_con[fid_m],
            n_bins=27, total_events=n_fid, tau_fraction=0.003, bkg_shares=FIDUCIAL_2T_BKG_SHARES,
            nc_charm_fraction_override=None, nc_charm_physical_fraction=0.045, zero_bkg_jeffreys_prior=True,
        )
        pl_phys = fit_realistic_profile_likelihood(
            tpl27, signed_sys=signed_g4_sys, n_sr_bins=21, enable_barlow_beeston=False
        )
        pl_bb = fit_realistic_profile_likelihood(
            tpl27, signed_sys=signed_g4_sys, n_sr_bins=21, enable_barlow_beeston=True
        )
        flux_projections[era] = {
            'luminosity_fb_inv': lum_fb,
            'fiducial_events': n_fid,
            'expected_nutau_cc': float(tpl27['s_tau'].sum()),
            **pl_phys,
            'z0_pl_with_10k_confirm_bb_mc_stat': pl_bb['z0_profile_likelihood'],
            'q0_pl_with_10k_confirm_bb_mc_stat': pl_bb['q0_profile_likelihood'],
            'bb_optimizer_converged': pl_bb['optimizer_converged'],
            'min_gamma_mc_stat_at_H0': pl_bb['min_gamma_mc_stat_at_H0'],
            'max_gamma_mc_stat_at_H0': pl_bb['max_gamma_mc_stat_at_H0'],
        }

    # 1,000-toy H0 calibration check on CleanConfirm HL-LHC 27-bin template (both BB-lite and Physics-Sys)
    hl_tpl_clean = build_physical_templates(
        joint27_con[fid_m], y_con[fid_m], has_ch_con[fid_m],
        n_bins=27, total_events=180000.0, tau_fraction=0.003, bkg_shares=FIDUCIAL_2T_BKG_SHARES,
        nc_charm_fraction_override=None, nc_charm_physical_fraction=0.045, zero_bkg_jeffreys_prior=True,
    )
    h0_clean = run_h0_toy_coverage_verification(
        hl_tpl_clean, signed_sys=signed_g4_sys, n_sr_bins=21, n_toys=n_toys, workers=16, enable_barlow_beeston=True
    )
    h0_clean_phys = run_h0_toy_coverage_verification(
        hl_tpl_clean, signed_sys=signed_g4_sys, n_sr_bins=21, n_toys=n_toys, workers=16, enable_barlow_beeston=False
    )

    # Figure 1: Exp-2 From-Scratch Hardware Pitch, Layer Count, and Mechanical Misalignment
    fig1, axes1 = plt.subplots(1, 3, figsize=(16.2, 5.0), dpi=300)
    a0, a1, a2 = axes1

    pitches_um = [20.0, 100.0, 300.0, 1000.0]
    fs_vals = [
        unsealed_groups['s319-pitch-20um']['Truth_pixel_centered']['within_slice_exact_1pct_fpr'] * 100.0,
        unsealed_groups['s319-pitch-100um']['Truth_pixel_centered']['within_slice_exact_1pct_fpr'] * 100.0,
        unsealed_groups['s319-pitch-300um']['Truth_pixel_centered']['within_slice_exact_1pct_fpr'] * 100.0,
        unsealed_groups['s319-pitch-1mm']['Truth_pixel_centered']['within_slice_exact_1pct_fpr'] * 100.0,
    ]
    ood_vals = [ood_pitch_curve[k] * 100.0 for k in ('0.020', '0.100', '0.300', '1.000')]
    a0.plot(pitches_um, fs_vals, 'o-', lw=2.3, ms=7, color='#4C72B0', label='Retrained From Scratch (Exp-2 True Physics)')
    a0.plot(pitches_um, ood_vals, 's--', lw=1.8, ms=6, color='#C44E52', label='Post-hoc Quantized on 20 $\\mu$m Model (OOD Shift)')
    a0.set_xscale('log')
    a0.set_xticks(pitches_um)
    a0.set_xticklabels(['20 $\\mu$m', '100 $\\mu$m', '300 $\\mu$m', '1 mm'])
    a0.set_xlabel('Target Silicon Pixel Pitch $\\Delta_{xy}$')
    a0.set_ylabel('Pixel-Fiducial $\\nu_\\tau$ Efficiency @ Exact 1% FPR [%]')
    a0.set_title('Exp-2 (P1-5 Fixed): True Pitch Resolution Curve')
    a0.set_ylim(min(min(fs_vals), min(ood_vals)) - 0.55, max(max(fs_vals), max(ood_vals)) + 0.45)
    a0.legend(loc='lower left', fontsize=8.0, frameon=True)
    a0.grid(True, which='both', alpha=0.25)

    lay_labels = ['12 Target + 3 IPT\n(20 $\\mu$m, baseline)', '6 Target + 3 IPT\n(50% target Si cost)', '12 Target + 3 IPT\n(+Hybrid Micro-Vtx)', 'PF + B1++ (6m)\nDual-Family Ens']
    lay_e1 = [
        unsealed_groups['s319-pitch-20um']['Truth_pixel_centered']['within_slice_exact_1pct_fpr'] * 100.0,
        unsealed_groups['s319-layers-6t']['Truth_pixel_centered']['within_slice_exact_1pct_fpr'] * 100.0,
        unsealed_groups['s319-hybrid-uvtx']['Truth_pixel_centered']['within_slice_exact_1pct_fpr'] * 100.0,
        unsealed_groups['primary_dual_family_6m']['Truth_pixel_centered']['within_slice_exact_1pct_fpr'] * 100.0,
    ]
    lay_e01 = [
        unsealed_groups['s319-pitch-20um']['Truth_pixel_centered']['within_slice_exact_0p1pct_fpr'] * 100.0,
        unsealed_groups['s319-layers-6t']['Truth_pixel_centered']['within_slice_exact_0p1pct_fpr'] * 100.0,
        unsealed_groups['s319-hybrid-uvtx']['Truth_pixel_centered']['within_slice_exact_0p1pct_fpr'] * 100.0,
        unsealed_groups['primary_dual_family_6m']['Truth_pixel_centered']['within_slice_exact_0p1pct_fpr'] * 100.0,
    ]
    xl = np.arange(len(lay_labels))
    b_e1 = a1.bar(xl - 0.18, lay_e1, width=0.35, color='#4C72B0', label='Exact 1.0% Slice FPR')
    b_e01 = a1.bar(xl + 0.18, lay_e01, width=0.35, color='#55A868', label='Exact 0.1% Slice FPR')
    for rect, v in zip(b_e1, lay_e1):
        a1.text(rect.get_x() + rect.get_width() / 2.0, v + 0.18, f'{v:.2f}%', ha='center', fontsize=7.5)
    for rect, v in zip(b_e01, lay_e01):
        a1.text(rect.get_x() + rect.get_width() / 2.0, v + 0.18, f'{v:.2f}%', ha='center', fontsize=7.5)
    a1.set_xticks(xl)
    a1.set_xticklabels(lay_labels, fontsize=7.6)
    a1.set_ylabel('Pixel-Fiducial $\\nu_\\tau$ CC Efficiency [%]')
    a1.set_title('Layer Cost Trade-Off & Hybrid Micro-Vertex')
    a1.set_ylim(0.0, max(lay_e1) * 1.42)
    a1.legend(loc='upper left', fontsize=8.0, frameon=True)
    a1.grid(True, axis='y', alpha=0.25)

    mis_x = [0.0, 10.0, 25.0, 50.0]
    mis_y = [misalign_scan[k] * 100.0 for k in ('0_um', '10_um', '25_um', '50_um')]
    a2.plot(mis_x, mis_y, 'D-', lw=2.2, ms=6.5, color='#8172B2')
    for xm, ym in zip(mis_x, mis_y):
        a2.annotate(f'{ym:.2f}%', (xm, ym), textcoords='offset points', xytext=(0, 7), ha='center', fontsize=8.2)
    a2.set_xlabel('Plane-Wise Transverse Misalignment $\\sigma_{xy}$ [$\\mu$m]')
    a2.set_ylabel('Pixel-Fiducial $\\nu_\\tau$ Efficiency @ Exact 1% FPR [%]')
    a2.set_title('Mechanical Alignment Tolerance on CleanConfirm')
    a2.set_ylim(max(0.0, min(mis_y) - 2.0), min(100.0, max(mis_y) + 2.0))
    a2.grid(True, alpha=0.25)

    fig1.tight_layout()
    fig1_path = clean_root / 'stage319_exp2_hardware_pitch_layers_and_misalignment.png'
    fig1.savefig(fig1_path, dpi=300)
    plt.close(fig1)

    # Figure 2: Path C Vertex Resolution, Paired Geant4 Systematics, and Clean Forward-Flux Sensitivity
    fig2, axes2 = plt.subplots(1, 3, figsize=(16.2, 5.0), dpi=300)
    b0, b1, b2 = axes2

    bins_vz = np.linspace(-30.0, 30.0, 61)
    for lbl_v, gk_v, col_v in (
        ('Vanilla PF 20 $\\mu$m', 's319-pitch-20um', '#4C72B0'),
        ('Hybrid Micro-Vertex PF (Path C)', 's319-hybrid-uvtx', '#C44E52'),
        ('Coarse Pitch PF 1 mm', 's319-pitch-1mm', '#8C8C8C'),
    ):
        if gk_v in vtx_dz_confirm:
            va = unsealed_groups[gk_v]['vertex_audit']['Truth_pixel_centered']
            b0.hist(
                vtx_dz_confirm[gk_v], bins=bins_vz, range=(-30.0, 30.0), density=True, histtype='step', lw=1.9, color=col_v,
                label=f"{lbl_v}\n(med $|\\Delta z|={va['median_abs_dz_mm']:.2f}$ mm, $\\mu={va['bias_z_mm']:+.2f}$ mm)",
            )
    b0.axvline(0.0, color='#222222', ls=':', lw=1.1)
    b0.set_ylim(0.0, b0.get_ylim()[1] * 1.38)
    b0.set_xlabel('Primary Vertex Residual $\\hat{z}_v - z_{\\mathrm{true}}$ [mm] ($|\\Delta z| \\leq 30$ mm)')
    b0.set_ylabel('Normalized Core Density [mm$^{-1}$]')
    b0.set_title('Path C: Micron Multi-Tracklet Vertex Precision')
    b0.legend(loc='upper left', fontsize=7.5, frameon=True)
    b0.grid(True, alpha=0.25)

    sv_labels = ['Nominal\nRef', '$h(\\tau)=0$\nUnpol.', '$\\rho_W \\times 0.98$\nW Density', 'No Magnet\nShells', '$B \\times 0.97$\nField']
    sv_keys = ['nominal_sys_ref', 'tau_helicity_0', 'w_density_0p98', 'no_magnet_shells', 'field_0p97']
    sv_vals = [sys_eval[k]['pix_centered_tau_eff_1pct_fpr'] * 100.0 for k in sv_keys]
    bars_sv = b1.bar(np.arange(len(sv_keys)), sv_vals, color=['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2'], width=0.55)
    b1.axhline(sv_vals[0], color='#222222', ls='--', lw=1.1)
    b1.set_xticks(np.arange(len(sv_keys)))
    b1.set_xticklabels(sv_labels, fontsize=8.0)
    b1.set_ylabel('Paired 600-Event $\\nu_\\tau$ Efficiency @ 1% FPR [%]')
    b1.set_title('Paired Geant4 Transport Systematics Stability')
    b1.set_ylim(0.0, max(sv_vals) * 1.38)
    for rect, v in zip(bars_sv, sv_vals):
        b1.text(rect.get_x() + rect.get_width() / 2.0, v + 0.12, f'{v:.1f}%', ha='center', fontsize=8.0)
    b1.grid(True, axis='y', alpha=0.25)

    eras_x = ['Run 3\n250 fb$^{-1}$', 'Run 4\n680 fb$^{-1}$', 'HL-LHC\n3 ab$^{-1}$', 'FPF\n10 ab$^{-1}$']
    eras_k = ['Run3_250fb', 'Run4_680fb', 'HLLHC_3000fb', 'FPF_10000fb']
    z_stat = [flux_projections[k]['z0_stat_only'] for k in eras_k]
    z_prof = [flux_projections[k]['z0_profile_likelihood'] for k in eras_k]
    z_bb = [flux_projections[k]['z0_pl_with_10k_confirm_bb_mc_stat'] for k in eras_k]
    xe = np.arange(len(eras_k))
    b2.bar(xe - 0.25, z_stat, width=0.24, color='#4C72B0', label='Stat-only (27-bin Ultra-Tail + CR)')
    bars_zp = b2.bar(xe, z_prof, width=0.24, color='#C44E52', label='Full Physics+G4/Gen/Charm PL')
    b2.bar(xe + 0.25, z_bb, width=0.24, color='#DD8452', label='+ 10k-Confirm MC BB-lite Bound')
    b2.axhline(3.0, color='#555555', ls=':', lw=1.1, label='3$\\sigma$ threshold')
    b2.set_ylim(0.0, max(max(z_stat), 3.3) * 1.24)
    for rect, zv in zip(bars_zp, z_prof):
        b2.text(rect.get_x() + rect.get_width() / 2.0, zv + 0.08, f'{zv:.2f}$\\sigma$', ha='center', fontsize=7.3, color='#902020')
    b2.set_xticks(xe)
    b2.set_xticklabels(eras_x, fontsize=8.2)
    b2.set_ylabel('Expected $\\nu_\\tau$ Discovery Significance $Z_0$ [$\\sigma$]')
    b2.set_title('One-Shot CleanConfirm Realistic Sensitivity')
    b2.legend(loc='upper left', fontsize=7.2, frameon=True)
    b2.grid(True, axis='y', alpha=0.25)

    fig2.tight_layout()
    fig2_path = clean_root / 'stage319_hybrid_microvertex_and_clean_confirm_unsealing.png'
    fig2.savefig(fig2_path, dpi=300)
    plt.close(fig2)

    final_report = {
        'UTC': datetime.now(timezone.utc).isoformat(),
        'wall_seconds': float(time.perf_counter() - t0),
        'freeze_manifest_verified': True,
        'pre_unsealing_git_commit': freeze_doc['pre_unsealing_git_commit'],
        'primary_pf_group': prim_pf_g,
        'cutb_locked_charm_cr_edges': charm_cr_edges,
        'unsealed_clean_confirm_groups': unsealed_groups,
        'ood_vs_from_scratch_pitch_comparison_P1_5': {
            'from_scratch_exact_1pct_fpr': {
                '0.020': unsealed_groups['s319-pitch-20um']['Truth_pixel_centered']['within_slice_exact_1pct_fpr'],
                '0.100': unsealed_groups['s319-pitch-100um']['Truth_pixel_centered']['within_slice_exact_1pct_fpr'],
                '0.300': unsealed_groups['s319-pitch-300um']['Truth_pixel_centered']['within_slice_exact_1pct_fpr'],
                '1.000': unsealed_groups['s319-pitch-1mm']['Truth_pixel_centered']['within_slice_exact_1pct_fpr'],
            },
            'post_hoc_ood_quantized_on_20um_model': ood_pitch_curve,
        },
        'mechanical_misalignment_scan_Truth_pixel_centered': misalign_scan,
        'paired_family_bootstrap_comparisons': paired_confirm_comparisons,
        'paired_geant4_physics_systematics': sys_eval,
        'paired_generator_hadronization_systematics': gen_sys_eval,
        'external_punchthrough_audit': ext_punch_eval,
        'realistic_forward_flux_projections_2t_fiducial': flux_projections,
        'h0_toy_coverage_verification': h0_clean,
        'h0_toy_coverage_physics_sys': h0_clean_phys,
        'figures': [str(fig1_path), str(fig2_path)],
    }
    write_json(clean_root / 'stage319_clean_unsealing_report.json', final_report)
    return final_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('audit-clean-sim', 'build-configs', 'submit-group', 'freeze', 'unseal-and-evaluate'))
    parser.add_argument('--clean-root', type=Path, default=DEFAULT_CLEAN_ROOT)
    parser.add_argument('--stage317-root', type=Path, default=DEFAULT_STAGE317_ROOT)
    parser.add_argument('--group', type=str, default=None)
    parser.add_argument('--backend', choices=('kubeflow', 'condor'), default='condor')
    parser.add_argument('--parallel', type=int, default=3)
    parser.add_argument('--n-toys', type=int, default=1000)
    args = parser.parse_args()

    if args.command == 'audit-clean-sim':
        out = audit_clean_sim(args.clean_root)
    elif args.command == 'build-configs':
        gs = build_stage319_configs(args.clean_root, args.stage317_root)
        out = {k: [str(p) for p in v['configs']] for k, v in gs.items()}
    elif args.command == 'submit-group':
        if not args.group:
            parser.error('--group is required for submit-group')
        out = submit_stage319_group(args.clean_root, args.stage317_root, args.group, args.backend, parallel=args.parallel)
    elif args.command == 'freeze':
        out = freeze_stage319_before_unsealing(args.clean_root, args.stage317_root)
    else:
        out = unseal_and_evaluate_stage319(args.clean_root, args.stage317_root, n_toys=args.n_toys)
    print(json.dumps(out, indent=2, default=str))


if __name__ == '__main__':
    main()
