"""Evaluate Stage 317 controlled 3-seed GPU ablations (Exp-1, Exp-3, Exp-5) strictly on Tune40k.

Enforces zero Confirm200k / Cal-A / Cut-B access (`Cal_A_Cut_B_Confirm_payload_read = False`).
Applies Stage 314 strict complexity retention gates (+1.0 pp tau@1% FPR and 95% paired family-bootstrap CI > 0)
to decide which components enter the Stage 319 clean Pinpoint_G4 training & one-shot unsealing campaign.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.flavour.inventory import write_json
from analysis.flavour.v2.partitions import FAMILY, KEY
from analysis.flavour.v2.r2.readout import grouped_folds
from analysis.flavour.v3.common.calibration import _tau_efficiency_at_fpr
from analysis.flavour.v3.v2_pinpointformer.evaluate import (
    compute_slice_metrics,
    load_tune_reference_frame,
)
from analysis.flavour.v3.v4_tau.ensemble import fit_oof_domain_ensemble


SEEDS = (17, 42, 2026)
DEFAULT_STAGE317_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v4_audit_2026-11-18/stage317_ablations')
W3_BASE = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15')
W3_11_ROOT = W3_BASE / 'W3-11'
EVENT_SIDECAR = W3_BASE / 'W3-08/tune40k_offline_audit/tune_event_audit.parquet'

LEGACY_SEED_DIRS = {
    ('w3_11_pf_v3_gn2_legacy', 17): W3_11_ROOT / 'experiments/pf_v3_300k_24p_gn2_s17',
    ('w3_11_pf_v3_gn2_legacy', 42): W3_BASE / 'W3-09/experiments/pf_v3_300k_24p_gn2_flagship',
    ('w3_11_pf_v3_gn2_legacy', 2026): W3_11_ROOT / 'experiments/pf_v3_300k_24p_gn2_s2026',
    ('w3_11_b1pp_dual_legacy', 17): W3_11_ROOT / 'experiments/b1pp_300k_24p_t4_s17',
    ('w3_11_b1pp_dual_legacy', 42): W3_BASE / 'W3-05/experiments/b1pp_300k_24p_t4',
    ('w3_11_b1pp_dual_legacy', 2026): W3_11_ROOT / 'experiments/b1pp_300k_24p_t4_s2026',
}

GROUP_SPECS = {
    'w3_11_pf_v3_gn2_legacy': (W3_11_ROOT, 'pf_v3_300k_24p_gn2'),
    'w3_11_b1pp_dual_legacy': (W3_11_ROOT, 'b1pp_300k_24p_t4'),
    'exp1_pf_v2_anchored': (DEFAULT_STAGE317_ROOT, 'pf_ctrl_v2_anchored'),
    'exp1_pf_v3_event4head': (DEFAULT_STAGE317_ROOT, 'pf_ctrl_v3_event4head'),
    'exp1_pf_v3_gn2_fixed': (DEFAULT_STAGE317_ROOT, 'pf_ctrl_v3_gn2_fixed'),
    'exp3_pf_pe_only_ev4': (DEFAULT_STAGE317_ROOT, 'pf_pe_only_ev4'),
    'exp5_pf_pe_only_nomad_phys': (DEFAULT_STAGE317_ROOT, 'pf_pe_only_nomad_phys'),
    'exp3_b1pp_pe_only': (DEFAULT_STAGE317_ROOT, 'b1pp_pe_only'),
}


def _select_seed_prediction_df(exp_dir: Path) -> tuple[pd.DataFrame, dict]:
    man_path = exp_dir / 'manifest.json'
    if not man_path.exists():
        raise FileNotFoundError(f'Missing manifest: {man_path}')
    man = json.loads(man_path.read_text())
    if man.get('status') != 'complete':
        raise RuntimeError(f'Experiment not complete: {exp_dir} status={man.get("status")}')

    best_pq = exp_dir / 'best_predictions.parquet'
    ema_pq = exp_dir / 'predictions.parquet'
    # Select strictly by 4k tune_screen NLL recorded in manifest
    best_nll = man.get('best_screen_nll', man.get('best_ema_screen_NLL', float('inf')))
    hist = man.get('history') or man.get('screen_history') or []
    last_ema_nll = float('inf')
    if hist:
        last = hist[-1]
        last_ema_nll = float(last.get('ema_nll', last.get('tune_screen_ema_NLL', float('inf'))))

    chosen_pq = best_pq if (best_pq.exists() and best_nll <= last_ema_nll) else ema_pq
    df = pd.read_parquet(chosen_pq)
    info = {
        'chosen_file': chosen_pq.name,
        'best_pass': man.get('best_pass'),
        'best_screen_nll': float(best_nll),
        'last_ema_screen_nll': float(last_ema_nll),
        'parameters': man.get('parameters'),
        'wall_seconds': man.get('wall_seconds'),
        'effective_attention_sharpness': man.get('effective_attention_sharpness', {}),
    }
    return df, info


def _align_probs(df: pd.DataFrame, ref_df: pd.DataFrame) -> np.ndarray:
    pos = pd.MultiIndex.from_frame(df[KEY]).get_indexer(pd.MultiIndex.from_frame(ref_df[KEY]))
    if (pos < 0).any():
        raise ValueError('Prediction Parquet missing Tune40k reference keys')
    al = df.iloc[pos].reset_index(drop=True)
    cols = ['p_0', 'p_1', 'p_2', 'p_3'] if 'p_0' in al.columns else ['p0', 'p1', 'p2', 'p3']
    p = al[cols].to_numpy(dtype=np.float64)
    return p / np.maximum(p.sum(axis=-1, keepdims=True), 1e-12)


def _compute_vertex_audit(dfs: list[pd.DataFrame], ref_df: pd.DataFrame, sidecar_df: pd.DataFrame) -> dict | None:
    if 'v_hat_z_mm' not in dfs[0].columns:
        return None
    s_pos = pd.MultiIndex.from_frame(sidecar_df[KEY]).get_indexer(pd.MultiIndex.from_frame(ref_df[KEY]))
    if (s_pos < 0).any():
        raise ValueError('Sidecar missing Tune40k keys')
    sc = sidecar_df.iloc[s_pos].reset_index(drop=True)
    valid = sc['vertex_valid'].to_numpy(dtype=int) > 0
    pix_c = ref_df['pixel_centered'].to_numpy(dtype=bool) & valid

    vx_true = sc['vertex_x_mm'].to_numpy(dtype=np.float64)
    vy_true = sc['vertex_y_mm'].to_numpy(dtype=np.float64)
    vz_true = sc['vertex_z_mm'].to_numpy(dtype=np.float64)

    vx_hats, vy_hats, vz_hats = [], [], []
    for df in dfs:
        pos = pd.MultiIndex.from_frame(df[KEY]).get_indexer(pd.MultiIndex.from_frame(ref_df[KEY]))
        al = df.iloc[pos].reset_index(drop=True)
        vx_hats.append(al['v_hat_x_mm'].to_numpy(dtype=np.float64))
        vy_hats.append(al['v_hat_y_mm'].to_numpy(dtype=np.float64))
        vz_hats.append(al['v_hat_z_mm'].to_numpy(dtype=np.float64))

    vx_mean = np.mean(vx_hats, axis=0)
    vy_mean = np.mean(vy_hats, axis=0)
    vz_mean = np.mean(vz_hats, axis=0)

    dx = vx_mean - vx_true
    dy = vy_mean - vy_true
    dz = vz_mean - vz_true

    def _stats(m: np.ndarray) -> dict:
        r3 = np.sqrt(dx[m] ** 2 + dy[m] ** 2 + dz[m] ** 2)
        return {
            'count': int(m.sum()),
            'bias_x_mm': float(np.mean(dx[m])),
            'bias_y_mm': float(np.mean(dy[m])),
            'bias_z_mm': float(np.mean(dz[m])),
            'median_abs_dx_mm': float(np.median(np.abs(dx[m]))),
            'median_abs_dy_mm': float(np.median(np.abs(dy[m]))),
            'median_abs_dz_mm': float(np.median(np.abs(dz[m]))),
            'median_3d_mm': float(np.median(r3)),
            'p68_abs_dz_mm': float(np.quantile(np.abs(dz[m]), 0.68)),
        }

    return {
        'All_active': _stats(valid),
        'Truth_pixel_centered': _stats(pix_c),
        '_dz_pix_centered': dz[pix_c],
    }


def paired_family_bootstrap_delta(
    score_a: np.ndarray,
    score_b: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    labels: np.ndarray,
    family_codes: np.ndarray,
    mask: np.ndarray,
    n_boot: int = 1000,
    seed: int = 20261118,
) -> dict:
    """Paired 1,000-replicate family block-bootstrap delta (A minus B) for tau@1%FPR, tau@0.1%FPR, BA4."""
    m = np.asarray(mask, dtype=bool)
    sa = score_a[m]
    sb = score_b[m]
    pa = probs_a[m].argmax(axis=1)
    pb = probs_b[m].argmax(axis=1)
    y = labels[m]
    fams = family_codes[m]

    uniq_f, inv = np.unique(fams, return_inverse=True)
    order = np.argsort(inv, kind='mergesort')
    _, sp = np.unique(inv[order], return_index=True)
    groups = [order[s:e] for s, e in zip(sp, np.r_[sp[1:], len(order)])]
    n_fam = len(groups)

    def _eval_pair(idx: np.ndarray) -> tuple[float, float, float]:
        yy = y[idx]
        e1_a = float(_tau_efficiency_at_fpr(sa[idx], yy, 0.01)['tau_eff'])
        e1_b = float(_tau_efficiency_at_fpr(sb[idx], yy, 0.01)['tau_eff'])
        e01_a = float(_tau_efficiency_at_fpr(sa[idx], yy, 0.001)['tau_eff'])
        e01_b = float(_tau_efficiency_at_fpr(sb[idx], yy, 0.001)['tau_eff'])
        ba_a = float(np.mean([np.mean(pa[idx][yy == c] == c) for c in range(4) if (yy == c).any()]))
        ba_b = float(np.mean([np.mean(pb[idx][yy == c] == c) for c in range(4) if (yy == c).any()]))
        return e1_a - e1_b, e01_a - e01_b, ba_a - ba_b

    pt_e1, pt_e01, pt_ba = _eval_pair(np.arange(len(y)))
    rng = np.random.default_rng(seed)
    d_e1 = np.empty(n_boot, dtype=np.float64)
    d_e01 = np.empty(n_boot, dtype=np.float64)
    d_ba = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        draw = rng.integers(0, n_fam, size=n_fam)
        b_idx = np.concatenate([groups[g] for g in draw])
        d_e1[b], d_e01[b], d_ba[b] = _eval_pair(b_idx)

    return {
        'delta_tau_eff_1pct_pp': float(pt_e1 * 100.0),
        'delta_tau_eff_1pct_ci95_pp': [float(np.quantile(d_e1, 0.025) * 100.0), float(np.quantile(d_e1, 0.975) * 100.0)],
        'delta_tau_eff_0p1pct_pp': float(pt_e01 * 100.0),
        'delta_tau_eff_0p1pct_ci95_pp': [float(np.quantile(d_e01, 0.025) * 100.0), float(np.quantile(d_e01, 0.975) * 100.0)],
        'delta_BA4_pp': float(pt_ba * 100.0),
        'delta_BA4_ci95_pp': [float(np.quantile(d_ba, 0.025) * 100.0), float(np.quantile(d_ba, 0.975) * 100.0)],
    }


def render_stage317_figure(group_results: dict, vtx_dz_curves: dict, fig_path: Path) -> None:
    """Render 3-panel 300-DPI English publication figure for Stage 317."""
    fig, axes = plt.subplots(1, 3, figsize=(16.4, 5.2), dpi=300)
    ax0, ax1, ax2 = axes

    # Panel A: Exp-1 Controlled 300k x 24p Ablations (Truth_pixel_centered within-slice exact 1% FPR)
    p0_keys = [
        ('W3-11 GN2\n(legacy loss)', 'w3_11_pf_v3_gn2_legacy', '#8C8C8C'),
        ('PF V2\n(anchored)', 'exp1_pf_v2_anchored', '#4C72B0'),
        ('PF V3\n(Event4Head)', 'exp1_pf_v3_event4head', '#55A868'),
        ('PF V3\n(+GN2 dense)', 'exp1_pf_v3_gn2_fixed', '#DD8452'),
    ]
    y_pos = np.arange(len(p0_keys))
    vals_0 = [group_results[k]['oof_3seed_ensemble']['Truth_pixel_centered']['within_slice_exact_1pct_fpr'] * 100.0 for _, k, _ in p0_keys]
    stds_0 = [group_results[k]['seed_mean_std_Truth_pixel_centered']['tau_at_1pct_fpr']['std'] * 100.0 for _, k, _ in p0_keys]
    bars0 = ax0.barh(y_pos, vals_0, xerr=stds_0, color=[c for _, _, c in p0_keys], height=0.56, capsize=4, alpha=0.9)
    ax0.set_yticks(y_pos)
    ax0.set_yticklabels([lbl for lbl, _, _ in p0_keys], fontsize=8.8)
    ax0.set_xlabel('Pixel-Fiducial $\\nu_\\tau$ Efficiency @ Exact 1% FPR [%]')
    ax0.set_title('Exp-1: Exposure-Matched 300k$\\times$24p Ablation')
    max_right = max(v + s for v, s in zip(vals_0, stds_0))
    ax0.set_xlim(max(0.0, min(v - s for v, s in zip(vals_0, stds_0)) - 5.0), min(100.0, max_right + 5.8))
    for b_rect, v, s in zip(bars0, vals_0, stds_0):
        ax0.text(v + s + 0.45, b_rect.get_y() + b_rect.get_height() / 2.0, f'{v:.2f}%', va='center', fontsize=8.2)
    ax0.grid(True, axis='x', alpha=0.25)

    # Panel B: Exp-3 & Exp-5 Physical PE-Only Readout vs Dual-Channel Leakage (All_Tune40k & Pixel-Fiducial)
    p1_keys = [
        ('PF V3\nDual-Ch (leak)', 'exp1_pf_v3_event4head', 'oof_3seed_ensemble', '#937860'),
        ('PF V3\nPE-Only', 'exp3_pf_pe_only_ev4', 'oof_3seed_ensemble', '#4C72B0'),
        ('PF V4 NOMAD\nPE-Only', 'exp5_pf_pe_only_nomad_phys', 'oof_3seed_ensemble', '#8172B2'),
        ('B1++\nDual-Ch (leak)', 'w3_11_b1pp_dual_legacy', 'oof_3seed_ensemble', '#CCB974'),
        ('B1++\nPE-Only', 'exp3_b1pp_pe_only', 'oof_3seed_ensemble', '#64B5CD'),
        ('PF+B1++ (6m)\nPE-Only Ens', 'exp3_dual_family_pe_only_6m', 'oof_6model_ensemble', '#C44E52'),
    ]
    x_idx = np.arange(len(p1_keys))
    w_bar = 0.36
    v_pc = [group_results[k][ek]['Truth_pixel_centered']['within_slice_exact_1pct_fpr'] * 100.0 for _, k, ek, _ in p1_keys]
    v_all = [group_results[k][ek]['All_Tune40k']['within_slice_exact_1pct_fpr'] * 100.0 for _, k, ek, _ in p1_keys]
    ax1.bar(x_idx - w_bar / 2, v_pc, width=w_bar, color='#4C72B0', label='Pixel-Fiducial (exact 1% FPR)')
    ax1.bar(x_idx + w_bar / 2, v_all, width=w_bar, color='#C44E52', label='All Tune40k (exact 1% FPR)')
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels([lbl for lbl, _, _, _ in p1_keys], fontsize=7.6)
    ax1.set_ylabel('$\\nu_\\tau$ CC Efficiency @ Exact 1% Slice FPR [%]')
    ax1.set_title('Exp-3 & Exp-5: Physical PE-Only vs Dual-Ch Readout')
    ax1.set_ylim(0.0, min(100.0, max(v_pc) + 12.0))
    ax1.legend(loc='upper right', fontsize=8.0, frameon=True)
    ax1.grid(True, axis='y', alpha=0.25)

    # Panel C: P2-2 Longitudinal Primary Vertex Residual Distribution (hat{z}_v - z_true)
    bins_z = np.linspace(-60.0, 60.0, 121)
    for label_v, gkey_v, col_v, ls_v in (
        ('W3-11 Legacy (biased z-loss)', 'w3_11_pf_v3_gn2_legacy', '#8C8C8C', '--'),
        ('Stage 317 PF V2 (unbiased)', 'exp1_pf_v2_anchored', '#4C72B0', '-'),
        ('Stage 317 PF V3 PE-Only (fixed)', 'exp3_pf_pe_only_ev4', '#C44E52', '-'),
    ):
        if gkey_v in vtx_dz_curves:
            dz_arr = np.clip(vtx_dz_curves[gkey_v], -59.5, 59.5)
            va = group_results[gkey_v]['vertex_audit']['Truth_pixel_centered']
            ax2.hist(
                dz_arr, bins=bins_z, density=True, histtype='step', lw=1.9, color=col_v, ls=ls_v,
                label=f"{label_v}\n($\\mu={va['bias_z_mm']:+.2f}$ mm, med $|\\Delta z|={va['median_abs_dz_mm']:.2f}$ mm)",
            )
    ax2.axvline(0.0, color='#222222', ls=':', lw=1.1)
    ax2.set_ylim(0.0, ax2.get_ylim()[1] * 1.28)
    ax2.set_xlabel('Primary Vertex Longitudinal Residual $\\hat{z}_v - z_{\\mathrm{true}}$ [mm]')
    ax2.set_ylabel('Normalized Event Density [mm$^{-1}$]')
    ax2.set_title('P2-2: Unbiased Vertex Regression Calibration')
    ax2.legend(loc='upper left', fontsize=7.3, frameon=True)
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)


def evaluate_all_stage317(stage317_root: Path, n_boot: int = 1000) -> dict:
    t0 = time.perf_counter()
    ref_df = load_tune_reference_frame()
    sidecar_df = pd.read_parquet(EVENT_SIDECAR)
    labels = ref_df['label'].to_numpy(dtype=np.int64)
    obs_dom = ref_df['obs_domain_id'].to_numpy(dtype=np.int64)
    fam_hash = ref_df['family_hi'].to_numpy(dtype=np.uint64) * np.uint64(1000003) + ref_df['family_lo'].to_numpy(dtype=np.uint64)
    _, fam_codes = np.unique(fam_hash, return_inverse=True)
    folds, _ = grouped_folds(ref_df[KEY + FAMILY + ['label']].reset_index(drop=True), folds=5, seed=20261117)

    slices = {
        'All_Tune40k': np.ones(len(ref_df), dtype=bool),
        'Truth_pixel_centered': ref_df['pixel_centered'].to_numpy(dtype=bool),
        'D_pix_trk': obs_dom == 1,
        'D_cal_core': obs_dom == 2,
        'E_le_2p3TeV': ref_df['is_E_le_2p3TeV'].to_numpy(dtype=bool),
        'HighE_non_tau': ref_df['is_highE_non_tau'].to_numpy(dtype=bool),
    }

    group_results = {}
    ens_cache = {}
    vtx_dz_curves = {}
    member_frames_by_group: dict[str, dict[str, pd.DataFrame]] = {}

    for gkey, (groot, prefix) in GROUP_SPECS.items():
        m_frames = {}
        seed_infos = {}
        seed_metrics_pc = []
        seed_metrics_all = []
        for s in SEEDS:
            sname = f'{prefix}_s{s}'
            groot_eff = stage317_root if groot == DEFAULT_STAGE317_ROOT else groot
            exp_dir = LEGACY_SEED_DIRS.get((gkey, s), groot_eff / 'experiments' / sname)
            df, info = _select_seed_prediction_df(exp_dir)
            m_frames[sname] = df
            seed_infos[sname] = info
            probs = _align_probs(df, ref_df)
            seed_metrics_pc.append(compute_slice_metrics(labels, probs, slices['Truth_pixel_centered']))
            seed_metrics_all.append(compute_slice_metrics(labels, probs, slices['All_Tune40k']))

        member_frames_by_group[gkey] = m_frames
        ens = fit_oof_domain_ensemble(m_frames, ref_df, slices, n_folds=5, folds=folds)
        ens_cache[gkey] = ens

        vtx_aud = _compute_vertex_audit(list(m_frames.values()), ref_df, sidecar_df)
        if vtx_aud is not None:
            vtx_dz_curves[gkey] = vtx_aud.pop('_dz_pix_centered')

        def _seed_summary(m_list: list[dict], k: str) -> dict:
            vals = np.array([m[k] for m in m_list], dtype=np.float64)
            return {'mean': float(vals.mean()), 'std': float(vals.std(ddof=1))}

        ens_slices = {}
        for sl_name, sl_mask in slices.items():
            sm = compute_slice_metrics(labels, ens['oof_probs'], sl_mask)
            if (labels[sl_mask] == 2).any() and (labels[sl_mask] != 2).any():
                sm['within_slice_exact_1pct_fpr'] = float(_tau_efficiency_at_fpr(ens['s_tau'][sl_mask], labels[sl_mask], 0.01)['tau_eff'])
                sm['within_slice_exact_0p1pct_fpr'] = float(_tau_efficiency_at_fpr(ens['s_tau'][sl_mask], labels[sl_mask], 0.001)['tau_eff'])
                sm['global_budget_np_1pct_fpr'] = ens['slices'][sl_name]['np_1pct']
            ens_slices[sl_name] = sm

        group_results[gkey] = {
            'seeds': seed_infos,
            'seed_mean_std_Truth_pixel_centered': {
                k: _seed_summary(seed_metrics_pc, k) for k in ('BA4', 'cond_BA3', 'nll', 'tau_at_1pct_fpr', 'tau_at_0p1pct_fpr')
            },
            'seed_mean_std_All_Tune40k': {
                k: _seed_summary(seed_metrics_all, k) for k in ('BA4', 'cond_BA3', 'nll', 'tau_at_1pct_fpr', 'tau_at_0p1pct_fpr')
            },
            'oof_3seed_ensemble': ens_slices,
            'oof_asimov_z0_18bin': ens['asimov_z0_oof_18bin_multidomain'],
            'vertex_audit': vtx_aud,
        }

    # Also build the 6-model physical PE-only dual-family ensemble (PF_PE_only + B1++_PE_only)
    dual_pe_frames = {**member_frames_by_group['exp3_pf_pe_only_ev4'], **member_frames_by_group['exp3_b1pp_pe_only']}
    dual_pe_ens = fit_oof_domain_ensemble(dual_pe_frames, ref_df, slices, n_folds=5, folds=folds)
    ens_cache['exp3_dual_family_pe_only_6m'] = dual_pe_ens
    dual_pe_slices = {}
    for sl_name, sl_mask in slices.items():
        sm = compute_slice_metrics(labels, dual_pe_ens['oof_probs'], sl_mask)
        if (labels[sl_mask] == 2).any() and (labels[sl_mask] != 2).any():
            sm['within_slice_exact_1pct_fpr'] = float(_tau_efficiency_at_fpr(dual_pe_ens['s_tau'][sl_mask], labels[sl_mask], 0.01)['tau_eff'])
            sm['within_slice_exact_0p1pct_fpr'] = float(_tau_efficiency_at_fpr(dual_pe_ens['s_tau'][sl_mask], labels[sl_mask], 0.001)['tau_eff'])
            sm['global_budget_np_1pct_fpr'] = dual_pe_ens['slices'][sl_name]['np_1pct']
        dual_pe_slices[sl_name] = sm
    group_results['exp3_dual_family_pe_only_6m'] = {
        'oof_6model_ensemble': dual_pe_slices,
        'oof_asimov_z0_18bin': dual_pe_ens['asimov_z0_oof_18bin_multidomain'],
    }

    def _pair_cmp(a_key: str, b_key: str) -> dict:
        ea, eb = ens_cache[a_key], ens_cache[b_key]
        return {
            sl: paired_family_bootstrap_delta(
                ea['s_tau'], eb['s_tau'], ea['oof_probs'], eb['oof_probs'],
                labels, fam_codes, slices[sl], n_boot=n_boot,
            )
            for sl in ('Truth_pixel_centered', 'D_pix_trk', 'All_Tune40k', 'E_le_2p3TeV')
        }

    comparisons = {
        'Exp1_V3_Event4Head_vs_V2_Anchored_300kx24p': _pair_cmp('exp1_pf_v3_event4head', 'exp1_pf_v2_anchored'),
        'Exp1_V3_GN2_Fixed_vs_V3_Event4Head_300kx24p': _pair_cmp('exp1_pf_v3_gn2_fixed', 'exp1_pf_v3_event4head'),
        'Exp1_V3_GN2_Fixed_vs_W3_11_Legacy_GN2': _pair_cmp('exp1_pf_v3_gn2_fixed', 'w3_11_pf_v3_gn2_legacy'),
        'Exp3_PF_PE_Only_vs_PF_DualChannel': _pair_cmp('exp3_pf_pe_only_ev4', 'exp1_pf_v3_event4head'),
        'Exp3_B1PP_PE_Only_vs_B1PP_DualChannel': _pair_cmp('exp3_b1pp_pe_only', 'w3_11_b1pp_dual_legacy'),
        'Exp3_DualFamily_PE_Only_vs_PF_PE_Only': _pair_cmp('exp3_dual_family_pe_only_6m', 'exp3_pf_pe_only_ev4'),
        'Exp5_PF_PE_NOMAD_Phys_vs_PF_PE_Only': _pair_cmp('exp5_pf_pe_only_nomad_phys', 'exp3_pf_pe_only_ev4'),
    }

    # Strict Stage 314 complexity retention decisions on Truth_pixel_centered
    gn2_cmp = comparisons['Exp1_V3_GN2_Fixed_vs_V3_Event4Head_300kx24p']['Truth_pixel_centered']
    ev4_cmp = comparisons['Exp1_V3_Event4Head_vs_V2_Anchored_300kx24p']['Truth_pixel_centered']
    nom_cmp = comparisons['Exp5_PF_PE_NOMAD_Phys_vs_PF_PE_Only']['Truth_pixel_centered']

    retention_decisions = {
        'rule': 'Stage 314 strict gate: retain heavy auxiliary module iff delta_tau_eff_1pct_pp >= +1.0 pp AND 95% paired CI > 0 on Truth_pixel_centered',
        'retain_event4head_multitask': bool(
            ev4_cmp['delta_tau_eff_1pct_pp'] >= 1.0 and ev4_cmp['delta_tau_eff_1pct_ci95_pp'][0] > 0.0
        ),
        'retain_dense_gn2_heads': bool(
            gn2_cmp['delta_tau_eff_1pct_pp'] >= 1.0 and gn2_cmp['delta_tau_eff_1pct_ci95_pp'][0] > 0.0
        ),
        'retain_calibrated_nomad_head': bool(
            nom_cmp['delta_tau_eff_1pct_pp'] >= 1.0 and nom_cmp['delta_tau_eff_1pct_ci95_pp'][0] > 0.0
        ),
        'measured_deltas_Truth_pixel_centered': {
            'V3_Event4Head_minus_V2_Anchored_pp': ev4_cmp,
            'V3_GN2_Fixed_minus_V3_Event4Head_pp': gn2_cmp,
            'V4_NOMAD_Phys_minus_V3_Event4Head_PE_Only_pp': nom_cmp,
        },
    }

    fig_path = stage317_root / 'stage317_controlled_ablations_and_vertex_calibration.png'
    render_stage317_figure(group_results, vtx_dz_curves, fig_path)

    out_report = {
        'UTC': datetime.now(timezone.utc).isoformat(),
        'wall_seconds': float(time.perf_counter() - t0),
        'Cal_A_Cut_B_Confirm_payload_read': False,
        'n_bootstrap_replicates': n_boot,
        'groups': group_results,
        'paired_family_bootstrap_comparisons': comparisons,
        'stage314_complexity_retention_decisions': retention_decisions,
        'figure_path': str(fig_path),
    }
    write_json(stage317_root / 'stage317_ablation_evaluation_report.json', out_report)
    return out_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_STAGE317_ROOT)
    parser.add_argument('--n-boot', type=int, default=1000)
    args = parser.parse_args()
    rep = evaluate_all_stage317(args.root, n_boot=args.n_boot)
    print(json.dumps(rep['stage314_complexity_retention_decisions'], indent=2))


if __name__ == '__main__':
    main()
