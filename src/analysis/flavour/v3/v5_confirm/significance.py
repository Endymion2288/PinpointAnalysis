"""Cal-A calibration, Cut-B threshold/bin locking, Confirm200k evaluation, and Profile-Likelihood significance."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax

from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.v2.partitions import FAMILY, KEY
from analysis.flavour.v2.r1.metrics import Metrics, three_from_four, three_targets
from analysis.flavour.v3.common.calibration import _tau_efficiency_at_fpr
from analysis.flavour.v3.common.domains import OBS_DOMAIN_NAMES
from analysis.flavour.v3.v4_tau.ensemble import _fit_domain_convex_blend, extract_member_arrays
from analysis.flavour.v3.v5_confirm import DEFAULT_V3_ROOT, ROLE_ORDER, SEEDS
from analysis.flavour.v3.v5_confirm.flux_significance import (
    LHC_FORWARD_SCENARIOS,
    assign_signal_and_charm_cr_bins,
    build_physical_templates,
    extract_signed_systematic_templates,
    fit_realistic_profile_likelihood,
)


TEMPLATE_QUANTILES = (0.0, 0.50, 0.80, 0.92, 0.97, 0.99, 1.0)
UNIFORM_QUANTILES = (0.0, 1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0, 4.0 / 6.0, 5.0 / 6.0, 1.0)
NP_RATIO_GRID = (
    (1.0, 1.0, 1.0),
    (1.5, 1.0, 0.6),
    (2.0, 1.0, 0.5),
    (3.0, 1.0, 0.3),
    (4.0, 0.8, 0.2),
)


def _balanced_weights(labels: np.ndarray) -> np.ndarray:
    cnt = np.bincount(labels, minlength=4)
    return (len(labels) / (4.0 * np.maximum(cnt, 1)))[labels]


def _apply_cal_a_probs(log_p_stack: np.ndarray, obs_dom: np.ndarray, domain_params: dict[str, dict]) -> np.ndarray:
    """Apply frozen Cal-A domain weights/temperature/bias; D_empty (0) stays strictly uniform."""
    n = log_p_stack.shape[1]
    probs = np.full((n, 4), 0.25, dtype=np.float64)
    for d in (1, 2, 3):
        dm = obs_dom == d
        if not dm.any():
            continue
        p = domain_params[OBS_DOMAIN_NAMES[d]]
        w = np.asarray(p['weights'], dtype=np.float64)
        mix_lp = np.tensordot(w, log_p_stack[:, dm, :], axes=(0, 0))
        scaled = float(p['inv_T']) * mix_lp + np.asarray(p['bias'], dtype=np.float64)[None, :]
        probs[dm] = softmax(scaled, axis=1)
    return probs


def _compute_stau(
    probs: np.ndarray,
    charm_prob: np.ndarray | None,
    obs_dom: np.ndarray,
    lambda_by_domain: dict[str, float],
) -> np.ndarray:
    s_tau = np.full(len(probs), -25.0, dtype=np.float64)
    for d in (1, 2, 3):
        dm = obs_dom == d
        if not dm.any():
            continue
        lam = float(lambda_by_domain.get(OBS_DOMAIN_NAMES[d], 0.0))
        p_t = np.clip(probs[dm, 2], 1e-12, 1.0 - 1e-12)
        p_b = np.clip(1.0 - p_t, 1e-12, 1.0)
        ch = charm_prob[dm] if charm_prob is not None else 0.0
        s_tau[dm] = np.log(p_t / (p_b + lam * ch + 1e-12))
    return s_tau


def fit_cal_a_calibrators(
    member_frames_cal_a: dict[str, pd.DataFrame],
    ref_cal_a: pd.DataFrame,
    member_names: list[str] | None = None,
) -> dict:
    """Step 1 (Cal-A40k): Fit & freeze per-domain convex weights, temperature, and bias."""
    names = member_names if member_names is not None else list(member_frames_cal_a.keys())
    aligned = [extract_member_arrays(member_frames_cal_a[n], ref_cal_a) for n in names]
    log_p_stack = np.stack([a['log_p'] for a in aligned], axis=0)
    labels = ref_cal_a['label'].to_numpy(dtype=np.int64)
    obs_dom = ref_cal_a['obs_domain_id'].to_numpy(dtype=np.int64)

    domain_params: dict[str, dict] = {
        OBS_DOMAIN_NAMES[0]: {
            'weights': [1.0 / len(names)] * len(names),
            'inv_T': 0.0,
            'T': None,
            'bias': [0.0, 0.0, 0.0, 0.0],
            'fixed_uniform': True,
        }
    }
    for d in (1, 2, 3):
        dm = obs_dom == d
        domain_params[OBS_DOMAIN_NAMES[d]] = _fit_domain_convex_blend(log_p_stack[:, dm, :], labels[dm])

    cal_probs = _apply_cal_a_probs(log_p_stack, obs_dom, domain_params)
    all_nll = float(-np.log(np.clip(cal_probs[np.arange(len(labels)), labels], 1e-8, 1.0)).mean())
    all_ba4 = float(Metrics(labels, cal_probs).report()['BA'])
    return {
        'fit_split': 'Cal-A40k',
        'events': int(len(labels)),
        'member_names': list(names),
        'domains': domain_params,
        'cal_a_post_fit_diagnostics': {
            'All_nll': all_nll,
            'All_BA4': all_ba4,
            'D_empty_count': int((obs_dom == 0).sum()),
        },
    }


def _fit_cut_b_np_cuts(
    s_tau: np.ndarray,
    labels: np.ndarray,
    obs_dom: np.ndarray,
    w_all: np.ndarray,
    target_fpr: float,
) -> dict:
    """Lock Strategy-II per-domain Neyman-Pearson thresholds solely on Cut-B160k."""
    tot_bkg = float(w_all[labels != 2].sum())
    best_eff = -1.0
    best_cuts: dict[str, float] = {OBS_DOMAIN_NAMES[d]: float('inf') for d in (1, 2, 3)}
    best_ratios: tuple[float, float, float] | None = None
    for r1, r2, r3 in NP_RATIO_GRID:
        r_vec = {1: r1, 2: r2, 3: r3}
        denom = sum(r_vec[d] * float(w_all[(obs_dom == d) & (labels != 2)].sum()) for d in (1, 2, 3))
        scale = (target_fpr * tot_bkg) / max(denom, 1e-30)
        cuts: dict[str, float] = {}
        tau_sel = 0.0
        for d in (1, 2, 3):
            dm = obs_dom == d
            dn = OBS_DOMAIN_NAMES[d]
            if not dm.any() or not (dm & (labels != 2)).any():
                cuts[dn] = float('inf')
                continue
            f_d = min(0.5, scale * r_vec[d])
            info_d = _tau_efficiency_at_fpr(s_tau[dm], labels[dm], f_d, w_all[dm])
            thr = float(info_d['threshold']) if info_d['threshold'] is not None else float('inf')
            cuts[dn] = thr
            tau_sel += float(w_all[dm & (labels == 2) & (s_tau >= thr)].sum())
        if tau_sel > best_eff:
            best_eff = tau_sel
            best_cuts = cuts
            best_ratios = (r1, r2, r3)
    return {
        'target_fpr': float(target_fpr),
        'chosen_ratios': list(best_ratios) if best_ratios else None,
        'thresholds_by_domain': best_cuts,
        'cut_b_weighted_tau_eff': float(best_eff / max(float(w_all[labels == 2].sum()), 1e-30)),
    }


def _fit_bin_edges(s_tau: np.ndarray, obs_dom: np.ndarray, quantiles: tuple[float, ...]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for d in (1, 2, 3):
        dn = OBS_DOMAIN_NAMES[d]
        dm = obs_dom == d
        vals = s_tau[dm]
        if len(vals) < 10:
            edges = np.linspace(-5.0, 5.0, len(quantiles), dtype=np.float64)
        else:
            edges = np.quantile(vals, quantiles).astype(np.float64)
            # Ensure strictly non-decreasing finite interior knots
            for i in range(1, len(edges) - 1):
                if edges[i] <= edges[i - 1]:
                    edges[i] = edges[i - 1] + 1e-6
        edges[0] = -1e9
        edges[-1] = 1e9
        out[dn] = edges.tolist()
    return out


def lock_cut_b_operating_points(
    member_frames_cut_b: dict[str, pd.DataFrame],
    ref_cut_b: pd.DataFrame,
    cal_a_params: dict,
    lambda_grid: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0),
) -> dict:
    """Step 2 (Cut-B160k): Freeze lambda_charm, Strategy I/II cuts, and 18-bin template edges."""
    names = list(cal_a_params['member_names'])
    aligned = [extract_member_arrays(member_frames_cut_b[n], ref_cut_b) for n in names]
    log_p_stack = np.stack([a['log_p'] for a in aligned], axis=0)
    charms = [a['charm'] for a in aligned if a['charm'] is not None]
    mean_charm = np.mean(charms, axis=0) if charms else None

    labels = ref_cut_b['label'].to_numpy(dtype=np.int64)
    obs_dom = ref_cut_b['obs_domain_id'].to_numpy(dtype=np.int64)
    probs = _apply_cal_a_probs(log_p_stack, obs_dom, cal_a_params['domains'])

    lambda_by_domain: dict[str, float] = {OBS_DOMAIN_NAMES[0]: 0.0}
    for d in (1, 2, 3):
        dn = OBS_DOMAIN_NAMES[d]
        dm = obs_dom == d
        best_lam = 0.0
        if mean_charm is not None and (dm & (labels == 2)).any() and (dm & (labels != 2)).any():
            best_obj = -1.0
            p_t = np.clip(probs[dm, 2], 1e-12, 1.0 - 1e-12)
            p_b = np.clip(1.0 - p_t, 1e-12, 1.0)
            ch = mean_charm[dm]
            wd = _balanced_weights(labels[dm])
            for lam in lambda_grid:
                sf = np.log(p_t / (p_b + float(lam) * ch + 1e-12))
                e1 = _tau_efficiency_at_fpr(sf, labels[dm], 0.01, wd)['tau_eff'] or 0.0
                e01 = _tau_efficiency_at_fpr(sf, labels[dm], 0.001, wd)['tau_eff'] or 0.0
                obj = e1 + 0.5 * e01
                if obj > best_obj + 1e-9:
                    best_obj = obj
                    best_lam = float(lam)
        lambda_by_domain[dn] = best_lam

    s_tau = _compute_stau(probs, mean_charm, obs_dom, lambda_by_domain)
    w_all = _balanced_weights(labels)

    cut_1p = float(_tau_efficiency_at_fpr(s_tau, labels, 0.01, w_all)['threshold'])
    cut_01p = float(_tau_efficiency_at_fpr(s_tau, labels, 0.001, w_all)['threshold'])
    np_1p = _fit_cut_b_np_cuts(s_tau, labels, obs_dom, w_all, 0.01)
    np_01p = _fit_cut_b_np_cuts(s_tau, labels, obs_dom, w_all, 0.001)

    edges_tail = _fit_bin_edges(s_tau, obs_dom, TEMPLATE_QUANTILES)
    edges_unif = _fit_bin_edges(s_tau, obs_dom, UNIFORM_QUANTILES)

    return {
        'lock_split': 'Cut-B160k',
        'events': int(len(labels)),
        'lambda_charm_by_domain': lambda_by_domain,
        'strategy_i_global_cuts': {
            '1pct_fpr': cut_1p,
            '0p1pct_fpr': cut_01p,
        },
        'strategy_ii_np_cuts': {
            '1pct_fpr': np_1p,
            '0p1pct_fpr': np_01p,
        },
        'template_edges_tail_resolved': {
            'quantiles': list(TEMPLATE_QUANTILES),
            'n_bins_per_domain': len(TEMPLATE_QUANTILES) - 1,
            'edges_by_domain': edges_tail,
        },
        'template_edges_uniform': {
            'quantiles': list(UNIFORM_QUANTILES),
            'n_bins_per_domain': len(UNIFORM_QUANTILES) - 1,
            'edges_by_domain': edges_unif,
        },
        'cut_b_diagnostics': {
            'All_nll': float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-8, 1.0)).mean()),
            'All_BA4': float(Metrics(labels, probs).report()['BA']),
            'cut_b_global_tau_1pct': float((s_tau[labels == 2] >= cut_1p).mean()),
            'cut_b_global_fpr_1pct': float((s_tau[labels != 2] >= cut_1p).mean()),
            'cut_b_np_tau_1pct': float(_apply_np_mask(s_tau, obs_dom, np_1p['thresholds_by_domain'])[labels == 2].mean()),
            'cut_b_np_fpr_1pct': float(_apply_np_mask(s_tau, obs_dom, np_1p['thresholds_by_domain'])[labels != 2].mean()),
        },
    }


def _apply_np_mask(s_tau: np.ndarray, obs_dom: np.ndarray, thresholds_by_domain: dict[str, float]) -> np.ndarray:
    sel = np.zeros(len(s_tau), dtype=bool)
    for d in (1, 2, 3):
        dm = obs_dom == d
        thr = float(thresholds_by_domain[OBS_DOMAIN_NAMES[d]])
        sel[dm] = s_tau[dm] >= thr
    return sel


def _assign_18_bins(s_tau: np.ndarray, obs_dom: np.ndarray, edges_by_domain: dict[str, list[float]]) -> np.ndarray:
    n_bins = 0
    for d in (1, 2, 3):
        n_bins = len(edges_by_domain[OBS_DOMAIN_NAMES[d]]) - 1
        break
    bin_ids = np.full(len(s_tau), -1, dtype=np.int64)
    for d in (1, 2, 3):
        dm = obs_dom == d
        if not dm.any():
            continue
        edges = np.asarray(edges_by_domain[OBS_DOMAIN_NAMES[d]], dtype=np.float64)
        idx = np.clip(np.searchsorted(edges[1:-1], s_tau[dm], side='right'), 0, n_bins - 1)
        bin_ids[dm] = (d - 1) * n_bins + idx
    return bin_ids


def apply_frozen_pipeline(
    member_frames: dict[str, pd.DataFrame],
    ref_df: pd.DataFrame,
    cal_a_params: dict,
    cut_b_locks: dict,
) -> dict[str, np.ndarray]:
    """Zero-refit forward application of frozen Cal-A calibrators and frozen Cut-B cuts."""
    names = list(cal_a_params['member_names'])
    aligned = [extract_member_arrays(member_frames[n], ref_df) for n in names]
    log_p_stack = np.stack([a['log_p'] for a in aligned], axis=0)
    charms = [a['charm'] for a in aligned if a['charm'] is not None]
    mean_charm = np.mean(charms, axis=0) if charms else None
    obs_dom = ref_df['obs_domain_id'].to_numpy(dtype=np.int64)

    probs = _apply_cal_a_probs(log_p_stack, obs_dom, cal_a_params['domains'])
    s_tau = _compute_stau(probs, mean_charm, obs_dom, cut_b_locks['lambda_charm_by_domain'])
    g_cuts = cut_b_locks['strategy_i_global_cuts']
    np_cuts = cut_b_locks['strategy_ii_np_cuts']

    pass_g1 = (obs_dom != 0) & (s_tau >= float(g_cuts['1pct_fpr']))
    pass_g01 = (obs_dom != 0) & (s_tau >= float(g_cuts['0p1pct_fpr']))
    pass_np1 = _apply_np_mask(s_tau, obs_dom, np_cuts['1pct_fpr']['thresholds_by_domain'])
    pass_np01 = _apply_np_mask(s_tau, obs_dom, np_cuts['0p1pct_fpr']['thresholds_by_domain'])

    bin_tail = _assign_18_bins(s_tau, obs_dom, cut_b_locks['template_edges_tail_resolved']['edges_by_domain'])
    bin_unif = _assign_18_bins(s_tau, obs_dom, cut_b_locks['template_edges_uniform']['edges_by_domain'])
    return {
        'probs': probs,
        'charm_prob': mean_charm,
        's_tau': s_tau,
        'pass_global_1pct': pass_g1,
        'pass_global_0p1pct': pass_g01,
        'pass_np_1pct': pass_np1,
        'pass_np_0p1pct': pass_np01,
        'bin_ids_tail': bin_tail,
        'bin_ids_unif': bin_unif,
    }


def compute_locked_slice_report(
    labels: np.ndarray,
    probs: np.ndarray,
    s_tau: np.ndarray,
    applied: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict:
    """Compute continuous + Cut-B-locked operating-point metrics on one Confirm slice."""
    m = np.asarray(mask, dtype=bool)
    n = int(m.sum())
    if n == 0:
        return {'count': 0}
    ym = labels[m]
    pm = probs[m]
    pm = pm / np.maximum(pm.sum(axis=-1, keepdims=True), 1e-12)
    sm = s_tau[m]

    nll = float(-np.log(np.clip(pm[np.arange(n), ym], 1e-8, 1.0)).mean())
    present = np.bincount(ym, minlength=4) > 0
    rep = Metrics(ym, pm).report()
    pc = rep['per_class']
    ba4 = float(rep['BA']) if present.all() else float('nan')

    nt = ym != 2
    if nt.any() and len(np.unique(ym[nt])) == 3:
        y3 = three_targets(ym[nt])
        p3 = three_from_four(pm[nt])
        ba3 = float(Metrics(y3, p3).report()['BA'])
    else:
        ba3 = float('nan')

    has_tau = bool((ym == 2).any())
    has_bkg = bool(nt.any())
    out = {
        'count': n,
        'nll': nll,
        'BA4': ba4,
        'cond_BA3': ba3,
        'recall_e': float(pc[0]['recall']) if pc[0]['recall'] is not None else float('nan'),
        'recall_mu': float(pc[1]['recall']) if pc[1]['recall'] is not None else float('nan'),
        'recall_tau': float(pc[2]['recall']) if pc[2]['recall'] is not None else float('nan'),
        'recall_NC': float(pc[3]['recall']) if pc[3]['recall'] is not None else float('nan'),
        'auroc_tau': float(pc[2]['AUROC']) if pc[2]['AUROC'] is not None else float('nan'),
    }
    if has_tau and has_bkg:
        ex1 = _tau_efficiency_at_fpr(sm, ym, 0.01)
        ex01 = _tau_efficiency_at_fpr(sm, ym, 0.001)
        out['within_slice_S_tau_1pct'] = float(ex1['tau_eff'])
        out['within_slice_S_tau_0p1pct'] = float(ex01['tau_eff'])
        out['within_slice_exact_1pct_fpr'] = ex1
        out['within_slice_exact_0p1pct_fpr'] = ex01

    # Realized performance at the locked Cut-B thresholds (explicitly distinguish global-budget
    # variable-local-FPR Strategy-II from within-slice fixed-FPR benchmarks per Audit P1-4)
    for key in ('pass_global_1pct', 'pass_global_0p1pct', 'pass_np_1pct', 'pass_np_0p1pct'):
        sel = applied[key][m]
        tau_tot = int((ym == 2).sum())
        bkg_tot = int(nt.sum())
        tau_pass = int((sel & (ym == 2)).sum())
        bkg_pass = int((sel & nt).sum())
        op_type = (
            'global_budget_variable_local_fpr'
            if '_np_' in key
            else 'global_uniform_score_threshold'
        )
        out[f'locked_{key}'] = {
            'operating_point_type': op_type,
            'tau_eff': float(tau_pass / tau_tot) if tau_tot > 0 else None,
            'actual_fpr': float(bkg_pass / bkg_tot) if bkg_tot > 0 else None,
            'tau_passed': tau_pass,
            'tau_total': tau_tot,
            'bkg_passed': bkg_pass,
            'bkg_total': bkg_tot,
        }
    return out


def vectorized_family_block_bootstrap(
    labels: np.ndarray,
    probs: np.ndarray,
    s_tau: np.ndarray,
    applied: dict[str, np.ndarray],
    fam_codes: np.ndarray,
    mask: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Fast vectorized n_boot=1000 physical-family block-bootstrap 95% CIs over large splits."""
    m = np.asarray(mask, dtype=bool)
    ym = labels[m]
    pm = probs[m]
    pm = pm / np.maximum(pm.sum(axis=-1, keepdims=True), 1e-12)
    sm = s_tau[m]
    g1 = applied['pass_global_1pct'][m].astype(np.float64)
    np1 = applied['pass_np_1pct'][m].astype(np.float64)
    np01 = applied['pass_np_0p1pct'][m].astype(np.float64)

    _, fam_idx = np.unique(fam_codes[m], return_inverse=True)
    n_fam = int(fam_idx.max()) + 1 if len(fam_idx) else 0
    if n_fam < 10:
        return {'n_families': n_fam, 'n_boot': 0}

    pred4 = pm.argmax(axis=1)
    pred3_nt = three_from_four(pm).argmax(axis=1)
    true_nll = -np.log(np.clip(pm[np.arange(len(ym)), ym], 1e-8, 1.0))

    # Per-family sufficient statistics: shape (n_fam, 21)
    # 0..3: class counts y in {0,1,2,3}
    # 4..7: correct class counts (pred4 == y)
    # 8..10: cond_BA3 correct counts for y in {0,1,3} (non-tau)
    # 11: nll sum, 12: g1_tau, 13: g1_bkg, 14: np1_tau, 15: np1_bkg, 16: np01_tau, 17: np01_bkg
    fam_stats = np.zeros((n_fam, 18), dtype=np.float64)
    for c in range(4):
        cm = ym == c
        if cm.any():
            np.add.at(fam_stats[:, c], fam_idx[cm], 1.0)
            ok4 = cm & (pred4 == c)
            if ok4.any():
                np.add.at(fam_stats[:, 4 + c], fam_idx[ok4], 1.0)
    for k, c in enumerate((0, 1, 3)):
        ok3 = (ym == c) & (pred3_nt == three_targets(np.array([c]))[0])
        if ok3.any():
            np.add.at(fam_stats[:, 8 + k], fam_idx[ok3], 1.0)
    np.add.at(fam_stats[:, 11], fam_idx, true_nll)
    tau_m = ym == 2
    bkg_m = ym != 2
    if tau_m.any():
        np.add.at(fam_stats[:, 12], fam_idx[tau_m], g1[tau_m])
        np.add.at(fam_stats[:, 14], fam_idx[tau_m], np1[tau_m])
        np.add.at(fam_stats[:, 16], fam_idx[tau_m], np01[tau_m])
    if bkg_m.any():
        np.add.at(fam_stats[:, 13], fam_idx[bkg_m], g1[bkg_m])
        np.add.at(fam_stats[:, 15], fam_idx[bkg_m], np1[bkg_m])
        np.add.at(fam_stats[:, 17], fam_idx[bkg_m], np01[bkg_m])

    # Tail-refined adaptive score histogram + pure-signal overflow bin above max background score
    q_probs = np.concatenate([np.linspace(0.0, 0.98, 256, endpoint=False), np.linspace(0.98, 1.0, 1025)])
    have_roc = bool(tau_m.any() and bkg_m.any())
    if have_roc:
        q_inner = np.quantile(sm[bkg_m], q_probs[:-1])
        q_edges = np.concatenate([q_inner, [float(np.max(sm[bkg_m])) + 1e-9, np.inf]])
        q_edges[0] = -np.inf
        n_qbins = len(q_edges) - 1
        sbin = np.clip(np.searchsorted(q_edges[1:-1], sm, side='right'), 0, n_qbins - 1)
        tau_hist = np.zeros((n_fam, n_qbins), dtype=np.float32)
        bkg_hist = np.zeros((n_fam, n_qbins), dtype=np.float32)
        if tau_m.any():
            np.add.at(tau_hist, (fam_idx[tau_m], sbin[tau_m]), 1.0)
        if bkg_m.any():
            np.add.at(bkg_hist, (fam_idx[bkg_m], sbin[bkg_m]), 1.0)

    rng = np.random.default_rng(seed)
    p_unif = np.full(n_fam, 1.0 / n_fam, dtype=np.float64)
    chunk = 250
    ba4_b, ba3_b, nll_b = [], [], []
    g1t_b, g1f_b, np1t_b, np1f_b, np01t_b, np01f_b = [], [], [], [], [], []
    roc1_b, roc01_b = [], []

    for start in range(0, n_boot, chunk):
        bs = min(chunk, n_boot - start)
        w = rng.multinomial(n_fam, p_unif, size=bs).astype(np.float64)  # (bs, n_fam)
        agg = w @ fam_stats  # (bs, 18)
        cnts = agg[:, 0:4]
        tot_n = np.maximum(cnts.sum(axis=1), 1.0)
        nll_b.append(agg[:, 11] / tot_n)
        with np.errstate(divide='ignore', invalid='ignore'):
            rec4 = agg[:, 4:8] / np.where(cnts > 0, cnts, np.nan)
            ba4_b.append(np.nanmean(rec4, axis=1))
            cnt3 = cnts[:, [0, 1, 3]]
            rec3 = agg[:, 8:11] / np.where(cnt3 > 0, cnt3, np.nan)
            ba3_b.append(np.nanmean(rec3, axis=1))
            if tau_m.any():
                g1t_b.append(agg[:, 12] / np.maximum(cnts[:, 2], 1.0))
                np1t_b.append(agg[:, 14] / np.maximum(cnts[:, 2], 1.0))
                np01t_b.append(agg[:, 16] / np.maximum(cnts[:, 2], 1.0))
            if bkg_m.any():
                nb = np.maximum(cnts[:, 0] + cnts[:, 1] + cnts[:, 3], 1.0)
                g1f_b.append(agg[:, 13] / nb)
                np1f_b.append(agg[:, 15] / nb)
                np01f_b.append(agg[:, 17] / nb)

        if have_roc:
            wf = w.astype(np.float32)
            th = (wf @ tau_hist)[:, ::-1].cumsum(axis=1)[:, ::-1]
            bh = (wf @ bkg_hist)[:, ::-1].cumsum(axis=1)[:, ::-1]
            t_tot = np.maximum(th[:, 0], 1.0)
            b_tot = np.maximum(bh[:, 0], 1.0)
            fpr_mat = bh / b_tot[:, None]
            tpr_mat = th / t_tot[:, None]
            for target_f, dest in ((0.01, roc1_b), (0.001, roc01_b)):
                # First descending-score bin from right (highest threshold) just above target FPR
                vals = np.empty(bs, dtype=np.float64)
                for r in range(bs):
                    # fpr_mat[r] decreases from 1.0 at idx=0 to ~0 at idx=n_qbins-1
                    ok = np.flatnonzero(fpr_mat[r] <= target_f)
                    if len(ok):
                        idx0 = int(ok[0])
                        vals[r] = float(tpr_mat[r, idx0])
                    else:
                        vals[r] = 0.0
                dest.append(vals)

    def _pack(chunks: list[np.ndarray]) -> dict | None:
        if not chunks:
            return None
        arr = np.concatenate(chunks, axis=0)
        arr = arr[np.isfinite(arr)]
        if not len(arr):
            return None
        return {
            'ci95': [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
            'se': float(np.std(arr, ddof=1)),
        }

    return {
        'n_families': n_fam,
        'n_boot': int(n_boot),
        'nll': _pack(nll_b),
        'BA4': _pack(ba4_b),
        'cond_BA3': _pack(ba3_b),
        'within_slice_S_tau_1pct': _pack(roc1_b),
        'within_slice_S_tau_0p1pct': _pack(roc01_b),
        'locked_global_1pct_tau_eff': _pack(g1t_b),
        'locked_global_1pct_fpr': _pack(g1f_b),
        'locked_np_1pct_tau_eff': _pack(np1t_b),
        'locked_np_1pct_fpr': _pack(np1f_b),
        'locked_np_0p1pct_tau_eff': _pack(np01t_b),
        'locked_np_0p1pct_fpr': _pack(np01f_b),
    }


def compute_migration_matrices(
    truth_meta: pd.DataFrame,
    probs: np.ndarray,
    applied: dict[str, np.ndarray],
) -> dict:
    """Compute 4x4 flavour migration and 8-subchannel physics migration/pass rates."""
    y = truth_meta['label'].to_numpy(dtype=np.int64)
    obs_dom = truth_meta['obs_domain_id'].to_numpy(dtype=np.int64)
    pred = probs.argmax(axis=1)
    class_names = ['nu_e_CC', 'nu_mu_CC', 'nu_tau_CC', 'NC']

    def _mat(mask: np.ndarray) -> dict:
        cm = np.zeros((4, 4), dtype=np.int64)
        np.add.at(cm, (y[mask], pred[mask]), 1)
        row_norm = (cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)).tolist()
        return {'counts': cm.tolist(), 'row_normalized': row_norm}

    matrices = {
        'All': _mat(np.ones(len(y), dtype=bool)),
        'D_pix_contained': _mat(obs_dom == 1),
        'D_scint_contained': _mat(obs_dom == 2),
        'Truth_pixel_centered': _mat(truth_meta['pixel_centered'].to_numpy(dtype=bool)),
    }

    tau_br = truth_meta['tau_branch'].astype(str).to_numpy()
    has_ch = truth_meta['has_charm'].to_numpy(dtype=bool)
    sub_specs = [
        ('nu_e_CC', (y == 0)),
        ('nu_mu_CC_no_charm', (y == 1) & (~has_ch)),
        ('nu_mu_CC_charm', (y == 1) & has_ch),
        ('nu_tau_e', (y == 2) & (tau_br == 'electron')),
        ('nu_tau_mu', (y == 2) & (tau_br == 'muon')),
        ('nu_tau_had', (y == 2) & (tau_br == 'hadronic')),
        ('NC_no_charm', (y == 3) & (~has_ch)),
        ('NC_charm', (y == 3) & has_ch),
    ]
    subchannels = {}
    for sname, sm in sub_specs:
        cnt = int(sm.sum())
        if cnt == 0:
            continue
        p_counts = np.bincount(pred[sm], minlength=4)
        subchannels[sname] = {
            'count': cnt,
            'pred_fractions': {class_names[c]: float(p_counts[c] / cnt) for c in range(4)},
            'locked_global_1pct_pass_rate': float(applied['pass_global_1pct'][sm].mean()),
            'locked_np_1pct_pass_rate': float(applied['pass_np_1pct'][sm].mean()),
            'locked_np_0p1pct_pass_rate': float(applied['pass_np_0p1pct'][sm].mean()),
        }

    return {'class_names': class_names, 'matrices_4x4': matrices, 'physics_subchannels': subchannels}


def profile_likelihood_significance(
    bin_ids: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 18,
    tau_prior: float = 0.01,
    benchmark_total: float = 10000.0,
    sigma_bkg_norm: float = 0.05,
    sigma_dom_norm: float = 0.05,
    delta_bkg_sys: np.ndarray | None = None,
) -> dict:
    """Compute both stat-only Asimov Z0 and profiled-likelihood Z0_PL with nuisance parameters."""
    valid = (bin_ids >= 0) & (bin_ids < n_bins)
    is_tau = labels == 2
    n_tau_tot = max(1, int(is_tau.sum()))
    n_bkg_tot = max(1, int((~is_tau).sum()))
    s_scale = (benchmark_total * tau_prior) / n_tau_tot
    b_scale = (benchmark_total * (1.0 - tau_prior)) / n_bkg_tot

    s_bins = np.zeros(n_bins, dtype=np.float64)
    b_bins = np.zeros(n_bins, dtype=np.float64)
    for b in range(n_bins):
        bm = valid & (bin_ids == b)
        s_bins[b] = float((bm & is_tau).sum()) * s_scale
        b_bins[b] = float((bm & (~is_tau)).sum()) * b_scale

    z0_sq_stat = 0.0
    per_bin = []
    bins_per_dom = n_bins // 3
    for b in range(n_bins):
        s, bk = float(s_bins[b]), float(b_bins[b])
        dom_idx = b // bins_per_dom + 1
        term = 2.0 * ((s + bk) * np.log(1.0 + s / bk) - s) if (s > 0.0 and bk > 0.0) else 0.0
        z0_sq_stat += term
        per_bin.append({
            'bin': int(b),
            'domain': OBS_DOMAIN_NAMES[dom_idx],
            's_expected': s,
            'b_expected': bk,
            's_over_b': float(s / max(bk, 1e-12)),
            'z0_sq_stat_contrib': float(term),
        })
    z0_stat = float(np.sqrt(max(z0_sq_stat, 0.0)))

    # Profile likelihood under H0 (mu = 0) with nuisance parameters theta = [eta_glob, eta_d1, eta_d2, eta_d3, alpha_sys]
    obs_asimov = s_bins + b_bins
    dom_of_bin = np.arange(n_bins) // bins_per_dom  # 0, 1, 2
    rel_sys = np.zeros(n_bins, dtype=np.float64) if delta_bkg_sys is None else np.asarray(delta_bkg_sys, dtype=np.float64)
    pos_mask = b_bins > 1e-12

    def _q0_obj(theta: np.ndarray) -> tuple[float, np.ndarray]:
        eta_g = float(theta[0])
        eta_d = theta[1:4]
        alpha = float(theta[4])
        expo = np.clip(sigma_bkg_norm * eta_g + sigma_dom_norm * eta_d[dom_of_bin] + rel_sys * alpha, -3.0, 3.0)
        nu = b_bins * np.exp(expo)

        q0_val = float(np.dot(theta, theta))
        grad_nu = np.zeros(n_bins, dtype=np.float64)
        for b in np.flatnonzero(pos_mask):
            n_b = obs_asimov[b]
            v_b = nu[b]
            if n_b > 0.0:
                q0_val += 2.0 * (v_b - n_b + n_b * np.log(n_b / v_b))
                grad_nu[b] = 2.0 * v_b * (1.0 - n_b / v_b)
            else:
                q0_val += 2.0 * v_b
                grad_nu[b] = 2.0 * v_b

        grad = 2.0 * theta.copy()
        grad[0] += sigma_bkg_norm * float(grad_nu.sum())
        for di in range(3):
            grad[1 + di] += sigma_dom_norm * float(grad_nu[dom_of_bin == di].sum())
        grad[4] += float(np.dot(grad_nu, rel_sys))
        return q0_val, grad

    res = minimize(
        _q0_obj,
        x0=np.zeros(5, dtype=np.float64),
        jac=True,
        method='L-BFGS-B',
        bounds=[(-5.0, 5.0)] * 5,
        options={'gtol': 1e-10, 'ftol': 1e-14, 'maxiter': 500},
    )
    q0_pl = float(max(0.0, res.fun))
    z0_pl = float(np.sqrt(q0_pl))
    return {
        'tau_prior': float(tau_prior),
        'benchmark_total': float(benchmark_total),
        'z0_stat_only': z0_stat,
        'z0_profile_likelihood': z0_pl,
        'q0_profile_likelihood': q0_pl,
        'nuisance_pulls_at_H0': {
            'eta_bkg_norm': float(res.x[0]),
            'eta_D_pix': float(res.x[1]),
            'eta_D_scint': float(res.x[2]),
            'eta_D_partial': float(res.x[3]),
            'alpha_readout_sys': float(res.x[4]),
        },
        'per_bin': per_bin,
    }


def _single_stream_asimov_z0(
    s_tau: np.ndarray,
    labels: np.ndarray,
    obs_dom: np.ndarray,
    tau_prior: float = 0.01,
    benchmark_total: float = 10000.0,
    edges: np.ndarray | None = None,
) -> float:
    active = obs_dom != 0
    is_tau = labels == 2
    s_scale = (benchmark_total * tau_prior) / max(1, int(is_tau.sum()))
    b_scale = (benchmark_total * (1.0 - tau_prior)) / max(1, int((~is_tau).sum()))
    if edges is None:
        edges = np.quantile(s_tau[active], np.linspace(0.0, 1.0, 19))
        edges[0], edges[-1] = -np.inf, np.inf
    else:
        edges = np.asarray(edges, dtype=np.float64)
    n_bins = len(edges) - 1
    b_idx = np.clip(np.searchsorted(edges[1:-1], s_tau[active], side='right'), 0, n_bins - 1)
    z0_sq = 0.0
    t_a = is_tau[active]
    for b in range(n_bins):
        bm = b_idx == b
        s = float((bm & t_a).sum()) * s_scale
        bk = float((bm & (~t_a)).sum()) * b_scale
        if s > 0.0 and bk > 0.0:
            z0_sq += 2.0 * ((s + bk) * np.log(1.0 + s / bk) - s)
    return float(np.sqrt(max(z0_sq, 0.0)))


def evaluate_14_stress_conditions(
    stress_meta: pd.DataFrame,
    stress_member_frames: dict[str, pd.DataFrame],
    cal_a_params: dict,
    cut_b_locks: dict,
    *,
    n_boot: int = 1000,
) -> dict:
    """Evaluate all 14 paired detector readout/systematic stress variants under frozen Cal-A + Cut-B."""
    variants = list(dict.fromkeys(stress_meta['variant'].tolist()))
    member_names = list(cal_a_params['member_names'])
    key_cols = KEY + ['variant']

    # Align on KEY + variant cleanly
    aligned_vars = {}
    ref_idx = pd.MultiIndex.from_frame(stress_meta[key_cols])
    for name in member_names:
        df = stress_member_frames[name]
        pos = pd.MultiIndex.from_frame(df[key_cols]).get_indexer(ref_idx)
        if (pos < 0).any():
            raise ValueError(f'Stress prediction frame {name} missing KEY+variant rows')
        sub = df.iloc[pos].reset_index(drop=True)
        p_cols = ['p_0', 'p_1', 'p_2', 'p_3']
        probs = sub[p_cols].to_numpy(dtype=np.float64)
        probs = probs / np.maximum(probs.sum(axis=-1, keepdims=True), 1e-12)
        lp = np.log(np.clip(probs, 1e-8, 1.0))
        lp = lp - lp.mean(axis=-1, keepdims=True)
        charm = np.clip(sub['charm_prob'].to_numpy(dtype=np.float64), 0.0, 1.0) if 'charm_prob' in sub.columns else None
        aligned_vars[name] = {'log_p': lp, 'charm': charm}

    log_p_stack = np.stack([aligned_vars[n]['log_p'] for n in member_names], axis=0)
    charms = [aligned_vars[n]['charm'] for n in member_names if aligned_vars[n]['charm'] is not None]
    mean_charm = np.mean(charms, axis=0) if charms else None
    obs_dom = stress_meta['obs_domain_id'].to_numpy(dtype=np.int64)
    labels = stress_meta['label'].to_numpy(dtype=np.int64)
    fam_codes, _ = pd.factorize(pd.MultiIndex.from_frame(stress_meta[FAMILY]))

    probs = _apply_cal_a_probs(log_p_stack, obs_dom, cal_a_params['domains'])
    s_tau = _compute_stau(probs, mean_charm, obs_dom, cut_b_locks['lambda_charm_by_domain'])
    g_cuts = cut_b_locks['strategy_i_global_cuts']
    np_cuts = cut_b_locks['strategy_ii_np_cuts']
    applied = {
        'pass_global_1pct': (obs_dom != 0) & (s_tau >= float(g_cuts['1pct_fpr'])),
        'pass_global_0p1pct': (obs_dom != 0) & (s_tau >= float(g_cuts['0p1pct_fpr'])),
        'pass_np_1pct': _apply_np_mask(s_tau, obs_dom, np_cuts['1pct_fpr']['thresholds_by_domain']),
        'pass_np_0p1pct': _apply_np_mask(s_tau, obs_dom, np_cuts['0p1pct_fpr']['thresholds_by_domain']),
        'bin_ids_tail': _assign_18_bins(s_tau, obs_dom, cut_b_locks['template_edges_tail_resolved']['edges_by_domain']),
        'bin_ids_unif': _assign_18_bins(s_tau, obs_dom, cut_b_locks['template_edges_uniform']['edges_by_domain']),
    }

    var_results = {}
    nom_bin_bkg = None
    nom_bin_bkg_unif = None
    bkg_var_fracs = []
    bkg_var_fracs_unif = []
    for vname in variants:
        vm = stress_meta['variant'].to_numpy() == vname
        rep = compute_locked_slice_report(labels, probs, s_tau, applied, vm)
        boot = vectorized_family_block_bootstrap(labels, probs, s_tau, applied, fam_codes, vm, n_boot=n_boot, seed=42)
        rep['bootstrap_ci95'] = boot
        # Track 18-bin background template yield variation across the 14 stress conditions (both tail and uniform binnings)
        bids_v = applied['bin_ids_tail'][vm]
        bids_u = applied['bin_ids_unif'][vm]
        is_bkg_v = labels[vm] != 2
        counts = np.array([int(((bids_v == b) & is_bkg_v).sum()) for b in range(18)], dtype=np.float64)
        fracs = counts / max(float(counts.sum()), 1.0)
        counts_u = np.array([int(((bids_u == b) & is_bkg_v).sum()) for b in range(18)], dtype=np.float64)
        fracs_u = counts_u / max(float(counts_u.sum()), 1.0)
        if vname == 'nominal':
            nom_bin_bkg = np.maximum(fracs, 1e-6)
            nom_bin_bkg_unif = np.maximum(fracs_u, 1e-6)
        var_results[vname] = rep
        bkg_var_fracs.append((vname, fracs))
        bkg_var_fracs_unif.append((vname, fracs_u))

    if nom_bin_bkg is None:
        nom_bin_bkg = np.full(18, 1.0 / 18.0, dtype=np.float64)
    if nom_bin_bkg_unif is None:
        nom_bin_bkg_unif = np.full(18, 1.0 / 18.0, dtype=np.float64)
    rel_devs = np.stack([(fr - nom_bin_bkg) / nom_bin_bkg for vname, fr in bkg_var_fracs if vname != 'nominal'], axis=0)
    rel_devs_u = np.stack([(fr - nom_bin_bkg_unif) / nom_bin_bkg_unif for vname, fr in bkg_var_fracs_unif if vname != 'nominal'], axis=0)
    # RMS relative bin yield perturbation across the 13 non-nominal detector readout stress conditions (capped safely at [0, 0.5])
    delta_bkg_sys = np.clip(np.sqrt(np.mean(rel_devs ** 2, axis=0)), 0.0, 0.50)
    delta_bkg_sys_unif = np.clip(np.sqrt(np.mean(rel_devs_u ** 2, axis=0)), 0.0, 0.50)

    return {
        'variants': var_results,
        'delta_bkg_sys_18bin_rms': delta_bkg_sys.tolist(),
        'delta_bkg_sys_18bin_unif_rms': delta_bkg_sys_unif.tolist(),
        'stress_bin_ids_tail': applied['bin_ids_tail'].tolist(),
        'stress_bin_ids_unif': applied['bin_ids_unif'].tolist(),
    }


def run_sequential_unsealing(v3_root: Path = DEFAULT_V3_ROOT, n_boot: int = 1000) -> dict:
    """Execute strict sequential Cal-A -> Cut-B -> Confirm200k evaluation and Profile-Likelihood."""
    w3_12 = v3_root / 'W3-12'
    meta_dir = w3_12 / 'offline_truth_meta'
    pred_dir = w3_12 / 'predictions'

    def _load_split_frames(split: str, role_frames: dict[str, pd.DataFrame] | None = None) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        ref = pd.read_parquet(meta_dir / f'{split}_truth_meta.parquet').sort_values(KEY).reset_index(drop=True)
        # Build both individual member DataFrames (15) and 3-seed averaged role DataFrames (5)
        members: dict[str, pd.DataFrame] = {}
        for role in ROLE_ORDER:
            seed_dfs = []
            for s in SEEDS:
                mid = f'{role}_s{s}'
                df = pd.read_parquet(pred_dir / split / f'{mid}.parquet')
                members[mid] = df
                seed_dfs.append(extract_member_arrays(df, ref))
            mean_p = np.mean([a['probs'] for a in seed_dfs], axis=0)
            mean_p = mean_p / np.maximum(mean_p.sum(axis=-1, keepdims=True), 1e-12)
            out_r = ref[KEY].copy().reset_index(drop=True)
            for c in range(4):
                out_r[f'p_{c}'] = mean_p[:, c].astype(np.float32)
            ch = [a['charm'] for a in seed_dfs if a['charm'] is not None]
            if ch:
                out_r['charm_prob'] = np.mean(ch, axis=0).astype(np.float32)
            members[f'{role}_3seed'] = out_r
        return ref, members

    ref_cal, mem_cal = _load_split_frames('cal_a')
    ref_cut, mem_cut = _load_split_frames('cut_b')
    ref_conf, mem_conf = _load_split_frames('confirm')

    role3_names = [f'{r}_3seed' for r in ROLE_ORDER]

    # Step 1: Fit calibrators strictly on Cal-A40k
    cal_a_ens = fit_cal_a_calibrators(mem_cal, ref_cal, member_names=role3_names)
    cal_a_pf_v4 = fit_cal_a_calibrators(mem_cal, ref_cal, member_names=['pf_v4_nomad_pauc_3seed'])
    cal_a_b1_v4 = fit_cal_a_calibrators(mem_cal, ref_cal, member_names=['b1pp_v4_nomad_pauc_3seed'])
    write_json(w3_12 / 'cal_a_frozen_calibrators.json', {'ens_v5': cal_a_ens, 'pf_v4_3seed': cal_a_pf_v4, 'b1pp_v4_3seed': cal_a_b1_v4})

    # Step 2: Lock operating points and 18-bin template edges strictly on Cut-B160k
    cut_b_ens = lock_cut_b_operating_points(mem_cut, ref_cut, cal_a_ens)
    cut_b_pf_v4 = lock_cut_b_operating_points(mem_cut, ref_cut, cal_a_pf_v4)
    cut_b_b1_v4 = lock_cut_b_operating_points(mem_cut, ref_cut, cal_a_b1_v4)
    write_json(w3_12 / 'cut_b_locked_thresholds.json', {'ens_v5': cut_b_ens, 'pf_v4_3seed': cut_b_pf_v4, 'b1pp_v4_3seed': cut_b_b1_v4})

    # Step 3: One-shot evaluation on Confirm200k (zero refit / zero retuning)
    obs_dom = ref_conf['obs_domain_id'].to_numpy(dtype=np.int64)
    labels = ref_conf['label'].to_numpy(dtype=np.int64)
    fam_codes, _ = pd.factorize(pd.MultiIndex.from_frame(ref_conf[FAMILY]))
    slices_conf = {
        'All': np.ones(len(ref_conf), dtype=bool),
        'D_empty': obs_dom == 0,
        'D_pix_contained': obs_dom == 1,
        'D_scint_contained': obs_dom == 2,
        'D_partial_or_edge': obs_dom == 3,
        'Truth_pixel_centered': ref_conf['pixel_centered'].to_numpy(dtype=bool),
        'Truth_large_scintillator': ref_conf['large_scintillator'].to_numpy(dtype=bool),
        'Confirm_E_le_2p3TeV': ref_conf['is_E_le_2p3TeV'].to_numpy(dtype=bool),
        'HighE_gt_2p3TeV_non_tau': ref_conf['is_highE_non_tau'].to_numpy(dtype=bool),
    }

    applied_ens = apply_frozen_pipeline(mem_conf, ref_conf, cal_a_ens, cut_b_ens)
    applied_pf_v4 = apply_frozen_pipeline(mem_conf, ref_conf, cal_a_pf_v4, cut_b_pf_v4)
    applied_b1_v4 = apply_frozen_pipeline(mem_conf, ref_conf, cal_a_b1_v4, cut_b_b1_v4)

    # Save frozen Confirm200k primary predictions for plots
    conf_out_df = ref_conf[KEY + FAMILY + ['label', 'obs_domain_id', 'pixel_centered', 'large_scintillator', 'is_E_le_2p3TeV', 'is_highE_non_tau', 'tau_branch', 'has_charm']].copy()
    for c in range(4):
        conf_out_df[f'p_{c}'] = applied_ens['probs'][:, c].astype(np.float32)
    conf_out_df['s_tau'] = applied_ens['s_tau'].astype(np.float32)
    conf_out_df['bin_id_tail'] = applied_ens['bin_ids_tail'].astype(np.int32)
    conf_out_df.to_parquet(pred_dir / 'confirm' / 'ens_v5_3seed_dual_regime_confirm.parquet', index=False, compression=COMPRESSION)

    def _summarize_model(applied: dict[str, np.ndarray], do_boot: bool = True) -> dict:
        out_slices = {}
        for sname, smask in slices_conf.items():
            rep = compute_locked_slice_report(labels, applied['probs'], applied['s_tau'], applied, smask)
            if do_boot and int(smask.sum()) >= 50:
                rep['bootstrap_ci95'] = vectorized_family_block_bootstrap(
                    labels, applied['probs'], applied['s_tau'], applied, fam_codes, smask, n_boot=n_boot, seed=42
                )
            out_slices[sname] = rep
        return out_slices

    ens_slices = _summarize_model(applied_ens, do_boot=True)
    pf_v4_slices = _summarize_model(applied_pf_v4, do_boot=True)
    b1_v4_slices = _summarize_model(applied_b1_v4, do_boot=True)

    # Also evaluate single-seed ensembles & unblended checkpoints on Confirm200k for multi-seed stability comparison
    per_checkpoint_confirm = {}
    for role in ROLE_ORDER:
        for s in SEEDS:
            mid = f'{role}_s{s}'
            arr_m = extract_member_arrays(mem_conf[mid], ref_conf)
            app_dummy = {
                'pass_global_1pct': np.zeros(len(ref_conf), dtype=bool),
                'pass_global_0p1pct': np.zeros(len(ref_conf), dtype=bool),
                'pass_np_1pct': np.zeros(len(ref_conf), dtype=bool),
                'pass_np_0p1pct': np.zeros(len(ref_conf), dtype=bool),
            }
            p_m = arr_m['probs']
            s_m = np.log(np.clip(p_m[:, 2], 1e-8, 1.0 - 1e-8) / np.clip(1.0 - p_m[:, 2], 1e-8, 1.0))
            per_checkpoint_confirm[mid] = {
                sn: compute_locked_slice_report(labels, p_m, s_m, app_dummy, slices_conf[sn])
                for sn in ('All', 'D_pix_contained', 'D_scint_contained', 'Truth_pixel_centered', 'Confirm_E_le_2p3TeV')
            }

    # Evaluate 14 paired readout stress conditions on Confirm200k (N=8,000 paired)
    stress_pf = pd.read_parquet(pred_dir / 'stress_pf.parquet')
    stress_b1 = pd.read_parquet(pred_dir / 'stress_b1pp.parquet')
    # Nominal stress replay parity check against primary Confirm200k predictions
    nom_pf = stress_pf[stress_pf['variant'] == 'nominal'].sort_values(KEY).reset_index(drop=True)
    nom_pos = pd.MultiIndex.from_frame(ref_conf[KEY]).get_indexer(pd.MultiIndex.from_frame(nom_pf[KEY]))
    max_nom_diff = 0.0
    per_ckpt_parity: dict[str, dict[str, float]] = {}
    for role in ROLE_ORDER:
        if not role.startswith('pf_'):
            continue
        for s in SEEDS:
            mid = f'{role}_s{s}'
            p_primary = extract_member_arrays(mem_conf[mid], ref_conf)['probs'][nom_pos]
            p_stress = nom_pf[[f'{mid}__p{c}' for c in range(4)]].to_numpy(dtype=np.float64)
            d_ev = np.max(np.abs(p_primary - p_stress), axis=1)
            mx = float(np.max(d_ev))
            max_nom_diff = max(max_nom_diff, mx)
            per_ckpt_parity[mid] = {
                'median_prob_diff': float(np.median(d_ev)),
                'p99_prob_diff': float(np.quantile(d_ev, 0.99)),
                'max_prob_diff': mx,
            }

    nom_b1 = stress_b1[stress_b1['variant'] == 'nominal'].sort_values(KEY).reset_index(drop=True)
    nom_b1_pos = pd.MultiIndex.from_frame(ref_conf[KEY]).get_indexer(pd.MultiIndex.from_frame(nom_b1[KEY]))
    for role in ROLE_ORDER:
        if not role.startswith('b1pp_'):
            continue
        for s in SEEDS:
            mid = f'{role}_s{s}'
            p_primary = extract_member_arrays(mem_conf[mid], ref_conf)['probs'][nom_b1_pos]
            p_stress = nom_b1[[f'{mid}__p{c}' for c in range(4)]].to_numpy(dtype=np.float64)
            d_ev = np.max(np.abs(p_primary - p_stress), axis=1)
            mx = float(np.max(d_ev))
            max_nom_diff = max(max_nom_diff, mx)
            per_ckpt_parity[mid] = {
                'median_prob_diff': float(np.median(d_ev)),
                'p99_prob_diff': float(np.quantile(d_ev, 0.99)),
                'max_prob_diff': mx,
            }

    bad_ckpt = [m for m, st in per_ckpt_parity.items() if st['median_prob_diff'] >= 1e-3]
    if bad_ckpt:
        raise RuntimeError(f'Nominal stress replay checkpoint median diff >= 1e-3: {bad_ckpt}')

    # Build role_3seed averaged stress DataFrames keyed by (KEY + ['variant'])
    stress_meta = stress_pf[KEY + ['label', 'variant', 'obs_domain_id']].merge(
        stress_b1[KEY + ['variant']], on=KEY + ['variant'], validate='one_to_one'
    )
    stress_meta = stress_meta.merge(ref_conf[KEY + FAMILY], on=KEY, how='left', validate='many_to_one')
    stress_member_frames: dict[str, pd.DataFrame] = {}
    st_ref_idx = pd.MultiIndex.from_frame(stress_meta[KEY + ['variant']])
    b1_pos = pd.MultiIndex.from_frame(stress_b1[KEY + ['variant']]).get_indexer(st_ref_idx)
    pf_pos = pd.MultiIndex.from_frame(stress_pf[KEY + ['variant']]).get_indexer(st_ref_idx)

    for role in ROLE_ORDER:
        src = stress_pf.iloc[pf_pos] if role.startswith('pf_') else stress_b1.iloc[b1_pos]
        probs_stack = [src[[f'{role}_s{s}__p{c}' for c in range(4)]].to_numpy(dtype=np.float64) for s in SEEDS]
        mp = np.mean(probs_stack, axis=0)
        mp = mp / np.maximum(mp.sum(axis=-1, keepdims=True), 1e-12)
        df_s = stress_meta[KEY + ['variant']].copy().reset_index(drop=True)
        for c in range(4):
            df_s[f'p_{c}'] = mp[:, c].astype(np.float32)
        if f'{role}_s42__charm' in src.columns:
            df_s['charm_prob'] = np.mean([src[f'{role}_s{s}__charm'].to_numpy(dtype=np.float64) for s in SEEDS], axis=0).astype(np.float32)
        stress_member_frames[f'{role}_3seed'] = df_s

    stress_eval = evaluate_14_stress_conditions(
        stress_meta, stress_member_frames, cal_a_ens, cut_b_ens, n_boot=n_boot
    )
    delta_sys_18 = np.asarray(stress_eval['delta_bkg_sys_18bin_rms'], dtype=np.float64)
    delta_sys_18_unif = np.asarray(stress_eval['delta_bkg_sys_18bin_unif_rms'], dtype=np.float64)

    # Heterogeneous-GPU ensemble nominal stress replay parity over all N=8,000 paired events
    nom_meta = stress_meta[stress_meta['variant'] == 'nominal'].sort_values(KEY).reset_index(drop=True)
    nom_mems = {
        mname: mdf[mdf['variant'] == 'nominal'].sort_values(KEY).reset_index(drop=True)
        for mname, mdf in stress_member_frames.items()
    }
    app_nom = apply_frozen_pipeline(nom_mems, nom_meta, cal_a_ens, cut_b_ens)
    pos_primary = pd.MultiIndex.from_frame(ref_conf[KEY]).get_indexer(pd.MultiIndex.from_frame(nom_meta[KEY]))
    ediff = np.max(np.abs(applied_ens['probs'][pos_primary] - app_nom['probs']), axis=1)
    y_nom = nom_meta['label'].to_numpy(dtype=np.int64)
    all_nom_mask = np.ones(len(y_nom), dtype=bool)
    r_prim_8k = compute_locked_slice_report(
        y_nom,
        applied_ens['probs'][pos_primary],
        applied_ens['s_tau'][pos_primary],
        {k: v[pos_primary] for k, v in applied_ens.items() if k.startswith('pass_')},
        all_nom_mask,
    )
    r_snom_8k = compute_locked_slice_report(y_nom, app_nom['probs'], app_nom['s_tau'], app_nom, all_nom_mask)
    nominal_parity = {
        'n_events': int(len(y_nom)),
        'ens_median_prob_diff': float(np.median(ediff)),
        'ens_mean_prob_diff': float(np.mean(ediff)),
        'ens_p99_prob_diff': float(np.quantile(ediff, 0.99)),
        'ens_p99_9_prob_diff': float(np.quantile(ediff, 0.999)),
        'ens_max_prob_diff': float(np.max(ediff)),
        'primary_8k_nll': float(r_prim_8k['nll']),
        'stress_nominal_8k_nll': float(r_snom_8k['nll']),
        'ens_abs_nll_diff': float(abs(r_prim_8k['nll'] - r_snom_8k['nll'])),
        'primary_8k_ba4': float(r_prim_8k['BA4']),
        'stress_nominal_8k_ba4': float(r_snom_8k['BA4']),
        'ens_abs_ba4_diff': float(abs(r_prim_8k['BA4'] - r_snom_8k['BA4'])),
        'primary_8k_locked_np1_tau_eff': float(r_prim_8k['locked_pass_np_1pct']['tau_eff']),
        'stress_nominal_8k_locked_np1_tau_eff': float(r_snom_8k['locked_pass_np_1pct']['tau_eff']),
        'ens_abs_locked_np1_tau_eff_diff': float(
            abs(r_prim_8k['locked_pass_np_1pct']['tau_eff'] - r_snom_8k['locked_pass_np_1pct']['tau_eff'])
        ),
        'per_checkpoint': per_ckpt_parity,
    }
    if nominal_parity['ens_median_prob_diff'] >= 1e-3:
        raise RuntimeError(f"Ensemble median nominal diff >= 1e-3: {nominal_parity['ens_median_prob_diff']:.6f}")
    if nominal_parity['ens_p99_prob_diff'] >= 3e-3:
        raise RuntimeError(f"Ensemble p99 nominal diff >= 3e-3: {nominal_parity['ens_p99_prob_diff']:.6f}")
    if nominal_parity['ens_abs_nll_diff'] >= 1e-3:
        raise RuntimeError(f"Ensemble |delta NLL| >= 1e-3: {nominal_parity['ens_abs_nll_diff']:.6f}")
    if nominal_parity['ens_abs_ba4_diff'] >= 2e-3:
        raise RuntimeError(f"Ensemble |delta BA4| >= 2e-3: {nominal_parity['ens_abs_ba4_diff']:.6f}")

    # Profile-likelihood and stat-only Asimov significance across priors and exposures
    sig_scan = {}
    for n_tot in (10000.0, 50000.0):
        key_n = f'N_{int(n_tot)}'
        sig_scan[key_n] = []
        for prior in (0.002, 0.005, 0.01, 0.02, 0.03, 0.05):
            pl_tail = profile_likelihood_significance(
                applied_ens['bin_ids_tail'], labels, tau_prior=prior, benchmark_total=n_tot, delta_bkg_sys=delta_sys_18
            )
            pl_unif = profile_likelihood_significance(
                applied_ens['bin_ids_unif'], labels, tau_prior=prior, benchmark_total=n_tot, delta_bkg_sys=delta_sys_18_unif
            )
            z_single = _single_stream_asimov_z0(applied_ens['s_tau'], labels, obs_dom, tau_prior=prior, benchmark_total=n_tot)
            sig_scan[key_n].append({
                'tau_prior': float(prior),
                'z0_single_stream_stat': z_single,
                'tail_18bin': pl_tail,
                'uniform_18bin': pl_unif,
            })

    migration = compute_migration_matrices(ref_conf, applied_ens['probs'], applied_ens)

    # Realistic forward-flux + signed morphing + Charm-CR + Barlow-Beeston lite PL scan (Audit P0-1)
    signed_sys = extract_signed_systematic_templates(
        stress_meta, np.asarray(stress_eval['stress_bin_ids_tail'], dtype=np.int64), n_bins=18
    )
    sr_bins_18, joint_bins_24 = assign_signal_and_charm_cr_bins(
        applied_ens['s_tau'],
        applied_ens.get('charm_prob'),
        obs_dom,
        cut_b_ens['template_edges_tail_resolved']['edges_by_domain'],
    )
    has_ch_conf = ref_conf['has_charm'].to_numpy(dtype=bool)
    realistic_flux_scan = {}
    for scen_name, scen in LHC_FORWARD_SCENARIOS.items():
        tpl_18 = build_physical_templates(
            sr_bins_18,
            labels,
            has_ch_conf,
            n_bins=18,
            total_events=scen['expected_total_events'],
            tau_fraction=scen['tau_fraction'],
            bkg_shares=scen['bkg_shares'],
        )
        tpl_24 = build_physical_templates(
            joint_bins_24,
            labels,
            has_ch_conf,
            n_bins=24,
            total_events=scen['expected_total_events'],
            tau_fraction=scen['tau_fraction'],
            bkg_shares=scen['bkg_shares'],
        )
        realistic_flux_scan[scen_name] = {
            'scenario': scen,
            'sr_18bin_only': fit_realistic_profile_likelihood(tpl_18, signed_sys=signed_sys, n_sr_bins=18),
            'sr_plus_charm_cr_24bin': fit_realistic_profile_likelihood(tpl_24, signed_sys=signed_sys, n_sr_bins=18),
        }

    probes_path_pf = w3_12 / 'probes_confirm_pf.json'
    probes_path_b1 = w3_12 / 'probes_confirm_b1pp.json'
    probes_pf = json.loads(probes_path_pf.read_text()) if probes_path_pf.exists() else {}
    probes_b1 = json.loads(probes_path_b1.read_text()) if probes_path_b1.exists() else {}

    out = {
        'status': 'complete',
        'UTC': datetime.now(timezone.utc).isoformat(),
        'cal_a': cal_a_ens,
        'cut_b': cut_b_ens,
        'confirm200k': {
            'events': int(len(ref_conf)),
            'families': int(fam_codes.max() + 1),
            'ens_v5_3seed_dual_regime': ens_slices,
            'pf_v4_nomad_pauc_3seed': pf_v4_slices,
            'b1pp_v4_nomad_pauc_3seed': b1_v4_slices,
            'per_checkpoint_raw': per_checkpoint_confirm,
            'migration': migration,
            'stress_14_conditions': stress_eval,
            'nominal_stress_replay_max_prob_diff': float(max_nom_diff),
            'nominal_stress_replay_parity': nominal_parity,
            'significance_scan': sig_scan,
            'realistic_forward_flux_significance': realistic_flux_scan,
            'probes_pf': probes_pf,
            'probes_b1pp': probes_b1,
        },
    }
    write_json(w3_12 / 'w3_12_confirmation_results.json', out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v3-root', type=Path, default=DEFAULT_V3_ROOT)
    parser.add_argument('--n-boot', type=int, default=1000)
    args = parser.parse_args()
    res = run_sequential_unsealing(v3_root=args.v3_root, n_boot=args.n_boot)
    e = res['confirm200k']['ens_v5_3seed_dual_regime']
    n10 = next(r for r in res['confirm200k']['significance_scan']['N_10000'] if r['tau_prior'] == 0.01)
    np_summary = {k: v for k, v in res['confirm200k']['nominal_stress_replay_parity'].items() if k != 'per_checkpoint'}
    print(json.dumps({
        'status': res['status'],
        'confirm_events': res['confirm200k']['events'],
        'confirm_families': res['confirm200k']['families'],
        'All': e['All'],
        'D_scint_contained': e['D_scint_contained'],
        'Truth_pixel_centered': e['Truth_pixel_centered'],
        'nominal_diff': res['confirm200k']['nominal_stress_replay_max_prob_diff'],
        'nominal_parity': np_summary,
        'sig_10k_1pct': {
            'z0_single_stream_stat': n10['z0_single_stream_stat'],
            'z0_18bin_tail_stat': n10['tail_18bin']['z0_stat_only'],
            'z0_18bin_tail_PL': n10['tail_18bin']['z0_profile_likelihood'],
        },
    }, indent=2))


if __name__ == '__main__':
    main()
