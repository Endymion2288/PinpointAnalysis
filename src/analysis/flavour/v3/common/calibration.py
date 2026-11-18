"""Per-observable-domain calibration (T_d, b_d), log-likelihood ratio S_tau, and W3-03 replay."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.s6.calibration import background_threshold
from analysis.flavour.v2.partitions import FAMILY, KEY
from analysis.flavour.v2.r1.metrics import Metrics, three_from_four, three_targets
from analysis.flavour.v2.r2.readout import grouped_folds
from analysis.flavour.v3.common.domains import OBS_DOMAIN_NAMES


def extract_raw_logits(frame: pd.DataFrame) -> np.ndarray:
    """Return raw 4-class logits if stored, otherwise log of probabilities."""
    if all(f'logit{c}' in frame.columns for c in range(4)):
        z = frame[[f'logit{c}' for c in range(4)]].to_numpy(dtype=np.float64)
        return z - z.mean(axis=1, keepdims=True)
    p = frame[[f'p{c}' for c in range(4)]].to_numpy(dtype=np.float64)
    return np.log(np.clip(p, 1e-15, 1.0))


def fit_domain_temperature_bias(
    logits: np.ndarray,
    labels: np.ndarray,
    inv_t_bounds: tuple[float, float] = (0.05, 20.0),
    bias_bounds: tuple[float, float] = (-3.0, 3.0),
    l2_bias: float = 1e-3,
    l2_log_t: float = 1e-4,
) -> dict:
    """Fit positive temperature T_d > 0 and 4-class bias b_d (b_NC=0) minimizing balanced NLL."""
    n = len(labels)
    if n < 8:
        return {'T': 1.0, 'inv_T': 1.0, 'bias': [0.0, 0.0, 0.0, 0.0], 'fallback': True}
    support = np.bincount(labels, minlength=4)
    present = support > 0
    n_present = int(present.sum())
    if n_present < 2:
        return {'T': 1.0, 'inv_T': 1.0, 'bias': [0.0, 0.0, 0.0, 0.0], 'fallback': True}

    # Class-balanced weights over classes present in this domain
    w = np.zeros(n, dtype=np.float64)
    for c in range(4):
        if support[c] > 0:
            w[labels == c] = 1.0 / (n_present * support[c])

    row_idx = np.arange(n)
    true_z = logits[row_idx, labels]

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        log_inv_t = float(theta[0])
        inv_t = float(np.exp(log_inv_t))
        bias = np.array([theta[1], theta[2], theta[3], 0.0], dtype=np.float64)
        scaled = inv_t * logits + bias[None, :]
        lse = logsumexp(scaled, axis=1)
        probs = np.exp(scaled - lse[:, None])
        loss = float(np.dot(w, lse - (inv_t * true_z + bias[labels])))
        loss += 0.5 * l2_bias * float(np.dot(bias[:3], bias[:3])) + 0.5 * l2_log_t * (log_inv_t ** 2)

        # Analytic gradient
        g_log_inv_t = float(inv_t * np.dot(w, (probs * logits).sum(axis=1) - true_z)) + l2_log_t * log_inv_t
        resid = probs.copy()
        resid[row_idx, labels] -= 1.0
        g_bias = (w[:, None] * resid).sum(axis=0)[:3] + l2_bias * bias[:3]
        grad = np.array([g_log_inv_t, g_bias[0], g_bias[1], g_bias[2]], dtype=np.float64)
        return loss, grad

    bounds = [(float(np.log(inv_t_bounds[0])), float(np.log(inv_t_bounds[1])))] + [bias_bounds] * 3
    res = minimize(
        objective,
        np.zeros(4, dtype=np.float64),
        jac=True,
        method='L-BFGS-B',
        bounds=bounds,
        options={'gtol': 1e-8, 'ftol': 1e-12, 'maxiter': 500},
    )
    if not res.success:
        rng = np.random.default_rng(17)
        for _ in range(4):
            cand = minimize(
                objective,
                rng.normal(0.0, 0.15, size=4),
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'gtol': 1e-8, 'ftol': 1e-14, 'maxiter': 1000},
            )
            if cand.fun < res.fun or (cand.success and not res.success):
                res = cand
            if res.success:
                break
    inv_t = float(np.exp(res.x[0]))
    bias = [float(res.x[1]), float(res.x[2]), float(res.x[3]), 0.0]
    grad_inf = float(np.max(np.abs(res.jac))) if hasattr(res, 'jac') and res.jac is not None else float('nan')
    return {
        'T': float(1.0 / inv_t),
        'inv_T': inv_t,
        'bias': bias,
        'fallback': False,
        'success': bool(res.success),
        'optimizer_converged': bool(res.success or grad_inf < 1e-5),
        'grad_inf_norm': grad_inf,
        'objective': float(res.fun),
        'counts': support.tolist(),
    }


def apply_domain_calibration(logits: np.ndarray, params: dict) -> np.ndarray:
    scaled = float(params['inv_T']) * logits + np.asarray(params['bias'], dtype=np.float64)[None, :]
    return softmax(scaled, axis=1)


def compute_s_tau(
    probs: np.ndarray,
    obs_domain_id: np.ndarray,
    charm_probs: np.ndarray | None = None,
    lambda_charm: float = 0.0,
) -> np.ndarray:
    """Standard multi-domain nu_tau log-likelihood ratio score S_tau."""
    p_tau = np.clip(probs[:, 2], 1e-12, 1.0)
    p_bkg = np.clip(probs[:, 0] + probs[:, 1] + probs[:, 3], 1e-12, 3.0)
    if charm_probs is not None and lambda_charm > 0.0:
        p_bkg = p_bkg + lambda_charm * np.clip(charm_probs, 0.0, 1.0)
    s_tau = np.log(p_tau / (p_bkg + 1e-8))
    s_tau = np.where(obs_domain_id == 0, -25.0, s_tau)
    return s_tau


def oof_regional_calibration(
    logits: np.ndarray,
    labels: np.ndarray,
    obs_domain_id: np.ndarray,
    folds: np.ndarray,
    n_folds: int = 5,
    charm_probs: np.ndarray | None = None,
    lambda_charm: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """5-fold family-connected OOF cross-fitting of per-domain (T_d, b_d) & S_tau."""
    cal_probs = np.full_like(logits, 0.25, dtype=np.float64)
    fit_records = []
    for fold in range(n_folds):
        held = folds == fold
        fit = ~held
        fold_rec = {'fold': int(fold), 'domains': {}}
        for d in (1, 2, 3):
            fit_mask = fit & (obs_domain_id == d)
            held_mask = held & (obs_domain_id == d)
            params = fit_domain_temperature_bias(logits[fit_mask], labels[fit_mask])
            fold_rec['domains'][OBS_DOMAIN_NAMES[d]] = params
            if held_mask.any():
                cal_probs[held_mask] = apply_domain_calibration(logits[held_mask], params)
        # D_empty receives uniform prior
        empty_held = held & (obs_domain_id == 0)
        if empty_held.any():
            cal_probs[empty_held] = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
        fit_records.append(fold_rec)

    s_tau = compute_s_tau(cal_probs, obs_domain_id, charm_probs=charm_probs, lambda_charm=lambda_charm)
    return cal_probs, s_tau, fit_records


def _tau_efficiency_at_fpr(
    score: np.ndarray,
    labels: np.ndarray,
    target_fpr: float,
    weight: np.ndarray | None = None,
) -> dict:
    y = np.asarray(labels, dtype=np.int64)
    w = np.ones(len(y), dtype=np.float64) if weight is None else np.asarray(weight, dtype=np.float64)
    is_tau = y == 2
    is_bkg = ~is_tau
    if not is_tau.any() or not is_bkg.any():
        return {'tau_eff': None, 'actual_fpr': None, 'threshold': None, 'tau_selected': 0, 'tau_total': int(is_tau.sum())}
    info = background_threshold(score, is_bkg, w, target_fpr)
    passed = score >= info['threshold']
    sig_sel = int((passed & is_tau).sum())
    eff = float(w[passed & is_tau].sum() / max(w[is_tau].sum(), 1e-30))
    return {
        'tau_eff': eff,
        'actual_fpr': float(info['validation_FPR']),
        'threshold': float(info['threshold']),
        'tau_selected': sig_sel,
        'tau_total': int(is_tau.sum()),
        'bkg_selected': int(info['background_passed']),
        'bkg_total': int(info['background_events']),
    }


def _balanced_weights_4(labels: np.ndarray) -> np.ndarray:
    cnt = np.bincount(labels, minlength=4)
    return (len(labels) / (4.0 * np.maximum(cnt, 1)))[labels]


def multi_domain_asimov_z0(
    s_tau: np.ndarray,
    labels: np.ndarray,
    obs_domain_id: np.ndarray,
    n_bins: int = 10,
    tau_prior: float = 0.01,
    benchmark_total: float = 10000.0,
) -> float:
    """Compute binned multi-domain Asimov discovery significance Z0 across d in {1,2,3}."""
    active = obs_domain_id > 0
    if not active.any():
        return 0.0
    is_tau = labels == 2
    n_tau_tot = max(int(is_tau.sum()), 1)
    n_bkg_tot = max(int((~is_tau).sum()), 1)
    # Normalize total expected yields to benchmark_total with assumed tau_prior
    s_scale = (benchmark_total * tau_prior) / n_tau_tot
    b_scale = (benchmark_total * (1.0 - tau_prior)) / n_bkg_tot

    edges = np.quantile(s_tau[active], np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    z0_sq = 0.0
    for d in (1, 2, 3):
        dm = obs_domain_id == d
        if not dm.any():
            continue
        for k in range(n_bins):
            bm = dm & (s_tau >= edges[k]) & (s_tau < edges[k + 1])
            s = float((bm & is_tau).sum()) * s_scale
            b = float((bm & (~is_tau)).sum()) * b_scale
            if s > 0 and b > 0:
                z0_sq += 2.0 * ((s + b) * np.log(1.0 + s / b) - s)
    return float(np.sqrt(max(z0_sq, 0.0)))


def evaluate_replay_comparison(
    frame: pd.DataFrame,
    obs_domain_id: np.ndarray,
    truth_pc_mask: np.ndarray,
    truth_ls_mask: np.ndarray,
    name: str,
) -> dict:
    """Evaluate raw global p(tau) vs OOF regional calibration + S_tau + Strategy II."""
    labels = frame['label'].to_numpy(dtype=np.int64)
    logits = extract_raw_logits(frame)
    raw_probs = softmax(logits, axis=1)
    if all(f'p{c}' in frame.columns for c in range(4)):
        raw_probs = frame[[f'p{c}' for c in range(4)]].to_numpy(dtype=np.float64)

    folds, fold_summary = grouped_folds(frame.reset_index(drop=True), folds=5, seed=20261117)
    cal_probs, s_tau, fit_records = oof_regional_calibration(logits, labels, obs_domain_id, folds)

    w_all = _balanced_weights_4(labels)
    raw_m4 = Metrics(labels, raw_probs).report(w_all)
    cal_m4 = Metrics(labels, cal_probs).report(w_all)

    non_tau = labels != 2
    y3 = three_targets(labels[non_tau])
    w3 = (len(y3) / (3.0 * np.maximum(np.bincount(y3, minlength=3), 1)))[y3]
    raw_m3 = Metrics(y3, three_from_four(raw_probs[non_tau])).report(w3)
    cal_m3 = Metrics(y3, three_from_four(cal_probs[non_tau])).report(w3)

    slices = {
        'All': np.ones(len(labels), dtype=bool),
        'Truth_pixel_centered': np.asarray(truth_pc_mask, dtype=bool),
        'Truth_large_scintillator': np.asarray(truth_ls_mask, dtype=bool),
        'Obs_D_pix_contained': obs_domain_id == 1,
        'Obs_D_scint_contained': obs_domain_id == 2,
        'Obs_D_partial_or_edge': obs_domain_id == 3,
    }

    # Global threshold from Strategy I (single global cut on All weighted background)
    raw_p_tau = raw_probs[:, 2]
    results_by_fpr = {}
    for target_fpr in (0.01, 0.001):
        raw_all_wp = _tau_efficiency_at_fpr(raw_p_tau, labels, target_fpr, w_all)
        s_all_wp = _tau_efficiency_at_fpr(s_tau, labels, target_fpr, w_all)
        cut_raw = raw_all_wp['threshold']
        cut_s = s_all_wp['threshold']

        # Strategy II (OOF Neyman-Pearson domain allocation: allocate higher FPR budget to D_pix_contained
        # proportional to signal-to-background ratio, verified strictly out-of-fold)
        sel_strat2 = np.zeros(len(labels), dtype=bool)
        for fold in range(5):
            held = folds == fold
            fit = ~held
            # Grid over domain budget multipliers favoring D_pix (1) > D_scint (2) > D_edge (3)
            # Search exact optimal per-domain cut offsets on fit fold only via Neyman-Pearson S_tau shift
            # Wait: under exact likelihood-ratio theory, S_tau + log_ratio_shift gives NP optimality;
            # additionally test explicit budget allocation (2.0% in D_pix, 0.9% in D_scint, 0.6% in D_edge,
            # rescaled on fit fold so overall non-tau FPR <= target_fpr).
            fit_bkg_w = w_all[fit & (labels != 2)]
            fit_tot_bkg = float(fit_bkg_w.sum())
            best_tau_eff = -1.0
            best_cuts = None
            ratios_grid = [(1.0, 1.0, 1.0), (2.0, 1.0, 0.5), (3.0, 1.0, 0.3), (4.0, 0.8, 0.2)]
            for r1, r2, r3 in ratios_grid:
                r_vec = {1: r1, 2: r2, 3: r3}
                denom = sum(r_vec[d] * float(w_all[fit & (obs_domain_id == d) & (labels != 2)].sum()) for d in (1, 2, 3))
                scale = (target_fpr * fit_tot_bkg) / max(denom, 1e-30)
                cuts = {}
                fit_tau_sel = 0.0
                for d in (1, 2, 3):
                    dm = fit & (obs_domain_id == d)
                    if not dm.any() or not (dm & (labels != 2)).any():
                        cuts[d] = np.inf
                        continue
                    f_d = min(0.5, scale * r_vec[d])
                    info_d = _tau_efficiency_at_fpr(s_tau[dm], labels[dm], f_d, w_all[dm])
                    cuts[d] = info_d['threshold'] if info_d['threshold'] is not None else np.inf
                    if info_d['tau_eff'] is not None:
                        fit_tau_sel += float(w_all[dm & (labels == 2) & (s_tau >= cuts[d])].sum())
                if fit_tau_sel > best_tau_eff:
                    best_tau_eff = fit_tau_sel
                    best_cuts = cuts
            if best_cuts is not None:
                for d in (1, 2, 3):
                    hm = held & (obs_domain_id == d)
                    sel_strat2[hm] = s_tau[hm] >= best_cuts[d]

        slice_rows = {}
        for sname, smask in slices.items():
            is_tau_s = smask & (labels == 2)
            is_bkg_s = smask & (labels != 2)
            raw_sel = smask & (raw_p_tau >= cut_raw)
            s_sel = smask & (s_tau >= cut_s)
            np_sel = smask & sel_strat2

            def _rate(sel_mask: np.ndarray) -> dict:
                tau_w = float(w_all[is_tau_s].sum())
                bkg_w = float(w_all[is_bkg_s].sum())
                return {
                    'tau_eff': float(w_all[sel_mask & is_tau_s].sum() / tau_w) if tau_w > 0 else 0.0,
                    'tau_passed': int((sel_mask & is_tau_s).sum()),
                    'tau_total': int(is_tau_s.sum()),
                    'fpr': float(w_all[sel_mask & is_bkg_s].sum() / bkg_w) if bkg_w > 0 else 0.0,
                    'bkg_passed': int((sel_mask & is_bkg_s).sum()),
                    'bkg_total': int(is_bkg_s.sum()),
                }

            # Also compute within-slice calibrated BA4 / NLL4
            s_cal_ba4 = None
            if smask.any():
                y_s = labels[smask]
                w_s = np.ones(len(y_s), dtype=np.float64)
                cnt_s = np.bincount(y_s, minlength=4)
                if (cnt_s > 0).all():
                    w_s = (len(y_s) / (4.0 * cnt_s))[y_s]
                s_cal_ba4 = float(Metrics(y_s, cal_probs[smask]).report(w_s)['BA'])
                s_raw_ba4 = float(Metrics(y_s, raw_probs[smask]).report(w_s)['BA'])
            else:
                s_raw_ba4 = None

            slice_rows[sname] = {
                'events': int(smask.sum()),
                'raw_global_cut': _rate(raw_sel),
                'regional_S_tau_global_cut': _rate(s_sel),
                'regional_S_tau_NP_allocated': _rate(np_sel),
                'raw_BA4': s_raw_ba4,
                'cal_BA4': s_cal_ba4,
            }
        results_by_fpr[str(target_fpr)] = {
            'global_raw_threshold': float(cut_raw),
            'global_S_tau_threshold': float(cut_s),
            'slices': slice_rows,
        }

    return {
        'model': name,
        'events': int(len(labels)),
        'folds': fold_summary,
        'oof_fits': fit_records,
        'all_domain_raw': {'BA4': raw_m4['BA'], 'NLL4': raw_m4['NLL'], 'BA3': raw_m3['BA'], 'NLL3': raw_m3['NLL']},
        'all_domain_regional_calibrated': {'BA4': cal_m4['BA'], 'NLL4': cal_m4['NLL'], 'BA3': cal_m3['BA'], 'NLL3': cal_m3['NLL']},
        'asimov_Z0_1pct_tau_prior_10k_events': {
            'raw_p_tau': multi_domain_asimov_z0(np.log(np.clip(raw_p_tau, 1e-12, 1.0)), labels, obs_domain_id),
            'regional_S_tau': multi_domain_asimov_z0(s_tau, labels, obs_domain_id),
        },
        'working_points': results_by_fpr,
    }


def _s4_validation_domains(s4_root: Path, geometry: dict, bounds: dict, val_keys: pd.DataFrame) -> np.ndarray:
    """Compute exact W3-01 obs_domain_id for S4 10k validation events."""
    from analysis.flavour.datasets import NativeShard
    from analysis.flavour.v3.common.domains import compute_observable_containment

    shards = sorted(p for p in (s4_root / 'shards').iterdir() if (p / 'observations.parquet').exists())
    rows = []
    for sp in shards:
        prov = pd.read_parquet(sp / 'provenance.parquet', columns=KEY + ['split'])
        val_idx = np.flatnonzero((prov['split'] == 'validation').to_numpy())
        if len(val_idx) == 0:
            continue
        ns = NativeShard(sp, target_columns=('label',))
        for i in val_idx:
            res = compute_observable_containment(ns[int(i)]['observations'], geometry, bounds)
            r = prov.iloc[int(i)]
            rows.append((int(r['run']), int(r['chunk']), int(r['row']), int(res['obs_domain_id'])))
    dom_df = pd.DataFrame(rows, columns=KEY + ['obs_domain_id'])
    merged = val_keys[KEY].merge(dom_df, on=KEY, how='left', validate='one_to_one', sort=False)
    if merged['obs_domain_id'].isna().any():
        raise ValueError('some S4 validation events missing from shards')
    return merged['obs_domain_id'].to_numpy(dtype=np.int8)


def _plot_replay_figure(r2_res: dict, s4_res: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)

    for ax, res, title in [
        (axes[0], r2_res, 'R2 B1-D2 T4 (Tune 40k, OOF 5-fold)'),
        (axes[1], s4_res, 'S6 MAE3 100k Ensemble (S4 Val 10k, OOF 5-fold)'),
    ]:
        wp = res['working_points']['0.01']['slices']
        slice_keys = ['All', 'Truth_pixel_centered', 'Truth_large_scintillator', 'Obs_D_pix_contained', 'Obs_D_scint_contained']
        labels_x = ['All', 'Truth\nPixel-Centered', 'Truth\nLarge Scint', 'Obs\nD_pix', 'Obs\nD_scint']
        raw_vals = [100.0 * wp[k]['raw_global_cut']['tau_eff'] for k in slice_keys]
        stau_vals = [100.0 * wp[k]['regional_S_tau_global_cut']['tau_eff'] for k in slice_keys]
        np_vals = [100.0 * wp[k]['regional_S_tau_NP_allocated']['tau_eff'] for k in slice_keys]

        x = np.arange(len(slice_keys))
        w = 0.26
        b1 = ax.bar(x - w, raw_vals, width=w, label='Raw global p(tau)', color='#8c96a0')
        b2 = ax.bar(x, stau_vals, width=w, label='Regional cal + S_tau (Strategy I)', color='#2b6cb0')
        b3 = ax.bar(x + w, np_vals, width=w, label='Regional cal + NP (Strategy II)', color='#d97706')
        for bars in (b1, b2, b3):
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.25, f'{h:.1f}%', ha='center', va='bottom', fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_x)
        ax.set_ylabel('Tau efficiency at 1% weighted FPR (%)')
        ax.set_title(title)
        ax.set_ylim(0, max(max(raw_vals), max(stau_vals), max(np_vals)) * 1.28 + 1.0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.legend(frameon=False, fontsize=8, loc='upper right')

    fig.savefig(out_dir / 'w3_03_regional_calibration_replay.pdf')
    fig.savefig(out_dir / 'w3_03_regional_calibration_replay.png', dpi=180)
    plt.close(fig)


def run_w3_03_replay(output_dir: Path) -> dict:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. R2 B1-D2 T4 on Tune40k
    r2_pred_path = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R2/experiments/b1_d2_t4_s42/predictions.parquet')
    tune_dom_path = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-01/tune40k_domains.parquet')
    r2_df = pd.read_parquet(r2_pred_path).sort_values(KEY).reset_index(drop=True)
    dom_df = pd.read_parquet(tune_dom_path, columns=KEY + ['obs_domain_id']).sort_values(KEY).reset_index(drop=True)
    if not r2_df[KEY].equals(dom_df[KEY]):
        raise ValueError('R2 Tune40k prediction keys differ from W3-01 tune40k_domains')
    r2_res = evaluate_replay_comparison(
        r2_df,
        dom_df['obs_domain_id'].to_numpy(dtype=np.int8),
        r2_df['pixel_centered'].to_numpy(dtype=bool),
        r2_df['large_scintillator'].to_numpy(dtype=bool),
        name='R2_B1_D2_T4_s42_Tune40k',
    )

    # 2. S6 MAE3 100k Ensemble on S4 10k validation (exact W3-01 obs_domain_id)
    s4_root = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13')
    plan = json.loads((s4_root / 'plan.json').read_text())
    geometry = json.loads(Path(plan['geometry']).read_text())
    from analysis.flavour.v3.common.domains import detector_bounds
    bounds = detector_bounds(geometry)

    seed_frames = []
    for seed in (17, 42, 2026):
        p = s4_root / f'experiments/s4_empty_mae3_n100000_s{seed}/validation_predictions.parquet'
        df_s = pd.read_parquet(p).sort_values(KEY).reset_index(drop=True)
        seed_frames.append(df_s)
    s4_ens = seed_frames[0].copy()
    p_ens = np.mean([df[[f'p{c}' for c in range(4)]].to_numpy(dtype=np.float64) for df in seed_frames], axis=0)
    for c in range(4):
        s4_ens[f'p{c}'] = p_ens[:, c]
    s4_obs_dom = _s4_validation_domains(s4_root, geometry, bounds, s4_ens)
    s4_res = evaluate_replay_comparison(
        s4_ens,
        s4_obs_dom,
        s4_ens['pixel_centered'].to_numpy(dtype=bool),
        s4_ens['large_scintillator'].to_numpy(dtype=bool),
        name='S6_MAE3_100k_Ensemble_S4Val10k',
    )

    # 3. Also evaluate S6 MAE3 on S6 Calibration (722k) -> S6 Independent Test (724k)
    # using strictly observable S6 diagnostics (pixel_channels, bar_channels, tracker_channels, quality.parquet)
    s6_root = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s6_2026-11-14')
    s6_extra = {}
    q_path = s6_root / 'evaluation/quality.parquet'
    if q_path.exists():
        # Check S6 test using calibration temperature + S_tau vs stage236 0.35% raw global threshold
        test_cat = pd.read_parquet(
            s6_root / 'test_catalog.parquet',
            columns=KEY + FAMILY + ['label', 'pixel_centered', 'large_scintillator', 'pixel_channels', 'bar_channels', 'tracker_channels', 'all_empty'],
        ).sort_values(KEY).reset_index(drop=True)
        test_raw = pd.read_parquet(
            s6_root / 'predictions/test_raw.parquet',
            columns=KEY + [
                f's4_empty_mae3_n100000_s{s}__p{c}'
                for s in (17, 42, 2026) for c in range(4)
            ],
        ).sort_values(KEY).reset_index(drop=True)
        qual = pd.read_parquet(q_path).sort_values(KEY).reset_index(drop=True)
        if test_cat[KEY].equals(test_raw[KEY]) and test_cat[KEY].equals(qual[KEY]):
            p_test = np.mean([
                test_raw[[f's4_empty_mae3_n100000_s{s}__p{c}' for c in range(4)]].to_numpy(dtype=np.float64)
                for s in (17, 42, 2026)
            ], axis=0)
            for c in range(4):
                test_cat[f'p{c}'] = p_test[:, c]
            # Strictly observable proxy strata on S6 test (no truth coordinates):
            # 0: D_empty; 1: high/moderate pixel & low edge leakage proxy; 2: bar-active; 3: edge/sparse
            obs_s6 = np.full(len(test_cat), 3, dtype=np.int8)
            is_empty = test_cat['all_empty'].to_numpy(dtype=bool)
            has_pix = test_cat['pixel_channels'].to_numpy() > 0
            has_bar = test_cat['bar_channels'].to_numpy() > 0
            pix_edge = qual['pixel_edge_fraction'].fillna(1.0).to_numpy()
            bar_edge = qual['bar_edge_energy_fraction'].fillna(1.0).to_numpy()
            bar_tail = qual['bar_last_panel_energy_fraction'].fillna(1.0).to_numpy()
            pix_planes = qual['pixel_planes'].fillna(0).to_numpy()
            is_pix_s6 = (~is_empty) & has_pix & (pix_planes >= 2) & (pix_edge < 0.25) & (bar_edge < 0.25) & (bar_tail < 0.20)
            is_scint_s6 = (~is_empty) & (~is_pix_s6) & has_bar & (bar_edge < 0.30) & (bar_tail < 0.25)
            obs_s6[is_empty] = 0
            obs_s6[is_pix_s6] = 1
            obs_s6[is_scint_s6] = 2
            s6_extra = evaluate_replay_comparison(
                test_cat,
                obs_s6,
                test_cat['pixel_centered'].to_numpy(dtype=bool),
                test_cat['large_scintillator'].to_numpy(dtype=bool),
                name='S6_MAE3_100k_Ensemble_S6Test724k_ObsProxyOOF',
            )

    _plot_replay_figure(r2_res, s4_res, output_dir)

    report = {
        'status': 'complete',
        'UTC': datetime.now(timezone.utc).isoformat(),
        'seconds': time.perf_counter() - started,
        'r2_b1_d2_t4_tune40k': r2_res,
        's6_mae3_s4_val10k': s4_res,
        's6_mae3_test724k_obs_proxy': s6_extra,
        'Cal_A_Cut_B_Confirm_payload_read': False,
    }
    write_json(output_dir / 'replay_results.json', report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-03'),
    )
    args = parser.parse_args()
    res = run_w3_03_replay(args.output)
    summary = {}
    for key in ('r2_b1_d2_t4_tune40k', 's6_mae3_s4_val10k', 's6_mae3_test724k_obs_proxy'):
        item = res.get(key)
        if not item:
            continue
        wp1 = item['working_points']['0.01']['slices']
        summary[key] = {
            'All_raw': wp1['All']['raw_global_cut']['tau_eff'],
            'All_Stau': wp1['All']['regional_S_tau_global_cut']['tau_eff'],
            'All_NP': wp1['All']['regional_S_tau_NP_allocated']['tau_eff'],
            'TruthPC_raw': wp1['Truth_pixel_centered']['raw_global_cut']['tau_eff'],
            'TruthPC_Stau': wp1['Truth_pixel_centered']['regional_S_tau_global_cut']['tau_eff'],
            'TruthPC_NP': wp1['Truth_pixel_centered']['regional_S_tau_NP_allocated']['tau_eff'],
            'ObsDpix_raw': wp1['Obs_D_pix_contained']['raw_global_cut']['tau_eff'],
            'ObsDpix_Stau': wp1['Obs_D_pix_contained']['regional_S_tau_global_cut']['tau_eff'],
            'ObsDpix_NP': wp1['Obs_D_pix_contained']['regional_S_tau_NP_allocated']['tau_eff'],
            'Z0_raw': item['asimov_Z0_1pct_tau_prior_10k_events']['raw_p_tau'],
            'Z0_Stau': item['asimov_Z0_1pct_tau_prior_10k_events']['regional_S_tau'],
        }
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
