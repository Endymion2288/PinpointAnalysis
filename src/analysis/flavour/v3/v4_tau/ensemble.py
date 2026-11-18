"""Strictly out-of-fold 5-fold cross-fitted dual-regime ensemble (PinpointFormer x B1++)."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax

from analysis.flavour.inventory import COMPRESSION
from analysis.flavour.v2.partitions import FAMILY, KEY
from analysis.flavour.v2.r1.metrics import Metrics, three_from_four, three_targets
from analysis.flavour.v2.r2.readout import grouped_folds
from analysis.flavour.v3.common.calibration import (
    _tau_efficiency_at_fpr,
    multi_domain_asimov_z0,
    oof_regional_calibration,
)
from analysis.flavour.v3.common.domains import OBS_DOMAIN_NAMES


def extract_member_arrays(frame: pd.DataFrame, ref_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Align a member prediction Parquet onto ref_df KEY order and extract probs4, log_probs4, charm."""
    pos = pd.MultiIndex.from_frame(frame[KEY]).get_indexer(pd.MultiIndex.from_frame(ref_df[KEY]))
    if (pos < 0).any():
        raise ValueError('prediction parquet missing reference keys')
    aligned = frame.iloc[pos].reset_index(drop=True)
    p_cols = ['p_0', 'p_1', 'p_2', 'p_3'] if 'p_0' in aligned.columns else ['p0', 'p1', 'p2', 'p3']
    probs = aligned[p_cols].to_numpy(dtype=np.float64)
    probs = probs / np.maximum(probs.sum(axis=-1, keepdims=True), 1e-12)
    log_p = np.log(np.clip(probs, 1e-8, 1.0))
    log_p = log_p - log_p.mean(axis=-1, keepdims=True)
    charm = (
        np.clip(aligned['charm_prob'].to_numpy(dtype=np.float64), 0.0, 1.0)
        if 'charm_prob' in aligned.columns
        else None
    )
    return {'probs': probs, 'log_p': log_p, 'charm': charm}


def _fit_domain_convex_blend(log_p_stack: np.ndarray, labels: np.ndarray) -> dict:
    """Fit convex member weights, domain temperature, and zero-sum class bias minimizing balanced NLL."""
    # log_p_stack: (M, N, 4)
    m, n, _ = log_p_stack.shape
    if n < 12:
        w = np.full(m, 1.0 / m, dtype=np.float64)
        return {'weights': w.tolist(), 'inv_T': 1.0, 'T': 1.0, 'bias': [0.0, 0.0, 0.0, 0.0], 'fallback': True}

    support = np.bincount(labels, minlength=4)
    present = support > 0
    n_pres = int(present.sum())
    if n_pres < 2:
        w = np.full(m, 1.0 / m, dtype=np.float64)
        return {'weights': w.tolist(), 'inv_T': 1.0, 'T': 1.0, 'bias': [0.0, 0.0, 0.0, 0.0], 'fallback': True}

    sample_w = np.zeros(n, dtype=np.float64)
    for c in range(4):
        if support[c] > 0:
            sample_w[labels == c] = 1.0 / (n_pres * support[c])
    row_idx = np.arange(n)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        alpha = theta[:m]
        alpha_c = alpha - np.max(alpha)
        exp_a = np.exp(alpha_c)
        w = exp_a / np.sum(exp_a)
        log_inv_t = float(theta[m])
        inv_t = float(np.exp(log_inv_t))
        bias = np.array([theta[m + 1], theta[m + 2], theta[m + 3], 0.0], dtype=np.float64)

        mix_lp = np.tensordot(w, log_p_stack, axes=(0, 0))  # (N, 4)
        scaled = inv_t * mix_lp + bias[None, :]
        max_s = np.max(scaled, axis=1, keepdims=True)
        exp_s = np.exp(scaled - max_s)
        sum_s = np.sum(exp_s, axis=1, keepdims=True)
        probs = exp_s / sum_s
        lse = np.squeeze(max_s, axis=1) + np.log(np.squeeze(sum_s, axis=1))

        true_scaled = scaled[row_idx, labels]
        nll = float(np.dot(sample_w, lse - true_scaled))
        reg = 0.5 * (1e-3 * float(np.dot(bias[:3], bias[:3])) + 1e-4 * (log_inv_t ** 2) + 1e-4 * float(np.dot(alpha_c, alpha_c)))

        resid = probs.copy()
        resid[row_idx, labels] -= 1.0
        w_resid = sample_w[:, None] * resid  # (N, 4)

        g_mix = inv_t * w_resid  # (N, 4)
        g_w = np.sum(log_p_stack * g_mix[None, :, :], axis=(1, 2))
        g_alpha = w * (g_w - float(np.dot(w, g_w))) + 1e-4 * (alpha_c - float(np.mean(alpha_c)))

        g_log_inv_t = float(inv_t * np.sum(w_resid * mix_lp)) + 1e-4 * log_inv_t
        g_bias = np.sum(w_resid, axis=0)[:3] + 1e-3 * bias[:3]
        grad = np.concatenate([g_alpha, [g_log_inv_t], g_bias])
        return nll + reg, grad

    theta0 = np.zeros(m + 4, dtype=np.float64)
    bounds = [(-6.0, 6.0)] * m + [(float(np.log(0.25)), float(np.log(4.0)))] + [(-2.5, 2.5)] * 3
    res = minimize(
        objective,
        theta0,
        jac=True,
        method='L-BFGS-B',
        bounds=bounds,
        options={'gtol': 1e-7, 'ftol': 1e-12, 'maxiter': 600},
    )
    if not res.success:
        rng = np.random.default_rng(29)
        for _ in range(4):
            cand = minimize(
                objective,
                rng.normal(0.0, 0.15, size=m + 4),
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'gtol': 1e-7, 'ftol': 1e-14, 'maxiter': 1200},
            )
            if cand.fun < res.fun or (cand.success and not res.success):
                res = cand
            if res.success:
                break
    alpha = res.x[:m]
    w = softmax(alpha)
    inv_t = float(np.exp(res.x[m]))
    bias = [float(res.x[m + 1]), float(res.x[m + 2]), float(res.x[m + 3]), 0.0]
    grad_inf = float(np.max(np.abs(res.jac))) if hasattr(res, 'jac') and res.jac is not None else float('nan')
    return {
        'weights': w.tolist(),
        'inv_T': inv_t,
        'T': float(1.0 / inv_t),
        'bias': bias,
        'fallback': False,
        'success': bool(res.success),
        'optimizer_converged': bool(res.success or grad_inf < 1e-5),
        'grad_inf_norm': grad_inf,
        'objective': float(res.fun),
    }


def _apply_domain_convex_blend(log_p_stack: np.ndarray, params: dict) -> np.ndarray:
    w = np.asarray(params['weights'], dtype=np.float64)
    mix_lp = np.tensordot(w, log_p_stack, axes=(0, 0))
    scaled = float(params['inv_T']) * mix_lp + np.asarray(params['bias'], dtype=np.float64)[None, :]
    return softmax(scaled, axis=1)


def _oof_neyman_pearson_rates(
    s_tau: np.ndarray,
    labels: np.ndarray,
    obs_domain_id: np.ndarray,
    folds: np.ndarray,
    n_folds: int,
    target_fpr: float,
    masks: dict[str, np.ndarray],
) -> dict[str, dict]:
    """Strictly OOF Strategy-II Neyman-Pearson domain FPR budget allocation."""
    cnt = np.bincount(labels, minlength=4)
    w_all = (len(labels) / (4.0 * np.maximum(cnt, 1)))[labels]
    sel = np.zeros(len(labels), dtype=bool)
    ratios_grid = [(1.0, 1.0, 1.0), (2.0, 1.0, 0.5), (3.0, 1.0, 0.3), (4.0, 0.8, 0.2)]
    for fold in range(n_folds):
        held = folds == fold
        fit = ~held
        fit_tot_bkg = float(w_all[fit & (labels != 2)].sum())
        best_eff = -1.0
        best_cuts = None
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
            if fit_tau_sel > best_eff:
                best_eff = fit_tau_sel
                best_cuts = cuts
        if best_cuts is not None:
            for d in (1, 2, 3):
                hm = held & (obs_domain_id == d)
                sel[hm] = s_tau[hm] >= best_cuts[d]

    out = {}
    for sname, smask in masks.items():
        m = np.asarray(smask, dtype=bool)
        itau = m & (labels == 2)
        ibkg = m & (labels != 2)
        out[sname] = {
            'tau_eff': float((sel & itau).sum() / max(1, int(itau.sum()))) if itau.any() else None,
            'actual_fpr': float((sel & ibkg).sum() / max(1, int(ibkg.sum()))) if ibkg.any() else None,
            'tau_passed': int((sel & itau).sum()),
            'tau_total': int(itau.sum()),
            'bkg_passed': int((sel & ibkg).sum()),
            'bkg_total': int(ibkg.sum()),
        }
    return out


def _oof_multidomain_asimov_z0(
    s_tau: np.ndarray,
    labels: np.ndarray,
    obs_domain_id: np.ndarray,
    folds: np.ndarray,
    n_folds: int = 5,
    n_bins_per_domain: int = 6,
    tau_prior: float = 0.01,
    benchmark_total: float = 10000.0,
) -> float:
    """Strictly OOF 18-bin (6 bins x 3 active domains) Asimov Z0 where bin edges come from fit folds."""
    is_tau = labels == 2
    n_tau_tot = max(1, int(is_tau.sum()))
    n_bkg_tot = max(1, int((~is_tau).sum()))
    s_scale = (benchmark_total * tau_prior) / n_tau_tot
    b_scale = (benchmark_total * (1.0 - tau_prior)) / n_bkg_tot

    bin_ids = np.full(len(labels), -1, dtype=np.int64)
    for fold in range(n_folds):
        held = folds == fold
        fit = ~held
        for d in (1, 2, 3):
            dm_fit = fit & (obs_domain_id == d)
            dm_held = held & (obs_domain_id == d)
            if not dm_fit.any() or not dm_held.any():
                continue
            edges = np.quantile(s_tau[dm_fit], np.linspace(0.0, 1.0, n_bins_per_domain + 1))
            edges[0] = -np.inf
            edges[-1] = np.inf
            b_idx = np.clip(np.searchsorted(edges[1:-1], s_tau[dm_held], side='right'), 0, n_bins_per_domain - 1)
            bin_ids[dm_held] = d * n_bins_per_domain + b_idx

    z0_sq = 0.0
    for d in (1, 2, 3):
        for k in range(n_bins_per_domain):
            bm = bin_ids == (d * n_bins_per_domain + k)
            if not bm.any():
                continue
            s = float((bm & is_tau).sum()) * s_scale
            b = float((bm & (~is_tau)).sum()) * b_scale
            if s > 0.0 and b > 0.0:
                z0_sq += 2.0 * ((s + b) * np.log(1.0 + s / b) - s)
    return float(np.sqrt(max(z0_sq, 0.0)))


def fit_oof_domain_ensemble(
    member_frames: dict[str, pd.DataFrame],
    ref_df: pd.DataFrame,
    slices: dict[str, np.ndarray],
    n_folds: int = 5,
    folds: np.ndarray | None = None,
    lambda_grid: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0),
    save_path: Path | None = None,
) -> dict:
    """Fit 5-fold family-grouped OOF convex domain ensemble + charm penalty + NP allocation."""
    member_names = list(member_frames.keys())
    if not member_names:
        raise ValueError('member_frames must be non-empty')

    aligned = {name: extract_member_arrays(df, ref_df) for name, df in member_frames.items()}
    log_p_stack = np.stack([aligned[n]['log_p'] for n in member_names], axis=0)  # (M, N, 4)
    charm_list = [aligned[n]['charm'] for n in member_names if aligned[n]['charm'] is not None]
    mean_charm = np.mean(charm_list, axis=0) if charm_list else None

    labels = ref_df['label'].to_numpy(dtype=np.int64)
    obs_dom = ref_df['obs_domain_id'].to_numpy(dtype=np.int64)
    if folds is None:
        folds, fold_summary = grouped_folds(
            ref_df[KEY + FAMILY + ['label']].reset_index(drop=True), folds=n_folds, seed=20261117
        )
    else:
        folds = np.asarray(folds, dtype=np.int64)
        fold_summary = {'folds': int(n_folds), 'precomputed': True}

    oof_probs = np.full((len(labels), 4), 0.25, dtype=np.float64)
    fold_records = []
    for k in range(n_folds):
        held = folds == k
        fit = ~held
        rec = {'fold': int(k), 'domains': {}}
        for d in (1, 2, 3):
            fm = fit & (obs_dom == d)
            hm = held & (obs_dom == d)
            params = _fit_domain_convex_blend(log_p_stack[:, fm, :], labels[fm])
            rec['domains'][OBS_DOMAIN_NAMES[d]] = params
            if hm.any():
                oof_probs[hm] = _apply_domain_convex_blend(log_p_stack[:, hm, :], params)
        fold_records.append(rec)

    # Fit per-domain OOF lambda_charm (purely cross-fitted on fit folds)
    s_tau = np.full(len(labels), -25.0, dtype=np.float64)
    chosen_lambdas: dict[str, list[float]] = {OBS_DOMAIN_NAMES[d]: [] for d in (1, 2, 3)}
    for k in range(n_folds):
        held = folds == k
        fit = ~held
        for d in (1, 2, 3):
            fm = fit & (obs_dom == d)
            hm = held & (obs_dom == d)
            p_tau_f = np.clip(oof_probs[fm, 2], 1e-12, 1.0)
            p_bkg_f = np.clip(1.0 - p_tau_f, 1e-12, 1.0)
            p_tau_h = np.clip(oof_probs[hm, 2], 1e-12, 1.0)
            p_bkg_h = np.clip(1.0 - p_tau_h, 1e-12, 1.0)

            best_lam = 0.0
            if mean_charm is not None and (labels[fm] == 2).any() and (labels[fm] != 2).any():
                best_obj = -1.0
                cf = mean_charm[fm]
                for lam in lambda_grid:
                    sf = np.log(p_tau_f / (p_bkg_f + lam * cf + 1e-12))
                    e1 = _tau_efficiency_at_fpr(sf, labels[fm], 0.01)['tau_eff'] or 0.0
                    e01 = _tau_efficiency_at_fpr(sf, labels[fm], 0.001)['tau_eff'] or 0.0
                    obj = e1 + 0.5 * e01
                    if obj > best_obj + 1e-9:
                        best_obj = obj
                        best_lam = float(lam)
            chosen_lambdas[OBS_DOMAIN_NAMES[d]].append(best_lam)
            if hm.any():
                ch = mean_charm[hm] if mean_charm is not None else 0.0
                s_tau[hm] = np.log(p_tau_h / (p_bkg_h + best_lam * ch + 1e-12))

    # Build raw S_tau = logit(p_tau) as well
    p_tau_raw = np.clip(oof_probs[:, 2], 1e-8, 1.0 - 1e-8)
    s_raw = np.where(obs_dom == 0, -25.0, np.log(p_tau_raw / (1.0 - p_tau_raw)))

    # Mean learned domain weights across folds
    mean_weights = {}
    for d in (1, 2, 3):
        dn = OBS_DOMAIN_NAMES[d]
        w_mat = np.array([fr['domains'][dn]['weights'] for fr in fold_records], dtype=np.float64)
        mean_weights[dn] = {name: float(w_mat[:, i].mean()) for i, name in enumerate(member_names)}

    # Evaluate across all slices using both Strategy I (global cut on s_tau & s_raw) and Strategy II (NP)
    slice_out = {}
    np_1p = _oof_neyman_pearson_rates(s_tau, labels, obs_dom, folds, n_folds, 0.01, slices)
    np_01p = _oof_neyman_pearson_rates(s_tau, labels, obs_dom, folds, n_folds, 0.001, slices)

    cnt_all = np.bincount(labels, minlength=4)
    w_all = (len(labels) / (4.0 * np.maximum(cnt_all, 1)))[labels]
    cut_s_1p = float(_tau_efficiency_at_fpr(s_tau, labels, 0.01, w_all)['threshold'])
    cut_s_01p = float(_tau_efficiency_at_fpr(s_tau, labels, 0.001, w_all)['threshold'])

    for sname, smask in slices.items():
        m = np.asarray(smask, dtype=bool)
        ym = labels[m]
        pm = oof_probs[m]
        rep = Metrics(ym, pm).report() if (np.bincount(ym, minlength=4) > 0).all() else {'BA': float('nan')}
        nt = ym != 2
        ba3 = (
            float(Metrics(three_targets(ym[nt]), three_from_four(pm[nt])).report()['BA'])
            if nt.any() and len(np.unique(ym[nt])) == 3
            else float('nan')
        )
        itau = m & (labels == 2)
        ibkg = m & (labels != 2)
        row = {
            'count': int(m.sum()),
            'nll': float(-np.log(np.clip(pm[np.arange(len(ym)), ym], 1e-8, 1.0)).mean()),
            'BA4': float(rep['BA']),
            'cond_BA3': ba3,
            'np_1pct': np_1p[sname],
            'np_0p1pct': np_01p[sname],
            'global_S_tau_at_1pct_fpr': {
                'tau_eff': float(((s_tau >= cut_s_1p) & itau).sum() / max(1, int(itau.sum()))) if itau.any() else None,
                'actual_fpr': float(((s_tau >= cut_s_1p) & ibkg).sum() / max(1, int(ibkg.sum()))) if ibkg.any() else None,
            },
            'global_S_tau_at_0p1pct_fpr': {
                'tau_eff': float(((s_tau >= cut_s_01p) & itau).sum() / max(1, int(itau.sum()))) if itau.any() else None,
                'actual_fpr': float(((s_tau >= cut_s_01p) & ibkg).sum() / max(1, int(ibkg.sum()))) if ibkg.any() else None,
            },
        }
        if (labels[m] == 2).any() and (labels[m] != 2).any():
            row['within_slice_p_tau_1pct'] = float(_tau_efficiency_at_fpr(s_raw[m], ym, 0.01)['tau_eff'])
            row['within_slice_p_tau_0p1pct'] = float(_tau_efficiency_at_fpr(s_raw[m], ym, 0.001)['tau_eff'])
            row['within_slice_S_tau_1pct'] = float(_tau_efficiency_at_fpr(s_tau[m], ym, 0.01)['tau_eff'])
            row['within_slice_S_tau_0p1pct'] = float(_tau_efficiency_at_fpr(s_tau[m], ym, 0.001)['tau_eff'])
        slice_out[sname] = row

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df_out = ref_df[KEY + FAMILY + ['label', 'obs_domain_id']].copy().reset_index(drop=True)
        for c in range(4):
            df_out[f'p_{c}'] = oof_probs[:, c].astype(np.float32)
        df_out['s_tau'] = s_tau.astype(np.float32)
        df_out.to_parquet(save_path, index=False, compression=COMPRESSION)

    return {
        'members': member_names,
        'folds': fold_summary,
        'mean_domain_weights': mean_weights,
        'mean_lambda_charm': {dn: float(np.mean(vs)) for dn, vs in chosen_lambdas.items()},
        'asimov_z0_single_stream': float(multi_domain_asimov_z0(s_tau, labels, obs_dom, n_bins=10)),
        'asimov_z0_oof_18bin_multidomain': _oof_multidomain_asimov_z0(s_tau, labels, obs_dom, folds, n_folds=n_folds),
        'slices': slice_out,
        'oof_probs': oof_probs,
        's_tau': s_tau,
        's_raw': s_raw,
    }


def calibrate_single_member_oof(
    df: pd.DataFrame,
    ref_df: pd.DataFrame,
    slices: dict[str, np.ndarray],
    n_folds: int = 5,
) -> dict:
    """Compute consistent Strategy I / Strategy II / OOF Asimov Z0 for any single member Parquet."""
    arr = extract_member_arrays(df, ref_df)
    labels = ref_df['label'].to_numpy(dtype=np.int64)
    obs_dom = ref_df['obs_domain_id'].to_numpy(dtype=np.int64)
    folds, _ = grouped_folds(ref_df[KEY + FAMILY + ['label']].reset_index(drop=True), folds=n_folds, seed=20261117)
    cal_probs, s_tau, _ = oof_regional_calibration(
        arr['log_p'], labels, obs_dom, folds, n_folds=n_folds, charm_probs=arr['charm'], lambda_charm=0.0
    )
    cnt_all = np.bincount(labels, minlength=4)
    w_all = (len(labels) / (4.0 * np.maximum(cnt_all, 1)))[labels]
    cut_s_1p = float(_tau_efficiency_at_fpr(s_tau, labels, 0.01, w_all)['threshold'])
    cut_s_01p = float(_tau_efficiency_at_fpr(s_tau, labels, 0.001, w_all)['threshold'])
    np_1p = _oof_neyman_pearson_rates(s_tau, labels, obs_dom, folds, n_folds, 0.01, slices)
    np_01p = _oof_neyman_pearson_rates(s_tau, labels, obs_dom, folds, n_folds, 0.001, slices)
    return {
        'oof_calibrated_probs': cal_probs,
        's_tau': s_tau,
        'asimov_z0_single_stream': float(multi_domain_asimov_z0(s_tau, labels, obs_dom, n_bins=10)),
        'asimov_z0_oof_18bin_multidomain': _oof_multidomain_asimov_z0(s_tau, labels, obs_dom, folds, n_folds=n_folds),
        'global_S_tau_cut_1pct': cut_s_1p,
        'global_S_tau_cut_0p1pct': cut_s_01p,
        'np_1pct': np_1p,
        'np_0p1pct': np_01p,
    }
