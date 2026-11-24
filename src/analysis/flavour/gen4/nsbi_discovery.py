"""POT-GPD semi-parametric tail calibrator and Golden Micro-Channel unbinned/fine-binned NSBI-PL engine.

Shared statistical discovery engine for Pinpoint-Gen4:
  1. `POT-GPD` calibrator: Harrell-Davis 95% Beta order-statistic anchor + shape-constrained
     Generalized Pareto Distribution (`xi <= 0.20`) tail extrapolation with optional NP-shift.
  2. Purely-observable router into 4 Golden Discovery Micro-Channels:
       - `Golden-3Prong-ZeroMu` (C1)
       - `Golden-Kink-Mu`       (C2)
       - `Golden-Rho-1Prong`    (C3)
       - `Fortune-HighMT-NSBI`  (C4)
     plus low-energy ambiguity abstention (`Abstain-LowE`) and 3 high-statistics in-situ
     Control Regions (`CR-mumu`, `CR-1mu`, `CR-pi0`) locking charm, numu flux, and NC scales.
  3. Joint Profiled Likelihood (`NSBI-PL`) significance calculator supporting both:
       - Fine-binned discovery micro-channels + CRs with closed-form Barlow-Beeston lite (`BB-lite`);
       - Continuous unbinned event-level density-ratio likelihood inside discovery channels + CRs;
       - Direct apples-to-apples comparison against single-bin `1% FPR` counting under `MW2t` mass
         weighting and realistic forward flux (`f_tau = 0.3%`, `delta_sys`).
  4. Frequentist H0 / H1 Toy-MC closure verification.

Strict Seal & Anti-Leakage Compliance:
  All routing and threshold calibration consume ONLY pure detector observables (`OBSERVABLE_SUMMARY_KEYS`).
  Truth labels (`label`, `has_charm`, `geom_weight`) are used exclusively inside template/Asimov/Toy-MC
  evaluation functions and never inside observable routing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import betainc
from scipy.stats import norm

from analysis.flavour.v3.v5_confirm.flux_significance import (
    FIDUCIAL_2T_BKG_SHARES,
    _barlow_beeston_inner_deviance,
)


CHANNEL_NAMES: tuple[str, ...] = (
    'Golden-3Prong-ZeroMu',   # 0: C1
    'Golden-Kink-Mu',         # 1: C2
    'Golden-Rho-1Prong',      # 2: C3
    'Fortune-HighMT-NSBI',    # 3: C4
    'Abstain-LowE',           # 4: C5 low-energy ambiguous abstention
    'CR-mumu',                # 5: In-situ charm rate control region
    'CR-1mu',                 # 6: In-situ numu flux/norm control region
    'CR-pi0',                 # 7: In-situ NC pi0/scale control region
    'Abstain-Bulk',           # 8: Remaining ambiguous/empty bulk
)
DISCOVERY_CHANNEL_IDS: tuple[int, ...] = (0, 1, 2, 3)
CR_CHANNEL_IDS: tuple[int, ...] = (5, 6, 7)

OBSERVABLE_SUMMARY_KEYS: tuple[str, ...] = (
    'obs_domain_id',
    'e_vis_gev',
    'pt_miss_gev',
    'mt_gev',
    'tau_llr',
    'charm_prob',
    'n_sv_prongs',
    'sv_lxy_sig',
    'sv_mass_gev',
    'n_pen_muons',
    'kink_llr',
    'dphi_mu_had',
    'ip_sig_max',
    'pi0_em_frac',
    'rho_vis_mass_gev',
    'has_prompt_e',
    'has_front_pixel',
    'has_target_pixel',
    # [gen4-contract] section 5.2 nsbi_obs alias keys
    'sl_xy',
    'm_sv_corr_gev',
    'n_mu_veto',
    'q_mu1',
    'q_mu2',
    'lambda_kink',
    'dphi_mu_ht',
    'm_vis_rho_gev',
    'e_vis_hat_gev',
    'pt_miss_hat_gev',
    'm_gg_pi0_gev',
)


# ===========================================================================
# 1. POT-GPD Semi-Parametric Extreme-Value Tail Calibrator
# ===========================================================================
def harrell_davis_quantile(
    values: np.ndarray,
    q: float,
    weights: np.ndarray | None = None,
) -> float:
    """Compute the Harrell-Davis Beta order-statistic quantile estimator at probability `q` in (0, 1)."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or len(x) == 0:
        raise ValueError('harrell_davis_quantile requires a non-empty 1D array')
    if len(x) == 1:
        return float(x[0])
    q_c = float(np.clip(q, 1e-6, 1.0 - 1e-6))

    order = np.argsort(x, kind='mergesort')
    xs = x[order]
    n = len(xs)

    if weights is None:
        u = np.linspace(0.0, 1.0, n + 1, dtype=np.float64)
        n_eff = float(n)
    else:
        w = np.maximum(np.asarray(weights, dtype=np.float64)[order], 0.0)
        w_sum = float(w.sum())
        if w_sum <= 0.0:
            raise ValueError('Weights must have positive sum in harrell_davis_quantile')
        w_norm = w / w_sum
        u = np.concatenate([[0.0], np.cumsum(w_norm)])
        u[-1] = 1.0
        n_eff = float((w_sum * w_sum) / max(float(np.sum(w * w)), 1e-30))

    a = q_c * (n_eff + 1.0)
    b = (1.0 - q_c) * (n_eff + 1.0)
    cdf_vals = betainc(a, b, u)
    hd_w = np.maximum(cdf_vals[1:] - cdf_vals[:-1], 0.0)
    s_w = float(hd_w.sum())
    if s_w <= 0.0:
        return float(np.quantile(xs, q_c))
    return float(np.dot(hd_w / s_w, xs))


@dataclass
class POTGPDFitResult:
    """Fitted POT-GPD extreme-value tail calibration parameters."""

    anchor_q: float
    t0: float
    p_exceed: float
    n_total: int
    n_exceed: int
    xi: float
    beta: float
    std_xi: float
    std_beta: float
    cov_xi_beta: float
    np_shift_z: float
    converged: bool
    fallback_empirical: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_pot_gpd(
    scores_bg: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    anchor_q: float = 0.95,
    xi_bounds: tuple[float, float] = (-0.35, 0.20),
    xi_prior_std: float = 0.05,
    np_shift_z: float = 0.0,
    min_exceedances: int = 12,
    target_min_exceedances: int = 200,
) -> POTGPDFitResult:
    """Fit Harrell-Davis anchor + Coles-Dixon penalized Generalized Pareto tail GPD(xi, beta)."""
    s = np.asarray(scores_bg, dtype=np.float64)
    finite_m = np.isfinite(s)
    if not finite_m.any():
        raise ValueError('No finite background scores in fit_pot_gpd')
    s = s[finite_m]
    w = None if weights is None else np.maximum(np.asarray(weights, dtype=np.float64)[finite_m], 0.0)

    n_tot = int(len(s))
    q_eff = float(anchor_q)
    if target_min_exceedances > 0 and n_tot * (1.0 - q_eff) < float(target_min_exceedances) and n_tot >= 800:
        q_eff = max(0.88, 1.0 - float(target_min_exceedances) / float(n_tot))

    t0 = harrell_davis_quantile(s, q_eff, weights=w)
    exc_mask = s > t0
    n_exc = int(exc_mask.sum())

    if n_exc < min_exceedances:
        # Step anchor slightly down if extreme ties/discreteness leave too few exceedances
        q_alt = max(0.80, 1.0 - max(min_exceedances, int(0.05 * n_tot)) / max(n_tot, 1))
        t0 = harrell_davis_quantile(s, q_alt, weights=w)
        exc_mask = s > t0
        n_exc = int(exc_mask.sum())

    if n_exc < 5:
        spread = float(np.std(s)) if n_tot > 1 else 1.0
        return POTGPDFitResult(
            anchor_q=float(anchor_q),
            t0=float(t0),
            p_exceed=max(1.0 / max(n_tot, 1), 1.0 - float(anchor_q)),
            n_total=n_tot,
            n_exceed=n_exc,
            xi=0.0,
            beta=max(spread * 0.25, 1e-4),
            std_xi=0.0,
            std_beta=0.0,
            cov_xi_beta=0.0,
            np_shift_z=float(np_shift_z),
            converged=False,
            fallback_empirical=True,
        )

    u = s[exc_mask] - t0
    if w is not None:
        w_tot = float(w.sum())
        w_exc = w[exc_mask]
        p_exc = float(w_exc.sum() / max(w_tot, 1e-30))
        w_norm = w_exc / max(float(w_exc.mean()), 1e-30)
    else:
        p_exc = float(n_exc / max(n_tot, 1))
        w_norm = np.ones(n_exc, dtype=np.float64)

    u_max = float(np.max(u))
    mean_u = float(np.average(u, weights=w_norm))
    var_u = float(np.average((u - mean_u) ** 2, weights=w_norm))
    mean_u = max(mean_u, 1e-6)
    var_u = max(var_u, 1e-10)

    # Method-of-moments initialization clamped inside bounds
    xi_mom = 0.5 * (1.0 - (mean_u * mean_u) / var_u)
    xi_init = float(np.clip(xi_mom, xi_bounds[0] + 0.02, xi_bounds[1] - 0.02))
    beta_init = max(mean_u * (1.0 - xi_init), 1e-4)
    if xi_init < 0.0:
        beta_init = max(beta_init, -xi_init * u_max * 1.08)

    def _nll(params: np.ndarray) -> float:
        xi_v = float(params[0])
        beta_v = float(math.exp(params[1]))
        if xi_v < xi_bounds[0] or xi_v > xi_bounds[1] or beta_v <= 1e-9:
            return 1e18
        # Coles-Dixon (1999) penalized likelihood prior on GPD shape xi to prevent small-sample variance explosion
        pen_xi = 0.5 * ((xi_v / max(float(xi_prior_std), 1e-3)) ** 2) if xi_prior_std > 0.0 else 0.0
        if abs(xi_v) < 1e-5:
            ll = -math.log(beta_v) - u / beta_v
            return float(-np.sum(w_norm * ll)) + pen_xi
        z = 1.0 + (xi_v / beta_v) * u
        if np.any(z <= 1e-9):
            # Smooth barrier penalty outside support
            min_z = float(np.min(z))
            return 1e12 + 1e10 * (1e-6 - min_z) ** 2
        ll = -math.log(beta_v) - (1.0 + 1.0 / xi_v) * np.log(z)
        return float(-np.sum(w_norm * ll)) + pen_xi

    x0 = np.array([xi_init, math.log(max(beta_init, 1e-6))], dtype=np.float64)
    bnds = [xi_bounds, (math.log(1e-6), math.log(max(u_max * 50.0, 100.0)))]
    res = minimize(_nll, x0=x0, method='L-BFGS-B', bounds=bnds, options={'ftol': 1e-12, 'gtol': 1e-8, 'maxiter': 300})
    if not res.success:
        res_alt = minimize(_nll, x0=np.array([0.0, math.log(mean_u)]), method='L-BFGS-B', bounds=bnds)
        if res_alt.fun < res.fun:
            res = res_alt

    xi_hat = float(np.clip(res.x[0], xi_bounds[0], xi_bounds[1]))
    beta_hat = float(math.exp(res.x[1]))
    if xi_hat < 0.0 and beta_hat <= -xi_hat * u_max:
        beta_hat = float(-xi_hat * u_max * 1.001)

    # Observed Fisher information in (xi, beta) coordinates for uncertainty / NP-shift
    n_eff_exc = float(np.sum(w_norm) ** 2 / max(float(np.sum(w_norm ** 2)), 1e-12))
    # Asymptotic GPD MLE Fisher matrix (valid for xi > -0.5):
    # I_11 = 2 / ((1 + xi)(1 + 2*xi)), I_12 = 1 / (beta*(1 + xi)*(1 + 2*xi)), I_22 = 1 / (beta^2 * (1 + 2*xi))
    denom_f = max(1.0 + 2.0 * xi_hat, 0.15)
    one_p_xi = max(1.0 + xi_hat, 0.25)
    # Fisher inverse for (xi, beta): cov = (1 + xi) / n_eff * [[1+xi, -beta], [-beta, 2*beta^2]]
    cov_11 = ((1.0 + xi_hat) ** 2) / max(n_eff_exc, 1.0)
    cov_22 = (2.0 * one_p_xi * beta_hat * beta_hat) / max(n_eff_exc, 1.0)
    cov_12 = (-one_p_xi * beta_hat) / max(n_eff_exc, 1.0)
    if xi_hat <= -0.45 or denom_f <= 0.15:
        cov_11 = max(cov_11, 0.01)

    return POTGPDFitResult(
        anchor_q=float(anchor_q),
        t0=float(t0),
        p_exceed=float(p_exc),
        n_total=n_tot,
        n_exceed=n_exc,
        xi=xi_hat,
        beta=beta_hat,
        std_xi=float(math.sqrt(max(cov_11, 0.0))),
        std_beta=float(math.sqrt(max(cov_22, 0.0))),
        cov_xi_beta=float(cov_12),
        np_shift_z=float(np_shift_z),
        converged=bool(res.success),
        fallback_empirical=False,
    )


def pot_gpd_threshold(
    fit: POTGPDFitResult,
    alpha_fpr: float,
    *,
    scores_bg_fallback: np.ndarray | None = None,
    use_np_shift: bool = True,
) -> float:
    """Evaluate analytical POT-GPD tail threshold at target background FPR `alpha_fpr`."""
    alpha = float(np.clip(alpha_fpr, 1e-7, 1.0 - 1e-6))
    if alpha >= fit.p_exceed:
        if scores_bg_fallback is not None and len(scores_bg_fallback) > 0:
            return harrell_davis_quantile(scores_bg_fallback, 1.0 - alpha)
        # Mild downward exponential continuation below t0
        ratio = max(alpha / max(fit.p_exceed, 1e-8), 1e-6)
        return float(fit.t0 - fit.beta * math.log(ratio))

    r = max(fit.p_exceed / alpha, 1.0)
    xi = fit.xi
    beta = fit.beta
    if abs(xi) < 1e-5:
        ln_r = math.log(r)
        thr = fit.t0 + beta * ln_r
        dg_dxi = 0.5 * beta * (ln_r ** 2)
        dg_dbeta = ln_r
    else:
        r_xi = r ** xi
        thr = fit.t0 + (beta / xi) * (r_xi - 1.0)
        ln_r = math.log(r)
        dg_dxi = (beta / (xi * xi)) * (xi * r_xi * ln_r - (r_xi - 1.0))
        dg_dbeta = (r_xi - 1.0) / xi

    if use_np_shift and fit.np_shift_z != 0.0:
        var_thr = (
            (dg_dxi ** 2) * (fit.std_xi ** 2)
            + (dg_dbeta ** 2) * (fit.std_beta ** 2)
            + 2.0 * dg_dxi * dg_dbeta * fit.cov_xi_beta
        )
        # Include anchor quantile variance contribution ~ p0(1-p0)/(n * f(t0)^2), f(t0) ~ p0/beta
        var_t0 = ((1.0 - fit.p_exceed) / max(fit.p_exceed * max(fit.n_total, 1), 1.0)) * (fit.beta ** 2)
        sigma_thr = math.sqrt(max(var_thr + var_t0, 0.0))
        thr = thr + fit.np_shift_z * sigma_thr
    return float(thr)


class DomainPOTGPDCalibrator:
    """Per-observable-domain POT-GPD extreme-value calibrator with global fallback."""

    def __init__(
        self,
        *,
        anchor_q: float = 0.95,
        xi_bounds: tuple[float, float] = (-0.35, 0.20),
        np_shift_z: float = 0.0,
        min_domain_events: int = 250,
    ) -> None:
        self.anchor_q = float(anchor_q)
        self.xi_bounds = xi_bounds
        self.np_shift_z = float(np_shift_z)
        self.min_domain_events = int(min_domain_events)
        self.global_fit: POTGPDFitResult | None = None
        self.domain_fits: dict[int, POTGPDFitResult] = {}

    def fit(
        self,
        scores: np.ndarray,
        is_background: np.ndarray,
        obs_domain_id: np.ndarray | None = None,
        weights: np.ndarray | None = None,
    ) -> 'DomainPOTGPDCalibrator':
        s = np.asarray(scores, dtype=np.float64)
        bg = np.asarray(is_background, dtype=bool)
        w_bg = None if weights is None else np.asarray(weights, dtype=np.float64)[bg]
        self.global_fit = fit_pot_gpd(
            s[bg], w_bg, anchor_q=self.anchor_q, xi_bounds=self.xi_bounds, np_shift_z=self.np_shift_z
        )
        self.domain_fits.clear()
        if obs_domain_id is not None:
            dom = np.asarray(obs_domain_id, dtype=np.int64)
            for d in np.unique(dom):
                d_int = int(d)
                if d_int <= 0:
                    continue
                dm = bg & (dom == d_int)
                if int(dm.sum()) >= self.min_domain_events:
                    w_d = None if weights is None else np.asarray(weights, dtype=np.float64)[dm]
                    self.domain_fits[d_int] = fit_pot_gpd(
                        s[dm], w_d, anchor_q=self.anchor_q, xi_bounds=self.xi_bounds, np_shift_z=self.np_shift_z
                    )
        return self

    def threshold(self, alpha_fpr: float, domain_id: int | None = None) -> float:
        if self.global_fit is None:
            raise RuntimeError('DomainPOTGPDCalibrator must be fitted before calling threshold()')
        fit = self.domain_fits.get(int(domain_id), self.global_fit) if domain_id is not None else self.global_fit
        return pot_gpd_threshold(fit, alpha_fpr)


def compare_pot_gpd_vs_empirical_split_half(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    fpr_targets: Sequence[float] = (0.01, 0.001),
    n_splits: int = 200,
    seed: int = 20263250,
    anchor_q: float = 0.95,
    xi_bounds: tuple[float, float] = (-0.35, 0.20),
    weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compare POT-GPD vs empirical quantile threshold variance & FPR stability via Split-Half resampling."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    w = np.ones(len(s), dtype=np.float64) if weights is None else np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    bg_idx = np.flatnonzero(y != 2)
    sg_idx = np.flatnonzero(y == 2)
    if len(bg_idx) < 100 or len(sg_idx) < 10:
        raise ValueError('Insufficient events for split-half POT-GPD comparison')

    rng = np.random.default_rng(seed)
    n_bg_half = len(bg_idx) // 2
    n_sg_half = len(sg_idx) // 2

    results: dict[str, Any] = {'n_splits': int(n_splits), 'n_bg_half': int(n_bg_half), 'targets': {}}
    emp_thrs: dict[float, list[float]] = {float(a): [] for a in fpr_targets}
    gpd_thrs: dict[float, list[float]] = {float(a): [] for a in fpr_targets}
    emp_fprs: dict[float, list[float]] = {float(a): [] for a in fpr_targets}
    gpd_fprs: dict[float, list[float]] = {float(a): [] for a in fpr_targets}
    emp_effs: dict[float, list[float]] = {float(a): [] for a in fpr_targets}
    gpd_effs: dict[float, list[float]] = {float(a): [] for a in fpr_targets}

    for _ in range(int(n_splits)):
        perm_b = rng.permutation(bg_idx)
        perm_s = rng.permutation(sg_idx)
        b_lock, b_held = perm_b[:n_bg_half], perm_b[n_bg_half:]
        s_held = perm_s[n_sg_half:]

        sb_lock = s[b_lock]
        wb_lock = w[b_lock]
        sb_held = s[b_held]
        wb_held = w[b_held]
        ss_held = s[s_held]
        ws_held = w[s_held]

        wb_held_sum = max(float(wb_held.sum()), 1e-30)
        ws_held_sum = max(float(ws_held.sum()), 1e-30)

        gpd_fit = fit_pot_gpd(sb_lock, weights=None if weights is None else wb_lock, anchor_q=anchor_q, xi_bounds=xi_bounds)

        # Weighted empirical quantile helper
        ord_b = np.argsort(-sb_lock, kind='mergesort')
        sv_b = sb_lock[ord_b]
        wv_b = wb_lock[ord_b]
        cum_w = np.cumsum(wv_b) / max(float(wv_b.sum()), 1e-30)

        for a in fpr_targets:
            af = float(a)
            # Empirical cut
            allowed = np.flatnonzero(cum_w <= af + 1e-14)
            t_emp = float(sv_b[allowed[-1]]) if len(allowed) else float(sv_b[0] + 1e-6)
            t_gpd = pot_gpd_threshold(gpd_fit, af, scores_bg_fallback=sb_lock, use_np_shift=False)

            emp_thrs[af].append(t_emp)
            gpd_thrs[af].append(t_gpd)
            emp_fprs[af].append(float(wb_held[sb_held >= t_emp].sum() / wb_held_sum))
            gpd_fprs[af].append(float(wb_held[sb_held >= t_gpd].sum() / wb_held_sum))
            emp_effs[af].append(float(ws_held[ss_held >= t_emp].sum() / ws_held_sum))
            gpd_effs[af].append(float(ws_held[ss_held >= t_gpd].sum() / ws_held_sum))

    for a in fpr_targets:
        af = float(a)
        te = np.asarray(emp_thrs[af], dtype=np.float64)
        tg = np.asarray(gpd_thrs[af], dtype=np.float64)
        fe = np.asarray(emp_fprs[af], dtype=np.float64)
        fg = np.asarray(gpd_fprs[af], dtype=np.float64)
        ee = np.asarray(emp_effs[af], dtype=np.float64)
        eg = np.asarray(gpd_effs[af], dtype=np.float64)

        var_e = float(np.var(te, ddof=1))
        var_g = float(np.var(tg, ddof=1))
        fpr_var_e = float(np.var(fe, ddof=1))
        fpr_var_g = float(np.var(fg, ddof=1))
        results['targets'][f'{af:.4g}'] = {
            'target_fpr': af,
            'empirical_threshold_mean': float(np.mean(te)),
            'empirical_threshold_std': float(np.sqrt(var_e)),
            'empirical_threshold_var': var_e,
            'pot_gpd_threshold_mean': float(np.mean(tg)),
            'pot_gpd_threshold_std': float(np.sqrt(var_g)),
            'pot_gpd_threshold_var': var_g,
            'threshold_variance_reduction_factor': float(var_e / max(var_g, 1e-12)),
            'threshold_std_reduction_factor': float(math.sqrt(var_e / max(var_g, 1e-12))),
            'empirical_held_fpr_mean': float(np.mean(fe)),
            'empirical_held_fpr_std': float(math.sqrt(fpr_var_e)),
            'empirical_held_fpr_rmse': float(np.sqrt(np.mean((fe - af) ** 2))),
            'pot_gpd_held_fpr_mean': float(np.mean(fg)),
            'pot_gpd_held_fpr_std': float(math.sqrt(fpr_var_g)),
            'pot_gpd_held_fpr_rmse': float(np.sqrt(np.mean((fg - af) ** 2))),
            'fpr_variance_reduction_factor': float(fpr_var_e / max(fpr_var_g, 1e-15)),
            'empirical_tau_eff_mean_pct': float(np.mean(ee) * 100.0),
            'empirical_tau_eff_std_pp': float(np.std(ee, ddof=1) * 100.0),
            'pot_gpd_tau_eff_mean_pct': float(np.mean(eg) * 100.0),
            'pot_gpd_tau_eff_std_pp': float(np.std(eg, ddof=1) * 100.0),
        }
    return results


# ===========================================================================
# 2. Pure-Observable Golden Micro-Channel & Control-Region Router
# ===========================================================================
@dataclass
class DiscoveryRouterConfig:
    """Purely observable routing cuts for the 4 Golden Micro-Channels, Low-E Abstention & 3 CRs."""

    # C1: Golden-3Prong-ZeroMu
    c1_lxy_sig_min: float = 3.5
    c1_mass_min: float = 0.35
    c1_mass_max: float = 6.00
    c1_llr_min: float = -0.25

    # C2: Golden-Kink-Mu
    c2_kink_min: float = 12.0
    c2_dphi_min: float = 2.1
    c2_llr_min: float = -0.50

    # C3: Golden-Rho-1Prong
    c3_ip_sig_min: float = 4.5
    c3_rho_mass_min: float = 0.30
    c3_rho_mass_max: float = 1.20
    c3_pi0_frac_min: float = 0.15
    c3_llr_min: float = -0.30

    # C4: Fortune-HighMT-NSBI
    c4_e_vis_min: float = 250.0
    c4_pt_miss_min: float = 1.8
    c4_llr_min: float = 0.25
    c4_floor_llr: float = -1.25

    # C5: Low-E Ambiguity Abstention
    low_e_abstain_gev: float = 140.0

    # In-situ Control Regions
    cr_tau_veto_max: float = 0.65
    cr_mumu_charm_min: float = 0.45
    cr_1mu_kink_max: float = 8.0
    cr_1mu_lxy_max: float = 2.5
    cr_pi0_frac_min: float = 0.25


def _extract_obs_array(
    obs: Mapping[str, Any],
    key: str,
    n_events: int,
    default: float | bool,
    dtype: Any,
) -> np.ndarray:
    if key in obs and obs[key] is not None:
        arr = np.asarray(obs[key], dtype=dtype).reshape(-1)
        if len(arr) != n_events:
            raise ValueError(f'Observable {key!r} length {len(arr)} != {n_events}')
        return arr
    return np.full(n_events, default, dtype=dtype)


def route_golden_micro_channels(
    obs_summary: Mapping[str, Any],
    cfg: DiscoveryRouterConfig | None = None,
) -> dict[str, Any]:
    """Mutually-exclusive pure-observable routing into 4 Golden Micro-Channels, Low-E Abstention & 3 CRs.

    Strictly ignores any non-observable truth keys even if present in `obs_summary`.
    """
    c = cfg or DiscoveryRouterConfig()
    clean_obs = {k: obs_summary[k] for k in OBSERVABLE_SUMMARY_KEYS if k in obs_summary}
    if 'tau_llr' not in clean_obs:
        raise KeyError('obs_summary must contain pure observable `tau_llr`')

    tau_llr = np.asarray(clean_obs['tau_llr'], dtype=np.float64).reshape(-1)
    n = len(tau_llr)

    # Support both canonical keys and [gen4-contract] Section 5.2 nsbi_obs aliases
    obs_dom = _extract_obs_array(clean_obs, 'obs_domain_id', n, 1, np.int64)
    e_vis = _extract_obs_array(
        clean_obs, 'e_vis_gev' if 'e_vis_gev' in clean_obs else 'e_vis_hat_gev', n, 400.0, np.float64
    )
    pt_miss = _extract_obs_array(
        clean_obs, 'pt_miss_gev' if 'pt_miss_gev' in clean_obs else 'pt_miss_hat_gev', n, 0.0, np.float64
    )
    charm_prob = _extract_obs_array(clean_obs, 'charm_prob', n, 0.0, np.float64)
    n_sv_prongs = _extract_obs_array(clean_obs, 'n_sv_prongs', n, 0, np.int64)
    sv_lxy_sig = _extract_obs_array(
        clean_obs, 'sv_lxy_sig' if 'sv_lxy_sig' in clean_obs else 'sl_xy', n, 0.0, np.float64
    )
    sv_mass = _extract_obs_array(
        clean_obs, 'sv_mass_gev' if 'sv_mass_gev' in clean_obs else 'm_sv_corr_gev', n, 0.0, np.float64
    )
    n_pen_mu = _extract_obs_array(
        clean_obs, 'n_pen_muons' if 'n_pen_muons' in clean_obs else 'n_mu_veto', n, 0, np.int64
    )
    kink_llr = _extract_obs_array(
        clean_obs, 'kink_llr' if 'kink_llr' in clean_obs else 'lambda_kink', n, 0.0, np.float64
    )
    dphi_mu = _extract_obs_array(
        clean_obs, 'dphi_mu_had' if 'dphi_mu_had' in clean_obs else 'dphi_mu_ht', n, 0.0, np.float64
    )
    ip_sig = _extract_obs_array(clean_obs, 'ip_sig_max', n, 0.0, np.float64)
    if 'pi0_em_frac' in clean_obs:
        pi0_frac = _extract_obs_array(clean_obs, 'pi0_em_frac', n, 0.0, np.float64)
    elif 'm_gg_pi0_gev' in clean_obs:
        m_gg = _extract_obs_array(clean_obs, 'm_gg_pi0_gev', n, 0.0, np.float64)
        pi0_frac = np.where((m_gg > 0.05) & (m_gg < 0.25), 0.45, 0.0)
    else:
        pi0_frac = np.zeros(n, dtype=np.float64)
    rho_mass = _extract_obs_array(
        clean_obs, 'rho_vis_mass_gev' if 'rho_vis_mass_gev' in clean_obs else 'm_vis_rho_gev', n, 0.0, np.float64
    )
    has_prompt_e = _extract_obs_array(clean_obs, 'has_prompt_e', n, False, bool)
    has_front_pix = _extract_obs_array(clean_obs, 'has_front_pixel', n, False, bool)
    has_tgt_pix = _extract_obs_array(clean_obs, 'has_target_pixel', n, False, bool) | has_front_pix
    has_q_mu1 = 'q_mu1' in clean_obs and clean_obs['q_mu1'] is not None
    q_mu1 = _extract_obs_array(clean_obs, 'q_mu1', n, 0.0, np.float64)

    active = obs_dom > 0
    # Detect scintillator active GeV scale (MC8588 ~5 GeV) vs calibrated neutrino GeV scale (~220 GeV)
    scint_scale = bool(n > 0 and float(np.median(e_vis[active] if active.any() else e_vis)) < 40.0)
    zero_spec_mu = ((n_pen_mu == 0) | ((n_pen_mu == 1) & (np.abs(q_mu1) < 0.5))) if (scint_scale and has_q_mu1) else (n_pen_mu == 0)
    has_spec_mu = (n_pen_mu == 1) & (np.abs(q_mu1) >= 0.5) if (scint_scale and has_q_mu1) else (n_pen_mu == 1)
    if scint_scale:
        q60 = float(np.quantile(tau_llr, 0.60))
        q75 = float(np.quantile(tau_llr, 0.75))
        q88 = float(np.quantile(tau_llr, 0.88))
        low_e_cut = 2.0
        c4_e_cut = 5.5
        c4_pt_cut = 0.70
        c4_llr_cut = q88
        c4_floor_cut = q75
        c1_llr_cut = q60
        c2_dphi_cut = 1.70
        c2_llr_cut = q75
        c3_llr_cut = q60
        cr_veto_cut = q75
    else:
        low_e_cut = c.low_e_abstain_gev
        c4_e_cut = c.c4_e_vis_min
        c4_pt_cut = c.c4_pt_miss_min
        c4_llr_cut = c.c4_llr_min
        c4_floor_cut = c.c4_floor_llr
        c1_llr_cut = c.c1_llr_min
        c2_dphi_cut = c.c2_dphi_min
        c2_llr_cut = c.c2_llr_min
        c3_llr_cut = c.c3_llr_min
        cr_veto_cut = c.cr_tau_veto_max

    channel_id = np.full(n, 8, dtype=np.int64)  # default Abstain-Bulk
    assigned = ~active

    # 1. CR-mumu (id=5): dimuon or high-charm single-muon non-tail region anchoring charm rate
    m_cr_mumu = (~assigned) & (tau_llr < cr_veto_cut) & (
        (n_pen_mu >= 2)
        | ((n_pen_mu == 1) & (charm_prob >= c.cr_mumu_charm_min))
    )
    channel_id[m_cr_mumu] = 5
    assigned |= m_cr_mumu

    # 2. C1: Golden-3Prong-ZeroMu (id=0)
    m_c1 = (
        (~assigned)
        & has_front_pix
        & zero_spec_mu
        & (~has_prompt_e)
        & (n_sv_prongs >= 3)
        & (sv_lxy_sig >= c.c1_lxy_sig_min)
        & (sv_mass >= c.c1_mass_min)
        & (sv_mass <= c.c1_mass_max)
        & (tau_llr >= c1_llr_cut)
    )
    channel_id[m_c1] = 0
    assigned |= m_c1

    # 3. C2: Golden-Kink-Mu (id=1)
    m_c2 = (
        (~assigned)
        & has_tgt_pix
        & ((n_pen_mu >= 1) if scint_scale else (n_pen_mu == 1))
        & (kink_llr >= c.c2_kink_min)
        & (dphi_mu >= c2_dphi_cut)
        & (tau_llr >= c2_llr_cut)
    )
    channel_id[m_c2] = 1
    assigned |= m_c2

    # 4. C3: Golden-Rho-1Prong (id=2)
    m_c3 = (
        (~assigned)
        & has_front_pix
        & zero_spec_mu
        & (~has_prompt_e)
        & (ip_sig >= c.c3_ip_sig_min)
        & (rho_mass >= c.c3_rho_mass_min)
        & (rho_mass <= c.c3_rho_mass_max)
        & (pi0_frac >= c.c3_pi0_frac_min)
        & (tau_llr >= c3_llr_cut)
    )
    channel_id[m_c3] = 2
    assigned |= m_c3

    # 5. CR-1mu (id=6): prompt zero-kink single muon anchoring numu flux & normalization
    m_cr_1mu = (
        (~assigned)
        & has_spec_mu
        & (kink_llr < c.cr_1mu_kink_max)
        & (sv_lxy_sig < c.cr_1mu_lxy_max)
        & (tau_llr < cr_veto_cut)
    )
    channel_id[m_cr_1mu] = 6
    assigned |= m_cr_1mu

    # 6. C5: Abstain-LowE (id=4): low-energy ambiguous events without clear silicon vertex
    m_low_e = (
        (~assigned)
        & (e_vis < low_e_cut)
        & (sv_lxy_sig < c.c1_lxy_sig_min)
        & (kink_llr < c.c2_kink_min)
        & (ip_sig < c.c3_ip_sig_min)
    )
    channel_id[m_low_e] = 4
    assigned |= m_low_e

    # 7. C4: Fortune-HighMT-NSBI (id=3): continuous high-E / high-pTmiss / high-LR discovery stream
    m_c4 = (
        (~assigned)
        & (
            ((e_vis >= c4_e_cut) & (pt_miss >= c4_pt_cut) & (tau_llr >= c4_floor_cut))
            | (tau_llr >= c4_llr_cut)
        )
    )
    channel_id[m_c4] = 3
    assigned |= m_c4

    # 8. CR-pi0 (id=7): prompt zero-muon EM/pi0-rich bulk anchoring NC energy scale & hadronization
    m_cr_pi0 = (
        (~assigned)
        & zero_spec_mu
        & (pi0_frac >= c.cr_pi0_frac_min)
        & (tau_llr < cr_veto_cut)
    )
    channel_id[m_cr_pi0] = 7
    assigned |= m_cr_pi0

    counts = {name: int(np.sum(channel_id == idx)) for idx, name in enumerate(CHANNEL_NAMES)}
    return {
        'channel_id': channel_id,
        'channel_names': CHANNEL_NAMES,
        'is_discovery_sr': np.isin(channel_id, DISCOVERY_CHANNEL_IDS),
        'is_control_region': np.isin(channel_id, CR_CHANNEL_IDS),
        'is_abstained': np.isin(channel_id, (4, 8)),
        'counts': counts,
    }


# ===========================================================================
# 3. Physical Forward-Flux Weighting & Joint NSBI-PL Significance Engine
# ===========================================================================
def compute_physical_event_weights(
    labels: np.ndarray,
    has_charm: np.ndarray | None = None,
    *,
    total_events: float = 180000.0,
    tau_fraction: float = 0.003,
    bkg_shares: Mapping[str, float] | None = None,
    mw2t_weights: np.ndarray | None = None,
    nc_charm_physical_fraction: float | None = 0.045,
) -> np.ndarray:
    """Scale MC events to expected HL-LHC 2t forward-flux yields under MW2t mass weighting."""
    y = np.asarray(labels, dtype=np.int64)
    n = len(y)
    base_w = np.ones(n, dtype=np.float64) if mw2t_weights is None else np.maximum(np.asarray(mw2t_weights, dtype=np.float64), 0.0)
    hc = np.zeros(n, dtype=bool) if has_charm is None else np.asarray(has_charm, dtype=bool)
    shares = dict(FIDUCIAL_2T_BKG_SHARES if bkg_shares is None else bkg_shares)

    n_sig_exp = float(total_events) * float(tau_fraction)
    n_bkg_exp = float(total_events) * (1.0 - float(tau_fraction))

    w = np.zeros(n, dtype=np.float64)
    m_tau = y == 2
    if m_tau.any() and float(base_w[m_tau].sum()) > 0.0:
        w[m_tau] = base_w[m_tau] * (n_sig_exp / float(base_w[m_tau].sum()))

    for cls_id, key in ((0, 'nue_cc'), (1, 'numu_cc'), (3, 'nc')):
        mc = y == cls_id
        if not mc.any():
            continue
        if cls_id == 3 and nc_charm_physical_fraction is not None and (mc & hc).any() and (mc & (~hc)).any():
            f_c = float(nc_charm_physical_fraction)
            mc_c = mc & hc
            mc_l = mc & (~hc)
            w[mc_c] = base_w[mc_c] * ((n_bkg_exp * float(shares['nc']) * f_c) / max(float(base_w[mc_c].sum()), 1e-30))
            w[mc_l] = base_w[mc_l] * ((n_bkg_exp * float(shares['nc']) * (1.0 - f_c)) / max(float(base_w[mc_l].sum()), 1e-30))
        elif float(base_w[mc].sum()) > 0.0:
            w[mc] = base_w[mc] * ((n_bkg_exp * float(shares[key])) / float(base_w[mc].sum()))
    return w


def build_nsbi_channel_templates(
    tau_llr: np.ndarray,
    channel_id: np.ndarray,
    labels: np.ndarray,
    *,
    has_charm: np.ndarray | None = None,
    mw2t_weights: np.ndarray | None = None,
    total_events: float = 180000.0,
    tau_fraction: float = 0.003,
    n_subbins_per_channel: int = 4,
    nc_charm_proxy_frac: float = 0.035,
    zero_bkg_jeffreys_prior: bool = True,
) -> dict[str, Any]:
    """Build fine-binned Discovery Micro-Channel + 3 In-Situ Control Region templates with MC variances.

    Bin layout (`n_sr_bins = 4 * n_subbins_per_channel`, followed by 3 CR bins):
      0 .. n_sr_bins - 1 : sub-bins of C1, C2, C3, C4 ordered by `tau_llr`
      n_sr_bins + 0      : CR-mumu (channel_id == 5)
      n_sr_bins + 1      : CR-1mu  (channel_id == 6)
      n_sr_bins + 2      : CR-pi0  (channel_id == 7)
    """
    llr = np.asarray(tau_llr, dtype=np.float64)
    ch_id = np.asarray(channel_id, dtype=np.int64)
    y = np.asarray(labels, dtype=np.int64)
    hc = np.zeros(len(y), dtype=bool) if has_charm is None else np.asarray(has_charm, dtype=bool)
    w = compute_physical_event_weights(
        y, hc, total_events=total_events, tau_fraction=tau_fraction, mw2t_weights=mw2t_weights
    )

    k_sub = max(1, int(n_subbins_per_channel))
    n_sr_bins = 4 * k_sub
    n_bins = n_sr_bins + 3
    bin_ids = np.full(len(y), -1, dtype=np.int64)

    sr_edges: dict[str, list[float]] = {}
    for slot_idx, cid in enumerate(DISCOVERY_CHANNEL_IDS):
        cm = ch_id == cid
        cname = CHANNEL_NAMES[cid]
        if not cm.any():
            sr_edges[cname] = [-np.inf] + [0.0] * (k_sub - 1) + [np.inf]
            continue
        vals = llr[cm]
        k_eff = min(k_sub, max(1, len(vals) // 12))
        if k_eff == 1:
            bin_ids[cm] = slot_idx * k_sub
            sr_edges[cname] = [-np.inf, np.inf]
            continue
        qs = np.linspace(0.0, 1.0, k_eff + 1)
        edges = np.quantile(vals, qs)
        for j in range(1, k_eff):
            if edges[j] <= edges[j - 1]:
                edges[j] = edges[j - 1] + 1e-4
        edges[0] = -np.inf
        edges[-1] = np.inf
        sr_edges[cname] = edges.tolist()
        loc = np.clip(np.searchsorted(edges[1:-1], vals, side='right'), 0, k_eff - 1)
        bin_ids[cm] = slot_idx * k_sub + loc

    for cr_offset, cid in enumerate(CR_CHANNEL_IDS):
        bin_ids[ch_id == cid] = n_sr_bins + cr_offset

    valid = (bin_ids >= 0) & (bin_ids < n_bins)

    def _hist_var(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m = valid & mask
        h = np.zeros(n_bins, dtype=np.float64)
        v = np.zeros(n_bins, dtype=np.float64)
        if m.any():
            np.add.at(h, bin_ids[m], w[m])
            np.add.at(v, bin_ids[m], w[m] ** 2)
        return h, v

    s_tau, var_s = _hist_var(y == 2)
    b_nue_l, v_nue_l = _hist_var((y == 0) & (~hc))
    b_nue_c, v_nue_c = _hist_var((y == 0) & hc)
    b_numu_l, v_numu_l = _hist_var((y == 1) & (~hc))
    b_numu_c, v_numu_c = _hist_var((y == 1) & hc)
    b_nc_l, v_nc_l = _hist_var((y == 3) & (~hc))
    b_nc_c, v_nc_c = _hist_var((y == 3) & hc)

    if b_nc_c.sum() == 0.0 and nc_charm_proxy_frac > 0.0 and b_nc_l.sum() > 0.0:
        f_ncc = float(nc_charm_proxy_frac)
        tot_nc = float(b_nc_l.sum())
        if b_numu_c.sum() > 0.0:
            proxy = 0.5 * (b_nc_l / max(tot_nc, 1e-12)) + 0.5 * (b_numu_c / max(float(b_numu_c.sum()), 1e-12))
        else:
            proxy = b_nc_l / max(tot_nc, 1e-12)
        proxy /= max(float(proxy.sum()), 1e-12)
        b_nc_c = (f_ncc * tot_nc) * proxy
        b_nc_l = (1.0 - f_ncc) * b_nc_l
        v_nc_c = (f_ncc ** 2) * v_nc_l

    n_raw_bkg = np.zeros(n_bins, dtype=np.int64)
    m_bkg_v = valid & (y != 2)
    if m_bkg_v.any():
        np.add.at(n_raw_bkg, bin_ids[m_bkg_v], 1)

    b_tot = b_nue_l + b_nue_c + b_numu_l + b_numu_c + b_nc_l + b_nc_c
    var_bkg = v_nue_l + v_nue_c + v_numu_l + v_numu_c + v_nc_l + v_nc_c

    if zero_bkg_jeffreys_prior and int(m_bkg_v.sum()) > 0:
        w_bkg_mean = float(total_events * (1.0 - tau_fraction)) / float(m_bkg_v.sum())
        empty_sig = (b_tot <= 0.0) & (s_tau > 0.0)
        if np.any(empty_sig):
            b_jeff = 0.5 * w_bkg_mean
            b_numu_l[empty_sig] = b_jeff
            b_tot[empty_sig] = b_jeff
            var_bkg[empty_sig] = 2.0 * b_jeff * b_jeff

    return {
        'n_bins': n_bins,
        'n_sr_bins': n_sr_bins,
        'n_subbins_per_channel': k_sub,
        'sr_edges': sr_edges,
        'bin_ids': bin_ids,
        'event_weights': w,
        's_tau': s_tau,
        'b_nue_light': b_nue_l,
        'b_nue_charm': b_nue_c,
        'b_numu_light': b_numu_l,
        'b_numu_charm': b_numu_c,
        'b_nc_light': b_nc_l,
        'b_nc_charm': b_nc_c,
        'b_total': b_tot,
        'var_mc_bkg': var_bkg,
        'var_mc_sig': var_s,
        'n_raw_bkg': n_raw_bkg,
    }


def fit_nsbi_profiled_likelihood(
    templates: Mapping[str, Any],
    *,
    delta_sys: float = 0.0159,
    sigma_lumi: float = 0.05,
    sigma_charm: float = 0.30,
    sigma_nc: float = 0.10,
    sigma_numu: float = 0.08,
    sigma_nue: float = 0.10,
    enable_barlow_beeston: bool = True,
    n_obs_override: np.ndarray | None = None,
    theta_obs_override: np.ndarray | None = None,
    beta_obs_override: np.ndarray | None = None,
    fit_signal_strength: bool = False,
) -> dict[str, Any]:
    """Joint Profiled Likelihood over 4 Golden Micro-Channels + 3 In-Situ CRs with Barlow-Beeston lite."""
    s_tau = np.asarray(templates['s_tau'], dtype=np.float64)
    b_nue_l = np.asarray(templates['b_nue_light'], dtype=np.float64)
    b_nue_c = np.asarray(templates['b_nue_charm'], dtype=np.float64)
    b_numu_l = np.asarray(templates['b_numu_light'], dtype=np.float64)
    b_numu_c = np.asarray(templates['b_numu_charm'], dtype=np.float64)
    b_nc_l = np.asarray(templates['b_nc_light'], dtype=np.float64)
    b_nc_c = np.asarray(templates['b_nc_charm'], dtype=np.float64)
    b_tot = b_nue_l + b_nue_c + b_numu_l + b_numu_c + b_nc_l + b_nc_c
    var_bkg = np.asarray(templates['var_mc_bkg'], dtype=np.float64)

    n_bins = len(b_tot)
    n_sr = int(templates.get('n_sr_bins', max(0, n_bins - 3)))
    n_obs = (s_tau + b_tot) if n_obs_override is None else np.asarray(n_obs_override, dtype=np.float64)
    delta_sq = np.where(b_tot > 1e-9, np.clip(var_bkg / np.maximum(b_tot * b_tot, 1e-18), 0.0, 2.5), 0.0)

    pos_sb = (s_tau > 0.0) & (b_tot > 0.0)
    z0_stat = float(
        math.sqrt(
            max(0.0, float(np.sum(2.0 * ((s_tau[pos_sb] + b_tot[pos_sb]) * np.log1p(s_tau[pos_sb] / b_tot[pos_sb]) - s_tau[pos_sb]))))
        )
    )

    # Nuisance layout: [eta_lumi, theta_charm, eta_nc, eta_numu, eta_nue, eta_sr_sys, eta_c1..c4_shape(4)]
    k_sub = max(1, n_sr // 4) if n_sr >= 4 else 1
    n_nuis = 10
    theta_obs = np.zeros(n_nuis, dtype=np.float64) if theta_obs_override is None else np.asarray(theta_obs_override, dtype=np.float64)

    sr_mask = np.zeros(n_bins, dtype=np.float64)
    sr_mask[:n_sr] = 1.0
    ch_masks = np.zeros((4, n_bins), dtype=np.float64)
    for c_i in range(4):
        lo = c_i * k_sub
        hi = min(n_sr, (c_i + 1) * k_sub)
        if lo < hi:
            ch_masks[c_i, lo:hi] = 1.0

    def _rates_jac(theta: np.ndarray, mu: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        e_lumi, th_ch, e_nc, e_numu, e_nue, e_sys = [float(theta[k]) for k in range(6)]
        e_ch_sh = theta[6:10]
        ch_log = np.tensordot(e_ch_sh, ch_masks, axes=(0, 0)) * float(delta_sys)
        common_b = np.clip(sigma_lumi * e_lumi + float(delta_sys) * e_sys * sr_mask + ch_log, -3.5, 3.5)

        r_nue_l = b_nue_l * np.exp(np.clip(common_b + sigma_nue * e_nue, -4.0, 4.0))
        r_nue_c = b_nue_c * np.exp(np.clip(common_b + sigma_nue * e_nue + sigma_charm * th_ch, -4.0, 4.0))
        r_numu_l = b_numu_l * np.exp(np.clip(common_b + sigma_numu * e_numu, -4.0, 4.0))
        r_numu_c = b_numu_c * np.exp(np.clip(common_b + sigma_numu * e_numu + sigma_charm * th_ch, -4.0, 4.0))
        r_nc_l = b_nc_l * np.exp(np.clip(common_b + sigma_nc * e_nc, -4.0, 4.0))
        r_nc_c = b_nc_c * np.exp(np.clip(common_b + sigma_nc * e_nc + sigma_charm * th_ch, -4.0, 4.0))

        sig_unit = s_tau * np.exp(np.clip(sigma_lumi * e_lumi, -3.5, 3.5))
        r_sig = mu * sig_unit
        b_pred = r_nue_l + r_nue_c + r_numu_l + r_numu_c + r_nc_l + r_nc_c
        nu_tot = b_pred + r_sig

        jac = np.zeros((n_nuis, n_bins), dtype=np.float64)
        jac[0] = sigma_lumi * (b_pred + r_sig)
        jac[1] = sigma_charm * (r_nue_c + r_numu_c + r_nc_c)
        jac[2] = sigma_nc * (r_nc_l + r_nc_c)
        jac[3] = sigma_numu * (r_numu_l + r_numu_c)
        jac[4] = sigma_nue * (r_nue_l + r_nue_c)
        jac[5] = float(delta_sys) * sr_mask * b_pred
        for c_i in range(4):
            jac[6 + c_i] = float(delta_sys) * ch_masks[c_i] * b_pred
        return nu_tot, jac, sig_unit

    def _obj_h0(theta: np.ndarray) -> tuple[float, np.ndarray]:
        nu, jac, _ = _rates_jac(theta, mu=0.0)
        dev, g_nu, _ = _barlow_beeston_inner_deviance(n_obs, nu, delta_sq, enable_barlow_beeston, beta_obs=beta_obs_override)
        diff = theta - theta_obs
        return dev + float(np.dot(diff, diff)), jac @ g_nu + 2.0 * diff

    bnds_h0 = [(-6.0, 6.0)] * n_nuis
    res0 = minimize(_obj_h0, x0=np.clip(theta_obs, -4.0, 4.0), jac=True, method='L-BFGS-B', bounds=bnds_h0, options={'gtol': 1e-7, 'ftol': 1e-11, 'maxiter': 400})
    val_h0 = float(res0.fun)
    val_hat = 0.0
    mu_hat = 1.0

    if fit_signal_strength or n_obs_override is not None:
        def _obj_free(x: np.ndarray) -> tuple[float, np.ndarray]:
            mu_v = float(x[0])
            th = x[1:]
            nu, jac_th, sig_unit = _rates_jac(th, mu=mu_v)
            dev, g_nu, _ = _barlow_beeston_inner_deviance(n_obs, nu, delta_sq, enable_barlow_beeston, beta_obs=beta_obs_override)
            diff = th - theta_obs
            g_mu = float(np.dot(g_nu, sig_unit))
            g_th = jac_th @ g_nu + 2.0 * diff
            return dev + float(np.dot(diff, diff)), np.concatenate([[g_mu], g_th])

        if n_obs_override is None:
            x0 = np.zeros(1 + n_nuis, dtype=np.float64)
            x0[0] = 1.0
            rf = minimize(_obj_free, x0=x0, jac=True, method='L-BFGS-B', bounds=[(0.0, 20.0)] + bnds_h0, options={'gtol': 1e-7, 'ftol': 1e-11, 'maxiter': 400})
            mu_hat = float(rf.x[0])
            val_hat = float(rf.fun)
        else:
            x_kkt = np.concatenate([[0.0], res0.x])
            _, g_kkt = _obj_free(x_kkt)
            if float(g_kkt[0]) >= -1e-7:
                mu_hat = 0.0
                val_hat = val_h0
            else:
                x0 = np.concatenate([[1e-3], res0.x])
                rf = minimize(_obj_free, x0=x0, jac=True, method='L-BFGS-B', bounds=[(0.0, 20.0)] + bnds_h0, options={'gtol': 1e-7, 'ftol': 1e-11, 'maxiter': 400})
                mu_hat = float(rf.x[0])
                val_hat = float(rf.fun)

    q0 = float(max(0.0, val_h0 - val_hat)) if mu_hat > 0.0 else 0.0
    z0_pl = float(math.sqrt(q0))
    return {
        'n_bins': int(n_bins),
        'n_sr_bins': int(n_sr),
        'expected_sr_signal': float(s_tau[:n_sr].sum()),
        'expected_sr_background': float(b_tot[:n_sr].sum()),
        'expected_cr_background': float(b_tot[n_sr:].sum()),
        'z0_stat_only': z0_stat,
        'z0_profile_likelihood': z0_pl,
        'q0_profile_likelihood': q0,
        'mu_hat': mu_hat,
        'enable_barlow_beeston': bool(enable_barlow_beeston),
        'optimizer_converged': bool(res0.success),
    }


def fit_unbinned_nsbi_pl(
    tau_llr: np.ndarray,
    channel_id: np.ndarray,
    labels: np.ndarray,
    *,
    has_charm: np.ndarray | None = None,
    mw2t_weights: np.ndarray | None = None,
    total_events: float = 180000.0,
    tau_fraction: float = 0.003,
    delta_sys: float = 0.0159,
    sigma_lumi: float = 0.05,
    sigma_charm: float = 0.30,
    sigma_nc: float = 0.10,
    sigma_numu: float = 0.08,
    sigma_nue: float = 0.10,
) -> dict[str, Any]:
    """Asimov Unbinned NSBI Profiled Likelihood across C1..C4 + Poisson In-Situ Control Regions.

    Uses the continuous event-level log density ratio `tau_llr` (clipped and calibrated per channel
    to respect expected signal/background yields) inside discovery micro-channels while profiling
    nuisance parameters simultaneously against the 3 in-situ Control Regions.
    """
    llr = np.asarray(tau_llr, dtype=np.float64)
    ch_id = np.asarray(channel_id, dtype=np.int64)
    y = np.asarray(labels, dtype=np.int64)
    hc = np.zeros(len(y), dtype=bool) if has_charm is None else np.asarray(has_charm, dtype=bool)
    w = compute_physical_event_weights(
        y, hc, total_events=total_events, tau_fraction=tau_fraction, mw2t_weights=mw2t_weights
    )

    is_sr = np.isin(ch_id, DISCOVERY_CHANNEL_IDS)
    is_tau = y == 2

    # Build event-level calibrated density ratio r_i = s(x_i) / b(x_i) inside each discovery channel
    # by calibrating the additive log-prior offset per channel so E_b[r(x)] == S_c / B_c (exact normalization!).
    is_bg_all = ~is_tau
    w_bkg_mean = float(total_events * (1.0 - tau_fraction)) / max(int(is_bg_all.sum()), 1)
    r_ev = np.zeros(len(y), dtype=np.float64)
    jeff_b_extra = 0.0
    for cid in DISCOVERY_CHANNEL_IDS:
        cm = ch_id == cid
        if not cm.any():
            continue
        w_s_c = float(w[cm & is_tau].sum())
        w_b_c = float(w[cm & (~is_tau)].sum())
        if w_s_c <= 0.0:
            continue
        if w_b_c <= 0.0:
            # Zero-MC-background golden channel: apply Gamma(1/2, 1) Jeffreys floor
            b_jeff = 0.5 * w_bkg_mean
            jeff_b_extra += b_jeff
            raw_r = np.exp(np.clip(llr[cm], -8.0, 8.0))
            raw_mean = max(float(np.mean(raw_r)), 1e-12)
            r_ev[cm] = np.clip(raw_r * ((w_s_c / b_jeff) / raw_mean), 0.0, 1e4)
            continue
        raw_r = np.exp(np.clip(llr[cm], -8.0, 8.0))
        w_b_arr = np.maximum(w[cm & (~is_tau)], 1e-12)
        raw_mean_b = float(np.average(raw_r[~is_tau[cm]], weights=w_b_arr))
        scale_c = (w_s_c / w_b_c) / max(raw_mean_b, 1e-12)
        r_c = np.clip(raw_r * scale_c, 0.0, 1e4)
        # Re-normalize after clip so E_b[r] == S_c / B_c
        clipped_mean_b = float(np.average(r_c[~is_tau[cm]], weights=w_b_arr))
        if clipped_mean_b > 1e-12:
            r_c = r_c * ((w_s_c / w_b_c) / clipped_mean_b)
        r_ev[cm] = r_c

    sr_idx = np.flatnonzero(is_sr)
    sr_w = w[sr_idx]
    sr_r = r_ev[sr_idx]
    sr_y = y[sr_idx]
    sr_hc = hc[sr_idx]
    s_tot_sr = float(sr_w[sr_y == 2].sum())

    # Process weights in SR and in the 3 CRs
    def _proc_yields(mask: np.ndarray) -> np.ndarray:
        return np.array([
            float(w[mask & (y == 0) & (~hc)].sum()),
            float(w[mask & (y == 0) & hc].sum()),
            float(w[mask & (y == 1) & (~hc)].sum()),
            float(w[mask & (y == 1) & hc].sum()),
            float(w[mask & (y == 3) & (~hc)].sum()),
            float(w[mask & (y == 3) & hc].sum()),
        ], dtype=np.float64)

    sr_proc_b = _proc_yields(is_sr)
    if jeff_b_extra > 0.0:
        sr_proc_b[2] += jeff_b_extra
    cr_proc_b = np.stack([_proc_yields(ch_id == cid) for cid in CR_CHANNEL_IDS], axis=0)  # (3, 6)
    cr_obs = cr_proc_b.sum(axis=1)

    # Stat-only unbinned Asimov Z0 = sqrt(2 * [ sum_{i in SR} w_i log(1 + r_i) - S_tot ])
    z0_stat_sq = 2.0 * (float(np.sum(sr_w * np.log1p(sr_r))) - s_tot_sr)
    z0_stat = float(math.sqrt(max(0.0, z0_stat_sq)))

    # Profiled unbinned likelihood over 6 nuisances: [eta_lumi, theta_charm, eta_nc, eta_numu, eta_nue, eta_sr_sys]
    # Map every MC background event to its exact process index 0..5, and signal events to channel-mean bkg mult
    # so the unbinned extended Poisson KL divergence is strictly >= 0 at (mu=1, theta=0) with global min 0.0.
    sr_proc_idx = np.full(len(sr_idx), -1, dtype=np.int64)
    sr_proc_idx[(sr_y == 0) & (~sr_hc)] = 0
    sr_proc_idx[(sr_y == 0) & sr_hc] = 1
    sr_proc_idx[(sr_y == 1) & (~sr_hc)] = 2
    sr_proc_idx[(sr_y == 1) & sr_hc] = 3
    sr_proc_idx[(sr_y == 3) & (~sr_hc)] = 4
    sr_proc_idx[(sr_y == 3) & sr_hc] = 5
    sr_is_sig_ev = sr_y == 2
    b_tot_sr0 = float(sr_proc_b.sum())
    ll_asimov_ref = float(np.sum(sr_w * np.log1p(sr_r)))

    def _nll_2x(theta: np.ndarray, mu: float) -> float:
        e_lumi, th_ch, e_nc, e_numu, e_nue, e_sys = [float(x) for x in theta]
        proc_mult = np.array([
            math.exp(np.clip(sigma_lumi * e_lumi + sigma_nue * e_nue, -4.0, 4.0)),
            math.exp(np.clip(sigma_lumi * e_lumi + sigma_nue * e_nue + sigma_charm * th_ch, -4.0, 4.0)),
            math.exp(np.clip(sigma_lumi * e_lumi + sigma_numu * e_numu, -4.0, 4.0)),
            math.exp(np.clip(sigma_lumi * e_lumi + sigma_numu * e_numu + sigma_charm * th_ch, -4.0, 4.0)),
            math.exp(np.clip(sigma_lumi * e_lumi + sigma_nc * e_nc, -4.0, 4.0)),
            math.exp(np.clip(sigma_lumi * e_lumi + sigma_nc * e_nc + sigma_charm * th_ch, -4.0, 4.0)),
        ], dtype=np.float64)

        sys_sr_mult = math.exp(np.clip(delta_sys * e_sys, -3.0, 3.0))
        sig_mult = mu * math.exp(np.clip(sigma_lumi * e_lumi, -3.5, 3.5))

        b_sr_pred = float(np.dot(sr_proc_b, proc_mult)) * sys_sr_mult
        s_sr_pred = sig_mult * s_tot_sr
        mean_b_mult = b_sr_pred / max(b_tot_sr0, 1e-12)

        ev_b_scale = np.where(sr_is_sig_ev, mean_b_mult, proc_mult[np.maximum(sr_proc_idx, 0)] * sys_sr_mult)
        ll_sr_events = float(np.sum(sr_w * np.log(np.maximum(sig_mult * sr_r + ev_b_scale, 1e-12))))
        dev_sr = 2.0 * ((s_sr_pred + b_sr_pred) - (s_tot_sr + b_tot_sr0) - (ll_sr_events - ll_asimov_ref))

        # CR Poisson terms
        cr_pred = cr_proc_b @ proc_mult
        dev_cr = 0.0
        for c_k in range(3):
            if cr_pred[c_k] > 1e-12 and cr_obs[c_k] > 0.0:
                dev_cr += 2.0 * (cr_pred[c_k] - cr_obs[c_k] + cr_obs[c_k] * math.log(cr_obs[c_k] / cr_pred[c_k]))
            else:
                dev_cr += 2.0 * cr_pred[c_k]

        pen = float(np.dot(theta, theta))
        return dev_sr + dev_cr + pen

    bnds = [(-6.0, 6.0)] * 6
    res0 = minimize(lambda th: _nll_2x(th, 0.0), x0=np.zeros(6), method='L-BFGS-B', bounds=bnds)
    res1 = minimize(lambda th: _nll_2x(th, 1.0), x0=np.zeros(6), method='L-BFGS-B', bounds=bnds)
    q0 = float(max(0.0, res0.fun - res1.fun))
    return {
        'z0_unbinned_stat_only': z0_stat,
        'z0_unbinned_profile_likelihood': float(math.sqrt(q0)),
        'q0_unbinned_profile_likelihood': q0,
        'expected_sr_signal': s_tot_sr,
        'expected_sr_background': float(sr_proc_b.sum()),
        'expected_cr_background': float(cr_obs.sum()),
        'optimizer_converged': bool(res0.success and res1.success),
    }


def fit_single_bin_1pct_baseline(
    tau_llr: np.ndarray,
    labels: np.ndarray,
    *,
    has_charm: np.ndarray | None = None,
    mw2t_weights: np.ndarray | None = None,
    total_events: float = 180000.0,
    tau_fraction: float = 0.003,
    fpr_target: float = 0.01,
    delta_sys: float = 0.0159,
    use_pot_gpd: bool = False,
) -> dict[str, Any]:
    """Single-bin `1% FPR` counting significance (stat-only, PL without BB-lite, PL with BB-lite)."""
    llr = np.asarray(tau_llr, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    hc = np.zeros(len(y), dtype=bool) if has_charm is None else np.asarray(has_charm, dtype=bool)
    w = compute_physical_event_weights(
        y, hc, total_events=total_events, tau_fraction=tau_fraction, mw2t_weights=mw2t_weights
    )
    is_bg = y != 2
    if use_pot_gpd:
        gpd = fit_pot_gpd(llr[is_bg])
        thr = pot_gpd_threshold(gpd, fpr_target, scores_bg_fallback=llr[is_bg], use_np_shift=False)
    else:
        thr = float(np.quantile(llr[is_bg], 1.0 - float(fpr_target)))

    sel = llr >= thr
    s_1bin = float(w[sel & (y == 2)].sum())
    b_1bin = float(w[sel & is_bg].sum())
    var_b = float(np.sum(w[sel & is_bg] ** 2))
    tau_eff_mw2t = float(s_1bin / max(float(total_events) * float(tau_fraction), 1e-12))

    if s_1bin <= 0.0 or b_1bin <= 0.0:
        return {
            'threshold': thr,
            'tau_eff_mw2t_pct': tau_eff_mw2t * 100.0,
            'S': s_1bin,
            'B': b_1bin,
            'z0_stat_only': 0.0,
            'z0_pl_no_bb': 0.0,
            'z0_pl_bb_lite': 0.0,
        }

    z0_stat = float(math.sqrt(max(0.0, 2.0 * ((s_1bin + b_1bin) * math.log1p(s_1bin / b_1bin) - s_1bin))))

    def _single_bin_pl(use_bb: bool) -> float:
        dsq = np.array([min(var_b / max(b_1bin * b_1bin, 1e-18), 2.5)], dtype=np.float64)
        n_obs = np.array([s_1bin + b_1bin], dtype=np.float64)

        def _obj(th: np.ndarray) -> float:
            nu = np.array([b_1bin * math.exp(np.clip(delta_sys * float(th[0]), -3.5, 3.5))], dtype=np.float64)
            dev, _, _ = _barlow_beeston_inner_deviance(n_obs, nu, dsq, use_bb)
            return dev + float(th[0] * th[0])

        res0 = minimize(_obj, x0=np.zeros(1), method='L-BFGS-B', bounds=[(-6.0, 6.0)])
        return float(math.sqrt(max(0.0, res0.fun)))

    return {
        'threshold': thr,
        'tau_eff_mw2t_pct': tau_eff_mw2t * 100.0,
        'S': s_1bin,
        'B': b_1bin,
        'S_over_B': float(s_1bin / max(b_1bin, 1e-12)),
        'z0_stat_only': z0_stat,
        'z0_pl_no_bb': _single_bin_pl(False),
        'z0_pl_bb_lite': _single_bin_pl(True),
    }


def evaluate_nsbi_vs_single_bin_closure(
    obs_summary: Mapping[str, Any],
    labels: np.ndarray,
    *,
    has_charm: np.ndarray | None = None,
    mw2t_weights: np.ndarray | None = None,
    total_events: float = 180000.0,
    tau_fraction: float = 0.003,
    delta_sys: float = 0.0159,
    n_subbins_per_channel: int = 4,
    router_cfg: DiscoveryRouterConfig | None = None,
) -> dict[str, Any]:
    """End-to-end Asimov comparison: Golden Micro-Channels + CRs (NSBI-PL) vs 1% FPR single-bin counting."""
    routing = route_golden_micro_channels(obs_summary, cfg=router_cfg)
    tau_llr = np.asarray(obs_summary['tau_llr'], dtype=np.float64)
    ch_id = routing['channel_id']

    tpl = build_nsbi_channel_templates(
        tau_llr,
        ch_id,
        labels,
        has_charm=has_charm,
        mw2t_weights=mw2t_weights,
        total_events=total_events,
        tau_fraction=tau_fraction,
        n_subbins_per_channel=n_subbins_per_channel,
    )
    pl_no_bb = fit_nsbi_profiled_likelihood(tpl, delta_sys=delta_sys, enable_barlow_beeston=False)
    pl_bb = fit_nsbi_profiled_likelihood(tpl, delta_sys=delta_sys, enable_barlow_beeston=True)
    unbinned = fit_unbinned_nsbi_pl(
        tau_llr, ch_id, labels, has_charm=has_charm, mw2t_weights=mw2t_weights,
        total_events=total_events, tau_fraction=tau_fraction, delta_sys=delta_sys,
    )
    single_1pct = fit_single_bin_1pct_baseline(
        tau_llr, labels, has_charm=has_charm, mw2t_weights=mw2t_weights,
        total_events=total_events, tau_fraction=tau_fraction, fpr_target=0.01, delta_sys=delta_sys,
    )

    per_channel = {}
    for cid in DISCOVERY_CHANNEL_IDS + CR_CHANNEL_IDS:
        cm = ch_id == cid
        w_ev = tpl['event_weights']
        s_c = float(w_ev[cm & (np.asarray(labels) == 2)].sum())
        b_c = float(w_ev[cm & (np.asarray(labels) != 2)].sum())
        per_channel[CHANNEL_NAMES[cid]] = {
            'raw_count': int(cm.sum()),
            'expected_S': s_c,
            'expected_B': b_c,
            'S_over_B': float(s_c / max(b_c, 1e-12)) if b_c > 0 else 0.0,
        }

    return {
        'routing_counts': routing['counts'],
        'per_channel_yields': per_channel,
        'single_bin_1pct_fpr': single_1pct,
        'nsbi_pl_no_bb': pl_no_bb,
        'nsbi_pl_bb_lite': pl_bb,
        'nsbi_unbinned_pl': unbinned,
        'z0_gain_factor_pl_no_bb': float(pl_no_bb['z0_profile_likelihood'] / max(single_1pct['z0_pl_no_bb'], 1e-6)),
        'z0_gain_factor_pl_bb_lite': float(pl_bb['z0_profile_likelihood'] / max(single_1pct['z0_pl_bb_lite'], 1e-6)),
        'z0_gain_factor_unbinned_vs_1bin_pl': float(
            unbinned['z0_unbinned_profile_likelihood'] / max(single_1pct['z0_pl_no_bb'], 1e-6)
        ),
        'templates': tpl,
    }


def run_nsbi_toy_mc_closure(
    templates: Mapping[str, Any],
    *,
    delta_sys: float = 0.0159,
    n_toys_h0: int = 300,
    n_toys_h1: int = 300,
    enable_barlow_beeston: bool = True,
    seed: int = 20263250,
) -> dict[str, Any]:
    """Run frequentist H0 (mu=0) and H1 (mu=1) Toy-MC closure verification on NSBI-PL."""
    s_tau = np.asarray(templates['s_tau'], dtype=np.float64)
    b_tot = np.asarray(templates['b_total'], dtype=np.float64)
    var_bkg = np.asarray(templates['var_mc_bkg'], dtype=np.float64)
    n_bins = len(b_tot)
    n_nuis = 10
    dsq = np.where(b_tot > 1e-9, np.clip(var_bkg / np.maximum(b_tot * b_tot, 1e-18), 0.0, 2.5), 0.0)

    rng = np.random.default_rng(seed)

    def _run_batch(mu_true: float, n_toys: int) -> tuple[np.ndarray, np.ndarray]:
        mu_arr = np.full(n_toys, mu_true, dtype=np.float64) * 0.0 + mu_true
        mean_counts = np.maximum(b_tot + mu_arr[0] * s_tau, 0.0)
        counts = rng.poisson(mean_counts[None, :], size=(n_toys, n_bins)).astype(np.float64)
        th_obs = rng.normal(0.0, 1.0, size=(n_toys, n_nuis))
        beta_obs = None
        if enable_barlow_beeston:
            beta_obs = np.ones((n_toys, n_bins), dtype=np.float64)
            for b_i in range(n_bins):
                if dsq[b_i] > 1e-8:
                    m_eff = 1.0 / dsq[b_i]
                    beta_obs[:, b_i] = np.maximum(rng.poisson(m_eff, size=n_toys) / m_eff, 1e-4)

        q0s = np.zeros(n_toys, dtype=np.float64)
        z0s = np.zeros(n_toys, dtype=np.float64)
        for i in range(n_toys):
            r = fit_nsbi_profiled_likelihood(
                templates,
                delta_sys=delta_sys,
                enable_barlow_beeston=enable_barlow_beeston,
                n_obs_override=counts[i],
                theta_obs_override=th_obs[i],
                beta_obs_override=None if beta_obs is None else beta_obs[i],
                fit_signal_strength=True,
            )
            q0s[i] = r['q0_profile_likelihood']
            z0s[i] = r['z0_profile_likelihood']
        return q0s, z0s

    q0_h0, z0_h0 = _run_batch(0.0, n_toys_h0)
    q0_h1, z0_h1 = _run_batch(1.0, n_toys_h1)
    asimov = fit_nsbi_profiled_likelihood(templates, delta_sys=delta_sys, enable_barlow_beeston=enable_barlow_beeston)

    return {
        'asimov_z0_pl': asimov['z0_profile_likelihood'],
        'h0': {
            'n_toys': int(n_toys_h0),
            'zero_fraction': float(np.mean(q0_h0 <= 1e-6)),
            'expected_zero_fraction_asymptotic': 0.50,
            'empirical_p_z_gt_1sigma': float(np.mean(z0_h0 >= 1.0)),
            'asymptotic_p_z_gt_1sigma': float(norm.sf(1.0)),
            'empirical_p_z_gt_2sigma': float(np.mean(z0_h0 >= 2.0)),
            'asymptotic_p_z_gt_2sigma': float(norm.sf(2.0)),
            'q0_p95': float(np.quantile(q0_h0, 0.95)),
            'asymptotic_q0_p95': float(norm.ppf(0.95) ** 2),
        },
        'h1': {
            'n_toys': int(n_toys_h1),
            'median_z0': float(np.median(z0_h1)),
            'mean_z0': float(np.mean(z0_h1)),
            'p16_z0': float(np.quantile(z0_h1, 0.16)),
            'p84_z0': float(np.quantile(z0_h1, 0.84)),
            'asimov_z0_pl': asimov['z0_profile_likelihood'],
            'median_over_asimov_ratio': float(np.median(z0_h1) / max(asimov['z0_profile_likelihood'], 1e-6)),
        },
    }
