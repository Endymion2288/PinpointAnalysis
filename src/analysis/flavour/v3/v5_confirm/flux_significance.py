"""Realistic LHC forward-flux, signed morphing, Charm-CR, and Barlow-Beeston lite PL significance.

Resolves Audit P0-1 and implements Decisive Experiment Exp-6 & Path D:
1. Physical LHC forward-flux compositions (measured Trench GENIE yields at 10 ab^-1:
   CC nue : CC numu : NC : CC nutau = 27949 : 154088 : 57661 : 502 => f_tau = 0.2090%,
   background flavour fractions nue : numu : NC = 11.660% : 64.284% : 24.056%) across
   Run 3 (250 fb^-1), Run 4 (680 fb^-1), HL-LHC (3 ab^-1), and FPF Trench (10 ab^-1).
2. Signed asymmetric Up/Down log-template morphing per systematic source (replacing unsigned
   positive RMS delta_bkg_sys that locked all 18 bins to a single one-sided scalar).
3. Independent per-domain tail-slope nuisances, independent nue/numu & NC/CC flux-ratio
   nuisances, and an unconstrained/30%-prior forward charm rate nuisance (theta_charm).
4. Simultaneous Signal-Region (18 bins) + Charm-Enriched Control-Region (6 bins, 2 per
   active domain at low S_tau & high charm_prob) fit constraining charm in situ (Path D).
5. Exact closed-form inner Barlow-Beeston lite profiling of per-bin MC template statistical
   uncertainties + multi-threaded frequentist H0 toy Monte Carlo coverage verification.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from analysis.flavour.v3.common.domains import OBS_DOMAIN_NAMES


# Documented Trench forward-flux GENIE yields at 10 ab^-1 (/eos/project-f/fasersim-bonn/public/pinpoint/2025-11-26-genie)
TRENCH_10AB_YIELDS = {
    'nue_cc': 27949.0,
    'numu_cc': 154088.0,
    'nc': 57661.0,
    'nutau_cc': 502.0,
}
TRENCH_10AB_TOTAL = sum(TRENCH_10AB_YIELDS.values())  # 240,200.0
PHYSICAL_TAU_FRACTION = TRENCH_10AB_YIELDS['nutau_cc'] / TRENCH_10AB_TOTAL  # 0.002089925
PHYSICAL_BKG_SHARES = {
    'nue_cc': TRENCH_10AB_YIELDS['nue_cc'] / (TRENCH_10AB_TOTAL - TRENCH_10AB_YIELDS['nutau_cc']),
    'numu_cc': TRENCH_10AB_YIELDS['numu_cc'] / (TRENCH_10AB_TOTAL - TRENCH_10AB_YIELDS['nutau_cc']),
    'nc': TRENCH_10AB_YIELDS['nc'] / (TRENCH_10AB_TOTAL - TRENCH_10AB_YIELDS['nutau_cc']),
}

LHC_FORWARD_SCENARIOS = {
    'FASER_Run3_250fb': {
        'luminosity_fb_inv': 250.0,
        'expected_total_events': TRENCH_10AB_TOTAL * (250.0 / 10000.0),
        'tau_fraction': PHYSICAL_TAU_FRACTION,
        'bkg_shares': PHYSICAL_BKG_SHARES,
    },
    'FASER_Run4_680fb': {
        'luminosity_fb_inv': 680.0,
        'expected_total_events': TRENCH_10AB_TOTAL * (680.0 / 10000.0),
        'tau_fraction': PHYSICAL_TAU_FRACTION,
        'bkg_shares': PHYSICAL_BKG_SHARES,
    },
    'HL_LHC_3ab': {
        'luminosity_fb_inv': 3000.0,
        'expected_total_events': TRENCH_10AB_TOTAL * (3000.0 / 10000.0),
        'tau_fraction': PHYSICAL_TAU_FRACTION,
        'bkg_shares': PHYSICAL_BKG_SHARES,
    },
    'FPF_Trench_10ab': {
        'luminosity_fb_inv': 10000.0,
        'expected_total_events': TRENCH_10AB_TOTAL,
        'tau_fraction': PHYSICAL_TAU_FRACTION,
        'bkg_shares': PHYSICAL_BKG_SHARES,
    },
    'Benchmark_10k_PhysFlux': {
        'luminosity_fb_inv': 10000.0 * (10000.0 / TRENCH_10AB_TOTAL),
        'expected_total_events': 10000.0,
        'tau_fraction': PHYSICAL_TAU_FRACTION,
        'bkg_shares': PHYSICAL_BKG_SHARES,
    },
}

SYSTEMATIC_PAIRS = (
    ('pixel_eff', 'pixel_eff_97', 'pixel_eff_93'),
    ('bar_threshold', 'bar_thr_0p10MeV', 'bar_thr_0p25MeV'),
    ('bar_pe_scale', 'bar_pe_plus5', 'bar_pe_minus5'),
    ('bar_attenuation', 'bar_att_plus10', 'bar_att_minus10'),
    ('spec_resolution', 'spec_smear_15um', 'spec_smear_30um'),
    ('spec_efficiency', 'spec_eff_95', 'spec_drop_st1'),
)


def extract_signed_systematic_templates(
    stress_meta: pd.DataFrame,
    bin_ids_stress: np.ndarray,
    *,
    n_bins: int = 18,
    labels_col: str = 'label',
    bkg_shares: dict[str, float] | None = None,
) -> dict[str, dict[str, list[float]]]:
    """Build signed log-ratio shift vectors (up, down) per systematic family across n_bins."""
    shares = PHYSICAL_BKG_SHARES if bkg_shares is None else bkg_shares
    variants = stress_meta['variant'].to_numpy()
    labels = stress_meta[labels_col].to_numpy(dtype=np.int64)

    # Compute flux-weighted background yield per bin for each variant
    def _variant_bkg_hist(vname: str) -> np.ndarray:
        vm = (variants == vname) & (bin_ids_stress >= 0) & (bin_ids_stress < n_bins)
        y = labels[vm]
        b = bin_ids_stress[vm]
        hist = np.zeros(n_bins, dtype=np.float64)
        for cls_id, key in ((0, 'nue_cc'), (1, 'numu_cc'), (3, 'nc')):
            cm = y == cls_id
            n_cls = int((labels[variants == vname] == cls_id).sum())
            if n_cls > 0 and cm.any():
                w_cls = float(shares[key]) / float(n_cls)
                np.add.at(hist, b[cm], w_cls)
        tot = float(hist.sum())
        return hist / max(tot, 1e-12)

    nom = np.maximum(_variant_bkg_hist('nominal'), 1e-6)
    avail = set(variants.tolist())
    signed: dict[str, dict[str, list[float]]] = {}
    for sys_name, v_up, v_dn in SYSTEMATIC_PAIRS:
        if v_up not in avail and v_dn not in avail:
            continue
        h_up = _variant_bkg_hist(v_up) if v_up in avail else nom
        h_dn = _variant_bkg_hist(v_dn) if v_dn in avail else (2.0 * nom - h_up)
        # Signed log-ratio shifts (preserving exact anti-correlations across bins!)
        d_up = np.clip(np.log(np.maximum(h_up, 1e-6) / nom), -0.60, 0.60)
        d_dn = np.clip(np.log(np.maximum(h_dn, 1e-6) / nom), -0.60, 0.60)
        signed[sys_name] = {
            'up_variant': v_up,
            'down_variant': v_dn,
            'delta_log_up': d_up.tolist(),
            'delta_log_down': d_dn.tolist(),
        }
    return signed


def assign_signal_and_charm_cr_bins(
    s_tau: np.ndarray,
    charm_prob: np.ndarray | None,
    obs_dom: np.ndarray,
    sr_edges_by_domain: dict[str, Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Partition events into 18 Signal-Region bins (top 4 quantiles per domain + 2 low bins)
    or split the lowest-S_tau bulk (bins 0..1 of each domain, <= 80% quantile) into a
    disjoint 6-bin Charm-Enriched Control Region (2 charm_prob bins x 3 domains) for Path D.

    Returns:
      sr_only_bins: (N,) in {0..17} (-1 for D_empty)
      joint_24_bins: (N,) in {0..23} where 0..17 are SR (with low-S_tau Charm-tagged events
                     routed to 18..23 CR_charm: 2 bins per domain [medium, high charm_prob]).
    """
    n = len(s_tau)
    sr_bins = np.full(n, -1, dtype=np.int64)
    joint_bins = np.full(n, -1, dtype=np.int64)
    ch = np.zeros(n, dtype=np.float64) if charm_prob is None else np.asarray(charm_prob, dtype=np.float64)

    for d in (1, 2, 3):
        dn = OBS_DOMAIN_NAMES[d]
        dm = obs_dom == d
        if not dm.any():
            continue
        edges = np.asarray(sr_edges_by_domain[dn], dtype=np.float64)
        n_b = len(edges) - 1
        local_idx = np.clip(np.searchsorted(edges[1:-1], s_tau[dm], side='right'), 0, n_b - 1)
        base_b = (d - 1) * n_b + local_idx
        sr_bins[dm] = base_b

        # Route low-S_tau events (local_idx <= 1, i.e. below 80% S_tau quantile, far from signal tail)
        # with elevated charm score into two disjoint Charm-CR bins per domain:
        # CR bin 18 + 2*(d-1) + 0: 0.25 <= charm_prob < 0.55
        # CR bin 18 + 2*(d-1) + 1: charm_prob >= 0.55
        ch_d = ch[dm]
        in_low_stau = local_idx <= 1
        cr_hi = in_low_stau & (ch_d >= 0.55)
        cr_lo = in_low_stau & (ch_d >= 0.25) & (ch_d < 0.55)
        jb = base_b.copy()
        jb[cr_lo] = 3 * n_b + 2 * (d - 1) + 0
        jb[cr_hi] = 3 * n_b + 2 * (d - 1) + 1
        joint_bins[dm] = jb

    return sr_bins, joint_bins


def build_physical_templates(
    bin_ids: np.ndarray,
    labels: np.ndarray,
    has_charm: np.ndarray,
    *,
    n_bins: int,
    total_events: float,
    tau_fraction: float = PHYSICAL_TAU_FRACTION,
    bkg_shares: dict[str, float] | None = None,
    nc_charm_fraction_override: float | None = 0.035,
) -> dict[str, np.ndarray]:
    """Construct physical forward-flux templates & MC variance arrays across n_bins.

    Components:
      s_tau:      nu_tau CC signal
      b_nue:      nu_e CC (light + negligible charm)
      b_numu_l:   nu_mu CC without charm
      b_numu_c:   nu_mu CC with charmed hadron
      b_nc_l:     NC without charm
      b_nc_c:     NC with c-cbar pair (if GENIE input has 0 NC_charm, optionally injects a
                  conservative charm-like tail proxy scaled to nc_charm_fraction_override=3.5%
                  of NC until clean Pythia8 NC-charm simulation shards are loaded).
      var_mc_bkg: Sum of squared event weights per bin for exact Barlow-Beeston lite.
    """
    shares = PHYSICAL_BKG_SHARES if bkg_shares is None else bkg_shares
    valid = (bin_ids >= 0) & (bin_ids < n_bins)
    y = np.asarray(labels, dtype=np.int64)
    hc = np.asarray(has_charm, dtype=bool)

    n_sig_exp = float(total_events) * float(tau_fraction)
    n_bkg_exp = float(total_events) * (1.0 - float(tau_fraction))

    # Event-level physical weights
    w = np.zeros(len(y), dtype=np.float64)
    m_tau = y == 2
    if m_tau.any():
        w[m_tau] = n_sig_exp / float(m_tau.sum())
    for cls_id, key in ((0, 'nue_cc'), (1, 'numu_cc'), (3, 'nc')):
        mc = y == cls_id
        if mc.any():
            w[mc] = (n_bkg_exp * float(shares[key])) / float(mc.sum())

    def _hist_and_var(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m = valid & mask
        h = np.zeros(n_bins, dtype=np.float64)
        v = np.zeros(n_bins, dtype=np.float64)
        if m.any():
            np.add.at(h, bin_ids[m], w[m])
            np.add.at(v, bin_ids[m], w[m] ** 2)
        return h, v

    s_tau, var_s = _hist_and_var(y == 2)
    b_nue, v_nue = _hist_and_var(y == 0)
    b_numu_l, v_numu_l = _hist_and_var((y == 1) & (~hc))
    b_numu_c, v_numu_c = _hist_and_var((y == 1) & hc)
    b_nc_l, v_nc_l = _hist_and_var((y == 3) & (~hc))
    b_nc_c, v_nc_c = _hist_and_var((y == 3) & hc)

    # Audit P2-1 / P0-1 safeguard: legacy GENIE 3.04 G18_02a has 0 NC_charm events!
    # If NC_charm is absent in legacy sample and nc_charm_fraction_override is set, split
    # nc_charm_fraction_override of total NC yield into a charm-shaped template modeled from
    # observed CC charm shape in each domain (hadronic secondary vertex without primary lepton).
    if b_nc_c.sum() == 0.0 and nc_charm_fraction_override is not None and nc_charm_fraction_override > 0.0:
        f_ncc = float(nc_charm_fraction_override)
        tot_nc = float(b_nc_l.sum())
        if tot_nc > 0.0 and b_numu_c.sum() > 0.0:
            # Blend 50% NC kinematics + 50% charm secondary-vertex tail shape normalized to f_ncc * tot_nc
            nc_shape = b_nc_l / max(tot_nc, 1e-12)
            ch_shape = b_numu_c / max(float(b_numu_c.sum()), 1e-12)
            proxy_shape = 0.5 * nc_shape + 0.5 * ch_shape
            proxy_shape /= max(float(proxy_shape.sum()), 1e-12)
            b_nc_c = (f_ncc * tot_nc) * proxy_shape
            b_nc_l = (1.0 - f_ncc) * b_nc_l
            v_nc_c = (f_ncc ** 2) * v_nc_l + ((0.5 * f_ncc * tot_nc / max(float(b_numu_c.sum()), 1.0)) ** 2) * v_numu_c

    var_bkg = v_nue + v_numu_l + v_numu_c + v_nc_l + v_nc_c
    b_tot = b_nue + b_numu_l + b_numu_c + b_nc_l + b_nc_c
    return {
        's_tau': s_tau,
        'b_nue': b_nue,
        'b_numu_light': b_numu_l,
        'b_numu_charm': b_numu_c,
        'b_nc_light': b_nc_l,
        'b_nc_charm': b_nc_c,
        'b_total': b_tot,
        'var_mc_bkg': var_bkg,
        'var_mc_sig': var_s,
    }


def _barlow_beeston_inner_deviance(
    n_obs: np.ndarray,
    nu_pred: np.ndarray,
    delta_sq: np.ndarray,
    enable_bb: bool,
    beta_obs: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Exact closed-form inner profile likelihood over per-bin MC stat multipliers gamma_b > 0.

    Minimizes over gamma_b > 0:
      sum_b [ 2 * (gamma_b * nu_b - n_b + n_b * log(n_b / (gamma_b * nu_b))) + (gamma_b - beta_obs_b)^2 / delta_b^2 ]
    Returns:
      (total_deviance, d_deviance_d_nu, gamma_star)
    """
    pos = nu_pred > 1e-12
    dev = 0.0
    grad_nu = np.zeros_like(nu_pred, dtype=np.float64)
    gamma_star = np.ones_like(nu_pred, dtype=np.float64)
    if not pos.any():
        return dev, grad_nu, gamma_star

    nu = nu_pred[pos]
    n = n_obs[pos]
    dsq = delta_sq[pos] if enable_bb else np.zeros_like(nu)
    b_obs = np.ones_like(nu) if beta_obs is None else np.asarray(beta_obs, dtype=np.float64)[pos]

    use_bb = enable_bb & (dsq > 1e-10)
    if np.any(use_bb):
        b_term = b_obs[use_bb] - dsq[use_bb] * nu[use_bb]
        disc = np.sqrt(np.maximum(b_term * b_term + 4.0 * dsq[use_bb] * n[use_bb], 1e-30))
        g_bb = np.maximum(0.5 * (b_term + disc), 1e-8)
        g = np.ones_like(nu)
        g[use_bb] = g_bb
    else:
        g = np.ones_like(nu)

    eff_nu = g * nu
    with np.errstate(divide='ignore', invalid='ignore'):
        pois = np.where(n > 0.0, 2.0 * (eff_nu - n + n * np.log(n / eff_nu)), 2.0 * eff_nu)
        pen = np.where(use_bb, ((g - b_obs) ** 2) / np.maximum(dsq, 1e-30), 0.0)
        d_dnu = np.where(n > 0.0, 2.0 * (g - n / nu), 2.0 * g)

    dev = float(np.sum(pois + pen))
    grad_nu[pos] = d_dnu
    gamma_star[pos] = g
    return dev, grad_nu, gamma_star


def fit_realistic_profile_likelihood(
    templates: dict[str, np.ndarray],
    *,
    signed_sys: dict[str, dict[str, list[float]]] | None = None,
    n_sr_bins: int = 18,
    sigma_lumi: float = 0.05,
    sigma_dom: float = 0.06,
    sigma_dom_slope: float = 0.08,
    sigma_nue_ratio: float = 0.10,
    sigma_nc_ratio: float = 0.10,
    sigma_charm: float = 0.30,
    enable_barlow_beeston: bool = True,
    n_obs_override: np.ndarray | None = None,
    theta_obs_override: np.ndarray | None = None,
    beta_obs_override: np.ndarray | None = None,
    fit_signal_strength: bool = False,
) -> dict:
    """Profile likelihood under H0 (mu=0) [and optionally free mu>=0] with full nuisance model."""
    s_tau = np.asarray(templates['s_tau'], dtype=np.float64)
    b_nue = np.asarray(templates['b_nue'], dtype=np.float64)
    b_numu_l = np.asarray(templates['b_numu_light'], dtype=np.float64)
    b_numu_c = np.asarray(templates['b_numu_charm'], dtype=np.float64)
    b_nc_l = np.asarray(templates['b_nc_light'], dtype=np.float64)
    b_nc_c = np.asarray(templates['b_nc_charm'], dtype=np.float64)
    b_tot = b_nue + b_numu_l + b_numu_c + b_nc_l + b_nc_c
    var_bkg = np.asarray(templates['var_mc_bkg'], dtype=np.float64)

    n_bins = len(b_tot)
    n_obs = (s_tau + b_tot) if n_obs_override is None else np.asarray(n_obs_override, dtype=np.float64)
    delta_sq = var_bkg / np.maximum(b_tot * b_tot, 1e-18)
    delta_sq = np.where(b_tot > 1e-9, np.clip(delta_sq, 0.0, 2.0), 0.0)

    # Stat-only Asimov Z0
    pos_sb = (s_tau > 0.0) & (b_tot > 0.0)
    z0_sq_stat = float(
        np.sum(
            2.0 * ((s_tau[pos_sb] + b_tot[pos_sb]) * np.log(1.0 + s_tau[pos_sb] / b_tot[pos_sb]) - s_tau[pos_sb])
        )
    )
    z0_stat = float(math.sqrt(max(0.0, z0_sq_stat)))

    # Domain assignment & within-domain centered slope basis z_tilde in [-1, +1]
    bins_per_dom = n_sr_bins // 3
    dom_of_bin = np.zeros(n_bins, dtype=np.int64)
    slope_basis = np.zeros((3, n_bins), dtype=np.float64)
    for d_idx in range(3):
        sr_slice = slice(d_idx * bins_per_dom, (d_idx + 1) * bins_per_dom)
        dom_of_bin[sr_slice] = d_idx
        slope_basis[d_idx, sr_slice] = np.linspace(-1.0, 1.0, bins_per_dom)
    if n_bins > n_sr_bins:
        n_cr_per_dom = (n_bins - n_sr_bins) // 3
        for d_idx in range(3):
            cr_slice = slice(n_sr_bins + d_idx * n_cr_per_dom, n_sr_bins + (d_idx + 1) * n_cr_per_dom)
            dom_of_bin[cr_slice] = d_idx
            # CR bins sit in the low-S_tau bulk (slope basis = -0.8)
            slope_basis[d_idx, cr_slice] = -0.8

    # Signed systematic up/down matrices (M_sys, n_bins)
    sys_names: list[str] = []
    up_list, dn_list = [], []
    if signed_sys:
        for k, spec in signed_sys.items():
            u = np.asarray(spec['delta_log_up'], dtype=np.float64)
            d = np.asarray(spec['delta_log_down'], dtype=np.float64)
            if len(u) < n_bins:
                u = np.pad(u, (0, n_bins - len(u)), mode='constant')
                d = np.pad(d, (0, n_bins - len(d)), mode='constant')
            sys_names.append(k)
            up_list.append(u[:n_bins])
            dn_list.append(d[:n_bins])
    m_sys = len(sys_names)
    up_mat = np.stack(up_list, axis=0) if m_sys > 0 else np.zeros((0, n_bins), dtype=np.float64)
    dn_mat = np.stack(dn_list, axis=0) if m_sys > 0 else np.zeros((0, n_bins), dtype=np.float64)

    # Parameter vector theta layout (10 + m_sys):
    # 0: eta_lumi
    # 1..3: eta_dom[0..2]
    # 4..6: beta_dom_slope[0..2]
    # 7: eta_nue
    # 8: eta_nc
    # 9: theta_charm
    # 10..10+m_sys-1: alpha_sys[0..m_sys-1]
    n_nuis = 10 + m_sys
    theta_obs = (
        np.zeros(n_nuis, dtype=np.float64)
        if theta_obs_override is None
        else np.asarray(theta_obs_override, dtype=np.float64)
    )

    def _eval_rates_and_jac(theta: np.ndarray, mu: float) -> tuple[np.ndarray, np.ndarray]:
        eta_lumi = float(theta[0])
        eta_dom = theta[1:4]
        beta_sl = theta[4:7]
        eta_nue = float(theta[7])
        eta_nc = float(theta[8])
        th_ch = float(theta[9])
        alpha_s = theta[10:]

        # Smooth HistFactory-like C^inf asymmetric morphing:
        # shift_m(alpha) = 0.5 * alpha * ((u - d) + tanh(4 * alpha) * (u + d))
        # At alpha = +1 -> ~ +u; at alpha = -1 -> ~ +d; at alpha = 0 -> 0 with slope 0.5*(u - d)
        if m_sys > 0:
            th4 = np.tanh(4.0 * alpha_s)  # (M,)
            sech2 = 1.0 - th4 * th4
            diff_ud = up_mat - dn_mat
            sum_ud = up_mat + dn_mat
            morph = np.sum(
                0.5 * alpha_s[:, None] * (diff_ud + th4[:, None] * sum_ud),
                axis=0,
            )
            dmorph_dalpha = 0.5 * (diff_ud + (th4[:, None] + 4.0 * alpha_s[:, None] * sech2[:, None]) * sum_ud)
        else:
            morph = 0.0
            dmorph_dalpha = np.zeros((0, n_bins), dtype=np.float64)

        common_log = (
            sigma_lumi * eta_lumi
            + sigma_dom * eta_dom[dom_of_bin]
            + sigma_dom_slope * np.tensordot(beta_sl, slope_basis, axes=(0, 0))
            + morph
        )
        common_log = np.clip(common_log, -3.5, 3.5)

        r_nue = b_nue * np.exp(np.clip(common_log + sigma_nue_ratio * eta_nue, -4.0, 4.0))
        r_numu_l = b_numu_l * np.exp(common_log)
        r_numu_c = b_numu_c * np.exp(np.clip(common_log + sigma_charm * th_ch, -4.0, 4.0))
        r_nc_l = b_nc_l * np.exp(np.clip(common_log + sigma_nc_ratio * eta_nc, -4.0, 4.0))
        r_nc_c = b_nc_c * np.exp(np.clip(common_log + sigma_nc_ratio * eta_nc + sigma_charm * th_ch, -4.0, 4.0))
        r_sig = mu * s_tau * np.exp(np.clip(sigma_lumi * eta_lumi + sigma_dom * eta_dom[dom_of_bin] + morph, -3.5, 3.5))

        b_pred = r_nue + r_numu_l + r_numu_c + r_nc_l + r_nc_c
        nu_tot = b_pred + r_sig

        # Jacobian d(nu_tot)/d(theta) of shape (n_nuis, n_bins)
        jac = np.zeros((n_nuis, n_bins), dtype=np.float64)
        jac[0] = sigma_lumi * nu_tot
        for di in range(3):
            dm = dom_of_bin == di
            jac[1 + di, dm] = sigma_dom * nu_tot[dm]
            jac[4 + di] = sigma_dom_slope * slope_basis[di] * b_pred
        jac[7] = sigma_nue_ratio * r_nue
        jac[8] = sigma_nc_ratio * (r_nc_l + r_nc_c)
        jac[9] = sigma_charm * (r_numu_c + r_nc_c)
        if m_sys > 0:
            jac[10:] = dmorph_dalpha * nu_tot[None, :]
        return nu_tot, jac

    def _obj_h0(theta: np.ndarray) -> tuple[float, np.ndarray]:
        nu_tot, jac = _eval_rates_and_jac(theta, mu=0.0)
        dev, grad_nu, _ = _barlow_beeston_inner_deviance(
            n_obs, nu_tot, delta_sq, enable_barlow_beeston, beta_obs=beta_obs_override
        )
        diff = theta - theta_obs
        val = dev + float(np.dot(diff, diff))
        grad = jac @ grad_nu + 2.0 * diff
        return val, grad

    bounds_h0 = [(-6.0, 6.0)] * n_nuis
    res0 = minimize(
        _obj_h0,
        x0=np.clip(theta_obs, -4.0, 4.0),
        jac=True,
        method='L-BFGS-B',
        bounds=bounds_h0,
        options={'gtol': 1e-9, 'ftol': 1e-13, 'maxiter': 500},
    )
    if not res0.success:
        res0_alt = minimize(
            _obj_h0,
            x0=np.zeros(n_nuis, dtype=np.float64),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds_h0,
            options={'gtol': 1e-9, 'ftol': 1e-14, 'maxiter': 1000},
        )
        if res0_alt.fun < res0.fun:
            res0 = res0_alt

    val_h0 = float(res0.fun)
    val_hat = 0.0
    mu_hat = 1.0

    if fit_signal_strength or n_obs_override is not None:
        def _obj_free(x: np.ndarray) -> tuple[float, np.ndarray]:
            mu = float(x[0])
            th = x[1:]
            nu_tot, jac_th = _eval_rates_and_jac(th, mu=mu)
            dev, grad_nu, _ = _barlow_beeston_inner_deviance(
                n_obs, nu_tot, delta_sq, enable_barlow_beeston, beta_obs=beta_obs_override
            )
            # d(nu_tot)/d(mu)
            eta_lumi = float(th[0])
            eta_dom = th[1:4]
            alpha_s = th[10:]
            if m_sys > 0:
                th4 = np.tanh(4.0 * alpha_s)
                morph = np.sum(0.5 * alpha_s[:, None] * ((up_mat - dn_mat) + th4[:, None] * (up_mat + dn_mat)), axis=0)
            else:
                morph = 0.0
            ds_dmu = s_tau * np.exp(np.clip(sigma_lumi * eta_lumi + sigma_dom * eta_dom[dom_of_bin] + morph, -3.5, 3.5))
            diff = th - theta_obs
            g_mu = float(np.dot(grad_nu, ds_dmu))
            g_th = jac_th @ grad_nu + 2.0 * diff
            return dev + float(np.dot(diff, diff)), np.concatenate([[g_mu], g_th])

        if n_obs_override is None:
            x_init = np.zeros(1 + n_nuis, dtype=np.float64)
            x_init[0] = 1.0
            res_free = minimize(
                _obj_free,
                x0=x_init,
                jac=True,
                method='L-BFGS-B',
                bounds=[(0.0, 20.0)] + bounds_h0,
                options={'gtol': 1e-9, 'ftol': 1e-13, 'maxiter': 500},
            )
            mu_hat = float(res_free.x[0])
            val_hat = float(res_free.fun)
        else:
            # Exact KKT check at H0 optimum (mu=0, theta=res0.x):
            # If dL/dmu(0, theta_hat_0) >= 0, mu_hat = 0 is the exact global constrained optimum!
            x_kkt = np.concatenate([[0.0], res0.x])
            _, grad_kkt = _obj_free(x_kkt)
            if float(grad_kkt[0]) >= -1e-7:
                mu_hat = 0.0
                val_hat = val_h0
            else:
                x_init = np.concatenate([[1e-3], res0.x])
                res_free = minimize(
                    _obj_free,
                    x0=x_init,
                    jac=True,
                    method='L-BFGS-B',
                    bounds=[(0.0, 20.0)] + bounds_h0,
                    options={'gtol': 1e-9, 'ftol': 1e-13, 'maxiter': 500},
                )
                mu_hat = float(res_free.x[0])
                val_hat = float(res_free.fun)

    q0 = float(max(0.0, val_h0 - val_hat)) if mu_hat > 0.0 else 0.0
    z0_pl = float(math.sqrt(q0))

    nu_h0, _ = _eval_rates_and_jac(res0.x, mu=0.0)
    _, _, gamma_h0 = _barlow_beeston_inner_deviance(n_obs, nu_h0, delta_sq, enable_barlow_beeston)

    nuis_dict = {
        'eta_lumi': float(res0.x[0]),
        'eta_D_pix': float(res0.x[1]),
        'eta_D_scint': float(res0.x[2]),
        'eta_D_partial': float(res0.x[3]),
        'slope_D_pix': float(res0.x[4]),
        'slope_D_scint': float(res0.x[5]),
        'slope_D_partial': float(res0.x[6]),
        'eta_nue_flux': float(res0.x[7]),
        'eta_nc_flux': float(res0.x[8]),
        'theta_charm_rate': float(res0.x[9]),
    }
    for idx_s, sname in enumerate(sys_names):
        nuis_dict[f'sys_{sname}'] = float(res0.x[10 + idx_s])

    return {
        'n_bins': int(n_bins),
        'expected_signal': float(s_tau.sum()),
        'expected_background': float(b_tot.sum()),
        'expected_charm_background': float((b_numu_c + b_nc_c).sum()),
        'z0_stat_only': z0_stat,
        'z0_profile_likelihood': z0_pl,
        'q0_profile_likelihood': q0,
        'mu_hat': mu_hat,
        'optimizer_converged': bool(res0.success),
        'enable_barlow_beeston': bool(enable_barlow_beeston),
        'min_gamma_mc_stat_at_H0': float(gamma_h0.min()),
        'max_gamma_mc_stat_at_H0': float(gamma_h0.max()),
        'nuisance_pulls_at_H0': nuis_dict,
    }


def run_h0_toy_coverage_verification(
    templates: dict[str, np.ndarray],
    *,
    signed_sys: dict[str, dict[str, list[float]]] | None = None,
    n_sr_bins: int = 18,
    n_toys: int = 2000,
    seed: int = 20261118,
    workers: int = 16,
    enable_barlow_beeston: bool = True,
) -> dict:
    """Frequentist H0 toy Monte Carlo verification of q0 asymptotic chi^2_1/2 calibration.

    Draws nuisance vectors theta_true ~ N(0, I), MC bin multipliers gamma_true ~ TruncNorm(1, delta_b),
    and Poisson counts n_toy ~ Poisson(nu_H0(theta_true) * gamma_true), then computes full profiled
    q0 = L_H0 - L_mu>=0 on every toy in parallel.
    """
    b_nue = np.asarray(templates['b_nue'], dtype=np.float64)
    b_numu_l = np.asarray(templates['b_numu_light'], dtype=np.float64)
    b_numu_c = np.asarray(templates['b_numu_charm'], dtype=np.float64)
    b_nc_l = np.asarray(templates['b_nc_light'], dtype=np.float64)
    b_nc_c = np.asarray(templates['b_nc_charm'], dtype=np.float64)
    b_tot = b_nue + b_numu_l + b_numu_c + b_nc_l + b_nc_c
    var_bkg = np.asarray(templates['var_mc_bkg'], dtype=np.float64)
    n_bins = len(b_tot)
    delta = np.where(b_tot > 1e-9, np.sqrt(np.clip(var_bkg / np.maximum(b_tot ** 2, 1e-18), 0.0, 2.0)), 0.0)

    bins_per_dom = n_sr_bins // 3
    dom_of_bin = np.zeros(n_bins, dtype=np.int64)
    slope_basis = np.zeros((3, n_bins), dtype=np.float64)
    for d_idx in range(3):
        sr_slice = slice(d_idx * bins_per_dom, (d_idx + 1) * bins_per_dom)
        dom_of_bin[sr_slice] = d_idx
        slope_basis[d_idx, sr_slice] = np.linspace(-1.0, 1.0, bins_per_dom)
    if n_bins > n_sr_bins:
        n_cr_per_dom = (n_bins - n_sr_bins) // 3
        for d_idx in range(3):
            cr_slice = slice(n_sr_bins + d_idx * n_cr_per_dom, n_sr_bins + (d_idx + 1) * n_cr_per_dom)
            dom_of_bin[cr_slice] = d_idx
            slope_basis[d_idx, cr_slice] = -0.8

    up_list, dn_list = [], []
    if signed_sys:
        for spec in signed_sys.values():
            u = np.asarray(spec['delta_log_up'], dtype=np.float64)
            d = np.asarray(spec['delta_log_down'], dtype=np.float64)
            if len(u) < n_bins:
                u = np.pad(u, (0, n_bins - len(u)))
                d = np.pad(d, (0, n_bins - len(d)))
            up_list.append(u[:n_bins])
            dn_list.append(d[:n_bins])
    m_sys = len(up_list)
    up_mat = np.stack(up_list, axis=0) if m_sys > 0 else np.zeros((0, n_bins), dtype=np.float64)
    dn_mat = np.stack(dn_list, axis=0) if m_sys > 0 else np.zeros((0, n_bins), dtype=np.float64)

    rng = np.random.default_rng(seed)
    m_sys = len(signed_sys) if signed_sys else 0
    theta_obs_all = rng.normal(0.0, 1.0, size=(n_toys, 10 + m_sys))
    toy_counts = rng.poisson(np.maximum(b_tot, 0.0)[None, :], size=(n_toys, n_bins)).astype(np.float64)
    if enable_barlow_beeston:
        dsq = np.clip(delta * delta, 0.0, 2.0)
        beta_obs_all = np.ones((n_toys, n_bins), dtype=np.float64)
        for b_idx in range(n_bins):
            if dsq[b_idx] > 1e-8:
                m_eff = 1.0 / dsq[b_idx]
                beta_obs_all[:, b_idx] = np.maximum(rng.poisson(m_eff, size=n_toys) / m_eff, 1e-4)
    else:
        beta_obs_all = None

    def _solve_chunk(idx_range: tuple[int, int]) -> list[tuple[float, float]]:
        out_c = []
        for idx in range(idx_range[0], idx_range[1]):
            r = fit_realistic_profile_likelihood(
                templates,
                signed_sys=signed_sys,
                n_sr_bins=n_sr_bins,
                enable_barlow_beeston=enable_barlow_beeston,
                n_obs_override=toy_counts[idx],
                theta_obs_override=theta_obs_all[idx],
                beta_obs_override=None if beta_obs_all is None else beta_obs_all[idx],
                fit_signal_strength=True,
            )
            out_c.append((r['q0_profile_likelihood'], r['z0_profile_likelihood']))
        return out_c

    step = max(1, math.ceil(n_toys / max(1, workers)))
    ranges = [(s, min(n_toys, s + step)) for s in range(0, n_toys, step)]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = [item for sub in pool.map(_solve_chunk, ranges) for item in sub]

    q0_arr = np.array([x[0] for x in results], dtype=np.float64)
    z0_arr = np.array([x[1] for x in results], dtype=np.float64)

    return {
        'n_toys': int(n_toys),
        'zero_fraction': float(np.mean(q0_arr <= 1e-6)),
        'expected_zero_fraction_asymptotic': 0.50,
        'empirical_p_z_gt_1sigma': float(np.mean(z0_arr >= 1.0)),
        'asymptotic_p_z_gt_1sigma': float(norm.sf(1.0)),
        'empirical_p_z_gt_2sigma': float(np.mean(z0_arr >= 2.0)),
        'asymptotic_p_z_gt_2sigma': float(norm.sf(2.0)),
        'empirical_p_z_gt_3sigma': float(np.mean(z0_arr >= 3.0)),
        'asymptotic_p_z_gt_3sigma': float(norm.sf(3.0)),
        'q0_quantiles': {
            'p50': float(np.quantile(q0_arr, 0.50)),
            'p90': float(np.quantile(q0_arr, 0.90)),
            'p95': float(np.quantile(q0_arr, 0.95)),
            'p99': float(np.quantile(q0_arr, 0.99)),
            'max': float(np.max(q0_arr)),
        },
        'asymptotic_chi2_half_quantiles': {
            'p90': float(norm.ppf(0.90) ** 2),
            'p95': float(norm.ppf(0.95) ** 2),
            'p99': float(norm.ppf(0.99) ** 2),
        },
    }
