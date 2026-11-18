"""Decisive Experiment Exp-6 (Audit P0-1 & Path D): Realistic LHC forward-flux significance & H0 coverage."""

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
from scipy.stats import chi2

from analysis.flavour.inventory import write_json
from analysis.flavour.v2.partitions import FAMILY, KEY
from analysis.flavour.v3.v4_tau.ensemble import extract_member_arrays
from analysis.flavour.v3.v5_confirm import DEFAULT_V3_ROOT, ROLE_ORDER, SEEDS
from analysis.flavour.v3.v5_confirm.flux_significance import (
    LHC_FORWARD_SCENARIOS,
    PHYSICAL_BKG_SHARES,
    PHYSICAL_TAU_FRACTION,
    assign_signal_and_charm_cr_bins,
    build_physical_templates,
    extract_signed_systematic_templates,
    fit_realistic_profile_likelihood,
    run_h0_toy_coverage_verification,
)
from analysis.flavour.v3.v5_confirm.significance import (
    apply_frozen_pipeline,
    compute_locked_slice_report,
    profile_likelihood_significance,
)


def run_exp6(
    v3_root: Path,
    out_dir: Path,
    *,
    n_toys: int = 2000,
    workers: int = 24,
) -> dict:
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    w3_12 = v3_root / 'W3-12'
    meta_dir = w3_12 / 'offline_truth_meta'
    pred_dir = w3_12 / 'predictions'

    cal_a_all = json.loads((w3_12 / 'cal_a_frozen_calibrators.json').read_text())
    cut_b_all = json.loads((w3_12 / 'cut_b_locked_thresholds.json').read_text())
    cal_a_ens = cal_a_all['ens_v5']
    cut_b_ens = cut_b_all['ens_v5']

    ref_conf = pd.read_parquet(meta_dir / 'confirm_truth_meta.parquet').sort_values(KEY).reset_index(drop=True)
    mem_conf: dict[str, pd.DataFrame] = {}
    for role in ROLE_ORDER:
        seed_dfs = []
        for s in SEEDS:
            mid = f'{role}_s{s}'
            df = pd.read_parquet(pred_dir / 'confirm' / f'{mid}.parquet')
            seed_dfs.append(extract_member_arrays(df, ref_conf))
        mean_p = np.mean([a['probs'] for a in seed_dfs], axis=0)
        mean_p /= np.maximum(mean_p.sum(axis=-1, keepdims=True), 1e-12)
        out_r = ref_conf[KEY].copy().reset_index(drop=True)
        for c in range(4):
            out_r[f'p_{c}'] = mean_p[:, c].astype(np.float32)
        ch = [a['charm'] for a in seed_dfs if a['charm'] is not None]
        if ch:
            out_r['charm_prob'] = np.mean(ch, axis=0).astype(np.float32)
        mem_conf[f'{role}_3seed'] = out_r

    applied = apply_frozen_pipeline(mem_conf, ref_conf, cal_a_ens, cut_b_ens)
    labels = ref_conf['label'].to_numpy(dtype=np.int64)
    obs_dom = ref_conf['obs_domain_id'].to_numpy(dtype=np.int64)
    has_charm = ref_conf['has_charm'].to_numpy(dtype=bool)

    # P1-4 exact within-slice vs global-budget FPR verification table across domains
    slice_fpr_audit = {}
    for d_id, d_name in ((0, 'All'), (1, 'D_pix_contained'), (2, 'D_scint_contained'), (3, 'D_partial_or_edge')):
        m = np.ones(len(labels), dtype=bool) if d_id == 0 else (obs_dom == d_id)
        rep = compute_locked_slice_report(labels, applied['probs'], applied['s_tau'], applied, m)
        slice_fpr_audit[d_name] = {
            'count': rep['count'],
            'within_slice_exact_1pct_fpr': rep.get('within_slice_exact_1pct_fpr'),
            'within_slice_exact_0p1pct_fpr': rep.get('within_slice_exact_0p1pct_fpr'),
            'locked_global_uniform_1pct': rep.get('locked_pass_global_1pct'),
            'locked_np_budget_1pct': rep.get('locked_pass_np_1pct'),
            'locked_np_budget_0p1pct': rep.get('locked_pass_np_0p1pct'),
        }

    # Build signed asymmetric systematic templates from paired stress predictions
    stress_pf = pd.read_parquet(pred_dir / 'stress_pf.parquet')
    stress_b1 = pd.read_parquet(pred_dir / 'stress_b1pp.parquet')
    stress_meta = stress_pf[KEY + ['label', 'variant', 'obs_domain_id']].merge(
        stress_b1[KEY + ['variant']], on=KEY + ['variant'], validate='one_to_one'
    )
    st_ref_idx = pd.MultiIndex.from_frame(stress_meta[KEY + ['variant']])
    b1_pos = pd.MultiIndex.from_frame(stress_b1[KEY + ['variant']]).get_indexer(st_ref_idx)
    pf_pos = pd.MultiIndex.from_frame(stress_pf[KEY + ['variant']]).get_indexer(st_ref_idx)
    stress_member_frames: dict[str, pd.DataFrame] = {}
    for role in ROLE_ORDER:
        src = stress_pf.iloc[pf_pos] if role.startswith('pf_') else stress_b1.iloc[b1_pos]
        mp = np.mean([src[[f'{role}_s{s}__p{c}' for c in range(4)]].to_numpy(dtype=np.float64) for s in SEEDS], axis=0)
        mp /= np.maximum(mp.sum(axis=-1, keepdims=True), 1e-12)
        df_s = stress_meta[KEY + ['variant']].copy().reset_index(drop=True)
        for c in range(4):
            df_s[f'p_{c}'] = mp[:, c].astype(np.float32)
        if f'{role}_s42__charm' in src.columns:
            df_s['charm_prob'] = np.mean([src[f'{role}_s{s}__charm'].to_numpy(dtype=np.float64) for s in SEEDS], axis=0).astype(np.float32)
        stress_member_frames[f'{role}_3seed'] = df_s

    from analysis.flavour.v3.v5_confirm.significance import evaluate_14_stress_conditions
    stress_meta_with_fam = stress_meta.merge(ref_conf[KEY + FAMILY], on=KEY, how='left', validate='many_to_one')
    stress_eval = evaluate_14_stress_conditions(
        stress_meta_with_fam, stress_member_frames, cal_a_ens, cut_b_ens, n_boot=100
    )
    signed_sys = extract_signed_systematic_templates(
        stress_meta, np.asarray(stress_eval['stress_bin_ids_tail'], dtype=np.int64), n_bins=18
    )

    # Also extract legacy unsigned 18-bin RMS for direct head-to-head audit comparison
    w3_12_old = json.loads((w3_12 / 'w3_12_confirmation_results.json').read_text())
    delta_rms_old = np.asarray(w3_12_old['confirm200k']['stress_14_conditions']['delta_bkg_sys_18bin_rms'], dtype=np.float64)

    sr_bins_18, joint_bins_24 = assign_signal_and_charm_cr_bins(
        applied['s_tau'],
        applied.get('charm_prob'),
        obs_dom,
        cut_b_ens['template_edges_tail_resolved']['edges_by_domain'],
    )

    scenarios_out = {}
    for scen_name, scen in LHC_FORWARD_SCENARIOS.items():
        tpl_18 = build_physical_templates(
            sr_bins_18,
            labels,
            has_charm,
            n_bins=18,
            total_events=scen['expected_total_events'],
            tau_fraction=scen['tau_fraction'],
            bkg_shares=scen['bkg_shares'],
            nc_charm_fraction_override=0.035,
        )
        tpl_24 = build_physical_templates(
            joint_bins_24,
            labels,
            has_charm,
            n_bins=24,
            total_events=scen['expected_total_events'],
            tau_fraction=scen['tau_fraction'],
            bkg_shares=scen['bkg_shares'],
            nc_charm_fraction_override=0.035,
        )
        legacy_same_ntot = profile_likelihood_significance(
            sr_bins_18,
            labels,
            tau_prior=scen['tau_fraction'],
            benchmark_total=scen['expected_total_events'],
            delta_bkg_sys=delta_rms_old,
        )
        scenarios_out[scen_name] = {
            'scenario': scen,
            'legacy_equal_bkg_unsigned_rms': {
                'z0_stat_only': legacy_same_ntot['z0_stat_only'],
                'z0_profile_likelihood': legacy_same_ntot['z0_profile_likelihood'],
            },
            'physical_sr_18bin_no_bb': fit_realistic_profile_likelihood(
                tpl_18, signed_sys=signed_sys, n_sr_bins=18, enable_barlow_beeston=False
            ),
            'physical_sr_18bin_bb_lite': fit_realistic_profile_likelihood(
                tpl_18, signed_sys=signed_sys, n_sr_bins=18, enable_barlow_beeston=True
            ),
            'physical_sr_plus_charm_cr_24bin_bb_lite': fit_realistic_profile_likelihood(
                tpl_24, signed_sys=signed_sys, n_sr_bins=18, enable_barlow_beeston=True
            ),
        }

    # Continuous luminosity curve from 100 fb^-1 to 10,000 fb^-1
    lumi_grid_fb = [100.0, 250.0, 500.0, 680.0, 1000.0, 1500.0, 2000.0, 3000.0, 5000.0, 10000.0]
    lumi_curve = []
    for lum in lumi_grid_fb:
        ntot = 240200.0 * (lum / 10000.0)
        t18 = build_physical_templates(sr_bins_18, labels, has_charm, n_bins=18, total_events=ntot)
        t24 = build_physical_templates(joint_bins_24, labels, has_charm, n_bins=24, total_events=ntot)
        r18 = fit_realistic_profile_likelihood(t18, signed_sys=signed_sys, n_sr_bins=18, enable_barlow_beeston=True)
        r24 = fit_realistic_profile_likelihood(t24, signed_sys=signed_sys, n_sr_bins=18, enable_barlow_beeston=True)
        leg = profile_likelihood_significance(sr_bins_18, labels, tau_prior=PHYSICAL_TAU_FRACTION, benchmark_total=ntot, delta_bkg_sys=delta_rms_old)
        lumi_curve.append({
            'luminosity_fb_inv': lum,
            'expected_total_events': ntot,
            'expected_nutau_cc': ntot * PHYSICAL_TAU_FRACTION,
            'z0_stat_phys_flux': r24['z0_stat_only'],
            'z0_pl_sr18_bb': r18['z0_profile_likelihood'],
            'z0_pl_sr24_charm_cr_bb': r24['z0_profile_likelihood'],
            'z0_pl_legacy_unsigned_equal_bkg': leg['z0_profile_likelihood'],
        })

    # 2,000-toy frequentist H0 coverage verification at HL-LHC (3 ab^-1) 24-bin SR + Charm-CR
    hl_tpl_24 = build_physical_templates(
        joint_bins_24,
        labels,
        has_charm,
        n_bins=24,
        total_events=LHC_FORWARD_SCENARIOS['HL_LHC_3ab']['expected_total_events'],
    )
    h0_cov = run_h0_toy_coverage_verification(
        hl_tpl_24,
        signed_sys=signed_sys,
        n_sr_bins=18,
        n_toys=n_toys,
        workers=workers,
        enable_barlow_beeston=True,
    )

    # Render publication-grade 2-panel English figure (300 DPI)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), dpi=300)
    ax0, ax1 = axes

    xs = [row['luminosity_fb_inv'] for row in lumi_curve]
    ax0.plot(xs, [r['z0_stat_phys_flux'] for r in lumi_curve], 'o--', color='#4C72B0', lw=1.8, label='Physical flux (stat-only)')
    ax0.plot(xs, [r['z0_pl_sr24_charm_cr_bb'] for r in lumi_curve], 's-', color='#DD8452', lw=2.3, label='Physical flux PL: SR + Charm-CR + BB-lite (24-bin)')
    ax0.plot(xs, [r['z0_pl_sr18_bb'] for r in lumi_curve], '^-', color='#55A868', lw=1.8, label='Physical flux PL: SR only + BB-lite (18-bin)')
    ax0.axhline(3.0, color='#666666', ls=':', lw=1.1, label='3 sigma evidence / 5 sigma discovery')
    ax0.axhline(5.0, color='#222222', ls='-.', lw=1.2)
    for x_m, tag in ((250.0, 'Run 3\n250 fb$^{-1}$'), (680.0, 'Run 4\n680 fb$^{-1}$'), (3000.0, 'HL-LHC\n3 ab$^{-1}$'), (10000.0, 'FPF\n10 ab$^{-1}$')):
        ax0.axvline(x_m, color='#BBBBBB', ls=':', lw=0.9)
        ax0.text(x_m * 1.04, 0.35, tag, fontsize=7.5, color='#333333')
    ax0.set_xscale('log')
    ax0.set_xlabel('Integrated Luminosity [fb$^{-1}$]')
    ax0.set_ylabel('Expected Discovery Significance $Z_0$ [$\\sigma$]')
    ax0.set_title('Forward $\\nu_\\tau$ CC Discovery Projection ($f_\\tau = 0.209\\%$)')
    ax0.grid(True, which='both', alpha=0.25)
    ax0.legend(loc='upper left', fontsize=7.8, frameon=True)

    # Panel B: Frequentist H0 tail exceedance curve vs asymptotic 0.5*chi^2_1
    z_grid = np.linspace(0.1, 3.2, 60)
    # Recompute empirical exceedance on a quick 400-toy sample or plot exact quantiles + markers
    q_grid = z_grid ** 2
    asym_sf = 0.5 * chi2.sf(q_grid, df=1)
    ax1.plot(z_grid, asym_sf, '-', color='#222222', lw=2.0, label='Asymptotic $\\frac{1}{2}\\chi^2_1$ (Wilks)')
    emp_pts_z = [1.0, 2.0, 3.0]
    emp_pts_p = [
        max(h0_cov['empirical_p_z_gt_1sigma'], 1e-4),
        max(h0_cov['empirical_p_z_gt_2sigma'], 1e-4),
        max(h0_cov['empirical_p_z_gt_3sigma'], 1e-4),
    ]
    emp_err = [math.sqrt(p * (1.0 - p) / n_toys) for p in emp_pts_p] if 'math' in globals() else [np.sqrt(p * (1.0 - p) / n_toys) for p in emp_pts_p]
    ax1.errorbar(emp_pts_z, emp_pts_p, yerr=emp_err, fmt='o', color='#C44E52', ms=6.5, capsize=4, label=f'Frequentist $H_0$ Toys ($N={n_toys}$)')
    ax1.set_yscale('log')
    ax1.set_ylim(2e-4, 0.6)
    ax1.set_xlabel('Test Statistic Threshold $Z_0 = \\sqrt{q_0}$ [$\\sigma$]')
    ax1.set_ylabel('Tail Exceedance Probability $P(Z_0 \\geq z \\mid H_0)$')
    ax1.set_title(f'$H_0$ Profile-Likelihood Calibration ($q_0=0$ frac: {h0_cov["zero_fraction"]:.3f})')
    ax1.grid(True, which='both', alpha=0.25)
    ax1.legend(loc='upper right', fontsize=8.5, frameon=True)

    fig.tight_layout()
    fig_path = out_dir / 'exp6_forward_flux_significance_and_h0_coverage.png'
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    report = {
        'UTC': datetime.now(timezone.utc).isoformat(),
        'wall_seconds': float(time.perf_counter() - t0),
        'physical_tau_fraction': PHYSICAL_TAU_FRACTION,
        'physical_bkg_shares': PHYSICAL_BKG_SHARES,
        'legacy_v5_10k_1pct_claim_reference': w3_12_old['confirm200k']['significance_scan']['N_10000'][2],
        'slice_fpr_operating_point_audit_P1_4': slice_fpr_audit,
        'signed_systematic_templates_18bin': signed_sys,
        'lhc_forward_scenarios': scenarios_out,
        'luminosity_curve': lumi_curve,
        'h0_toy_coverage_HL_LHC_24bin': h0_cov,
        'figure_path': str(fig_path),
    }
    out_json = out_dir / 'exp6_realistic_flux_significance_report.json'
    write_json(out_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v3-root', type=Path, default=DEFAULT_V3_ROOT)
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v4_audit_2026-11-18/exp6_flux_significance'),
    )
    parser.add_argument('--n-toys', type=int, default=2000)
    parser.add_argument('--workers', type=int, default=24)
    args = parser.parse_args()
    rep = run_exp6(args.v3_root, args.out_dir, n_toys=args.n_toys, workers=args.workers)
    print(json.dumps({
        'wall_seconds': rep['wall_seconds'],
        'scenarios': {
            k: {
                'N_tot': v['scenario']['expected_total_events'],
                'z0_stat': v['physical_sr_plus_charm_cr_24bin_bb_lite']['z0_stat_only'],
                'z0_sr18_bb': v['physical_sr_18bin_bb_lite']['z0_profile_likelihood'],
                'z0_sr24_charm_cr_bb': v['physical_sr_plus_charm_cr_24bin_bb_lite']['z0_profile_likelihood'],
            }
            for k, v in rep['lhc_forward_scenarios'].items()
        },
        'h0_toy_coverage': rep['h0_toy_coverage_HL_LHC_24bin'],
        'figure_path': rep['figure_path'],
    }, indent=2))


if __name__ == '__main__':
    main()
