"""Decisive Experiment Exp-6 (Audit P0-1 & Path D): Realistic LHC forward-flux significance & H0 coverage."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
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
    FIDUCIAL_2T_BKG_SHARES,
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
    _assign_18_bins,
    _fit_bin_edges,
    apply_frozen_pipeline,
    compute_locked_slice_report,
    evaluate_14_stress_conditions,
    profile_likelihood_significance,
)

ULTRA_TAIL_QUANTILES = (0.0, 0.50, 0.85, 0.95, 0.985, 0.995, 0.9985, 1.0)


def _load_role3_split(w3_12: Path, split: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    ref = pd.read_parquet(w3_12 / 'offline_truth_meta' / f'{split}_truth_meta.parquet').sort_values(KEY).reset_index(drop=True)
    mem: dict[str, pd.DataFrame] = {}
    for role in ROLE_ORDER:
        seed_dfs = [extract_member_arrays(pd.read_parquet(w3_12 / 'predictions' / split / f'{role}_s{s}.parquet'), ref) for s in SEEDS]
        mean_p = np.mean([a['probs'] for a in seed_dfs], axis=0)
        mean_p /= np.maximum(mean_p.sum(axis=-1, keepdims=True), 1e-12)
        out_r = ref[KEY].copy().reset_index(drop=True)
        for c in range(4):
            out_r[f'p_{c}'] = mean_p[:, c].astype(np.float32)
        ch = [a['charm'] for a in seed_dfs if a['charm'] is not None]
        if ch:
            out_r['charm_prob'] = np.mean(ch, axis=0).astype(np.float32)
        mem[f'{role}_3seed'] = out_r
    return ref, mem


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
    pred_dir = w3_12 / 'predictions'

    cal_a_ens = json.loads((w3_12 / 'cal_a_frozen_calibrators.json').read_text())['ens_v5']
    cut_b_ens = json.loads((w3_12 / 'cut_b_locked_thresholds.json').read_text())['ens_v5']

    # Lock 21-bin ultra-tail edges strictly on Cut-B160k (zero Confirm200k leakage!)
    ref_cut_b, mem_cut_b = _load_role3_split(w3_12, 'cut_b')
    app_cut_b = apply_frozen_pipeline(mem_cut_b, ref_cut_b, cal_a_ens, cut_b_ens)
    ultra_edges_by_dom = _fit_bin_edges(
        app_cut_b['s_tau'],
        ref_cut_b['obs_domain_id'].to_numpy(dtype=np.int64),
        ULTRA_TAIL_QUANTILES,
    )

    ref_conf, mem_conf = _load_role3_split(w3_12, 'confirm')
    applied = apply_frozen_pipeline(mem_conf, ref_conf, cal_a_ens, cut_b_ens)
    labels = ref_conf['label'].to_numpy(dtype=np.int64)
    obs_dom = ref_conf['obs_domain_id'].to_numpy(dtype=np.int64)
    has_charm = ref_conf['has_charm'].to_numpy(dtype=bool)
    fid_mask = ref_conf['large_scintillator'].to_numpy(dtype=bool) & ref_conf['is_E_le_2p3TeV'].to_numpy(dtype=bool)

    # P1-4 exact within-slice vs global-budget FPR verification table across slices
    slice_fpr_audit = {}
    for s_name, m in (
        ('All', np.ones(len(labels), dtype=bool)),
        ('D_pix_contained', obs_dom == 1),
        ('D_scint_contained', obs_dom == 2),
        ('D_partial_or_edge', obs_dom == 3),
        ('Truth_pixel_centered', ref_conf['pixel_centered'].to_numpy(dtype=bool)),
    ):
        rep = compute_locked_slice_report(labels, applied['probs'], applied['s_tau'], applied, m)
        slice_fpr_audit[s_name] = {
            'count': rep['count'],
            'within_slice_exact_1pct_fpr': rep.get('within_slice_exact_1pct_fpr'),
            'within_slice_exact_0p1pct_fpr': rep.get('within_slice_exact_0p1pct_fpr'),
            'locked_global_uniform_1pct': rep.get('locked_pass_global_1pct'),
            'locked_np_budget_1pct': rep.get('locked_pass_np_1pct'),
            'locked_np_budget_0p1pct': rep.get('locked_pass_np_0p1pct'),
        }

    # Build signed asymmetric systematic templates for ALL 8 physical detector stress families
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

    stress_meta_with_fam = stress_meta.merge(ref_conf[KEY + FAMILY], on=KEY, how='left', validate='many_to_one')
    stress_eval = evaluate_14_stress_conditions(
        stress_meta_with_fam, stress_member_frames, cal_a_ens, cut_b_ens, n_boot=50
    )
    stress_bin_18 = np.asarray(stress_eval['stress_bin_ids_tail'], dtype=np.int64)
    signed_sys_18 = extract_signed_systematic_templates(stress_meta, stress_bin_18, n_bins=18)

    # Also compute stress s_tau + charm_prob for the 21-bin + 6-CR = 27-bin ultra-tail layout
    cut_b_ultra = dict(cut_b_ens)
    cut_b_ultra['template_edges_tail_resolved'] = {
        'quantiles': list(ULTRA_TAIL_QUANTILES),
        'n_bins_per_domain': 7,
        'edges_by_domain': ultra_edges_by_dom,
    }
    stress_eval_ultra = evaluate_14_stress_conditions(
        stress_meta_with_fam, stress_member_frames, cal_a_ens, cut_b_ultra, n_boot=10
    )
    signed_sys_21 = extract_signed_systematic_templates(
        stress_meta, np.asarray(stress_eval_ultra['stress_bin_ids_tail'], dtype=np.int64), n_bins=21
    )

    w3_12_old = json.loads((w3_12 / 'w3_12_confirmation_results.json').read_text())
    delta_rms_old = np.asarray(w3_12_old['confirm200k']['stress_14_conditions']['delta_bkg_sys_18bin_rms'], dtype=np.float64)

    sr_bins_18, joint_bins_24 = assign_signal_and_charm_cr_bins(
        applied['s_tau'],
        applied.get('charm_prob'),
        obs_dom,
        cut_b_ens['template_edges_tail_resolved']['edges_by_domain'],
    )
    sr_bins_21, joint_bins_27 = assign_signal_and_charm_cr_bins(
        applied['s_tau'],
        applied.get('charm_prob'),
        obs_dom,
        ultra_edges_by_dom,
    )

    scenarios_out = {}
    for scen_name, scen in LHC_FORWARD_SCENARIOS.items():
        use_fid = bool(scen.get('fiducial_slice', False))
        m_sc = fid_mask if use_fid else np.ones(len(labels), dtype=bool)
        s_sys18 = extract_signed_systematic_templates(stress_meta, stress_bin_18, n_bins=18, bkg_shares=scen['bkg_shares'])
        s_sys21 = extract_signed_systematic_templates(
            stress_meta, np.asarray(stress_eval_ultra['stress_bin_ids_tail'], dtype=np.int64), n_bins=21, bkg_shares=scen['bkg_shares']
        )

        tpl_18 = build_physical_templates(
            sr_bins_18[m_sc], labels[m_sc], has_charm[m_sc],
            n_bins=18, total_events=scen['expected_total_events'],
            tau_fraction=scen['tau_fraction'], bkg_shares=scen['bkg_shares'],
        )
        tpl_24 = build_physical_templates(
            joint_bins_24[m_sc], labels[m_sc], has_charm[m_sc],
            n_bins=24, total_events=scen['expected_total_events'],
            tau_fraction=scen['tau_fraction'], bkg_shares=scen['bkg_shares'],
        )
        tpl_27_ultra = build_physical_templates(
            joint_bins_27[m_sc], labels[m_sc], has_charm[m_sc],
            n_bins=27, total_events=scen['expected_total_events'],
            tau_fraction=scen['tau_fraction'], bkg_shares=scen['bkg_shares'],
        )
        leg = profile_likelihood_significance(
            sr_bins_18[m_sc], labels[m_sc],
            tau_prior=scen['tau_fraction'], benchmark_total=scen['expected_total_events'], delta_bkg_sys=delta_rms_old,
        )
        scenarios_out[scen_name] = {
            'scenario': scen,
            'legacy_equal_bkg_unsigned_rms_18bin': {
                'z0_stat_only': leg['z0_stat_only'],
                'z0_profile_likelihood': leg['z0_profile_likelihood'],
            },
            'physical_sr_18bin_bb_lite': fit_realistic_profile_likelihood(
                tpl_18, signed_sys=s_sys18, n_sr_bins=18, enable_barlow_beeston=True
            ),
            'physical_sr_plus_charm_cr_24bin_bb_lite': fit_realistic_profile_likelihood(
                tpl_24, signed_sys=s_sys18, n_sr_bins=18, enable_barlow_beeston=True
            ),
            'physical_ultra_tail_sr21_plus_charm_cr_27bin_bb_lite': fit_realistic_profile_likelihood(
                tpl_27_ultra, signed_sys=s_sys21, n_sr_bins=21, enable_barlow_beeston=True
            ),
        }

    # Continuous luminosity curve for 2t Fiducial Pinpoint target (100 fb^-1 to 10,000 fb^-1; 60 events / fb^-1)
    lumi_grid_fb = [100.0, 250.0, 500.0, 680.0, 1000.0, 1500.0, 2000.0, 3000.0, 5000.0, 10000.0]
    s_sys18_fid = extract_signed_systematic_templates(stress_meta, stress_bin_18, n_bins=18, bkg_shares=FIDUCIAL_2T_BKG_SHARES)
    s_sys21_fid = extract_signed_systematic_templates(
        stress_meta, np.asarray(stress_eval_ultra['stress_bin_ids_tail'], dtype=np.int64), n_bins=21, bkg_shares=FIDUCIAL_2T_BKG_SHARES
    )
    lumi_curve = []
    for lum in lumi_grid_fb:
        nfid = 60.0 * lum  # 15,000 at 250 fb^-1, 180,000 at 3,000 fb^-1 (matching Stage 314 2t W target!)
        t18 = build_physical_templates(sr_bins_18[fid_mask], labels[fid_mask], has_charm[fid_mask], n_bins=18, total_events=nfid, tau_fraction=0.003, bkg_shares=FIDUCIAL_2T_BKG_SHARES)
        t24 = build_physical_templates(joint_bins_24[fid_mask], labels[fid_mask], has_charm[fid_mask], n_bins=24, total_events=nfid, tau_fraction=0.003, bkg_shares=FIDUCIAL_2T_BKG_SHARES)
        t27 = build_physical_templates(joint_bins_27[fid_mask], labels[fid_mask], has_charm[fid_mask], n_bins=27, total_events=nfid, tau_fraction=0.003, bkg_shares=FIDUCIAL_2T_BKG_SHARES)
        r18 = fit_realistic_profile_likelihood(t18, signed_sys=s_sys18_fid, n_sr_bins=18, enable_barlow_beeston=True)
        r24 = fit_realistic_profile_likelihood(t24, signed_sys=s_sys18_fid, n_sr_bins=18, enable_barlow_beeston=True)
        r27 = fit_realistic_profile_likelihood(t27, signed_sys=s_sys21_fid, n_sr_bins=21, enable_barlow_beeston=True)
        lumi_curve.append({
            'luminosity_fb_inv': lum,
            'expected_fiducial_events': nfid,
            'expected_nutau_cc': nfid * 0.003,
            'z0_stat_18bin': r18['z0_stat_only'],
            'z0_stat_27bin_ultra': r27['z0_stat_only'],
            'z0_pl_sr18_bb': r18['z0_profile_likelihood'],
            'z0_pl_sr24_charm_cr_bb': r24['z0_profile_likelihood'],
            'z0_pl_ultra27_charm_cr_bb': r27['z0_profile_likelihood'],
        })

    # 2,000-toy frequentist H0 coverage verification on HL-LHC 2t Fiducial 27-bin Ultra-Tail + Charm-CR
    hl_tpl_27 = build_physical_templates(
        joint_bins_27[fid_mask], labels[fid_mask], has_charm[fid_mask],
        n_bins=27, total_events=180000.0, tau_fraction=0.003, bkg_shares=FIDUCIAL_2T_BKG_SHARES,
    )
    h0_cov = run_h0_toy_coverage_verification(
        hl_tpl_27,
        signed_sys=s_sys21_fid,
        n_sr_bins=21,
        n_toys=n_toys,
        workers=workers,
        enable_barlow_beeston=True,
    )

    # Render publication-grade 2-panel English figure (300 DPI, clean label placement)
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.4), dpi=300)
    ax0, ax1 = axes

    xs = [row['luminosity_fb_inv'] for row in lumi_curve]
    ax0.plot(xs, [r['z0_stat_27bin_ultra'] for r in lumi_curve], 'o--', color='#4C72B0', lw=1.8, label='Stat-only (21-bin ultra-tail)')
    ax0.plot(xs, [r['z0_pl_ultra27_charm_cr_bb'] for r in lumi_curve], 'D-', color='#C44E52', lw=2.3, label='Full PL: Ultra-tail SR21 + Charm-CR + BB-lite')
    ax0.plot(xs, [r['z0_pl_sr24_charm_cr_bb'] for r in lumi_curve], 's-', color='#DD8452', lw=1.9, label='Full PL: Coarse SR18 + Charm-CR + BB-lite')
    ax0.plot(xs, [r['z0_pl_sr18_bb'] for r in lumi_curve], '^-', color='#55A868', lw=1.7, label='Full PL: Coarse SR18 only + BB-lite')
    ax0.axhline(3.0, color='#555555', ls=':', lw=1.1, label='3$\\sigma$ evidence / 5$\\sigma$ discovery')
    ax0.axhline(5.0, color='#222222', ls='-.', lw=1.2)
    y_top = max(5.6, max(r['z0_stat_27bin_ultra'] for r in lumi_curve) * 1.08)
    ax0.set_ylim(0.0, y_top)
    for x_m, tag, y_fr, x_mul, ha in (
        (250.0, 'Run 3\n250 fb$^{-1}$', 0.60, 1.05, 'left'),
        (680.0, 'Run 4\n680 fb$^{-1}$', 0.60, 1.05, 'left'),
        (3000.0, 'HL-LHC\n3 ab$^{-1}$', 0.88, 1.05, 'left'),
        (10000.0, 'FPF\n10 ab$^{-1}$', 0.88, 0.94, 'right'),
    ):
        ax0.axvline(x_m, color='#CCCCCC', ls=':', lw=0.9)
        ax0.text(x_m * x_mul, y_top * y_fr, tag, fontsize=7.5, color='#333333', va='top', ha=ha)
    ax0.set_xscale('log')
    ax0.set_xlabel('Integrated Luminosity [fb$^{-1}$]')
    ax0.set_ylabel('Expected Discovery Significance $Z_0$ [$\\sigma$]')
    ax0.set_title('2t Pinpoint Fiducial $\\nu_\\tau$ CC Sensitivity ($f_\\tau = 0.30\\%$, 8 Signed Sys + Charm)')
    ax0.grid(True, which='both', alpha=0.25)
    ax0.legend(loc='upper left', fontsize=7.6, frameon=True)

    z_grid = np.linspace(0.1, 3.2, 60)
    asym_sf = 0.5 * chi2.sf(z_grid ** 2, df=1)
    ax1.plot(z_grid, asym_sf, '-', color='#222222', lw=2.0, label='Asymptotic $\\frac{1}{2}\\chi^2_1$ (Wilks)')
    emp_pts_z = [1.0, 2.0, 3.0]
    emp_pts_p = [
        max(h0_cov['empirical_p_z_gt_1sigma'], 2e-4),
        max(h0_cov['empirical_p_z_gt_2sigma'], 2e-4),
        max(h0_cov['empirical_p_z_gt_3sigma'], 2e-4),
    ]
    emp_err = [math.sqrt(p * (1.0 - p) / n_toys) for p in emp_pts_p]
    ax1.errorbar(emp_pts_z, emp_pts_p, yerr=emp_err, fmt='o', color='#C44E52', ms=6.5, capsize=4, label=f'Frequentist $H_0$ Toys ($N={n_toys}$)')
    ax1.set_yscale('log')
    ax1.set_ylim(1.5e-4, 0.6)
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
        'physical_tau_fraction_trench': PHYSICAL_TAU_FRACTION,
        'physical_bkg_shares_trench': PHYSICAL_BKG_SHARES,
        'fiducial_2t_bkg_shares': FIDUCIAL_2T_BKG_SHARES,
        'legacy_v5_10k_1pct_claim_reference': w3_12_old['confirm200k']['significance_scan']['N_10000'][2],
        'slice_fpr_operating_point_audit_P1_4': slice_fpr_audit,
        'signed_systematic_templates_18bin': signed_sys_18,
        'signed_systematic_templates_21bin_ultra': signed_sys_21,
        'lhc_forward_scenarios': scenarios_out,
        'luminosity_curve_2t_fiducial': lumi_curve,
        'h0_toy_coverage_HLLHC_27bin': h0_cov,
        'figure_path': str(fig_path),
    }
    write_json(out_dir / 'exp6_realistic_flux_significance_report.json', report)
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
        'signed_sys_families': list(rep['signed_systematic_templates_18bin'].keys()),
        'scenarios': {
            k: {
                'N_tot': v['scenario']['expected_total_events'],
                'z0_sr18_stat': v['physical_sr_18bin_bb_lite']['z0_stat_only'],
                'z0_sr18_bb': v['physical_sr_18bin_bb_lite']['z0_profile_likelihood'],
                'z0_sr24_charm_cr_bb': v['physical_sr_plus_charm_cr_24bin_bb_lite']['z0_profile_likelihood'],
                'z0_ultra27_charm_cr_bb': v['physical_ultra_tail_sr21_plus_charm_cr_27bin_bb_lite']['z0_profile_likelihood'],
            }
            for k, v in rep['lhc_forward_scenarios'].items()
        },
        'h0_toy_coverage': rep['h0_toy_coverage_HLLHC_27bin'],
        'figure_path': rep['figure_path'],
    }, indent=2))


if __name__ == '__main__':
    main()
