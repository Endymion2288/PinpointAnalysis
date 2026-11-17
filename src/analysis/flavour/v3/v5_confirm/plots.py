"""Publication-grade English PNG (300 DPI) + PDF figures for Phase V5 (W3-11 + W3-12)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.flavour.v3.v5_confirm import DEFAULT_V3_ROOT, ROLE_ORDER, SEEDS


plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
})

ROLE_LABELS = {
    'pf_v3_gn2': 'PF-V3 (GN2)',
    'pf_v4_nomad_only': 'PF-V4 (NOMAD)',
    'pf_v4_nomad_pauc': 'PF-V4 (NOMAD+pAUC)',
    'b1pp_v3': 'B1++ (V1 Base)',
    'b1pp_v4_nomad_pauc': 'B1++-V4 (NOMAD+pAUC)',
}


def _save_fig(fig: plt.Figure, out_stem: Path) -> list[str]:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    png_p = out_stem.with_suffix('.png')
    pdf_p = out_stem.with_suffix('.pdf')
    fig.savefig(png_p, dpi=300)
    fig.savefig(pdf_p)
    plt.close(fig)
    return [str(png_p), str(pdf_p)]


def plot_fig1_multiseed_stability(w3_11: dict, w3_12: dict, out_dir: Path) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    roles = list(ROLE_ORDER)
    x = np.arange(len(roles) + 1)
    labels = [ROLE_LABELS[r] for r in roles] + ['Dual-Regime\nEnsemble']
    seed_colors = {17: '#1f77b4', 42: '#2ca02c', 2026: '#ff7f0e'}

    # Left: All BA4 (%), Right: All tau@1%FPR (%) on Tune40k + Confirm200k overlay
    for ax_idx, (metric_key, ens_key, title, ylabel) in enumerate((
        ('BA4', 'BA4', '4-Class Balanced Accuracy across Seeds {17, 42, 2026}', 'All-Event BA4 (%)'),
        ('tau_at_1pct_fpr', 'within_slice_S_tau_1pct', r'Tau Efficiency at 1% Non-Tau FPR across Seeds', r'All $\nu_\tau$ Eff. @ 1% FPR (%)'),
    )):
        ax = axes[ax_idx]
        tune_means, tune_stds = [], []
        for r in roles:
            st = w3_11['role_3seed_stability'][r]['All'][metric_key]
            tune_means.append(st['mean'] * 100.0)
            tune_stds.append(st['std'] * 100.0)
            for s in SEEDS:
                ax.scatter(
                    roles.index(r) + ({17: -0.14, 42: 0.0, 2026: 0.14}[s]),
                    st['per_seed'][str(s)] * 100.0,
                    color=seed_colors[s],
                    s=36,
                    zorder=4,
                    label=f'Tune40k seed {s}' if (r == roles[0] and ax_idx == 0) else None,
                )
        ens_st = w3_11['ensemble_3seed_stability']['All'][ens_key]
        tune_means.append(ens_st['mean'] * 100.0)
        tune_stds.append(ens_st['std'] * 100.0)
        for s in SEEDS:
            ax.scatter(
                len(roles) + ({17: -0.14, 42: 0.0, 2026: 0.14}[s]),
                ens_st['per_seed'][str(s)] * 100.0,
                color=seed_colors[s],
                s=42,
                zorder=4,
            )

        ax.errorbar(
            x - 0.06,
            tune_means,
            yerr=tune_stds,
            fmt='o',
            color='#111111',
            capsize=4,
            markersize=6,
            linewidth=1.5,
            label='Tune40k 3-seed mean +/- std' if ax_idx == 0 else None,
            zorder=5,
        )

        # Confirm200k 3-seed raw per-role means + frozen ens_v5 Confirm200k point
        conf_vals = []
        raw_conf = w3_12['confirm200k']['per_checkpoint_raw']
        for r in roles:
            vals = [raw_conf[f'{r}_s{s}']['All'][metric_key if ax_idx == 0 else 'within_slice_S_tau_1pct'] * 100.0 for s in SEEDS]
            conf_vals.append(float(np.mean(vals)))
        ens_c = w3_12['confirm200k']['ens_v5_3seed_dual_regime']['All']
        conf_vals.append((ens_c['BA4'] if ax_idx == 0 else ens_c['locked_pass_np_1pct']['tau_eff']) * 100.0)
        ax.plot(x + 0.18, conf_vals, 'D--', color='#d62728', markersize=6.5, linewidth=1.4, label='Confirm200k (one-shot unsealed)' if ax_idx == 0 else None, zorder=5)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=22, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.28, linestyle='--')

    axes[0].legend(loc='lower left', frameon=True, fontsize=8.0)
    fig.suptitle('W3-11 Three-Seed Stability and Zero-Overfitting Transfer to Confirm200k', y=1.02)
    fig.tight_layout()
    return _save_fig(fig, out_dir / 'fig1_w3_11_multiseed_stability')


def _roc_curve_log_fpr(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    is_tau = labels == 2
    s_pos = scores[is_tau]
    s_neg = scores[~is_tau]
    fpr_grid = np.geomspace(1e-4, 0.5, 240)
    thrs = np.quantile(s_neg, 1.0 - fpr_grid)
    tpr = np.array([(s_pos >= t).mean() for t in thrs], dtype=np.float64)
    return fpr_grid, tpr


def plot_fig2_confirm_roc(confirm_df: pd.DataFrame, w3_12: dict, out_dir: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    y = confirm_df['label'].to_numpy(dtype=np.int64)
    s_tau = confirm_df['s_tau'].to_numpy(dtype=np.float64)
    dom = confirm_df['obs_domain_id'].to_numpy(dtype=np.int64)
    pix_c = confirm_df['pixel_centered'].to_numpy(dtype=bool)

    curves = [
        ('All Confirm200k (N=200k)', np.ones(len(y), dtype=bool), '#111111', '-'),
        ('D_scint_contained (observable)', dom == 2, '#1f77b4', '-'),
        ('D_pix_contained (observable)', dom == 1, '#2ca02c', '-'),
        ('Truth_pixel_centered', pix_c, '#9467bd', '--'),
    ]
    for label, mask, col, ls in curves:
        fpr, tpr = _roc_curve_log_fpr(s_tau[mask], y[mask])
        ax.plot(fpr * 100.0, tpr * 100.0, color=col, linestyle=ls, linewidth=2.0, label=label)

    # Mark exact Cut-B160k locked working points on Confirm200k All
    e_all = w3_12['confirm200k']['ens_v5_3seed_dual_regime']['All']
    for key, marker, col, lbl in (
        ('locked_pass_global_1pct', 'o', '#d62728', 'Cut-B Locked Strategy I (Global 1%)'),
        ('locked_pass_np_1pct', '*', '#ff7f0e', 'Cut-B Locked Strategy II (NP 1%)'),
        ('locked_pass_global_0p1pct', 's', '#d62728', 'Cut-B Locked Strategy I (Global 0.1%)'),
        ('locked_pass_np_0p1pct', 'D', '#ff7f0e', 'Cut-B Locked Strategy II (NP 0.1%)'),
    ):
        pt = e_all[key]
        ax.scatter(
            pt['actual_fpr'] * 100.0,
            pt['tau_eff'] * 100.0,
            marker=marker,
            s=110 if marker == '*' else 65,
            color=col,
            edgecolors='black',
            linewidth=0.7,
            zorder=6,
            label=f"{lbl}: ({pt['actual_fpr']*100:.2f}%, {pt['tau_eff']*100:.2f}%)",
        )

    ax.set_xscale('log')
    ax.set_xlim(0.01, 25.0)
    ax.set_ylim(0.0, 52.0)
    ax.set_xlabel('Confirm200k Realized Non-Tau False Positive Rate (%)')
    ax.set_ylabel(r'$\nu_\tau$ CC Selection Efficiency (%)')
    ax.set_title(r'Confirm200k Low-FPR $\nu_\tau$ Discovery ROC with Frozen Cut-B160k Operating Points')
    ax.grid(True, which='both', alpha=0.25, linestyle='--')
    ax.legend(loc='upper left', fontsize=8.0, frameon=True)
    fig.tight_layout()
    return _save_fig(fig, out_dir / 'fig2_confirm200k_roc_and_locked_cuts')


def plot_fig3_migration_matrices(w3_12: dict, out_dir: Path) -> list[str]:
    mig = w3_12['confirm200k']['migration']
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), gridspec_kw={'width_ratios': [1.0, 1.25]})

    # Panel A: 4x4 Row-Normalized Flavour Migration (All Confirm200k)
    ax0 = axes[0]
    mat = np.asarray(mig['matrices_4x4']['All']['row_normalized'], dtype=np.float64) * 100.0
    mat_pix = np.asarray(mig['matrices_4x4']['Truth_pixel_centered']['row_normalized'], dtype=np.float64) * 100.0
    cnames = [r'$\nu_e$ CC', r'$\nu_\mu$ CC', r'$\nu_\tau$ CC', 'NC']
    im = ax0.imshow(mat, cmap='Blues', vmin=0.0, vmax=85.0)
    for i in range(4):
        for j in range(4):
            txt = f'{mat[i, j]:.1f}%\n({mat_pix[i, j]:.1f}%)'
            ax0.text(j, i, txt, ha='center', va='center', fontsize=8.3, color='white' if mat[i, j] > 45.0 else 'black')
    ax0.set_xticks(range(4))
    ax0.set_xticklabels(cnames)
    ax0.set_yticks(range(4))
    ax0.set_yticklabels(cnames)
    ax0.set_xlabel('Predicted Class')
    ax0.set_ylabel('True Neutrino Interaction')
    ax0.set_title('Flavour Migration: All Confirm200k\n(parentheses: Truth_pixel_centered)')
    fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04, label='Row Efficiency / Migration (%)')

    # Panel B: 8 physics subchannels locked NP 1% and 0.1% pass rates
    ax1 = axes[1]
    sub = mig['physics_subchannels']
    keys = list(sub.keys())
    nice = {
        'nu_e_CC': r'$\nu_e$ CC',
        'nu_mu_CC_no_charm': r'$\nu_\mu$ CC (no charm)',
        'nu_mu_CC_charm': r'$\nu_\mu$ CC (+charm)',
        'nu_tau_e': r'$\nu_\tau\to e$',
        'nu_tau_mu': r'$\nu_\tau\to \mu$',
        'nu_tau_had': r'$\nu_\tau\to$ had.',
        'NC_no_charm': 'NC (no charm)',
        'NC_charm': 'NC (+charm)',
    }
    y_pos = np.arange(len(keys))
    p1 = [sub[k]['locked_np_1pct_pass_rate'] * 100.0 for k in keys]
    p01 = [sub[k]['locked_np_0p1pct_pass_rate'] * 100.0 for k in keys]
    h = 0.36
    ax1.barh(y_pos - h / 2, p1, height=h, color='#1f77b4', label='Cut-B Locked NP 1% FPR Cut')
    ax1.barh(y_pos + h / 2, p01, height=h, color='#ff7f0e', label='Cut-B Locked NP 0.1% FPR Cut')
    for idx, (v1, v01) in enumerate(zip(p1, p01)):
        ax1.text(v1 + 0.3, idx - h / 2, f'{v1:.2f}%', va='center', fontsize=8.0)
        ax1.text(v01 + 0.3, idx + h / 2, f'{v01:.2f}%', va='center', fontsize=8.0)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([nice.get(k, k) for k in keys])
    ax1.invert_yaxis()
    ax1.set_xlim(0.0, max(p1) * 1.24)
    ax1.set_xlabel(r'$\nu_\tau$-Selection Pass Rate on Confirm200k (%)')
    ax1.set_title('Physics Subchannel Signal Efficiency vs Charm/NC Leakage')
    ax1.grid(True, axis='x', alpha=0.25, linestyle='--')
    ax1.legend(loc='upper right', frameon=True)

    fig.tight_layout()
    return _save_fig(fig, out_dir / 'fig3_confirm200k_migration_matrices')


def plot_fig4_stress_conditions(w3_12: dict, out_dir: Path) -> list[str]:
    vmap = w3_12['confirm200k']['stress_14_conditions']['variants']
    vnames = list(vmap.keys())
    y = np.arange(len(vnames))

    ba4, ba4_lo, ba4_hi = [], [], []
    eff, eff_lo, eff_hi = [], [], []
    fpr, fpr_lo, fpr_hi = [], [], []
    for vn in vnames:
        r = vmap[vn]
        ci = r['bootstrap_ci95']
        ba4.append(r['BA4'] * 100.0)
        ba4_lo.append((r['BA4'] - ci['BA4']['ci95'][0]) * 100.0)
        ba4_hi.append((ci['BA4']['ci95'][1] - r['BA4']) * 100.0)

        ev = r['locked_pass_np_1pct']['tau_eff'] * 100.0
        eff.append(ev)
        eff_lo.append(ev - ci['locked_np_1pct_tau_eff']['ci95'][0] * 100.0)
        eff_hi.append(ci['locked_np_1pct_tau_eff']['ci95'][1] * 100.0 - ev)

        fv = r['locked_pass_np_1pct']['actual_fpr'] * 100.0
        fpr.append(fv)
        fpr_lo.append(fv - ci['locked_np_1pct_fpr']['ci95'][0] * 100.0)
        fpr_hi.append(ci['locked_np_1pct_fpr']['ci95'][1] * 100.0 - fv)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.2), sharey=True)
    axes[0].errorbar(ba4, y, xerr=[ba4_lo, ba4_hi], fmt='o', color='#1f77b4', capsize=3, markersize=5)
    axes[0].axvline(ba4[0], color='#444444', linestyle='--', linewidth=1.0, label=f'Nominal ({ba4[0]:.2f}%)')
    axes[0].set_xlabel('4-Class Balanced Accuracy BA4 (%) [95% Block CI]')
    axes[0].set_title('Flavour Classification Under 14 Readout Conditions')
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(vnames)
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.25, linestyle='--')
    axes[0].legend(loc='lower left')

    axes[1].errorbar(eff, y - 0.12, xerr=[eff_lo, eff_hi], fmt='s', color='#2ca02c', capsize=3, markersize=4.8, label=r'$\nu_\tau$ Eff. (Cut-B NP 1%)')
    axes[1].errorbar(fpr, y + 0.12, xerr=[fpr_lo, fpr_hi], fmt='^', color='#d62728', capsize=3, markersize=4.8, label='Realized Non-Tau FPR (%)')
    axes[1].axvline(1.0, color='#d62728', linestyle=':', linewidth=1.1, label='Nominal 1.0% FPR target')
    axes[1].set_xlabel('Realized Rate (%) under Unchanged Cal-A + Cut-B Locks')
    axes[1].set_title('Frozen Operating Point Stability (N=8,000 Paired Events)')
    axes[1].set_xlim(-0.3, max(eff_hi[i] + eff[i] for i in range(len(eff))) * 1.15)
    axes[1].grid(True, alpha=0.25, linestyle='--')
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.40, 0.98), frameon=True, fontsize=8.0)

    fig.tight_layout()
    return _save_fig(fig, out_dir / 'fig4_confirm200k_14_stress_conditions')


def plot_fig5_profile_likelihood(w3_12: dict, out_dir: Path) -> list[str]:
    sig = w3_12['confirm200k']['significance_scan']
    n10_1p = next(r for r in sig['N_10000'] if abs(r['tau_prior'] - 0.01) < 1e-9)
    bins = n10_1p['tail_18bin']['per_bin']

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1))
    # Panel A: 18-bin template expected S & B at 10k events, 1% tau prior
    ax0 = axes[0]
    bx = np.arange(len(bins))
    s_exp = np.array([b['s_expected'] for b in bins], dtype=np.float64)
    b_exp = np.array([b['b_expected'] for b in bins], dtype=np.float64)
    ax0.bar(bx - 0.18, b_exp, width=0.36, color='#aec7e8', label=r'Non-$\tau$ Background ($b_i$)')
    ax0.bar(bx + 0.18, s_exp, width=0.36, color='#d62728', label=r'$\nu_\tau$ CC Signal ($s_i$, 1% prior)')
    ax0.set_yscale('log')
    y_top = max(b_exp) * 14.0
    ax0.set_ylim(0.15, y_top)
    ax0.axvline(5.5, color='#333333', linestyle='--', linewidth=1.0)
    ax0.axvline(11.5, color='#333333', linestyle='--', linewidth=1.0)
    ax0.text(2.5, y_top * 0.88, 'D_pix_contained\n(bins 0-5)', ha='center', va='top', fontsize=8.3)
    ax0.text(8.5, y_top * 0.88, 'D_scint_contained\n(bins 6-11)', ha='center', va='top', fontsize=8.3)
    ax0.text(14.5, y_top * 0.88, 'D_partial_or_edge\n(bins 12-17)', ha='center', va='top', fontsize=8.3)
    ax0.set_xticks(bx)
    ax0.set_xticklabels([f'B{i}' for i in range(len(bins))], fontsize=8)
    ax0.set_ylabel('Expected Events (10,000 Total, 1% Tau Prior)')
    ax0.set_title('Cut-B Locked 18-Bin Multi-Domain Discovery Template on Confirm200k')
    ax0.grid(True, which='both', axis='y', alpha=0.25, linestyle='--')
    ax0.legend(loc='upper left', bbox_to_anchor=(0.015, 0.83), frameon=True, fontsize=8.0)

    # Panel B: Stat-only vs Profile-Likelihood Z0 across tau prior & exposure
    ax1 = axes[1]
    priors = [r['tau_prior'] * 100.0 for r in sig['N_10000']]
    z10_stat = [r['tail_18bin']['z0_stat_only'] for r in sig['N_10000']]
    z10_pl = [r['tail_18bin']['z0_profile_likelihood'] for r in sig['N_10000']]
    z10_ss = [r['z0_single_stream_stat'] for r in sig['N_10000']]
    z50_stat = [r['tail_18bin']['z0_stat_only'] for r in sig['N_50000']]
    z50_pl = [r['tail_18bin']['z0_profile_likelihood'] for r in sig['N_50000']]

    ax1.plot(priors, z10_stat, 'o--', color='#1f77b4', alpha=0.75, label='N=10k Stat-Only 18-bin Asimov Z0')
    ax1.plot(priors, z10_pl, 'o-', color='#1f77b4', linewidth=2.2, label='N=10k Profile-Likelihood Z0 (w/ syst.)')
    ax1.plot(priors, z10_ss, 'v:', color='#7f7f7f', label='N=10k Single-Stream Stat-Only Z0')
    ax1.plot(priors, z50_stat, 's--', color='#d62728', alpha=0.75, label='N=50k Stat-Only 18-bin Asimov Z0')
    ax1.plot(priors, z50_pl, 's-', color='#d62728', linewidth=2.2, label='N=50k Profile-Likelihood Z0 (w/ syst.)')
    ax1.axhline(3.0, color='#2ca02c', linestyle='--', linewidth=1.1, label=r'3$\sigma$ evidence')
    ax1.axhline(5.0, color='#9467bd', linestyle=':', linewidth=1.1, label=r'5$\sigma$ discovery')
    ax1.set_xlabel(r'Assumed $\nu_\tau$ CC Signal Fraction (%)')
    ax1.set_ylabel(r'Expected Discovery Significance ($\sigma$)')
    ax1.set_title('Confirm200k Stat-Only vs Profile-Likelihood Discovery Sensitivity')
    ax1.grid(True, alpha=0.28, linestyle='--')
    ax1.legend(loc='upper left', fontsize=8.0, frameon=True)

    fig.tight_layout()
    return _save_fig(fig, out_dir / 'fig5_profile_likelihood_significance')


def generate_all_v5_figures(v3_root: Path = DEFAULT_V3_ROOT) -> list[str]:
    w3_11 = json.loads((v3_root / 'W3-11' / 'w3_11_multiseed_results.json').read_text())
    w3_12 = json.loads((v3_root / 'W3-12' / 'w3_12_confirmation_results.json').read_text())
    confirm_df = pd.read_parquet(v3_root / 'W3-12' / 'predictions' / 'confirm' / 'ens_v5_3seed_dual_regime_confirm.parquet')
    out_dir = v3_root / 'W3-12' / 'figures'

    paths: list[str] = []
    paths.extend(plot_fig1_multiseed_stability(w3_11, w3_12, out_dir))
    paths.extend(plot_fig2_confirm_roc(confirm_df, w3_12, out_dir))
    paths.extend(plot_fig3_migration_matrices(w3_12, out_dir))
    paths.extend(plot_fig4_stress_conditions(w3_12, out_dir))
    paths.extend(plot_fig5_profile_likelihood(w3_12, out_dir))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v3-root', type=Path, default=DEFAULT_V3_ROOT)
    args = parser.parse_args()
    out = generate_all_v5_figures(args.v3_root)
    print(json.dumps({'figures': out}, indent=2))


if __name__ == '__main__':
    main()
