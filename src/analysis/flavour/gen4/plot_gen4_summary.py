"""Generate 300-DPI 100%-English summary figures for Stage 329 (Pinpoint-Gen4 on MC 10085-10088).

Reads only unsealed Stage-I / Stage-II & independent audit JSONs under `SealAuditGuard`
and renders publication figures into:
  /eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage329_gen4_mc8588/figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analysis.flavour.nextgen.seal_guard import SealAuditGuard, assert_unsealed_path


GEN4_RES = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage329_gen4_mc8588')
DEFAULT_AUDIT_JSON = GEN4_RES / 'verification/independent_tune6k_audit.json'
DEFAULT_DP_S2 = GEN4_RES / 'exp_diffphys_sbi/stage2/stage2_diffphys_results.json'
DEFAULT_SUP_DIR = GEN4_RES / 'exp_gen4_supervised'
DEFAULT_SSL_S1 = GEN4_RES / 'exp_gen4_ssl_tail/stage1/stage1_summary.json'
DEFAULT_SSL_S2 = GEN4_RES / 'exp_gen4_ssl_tail/stage2/stage2_summary.json'
DEFAULT_FIG_DIR = GEN4_RES / 'figures'

CJK_RE = re.compile(r'[\u4e00-\u9fff]')

PALETTE = {
    'blue': '#0072B2',
    'orange': '#D55E00',
    'green': '#009E73',
    'purple': '#CC79A7',
    'amber': '#E69F00',
    'sky': '#56B4E9',
    'grey': '#4D4D4D',
    'lightgrey': '#999999',
}


def _configure_style() -> None:
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.titlesize': 11.5,
        'axes.titleweight': 'bold',
        'axes.labelsize': 10.5,
        'xtick.labelsize': 9.0,
        'ytick.labelsize': 9.0,
        'legend.fontsize': 8.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linestyle': '--',
        'figure.dpi': 300,
        'savefig.dpi': 300,
    })


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    p = assert_unsealed_path(path)
    return json.loads(p.read_text())


def _assert_ascii_english_figure(fig: plt.Figure, out_path: Path) -> None:
    texts: list[str] = []
    if fig._suptitle is not None:
        texts.append(fig._suptitle.get_text())
    for ax in fig.get_axes():
        texts.append(ax.get_title())
        texts.append(ax.get_xlabel())
        texts.append(ax.get_ylabel())
        for t in ax.texts:
            texts.append(t.get_text())
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                texts.append(t.get_text())
    for txt in texts:
        if CJK_RE.search(txt):
            raise ValueError(f'Non-English CJK text detected in {out_path.name}: {txt!r}')


def _save_fig(fig: plt.Figure, out_path: Path) -> str:
    _assert_ascii_english_figure(fig, out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return str(out_path)


def plot_fig1_tune6k_overall_and_slices(audit: dict[str, Any], fig_dir: Path) -> str | None:
    models = audit.get('models', {})
    if not models:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.4, 5.6), gridspec_kw={'width_ratios': [1.15, 1.15], 'wspace': 0.42})

    order = [
        ('Ref-BDT', 'Ref-BDT', PALETTE['lightgrey']),
        ('Ref-L1-GN2+DRO', 'Ref-L1-GN2\n+KL-DRO', PALETTE['grey']),
        ('DiffPhys-SBI', 'DiffPhys-SBI\n(3-seed)', PALETTE['amber']),
        ('PinpointGen4_SupOnly', 'Gen4 SupOnly\n(3-seed)', PALETTE['sky']),
        ('PinpointGen4_SSL_Tail', 'Gen4+SSL+Tail\n(3-seed)', PALETTE['blue']),
        ('PinpointGen4_Full_Blend', 'Gen4 Full Blend\n(+DiffPhys)', PALETTE['green']),
    ]
    present = [(k, lbl, col) for k, lbl, col in order if k in models]
    if present:
        x = np.arange(len(present))
        ba4 = [float(models[k]['eval_canonical_s_tau']['BA4_pct']) for k, _, _ in present]
        nll = [float(models[k]['eval_canonical_s_tau']['NLL']) for k, _, _ in present]
        cols = [col for _, _, col in present]

        bars = ax1.bar(x - 0.18, ba4, width=0.35, color=cols, edgecolor='black', linewidth=0.7, label='BA4 (%) [left]')
        for b, v in zip(bars, ba4):
            ax1.text(b.get_x() + b.get_width() / 2.0, v + 0.12, f'{v:.2f}%', ha='center', va='bottom', fontsize=7.8, fontweight='bold')

        ax1_r = ax1.twinx()
        ax1_r.spines['top'].set_visible(False)
        ax1_r.grid(False)
        bars_n = ax1_r.bar(x + 0.18, nll, width=0.35, color=cols, alpha=0.45, hatch='//', edgecolor='black', linewidth=0.7, label='NLL [right]')
        for b, v in zip(bars_n, nll):
            ax1_r.text(b.get_x() + b.get_width() / 2.0, v + 0.002, f'{val_fmt(v)}', ha='center', va='bottom', fontsize=7.5)

        ax1.set_xticks(x)
        ax1.set_xticklabels([lbl for _, lbl, _ in present], fontsize=8.0)
        ax1.set_ylim(max(0.0, min(ba4) - 2.5), min(100.0, max(ba4) + 2.0))
        ax1_r.set_ylim(max(0.0, min(nll) - 0.04), max(nll) + 0.04)
        ax1.set_ylabel('Tune6k Balanced 4-Class Accuracy (%)')
        ax1_r.set_ylabel('Tune6k Cross-Entropy NLL (lower is better)')
        ax1.set_title('Tune6k (N=6,000) Overall Accuracy & Calibration')
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax1_r.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc='upper left', frameon=True)

    # Panel B: Detector Region & Mahlstedt BA3 Comparison
    comp_keys = [k for k in ('Ref-BDT', 'Ref-L1-GN2+DRO', 'DiffPhys-SBI', 'PinpointGen4_SSL_Tail', 'PinpointGen4_Full_Blend') if k in models]
    slice_specs = [
        ('All_Tune6k', 'BA4_pct', 'All Tune6k'),
        ('D1_obs', 'D1_obs_BA4_pct', 'D1 (Pinpoint)'),
        ('Front30', 'Front30_BA4_pct', 'Front30'),
        ('Interm150', 'Interm150_BA4_pct', 'Interm150'),
        ('Fortune_ScintOnly', 'Fortune_ScintOnly_BA4_pct', 'Fortune Scint'),
        ('Mahlstedt', 'BA3_pct', 'Mahlstedt BA3'),
    ]
    if comp_keys:
        xs = np.arange(len(slice_specs))
        w = 0.78 / len(comp_keys)
        col_map = {k: c for k, _, c in order}
        lbl_map = {k: l.replace('\n', ' ') for k, l, _ in order}
        for idx, mk in enumerate(comp_keys):
            ev = models[mk]['eval_canonical_s_tau']
            m_ba3 = models[mk]['mahlstedt_pc_ba3']
            vals = []
            for sk, field, _ in slice_specs:
                if sk == 'Mahlstedt':
                    vals.append(float(m_ba3.get('BA3_pct', 0.0)))
                elif sk == 'All_Tune6k':
                    vals.append(float(ev.get('BA4_pct', 0.0)))
                else:
                    vals.append(float(ev.get(field, 0.0)))
            offset = (idx - 0.5 * (len(comp_keys) - 1)) * w
            ax2.bar(xs + offset, vals, width=w, color=col_map[mk], edgecolor='black', linewidth=0.5, label=lbl_map[mk])
        ax2.set_xticks(xs)
        ax2.set_xticklabels([s[2] for s in slice_specs], fontsize=8.2)
        ax2.set_ylim(45.0, 82.0)
        ax2.set_ylabel('Balanced Accuracy (%)')
        ax2.set_title('Detector-Region & Section-12 Mahlstedt BA3 Breakdown')
        ax2.legend(loc='upper left', ncol=2, frameon=True, fontsize=7.8)

    fig.suptitle('Pinpoint-Gen4 on Official MC 10085-10088: Tune6k Performance & Regional Synergy', fontsize=13, fontweight='bold', y=1.02)
    return _save_fig(fig, fig_dir / 'fig1_gen4_mc8588_overall_tune6k_and_slices.png')


def val_fmt(v: float) -> str:
    return f'{v:.3f}'


def plot_fig2_ssl_scaling_and_ablations(
    ssl_s1: dict[str, Any] | None,
    sup_dir: Path,
    fig_dir: Path,
) -> str | None:
    if ssl_s1 is None and not (sup_dir / 'stage1').is_dir():
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.6, 5.3), gridspec_kw={'wspace': 0.36})

    # Panel A: SSL-v2 vs Scratch Label Efficiency
    if ssl_s1 and 'variants' in ssl_s1:
        v = ssl_s1['variants']
        fracs = [10, 25, 100]
        xpos = [0, 1, 2]
        ssl_ba4 = [float(v[f'ssl_v2_ft_{f}pct']['BA4_pct']) for f in fracs if f'ssl_v2_ft_{f}pct' in v]
        scr_ba4 = [float(v[f'scratch_{f}pct']['BA4_pct']) for f in fracs if f'scratch_{f}pct' in v]
        ssl_nll = [float(v[f'ssl_v2_ft_{f}pct']['NLL']) for f in fracs if f'ssl_v2_ft_{f}pct' in v]
        scr_nll = [float(v[f'scratch_{f}pct']['NLL']) for f in fracs if f'scratch_{f}pct' in v]

        if ssl_ba4 and scr_ba4:
            ax1.plot(xpos[:len(ssl_ba4)], ssl_ba4, marker='o', linewidth=2.3, markersize=7, color=PALETTE['blue'], label='PinpointSSL-v2 Fine-Tune (BA4)')
            ax1.plot(xpos[:len(scr_ba4)], scr_ba4, marker='s', linestyle='--', linewidth=2.0, markersize=6.5, color=PALETTE['orange'], label='From Scratch (BA4)')
            for xi, ys, y0 in zip(xpos, ssl_ba4, scr_ba4):
                ax1.text(xi, ys + 0.28, f'{ys:.2f}%', ha='center', va='bottom', fontsize=8.3, color=PALETTE['blue'], fontweight='bold')
                ax1.text(xi, y0 - 0.32, f'{y0:.2f}%', ha='center', va='top', fontsize=8.3, color=PALETTE['orange'], fontweight='bold')

            ax1_r = ax1.twinx()
            ax1_r.spines['top'].set_visible(False)
            ax1_r.grid(False)
            ax1_r.plot(xpos[:len(ssl_nll)], ssl_nll, marker='^', linestyle=':', linewidth=1.8, color=PALETTE['green'], label='SSL-v2 NLL [right]')
            ax1_r.plot(xpos[:len(scr_nll)], scr_nll, marker='v', linestyle=':', linewidth=1.8, color=PALETTE['purple'], label='Scratch NLL [right]')
            ax1_r.set_ylabel('Validation NLL (lower is better)')
            ax1_r.set_ylim(1.02, 1.42)

            ax1.set_xticks(xpos)
            ax1.set_xticklabels(['10%\n(2,560)', '25%\n(6,400)', '100%\n(25,600)'])
            ax1.set_ylim(44.0, 54.5)
            ax1.set_xlabel('Labeled Training Fraction (internal_train)')
            ax1.set_ylabel('Balanced 4-Class Accuracy (%)')
            ax1.set_title('PinpointSSL-v2 Positive Transfer Across Label Budgets')
            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax1_r.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2, loc='center right', frameon=True, fontsize=7.8)

    # Panel B: Supervised Stage-I Causal Ablations
    s1_dir = sup_dir / 'stage1'
    ab_order = [
        ('gen4_full', 'Gen4 Full\n(w/ edep)', PALETTE['blue']),
        ('abl_binary_no_edep', 'Abl: Binary\nNo edep', PALETTE['orange']),
        ('abl_no_sandwich_no_ndive', 'Abl: No\nNDIVE/Slab', PALETTE['sky']),
        ('abl_no_m2_no_kink_stub', 'Abl: No M2\n& KinkStub', PALETTE['amber']),
        ('abl_no_desat_sinkhorn_fmpe', 'Abl: No DeSat\nSinkhorn', PALETTE['purple']),
    ]
    ab_recs = []
    for k, lbl, col in ab_order:
        mp = s1_dir / k / 'metrics.json'
        if mp.is_file():
            d = json.loads(mp.read_text())
            m = d.get('metrics', d)
            ab_recs.append((lbl, col, float(m.get('BA4_pct', 0.0)), float(m.get('D1_obs_BA4_pct', 0.0)), float(m.get('tau_1pct_fpr_llr_pct', 0.0))))
    if ab_recs:
        x = np.arange(len(ab_recs))
        w = 0.28
        b_all = ax2.bar(x - w, [r[2] for r in ab_recs], width=w, color=PALETTE['blue'], edgecolor='black', linewidth=0.6, label='All BA4 (%)')
        b_d1 = ax2.bar(x, [r[3] for r in ab_recs], width=w, color=PALETTE['green'], edgecolor='black', linewidth=0.6, label='D1_obs BA4 (%)')
        ax2_r = ax2.twinx()
        ax2_r.spines['top'].set_visible(False)
        ax2_r.grid(False)
        b_tau = ax2_r.bar(x + w, [r[4] for r in ab_recs], width=w, color=PALETTE['orange'], alpha=0.80, edgecolor='black', linewidth=0.6, label='Tau @ 1% FPR (%) [right]')
        for b in b_tau:
            v = b.get_height()
            ax2_r.text(b.get_x() + b.get_width() / 2.0, v + 0.05, f'{v:.2f}%', ha='center', va='bottom', fontsize=7.3)
        ax2.set_xticks(x)
        ax2.set_xticklabels([r[0] for r in ab_recs], fontsize=8.0)
        ax2.set_ylim(49.0, 60.5)
        ax2_r.set_ylim(0.0, 3.8)
        ax2.set_ylabel('Balanced 4-Class Accuracy (%)')
        ax2_r.set_ylabel('Tau Efficiency @ 1% FPR (%)')
        ax2.set_title('Stage-I Causal Ablations (internal_val_eval, N=4,800)')
        h1, l1 = ax2.get_legend_handles_labels()
        h2, l2 = ax2_r.get_legend_handles_labels()
        ax2.legend(h1 + h2, l1 + l2, loc='upper right', frameon=True, fontsize=7.8)

    fig.suptitle('Stage 329 Representation & Physics Ablations: SSL-v2 Scaling & Single-Pixel edep', fontsize=13, fontweight='bold', y=1.02)
    return _save_fig(fig, fig_dir / 'fig2_gen4_ssl_scaling_and_ablations.png')


def plot_fig3_submodes_potgpd_and_nsbi(
    audit: dict[str, Any],
    dp_s2: dict[str, Any] | None,
    ssl_s2: dict[str, Any] | None,
    fig_dir: Path,
) -> str | None:
    models = audit.get('models', {})
    if not models:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 5.4), gridspec_kw={'wspace': 0.34})

    # Panel A: Tau Submodes & L_tau < 30 mm @ 1% FPR (using canonical s_tau)
    sub_order = [
        ('tau_e_1pct_fpr_pct', 'tau -> e'),
        ('tau_mu_1pct_fpr_pct', 'tau -> mu'),
        ('tau_1h_1pct_fpr_pct', 'tau -> 1h'),
        ('tau_3h_1pct_fpr_pct', 'tau -> 3h'),
        ('tau_L_lt_30mm_1pct_fpr_pct', 'L_tau < 30 mm'),
    ]
    comp_models = [k for k in ('Ref-BDT', 'Ref-L1-GN2+DRO', 'DiffPhys-SBI', 'PinpointGen4_SupOnly', 'PinpointGen4_Full_Blend') if k in models]
    c_cols = {'Ref-BDT': PALETTE['lightgrey'], 'Ref-L1-GN2+DRO': PALETTE['grey'], 'DiffPhys-SBI': PALETTE['amber'], 'PinpointGen4_SupOnly': PALETTE['sky'], 'PinpointGen4_Full_Blend': PALETTE['green']}
    if comp_models:
        x = np.arange(len(sub_order))
        w = 0.78 / len(comp_models)
        max_sub = 0.0
        for idx, mk in enumerate(comp_models):
            ev = models[mk]['eval_canonical_s_tau']
            vals = [float(ev.get(sk, 0.0)) for sk, _ in sub_order]
            max_sub = max(max_sub, max(vals, default=0.0))
            off = (idx - 0.5 * (len(comp_models) - 1)) * w
            ax1.bar(x + off, vals, width=w, color=c_cols[mk], edgecolor='black', linewidth=0.5, label=mk.replace('_', ' '))
        ax1.set_xticks(x)
        ax1.set_xticklabels([s[1] for s in sub_order])
        ax1.set_ylim(0.0, max(7.2, max_sub * 1.28))
        ax1.set_ylabel('Signal Efficiency @ 1% Bkg FPR (%)')
        ax1.set_title('Tune6k Tau Decay Submodes & Short-L_tau @ 1% FPR')
        ax1.legend(loc='upper left', ncol=2, frameon=True, fontsize=7.6)

    # Panel B: NSBI-PL Z0 Discovery Significance vs Single-Bin 1% FPR
    nsbi_entries: list[tuple[str, float, float, float]] = []
    if dp_s2 and 'nsbi_pl_closure' in dp_s2:
        nc = dp_s2['nsbi_pl_closure']
        nsbi_entries.append((
            'DiffPhys-SBI',
            float(nc['single_bin_1pct_fpr']['z0_pl_no_bb']),
            float(nc['nsbi_pl_no_bb']['z0_profile_likelihood']),
            float(nc['nsbi_pl_no_bb']['z0_stat_only']),
        ))
    if ssl_s2 and 'nsbi_pl' in ssl_s2:
        g4_only = ssl_s2['nsbi_pl'].get('gen4_only_closure', {})
        if g4_only:
            nsbi_entries.append((
                'Gen4+SSL+Tail',
                float(g4_only['single_bin_1pct_fpr']['z0_pl_no_bb']),
                float(g4_only['nsbi_pl_no_bb']['z0_profile_likelihood']),
                float(g4_only['nsbi_pl_no_bb']['z0_stat_only']),
            ))
        np_f = ssl_s2['nsbi_pl']
        nsbi_entries.append((
            'Gen4 Full Blend',
            float(np_f['single_bin_1pct_fpr']['z0_pl_no_bb']),
            float(np_f['nsbi_pl_no_bb']['z0_profile_likelihood']),
            float(np_f['nsbi_pl_no_bb']['z0_stat_only']),
        ))

    if nsbi_entries:
        x = np.arange(len(nsbi_entries))
        w = 0.25
        b1 = ax2.bar(x - w, [e[1] for e in nsbi_entries], width=w, color=PALETTE['lightgrey'], edgecolor='black', linewidth=0.6, label='Single-Bin 1% FPR (PL)')
        b2 = ax2.bar(x, [e[2] for e in nsbi_entries], width=w, color=PALETTE['blue'], edgecolor='black', linewidth=0.6, label='4-SR + 3-CR NSBI-PL')
        b3 = ax2.bar(x + w, [e[3] for e in nsbi_entries], width=w, color=PALETTE['green'], alpha=0.80, edgecolor='black', linewidth=0.6, label='4-SR + 3-CR Stat-Only Z0')
        for grp in (b1, b2, b3):
            for b in grp:
                v = b.get_height()
                ax2.text(b.get_x() + b.get_width() / 2.0, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=7.6)
        ax2.set_xticks(x)
        ax2.set_xticklabels([e[0] for e in nsbi_entries])
        ax2.set_ylim(0.0, max(e[3] for e in nsbi_entries) * 1.18)
        ax2.set_ylabel('Asimov Discovery Significance Z0 (sigma)')
        ax2.set_title('Golden Micro-Channel + 3-CR NSBI-PL Significance (f_tau=0.3%)')
        ax2.legend(loc='upper right', frameon=True, fontsize=8.0)

    fig.suptitle('Submode Tail Efficiency & Golden Micro-Channel NSBI Profile-Likelihood Gain', fontsize=13, fontweight='bold', y=1.02)
    return _save_fig(fig, fig_dir / 'fig3_gen4_submodes_and_nsbi_z0.png')


def plot_fig4_pv_pulls_and_stress(
    audit: dict[str, Any],
    ssl_s2: dict[str, Any] | None,
    dp_s2: dict[str, Any] | None,
    fig_dir: Path,
) -> str | None:
    models = audit.get('models', {})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 5.4), gridspec_kw={'wspace': 0.38})

    # Panel A: 9/9 PV Pulls on Tune6k
    pv_key = 'PinpointGen4_SSL_Tail' if 'PinpointGen4_SSL_Tail' in models and 'pv_pulls' in models['PinpointGen4_SSL_Tail'] else (
        'PinpointGen4_SupOnly' if 'PinpointGen4_SupOnly' in models and 'pv_pulls' in models['PinpointGen4_SupOnly'] else None
    )
    if pv_key:
        pva = models[pv_key]['pv_pulls']
        regs = [('All_valid', 'All Valid PV'), ('Front30', 'Front30'), ('Interm150', 'Interm150')]
        pres = [(rk, rl) for rk, rl in regs if rk in pva]
        x = np.arange(len(pres))
        w = 0.24
        cols = [PALETTE['blue'], PALETTE['green'], PALETTE['orange']]
        max_pull = 1.2
        for ai, aname in enumerate(('x', 'y', 'z')):
            vals = [float(pva[rk]['p68_abs_pull'][ai]) for rk, _ in pres]
            max_pull = max(max_pull, max(vals, default=1.2))
            bars = ax1.bar(x + (ai - 1) * w, vals, width=w, color=cols[ai], edgecolor='black', linewidth=0.6, label=f'PV {aname} 68% |pull|')
            for b, v in zip(bars, vals):
                ax1.text(b.get_x() + b.get_width() / 2.0, v + 0.03, f'{v:.2f}', ha='center', va='bottom', fontsize=7.6)
        ax1.axhspan(0.85, 1.15, color=PALETTE['green'], alpha=0.14, label='Target Band [0.85, 1.15]')
        ax1.axhline(1.0, color='black', linestyle='--', linewidth=1.0)
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"{rl}\n(|dz|={pva[rk]['median_abs_dz_mm']:.2f} mm)" for rk, rl in pres])
        ax1.set_ylim(0.4, max(1.45, max_pull * 1.20))
        ax1.set_ylabel('Normalized 68% |Pull| / q_68(t_3)')
        ax1.set_title(f'Tune6k Student-t(nu=3) PV Pull Calibration ({pv_key.replace("_", " ")})')
        ax1.legend(loc='upper left', ncol=1, frameon=True, fontsize=7.6)

    # Panel B: Stress400 & Diag_OOD
    ssl_str = (ssl_s2 or {}).get('stress400_ens3', {})
    dp_str = (dp_s2 or {}).get('stress400_tests', {})
    ssl_ood = (ssl_s2 or {}).get('diag_ood_ens3', {})
    dp_ood = (dp_s2 or {}).get('ood_evaluation', {})
    labels_y, v_ssl, v_dp = [], [], []
    for k in ('nominal', 'pixel_dead_2pct', 'pixel_dead_5pct', 'pixel_dark_1e-5', 'tracker_dead_2pct_noise1', 'tracker_dead_5pct_noise2', 'sipm_sat_cap_0p95'):
        if k in ssl_str or k in dp_str:
            labels_y.append(f'Stress: {k}')
            v_ssl.append(float(ssl_str.get(k, {}).get('BA4_pct', np.nan)))
            v_dp.append(float(dp_str.get(k, {}).get('BA4_pct', np.nan)))
    for k in ('NonW_Al_Scint', 'HighE_gt1p5TeV'):
        if k in ssl_ood or k in dp_ood:
            labels_y.append(f'OOD: {k}')
            v_ssl.append(float(ssl_ood.get(k, {}).get('BA4_pct', np.nan)))
            v_dp.append(float(dp_ood.get(k, {}).get('BA4_pct', np.nan)))

    if labels_y:
        y = np.arange(len(labels_y))
        h = 0.36
        ax2.barh(y - h / 2, [0.0 if not np.isfinite(x) else x for x in v_ssl], height=h, color=PALETTE['blue'], edgecolor='black', linewidth=0.5, label='Gen4+SSL+Tail (3-seed)')
        ax2.barh(y + h / 2, [0.0 if not np.isfinite(x) else x for x in v_dp], height=h, color=PALETTE['amber'], edgecolor='black', linewidth=0.5, label='DiffPhys-SBI (3-seed)')
        ax2.set_yticks(y)
        ax2.set_yticklabels(labels_y, fontsize=8.0)
        ax2.invert_yaxis()
        ax2.set_xlim(38.0, 60.0)
        ax2.set_xlabel('Balanced 4-Class Accuracy (%)')
        ax2.set_title('Stress400 Hardware Degradation & Diag_OOD Extrapolation')
        ax2.legend(loc='lower right', frameon=True, fontsize=8.0)

    fig.suptitle('Primary Vertex Pull Calibration, Hardware Stress Resilience & OOD Generalization', fontsize=13, fontweight='bold', y=1.02)
    return _save_fig(fig, fig_dir / 'fig4_gen4_pv_pulls_and_stress.png')


def main() -> None:
    with SealAuditGuard():
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--audit-json', type=Path, default=DEFAULT_AUDIT_JSON)
        parser.add_argument('--dp-s2', type=Path, default=DEFAULT_DP_S2)
        parser.add_argument('--sup-dir', type=Path, default=DEFAULT_SUP_DIR)
        parser.add_argument('--ssl-s1', type=Path, default=DEFAULT_SSL_S1)
        parser.add_argument('--ssl-s2', type=Path, default=DEFAULT_SSL_S2)
        parser.add_argument('--out-dir', type=Path, default=DEFAULT_FIG_DIR)
        args = parser.parse_args()

        _configure_style()
        audit = _load_json_if_exists(args.audit_json)
        dp_s2 = _load_json_if_exists(args.dp_s2)
        ssl_s1 = _load_json_if_exists(args.ssl_s1)
        ssl_s2 = _load_json_if_exists(args.ssl_s2)

        generated: list[str] = []
        if audit is not None:
            for fn in (
                lambda: plot_fig1_tune6k_overall_and_slices(audit, args.out_dir),
                lambda: plot_fig3_submodes_potgpd_and_nsbi(audit, dp_s2, ssl_s2, args.out_dir),
                lambda: plot_fig4_pv_pulls_and_stress(audit, ssl_s2, dp_s2, args.out_dir),
            ):
                p = fn()
                if p:
                    generated.append(p)
        p2 = plot_fig2_ssl_scaling_and_ablations(ssl_s1, args.sup_dir, args.out_dir)
        if p2:
            generated.append(p2)

        print(json.dumps({'generated_figures': generated, 'count': len(generated)}, indent=2))


if __name__ == '__main__':
    main()
