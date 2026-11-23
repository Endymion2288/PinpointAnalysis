"""Generate 300-DPI 100%-English summary figures for Stage 328 (Experiments L1 & L2).

Reads only unsealed Stage-I / Stage-II result JSONs under `SealAuditGuard` and
renders publication figures into:
  /eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage328_figures/
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


NEXTGEN_RES = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research')
DEFAULT_L2_S1 = NEXTGEN_RES / 'exp_l2_jepa/stage1_l2_jepa_results.json'
DEFAULT_L2_S2 = NEXTGEN_RES / 'exp_l2_jepa/stage2/stage2_l2_jepa_results.json'
DEFAULT_L1_S1 = NEXTGEN_RES / 'exp_l1_gn2/stage1_l1_gn2_results.json'
DEFAULT_L1_S2 = NEXTGEN_RES / 'exp_l1_gn2/stage2/stage2_l1_gn2_results.json'
DEFAULT_FIG_DIR = NEXTGEN_RES / 'stage328_figures'

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
        'xtick.labelsize': 9.2,
        'ytick.labelsize': 9.2,
        'legend.fontsize': 8.8,
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


def plot_l2_equal_compute_and_label_efficiency(l2_s1: dict[str, Any], fig_dir: Path) -> str | None:
    models = l2_s1.get('models', {})
    if not models:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.3), gridspec_kw={'width_ratios': [1.28, 1.0], 'wspace': 0.46})

    # Panel A: Equal-Compute & SSL Ablations (100% = 25.6k labels)
    order = [
        ('sup_scratch_20p', 'Scratch\n(20 passes)', PALETTE['lightgrey']),
        ('sup_scratch_equal_compute', 'Scratch Equal\nCompute', PALETTE['grey']),
        ('abl_ssl_random_point_mask_shortcut_ft_100pct', 'Abl: Random\nPoint Mask', PALETTE['amber']),
        ('abl_ssl_no_spec_precal_ft_100pct', 'Abl: No Spec\nPre-Calib', PALETTE['sky']),
        ('ssl300k_jepa_ft_100pct', 'PinpointSSL-JEPA\n(300k + FT)', PALETTE['blue']),
    ]
    present = [(k, lbl, col) for k, lbl, col in order if k in models]
    if present:
        x = np.arange(len(present))
        ba4 = [float(models[k]['internal_val_eval_all']['BA4_pct']) for k, _, _ in present]
        nll = [float(models[k]['internal_val_eval_all']['NLL']) for k, _, _ in present]
        cols = [col for _, _, col in present]

        bars = ax1.bar(x - 0.18, ba4, width=0.35, color=cols, edgecolor='black', linewidth=0.7, label='BA4 (%) [left]')
        for b, val in zip(bars, ba4):
            ax1.text(b.get_x() + b.get_width() / 2.0, val + 0.18, f'{val:.2f}%', ha='center', va='bottom', fontsize=8.3, fontweight='bold')

        ax1_r = ax1.twinx()
        ax1_r.spines['top'].set_visible(False)
        ax1_r.grid(False)
        bars_n = ax1_r.bar(x + 0.18, nll, width=0.35, color=cols, alpha=0.45, hatch='//', edgecolor='black', linewidth=0.7, label='NLL [right]')
        for b, val in zip(bars_n, nll):
            ax1_r.text(b.get_x() + b.get_width() / 2.0, val + 0.004, f'{val:.3f}', ha='center', va='bottom', fontsize=8.0)

        ax1.set_xticks(x)
        ax1.set_xticklabels([lbl for _, lbl, _ in present])
        ax1.set_ylabel('Balanced 4-Class Accuracy (%)')
        ax1_r.set_ylabel('Cross-Entropy NLL (lower is better)')
        bmin, bmax = min(ba4), max(ba4)
        ax1.set_ylim(max(0.0, bmin - 3.5), min(100.0, bmax + 2.5))
        nmin, nmax = min(nll), max(nll)
        ax1_r.set_ylim(max(0.0, nmin - 0.06), nmax + 0.07)
        ax1.set_title('Stage-I Controlled Benchmark (internal_val_eval, 4.8k)')
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax1_r.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc='upper left', frameon=True)

    # Panel B: Low-Label Efficiency (10%, 25%, 100%)
    fracs = [10, 25, 100]
    ssl_keys = [f'ssl300k_jepa_ft_{f}pct' for f in fracs]
    sup_keys = ['sup_scratch_10pct', 'sup_scratch_25pct', 'sup_scratch_20p']
    pts_ssl = [(f, models[k]['internal_val_eval_all']['BA4_pct'], models[k]['internal_val_eval_all']['NLL']) for f, k in zip(fracs, ssl_keys) if k in models]
    pts_sup = [(f, models[k]['internal_val_eval_all']['BA4_pct'], models[k]['internal_val_eval_all']['NLL']) for f, k in zip(fracs, sup_keys) if k in models]

    if pts_ssl or pts_sup:
        xpos = {10: 0, 25: 1, 100: 2}
        if pts_ssl:
            xs = [xpos[p[0]] for p in pts_ssl]
            ys = [p[1] for p in pts_ssl]
            ax2.plot(xs, ys, marker='o', linewidth=2.2, markersize=7, color=PALETTE['blue'], label='PinpointSSL-JEPA Fine-Tune')
            for xi, yi in zip(xs, ys):
                ax2.text(xi, yi - 0.30, f'{yi:.2f}%', ha='center', va='top', fontsize=8.5, color=PALETTE['blue'], fontweight='bold')
        if pts_sup:
            xs = [xpos[p[0]] for p in pts_sup]
            ys = [p[1] for p in pts_sup]
            ax2.plot(xs, ys, marker='s', linestyle='--', linewidth=2.0, markersize=6.5, color=PALETTE['orange'], label='Supervised From Scratch')
            for xi, yi in zip(xs, ys):
                ax2.text(xi, yi + 0.28, f'{yi:.2f}%', ha='center', va='bottom', fontsize=8.5, color=PALETTE['orange'], fontweight='bold')
        if 'sup_scratch_equal_compute' in models:
            eq_b = float(models['sup_scratch_equal_compute']['internal_val_eval_all']['BA4_pct'])
            ax2.axhline(eq_b, color=PALETTE['grey'], linestyle=':', linewidth=1.6, label=f'Scratch Equal-Compute (100%): {eq_b:.2f}%')
        ax2.set_xticks([0, 1, 2])
        ax2.set_xticklabels(['10%\n(2,560)', '25%\n(6,400)', '100%\n(25,600)'])
        ax2.set_ylim(50.8, 59.5)
        ax2.set_xlabel('Labeled Training Subset (internal_train)')
        ax2.set_ylabel('Balanced 4-Class Accuracy (%)')
        ax2.set_title('Label-Efficiency Scaling on internal_val_eval')
        ax2.legend(loc='lower right', frameon=True)

    fig.suptitle('Experiment L2: PinpointSSL-JEPA Equal-Compute Benchmark & Label Efficiency', fontsize=13, fontweight='bold', y=1.02)
    return _save_fig(fig, fig_dir / 'fig1_l2_ssl_equal_compute_and_label_efficiency.png')


def plot_l2_precal_and_pv_calibration(l2_s1: dict[str, Any], fig_dir: Path) -> str | None:
    ssl_pre = l2_s1.get('ssl_pretraining', {})
    models = l2_s1.get('models', {})
    if not ssl_pre and not models:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.1), gridspec_kw={'wspace': 0.30})

    # Panel A: Unlabeled Cross-Magnet-0 Pre-Calibration
    variants = [
        ('ssl_full', 'PinpointSSL-JEPA\n(Full)', PALETTE['blue']),
        ('abl_ssl_random_point_mask_shortcut', 'Abl: Random\nPoint Mask', PALETTE['amber']),
        ('abl_ssl_no_spec_precal', 'Abl: No Spec\nPre-Calib', PALETTE['lightgrey']),
    ]
    v_pres = [(k, lbl, col) for k, lbl, col in variants if k in ssl_pre and ssl_pre[k].get('precalibration')]
    if v_pres:
        x = np.arange(len(v_pres))
        aurocs, r_sy, r_ty = [], [], []
        for k, _, _ in v_pres:
            pc = ssl_pre[k]['precalibration']
            ev = pc.get('ssl_val_calib') or pc.get('internal_val_eval') or {}
            aurocs.append(float(ev.get('mu_presence_AUROC', 0.0)))
            r_sy.append(float(ev.get('ln_abs_sy_pearson_r', 0.0)))
            r_ty.append(float(ev.get('exit_slope_ty_pearson_r', 0.0)))
        w = 0.25
        b1 = ax1.bar(x - w, aurocs, width=w, color=PALETTE['blue'], edgecolor='black', linewidth=0.6, label='Penetrating-mu AUROC')
        b2 = ax1.bar(x, r_sy, width=w, color=PALETTE['green'], edgecolor='black', linewidth=0.6, label='ln|sagitta_y| Pearson r')
        b3 = ax1.bar(x + w, r_ty, width=w, color=PALETTE['purple'], edgecolor='black', linewidth=0.6, label='Exit Slope t_y Pearson r')
        for grp in (b1, b2, b3):
            for b in grp:
                val = b.get_height()
                if val >= 0:
                    ax1.text(b.get_x() + b.get_width() / 2.0, val + 0.015, f'{val:.3f}', ha='center', va='bottom', fontsize=7.8)
                else:
                    ax1.text(b.get_x() + b.get_width() / 2.0, val - 0.02, f'{val:.3f}', ha='center', va='top', fontsize=7.8)
        ax1.axhline(0.0, color='black', linewidth=0.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels([lbl for _, lbl, _ in v_pres])
        ax1.set_ylim(-0.42, 1.12)
        ax1.set_ylabel('Unlabeled Pre-Calibration Metric')
        ax1.set_title('Task 3: Cross-Magnet-0 Spectrometer Drop Pre-Calibration')
        ax1.legend(loc='upper right', frameon=True)
    else:
        ax1.text(0.5, 0.5, 'Pre-calibration metrics unavailable', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Task 3: Cross-Magnet-0 Spectrometer Drop Pre-Calibration')

    # Panel B: 9/9 Student-t(nu=3) PV Pull Audit on ssl300k_jepa_ft_100pct (or best available)
    rep_key = 'ssl300k_jepa_ft_100pct' if 'ssl300k_jepa_ft_100pct' in models else next(iter(models), None)
    if rep_key and models[rep_key].get('pv_audit'):
        pva = models[rep_key]['pv_audit']
        regs = [('All_valid', 'All Valid PV'), ('FrontPinpoint_30mm', 'Front 30mm'), ('IntermPinpoint_150mm', 'Interm 150mm')]
        regs_pres = [(rk, rlbl) for rk, rlbl in regs if rk in pva]
        x = np.arange(len(regs_pres))
        w = 0.24
        ax_cols = [PALETTE['blue'], PALETTE['green'], PALETTE['orange']]
        for ai, aname in enumerate(('x', 'y', 'z')):
            vals = [float(pva[rk]['p68_abs_pull'][ai]) for rk, _ in regs_pres]
            bars = ax2.bar(x + (ai - 1) * w, vals, width=w, color=ax_cols[ai], edgecolor='black', linewidth=0.6, label=f'PV {aname} 68% |pull|')
            for b, v in zip(bars, vals):
                ax2.text(b.get_x() + b.get_width() / 2.0, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=7.8)
        ax2.axhspan(0.85, 1.15, color=PALETTE['green'], alpha=0.14, label='Target Band [0.85, 1.15]')
        ax2.axhline(1.0, color='black', linestyle='--', linewidth=1.0)
        ax2.set_xticks(x)
        ax2.set_xticklabels([
            f"{rlbl}\n(med |dz|={pva[rk]['median_abs_dz_mm']:.2f} mm)"
            for rk, rlbl in regs_pres
        ])
        ax2.set_ylim(0.5, 1.45)
        ax2.set_ylabel('68% Quantile of Normalized |Pull| / q_68(t_3)')
        ax2.set_title(f'Student-t(nu=3) PV Calibration ({rep_key})')
        ax2.legend(loc='upper right', frameon=True, ncol=2)

    fig.suptitle('Experiment L2: Spectrometer Pre-Calibration & Primary Vertex Pull Audit', fontsize=13, fontweight='bold', y=1.02)
    return _save_fig(fig, fig_dir / 'fig2_l2_unlabeled_probe_and_pv_calibration.png')


def plot_l1_gn2_summary(l1_s1: dict[str, Any], fig_dir: Path) -> str | None:
    models = l1_s1.get('models', {}) or l1_s1.get('variants', {})
    if not models:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.3), gridspec_kw={'wspace': 0.42})
    names = list(models.keys())
    ba4_vals, nll_vals, tau1_vals = [], [], []
    for k in names:
        rec = models[k]
        ev = rec.get('internal_val_eval_all') or rec.get('val_eval') or rec.get('metrics') or {}
        ba4_vals.append(float(ev.get('BA4_pct', 0.0)))
        nll_vals.append(float(ev.get('NLL', 0.0)))
        t1 = rec.get('tau_tail', {}).get('within_slice_1pct_All', {}).get('tau_eff_pct')
        if t1 is None:
            t1 = rec.get('tau_eff_1pct_pct', 0.0)
        tau1_vals.append(float(t1))

    x = np.arange(len(names))
    cols = [PALETTE['grey'] if 'ctrl' in n else (PALETTE['blue'] if 'full' in n else PALETTE['sky']) for n in names]
    bars = ax1.bar(x - 0.18, ba4_vals, width=0.35, color=cols, edgecolor='black', linewidth=0.7, label='BA4 (%) [left]')
    for b, v in zip(bars, ba4_vals):
        ax1.text(b.get_x() + b.get_width() / 2.0, v + 0.15, f'{v:.2f}%', ha='center', va='bottom', fontsize=8.2)
    ax1_r = ax1.twinx()
    ax1_r.spines['top'].set_visible(False)
    ax1_r.grid(False)
    bars2 = ax1_r.bar(x + 0.18, tau1_vals, width=0.35, color=cols, alpha=0.45, hatch='//', edgecolor='black', linewidth=0.7, label='Tau Eff @ 1% FPR (%) [right]')
    for b, v in zip(bars2, tau1_vals):
        ax1_r.text(b.get_x() + b.get_width() / 2.0, v + 0.25, f'{v:.1f}%', ha='center', va='bottom', fontsize=8.0)

    ax1.set_xticks(x)
    ax1.set_xticklabels([n.replace('_', '\n') for n in names], fontsize=7.8)
    ax1.set_ylim(max(0.0, min(ba4_vals) - 2.5), min(100.0, max(ba4_vals) + 2.0))
    if tau1_vals and max(tau1_vals) > 0:
        ax1_r.set_ylim(max(0.0, min(tau1_vals) - 2.5), min(100.0, max(tau1_vals) + 2.5))
    ax1.set_ylabel('Balanced 4-Class Accuracy (%)')
    ax1_r.set_ylabel('Tau Efficiency @ 1% FPR (%)')
    ax1.set_title('L1 Causal Control vs GN2 Full & Ablations (L1_internal_val_eval)')

    # Panel B: Microscopic GN2 Probe AUCs if present, else NLL comparison
    probes = l1_s1.get('microscopic_probes') or {}
    if not probes:
        for k in names:
            for pk in ('microscopic_probe_aucs_val_eval', 'microscopic_probes'):
                if pk in models[k]:
                    probes[k] = models[k][pk]
                    break
    if probes:
        pkeys = list(probes.keys())
        px = np.arange(len(pkeys))
        ghost = [float(probes[k].get('purity_auroc', probes[k].get('ghost_auroc', probes[k].get('ghost_vs_real_auroc', 0.0)))) for k in pkeys]
        conv = [float(probes[k].get('conversion_auroc', probes[k].get('conversion_vs_primary_auroc', 0.0))) for k in pkeys]
        had = [float(probes[k].get('tau_prong_auroc', probes[k].get('tau_or_hadronic_auroc', probes[k].get('hf_disp_auroc', 0.5)))) for k in pkeys]
        w = 0.25
        ax2.bar(px - w, ghost, width=w, color=PALETTE['blue'], edgecolor='black', linewidth=0.6, label='Purity (Real vs Ghost) AUROC')
        ax2.bar(px, conv, width=w, color=PALETTE['green'], edgecolor='black', linewidth=0.6, label='Conversion e+e- AUROC')
        ax2.bar(px + w, had, width=w, color=PALETTE['purple'], edgecolor='black', linewidth=0.6, label='Tau Prong AUROC')
        ax2.set_xticks(px)
        ax2.set_xticklabels([k.replace('_', '\n') for k in pkeys], fontsize=7.8)
        ax2.set_ylim(0.45, 1.18)
        ax2.set_ylabel('Microscopic Origin Probe AUROC')
        ax2.set_title('GN2 Auxiliary Tracklet Origin Probes')
        ax2.legend(loc='upper center', ncol=2, frameon=True, fontsize=7.8)
    else:
        bars_n = ax2.bar(x, nll_vals, width=0.5, color=cols, edgecolor='black', linewidth=0.7)
        for b, v in zip(bars_n, nll_vals):
            ax2.text(b.get_x() + b.get_width() / 2.0, v + 0.005, f'{v:.4f}', ha='center', va='bottom', fontsize=8.4)
        ax2.set_xticks(x)
        ax2.set_xticklabels([n.replace('_', '\n') for n in names], fontsize=8.2)
        ax2.set_ylabel('Cross-Entropy NLL (lower is better)')
        ax2.set_title('L1 Stage-I Cross-Entropy NLL')

    fig.suptitle('Experiment L1: TrackletIPFormer-GN2 Causal Control & Microscopic Origin Probes', fontsize=13, fontweight='bold', y=1.02)
    return _save_fig(fig, fig_dir / 'fig3_l1_gn2_causal_ablation_and_probes.png')


def plot_stage2_synergy_and_robustness(
    l2_s2: dict[str, Any] | None,
    l1_s2: dict[str, Any] | None,
    fig_dir: Path,
) -> str | None:
    if l2_s2 is None and l1_s2 is None:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.4, 5.8), gridspec_kw={'width_ratios': [1.22, 1.0], 'wspace': 0.68})

    def _l1_all_slice(block: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(block, dict):
            return {}
        if 'BA4_pct' in block:
            return block
        return block.get('slices', {}).get('All_Tune6k', {})

    # Panel A: Tune6k Comparison & Blends
    entries: list[tuple[str, float, float, str]] = []
    l1_raw = _l1_all_slice((l1_s2 or {}).get('tune6k_ens3_raw') or (l1_s2 or {}).get('gn2_ens3_raw_tune6k'))
    l1_dro = _l1_all_slice((l1_s2 or {}).get('gn2_ens3_dro_tune6k'))
    l1_dual_dro = _l1_all_slice((l1_s2 or {}).get('blend_gn2_dual_dro_tune6k'))
    if 'BA4_pct' in l1_raw:
        entries.append(('L1 GN2\n(3-seed)', float(l1_raw['BA4_pct']), float(l1_raw['NLL']), PALETTE['green']))
    if 'BA4_pct' in l1_dro:
        entries.append(('L1 GN2\n+ KL-DRO', float(l1_dro['BA4_pct']), float(l1_dro['NLL']), PALETTE['sky']))
    if l2_s2 and 'tune6k_ens3_raw' in l2_s2:
        m = l2_s2['tune6k_ens3_raw']
        entries.append(('L2 SSL-JEPA\n(3-seed)', float(m['BA4_pct']), float(m['NLL']), PALETTE['blue']))
    blends = dict((l2_s2 or {}).get('blends') or (l1_s2 or {}).get('blends') or {})
    if 'BA4_pct' in l1_dual_dro:
        blends['blend_gn2_dual_dro_tune6k'] = l1_dual_dro
    for bk, blbl, bcol in (
        ('blend_gn2_dual_dro_tune6k', 'L1 + Dual\n+ KL-DRO', PALETTE['orange']),
        ('blend_50_50_jepa_dual', '50/50\nJEPA+Dual', PALETTE['sky']),
        ('blend_50_50_jepa_l1', '50/50\nJEPA+L1', PALETTE['amber']),
        ('blend_3way_l1_dual_jepa', '3-Way\nL1+Dual+JEPA', PALETTE['purple']),
    ):
        if bk in blends and 'BA4_pct' in blends[bk]:
            entries.append((blbl, float(blends[bk]['BA4_pct']), float(blends[bk]['NLL']), bcol))

    if entries:
        x = np.arange(len(entries))
        ba4 = [e[1] for e in entries]
        nll = [e[2] for e in entries]
        cols = [e[3] for e in entries]
        bars = ax1.bar(x - 0.18, ba4, width=0.35, color=cols, edgecolor='black', linewidth=0.7, label='BA4 (%) [left]')
        for b, v in zip(bars, ba4):
            ax1.text(b.get_x() + b.get_width() / 2.0, v + 0.10, f'{v:.2f}%', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
        ax1_r = ax1.twinx()
        ax1_r.spines['top'].set_visible(False)
        ax1_r.grid(False)
        bars2 = ax1_r.bar(x + 0.18, nll, width=0.35, color=cols, alpha=0.45, hatch='//', edgecolor='black', linewidth=0.7, label='NLL [right]')
        for b, v in zip(bars2, nll):
            ax1_r.text(b.get_x() + b.get_width() / 2.0, v + 0.003, f'{v:.3f}', ha='center', va='bottom', fontsize=7.3)
        ax1.set_xticks(x)
        ax1.set_xticklabels([e[0] for e in entries], fontsize=7.6)
        ax1.set_ylim(max(0.0, min(ba4) - 2.5), min(100.0, max(ba4) + 2.0))
        ax1_r.set_ylim(max(0.0, min(nll) - 0.04), max(nll) + 0.05)
        ax1.set_ylabel('Tune6k Balanced 4-Class Accuracy (%)')
        ax1_r.set_ylabel('Tune6k Cross-Entropy NLL')
        ax1.set_title('Stage-II Tune6k (6,000 events): Individual & Blend Synergy')
        h1, l1_leg = ax1.get_legend_handles_labels()
        h2, l2_leg = ax1_r.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1_leg + l2_leg, loc='upper left', ncol=2, frameon=True, fontsize=7.8)

    # Panel B: OOD & Stress BA4 across L2/L1 Stage-II
    l2_ood = (l2_s2 or {}).get('ood_ens3') or {}
    l1_ood = (l1_s2 or {}).get('ood_ens3') or (l1_s2 or {}).get('ood_evaluation') or {}
    l2_str = (l2_s2 or {}).get('stress_shard080_ens3') or {}
    l1_str = (l1_s2 or {}).get('stress_shard080_ens3') or (l1_s2 or {}).get('shard080_stress_tests') or {}
    ood_src = l2_ood or l1_ood
    stress_src = l2_str or l1_str
    rob_labels, l2_vals, l1_vals = [], [], []
    for k, v in ood_src.items():
        if isinstance(v, dict) and 'BA4_pct' in v:
            rob_labels.append(k.replace('gen_ood_', 'OOD: '))
            l2_vals.append(float(l2_ood.get(k, {}).get('BA4_pct', np.nan)))
            l1_entry = l1_ood.get(k, {})
            if isinstance(l1_entry, dict) and 'gn2_raw' in l1_entry:
                l1_entry = l1_entry['gn2_raw']
            l1_vals.append(float(l1_entry.get('BA4_pct', np.nan)) if isinstance(l1_entry, dict) else np.nan)
    for k, v in stress_src.items():
        if isinstance(v, dict) and 'BA4_pct' in v:
            rob_labels.append(f'Stress: {k}')
            l2_vals.append(float(l2_str.get(k, {}).get('BA4_pct', np.nan)))
            l1_entry = l1_str.get(k, {})
            if isinstance(l1_entry, dict) and 'gn2_raw' in l1_entry:
                l1_entry = l1_entry['gn2_raw']
            l1_vals.append(float(l1_entry.get('BA4_pct', np.nan)) if isinstance(l1_entry, dict) else np.nan)

    if rob_labels:
        y = np.arange(len(rob_labels))
        h = 0.36
        if any(np.isfinite(l2_vals)):
            ax2.barh(y - h / 2, [0.0 if not np.isfinite(v) else v for v in l2_vals], height=h, color=PALETTE['blue'], edgecolor='black', linewidth=0.5, label='L2 PinpointSSL-JEPA')
        if any(np.isfinite(l1_vals)):
            ax2.barh(y + h / 2, [0.0 if not np.isfinite(v) else v for v in l1_vals], height=h, color=PALETTE['green'], edgecolor='black', linewidth=0.5, label='L1 TrackletIPFormer-GN2')
        ax2.set_yticks(y)
        ax2.set_yticklabels(rob_labels, fontsize=8.0)
        ax2.invert_yaxis()
        ax2.set_xlim(0.0, 68.0)
        ax2.set_xlabel('Balanced 4-Class Accuracy (%)')
        ax2.set_title('Stage-II OOD & Shard-080 Stress Robustness')
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=True)

    fig.suptitle('Stage 328 Stage-II Tune6k Synergy, OOD Generalization & Stress Robustness', fontsize=13, fontweight='bold', y=1.02)
    return _save_fig(fig, fig_dir / 'fig4_stage328_stage2_synergy_ood_stress.png')


def main() -> None:
    guard = SealAuditGuard()
    guard.__enter__()
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--l2-s1', type=Path, default=DEFAULT_L2_S1)
        parser.add_argument('--l2-s2', type=Path, default=DEFAULT_L2_S2)
        parser.add_argument('--l1-s1', type=Path, default=DEFAULT_L1_S1)
        parser.add_argument('--l1-s2', type=Path, default=DEFAULT_L1_S2)
        parser.add_argument('--out-dir', type=Path, default=DEFAULT_FIG_DIR)
        args = parser.parse_args()

        _configure_style()
        l2_s1 = _load_json_if_exists(args.l2_s1)
        l2_s2 = _load_json_if_exists(args.l2_s2)
        l1_s1 = _load_json_if_exists(args.l1_s1)
        l1_s2 = _load_json_if_exists(args.l1_s2)

        generated: list[str] = []
        if l2_s1 is not None:
            for fn in (plot_l2_equal_compute_and_label_efficiency, plot_l2_precal_and_pv_calibration):
                p = fn(l2_s1, args.out_dir)
                if p:
                    generated.append(p)
        if l1_s1 is not None:
            p = plot_l1_gn2_summary(l1_s1, args.out_dir)
            if p:
                generated.append(p)
        if l2_s2 is not None or l1_s2 is not None:
            p = plot_stage2_synergy_and_robustness(l2_s2, l1_s2, args.out_dir)
            if p:
                generated.append(p)

        print(json.dumps({'generated_figures': generated, 'count': len(generated)}, indent=2))
    finally:
        guard.__exit__(None, None, None)


if __name__ == '__main__':
    main()
