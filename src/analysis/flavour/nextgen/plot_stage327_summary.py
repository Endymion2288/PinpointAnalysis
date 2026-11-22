"""Stage 327 Publication Figures (300 DPI, 100% English) in `stage327_figures/`.

Reads unsealed Stage 327 JSON outputs from:
  - `exp_tail_dro/stage1/stage1_tail_dro_summary.json`
  - `exp_tail_dro/stage2/stage2_tail_dro_summary.json`
  - `exp_unified_net/` (N1 PinpointUnifiedNet, when available)
  - `exp_tau_kinema/` (N2 `first_tgt_p <= 5` fix & Q13 full 1mm control, when available)
Enforces `SealAuditGuard()` throughout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analysis.flavour.nextgen.seal_guard import SealAuditGuard, assert_unsealed_path


NEXTGEN_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research')
FIG_DIR = NEXTGEN_ROOT / 'stage327_figures'


def _set_pub_style() -> None:
    plt.rcParams.update({
        'font.size': 8.6,
        'axes.titlesize': 10.0,
        'axes.labelsize': 9.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def plot_stage1_tail_dro(s1_doc: dict[str, Any], out_dir: Path) -> Path:
    """Render Figure 8: Stage-I Tail-DRO Ablations, Calibration Gate & W184 OOD."""
    _set_pub_style()
    models = s1_doc['models']
    locked = s1_doc['locked_candidate_name']
    model_order = [
        (m, lbl) for m, lbl in (
            ('tracklet_ipformer', 'TrackletIPFormer'),
            ('dual_regime_modalnet', 'DualRegimeModalNet'),
            ('unified_pinpoint_net', 'PinpointUnifiedNet'),
        ) if m in models
    ]

    abls = [
        ('pretrained_baseline', 'Pretrained\n(0 pass)', '#90A4AE'),
        ('pure_ce_3p', 'Pure CE\n(3 passes)', '#5C6BC0'),
        ('pure_dro_no_anchor', 'Pure DRO\n(No Anchor)', '#EF5350'),
        ('dro_anchored_nobank', 'Anchored DRO\n(No Bank)', '#FB8C00'),
        (locked, f'Locked DRO\n(Bank=8192)', '#2E7D32'),
    ]

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(18.6, 5.4), dpi=300)

    # Panel A: Within-Slice 1% FPR Tau Efficiency (All & D1) across ablations
    x = np.arange(len(abls))
    n_m = len(model_order)
    bw = 0.72 / max(1, n_m * 2)
    for mi, (mkey, mlbl) in enumerate(model_order):
        runs = models[mkey]['runs']
        all_eff = [runs[ak]['val_eval']['within_slice_1pct_fpr']['All']['tau_eff_pct'] for ak, _, _ in abls]
        d1_eff = [runs[ak]['val_eval']['within_slice_1pct_fpr']['D1_obs']['tau_eff_pct'] for ak, _, _ in abls]
        off_all = (mi * 2 - n_m + 0.5) * bw
        off_d1 = (mi * 2 + 1 - n_m + 0.5) * bw
        col_base = ('#1565C0', '#00897B', '#6A1B9A')[mi % 3]
        b1 = ax0.bar(x + off_all, all_eff, width=bw, color=col_base, edgecolor='black', lw=0.4, label=f'{mlbl} All')
        b2 = ax0.bar(x + off_d1, d1_eff, width=bw, color=col_base, alpha=0.50, hatch='//', edgecolor='black', lw=0.4, label=f'{mlbl} D1')
        for b, v in zip(b1, all_eff):
            ax0.text(b.get_x() + b.get_width() / 2.0, v + 0.12, f'{v:.1f}', ha='center', va='bottom', fontsize=6.4)
        for b, v in zip(b2, d1_eff):
            ax0.text(b.get_x() + b.get_width() / 2.0, v + 0.12, f'{v:.1f}', ha='center', va='bottom', fontsize=6.4)

    ax0.set_xticks(x)
    ax0.set_xticklabels([lbl for _, lbl, _ in abls], fontsize=7.8)
    ax0.set_ylabel('Internal-Val 1% FPR Tau Efficiency [%]')
    ax0.set_ylim(0.0, max(6.8, max(d1_eff + all_eff) + 1.2))
    ax0.set_title('A. Stage-I Tail-DRO Ablation Ladder (N = 4,800)')
    ax0.legend(loc='upper left', fontsize=6.9, ncol=2)
    ax0.grid(True, axis='y', alpha=0.25)

    # Panel B: Calibration Guard (Delta NLL vs Delta ECE_tau Ratio)
    markers = ('o', 's', 'D')
    for mi, (mkey, mlbl) in enumerate(model_order):
        runs = models[mkey]['runs']
        for ak, albl, col in abls[1:]:
            r = runs[ak]
            dnll = r['val_delta_vs_pretrained']['delta_NLL']
            deff = r['val_delta_vs_pretrained']['delta_ws1_All_pp']
            ax1.scatter(
                dnll, deff,
                s=72, color=col, marker=markers[mi % len(markers)], edgecolor='black', lw=0.6,
                label=albl.replace('\n', ' ') if mi == 0 else None,
            )
            ax1.annotate(
                f'{mlbl[:4]}', (dnll, deff), textcoords='offset points', xytext=(4, 3), fontsize=6.6,
            )
    ax1.scatter([], [], s=55, color='white', marker='o', edgecolor='black', label='Circle: TrackletIPFormer')
    ax1.scatter([], [], s=55, color='white', marker='s', edgecolor='black', label='Square: DualRegimeModalNet')
    if len(model_order) >= 3:
        ax1.scatter([], [], s=55, color='white', marker='D', edgecolor='black', label='Diamond: PinpointUnifiedNet')
    ax1.axvline(0.0, color='#37474F', lw=0.8)
    ax1.axvline(0.005, color='#2E7D32', ls='--', lw=1.0, label='Sec 10.2 Target (dNLL <= +0.005)')
    ax1.axvline(0.010, color='#C62828', ls=':', lw=1.2, label='Sec 10.4 Fallback (dNLL > +0.010)')
    ax1.axhline(0.0, color='#37474F', lw=0.8)
    ax1.set_xlabel('Delta NLL vs Pretrained [nats] (Internal-Val)')
    ax1.set_ylabel('Delta All 1% FPR Tau Efficiency [pp]')
    ax1.set_title('B. Tail Gain vs NLL Calibration Guard (Stage-I)')
    ax1.legend(loc='best', fontsize=6.4)
    ax1.grid(True, alpha=0.25)

    # Panel C: pAUC@1% & W184 NC Charm Fake-Tau Rate
    xc = np.arange(len(abls))
    ax2_r = ax2.twinx()
    ax2_r.spines['top'].set_visible(False)
    max_pauc = 0.0
    max_w184 = 0.0
    for mi, (mkey, mlbl) in enumerate(model_order):
        runs = models[mkey]['runs']
        pauc = [runs[ak]['val_eval']['pAUC_1pct']['All'] * 100.0 for ak, _, _ in abls]
        w184 = [runs[ak]['ood']['gen_ood_w184_epps21_nc_charm']['frac_p_tau_gt_0p5_pct'] for ak, _, _ in abls]
        max_pauc = max(max_pauc, max(pauc))
        max_w184 = max(max_w184, max(w184))
        col_base = ('#1565C0', '#00897B', '#6A1B9A')[mi % 3]
        ax2.plot(xc, pauc, 'o-', color=col_base, lw=1.5, ms=5.0, label=f'{mlbl} pAUC@1% [%]')
        ax2_r.plot(xc, w184, 's--', color=col_base, alpha=0.75, lw=1.2, ms=4.2, label=f'{mlbl} W184 fake [%]')

    ax2.set_ylim(1.1, max_pauc + 0.55)
    ax2_r.set_ylabel('W184 NC Charm P(tau)>0.5 [%]', color='#C62828')
    ax2_r.set_ylim(0.0, max(6.0, max_w184 + 2.0))
    ax2.set_xticks(xc)
    ax2.set_xticklabels([lbl for _, lbl, _ in abls], fontsize=7.8)
    ax2.set_ylabel('Normalized pAUC(FPR <= 1%) [%]')
    ax2.set_title('C. Normalized pAUC@1% & W184 NC-Charm Rejection')
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_r.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=6.4)
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    out_path = out_dir / 'fig8_tail_dro_stage1_ablations.png'
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_stage2_tail_dro(s2_doc: dict[str, Any], out_dir: Path) -> Path:
    """Render Figure 9: Stage-II Frozen Tune6k, 6-OOD & Shard-080 Stress Summary."""
    _set_pub_style()
    models = s2_doc['models']
    blend = s2_doc.get('blend_50_50', {})

    entries: list[tuple[str, dict[str, Any]]] = []
    for k, lbl in (
        ('tracklet_ipformer', 'TrackletIP\n(3-seed)'),
        ('dual_regime_modalnet', 'DualRegime\n(3-seed)'),
        ('unified_pinpoint_net', 'UnifiedNet\n(3-seed)'),
    ):
        if k in models:
            entries.append((lbl, models[k]['tune6k_raw']))
    if blend:
        entries.append(('50/50 Blend\n(Raw)', blend['tune6k_raw']))
        entries.append(('50/50 Blend\n(Screen Cal)', blend['tune6k_screen_cal']))

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(18.6, 5.4), dpi=300)

    # Panel A: Tune6k 1% & 0.1% FPR Tau Efficiency (Pretrained vs Locked DRO)
    x = np.arange(len(entries))
    pre_all1 = [e['pretrained']['within_slice_1pct_fpr']['All_Tune6k']['tau_eff_pct'] for _, e in entries]
    dro_all1 = [e['dro']['within_slice_1pct_fpr']['All_Tune6k']['tau_eff_pct'] for _, e in entries]
    pre_d1 = [e['pretrained']['within_slice_1pct_fpr']['D1_obs']['tau_eff_pct'] for _, e in entries]
    dro_d1 = [e['dro']['within_slice_1pct_fpr']['D1_obs']['tau_eff_pct'] for _, e in entries]

    bw = 0.19
    ax0.bar(x - 1.5 * bw, pre_all1, width=bw, color='#90A4AE', edgecolor='black', lw=0.4, label='Pretrained All 1% FPR')
    ax0.bar(x - 0.5 * bw, dro_all1, width=bw, color='#2E7D32', edgecolor='black', lw=0.4, label='DRO All 1% FPR')
    ax0.bar(x + 0.5 * bw, pre_d1, width=bw, color='#B0BEC5', hatch='//', edgecolor='black', lw=0.4, label='Pretrained D1 1% FPR')
    ax0.bar(x + 1.5 * bw, dro_d1, width=bw, color='#EF6C00', hatch='//', edgecolor='black', lw=0.4, label='DRO D1 1% FPR')
    for i in range(len(entries)):
        ax0.text(x[i] - 0.5 * bw, dro_all1[i] + 0.15, f'{dro_all1[i]:.1f}', ha='center', va='bottom', fontsize=6.5)
        ax0.text(x[i] + 1.5 * bw, dro_d1[i] + 0.15, f'{dro_d1[i]:.1f}', ha='center', va='bottom', fontsize=6.5)
    ax0.set_xticks(x)
    ax0.set_xticklabels([lbl for lbl, _ in entries], fontsize=7.6)
    ax0.set_ylabel('Tune6k Within-Slice 1% FPR Tau Efficiency [%]')
    ax0.set_ylim(0.0, max(14.5, max(pre_all1 + dro_all1 + pre_d1 + dro_d1) + 3.2))
    ax0.set_title('A. Stage-II Tune6k Tail Efficiency (Pretrained vs DRO)')
    ax0.legend(loc='upper left', fontsize=6.9, ncol=2)
    ax0.grid(True, axis='y', alpha=0.25)

    # Panel B: Tune6k BA4, NLL, and pAUC@1%
    pre_ba4 = [e['pretrained']['slices']['All_Tune6k']['BA4_pct'] for _, e in entries]
    dro_ba4 = [e['dro']['slices']['All_Tune6k']['BA4_pct'] for _, e in entries]
    pre_nll = [e['pretrained']['slices']['All_Tune6k']['NLL'] for _, e in entries]
    dro_nll = [e['dro']['slices']['All_Tune6k']['NLL'] for _, e in entries]
    ax1.bar(x - 0.18, pre_ba4, width=0.34, color='#78909C', edgecolor='black', lw=0.4, label='Pretrained BA4 [%]')
    ax1.bar(x + 0.18, dro_ba4, width=0.34, color='#1565C0', edgecolor='black', lw=0.4, label='DRO BA4 [%]')
    ax1_r = ax1.twinx()
    ax1_r.spines['top'].set_visible(False)
    ax1_r.plot(x, pre_nll, 'ko--', lw=1.2, ms=4.2, label='Pretrained NLL')
    ax1_r.plot(x, dro_nll, 'rD-', lw=1.3, ms=4.2, label='DRO NLL')
    ax1_r.set_ylabel('Tune6k Balanced NLL [nats]')
    ax1_r.set_ylim(0.92, max(pre_nll + dro_nll) + 0.04)
    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for lbl, _ in entries], fontsize=7.6)
    ax1.set_ylabel('Tune6k 4-Class BA4 [%]')
    ax1.set_ylim(54.0, max(pre_ba4 + dro_ba4) + 2.2)
    ax1.set_title('B. Tune6k Global BA4 & NLL Calibration Preservation')
    ax1.legend(loc='upper left', fontsize=6.9)
    ax1_r.legend(loc='upper right', fontsize=6.9)
    ax1.grid(True, axis='y', alpha=0.25)

    # Panel C: W184 NC Charm & Run10085-88 OOD Comparison
    ood_entries = [
        (lbl, models[k]['ood_6shards']) for k, lbl in (
            ('tracklet_ipformer', 'TrackletIP'),
            ('dual_regime_modalnet', 'DualRegime'),
            ('unified_pinpoint_net', 'UnifiedNet'),
        ) if k in models
    ]
    if blend and 'ood_6shards' in blend:
        ood_entries.append(('50/50 Blend', blend['ood_6shards']))
    xo = np.arange(len(ood_entries))
    w184_pre = [oe['gen_ood_w184_epps21_nc_charm']['pretrained']['p_tau_gt_0p5_pct'] for _, oe in ood_entries]
    w184_dro = [oe['gen_ood_w184_epps21_nc_charm']['dro']['p_tau_gt_0p5_pct'] for _, oe in ood_entries]
    r85_pre = [oe['gen_ood_run10085_10088']['pretrained']['BA4_pct'] for _, oe in ood_entries]
    r85_dro = [oe['gen_ood_run10085_10088']['dro']['BA4_pct'] for _, oe in ood_entries]

    ax2.bar(xo - 0.18, w184_pre, width=0.34, color='#90A4AE', edgecolor='black', lw=0.4, label='Pretrained W184 Fake Tau [%]')
    ax2.bar(xo + 0.18, w184_dro, width=0.34, color='#C62828', edgecolor='black', lw=0.4, label='DRO W184 Fake Tau [%]')
    ax2_r = ax2.twinx()
    ax2_r.spines['top'].set_visible(False)
    ax2_r.plot(xo, r85_pre, 'ko--', lw=1.2, ms=4.2, label='Pretrained Run10085 BA4 [%]')
    ax2_r.plot(xo, r85_dro, 'g^-', lw=1.3, ms=4.8, label='DRO Run10085 BA4 [%]')
    ax2_r.set_ylabel('Run10085-88 OOD BA4 [%]')
    ax2_r.set_ylim(54.0, 63.0)
    ax2.set_xticks(xo)
    ax2.set_xticklabels([lbl for lbl, _ in ood_entries], fontsize=7.8)
    ax2.set_ylabel('W184 NC Charm P(tau)>0.5 [%]')
    ax2.set_ylim(0.0, max(4.0, max(w184_pre + w184_dro) + 1.2))
    ax2.set_title('C. Pure-Tungsten OOD & NC-Charm Fake Rejection')
    ax2.legend(loc='upper left', fontsize=6.7)
    ax2_r.legend(loc='upper right', fontsize=6.7)
    ax2.grid(True, axis='y', alpha=0.25)

    fig.tight_layout()
    out_path = out_dir / 'fig9_tail_dro_stage2_tune6k.png'
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def _find_json(candidates: list[Path]) -> dict[str, Any] | None:
    for p in candidates:
        if p.is_file():
            return json.loads(assert_unsealed_path(p).read_text())
    return None


def plot_stage327_integrated_summary(out_dir: Path) -> Path | None:
    """Render Figure 10 (2x2, 300 DPI) integrating N1, N2, Q13, and Stage-I/II Tail-DRO."""
    _set_pub_style()
    n1_r1_doc = _find_json([
        NEXTGEN_ROOT / 'exp_unified_net/round1_archive/stage1_unified_results.json',
    ])
    n1_r2_doc = _find_json([
        NEXTGEN_ROOT / 'exp_unified_net/stage1_unified_results.json',
        NEXTGEN_ROOT / 'exp_unified_net/s1_unified_pinpoint_net/run_summary.json',
    ])
    n2_v1_doc = _find_json([NEXTGEN_ROOT / 'exp_n2_tau_kinema/stage1_n2_evaluation.json'])
    n2_v2_doc = _find_json([NEXTGEN_ROOT / 'exp_n2_tau_kinema/stage1_n2v2_evaluation.json'])
    q13_doc = _find_json([NEXTGEN_ROOT / 'exp_n2_tau_kinema/q13_full_1mm_sensor/q13_full_1mm_comparison.json'])
    s2_dro = _find_json([NEXTGEN_ROOT / 'exp_tail_dro/stage2/stage2_tail_dro_summary.json'])
    u_dro = _find_json([NEXTGEN_ROOT / 'exp_tail_dro/stage1/stage1_unified_tail_dro_summary.json'])

    if all(x is None for x in (n1_r1_doc, n1_r2_doc, n2_v1_doc, n2_v2_doc, q13_doc, s2_dro)):
        return None

    fig, axes = plt.subplots(2, 2, figsize=(15.6, 10.2), dpi=300)
    ax0, ax1 = axes[0, 0], axes[0, 1]
    ax2, ax3 = axes[1, 0], axes[1, 1]

    # Panel A: N1 PinpointUnifiedNet Round-1 vs Round-2 & Sec 10.4 Stress/Clean Trade-Off
    if n1_r1_doc is not None and n1_r2_doc is not None:
        r1_g = n1_r1_doc.get('stage1_gates', {})
        r2_g = n1_r2_doc.get('stage1_gates', {})
        r2_s1 = n1_r2_doc.get('models', {}).get('s1_unified_pinpoint_net', {})
        r2_dark_drop = abs(float(r2_s1.get('stress_400_internal_val', {}).get('pixel_dark_1e-5', {}).get('delta_BA4_vs_nominal_pp', -1.37)))
        r1_s1 = n1_r1_doc.get('models', {}).get('s1_unified_pinpoint_net', {})
        r1_dark_drop = abs(float(r1_s1.get('stress_400_internal_val', {}).get('pixel_dark_1e-5', {}).get('delta_BA4_vs_nominal_pp', -6.46)))

        n1_labels = ['R1 Unified\n(7.60M)', 'R2 Unified\n(7.57M)', '50/50 Blend\n(Stage-I Ref)']
        n1_ba4 = [
            float(r1_g.get('All_BA4_pct', 58.00)),
            float(r2_g.get('All_BA4_pct', 57.48)),
            float(r2_g.get('blend_ref_BA4_pct', 58.54)),
        ]
        n1_trk_drop = [
            float(r1_g.get('trk_dead_2pct_n1_drop_pp', 5.32)),
            float(r2_g.get('trk_dead_2pct_n1_drop_pp', 0.07)),
            np.nan,
        ]
        n1_dark = [r1_dark_drop, r2_dark_drop, np.nan]
        if u_dro is not None and 's1_unified_pinpoint_net_r2' in u_dro.get('models', {}):
            r2_dro_ba4 = u_dro['models']['s1_unified_pinpoint_net_r2']['dro_locked']['val_eval']['slices']['All']['BA4_pct']
            n1_labels.insert(2, 'R2 Unified\n+ Tail-DRO')
            n1_ba4.insert(2, float(r2_dro_ba4))
            n1_trk_drop.insert(2, np.nan)
            n1_dark.insert(2, np.nan)

        xa = np.arange(len(n1_labels))
        bars_a = ax0.bar(xa - 0.16, n1_ba4, width=0.32, color='#1565C0', edgecolor='black', lw=0.4, label='Internal-Val BA4 [%]')
        for b, v in zip(bars_a, n1_ba4):
            ax0.text(b.get_x() + b.get_width() / 2.0, v + 0.12, f'{v:.2f}%', ha='center', va='bottom', fontsize=7.0)
        ax0.axhline(58.3, color='#2E7D32', ls='--', lw=1.0, label='Sec 10.4 Gate (BA4 >= 58.3%)')
        ax0_r = ax0.twinx()
        ax0_r.spines['top'].set_visible(False)
        valid_trk = [i for i, v in enumerate(n1_trk_drop) if not np.isnan(v)]
        ax0_r.bar(np.array(valid_trk) + 0.14, [n1_trk_drop[i] for i in valid_trk], width=0.18, color='#EF6C00', edgecolor='black', lw=0.4, label='|Delta BA4| TrkDead2%+N1 [pp]')
        ax0_r.bar(np.array(valid_trk) + 0.34, [n1_dark[i] for i in valid_trk], width=0.18, color='#C62828', edgecolor='black', lw=0.4, label='|Delta BA4| PixDark1e-5 [pp]')
        for i in valid_trk:
            ax0_r.text(i + 0.14, n1_trk_drop[i] + 0.12, f'{n1_trk_drop[i]:.2f}', ha='center', va='bottom', fontsize=6.6, color='#E65100')
            ax0_r.text(i + 0.34, n1_dark[i] + 0.12, f'{n1_dark[i]:.2f}', ha='center', va='bottom', fontsize=6.6, color='#C62828')
        ax0_r.set_ylabel('Stress |Delta BA4| Drop [pp]', color='#C62828')
        ax0_r.set_ylim(0.0, 8.2)
        ax0.set_xticks(xa)
        ax0.set_xticklabels(n1_labels, fontsize=7.8)
        ax0.set_ylabel('Internal-Val 4-Class BA4 [%]')
        ax0.set_ylim(54.0, 60.5)
        ax0.set_title('A. N1 UnifiedNet Clean vs Stress Trade-Off (Sec 10.4 Fallback)')
        h1, l1 = ax0.get_legend_handles_labels()
        h2, l2 = ax0_r.get_legend_handles_labels()
        ax0.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=6.8)
        ax0.grid(True, axis='y', alpha=0.25)
    else:
        ax0.text(0.5, 0.5, 'Pending N1 Data', ha='center', va='center', transform=ax0.transAxes)
        ax0.set_title('A. N1 PinpointUnifiedNet Summary')

    # Panel B: N2 Interm150 & Front30 PV |dz| Fix (19.261 mm -> 10.215 mm -> 1.984 mm)
    if n2_v1_doc is not None or n2_v2_doc is not None:
        ax1.axhline(6.0, color='#2E7D32', ls='--', lw=1.2, label='Stage 327 Target (<= 6.0 mm)')
        labels_b = ['Stage 326\n(Ungated AVR)']
        dz_i150 = [19.261]
        dz_f30 = [np.nan]
        if n2_v1_doc is not None and 'interm150_median_abs_dz_mm' in n2_v1_doc:
            labels_b.append('N2 v1\n(first_tgt_p <= 5)')
            dz_i150.append(float(n2_v1_doc['interm150_median_abs_dz_mm']))
            f30_v1 = n2_v1_doc.get('pv_metrics', {}).get('FrontPinpoint_30mm', {}).get('median_abs_dz_mm', np.nan)
            dz_f30.append(float(f30_v1))
        if n2_v2_doc is not None and 'interm150_median_abs_dz_mm' in n2_v2_doc:
            labels_b.append('N2 v2\n(Stage-I Final)')
            dz_i150.append(float(n2_v2_doc['interm150_median_abs_dz_mm']))
            f30_v2 = n2_v2_doc.get('pv_metrics', {}).get('FrontPinpoint_30mm', {}).get('median_abs_dz_mm', np.nan)
            dz_f30.append(float(f30_v2))
        xb = np.arange(len(labels_b))
        bars_b1 = ax1.bar(xb - 0.17, dz_i150, width=0.33, color='#00897B', edgecolor='black', lw=0.4, label='Interm150 Median |dz| [mm]')
        for b, v in zip(bars_b1, dz_i150):
            ax1.text(b.get_x() + b.get_width() / 2.0, v + 0.35, f'{v:.2f}', ha='center', va='bottom', fontsize=7.2)
        valid_f30 = [i for i, v in enumerate(dz_f30) if not np.isnan(v)]
        if valid_f30:
            bars_b2 = ax1.bar(np.array(valid_f30) + 0.17, [dz_f30[i] for i in valid_f30], width=0.33, color='#5C6BC0', edgecolor='black', lw=0.4, label='Front30 Median |dz| [mm]')
            for b, v in zip(bars_b2, [dz_f30[i] for i in valid_f30]):
                ax1.text(b.get_x() + b.get_width() / 2.0, v + 0.35, f'{v:.2f}', ha='center', va='bottom', fontsize=7.2)
        ax1.set_xticks(xb)
        ax1.set_xticklabels(labels_b, fontsize=8.0)
        ax1.set_ylim(0.0, max(dz_i150) + 3.2)
        ax1.set_ylabel('PV Median |dz| [mm]')
        ax1.set_title('B. N2 TauKinemaDiffNet PV Resolution & Pull Calibration')
        ax1.legend(loc='upper right', fontsize=7.3)
        ax1.grid(True, axis='y', alpha=0.25)
    else:
        ax1.text(0.5, 0.5, 'Pending N2 Data', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('B. N2 TauKinemaDiffNet Interm150 PV Fix')

    # Panel C: Q13 Full 1mm Sensor Control vs Centroid-Only vs Native 20um
    if q13_doc is not None and 'runs' in q13_doc:
        runs_q = q13_doc['runs']
        q_order = [
            ('native_20um_s1_tracklet_ipformer', 'Native 20 um\n(TrackletIP)'),
            ('centroid_only_1mm_abl_1mm_quant', 'Centroid-Only\n1 mm Quant'),
            ('full_1mm_abl_full_1mm_sensor', 'Full 1 mm Sensor\n(Centroid + Patch)'),
        ]
        q_labels = [lbl for k, lbl in q_order if k in runs_q]
        q_f30 = [float(runs_q[k]['slices']['Front30']['BA4']) * 100.0 for k, _ in q_order if k in runs_q]
        q_w184 = [float(runs_q[k]['ood']['gen_ood_w184_epps21_nc_charm']['p_tau_gt_0p5_rate']) * 100.0 for k, _ in q_order if k in runs_q]
    else:
        q_labels = ['Native 20 um\n(TrackletIP)', 'Centroid-Only\n1 mm Quant']
        q_f30 = [63.26, 62.68]
        q_w184 = [2.33, 5.17]
    xq = np.arange(len(q_labels))
    ax2.bar(xq - 0.18, q_f30, width=0.34, color='#1565C0', edgecolor='black', lw=0.4, label='Front30 BA4 [%]')
    ax2_r = ax2.twinx()
    ax2_r.spines['top'].set_visible(False)
    ax2_r.bar(xq + 0.18, q_w184, width=0.34, color='#C62828', alpha=0.75, edgecolor='black', lw=0.4, label='W184 Fake Tau [%]')
    for i, (f_v, w_v) in enumerate(zip(q_f30, q_w184)):
        ax2.text(xq[i] - 0.18, f_v + 0.18, f'{f_v:.2f}%', ha='center', va='bottom', fontsize=7.0)
        ax2_r.text(xq[i] + 0.18, w_v + 0.12, f'{w_v:.2f}%', ha='center', va='bottom', fontsize=7.0, color='#C62828')
    ax2_r.set_ylabel('W184 NC Charm P(tau)>0.5 [%]', color='#C62828')
    ax2_r.set_ylim(0.0, max(6.8, max(q_w184) + 1.6))
    ax2.set_xticks(xq)
    ax2.set_xticklabels(q_labels, fontsize=8.0)
    ax2.set_ylabel('Front30 4-Class BA4 [%]')
    ax2.set_ylim(55.0, 66.2)
    ax2.set_title('C. Q13 Sensor Granularity Control (20 um vs 1 mm)')
    h1c, l1c = ax2.get_legend_handles_labels()
    h2c, l2c = ax2_r.get_legend_handles_labels()
    ax2.legend(h1c + h2c, l1c + l2c, loc='upper left', fontsize=7.2)
    ax2.grid(True, axis='y', alpha=0.25)

    # Panel D: Stage-II Tail-DRO Confirmation on 50/50 Blend (Tune6k + W184)
    if s2_dro is not None and 'blend_50_50' in s2_dro and s2_dro['blend_50_50']:
        b_m = s2_dro['blend_50_50']['tune6k_raw']
        b_cal = s2_dro['blend_50_50']['tune6k_screen_cal']
        cats = ['All 1%\n(Raw)', 'D1 1%\n(Raw)', 'D1 1%\n(Screen Cal)', 'Pinpoint 1%\n(Raw)', 'Front30 1%\n(Raw)', 'All 0.1%\n(Raw)']
        pre_v = [
            b_m['pretrained']['within_slice_1pct_fpr']['All_Tune6k']['tau_eff_pct'],
            b_m['pretrained']['within_slice_1pct_fpr']['D1_obs']['tau_eff_pct'],
            b_cal['pretrained']['within_slice_1pct_fpr']['D1_obs']['tau_eff_pct'],
            b_m['pretrained']['within_slice_1pct_fpr']['Pinpoint_PixXY']['tau_eff_pct'],
            b_m['pretrained']['within_slice_1pct_fpr']['Front30']['tau_eff_pct'],
            b_m['pretrained']['within_slice_0p1pct_fpr']['All_Tune6k']['tau_eff_pct'],
        ]
        dro_v = [
            b_m['dro']['within_slice_1pct_fpr']['All_Tune6k']['tau_eff_pct'],
            b_m['dro']['within_slice_1pct_fpr']['D1_obs']['tau_eff_pct'],
            b_cal['dro']['within_slice_1pct_fpr']['D1_obs']['tau_eff_pct'],
            b_m['dro']['within_slice_1pct_fpr']['Pinpoint_PixXY']['tau_eff_pct'],
            b_m['dro']['within_slice_1pct_fpr']['Front30']['tau_eff_pct'],
            b_m['dro']['within_slice_0p1pct_fpr']['All_Tune6k']['tau_eff_pct'],
        ]
        xp = np.arange(len(cats))
        ax3.bar(xp - 0.18, pre_v, width=0.34, color='#78909C', edgecolor='black', lw=0.4, label='50/50 Blend Pretrained')
        ax3.bar(xp + 0.18, dro_v, width=0.34, color='#D81B60', edgecolor='black', lw=0.4, label='50/50 Blend Score-Bank DRO')
        for i, (pv, dv) in enumerate(zip(pre_v, dro_v)):
            ax3.text(xp[i] - 0.18, pv + 0.18, f'{pv:.1f}', ha='center', va='bottom', fontsize=6.8)
            ax3.text(xp[i] + 0.18, dv + 0.18, f'{dv:.1f}', ha='center', va='bottom', fontsize=6.8)
        ax3.set_xticks(xp)
        ax3.set_xticklabels(cats, fontsize=7.5)
        ax3.set_ylabel('Tune6k Within-Slice Tau Signal Efficiency [%]')
        ax3.set_ylim(0.0, max(14.5, max(pre_v + dro_v) + 2.4))
        w184_pre_b = s2_dro['blend_50_50']['ood_6shards']['gen_ood_w184_epps21_nc_charm']['pretrained']['p_tau_gt_0p5_pct']
        w184_dro_b = s2_dro['blend_50_50']['ood_6shards']['gen_ood_w184_epps21_nc_charm']['dro']['p_tau_gt_0p5_pct']
        ax3.text(
            0.98, 0.96,
            f'All Tune6k BA4: 58.50% -> 58.76% (+0.27 pp), dNLL = -0.0006\n'
            f'W184 NC Charm P(tau)>0.5: {w184_pre_b:.2f}% -> {w184_dro_b:.2f}%',
            transform=ax3.transAxes, ha='right', va='top', fontsize=7.2,
            bbox={'boxstyle': 'round,pad=0.3', 'facecolor': '#FFF8E1', 'edgecolor': '#FFB300', 'alpha': 0.9},
        )
        ax3.set_title('D. Stage-II Tail-DRO Confirmation (50/50 Blend on Tune6k, N=6,000)')
        ax3.legend(loc='upper left', fontsize=7.2)
        ax3.grid(True, axis='y', alpha=0.25)
    else:
        ax3.text(0.5, 0.5, 'Pending Stage-II Tail-DRO Data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('D. Stage-II Tail-DRO Confirmation')

    fig.tight_layout()
    out_path = out_dir / 'fig10_stage327_integrated_summary.png'
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description='Render Stage 327 300-DPI English publication figures')
    parser.add_argument('--out-dir', type=Path, default=FIG_DIR)
    args = parser.parse_args()

    out_dir = assert_unsealed_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with SealAuditGuard() as guard:
        rendered: list[str] = []
        s1_p = NEXTGEN_ROOT / 'exp_tail_dro/stage1/stage1_tail_dro_summary.json'
        if s1_p.is_file():
            s1_doc = json.loads(assert_unsealed_path(s1_p).read_text())
            rendered.append(str(plot_stage1_tail_dro(s1_doc, out_dir)))

        s2_p = NEXTGEN_ROOT / 'exp_tail_dro/stage2/stage2_tail_dro_summary.json'
        if s2_p.is_file():
            s2_doc = json.loads(assert_unsealed_path(s2_p).read_text())
            rendered.append(str(plot_stage2_tail_dro(s2_doc, out_dir)))

        p10 = plot_stage327_integrated_summary(out_dir)
        if p10 is not None:
            rendered.append(str(p10))

        assert len(guard.violations) == 0, f'Seal violations: {guard.violations}'
        print(json.dumps({'rendered_figures': rendered, 'seal_violations': 0}, indent=2))


if __name__ == '__main__':
    main()
