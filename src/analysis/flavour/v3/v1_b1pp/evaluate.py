"""M1 Collaboration benchmark, modality ablations, OOF regional calibration, and 3 figures (W3-05)."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import softmax

from analysis.flavour.inventory import write_json
from analysis.flavour.s3.metrics import bootstrap_weights
from analysis.flavour.v2.partitions import KEY
from analysis.flavour.v2.r1.metrics import Metrics, three_from_four, three_targets
from analysis.flavour.v2.r2.readout import grouped_folds
from analysis.flavour.v3.common.calibration import (
    _tau_efficiency_at_fpr,
    multi_domain_asimov_z0,
    oof_regional_calibration,
)


SLICE_ORDER = (
    'All',
    'Obs_D_pix_contained',
    'Obs_D_scint_contained',
    'Obs_D_partial_or_edge',
    'Truth_pixel_centered',
    'Truth_large_scintillator',
    'Tune_E_le_2p3TeV',
    'HighE_gt_2p3TeV_non_tau',
)


def _balanced_weights(y: np.ndarray, k: int) -> np.ndarray:
    cnt = np.bincount(y, minlength=k)
    return (len(y) / (float(k) * np.maximum(cnt, 1)))[y]


def _extract_logits(df: pd.DataFrame, k: int) -> np.ndarray:
    if all(f'z{c}' in df.columns for c in range(k)):
        z = df[[f'z{c}' for c in range(k)]].to_numpy(dtype=np.float64)
        return z - z.mean(axis=1, keepdims=True)
    p = df[[f'p{c}' for c in range(k)]].to_numpy(dtype=np.float64)
    return np.log(np.clip(p, 1e-15, 1.0))


def _bootstrap_ci(vals: np.ndarray) -> dict:
    finite = vals[np.isfinite(vals)]
    if len(finite) == 0:
        return {'median': None, 'ci_low': None, 'ci_high': None}
    return {
        'median': float(np.median(finite)),
        'ci_low': float(np.quantile(finite, 0.025)),
        'ci_high': float(np.quantile(finite, 0.975)),
    }


def _compute_slice_metrics_t3(
    probs3: np.ndarray,
    logits3: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    boot_w: np.ndarray | None = None,
) -> dict:
    keep = mask & (labels != 2)
    if not keep.any():
        return {'events': 0}
    y3 = three_targets(labels[keep])
    p3 = probs3[keep]
    z3 = logits3[keep]
    cnt = np.bincount(y3, minlength=3)
    present = cnt > 0
    w_bal = np.zeros(len(y3), dtype=np.float64)
    n_p = int(present.sum())
    for c in range(3):
        if cnt[c] > 0:
            w_bal[y3 == c] = len(y3) / (float(n_p) * cnt[c])

    rep = Metrics(y3, p3).report(w_bal)
    c_names = ('nu_e', 'nu_mu', 'NC')
    recalls = {cname: rep['per_class'][c]['recall'] for c, cname in enumerate(c_names)}
    aurocs = {cname: rep['per_class'][c].get('AUROC') for c, cname in enumerate(c_names)}
    eff_purity_80 = {cname: rep['per_class'][c].get('eff_purity_0.8') for c, cname in enumerate(c_names)}
    eff_purity_90 = {cname: rep['per_class'][c].get('eff_purity_0.9') for c, cname in enumerate(c_names)}
    out = {
        'events': int(keep.sum()),
        'raw_counts': cnt.tolist(),
        'BA3': float(rep['BA']),
        'NLL3': float(rep['NLL']),
        'macro_AP3': rep['macro_AP'],
        'macro_AUROC3': rep['macro_AUROC'],
        'recalls': recalls,
        'aurocs': aurocs,
        'eff_purity_80': eff_purity_80,
        'eff_purity_90': eff_purity_90,
    }
    if boot_w is not None:
        pred3 = p3.argmax(axis=1)
        bw = boot_w[:, keep].astype(np.float64)
        correct_c = [(y3 == c) & (pred3 == c) for c in range(3)]
        is_c = [y3 == c for c in range(3)]
        ba_samples = np.zeros(bw.shape[0], dtype=np.float64)
        valid_classes = 0
        for c in range(3):
            if not is_c[c].any():
                continue
            valid_classes += 1
            num = bw[:, correct_c[c]].sum(axis=1)
            den = np.maximum(bw[:, is_c[c]].sum(axis=1), 1e-12)
            ba_samples += num / den
        ba_samples /= max(1, valid_classes)
        out['BA3_ci95'] = _bootstrap_ci(ba_samples)
    return out


def _evaluate_t4_model(
    df: pd.DataFrame,
    slice_masks: dict[str, np.ndarray],
    boot_w: np.ndarray,
    folds: np.ndarray,
) -> dict:
    labels = df['label'].to_numpy(dtype=np.int64)
    obs_dom = df['obs_domain_id'].to_numpy(dtype=np.int64)
    logits = _extract_logits(df, 4)
    raw_probs = df[[f'p{c}' for c in range(4)]].to_numpy(dtype=np.float64)

    cal_probs, s_tau, fit_records = oof_regional_calibration(logits, labels, obs_dom, folds)
    z0_s_tau = multi_domain_asimov_z0(s_tau, labels, obs_dom)
    w_all = _balanced_weights(labels, 4)

    # Global thresholds (raw p_tau and regional S_tau) computed on AllTune
    raw_p_tau = raw_probs[:, 2]
    cuts = {}
    for fpr in (0.01, 0.001):
        r_wp = _tau_efficiency_at_fpr(raw_p_tau, labels, fpr, w_all)
        s_wp = _tau_efficiency_at_fpr(s_tau, labels, fpr, w_all)
        # OOF Strategy II Neyman-Pearson domain allocation
        sel_np = np.zeros(len(labels), dtype=bool)
        for fold in range(5):
            held = folds == fold
            fit = ~held
            fit_tot_bkg = float(w_all[fit & (labels != 2)].sum())
            best_eff = -1.0
            best_cuts = {}
            for r1, r2, r3 in ((1.0, 1.0, 1.0), (2.0, 1.0, 0.5), (3.0, 1.0, 0.3), (4.0, 0.8, 0.2)):
                r_vec = {1: r1, 2: r2, 3: r3}
                denom = sum(r_vec[d] * float(w_all[fit & (obs_dom == d) & (labels != 2)].sum()) for d in (1, 2, 3))
                scale = (fpr * fit_tot_bkg) / max(denom, 1e-30)
                c_map = {}
                tau_sel = 0.0
                for d in (1, 2, 3):
                    dm = fit & (obs_dom == d)
                    if not dm.any() or not (dm & (labels != 2)).any():
                        c_map[d] = np.inf
                        continue
                    f_d = min(0.5, scale * r_vec[d])
                    info_d = _tau_efficiency_at_fpr(s_tau[dm], labels[dm], f_d, w_all[dm])
                    c_map[d] = info_d['threshold'] if info_d['threshold'] is not None else np.inf
                    tau_sel += float(w_all[dm & (labels == 2) & (s_tau >= c_map[d])].sum())
                if tau_sel > best_eff:
                    best_eff = tau_sel
                    best_cuts = c_map
            for d in (1, 2, 3):
                hm = held & (obs_dom == d)
                sel_np[hm] = s_tau[hm] >= best_cuts.get(d, np.inf)

        cuts[str(fpr)] = {
            'raw_threshold': r_wp['threshold'],
            's_tau_threshold': s_wp['threshold'],
            'np_selected_mask': sel_np,
        }

    # Conditional 3-class probabilities
    raw_p3 = three_from_four(raw_probs)
    cal_p3 = three_from_four(cal_probs)

    slices_out = {}
    for sname, smask in slice_masks.items():
        if not smask.any():
            continue
        ys = labels[smask]
        raw_ps = raw_probs[smask]
        cal_ps = cal_probs[smask]
        cnt4 = np.bincount(ys, minlength=4)
        present4 = cnt4 > 0
        row = {'events': int(smask.sum()), 'class_counts': cnt4.tolist()}

        # Conditional 3-class T3-equivalent metrics
        t3_raw = _compute_slice_metrics_t3(raw_p3, logits[:, [0, 1, 3]], labels, smask, boot_w)
        row.update({
            'BA3': t3_raw.get('BA3'),
            'BA3_ci95': t3_raw.get('BA3_ci95'),
            'NLL3': t3_raw.get('NLL3'),
            'macro_AP3': t3_raw.get('macro_AP3'),
            'macro_AUROC3': t3_raw.get('macro_AUROC3'),
            'recalls3': t3_raw.get('recalls'),
            'aurocs3': t3_raw.get('aurocs'),
            'eff_purity_80_3': t3_raw.get('eff_purity_80'),
            'eff_purity_90_3': t3_raw.get('eff_purity_90'),
        })

        if present4.all():
            w4 = np.array([len(ys) / (4.0 * cnt4[c]) for c in ys], dtype=np.float64)
            m4_raw = Metrics(ys, raw_ps).report(w4)
            m4_cal = Metrics(ys, cal_ps).report(w4)
            row.update({
                'BA4_raw': float(m4_raw['BA']),
                'NLL4_raw': float(m4_raw['NLL']),
                'BA4_cal': float(m4_cal['BA']),
                'NLL4_cal': float(m4_cal['NLL']),
                'macro_AP4_raw': m4_raw['macro_AP'],
                'macro_AUROC4_raw': m4_raw['macro_AUROC'],
            })
            # 400x block bootstrap on BA4
            pred4 = raw_ps.argmax(axis=1)
            bw = boot_w[:, smask].astype(np.float64)
            ba4_boot = np.zeros(bw.shape[0], dtype=np.float64)
            for c in range(4):
                num = bw[:, (ys == c) & (pred4 == c)].sum(axis=1)
                den = np.maximum(bw[:, ys == c].sum(axis=1), 1e-12)
                ba4_boot += num / den
            ba4_boot /= 4.0
            row['BA4_ci95'] = _bootstrap_ci(ba4_boot)

        # Tau efficiencies / high-E false-tau rates at 1% & 0.1% global FPR cuts
        wp_dict = {}
        for fpr_str, cinfo in cuts.items():
            raw_pass = smask & (raw_p_tau >= cinfo['raw_threshold'])
            s_pass = smask & (s_tau >= cinfo['s_tau_threshold'])
            np_pass = smask & cinfo['np_selected_mask']
            is_tau_s = smask & (labels == 2)
            is_bkg_s = smask & (labels != 2)

            def _wp_stat(sel: np.ndarray, prefix: str = '') -> dict:
                tw = float(w_all[is_tau_s].sum())
                bw_s = float(w_all[is_bkg_s].sum())
                st = {
                    'tau_eff': float(w_all[sel & is_tau_s].sum() / tw) if tw > 0 else None,
                    'bkg_fpr': float(w_all[sel & is_bkg_s].sum() / bw_s) if bw_s > 0 else None,
                    'tau_selected': int((sel & is_tau_s).sum()),
                    'bkg_selected': int((sel & is_bkg_s).sum()),
                }
                if tw > 0:
                    bw_arr = boot_w[:, is_tau_s].astype(np.float64)
                    hit_arr = (sel[is_tau_s])[None, :].astype(np.float64)
                    # Bootstrap tau_eff
                    st['tau_eff_ci95'] = _bootstrap_ci(
                        (bw_arr * hit_arr).sum(axis=1) / np.maximum(bw_arr.sum(axis=1), 1e-12)
                    )
                elif bw_s > 0:
                    # HighE_gt_2p3TeV_non_tau false-tau CI
                    bw_arr = boot_w[:, is_bkg_s].astype(np.float64)
                    hit_arr = (sel[is_bkg_s])[None, :].astype(np.float64)
                    st['bkg_fpr_ci95'] = _bootstrap_ci(
                        (bw_arr * hit_arr).sum(axis=1) / np.maximum(bw_arr.sum(axis=1), 1e-12)
                    )
                return st

            wp_dict[fpr_str] = {
                'raw_global': _wp_stat(raw_pass),
                'regional_S_tau': _wp_stat(s_pass),
                'regional_NP': _wp_stat(np_pass),
            }
        row['working_points'] = wp_dict
        slices_out[sname] = row

    return {
        'asimov_z0_s_tau': float(z0_s_tau),
        'oof_fit_fold0': fit_records[0]['domains'],
        'slices': slices_out,
    }


def _load_r1_r2_references() -> dict:
    r1_csv = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R1/evaluation/model_domain_metrics.csv')
    r1_json_path = Path('/eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/roadmap/v2/r1_results.json')
    r2_csv = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R2/evaluation/model_domain_metrics.csv')
    w303_json = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-03/replay_results.json')

    refs: dict = {
        'FORTVNE_Wu_2026': {
            'paper': 'Wu et al. (2026), FORTVNE GNN contained large scintillator (e/mu/NC)',
            'large_scintillator_BA3': 0.750,
            'note': 'Published collaboration M1 GNN reference (~75% 3-class balanced accuracy on contained large scintillator)',
        },
        'FORTVNE_Mahlstedt_2026': {
            'paper': 'Mahlstedt (2026), FORTVNE 3D CNN baseline on FASERv/Pinpoint',
            'all_BA3': 0.566,
            'large_scintillator_BA3': 0.690,
            'note': 'Matched CNN baseline class represented internally by R1 B1',
        },
    }
    if r1_csv.exists():
        df1 = pd.read_csv(r1_csv)
        for name in ('final_b1_t3_s42', 'final_b1_t4_s42'):
            sub = df1[df1['name'] == name]
            refs[name] = {
                row['domain']: {
                    'BA3': None if pd.isna(row['BA3']) else float(row['BA3']),
                    'NLL3': None if pd.isna(row['NLL3']) else float(row['NLL3']),
                    'BA4': None if pd.isna(row['BA4']) else float(row['BA4']),
                    'NLL4': None if pd.isna(row['NLL4']) else float(row['NLL4']),
                    'tau_eff_1pct': None if pd.isna(row['tau_eff_1pct']) else float(row['tau_eff_1pct']),
                }
                for _, row in sub.iterrows()
            }
    if r1_json_path.exists():
        r1_full = json.loads(r1_json_path.read_text()).get('domains', {})
        r1_specs = (
            ('final_b1_t3_s42', 'R1 B1 T3 (4p, 100k)'),
            ('final_b1_t4_s42', 'R1 B1 T4 cond. (4p, 100k)'),
            ('final_b1_t3_s42_without_tracker_fp32', 'R1 B1 T3 No Tracker'),
            ('control_b1_cnn_t3_s42', 'R1 Plain CNN T3 (4p, 100k)'),
            ('screen_b0_t3_lr0.05_s42', 'R1 HGB T3 (100k)'),
        )
        r1_detailed = {}
        for r1_key, label_name in r1_specs:
            if r1_key not in r1_full:
                continue
            dom_map = {}
            for dname in ('all', 'pixel_centered', 'large_scintillator'):
                drec = r1_full[r1_key].get(dname)
                if not drec or 'T3' not in drec:
                    continue
                t3 = drec['T3']
                pc = t3['per_class']
                entry = {
                    'BA3': float(t3['BA']),
                    'NLL3': float(t3['NLL']),
                    'macro_AUROC3': t3.get('macro_AUROC'),
                    'aurocs': {'nu_e': pc[0].get('AUROC'), 'nu_mu': pc[1].get('AUROC'), 'NC': pc[2].get('AUROC')},
                    'eff_purity_80': {'nu_e': pc[0].get('eff_purity_0.8'), 'nu_mu': pc[1].get('eff_purity_0.8'), 'NC': pc[2].get('eff_purity_0.8')},
                    'eff_purity_90': {'nu_e': pc[0].get('eff_purity_0.9'), 'nu_mu': pc[1].get('eff_purity_0.9'), 'NC': pc[2].get('eff_purity_0.9')},
                    'BA4': float(drec['T4']['BA']) if 'T4' in drec else None,
                    'tau_eff_1pct': drec['T4']['per_class'][2].get('eff_FPR_0.01') if 'T4' in drec else None,
                }
                dom_map[dname] = entry
            r1_detailed[r1_key] = {'label': label_name, 'domains': dom_map}
        refs['r1_detailed'] = r1_detailed
    if r2_csv.exists():
        df2 = pd.read_csv(r2_csv)
        for exp in ('b1_d2_t3_s42', 'b1_d2_t4_s42'):
            sub = df2[(df2['experiment'] == exp) & (df2['kind'] == 'endpoint')]
            refs[f'R2_{exp}'] = {
                row['domain']: {
                    'BA3': None if pd.isna(row['BA3']) else float(row['BA3']),
                    'NLL3': None if pd.isna(row['NLL3']) else float(row['NLL3']),
                    'BA4': None if pd.isna(row['BA4']) else float(row['BA4']),
                    'NLL4': None if pd.isna(row['NLL4']) else float(row['NLL4']),
                    'tau_eff_0.01': None if pd.isna(row['tau_eff_0.01']) else float(row['tau_eff_0.01']),
                    'tau_eff_0.001': None if pd.isna(row['tau_eff_0.001']) else float(row['tau_eff_0.001']),
                }
                for _, row in sub.iterrows()
            }
    if w303_json.exists():
        refs['W3_03_R2_D2_replay'] = json.loads(w303_json.read_text()).get('r2_b1_d2_t4_tune40k')
    return refs


def _make_figures(res: dict, refs: dict, manifests: dict, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'legend.fontsize': 8.5,
        'figure.dpi': 150,
    })

    models = res['models']

    # --- Figure 1: Collaboration M1 Benchmark + Modality Ablations ---
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
    ax0, ax1 = axes

    bar_specs = [
        ('R1 B1 T4\n(4p, 100k)', refs['final_b1_t4_s42']['pixel_centered']['BA3'], refs['final_b1_t4_s42']['large_scintillator']['BA3'], refs['final_b1_t4_s42']['all']['BA3'], '#8c96a5'),
        ('R2 B1-D2 T4\n(4p, 300k)', refs['R2_b1_d2_t4_s42']['pixel_centered']['BA3'], refs['R2_b1_d2_t4_s42']['large_scintillator']['BA3'], refs['R2_b1_d2_t4_s42']['all']['BA3'], '#5a7899'),
        ('B1++ T3\n(16p, 100k)', models['b1pp_100k_16p_t3']['slices']['Truth_pixel_centered']['BA3'], models['b1pp_100k_16p_t3']['slices']['Truth_large_scintillator']['BA3'], models['b1pp_100k_16p_t3']['slices']['All']['BA3'], '#2c8c69'),
        ('B1++ T4\n(16p, 100k)', models['b1pp_100k_16p_t4']['slices']['Truth_pixel_centered']['BA3'], models['b1pp_100k_16p_t4']['slices']['Truth_large_scintillator']['BA3'], models['b1pp_100k_16p_t4']['slices']['All']['BA3'], '#e08b36'),
        ('B1++ T4\n(24p, 300k)', models['b1pp_300k_24p_t4']['slices']['Truth_pixel_centered']['BA3'], models['b1pp_300k_24p_t4']['slices']['Truth_large_scintillator']['BA3'], models['b1pp_300k_24p_t4']['slices']['All']['BA3'], '#c94338'),
    ]
    x = np.arange(len(bar_specs))
    w = 0.25
    pix_vals = [b[1] * 100 for b in bar_specs]
    lrg_vals = [b[2] * 100 for b in bar_specs]
    all_vals = [b[3] * 100 for b in bar_specs]
    b1 = ax0.bar(x - w, all_vals, width=w, label='All Tune40k', color='#4c6a92')
    b2 = ax0.bar(x, lrg_vals, width=w, label='Truth large_scintillator', color='#2c8c69')
    b3 = ax0.bar(x + w, pix_vals, width=w, label='Truth pixel_centered', color='#d96b27')
    ax0.axhline(73.0, color='#1b5e20', linestyle='--', linewidth=1.3, label='V1 Gate (Large >= 73.0%)')
    ax0.axhline(75.0, color='#333333', linestyle=':', linewidth=1.3, label='FORTVNE Wu 2026 (~75%)')
    for bars in (b1, b2, b3):
        for rect in bars:
            h = rect.get_height()
            ax0.text(rect.get_x() + rect.get_width() / 2, h + 0.6, f'{h:.1f}', ha='center', va='bottom', fontsize=7.5)
    ax0.set_xticks(x)
    ax0.set_xticklabels([b[0] for b in bar_specs], fontsize=8.5)
    ax0.set_ylabel('Conditional 3-Class Balanced Accuracy BA3 (%)')
    ax0.set_ylim(48, 92)
    ax0.set_title('Collaboration M1 Benchmark on Tune40k (seed=42)')
    ax0.grid(axis='y', alpha=0.25)
    ax0.legend(loc='upper left', ncol=2, frameon=True)

    # Panel B: Modality ablations on T3 (All, Pixel-centered, Large-scintillator + nu_mu recall)
    abl_names = ['B1++ T3 Full', 'B1++ T3 No Pixel', 'B1++ T3 No Tracker']
    abl_keys = ['b1pp_100k_16p_t3', 'b1pp_100k_16p_t3_no_pixel', 'b1pp_100k_16p_t3_no_tracker']
    xa = np.arange(len(abl_keys))
    wa = 0.22
    v_all = [models[k]['slices']['All']['BA3'] * 100 for k in abl_keys]
    v_pix = [models[k]['slices']['Truth_pixel_centered']['BA3'] * 100 for k in abl_keys]
    v_lrg = [models[k]['slices']['Truth_large_scintillator']['BA3'] * 100 for k in abl_keys]
    v_mu = [models[k]['slices']['Truth_large_scintillator']['recalls']['nu_mu'] * 100 for k in abl_keys]
    ab1 = ax1.bar(xa - 1.5 * wa, v_all, width=wa, label='BA3 All', color='#4c6a92')
    ab2 = ax1.bar(xa - 0.5 * wa, v_lrg, width=wa, label='BA3 Large Scint.', color='#2c8c69')
    ab3 = ax1.bar(xa + 0.5 * wa, v_pix, width=wa, label='BA3 Pixel Cent.', color='#d96b27')
    ab4 = ax1.bar(xa + 1.5 * wa, v_mu, width=wa, label='nu_mu Recall (Large)', color='#7b52ab')
    for bars in (ab1, ab2, ab3, ab4):
        for rect in bars:
            h = rect.get_height()
            ax1.text(rect.get_x() + rect.get_width() / 2, h + 0.6, f'{h:.1f}', ha='center', va='bottom', fontsize=7.5)
    ax1.set_xticks(xa)
    ax1.set_xticklabels(abl_names, fontsize=9)
    ax1.set_ylabel('Balanced Accuracy / Recall (%)')
    ax1.set_ylim(44, 92)
    ax1.set_title('T3 Modality Ablation Contributions (Train100k x 16p)')
    ax1.grid(axis='y', alpha=0.25)
    ax1.legend(loc='upper left', ncol=2, frameon=True)

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(fig_dir / f'w3_05_m1_benchmark_and_ablations.{ext}', bbox_inches='tight')
    plt.close(fig)

    # --- Figure 2: Nu_tau Slice Efficiency & High-Energy Spectrum Bias ---
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
    ax0, ax1 = axes
    s_names = ['All', 'Obs_D_pix_contained', 'Truth_pixel_centered', 'Truth_large_scintillator']
    s_short = ['All (40k)', 'Obs D_pix', 'Truth Pix-Cent', 'Truth Large']
    xs = np.arange(len(s_names))
    ws = 0.20

    r2_cal = [refs['W3_03_R2_D2_replay']['working_points']['0.01']['slices'][s]['regional_S_tau_global_cut']['tau_eff'] * 100 for s in s_names]
    b100_cal = [models['b1pp_100k_16p_t4']['slices'][s]['working_points']['0.01']['regional_S_tau']['tau_eff'] * 100 for s in s_names]
    b300_raw = [models['b1pp_300k_24p_t4']['slices'][s]['working_points']['0.01']['raw_global']['tau_eff'] * 100 for s in s_names]
    b300_cal = [models['b1pp_300k_24p_t4']['slices'][s]['working_points']['0.01']['regional_S_tau']['tau_eff'] * 100 for s in s_names]

    bars_t = [
        ax0.bar(xs - 1.5 * ws, r2_cal, width=ws, label='R2 B1-D2 Reg. S_tau', color='#7c8d9e'),
        ax0.bar(xs - 0.5 * ws, b100_cal, width=ws, label='B1++ 100k Reg. S_tau', color='#e08b36'),
        ax0.bar(xs + 0.5 * ws, b300_raw, width=ws, label='B1++ 300k Raw p_tau', color='#5082b5'),
        ax0.bar(xs + 1.5 * ws, b300_cal, width=ws, label='B1++ 300k Reg. S_tau', color='#c94338'),
    ]
    for bars in bars_t:
        for rect in bars:
            h = rect.get_height()
            ax0.text(rect.get_x() + rect.get_width() / 2, h + 0.35, f'{h:.1f}', ha='center', va='bottom', fontsize=7.5)
    ax0.set_xticks(xs)
    ax0.set_xticklabels(s_short, fontsize=9)
    ax0.set_ylabel('nu_tau Selection Efficiency at 1% Global FPR (%)')
    ax0.set_ylim(0, max(max(b300_cal), max(b300_raw), 15) * 1.32)
    ax0.set_title('nu_tau Selection Efficiency by Containment Domain (1% FPR)')
    ax0.grid(axis='y', alpha=0.25)
    ax0.legend(loc='upper left', frameon=True)

    # Panel B: Energy spectrum bias check (E <= 2.3 TeV BA4/tau_eff vs HighE > 2.3 TeV false-tau FPR)
    spec_keys = ['b1pp_100k_16p_t4', 'b1pp_100k_16p_t4_ematched', 'b1pp_300k_24p_t4']
    spec_lbls = ['100k 16p\nUnweighted', '100k 16p\nE-Matched', '300k 24p\nFull Pool']
    xe = np.arange(len(spec_keys))
    we = 0.25
    cs_ba4 = [models[k]['slices']['Tune_E_le_2p3TeV']['BA4_raw'] * 100 for k in spec_keys]
    cs_tau1 = [models[k]['slices']['Tune_E_le_2p3TeV']['working_points']['0.01']['regional_S_tau']['tau_eff'] * 100 for k in spec_keys]
    he_fpr1 = [models[k]['slices']['HighE_gt_2p3TeV_non_tau']['working_points']['0.01']['regional_S_tau']['bkg_fpr'] * 100 for k in spec_keys]
    he_fpr_raw = [models[k]['slices']['HighE_gt_2p3TeV_non_tau']['working_points']['0.01']['raw_global']['bkg_fpr'] * 100 for k in spec_keys]

    ax1_r = ax1.twinx()
    eb1 = ax1.bar(xe - we / 2, cs_ba4, width=we, label='BA4 (E <= 2.3 TeV)', color='#4c6a92')
    eb2 = ax1.bar(xe + we / 2, cs_tau1, width=we, label='nu_tau Eff @1% FPR (E <= 2.3 TeV)', color='#2c8c69')
    ax1_r.plot(xe, he_fpr1, 'o-', color='#c94338', linewidth=2.0, label='High-E (>2.3 TeV) False-tau FPR (%) [Reg S_tau]')
    ax1_r.plot(xe, he_fpr_raw, 's--', color='#d96b27', linewidth=1.6, label='High-E (>2.3 TeV) False-tau FPR (%) [Raw p_tau]')
    for rect in list(eb1) + list(eb2):
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width() / 2, h + 0.6, f'{h:.1f}', ha='center', va='bottom', fontsize=7.5)
    for xi, yi in zip(xe, he_fpr1):
        ax1_r.text(xi, yi - 0.16, f'{yi:.2f}%', ha='center', va='top', color='#c94338', fontsize=8.5, fontweight='bold')
    ax1.set_xticks(xe)
    ax1.set_xticklabels(spec_lbls, fontsize=9)
    ax1.set_ylabel('Common Support Metric (%)')
    ax1.set_ylim(0, 78)
    ax1_r.set_ylabel('High-E (>2.3 TeV) False-tau Rate (%)', color='#c94338')
    ax1_r.set_ylim(0, max(5.5, max(he_fpr1 + he_fpr_raw) * 1.45))
    ax1.set_title('Spectrum Support Audit (<=2.3 TeV vs >2.3 TeV Non-tau Tail)')
    ax1.grid(axis='y', alpha=0.25)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=7.5, frameon=True)

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(fig_dir / f'w3_05_tau_slices_and_spectrum.{ext}', bbox_inches='tight')
    plt.close(fig)

    # --- Figure 3: Training Dynamics, best.pt Selection Star, and Generalization Gap ---
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
    ax0, ax1 = axes
    curve_specs = [
        ('b1pp_300k_24p_t4', '300k 24p T4 (EMA)', '#c94338', '-'),
        ('b1pp_100k_16p_t4', '100k 16p T4 (EMA)', '#e08b36', '-'),
        ('b1pp_100k_16p_t4_ematched', '100k 16p T4 E-match (EMA)', '#8c564b', '--'),
        ('b1pp_100k_16p_t3', '100k 16p T3 (EMA)', '#2c8c69', '-'),
    ]
    for name, lbl, col, ls in curve_specs:
        man = manifests[name]
        hist = man['screen_history']
        ps = [h['pass'] for h in hist]
        ema_nll = [h['tune_screen_ema_NLL'] for h in hist]
        raw_nll = [h['tune_screen_raw_NLL'] for h in hist]
        gap = [h['gap_ema_NLL'] for h in hist]
        ax0.plot(ps, ema_nll, marker='o', markersize=3.5, linestyle=ls, color=col, linewidth=1.8, label=lbl)
        ax0.plot(ps, raw_nll, linestyle=':', color=col, alpha=0.55, linewidth=1.1)
        ax1.plot(ps, gap, marker='s', markersize=3.5, linestyle=ls, color=col, linewidth=1.8, label=lbl)
        bp = man.get('best_pass')
        b_row = next((h for h in hist if h['pass'] == bp), None)
        if b_row is not None:
            ax0.plot(bp, b_row['tune_screen_ema_NLL'], marker='*', markersize=11, color=col, markeredgecolor='#111111', markeredgewidth=0.6, zorder=5)
            ax1.plot(bp, b_row['gap_ema_NLL'], marker='*', markersize=11, color=col, markeredgecolor='#111111', markeredgewidth=0.6, zorder=5)

    ax0.plot([], [], marker='*', linestyle='None', color='#444444', markeredgecolor='#111111', markersize=10, label='Star = best.pt (min Tune4k NLL)')
    ax1.plot([], [], marker='*', linestyle='None', color='#444444', markeredgecolor='#111111', markersize=10, label='Star = best.pt checkpoint')
    ax0.set_xlabel('Training Pass')
    ax0.set_ylabel('Tune4k Screen NLL (Solid: EMA, Dotted: Raw)')
    ax0.set_ylim(0.78, 1.32)
    ax0.set_title('W3-05 Screen NLL Trajectories Across Passes')
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc='upper right', ncol=2, fontsize=8.0, frameon=True)

    ax1.axhline(0.0, color='#333333', linestyle='--', linewidth=1.0)
    ax1.set_xlabel('Training Pass')
    ax1.set_ylabel('Generalization Gap (Tune4k EMA NLL - Train4k NLL)')
    ax1.set_title('Tune-Train NLL Generalization Gap vs Pass')
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc='upper left', frameon=True)

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(fig_dir / f'w3_05_training_dynamics_and_gap.{ext}', bbox_inches='tight')
    plt.close(fig)


def evaluate_all(w305_root: Path, w301_root: Path) -> dict:
    dom_df = pd.read_parquet(w301_root / 'tune40k_domains.parquet')
    le23 = pd.read_parquet(w301_root / 'tune_E_le_2p3TeV_keys.parquet', columns=KEY)
    gt23 = pd.read_parquet(w301_root / 'highE_gt_2p3TeV_non_tau_keys.parquet', columns=KEY)

    dom_idx = pd.MultiIndex.from_frame(dom_df[KEY])
    le23_mask = dom_idx.isin(pd.MultiIndex.from_frame(le23[KEY]))
    gt23_mask = dom_idx.isin(pd.MultiIndex.from_frame(gt23[KEY]))

    model_names = (
        'b1pp_300k_24p_t4',
        'b1pp_100k_16p_t4',
        'b1pp_100k_16p_t4_ematched',
        'b1pp_100k_16p_t3',
        'b1pp_100k_16p_t3_no_pixel',
        'b1pp_100k_16p_t3_no_tracker',
    )

    # Reconstruct exact deterministic 4k tune_screen selection used in train.py
    ref_raw_p = w305_root / 'experiments' / model_names[0] / 'best_predictions.parquet'
    if not ref_raw_p.exists():
        ref_raw_p = w305_root / 'experiments' / model_names[0] / 'predictions.parquet'
    ref_raw_df = pd.read_parquet(ref_raw_p, columns=KEY + ['label'])
    scr_pos_t4 = np.sort(np.random.default_rng(2026).choice(np.arange(len(ref_raw_df)), size=min(4000, len(ref_raw_df)), replace=False))
    non_tau_raw = np.flatnonzero(ref_raw_df['label'].to_numpy(dtype=np.int64) != 2)
    scr_pos_t3 = np.sort(np.random.default_rng(2026).choice(non_tau_raw, size=min(4000, len(non_tau_raw)), replace=False))
    scr_keys = pd.MultiIndex.from_frame(
        pd.concat([ref_raw_df.iloc[scr_pos_t4][KEY], ref_raw_df.iloc[scr_pos_t3][KEY]]).drop_duplicates()
    )
    non_screen_mask = ~dom_idx.isin(scr_keys)

    obs_dom = dom_df['obs_domain_id'].to_numpy(dtype=np.int64)
    slice_masks = {
        'All': np.ones(len(dom_df), dtype=bool),
        'Obs_D_pix_contained': obs_dom == 1,
        'Obs_D_scint_contained': obs_dom == 2,
        'Obs_D_partial_or_edge': obs_dom == 3,
        'Truth_pixel_centered': dom_df['pixel_centered'].to_numpy(dtype=bool),
        'Truth_large_scintillator': dom_df['large_scintillator'].to_numpy(dtype=bool),
        'Tune_E_le_2p3TeV': le23_mask,
        'HighE_gt_2p3TeV_non_tau': gt23_mask,
        'Tune_non_screen_all': non_screen_mask,
        'Tune_non_screen_large_scintillator': non_screen_mask & dom_df['large_scintillator'].to_numpy(dtype=bool),
    }

    boot_w, boot_summary = bootstrap_weights(dom_df[KEY + ['label']].copy(), replicates=400, seed=2026)
    folds, fold_summary = grouped_folds(dom_df[['run', 'chunk', 'family_hi', 'family_lo', 'label']].copy(), folds=5, seed=20261117)

    manifests = {}
    models_out = {}
    pred_dfs = {}

    for mname in model_names:
        exp_dir = w305_root / 'experiments' / mname
        man = json.loads((exp_dir / 'manifest.json').read_text())
        if man.get('status') != 'complete':
            raise RuntimeError(f'experiment {mname} is not complete: {man.get("status")}')
        manifests[mname] = man

        best_p = exp_dir / 'best_predictions.parquet'
        primary_p = best_p if best_p.exists() else (exp_dir / 'predictions.parquet')
        df = pd.read_parquet(primary_p)
        pos = pd.MultiIndex.from_frame(df[KEY]).get_indexer(dom_idx)
        if (pos < 0).any():
            raise ValueError(f'{mname} predictions missing Tune40k events')
        df = df.iloc[pos].reset_index(drop=True)
        # Attach obs_domain_id and verified keys directly
        if not np.array_equal(df['label'].to_numpy(), dom_df['label'].to_numpy()):
            raise ValueError(f'{mname} prediction labels misaligned with tune40k_domains')
        pred_dfs[mname] = df

        task = man['config']['task']
        if task == 'T4':
            ev = _evaluate_t4_model(df, slice_masks, boot_w, folds)
        else:
            logits3 = _extract_logits(df, 3)
            probs3 = df[[f'p{c}' for c in range(3)]].to_numpy(dtype=np.float64)
            labels = df['label'].to_numpy(dtype=np.int64)
            slices_t3 = {}
            for sname, smask in slice_masks.items():
                slices_t3[sname] = _compute_slice_metrics_t3(probs3, logits3, labels, smask, boot_w)
            ev = {'slices': slices_t3}

        # Endpoint comparison (Best-Screen vs EMA-Final vs Exposure-Final)
        ep_comp = {}
        for ep_name, fname in (
            ('best_screen', 'best_predictions.parquet'),
            ('ema_final', 'predictions.parquet'),
            ('exposure_final', 'exposure_predictions.parquet'),
        ):
            ep_path = exp_dir / fname
            if not ep_path.exists():
                continue
            ep_df = pd.read_parquet(ep_path)
            ep_pos = pd.MultiIndex.from_frame(ep_df[KEY]).get_indexer(dom_idx)
            ep_df = ep_df.iloc[ep_pos].reset_index(drop=True)
            k = 4 if task == 'T4' else 3
            ep_l = ep_df['label'].to_numpy(dtype=np.int64)
            ep_z = _extract_logits(ep_df, k)
            ep_p = ep_df[[f'p{c}' for c in range(k)]].to_numpy(dtype=np.float64)
            p3 = three_from_four(ep_p) if task == 'T4' else ep_p
            z3 = ep_z[:, [0, 1, 3]] if task == 'T4' else ep_z
            s_all = _compute_slice_metrics_t3(p3, z3, ep_l, slice_masks['All'])
            s_pix = _compute_slice_metrics_t3(p3, z3, ep_l, slice_masks['Truth_pixel_centered'])
            s_lrg = _compute_slice_metrics_t3(p3, z3, ep_l, slice_masks['Truth_large_scintillator'])
            entry = {
                'All_BA3': s_all['BA3'],
                'All_NLL3': s_all['NLL3'],
                'Pix_BA3': s_pix['BA3'],
                'Large_BA3': s_lrg['BA3'],
            }
            if task == 'T4':
                w4 = _balanced_weights(ep_l, 4)
                m4 = Metrics(ep_l, ep_p).report(w4)
                entry.update({'All_BA4': float(m4['BA']), 'All_NLL4': float(m4['NLL']), 'All_tau_eff_1pct_raw': m4['per_class'][2]['eff_FPR_0.01']})
            ep_comp[ep_name] = entry

        ev.update({
            'task': task,
            'primary_checkpoint': 'best_screen' if best_p.exists() else 'ema_final',
            'parameters': man['parameters'],
            'wall_seconds': man['wall_seconds'],
            'peak_GPU_MiB': man['peak_GPU_MiB'],
            'GPU': man['GPU'],
            'precision': man['precision'],
            'best_pass': man['best_pass'],
            'final_screen': man['screen_history'][-1],
            'endpoint_comparison': ep_comp,
        })
        models_out[mname] = ev

    # Paired modality ablation deltas (T3 Full vs No Pixel / No Tracker)
    ablations = {}
    full_df = pred_dfs['b1pp_100k_16p_t3']
    full_pred = full_df[['p0', 'p1', 'p2']].to_numpy(dtype=np.float64).argmax(axis=1)
    labels_t3 = full_df['label'].to_numpy(dtype=np.int64)
    for abl_key, abl_name in (
        ('b1pp_100k_16p_t3_no_pixel', 'pixel_drop_delta'),
        ('b1pp_100k_16p_t3_no_tracker', 'tracker_drop_delta'),
    ):
        abl_df = pred_dfs[abl_key]
        abl_pred = abl_df[['p0', 'p1', 'p2']].to_numpy(dtype=np.float64).argmax(axis=1)
        deltas = {}
        for sname in ('All', 'Truth_pixel_centered', 'Truth_large_scintillator', 'Obs_D_pix_contained', 'Obs_D_scint_contained', 'HighE_gt_2p3TeV_non_tau'):
            keep = slice_masks[sname] & (labels_t3 != 2)
            y3 = three_targets(labels_t3[keep])
            bw = boot_w[:, keep].astype(np.float64)
            diff_ba = np.zeros(bw.shape[0], dtype=np.float64)
            nc = 0
            for c in range(3):
                if not (y3 == c).any():
                    continue
                nc += 1
                den = np.maximum(bw[:, y3 == c].sum(axis=1), 1e-12)
                r_full = bw[:, (y3 == c) & (full_pred[keep] == c)].sum(axis=1) / den
                r_abl = bw[:, (y3 == c) & (abl_pred[keep] == c)].sum(axis=1) / den
                diff_ba += (r_full - r_abl)
            diff_ba /= max(1, nc)
            point_delta = models_out['b1pp_100k_16p_t3']['slices'][sname]['BA3'] - models_out[abl_key]['slices'][sname]['BA3']
            deltas[sname] = {'delta_BA3': float(point_delta), 'delta_BA3_ci95': _bootstrap_ci(diff_ba)}
        ablations[abl_name] = deltas

    refs = _load_r1_r2_references()
    large_300k_ba3 = models_out['b1pp_300k_24p_t4']['slices']['Truth_large_scintillator']['BA3']
    large_300k_non_screen_ba3 = models_out['b1pp_300k_24p_t4']['slices']['Tune_non_screen_large_scintillator']['BA3']
    large_100k_t3_ba3 = models_out['b1pp_100k_16p_t3']['slices']['Truth_large_scintillator']['BA3']
    v1_gate_passed = bool(max(large_300k_ba3, large_100k_t3_ba3) >= 0.730 and large_300k_non_screen_ba3 >= 0.730)

    summary = {
        'status': 'complete',
        'UTC': datetime.now(timezone.utc).isoformat(),
        'primary_checkpoint': 'best_screen',
        'v1_gate_large_scintillator_ba3_ge_0.73': v1_gate_passed,
        'best_large_scintillator_BA3': float(max(large_300k_ba3, large_100k_t3_ba3)),
        'tune_non_screen_300k_large_scintillator_BA3': float(large_300k_non_screen_ba3),
        'bootstrap_summary': boot_summary,
        'fold_summary': fold_summary,
        'models': models_out,
        'ablations': ablations,
        'references': refs,
        'Cal_A_Cut_B_Confirm_payload_read': False,
    }
    write_json(w305_root / 'evaluation_results.json', summary)

    # Save Collaboration M1 aligned CSV table
    def _m1_row(model_lbl: str, sl_name: str, ba3, nll3, aurocs: dict, p80: dict, p90: dict, ba4=None, tau_raw=None, tau_reg=None) -> dict:
        return {
            'model': model_lbl,
            'slice': sl_name,
            'BA3': ba3,
            'NLL3': nll3,
            'AUROC_e': aurocs.get('nu_e'),
            'AUROC_mu': aurocs.get('nu_mu'),
            'AUROC_NC': aurocs.get('NC'),
            'eff_pur80_e': p80.get('nu_e'),
            'eff_pur80_mu': p80.get('nu_mu'),
            'eff_pur80_NC': p80.get('NC'),
            'eff_pur90_e': p90.get('nu_e'),
            'eff_pur90_mu': p90.get('nu_mu'),
            'eff_pur90_NC': p90.get('NC'),
            'BA4': ba4,
            'tau_eff_1pct_raw': tau_raw,
            'tau_eff_1pct_reg_S_tau': tau_reg,
        }

    rows = [
        _m1_row('FORTVNE Wu 2026 (ref)', 'Truth_large_scintillator', refs['FORTVNE_Wu_2026']['large_scintillator_BA3'], None, {}, {}, {}),
        _m1_row('FORTVNE Mahlstedt 2026 (ref)', 'All', refs['FORTVNE_Mahlstedt_2026']['all_BA3'], None, {}, {}, {}),
        _m1_row('FORTVNE Mahlstedt 2026 (ref)', 'Truth_large_scintillator', refs['FORTVNE_Mahlstedt_2026']['large_scintillator_BA3'], None, {}, {}, {}),
    ]
    dom_to_slice = (('all', 'All'), ('pixel_centered', 'Truth_pixel_centered'), ('large_scintillator', 'Truth_large_scintillator'))
    for r1_rec in refs.get('r1_detailed', {}).values():
        for d_key, s_lbl in dom_to_slice:
            dm = r1_rec['domains'].get(d_key)
            if dm:
                rows.append(_m1_row(
                    r1_rec['label'], s_lbl, dm['BA3'], dm['NLL3'],
                    dm.get('aurocs', {}), dm.get('eff_purity_80', {}), dm.get('eff_purity_90', {}),
                    dm.get('BA4'), dm.get('tau_eff_1pct'), None,
                ))
    r2_d2 = refs.get('R2_b1_d2_t4_s42', {})
    r2_rep = refs.get('W3_03_R2_D2_replay', {}).get('working_points', {}).get('0.01', {}).get('slices', {})
    for d_key, s_lbl in dom_to_slice:
        dm = r2_d2.get(d_key)
        if dm:
            rp = r2_rep.get(s_lbl, {})
            rows.append(_m1_row(
                'R2 B1-D2 T4 (4p, 300k)', s_lbl, dm['BA3'], dm['NLL3'], {}, {}, {},
                dm.get('BA4'), rp.get('raw_global_cut', {}).get('tau_eff'), rp.get('regional_S_tau_global_cut', {}).get('tau_eff'),
            ))

    for mname in model_names:
        ev = models_out[mname]
        for sname in ('All', 'Truth_pixel_centered', 'Truth_large_scintillator', 'Tune_non_screen_large_scintillator', 'Tune_E_le_2p3TeV', 'HighE_gt_2p3TeV_non_tau'):
            sl = ev['slices'][sname]
            wp1 = sl.get('working_points', {}).get('0.01', {})
            rows.append(_m1_row(
                mname, sname, sl.get('BA3'), sl.get('NLL3'),
                sl.get('aurocs3') or sl.get('aurocs') or {},
                sl.get('eff_purity_80_3') or sl.get('eff_purity_80') or {},
                sl.get('eff_purity_90_3') or sl.get('eff_purity_90') or {},
                sl.get('BA4_raw'), wp1.get('raw_global', {}).get('tau_eff'), wp1.get('regional_S_tau', {}).get('tau_eff'),
            ))
    pd.DataFrame(rows).to_csv(w305_root / 'm1_collaboration_table.csv', index=False)

    _make_figures(summary, refs, manifests, w305_root / 'figures')
    print(json.dumps({
        'status': 'complete',
        'primary_checkpoint': 'best_screen',
        'v1_gate_passed': v1_gate_passed,
        'best_large_scintillator_BA3': summary['best_large_scintillator_BA3'],
        '300k_All_BA4': models_out['b1pp_300k_24p_t4']['slices']['All']['BA4_raw'],
        '300k_All_BA3': models_out['b1pp_300k_24p_t4']['slices']['All']['BA3'],
        '300k_Pix_BA3': models_out['b1pp_300k_24p_t4']['slices']['Truth_pixel_centered']['BA3'],
        '300k_Large_BA3': large_300k_ba3,
        '300k_Tune_non_screen_Large_BA3': large_300k_non_screen_ba3,
        '100k_T3_Large_BA3': large_100k_t3_ba3,
    }, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--w305-root', type=Path, default=Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-05'))
    parser.add_argument('--w301-root', type=Path, default=Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-01'))
    args = parser.parse_args()
    evaluate_all(args.w305_root, args.w301_root)


if __name__ == '__main__':
    main()
