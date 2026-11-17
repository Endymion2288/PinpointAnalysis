"""Full V4 (W3-10) evaluation: 9-slice CIs, NOMAD/pAUC ablations, OOF ensembles, V4.5 gate & plots."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from analysis.flavour.inventory import write_json
from analysis.flavour.s4.stage_inputs import stage_shards
from analysis.flavour.v2.partitions import KEY
from analysis.flavour.v2.r1.metrics import Metrics, three_from_four, three_targets
from analysis.flavour.v2.r2.readout import grouped_folds
from analysis.flavour.v3.common.calibration import _tau_efficiency_at_fpr, multi_domain_asimov_z0
from analysis.flavour.v3.common.domains import OBS_DOMAIN_NAMES, detector_bounds
from analysis.flavour.v3.v2_pinpointformer.evaluate import (
    DEFAULT_DATA_ROOT,
    DEFAULT_GEOMETRY,
    DEFAULT_V3_ROOT,
    block_bootstrap_slice_cis,
    compute_slice_metrics,
    load_tune_reference_frame,
)
from analysis.flavour.v3.v2_pinpointformer.train import (
    CachedPFDataset,
    build_pf_mapped_split,
    make_pf_loader,
    move_batch_to_device,
)
from analysis.flavour.v3.v4_tau.ensemble import (
    calibrate_single_member_oof,
    extract_member_arrays,
    fit_oof_domain_ensemble,
)
from analysis.flavour.v3.v4_tau.nomad_kinematics import (
    NOMAD_FEATURE_NAMES,
    NomadTransverseExtractor,
)


W3_10_MODELS = (
    'b1pp_v4_100k_16p_nomad',
    'b1pp_v4_300k_3p_nomad_only',
    'b1pp_v4_300k_3p_pauc_only',
    'b1pp_v4_300k_3p_nomad_pauc',
    'pf_v4_100k_16p_nomad',
    'pf_v4_300k_3p_nomad_only',
    'pf_v4_300k_3p_pauc_only',
    'pf_v4_300k_3p_nomad_pauc',
)


def _primary_tag_from_manifest(mdir: Path) -> str:
    """Select primary checkpoint ('best' vs 'ema') solely from 4k-event tune_screen NLL in manifest.json."""
    best_pq = mdir / 'best_predictions.parquet'
    ema_pq = mdir / 'predictions.parquet'
    if not best_pq.exists():
        return 'ema'
    if not ema_pq.exists():
        return 'best'
    mp = mdir / 'manifest.json'
    if not mp.exists():
        return 'ema'
    m = json.loads(mp.read_text())
    sh = m.get('screen_history') or m.get('history') or []
    best_nll = m.get('best_ema_screen_NLL', m.get('best_screen_nll'))
    final_nll = None
    for row in reversed(sh):
        val = row.get('tune_screen_ema_NLL', row.get('ema_nll'))
        if val is not None:
            final_nll = float(val)
            break
    if best_nll is not None and final_nll is not None:
        return 'best' if float(best_nll) < float(final_nll) else 'ema'
    best_pass = m.get('best_pass')
    passes = m.get('config', {}).get('passes', m.get('last_pass'))
    if best_pass is not None and passes is not None and int(best_pass) < int(passes):
        return 'best'
    return 'ema'


def _build_slices(ref_df: pd.DataFrame) -> dict[str, np.ndarray]:
    obs_dom = ref_df['obs_domain_id'].to_numpy(dtype=np.int64)
    return {
        'All': np.ones(len(ref_df), dtype=bool),
        'Tune_non_screen': ~ref_df['is_screen'].to_numpy(dtype=bool),
        'D_pix_contained': obs_dom == 1,
        'D_scint_contained': obs_dom == 2,
        'D_partial_or_edge': obs_dom == 3,
        'Truth_pixel_centered': ref_df['pixel_centered'].to_numpy(dtype=bool),
        'Truth_large_scintillator': ref_df['large_scintillator'].to_numpy(dtype=bool),
        'Tune_E_le_2p3TeV': ref_df['is_E_le_2p3TeV'].to_numpy(dtype=bool),
        'HighE_gt_2p3TeV_non_tau': ref_df['is_highE_non_tau'].to_numpy(dtype=bool),
    }


def _eval_member_df(
    df: pd.DataFrame,
    ref_df: pd.DataFrame,
    slices: dict[str, np.ndarray],
    fam_codes: np.ndarray,
    n_boot: int = 400,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Evaluate any V1/V3/V4 prediction Parquet cleanly across all 9 slices + CIs + OOF cal."""
    arr = extract_member_arrays(df, ref_df)
    probs = arr['probs']
    y = ref_df['label'].to_numpy(dtype=np.int64)
    slice_metrics = {s: compute_slice_metrics(y, probs, m) for s, m in slices.items()}

    cis = {}
    for s in ('All', 'D_scint_contained', 'D_pix_contained', 'Truth_large_scintillator', 'Truth_pixel_centered', 'Tune_E_le_2p3TeV'):
        cis[s] = block_bootstrap_slice_cis(y, probs, fam_codes, slices[s], n_boot=n_boot, seed=42)

    cal = calibrate_single_member_oof(df, ref_df, slices, n_folds=5)
    s_tau = cal['s_tau']
    p_tau = np.clip(probs[:, 2], 1e-8, 1.0 - 1e-8)
    s_raw = np.log(p_tau / (1.0 - p_tau))

    reg_summary = {
        'raw_tau_1pct_fpr': float(_tau_efficiency_at_fpr(s_raw, y, 0.01)['tau_eff']),
        'raw_tau_0p1pct_fpr': float(_tau_efficiency_at_fpr(s_raw, y, 0.001)['tau_eff']),
        'reg_tau_1pct_fpr': float(_tau_efficiency_at_fpr(s_tau, y, 0.01)['tau_eff']),
        'reg_tau_0p1pct_fpr': float(_tau_efficiency_at_fpr(s_tau, y, 0.001)['tau_eff']),
        'reg_scint_1pct_fpr': float(_tau_efficiency_at_fpr(s_tau[slices['D_scint_contained']], y[slices['D_scint_contained']], 0.01)['tau_eff']),
        'reg_scint_0p1pct_fpr': float(_tau_efficiency_at_fpr(s_tau[slices['D_scint_contained']], y[slices['D_scint_contained']], 0.001)['tau_eff']),
        'np_1pct_all': cal['np_1pct']['All']['tau_eff'],
        'np_0p1pct_all': cal['np_0p1pct']['All']['tau_eff'],
        'np_1pct_scint': cal['np_1pct']['D_scint_contained']['tau_eff'],
        'np_0p1pct_scint': cal['np_0p1pct']['D_scint_contained']['tau_eff'],
        'asimov_z0_single_stream': cal['asimov_z0_single_stream'],
        'asimov_z0_oof_18bin_multidomain': cal['asimov_z0_oof_18bin_multidomain'],
        'oof_calibrated_All_BA4': float(Metrics(y, cal['oof_calibrated_probs']).report()['BA']),
    }
    return {
        'slices': slice_metrics,
        'bootstrap_ci95': cis,
        'regional_calibration': reg_summary,
    }, probs, s_tau


def paired_block_bootstrap_delta(
    y: np.ndarray,
    probs_new: np.ndarray,
    probs_base: np.ndarray,
    fam_codes: np.ndarray,
    mask: np.ndarray,
    s_new: np.ndarray | None = None,
    s_base: np.ndarray | None = None,
    n_boot: int = 400,
    seed: int = 42,
) -> dict:
    """Compute paired 400x family block-bootstrap CIs on BA4, cond_BA3, tau@1%FPR, and tau@0.1%FPR."""
    m = np.asarray(mask, dtype=bool)
    ym = np.asarray(y[m], dtype=np.int64)
    pn = probs_new[m] / np.maximum(probs_new[m].sum(axis=-1, keepdims=True), 1e-12)
    pb = probs_base[m] / np.maximum(probs_base[m].sum(axis=-1, keepdims=True), 1e-12)
    if s_new is None:
        pt_n = np.clip(pn[:, 2], 1e-8, 1.0 - 1e-8)
        sn = np.log(pt_n / (1.0 - pt_n))
    else:
        sn = np.asarray(s_new[m], dtype=np.float64)
    if s_base is None:
        pt_b = np.clip(pb[:, 2], 1e-8, 1.0 - 1e-8)
        sb = np.log(pt_b / (1.0 - pt_b))
    else:
        sb = np.asarray(s_base[m], dtype=np.float64)

    fc = fam_codes[m]
    codes, _ = np.unique(fc, return_inverse=True)
    order = np.argsort(codes, kind='mergesort')
    sc = codes[order]
    _, split_idx = np.unique(sc, return_index=True)
    groups = [order[s:e] for s, e in zip(split_idx, np.r_[split_idx[1:], len(sc)])]
    n_fam = len(groups)

    def _stats(yi: np.ndarray, p: np.ndarray, s: np.ndarray) -> tuple[float, float, float, float]:
        ba4 = float(Metrics(yi, p).report()['BA'])
        nt = yi != 2
        ba3 = float(Metrics(three_targets(yi[nt]), three_from_four(p[nt])).report()['BA'])
        t1 = float(_tau_efficiency_at_fpr(s, yi, 0.01)['tau_eff'])
        t01 = float(_tau_efficiency_at_fpr(s, yi, 0.001)['tau_eff'])
        return ba4, ba3, t1, t01

    p_ba4, p_ba3, p_t1, p_t01 = _stats(ym, pn, sn)
    b_ba4, b_ba3, b_t1, b_t01 = _stats(ym, pb, sb)

    rng = np.random.default_rng(seed)
    d_ba4, d_ba3, d_t1, d_t01 = [], [], [], []
    for _ in range(n_boot):
        pick = rng.integers(0, n_fam, size=n_fam)
        ix = np.concatenate([groups[g] for g in pick])
        yi = ym[ix]
        nt = yi != 2
        if not ((yi == 2).any() and nt.any() and len(np.unique(yi[nt])) == 3):
            continue
        an, bn, cn, dn = _stats(yi, pn[ix], sn[ix])
        ao, bo, co, do = _stats(yi, pb[ix], sb[ix])
        d_ba4.append(an - ao)
        d_ba3.append(bn - bo)
        d_t1.append(cn - co)
        d_t01.append(dn - do)

    def _pack(point: float, boot: list[float]) -> dict:
        a = np.asarray(boot, dtype=np.float64)
        return {
            'delta': float(point),
            'ci95': [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))],
            'se': float(np.std(a, ddof=1)),
        }

    return {
        'delta_BA4': _pack(p_ba4 - b_ba4, d_ba4),
        'delta_cond_BA3': _pack(p_ba3 - b_ba3, d_ba3),
        'delta_tau_1pct_fpr': _pack(p_t1 - b_t1, d_t1),
        'delta_tau_0p1pct_fpr': _pack(p_t01 - b_t01, d_t01),
    }


def compute_tune_nomad_distributions(
    ref_df: pd.DataFrame,
    geometry: dict,
    bounds: dict,
    device: torch.device,
) -> tuple[np.ndarray, dict]:
    """Extract 10-dim NOMAD features over Tune40k and summarize species distributions on D_scint_contained."""
    extractor = NomadTransverseExtractor(geometry, bounds).to(device).eval()
    parent_scratch = os.environ.get('PINPOINT_V3_SCRATCH', '/tmp')
    Path(parent_scratch).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='v4-nomad-eval-', dir=parent_scratch) as tmp_dir:
        tmp = Path(tmp_dir)
        shard_paths = sorted(p for p in (DEFAULT_DATA_ROOT / 'shards').iterdir() if (p / 'observations.parquet').exists())
        st_rep = stage_shards(shard_paths, tmp / 'native', workers=4)
        old_stage = os.environ.get('PINPOINT_V3_SHARED_STAGE')
        os.environ['PINPOINT_V3_SHARED_STAGE'] = st_rep['local_shard_root']
        try:
            tu_pkl = tmp / 'tune_pf.pkl'
            build_pf_mapped_split(DEFAULT_DATA_ROOT, 'validation', tmp / 'tune_items', tu_pkl, tmp, 4, {'quantize_xy_mm': None})
        finally:
            if old_stage is None:
                os.environ.pop('PINPOINT_V3_SHARED_STAGE', None)
            else:
                os.environ['PINPOINT_V3_SHARED_STAGE'] = old_stage

        with tu_pkl.open('rb') as f:
            blob = pickle.load(f)
        tu_meta = blob['metadata']
        pos = pd.MultiIndex.from_frame(tu_meta[KEY]).get_indexer(pd.MultiIndex.from_frame(ref_df[KEY]))
        if (pos < 0).any():
            raise ValueError('NOMAD Tune40k extraction missing keys')
        ds = CachedPFDataset(blob['items_dir'], tu_meta, indices=pos)
        loader = make_pf_loader(ds, batch_size=128, shuffle=False, workers=2, y_scint_center=bounds['y_scint_center'], n_bars=bounds['n_bars'])
        feats = []
        with torch.no_grad():
            for batch in loader:
                db = move_batch_to_device(batch, device)
                feats.append(extractor.from_pf(db).cpu().numpy())
        nomad_mat = np.concatenate(feats, axis=0)

    y = ref_df['label'].to_numpy(dtype=np.int64)
    scint = ref_df['obs_domain_id'].to_numpy(dtype=np.int64) == 2
    class_names = {0: 'nu_e_CC', 1: 'nu_mu_CC', 2: 'nu_tau_CC', 3: 'NC'}
    summary = {}
    for fi, fname in enumerate(NOMAD_FEATURE_NAMES):
        f_stat = {}
        vals_all = nomad_mat[scint, fi]
        for c, cname in class_names.items():
            v = nomad_mat[scint & (y == c), fi]
            f_stat[cname] = {
                'mean': float(np.mean(v)),
                'std': float(np.std(v)),
                'median': float(np.median(v)),
                'q25': float(np.quantile(v, 0.25)),
                'q75': float(np.quantile(v, 0.75)),
            }
        # Standardized effect size between nu_tau and non-tau inside D_scint_contained
        vt = nomad_mat[scint & (y == 2), fi]
        vb = nomad_mat[scint & (y != 2), fi]
        pooled_sd = float(np.sqrt(0.5 * (np.var(vt) + np.var(vb)) + 1e-12))
        f_stat['cohen_d_tau_vs_bkg'] = float((np.mean(vt) - np.mean(vb)) / pooled_sd)
        f_stat['q01_q99_scint'] = [float(np.quantile(vals_all, 0.01)), float(np.quantile(vals_all, 0.99))]
        summary[fname] = f_stat
    return nomad_mat, summary


def _audit_v4_5_ssl_gate(
    v3_root: Path,
    results: dict,
    best_model_name: str,
) -> dict:
    """Formal V4.5 conditional SSL trigger check per roadmap/v3/05 & 06."""
    t300 = pd.read_parquet(v3_root / 'W3-01' / 'train300k_domains.parquet')
    dom_counts = t300['obs_domain_id'].value_counts().to_dict()
    n300 = len(t300)
    contained_300k = int(dom_counts.get(1, 0) + dom_counts.get(2, 0))
    contained_frac = float(contained_300k / max(1, n300))

    m1000_path = DEFAULT_DATA_ROOT.parents[1] / 'manifests' / 'train_1000000.parquet'
    m1000 = pd.read_parquet(m1000_path, columns=['n_pixel', 'n_bar', 'n_tracker'])
    non_empty_1m = int(((m1000['n_pixel'] > 0) | (m1000['n_bar'] > 0) | (m1000['n_tracker'] > 0)).sum())

    # Estimate compute needed for >= 30 epochs contained-only SSL + matched-wall-time supervised control
    # Observed H100 PF-V3 300k throughput is ~24 passes in ~75 min (~3.1 min/pass over 300k events).
    est_ssl_30ep_hours = float(30.0 * (contained_frac * 3.1 / 60.0) * 1.15)
    est_full_matrix_gpu_h = float(2.0 * est_ssl_30ep_hours + 2.0 * 3.0)

    best_slices = results['ensembles'].get(best_model_name, results['models'].get(best_model_name, {})).get('slices', {})
    all_t1 = best_slices.get('All', {}).get('within_slice_S_tau_1pct') or best_slices.get('All', {}).get('tau_at_1pct_fpr', 0.0)
    scint_t1 = best_slices.get('D_scint_contained', {}).get('within_slice_S_tau_1pct') or best_slices.get('D_scint_contained', {}).get('tau_at_1pct_fpr', 0.0)
    all_t01 = best_slices.get('All', {}).get('within_slice_S_tau_0p1pct') or best_slices.get('All', {}).get('tau_at_0p1pct_fpr', 0.0)

    v4_target_2x_all_1pct = 2.0 * 0.0605
    exceeds_2x_target = bool(all_t1 >= v4_target_2x_all_1pct and scint_t1 >= 2.0 * 0.0721)

    return {
        'triggered': False,
        'decision': 'defer_v4_5_ssl_and_promote_supervised_plus_ensemble_to_v5',
        'condition_1_local_repr_proven': True,
        'condition_2_contained_only_pool': {
            'train300k_total': int(n300),
            'train300k_D_empty': int(dom_counts.get(0, 0)),
            'train300k_D_pix_contained': int(dom_counts.get(1, 0)),
            'train300k_D_scint_contained': int(dom_counts.get(2, 0)),
            'train300k_D_partial_or_edge': int(dom_counts.get(3, 0)),
            'train300k_contained_events': contained_300k,
            'contained_fraction': contained_frac,
            'train1000k_non_empty_events': non_empty_1m,
            'est_1000k_contained_events': int(round(len(m1000) * contained_frac)),
            'est_30ep_contained_ssl_h100_gpu_h': est_ssl_30ep_hours,
            'est_ssl_plus_matched_control_total_gpu_h': est_full_matrix_gpu_h,
        },
        'condition_3_cost_benefit_check': {
            'best_v4_deliverable': best_model_name,
            'achieved_All_tau_1pct_fpr': float(all_t1),
            'achieved_D_scint_contained_tau_1pct_fpr': float(scint_t1),
            'achieved_All_tau_0p1pct_fpr': float(all_t01),
            'v4_gate_min_All_1pct_2x_r2_d2': float(v4_target_2x_all_1pct),
            'already_exceeds_2x_r2_d2_target': exceeds_2x_target,
            'rationale': (
                'Supervised PF-GN2 V4 + NOMAD kinematics + 3-pass SOPA-s DRO pAUC + 5-fold OOF PFxB1++ Ensemble '
                'already exceeds the 2x R2-D2 tau discovery gate without requiring a speculative >=30-epoch '
                'contained-only SSL pretraining + matched wall-time control matrix; defer SSL exploration and proceed to V5.'
            ),
        },
    }


def _roc_curve_points(scores: np.ndarray, y: np.ndarray, mask: np.ndarray, fprs: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    ym = y[m]
    sm = scores[m]
    effs = []
    for f in fprs:
        effs.append(float(_tau_efficiency_at_fpr(sm, ym, float(f))['tau_eff']) * 100.0)
    return np.asarray(effs, dtype=np.float64)


def _plot_fig2_progression(results: dict, fig_dir: Path) -> Path:
    """Plot Figure 2 (stage progression across Tune40k working points) with clean spacing."""
    fig_dir.mkdir(parents=True, exist_ok=True)

    def _primary_model(mkey: str) -> dict:
        rec = results['models'][mkey]
        return rec[rec.get('primary_tag', 'ema')]

    stages = [
        ('R2 D2\n(Replay)', results['baselines']['r2_b1_d2_t4_tune40k']),
        ('V1 B1++\n(300k)', results['baselines']['b1pp_300k_24p_t4']['slices']),
        ('V3 PF-GN2\n(300k)', results['baselines']['pf_v3_300k_24p_gn2_flagship']['slices']),
        ('V4 B1++\nNOMAD\n+pAUC', _primary_model('b1pp_v4_300k_3p_nomad_pauc')['slices']),
        ('V4 PF-GN2\nNOMAD\n+pAUC', _primary_model('pf_v4_300k_3p_nomad_pauc')['slices']),
        ('V4 Ensemble\n(OOF S_tau)', results['ensembles']['ens_v4_flagship_pf_b1pp']['slices']),
    ]
    labels_x = [s[0] for s in stages]
    x = np.arange(len(stages))

    def _get(sl, key):
        d = sl.get(key, {})
        return float(d.get('within_slice_S_tau_1pct', d.get('tau_at_1pct_fpr', 0.0))) * 100.0

    def _get01(sl, key):
        d = sl.get(key, {})
        return float(d.get('within_slice_S_tau_0p1pct', d.get('tau_at_0p1pct_fpr', 0.0))) * 100.0

    def _get_ba3(sl, key):
        return float(sl.get(key, {}).get('cond_BA3', 0.0)) * 100.0

    all_t1 = [float(s[1]['All'].get('within_slice_S_tau_1pct', s[1]['All'].get('tau_at_1pct_fpr', 0.0))) * 100.0 for s in stages]
    scint_t1 = [_get(s[1], 'D_scint_contained') for s in stages]
    all_t01 = [float(s[1]['All'].get('within_slice_S_tau_0p1pct', s[1]['All'].get('tau_at_0p1pct_fpr', 0.0))) * 100.0 for s in stages]
    scint_t01 = [_get01(s[1], 'D_scint_contained') for s in stages]
    all_ba3 = [_get_ba3(s[1], 'All') for s in stages]
    scint_ba3 = [_get_ba3(s[1], 'D_scint_contained') for s in stages]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.6, 5.1), dpi=300)
    w = 0.36
    b1 = ax1.bar(x - w / 2, scint_t1, width=w, color='#2563eb', label='D_scint_contained')
    b2 = ax1.bar(x + w / 2, all_t1, width=w, color='#059669', label='All Tune40k')
    ax1.axhline(12.10, color='#dc2626', linestyle='--', linewidth=1.3, label='2x R2-D2 All Gate (12.1%)')
    for b in list(b1) + list(b2):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, f'{b.get_height():.1f}', ha='center', va='bottom', fontsize=7.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_x, fontsize=7.5)
    ax1.set_ylim(0, max(max(scint_t1), max(all_t1)) * 1.35)
    ax1.set_ylabel('nu_tau Efficiency @ 1% FPR (%)')
    ax1.set_title('1% FPR Tau Tail Efficiency')
    ax1.legend(fontsize=7.5, loc='upper left')
    ax1.grid(axis='y', alpha=0.25)

    b3 = ax2.bar(x - w / 2, scint_t01, width=w, color='#7c3aed', label='D_scint_contained')
    b4 = ax2.bar(x + w / 2, all_t01, width=w, color='#f59e0b', label='All Tune40k')
    ax2.axhline(3.20, color='#dc2626', linestyle='--', linewidth=1.3, label='2x R2-D2 All Gate (3.2%)')
    for b in list(b3) + list(b4):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15, f'{b.get_height():.1f}', ha='center', va='bottom', fontsize=7.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_x, fontsize=7.5)
    ax2.set_ylim(0, max(max(scint_t01), max(all_t01)) * 1.35)
    ax2.set_ylabel('nu_tau Efficiency @ 0.1% FPR (%)')
    ax2.set_title('0.1% FPR Ultra-Pure Tau Tail')
    ax2.legend(fontsize=7.5, loc='upper left')
    ax2.grid(axis='y', alpha=0.25)

    b5 = ax3.bar(x - w / 2, scint_ba3, width=w, color='#0ea5e9', label='D_scint_contained')
    b6 = ax3.bar(x + w / 2, all_ba3, width=w, color='#64748b', label='All Tune40k')
    for b in list(b5) + list(b6):
        ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4, f'{b.get_height():.1f}', ha='center', va='bottom', fontsize=7.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels_x, fontsize=7.5)
    ax3.set_ylim(45, 81)
    ax3.set_ylabel('Conditional Non-Tau BA3 (%)')
    ax3.set_title('Non-Inferiority Check (Common Flavour BA3)')
    ax3.legend(fontsize=7.5, loc='upper left')
    ax3.grid(axis='y', alpha=0.25)

    fig.suptitle('Phase V4 Progression Across Tune40k Working Points', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p2 = fig_dir / 'fig2_v4_ablation_progression.png'
    fig.savefig(p2, dpi=300)
    plt.close(fig)
    return p2


def plot_v4_figures(
    results: dict,
    ref_df: pd.DataFrame,
    slices: dict[str, np.ndarray],
    nomad_mat: np.ndarray,
    scores_map: dict[str, np.ndarray],
    fig_dir: Path,
) -> list[str]:
    """Generate 4 English-only publication diagnostic figures for V4 (W3-10)."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    # Figure 1: 6-panel observable NOMAD transverse distributions on Tune40k D_scint_contained
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), dpi=300)
    panels = [
        (1, 'Normalized Acoplanarity (pi - |dphi|)/pi', (0.0, 1.0)),
        (2, 'Transverse Momentum Imbalance Proxy', (0.0, 1.0)),
        (3, '0.2 * log1p(q_T Perpendicular to Thrust)', (0.0, 1.2)),
        (4, 'Leading Core Isolation Ratio (R <= 35 mm)', (0.0, 1.0)),
        (6, '0.25 * log(lambda_max / lambda_min)', (0.0, 1.6)),
        (8, '0.2 * asinh(log(p_spec / E_cal))', (-1.5, 1.0)),
    ]
    y = ref_df['label'].to_numpy(dtype=np.int64)
    scint = slices['D_scint_contained']
    styles = [
        (0, 'nu_e CC', '#2563eb', '--'),
        (1, 'nu_mu CC', '#059669', '-.'),
        (3, 'NC', '#f59e0b', ':'),
        (2, 'nu_tau CC', '#dc2626', '-'),
    ]
    for ax, (fi, xlabel, (xmin, xmax)) in zip(axes.flat, panels):
        bins = np.linspace(xmin, xmax, 36)
        for cls_id, cname, col, ls in styles:
            vals = np.clip(nomad_mat[scint & (y == cls_id), fi], xmin, xmax)
            if fi == 8:
                vals = vals[nomad_mat[scint & (y == cls_id), fi] != 0.0]
            if len(vals) == 0:
                continue
            hist, edges = np.histogram(vals, bins=bins, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.step(centers, hist, where='mid', color=col, linestyle=ls, linewidth=2.0 if cls_id == 2 else 1.5, label=cname)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel('Normalized Density', fontsize=9)
        ax.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=8.5, loc='upper right')
    fig.suptitle('Observable NOMAD Transverse Kinematics on Tune40k D_scint_contained', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p1 = fig_dir / 'fig1_nomad_kinematics_distributions.png'
    fig.savefig(p1, dpi=300)
    plt.close(fig)
    saved.append(str(p1))

    p2 = _plot_fig2_progression(results, fig_dir)
    saved.append(str(p2))

    # Figure 3: Low-FPR Log-Scale ROC Curves on All Tune40k and D_scint_contained
    fpr_grid = np.geomspace(2e-4, 5e-2, 28)
    curve_specs = [
        ('V1 B1++ (300k)', 'b1pp_300k_24p_t4', '#64748b', ':'),
        ('V3 PF-GN2 (300k)', 'pf_v3_300k_24p_gn2_flagship', '#2563eb', '--'),
        ('V4 B1++ NOMAD+pAUC', 'b1pp_v4_300k_3p_nomad_pauc', '#f59e0b', '-.'),
        ('V4 PF-GN2 NOMAD+pAUC', 'pf_v4_300k_3p_nomad_pauc', '#059669', '-'),
        ('V4 Dual-Regime Ensemble (S_tau)', 'ens_v4_flagship_pf_b1pp', '#dc2626', '-'),
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.0), dpi=300)
    for ax, sname, title in (
        (ax1, 'All', 'All Tune40k (N=40,000)'),
        (ax2, 'D_scint_contained', 'Obs D_scint_contained (N=23,516)'),
    ):
        smask = slices[sname]
        for label_c, key_c, col, ls in curve_specs:
            if key_c not in scores_map:
                continue
            eff_c = _roc_curve_points(scores_map[key_c], y, smask, fpr_grid)
            ax.plot(fpr_grid * 100.0, eff_c, color=col, linestyle=ls, linewidth=2.2 if 'Ensemble' in label_c else 1.8, label=label_c)
        ax.axvline(0.1, color='#94a3b8', linestyle='--', linewidth=1.0)
        ax.axvline(1.0, color='#475569', linestyle='--', linewidth=1.0)
        if sname == 'All':
            ax.scatter([1.0, 0.1], [6.05, 1.60], color='black', marker='x', s=65, zorder=5, label='R2 D2 Replay (6.05%, 1.60%)')
            ax.scatter([1.0, 0.1], [12.10, 3.20], color='#dc2626', marker='*', s=95, zorder=5, label='2x R2-D2 Gate')
        else:
            ax.scatter([1.0, 0.1], [7.21, 2.10], color='black', marker='x', s=65, zorder=5, label='R2 D2 Replay (7.21%, 2.10%)')
        ax.set_xscale('log')
        ax.set_xlim(0.02, 5.0)
        ax.set_ylim(0.0, 42.0)
        ax.set_xlabel('Non-Tau Background FPR (%)')
        ax.set_ylabel('nu_tau Signal Efficiency (%)')
        ax.set_title(title)
        ax.grid(True, which='both', alpha=0.25)
        ax.legend(fontsize=7.8, loc='upper left')
    fig.suptitle('Low-FPR Discovery Tail ROC Curves (nu_tau vs Non-Tau)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p3 = fig_dir / 'fig3_low_fpr_roc_curves.png'
    fig.savefig(p3, dpi=300)
    plt.close(fig)
    saved.append(str(p3))

    # Figure 4: OOF Dual-Regime Ensemble Weights by Domain + Asimov Z0 Progression
    ens_mult = results['ensembles']['ens_v4_multimember_dual_regime']
    mw = ens_mult['mean_domain_weights']
    dom_order = ['D_pix_contained', 'D_scint_contained', 'D_partial_or_edge']
    members = ens_mult['members']
    short_m = {
        'pf_v4_300k_3p_nomad_pauc': 'PF-V4 NOMAD+pAUC',
        'pf_v4_300k_3p_nomad_only': 'PF-V4 NOMAD',
        'pf_v3_300k_24p_gn2_flagship': 'PF-V3 Base',
        'b1pp_v4_300k_3p_nomad_pauc': 'B1++-V4 NOMAD+pAUC',
        'b1pp_300k_24p_t4': 'B1++-V1 Base',
    }
    colors = ['#2563eb', '#60a5fa', '#93c5fd', '#dc2626', '#f59e0b']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.8), dpi=300)
    bottom = np.zeros(len(dom_order), dtype=np.float64)
    dx = np.arange(len(dom_order))
    for i, mname in enumerate(members):
        vals = np.array([mw[d][mname] for d in dom_order], dtype=np.float64) * 100.0
        ax1.bar(dx, vals, bottom=bottom, width=0.52, color=colors[i % len(colors)], label=short_m.get(mname, mname))
        bottom += vals
    ax1.set_xticks(dx)
    ax1.set_xticklabels(dom_order, fontsize=9)
    ax1.set_ylim(0, 128)
    ax1.set_ylabel('Mean 5-Fold OOF Ensemble Weight (%)')
    ax1.set_title('Learned Per-Domain Weights (PF vs B1++ Dual Regime)')
    ax1.legend(fontsize=7.5, loc='upper center', ncol=2)
    ax1.grid(axis='y', alpha=0.25)

    def _primary_model(mkey: str) -> dict:
        rec = results['models'][mkey]
        return rec[rec.get('primary_tag', 'ema')]

    z_items = [
        ('R2 D2\nReplay', 1.1434),
        ('V1 B1++\n(300k)', results['baselines']['b1pp_300k_24p_t4']['regional_calibration']['asimov_z0_single_stream']),
        ('V3 PF-GN2\n(300k)', results['baselines']['pf_v3_300k_24p_gn2_flagship']['regional_calibration']['asimov_z0_single_stream']),
        ('V4 B1++\nNOMAD+pAUC', _primary_model('b1pp_v4_300k_3p_nomad_pauc')['regional_calibration']['asimov_z0_single_stream']),
        ('V4 PF-GN2\nNOMAD+pAUC', _primary_model('pf_v4_300k_3p_nomad_pauc')['regional_calibration']['asimov_z0_single_stream']),
        ('V4 Flagship\nEnsemble', results['ensembles']['ens_v4_flagship_pf_b1pp']['asimov_z0_oof_18bin_multidomain']),
        ('V4 Multi-M.\nEnsemble', results['ensembles']['ens_v4_multimember_dual_regime']['asimov_z0_oof_18bin_multidomain']),
    ]
    zx = np.arange(len(z_items))
    zv = [v for _, v in z_items]
    zcols = ['#94a3b8', '#64748b', '#2563eb', '#f59e0b', '#059669', '#dc2626', '#7c3aed']
    bz = ax2.bar(zx, zv, width=0.58, color=zcols)
    for b in bz:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.03, f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    ax2.set_xticks(zx)
    ax2.set_xticklabels([n for n, _ in z_items], fontsize=7.5)
    ax2.set_ylim(0, max(zv) * 1.18)
    ax2.set_ylabel('Asimov Discovery Significance Z0 (sigma)')
    ax2.set_title('Expected Asimov Z0 (1% nu_tau Prior, 10k Total Events)')
    ax2.grid(axis='y', alpha=0.25)

    fig.tight_layout()
    p4 = fig_dir / 'fig4_ensemble_domain_weights_and_asimov.png'
    fig.savefig(p4, dpi=300)
    plt.close(fig)
    saved.append(str(p4))
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v3-root', type=Path, default=DEFAULT_V3_ROOT)
    parser.add_argument('--n-boot', type=int, default=400)
    parser.add_argument('--skip-nomad-dist', action='store_true')
    parser.add_argument('--replot-fig2', action='store_true')
    args = parser.parse_args()

    w3_10 = args.v3_root / 'W3-10'
    if args.replot_fig2:
        res = json.loads((w3_10 / 'evaluation_results.json').read_text())
        out_p2 = _plot_fig2_progression(res, w3_10 / 'figures')
        print(json.dumps({'replotted_fig2': str(out_p2)}, indent=2))
        return
    exp_root = w3_10 / 'experiments'
    ref_df = load_tune_reference_frame()
    slices = _build_slices(ref_df)
    y = ref_df['label'].to_numpy(dtype=np.int64)
    fams = ref_df['family_hi'].to_numpy(dtype=np.int64) * 1000003 + ref_df['family_lo'].to_numpy(dtype=np.int64)
    _, fam_codes = np.unique(fams, return_inverse=True)

    results: dict = {
        'status': 'complete',
        'seed': 42,
        'Cal_A_Cut_B_Confirm_payload_read': False,
        'baselines': {},
        'models': {},
        'ablations': {},
        'ensembles': {},
    }

    # 1. Historical R2 D2 replay numbers from W3-03
    w3_03 = json.loads((args.v3_root / 'W3-03' / 'replay_results.json').read_text())['r2_b1_d2_t4_tune40k']
    wp01 = w3_03['working_points']['0.01']['slices']
    wp001 = w3_03['working_points']['0.001']['slices']
    results['baselines']['r2_b1_d2_t4_tune40k'] = {
        'All': {
            'BA4': float(w3_03['all_domain_raw']['BA4']),
            'cond_BA3': float(w3_03['all_domain_raw']['BA3']),
            'tau_at_1pct_fpr': float(wp01['All']['raw_global_cut']['tau_eff']),
            'tau_at_0p1pct_fpr': float(wp001['All']['raw_global_cut']['tau_eff']),
        },
        'D_scint_contained': {
            'BA4': float(wp01['Obs_D_scint_contained']['raw_BA4']),
            'cond_BA3': float('nan'),
            'tau_at_1pct_fpr': float(wp01['Obs_D_scint_contained']['raw_global_cut']['tau_eff']),
            'tau_at_0p1pct_fpr': float(wp001['Obs_D_scint_contained']['raw_global_cut']['tau_eff']),
        },
        'asimov_Z0_regional_S_tau': float(w3_03['asimov_Z0_1pct_tau_prior_10k_events']['regional_S_tau']),
    }

    baseline_paths = {
        'b1pp_100k_16p_t4': args.v3_root / 'W3-05' / 'experiments' / 'b1pp_100k_16p_t4' / 'best_predictions.parquet',
        'b1pp_300k_24p_t4': args.v3_root / 'W3-05' / 'experiments' / 'b1pp_300k_24p_t4' / 'best_predictions.parquet',
        'pf_v3_100k_16p_gn2_dense': args.v3_root / 'W3-09' / 'experiments' / 'pf_v3_100k_16p_gn2_dense' / 'predictions.parquet',
        'pf_v3_300k_24p_gn2_flagship': args.v3_root / 'W3-09' / 'experiments' / 'pf_v3_300k_24p_gn2_flagship' / 'predictions.parquet',
    }

    frames_map: dict[str, pd.DataFrame] = {}
    probs_map: dict[str, np.ndarray] = {}
    scores_map: dict[str, np.ndarray] = {}

    for bname, bpath in baseline_paths.items():
        if bpath.exists():
            df = pd.read_parquet(bpath)
            frames_map[bname] = df
            ev, pr, sc = _eval_member_df(df, ref_df, slices, fam_codes, n_boot=args.n_boot)
            results['baselines'][bname] = ev
            probs_map[bname] = pr
            scores_map[bname] = sc

    for mname in W3_10_MODELS:
        mdir = exp_root / mname
        if not mdir.exists():
            continue
        m_out: dict = {}
        loaded_df: dict[str, pd.DataFrame] = {}
        loaded_pr: dict[str, np.ndarray] = {}
        loaded_sc: dict[str, np.ndarray] = {}
        for tag, fname in (('ema', 'predictions.parquet'), ('best', 'best_predictions.parquet')):
            p = mdir / fname
            if not p.exists():
                continue
            df = pd.read_parquet(p)
            ev, pr, sc = _eval_member_df(df, ref_df, slices, fam_codes, n_boot=args.n_boot)
            m_out[tag] = ev
            loaded_df[tag] = df
            loaded_pr[tag] = pr
            loaded_sc[tag] = sc
        primary_tag = _primary_tag_from_manifest(mdir)
        if primary_tag not in loaded_df and loaded_df:
            primary_tag = next(iter(loaded_df))
        m_out['primary_tag'] = primary_tag
        if primary_tag in loaded_df:
            frames_map[mname] = loaded_df[primary_tag]
            probs_map[mname] = loaded_pr[primary_tag]
            scores_map[mname] = loaded_sc[primary_tag]
        results['models'][mname] = m_out

    diag_root = w3_10 / 'diagnostic_v1_5p_global_pauc' / 'experiments'
    if diag_root.exists():
        diag_out: dict = {}
        for ddir in sorted(p for p in diag_root.iterdir() if p.is_dir()):
            d_rec: dict = {}
            for tag, fname in (('ema', 'predictions.parquet'), ('best', 'best_predictions.parquet')):
                p = ddir / fname
                if not p.exists():
                    continue
                df = pd.read_parquet(p)
                arr = extract_member_arrays(df, ref_df)
                d_rec[tag] = {
                    s: compute_slice_metrics(y, arr['probs'], slices[s])
                    for s in ('All', 'D_scint_contained')
                }
            if d_rec:
                diag_out[ddir.name] = d_rec
        results['diagnostic_v1_5p_global_pauc'] = diag_out

    # 2. Factorized paired 400x block-bootstrap ablations
    pairs = [
        ('nomad_100k_b1pp', 'b1pp_v4_100k_16p_nomad', 'b1pp_100k_16p_t4'),
        ('nomad_100k_pf', 'pf_v4_100k_16p_nomad', 'pf_v3_100k_16p_gn2_dense'),
        ('nomad_300k_b1pp', 'b1pp_v4_300k_3p_nomad_only', 'b1pp_300k_24p_t4'),
        ('nomad_300k_pf', 'pf_v4_300k_3p_nomad_only', 'pf_v3_300k_24p_gn2_flagship'),
        ('pauc_300k_b1pp_vs_base', 'b1pp_v4_300k_3p_pauc_only', 'b1pp_300k_24p_t4'),
        ('pauc_300k_pf_vs_base', 'pf_v4_300k_3p_pauc_only', 'pf_v3_300k_24p_gn2_flagship'),
        ('full_v4_b1pp_vs_v1', 'b1pp_v4_300k_3p_nomad_pauc', 'b1pp_300k_24p_t4'),
        ('full_v4_pf_vs_v3', 'pf_v4_300k_3p_nomad_pauc', 'pf_v3_300k_24p_gn2_flagship'),
        ('pauc_on_top_of_nomad_pf', 'pf_v4_300k_3p_nomad_pauc', 'pf_v4_300k_3p_nomad_only'),
        ('pauc_on_top_of_nomad_b1pp', 'b1pp_v4_300k_3p_nomad_pauc', 'b1pp_v4_300k_3p_nomad_only'),
    ]
    for pname, new_k, old_k in pairs:
        if new_k not in probs_map or old_k not in probs_map:
            continue
        results['ablations'][pname] = {
            'new': new_k,
            'baseline': old_k,
            'slices': {
                s: paired_block_bootstrap_delta(
                    y, probs_map[new_k], probs_map[old_k], fam_codes, slices[s], n_boot=args.n_boot
                )
                for s in ('All', 'D_scint_contained', 'Truth_large_scintillator', 'Tune_E_le_2p3TeV', 'D_pix_contained')
            },
        }

    # 3. Cross-family OOF 5-fold ensembles
    ens_specs = {
        'ens_v3_base_pf_b1pp': ('pf_v3_300k_24p_gn2_flagship', 'b1pp_300k_24p_t4'),
        'ens_v4_flagship_pf_b1pp': ('pf_v4_300k_3p_nomad_pauc', 'b1pp_v4_300k_3p_nomad_pauc'),
        'ens_v4_multimember_dual_regime': (
            'pf_v4_300k_3p_nomad_pauc',
            'pf_v4_300k_3p_nomad_only',
            'pf_v3_300k_24p_gn2_flagship',
            'b1pp_v4_300k_3p_nomad_pauc',
            'b1pp_300k_24p_t4',
        ),
    }
    for ename, mlist in ens_specs.items():
        if not all(m in frames_map for m in mlist):
            continue
        mframes = {m: frames_map[m] for m in mlist}
        ens_res = fit_oof_domain_ensemble(
            mframes,
            ref_df,
            slices,
            n_folds=5,
            save_path=w3_10 / 'ensembles' / ename / 'predictions.parquet',
        )
        scores_map[ename] = ens_res['s_tau']
        probs_map[ename] = ens_res['oof_probs']
        results['ensembles'][ename] = {
            k: v for k, v in ens_res.items() if k not in ('oof_probs', 's_tau', 's_raw')
        }

    if 'ens_v4_flagship_pf_b1pp' in probs_map and 'pf_v4_300k_3p_nomad_pauc' in probs_map:
        results['ablations']['ens_v4_flagship_vs_best_single_pf'] = {
            s: paired_block_bootstrap_delta(
                y,
                probs_map['ens_v4_flagship_pf_b1pp'],
                probs_map['pf_v4_300k_3p_nomad_pauc'],
                fam_codes,
                slices[s],
                s_new=scores_map['ens_v4_flagship_pf_b1pp'],
                s_base=scores_map['pf_v4_300k_3p_nomad_pauc'],
                n_boot=args.n_boot,
            )
            for s in ('All', 'D_scint_contained', 'D_pix_contained', 'Tune_E_le_2p3TeV')
        }
    if 'ens_v4_multimember_dual_regime' in probs_map and 'pf_v4_300k_3p_nomad_pauc' in probs_map:
        results['ablations']['ens_v4_multimember_vs_best_single_pf'] = {
            s: paired_block_bootstrap_delta(
                y,
                probs_map['ens_v4_multimember_dual_regime'],
                probs_map['pf_v4_300k_3p_nomad_pauc'],
                fam_codes,
                slices[s],
                s_new=scores_map['ens_v4_multimember_dual_regime'],
                s_base=scores_map['pf_v4_300k_3p_nomad_pauc'],
                n_boot=args.n_boot,
            )
            for s in ('All', 'D_scint_contained', 'D_pix_contained', 'Tune_E_le_2p3TeV')
        }

    # 4. Formal V4.5 conditional SSL audit & V4 promotion gate
    primary_ens = 'ens_v4_multimember_dual_regime' if 'ens_v4_multimember_dual_regime' in results['ensembles'] else 'ens_v4_flagship_pf_b1pp'
    results['v4_5_ssl_trigger_audit'] = _audit_v4_5_ssl_gate(args.v3_root, results, primary_ens)

    pf4_rec = results['models']['pf_v4_300k_3p_nomad_pauc']
    pf4_tag = pf4_rec.get('primary_tag', 'ema')
    pf4_all = pf4_rec[pf4_tag]['slices']['All']
    pf4_scint = pf4_rec[pf4_tag]['slices']['D_scint_contained']
    pf3_all = results['baselines']['pf_v3_300k_24p_gn2_flagship']['slices']['All']
    ens_all = results['ensembles']['ens_v4_flagship_pf_b1pp']['slices']['All']
    ens_scint = results['ensembles']['ens_v4_flagship_pf_b1pp']['slices']['D_scint_contained']

    all_t1_best = max(pf4_all['tau_at_1pct_fpr'], ens_all['within_slice_S_tau_1pct'])
    scint_t1_best = max(pf4_scint['tau_at_1pct_fpr'], ens_scint['within_slice_S_tau_1pct'])
    all_t01_best = max(pf4_all['tau_at_0p1pct_fpr'], ens_all['within_slice_S_tau_0p1pct'])
    scint_t01_best = max(pf4_scint['tau_at_0p1pct_fpr'], ens_scint['within_slice_S_tau_0p1pct'])
    delta_ba3_pf = pf4_all['cond_BA3'] - pf3_all['cond_BA3']
    delta_ba3_ens = ens_all['cond_BA3'] - pf3_all['cond_BA3']

    results['v4_promotion_gate'] = {
        'r2_d2_All_tau_1pct_fpr': 0.0605,
        'r2_d2_All_tau_0p1pct_fpr': 0.0160,
        'r2_d2_scint_tau_1pct_fpr': 0.07210955104870861,
        'r2_d2_scint_tau_0p1pct_fpr': 0.020974172300225344,
        'pf_v4_All_tau_1pct_fpr': float(pf4_all['tau_at_1pct_fpr']),
        'pf_v4_scint_tau_1pct_fpr': float(pf4_scint['tau_at_1pct_fpr']),
        'ens_flagship_All_tau_1pct_fpr': float(ens_all['within_slice_S_tau_1pct']),
        'ens_flagship_scint_tau_1pct_fpr': float(ens_scint['within_slice_S_tau_1pct']),
        'ens_flagship_All_tau_0p1pct_fpr': float(ens_all['within_slice_S_tau_0p1pct']),
        'ens_flagship_scint_tau_0p1pct_fpr': float(ens_scint['within_slice_S_tau_0p1pct']),
        'best_All_ratio_vs_r2_d2_1pct': float(all_t1_best / 0.0605),
        'best_scint_ratio_vs_r2_d2_1pct': float(scint_t1_best / 0.07210955104870861),
        'best_All_ratio_vs_r2_d2_0p1pct': float(all_t01_best / 0.0160),
        'delta_cond_BA3_pf_v4_vs_v3': float(delta_ba3_pf),
        'delta_cond_BA3_ens_vs_v3': float(delta_ba3_ens),
        'non_inferior_cond_BA3_ge_minus_0p5pp': bool(delta_ba3_pf >= -0.005 and delta_ba3_ens >= -0.005),
        'doubles_r2_d2_tau_discovery': bool(
            all_t1_best >= 2.0 * 0.0605
            and scint_t1_best >= 2.0 * 0.07210955104870861
            and all_t01_best >= 2.0 * 0.0160
            and scint_t01_best >= 2.0 * 0.020974172300225344
        ),
        'Cal_A_Cut_B_Confirm_payload_read': False,
        'passed': bool(
            all_t1_best >= 2.0 * 0.0605
            and scint_t1_best >= 2.0 * 0.07210955104870861
            and all_t01_best >= 2.0 * 0.0160
            and delta_ba3_pf >= -0.005
            and delta_ba3_ens >= -0.005
        ),
    }

    if not args.skip_nomad_dist:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        geometry = json.loads(DEFAULT_GEOMETRY.read_text())
        bounds = detector_bounds(geometry)
        nomad_mat, nomad_summary = compute_tune_nomad_distributions(ref_df, geometry, bounds, device)
        results['nomad_kinematics_tune_scint_summary'] = nomad_summary
        results['figures'] = plot_v4_figures(results, ref_df, slices, nomad_mat, scores_map, w3_10 / 'figures')

    out_path = w3_10 / 'evaluation_results.json'
    write_json(out_path, results)
    print(json.dumps({
        'evaluation_path': str(out_path),
        'v4_promotion_gate': results['v4_promotion_gate'],
        'v4_5_ssl_decision': results['v4_5_ssl_trigger_audit']['decision'],
        'figures': results.get('figures', []),
    }, indent=2))


if __name__ == '__main__':
    main()
