"""Evaluation, 8-slice metrics, 400x CIs, Probe A/B, and figures for PinpointFormer V2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from analysis.flavour.inventory import write_json
from analysis.flavour.s4.stage_inputs import stage_shards
from analysis.flavour.v2.partitions import FAMILY, KEY
from analysis.flavour.v2.r1.metrics import Metrics, three_from_four, three_targets
from analysis.flavour.v3.common.calibration import (
    _tau_efficiency_at_fpr,
    multi_domain_asimov_z0,
    oof_regional_calibration,
)
from analysis.flavour.v3.common.datasets import V3EvalDataset
from analysis.flavour.v3.common.domains import detector_bounds
from analysis.flavour.v3.common.probes import compute_probe_a, compute_probe_b, run_model_with_entropy
from analysis.flavour.v3.v2_pinpointformer.models import PinpointFormer
from analysis.flavour.v3.v2_pinpointformer.train import (
    CachedPFDataset,
    DEFAULT_DATA_ROOT,
    DEFAULT_GEOMETRY,
    build_pf_mapped_split,
    make_pf_loader,
)


DEFAULT_DOMAINS_TUNE = Path(
    '/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-01/tune40k_domains.parquet'
)
DEFAULT_TUNE_LE = Path(
    '/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-01/tune_E_le_2p3TeV_keys.parquet'
)
DEFAULT_HIGHE = Path(
    '/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-01/highE_gt_2p3TeV_non_tau_keys.parquet'
)
DEFAULT_SCREEN = Path(
    '/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R1/data/tune_screen.parquet'
)
DEFAULT_V3_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15')


def load_tune_reference_frame(
    domains_path: Path = DEFAULT_DOMAINS_TUNE,
    le_path: Path = DEFAULT_TUNE_LE,
    he_path: Path = DEFAULT_HIGHE,
    screen_path: Path = DEFAULT_SCREEN,
) -> pd.DataFrame:
    """Load pure Tune40k domain & fixed slice membership flags (indexed by KEY)."""
    dom = pd.read_parquet(domains_path)
    le_df = pd.read_parquet(le_path, columns=KEY + ['energy_nu_GeV'])
    he_df = pd.read_parquet(he_path, columns=KEY)
    sc_df = pd.read_parquet(screen_path, columns=KEY)

    le_idx = pd.MultiIndex.from_frame(le_df[KEY])
    he_idx = pd.MultiIndex.from_frame(he_df[KEY])
    sc_idx = pd.MultiIndex.from_frame(sc_df[KEY])
    dom_idx = pd.MultiIndex.from_frame(dom[KEY])

    dom['is_E_le_2p3TeV'] = dom_idx.isin(le_idx)
    dom['is_highE_non_tau'] = dom_idx.isin(he_idx)
    dom['is_screen'] = dom_idx.isin(sc_idx)
    return dom


def compute_slice_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    mask: np.ndarray | None = None,
    weight: np.ndarray | None = None,
) -> dict:
    """Compute unified T4 and conditional T3 metrics on any boolean slice."""
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        y = np.asarray(labels[m], dtype=np.int64)
        p = np.asarray(probs[m], dtype=np.float64)
        w = None if weight is None else np.asarray(weight[m], dtype=np.float64)
    else:
        y = np.asarray(labels, dtype=np.int64)
        p = np.asarray(probs, dtype=np.float64)
        w = None if weight is None else np.asarray(weight, dtype=np.float64)
    if len(y) == 0:
        return {'count': 0}

    def _opt_float(x) -> float:
        return float('nan') if x is None else float(x)

    p = p / np.maximum(p.sum(axis=-1, keepdims=True), 1e-12)
    rep = Metrics(y, p).report(w)
    pc = rep['per_class']
    non_tau = y != 2
    rec_3 = [_opt_float(pc[0]['recall']), _opt_float(pc[1]['recall']), _opt_float(pc[3]['recall'])]
    out = {
        'count': int(len(y)),
        'nll': float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-8, 1.0)).mean()),
        'BA4': _opt_float(rep['BA']),
        'BA3_t4_diag': float(np.nanmean(rec_3)),
        'recall_e': _opt_float(pc[0]['recall']),
        'recall_mu': _opt_float(pc[1]['recall']),
        'recall_tau': _opt_float(pc[2]['recall']),
        'recall_NC': _opt_float(pc[3]['recall']),
        'auroc_tau': _opt_float(pc[2]['AUROC']),
    }
    if (y == 2).any() and non_tau.any():
        p_tau = np.clip(p[:, 2], 1e-8, 1.0 - 1e-8)
        score = np.log(p_tau / (1.0 - p_tau))
        out['tau_at_1pct_fpr'] = float(_tau_efficiency_at_fpr(score, y, 0.01, w)['tau_eff'])
        out['tau_at_0p1pct_fpr'] = float(_tau_efficiency_at_fpr(score, y, 0.001, w)['tau_eff'])
    if non_tau.any() and len(np.unique(y[non_tau])) == 3:
        y3 = three_targets(y[non_tau])
        p3 = three_from_four(p[non_tau])
        m3 = Metrics(y3, p3).report(None if w is None else w[non_tau])
        pc3 = m3['per_class']
        out['cond_BA3'] = _opt_float(m3['BA'])
        out['cond_auroc_e'] = _opt_float(pc3[0]['AUROC'])
        out['cond_auroc_mu'] = _opt_float(pc3[1]['AUROC'])
        out['cond_auroc_NC'] = _opt_float(pc3[2]['AUROC'])
    else:
        out['cond_BA3'] = float('nan')
    if (y != 2).all():
        pred4 = p.argmax(axis=1)
        out['tau_false_positive_rate'] = float((pred4 == 2).mean())
    return out


def block_bootstrap_slice_cis(
    labels: np.ndarray,
    probs: np.ndarray,
    family_keys: np.ndarray,
    mask: np.ndarray,
    n_boot: int = 400,
    seed: int = 42,
) -> dict:
    """Compute 400x family block-bootstrap 95% CIs for BA4, cond_BA3, and tau@1% FPR."""
    m = np.asarray(mask, dtype=bool)
    y = np.asarray(labels[m], dtype=np.int64)
    p = np.asarray(probs[m], dtype=np.float64)
    fams = np.asarray(family_keys)[m]
    codes, _ = np.unique(fams, return_inverse=True)
    order = np.argsort(codes, kind='mergesort')
    sorted_codes = codes[order]
    _, split_idx = np.unique(sorted_codes, return_index=True)
    groups = [order[s:e] for s, e in zip(split_idx, np.r_[split_idx[1:], len(sorted_codes)])]
    n_fam = len(groups)

    p_norm = p / np.maximum(p.sum(axis=-1, keepdims=True), 1e-12)
    p_tau = np.clip(p_norm[:, 2], 1e-8, 1.0 - 1e-8)
    scores = np.log(p_tau / (1.0 - p_tau))

    rng = np.random.default_rng(seed)
    ba4_v, ba3_v, t1_v = [], [], []
    for _ in range(n_boot):
        pick = rng.integers(0, n_fam, size=n_fam)
        ix = np.concatenate([groups[g] for g in pick])
        yi = y[ix]
        pi = p_norm[ix]
        rep = Metrics(yi, pi).report()
        ba4_v.append(float(rep['BA']))
        nt = yi != 2
        if nt.any() and len(np.unique(yi[nt])) == 3:
            ba3_v.append(float(Metrics(three_targets(yi[nt]), three_from_four(pi[nt])).report()['BA']))
        if (yi == 2).any() and nt.any():
            t1_v.append(float(_tau_efficiency_at_fpr(scores[ix], yi, 0.01)['tau_eff']))

    def _ci(arr: list[float]) -> list[float] | None:
        if not arr:
            return None
        a = np.asarray(arr, dtype=np.float64)
        return [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]

    return {
        'n_families': n_fam,
        'n_boot': n_boot,
        'BA4_ci95': _ci(ba4_v),
        'cond_BA3_ci95': _ci(ba3_v),
        'tau_at_1pct_fpr_ci95': _ci(t1_v),
    }


def evaluate_predictions_frame(
    pred_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    audit_df: pd.DataFrame | None = None,
    charm_probs: np.ndarray | None = None,
    lambda_charm: float = 0.0,
) -> dict:
    """Compute complete 8-slice metrics, non-screen check, OOF regional calibration, and vertex errors."""
    pos = pd.MultiIndex.from_frame(pred_df[KEY]).get_indexer(pd.MultiIndex.from_frame(ref_df[KEY]))
    if (pos < 0).any():
        raise ValueError('predictions parquet missing Tune40k reference keys')
    aligned = pred_df.iloc[pos].reset_index(drop=True)

    y = ref_df['label'].to_numpy(dtype=np.int64)
    probs = aligned[['p_0', 'p_1', 'p_2', 'p_3']].to_numpy(dtype=np.float64)
    logit_cols = [c for c in aligned.columns if c.startswith('logit_')]
    logits = aligned[logit_cols].to_numpy(dtype=np.float64)
    # For 7-way logits, collapse logits to marginal 4-class log-probs before regional T/b scaling
    if logits.shape[1] == 7:
        l4 = np.log(np.clip(probs, 1e-8, 1.0))
    else:
        l4 = logits

    obs_dom = ref_df['obs_domain_id'].to_numpy(dtype=np.int64)
    fams = ref_df['family_hi'].to_numpy(dtype=np.int64) * 1000003 + ref_df['family_lo'].to_numpy(dtype=np.int64)
    _, fam_codes = np.unique(fams, return_inverse=True)
    folds = (fam_codes % 5).astype(np.int64)

    slices = {
        'All': np.ones(len(y), dtype=bool),
        'Tune_non_screen': ~ref_df['is_screen'].to_numpy(dtype=bool),
        'D_pix_contained': obs_dom == 1,
        'D_scint_contained': obs_dom == 2,
        'D_partial_or_edge': obs_dom == 3,
        'Truth_pixel_centered': ref_df['pixel_centered'].to_numpy(dtype=bool),
        'Truth_large_scintillator': ref_df['large_scintillator'].to_numpy(dtype=bool),
        'Tune_E_le_2p3TeV': ref_df['is_E_le_2p3TeV'].to_numpy(dtype=bool),
        'HighE_gt_2p3TeV_non_tau': ref_df['is_highE_non_tau'].to_numpy(dtype=bool),
    }

    slice_metrics = {sname: compute_slice_metrics(y, probs, mask) for sname, mask in slices.items()}

    # 400x family block-bootstrap CIs on core decision slices
    cis = {}
    for sname in ('All', 'Truth_pixel_centered', 'Truth_large_scintillator', 'Tune_E_le_2p3TeV'):
        cis[sname] = block_bootstrap_slice_cis(y, probs, fam_codes, slices[sname], n_boot=400, seed=42)

    # 5-fold OOF regional calibration and Asimov Z0
    cal_probs, s_tau, oof_folds = oof_regional_calibration(
        l4, y, obs_dom, folds, n_folds=5, charm_probs=charm_probs, lambda_charm=lambda_charm
    )
    p_tau_raw = np.clip(probs[:, 2], 1e-8, 1.0 - 1e-8)
    s_raw = np.log(p_tau_raw / (1.0 - p_tau_raw))

    z0_raw = multi_domain_asimov_z0(s_raw, y, obs_dom)
    z0_reg = multi_domain_asimov_z0(s_tau, y, obs_dom)
    reg_summary = {
        'raw_tau_1pct_fpr': float(_tau_efficiency_at_fpr(s_raw, y, 0.01)['tau_eff']),
        'raw_tau_0p1pct_fpr': float(_tau_efficiency_at_fpr(s_raw, y, 0.001)['tau_eff']),
        'reg_tau_1pct_fpr': float(_tau_efficiency_at_fpr(s_tau, y, 0.01)['tau_eff']),
        'reg_tau_0p1pct_fpr': float(_tau_efficiency_at_fpr(s_tau, y, 0.001)['tau_eff']),
        'raw_pixel_centered_tau_1pct': float(_tau_efficiency_at_fpr(s_raw[slices['Truth_pixel_centered']], y[slices['Truth_pixel_centered']], 0.01)['tau_eff']),
        'reg_pixel_centered_tau_1pct': float(_tau_efficiency_at_fpr(s_tau[slices['Truth_pixel_centered']], y[slices['Truth_pixel_centered']], 0.01)['tau_eff']),
        'raw_E_le_2p3TeV_tau_1pct': float(_tau_efficiency_at_fpr(s_raw[slices['Tune_E_le_2p3TeV']], y[slices['Tune_E_le_2p3TeV']], 0.01)['tau_eff']),
        'reg_E_le_2p3TeV_tau_1pct': float(_tau_efficiency_at_fpr(s_tau[slices['Tune_E_le_2p3TeV']], y[slices['Tune_E_le_2p3TeV']], 0.01)['tau_eff']),
        'asimov_z0_raw': float(z0_raw),
        'asimov_z0_reg': float(z0_reg),
        'oof_calibrated_All_BA4': float(Metrics(y, cal_probs).report()['BA']),
        'mean_domain_params': {
            dname: {
                'T': float(np.mean([f['domains'][dname]['T'] for f in oof_folds if dname in f['domains']])),
                'bias': np.mean([f['domains'][dname]['bias'] for f in oof_folds if dname in f['domains']], axis=0).tolist(),
            }
            for dname in ('D_empty', 'D_pix_contained', 'D_scint_contained', 'D_partial_or_edge')
            if any(dname in f['domains'] for f in oof_folds)
        },
    }

    # Offline vertex residual evaluation against Tune40k audit (never used at model runtime)
    vertex_metrics = {}
    if audit_df is not None and 'v_hat_x_mm' in aligned.columns:
        apos = pd.MultiIndex.from_frame(audit_df[KEY]).get_indexer(pd.MultiIndex.from_frame(ref_df[KEY]))
        aud = audit_df.iloc[apos].reset_index(drop=True)
        valid_v = aud['vertex_valid'].to_numpy(dtype=int) > 0
        dx = aligned['v_hat_x_mm'].to_numpy(dtype=np.float64) - aud['vertex_x_mm'].to_numpy(dtype=np.float64)
        dy = aligned['v_hat_y_mm'].to_numpy(dtype=np.float64) - aud['vertex_y_mm'].to_numpy(dtype=np.float64)
        dz = aligned['v_hat_z_mm'].to_numpy(dtype=np.float64) - aud['vertex_z_mm'].to_numpy(dtype=np.float64)
        for sname in ('All', 'D_pix_contained', 'Truth_pixel_centered', 'D_scint_contained'):
            mv = valid_v & slices[sname]
            if mv.any():
                r3 = np.sqrt(dx[mv] ** 2 + dy[mv] ** 2 + dz[mv] ** 2)
                vertex_metrics[sname] = {
                    'count': int(mv.sum()),
                    'mae_x_mm': float(np.mean(np.abs(dx[mv]))),
                    'mae_y_mm': float(np.mean(np.abs(dy[mv]))),
                    'mae_z_mm': float(np.mean(np.abs(dz[mv]))),
                    'median_x_mm': float(np.median(np.abs(dx[mv]))),
                    'median_y_mm': float(np.median(np.abs(dy[mv]))),
                    'median_z_mm': float(np.median(np.abs(dz[mv]))),
                    'median_3d_mm': float(np.median(r3)),
                    'rmse_3d_mm': float(np.sqrt(np.mean(r3 ** 2))),
                    'frac_abs_dz_le_10mm': float((np.abs(dz[mv]) <= 10.0).mean()),
                }

    return {
        'slices': slice_metrics,
        'bootstrap_ci95': cis,
        'regional_calibration': reg_summary,
        'vertex_metrics': vertex_metrics,
    }


def _evaluate_probe_a_b_for_v2(
    w3_07_root: Path,
    ref_df: pd.DataFrame,
    geometry: dict,
    bounds: dict,
    device: torch.device,
    audit_df: pd.DataFrame | None = None,
) -> tuple[dict, dict]:
    """Run Probe A on anchored vs no_anchor and Probe B (micron vs 1mm) on GPU/CPU."""
    exp_dir = w3_07_root / 'experiments'
    models_to_probe = ('pf_v2_100k_16p_anchored', 'pf_v2_100k_16p_no_anchor')
    probe_a_out = {}

    y = ref_df['label'].to_numpy(dtype=np.int64)
    le_mask = ref_df['is_E_le_2p3TeV'].to_numpy(dtype=bool)
    pix_cent = ref_df['pixel_centered'].to_numpy(dtype=bool)
    fams = ref_df['family_hi'].to_numpy(dtype=np.int64) * 1000003 + ref_df['family_lo'].to_numpy(dtype=np.int64)

    parent_scratch = os.environ.get('PINPOINT_V3_SCRATCH', '/tmp')
    Path(parent_scratch).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='pf-v2-eval-', dir=parent_scratch) as tmp_dir:
        tmp = Path(tmp_dir)
        shard_paths = sorted(p for p in (DEFAULT_DATA_ROOT / 'shards').iterdir() if (p / 'observations.parquet').exists())
        st_rep = stage_shards(shard_paths, tmp / 'native', workers=4)
        old_stage = os.environ.get('PINPOINT_V3_SHARED_STAGE')
        os.environ['PINPOINT_V3_SHARED_STAGE'] = st_rep['local_shard_root']
        try:
            nat_pkl = tmp / 'tune_native.pkl'
            q1m_pkl = tmp / 'tune_q1mm.pkl'
            build_pf_mapped_split(DEFAULT_DATA_ROOT, 'validation', tmp / 'nat_items', nat_pkl, tmp, 4, {'quantize_xy_mm': None})
            build_pf_mapped_split(DEFAULT_DATA_ROOT, 'validation', tmp / 'q1m_items', q1m_pkl, tmp, 4, {'quantize_xy_mm': 1.0})
        finally:
            if old_stage is None:
                os.environ.pop('PINPOINT_V3_SHARED_STAGE', None)
            else:
                os.environ['PINPOINT_V3_SHARED_STAGE'] = old_stage

        import pickle
        with nat_pkl.open('rb') as f:
            nat_blob = pickle.load(f)
        with q1m_pkl.open('rb') as f:
            q1m_blob = pickle.load(f)

        nat_meta = nat_blob['metadata']
        pos = pd.MultiIndex.from_frame(nat_meta[KEY]).get_indexer(pd.MultiIndex.from_frame(ref_df[KEY]))
        nat_ds = CachedPFDataset(nat_blob['items_dir'], nat_meta, indices=pos)
        q1m_ds = CachedPFDataset(q1m_blob['items_dir'], q1m_blob['metadata'], indices=pos)
        nat_loader = make_pf_loader(nat_ds, batch_size=96, shuffle=False, workers=2, y_scint_center=bounds['y_scint_center'], n_bars=bounds['n_bars'])
        q1m_loader = make_pf_loader(q1m_ds, batch_size=96, shuffle=False, workers=2, y_scint_center=bounds['y_scint_center'], n_bars=bounds['n_bars'])

        nat_probs_anchored = None
        inf_q1m_probs = None
        for mname in models_to_probe:
            man = json.loads((exp_dir / mname / 'manifest.json').read_text())
            cfg = man['config']
            ckpt_p = exp_dir / mname / 'ema.pt'
            if not ckpt_p.exists():
                ckpt_p = exp_dir / mname / 'best.pt'
            state = torch.load(ckpt_p, map_location=device, weights_only=False)
            model = PinpointFormer(geometry, cfg).to(device)
            model.load_state_dict(state['model'])
            res = run_model_with_entropy(model, nat_loader, device, use_bf16=(device.type == 'cuda' and torch.cuda.get_device_capability(0)[0] >= 8))
            probe_a_out[mname] = {
                'All_Tune40k': compute_probe_a(
                    res['probs'], y, res['raw_pixel_hits'], res['mean_entropy'],
                    learned_scales=res['learned_scales'], layer_entropy=res['layer_entropy'],
                ),
                'Tune_E_le_2p3TeV': compute_probe_a(
                    res['probs'], y, res['raw_pixel_hits'], res['mean_entropy'],
                    learned_scales=res['learned_scales'], layer_entropy=res['layer_entropy'],
                    subset_mask=le_mask,
                ),
            }
            if mname == 'pf_v2_100k_16p_anchored':
                nat_probs_anchored = res['probs']
                res_q1m = run_model_with_entropy(model, q1m_loader, device, use_bf16=(device.type == 'cuda' and torch.cuda.get_device_capability(0)[0] >= 8))
                inf_q1m_probs = res_q1m['probs']

        # Load matched from-scratch q1mm control predictions
        q_df = pd.read_parquet(exp_dir / 'pf_v2_100k_16p_anchored_q1mm' / 'predictions.parquet')
        qpos = pd.MultiIndex.from_frame(q_df[KEY]).get_indexer(pd.MultiIndex.from_frame(ref_df[KEY]))
        scratch_q1m_probs = q_df.iloc[qpos][['p_0', 'p_1', 'p_2', 'p_3']].to_numpy(dtype=np.float64)

        assert nat_probs_anchored is not None and inf_q1m_probs is not None
        probe_b_out = {
            'inference_quantized_1mm': compute_probe_b(
                y, nat_probs_anchored, inf_q1m_probs, pix_cent, family_keys=fams, n_boot=400
            ),
            'matched_from_scratch_q1mm': compute_probe_b(
                y, nat_probs_anchored, scratch_q1m_probs, pix_cent, family_keys=fams, n_boot=400
            ),
        }
        if audit_df is not None:
            apos = pd.MultiIndex.from_frame(audit_df[KEY]).get_indexer(pd.MultiIndex.from_frame(ref_df[KEY]))
            aud_aligned = audit_df.iloc[apos].reset_index(drop=True)
            pinpoint_6layer_mask = (
                pix_cent
                & (aud_aligned['vertex_valid'].to_numpy(dtype=int) > 0)
                & (aud_aligned['vertex_z_mm'].to_numpy(dtype=np.float64) <= -2815.0)
            )
            if pinpoint_6layer_mask.any():
                probe_b_out['pinpoint_6layer_vertex_block'] = compute_probe_b(
                    y, nat_probs_anchored, inf_q1m_probs, pinpoint_6layer_mask, family_keys=fams, n_boot=400
                )
                probe_b_out['pinpoint_6layer_matched_scratch'] = compute_probe_b(
                    y, nat_probs_anchored, scratch_q1m_probs, pinpoint_6layer_mask, family_keys=fams, n_boot=400
                )
    return probe_a_out, probe_b_out


def plot_v2_figures(results: dict, ref_df: pd.DataFrame, anchored_pred: pd.DataFrame, audit_df: pd.DataFrame | None, fig_dir: Path) -> list[str]:
    """Generate English publication figures for Phase V2 (W3-07)."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    # 1. Ablation overview bar chart
    models = [
        ('pf_v2_100k_16p_no_anchor', 'PF_V2 No-Anchor\n(100k, 16p)'),
        ('pf_v2_100k_16p_anchored_q1mm', 'PF_V2 Anchored 1mm\n(100k, 16p)'),
        ('pf_v2_100k_16p_anchored', 'PF_V2 Anchored Micron\n(100k, 16p)'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), dpi=300)
    ba4_pc = [results['models'][m]['ema']['slices']['Truth_pixel_centered']['BA4'] * 100.0 for m, _ in models]
    ba3_ls = [results['models'][m]['ema']['slices']['Truth_large_scintillator']['cond_BA3'] * 100.0 for m, _ in models]
    tau_pc = [results['models'][m]['ema']['slices']['Truth_pixel_centered']['tau_at_1pct_fpr'] * 100.0 for m, _ in models]
    x = np.arange(len(models))
    labels = [lbl for _, lbl in models]
    colors = ['#64748b', '#f59e0b', '#2563eb']

    for ax, vals, ref_val, ref_lbl, title in (
        (axes[0], ba4_pc, 55.92, 'B0 HGB (55.92%)', 'Truth Pixel-Centered BA4 (%)'),
        (axes[1], ba3_ls, 74.11, 'V1 B1++ (74.11%)', 'Truth Large-Scint Cond BA3 (%)'),
        (axes[2], tau_pc, 4.36, 'B0 HGB (4.36%)', 'Truth Pixel-Centered tau@1% FPR (%)'),
    ):
        bars = ax.bar(x, vals, color=colors, width=0.58)
        ax.axhline(ref_val, color='#dc2626', linestyle='--', linewidth=1.4, label=ref_lbl)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2.0, v + 0.6, f'{v:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylim(0, max(max(vals), ref_val) * 1.18)
        ax.set_title(title, fontsize=10.5)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(axis='y', alpha=0.25)

    fig.suptitle('Phase V2 PinpointFormer Single-Factor Ablations (Tune40k EMA)', fontsize=12)
    fig.tight_layout()
    p1 = fig_dir / 'v2_ablation_overview.png'
    fig.savefig(p1, dpi=300)
    plt.close(fig)
    saved.append(str(p1))

    # 2. Probe A: occupancy bins + normalized attention entropy
    pa = results['probe_a']
    bins = ['0', '1-100', '100-1000', '1000-10000', '10000-100000']
    bx = np.arange(len(bins))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.5), dpi=300)
    for mname, lbl, col, mk in (
        ('pf_v2_100k_16p_anchored', 'PF_V2 Anchored', '#2563eb', 'o'),
        ('pf_v2_100k_16p_no_anchor', 'PF_V2 No-Anchor', '#64748b', 's'),
    ):
        effs = []
        ents = []
        for bname in bins:
            bd = pa[mname]['All_Tune40k']['bins'].get(bname, {})
            ev = bd.get('tau_eff_at_global_1pct_fpr')
            effs.append(np.nan if ev is None else ev * 100.0)
            ents.append(bd.get('entropy_mean', np.nan))
        ax1.plot(bx, effs, marker=mk, linewidth=2.0, color=col, label=lbl)
        ax2.plot(bx, ents, marker=mk, linewidth=2.0, color=col, label=lbl)

    ax1.scatter([3], [0.29], color='#dc2626', s=90, zorder=5, marker='X', label='S6 MAE3 [1000,10000] (0.29%)')
    ax1.set_xticks(bx)
    ax1.set_xticklabels(bins, rotation=15, fontsize=8.5)
    ax1.set_ylabel('tau Efficiency at Global 1% FPR (%)')
    ax1.set_xlabel('Raw Pixel Hits Occupancy Bin')
    ax1.set_title('Probe A: High-Occupancy tau Efficiency')
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8)

    ax2.set_xticks(bx)
    ax2.set_xticklabels(bins, rotation=15, fontsize=8.5)
    ax2.set_ylabel('Normalized Attention Entropy H_norm')
    ax2.set_xlabel('Raw Pixel Hits Occupancy Bin')
    ax2.set_title('ScaledQKNormAttention Entropy vs Occupancy')
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8)

    fig.suptitle('Probe A: ScaledQKNormAttention Prevents High-Occupancy Collapse', fontsize=11.5)
    fig.tight_layout()
    p2 = fig_dir / 'v2_probe_a_occupancy_entropy.png'
    fig.savefig(p2, dpi=300)
    plt.close(fig)
    saved.append(str(p2))

    # 3. Probe B: micron vs 1mm + vertex error summary
    pb = results['probe_b']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.4), dpi=300)
    cats = ['Native\nMicron', '1.0 mm\n(Inference)', '1.0 mm\n(Trained Control)']
    cx = np.arange(3)
    ba4_vals = [
        pb['inference_quantized_1mm']['native']['BA4'] * 100.0,
        pb['inference_quantized_1mm']['q1mm']['BA4'] * 100.0,
        pb['matched_from_scratch_q1mm']['q1mm']['BA4'] * 100.0,
    ]
    tau_vals = [
        pb['inference_quantized_1mm']['native']['tau_at_1pct_fpr'] * 100.0,
        pb['inference_quantized_1mm']['q1mm']['tau_at_1pct_fpr'] * 100.0,
        pb['matched_from_scratch_q1mm']['q1mm']['tau_at_1pct_fpr'] * 100.0,
    ]
    w = 0.34
    b1 = ax1.bar(cx - w / 2, ba4_vals, width=w, color='#2563eb', label='BA4 (%)')
    b2 = ax1.bar(cx + w / 2, tau_vals, width=w, color='#059669', label='tau@1% FPR (%)')
    for b in list(b1) + list(b2):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f'{b.get_height():.1f}%', ha='center', va='bottom', fontsize=8)
    ax1.set_xticks(cx)
    ax1.set_xticklabels(cats, fontsize=8.5)
    ax1.set_ylabel('Percentage (%)')
    ax1.set_title('Probe B: Micron vs 1.0 mm Pixel Resolution (Truth Pixel-Centered)')
    ax1.set_ylim(0, max(max(ba4_vals), max(tau_vals)) * 1.18)
    ax1.grid(axis='y', alpha=0.25)
    ax1.legend(fontsize=8.5)

    if audit_df is not None and 'v_hat_z_mm' in anchored_pred.columns:
        pos = pd.MultiIndex.from_frame(anchored_pred[KEY]).get_indexer(pd.MultiIndex.from_frame(ref_df[KEY]))
        al = anchored_pred.iloc[pos].reset_index(drop=True)
        apos = pd.MultiIndex.from_frame(audit_df[KEY]).get_indexer(pd.MultiIndex.from_frame(ref_df[KEY]))
        aud = audit_df.iloc[apos].reset_index(drop=True)
        m_v = (aud['vertex_valid'].to_numpy(dtype=int) > 0) & ref_df['pixel_centered'].to_numpy(dtype=bool)
        dz = np.abs(al['v_hat_z_mm'].to_numpy()[m_v] - aud['vertex_z_mm'].to_numpy()[m_v])
        dr = np.hypot(
            al['v_hat_x_mm'].to_numpy()[m_v] - aud['vertex_x_mm'].to_numpy()[m_v],
            al['v_hat_y_mm'].to_numpy()[m_v] - aud['vertex_y_mm'].to_numpy()[m_v],
        )
        ax2.hist(np.clip(dz, 0, 100), bins=40, alpha=0.75, color='#2563eb', label=f'|dz| median={np.median(dz):.1f} mm')
        ax2.hist(np.clip(dr, 0, 100), bins=40, alpha=0.65, color='#f59e0b', label=f'd_xy median={np.median(dr):.1f} mm')
        ax2.set_xlabel('Primary Vertex Prediction Error (mm, clipped at 100 mm)')
        ax2.set_ylabel('Events')
        ax2.set_title('Stage-1 Vertex Anchor Accuracy (Truth Pixel-Centered)')
        ax2.legend(fontsize=8.5)
        ax2.grid(alpha=0.25)
    else:
        scales = pa['pf_v2_100k_16p_anchored']['All_Tune40k']['learned_scales']
        vals = [v for vs in scales['per_layer'].values() for v in vs]
        ax2.hist(vals, bins=20, color='#2563eb', alpha=0.85)
        ax2.axvline(4.90, color='#dc2626', linestyle='--', linewidth=1.4, label='R2/S6 Collapse Max (4.90)')
        ax2.set_xlabel('Learned Head Scale gamma_h')
        ax2.set_ylabel('Attention Heads')
        ax2.set_title('ScaledQKNormAttention Learned Temperatures')
        ax2.legend(fontsize=8.5)
        ax2.grid(alpha=0.25)

    fig.tight_layout()
    p3 = fig_dir / 'v2_probe_b_micron_vs_1mm.png'
    fig.savefig(p3, dpi=300)
    plt.close(fig)
    saved.append(str(p3))
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_V3_ROOT / 'W3-07')
    parser.add_argument('--offline-audit-parquet', type=Path, default=DEFAULT_V3_ROOT / 'W3-08' / 'tune40k_offline_audit' / 'tune_event_audit.parquet')
    parser.add_argument('--skip-probes', action='store_true')
    args = parser.parse_args()

    ref_df = load_tune_reference_frame()
    audit_df = pd.read_parquet(args.offline_audit_parquet) if args.offline_audit_parquet.exists() else None
    exp_root = args.root / 'experiments'
    model_names = [
        'pf_v2_100k_16p_anchored',
        'pf_v2_100k_16p_no_anchor',
        'pf_v2_100k_16p_anchored_q1mm',
    ]

    results: dict = {'models': {}, 'v2_promotion_gate': {}}
    anchored_pred_df = None
    for mname in model_names:
        mdir = exp_root / mname
        if not mdir.exists():
            continue
        m_out: dict = {}
        for tag, fname in (('ema', 'predictions.parquet'), ('best', 'best_predictions.parquet'), ('exposure', 'exposure_predictions.parquet')):
            p = mdir / fname
            if p.exists():
                df = pd.read_parquet(p)
                if mname == 'pf_v2_100k_16p_anchored' and tag == 'ema':
                    anchored_pred_df = df
                m_out[tag] = evaluate_predictions_frame(df, ref_df, audit_df=audit_df)
        results['models'][mname] = m_out

    if not args.skip_probes:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        geometry = json.loads(DEFAULT_GEOMETRY.read_text())
        bounds = detector_bounds(geometry)
        probe_a, probe_b = _evaluate_probe_a_b_for_v2(
            args.root, ref_df, geometry, bounds, device, audit_df=audit_df
        )
        results['probe_a'] = probe_a
        results['probe_b'] = probe_b

        # Evaluate formal V2 Promotion Gate
        ba4_anch = results['models']['pf_v2_100k_16p_anchored']['ema']['slices']['Truth_pixel_centered']['BA4']
        ba4_no = results['models']['pf_v2_100k_16p_no_anchor']['ema']['slices']['Truth_pixel_centered']['BA4']
        pa_anch = probe_a['pf_v2_100k_16p_anchored']['All_Tune40k']
        pb_inf = probe_b['inference_quantized_1mm']
        pb_scr = probe_b['matched_from_scratch_q1mm']
        results['v2_promotion_gate'] = {
            'anchored_pixel_centered_BA4': ba4_anch,
            'no_anchor_pixel_centered_BA4': ba4_no,
            'b0_hgb_reference_BA4': 0.5592,
            'beats_no_anchor': bool(ba4_anch > ba4_no),
            'beats_b0_hgb': bool(ba4_anch > 0.5592),
            'probe_a_collapse_eliminated': bool(pa_anch['collapse_eliminated']),
            'probe_a_learned_scales_gt_4p90': bool(pa_anch['learned_scales']['n_heads_gt_4p90'] > 0),
            'probe_b_inference_passes': bool(pb_inf['passes_probe_b']),
            'probe_b_matched_scratch_passes': bool(pb_scr['passes_probe_b']),
            'passed': bool(
                (ba4_anch > ba4_no)
                and (ba4_anch > 0.5592)
                and pa_anch['collapse_eliminated']
                and (pb_inf['passes_probe_b'] or pb_scr['passes_probe_b'])
            ),
        }
        assert anchored_pred_df is not None
        results['figures'] = plot_v2_figures(results, ref_df, anchored_pred_df, audit_df, args.root / 'figures')

    write_json(args.root / 'evaluation_results.json', results)
    print(json.dumps({
        'v2_promotion_gate': results['v2_promotion_gate'],
        'figures': results.get('figures', []),
    }, indent=2))


if __name__ == '__main__':
    main()
