"""Full V3 evaluation: 8-slice CIs, Charm FPR rejection, Cluster Origin confusion/PR, Probe A/B."""

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
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
import torch

from analysis.flavour.inventory import write_json
from analysis.flavour.s4.stage_inputs import stage_shards
from analysis.flavour.v2.partitions import KEY
from analysis.flavour.v3.common.calibration import _tau_efficiency_at_fpr, oof_regional_calibration
from analysis.flavour.v3.common.domains import detector_bounds
from analysis.flavour.v3.common.probes import compute_probe_a, compute_probe_b, run_model_with_entropy
from analysis.flavour.v3.v2_pinpointformer.evaluate import (
    DEFAULT_DATA_ROOT,
    DEFAULT_GEOMETRY,
    DEFAULT_V3_ROOT,
    evaluate_predictions_frame,
    load_tune_reference_frame,
)
from analysis.flavour.v3.v2_pinpointformer.train import (
    CachedPFDataset,
    build_pf_mapped_split,
    make_pf_loader,
    move_batch_to_device,
)
from analysis.flavour.v3.v3_multitask.dense_heads import CLUSTER_ORIGIN_NAMES
from analysis.flavour.v3.v3_multitask.models import PinpointFormerMultitask


def _tau_score_from_probs(probs: np.ndarray) -> np.ndarray:
    p = probs / np.maximum(probs.sum(axis=-1, keepdims=True), 1e-12)
    pt = np.clip(p[:, 2], 1e-8, 1.0 - 1e-8)
    return np.log(pt / (1.0 - pt))


def compute_charm_rejection_audit(
    ref_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    model_preds: dict[str, pd.DataFrame],
    lambda_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0),
) -> dict:
    """Measure Charm background contamination in the 1% / 0.5% FPR tau tail across V1/V2/V3."""
    apos = pd.MultiIndex.from_frame(audit_df[KEY]).get_indexer(pd.MultiIndex.from_frame(ref_df[KEY]))
    if (apos < 0).any():
        raise ValueError('Tune40k offline audit missing Tune40k reference keys')
    aud = audit_df.iloc[apos].reset_index(drop=True)

    y = ref_df['label'].to_numpy(dtype=np.int64)
    pix_c = ref_df['pixel_centered'].to_numpy(dtype=bool)
    non_tau = y != 2
    charm_valid = aud['charm_valid'].to_numpy(dtype=int) > 0
    charm_pos = (aud['charm_flag'].to_numpy(dtype=int) == 1) & charm_valid
    non_tau_charm = non_tau & charm_pos
    non_tau_nocharm = non_tau & charm_valid & (~charm_pos)

    obs_dom = ref_df['obs_domain_id'].to_numpy(dtype=np.int64)
    fams = ref_df['family_hi'].to_numpy(dtype=np.int64) * 1000003 + ref_df['family_lo'].to_numpy(dtype=np.int64)
    _, fam_codes = np.unique(fams, return_inverse=True)
    folds = (fam_codes % 5).astype(np.int64)

    out: dict = {
        'tune_counts': {
            'non_tau_total': int(non_tau.sum()),
            'non_tau_charm': int(non_tau_charm.sum()),
            'pixel_centered_non_tau_charm': int((pix_c & non_tau_charm).sum()),
        },
        'models': {},
        'lambda_charm_scan': {},
    }

    for name, df in model_preds.items():
        pos = pd.MultiIndex.from_frame(df[KEY]).get_indexer(pd.MultiIndex.from_frame(ref_df[KEY]))
        if (pos < 0).any():
            continue
        al = df.iloc[pos].reset_index(drop=True)
        p_cols = ['p_0', 'p_1', 'p_2', 'p_3'] if 'p_0' in al.columns else ['p0', 'p1', 'p2', 'p3']
        probs = al[p_cols].to_numpy(dtype=np.float64)
        score = _tau_score_from_probs(probs)

        def _tail_stats(sc: np.ndarray, mask_slice: np.ndarray | None = None) -> dict:
            m = np.ones(len(y), dtype=bool) if mask_slice is None else mask_slice
            nt_m = m & non_tau
            thr_1p = float(np.quantile(sc[nt_m], 0.99))
            thr_05p = float(np.quantile(sc[nt_m], 0.995))
            fp_1p = nt_m & (sc >= thr_1p)
            fp_05p = nt_m & (sc >= thr_05p)
            chm_m = m & non_tau_charm
            noc_m = m & non_tau_nocharm
            return {
                'tau_eff_at_1pct_fpr': float(_tau_efficiency_at_fpr(sc[m], y[m], 0.01)['tau_eff']),
                'tau_eff_at_0p5pct_fpr': float(_tau_efficiency_at_fpr(sc[m], y[m], 0.005)['tau_eff']),
                'charm_fpr_at_global_1pct': float((sc[chm_m] >= thr_1p).mean()) if chm_m.any() else None,
                'noncharm_fpr_at_global_1pct': float((sc[noc_m] >= thr_1p).mean()) if noc_m.any() else None,
                'charm_share_of_1pct_fp': float((fp_1p & charm_pos).sum() / max(1, int(fp_1p.sum()))),
                'charm_fpr_at_global_0p5pct': float((sc[chm_m] >= thr_05p).mean()) if chm_m.any() else None,
                'charm_share_of_0p5pct_fp': float((fp_05p & charm_pos).sum() / max(1, int(fp_05p.sum()))),
                'n_charm_fp_at_1pct': int((fp_1p & charm_pos).sum()),
            }

        entry: dict = {
            'All_Tune40k': _tail_stats(score, None),
            'Truth_pixel_centered': _tail_stats(score, pix_c),
        }
        if 'charm_prob' in al.columns:
            ch_p = al['charm_prob'].to_numpy(dtype=np.float64)
            entry['charm_auroc_non_tau'] = float(roc_auc_score(charm_pos[non_tau & charm_valid].astype(int), ch_p[non_tau & charm_valid]))
            entry['charm_auroc_pixel_centered_non_tau'] = float(
                roc_auc_score(charm_pos[pix_c & non_tau & charm_valid].astype(int), ch_p[pix_c & non_tau & charm_valid])
            )
            l4 = np.log(np.clip(probs, 1e-8, 1.0))
            scan = {}
            best_lam = 0.0
            best_eff = -1.0
            for lam in lambda_grid:
                _, st, _ = oof_regional_calibration(l4, y, obs_dom, folds, n_folds=5, charm_probs=ch_p, lambda_charm=float(lam))
                s_all = _tail_stats(st, None)
                s_pc = _tail_stats(st, pix_c)
                scan[str(lam)] = {'All_Tune40k': s_all, 'Truth_pixel_centered': s_pc}
                if s_pc['tau_eff_at_1pct_fpr'] > best_eff:
                    best_eff = s_pc['tau_eff_at_1pct_fpr']
                    best_lam = float(lam)
            out['lambda_charm_scan'][name] = {'best_lambda_charm': best_lam, 'grid': scan}
            entry['best_lambda_charm'] = best_lam
            entry['with_best_lambda_charm'] = scan[str(best_lam)]
        out['models'][name] = entry
    return out


def compute_origin_confusion_and_pr(
    probs_or: np.ndarray,
    true_or: np.ndarray,
) -> dict:
    """Compute 5x5 confusion matrix (counts & row-norm) and per-class precision/recall/F1/PR-AUC."""
    valid = (true_or >= 0) & (true_or < 5)
    y = true_or[valid].astype(np.int64)
    p = probs_or[valid].astype(np.float64)
    pred = p.argmax(axis=-1)
    cm = confusion_matrix(y, pred, labels=[0, 1, 2, 3, 4])
    row_sums = np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cm_norm = cm.astype(np.float64) / row_sums

    per_class = {}
    for c, cname in enumerate(CLUSTER_ORIGIN_NAMES):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        prec = float(tp / max(tp + fp, 1))
        rec = float(tp / max(tp + fn, 1))
        f1 = float(2.0 * prec * rec / max(prec + rec, 1e-12))
        y_bin = (y == c).astype(int)
        ap = float(average_precision_score(y_bin, p[:, c])) if y_bin.any() else float('nan')
        per_class[cname] = {
            'support': int(cm[c, :].sum()),
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'pr_auc': ap,
        }

    return {
        'n_clusters': int(len(y)),
        'labels': list(CLUSTER_ORIGIN_NAMES),
        'confusion_counts': cm.tolist(),
        'confusion_row_norm': cm_norm.tolist(),
        'macro_f1': float(np.mean([v['f1'] for v in per_class.values()])),
        'macro_pr_auc': float(np.nanmean([v['pr_auc'] for v in per_class.values()])),
        'per_class': per_class,
    }


def _evaluate_flagship_probes_and_origin(
    w3_09_root: Path,
    ref_df: pd.DataFrame,
    audit_root: Path,
    geometry: dict,
    bounds: dict,
    device: torch.device,
) -> tuple[dict, dict, dict]:
    """Run Cluster Origin evaluation, Probe A, and Probe B on flagship PinpointFormer-GN2."""
    flag_dir = w3_09_root / 'experiments' / 'pf_v3_300k_24p_gn2_flagship'
    man = json.loads((flag_dir / 'manifest.json').read_text())
    cfg = man['config']
    ckpt_p = flag_dir / 'ema.pt'
    if not ckpt_p.exists():
        ckpt_p = flag_dir / 'best.pt'
    state = torch.load(ckpt_p, map_location=device, weights_only=False)
    model = PinpointFormerMultitask(geometry, cfg).to(device)
    model.load_state_dict(state['model'])
    model.eval()
    use_bf16 = device.type == 'cuda' and torch.cuda.get_device_capability(0)[0] >= 8

    y = ref_df['label'].to_numpy(dtype=np.int64)
    le_mask = ref_df['is_E_le_2p3TeV'].to_numpy(dtype=bool)
    pix_cent = ref_df['pixel_centered'].to_numpy(dtype=bool)
    fams = ref_df['family_hi'].to_numpy(dtype=np.int64) * 1000003 + ref_df['family_lo'].to_numpy(dtype=np.int64)

    parent_scratch = os.environ.get('PINPOINT_V3_SCRATCH', '/tmp')
    Path(parent_scratch).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='pf-v3-eval-', dir=parent_scratch) as tmp_dir:
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

        res_nat = run_model_with_entropy(model, nat_loader, device, use_bf16=use_bf16)
        res_q1m = run_model_with_entropy(model, q1m_loader, device, use_bf16=use_bf16)

        probe_a = {
            'All_Tune40k': compute_probe_a(
                res_nat['probs'], y, res_nat['raw_pixel_hits'], res_nat['mean_entropy'],
                learned_scales=res_nat['learned_scales'], layer_entropy=res_nat['layer_entropy'],
            ),
            'Tune_E_le_2p3TeV': compute_probe_a(
                res_nat['probs'], y, res_nat['raw_pixel_hits'], res_nat['mean_entropy'],
                learned_scales=res_nat['learned_scales'], layer_entropy=res_nat['layer_entropy'],
                subset_mask=le_mask,
            ),
        }
        probe_b = compute_probe_b(
            y, res_nat['probs'], res_q1m['probs'], pix_cent, family_keys=fams, n_boot=400
        )

        # Evaluate 5-class Cluster Origin confusion & PR on near-vertex clusters against offline cluster_audit
        ev_aud = pd.read_parquet(audit_root / 'tune_event_audit.parquet')
        cl_aud_dir = audit_root / 'cluster_audit'
        a_off = np.memmap(cl_aud_dir / 'pix_offsets.bin', dtype='<i8', mode='r')
        n_aud_cl = int(a_off[-1])
        a_cid = np.memmap(cl_aud_dir / 'cluster_first_channel_id.bin', dtype='<u8', mode='r', shape=(n_aud_cl,))
        a_or = np.memmap(cl_aud_dir / 'cluster_origin.bin', dtype='|i1', mode='r', shape=(n_aud_cl,))
        apos = pd.MultiIndex.from_frame(ev_aud[KEY]).get_indexer(pd.MultiIndex.from_frame(ref_df[KEY]))
        aud_aligned = ev_aud.iloc[apos].reset_index(drop=True)
        aud_si = aud_aligned['audit_index'].to_numpy(dtype=np.int64)

        pinpoint_6layer_mask = (
            pix_cent
            & (aud_aligned['vertex_valid'].to_numpy(dtype=int) > 0)
            & (aud_aligned['vertex_z_mm'].to_numpy(dtype=np.float64) <= -2815.0)
        )
        if pinpoint_6layer_mask.any():
            probe_b['pinpoint_6layer_vertex_block'] = compute_probe_b(
                y, res_nat['probs'], res_q1m['probs'], pinpoint_6layer_mask, family_keys=fams, n_boot=400
            )

        all_or_probs: list[np.ndarray] = []
        all_or_true: list[np.ndarray] = []
        pc_or_probs: list[np.ndarray] = []
        pc_or_true: list[np.ndarray] = []
        raw_items = nat_ds.items

        with torch.no_grad():
            for b_i, batch in enumerate(nat_loader):
                db = move_batch_to_device(batch, device)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
                    out = model(db, return_aux=True)
                or_p = torch.softmax(out['dense']['origin_logits'].float(), dim=-1).cpu().numpy()
                vtx_idx = out['aux']['vtx_idx'].cpu().numpy()
                vtx_mask = out['aux']['vtx_mask'].cpu().numpy()
                ev_indices = batch['index'].numpy()

                for row_i, ref_pos in enumerate(ev_indices):
                    raw_i = int(pos[ref_pos])
                    p_fcid = np.asarray(raw_items[raw_i]['pix_first_cid'], dtype=np.uint64)
                    vm = vtx_mask[row_i]
                    if not vm.any():
                        continue
                    sel_cl = vtx_idx[row_i, vm]
                    sel_fcid = p_fcid[sel_cl]
                    si = int(aud_si[ref_pos])
                    s0, s1 = int(a_off[si]), int(a_off[si + 1])
                    sc_cids = np.asarray(a_cid[s0:s1], dtype=np.uint64)
                    sc_ors = np.asarray(a_or[s0:s1], dtype=np.int64)
                    if len(sc_cids) == 0:
                        continue
                    mpos = np.searchsorted(sc_cids, sel_fcid)
                    safe = np.minimum(mpos, len(sc_cids) - 1)
                    ok = (mpos < len(sc_cids)) & (sc_cids[safe] == sel_fcid)
                    if not ok.any():
                        continue
                    probs_ev = or_p[row_i, vm][ok]
                    true_ev = sc_ors[safe[ok]]
                    all_or_probs.append(probs_ev)
                    all_or_true.append(true_ev)
                    if pix_cent[ref_pos]:
                        pc_or_probs.append(probs_ev)
                        pc_or_true.append(true_ev)

        origin_eval = {
            'All_Tune40k': compute_origin_confusion_and_pr(
                np.concatenate(all_or_probs, axis=0), np.concatenate(all_or_true, axis=0)
            ),
            'Truth_pixel_centered': compute_origin_confusion_and_pr(
                np.concatenate(pc_or_probs, axis=0), np.concatenate(pc_or_true, axis=0)
            ),
        }

    return origin_eval, probe_a, probe_b


def plot_v3_figures(results: dict, fig_dir: Path) -> list[str]:
    """Generate clean English publication figures for Phase V3 (W3-09)."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    # 1. Roadmap progression across B0 -> V1 B1++ -> V2 -> V3
    labels = [
        'B0\nHGB',
        'V1 B1++\n(300k)',
        'V2 PF\nAnchored',
        'V3 Event\n4-Head',
        'V3 GN2\n(100k)',
        'V3 GN2\nFlagship (300k)',
    ]
    m_v3 = results['models']
    ba4_pc = [
        results['references']['b0_pixel_centered_BA4'] * 100.0,
        results['references']['b1pp_pixel_centered_BA4'] * 100.0,
        results['references']['pf_v2_anchored_pixel_centered_BA4'] * 100.0,
        m_v3['pf_v3_100k_16p_event4head']['ema']['slices']['Truth_pixel_centered']['BA4'] * 100.0,
        m_v3['pf_v3_100k_16p_gn2_dense']['ema']['slices']['Truth_pixel_centered']['BA4'] * 100.0,
        m_v3['pf_v3_300k_24p_gn2_flagship']['ema']['slices']['Truth_pixel_centered']['BA4'] * 100.0,
    ]
    tau_pc = [
        results['references']['b0_pixel_centered_tau1'] * 100.0,
        results['references']['b1pp_pixel_centered_tau1'] * 100.0,
        results['references']['pf_v2_anchored_pixel_centered_tau1'] * 100.0,
        m_v3['pf_v3_100k_16p_event4head']['ema']['slices']['Truth_pixel_centered']['tau_at_1pct_fpr'] * 100.0,
        m_v3['pf_v3_100k_16p_gn2_dense']['ema']['slices']['Truth_pixel_centered']['tau_at_1pct_fpr'] * 100.0,
        m_v3['pf_v3_300k_24p_gn2_flagship']['ema']['slices']['Truth_pixel_centered']['tau_at_1pct_fpr'] * 100.0,
    ]
    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.6), dpi=300)
    cols = ['#94a3b8', '#64748b', '#0ea5e9', '#6366f1', '#10b981', '#2563eb']
    b1 = ax1.bar(x, ba4_pc, width=0.58, color=cols)
    ax1.axhline(59.92, color='#dc2626', linestyle='--', linewidth=1.3, label='V3 Gate: B0 + 4.0 pp (59.92%)')
    for b in b1:
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f'{b.get_height():.2f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel('Balanced Accuracy BA4 (%)')
    ax1.set_title('Truth Pixel-Centered 4-Class BA4 Progression')
    ax1.set_ylim(0, max(ba4_pc) * 1.18)
    ax1.legend(fontsize=8)
    ax1.grid(axis='y', alpha=0.25)

    b2 = ax2.bar(x, tau_pc, width=0.58, color=cols)
    for b in b2:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f'{b.get_height():.2f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel('tau Efficiency @ 1% FPR (%)')
    ax2.set_title('Truth Pixel-Centered tau@1% FPR Progression')
    ax2.set_ylim(0, max(tau_pc) * 1.18)
    ax2.grid(axis='y', alpha=0.25)

    fig.suptitle('Phase V3 PinpointFormer-GN2 Multitask Progression (Tune40k EMA)', fontsize=12)
    fig.tight_layout()
    p1 = fig_dir / 'v3_progression_overview.png'
    fig.savefig(p1, dpi=300)
    plt.close(fig)
    saved.append(str(p1))

    # 2. Charm rejection & lambda_charm scan
    chm = results['charm_rejection_audit']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.5), dpi=300)
    c_models = [(k, v['Truth_pixel_centered']) for k, v in chm['models'].items() if v.get('Truth_pixel_centered')]
    cx = np.arange(len(c_models))
    ch_fpr = [(v['charm_fpr_at_global_1pct'] or 0.0) * 100.0 for _, v in c_models]
    ch_names = [k.replace('pf_v3_', 'V3 ').replace('pf_v2_', 'V2 ').replace('b1pp_', 'B1++ ').replace('_', '\n') for k, _ in c_models]
    bars = ax1.bar(cx, ch_fpr, width=0.55, color='#f97316')
    for b in bars:
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15, f'{b.get_height():.2f}%', ha='center', va='bottom', fontsize=8)
    ax1.set_xticks(cx)
    ax1.set_xticklabels(ch_names, fontsize=7.5)
    ax1.set_ylabel('Charm Background FPR at 1% Global Cut (%)')
    ax1.set_title('Concurrent Charm Mis-Tag Rate (Truth Pixel-Centered)')
    ax1.grid(axis='y', alpha=0.25)

    flag_scan = chm['lambda_charm_scan'].get('pf_v3_300k_24p_gn2_flagship', {}).get('grid', {})
    if flag_scan:
        lams = sorted(float(k) for k in flag_scan)
        tau_ls = [flag_scan[str(l)]['Truth_pixel_centered']['tau_eff_at_1pct_fpr'] * 100.0 for l in lams]
        chm_ls = [flag_scan[str(l)]['Truth_pixel_centered']['charm_fpr_at_global_1pct'] * 100.0 for l in lams]
        ax2.plot(lams, tau_ls, marker='o', color='#2563eb', linewidth=2.0, label='tau@1% FPR (%)')
        ax2_t = ax2.twinx()
        ax2_t.plot(lams, chm_ls, marker='s', linestyle='--', color='#dc2626', linewidth=2.0, label='Charm FPR (%)')
        ax2.set_xlabel('Charm Penalty Weight lambda_charm')
        ax2.set_ylabel('tau@1% FPR (%)', color='#2563eb')
        ax2_t.set_ylabel('Charm FPR (%)', color='#dc2626')
        ax2.set_title('OOF Regional S_tau(lambda_charm) Scan (Flagship)')
        ax2.grid(alpha=0.25)
    fig.tight_layout()
    p2 = fig_dir / 'v3_charm_rejection_and_stau.png'
    fig.savefig(p2, dpi=300)
    plt.close(fig)
    saved.append(str(p2))

    # 3. Cluster Origin 5x5 confusion matrix & per-class PR-AUC
    oe = results['cluster_origin_eval']['Truth_pixel_centered']
    cm = np.asarray(oe['confusion_row_norm'], dtype=np.float64) * 100.0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.0), dpi=300)
    im = ax1.imshow(cm, cmap='Blues', vmin=0, vmax=100)
    short_names = ['Prim.\nLepton', 'Prim.\nHadron', 'Tau\nDaughter', 'Charm\nDaughter', 'Sec.\nShower']
    ax1.set_xticks(np.arange(5))
    ax1.set_yticks(np.arange(5))
    ax1.set_xticklabels(short_names, fontsize=8)
    ax1.set_yticklabels(short_names, fontsize=8)
    ax1.set_xlabel('Predicted Origin')
    ax1.set_ylabel('True Origin')
    ax1.set_title('Near-Vertex Cluster Origin Confusion (Row-Norm %)')
    for i in range(5):
        for j in range(5):
            val = cm[i, j]
            ax1.text(j, i, f'{val:.1f}%', ha='center', va='center', fontsize=8, color='white' if val > 55 else 'black')
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    pr_aucs = [oe['per_class'][c]['pr_auc'] for c in CLUSTER_ORIGIN_NAMES]
    f1s = [oe['per_class'][c]['f1'] for c in CLUSTER_ORIGIN_NAMES]
    ox = np.arange(5)
    w = 0.36
    b1 = ax2.bar(ox - w / 2, pr_aucs, width=w, color='#2563eb', label='PR-AUC (AP)')
    b2 = ax2.bar(ox + w / 2, f1s, width=w, color='#10b981', label='F1 Score')
    for b in list(b1) + list(b2):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f'{b.get_height():.2f}', ha='center', va='bottom', fontsize=8)
    ax2.set_xticks(ox)
    ax2.set_xticklabels(short_names, fontsize=8)
    ax2.set_ylim(0, 1.12)
    ax2.set_title('5-Class Cluster Origin PR-AUC & F1 (Truth Pixel-Centered)')
    ax2.legend(fontsize=8.5)
    ax2.grid(axis='y', alpha=0.25)

    fig.tight_layout()
    p3 = fig_dir / 'v3_cluster_origin_confusion_pr.png'
    fig.savefig(p3, dpi=300)
    plt.close(fig)
    saved.append(str(p3))

    # 4. Flagship Probe A & Probe B summary figure
    pa = results['flagship_probe_a']['All_Tune40k']
    pb = results['flagship_probe_b']
    bins = ['0', '1-100', '100-1000', '1000-10000', '10000-100000']
    bx = np.arange(len(bins))
    effs = [
        np.nan if pa['bins'].get(b, {}).get('tau_eff_at_global_1pct_fpr') is None
        else pa['bins'][b]['tau_eff_at_global_1pct_fpr'] * 100.0
        for b in bins
    ]
    ents = [pa['bins'].get(b, {}).get('entropy_mean', np.nan) for b in bins]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.4), dpi=300)
    ax1.plot(bx, effs, marker='o', linewidth=2.0, color='#2563eb', label='V3 Flagship tau@1% FPR')
    ax1.scatter([3], [0.29], marker='X', s=85, color='#dc2626', label='S6 MAE3 [1k,10k] (0.29%)')
    ax1.set_xticks(bx)
    ax1.set_xticklabels(bins, rotation=15, fontsize=8.5)
    ax1.set_ylabel('tau Efficiency @ Global 1% FPR (%)', color='#2563eb')
    ax1.set_xlabel('Raw Pixel Hits Occupancy Bin')
    ax1_t = ax1.twinx()
    ax1_t.plot(bx, ents, marker='^', linestyle='--', color='#059669', linewidth=1.8, label='Mean H_norm')
    ax1_t.set_ylabel('Normalized Entropy H_norm', color='#059669')
    ax1_t.set_ylim(0, 1.05)
    ax1.set_title('Flagship Probe A: Occupancy Robustness & Entropy')
    ax1.grid(alpha=0.25)

    cats = ['Native Micron\n(V3 Flagship)', '1.0 mm Quantized\n(Inference)']
    vals_ba4 = [pb['native']['BA4'] * 100.0, pb['q1mm']['BA4'] * 100.0]
    vals_t1 = [pb['native']['tau_at_1pct_fpr'] * 100.0, pb['q1mm']['tau_at_1pct_fpr'] * 100.0]
    px = np.arange(2)
    pw = 0.34
    b1 = ax2.bar(px - pw / 2, vals_ba4, width=pw, color='#2563eb', label='BA4 (%)')
    b2 = ax2.bar(px + pw / 2, vals_t1, width=pw, color='#059669', label='tau@1% FPR (%)')
    for b in list(b1) + list(b2):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f'{b.get_height():.1f}%', ha='center', va='bottom', fontsize=8.5)
    ax2.set_xticks(px)
    ax2.set_xticklabels(cats, fontsize=9)
    ax2.set_ylim(0, max(max(vals_ba4), max(vals_t1)) * 1.18)
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Flagship Probe B: Micron vs 1.0 mm (Truth Pixel-Centered)')
    ax2.legend(fontsize=8.5)
    ax2.grid(axis='y', alpha=0.25)

    fig.tight_layout()
    p4 = fig_dir / 'v3_flagship_probes.png'
    fig.savefig(p4, dpi=300)
    plt.close(fig)
    saved.append(str(p4))
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_V3_ROOT / 'W3-09')
    parser.add_argument('--audit-root', type=Path, default=DEFAULT_V3_ROOT / 'W3-08' / 'tune40k_offline_audit')
    parser.add_argument('--v2-eval', type=Path, default=DEFAULT_V3_ROOT / 'W3-07' / 'evaluation_results.json')
    parser.add_argument('--v1-eval', type=Path, default=DEFAULT_V3_ROOT / 'W3-05' / 'evaluation_results.json')
    parser.add_argument('--skip-probes', action='store_true')
    args = parser.parse_args()

    ref_df = load_tune_reference_frame()
    ev_aud = pd.read_parquet(args.audit_root / 'tune_event_audit.parquet')

    model_names = (
        'pf_v3_100k_16p_event4head',
        'pf_v3_100k_16p_gn2_dense',
        'pf_v3_300k_24p_gn2_flagship',
    )
    exp_root = args.root / 'experiments'
    results: dict = {'models': {}, 'references': {}, 'v3_promotion_gate': {}}
    pred_dfs_for_charm: dict[str, pd.DataFrame] = {}

    b1pp_p = DEFAULT_V3_ROOT / 'W3-05' / 'experiments' / 'b1pp_300k_24p_t4' / 'predictions.parquet'
    if b1pp_p.exists():
        pred_dfs_for_charm['b1pp_300k_24p_t4'] = pd.read_parquet(b1pp_p)
    v2_p = DEFAULT_V3_ROOT / 'W3-07' / 'experiments' / 'pf_v2_100k_16p_anchored' / 'predictions.parquet'
    if v2_p.exists():
        pred_dfs_for_charm['pf_v2_100k_16p_anchored'] = pd.read_parquet(v2_p)

    for mname in model_names:
        mdir = exp_root / mname
        if not mdir.exists():
            continue
        m_out: dict = {}
        for tag, fname in (('ema', 'predictions.parquet'), ('best', 'best_predictions.parquet'), ('exposure', 'exposure_predictions.parquet')):
            p = mdir / fname
            if p.exists():
                df = pd.read_parquet(p)
                if tag == 'ema':
                    pred_dfs_for_charm[mname] = df
                chm_p = df['charm_prob'].to_numpy(dtype=np.float64) if 'charm_prob' in df.columns else None
                m_out[tag] = evaluate_predictions_frame(df, ref_df, audit_df=ev_aud, charm_probs=chm_p, lambda_charm=0.0)
        results['models'][mname] = m_out

    results['charm_rejection_audit'] = compute_charm_rejection_audit(ref_df, ev_aud, pred_dfs_for_charm)

    # Record clean baseline references from V1 & V2
    results['references']['b0_pixel_centered_BA4'] = 0.5592
    results['references']['b0_pixel_centered_tau1'] = 0.0436
    if args.v1_eval.exists():
        v1 = json.loads(args.v1_eval.read_text())
        v1_m = v1.get('models', {}).get('b1pp_300k_24p_t4', {})
        v1_pc = v1_m.get('slices', v1_m.get('ema', {}).get('slices', {})).get('Truth_pixel_centered', {})
        results['references']['b1pp_pixel_centered_BA4'] = float(v1_pc.get('BA4', v1_pc.get('BA4_raw', 0.6352)))
        v1_t1 = v1_pc.get('tau_at_1pct_fpr', v1_pc.get('working_points', {}).get('0.01', {}).get('raw_global', {}).get('tau_eff', 0.1057))
        results['references']['b1pp_pixel_centered_tau1'] = float(v1_t1)
    if args.v2_eval.exists():
        v2 = json.loads(args.v2_eval.read_text())
        v2_pc = v2.get('models', {}).get('pf_v2_100k_16p_anchored', {}).get('ema', {}).get('slices', {}).get('Truth_pixel_centered', {})
        results['references']['pf_v2_anchored_pixel_centered_BA4'] = float(v2_pc.get('BA4', 0.0))
        results['references']['pf_v2_anchored_pixel_centered_tau1'] = float(v2_pc.get('tau_at_1pct_fpr', 0.0))

    if not args.skip_probes:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        geometry = json.loads(DEFAULT_GEOMETRY.read_text())
        bounds = detector_bounds(geometry)
        or_eval, pa, pb = _evaluate_flagship_probes_and_origin(args.root, ref_df, args.audit_root, geometry, bounds, device)
        results['cluster_origin_eval'] = or_eval
        results['flagship_probe_a'] = pa
        results['flagship_probe_b'] = pb

        flag_pc = results['models']['pf_v3_300k_24p_gn2_flagship']['ema']['slices']['Truth_pixel_centered']
        ch_base = results['charm_rejection_audit']['models'].get('pf_v2_100k_16p_anchored', {}).get('Truth_pixel_centered', {}).get('charm_fpr_at_global_1pct')
        ch_flag = results['charm_rejection_audit']['models']['pf_v3_300k_24p_gn2_flagship']['with_best_lambda_charm']['Truth_pixel_centered']['charm_fpr_at_global_1pct']
        results['v3_promotion_gate'] = {
            'flagship_pixel_centered_BA4': float(flag_pc['BA4']),
            'gate_min_BA4': 0.5992,
            'beats_b0_plus_4pp': bool(flag_pc['BA4'] >= 0.5992),
            'charm_fpr_reduced_vs_v2': bool(ch_base is None or (ch_flag is not None and ch_flag < ch_base)),
            'probe_a_collapse_eliminated': bool(pa['All_Tune40k']['collapse_eliminated']),
            'probe_b_passes': bool(pb['passes_probe_b']),
            'cluster_origin_macro_pr_auc': float(or_eval['Truth_pixel_centered']['macro_pr_auc']),
            'passed': bool(
                flag_pc['BA4'] >= 0.5992
                and pa['All_Tune40k']['collapse_eliminated']
                and pb['passes_probe_b']
            ),
        }
        results['figures'] = plot_v3_figures(results, args.root / 'figures')

    write_json(args.root / 'evaluation_results.json', results)
    print(json.dumps({
        'v3_promotion_gate': results.get('v3_promotion_gate', {}),
        'figures': results.get('figures', []),
    }, indent=2))


if __name__ == '__main__':
    main()
