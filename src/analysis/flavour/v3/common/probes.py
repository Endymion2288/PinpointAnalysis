"""Mandatory physics diagnostics: Probe A (high-occupancy entropy/efficiency) & Probe B (micron vs 1mm)."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from analysis.flavour.v2.r1.metrics import Metrics
from analysis.flavour.v3.common.calibration import _tau_efficiency_at_fpr
from analysis.flavour.v3.v2_pinpointformer.attention import (
    collect_attention_scales,
    collect_effective_attention_sharpness,
)


PIXEL_OCCUPANCY_BINS = (
    ('0', 0, 0),
    ('1-100', 1, 100),
    ('100-1000', 100, 1000),
    ('1000-10000', 1000, 10000),
    ('10000-100000', 10000, 100000),
)


def _bin_mask(raw_hits: np.ndarray, lo: int, hi: int) -> np.ndarray:
    if lo == 0 and hi == 0:
        return raw_hits == 0
    if lo == 10000:
        return raw_hits >= 10000
    return (raw_hits >= lo) & (raw_hits < hi)


def _tau_log_score(probs: np.ndarray) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float64)
    p = p / np.maximum(p.sum(axis=-1, keepdims=True), 1e-12)
    p_tau = np.clip(p[:, 2], 1e-8, 1.0 - 1e-8)
    return np.log(p_tau / (1.0 - p_tau))


def compute_probe_a(
    probs: np.ndarray,
    labels: np.ndarray,
    raw_pixel_hits: np.ndarray,
    mean_entropy: np.ndarray,
    learned_scales: dict[str, list[float]] | None = None,
    layer_entropy: dict[str, np.ndarray] | None = None,
    subset_mask: np.ndarray | None = None,
) -> dict:
    """Measure tau efficiency @ 1% FPR and normalized attention entropy across pixel occupancy bins."""
    if subset_mask is not None:
        m = np.asarray(subset_mask, dtype=bool)
        probs = probs[m]
        labels = labels[m]
        raw_pixel_hits = raw_pixel_hits[m]
        mean_entropy = mean_entropy[m]
        if layer_entropy is not None:
            layer_entropy = {k: v[m] for k, v in layer_entropy.items()}

    scores = _tau_log_score(probs)
    neg = labels != 2
    if int(neg.sum()) > 0:
        global_thr_1pct = float(np.quantile(scores[neg], 0.99))
        global_thr_01pct = float(np.quantile(scores[neg], 0.999))
    else:
        global_thr_1pct = float('inf')
        global_thr_01pct = float('inf')

    bins_out: dict[str, dict] = {}
    for name, lo, hi in PIXEL_OCCUPANCY_BINS:
        bm = _bin_mask(raw_pixel_hits, lo, hi)
        n_tot = int(bm.sum())
        n_tau = int((bm & (labels == 2)).sum())
        n_neg = int((bm & (labels != 2)).sum())
        if n_tot == 0:
            bins_out[name] = {'count': 0, 'n_tau': 0, 'n_non_tau': 0}
            continue
        ent_b = mean_entropy[bm]
        tau_b = bm & (labels == 2)
        neg_b = bm & (labels != 2)
        eff_global_1pct = float((scores[tau_b] >= global_thr_1pct).mean()) if n_tau > 0 else None
        fpr_global_1pct = float((scores[neg_b] >= global_thr_1pct).mean()) if n_neg > 0 else None
        eff_within_1pct = float(_tau_efficiency_at_fpr(scores[bm], labels[bm], 0.01)['tau_eff']) if (n_tau > 0 and n_neg > 0) else None
        eff_within_01pct = float(_tau_efficiency_at_fpr(scores[bm], labels[bm], 0.001)['tau_eff']) if (n_tau > 0 and n_neg > 0) else None
        entry = {
            'count': n_tot,
            'n_tau': n_tau,
            'n_non_tau': n_neg,
            'tau_eff_at_global_1pct_fpr': eff_global_1pct,
            'bin_fpr_at_global_1pct_cut': fpr_global_1pct,
            'within_bin_tau_eff_at_1pct_fpr': eff_within_1pct,
            'within_bin_tau_eff_at_0p1pct_fpr': eff_within_01pct,
            'entropy_mean': float(np.mean(ent_b)),
            'entropy_std': float(np.std(ent_b)),
            'entropy_median': float(np.median(ent_b)),
        }
        if layer_entropy is not None:
            entry['layer_entropy_means'] = {k: float(np.mean(v[bm])) for k, v in layer_entropy.items()}
        bins_out[name] = entry

    b_1k_10k = bins_out.get('1000-10000', {})
    eff_1k_10k = b_1k_10k.get('tau_eff_at_global_1pct_fpr')
    if eff_1k_10k is None:
        eff_1k_10k = b_1k_10k.get('within_bin_tau_eff_at_1pct_fpr')

    scale_summary: dict = {}
    if learned_scales:
        flat_scales = np.concatenate([np.asarray(v, dtype=np.float64) for v in learned_scales.values()])
        scale_summary = {
            'min': float(flat_scales.min()),
            'median': float(np.median(flat_scales)),
            'mean': float(flat_scales.mean()),
            'max': float(flat_scales.max()),
            'n_heads_total': int(len(flat_scales)),
            'n_heads_gt_4p90': int((flat_scales > 4.90).sum()),
            'per_layer': learned_scales,
        }

    return {
        'global_thr_1pct_fpr': global_thr_1pct,
        'global_thr_0p1pct_fpr': global_thr_01pct,
        'bins': bins_out,
        's6_reference_1000_10000_tau_eff': 0.0029,
        'bin_1000_10000_tau_eff': eff_1k_10k,
        'collapse_eliminated': bool(eff_1k_10k is not None and eff_1k_10k > 0.02),
        'learned_scales': scale_summary,
    }


def _bootstrap_probe_b(
    labels: np.ndarray,
    family_keys: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    n_boot: int = 400,
    seed: int = 42,
) -> dict:
    """Block-bootstrap by family id for Probe B native micron vs 1mm metrics and paired deltas."""
    codes, _ = np.unique(family_keys, return_inverse=True)
    order = np.argsort(codes, kind='mergesort')
    sorted_codes = codes[order]
    _, split_idx = np.unique(sorted_codes, return_index=True)
    groups = [order[s:e] for s, e in zip(split_idx, np.r_[split_idx[1:], len(sorted_codes)])]
    n_fam = len(groups)

    scores_a = _tau_log_score(probs_a)
    scores_b = _tau_log_score(probs_b)
    rng = np.random.default_rng(seed)

    ba_a, ba_b, t1_a, t1_b = [], [], [], []
    d_ba, d_t1 = [], []
    for _ in range(n_boot):
        pick = rng.integers(0, n_fam, size=n_fam)
        ix = np.concatenate([groups[g] for g in pick])
        y = labels[ix]
        ma = Metrics(y, probs_a[ix]).report()
        mb = Metrics(y, probs_b[ix]).report()
        e1a = float(_tau_efficiency_at_fpr(scores_a[ix], y, 0.01)['tau_eff'])
        e1b = float(_tau_efficiency_at_fpr(scores_b[ix], y, 0.01)['tau_eff'])
        ba_a.append(float(ma['BA']))
        ba_b.append(float(mb['BA']))
        t1_a.append(e1a)
        t1_b.append(e1b)
        d_ba.append((float(ma['BA']) - float(mb['BA'])) * 100.0)
        d_t1.append((e1a - e1b) * 100.0)

    def _ci(vals: list[float]) -> list[float]:
        arr = np.asarray(vals, dtype=np.float64)
        return [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))]

    return {
        'n_families': n_fam,
        'n_boot': n_boot,
        'native_ba4_ci95': _ci(ba_a),
        'q1mm_ba4_ci95': _ci(ba_b),
        'native_tau_1pct_ci95': _ci(t1_a),
        'q1mm_tau_1pct_ci95': _ci(t1_b),
        'delta_ba4_pp_ci95': _ci(d_ba),
        'delta_tau_1pct_pp_ci95': _ci(d_t1),
    }


def compute_probe_b(
    labels: np.ndarray,
    probs_native: np.ndarray,
    probs_q1mm: np.ndarray,
    pixel_centered_mask: np.ndarray,
    family_keys: np.ndarray | None = None,
    n_boot: int = 400,
) -> dict:
    """Evaluate native micron vs 1.0 mm quantized pixel resolution on Truth_pixel_centered."""
    m = np.asarray(pixel_centered_mask, dtype=bool)
    y = np.asarray(labels[m], dtype=np.int64)
    pn = np.asarray(probs_native[m], dtype=np.float64)
    pq = np.asarray(probs_q1mm[m], dtype=np.float64)

    mn = Metrics(y, pn).report()
    mq = Metrics(y, pq).report()
    sn = _tau_log_score(pn)
    sq = _tau_log_score(pq)
    t1_n = float(_tau_efficiency_at_fpr(sn, y, 0.01)['tau_eff'])
    t1_q = float(_tau_efficiency_at_fpr(sq, y, 0.01)['tau_eff'])
    t01_n = float(_tau_efficiency_at_fpr(sn, y, 0.001)['tau_eff'])
    t01_q = float(_tau_efficiency_at_fpr(sq, y, 0.001)['tau_eff'])

    d_ba4_pp = float((float(mn['BA']) - float(mq['BA'])) * 100.0)
    d_tau1_pp = float((t1_n - t1_q) * 100.0)
    d_tau01_pp = float((t01_n - t01_q) * 100.0)

    out = {
        'n_events': int(len(y)),
        'native': {
            'BA4': float(mn['BA']),
            'auroc_tau': float(mn['per_class'][2]['AUROC']),
            'tau_at_1pct_fpr': float(t1_n),
            'tau_at_0p1pct_fpr': float(t01_n),
        },
        'q1mm': {
            'BA4': float(mq['BA']),
            'auroc_tau': float(mq['per_class'][2]['AUROC']),
            'tau_at_1pct_fpr': float(t1_q),
            'tau_at_0p1pct_fpr': float(t01_q),
        },
        'delta_ba4_pp': d_ba4_pp,
        'delta_tau_1pct_pp': d_tau1_pp,
        'delta_tau_0p1pct_pp': d_tau01_pp,
        'passes_ba4_ge_2pp': bool(d_ba4_pp >= 2.0),
        'passes_tau1_ge_2pp': bool(d_tau1_pp >= 2.0),
        'passes_probe_b': bool(d_ba4_pp >= 2.0 and d_tau1_pp >= 2.0),
    }
    if family_keys is not None and n_boot > 0:
        out['bootstrap'] = _bootstrap_probe_b(y, np.asarray(family_keys)[m], pn, pq, n_boot=n_boot)
    return out


@torch.no_grad()
def run_model_with_entropy(
    model: nn.Module,
    loader,
    device: torch.device,
    use_bf16: bool = False,
) -> dict:
    """Run forward pass collecting logits/probs, raw_pixel_hits, and per-layer attention entropy."""
    model.eval()
    probs_list: list[np.ndarray] = []
    logits_list: list[np.ndarray] = []
    hits_list: list[np.ndarray] = []
    vhat_list: list[np.ndarray] = []
    ent_per_layer: list[list[np.ndarray]] | None = None

    amp_enabled = use_bf16 and device.type == 'cuda'
    for batch in loader:
        dev_batch = {
            k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=amp_enabled):
            out = model(dev_batch, return_aux=True, return_entropy=True)
        logits_ev = out['logits']
        if logits_ev.shape[-1] == 7:
            p7 = torch.softmax(logits_ev.float(), dim=-1)
            probs = torch.stack([p7[:, 0], p7[:, 1] + p7[:, 2], p7[:, 3] + p7[:, 4] + p7[:, 5], p7[:, 6]], dim=-1)
        else:
            probs = torch.softmax(logits_ev.float(), dim=-1)
        logits_list.append(logits_ev.float().cpu().numpy())
        probs_list.append(probs.cpu().numpy())
        hits_list.append(batch['raw_pixel_hits'].numpy())
        vhat_list.append(out['v_hat'].float().cpu().numpy())

        ents = out['entropies']
        if ent_per_layer is None:
            ent_per_layer = [[] for _ in range(len(ents))]
        for li, et in enumerate(ents):
            # et shape: (B, heads, nq) -> per-event mean normalized entropy
            ent_per_layer[li].append(et.float().mean(dim=(1, 2)).cpu().numpy())

    assert ent_per_layer is not None
    layer_arrs = {
        f'layer_{i:02d}': np.concatenate(chunks, axis=0)
        for i, chunks in enumerate(ent_per_layer)
    }
    stacked = np.stack(list(layer_arrs.values()), axis=1)
    return {
        'logits': np.concatenate(logits_list, axis=0),
        'probs': np.concatenate(probs_list, axis=0),
        'raw_pixel_hits': np.concatenate(hits_list, axis=0),
        'v_hat': np.concatenate(vhat_list, axis=0),
        'mean_entropy': stacked.mean(axis=1),
        'layer_entropy': layer_arrs,
        'learned_scales': collect_attention_scales(model),
        'effective_sharpness': collect_effective_attention_sharpness(model),
    }
