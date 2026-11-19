"""Reconstruct Stage 321 metrics, disjoint bins and Asimov fits from event scores."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax, xlogy
from scipy.stats import beta, binom, norm

KEY = ['run', 'chunk', 'row']
SEEDS = (17, 42, 2026)
DOMAINS = ('D_pix_contained', 'D_scint_contained', 'D_partial_or_edge')
PRIMARY = 'v2_pf_hybrid_uvtx'
B1 = 'v2_b1pp_clean'


def cache_path(root, data, split, prefix, frozen):
    request = dict(root=str(data), split=split, prefix=prefix, override={}, precision='fp32',
                   shas=[frozen[f'{prefix}_s{s}']['sha256'] for s in SEEDS])
    tag = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:16]
    return root / f'stage321/unsealed_preds_fp32/{prefix}_{split}_{tag}'


def inputs(root, out, partition, frozen, prefix=PRIMARY, dual=False):
    main = partition in ('cal_a', 'cut_b', 'confirm')
    data = root / 'data' if main else root / 'sys_shards' / partition
    meta_path = root / f'sealed_truth_meta/{partition}_truth_meta.parquet' if main else data / 'truth_meta.parquet'
    meta = pd.read_parquet(meta_path)
    cache = cache_path(root, data, partition if main else 'confirm', prefix, frozen)
    logs, charm, vertex = [], [], []
    keys = pd.MultiIndex.from_frame(meta[KEY])
    for seed in SEEDS:
        name = f'{prefix}_s{seed}'
        score = pd.read_parquet(cache / f'{name}.parquet').set_index(KEY).reindex(keys)
        probs = score[[f'p_{i}' for i in range(4)]].to_numpy(dtype=np.float64)
        assert np.isfinite(probs).all(), (partition, seed)
        logs.append(np.log(np.maximum(probs / probs.sum(axis=1, keepdims=True), 1e-8)))
        charm.append(score['charm_prob'].to_numpy(dtype=np.float64))
        vertex.append(score[[f'v_hat_{axis}_mm' for axis in 'xyz']].to_numpy())
    if dual:
        b1 = pd.read_parquet(out / f'b1_{partition}.parquet').set_index(KEY).reindex(keys)
        assert not b1.isna().any().any(), partition
        meta['recomputed_domain'] = b1['domain_recomputed'].to_numpy()
        for seed in SEEDS:
            probs = b1[[f'{B1}_s{seed}_p{i}' for i in range(4)]].to_numpy(dtype=np.float64)
            logs.append(np.log(np.maximum(probs / probs.sum(axis=1, keepdims=True), 1e-8)))
    return meta, np.stack(logs), np.mean(charm, axis=0), np.mean(vertex, axis=0)


def scores(meta, logs, charm, params, lambdas):
    dom = meta.obs_domain_id.to_numpy()
    p = np.full((len(meta), 4), .25)
    s = np.full(len(meta), -25.)
    for d in (1, 2, 3):
        m = dom == d
        cfg = params[str(d)]
        logits = np.einsum('m,mnc->nc', cfg['weights'], logs[:, m])
        p[m] = softmax(cfg['inv_T'] * logits + cfg['bias'], axis=1)
        s[m] = np.log(p[m, 2] / (1 - p[m, 2] + lambdas[str(d)] * charm[m] + 1e-12))
    return p, s


def rates(y, mask, score, threshold):
    signal = mask & (y == 2)
    background = mask & (y != 2)
    passed = score >= threshold
    tp, fp = int((passed & signal).sum()), int((passed & background).sum())
    n_s, n_b = int(signal.sum()), int(background.sum())
    low = float(beta.ppf(.025, fp, n_b - fp + 1)) if fp else 0.
    high = float(beta.ppf(.975, fp + 1, n_b - fp)) if fp < n_b else 1.
    return {'tp': tp, 'n_signal': n_s, 'fp': fp, 'n_background': n_b,
            'eff': tp / n_s, 'fpr': fp / n_b, 'fixed_threshold_fpr_ci95': [low, high]}


def ranked_threshold(values, target):
    ranked = np.sort(values)[::-1]
    whole_groups = np.r_[np.flatnonzero(ranked[1:] != ranked[:-1]) + 1, len(ranked)]
    allowed = whole_groups[whole_groups <= target * len(ranked) + 1e-12]
    return ranked[allowed[-1] - 1] if len(allowed) else np.nextafter(ranked[0], np.inf)


def metric_report(meta, probability, score, vertex, thresholds=None):
    class_labels = np.asarray(meta.label)
    domain = np.asarray(meta.obs_domain_id)
    weights = np.asarray(meta.geom_weight)
    masks = {'all': np.ones(len(meta), dtype=bool), 'pinpoint': meta.pixel_centered.to_numpy(),
             'front': meta.front_pinpoint_pixel_centered.to_numpy(), 'D1': domain == 1,
             'D2': domain == 2, 'D3': domain == 3, 'mass_fid': meta.large_scintillator.to_numpy(),
             'mass_all': np.ones(len(meta), dtype=bool)}
    results = {}
    labels3 = np.array([0, 1, 3])
    for name, mask in masks.items():
        y = class_labels[mask]
        w = weights[mask] if name.startswith('mass_') else np.ones(len(y))
        p = probability[mask]
        yhat = p.argmax(axis=1)
        conf = np.zeros((4, 4), dtype=float)
        np.add.at(conf, (y, yhat), w)
        yhat3 = labels3[p[:, labels3].argmax(axis=1)]
        res = {
            'count': len(y), 'BA4': float(np.mean(np.diag(conf) / conf.sum(axis=1))),
            'conditional_BA3': float(np.mean([np.average(yhat3[y == c] == c, weights=w[y == c]) for c in labels3])),
            'NLL': float(np.average(-np.log(p[np.arange(len(y)), y]), weights=w)),
            'confusion': conf.tolist(),
        }
        delta = np.abs(vertex[mask] - meta.loc[mask, [f'vertex_{a}_mm' for a in 'xyz']].to_numpy())
        res['median_abs_vertex_errors_mm'] = np.median(delta, axis=0).tolist()
        for fpr, label in ((.01, '1pct'), (.001, '0p1pct')):
            for field, values in (('p_tau', probability[:, 2]), ('s_tau', score)):
                res[f'{field}_{label}'] = rates(class_labels, mask, values,
                                                ranked_threshold(values[mask & (class_labels != 2)], fpr))
            if thresholds and name in ('D1', 'D2', 'D3'):
                thr = thresholds[DOMAINS[int(name[1]) - 1]][label + '_fpr']
                res[f'locked_{label}'] = rates(class_labels, mask, score, thr)
        results[name] = res
    return results


def bins(meta, score, charm, edges, cr_edges):
    result = np.full(len(meta), -1, dtype=int)
    for d, name in enumerate(DOMAINS, 1):
        mask = meta.obs_domain_id.to_numpy() == d
        local = np.searchsorted(edges[name], score[mask], side='right')
        low, high = cr_edges[name]
        selected = (local <= 1) & (charm[mask] >= low)
        result[mask] = np.where(selected, 21 + 2 * (d - 1) + (charm[mask] >= high), 7 * (d - 1) + local)
    return result


def hist(indices, weight):
    m = indices >= 0
    return np.bincount(indices[m], weights=np.asarray(weight)[m], minlength=27)


def physical_components(n_fid, meta, indices, ext_shape, veto=False, ext_count=None):
    y = meta.label.to_numpy()
    charm = meta.has_charm.to_numpy()
    fid = meta.large_scintillator.to_numpy()
    if veto:
        indices = np.where(meta.p0_pixel_hits.to_numpy() == 0, indices, -1)
    unit = meta.geom_weight.to_numpy()
    processes = [(0, False), (0, True), (1, False), (1, True), (3, False), (3, True)]
    weights = np.zeros(len(meta))
    for c, frac in [(0, .170), (1, .580), (2, .003), (3, .247)]:
        m = fid & (y == c)
        categories = ((m & charm, .045), (m & ~charm, .955)) if c == 3 else ((m, 1),)
        for sub, share in categories:
            weights[sub] = unit[sub] * (n_fid * frac * share) / unit[sub].sum()
    signal = hist(indices, np.where(y == 2, weights, 0))
    parts = np.stack([hist(indices, np.where((y == c) & (charm == ch), weights, 0)) for c, ch in processes])
    external = ext_shape * (.02 * n_fid)
    var = hist(indices, np.where(y != 2, weights ** 2, 0))
    if ext_count is not None:
        var += external * (.02 * n_fid) / ext_count
    return signal, parts, external, var


def shape_hist(meta, indices):
    y = meta.label.to_numpy()
    w = meta.geom_weight.to_numpy()
    h = []
    for c in range(4):
        m = y == c
        h.append(hist(indices[m], w[m] / w[m].sum()))
    return h[2], (h[0] * .170 + h[1] * .580 + h[3] * .247) / .997


def independent_profile(signal, parts, external, variance, shapes, mode):
    sums = parts.sum(axis=0) + external
    observations = sums + signal
    design = np.zeros((6, 27, 11))
    domain = np.r_[np.repeat(np.arange(3), 7), np.repeat(np.arange(3), 2)]
    slope = np.r_[np.tile(np.linspace(-1, 1, 7), 3), np.repeat(-.8, 6)]
    design[:, :, 0] = .05
    for d in range(3):
        design[:, domain == d, 1 + d] = .06
        design[:, domain == d, 4 + d] = .08 * slope[domain == d]
    design[:2, :, 7] = .10
    design[4:, :, 8] = .10
    design[[1, 3, 5], :, 9] = .30
    bstat = np.clip(variance / np.maximum(sums ** 2, 1e-18), 0, 2.5)
    use = bstat > 1e-10

    def objective(theta):
        shifts = design @ theta[:11]
        if len(shapes):
            a = theta[11:, None]
            up, down = shapes[:, 0], shapes[:, 1]
            shifts += np.sum(a * ((up - down) + np.tanh(4 * a) * (up + down)) / 2, axis=0)
        rate = np.sum(parts * np.exp(shifts), axis=0) + external * np.exp(.5 * theta[10])
        gamma = np.ones_like(rate)
        penalty = np.dot(theta, theta)
        if mode == 'gaussian':
            b = 1 - bstat[use] * rate[use]
            gamma[use] = np.maximum((b + np.sqrt(b*b + 4*bstat[use]*observations[use])) / 2, 1e-8)
            penalty += np.sum((gamma[use]-1)**2 / bstat[use])
        elif mode == 'poisson':
            auxiliary = 1 / bstat[use]
            gamma[use] = (observations[use] + auxiliary) / (rate[use] + auxiliary)
            penalty += 2 * np.sum(auxiliary * (gamma[use] - 1 - np.log(gamma[use])))
        mean = gamma * rate
        return float(2 * np.sum(mean - observations + xlogy(observations, observations / np.maximum(mean, 1e-300))) + penalty)

    fit = minimize(objective, np.zeros(11 + len(shapes)), method='L-BFGS-B',
                   bounds=[(-6, 6)] * (11 + len(shapes)), options={'ftol': 1e-12, 'maxiter': 600})
    positive = sums > 0
    unsupported = (sums == 0) & (signal > 0)
    qstat = 2 * np.sum((sums[positive] + signal[positive]) *
                       np.log1p(signal[positive] / sums[positive]) - signal[positive])
    return {'z_stat': None if unsupported.any() else float(np.sqrt(qstat)),
            'unsupported_signal_bins': np.flatnonzero(unsupported).tolist(), 'z_profile': float(np.sqrt(max(fit.fun, 0))),
            'q0': float(fit.fun), 'converged': bool(fit.success),
            'signal': float(signal.sum()), 'background': float(sums.sum()),
            'max_abs_pull': float(np.max(np.abs(fit.x)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--classification-only', action='store_true')
    args = parser.parse_args()
    root, out = args.root, args.output
    frozen = json.loads((root / 'v7_freeze_manifest.json').read_text())['checkpoints']
    stage = root / 'stage321'
    blend = json.loads((stage / 'cal_a_frozen_params.json').read_text())
    cut = json.loads((stage / 'cut_b_locked_params.json').read_text())
    primary_meta, lp, ch, vert = inputs(root, out, 'confirm', frozen)
    pf_params = blend['primary_pf_group']['dom_params']
    pf_cut = cut['primary_pf_group']
    prob, stau = scores(primary_meta, lp, ch, pf_params, pf_cut['lam_by_dom'])
    metrics = metric_report(primary_meta, prob, stau, vert, pf_cut['obs_cutb_thresholds'])
    domain = primary_meta.obs_domain_id.to_numpy()
    y = primary_meta.label.to_numpy()
    px = primary_meta.pixel_centered.to_numpy()
    threshold = ranked_threshold(stau[px & (y != 2)], .01)
    metrics['global_pinpoint_cut_domain_composition'] = [rates(y, px & (domain == d), stau, threshold)
                                                       for d in (1, 2, 3)]
    d1_bkg = (domain == 1) & (y != 2)
    neg_pass = stau >= pf_cut['obs_cutb_thresholds'][DOMAINS[0]]['1pct_fpr']
    charm_flag = primary_meta.has_charm.to_numpy()
    for name, mask in [('NC_charm', d1_bkg & (y == 3) & charm_flag),
                       ('CC_charm', d1_bkg & (y != 3) & charm_flag),
                       ('light', d1_bkg & ~charm_flag)]:
        metrics['D1'].setdefault('locked_1pct_background_composition', {})[name] = [
            int(np.count_nonzero(mask & neg_pass)), int(mask.sum())]
    (out / 'classification.json').write_text(json.dumps(metrics, indent=2))
    if args.classification_only:
        return

    cal_param = blend['primary_dual_family_6m']['dom_params']
    locked = cut['primary_dual_family_6m']
    lambdas = locked['lam_by_dom']
    partitions = ['cut_b', 'confirm', 'external_punchthrough', 'nominal_sys_ref', 'tau_helicity_0',
                  'w_density_0p98', 'no_magnet_shells', 'field_0p97', 'pythia6_std_ref', 'pythia6_nux']

    def load(partition):
        meta, logs, charm, vertex = inputs(root, out, partition, frozen, dual=True)
        p, s = scores(meta, logs, charm, cal_param, lambdas)
        return meta, p, s, charm, vertex

    with ThreadPoolExecutor(max_workers=4) as pool:
        frames = dict(zip(partitions, pool.map(load, partitions)))
    mcut, _, scut, ccut, _ = frames['cut_b']
    edges, cr = {}, {}
    for d, domain in enumerate(DOMAINS, 1):
        md = mcut.obs_domain_id.to_numpy() == d
        edges[domain] = np.quantile(scut[md], [.50, .85, .95, .985, .995, .9985])
        is_low = np.searchsorted(edges[domain], scut[md], side='right') <= 1
        cr[domain] = np.quantile(ccut[md][is_low], [.85, .96])
    by_partition = {k: bins(m, s, c, edges, cr) for k, (m, p, s, c, v) in frames.items()}
    conf, pp, ss, cc, vv = frames['confirm']
    metrics['dual'] = metric_report(conf, pp, ss, vv, locked['obs_cutb_thresholds'])
    (out / 'classification.json').write_text(json.dumps(metrics, indent=2))
    m_ext = frames['external_punchthrough'][0]
    ext_ids = by_partition['external_punchthrough']
    ext_eligible = m_ext.ext_subtype.isin(['neutron', 'k0l']).to_numpy() & (ext_ids >= 0)
    ext_veto = ext_eligible & (m_ext.p0_pixel_hits.to_numpy() == 0)
    ext_all_count = hist(ext_ids, ext_eligible.astype(int))
    ext_veto_count = hist(ext_ids, ext_veto.astype(int))
    ext_shape = ext_veto_count / ext_veto_count.sum()

    hs = {k: shape_hist(frames[k][0], by_partition[k]) for k in partitions[3:]}
    shapes = []
    for name in partitions[4:]:
        if name == 'pythia6_std_ref':
            continue
        ref = 'pythia6_std_ref' if name == 'pythia6_nux' else 'nominal_sys_ref'
        upper = np.clip(np.log((hs[name][1] + 1e-3) / (hs[ref][1] + 1e-3)), -.35, .35)
        if name == 'tau_helicity_0':
            upper[:] = 0
        shapes.append(np.stack([upper, (-1 if name == 'pythia6_nux' else -.5) * upper]))
    shapes = np.stack(shapes)
    official = json.loads((root / 'stage321_clean_v2_unsealing_report.json').read_text())
    result = {'recomputed_domain_mismatches': {k: int((m.recomputed_domain != m.obs_domain_id).sum())
                                              for k, (m, *_) in frames.items()},
              'charm_cut_edge_max_abs': float(max(np.max(np.abs(cr[k] - np.array(v))) for k, v in
                                                 official['cutb_locked_charm_cr_edges'].items())),
              'external_counts_after_veto': ext_veto_count.tolist(),
              'external_counts_before_veto': ext_all_count.tolist(),
              'external_eligible_before_veto': int(ext_eligible.sum()),
              'external_eligible_after_veto': int(ext_veto.sum()), 'eras': {}}
    for era, count in (('Run3_250fb', 15000), ('Run4_680fb', 40800), ('HLLHC_3000fb', 180000), ('FPF_10000fb', 600000)):
        components = physical_components(count, conf, by_partition['confirm'], ext_shape)
        variants = {
            mode: independent_profile(*components, shapes, mode) for mode in ('none', 'gaussian', 'poisson')
        }
        # Assume the external normalization was post-veto, as in the archived projection.
        coherent = physical_components(count, conf, by_partition['confirm'], ext_shape, veto=True,
                                       ext_count=ext_veto_count.sum())
        c_sig, c_parts, c_ext, c_var = coherent
        unsupported = (c_parts.sum(axis=0) + c_ext == 0) & (c_sig > 0)
        keep_bkg = conf.large_scintillator & (conf.label != 2) & (conf.p0_pixel_hits == 0)
        keep_bkg &= by_partition['confirm'] >= 0
        zero_mc_scale = .997 * count / keep_bkg.sum()
        c_parts[2, unsupported] = .5 * zero_mc_scale
        c_var[unsupported] = .5 * zero_mc_scale ** 2
        alternatives = independent_profile(c_sig, c_parts, c_ext, c_var, shapes, 'poisson')
        alternatives['prior_only_bin_ids'] = np.flatnonzero(unsupported).tolist()
        alternatives['prior_only_bin_signal'] = c_sig[unsupported].tolist()
        alternatives['prior_mean_per_bin'] = float(.5 * zero_mc_scale)
        variants['veto_all_poisson_with_ext_mc_and_empty_bin_continuation'] = alternatives
        result['eras'][era] = variants
    rng = np.random.default_rng(322)
    n, aux = rng.poisson(10, 200_000), rng.poisson(1, 200_000)
    total = n + aux
    h0 = total / 11
    q0 = 2 * (xlogy(n, n / np.maximum(10 * h0, 1e-300)) +
              xlogy(aux, aux / np.maximum(h0, 1e-300)))
    q0[n <= 10 * aux] = 0
    exact = binom.sf(n - 1, total, 10 / 11)
    result['one_bin_exact_control'] = {
        'toys': len(n), 'nominal_background': 10, 'expected_auxiliary_count': 1,
        'zero_fraction': float(np.mean(q0 < 1e-10)),
        'asymptotic_rejection_at_1_2_3_sigma': [float(np.mean(q0 >= z ** 2)) for z in (1, 2, 3)],
        'exact_rejection_at_1_2_3_sigma': [float(np.mean(exact <= norm.sf(z))) for z in (1, 2, 3)],
    }
    (out / 'independent_likelihood.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    np.savez(out / 'template_hllhc.npz', signal=physical_components(180000, conf, by_partition['confirm'], ext_shape)[0],
             parts=physical_components(180000, conf, by_partition['confirm'], ext_shape)[1],
             external=3600 * ext_shape, shape_log_variations=shapes)


if __name__ == '__main__':
    main()
