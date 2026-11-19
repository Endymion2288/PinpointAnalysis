"""Adversarial checks for hardware isolation, timing, calibration and spectrum."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from scipy.stats import beta, rankdata
import torch

from analysis.flavour.readout import bar_channels
from analysis.flavour.v3.common.domains import compute_observable_containment, detector_bounds
from analysis.flavour.v3.common.layers import ScaledQKNormAttention
from analysis.flavour.v3.v2_pinpointformer.pixel_tokenizer import extract_pf_item
from analysis.flavour.v3.v4_tau.ensemble import _domain_blend_objective_and_grad, _fit_domain_convex_blend
from infer import observe_shard
from recalculate import inputs, rates, scores


def hardware(root, g, b):
    _, events = observe_shard(root / 'data/shards/125', 32)
    nx, ny = len(g['pixel_Xpos']), len(g['pixel_Ypos'])
    col = np.array([6000, 6010])
    row = 4000
    artificial = {
        'pixel': {'channel_id': (12 * ny + row) * nx + col,
                  'xyz_mm': np.column_stack([np.asarray(g['pixel_Xpos'])[col],
                                             np.full(2, g['pixel_Ypos'][row]),
                                             np.full(2, g['pixel_Zpos'][12])])},
        'bar': {'panel': [], 'horizontal': [], 'bar': [], 'bounds_mm': [], 'pe_proxy': []},
        'tracker_hypothesis': {'station': [], 'xy_mm': [], 'sigma_xy_mm': []},
    }
    response = []
    for observation in [artificial, *events]:
        row_result = {}
        for pitch in (None, 0.1, .3, 1.0):
            conf = {'pe_only': True, 'quantize_xy_mm': pitch, 'quantize_target_only': True}
            dom = compute_observable_containment(observation, g, b, config=conf)
            obj = extract_pf_item(0, observation, -1, dom, g, b, conf)
            row_result[str(pitch)] = (int((obj['pix_plane'] >= 12).sum()), int(obj['obs_domain_id']),
                                      obj['pix_xyz_mm'][obj['pix_plane'] >= 12])
        response.append(row_result)
    counter = {'ipt_cluster_counts': {p: x[0] for p, x in response[0].items()},
               'ipt_cluster_centroids': {p: x[2].tolist() for p, x in response[0].items()}}
    counter['real_examples'] = {
        p: {'events_with_ipt_count_change': sum(v[p][0] != v['None'][0] for v in response[1:]),
            'events_with_routing_change': sum(v[p][1] != v['None'][1] for v in response[1:]),
            'total_retained_ipt_tokens': sum(v[p][0] for v in response[1:])}
        for p in ('None', '0.1', '0.3', '1.0')
    }
    record = dict(panelID=np.array([0, 0]), layerID=np.array([0, 0]),
                  isHorizontal=np.array([0, 0]), colID=np.array([0, 0]), rowID=np.array([-1, -1]),
                  nPE=np.array([0, 40]), hitTime=np.array([1., 5.]), edep=np.array([.01, 1.]))
    channels, _ = bar_channels(record, g, include_truth_edep=False)
    counter['timing_counterexample'] = {k: np.asarray(channels[k]).tolist() for k in ('pe_proxy', 'hit_time_ns')}
    return counter


def math_checks():
    rng = np.random.default_rng(322)
    lp = rng.normal(size=(4, 120, 4))
    y = rng.integers(0, 4, 120)
    w = np.full(len(y), 1 / len(y))
    theta = rng.normal(0, .4, size=8)

    def reference(t):
        alpha, inv, bias = t[:4], np.exp(t[4]), np.r_[t[5:], -t[5:].sum()]
        logits = np.einsum('m,mnc->nc', softmax(alpha), lp) * inv + bias
        loss = np.dot(w, logsumexp(logits, axis=1) - logits[np.arange(len(y)), y])
        return loss + 0.5e-4 * (np.var(alpha) * len(alpha) + t[4] ** 2) + 0.5e-3 * (bias @ bias)

    values = []
    for direction in np.eye(len(theta)) * 1e-5:
        values.append((reference(theta + direction) - reference(theta - direction)) / 2e-5)
    value, gradient = _domain_blend_objective_and_grad(theta, lp, y, w, np.arange(len(y)))
    reference_fit = _fit_domain_convex_blend(lp, y)
    target = softmax(np.einsum('m,mnc->nc', reference_fit['weights'], lp) *
                     reference_fit['inv_T'] + reference_fit['bias'], axis=1)
    differences = []
    for perm in itertools.permutations(range(4)):
        v = lp[list(perm)]
        fit = _fit_domain_convex_blend(v, y)
        p = softmax(np.einsum('m,mnc->nc', fit['weights'], v) * fit['inv_T'] + fit['bias'], axis=1)
        differences.append(float(np.max(np.abs(p - target))))
    torch.manual_seed(322)
    layer = ScaledQKNormAttention(dim=16, heads=2, init_scale=2).eval()
    q, k = torch.randn(3, 3, 16), torch.randn(3, 6, 16)
    masks = torch.tensor([[0] * 6, [1, 1, 0, 0, 0, 0], [1] * 6], dtype=torch.bool)
    qmask = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    with torch.no_grad():
        _, actual = layer(q, k, masks, query_mask=qmask, return_entropy=True)
        q_weights = layer.q_proj.weight.numpy()
        k_weights = layer.k_proj.weight.numpy()

    def project(x, weights):
        array = x.numpy().astype(np.float64)
        normal = (array - array.mean(axis=-1, keepdims=True)) / np.sqrt(
            array.var(axis=-1, keepdims=True) + 1e-5)
        return (normal @ weights.T).reshape(len(array), array.shape[1], 2, 8)

    qn, kn = project(q, q_weights), project(k, k_weights)
    expected = np.zeros((3, 2))
    for event in (1, 2):
        for head in range(2):
            v = qn[event, qmask[event], head]
            u = kn[event, masks[event], head]
            dot = v @ u.T / np.outer(np.linalg.norm(v, axis=1), np.linalg.norm(u, axis=1))
            scale = 2 * (1 + np.log1p(np.exp(-2)) * np.log(len(u)))
            p = softmax(scale * dot, axis=1)
            expected[event, head] = np.mean(-np.sum(p * np.log(p), axis=1) / np.log(len(u)))
    return {'calibration_objective_abs_difference': float(abs(value - reference(theta))),
            'calibration_gradient_max_abs_difference': float(np.max(np.abs(gradient - values))),
            'fit_permutations': len(differences), 'permutation_max_score_difference': max(differences),
            'nonuniform_attention_entropy_max_difference': float(np.max(np.abs(actual.numpy()[:, :, 0] - expected))),
            'tracker_sigma_y_with_grid_mm': float(np.hypot(.017, .1 / np.sqrt(12))),
            'tracker_single_triplet_sagitta_sigma_mm': float(np.sqrt(1.5) * np.hypot(.017, .1 / np.sqrt(12))),
            'zero_pass_500_upper_95pct': float(beta.ppf(.95, 1, 500))}


def sample_audit(root):
    data = pd.read_parquet(root / 'planned_manifest.parquet')
    data['source_type'] = data.source_tag.str.split(':').str[0]
    data['source_label_count'] = data.groupby('source_tag').label.transform('nunique')
    result = {
        'sources_crossing_splits': int((data.groupby('source_tag').split.nunique() > 1).sum()),
        'total_sources': int(data.source_tag.nunique()),
        'event_fraction_label_unique_given_source': float((data.source_label_count == 1).mean()),
        'counts_by_source_and_class': pd.crosstab(data.source_type, data.label).to_dict(),
        'energy_GeV_quantiles_by_class': data.groupby('label').Ev_GeV.quantile([.1, .5, .9]).unstack().to_numpy().tolist(),
    }
    conf = pd.read_parquet(root / 'sealed_truth_meta/confirm_truth_meta.parquet')
    veto = conf.p0_pixel_hits.to_numpy() >= 1
    active = conf.large_scintillator.to_numpy()
    front = conf.front_pinpoint_pixel_centered.to_numpy()
    w = conf.geom_weight.to_numpy()
    y = conf.label.to_numpy()
    result['veto'] = {key: {str(c): float(np.average(veto[m & (y == c)], weights=w[m & (y == c)]))
                            for c in range(4)}
                      for key, m in [('all', w > 0), ('fid', active), ('front', front)]}
    result['nominal_tau_unweighted_front_p0_losses'] = [int(np.sum(veto & front & (y == 2))), int(np.sum(front & (y == 2)))]
    ext = pd.read_parquet(root / 'sys_shards/external_punchthrough/truth_meta.parquet')
    result['external_veto_counts'] = {name: [int((rows.p0_pixel_hits > 0).sum()), len(rows)]
                                      for name, rows in ext.groupby('ext_subtype')}
    return result


def observable_baseline(root, geometry, bounds):
    # Fixed, evenly spaced shards from each split avoid a sequential block selection.
    manifest = pd.read_parquet(root / 'planned_manifest.parquet', columns=['split', 'chunk', 'row', 'label'])
    split_chunks = {}
    for split, count in [('train', 6), ('confirm', 4)]:
        chunks = np.sort(manifest.loc[manifest.split == split, 'chunk'].unique())
        split_chunks[split] = chunks[np.linspace(0, len(chunks) - 1, count, dtype=int)]

    def features(chunk):
        keys, observations = observe_shard(root / 'data/shards' / f'{chunk:03d}')
        labels = keys.merge(manifest[['chunk', 'row', 'label']], on=['chunk', 'row'],
                            validate='one_to_one')['label'].to_numpy()
        config = {'pe_only': True}
        covariates = [compute_observable_containment(event, geometry, bounds, config=config)['c_obs']
                      for event in observations]
        return np.stack(covariates), labels

    with ThreadPoolExecutor(max_workers=4) as pool:
        train, test = [list(pool.map(features, split_chunks[split])) for split in ('train', 'confirm')]
    train_x, train_y = (np.concatenate(cols) for cols in zip(*train))
    test_x, test_y = (np.concatenate(cols) for cols in zip(*test))
    result = {split + '_shards': chunks.tolist() for split, chunks in split_chunks.items()}
    result.update({'train_events': len(train_y), 'confirm_events': len(test_y)})
    groups = {'presence_and_counts': [0, 1, 2, 3, 13, 14], 'all_observable_containment': list(range(16))}
    for group, columns in groups.items():
        mean = train_x[:, columns].mean(axis=0)
        scale = np.maximum(train_x[:, columns].std(axis=0), 1e-6)
        x = np.column_stack([np.ones(len(train_x)), (train_x[:, columns] - mean) / scale])
        v = np.column_stack([np.ones(len(test_x)), (test_x[:, columns] - mean) / scale])
        t = np.eye(4)[train_y]

        def objective(theta):
            weights = theta.reshape(x.shape[1], 4)
            logits = x @ weights
            loss = np.mean(logsumexp(logits, axis=1) - np.sum(t * logits, axis=1))
            gradient = x.T @ (softmax(logits, axis=1) - t) / len(x)
            return loss + .0005 * np.sum(weights ** 2), (gradient + .001 * weights).ravel()

        fit = minimize(objective, np.zeros(x.shape[1] * 4), jac=True, method='L-BFGS-B')
        probability = softmax(v @ fit.x.reshape(v.shape[1], 4), axis=1)
        prediction = probability.argmax(axis=1)
        signal = test_y == 2
        ranks = rankdata(probability[:, 2])
        n_signal = int(signal.sum())
        auc = (ranks[signal].sum() - n_signal * (n_signal + 1) / 2) / (n_signal * (~signal).sum())
        result[group] = {'ba4': float(np.mean([np.mean(prediction[test_y == c] == c) for c in range(4)])),
                         'tau_auc': float(auc), 'converged': bool(fit.success)}
    return result


def compare_groups(root, out):
    manifest = json.loads((root / 'v7_freeze_manifest.json').read_text())['checkpoints']
    groups = ('v2_pf_pitch_20um', 'v2_pf_pitch_1mm_tgtonly', 'v2_pf_6layer_20um',
              'v2_pf_hybrid_uvtx', 'v2_pf_hybrid_shuffled', 'v2_pf_hybrid_nomad')
    models, metrics = {}, {}
    for prefix in groups:
        cal, lp, _, _ = inputs(root, out, 'cal_a', manifest, prefix)
        conf, lq, charm, vtx = inputs(root, out, 'confirm', manifest, prefix)
        par = {}
        for d in (1, 2, 3):
            mask = cal.obs_domain_id.to_numpy() == d
            par[str(d)] = _fit_domain_convex_blend(lp[:, mask], cal.label.to_numpy()[mask])
        probability, _ = scores(conf, lq, charm, par, {str(d): 0 for d in range(1, 4)})
        p_mask = conf.pixel_centered.to_numpy()
        y = conf.label.to_numpy()[p_mask]
        pv = conf.loc[p_mask, ['vertex_x_mm', 'vertex_y_mm', 'vertex_z_mm']].to_numpy()
        residual = vtx[p_mask] - pv
        predict = probability[p_mask].argmax(axis=1)
        p_tau = probability[:, 2]
        neg = p_mask & (conf.label.to_numpy() != 2)
        threshold = np.sort(p_tau[neg])[-int(.01 * neg.sum())]
        models[prefix] = residual, predict
        metrics[prefix] = {
            'BA4': float(np.mean([np.mean(predict[y == c] == c) for c in range(4)])),
            'pv_xyz_median_absolute_mm': np.median(np.abs(residual), axis=0).tolist(),
            'pv_3d_median_mm': float(np.median(np.linalg.norm(residual, axis=1))),
            'confirm_ranked_p_tau_1pct': rates(conf.label.to_numpy(), p_mask, p_tau, threshold),
        }
    rng = np.random.default_rng(322)
    contrast = {}
    for first, second in ((0, 1), (2, 0), (3, 0), (3, 4), (5, 3)):
        va, ya = models[groups[first]]
        vb, yb = models[groups[second]]
        out_arr = np.zeros((1000, 4))
        for i in range(len(out_arr)):
            k = rng.integers(0, len(y), len(y))
            out_arr[i, :3] = np.median(np.abs(va[k]), axis=0) - np.median(np.abs(vb[k]), axis=0)
            out_arr[i, 3] = np.mean([np.mean(ya[k][y[k] == c] == c) - np.mean(yb[k][y[k] == c] == c)
                                     for c in range(4)])
        contrast[groups[first] + ' - ' + groups[second]] = np.quantile(out_arr, [.025, .5, .975], axis=0).tolist()
    return metrics, contrast


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    g = json.loads((args.root / 'geometry.json').read_text())
    torch.set_num_threads(2)
    summary = {'hardware': hardware(args.root, g, detector_bounds(g)), 'math': math_checks(),
               'data': sample_audit(args.root)}
    (args.output / 'counterexamples.json').write_text(json.dumps(summary, indent=2))
    metrics, comparisons = compare_groups(args.root, args.output)
    summary.update({'groups': metrics, 'paired_bootstrap': comparisons,
                    'observable_baseline': observable_baseline(args.root, g, detector_bounds(g))})
    (args.output / 'counterexamples.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
