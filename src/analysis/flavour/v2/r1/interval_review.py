"""Read back core paired intervals using class-weighted event sufficient statistics."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from analysis.flavour.inventory import write_json
from analysis.flavour.s3.metrics import bootstrap_weights
from analysis.flavour.v2.partitions import KEY, equal_identity_rows


CORE = ['BA3', 'NLL3', 'recall_e', 'recall_mu', 'recall_NC', 'BA4', 'NLL4']


def domain_mask(catalog, domain):
    if domain == 'all':
        return np.ones(len(catalog), dtype=bool)
    if domain.startswith('occupancy_'):
        _, left, right = domain.split('_')
        channels = catalog.pixel_channels+catalog.bar_channels+catalog.tracker_channels
        return channels.ge(float(left)).to_numpy() & channels.lt(float(right)).to_numpy()
    return catalog[domain].to_numpy(dtype=bool)


def class_statistics(labels, probabilities, weights):
    """Return per-draw BA, NLL and recall without calling the evaluation metrics."""
    p = np.asarray(probabilities, dtype='float64')
    predicted = p.argmax(1)
    loss = -np.log(np.clip(p[np.arange(len(labels)), labels], 1e-15, 1.))
    per_class = []
    for c in range(p.shape[1]):
        keep = labels == c
        w = weights[:, keep].astype('float64')
        support = w.sum(1)
        payload = np.column_stack([predicted[keep] == c, loss[keep]])
        sums = w @ payload
        per_class.append(np.divide(sums, support[:, None],
            out=np.full_like(sums, np.nan), where=support[:, None] > 0))
    values = np.stack(per_class, axis=1)
    return np.column_stack([np.nanmean(values[:, :, 0], axis=1),
                            np.nanmean(values[:, :, 1], axis=1), values[:, :, 0]])


def review_intervals(root, config, evaluation):
    started = time.perf_counter()
    catalog = pd.read_parquet(root/'data/catalog.parquet',
        filters=[('split', '=', 'validation')]).sort_values(KEY).reset_index(drop=True)
    # Reuse the frozen physical-block design and RNG draws. All metric and
    # contrast arithmetic below is independent of Metrics/scalar_summary.
    draws, design = bootstrap_weights(catalog, evaluation['replicates'], 20261116)
    weights = np.vstack([np.ones((1, len(catalog)), dtype=draws.dtype), draws])
    needed = sorted({(pair[side], pair['domain'])
                     for pair in config['pairs'] for side in ['a', 'b']})
    names = {entry['name']: entry for entry in config['models']}
    checks, cache = [], {}
    largest = {'point': 0., 'interval_endpoint': 0., 'seed_SD': 0.}

    def check(name, passed, **details):
        checks.append({'name': name, 'passed': bool(passed), **details})

    def same(name, actual, expected, kind):
        if np.isfinite(expected).all():
            observed = np.asarray(actual, dtype='float64')
            error = float(np.max(np.abs(observed-expected)))
            largest[kind] = max(largest[kind], error)
            check(name, bool(np.isfinite(observed).all()) and error < 2e-9,
                  maximum_absolute_error=error)
        else:
            check(name, actual is None if np.ndim(expected) == 0
                  else all(value is None for value in actual))

    for name in sorted({key[0] for key in needed}):
        frame = pd.read_parquet(root/names[name]['predictions']).sort_values(KEY).reset_index(drop=True)
        check(name+'_frozen_Tune', equal_identity_rows(frame, catalog)
              and np.array_equal(frame.label, catalog.label))
        y = frame.label.to_numpy()
        columns = ['p0', 'p1', 'p2', 'p3'] if 'p2' in frame else ['p0', 'p1', 'p3']
        original = frame[columns].to_numpy()
        common = original if len(columns) == 3 else original[:, [0, 1, 3]]
        if len(columns) == 4:
            common = common/np.maximum(common.sum(1, keepdims=True), 1e-15)
        for _, domain in [key for key in needed if key[0] == name]:
            mask = domain_mask(catalog, domain)
            non_tau = mask & (y != 2)
            labels = np.searchsorted([0, 1, 3], y[non_tau])
            three = class_statistics(labels, common[non_tau], weights[:, non_tau])
            four = class_statistics(y[mask], original[mask], weights[:, mask])[:, :2] \
                if len(columns) == 4 else np.full((len(weights), 2), np.nan)
            cache[(name, domain)] = np.column_stack([three, four])

    def contrast(prefix, samples, reported):
        for i, metric in enumerate(CORE):
            row = reported[metric]
            same(prefix+'_'+metric+'_point', row['difference_a_minus_b'], samples[0, i], 'point')
            valid = np.isfinite(samples[1:, i])
            bounds = np.quantile(samples[1:, i][valid], [.025, .975]) \
                if valid.any() else np.full(2, np.nan)
            same(prefix+'_'+metric+'_interval', row['95pct'], bounds, 'interval_endpoint')

    by_pair = {(row['a'], row['b'], row['domain'], row['reason']): row
               for row in evaluation['pairs']}
    groups = defaultdict(list)
    for index, pair in enumerate(config['pairs']):
        identity = tuple(pair[k] for k in ['a', 'b', 'domain', 'reason'])
        samples = cache[(pair['a'], pair['domain'])]-cache[(pair['b'], pair['domain'])]
        contrast(f'pair{index}', samples, by_pair[identity]['metrics'])
        a, b = names[pair['a']], names[pair['b']]
        if a['seed'] == b['seed']:
            group = tuple((entry['family'], entry['task'], entry.get('kind', 'main'))
                          for entry in [a, b])+(pair['domain'], pair['reason'])
            groups[group].append((a['seed'], samples))
    mean_count = 0
    for index, pair in enumerate(evaluation['mean_seed_pairs']):
        key = (tuple(pair['a_family_task_kind']), tuple(pair['b_family_task_kind']),
               pair['domain'], pair['reason'])
        rows = groups[key]
        check(f'mean{index}_exact_seed_order', [seed for seed, _ in rows] == pair['seeds']
              and sorted(pair['seeds']) == [17, 42, 2026])
        values = np.asarray([value for _, value in rows])
        contrast(f'mean{index}', values.mean(0), pair['metrics'])
        for i, metric in enumerate(CORE):
            points = values[:, 0, i]
            same(f'mean{index}_{metric}_seed_points', pair['metrics'][metric]['per_seed_differences'],
                 points, 'point')
            sd = points.std(ddof=1) if np.isfinite(points).all() else np.nan
            same(f'mean{index}_{metric}_seed_SD', pair['metrics'][metric]['training_seed_sample_SD'],
                 sd, 'seed_SD')
        mean_count += 1
    failed = [row['name'] for row in checks if not row['passed']]
    return {'status': 'complete' if not failed else 'failed', 'passed': len(checks)-len(failed),
        'failed': failed, 'checks': checks, 'model_domains': len(cache),
        'pairs': len(config['pairs']), 'mean_seed_pairs': mean_count,
        'metrics': CORE, 'replicates': evaluation['replicates'],
        'largest_absolute_errors': largest, 'bootstrap_design': design,
        'seconds': time.perf_counter()-started,
        'scope': 'independent class-weighted event arithmetic for core points, paired percentile intervals and realized seed SD; same frozen physical-block draws; no independent confirmation',
        'Cal_A_Cut_B_Confirm_payload_read': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--evaluation', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    config = json.loads((args.config or args.root/'evaluation_config.json').read_text())
    evaluation = json.loads((args.evaluation or args.root/'evaluation/results.json').read_text())
    result = review_intervals(args.root, config, evaluation)
    write_json(args.output or args.root/'evaluation/independent_core_intervals.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'checks'}), flush=True)
    if result['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
