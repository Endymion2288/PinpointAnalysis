"""Locate a model's class tradeoffs with paired physical-block occupancy strata."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.flavour.inventory import write_json
from analysis.flavour.s3.metrics import bootstrap_weights
from analysis.flavour.v2.partitions import KEY, equal_identity_rows


def common_prediction(frame):
    common = frame[['p0', 'p1', 'p3']].to_numpy()
    if 'p2' in frame:
        common = common/np.maximum(common.sum(1, keepdims=True), 1e-15)
    return common.argmax(1)


def interval(point, samples):
    valid = np.isfinite(samples)
    return {'difference_a_minus_b': float(point),
        '95pct': np.quantile(samples[valid], [.025, .975]).tolist() if valid.any() else [None, None],
        'valid_replicates': int(valid.sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--left', default='final_b1_t3_s42')
    parser.add_argument('--right', default='final_b4_t3_s42')
    args = parser.parse_args()
    catalog = pd.read_parquet(args.root/'data/catalog.parquet', filters=[('split', '=', 'validation')])
    catalog = catalog.sort_values(KEY).reset_index(drop=True)
    frames = [pd.read_parquet(args.root/'experiments'/name/'predictions.parquet').sort_values(KEY).reset_index(drop=True)
              for name in [args.left, args.right]]
    for frame in frames:
        if not equal_identity_rows(frame, catalog) or not np.array_equal(frame.label, catalog.label):
            raise ValueError('occupancy diagnostic requires the same frozen Tune rows and labels')
        if any(not np.array_equal(frame[c], catalog[c])
               for c in ['pixel_channels', 'bar_channels', 'tracker_channels']):
            raise ValueError('occupancy metadata differs from the catalog')
    y = catalog.label.to_numpy()
    target = np.searchsorted([0, 1, 3], y)
    target[y==2] = -1
    correct = [common_prediction(frame)==target for frame in frames]
    delta = correct[0].astype('int8')-correct[1].astype('int8')
    weights, design = bootstrap_weights(catalog, 400, 20261116)
    occupancy = (catalog.pixel_channels+catalog.bar_channels+catalog.tracker_channels).to_numpy()
    strata = [(0, 100), (100, 1000), (1000, 5000), (5000, np.inf)]
    rows, contribution_sum = [], np.zeros((len(weights), 3))
    all_effects = []
    for label in [0, 1, 3]:
        mask = y==label
        all_effects.append(weights[:, mask]@delta[mask]/weights[:, mask].sum(1))
    for low, high in strata:
        mask = (occupancy>=low)&(occupancy<high)
        report = {'occupancy_min_inclusive': low, 'occupancy_max_exclusive': None if np.isinf(high) else high,
            'raw_counts_e_mu_tau_NC': np.bincount(y[mask], minlength=4).tolist(), 'classes': {}}
        class_points, class_samples = [], []
        for c, (label, name) in enumerate([(0, 'e'), (1, 'mu'), (3, 'NC')]):
            keep = mask & (y==label)
            support = int(keep.sum())
            numerator = weights[:, keep]@delta[keep]
            denominator = weights[:, keep].sum(1)
            samples = np.divide(numerator, denominator, out=np.full(len(weights), np.nan), where=denominator>0)
            contribution = numerator/weights[:, y==label].sum(1)
            point = float(delta[keep].mean())
            class_points.append(point)
            class_samples.append(samples)
            contribution_sum[:, c] += contribution
            report['classes'][name] = {'events': support,
                'a_correct': int(correct[0][keep].sum()), 'b_correct': int(correct[1][keep].sum()),
                'a_only_correct': int((keep & correct[0] & ~correct[1]).sum()),
                'b_only_correct': int((keep & correct[1] & ~correct[0]).sum()),
                'recall_difference': interval(point, samples),
                'contribution_to_all_class_recall': interval(float(delta[keep].sum()/(y==label).sum()), contribution)}
        report['BA3_difference'] = interval(np.mean(class_points), np.mean(class_samples, axis=0))
        rows.append(report)
    error = float(np.max(np.abs(contribution_sum-np.asarray(all_effects).T)))
    if error>2e-12:
        raise ValueError('occupancy contributions fail the per-draw all-domain identity')
    expected_all = [float(delta[y==label].mean()) for label in [0, 1, 3]]
    output = {'status': 'complete', 'a': args.left, 'b': args.right,
        'scope': 'post-hoc descriptive Tune localization; one trained seed; occupancy correlates with geometry and acceptance, not a causal effect',
        'occupancy_definition': 'pre-removal accepted pixel_channels+bar_channels+tracker_channels',
        'replicates': 400, 'bootstrap_seed': 20261116, 'bootstrap': design, 'strata': rows,
        'all_class_recall_differences': dict(zip(['e', 'mu', 'NC'], expected_all)),
        'maximum_contribution_identity_error': error,
        'Cal_A_Cut_B_Confirm_payload_read': False}
    path = args.root/'evaluation/occupancy_localization.json'
    path.parent.mkdir(exist_ok=True)
    write_json(path, output)
    print(json.dumps({'status': 'complete', 'output': str(path), 'contribution_identity_error': error}), flush=True)


if __name__=='__main__':
    main()
