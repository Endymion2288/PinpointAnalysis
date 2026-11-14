"""Predeclared exact-target joint overlap diagnostic on the independent test."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.flavour.inventory import write_json
from analysis.flavour.s3.metrics import MetricCache
from analysis.flavour.s5.scope_diagnostics import target_support_weights
from analysis.flavour.s6 import bootstrap
from analysis.flavour.s6.calibration import extract, transform, probability_columns
from analysis.flavour.s6.evaluate import FixedCache, METRIC_NAMES, METRIC_INDEX
from analysis.flavour.s6.prepare import KEY


MODELS = ['mae3_100k_ensemble', 'scratch_100k_ensemble', 's4_repaired_rel_channel5_n100000_s42',
    's3r_multiscale_1mm_s42_e4', 's3r_multiscale_native_s42_e4', 's3r_multiscale_tracker_s42_e4',
    's5_confirm8_direct4_n10000_s42', 's5_confirm8_vertex_kinematics_n10000_s42', 's5_confirm8_six_n10000_s42']


def one(args):
    freeze = json.loads((args.root/'analysis_freeze.json').read_text())
    name = MODELS[args.task]
    _, columns = probability_columns(name, freeze['candidates'])
    raw = pd.read_parquet(args.root/'predictions/test_raw.parquet', columns=KEY+['label']+columns)
    catalog = pd.read_parquet(args.root/'test_catalog.parquet')
    frame = raw[KEY+['label']].merge(catalog.drop(columns=['label']), on=KEY, validate='one_to_one', sort=False)
    weight, summary = target_support_weights(frame)
    keep = weight > 0
    # Zero-weight events do not contribute to this predeclared overlap. Drop
    # them before caching ranking statistics; support/cell definition is fixed.
    raw, weight = raw.loc[keep].reset_index(drop=True), weight[keep]
    matrix, blocks = bootstrap.load(args.root, raw)
    y = raw.label.to_numpy()
    cal = freeze['calibrators'][name]
    p = transform(extract(raw, name, freeze['candidates']), cal['temperature'], [.25]*4)
    cache = MetricCache(y, p)
    fixed = FixedCache(y, p, [v for v in cal['thresholds'] if v['scenario'] == 'balanced'])
    metric_names = METRIC_NAMES+fixed.names
    observed = np.r_[cache.compute(weight)[0][METRIC_INDEX], fixed.vector(weight)]
    samples = []
    for m in matrix:
        w = weight*m[blocks]
        count = np.bincount(y, weights=w, minlength=4)
        w *= np.divide(.25, count, out=np.zeros(4), where=count > 0)[y]
        samples.append(np.r_[cache.compute(w)[0][METRIC_INDEX], fixed.vector(w)])
    samples = np.asarray(samples)
    destination = args.root/'evaluation/support'/name
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination/'bootstrap.npz', names=metric_names, values=samples)
    write_json(destination/'result.json', {'name': name, 'status': 'complete', 'support': summary,
        'positive_weight_events': int(keep.sum()), 'metrics': {n: {'value': float(observed[i]), **bootstrap.ci(samples[:, i])}
            for i, n in enumerate(metric_names)}})


def compare(args):
    reports, arrays = {}, {}
    summary = None
    for name in MODELS:
        path = args.root/'evaluation/support'/name
        result = json.loads((path/'result.json').read_text())
        summary = result['support'] if summary is None else summary
        if result['support'] != summary:
            raise ValueError('fixed common support differs between models')
        reports[name] = result['metrics']
        arrays[name] = np.load(path/'bootstrap.npz')
    primary = 'mae3_100k_ensemble'
    metric_names = arrays[primary]['names']
    contrasts = {}
    for name in MODELS[1:]:
        if not np.array_equal(arrays[name]['names'], metric_names):
            raise ValueError('support metric order differs')
        contrasts[name] = {str(n): {'delta': reports[primary][str(n)]['value']-reports[name][str(n)]['value'],
            **bootstrap.ci(arrays[primary]['values'][:, i]-arrays[name]['values'][:, i])}
            for i, n in enumerate(metric_names)}
    write_json(args.root/'evaluation/exact_target_overlap.json', {'support': summary, 'models': reports,
        'primary_minus_control': contrasts, 'bootstrap': 'fixed accepted-family-connected blocks and fixed joint-cell support; balanced prior normalized per replicate'})


def evaluate(args):
    freeze = json.loads((args.root/'analysis_freeze.json').read_text())
    raw = pd.read_parquet(args.root/'predictions/test_raw.parquet')
    catalog = pd.read_parquet(args.root/'test_catalog.parquet')
    frame = raw[KEY+['label']].merge(catalog.drop(columns=['label']), on=KEY, validate='one_to_one', sort=False)
    weight, summary = target_support_weights(frame)
    matrix, blocks = bootstrap.load(args.root, raw)
    names = ['mae3_100k_ensemble', 'scratch_100k_ensemble', 's4_repaired_rel_channel5_n100000_s42',
        's3r_multiscale_1mm_s42_e4', 's3r_multiscale_native_s42_e4', 's3r_multiscale_tracker_s42_e4',
        's5_confirm8_direct4_n10000_s42', 's5_confirm8_vertex_kinematics_n10000_s42', 's5_confirm8_six_n10000_s42']
    reports, arrays = {}, {}
    y = frame.label.to_numpy()
    for name in names:
        cal = freeze['calibrators'][name]
        p = transform(extract(raw, name, freeze['candidates']), cal['temperature'], [.25]*4)
        cache = MetricCache(y, p)
        cuts = [v for v in cal['thresholds'] if v['scenario'] == 'balanced']
        fixed = FixedCache(y, p, cuts)
        metric_names = METRIC_NAMES+fixed.names
        observed = np.r_[cache.compute(weight)[0][METRIC_INDEX], fixed.vector(weight)]
        samples = []
        for m in matrix:
            w = weight*m[blocks]
            # Preserve the declared balanced prior without re-estimating the
            # overlap cells in each bootstrap replicate.
            count = np.bincount(y, weights=w, minlength=4)
            w *= np.divide(.25, count, out=np.zeros(4), where=count > 0)[y]
            samples.append(np.r_[cache.compute(w)[0][METRIC_INDEX], fixed.vector(w)])
        arrays[name] = np.asarray(samples)
        reports[name] = {n: {'value': float(observed[i]), **bootstrap.ci(arrays[name][:, i])}
            for i, n in enumerate(metric_names)}
    contrasts = {name: {n: {'delta': reports['mae3_100k_ensemble'][n]['value']-reports[name][n]['value'],
        **bootstrap.ci(arrays['mae3_100k_ensemble'][:, i]-arrays[name][:, i])}
        for i, n in enumerate(metric_names)} for name in names if name != 'mae3_100k_ensemble'}
    write_json(args.root/'evaluation/exact_target_overlap.json', {'support': summary, 'models': reports,
        'primary_minus_control': contrasts, 'bootstrap': 'fixed accepted-family-connected blocks and fixed joint-cell support; balanced prior normalized per replicate'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['evaluate', 'one', 'compare'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--task', type=int, default=0)
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == '__main__':
    main()
