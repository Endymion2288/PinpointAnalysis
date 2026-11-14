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
from analysis.flavour.s6.calibration import extract, transform
from analysis.flavour.s6.evaluate import FixedCache, METRIC_NAMES, METRIC_INDEX
from analysis.flavour.s6.prepare import KEY


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
    parser.add_argument('command', choices=['evaluate'])
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == '__main__':
    main()
