"""Paired condition deltas and independent matched generator comparisons."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.s3.metrics import MetricCache
from analysis.flavour.s6 import bootstrap
from analysis.flavour.s6.calibration import extract, transform
from analysis.flavour.s6.evaluate import FixedCache, METRIC_INDEX, METRIC_NAMES, reliability
from analysis.flavour.s6.prepare import KEY


def probability(frame, name, campaign, calibrators, prior):
    return transform(extract(frame, name, campaign), calibrators[name]['temperature'], prior)


def evaluate(args):
    started = time.monotonic()
    freeze = json.loads((args.root/'analysis_freeze.json').read_text())
    campaign, calibrators = freeze['candidates'], freeze['calibrators']
    main = 'mae3_100k_ensemble'
    members = next(e['members'] for e in campaign['ensembles'] if e['name'] == main)
    raw = pd.read_parquet(args.root/'predictions'/f'{args.scope}_raw.parquet')
    condition_key = 'variant' if args.scope == 'variants' else 'condition'
    conditions = sorted(raw[condition_key].unique())
    condition = conditions[args.task]
    current = raw[raw[condition_key].eq(condition)].copy().reset_index(drop=True)
    reference = raw[raw[condition_key].eq('nominal')].copy().reset_index(drop=True)
    if args.scope != 'generator':
        reference = current[KEY].merge(reference, on=KEY, how='left', validate='one_to_one', sort=False)
        if reference.label.isna().any():
            raise ValueError('paired condition reference missing')
        multiplicity, blocks = bootstrap.load(args.root, current)
    result = {'scope': args.scope, 'condition': condition, 'events': len(current),
        'class_counts': np.bincount(current.label, minlength=4).tolist(), 'scenarios': {},
        'thresholds_retuned': False, 'hardware_amplitude_calibrated': False}
    y = current.label.to_numpy()
    for scenario in calibrators[main]['priors']:
        name, prior = scenario['name'], scenario['prior']
        points = [p for p in calibrators[main]['thresholds'] if p['scenario'] == name]
        p = probability(current, main, campaign, calibrators, prior)
        p0 = probability(reference, main, campaign, calibrators, prior)
        cache, base = MetricCache(y, p), MetricCache(reference.label.to_numpy(), p0)
        fixed, fixed0 = FixedCache(y, p, points), FixedCache(reference.label.to_numpy(), p0, points)
        w = bootstrap.weights(y, prior)
        observed = np.r_[cache.compute(w)[0][METRIC_INDEX], fixed.vector(w)]
        before = np.r_[base.compute(w)[0][METRIC_INDEX], fixed0.vector(w)]
        names = METRIC_NAMES+fixed.names
        samples = []
        for r in range(len(multiplicity)):
            weight = bootstrap.weights(y, prior, multiplicity[r, blocks])
            samples.append(np.r_[cache.compute(weight)[0][METRIC_INDEX], fixed.vector(weight)]-
                np.r_[base.compute(weight)[0][METRIC_INDEX], fixed0.vector(weight)])
        samples = np.asarray(samples)
        seed_deltas = []
        for member in members:
            score = probability(current, member, campaign, calibrators, prior)
            score0 = probability(reference, member, campaign, calibrators, prior)
            cut = [t for t in calibrators[member]['thresholds'] if t['scenario'] == name]
            values = np.r_[MetricCache(y, score).compute(w)[0][METRIC_INDEX], FixedCache(y, score, cut).vector(w)]
            values0 = np.r_[MetricCache(y, score0).compute(w)[0][METRIC_INDEX], FixedCache(y, score0, cut).vector(w)]
            seed_deltas.append(values-values0)
        seed_deltas = np.asarray(seed_deltas)
        migration = np.bincount(p0.argmax(1)*4+p.argmax(1), minlength=16).reshape(4, 4)
        result['scenarios'][name] = {'metrics': {n: {'value': float(observed[i]), 'nominal_value': float(before[i]),
            'delta': float(observed[i]-before[i]), **bootstrap.ci(samples[:, i]),
            'training_seed_deltas': seed_deltas[:, i].tolist(), 'training_seed_sample_SD': float(seed_deltas[:, i].std(ddof=1))}
            for i, n in enumerate(names)}, 'fixed_raw_counts': fixed.counts(w),
            'probability_mean_absolute_delta': np.abs(p-p0).mean(0).tolist(),
            'probability_delta_quantiles': np.quantile(p-p0, [.01, .5, .99], axis=0).tolist(),
            'class_migration_from_nominal': migration.tolist(), 'Neff': float(w.sum()**2/(w@w)),
            'reliability': reliability(y, p, w, freeze['evaluation']['reliability_bins'])}
    destination = args.root/'evaluation/conditions'/args.scope
    write_json(destination/(condition+'.json'), {**result, 'status': 'complete', 'seconds': time.monotonic()-started})
    print(json.dumps({'scope': args.scope, 'condition': condition, 'seconds': time.monotonic()-started}), flush=True)


def generator(args):
    freeze = json.loads((args.root/'analysis_freeze.json').read_text())
    campaign, calibrators = freeze['candidates'], freeze['calibrators']
    raw = pd.read_parquet(args.root/'predictions/generator_raw.parquet')
    name = 'mae3_100k_ensemble'
    # Independent generator source-file blocks within each matched benchmark
    # cell. Shared benchmark vertex slots do not make hard scatters paired.
    raw['cell'] = raw.benchmark_cell
    tunes = ['Pythia6_Standard', 'Pythia6_NUX']
    priors = calibrators[name]['priors']
    result = {'comparison': tunes, 'hard_scatter_events_paired': False, 'benchmark_cells_matched': True,
        'class_counts': raw.groupby(['tune', 'label']).size().rename('events').reset_index().to_dict('records'),
        'generator_input': json.loads((args.root/'generator/inventory.json').read_text()), 'scenarios': {}}
    rng = np.random.default_rng(2026)
    multiplicities = {}
    for tune in tunes:
        f = raw[raw.tune.eq(tune)]
        source_id, source = pd.factorize(f.source_path)
        sampled = np.array([np.bincount(rng.integers(len(source), size=len(source)), minlength=len(source)) for _ in range(400)])
        multiplicities[tune] = sampled[:, source_id]
    for scenario in priors:
        prior, scenario_name = scenario['prior'], scenario['name']
        points = [v for v in calibrators[name]['thresholds'] if v['scenario'] == scenario_name]
        reports, samples = {}, {}
        names = None
        for tune in tunes:
            frame = raw[raw.tune.eq(tune)].reset_index(drop=True)
            y = frame.label.to_numpy()
            p = probability(frame, name, campaign, calibrators, prior)
            cache, fixed = MetricCache(y, p), FixedCache(y, p, points)
            names = METRIC_NAMES+fixed.names
            weight = bootstrap.weights(y, prior)
            values = np.r_[cache.compute(weight)[0][METRIC_INDEX], fixed.vector(weight)]
            reports[tune] = {'metrics': dict(zip(names, values.tolist())), 'fixed_raw_counts': fixed.counts(weight),
                'Neff': float(weight.sum()**2/(weight@weight)), 'weighted_reliability': reliability(y, p, weight),
                'charm_subsets': {str(flag): {'events': int(frame.charm_primary.eq(flag).sum()),
                    'raw_tau_false_positives_at_1pct': int(((y != 2) & frame.charm_primary.eq(flag).to_numpy() & fixed.passed[0][:, 2]).sum())}
                    for flag in [False, True]}}
            values_sample = []
            for r in range(400):
                m = multiplicities[tune][r].astype(float)
                # Restore the common coarse-cell distribution separately in
                # each generator resample, without event correspondence.
                for _, ix in frame.groupby('cell').indices.items():
                    if m[ix].sum():
                        m[ix] *= len(ix)/m[ix].sum()
                w = bootstrap.weights(y, prior, m)
                values_sample.append(np.r_[cache.compute(w)[0][METRIC_INDEX], fixed.vector(w)])
            samples[tune] = np.asarray(values_sample)
        a, b = tunes
        result['scenarios'][scenario_name] = {'tunes': reports, 'NUX_minus_Standard': {
            n: {'delta': reports[b]['metrics'][n]-reports[a]['metrics'][n],
                **bootstrap.ci(samples[b][:, i]-samples[a][:, i])} for i, n in enumerate(names)},
            'interval_scope': 'independent generator source-file block bootstrap conditional on common benchmark support; each cell reweighted to nominal occupancy when present'}
    write_json(args.root/'evaluation/conditions/generator_comparison.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['evaluate', 'generator'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--scope', choices=['variants', 'transport'], default='variants')
    parser.add_argument('--task', type=int, default=0)
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == '__main__':
    main()
