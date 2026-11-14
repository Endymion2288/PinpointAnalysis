"""Independent test statistics with frozen calibration and validation cuts."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import beta
from sklearn.metrics import roc_curve, precision_recall_curve

from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.s3.metrics import MetricCache, NAMES
from analysis.flavour.s6 import bootstrap
from analysis.flavour.s6.calibration import extract, model_names, probability_columns, selection, transform
from analysis.flavour.s6.prepare import KEY, require_freeze


METRIC_NAMES = [n for n in NAMES if 'eff_at_' not in n]
METRIC_INDEX = [NAMES.index(n) for n in METRIC_NAMES]


def exact_interval(passed, total):
    if not total:
        return [None, None]
    return [float(beta.ppf(.025, passed, total-passed+1)) if passed else 0.,
            float(beta.ppf(.975, passed+1, total-passed)) if passed < total else 1.]


def report(cache, weight):
    result = cache.report(weight)
    return {k: v for k, v in result.items() if 'eff_at_' not in k}


def reliability(y, p, weight, bins=20):
    output = {}
    for name, score, outcome in [('top_label', p.max(1), p.argmax(1) == y)]+[
        (f'class_{c}', p[:, c], y == c) for c in range(4)]:
        index = np.minimum((score*bins).astype(int), bins-1)
        count = np.bincount(index, minlength=bins)
        mass = np.bincount(index, weights=weight, minlength=bins)
        square = np.bincount(index, weights=weight**2, minlength=bins)
        acc = np.divide(np.bincount(index, weights=weight*outcome, minlength=bins), mass, out=np.zeros(bins), where=mass > 0)
        confidence = np.divide(np.bincount(index, weights=weight*score, minlength=bins), mass, out=np.zeros(bins), where=mass > 0)
        output[name] = {'edges': np.linspace(0, 1, bins+1).tolist(), 'counts': count.tolist(),
            'weight': mass.tolist(), 'accuracy': acc.tolist(), 'confidence': confidence.tolist(),
            'Neff': np.divide(mass**2, square, out=np.zeros(bins), where=square > 0).tolist(),
            'ECE': float(np.dot(mass, np.abs(acc-confidence))/mass.sum())}
    return output


class FixedCache:
    def __init__(self, y, probability, points):
        self.y = y
        self.points = points
        self.passed = []
        self.final = []
        self.names = []
        for point in points:
            fpr = point['target_FPR']
            cuts = [v['threshold'] for v in point['classes']]
            self.passed.append(probability >= np.asarray(cuts))
            self.final.append(selection(probability, cuts))
            for c in range(4):
                self.names += [f'class_{c}_{n}@{fpr:g}' for n in ['efficiency', 'FPR', 'purity', 'resolved_efficiency']]
            self.names += [f'reject_rate@{fpr:g}']

    def vector(self, weight):
        output = []
        ratio = lambda a, b: float(a/b) if b else np.nan
        for passed, final in zip(self.passed, self.final):
            for c in range(4):
                signal = self.y == c
                tp = weight[signal & passed[:, c]].sum()
                fp = weight[~signal & passed[:, c]].sum()
                output += [ratio(tp, weight[signal].sum()), ratio(fp, weight[~signal].sum()), ratio(tp, tp+fp),
                           ratio(weight[signal & (final == c)].sum(), weight[signal].sum())]
            output += [ratio(weight[final == 4].sum(), weight.sum())]
        return np.asarray(output)

    def counts(self, weight):
        output = []
        for point, passed, final in zip(self.points, self.passed, self.final):
            classes = []
            for c in range(4):
                signal = self.y == c
                tp, fp = int((signal & passed[:, c]).sum()), int((~signal & passed[:, c]).sum())
                nsignal, nbackground = int(signal.sum()), int((~signal).sum())
                classes.append({'class': c, 'threshold': point['classes'][c]['threshold'],
                    'signal_passed': tp, 'signal_events': nsignal, 'background_passed': fp, 'background_events': nbackground,
                    'signal_efficiency_exact_CI95': exact_interval(tp, nsignal),
                    'raw_background_FPR_exact_CI95': exact_interval(fp, nbackground),
                    'background_passed_below_25': fp < 25,
                    'selected_Neff': float(weight[passed[:, c]].sum()**2/(weight[passed[:, c]]@weight[passed[:, c]]))
                        if passed[:, c].any() and (weight[passed[:, c]]@weight[passed[:, c]]) > 0 else 0.,
                    'background_class_counts': [int(((self.y == k) & passed[:, c]).sum()) if k != c else None for k in range(4)]})
            raw = np.bincount(self.y*5+final, minlength=20).reshape(4, 5)
            weighted = np.bincount(self.y*5+final, weights=weight, minlength=20).reshape(4, 5)
            output.append({'target_FPR': point['target_FPR'], 'classes': classes,
                'resolved_confusion_raw_4x5': raw.tolist(), 'resolved_confusion_weighted_4x5': weighted.tolist(),
                'raw_rejected': int((final == 4).sum()),
                'conflicting_passes': int((passed.sum(1) > 1).sum())})
        return output


def joined(root, raw):
    catalog = pd.read_parquet(root/'test_catalog.parquet')
    truth = pd.read_parquet(root/'test_truth.parquet')
    frame = raw.merge(catalog.drop(columns=[c for c in raw if c not in KEY and c in catalog]), on=KEY,
        how='left', validate='one_to_one', sort=False)
    columns = KEY+[c for c in truth if c not in frame]
    frame = frame.merge(truth[columns], on=KEY, how='left', validate='one_to_one', sort=False)
    quality_path = root/'evaluation/quality.parquet'
    if quality_path.exists():
        frame = frame.merge(pd.read_parquet(quality_path), on=KEY, how='left', validate='one_to_one', sort=False)
    plan = json.loads((root/'plan.json').read_text())
    geometry = json.loads(Path(plan['geometry']).read_text())
    # Declared target envelope is distinct from the much smaller pixel field.
    x, y, z = [frame['init'+a].to_numpy() for a in 'XYZ']
    cx, cy = np.mean(geometry['scint_bar_Xcenter']), np.mean(geometry['scint_bar_Ycenter'])
    frame['target_envelope'] = ((np.abs(x-cx) <= geometry['scint_detector_width']/2) &
        (np.abs(y-cy) <= geometry['scint_detector_height']/2) &
        (z >= min(geometry['tungsten_Zpos'])-geometry['tungsten_thickness']/2) &
        (z <= max(geometry['tungsten_Zpos'])+geometry['tungsten_thickness']/2))
    frame['W_z_slab'] = np.abs(z[:, None]-np.asarray(geometry['tungsten_Zpos'])).min(1) <= geometry['tungsten_thickness']/2
    return frame


def strata(frame, config):
    groups = {'all': np.ones(len(frame), dtype=bool), 'pixel_centered': frame.pixel_centered.to_numpy(),
        'target_envelope': frame.target_envelope.to_numpy(), 'target_envelope_and_W_z_slab':
            frame.target_envelope.to_numpy() & frame.W_z_slab.to_numpy(),
        'large_scintillator': frame.large_scintillator.to_numpy(), 'active': ~frame.all_empty.to_numpy(),
        'empty_response': frame.all_empty.to_numpy(), 'neutrino': frame.initPDG.to_numpy() > 0,
        'antineutrino': frame.initPDG.to_numpy() < 0}
    for name, key, edges, factor in [('energy_GeV', 'initE', config['energy_edges_GeV'], .001),
        ('vertex_z_mm', 'initZ', config['vertex_z_edges_mm'], 1.),
        ('pixel_occupancy', 'pixel_channels', config['pixel_occupancy_edges'], 1.)]:
        values = frame[key].to_numpy()*factor
        for lo, hi in zip(edges[:-1], edges[1:]):
            groups[f'{name}_[{lo},{hi})'] = (values >= lo) & (values < hi)
    for key in ['tau_branch', 'tau_decay_location', 'charm_branch', 'charm_presence', 'tau_decay_contained']:
        if key not in frame:
            continue
        for value in frame[key].dropna().unique():
            if key.startswith('tau_'):
                # Keep ALL non-tau background for every tau conditional signal.
                groups[f'{key}={value}'] = frame.label.ne(2).to_numpy() | frame[key].eq(value).to_numpy()
            else:
                groups[f'{key}={value}'] = frame[key].eq(value).to_numpy()
    for modality in ['pixel', 'bar', 'tracker']:
        groups[f'{modality}_present'] = frame[f'{modality}_channels'].to_numpy() > 0
        groups[f'{modality}_absent'] = frame[f'{modality}_channels'].to_numpy() == 0
    if 'tracker_stations' in frame:
        for n in range(4):
            groups[f'tracker_stations={n}'] = frame.tracker_stations.eq(n).to_numpy()
        for key in ['pixel_segments', 'tracker_associations', 'vertex_candidate']:
            groups[f'{key}_present'] = frame[key].to_numpy() > 0
            groups[f'{key}_absent'] = frame[key].to_numpy() == 0
        groups['pixel_edge_fraction_gt_5pct'] = frame.pixel_edge_fraction.to_numpy() > .05
        groups['last_pixel_plane_active'] = frame.pixel_last_plane_fraction.to_numpy() > 0
        groups['last_bar_panel_active'] = frame.bar_last_panel_energy_fraction.to_numpy() > 0
    if 'model_active' in frame:
        groups['model_active'] = frame.model_active.to_numpy()
        groups['model_empty'] = ~frame.model_active.to_numpy()
    return groups


def fixed_block_intervals(y, fixed, prior, multiplicity, blocks):
    """Exact frozen-selection block resamples from small sufficient counts.

    Aggregate by true class before normalizing each replicate to its declared
    prior. This is equivalent to event weights and avoids a 400 x N array.
    """
    b = multiplicity.shape[1]
    index = blocks*4+y
    aggregate = lambda value: np.bincount(index, weights=value, minlength=b*4).reshape(b, 4)
    totals = multiplicity @ aggregate(np.ones(len(y)))
    scales = np.divide(prior, totals, out=np.zeros_like(totals, dtype=float), where=totals > 0)
    masses = totals*scales
    ratio = lambda a, d: np.divide(a, d, out=np.full_like(a, np.nan, dtype=float), where=d > 0)
    columns = []
    for passed, final in zip(fixed.passed, fixed.final):
        for c in range(4):
            accepted = (multiplicity @ aggregate(passed[:, c]))*scales
            signal = accepted[:, c]
            total = accepted.sum(1)
            resolved = (multiplicity @ aggregate(final == c))*scales
            columns.extend([ratio(signal, masses[:, c]), ratio(total-signal, masses.sum(1)-masses[:, c]),
                ratio(signal, total), ratio(resolved[:, c], masses[:, c])])
        rejected = ((multiplicity @ aggregate(final == 4))*scales).sum(1)
        columns.append(ratio(rejected, masses.sum(1)))
    return {name: bootstrap.ci(value) for name, value in zip(fixed.names, columns)}


def conditional(frame, probability, prior, points, config, multiplicity=None, blocks=None):
    output = []
    y = frame.label.to_numpy()
    for name, mask in strata(frame, config).items():
        count = np.bincount(y[mask], minlength=4)
        if not mask.any():
            continue
        p, labels = probability[mask], y[mask]
        w = bootstrap.weights(labels, prior)
        metrics = report(MetricCache(labels, p), w)
        fixed = FixedCache(labels, p, points)
        output.append({'stratum': name, 'class_counts': count.tolist(),
            'prior_mass_present': float(np.asarray(prior)[count > 0].sum()),
            'metrics': metrics, 'fixed': dict(zip(fixed.names, fixed.vector(w))),
            'fixed_raw_counts': fixed.counts(w),
            **({'fixed_block_intervals': fixed_block_intervals(labels, fixed, prior, multiplicity, blocks[mask])}
               if multiplicity is not None else {})})
    return output


def curves(y, p, w, destination):
    for c in range(4):
        fpr, tpr, thresholds = roc_curve(y == c, p[:, c], sample_weight=w, drop_intermediate=False)
        pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'threshold': thresholds}).to_parquet(
            destination/f'roc_class{c}.parquet', index=False, compression=COMPRESSION)
        precision, recall, thresholds = precision_recall_curve(y == c, p[:, c], sample_weight=w, drop_intermediate=False)
        pd.DataFrame({'precision': precision, 'recall': recall, 'threshold': np.r_[thresholds, np.nan]}).to_parquet(
            destination/f'pr_class{c}.parquet', index=False, compression=COMPRESSION)


def one(args):
    require_freeze(args.root, 'test')
    started = time.monotonic()
    freeze = json.loads((args.root/'analysis_freeze.json').read_text())
    campaign, config = freeze['candidates'], freeze['evaluation']
    names = model_names(campaign)
    name, scenario = names[args.task//4], campaign['prior_scenarios'][args.task % 4]['name']
    cal = freeze['calibrators'][name]
    prior = next(s['prior'] for s in cal['priors'] if s['name'] == scenario)
    points = [p for p in cal['thresholds'] if p['scenario'] == scenario]
    _, columns = probability_columns(name, campaign)
    raw = pd.read_parquet(args.root/'predictions/test_raw.parquet', columns=KEY+['label']+columns)
    p_raw = extract(raw, name, campaign)
    p = transform(p_raw, cal['temperature'], prior)
    y = raw.label.to_numpy()
    cache, weight = MetricCache(y, p), bootstrap.weights(y, prior)
    fixed = FixedCache(y, p, points)
    destination = args.root/'evaluation/models'/name/scenario
    destination.mkdir(parents=True, exist_ok=True)
    scores = raw[KEY+['label']].copy()
    for c in range(4):
        scores[f'raw_p{c}'], scores[f'p{c}'] = p_raw[:, c], p[:, c].astype('float32')
    scores.to_parquet(destination/'scores.parquet', index=False, compression=COMPRESSION)
    frame = joined(args.root, raw)
    members, _ = probability_columns(name, campaign)
    candidate = next(m for m in campaign['models'] if m['name'] == members[0])
    modalities = candidate['config'].get('modalities', ['pixel', 'bar'])
    frame['model_active'] = np.logical_or.reduce([frame[m+'_channels'].to_numpy() > 0 for m in modalities])
    result = {'name': name, 'scenario': scenario, 'prior': prior, 'temperature': cal['temperature'],
        'metrics': report(cache, weight), 'raw_balanced_metrics': report(MetricCache(y, p_raw), bootstrap.weights(y, [.25]*4)),
        'class_counts': np.bincount(y, minlength=4).tolist(), 'confusion_raw_counts':
            np.bincount(y*4+p.argmax(1), minlength=16).reshape(4, 4).tolist(),
        'fixed': dict(zip(fixed.names, fixed.vector(weight))), 'fixed_raw_counts': fixed.counts(weight),
        'reliability': reliability(y, p, weight, config['reliability_bins']),
        'reliability_before_temperature': reliability(y, transform(p_raw, 1., prior), weight, config['reliability_bins']),
        'Neff': float(weight.sum()**2/(weight@weight)),
        'weight_quantiles': np.quantile(weight, [0, .01, .5, .99, 1]).tolist()}
    result['activity_definition'] = {'detector_active': 'any pixel/bar/tracker measurement',
        'model_active': 'any supplied modality measurement', 'supplied_modalities': modalities,
        'model_active_events': int(frame.model_active.sum())}
    matrix, blocks = bootstrap.load(args.root, raw)
    sample_names = METRIC_NAMES+fixed.names
    samples = np.zeros((len(matrix), len(sample_names)))
    for r in range(len(matrix)):
        w = bootstrap.weights(y, prior, matrix[r, blocks])
        samples[r] = np.r_[cache.compute(w)[0][METRIC_INDEX], fixed.vector(w)]
    np.savez_compressed(destination/'bootstrap.npz', names=sample_names, values=samples)
    result['intervals'] = {n: bootstrap.ci(samples[:, i]) for i, n in enumerate(sample_names)}
    result['conditional'] = conditional(frame, p, prior, points, config, matrix, blocks)
    cutflow = []
    for point, resolved in zip(points, fixed.final):
        active = ~frame.all_empty.to_numpy()
        target = frame.pixel_centered.to_numpy()
        envelope = frame.target_envelope.to_numpy()
        for n, mask in [('all', np.ones(len(frame), bool)), ('target_envelope', envelope),
            ('target_envelope_and_active', envelope & active),
            ('target_envelope_active_selected', envelope & active & (resolved != 4)), ('pixel_centered', target),
            ('target_envelope_and_model_active', envelope & frame.model_active.to_numpy()),
            ('target_envelope_model_active_selected', envelope & frame.model_active.to_numpy() & (resolved != 4)),
            ('pixel_centered_and_active', target & active),
            ('pixel_centered_active_and_frozen_selection', target & active & (resolved != 4))]:
            cutflow.append({'target_FPR': point['target_FPR'], 'cut': n,
                'class_counts': np.bincount(y[mask], minlength=4).tolist(),
                'weighted_mass': float(weight[mask].sum()), 'correctly_resolved': int(((resolved == y) & mask).sum())})
    result['cutflow'] = cutflow
    curves(y, p, weight, destination)
    result.update(status='complete', events=len(raw), seconds=time.monotonic()-started,
        scope='frozen independent test; intervals conditional on calibrator, thresholds and checkpoints',
        exposure='unmeasured', thresholds_selected_on_test=False)
    write_json(destination/'result.json', result)
    print(json.dumps({'name': name, 'scenario': scenario, 'BA': result['metrics']['balanced_accuracy'],
        'NLL': result['metrics']['NLL'], 'seconds': result['seconds']}), flush=True)


def design(args):
    frame = pd.read_parquet(args.root/'test_catalog.parquet')
    freeze = json.loads((args.root/'analysis_freeze.json').read_text())
    summary = bootstrap.design(args.root, frame, freeze['statistics']['replicates'], freeze['statistics']['seed'])
    print(json.dumps(summary), flush=True)


def compare(args):
    freeze = json.loads((args.root/'analysis_freeze.json').read_text())
    names = model_names(freeze['candidates'])
    rows, contrasts = [], []
    for name in names:
        for scenario in ['balanced', 'production_sampling', 'tau_1_percent', 'tau_0p1_percent']:
            path = args.root/'evaluation/models'/name/scenario
            result = json.loads((path/'result.json').read_text())
            rows.append({'name': name, 'scenario': scenario, **result['metrics'], **result['fixed']})
    pd.DataFrame(rows).to_csv(args.root/'evaluation/model_comparison.csv', index=False)
    primary = 'mae3_100k_ensemble'
    controls = [n for n in names if n != primary]
    for scenario in ['balanced', 'production_sampling', 'tau_1_percent', 'tau_0p1_percent']:
        a = np.load(args.root/'evaluation/models'/primary/scenario/'bootstrap.npz')
        for control in controls:
            b = np.load(args.root/'evaluation/models'/control/scenario/'bootstrap.npz')
            if not np.array_equal(a['names'], b['names']):
                raise ValueError('paired metric names differ')
            ar = json.loads((args.root/'evaluation/models'/primary/scenario/'result.json').read_text())
            br = json.loads((args.root/'evaluation/models'/control/scenario/'result.json').read_text())
            av, bv = {**ar['metrics'], **ar['fixed']}, {**br['metrics'], **br['fixed']}
            contrasts.append({'scenario': scenario, 'primary': primary, 'control': control,
                'metrics': {str(n): {'delta': av[str(n)]-bv[str(n)], **bootstrap.ci(a['values'][:, i]-b['values'][:, i])}
                    for i, n in enumerate(a['names'])}, 'replicates_paired': True})
    spread = {}
    for initialization in ['mae3', 'scratch']:
        members = [f's4_empty_{initialization}_n100000_s{s}' for s in [17, 42, 2026]]
        subset = pd.DataFrame(rows).query('scenario == "balanced"').set_index('name').loc[members]
        spread[initialization] = {n: {'seed_values': subset[n].tolist(), 'sample_SD': float(subset[n].std(ddof=1))}
            for n in METRIC_NAMES}
    write_json(args.root/'evaluation/paired_comparisons.json', {'contrasts': contrasts,
        'training_seed_spread_separate_from_event_intervals': spread})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['design', 'one', 'compare'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--task', type=int, default=0)
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == '__main__':
    main()
