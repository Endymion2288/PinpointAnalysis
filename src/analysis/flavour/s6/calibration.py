"""Scalar temperature on calibration; prior scenarios and validation thresholds."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import logsumexp

from analysis.flavour.inventory import write_json
from analysis.flavour.s6.prepare import KEY


def log_probabilities(probability):
    values = np.asarray(probability, dtype='float64')
    if not np.isfinite(values).all() or np.any(values < 0) or not np.allclose(values.sum(1), 1, atol=2e-6):
        raise ValueError('invalid raw probabilities')
    return np.log(values.clip(1e-30, 1.))


def class_weights(y, prior):
    counts = np.bincount(y, minlength=4)
    if np.any(counts == 0):
        raise ValueError('all four calibration classes required')
    return (np.asarray(prior)/counts)[y]


def transform(probability, temperature=1., prior=None):
    z = log_probabilities(probability)/temperature
    if prior is not None:
        z += np.log(np.asarray(prior)/.25)
    return np.exp(z-logsumexp(z, axis=1, keepdims=True))


def temperature_fit(probability, y, prior=(.25, .25, .25, .25), bounds=(.05, 20.)):
    """NLL is convex in inverse temperature; solve its monotone derivative."""
    z = log_probabilities(probability)
    weight = class_weights(y, prior)
    true = z[np.arange(len(y)), y]

    def derivative(inverse):
        scaled = inverse*z
        p = np.exp(scaled-logsumexp(scaled, axis=1, keepdims=True))
        return float(weight @ ((p*z).sum(1)-true))

    left, right = 1/bounds[1], 1/bounds[0]
    dleft, dright = derivative(left), derivative(right)
    inverse = left if dleft >= 0 else right if dright <= 0 else brentq(derivative, left, right, xtol=1e-10)
    scaled = inverse*z
    p = np.exp(scaled-logsumexp(scaled, axis=1, keepdims=True))
    hessian = float(weight @ ((p*z*z).sum(1)-(p*z).sum(1)**2))
    nll = lambda scale: float(weight @ (logsumexp(scale*z, axis=1)-scale*true))
    return {'temperature': float(1/inverse), 'inverse_temperature': float(inverse),
            'at_bound': bool(inverse in [left, right]), 'bounds': list(bounds),
            'balanced_NLL_before': nll(1.), 'balanced_NLL_after': nll(inverse),
            'derivative_at_solution': derivative(inverse), 'inverse_temperature_hessian': hessian,
            'events': len(y), 'class_counts': np.bincount(y, minlength=4).tolist(), 'fit_prior': list(prior)}


def background_threshold(score, background, weight, target):
    """Inclusive >= cut; ties remain intact and validation FPR never exceeds target."""
    values, weights = np.asarray(score)[background], np.asarray(weight)[background]
    order = np.argsort(-values, kind='stable')
    values, weights = values[order], weights[order]
    ends = np.r_[np.flatnonzero(np.diff(values)) + 1, len(values)]-1
    cumulative = np.cumsum(weights)[ends]/weights.sum()
    allowed = np.flatnonzero(cumulative <= target+1e-14)
    cut = float(values[ends[allowed[-1]]]) if len(allowed) else float(np.nextafter(values[0], np.inf))
    passed = np.asarray(score) >= cut
    selected = int(np.sum(passed & background))
    return {'threshold': cut, 'validation_FPR': float(weight[passed & background].sum()/weight[background].sum()),
            'background_passed': selected, 'background_events': int(background.sum()),
            'background_passed_below_25': selected < 25, 'target_FPR': target, 'tie_rule': 'inclusive >=; conservative whole tie groups'}


def selection(probability, thresholds):
    thresholds = np.asarray(thresholds)
    passed = probability >= thresholds[None]
    margin = np.log(np.maximum(probability, 1e-30)/thresholds[None])
    result = np.where(passed, margin, -np.inf).argmax(1)
    result[~passed.any(1)] = 4
    return result


def model_names(campaign):
    return [m['name'] for m in campaign['models']]+[e['name'] for e in campaign['ensembles']]


def probability_columns(name, campaign):
    members = next((e['members'] for e in campaign['ensembles'] if e['name'] == name), [name])
    return members, [f'{member}__p{c}' for member in members for c in range(4)]


def extract(frame, name, campaign):
    members, _ = probability_columns(name, campaign)
    return np.mean([frame[[f'{member}__p{c}' for c in range(4)]].to_numpy() for member in members], axis=0)


def validation(name, campaign):
    members, _ = probability_columns(name, campaign)
    models = {m['name']: m for m in campaign['models']}
    reference, arrays = None, []
    for member in members:
        frame = pd.read_parquet(models[member]['validation_predictions'])
        if reference is None:
            reference = frame[KEY+['label']].copy()
        frame = reference[KEY].merge(frame, on=KEY, how='left', validate='one_to_one', sort=False)
        if frame.p0.isna().any():
            raise ValueError('ensemble validation keys differ')
        arrays.append(frame[[f'p{c}' for c in range(4)]].to_numpy())
    return reference, np.mean(arrays, axis=0)


def fit(args):
    campaign = json.loads(args.campaign.read_text())
    name = model_names(campaign)[args.task]
    _, columns = probability_columns(name, campaign)
    frame = pd.read_parquet(args.root/'predictions/calibration_raw.parquet', columns=KEY+['label']+columns)
    expected = sum(t['events'] for t in json.loads((args.root/'plan.json').read_text())['tasks'] if t['split'] == 'calibration')
    if len(frame) != expected:
        raise ValueError('temperature fit requires all accepted calibration events')
    probability = extract(frame, name, campaign)
    result = temperature_fit(probability, frame.label.to_numpy(), campaign['calibration']['fit_prior'], campaign['calibration']['temperature_bounds'])
    reference, raw = validation(name, campaign)
    priors, thresholds = [], []
    for scenario in campaign['prior_scenarios']:
        prior = scenario['prior'] or (np.bincount(frame.label, minlength=4)/len(frame)).tolist()
        priors.append({'name': scenario['name'], 'prior': prior})
        calibrated = transform(raw, result['temperature'], prior)
        y = reference.label.to_numpy()
        weight = class_weights(y, prior)
        for fpr in campaign['selection']['per_class_background_acceptance']:
            points = [background_threshold(calibrated[:, c], y != c, weight, fpr) for c in range(4)]
            thresholds.append({'scenario': scenario['name'], 'target_FPR': fpr, 'classes': points})
    result.update(name=name, priors=priors, validation_events=len(reference), thresholds=thresholds,
        method=campaign['calibration'], UTC=datetime.now(timezone.utc).isoformat(),
        validation_keys_path=campaign['models'][0]['validation_predictions'], test_used=False)
    destination = args.root/'calibration'/f'{name}.json'
    write_json(destination, result)
    print(json.dumps({k: result[k] for k in ['name', 'temperature', 'balanced_NLL_before', 'balanced_NLL_after', 'at_bound']}), flush=True)


def freeze(args):
    campaign = json.loads(args.campaign.read_text())
    calibrators = {name: json.loads((args.root/'calibration'/f'{name}.json').read_text()) for name in model_names(campaign)}
    conditions = json.loads(args.conditions.read_text())
    policy = json.loads((args.root/'inference_policy.json').read_text())
    evaluation = json.loads(args.evaluation.read_text())
    plan = json.loads((args.root/'plan.json').read_text())
    result = {'stage': 'S6', 'status': 'frozen', 'UTC': datetime.now(timezone.utc).isoformat(),
        'calibration_and_validation_rules_frozen': True, 'test_response_opened_before_freeze': False,
        'candidates': campaign, 'calibrators': calibrators, 'inference_policy': policy, 'conditions': conditions,
        'evaluation': evaluation,
        'sample_scope': {'selection': plan['selection'], 'counts': plan['counts'], 'tasks': plan['tasks']},
        'statistics': campaign['bootstrap'], 'selection': campaign['selection'],
        'interval_scope': 'pointwise family-connected chunk bootstrap conditional on frozen checkpoints, calibration and thresholds; seed spread separate',
        'physical_exposure': 'unmeasured; named prior scenarios only', 'test_threshold_retuning': False}
    destination = args.root/'analysis_freeze.json'
    if destination.exists():
        raise ValueError('analysis freeze already exists')
    write_json(destination, result)
    print(json.dumps({'status': 'frozen', 'calibrators': len(calibrators), 'test_events': sum(t['events'] for t in plan['tasks'] if t['split'] == 'test')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['fit', 'freeze'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--conditions', type=Path)
    parser.add_argument('--evaluation', type=Path)
    parser.add_argument('--task', type=int, default=0)
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == '__main__':
    main()
