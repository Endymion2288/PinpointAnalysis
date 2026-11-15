"""Development-only grouped readout diagnostic and Train/Tune convergence summary."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp, softmax

from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.s2.evaluate import resampling_groups
from analysis.flavour.v2.partitions import FAMILY, KEY, equal_identity_rows
from analysis.flavour.v2.r1.metrics import Metrics
from analysis.flavour.v2.r1.review import conditional_probabilities
from analysis.flavour.v2.r2.evaluate import configuration, domains, independent_points


def grouped_folds(frame, folds=5, seed=20261117):
    """Keep every family-connected chunk component in a single held-out fold."""
    groups, summary = resampling_groups(frame)
    rng = np.random.default_rng(seed)
    blocks = [block for stratum in sorted(groups) for block in groups[stratum]]
    order = rng.permutation(len(blocks))
    order = sorted(order, key=lambda i: -len(blocks[i]))
    assignment = np.full(len(frame), -1, dtype='int8')
    counts = np.zeros((folds, 4), dtype='int64')
    total = np.bincount(frame.label, minlength=4)
    for i in order:
        block = blocks[i]
        support = np.bincount(frame.label.to_numpy()[block], minlength=4)
        increase = ((2*counts*support+support**2)/np.maximum(total, 1)**2).sum(1)
        fold = int(np.argmin(increase))
        assignment[block] = fold
        counts[fold] += support
    if (assignment<0).any() or (counts==0).any():
        raise ValueError('grouped folds need complete assignment and all classes in every fold')
    return assignment, {**summary, 'fold_class_counts': counts.tolist(), 'folds': folds, 'seed': seed}


def plan(root):
    folder = root/'readout'
    folder.mkdir(exist_ok=True)
    path = folder/'contract.json'
    if path.exists():
        raise FileExistsError('readout diagnostic contract already frozen')
    frame = pd.read_parquet(root/'data/catalog.parquet', filters=[('split', '=', 'validation')])
    frame = frame.sort_values(KEY).reset_index(drop=True)
    assignment, summary = grouped_folds(frame)
    keys = frame[KEY+FAMILY+['label']].copy()
    keys['fold'] = assignment
    keys.to_parquet(folder/'folds.parquet', index=False, compression=COMPRESSION)
    write_json(path, {
        'status': 'frozen', 'UTC': datetime.now(timezone.utc).isoformat(),
        'scope': 'Tune development diagnostic; base model selection already used Tune; no independent performance claim',
        'motivation': 'R1 and R2 D0 screens show class decision shifts and weak low-occupancy muon ranking',
        'models': 'each completed campaign endpoint; best-screen predictions remain raw in primary evaluation',
        'folds': summary, 'task': 'common e/mu/NC, true taus excluded from fitting and evaluation',
        'input': 'log of stored conditional probabilities, clipped at 1e-15',
        'variants': ['raw', 'positive_scalar_temperature', 'two_global_class_biases'],
        'bias_anchor': 'NC=0', 'bias_bounds': [-3., 3.], 'bias_l2': .001,
        'temperature_inverse_bounds': [.05, 20.], 'objective': 'fit-fold class-balanced mean NLL',
        'hyperparameter_selection': 'one fixed setting, no scan; no occupancy-dependent parameters',
        'assessment': 'held-out predictions from five family-connected-chunk folds; report all domains and each fold',
        'intervals': 'not estimated for fitted readout; primary raw-model intervals remain in evaluation/',
        'interpretation': 'measures recoverable decision bias; added bias can change multiclass rankings; no added detector information',
        'Cal_A_Cut_B_Confirm_payload_read': False})
    print(json.dumps({'status': 'frozen', **summary}), flush=True)


def fit_bias(log_probability, labels, l2=.001, bounds=(-3., 3.)):
    """Convex two-parameter CE fit, with NC anchored and an analytic gradient."""
    support = np.bincount(labels, minlength=3)
    if (support==0).any():
        raise ValueError('bias fitting requires all three classes')
    weight = 1/(3*support[labels])

    def objective(value):
        shifted = log_probability+np.r_[value, 0.]
        probability = softmax(shifted, axis=1)
        residual = probability.copy()
        residual[np.arange(len(labels)), labels] -= 1
        loss = np.dot(weight, logsumexp(shifted, axis=1)-shifted[np.arange(len(labels)), labels])
        gradient = (weight[:, None]*residual).sum(0)[:2]+l2*value
        return float(loss+.5*l2*np.dot(value, value)), gradient

    result = minimize(objective, np.zeros(2), jac=True, method='L-BFGS-B',
        bounds=[bounds]*2, options={'gtol': 1e-9, 'ftol': 1e-13, 'maxiter': 200})
    if not result.success or not np.isfinite(result.fun):
        raise ValueError(f'bias optimization failed: {result.message}')
    return np.r_[result.x, 0.], {
        'success': bool(result.success), 'iterations': int(result.nit),
        'objective': float(result.fun), 'gradient': result.jac.tolist(),
        'at_bound': bool(np.any(np.isclose(result.x, bounds[0]) | np.isclose(result.x, bounds[1])))}


def fit_temperature(log_probability, labels, bounds=(.05, 20.)):
    support = np.bincount(labels, minlength=3)
    if (support==0).any():
        raise ValueError('temperature fitting requires all three classes')
    weight = 1/(3*support[labels])

    def objective(inverse):
        z = log_probability*inverse
        return float(np.dot(weight, logsumexp(z, axis=1)-z[np.arange(len(labels)), labels]))

    result = minimize_scalar(objective, method='bounded', bounds=bounds, options={'xatol': 1e-9})
    if not result.success:
        raise ValueError('temperature optimization failed')
    return float(result.x)


def crossfit(probabilities, labels, folds, contract):
    logits = np.log(np.clip(probabilities, 1e-15, 1.))
    variants = {'raw': probabilities.copy(), 'positive_scalar_temperature': np.zeros_like(probabilities),
        'two_global_class_biases': np.zeros_like(probabilities)}
    fits = []
    for fold in range(contract['folds']['folds']):
        held = folds==fold
        fit = ~held
        bias, health = fit_bias(logits[fit], labels[fit], contract['bias_l2'], tuple(contract['bias_bounds']))
        inverse = fit_temperature(logits[fit], labels[fit], tuple(contract['temperature_inverse_bounds']))
        variants['two_global_class_biases'][held] = softmax(logits[held]+bias, axis=1)
        variants['positive_scalar_temperature'][held] = softmax(logits[held]*inverse, axis=1)
        fits.append({'fold': fold, 'fit_class_counts': np.bincount(labels[fit], minlength=3).tolist(),
            'held_class_counts': np.bincount(labels[held], minlength=3).tolist(),
            'bias_e_mu_NC': bias.tolist(), 'inverse_temperature': inverse, 'bias_health': health})
    if not np.array_equal(probabilities.argmax(1), variants['positive_scalar_temperature'].argmax(1)):
        raise ValueError('positive scalar temperature unexpectedly changed argmax')
    return variants, fits


def report(labels, probability):
    support = np.bincount(labels, minlength=3)
    weights = len(labels)/(3*support[labels]) if (support>0).all() else np.ones(len(labels))
    result = Metrics(labels, probability).report(weights)
    result['argmax_raw_counts'] = np.bincount(probability.argmax(1), minlength=3).tolist()
    return result


def convergence(record, root):
    task = record['config']['task']
    history = record['history']
    tail = history[-3:]
    nll = np.array([row['metrics'][task]['NLL'] for row in tail])
    ba = np.array([row['metrics']['T3']['BA'] for row in tail])
    four = history[-4:]
    changes = np.diff([row['metrics'][task]['NLL'] for row in four])
    gaps = {}
    for kind, key in [('endpoint', 'metrics'), ('best_screen', 'best_metrics')]:
        path = root/'gap'/record['config']['name']/kind/'report.json'
        if not path.exists():
            continue
        gap = json.loads(path.read_text())
        train, tune = gap['reports']['all'], record[key]
        gaps[kind] = {t: {metric: {'Train': train[t][metric], 'Tune': tune[t][metric],
            'Tune_minus_Train': tune[t][metric]-train[t][metric]} for metric in ['NLL', 'BA']}
            for t in ['T3', 'T4'] if t in train and t in tune}
    return {'last_three_screen_exposures': [h['non_tau_exposures'] for h in tail],
        'last_three_own_task_NLL': nll.tolist(), 'last_three_common_BA': ba.tolist(),
        'adjacent_tail_NLL_differences': np.diff(nll).tolist(),
        'last_window_NLL_change': float(nll[-1]-nll[-2]) if len(nll)>1 else None,
        'tail_all_NLL_changes_below_0p001': bool(len(tail)==3 and (np.abs(np.diff(nll))<.001).all()),
        'last_four_screen_exposures': [h['non_tau_exposures'] for h in four],
        'three_successive_window_NLL_changes': changes.tolist(),
        'three_successive_NLL_changes_below_0p001': bool(len(changes)==3 and (np.abs(changes)<.001).all()),
        'tail_BA_range': float(np.ptp(ba)), 'Train_Tune_gaps': gaps,
        'scope': 'four screen points give three successive changes; engineering plateau aid, no asymptotic convergence claim'}


def run(root, partial=False):
    contract = json.loads((root/'readout/contract.json').read_text())
    fold_keys = pd.read_parquet(root/'readout/folds.parquet').sort_values(KEY).reset_index(drop=True)
    config = configuration(root, complete_only=partial)
    evaluation = root/('evaluation_partial' if partial else 'evaluation')
    evaluated = json.loads((evaluation/'results.json').read_text())
    if partial:
        config['models'] = [entry for entry in config['models'] if entry['name'] in evaluated['models']]
    elif {entry['name'] for entry in config['models']}!=set(evaluated['models']):
        raise ValueError('formal readout review requires the complete evaluated matrix')
    frames = {}
    for entry in config['models']:
        frame = pd.read_parquet(root/entry['predictions']).sort_values(KEY).reset_index(drop=True)
        if not equal_identity_rows(frame, fold_keys) or not np.array_equal(frame.label, fold_keys.label):
            raise ValueError('readout diagnostic requires exactly frozen Tune identities and labels')
        frames[entry['name']] = frame
    ranking = independent_points(frames, evaluated['models'])
    destination = root/('readout_partial' if partial else 'readout')
    write_json(destination/'independent_scores.json', ranking)
    if ranking['failed']:
        raise ValueError('independent probability/ranking review failed')
    models = {}
    for entry in config['models']:
        if entry['kind']!='endpoint':
            continue
        frame = frames[entry['name']]
        non_tau = frame.label.to_numpy()!=2
        labels = np.searchsorted([0, 1, 3], frame.label.to_numpy()[non_tau])
        probability = conditional_probabilities(frame)[1][non_tau]
        folds = fold_keys.fold.to_numpy()[non_tau]
        variants, fits = crossfit(probability, labels, folds, contract)
        masks = {domain: mask[non_tau] for domain, mask in domains(frame).items()}
        record = json.loads((root/'experiments'/entry['experiment']/'manifest.json').read_text())
        models[entry['experiment']] = {
            'fits': fits, 'convergence': convergence(record, root),
            'domains': {domain: {name: report(labels[mask], p[mask]) for name, p in variants.items()}
                for domain, mask in masks.items() if mask.any()},
            'fold_metrics': {str(fold): {name: report(labels[folds==fold], p[folds==fold])
                for name, p in variants.items()} for fold in range(contract['folds']['folds'])},
            'temperature_argmax_identical': True}
    result = {'status': 'complete', 'models': models, 'contract': str(root/'readout/contract.json'),
        'fold_family_integrity': bool(fold_keys.groupby(FAMILY).fold.nunique().max()==1),
        'fold_chunk_integrity': bool(fold_keys.groupby(['run', 'chunk']).fold.nunique().max()==1),
        'independent_score_checks': ranking['passed'], 'development_only': True,
        'Cal_A_Cut_B_Confirm_payload_read': False}
    if not result['fold_family_integrity'] or not result['fold_chunk_integrity']:
        raise ValueError('readout fold integrity failed')
    write_json(destination/'results.json', result)
    print(json.dumps({'status': 'complete', 'models': len(models), 'score_checks': ranking['passed'],
        'output': str(destination)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'run'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--complete-only', action='store_true')
    args = parser.parse_args()
    if args.command=='plan':
        plan(args.root)
    else:
        run(args.root, args.complete_only)


if __name__=='__main__':
    main()
