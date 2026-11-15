"""Matched-domain R1 development tables, paired block intervals and tau contamination."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.s3.metrics import bootstrap_weights
from analysis.flavour.v2.partitions import KEY, FAMILY, equal_identity_rows
from analysis.flavour.v2.r1.metrics import METRICS_REVISION, Metrics, three_from_four, three_targets


def probability_columns(frame):
    return ['p0', 'p1', 'p2', 'p3'] if 'p2' in frame else ['p0', 'p1', 'p3']


def domain_report(frame, mask):
    y = frame.label.to_numpy()
    p = frame[probability_columns(frame)].to_numpy()
    common = mask & (y != 2)
    p3 = p[common] if p.shape[1] == 3 else three_from_four(p[common])
    result = {'events': int(mask.sum()), 'non_tau_events': int(common.sum()),
        'raw_counts_original_labels': np.bincount(y[mask], minlength=4).tolist(),
        'T3': Metrics(three_targets(y[common]), p3).report() if common.any() else None}
    if common.any():
        target = three_targets(y[common])
        support = np.bincount(target, minlength=3)
        if (support > 0).all():
            result['T3_observed_mixture'] = result['T3']
            result['T3'] = Metrics(target, p3).report(len(target)/(3*support[target]))
    if mask.any() and p.shape[1] == 4:
        result['T4'] = Metrics(y[mask], p[mask]).report()
        support = np.bincount(y[mask], minlength=4)
        if (support > 0).all():
            result['T4_observed_mixture'] = result['T4']
            result['T4'] = Metrics(y[mask], p[mask]).report(mask.sum()/(4*support[y[mask]]))
    # Conditional T3 decisions are forced for true taus; they are not relabeled NC.
    conditional = p if p.shape[1] == 3 else three_from_four(p)
    assigned = conditional.argmax(1)
    tau = mask & (y == 2)
    result['tau_forced_assignments_e_mu_NC'] = np.bincount(assigned[tau], minlength=3).tolist()
    contamination = []
    if common.any():
        cuts = result['T3']['per_class']
        for c in range(3):
            row = {'original_label': [0, 1, 3][c]}
            for target in [.01, .001]:
                threshold = cuts[c].get(f'threshold_FPR_{target}')
                selected = mask & (conditional[:, c] >= threshold) if threshold is not None else np.zeros(len(y), dtype=bool)
                n_tau = int((selected & (y == 2)).sum())
                n_total = int(selected.sum())
                selected_counts = np.bincount(y[selected], minlength=4)
                support = np.bincount(y[mask], minlength=4)
                acceptance = np.divide(selected_counts, support, out=np.zeros(4), where=support>0)
                scenarios = {}
                supported = bool((support>0).all())
                for prior_tau in [.25, .01, .001]:
                    prior = np.array([(1-prior_tau)/3, (1-prior_tau)/3, prior_tau, (1-prior_tau)/3])
                    weighted_selected = prior*acceptance
                    scenarios[str(prior_tau)] = {'prior_e_mu_tau_NC': prior.tolist(),
                        'all_classes_observed': supported,
                        'tau_fraction': float(weighted_selected[2]/weighted_selected.sum()) if supported and weighted_selected.sum()>0 else None,
                        'input_coverage': float(weighted_selected.sum()) if supported else None,
                        'interpretation': 'benchmark prior scenario, conditional response held fixed; not a physical flux estimate'}
                row[str(target)] = {'threshold': threshold, 'selected_total': n_total, 'selected_tau': n_tau,
                    'tau_fraction_observed_domain_mixture': n_tau/n_total if n_total else None,
                    'tau_acceptance': n_tau/int(tau.sum()) if tau.any() else None,
                    'raw_selected_original_labels': selected_counts.tolist(), 'prior_scenarios': scenarios}
            contamination.append(row)
    result['tau_contamination'] = contamination
    return result


def scalar_summary(frame):
    y = frame.label.to_numpy()
    p = frame[probability_columns(frame)].to_numpy()
    keep = y != 2
    cache3 = Metrics(three_targets(y[keep]), p[keep] if p.shape[1] == 3 else three_from_four(p[keep]))
    cache4 = Metrics(y, p) if p.shape[1] == 4 else None
    def compute(weight=None):
        w = np.ones(len(y)) if weight is None else np.asarray(weight, dtype='float64')
        def balanced(labels, values, classes):
            support = np.bincount(labels, weights=values, minlength=classes)
            return values/np.maximum(support[labels], 1e-30)
        r3 = cache3.report(balanced(cache3.y, w[keep], 3))
        r4 = cache4.report(balanced(y, w, 4)) if cache4 is not None else None
        return np.array([r3['BA'], r3['NLL'], r3['macro_AP'], r3['macro_AUROC'],
            *(r3['per_class'][c]['recall'] for c in range(3)),
            r4['per_class'][2]['eff_FPR_0.01'] if r4 else np.nan,
            r4['per_class'][2]['eff_FPR_0.001'] if r4 else np.nan,
            r4['NLL'] if r4 else np.nan,
            *(r3['per_class'][c][f'eff_FPR_{fpr}'] for c in range(3) for fpr in [.01, .001]),
            *(r3['per_class'][c][f'eff_purity_{purity}'] for c in range(3) for purity in [.8, .9]),
            r4['BA'] if r4 else np.nan])
    return compute


SCALARS = ['BA3', 'NLL3', 'macro_AP3', 'macro_AUROC3', 'recall_e', 'recall_mu', 'recall_NC',
           'tau_eff_1pct', 'tau_eff_0p1pct', 'NLL4',
           *[f'{name}_eff_FPR_{fpr}' for name in ['e', 'mu', 'NC'] for fpr in [.01, .001]],
           *[f'{name}_eff_purity_{purity}' for name in ['e', 'mu', 'NC'] for purity in [.8, .9]], 'BA4']


def interval(point, samples):
    result = {}
    for c, name in enumerate(SCALARS):
        valid = np.isfinite(samples[:, c])
        result[name] = {'difference_a_minus_b': float(point[c]) if np.isfinite(point[c]) else None,
            '95pct': np.quantile(samples[valid, c], [.025, .975]).tolist() if valid.any() else [None, None]}
    return result


def paired(a, b, weights):
    left, right = scalar_summary(a), scalar_summary(b)
    return interval(left()-right(), np.array([left(w)-right(w) for w in weights]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--replicates', type=int, default=400)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    started = time.perf_counter()
    output = args.root/'evaluation'
    output.mkdir(exist_ok=True)
    config = json.loads(args.config.read_text())
    catalog = pd.read_parquet(args.root/'data/catalog.parquet', filters=[('split', '=', 'validation')]).sort_values(KEY).reset_index(drop=True)
    frames, reports = {}, {}
    rows = []
    for entry in config['models']:
        frame = pd.read_parquet(args.root/entry['predictions']).sort_values(KEY).reset_index(drop=True)
        if not equal_identity_rows(frame, catalog) or not np.array_equal(frame.label, catalog.label):
            raise ValueError('R1 candidates must use exactly the same frozen Tune rows and labels')
        frames[entry['name']] = frame
        domains = {'all': np.ones(len(frame), dtype=bool), 'pixel_centered': frame.pixel_centered.to_numpy(),
            'large_scintillator': frame.large_scintillator.to_numpy(), 'active': ~frame.all_empty.to_numpy(),
            'empty': frame.all_empty.to_numpy(), 'tracker_active': frame.tracker_channels.to_numpy()>0,
            'tracker_empty': frame.tracker_channels.to_numpy()==0}
        for left, right in [(0, 100), (100, 1000), (1000, 5000), (5000, np.inf)]:
            channels = frame.pixel_channels+frame.bar_channels+frame.tracker_channels
            domains[f'occupancy_{left}_{right}'] = channels.ge(left).to_numpy() & channels.lt(right).to_numpy()
        reports[entry['name']] = {name: domain_report(frame, mask) for name,mask in domains.items()}
        for domain, report in reports[entry['name']].items():
            if report['T3']:
                rows.append({'name': entry['name'], 'family': entry['family'], 'task': entry['task'],
                    'kind': entry.get('kind', 'main'),
                    'seed': entry['seed'], 'domain': domain, 'events': report['events'],
                    'non_tau_events': report['non_tau_events'],
                    **{f'{key}3': report['T3'][key] for key in ['BA', 'NLL', 'macro_AP', 'macro_AUROC']},
                    'BA4': report.get('T4', {}).get('BA'), 'NLL4': report.get('T4', {}).get('NLL'),
                    'tau_eff_1pct': report['T4']['per_class'][2].get('eff_FPR_0.01') if 'T4' in report else None})
    weights, design = bootstrap_weights(catalog, args.replicates, 20261116)
    catalog[KEY+FAMILY+['label']].to_parquet(output/'bootstrap_keys.parquet', index=False, compression=COMPRESSION)
    write_json(output/'bootstrap_design.json', {**design, 'replicates': args.replicates, 'seed': 20261116,
        'paired': True, 'scope': 'development Tune intervals; training seed spread reported separately'})
    comparisons = []
    cache = {}
    groups = {}
    entries = {e['name']: e for e in config['models']}
    needed = sorted({(pair[side], pair.get('domain', 'all'))
                     for pair in config.get('pairs', []) for side in ['a', 'b']})
    def resample(key):
        name, domain = key
        mask = np.ones(len(catalog), dtype=bool) if domain == 'all' else catalog[domain].to_numpy(dtype=bool)
        compute = scalar_summary(frames[name].loc[mask].reset_index(drop=True))
        return key, (compute(), np.asarray([compute(w[mask]) for w in weights]))
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for key, value in executor.map(resample, needed):
            cache[key] = value
    for pair in config.get('pairs', []):
        domain = pair.get('domain', 'all')
        left, right = cache[(pair['a'], domain)], cache[(pair['b'], domain)]
        comparisons.append({**pair, 'domain': domain, 'metrics': interval(left[0]-right[0], left[1]-right[1])})
        a, b = entries[pair['a']], entries[pair['b']]
        if a['seed']==b['seed']:
            key = tuple((e['family'], e['task'], e.get('kind', 'main')) for e in [a, b])+(domain, pair.get('reason', ''))
            groups.setdefault(key, []).append((a['seed'], left[0]-right[0], left[1]-right[1]))
    mean_seed_pairs = []
    for key, values in groups.items():
        seeds = [v[0] for v in values]
        if set(seeds)!={17,42,2026} or len(seeds)!=3:
            continue
        points = np.asarray([v[1] for v in values])
        metrics = interval(points.mean(0), np.asarray([v[2] for v in values]).mean(0))
        for c, name in enumerate(SCALARS):
            finite = np.isfinite(points[:, c])
            metrics[name]['per_seed_differences'] = [float(v) if np.isfinite(v) else None for v in points[:, c]]
            metrics[name]['training_seed_sample_SD'] = float(points[finite, c].std(ddof=1)) if finite.sum()>1 else None
        mean_seed_pairs.append({'a_family_task_kind': key[0], 'b_family_task_kind': key[1],
            'domain': key[2], 'reason': key[3], 'seeds': seeds, 'metrics': metrics,
            'scope': 'paired physical-event interval conditional on these three trained seeds; seed SD separate'})
    spread = []
    table = pd.DataFrame(rows)
    for key, group in table[table.domain.eq('all') & table.kind.eq('main')].groupby(['family', 'task']):
        if group.seed.nunique() > 1:
            spread.append({'family': key[0], 'task': key[1], 'seeds': group.seed.tolist(),
                'BA3_values': group.BA3.tolist(), 'BA3_mean': float(group.BA3.mean()),
                'BA3_sample_SD': float(group.BA3.std(ddof=1)), 'NLL3_mean': float(group.NLL3.mean())})
    table.to_csv(output/'model_domain_metrics.csv', index=False)
    write_json(output/'results.json', {'status': 'complete', 'models': reports, 'pairs': comparisons,
        'metrics_revision': METRICS_REVISION,
        'mean_seed_pairs': mean_seed_pairs,
        'seed_spread': spread, 'bootstrap': design, 'replicates': args.replicates,
        'benchmark_prior': {'T3': [1/3]*3, 'T4': [.25]*4},
        'bootstrap_workers': args.workers,
        'scope': 'development Tune; raw uncalibrated probabilities; descriptive score-selected operating points',
        'Cal_A_Cut_B_Confirm_payload_read': False, 'seconds': time.perf_counter()-started})
    print(json.dumps({'models': len(frames), 'comparisons': len(comparisons), 'status': 'complete'}), flush=True)


if __name__ == '__main__':
    main()
