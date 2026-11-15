"""Independently read back R1 identities, exposures, class means and selected counts."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.flavour.inventory import write_json
from analysis.flavour.v2.partitions import KEY, FAMILY, equal_identity_rows


def conditional_probabilities(frame):
    columns = ['p0', 'p1', 'p2', 'p3'] if 'p2' in frame else ['p0', 'p1', 'p3']
    original = frame[columns].to_numpy()
    common = original if len(columns)==3 else original[:, [0, 1, 3]]
    if len(columns)==4:
        # Preserve the stored dtype's normalization, as specified for evaluation.
        common = common/np.maximum(common.sum(1, keepdims=True), 1e-15)
    return original.astype('float64'), common.astype('float64')


def direct_metrics(y, p):
    support = np.bincount(y, minlength=p.shape[1])
    balanced = bool((support>0).all())
    predicted = p.argmax(1)
    recall = [np.mean(predicted[y==c]==c) for c in range(p.shape[1]) if support[c]]
    loss = -np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1.))
    nll = np.mean([loss[y==c].mean() for c in range(p.shape[1])]) if balanced else loss.mean()
    return {'BA': float(np.mean(recall)), 'NLL': float(nll)}, support, balanced


def selected_rates(y, p, c, cut, balanced):
    selected = p[:, c]>=cut if cut is not None else np.zeros(len(y), dtype=bool)
    support = np.bincount(y, minlength=p.shape[1])
    counts = np.bincount(y[selected], minlength=p.shape[1])
    acceptance = np.divide(counts, support, out=np.zeros(p.shape[1]), where=support>0)
    background = np.arange(p.shape[1])!=c
    fpr = acceptance[background].mean() if balanced else counts[background].sum()/max(1, support[background].sum())
    return selected, counts, acceptance, float(fpr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--require-full', action='store_true')
    args = parser.parse_args()
    root = args.root
    config = json.loads((args.config or root/'evaluation_config.json').read_text())
    evaluation = json.loads((root/'evaluation/results.json').read_text())
    catalog = pd.read_parquet(root/'data/catalog.parquet').sort_values(KEY).reset_index(drop=True)
    tune = catalog[catalog.split.eq('validation')].reset_index(drop=True)
    checks, models, core = [], [], {}
    largest = {'BA': 0., 'NLL': 0., 'working_point': 0., 'prior_purity': 0.}

    def check(name, value, **details):
        checks.append({'name': name, 'passed': bool(value), **details})

    for entry in config['models']:
        name = entry['name']
        folder = root/'experiments'/name
        manifest = json.loads((folder/'manifest.json').read_text())
        frame = pd.read_parquet(root/entry['predictions']).sort_values(KEY).reset_index(drop=True)
        fields = [c for c in [*KEY, *FAMILY, 'label', 'pixel_centered', 'large_scintillator',
                  'pixel_channels', 'bar_channels', 'tracker_channels', 'all_empty'] if c in tune]
        check(name+'_catalog_metadata', all(np.array_equal(frame[c], tune[c]) for c in fields))
        original, common = conditional_probabilities(frame)
        check(name+'_valid_probabilities', np.isfinite(original).all() and (original>=0).all()
              and (original<=1).all() and np.allclose(original.sum(1), 1., atol=2e-6))
        y = frame.label.to_numpy()
        channels = frame.pixel_channels+frame.bar_channels+frame.tracker_channels
        domains = {'all': np.ones(len(y), dtype=bool), 'pixel_centered': frame.pixel_centered.to_numpy(),
            'large_scintillator': frame.large_scintillator.to_numpy(), 'active': ~frame.all_empty.to_numpy(),
            'empty': frame.all_empty.to_numpy(), 'tracker_active': frame.tracker_channels.to_numpy()>0,
            'tracker_empty': frame.tracker_channels.to_numpy()==0}
        for left, right in [(0, 100), (100, 1000), (1000, 5000), (5000, np.inf)]:
            domains[f'occupancy_{left}_{right}'] = channels.ge(left).to_numpy() & channels.lt(right).to_numpy()
        errors, working_errors, prior_errors = [], [], []
        for domain, mask in domains.items():
            reported = evaluation['models'][name][domain]
            check(name+'_'+domain+'_raw_counts', reported['raw_counts_original_labels']==
                  np.bincount(y[mask], minlength=4).tolist())
            for task in ['T3', 'T4']:
                if task=='T4' and original.shape[1]!=4:
                    continue
                keep = mask & (y!=2) if task=='T3' else mask
                if not keep.any():
                    continue
                labels = np.searchsorted([0, 1, 3], y[keep]) if task=='T3' else y[keep]
                p = common[keep] if task=='T3' else original[keep]
                measured, support, balanced = direct_metrics(labels, p)
                for key, value in measured.items():
                    error = abs(value-reported[task][key])
                    largest[key] = max(largest[key], error)
                    errors.append(error)
                if domain=='all' and task=='T3':
                    core[name] = measured
                for c in range(p.shape[1]):
                    if not support[c] or support[c]==len(labels):
                        continue
                    row = reported[task]['per_class'][c]
                    for target in [.01, .001]:
                        cut = row[f'threshold_FPR_{target}']
                        selected, counts, acceptance, fpr = selected_rates(labels, p, c, cut, balanced)
                        working_errors += [abs(fpr-row[f'actual_FPR_{target}']),
                            abs(acceptance[c]-row[f'eff_FPR_{target}']), max(0., fpr-target)]
                        if task=='T4' and c==2 and (args.require_full or 'T4_tau_prior_scenarios' in reported):
                            scenario = reported['T4_tau_prior_scenarios'][str(target)]
                            check(name+'_'+domain+f'_tau_counts_{target}', scenario['raw_selected_original_labels']==counts.tolist())
                            for pi in [.25, .01, .001]:
                                prior = np.array([(1-pi)/3, (1-pi)/3, pi, (1-pi)/3])
                                total = float(np.dot(prior, acceptance))
                                expected = pi*acceptance[2]/total if balanced and total>0 else None
                                actual = scenario['prior_scenarios'][str(pi)]['tau_fraction']
                                if expected is None:
                                    check(name+'_'+domain+f'_unsupported_prior_{target}_{pi}', actual is None)
                                else:
                                    prior_errors.append(abs(actual-expected))
        largest['working_point'] = max(largest['working_point'], max(working_errors, default=0.))
        largest['prior_purity'] = max(largest['prior_purity'], max(prior_errors, default=0.))
        check(name+'_direct_class_mean_metrics', max(errors, default=0.)<2e-10, maximum_absolute_error=max(errors, default=0.))
        check(name+'_direct_working_points', max(working_errors, default=0.)<2e-10,
              maximum_absolute_error=max(working_errors, default=0.))
        check(name+'_direct_prior_purity', max(prior_errors, default=0.)<2e-10)
        item = {**entry, 'direct_common_three': core[name], 'GPU': manifest.get('GPU', 'CPU'),
                'precision': manifest['config'].get('precision'), 'wall_seconds': manifest['wall_seconds']}
        if entry['family']!='B0':
            exposures = pd.read_parquet(folder/'train_exposures.parquet')
            expected = catalog[catalog.split.eq('train') & (catalog.label.ne(2) if entry['task']=='T3' else True)]
            check(name+'_training_keys_labels', equal_identity_rows(exposures, expected)
                  and exposures.sort_values(KEY).label.tolist()==expected.sort_values(KEY).label.tolist())
            by_class = [int(exposures.loc[exposures.label.eq(c), 'exposures'].sum()) for c in range(4)]
            check(name+'_class_exposure_counters', by_class==manifest['counters']['exposure_by_class'])
            item.update(exposures_by_class=by_class, minimum_exposures=int(exposures.exposures.min()),
                        maximum_exposures=int(exposures.exposures.max()), updates=manifest['counters']['updates'])
        models.append(item)

    if args.require_full:
        ranking = json.loads((root/'architecture_ranking.json').read_text())
        measured = sorted([{'family': family,
            'mean_BA3': np.mean([core[f'final_{family.lower()}_{task}_s42']['BA'] for task in ['t3','t4']]),
            'mean_NLL3': np.mean([core[f'final_{family.lower()}_{task}_s42']['NLL'] for task in ['t3','t4']])}
            for family in ['B1','B2','B3','B4']], key=lambda r: (-r['mean_BA3'], r['mean_NLL3']))
        check('ranking_from_actual_probabilities', [r['family'] for r in measured]==[r['family'] for r in ranking['ranking']])
        stress_rows = pd.read_parquet(root/'data/stress_rows.parquet')
        stress = json.loads((root/'evaluation/stress_results.json').read_text())
        paths = []
        for entry in [e for e in config['models'] if e.get('kind', 'main')=='main' and e['seed']==42]:
            for condition in ['nominal','noise','combined','remove_pixel','remove_tracker']:
                path = root/'diagnostics'/f"{entry['name']}_{condition}"
                record = json.loads((path/'manifest.json').read_text())
                prediction = pd.read_parquet(path/'predictions.parquet').sort_values(KEY).reset_index(drop=True)
                reference = stress_rows.sort_values(KEY).reset_index(drop=True)
                check(path.name+'_frozen_stress_rows', equal_identity_rows(prediction, reference)
                      and np.array_equal(prediction.label, reference.label))
                check(path.name+'_terminal_inference', record['status']=='complete' and not record['Cal_A_Cut_B_Confirm_payload_read'])
                paths.append(path.name)
        check('eleven_models_five_conditions', len(paths)==55 and len(stress['models'])==11)

    failed = [c['name'] for c in checks if not c['passed']]
    output = {'status': 'complete' if not failed else 'failed', 'require_full': args.require_full,
        'scope': 'independent array/count readback of development results; uncertainty and scientific promotion reviewed separately',
        'models': models, 'largest_absolute_errors': largest, 'checks': checks,
        'passed': len(checks)-len(failed), 'failed': failed, 'Cal_A_Cut_B_Confirm_payload_read': False}
    write_json(root/'evaluation/independent_review.json', output)
    print(json.dumps({key: output[key] for key in ['status','require_full','passed','failed','largest_absolute_errors']}), flush=True)
    if failed:
        raise SystemExit(1)


if __name__=='__main__':
    main()
