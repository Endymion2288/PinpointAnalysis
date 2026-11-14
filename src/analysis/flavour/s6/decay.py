"""Compact resimulated decay evidence and frozen tau-selection counts."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.flavour.inventory import write_json
from analysis.flavour.s6.calibration import extract, transform
from analysis.flavour.s6.evaluate import exact_interval
from analysis.flavour.s6.prepare import KEY, require_freeze


def evaluate(args):
    require_freeze(args.root, 'test')
    base = args.root/args.scope
    completion = json.loads((base/'truth_completion.json').read_text())
    if not completion['all_observations_exactly_equal']:
        raise ValueError('decay replay response mismatch')
    truth = pd.read_parquet(base/'decay_truth.parquet')
    raw = pd.read_parquet(args.root/'predictions'/f'{args.scope}_raw.parquet')
    frame = raw.drop(columns=[c for c in raw if c.startswith('resim_')]).merge(
        truth, on=KEY+['condition'], how='left', validate='one_to_one', sort=False)
    if frame.resim_tau_branch.isna().any():
        raise ValueError('decay truth coverage missing')
    freeze = json.loads((args.root/'analysis_freeze.json').read_text())
    name = 'mae3_100k_ensemble'
    cal = freeze['calibrators'][name]
    groups = 'tune' if args.scope == 'generator' else 'condition'
    nominal = frame[frame.condition.eq('nominal')][KEY+['resim_tau_branch']]
    output = {}
    for group, current in frame.groupby(groups, sort=True):
        tau = current[current.label.eq(2)]
        branches = tau.resim_tau_branch.value_counts().to_dict()
        valid = tau.resim_tau_decay_valid.to_numpy()
        vertices = np.array(tau.loc[valid, 'resim_tau_decay_vertex_mm'].tolist()).reshape(-1, 3)
        row = {'events': len(current), 'tau_events': len(tau), 'tau_decay_valid': int(valid.sum()),
            'tau_branches': branches, 'tau_branch_fractions': {k: {'events': int(v),
                'fraction': v/len(tau), 'raw_binomial_CI95': exact_interval(v, len(tau))} for k, v in branches.items()},
            'tau_decay_z_quantiles_mm': np.quantile(vertices[:, 2], [.01, .5, .99]).tolist() if len(vertices) else [],
            'charm_primary_events': int(current.resim_charm_primary.sum()),
            'charm_primary_with_recorded_children': int((current.resim_charm_primary & current.resim_charm_tree_rows.gt(0)).sum()),
            'charm_direct_child_e_events': int(current.resim_charm_e.sum()),
            'charm_direct_child_mu_events': int(current.resim_charm_mu.sum()), 'scenarios': {}}
        if args.scope == 'transport':
            paired = tau[KEY+['resim_tau_branch']].merge(nominal, on=KEY, suffixes=('_current', '_nominal'),
                how='left', validate='one_to_one')
            row['paired_tau_branch_migration'] = paired.groupby(['resim_tau_branch_nominal', 'resim_tau_branch_current']).size().rename(
                'events').reset_index().to_dict('records')
        for scenario in cal['priors']:
            probability = transform(extract(current, name, freeze['candidates']), cal['temperature'], scenario['prior'])
            selections = []
            for point in [p for p in cal['thresholds'] if p['scenario'] == scenario['name']]:
                passed = probability[:, 2] >= point['classes'][2]['threshold']
                by_branch = {}
                for branch in branches:
                    mask = current.label.eq(2).to_numpy() & current.resim_tau_branch.eq(branch).to_numpy()
                    n, k = int(mask.sum()), int(passed[mask].sum())
                    by_branch[branch] = {'passed': k, 'total': n, 'efficiency': k/n,
                        'raw_binomial_CI95': exact_interval(k, n)}
                charm = {}
                for flag in [False, True]:
                    mask = current.label.ne(2).to_numpy() & current.resim_charm_primary.eq(flag).to_numpy()
                    n, k = int(mask.sum()), int(passed[mask].sum())
                    charm[str(flag)] = {'passed': k, 'background_events': n,
                        'raw_false_positive_fraction': k/n if n else None, 'raw_binomial_CI95': exact_interval(k, n),
                        'class_counts': np.bincount(current.loc[mask, 'label'], minlength=4).tolist()}
                selections.append({'target_FPR': point['target_FPR'], 'threshold': point['classes'][2]['threshold'],
                    'tau_by_branch': by_branch, 'raw_background_by_primary_charm': charm})
            row['scenarios'][scenario['name']] = selections
        output[str(group)] = row
    write_json(args.root/'evaluation/conditions'/f'{args.scope}_decay.json', {'status': 'complete',
        'groups': output, 'replay': completion, 'thresholds_retuned': False,
        'charm_child_scope': 'all direct children of the supported primary charm species, including interaction/ionization secondaries; electron/muon presence is not a measured charm branching fraction',
        'raw_interval_scope': 'descriptive exact binomial intervals on integer branch/selection counts; block-paired condition effects reported separately',
        'polarization': 'frozen tau-minus helicity proxies -1, 0, +1; charge-conjugate tau-plus; Pythia SPINUP=0 not identified as an unpolarized mixture',
        'secondary_implementation': 'FTFP_BERT+PY8DK, Pythia8.313; original implementation unchanged; compact trajectory origins are truth-only'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['evaluate'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--scope', choices=['transport', 'generator'], required=True)
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == '__main__':
    main()
