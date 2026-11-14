"""Freeze or resolve train-inner optimization without consulting validation."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from analysis.flavour.inventory import write_json
from analysis.flavour.s4.dataset import LabelDataset
from analysis.flavour.s4.low_label import CompletePassSampler, TrainInnerDataset, inner_folds


RECIPES = {
    'early': {'encoder_lr_scale': 1., 'weight_decay': .01, 'classifier_dropout': .1, 'linear_probe_passes': 0},
    'slow': {'encoder_lr_scale': .1, 'weight_decay': .01, 'classifier_dropout': .1, 'linear_probe_passes': 0},
    'regularized': {'encoder_lr_scale': .1, 'weight_decay': .1, 'classifier_dropout': .3, 'linear_probe_passes': 0},
    'lpft': {'encoder_lr_scale': .1, 'weight_decay': .1, 'classifier_dropout': .3, 'linear_probe_passes': 2},
}


def plan(root, source):
    experiments, partitions = [], []
    originals = json.loads(source.read_text())['experiments']
    for budget in [1000, 10000]:
        for initialization in ['scratch', 'mae3']:
            original = next(c for c in originals if c['label_budget'] == budget and
                            c['initialization'] == initialization and c['seed'] == 42)
            base = {**original, 'inner_folds': 3, 'inner_split_seed': 20261114, 'epochs': 12,
                'warmup_steps': 32 if budget == 1000 else 128, 'minimum_learning_rate': 1e-5,
                'health_every_steps': 100, 'health_initial_steps': 8, 'progress_every_steps': 100,
                'selection_scope': 'original train only; family-grouped inner folds; same nested label budget'}
            base.pop('steps_per_epoch', None)
            data = LabelDataset(root, 'train', base)
            assignment = inner_folds(data.metadata, base['inner_folds'], base['inner_split_seed'])
            if initialization == 'scratch':
                partitions.append({'label_budget': budget, 'folds': [
                    {'fold': fold, 'fit_events': int((assignment != fold).sum()),
                     'selection_events': int((assignment == fold).sum()),
                     'selection_class_counts': dict(Counter(map(int, data.metadata.label[assignment == fold]))),
                     'selection_tau_counts': dict(Counter(data.metadata.tau_branch[(assignment == fold) & data.metadata.label.eq(2)]))}
                    for fold in range(3)]})
            data.native.close()
            for fold in range(3):
                c = {**base, 'selection_fold': fold}
                fit = TrainInnerDataset(root, 'train', c)
                sampler = CompletePassSampler(fit, c['batch_size'], c['seed'])
                steps = len(sampler)
                fit.native.close()
                for recipe, settings in RECIPES.items():
                    experiments.append({**c, **settings,
                        'name': f's4_inner_{recipe}_{initialization}_n{budget}_f{fold}_s42',
                        'transfer_recipe': recipe, 'complete_pass_steps': steps,
                        'eligible_selection_first_epoch': 3 if recipe == 'lpft' else 1})
    return {'stage': 'S4_train_inner_optimization', 'organization_device_hours_cap': 60,
        'designed_after_observing_empty_repair_validation': True, 'calibration_test_used': False,
        'original_validation_used_for_selection': False,
        'selection_rule': 'For each label budget minimize equally weighted Scratch/MAE3 OOF NLL over recipe and eligible epoch; shared recipe/passes; ties prefer fewer updates then declared recipe order.',
        'refit_seeds': [17, 42, 2026], 'partitions': partitions, 'experiments': experiments}


def resolve(root, campaign, budgets=(1000, 10000)):
    specification = json.loads(campaign.read_text())
    configurations = [c for c in specification['experiments'] if c['label_budget'] in budgets]
    rows = []
    for c in configurations:
        directory = root / 'experiments' / c['name']
        manifest = json.loads((directory / 'manifest.json').read_text())
        if manifest['status'] != 'complete' or manifest['config'] != c:
            raise ValueError(c['name'] + ': complete frozen inner fit required')
        if manifest.get('selection') != 'train_inner_NLL' or manifest.get('original_validation_used') is not False:
            raise ValueError('recipe selection must use only original train')
        history = json.loads((directory / 'history.json').read_text())
        if [r['epoch'] for r in history] != list(range(1, c['epochs'] + 1)) or any(
            r['train_exposures'] != manifest['train_events'] or
            r['optimizer_steps_cumulative'] != r['epoch'] * c['complete_pass_steps'] for r in history):
            raise ValueError('complete once-per-pass training history required')
        for row in history:
            rows.append({'budget': c['label_budget'], 'recipe': c['transfer_recipe'], 'fold': c['selection_fold'],
                'initialization': c['initialization'], 'epoch': row['epoch'],
                'eligible': row['epoch'] >= c['eligible_selection_first_epoch'],
                'NLL': row['validation']['NLL'], 'events': manifest['validation_events'],
                'updates': row['optimizer_steps_cumulative']})
    selections, refits = [], []
    for budget in budgets:
        candidates = []
        for recipe in RECIPES:
            for epoch in range(1, 13):
                measurements = [r for r in rows if (r['budget'], r['recipe'], r['epoch']) == (budget, recipe, epoch) and r['eligible']]
                if not measurements:
                    continue
                if len(measurements) != 6 or any(not np.isfinite(r['NLL']) for r in measurements):
                    raise ValueError('all three inner folds of both initializations required')
                by_kind = {kind: np.average([r['NLL'] for r in measurements if r['initialization'] == kind],
                    weights=[r['events'] for r in measurements if r['initialization'] == kind]) for kind in ['scratch', 'mae3']}
                candidates.append({'recipe': recipe, 'epochs': epoch, 'mean_OOF_NLL': float(np.mean(list(by_kind.values()))),
                    'OOF_NLL_by_initialization': {k: float(v) for k, v in by_kind.items()},
                    'mean_updates': float(np.mean([r['updates'] for r in measurements]))})
        selected = min(candidates, key=lambda r: (r['mean_OOF_NLL'], r['mean_updates'], list(RECIPES).index(r['recipe'])))
        selections.append({'label_budget': budget, 'selected': selected, 'candidates': candidates})
        for initialization in ['scratch', 'mae3']:
            base = next(c for c in configurations if c['label_budget'] == budget and
                        c['initialization'] == initialization and c['transfer_recipe'] == selected['recipe'])
            base = {k: v for k, v in base.items() if k not in {
                'selection_fold', 'eligible_selection_first_epoch', 'complete_pass_steps'}}
            data = LabelDataset(root, 'train', base)
            steps = len(CompletePassSampler(data, base['batch_size'], base['seed']))
            data.native.close()
            for seed in specification['refit_seeds']:
                c = {**base, 'seed': seed, 'epochs': selected['epochs'], 'complete_pass_steps': steps,
                    'name': f's4_optimized_{initialization}_n{budget}_s{seed}', 'select_final_epoch': True,
                    'selection_scope': 'recipe and epoch frozen by original train-inner OOF NLL; evaluate final epoch only for selection',
                    'inner_selection_source': str(campaign.resolve()), 'inner_selected_recipe': selected}
                if initialization == 'mae3':
                    parent = f's4_mae_m75_e3_s{seed}'
                    c.update(pretrained_checkpoint=str(root / 'experiments' / parent / 'best.pt'),
                        dependencies=[parent], required_ssl_readbacks=[parent])
                refits.append(c)
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'selections': selections,
        'selection_rule': specification['selection_rule'], 'original_validation_used': False,
        'calibration_test_used': False, 'training_seed_scope': 'recipe selection seed42 only; refit with 17/42/2026',
        'scope': 'conditional OOF tuning after prior validation observations; not independent final generalization evidence'}
    return report, {'stage': 'S4_train_inner_selected_refits', 'calibration_test_used': False, 'experiments': refits}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'resolve'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selection-report', type=Path)
    parser.add_argument('--budgets', type=int, nargs='+', choices=[1000, 10000], default=[1000, 10000])
    args = parser.parse_args()
    if args.action == 'plan':
        write_json(args.output, plan(args.root, args.source))
    else:
        if args.selection_report is None:
            parser.error('resolve requires --selection-report')
        report, campaign = resolve(args.root, args.source, args.budgets)
        write_json(args.selection_report, report)
        write_json(args.output, campaign)


if __name__ == '__main__':
    main()
