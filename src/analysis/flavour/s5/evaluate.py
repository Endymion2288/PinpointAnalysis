"""S5 paired validation, difficult channels, objects and component resolutions."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import beta, spearmanr
from sklearn.metrics import average_precision_score, confusion_matrix, log_loss, roc_auc_score

from analysis.flavour.inventory import write_json
from analysis.flavour.s2.evaluate import common_support_weights, tau_metrics
from analysis.flavour.s3.metrics import MetricCache, NAMES, bootstrap_weights, interval_report, prediction_diagnostics
from analysis.flavour.s4.evaluate import tau_statistics, tau_interval, TAU_NAMES


TRUTH_FIELDS = ['run', 'chunk', 'row', 'label', 'tau_branch', 'tau_decay_valid', 'tau_decay_vertex_mm', 'tau_decay_z_mm',
                'tau_decay_length_mm', 'tau_decay_location', 'tau_decay_contained', 'tau_direct_prongs', 'tau_direct_prongs_hit',
                'tau_visible_any', 'tau_visible_pixel', 'tau_visible_bar', 'tau_visible_tracker', 'charm_presence',
                'charm_branch', 'vertex_mm', 'vertex_supervision_valid', 'visible_p3_GeV', 'visible_E_GeV', 'visible_target_valid', 'six_label']
TRUTH_FIELDS += ['bar_semantic_fractions', 'bar_semantic_valid', 'charm_label_valid']


def binomial(k, n):
    return {'selected': int(k), 'events': int(n), 'rate': float(k/n) if n else None,
            'CI95': [float(beta.ppf(.025, k, n-k+1)) if k else 0., float(beta.ppf(.975, k+1, n-k)) if k < n else 1.] if n else [None, None]}


def tau_domains(frame):
    signal = frame.label.to_numpy() == 2
    groups = {f'branch_{b}': frame.tau_branch.eq(b).to_numpy() for b in ['electron', 'muon', 'hadronic', 'unknown']}
    for flag in ['tau_visible_any', 'tau_visible_pixel', 'tau_visible_tracker', 'tau_decay_contained', 'charm_presence']:
        for value in [False, True]:
            groups[f'{flag}_{int(value)}'] = frame[flag].eq(value).to_numpy()
    if 'response_contained_proxy' in frame:
        for value in [False, True]:
            groups[f'response_contained_proxy_{int(value)}'] = frame.response_contained_proxy.eq(value).to_numpy()
    for prong in [0, 1, 3]:
        groups[f'prongs_{prong}'] = frame.tau_direct_prongs.eq(prong).to_numpy()
    for location in sorted(frame.loc[signal, 'tau_decay_location'].unique()):
        groups['decay_'+location] = frame.tau_decay_location.eq(location).to_numpy()
    return {k: ~signal | (signal & mask) for k, mask in groups.items()}


def operating_tables(frame, common):
    inclusive = tau_metrics(frame)
    result = {'inclusive': inclusive, 'strata': [], 'charm_fake_tau': [], 'charm_branch_fake_tau': [],
              'threshold_scope': 'empirical validation ROC; calibrated final threshold belongs to S6',
              'count_interval_scope': 'descriptive Clopper-Pearson counts at validation-selected threshold; no independent operating-point coverage claim'}
    signal = frame.label.eq(2).to_numpy()
    for fpr in [.01, .001]:
        point = inclusive.get(f'FPR_{fpr:g}', {})
        cut = point.get('threshold')
        selected = frame.p2.to_numpy() >= cut if cut is not None else np.zeros(len(frame), bool)
        background_count = int(np.sum(selected & ~signal))
        for name, mask in tau_domains(frame).items():
            eligible = signal & mask
            n, k = int(eligible.sum()), int(np.sum(selected & eligible))
            result['strata'].append({'stratum': name, 'FPR': fpr, **binomial(k, n),
                'fixed_background_selected': background_count, 'branch_vs_all_background_purity': k/(k+background_count) if k+background_count else None,
                'common_support_weight_sum': float(common[eligible].sum()),
                'common_support_efficiency': float(np.sum(common[selected & eligible])/np.sum(common[eligible])) if np.sum(common[eligible]) else None})
        for label in [0, 1, 3]:
            for charm in [False, True]:
                mask = frame.label.eq(label).to_numpy() & frame.charm_presence.eq(charm).to_numpy()
                result['charm_fake_tau'].append({'label': label, 'charm': charm, 'FPR': fpr,
                                                **binomial(int(np.sum(mask & selected)), int(mask.sum()))})
            for branch in ['electron', 'muon', 'hadronic']:
                mask = frame.label.eq(label).to_numpy() & frame.charm_presence.eq(True).to_numpy() & frame.charm_branch.eq(branch).to_numpy()
                result['charm_branch_fake_tau'].append({'label': label, 'branch': branch, 'FPR': fpr,
                                                       **binomial(int(np.sum(mask & selected)), int(mask.sum()))})
    return result


def conditional_correlations(frame):
    result = []
    columns = ['pixel_channels', 'bar_channels', 'tracker_channels', 'pixel_tracks', 'tracker_tracks',
               'cluster_count', 'vertex_sigma_z_mm', 'pixel_edge_fraction', 'pixel_end_fraction',
               'bar_edge_signal_fraction', 'bar_end_signal_fraction', 'centroid_slope_change']
    for label in range(4):
        selected = frame.label.eq(label)
        for observable in columns:
            if observable not in frame:
                continue
            x = frame.loc[selected, observable].to_numpy(dtype=float)
            y = frame.loc[selected, 'p2'].to_numpy()
            valid = np.isfinite(x)
            rho = float(spearmanr(x[valid], y[valid]).statistic) if valid.sum() > 2 and np.ptp(x[valid]) > 0 else None
            result.append({'true_class': label, 'observable': observable, 'events': int(valid.sum()), 'tau_score_spearman': rho})
    return result


def resolution(values, mask):
    values = np.asarray(values)
    finite = np.isfinite(values).all(1) if values.ndim > 1 else np.isfinite(values)
    mask = np.asarray(mask) & finite
    x = values[mask]
    return {'events': int(mask.sum()), 'median_signed': np.median(x, axis=0).tolist() if len(x) else None,
            'absolute_quantiles_50_68_95': np.quantile(np.abs(x), [.5, .68, .95], axis=0).tolist() if len(x) else None}


def kinematics(frame, config):
    result = {'all_events': len(frame), 'groups': [], 'definition': 'non_nuclear_G4_initial_primary_system_replace_tau_exclude_neutrinos',
              'residual_sign': 'prediction minus target', 'component_residual': 'absolute GeV; pT relative only for target pT > 0.1 GeV',
              'coverage': 'fraction of all events passing observable classification-confidence cuts; quantiles are error summaries, not calibrated predictive intervals'}
    tasks = config.get('auxiliaries', [])
    confidence = frame[[f'p{c}' for c in range(4)]].max(axis=1).to_numpy()
    reco_vertex = frame[[f'pred_vertex_{c}' for c in range(3)]].to_numpy()*[210, 210, 3000] if 'vertex' in tasks else None
    true_vertex = np.stack(frame.vertex_mm)
    if 'kinematics' in tasks:
        encoded = frame[[f'pred_kinematics_{c}' for c in range(4)]].to_numpy()*7.
        if not np.isfinite(encoded).all() or np.max(np.abs(encoded)) > 30:
            raise RuntimeError('kinematic head output requires numerical/architecture review')
        reco = np.column_stack([np.sinh(encoded[:, 0]), np.sinh(encoded[:, 1]), np.expm1(encoded[:, 2])])
        true = np.stack(frame.visible_p3_GeV)
        pt, pt_true = np.linalg.norm(reco[:, :2], axis=1), np.linalg.norm(true[:, :2], axis=1)
        theta = np.arctan2(pt, reco[:, 2])-np.arctan2(pt_true, true[:, 2])
        phi = np.arctan2(reco[:, 1], reco[:, 0])-np.arctan2(true[:, 1], true[:, 0])
        wrapped_phi = np.arctan2(np.sin(phi), np.cos(phi))
        residuals = {'px_GeV': reco[:, 0]-true[:, 0], 'py_GeV': reco[:, 1]-true[:, 1],
                     'pT_GeV': pt-pt_true, 'pT_relative': (pt-pt_true)/np.maximum(pt_true, .1),
                     'delta_theta_rad': theta, 'wrapped_delta_phi_rad': wrapped_phi,
                     'visible_E_relative': (np.expm1(encoded[:, 3])-frame.visible_E_GeV.to_numpy())/np.maximum(frame.visible_E_GeV.to_numpy(), .1)}
    else:
        residuals = {}
    for label in ['all', 0, 1, 2, 3]:
        class_mask = np.ones(len(frame), bool) if label == 'all' else frame.label.eq(label).to_numpy()
        for cut in [0., .4, .6, .8, .9]:
            selected = class_mask & (confidence >= cut)
            row = {'class': label, 'confidence_cut': cut, 'all_class_events': int(class_mask.sum()), 'selected_events': int(selected.sum()),
                   'coverage': float(selected.sum()/max(class_mask.sum(), 1)), 'resolutions': {}}
            if reco_vertex is not None:
                row['resolutions']['vertex_mm'] = resolution(reco_vertex-true_vertex, selected & frame.vertex_supervision_valid.to_numpy())
            for name, value in residuals.items():
                mask = selected & frame.visible_target_valid.to_numpy()
                if name in ['wrapped_delta_phi_rad', 'pT_relative']:
                    mask &= pt_true > .1
                row['resolutions'][name] = resolution(value, mask)
            result['groups'].append(row)
    if residuals:
        # Fixed energy/position bins for the component-resolution contract.
        result['strata'] = []
        for field, bins in [('initE', [0, 1e5, 5e5, 1e6, np.inf]), ('initZ', [-np.inf, -2700, -2300, -1900, np.inf])]:
            for a, b in zip(bins[:-1], bins[1:]):
                mask = (frame[field].to_numpy() >= a)&(frame[field].to_numpy() < b)&frame.visible_target_valid.to_numpy()
                result['strata'].append({'field': field, 'range': [float(a), float(b)], 'events': int(mask.sum()),
                    'resolution': {name: resolution(value, mask & (pt_true > .1) if name in ['wrapped_delta_phi_rad', 'pT_relative'] else mask) for name, value in residuals.items()}})
    return result


def auxiliary_quality(frame, config, reference, weights=None):
    result = {}
    for task in config.get('auxiliaries', []):
        count = {'vertex': 3, 'kinematics': 4, 'semantics': 3, 'charm': 1}[task]
        prediction = frame[[f'pred_{task}_{i}' for i in range(count)]].to_numpy()
        if task == 'vertex':
            target = np.stack(frame.vertex_mm)/[210, 210, 3000]
            valid = frame.vertex_supervision_valid.to_numpy()
        elif task == 'kinematics':
            momentum = np.stack(frame.visible_p3_GeV)
            target = np.column_stack([np.arcsinh(momentum[:, 0]), np.arcsinh(momentum[:, 1]), np.log1p(np.maximum(momentum[:, 2], 0)), np.log1p(frame.visible_E_GeV)])/7.
            valid = frame.visible_target_valid.to_numpy() & (momentum[:, 2] >= 0)
        elif task == 'semantics':
            target = np.stack(frame.bar_semantic_fractions)
            valid = frame.bar_semantic_valid.to_numpy()
        else:
            target = frame.charm_presence.to_numpy(dtype=float)[:, None]
            valid = frame.charm_label_valid.to_numpy()
        prediction, target = prediction[valid], target[valid]
        mean = np.asarray(reference[task])
        row = {'events': int(valid.sum()), 'prediction_std': prediction.std(0).tolist(), 'target_std': target.std(0).tolist(), 'reference': 'constant mean fitted only to selected 10k train labels'}
        if task in ['vertex', 'kinematics']:
            squared_error = (prediction-target)**2
            constant_error = (mean-target)**2
            row.update(component_MSE=squared_error.mean(0).tolist(),
                       constant_train_mean_MSE=constant_error.mean(0).tolist(),
                       component_bias=np.mean(prediction-target, axis=0).tolist(),
                       component_correlation=[float(np.corrcoef(prediction[:, i], target[:, i])[0, 1]) if prediction[:, i].std() > 0 else None for i in range(count)])
            if weights is not None:
                eligible_weights = weights[:, valid]
                denominator = eligible_weights.sum(1)
                samples = np.divide(eligible_weights@(squared_error-constant_error), denominator[:, None],
                                    out=np.full((len(weights), count), np.nan), where=denominator[:, None] > 0)
                row['component_MSE_minus_constant'] = [
                    {'estimate': float(delta),
                     'block_CI95': np.nanquantile(samples[:, i], [.025, .975]).tolist(),
                     'valid_replicates': int(np.isfinite(samples[:, i]).sum())}
                    for i, delta in enumerate((squared_error-constant_error).mean(0))]
            if task == 'kinematics':
                row['negative_encoded_pz_E_counts'] = (prediction[:, 2:] < 0).sum(0).tolist()
        elif task == 'semantics':
            p = np.exp(prediction-prediction.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            row.update(logit_std=row['prediction_std'], prediction_std=p.std(0).tolist(), score_space='semantic fractions',
                       soft_cross_entropy=float(-np.mean(np.sum(target*np.log(np.maximum(p, 1e-15)), axis=1))),
                       constant_train_mean_cross_entropy=float(-np.mean(np.sum(target*np.log(np.maximum(mean, 1e-15)), axis=1))))
        else:
            score = prediction[:, 0]
            probability = np.exp(-np.logaddexp(0., -score))
            row.update(logit_std=row['prediction_std'], prediction_std=[float(probability.std())], score_space='charm probability',
                       constant_train_mean_BCE=float(-np.mean(target*np.log(np.maximum(mean, 1e-15))+(1-target)*np.log(np.maximum(1-mean, 1e-15)))),
                       AP=float(average_precision_score(target[:, 0], score)), AUROC=float(roc_auc_score(target[:, 0], score)),
                       BCE=float(np.mean(np.logaddexp(0, score)-target[:, 0]*score)), prevalence=float(target.mean()))
        result[task] = row
    return result


def calculate(entry, common, weights, reference):
    name, frame, manifest = entry
    config = manifest['config']
    cache = MetricCache(frame.label.to_numpy(), frame[[f'p{c}' for c in range(4)]].to_numpy())
    domains = {'raw': np.ones(len(frame)), 'common_support': common}
    point = {k: cache.compute(w)[0] for k, w in domains.items()}
    samples = {k: np.stack([cache.compute(bw*w)[0] for bw in weights]) for k, w in domains.items()}
    tau_masks = tau_domains(frame)
    paired_tau = {k: v for k, v in tau_masks.items() if k.startswith('branch_') and k != 'branch_unknown' or k.startswith('tau_visible_any_')}
    tau_point = {k: tau_statistics(cache, v.astype(float)) for k, v in paired_tau.items()}
    tau_samples = {k: np.stack([tau_statistics(cache, v*bw) for bw in weights]) for k, v in paired_tau.items()}
    report = {'name': name, 'variant': config.get('variant'), 'seed': config.get('seed'),
              'metrics': cache.report(), 'common_support': cache.report(common),
              'probabilities': prediction_diagnostics(cache.p), 'operating_points': operating_tables(frame, common),
              'score_observable_correlations': conditional_correlations(frame), 'kinematics': kinematics(frame, config) if 'pred_vertex_0' in frame else None,
              'all_empty': cache.report(frame.all_empty.to_numpy(dtype=float))}
    report['auxiliary_quality'] = auxiliary_quality(frame, config, reference, weights)
    report['tau_domain_block_intervals'] = {}
    for domain in paired_tau:
        intervals = tau_interval(tau_point[domain], tau_samples[domain])
        for value in intervals.values():
            value['estimate'] = value.pop('delta')
        report['tau_domain_block_intervals'][domain] = intervals
    report['tau_domain_metrics'] = {k: dict(zip(TAU_NAMES, tau_statistics(cache, mask.astype(float)).tolist())) for k, mask in tau_masks.items()}
    report['common_support_tau_domain_metrics'] = {k: dict(zip(TAU_NAMES, tau_statistics(cache, mask*common).tolist())) for k, mask in tau_masks.items()}
    if all(f'q{i}' in frame for i in range(6)):
        q = frame[[f'q{i}' for i in range(6)]].to_numpy()
        mask = frame.six_label.to_numpy() >= 0
        y = frame.six_label.to_numpy()[mask]
        report['six_way'] = {'events': int(mask.sum()), 'masked_events': int((~mask).sum()),
            'NLL': float(log_loss(y, q[mask], labels=range(6))), 'confusion_counts': confusion_matrix(y, q[mask].argmax(1), labels=range(6)).tolist(),
            'class_AP': [float(average_precision_score(y == i, q[mask, i])) if (y == i).any() else None for i in range(6)],
            'class_probability_std': q[mask].std(0).tolist(),
            'class_true_counts': np.bincount(y, minlength=6).tolist(),
            'class_prevalence': (np.bincount(y, minlength=6)/len(y)).tolist(),
            'class_predicted_counts': np.bincount(q[mask].argmax(1), minlength=6).tolist(),
            'maximum_tau_sum_difference': float(np.max(np.abs(q[:, 2:5].sum(1)-frame.p2.to_numpy())))}
        if report['six_way']['maximum_tau_sum_difference'] > 2e-6:
            raise ValueError('six-way to four-way sum inconsistent')
    return name, report, point, samples, tau_point, tau_samples


def object_coverage(objects, truth, weights):
    key = ['run', 'chunk', 'row']
    frame = truth[key+['label', 'tau_decay_valid']].merge(objects.drop(columns=['label', 'split'], errors='ignore'), on=key, validate='one_to_one').sort_values(key).reset_index(drop=True)
    rows = []
    for label in ['all', 0, 1, 2, 3]:
        mask = np.ones(len(frame), bool) if label == 'all' else frame.label.eq(label).to_numpy()
        for cut in [np.inf, 100., 20., 5., 1.]:
            selected = mask & frame.vertex_present.to_numpy() & (frame.vertex_sigma_z_mm.fillna(np.inf).to_numpy() <= cut)
            residual = np.stack(frame.vertex_primary_residual_mm)
            tau_residual = np.stack(frame.vertex_tau_residual_mm)
            primary_ok = selected & (np.linalg.norm(residual[:, :2], axis=1) < 1)&(np.abs(residual[:, 2]) < 10)
            tau_ok = selected & (np.linalg.norm(tau_residual[:, :2], axis=1) < 1)&(np.abs(tau_residual[:, 2]) < 10)
            eligible = mask & frame.primary_vertex_reconstructable.to_numpy()
            tau_eligible = mask & frame.tau_vertex_reconstructable.to_numpy()
            def efficiency_interval(passed, denominator):
                sums = weights@denominator.astype(float)
                values = np.divide(weights@passed.astype(float), sums, out=np.full(len(sums), np.nan), where=sums > 0)
                finite = values[np.isfinite(values)]
                return {'rate': float(passed.sum()/denominator.sum()) if denominator.sum() else None,
                        'block_CI95': np.quantile(finite, [.025, .975]).tolist() if len(finite) else [None, None]}
            rows.append({'class': label, 'sigma_z_cut_mm': cut, 'all_events': int(mask.sum()), 'selected': int(selected.sum()),
                'coverage': float(selected.sum()/max(mask.sum(), 1)), 'primary_eligible': int(eligible.sum()), 'tau_eligible': int(tau_eligible.sum()),
                'primary_matched_all': int(primary_ok.sum()), 'primary_matched_eligible': int((primary_ok & eligible).sum()),
                'tau_matched_all': int(tau_ok.sum()), 'tau_matched_eligible': int((tau_ok & tau_eligible).sum()),
                'primary_efficiency': efficiency_interval(primary_ok & eligible, eligible),
                'tau_efficiency': efficiency_interval(tau_ok & tau_eligible, tau_eligible),
                'primary_resolution_mm': resolution(residual, selected), 'tau_resolution_mm': resolution(tau_residual, selected & frame.tau_decay_valid.to_numpy())})
    return rows


def sample_strata(frame):
    tau = frame[frame.label.eq(2)]
    branches = []
    for branch, group in tau.groupby('tau_branch', sort=True):
        branches.append({'branch': branch, 'events': len(group),
            **{flag: int(group[flag].eq(True).sum()) for flag in ['tau_visible_any', 'tau_visible_pixel', 'tau_visible_tracker', 'tau_decay_contained']},
            'direct_prongs': {str(k): int(v) for k, v in group.tau_direct_prongs.value_counts(dropna=False).items()},
            'direct_prongs_hit': {str(k): int(v) for k, v in group.tau_direct_prongs_hit.value_counts(dropna=False).items()}})
    return {'validation_events': len(frame), 'events_by_class': frame.label.value_counts().sort_index().to_dict(),
            'tau_branches': branches, 'tau_decay_location': tau.tau_decay_location.value_counts().to_dict(),
            'tau_decay_valid_events': int(tau.tau_decay_valid.sum()),
            'charm_events_by_class': frame.groupby('label').charm_presence.sum().to_dict(),
            'empty_events_by_class': frame.groupby('label').all_empty.sum().to_dict(),
            'measurement_containment_valid': int(frame.response_containment_valid.sum()),
            'measurement_contained_proxy': int(frame.response_contained_proxy.eq(True).sum()),
            'scope': 'accepted full validation, no truth-conditioned event rejection'}


def run(args):
    started = time.monotonic()
    key = ['run', 'chunk', 'row']
    truth = pd.read_parquet(args.root/'truth.parquet', columns=TRUTH_FIELDS)
    catalog = pd.read_parquet(args.root/'catalog.parquet')
    objects = pd.read_parquet(args.root/'objects.parquet')
    configs = [c for p in args.campaign for c in json.loads(p.read_text())['experiments']]
    directories = [args.root/'experiments'/c['name'] for c in configs]
    directories += sorted((args.root/'experiments').glob('s5_hgb_*'))
    entries = []
    for directory in directories:
        manifest = json.loads((directory/'manifest.json').read_text())
        if manifest['status'] != 'complete':
            raise RuntimeError('incomplete experiment '+directory.name)
        frame = pd.read_parquet(directory/'validation_predictions.parquet').sort_values(key).reset_index(drop=True)
        for supplement in [truth, catalog[key+['pixel_channels', 'bar_channels', 'tracker_channels', 'all_empty']],
                           objects[key+['pixel_tracks', 'tracker_tracks', 'cluster_count', 'vertex_sigma_z_mm']]]:
            columns = [c for c in supplement if c not in key and c not in frame]
            if 'tau_branch' in supplement and 'tau_branch' in frame:
                frame = frame.drop(columns='tau_branch')
                columns.append('tau_branch')
            if columns:
                frame = frame.merge(supplement[key+columns], on=key, how='left', validate='one_to_one', sort=False)
        diagnostics = pd.read_parquet(args.root/'measurement_diagnostics.parquet')
        frame = frame.merge(diagnostics, on=key, how='left', validate='one_to_one', sort=False)
        if len(frame) != 10000 or frame.six_label.isna().any():
            raise ValueError('incomplete full validation or missing truth join')
        if entries and not frame[key].equals(entries[0][1][key]):
            raise ValueError('paired validation event keys differ')
        entries.append((directory.name, frame, manifest))
    common, support = common_support_weights(entries[0][1])
    weights, block_summary = bootstrap_weights(entries[0][1], args.bootstrap)
    from analysis.flavour.s5.dataset import PhysicsDataset
    reference_dataset = PhysicsDataset(args.root, 'train', {**configs[0], 'use_objects': False, 'bucket_size': 0})
    reference = {k: reference_dataset.targets[k][reference_dataset.targets[k+'_mask']].mean(0).tolist() for k in ['vertex', 'kinematics', 'semantics', 'charm']}
    reference_dataset.native.close()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        computed = list(executor.map(lambda entry: calculate(entry, common, weights, reference), entries))
    results = {row[0]: row for row in computed}
    # Four-pass screening and longer confirmation must never share a control.
    def cohort(config):
        return config.get('analysis_cohort', f"passes{config['epochs']}_labels{config['label_budget']}")
    index = {(cohort(row[2]['config']), row[2]['config'].get('variant'), row[2]['config'].get('seed')): row[0] for row in entries}
    if len(index) != len(entries):
        raise ValueError('ambiguous cohort/variant/seed in evaluation matrix')
    pairs = []
    for group, variant, seed in index:
        if variant == 'direct4' or (group, 'direct4', seed) not in index:
            continue
        pairs.append((group, index[(group, 'direct4', seed)], index[(group, variant, seed)]))
    for a, b in [('vertex', 'relative'), ('relative', 'relative_objects'), ('vertex_kinematics', 'kinematics_relative'),
                 ('absolute_coordinates', 'relative'), ('kinematics_absolute', 'kinematics_relative'),
                 ('vertex_kinematics', 'kinematics_standard_linear'),
                 ('vertex_kinematics', 'kinematics_standard_mlp'),
                 ('kinematics_standard_linear', 'kinematics_standard_mlp')]:
        for group in sorted({g for g, v, s in index}):
            for seed in [17, 42, 2026]:
                if (group, a, seed) in index and (group, b, seed) in index:
                    pairs.append((group, index[(group, a, seed)], index[(group, b, seed)]))
    comparisons = []
    for group, a, b in pairs:
        ra, rb = results[a], results[b]
        comparisons.append({'cohort': group, 'a': a, 'b': b,
            'metrics': {d: interval_report(rb[2][d]-ra[2][d], rb[3][d]-ra[3][d]) for d in ['raw', 'common_support']},
            'tau_domains': {d: tau_interval(rb[4][d]-ra[4][d], rb[5][d]-ra[5][d]) for d in ra[4]}})
    seed_means = []
    for group, variant in sorted({(g, v) for g, v, s in index if v != 'direct4'}):
        seeds = [s for s in [17, 42, 2026] if (group, variant, s) in index and (group, 'direct4', s) in index]
        if len(seeds) != 3:
            continue
        rows = [(results[index[(group, 'direct4', s)]], results[index[(group, variant, s)]]) for s in seeds]
        seed_means.append({'cohort': group, 'variant': variant, 'seeds': seeds, 'metrics': {d: interval_report(
            np.mean([b[2][d]-a[2][d] for a, b in rows], axis=0), np.mean([b[3][d]-a[3][d] for a, b in rows], axis=0),
            np.stack([b[2][d]-a[2][d] for a, b in rows])) for d in ['raw', 'common_support']}})
    output = args.root/'evaluation'/args.tag
    output.mkdir(parents=True, exist_ok=True)
    validation_truth = catalog[catalog.split.eq('validation')][key].merge(truth, on=key, validate='one_to_one')
    validation_objects = validation_truth[key].merge(objects, on=key, validate='one_to_one')
    report = {'experiments': [r[1] for r in computed], 'comparisons': comparisons, 'seed_means': seed_means,
              'sample_strata': sample_strata(entries[0][1]),
              'object_resolution_coverage': object_coverage(validation_objects, validation_truth, weights),
              'bootstrap': {'replicates': args.bootstrap, 'blocks': block_summary, 'scope': 'pointwise event-block CI95 conditional on trained weights; reused validation, no test claim'},
              'common_support': support, 'calibration_test_used': False, 'elapsed_seconds': time.monotonic()-started}
    write_json(output/'summary.json', report)
    pd.DataFrame([{'name': r[0], **{k: r[1]['metrics'][k] for k in ['macro_AP', 'macro_AUROC', 'NLL', 'class_2_AP', 'class_2_eff_at_0.01_FPR', 'class_2_eff_at_0.001_FPR']}} for r in computed]).to_csv(output/'metrics.csv', index=False)
    from analysis.flavour.s5.plots import plot_results
    plot_results(output, entries, report, args.root)
    print(json.dumps({'output': str(output), 'models': len(entries), 'pairs': len(comparisons), 'seconds': time.monotonic()-started}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, nargs='+', required=True)
    parser.add_argument('--bootstrap', type=int, default=400)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--tag', default='s5_screen')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
