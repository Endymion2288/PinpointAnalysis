"""Paired four-class S3 evaluation on frozen validation events only."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

from analysis.flavour.inventory import write_json
from analysis.flavour.s2.evaluate import (common_support_weights, error_groups,
    input_response_activity, resource_metrics, tau_metrics, tau_branch_metrics)
from analysis.flavour.s3.metrics import MetricCache, NAMES, bootstrap_weights, interval_report, prediction_diagnostics


KEY = ['run', 'chunk', 'row']


def align_frame(frame, reference):
    frame = frame.sort_values(KEY).reset_index(drop=True)
    if not frame[KEY + ['label']].equals(reference[KEY + ['label']]):
        raise ValueError('different validation events or labels in paired comparison')
    for columns in [['family_hi', 'family_lo'], ['initE', 'initX', 'initY', 'initZ']]:
        if not frame[columns].equals(reference[columns]):
            raise ValueError('different accepted family or common-support metadata')
    return frame


def domains(frame, config, hardware_model):
    active, _ = input_response_activity(frame, config, hardware_model)
    masks = {'all': np.ones(len(frame), bool), 'detector_active': ~frame.all_empty,
             'all_empty': frame.all_empty, 'pixel_centered': frame.pixel_centered,
             'large_scintillator': frame.large_scintillator,
             'tracker_active': frame.tracker_channels.gt(0), 'tracker_empty': frame.tracker_channels.eq(0),
             'tracker_only_response': frame.pixel_channels.eq(0) & frame.bar_channels.eq(0) & frame.tracker_channels.gt(0),
             'pixel_empty': frame.pixel_channels.eq(0), 'pixel_1_99': frame.pixel_channels.between(1, 99),
             'pixel_100_999': frame.pixel_channels.between(100, 999), 'pixel_ge_1000': frame.pixel_channels.ge(1000)}
    if active is not None:
        masks.update(model_input_active=active, model_input_empty=~active)
    for low, high in [(0, 100), (100, 300), (300, 1000), (1000, 2500), (2500, 6000)]:
        masks[f'energy_{low}_{high}_GeV'] = frame.initE.ge(low * 1000) & frame.initE.lt(high * 1000)
    for low, high in [(-2993, -2600), (-2600, -2200), (-2200, -1800)]:
        masks[f'vertex_z_{low}_{high}_mm'] = frame.initZ.ge(low) & frame.initZ.lt(high)
    return {name: np.asarray(mask, dtype=bool) for name, mask in masks.items()}


def contrast(names, coefficients, point, samples, group_summary, per_seed=None):
    """Every model and seed sees the very same physical block multiplicities."""
    result = {}
    for domain in ['raw', 'common_support']:
        observed = np.einsum('i,ij->j', coefficients, np.stack([point[n][domain] for n in names]))
        boot = np.einsum('i,ijk->jk', coefficients, np.stack([samples[n][domain] for n in names]))
        seed_values = None
        if per_seed:
            seed_values = np.stack([point[b][domain] - point[a][domain] for a, b in per_seed])
        result[domain] = interval_report(observed, boot, seed_values)
    return {'metrics': result, 'blocks': group_summary,
            'resampling': 'paired_class_signature_stratified_accepted_family_connected_chunk_bootstrap',
            'interval_scope': 'pointwise physical-event intervals conditional on trained seeds and fixed common-support weights',
            'replicates': len(samples[names[0]]['raw'])}


def plot_all(output, experiments, frames, root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from sklearn.metrics import roc_curve, precision_recall_curve
    from analysis.flavour.s3.metrics import CLASSES
    with PdfPages(output / 'four_class_diagnostics.pdf') as pdf:
        for item in experiments:
            name = item['name']; frame = frames[name]
            fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
            matrix = np.asarray(item['four_class']['confusion_weighted_counts'])
            norm = matrix / np.maximum(matrix.sum(1, keepdims=True), 1)
            axes[0, 0].imshow(norm, vmin=0, vmax=1, cmap='Blues')
            for i in range(4):
                for j in range(4):
                    axes[0, 0].text(j, i, f'{norm[i,j]:.1%}\n({int(matrix[i,j])})', ha='center', va='center', fontsize=7)
            axes[0, 0].set(xticks=range(4), xticklabels=CLASSES, yticks=range(4), yticklabels=CLASSES,
                           xlabel='Argmax prediction', ylabel='True class', title='Row-normalized confusion')
            for c, label in enumerate(CLASSES):
                truth = frame.label.eq(c)
                fpr, tpr, _ = roc_curve(truth, frame[f'p{c}'])
                precision, recall, _ = precision_recall_curve(truth, frame[f'p{c}'])
                axes[0, 1].plot(fpr, tpr, label=label)
                axes[1, 0].step(recall, precision, where='post', label=label)
            axes[0, 1].set(xlabel='Background acceptance', ylabel='Class efficiency', title='One-vs-rest ROC')
            axes[1, 0].set(xlabel='Recall', ylabel='Precision', title='One-vs-rest precision-recall')
            axes[0, 1].legend(fontsize=7); axes[1, 0].legend(fontsize=7)
            history = json.loads((root / 'experiments' / name / 'history.json').read_text())
            x = [h.get('epoch', h.get('iteration')) for h in history]
            axes[1, 1].plot(x, [h['validation']['NLL'] for h in history], color='C0', marker='.')
            other = axes[1, 1].twinx()
            other.plot(x, [np.mean([h['validation']['classwise'][str(c)]['PR_AUC'] for c in range(4)])
                           for h in history], color='C1', marker='.')
            axes[1, 1].set(xlabel='Epoch / boosting iteration', ylabel='Validation NLL', title='NLL checkpoint selection')
            other.set_ylabel('Four-class macro AP', color='C1')
            fig.suptitle(f'{name}: validation sample priors; uncalibrated scores', fontsize=10)
            pdf.savefig(fig); plt.close(fig)
    plot_resources(output, experiments)


def plot_resources(output, experiments):
    """Keep GPU subtypes separate and identify points without overlapping names."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    rows = [e for e in experiments if e.get('allocated_gpu_hours') is not None]
    if not rows:
        return
    with PdfPages(output / 'four_class_resource_tradeoff.pdf') as pdf:
        for gpu in sorted({e.get('GPU') or 'Unspecified GPU' for e in rows}):
            group = [e for e in rows if (e.get('GPU') or 'Unspecified GPU') == gpu]
            # Small pages remain readable when all confirmation seeds are added.
            for start in range(0, len(group), 12):
                page = group[start:start + 12]
                fig = plt.figure(figsize=(12, 9), layout='constrained')
                grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
                axes = [fig.add_subplot(grid[0, i]) for i in range(2)]
                table_rows = []
                for index, item in enumerate(page, start + 1):
                    cost = item['allocated_gpu_hours']
                    for axis, metric in zip(axes, ['macro_AP', 'NLL']):
                        axis.scatter(cost, item['four_class'][metric], s=25)
                        axis.annotate(str(index), (cost, item['four_class'][metric]),
                                      xytext=(4, 4), textcoords='offset points', fontsize=8)
                        axis.set(xlabel='Allocated device-hours (bounds / shared charges included)', ylabel=metric)
                    table_rows.append([index, item['name'], f'{cost:.4f}',
                                       f'{item["four_class"]["macro_AP"]:.4f}',
                                       f'{item["four_class"]["NLL"]:.4f}'])
                table_axis = fig.add_subplot(grid[1, :]); table_axis.axis('off')
                table = table_axis.table(cellText=table_rows,
                    colLabels=['ID', 'Frozen experiment', 'Device-hours', 'Macro AP', 'NLL'],
                    colWidths=[.04, .63, .11, .11, .11], loc='center', cellLoc='left')
                table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.6)
                fig.suptitle(f'{gpu}: frozen validation quality and allocation charges\n'
                             'GPU subtypes shown separately; operational costs do not measure model-only speedup', fontsize=11)
                pdf.savefig(fig); plt.close(fig)


def run(args):
    started = time.perf_counter(); args.output.mkdir(parents=True, exist_ok=True)
    campaigns = [json.loads(p.read_text()) for p in [args.campaign, args.confirmation] if p is not None]
    required = [c['name'] for campaign in campaigns for c in campaign['experiments']]
    catalog = pd.read_parquet(args.root / 'catalog.parquet', columns=KEY + ['all_empty', 'pixel_channels', 'bar_channels', 'tracker_channels'])
    paths = sorted((args.root / 'experiments').iterdir())
    completed, pending = [], []
    for path in paths:
        mf = path / 'manifest.json'
        if mf.exists() and json.loads(mf.read_text()).get('status') == 'complete':
            completed.append((path, json.loads(mf.read_text())))
    completed_names = {p.name for p, _ in completed}
    pending += [{'type': 'experiment', 'name': n} for n in required if n not in completed_names]
    if not completed:
        raise ValueError('no complete experiments')
    reference = pd.read_parquet(completed[0][0] / 'validation_predictions.parquet').sort_values(KEY).reset_index(drop=True)
    fixed_weight, support = common_support_weights(reference)
    block_weights, block_summary = bootstrap_weights(reference, args.bootstrap)
    frames, point, samples = {}, {}, {}
    experiments, strata, examples = [], [], []
    allocation_reports = {}
    for path, manifest in completed:
        frame = align_frame(pd.read_parquet(path / 'validation_predictions.parquet'), reference)
        frame = frame.merge(catalog, on=KEY, validate='one_to_one', how='left')
        if frame.all_empty.isna().any():
            raise ValueError('missing accepted response catalog row')
        frames[path.name] = frame
        cache = MetricCache(frame.label.to_numpy(), frame[[f'p{c}' for c in range(4)]].to_numpy())
        point[path.name] = {domain: cache.compute(w)[0] for domain, w in [('raw', None), ('common_support', fixed_weight)]}
        samples[path.name] = {}
        for domain, w in [('raw', np.ones(len(frame))), ('common_support', fixed_weight)]:
            samples[path.name][domain] = np.stack([cache.compute(bw * w)[0] for bw in block_weights])
        # Baseline symlinks consume their existing, read-only S2 allocation report.
        source_root = path.resolve().parent.parent
        if source_root not in allocation_reports:
            acc = source_root / 'campaign_accounting.json'
            allocation_reports[source_root] = json.loads(acc.read_text()).get('experiments', {}) if acc.exists() else {}
        resource = resource_metrics(manifest, allocation_reports[source_root].get(path.name))
        tau = tau_metrics(frame)
        experiments.append({'name': path.name, 'config': manifest['config'], 'source_root': str(source_root),
            'four_class': cache.report(), 'common_support_four_class': cache.report(fixed_weight), 'tau': tau,
            'prediction_diagnostics': prediction_diagnostics(cache.p),
            'parameters': manifest['parameters'], 'GPU': manifest['gpu_name'],
            'peak_gpu_mib': manifest.get('peak_gpu_allocated_mib'), **resource})
        for domain, mask in domains(frame, manifest['config'], manifest.get('hardware_model', True)).items():
            conditional = MetricCache(frame.loc[mask, 'label'].to_numpy(), frame.loc[mask, [f'p{c}' for c in range(4)]].to_numpy())
            strata.append({'experiment': path.name, 'stratum': domain, 'four_class': conditional.report(),
                           'tau': tau_metrics(frame.loc[mask])})
        for branch in ['electron', 'muon', 'hadronic', 'unknown', 'ambiguous']:
            strata.append({'experiment': path.name, 'stratum': 'tau_' + branch,
                'tau': tau_branch_metrics(frame, branch, tau.get('FPR_0.01', {}).get('threshold'))})
        examples.append(error_groups(frame, path.name))
        print(path.name, 'four-class bootstrap complete', flush=True)
    pair_spec = json.loads(args.comparisons.read_text())
    paired, seed_means, interactions = [], [], []
    for pair in pair_spec['pairs']:
        names = [pair['a'], pair['b']]
        if any(n not in point for n in names):
            pending.append({'type': 'pair', **pair}); continue
        paired.append({**pair, **contrast(names, [-1, 1], point, samples, block_summary)})
    for group in pair_spec.get('seed_mean_comparisons', []):
        names = [n for p in group['pairs'] for n in [p['a'], p['b']]]
        if any(n not in point for n in names):
            pending.append({'type': 'seed_mean', **group}); continue
        pairs = [(p['a'], p['b']) for p in group['pairs']]
        coefficients = np.tile([-1., 1.], len(pairs)) / len(pairs)
        seed_means.append({**group, **contrast(names, coefficients, point, samples, block_summary, pairs)})
    names = ['s3_f16_context_latent_s42_e4', 's3_f16_context_latent_1mm_s42_e4',
             's3_a4_context_latent_s42_e4', 's3_a4_context_latent_1mm_s42_e4']
    if all(n in point for n in names):
        interactions.append({'experiments': names, 'definition': '(fine_native-fine_1mm)-(4mm_native-4mm_1mm)',
                             **contrast(names, [1, -1, -1, 1], point, samples, block_summary)})
    else:
        pending.append({'type': 'factorial_interaction', 'experiments': names})
    spread_groups = {}
    for e in experiments:
        if 'seed' not in e['config']:
            continue
        identity = {k: v for k, v in e['config'].items() if k not in ['name', 'seed', 'role']}
        spread_groups.setdefault(json.dumps(identity, sort_keys=True), []).append(e)
    spread = []
    for identity, group in spread_groups.items():
        seeds = [e['config']['seed'] for e in group]
        if len(seeds) != len(set(seeds)):
            raise ValueError('duplicate trained seeds in spread group')
        values = np.stack([point[e['name']]['raw'] for e in group])
        spread.append({'config': json.loads(identity), 'seeds': seeds, 'experiments': [e['name'] for e in group],
            'three_seed_confirmation_complete': set([17, 42, 2026]).issubset(seeds),
            'mean': dict(zip(NAMES, np.mean(values, axis=0).tolist())),
            'sample_SD': dict(zip(NAMES, np.std(values, axis=0, ddof=1).tolist())) if len(values) > 1 else None})
    summary = {'utc': datetime.now(timezone.utc).isoformat(), 'scope': 'S3 frozen validation only; calibration/test unused',
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'experiments': experiments, 'strata': strata, 'paired_comparisons': paired, 'seed_mean_comparisons': seed_means,
        'factorial_interactions': interactions, 'training_seed_spread': spread, 'common_support': support,
        'resampling_blocks': block_summary, 'bootstrap_replicates': args.bootstrap,
        'pending': pending, 'requested_screen_and_pairs_complete': not pending,
        'confirmation_configuration_present': args.confirmation is not None, 'S3_acceptance_asserted': False,
        'metric_definitions': {'AP': 'noninterpolated one-vs-rest AP with exact ties retained',
            'macro_AP_AUROC': 'unweighted mean of four classes; null if any class statistic undefined',
            'macro_F1': 'four argmax classes, zero division=0',
            'balanced_accuracy': 'mean recall of classes with positive evaluation weight',
            'NLL': 'float64 clipped normalized four-class log loss', 'Brier': 'sum of four squared probability errors per event',
            'common_support': 'fixed accepted S2 coarse E/vertex weights; target Z unmatched',
            'CI95': 'pointwise paired block percentile interval; does not include training stochasticity',
            'training_seed_SD': 'sample SD across separately trained seeds; not event replication'},
        'program_wall_seconds': time.perf_counter() - started}
    # MetricCache reports undefined values as null in domains with missing classes.
    write_json(args.output / 'summary.json', summary)
    pd.concat(examples, ignore_index=True).to_parquet(args.output / 'error_groups.parquet', index=False)
    np.savez_compressed(args.output / 'bootstrap_statistics.npz', metric_names=np.array(NAMES),
        **{f'{name}__{domain}': values for name, d in samples.items() for domain, values in d.items()})
    if not args.no_plots:
        plot_all(args.output, experiments, frames, args.root)
    if args.strict and pending:
        raise ValueError(f'{len(pending)} requested experiments or contrasts pending; see summary.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--confirmation', type=Path)
    parser.add_argument('--comparisons', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=400)
    parser.add_argument('--strict', action='store_true')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    if args.bootstrap <= 0:
        parser.error('--bootstrap must be positive')
    run(args)


if __name__ == '__main__':
    main()
