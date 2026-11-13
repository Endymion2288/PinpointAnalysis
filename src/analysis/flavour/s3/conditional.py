"""Four-class domain contrasts and empirical background/purity diagnostics."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from analysis.flavour.inventory import write_json
from analysis.flavour.s3.evaluate import KEY, align_frame, contrast, domains
from analysis.flavour.s3.metrics import CLASSES, MetricCache, bootstrap_weights
from analysis.flavour.s2.evaluate import common_support_weights


def operating_points(truth, score, inclusive_threshold=None):
    """No interpolation beyond observed tied-score thresholds or empty tails."""
    truth = np.asarray(truth, dtype=bool); score = np.asarray(score)
    positive, negative = int(truth.sum()), int((~truth).sum())
    if positive == 0 or negative == 0:
        return {'positive_events': positive, 'negative_events': negative, 'defined': False}, []
    fpr, tpr, cut = roc_curve(truth, score, drop_intermediate=False)
    precision, recall, pr_cut = precision_recall_curve(truth, score)
    out = {'positive_events': positive, 'negative_events': negative, 'defined': True,
           'observed_class_prior': positive / len(truth)}
    retained = set(np.linspace(0, len(fpr) - 1, min(256, len(fpr)), dtype=int))
    for rate in [.01, .001]:
        i = int(np.flatnonzero(fpr <= rate)[-1]); threshold = cut[i]
        retained.add(i); selected = score >= threshold
        bg = int((selected & ~truth).sum()); sig = int((selected & truth).sum())
        out[f'FPR_{rate:g}'] = {'threshold': float(threshold) if np.isfinite(threshold) else None,
            'efficiency': float(tpr[i]), 'background_acceptance': float(fpr[i]),
            'signal_selected': sig, 'background_selected': bg, 'tail_statistics_sufficient': bg >= 25}
    for purity in [.5, .8, .9]:
        possible = np.flatnonzero(precision[:-1] >= purity)
        if not len(possible):
            out[f'purity_{purity:g}'] = {'reachable_with_nonempty_selection': False}
            continue
        i = int(possible[np.argmax(recall[possible])]); threshold = pr_cut[i]
        selected = score >= threshold
        bg = int((selected & ~truth).sum()); sig = int((selected & truth).sum())
        out[f'purity_{purity:g}'] = {'reachable_with_nonempty_selection': True, 'threshold': float(threshold),
            'efficiency': float(recall[i]), 'purity': float(precision[i]), 'signal_selected': sig,
            'background_selected': bg, 'selected_statistics_sufficient': sig + bg >= 25}
    if inclusive_threshold is not None:
        threshold = np.inf if inclusive_threshold['threshold'] is None else inclusive_threshold['threshold']
        selected = score >= threshold
        out['at_full_validation_1pct_FPR_threshold'] = {'threshold': inclusive_threshold['threshold'],
            'efficiency': float((selected & truth).sum() / positive),
            'background_acceptance': float((selected & ~truth).sum() / negative),
            'signal_selected': int((selected & truth).sum()), 'background_selected': int((selected & ~truth).sum())}
    curves = [{'curve': 'ROC', 'x': float(fpr[i]), 'y': float(tpr[i])} for i in sorted(retained)]
    kept = np.unique(np.linspace(0, len(precision) - 1, min(256, len(precision)), dtype=int))
    curves += [{'curve': 'PR', 'x': float(recall[i]), 'y': float(precision[i])} for i in kept]
    return out, curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--comparisons', type=Path, required=True)
    parser.add_argument('--experiments', nargs='+', required=True)
    parser.add_argument('--bootstrap', type=int, default=400)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    frames, caches, manifests = {}, {}, {}
    for name in args.experiments:
        path = args.root / 'experiments' / name
        manifest = json.loads((path / 'manifest.json').read_text())
        checks = json.loads((path / 'final_checkpoint_checks.json').read_text())
        if manifest['status'] != 'complete' or checks['status'] != 'pass':
            raise ValueError(f'{name}: complete training and CPU readback required')
        manifests[name] = manifest
        frames[name] = pd.read_parquet(path / 'validation_predictions.parquet').sort_values(KEY).reset_index(drop=True)
    reference = next(iter(frames.values()))
    weights, block_summary = bootstrap_weights(reference, args.bootstrap)
    fixed, support = common_support_weights(reference)
    catalog = pd.read_parquet(args.root / 'catalog.parquet', columns=KEY + ['all_empty', 'pixel_channels', 'bar_channels', 'tracker_channels'])
    masks = domains(reference.merge(catalog, on=KEY, validate='one_to_one'), {'modalities': ['pixel', 'bar']}, True)
    condition_names = ['pixel_centered', 'large_scintillator', 'detector_active', 'all_empty',
                       'pixel_ge_1000', 'tracker_active', 'tracker_empty']
    curve_names = ['all', *condition_names]
    curves, working, learning = [], [], []
    for name, frame in frames.items():
        align_frame(frame, reference)
        caches[name] = MetricCache(frame.label.to_numpy(), frame[[f'p{c}' for c in range(4)]].to_numpy())
        inclusive = [operating_points(frame.label.eq(c), frame[f'p{c}'])[0].get('FPR_0.01') for c in range(4)]
        for domain in curve_names:
            mask = masks[domain]
            for c in range(4):
                measured, sampled = operating_points(frame.loc[mask, 'label'].eq(c), frame.loc[mask, f'p{c}'], inclusive[c])
                working.append({'experiment': name, 'domain': domain, 'class': c, **measured})
                curves += [{'experiment': name, 'domain': domain, 'class': c, **row} for row in sampled]
        # Epoch predictions use validation dataset indices, not sorted physical keys.
        original = pd.read_parquet(args.root / 'experiments' / name / 'validation_predictions.parquet')
        epochs = pd.read_parquet(args.root / 'experiments' / name / 'epoch_predictions.parquet')
        domain_original = original.merge(catalog, on=KEY, validate='one_to_one')
        original_masks = domains(domain_original, {'modalities': ['pixel', 'bar']}, True)
        for epoch, group in epochs.groupby('epoch'):
            group = group.sort_values('row')
            if not np.array_equal(group.row, np.arange(len(original))):
                raise ValueError('incomplete epoch validation indices')
            cache = MetricCache(original.label.to_numpy(), group[[f'p{c}' for c in range(4)]].to_numpy())
            for domain in ['all', 'pixel_centered', 'large_scintillator', 'tracker_active', 'tracker_empty']:
                learning.append({'experiment': name, 'epoch': int(epoch), 'domain': domain,
                                 'primary_NLL_epoch': manifests[name]['best_epoch'], **cache.report(original_masks[domain])})
    pair_spec = json.loads(args.comparisons.read_text()); pending, reports = [], []
    # Only comparisons whose models were explicitly requested belong to this diagnostic.
    pairs = [p for p in pair_spec['pairs'] if p['a'] in frames and p['b'] in frames]
    means = [g for g in pair_spec.get('seed_mean_comparisons', []) if all(
        p['a'] in frames and p['b'] in frames for p in g['pairs'])]
    if not pairs and not means:
        pending.append({'reason': 'no preregistered contrasts covered by requested models'})
    for domain in condition_names:
        mask = masks[domain]; point, samples = {}, {}
        for name, cache in caches.items():
            point[name], samples[name] = {}, {}
            for scope, w in [('raw', mask.astype(float)), ('common_support', fixed * mask)]:
                point[name][scope] = cache.compute(w)[0]
                samples[name][scope] = np.stack([cache.compute(b * w)[0] for b in weights])
        conditional = []
        for pair in pairs:
            conditional.append({**pair, **contrast([pair['a'], pair['b']], [-1, 1], point, samples, block_summary)})
        seed_conditional = []
        for group in means:
            ps = [(p['a'], p['b']) for p in group['pairs']]
            seed_conditional.append({**group, **contrast([n for p in ps for n in p],
                np.tile([-1., 1.], len(ps)) / len(ps), point, samples, block_summary, ps)})
        reports.append({'domain': domain, 'events': int(mask.sum()),
                        'paired_comparisons': conditional, 'seed_mean_comparisons': seed_conditional})
        print(domain, 'four-class conditional bootstrap complete', flush=True)
    pd.DataFrame(curves).to_parquet(args.output / 'sampled_curves.parquet', index=False)
    write_json(args.output / 'summary.json', {'utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'S3 conditional validation diagnostics, unchanged full-validation NLL checkpoints',
        'experiments': args.experiments, 'requested_complete': not pending, 'pending': pending,
        'paired_domains': reports, 'operating_points': working, 'epoch_learning_curves': learning,
        'common_support': support, 'bootstrap_replicates': args.bootstrap,
        'denominators': 'all events of stated domain, including absent/empty response; each class versus all other classes',
        'threshold_scope': 'empirical validation; domain-specific and inclusive-threshold efficiencies separately recorded; no calibrated deployment threshold',
        'purity_scope': 'observed benchmark domain priors; no physical flux or yield assumption',
        'curve_storage': 'up to 256 points per ROC/PR curve for plotting; operating points computed on full tied-score thresholds',
        'interval_scope': 'pointwise shared accepted physical block bootstrap, conditional on fitted seeds; fixed S2 common support, not refitted by domain',
        'S4_S6_executed': False})
    plot_curves(pd.DataFrame(curves), args.output)


def plot_curves(frame, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(output / 'four_class_domain_curves.pdf') as pdf:
        for domain, part in frame.groupby('domain', sort=False):
            names = list(part.experiment.unique())
            for start in range(0, len(names), 8):
                page = names[start:start + 8]
                fig, axes = plt.subplots(2, 4, figsize=(15, 9))
                fig.subplots_adjust(bottom=.24, top=.91, hspace=.3, wspace=.3)
                for c, label in enumerate(CLASSES):
                    for index, name in enumerate(page):
                        values = part[part['class'].eq(c) & part.experiment.eq(name)]
                        for row, curve in enumerate(['ROC', 'PR']):
                            v = values[values.curve.eq(curve)]
                            axes[row, c].plot(v.x, v.y, label=name, color=f'C{index}', linewidth=1,
                                              drawstyle='steps-post' if curve == 'PR' else 'default')
                    axes[0, c].set(title=label, xlabel='Background acceptance', ylabel='Class efficiency',
                                   xlim=(0, 1), ylim=(0, 1))
                    axes[0, c].set_xscale('symlog', linthresh=.001, linscale=.5)
                    axes[1, c].set(xlabel='Class recall', ylabel='Precision', xlim=(0, 1), ylim=(0, 1))
                handles, labels = axes[0, 0].get_legend_handles_labels()
                fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=8,
                           bbox_to_anchor=(.5, .01))
                fig.suptitle(f'{domain}: validation domain priors; full-domain NLL-selected checkpoints\n'
                             f'Models {start + 1}–{start + len(page)} of {len(names)}', fontsize=11)
                pdf.savefig(fig); plt.close(fig)


if __name__ == '__main__':
    main()
