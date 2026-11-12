"""Render S2 resolution and paired-effect figures from completed evaluations."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
import numpy as np

from analysis.flavour.inventory import write_json


FAMILIES = {'part': 'ParT', 'particlenet': 'ParticleNet'}


def short_name(name):
    return name.removesuffix('_e4_v2')


def resolution_figure(output, experiments):
    """Keep the prespecified seed42 spatial screen separate from its controls."""
    screen = [e for e in experiments if e['config'].get('role') == 'spatial_screen'
              and e['config'].get('response_summary_version') == 2
              and e['config'].get('seed') == 42]
    if not screen:
        return []
    pitches = [0., .1, .2, 1.]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    getters = [lambda e: e['tau']['PR_AUC'], lambda e: e['matched_tau'].get('PR_AUC'),
               lambda e: e['tau']['FPR_0.01']['tau_efficiency'],
               lambda e: e['metrics']['balanced_accuracy']]
    titles = ['Tau average precision', 'Tau AP on fixed E/vertex common support',
              'Tau efficiency at 1% background acceptance', 'Four-class balanced accuracy']
    for family, label in FAMILIES.items():
        color = 'tab:blue' if family == 'part' else 'tab:orange'
        for projection in [False, True]:
            cells = {e['config']['pitch_mm']: e for e in screen
                     if e['config']['architecture'] == family
                     and e['config'].get('projection', False) == projection}
            if not cells:
                continue
            for axis, getter in zip(axes.flat, getters):
                values = [getter(cells[p]) if p in cells else None for p in pitches]
                axis.plot(range(4), [np.nan if v is None else v for v in values],
                          color=color, marker='s' if projection else 'o',
                          linestyle='none' if projection else '-',
                          label=f'{label}: {"xz/yz projections" if projection else "joint positions"}')
    for axis, title in zip(axes.flat, titles):
        axis.set(xticks=range(4), xticklabels=['Native', '100 um', '200 um', '1 mm'],
                 xlabel='Pixel representation (bar response fixed)', ylabel=title)
        axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('S2 validation: seed42, four epochs, NLL-selected checkpoints\n'
                 'Only completed spatial runs; missing points are unmeasured; sample priors', fontsize=10)
    for suffix in ['png', 'pdf']:
        fig.savefig(output/f'resolution_comparison.{suffix}', dpi=160)
    plt.close(fig)
    return [e['name'] for e in screen]


def interval_panel(axis, rows, key):
    """Draw intervals directly so a biased bootstrap CI need not contain its estimate."""
    for i, row in enumerate(rows):
        effect = row if key is None else row.get(key, {})
        if effect and effect.get('delta') is not None:
            if effect.get('CI95') is not None:
                low, high = effect['CI95']
                axis.hlines(i, low, high, color='tab:blue', linewidth=1.6)
                axis.plot([low, high], [i, i], '|', color='tab:blue', markersize=7)
            axis.plot(effect['delta'], i, 'o', color='tab:blue', markersize=4)
        else:
            axis.text(.5, i, 'Not measured', ha='center', va='center',
                      transform=axis.get_yaxis_transform(), fontsize=7)
    axis.axvline(0., color='grey', linewidth=.8)
    axis.axvline(.01, color='grey', linestyle=':', linewidth=.8)
    axis.set(ylim=(len(rows)-.5, -.5), xlabel='Tau AP difference (B - A)')
    axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.grid(axis='x', alpha=.2)


def paired_figures(output, rows, seed_mean=False):
    if not rows:
        return []
    basename = 'seed_mean_comparisons' if seed_mean else 'paired_comparisons'
    pages = []
    with PdfPages(output/f'{basename}.pdf') as pdf:
        for start in range(0, len(rows), 10):
            selected = rows[start:start+10]
            show_time = not seed_mean and any(r.get('matched_allocated_time', {}).get('comparison_measured')
                                             for r in selected)
            fig, axes = plt.subplots(1, 3 if show_time else 2, sharey=True,
                                     figsize=(15 if show_time else 11, max(4, len(selected)*.68+2)),
                                     constrained_layout=True)
            interval_panel(axes[0], selected, None)
            interval_panel(axes[1], selected, 'common_support_tau_AP')
            axes[0].set_title('Sample priors: 95% event bootstrap CI', fontsize=10)
            axes[1].set_title('Common support: conditional 95% CI', fontsize=10)
            if show_time:
                timed = []
                for row in selected:
                    result = row.get('matched_allocated_time', {})
                    timed.append({'delta': result['tau_AP_B_minus_A']}
                                 if result.get('comparison_measured') else {})
                interval_panel(axes[2], timed, None)
                axes[2].set_title('Matched GPU allocation: epoch resolution\nPoint estimates; no interval computed', fontsize=10)
            labels = [row['name'].replace('_', ' ') if seed_mean else
                      f"B: {short_name(row['b'])}\nA: {short_name(row['a'])}" for row in selected]
            axes[0].set_yticks(range(len(selected)), labels, fontsize=7)
            fig.suptitle(('S2 mean of per-seed AP differences; shared physical-event bootstrap\n'
                          'Intervals conditional on trained seeds; training spread is reported separately'
                          if seed_mean else
                          'S2 validation paired effects; identical physical events, accepted family/chunk blocks')
                         + '\nDotted line: prespecified +0.01 AP gain; pointwise intervals', fontsize=10)
            pdf.savefig(fig)
            if start == 0:
                fig.savefig(output/f'{basename}_first_page.png', dpi=160)
            plt.close(fig)
            pages.append({'page': len(pages)+1,
                          'comparisons': [r['name'] if seed_mean else
                                          {'a': r['a'], 'b': r['b'], 'question': r['question']}
                                          for r in selected]})
    return pages


def seed_spread_figure(output, summary):
    """Show fitted-seed spread separately from physical-event intervals."""
    groups = [g for g in summary.get('seed_spread', [])
              if g['three_seed_confirmation_complete']]
    if not groups:
        return []
    seeds = [17, 42, 2026]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    labels, records = [], []
    for number, group in enumerate(groups):
        selected = {}
        for experiment in summary['experiments']:
            config = {k: v for k, v in experiment['config'].items()
                      if k not in ['name', 'role', 'seed']}
            if config == group['config']:
                selected[experiment['config']['seed']] = experiment
        models = [selected[seed] for seed in seeds]
        config = group['config']
        pitch = 'Native' if config['pitch_mm'] == 0 else f"{config['pitch_mm']:g} mm"
        labels.append(f"{FAMILIES[config['architecture']]}\n{pitch}, joint")
        record = {'experiments': [m['name'] for m in models], 'seeds': seeds}
        for axis, key, getter in zip(axes, ['raw', 'common_support'],
                [lambda m: m['tau']['PR_AUC'], lambda m: m['matched_tau']['PR_AUC']]):
            values = np.asarray([getter(model) for model in models])
            mean, std = float(values.mean()), float(values.std(ddof=1))
            for j, (seed, value) in enumerate(zip(seeds, values)):
                axis.scatter(number + (j - 1) * .13, value, color=f'C{j}',
                             label=f'Seed {seed}' if number == 0 else None, s=25)
            axis.errorbar(number, mean, yerr=std, fmt='D', color='black',
                          markersize=4, capsize=5,
                          label='Mean +/- sample SD' if number == 0 else None)
            record[key] = {'AP_values': values.tolist(), 'mean': mean, 'sample_std': std}
        records.append(record)
    for axis, title in zip(axes, ['Tau AP: observed sample priors',
                                  'Tau AP: fixed E/vertex common support']):
        axis.set(title=title, ylabel='Tau average precision',
                 xticks=range(len(labels)), xticklabels=labels,
                 xlim=(-.5, len(labels) - .5))
        axis.legend(fontsize=7)
        axis.grid(axis='y', alpha=.2)
    figure.suptitle('S2: three trained seeds on the same physical validation events\n'
                   'Sample standard deviation across fits; not a confidence interval or an ensemble',
                   fontsize=10)
    for suffix in ['png', 'pdf']:
        figure.savefig(output/f'training_seed_spread.{suffix}', dpi=160)
    plt.close(figure)
    return records


def plot_evaluation(output):
    summary = json.loads((output/'summary.json').read_text())
    screen = resolution_figure(output, summary['experiments'])
    pairs = paired_figures(output, summary['paired_comparisons'])
    seeds = paired_figures(output, summary.get('seed_mean_comparisons', []), seed_mean=True)
    spread = seed_spread_figure(output, summary)
    write_json(output/'comparison_figures.json', {
        'source': str(output/'summary.json'), 'evaluation_code_commit': summary['code_commit'],
        'scope': summary['scope'], 'requested_comparisons_complete': summary['requested_comparisons_complete'],
        'pending_comparisons': len(summary['pending_comparisons']),
        'resolution_experiments': screen, 'paired_pages': pairs, 'seed_mean_pages': seeds,
        'training_seed_spread': spread,
        'statistics': 'read existing estimates and intervals; no refitting or resampling',
        'training_seed_spread_definition': 'sample standard deviation of AP across fitted seeds 17/42/2026 on the same events; not a confidence interval or an ensemble',
        'resolution_axis': 'categorical native/100um/200um/1mm; no numeric pitch assigned to native',
        'uncertainty': 'pointwise physical-event bootstrap; common-support weights and fitted seeds conditional; not simultaneous intervals',
        'matched_time': 'existing epoch-resolution point estimates; no confidence interval inferred'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    plot_evaluation(parser.parse_args().evaluation)


if __name__ == '__main__':
    main()
