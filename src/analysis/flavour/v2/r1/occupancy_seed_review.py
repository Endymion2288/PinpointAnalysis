"""Read back three-seed occupancy contributions to the R1 class tradeoffs."""

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pandas as pd

from analysis.flavour.inventory import write_json


SEEDS = [17, 42, 2026]
CLASSES = [0, 1, 3]
DOMAINS = ['all', 'empty', 'occupancy_0_100', 'occupancy_100_1000',
           'occupancy_1000_5000', 'occupancy_5000_inf']
CONTRASTS = [
    (('B1', 'T3', 'main'), ('B4', 'T3', 'main')),
    (('B1', 'T4', 'main'), ('B4', 'T4', 'main')),
    (('B1', 'T3', 'matched_compute'), ('B4', 'T3', 'matched_compute')),
    (('B1', 'T3', 'matched_compute'), ('B1', 'T3', 'main')),
]


def readback(root):
    source = root/'interim/b1_three_seed_compute'
    evaluation = json.loads((source/'evaluation/results.json').read_text())
    review = json.loads((source/'evaluation/independent_review.json').read_text())
    config = json.loads((source/'evaluation_config.json').read_text())
    if evaluation['status'] != 'complete' or review['status'] != 'complete' or review['failed']:
        raise ValueError('Complete evaluation and independent review required')
    entries = {(e['family'], e['task'], e.get('kind', 'main'), e['seed']): e
               for e in config['models']}
    needed = {(*group, seed) for pair in CONTRASTS for group in pair for seed in SEEDS}
    raw, passed = {}, 0
    for key in sorted(needed):
        entry = entries[key]
        frame = pd.read_parquet(root/entry['predictions'])
        labels = frame.label.to_numpy()
        channels = (frame.pixel_channels+frame.bar_channels+frame.tracker_channels).to_numpy()
        masks = dict(all=np.ones(len(frame), dtype=bool), empty=frame.all_empty.to_numpy())
        for low, high in [(0, 100), (100, 1000), (1000, 5000), (5000, np.inf)]:
            masks[f'occupancy_{low}_{high}'] = (channels >= low) & (channels < high)
        predicted = frame[['p0', 'p1', 'p3']].to_numpy(dtype='float64').argmax(1)
        truth = np.where(labels == 3, 2, labels)
        raw[key] = {}
        for domain in DOMAINS:
            mask = masks[domain]
            common = mask & (labels != 2)
            counts = np.bincount(labels[mask], minlength=4)
            confusion = np.bincount(truth[common]*3+predicted[common], minlength=9).reshape(3, 3)
            report = evaluation['models'][entry['name']][domain]
            observed = report.get('T3_observed_mixture', report['T3'])
            if not np.array_equal(counts, report['raw_counts_original_labels']):
                raise ValueError(f'Raw class counts differ: {entry["name"]} {domain}')
            if not np.allclose(confusion, observed['confusion_counts'], atol=1e-7, rtol=0.):
                raise ValueError(f'Raw confusion differs: {entry["name"]} {domain}')
            passed += 13
            raw[key][domain] = {'support': counts[CLASSES], 'correct': confusion.diagonal()}
        for field in ['support', 'correct']:
            if not np.array_equal(sum(raw[key][d][field] for d in DOMAINS[2:]), raw[key]['all'][field]):
                raise ValueError(f'Occupancy partition does not recover all events: {entry["name"]}')
            passed += 3
    contrasts = []
    for left, right in CONTRASTS:
        domains = {}
        for domain in DOMAINS:
            differences, contributions, correct = [], [], []
            for seed in SEEDS:
                a, b = raw[(*left, seed)][domain], raw[(*right, seed)][domain]
                if not np.array_equal(a['support'], b['support']):
                    raise ValueError('Paired domains need identical class support')
                if not np.array_equal(a['support'], raw[(*left, SEEDS[0])][domain]['support']):
                    raise ValueError('Seeds need identical class support')
                passed += 6
                delta = a['correct']-b['correct']
                correct.append(delta.tolist())
                differences.append((100*delta/a['support']).tolist())
                contributions.append((100*delta/raw[(*left, seed)]['all']['support']).tolist())
            domains[domain] = {
                'raw_support_e_mu_NC': a['support'].tolist(),
                'delta_correct_by_seed_e_mu_NC': correct,
                'delta_recall_pp_by_seed_e_mu_NC': differences,
                'delta_recall_pp_mean_e_mu_NC': [mean(row[c] for row in differences) for c in range(3)],
                'delta_recall_pp_seed_sample_SD_e_mu_NC': [stdev(row[c] for row in differences) for c in range(3)],
                'contribution_to_all_recall_pp_mean_e_mu_NC': [mean(row[c] for row in contributions) for c in range(3)],
            }
        for seed_index in range(3):
            total = np.sum([domains[d]['delta_correct_by_seed_e_mu_NC'][seed_index] for d in DOMAINS[2:]], axis=0)
            if not np.array_equal(total, domains['all']['delta_correct_by_seed_e_mu_NC'][seed_index]):
                raise ValueError('Paired occupancy contributions do not sum to global difference')
            passed += 3
        contrasts.append({'a_family_task_kind': list(left), 'b_family_task_kind': list(right), 'domains': domains})
    result = {'status': 'complete', 'source': str(source/'evaluation/results.json'),
              'seeds': SEEDS, 'models': len(needed), 'contrasts': contrasts,
              'raw_array_checks_passed': passed, 'source_review_checks_passed': review['passed'],
              'uncertainty': 'post-hoc development point estimates and realized seed sample SD; no new event or seed-population interval',
              'compute_scope': 'each compute control uses its own family T4 operation budget; cross-family compute contrasts are not a common device-compute comparison',
              'domain_definition': 'original accepted pixel+bar+tracker channel count; four occupancy bins partition all; empty is an overlapping diagnostic',
              'Cal_A_Cut_B_Confirm_payload_read': False}
    return result


def plot(root, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    domains = DOMAINS[2:]
    labels = ['0-99', '100-999', '1000-4999', '5000+']
    for j, (contrast, label) in enumerate(zip(result['contrasts'][:3], ['T3 four passes', 'T4 conditional', 'T3 at own T4 wall'])):
        for seed_index, marker in enumerate(['o', 's', '^']):
            values = [contrast['domains'][d]['delta_recall_pp_by_seed_e_mu_NC'][seed_index][1] for d in domains]
            axes[0].plot(np.arange(4)+(j-1)*.14, values, marker, color=f'C{j}', alpha=.75,
                         label=f'{label}, seed {SEEDS[seed_index]}')
    axes[0].set(xticks=range(4), xticklabels=labels, xlabel='Original accepted channel count',
                ylabel='Muon recall difference (percentage points)', title='B1 minus B4: realized three-seed results')
    axes[0].legend(fontsize=7)
    row = result['contrasts'][3]['domains']['occupancy_0_100']
    for seed_index, marker in enumerate(['o', 's', '^']):
        axes[1].plot(np.arange(3)+(seed_index-1)*.1, row['delta_recall_pp_by_seed_e_mu_NC'][seed_index],
                     marker, label=f'Seed {SEEDS[seed_index]}')
    axes[1].set(xticks=range(3), xticklabels=['Electron', 'Muon', 'NC'], ylabel='Recall difference (percentage points)',
                title='B1 compute control minus main T3\nOriginal channel count below 100')
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.axhline(0., color='black', linewidth=.8, linestyle=':')
        axis.grid(axis='y', alpha=.2)
    figure.suptitle('Development Tune occupancy diagnostic; points are fitted seeds\n'
                   'Each compute control uses its own family T4 budget')
    (root/'figures').mkdir(exist_ok=True)
    for suffix in ['png', 'pdf']:
        figure.savefig(root/'figures'/f'r1_occupancy_seeds.{suffix}', dpi=170)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    result = readback(args.root)
    write_json(args.root/'audit/occupancy_seed_review.json', result)
    plot(args.root, result)
    print(json.dumps({key: result[key] for key in ['status', 'models', 'raw_array_checks_passed']}), flush=True)


if __name__ == '__main__':
    main()
