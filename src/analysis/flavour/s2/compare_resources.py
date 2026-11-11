"""Pair complete-event resource measurements from two execution environments."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.flavour.inventory import COMPRESSION, write_json


def run(args):
    inputs = [args.reference, args.target]
    summaries = [json.loads((path/'summary.json').read_text()) for path in inputs]
    frames = [pd.read_parquet(path/'measurements.parquet') for path in inputs]
    group = ['architecture', 'pitch_mm', 'projection', 'mode']
    key = ['run', 'chunk', 'row'] + group
    for frame in frames:
        frame['cold_event'] = frame.groupby(group, sort=False).cumcount().eq(0)
    paired = frames[0].merge(frames[1], on=key, suffixes=('_reference', '_target'), validate='one_to_one')
    if len(paired) != len(frames[0]) or len(paired) != len(frames[1]):
        raise ValueError('resource profiles cover different physical event/representation/mode keys')
    for column in ['nodes', 'blocks', 'parameters', 'dim', 'cold_event']:
        if not paired[f'{column}_reference'].eq(paired[f'{column}_target']).all():
            raise ValueError(f'resource profiles have different {column}')
    for side in ['reference', 'target']:
        paired[f'compute_seconds_{side}'] = paired[f'forward_seconds_{side}'] + paired[f'backward_seconds_{side}']
    measured = paired.loc[~paired.cold_event_reference].copy()
    rows = []
    for identity, data in measured.groupby(group):
        rows.append({'architecture': str(identity[0]), 'pitch_mm': float(identity[1]),
            'projection': bool(identity[2]), 'mode': str(identity[3]), 'events': len(data),
            'reference_compute_median_ms': float(data.compute_seconds_reference.median()*1000),
            'target_compute_median_ms': float(data.compute_seconds_target.median()*1000),
            'reference_over_target_sum_ratio': float(data.compute_seconds_reference.sum()/data.compute_seconds_target.sum()),
            'paired_ratio_median': float((data.compute_seconds_reference/data.compute_seconds_target).median()),
            'reference_peak_gpu_mib': float(data.peak_gpu_mib_reference.max()),
            'target_peak_gpu_mib': float(data.peak_gpu_mib_target.max())})
    output = args.output or args.target/'comparison_reference'
    output.mkdir(parents=True, exist_ok=True)
    paired.to_parquet(output/'paired_measurements.parquet', index=False, compression=COMPRESSION)
    pd.DataFrame(rows).to_csv(output/'comparison.csv', index=False)
    labels = [f"{s['gpu']}, Torch {s['torch']}, {s['precision'].removeprefix('torch.')}" for s in summaries]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for axis, mode in zip(axes, ['inference', 'training']):
        data = measured[measured['mode'].eq(mode)]
        for architecture, color in [('part', 'tab:blue'), ('particlenet', 'tab:orange')]:
            subset = data[data.architecture.eq(architecture)]
            axis.scatter(subset.compute_seconds_reference*1000, subset.compute_seconds_target*1000,
                         s=14, alpha=.6, color=color, label='ParT' if architecture == 'part' else 'ParticleNet')
        limits = [min(data.compute_seconds_reference.min(), data.compute_seconds_target.min())*800,
                  max(data.compute_seconds_reference.max(), data.compute_seconds_target.max())*1200]
        x = np.asarray(limits)
        axis.plot(x, x, color='grey', linewidth=.8, label='Equal time')
        axis.plot(x, x/2, color='grey', linestyle=':', linewidth=.8, label='Target takes half the time')
        axis.set(xscale='log', yscale='log', xlim=limits, ylim=limits,
                 xlabel=f'Reference compute [ms]\n{labels[0]}',
                 ylabel=f'Target compute [ms]\n{labels[1]}',
                 title='Inference forward' if mode == 'inference' else 'Training forward + backward')
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle('Paired batch1 stress measurements: same events, nodes, blocks and parameters\n'
                 'First event per model/representation omitted; GPU and software both differ', fontsize=10)
    for suffix in ['png', 'pdf']:
        fig.savefig(output/f'paired_compute.{suffix}', dpi=160)
    plt.close(fig)
    write_json(output/'summary.json', {
        'reference': {'path': str(args.reference), **summaries[0]},
        'target': {'path': str(args.target), **summaries[1]},
        'matched_rows': len(paired), 'noncold_rows': len(measured),
        'all_losses_finite': bool(paired.loss_finite_reference.all() and paired.loss_finite_target.all()),
        'identical_fields': ['physical_event', 'representation', 'mode', 'nodes', 'blocks', 'parameters', 'dim'],
        'cold_event_rule': 'omit the first physical event in each model/representation/mode group on both sides',
        'scope': 'batch1 occupancy-rank stress sampling; compute excludes loading/collation; not population mean or full-epoch training time',
        'interpretation': 'joint execution-environment comparison; GPU, software and precision differ; no pure hardware attribution',
        'groups': rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
