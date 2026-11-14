"""English, standalone R0 timing and memory figures from measured acceptance."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def draw(root):
    record = json.loads((root/'acceptance.json').read_text())
    output = root/'figures'
    output.mkdir(exist_ok=True)
    profile = record['profiles']['stream']
    rows = [row for row in profile['cases'] if row['training']]
    labels = [row['scenario'].replace('_', ' ') for row in rows]
    x = np.arange(len(rows))
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    axes[0].bar(x-.18, [10000/row['cold_seconds'] for row in rows], .36, label='Cold pass, includes startup')
    axes[0].bar(x+.18, [row['warm_events_per_second'] for row in rows], .36, label='Warm pass')
    axes[0].set(ylabel='Training events / second', title=f'10k Train events; {profile["GPU"]}')
    axes[0].legend(fontsize=8)
    axes[1].bar(x, [row['steady_data_wait_fraction']*100 for row in rows])
    axes[1].axhline(20, color='tab:red', linestyle='--', label='20% planning target')
    axes[1].set(ylabel='Steady blocking data wait (%)', title='Warm pass; first batch excluded')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(x, labels, rotation=35, ha='right')
        ax.grid(axis='y', alpha=.2)
    for extension in ['pdf', 'png']:
        figure.savefig(output/f'r0_loader_profile.{extension}', dpi=170)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    axes[0].bar(x-.18, [row['peak_memory']['aggregate_RSS_MiB']/1024 for row in rows], .36, label='Parent + workers RSS')
    axes[0].bar(x+.18, [row['peak_memory']['GPU_allocated_MiB']/1024 for row in rows], .36, label='Peak GPU allocation')
    axes[0].set(ylabel='Memory (GiB)', title='Peak memory; RSS may count shared mappings twice')
    axes[0].legend(fontsize=8)
    axes[0].set_xticks(x, labels, rotation=35, ha='right')
    latency = pd.DataFrame(profile['batch_occupancy_latency'])
    for scenario, group in latency.groupby('scenario', sort=True):
        axes[1].plot(group.mean_channels_per_event, group.compute_ms_per_event, 'o-', label=scenario.replace('_', ' '), alpha=.8)
    axes[1].set(xlabel='Mean native channels / event in batch quartile', ylabel='GPU compute time (ms / event)',
                title='Occupancy-dependent training compute')
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=.2)
    for extension in ['pdf', 'png']:
        figure.savefig(output/f'r0_memory_and_occupancy.{extension}', dpi=170)
    plt.close(figure)
    pd.DataFrame([{key: row[key] for key in ['scenario', 'workers', 'batch_size', 'cold_seconds', 'warm_seconds',
        'warm_events_per_second', 'cold_data_wait_fraction_including_startup',
        'warm_data_wait_fraction_including_startup', 'steady_data_wait_fraction',
        'channel_padding_utilization', 'event_block_padding_utilization']} for row in profile['cases']]).to_csv(
            output/'r0_stream_profile.csv', index=False)
    compiled = record['profiles']['stream_compiled']
    rows = [row for row in compiled['cases'] if row['training']]
    x = np.arange(len(rows))
    labels = [row['scenario'].replace('_', ' ') for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].bar(x, [row['warm_events_per_second'] for row in rows])
    axes[0].set(title=f'Compiled local encoder; {compiled["GPU"]}', ylabel='Warm training events / second')
    axes[1].bar(x, [row['steady_data_wait_fraction']*100 for row in rows])
    axes[1].axhline(20, color='tab:red', linestyle='--')
    axes[1].set(title='Warm pass, first batch excluded', ylabel='Steady blocking data wait (%)')
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.grid(axis='y', alpha=.2)
    for extension in ['pdf', 'png']:
        figure.savefig(output/f'r0_compiled_pipeline.{extension}', dpi=170)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    draw(parser.parse_args().root)


if __name__ == '__main__':
    main()
