"""S1 stratified displays, signal/occupancy diagnostics and loader resource audit."""

import argparse
import json
from pathlib import Path
import resource
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

from analysis.flavour import RUN_LABELS
from analysis.flavour.datasets import NativeShard, collate_native
from analysis.flavour.inventory import write_json


def display(event, title, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    pixel = np.asarray(event['pixel']['xyz_mm']).reshape(-1, 3)
    bars = event['bar']
    bounds = np.asarray(bars['bounds_mm']).reshape(-1, 5)
    horizontal = np.asarray(bars['horizontal']) == 1
    signal = np.log1p(bars['pe_proxy'])
    for ax, coordinate in zip(axes[:2], [0, 1]):
        ax.scatter(pixel[:, 2], pixel[:, coordinate], s=2, c='black', label='Native binary pixels')
        selected = ~horizontal if coordinate == 0 else horizontal
        center = (bounds[:, coordinate * 2] + bounds[:, coordinate * 2 + 1]) / 2
        shown = ax.scatter(bounds[selected, 4], center[selected], c=signal[selected], s=9,
                           cmap='viridis', label='Measured bar coordinate')
        ax.set(xlabel='z [mm]', ylabel=('x' if coordinate == 0 else 'y') + ' [mm]')
    axes[0].legend(fontsize=7)
    fig.colorbar(shown, ax=axes[:2], label='log(1 + simulated PE proxy)', shrink=.7)
    tracker = event['tracker_hypothesis']
    xy = np.asarray(tracker['xy_mm']).reshape(-1, 2)
    axes[2].scatter(xy[:, 0], xy[:, 1], c=tracker['station'], cmap='tab10', s=15)
    axes[2].set(xlabel='Tracker x [mm]', ylabel='Tracker y [mm]', title='Unvalidated digitization hypothesis')
    fig.suptitle(title)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def evaluate(args):
    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    sizes, cutflow, load, stats, selected = {}, {}, {}, [], []
    channel_maps = {}
    for shard in sorted(args.input.iterdir()):
        if not (shard / 'observations.parquet').exists():
            continue
        run = int(shard.name)
        start = time.perf_counter()
        dataset = NativeShard(shard)
        load_seconds = time.perf_counter() - start
        target = pq.read_table(shard / 'targets.parquet').to_pandas()
        prov = pq.read_table(shard / 'provenance.parquet').to_pandas()
        summary = json.loads((shard / 'summary.json').read_text())
        rows = dataset.observations.to_pylist()
        geometry = json.loads((shard / 'geometry.json').read_text())
        associations = sorted({(int(l), int(p), int(h)) for e in rows
                               for l, p, h in zip(e['bar']['layer'], e['bar']['panel'], e['bar']['horizontal'])})
        channel_maps[str(run)] = {
            'geometry': str(shard / 'geometry.json'), 'positions_unit': 'mm',
            'pixel_channel_id': '((plane * n_y) + row) * n_x + col',
            'pixel_dimensions': [len(geometry[k]) for k in ['pixel_Zpos', 'pixel_Ypos', 'pixel_Xpos']],
            'bar_channel_id': '(layer * n_panels + panel) * (2 * n_bars) + horizontal * n_bars + bar',
            'bar_dimensions': {'n_panels': len(geometry['scint_Zpos']), 'n_bars': len(geometry['scint_bar_Xcenter'])},
            'observed_layer_panel_orientation': [list(a) for a in associations],
            'observed_panels': len({a[1] for a in associations}),
            'panel_association_status': 'empirical_from_pilot_not_production_geometry_validation',
            'tracker_station_z_map': None}
        counts = []
        for i, event in enumerate(rows):
            n_pixel, n_bar = len(event['pixel']['channel_id']), len(event['bar']['channel_id'])
            counts.append(n_pixel)
            stats.append({'run': run, 'pixel': n_pixel, 'bar': n_bar,
                          'tracker': len(event['tracker_hypothesis']['station']),
                          'edep_MeV': sum(event['bar']['edep_proxy_MeV']),
                          'pe': sum(event['bar']['pe_proxy']), 'all_empty': not any(event['has_response'])})
        order = np.argsort(counts, kind='stable')
        chosen = [int(order[j]) for j in [0, len(order)//4, len(order)//2, int(.95 * (len(order)-1))]]
        for category, index in zip(['lowest', 'q25', 'median', 'q95'], chosen):
            selected.append({'run': run, 'row': index, 'evtID': int(prov.iloc[index].evtID), 'category': category})
            display(rows[index], f'{RUN_LABELS[run]} | row {index} | pixel occupancy {category}',
                    args.output / f'event_{run}_{category}.png')
        t = time.perf_counter()
        n, tokens = 0, 0
        for batch in DataLoader(dataset, batch_size=16, collate_fn=collate_native, num_workers=0):
            n += len(batch['targets']['label'])
            tokens += sum(len(batch['observations'][d]['values']) for d in ['pixel', 'bar', 'tracker_hypothesis'])
        collate_seconds = time.perf_counter() - t
        load[str(run)] = {'read_seconds': load_seconds, 'collate_seconds': collate_seconds,
                          'events_per_second_collate': n / collate_seconds, 'tokens': tokens}
        cutflow[str(run)] = {'all_events': len(rows), 'label_consistent': int((target.label_status == 'consistent').sum()),
                             'join_usable': int(prov.usable_join.sum()), 'zero_pixel': int(np.count_nonzero(np.asarray(counts) == 0)),
                             'all_empty_digitized': sum(not any(e['has_response']) for e in rows),
                             'threshold_cut': 'none', 'fiducial_cut': 'none', 'minimum_nodes_cut': 'none'}
        sizes[str(run)] = {'events': len(rows), 'observation_bytes': summary['parquet_bytes']['observations.parquet'],
                           'bytes_per_event': summary['parquet_bytes']['observations.parquet'] / len(rows)}
    frame = pd.DataFrame(stats)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for run, group in frame.groupby('run'):
        for ax, field in zip(axes, ['pixel', 'bar', 'pe']):
            ax.hist(np.log10(1 + group[field]), bins=50, histtype='step', density=True, label=RUN_LABELS[run])
            ax.set(xlabel=f'log10(1 + {field})', ylabel='Density')
    axes[0].legend()
    fig.savefig(args.output / 'occupancy_signal.png', dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    labels = ['Generated', 'Metadata-consistent', 'Any digitized response']
    x = np.arange(len(labels))
    for i, (run, values) in enumerate(cutflow.items()):
        ax.bar(x + (i-1.5)*.18, [values['all_events'], values['label_consistent'],
                                values['all_events']-values['all_empty_digitized']], width=.18,
               label=RUN_LABELS[int(run)])
    ax.set(xticks=x, xticklabels=labels, ylabel='Events', title='Pilot event counts; no response requirement in training')
    ax.legend()
    fig.savefig(args.output / 'cutflow.png', dpi=140)
    plt.close(fig)
    quantiles = {str(run): group[['pixel', 'bar', 'tracker', 'edep_MeV', 'pe']].quantile([0, .5, .9, .95, .99, .999, 1]).to_dict()
                 for run, group in frame.groupby('run')}
    write_json(args.output / 'channel_maps.json', channel_maps)
    write_json(args.output / 'summary.json', {'cutflow': cutflow, 'loader': load, 'sizes': sizes,
               'quantiles': quantiles, 'selected_events': selected,
               'selection': 'four deterministic pixel-occupancy strata per class; no truth selection',
               'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
               'torch_version': torch.__version__, 'shard_target_mib': [256, 1024],
               'pilot_shards_smaller_than_target': True,
               'full_scale_size_estimate_status': 'calculated_separately_from_inventory_counts_and_pilot_sizes'})
    print(json.dumps({'events': len(frame), 'displays': len(selected), 'output': str(args.output)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    evaluate(parser.parse_args())


if __name__ == '__main__':
    main()
