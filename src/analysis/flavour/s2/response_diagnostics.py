"""Measure coarse-cell multiplicity removal in the accepted validation input."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score

from analysis.flavour.inventory import write_json
from analysis.flavour.s2.representation import pixel_view


def measure_shard(arguments):
    path, geometry = arguments
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    meta = pd.read_parquet(path / 'provenance.parquet', columns=[
        'run', 'chunk', 'row', 'label', 'split', 'pixel_centered', 'large_scintillator'])
    selected = np.flatnonzero(meta.split.eq('validation'))
    pixel = pq.read_table(path / 'observations.parquet', columns=['pixel'],
                          use_threads=False)['pixel'].combine_chunks()
    ids = pixel.field('channel_id')
    offsets = ids.offsets.to_numpy()
    ids = ids.flatten().to_numpy()
    xyz = pixel.field('xyz_mm').flatten().flatten().to_numpy().reshape(-1, 3)
    records = []
    for index in selected:
        start, end = offsets[index:index + 2]
        event = {'channel_id': ids[start:end], 'xyz_mm': xyz[start:end]}
        count = int(end - start)
        identity = meta.iloc[index].drop('split').to_dict()
        for pitch, shift in [(.1, 0.), (.2, 0.), (1., 0.), (1., .5)]:
            view = pixel_view(event, geometry, pitch_mm=pitch, origin_shift=shift)
            multiplicity = view['count']
            cells = len(multiplicity)
            records.append({**identity, 'pitch_mm': pitch, 'origin_shift': shift,
                'native_pixels': count, 'occupied_cells': cells,
                'cells_with_multiplicity': int(np.count_nonzero(multiplicity > 1)),
                'pixels_in_merged_cells': int(multiplicity[multiplicity > 1].sum()),
                'maximum_cell_multiplicity': int(multiplicity.max()) if cells else 0,
                'excess_native_pixels': count - cells,
                'excess_fraction': (count - cells) / count if count else 0.})
    return records


def group_summary(frame):
    active = frame.native_pixels.gt(0)
    merged = frame.excess_native_pixels.gt(0)
    return {
        'events': len(frame), 'tau_events': int(frame.label.eq(2).sum()),
        'pixel_active_events': int(active.sum()),
        'events_with_merged_cells': int(merged.sum()),
        'fraction_merged_among_pixel_active': float(merged.sum() / active.sum()) if active.any() else None,
        'total_native_pixels': int(frame.native_pixels.sum()),
        'total_occupied_cells': int(frame.occupied_cells.sum()),
        'total_excess_native_pixels': int(frame.excess_native_pixels.sum()),
        'pooled_excess_fraction': float(frame.excess_native_pixels.sum() / frame.native_pixels.sum())
            if active.any() else None,
        'active_event_quantiles': frame.loc[active, ['native_pixels', 'occupied_cells',
            'excess_fraction', 'maximum_cell_multiplicity']].quantile([0, .25, .5, .75, .9, .99, 1]).to_dict(),
    }


def score_strata(root, measurements):
    """Exploratory groups; retain group prevalence and no interval claims."""
    rows = []
    keys = ['run', 'chunk', 'row', 'label']
    for family in ['part', 'pn']:
        paths = [root / 'experiments' / f'{family}_1mm_{kind}_s42_e4_v2'
                 for kind in ['joint', 'or']]
        if not all((p / 'manifest.json').exists() and
                   json.loads((p / 'manifest.json').read_text())['status'] == 'complete' for p in paths):
            continue
        count, binary = [pd.read_parquet(p / 'validation_predictions.parquet',
                         columns=keys + ['p2']) for p in paths]
        merged = measurements.merge(count, on=keys, validate='one_to_one').rename(columns={'p2': 'count_score'})
        merged = merged.merge(binary, on=keys, validate='one_to_one').rename(columns={'p2': 'or_score'})
        masks = {
            'pixel_empty': merged.native_pixels.eq(0),
            'pixel_active_no_merge': merged.native_pixels.gt(0) & merged.excess_native_pixels.eq(0),
            'pixel_active_excess_gt0_to_0.25': merged.excess_fraction.gt(0) & merged.excess_fraction.le(.25),
            'pixel_active_excess_gt0.25_to_0.75': merged.excess_fraction.gt(.25) & merged.excess_fraction.le(.75),
            'pixel_active_excess_gt0.75': merged.excess_fraction.gt(.75),
        }
        for name, mask in masks.items():
            selected = merged.loc[mask]
            y = selected.label.eq(2)
            ap = [float(average_precision_score(y, selected[c])) if y.any() and (~y).any() else None
                  for c in ['count_score', 'or_score']]
            rows.append({'family': family, 'stratum': name, 'events': len(selected),
                'tau_events': int(y.sum()), 'tau_prior': float(y.mean()) if len(y) else None,
                'count_tau_AP': ap[0], 'OR_tau_AP': ap[1],
                'count_minus_OR_AP': ap[0] - ap[1] if all(x is not None for x in ap) else None,
                'mean_absolute_tau_score_difference': float(
                    (selected.count_score - selected.or_score).abs().mean()) if len(y) else None})
    return rows


def run(args):
    started = time.perf_counter()
    plan = json.loads((args.root / 'plan.json').read_text())
    geometry = json.loads(Path(plan['geometry']).read_text())
    paths = sorted(p for p in (args.root / 'shards').iterdir() if (p / 'observations.parquet').exists())
    with ProcessPoolExecutor(max_workers=args.cpus) as pool:
        groups = pool.map(measure_shard, ((path, geometry) for path in paths))
        frame = pd.DataFrame([row for group in groups for row in group])
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output / 'coarse_multiplicity.parquet', index=False)
    summaries = []
    for (pitch, shift), selected in frame.groupby(['pitch_mm', 'origin_shift']):
        masks = {'all': np.ones(len(selected), dtype=bool),
                 'pixel_centered': selected.pixel_centered, 'large_scintillator': selected.large_scintillator}
        masks.update({f'class_{c}': selected.label.eq(c) for c in range(4)})
        for name, mask in masks.items():
            summaries.append({'pitch_mm': pitch, 'origin_shift': shift, 'stratum': name,
                              **group_summary(selected.loc[mask])})
    native_grid = frame.loc[frame.pitch_mm.eq(1) & frame.origin_shift.eq(0)]
    diagnostics = score_strata(args.root, native_grid)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for (pitch, shift), selected in frame.groupby(['pitch_mm', 'origin_shift']):
        active = selected.loc[selected.native_pixels.gt(0), 'excess_fraction'].sort_values().to_numpy()
        label = f'{pitch:g} mm' + ('; half-cell shift' if shift else '')
        axes[0].plot(active, np.arange(1, len(active) + 1) / len(active), label=label)
    axes[0].set(xlabel='(Native pixels - occupied coarse cells) / native pixels',
                ylabel='Fraction of pixel-active validation events', title='Multiplicity removed by coarse OR',
                xlim=(0, 1), ylim=(0, 1))
    axes[0].legend(fontsize=8)
    for c, label in enumerate(['CC nu_e', 'CC nu_mu', 'CC nu_tau', 'NC']):
        values = native_grid.loc[native_grid.label.eq(c) & native_grid.native_pixels.gt(0), 'excess_fraction']
        axes[1].hist(values, bins=np.linspace(0, 1, 26), density=True, histtype='step', label=label)
    axes[1].set(xlabel='Excess native-pixel fraction', ylabel='Density in pixel-active class',
                title='1 mm reference origin; class-conditional input')
    axes[1].legend(fontsize=8)
    fig.suptitle('S2 accepted validation input; response diagnostic, no calibration or test')
    for suffix in ['png', 'pdf']:
        fig.savefig(args.output / f'coarse_multiplicity.{suffix}', dpi=160)
    plt.close(fig)
    write_json(args.output / 'summary.json', {
        'utc': datetime.now(timezone.utc).isoformat(),
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'root': str(args.root), 'workers': args.cpus, 'seconds': time.perf_counter() - started,
        'scope': 'post-result exploratory response/score diagnostic on accepted validation; no new fit or sample certification',
        'definition': 'excess_fraction=(native_active_pixel_channels-occupied_coarse_cells)/native_active_pixel_channels; zero for pixel-empty events',
        'weighting': 'event and pooled-channel summaries separate; score strata retain their observed tau priors',
        'intervals_measured': False, 'calibration_test_used': False,
        'representation_summaries': summaries, 'exploratory_count_OR_score_strata': diagnostics})
    print(json.dumps({'rows': len(frame), 'events': len(native_grid), 'output': str(args.output)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpus', type=int, default=2)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
