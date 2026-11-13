"""Extend the accepted corpus and write separate per-bar simulation targets."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import resource
import time
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import uproot

from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.s2.prepare import convert, regions


def plan(args):
    root, base = args.root, args.base
    previous = json.loads((base / 'plan.json').read_text())
    old = pd.read_parquet(base / 'manifest.parquet')
    inventory = pd.read_parquet(previous['inventory'], columns=[c for c in old if c not in {
        'pixel_centered', 'large_scintillator', 'shard'}])
    rng = np.random.default_rng(20261113)
    chunks, selected = {}, []
    for run in [10081, 10082, 10083, 10084]:
        pool = inventory[inventory.run.eq(run) & inventory.split.eq('train')
            & inventory.usable_join & inventory.label_status.eq('consistent')
            & ~inventory.chunk.isin(previous['chunks'][str(run)])]
        available = np.sort(pool.chunk.unique())
        chosen = sorted(map(int, rng.choice(available, 8, replace=False)))
        pool = pool[pool.chunk.isin(chosen)]
        picked = pool.iloc[rng.permutation(len(pool))[:5000]].copy()
        if len(picked) != 5000:
            raise ValueError('extension has fewer than 5000 train rows per class')
        picked['shard'] = picked.chunk.map({chunk: i for i, chunk in enumerate(chosen)})
        chunks[str(run)] = chosen
        selected.append(picked)
    frame = pd.concat(selected).sort_values(['run', 'chunk', 'row']).reset_index(drop=True)
    geometry = json.loads(Path(previous['geometry']).read_text())
    frame['pixel_centered'], frame['large_scintillator'], _ = regions(frame, geometry)
    extension = root / 'extension'
    extension.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(extension / 'manifest.parquet', index=False, compression=COMPRESSION)
    write_json(extension / 'plan.json', {**previous, 'seed': 20261113, 'chunks': chunks,
        'tasks': [list(range(i, i + 2)) for i in range(0, 8, 2)],
        'max_train_per_class': 5000, 'max_validation_per_class': 0,
        'selection': 'additional_train_only_rows_from_eight_disjoint_chunks_per_class',
        'counts': frame.groupby(['split', 'run']).size().rename('events').reset_index().to_dict('records')})
    (root / 'shards').mkdir(exist_ok=True)
    for i in range(40):
        source = (base / 'shards' / f'{i:03d}' if i < 32 else extension / 'shards' / f'{i-32:03d}')
        path = root / 'shards' / f'{i:03d}'
        if not path.is_symlink():
            path.symlink_to(source.resolve(), target_is_directory=True)
    added = frame.copy()
    added['shard'] += 32
    combined = pd.concat([old, added], ignore_index=True)
    combined.to_parquet(root / 'manifest.parquet', index=False, compression=COMPRESSION)
    write_json(root / 'plan.json', {**previous, 'base': str(base), 'extension': str(extension),
        'stage': 'S4', 'train_events': 100000, 'validation_events': 10000,
        'counts': combined.groupby(['split', 'run']).size().rename('events').reset_index().to_dict('records'),
        'selection': 'unchanged_S2_80k_train_10k_validation_plus_20k_train_only',
        'rel_definition': 'bar_per_track_edep_fraction_EM_muon_hadron_other; no_pixel_soft_truth',
        'calibration_test_payloads_used': False})
    print(json.dumps({'root': str(root), 'train': 100000, 'validation': 10000, 'extension_tasks': 4}))


def bar_contributions(record, geometry):
    """Energy-weighted mixtures before hardware channel aggregation (MeV)."""
    layer, panel, horizontal, row, col, pdg, energy = [np.asarray(record[k]) for k in
        ['layerID', 'panelID', 'isHorizontal', 'rowID', 'colID', 'pdg', 'edep']]
    nbar, npanel = len(geometry['scint_bar_Xcenter']), len(geometry['scint_Zpos'])
    ids = (layer * npanel + panel) * (2 * nbar) + horizontal * nbar + np.where(horizontal == 1, row, col)
    unique, inverse = np.unique(ids, return_inverse=True)
    category = np.full(len(pdg), 2, dtype='int64')
    category[np.isin(np.abs(pdg), [11, 22])] = 0
    category[np.abs(pdg) == 13] = 1
    category[(np.abs(pdg) <= 100) & ~np.isin(np.abs(pdg), [11, 13, 22])] = 3
    sums = np.zeros((len(unique), 4), dtype='float64')
    np.add.at(sums, (inverse, category), energy)
    totals = sums.sum(-1)
    valid = (totals > 0) & np.isfinite(sums).all(-1) & (sums >= 0).all(-1)
    fractions = sums / np.maximum(totals[:, None], 1e-30)
    return unique, totals, fractions, valid


def rel_shard(args, shard):
    started = time.perf_counter()
    source = args.root / 'shards' / f'{shard:03d}'
    metadata = pd.read_parquet(source / 'provenance.parquet')
    # Targets are generated only for train, and never stored inside observations.
    selected = metadata[metadata.split.eq('train')].copy()
    selected['source_row'] = selected.index
    plan_data = json.loads((args.root / 'plan.json').read_text())
    geometry = json.loads(Path(plan_data['geometry']).read_text())
    columns = ['layerID', 'panelID', 'isHorizontal', 'rowID', 'colID', 'pdg', 'edep']
    records, channels, mixtures, unknown_energy, total_energy = [], 0, 0, 0., 0.
    with ThreadPoolExecutor(args.cpus) as executor:
        for (run, chunk), group in selected.groupby(['run', 'chunk'], sort=True):
            path = Path(plan_data['input_root']) / str(run) / f'{run}_{int(chunk):03d}.root'
            with uproot.open(path, object_cache=None, array_cache=None,
                decompression_executor=executor, interpretation_executor=executor) as file:
                tree = file['Hits/scintHits']
                for block, rows in group.groupby(group.row // 64, sort=True):
                    start = int(block) * 64
                    arrays = tree.arrays(columns, entry_start=start, entry_stop=start+64, library='ak')
                    for event in rows.itertuples(index=False):
                        ids, totals, fractions, valid = bar_contributions(arrays[int(event.row)-start], geometry)
                        records.append({'source_row': int(event.source_row), 'run': int(run), 'chunk': int(chunk),
                            'row': int(event.row), 'channel_id': ids.astype('uint64').tolist(),
                            'energy_MeV': totals.astype('float32').tolist(),
                            'fractions': fractions.astype('float32').tolist(), 'valid': valid.tolist()})
                        channels += len(ids)
                        mixtures += int((fractions.max(-1) < .999).sum())
                        unknown_energy += float((totals * fractions[:, 3]).sum())
                        total_energy += float(totals.sum())
    output = args.root / 'rel'
    output.mkdir(exist_ok=True)
    pq.write_table(pa.Table.from_pylist(sorted(records, key=lambda x: x['source_row'])),
        output / f'{shard:03d}.parquet', compression=COMPRESSION)
    write_json(output / f'{shard:03d}.json', {'shard': shard, 'events': len(records), 'channels': channels,
        'mixed_channels': mixtures, 'unknown_energy_MeV': unknown_energy, 'energy_MeV': total_energy,
        'seconds': time.perf_counter()-started, 'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'source_semantics': 'ScintSD.cc barEnergyMap keyed by channel,track,PDG; exact accumulated deposits'})
    print(json.dumps({'rel_shard': shard, 'events': len(records), 'channels': channels}), flush=True)


def create(args):
    if args.task < 4:
        convert(SimpleNamespace(root=args.root/'extension', output_root=None,
            task=args.task, cpus=args.cpus, smoke_per_class=0))
    for shard in range(args.task*2, min(args.task*2+2, 40)):
        rel_shard(args, shard)
    # The four extension-producing tasks also own their extension sidecars.
    if args.task < 4:
        for shard in range(32+args.task*2, 34+args.task*2):
            rel_shard(args, shard)


def finalize(args):
    plan_data = json.loads((args.root / 'plan.json').read_text())
    catalog = pd.read_parquet(Path(plan_data['base']) / 'catalog.parquet')
    frames, audit = [catalog], []
    for i, path in enumerate(sorted((args.root / 'shards').iterdir())):
        meta = pd.read_parquet(path / 'provenance.parquet')
        obs = pq.read_table(path / 'observations.parquet', columns=['pixel', 'bar', 'tracker_hypothesis'])
        if i >= 32:
            for name, field, column in [('pixel', 'channel_id', 'pixel_channels'),
                ('bar', 'channel_id', 'bar_channels'), ('tracker_hypothesis', 'station', 'tracker_channels')]:
                meta[column] = np.diff(obs[name].combine_chunks().field(field).offsets.to_numpy())
            meta['shard'] = i
            frames.append(meta)
        sidecar = pq.read_table(args.root / 'rel' / f'{i:03d}.parquet').to_pylist()
        bars = obs['bar'].combine_chunks()
        max_error, matched = 0., 0
        for row in sidecar:
            j = row['source_row']
            value = bars[j].as_py()
            ids = np.asarray(value['channel_id'], dtype='uint64')
            actual = np.asarray(value['edep_proxy_MeV'], dtype='float64')
            expected = np.asarray(row['energy_MeV'])
            if not np.array_equal(ids, row['channel_id']):
                raise ValueError('Rel/hardware bar channel mismatch')
            difference = np.abs(actual-expected)/np.maximum(np.abs(actual), 1e-6)
            max_error = max(max_error, float(difference.max(initial=0)))
            if not np.isfinite(np.asarray(row['fractions'])).all() or not all(row['valid']):
                raise ValueError('invalid bar contribution targets')
            matched += len(ids)
        report = json.loads((args.root/'rel'/f'{i:03d}.json').read_text())
        audit.append({**report, 'matched_channels': matched, 'maximum_relative_energy_error': max_error})
    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(args.root / 'catalog.parquet', index=False, compression=COMPRESSION)
    checks = {'100k_train': int(combined.split.eq('train').sum()) == 100000,
        '10k_unchanged_validation': int(combined.split.eq('validation').sum()) == 10000,
        'all_bar_targets_match_hardware_energy': max(x['maximum_relative_energy_error'] for x in audit) < 2e-6,
        'all_bar_targets_train_only': sum(x['events'] for x in audit) == 100000}
    write_json(args.root / 'data_summary.json', {'checks': checks, 'passed': all(checks.values()),
        'counts': combined.groupby(['split','label']).size().rename('events').reset_index().to_dict('records'),
        'rel': audit, 'calibration_test_used': False})
    print(json.dumps(checks))
    if not all(checks.values()):
        raise RuntimeError('S4 data assembly or new Rel target check failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'create', 'finalize'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--base', type=Path)
    parser.add_argument('--task', type=int)
    parser.add_argument('--cpus', type=int, default=4)
    args = parser.parse_args()
    pa.set_cpu_count(args.cpus); pa.set_io_thread_count(args.cpus)
    globals()[args.command](args)


if __name__ == '__main__':
    main()
