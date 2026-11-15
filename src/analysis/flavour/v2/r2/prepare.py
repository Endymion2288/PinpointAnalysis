"""Reuse R1 observations and convert only the additional frozen Train events."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.s2.prepare import regions
from analysis.flavour.v2.partitions import FAMILY, KEY, equal_identity_rows, row_keys
from analysis.flavour.v2.r1.prepare import TRUTH, assign_shards, convert_one, observation_counts


def plan(root, r1, size=300000):
    folder = root/'data'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'plan.json').exists():
        raise FileExistsError('R2 conversion plan already frozen')
    r0 = Path(json.loads((r1/'campaign.json').read_text())['r0'])
    partitions = json.loads((r0/'partitions.json').read_text())
    original = pd.read_parquet(r1/'data/manifest.parquet')
    selected = pd.read_parquet(partitions['manifests'][f'train_{size}']['manifest'])
    smaller = original[original.split.eq('train')]
    if not set(row_keys(smaller)).issubset(set(row_keys(selected))):
        raise ValueError('Train100k is not nested in the requested pool')
    additional = selected.loc[~pd.Index(row_keys(selected)).isin(row_keys(smaller))].copy()
    inventory = pd.read_parquet(partitions['config']['inventory'], columns=KEY+FAMILY+TRUTH)
    positions = pd.Index(row_keys(inventory)).get_indexer(row_keys(additional))
    actual = inventory.iloc[positions]
    if (positions<0).any() or any(not np.array_equal(actual[c], additional[c]) for c in FAMILY):
        raise ValueError('additional frozen physical identities differ from accepted inventory')
    for name in TRUTH:
        additional[name] = actual[name].to_numpy()
    additional['partition'] = f'train_{size}'
    additional['split'] = 'train'
    geometry = json.loads(Path(partitions['config']['geometry']).read_text())
    center, large, definition = regions(additional, geometry)
    additional['pixel_centered'], additional['large_scintillator'] = center, large
    first = int(original.shard.max())+1
    additional, stop = assign_shards(additional, first, 20261115+first)
    complete = pd.concat([original, additional], ignore_index=True)
    if complete.duplicated(KEY).any() or not equal_identity_rows(complete[complete.split.eq('train')], selected):
        raise ValueError('planned expanded data does not equal the frozen pool')
    (folder/'shards').mkdir(exist_ok=True)
    for path in sorted((r1/'data/shards').iterdir()):
        (folder/'shards'/path.name).symlink_to(path.resolve(), target_is_directory=True)
    (root/'data_100000').symlink_to((r1/'data').resolve(), target_is_directory=True)
    complete.sort_values(KEY).to_parquet(folder/'manifest.parquet', index=False, compression=COMPRESSION)
    write_json(folder/'plan.json', {'stage': 'R2', 'seed': 42,
        'geometry': partitions['config']['geometry'], 'input_root': partitions['config']['input_root'],
        'source_partitions': str(r0/'partitions.json'), 'source_R1': str(r1), 'pool_events': size,
        'reused_shards': sorted(original.shard.unique().tolist()),
        'tasks': [[i] for i in range(first, stop)], 'regions': definition,
        'counts': complete.groupby(['split','label']).size().rename('events').reset_index().to_dict('records'),
        'new_events': len(additional), 'selection': 'reuse exact R1 Train100k/Tune responses; only convert additional frozen Train rows',
        'Cal_A_Cut_B_Confirm_payload_read': False})
    print(json.dumps({'status':'planned', 'new_events':len(additional), 'new_shards':stop-first}), flush=True)


def finalize(root):
    started = time.perf_counter()
    folder = root/'data'
    spec = json.loads((folder/'plan.json').read_text())
    expected = pd.read_parquet(folder/'manifest.parquet')
    reused = set(spec['reused_shards'])
    frames, summaries = [], []
    r1 = Path(spec['source_R1'])
    original = pd.read_parquet(r1/'data/catalog.parquet')
    for shard in sorted(expected.shard.unique()):
        path = folder/'shards'/f'{shard:03d}'
        if int(shard) in reused:
            frames.append(original[original.shard.eq(shard)].copy())
            continue
        frame = pd.read_parquet(path/'provenance.parquet')
        response = pq.read_table(path/'observations.parquet', columns=[
            'pixel.channel_id', 'bar.channel_id', 'tracker_hypothesis'])
        for name, values in observation_counts(response).items():
            frame[name] = values
        target = pq.read_table(path/'targets.parquet', columns=['label']).column(0).to_numpy()
        summary = json.loads((path/'summary.json').read_text())
        if not np.array_equal(target, frame.label) or summary['events']!=len(frame):
            raise ValueError(f'target/count mismatch in shard {shard}')
        if summary['PE_input'] != summary['PE_output']:
            raise ValueError(f'PE conservation failed in shard {shard}')
        frame['all_empty'] = frame[['pixel_channels','bar_channels','tracker_channels']].sum(axis=1).eq(0)
        frames.append(frame)
        summaries.append(summary)
    catalog = pd.concat(frames, ignore_index=True)
    if not equal_identity_rows(catalog, expected) or catalog.duplicated(KEY).any():
        raise ValueError('expanded catalog differs from frozen physical identities')
    catalog.to_parquet(folder/'catalog.parquet', index=False, compression=COMPRESSION)
    for name in ['tune_screen.parquet', 'stress_rows.parquet']:
        destination = folder/name
        if not destination.exists():
            destination.symlink_to((r1/'data'/name).resolve())
    write_json(root/'preparation.json', {'status':'complete', 'events':len(catalog),
        'pool_events': spec['pool_events'], 'new_events':spec['new_events'],
        'counts':spec['counts'], 'reused_response_shards':len(reused),
        'new_response_shards':len(summaries), 'new_source_summaries':summaries,
        'all_event_family_keys_exact':True, 'PE_conserved':True,
        'seconds':time.perf_counter()-started, 'Cal_A_Cut_B_Confirm_payload_read':False})
    print(json.dumps({'status':'complete','events':len(catalog)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan','convert','finalize'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--r1', type=Path)
    parser.add_argument('--size', type=int, default=300000)
    parser.add_argument('--task-start', type=int, default=0)
    parser.add_argument('--task-count', type=int, default=10)
    parser.add_argument('--processes', type=int, default=4)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    pa.set_cpu_count(args.threads)
    pa.set_io_thread_count(args.threads)
    if args.command=='plan':
        plan(args.root, args.r1, args.size)
    elif args.command=='finalize':
        finalize(args.root)
    else:
        spec = json.loads((args.root/'data/plan.json').read_text())
        stop = min(len(spec['tasks']), args.task_start+args.task_count)
        with ProcessPoolExecutor(args.processes, mp_context=multiprocessing.get_context('spawn')) as pool:
            list(pool.map(convert_one, [(args.root/'data', i, args.threads)
                                       for i in range(args.task_start, stop)]))


if __name__=='__main__':
    main()
