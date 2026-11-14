"""Convert frozen Train100k/Tune40k, grouping ROOT reads within each partition."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.s2.prepare import convert, regions
from analysis.flavour.v2.partitions import FAMILY, KEY, equal_identity_rows, row_keys
from analysis.flavour.v2.r1.data import tracker_features, SUMMARY_NAMES
from analysis.flavour.s6.conditions import select_families


TRUTH = ['initE', 'initX', 'initY', 'initZ', 'initPDG', 'fslPDG',
         'intType', 'scatteringType', 'tgtPDG']


def assign_shards(frame, first, seed, target=2500):
    """Keep all selected rows of each ROOT file in one shard, mix classes."""
    count = math.ceil(len(frame)/target)
    rng = np.random.default_rng(seed)
    loads = np.zeros((count, 4), dtype='int64')
    selected = []
    for label, rows in frame.groupby('label', sort=True):
        chunks = list(rows.groupby(KEY[:2], sort=True))
        for position in rng.permutation(len(chunks)):
            _, group = chunks[position]
            # The per-class term keeps even sparse Tune shards mixed.
            score = loads[:, int(label)]*4 + loads.sum(1)
            shard = int(score.argmin())
            group = group.copy()
            group['shard'] = first+shard
            loads[shard, int(label)] += len(group)
            selected.append(group)
    return pd.concat(selected, ignore_index=True), first+count


def plan(root, r0):
    folder = root/'data'
    if (folder/'plan.json').exists():
        raise FileExistsError('frozen R1 preparation plan exists')
    frozen = json.loads((r0/'partitions.json').read_text())
    config = frozen['config']
    inventory = pd.read_parquet(config['inventory'], columns=KEY+FAMILY+TRUTH)
    lookup = pd.Index(row_keys(inventory))
    geometry = json.loads(Path(config['geometry']).read_text())
    parts, first = [], 0
    for name, split in [('train_100000', 'train'), ('tune', 'validation')]:
        selected = pd.read_parquet(frozen['manifests'][name]['manifest'])
        positions = lookup.get_indexer(row_keys(selected))
        if (positions < 0).any():
            raise ValueError('frozen selected rows absent from accepted inventory')
        actual = inventory.iloc[positions]
        if any(not np.array_equal(actual[c].to_numpy(), selected[c].to_numpy()) for c in FAMILY):
            raise ValueError('accepted/frozen family identity mismatch')
        for column in TRUTH:
            selected[column] = actual[column].to_numpy()
        selected['partition'] = name
        selected['split'] = split
        center, large, definition = regions(selected, geometry)
        selected['pixel_centered'], selected['large_scintillator'] = center, large
        selected, first = assign_shards(selected, first, 20261115+first)
        parts.append(selected)
    manifest = pd.concat(parts, ignore_index=True).sort_values(KEY).reset_index(drop=True)
    folder.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(folder/'manifest.parquet', index=False, compression=COMPRESSION)
    write_json(folder/'plan.json', {'stage': 'R1', 'seed': 42, 'geometry': config['geometry'],
        'input_root': config['input_root'], 'source_partitions': str(r0/'partitions.json'),
        'tasks': [[int(shard)] for shard in sorted(manifest.shard.unique())], 'regions': definition,
        'counts': manifest.groupby(['split', 'label']).size().rename('events').reset_index().to_dict('records'),
        'ROOT_files': manifest.groupby('split')[KEY[:2]].apply(lambda g: len(g.drop_duplicates())).to_dict(),
        'selection': 'complete frozen Train100k and Tune40k; whole selected ROOT chunks per output shard',
        'truth_usage': 'provenance/evaluation only; never model features',
        'Cal_A_Cut_B_Confirm_payload_read': False})
    print(json.dumps({'events': len(manifest), 'shards': first}), flush=True)


def convert_one(arguments):
    folder, task, threads = arguments
    plan_data = json.loads((folder/'plan.json').read_text())
    shard = plan_data['tasks'][task][0]
    output = folder/'shards'/f'{shard:03d}'
    if (output/'summary.json').exists() and all((output/name).is_file() for name in [
            'observations.parquet', 'targets.parquet', 'provenance.parquet', 'b0_features.parquet']):
        return
    convert(SimpleNamespace(root=folder, task=task, cpus=threads, output_root=None, smoke_per_class=0))


def observation_counts(response):
    result = {}
    for position, name in [(0, 'pixel_channels'), (1, 'bar_channels')]:
        result[name] = np.diff(response.column(position).combine_chunks().offsets.to_numpy())
    result['tracker_channels'] = np.diff(response['tracker_hypothesis'].combine_chunks().field('station').offsets.to_numpy())
    return result


def finalize(root):
    folder = root/'data'
    expected = pd.read_parquet(folder/'manifest.parquet')
    plan_data = json.loads((folder/'plan.json').read_text())
    frames, summaries, feature_frames = [], [], []
    for task in plan_data['tasks']:
        path = folder/'shards'/f'{task[0]:03d}'
        frame = pd.read_parquet(path/'provenance.parquet')
        response = pq.read_table(path/'observations.parquet', columns=[
            'pixel.channel_id', 'bar.channel_id', 'tracker_hypothesis'])
        # Dotted Arrow projection flattens both channel_id names; preserve the
        # positional detector distinction instead of resolving duplicate names.
        for name, counts in observation_counts(response).items():
            frame[name] = counts
        target = pq.read_table(path/'targets.parquet', columns=['label']).column('label').to_numpy()
        if not np.array_equal(target, frame.label):
            raise ValueError('target/provenance label mismatch')
        frame['all_empty'] = frame[['pixel_channels', 'bar_channels', 'tracker_channels']].sum(axis=1).eq(0)
        tracker = response['tracker_hypothesis'].combine_chunks()
        tracker_summary = []
        for i in range(len(frame)):
            fields = {key.name: np.asarray(tracker.field(key.name)[i].as_py()) for key in tracker.type}
            tracker_summary.append(tracker_features(fields)[1])
        b0 = pd.read_parquet(path/'b0_features.parquet')
        b0 = pd.DataFrame(np.column_stack([b0.to_numpy(), np.asarray(tracker_summary)]), columns=SUMMARY_NAMES)
        feature_frames.append(pd.concat([frame[KEY+['label', 'split']].reset_index(drop=True), b0], axis=1))
        frames.append(frame)
        summary = json.loads((path/'summary.json').read_text())
        if summary['PE_input'] != summary['PE_output'] or len(frame) != summary['events']:
            raise ValueError('converted response conservation/count check failed')
        summaries.append(summary)
    catalog = pd.concat(frames, ignore_index=True)
    if not equal_identity_rows(catalog, expected) or catalog.duplicated(KEY).any():
        raise ValueError('R1 conversion differs from frozen physical identities')
    catalog.to_parquet(folder/'catalog.parquet', index=False, compression=COMPRESSION)
    pd.concat(feature_frames, ignore_index=True).to_parquet(folder/'b0.parquet', index=False, compression=COMPRESSION)
    tune = catalog[catalog.split.eq('validation')]
    select_families(tune, 1000, 20261115).to_parquet(folder/'tune_screen.parquet', index=False, compression=COMPRESSION)
    select_families(tune, 400, 20261116).to_parquet(folder/'stress_rows.parquet', index=False, compression=COMPRESSION)
    write_json(root/'preparation.json', {'status': 'complete', 'events': len(catalog),
        'counts': catalog.groupby(['split', 'label']).size().rename('events').reset_index().to_dict('records'),
        'source_summaries': summaries, 'all_event_family_keys_exact': True, 'PE_conserved': True,
        'Cal_A_Cut_B_Confirm_payload_read': False, 'data_root': str(folder)})
    print(json.dumps({'status': 'complete', 'events': len(catalog)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'convert', 'finalize'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--r0', type=Path)
    parser.add_argument('--processes', type=int, default=4)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--task-start', type=int, default=0)
    parser.add_argument('--task-count', type=int)
    args = parser.parse_args()
    pa.set_cpu_count(args.threads)
    pa.set_io_thread_count(args.threads)
    if args.command == 'plan':
        plan(args.root, args.r0)
    elif args.command == 'convert':
        data = json.loads((args.root/'data/plan.json').read_text())
        stop = min(len(data['tasks']), args.task_start+(args.task_count or len(data['tasks'])))
        with ProcessPoolExecutor(args.processes, mp_context=multiprocessing.get_context('spawn')) as pool:
            list(pool.map(convert_one, [(args.root/'data', i, args.threads)
                                       for i in range(args.task_start, stop)]))
        if args.task_start == 0 and stop == len(data['tasks']):
            finalize(args.root)
    else:
        finalize(args.root)


if __name__ == '__main__':
    main()
