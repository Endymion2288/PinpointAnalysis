"""Prepare the accepted calibration/test partitions without resampling splits.

Metadata planning may list test keys. Test response conversion and inference
require the final analysis freeze; calibration preparation is allowed earlier.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import resource
import time

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import uproot

from analysis.flavour.create_data import BRANCHES, ROOT_TREES, lists
from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.readout import pixel_channels, bar_channels, tracker_measurements, modality_state
from analysis.flavour.s2.prepare import regions
from analysis.flavour.s5.truth import event_truth, group_flat


KEY = ['run', 'chunk', 'row']


def plan(args):
    args.root.mkdir(parents=True, exist_ok=True)
    source = json.loads((args.source/'plan.json').read_text())
    columns = KEY+['evtID', 'label', 'label_status', 'usable_join', 'split', 'geometry_id', 'family_hi', 'family_lo',
        'initE', 'initX', 'initY', 'initZ', 'initPDG', 'fslPDG', 'intType', 'scatteringType', 'tgtPDG',
        'n_pixel', 'n_bar', 'n_tracker', 'weight', 'xs', 'Q2', 'xBj', 'y', 'W']
    available = pq.read_schema(source['inventory']).names
    frame = pd.read_parquet(source['inventory'], columns=[c for c in columns if c in available])
    frame = frame[frame.split.isin(['calibration', 'test']) & frame.usable_join & frame.label_status.eq('consistent')].copy()
    frame = frame.sort_values(KEY).reset_index(drop=True)
    geometry = json.loads(Path(source['geometry']).read_text())
    frame['pixel_centered'], frame['large_scintillator'], definition = regions(frame, geometry)
    pairs = frame[['run', 'chunk']].drop_duplicates().reset_index(drop=True)
    pairs['shard'] = np.arange(len(pairs))//args.files_per_shard
    frame = frame.merge(pairs, on=['run', 'chunk'], validate='many_to_one', sort=False)
    frame.to_parquet(args.root/'manifest.parquet', index=False, compression=COMPRESSION)
    tasks = [{'split': split, 'shard': int(shard), 'events': len(group)}
             for (split, shard), group in frame.groupby(['split', 'shard'], sort=True)]
    result = {**source, 'stage': 'S6', 'source': str(args.source),
        'selection': 'all usable consistent events of the existing accepted calibration/test splits; no new split',
        'seed': 20261114, 'files_per_shard': args.files_per_shard, 'tasks': tasks,
        'counts': frame.groupby(['split', 'label']).size().rename('events').reset_index().to_dict('records'),
        'regions': definition, 'test_response_opened': False,
        'calibration_test_payloads_used': False,
        'split_role': {'validation': 'existing frozen 10000 events; candidate/threshold decisions',
                       'calibration': 'scalar temperature fit only', 'test': 'frozen final evaluation; no model/threshold tuning'}}
    write_json(args.root/'plan.json', result)
    for name in ['shards', 'truth_parts', 'logs', 'runtime', 'code', 'kubeflow', 'predictions', 'calibration', 'evaluation', 'audit', 'literature']:
        (args.root/name).mkdir(exist_ok=True)
    print(json.dumps({'events': len(frame), 'tasks': len(tasks), 'counts': result['counts']}), flush=True)


def require_freeze(root, split):
    path = root/'analysis_freeze.json'
    if split == 'test':
        data = json.loads(path.read_text())
        if data.get('status') != 'frozen' or not data.get('calibration_and_validation_rules_frozen'):
            raise ValueError('test requires the final model, calibration, threshold and selection freeze')


def convert(args):
    started = time.monotonic()
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    plan_data = json.loads((args.root/'plan.json').read_text())
    task = plan_data['tasks'][args.task]
    require_freeze(args.root, task['split'])
    frame = pd.read_parquet(args.root/'manifest.parquet', filters=[('split', '=', task['split']), ('shard', '=', task['shard'])])
    if args.limit:
        frame = frame.groupby('run', group_keys=False).head(args.limit)
    destination = args.root/'shards'/f"{task['split']}_{task['shard']:04d}"
    if args.limit:
        destination = args.root/'audit'/f"smoke_{args.task}"
    if (destination/'summary.json').exists():
        summary = json.loads((destination/'summary.json').read_text())
        if summary['events'] != len(frame):
            raise ValueError('existing S6 shard has different event coverage')
        print(json.dumps({'already_complete': str(destination)}), flush=True)
        return
    geometry = json.loads(Path(plan_data['geometry']).read_text())
    observations, provenance, truth, ideal_rows = [], [], [], []
    fields = {name: list(value) for name, value in BRANCHES.items()}
    fields['pixel'] += ['trackID', 'fromTau']
    fields['bar'] += ['trackID', 'pdg', 'fromTau']
    fields['tracker'] += ['trackID']
    with ThreadPoolExecutor(max_workers=max(1, args.cpus-1)) as executor:
        for (run, chunk), group in frame.groupby(['run', 'chunk'], sort=True):
            path = Path(plan_data['input_root'])/str(run)/f'{run}_{chunk:03d}.root'
            with uproot.open(path, decompression_executor=executor, interpretation_executor=executor,
                             object_cache=None, array_cache=None) as source:
                flat = {name: group_flat(source[name]) for name in ['primaries', 'tau', 'charm', 'trajectories']}
                for block, selected in group.groupby(group.row//32, sort=True):
                    start = int(block)*32
                    stop = min(start+32, source['event'].num_entries)
                    batch = {name: source[ROOT_TREES[name]].arrays(keys, entry_start=start, entry_stop=stop, library='ak')
                             for name, keys in fields.items()}
                    for meta in selected.to_dict('records'):
                        h = {name: {key: ak.to_numpy(value[key][int(meta['row'])-start]) for key in value.fields}
                             for name, value in batch.items()}
                        pixel = pixel_channels(h['pixel'], geometry)
                        bars, _ = bar_channels(h['bar'], geometry)
                        seed = int(meta['family_hi']) ^ int(meta['family_lo']) ^ 20261111
                        ideal, tracker = tracker_measurements(h['tracker'], seed)
                        state = modality_state(pixel, bars, tracker)
                        observations.append({'pixel': lists(pixel), 'bar': lists(bars), 'tracker_hypothesis': lists(tracker),
                            'has_response': state['has_response'].tolist(), 'outside_acceptance': [-1, -1, -1],
                            'fit_attempted': False, 'momentum_censored': -1})
                        ideal_rows.append(lists(ideal))
                        meta.update(pixel_channels=len(pixel['channel_id']), bar_channels=len(bars['channel_id']),
                                    tracker_channels=len(tracker['station']), all_empty=not state['has_response'].any(), tau_branch='unknown' if run == 10083 else 'not_tau')
                        event = int(meta['evtID'])
                        target = event_truth(meta, *[flat[n].get(event, {}) for n in ['primaries', 'tau', 'charm', 'trajectories']],
                                             h['pixel'], h['bar'], h['tracker'], geometry)
                        target.pop('pixel_truth_rows', None)
                        target.pop('tracker_truth_rows', None)
                        meta['tau_branch'] = target['tau_branch']
                        provenance.append(meta)
                        truth.append(target)
    destination.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(observations), destination/'observations.parquet', compression=COMPRESSION, row_group_size=64)
    pq.write_table(pa.Table.from_pylist([{'label': int(m['label'])} for m in provenance]), destination/'targets.parquet', compression=COMPRESSION)
    pd.DataFrame(provenance).to_parquet(destination/'provenance.parquet', index=False, compression=COMPRESSION)
    pd.DataFrame(truth).to_parquet(destination/'truth.parquet', index=False, compression=COMPRESSION)
    pq.write_table(pa.Table.from_pylist(ideal_rows), destination/'ideal_tracker.parquet', compression=COMPRESSION)
    summary = {'task': args.task, 'split': task['split'], 'events': len(provenance),
        'seconds': time.monotonic()-started, 'peak_rss_MiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'bytes': sum(p.stat().st_size for p in destination.glob('*.parquet')), 'test_response_used': task['split'] == 'test'}
    write_json(destination/'summary.json', summary)
    print(json.dumps(summary), flush=True)


def finalize(args):
    plan_data = json.loads((args.root/'plan.json').read_text())
    tasks = [t for t in plan_data['tasks'] if t['split'] == args.split]
    paths = [args.root/'shards'/f"{t['split']}_{t['shard']:04d}" for t in tasks]
    summaries = [json.loads((p/'summary.json').read_text()) for p in paths]
    catalog = pd.concat([pd.read_parquet(p/'provenance.parquet') for p in paths], ignore_index=True)
    expected = pd.read_parquet(args.root/'manifest.parquet', filters=[('split', '=', args.split)])
    if len(catalog) != len(expected) or not catalog.sort_values(KEY)[KEY].reset_index(drop=True).equals(expected.sort_values(KEY)[KEY].reset_index(drop=True)):
        raise ValueError('S6 converted keys differ from accepted split')
    catalog.to_parquet(args.root/f'{args.split}_catalog.parquet', index=False, compression=COMPRESSION)
    truth = pd.concat([pd.read_parquet(p/'truth.parquet') for p in paths], ignore_index=True)
    truth.to_parquet(args.root/f'{args.split}_truth.parquet', index=False, compression=COMPRESSION)
    write_json(args.root/f'{args.split}_preparation.json', {'status': 'complete', 'events': len(catalog), 'shards': len(paths),
        'task_seconds_sum': sum(s['seconds'] for s in summaries), 'bytes': sum(s['bytes'] for s in summaries),
        'maximum_task_RSS_MiB': max(s['peak_rss_MiB'] for s in summaries), 'split_unchanged': True})
    print(json.dumps({'split': args.split, 'events': len(catalog), 'status': 'complete'}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'convert', 'finalize'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--files-per-shard', type=int, default=32)
    parser.add_argument('--task', type=int, default=0)
    parser.add_argument('--cpus', type=int, default=4)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--split', choices=['calibration', 'test'], default='calibration')
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == '__main__':
    main()
