"""Batch truth extraction and compact object/label sidecars for accepted S4 events."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import uproot

from analysis.flavour.s5.truth import event_truth, group_flat


def initialize(root, source):
    root.mkdir(parents=True, exist_ok=True)
    for name in ['truth_parts', 'object_parts', 'logs', 'code', 'runtime', 'experiments', 'evaluation', 'figures', 'kubeflow']:
        (root/name).mkdir(exist_ok=True)
    for name in ['plan.json', 'catalog.parquet', 'manifest.parquet', 'shards', 'audit']:
        path = root/name
        if not path.exists():
            path.symlink_to(source/name, target_is_directory=(source/name).is_dir())
    frame = pd.read_parquet(root/'catalog.parquet')
    groups = [{'run': int(r), 'chunk': int(c), 'events': len(g)} for (r, c), g in frame.groupby(['run', 'chunk'], sort=True)]
    plan = {'source': str(source), 'tasks': groups, 'events': len(frame), 'truth_input_only_for_supervision_and_evaluation': True,
            'calibration_test_used': False, 'units': {'position': 'mm', 'momentum': 'GeV', 'energy': 'GeV'}}
    (root/'s5_plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    print(json.dumps({'truth_tasks': len(groups), 'events': len(frame)}))


def create(root, task, limit=None):
    started = time.monotonic()
    plan = json.loads((root/'s5_plan.json').read_text())
    item = plan['tasks'][task]
    frame = pd.read_parquet(root/'catalog.parquet', filters=[('run', '=', item['run']), ('chunk', '=', item['chunk'])]).sort_values('row')
    if limit:
        frame = frame.iloc[:limit]
    source_plan = json.loads((root/'plan.json').read_text())
    geometry = json.loads(Path(source_plan['geometry']).read_text())
    path = Path(source_plan['input_root'])/str(item['run'])/f"{item['run']}_{item['chunk']:03d}.root"
    output = []
    with ThreadPoolExecutor(max_workers=2) as executor, uproot.open(path, decompression_executor=executor) as file:
        flat = {name: group_flat(file[name]) for name in ['primaries', 'tau', 'charm', 'trajectories']}
        fields = {'pixel': ('Hits/pixelHits', ['layerID', 'rowID', 'colID', 'trackID', 'fromTau']),
                  'bar': ('Hits/scintHits', ['trackID', 'pdg', 'edep', 'fromTau']),
                  'tracker': ('Hits/faserHits', ['trackerID', 'trackID', 'x', 'y', 'z'])}
        # One bounded source window at a time; no per-event remote requests.
        for start in range(0, 1000, 64):
            selected = frame[(frame.row >= start) & (frame.row < start+64)]
            if selected.empty:
                continue
            hits = {key: file[name].arrays(columns, entry_start=start, entry_stop=start+64, library='ak') for key, (name, columns) in fields.items()}
            for meta in selected.to_dict('records'):
                event = int(meta['evtID'])
                h = {key: {k: ak.to_numpy(value[k][int(meta['row'])-start]) for k in value.fields} for key, value in hits.items()}
                output.append(event_truth(meta, *[flat[n].get(event, {}) for n in ['primaries', 'tau', 'charm', 'trajectories']], h['pixel'], h['bar'], h['tracker'], geometry))
    destination = root/'truth_parts'/f'{task:03d}.parquet' if not limit else root/f'truth_smoke_{task:03d}.parquet'
    table = pa.Table.from_pylist([{k: v for k, v in r.items() if k not in ['family_hi', 'family_lo']} for r in output])
    for key in ['family_hi', 'family_lo']:
        table = table.append_column(key, pa.array([r[key] for r in output], type=pa.uint64()))
    pq.write_table(table, destination, compression='gzip')
    summary = {'task': task, 'events': len(output), 'seconds': time.monotonic()-started, 'source': str(path), 'output': str(destination),
               'branch_disagreements': sum(r['tau_branch'] != r['tau_branch_catalog'] for r in output),
               'valid_tau_decays': sum(r['tau_decay_valid'] for r in output), 'charm': sum(r['charm_presence'] for r in output)}
    print(json.dumps(summary), flush=True)
    if not limit:
        (root/'truth_parts'/f'{task:03d}.json').write_text(json.dumps(summary)+'\n')


def finalize(root):
    plan = json.loads((root/'s5_plan.json').read_text())
    parts = [root/'truth_parts'/f'{i:03d}.parquet' for i in range(len(plan['tasks']))]
    tables = [pq.read_table(p) for p in parts]
    table = pa.concat_tables(tables, promote_options='permissive')
    frame = table.to_pandas()
    catalog = pd.read_parquet(root/'catalog.parquet')
    key = ['run', 'chunk', 'row']
    joined = catalog[key+['label']].merge(frame[key+['label']], on=key, validate='one_to_one', suffixes=('', '_side'))
    if len(joined) != len(catalog) or len(frame) != len(catalog) or not (joined.label == joined.label_side).all():
        raise ValueError('sidecar event-key/label mismatch')
    pq.write_table(table, root/'truth.parquet', compression='gzip')
    summary = {'events': len(frame), 'tau_branches': frame[frame.label.eq(2)].tau_branch.value_counts().to_dict(),
               'branch_disagreements': int((frame.tau_branch != frame.tau_branch_catalog).sum()),
               'mask_counts': {k: int(frame[k].sum()) for k in ['vertex_valid', 'tau_branch_valid', 'tau_prong_valid', 'tau_decay_valid', 'bar_semantic_valid', 'charm_label_valid', 'visible_target_valid']},
               'charm_counts_by_class': frame.groupby('label').charm_presence.sum().to_dict(),
               'tau_visibility': frame[frame.label.eq(2)].groupby('tau_branch')[['tau_visible_any', 'tau_visible_pixel', 'tau_visible_bar', 'tau_visible_tracker']].sum().to_dict(),
               'calibration_test_used': False}
    (root/'truth_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['initialize', 'create', 'finalize', 'objects', 'finalize_objects'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--task', type=int, default=0)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.stage == 'initialize':
        initialize(args.root, args.source)
    elif args.stage == 'create':
        create(args.root, args.task, args.limit)
    elif args.stage == 'finalize':
        finalize(args.root)
    elif args.stage == 'finalize_objects':
        from analysis.flavour.s5.objects import finalize_objects
        finalize_objects(args.root)
    else:
        from analysis.flavour.s5.objects import create_shard
        create_shard(args.root, args.task, args.limit)


if __name__ == '__main__':
    main()
