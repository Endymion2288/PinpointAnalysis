"""Convert only the R0 10k train stream and compact 256-event micro sample."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.s2.prepare import convert
from analysis.flavour.v2.partitions import KEY, FAMILY, equal_identity_rows


def plan(root):
    if (root/'profile_data/plan.json').exists():
        raise ValueError('profile event plan already exists; resume selected conversion tasks from its frozen manifest')
    partitions = json.loads((root/'partitions.json').read_text())
    config = partitions['config']
    train = pd.read_parquet(partitions['manifests'][config['profile_train_pool']]['manifest'])
    rows = []
    per_class = config['profile_events']//4
    per_shard_class = per_class//4
    for _, group in train.groupby('label', sort=True):
        selected = group.sort_values('class_train_rank').head(per_class).copy()
        selected['shard'] = np.arange(len(selected))//per_shard_class
        rows.append(selected)
    frame = pd.concat(rows, ignore_index=True).sort_values(KEY)
    if len(frame) != config['profile_events'] or set(frame.shard) != set(range(4)):
        raise ValueError('R0 profile requires four complete balanced shards')
    folder = root/'profile_data'
    folder.mkdir(exist_ok=True)
    frame.to_parquet(folder/'manifest.parquet', index=False, compression=COMPRESSION)
    write_json(folder/'plan.json', {'seed': config['seed'], 'geometry': config['geometry'],
        'input_root': config['input_root'], 'tasks': [[i] for i in range(4)],
        'source': str(root/'partitions.json'), 'split': 'train',
        'selection': 'first 2500 per class in frozen chunk-efficient train order',
        'ROOT_files': len(frame[KEY[:2]].drop_duplicates()), 'Confirm_responses_read': False})


def convert_one(arguments):
    root, task, cpus = arguments
    convert(SimpleNamespace(root=root/'profile_data', output_root=None,
                            task=task, cpus=cpus, smoke_per_class=0))


def finalize(root):
    folder = root/'profile_data'
    frames = []
    summaries = []
    for path in sorted((folder/'shards').iterdir()):
        frame = pd.read_parquet(path/'provenance.parquet')
        table = pq.read_table(path/'observations.parquet', columns=['pixel','bar','tracker_hypothesis'])
        for group, field, output in [('pixel','channel_id','pixel_channels'), ('bar','channel_id','bar_channels'),
                                     ('tracker_hypothesis','station','tracker_channels')]:
            frame[output] = np.diff(table[group].combine_chunks().field(field).offsets.to_numpy())
        frame['source_shard'] = int(path.name)
        frame['source_row'] = np.arange(len(frame))
        frames.append(frame)
        summary = json.loads((path/'summary.json').read_text())
        if summary['PE_input'] != summary['PE_output']:
            raise ValueError('converted bar PE total changed')
        summaries.append(summary)
    catalog = pd.concat(frames, ignore_index=True)
    expected = pd.read_parquet(folder/'manifest.parquet')
    if not equal_identity_rows(catalog, expected):
        raise ValueError('profile conversion changed event/family identities')
    catalog['profile_index'] = np.arange(len(catalog))
    catalog.to_parquet(folder/'catalog.parquet', index=False, compression=COMPRESSION)
    work = catalog[['pixel_channels','bar_channels','tracker_channels']].sum(axis=1)
    chosen = []
    for _, group in catalog.groupby('label', sort=True):
        ordered = group.index[np.argsort(work[group.index], kind='stable')]
        chosen.extend(ordered[np.linspace(0,len(ordered)-1,64).astype(int)].tolist())
    micro = catalog.iloc[chosen].copy().reset_index(drop=True)
    micro['micro_index'] = np.arange(len(micro))
    micro['occupancy_band'] = np.tile(np.repeat(['low','medium','high'],[22,21,21]),4)
    micro.to_parquet(root/'manifests/micro_256.parquet', index=False, compression=COMPRESSION)
    compact = root/'micro_data'
    for shard, group in micro.groupby('source_shard', sort=True):
        destination = compact/'shards'/f'{shard:03d}'
        destination.mkdir(parents=True, exist_ok=True)
        indexes = pa.array(group.source_row.to_numpy())
        for name in ['observations.parquet','targets.parquet']:
            table = pq.read_table(folder/'shards'/f'{shard:03d}'/name).take(indexes)
            pq.write_table(table,destination/name,compression=COMPRESSION,row_group_size=64)
        group.to_parquet(destination/'provenance.parquet',index=False,compression=COMPRESSION)
    config = json.loads((folder/'plan.json').read_text())
    write_json(compact/'plan.json', {**config, 'role':'fixed 256 real Train events at per-class occupancy quantiles'})
    micro.to_parquet(compact/'catalog.parquet',index=False,compression=COMPRESSION)
    write_json(root/'preparation.json', {'status':'complete','profile_events':len(catalog),'micro_events':len(micro),
        'profile_class_counts':catalog.groupby('label').size().to_dict(),
        'micro_class_counts':micro.groupby('label').size().to_dict(),
        'native_workload_quantiles': {str(q):float(v) for q,v in work.quantile([0,.5,.9,.99,1]).items()},
        'exact_event_family_readback':True, 'bar_PE_conserved':True,
        'source_summaries':summaries, 'Confirm_responses_read':False})
    print(json.dumps({'profile_events':len(catalog),'micro_events':len(micro),'status':'complete'}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--processes',type=int,default=4)
    parser.add_argument('--threads-per-process',type=int,default=2)
    args=parser.parse_args()
    plan(args.root)
    with ProcessPoolExecutor(args.processes,mp_context=multiprocessing.get_context('spawn')) as executor:
        list(executor.map(convert_one,[(args.root,i,args.threads_per_process) for i in range(4)]))
    finalize(args.root)


if __name__=='__main__':
    main()
