"""Fixed small Tune stress: accepted noise/combined proxies, paired ideal tracker RNG."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import time

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import uproot

from analysis.flavour.create_data import ROOT_TREES, lists
from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.readout import tracker_measurements, modality_state
from analysis.flavour.s2.dataset import SpatialDataset
from analysis.flavour.s6.conditions import Condition, family_seed
from analysis.flavour.v2.partitions import KEY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--conditions', type=Path, required=True)
    parser.add_argument('--hardware', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    started = time.perf_counter()
    pa.set_cpu_count(args.threads)
    selected = pd.read_parquet(args.root/'data/stress_rows.parquet')
    plan = json.loads((args.root/'data/plan.json').read_text())
    geometry = json.loads(Path(plan['geometry']).read_text())
    hardware = json.loads(args.hardware.read_text())
    accepted = json.loads(args.conditions.read_text())
    names = {'nominal': 'nominal', 'noise': 'sparse_noise', 'combined': 'combined'}
    parameters = {name: next(v['parameters'] for v in accepted['variants'] if v['name'] == source)
                  for name, source in names.items()}
    output = args.root/'stress'
    output.mkdir(exist_ok=False)
    ideals = {}
    with ThreadPoolExecutor(args.threads) as executor:
        for (run, chunk), group in selected.groupby(KEY[:2], sort=True):
            path = Path(plan['input_root'])/str(run)/f'{run}_{chunk:03d}.root'
            with uproot.open(path, array_cache=None, object_cache=None, decompression_executor=executor,
                             interpretation_executor=executor) as source:
                for block, rows in group.groupby(group.row//32, sort=True):
                    start = int(block)*32
                    raw = source[ROOT_TREES['tracker']].arrays(['trackerID', 'x', 'y', 'z'], entry_start=start,
                        entry_stop=min(start+32, source['event'].num_entries), library='ak')
                    for meta in rows.to_dict('records'):
                        event = {name: ak.to_numpy(raw[name][meta['row']-start]) for name in raw.fields}
                        ideal, _ = tracker_measurements(event, family_seed(meta))
                        ideals[tuple(meta[k] for k in KEY)] = ideal
    conditions = {name: Condition(geometry, p, hardware) for name,p in parameters.items()}
    observations = {name: [] for name in names}
    metadata = []
    with tempfile.TemporaryDirectory(prefix='r1-stress-') as scratch:
        dataset = SpatialDataset(args.root/'data', 'validation', {'cache_shards': 2}, cache_dir=scratch)
        positions = pd.MultiIndex.from_frame(dataset.metadata[KEY]).get_indexer(pd.MultiIndex.from_frame(selected[KEY]))
        if (positions < 0).any():
            raise ValueError('stress physical keys absent from nominal Tune')
        for position in sorted(positions):
            meta = dataset.metadata.iloc[position].to_dict()
            original = dataset.native[int(dataset.indexes[position])]['observations']
            ideal = ideals[tuple(meta[k] for k in KEY)]
            # Saved float32 crossings must reproduce the accepted nominal measurements.
            reproduced = Condition(geometry, {'tracker_sigma_scale': 1.}, hardware)(original, meta, ideal)
            if not all(np.array_equal(v, reproduced['tracker_hypothesis'][k]) for k,v in original['tracker_hypothesis'].items()):
                raise ValueError('ideal tracker crossings fail nominal RNG readback')
            for name, condition in conditions.items():
                changed = condition(original, meta, ideal)
                observations[name].append({**{detector: lists(changed[detector]) for detector in
                    ['pixel', 'bar', 'tracker_hypothesis']}, 'has_response': changed['has_response'].tolist(),
                    'outside_acceptance': [-1, -1, -1], 'fit_attempted': False, 'momentum_censored': -1})
            metadata.append(meta)
        dataset.native.close()
    frame = pd.DataFrame(metadata)
    for name in names:
        folder = output/name
        folder.mkdir()
        changed_geometry = output/f'{name}_geometry.json'
        write_json(changed_geometry, conditions[name].geometry)
        write_json(folder/'plan.json', {**plan, 'geometry': str(changed_geometry), 'condition': parameters[name]})
        for shard, start in enumerate(range(0, len(frame), 400)):
            destination = folder/'shards'/f'{shard:03d}'
            destination.mkdir(parents=True)
            rows = frame.iloc[start:start+400].copy()
            rows['shard'] = shard
            current = observations[name][start:start+400]
            for detector, field, column in [('pixel', 'channel_id', 'pixel_channels'),
                    ('bar', 'channel_id', 'bar_channels'), ('tracker_hypothesis', 'station', 'tracker_channels')]:
                rows[column] = [len(event[detector][field]) for event in current]
            rows['all_empty'] = rows[['pixel_channels', 'bar_channels', 'tracker_channels']].sum(1).eq(0)
            pq.write_table(pa.Table.from_pylist(current), destination/'observations.parquet', compression=COMPRESSION)
            pq.write_table(pa.Table.from_pylist([{'label': int(y), 'label_status': 'consistent'} for y in rows.label]),
                           destination/'targets.parquet', compression=COMPRESSION)
            rows.to_parquet(destination/'provenance.parquet', index=False, compression=COMPRESSION)
        frame.to_parquet(folder/'catalog.parquet', index=False, compression=COMPRESSION)
    write_json(output/'preparation.json', {'status': 'complete', 'events': len(frame),
        'raw_counts': frame.label.value_counts().sort_index().to_dict(), 'conditions': parameters,
        'selection': 'frozen Tune family hash, 400/class', 'nominal_tracker_redigitization_exact': True,
        'heldout_pattern_used': False, 'Cal_A_Cut_B_Confirm_payload_read': False,
        'seconds': time.perf_counter()-started})


if __name__ == '__main__':
    main()
