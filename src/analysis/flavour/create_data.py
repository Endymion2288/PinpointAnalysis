"""Convert one inventory-approved ROOT chunk into compact variable-length Parquet.

Observations, labels and provenance are physically separate files. Readout level
is a simulated detector proxy; L3 tracker digitization remains a hypothesis.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import platform
import resource
import time

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import uproot

from analysis.flavour.features import event_features, FEATURE_NAMES
from analysis.flavour.inventory import scan_file, write_json, COMPRESSION
from analysis.flavour.readout import pixel_channels, bar_channels, tracker_measurements, modality_state

BRANCHES = {
    'pixel': ['event_id', 'layerID', 'rowID', 'colID'],
    'bar': ['event_id', 'layerID', 'panelID', 'rowID', 'colID', 'isHorizontal', 'edep', 'nPE'],
    'tracker': ['event_id', 'trackerID', 'x', 'y', 'z'],
}
ROOT_TREES = {'pixel': 'Hits/pixelHits', 'bar': 'Hits/scintHits', 'tracker': 'Hits/faserHits'}


def lists(mapping):
    return {key: value.tolist() for key, value in mapping.items()}


def convert(args):
    pa.set_cpu_count(args.cpus)
    pa.set_io_thread_count(args.cpus)
    started = time.perf_counter()
    timing = dict(root_read=0., aggregation=0., write=0.)
    args.output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.cpus) as executor:
        metadata, audit, geometry, schema = scan_file(args.input, executor)
        if not audit['usable_join']:
            raise ValueError('file alignment validation failed')
        count = min(len(metadata), args.events) if args.events else len(metadata)
        metadata = metadata.iloc[:count].copy()
        observations, ideal_rows, features, conservation = [], [], [], []
        with uproot.open(args.input, decompression_executor=executor, interpretation_executor=executor,
                         object_cache=None, array_cache=None) as root:
            for start in range(0, count, args.batch_events):
                t = time.perf_counter()
                stop = min(start + args.batch_events, count)
                batch = {name: root[ROOT_TREES[name]].arrays(keys, entry_start=start,
                         entry_stop=stop, library='ak') for name, keys in BRANCHES.items()}
                timing['root_read'] += time.perf_counter() - t
                t = time.perf_counter()
                for local in range(stop - start):
                    row = start + local
                    pixel = pixel_channels(batch['pixel'][local], geometry)
                    bars, conserved = bar_channels(batch['bar'][local], geometry)
                    seed = int(metadata.iloc[row].family_hi) ^ int(metadata.iloc[row].family_lo) ^ 20261111
                    ideal, digitized = tracker_measurements(batch['tracker'][local], seed)
                    state = modality_state(pixel, bars, digitized)
                    if not np.isclose(conserved['input_edep_MeV'], conserved['output_edep_MeV'], rtol=1e-10, atol=1e-8):
                        raise ValueError('bar energy is not conserved')
                    if conserved['input_pe'] != conserved['output_pe']:
                        raise ValueError('bar PE is not conserved')
                    observations.append({'pixel': lists(pixel), 'bar': lists(bars),
                                         'tracker_hypothesis': lists(digitized),
                                         'has_response': state['has_response'].tolist(),
                                         'outside_acceptance': state['outside_acceptance'].tolist(),
                                         'fit_attempted': state['fit_attempted'],
                                         'momentum_censored': int(state['momentum_censored'])})
                    ideal_rows.append(lists(ideal))
                    features.append(event_features(pixel, bars))
                    conservation.append(conserved)
                timing['aggregation'] += time.perf_counter() - t
    t = time.perf_counter()
    table = pa.Table.from_pylist(observations)
    pq.write_table(table, args.output / 'observations.parquet', compression=COMPRESSION, row_group_size=64)
    pq.write_table(pa.Table.from_pylist(ideal_rows), args.output / 'ideal_tracker_proxy.parquet', compression=COMPRESSION)
    feature_table = pa.table({name: np.asarray(features)[:, i] for i, name in enumerate(FEATURE_NAMES)})
    pq.write_table(feature_table, args.output / 'b0_features.parquet', compression=COMPRESSION)
    provenance_keys = ['row', 'run', 'chunk', 'evtID', 'vtxID', 'geometry_id', 'family_hi', 'family_lo', 'usable_join']
    metadata[provenance_keys].to_parquet(args.output / 'provenance.parquet', index=False, compression=COMPRESSION)
    metadata.drop(columns=provenance_keys).to_parquet(args.output / 'targets.parquet', index=False, compression=COMPRESSION)
    write_json(args.output / 'geometry.json', geometry)
    timing['write'] = time.perf_counter() - t
    sizes = {p.name: p.stat().st_size for p in args.output.glob('*.parquet')}
    conserved = pd.DataFrame(conservation)
    quantiles = {key: {str(q): float(v) for q,v in conserved[key].quantile([0, .5, .9, .95, .99, .999, 1]).items()}
                 for key in ['records', 'channels', 'input_edep_MeV', 'input_pe']}
    summary = {'input': str(args.input), 'events': count, 'alignment': audit['alignment'],
               'inventory_seconds': audit['seconds'], 'timing_seconds': timing,
               'seconds': time.perf_counter() - started, 'parquet_bytes': sizes,
               'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
               'quantiles': quantiles, 'bar_edep_conservation_max_abs_MeV':
               float(np.max(np.abs(conserved.input_edep_MeV - conserved.output_edep_MeV))),
               'bar_pe_conservation': bool((conserved.input_pe == conserved.output_pe).all()),
               'readout_level': 'simulated_detector_proxy_unvalidated_electronics',
               'tracker': {'hypothesis': 'L3_resolution_efficiency_plus_0.1mm_grid_merge',
                           'sigma_xy_mm': [.58, .017], 'efficiency': .99, 'production_validated': False,
                           'noise': False, 'momentum_fit': False, 'field_map': None},
               'pixel': 'native_binary_no_edep_feature', 'timing_used': False, 'compression': COMPRESSION,
               'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                            'awkward': ak.__version__, 'uproot': uproot.__version__},
               'feature_names': FEATURE_NAMES}
    write_json(args.output / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--events', type=int, default=0, help='0 means full chunk')
    parser.add_argument('--batch-events', type=int, default=16)
    parser.add_argument('--cpus', type=int, default=2)
    convert(parser.parse_args())


if __name__ == '__main__':
    main()
