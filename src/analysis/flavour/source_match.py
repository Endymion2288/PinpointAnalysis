"""Test a candidate gFaser source against the pilot, without assuming row identity.

Exact scalar signatures are a provenance lead, not proof of production settings.
Only scalar branches are read; source particle records are not materialized.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import uproot

from analysis.flavour.inventory import family_signatures, write_json, COMPRESSION

FIELDS = {'neu': 'initPDG', 'pxv': 'initPx', 'pyv': 'initPy', 'pzv': 'initPz', 'Ev': 'initE',
          'tgt': 'tgtPDG', 'hitnuc': 'hitnucPDG', 'Q2': 'Q2', 'x': 'xBj', 'y': 'y', 'W': 'W',
          'scatteringType': 'scatteringType', 'fspl': 'fslPDG'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--run', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    started = time.perf_counter()
    pilot = pd.read_parquet(args.inventory, filters=[('run', '==', args.run)])
    matches, total = [], 0
    with ThreadPoolExecutor(2) as executor:
        with uproot.open(args.source, decompression_executor=executor, interpretation_executor=executor,
                         object_cache=None, array_cache=None) as root:
            tree = root['gFaser']
            for batch in tree.iterate(list(FIELDS) + ['iev'], step_size=50000, library='np'):
                source = pd.DataFrame({target: batch[name] for name, target in FIELDS.items()})
                for key in ['initPx', 'initPy', 'initPz', 'initE']:
                    source[key] = source[key].to_numpy(dtype='float64') * 1000.
                sig = family_signatures(source)
                candidates = pd.DataFrame({'family_hi': sig[:, 0], 'family_lo': sig[:, 1],
                                            'source_entry': np.arange(total, total+len(source)), 'source_iev': batch['iev']})
                found = pilot[['run', 'chunk', 'row', 'evtID', 'family_hi', 'family_lo']].merge(candidates, on=['family_hi', 'family_lo'], how='inner')
                if len(found):
                    matches.append(found)
                total += len(source)
    args.output.mkdir(parents=True, exist_ok=True)
    found = pd.concat(matches, ignore_index=True) if matches else pd.DataFrame()
    if len(found):
        found.to_parquet(args.output / f'matches_{args.run}.parquet', index=False, compression=COMPRESSION)
    write_json(args.output / f'source_{args.run}.json', {'run': args.run, 'source': str(args.source),
               'source_entries_scanned': total, 'pilot_events': len(pilot), 'exact_signature_matches': len(found),
               'matched_pilot_events': len(found.drop_duplicates(['run', 'chunk', 'row'])) if len(found) else 0,
               'row_identity_matches': int((found.source_entry == found.evtID).sum()) if len(found) else 0,
               'method': 'all thirteen canonical scalar signature fields, source GeV converted to MeV',
               'production_macros_verified': False, 'seconds': time.perf_counter()-started})
    print(args.run, total, 'source entries;', len(found), 'signature matches')


if __name__ == '__main__':
    main()
