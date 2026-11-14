"""Train-only tiny end-to-end runner check; simulated validation is not Tune."""

import argparse
import json
from pathlib import Path
import tempfile
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.datasets import NativeShard
from analysis.flavour.create_data import lists
from analysis.flavour.v2.fit import fit_event_positions
from analysis.flavour.v2.r1.config import recipe
from analysis.flavour.v2.r1.data import R1Dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--r0', type=Path, required=True)
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    output = args.root/'smoke'/args.attempt
    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='r1-integrity-') as scratch:
        source = R1Dataset(args.r0/'micro_data', 'train', recipe(args.r0, 'B1'), cache_dir=scratch)
        selected = pd.read_parquet(args.r0/'fit/fit32_a2_events.parquet').copy()
        positions = fit_event_positions(selected, source.metadata)
        selected['split'] = np.where(selected.groupby('label').cumcount()<2, 'validation', 'train')
        for column in ['pixel_centered', 'large_scintillator']:
            selected[column] = True
        observed = []
        for index in positions:
            native = source.native[int(source.indexes[index])]['observations']
            observed.append({**{name: lists(native[name]) for name in ['pixel', 'bar', 'tracker_hypothesis']},
                'has_response': np.asarray(native['has_response']).tolist(), 'outside_acceptance': [-1]*3,
                'fit_attempted': False, 'momentum_censored': -1})
        source.native.close()
        shard = output/'data/shards/000'
        shard.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(observed), shard/'observations.parquet', compression=COMPRESSION)
        pq.write_table(pa.Table.from_pylist([{'label': int(y), 'label_status': 'consistent'} for y in selected.label]),
                       shard/'targets.parquet', compression=COMPRESSION)
        selected.to_parquet(shard/'provenance.parquet', index=False, compression=COMPRESSION)
        selected.to_parquet(output/'data/catalog.parquet', index=False, compression=COMPRESSION)
        selected[selected.split.eq('validation')].to_parquet(output/'data/tune_screen.parquet', index=False, compression=COMPRESSION)
        write_json(output/'data/plan.json', json.loads((args.r0/'micro_data/plan.json').read_text()))
    for family in ['B1', 'B2', 'B3', 'B4']:
        for task in ['T3', 'T4']:
            config = recipe(args.r0, family, task)
            config.update(name=f'smoke_{family.lower()}_{task.lower()}', data_root=str(output/'data'),
                passes=1, workers=0, learning_rate=.0005, bf16=False)
            path = output/'configs'/f"{config['name']}.json"
            write_json(path, config)
            subprocess.run([sys.executable, '-m', 'analysis.flavour.v2.r1.train', '--root', str(output),
                '--config', str(path)], check=True)
    write_json(output/'integrity.json', {'status': 'complete', 'cases': 8,
        'events': 32, 'source': 'frozen Train only; runtime validation is a subset of Train',
        'actual_Tune_used': False, 'Cal_A_Cut_B_Confirm_payload_read': False,
        'scope': 'runner/score-table/coverage/checkpoint integration, not model generalization'})


if __name__ == '__main__':
    main()
