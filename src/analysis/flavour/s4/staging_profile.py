"""Measure EOS/restaged-local readers on the same immutable native events."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
import pyarrow as pa
import torch

from analysis.flavour.datasets import CachedNativeDataset, collate_native
from analysis.flavour.inventory import write_json
from analysis.flavour.s4.stage_inputs import stage_shards


def tensors(value, prefix=''):
    if isinstance(value, dict):
        return {key: item for k, v in value.items() for key, item in tensors(v, prefix + '/' + k).items()}
    return {prefix: value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--task', type=int, required=True)
    args = parser.parse_args()
    config = json.loads(args.campaign.read_text())['experiments'][args.task]
    output = args.root / 'experiments' / config['name']
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1); pa.set_cpu_count(1); pa.set_io_thread_count(1)
    started = time.perf_counter()
    manifest = {'utc': datetime.now(timezone.utc).isoformat(), 'config': config, 'status': 'running', 'GPU': None,
        'code_commit': os.environ.get('PINPOINT_CODE_COMMIT') or subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
    write_json(output / 'manifest.json', manifest)
    all_paths = sorted(p for p in (args.root / 'shards').iterdir() if (p / 'observations.parquet').exists())
    paths = [all_paths[i] for i in np.linspace(0, len(all_paths)-1, config.get('shards', 4)).astype(int)]
    with tempfile.TemporaryDirectory(prefix='pinpoint-shared-input-', dir=os.environ.get('_CONDOR_SCRATCH_DIR')) as temporary:
        temporary = Path(temporary)
        staging = stage_shards(paths, temporary / 'shared', workers=3)
        readers = {name: CachedNativeDataset(source, cache_dir=temporary / name, max_shards=1)
            for name, source in [('EOS', paths), ('local', [temporary / 'shared' / p.name for p in paths])]}
        reference = {}
        totals = {name: 0. for name in readers}
        for repeat in range(config.get('repeats', 2)):
            for name in (['EOS', 'local'] if repeat % 2 == 0 else ['local', 'EOS']):
                reader = readers[name]
                for index in reader.boundaries[:-1]:
                    before = time.perf_counter()
                    batch = tensors(collate_native([reader[index + i] for i in range(config.get('events_per_shard', 8))]))
                    totals[name] += time.perf_counter() - before
                    if index not in reference:
                        reference[index] = batch
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            if not torch.equal(value, reference[index][key]):
                                raise AssertionError('native staged tensor differs: ' + key)
                        elif value != reference[index][key]:
                            raise AssertionError('native staged metadata differs: ' + key)
        statistics = {name: reader.cache_stats() for name, reader in readers.items()}
        for reader in readers.values():
            reader.close()
    report = {'shards': [p.name for p in paths], 'events_per_shard': config.get('events_per_shard', 8),
        'repeats': config.get('repeats', 2), 'all_native_tensors_and_metadata_bitwise_equal': True,
        'initial_shared_staging': staging, 'native_read_and_collate_seconds': totals,
        'reader_statistics': statistics,
        'scope': 'bounded one-shard decoded LRU, alternating EOS/local-source passes; timing includes native staging/decoding/collation and excludes model; not end-to-end GPU throughput'}
    write_json(output / 'equivalence.json', report)
    manifest.update(status='complete', seconds=time.perf_counter() - started)
    write_json(output / 'manifest.json', manifest)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
