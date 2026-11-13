"""Compare cached classification tensors and timing on a fixed label subset."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import pyarrow as pa
import torch

from analysis.flavour.inventory import write_json
from analysis.flavour.s3.dataset import collate_planes
from analysis.flavour.s4.dataset import LabelDataset
from analysis.flavour.s4.stage_inputs import stage_shards


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--task', type=int, required=True)
    args = parser.parse_args()
    config = json.loads(args.campaign.read_text())['experiments'][args.task]
    output = args.root / 'experiments' / config['name']; output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1); pa.set_cpu_count(1); pa.set_io_thread_count(1)
    started = time.perf_counter()
    manifest = {'utc': datetime.now(timezone.utc).isoformat(), 'config': config, 'status': 'running', 'GPU': None,
        'code_commit': os.environ.get('PINPOINT_CODE_COMMIT') or subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
    write_json(output / 'manifest.json', manifest)
    paths = sorted(p for p in (args.root / 'shards').iterdir() if (p / 'observations.parquet').exists())
    with tempfile.TemporaryDirectory(prefix='pinpoint-classification-cache-', dir=os.environ.get('_CONDOR_SCRATCH_DIR')) as temporary:
        staging = stage_shards(paths, Path(temporary) / 'native', workers=3)
        os.environ['PINPOINT_NATIVE_SHARD_ROOT'] = str(Path(temporary) / 'native')
        readers = {}
        for name, mib in [('original', 0), ('cached', 1024)]:
            os.environ['PINPOINT_CLASSIFICATION_CACHE_MIB'] = str(mib)
            readers[name] = LabelDataset(args.root, 'train', config['settings'], cache_dir=Path(temporary) / name)
        if not readers['original'].metadata.equals(readers['cached'].metadata):
            raise AssertionError('classification cache changed selected metadata')
        seconds = {key: [] for key in readers}
        for repeat in range(2):
            elapsed = dict.fromkeys(readers, 0.)
            for start in range(0, len(readers['original']), 16):
                batches = {}
                for name in (['original', 'cached'] if repeat == 0 else ['cached', 'original']):
                    before = time.perf_counter()
                    batches[name] = collate_planes([readers[name][i] for i in range(start, min(start+16, len(readers[name])))])
                    elapsed[name] += time.perf_counter() - before
                for key in batches['original']:
                    if not torch.equal(batches['original'][key], batches['cached'][key]):
                        raise AssertionError('classification cache changed tensor: ' + key)
            for name in readers:
                seconds[name].append(elapsed[name])
        report = {'all_collated_tensors_and_metadata_bitwise_equal': True, 'events': len(readers['original']),
            'read_transform_collate_seconds_by_pass': seconds, 'initial_shared_staging': staging,
            'cache': readers['cached'].event_cache_stats(),
            'native_reader_statistics': {name: data.native.cache_stats() for name, data in readers.items()},
            'scope': 'two complete sequential passes through fixed 1k nested-label subset; alternating reader order, same shared native files; no model/GPU timing'}
        for data in readers.values(): data.native.close()
    write_json(output / 'equivalence.json', report)
    manifest.update(status='complete', seconds=time.perf_counter()-started)
    write_json(output / 'manifest.json', manifest)
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
