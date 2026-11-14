"""Share compressed native inputs and claim independent fits across backends."""
import argparse
import json
import os
from pathlib import Path
import sys

from analysis.flavour.s3.parallel import main as parallel_main
from analysis.flavour.s4.stage_inputs import stage_shards


def main():
    # Parse only staging arguments here; the existing scheduler owns children.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--name', required=True)
    known, _ = parser.parse_known_args()
    scratch = Path(os.environ.get('PINPOINT_S5_SCRATCH', os.environ.get('_CONDOR_SCRATCH_DIR', '/tmp')))
    local = scratch/'native_inputs'
    paths = sorted(p for p in (known.root/'shards').iterdir() if (p/'observations.parquet').exists())
    report = stage_shards(paths, local, workers=3)
    (known.root/'runtime'/f'{known.name}.staging.json').write_text(json.dumps(report, indent=2)+'\n')
    os.environ['PINPOINT_NATIVE_SHARD_ROOT'] = str(local)
    os.environ['PINPOINT_CLASSIFICATION_CACHE_MIB'] = '256'
    sys.argv.extend(['--training-module', 'analysis.flavour.s5.train', '--claim-tasks'])
    parallel_main()


if __name__ == '__main__':
    main()
