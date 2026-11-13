"""Copy immutable compressed observations once per formally allocated worker."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time


FILES = ('observations.parquet', 'targets.parquet')


def stage_shards(paths, output, workers=3):
    output = Path(output).resolve()
    if str(output).startswith(('/eos/', '/afs/')):
        raise ValueError('compressed input staging requires local allocation scratch')
    paths = list(paths)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    def copy(path):
        destination = output / path.name
        destination.mkdir(exist_ok=True)
        size = 0
        for name in FILES:
            shutil.copyfile(path / name, destination / name)
            size += (destination / name).stat().st_size
        return {'shard': path.name, 'bytes': size}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(copy, paths))
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'shards': rows,
        'compressed_bytes': sum(row['bytes'] for row in rows),
        'copy_seconds': time.perf_counter() - started, 'copy_threads': workers,
        'local_shard_root': str(output), 'files': list(FILES),
        'scope': 'allocation-owned byte copies, shared by child readers; bounded decoded-shard LRU retained; no tensor, sampler, target or normalization changes'}
    (output / 'ready.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    paths = sorted(p for p in (args.root / 'shards').iterdir() if (p / 'observations.parquet').exists())
    report = stage_shards(paths, args.output, args.workers)
    from analysis.flavour.inventory import write_json
    write_json(args.record, report)
    print(json.dumps({key: value for key, value in report.items() if key != 'shards'}), flush=True)


if __name__ == '__main__':
    main()
