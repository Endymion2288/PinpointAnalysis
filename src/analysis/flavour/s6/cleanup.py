"""Compact completed S6 artifacts after all consumers and backends close."""
import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import shutil
import tarfile

from analysis.flavour.inventory import write_json


def run(args):
    root = args.root.resolve()
    audit = json.loads((root/'evaluation/scope_completion_audit.json').read_text())
    remaining = [n for n in audit['open_requirements'] if n != 'resource allocation evidence, backend closure and compact outputs']
    cost = json.loads((root/'evaluation/resource_accounting.json').read_text())
    closure = json.loads((root/'evaluation/backend_closure.json').read_text())
    if remaining or cost['active_jobs'] or not cost['backend_queries_complete']:
        raise ValueError('analysis or resource evidence still incomplete')
    if any(f['unresolved_rows'] for f in cost['submitted_condor_factories']):
        raise ValueError('unresolved backend records')
    if closure['condor_active_jobs'] or closure['kubeflow_active_pods']:
        raise ValueError('backend still active')
    targets = ['shards', 'predictions/nominal', 'predictions/variants', 'predictions/transport', 'predictions/generator',
        'transport/shards', 'generator/shards', 'transport/truth_replay', 'generator/truth_replay', 'transport/input_parts']
    sizes = {}
    for name in targets:
        path = root/name
        sizes[name] = sum(p.stat().st_size for p in path.rglob('*') if p.is_file()) if path.exists() else 0
    record = {'UTC': datetime.now(timezone.utc).isoformat(), 'apply': args.apply,
        'removable_bytes_by_directory': sizes, 'removable_total_bytes': sum(sizes.values()),
        'retained': ['full raw and per-model/scenario calibrated scores', 'compact merged catalog/truth/quality and decay truth',
            'all metrics, curves, bootstrap replicates and English figures', 'frozen configuration, source snapshots and G4 executables',
            'backend terminal/cancellation evidence', 'compressed per-task summaries and logs']}
    if not args.apply:
        print(json.dumps(record), flush=True)
        return
    summary_archive = root/'audit/compacted_task_summaries.jsonl.gz'
    count = 0
    with gzip.open(summary_archive, 'wt', encoding='utf-8') as stream:
        for name in targets:
            for p in sorted((root/name).rglob('summary.json')):
                stream.write(json.dumps({'path': str(p.relative_to(root)), 'summary': json.loads(p.read_text())})+'\n')
                count += 1
    # Opening the complete archive again catches a truncated stream before any
    # source directory is removed. No checksum or extra event data scan.
    with gzip.open(summary_archive, 'rt', encoding='utf-8') as stream:
        if sum(1 for _ in stream) != count:
            raise ValueError('summary archive readback differs')
    logs = root/'logs'
    log_bytes = sum(p.stat().st_size for p in logs.rglob('*') if p.is_file())
    archive = root/'audit/completed_logs.tar.gz'
    with tarfile.open(archive, 'w:gz') as stream:
        stream.add(logs, arcname='logs')
    with tarfile.open(archive, 'r:gz') as stream:
        log_members = len(stream.getmembers())
    for name in targets:
        path = root/name
        if path.exists():
            shutil.rmtree(path)
    shutil.rmtree(logs)
    logs.mkdir()
    afs_base = Path.home()/'condor/Pinpoint/flavour_s6'
    removed_scripts = []
    for p in (root/'runtime').glob('*.submission.json'):
        value = json.loads(p.read_text())
        if value.get('backend') == 'condor' and value.get('submit_file'):
            directory = Path(value['submit_file']).parent
            if directory.is_relative_to(afs_base) and directory.exists():
                removed_scripts.append(str(directory))
                shutil.rmtree(directory)
    record.update(status='complete', task_summaries=count, task_summary_archive=str(summary_archive),
        task_summary_archive_bytes=summary_archive.stat().st_size, log_archive=str(archive),
        log_archive_bytes=archive.stat().st_size, log_archive_members=log_members,
        original_log_bytes=log_bytes,
        removed_AFS_directories=removed_scripts,
        net_released_bytes=sum(sizes.values())+log_bytes-summary_archive.stat().st_size-archive.stat().st_size)
    write_json(root/'evaluation/cleanup.json', record)
    print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
