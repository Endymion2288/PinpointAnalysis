"""Consolidate validated S5 outputs while retaining selected models and evidence."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pyarrow.parquet as pq

from analysis.flavour.s2.train import atomic_output


def run(root):
    acceptance = json.loads((root/'acceptance.json').read_text())
    if not acceptance['scientific_complete'] or acceptance['neural_models'] != 26:
        raise ValueError('full scientific audit must precede output consolidation')
    expected = {row['name'] for row in acceptance['models']}
    rows = []
    def remove(path, reason):
        size = path.stat().st_size
        path.unlink()
        rows.append({'path': str(path.relative_to(root)), 'bytes': size, 'reason': reason})
    truth = root/'truth.parquet'
    fields = pq.read_schema(truth).names
    excluded = ['pixel_truth_rows', 'tracker_truth_rows']
    if any(k in fields for k in excluded):
        before = truth.stat().st_size
        table = pq.read_table(truth, columns=[k for k in fields if k not in excluded])
        if table.num_rows != 110000:
            raise ValueError('truth coverage differs before compaction')
        atomic_output(truth, lambda stream: pq.write_table(table, stream, compression='gzip'))
        rows.append({'path': 'truth.parquet', 'bytes': before-truth.stat().st_size,
                     'reason': 'drop response-level matching arrays after validated object matching; ROOT source and extraction code retained'})
        (root/'truth_compaction.json').write_text(json.dumps({'rows': table.num_rows, 'dropped_columns': excluded,
            'kept_columns': table.schema.names, 'source': str(root/'s5_plan.json'),
            'regeneration': 'analysis.flavour.s5.prepare create tasks 0-159, then finalize before rerunning object matching'}, indent=2)+'\n')
    summaries = []
    for path in sorted((root/'truth_parts').glob('*.json')):
        summaries.append(json.loads(path.read_text()))
    if summaries:
        (root/'runtime/truth_extraction_tasks.json').write_text(json.dumps(summaries, indent=2)+'\n')
    for directory in [root/'truth_parts', root/'object_parts']:
        if not directory.exists():
            continue
        for path in sorted(directory.iterdir()):
            if path.is_file():
                remove(path, 'merged into validated compact sidecar and task summary')
        directory.rmdir()
    for directory in sorted((root/'experiments').iterdir()):
        if not directory.is_dir() or not (directory/'manifest.json').exists():
            continue
        status = json.loads((directory/'manifest.json').read_text())['status']
        if directory.name in expected:
            if status != 'complete' or not (directory/'best.pt').exists():
                raise ValueError('selected checkpoint missing during consolidation')
            if (directory/'last.pt').exists():
                remove(directory/'last.pt', 'selected best checkpoint and full predictions retained')
        elif status in ['interrupted', 'failed', 'superseded']:
            for path in directory.glob('*.pt'):
                remove(path, 'interrupted fit excluded; retain manifest, learning history, health and retry provenance')
    for pattern in ['truth_smoke.parquet', 'truth_smoke_*.parquet', 'object_smoke_*.parquet', 'objects_smoke_*.parquet']:
        for path in root.glob(pattern):
            remove(path, 'complete validated sidecar supersedes small preparation output')
    intermediate = []
    final = root/'evaluation/final'
    for directory in sorted((root/'evaluation').iterdir()):
        if directory == final or not directory.is_dir():
            continue
        summary_path = directory/'summary.json'
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            intermediate.append({'tag': directory.name, 'models': [e['name'] for e in summary['experiments']],
                                 'seconds': summary['elapsed_seconds'], 'replacement': str(final)})
        for path in directory.iterdir():
            if path.is_file():
                remove(path, 'same model predictions and expanded diagnostics retained in final evaluation')
        directory.rmdir()
    (root/'runtime/intermediate_evaluations.json').write_text(json.dumps(intermediate, indent=2)+'\n')
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'complete': True,
              'bytes_released': sum(row['bytes'] for row in rows), 'files_removed_or_compacted': rows,
              'AFS_cleanup_complete': False, 'AFS_cleanup_scope': 'run from submission host after final Condor terminal evidence',
              'best_checkpoints_retained': len(expected),
              'preserved': 'best checkpoints, HGB fits, full validation predictions, final figures/tables, health/history, immutable source/config provenance and all backend evidence',
              'calibration_test_used': False}
    (root/'cleanup.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'files_removed_or_compacted'}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)


if __name__ == '__main__':
    main()
