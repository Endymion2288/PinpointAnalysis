"""Lightweight, resumable ROOT inventory and provisional source-signature split.

Run ``python -m analysis.flavour.inventory --help`` for the three stages.
Only scan selected scalar metadata, IDs and one vector per detector, never truth
hit payloads. A signature is a conservative duplicate check, not source provenance.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import time

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow as pa
import uproot

from analysis.flavour import RUN_LABELS

INPUT_ROOT = Path('/eos/project-f/fasersim-bonn/public/pinpoint/data')
TREES = {'pixel': ('Hits/pixelHits', 'layerID'),
         'bar': ('Hits/scintHits', 'layerID'),
         'tracker': ('Hits/faserHits', 'trackerID')}
SCALARS = ['evtID', 'vtxID', 'initPDG', 'fslPDG', 'intType', 'scatteringType',
           'initX', 'initY', 'initZ', 'initPx', 'initPy', 'initPz', 'initE',
           'tgtPDG', 'tgtA', 'tgtZ', 'hitnucPDG', 'Q2', 'xBj', 'y', 'W', 'xs', 'weight']
SIGNATURE = ['initPDG', 'initPx', 'initPy', 'initPz', 'initE', 'tgtPDG',
             'hitnucPDG', 'Q2', 'xBj', 'y', 'W', 'scatteringType', 'fslPDG']
COMPRESSION = next((c for c in ['zstd', 'snappy', 'gzip'] if pa.Codec.is_available(c)), None)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def short_id(value):
    return hashlib.blake2b(json.dumps(value, sort_keys=True).encode(), digest_size=12).hexdigest()


def label_status(run, frame):
    nu, lepton, interaction = np.abs(frame.initPDG), np.abs(frame.fslPDG), frame.intType
    if run == 10084:
        valid = (interaction == 3) & (frame.initPDG == frame.fslPDG) & np.isin(nu, [12, 14, 16])
    else:
        expected = {10081: 12, 10082: 14, 10083: 16}[run]
        valid = ((interaction == 2) & (nu == expected) & (lepton == expected - 1)
                 & (np.sign(frame.initPDG) == np.sign(frame.fslPDG)))
    # Encoding 7 is NuElectronElastic in the inspected converter; retain an
    # explicit candidate status until the actual production converter is known.
    return np.where(valid, 'consistent', np.where(frame.scatteringType == 7,
                    'nu_e_elastic_candidate', 'unresolved_metadata'))


def family_signatures(frame):
    # Canonical little-endian doubles, no randomized vertex or file/run/entry.
    values = frame[SIGNATURE].to_numpy(dtype='<f8', copy=True)
    values[values == 0] = 0  # canonicalize signed zero
    digests = [hashlib.blake2b(row.tobytes(), digest_size=16).digest() for row in values]
    return np.frombuffer(b''.join(digests), dtype='<u8').reshape(-1, 2)


def validate_alignment(event_ids, hit_ids, counts, detector):
    if len(hit_ids) != len(event_ids):
        return {'ok': False, 'reason': 'entry_count_mismatch', 'repaired_empty_ids': 0}
    mismatched = hit_ids != event_ids
    allowed = (counts == 0) if detector == 'pixel' else np.zeros(len(counts), dtype=bool)
    return {'ok': not bool(np.any(mismatched & ~allowed)),
            'reason': 'aligned_or_empty_pixel_only',
            'id_mismatches': int(mismatched.sum()),
            'nonempty_mismatches': int(np.count_nonzero(mismatched & (counts > 0))),
            'repaired_empty_ids': int(np.count_nonzero(mismatched & allowed))}


def split_names(family_hi, family_lo, seed=20261111):
    # SplitMix64 avalanche; seed changes all bits. Signature is independent of
    # the chosen split seed and both 64-bit halves contribute to assignment.
    mixed = np.asarray(family_hi, dtype=np.uint64) ^ np.asarray(family_lo, dtype=np.uint64) ^ np.uint64(seed)
    mixed = mixed + np.uint64(0x9e3779b97f4a7c15)
    mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
    mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
    mixed = mixed ^ (mixed >> np.uint64(31))
    fraction = (mixed >> np.uint64(11)).astype('float64') / 2**53
    return np.asarray(['train', 'validation', 'calibration', 'test'])[np.searchsorted([.7, .8, .9], fraction)]


def scan_file(path, executor):
    started = time.perf_counter()
    run, chunk = map(int, Path(path).stem.split('_'))
    with uproot.open(path, decompression_executor=executor, interpretation_executor=executor,
                     object_cache=None, array_cache=None) as root:
        geometry = ak.to_list(root['geometry'].arrays(library='ak'))
        if len(geometry) != 1:
            raise ValueError(f'{path}: expected one geometry entry')
        geom = geometry[0]
        schema = {name: root[name].typenames() for name in ['event', *[v[0] for v in TREES.values()]]}
        frame = pd.DataFrame(root['event'].arrays(SCALARS, library='np'))
        event_ids = frame.evtID.to_numpy()
        unique = not frame.evtID.duplicated().any()
        record = {'path': str(path), 'run': run, 'chunk': chunk,
                  'bytes': Path(path).stat().st_size, 'events': len(frame),
                  'unique_event_ids': bool(unique), 'geometry_id': short_id(geom),
                  'schema_id': short_id(schema), 'tree_cycles': root.keys(), 'alignment': {}}
        for detector, (tree_name, count_field) in TREES.items():
            tree = root[tree_name]
            hit_ids = tree['event_id'].array(library='np')
            counts = np.concatenate([ak.to_numpy(ak.num(batch[count_field], axis=1))
                      for batch in tree.iterate([count_field], step_size='16 MB', library='ak')])
            alignment = validate_alignment(event_ids, hit_ids, counts, detector)
            record['alignment'][detector] = alignment
            frame['n_' + detector] = counts.astype('int32') if len(counts) == len(frame) else -1
        record['usable_join'] = bool(unique and all(v['ok'] for v in record['alignment'].values()))
        frame['row'] = np.arange(len(frame), dtype='int32')
        frame['run'], frame['chunk'] = np.int32(run), np.int32(chunk)
        frame['label'] = np.int8(list(RUN_LABELS).index(run))
        frame['label_status'] = label_status(run, frame)
        frame['usable_join'] = record['usable_join']
        frame['geometry_id'] = record['geometry_id']
        signatures = family_signatures(frame)
        frame['family_hi'], frame['family_lo'] = signatures[:, 0], signatures[:, 1]
        record['seconds'] = time.perf_counter() - started
        return frame, record, geom, schema


def plan(args):
    tasks, files = [], []
    for run in RUN_LABELS:
        paths = sorted((args.input / str(run)).glob('*.root'), key=lambda p: int(p.stem.split('_')[1]))
        if args.pilot:
            paths = paths[:1]
        files.extend(str(p) for p in paths)
        tasks.extend([[str(p) for p in paths[i:i + args.files_per_task]]
                      for i in range(0, len(paths), args.files_per_task)])
    write_json(args.output, {'created_utc': datetime.now(timezone.utc).isoformat(),
                            'tasks': tasks, 'files': len(files), 'input': str(args.input)})
    print(f'{len(files)} files, {len(tasks)} tasks -> {args.output}')


def scan(args):
    paths = json.loads(args.plan.read_text())['tasks'][args.task]
    args.output.mkdir(parents=True, exist_ok=True)
    tables, records, geometries, schemas = [], [], {}, {}
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.cpus) as executor:
        for path in paths:
            try:
                frame, record, geom, schema = scan_file(path, executor)
                tables.append(frame)
                geometries[record['geometry_id']] = geom
                schemas[record['schema_id']] = schema
                records.append(record)
            except Exception as exc:
                records.append({'path': path, 'error': f'{type(exc).__name__}: {exc}'})
    if tables:
        output = args.output / f'events_{args.task:04d}.parquet'
        pd.concat(tables, ignore_index=True).to_parquet(output, index=False, compression=COMPRESSION)
    report = {'task': args.task, 'files': records, 'geometries': geometries, 'schemas': schemas,
              'seconds': time.perf_counter() - start,
              'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    write_json(args.output / f'audit_{args.task:04d}.json', report)
    print(json.dumps({'task': args.task, 'files': len(records), 'errors': sum('error' in r for r in records),
                      'events': sum(len(t) for t in tables), 'seconds': report['seconds']}))
    if any('error' in r for r in records):
        raise SystemExit(1)


def merge(args):
    # This reduction is a batch task too; peak memory is reported in summary.json.
    start = time.perf_counter()
    expected = json.loads(args.plan.read_text())['tasks']
    reports = [json.loads((args.input / f'audit_{i:04d}.json').read_text()) for i in range(len(expected))]
    records = [r for report in reports for r in report['files']]
    expected_paths = [path for task in expected for path in task]
    actual_paths = [r['path'] for r in records]
    if len(set(actual_paths)) != len(actual_paths) or set(actual_paths) != set(expected_paths):
        raise ValueError('audits do not cover the planned files exactly once')
    if any('error' in r for r in records):
        raise ValueError('file errors remain; inspect audits and resubmit failed tasks')
    tables = [pd.read_parquet(args.input / f'events_{i:04d}.parquet') for i in range(len(expected))]
    frame = pd.concat(tables, ignore_index=True)
    del tables
    if frame.duplicated(['run', 'chunk', 'row']).any():
        raise ValueError('duplicate observation identity')
    if sum(r['events'] for r in records) != len(frame):
        raise ValueError('file audit and scalar manifest row counts differ')
    # Same signature always shares one partition across runs/conditions. Ratios
    # are probabilistic within strata, checked below, not forced by row order.
    frame['split'] = split_names(frame.family_hi, frame.family_lo)
    family = frame.groupby(['family_hi', 'family_lo'], sort=False, observed=True)
    sizes = family.size()
    duplicate = sizes[sizes > 1]
    if family.split.nunique().max() != 1:
        raise ValueError('signature leakage across partitions')
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output / 'events.parquet', index=False, compression=COMPRESSION)
    frame.loc[frame.label_status != 'consistent'].to_parquet(args.output / 'label_exceptions.parquet', index=False, compression=COMPRESSION)
    if len(duplicate):
        duplicate.rename('observations').to_frame().to_parquet(args.output / 'repeated_signatures.parquet', compression=COMPRESSION)
    geometries = {k: v for r in reports for k, v in r['geometries'].items()}
    schemas = {k: v for r in reports for k, v in r['schemas'].items()}
    write_json(args.output / 'geometries.json', geometries)
    write_json(args.output / 'schemas.json', schemas)
    write_json(args.output / 'files.json', records)
    summaries = {}
    for run, group in frame.groupby('run', sort=True):
        chunks = sorted(group.chunk.unique().tolist())
        summaries[str(run)] = {
            'label': RUN_LABELS[run], 'files': len(chunks), 'events': len(group),
            'missing_chunks': sorted(set(range(max(chunks) + 1)) - set(chunks)),
            'labels': group.label_status.value_counts().to_dict(),
            'initPDG': {str(k): int(v) for k, v in group.initPDG.value_counts().items()},
            'interaction_scattering_counts': group.groupby(['intType', 'scatteringType']).size().rename('count').reset_index().to_dict('records'),
            'splits': group.groupby(['label_status', 'split']).size().rename('count').reset_index().to_dict('records'),
            'join_failures': int((~group.usable_join).sum()),
            'empty': {d: int((group['n_' + d] == 0).sum()) for d in TREES},
            'all_empty': int((group[[f'n_{d}' for d in TREES]] == 0).all(axis=1).sum()),
            'ranges': {key: {str(q): float(value) for q, value in group[key].quantile([0, .5, .9, .95, .99, .999, 1]).items()}
                       for key in ['initE', 'initX', 'initY', 'initZ', 'n_pixel', 'n_bar', 'n_tracker']}}
    summary = {'events': len(frame), 'files': len(records), 'runs': summaries,
               'unique_metadata_event_keys': int(frame.groupby(['run', 'chunk']).evtID.nunique().sum()),
               'files_with_duplicate_event_ids': sum(not r['unique_event_ids'] for r in records),
               'geometry_ids': list(geometries), 'schema_ids': list(schemas),
               'repeated_signature_families': len(duplicate),
               'observations_in_repeated_families': int(duplicate.sum()),
               'cross_run_signature_families': int((family.run.nunique() > 1).sum()),
               'signature_split_leakage': 0, 'source_provenance_status': 'unverified',
               'split_status': 'provisional_signature_grouped_development_only',
               'signature_fields': SIGNATURE, 'seed': 20261111,
               'repaired_empty_pixel_ids': sum(r['alignment']['pixel']['repaired_empty_ids'] for r in records),
               'seconds': time.perf_counter() - start,
               'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    write_json(args.output / 'summary.json', summary)
    pd.DataFrame([{'run': int(run), 'generated': g['events'], 'join_usable': g['events']-g['join_failures'],
                   'metadata_consistent': g['labels'].get('consistent', 0),
                   'all_empty_raw': g['all_empty'], 'training_response_cut': False,
                   'training_fiducial_cut': False} for run, g in summaries.items()]).to_csv(args.output / 'cutflow.csv', index=False)
    plot_inventory(frame, args.output)
    print(json.dumps({'events': len(frame), 'files': len(records), 'output': str(args.output)}))


def plot_inventory(frame, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for run, group in frame.groupby('run'):
        for ax, key, title in zip(axes.flat, ['initE', 'initX', 'initY', 'initZ', 'n_pixel', 'n_bar'],
                                 ['Incident energy [GeV]', 'True vertex x [mm]', 'True vertex y [mm]',
                                  'True vertex z [mm]', 'Native pixel records', 'Scintillator track records']):
            values = group[key].to_numpy()
            if key == 'initE':
                values = np.log10(np.maximum(values / 1000, 1e-6))
                title = 'log10 incident energy [GeV]'
            elif key.startswith('n_'):
                values = np.log10(1 + values)
                title = 'log10(1 + ' + title.lower() + ')'
            ax.hist(values, bins=70, histtype='step', density=True, label=RUN_LABELS[run])
            ax.set(xlabel=title, ylabel='Density')
    axes[0, 0].legend()
    fig.savefig(output / 'data_ranges.png', dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('plan')
    p.add_argument('--input', type=Path, default=INPUT_ROOT)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--files-per-task', type=int, default=320)
    p.add_argument('--pilot', action='store_true')
    p = sub.add_parser('scan')
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--task', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cpus', type=int, default=2)
    p = sub.add_parser('merge')
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cpus', type=int, default=8)
    args = parser.parse_args()
    pa.set_cpu_count(getattr(args, 'cpus', 2))
    pa.set_io_thread_count(getattr(args, 'cpus', 2))
    {'plan': plan, 'scan': scan, 'merge': merge}[args.command](args)


if __name__ == '__main__':
    main()
