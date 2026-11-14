"""Observation-only reconstruction quality and visible leakage indicators."""
import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd
import pyarrow as pa

from analysis.flavour.datasets import NativeShard
from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.s5.objects import reconstruct
from analysis.flavour.s6.prepare import KEY, require_freeze


def summarize(observation, geometry):
    reco = reconstruct(observation, geometry)
    pixel, bar = observation['pixel'], observation['bar']
    xyz = np.asarray(pixel['xyz_mm']).reshape(-1, 3)
    b = np.asarray(bar['bounds_mm']).reshape(-1, 5)
    energy = np.asarray(bar['edep_proxy_MeV'])
    nx, ny = len(geometry['pixel_Xpos']), len(geometry['pixel_Ypos'])
    planes = np.asarray(pixel['channel_id'], dtype='int64')//(nx*ny)
    xlim, ylim = [np.asarray(geometry[f'pixel_{a}pos'])[[0, -1]] for a in 'XY']
    edge = ((xyz[:, 0] <= xlim[0]+1) | (xyz[:, 0] >= xlim[1]-1) |
            (xyz[:, 1] <= ylim[0]+1) | (xyz[:, 1] >= ylim[1]-1))
    last = planes == len(geometry['pixel_Zpos'])-1
    bars_edge = (np.asarray(bar['bar']) == 0) | (np.asarray(bar['bar']) == len(geometry['scint_bar_Xcenter'])-1)
    last_panel = np.asarray(bar['panel']) == len(geometry['scint_Zpos'])-1
    v = reco['vertex']
    return {'clusters': len(reco['clusters']), 'pixel_segments': len(reco['tracks']),
        'tracker_associations': len(reco['tracker_tracks']), 'tracker_stations': len(np.unique(observation['tracker_hypothesis']['station'])),
        'pixel_planes': len(np.unique(planes)), 'vertex_candidate': v is not None,
        'vertex_candidate_count': reco['vertex_count'], 'vertex_dca_mm': v['dca'] if v else None,
        'vertex_sigma_z_mm': v['sigma_z'] if v else None,
        'pixel_edge_fraction': float(edge.mean()) if len(edge) else 0.,
        'pixel_last_plane_fraction': float(last.mean()) if len(last) else 0.,
        'bar_edge_energy_fraction': float(energy[bars_edge].sum()/energy.sum()) if energy.sum() else 0.,
        'bar_last_panel_energy_fraction': float(energy[last_panel].sum()/energy.sum()) if energy.sum() else 0.,
        'bar_energy_MeV': float(energy.sum())}


def task(args):
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    require_freeze(args.root, 'test')
    started = time.monotonic()
    plan = json.loads((args.root/'plan.json').read_text())
    spec = plan['tasks'][args.task]
    path = args.root/'shards'/f"{spec['split']}_{spec['shard']:04d}"
    geometry = json.loads(Path(plan['geometry']).read_text())
    metadata = pd.read_parquet(path/'provenance.parquet')
    native = NativeShard(path)
    rows = []
    count = min(len(native), args.limit) if args.limit else len(native)
    for i in range(count):
        rows.append({**metadata.iloc[i][KEY].to_dict(), **summarize(native[i]['observations'], geometry)})
    destination = args.root/'evaluation/diagnostics'/f'{args.task:04d}'
    if args.limit:
        destination = args.root/'audit'/f'diagnostic_pilot_{args.task:04d}'
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(destination/'quality.parquet', index=False, compression=COMPRESSION)
    write_json(destination/'summary.json', {'status': 'complete', 'events': len(rows),
        'seconds': time.monotonic()-started, 'peak_RSS_MiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})


def finalize(args):
    plan = json.loads((args.root/'plan.json').read_text())
    parts = [args.root/'evaluation/diagnostics'/f'{i:04d}/quality.parquet'
        for i, t in enumerate(plan['tasks']) if t['split'] == 'test']
    frame = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    expected = pd.read_parquet(args.root/'manifest.parquet', filters=[('split', '=', 'test')])
    if not np.array_equal(frame.sort_values(KEY)[KEY].to_numpy(), expected.sort_values(KEY)[KEY].to_numpy()):
        raise ValueError('diagnostic keys differ from test')
    frame.to_parquet(args.root/'evaluation/quality.parquet', index=False, compression=COMPRESSION)


def worker(args):
    plan = json.loads((args.root/'plan.json').read_text())
    specs = [(i, s) for i, s in enumerate(plan['tasks']) if s['split'] == 'test']
    claims = args.root/'runtime/diagnostic_claims'
    claims.mkdir(exist_ok=True)
    start = time.monotonic()
    while time.monotonic()-start < 21600:
        incomplete, chosen = False, None
        for i, spec in specs:
            if (args.root/'evaluation/diagnostics'/f'{i:04d}/summary.json').exists():
                continue
            incomplete = True
            if not (args.root/'shards'/f"test_{spec['shard']:04d}/summary.json").exists():
                continue
            claim = claims/f'{i:04d}'
            try:
                claim.mkdir()
            except FileExistsError:
                continue
            chosen = i
            break
        if not incomplete:
            break
        if chosen is None:
            time.sleep(20)
            continue
        try:
            args.task = chosen
            task(args)
            print(json.dumps({'diagnostic_task': chosen, 'status': 'complete'}), flush=True)
        finally:
            claim.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['task', 'finalize', 'worker'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--task', type=int, default=227)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == '__main__':
    main()
