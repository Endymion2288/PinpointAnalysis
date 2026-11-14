"""Measurement-defined containment and shower-axis proxies on validation only."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.flavour.datasets import NativeShard
from analysis.flavour.s2.train import save_parquet


def extract_shard(arguments):
    root, task = arguments
    paths = sorted(p for p in (root/'shards').iterdir() if (p/'observations.parquet').exists())
    path = paths[task]
    metadata = pd.read_parquet(path/'provenance.parquet')
    eligible = pd.read_parquet(root/'catalog.parquet', filters=[('split', '=', 'validation')], columns=['run', 'chunk', 'row'])
    keys = set(map(tuple, eligible.to_numpy()))
    selected = [i for i, key in enumerate(metadata[['run', 'chunk', 'row']].itertuples(index=False, name=None)) if key in keys]
    if not selected:
        return []
    geometry = json.loads(Path(json.loads((root/'plan.json').read_text())['geometry']).read_text())
    native = NativeShard(path)
    result = []
    for index in selected:
        meta = metadata.iloc[index]
        observation = native[index]['observations']
        xyz = np.asarray(observation['pixel']['xyz_mm']).reshape(-1, 3)
        b = np.asarray(observation['bar']['bounds_mm']).reshape(-1, 5)
        energy = np.asarray(observation['bar']['edep_proxy_MeV'])
        h = np.asarray(observation['bar']['horizontal'])
        row = {k: int(meta[k]) for k in ['run', 'chunk', 'row']}
        if len(xyz):
            limits = [[min(geometry['pixel_Xpos']), max(geometry['pixel_Xpos'])], [min(geometry['pixel_Ypos']), max(geometry['pixel_Ypos'])]]
            edge = np.min(np.stack([xyz[:, 0]-limits[0][0], limits[0][1]-xyz[:, 0], xyz[:, 1]-limits[1][0], limits[1][1]-xyz[:, 1]]), axis=0)
            pixel_edge = float(np.mean(edge < 1.))
            pixel_end = float(np.mean(xyz[:, 2] > max(geometry['pixel_Zpos'])-100.))
            z = np.unique(xyz[:, 2])
            centers = np.stack([xyz[xyz[:, 2] == value, :2].mean(0) for value in z])
            if len(z) >= 6:
                n = len(z)//2
                slopes = [np.linalg.lstsq(np.column_stack([np.ones(len(zz)), zz-zz.mean()]), xy, rcond=None)[0][1]
                          for zz, xy in [(z[:n], centers[:n]), (z[n:], centers[n:])]]
                change = float(np.linalg.norm(slopes[0]-slopes[1]))
            else:
                change = None
        else:
            pixel_edge = pixel_end = change = None
        if len(b) and energy.sum() > 0:
            center_x, center_y = np.mean(geometry['scint_bar_Xcenter']), np.mean(geometry['scint_bar_Ycenter'])
            coordinate = np.where(h, (b[:, 2]+b[:, 3])/2-center_y, (b[:, 0]+b[:, 1])/2-center_x)
            half_extent = np.where(h, geometry['scint_detector_height']/2, geometry['scint_detector_width']/2)
            bar_width = np.where(h, geometry['scint_bar_height'], geometry['scint_bar_width'])
            edge = half_extent-np.abs(coordinate) < bar_width
            bar_edge = float(energy[edge].sum()/energy.sum())
            bar_end = float(energy[b[:, 4] > max(geometry['scint_Zpos'])-100.].sum()/energy.sum())
            contained = bar_edge <= .01 and bar_end <= .05 and (pixel_edge is None or pixel_edge <= .01)
        else:
            bar_edge = bar_end = contained = None
        row.update(pixel_edge_fraction=pixel_edge, pixel_end_fraction=pixel_end,
                   bar_edge_signal_fraction=bar_edge, bar_end_signal_fraction=bar_end,
                   response_contained_proxy=contained, response_containment_valid=contained is not None,
                   centroid_slope_change=change)
        result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    paths = list((args.root/'shards').iterdir())
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = [row for part in executor.map(extract_shard, [(args.root, i) for i in range(len(paths))]) for row in part]
    frame = pd.DataFrame(rows)
    if len(frame) != 10000 or frame.duplicated(['run', 'chunk', 'row']).any():
        raise ValueError('measurement diagnostics lack complete validation keys')
    save_parquet(frame, args.root/'measurement_diagnostics.parquet')
    summary = {'events': len(frame), 'response_containment_valid': int(frame.response_containment_valid.sum()),
               'contained_proxy': int(frame.response_contained_proxy.eq(True).sum()),
               'definition': 'bar measured edge energy <=1%, last 100mm bar energy <=5%, pixel edge occupancy <=1% if present',
               'scope': 'measurement leakage proxy; no claim of true shower containment; centroid slope change is not a fitted particle kink',
               'calibration_test_used': False}
    (args.root/'measurement_diagnostics.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
