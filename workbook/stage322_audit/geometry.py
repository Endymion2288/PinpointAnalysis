"""Independent placement, ROOT hit and serialized Clean v2 geometry checks."""
import argparse
import csv
import json
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import uproot


def placements():
    """Replay the volume stack in DetectorConstruction.cc and the production macro."""
    plates, pixel, bars = [], [], []
    z = -3000.0

    def pinpoint(copy):
        nonlocal z
        pixel.append(z + 8.005)
        plates.extend([(z + 15.51, 'PinpointBlock', copy), (z + 25.51, 'PinpointBlock', copy)])
        bars.append((z + 20.51, 0))
        z += 30.01

    for copy in range(6):
        pinpoint(copy)
    for copy in range(7):
        for _ in range(6):
            for horizontal in (0, 1):
                plates.append((z + 2.5, 'FortuneBlock', copy))
                bars.append((z + 7.5, horizontal))
                z += 10
        if copy < 6:
            pinpoint(6 + copy)
    pixel.extend(z + 50 + (np.arange(3) + 0.5) * 30.01)
    return plates, np.asarray(pixel), np.asarray(bars), z


def transport_factor(z0, z):
    # Integrate (z-u) across the measured G4 bore z interval for a unit curvature.
    result = np.zeros_like(z, dtype=float)
    for low, high in ((-1565.3, -65.3), (137.4, 1137.4), (1337.4, 2337.4)):
        lower = max(low, z0)
        upper = np.maximum(lower, np.minimum(z, high))
        result += (upper - lower) * (z - (upper + lower) / 2)
    return result


def inspect_probes(out, pixel, bars, xpos, ypos):
    probes = {}
    for path in sorted(out.glob('probe_*.root')):
        root = uproot.open(path)
        event = 'event_id'
        pix = root['Hits/pixelHits'].arrays([event, 'trackID', 'layerID', 'rowID', 'colID'])
        faser = root['Hits/faserHits'].arrays([event, 'trackID', 'trackerID', 'x', 'y', 'z', 'px', 'py', 'pz'])
        scint = root['Hits/scintHits'].arrays([event, 'panelID', 'isHorizontal', 'rowID', 'colID', 'nPE', 'hitTime', 'trackID'])
        p0 = (pix.trackID == 1) & (pix.layerID == 0)
        x = int(path.name.split('_')[1])
        xres = xpos[ak.to_numpy(ak.flatten(pix.colID[p0]))] - x
        yres = ypos[ak.to_numpy(ak.flatten(pix.rowID[p0]))] + 10
        faser = ak.zip({k: faser[k] for k in faser.fields if k != event})
        scint = ak.zip({k: scint[k] for k in scint.fields if k != event})
        s_max, root_times, errors, sagitta = [], [], [], []
        orientation_errors = 0
        z_st = np.array([47.4, 1237.4, 2427.4])
        # Extract a second difference of the integration kernel, independent of fitted slope.
        geom_factor = np.dot([-0.5, 1, -0.5], transport_factor(-3010, z_st))
        charge = 1 if 'mu+' in path.name else -1
        for hits, sensor in zip(faser, scint):
            y = {}
            p = {}
            for hit in hits:
                if hit.trackID == 1 and hit.trackerID not in y:
                    y[hit.trackerID] = hit.y
                    p[hit.trackerID] = np.linalg.norm([hit.px, hit.py, hit.pz]) / 1000
            if len(y) == 3:
                s = y[1] - (y[0] + y[2]) / 2
                p_fit = 0.000299792458 * 0.57 * geom_factor * charge / s
                errors.append(float((p_fit - p[0]) / p[0]))
                sagitta.append(float(s))
            pe = {}
            for hit in sensor:
                key = (hit.panelID, hit.isHorizontal, hit.rowID if hit.isHorizontal else hit.colID)
                pe[key] = pe.get(key, 0) + hit.nPE
                orientation_errors += int(bars[hit.panelID, 1] != hit.isHorizontal)
                if hit.nPE == 0 and hit.hitTime >= 0:
                    root_times.append(float(hit.hitTime))
            s_max.extend(pe.values())
        probes[path.stem] = {
            'primary_p0_events': int(np.count_nonzero(np.asarray(ak.sum(p0, axis=1)))),
            'p0_max_dx_mm': float(np.max(np.abs(xres))),
            'p0_max_dy_mm': float(np.max(np.abs(yres))),
            'tracker_triplets': len(errors),
            'sagitta_mm': sagitta,
            'p_relative_error': errors,
            'scint_orientation_errors': orientation_errors,
            'bar_max_pe': int(max(s_max)),
            'zero_pe_track_records_with_timestamp': len(root_times),
            'sagitta_kernel_mm2': float(geom_factor),
        }
    return probes


def inspect_clean_readout(root):
    output = {}
    for shard in ('080', '125'):
        table = pq.ParquetFile(root / f'data/shards/{shard}/observations.parquet')
        channels = table.read(columns=['bar'])['bar'].combine_chunks()
        pe = channels.field('pe_proxy').flatten().to_numpy()
        time = channels.field('hit_time_ns').flatten().to_numpy()
        output[shard] = {
            'records': len(pe),
            'has_edep_proxy': 'edep_proxy_MeV' in [field.name for field in channels.type],
            'max_pe': int(pe.max()),
            'count_above_double_saturation_cap': int(np.count_nonzero(pe > 14400 * (1 - np.e ** -1))),
            'count_at_cell_limit': int(np.count_nonzero(pe == 14400)),
            'count_above_threshold': int(np.count_nonzero(pe >= 2.5)),
            'zero_pe_channels_with_nonnegative_time': int(np.count_nonzero((pe == 0) & (time >= 0))),
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('clean', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    g = json.loads((args.clean / 'geometry.json').read_text())
    plates, pixel, bars, end = placements()
    wz = np.array([record[0] for record in plates])
    xpos = -133 + (np.arange(int(266 / .0208)) + .5) * .0208
    ypos = -110 + (np.arange(int(196 / .0228)) + .5) * .0228
    references = {
        'pixel_Xpos': xpos, 'pixel_Ypos': ypos, 'pixel_Zpos': pixel,
        'scint_bar_Xcenter': -133 + (np.arange(40) + .5) * 10.5,
        'scint_bar_Ycenter': -222 + (np.arange(40) + .5) * 10.5,
        'scint_Zpos': bars[:, 0], 'tungsten_Zpos': wz,
    }
    live = uproot.open(args.output / 'probe_0_mu-.root')['geometry'].arrays(library='ak')[0].to_list()
    residual = {f'{origin}:{key}': float(np.max(np.abs(expected - array[key])))
                for origin, array in [('live_root', live), ('campaign_json', g)]
                for key, expected in references.items()}
    plan = pd.read_parquet(args.clean / 'planned_manifest.parquet')
    slab = plan.slab_id.to_numpy()
    # W centers from the 77 mm macro shift, (+5,+5) mm W corner shift, and -12 mm stack shift.
    x, y, z = (plan[k].to_numpy() for k in ('initX', 'initY', 'initZ'))
    dz = np.abs(z - wz[slab])
    outside = (np.abs(x - 82) > 215) | (np.abs(y + 7) > 215) | (dz > 2.5)
    expect = np.array([row[1] for row in plates])[slab]
    table = pq.ParquetFile(args.clean / 'data/shards/125/observations.parquet')
    pixel_flat = table.read_row_group(0, columns=['pixel'])['pixel'].combine_chunks()
    ids = pixel_flat.field('channel_id').flatten().to_numpy()
    xyz = pixel_flat.field('xyz_mm').flatten().flatten().to_numpy().reshape(-1, 3)
    layer, rem = np.divmod(ids, len(xpos) * len(ypos))
    row, col = np.divmod(rem, len(xpos))
    xyz_expected = np.column_stack([xpos[col], ypos[row], pixel[layer]])
    weights = plan.geom_weight.to_numpy()
    full_area = 430 ** 2
    inset_fraction = (429 / 430) ** 2 * 4.9 / 5
    result = {
        'max_geometry_residual_mm': residual,
        'stack_end_mm': end,
        'w_total_kg_rho_19p3': 108 * full_area * 5 * 19.3e-6,
        'w_generated_support_fraction': inset_fraction,
        'max_vertex_dz_mm': float(dz.max()),
        'vertex_outside_count': int(outside.sum()),
        'module_mismatch_count': int(np.count_nonzero(expect != plan.module_type.to_numpy())),
        'module_weighted_pinpoint_fraction': float(weights[expect == 'PinpointBlock'].sum() / weights.sum()),
        'geometry_weight_effective_count': float(weights.sum() ** 2 / np.dot(weights, weights)),
        'duplicate_family_keys': int(plan.duplicated(['family_hi', 'family_lo']).sum()),
        'duplicate_event_keys': int(plan.duplicated(['run', 'chunk', 'row']).sum()),
        'serialized_hit_count': int(len(ids)),
        'serialized_xyz_max_residual_mm': float(np.max(np.abs(xyz_expected - xyz))),
        'live_probes': inspect_probes(args.output, pixel, bars, xpos, ypos),
        'clean_readout': inspect_clean_readout(args.clean),
    }
    (args.output / 'geometry_closure.json').write_text(json.dumps(result, indent=2))
    with (args.output / 'placement_table.csv').open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(['slab', 'z_mm', 'module', 'copy'])
        writer.writerows((i, *record) for i, record in enumerate(plates))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
