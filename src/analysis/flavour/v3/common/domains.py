"""Pure-observation containment features (c_obs) and mutually exclusive calibration domains.

Strictly observation-only: truth vertex/energy fields are never accepted by
compute_observable_containment(). Truth domain labels (pixel_centered,
large_only, outside_large, empty) are joined offline in the CLI diagnostic
only to verify purity and coverage against R0-frozen manifests.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from analysis.flavour.datasets import NativeShard
from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.v2.partitions import FAMILY, KEY, equal_identity_rows


OBS_DOMAIN_NAMES = (
    'D_empty',
    'D_pix_contained',
    'D_scint_contained',
    'D_partial_or_edge',
)
TRUTH_DOMAIN_NAMES = (
    'empty',
    'pixel_centered',
    'large_only',
    'outside_large',
)
COBS_FEATURE_NAMES = (
    'has_pixel',
    'has_bar',
    'has_tracker_any',
    'has_tracker_3station',
    'z_first_active_norm',
    'z_last_active_norm',
    'upstream_veto_active',
    'downstream_tail_energy_frac',
    'transverse_edge_energy_frac',
    'bar_cog_x_norm',
    'bar_cog_y_norm',
    'bar_rms_x_norm',
    'bar_rms_y_norm',
    'log1p_pixel_hits',
    'log1p_bar_channels',
    'log1p_bar_edep_norm',
)

# Radiation lengths in mm (PDG 2024 material properties matching Pinpoint_G4)
_X0_W_MM = 3.504
_X0_SI_MM = 93.70
_X0_SCINT_MM = 425.4
_X0_AL_MM = 88.97


def derive_panel_orientations(geometry: dict) -> np.ndarray:
    """Derive boolean horizontal flag (0=vertical/X, 1=horizontal/Y) for all scint panels."""
    n_pin = int(geometry['nPinpointBlocks'])
    n_for = int(geometry['nFortuneBlocks'])
    n_sl = int(geometry['numScintLayers'])
    n_sp = int(geometry['num_scint_panels_layer'])
    flags: list[int] = [0] * n_pin
    for f in range(n_for):
        for _ in range(n_sl):
            if n_sp >= 1:
                flags.append(0)
            if n_sp >= 2:
                flags.append(1)
        if f < n_for - 1:
            flags.append(0)
    arr = np.asarray(flags, dtype=np.int8)
    if len(arr) != len(geometry['scint_Zpos']):
        raise ValueError(
            f'derived panel count {len(arr)} != geometry scint_Zpos {len(geometry["scint_Zpos"])}'
        )
    return arr


def cumulative_radiation_length(z_mm: float, geometry: dict) -> float:
    """Approximate cumulative radiation length X0 from target entrance up to plane z_mm."""
    w_z = np.asarray(geometry['tungsten_Zpos'], dtype=np.float64)
    w_t = float(geometry['tungsten_thickness'])
    s_z = np.asarray(geometry['scint_Zpos'], dtype=np.float64)
    s_t = float(geometry['scint_thickness'])
    p_z = np.asarray(geometry['pixel_Zpos'], dtype=np.float64)
    p_t = float(geometry['silicon_thickness']) / 1000.0
    n_pin_blocks = int(geometry['nPinpointBlocks']) + max(0, int(geometry['nFortuneBlocks']) - 1)
    pin_z = p_z[:n_pin_blocks]

    x0 = (
        np.count_nonzero(w_z < z_mm) * (w_t / _X0_W_MM)
        + np.count_nonzero(s_z < z_mm) * (s_t / _X0_SCINT_MM)
        + np.count_nonzero(p_z < z_mm) * (p_t / _X0_SI_MM)
        + np.count_nonzero(pin_z < z_mm) * (9.0 / _X0_AL_MM)
    )
    return float(x0)


def detector_bounds(geometry: dict) -> dict:
    """Precompute exact sensor bounds, centers, panel maps, and target depth thresholds."""
    px = np.asarray(geometry['pixel_Xpos'], dtype=np.float64)
    py = np.asarray(geometry['pixel_Ypos'], dtype=np.float64)
    pz = np.asarray(geometry['pixel_Zpos'], dtype=np.float64)
    dx = float(np.median(np.diff(px)))
    dy = float(np.median(np.diff(py)))
    px_min = float(px.min() - dx / 2.0)
    px_max = float(px.max() + dx / 2.0)
    py_min = float(py.min() - dy / 2.0)
    py_max = float(py.max() + dy / 2.0)

    bx = np.asarray(geometry['scint_bar_Xcenter'], dtype=np.float64)
    by = np.asarray(geometry['scint_bar_Ycenter'], dtype=np.float64)
    sz = np.asarray(geometry['scint_Zpos'], dtype=np.float64)
    tz = np.asarray(geometry['tungsten_Zpos'], dtype=np.float64)
    w_t = float(geometry['tungsten_thickness'])
    z_target_min = float(tz.min() - w_t / 2.0)
    z_target_max = float(tz.max() + w_t / 2.0)
    l_target = z_target_max - z_target_min

    panel_is_horiz = derive_panel_orientations(geometry)
    vert_panels = np.flatnonzero(panel_is_horiz == 0)
    horiz_panels = np.flatnonzero(panel_is_horiz == 1)
    panel_to_vert = np.full(len(sz), -1, dtype=np.int32)
    panel_to_horiz = np.full(len(sz), -1, dtype=np.int32)
    panel_to_vert[vert_panels] = np.arange(len(vert_panels), dtype=np.int32)
    panel_to_horiz[horiz_panels] = np.arange(len(horiz_panels), dtype=np.int32)

    return {
        'nx': len(px),
        'ny': len(py),
        'nz': len(pz),
        'dx_mm': dx,
        'dy_mm': dy,
        'px_min': px_min,
        'px_max': px_max,
        'py_min': py_min,
        'py_max': py_max,
        'x_pix_center': 0.5 * (px_min + px_max),
        'y_pix_center': 0.5 * (py_min + py_max),
        'pixel_Zpos': pz,
        'scint_Zpos': sz,
        'scint_bar_Xcenter': bx,
        'scint_bar_Ycenter': by,
        'n_bars': len(bx),
        'x_scint_center': float(bx.mean()),
        'y_scint_center': float(by.mean()),
        'z_target_min': z_target_min,
        'z_target_max': z_target_max,
        'L_target': l_target,
        'z_pinpoint_max': z_target_min + 500.0,
        'z_85': z_target_min + 0.85 * l_target,
        'z_mid': 0.5 * (z_target_min + z_target_max),
        'panel_is_horiz': panel_is_horiz,
        'vert_panels': vert_panels,
        'horiz_panels': horiz_panels,
        'panel_to_vert': panel_to_vert,
        'panel_to_horiz': panel_to_horiz,
        'n_scint_panels': len(sz),
    }


def compute_observable_containment(
    observation: dict,
    geometry: dict,
    bounds: dict | None = None,
) -> dict:
    """Extract 16-dim c_obs and mutually exclusive obs_domain_id purely from observables."""
    if bounds is None:
        bounds = detector_bounds(geometry)

    pixel = observation['pixel']
    bar = observation['bar']
    tracker = observation['tracker_hypothesis']

    pxyz = np.asarray(pixel['xyz_mm'], dtype=np.float64).reshape(-1, 3)
    pcid = np.asarray(pixel['channel_id'], dtype=np.int64)
    bb = np.asarray(bar['bounds_mm'], dtype=np.float64).reshape(-1, 5)
    be = np.asarray(bar['edep_proxy_MeV'], dtype=np.float64)
    bp = np.asarray(bar['panel'], dtype=np.int64)
    bh = np.asarray(bar['horizontal'], dtype=np.int64)
    bi = np.asarray(bar['bar'], dtype=np.int64)
    tr_s = np.asarray(tracker['station'], dtype=np.int64)

    n_pix = len(pcid)
    n_bar = len(be)
    n_trk = len(tr_s)
    has_pix = n_pix > 0
    has_bar = n_bar > 0
    has_trk = n_trk > 0
    has_trk3 = len(np.unique(tr_s)) == 3 if has_trk else False

    z_min_t = bounds['z_target_min']
    l_target = bounds['L_target']
    y_scint_c = bounds['y_scint_center']
    n_scint = bounds['n_scint_panels']
    n_bars = bounds['n_bars']

    c_obs = np.zeros(16, dtype=np.float32)
    c_obs[0] = float(has_pix)
    c_obs[1] = float(has_bar)
    c_obs[2] = float(has_trk)
    c_obs[3] = float(has_trk3)
    c_obs[13] = float(np.log1p(n_pix) / 10.0)
    c_obs[14] = float(np.log1p(n_bar) / 6.0)

    if not has_pix and not has_bar and not has_trk:
        c_obs[4] = 1.5
        c_obs[5] = 1.5
        return {
            'c_obs': c_obs,
            'obs_domain_id': 0,
            'obs_domain_name': OBS_DOMAIN_NAMES[0],
            'cal_start': (0.0, 0.0, 0.0, False),
        }

    # Active Z range
    if has_pix and has_bar:
        z_first = float(min(pxyz[:, 2].min(), bb[:, 4].min()))
        z_last = float(max(pxyz[:, 2].max(), bb[:, 4].max()))
    elif has_pix:
        z_first = float(pxyz[:, 2].min())
        z_last = float(pxyz[:, 2].max())
    elif has_bar:
        z_first = float(bb[:, 4].min())
        z_last = float(bb[:, 4].max())
    else:
        z_first = bounds['z_target_max'] + 0.5 * l_target
        z_last = z_first

    c_obs[4] = float((z_first - z_min_t) / l_target)
    c_obs[5] = float((z_last - z_min_t) / l_target)

    total_edep = float(be.sum()) if has_bar else 0.0
    c_obs[15] = float(np.log1p(total_edep) / 10.0)

    # Panel 0 sits at z=-2995.5 mm upstream of tungsten plate 0 (-2990.5 mm).
    # Soft neutron/photon backscatter into panel 0 is distinguished from true
    # upstream/Al-wall activity by requiring > 5 MeV and > 2% of total bar edep.
    p0_edep = float(be[bp == 0].sum()) if has_bar else 0.0
    upstream_veto = (p0_edep > 5.0) and (p0_edep > 0.02 * max(total_edep, 1e-12))
    c_obs[6] = float(upstream_veto)

    if has_bar and total_edep > 0.0:
        tail_frac = float(be[bp >= (n_scint - 6)].sum() / total_edep)
        edge_mask = (bi <= 1) | (bi >= n_bars - 2)
        edge_frac = float(be[edge_mask].sum() / total_edep)

        bx_c = 0.5 * (bb[:, 0] + bb[:, 1])
        by_c = 0.5 * (bb[:, 2] + bb[:, 3])
        vmask = bh == 0
        hmask = bh == 1
        ve = float(be[vmask].sum()) if vmask.any() else 0.0
        he = float(be[hmask].sum()) if hmask.any() else 0.0

        if ve > 0.0:
            cog_x = float(np.dot(bx_c[vmask], be[vmask]) / ve)
            rms_x = float(np.sqrt(np.dot((bx_c[vmask] - cog_x) ** 2, be[vmask]) / ve))
        else:
            cog_x, rms_x = bounds['x_scint_center'], 0.0
        if he > 0.0:
            cog_y = float(np.dot(by_c[hmask], be[hmask]) / he)
            rms_y = float(np.sqrt(np.dot((by_c[hmask] - cog_y) ** 2, be[hmask]) / he))
        else:
            cog_y, rms_y = y_scint_c, 0.0

        order_z = np.argsort(bb[:, 4], kind='mergesort')
        cum_e = np.cumsum(be[order_z]) / total_edep
        idx_01 = min(len(order_z) - 1, int(np.searchsorted(cum_e, 0.01)))
        idx_05 = min(len(order_z) - 1, int(np.searchsorted(cum_e, 0.05)))
        z_e01 = float(bb[order_z[idx_01], 4])
        z_e05 = float(bb[order_z[idx_05], 4])

        fv = vmask & (bb[:, 4] >= z_e01 - 20.0) & (bb[:, 4] <= z_e05 + 150.0)
        fh = hmask & (bb[:, 4] >= z_e01 - 20.0) & (bb[:, 4] <= z_e05 + 200.0)
        fcog_x = float(np.dot(bx_c[fv], be[fv]) / be[fv].sum()) if fv.any() and be[fv].sum() > 0 else cog_x
        fcog_y = float(np.dot(by_c[fh], be[fh]) / be[fh].sum()) if fh.any() and be[fh].sum() > 0 else cog_y
    else:
        tail_frac = 0.0
        edge_frac = 0.0
        cog_x, cog_y = bounds['x_scint_center'], y_scint_c
        rms_x, rms_y = 0.0, 0.0
        z_e01 = np.inf
        fcog_x = np.nan
        fcog_y = np.nan

    c_obs[7] = tail_frac
    c_obs[8] = edge_frac
    c_obs[9] = float(cog_x / 210.0)
    c_obs[10] = float((cog_y - y_scint_c) / 210.0)
    c_obs[11] = float(rms_x / 100.0)
    c_obs[12] = float(rms_y / 100.0)

    if has_pix:
        planes = pcid // (bounds['nx'] * bounds['ny'])
        p_planes, p_counts = np.unique(planes, return_counts=True)
        z_p1 = float(bounds['pixel_Zpos'][p_planes[0]])
        dense_planes = p_planes[p_counts >= 3]
        z_p3 = float(bounds['pixel_Zpos'][dense_planes[0]]) if len(dense_planes) else z_p1
        f_mask = pxyz[:, 2] <= z_p3 + 20.0
        fpx = float(np.median(pxyz[f_mask, 0]))
        fpy = float(np.median(pxyz[f_mask, 1]))
    else:
        z_p1 = np.inf
        fpx, fpy = np.nan, np.nan

    px_in_min, px_in_max = bounds['px_min'] + 20.0, bounds['px_max'] - 20.0
    py_in_min, py_in_max = bounds['py_min'] + 20.0, bounds['py_max'] - 20.0

    is_pix_contained = (
        has_pix
        and (not upstream_veto)
        and (z_p1 <= bounds['z_pinpoint_max'] + 20.0)
        and ((not has_bar) or (total_edep <= 0.0) or (z_e01 <= bounds['z_pinpoint_max']))
        and (edge_frac < 0.25)
        and (px_in_min <= fpx <= px_in_max)
        and (py_in_min <= fpy <= py_in_max)
        and (np.isnan(fcog_x) or (px_in_min <= fcog_x <= px_in_max))
        and (np.isnan(fcog_y) or (py_in_min <= fcog_y <= py_in_max))
    )

    if is_pix_contained:
        obs_domain_id = 1
    elif has_bar and (not upstream_veto) and (z_first <= bounds['z_85']) and (edge_frac < 0.30):
        obs_domain_id = 2
    else:
        obs_domain_id = 3

    # Pure-observation shower start for downstream spectrometer back-extrapolation
    if has_bar and total_edep > 0.0:
        start_z = min(z_p1, z_e01) if (has_pix and z_e01 - z_p1 < 100.0) else z_e01
        start_x = fpx if (is_pix_contained and np.isfinite(fpx)) else fcog_x
        start_y = fpy if (is_pix_contained and np.isfinite(fpy)) else fcog_y
        cal_start = (float(start_z), float(start_x), float(start_y), True)
    elif has_pix:
        cal_start = (float(z_p1), float(fpx), float(fpy), True)
    else:
        cal_start = (0.0, 0.0, 0.0, False)

    return {
        'c_obs': c_obs,
        'obs_domain_id': obs_domain_id,
        'obs_domain_name': OBS_DOMAIN_NAMES[obs_domain_id],
        'cal_start': cal_start,
    }


def _process_shard(task: tuple[str, dict, dict]) -> pd.DataFrame:
    shard_str, geometry, bounds = task
    shard_path = Path(shard_str)
    prov = pd.read_parquet(shard_path / 'provenance.parquet')
    ns = NativeShard(shard_path, target_columns=('label',))
    n = len(prov)
    c_obs_arr = np.empty((n, 16), dtype=np.float32)
    dom_arr = np.empty(n, dtype=np.int8)
    start_arr = np.empty((n, 3), dtype=np.float32)
    start_valid = np.empty(n, dtype=bool)
    for i in range(n):
        res = compute_observable_containment(ns[i]['observations'], geometry, bounds)
        c_obs_arr[i] = res['c_obs']
        dom_arr[i] = res['obs_domain_id']
        sz, sx, sy, sv = res['cal_start']
        start_arr[i] = (sz, sx, sy)
        start_valid[i] = sv
    out = prov[KEY + FAMILY + ['evtID', 'label', 'split', 'pixel_centered', 'large_scintillator']].copy()
    out['obs_domain_id'] = dom_arr
    out['obs_domain_name'] = [OBS_DOMAIN_NAMES[int(d)] for d in dom_arr]
    out['cal_start_z_mm'] = start_arr[:, 0]
    out['cal_start_x_mm'] = start_arr[:, 1]
    out['cal_start_y_mm'] = start_arr[:, 2]
    out['cal_start_valid'] = start_valid
    for j, name in enumerate(COBS_FEATURE_NAMES):
        out[name] = c_obs_arr[:, j]
    return out


def _truth_domain_series(df: pd.DataFrame) -> np.ndarray:
    empty = (df['has_pixel'] == 0) & (df['has_bar'] == 0) & (df['has_tracker_any'] == 0)
    pc = df['pixel_centered'].to_numpy(dtype=bool)
    ls = df['large_scintillator'].to_numpy(dtype=bool)
    out = np.full(len(df), 'outside_large', dtype=object)
    out[ls] = 'large_only'
    out[pc] = 'pixel_centered'
    out[empty.to_numpy()] = 'empty'
    return out


def _confusion_summary(df: pd.DataFrame) -> dict:
    t_dom = _truth_domain_series(df)
    o_dom = df['obs_domain_id'].to_numpy()
    matrix = np.zeros((4, 4), dtype=np.int64)
    for oi in range(4):
        om = o_dom == oi
        for ti, tname in enumerate(TRUTH_DOMAIN_NAMES):
            matrix[oi, ti] = int(np.count_nonzero(om & (t_dom == tname)))
    d1 = o_dom == 1
    pc = df['pixel_centered'].to_numpy(dtype=bool)
    lo = t_dom == 'large_only'
    ls = df['large_scintillator'].to_numpy(dtype=bool)
    return {
        'events': int(len(df)),
        'obs_domain_names': list(OBS_DOMAIN_NAMES),
        'truth_domain_names': list(TRUTH_DOMAIN_NAMES),
        'matrix_obs_rows_by_truth_cols': matrix.tolist(),
        'obs_domain_counts': {OBS_DOMAIN_NAMES[i]: int(matrix[i].sum()) for i in range(4)},
        'truth_domain_counts': {TRUTH_DOMAIN_NAMES[j]: int(matrix[:, j].sum()) for j in range(4)},
        'D_pix_contained_purity_pixel_centered': float(pc[d1].mean()) if d1.any() else 0.0,
        'D_pix_contained_coverage_pixel_centered': float(d1[pc].mean()) if pc.any() else 0.0,
        'D_scint_contained_large_scintillator_fraction': float(ls[o_dom == 2].mean()) if (o_dom == 2).any() else 0.0,
        'D_scint_contained_coverage_large_only': float((o_dom == 2)[lo].mean()) if lo.any() else 0.0,
    }


def run_w3_01_domains(
    data_root: Path,
    manifests_root: Path,
    output_dir: Path,
    workers: int = 16,
) -> dict:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = json.loads((data_root / 'plan.json').read_text())
    geometry = json.loads(Path(plan['geometry']).read_text())
    bounds = detector_bounds(geometry)
    shards = sorted(p for p in (data_root / 'shards').iterdir() if (p / 'observations.parquet').exists())
    tasks = [(str(p), geometry, bounds) for p in shards]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        frames = list(pool.map(_process_shard, tasks))
    all_df = pd.concat(frames, ignore_index=True)

    train_df = all_df[all_df['split'] == 'train'].sort_values(KEY).reset_index(drop=True)
    tune_df = all_df[all_df['split'] == 'validation'].sort_values(KEY).reset_index(drop=True)

    m300 = pd.read_parquet(manifests_root / 'train_300000.parquet')
    m100 = pd.read_parquet(manifests_root / 'train_100000.parquet')
    mtune = pd.read_parquet(manifests_root / 'tune.parquet')
    if not equal_identity_rows(train_df, m300):
        raise ValueError('Train300k domain table identity mismatch against frozen R0 manifest')
    if not equal_identity_rows(tune_df, mtune):
        raise ValueError('Tune40k domain table identity mismatch against frozen R0 manifest')

    # Verify Train100k exact nesting
    m100_keys = pd.MultiIndex.from_frame(m100[KEY])
    train_keys = pd.MultiIndex.from_frame(train_df[KEY])
    if not m100_keys.isin(train_keys).all():
        raise ValueError('Train100k rows missing from Train300k domain table')

    train_df.to_parquet(output_dir / 'train300k_domains.parquet', index=False, compression=COMPRESSION)
    tune_df.to_parquet(output_dir / 'tune40k_domains.parquet', index=False, compression=COMPRESSION)

    summary = {
        'status': 'complete',
        'seconds': time.perf_counter() - started,
        'workers': workers,
        'shards': len(shards),
        'bounds': {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in bounds.items()
            if k not in ('panel_to_vert', 'panel_to_horiz')
        },
        'train300k': _confusion_summary(train_df),
        'tune40k': _confusion_summary(tune_df),
        'Cal_A_Cut_B_Confirm_payload_read': False,
    }
    write_json(output_dir / 'domain_confusion_4x4.json', summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--data-root',
        type=Path,
        default=Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R2/data'),
    )
    parser.add_argument(
        '--manifests-root',
        type=Path,
        default=Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/manifests'),
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-01'),
    )
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    summary = run_w3_01_domains(args.data_root, args.manifests_root, args.output, args.workers)
    print(json.dumps({
        'status': summary['status'],
        'train300k_pix_purity': summary['train300k']['D_pix_contained_purity_pixel_centered'],
        'train300k_pix_coverage': summary['train300k']['D_pix_contained_coverage_pixel_centered'],
        'tune40k_pix_purity': summary['tune40k']['D_pix_contained_purity_pixel_centered'],
        'tune40k_pix_coverage': summary['tune40k']['D_pix_contained_coverage_pixel_centered'],
        'seconds': summary['seconds'],
    }, indent=2))


if __name__ == '__main__':
    main()
