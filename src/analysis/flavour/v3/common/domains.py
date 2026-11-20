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

# Mean bar-center photoelectron calibration constant (BC408: 10000/MeV * 0.10 * 0.40 * exp(-21 cm / 380 cm))
PE_PER_MEV_CENTER_CAL = 10000.0 * 0.10 * 0.40 * float(np.exp(-21.0 / 380.0))  # 378.514 PE / MeV
DEFAULT_BAR_THRESHOLD_PE = 2.5
DEFAULT_SIPM_NPIX = 14400.0


def validate_finite_observation(observation: dict, *, check_edep: bool = True) -> None:
    """Raise ValueError immediately if any sensor coordinate or readout value is NaN/Inf (P3-2)."""
    pxyz = np.asarray(observation['pixel']['xyz_mm'], dtype=np.float64)
    if pxyz.size and not np.isfinite(pxyz).all():
        raise ValueError('Non-finite pixel xyz_mm coordinate in observation')
    bar = observation['bar']
    bb = np.asarray(bar['bounds_mm'], dtype=np.float64)
    if bb.size and not np.isfinite(bb).all():
        raise ValueError('Non-finite scintillator bounds_mm in observation')
    bpe = np.asarray(bar['pe_proxy'], dtype=np.float64)
    if bpe.size and not np.isfinite(bpe).all():
        raise ValueError('Non-finite scintillator pe_proxy in observation')
    if check_edep and 'edep_proxy_MeV' in bar:
        be = np.asarray(bar['edep_proxy_MeV'], dtype=np.float64)
        if be.size and not np.isfinite(be).all():
            raise ValueError('Non-finite scintillator edep_proxy_MeV in observation')
    trk = observation['tracker_hypothesis']
    txy = np.asarray(trk['xy_mm'], dtype=np.float64)
    if txy.size and not np.isfinite(txy).all():
        raise ValueError('Non-finite tracker xy_mm coordinate in observation')
    if 'sigma_xy_mm' in trk:
        tsig = np.asarray(trk['sigma_xy_mm'], dtype=np.float64)
        if tsig.size and not np.isfinite(tsig).all():
            raise ValueError('Non-finite tracker sigma_xy_mm in observation')


def calibrated_bar_channels(
    bar: dict,
    config: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract active bar mask, calibrated energy proxy (MeV), and PE signal.

    When config['pe_only'] is True (resolving Audit P1-1 & Stage 320 double-saturation P1):
      - `edep_proxy_MeV` is NEVER read or exposed (pure experimental PE observable only);
      - Canonical Pinpoint_G4 (`ScintSD.cc:321-354`) ALREADY applies 14,400-cell SiPM saturation,
        so `apply_sipm_saturation` defaults to False (set True ONLY when emulating SiPM saturation
        on legacy unsaturated pre-Stage315 data);
      - A physical photoelectron threshold (`bar_threshold_pe`, default 2.5 PE) filters sub-threshold hits;
      - Energy proxy is derived purely from single-response inverse SiPM calibration
        (`invert_sipm_saturation`, default True when `sipm_npix > 0`).
    """
    cfg = config or {}
    pe_only = bool(cfg.get('pe_only', False))
    bpe_raw = np.asarray(bar['pe_proxy'], dtype=np.float64)
    n_raw = len(bpe_raw)
    if n_raw == 0:
        empty_b = np.zeros(0, dtype=bool)
        empty_f = np.empty(0, dtype=np.float64)
        return empty_b, empty_f, empty_f

    if not pe_only:
        if 'edep_proxy_MeV' not in bar:
            raise ValueError('edep_proxy_MeV absent from observation; set pe_only=True for pure observable readout')
        be = np.maximum(0.0, np.asarray(bar['edep_proxy_MeV'], dtype=np.float64))
        bpe = np.maximum(0.0, bpe_raw)
        keep = np.ones(n_raw, dtype=bool)
        return keep, be, bpe

    bpe = np.maximum(0.0, bpe_raw)
    sipm_npix = float(cfg.get('sipm_npix', DEFAULT_SIPM_NPIX))
    # Default False: Pinpoint_G4 ScintSD already saturates bar PE once at N_pix=14,400!
    apply_sat = bool(cfg.get('apply_sipm_saturation', False))
    invert_sat = bool(cfg.get('invert_sipm_saturation', True))
    max_occ = float(cfg.get('sipm_invert_max_occupancy', cfg.get('sipm_sat_cap', 0.98)))
    if apply_sat and sipm_npix > 0.0:
        bpe = sipm_npix * (1.0 - np.exp(-bpe / sipm_npix))

    thr_pe = float(cfg.get('bar_threshold_pe', DEFAULT_BAR_THRESHOLD_PE))
    keep = bpe >= thr_pe
    bpe_kept = bpe[keep]
    # Invert the single physical SiPM saturation curve once (capped safely below full occupancy)
    if invert_sat and sipm_npix > 0.0 and len(bpe_kept):
        frac = np.clip(bpe_kept / sipm_npix, 0.0, max_occ)
        pe_lin = -sipm_npix * np.log(1.0 - frac)
    else:
        pe_lin = bpe_kept
    e_cal_mev = pe_lin / PE_PER_MEV_CENTER_CAL
    return keep, e_cal_mev, bpe_kept


# Physical tungsten plate dimensions from Pinpoint_G4/Pinpoint/include/DetectorConstruction.hh:237-246
# W plate is 430x430 mm (grown +10 mm from 420x420 scintillator preserving the (-X,-Y) corner -> +5 mm shift in X,Y)
PINPOINT_W_WIDTH_MM = 430.0
PINPOINT_W_HEIGHT_MM = 430.0
PINPOINT_W_CORNER_SHIFT_X_MM = 5.0
PINPOINT_W_CORNER_SHIFT_Y_MM = 5.0


def pinpoint_slab_catalog(geometry: dict) -> dict:
    """Build exact physical-volume catalog for all tungsten plates matching Pinpoint_G4 DetectorConstruction.cc.

    Every tungsten slab k in 0..N_W-1 is mapped deterministically to its physical parent module
    (`PinpointBlock` vs `FortuneBlock`), G4 block copy number, within-block plate index, exact 3D
    physical box bounds in world coordinates, and nearest downstream silicon pixel plane.
    """
    w_z = np.asarray(geometry['tungsten_Zpos'], dtype=np.float64)
    p_z = np.asarray(geometry['pixel_Zpos'], dtype=np.float64)
    w_t = float(geometry['tungsten_thickness'])
    n_pin_front = int(geometry['nPinpointBlocks'])
    n_for = int(geometry['nFortuneBlocks'])
    n_sl = int(geometry['numScintLayers'])
    n_interm = max(0, n_for - 1)
    n_w = len(w_z)

    expected_w = 2 * n_pin_front + n_for * (2 * n_sl) + 2 * n_interm
    if n_w != expected_w:
        raise ValueError(f'tungsten_Zpos length {n_w} != DetectorConstruction expected {expected_w}')

    module_type: list[str] = []
    block_copy_no: list[int] = []
    plate_in_block: list[int] = []
    is_pinpoint = np.zeros(n_w, dtype=bool)
    is_front_pinpoint = np.zeros(n_w, dtype=bool)
    is_interm_pinpoint = np.zeros(n_w, dtype=bool)

    cursor = 0
    # 1. Front PinpointBlocks: copyNo 0 .. n_pin_front-1 (each has 2 W plates)
    for b in range(n_pin_front):
        for p_local in range(2):
            module_type.append('PinpointBlock')
            block_copy_no.append(b)
            plate_in_block.append(p_local)
            is_pinpoint[cursor] = True
            is_front_pinpoint[cursor] = True
            cursor += 1
    # 2. Interleaved FortuneBlock i (2*n_sl W plates) + intermediate PinpointBlock (n_pin_front+i) (2 W plates)
    for f in range(n_for):
        for p_local in range(2 * n_sl):
            module_type.append('FortuneBlock')
            block_copy_no.append(f)
            plate_in_block.append(p_local)
            cursor += 1
        if f < n_interm:
            pb_copy = n_pin_front + f
            for p_local in range(2):
                module_type.append('PinpointBlock')
                block_copy_no.append(pb_copy)
                plate_in_block.append(p_local)
                is_pinpoint[cursor] = True
                is_interm_pinpoint[cursor] = True
                cursor += 1

    assert cursor == n_w

    # Cross-check against physical G4PVPlacement offsets (+7.505 mm, +17.505 mm from PixelSensor in PinpointBlock)
    # when geometry is clean physical G4 (pixel_Zpos[0] < tungsten_Zpos[0]).
    is_physical_g4 = float(p_z[0]) < float(w_z[0])
    if is_physical_g4:
        n_target_pix = n_pin_front + n_interm
        for idx in np.flatnonzero(is_pinpoint):
            pb_c = block_copy_no[int(idx)]
            pl_c = plate_in_block[int(idx)]
            if pb_c >= n_target_pix:
                raise ValueError(f'PinpointBlock copyNo {pb_c} exceeds target pixel planes {n_target_pix}')
            expected_z = float(p_z[pb_c]) + (7.505 if pl_c == 0 else 17.505)
            if abs(float(w_z[int(idx)]) - expected_z) > 0.05:
                raise ValueError(
                    f'PinpointBlock slab {idx} (copy={pb_c}, plate={pl_c}) z={w_z[int(idx)]:.3f} mm '
                    f'!= DetectorConstruction expected {expected_z:.3f} mm'
                )

    bx = np.asarray(geometry['scint_bar_Xcenter'], dtype=np.float64)
    by = np.asarray(geometry['scint_bar_Ycenter'], dtype=np.float64)
    w_cx = float(bx.mean()) + PINPOINT_W_CORNER_SHIFT_X_MM
    w_cy = float(by.mean()) + PINPOINT_W_CORNER_SHIFT_Y_MM
    w_x_min = w_cx - 0.5 * PINPOINT_W_WIDTH_MM
    w_x_max = w_cx + 0.5 * PINPOINT_W_WIDTH_MM
    w_y_min = w_cy - 0.5 * PINPOINT_W_HEIGHT_MM
    w_y_max = w_cy + 0.5 * PINPOINT_W_HEIGHT_MM

    next_pix = np.full(n_w, -1, dtype=np.int32)
    dz_next_pix = np.full(n_w, np.inf, dtype=np.float64)
    for k in range(n_w):
        ds = np.flatnonzero(p_z > w_z[k])
        if len(ds):
            next_pix[k] = int(ds[0])
            dz_next_pix[k] = float(p_z[ds[0]] - w_z[k])

    return {
        'n_slabs': int(n_w),
        'slab_id': np.arange(n_w, dtype=np.int32),
        'module_type': module_type,
        'block_copy_no': np.asarray(block_copy_no, dtype=np.int32),
        'plate_in_block': np.asarray(plate_in_block, dtype=np.int32),
        'is_pinpoint_slab': is_pinpoint,
        'is_front_pinpoint': is_front_pinpoint,
        'is_intermediate_pinpoint': is_interm_pinpoint,
        'is_fortune_slab': ~is_pinpoint,
        'pinpoint_indices': np.flatnonzero(is_pinpoint).astype(np.int32),
        'front_pinpoint_indices': np.flatnonzero(is_front_pinpoint).astype(np.int32),
        'intermediate_pinpoint_indices': np.flatnonzero(is_interm_pinpoint).astype(np.int32),
        'fortune_indices': np.flatnonzero(~is_pinpoint).astype(np.int32),
        'z_center_mm': w_z,
        'w_thickness_mm': w_t,
        'w_center_x_mm': w_cx,
        'w_center_y_mm': w_cy,
        'w_width_mm': PINPOINT_W_WIDTH_MM,
        'w_height_mm': PINPOINT_W_HEIGHT_MM,
        'w_x_min_mm': w_x_min,
        'w_x_max_mm': w_x_max,
        'w_y_min_mm': w_y_min,
        'w_y_max_mm': w_y_max,
        'next_pixel_plane': next_pix,
        'dz_to_next_pixel_mm': dz_next_pix,
        'is_physical_g4': is_physical_g4,
    }


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
    """Cumulative radiation length X0 from target entrance up to plane z_mm matching DetectorConstruction.cc."""
    w_z = np.asarray(geometry['tungsten_Zpos'], dtype=np.float64)
    w_t = float(geometry['tungsten_thickness'])
    s_z = np.asarray(geometry['scint_Zpos'], dtype=np.float64)
    s_t = float(geometry['scint_thickness'])
    p_z = np.asarray(geometry['pixel_Zpos'], dtype=np.float64)
    p_t = float(geometry['silicon_thickness']) / 1000.0
    n_pin_blocks = int(geometry['nPinpointBlocks']) + max(0, int(geometry['nFortuneBlocks']) - 1)
    pin_z = p_z[:n_pin_blocks]

    if float(p_z[0]) < float(w_z[0]):
        # Exact physical placements inside PinpointBlock relative to PixelSensor center:
        # AlWall_Back (2 mm) at -7.005 mm, AlCoolingPlate (5 mm) at +2.505 mm, AlWall_Front (2 mm) at +21.005 mm
        al_x0 = (
            np.count_nonzero((pin_z - 7.005) < z_mm) * (2.0 / _X0_AL_MM)
            + np.count_nonzero((pin_z + 2.505) < z_mm) * (5.0 / _X0_AL_MM)
            + np.count_nonzero((pin_z + 21.005) < z_mm) * (2.0 / _X0_AL_MM)
        )
    else:
        al_x0 = np.count_nonzero(pin_z < z_mm) * (9.0 / _X0_AL_MM)

    x0 = (
        np.count_nonzero(w_z < z_mm) * (w_t / _X0_W_MM)
        + np.count_nonzero(s_z < z_mm) * (s_t / _X0_SCINT_MM)
        + np.count_nonzero(p_z < z_mm) * (p_t / _X0_SI_MM)
        + al_x0
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
        'z_w0': float(tz[0]),
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
    config: dict | None = None,
) -> dict:
    """Extract 16-dim c_obs and mutually exclusive obs_domain_id purely from observables."""
    if bounds is None:
        bounds = detector_bounds(geometry)
    pe_only = bool((config or {}).get('pe_only', False))
    validate_finite_observation(observation, check_edep=not pe_only)

    pixel = observation['pixel']
    bar = observation['bar']
    tracker = observation['tracker_hypothesis']

    pxyz = np.asarray(pixel['xyz_mm'], dtype=np.float64).reshape(-1, 3)
    pcid = np.asarray(pixel['channel_id'], dtype=np.int64)
    keep_bar, be, _ = calibrated_bar_channels(bar, config)
    bb = np.asarray(bar['bounds_mm'], dtype=np.float64).reshape(-1, 5)[keep_bar]
    bp = np.asarray(bar['panel'], dtype=np.int64)[keep_bar]
    bh = np.asarray(bar['horizontal'], dtype=np.int64)[keep_bar]
    bi = np.asarray(bar['bar'], dtype=np.int64)[keep_bar]
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

    # Geometry-aware upstream veto:
    # - In legacy geometry.json, scint panel 0 (-2995.5 mm) sat upstream of W0 (-2990.5 mm).
    # - In physical Geant4 geometry (P0-2 fixed), pixel plane 0 (-2991.995 mm) sits upstream of W0 (-2984.49 mm),
    #   while scint panel 0 (-2979.49 mm) sits downstream of W0.
    z_w0 = float(bounds.get('z_w0', bounds['z_target_min'] + 2.5))
    scint_p0_upstream = float(bounds['scint_Zpos'][0]) < z_w0
    pix_p0_upstream = float(bounds['pixel_Zpos'][0]) < z_w0
    if scint_p0_upstream:
        p0_edep = float(be[bp == 0].sum()) if has_bar else 0.0
        upstream_veto = (p0_edep > 5.0) and (p0_edep > 0.02 * max(total_edep, 1e-12))
    elif pix_p0_upstream and has_pix:
        p0_planes = pcid // (bounds['nx'] * bounds['ny'])
        p0_hits = int(np.count_nonzero(p0_planes == 0))
        # Physical Geant4 closure (Stage 321): an incoming 50-1000 GeV rock muon leaves
        # exactly 1 hit in 10 um silicon PixelSensor[0] (-2991.995 mm, upstream of W0 -2984.49 mm).
        upstream_veto = p0_hits >= 1
    else:
        upstream_veto = False
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
