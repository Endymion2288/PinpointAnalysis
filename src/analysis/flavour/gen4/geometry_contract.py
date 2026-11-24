"""Exact non-coaxial PINPOINT geometry contract, piecewise X0(z), conditional symmetry orbits,
and in-place NC + tau->mu additive surgery with 14,400-cell SiPM inverse-saturation overlay.

All detector constants are parsed directly from `10085_000.root:geometry` / `geometry_mc8588.json`
(matching `10085-10088` PinpointBlock placement: AlWall -> Scint_V -> W_back -> AirGap -> Si -> AlCool -> W_front -> AlWall).
Zero sealed chunks (`chunk_id % 20 >= 16`) and zero deprecated `v6_clean_v3` / `10081-10084` files are ever touched.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from analysis.flavour.nextgen.seal_guard import assert_unsealed_path
from analysis.flavour.v3.common.domains import (
    _X0_AL_MM,
    _X0_SCINT_MM,
    _X0_SI_MM,
    _X0_W_MM,
    detector_bounds,
    pinpoint_slab_catalog,
)
from analysis.flavour.v3.v1_b1pp.spectrometer import (
    B_FIELD_TESLA,
    ENTRY_SLOPE_SAGITTA_COEFF,
    MAGNET0_LENGTH_MM,
    MAGNET0_Z_CENTER_MM,
    MAGNET0_Z_MAX_MM,
    MAGNET0_Z_MIN_MM,
    MAGNET1_Z_MAX_MM,
    MAGNET1_Z_MIN_MM,
    MAGNET2_Z_MAX_MM,
    MAGNET2_Z_MIN_MM,
    MAGNET_BORE_FID_R_MM,
    MAGNET_BORE_R_MM,
    MAGNET_BORE_X_MM,
    MAGNET_BORE_Y_MM,
    SAGITTA_CONST_MEV_MM,
    SAGITTA_INTEGRAL_MM2,
    SIGMA_X_MM,
    SIGMA_Y_MM,
    TRACKER_Z_MM,
)


DEFAULT_GEOMETRY_JSON = Path(
    '/eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage329_gen4_mc8588/cache/geometry_mc8588.json'
)
DEFAULT_GEOMETRY_ROOT = Path(
    '/eos/project-f/fasersim-bonn/public/pinpoint/data/10085/10085_000.root'
)

# Physical constants matching Pinpoint_G4 DetectorConstruction.cc, ScintSD.cc & PixelSD.cc
SPEED_OF_LIGHT_MM_PER_NS = 299.792458
BC408_N_REF = 1.58
WLS_GROUP_VELOCITY_MM_PER_NS = SPEED_OF_LIGHT_MM_PER_NS / BC408_N_REF  # 189.742062... mm/ns
BC408_TAU_SCINT_NS = 2.1
BC408_ATTENUATION_MM = 3800.0
BC408_LIGHT_YIELD_PH_PER_MEV = 10000.0
BC408_TRANSPORT_EFF = 0.10
SIPM_QUANTUM_EFF = 0.40
SIPM_ETA0_PE_PER_MEV = BC408_LIGHT_YIELD_PH_PER_MEV * BC408_TRANSPORT_EFF * SIPM_QUANTUM_EFF  # 400.0 PE/MeV
SIPM_NPIX = 14400.0
SIPM_TIME_THR_PE = 3
SIPM_TTS_NS = 0.15

X0_AIR_MM = 303900.0
RM_W_MM = 9.30
LAMBDA_I_W_MM = 99.46
AIR_CONV_SUPPRESSION = 1.0 / 87000.0  # gamma->e+e- pair production ratio: 6mm Air vs 5mm W

# Pixel sensor physical digitization parameters (PixelSD.cc)
PIXEL_DIFFUSION_SIGMA_UM = 3.0
PIXEL_THR_E = 100.0
PIXEL_NOISE_E = 10.0
PIXEL_MIP_E = 722.0  # ~2.6 keV / 3.6 eV

# Magnetic field deflection constants (B_x = +0.57 T -> Lorentz force in y-z plane)
KAPPA_B_CONST_MEV_MM = 0.299792458 * B_FIELD_TESLA  # 0.17088170106 MeV/(mm)
# IPT (z_ref = -1704.865 mm) -> Magnet0 [-1565.3, -65.3] (L=1500 mm) -> Tracker 1 (z=47.4 mm)
IPT_Z_REF_MM = -1704.865
IPT_X0_FLOOR = 1.0e-4  # Fixed from legacy 0.05 X0 clamp in tracklet_ops.py
MAGNET0_BEND_INTEGRAL_MM2 = (
    0.5 * (MAGNET0_LENGTH_MM ** 2) + MAGNET0_LENGTH_MM * (TRACKER_Z_MM[0] - MAGNET0_Z_MAX_MM)
)  # 1,294,050.0 mm^2
M0_DEFLECTION_INT_MM2 = MAGNET0_BEND_INTEGRAL_MM2
MAGNET0_BEND_CONST_MEV_MM = MAGNET0_BEND_INTEGRAL_MM2 * KAPPA_B_CONST_MEV_MM  # 221,129.47 MeV*mm

# Contract Section I aliases & exact 54-station interleaved topology
X_SIPM_Y = -133.0
Y_SIPM_X = 198.0
S_FRONT: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
S_INTER: tuple[int, ...] = (12, 19, 26, 33, 40, 47)
SI_TO_DOWNSTREAM_SCINT_STATION: tuple[int, ...] = S_FRONT + S_INTER

# Tracker transverse bounds (DetectorConstruction.cc: 250x250 mm centered at (0, +15) mm)
TRACKER_CENTER_X_MM = 0.0
TRACKER_CENTER_Y_MM = 15.0
TRACKER_X_MIN_MM = -125.0
TRACKER_X_MAX_MM = 125.0
TRACKER_Y_MIN_MM = -110.0
TRACKER_Y_MAX_MM = 140.0

# Exact sub-aperture symmetry axes & discrete pitch lattice shift multipliers
SI_SUB_X_AXIS_MM = -1.75        # Center of vertical bar 12 (-127.75 + 12 * 10.5 = -1.75 mm)
SI_SUB_BAR_SUM = 24             # bar_x in 0..24 -> 24 - bar_x
SCINT_SUB_X_AXIS_MM = 77.0      # Scintillator panel X center
SCINT_SUB_BAR_SUM = 39          # bar_x in 0..39 -> 39 - bar_x
COMMON_Y_AXIS_MM = -12.0        # Silicon & Scintillator common Y center
SCINT_Y_BAR_SUM = 39            # bar_y in 0..39 -> 39 - bar_y
SI_REFL_COL_SUM = 12619         # round((-3.50 - 2*px_min)/dx) - 1 -> exact pixel col involution
SI_REFL_ROW_SUM = 8595          # ny - 1 -> exact pixel row involution
LATTICE_SHIFT_PIX_X = 505       # 505 * 0.0208 mm = 10.504 mm (~1 bar pitch 10.5 mm)
LATTICE_SHIFT_PIX_Y = 460       # 460 * 0.0228 mm = 10.488 mm (~1 bar pitch 10.5 mm)

# Material IDs for 1D z-profile
MAT_AIR = 0
MAT_AL = 1
MAT_SI = 2
MAT_W = 3
MAT_SCINT = 4


@dataclass(frozen=True)
class Gen4GeometryContract:
    """Immutable geometry contract parsed from 10085-10088 `geometry` TTree / `geometry_mc8588.json`."""

    raw_geometry: dict[str, Any]
    nx: int
    ny: int
    nz: int
    n_target_planes: int
    n_ipt_planes: int
    dx_mm: float
    dy_mm: float
    px_min: float
    px_max: float
    py_min: float
    py_max: float
    x_pix_center: float
    y_pix_center: float
    pixel_Zpos: np.ndarray            # (15,) float64
    pixel_X0: np.ndarray              # (15,) float64
    n_bars: int
    bar_pitch_mm: float
    bar_width_mm: float
    x_scint_center: float
    y_scint_center: float
    scint_x_min: float
    scint_x_max: float
    scint_y_min: float
    scint_y_max: float
    x_sipm_horiz_mm: float            # -133.0 mm
    y_sipm_vert_mm: float             # +198.0 mm
    bar_x_centers_mm: np.ndarray      # (40,) float64
    bar_y_centers_mm: np.ndarray      # (40,) float64
    n_stations: int                   # 54
    n_doublets: int                   # 42
    station_z_mm: np.ndarray          # (54,) float64
    station_x0: np.ndarray            # (54,) float64
    station_has_y: np.ndarray         # (54,) bool
    s_front_xonly: np.ndarray         # (6,) -> [0,1,2,3,4,5]
    s_inter_xonly: np.ndarray         # (6,) -> [12,19,26,33,40,47]
    s_all_xonly: np.ndarray           # (12,)
    s_stereo: np.ndarray              # (42,)
    station_to_doublet: np.ndarray    # (54,) int64 (-1 if x-only)
    panel_to_station: np.ndarray      # (96,) int64
    panel_to_xslot: np.ndarray        # (96,) int64
    panel_to_yslot: np.ndarray        # (96,) int64
    PANEL_TO_STATION_VIEW: np.ndarray # (96, 2) int64 [station_idx 0..53, view 0:X_vert / 1:Y_horiz]
    STATION_HAS_Y: np.ndarray         # (54,) bool
    pinpoint_scint_stations: np.ndarray     # (12,) int64 -> station index of Scint[p] inside PinpointBlock p
    pinpoint_sand_scint_station: np.ndarray # (12,) int64 -> BackConeSlabPrior sandwich station (-1 at p0=0; p0 at 1..5; pin_scint_st[p0] at 6..11)
    pinpoint_upstream_w_front_z: np.ndarray # (12,) float64 -> upstream W plate center before sandwich/Fortune
    pinpoint_fortune_block_id: np.ndarray   # (12,) int64 -> -1 for Front30, p0-6 for Interm150
    fortune_station_mask: np.ndarray        # (7, 54) bool -> 6 doublet stations per FortuneBlock
    is_mc8588_block_order: bool             # True for 10085-10088 (Scint -> W_back -> Air -> Si -> AlCool -> W_front)
    w_center_x_mm: float              # +82.0 mm
    w_center_y_mm: float              # -7.0 mm
    w_x_min_mm: float                 # -133.0 mm
    w_x_max_mm: float                 # +297.0 mm
    w_y_min_mm: float                 # -222.0 mm
    w_y_max_mm: float                 # +208.0 mm
    tungsten_Zpos: np.ndarray         # (108,) float64
    w_thickness_mm: float             # 5.0 mm
    air_gap_6mm_bounds: np.ndarray    # (12, 2) float64 [z_lo, z_hi] for 6mm air gap upstream of Si[0..11]
    tungsten_bounds: np.ndarray       # (108, 2) float64 [z_lo, z_hi]
    pinpoint_w_back_z: np.ndarray     # (12,) float64
    pinpoint_w_front_z: np.ndarray    # (12,) float64
    x0_knots_z: np.ndarray            # (K,) float64 monotonic z breakpoints for piecewise-linear X0(z)
    x0_knots_val: np.ndarray          # (K,) float64 cumulative X0 at each knot
    material_intervals_z: np.ndarray  # (M, 2) float64
    material_intervals_id: np.ndarray # (M,) int8
    bounds_dict: dict[str, Any]

    def optical_distance_to_sipm(
        self,
        horiz: np.ndarray | int,
        x_mm: np.ndarray | float,
        y_mm: np.ndarray | float,
        s_mir: int = 1,
    ) -> np.ndarray | float:
        """Exact optical distance [0, 420] mm from hit (x, y) to SiPM readout edge.

        Matches ScintSD.cc:250-253:
          - Vertical X bars (`horiz == 0`): SiPM at +Y (world +198.0 mm when s_mir=+1):
              d_X = 210.0 - s_mir * (y + 12.0)
          - Horizontal Y bars (`horiz == 1`): SiPM at -X (world -133.0 mm):
              d_Y = 210.0 + (x - 77.0) = x + 133.0
        """
        if s_mir not in (1, -1):
            raise ValueError(f's_mir must be +1 or -1, got {s_mir}')
        half_l = 0.5 * float(self.raw_geometry['scint_detector_width'])  # 210.0 mm
        d_x_bar = half_l - float(s_mir) * (np.asarray(y_mm, dtype=np.float64) - self.y_scint_center)
        d_y_bar = half_l + (np.asarray(x_mm, dtype=np.float64) - self.x_scint_center)
        h_arr = np.asarray(horiz)
        res = np.where(h_arr == 1, d_y_bar, d_x_bar)
        res = np.clip(res, 0.0, 2.0 * half_l)
        if np.ndim(horiz) == 0 and np.ndim(x_mm) == 0 and np.ndim(y_mm) == 0:
            return float(res)
        return res

    def interp_x0(self, z_mm: np.ndarray | float) -> np.ndarray | float:
        """Exact continuous piecewise-linear cumulative radiation length X0(z) from target entrance."""
        val = np.interp(
            np.asarray(z_mm, dtype=np.float64),
            self.x0_knots_z,
            self.x0_knots_val,
            left=float(self.x0_knots_val[0]),
            right=float(self.x0_knots_val[-1]),
        )
        if np.ndim(z_mm) == 0:
            return float(val)
        return val

    def interp_x0_torch(self, z_mm: torch.Tensor) -> torch.Tensor:
        """Differentiable piecewise-linear X0(z) on CPU or GPU tensors."""
        rz = torch.as_tensor(self.x0_knots_z, dtype=torch.float32, device=z_mm.device)
        rx = torch.as_tensor(self.x0_knots_val, dtype=torch.float32, device=z_mm.device)
        z_c = z_mm.float().clamp(min=float(rz[0].item()), max=float(rz[-1].item()))
        idx_hi = torch.searchsorted(rz, z_c.contiguous()).clamp(min=1, max=len(rz) - 1)
        idx_lo = idx_hi - 1
        z0 = rz[idx_lo]
        z1 = rz[idx_hi]
        x0 = rx[idx_lo]
        x1 = rx[idx_hi]
        frac = (z_c - z0) / (z1 - z0).clamp_min(1e-5)
        return x0 + frac * (x1 - x0)

    def air_gap_6mm_mask(self, z_mm: np.ndarray | float) -> np.ndarray | bool:
        """Return True where z_mm lies inside one of the 12 6mm air gaps upstream of Target Si[0..11]."""
        z_arr = np.asarray(z_mm, dtype=np.float64)
        lo = self.air_gap_6mm_bounds[:, 0]
        hi = self.air_gap_6mm_bounds[:, 1]
        in_gap = ((z_arr[..., None] >= lo) & (z_arr[..., None] <= hi)).any(axis=-1)
        if np.ndim(z_mm) == 0:
            return bool(in_gap)
        return in_gap

    def tungsten_mask(self, z_mm: np.ndarray | float) -> np.ndarray | bool:
        """Return True where z_mm lies inside one of the 108 5mm tungsten plates."""
        z_arr = np.asarray(z_mm, dtype=np.float64)
        lo = self.tungsten_bounds[:, 0]
        hi = self.tungsten_bounds[:, 1]
        in_w = ((z_arr[..., None] >= lo) & (z_arr[..., None] <= hi)).any(axis=-1)
        if np.ndim(z_mm) == 0:
            return bool(in_w)
        return in_w

    def material_conversion_prior(self, z_mm: np.ndarray | float) -> np.ndarray | float:
        """Relative photon conversion probability density at longitudinal coordinate z_mm.

        Returns 1.0 inside G4_W, ~0.04 inside Al/Scint/Si, and 1/87000 inside Air gaps.
        """
        z_arr = np.asarray(z_mm, dtype=np.float64)
        in_w = self.tungsten_mask(z_arr)
        in_gap = self.air_gap_6mm_mask(z_arr)
        # Check Al & Scint intervals
        idx = np.searchsorted(self.material_intervals_z[:, 1], z_arr, side='left')
        idx = np.clip(idx, 0, len(self.material_intervals_id) - 1)
        in_int = (z_arr >= self.material_intervals_z[idx, 0]) & (z_arr <= self.material_intervals_z[idx, 1])
        mat_id = np.where(in_int, self.material_intervals_id[idx], MAT_AIR)
        prob = np.full_like(z_arr, AIR_CONV_SUPPRESSION, dtype=np.float64)
        prob = np.where(mat_id == MAT_SCINT, 0.008, prob)
        prob = np.where(mat_id == MAT_AL, 0.040, prob)
        prob = np.where(mat_id == MAT_SI, 0.037, prob)
        prob = np.where(in_w, 1.0, prob)
        prob = np.where(in_gap, AIR_CONV_SUPPRESSION, prob)
        if np.ndim(z_mm) == 0:
            return float(prob)
        return prob

    def slab_index_at_z(self, z_mm: float) -> int:
        """Return tungsten slab index [0..107] containing z_mm, or -1 if outside tungsten."""
        lo = self.tungsten_bounds[:, 0]
        hi = self.tungsten_bounds[:, 1]
        hits = np.flatnonzero((z_mm >= lo) & (z_mm <= hi))
        return int(hits[0]) if len(hits) else -1

    def upstream_sandwich_stations_for_si(self, p0: int) -> tuple[int, ...]:
        """Scintillator station index(es) immediately upstream of Si[p0] for W_back vs W_front disambiguation.

        - p0 == 0: upstream is W_back[0] only (no upstream sandwich station).
        - Front30 (p0 in 1..5):
            - 10085-10088 layout: Scint_V[p0] at z_si[p0] - 13.505 mm (between W_front[p0-1] and W_back[p0]).
            - Legacy layout: Scint_V[p0-1] at z_si[p0-1] + 12.505 mm.
        - Interm150 (p0 in 6..11): the 6 stereo stations of FortuneBlock[p0 - 6] (+ Scint_V[p0] in 10085-10088).
        """
        if p0 <= 0 or p0 >= self.n_target_planes:
            return ()
        if p0 <= 5:
            return (int(self.pinpoint_sand_scint_station[p0]),)
        f_idx = p0 - 6
        st_in = np.flatnonzero(self.fortune_station_mask[f_idx])
        if self.is_mc8588_block_order:
            return tuple(int(s) for s in st_in.tolist()) + (int(self.pinpoint_scint_stations[p0]),)
        return tuple(int(s) for s in st_in.tolist())


def _build_scint_stations_gen4(geometry: dict[str, Any], bounds: dict[str, Any]) -> dict[str, Any]:
    """Build exact 54-station map fixing v3's static_scint_stations mispairing on 10085-10088.

    In 10085-10088, FortuneBlock group 5's horizontal Y panel sits at -2702.44 mm
    (10.0 mm downstream of its paired vertical X panel at -2712.44 mm), while the next
    PinpointBlock's X-only vertical panel sits only 7.0 mm downstream at -2695.44 mm.
    Requiring 0.0 < (z_horiz - z_vert) <= 15.0 mm guarantees exact pairing:
      S_FRONT = (0..5), S_INTER = (12, 19, 26, 33, 40, 47).
    """
    sz = np.asarray(bounds['scint_Zpos'], dtype=np.float64)
    vp = np.asarray(bounds['vert_panels'], dtype=np.int64)
    hp = np.asarray(bounds['horiz_panels'], dtype=np.int64)

    v_order = np.argsort(sz[vp], kind='mergesort')
    vp_sorted = vp[v_order]
    n_st = len(vp_sorted)
    v_z = sz[vp_sorted]

    h_order = np.argsort(sz[hp], kind='mergesort')
    hp_sorted = hp[h_order]

    station_has_y = np.zeros(n_st, dtype=bool)
    station_h_panel = np.full(n_st, -1, dtype=np.int64)
    station_z = v_z.copy()

    for p_h in hp_sorted:
        hz = float(sz[p_h])
        dz_pos = hz - v_z  # Fortune H panel is strictly downstream (+10.0 mm) of its paired V panel
        valid = (dz_pos > 0.0) & (dz_pos <= 15.0) & (~station_has_y)
        if not np.any(valid):
            raise ValueError(f'Horizontal panel {int(p_h)} at z={hz} failed causal pairing (0 < h_z - v_z <= 15)')
        cand_idx = np.flatnonzero(valid)
        s = int(cand_idx[int(np.argmin(dz_pos[cand_idx]))])
        station_has_y[s] = True
        station_h_panel[s] = int(p_h)
        station_z[s] = 0.5 * (v_z[s] + hz)

    doublet_stations = np.flatnonzero(station_has_y).astype(np.int64)
    station_to_doublet = np.full(n_st, -1, dtype=np.int64)
    station_to_doublet[doublet_stations] = np.arange(len(doublet_stations), dtype=np.int64)

    panel_to_station = np.full(len(sz), -1, dtype=np.int64)
    panel_to_xslot = np.full(len(sz), -1, dtype=np.int64)
    panel_to_yslot = np.full(len(sz), -1, dtype=np.int64)
    for s in range(n_st):
        pv = int(vp_sorted[s])
        panel_to_station[pv] = s
        panel_to_xslot[pv] = s
        ph = int(station_h_panel[s])
        if ph >= 0:
            panel_to_station[ph] = s
            panel_to_yslot[ph] = int(station_to_doublet[s])

    return {
        'n_stations': n_st,
        'n_doublets': len(doublet_stations),
        'station_z_mm': station_z,
        'station_has_y': station_has_y,
        'doublet_stations': doublet_stations,
        'station_to_doublet': station_to_doublet,
        'panel_to_station': panel_to_station,
        'panel_to_xslot': panel_to_xslot,
        'panel_to_yslot': panel_to_yslot,
    }


def _build_piecewise_x0_profile(
    geometry: dict[str, Any],
    bounds: dict[str, Any],
    cat: dict[str, Any],
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray, bool]:
    """Construct exact non-overlapping material slab table and continuous piecewise-linear X0(z)."""
    w_z = np.asarray(geometry['tungsten_Zpos'], dtype=np.float64)
    w_t = float(geometry['tungsten_thickness'])
    s_z = np.asarray(geometry['scint_Zpos'], dtype=np.float64)
    s_t = float(geometry['scint_thickness'])
    p_z = np.asarray(geometry['pixel_Zpos'], dtype=np.float64)
    p_t = float(geometry['silicon_thickness']) / 1000.0
    n_pin = int(geometry['nPinpointBlocks']) + max(0, int(geometry['nFortuneBlocks']) - 1)
    is_mc8588_order = bool(float(p_z[0]) > float(w_z[0]))

    intervals: list[tuple[float, float, int, float]] = []
    air_gaps: list[tuple[float, float]] = []

    for p in range(n_pin):
        zp = float(p_z[p])
        if is_mc8588_order:
            # 10085-10088 PinpointBlock layout relative to PixelSensor center zp:
            # AlWall_Back (2mm) [-18.005, -16.005], Scint_V (5mm) [-16.005, -11.005],
            # W_back (5mm) [-11.005, -6.005], AirGap (6mm) [-6.005, -0.005],
            # Si (0.01mm) [-0.005, +0.005], AlCool (5mm) [+0.005, +5.005],
            # W_front (5mm) [+5.005, +10.005], AlWall_Front (2mm) [+10.005, +12.005]
            intervals.append((zp - 18.005, zp - 16.005, MAT_AL, 2.0 / _X0_AL_MM))
            intervals.append((zp - 6.005, zp - 0.005, MAT_AIR, 6.0 / X0_AIR_MM))
            air_gaps.append((zp - 6.005, zp - 0.005))
            intervals.append((zp - 0.005, zp + 0.005, MAT_SI, p_t / _X0_SI_MM))
            intervals.append((zp + 0.005, zp + 5.005, MAT_AL, 5.0 / _X0_AL_MM))
            intervals.append((zp + 10.005, zp + 12.005, MAT_AL, 2.0 / _X0_AL_MM))
        else:
            # Legacy v6_clean_v3 PinpointBlock order
            intervals.append((zp - 8.005, zp - 6.005, MAT_AL, 2.0 / _X0_AL_MM))
            intervals.append((zp - 6.005, zp - 0.005, MAT_AIR, 6.0 / X0_AIR_MM))
            air_gaps.append((zp - 6.005, zp - 0.005))
            intervals.append((zp - 0.005, zp + 0.005, MAT_SI, p_t / _X0_SI_MM))
            intervals.append((zp + 0.005, zp + 5.005, MAT_AL, 5.0 / _X0_AL_MM))
            intervals.append((zp + 20.005, zp + 22.005, MAT_AL, 2.0 / _X0_AL_MM))

    # All 108 Tungsten plates (each 5 mm)
    w_bounds = np.column_stack([w_z - 0.5 * w_t, w_z + 0.5 * w_t])
    for zw in w_z:
        intervals.append((float(zw - 0.5 * w_t), float(zw + 0.5 * w_t), MAT_W, w_t / _X0_W_MM))

    # All 96 Scintillator panels (each 5 mm)
    for zs in s_z:
        intervals.append((float(zs - 0.5 * s_t), float(zs + 0.5 * s_t), MAT_SCINT, s_t / _X0_SCINT_MM))

    # IPT silicon planes (3 x 10 um)
    for zp in p_z[n_pin:]:
        intervals.append((float(zp - 0.5 * p_t), float(zp + 0.5 * p_t), MAT_SI, p_t / _X0_SI_MM))

    intervals.sort(key=lambda item: item[0])
    full_intervals: list[tuple[float, float, int, float]] = []
    cur_z = intervals[0][0] - 5.0
    for z0, z1, mid, dx0 in intervals:
        if z0 > cur_z + 1e-6:
            full_intervals.append((cur_z, z0, MAT_AIR, (z0 - cur_z) / X0_AIR_MM))
        elif z0 < cur_z - 1e-4:
            z0 = cur_z
        full_intervals.append((z0, z1, mid, dx0))
        cur_z = z1
    full_intervals.append((cur_z, cur_z + 200.0, MAT_AIR, 200.0 / X0_AIR_MM))

    knots_z = [full_intervals[0][0]]
    knots_x0 = [0.0]
    acc = 0.0
    for z0, z1, _, dx0 in full_intervals:
        acc += dx0
        knots_z.append(z1)
        knots_x0.append(acc)

    mat_z = np.array([[a, b] for a, b, _, _ in full_intervals], dtype=np.float64)
    mat_id = np.array([mid for _, _, mid, _ in full_intervals], dtype=np.int8)
    return (
        np.asarray(knots_z, dtype=np.float64),
        np.asarray(knots_x0, dtype=np.float64),
        mat_z,
        mat_id,
        np.asarray(air_gaps, dtype=np.float64),
    ), w_bounds, is_mc8588_order


_CONTRACT_CACHE: dict[str, Gen4GeometryContract] = {}


def _read_geometry_from_root(root_path: Path) -> dict[str, Any]:
    import uproot

    f = uproot.open(str(root_path))
    g = f['geometry'].arrays(library='np')
    geom_out: dict[str, Any] = {}
    for k, v in g.items():
        arr = np.asarray(v[0] if len(v) == 1 else v)
        if arr.ndim == 0:
            geom_out[k] = float(arr) if np.issubdtype(arr.dtype, np.floating) else int(arr)
        else:
            geom_out[k] = arr.tolist()
    return geom_out


def load_geometry_contract(
    geometry_or_path: Path | str | dict[str, Any] | None = None,
) -> Gen4GeometryContract:
    """Parse 10085-10088 geometry (`geometry_mc8588.json` or `.root:geometry`) into Gen4GeometryContract."""
    if isinstance(geometry_or_path, dict):
        geom = geometry_or_path
        cache_key = None
    else:
        if geometry_or_path is None:
            p_cand = DEFAULT_GEOMETRY_JSON if DEFAULT_GEOMETRY_JSON.exists() else DEFAULT_GEOMETRY_ROOT
        else:
            p_cand = Path(geometry_or_path)
        p = assert_unsealed_path(p_cand)
        cache_key = str(Path(p).resolve())
        if cache_key in _CONTRACT_CACHE:
            return _CONTRACT_CACHE[cache_key]
        if p.suffix == '.root':
            geom = _read_geometry_from_root(p)
        else:
            geom = json.loads(p.read_text())

    bounds = detector_bounds(geom)
    stations = _build_scint_stations_gen4(geom, bounds)
    cat = pinpoint_slab_catalog(geom)

    n_pin_blocks = int(geom['nPinpointBlocks']) + max(0, int(geom['nFortuneBlocks']) - 1)
    n_fortune = int(geom['nFortuneBlocks'])
    n_pix_planes = int(bounds['nz'])
    n_ipt = n_pix_planes - n_pin_blocks
    p_z = np.asarray(bounds['pixel_Zpos'], dtype=np.float64)

    (knots_z, knots_x0, mat_z, mat_id, air_gaps), w_bounds, is_mc8588_order = _build_piecewise_x0_profile(
        geom, bounds, cat
    )
    p_x0 = np.interp(p_z, knots_z, knots_x0, left=knots_x0[0], right=knots_x0[-1])

    bx = np.asarray(bounds['scint_bar_Xcenter'], dtype=np.float64)
    by = np.asarray(bounds['scint_bar_Ycenter'], dtype=np.float64)
    half_scint_w = 0.5 * float(geom['scint_detector_width'])
    half_scint_h = 0.5 * float(geom['scint_detector_height'])
    x_sc_c = float(bounds['x_scint_center'])
    y_sc_c = float(bounds['y_scint_center'])

    st_z = np.asarray(stations['station_z_mm'], dtype=np.float64)
    st_x0 = np.interp(st_z, knots_z, knots_x0, left=knots_x0[0], right=knots_x0[-1])
    st_has_y = np.asarray(stations['station_has_y'], dtype=bool)
    s_xonly = np.flatnonzero(~st_has_y).astype(np.int64)
    s_stereo = np.asarray(stations['doublet_stations'], dtype=np.int64)
    st_to_dbl = np.asarray(stations['station_to_doublet'], dtype=np.int64)

    pin_scint_st = np.empty(n_pin_blocks, dtype=np.int64)
    w_back_z = np.empty(n_pin_blocks, dtype=np.float64)
    w_front_z = np.empty(n_pin_blocks, dtype=np.float64)
    for p in range(n_pin_blocks):
        if is_mc8588_order:
            pin_scint_st[p] = int(np.argmin(np.abs(st_z - (p_z[p] - 13.505))))
            w_back_z[p] = p_z[p] - 8.505
            w_front_z[p] = p_z[p] + 7.505
        else:
            pin_scint_st[p] = int(np.argmin(np.abs(st_z - (p_z[p] + 12.505))))
            w_back_z[p] = p_z[p] + 7.505
            w_front_z[p] = p_z[p] + 17.505

    # FortuneBlock[f] doublet masks (7 blocks x 6 stereo stations = 42 doublets)
    fortune_station_mask = np.zeros((n_fortune, int(stations['n_stations'])), dtype=bool)
    n_sl_per_f = int(geom['numScintLayers'])
    for f_idx in range(n_fortune):
        f_doublets = s_stereo[f_idx * n_sl_per_f : (f_idx + 1) * n_sl_per_f]
        fortune_station_mask[f_idx, f_doublets] = True

    # BackConeSlabPrior arrays (length 12, decoupled from arithmetic offsets)
    pin_sand_scint = np.full(n_pin_blocks, -1, dtype=np.int64)
    pin_up_w_front = np.full(n_pin_blocks, np.nan, dtype=np.float64)
    pin_fortune_id = np.full(n_pin_blocks, -1, dtype=np.int64)
    w_z_all = np.asarray(cat['z_center_mm'], dtype=np.float64)
    for p0 in range(1, n_pin_blocks):
        if p0 <= 5:
            pin_sand_scint[p0] = int(pin_scint_st[p0]) if is_mc8588_order else int(pin_scint_st[p0 - 1])
            pin_up_w_front[p0] = w_front_z[p0 - 1]
        else:
            f_idx = p0 - 6
            pin_fortune_id[p0] = f_idx
            pin_sand_scint[p0] = int(pin_scint_st[p0])
            up_w = w_z_all[w_z_all < w_back_z[p0] - 1.0]
            pin_up_w_front[p0] = float(up_w[-1]) if len(up_w) else np.nan

    panel_to_station_arr = np.asarray(stations['panel_to_station'], dtype=np.int64)
    panel_view_arr = np.asarray(bounds['panel_is_horiz'], dtype=np.int64)
    panel_to_station_view = np.column_stack([panel_to_station_arr, panel_view_arr]).astype(np.int64)

    contract = Gen4GeometryContract(
        raw_geometry=geom,
        nx=int(bounds['nx']),
        ny=int(bounds['ny']),
        nz=n_pix_planes,
        n_target_planes=n_pin_blocks,
        n_ipt_planes=n_ipt,
        dx_mm=float(bounds['dx_mm']),
        dy_mm=float(bounds['dy_mm']),
        px_min=float(bounds['px_min']),
        px_max=float(bounds['px_max']),
        py_min=float(bounds['py_min']),
        py_max=float(bounds['py_max']),
        x_pix_center=float(bounds['x_pix_center']),
        y_pix_center=float(bounds['y_pix_center']),
        pixel_Zpos=p_z,
        pixel_X0=p_x0,
        n_bars=int(bounds['n_bars']),
        bar_pitch_mm=float(np.median(np.diff(bx))),
        bar_width_mm=float(geom['scint_bar_width']),
        x_scint_center=x_sc_c,
        y_scint_center=y_sc_c,
        scint_x_min=x_sc_c - half_scint_w,
        scint_x_max=x_sc_c + half_scint_w,
        scint_y_min=y_sc_c - half_scint_h,
        scint_y_max=y_sc_c + half_scint_h,
        x_sipm_horiz_mm=x_sc_c - half_scint_w,  # -133.0 mm
        y_sipm_vert_mm=y_sc_c + half_scint_h,   # +198.0 mm
        bar_x_centers_mm=bx,
        bar_y_centers_mm=by,
        n_stations=int(stations['n_stations']),
        n_doublets=int(stations['n_doublets']),
        station_z_mm=st_z,
        station_x0=st_x0,
        station_has_y=st_has_y,
        s_front_xonly=s_xonly[: int(geom['nPinpointBlocks'])],
        s_inter_xonly=s_xonly[int(geom['nPinpointBlocks']) :],
        s_all_xonly=s_xonly,
        s_stereo=s_stereo,
        station_to_doublet=st_to_dbl,
        panel_to_station=panel_to_station_arr,
        panel_to_xslot=np.asarray(stations['panel_to_xslot'], dtype=np.int64),
        panel_to_yslot=np.asarray(stations['panel_to_yslot'], dtype=np.int64),
        PANEL_TO_STATION_VIEW=panel_to_station_view,
        STATION_HAS_Y=st_has_y,
        pinpoint_scint_stations=pin_scint_st,
        pinpoint_sand_scint_station=pin_sand_scint,
        pinpoint_upstream_w_front_z=pin_up_w_front,
        pinpoint_fortune_block_id=pin_fortune_id,
        fortune_station_mask=fortune_station_mask,
        is_mc8588_block_order=is_mc8588_order,
        w_center_x_mm=float(cat['w_center_x_mm']),
        w_center_y_mm=float(cat['w_center_y_mm']),
        w_x_min_mm=float(cat['w_x_min_mm']),
        w_x_max_mm=float(cat['w_x_max_mm']),
        w_y_min_mm=float(cat['w_y_min_mm']),
        w_y_max_mm=float(cat['w_y_max_mm']),
        tungsten_Zpos=w_z_all,
        w_thickness_mm=float(cat['w_thickness_mm']),
        air_gap_6mm_bounds=air_gaps,
        tungsten_bounds=w_bounds,
        pinpoint_w_back_z=w_back_z,
        pinpoint_w_front_z=w_front_z,
        x0_knots_z=knots_z,
        x0_knots_val=knots_x0,
        material_intervals_z=mat_z,
        material_intervals_id=mat_id,
        bounds_dict=bounds,
    )
    if cache_key is not None:
        _CONTRACT_CACHE[cache_key] = contract
    return contract


# =============================================================================
# Sub-Aperture Conditional Symmetry Augmentation Operators (Zero-Artifact Orbits)
# =============================================================================

def _clone_obs(obs: dict[str, Any]) -> dict[str, Any]:
    out = {
        'pixel': {k: np.array(v, copy=True) for k, v in obs['pixel'].items()},
        'bar': {k: np.array(v, copy=True) for k, v in obs['bar'].items()},
        'tracker_hypothesis': {k: np.array(v, copy=True) for k, v in obs['tracker_hypothesis'].items()},
    }
    for k in ('has_response', 'outside_acceptance', 'fit_attempted', 'momentum_censored'):
        if k in obs:
            out[k] = obs[k]
    return out


def _estimate_station_centers(
    bar_panel: np.ndarray,
    bar_horiz: np.ndarray,
    bar_idx: np.ndarray,
    bar_pe: np.ndarray,
    contract: Gen4GeometryContract,
    *,
    pxyz: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate equivariant per-station (x_hat_st, y_hat_st) using linear bar PE weights."""
    n_st = contract.n_stations
    bpe = np.asarray(bar_pe, dtype=np.float64)
    frac = np.clip(bpe / SIPM_NPIX, 0.0, 0.995)
    pe_lin = np.where(bpe > 0.0, -SIPM_NPIX * np.log(1.0 - frac), 0.0)
    st = contract.panel_to_station[np.asarray(bar_panel, dtype=np.int64)]
    horiz = np.asarray(bar_horiz, dtype=np.int64)
    bi = np.asarray(bar_idx, dtype=np.int64)

    x_st = np.full(n_st, contract.x_scint_center, dtype=np.float64)
    y_st = np.full(n_st, contract.y_scint_center, dtype=np.float64)

    xm = horiz == 0
    if xm.any():
        wx = pe_lin[xm] + 1e-3
        sx = np.bincount(st[xm], weights=wx, minlength=n_st)
        vx = np.bincount(st[xm], weights=wx * contract.bar_x_centers_mm[bi[xm]], minlength=n_st)
        act_x = sx > 0.0
        x_st[act_x] = vx[act_x] / sx[act_x]
        no_x = ~act_x
        if no_x.any() and act_x.any():
            act_idx = np.flatnonzero(act_x)
            z_act = contract.station_z_mm[act_idx]
            for s_miss in np.flatnonzero(no_x):
                nearest = act_idx[int(np.argmin(np.abs(z_act - contract.station_z_mm[s_miss])))]
                x_st[s_miss] = x_st[nearest]
    elif pxyz is not None and len(pxyz) > 0:
        x_st[:] = float(np.median(pxyz[:, 0]))

    ym = horiz == 1
    if ym.any():
        wy = pe_lin[ym] + 1e-3
        sy = np.bincount(st[ym], weights=wy, minlength=n_st)
        vy = np.bincount(st[ym], weights=wy * contract.bar_y_centers_mm[bi[ym]], minlength=n_st)
        act_y = sy > 0.0
        y_st[act_y] = vy[act_y] / sy[act_y]
        # Interpolate X-only stations from nearest active doublet stations (equivariant under CP_y)
        no_y = ~act_y
        if no_y.any() and act_y.any():
            act_idx = np.flatnonzero(act_y)
            z_act = contract.station_z_mm[act_idx]
            for s_miss in np.flatnonzero(no_y):
                nearest = act_idx[int(np.argmin(np.abs(z_act - contract.station_z_mm[s_miss])))]
                y_st[s_miss] = y_st[nearest]
    elif pxyz is not None and len(pxyz) > 0:
        y_st[:] = float(np.median(pxyz[:, 1]))

    return x_st, y_st


def _compensate_bar_attenuation_and_time(
    obs: dict[str, Any],
    horiz_target: int,
    delta_d_per_bar: np.ndarray,
    *,
    continuous_pe: bool = False,
) -> None:
    """Apply exact optical-path attenuation & WLS group-velocity time shift to bars with `horiz == horiz_target`.

    For exact involution T(T(x)) == x:
      - If `continuous_pe=True`, preserves exact float linear PE in `bar['_pe_lin_float']` across chained transforms.
      - Otherwise updates `bar['pe_proxy']` via 14,400-cell inverse-saturation -> exp(dd/lambda) -> re-saturation.
    """
    bar = obs['bar']
    horiz = np.asarray(bar['horizontal'], dtype=np.int64)
    mask = horiz == horiz_target
    if not mask.any():
        return

    bpe = np.asarray(bar['pe_proxy'], dtype=np.float64)
    if '_pe_lin_float' in bar:
        pe_lin = np.asarray(bar['_pe_lin_float'], dtype=np.float64).copy()
    else:
        frac = np.clip(bpe / SIPM_NPIX, 0.0, 0.995)
        pe_lin = np.where(bpe > 0.0, -SIPM_NPIX * np.log(1.0 - frac), 0.0)

    dd = delta_d_per_bar[mask]
    # Moving farther from SiPM (dd > 0) reduces PE by exp(-dd / lambda_att) and increases time by dd / v_g
    pe_lin[mask] = pe_lin[mask] * np.exp(-dd / BC408_ATTENUATION_MM)
    pe_sat = SIPM_NPIX * (1.0 - np.exp(-pe_lin / SIPM_NPIX))

    if continuous_pe:
        bar['_pe_lin_float'] = pe_lin
        bar['pe_proxy'] = pe_sat.astype(np.float64)
    else:
        bar['pe_proxy'] = np.clip(np.rint(pe_sat), 0, int(SIPM_NPIX)).astype(np.int64)

    if 'hit_time_ns' in bar and len(bar['hit_time_ns']) == len(bpe):
        ht = np.asarray(bar['hit_time_ns'], dtype=np.float64).copy()
        valid_t = mask & (ht >= 0.0)
        if valid_t.any():
            ht[valid_t] = np.maximum(0.0, ht[valid_t] + delta_d_per_bar[valid_t] / WLS_GROUP_VELOCITY_MM_PER_NS)
        bar['hit_time_ns'] = ht.astype(np.float32)


def _rebuild_bar_channel_ids_and_bounds(
    bar: dict[str, Any],
    contract: Gen4GeometryContract,
) -> None:
    """Recompute exact canonical `channel_id` and `bounds_mm` from `(layer, panel, horizontal, bar)`."""
    n = len(bar['panel'])
    if n == 0:
        return
    lay = np.asarray(bar['layer'], dtype=np.int64)
    pan = np.asarray(bar['panel'], dtype=np.int64)
    hor = np.asarray(bar['horizontal'], dtype=np.int64)
    bi = np.asarray(bar['bar'], dtype=np.int64)
    n_scint = len(contract.raw_geometry['scint_Zpos'])
    n_bars = contract.n_bars

    bar['channel_id'] = ((lay * n_scint + pan) * (2 * n_bars) + hor * n_bars + bi).astype(np.uint64)
    x = np.where(hor == 1, contract.x_scint_center, contract.bar_x_centers_mm[bi])
    y = np.where(hor == 1, contract.bar_y_centers_mm[bi], contract.y_scint_center)
    hx = np.where(hor == 1, 0.5 * float(contract.raw_geometry['scint_detector_width']), 0.5 * contract.bar_width_mm)
    hy = np.where(hor == 1, 0.5 * contract.bar_width_mm, 0.5 * float(contract.raw_geometry['scint_detector_height']))
    z = np.asarray(contract.raw_geometry['scint_Zpos'], dtype=np.float64)[pan]
    bar['bounds_mm'] = np.column_stack([x - hx, x + hx, y - hy, y + hy, z]).astype(np.float32)


def _flip_label7(label_7: int | None) -> int | None:
    """Charge-conjugation label_7 flip: mu_minus_CC (1) <-> mu_plus_CC (2)."""
    if label_7 is None:
        return None
    if label_7 == 1:
        return 2
    if label_7 == 2:
        return 1
    return int(label_7)


def check_symmetry_eligibility(
    obs: dict[str, Any],
    contract: Gen4GeometryContract,
) -> dict[str, Any]:
    """Determine exact artifact-free eligibility for P_{x,sub}^{Si}, P_{x,sub}^{Scint}, CP_y^{fid}, and T_{kx,ky}^{fid}."""
    pxyz = np.asarray(obs['pixel']['xyz_mm'], dtype=np.float64).reshape(-1, 3)
    pcid = np.asarray(obs['pixel']['channel_id'], dtype=np.int64)
    bar = obs['bar']
    bh = np.asarray(bar['horizontal'], dtype=np.int64)
    bi = np.asarray(bar['bar'], dtype=np.int64)
    bpe = np.asarray(bar['pe_proxy'], dtype=np.float64)
    active_b = bpe > 0.0
    bx_act = bi[(bh == 0) & active_b]
    by_act = bi[(bh == 1) & active_b]

    trk = obs['tracker_hypothesis']
    txy = np.asarray(trk['xy_mm'], dtype=np.float64).reshape(-1, 2)
    n_pix = len(pxyz)
    n_trk = len(txy)

    # 1A: P_{x,sub}^{Si} around x_axis = -1.75 mm (b_x in {0..24} -> 24 - b_x)
    # Symmetric pixel gate around -1.75 mm: [-126.0, +122.5] guarantees elig(P_x(x)) == elig(x)
    ok_bx_si = (len(bx_act) == 0) or bool(np.all((bx_act >= 2) & (bx_act <= 22)))
    ok_pix_si = (n_pix == 0) or bool(np.all((pxyz[:, 0] >= -126.0) & (pxyz[:, 0] <= 122.5)))
    if n_trk > 0:
        tx_refl = 2.0 * SI_SUB_X_AXIS_MM - txy[:, 0]
        r_bore_refl = np.hypot(tx_refl - MAGNET_BORE_X_MM, txy[:, 1] - MAGNET_BORE_Y_MM)
        ok_trk_si = bool(
            np.all((tx_refl >= TRACKER_X_MIN_MM) & (tx_refl <= TRACKER_X_MAX_MM))
            and np.all(r_bore_refl <= MAGNET_BORE_FID_R_MM)
        )
    else:
        ok_trk_si = True
    elig_px_si = ok_bx_si and ok_pix_si and ok_trk_si

    # 1B: P_{x,sub}^{Scint} around x_scint = +77.0 mm (only zero-pixel, zero-tracker Fortune_ScintOnly)
    ok_bx_scint = (len(bx_act) == 0) or bool(np.all((bx_act >= 2) & (bx_act <= 37)))
    elig_px_scint = (n_pix == 0) and (n_trk == 0) and ok_bx_scint

    # 2: CP_y^{fid} around y_c = -12.0 mm (q -> -q)
    ok_by = (len(by_act) == 0) or bool(np.all((by_act >= 2) & (by_act <= 37)))
    ok_py = (n_pix == 0) or bool(np.all((pxyz[:, 1] >= -105.0) & (pxyz[:, 1] <= 81.0)))
    if n_trk > 0:
        ty_refl = 2.0 * COMMON_Y_AXIS_MM - txy[:, 1]
        r_bore_yrefl = np.hypot(txy[:, 0] - MAGNET_BORE_X_MM, ty_refl - MAGNET_BORE_Y_MM)
        ok_trk_y = bool(
            np.all((ty_refl >= TRACKER_Y_MIN_MM) & (ty_refl <= TRACKER_Y_MAX_MM))
            and np.all(r_bore_yrefl <= MAGNET_BORE_FID_R_MM)
        )
    else:
        ok_trk_y = True
    elig_cp_y = ok_by and ok_py and ok_trk_y

    # 3: Valid discrete bar-pitch shifts (kx, ky) in {-2..2}^2 \ {(0,0)}
    valid_shifts: list[tuple[int, int]] = []
    dx_step = LATTICE_SHIFT_PIX_X * contract.dx_mm  # 10.504 mm
    dy_step = LATTICE_SHIFT_PIX_Y * contract.dy_mm  # 10.488 mm
    has_pix = n_pix > 0
    for kx in (-2, -1, 0, 1, 2):
        for ky in (-2, -1, 0, 1, 2):
            if kx == 0 and ky == 0:
                continue
            dx_s = kx * dx_step
            dy_s = ky * dy_step
            if has_pix:
                xs = pxyz[:, 0] + dx_s
                ys = pxyz[:, 1] + dy_s
                if not (
                    np.all((xs >= contract.px_min + 10.0) & (xs <= contract.px_max - 10.0))
                    and np.all((ys >= contract.py_min + 10.0) & (ys <= contract.py_max - 10.0))
                ):
                    continue
                if len(bx_act) > 0 and not np.all((bx_act + kx >= 2) & (bx_act + kx <= 23)):
                    continue
            else:
                if len(bx_act) > 0 and not np.all((bx_act + kx >= 2) & (bx_act + kx <= 37)):
                    continue
            if len(by_act) > 0 and not np.all((by_act + ky >= 2) & (by_act + ky <= 37)):
                continue
            if n_trk > 0:
                txs = txy[:, 0] + dx_s
                tys = txy[:, 1] + dy_s
                if not (
                    np.all((txs >= TRACKER_X_MIN_MM) & (txs <= TRACKER_X_MAX_MM))
                    and np.all((tys >= TRACKER_Y_MIN_MM) & (tys <= TRACKER_Y_MAX_MM))
                    and np.all(np.hypot(txs - MAGNET_BORE_X_MM, tys - MAGNET_BORE_Y_MM) <= MAGNET_BORE_FID_R_MM)
                ):
                    continue
            valid_shifts.append((kx, ky))

    return {
        'px_sub_si': bool(elig_px_si),
        'px_sub_scint': bool(elig_px_scint),
        'cp_y_fid': bool(elig_cp_y),
        'valid_shifts': valid_shifts,
        'n_pixels': int(n_pix),
        'n_tracker': int(n_trk),
    }


def apply_px_sub_si(
    obs: dict[str, Any],
    contract: Gen4GeometryContract,
    *,
    label_7: int | None = None,
    continuous_pe: bool = False,
    enforce_gate: bool = True,
) -> tuple[dict[str, Any], int | None]:
    """Sub-aperture silicon-contained X reflection around x = -1.75 mm (b_x: 0..24 -> 24 - b_x)."""
    if enforce_gate and not check_symmetry_eligibility(obs, contract)['px_sub_si']:
        raise ValueError('Observation violates Omega_x^Si gate for P_{x,sub}^{Si}')

    out = _clone_obs(obs)
    pxyz = np.asarray(out['pixel']['xyz_mm'], dtype=np.float64).reshape(-1, 3)
    pcid = np.asarray(out['pixel']['channel_id'], dtype=np.int64)
    if len(pcid) > 0:
        stride = contract.nx * contract.ny
        planes = pcid // stride
        rows = (pcid // contract.nx) % contract.ny
        cols = pcid % contract.nx
        cols_new = np.clip(SI_REFL_COL_SUM - cols, 0, contract.nx - 1)
        out['pixel']['channel_id'] = ((planes * contract.ny + rows) * contract.nx + cols_new).astype(np.uint64)
        pxyz[:, 0] = 2.0 * SI_SUB_X_AXIS_MM - pxyz[:, 0]
        out['pixel']['xyz_mm'] = pxyz.astype(np.float32)

    bar = out['bar']
    if len(bar['bar']) > 0:
        bp = np.asarray(bar['panel'], dtype=np.int64)
        bh = np.asarray(bar['horizontal'], dtype=np.int64)
        bi = np.asarray(bar['bar'], dtype=np.int64)
        bpe = np.asarray(bar['pe_proxy'], dtype=np.float64)
        x_st, _ = _estimate_station_centers(bp, bh, bi, bpe, contract, pxyz=pxyz)

        # Reflect vertical X bars bi -> 24 - bi (in 0..24)
        xm = bh == 0
        bi_new = bi.copy()
        bi_new[xm] = SI_SUB_BAR_SUM - bi[xm]
        bar['bar'] = bi_new.astype(np.int16)
        _rebuild_bar_channel_ids_and_bounds(bar, contract)

        # Horizontal Y bars change distance to x_SiPM = -133.0 mm by dx = (x' + 133) - (x + 133) = -2*x_st - 3.50
        st_idx = contract.panel_to_station[bp]
        delta_d = -2.0 * x_st[st_idx] + 2.0 * SI_SUB_X_AXIS_MM
        _compensate_bar_attenuation_and_time(out, horiz_target=1, delta_d_per_bar=delta_d, continuous_pe=continuous_pe)

    trk = out['tracker_hypothesis']
    txy = np.asarray(trk['xy_mm'], dtype=np.float64).reshape(-1, 2)
    if len(txy) > 0:
        txy[:, 0] = 2.0 * SI_SUB_X_AXIS_MM - txy[:, 0]
        trk['xy_mm'] = txy.astype(np.float32)

    return out, label_7


def apply_px_sub_scint(
    obs: dict[str, Any],
    contract: Gen4GeometryContract,
    *,
    label_7: int | None = None,
    continuous_pe: bool = False,
    enforce_gate: bool = True,
) -> tuple[dict[str, Any], int | None]:
    """Pure-calorimeter X reflection around x_scint = +77.0 mm (b_x: 0..39 -> 39 - b_x)."""
    if enforce_gate and not check_symmetry_eligibility(obs, contract)['px_sub_scint']:
        raise ValueError('Observation violates Scint-only gate for P_{x,sub}^{Scint}')

    out = _clone_obs(obs)
    bar = out['bar']
    if len(bar['bar']) > 0:
        bp = np.asarray(bar['panel'], dtype=np.int64)
        bh = np.asarray(bar['horizontal'], dtype=np.int64)
        bi = np.asarray(bar['bar'], dtype=np.int64)
        bpe = np.asarray(bar['pe_proxy'], dtype=np.float64)
        x_st, _ = _estimate_station_centers(bp, bh, bi, bpe, contract)

        xm = bh == 0
        bi_new = bi.copy()
        bi_new[xm] = SCINT_SUB_BAR_SUM - bi[xm]
        bar['bar'] = bi_new.astype(np.int16)
        _rebuild_bar_channel_ids_and_bounds(bar, contract)

        # Delta_d_x = (154.0 - x_st) - x_st = 154.0 - 2 * x_st
        st_idx = contract.panel_to_station[bp]
        delta_d = 2.0 * SCINT_SUB_X_AXIS_MM - 2.0 * x_st[st_idx]
        _compensate_bar_attenuation_and_time(out, horiz_target=1, delta_d_per_bar=delta_d, continuous_pe=continuous_pe)

    return out, label_7


def apply_cp_y_fid(
    obs: dict[str, Any],
    contract: Gen4GeometryContract,
    *,
    label_7: int | None = None,
    continuous_pe: bool = False,
    enforce_gate: bool = True,
) -> tuple[dict[str, Any], int | None]:
    """Charge-conjugated Y reflection around y = -12.0 mm (b_y: 0..39 -> 39 - b_y, q -> -q)."""
    if enforce_gate and not check_symmetry_eligibility(obs, contract)['cp_y_fid']:
        raise ValueError('Observation violates fiducial Y/bore/tracker gate for CP_y^{fid}')

    out = _clone_obs(obs)
    pxyz = np.asarray(out['pixel']['xyz_mm'], dtype=np.float64).reshape(-1, 3)
    pcid = np.asarray(out['pixel']['channel_id'], dtype=np.int64)
    if len(pcid) > 0:
        stride = contract.nx * contract.ny
        planes = pcid // stride
        rows = (pcid // contract.nx) % contract.ny
        cols = pcid % contract.nx
        rows_new = np.clip(SI_REFL_ROW_SUM - rows, 0, contract.ny - 1)
        out['pixel']['channel_id'] = ((planes * contract.ny + rows_new) * contract.nx + cols).astype(np.uint64)
        pxyz[:, 1] = 2.0 * COMMON_Y_AXIS_MM - pxyz[:, 1]
        out['pixel']['xyz_mm'] = pxyz.astype(np.float32)

    bar = out['bar']
    if len(bar['bar']) > 0:
        bp = np.asarray(bar['panel'], dtype=np.int64)
        bh = np.asarray(bar['horizontal'], dtype=np.int64)
        bi = np.asarray(bar['bar'], dtype=np.int64)
        bpe = np.asarray(bar['pe_proxy'], dtype=np.float64)
        _, y_st = _estimate_station_centers(bp, bh, bi, bpe, contract, pxyz=pxyz)

        ym = bh == 1
        bi_new = bi.copy()
        bi_new[ym] = SCINT_Y_BAR_SUM - bi[ym]
        bar['bar'] = bi_new.astype(np.int16)
        _rebuild_bar_channel_ids_and_bounds(bar, contract)

        # Vertical X bars have SiPM at y_SiPM = +198.0 mm -> d_X(y) = 198 - y
        # d_X(y') - d_X(y) = y - y' = y - (-24 - y) = 2 * (y + 12.0)
        st_idx = contract.panel_to_station[bp]
        delta_d = 2.0 * (y_st[st_idx] - COMMON_Y_AXIS_MM)
        _compensate_bar_attenuation_and_time(out, horiz_target=0, delta_d_per_bar=delta_d, continuous_pe=continuous_pe)

    trk = out['tracker_hypothesis']
    txy = np.asarray(trk['xy_mm'], dtype=np.float64).reshape(-1, 2)
    if len(txy) > 0:
        txy[:, 1] = 2.0 * COMMON_Y_AXIS_MM - txy[:, 1]
        trk['xy_mm'] = txy.astype(np.float32)

    return out, _flip_label7(label_7)


def apply_discrete_shift_fid(
    obs: dict[str, Any],
    contract: Gen4GeometryContract,
    kx: int,
    ky: int,
    *,
    continuous_pe: bool = False,
    enforce_gate: bool = True,
) -> dict[str, Any]:
    """Discrete bar-pitch lattice translation (10.504*kx, 10.488*ky) mm = (505*kx*dx, 460*ky*dy)."""
    if kx == 0 and ky == 0:
        return _clone_obs(obs)
    if enforce_gate:
        elig = check_symmetry_eligibility(obs, contract)
        if (kx, ky) not in elig['valid_shifts']:
            raise ValueError(f'Shift ({kx}, {ky}) violates fiducial bounds for this event')

    out = _clone_obs(obs)
    dx_mm = float(kx) * LATTICE_SHIFT_PIX_X * contract.dx_mm
    dy_mm = float(ky) * LATTICE_SHIFT_PIX_Y * contract.dy_mm

    pxyz = np.asarray(out['pixel']['xyz_mm'], dtype=np.float64).reshape(-1, 3)
    pcid = np.asarray(out['pixel']['channel_id'], dtype=np.int64)
    if len(pcid) > 0:
        stride = contract.nx * contract.ny
        planes = pcid // stride
        rows = (pcid // contract.nx) % contract.ny
        cols = pcid % contract.nx
        cols_new = np.clip(cols + kx * LATTICE_SHIFT_PIX_X, 0, contract.nx - 1)
        rows_new = np.clip(rows + ky * LATTICE_SHIFT_PIX_Y, 0, contract.ny - 1)
        out['pixel']['channel_id'] = ((planes * contract.ny + rows_new) * contract.nx + cols_new).astype(np.uint64)
        pxyz[:, 0] += dx_mm
        pxyz[:, 1] += dy_mm
        out['pixel']['xyz_mm'] = pxyz.astype(np.float32)

    bar = out['bar']
    if len(bar['bar']) > 0:
        bh = np.asarray(bar['horizontal'], dtype=np.int64)
        bi = np.asarray(bar['bar'], dtype=np.int64)
        bi_new = np.where(bh == 0, bi + kx, bi + ky)
        bar['bar'] = np.clip(bi_new, 0, contract.n_bars - 1).astype(np.int16)
        _rebuild_bar_channel_ids_and_bounds(bar, contract)

        # Horizontal Y bars: SiPM at -X, so +dx_mm increases light path by +dx_mm
        if kx != 0:
            _compensate_bar_attenuation_and_time(
                out, horiz_target=1, delta_d_per_bar=np.full(len(bi), dx_mm, dtype=np.float64), continuous_pe=continuous_pe
            )
        # Vertical X bars: SiPM at +Y, so +dy_mm decreases light path by -dy_mm
        if ky != 0:
            _compensate_bar_attenuation_and_time(
                out, horiz_target=0, delta_d_per_bar=np.full(len(bi), -dy_mm, dtype=np.float64), continuous_pe=continuous_pe
            )

    trk = out['tracker_hypothesis']
    txy = np.asarray(trk['xy_mm'], dtype=np.float64).reshape(-1, 2)
    if len(txy) > 0:
        txy[:, 0] += dx_mm
        txy[:, 1] += dy_mm
        trk['xy_mm'] = txy.astype(np.float32)

    return out


def apply_random_valid_orbit(
    obs: dict[str, Any],
    contract: Gen4GeometryContract,
    rng: np.random.Generator,
    *,
    label_7: int | None = None,
    enable_shift: bool = True,
) -> tuple[dict[str, Any], int | None, dict[str, Any]]:
    """Sample a random artifact-free composition from {Id, P_x} x {Id, CP_y} x {Id, T_{kx,ky}}."""
    cur = _clone_obs(obs)
    cur_l7 = label_7
    applied: dict[str, Any] = {'px': 'none', 'cp_y': False, 'shift': (0, 0)}

    elig = check_symmetry_eligibility(cur, contract)
    if elig['px_sub_si'] and rng.random() < 0.5:
        cur, cur_l7 = apply_px_sub_si(cur, contract, label_7=cur_l7, enforce_gate=False)
        applied['px'] = 'si'
    elif elig['px_sub_scint'] and rng.random() < 0.5:
        cur, cur_l7 = apply_px_sub_scint(cur, contract, label_7=cur_l7, enforce_gate=False)
        applied['px'] = 'scint'

    elig2 = check_symmetry_eligibility(cur, contract)
    if elig2['cp_y_fid'] and rng.random() < 0.5:
        cur, cur_l7 = apply_cp_y_fid(cur, contract, label_7=cur_l7, enforce_gate=False)
        applied['cp_y'] = True

    if enable_shift:
        elig3 = check_symmetry_eligibility(cur, contract)
        shifts = elig3['valid_shifts']
        if shifts and rng.random() < 0.5:
            kx, ky = shifts[int(rng.integers(0, len(shifts)))]
            cur = apply_discrete_shift_fid(cur, contract, kx, ky, enforce_gate=False)
            applied['shift'] = (kx, ky)

    return cur, cur_l7, applied


# =============================================================================
# In-Place Host-Vertex Factorized Particle Surgery (NC + tau->mu Overlay)
# =============================================================================

def inverse_sat_additive_pe(
    pe_host: np.ndarray | float,
    pe_inject: np.ndarray | float,
    n_pix: float = SIPM_NPIX,
) -> np.ndarray | float:
    """Exact 14,400-cell SiPM inverse-saturation multiplicative overlay:

      N_surg = N_pix * [1 - (1 - N_A / N_pix) * (1 - N_mu / N_pix)].

    Algebraically identical to summing linear raw photoelectrons N_raw_A + N_raw_mu
    prior to applying N_pix * (1 - exp(-N_raw / N_pix)).
    """
    fa = np.clip(np.asarray(pe_host, dtype=np.float64) / n_pix, 0.0, 1.0)
    fb = np.clip(np.asarray(pe_inject, dtype=np.float64) / n_pix, 0.0, 1.0)
    out = n_pix * (1.0 - (1.0 - fa) * (1.0 - fb))
    if np.ndim(pe_host) == 0 and np.ndim(pe_inject) == 0:
        return float(out)
    return out


def sample_polarized_tau_to_mu(
    tau_p4_gev: np.ndarray,
    rng: np.random.Generator,
    *,
    helicity: float = -1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample polarized V-A tau -> mu nu_tau nubar_mu decay (rest-frame Michel spectrum boosted to lab).

    Returns (mu_p4_gev, tau_boost_beta) in GeV.
    """
    m_tau = 1.77686
    m_mu = 0.105658
    e_tau = float(tau_p4_gev[3])
    p_tau = np.asarray(tau_p4_gev[:3], dtype=np.float64)
    p_norm = float(np.linalg.norm(p_tau))
    beta_vec = p_tau / max(e_tau, m_tau)
    gamma = e_tau / m_tau
    u_tau = p_tau / max(p_norm, 1e-9)

    # Rejection sampling from dGamma / (dx dcos_th) ~ x^2 * [(3 - 2x) + hel * (1 - 2x) * cos_th]
    # Note for tau^- (helicity = -1) in V-A, high-energy mu is emitted opposite to tau spin -> cos_th < 0 suppressed
    while True:
        x = float(rng.uniform(2.0 * m_mu / m_tau, 1.0))
        c_th = float(rng.uniform(-1.0, 1.0))
        w = (x * x) * ((3.0 - 2.0 * x) + float(helicity) * (1.0 - 2.0 * x) * c_th)
        if rng.random() * 2.0 <= max(0.0, w):
            break

    e_mu_rf = 0.5 * m_tau * x
    p_mu_rf_mag = math.sqrt(max(0.0, e_mu_rf * e_mu_rf - m_mu * m_mu))
    s_th = math.sqrt(max(0.0, 1.0 - c_th * c_th))
    phi = float(rng.uniform(0.0, 2.0 * math.pi))

    # Build orthonormal frame around u_tau
    ref = np.array([1.0, 0.0, 0.0]) if abs(u_tau[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(u_tau, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u_tau, e1)
    p_mu_rf = p_mu_rf_mag * (s_th * math.cos(phi) * e1 + s_th * math.sin(phi) * e2 + c_th * u_tau)

    # Lorentz boost from tau rest frame to lab frame
    b2 = float(np.dot(beta_vec, beta_vec))
    bp = float(np.dot(beta_vec, p_mu_rf))
    factor = (gamma - 1.0) / max(b2, 1e-12) * bp + gamma * e_mu_rf
    p_mu_lab = p_mu_rf + factor * beta_vec
    e_mu_lab = gamma * (e_mu_rf + bp)
    return np.array([p_mu_lab[0], p_mu_lab[1], p_mu_lab[2], e_mu_lab], dtype=np.float64), beta_vec


def _propagate_muon_trajectory_yz_bending(
    sv_xyz_mm: np.ndarray,
    mu_p4_gev: np.ndarray,
    q_mu: int,
    z_samples_mm: np.ndarray,
    contract: Gen4GeometryContract,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Propagate a muon from sv_xyz_mm through target W MCS + Magnet 0/1/2 B_x=+0.57 T field."""
    p_tot_gev = float(np.linalg.norm(mu_p4_gev[:3]))
    p_mev = max(500.0, p_tot_gev * 1000.0)
    tx = float(mu_p4_gev[0] / max(mu_p4_gev[2], 1e-3))
    ty = float(mu_p4_gev[1] / max(mu_p4_gev[2], 1e-3))
    kappa = float(q_mu) * KAPPA_B_CONST_MEV_MM / p_mev  # dy^2/dz^2

    n = len(z_samples_mm)
    xs = np.empty(n, dtype=np.float64)
    ys = np.empty(n, dtype=np.float64)
    txs = np.empty(n, dtype=np.float64)
    tys = np.empty(n, dtype=np.float64)

    cur_z = float(sv_xyz_mm[2])
    cur_x = float(sv_xyz_mm[0])
    cur_y = float(sv_xyz_mm[1])
    cur_tx = tx
    cur_ty = ty
    x0_cur = float(contract.interp_x0(cur_z))

    order = np.argsort(z_samples_mm)
    for idx in order:
        zt = float(z_samples_mm[idx])
        if zt <= cur_z:
            xs[idx], ys[idx], txs[idx], tys[idx] = cur_x, cur_y, cur_tx, cur_ty
            continue
        dz = zt - cur_z
        # Highland MCS across material between cur_z and zt
        x0_next = float(contract.interp_x0(zt))
        dx0 = max(0.0, x0_next - x0_cur)
        if dx0 > 1e-4:
            th_ms = (13.6 / p_mev) * math.sqrt(dx0) * (1.0 + 0.038 * math.log(max(dx0, 1.0)))
            cur_tx += float(rng.normal(0.0, th_ms))
            cur_ty += float(rng.normal(0.0, th_ms))

        # Straight line in X
        cur_x += cur_tx * dz
        # Piecewise integration of B_x = +0.57 T across Magnet0, Magnet1, Magnet2
        cur_y += cur_ty * dz
        for m_lo, m_hi in (
            (MAGNET0_Z_MIN_MM, MAGNET0_Z_MAX_MM),
            (MAGNET1_Z_MIN_MM, MAGNET1_Z_MAX_MM),
            (MAGNET2_Z_MIN_MM, MAGNET2_Z_MAX_MM),
        ):
            a = max(cur_z, m_lo)
            b = min(zt, m_hi)
            if b > a:
                dm = b - a
                # Impulse at midpoint + drift to zt
                d_ty = kappa * dm
                cur_y += 0.5 * kappa * dm * dm + d_ty * (zt - b)
                cur_ty += d_ty

        xs[idx], ys[idx], txs[idx], tys[idx] = cur_x, cur_y, cur_tx, cur_ty
        cur_z = zt
        x0_cur = x0_next

    return xs, ys, txs, tys


def apply_inplace_nc_tau_mu_surgery(
    host_obs: dict[str, Any],
    host_vertex_mm: np.ndarray,
    host_p_had_t_gev: np.ndarray,
    contract: Gen4GeometryContract,
    rng: np.random.Generator,
    *,
    tau_energy_gev: float | None = None,
    decay_length_mm: float | None = None,
    q_mu: int = -1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """In-place factorized NC + (tau -> mu) additive particle surgery.

    Preserves the host NC hadronic shower and asymmetric detector boundaries untouched,
    injects a transverse-momentum-balanced polarized tau -> mu decay at SV = PV + L_tau * u_tau,
    and overlays silicon pixel hits (set union), scintillator bars (exact 14,400-cell
    inverse-saturation multiplicative overlay + earliest time merge), and tracker hits.
    """
    out = _clone_obs(host_obs)
    pv = np.asarray(host_vertex_mm, dtype=np.float64).reshape(3)
    p_ht = np.asarray(host_p_had_t_gev, dtype=np.float64).reshape(2)

    m_tau = 1.77686
    e_tau = float(tau_energy_gev) if tau_energy_gev is not None else float(rng.uniform(35.0, 280.0))
    # Balance transverse momentum against host hadronic recoil: p_T(tau) = -p_T(H) + small k_T kick
    pt_tau_xy = -p_ht + rng.normal(0.0, 0.35, size=2)
    pz_tau = math.sqrt(max(1.0, e_tau * e_tau - m_tau * m_tau - float(np.dot(pt_tau_xy, pt_tau_xy))))
    tau_p4 = np.array([pt_tau_xy[0], pt_tau_xy[1], pz_tau, e_tau], dtype=np.float64)
    u_tau = tau_p4[:3] / np.linalg.norm(tau_p4[:3])

    if decay_length_mm is None:
        # Target short-to-medium decay window [0.8, 32.0] mm where Train32k is starved
        decay_length_mm = float(rng.uniform(0.8, 32.0))
    sv = pv + float(decay_length_mm) * u_tau

    helicity = -1.0 if q_mu < 0 else 1.0
    mu_p4, _ = sample_polarized_tau_to_mu(tau_p4, rng, helicity=helicity)
    p_mu_gev = float(np.linalg.norm(mu_p4[:3]))

    # 1. Propagate parent tau stub from PV to SV across downstream silicon planes in [PV_z, SV_z]
    #    and daughter muon from SV downstream through Target + IPT + Spectrometer
    add_cids: list[int] = []
    add_xyzs: list[np.ndarray] = []

    tau_planes = np.flatnonzero((contract.pixel_Zpos > pv[2]) & (contract.pixel_Zpos <= sv[2]))
    for pl in tau_planes:
        zp = float(contract.pixel_Zpos[pl])
        x_p = pv[0] + (u_tau[0] / u_tau[2]) * (zp - pv[2])
        y_p = pv[1] + (u_tau[1] / u_tau[2]) * (zp - pv[2])
        if contract.px_min < x_p < contract.px_max and contract.py_min < y_p < contract.py_max:
            col = int(np.clip(math.floor((x_p - contract.px_min) / contract.dx_mm), 0, contract.nx - 1))
            row = int(np.clip(math.floor((y_p - contract.py_min) / contract.dy_mm), 0, contract.ny - 1))
            add_cids.append((int(pl) * contract.ny + row) * contract.nx + col)
            add_xyzs.append(np.array([
                contract.px_min + (col + 0.5) * contract.dx_mm,
                contract.py_min + (row + 0.5) * contract.dy_mm,
                zp,
            ], dtype=np.float32))

    # Daughter muon propagation through Silicon planes > sv[2], Scintillator panels > sv[2], and Trackers 0..2
    mu_planes = np.flatnonzero(contract.pixel_Zpos > sv[2])
    mu_panels = np.flatnonzero(np.asarray(contract.raw_geometry['scint_Zpos'], dtype=np.float64) > sv[2])
    z_trk = np.asarray(TRACKER_Z_MM, dtype=np.float64)
    z_all = np.concatenate([
        contract.pixel_Zpos[mu_planes],
        np.asarray(contract.raw_geometry['scint_Zpos'], dtype=np.float64)[mu_panels],
        z_trk,
    ])
    mx, my, _, _ = _propagate_muon_trajectory_yz_bending(sv, mu_p4, q_mu, z_all, contract, rng)
    n_mp = len(mu_planes)
    n_ms = len(mu_panels)

    # Silicon hits for daughter muon (1-pixel + optional edep_mev for 10085-10088 Track-B)
    add_edeps: list[float] = []
    # Match tau stubs above with MIP edep
    for _ in range(len(add_cids)):
        add_edeps.append(float(np.clip(0.00309 * math.exp(float(rng.normal(0.0, 0.35))), 0.001, 0.025)))

    for idx_k, pl in enumerate(mu_planes):
        xp, yp, zp = float(mx[idx_k]), float(my[idx_k]), float(contract.pixel_Zpos[pl])
        if contract.px_min < xp < contract.px_max and contract.py_min < yp < contract.py_max:
            col = int(np.clip(math.floor((xp - contract.px_min) / contract.dx_mm), 0, contract.nx - 1))
            row = int(np.clip(math.floor((yp - contract.py_min) / contract.dy_mm), 0, contract.ny - 1))
            add_cids.append((int(pl) * contract.ny + row) * contract.nx + col)
            add_xyzs.append(np.array([
                contract.px_min + (col + 0.5) * contract.dx_mm,
                contract.py_min + (row + 0.5) * contract.dy_mm,
                zp,
            ], dtype=np.float32))
            add_edeps.append(float(np.clip(0.00309 * math.exp(float(rng.normal(0.0, 0.35))), 0.001, 0.025)))

    if add_cids:
        old_cid = np.asarray(out['pixel']['channel_id'], dtype=np.uint64)
        old_xyz = np.asarray(out['pixel']['xyz_mm'], dtype=np.float32).reshape(-1, 3)
        has_edep = 'edep_mev' in out['pixel'] and len(out['pixel']['edep_mev']) == len(old_cid)
        comb_cid = np.concatenate([old_cid, np.asarray(add_cids, dtype=np.uint64)]) if len(old_cid) else np.asarray(add_cids, dtype=np.uint64)
        comb_xyz = np.vstack([old_xyz, np.asarray(add_xyzs, dtype=np.float32)]) if len(old_xyz) else np.asarray(add_xyzs, dtype=np.float32)
        uniq_cid, inv_idx, uniq_pos = np.unique(comb_cid, return_inverse=True, return_index=True)
        out['pixel']['channel_id'] = uniq_cid
        out['pixel']['xyz_mm'] = comb_xyz[uniq_pos]
        if has_edep:
            old_edep = np.asarray(out['pixel']['edep_mev'], dtype=np.float64)
            comb_edep = np.concatenate([old_edep, np.asarray(add_edeps, dtype=np.float64)]) if len(old_edep) else np.asarray(add_edeps, dtype=np.float64)
            out['pixel']['edep_mev'] = np.bincount(inv_idx, weights=comb_edep, minlength=len(uniq_cid)).astype(np.float32)

    # 2. Scintillator MIP overlay with exact 14,400 inverse-saturation multiplicative formula
    sc_z = np.asarray(contract.raw_geometry['scint_Zpos'], dtype=np.float64)
    p_is_h = contract.bounds_dict['panel_is_horiz']
    # Map panel -> layer (matching readout.py / Geometry)
    bar_dict = out['bar']
    n_scint_tot = len(sc_z)
    # Build existing channel map keyed by (panel, horiz, bar)
    existing_map: dict[tuple[int, int, int], int] = {}
    if len(bar_dict['panel']) > 0:
        for idx_b, (p_b, h_b, b_b) in enumerate(zip(bar_dict['panel'], bar_dict['horizontal'], bar_dict['bar'])):
            existing_map[(int(p_b), int(h_b), int(b_b))] = idx_b

    b_lay = list(np.asarray(bar_dict['layer'], dtype=np.int64))
    b_pan = list(np.asarray(bar_dict['panel'], dtype=np.int64))
    b_hor = list(np.asarray(bar_dict['horizontal'], dtype=np.int64))
    b_bar = list(np.asarray(bar_dict['bar'], dtype=np.int64))
    b_pe = list(np.asarray(bar_dict['pe_proxy'], dtype=np.float64))
    b_ht = list(np.asarray(bar_dict['hit_time_ns'], dtype=np.float64)) if 'hit_time_ns' in bar_dict else [-1.0] * len(b_pan)

    for j, pan_id in enumerate(mu_panels):
        xm, ym = float(mx[n_mp + j]), float(my[n_mp + j])
        if not (contract.scint_x_min <= xm <= contract.scint_x_max and contract.scint_y_min <= ym <= contract.scint_y_max):
            continue
        is_h = int(p_is_h[pan_id])
        if is_h == 0:
            bi_mu = int(np.argmin(np.abs(contract.bar_x_centers_mm - xm)))
        else:
            bi_mu = int(np.argmin(np.abs(contract.bar_y_centers_mm - ym)))
        d_sipm = float(contract.optical_distance_to_sipm(is_h, xm, ym, s_mir=1))
        # MIP dE/dx ~ 1.0 MeV in 5mm BC408 -> ~400 * exp(-d_sipm/3800) PE
        mu_raw_pe = float(rng.poisson(max(1.0, 1.05 * SIPM_ETA0_PE_PER_MEV * math.exp(-d_sipm / BC408_ATTENUATION_MM))))
        mu_sat_pe = SIPM_NPIX * (1.0 - math.exp(-mu_raw_pe / SIPM_NPIX))
        tof_ns = float(np.linalg.norm(np.array([xm, ym, float(sc_z[pan_id])]) - pv) / SPEED_OF_LIGHT_MM_PER_NS)
        mu_t_ns = tof_ns + d_sipm / WLS_GROUP_VELOCITY_MM_PER_NS + 0.05

        key = (int(pan_id), is_h, bi_mu)
        if key in existing_map:
            pos = existing_map[key]
            b_pe[pos] = inverse_sat_additive_pe(b_pe[pos], mu_sat_pe, SIPM_NPIX)
            if mu_sat_pe >= SIPM_TIME_THR_PE:
                b_ht[pos] = mu_t_ns if b_ht[pos] < 0.0 else min(b_ht[pos], mu_t_ns)
        else:
            existing_map[key] = len(b_pan)
            b_lay.append(int(contract.panel_to_station[pan_id]))
            b_pan.append(int(pan_id))
            b_hor.append(is_h)
            b_bar.append(bi_mu)
            b_pe.append(mu_sat_pe)
            b_ht.append(mu_t_ns if mu_sat_pe >= SIPM_TIME_THR_PE else -1.0)

    bar_dict['layer'] = np.asarray(b_lay, dtype=np.int16)
    bar_dict['panel'] = np.asarray(b_pan, dtype=np.int16)
    bar_dict['horizontal'] = np.asarray(b_hor, dtype=np.int8)
    bar_dict['bar'] = np.asarray(b_bar, dtype=np.int16)
    bar_dict['pe_proxy'] = np.clip(np.rint(np.asarray(b_pe)), 0, int(SIPM_NPIX)).astype(np.int64)
    bar_dict['hit_time_ns'] = np.asarray(b_ht, dtype=np.float32)
    _rebuild_bar_channel_ids_and_bounds(bar_dict, contract)

    # 3. Tracker hits (checking bore acceptance R_bore = 100 mm in Magnet0/1/2)
    trk = out['tracker_hypothesis']
    t_st = list(np.asarray(trk['station'], dtype=np.int8))
    t_xy = list(np.asarray(trk['xy_mm'], dtype=np.float32).reshape(-1, 2))
    t_sig = list(np.asarray(trk['sigma_xy_mm'], dtype=np.float32).reshape(-1, 2)) if 'sigma_xy_mm' in trk else []

    in_bore_all = True
    for s_id in range(3):
        tx_s = float(mx[n_mp + n_ms + s_id])
        ty_s = float(my[n_mp + n_ms + s_id])
        r_b = math.hypot(tx_s - MAGNET_BORE_X_MM, ty_s - MAGNET_BORE_Y_MM)
        if r_b > MAGNET_BORE_R_MM:
            in_bore_all = False
            break
        if TRACKER_X_MIN_MM <= tx_s <= TRACKER_X_MAX_MM and TRACKER_Y_MIN_MM <= ty_s <= TRACKER_Y_MAX_MM:
            if rng.random() < 0.99:
                t_st.append(s_id)
                t_xy.append(np.array([
                    tx_s + rng.normal(0.0, SIGMA_X_MM),
                    ty_s + rng.normal(0.0, SIGMA_Y_MM),
                ], dtype=np.float32))
                if len(t_sig) == len(t_st) - 1:
                    t_sig.append(np.array([SIGMA_X_MM, SIGMA_Y_MM], dtype=np.float32))

    trk['station'] = np.asarray(t_st, dtype=np.int8)
    trk['xy_mm'] = np.asarray(t_xy, dtype=np.float32).reshape(-1, 2)
    if t_sig:
        trk['sigma_xy_mm'] = np.asarray(t_sig, dtype=np.float32).reshape(-1, 2)

    surg_meta = {
        'label': 2,
        'label_7': 4,  # tau_mu
        'charm_flag': 0,
        'vertex_xyz_mm': pv.astype(np.float32),
        'true_sv_xyz_mm': sv.astype(np.float32),
        'tau_decay_L3d_mm': float(decay_length_mm),
        'tau_p4_gev': tau_p4.astype(np.float32),
        'mu_p4_gev': mu_p4.astype(np.float32),
        'p_mu_gev': float(p_mu_gev),
        'q_mu': int(q_mu),
        'in_3station_bore': bool(in_bore_all),
        'is_synthetic_surgery': True,
    }
    return out, surg_meta
