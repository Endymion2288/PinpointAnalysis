"""Zero-parameter inverse-physics operators and low-dimensional generative DiffPhys-SBI (Pillar IV).

Implements:
  1. Zero-learnable-parameter 54-station staggered scintillator de-saturation & Rényi time-walk
     unfolding (`ScintStationDesaturator`), aligning exact non-coaxial SiPM coordinates
     (`x_SiPM = -133.0 mm`, `y_SiPM = +198.0 mm`) and staggered single-view stations
     (`S_front = {0..5}`, `S_inter = {12,19,26,33,40,47}`, `S_stereo` 42 stations).
  2. Zero-learnable-parameter Rossi-Greisen X0 analytical-kernel sparse FISTA non-negative
     deconvolution shower stripper (`RossiGreisenFISTAShowerStripper`), stripping `gamma -> e+e-`
     cascade fog while preserving isolated MIP / secondary-prong weights `w_c^MIP`.
  3. Layer-by-layer parallel broken-line Kalman single-track kink likelihood-ratio detector
     `Lambda_kink(k*)` and 54-station calorimeter MCS momentum estimator (`BrokenLineKalmanKinkDetector`),
     embedding exact `B_x = +0.57 T` y-z magnetic deflection.
  4. 32D physical sufficient statistics `z_hat(x)` + analytical uncertainty covariance `Sigma_z(x)`
     and lightweight (~127k param) conditional Rational-Quadratic Neural Spline Flow `DiffPhysSBIModel`
     with covariance marginalization over `Sigma_z(x)`.

Strict Seal & Anti-Leakage Compliance:
  All feature extraction and inference paths consume pure detector observables ONLY.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from analysis.flavour.nextgen.protocols import (
    NextGenFlavourModel,
    NextGenModelOutput,
    marginal_four_from_seven,
    student_t_vertex_loss,
)
from analysis.flavour.nextgen.seal_guard import TRUTH_BATCH_KEYS
from analysis.flavour.v3.common.domains import (
    OBS_DOMAIN_NAMES,
    compute_observable_containment,
    cumulative_radiation_length,
    detector_bounds,
)
from analysis.flavour.v3.v1_b1pp.spectrometer import (
    B_FIELD_TESLA,
    MAGNET0_Z_MAX_MM,
    MAGNET0_Z_MIN_MM,
    MAGNET_BORE_FID_R_MM,
    MAGNET_BORE_X_MM,
    MAGNET_BORE_Y_MM,
    SAGITTA_CONST_MEV_MM,
    SAGITTA_INTEGRAL_MM2,
    SIGMA_X_MM,
    SIGMA_Y_MM,
    TRACKER_Z_MM,
)
from analysis.flavour.v3.common.pixel_clusters import cluster_pixel_hits
from analysis.flavour.v3.v2_pinpointformer.stereo_bars import static_scint_stations
from analysis.flavour.v3.v3_multitask.losses import focal_charm_bce, marginal_tau_branch_ce


# Exact Pinpoint_G4 / PDG 2024 material & detector constants
X0_W_MM = 3.504
R_MOLIERE_W_MM = 9.327
EC_W_MEV = 7.97
SPEED_OF_LIGHT_MM_PER_NS = 299.792458
WLS_V_EFF_MM_PER_NS = 189.742
LAMBDA_ATT_MM = 3800.0
SIPM_NPIX = 14400.0
PE_PER_MEV_CENTER_CAL = 378.514
X_SIPM_YBAR_MM = -133.0
Y_SIPM_XBAR_MM = 198.0

# Exact 54-station staggered topology (verified against Clean v3 geometry.json scint_Zpos)
S_FRONT_SINGLE_X: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
S_INTER_SINGLE_X: tuple[int, ...] = (12, 19, 26, 33, 40, 47)

# 8 fine-grained physical processes for generative density modeling
DIFFPHYS_PROCESS_NAMES: tuple[str, ...] = (
    'nu_e_CC',     # 0 -> label_7 = 0
    'nu_mu_neg',   # 1 -> label_7 = 1
    'nu_mu_pos',   # 2 -> label_7 = 2
    'tau_e',       # 3 -> label_7 = 3
    'tau_mu',      # 4 -> label_7 = 4
    'tau_had',     # 5 -> label_7 = 5
    'NC_light',    # 6 -> label_7 = 6 (charm == 0)
    'Charm',       # 7 -> label_7 = 6 or CC with charm == 1
)

# 32D physical sufficient statistics (strictly within 28D-36D contract)
DIFFPHYS_STAT_NAMES: tuple[str, ...] = (
    'log1p_n_stripped_mip',         # 0
    'em_strip_fraction',            # 1
    'front30_mip_ratio',            # 2
    'mean_bipole_split_score',      # 3
    'sv_lxy_sig',                   # 4
    'sv_m_corr_gev',                # 5
    'log1p_max_kink_llr',           # 6
    'kink_pt_gev',                  # 7
    'max_ip_sig',                   # 8
    'log1p_n_high_ip_mips',         # 9
    'n_sv_prongs_stripped',         # 10
    'max_opening_angle_rad',        # 11
    'log1p_e_vis_desat_gev',        # 12
    'em_core_energy_frac',          # 13
    'shower_max_x0',                # 14
    'longitudinal_width_x0',        # 15
    'transverse_core_rms_mm',       # 16
    'prompt_energy_frac',           # 17
    'log1p_delayed_neutron_gev',    # 18
    'first_active_x0',              # 19
    'log1p_pt_miss_gev',            # 20
    'log1p_mt_gev',                 # 21
    'dphi_lead_had_rad',            # 22
    'pt_balance_asym',              # 23
    'nomad_acoplanarity',           # 24
    'rho_pi0_vis_mass_gev',         # 25
    'n_pen_muons_norm',             # 26
    'mu_q_over_p_signed',           # 27
    'log1p_mu_p_best_gev',          # 28
    'mu_pt_gev',                    # 29
    'mu_kink_compat',               # 30
    'charm_dimuon_veto_score',      # 31
    'has_edep_flag',                # 32 (Mode-B: 10085-10088 single-pixel edep)
    'log1p_mean_mip_edep',          # 33
    'frac_2mip_conversion_clusters',# 34
    'stripped_mip_edep_ratio',      # 35
)
DIFFPHYS_STAT_DIM = len(DIFFPHYS_STAT_NAMES)
PIXEL_MIP_EDEP_KEV = 3.09  # Measured 10085-10088 single-pixel MIP median 3.09 keV


def _highland_sigma_rad(delta_x0: float, p_mev: float) -> float:
    x0 = max(float(delta_x0), 1e-4)
    p = max(float(p_mev), 150.0)
    return (13.6 / p) * math.sqrt(x0) * max(0.2, 1.0 + 0.038 * math.log(x0))


# ===========================================================================
# 1. Zero-Parameter 54-Station Staggered Scintillator De-Saturator
# ===========================================================================
class ScintStationDesaturator:
    """Zero-parameter 8-step damped Newton SiPM de-saturator & Rényi time-walk separator.

    Processes all 54 longitudinal scintillator stations (`S_front={0..5}`, `S_inter={12,19,26,33,40,47}`,
    and 42 `S_stereo` stations) including `< 2.5 PE` halo hits when present in `observations`.
    """

    def __init__(self, geometry: dict[str, Any], bounds: dict[str, Any] | None = None) -> None:
        self.geometry = geometry
        self.bounds = bounds or detector_bounds(geometry)
        self.stations = static_scint_stations(geometry, self.bounds)
        self.n_stations = int(self.stations['n_stations'])  # 54
        self.n_bars = int(self.bounds['n_bars'])             # 40
        self.bx = np.asarray(self.bounds['scint_bar_Xcenter'], dtype=np.float64)
        self.by = np.asarray(self.bounds['scint_bar_Ycenter'], dtype=np.float64)
        self.station_z = np.asarray(self.stations.get('station_z_mm', self.stations.get('station_z')), dtype=np.float64)
        self.panel_to_station = np.asarray(self.stations['panel_to_station'], dtype=np.int64)
        self.panel_is_horiz = np.asarray(self.bounds['panel_is_horiz'], dtype=np.int8)
        self.has_y_station = np.asarray(self.stations['station_has_y'], dtype=bool)
        self.station_x0 = np.array(
            [cumulative_radiation_length(float(z), geometry) for z in self.station_z],
            dtype=np.float64,
        )

    def _time_walk_corrected_dt(
        self,
        t_hit_ns: np.ndarray,
        z_mm: np.ndarray,
        d_sipm_mm: np.ndarray,
        pe_raw: np.ndarray,
    ) -> np.ndarray:
        # Rényi 3rd-PE threshold time-walk correction ~ tau_scint / sqrt(max(pe, 1))
        t_tof = z_mm / SPEED_OF_LIGHT_MM_PER_NS
        t_wls = d_sipm_mm / WLS_V_EFF_MM_PER_NS
        t_walk = 0.85 / np.sqrt(np.maximum(pe_raw, 1.0))
        return t_hit_ns - t_tof - t_wls - t_walk

    def reconstruct(self, bar: Mapping[str, Any], *, min_pe: float = 0.5) -> dict[str, Any]:
        """De-saturate bar PE, separate prompt vs delayed neutron halo, and close 54 stations."""
        pe_raw = np.asarray(bar.get('pe_proxy', ()), dtype=np.float64)
        n_raw = len(pe_raw)
        e_station = np.zeros(self.n_stations, dtype=np.float64)
        e_prompt_st = np.zeros(self.n_stations, dtype=np.float64)
        e_delayed_st = np.zeros(self.n_stations, dtype=np.float64)
        x_st = np.full(self.n_stations, float(self.bounds['x_scint_center']), dtype=np.float64)
        y_st = np.full(self.n_stations, float(self.bounds['y_scint_center']), dtype=np.float64)
        x_var_st = np.full(self.n_stations, (420.0 ** 2) / 12.0, dtype=np.float64)
        y_var_st = np.full(self.n_stations, (420.0 ** 2) / 12.0, dtype=np.float64)
        n_sat_bars = 0

        if n_raw == 0:
            return {
                'e_station_mev': e_station,
                'e_prompt_station_mev': e_prompt_st,
                'e_delayed_station_mev': e_delayed_st,
                'x_station_mm': x_st,
                'y_station_mm': y_st,
                'x_var_station_mm2': x_var_st,
                'y_var_station_mm2': y_var_st,
                'e_vis_gev': 0.0,
                'e_prompt_gev': 0.0,
                'e_delayed_gev': 0.0,
                'e_vis_sigma_gev': 0.5,
                'px_mev': 0.0,
                'py_mev': 0.0,
                'pt_miss_gev': 0.0,
                'pt_miss_sigma_gev': 1.0,
                'n_sat_bars': 0,
            }

        panels = np.asarray(bar['panel'], dtype=np.int64)
        bars_idx = np.asarray(bar['bar'], dtype=np.int64)
        horiz = np.asarray(bar['horizontal'], dtype=np.int8)
        t_hits = np.asarray(bar.get('hit_time_ns', np.zeros(n_raw)), dtype=np.float64)

        keep = pe_raw >= float(min_pe)
        if not keep.any():
            keep = pe_raw > 0.0
        pe_k = np.maximum(pe_raw[keep], 0.0)
        pan_k = np.clip(panels[keep], 0, len(self.panel_to_station) - 1)
        st_k = self.panel_to_station[pan_k]
        b_k = np.clip(bars_idx[keep], 0, self.n_bars - 1)
        h_k = horiz[keep]
        t_k = t_hits[keep]

        # 8-step damped Newton / censored Tobit flank de-saturation per view-station
        occ = np.clip(pe_k / SIPM_NPIX, 0.0, 0.996)
        n_sat_bars = int(np.sum(occ >= 0.95))
        pe_lin = -SIPM_NPIX * np.log(1.0 - occ)

        # Estimate station-level centroid first from unsaturated + linear PE
        st_x_grid = np.zeros((self.n_stations, self.n_bars), dtype=np.float64)
        st_y_grid = np.zeros((self.n_stations, self.n_bars), dtype=np.float64)
        st_x_sat = np.zeros((self.n_stations, self.n_bars), dtype=bool)
        st_y_sat = np.zeros((self.n_stations, self.n_bars), dtype=bool)

        for idx_hit in range(len(pe_lin)):
            s = int(st_k[idx_hit])
            bi = int(b_k[idx_hit])
            if h_k[idx_hit] == 0:
                st_x_grid[s, bi] += pe_lin[idx_hit]
                if occ[idx_hit] >= 0.95:
                    st_x_sat[s, bi] = True
            else:
                st_y_grid[s, bi] += pe_lin[idx_hit]
                if occ[idx_hit] >= 0.95:
                    st_y_sat[s, bi] = True

        # 8-step Cauchy-Molière flank closure for saturated core bars (zero learnable weights)
        def _desat_1d_profile(grid: np.ndarray, sat_mask: np.ndarray, centers: np.ndarray) -> np.ndarray:
            out = grid.copy()
            if not sat_mask.any():
                return out
            half_width_mm = 11.5  # Cauchy/Moliere core scale in W+Scint
            for s_idx in np.flatnonzero(sat_mask.any(axis=1)):
                sat_cols = np.flatnonzero(sat_mask[s_idx])
                unsat_cols = np.flatnonzero((~sat_mask[s_idx]) & (grid[s_idx] > 0.0))
                if len(unsat_cols) >= 2:
                    # Fit core center & amplitude from unsaturated flanks via 8 damped Gauss-Newton steps
                    c_est = float(np.average(centers[sat_cols], weights=np.maximum(grid[s_idx, sat_cols], 1.0)))
                    flank_d2 = (centers[unsat_cols] - c_est) ** 2 + half_width_mm ** 2
                    flank_kern = (half_width_mm ** 2) / flank_d2
                    amp = float(np.sum(grid[s_idx, unsat_cols]) / max(float(np.sum(flank_kern)), 1e-6))
                    for _ in range(8):
                        pred_f = amp * (half_width_mm ** 2) / ((centers[unsat_cols] - c_est) ** 2 + half_width_mm ** 2)
                        res_f = grid[s_idx, unsat_cols] - pred_f
                        d_amp = float(np.mean(res_f / np.maximum(flank_kern, 1e-4)))
                        amp = max(float(grid[s_idx, sat_cols].max()), amp + 0.5 * d_amp)
                    core_pred = amp * (half_width_mm ** 2) / ((centers[sat_cols] - c_est) ** 2 + half_width_mm ** 2)
                    out[s_idx, sat_cols] = np.maximum(out[s_idx, sat_cols], np.minimum(core_pred, 8.0 * SIPM_NPIX))
            return out

        st_x_grid = _desat_1d_profile(st_x_grid, st_x_sat, self.bx)
        st_y_grid = _desat_1d_profile(st_y_grid, st_y_sat, self.by)

        # Solve 2D X/Y station centers first to apply exact light-attenuation & time-walk unfolding
        sum_x = st_x_grid.sum(axis=1)
        sum_y = st_y_grid.sum(axis=1)
        for s in range(self.n_stations):
            if sum_x[s] > 0.0:
                wx = st_x_grid[s] / sum_x[s]
                x_st[s] = float(np.dot(wx, self.bx))
                x_var_st[s] = max(float(np.dot(wx, (self.bx - x_st[s]) ** 2)) + (10.5 ** 2) / 12.0, 9.0)
            if self.has_y_station[s] and sum_y[s] > 0.0:
                wy = st_y_grid[s] / sum_y[s]
                y_st[s] = float(np.dot(wy, self.by))
                y_var_st[s] = max(float(np.dot(wy, (self.by - y_st[s]) ** 2)) + (10.5 ** 2) / 12.0, 9.0)

        # Bidirectional stereo interpolation for interleaved single-view stations S_inter = {12,19,26,33,40,47}
        for s_int in S_INTER_SINGLE_X:
            if s_int < self.n_stations:
                s_lo = max(0, s_int - 1)
                s_hi = min(self.n_stations - 1, s_int + 1)
                w_lo = sum_y[s_lo] if self.has_y_station[s_lo] else 0.0
                w_hi = sum_y[s_hi] if self.has_y_station[s_hi] else 0.0
                if w_lo + w_hi > 0.0:
                    y_st[s_int] = (w_lo * y_st[s_lo] + w_hi * y_st[s_hi]) / (w_lo + w_hi)
                    y_var_st[s_int] = 0.5 * (y_var_st[s_lo] + y_var_st[s_hi]) + 25.0

        # Front 6 single-view stations S_front={0..5}: backward flow from s=6 first stereo station if active
        if sum_y[6] > 0.0:
            for s_f in S_FRONT_SINGLE_X:
                if sum_x[s_f] > 0.0:
                    y_st[s_f] = y_st[6]
                    # Explicit rank-deficient variance for outer X > +133 mm
                    y_var_st[s_f] = y_var_st[6] + (2500.0 if x_st[s_f] > 133.0 else 225.0)

        # Exact SiPM light-attenuation unfolding & prompt/delayed time separation
        # Horizontal Y-bars: SiPM at x_SiPM = -133.0 mm -> d_Y(x) = x + 133.0
        # Vertical X-bars: SiPM at y_SiPM = +198.0 mm -> d_X(y) = 198.0 - y
        # Subtract baseline center attenuation offset exp(-210/3800) used in PE_PER_MEV_CENTER_CAL
        e_hit_mev = np.zeros(len(pe_lin), dtype=np.float64)
        is_delayed = np.zeros(len(pe_lin), dtype=bool)
        for idx_hit in range(len(pe_lin)):
            s = int(st_k[idx_hit])
            bi = int(b_k[idx_hit])
            if h_k[idx_hit] == 0:
                d_sipm = float(np.clip(Y_SIPM_XBAR_MM - y_st[s], 0.0, 420.0))
                pe_ref = st_x_grid[s, bi] if st_x_sat[s, bi] else pe_lin[idx_hit]
            else:
                d_sipm = float(np.clip(x_st[s] - X_SIPM_YBAR_MM, 0.0, 420.0))
                pe_ref = st_y_grid[s, bi] if st_y_sat[s, bi] else pe_lin[idx_hit]
            att_corr = math.exp((d_sipm - 210.0) / LAMBDA_ATT_MM)
            e_mev = (pe_ref * att_corr) / PE_PER_MEV_CENTER_CAL
            e_hit_mev[idx_hit] = e_mev

            dt_ns = float(self._time_walk_corrected_dt(t_k[idx_hit], self.station_z[s], d_sipm, pe_k[idx_hit]))
            delayed = (dt_ns > 1.5) and (dt_ns <= 4360.0)
            is_delayed[idx_hit] = delayed

            if self.has_y_station[s]:
                w_view = 0.5
            else:
                w_view = 1.0
            e_station[s] += w_view * e_mev
            if delayed:
                e_delayed_st[s] += w_view * e_mev
            else:
                e_prompt_st[s] += w_view * e_mev

        e_vis_gev = float(np.sum(e_station)) * 1e-3
        e_prompt_gev = float(np.sum(e_prompt_st)) * 1e-3
        e_delayed_gev = float(np.sum(e_delayed_st)) * 1e-3

        # Transverse momentum balance from prompt stations relative to mean interaction axis
        active_s = np.flatnonzero(e_prompt_st > 1.0)
        if len(active_s) >= 2:
            z0 = float(self.station_z[active_s[0]] - 15.0)
            dz = np.maximum(self.station_z[active_s] - z0, 15.0)
            e_p = e_prompt_st[active_s]
            x_ref = float(np.average(x_st[active_s[:min(3, len(active_s))]], weights=e_p[:min(3, len(active_s))]))
            y_ref = float(np.average(y_st[active_s[:min(3, len(active_s))]], weights=e_p[:min(3, len(active_s))]))
            tx_s = (x_st[active_s] - x_ref) / dz
            ty_s = (y_st[active_s] - y_ref) / dz
            norm_s = np.sqrt(1.0 + tx_s ** 2 + ty_s ** 2)
            px_mev = float(np.sum(e_p * tx_s / norm_s))
            py_mev = float(np.sum(e_p * ty_s / norm_s))
            # Uncertainty propagation including rank-deficient S_front Y variance
            var_px = float(np.sum((e_p / dz) ** 2 * x_var_st[active_s]))
            var_py = float(np.sum((e_p / dz) ** 2 * y_var_st[active_s]))
            pt_miss_gev = math.hypot(px_mev, py_mev) * 1e-3
            pt_sigma_gev = max(0.25, math.sqrt(var_px + var_py) * 1e-3)
        else:
            px_mev, py_mev = 0.0, 0.0
            pt_miss_gev, pt_sigma_gev = 0.0, 1.0

        e_sigma_gev = max(0.35, 0.08 * e_vis_gev + 0.06 * n_sat_bars * max(e_vis_gev, 5.0))
        return {
            'e_station_mev': e_station,
            'e_prompt_station_mev': e_prompt_st,
            'e_delayed_station_mev': e_delayed_st,
            'x_station_mm': x_st,
            'y_station_mm': y_st,
            'x_var_station_mm2': x_var_st,
            'y_var_station_mm2': y_var_st,
            'e_vis_gev': e_vis_gev,
            'e_prompt_gev': e_prompt_gev,
            'e_delayed_gev': e_delayed_gev,
            'e_vis_sigma_gev': e_sigma_gev,
            'px_mev': px_mev,
            'py_mev': py_mev,
            'pt_miss_gev': pt_miss_gev,
            'pt_miss_sigma_gev': pt_sigma_gev,
            'n_sat_bars': n_sat_bars,
        }


# ===========================================================================
# 2. Zero-Parameter Rossi-Greisen X0 Sparse FISTA Shower Stripper
# ===========================================================================
class RossiGreisenFISTAShowerStripper:
    """Zero-parameter analytical X0 Rossi-Greisen + V0 sparse FISTA non-negative deconvolution.

    Solves `min_{alpha >= 0} 0.5 * ||y - A alpha||_2^2 + lambda_l1 * ||alpha||_1` via 10 FISTA
    iterations to estimate the electromagnetic conversion/cascade fog density `rho_EM(c)` at every
    silicon cluster `c` and assign isolated MIP / secondary-prong posterior weights:
        `w_c^MIP = rho_MIP0 / (rho_EM(c) + rho_MIP0)`.
    """

    def __init__(
        self,
        geometry: dict[str, Any],
        *,
        n_fista_steps: int = 10,
        lambda_l1: float = 0.12,
        rho_mip0: float = 0.85,
        max_sources: int = 48,
        pixel_x0: np.ndarray | None = None,
    ) -> None:
        self.geometry = geometry
        self.n_fista_steps = int(n_fista_steps)
        self.lambda_l1 = float(lambda_l1)
        self.rho_mip0 = float(rho_mip0)
        self.max_sources = int(max_sources)
        self.pz = np.asarray(geometry['pixel_Zpos'], dtype=np.float64)
        if pixel_x0 is not None:
            self.plane_x0 = np.asarray(pixel_x0, dtype=np.float64)
        else:
            self.plane_x0 = np.array([cumulative_radiation_length(float(z), geometry) for z in self.pz], dtype=np.float64)

    def strip(
        self,
        pix_xyz_mm: np.ndarray,
        pix_plane: np.ndarray,
        bipole_scores: np.ndarray | None = None,
        cluster_n_hits: np.ndarray | None = None,
        cluster_edep_mip: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute per-cluster EM cascade density `rho_em` and isolated MIP weight `w_mip`."""
        xyz = np.asarray(pix_xyz_mm, dtype=np.float64)
        pl = np.asarray(pix_plane, dtype=np.int64)
        n_cl = len(xyz)
        if n_cl == 0:
            return {
                'w_mip': np.empty(0, dtype=np.float64),
                'rho_em': np.empty(0, dtype=np.float64),
                'local_density': np.empty(0, dtype=np.float64),
                'alpha_sources': np.empty(0, dtype=np.float64),
            }

        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]
        x0_c = self.plane_x0[np.clip(pl, 0, len(self.plane_x0) - 1)]

        # Local intra-plane transverse density proxy y_obs (high in dense EM showers, ~0 for isolated MIPs)
        y_local = np.zeros(n_cl, dtype=np.float64)
        for p_id in np.unique(pl):
            idx = np.flatnonzero(pl == p_id)
            if len(idx) <= 1:
                continue
            dx = x[idx, None] - x[None, idx]
            dy = y[idx, None] - y[None, idx]
            r2 = dx * dx + dy * dy
            # Neighbor kernel at 2.8 mm inside W+Si sandwich excluding self
            k_mat = np.exp(-r2 / (2.0 * (2.8 ** 2)))
            np.fill_diagonal(k_mat, 0.0)
            y_local[idx] = k_mat.sum(axis=1)

        # Subtract isolated 1-neighbor baseline so 2-3 prong MIP vertices do not self-strip
        y_obs = np.maximum(y_local - 0.85, 0.0)
        nh_arr = np.asarray(cluster_n_hits, dtype=np.float64) if (cluster_n_hits is not None and len(cluster_n_hits) == n_cl) else None
        if nh_arr is not None:
            y_obs = y_obs + 0.28 * np.maximum(nh_arr - 4.0, 0.0)
        if bipole_scores is not None and len(bipole_scores) == n_cl:
            bip_arr = np.clip(np.asarray(bipole_scores, dtype=np.float64), 0.0, 2.0)
            if nh_arr is not None:
                bip_arr = np.where(nh_arr >= 3.5, bip_arr, 0.0)
            y_obs = y_obs + 0.65 * bip_arr
        if cluster_edep_mip is not None and len(cluster_edep_mip) == n_cl:
            ed_mip = np.maximum(np.asarray(cluster_edep_mip, dtype=np.float64), 0.0)
            # Mode-B 10085-10088: 2-MIP merged e+e- conversion pairs (>= 1.85 MIP, above 1-MIP Landau tail)
            y_obs = y_obs + 0.85 * np.clip(ed_mip - 1.75, 0.0, 2.5)

        if float(y_obs.max()) < 0.25 or n_cl < 3:
            w_mip = self.rho_mip0 / (y_obs + self.rho_mip0)
            return {
                'w_mip': w_mip,
                'rho_em': y_obs,
                'local_density': y_obs,
                'alpha_sources': np.zeros(0, dtype=np.float64),
            }

        # Select candidate EM conversion / cascade source anchors (high local density, non-IPT target planes)
        tgt_mask = pl < 12
        cand_idx = np.flatnonzero(tgt_mask & (y_obs >= 0.20))
        if len(cand_idx) == 0:
            cand_idx = np.flatnonzero(tgt_mask)
        if len(cand_idx) == 0:
            cand_idx = np.arange(n_cl)
        if len(cand_idx) > self.max_sources:
            top_ord = np.argsort(-y_obs[cand_idx], kind='mergesort')[:self.max_sources]
            cand_idx = cand_idx[top_ord]

        # Estimate primary shower axis from density-weighted first/last active plane centroids
        p_act = np.unique(pl[tgt_mask])
        if len(p_act) >= 2:
            p0_m = pl == p_act[0]
            p1_m = pl == p_act[-1]
            w0 = y_obs[p0_m] + 0.1
            w1 = y_obs[p1_m] + 0.1
            dz_ax = max(float(z[p1_m].mean() - z[p0_m].mean()), 10.0)
            tx_ax = (float(np.average(x[p1_m], weights=w1)) - float(np.average(x[p0_m], weights=w0))) / dz_ax
            ty_ax = (float(np.average(y[p1_m], weights=w1)) - float(np.average(y[p0_m], weights=w0))) / dz_ax
            tx_ax = float(np.clip(tx_ax, -0.25, 0.25))
            ty_ax = float(np.clip(ty_ax, -0.25, 0.25))
        else:
            tx_ax, ty_ax = 0.0, 0.0

        # Build analytical Rossi-Greisen EM + tight V0 pair dictionary matrix A: (N_cl, N_src)
        dz_ca = z[:, None] - z[None, cand_idx]
        dt_ca = x0_c[:, None] - x0_c[None, cand_idx]
        causal = (dz_ca >= -0.5) & (dt_ca >= -0.05) & (dt_ca <= 28.0)

        dx_proj = (x[:, None] - x[None, cand_idx]) - tx_ax * np.maximum(dz_ca, 0.0)
        dy_proj = (y[:, None] - y[None, cand_idx]) - ty_ax * np.maximum(dz_ca, 0.0)
        r_perp = np.hypot(dx_proj, dy_proj)

        # Rossi-Greisen longitudinal profile (alpha=1.6, beta=0.50 in X0 units) + immediate V0 pair
        dt_pos = np.maximum(dt_ca, 0.0)
        long_rg = ((dt_pos + 0.35) ** 1.6) * np.exp(-0.48 * dt_pos)
        long_rg = long_rg / max(float(long_rg.max()), 1.0)
        sigma_em = np.sqrt(0.35 ** 2 + (0.28 * R_MOLIERE_W_MM * np.minimum(1.0, (dt_pos + 0.2) / 3.5)) ** 2)
        k_em = long_rg * np.exp(-r_perp / sigma_em)
        # Immediate conversion V0 core (same or next plane)
        is_v0_zone = (dt_ca >= -0.05) & (dt_ca <= 3.5)
        k_v0 = np.where(is_v0_zone, 0.85 * np.exp(-(r_perp ** 2) / (2.0 * (0.45 ** 2))), 0.0)

        A = np.where(causal, k_em + k_v0, 0.0)

        # Exact Lipschitz upper bound via 4 power iterations on A^T A
        n_src = A.shape[1]
        v = np.ones(n_src, dtype=np.float64) / math.sqrt(max(n_src, 1))
        L = 1.0
        for _ in range(4):
            Av = A @ v
            AtAv = A.T @ Av
            norm_v = float(np.linalg.norm(AtAv))
            if norm_v <= 1e-12:
                L = 1.0
                break
            v = AtAv / norm_v
            L = norm_v
        eta = 1.0 / max(L * 1.05, 1e-6)

        # 10-step FISTA non-negative sparse deconvolution
        alpha = np.zeros(n_src, dtype=np.float64)
        z_fista = np.zeros(n_src, dtype=np.float64)
        t_fista = 1.0
        At = A.T
        thresh = eta * self.lambda_l1
        for _ in range(self.n_fista_steps):
            grad = At @ (A @ z_fista - y_obs)
            alpha_next = np.maximum(0.0, z_fista - eta * grad - thresh)
            t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t_fista * t_fista))
            z_fista = alpha_next + ((t_fista - 1.0) / t_next) * (alpha_next - alpha)
            alpha = alpha_next
            t_fista = t_next

        rho_em = np.maximum(A @ alpha, 0.0)
        # Mild local self-density blend to guard isolated dense cores not captured by sparse anchors
        rho_eff = np.maximum(rho_em, 0.35 * y_obs)
        w_mip = self.rho_mip0 / (rho_eff + self.rho_mip0)
        return {
            'w_mip': w_mip,
            'rho_em': rho_eff,
            'local_density': y_obs,
            'alpha_sources': alpha,
        }


# ===========================================================================
# 3. Broken-Line Kalman Kink Detector & 54-Station Muon MCS Estimator
# ===========================================================================
class BrokenLineKalmanKinkDetector:
    """Layer-by-layer parallel broken-line Kalman kink LR detector Lambda_kink(k*) + MCS momentum.

    Embeds exact `B_x = +0.57 T` y-z magnetic deflection and Highland scattering matrices across
    `Front30` (2.97 X0), `Interm150` (20.23 X0), and `IPT` (1e-4 X0 air gap).
    """

    def __init__(
        self,
        geometry: dict[str, Any],
        bounds: dict[str, Any] | None = None,
        *,
        pixel_x0: np.ndarray | None = None,
    ) -> None:
        self.geometry = geometry
        self.bounds = bounds or detector_bounds(geometry)
        self.pz = np.asarray(geometry['pixel_Zpos'], dtype=np.float64)
        if pixel_x0 is not None:
            self.plane_x0 = np.asarray(pixel_x0, dtype=np.float64)
        else:
            self.plane_x0 = np.array([cumulative_radiation_length(float(z), geometry) for z in self.pz], dtype=np.float64)
        self.sig_x_mm = 0.0208 / math.sqrt(12.0)
        self.sig_y_mm = 0.0228 / math.sqrt(12.0)

    def _estimate_spectrometer_muons(
        self,
        trk_hyp: Mapping[str, Any],
        ipt_xyz: np.ndarray,
    ) -> dict[str, Any]:
        """Unified M1 (3-station sagitta) + M2 (IPT+Magnet0+Tracker1) muon momentum & charge."""
        stations = np.asarray(trk_hyp.get('station', ()), dtype=np.int64)
        xy = np.asarray(trk_hyp.get('xy_mm', np.empty((0, 2))), dtype=np.float64)
        if len(stations) == 0 or len(xy) == 0:
            return {'n_mu_spec': 0, 'q_over_p': 0.0, 'p_gev': 0.0, 'pt_gev': 0.0, 'tx': 0.0, 'ty': 0.0, 'has_m1': False, 'has_m2': False}

        s0 = xy[stations == 0]
        s1 = xy[stations == 1]
        s2 = xy[stations == 2]

        # M1: 3-station bore-fiducial trajectories
        best_chi2 = 1e18
        best_m1 = None
        if len(s0) > 0 and len(s1) > 0 and len(s2) > 0:
            dz_02 = TRACKER_Z_MM[2] - TRACKER_Z_MM[0]
            for p0 in s0[:6]:
                for p2 in s2[:6]:
                    tx = (p2[0] - p0[0]) / dz_02
                    xm = 0.5 * (p0[0] + p2[0])
                    ym_line = 0.5 * (p0[1] + p2[1])
                    for p1 in s1[:6]:
                        dx_m = p1[0] - xm
                        sy = p1[1] - ym_line
                        chi2_x = (dx_m / (math.sqrt(1.5) * SIGMA_X_MM)) ** 2
                        if chi2_x > 25.0 or abs(sy) > 45.0:
                            continue
                        kappa = -sy / SAGITTA_INTEGRAL_MM2
                        ty0 = (p1[1] - p0[1]) / 1190.0 + (60.0 / 59.0) * (sy / 1190.0)
                        # Bore check around (0, -10) mm
                        r_mid = math.hypot(xm - MAGNET_BORE_X_MM, p1[1] - MAGNET_BORE_Y_MM)
                        if r_mid > MAGNET_BORE_FID_R_MM:
                            continue
                        if chi2_x < best_chi2:
                            best_chi2 = chi2_x
                            p_mev = SAGITTA_CONST_MEV_MM / max(abs(sy), 1e-3)
                            q = -1.0 if kappa < 0.0 else (1.0 if kappa > 0.0 else 0.0)
                            # Approximate number of distinct muon tracks at station 0/2
                            n_trk = min(3, min(len(s0), len(s2)))
                            best_m1 = {
                                'n_mu_spec': int(n_trk),
                                'q_over_p': float(q / max(p_mev * 1e-3, 0.5)),
                                'p_gev': float(np.clip(p_mev * 1e-3, 0.5, 2500.0)),
                                'pt_gev': float(np.clip(p_mev * 1e-3 * math.hypot(tx, ty0), 0.0, 200.0)),
                                'tx': float(tx),
                                'ty': float(ty0),
                                'has_m1': True,
                                'has_m2': False,
                            }
        if best_m1 is not None:
            return best_m1

        # M2: IPT + Magnet-0 + Tracker-1 soft muon deflection lever arm (221.1 GeV*mm)
        # Deflection across Magnet-0 [z_min=-1565.3, z_max=-65.3] (L=1500 mm) observed at z_trk0=+47.4 mm:
        # y(z_trk0) - [y_ipt + ty_ipt * (z_trk0 - z_ipt)] = kappa * L_mag * (z_trk0 - z_mag_mid)
        if len(s0) > 0 and len(ipt_xyz) >= 2:
            z_mag_mid = 0.5 * (MAGNET0_Z_MIN_MM + MAGNET0_Z_MAX_MM)
            l_mag = MAGNET0_Z_MAX_MM - MAGNET0_Z_MIN_MM
            lever_mm2 = l_mag * (TRACKER_Z_MM[0] - z_mag_mid)  # ~1,294,050 mm^2
            # Pair IPT clusters on different planes
            z_i = ipt_xyz[:, 2]
            best_m2_chi2 = 1e18
            best_m2 = None
            for i in range(min(len(ipt_xyz), 16)):
                for j in range(min(len(ipt_xyz), 16)):
                    dz_ij = z_i[j] - z_i[i]
                    if dz_ij < 5.0:
                        continue
                    tx_i = (ipt_xyz[j, 0] - ipt_xyz[i, 0]) / dz_ij
                    ty_i = (ipt_xyz[j, 1] - ipt_xyz[i, 1]) / dz_ij
                    if math.hypot(tx_i, ty_i) > 0.25:
                        continue
                    x_pred0 = ipt_xyz[i, 0] + tx_i * (TRACKER_Z_MM[0] - z_i[i])
                    y_line0 = ipt_xyz[i, 1] + ty_i * (TRACKER_Z_MM[0] - z_i[i])
                    for p0 in s0[:6]:
                        dx_m2 = p0[0] - x_pred0
                        chi2_x = (dx_m2 / 2.5) ** 2
                        if chi2_x > 16.0:
                            continue
                        dy_bend = p0[1] - y_line0
                        kappa = dy_bend / lever_mm2
                        p_gev = (0.299792458 * B_FIELD_TESLA / max(abs(kappa), 1e-8)) * 1e-3
                        if 1.0 <= p_gev <= 300.0 and chi2_x < best_m2_chi2:
                            best_m2_chi2 = chi2_x
                            q = -1.0 if kappa < 0.0 else 1.0
                            best_m2 = {
                                'n_mu_spec': 1,
                                'q_over_p': float(q / max(p_gev, 0.5)),
                                'p_gev': float(p_gev),
                                'pt_gev': float(p_gev * math.hypot(tx_i, ty_i)),
                                'tx': float(tx_i),
                                'ty': float(ty_i),
                                'has_m1': False,
                                'has_m2': True,
                            }
            if best_m2 is not None:
                return best_m2

        return {'n_mu_spec': 0, 'q_over_p': 0.0, 'p_gev': 0.0, 'pt_gev': 0.0, 'tx': 0.0, 'ty': 0.0, 'has_m1': False, 'has_m2': False}

    def analyze(
        self,
        pix_xyz_mm: np.ndarray,
        pix_plane: np.ndarray,
        w_mip: np.ndarray,
        trk_hyp: Mapping[str, Any],
        desat_cal: Mapping[str, Any],
        *,
        mip_keep_threshold: float = 0.35,
    ) -> dict[str, Any]:
        """Run broken-line Kalman kink LR & 54-station calorimeter MCS analysis on stripped MIP clusters."""
        xyz = np.asarray(pix_xyz_mm, dtype=np.float64)
        pl = np.asarray(pix_plane, dtype=np.int64)
        wm = np.asarray(w_mip, dtype=np.float64)
        ipt_xyz = xyz[pl >= 12] if len(xyz) else np.empty((0, 3), dtype=np.float64)
        spec_mu = self._estimate_spectrometer_muons(trk_hyp, ipt_xyz)

        # Estimate penetrating muon via 54-station calorimeter MIP tail (M4) if spectrometer silent
        e_st = np.asarray(desat_cal['e_station_mev'], dtype=np.float64)
        active_st = np.flatnonzero(e_st > 0.8)
        n_tail_st = int(np.sum(e_st[30:] > 1.2)) if len(e_st) >= 54 else 0
        has_cal_mu = (spec_mu['n_mu_spec'] > 0) or (n_tail_st >= 6)
        n_pen_mu = int(spec_mu['n_mu_spec']) if spec_mu['n_mu_spec'] > 0 else (1 if has_cal_mu else 0)

        p_best_gev = float(spec_mu['p_gev'])
        if p_best_gev <= 0.0 and has_cal_mu:
            # M4: 54-station calorimeter Highland MCS momentum estimator
            x_st = np.asarray(desat_cal['x_station_mm'], dtype=np.float64)
            z_st = self.geometry['scint_Zpos']
            # Station z approximate centers for active tail
            st_z_arr = np.linspace(float(min(z_st)), float(max(z_st)), len(e_st))
            if len(active_st) >= 5:
                xs = x_st[active_st]
                zs = st_z_arr[active_st]
                slopes = np.diff(xs) / np.maximum(np.diff(zs), 10.0)
                d_sl = np.diff(slopes)
                rms_theta = max(float(np.sqrt(np.mean(d_sl * d_sl))), 1e-4) if len(d_sl) > 0 else 0.02
                # ~3.0 X0 between stations
                p_best_gev = float(np.clip((13.6e-3 * math.sqrt(3.0)) / rms_theta, 2.0, 250.0))
            else:
                p_best_gev = 12.0

        p_ref_mev = max(p_best_gev * 1e3, 8000.0)

        # Filter to isolated MIP clusters (preserving secondary prongs)
        keep_m = wm >= float(mip_keep_threshold)
        if int(keep_m.sum()) < 2 and len(xyz) >= 2:
            top_k = np.argsort(-wm, kind='mergesort')[:min(len(xyz), 24)]
            keep_m = np.zeros(len(xyz), dtype=bool)
            keep_m[top_k] = True

        xyz_s = xyz[keep_m]
        pl_s = pl[keep_m]
        wm_s = wm[keep_m]

        if len(xyz_s) < 3:
            return {
                'max_kink_llr': 0.0,
                'kink_pt_gev': 0.0,
                'max_ip_sig': 0.0,
                'n_high_ip_mips': 0,
                'sv_lxy_sig': 0.0,
                'sv_m_corr_gev': 0.0,
                'n_sv_prongs': 0,
                'max_opening_angle_rad': 0.0,
                'pv_xyz_mm': np.array([0.0, -12.0, -2950.0], dtype=np.float64),
                'pv_sigma_mm': np.array([15.0, 15.0, 25.0], dtype=np.float64),
                'n_pen_muons': n_pen_mu,
                'mu_q_over_p': float(spec_mu['q_over_p']),
                'mu_p_best_gev': p_best_gev,
                'mu_pt_gev': float(spec_mu['pt_gev']),
                'mu_kink_compat': 0.0,
                'n_stripped_mip': float(np.sum(wm)),
            }

        # Build clean non-exclusive 2-hit/3-hit tracklets and chains across stripped clusters
        # Group by plane (keep top 24 highest w_mip clusters per plane for fast O(N) broken-line)
        planes_act = sorted(np.unique(pl_s).tolist())
        by_p: dict[int, np.ndarray] = {}
        for p_id in planes_act:
            idx = np.flatnonzero(pl_s == p_id)
            if len(idx) > 24:
                idx = idx[np.argsort(-wm_s[idx], kind='mergesort')[:24]]
            by_p[p_id] = idx

        # Robust PV estimate from first active silicon plane median (immune to shower outliers)
        p0 = planes_act[0]
        i0 = by_p[p0]
        w0 = wm_s[i0]
        pv_x = float(np.median(xyz_s[i0, 0]))
        pv_y = float(np.median(xyz_s[i0, 1]))
        pv_z = float(xyz_s[i0[0], 2] - (10.0 if p0 < 6 else 18.0))
        pv_xyz = np.array([pv_x, pv_y, pv_z], dtype=np.float64)
        pv_xy_unc = (0.25 if p0 < 6 else 0.85) + 0.15 * max(len(i0) - 2, 0)

        # Layer-by-layer parallel broken-line Kalman kink detection across triples (p_a -> p_b -> p_c)
        max_kink_llr = 0.0
        kink_pt_gev = 0.0
        max_ip_sig = 0.0
        high_ip_mips = 0
        seg_vecs: list[tuple[np.ndarray, np.ndarray, float]] = []  # (origin_xyz, slope_xy, ip_sig)

        for idx_p in range(len(planes_act) - 2):
            pa, pb, pc = planes_act[idx_p], planes_act[idx_p + 1], planes_act[idx_p + 2]
            if (pa < 12) != (pc < 12):
                continue
            ia, ib, ic = by_p[pa], by_p[pb], by_p[pc]
            dz_ab = float(self.pz[pb] - self.pz[pa])
            dz_bc = float(self.pz[pc] - self.pz[pb])
            if dz_ab <= 1.0 or dz_bc <= 1.0:
                continue
            dx0_ab = max(1e-4, float(self.plane_x0[pb] - self.plane_x0[pa]))
            dx0_bc = max(1e-4, float(self.plane_x0[pc] - self.plane_x0[pb]))
            th_ms_b = _highland_sigma_rad(0.5 * (dx0_ab + dx0_bc), p_ref_mev)
            var_tx_b = (self.sig_x_mm ** 2) * (1.0 / (dz_ab ** 2) + 1.0 / (dz_bc ** 2) + 2.0 / (dz_ab * dz_bc)) + th_ms_b ** 2
            var_ty_b = (self.sig_y_mm ** 2) * (1.0 / (dz_ab ** 2) + 1.0 / (dz_bc ** 2) + 2.0 / (dz_ab * dz_bc)) + th_ms_b ** 2

            # Vectorized search over (a, b, c) triples within max slope 0.25
            xa, ya = xyz_s[ia, 0], xyz_s[ia, 1]
            xb, yb = xyz_s[ib, 0], xyz_s[ib, 1]
            xc, yc = xyz_s[ic, 0], xyz_s[ic, 1]

            tx_ab = (xb[None, :] - xa[:, None]) / dz_ab
            ty_ab = (yb[None, :] - ya[:, None]) / dz_ab
            sl_ab_ok = (np.hypot(tx_ab, ty_ab) <= 0.25)

            tx_bc = (xc[None, :] - xb[:, None]) / dz_bc
            ty_bc = (yc[None, :] - yb[:, None]) / dz_bc
            sl_bc_ok = (np.hypot(tx_bc, ty_bc) <= 0.25)

            for a_i in range(len(ia)):
                b_cands = np.flatnonzero(sl_ab_ok[a_i])
                for b_i in b_cands:
                    c_cands = np.flatnonzero(sl_bc_ok[b_i])
                    if len(c_cands) == 0:
                        continue
                    dtx = tx_bc[b_i, c_cands] - tx_ab[a_i, b_i]
                    dty = ty_bc[b_i, c_cands] - ty_ab[a_i, b_i]
                    kink_chi2 = (dtx * dtx) / var_tx_b + (dty * dty) / var_ty_b
                    # Require physical kink angle > 1.5 mrad and < 180 mrad (suppress accidental huge cross-matches)
                    dth = np.hypot(dtx, dty)
                    # Best downstream match (smallest chi2 if straight, or distinct kink candidate in [0.002, 0.16])
                    min_idx = int(np.argmin(kink_chi2))
                    # Check if best c has a clean kink (2..160 mrad) and both b,c have high MIP weight
                    w_triple = float(min(wm_s[ia[a_i]], wm_s[ib[b_i]], wm_s[ic[c_cands[min_idx]]]))
                    if w_triple >= 0.45 and 0.002 <= float(dth[min_idx]) <= 0.16:
                        llr_val = float(kink_chi2[min_idx]) * w_triple
                        if llr_val > max_kink_llr:
                            max_kink_llr = llr_val
                            kink_pt_gev = (p_ref_mev * 1e-3) * float(dth[min_idx])

                    # Also record downstream segments (pb -> pc) for IP & SV calculation
                    best_c = int(c_cands[min_idx])
                    sl_seg = np.array([tx_bc[b_i, best_c], ty_bc[b_i, best_c]], dtype=np.float64)
                    org = xyz_s[ib[b_i]]
                    # Project back to pv_z
                    dz_pv = pv_z - org[2]
                    ip_x = (org[0] + sl_seg[0] * dz_pv) - pv_x
                    ip_y = (org[1] + sl_seg[1] * dz_pv) - pv_y
                    ip_d = math.hypot(ip_x, ip_y)
                    sig_ip = math.hypot(pv_xy_unc, th_ms_b * abs(dz_pv)) + 0.05
                    ip_sig = ip_d / sig_ip
                    if w_triple >= 0.45:
                        seg_vecs.append((org, sl_seg, ip_sig))
                        if ip_sig > max_ip_sig:
                            max_ip_sig = ip_sig
                        if ip_sig > 3.0:
                            high_ip_mips += 1

        # Also check 2-plane Front30 segments if <3 planes produced segments
        if len(seg_vecs) < 2 and len(planes_act) >= 2:
            for idx_p in range(min(3, len(planes_act) - 1)):
                pa, pb = planes_act[idx_p], planes_act[idx_p + 1]
                ia, ib = by_p[pa], by_p[pb]
                dz_ab = float(self.pz[pb] - self.pz[pa])
                if dz_ab <= 1.0:
                    continue
                th_ms = _highland_sigma_rad(max(1e-4, float(self.plane_x0[pb] - self.plane_x0[pa])), p_ref_mev)
                for a_i in ia[:10]:
                    for b_i in ib[:10]:
                        tx = (xyz_s[b_i, 0] - xyz_s[a_i, 0]) / dz_ab
                        ty = (xyz_s[b_i, 1] - xyz_s[a_i, 1]) / dz_ab
                        if math.hypot(tx, ty) > 0.25:
                            continue
                        w_p = min(wm_s[a_i], wm_s[b_i])
                        if w_p < 0.45:
                            continue
                        dz_pv = pv_z - xyz_s[a_i, 2]
                        ip_d = math.hypot(xyz_s[a_i, 0] + tx * dz_pv - pv_x, xyz_s[a_i, 1] + ty * dz_pv - pv_y)
                        ip_sig = ip_d / (math.hypot(pv_xy_unc, th_ms * abs(dz_pv)) + 0.05)
                        seg_vecs.append((xyz_s[a_i], np.array([tx, ty]), ip_sig))
                        max_ip_sig = max(max_ip_sig, ip_sig)
                        if ip_sig > 3.0:
                            high_ip_mips += 1

        # Compute SV displacement significance, corrected mass M_corr, and opening angle from displaced segments
        disp_segs = [sv for sv in seg_vecs if sv[2] > 2.5]
        max_open = 0.0
        sv_lxy_sig = 0.0
        sv_m_corr = 0.0
        if len(disp_segs) >= 2:
            # Pairwise closest-approach vertex among displaced segments
            best_lxy_sig = 0.0
            for u_i in range(min(len(disp_segs), 8)):
                o1, s1, _ = disp_segs[u_i]
                for v_i in range(u_i + 1, min(len(disp_segs), 8)):
                    o2, s2, _ = disp_segs[v_i]
                    d_ang = math.hypot(s1[0] - s2[0], s1[1] - s2[1])
                    max_open = max(max_open, d_ang)
                    # Solve closest approach z_sv
                    dv_x, dv_y = s1[0] - s2[0], s1[1] - s2[1]
                    denom = dv_x * dv_x + dv_y * dv_y
                    if denom > 1e-8:
                        dx0 = (o2[0] - s2[0] * o2[2]) - (o1[0] - s1[0] * o1[2])
                        dy0 = (o2[1] - s2[1] * o2[2]) - (o1[1] - s1[1] * o1[2])
                        z_sv = float(np.clip(-(dx0 * dv_x + dy0 * dv_y) / denom, pv_z, pv_z + 180.0))
                        x_sv = 0.5 * ((o1[0] + s1[0] * (z_sv - o1[2])) + (o2[0] + s2[0] * (z_sv - o2[2])))
                        y_sv = 0.5 * ((o1[1] + s1[1] * (z_sv - o1[2])) + (o2[1] + s2[1] * (z_sv - o2[2])))
                        lxy = math.hypot(x_sv - pv_x, y_sv - pv_y)
                        sig_lxy = 0.045 + 0.002 * max(z_sv - pv_z, 0.0)
                        best_lxy_sig = max(best_lxy_sig, lxy / sig_lxy)
                        # Approximate invariant + pT-corrected mass for pion prongs
                        m_vis = math.sqrt(max(0.0, 2.0 * 0.1396 ** 2 + 2.0 * (10.0 * 10.0) * (1.0 - math.cos(d_ang))))
                        pt_rel = 10.0 * d_ang * 0.5
                        sv_m_corr = max(sv_m_corr, math.sqrt(m_vis * m_vis + pt_rel * pt_rel) + pt_rel)
            sv_lxy_sig = best_lxy_sig

        # Spectrometer muon compatibility with kink/target segments
        mu_compat = 0.0
        if spec_mu['n_mu_spec'] > 0 and len(seg_vecs) > 0:
            min_d = min(math.hypot(s[0] - spec_mu['tx'], s[1] - spec_mu['ty']) for _, s, _ in seg_vecs)
            mu_compat = math.exp(-min_d / 0.03)

        # MCS uncertainty scale on PV in Interm150 vs Front30
        pv_sig_xy = 0.08 if p0 < 6 else 0.45
        pv_sig_z = 1.44 if p0 < 6 else 6.0
        return {
            'max_kink_llr': float(np.clip(max_kink_llr, 0.0, 250.0)),
            'kink_pt_gev': float(np.clip(kink_pt_gev, 0.0, 10.0)),
            'max_ip_sig': float(np.clip(max_ip_sig, 0.0, 50.0)),
            'n_high_ip_mips': int(high_ip_mips),
            'sv_lxy_sig': float(np.clip(sv_lxy_sig, 0.0, 40.0)),
            'sv_m_corr_gev': float(np.clip(sv_m_corr, 0.0, 8.0)),
            'n_sv_prongs': int(min(len(disp_segs), 8)),
            'max_opening_angle_rad': float(np.clip(max_open, 0.0, 0.5)),
            'pv_xyz_mm': pv_xyz,
            'pv_sigma_mm': np.array([pv_sig_xy, pv_sig_xy, pv_sig_z], dtype=np.float64),
            'n_pen_muons': n_pen_mu,
            'mu_q_over_p': float(spec_mu['q_over_p']),
            'mu_p_best_gev': float(p_best_gev),
            'mu_pt_gev': float(spec_mu['pt_gev']),
            'mu_kink_compat': float(mu_compat),
            'n_stripped_mip': float(np.sum(wm)),
        }


# ===========================================================================
# 4. 32D Sufficient Statistics + Analytical Covariance Extractor
# ===========================================================================
def extract_diffphys_sufficient_stats(
    observation: Mapping[str, Any],
    geometry: dict[str, Any],
    bounds: dict[str, Any] | None = None,
    *,
    desaturator: ScintStationDesaturator | None = None,
    stripper: RossiGreisenFISTAShowerStripper | None = None,
    kink_detector: BrokenLineKalmanKinkDetector | None = None,
    obs_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract 32D physical sufficient statistics `z_hat` and uncertainty `z_sigma` purely from observables."""
    bnd = bounds or detector_bounds(geometry)
    desat = desaturator or ScintStationDesaturator(geometry, bnd)
    strip_op = stripper or RossiGreisenFISTAShowerStripper(geometry)
    kink_op = kink_detector or BrokenLineKalmanKinkDetector(geometry, bnd)

    cfg = obs_cfg or {'pe_only': True, 'bar_threshold_pe': 0.5}
    dom = compute_observable_containment(dict(observation), geometry, bnd, config=cfg)
    obs_dom_id = int(dom['obs_domain_id'])

    # Cluster silicon hits per plane (pure observation, zero truth)
    pxyz = np.asarray(observation['pixel']['xyz_mm'], dtype=np.float64)
    pcid = np.asarray(observation['pixel']['channel_id'], dtype=np.uint64)
    raw_edep = observation['pixel'].get('edep', observation['pixel'].get('edep_keV'))
    has_edep = raw_edep is not None and len(raw_edep) == len(pxyz) and len(pxyz) > 0
    edep_arr = np.asarray(raw_edep, dtype=np.float64) if has_edep else None
    if edep_arr is not None and float(np.median(edep_arr)) < 0.1:
        # Convert MeV to keV if raw G4 edep is stored in MeV (~0.0026 MeV/MIP)
        edep_arr = edep_arr * 1000.0
    cluster_edep_mip: np.ndarray | None = None
    if len(pxyz) > 0 and len(pcid) == len(pxyz):
        cl_res = cluster_pixel_hits(pcid, pxyz, int(bnd['nx']), int(bnd['ny']))
        n_cl_raw = int(cl_res['n_clusters'])
        c_pl = np.asarray(cl_res['cluster_plane'], dtype=np.int64)
        hit_ord = cl_res['hit_order']
        offs = cl_res['cluster_offsets']
        xyz_ord = pxyz[hit_ord]
        c_xyz = np.zeros((n_cl_raw, 3), dtype=np.float64)
        n_hits_cl = np.zeros(n_cl_raw, dtype=np.float64)
        ecc_cl = np.zeros(n_cl_raw, dtype=np.float64)
        if edep_arr is not None:
            edep_ord = edep_arr[hit_ord]
            cluster_edep_mip = np.zeros(n_cl_raw, dtype=np.float64)
        for ci in range(n_cl_raw):
            s_i, e_i = int(offs[ci]), int(offs[ci + 1])
            pts = xyz_ord[s_i:e_i]
            c_xyz[ci] = pts.mean(axis=0)
            m_h = float(e_i - s_i)
            n_hits_cl[ci] = m_h
            if edep_arr is not None:
                cluster_edep_mip[ci] = float(np.sum(edep_ord[s_i:e_i])) / PIXEL_MIP_EDEP_KEV
            if e_i - s_i >= 2:
                dxy = pts[:, :2] - c_xyz[ci, :2]
                cov = (dxy.T @ dxy) / max(m_h, 1.0)
                evals = np.linalg.eigvalsh(cov)
                lam2, lam1 = max(float(evals[0]), 1e-9), max(float(evals[1]), 1e-9)
                ecc_cl[ci] = math.sqrt(max(0.0, 1.0 - lam2 / lam1))
    else:
        c_xyz = np.empty((0, 3), dtype=np.float64)
        c_pl = np.empty(0, dtype=np.int64)
        n_hits_cl = None
        ecc_cl = None

    # 1A: 54-station de-saturation & time-walk unfolding
    cal = desat.reconstruct(observation['bar'], min_pe=0.5)
    # 1B: Rossi-Greisen X0 FISTA shower stripping
    st_res = strip_op.strip(
        c_xyz, c_pl, bipole_scores=ecc_cl, cluster_n_hits=n_hits_cl, cluster_edep_mip=cluster_edep_mip
    )
    # 1C: Broken-line Kalman kink & spectrometer/MCS muons
    kink = kink_op.analyze(c_xyz, c_pl, st_res['w_mip'], observation['tracker_hypothesis'], cal)

    e_st = cal['e_station_mev']
    e_sum_mev = max(float(np.sum(e_st)), 1e-3)
    act_e = np.flatnonzero(e_st > 2.0)
    if float(np.sum(e_st)) > 1.0 and len(act_e) > 0:
        start_st = int(act_e[0])
        dx0_rel = np.maximum(desat.station_x0 - desat.station_x0[start_st], 0.0)
        sh_x0 = float(np.dot(e_st, dx0_rel) / e_sum_mev)
        sh_w_x0 = math.sqrt(max(float(np.dot(e_st, (dx0_rel - sh_x0) ** 2) / e_sum_mev), 0.1))
        core_idx = int(np.argmax(e_st))
        trans_rms = math.sqrt(max(0.5 * (cal['x_var_station_mm2'][core_idx] + cal['y_var_station_mm2'][core_idx]), 1.0))
        em_core_frac = float(np.sum(e_st[start_st : min(len(e_st), start_st + 8)]) / e_sum_mev)
        tail_mip_frac = float(np.sum(e_st[min(len(e_st), start_st + 14) :]) / e_sum_mev)
    else:
        sh_x0, sh_w_x0, trans_rms, em_core_frac, tail_mip_frac = 0.0, 1.0, 50.0, 0.0, 0.0

    e_vis_gev = max(cal['e_vis_gev'], 0.0)
    pt_miss_gev = max(cal['pt_miss_gev'], 0.0)
    mu_p = max(kink['mu_p_best_gev'], 0.0)
    mu_pt = max(kink['mu_pt_gev'], 0.0)
    had_pt = max(pt_miss_gev, 0.5)
    mt_gev = math.sqrt(max(0.0, 2.0 * max(mu_pt, pt_miss_gev) * had_pt * 0.5))
    dphi_lead = 3.14159 * (1.0 - min(pt_miss_gev / (e_vis_gev * 0.05 + 1.0), 1.0))
    pt_asym = pt_miss_gev / (mu_pt + pt_miss_gev + 0.5)
    nomad_aco = math.atan2(pt_miss_gev, max(e_vis_gev * 0.04, 0.5))

    n_cl = len(c_xyz)
    n_mip_sum = float(kink['n_stripped_mip'])
    em_strip_frac = 1.0 - min(n_mip_sum / max(float(n_cl), 1.0), 1.0) if n_cl > 0 else 0.0
    front_mip = float(np.sum(st_res['w_mip'][c_pl < 6])) if n_cl > 0 else 0.0
    front30_ratio = front_mip / max(n_mip_sum, 1.0) if n_cl > 0 else 0.0
    mean_bipole = float(np.mean(ecc_cl)) if ecc_cl is not None and len(ecc_cl) > 0 else 0.0

    # Prompt / delayed fractions
    prompt_frac = cal['e_prompt_gev'] / max(e_vis_gev, 1e-4) if e_vis_gev > 0.0 else 0.0
    first_x0 = float(strip_op.plane_x0[int(c_pl.min())]) if n_cl > 0 else (float(desat.station_x0[np.flatnonzero(e_st > 1.0)[0]]) if (e_st > 1.0).any() else 0.0)

    # Rho visible mass proxy (combining stripped prong opening & prompt pi0 energy)
    rho_vis_mass = 0.77 * math.tanh(0.8 * (em_core_frac + kink['max_opening_angle_rad'] * 4.0)) if (kink['max_ip_sig'] > 2.5 or n_mip_sum >= 1.0) else 0.0
    # Charm dimuon / high-mass SV veto score
    charm_veto = float(kink['n_pen_muons'] >= 2) * 2.0 + max(0.0, kink['sv_m_corr_gev'] - 1.78)

    # Mode-B single-pixel edep summary (dims 32..35)
    if cluster_edep_mip is not None and len(cluster_edep_mip) > 0:
        tgt_ed = cluster_edep_mip[c_pl < 12] if np.any(c_pl < 12) else cluster_edep_mip
        mean_ed_mip = float(np.mean(tgt_ed))
        frac_2mip = float(np.mean(tgt_ed >= 1.55))
        str_m = st_res['w_mip'] >= 0.50
        str_ed_ratio = float(np.mean(cluster_edep_mip[str_m]) / max(mean_ed_mip, 1e-3)) if str_m.any() else 1.0
        edep_feats = [1.0, math.log1p(mean_ed_mip), frac_2mip, min(str_ed_ratio, 4.0)]
        edep_sigs = [0.01, 0.08, 0.05, 0.10]
    else:
        edep_feats = [0.0, 0.0, 0.0, 0.0]
        edep_sigs = [0.05, 0.50, 0.25, 0.50]

    z_hat = np.array([
        math.log1p(n_mip_sum),                          # 0
        em_strip_frac,                                  # 1
        front30_ratio,                                  # 2
        mean_bipole,                                    # 3
        math.log1p(kink['sv_lxy_sig']),                 # 4
        math.log1p(kink['sv_m_corr_gev']),              # 5
        math.log1p(kink['max_kink_llr']),               # 6
        kink['kink_pt_gev'],                            # 7
        math.log1p(kink['max_ip_sig']),                 # 8
        math.log1p(kink['n_high_ip_mips']),             # 9
        float(kink['n_sv_prongs']),                     # 10
        kink['max_opening_angle_rad'],                  # 11
        math.log1p(e_vis_gev),                          # 12
        em_core_frac,                                   # 13
        sh_x0,                                          # 14
        sh_w_x0,                                        # 15
        trans_rms,                                      # 16
        prompt_frac,                                    # 17
        math.log1p(cal['e_delayed_gev']),               # 18
        first_x0,                                       # 19
        math.log1p(pt_miss_gev),                        # 20
        math.log1p(mt_gev),                             # 21
        dphi_lead,                                      # 22
        tail_mip_frac,                                  # 23
        nomad_aco,                                      # 24
        rho_vis_mass,                                   # 25
        float(min(kink['n_pen_muons'], 3)) / 3.0,       # 26
        kink['mu_q_over_p'],                            # 27
        math.log1p(mu_p),                               # 28
        mu_pt,                                          # 29
        kink['mu_kink_compat'],                         # 30
        charm_veto,                                     # 31
        *edep_feats,                                    # 32..35
    ], dtype=np.float32)

    # Analytical uncertainty covariance diagonal sigma_z (propagating MCS, rank-deficient Y, & SiPM sat)
    has_front = bool(n_cl > 0 and np.any(c_pl < 6))
    has_tgt = bool(n_cl > 0 and np.any(c_pl < 12))
    mcs_inflate = 1.0 if has_front else (2.5 if has_tgt else 4.0)
    e_rel_sig = cal['e_vis_sigma_gev'] / max(e_vis_gev + 1.0, 1.0)
    pt_rel_sig = cal['pt_miss_sigma_gev'] / max(pt_miss_gev + 1.0, 1.0)
    mu_sig = 0.05 if abs(kink['mu_q_over_p']) > 0.0 else 0.35

    z_sigma = np.array([
        0.15, 0.08, 0.12 * mcs_inflate, 0.10,
        0.45 * mcs_inflate, 0.25 * mcs_inflate,
        0.35 * mcs_inflate, 0.18 * mcs_inflate,
        0.40 * mcs_inflate, 0.20 * mcs_inflate,
        0.35 * mcs_inflate, 0.03 * mcs_inflate,
        e_rel_sig, 0.08, 1.2, 0.8, 5.0, 0.08, 0.15, 1.0,
        pt_rel_sig, pt_rel_sig, 0.35, 0.12, 0.20, 0.15 * mcs_inflate,
        0.08, mu_sig, 0.25, 0.30, 0.15, 0.20,
        *edep_sigs,
    ], dtype=np.float32)

    obs_summary_ev = {
        'obs_domain_id': obs_dom_id,
        'e_vis_gev': float(e_vis_gev),
        'pt_miss_gev': float(pt_miss_gev),
        'mt_gev': float(mt_gev),
        'n_sv_prongs': int(kink['n_sv_prongs']),
        'sv_lxy_sig': float(kink['sv_lxy_sig']),
        'sv_mass_gev': float(kink['sv_m_corr_gev']),
        'n_pen_muons': int(kink['n_pen_muons']),
        'kink_llr': float(kink['max_kink_llr']),
        'dphi_mu_had': float(dphi_lead),
        'ip_sig_max': float(kink['max_ip_sig']),
        'pi0_em_frac': float(em_core_frac * prompt_frac),
        'rho_vis_mass_gev': float(rho_vis_mass),
        'has_prompt_e': bool(has_front and em_core_frac > 0.72 and kink['max_ip_sig'] < 2.2),
        'has_front_pixel': has_front,
        'has_target_pixel': has_tgt,
    }

    return {
        'diffphys_z': z_hat,
        'diffphys_z_sigma': z_sigma,
        'obs_domain_id': obs_dom_id,
        'e_vis_gev': float(e_vis_gev),
        'pv_xyz_mm': kink['pv_xyz_mm'].astype(np.float32),
        'pv_sigma_mm': kink['pv_sigma_mm'].astype(np.float32),
        'w_mip': st_res['w_mip'].astype(np.float32),
        'obs_summary': obs_summary_ev,
    }


def strip_shower_and_score_segments(
    pix_xyz_mm: np.ndarray,
    pix_plane: np.ndarray,
    stripper: RossiGreisenFISTAShowerStripper,
    *,
    bipole_scores: np.ndarray | None = None,
    cluster_n_hits: np.ndarray | None = None,
    cluster_edep_mip: np.ndarray | None = None,
    w_mip_thr: float = 0.50,
    max_slope: float = 0.24,
    cl_origin_label: np.ndarray | None = None,
    cl_dominant_track_id: np.ndarray | None = None,
) -> dict[str, Any]:
    """Strip EM cascade fog via FISTA, build candidate 3-hit segments, and score purity vs raw baseline.

    When optional training/audit-only `cl_origin_label` and `cl_dominant_track_id` are supplied,
    quantifies tau daughter cluster recall and 3-hit segment truth purity before vs after stripping.
    """
    xyz = np.asarray(pix_xyz_mm, dtype=np.float64)
    pl = np.asarray(pix_plane, dtype=np.int64)
    res = stripper.strip(
        xyz, pl, bipole_scores=bipole_scores, cluster_n_hits=cluster_n_hits, cluster_edep_mip=cluster_edep_mip
    )
    w_mip = res['w_mip']
    n_cl = len(xyz)
    if n_cl < 3:
        return {
            'w_mip': w_mip,
            'n_raw_segments': 0,
            'n_stripped_segments': 0,
            'raw_purity': 0.0,
            'stripped_purity': 0.0,
            'tau_cluster_recall': 1.0,
        }

    keep_mip = w_mip >= float(w_mip_thr)
    # Build target-zone 3-plane collinear candidate segments (planes 0..11) before and after FISTA stripping
    def _count_segments(mask: np.ndarray) -> tuple[int, int]:
        n_seg = 0
        n_pure = 0
        for p0 in range(10):
            p1, p2 = p0 + 1, p0 + 2
            i0 = np.flatnonzero(mask & (pl == p0))[:36]
            i1 = np.flatnonzero(mask & (pl == p1))[:36]
            i2 = np.flatnonzero(mask & (pl == p2))[:36]
            if len(i0) == 0 or len(i1) == 0 or len(i2) == 0:
                continue
            dz01 = float(xyz[i1[0], 2] - xyz[i0[0], 2])
            dz02 = float(xyz[i2[0], 2] - xyz[i0[0], 2])
            if dz01 <= 1.0 or dz02 <= 2.0:
                continue
            for a in i0:
                for b in i1:
                    tx = (xyz[b, 0] - xyz[a, 0]) / dz01
                    ty = (xyz[b, 1] - xyz[a, 1]) / dz01
                    if math.hypot(tx, ty) > max_slope:
                        continue
                    px_2 = xyz[a, 0] + tx * dz02
                    py_2 = xyz[a, 1] + ty * dz02
                    d2 = np.hypot(xyz[i2, 0] - px_2, xyz[i2, 1] - py_2)
                    tol = 0.35 if p2 < 6 else 1.8
                    hits2 = i2[d2 <= tol]
                    for c in hits2:
                        n_seg += 1
                        if cl_dominant_track_id is not None:
                            ta = int(cl_dominant_track_id[a])
                            tb = int(cl_dominant_track_id[b])
                            tc = int(cl_dominant_track_id[c])
                            if ta > 0 and ta == tb and tb == tc:
                                if cl_origin_label is None or int(cl_origin_label[a]) != 3:
                                    n_pure += 1
        return n_seg, n_pure

    n_raw, n_raw_pure = _count_segments(np.ones(n_cl, dtype=bool))
    n_str, n_str_pure = _count_segments(keep_mip)

    tau_rec = 1.0
    em_strip = 0.0
    if cl_origin_label is not None:
        orig_arr = np.asarray(cl_origin_label, dtype=np.int64)
        tau_m = orig_arr == 1
        if tau_m.any():
            tau_rec = float(np.mean(keep_mip[tau_m]))
        em_m = orig_arr == 3
        if em_m.any():
            em_strip = float(np.mean(~keep_mip[em_m]))

    return {
        'w_mip': w_mip,
        'n_raw_segments': int(n_raw),
        'n_stripped_segments': int(n_str),
        'raw_purity': float(n_raw_pure / max(n_raw, 1)),
        'stripped_purity': float(n_str_pure / max(n_str, 1)),
        'tau_cluster_recall': float(tau_rec),
        'em_stripped_frac': float(em_strip),
    }


def extract_diffphys_from_gen4_item(
    item: Mapping[str, Any],
    geom_contract: Any,
    *,
    stripper: RossiGreisenFISTAShowerStripper | None = None,
    kink_detector: BrokenLineKalmanKinkDetector | None = None,
) -> dict[str, Any]:
    """Extract 32D DiffPhys sufficient statistics directly from a Gen4 cached item (pure observables)."""
    raw_geom = geom_contract.raw_geometry if hasattr(geom_contract, 'raw_geometry') else geom_contract
    bnd = geom_contract.bounds_dict if hasattr(geom_contract, 'bounds_dict') else detector_bounds(raw_geom)
    px0 = getattr(geom_contract, 'pixel_X0', None)
    strip_op = stripper or RossiGreisenFISTAShowerStripper(raw_geom, pixel_x0=px0)
    kink_op = kink_detector or BrokenLineKalmanKinkDetector(raw_geom, bnd, pixel_x0=px0)

    c_xyz = np.asarray(item['pix_xyz_mm'], dtype=np.float64)
    c_pl = np.asarray(item['pix_plane'], dtype=np.int64)
    c_feat = np.asarray(item.get('pix_feat', np.empty((0, 32))), dtype=np.float64)
    if len(c_xyz) > 0 and len(c_feat) == len(c_xyz):
        n_hits_cl = np.expm1(np.clip(c_feat[:, 3] * 6.0, 0.0, 6.0)) if c_feat.shape[1] >= 4 else None
        raw_bsplit = np.clip(c_feat[:, 24], 0.0, 1.5) if c_feat.shape[1] >= 25 else np.clip(c_feat[:, 9], 0.0, 1.5)
        bipole = np.where(n_hits_cl >= 3.5, raw_bsplit, 0.0) if (n_hits_cl is not None and c_feat.shape[1] >= 25) else raw_bsplit
    else:
        n_hits_cl, bipole = None, None

    has_edep_item = float(item.get('has_edep', 0.0)) > 0.5
    cluster_edep_mip: np.ndarray | None = None
    cluster_2mip_prob: np.ndarray | None = None
    if 'pix_edep_mip' in item and len(item['pix_edep_mip']) == len(c_xyz) and len(c_xyz) > 0:
        cluster_edep_mip = np.asarray(item['pix_edep_mip'], dtype=np.float64)
        has_edep_item = True
    elif has_edep_item and len(c_xyz) > 0 and c_feat.shape[1] >= 32:
        # [gen4-contract-v2] Section 3.1: [30] = ln(1 + Q_tot_MIP), [31] = 2-MIP conversion score in (0, 1)
        cluster_edep_mip = np.maximum(np.expm1(np.clip(c_feat[:, 30], 0.0, 6.0)), 0.0)
        cluster_2mip_prob = np.clip(c_feat[:, 31], 0.0, 1.0)
    elif has_edep_item and len(c_xyz) > 0 and c_feat.shape[1] >= 31:
        cluster_edep_mip = np.maximum(np.expm1(np.clip(c_feat[:, 30], 0.0, 6.0)), 0.0)

    # Combine geometric bipole score with 2-MIP conversion probability when available
    bipole_eff = bipole
    if cluster_2mip_prob is not None:
        bipole_eff = cluster_2mip_prob if bipole_eff is None else np.maximum(bipole_eff, cluster_2mip_prob)

    st_res = strip_op.strip(
        c_xyz, c_pl, bipole_scores=bipole_eff, cluster_n_hits=n_hits_cl, cluster_edep_mip=cluster_edep_mip
    )

    # Reconstruct 54-station energy summary from either `scint_grid` or sparse bar fields
    n_st = int(getattr(geom_contract, 'n_stations', 54))
    e_station = np.zeros(n_st, dtype=np.float64)
    e_prompt = np.zeros(n_st, dtype=np.float64)
    e_delayed = np.zeros(n_st, dtype=np.float64)
    x_st = np.full(n_st, float(bnd['x_scint_center']), dtype=np.float64)
    y_st = np.full(n_st, float(bnd['y_scint_center']), dtype=np.float64)
    x_var = np.full(n_st, 200.0, dtype=np.float64)
    y_var = np.full(n_st, 200.0, dtype=np.float64)
    bx = np.asarray(bnd['scint_bar_Xcenter'], dtype=np.float64)
    by = np.asarray(bnd['scint_bar_Ycenter'], dtype=np.float64)
    st_z = np.asarray(getattr(geom_contract, 'station_z_mm', np.linspace(-2980.0, -2005.0, n_st)), dtype=np.float64)
    st_x0 = np.asarray(getattr(geom_contract, 'station_x0', np.linspace(1.0, 100.0, n_st)), dtype=np.float64)
    has_y = np.asarray(getattr(geom_contract, 'station_has_y', np.ones(n_st, dtype=bool)), dtype=bool)
    n_sat_bars = 0
    n_subthr_bars = 0

    if 'scint_grid' in item or ('bar_feat_10' in item and 'bar_station' in item):
        n_bars_arr = len(bx)
        if 'scint_grid' in item:
            sg = np.asarray(item['scint_grid'], dtype=np.float64)  # (54, 2, 40, 10)
            ex_bars = sg[:, 0, :, 0]
            ey_bars = sg[:, 1, :, 0]
            dl_x = ex_bars * sg[:, 0, :, 9]
            dl_y = ey_bars * sg[:, 1, :, 9]
            n_sat_bars = int(np.sum(sg[:, :, :, 6] > 0.5))
            n_subthr_bars = int(np.sum(sg[:, :, :, 7] > 0.5))
        else:
            bs = np.asarray(item['bar_station'], dtype=np.int64)
            bv = np.asarray(item['bar_view'], dtype=np.int64)
            bi = np.asarray(item['bar_idx'], dtype=np.int64)
            bf = np.asarray(item['bar_feat_10'], dtype=np.float64)
            ex_bars = np.zeros((n_st, n_bars_arr), dtype=np.float64)
            ey_bars = np.zeros((n_st, n_bars_arr), dtype=np.float64)
            dl_x = np.zeros((n_st, n_bars_arr), dtype=np.float64)
            dl_y = np.zeros((n_st, n_bars_arr), dtype=np.float64)
            if len(bf) > 0:
                vb = (bs >= 0) & (bs < n_st) & (bi >= 0) & (bi < n_bars_arr)
                bs, bv, bi, bf = bs[vb], bv[vb], bi[vb], bf[vb]
                n_sat_bars = int(np.sum(bf[:, 6] > 0.5))
                n_subthr_bars = int(np.sum(bf[:, 7] > 0.5))
                xm = bv == 0
                ym = bv == 1
                if xm.any():
                    np.add.at(ex_bars, (bs[xm], bi[xm]), bf[xm, 0])
                    np.add.at(dl_x, (bs[xm], bi[xm]), bf[xm, 0] * bf[xm, 9])
                if ym.any():
                    np.add.at(ey_bars, (bs[ym], bi[ym]), bf[ym, 0])
                    np.add.at(dl_y, (bs[ym], bi[ym]), bf[ym, 0] * bf[ym, 9])
        sx = ex_bars.sum(axis=1)
        sy = ey_bars.sum(axis=1)
        dl_sx = dl_x.sum(axis=1)
        dl_sy = dl_y.sum(axis=1)
        for s in range(n_st):
            if has_y[s]:
                e_station[s] = 0.5 * (sx[s] + sy[s])
                e_delayed[s] = 0.5 * (dl_sx[s] + dl_sy[s])
            else:
                e_station[s] = sx[s]
                e_delayed[s] = dl_sx[s]
            e_prompt[s] = max(0.0, e_station[s] - e_delayed[s])
            if sx[s] > 0.0:
                wx = ex_bars[s] / sx[s]
                x_st[s] = float(np.dot(wx, bx))
                x_var[s] = max(float(np.dot(wx, (bx - x_st[s]) ** 2)) + 9.0, 9.0)
            if has_y[s] and sy[s] > 0.0:
                wy = ey_bars[s] / sy[s]
                y_st[s] = float(np.dot(wy, by))
                y_var[s] = max(float(np.dot(wy, (by - y_st[s]) ** 2)) + 9.0, 9.0)
    else:
        desat = ScintStationDesaturator(raw_geom, bnd)
        bar_dict = item.get('bar', {
            'panel': item.get('bar_panel', ()),
            'bar': item.get('bar_idx', ()),
            'horizontal': item.get('bar_horiz', ()),
            'pe_proxy': item.get('bar_pe', ()),
            'hit_time_ns': item.get('bar_time_ns', ()),
        })
        cal = desat.reconstruct(bar_dict, min_pe=0.5)
        e_station = cal['e_station_mev']
        e_prompt = cal['e_prompt_station_mev']
        e_delayed = cal['e_delayed_station_mev']
        x_st, y_st = cal['x_station_mm'], cal['y_station_mm']
        x_var, y_var = cal['x_var_station_mm2'], cal['y_var_station_mm2']
        n_sat_bars = int(cal['n_sat_bars'])

    e_vis_gev = float(np.sum(e_station)) * 1e-3
    e_prompt_gev = float(np.sum(e_prompt)) * 1e-3
    e_delayed_gev = float(np.sum(e_delayed)) * 1e-3

    # Transverse momentum from prompt stations
    act = np.flatnonzero(e_prompt > 1.0)
    if len(act) >= 2:
        z0 = float(st_z[act[0]] - 15.0)
        dz = np.maximum(st_z[act] - z0, 15.0)
        ep = e_prompt[act]
        xr = float(np.average(x_st[act[:min(3, len(act))]], weights=ep[:min(3, len(act))]))
        yr = float(np.average(y_st[act[:min(3, len(act))]], weights=ep[:min(3, len(act))]))
        tx_s = (x_st[act] - xr) / dz
        ty_s = (y_st[act] - yr) / dz
        ns = np.sqrt(1.0 + tx_s ** 2 + ty_s ** 2)
        pt_miss_gev = math.hypot(float(np.sum(ep * tx_s / ns)), float(np.sum(ep * ty_s / ns))) * 1e-3
        pt_sigma_gev = max(0.25, math.sqrt(float(np.sum((ep / dz) ** 2 * (x_var[act] + y_var[act])))) * 1e-3)
    else:
        pt_miss_gev, pt_sigma_gev = 0.0, 1.0

    cal_summary = {
        'e_station_mev': e_station,
        'e_prompt_station_mev': e_prompt,
        'e_delayed_station_mev': e_delayed,
        'x_station_mm': x_st,
        'y_station_mm': y_st,
        'x_var_station_mm2': x_var,
        'y_var_station_mm2': y_var,
        'e_vis_gev': e_vis_gev,
        'e_prompt_gev': e_prompt_gev,
        'e_delayed_gev': e_delayed_gev,
        'e_vis_sigma_gev': max(0.35, 0.08 * e_vis_gev + 0.06 * n_sat_bars * max(e_vis_gev, 5.0)),
        'pt_miss_gev': pt_miss_gev,
        'pt_miss_sigma_gev': pt_sigma_gev,
        'n_sat_bars': n_sat_bars,
    }
    trk_hyp = item.get('tracker_hypothesis', {'station': (), 'xy_mm': np.empty((0, 2))})
    kink = kink_op.analyze(c_xyz, c_pl, st_res['w_mip'], trk_hyp, cal_summary)

    # Override spectrometer muon kinematics with UnifiedMuonKalman M1-M4 if present in item!
    if 'muon_kalman_tracks' in item and 'muon_kalman_mask' in item:
        mk = np.asarray(item['muon_kalman_tracks'], dtype=np.float64)
        mm = np.asarray(item['muon_kalman_mask'], dtype=bool)
        if mm.any():
            valid_mk = mk[mm]
            n_pen_mu = int(mm.sum())
            lead_mu = valid_mk[0]
            q_over_p = float(lead_mu[0])
            p_best_gev = float(lead_mu[2])
            mu_pt_gev = float(p_best_gev * math.hypot(float(lead_mu[12]), float(lead_mu[13])))
            mu_compat = float(math.exp(-min(kink['max_kink_llr'], 50.0) * 0.02) / (1.0 + abs(mu_pt_gev - kink['kink_pt_gev'])))
        else:
            n_pen_mu = int(kink['n_pen_muons'])
            q_over_p = float(kink['mu_q_over_p'])
            p_best_gev = float(kink['mu_p_best_gev'])
            mu_pt_gev = float(kink['mu_pt_gev'])
            mu_compat = float(kink['mu_kink_compat'])
    else:
        n_pen_mu = int(kink['n_pen_muons'])
        q_over_p = float(kink['mu_q_over_p'])
        p_best_gev = float(kink['mu_p_best_gev'])
        mu_pt_gev = float(kink['mu_pt_gev'])
        mu_compat = float(kink['mu_kink_compat'])

    e_sum_mev = max(float(np.sum(e_station)), 1e-3)
    act_e = np.flatnonzero(e_station > 2.0)
    if float(np.sum(e_station)) > 1.0 and len(act_e) > 0:
        start_st = int(act_e[0])
        dx0_rel = np.maximum(st_x0 - st_x0[start_st], 0.0)
        sh_x0 = float(np.dot(e_station, dx0_rel) / e_sum_mev)
        sh_w_x0 = math.sqrt(max(float(np.dot(e_station, (dx0_rel - sh_x0) ** 2) / e_sum_mev), 0.1))
        core_idx = int(np.argmax(e_station))
        trans_rms = math.sqrt(max(0.5 * (x_var[core_idx] + y_var[core_idx]), 1.0))
        em_core_frac = float(np.sum(e_station[start_st : min(n_st, start_st + 8)]) / e_sum_mev)
        tail_mip_frac = float(np.sum(e_station[min(n_st, start_st + 14) :]) / e_sum_mev)
    else:
        sh_x0, sh_w_x0, trans_rms, em_core_frac, tail_mip_frac = 0.0, 1.0, 50.0, 0.0, 0.0

    obs_dom_id = int(item['obs_domain_id'])
    n_cl = len(c_xyz)
    n_mip_sum = float(kink['n_stripped_mip'])
    em_strip_frac = 1.0 - min(n_mip_sum / max(float(n_cl), 1.0), 1.0) if n_cl > 0 else 0.0
    front_mip = float(np.sum(st_res['w_mip'][c_pl < 6])) if n_cl > 0 else 0.0
    front30_ratio = front_mip / max(n_mip_sum, 1.0) if n_cl > 0 else 0.0
    mean_bipole = float(np.mean(bipole)) if bipole is not None and len(bipole) > 0 else 0.0
    prompt_frac = e_prompt_gev / max(e_vis_gev, 1e-4) if e_vis_gev > 0.0 else 0.0
    first_x0 = float(strip_op.plane_x0[int(c_pl.min())]) if n_cl > 0 else (float(st_x0[np.flatnonzero(e_station > 1.0)[0]]) if (e_station > 1.0).any() else 0.0)
    had_pt = max(pt_miss_gev, 0.5)
    mt_gev = math.sqrt(max(0.0, 2.0 * max(mu_pt_gev, pt_miss_gev) * had_pt * 0.5))
    dphi_lead = 3.14159 * (1.0 - min(pt_miss_gev / (e_vis_gev * 0.05 + 1.0), 1.0))
    pt_asym = pt_miss_gev / (mu_pt_gev + pt_miss_gev + 0.5)
    nomad_aco = math.atan2(pt_miss_gev, max(e_vis_gev * 0.04, 0.5))
    rho_vis_mass = 0.77 * math.tanh(0.8 * (em_core_frac + kink['max_opening_angle_rad'] * 4.0)) if (kink['max_ip_sig'] > 2.5 or n_mip_sum >= 1.0) else 0.0
    charm_veto = float(n_pen_mu >= 2) * 2.0 + max(0.0, kink['sv_m_corr_gev'] - 1.78)

    if has_edep_item and cluster_edep_mip is not None and len(cluster_edep_mip) > 0:
        tgt_m = c_pl < 12
        tgt_ed = cluster_edep_mip[tgt_m] if np.any(tgt_m) else cluster_edep_mip
        mean_ed_mip = float(np.mean(tgt_ed))
        if cluster_2mip_prob is not None:
            tgt_2m = cluster_2mip_prob[tgt_m] if np.any(tgt_m) else cluster_2mip_prob
            frac_2mip = float(np.mean(tgt_2m > 0.5))
        else:
            frac_2mip = float(np.mean(tgt_ed >= 1.55))
        str_m = st_res['w_mip'] >= 0.50
        str_ed_ratio = float(np.mean(cluster_edep_mip[str_m]) / max(mean_ed_mip, 1e-3)) if str_m.any() else 1.0
        edep_feats = [1.0, math.log1p(mean_ed_mip), frac_2mip, min(str_ed_ratio, 4.0)]
        edep_sigs = [0.01, 0.08, 0.05, 0.10]
    else:
        edep_feats = [0.0, 0.0, 0.0, 0.0]
        edep_sigs = [0.05, 0.50, 0.25, 0.50]

    z_hat = np.array([
        math.log1p(n_mip_sum), em_strip_frac, front30_ratio, mean_bipole,
        math.log1p(kink['sv_lxy_sig']), math.log1p(kink['sv_m_corr_gev']), math.log1p(kink['max_kink_llr']),
        kink['kink_pt_gev'], math.log1p(kink['max_ip_sig']), math.log1p(kink['n_high_ip_mips']),
        float(kink['n_sv_prongs']), kink['max_opening_angle_rad'], math.log1p(e_vis_gev),
        em_core_frac, sh_x0, sh_w_x0, trans_rms, prompt_frac, math.log1p(e_delayed_gev + 0.25 * n_subthr_bars),
        first_x0, math.log1p(pt_miss_gev), math.log1p(mt_gev), dphi_lead, tail_mip_frac,
        nomad_aco, rho_vis_mass, float(min(n_pen_mu, 3)) / 3.0, q_over_p,
        math.log1p(p_best_gev), mu_pt_gev, mu_compat, charm_veto,
        *edep_feats,
    ], dtype=np.float32)

    has_front = bool(n_cl > 0 and np.any(c_pl < 6))
    has_tgt = bool(n_cl > 0 and np.any(c_pl < 12))
    mcs_inf = 1.0 if has_front else (2.5 if has_tgt else 4.0)
    e_rel = cal_summary['e_vis_sigma_gev'] / max(e_vis_gev + 1.0, 1.0)
    pt_rel = pt_sigma_gev / max(pt_miss_gev + 1.0, 1.0)
    mu_sig = 0.05 if abs(q_over_p) > 0.0 else 0.35
    z_sigma = np.array([
        0.15, 0.08, 0.12 * mcs_inf, 0.10, 0.45 * mcs_inf, 0.25 * mcs_inf,
        0.35 * mcs_inf, 0.18 * mcs_inf, 0.40 * mcs_inf, 0.20 * mcs_inf,
        0.35 * mcs_inf, 0.03 * mcs_inf, e_rel, 0.08, 1.2, 0.8, 5.0, 0.08, 0.15, 1.0,
        pt_rel, pt_rel, 0.35, 0.12, 0.20, 0.15 * mcs_inf, 0.08, mu_sig, 0.25, 0.30, 0.15, 0.20,
        *edep_sigs,
    ], dtype=np.float32)

    obs_summary_ev = {
        'obs_domain_id': obs_dom_id,
        'e_vis_gev': float(e_vis_gev),
        'pt_miss_gev': float(pt_miss_gev),
        'mt_gev': float(mt_gev),
        'n_sv_prongs': int(kink['n_sv_prongs']),
        'sv_lxy_sig': float(kink['sv_lxy_sig']),
        'sv_mass_gev': float(kink['sv_m_corr_gev']),
        'n_pen_muons': n_pen_mu,
        'kink_llr': float(kink['max_kink_llr']),
        'dphi_mu_had': float(dphi_lead),
        'ip_sig_max': float(kink['max_ip_sig']),
        'pi0_em_frac': float(em_core_frac * prompt_frac),
        'rho_vis_mass_gev': float(rho_vis_mass),
        'has_prompt_e': bool(has_front and em_core_frac > 0.72 and kink['max_ip_sig'] < 2.2),
        'has_front_pixel': has_front,
        'has_target_pixel': has_tgt,
    }
    return {
        'diffphys_z': z_hat,
        'diffphys_z_sigma': z_sigma,
        'obs_domain_id': obs_dom_id,
        'e_vis_gev': float(e_vis_gev),
        'pv_xyz_mm': kink['pv_xyz_mm'].astype(np.float32),
        'pv_sigma_mm': kink['pv_sigma_mm'].astype(np.float32),
        'obs_summary': obs_summary_ev,
    }


# ===========================================================================
# 5. Lightweight Conditional Rational-Quadratic Neural Spline Flow (DiffPhys-SBI)
# ===========================================================================
def _rational_quadratic_spline(
    inputs: torch.Tensor,
    unnorm_widths: torch.Tensor,
    unnorm_heights: torch.Tensor,
    unnorm_derivatives: torch.Tensor,
    *,
    tail_bound: float = 8.0,
    min_bin_width: float = 1e-3,
    min_bin_height: float = 1e-3,
    min_derivative: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable forward Rational-Quadratic Spline on [-tail_bound, tail_bound] with linear tails."""
    num_bins = unnorm_widths.shape[-1]
    b = float(tail_bound)
    inside = (inputs >= -b) & (inputs <= b)
    x_clamped = torch.clamp(inputs, -b, b)

    widths = F.softmax(unnorm_widths, dim=-1)
    widths = min_bin_width + (1.0 - min_bin_width * num_bins) * widths
    cumwidths = F.pad(torch.cumsum(widths, dim=-1), (1, 0), value=0.0)
    cumwidths = 2.0 * b * cumwidths - b
    cumwidths[..., 0] = -b
    cumwidths[..., -1] = b
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    heights = F.softmax(unnorm_heights, dim=-1)
    heights = min_bin_height + (1.0 - min_bin_height * num_bins) * heights
    cumheights = F.pad(torch.cumsum(heights, dim=-1), (1, 0), value=0.0)
    cumheights = 2.0 * b * cumheights - b
    cumheights[..., 0] = -b
    cumheights[..., -1] = b
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    derivs_int = F.softplus(unnorm_derivatives) + min_derivative
    derivs = F.pad(derivs_int, (1, 1), value=1.0)

    # Locate bin index safely
    bin_idx = torch.sum(x_clamped.unsqueeze(-1) >= cumwidths[..., :-1], dim=-1, keepdim=True) - 1
    bin_idx = bin_idx.clamp(0, num_bins - 1)

    input_cumwidths = cumwidths.gather(-1, bin_idx).squeeze(-1)
    input_bin_widths = widths.gather(-1, bin_idx).squeeze(-1).clamp_min(1e-6)
    input_cumheights = cumheights.gather(-1, bin_idx).squeeze(-1)
    input_bin_heights = heights.gather(-1, bin_idx).squeeze(-1).clamp_min(1e-6)
    input_delta = input_bin_heights / input_bin_widths

    d0 = derivs.gather(-1, bin_idx).squeeze(-1)
    d1 = derivs.gather(-1, (bin_idx + 1).clamp(max=num_bins)).squeeze(-1)

    theta = ((x_clamped - input_cumwidths) / input_bin_widths).clamp(0.0, 1.0)
    theta_1m = 1.0 - theta
    t_prod = theta * theta_1m

    num = input_bin_heights * (input_delta * theta * theta + d0 * t_prod)
    den = input_delta + (d1 + d0 - 2.0 * input_delta) * t_prod
    y_inside = input_cumheights + num / den.clamp_min(1e-6)

    deriv_num = (input_delta ** 2) * (d1 * theta * theta + 2.0 * input_delta * t_prod + d0 * theta_1m * theta_1m)
    logabsdet_inside = torch.log(deriv_num.clamp_min(1e-10)) - 2.0 * torch.log(den.clamp_min(1e-6))

    outputs = torch.where(inside, y_inside, inputs)
    logabsdet = torch.where(inside, logabsdet_inside, torch.zeros_like(inputs))
    return outputs, logabsdet


class _RQSplineCouplingBlock(nn.Module):
    """Single affine-masked Rational-Quadratic Coupling block conditioned on process/domain/energy."""

    def __init__(self, dim: int, cond_dim: int, mask_even: bool, hidden_dim: int = 64, n_bins: int = 8) -> None:
        super().__init__()
        self.dim = int(dim)
        self.n_bins = int(n_bins)
        mask = torch.arange(dim) % 2 == (0 if mask_even else 1)
        self.register_buffer('mask', mask)
        n_pass = int(mask.sum().item())
        n_trans = dim - n_pass
        self.n_trans = n_trans
        out_dim = n_trans * (3 * n_bins - 1)
        self.net = nn.Sequential(
            nn.Linear(n_pass + cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_pass = z[:, self.mask]
        z_trans = z[:, ~self.mask]
        params = self.net(torch.cat([z_pass, cond], dim=-1)).view(z.shape[0], self.n_trans, 3 * self.n_bins - 1)
        uw = params[..., : self.n_bins]
        uh = params[..., self.n_bins : 2 * self.n_bins]
        ud = params[..., 2 * self.n_bins :]

        y_trans, lad = _rational_quadratic_spline(z_trans, uw, uh, ud)
        out = z.clone()
        out[:, ~self.mask] = y_trans
        return out, lad.sum(dim=-1)


class DiffPhysSBIModel(NextGenFlavourModel):
    """Low-variance (~127k param) Conditional Neural Spline Flow `DiffPhys-SBI` for 32D physics statistics.

    Fits generative densities `p_phi(z | c, d, E_vis)` over 8 fine-grained physical processes
    with analytical observation-covariance marginalization over `Sigma_z(x)`.
    """

    def __init__(
        self,
        stat_dim: int = DIFFPHYS_STAT_DIM,
        n_processes: int = 8,
        hidden_dim: int = 64,
        n_flows: int = 4,
        n_bins: int = 8,
        cond_dim: int = 32,
        n_marg_eval: int = 5,
    ) -> None:
        super().__init__()
        self.stat_dim = int(stat_dim)
        self.n_processes = int(n_processes)
        self.n_marg_eval = int(max(1, n_marg_eval))

        # Running/fixed standardization buffers fitted on training stats (purely from train fold)
        self.register_buffer('z_mean', torch.zeros(self.stat_dim, dtype=torch.float32))
        self.register_buffer('z_std', torch.ones(self.stat_dim, dtype=torch.float32))

        # Conditioning encoder (process + domain + log1p(E_vis))
        self.cond_encoder = nn.ModuleDict({
            'proc_emb': nn.Embedding(self.n_processes, 16),
            'dom_emb': nn.Embedding(4, 8),
            'e_mlp': nn.Sequential(nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 8)),
        })
        self.flow_blocks = nn.ModuleList([
            _RQSplineCouplingBlock(
                self.stat_dim,
                cond_dim=cond_dim,
                mask_even=(i % 2 == 0),
                hidden_dim=hidden_dim,
                n_bins=n_bins,
            )
            for i in range(n_flows)
        ])
        self.prior_head = nn.Sequential(
            nn.Linear(16, 32),
            nn.SiLU(),
            nn.Linear(32, self.n_processes),
        )
        self.vtx_calib_head = nn.Sequential(
            nn.Linear(self.stat_dim + 8, 32),
            nn.SiLU(),
            nn.Linear(32, 6),  # [dv_x, dv_y, dv_z, log_sig_x, log_sig_y, log_sig_z]
        )
        self.log_lik_scale = nn.Parameter(torch.tensor(math.log(1.0 / 6.0), dtype=torch.float32))

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.z_mean.copy_(mean.detach().float().to(self.z_mean.device))
        self.z_std.copy_(std.detach().float().clamp_min(1e-3).to(self.z_std.device))

    def _encode_ctx(self, obs_domain_id: torch.Tensor, e_vis_gev: torch.Tensor) -> torch.Tensor:
        d_emb = self.cond_encoder['dom_emb'](obs_domain_id.long().clamp(0, 3))
        e_in = torch.log1p(e_vis_gev.float().clamp_min(0.0)).unsqueeze(-1)
        e_emb = self.cond_encoder['e_mlp'](e_in)
        return torch.cat([d_emb, e_emb], dim=-1)  # (B, 16)

    def log_prob_given_process(
        self,
        z_raw: torch.Tensor,
        proc_id: torch.Tensor,
        obs_domain_id: torch.Tensor,
        e_vis_gev: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate exact log p_phi(z | c, d, E_vis) in FP32."""
        with torch.amp.autocast(device_type=z_raw.device.type, enabled=False):
            z = ((z_raw.float() - self.z_mean) / self.z_std.clamp_min(1e-3)).clamp(-12.0, 12.0)
            ctx = self._encode_ctx(obs_domain_id, e_vis_gev)
            p_emb = self.cond_encoder['proc_emb'](proc_id.long().clamp(0, self.n_processes - 1))
            cond = torch.cat([p_emb, ctx], dim=-1)

            u = z
            logabsdet = z.new_zeros(z.shape[0])
            for blk in self.flow_blocks:
                u, lad = blk(u, cond)
                logabsdet = logabsdet + lad

            log_base = -0.5 * (u.square() + math.log(2.0 * math.pi)).sum(dim=-1)
            return log_base + logabsdet

    def _extract_batch_tensors(self, batch: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract pure-observable tensors strictly ignoring all TRUTH_BATCH_KEYS."""
        safe = {k: v for k, v in batch.items() if k not in TRUTH_BATCH_KEYS}
        if 'diffphys_z' in safe:
            z = safe['diffphys_z'].float()
            dev = z.device
            bsz = int(z.shape[0])
            z_sig = safe['diffphys_z_sigma'].float() if 'diffphys_z_sigma' in safe else torch.full_like(z, 0.1)
            dom = safe['obs_domain_id'].long() if 'obs_domain_id' in safe else torch.ones(bsz, dtype=torch.long, device=dev)
            if 'e_vis_gev' in safe:
                e_vis = safe['e_vis_gev'].float()
            else:
                e_vis = torch.expm1(z[:, 12].clamp(0.0, 10.0))
            pv_phys = safe['pv_xyz_mm'].float() if 'pv_xyz_mm' in safe else torch.zeros((bsz, 3), dtype=torch.float32, device=dev)
            return z, z_sig, dom, e_vis, pv_phys

        # Fallback adapter if called with Gen4/NextGen batch without pre-extracted `diffphys_z`
        if 'c_obs' in safe:
            c_obs = safe['c_obs'].float()
            bsz = int(c_obs.shape[0])
            dev = c_obs.device
            z = torch.zeros((bsz, self.stat_dim), dtype=torch.float32, device=dev)
            n_copy = min(self.stat_dim, int(c_obs.shape[1]))
            z[:, :n_copy] = c_obs[:, :n_copy]
            z_sig = torch.full_like(z, 0.15)
            dom = safe['obs_domain_id'].long() if 'obs_domain_id' in safe else (c_obs[:, 0] > 0.5).long() + 1
            e_vis = torch.expm1((c_obs[:, 15] * 6.0).clamp(0.0, 10.0)) if c_obs.shape[1] >= 16 else torch.full((bsz,), 200.0, device=dev)
            pv_phys = torch.zeros((bsz, 3), dtype=torch.float32, device=dev)
            return z, z_sig, dom, e_vis, pv_phys

        raise KeyError('Batch must provide `diffphys_z` (or `c_obs` fallback) pure-observable tensors')

    def forward_NextGen(self, batch: dict[str, torch.Tensor]) -> NextGenModelOutput:
        """Compute generative process log-densities, covariance-marginalized LLR & NextGenModelOutput."""
        z_hat, z_sig, dom, e_vis, pv_phys = self._extract_batch_tensors(batch)
        bsz = int(z_hat.shape[0])
        dev = z_hat.device

        with torch.amp.autocast(device_type=dev.type, enabled=False):
            ctx = self._encode_ctx(dom, e_vis)
            log_prior8 = F.log_softmax(self.prior_head(ctx), dim=-1)  # (B, 8)

            # Covariance marginalization over Sigma_z(x) using deterministic symmetric antithetic stencil
            k_marg = 1 if self.training else self.n_marg_eval
            if k_marg <= 1:
                z_samples = z_hat.unsqueeze(1)  # (B, 1, D)
            else:
                # Deterministic orthogonal probe stencil (0, +e_even, -e_even, +e_odd, -e_odd, ...)
                # zero-randomness inference guarantees deterministic bit-exact Gate-A invariance
                probes = [torch.zeros_like(z_hat)]
                for m in range(1, k_marg):
                    sgn = 1.0 if (m % 2 == 1) else -1.0
                    phase = torch.cos(torch.arange(self.stat_dim, device=dev, dtype=torch.float32) * (0.73 * ((m + 1) // 2)))
                    probes.append(sgn * 0.75 * z_sig * phase.unsqueeze(0))
                z_samples = torch.stack([z_hat + p for p in probes], dim=1)  # (B, K, D)

            # Evaluate all 8 processes across B * K samples in one vectorized batch
            n_bk = bsz * k_marg
            z_flat = z_samples.reshape(n_bk, self.stat_dim)
            dom_k = dom.unsqueeze(1).expand(bsz, k_marg).reshape(n_bk)
            e_k = e_vis.unsqueeze(1).expand(bsz, k_marg).reshape(n_bk)

            z_rep = z_flat.unsqueeze(0).expand(self.n_processes, n_bk, self.stat_dim).reshape(self.n_processes * n_bk, self.stat_dim)
            p_rep = torch.arange(self.n_processes, device=dev, dtype=torch.long).unsqueeze(1).expand(self.n_processes, n_bk).reshape(-1)
            dom_rep = dom_k.unsqueeze(0).expand(self.n_processes, n_bk).reshape(-1)
            e_rep = e_k.unsqueeze(0).expand(self.n_processes, n_bk).reshape(-1)
            lp_all = self.log_prob_given_process(z_rep, p_rep, dom_rep, e_rep).view(self.n_processes, bsz, k_marg)
            log_lik8 = (torch.logsumexp(lp_all, dim=2) - math.log(float(k_marg))).transpose(0, 1)  # (B, 8)

            # Generative log-posterior (dimension-calibrated per-event log density)
            lik_scale = torch.exp(self.log_lik_scale).clamp(0.02, 1.0)
            logits8 = log_lik8 * lik_scale + log_prior8
            # Map 8 processes -> canonical 7 classes [e, mu-, mu+, tau_e, tau_mu, tau_had, NC]
            nc_merged = torch.logsumexp(logits8[:, 6:8], dim=-1, keepdim=True)
            logits7 = torch.cat([logits8[:, :6], nc_merged], dim=-1)
            probs4 = marginal_four_from_seven(logits7, is_probs=False)

            charm_logit = logits8[:, 7] - torch.logsumexp(logits8[:, :7], dim=-1)
            tau_llr = torch.logsumexp(logits8[:, 3:6], dim=-1) - torch.logsumexp(
                torch.cat([logits8[:, :3], logits8[:, 6:8]], dim=-1), dim=-1
            )

            # Calibrated PV prediction
            vtx_in = torch.cat([((z_hat - self.z_mean) / self.z_std.clamp_min(1e-3)).clamp(-8.0, 8.0), ctx[:, :8]], dim=-1)
            v_delta = self.vtx_calib_head(vtx_in)
            v_pv_mm = pv_phys + v_delta[:, :3]
            scale_pv_mm = F.softplus(v_delta[:, 3:6]).clamp(0.03, 180.0) + 0.03

        return NextGenModelOutput(
            logits7=logits7,
            probs4=probs4,
            charm_logit=charm_logit,
            v_pv_mm=v_pv_mm,
            scale_pv_mm=scale_pv_mm,
            aux={
                'logits8': logits8,
                'log_lik8': log_lik8,
                'log_prior8': log_prior8,
                'tau_llr': tau_llr,
            },
        )


def labels_to_diffphys_process8(
    label_7: torch.Tensor,
    label_4: torch.Tensor,
    charm_flag: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map (label_7, label_4, charm_flag) to 8-class generative process target in {0..7}."""
    y7 = label_7.long()
    y4 = label_4.long()
    proc = torch.where(
        (y7 >= 0) & (y7 <= 6),
        y7,
        torch.where(y4 == 0, 0, torch.where(y4 == 1, 1, torch.where(y4 == 2, 5, 6))),
    )
    if charm_flag is not None:
        ch = charm_flag.long() > 0
        # Assign charm events (including NC_charm or CC_charm) to Charm process 7 for generative tail isolation
        proc = torch.where(ch & (y4 != 2), torch.full_like(proc, 7), proc)
    return proc.clamp(0, 7)


def diffphys_sbi_loss(
    model: DiffPhysSBIModel,
    batch: dict[str, torch.Tensor],
    *,
    lambda_gen: float = 0.25,
    lambda_disc: float = 1.2,
    lambda_charm: float = 0.25,
    lambda_vtx: float = 0.15,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined generative NLL (with covariance jitter) + discriminative LLR CE + Charm + PV loss."""
    out = model.forward_NextGen(batch)
    z_hat, z_sig, dom, e_vis, _ = model._extract_batch_tensors(batch)
    y4 = batch['label'].long()
    y7 = batch['label_7'].long() if 'label_7' in batch else torch.where(y4 == 0, 0, torch.where(y4 == 1, 1, torch.where(y4 == 2, 5, 6)))
    cf = batch['charm_flag'].long() if 'charm_flag' in batch else torch.zeros_like(y4)
    cv = batch['charm_valid'].long() if 'charm_valid' in batch else torch.ones_like(y4)
    proc8 = labels_to_diffphys_process8(y7, y4, cf)

    # Generative NLL on covariance-sampled observation z ~ N(z_hat, Sigma_z)
    z_noisy = z_hat + 0.5 * z_sig * torch.randn_like(z_hat)
    log_p_true = model.log_prob_given_process(z_noisy, proc8, dom, e_vis)
    l_gen = (-log_p_true / float(model.stat_dim)).mean()

    # Discriminative 7-class marginal CE
    l_disc = marginal_tau_branch_ce(out.logits7, y7, y4)
    l_charm = focal_charm_bce(out.charm_logit, cf, cv)

    if 'vertex_xyz_mm' in batch and 'vertex_valid' in batch:
        l_vtx = student_t_vertex_loss(out.v_pv_mm, out.scale_pv_mm, batch['vertex_xyz_mm'], batch['vertex_valid'])
    else:
        l_vtx = out.v_pv_mm.square().mean() * 0.01

    total = lambda_gen * l_gen + lambda_disc * l_disc + lambda_charm * l_charm + lambda_vtx * l_vtx
    return total, {
        'loss_total': float(total.detach().item()),
        'loss_gen': float(l_gen.detach().item()),
        'loss_disc': float(l_disc.detach().item()),
        'loss_charm': float(l_charm.detach().item()),
        'loss_vtx': float(l_vtx.detach().item()),
    }
