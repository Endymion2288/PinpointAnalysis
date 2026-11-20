"""Phase 1 minimal representation probes on Pinpoint_G4-consistent synthetic physics."""

from __future__ import annotations

import math
import os
import unittest

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '4')

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import torch

from analysis.flavour.v3.common.domains import (
    compute_observable_containment,
    cumulative_radiation_length,
    detector_bounds,
)
from analysis.flavour.v3.common.fast_kernels import get_fast_lib, reconstruct_micron_vertex
from analysis.flavour.v3.common.layers import ScaledQKNormAttention
from analysis.flavour.v3.common.pixel_clusters import cluster_pixel_hits
from analysis.flavour.v3.v1_b1pp.spectrometer import (
    B_FIELD_TESLA,
    MAGNET1_Z_MAX_MM,
    MAGNET1_Z_MIN_MM,
    MAGNET2_Z_MAX_MM,
    MAGNET2_Z_MIN_MM,
    MAGNET_BORE_Y_MM,
    SAGITTA_CONST_MEV_MM,
    SAGITTA_INTEGRAL_MM2,
    TRACKER_Z_MM,
    extract_spectrometer_kinematics,
)
from analysis.flavour.v3.v2_pinpointformer.models import PinpointFormer
from analysis.flavour.v3.v2_pinpointformer.pixel_tokenizer import (
    apply_y_mirror_pf_item,
    collate_pf,
    extract_pf_item,
)
from analysis.flavour.v3.v2_pinpointformer.train import move_batch_to_device
from analysis.flavour.v3.v2_pinpointformer.vertex_anchor import (
    compute_near_vertex_pair_features,
    compute_stage2_polar_features,
)
from analysis.flavour.v3.v3_multitask.models import PinpointFormerMultitask
from tests.test_nextgen_seal_and_splits import build_minimal_geometry


class SyntheticPinpointPhysicsGenerator:
    """Fast synthetic Pinpoint_G4-aligned track, vertex, and decay topology generator."""

    def __init__(self, geometry: dict | None = None, seed: int = 20261120) -> None:
        self.geometry = geometry if geometry is not None else build_minimal_geometry()
        self.bounds = detector_bounds(self.geometry)
        self.rng = np.random.default_rng(seed)
        self.px = np.asarray(self.geometry['pixel_Xpos'], dtype=np.float64)
        self.py = np.asarray(self.geometry['pixel_Ypos'], dtype=np.float64)
        self.pz = np.asarray(self.geometry['pixel_Zpos'], dtype=np.float64)
        self.x0_planes = np.array(
            [cumulative_radiation_length(float(z), self.geometry) for z in self.pz],
            dtype=np.float64,
        )

    def _empty_bar_and_tracker(self, z_bar_hint: float | None = None) -> tuple[dict, dict]:
        vp_indices = np.asarray(self.bounds['vert_panels'], dtype=np.int64)
        sz = np.asarray(self.geometry['scint_Zpos'], dtype=np.float64)
        if z_bar_hint is None:
            vp = int(vp_indices[1])
        else:
            after = vp_indices[sz[vp_indices] >= float(z_bar_hint)]
            vp = int(after[0]) if len(after) else int(vp_indices[-1])
        hp_indices = np.asarray(self.bounds['horiz_panels'], dtype=np.int64)
        after_h = hp_indices[sz[hp_indices] >= float(sz[vp])]
        hp = int(after_h[0]) if len(after_h) else int(hp_indices[0])
        xc = self.geometry['scint_bar_Xcenter'][12]
        yc = self.geometry['scint_bar_Ycenter'][20]
        bar = {
            'channel_id': np.array([1, 2], dtype=np.int64),
            'layer': np.array([vp, hp], dtype=np.int16),
            'panel': np.array([vp, hp], dtype=np.int16),
            'horizontal': np.array([0, 1], dtype=np.int8),
            'bar': np.array([12, 20], dtype=np.int16),
            'bounds_mm': np.array([
                [xc - 5.25, xc + 5.25, -222.0, 198.0, self.geometry['scint_Zpos'][vp]],
                [-133.0, 287.0, yc - 5.25, yc + 5.25, self.geometry['scint_Zpos'][hp]],
            ], dtype=np.float32),
            'pe_proxy': np.array([400, 420], dtype=np.int64),
            'hit_time_ns': np.array([3.5, 4.1], dtype=np.float32),
        }
        tracker = {
            'station': np.empty(0, dtype=np.int8),
            'xy_mm': np.empty((0, 2), dtype=np.float32),
            'sigma_xy_mm': np.empty((0, 2), dtype=np.float32),
        }
        return bar, tracker

    def propagate_track(
        self,
        origin_xyz_mm: np.ndarray,
        slope_xy: tuple[float, float],
        *,
        planes: list[int] | tuple[int, ...],
        momentum_gev: float | None = None,
        hit_drop_prob: float = 0.0,
        cluster_2px_prob: float = 0.25,
    ) -> list[tuple[int, int, int]]:
        """Propagate a track with optional Highland MCS across given pixel planes."""
        x, y, z = float(origin_xyz_mm[0]), float(origin_xyz_mm[1]), float(origin_xyz_mm[2])
        tx, ty = float(slope_xy[0]), float(slope_xy[1])
        nx, ny = int(self.bounds['nx']), int(self.bounds['ny'])
        dx, dy = float(self.bounds['dx_mm']), float(self.bounds['dy_mm'])
        px_min, py_min = float(self.bounds['px_min']), float(self.bounds['py_min'])

        hits: list[tuple[int, int, int]] = []
        prev_x0 = cumulative_radiation_length(z, self.geometry)
        prev_z = z

        for pl in planes:
            z_pl = float(self.pz[pl])
            if z_pl <= prev_z:
                continue
            dz = z_pl - prev_z
            if momentum_gev is not None and momentum_gev > 0.0:
                dx0 = max(1e-4, float(self.x0_planes[pl]) - prev_x0)
                theta0 = (0.0136 / float(momentum_gev)) * math.sqrt(dx0) * (1.0 + 0.038 * math.log(dx0))
                kx = float(self.rng.normal(0.0, theta0))
                ky = float(self.rng.normal(0.0, theta0))
                x += (tx + 0.5 * kx) * dz
                y += (ty + 0.5 * ky) * dz
                tx += kx
                ty += ky
                prev_x0 = float(self.x0_planes[pl])
            else:
                x += tx * dz
                y += ty * dz
            prev_z = z_pl

            if hit_drop_prob > 0.0 and self.rng.random() < hit_drop_prob:
                continue

            col = int(math.floor((x - px_min) / dx))
            row = int(math.floor((y - py_min) / dy))
            if 0 <= col < nx and 0 <= row < ny:
                hits.append((pl, row, col))
                if cluster_2px_prob > 0.0 and self.rng.random() < cluster_2px_prob and (col + 1) < nx:
                    hits.append((pl, row, col + 1))
        return hits

    def hits_to_observation(self, hits: list[tuple[int, int, int]], z_bar_hint: float | None = None) -> dict:
        """Convert (plane, row, col) pixel hits to canonical Pinpoint observation dict."""
        nx, ny = int(self.bounds['nx']), int(self.bounds['ny'])
        uniq_hits = sorted(set(hits))
        if uniq_hits:
            cids = np.array([(p * ny + r) * nx + c for p, r, c in uniq_hits], dtype=np.int64)
            xyzs = np.array([[self.px[c], self.py[r], self.pz[p]] for p, r, c in uniq_hits], dtype=np.float32)
            if z_bar_hint is None:
                z_bar_hint = float(min(self.pz[p] for p, _, _ in uniq_hits))
        else:
            cids = np.empty(0, dtype=np.int64)
            xyzs = np.empty((0, 3), dtype=np.float32)
        bar, tracker = self._empty_bar_and_tracker(z_bar_hint=z_bar_hint)
        return {
            'pixel': {'channel_id': cids, 'xyz_mm': xyzs},
            'bar': bar,
            'tracker_hypothesis': tracker,
        }

    def generate_prompt_vs_displaced_event(
        self,
        *,
        displaced: bool,
        n_prongs: int = 1,
        dz_sv_mm: float | None = None,
        d0_target_mm: float | None = None,
        add_crossing_bg: bool = False,
        hit_drop_prob: float = 0.0,
        momentum_gev: float = 80.0,
        planes: tuple[int, ...] = (1, 2, 3),
    ) -> tuple[dict, float]:
        """Generate a front-module PV event with optional displaced 1p/3p secondary vertex."""
        pv_x = float(self.rng.uniform(-10.0, 10.0))
        pv_y = float(self.bounds['y_scint_center'] + self.rng.uniform(-10.0, 10.0))
        # W0/W1 region inside front Pinpoint block 0 (between plane 0 and plane 1)
        pv_z = float(self.pz[0] + 10.5)
        pv = np.array([pv_x, pv_y, pv_z], dtype=np.float64)

        hits: list[tuple[int, int, int]] = []
        # Three non-collinear stiff prompt primary hadron tracks (d0 = 0) anchoring PV
        base_angles = (0.0, 2.1, 4.2)
        for ang_b in base_angles:
            ang = ang_b + float(self.rng.uniform(-0.20, 0.20))
            r_sl = float(self.rng.uniform(0.040, 0.065))
            sl = (r_sl * math.cos(ang), r_sl * math.sin(ang))
            hits.extend(
                self.propagate_track(
                    pv, sl, planes=planes, momentum_gev=momentum_gev,
                    hit_drop_prob=hit_drop_prob, cluster_2px_prob=0.15,
                )
            )

        if not displaced:
            # Additional prompt primary track(s) from PV (d0 == 0)
            for k in range(max(1, n_prongs)):
                ang = 1.0 + k * 2.0 + float(self.rng.uniform(-0.20, 0.20))
                r_sl = float(self.rng.uniform(0.035, 0.060))
                sl = (r_sl * math.cos(ang), r_sl * math.sin(ang))
                hits.extend(
                    self.propagate_track(
                        pv, sl, planes=planes, momentum_gev=momentum_gev,
                        hit_drop_prob=hit_drop_prob, cluster_2px_prob=0.15,
                    )
                )
            true_d0 = 0.0
        else:
            dz_sv = float(dz_sv_mm if dz_sv_mm is not None else self.rng.uniform(0.5, 5.0))
            d0_val = float(d0_target_mm if d0_target_mm is not None else self.rng.uniform(0.02, 0.30))
            parent_sl = (float(self.rng.uniform(-0.012, 0.012)), float(self.rng.uniform(-0.012, 0.012)))
            sv = np.array([
                pv_x + parent_sl[0] * dz_sv,
                pv_y + parent_sl[1] * dz_sv,
                pv_z + dz_sv,
            ], dtype=np.float64)
            # Exact unclamped d0 = dz_sv * dtheta_req across full [0.02, 0.30] mm x [0.5, 5.0] mm domain
            dtheta_req = min(0.60, d0_val / max(dz_sv, 0.2))
            base_phi = float(self.rng.uniform(0.0, 2.0 * math.pi))
            for k in range(n_prongs):
                phi = base_phi + (2.0 * math.pi * k) / max(1, n_prongs)
                sl = (
                    parent_sl[0] + dtheta_req * math.cos(phi),
                    parent_sl[1] + dtheta_req * math.sin(phi),
                )
                hits.extend(
                    self.propagate_track(
                        sv, sl, planes=planes, momentum_gev=momentum_gev,
                        hit_drop_prob=hit_drop_prob, cluster_2px_prob=0.15,
                    )
                )
            true_d0 = dz_sv * dtheta_req

        if add_crossing_bg:
            # Unrelated far crossing track (>15 mm from PV) plus an overlapping low-angle prompt pair at PV
            bg_orig = np.array([pv_x + 16.0, pv_y - 14.0, pv_z], dtype=np.float64)
            hits.extend(
                self.propagate_track(
                    bg_orig, (-0.02, 0.015), planes=planes, momentum_gev=momentum_gev,
                    hit_drop_prob=hit_drop_prob, cluster_2px_prob=0.15,
                )
            )
            hits.extend(
                self.propagate_track(
                    pv, (0.012, -0.008), planes=planes, momentum_gev=momentum_gev,
                    hit_drop_prob=hit_drop_prob, cluster_2px_prob=0.15,
                )
            )

        return self.hits_to_observation(hits), float(true_d0)

    def generate_1prong_kink_event(
        self,
        *,
        has_kink: bool,
        kink_mrad: float = 45.0,
        hit_drop_prob: float = 0.0,
    ) -> dict:
        """Generate 1-prong parent-stub -> kinked daughter vs straight-track topology."""
        pv_x = float(self.rng.uniform(-8.0, 8.0))
        pv_y = float(self.bounds['y_scint_center'] + self.rng.uniform(-8.0, 8.0))
        pv_z = float(self.pz[0] + 10.5)
        pv = np.array([pv_x, pv_y, pv_z], dtype=np.float64)
        hits: list[tuple[int, int, int]] = []

        # Three prompt stiff tracks establishing PV on planes 1..4 (keep planes 1..2 intact, dropout downstream)
        for sl in ((-0.05, 0.03), (0.04, -0.05), (0.01, 0.06)):
            hits.extend(self.propagate_track(pv, sl, planes=(1, 2), momentum_gev=60.0, hit_drop_prob=0.0))
            last_h = hits[-1]
            p2_pt = np.array([self.px[last_h[2]], self.py[last_h[1]], self.pz[2]], dtype=np.float64)
            hits.extend(
                self.propagate_track(p2_pt, sl, planes=(3, 4), momentum_gev=60.0, hit_drop_prob=hit_drop_prob)
            )

        # Parent tau / straight track starts at PV, hits plane 1, then kinks at z_kink midpoint before plane 2
        tau_sl = (0.015, -0.010)
        hits.extend(self.propagate_track(pv, tau_sl, planes=(1,), momentum_gev=None, hit_drop_prob=0.0))
        dz_to_kink = (float(self.pz[1]) - pv_z) + 15.0  # midpoint between plane 1 and plane 2
        kink_pt = np.array([
            pv_x + tau_sl[0] * dz_to_kink,
            pv_y + tau_sl[1] * dz_to_kink,
            pv_z + dz_to_kink,
        ], dtype=np.float64)
        ang = float(self.rng.uniform(0.0, 2.0 * math.pi))
        dk = (kink_mrad * 1e-3) if has_kink else 0.0
        d_sl = (tau_sl[0] + dk * math.cos(ang), tau_sl[1] + dk * math.sin(ang))
        hits.extend(self.propagate_track(kink_pt, d_sl, planes=(2, 3), momentum_gev=60.0, hit_drop_prob=0.0))
        last_d = hits[-1]
        p3_pt = np.array([self.px[last_d[2]], self.py[last_d[1]], self.pz[3]], dtype=np.float64)
        hits.extend(
            self.propagate_track(
                p3_pt, d_sl, planes=(4, 5), momentum_gev=60.0, hit_drop_prob=hit_drop_prob
            )
        )
        return self.hits_to_observation(hits)


def _fit_line_residuals_mm(xyz: np.ndarray) -> float:
    """Fit 3D straight line x(z), y(z) to hits and return 2D transverse RMS residual (mm)."""
    z = xyz[:, 2]
    zm = z.mean()
    dz = z - zm
    szz = float(np.dot(dz, dz))
    tx = float(np.dot(dz, xyz[:, 0] - xyz[:, 0].mean()) / szz)
    ty = float(np.dot(dz, xyz[:, 1] - xyz[:, 1].mean()) / szz)
    rx = xyz[:, 0] - (xyz[:, 0].mean() + tx * dz)
    ry = xyz[:, 1] - (xyz[:, 1].mean() + ty * dz)
    return float(np.sqrt(np.mean(rx * rx + ry * ry)))


def _reconstruct_3plane_tracklets(
    pix_xyz_mm: np.ndarray,
    pix_plane: np.ndarray,
    planes: tuple[int, int, int] = (1, 2, 3),
    max_slope: float = 0.65,
    max_mid_res_mm: float = 0.08,
) -> np.ndarray:
    """Reconstruct greedy 3-plane tracklets returning (N_trk, 5): [x_at_0, y_at_0, tx, ty, chi2]."""
    p0, p1, p2 = planes
    i0 = np.flatnonzero(pix_plane == p0)
    i1 = np.flatnonzero(pix_plane == p1)
    i2 = np.flatnonzero(pix_plane == p2)
    if len(i0) == 0 or len(i1) == 0 or len(i2) == 0:
        return np.empty((0, 5), dtype=np.float64)

    xyz = pix_xyz_mm.astype(np.float64)
    cands: list[tuple[float, int, int, int, float, float, float, float]] = []
    for a in i0:
        xa, ya, za = xyz[a]
        for c in i2:
            xc, yc, zc = xyz[c]
            dz_ac = zc - za
            tx = (xc - xa) / dz_ac
            ty = (yc - ya) / dz_ac
            if math.hypot(tx, ty) > max_slope:
                continue
            for b in i1:
                xb, yb, zb = xyz[b]
                dz_ab = zb - za
                rx = xb - (xa + tx * dz_ab)
                ry = yb - (ya + ty * dz_ab)
                res = math.hypot(rx, ry)
                if res <= max_mid_res_mm:
                    zm = (za + zb + zc) / 3.0
                    xm = (xa + xb + xc) / 3.0
                    ym = (ya + yb + yc) / 3.0
                    szz = (za - zm) ** 2 + (zb - zm) ** 2 + (zc - zm) ** 2
                    tx_f = ((za - zm) * (xa - xm) + (zb - zm) * (xb - xm) + (zc - zm) * (xc - xm)) / szz
                    ty_f = ((za - zm) * (ya - ym) + (zb - zm) * (yb - ym) + (zc - zm) * (yc - ym)) / szz
                    ax = xm - tx_f * zm
                    ay = ym - ty_f * zm
                    cands.append((res * res, int(a), int(b), int(c), ax, ay, tx_f, ty_f))

    cands.sort(key=lambda t: t[0])
    used: set[int] = set()
    trks: list[list[float]] = []
    for r2, a, b, c, ax, ay, tx_f, ty_f in cands:
        if a in used or b in used or c in used:
            continue
        used.update((a, b, c))
        trks.append([ax, ay, tx_f, ty_f, r2])
    return np.asarray(trks, dtype=np.float64) if trks else np.empty((0, 5), dtype=np.float64)


def _fit_pv_and_tracklet_d0(trks: np.ndarray, z_prior: float) -> tuple[np.ndarray, np.ndarray]:
    """IRLS 3D vertex fit from tracklets [ax, ay, tx, ty, r2] -> (pv_xyz, per_tracklet_d0_mm)."""
    if len(trks) < 2:
        return np.array([0.0, -12.0, z_prior], dtype=np.float64), np.zeros(len(trks), dtype=np.float64)
    w = np.ones(len(trks), dtype=np.float64)
    pv = np.array([0.0, -12.0, z_prior], dtype=np.float64)
    for temp in (0.15, 0.04, 0.015):
        a_mat = np.diag([1e-4, 1e-4, 1.0 / (15.0 ** 2)])
        rhs = np.array([0.0, -12.0 * 1e-4, z_prior / (15.0 ** 2)], dtype=np.float64)
        for i in range(len(trks)):
            wi = w[i]
            ax, ay, tx, ty, _ = trks[i]
            a_mat[0, 0] += wi
            a_mat[0, 2] -= wi * tx
            a_mat[2, 0] -= wi * tx
            a_mat[1, 1] += wi
            a_mat[1, 2] -= wi * ty
            a_mat[2, 1] -= wi * ty
            a_mat[2, 2] += wi * (tx * tx + ty * ty)
            rhs[0] += wi * ax
            rhs[1] += wi * ay
            rhs[2] -= wi * (tx * ax + ty * ay)
        pv = np.linalg.solve(a_mat, rhs)
        dx = (trks[:, 0] + trks[:, 2] * pv[2]) - pv[0]
        dy = (trks[:, 1] + trks[:, 3] * pv[2]) - pv[1]
        d2 = dx * dx + dy * dy
        w = 1.0 / (1.0 + np.exp(np.clip((d2 - temp * temp) / (0.5 * temp * temp), -20.0, 20.0)))
    dx = (trks[:, 0] + trks[:, 2] * pv[2]) - pv[0]
    dy = (trks[:, 1] + trks[:, 3] * pv[2]) - pv[1]
    return pv, np.hypot(dx, dy)


class TestNextgenRepresentationProbes(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(4)
        cls.gen = SyntheticPinpointPhysicsGenerator(seed=20261120)
        cls.geometry = cls.gen.geometry
        cls.bounds = cls.gen.bounds
        cls.fast_lib = get_fast_lib()

    def test_probe_a_straight_and_mcs_tracks_front_and_mid(self) -> None:
        """Straight vs Highland MCS tracks across 30 mm dense front and 150 mm sparse mid modules."""
        self.assertIsNotNone(self.fast_lib, 'C++17 fast_kernels library must be available')

        # 1. Track residuals: straight vs 8 GeV MCS in front (30 mm) and mid (150 mm)
        rms_front_str, rms_front_mcs = [], []
        rms_mid_str, rms_mid_mcs = [], []
        for _ in range(25):
            for mcs_p, out_f, out_m in (
                (None, rms_front_str, rms_mid_str),
                (8.0, rms_front_mcs, rms_mid_mcs),
            ):
                pv_f = np.array([0.0, -12.0, float(self.gen.pz[0] + 10.0)])
                hf = self.gen.propagate_track(pv_f, (0.03, -0.02), planes=(1, 2, 3, 4, 5), momentum_gev=mcs_p, cluster_2px_prob=0.0)
                obs_f = self.gen.hits_to_observation(hf)
                out_f.append(_fit_line_residuals_mm(obs_f['pixel']['xyz_mm']))

                pv_m = np.array([0.0, -12.0, float(self.gen.pz[5] + 60.0)])
                hm = self.gen.propagate_track(pv_m, (0.03, -0.02), planes=(6, 7, 8, 9), momentum_gev=mcs_p, cluster_2px_prob=0.0)
                obs_m = self.gen.hits_to_observation(hm)
                out_m.append(_fit_line_residuals_mm(obs_m['pixel']['xyz_mm']))

        self.assertLess(float(np.mean(rms_front_str)), 0.025)
        self.assertLess(float(np.mean(rms_mid_str)), 0.025)
        self.assertGreater(float(np.mean(rms_front_mcs)), float(np.mean(rms_front_str)) * 2.0)
        self.assertGreater(float(np.mean(rms_mid_mcs)), float(np.mean(rms_front_mcs)) * 3.0)

        # 2. Multi-track PV reconstruction succeeds in both dense front (30 mm) and sparse mid (150 mm)
        for region_pv, planes in (
            (np.array([2.0, -10.0, float(self.gen.pz[0] + 10.5)]), (1, 2, 3, 4)),
            (np.array([2.0, -10.0, float(self.gen.pz[5] + 120.0)]), (6, 7, 8, 9)),
        ):
            hits = []
            for sl in ((-0.05, 0.02), (0.04, 0.03), (-0.01, -0.06)):
                hits.extend(self.gen.propagate_track(region_pv, sl, planes=planes, momentum_gev=30.0))
            obs = self.gen.hits_to_observation(hits)
            dom = compute_observable_containment(obs, self.geometry, self.bounds, {'pe_only': True})
            item = extract_pf_item(0, obs, 2, dom, self.geometry, self.bounds, {'pe_only': True})
            self.assertEqual(float(item['avr_pv_mm'][3]), 1.0, msg=f'PV failed for planes={planes}')
            self.assertGreater(float(item['micro_vtx_feat'][0]), 0.3, msg=f'No tracklets built for planes={planes}')

    def test_probe_b_displaced_sv_20um_vs_1mm_and_mean_pooling(self) -> None:
        """20 um + tracklet IP & relational SV preserve displaced separation that 1 mm or scalar pooling destroys."""
        # 1. Full unclamped brief #4 domain: dz_sv in [0.5, 5.0] mm, d0 in [0.02, 0.30] mm
        n_per_class = 45
        y_true = np.array([0] * n_per_class + [1] * n_per_class, dtype=np.int64)
        score_20um = []
        score_1mm = []
        pooled_16d_feats = []
        sampled_d0s = []
        z_pv_prior = float(self.gen.pz[0] + 10.5)

        for label in y_true:
            obs, true_d0 = self.gen.generate_prompt_vs_displaced_event(
                displaced=bool(label == 1),
                n_prongs=1,
                dz_sv_mm=float(self.gen.rng.uniform(0.5, 5.0)) if label == 1 else 0.0,
                d0_target_mm=float(self.gen.rng.uniform(0.02, 0.30)) if label == 1 else 0.0,
                momentum_gev=150.0,
            )
            if label == 1:
                sampled_d0s.append(true_d0)
            dom = compute_observable_containment(obs, self.geometry, self.bounds, {'pe_only': True})
            it_20 = extract_pf_item(0, obs, int(label), dom, self.geometry, self.bounds, {'pe_only': True})
            it_1m = extract_pf_item(
                0, obs, int(label), dom, self.geometry, self.bounds,
                {'pe_only': True, 'quantize_cluster_coords_only_mm': 1.0},
            )
            trks_20 = _reconstruct_3plane_tracklets(it_20['pix_xyz_mm'], it_20['pix_plane'], max_mid_res_mm=0.08)
            _, d0s_20 = _fit_pv_and_tracklet_d0(trks_20, z_pv_prior)
            score_20um.append(float(d0s_20.max()) if len(d0s_20) else 0.0)

            trks_1m = _reconstruct_3plane_tracklets(it_1m['pix_xyz_mm'], it_1m['pix_plane'], max_mid_res_mm=1.20)
            _, d0s_1m = _fit_pv_and_tracklet_d0(trks_1m, z_pv_prior)
            score_1mm.append(float(d0s_1m.max()) if len(d0s_1m) else 0.0)

            # 16-d scalar mean-pooling over unordered cluster features (destroys tracklet alignment)
            pooled_16d_feats.append(it_20['pix_feat'][:, :16].mean(axis=0))

        # Verify full [0.02, 0.30] mm domain was genuinely sampled without clamping distortion
        self.assertLess(min(sampled_d0s), 0.045)
        self.assertGreater(max(sampled_d0s), 0.22)

        auc_20um = float(roc_auc_score(y_true, score_20um))
        auc_1mm = float(roc_auc_score(y_true, score_1mm))

        X_pool = np.vstack(pooled_16d_feats)
        oof_pool = np.zeros(len(y_true), dtype=np.float64)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for tr, va in cv.split(X_pool, y_true):
            clf = LogisticRegression(C=1.0, max_iter=500).fit(X_pool[tr], y_true[tr])
            oof_pool[va] = clf.predict_proba(X_pool[va])[:, 1]
        auc_pool = float(roc_auc_score(y_true, oof_pool))

        self.assertGreater(auc_20um, 0.85, f'Expected strong 20 um IP separation, got AUC={auc_20um:.3f}')
        self.assertGreater(auc_20um - auc_1mm, 0.15, f'1 mm should degrade separation: 20um={auc_20um:.3f}, 1mm={auc_1mm:.3f}')
        self.assertGreater(auc_20um - auc_pool, 0.15, f'16-d scalar mean-pool should degrade separation: 20um={auc_20um:.3f}, pool={auc_pool:.3f}')

        # 2. Multi-tracklet relational intersection vs global tracklet scalar summary under off-vertex distractors:
        # lbl=0 has 3 prompt tracks + 3 independent off-vertex conversion/crossing distractor tracks (d0 in [0.04, 0.28] mm);
        # lbl=1 has 3 prompt tracks + 3 displaced prongs from ONE shared secondary vertex (dz in [0.8, 5.0] mm, d0 in [0.04, 0.28] mm).
        y_rel = np.array([0] * 40 + [1] * 40, dtype=np.int64)
        rel_pair_scores = []
        global_trk_scalar_feats = []

        for lbl in y_rel:
            pv_x = float(self.gen.rng.uniform(-8.0, 8.0))
            pv_y = float(self.bounds['y_scint_center'] + self.gen.rng.uniform(-8.0, 8.0))
            pv = np.array([pv_x, pv_y, z_pv_prior], dtype=np.float64)
            hits = []
            for ang_b in (0.0, 2.1, 4.2):
                ang = ang_b + float(self.gen.rng.uniform(-0.15, 0.15))
                r_sl = float(self.gen.rng.uniform(0.045, 0.065))
                hits.extend(self.gen.propagate_track(pv, (r_sl * math.cos(ang), r_sl * math.sin(ang)), planes=(1, 2, 3), momentum_gev=150.0, cluster_2px_prob=0.1))

            dz_draw = float(self.gen.rng.uniform(0.5, 5.0))
            d0_draw = float(self.gen.rng.uniform(0.04, 0.28))
            dth = min(0.50, d0_draw / max(dz_draw, 0.2))
            base_phi = float(self.gen.rng.uniform(0.0, 2.0 * math.pi))

            # Shared off-vertex gamma->ee conversion / crossing distractor tracklet on BOTH classes (lbl=0 & lbl=1)
            d0_conv = float(self.gen.rng.uniform(0.35, 1.00))
            ang_conv = float(self.gen.rng.uniform(0.0, 2.0 * math.pi))
            orig_conv = pv + np.array([d0_conv * math.cos(ang_conv), d0_conv * math.sin(ang_conv), 0.0])
            hits.extend(self.gen.propagate_track(orig_conv, (-0.04 * math.sin(ang_conv), 0.04 * math.cos(ang_conv)), planes=(1, 2, 3), momentum_gev=150.0, cluster_2px_prob=0.1))

            if lbl == 1:
                # Coherent 3-prong secondary vertex at pv + [0, 0, dz_draw]
                sv = pv + np.array([0.0, 0.0, dz_draw])
                for k in range(3):
                    phi = base_phi + (2.0 * math.pi * k) / 3.0
                    hits.extend(self.gen.propagate_track(sv, (dth * math.cos(phi), dth * math.sin(phi)), planes=(1, 2, 3), momentum_gev=150.0, cluster_2px_prob=0.1))
            else:
                # 3 incoherent off-vertex conversion/crossing distractor tracks with matched marginal d0 & slope!
                for k in range(3):
                    phi = base_phi + (2.0 * math.pi * k) / 3.0
                    psi = phi + 0.5 * math.pi  # orthogonal transverse shift -> skew lines in 3D with identical d0_draw!
                    orig_k = pv + np.array([d0_draw * math.cos(psi), d0_draw * math.sin(psi), 0.0])
                    hits.extend(self.gen.propagate_track(orig_k, (dth * math.cos(phi), dth * math.sin(phi)), planes=(1, 2, 3), momentum_gev=150.0, cluster_2px_prob=0.1))

            obs_r = self.gen.hits_to_observation(hits)
            dom_r = compute_observable_containment(obs_r, self.geometry, self.bounds, {'pe_only': True})
            it_r = extract_pf_item(0, obs_r, int(lbl), dom_r, self.geometry, self.bounds, {'pe_only': True})
            trks = _reconstruct_3plane_tracklets(it_r['pix_xyz_mm'], it_r['pix_plane'], max_mid_res_mm=0.08)
            pv_fit, d0s = _fit_pv_and_tracklet_d0(trks, z_pv_prior)

            # Global tracklet scalar summary (mean/max of d0, |slope|, residual)
            if len(trks):
                sl_mag = np.hypot(trks[:, 2], trks[:, 3])
                d0_s = np.sort(d0s)[::-1]
                sec_d0 = float(d0_s[1]) if len(d0_s) >= 2 else 0.0
                global_trk_scalar_feats.append([
                    float(len(trks)), float(d0s.mean()), float(d0s.max()), sec_d0,
                    float((d0s > 0.025).sum()), float(sl_mag.mean()), float(sl_mag.max()), float(trks[:, 4].mean()),
                ])
            else:
                global_trk_scalar_feats.append([0.0] * 8)

            # Per-tracklet relational pairwise 3D closest-approach compatibility among displaced candidates
            disp_idx = np.flatnonzero(d0s > 0.020)
            pair_compat = 0.0
            for ii in range(len(disp_idx)):
                for jj in range(ii + 1, len(disp_idx)):
                    ti = trks[disp_idx[ii]]
                    tj = trks[disp_idx[jj]]
                    dax = ti[0] - tj[0]
                    day = ti[1] - tj[1]
                    dtx = ti[2] - tj[2]
                    dty = ti[3] - tj[3]
                    denom = dtx * dtx + dty * dty
                    if denom < 1e-6:
                        continue
                    z_star = -(dax * dtx + day * dty) / denom
                    if (pv_fit[2] + 0.3) <= z_star <= (pv_fit[2] + 8.0):
                        dca_xy = math.hypot(dax + dtx * z_star, day + dty * z_star)
                        pair_compat += math.exp(-dca_xy / 0.020)
            rel_pair_scores.append(pair_compat)

        auc_rel = float(roc_auc_score(y_rel, rel_pair_scores))
        X_g = np.asarray(global_trk_scalar_feats, dtype=np.float64)
        oof_g = np.zeros(len(y_rel), dtype=np.float64)
        for tr, va in cv.split(X_g, y_rel):
            clf_g = LogisticRegression(C=1.0, max_iter=500).fit(X_g[tr], y_rel[tr])
            oof_g[va] = clf_g.predict_proba(X_g[va])[:, 1]
        auc_global_scalars = float(roc_auc_score(y_rel, oof_g))

        self.assertGreater(auc_rel, 0.85, f'Relational multi-tracklet SV compatibility AUC too low: {auc_rel:.3f}')
        self.assertGreater(
            auc_rel - auc_global_scalars,
            0.15,
            f'Global tracklet scalar collapse failed to show bottleneck: rel={auc_rel:.3f}, global={auc_global_scalars:.3f}',
        )

    def test_probe_c_tau_1prong_kink_and_3prong_with_dropout(self) -> None:
        """1-prong kink and 3-prong displaced decay topologies separate from crossing primary tracks & survive dropout."""
        # 1. 1-prong kink vs straight track (clean and with 10% downstream missing-hit drop-out)
        for drop_p, min_auc in ((0.0, 0.90), (0.10, 0.80)):
            y_1p, s_1p = [], []
            for lbl in ([0] * 30 + [1] * 30):
                obs = self.gen.generate_1prong_kink_event(
                    has_kink=bool(lbl == 1),
                    kink_mrad=float(self.gen.rng.uniform(25.0, 75.0)),
                    hit_drop_prob=drop_p,
                )
                dom = compute_observable_containment(obs, self.geometry, self.bounds, {'pe_only': True})
                it = extract_pf_item(0, obs, lbl, dom, self.geometry, self.bounds, {'pe_only': True})
                mv = it['micro_vtx_feat']
                y_1p.append(lbl)
                # Kink angle + displaced daughter IP signal
                s_1p.append(float(mv[15] + 0.35 * mv[4]))
            auc_1p = float(roc_auc_score(y_1p, s_1p))
            self.assertGreater(auc_1p, min_auc, f'1-prong kink AUC (drop={drop_p}) too low: {auc_1p:.3f}')

        # 2. 3-prong displaced decay vs crossing/overlapping primary tracks (clean & with missing-hit drop-out)
        for drop_p, min_auc in ((0.0, 0.90), (0.05, 0.80)):
            y_3p, s_3p = [], []
            for lbl in ([0] * 25 + [1] * 25):
                obs, _ = self.gen.generate_prompt_vs_displaced_event(
                    displaced=bool(lbl == 1),
                    n_prongs=3,
                    dz_sv_mm=float(self.gen.rng.uniform(1.5, 4.5)) if lbl == 1 else 0.0,
                    d0_target_mm=float(self.gen.rng.uniform(0.06, 0.25)) if lbl == 1 else 0.0,
                    add_crossing_bg=bool(lbl == 0),
                    hit_drop_prob=drop_p,
                    planes=(1, 2, 3),
                )
                dom = compute_observable_containment(obs, self.geometry, self.bounds, {'pe_only': True})
                it = extract_pf_item(0, obs, lbl, dom, self.geometry, self.bounds, {'pe_only': True})
                mv = it['micro_vtx_feat']
                y_3p.append(lbl)
                s_3p.append(float(mv[4] + mv[6] + mv[8] + mv[12]))
            auc_3p = float(roc_auc_score(y_3p, s_3p))
            self.assertGreater(auc_3p, min_auc, f'3-prong displaced vs crossing primary AUC (drop={drop_p}) too low: {auc_3p:.3f}')

    def test_probe_d_high_occupancy_stability_n64_vs_n2048(self) -> None:
        """Attention entropy and output variance must not collapse as O(1/N) from N=64 to N=2048."""
        torch.manual_seed(325)
        attn = ScaledQKNormAttention(dim=64, heads=4, init_scale=12.0, drop_path=0.0).eval()
        bsz = 16
        q = torch.randn(bsz, 4, 64)

        variances_attn: dict[int, float] = {}
        variances_mean: dict[int, float] = {}
        entropies: dict[int, float] = {}
        with torch.inference_mode():
            for n_tok in (64, 2048):
                # Background noise tokens + 4 salient signal tokens whose amplitude varies across events
                bg = torch.randn(bsz, n_tok, 64)
                salient = torch.randn(bsz, 4, 64) * 3.0
                bg[:, :4, :] = salient
                mask = torch.ones((bsz, n_tok), dtype=torch.bool)
                out, norm_ent = attn(q, bg, mask, return_entropy=True)
                # Variance across batch of the attended delta relative to query
                delta = out - q
                variances_attn[n_tok] = float(delta.var(dim=0, unbiased=False).mean().item())
                variances_mean[n_tok] = float(bg.mean(dim=1).var(dim=0, unbiased=False).mean().item())
                entropies[n_tok] = float(norm_ent.mean().item())

        # Mean pooling collapses by ~ 64/2048 = 0.03125; ScaledQKNormAttention must resist O(1/N) collapse
        mean_ratio = variances_mean[2048] / max(variances_mean[64], 1e-12)
        attn_ratio = variances_attn[2048] / max(variances_attn[64], 1e-12)
        self.assertLess(mean_ratio, 0.10, f'Control mean-pool did not show expected O(1/N) collapse: {mean_ratio:.4f}')
        self.assertGreater(attn_ratio, 0.25, f'ScaledQKNormAttention variance collapsed O(1/N): ratio={attn_ratio:.4f}')
        self.assertLess(entropies[2048], 0.98, f'Attention entropy saturated to uniform 1/N: {entropies[2048]:.4f}')

    def test_probe_e_permutation_and_detector_symmetry_with_bfield(self) -> None:
        """Permutation invariance and exact X/Y reflection symmetries respecting B_x=+0.57T y-z bending."""
        # 1. Magnetic spectrometer y-z bending sign & Y-reflection charge conjugation (mu+ <-> mu-)
        # For B_x = +0.57 T, v along +z: Lorentz force q(v x B) pushes q>0 toward +y (kappa > 0, s_y < 0).
        p_gev = 80.0
        p_mev = p_gev * 1000.0
        kappa_mu_plus = +(0.299792458 * B_FIELD_TESLA) / p_mev  # +1/mm
        y0 = MAGNET_BORE_Y_MM
        ty0 = 0.002

        def _propagate_spec(y_start: float, ty_start: float, kappa: float) -> np.ndarray:
            # Exact piecewise trajectory across Tracker 1 (47.4), Magnet 1 [137.4, 1137.4],
            # Tracker 2 (1237.4), Magnet 2 [1337.4, 2337.4], Tracker 3 (2427.4)
            z0, z1, z2 = TRACKER_Z_MM
            ym1_in = y_start + ty_start * (MAGNET1_Z_MIN_MM - z0)
            l1 = MAGNET1_Z_MAX_MM - MAGNET1_Z_MIN_MM
            ym1_out = ym1_in + ty_start * l1 + 0.5 * kappa * (l1 ** 2)
            ty1 = ty_start + kappa * l1
            y1 = ym1_out + ty1 * (z1 - MAGNET1_Z_MAX_MM)

            ym2_in = y1 + ty1 * (MAGNET2_Z_MIN_MM - z1)
            l2 = MAGNET2_Z_MAX_MM - MAGNET2_Z_MIN_MM
            ym2_out = ym2_in + ty1 * l2 + 0.5 * kappa * (l2 ** 2)
            ty2 = ty1 + kappa * l2
            y2 = ym2_out + ty2 * (z2 - MAGNET2_Z_MAX_MM)
            return np.array([[0.0, y_start], [0.0, y1], [0.0, y2]], dtype=np.float64)

        xy_plus = _propagate_spec(y0, ty0, +kappa_mu_plus)
        xy_minus = _propagate_spec(y0, ty0, -kappa_mu_plus)
        sy_plus = xy_plus[1, 1] - 0.5 * (xy_plus[0, 1] + xy_plus[2, 1])
        sy_minus = xy_minus[1, 1] - 0.5 * (xy_minus[0, 1] + xy_minus[2, 1])

        self.assertLess(sy_plus, 0.0, 'mu+ must have negative sagitta under B_x = +0.57 T')
        self.assertGreater(sy_minus, 0.0, 'mu- must have positive sagitta under B_x = +0.57 T')
        self.assertAlmostEqual(abs(p_mev * sy_plus), SAGITTA_CONST_MEV_MM, delta=100.0)
        self.assertAlmostEqual(sy_plus, -590000.0 * kappa_mu_plus, places=6)

        # Y-reflection about bore center maps mu+(y, ty) -> mu-(2*yc-y, -ty) and flips s_y -> -s_y
        xy_refl = xy_plus.copy()
        xy_refl[:, 1] = 2.0 * MAGNET_BORE_Y_MM - xy_refl[:, 1]
        trk_plus = {'station': np.array([0, 1, 2]), 'xy_mm': xy_plus, 'sigma_xy_mm': np.full((3, 2), 0.1)}
        trk_refl = {'station': np.array([0, 1, 2]), 'xy_mm': xy_refl, 'sigma_xy_mm': np.full((3, 2), 0.1)}
        sp_p = extract_spectrometer_kinematics(trk_plus, (0.0, 0.0, 0.0, False), self.bounds)
        sp_r = extract_spectrometer_kinematics(trk_refl, (0.0, 0.0, 0.0, False), self.bounds)
        # Sagitta features (indices 6, 7) and y0/ty0 (3, 5) negate; |s_y| momentum proxy (8, 13) & bore flag (0) invariant
        self.assertAlmostEqual(float(sp_p['cand'][0, 6]), -float(sp_r['cand'][0, 6]), places=5)
        self.assertAlmostEqual(float(sp_p['cand'][0, 7]), -float(sp_r['cand'][0, 7]), places=5)
        self.assertAlmostEqual(float(sp_p['cand'][0, 8]), float(sp_r['cand'][0, 8]), places=5)
        self.assertAlmostEqual(float(sp_p['cand'][0, 13]), float(sp_r['cand'][0, 13]), places=5)
        self.assertEqual(float(sp_p['cand'][0, 0]), float(sp_r['cand'][0, 0]))

        # 2. Pixel/cluster permutation invariance & micro-vertex X/Y reflection invariance
        obs, _ = self.gen.generate_prompt_vs_displaced_event(displaced=True, n_prongs=3, dz_sv_mm=2.5, d0_target_mm=0.12)
        dom = compute_observable_containment(obs, self.geometry, self.bounds, {'pe_only': True})
        it_orig = extract_pf_item(0, obs, 2, dom, self.geometry, self.bounds, {'pe_only': True})

        perm = self.gen.rng.permutation(len(obs['pixel']['channel_id']))
        obs_perm = {
            **obs,
            'pixel': {
                'channel_id': obs['pixel']['channel_id'][perm],
                'xyz_mm': obs['pixel']['xyz_mm'][perm],
            },
        }
        dom_perm = compute_observable_containment(obs_perm, self.geometry, self.bounds, {'pe_only': True})
        it_perm = extract_pf_item(0, obs_perm, 2, dom_perm, self.geometry, self.bounds, {'pe_only': True})
        np.testing.assert_allclose(it_orig['pix_xyz_mm'], it_perm['pix_xyz_mm'], atol=1e-6)
        np.testing.assert_allclose(it_orig['pix_feat'], it_perm['pix_feat'], atol=1e-6)
        np.testing.assert_allclose(it_orig['micro_vtx_feat'], it_perm['micro_vtx_feat'], atol=1e-6)

        torch.manual_seed(7)
        dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        pf_model = PinpointFormerMultitask(
            self.geometry,
            {'task': 'T4', 'dim': 32, 'heads': 2, 'n_local_layers': 1, 'n_global_layers': 1, 'k_vtx': 16, 'n_queries': 4, 'drop_path': 0.0, 'enable_micro_vertex': True},
        ).to(dev).eval()
        with torch.inference_mode():
            l_orig = pf_model(move_batch_to_device(collate_pf([it_orig]), dev))['logits'].cpu().numpy()
            l_perm = pf_model(move_batch_to_device(collate_pf([it_perm]), dev))['logits'].cpu().numpy()
        np.testing.assert_allclose(l_orig, l_perm, atol=1e-6)

        # Y-mirror on pixel clusters + Stage-2 polar/pair features
        yc = float(self.bounds['y_scint_center'])
        it_mir = apply_y_mirror_pf_item(it_orig, y_scint_center=yc, n_bars=self.bounds['n_bars'])
        xyz_t = torch.from_numpy(it_orig['pix_xyz_mm']).unsqueeze(0)
        xyz_m = torch.from_numpy(it_mir['pix_xyz_mm']).unsqueeze(0)
        mxy = torch.ones((1, len(it_orig['pix_xyz_mm']), 2), dtype=torch.float32)
        v0 = torch.tensor([[0.5, yc + 1.5, -2980.0]], dtype=torch.float32)
        v0_m = torch.tensor([[0.5, yc - 1.5, -2980.0]], dtype=torch.float32)
        p1 = compute_stage2_polar_features(xyz_t, mxy, v0)
        p2 = compute_stage2_polar_features(xyz_m, mxy, v0_m)
        np.testing.assert_allclose(p1[..., [0, 2, 3, 4, 6, 7]].numpy(), p2[..., [0, 2, 3, 4, 6, 7]].numpy(), atol=1e-5)
        np.testing.assert_allclose(p1[..., [1, 5]].numpy(), -p2[..., [1, 5]].numpy(), atol=1e-5)
        mask = torch.ones((1, len(it_orig['pix_xyz_mm'])), dtype=torch.bool)
        x0 = torch.ones((1, len(it_orig['pix_xyz_mm'])), dtype=torch.float32)
        pf1 = compute_near_vertex_pair_features(xyz_t, x0, mask, v0)
        pf2 = compute_near_vertex_pair_features(xyz_m, x0, mask, v0_m)
        np.testing.assert_allclose(pf1.numpy(), pf2.numpy(), atol=1e-5)

        # X-reflection invariance of micron-scale vertex & IP features (B_x along X -> no x-z bending)
        xc = float(self.bounds['x_pix_center'])
        cl_p = it_orig['pix_plane'].astype(np.int64)
        cl_x = it_orig['pix_xyz_mm'][:, 0].astype(np.float64)
        cl_y = it_orig['pix_xyz_mm'][:, 1].astype(np.float64)
        cl_z = it_orig['pix_xyz_mm'][:, 2].astype(np.float64)
        nh = np.ones(len(cl_p), dtype=np.float64)
        lam2 = np.full(len(cl_p), 1e-4, dtype=np.float64)
        pv_a, ok_a, mv_a = reconstruct_micron_vertex(
            cl_p, cl_x, cl_y, cl_z, nh, lam2, 15, 0.0208, 0.0228, dom['cal_start'], (xc, yc, self.bounds['z_mid'])
        )
        cs_xmir = (dom['cal_start'][0], 2.0 * xc - dom['cal_start'][1], dom['cal_start'][2], dom['cal_start'][3])
        pv_b, ok_b, mv_b = reconstruct_micron_vertex(
            cl_p, 2.0 * xc - cl_x, cl_y, cl_z, nh, lam2, 15, 0.0208, 0.0228, cs_xmir, (xc, yc, self.bounds['z_mid'])
        )
        self.assertEqual(ok_a, ok_b)
        np.testing.assert_allclose(mv_a, mv_b, atol=1e-5)
        self.assertAlmostEqual(float(pv_a[0] + pv_b[0]), 2.0 * xc, places=4)


if __name__ == '__main__':
    unittest.main()
