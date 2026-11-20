"""Magnetic spectrometer y-z bending sagitta kinematics & back-extrapolation (V1 B1++)."""

from __future__ import annotations

import math
import numpy as np


TRACKER_Z_MM = (47.4, 1237.4, 2427.4)
DELTA_Z_MM = 1190.0
# Exact Pinpoint_G4/Pinpoint/include/DetectorConstruction.hh magnet geometry (B_x = 0.57 T)
# Magnet0: center -815.3 mm, length 1500 mm -> [-1565.3, -65.3] mm
# Magnet1: center +637.4 mm, length 1000 mm -> [+137.4, +1137.4] mm
# Magnet2: center +1837.4 mm, length 1000 mm -> [+1337.4, +2337.4] mm
MAGNET0_Z_CENTER_MM = -815.3
MAGNET0_LENGTH_MM = 1500.0
MAGNET0_Z_MIN_MM = MAGNET0_Z_CENTER_MM - 0.5 * MAGNET0_LENGTH_MM  # -1565.3 mm
MAGNET0_Z_MAX_MM = MAGNET0_Z_CENTER_MM + 0.5 * MAGNET0_LENGTH_MM  # -65.3 mm
MAGNET1_Z_MIN_MM = 137.4
MAGNET1_Z_MAX_MM = 1137.4
MAGNET2_Z_MIN_MM = 1337.4
MAGNET2_Z_MAX_MM = 2337.4
MAGNET_BORE_X_MM = 0.0
MAGNET_BORE_Y_MM = -10.0
MAGNET_BORE_R_MM = 100.0
# Fiducial bore radius (mm): excludes grazing/Sm2Co17-shell trajectories (Stage 324-06)
MAGNET_BORE_FID_R_MM = 96.0
# Exact piecewise-magnet sagitta integral between Trackers 1, 2, 3:
#   y1 - y0 = theta_y0 * 1190 + 600,000 * kappa
#   y2 - y0 = theta_y0 * 2380 + 2,380,000 * kappa
#   s_y = y1 - 0.5*(y0 + y2) = -590,000 * kappa  =>  kappa = -s_y / 590,000 mm^-1
#   theta_y0 = (y1 - y0) / 1190 + (60 / 59) * (s_y / 1190)
SAGITTA_INTEGRAL_MM2 = 590000.0
ENTRY_SLOPE_SAGITTA_COEFF = 60.0 / 59.0
B_FIELD_TESLA = 0.57
# Exact |p * s_y| = 590,000 mm^2 * 0.299792458 MeV/(T*mm) * 0.57 T = 100,820.18 MeV*mm
SAGITTA_CONST_MEV_MM = SAGITTA_INTEGRAL_MM2 * 0.299792458 * B_FIELD_TESLA

SIGMA_X_MM = 0.58
SIGMA_Y_MM = 0.017
SIGMA_SAGITTA_Y_MM = math.sqrt(1.5) * SIGMA_Y_MM
SIGMA_DELTA_X_MM = math.sqrt(1.5) * SIGMA_X_MM

# Candidate track feature indices that negate under Y-mirror (y -> 2*y_c - y)
CAND_Y_FLIP_IDX = (3, 5, 6, 7, 11)
# Raw hit feature indices that negate under Y-mirror
HIT_Y_FLIP_IDX = (2, 4)


def check_magnet12_bore_traversal(
    x0: np.ndarray,
    y0: np.ndarray,
    y1: np.ndarray,
    theta_x: np.ndarray,
    theta_y_entry: np.ndarray,
    kappa: np.ndarray,
    fid_radius_mm: float = MAGNET_BORE_FID_R_MM,
) -> tuple[np.ndarray, np.ndarray]:
    """Verify whether 3-station trajectories stay inside Magnet1 & Magnet2 air bores (Stage 324-06).

    Evaluates exact piecewise parabolic trajectory at entry/mid/exit of Magnet1 [137.4, 1137.4] mm
    and Magnet2 [1337.4, 2337.4] mm around bore center (0, -10) mm, R_bore = 100 mm.
    Returns (in_bore_mask, max_r_mm).
    """
    z0 = TRACKER_Z_MM[0]
    z1 = TRACKER_Z_MM[1]
    theta_y1 = theta_y_entry + kappa * 1000.0
    # Sample 6 critical z stations: Magnet1 (in, mid, out) and Magnet2 (in, mid, out)
    samples = (
        (MAGNET1_Z_MIN_MM, y0 + theta_y_entry * (MAGNET1_Z_MIN_MM - z0)),
        (0.5 * (MAGNET1_Z_MIN_MM + MAGNET1_Z_MAX_MM), y0 + theta_y_entry * (637.4 - z0) + 0.5 * kappa * (500.0 ** 2)),
        (MAGNET1_Z_MAX_MM, y0 + theta_y_entry * (MAGNET1_Z_MAX_MM - z0) + 0.5 * kappa * (1000.0 ** 2)),
        (MAGNET2_Z_MIN_MM, y1 + theta_y1 * (MAGNET2_Z_MIN_MM - z1)),
        (0.5 * (MAGNET2_Z_MIN_MM + MAGNET2_Z_MAX_MM), y1 + theta_y1 * (1837.4 - z1) + 0.5 * kappa * (500.0 ** 2)),
        (MAGNET2_Z_MAX_MM, y1 + theta_y1 * (MAGNET2_Z_MAX_MM - z1) + 0.5 * kappa * (1000.0 ** 2)),
    )
    max_r = np.zeros_like(x0, dtype=np.float64)
    for zs, ys in samples:
        xs = x0 + theta_x * (zs - z0)
        rs = np.hypot(xs - MAGNET_BORE_X_MM, ys - MAGNET_BORE_Y_MM)
        max_r = np.maximum(max_r, rs)
    return (max_r <= float(fid_radius_mm)), max_r


def back_extrapolate_y_through_magnet0(
    y0: np.ndarray,
    theta_y_entry: np.ndarray,
    kappa: np.ndarray,
    z_target: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact C^1 analytical back-extrapolation from Tracker 1 (z=47.4 mm) through Magnet0 [-1565.3, -65.3] mm.

    Returns (y_at_z_target, theta_y_at_z_target) in (mm, rad).
    """
    dz = z_target - TRACKER_Z_MM[0]
    y_lin = y0 + theta_y_entry * dz
    if z_target >= MAGNET0_Z_MAX_MM:
        return y_lin, theta_y_entry
    if z_target <= MAGNET0_Z_MIN_MM:
        y_ext = y_lin - kappa * MAGNET0_LENGTH_MM * (z_target - MAGNET0_Z_CENTER_MM)
        ty_ext = theta_y_entry - kappa * MAGNET0_LENGTH_MM
        return y_ext, ty_ext
    u = MAGNET0_Z_MAX_MM - z_target
    y_ext = y_lin + 0.5 * kappa * (u * u)
    ty_ext = theta_y_entry - kappa * u
    return y_ext, ty_ext


def extract_spectrometer_kinematics(
    tracker_obs: dict,
    cal_start: tuple[float, float, float, bool],
    bounds: dict,
    max_candidates: int = 4,
) -> dict:
    """Extract Top-K 3-station y-z bending triplets and 6-dim raw hit features."""
    xy = np.asarray(tracker_obs['xy_mm'], dtype=np.float64).reshape(-1, 2)
    station = np.asarray(tracker_obs['station'], dtype=np.int64)
    n_hits = len(station)

    x_c = float(bounds['x_pix_center'])
    y_c = float(bounds['y_scint_center'])
    z_start, x_start, y_start, has_start = cal_start

    cand = np.zeros((max_candidates, 14), dtype=np.float32)
    cand_mask = np.zeros(max_candidates, dtype=bool)

    if n_hits == 0:
        return {
            'cand': cand,
            'cand_mask': cand_mask,
            'hits': np.zeros((0, 6), dtype=np.float32),
        }

    # Build 6-dim raw hit features
    z_trk = np.array([TRACKER_Z_MM[int(s)] for s in station], dtype=np.float64)
    hits = np.zeros((n_hits, 6), dtype=np.float32)
    hits[:, 0] = (station / 2.0).astype(np.float32)
    hits[:, 1] = ((xy[:, 0] - x_c) / 150.0).astype(np.float32)
    hits[:, 2] = ((xy[:, 1] - y_c) / 150.0).astype(np.float32)
    if has_start:
        dz = np.maximum(100.0, z_trk - z_start)
        hits[:, 3] = np.clip((xy[:, 0] - x_start) / dz, -2.0, 2.0).astype(np.float32) * 10.0
        hits[:, 4] = np.clip((xy[:, 1] - y_start) / dz, -2.0, 2.0).astype(np.float32) * 10.0
        hits[:, 5] = 1.0

    # Partition hits by station 0, 1, 2 (cap per-station pool to keep worst-case bounded)
    idx_s = [np.flatnonzero(station == s) for s in (0, 1, 2)]
    max_per_station = 14
    for s in (0, 1, 2):
        ix = idx_s[s]
        if len(ix) > max_per_station:
            ref_x = x_start if has_start else x_c
            ref_y = y_start if has_start else y_c
            dist2 = (xy[ix, 0] - ref_x) ** 2 + (xy[ix, 1] - ref_y) ** 2
            idx_s[s] = ix[np.argsort(dist2, kind='mergesort')[:max_per_station]]

    used_by_station: dict[int, set[int]] = {0: set(), 1: set(), 2: set()}
    slot = 0

    if all(len(idx_s[s]) > 0 for s in (0, 1, 2)):
        i0, i1, i2 = idx_s
        g0, g1, g2 = np.meshgrid(i0, i1, i2, indexing='ij')
        g0, g1, g2 = g0.ravel(), g1.ravel(), g2.ravel()

        x0, y0 = xy[g0, 0], xy[g0, 1]
        x1, y1 = xy[g1, 0], xy[g1, 1]
        x2, y2 = xy[g2, 0], xy[g2, 1]

        # Bending plane is y-z (B along +x); non-bending straight-line plane is x-z
        s_y = y1 - 0.5 * (y0 + y2)
        kappa = -s_y / SAGITTA_INTEGRAL_MM2
        delta_x = x1 - 0.5 * (x0 + x2)
        chi2_x = (delta_x / SIGMA_DELTA_X_MM) ** 2
        theta_x = (x2 - x0) / (2.0 * DELTA_Z_MM)
        # Exact entry slope at Tracker 1 derived from Pinpoint_G4 Magnet1 & Magnet2 integrals
        theta_y_entry = (y1 - y0) / DELTA_Z_MM + ENTRY_SLOPE_SAGITTA_COEFF * (s_y / DELTA_Z_MM)
        # Stage 324-06 fix: bore-aware quality check rejecting Sm2Co17 shell-scattered wrong-sign sagitta
        in_bore, _ = check_magnet12_bore_traversal(x0, y0, y1, theta_x, theta_y_entry, kappa)
        kappa_eff = np.where(in_bore, kappa, 0.0)

        if has_start:
            dz_ext = z_start - TRACKER_Z_MM[0]
            dx_ext = (x0 + theta_x * dz_ext) - x_start
            y_ext, _ = back_extrapolate_y_through_magnet0(y0, theta_y_entry, kappa_eff, float(z_start))
            dy_ext = y_ext - y_start
            d_ext = np.hypot(dx_ext / 20.0, dy_ext / 20.0)
        else:
            dx_ext = np.zeros_like(s_y)
            dy_ext = np.zeros_like(s_y)
            d_ext = np.zeros_like(s_y)

        bore_pen = np.where(in_bore, 0.0, 25.0)
        q_score = chi2_x + bore_pen + (0.5 * (d_ext ** 2) if has_start else 0.0) + (np.abs(s_y) / 20.0) ** 2
        order = np.argsort(q_score, kind='mergesort')

        for k in order:
            a, b, c = int(g0[k]), int(g1[k]), int(g2[k])
            if a in used_by_station[0] or b in used_by_station[1] or c in used_by_station[2]:
                continue
            used_by_station[0].add(a)
            used_by_station[1].add(b)
            used_by_station[2].add(c)
            ib_k = bool(in_bore[k])
            sy_k = float(s_y[k]) if ib_k else 0.0
            p_proxy = (1.0 / (abs(sy_k) / 0.5 + 0.01)) if ib_k else 0.0
            cand[slot] = np.array([
                1.0 if ib_k else 0.70,
                1.0 / (1.0 + 0.1 * float(q_score[k])),
                (float(x0[k]) - x_c) / 150.0,
                (float(y0[k]) - y_c) / 150.0,
                float(theta_x[k]) * 10.0,
                float(theta_y_entry[k]) * 10.0,
                (math.asinh(sy_k / SIGMA_SAGITTA_Y_MM) / 5.0) if ib_k else 0.0,
                (float(np.clip(sy_k, -25.0, 25.0)) / 5.0) if ib_k else 0.0,
                (math.log1p(p_proxy) / 5.0) if ib_k else 0.0,
                math.log1p(float(chi2_x[k])) / 5.0,
                float(np.clip(dx_ext[k], -300.0, 300.0)) / 50.0,
                float(np.clip(dy_ext[k], -300.0, 300.0)) / 50.0,
                math.log1p(float(d_ext[k])) / 3.0,
                (math.log1p(abs(sy_k) / 0.05) / 5.0) if ib_k else 0.0,
            ], dtype=np.float32)
            cand_mask[slot] = True
            slot += 1
            if slot >= max_candidates:
                break

    # 2-station fallback candidates across ((0, 2), (0, 1), (1, 2)) for escaping / incomplete muons
    if slot < max_candidates:
        pair_records = []
        for sa, sb in ((0, 2), (0, 1), (1, 2)):
            rem_a = np.array([ix for ix in idx_s[sa] if int(ix) not in used_by_station[sa]], dtype=np.int64)
            rem_b = np.array([ix for ix in idx_s[sb] if int(ix) not in used_by_station[sb]], dtype=np.int64)
            if len(rem_a) == 0 or len(rem_b) == 0:
                continue
            ga, gb = np.meshgrid(rem_a, rem_b, indexing='ij')
            ga, gb = ga.ravel(), gb.ravel()
            xa, ya = xy[ga, 0], xy[ga, 1]
            xb, yb = xy[gb, 0], xy[gb, 1]
            dz_ab = TRACKER_Z_MM[sb] - TRACKER_Z_MM[sa]
            tx = (xb - xa) / dz_ab
            ty = (yb - ya) / dz_ab
            if has_start:
                dz_ext = z_start - TRACKER_Z_MM[sa]
                dx_ext = (xa + tx * dz_ext) - x_start
                dy_ext = (ya + ty * dz_ext) - y_start
                d_ext = np.hypot(dx_ext / 20.0, dy_ext / 20.0)
            else:
                dx_ext = np.zeros_like(tx)
                dy_ext = np.zeros_like(tx)
                d_ext = np.zeros_like(tx)
            q2 = 10.0 + (0.5 * (d_ext ** 2) if has_start else 0.0) + (tx * 10.0) ** 2 + (ty * 10.0) ** 2
            for k in range(len(ga)):
                pair_records.append((
                    float(q2[k]), sa, sb, int(ga[k]), int(gb[k]),
                    float(xa[k]), float(ya[k]), float(tx[k]), float(ty[k]),
                    float(dx_ext[k]), float(dy_ext[k]), float(d_ext[k]),
                ))
        pair_records.sort(key=lambda r: r[0])
        for rec in pair_records:
            q2_k, sa, sb, a, b, xa_k, ya_k, tx_k, ty_k, dxe_k, dye_k, de_k = rec
            if a in used_by_station[sa] or b in used_by_station[sb]:
                continue
            used_by_station[sa].add(a)
            used_by_station[sb].add(b)
            cand[slot] = np.array([
                0.5,
                1.0 / (1.0 + 0.1 * q2_k),
                (xa_k - x_c) / 150.0,
                (ya_k - y_c) / 150.0,
                tx_k * 10.0,
                ty_k * 10.0,
                0.0,
                0.0,
                0.0,
                float(sb - sa) / 2.0,
                float(np.clip(dxe_k, -300.0, 300.0)) / 50.0,
                float(np.clip(dye_k, -300.0, 300.0)) / 50.0,
                math.log1p(de_k) / 3.0,
                0.0,
            ], dtype=np.float32)
            cand_mask[slot] = True
            slot += 1
            if slot >= max_candidates:
                break

    return {'cand': cand, 'cand_mask': cand_mask, 'hits': hits}
