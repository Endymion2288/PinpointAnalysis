"""Differentiable PINPOINT physics operators: non-exclusive soft tracklets, vertex-free 3D skew-DCA,
Student-t Billoir/Kalman vertexing, relativistic bar ToF, and Magnet-0 IPT-spectrometer interlock.
"""

from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn

from analysis.flavour.v3.v1_b1pp.spectrometer import (
    MAGNET0_LENGTH_MM,
    MAGNET0_Z_CENTER_MM,
    MAGNET0_Z_MAX_MM,
    MAGNET0_Z_MIN_MM,
    SAGITTA_INTEGRAL_MM2,
    SIGMA_SAGITTA_Y_MM,
    TRACKER_Z_MM,
)


SPEED_OF_LIGHT_MM_PER_NS = 299.792458
WLS_EFFECTIVE_SPEED_MM_PER_NS = 160.0
PAIR_TRK_FEAT_DIM = 14
TRK_KIN_DIM = 12
TRK_MORPH_DIM = 8


def highland_scattering_rad(delta_x0: torch.Tensor | np.ndarray | float, p_ref_mev: float = 10000.0):
    """Highland multiple Coulomb scattering RMS angle (rad) across `delta_x0` radiation lengths."""
    base = 13.6 / max(float(p_ref_mev), 100.0)
    if isinstance(delta_x0, torch.Tensor):
        x0 = delta_x0.clamp_min(1e-3)
        return base * torch.sqrt(x0) * (1.0 + 0.038 * torch.log(x0.clamp_min(1.0)))
    x0_np = np.maximum(np.asarray(delta_x0, dtype=np.float64), 1e-3)
    return base * np.sqrt(x0_np) * (1.0 + 0.038 * np.log(np.maximum(x0_np, 1.0)))


def _solve_sym_3x3(mat: torch.Tensor, rhs: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact, fast, batch-vectorized FP32 analytical 3x3 SPD linear solver & inverse diagonal.

    Avoids `torch.linalg.solve` cuSOLVER launch overhead and singular-matrix crashes in BF16/MIG.
    mat: (B, 3, 3) symmetric positive definite; rhs: (B, 3).
    Returns: (sol: (B, 3), inv_diag: (B, 3)).
    """
    a00 = mat[:, 0, 0] + eps
    a01 = mat[:, 0, 1]
    a02 = mat[:, 0, 2]
    a11 = mat[:, 1, 1] + eps
    a12 = mat[:, 1, 2]
    a22 = mat[:, 2, 2] + eps

    c00 = a11 * a22 - a12 * a12
    c01 = -(a01 * a22 - a12 * a02)
    c02 = a01 * a12 - a11 * a02
    c11 = a00 * a22 - a02 * a02
    c12 = -(a00 * a12 - a01 * a02)
    c22 = a00 * a11 - a01 * a01

    det = (a00 * c00 + a01 * c01 + a02 * c02).clamp_min(1e-12)
    inv_det = 1.0 / det

    b0, b1, b2 = rhs[:, 0], rhs[:, 1], rhs[:, 2]
    x0 = (c00 * b0 + c01 * b1 + c02 * b2) * inv_det
    x1 = (c01 * b0 + c11 * b1 + c12 * b2) * inv_det
    x2 = (c02 * b0 + c12 * b1 + c22 * b2) * inv_det

    sol = torch.stack([x0, x1, x2], dim=-1)
    inv_diag = torch.stack([c00 * inv_det, c11 * inv_det, c22 * inv_det], dim=-1).clamp_min(1e-8)
    return sol, inv_diag


def extract_soft_tracklets_single(
    pix_xyz_mm: np.ndarray,
    pix_plane: np.ndarray,
    pix_feat: np.ndarray | None,
    plane_x0: np.ndarray,
    *,
    max_tracklets: int = 64,
    max_per_plane: int = 36,
    pitch_xy_mm: tuple[float, float] = (0.0208, 0.0228),
    max_slope: float = 0.24,
) -> dict[str, np.ndarray]:
    """Extract non-exclusive 3-plane & 2-plane soft tracklets across 30mm front, 150mm mid & 30mm IPT.

    Unlike `_fast_kernels.cpp`, this builder:
      1. NEVER applies greedy cluster exclusivity on upstream planes (`shared_root_flag` explicitly
         marks multiprong tracklets splitting from a common merged cluster on `pa`);
      2. Explicitly checks upstream active planes for a collinear parent stub (`has_upstream_stub`)
         vs an empty neutral gap (`photon_gap_flag`) to separate 1-prong tau kinks from gamma->e+e-;
      3. Propagates exact per-axis measurement + Highland multiple-scattering variances across BOTH
         2.85 X0 (30.01 mm) and 19.98 X0 (150.01 mm) detector spacings;
      4. Is 100% invariant to input cluster permutation and reflection-symmetric under y -> 2*y_c - y.
    """
    kin = np.zeros((max_tracklets, TRK_KIN_DIM), dtype=np.float32)
    morph = np.zeros((max_tracklets, TRK_MORPH_DIM), dtype=np.float32)
    mask = np.zeros(max_tracklets, dtype=bool)
    n_cl = len(pix_xyz_mm)
    if n_cl < 2:
        return {'trk_kin': kin, 'trk_morph': morph, 'trk_mask': mask}

    x = pix_xyz_mm[:, 0].astype(np.float64)
    y = pix_xyz_mm[:, 1].astype(np.float64)
    z = pix_xyz_mm[:, 2].astype(np.float64)
    pl = pix_plane.astype(np.int64)
    n_planes = int(len(plane_x0))

    # Approximate cluster size / minor-axis proxy from pix_feat if present, else uniform binary pitch
    if pix_feat is not None and len(pix_feat) == n_cl and pix_feat.shape[1] >= 10:
        log_nhits = pix_feat[:, 3].astype(np.float64) * 6.0
        sqrt_lam1 = 0.05 * np.sinh(np.clip(pix_feat[:, 7].astype(np.float64) * 3.0, 0.0, 6.0))
        sqrt_lam2 = 0.05 * np.sinh(np.clip(pix_feat[:, 8].astype(np.float64) * 3.0, 0.0, 6.0))
        ecc = pix_feat[:, 9].astype(np.float64)
    else:
        log_nhits = np.full(n_cl, math.log1p(1.0), dtype=np.float64)
        sqrt_lam1 = np.full(n_cl, pitch_xy_mm[0] / math.sqrt(12.0), dtype=np.float64)
        sqrt_lam2 = np.full(n_cl, pitch_xy_mm[1] / math.sqrt(12.0), dtype=np.float64)
        ecc = np.zeros(n_cl, dtype=np.float64)

    sig_px2 = (pitch_xy_mm[0] ** 2) / 12.0 + 1e-6
    sig_py2 = (pitch_xy_mm[1] ** 2) / 12.0 + 1e-6

    # Group clusters by plane with permutation-invariant radial sorting around plane centroid
    plane_groups: list[np.ndarray] = []
    for p in range(n_planes):
        idx = np.flatnonzero(pl == p)
        if len(idx) > 0:
            xc_p = float(np.mean(x[idx]))
            yc_p = float(np.mean(y[idx]))
            r2 = (x[idx] - xc_p) ** 2 + (y[idx] - yc_p) ** 2
            # Symmetric tie-break on |x - xc_p| and z-invariant properties (never raw array index!)
            ord_p = np.lexsort((np.abs(x[idx] - xc_p), -log_nhits[idx], r2))
            idx = idx[ord_p[:max_per_plane]]
        plane_groups.append(idx)

    active_planes = [p for p in range(n_planes) if len(plane_groups[p]) > 0]
    if len(active_planes) < 2:
        return {'trk_kin': kin, 'trk_morph': morph, 'trk_mask': mask}
    first_act_p = active_planes[0]

    records: list[tuple[float, int, np.ndarray, np.ndarray, int]] = []

    # Search both Target (0..11) and IPT (12..14) regimes seamlessly
    for pa in range(n_planes - 1):
        ia_arr = plane_groups[pa]
        if len(ia_arr) == 0:
            continue
        is_pa_ipt = pa >= 12
        for pb in range(pa + 1, min(n_planes, pa + 3)):
            if is_pa_ipt != (pb >= 12):
                continue
            ib_arr = plane_groups[pb]
            if len(ib_arr) == 0:
                continue
            dz_ab = float(z[ib_arr[0]] - z[ia_arr[0]])
            if dz_ab < 5.0 or dz_ab > 185.0:
                continue
            if pb > pa + 1 and dz_ab > 70.0:
                continue
            dx0_ab = max(0.05, float(plane_x0[pb] - plane_x0[pa]))
            th_ms_ab = float(highland_scattering_rad(dx0_ab, p_ref_mev=8000.0))

            for ia in ia_arr:
                xa, ya, za = x[ia], y[ia], z[ia]
                sxa2 = sig_px2 + 0.15 * (sqrt_lam2[ia] ** 2)
                sya2 = sig_py2 + 0.15 * (sqrt_lam2[ia] ** 2)
                for ib in ib_arr:
                    tx_ab = (x[ib] - xa) / dz_ab
                    ty_ab = (y[ib] - ya) / dz_ab
                    sl_ab = math.hypot(tx_ab, ty_ab)
                    if sl_ab > max_slope:
                        continue

                    found_3 = False
                    for pc in range(pb + 1, min(n_planes, pb + 3)):
                        if is_pa_ipt != (pc >= 12):
                            continue
                        ic_arr = plane_groups[pc]
                        if len(ic_arr) == 0:
                            continue
                        dz_bc = float(z[ic_arr[0]] - z[ib])
                        dz_ac = float(z[ic_arr[0]] - za)
                        if dz_bc < 5.0 or dz_bc > 185.0 or dz_ac > 340.0:
                            continue
                        if pc > pb + 1 and dz_bc > 70.0:
                            continue
                        dx0_ac = max(0.10, float(plane_x0[pc] - plane_x0[pa]))
                        lever = (dz_ab * dz_bc) / dz_ac
                        th_mid = float(highland_scattering_rad(min(dx0_ab, max(0.05, float(plane_x0[pc] - plane_x0[pb]))), 8000.0))
                        sig_mid2 = 1.5 * (sig_px2 + sig_py2) + 0.15 * (sqrt_lam2[ib] ** 2) + (2.0 * th_mid * lever) ** 2

                        xc_v = x[ic_arr]
                        yc_v = y[ic_arr]
                        tx_v = (xc_v - xa) / dz_ac
                        ty_v = (yc_v - ya) / dz_ac
                        sl_ok = np.hypot(tx_v, ty_v) <= max_slope
                        if not sl_ok.any():
                            continue
                        rx = x[ib] - (xa + tx_v * dz_ab)
                        ry = y[ib] - (ya + ty_v * dz_ab)
                        chi2_v = (rx * rx + ry * ry) / sig_mid2
                        valid_c = np.flatnonzero(sl_ok & (chi2_v <= 18.0))
                        for k_c in valid_c:
                            ic = int(ic_arr[k_c])
                            chi2_val = float(chi2_v[k_c])
                            zm = (za + z[ib] + z[ic]) / 3.0
                            xm = (xa + x[ib] + x[ic]) / 3.0
                            ym = (ya + y[ib] + y[ic]) / 3.0
                            szz = (za - zm) ** 2 + (z[ib] - zm) ** 2 + (z[ic] - zm) ** 2
                            tx_f = ((za - zm) * (xa - xm) + (z[ib] - zm) * (x[ib] - xm) + (z[ic] - zm) * (x[ic] - xm)) / szz
                            ty_f = ((za - zm) * (ya - ym) + (z[ib] - zm) * (y[ib] - ym) + (z[ic] - zm) * (y[ic] - ym)) / szz
                            x0_f = xm + tx_f * (za - zm)
                            y0_f = ym + ty_f * (za - zm)

                            half_g = min(40.0, 0.5 * dz_ab)
                            ms_pos2 = (0.8 * th_ms_ab * half_g) ** 2
                            sig_tx2 = (sxa2 + sig_px2) / (dz_ac * dz_ac) + (th_ms_ab ** 2)
                            sig_ty2 = (sya2 + sig_py2) / (dz_ac * dz_ac) + (th_ms_ab ** 2)

                            # Check upstream parent stub vs neutral photon gap if pa > first_act_p
                            has_stub = 0.0
                            gap_flag = 0.0
                            if pa > first_act_p and not is_pa_ipt:
                                min_d_up = 1e9
                                for pu in range(first_act_p, pa):
                                    iu_arr = plane_groups[pu]
                                    if len(iu_arr) == 0:
                                        continue
                                    zu = z[iu_arr[0]]
                                    if za - zu > 165.0:
                                        continue
                                    xp = x0_f + tx_f * (zu - za)
                                    yp = y0_f + ty_f * (zu - za)
                                    du = float(np.min(np.hypot(x[iu_arr] - xp, y[iu_arr] - yp)))
                                    if du < min_d_up:
                                        min_d_up = du
                                if min_d_up <= 0.35:
                                    has_stub = 1.0
                                elif min_d_up > 1.20:
                                    gap_flag = 1.0

                            k_vec = np.array([
                                x0_f, y0_f, za, tx_f, ty_f,
                                sxa2 + ms_pos2, sya2 + ms_pos2,
                                sig_tx2, sig_ty2,
                                chi2_val, dx0_ac, dz_ac,
                            ], dtype=np.float32)
                            m_vec = np.array([
                                1.0,  # 3-hit flag
                                1.0 if is_pa_ipt else 0.0,
                                float(log_nhits[ia]) / 5.0,
                                math.asinh(sqrt_lam1[ia] / 0.03) / 3.0,
                                math.asinh(sqrt_lam2[ia] / 0.03) / 3.0,
                                float(ecc[ia]),
                                has_stub,
                                gap_flag,
                            ], dtype=np.float32)
                            sort_score = chi2_val + 25.0 * (tx_f * tx_f + ty_f * ty_f)
                            records.append((sort_score, 3, k_vec, m_vec, int(ia)))
                            found_3 = True

                    if not found_3 and pb == pa + 1 and dz_ab <= 165.0 and sl_ab <= 0.16:
                        half_g = min(45.0, 0.5 * dz_ab)
                        ms_pos2 = (1.5 * th_ms_ab * half_g) ** 2
                        sig_tx2 = 2.0 * (sxa2 + sig_px2) / (dz_ab * dz_ab) + 1.5 * (th_ms_ab ** 2)
                        sig_ty2 = 2.0 * (sya2 + sig_py2) / (dz_ab * dz_ab) + 1.5 * (th_ms_ab ** 2)
                        p_chi2 = 4.0 + 180.0 * (tx_ab * tx_ab + ty_ab * ty_ab)
                        k_vec = np.array([
                            xa, ya, za, tx_ab, ty_ab,
                            2.0 * sxa2 + ms_pos2, 2.0 * sya2 + ms_pos2,
                            sig_tx2, sig_ty2,
                            p_chi2, dx0_ab, dz_ab,
                        ], dtype=np.float32)
                        m_vec = np.array([
                            0.0,
                            1.0 if is_pa_ipt else 0.0,
                            float(log_nhits[ia]) / 5.0,
                            math.asinh(sqrt_lam1[ia] / 0.03) / 3.0,
                            math.asinh(sqrt_lam2[ia] / 0.03) / 3.0,
                            float(ecc[ia]),
                            0.0,
                            0.0,
                        ], dtype=np.float32)
                        records.append((p_chi2 + 20.0, 2, k_vec, m_vec, int(ia)))

    if not records:
        return {'trk_kin': kin, 'trk_morph': morph, 'trk_mask': mask}

    # Permutation-invariant & Y-reflection-symmetric sorting: (-n_hits, sort_score, z0, |tx|)
    records.sort(key=lambda r: (-r[1], round(r[0], 7), round(float(r[2][2]), 4), round(abs(float(r[2][3])), 7)))

    # Allow up to 3 prongs to share a single upstream merged cluster (preserving 3-prong tau/D decays!)
    root_counts: dict[int, int] = {}
    selected: list[tuple[np.ndarray, np.ndarray, int]] = []
    for _, _, kv, mv, root_ia in records:
        c_cnt = root_counts.get(root_ia, 0)
        if c_cnt >= 3:
            continue
        root_counts[root_ia] = c_cnt + 1
        selected.append((kv, mv, root_ia))
        if len(selected) >= max_tracklets:
            break

    for idx_out, (kv, mv, root_ia) in enumerate(selected):
        kin[idx_out] = kv
        mv_out = mv.copy()
        # Encode shared-root multiplicity indicator seamlessly in slot 0 fractional part or morph
        if root_counts[root_ia] >= 2:
            mv_out[0] = mv_out[0] + 0.5  # 1.0 = isolated 3-hit, 1.5 = shared-root 3-hit; 0.0/0.5 for 2-hit
        morph[idx_out] = mv_out
        mask[idx_out] = True

    return {'trk_kin': kin, 'trk_morph': morph, 'trk_mask': mask}


def extract_soft_tracklets_batch(
    pix_xyz_mm: torch.Tensor,
    pix_plane: torch.Tensor,
    pix_mask: torch.Tensor,
    plane_x0: torch.Tensor,
    pix_feat: torch.Tensor | None = None,
    *,
    max_tracklets: int = 64,
) -> dict[str, torch.Tensor]:
    """Batch wrapper extracting `(trk_kin, trk_morph, trk_mask)` on CPU/GPU tensors."""
    b = int(pix_xyz_mm.shape[0])
    device = pix_xyz_mm.device
    xyz_np = pix_xyz_mm.detach().cpu().numpy()
    pl_np = pix_plane.detach().cpu().numpy()
    msk_np = pix_mask.detach().cpu().numpy()
    x0_np = plane_x0.detach().cpu().numpy()
    feat_np = pix_feat.detach().cpu().numpy() if pix_feat is not None else None

    kin_b = np.zeros((b, max_tracklets, TRK_KIN_DIM), dtype=np.float32)
    morph_b = np.zeros((b, max_tracklets, TRK_MORPH_DIM), dtype=np.float32)
    mask_b = np.zeros((b, max_tracklets), dtype=bool)

    for i in range(b):
        valid = msk_np[i]
        if not valid.any():
            continue
        f_i = feat_np[i][valid] if feat_np is not None else None
        res = extract_soft_tracklets_single(
            xyz_np[i][valid],
            pl_np[i][valid],
            f_i,
            x0_np,
            max_tracklets=max_tracklets,
        )
        kin_b[i] = res['trk_kin']
        morph_b[i] = res['trk_morph']
        mask_b[i] = res['trk_mask']

    return {
        'trk_kin': torch.from_numpy(kin_b).to(device),
        'trk_morph': torch.from_numpy(morph_b).to(device),
        'trk_mask': torch.from_numpy(mask_b).to(device),
    }


def pairwise_skew_dca_features(
    trk_kin: torch.Tensor,
    trk_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute 14-dim EXACT translation-invariant & Y-mirror-invariant 3D Skew-Line DCA matrix.

    Zero dependence on any primary vertex estimate `v_0`!
    trk_kin: (B, M, 12) [x0, y0, z0, tx, ty, sig_xx, sig_yy, sig_tx, sig_ty, chi2, dx0, span_dz]
    Returns: (B, M, M, 14) in FP32.
    """
    with torch.amp.autocast(device_type=trk_kin.device.type, enabled=False):
        # Compute 3D line-pair geometry in FP64 (64x64 matrix -> <0.05 ms) to prevent FP32 cancellation
        # when z0 ~ -3000 mm and TeV opening angles are ~ 1-5 mrad (||u_a x u_b||^2 ~ 1e-5).
        k = trk_kin.double()
        r0 = k[..., 0:3]   # (B, M, 3)
        tx = k[..., 3]
        ty = k[..., 4]
        sxx = k[..., 5].clamp_min(1e-7)
        syy = k[..., 6].clamp_min(1e-7)
        stx = k[..., 7].clamp_min(1e-8)
        sty = k[..., 8].clamp_min(1e-8)
        span = k[..., 11].clamp_min(1.0)

        u = torch.stack([tx, ty, torch.ones_like(tx)], dim=-1)  # (B, M, 3)
        u_norm = torch.linalg.norm(u, dim=-1, keepdim=True).clamp_min(1e-9)
        u_hat = u / u_norm

        ua = u[:, :, None, :]      # (B, M, 1, 3)
        ub = u[:, None, :, :]      # (B, 1, M, 3)
        dr = r0[:, :, None, :] - r0[:, None, :, :]  # r_a - r_b (strictly translation-invariant)

        cross = torch.linalg.cross(ua, ub, dim=-1)  # (B, M, M, 3)
        cross_sq = (cross * cross).sum(dim=-1)      # (B, M, M) == aa*bb - ab^2
        cross_norm = torch.sqrt(cross_sq + 1e-12)

        # Skew-line DCA = |(r_a - r_b) . (u_a x u_b)| / ||u_a x u_b||
        dca_skew = torch.abs((dr * cross).sum(dim=-1)) / cross_norm

        # Smooth fallback for nearly parallel rays: transverse distance evaluated at midpoint z
        half_dz0 = 0.5 * dr[..., 2]  # z0_a - z_mid0 = +0.5*(z0_a - z0_b)
        dx_mid = dr[..., 0] - tx[:, :, None] * half_dz0 - tx[:, None, :] * half_dz0
        dy_mid = dr[..., 1] - ty[:, :, None] * half_dz0 - ty[:, None, :] * half_dz0
        dca_par = torch.hypot(dx_mid, dy_mid)

        w_skew = cross_sq / (cross_sq + 1e-5)
        d_dca = w_skew * dca_skew + (1.0 - w_skew) * dca_par

        # Closest-approach parameters dz_a, dz_b along z (Tikhonov-regularized for milliradian pairs)
        aa = (ua * ua).sum(dim=-1)
        bb = (ub * ub).sum(dim=-1)
        ab = (ua * ub).sum(dim=-1)
        ra_dot = (dr * ua).sum(dim=-1)
        rb_dot = (dr * ub).sum(dim=-1)
        denom = cross_sq + 1e-5
        dz_a = ((ab * rb_dot - bb * ra_dot) / denom).clamp(-500.0, 500.0)
        dz_b = ((aa * rb_dot - ab * ra_dot) / denom).clamp(-500.0, 500.0)
        # Compute z_ca - min(z0_a, z0_b) purely from relative differences dr[..., 2] (zero large-z cancellation!)
        z_a_minus_min = torch.clamp(dr[..., 2], min=0.0)
        z_b_minus_min = torch.clamp(-dr[..., 2], min=0.0)
        z_rel_upstream = 0.5 * ((z_a_minus_min + dz_a) + (z_b_minus_min + dz_b))

        # Propagated transverse covariance at z_ca for covariance-normalized DCA significance S_DCA
        var_xa = sxx[:, :, None] + (dz_a.square()) * stx[:, :, None]
        var_ya = syy[:, :, None] + (dz_a.square()) * sty[:, :, None]
        var_xb = sxx[:, None, :] + (dz_b.square()) * stx[:, None, :]
        var_yb = syy[:, None, :] + (dz_b.square()) * sty[:, None, :]

        nx = cross[..., 0] / cross_norm
        ny = cross[..., 1] / cross_norm
        n_perp_sq = (nx * nx + ny * ny).clamp_min(1e-4)
        sig_dca_sq = (nx * nx * (var_xa + var_xb) + ny * ny * (var_ya + var_yb)) / n_perp_sq + 4e-5
        s_dca = d_dca / torch.sqrt(sig_dca_sq)

        # Pairwise 3D opening angle & sequential parent->daughter kink compatibility (strictly relative!)
        cos_ang = (u_hat[:, :, None, :] * u_hat[:, None, :, :]).sum(dim=-1).clamp(-1.0 + 1e-9, 1.0 - 1e-9)
        ang_rad = torch.sqrt((2.0 * (1.0 - cos_ang)).clamp_min(1e-14))

        gap_z_ba = -dr[..., 2] - span[:, :, None]  # z_start_b - (z_start_a + span_a)
        is_seq = (gap_z_ba >= -5.0) & (gap_z_ba <= 185.0)
        dz_kink_a = span[:, :, None] + 0.5 * gap_z_ba
        dz_kink_b = -0.5 * gap_z_ba
        dx_kink = dr[..., 0] + tx[:, :, None] * dz_kink_a - tx[:, None, :] * dz_kink_b
        dy_kink = dr[..., 1] + ty[:, :, None] * dz_kink_a - ty[:, None, :] * dz_kink_b
        kink_miss = torch.hypot(dx_kink, dy_kink)

        same_start_z = (torch.abs(dr[..., 2]) < 1.0)
        shared_root = same_start_z & (torch.hypot(dr[..., 0], dr[..., 1]) < 0.08)

        pair_valid = trk_mask[:, :, None] & trk_mask[:, None, :]
        eye = torch.eye(trk_kin.shape[1], dtype=torch.bool, device=trk_kin.device)[None, :, :]
        off_diag = pair_valid & (~eye)
        m_f = off_diag.double()

        feat = torch.stack([
            torch.log1p(d_dca / 0.025) / 4.0 * m_f,
            torch.asinh(d_dca / 0.100) / 3.0 * m_f,
            torch.log1p(s_dca) / 3.5 * m_f,
            torch.exp(-0.5 * (s_dca.clamp_max(10.0).square())) * m_f,
            torch.asinh(z_rel_upstream / 15.0) / 3.0 * m_f,
            (z_rel_upstream < -1.0).double() * m_f,
            torch.log1p(100.0 * ang_rad) / 4.0 * m_f,
            cos_ang * m_f,
            torch.asinh(-dr[..., 2] / 30.0) / 3.0 * m_f,
            is_seq.double() * m_f,
            torch.exp(-kink_miss / 0.10) * is_seq.double() * m_f,
            torch.log1p(200.0 * ang_rad) / 4.0 * torch.exp(-kink_miss / 0.12) * is_seq.double() * m_f,
            shared_root.double() * m_f,
            same_start_z.double() * m_f,
        ], dim=-1).float()
        return feat


class DifferentiableStudentTVertexLayer(nn.Module):
    """Differentiable Student-t (nu=3) IRLS Primary & Secondary Vertex Layer + Kalman Prior Fusion.

    Analytically solves the weighted 3D line-bundle normal equations in FP32 and fuses with the
    calorimeter start prior via inverse covariance:
      - In dense 30mm front Pinpoint: stiff sub-20um tracklets carry precision ~ 10^3 mm^-2,
        dominating the vertex to sub-mm / few-mm accuracy;
      - In sparse 150mm mid Pinpoint or zero-tracklet scintillator events: tracklet precision
        naturally shrinks via Highland MCS covariance, smoothly reverting to `v_cal_prior`
        (eliminating Stage 325 `324-07` 9.92 mm mid-module degradation by mathematical construction!).
    """

    def __init__(self, hidden_dim: int = 128, df: float = 3.0, n_iters: int = 4) -> None:
        super().__init__()
        self.df = float(df)
        self.n_iters = int(n_iters)
        self.pv_weight_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.sv_weight_head = nn.Sequential(
            nn.LayerNorm(hidden_dim + 4),
            nn.Linear(hidden_dim + 4, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.cal_log_prec = nn.Parameter(torch.tensor([-5.5, -5.5, -6.2], dtype=torch.float32))
        self.scale_cal = nn.Parameter(torch.zeros(3, dtype=torch.float32))

    def forward(
        self,
        trk_kin: torch.Tensor,
        trk_emb: torch.Tensor,
        trk_mask: torch.Tensor,
        v_cal_prior_mm: torch.Tensor,
        cal_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        with torch.amp.autocast(device_type=trk_kin.device.type, enabled=False):
            k = trk_kin.float()
            emb = trk_emb.float()
            prior = v_cal_prior_mm.float()
            b, m, _ = k.shape

            x0 = k[..., 0]
            y0 = k[..., 1]
            z0 = k[..., 2]
            tx = k[..., 3]
            ty = k[..., 4]
            sxx = k[..., 5].clamp_min(4e-5)
            syy = k[..., 6].clamp_min(4e-5)
            stx = k[..., 7].clamp_min(1e-7)
            sty = k[..., 8].clamp_min(1e-7)

            pi_pv = torch.sigmoid(self.pv_weight_head(emb).squeeze(-1)) * trk_mask.float()

            # Calibrated prior precision matrix P_cal (diagonal 3x3 in mm^-2)
            base_prec = torch.exp(self.cal_log_prec.clamp(-9.0, 2.0))[None, :]  # (1, 3)
            if cal_valid is not None:
                cv = cal_valid.float().view(b, 1)
                p_prior = cv * base_prec + (1.0 - cv) * 1e-5
            else:
                p_prior = base_prec.expand(b, 3)

            v_cur = prior.clone()
            nu = self.df

            for _ in range(self.n_iters):
                dz = (v_cur[:, None, 2] - z0).clamp(-350.0, 350.0)
                var_x = sxx + dz.square() * stx
                var_y = syy + dz.square() * sty
                rx = (x0 + tx * dz) - v_cur[:, None, 0]
                ry = (y0 + ty * dz) - v_cur[:, None, 1]
                chi2 = (rx.square() / var_x + ry.square() / var_y).clamp_max(1e4)

                w_rob = pi_pv / torch.pow(1.0 + chi2 / nu, 0.5 * (nu + 1.0))
                wx = w_rob / var_x
                wy = w_rob / var_y

                # Exact normal equations for 3D line-to-point distance:
                # r_x(v) = v_x - tx * v_z - (x0 - tx * z0),  r_y(v) = v_y - ty * v_z - (y0 - ty * z0)
                cx = x0 - tx * z0
                cy = y0 - ty * z0

                m00 = p_prior[:, 0] + wx.sum(dim=1)
                m11 = p_prior[:, 1] + wy.sum(dim=1)
                m02 = -(wx * tx).sum(dim=1)
                m12 = -(wy * ty).sum(dim=1)
                m22 = p_prior[:, 2] + (wx * tx.square() + wy * ty.square()).sum(dim=1)
                zeros = torch.zeros_like(m00)

                mat = torch.stack([
                    torch.stack([m00, zeros, m02], dim=-1),
                    torch.stack([zeros, m11, m12], dim=-1),
                    torch.stack([m02, m12, m22], dim=-1),
                ], dim=-2)

                rhs = torch.stack([
                    p_prior[:, 0] * prior[:, 0] + (wx * cx).sum(dim=1),
                    p_prior[:, 1] * prior[:, 1] + (wy * cy).sum(dim=1),
                    p_prior[:, 2] * prior[:, 2] - (wx * tx * cx + wy * ty * cy).sum(dim=1),
                ], dim=-1)

                v_cur, inv_diag = _solve_sym_3x3(mat, rhs)

            v_pv = v_cur
            scale_pv = (torch.sqrt(inv_diag) * torch.exp(self.scale_cal.clamp(-2.0, 3.0))[None, :]).clamp(0.02, 150.0)

            # Compute per-tracklet impact parameter features wrt converged (v_pv, inv_diag)
            dz_f = (v_pv[:, None, 2] - z0).clamp(-350.0, 350.0)
            var_xf = sxx + dz_f.square() * stx + inv_diag[:, None, 0]
            var_yf = syy + dz_f.square() * sty + inv_diag[:, None, 1]
            dx_pv = (x0 + tx * dz_f) - v_pv[:, None, 0]
            dy_pv = (y0 + ty * dz_f) - v_pv[:, None, 1]
            d0 = torch.hypot(dx_pv, dy_pv) * trk_mask.float()
            chi2_pv = (dx_pv.square() / var_xf + dy_pv.square() / var_yf) * trk_mask.float()
            s_d0 = torch.sqrt(chi2_pv.clamp_min(1e-10)) * trk_mask.float()

            # Thrust-axis signed impact parameter (mirror-invariant!)
            w_thr = trk_mask.float()
            ux = (tx * w_thr).sum(dim=1, keepdim=True)
            uy = (ty * w_thr).sum(dim=1, keepdim=True)
            u_n = torch.hypot(ux, uy).clamp_min(1e-5)
            signed_d0 = (-(dx_pv * (ux / u_n) + dy_pv * (uy / u_n))) * trk_mask.float()

            ip_feat = torch.stack([
                torch.asinh(d0 / 0.05) / 3.0,
                torch.log1p(s_d0) / 3.0,
                torch.asinh(signed_d0 / 0.05) / 3.0,
                pi_pv,
            ], dim=-1) * trk_mask.unsqueeze(-1).float()

            # Soft Outlier-Gated Secondary Vertex Fit (2V vs 1V Delta-Chi2)
            sv_in = torch.cat([emb, ip_feat], dim=-1)
            pi_sv = torch.sigmoid(self.sv_weight_head(sv_in).squeeze(-1)) * torch.sigmoid(s_d0 - 2.0) * trk_mask.float()

            # Weak downstream bias prior around v_pv + (0, 0, 2.5 mm)
            sv_prior = v_pv + v_pv.new_tensor([0.0, 0.0, 2.5])[None, :]
            p_sv0 = v_pv.new_tensor([1e-2, 1e-2, 5e-3])[None, :].expand(b, 3)
            v_sv = sv_prior.clone()
            for _ in range(2):
                dz_s = (v_sv[:, None, 2] - z0).clamp(-350.0, 350.0)
                vxs = sxx + dz_s.square() * stx
                vys = syy + dz_s.square() * sty
                wxs = pi_sv / vxs
                wys = pi_sv / vys
                cxs = x0 - tx * z0
                cys = y0 - ty * z0
                ms00 = p_sv0[:, 0] + wxs.sum(dim=1)
                ms11 = p_sv0[:, 1] + wys.sum(dim=1)
                ms02 = -(wxs * tx).sum(dim=1)
                ms12 = -(wys * ty).sum(dim=1)
                ms22 = p_sv0[:, 2] + (wxs * tx.square() + wys * ty.square()).sum(dim=1)
                zs = torch.zeros_like(ms00)
                mat_s = torch.stack([
                    torch.stack([ms00, zs, ms02], dim=-1),
                    torch.stack([zs, ms11, ms12], dim=-1),
                    torch.stack([ms02, ms12, ms22], dim=-1),
                ], dim=-2)
                rhs_s = torch.stack([
                    p_sv0[:, 0] * sv_prior[:, 0] + (wxs * cxs).sum(dim=1),
                    p_sv0[:, 1] * sv_prior[:, 1] + (wys * cys).sum(dim=1),
                    p_sv0[:, 2] * sv_prior[:, 2] - (wxs * tx * cxs + wys * ty * cys).sum(dim=1),
                ], dim=-1)
                v_sv, _ = _solve_sym_3x3(mat_s, rhs_s)

            # Enforce physical forward decay direction dz_sv >= 0 softly
            sv_disp = v_sv - v_pv
            dz_sv_f = (v_sv[:, None, 2] - z0).clamp(-350.0, 350.0)
            dx_sv = (x0 + tx * dz_sv_f) - v_sv[:, None, 0]
            dy_sv = (y0 + ty * dz_sv_f) - v_sv[:, None, 1]
            chi2_sv = (dx_sv.square() / (sxx + dz_sv_f.square() * stx) + dy_sv.square() / (syy + dz_sv_f.square() * sty))
            delta_chi2_2v = (pi_sv * (chi2_pv - chi2_sv).clamp(min=0.0, max=50.0)).sum(dim=1, keepdim=True)

            return {
                'v_pv_mm': v_pv,
                'scale_pv_mm': scale_pv,
                'v_sv_mm': v_sv,
                'sv_disp_mm': sv_disp,
                'delta_chi2_2v': delta_chi2_2v,
                'trk_ip_feat': ip_feat,
                'pi_pv': pi_pv,
                'pi_sv': pi_sv,
            }


def compute_scint_tof_residual_ns(
    hit_time_ns: np.ndarray | torch.Tensor,
    bar_z_mm: np.ndarray | torch.Tensor,
    wls_dist_mm: np.ndarray | torch.Tensor | None = None,
) -> np.ndarray | torch.Tensor:
    """Compute relativistic shower-front time-of-flight residual Delta_t_ToF (ns) for active bars."""
    if isinstance(hit_time_ns, torch.Tensor):
        valid = hit_time_ns >= 0.0
        if not valid.any():
            return torch.zeros_like(hit_time_ns, dtype=torch.float32)
        t = hit_time_ns.float()
        z = bar_z_mm.float()
        d_wls = wls_dist_mm.float() if wls_dist_mm is not None else torch.full_like(t, 210.0)
        t_corr = t - (z / SPEED_OF_LIGHT_MM_PER_NS) - (d_wls / WLS_EFFECTIVE_SPEED_MM_PER_NS)
        t_ref = torch.where(valid, t_corr, torch.full_like(t_corr, 1e9)).min()
        res = torch.where(valid, (t_corr - t_ref).clamp(-5.0, 50.0), torch.zeros_like(t_corr))
        return torch.asinh(res / 0.35) / 3.0

    t_np = np.asarray(hit_time_ns, dtype=np.float64)
    valid_np = t_np >= 0.0
    if not valid_np.any():
        return np.zeros_like(t_np, dtype=np.float32)
    z_np = np.asarray(bar_z_mm, dtype=np.float64)
    dw_np = np.asarray(wls_dist_mm, dtype=np.float64) if wls_dist_mm is not None else np.full_like(t_np, 210.0)
    tc = t_np - (z_np / SPEED_OF_LIGHT_MM_PER_NS) - (dw_np / WLS_EFFECTIVE_SPEED_MM_PER_NS)
    t_ref = float(np.min(tc[valid_np]))
    res_np = np.where(valid_np, np.clip(tc - t_ref, -5.0, 50.0), 0.0)
    return (np.arcsinh(res_np / 0.35) / 3.0).astype(np.float32)


def compute_ipt_spectrometer_interlock(
    ipt_kin: torch.Tensor,
    ipt_mask: torch.Tensor,
    trk_cand: torch.Tensor,
    trk_cand_mask: torch.Tensor,
    *,
    x_pix_center: float = 0.0,
    y_scint_center: float = -12.0,
    ipt_bounds_xy: tuple[float, float, float, float] = (-133.0, 133.0, -110.0, 86.0),
) -> torch.Tensor:
    """Cross-Magnet-0 4D trajectory interlock between IPT stubs (Planes 12..14) & Spectrometer candidates.

    Directly defeats the Stage 325 `TrkD2%+N1` single-noise-hit spectrometer vulnerability:
      - Real muons leaving the W target traverse IPT [-1735, -1675] mm before Magnet 0 [-1565.3, -65.3] mm,
        matching back-extrapolated position & slope within multiple-scattering tolerance;
      - Combinatorial spectrometer noise triplets fail IPT matching (`min_chi2 >> 25`) when inside IPT XY bounds!
    Returns per-spectrometer-candidate 4-dim interlock feature `(B, K_cand, 4)` in FP32:
      `[exp(-0.5 * min_chi2 / 9), log1p(min_chi2) / 5, in_ipt_aperture_flag, any_ipt_stub_flag]`.
    """
    with torch.amp.autocast(device_type=trk_cand.device.type, enabled=False):
        cand = trk_cand.float()
        ik = ipt_kin.float()
        b, kc, _ = cand.shape
        device = cand.device

        x_trk0 = float(x_pix_center) + 150.0 * cand[..., 2]
        y_trk0 = float(y_scint_center) + 150.0 * cand[..., 3]
        tx_s = 0.1 * cand[..., 4]
        ty_entry = 0.1 * cand[..., 5]
        is_bore_3st = (cand[..., 0] > 0.75) & trk_cand_mask
        sy_signed = SIGMA_SAGITTA_Y_MM * torch.sinh((5.0 * cand[..., 6]).clamp(-9.0, 9.0))
        kappa = torch.where(is_bore_3st, -sy_signed / SAGITTA_INTEGRAL_MM2, torch.zeros_like(sy_signed))

        # Reference IPT plane z = -1704.865 mm (upstream of Magnet0 [-1565.3, -65.3] mm)
        z_ref = -1704.865
        dz_s = z_ref - float(TRACKER_Z_MM[0])
        x_ext = x_trk0 + tx_s * dz_s
        y_ext = y_trk0 + ty_entry * dz_s - kappa * MAGNET0_LENGTH_MM * (z_ref - MAGNET0_Z_CENTER_MM)
        ty_ext = ty_entry - kappa * MAGNET0_LENGTH_MM

        xmin, xmax, ymin, ymax = ipt_bounds_xy
        in_ipt_ap = (x_ext >= xmin + 5.0) & (x_ext <= xmax - 5.0) & (y_ext >= ymin + 5.0) & (y_ext <= ymax - 5.0)

        # Propagate IPT stubs (B, M, 12) to z_ref
        dz_i = z_ref - ik[..., 2]
        xi = ik[..., 0] + ik[..., 3] * dz_i
        yi = ik[..., 1] + ik[..., 4] * dz_i
        txi = ik[..., 3]
        tyi = ik[..., 4]

        dx = x_ext[:, :, None] - xi[:, None, :]
        dy = y_ext[:, :, None] - yi[:, None, :]
        dtx = tx_s[:, :, None] - txi[:, None, :]
        dty = ty_ext[:, :, None] - tyi[:, None, :]

        # Tolerances across 1.75m Magnet0 air bore + tracker sigma_x=0.58mm: ~3.5 mm in x, ~2.5 mm in y, ~4 mrad slope
        chi2 = (dx / 3.5).square() + (dy / 2.5).square() + (dtx / 0.004).square() + (dty / 0.004).square()
        valid_pair = trk_cand_mask[:, :, None] & ipt_mask[:, None, :]
        chi2_masked = torch.where(valid_pair, chi2, torch.full_like(chi2, 400.0))
        min_chi2 = chi2_masked.min(dim=-1).values  # (B, K_cand)

        any_ipt = ipt_mask.any(dim=-1, keepdim=True).expand(b, kc).float()
        cm = trk_cand_mask.float()
        return torch.stack([
            torch.exp(-0.5 * min_chi2 / 9.0) * any_ipt * cm,
            torch.log1p(min_chi2) / 5.0 * cm,
            in_ipt_ap.float() * cm,
            any_ipt * cm,
        ], dim=-1)
