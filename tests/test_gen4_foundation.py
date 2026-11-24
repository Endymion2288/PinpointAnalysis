"""Unit tests for Pinpoint-Gen4 foundation: geometry_contract, physics_frontend, data_cache & seal_guard."""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from analysis.flavour.gen4.data_cache import (
    GEN4_TRUTH_BATCH_KEYS,
    RAW_MC_ROOT,
    _process_chunk_worker,
    assert_unsealed_mc8588_path,
    collate_gen4_batch,
    partition_mc8588_chunks,
)
from analysis.flavour.gen4.geometry_contract import (
    AIR_CONV_SUPPRESSION,
    IPT_X0_FLOOR,
    IPT_Z_REF_MM,
    MAGNET0_BEND_CONST_MEV_MM,
    S_FRONT,
    S_INTER,
    SIPM_NPIX,
    apply_cp_y_fid,
    apply_discrete_shift_fid,
    apply_inplace_nc_tau_mu_surgery,
    apply_px_sub_scint,
    apply_px_sub_si,
    check_symmetry_eligibility,
    inverse_sat_additive_pe,
    load_geometry_contract,
)
from analysis.flavour.gen4.physics_frontend import (
    PIX_FEAT_GEN4_DIM,
    PIXEL_MIP_KEV,
    UnifiedMuonKalman,
    dark_shield_filter_and_features,
    digitize_raw_scint_hits,
    extract_gen4_frontend_event,
    reconstruct_calorimeter_54_stations,
    unified_muon_kalman_event,
)
from analysis.flavour.nextgen.seal_guard import SealViolationError
from analysis.flavour.v3.v1_b1pp.spectrometer import TRACKER_Z_MM


class TestGen4Foundation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.geom = load_geometry_contract()

    def _make_clean_mip_obs(self, *, x0: float = 10.0, y0: float = -15.0, tx: float = 0.01, ty: float = -0.01, n_pix_per_hit: int = 2) -> dict:
        rows, cols, lays, edeps = [], [], [], []
        for l in range(self.geom.nz):
            dz = float(self.geom.pixel_Zpos[l] - self.geom.pixel_Zpos[0])
            x, y = x0 + tx * dz, y0 + ty * dz
            c0 = int(round((x - self.geom.px_min) / self.geom.dx_mm))
            r0 = int(round((y - self.geom.py_min) / self.geom.dy_mm))
            if 1 <= c0 < self.geom.nx - 1 and 1 <= r0 < self.geom.ny - 1:
                for k in range(n_pix_per_hit):
                    rows.append(r0)
                    cols.append(c0 + k)
                    lays.append(l)
                    edeps.append(0.00309 / n_pix_per_hit)
        r = np.asarray(rows, dtype=np.int64)
        c = np.asarray(cols, dtype=np.int64)
        l = np.asarray(lays, dtype=np.int64)
        cid = ((l * self.geom.ny + r) * self.geom.nx + c).astype(np.uint64)
        px = np.asarray(self.geom.raw_geometry['pixel_Xpos'], dtype=np.float64)
        py = np.asarray(self.geom.raw_geometry['pixel_Ypos'], dtype=np.float64)
        xyz = np.column_stack([px[c], py[r], self.geom.pixel_Zpos[l]]).astype(np.float32)

        # Active Scint bars along the same track (including <2.5 PE halo bar)
        n_sc = len(self.geom.raw_geometry['scint_Zpos'])
        p_is_h = self.geom.bounds_dict['panel_is_horiz']
        sz = np.asarray(self.geom.raw_geometry['scint_Zpos'], dtype=np.float64)
        pans, hors, bars, lays_b, npe_b, ht_b, bnds = [], [], [], [], [], [], []
        for pid in range(0, n_sc, 2):
            dz = float(sz[pid] - self.geom.pixel_Zpos[0])
            xb, yb = x0 + tx * dz, y0 + ty * dz
            ih = int(p_is_h[pid])
            bi = int(np.argmin(np.abs((self.geom.bar_y_centers_mm if ih else self.geom.bar_x_centers_mm) - (yb if ih else xb))))
            d_sipm = float(self.geom.optical_distance_to_sipm(ih, xb, yb))
            pe = 2 if pid == 0 else int(round(360.0 * math.exp(-d_sipm / 3800.0)))
            pans.append(pid)
            hors.append(ih)
            bars.append(bi)
            lays_b.append(int(self.geom.panel_to_station[pid]))
            npe_b.append(pe)
            ht_b.append(-1.0 if pe < 3 else 2.5 + d_sipm / 189.742)
            xc = self.geom.x_scint_center if ih else float(self.geom.bar_x_centers_mm[bi])
            yc = float(self.geom.bar_y_centers_mm[bi]) if ih else self.geom.y_scint_center
            hx = 210.0 if ih else 5.0
            hy = 5.0 if ih else 210.0
            bnds.append([xc - hx, xc + hx, yc - hy, yc + hy, float(sz[pid])])

        # Add one horizontal Y bar at station 6 (panel 7) for CP_y test
        pid_h = int(np.flatnonzero(p_is_h == 1)[0])
        bi_h = int(np.argmin(np.abs(self.geom.bar_y_centers_mm - y0)))
        pans.append(pid_h)
        hors.append(1)
        bars.append(bi_h)
        lays_b.append(int(self.geom.panel_to_station[pid_h]))
        npe_b.append(250)
        ht_b.append(3.2)
        yc = float(self.geom.bar_y_centers_mm[bi_h])
        bnds.append([self.geom.x_scint_center - 210.0, self.geom.x_scint_center + 210.0, yc - 5.0, yc + 5.0, float(sz[pid_h])])

        bar_dict = {
            'layer': np.asarray(lays_b, dtype=np.int16),
            'panel': np.asarray(pans, dtype=np.int16),
            'horizontal': np.asarray(hors, dtype=np.int8),
            'bar': np.asarray(bars, dtype=np.int16),
            'pe_proxy': np.asarray(npe_b, dtype=np.int64),
            'hit_time_ns': np.asarray(ht_b, dtype=np.float32),
            'bounds_mm': np.asarray(bnds, dtype=np.float32),
        }
        # Set canonical channel_id
        nb = self.geom.n_bars
        bar_dict['channel_id'] = (
            (bar_dict['layer'].astype(np.int64) * n_sc + bar_dict['panel'].astype(np.int64)) * (2 * nb)
            + bar_dict['horizontal'].astype(np.int64) * nb
            + bar_dict['bar'].astype(np.int64)
        ).astype(np.uint64)

        return {
            'pixel': {'channel_id': cid, 'xyz_mm': xyz, 'edep_mev': np.asarray(edeps, dtype=np.float32)},
            'bar': bar_dict,
            'tracker_hypothesis': {
                'station': np.array([0, 1, 2], dtype=np.int8),
                'xy_mm': np.array([[x0 + tx * 3000, y0 + ty * 3000], [x0 + tx * 4190, y0 + ty * 4190], [x0 + tx * 5380, y0 + ty * 5380]], dtype=np.float32),
                'sigma_xy_mm': np.tile([0.58, 0.017], (3, 1)).astype(np.float32),
            },
        }

    def test_geometry_contract_mc8588_and_54_stations(self) -> None:
        g = self.geom
        self.assertTrue(g.is_mc8588_block_order)
        self.assertEqual(tuple(g.s_front_xonly.tolist()), S_FRONT)
        self.assertEqual(tuple(g.s_inter_xonly.tolist()), S_INTER)
        self.assertEqual(g.n_stations, 54)
        self.assertEqual(g.n_doublets, 42)
        self.assertEqual(g.fortune_station_mask.shape, (7, 54))
        self.assertTrue(np.all(g.fortune_station_mask.sum(axis=1) == 6))
        self.assertEqual(g.pinpoint_sand_scint_station.tolist(), [-1, 1, 2, 3, 4, 5, 12, 19, 26, 33, 40, 47])
        # Zero air-gap false conversion & monotonic X0(z)
        self.assertTrue(g.air_gap_6mm_mask(g.pixel_Zpos[0] - 3.0))
        self.assertFalse(g.air_gap_6mm_mask(g.pixel_Zpos[0] + 2.0))
        self.assertAlmostEqual(float(g.material_conversion_prior(g.pixel_Zpos[0] - 3.0)), AIR_CONV_SUPPRESSION, places=9)
        self.assertGreater(float(np.diff(g.x0_knots_val).min()), 0.0)
        z_t = torch.tensor([-2990.0, -2500.0, -1700.0], dtype=torch.float32)
        x0_t = g.interp_x0_torch(z_t)
        self.assertEqual(x0_t.shape, (3,))
        self.assertGreater(IPT_X0_FLOOR, 0.0)
        self.assertLess(IPT_X0_FLOOR, 1e-3)

    def test_symmetry_bijective_involutions(self) -> None:
        g = self.geom
        obs = self._make_clean_mip_obs(x0=-5.0, y0=-14.0, tx=0.002, ty=-0.001)
        elig = check_symmetry_eligibility(obs, g)
        self.assertTrue(elig['px_sub_si'])
        self.assertTrue(elig['cp_y_fid'])
        self.assertIn((1, -1), elig['valid_shifts'])

        # 1) P_{x,sub}^{Si} involution
        ox1, l1 = apply_px_sub_si(obs, g, label_7=1, continuous_pe=True)
        self.assertEqual(check_symmetry_eligibility(ox1, g)['px_sub_si'], elig['px_sub_si'])
        ox2, l2 = apply_px_sub_si(ox1, g, label_7=l1, continuous_pe=True)
        self.assertEqual(l2, 1)
        self.assertTrue(np.array_equal(ox2['pixel']['channel_id'], obs['pixel']['channel_id']))
        self.assertTrue(np.allclose(ox2['pixel']['edep_mev'], obs['pixel']['edep_mev']))
        self.assertTrue(np.allclose(ox2['bar']['_pe_lin_float'], -SIPM_NPIX * np.log(1.0 - np.clip(obs['bar']['pe_proxy'] / SIPM_NPIX, 0, 0.995)), atol=1e-4))

        # 2) CP_y^{fid} involution + label_7 swap 1 <-> 2
        oy1, ly1 = apply_cp_y_fid(obs, g, label_7=1, continuous_pe=True)
        self.assertEqual(ly1, 2)
        oy2, ly2 = apply_cp_y_fid(oy1, g, label_7=ly1, continuous_pe=True)
        self.assertEqual(ly2, 1)
        self.assertTrue(np.array_equal(oy2['pixel']['channel_id'], obs['pixel']['channel_id']))
        self.assertTrue(np.allclose(oy2['tracker_hypothesis']['xy_mm'], obs['tracker_hypothesis']['xy_mm'], atol=1e-4))

        # 3) T_{kx,ky} inverse
        os1 = apply_discrete_shift_fid(obs, g, 1, -1, continuous_pe=True)
        os2 = apply_discrete_shift_fid(os1, g, -1, 1, continuous_pe=True)
        self.assertTrue(np.array_equal(os2['pixel']['channel_id'], obs['pixel']['channel_id']))

        # 4) Scint-only P_{x,sub}^{Scint}
        scint_only = {'pixel': {'channel_id': np.zeros(0, dtype=np.uint64), 'xyz_mm': np.zeros((0, 3), dtype=np.float32)},
                      'bar': obs['bar'], 'tracker_hypothesis': {'station': np.zeros(0, dtype=np.int8), 'xy_mm': np.zeros((0, 2), dtype=np.float32)}}
        self.assertTrue(check_symmetry_eligibility(scint_only, g)['px_sub_scint'])
        sx1, _ = apply_px_sub_scint(scint_only, g, continuous_pe=True)
        sx2, _ = apply_px_sub_scint(sx1, g, continuous_pe=True)
        self.assertTrue(np.array_equal(sx2['bar']['bar'], scint_only['bar']['bar']))

    def test_dark_shield_and_36dim_edep_features(self) -> None:
        g = self.geom
        obs_sig = self._make_clean_mip_obs(x0=0.0, y0=-12.0, n_pix_per_hit=2)
        n_sig = len(obs_sig['pixel']['channel_id'])

        # Inject 1e-5 dark rate (~1100 fake hits/plane = 16,500 fake hits)
        rng = np.random.default_rng(329)
        dr = rng.integers(0, g.ny, size=15 * 1100, dtype=np.int64)
        dc = rng.integers(0, g.nx, size=15 * 1100, dtype=np.int64)
        dl = np.repeat(np.arange(15, dtype=np.int64), 1100)
        d_cid = ((dl * g.ny + dr) * g.nx + dc).astype(np.uint64)
        px = np.asarray(g.raw_geometry['pixel_Xpos'], dtype=np.float64)
        py = np.asarray(g.raw_geometry['pixel_Ypos'], dtype=np.float64)
        d_xyz = np.column_stack([px[dc], py[dr], g.pixel_Zpos[dl]]).astype(np.float32)
        d_ed = np.full(len(dr), 0.00036, dtype=np.float32)

        obs_noisy = {
            'pixel': {
                'channel_id': np.concatenate([obs_sig['pixel']['channel_id'], d_cid]),
                'xyz_mm': np.vstack([obs_sig['pixel']['xyz_mm'], d_xyz]),
                'edep_mev': np.concatenate([obs_sig['pixel']['edep_mev'], d_ed]),
            },
            'bar': obs_sig['bar'],
            'tracker_hypothesis': obs_sig['tracker_hypothesis'],
        }
        ds = dark_shield_filter_and_features(obs_noisy, g, enable_dark_filter='auto')
        self.assertTrue(ds['dark_noise_detected'])
        keep = ds['raw_keep_mask_hits']
        sig_ret = float(keep[:n_sig].mean())
        dark_rej = 1.0 - float(keep[n_sig:].mean())
        self.assertGreater(sig_ret, 0.995)
        self.assertGreater(dark_rej, 0.998)
        self.assertEqual(ds['pix_feat'].shape[1], PIX_FEAT_GEN4_DIM)
        self.assertEqual(PIX_FEAT_GEN4_DIM, 36)

        # Clean mode preserves 100% of single-pixel hits
        obs_1pix = self._make_clean_mip_obs(x0=5.0, y0=-10.0, n_pix_per_hit=1)
        ds_clean = dark_shield_filter_and_features(obs_1pix, g, enable_dark_filter='auto')
        self.assertFalse(ds_clean['dark_noise_detected'])
        self.assertEqual(float(ds_clean['raw_keep_mask_hits'].mean()), 1.0)

        # Double-MIP conversion feature [31] cleanly separates 2*MIP (6.18 keV) from 1*MIP (3.09 keV)
        obs_2mip = self._make_clean_mip_obs(x0=5.0, y0=-10.0, n_pix_per_hit=1)
        obs_2mip['pixel']['edep_mev'] = obs_2mip['pixel']['edep_mev'] * 2.2
        ds_2mip = dark_shield_filter_and_features(obs_2mip, g, enable_dark_filter='auto')
        self.assertLess(float(ds_clean['pix_feat'][:, 31].mean()), 0.2)
        self.assertGreater(float(ds_2mip['pix_feat'][:, 31].mean()), 0.75)

    def test_scint_saturation_and_unified_muon_kalman_m2(self) -> None:
        g = self.geom
        # Scint saturation & <2.5 PE halo
        bar = digitize_raw_scint_hits(
            np.array([0, 0, 1]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([10, 10, 12]),
            np.array([40000.0, 20000.0, 2.0]), np.array([2.5, 2.9, 3.5]), g,
        )
        pe_dict = {int(b): int(p) for b, p in zip(bar['bar'], bar['pe_proxy'])}
        ht_dict = {int(b): float(t) for b, t in zip(bar['bar'], bar['hit_time_ns'])}
        self.assertLessEqual(pe_dict[10], 14400)
        self.assertGreater(pe_dict[10], 14000)
        self.assertEqual(pe_dict[12], 2)
        self.assertEqual(ht_dict[12], -1.0)

        # M2 soft muon deflection accuracy & sign in Magnet0 (B_x = +0.57 T)
        for q_true, p_true_gev in [(-1, 18.0), (1, 45.0)]:
            x_ipt, y_ipt, tx, ty_ipt = 4.0, -10.0, 0.003, -0.002
            dz = TRACKER_Z_MM[0] - IPT_Z_REF_MM
            dy_m0 = float(q_true) * (MAGNET0_BEND_CONST_MEV_MM / 1000.0) / p_true_gev
            y_t1 = y_ipt + ty_ipt * dz + dy_m0
            x_t1 = x_ipt + tx * dz
            # Build 3 IPT pixel hits on layers 12, 13, 14
            rows, cols, lays, edeps = [], [], [], []
            for l in (12, 13, 14):
                dz_l = float(g.pixel_Zpos[l] - IPT_Z_REF_MM)
                xl, yl = x_ipt + tx * dz_l, y_ipt + ty_ipt * dz_l
                cols.append(int(round((xl - g.px_min) / g.dx_mm)))
                rows.append(int(round((yl - g.py_min) / g.dy_mm)))
                lays.append(l)
                edeps.append(0.0031)
            r, c, l = np.asarray(rows), np.asarray(cols), np.asarray(lays)
            px = np.asarray(g.raw_geometry['pixel_Xpos'], dtype=np.float64)
            py = np.asarray(g.raw_geometry['pixel_Ypos'], dtype=np.float64)
            obs_m2 = {
                'pixel': {
                    'channel_id': ((l * g.ny + r) * g.nx + c).astype(np.uint64),
                    'xyz_mm': np.column_stack([px[c], py[r], g.pixel_Zpos[l]]).astype(np.float32),
                    'edep_mev': np.asarray(edeps, dtype=np.float32),
                },
                'bar': {'layer': np.zeros(0, np.int16), 'panel': np.zeros(0, np.int16), 'horizontal': np.zeros(0, np.int8),
                        'bar': np.zeros(0, np.int16), 'pe_proxy': np.zeros(0, np.int64), 'hit_time_ns': np.zeros(0, np.float32),
                        'bounds_mm': np.zeros((0, 5), np.float32), 'channel_id': np.zeros(0, np.uint64)},
                'tracker_hypothesis': {
                    'station': np.array([0], dtype=np.int8),
                    'xy_mm': np.array([[x_t1, y_t1]], dtype=np.float32),
                    'sigma_xy_mm': np.array([[0.58, 0.017]], dtype=np.float32),
                },
            }
            fe = extract_gen4_frontend_event(obs_m2, g)
            self.assertEqual(fe['muon_mode'], 2)
            m_trks = fe['muon_kalman_tracks']
            self.assertAlmostEqual(float(m_trks[0, 4]), float(q_true), places=3)
            tol_rel = 0.05 if p_true_gev < 25.0 else 0.10
            self.assertLess(abs(float(m_trks[0, 2]) - p_true_gev) / p_true_gev, tol_rel)

        # Differentiable M2 module check
        mod = UnifiedMuonKalman(g)
        y_i = torch.tensor([-10.0], requires_grad=True)
        ty_i = torch.tensor([0.0])
        z_i = torch.tensor([IPT_Z_REF_MM])
        y_t = torch.tensor([-22.28])
        out = mod.forward_m2_batch(y_i, ty_i, z_i, y_t, torch.tensor([0.01]), torch.tensor([1e-4]))
        out['p_gev'].sum().backward()
        self.assertTrue(torch.isfinite(y_i.grad).all())

    def test_surgery_collate_and_seal_guard(self) -> None:
        g = self.geom
        # Inverse saturation additive algebra
        pe_a = 14000.0
        pe_b = 350.0
        n_surg = inverse_sat_additive_pe(pe_a, pe_b, SIPM_NPIX)
        n_lin_exact = -SIPM_NPIX * math.log(1.0 - pe_a / SIPM_NPIX) - SIPM_NPIX * math.log(1.0 - pe_b / SIPM_NPIX)
        self.assertAlmostEqual(n_surg, SIPM_NPIX * (1.0 - math.exp(-n_lin_exact / SIPM_NPIX)), places=4)

        host = self._make_clean_mip_obs(x0=2.0, y0=-10.0, n_pix_per_hit=1)
        rng = np.random.default_rng(77)
        surg_obs, meta = apply_inplace_nc_tau_mu_surgery(
            host, np.array([2.0, -10.0, -2950.0]), np.array([0.5, -0.3]), g, rng, tau_energy_gev=120.0, decay_length_mm=8.0, q_mu=-1
        )
        self.assertEqual(meta['label_7'], 4)
        fe = extract_gen4_frontend_event(surg_obs, g)
        fe['label'] = 2
        fe['label_7'] = 4
        fe['vertex_xyz_mm'] = meta['vertex_xyz_mm']
        fe['vis_kinematics'] = np.zeros(4, dtype=np.float32)
        fe['true_sv_xyz_mm'] = meta['true_sv_xyz_mm']
        fe['true_branch_p4_gev'] = np.zeros((4, 4), dtype=np.float32)

        batch = collate_gen4_batch([fe, fe], g, aug_cfg={'drop_edep': False})
        self.assertEqual(batch['pix_feat'].shape[-1], 36)
        self.assertEqual(batch['scint_grid'].shape[1:], (54, 2, 40, 10))
        self.assertEqual(float(batch['has_edep'][0]), 1.0)

        batch_no_edep = collate_gen4_batch([fe, fe], g, aug_cfg={'drop_edep': True})
        self.assertEqual(float(batch_no_edep['has_edep'][0]), 0.0)
        self.assertEqual(float(batch_no_edep['pix_feat'][..., 30:36].abs().max()), 0.0)

        # Seal guard checks on chunk_id % 20 >= 16 & deprecated MCs
        for bad in (
            '/eos/project-f/fasersim-bonn/public/pinpoint/data/10085/10085_016.root',
            '/eos/project-f/fasersim-bonn/public/pinpoint/data/10087/10087_019.root',
            '/eos/project-f/fasersim-bonn/public/pinpoint/data/10081/10081_000.root',
            '/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v6_clean_v3_2026-11-19/clean_sim/geometry.json',
        ):
            with self.assertRaises(SealViolationError):
                assert_unsealed_mc8588_path(bad)
        splits = partition_mc8588_chunks()
        for sp_dict in splits.values():
            for ds_list in sp_dict.values():
                for cid, _ in ds_list:
                    self.assertLess(cid % 20, 16)
        self.assertIn('true_tau_submode', GEN4_TRUTH_BATCH_KEYS)

    def test_chunk_gt0_evtid_truth_alignment(self) -> None:
        p = RAW_MC_ROOT / '10087' / '10087_012.root'
        items_tr = _process_chunk_worker(('10087', 12, str(p), list(range(15)), 'train', 'r9_split', 12345))
        self.assertGreaterEqual(len(items_tr), 10)
        self.assertTrue(all(it['tau_branch_valid'] == 1 for it in items_tr))
        self.assertTrue(all(it['true_tau_submode'] in (0, 1, 2, 3) for it in items_tr))
        self.assertTrue(any(it['charm_flag'] == 1 for it in items_tr))
        self.assertTrue(any(it['tau_decay_L3d_mm'] > 0.0 for it in items_tr))

        items_tu = _process_chunk_worker(('10087', 12, str(p), list(range(5)), 'tune_with_meta', 'screen', 12345))
        self.assertGreaterEqual(len(items_tu), 3)
        for it in items_tu:
            for k in GEN4_TRUTH_BATCH_KEYS:
                self.assertNotIn(k, it)
            self.assertEqual(it['_meta']['tau_branch_valid'], 1)
            self.assertIn(it['_meta']['true_tau_submode'], (0, 1, 2, 3))


if __name__ == '__main__':
    unittest.main()
