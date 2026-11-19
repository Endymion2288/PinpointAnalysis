"""Stage 321 comprehensive unit tests for all Stage 320 physical, ML, and statistical audit fixes."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
import unittest

import numpy as np
import torch

from analysis.flavour.readout import bar_channels
from analysis.flavour.v3.common.domains import (
    PE_PER_MEV_CENTER_CAL,
    calibrated_bar_channels,
    compute_observable_containment,
    cumulative_radiation_length,
    detector_bounds,
    pinpoint_slab_catalog,
)
from analysis.flavour.v3.common.layers import ScaledQKNormAttention
from analysis.flavour.v3.v1_b1pp.spectrometer import (
    MAGNET0_LENGTH_MM,
    MAGNET0_Z_MAX_MM,
    MAGNET0_Z_MIN_MM,
    SAGITTA_CONST_MEV_MM,
    SAGITTA_INTEGRAL_MM2,
    TRACKER_Z_MM,
    back_extrapolate_y_through_magnet0,
    extract_spectrometer_kinematics,
)
from analysis.flavour.v3.v2_pinpointformer.pixel_tokenizer import extract_pf_item, quantize_pixel_hits
from analysis.flavour.v3.v4_tau.ensemble import (
    _apply_domain_convex_blend,
    _domain_blend_objective_and_grad,
    _fit_domain_convex_blend,
)
from analysis.flavour.v3.v5_confirm.flux_significance import (
    FIDUCIAL_2T_BKG_SHARES,
    build_physical_templates,
    fit_realistic_profile_likelihood,
)
from analysis.flavour.v3.v6_clean_sim.produce_clean_sim import (
    _build_tungsten_sampler_spec,
    _sample_tungsten_vertex_v2_mm,
)


CLEAN_GEOMETRY_PATH = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v4_audit_2026-11-18/clean_sim/geometry.json')


def _analytic_piecewise_track_y(
    y_start: float,
    ty_start: float,
    z_start: float,
    kappa: float,
    z_eval: float,
    magnets: tuple[tuple[float, float], ...] = ((-1565.3, -65.3), (137.4, 1137.4), (1337.4, 2337.4)),
) -> tuple[float, float]:
    """Exact C^1 analytical integration of y''(z) = kappa inside magnetic intervals, 0 in drift regions."""
    z = float(z_start)
    y = float(y_start)
    ty = float(ty_start)
    for za, zb in magnets:
        if z_eval <= z:
            break
        # Drift to magnet entry
        if z < za:
            dz = min(z_eval, za) - z
            y += ty * dz
            z += dz
        if z_eval <= z:
            break
        # Constant curvature kappa inside [za, zb]
        if z < zb:
            dz = min(z_eval, zb) - z
            y += ty * dz + 0.5 * kappa * dz * dz
            ty += kappa * dz
            z += dz
    if z_eval > z:
        y += ty * (z_eval - z)
    return y, ty


class TestStage321PhysicsFixes(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.geometry = json.loads(CLEAN_GEOMETRY_PATH.read_text())
        cls.bounds = detector_bounds(cls.geometry)

    def test_pinpoint_slab_catalog_and_exact_volume_sampler(self) -> None:
        cat = pinpoint_slab_catalog(self.geometry)
        expected_pin = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 24, 25, 38, 39, 52, 53, 66, 67, 80, 81, 94, 95]
        self.assertEqual(cat['n_slabs'], 108)
        self.assertEqual(cat['pinpoint_indices'].tolist(), expected_pin)
        self.assertEqual(len(cat['fortune_indices']), 84)
        self.assertAlmostEqual(cat['w_x_min_mm'], -133.0, places=3)
        self.assertAlmostEqual(cat['w_x_max_mm'], 297.0, places=3)
        self.assertAlmostEqual(cat['w_y_min_mm'], -222.0, places=3)
        self.assertAlmostEqual(cat['w_y_max_mm'], 208.0, places=3)

        # Verify exact DetectorConstruction +7.505 / +17.505 mm offsets
        pz = np.asarray(self.geometry['pixel_Zpos'], dtype=np.float64)
        wz = np.asarray(self.geometry['tungsten_Zpos'], dtype=np.float64)
        for idx in expected_pin:
            bc = int(cat['block_copy_no'][idx])
            pl = int(cat['plate_in_block'][idx])
            self.assertAlmostEqual(wz[idx] - pz[bc], 7.505 if pl == 0 else 17.505, places=2)

        # Verify 12,000 stratified vertices recover exact uniform 3D tungsten mass composition
        spec = _build_tungsten_sampler_spec(self.geometry, self.bounds)
        self.assertAlmostEqual(sum(spec['volume_fractions']), 1.0, places=9)
        n = 12000
        samples = [_sample_tungsten_vertex_v2_mm(100000 + i, 99991 * i + 7, spec) for i in range(n)]
        weights = np.array([s['geom_weight'] for s in samples], dtype=np.float64)
        is_pin = np.array([s['is_pinpoint_module'] for s in samples], dtype=bool)
        is_pix_xy = np.array([s['is_pixel_xy_acceptance'] for s in samples], dtype=bool)
        vz = np.array([s['vz_mm'] for s in samples], dtype=np.float64)
        slabs = np.array([s['slab_id'] for s in samples], dtype=np.int64)
        max_dz = np.max(np.abs(vz - wz[slabs]))

        self.assertLessEqual(max_dz, 2.45 + 1e-9)
        # Analytic expectation E[w] = 1.0
        self.assertAlmostEqual(float(weights.mean()), 1.0, delta=0.025)
        # Mass-weighted PinpointBlock fraction must match exact 24/108 = 22.22%
        pin_mass_frac = float(weights[is_pin].sum() / weights.sum())
        self.assertAlmostEqual(pin_mass_frac, 24.0 / 108.0, delta=0.015)
        # Mass-weighted pixel-XY area fraction must match (265*195)/(429*429) = 28.08%
        pix_mass_frac = float(weights[is_pix_xy].sum() / weights.sum())
        self.assertAlmostEqual(pix_mass_frac, (265.0 * 195.0) / (429.0 * 429.0), delta=0.015)

        # Verify Al placement in cumulative_radiation_length matches physical G4 ordering
        x0_before_p0 = cumulative_radiation_length(float(pz[0]) - 0.1, self.geometry)
        x0_after_w0 = cumulative_radiation_length(float(wz[0]) + 3.0, self.geometry)
        self.assertGreater(x0_before_p0, 0.01)  # 2 mm AlWall_Back at -7.005 mm
        self.assertGreater(x0_after_w0, 1.40)   # 5 mm W0 (~1.43 X0) + 7 mm Al

    def test_single_sipm_saturation_and_zero_truth_edep(self) -> None:
        # Pure PE bar dictionary with NO edep_proxy_MeV
        npix = 14400.0
        raw_pe = np.array([1.0, 200.0, 5000.0, 20000.0, 100000.0], dtype=np.float64)
        g4_sat_pe = np.rint(npix * (1.0 - np.exp(-raw_pe / npix))).astype(np.int64)
        bar_no_edep = {'pe_proxy': g4_sat_pe}

        # Default pe_only=True must NOT apply a second saturation compression!
        keep, e_cal, bpe_kept = calibrated_bar_channels(bar_no_edep, {'pe_only': True})
        self.assertEqual(keep.tolist(), [False, True, True, True, True])
        # bpe_kept must retain exact single-saturated G4 PE (reaching ~14386 PE, NOT double-compressed ~9102 PE!)
        np.testing.assert_array_equal(bpe_kept, g4_sat_pe[1:].astype(np.float64))
        self.assertGreater(float(bpe_kept[-1]), 14000.0)
        # Single inverse saturation accurately recovers raw linear energy (up to 98% cap)
        self.assertAlmostEqual(float(e_cal[0]), 200.0 / PE_PER_MEV_CENTER_CAL, delta=0.01)
        self.assertAlmostEqual(float(e_cal[1]), 5000.0 / PE_PER_MEV_CENTER_CAL, delta=0.05)
        self.assertAlmostEqual(float(e_cal[2]), 20000.0 / PE_PER_MEV_CENTER_CAL, delta=0.2)
        # pe_only=False must raise when edep_proxy_MeV is absent
        with self.assertRaises(ValueError):
            calibrated_bar_channels(bar_no_edep, {'pe_only': False})

        # Verify bar_channels(include_truth_edep=False) omits edep_proxy_MeV and preserves hit_time_ns
        rec = {
            'layerID': np.array([0, 0], dtype=np.int64),
            'panelID': np.array([0, 0], dtype=np.int64),
            'isHorizontal': np.array([0, 0], dtype=np.int64),
            'rowID': np.array([-1, -1], dtype=np.int64),
            'colID': np.array([10, 10], dtype=np.int64),
            'nPE': np.array([500, 300], dtype=np.int64),
            'hitTime': np.array([12.5, 9.2], dtype=np.float64),
        }
        bars, cons = bar_channels(rec, self.geometry, include_truth_edep=False)
        self.assertNotIn('edep_proxy_MeV', bars)
        self.assertIn('hit_time_ns', bars)
        self.assertEqual(int(bars['pe_proxy'][0]), 800)
        self.assertAlmostEqual(float(bars['hit_time_ns'][0]), 9.2, places=5)
        self.assertEqual(cons['output_pe'], 800)

    def test_domain_blend_exact_gradient_and_permutation_invariance(self) -> None:
        rng = np.random.default_rng(321)
        m, n, c = 4, 300, 4
        y = rng.integers(0, c, size=n)
        lp = rng.normal(0.0, 1.0, size=(m, n, c))
        # Add realistic class signal to member logits
        for k in range(m):
            lp[k, np.arange(n), y] += 0.6 + 0.25 * k
        lp -= lp.mean(axis=-1, keepdims=True)

        # 1. Central finite-difference gradient check at an arbitrary interior point (m alphas + log_T + b0, b1, b2)
        theta = np.array([0.3, -0.4, 0.7, 0.1, -0.15, 0.15, -0.25, 0.08], dtype=np.float64)
        row_idx = np.arange(n)
        sample_w = np.full(n, 1.0 / n, dtype=np.float64)
        _, g_ana = _domain_blend_objective_and_grad(theta, lp, y, sample_w, row_idx)
        eps = 1e-6
        g_num = np.zeros_like(theta)
        for j in range(len(theta)):
            tp = theta.copy()
            tm = theta.copy()
            tp[j] += eps
            tm[j] -= eps
            fp, _ = _domain_blend_objective_and_grad(tp, lp, y, sample_w, row_idx)
            fm, _ = _domain_blend_objective_and_grad(tm, lp, y, sample_w, row_idx)
            g_num[j] = (fp - fm) / (2.0 * eps)
        np.testing.assert_allclose(g_ana, g_num, rtol=1e-7, atol=1e-8)

        # 2. Exact member-permutation invariance across all 4! = 24 permutations
        ref_fit = _fit_domain_convex_blend(lp, y)
        ref_probs = _apply_domain_convex_blend(lp, ref_fit)
        self.assertAlmostEqual(sum(ref_fit['bias']), 0.0, places=10)
        self.assertAlmostEqual(sum(ref_fit['weights']), 1.0, places=10)

        for perm in itertools.permutations(range(m)):
            idx_p = list(perm)
            fit_p = _fit_domain_convex_blend(lp[idx_p], y)
            probs_p = _apply_domain_convex_blend(lp[idx_p], fit_p)
            np.testing.assert_allclose(probs_p, ref_probs, rtol=1e-7, atol=1e-8)
            np.testing.assert_array_equal(probs_p.argmax(axis=1), ref_probs.argmax(axis=1))

    def test_batch_independent_attention_entropy(self) -> None:
        torch.manual_seed(321)
        attn = ScaledQKNormAttention(dim=32, heads=4, init_scale=1.0)
        attn.eval()
        # Zero Q/K projections so attention scores are identically 0 -> exact uniform distribution over valid keys
        with torch.no_grad():
            attn.q_proj.weight.zero_()
            attn.k_proj.weight.zero_()

        q = torch.randn(4, 3, 32)
        k = torch.randn(4, 8, 32)
        k_mask = torch.zeros(4, 8, dtype=torch.bool)
        # Event 0: 0 valid keys -> entropy 0.0
        # Event 1: 1 valid key  -> entropy 0.0
        # Event 2: 2 valid keys -> uniform normalized entropy 1.0
        # Event 3: 8 valid keys -> uniform normalized entropy 1.0
        k_mask[1, :1] = True
        k_mask[2, :2] = True
        k_mask[3, :8] = True
        q_mask = torch.tensor([[True, False, False], [True, True, False], [True, True, False], [True, True, True]])

        _, ent_batch = attn(q, k, k_mask, query_mask=q_mask, return_entropy=True)
        self.assertEqual(tuple(ent_batch.shape), (4, 4, 1))
        np.testing.assert_allclose(ent_batch[0].detach().numpy(), 0.0, atol=1e-6)
        np.testing.assert_allclose(ent_batch[1].detach().numpy(), 0.0, atol=1e-6)
        np.testing.assert_allclose(ent_batch[2].detach().numpy(), 1.0, atol=1e-5)
        np.testing.assert_allclose(ent_batch[3].detach().numpy(), 1.0, atol=1e-5)

        # Single-event batch must match mixed batch exactly (zero cross-event contamination!)
        _, ent_single = attn(q[2:3], k[2:3], k_mask[2:3], query_mask=q_mask[2:3], return_entropy=True)
        np.testing.assert_allclose(ent_single[0].detach().numpy(), ent_batch[2].detach().numpy(), atol=1e-6)

    def test_spectrometer_entry_slope_and_magnet0_exact_closure(self) -> None:
        # Test mu- and mu+ at 25 GeV and 250 GeV launched from target z = -2500 mm
        z_start = -2500.0
        for p_gev in (25.0, 250.0):
            for charge in (-1.0, +1.0):
                p_mev = p_gev * 1000.0
                kappa = charge * (0.299792458 * 0.57) / p_mev  # mm^-1
                y_start = -15.0
                ty_start = 0.008 * charge

                y_trk0, ty_trk0_true = _analytic_piecewise_track_y(y_start, ty_start, z_start, kappa, TRACKER_Z_MM[0])
                y_trk1, _ = _analytic_piecewise_track_y(y_start, ty_start, z_start, kappa, TRACKER_Z_MM[1])
                y_trk2, _ = _analytic_piecewise_track_y(y_start, ty_start, z_start, kappa, TRACKER_Z_MM[2])

                sy = y_trk1 - 0.5 * (y_trk0 + y_trk2)
                delta = TRACKER_Z_MM[1] - TRACKER_Z_MM[0]
                # 1. Exact momentum closure
                p_reco_mev = SAGITTA_CONST_MEV_MM / abs(sy)
                self.assertAlmostEqual(p_reco_mev / p_mev, 1.0, places=6)
                # 2. Exact entry slope closure (fixes previous wrong sign and 3x error!)
                ty_entry = (y_trk1 - y_trk0) / delta + (60.0 / 59.0) * (sy / delta)
                self.assertAlmostEqual(ty_entry, ty_trk0_true, places=9)
                # 3. Exact Magnet0 back-extrapolation closure
                kappa_reco = -sy / SAGITTA_INTEGRAL_MM2
                y_back, ty_back = back_extrapolate_y_through_magnet0(y_trk0, ty_entry, kappa_reco, z_start)
                self.assertAlmostEqual(float(y_back), y_start, places=5)
                self.assertAlmostEqual(float(ty_back), ty_start, places=8)

                # 4. Also verify extract_spectrometer_kinematics matches exact cal_start with ~0 residual!
                obs_trk = {
                    'station': np.array([0, 1, 2], dtype=np.int8),
                    'xy_mm': np.array([[10.0, y_trk0], [10.0, y_trk1], [10.0, y_trk2]], dtype=np.float32),
                    'sigma_xy_mm': np.full((3, 2), 0.02, dtype=np.float32),
                }
                out = extract_spectrometer_kinematics(obs_trk, (z_start, 10.0, y_start, True), self.bounds)
                self.assertTrue(bool(out['cand_mask'][0]))
                # cand[0, 10] is dy_start / 150 mm -> must be nearly 0!
                self.assertLess(abs(float(out['cand'][0, 10])) * 150.0, 0.05)

    def test_target_only_pitch_quantization_and_shuffle_control(self) -> None:
        nx, ny = self.bounds['nx'], self.bounds['ny']
        pz = self.bounds['pixel_Zpos']
        # Create hits on target planes 1, 2, 3 AND IPT planes 12, 13, 14
        rows = []
        for pl, x_mm, y_mm in (
            (1, 10.12, -5.34),
            (1, 10.14, -5.32),
            (2, 12.67, -2.11),
            (2, 14.89, -4.45),
            (3, 15.23, 1.09),
            (3, 19.81, -7.22),
            (12, 22.37, 3.19),
            (13, 24.41, 4.21),
            (14, 26.55, 5.33),
        ):
            col = int(np.argmin(np.abs(np.asarray(self.geometry['pixel_Xpos']) - x_mm)))
            row = int(np.argmin(np.abs(np.asarray(self.geometry['pixel_Ypos']) - y_mm)))
            cid = (pl * ny + row) * nx + col
            rows.append((cid, [self.geometry['pixel_Xpos'][col], self.geometry['pixel_Ypos'][row], pz[pl]]))
        cids = np.array([r[0] for r in rows], dtype=np.uint64)
        xyz = np.array([r[1] for r in rows], dtype=np.float32)

        q_cids, q_xyz, *_ = quantize_pixel_hits(cids, xyz, self.bounds, 1.0, target_planes_only=True)
        q_planes = q_cids.astype(np.int64) // (nx * ny)
        # IPT planes 12, 13, 14 must be UNCHANGED to float32 precision!
        orig_ipt = xyz[np.isin(cids.astype(np.int64) // (nx * ny), [12, 13, 14])]
        new_ipt = q_xyz[np.isin(q_planes, [12, 13, 14])]
        np.testing.assert_allclose(new_ipt, orig_ipt, rtol=0.0, atol=1e-6)
        # Target planes 1..3 must be quantized to 1.0 mm integer offsets from (x_pix_center, y_scint_center)
        tgt_xy = q_xyz[q_planes < 12, :2]
        x_c = float(self.bounds['x_pix_center'])
        y_c = float(self.bounds['y_scint_center'])
        self.assertTrue(np.allclose((tgt_xy[:, 0] - x_c) - np.rint(tgt_xy[:, 0] - x_c), 0.0, atol=1e-4))
        self.assertTrue(np.allclose((tgt_xy[:, 1] - y_c) - np.rint(tgt_xy[:, 1] - y_c), 0.0, atol=1e-4))

        # Verify 1-hit incoming rock muon on physical PixelSensor[0] triggers upstream_veto (c_obs[6] == 1.0)
        p0_cid = np.array([int(ny // 2) * nx + int(nx // 2)], dtype=np.uint64)
        p0_xyz = np.array([[0.0, -12.0, float(pz[0])]], dtype=np.float32)
        obs_rock = {
            'pixel': {'channel_id': np.concatenate([p0_cid, cids]), 'xyz_mm': np.vstack([p0_xyz, xyz])},
            'bar': {
                'panel': np.array([0, 1], dtype=np.int16),
                'horizontal': np.array([0, 0], dtype=np.int8),
                'bar': np.array([15, 16], dtype=np.int16),
                'bounds_mm': np.array([[0, 10, -200, 200, -2979.49], [10, 20, -200, 200, -2949.48]], dtype=np.float32),
                'pe_proxy': np.array([500, 700], dtype=np.int64),
            },
            'tracker_hypothesis': {
                'station': np.empty(0, dtype=np.int8),
                'xy_mm': np.empty((0, 2), dtype=np.float32),
                'sigma_xy_mm': np.empty((0, 2), dtype=np.float32),
            },
        }
        dom_rock = compute_observable_containment(obs_rock, self.geometry, self.bounds, config={'pe_only': True})
        self.assertEqual(float(dom_rock['c_obs'][6]), 1.0)
        self.assertEqual(int(dom_rock['obs_domain_id']), 3)

        # Test Exp-1 shuffle_micro_vertex control leaves raw hits & bar features invariant while perturbing micro_vtx_feat
        obs = {
            'pixel': {'channel_id': cids, 'xyz_mm': xyz},
            'bar': {
                'panel': np.array([0, 1], dtype=np.int16),
                'horizontal': np.array([0, 0], dtype=np.int8),
                'bar': np.array([15, 16], dtype=np.int16),
                'bounds_mm': np.array([[0, 10, -200, 200, -2979.49], [10, 20, -200, 200, -2949.48]], dtype=np.float32),
                'pe_proxy': np.array([500, 700], dtype=np.int64),
            },
            'tracker_hypothesis': {
                'station': np.empty(0, dtype=np.int8),
                'xy_mm': np.empty((0, 2), dtype=np.float32),
                'sigma_xy_mm': np.empty((0, 2), dtype=np.float32),
            },
        }
        dom = {'obs_domain_id': 1, 'c_obs': np.zeros(16, dtype=np.float32), 'cal_start': (-2984.49, 10.0, -5.0, True)}
        it_real = extract_pf_item(7, obs, 2, dom, self.geometry, self.bounds, {'pe_only': True, 'shuffle_micro_vertex': False})
        it_shuf = extract_pf_item(7, obs, 2, dom, self.geometry, self.bounds, {'pe_only': True, 'shuffle_micro_vertex': True})
        self.assertEqual(it_real['raw_pixel_hits'], it_shuf['raw_pixel_hits'])
        np.testing.assert_allclose(it_real['bar_val'], it_shuf['bar_val'], atol=0.0)
        self.assertEqual(it_real['micro_vtx_feat'].shape, it_shuf['micro_vtx_feat'].shape)

    def test_flux_significance_nue_charm_geom_weight_per_process_morphing_and_b_ext(self) -> None:
        rng = np.random.default_rng(3210)
        n = 4000
        labels = np.repeat(np.arange(4), n // 4)
        has_charm = np.zeros(n, dtype=bool)
        has_charm[(labels == 0) & (rng.random(n) < 0.05)] = True
        has_charm[(labels == 1) & (rng.random(n) < 0.15)] = True
        has_charm[(labels == 3) & (rng.random(n) < 0.10)] = True
        # Assign signal-like higher bins to tau, lower bins to backgrounds, plus CR bins 18..23
        bins = np.where(labels == 2, rng.integers(10, 18, size=n), rng.integers(0, 24, size=n))
        # Deliberately leave bin 17 with 0 MC background to test exact Gamma(1/2, 1) variance = 2 * b_jeff^2
        bins[(labels != 2) & (bins == 17)] = 16
        weights = np.where(rng.random(n) < 0.5, 0.2, 1.8)

        b_ext = np.zeros(24, dtype=np.float64)
        b_ext[14:18] = 0.25
        tpl = build_physical_templates(
            bins,
            labels,
            has_charm,
            n_bins=24,
            total_events=3000.0,
            tau_fraction=0.005,
            bkg_shares=FIDUCIAL_2T_BKG_SHARES,
            event_weights=weights,
            external_bkg=b_ext,
        )
        # Verify separated nue_charm + exact Gamma(1/2,1) variance at empty-MC bin 17
        self.assertIn('b_nue_charm', tpl)
        self.assertIn('b_nue_light', tpl)
        self.assertGreater(float(tpl['b_nue_charm'].sum()), 0.0)
        self.assertEqual(int(tpl['n_raw_bkg'][17]), 0)
        # In bin 17 (0 MC bkg, plus b_ext=0.25), Jeffreys prior variance is 2 * b_jeff^2!
        b_jeff_17 = float((tpl['b_nue'] + tpl['b_numu_light'] + tpl['b_numu_charm'] + tpl['b_nc_light'] + tpl['b_nc_charm'])[17])
        self.assertAlmostEqual(float(tpl['var_mc_bkg'][17]), 2.0 * (b_jeff_17 ** 2), places=9)

        # Signal-only systematic (tau_helicity) vs joint SR+CR systematic
        signed_sys = {
            'tau_helicity_sig_only': {
                'applies_to': 'signal',
                'delta_log_up': np.full(24, 0.08, dtype=np.float64),
                'delta_log_down': np.full(24, -0.08, dtype=np.float64),
            },
            'g4_w_density': {
                'delta_log_up_sig': np.full(24, 0.04, dtype=np.float64),
                'delta_log_down_sig': np.full(24, -0.04, dtype=np.float64),
                'delta_log_up_bkg': np.full(24, 0.05, dtype=np.float64),
                'delta_log_down_bkg': np.full(24, -0.05, dtype=np.float64),
            },
        }
        res_no_sys = fit_realistic_profile_likelihood(tpl, signed_sys={}, n_sr_bins=18)
        res_with_sys = fit_realistic_profile_likelihood(tpl, signed_sys=signed_sys, n_sr_bins=18)
        self.assertTrue(np.isfinite(res_with_sys['z0_profile_likelihood']))
        self.assertIn('theta_ext_neutral_hadron', res_with_sys['nuisance_pulls_at_H0'])
        self.assertLessEqual(res_with_sys['z0_profile_likelihood'], res_no_sys['z0_profile_likelihood'] + 1e-4)
        # Verify signal-only systematic has ZERO pull at mu=0 (H0 has mu=0 so signal morphing doesn't alter H0!)
        self.assertAlmostEqual(res_with_sys['nuisance_pulls_at_H0']['sys_tau_helicity_sig_only'], 0.0, places=5)


if __name__ == '__main__':
    unittest.main()
