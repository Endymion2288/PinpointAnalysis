"""Unit tests verifying all Stage 314 independent audit remediations (Stage 316)."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import unittest

import numpy as np
import torch

from analysis.flavour.v3.common.domains import (
    calibrated_bar_channels,
    compute_observable_containment,
    detector_bounds,
    validate_finite_observation,
)
from analysis.flavour.v3.common.fast_kernels import reconstruct_micron_vertex
from analysis.flavour.v3.common.layers import is_weight_decay_exempt, split_adamw_param_groups
from analysis.flavour.v3.v1_b1pp.models import ZAlignedStereoNet
from analysis.flavour.v3.v1_b1pp.representation import extract_b1pp_item, static_plane_layout
from analysis.flavour.v3.v2_pinpointformer.models import PinpointFormer
from analysis.flavour.v3.v2_pinpointformer.pixel_tokenizer import collate_pf, extract_pf_item
from analysis.flavour.v3.v2_pinpointformer.stereo_bars import StereoBarNexus, static_scint_stations
from analysis.flavour.v3.v2_pinpointformer.vertex_anchor import heteroscedastic_vertex_loss
from analysis.flavour.v3.v3_multitask.losses import FixedWeightEventLoss
from analysis.flavour.v3.v3_multitask.models import PinpointFormerMultitask
from analysis.flavour.v3.v4_tau.models import PinpointFormerV4, ZAlignedStereoNetV4
from analysis.flavour.v3.v4_tau.nomad_kinematics import NomadTransverseExtractor, SAGITTA_CONST_MEV_MM
from analysis.flavour.v3.v5_confirm.flux_significance import (
    LHC_FORWARD_SCENARIOS,
    PHYSICAL_TAU_FRACTION,
    build_physical_templates,
    fit_realistic_profile_likelihood,
    run_h0_toy_coverage_verification,
)
from analysis.flavour.v3.v5_confirm.significance import compute_locked_slice_report


GEOMETRY_PATH = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s0_s1_2026-11-11/shards/10083/geometry.json')
REPO_ROOT = Path(__file__).resolve().parents[1]


class TestStage316AuditFixes(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.geometry = json.loads(GEOMETRY_PATH.read_text())
        cls.bounds = detector_bounds(cls.geometry)
        cls.layout = static_plane_layout(cls.geometry)

    def _base_obs(self) -> dict:
        pz0 = float(self.bounds['pixel_Zpos'][1])
        pz1 = float(self.bounds['pixel_Zpos'][2])
        pz2 = float(self.bounds['pixel_Zpos'][3])
        nx = int(self.bounds['nx'])
        ny = int(self.bounds['ny'])
        sz = np.asarray(self.bounds['scint_Zpos'], dtype=np.float32)
        panels = np.array([12, 13, 14, 15], dtype=np.int16)
        b_mm = np.column_stack([
            np.full(4, -50.0, dtype=np.float32),
            np.full(4, 50.0, dtype=np.float32),
            np.full(4, -50.0, dtype=np.float32),
            np.full(4, 50.0, dtype=np.float32),
            sz[panels],
        ])
        return {
            'pixel': {
                'channel_id': np.array([1 * nx * ny + 1000, 2 * nx * ny + 1005, 3 * nx * ny + 1010], dtype=np.int64),
                'xyz_mm': np.array(
                    [
                        [0.10, -10.0, pz0],
                        [0.25, -9.85, pz1],
                        [0.40, -9.70, pz2],
                    ],
                    dtype=np.float32,
                ),
            },
            'bar': {
                'channel_id': np.arange(4, dtype=np.int64),
                'layer': np.zeros(4, dtype=np.int16),
                'panel': panels,
                'horizontal': np.array([0, 1, 0, 1], dtype=np.int8),
                'bar': np.array([18, 22, 19, 21], dtype=np.int16),
                'bounds_mm': b_mm,
                'edep_proxy_MeV': np.array([10.0, 12.0, 8.0, 9.0], dtype=np.float32),
                'pe_proxy': np.array([850, 920, 1, 640], dtype=np.int64),
            },
            'tracker_hypothesis': {
                'station': np.array([0, 1, 2], dtype=np.int8),
                'xy_mm': np.array([[2.0, 5.0], [2.4, 5.9], [2.8, 7.814]], dtype=np.float32),
                'sigma_xy_mm': np.full((3, 2), 0.05, dtype=np.float32),
            },
        }

    def test_p1_1_pe_only_blocks_edep_leakage_and_applies_threshold_and_saturation(self) -> None:
        """Audit P1-1: pe_only mode must ignore bar_edep_proxy_MeV completely & enforce >=2.5 PE + saturation."""
        obs_a = self._base_obs()
        obs_b = copy.deepcopy(obs_a)
        # Perturb unphysical bar_edep_proxy_MeV by 100,000x (even NaN!) while keeping bar_pe_proxy identical!
        obs_b['bar']['edep_proxy_MeV'] = np.array([99999.0, 0.0001, 5000.0, 77777.0], dtype=np.float32)

        cfg = {'pe_only': True, 'bar_threshold_pe': 2.5, 'sipm_npix': 14400.0, 'apply_sipm_saturation': True}
        dom_a = compute_observable_containment(obs_a, self.geometry, self.bounds, config=cfg)
        dom_b = compute_observable_containment(obs_b, self.geometry, self.bounds, config=cfg)
        pf_a = extract_pf_item(0, obs_a, 2, dom_a, self.geometry, self.bounds, cfg)
        pf_b = extract_pf_item(0, obs_b, 2, dom_b, self.geometry, self.bounds, cfg)
        b1_a = extract_b1pp_item(0, obs_a, 2, dom_a, self.geometry, self.bounds, cfg)
        b1_b = extract_b1pp_item(0, obs_b, 2, dom_b, self.geometry, self.bounds, cfg)

        # Sub-threshold hit (1 PE < 2.5 PE) must be dropped -> 3 bars left instead of 4
        self.assertEqual(len(pf_a['bar_idx']), 3)
        np.testing.assert_allclose(pf_a['bar_val'], pf_b['bar_val'], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(pf_a['c_obs'], pf_b['c_obs'], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(b1_a['bx_val'], b1_b['bx_val'], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(b1_a['by_val'], b1_b['by_val'], rtol=0.0, atol=0.0)

        # Verify SiPM saturation monotonicity up to N_pix = 14400
        keep, e_cal, _ = calibrated_bar_channels(
            {
                'edep_proxy_MeV': np.array([1.0, 10.0, 100.0], dtype=np.float32),
                'pe_proxy': np.array([2, 1000, 50000], dtype=np.int64),
            },
            cfg,
        )
        self.assertEqual(keep.tolist(), [False, True, True])
        self.assertGreater(float(e_cal[1]), float(e_cal[0]))
        self.assertTrue(np.isfinite(e_cal).all())

    def test_p0_2_geometry_aware_upstream_veto(self) -> None:
        """Audit P0-2: upstream veto activates dynamically only when scint_Zpos[0] < z_target_min."""
        obs = self._base_obs()
        # Only panel 0 fires along with an interior shower, no side edge bars (bar=20), no downstream exit
        obs['tracker_hypothesis'] = {
            'station': np.empty(0, dtype=np.int8),
            'xy_mm': np.empty((0, 2), dtype=np.float32),
            'sigma_xy_mm': np.empty((0, 2), dtype=np.float32),
        }
        g_fixed = copy.deepcopy(self.geometry)
        sc_z = list(g_fixed['scint_Zpos'])
        sc_z[0] = -2979.490  # Behind first W plate (-2984.490 mm), so NOT an upstream veto!
        g_fixed['scint_Zpos'] = sc_z
        b_fixed = detector_bounds(g_fixed)

        obs['bar'] = {
            'channel_id': np.arange(3, dtype=np.int64),
            'layer': np.zeros(3, dtype=np.int16),
            'panel': np.array([0, 12, 13], dtype=np.int16),
            'horizontal': np.array([0, 0, 1], dtype=np.int8),
            'bar': np.array([20, 20, 20], dtype=np.int16),
            'bounds_mm': np.array(
                [[-5.0, 5.0, -50.0, 50.0, sc_z[0]], [-5.0, 5.0, -50.0, 50.0, sc_z[12]], [-50.0, 50.0, -5.0, 5.0, sc_z[13]]],
                dtype=np.float32,
            ),
            'edep_proxy_MeV': np.array([8.0, 10.0, 10.0], dtype=np.float32),
            'pe_proxy': np.array([2800, 3500, 3500], dtype=np.int64),
        }
        dom_fixed = compute_observable_containment(obs, g_fixed, b_fixed, config={'pe_only': True})
        # Contained event (D_pix_contained == 1), NOT forced to D_partial_or_edge (3) by stale panel-0 veto!
        self.assertEqual(dom_fixed['obs_domain_id'], 1)

        # In legacy geometry where panel 0 is at -2995.50 < z_w0 (-2984.49), panel 0 IS upstream veto -> domain 3!
        obs_leg = copy.deepcopy(obs)
        obs_leg['bar']['bounds_mm'][0, 4] = float(self.bounds['scint_Zpos'][0])
        dom_leg = compute_observable_containment(obs_leg, self.geometry, self.bounds, config={'pe_only': True})
        self.assertEqual(dom_leg['obs_domain_id'], 3)

    def test_p1_3_spectrometer_sagitta_momentum_calibration(self) -> None:
        """Audit P1-3: 3-station sagitta s_y=1.0 mm -> ~100.7 GeV (not ~0.48 GeV), 2-station -> p_spec=0."""
        ext = NomadTransverseExtractor(self.geometry, self.bounds, phys_calibrated=True)
        e_x = torch.zeros((2, ext.n_st, ext.n_bars), dtype=torch.float32)
        e_y = torch.zeros((2, ext.n_dbl, ext.n_bars), dtype=torch.float32)
        trk_cand = torch.zeros((2, 4, 14), dtype=torch.float32)
        trk_mask = torch.zeros((2, 4), dtype=torch.bool)
        trk_mask[:, 0] = True
        trk_cand[0, 0, 0] = 1.0  # 3-station
        trk_cand[0, 0, 13] = math.log1p(1.00 / 0.05) / 5.0
        trk_cand[1, 0, 0] = 0.5  # 2-station
        trk_cand[1, 0, 1] = 3.0

        # Active scintillator 7937 MeV -> 12.5 * 7937 = 99,212.5 MeV ~ p_spec_mev (100705 / 1.015 = 99216.7 MeV)!
        e_x[0, 14, 20] = 7937.0
        e_y[0, 2, 20] = 7937.0
        e_x[1, 14, 20] = 500.0
        e_y[1, 2, 20] = 500.0
        obs_dom = torch.tensor([1, 1], dtype=torch.long)
        feat = ext._compute_from_grids(e_x, e_y, trk_cand, trk_mask, obs_dom)
        self.assertLess(abs(float(feat[0, 8])), 0.01)
        self.assertEqual(float(feat[1, 8]), 0.0)
        self.assertAlmostEqual(SAGITTA_CONST_MEV_MM, 100705.0, places=1)

    def test_p2_2_unbiased_vertex_and_fixed_event_losses(self) -> None:
        """Audit P2-2: stationary point of heteroscedastic_vertex_loss matches true residual error."""
        v_true = torch.zeros((16, 3), dtype=torch.float32)
        v_hat = torch.full((16, 3), 0.02, dtype=torch.float32)
        valid = torch.ones(16, dtype=torch.long)
        log_var = torch.zeros((16, 3), dtype=torch.float32, requires_grad=True)
        opt = torch.optim.Adam([log_var], lr=0.1)
        for _ in range(150):
            opt.zero_grad()
            loss = heteroscedastic_vertex_loss(v_hat, log_var, v_true, valid, legacy_biased=False)
            loss.backward()
            opt.step()
        self.assertLess(float(log_var.mean()), -4.0)
        ev_loss = FixedWeightEventLoss()
        self.assertEqual(sum(p.numel() for p in ev_loss.parameters()), 0)

    def test_p2_3_zero_weight_decay_on_1d_and_scale_parameters(self) -> None:
        """Audit P2-3: LayerNorm, biases, log_scale, len_beta, gates, queries have weight_decay == 0.0."""
        cfg = {'task': 'T4', 'dim': 64, 'heads': 4, 'n_local_layers': 1, 'n_global_layers': 1, 'enable_micro_vertex': True}
        for model in (
            ZAlignedStereoNet(self.layout, {'task': 'T4', 'd_model': 64}),
            PinpointFormer(self.geometry, cfg),
            PinpointFormerMultitask(self.geometry, {**cfg, 'use_fixed_event_loss': True}),
            ZAlignedStereoNetV4(self.layout, {'task': 'T4', 'd_model': 64}, geometry=self.geometry),
            PinpointFormerV4(self.geometry, cfg),
        ):
            groups = model.build_param_groups(1e-3, 0.03) if hasattr(model, 'build_param_groups') else split_adamw_param_groups(model, 1e-3, 0.03)
            param_to_wd = {id(p): g['weight_decay'] for g in groups for p in g['params']}
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if is_weight_decay_exempt(name, p):
                    self.assertEqual(param_to_wd[id(p)], 0.0, msg=f'{type(model).__name__}.{name} had non-zero WD')
                else:
                    self.assertGreater(param_to_wd[id(p)], 0.0, msg=f'{type(model).__name__}.{name} missing WD')

    def test_p2_4_stereo_bar_nexus_exact_y_mirror_symmetry_on_ties(self) -> None:
        """Audit P2-4: equal-energy Y-symmetric bars must NOT jump by 10.5 mm under Y-reflection."""
        stations = static_scint_stations(self.geometry, self.bounds)
        nexus = StereoBarNexus(stations, dim=64, heads=4, drop_path=0.0).eval()

        obs = self._base_obs()
        sz = np.asarray(self.bounds['scint_Zpos'], dtype=np.float32)
        obs['bar'] = {
            'channel_id': np.arange(3, dtype=np.int64),
            'layer': np.zeros(3, dtype=np.int16),
            'panel': np.array([12, 13, 13], dtype=np.int16),
            'horizontal': np.array([0, 1, 1], dtype=np.int8),
            'bar': np.array([20, 10, 29], dtype=np.int16),
            'bounds_mm': np.column_stack([
                np.full(3, -50.0, dtype=np.float32),
                np.full(3, 50.0, dtype=np.float32),
                np.full(3, -50.0, dtype=np.float32),
                np.full(3, 50.0, dtype=np.float32),
                sz[[12, 13, 13]],
            ]),
            'edep_proxy_MeV': np.array([20.0, 15.0, 15.0], dtype=np.float32),
            'pe_proxy': np.array([1500, 1100, 1100], dtype=np.int64),
        }
        cfg = {'pe_only': True}
        dom = compute_observable_containment(obs, self.geometry, self.bounds, config=cfg)
        item = extract_pf_item(0, obs, 1, dom, self.geometry, self.bounds, cfg)
        b_orig = collate_pf([item], mirror_rng=None, y_scint_center=self.bounds['y_scint_center'], n_bars=self.bounds['n_bars'])

        class _AlwaysFlipRNG:
            def random(self, n: int) -> np.ndarray:
                return np.zeros(n, dtype=np.float64)

        b_flip = collate_pf([item], mirror_rng=_AlwaysFlipRNG(), y_scint_center=self.bounds['y_scint_center'], n_bars=self.bounds['n_bars'])
        # 1) Audit P2-4 exact tie bug: 2-bar equal-energy cluster at (bar 18, bar 19) -> flips to (bar 20, bar 21).
        # Previously index-0 tie-breaking chose bar 18 (-15.75 mm) vs bar 20 (+5.25 mm) -> 10.5 mm jump!
        obs_adj = copy.deepcopy(obs)
        obs_adj['bar']['bar'] = np.array([20, 18, 19], dtype=np.int16)
        dom_adj = compute_observable_containment(obs_adj, self.geometry, self.bounds, config=cfg)
        item_adj = extract_pf_item(0, obs_adj, 1, dom_adj, self.geometry, self.bounds, cfg)
        b_adj_orig = collate_pf([item_adj], mirror_rng=None, y_scint_center=self.bounds['y_scint_center'], n_bars=self.bounds['n_bars'])
        b_adj_flip = collate_pf([item_adj], mirror_rng=_AlwaysFlipRNG(), y_scint_center=self.bounds['y_scint_center'], n_bars=self.bounds['n_bars'])
        with torch.no_grad():
            o_adj = nexus(b_adj_orig)
            f_adj = nexus(b_adj_flip)
        yc = float(self.bounds['y_scint_center'])
        # Top-1 nexus crossing (index n_st + 0) must flip from bar 19 (-5.25 mm) to bar 20 (+5.25 mm) with 0.000 mm jump!
        n_st = int(stations['n_stations'])
        self.assertLess(abs(float((o_adj['cal_xyz_mm'][0, n_st, 1] - yc) + (f_adj['cal_xyz_mm'][0, n_st, 1] - yc))), 1e-4)

        # 2) Symmetric pair (bar 10, bar 29): station centroids match elementwise and nexus crossing set is exact
        with torch.no_grad():
            out_orig = nexus(b_orig)
            out_flip = nexus(b_flip)
        y_orig = out_orig['cal_xyz_mm'][0, :, 1] - yc
        y_flip = out_flip['cal_xyz_mm'][0, :, 1] - yc
        mask = out_orig['cal_mask'][0] & (out_orig['cal_measured_xy'][0, :, 1] > 0.5)
        st_mask = mask[:n_st]
        self.assertLess(float(torch.max(torch.abs(y_orig[:n_st][st_mask] + y_flip[:n_st][st_mask]))), 1e-4)
        nex_orig_sorted = torch.sort(y_orig[n_st:][mask[n_st:]]).values
        nex_flip_sorted = torch.sort(-y_flip[n_st:][mask[n_st:]]).values
        self.assertLess(float(torch.max(torch.abs(nex_orig_sorted - nex_flip_sorted))), 1e-4)

    def test_p3_1_and_p3_2_freeze_sha_and_nan_inf_guard(self) -> None:
        """Audit P3-1 & P3-2: freeze manifest commit SHA populated and NaN/Inf observations rejected."""
        freeze_json = json.loads((REPO_ROOT / 'roadmap/v3/v5_freeze_manifest.json').read_text())
        sha = freeze_json.get('pre_unsealing_git_commit')
        self.assertIsInstance(sha, str)
        self.assertEqual(len(sha), 40)

        bad_obs = self._base_obs()
        bad_obs['pixel']['xyz_mm'][0, 1] = np.nan
        with self.assertRaises(ValueError):
            validate_finite_observation(bad_obs)

    def test_p1_4_and_p0_1_slice_fpr_labels_and_realistic_flux_pl(self) -> None:
        """Audit P1-4 & P0-1: explicit operating_point_type + forward-flux PL + H0 toy calibration."""
        rng = np.random.default_rng(123)
        n = 6000
        labels = rng.choice([0, 1, 2, 3], size=n)
        dom = rng.choice([1, 2, 3], size=n)
        probs = rng.dirichlet([1.0, 1.0, 1.0, 1.0], size=n)
        s_tau = np.where(labels == 2, rng.normal(2.5, 1.0, size=n), rng.normal(-1.0, 1.0, size=n))
        applied = {
            'pass_global_1pct': s_tau > 1.3,
            'pass_global_0p1pct': s_tau > 2.1,
            'pass_np_1pct': s_tau > 1.1,
            'pass_np_0p1pct': s_tau > 1.9,
        }
        rep = compute_locked_slice_report(labels, probs, s_tau, applied, np.ones(n, dtype=bool))
        self.assertIn('within_slice_exact_1pct_fpr', rep)
        self.assertEqual(rep['locked_pass_np_1pct']['operating_point_type'], 'global_budget_variable_local_fpr')

        # Within each domain (d=1,2,3), assign 6 score bins 0..5 so signal peaks in bin 5 of every domain!
        local_bin = np.clip(((s_tau + 3.0) / 6.0 * 6.0).astype(np.int64), 0, 5)
        bin_ids = (dom - 1) * 6 + local_bin
        has_charm = (labels != 2) & (rng.random(n) < 0.08)
        scen = LHC_FORWARD_SCENARIOS['HL_LHC_3ab_Trench']
        self.assertAlmostEqual(PHYSICAL_TAU_FRACTION, 502.0 / 240200.0, places=6)
        tpl = build_physical_templates(
            bin_ids,
            labels,
            has_charm,
            n_bins=18,
            total_events=scen['expected_total_events'],
            tau_fraction=scen['tau_fraction'],
            bkg_shares=scen['bkg_shares'],
        )
        pl = fit_realistic_profile_likelihood(tpl, n_sr_bins=18, enable_barlow_beeston=True)
        self.assertTrue(pl['optimizer_converged'])
        self.assertGreater(pl['z0_profile_likelihood'], 0.5)
        self.assertLessEqual(pl['z0_profile_likelihood'], pl['z0_stat_only'] + 1e-6)

        cov = run_h0_toy_coverage_verification(tpl, n_sr_bins=18, n_toys=160, workers=8)
        self.assertGreater(cov['zero_fraction'], 0.30)
        self.assertLess(cov['zero_fraction'], 0.70)

    def test_path_c_micron_multi_track_vertex_and_hybrid_pinpointformer(self) -> None:
        """Path C: C++17 AVR primary vertex + displaced d0/sigma + kink features + hybrid model grad."""
        pz = np.asarray(self.bounds['pixel_Zpos'][:4], dtype=np.float64)
        v_true = np.array([0.0, -10.0, -2984.49], dtype=np.float64)
        slopes = [(0.02, 0.01), (-0.015, 0.02), (0.01, -0.025)]
        pts, planes = [], []
        for sx, sy in slopes:
            for pl_i in range(4):
                dz = float(pz[pl_i] - v_true[2])
                pts.append([v_true[0] + sx * dz, v_true[1] + sy * dz, float(pz[pl_i])])
                planes.append(pl_i)
        # Displaced daughter track shifted by +0.18 mm in Y
        for pl_i in range(1, 4):
            dz = float(pz[pl_i] - v_true[2])
            pts.append([v_true[0] - 0.02 * dz, v_true[1] + 0.18 + 0.03 * dz, float(pz[pl_i])])
            planes.append(pl_i)

        cl_xyz = np.asarray(pts, dtype=np.float64)
        cl_plane = np.asarray(planes, dtype=np.int64)
        n_hits = np.full(len(pts), 2.0, dtype=np.float64)
        lam2 = np.full(len(pts), (0.021 ** 2) / 12.0, dtype=np.float64)
        default_center = (self.bounds['x_pix_center'], self.bounds['y_scint_center'], self.bounds['z_mid'])
        avr_xyz, avr_ok, mv_feat = reconstruct_micron_vertex(
            cl_plane,
            cl_xyz[:, 0],
            cl_xyz[:, 1],
            cl_xyz[:, 2],
            n_hits,
            lam2,
            int(self.bounds['nz']),
            float(self.bounds['dx_mm']),
            float(self.bounds['dy_mm']),
            (0.0, 0.0, 0.0, False),
            default_center,
        )

        self.assertTrue(avr_ok)
        self.assertLess(abs(float(avr_xyz[0] - v_true[0])), 0.05)
        self.assertLess(abs(float(avr_xyz[1] - v_true[1])), 0.10)
        self.assertLess(abs(float(avr_xyz[2] - v_true[2])), 2.50)
        # Displaced track must produce strong IP significance (mv_feat[6] = max_ip_sig_asinh > 0.5)
        self.assertGreater(float(mv_feat[6]), 0.50)

        obs = self._base_obs()
        cfg = {'pe_only': True}
        dom = compute_observable_containment(obs, self.geometry, self.bounds, config=cfg)
        item = extract_pf_item(0, obs, 2, dom, self.geometry, self.bounds, cfg)
        batch = collate_pf([item, item], mirror_rng=None, y_scint_center=self.bounds['y_scint_center'], n_bars=self.bounds['n_bars'])
        model = PinpointFormer(
            self.geometry,
            {'task': 'T4', 'dim': 64, 'heads': 4, 'n_local_layers': 1, 'n_global_layers': 1, 'enable_micro_vertex': True},
        )
        logits = model(batch)
        loss = logits[:, 2].sum()
        loss.backward()
        self.assertIsNotNone(model.micro_vtx_gate.grad)
        self.assertTrue(torch.isfinite(model.micro_vtx_gate.grad).all())

    def test_mapped_items_consistent_bar_keep_mask_under_pe_only(self) -> None:
        """MappedItems must store consistent bar_keep_mask across mixed above/below-threshold events."""
        import tempfile
        from analysis.flavour.v2.r3.mapped_inputs import write_mapped_items

        cfg = {'pe_only': True, 'bar_threshold_pe': 2.5}
        obs_all_keep = self._base_obs()
        obs_all_keep['bar']['pe_proxy'] = np.array([100, 200, 300, 400], dtype=np.int64)
        obs_sub_keep = self._base_obs()
        obs_sub_keep['bar']['pe_proxy'] = np.array([1, 200, 300, 400], dtype=np.int64)

        items = [
            extract_pf_item(
                idx,
                ob,
                idx,
                compute_observable_containment(ob, self.geometry, self.bounds, config=cfg),
                self.geometry,
                self.bounds,
                cfg,
            )
            for idx, ob in enumerate([obs_all_keep, obs_sub_keep, obs_all_keep])
        ]
        with tempfile.TemporaryDirectory() as tmp:
            mi = write_mapped_items([items], Path(tmp) / 'ok')
            self.assertEqual(len(mi), 3)
            self.assertEqual(len(mi[0]['bar_keep_mask']), 4)
            self.assertEqual(len(mi[1]['bar_keep_mask']), 4)
            self.assertEqual(len(mi[2]['bar_keep_mask']), 4)

        with tempfile.TemporaryDirectory() as tmp_bad:
            bad_items = [
                {'x': np.array([1.0], dtype=np.float32)},
                {'x': np.array([2.0], dtype=np.float32), 'opt': np.array([1], dtype=np.int8)},
            ]
            with self.assertRaises(ValueError):
                write_mapped_items([bad_items], Path(tmp_bad) / 'bad')


if __name__ == '__main__':
    unittest.main()
