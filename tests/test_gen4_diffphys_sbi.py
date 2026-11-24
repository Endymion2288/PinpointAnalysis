"""Unit tests for Pinpoint-Gen4 Pillar IV: DiffPhys-SBI operators, POT-GPD, and NSBI-PL engine."""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from analysis.flavour.gen4.diffphys_sbi import (
    DIFFPHYS_STAT_DIM,
    BrokenLineKalmanKinkDetector,
    DiffPhysSBIModel,
    RossiGreisenFISTAShowerStripper,
    ScintStationDesaturator,
    diffphys_sbi_loss,
    extract_diffphys_sufficient_stats,
    strip_shower_and_score_segments,
)
from analysis.flavour.gen4.nsbi_discovery import (
    CHANNEL_NAMES,
    DomainPOTGPDCalibrator,
    compare_pot_gpd_vs_empirical_split_half,
    evaluate_nsbi_vs_single_bin_closure,
    fit_pot_gpd,
    harrell_davis_quantile,
    pot_gpd_threshold,
    route_golden_micro_channels,
    run_nsbi_toy_mc_closure,
)
from analysis.flavour.nextgen.protocols import check_branch_gradient_health
from analysis.flavour.nextgen.seal_guard import verify_inference_truth_invariance
from analysis.flavour.v3.common.domains import detector_bounds
from tests.test_nextgen_model_health import _build_32_event_4class_observations
from tests.test_nextgen_seal_and_splits import build_minimal_geometry


class TestGen4DiffPhysSBIAndNSBI(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.geometry = build_minimal_geometry()
        cls.bounds = detector_bounds(cls.geometry)
        cls.obs32, cls.lbl32 = _build_32_event_4class_observations(cls.geometry, cls.bounds, seed=329)

    def test_fista_shower_stripping_improves_segment_purity_and_preserves_tau(self) -> None:
        """Rossi-Greisen FISTA stripping boosts 3-hit segment purity vs dense EM fog while keeping tau recall >= 88%."""
        stripper = RossiGreisenFISTAShowerStripper(self.geometry, n_fista_steps=10)
        pz = np.asarray(self.geometry['pixel_Zpos'], dtype=np.float64)

        rng = np.random.default_rng(2026)
        # Build synthetic Front30 event: 1 clean 3-hit tau prong (trackID=99, origin=1) + dense gamma->e+e- EM fog (origin=3)
        pts = [
            [2.0, -10.0, pz[0]],
            [2.15, -9.85, pz[1]],
            [2.30, -9.70, pz[2]],
        ]
        pls = [0, 1, 2]
        orig = [1, 1, 1]
        tids = [99, 99, 99]

        # Inject 24 dense EM conversion clusters around (20.0, -25.0) across planes 0,1,2 with different random trackIDs
        for p_idx in (0, 1, 2):
            for k in range(8):
                pts.append([20.0 + float(rng.normal(0.0, 0.25)), -25.0 + float(rng.normal(0.0, 0.25)), pz[p_idx]])
                pls.append(p_idx)
                orig.append(3)
                tids.append(1000 + p_idx * 10 + k)

        res = strip_shower_and_score_segments(
            np.asarray(pts, dtype=np.float64),
            np.asarray(pls, dtype=np.int64),
            stripper,
            w_mip_thr=0.50,
            cl_origin_label=np.asarray(orig, dtype=np.int64),
            cl_dominant_track_id=np.asarray(tids, dtype=np.int64),
        )
        self.assertGreaterEqual(res['tau_cluster_recall'], 0.88)
        self.assertGreater(res['stripped_purity'], res['raw_purity'] * 3.0)
        self.assertGreater(res['stripped_purity'], 0.25)

        # Mode-B edep check: 2-MIP conversion cluster boosts FISTA stripping while preserving 1-MIP tau
        edep_mip = np.where(np.asarray(orig) == 1, 1.0, 2.0)
        res_ed = strip_shower_and_score_segments(
            np.asarray(pts, dtype=np.float64),
            np.asarray(pls, dtype=np.int64),
            stripper,
            cluster_edep_mip=edep_mip,
            w_mip_thr=0.50,
            cl_origin_label=np.asarray(orig, dtype=np.int64),
            cl_dominant_track_id=np.asarray(tids, dtype=np.int64),
        )
        self.assertGreaterEqual(res_ed['tau_cluster_recall'], 0.88)
        self.assertGreater(res_ed['stripped_purity'], 0.25)

    def test_broken_line_kalman_kink_detection_and_32d_extraction(self) -> None:
        """Broken-line Kalman detects single-plane kink > 16 chi2 and extracts finite 32D stats + covariance."""
        kink_det = BrokenLineKalmanKinkDetector(self.geometry, self.bounds)
        pz = np.asarray(self.geometry['pixel_Zpos'], dtype=np.float64)
        # Single track entering straight p0->p1, then kinking by 35 mrad at p1->p2
        dz01 = pz[1] - pz[0]
        dz12 = pz[2] - pz[1]
        pts = np.array([
            [0.0, -12.0, pz[0]],
            [0.002 * dz01, -12.0, pz[1]],
            [0.002 * dz01 + 0.037 * dz12, -12.0 + 0.015 * dz12, pz[2]],
        ], dtype=np.float64)
        pls = np.array([0, 1, 2], dtype=np.int64)
        wm = np.ones(3, dtype=np.float64)
        cal_dummy = {
            'e_station_mev': np.zeros(self.bounds['n_scint_panels'] // 2 + 6, dtype=np.float64),
            'x_station_mm': np.zeros(54, dtype=np.float64),
        }
        k_res = kink_det.analyze(pts, pls, wm, {'station': (), 'xy_mm': np.empty((0, 2))}, cal_dummy)
        self.assertGreater(k_res['max_kink_llr'], 16.0)
        self.assertGreater(k_res['kink_pt_gev'], 0.15)

        desat = ScintStationDesaturator(self.geometry, self.bounds)
        stripper = RossiGreisenFISTAShowerStripper(self.geometry)
        st = extract_diffphys_sufficient_stats(
            self.obs32[0], self.geometry, self.bounds,
            desaturator=desat, stripper=stripper, kink_detector=kink_det,
        )
        self.assertEqual(st['diffphys_z'].shape, (DIFFPHYS_STAT_DIM,))
        self.assertEqual(st['diffphys_z_sigma'].shape, (DIFFPHYS_STAT_DIM,))
        self.assertTrue(np.isfinite(st['diffphys_z']).all())
        self.assertTrue(np.all(st['diffphys_z_sigma'] > 0.0))

    def test_diffphys_sbi_model_param_budget_health_and_truth_invariance(self) -> None:
        """DiffPhysSBIModel stays in 85k-150k params, passes Gate-A grad health & atol=0.0 truth invariance."""
        torch.manual_seed(42)
        desat = ScintStationDesaturator(self.geometry, self.bounds)
        stripper = RossiGreisenFISTAShowerStripper(self.geometry)
        kink_det = BrokenLineKalmanKinkDetector(self.geometry, self.bounds)
        stats = [
            extract_diffphys_sufficient_stats(
                ob, self.geometry, self.bounds, desaturator=desat, stripper=stripper, kink_detector=kink_det
            )
            for ob in self.obs32
        ]
        z_t = torch.from_numpy(np.stack([s['diffphys_z'] for s in stats]))
        zs_t = torch.from_numpy(np.stack([s['diffphys_z_sigma'] for s in stats]))
        dom_t = torch.tensor([s['obs_domain_id'] for s in stats], dtype=torch.long)
        e_t = torch.tensor([s['e_vis_gev'] for s in stats], dtype=torch.float32)
        pv_t = torch.from_numpy(np.stack([s['pv_xyz_mm'] for s in stats]))
        y4_t = torch.tensor(self.lbl32, dtype=torch.long)
        y7_t = torch.where(y4_t == 0, 0, torch.where(y4_t == 1, 1, torch.where(y4_t == 2, 5, 6)))

        batch = {
            'diffphys_z': z_t,
            'diffphys_z_sigma': zs_t,
            'obs_domain_id': dom_t,
            'e_vis_gev': e_t,
            'pv_xyz_mm': pv_t,
            'label': y4_t,
            'label_7': y7_t,
            'charm_flag': torch.zeros_like(y4_t),
            'charm_valid': torch.ones_like(y4_t),
            'vertex_xyz_mm': pv_t.clone(),
            'vertex_valid': torch.ones_like(y4_t),
        }

        model = DiffPhysSBIModel()
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertGreaterEqual(n_params, 85000)
        self.assertLessEqual(n_params, 150000)
        model.set_normalization(z_t.mean(dim=0), z_t.std(dim=0).clamp_min(0.1))

        inv_res = verify_inference_truth_invariance(model, batch, atol=0.0)
        self.assertTrue(inv_res['passed'])
        self.assertEqual(inv_res['max_abs_diff'], 0.0)

        loss0, _ = diffphys_sbi_loss(model, batch)
        g_res = check_branch_gradient_health(model, loss0, min_grad_l2=1e-6)
        self.assertTrue(g_res['passed'], msg=str(g_res))

        # Verify 32-event overfit convergence
        opt = torch.optim.AdamW(model.parameters(), lr=4e-3)
        for _ in range(80):
            opt.zero_grad(set_to_none=True)
            loss, _ = diffphys_sbi_loss(model, batch)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            out = model.forward_NextGen(batch)
            acc = float((out.probs4.argmax(dim=-1) == y4_t).float().mean().item())
        self.assertGreaterEqual(acc, 0.90)

    def test_pot_gpd_calibrator_reduces_tail_variance(self) -> None:
        """POT-GPD reduces 0.1% FPR split-half threshold variance vs empirical quantile."""
        rng = np.random.default_rng(1234)
        # Exponential/Laplace tail representative of log-likelihood ratios across 4,500 background events
        bg = rng.laplace(loc=-1.0, scale=1.2, size=4500)
        sg = rng.normal(loc=2.5, scale=1.4, size=1500)
        scores = np.concatenate([bg, sg])
        labels = np.concatenate([np.ones(len(bg), dtype=int), np.full(len(sg), 2, dtype=int)])

        hd_95 = harrell_davis_quantile(bg, 0.95)
        self.assertTrue(np.isfinite(hd_95))

        cal = DomainPOTGPDCalibrator(anchor_q=0.95)
        cal.fit(scores, labels != 2, np.ones_like(labels))
        t_1pct = cal.threshold(0.01, domain_id=1)
        t_0p1pct = cal.threshold(0.001, domain_id=1)
        self.assertGreater(t_0p1pct, t_1pct)

        # Independent-draw variance comparison (matching 2,250 background events per split-half)
        thrs_emp, thrs_gpd = [], []
        for _ in range(120):
            draw = rng.laplace(loc=-1.0, scale=1.2, size=2250)
            thrs_emp.append(float(np.quantile(draw, 0.999)))
            fit = fit_pot_gpd(draw, anchor_q=0.95)
            thrs_gpd.append(pot_gpd_threshold(fit, 0.001))
        var_red = float(np.var(thrs_emp, ddof=1) / max(np.var(thrs_gpd, ddof=1), 1e-12))
        self.assertGreaterEqual(var_red, 3.0)

        sh = compare_pot_gpd_vs_empirical_split_half(scores, labels, fpr_targets=(0.01, 0.001), n_splits=50)
        self.assertIn('0.001', sh['targets'])

    def test_nsbi_golden_micro_channels_and_toy_mc_closure(self) -> None:
        """4 Golden Micro-Channels + 3 CRs NSBI-PL outperforms 1% FPR single-bin and passes H0/H1 closure."""
        rng = np.random.default_rng(777)
        n = 6000
        labels = rng.choice([0, 1, 2, 3], size=n, p=[0.25, 0.25, 0.25, 0.25])
        has_charm = (labels != 2) & (rng.random(n) < 0.05)
        obs_dom = rng.choice([1, 2, 3], size=n, p=[0.25, 0.55, 0.20])

        tau_llr = rng.normal(-1.5, 1.0, size=n)
        e_vis = rng.lognormal(mean=5.4, sigma=0.7, size=n)
        pt_miss = rng.exponential(1.0, size=n)
        n_sv_prongs = np.zeros(n, dtype=np.int64)
        sl_xy = np.zeros(n, dtype=np.float64)
        m_sv = np.zeros(n, dtype=np.float64)
        n_mu = np.where(labels == 1, 1, 0).astype(np.int64)
        kink_llr = rng.exponential(1.5, size=n)
        dphi_mu = rng.uniform(0.0, 3.14, size=n)
        ip_sig = rng.exponential(1.2, size=n)
        pi0_frac = np.where(labels == 3, 0.55, 0.10)
        rho_mass = np.zeros(n, dtype=np.float64)
        charm_prob = np.where(has_charm, 0.70, 0.05)

        # Populate realistic golden topologies for tau signal (with tiny background leakage)
        tau_idx = np.flatnonzero(labels == 2)
        for i in tau_idx:
            r = rng.random()
            if r < 0.25 and obs_dom[i] == 1:  # C1 3-prong
                n_mu[i] = 0
                n_sv_prongs[i] = 3
                sl_xy[i] = rng.uniform(4.5, 14.0)
                m_sv[i] = rng.uniform(0.6, 1.5)
                tau_llr[i] = rng.normal(2.8, 0.9)
            elif r < 0.45 and obs_dom[i] in (1, 3):  # C2 kink mu
                n_mu[i] = 1
                kink_llr[i] = rng.uniform(15.0, 40.0)
                dphi_mu[i] = rng.uniform(2.3, 3.1)
                tau_llr[i] = rng.normal(2.2, 0.9)
            elif r < 0.65 and obs_dom[i] == 1:  # C3 rho 1-prong
                n_mu[i] = 0
                ip_sig[i] = rng.uniform(5.0, 15.0)
                rho_mass[i] = rng.uniform(0.55, 0.95)
                pi0_frac[i] = 0.45
                tau_llr[i] = rng.normal(2.4, 0.9)
            else:  # C4 Fortune high MT
                pt_miss[i] = rng.uniform(2.5, 8.0)
                e_vis[i] = rng.uniform(350.0, 1200.0)
                tau_llr[i] = rng.normal(1.2, 1.0)

        # Inject a bit of dimuon charm into CR-mumu
        ch_idx = np.flatnonzero(has_charm)
        n_mu[ch_idx[:len(ch_idx) // 2]] = 2

        obs_summary = {
            'obs_domain_id': obs_dom,
            'e_vis_gev': e_vis,
            'pt_miss_gev': pt_miss,
            'tau_llr': tau_llr,
            'charm_prob': charm_prob,
            'n_sv_prongs': n_sv_prongs,
            'sl_xy': sl_xy,
            'm_sv_corr_gev': m_sv,
            'n_mu_veto': n_mu,
            'lambda_kink': kink_llr,
            'dphi_mu_ht': dphi_mu,
            'ip_sig_max': ip_sig,
            'pi0_em_frac': pi0_frac,
            'm_vis_rho_gev': rho_mass,
            'has_prompt_e': (labels == 0) & (rng.random(n) < 0.8),
            'has_front_pixel': obs_dom == 1,
            'has_target_pixel': np.isin(obs_dom, [1, 3]),
            # Truth key deliberately injected to verify router ignores it completely
            'label': labels,
        }

        routing1 = route_golden_micro_channels(obs_summary)
        obs_mut = dict(obs_summary)
        obs_mut['label'] = np.zeros_like(labels)
        routing2 = route_golden_micro_channels(obs_mut)
        self.assertTrue(np.array_equal(routing1['channel_id'], routing2['channel_id']))
        for cname in CHANNEL_NAMES[:4] + CHANNEL_NAMES[5:8]:
            self.assertGreater(routing1['counts'][cname], 0, msg=f'Empty channel {cname}')

        closure = evaluate_nsbi_vs_single_bin_closure(
            obs_summary, labels, has_charm=has_charm, total_events=180000.0, tau_fraction=0.003, delta_sys=0.0159
        )
        self.assertGreater(closure['z0_gain_factor_pl_bb_lite'], 2.0)
        self.assertGreater(
            closure['nsbi_unbinned_pl']['z0_unbinned_profile_likelihood'],
            closure['single_bin_1pct_fpr']['z0_pl_no_bb'],
        )

        toy = run_nsbi_toy_mc_closure(closure['templates'], delta_sys=0.0159, n_toys_h0=80, n_toys_h1=80)
        self.assertGreater(toy['h0']['zero_fraction'], 0.30)
        self.assertLess(toy['h0']['zero_fraction'], 0.70)
        self.assertGreater(toy['h1']['median_over_asimov_ratio'], 0.70)
        self.assertLess(toy['h1']['median_over_asimov_ratio'], 1.30)


if __name__ == '__main__':
    unittest.main()
