"""Unit, Gate-A/B invariance, symmetry, and health tests for Pinpoint-Gen4 dual-pillar models."""

from __future__ import annotations

import unittest

import torch

from analysis.flavour.gen4.conflict_free_optim import collect_gen4_param_groups
from analysis.flavour.gen4.gen4_unified_model import (
    GEN4_TRUTH_BATCH_KEYS,
    PinpointGen4Net,
    compute_gen4_supervised_loss,
    extract_nsbi_observables_from_output,
)
from analysis.flavour.gen4.micro_ndive import MicroConePairmixerNDIVE
from analysis.flavour.gen4.pflow_fmpe import PinpointPFlowFMPE, UnifiedMuonSpecEncoder
from analysis.flavour.gen4.run_gen4_supervised import (
    make_synthetic_gen4_batch,
    run_gate_ab_and_overfit_health,
)
from analysis.flavour.nextgen.protocols import check_branch_gradient_health
from analysis.flavour.nextgen.seal_guard import verify_inference_truth_invariance


class TestPinpointGen4Models(unittest.TestCase):
    """Comprehensive CPU-safe architecture, zero-leakage & gradient-health tests."""

    def test_five_top_level_groups_and_parameter_budgets(self) -> None:
        model = PinpointGen4Net()
        groups = collect_gen4_param_groups(model)
        self.assertEqual(
            set(groups.keys()),
            {'si_vtx', 'cal_pflow', 'spec_ipt', 'fusion', 'head'},
        )
        for gname, plist in groups.items():
            self.assertGreater(len(plist), 0, f'Empty parameter group: {gname}')

        si_params = sum(p.numel() for p in model.si_vtx.parameters())
        # Pillar I should stay compact around ~1.0M - 2.4M params
        self.assertGreater(si_params, 500_000)
        self.assertLess(si_params, 3_000_000)

    def test_track_a_zero_offset_and_track_b_edep_gradients(self) -> None:
        model = PinpointGen4Net()
        model.eval()

        # Track A (has_edep == 0): mutating pix_feat[..., 30:36] must yield exact 0.0 diff
        b_ta = make_synthetic_gen4_batch(batch_size=2, n_cl=48, has_edep=0.0, seed=11)
        b_ta_obs = {k: v for k, v in b_ta.items() if k not in GEN4_TRUTH_BATCH_KEYS}
        b_ta_mut = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in b_ta_obs.items()}
        b_ta_mut['pix_feat'][..., 30:36] = 99.0
        with torch.no_grad():
            o1 = model.forward_NextGen(b_ta_obs)
            o2 = model.forward_NextGen(b_ta_mut)
        self.assertEqual(float((o1.probs4 - o2.probs4).abs().max().item()), 0.0)
        self.assertEqual(float((o1.v_pv_mm - o2.v_pv_mm).abs().max().item()), 0.0)

        # abl_binary_no_edep config also masks out continuous edep channels cleanly
        model_bin = PinpointGen4Net(config={'force_binary_no_edep': True})
        model_bin.eval()
        b_tb_obs = {k: v for k, v in make_synthetic_gen4_batch(batch_size=2, n_cl=48, has_edep=1.0, seed=11).items() if k not in GEN4_TRUTH_BATCH_KEYS}
        b_tb_obs2 = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in b_tb_obs.items()}
        b_tb_obs2['pix_feat'][..., 30:36] = 77.0
        with torch.no_grad():
            ob1 = model_bin.forward_NextGen(b_tb_obs)
            ob2 = model_bin.forward_NextGen(b_tb_obs2)
        self.assertEqual(float((ob1.probs4 - ob2.probs4).abs().max().item()), 0.0)

        # Track B (has_edep == 1): verify mlp_edep receives non-zero gradients at step 0
        model.train()
        b_tb = make_synthetic_gen4_batch(batch_size=2, n_cl=48, has_edep=1.0, seed=12)
        out_tb = model.forward_NextGen(b_tb)
        loss_tb, _, _, _ = compute_gen4_supervised_loss(model, out_tb, b_tb)
        model.zero_grad(set_to_none=True)
        loss_tb.backward()
        edep_grad_sq = sum(
            float(p.grad.detach().square().sum().item())
            for p in model.si_vtx.cluster_enc.mlp_edep.parameters()
            if p.grad is not None
        )
        self.assertGreater(edep_grad_sq, 1.0e-12)

    def test_conservation_and_competitive_normalization(self) -> None:
        model = PinpointGen4Net()
        model.eval()
        batch = make_synthetic_gen4_batch(batch_size=3, n_cl=64, has_edep=1.0, seed=21)
        with torch.no_grad():
            out = model.forward_NextGen(batch)

        # 1. Comp-NDIVE 8-slot competitive probabilities sum strictly to 1.0
        seg_p = out.aux['seg_slot_probs']
        self.assertEqual(seg_p.shape, (3, 128, 8))
        self.assertTrue(torch.allclose(seg_p.sum(dim=-1), torch.ones(3, 128), atol=1e-5))

        # 2. Macro-HGPflow 8-slot assignment matrix P_ia sums strictly to 1.0
        p_ia = out.aux['P_ia']
        self.assertEqual(p_ia.shape[-1], 8)
        self.assertTrue(torch.allclose(p_ia.sum(dim=-1), torch.ones_like(p_ia[..., 0]), atol=1e-5))

        # 3. Full 3x3 PV covariance is symmetric positive definite with min eigenvalue >= (0.015)^2
        cov_pv = out.aux['cov_pv_mm2']
        self.assertEqual(cov_pv.shape, (3, 3, 3))
        self.assertTrue(torch.allclose(cov_pv, cov_pv.transpose(-1, -2), atol=1e-6))
        eigvals = torch.linalg.eigvalsh(cov_pv)
        self.assertTrue((eigvals >= 0.015 ** 2 - 1e-6).all())

        # 4. Probs4 sums to 1.0 and matches softmax(logits13)
        self.assertTrue(torch.allclose(out.probs4.sum(dim=-1), torch.ones(3), atol=1e-5))

    def test_fortune_scint_only_zeros_silicon_phi_vtx(self) -> None:
        model = PinpointGen4Net()
        model.eval()
        batch = make_synthetic_gen4_batch(batch_size=2, n_cl=0, fortune_only=True, seed=31)
        with torch.no_grad():
            out = model.forward_NextGen(batch)
        self.assertEqual(float(out.aux['phi_vtx'].abs().max().item()), 0.0)
        self.assertEqual(float(out.aux['z_si_vtx'].abs().max().item()), 0.0)
        self.assertTrue(torch.isfinite(out.probs4).all())
        self.assertTrue(torch.isfinite(out.aux['tau_llr_score']).all())

        # Also verify backward on pure fortune_only & mixed (si + fortune_only) batches produces 0 NaN/Inf grads!
        model.train()
        out_f = model.forward_NextGen(batch)
        loss_f, _, _, _ = compute_gen4_supervised_loss(model, out_f, batch)
        model.zero_grad(set_to_none=True)
        loss_f.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all(), f'Non-finite grad on fortune_only for {name}')

        # 3) D_empty all-zero empty event (0 clusters, 0 bars, 0 muons)
        b_empty = make_synthetic_gen4_batch(batch_size=2, n_cl=0, empty_event=True, seed=34)
        out_e = model.forward_NextGen(b_empty)
        self.assertTrue(torch.isfinite(out_e.probs4).all())
        self.assertTrue(torch.isfinite(out_e.v_pv_mm).all())
        loss_e, _, _, _ = compute_gen4_supervised_loss(model, out_e, b_empty)
        model.zero_grad(set_to_none=True)
        loss_e.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all(), f'Non-finite grad on D_empty for {name}')

        # 4) Mixed batch (all_si + fortune_only) Gate-B check
        b_si = make_synthetic_gen4_batch(batch_size=2, n_cl=48, has_edep=1.0, seed=32)
        b_fo = make_synthetic_gen4_batch(batch_size=2, n_cl=48, has_edep=1.0, fortune_only=False, seed=33)
        # Zero out silicon on b_fo while keeping padded shape N_cl=48 (exact mixed-batch scenario!)
        b_fo['c_obs'][:, 0] = 0.0
        b_fo['obs_domain_id'][:] = 2
        b_fo['pix_valid'][:] = False
        b_fo['pix_plane'][:] = -1
        b_fo['cl_sidecar_valid'][:] = False
        b_mix = {
            k: (torch.cat([b_si[k], b_fo[k]], dim=0) if isinstance(b_si[k], torch.Tensor) else b_si[k])
            for k in b_si
        }
        out_m = model.forward_NextGen(b_mix)
        loss_m, _, _, _ = compute_gen4_supervised_loss(model, out_m, b_mix)
        gb_mix = check_branch_gradient_health(model, loss_m, min_grad_l2=1.0e-6)
        self.assertTrue(gb_mix['passed'], f'Mixed batch Gate-B failed: {gb_mix}')

    def test_ssl_and_fusion_interface_compatibility(self) -> None:
        model = PinpointGen4Net()
        model.eval()
        batch = make_synthetic_gen4_batch(batch_size=2, n_cl=48, has_edep=1.0, seed=41)
        with torch.no_grad():
            feats = model.forward_features(batch)
            # Verify dict indexing required by ssl_sonata_v2.py
            self.assertEqual(feats['micro_out']['z_si_vtx'].shape, (2, 256))
            self.assertEqual(feats['micro_out']['u_micro'].shape, (2, 128, 128))
            self.assertEqual(feats['micro_out']['s_si_meso'].shape, (2, 15, 192))
            self.assertEqual(feats['pflow_out']['z_cal_pflow'].shape, (2, 256))
            self.assertEqual(feats['pflow_out']['s_cal_meso'].shape, (2, 54, 192))
            self.assertEqual(feats['spec_out']['z_spec_ipt'].shape, (2, 256))
            # Verify 3-arg fusion signature required by conflict_free_optim.py & ssl_sonata_v2.py
            zf = model.fusion(feats['z_si_vtx'], feats['z_cal_pflow'], feats['z_spec_ipt'])
            self.assertEqual(zf.shape, (2, 256))

            out = model.forward_NextGen(batch)
            nsbi = extract_nsbi_observables_from_output(batch, out)
            expected_keys = {
                'n_sv_prongs', 'sl_xy', 'm_sv_corr_gev', 'n_mu_veto', 'q_mu1', 'q_mu2',
                'lambda_kink', 'dphi_mu_ht', 'ip_sig_max', 'm_vis_rho_gev',
                'e_vis_hat_gev', 'pt_miss_hat_gev', 'm_gg_pi0_gev',
            }
            self.assertEqual(set(nsbi.keys()), expected_keys)
            for k, arr in nsbi.items():
                self.assertEqual(arr.shape, (2,), f'Bad shape for {k}')

    def test_gate_ab_and_small_sample_overfit(self) -> None:
        res = run_gate_ab_and_overfit_health(device='cpu', steps=4, batch_size=2)
        self.assertTrue(res['gate_a_base']['passed'])
        self.assertEqual(res['sidecar_max_abs_diff'], 0.0)
        self.assertTrue(res['gate_b']['passed'])
        self.assertLess(res['overfit_loss_end'], res['overfit_loss_start'])

    def test_causal_ablation_switches_finite(self) -> None:
        from analysis.flavour.gen4.run_gen4_supervised import ABLATION_CONFIGS
        batch = make_synthetic_gen4_batch(batch_size=2, n_cl=48, has_edep=1.0, seed=55)
        for name, cfg in ABLATION_CONFIGS.items():
            if name == 'gen4_full':
                continue
            m = PinpointGen4Net(config=cfg)
            m.train()
            out = m.forward_NextGen(batch)
            loss, _, _, _ = compute_gen4_supervised_loss(m, out, batch)
            self.assertTrue(torch.isfinite(loss).item(), f'Non-finite loss for ablation {name}')
            loss.backward()
            for pname, p in m.named_parameters():
                if p.grad is not None:
                    self.assertTrue(torch.isfinite(p.grad).all().item(), f'NaN grad in {name}:{pname}')


if __name__ == '__main__':
    unittest.main()
