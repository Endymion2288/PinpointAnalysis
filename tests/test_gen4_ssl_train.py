"""Unit and integration tests for Pinpoint-Gen4 Pillar-III:
  - PinpointSSL-v2 (Sonata-MS, LeJEPA SIGReg, KoLeo, 15 um ray interpolation, zero-leakage plane masking, HW-Inv)
  - Conflict-free optimization (Domain-CGGM active-subset gating, MMPareto norm-preserving QP, MCR synergy)
  - Pinpoint-UParT-13 (unbiased prior init, zero-param NC_ccbar, macro-balanced LORT, 256 KB state table,
    unfrozen top-65% NSWC-TPAUC, Quantile-Pivot, INFERNO Fisher proxy, Gate-A truth invariance)
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch
from torch import nn

from analysis.flavour.gen4.conflict_free_optim import (
    DomainCGGMConfig,
    DomainCGGMProbes,
    DomainCGGMTracker,
    apply_conflict_free_backward,
    collect_gen4_param_groups,
    compute_mcr_synergy_loss,
    compute_modality_presence_masks,
    compute_probe_loss,
    mmpareto_combine_gradients,
    solve_mmpareto_weights,
)
from analysis.flavour.gen4.ssl_sonata_v2 import (
    LeJEPASIGReg,
    PinpointSSLv2Model,
    SonataSSLConfig,
    build_sonata_student_view,
    compute_sonata_v2_ssl_loss,
    inject_online_dark_noise_and_degradation,
    intra_event_koleo_loss,
    sinkhorn_knopp_teacher,
)
from analysis.flavour.gen4.upart13_tpauc import (
    FullDatasetScoreTable,
    PinpointUParT13Head,
    bayesian_tau_llr_from_logits13,
    charm_logit_from_logits13,
    compute_inferno_fisher_proxy,
    compute_nswc_tpauc_loss,
    compute_quantile_pivot_and_rng_loss,
    compute_upart13_lort_loss,
    configure_unfrozen_top_backbone,
    logits7_from_logits13,
)
from analysis.flavour.nextgen.protocols import marginal_four_from_seven
from analysis.flavour.nextgen.seal_guard import verify_inference_truth_invariance


class _MockGen4Backbone(nn.Module):
    """Lightweight mock matching the contracted 5 top-level PinpointGen4Net modules."""

    def __init__(self) -> None:
        super().__init__()
        self.si_vtx = nn.Sequential(nn.Linear(32, 128), nn.SiLU(), nn.Linear(128, 128))
        self.cal_pflow = nn.Sequential(nn.Linear(10, 192), nn.SiLU(), nn.Linear(192, 192))
        self.spec_ipt = nn.Sequential(nn.Linear(16, 256), nn.SiLU(), nn.Linear(256, 256))
        self.si_to_macro = nn.Linear(128, 256)
        self.cal_to_macro = nn.Linear(192, 256)
        self.fusion = nn.Sequential(nn.Linear(256 * 3, 256), nn.SiLU(), nn.Linear(256, 256))
        self.head = PinpointUParT13Head()

    def forward_features(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        bsz = batch['c_obs'].shape[0]
        dev = batch['c_obs'].device
        u = self.si_vtx(batch['pix_feat'][:, :128, :32].float())
        m = batch['pix_valid'][:, :128].bool()
        # Ensure empty-silicon events get zero u contribution
        u = u * m.unsqueeze(-1).float()
        s_si = torch.zeros((bsz, 15, 192), device=dev)
        for p in range(15):
            pm = m & (batch['pix_plane'][:, :128] == p)
            cnt = pm.float().sum(dim=1, keepdim=True).clamp_min(1.0)
            s_si[:, p, :128] = (u * pm.unsqueeze(-1).float()).sum(dim=1) / cnt
        s_cal = self.cal_pflow(batch['scint_grid'][:, :, 0, 0, :].float())
        z_si = self.si_to_macro((u.sum(dim=1)) / m.float().sum(dim=1, keepdim=True).clamp_min(1.0))
        z_cal = self.cal_to_macro(s_cal.mean(dim=1))
        z_spec = self.spec_ipt(batch['muon_kalman_tracks'][:, 0, :].float())
        z_fuse = self.fusion(torch.cat([z_si, z_cal, z_spec], dim=-1))
        return {
            'u_micro': u,
            's_si_meso': s_si,
            's_cal_meso': s_cal,
            'z_si_vtx': z_si,
            'z_cal_pflow': z_cal,
            'z_spec_ipt': z_spec,
            'z_fuse': z_fuse,
            'seg_mask': m,
            'phi_vtx': torch.zeros((bsz, 36), device=dev),
            'psi_nomad': torch.zeros((bsz, 16), device=dev),
            'e_vis_hat_gev': torch.full((bsz,), 350.0, device=dev),
        }

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        feats = self.forward_features(batch)
        out = self.head(
            feats['z_fuse'],
            feats['phi_vtx'],
            feats['psi_nomad'],
            batch['obs_domain_id'],
            feats['e_vis_hat_gev'],
        )
        return {
            'logits': out['logits7'],
            'probs4': out['probs4'],
            'logits13': out['logits13'],
            'tau_llr_score': out['tau_llr_score'],
            'charm_logit': out['charm_logit'],
        }


def _make_dummy_batch(bsz: int = 8, seed: int = 42) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    c_obs = torch.zeros((bsz, 16), dtype=torch.float32)
    c_obs[:, 0] = torch.tensor([1.0 if i % 2 == 0 else 0.0 for i in range(bsz)])
    c_obs[:, 1] = 1.0
    c_obs[:, 2] = torch.tensor([1.0 if i % 3 == 0 else 0.0 for i in range(bsz)])
    obs_dom = torch.where(c_obs[:, 0] > 0.5, 1, 2).long()

    pix_xyz = torch.randn((bsz, 128, 3), generator=g) * 15.0
    pix_plane = torch.randint(0, 12, (bsz, 128), generator=g)
    for p in range(12):
        pix_xyz[:, p * 8 : (p + 1) * 8, 2] = -3000.0 + p * 25.0
        pix_plane[:, p * 8 : (p + 1) * 8] = p
    pix_valid = torch.ones((bsz, 128), dtype=torch.bool)
    pix_valid[c_obs[:, 0] == 0.0] = False
    pix_feat = torch.randn((bsz, 128, 32), generator=g) * 0.3
    pix_feat[:, :, 26] = 0.95

    return {
        'c_obs': c_obs,
        'obs_domain_id': obs_dom,
        'cal_start_mm': torch.tensor([[0.0, -12.0, -2950.0]] * bsz, dtype=torch.float32),
        'has_subthr_readout': torch.ones(bsz),
        'has_edep': torch.zeros(bsz),
        's_mir': torch.ones(bsz),
        'pix_xyz_mm': pix_xyz,
        'pix_plane': pix_plane,
        'pix_valid': pix_valid,
        'pix_feat': pix_feat,
        'pix_plane_summary': torch.randn((bsz, 15, 12), generator=g) * 0.2,
        'pix_patches': torch.zeros((bsz, 64, 2, 32, 32)),
        'patch_cluster_idx': torch.full((bsz, 64), -1, dtype=torch.long),
        'scint_grid': torch.rand((bsz, 54, 2, 40, 10), generator=g),
        'muon_kalman_tracks': torch.randn((bsz, 4, 16), generator=g) * 0.2,
        'muon_kalman_mask': (c_obs[:, 2:3] > 0.5).expand(bsz, 4).clone(),
        'trk_cand': torch.randn((bsz, 8, 16), generator=g) * 0.2,
        'trk_cand_mask': (c_obs[:, 2:3] > 0.5).expand(bsz, 8).clone(),
        'trk_station_counts': torch.zeros((bsz, 3)),
        'label': torch.arange(bsz, dtype=torch.long) % 4,
        'label_7': torch.tensor(([0, 1, 4, 6, 0, 2, 5, 6] * ((bsz + 7) // 8))[:bsz], dtype=torch.long),
        'charm_flag': torch.tensor(([0, 1, 0, 0, 1, 0, 0, 1] * ((bsz + 7) // 8))[:bsz], dtype=torch.long),
        'charm_valid': torch.ones(bsz, dtype=torch.long),
    }


class TestGen4SSLAndTrain(unittest.TestCase):
    def test_mmpareto_norm_and_primary_inner_product(self) -> None:
        g0 = [torch.tensor([3.0, -1.0, 2.0]), torch.tensor([0.5, 1.5])]
        g1 = [torch.tensor([-2.8, 0.9, -1.5]), torch.tensor([-0.4, -1.2])]
        g2 = [torch.tensor([0.1, 2.0, -0.5]), torch.tensor([1.0, -0.3])]
        comb, diag = mmpareto_combine_gradients([g0, g1, g2], gamma0_min=0.45)

        n0 = math.sqrt(sum(float(t.square().sum()) for t in g0))
        nc = math.sqrt(sum(float(t.square().sum()) for t in comb if t is not None))
        dot = sum(float((a * b).sum()) for a, b in zip(comb, g0) if a is not None)
        self.assertAlmostEqual(nc, n0, places=5)
        self.assertGreater(dot, 0.0)
        self.assertGreaterEqual(diag['gammas'][0], 0.449)

        # Solve directly for 4x4 Gram
        gram = torch.eye(4)
        gram[0, 1] = gram[1, 0] = -0.8
        w = solve_mmpareto_weights(gram, gamma0_min=0.45)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=5)
        self.assertGreaterEqual(float(w[0]), 0.449)

    def test_domain_cggm_active_gating_and_conflict_free_step(self) -> None:
        torch.manual_seed(7)
        model = _MockGen4Backbone()
        probes = DomainCGGMProbes(embed_dim=256, num_classes=4)
        model.probes = probes  # attach training-only probes

        # Pure Fortune_ScintOnly batch (a_si == 0 everywhere): kappa_si must stay 1.0!
        b_scint = _make_dummy_batch(4, seed=11)
        b_scint['c_obs'][:, 0] = 0.0
        b_scint['pix_valid'].zero_()
        masks_s = compute_modality_presence_masks(b_scint)
        self.assertFalse(masks_s['si_vtx'].any())

        feats_s = model.forward_features(b_scint)
        pl_s = probes({'si_vtx': feats_s['z_si_vtx'], 'cal_pflow': feats_s['z_cal_pflow'], 'spec_ipt': feats_s['z_spec_ipt']})
        tracker = DomainCGGMTracker()
        kappas_s, _ = tracker.update_and_get_kappas(pl_s, b_scint['label'], masks_s)
        self.assertEqual(kappas_s['si_vtx'], 1.0)

        # Mixed batch: run full conflict-free backward
        b_mix = _make_dummy_batch(8, seed=19)
        masks_m = compute_modality_presence_masks(b_mix)
        feats_m = model.forward_features(b_mix)
        out_m = model.head(
            feats_m['z_fuse'], feats_m['phi_vtx'], feats_m['psi_nomad'],
            b_mix['obs_domain_id'], feats_m['e_vis_hat_gev'],
        )
        pl_m = probes({'si_vtx': feats_m['z_si_vtx'], 'cal_pflow': feats_m['z_cal_pflow'], 'spec_ipt': feats_m['z_spec_ipt']})
        kappas_m, _ = tracker.update_and_get_kappas(pl_m, b_mix['label'], masks_m)
        l_prim, _ = compute_upart13_lort_loss(out_m['logits13'], b_mix)
        l_aux = feats_m['z_si_vtx'].square().mean() + feats_m['z_cal_pflow'].square().mean()
        l_probe, _ = compute_probe_loss(pl_m, b_mix['label'], masks_m)
        l_mcr, _ = compute_mcr_synergy_loss(
            lambda zs, zc, zp: model.fusion(torch.cat([zs, zc, zp], dim=-1)),
            lambda zf: model.head(
                zf,
                torch.zeros(len(zf), 36),
                torch.zeros(len(zf), 16),
                torch.ones(len(zf), dtype=torch.long),
                torch.full((len(zf),), 350.0),
            ),
            feats_m['z_si_vtx'], feats_m['z_cal_pflow'], feats_m['z_spec_ipt'],
            b_mix['label'], masks_m,
        )
        model.zero_grad(set_to_none=True)
        diag = apply_conflict_free_backward(
            model, l_prim, [l_aux],
            presence_masks=masks_m, kappas=kappas_m,
            probe_loss=l_probe, mcr_loss=l_mcr,
        )
        self.assertGreater(diag['mmpareto']['cos_with_primary'], 0.0)
        groups = collect_gen4_param_groups(model)
        for gname in ('si_vtx', 'cal_pflow', 'spec_ipt', 'fusion', 'head'):
            gnorm = sum(float(p.grad.square().sum()) for p in groups[gname] if p.grad is not None)
            self.assertGreater(gnorm, 0.0, f'Dead gradient in group {gname}')

    def test_upart13_unbiased_init_and_stage3_tpauc(self) -> None:
        head = PinpointUParT13Head()
        # Verify slot 12 has ZERO dedicated free parameters (weight matrix is 12xD, not 13xD)
        self.assertEqual(head.proj_12.weight.shape[0], 12)

        with torch.no_grad():
            z0 = torch.zeros(4, 256)
            out = head(z0, torch.zeros(4, 36), torch.zeros(4, 16), torch.tensor([1, 2, 1, 3]), torch.full((4,), 350.0))
            p4_mean = out['probs4'].mean(dim=0)
            # Initial weights small -> probs4 close to uniform [0.25, 0.25, 0.25, 0.25]
            self.assertTrue(torch.allclose(p4_mean, torch.full((4,), 0.25), atol=0.05))
            # Exact additivity check between logits13 and logits7 -> probs4
            p13 = torch.softmax(out['logits13'], dim=-1)
            p4_exact = torch.stack([
                p13[:, 4:6].sum(-1),
                p13[:, 6:10].sum(-1),
                p13[:, 0:4].sum(-1),
                p13[:, 10:13].sum(-1),
            ], dim=-1)
            self.assertLess((p4_exact - out['probs4']).abs().max().item(), 1e-6)

        # FullDatasetScoreTable 256 KB check at N=32,000
        tbl = FullDatasetScoreTable(
            32000,
            np.arange(32000) % 4,
            np.ones(32000, dtype=np.int64),
            charm_flag=(np.arange(32000) % 7 == 0).astype(np.int64),
            label_7=np.where(np.arange(32000) % 4 == 2, 4, 6),
        )
        self.assertEqual(tbl.nbytes, 256000)
        tbl.update(np.arange(1000), np.linspace(-2.0, 2.0, 1000))
        idx_b = tbl.sample_hard_batch_indices(np.random.default_rng(42))
        self.assertEqual(len(idx_b), 128)

        # Unfrozen top-65% NSWC-TPAUC + Quantile-Pivot + INFERNO
        model = _MockGen4Backbone()
        pgs, info = configure_unfrozen_top_backbone(model, top_frac=0.65)
        self.assertGreater(len(pgs), 0)
        self.assertGreater(info['trainable_frac'], 0.45)

        b = _make_dummy_batch(16, seed=31)
        out_f = model(b)
        l_tp, _ = compute_nswc_tpauc_loss(
            out_f['tau_llr_score'], b['label'] == 2, b['obs_domain_id'],
            is_charm_bg=(b['label'] != 2) & (b['charm_flag'] > 0),
        )
        l_piv = compute_quantile_pivot_and_rng_loss(
            out_f['tau_llr_score'], out_f['tau_llr_score'] + 0.01, b['label'] == 2, torch.full((16,), 350.0)
        )
        l_inf = compute_inferno_fisher_proxy(
            out_f['tau_llr_score'], b['label'] == 2, (b['label'] != 2) & (b['charm_flag'] > 0)
        )
        (l_tp + l_piv + l_inf).backward()
        self.assertTrue(torch.isfinite(l_tp) and torch.isfinite(l_piv) and torch.isfinite(l_inf))

    def test_sonata_v2_zero_coordinate_leakage_and_hw_inv(self) -> None:
        torch.manual_seed(9)
        bb = _MockGen4Backbone()
        tea = _MockGen4Backbone()
        ssl = PinpointSSLv2Model(bb)
        tea_heads = {'micro': ssl.proto_micro, 'meso': ssl.proto_meso, 'macro': ssl.proto_macro}

        b = _make_dummy_batch(6, seed=23)
        # Check zero-leakage: perturbing coordinates of dropped planes must produce 0.0 change in student view
        g1 = torch.Generator().manual_seed(100)
        stu1, masks1 = build_sonata_student_view(b, generator=g1)
        b_tamper = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
        for bi in range(6):
            for p in torch.nonzero(masks1['pix_plane_drop_mask'][bi], as_tuple=True)[0].tolist():
                pm = b_tamper['pix_plane'][bi] == p
                b_tamper['pix_xyz_mm'][bi, pm] += 999.0
                b_tamper['pix_feat'][bi, pm] += 42.0
                b_tamper['pix_plane_summary'][bi, p] += 77.0
        g2 = torch.Generator().manual_seed(100)
        stu2, _ = build_sonata_student_view(b_tamper, generator=g2)
        self.assertEqual(float((stu1['pix_xyz_mm'] - stu2['pix_xyz_mm']).abs().max().item()), 0.0)
        self.assertEqual(float((stu1['pix_feat'] - stu2['pix_feat']).abs().max().item()), 0.0)
        self.assertEqual(float((stu1['pix_plane_summary'] - stu2['pix_plane_summary']).abs().max().item()), 0.0)

        # Full SSL loss + backward + Gate-A truth invariance
        loss, metrics = compute_sonata_v2_ssl_loss(ssl, tea, tea_heads, b)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIn('ssl_sonata', metrics)
        self.assertIn('ssl_sigreg', metrics)
        self.assertIn('ssl_hw_inv', metrics)

        # Verify backbone export keys cover all 4 contracted prefixes
        exp = ssl.export_backbone_checkpoint_dict()
        self.assertTrue(all(k.startswith(('si_vtx.', 'cal_pflow.', 'spec_ipt.', 'fusion.')) for k in exp))

        # Gate-A truth invariance on mock Gen4 model
        res_inv = verify_inference_truth_invariance(bb, b, atol=0.0)
        self.assertTrue(res_inv['passed'])
        self.assertEqual(res_inv['max_abs_diff'], 0.0)

    def test_sonata_v2_mode_b_edep_ssl_and_subthr_masking(self) -> None:
        torch.manual_seed(17)
        bb = _MockGen4Backbone()
        tea = _MockGen4Backbone()
        ssl = PinpointSSLv2Model(bb)
        tea_heads = {'micro': ssl.proto_micro, 'meso': ssl.proto_meso, 'macro': ssl.proto_macro}

        b = _make_dummy_batch(4, seed=99)
        b['has_edep'] = torch.ones(4)
        b['pix_feat'][..., 30:32] = torch.randn(4, 128, 2) * 0.5

        loss, metrics = compute_sonata_v2_ssl_loss(ssl, tea, tea_heads, b)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(metrics['ssl_phys_ray'], 0.0)

        # v2 36-dim pix_feat[..., 30:36] continuous edep + double-MIP conversion channel check
        b36 = _make_dummy_batch(4, seed=103)
        b36['has_edep'] = torch.ones(4)
        pf36 = torch.zeros(4, 128, 36)
        pf36[..., :32] = b36['pix_feat']
        pf36[..., 30:36] = torch.randn(4, 128, 6) * 0.5
        b36['pix_feat'] = pf36
        ssl.zero_grad(set_to_none=True)
        loss36, m36 = compute_sonata_v2_ssl_loss(ssl, tea, tea_heads, b36)
        self.assertTrue(torch.isfinite(loss36))
        loss36.backward()
        self.assertGreater(m36['ssl_phys_ray'], 0.0)

    def test_inplace_nc_surgery_on_cached_item(self) -> None:
        from analysis.flavour.gen4.geometry_contract import load_geometry_contract
        from analysis.flavour.gen4.physics_frontend import PIX_FEAT_GEN4_DIM, PIX_PLANE_SUMMARY_DIM
        from analysis.flavour.gen4.run_gen4_ssl_and_tail import _overlay_inplace_nc_surgery_on_cached

        geom = load_geometry_contract()
        host = {
            'run': 10088,
            'chunk': 1,
            'evt_id': 7,
            'c_obs': np.array([1.0, 1.0, 0.0, 0.0, 0.0] + [0.0] * 11, dtype=np.float32),
            'obs_domain_id': 1,
            'cal_start_mm': np.array([0.0, -12.0, -2950.0], dtype=np.float32),
            'has_subthr_readout': 1.0,
            'has_edep': 1.0,
            's_mir': 1.0,
            'pix_xyz_mm': np.zeros((2, 3), dtype=np.float32),
            'pix_plane': np.array([0, 1], dtype=np.int16),
            'pix_feat': np.zeros((2, PIX_FEAT_GEN4_DIM), dtype=np.float32),
            'pix_plane_summary': np.zeros((geom.nz, PIX_PLANE_SUMMARY_DIM), dtype=np.float32),
            'pix_patches_u8': np.zeros((2, 2, 32, 32), dtype=np.uint8),
            'patch_cluster_idx': np.array([0, 1], dtype=np.int16),
            'bar_station': np.array([0, 1], dtype=np.int16),
            'bar_view': np.array([0, 1], dtype=np.int8),
            'bar_idx': np.array([12, 12], dtype=np.int16),
            'bar_feat_10': np.ones((2, 10), dtype=np.float32),
            'muon_kalman_tracks': np.zeros((4, 16), dtype=np.float32),
            'muon_kalman_mask': np.zeros((4,), dtype=bool),
            'trk_cand': np.zeros((8, 16), dtype=np.float32),
            'trk_cand_mask': np.zeros((8,), dtype=bool),
            'trk_station_counts': np.zeros((3,), dtype=np.float32),
            'label': 3,
            'label_7': 6,
            'vertex_xyz_mm': np.array([0.0, -12.0, -2985.0], dtype=np.float32),
            'vis_kinematics': np.array([1.0, -0.5, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        }
        out = _overlay_inplace_nc_surgery_on_cached(host, geom, np.random.default_rng(123))
        self.assertEqual(int(out['label']), 2)
        self.assertEqual(int(out['label_7']), 4)
        self.assertGreaterEqual(len(out['pix_plane']), len(host['pix_plane']))
        self.assertEqual(out['pix_feat'].shape[1], PIX_FEAT_GEN4_DIM)
        self.assertGreater(float(out['tau_decay_L3d_mm']), 0.0)


if __name__ == '__main__':
    unittest.main()
