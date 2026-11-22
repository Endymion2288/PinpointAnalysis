"""Gate A, Gate B, SealAuditGuard, Invariance & Gradient-Accumulation Tests for PinpointUnifiedNet."""

from __future__ import annotations

import copy
import math
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from analysis.flavour.nextgen.dual_regime_modalnet import (
    apply_y_mirror_dual_regime_item,
    collate_dual_regime,
)
from analysis.flavour.nextgen.protocols import NextGenFlavourModel, NextGenModelOutput, student_t_vertex_loss
from analysis.flavour.nextgen.seal_guard import (
    SealAuditGuard,
    SealViolationError,
    TRUTH_BATCH_KEYS,
    assert_unsealed_split,
    verify_inference_truth_invariance,
)
from analysis.flavour.nextgen.tracklet_ops import compute_ipt_spectrometer_interlock
from analysis.flavour.nextgen.unified_pinpoint_net import (
    PinpointUnifiedNet,
    compute_unified_loss,
    compute_unified_window_denominators,
    extract_unified_item,
)
from analysis.flavour.v3.common.domains import compute_observable_containment, cumulative_radiation_length, detector_bounds
from analysis.flavour.v3.v2_pinpointformer.train import move_batch_to_device
from tests.test_nextgen_model_health import (
    _build_32_event_4class_observations,
    run_gate_a_health_check,
    run_gate_b_overfit_32_check,
)
from tests.test_nextgen_models_integration import _zero_modalities
from tests.test_nextgen_seal_and_splits import build_minimal_geometry


BASE_OBS_CFG = {
    'pe_only': True,
    'bar_threshold_pe': 2.5,
    'sipm_npix': 14400.0,
    'apply_sipm_saturation': False,
    'invert_sipm_saturation': True,
}


def _attach_targets(items: list[dict]) -> list[dict]:
    out = []
    for i, it in enumerate(items):
        d = dict(it)
        lbl = int(d.get('label', i % 4))
        d['label'] = lbl
        # Map 4-class to deterministic 7-class for tests: e->0, mu->1+(i%2), tau->5, nc->6
        if lbl == 0:
            d['label_7'] = 0
        elif lbl == 1:
            d['label_7'] = 1 + (i % 2)
        elif lbl == 2:
            d['label_7'] = 5
        else:
            d['label_7'] = 6
        d['tau_branch_valid'] = 1
        d['charm_flag'] = int(i % 3 == 0)
        d['charm_valid'] = 1
        cs = d['cal_start']
        d['vertex_xyz_mm'] = np.array([cs[1], cs[2], cs[0]], dtype=np.float32)
        d['vertex_valid'] = 1
        d['vis_kinematics'] = np.zeros(4, dtype=np.float32)
        d['kin_valid'] = 1
        out.append(d)
    return out


class TestPinpointUnifiedNet(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(42)
        cls.geometry = build_minimal_geometry()
        cls.bounds = detector_bounds(cls.geometry)
        cls.bounds['_plane_x0'] = np.array(
            [cumulative_radiation_length(float(z), cls.geometry) for z in cls.bounds['pixel_Zpos']],
            dtype=np.float32,
        )
        cls.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        cls.yc = float(cls.bounds['y_scint_center'])
        cls.nb = int(cls.bounds['n_bars'])
        cls.obs32, cls.lbl32 = _build_32_event_4class_observations(cls.geometry, cls.bounds, seed=325)
        raw_items = [
            extract_unified_item(
                i,
                cls.obs32[i],
                cls.lbl32[i],
                compute_observable_containment(cls.obs32[i], cls.geometry, cls.bounds, BASE_OBS_CFG),
                cls.geometry,
                cls.bounds,
                BASE_OBS_CFG,
            )
            for i in range(32)
        ]
        cls.items32 = _attach_targets(raw_items)
        cls.b8_cpu = collate_dual_regime(cls.items32[:8], mirror_rng=None, y_scint_center=cls.yc, n_bars=cls.nb)
        cls.b8_dev = move_batch_to_device(cls.b8_cpu, cls.device)
        cls.b32_dev = move_batch_to_device(
            collate_dual_regime(cls.items32, mirror_rng=None, y_scint_center=cls.yc, n_bars=cls.nb),
            cls.device,
        )

    def test_01_seal_audit_guard_and_parameter_budget(self) -> None:
        """SealAuditGuard blocks sealed/Stage-1-forbidden paths and model meets ~7.4M (<8.5M) param budget."""
        with SealAuditGuard() as guard:
            assert_unsealed_split('internal_train', stage=1)
            assert_unsealed_split('internal_val', stage=1)
            with self.assertRaises(SealViolationError):
                assert_unsealed_split('tune6k', stage=1)
            with self.assertRaises(SealViolationError):
                assert_unsealed_split('confirm', stage=2)
            with self.assertRaises(SealViolationError):
                assert_unsealed_split('cal_a', stage=2)
            self.assertEqual(guard.violations, [])

        model = PinpointUnifiedNet(self.geometry, {'dim': 160, 'heads': 8}).to(self.device)
        self.assertIsInstance(model, NextGenFlavourModel)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'[PinpointUnifiedNet] trainable params = {n_params:,} ({n_params / 1e6:.3f}M)', flush=True)
        self.assertGreater(n_params, 6_000_000)
        self.assertLess(n_params, 8_500_000)

    def test_02_gate_a_health_all_variants_and_empty_modalities(self) -> None:
        """Gate A passes with 0 dead branches for full PinpointUnifiedNet and all 3 ablation configs."""
        variants = (
            ('full', {}),
            ('abl_no_recoil_hem', {'use_recoil_hem': False}),
            ('abl_no_online_spec_curriculum', {'enable_online_spec_curriculum': False}),
            ('abl_no_tracklet_rab', {'use_tracklet_rab': False}),
        )
        for vname, vcfg in variants:
            with self.subTest(variant=vname):
                torch.manual_seed(42)
                model = PinpointUnifiedNet(self.geometry, {'dim': 160, 'heads': 8, **vcfg}).to(self.device)

                def _loss_fn(_out: dict, b: dict) -> torch.Tensor:
                    tot, _ = compute_unified_loss(model, b)
                    return tot

                res = run_gate_a_health_check(
                    model,
                    self.b8_dev,
                    _loss_fn,
                    expected_logits_shape=(8, 7),
                    branch_groups_fn=lambda m: {
                        k: [(f'{k}_{i}', p) for i, p in enumerate(ps)]
                        for k, ps in m.branch_parameter_groups().items()
                    },
                )
                self.assertTrue(res['passed'])
                self.assertEqual(res['dead_branches'], [])

                # Empty modalities finite check
                model.eval()
                for tag, zp, zb, zs in (
                    ('zero_pixels', True, False, False),
                    ('zero_bars', False, True, False),
                    ('zero_spec', False, False, True),
                    ('all_empty', True, True, True),
                ):
                    mb = _zero_modalities(self.b8_dev, zp, zb, zs)
                    with torch.inference_mode():
                        out = model.forward_NextGen(mb)
                    self.assertIsInstance(out, NextGenModelOutput)
                    for fname in ('logits7', 'probs4', 'charm_logit', 'v_pv_mm', 'scale_pv_mm', 'sv_disp_mm', 'kin_pred'):
                        t = getattr(out, fname)
                        self.assertTrue(torch.isfinite(t).all(), msg=f'{vname}.{tag}.{fname} non-finite')

    def test_03_invariance_truth_bs1_permutation_ymirror(self) -> None:
        """39-key truth tamper == 0.0, bs=1 vs bs=8 <= 1e-6, hit permutation <= 1e-6, Y-mirror equivariance."""
        prev_mm = torch.backends.cuda.matmul.allow_tf32
        prev_cd = torch.backends.cudnn.allow_tf32
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        try:
            torch.manual_seed(42)
            model = PinpointUnifiedNet(self.geometry, {'dim': 160, 'heads': 8}).to(self.device)
            model.eval()

            # 1. 39-key truth invariance across ALL 7 outputs == 0.0
            res_inv = verify_inference_truth_invariance(model, self.b8_dev, atol=0.0)
            self.assertEqual(res_inv['max_abs_diff'], 0.0)

            clean_b = {k: v for k, v in self.b8_dev.items() if k not in TRUTH_BATCH_KEYS}
            mut_b = copy.copy(self.b8_dev)
            bsz = int(clean_b['c_obs'].shape[0])
            for tk in TRUTH_BATCH_KEYS:
                if tk == 'tau_branch':
                    mut_b[tk] = ['unknown'] * bsz
                elif tk in ('label', 'label_4', 'label_7', 'tau_branch_valid', 'charm_flag', 'charm_valid', 'vertex_valid', 'kin_valid', 'family_hi', 'family_lo'):
                    mut_b[tk] = torch.ones((bsz,), dtype=torch.long, device=self.device)
                elif tk == 'has_charm':
                    mut_b[tk] = torch.ones((bsz,), dtype=torch.bool, device=self.device)
                elif tk == 'vertex_xyz_mm':
                    mut_b[tk] = torch.full((bsz, 3), -987.0, dtype=torch.float32, device=self.device)
                elif tk == 'vis_kinematics':
                    mut_b[tk] = torch.full((bsz, 4), 12.3, dtype=torch.float32, device=self.device)
                elif tk == 'trackID':
                    mut_b[tk] = torch.full((bsz, 8), 54321, dtype=torch.long, device=self.device)
                else:
                    mut_b[tk] = torch.full((bsz,), 77.7, dtype=torch.float32, device=self.device)

            with torch.inference_mode():
                o_clean = model.forward_NextGen(clean_b)
                o_mut = model.forward_NextGen(mut_b)
            for fname in ('logits7', 'probs4', 'charm_logit', 'v_pv_mm', 'scale_pv_mm', 'sv_disp_mm', 'kin_pred'):
                d = float((getattr(o_clean, fname) - getattr(o_mut, fname)).abs().max().item())
                self.assertEqual(d, 0.0, msg=f'{fname} truth tamper diff={d}')

            # 2. bs=1 vs bs=8 <= 1e-6
            b1_dev = move_batch_to_device(
                collate_dual_regime(self.items32[:1], mirror_rng=None, y_scint_center=self.yc, n_bars=self.nb),
                self.device,
            )
            with torch.inference_mode():
                p_b8 = model.forward_NextGen(self.b8_dev).probs4[:1]
                p_b1 = model.forward_NextGen(b1_dev).probs4
            bs_diff = float((p_b8 - p_b1).abs().max().item())
            self.assertLessEqual(bs_diff, 1e-6)

            # 3. Hit permutation invariance <= 1e-6
            rng_p = np.random.default_rng(7)
            p_idx = rng_p.permutation(len(self.obs32[0]['pixel']['channel_id']))
            b_idx = rng_p.permutation(len(self.obs32[0]['bar']['panel']))
            ob_perm0 = {
                'pixel': {
                    'channel_id': np.asarray(self.obs32[0]['pixel']['channel_id'])[p_idx],
                    'xyz_mm': np.asarray(self.obs32[0]['pixel']['xyz_mm'])[p_idx],
                },
                'bar': {k: np.asarray(v)[b_idx] for k, v in self.obs32[0]['bar'].items()},
                'tracker_hypothesis': self.obs32[0]['tracker_hypothesis'],
            }
            if len(self.obs32[1]['tracker_hypothesis']['station']):
                t_idx = rng_p.permutation(len(self.obs32[1]['tracker_hypothesis']['station']))
                ob_perm1 = {
                    'pixel': self.obs32[1]['pixel'],
                    'bar': self.obs32[1]['bar'],
                    'tracker_hypothesis': {k: np.asarray(v)[t_idx] for k, v in self.obs32[1]['tracker_hypothesis'].items()},
                }
            else:
                ob_perm1 = self.obs32[1]
            it_p0 = extract_unified_item(0, ob_perm0, 0, compute_observable_containment(ob_perm0, self.geometry, self.bounds, BASE_OBS_CFG), self.geometry, self.bounds, BASE_OBS_CFG)
            it_p1 = extract_unified_item(1, ob_perm1, 1, compute_observable_containment(ob_perm1, self.geometry, self.bounds, BASE_OBS_CFG), self.geometry, self.bounds, BASE_OBS_CFG)
            bp = move_batch_to_device(collate_dual_regime([it_p0, it_p1], mirror_rng=None, y_scint_center=self.yc, n_bars=self.nb), self.device)
            b2 = move_batch_to_device(collate_dual_regime(self.items32[:2], mirror_rng=None, y_scint_center=self.yc, n_bars=self.nb), self.device)
            with torch.inference_mode():
                o_perm = model.forward_NextGen(bp)
                o_orig = model.forward_NextGen(b2)
            perm_diff = float((o_perm.probs4 - o_orig.probs4).abs().max().item())
            self.assertLessEqual(perm_diff, 1e-6)

            # 4. Y-mirror charge-conjugation & PV equivariance check
            mir_items = [apply_y_mirror_dual_regime_item(it, y_scint_center=self.yc, n_bars=self.nb) for it in self.items32[:4]]
            bm = move_batch_to_device(collate_dual_regime(mir_items, mirror_rng=None, y_scint_center=self.yc, n_bars=self.nb), self.device)
            b4 = move_batch_to_device(collate_dual_regime(self.items32[:4], mirror_rng=None, y_scint_center=self.yc, n_bars=self.nb), self.device)
            is_ipt_4 = b4['trk_mask'] & (b4['trk_morph'][..., 1] > 0.5)
            is_ipt_m = bm['trk_mask'] & (bm['trk_morph'][..., 1] > 0.5)
            il_orig = compute_ipt_spectrometer_interlock(b4['trk_kin'], is_ipt_4, b4['trk_cand'], b4['trk_cand_mask'])
            il_mir = compute_ipt_spectrometer_interlock(bm['trk_kin'], is_ipt_m, bm['trk_cand'], bm['trk_cand_mask'])
            self.assertLessEqual(float((il_orig - il_mir).abs().max().item()), 1e-5)
            self.assertEqual(mir_items[1]['label_7'], 3 - self.items32[1]['label_7'])  # mu- <-> mu+
        finally:
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = prev_mm
                torch.backends.cudnn.allow_tf32 = prev_cd

    def test_04_cpu_fp32_grad_accum_equivalence(self) -> None:
        """CPU FP32 gradient accumulation (bs=8 vs 2x4 microbatches) relative L2 error < 1e-4."""
        items = self.items32[:8]
        b8 = collate_dual_regime(items, mirror_rng=None, y_scint_center=self.yc, n_bars=self.nb)
        mb0 = collate_dual_regime(items[:4], mirror_rng=None, y_scint_center=self.yc, n_bars=self.nb)
        mb1 = collate_dual_regime(items[4:8], mirror_rng=None, y_scint_center=self.yc, n_bars=self.nb)

        cfg = {'dim': 160, 'heads': 8, 'drop_path': 0.0, 'head_dropout': 0.0, 'degradation_aug_prob': 0.0}
        torch.manual_seed(99)
        m_full = PinpointUnifiedNet(self.geometry, cfg).cpu()
        torch.manual_seed(99)
        m_acc = PinpointUnifiedNet(self.geometry, cfg).cpu()
        for mdl in (m_full, m_acc):
            mdl.train()
            for mod in mdl.modules():
                if isinstance(mod, torch.nn.Dropout):
                    mod.p = 0.0
                if hasattr(mod, 'drop_prob'):
                    mod.drop_prob = 0.0

        m_full.zero_grad(set_to_none=True)
        lf, _ = compute_unified_loss(m_full, b8)
        lf.backward()
        g_full = {n: p.grad.detach().float() for n, p in m_full.named_parameters() if p.grad is not None}

        m_acc.zero_grad(set_to_none=True)
        fd = compute_unified_window_denominators([mb0, mb1])
        la_sum = 0.0
        for mb in (mb0, mb1):
            la, _ = compute_unified_loss(m_acc, mb, fixed_denoms=fd)
            la.backward()
            la_sum += float(la.item())
        g_acc = {n: p.grad.detach().float() for n, p in m_acc.named_parameters() if p.grad is not None}

        gn_full = math.sqrt(sum(float(v.square().sum()) for v in g_full.values()))
        gn_diff = math.sqrt(sum(float((g_full[k] - g_acc[k]).square().sum()) for k in g_full))
        rel_diff = gn_diff / max(gn_full, 1e-8)
        print(f'[PinpointUnifiedNet] CPU FP32 grad_accum rel_L2_diff={rel_diff:.3e}', flush=True)
        self.assertLess(rel_diff, 1e-4)
        self.assertLess(abs(float(lf.item()) - la_sum), 1e-5)

    def test_05_gate_b_32_event_overfit(self) -> None:
        """Gate B: 32/32 balanced 4-class overfit convergence with final loss < 0.05."""
        torch.manual_seed(42)
        model = PinpointUnifiedNet(self.geometry, {'dim': 160, 'heads': 8, 'drop_path': 0.0, 'head_dropout': 0.0}).to(self.device)
        res = run_gate_b_overfit_32_check(
            model,
            self.b32_dev,
            lambda out, b: F.nll_loss(torch.log(out['probs4'].clamp(min=1e-6)), b['label']),
            steps=75,
            lr=2.5e-3,
            min_acc=31.0 / 32.0,
            max_final_loss=0.05,
        )
        print(f"[PinpointUnifiedNet] Gate B: acc={res['final_acc']*32:.0f}/32, final_loss={res['final_loss']:.4f}", flush=True)
        self.assertTrue(res['passed'])
        self.assertGreaterEqual(res['final_acc'], 31.0 / 32.0)
        self.assertLess(res['final_loss'], 0.05)

    def test_06_round2_monopixel_filter_and_flood_guard(self) -> None:
        """Round-2 checks: 3x3x3 scale buffer, NC_cc aux keys, monopixel filter & front_gate flood guard."""
        model = PinpointUnifiedNet(self.geometry, {'dim': 160, 'heads': 8}).to(self.device).eval()
        self.assertEqual(tuple(model.pv_regime_scale_corr.shape), (3, 3, 3))

        # Inject 4 isolated monopixels per plane across planes 0..11 on a zero-pixel event
        b_flood = _zero_modalities(self.b8_dev, True, False, False)
        bsz = int(b_flood['pix_mask'].shape[0])
        n_dark = 48
        p_xyz = torch.zeros((bsz, n_dark, 3), device=self.device, dtype=torch.float32)
        p_feat = torch.zeros((bsz, n_dark, 22), device=self.device, dtype=torch.float32)
        p_plane = torch.zeros((bsz, n_dark), device=self.device, dtype=torch.long)
        p_mask = torch.ones((bsz, n_dark), device=self.device, dtype=torch.bool)
        rng = np.random.default_rng(1234)
        for j in range(n_dark):
            pl = j % 12
            p_plane[:, j] = pl
            xr = float(rng.uniform(-110.0, 110.0))
            yr = self.yc + float(rng.uniform(-80.0, 80.0))
            zr = float(self.bounds['pixel_Zpos'][pl])
            p_xyz[:, j, 0] = xr
            p_xyz[:, j, 1] = yr
            p_xyz[:, j, 2] = zr
            p_feat[:, j, 0] = (xr - float(self.bounds['x_pix_center'])) / 135.0
            p_feat[:, j, 1] = (yr - self.yc) / 100.0
            p_feat[:, j, 2] = (zr - float(self.bounds['z_mid'])) / 1000.0
            p_feat[:, j, 3] = math.log1p(1.0) / 6.0
            p_feat[:, j, 6] = 1.0
        b_flood['pix_xyz_mm'] = p_xyz
        b_flood['pix_feat'] = p_feat
        b_flood['pix_plane'] = p_plane
        b_flood['pix_mask'] = p_mask
        with torch.inference_mode():
            out = model.forward_NextGen(b_flood)
        self.assertIn('nccc_active', out.aux)
        self.assertIn('nccc_boost', out.aux)
        # Isolated random dark monopixels should not trip front_gate above 0.5
        self.assertLess(float(out.aux['front_gate'].max().item()), 0.5)


if __name__ == '__main__':
    unittest.main()
