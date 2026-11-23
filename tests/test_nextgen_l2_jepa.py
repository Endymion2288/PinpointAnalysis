"""Unit, Anti-Shortcut (NOMAE), Gate A/B, SealAuditGuard, and Invariance Tests for PinpointSSLJEPA."""

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
from analysis.flavour.nextgen.pinpoint_ssl_jepa import (
    PinpointSSLJEPA,
    compute_jepa_finetune_loss,
    compute_jepa_finetune_window_denominators,
    compute_ssl_pretrain_loss,
    extract_jepa_item,
    sample_ssl_masks,
    vicreg_variance_covariance_loss,
)
from analysis.flavour.nextgen.protocols import NextGenFlavourModel, NextGenModelOutput
from analysis.flavour.nextgen.seal_guard import (
    SealAuditGuard,
    SealViolationError,
    TRUTH_BATCH_KEYS,
    assert_unsealed_split,
    verify_inference_truth_invariance,
)
from analysis.flavour.v3.common.domains import (
    compute_observable_containment,
    cumulative_radiation_length,
    detector_bounds,
)
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


class TestPinpointSSLJEPA(unittest.TestCase):
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
            extract_jepa_item(
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

    def test_01_seal_guard_param_budget_and_layer_lr_decay(self) -> None:
        """SealAuditGuard blocks sealed/Stage-1 paths, param count in 7.4M-7.8M target, and xi=0.78 LR decay."""
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

        model = PinpointSSLJEPA(self.geometry, {'dim': 176, 'heads': 8}).to(self.device)
        self.assertIsInstance(model, NextGenFlavourModel)
        n_online = model.online_finetune_param_count()
        n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f'[PinpointSSLJEPA] online+finetune params = {n_online:,} ({n_online / 1e6:.3f}M) | '
            f'with_ssl_heads = {n_total:,} ({n_total / 1e6:.3f}M)',
            flush=True,
        )
        self.assertGreaterEqual(n_online, 7_300_000)
        self.assertLessEqual(n_online, 7_900_000)

        pg = model.build_param_groups(base_lr=1.0e-3, weight_decay=0.03, xi=0.78, include_ssl_heads=False)
        lrs = sorted({round(g['lr'], 10) for g in pg})
        expected_lrs = sorted([round(1.0e-3 * (0.78 ** (4 - lvl)), 10) for lvl in range(5)])
        self.assertEqual(lrs, expected_lrs)

    def test_02_nomae_anti_shortcut_bit_identical(self) -> None:
        """NOMAE/Sonata anti-shortcut: corrupting masked-plane (x,y) by +999mm leaves student/predictor bit-identical (0.0)."""
        prev_mm = torch.backends.cuda.matmul.allow_tf32
        prev_cd = torch.backends.cudnn.allow_tf32
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        try:
            torch.manual_seed(42)
            model = PinpointSSLJEPA(self.geometry, {'dim': 176, 'heads': 8, 'drop_path': 0.0}).to(self.device).eval()

            gen = torch.Generator(device=self.device).manual_seed(123)
            masks = sample_ssl_masks(self.b8_dev, generator=gen, shortcut_point_mask=False, n_st=model.backbone.n_st)
            p_drop = masks['pix_plane_drop_mask']
            s_drop = masks['stereo_panel_drop_mask']
            sp_drop = masks['spec_drop_mask']
            assert p_drop is not None and s_drop is not None and sp_drop is not None
            self.assertTrue(p_drop.any(), 'Synthetic batch should have at least one masked active plane')

            with torch.inference_mode():
                out_ref = model.forward_ssl_student(
                    self.b8_dev,
                    pix_plane_drop_mask=p_drop,
                    stereo_panel_drop_mask=s_drop,
                    spec_drop_mask=sp_drop,
                )

            # Corrupt ALL masked-plane coordinates, features, summaries, patches, tracklets, and c_obs
            b_mut = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in self.b8_dev.items()}
            cl_plane = b_mut['pix_plane'].long().clamp(0, 14)
            is_masked_cl = torch.gather(p_drop, 1, cl_plane) & b_mut['pix_mask'].bool()
            self.assertTrue(is_masked_cl.any())

            b_mut['pix_xyz_mm'][is_masked_cl] = b_mut['pix_xyz_mm'][is_masked_cl] + b_mut['pix_xyz_mm'].new_tensor([999.0, -999.0, 0.0])
            b_mut['pix_feat'][is_masked_cl] = 999.0
            b_mut['pix_plane_summary'][p_drop] = 999.0
            if len(b_mut['pix_patches']):
                patch_masked = is_masked_cl[b_mut['patch_ev'], b_mut['patch_cl']]
                if patch_masked.any():
                    b_mut['pix_patches'][patch_masked] = 999.0
            b_mut['trk_kin'] = b_mut['trk_kin'] + 999.0
            b_mut['trk_morph'] = b_mut['trk_morph'] + 999.0
            b_mut['c_obs'] = b_mut['c_obs'] + 999.0

            # Also corrupt masked spectrometer inputs when spec_drop is True
            b_mut['trk_cand'][sp_drop] = 999.0
            b_mut['trk_hits'][sp_drop] = 999.0
            if 'trk_station_counts' in b_mut:
                b_mut['trk_station_counts'][sp_drop] = 999.0

            with torch.inference_mode():
                out_mut = model.forward_ssl_student(
                    b_mut,
                    pix_plane_drop_mask=p_drop,
                    stereo_panel_drop_mask=s_drop,
                    spec_drop_mask=sp_drop,
                )

            for k in ('z_stu_glob', 'z_pix_pred', 'gmm_raw', 'bar_pred', 'tof_pred', 'spec_pred'):
                diff = float((out_ref[k] - out_mut[k]).abs().max().item())
                self.assertEqual(diff, 0.0, msg=f'NOMAE anti-shortcut violated for {k}: diff={diff}')

            # Positive sanity check: corrupting an UNMASKED cluster must change student output
            b_unm = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in self.b8_dev.items()}
            is_unm_cl = (~torch.gather(p_drop, 1, cl_plane)) & b_unm['pix_mask'].bool()
            b_unm['pix_feat'][is_unm_cl] += 2.0
            with torch.inference_mode():
                out_unm = model.forward_ssl_student(
                    b_unm,
                    pix_plane_drop_mask=p_drop,
                    stereo_panel_drop_mask=s_drop,
                    spec_drop_mask=sp_drop,
                )
            unm_diff = float((out_ref['z_stu_glob'] - out_unm['z_stu_glob']).abs().max().item())
            self.assertGreater(unm_diff, 1e-4)
        finally:
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = prev_mm
                torch.backends.cudnn.allow_tf32 = prev_cd

    def test_03_ssl_loss_vicreg_finite_grads_and_non_collapse(self) -> None:
        """VICReg regularizer penalizes collapse, SSL loss is finite with non-zero gradients across all SSL branches."""
        torch.manual_seed(42)
        model = PinpointSSLJEPA(self.geometry, {'dim': 176, 'heads': 8}).to(self.device).train()
        teacher = copy.deepcopy(model.backbone).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        # VICReg non-collapse test
        z_collapsed = (torch.ones((256, 64), device=self.device) * 0.5).requires_grad_(True)
        z_diverse = (torch.randn((256, 64), device=self.device) * 1.5).requires_grad_(True)
        l_col, st_col = vicreg_variance_covariance_loss(z_collapsed)
        l_div, st_div = vicreg_variance_covariance_loss(z_diverse)
        self.assertTrue(torch.isfinite(l_col) and torch.isfinite(l_div))
        self.assertGreater(st_col['var_loss'], 0.95)
        self.assertEqual(st_div['var_loss'], 0.0)
        self.assertGreater(float(l_col.item()), float(l_div.item()) + 10.0)
        self.assertLess(st_col['std_mean'], 0.05)
        self.assertGreater(st_div['std_mean'], 1.0)
        l_col.backward()
        self.assertTrue(torch.isfinite(z_collapsed.grad).all())

        # Full 3-task SSL forward + backward
        gen = torch.Generator(device=self.device).manual_seed(77)
        masks = sample_ssl_masks(self.b8_dev, generator=gen, shortcut_point_mask=False, n_st=model.backbone.n_st)
        model.zero_grad(set_to_none=True)
        ssl_loss, metrics = compute_ssl_pretrain_loss(model, teacher, self.b8_dev, masks)
        self.assertTrue(torch.isfinite(ssl_loss))
        for k, v in metrics.items():
            self.assertTrue(math.isfinite(v), msg=f'Non-finite SSL metric {k}={v}')
        ssl_loss.backward()

        # Verify backbone + ssl_predictors receive finite, non-zero gradients
        for prefix in ('backbone.', 'ssl_predictors.'):
            sq = 0.0
            for n, p in model.named_parameters():
                if n.startswith(prefix) and p.grad is not None:
                    self.assertTrue(torch.isfinite(p.grad).all(), msg=f'Non-finite grad in {n}')
                    sq += float(p.grad.detach().float().square().sum().item())
            self.assertGreater(math.sqrt(sq), 1e-6, msg=f'Zero grad in {prefix}')

    def test_04_gate_a_health_truth_tamper_bs1_perm_empty(self) -> None:
        """Gate A health, empty modalities, 39-key truth tamper == 0.0, bs=1 vs bs=8 <= 1e-6, permutation <= 1e-6."""
        prev_mm = torch.backends.cuda.matmul.allow_tf32
        prev_cd = torch.backends.cudnn.allow_tf32
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        try:
            torch.manual_seed(42)
            model = PinpointSSLJEPA(self.geometry, {'dim': 176, 'heads': 8}).to(self.device)

            def _loss_fn(_out: dict, b: dict) -> torch.Tensor:
                tot, _ = compute_jepa_finetune_loss(model, b)
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
                    self.assertTrue(torch.isfinite(getattr(out, fname)).all(), msg=f'{tag}.{fname} non-finite')

            # 39-key truth tamper == 0.0
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
                self.assertEqual(float((getattr(o_clean, fname) - getattr(o_mut, fname)).abs().max().item()), 0.0)

            # bs=1 vs bs=8 <= 1e-6
            b1_dev = move_batch_to_device(
                collate_dual_regime(self.items32[:1], mirror_rng=None, y_scint_center=self.yc, n_bars=self.nb),
                self.device,
            )
            with torch.inference_mode():
                p_b8 = model.forward_NextGen(self.b8_dev).probs4[:1]
                p_b1 = model.forward_NextGen(b1_dev).probs4
            self.assertLessEqual(float((p_b8 - p_b1).abs().max().item()), 1e-6)

            # Hit permutation invariance <= 1e-6
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
            it_p0 = extract_jepa_item(
                0, ob_perm0, 0,
                compute_observable_containment(ob_perm0, self.geometry, self.bounds, BASE_OBS_CFG),
                self.geometry, self.bounds, BASE_OBS_CFG,
            )
            bp = move_batch_to_device(collate_dual_regime([it_p0], mirror_rng=None, y_scint_center=self.yc, n_bars=self.nb), self.device)
            with torch.inference_mode():
                o_perm = model.forward_NextGen(bp)
            self.assertLessEqual(float((o_perm.probs4 - p_b1).abs().max().item()), 1e-6)

            # Y-mirror test item check
            mir_items = [apply_y_mirror_dual_regime_item(it, y_scint_center=self.yc, n_bars=self.nb) for it in self.items32[:4]]
            self.assertEqual(mir_items[1]['label_7'], 3 - self.items32[1]['label_7'])
        finally:
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = prev_mm
                torch.backends.cudnn.allow_tf32 = prev_cd

    def test_05_grad_accum_and_gate_b_32_overfit(self) -> None:
        """CPU FP32 grad_accum relative L2 error < 1e-4 and Gate B 32/32 overfit < 0.05."""
        items = self.items32[:8]
        b8 = collate_dual_regime(items, mirror_rng=None, y_scint_center=self.yc, n_bars=self.nb)
        mb0 = collate_dual_regime(items[:4], mirror_rng=None, y_scint_center=self.yc, n_bars=self.nb)
        mb1 = collate_dual_regime(items[4:8], mirror_rng=None, y_scint_center=self.yc, n_bars=self.nb)

        cfg = {'dim': 176, 'heads': 8, 'drop_path': 0.0, 'head_dropout': 0.0}
        torch.manual_seed(99)
        m_full = PinpointSSLJEPA(self.geometry, cfg).cpu().train()
        torch.manual_seed(99)
        m_acc = PinpointSSLJEPA(self.geometry, cfg).cpu().train()
        for mdl in (m_full, m_acc):
            for mod in mdl.modules():
                if isinstance(mod, torch.nn.Dropout):
                    mod.p = 0.0
                if hasattr(mod, 'drop_prob'):
                    mod.drop_prob = 0.0

        m_full.zero_grad(set_to_none=True)
        lf, _ = compute_jepa_finetune_loss(m_full, b8)
        lf.backward()
        g_full = {n: p.grad.detach().float() for n, p in m_full.online_finetune_parameters() if p.grad is not None}

        m_acc.zero_grad(set_to_none=True)
        fd = compute_jepa_finetune_window_denominators([mb0, mb1])
        la_sum = 0.0
        for mb in (mb0, mb1):
            la, _ = compute_jepa_finetune_loss(m_acc, mb, fixed_denoms=fd)
            la.backward()
            la_sum += float(la.item())
        g_acc = {n: p.grad.detach().float() for n, p in m_acc.online_finetune_parameters() if p.grad is not None}

        gn_full = math.sqrt(sum(float(v.square().sum()) for v in g_full.values()))
        gn_diff = math.sqrt(sum(float((g_full[k] - g_acc[k]).square().sum()) for k in g_full))
        rel_diff = gn_diff / max(gn_full, 1e-8)
        top_diffs = sorted(
            ((float((g_full[k] - g_acc[k]).norm()), k) for k in g_full),
            reverse=True,
        )[:5]
        print(f'[PinpointSSLJEPA] CPU FP32 grad_accum rel_L2_diff={rel_diff:.3e} top={top_diffs[:3]}', flush=True)
        self.assertLess(rel_diff, 1e-4)
        self.assertLess(abs(float(lf.item()) - la_sum), 1e-5)

        # Gate B 32/32 overfit
        torch.manual_seed(42)
        m_ov = PinpointSSLJEPA(self.geometry, cfg).to(self.device)
        res = run_gate_b_overfit_32_check(
            m_ov,
            self.b32_dev,
            lambda out, b: F.nll_loss(torch.log(out['probs4'].clamp(min=1e-6)), b['label']),
            steps=75,
            lr=2.5e-3,
            min_acc=31.0 / 32.0,
            max_final_loss=0.05,
        )
        print(f"[PinpointSSLJEPA] Gate B: acc={res['final_acc']*32:.0f}/32, final_loss={res['final_loss']:.4f}", flush=True)
        self.assertTrue(res['passed'])
        self.assertGreaterEqual(res['final_acc'], 31.0 / 32.0)
        self.assertLess(res['final_loss'], 0.05)

    def test_06_csr_schema_and_flop_parity(self) -> None:
        """Verify 19d `pix_feat` CSR unpacking and FlopCounterMode SSL/Sup FLOP ratio."""
        from analysis.flavour.nextgen.run_l2_ssl_jepa import _csr_shard_event_to_item, _measure_step_flops

        sh = {
            'obs_domain_id': np.array([1], dtype=np.uint8),
            'c_obs': np.zeros((1, 18), dtype=np.float32),
            'cal_start': np.zeros((1, 4), dtype=np.float32),
            'pix_ptr': np.array([0, 4], dtype=np.int32),
            'pix_xyz_mm': np.array([[10.0, -5.0, 20.0], [10.1, -4.9, 50.0], [0.0, 0.0, 80.0], [-2.0, 1.0, 110.0]], dtype=np.float32),
            'pix_plane': np.array([0, 1, 2, 3], dtype=np.int8),
            'pix_feat_16': np.ones((4, 16), dtype=np.float16),
            'pix_plane_summary': np.zeros((1, 15, 12), dtype=np.float32),
            'pat_ptr': np.array([0, 4], dtype=np.int32),
            'pix_patches': np.ones((4, 7, 7), dtype=np.float16),
            'patch_cluster_idx': np.arange(4, dtype=np.int16),
            'bar_ptr': np.array([0, 2], dtype=np.int32),
            'bar_panel': np.array([0, 0], dtype=np.int16),
            'bar_horiz': np.array([0, 1], dtype=np.int8),
            'bar_idx': np.array([3, 4], dtype=np.int16),
            'bar_pe': np.array([120, 2], dtype=np.uint16),  # 2 < 2.5 threshold is filtered out
            'bar_hit_time_ns': np.array([12.5, 13.0], dtype=np.float16),
            'trk_cand': np.zeros((1, 8, 14), dtype=np.float32),
            'trk_cand_mask': np.zeros((1, 8), dtype=bool),
            'trk_station_counts': np.zeros((1, 3), dtype=np.float32),
            'hit_ptr': np.array([0, 1], dtype=np.int32),
            'trk_hits': np.zeros((1, 6), dtype=np.float32),
        }
        it = _csr_shard_event_to_item(sh, 0, float(self.bounds['x_pix_center']), self.yc, float(self.bounds['z_mid']))
        self.assertEqual(it['pix_feat'].shape, (4, 22))
        self.assertTrue(np.allclose(it['pix_feat'][:, 3:16], 1.0, atol=1e-3))
        self.assertTrue(np.allclose(it['pix_feat'][:, 19:22], 1.0, atol=1e-3))
        self.assertEqual(len(it['bar_val']), 1)

        model = PinpointSSLJEPA(self.geometry, {'dim': 176, 'heads': 8}).to(self.device)
        teacher = copy.deepcopy(model.backbone).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        finfo = _measure_step_flops(model, teacher, self.b8_dev, self.b8_dev)
        print(
            f"[PinpointSSLJEPA] FlopCounterMode: SSL={finfo['ssl_flops_per_sample']/1e9:.3f}G, "
            f"Sup={finfo['sup_flops_per_sample']/1e9:.3f}G, ratio={finfo['ssl_to_sup_ratio']:.3f}x",
            flush=True,
        )
        self.assertGreater(finfo['ssl_flops_per_sample'], 1e7)
        self.assertGreater(finfo['sup_flops_per_sample'], 1e7)
        self.assertGreater(finfo['ssl_to_sup_ratio'], 0.8)


if __name__ == '__main__':
    unittest.main()
