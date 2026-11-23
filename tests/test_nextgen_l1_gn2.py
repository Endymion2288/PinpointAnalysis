"""Gate A, Gate B, SealAuditGuard, 39-key + sidecar Invariance, and Grad-Accum Tests for TrackletIPFormerGN2."""

from __future__ import annotations

import copy
import math
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from analysis.flavour.nextgen.gn2_tracklet_ipformer import (
    GN2_ORIGIN_CLASSES,
    GN2_SIDECAR_BATCH_KEYS,
    TrackletIPFormerGN2,
    collate_tracklet_gn2,
    compute_gn2_window_denominators,
    derive_tracklet_gn2_targets,
    extract_soft_tracklets_with_indices,
    extract_tracklet_gn2_item,
    gn2_tracklet_ipformer_loss,
    match_tracklets_to_clusters_batch,
)
from analysis.flavour.nextgen.protocols import NextGenFlavourModel, NextGenModelOutput
from analysis.flavour.nextgen.seal_guard import (
    SealAuditGuard,
    SealViolationError,
    TRUTH_BATCH_KEYS,
    assert_unsealed_split,
    verify_inference_truth_invariance,
)
from analysis.flavour.nextgen.tracklet_ops import extract_soft_tracklets_single
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


def _attach_synthetic_sidecar_targets(items: list[dict]) -> list[dict]:
    """Attach deterministic event + synthetic cluster_sidecar targets for Gate A/B & tamper tests."""
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

        n_cl = len(d['pix_xyz_mm'])
        orig = np.full(n_cl, 4, dtype=np.int64)
        sv_gid = np.full(n_cl, -1, dtype=np.int64)
        tid = np.full(n_cl, -1, dtype=np.int64)
        if n_cl > 0:
            # Make ~35% clusters coherent single-track signals across planes so pos/neg pairs exist
            # Class 2 (tau_CC) gets origin=1 (tau_decay) with sv_group_id=1;
            # Class 0/1/3 get origin=0 (PV_primary, sv=0) + photon_conv_ee (3, sv=2) + charm (2, sv=3) + ghost (4)
            is_tgt = d['pix_plane'] < 12
            tgt_idx = np.flatnonzero(is_tgt)
            for k, ci in enumerate(tgt_idx):
                if k % 5 == 0:
                    orig[ci] = 1 if lbl == 2 else 0
                    sv_gid[ci] = 1 if lbl == 2 else 0
                    tid[ci] = 101
                elif k % 5 == 1:
                    orig[ci] = 2 if (i % 3 == 0) else 0
                    sv_gid[ci] = 2 if (i % 3 == 0) else 0
                    tid[ci] = 102
                elif k % 5 == 2:
                    orig[ci] = 3  # photon_conv_ee
                    sv_gid[ci] = 3
                    tid[ci] = 103
                elif k % 5 == 3:
                    orig[ci] = 0
                    sv_gid[ci] = 0
                    tid[ci] = 101 if lbl != 2 else 104
                else:
                    orig[ci] = 4  # ghost / had_sec
                    sv_gid[ci] = -1
                    tid[ci] = 200 + k
        d['cl_origin_label'] = orig
        d['cl_sv_group_id'] = sv_gid
        d['cl_dominant_track_id'] = tid
        out.append(d)
    return out


class TestTrackletIPFormerGN2(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(42)
        cls.geometry = build_minimal_geometry()
        cls.bounds = detector_bounds(cls.geometry)
        cls.plane_x0 = np.array(
            [cumulative_radiation_length(float(z), cls.geometry) for z in cls.bounds['pixel_Zpos']],
            dtype=np.float32,
        )
        cls.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        cls.yc = float(cls.bounds['y_scint_center'])
        cls.nb = int(cls.bounds['n_bars'])
        cls.obs32, cls.lbl32 = _build_32_event_4class_observations(cls.geometry, cls.bounds, seed=325)
        raw_items = [
            extract_tracklet_gn2_item(
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
        cls.items32 = _attach_synthetic_sidecar_targets(raw_items)
        cls.b8_cpu = collate_tracklet_gn2(cls.items32[:8], mirror_rng=None, y_scint_center=cls.yc, n_bars=cls.nb)
        cls.b8_dev = move_batch_to_device(cls.b8_cpu, cls.device)
        cls.b32_dev = move_batch_to_device(
            collate_tracklet_gn2(cls.items32, mirror_rng=None, y_scint_center=cls.yc, n_bars=cls.nb),
            cls.device,
        )

    def test_01_seal_audit_guard_and_tracklet_index_parity(self) -> None:
        """SealAuditGuard blocks sealed/Tune8k paths; tracklet_with_indices matches TrackletOps bit-exact."""
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

        # Bit-identity between extract_soft_tracklets_with_indices and extract_soft_tracklets_single
        pitch = (float(self.bounds['dx_mm']), float(self.bounds['dy_mm']))
        for it in self.items32[:8]:
            ref = extract_soft_tracklets_single(
                it['pix_xyz_mm'], it['pix_plane'], it['pix_feat'], self.plane_x0, pitch_xy_mm=pitch
            )
            got = extract_soft_tracklets_with_indices(
                it['pix_xyz_mm'], it['pix_plane'], it['pix_feat'], self.plane_x0, pitch_xy_mm=pitch
            )
            self.assertTrue(np.array_equal(ref['trk_mask'], got['trk_mask']))
            self.assertEqual(float(np.abs(ref['trk_kin'] - got['trk_kin']).max()), 0.0)
            self.assertEqual(float(np.abs(ref['trk_morph'] - got['trk_morph']).max()), 0.0)

        # Verify derive_tracklet_gn2_targets works both with precomputed trk_cl_idx and fallback matcher
        t_pre = derive_tracklet_gn2_targets(self.b8_dev)
        self.assertIsNotNone(t_pre)
        b_no_idx = {k: v for k, v in self.b8_dev.items() if k != 'trk_cl_idx'}
        t_fb = derive_tracklet_gn2_targets(b_no_idx)
        self.assertIsNotNone(t_fb)
        assert t_pre is not None and t_fb is not None
        self.assertTrue(t_pre['trk_gn2_valid'].any())
        self.assertTrue(t_pre['trk_covertex_valid'].any())
        # Check covertex target symmetry
        cov = t_pre['trk_covertex_target']
        self.assertEqual(float((cov - cov.transpose(1, 2)).abs().max().item()), 0.0)
        # Fallback matcher aligns closely with exact index
        cl_fb = match_tracklets_to_clusters_batch(
            self.b8_dev['trk_kin'], self.b8_dev['trk_morph'], self.b8_dev['trk_mask'],
            self.b8_dev['pix_xyz_mm'], self.b8_dev['pix_mask'],
        )
        m_v = self.b8_dev['trk_mask']
        agree = float((cl_fb[m_v][..., 0] == self.b8_dev['trk_cl_idx'][m_v][..., 0]).float().mean().item())
        self.assertGreaterEqual(agree, 0.95)

    def test_02_gate_a_health_all_variants_and_empty_modalities(self) -> None:
        """Gate A passes with 0 dead branches across gn2_full, 3 ablations, and l1_ctrl_no_gn2."""
        variants = (
            ('gn2_full', {}),
            ('l1_ctrl_no_gn2', {'enable_origin_purity': False, 'enable_covertex': False, 'enable_readout_gate': False}),
            ('abl_origin_purity_only', {'enable_covertex': False}),
            ('abl_covertex_only', {'enable_origin_purity': False}),
            ('abl_aux_loss_only_no_readout_gate', {'enable_readout_gate': False}),
        )
        for vname, vcfg in variants:
            with self.subTest(variant=vname):
                torch.manual_seed(42)
                model = TrackletIPFormerGN2(self.geometry, {'dim': 176, 'heads': 8, **vcfg}).to(self.device)
                self.assertIsInstance(model, NextGenFlavourModel)
                n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                self.assertGreater(n_params, 6_000_000)
                self.assertLess(n_params, 8_500_000)

                def _loss_fn(_out: dict, b: dict) -> torch.Tensor:
                    out_dc = model.forward_NextGen(b)
                    tot, _ = gn2_tracklet_ipformer_loss(out_dc, b)
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
                    if zp:
                        mb['trk_kin'] = torch.zeros_like(mb['trk_kin'])
                        mb['trk_morph'] = torch.zeros_like(mb['trk_morph'])
                        mb['trk_mask'] = torch.zeros_like(mb['trk_mask'])
                    with torch.inference_mode():
                        out = model.forward_NextGen(mb)
                    self.assertIsInstance(out, NextGenModelOutput)
                    for fname in ('logits7', 'probs4', 'charm_logit', 'v_pv_mm', 'scale_pv_mm', 'sv_disp_mm', 'kin_pred'):
                        t = getattr(out, fname)
                        self.assertTrue(torch.isfinite(t).all(), msg=f'{vname}.{tag}.{fname} non-finite')

    def test_03_invariance_truth_sidecar_bs1_permutation_and_symmetry(self) -> None:
        """39-key + cluster_sidecar tamper == 0.0; bs=1 vs bs=8 <= 1e-6; hit perm <= 1e-6; covtx symmetric."""
        prev_mm = torch.backends.cuda.matmul.allow_tf32
        prev_cd = torch.backends.cudnn.allow_tf32
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        try:
            torch.manual_seed(42)
            model = TrackletIPFormerGN2(self.geometry, {'dim': 176, 'heads': 8}).to(self.device)
            model.eval()

            # 1. 39-key truth invariance
            res_inv = verify_inference_truth_invariance(model, self.b8_dev, atol=0.0)
            self.assertEqual(res_inv['max_abs_diff'], 0.0)

            # 2. Full 39-key + cluster_sidecar tamper == 0.0 across ALL 7 outputs + GN2 aux predictions
            stripped_keys = set(TRUTH_BATCH_KEYS) | set(GN2_SIDECAR_BATCH_KEYS)
            clean_b = {k: v for k, v in self.b8_dev.items() if k not in stripped_keys}
            mut_b = copy.copy(self.b8_dev)
            bsz = int(clean_b['c_obs'].shape[0])
            n_cl = int(clean_b['pix_xyz_mm'].shape[1])
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
            mut_b['cl_origin_label'] = torch.full((bsz, n_cl), 1, dtype=torch.long, device=self.device)
            mut_b['cl_secondary_origin_label'] = torch.full((bsz, n_cl), 2, dtype=torch.long, device=self.device)
            mut_b['cl_sv_group_id'] = torch.full((bsz, n_cl), 99, dtype=torch.long, device=self.device)
            mut_b['cl_dominant_track_id'] = torch.full((bsz, n_cl), 8888, dtype=torch.long, device=self.device)
            mut_b['cl_second_track_id'] = torch.full((bsz, n_cl), 7777, dtype=torch.long, device=self.device)
            mut_b['cl_third_track_id'] = torch.full((bsz, n_cl), 6666, dtype=torch.long, device=self.device)
            mut_b['trk_cl_idx'] = torch.zeros((bsz, 64, 3), dtype=torch.long, device=self.device)

            with torch.inference_mode():
                o_clean = model.forward_NextGen(clean_b)
                o_mut = model.forward_NextGen(mut_b)
            for fname in ('logits7', 'probs4', 'charm_logit', 'v_pv_mm', 'scale_pv_mm', 'sv_disp_mm', 'kin_pred'):
                d = float((getattr(o_clean, fname) - getattr(o_mut, fname)).abs().max().item())
                self.assertEqual(d, 0.0, msg=f'{fname} 39-key+sidecar tamper diff={d}')
            for ak in ('tracklet_origin_logits', 'tracklet_purity_logit', 'pairwise_covertex_logits', 'hat_w_real', 'hat_p_disp_hf', 'hat_C_ab'):
                d = float((o_clean.aux[ak] - o_mut.aux[ak]).abs().max().item())
                self.assertEqual(d, 0.0, msg=f'aux.{ak} 39-key+sidecar tamper diff={d}')

            # Check covertex logits & predicted affinity symmetry
            cov_lg = o_clean.aux['pairwise_covertex_logits'].squeeze(-1)
            self.assertLessEqual(float((cov_lg - cov_lg.transpose(1, 2)).abs().max().item()), 1e-6)

            # 3. bs=1 vs bs=8 <= 1e-6
            b1_dev = move_batch_to_device(
                collate_tracklet_gn2(self.items32[:1], mirror_rng=None, y_scint_center=self.yc, n_bars=self.nb),
                self.device,
            )
            with torch.inference_mode():
                p_b8 = model.forward_NextGen(self.b8_dev).probs4[:1]
                p_b1 = model.forward_NextGen(b1_dev).probs4
            bs_diff = float((p_b8 - p_b1).abs().max().item())
            self.assertLessEqual(bs_diff, 1e-6)

            # 4. Hit permutation invariance <= 1e-6
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
            it_p0 = extract_tracklet_gn2_item(
                0, ob_perm0, 0, compute_observable_containment(ob_perm0, self.geometry, self.bounds, BASE_OBS_CFG),
                self.geometry, self.bounds, BASE_OBS_CFG,
            )
            it_p1 = extract_tracklet_gn2_item(
                1, ob_perm1, 1, compute_observable_containment(ob_perm1, self.geometry, self.bounds, BASE_OBS_CFG),
                self.geometry, self.bounds, BASE_OBS_CFG,
            )
            bp = move_batch_to_device(collate_tracklet_gn2([it_p0, it_p1], mirror_rng=None, y_scint_center=self.yc, n_bars=self.nb), self.device)
            b2 = move_batch_to_device(collate_tracklet_gn2(self.items32[:2], mirror_rng=None, y_scint_center=self.yc, n_bars=self.nb), self.device)
            with torch.inference_mode():
                o_perm = model.forward_NextGen(bp)
                o_orig = model.forward_NextGen(b2)
            perm_diff = float((o_perm.probs4 - o_orig.probs4).abs().max().item())
            self.assertLessEqual(perm_diff, 1e-6)
        finally:
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = prev_mm
                torch.backends.cudnn.allow_tf32 = prev_cd

    def test_04_fp32_grad_accum_equivalence(self) -> None:
        """FP32 gradient accumulation (bs=8 vs 2x4 microbatches) relative L2 error < 1e-4."""
        items = self.items32[:8]
        b8 = collate_tracklet_gn2(items, mirror_rng=None, y_scint_center=self.yc, n_bars=self.nb)
        mb0 = collate_tracklet_gn2(items[:4], mirror_rng=None, y_scint_center=self.yc, n_bars=self.nb)
        mb1 = collate_tracklet_gn2(items[4:8], mirror_rng=None, y_scint_center=self.yc, n_bars=self.nb)

        cfg = {'dim': 176, 'heads': 8, 'drop_path': 0.0, 'head_dropout': 0.0}
        torch.manual_seed(99)
        m_full = TrackletIPFormerGN2(self.geometry, cfg).cpu()
        torch.manual_seed(99)
        m_acc = TrackletIPFormerGN2(self.geometry, cfg).cpu()
        for mdl in (m_full, m_acc):
            mdl.train()
            for mod in mdl.modules():
                if isinstance(mod, torch.nn.Dropout):
                    mod.p = 0.0
                if hasattr(mod, 'drop_prob'):
                    mod.drop_prob = 0.0

        m_full.zero_grad(set_to_none=True)
        lf, _ = gn2_tracklet_ipformer_loss(m_full.forward_NextGen(b8), b8)
        lf.backward()
        g_full = {n: p.grad.detach().float() for n, p in m_full.named_parameters() if p.grad is not None}

        m_acc.zero_grad(set_to_none=True)
        fd = compute_gn2_window_denominators([mb0, mb1])
        la_sum = 0.0
        for mb in (mb0, mb1):
            la, _ = gn2_tracklet_ipformer_loss(m_acc.forward_NextGen(mb), mb, fixed_denoms=fd)
            la.backward()
            la_sum += float(la.item())
        g_acc = {n: p.grad.detach().float() for n, p in m_acc.named_parameters() if p.grad is not None}

        gn_full = math.sqrt(sum(float(v.square().sum()) for v in g_full.values()))
        gn_diff = math.sqrt(sum(float((g_full[k] - g_acc[k]).square().sum()) for k in g_full))
        rel_diff = gn_diff / max(gn_full, 1e-8)
        print(f'[TrackletIPFormerGN2] CPU FP32 grad_accum rel_L2_diff={rel_diff:.3e}', flush=True)
        self.assertLess(rel_diff, 1e-4)
        self.assertLess(abs(float(lf.item()) - la_sum), 1e-5)

    def test_05_gate_b_32_event_overfit_with_sidecar(self) -> None:
        """Gate B: 32/32 balanced 4-class overfit < 0.05 (both NLL-only and full GN2 sidecar multitask)."""
        torch.manual_seed(42)
        model = TrackletIPFormerGN2(
            self.geometry, {'dim': 176, 'heads': 8, 'drop_path': 0.0, 'head_dropout': 0.0}
        ).to(self.device)
        res = run_gate_b_overfit_32_check(
            model,
            self.b32_dev,
            lambda out, b: F.nll_loss(torch.log(out['probs4'].clamp(min=1e-6)), b['label']),
            steps=75,
            lr=2.0e-3,
            min_acc=31.0 / 32.0,
            max_final_loss=0.05,
            upstream_param_prefixes=('pix_in.', 'stereo_bars.'),
        )
        print(f"[TrackletIPFormerGN2] Gate B (NLL): acc={res['final_acc']*32:.0f}/32, final_loss={res['final_loss']:.4f}", flush=True)
        self.assertTrue(res['passed'])
        self.assertGreaterEqual(res['final_acc'], 31.0 / 32.0)
        self.assertLess(res['final_loss'], 0.05)

        # Also verify full multitask + GN2 sidecar targets converge cleanly on 32 events
        torch.manual_seed(42)
        m2 = TrackletIPFormerGN2(
            self.geometry, {'dim': 176, 'heads': 8, 'drop_path': 0.0, 'head_dropout': 0.0}
        ).to(self.device)
        opt = torch.optim.AdamW(m2.parameters(), lr=1.5e-3)
        m2.train()
        for _ in range(75):
            opt.zero_grad(set_to_none=True)
            out_dc = m2.forward_NextGen(self.b32_dev)
            tot, _ = gn2_tracklet_ipformer_loss(out_dc, self.b32_dev)
            tot.backward()
            torch.nn.utils.clip_grad_norm_(m2.parameters(), 5.0)
            opt.step()
        m2.eval()
        with torch.inference_mode():
            out_f = m2.forward_NextGen(self.b32_dev)
            nll_f = float(F.nll_loss(torch.log(out_f.probs4.clamp_min(1e-6)), self.b32_dev['label']).item())
            acc_f = float((out_f.probs4.argmax(dim=-1) == self.b32_dev['label']).float().mean().item())
        print(f'[TrackletIPFormerGN2] Gate B (+GN2 sidecar): acc={acc_f*32:.0f}/32, nll={nll_f:.4f}', flush=True)
        self.assertGreaterEqual(acc_f, 31.0 / 32.0)
        self.assertLess(nll_f, 0.05)


if __name__ == '__main__':
    unittest.main()
