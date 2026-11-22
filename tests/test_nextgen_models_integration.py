"""Integration tests across all unified PINPOINT next-generation architectures."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

import analysis.flavour.nextgen as nextgen_pkg
from analysis.flavour.nextgen import NextGenFlavourModel, NextGenModelOutput, student_t_vertex_loss
from analysis.flavour.nextgen.dual_regime_modalnet import (
    DualRegimeModalNet,
    collate_dual_regime,
    extract_dual_regime_item,
)
from analysis.flavour.nextgen.seal_guard import TRUTH_BATCH_KEYS, verify_inference_truth_invariance
from analysis.flavour.nextgen.tau_kinema_diffnet import TauKinemaDiffNet, compute_tau_kinema_loss
from analysis.flavour.nextgen.tracklet_ipformer import TrackletIPFormer, tracklet_ipformer_loss
from analysis.flavour.v3.common.domains import compute_observable_containment, detector_bounds
from tests.test_nextgen_model_health import _build_32_event_4class_observations
from tests.test_nextgen_seal_and_splits import build_minimal_geometry


def _zero_modalities(batch: dict, zero_pixels: bool, zero_bars: bool, zero_spec: bool) -> dict:
    """Return a shallow-cloned batch with selected detector modalities masked out."""
    out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    device = out['c_obs'].device
    if zero_pixels:
        out['pix_mask'] = torch.zeros_like(out['pix_mask'], dtype=torch.bool)
        out['pix_feat'] = torch.zeros_like(out['pix_feat'])
        out['pix_plane_summary'] = torch.zeros_like(out['pix_plane_summary'])
        out['pix_patches'] = torch.empty((0, 1, 32, 32), dtype=out['pix_patches'].dtype, device=device)
        out['patch_ev'] = torch.empty((0,), dtype=out['patch_ev'].dtype, device=device)
        out['patch_cl'] = torch.empty((0,), dtype=out['patch_cl'].dtype, device=device)
        if 'trk_mask' in out:
            out['trk_mask'] = torch.zeros_like(out['trk_mask'], dtype=torch.bool)
    if zero_bars:
        out['bar_ev'] = torch.empty((0,), dtype=torch.long, device=device)
        out['bar_panel'] = torch.empty((0,), dtype=torch.long, device=device)
        out['bar_horiz'] = torch.empty((0,), dtype=torch.long, device=device)
        out['bar_idx'] = torch.empty((0,), dtype=torch.long, device=device)
        out['bar_val'] = torch.empty((0,), dtype=torch.float32, device=device)
        out['bar_time_ns'] = torch.empty((0,), dtype=torch.float32, device=device)
        out['bar_hit_time_ns'] = torch.empty((0,), dtype=torch.float32, device=device)
    if zero_spec:
        out['trk_cand_mask'] = torch.zeros_like(out['trk_cand_mask'], dtype=torch.bool)
        out['trk_cand'] = torch.zeros_like(out['trk_cand'])
        out['trk_hit_mask'] = torch.zeros_like(out['trk_hit_mask'], dtype=torch.bool)
        out['trk_hits'] = torch.zeros_like(out['trk_hits'])
        if 'trk_station_counts' in out:
            out['trk_station_counts'] = torch.zeros_like(out['trk_station_counts'])
    return out


class TestNextgenModelsIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(42)
        cls.geometry = build_minimal_geometry()
        cls.bounds = detector_bounds(cls.geometry)
        cls.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        obs, labels = _build_32_event_4class_observations(cls.geometry, cls.bounds, seed=325)
        cfg = {'pe_only': True}
        items = []
        for idx in range(4):
            dom = compute_observable_containment(obs[idx], cls.geometry, cls.bounds, cfg)
            items.append(extract_dual_regime_item(idx, obs[idx], labels[idx], dom, cls.geometry, cls.bounds, cfg))
        batch = collate_dual_regime(
            items,
            y_scint_center=cls.bounds['y_scint_center'],
            n_bars=cls.bounds['n_bars'],
        )
        batch['bar_hit_time_ns'] = batch['bar_time_ns'].clone()
        bsz = len(items)
        batch['label_7'] = torch.tensor([0, 1, 5, 6], dtype=torch.long)
        batch['tau_branch_valid'] = torch.ones((bsz,), dtype=torch.long)
        batch['charm_flag'] = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        batch['charm_valid'] = torch.ones((bsz,), dtype=torch.long)
        batch['vertex_xyz_mm'] = torch.tensor(
            [[0.0, -12.0, -3050.0], [1.0, -11.0, -2900.0], [-0.5, -13.0, -2500.0], [0.0, -10.0, -2200.0]],
            dtype=torch.float32,
        )
        batch['vertex_valid'] = torch.ones((bsz,), dtype=torch.long)
        batch['vis_kinematics'] = torch.zeros((bsz, 4), dtype=torch.float32)
        batch['kin_valid'] = torch.ones((bsz,), dtype=torch.long)
        cls.batch = {
            k: (v.to(cls.device) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }

    def _build_models(self) -> list[tuple[str, NextGenFlavourModel]]:
        return [
            (
                'TrackletIPFormer',
                TrackletIPFormer(
                    self.geometry,
                    {'dim': 32, 'heads': 2, 'n_rel_pre': 1, 'n_rel_post': 1, 'n_global_layers': 1, 'drop_path': 0.0},
                ).to(self.device),
            ),
            (
                'DualRegimeModalNet',
                DualRegimeModalNet(
                    self.geometry,
                    {'dim': 32, 'heads': 2, 'drop_path': 0.0, 'enable_pixdark_noise_aug': False},
                ).to(self.device),
            ),
            (
                'TauKinemaDiffNet',
                TauKinemaDiffNet(
                    self.geometry,
                    {'dim': 32, 'heads': 2, 'n_global_layers': 1, 'drop_path': 0.0},
                ).to(self.device),
            ),
        ]

    def test_imports_and_contract_types(self) -> None:
        """All public nextgen symbols and model classes import cleanly."""
        self.assertTrue(hasattr(nextgen_pkg, 'NextGenFlavourModel'))
        self.assertTrue(hasattr(nextgen_pkg, 'NextGenModelOutput'))
        for name, model in self._build_models():
            self.assertIsInstance(model, NextGenFlavourModel, msg=name)

    def test_forward_backward_shapes_empty_modalities_and_truth_invariance(self) -> None:
        """All 3 nextgen models pass shape, backward, empty-modality, and zero-truth-leakage contracts."""
        bsz = int(self.batch['c_obs'].shape[0])
        for name, model in self._build_models():
            with self.subTest(model=name):
                model.eval()
                with torch.no_grad():
                    dc_out = model(self.batch, return_dataclass=True)
                self.assertIsInstance(dc_out, NextGenModelOutput)
                self.assertEqual(tuple(dc_out.logits7.shape), (bsz, 7))
                self.assertEqual(tuple(dc_out.probs4.shape), (bsz, 4))
                self.assertEqual(tuple(dc_out.charm_logit.shape), (bsz,))
                self.assertEqual(tuple(dc_out.v_pv_mm.shape), (bsz, 3))
                self.assertEqual(tuple(dc_out.scale_pv_mm.shape), (bsz, 3))
                self.assertEqual(tuple(dc_out.kin_pred.shape), (bsz, 4))
                self.assertTrue((dc_out.scale_pv_mm > 0.0).all())
                self.assertTrue(torch.allclose(dc_out.probs4.sum(dim=-1), torch.ones(bsz, device=self.device), atol=1e-5))

                # Verify zero truth leakage across all 7 standardized output tensors.
                verify_inference_truth_invariance(model, self.batch, atol=1e-7)
                clean_batch = {k: v for k, v in self.batch.items() if k not in TRUTH_BATCH_KEYS}
                mutated = dict(self.batch)
                mutated['trackID'] = torch.full((bsz, 8), 777777, dtype=torch.long, device=self.device)
                mutated['vertex_xyz_mm'] = torch.full((bsz, 3), -4321.0, dtype=torch.float32, device=self.device)
                mutated['vis_kinematics'] = torch.full((bsz, 4), 123.4, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    out_clean = model(clean_batch, return_dataclass=True)
                    out_mut = model(mutated, return_dataclass=True)
                for field_name in ('logits7', 'probs4', 'charm_logit', 'v_pv_mm', 'scale_pv_mm', 'kin_pred'):
                    diff = float((getattr(out_clean, field_name) - getattr(out_mut, field_name)).abs().max().item())
                    self.assertLessEqual(diff, 1e-7, msg=f'{name}.{field_name} leaked truth: {diff:.3e}')
                if out_clean.sv_disp_mm is not None and out_mut.sv_disp_mm is not None:
                    sv_diff = float((out_clean.sv_disp_mm - out_mut.sv_disp_mm).abs().max().item())
                    self.assertLessEqual(sv_diff, 1e-7, msg=f'{name}.sv_disp_mm leaked truth: {sv_diff:.3e}')

                # Verify finite outputs under empty pixel, bar, and spectrometer modalities.
                for tag, zp, zb, zs in (
                    ('zero_pixels', True, False, False),
                    ('zero_bars', False, True, False),
                    ('zero_spec', False, False, True),
                    ('all_empty', True, True, True),
                ):
                    masked_batch = _zero_modalities(self.batch, zp, zb, zs)
                    with torch.no_grad():
                        m_out = model(masked_batch, return_dataclass=True)
                    for field_name in ('logits7', 'probs4', 'charm_logit', 'v_pv_mm', 'scale_pv_mm', 'kin_pred'):
                        self.assertTrue(
                            torch.isfinite(getattr(m_out, field_name)).all(),
                            msg=f'{name}.{field_name} non-finite under {tag}',
                        )

                # Verify training backward pass & non-zero gradients.
                model.train()
                model.zero_grad(set_to_none=True)
                if isinstance(model, TrackletIPFormer):
                    loss, _ = tracklet_ipformer_loss(model(self.batch), self.batch)
                elif isinstance(model, TauKinemaDiffNet):
                    loss, _ = compute_tau_kinema_loss(model, model(self.batch, return_dataclass=True), self.batch)
                else:
                    dc_tr = model(self.batch, return_dataclass=True)
                    loss = (
                        F.nll_loss(torch.log(dc_tr.probs4.clamp_min(1e-7)), self.batch['label'])
                        + 0.25 * F.binary_cross_entropy_with_logits(dc_tr.charm_logit, self.batch['charm_flag'].float())
                        + 0.20 * student_t_vertex_loss(dc_tr.v_pv_mm, dc_tr.scale_pv_mm, self.batch['vertex_xyz_mm'], self.batch['vertex_valid'])
                        + 0.10 * F.smooth_l1_loss(dc_tr.kin_pred, self.batch['vis_kinematics'])
                    )
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                grad_norms = [
                    float(p.grad.detach().norm().item())
                    for p in model.parameters()
                    if p.requires_grad and p.grad is not None
                ]
                self.assertGreater(len(grad_norms), 0, msg=f'{name} produced zero parameter grads')
                self.assertGreater(sum(g > 0.0 for g in grad_norms), 0, msg=f'{name} all grads zero')


if __name__ == '__main__':
    unittest.main()
