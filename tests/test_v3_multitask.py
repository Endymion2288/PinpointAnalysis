"""Unit tests for Phase V3 multitask losses, GN2 dense heads, Y-mirror labels, and eval isolation."""

import ast
import inspect
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from analysis.flavour.v3.common import datasets, domains
from analysis.flavour.v3.v2_pinpointformer.pixel_tokenizer import (
    apply_y_mirror_pf_item,
    collate_pf,
    extract_pf_item,
)
from analysis.flavour.v3.v2_pinpointformer.vertex_anchor import heteroscedastic_vertex_loss
from analysis.flavour.v3.v3_multitask.losses import (
    HomoscedasticEventLoss,
    marginal_four_from_seven,
    marginal_tau_branch_ce,
)
from analysis.flavour.v3.v3_multitask.models import PinpointFormerMultitask, v3_branch_groups


GEOMETRY_PATH = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s0_s1_2026-11-11/shards/10083/geometry.json')


class TestMultitaskV3(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.geometry = json.loads(GEOMETRY_PATH.read_text())
        cls.bounds = domains.detector_bounds(cls.geometry)

    def test_v3_eval_dataset_untouched(self) -> None:
        src = inspect.getsource(datasets.V3EvalDataset)
        tree = ast.parse(src)
        strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        self.assertNotIn('sidecar_root', inspect.signature(datasets.V3EvalDataset.__init__).parameters)
        for tok in ('truth_sidecar', 'cluster_sidecar', 'event_sidecar', 'vis_kinematics', 'charm', 'Cal-A', 'Cut-B', 'Confirm200k'):
            self.assertFalse(any(tok in s for s in strings), msg=f'V3EvalDataset touched forbidden token {tok}')
        with self.assertRaises(ValueError):
            datasets.V3TrainDataset(
                '/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R2/data',
                None,
                split='validation',
            )

    def test_y_mirror_label7_and_vis_kinematics(self) -> None:
        obs = {
            'pixel': {
                'channel_id': np.array([10, 11, 12], dtype=np.int64),
                'xyz_mm': np.array([[0.0, -10.0, -3000.0], [0.02, -10.0, -3000.0], [0.04, -9.98, -3000.0]], dtype=np.float32),
            },
            'bar': {
                'channel_id': np.array([1], dtype=np.int64),
                'layer': np.array([6], dtype=np.int16),
                'panel': np.array([6], dtype=np.int16),
                'horizontal': np.array([1], dtype=np.int8),
                'bar': np.array([5], dtype=np.int16),
                'bounds_mm': np.array([[-123.0, 277.0, -170.0, -159.5, -2800.0]], dtype=np.float32),
                'edep_proxy_MeV': np.array([15.0], dtype=np.float32),
                'pe_proxy': np.array([300], dtype=np.int64),
            },
            'tracker_hypothesis': {
                'station': np.empty(0, dtype=np.int8),
                'xy_mm': np.empty((0, 2), dtype=np.float32),
                'sigma_xy_mm': np.empty((0, 2), dtype=np.float32),
            },
        }
        dom = domains.compute_observable_containment(obs, self.geometry, self.bounds)
        item = extract_pf_item(0, obs, 1, dom, self.geometry, self.bounds, {})
        item['label_7'] = 1  # mu_minus_CC -> flips to 2 (mu_plus_CC) under Y-mirror
        item['vertex_xyz_mm'] = np.array([2.0, -8.0, -2980.0], dtype=np.float32)
        item['vertex_valid'] = 1
        item['vis_kinematics'] = np.array([4.0, 1.2, -0.5, 0.866], dtype=np.float32)
        item['kin_valid'] = 1

        yc = self.bounds['y_scint_center']
        mir = apply_y_mirror_pf_item(item, y_scint_center=yc)
        self.assertEqual(mir['label_7'], 2)
        self.assertAlmostEqual(float(mir['vertex_xyz_mm'][1]), 2.0 * yc - (-8.0), places=5)
        self.assertAlmostEqual(float(mir['vis_kinematics'][3]), -0.866, places=5)
        self.assertAlmostEqual(float(mir['vis_kinematics'][2]), -0.5, places=5)

    def test_marginal_tau_loss_and_heteroscedastic_positivity(self) -> None:
        logits7 = torch.tensor([[0.2, 0.3, 0.1, 1.0, 0.5, 0.2, -0.4], [0.2, 0.3, 0.1, 1.0, 0.5, 0.2, -0.4]], dtype=torch.float32)
        label7 = torch.tensor([0, -1], dtype=torch.long)
        label4 = torch.tensor([0, 2], dtype=torch.long)
        loss = marginal_tau_branch_ce(logits7, label7, label4)
        p7 = torch.softmax(logits7, dim=-1)
        expected = 0.5 * (-torch.log(p7[0, 0]) - torch.log(p7[1, 3] + p7[1, 4] + p7[1, 5]))
        self.assertAlmostEqual(float(loss), float(expected), places=5)

        p4 = marginal_four_from_seven(logits7)
        np.testing.assert_allclose(p4.sum(dim=-1).numpy(), np.ones(2), atol=1e-6)

        # Even at exact zero vertex residual and extreme negative log_var_v, loss stays positive & bounded
        lv = torch.full((2, 3), -4.0)
        v = torch.zeros((2, 3))
        vtx_loss = heteroscedastic_vertex_loss(v, lv, v, torch.ones(2, dtype=torch.long))
        self.assertGreater(float(vtx_loss), 0.5)
        vtx_loss_legacy = heteroscedastic_vertex_loss(v, lv, v, torch.ones(2, dtype=torch.long), legacy_biased=True)
        self.assertGreater(float(vtx_loss_legacy), 10.0)

        # Zero-valid masks do not produce NaN/Inf
        hl = HomoscedasticEventLoss()
        b_zero = {
            'label': torch.tensor([0, 1]),
            'label_7': torch.tensor([-1, -1]),
            'charm_flag': torch.tensor([0, 1]),
            'charm_valid': torch.tensor([0, 0]),
            'vertex_xyz_mm': torch.zeros((2, 3)),
            'vertex_valid': torch.tensor([0, 0]),
            'vis_kinematics': torch.zeros((2, 4)),
            'kin_valid': torch.tensor([0, 0]),
        }
        tot, _ = hl(logits7, torch.zeros(2), torch.zeros((2, 3)), torch.zeros((2, 3)), torch.zeros((2, 4)), b_zero)
        self.assertTrue(torch.isfinite(tot))

    def test_v3_dense_gn2_and_event4head_forward_backward(self) -> None:
        nx = self.bounds['nx']
        stride_p = nx * self.bounds['ny']
        c0, r0 = 100, 100
        c1, r1 = 105, 105
        obs = {
            'pixel': {
                'channel_id': np.array([
                    r0 * nx + c0, r0 * nx + c0 + 1, (r0 + 1) * nx + c0 + 2,
                    stride_p + r1 * nx + c1, stride_p + r1 * nx + c1 + 1, stride_p + (r1 + 1) * nx + c1 + 2,
                ], dtype=np.int64),
                'xyz_mm': np.array([
                    [1.0, -10.0, self.geometry['pixel_Zpos'][0]],
                    [1.02, -10.0, self.geometry['pixel_Zpos'][0]],
                    [1.045, -9.97, self.geometry['pixel_Zpos'][0]],
                    [1.1, -9.8, self.geometry['pixel_Zpos'][1]],
                    [1.12, -9.8, self.geometry['pixel_Zpos'][1]],
                    [1.145, -9.77, self.geometry['pixel_Zpos'][1]],
                ], dtype=np.float32),
            },
            'bar': {
                'channel_id': np.array([1, 2], dtype=np.int64),
                'layer': np.array([6, 7], dtype=np.int16),
                'panel': np.array([6, 7], dtype=np.int16),
                'horizontal': np.array([0, 1], dtype=np.int8),
                'bar': np.array([20, 15], dtype=np.int16),
                'bounds_mm': np.array([[-5.0, 5.0, -220.0, 198.0, -2800.0], [-120.0, 270.0, -17.0, -6.5, -2790.0]], dtype=np.float32),
                'edep_proxy_MeV': np.array([25.0, 30.0], dtype=np.float32),
                'pe_proxy': np.array([400, 480], dtype=np.int64),
            },
            'tracker_hypothesis': {
                'station': np.array([0, 1, 2], dtype=np.int8),
                'xy_mm': np.array([[2.0, -10.0], [2.1, -10.5], [2.2, -9.5]], dtype=np.float32),
                'sigma_xy_mm': np.full((3, 2), 0.1, dtype=np.float32),
            },
        }
        dom = domains.compute_observable_containment(obs, self.geometry, self.bounds)
        items = []
        for idx, (l4, l7) in enumerate(((2, -1), (0, 0))):
            it = extract_pf_item(idx, obs, l4, dom, self.geometry, self.bounds, {})
            n_cl = len(it['pix_first_cid'])
            it['label_7'] = l7
            it['tau_branch_valid'] = 1 if l7 >= 0 else 0
            it['charm_flag'] = idx
            it['charm_valid'] = 1
            it['vertex_xyz_mm'] = np.array([0.0, -12.0, -2980.0], dtype=np.float32)
            it['vertex_valid'] = 1
            it['vis_kinematics'] = np.array([4.0, 1.0, 0.1, 0.2], dtype=np.float32)
            it['kin_valid'] = 1
            order = np.argsort(it['pix_first_cid'])
            it['sc_first_cid'] = it['pix_first_cid'][order]
            it['sc_origin'] = np.array([0, 2][:n_cl], dtype=np.int8)
            it['sc_ancestry'] = np.array([[10, 0], [20, 1]][:n_cl], dtype=np.int32)
            it['sc_pix_rel'] = np.array([[0.7, 0.1, 0.2], [0.1, 0.1, 0.8]][:n_cl], dtype=np.float16)
            it['sc_bar_rel'] = np.array([[0.8, 0.0, 0.2], [0.7, 0.0, 0.3]], dtype=np.float16)
            items.append(it)
        batch = collate_pf(items, y_scint_center=self.bounds['y_scint_center'], n_bars=self.bounds['n_bars'])

        for dense_on in (False, True):
            model = PinpointFormerMultitask(self.geometry, {'task': 'T4', 'dim': 128, 'heads': 8, 'enable_dense_gn2': dense_on})
            out = model(batch, grad_blend=0.3, return_aux=True, return_entropy=True)
            self.assertEqual(out['logits'].shape, (2, 7))
            self.assertEqual(out['probs4'].shape, (2, 4))
            self.assertEqual(out['dense']['topo_summary'].shape, (2, 8))
            loss, _ = model.event_loss(out['logits'], out['charm_logit'], out['v_hat'], out['log_var_v'], out['kin_pred'], batch)
            if dense_on:
                dl = model.dense_heads.compute_losses(out['dense'], out['aux'], batch)
                loss = loss + 0.35 * dl['l_rel'] + 0.50 * dl['l_origin'] + 0.30 * dl['l_compat']
            loss.backward()
            for br, plist in v3_branch_groups(model).items():
                gnorm = sum(float(p.grad.pow(2).sum()) for _, p in plist if p.grad is not None)
                self.assertGreater(gnorm, 0.0, msg=f'zero grad in V3 branch {br} (dense={dense_on})')

    def test_ast_no_truth_leak_in_v3_modules(self) -> None:
        base = Path(__file__).resolve().parent.parent / 'src' / 'analysis' / 'flavour' / 'v3' / 'v3_multitask'
        forbidden = {'initX', 'initY', 'initZ', 'initE', 'Cal-A', 'Cut-B', 'Confirm200k', 'sealed'}
        for fname in ('dense_heads.py', 'models.py', 'evaluate.py'):
            tree = ast.parse((base / fname).read_text())
            strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
            leaks = [s for s in strings for f in forbidden if f in s]
            self.assertEqual(leaks, [], msg=f'Forbidden literals in {fname}: {leaks}')


if __name__ == '__main__':
    unittest.main()
