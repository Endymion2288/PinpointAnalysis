"""Unit tests for Phase V2 PinpointFormer representation, geometry equivariance, and isolation."""

import ast
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from analysis.flavour.v3.common import domains
from analysis.flavour.v3.common.datasets import V3TrainDataset
from analysis.flavour.v3.v2_pinpointformer.models import PinpointFormer, pf_branch_groups
from analysis.flavour.v3.v2_pinpointformer.pixel_tokenizer import (
    N_MAX_CLUSTERS,
    apply_y_mirror_pf_item,
    collate_pf,
    extract_pf_item,
)
from analysis.flavour.v3.v2_pinpointformer.stereo_bars import static_scint_stations
from analysis.flavour.v3.v2_pinpointformer.vertex_anchor import (
    compute_near_vertex_pair_features,
    compute_stage2_polar_features,
    select_near_vertex_clusters,
)


GEOMETRY_PATH = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s0_s1_2026-11-11/shards/10083/geometry.json')
DATA_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R2/data')
SIDECAR_ROOT = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/truth_sidecar')


class TestPinpointFormerV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.geometry = json.loads(GEOMETRY_PATH.read_text())
        cls.bounds = domains.detector_bounds(cls.geometry)
        cls.stations = static_scint_stations(cls.geometry, cls.bounds)

    def _synthetic_observation(self) -> dict:
        nx, ny = self.bounds['nx'], self.bounds['ny']
        yc = self.bounds['y_scint_center']
        col0, row0, pl0 = nx // 2 + 12, ny // 2 + 40, 1
        col1, row1, pl1 = nx // 2 + 15, ny // 2 + 45, 3
        hits_p0 = [
            (pl0, row0 + dr, col0 + dc)
            for dr, dc in ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2))
        ]
        hits_p1 = [
            (pl1, row1 + dr, col1 + dc)
            for dr, dc in ((0, 0), (1, 0), (1, 1), (2, 2))
        ]
        all_h = hits_p0 + hits_p1
        pcid = np.array([(p * ny + r) * nx + c for p, r, c in all_h], dtype=np.int64)
        pxyz = np.array(
            [
                [
                    self.geometry['pixel_Xpos'][c],
                    self.geometry['pixel_Ypos'][r],
                    self.geometry['pixel_Zpos'][p],
                ]
                for p, r, c in all_h
            ],
            dtype=np.float32,
        )
        # Shift y slightly so y -> 2*yc - y is exact wrt yc = -12.0 mm
        dy_shift = yc - self.bounds['y_pix_center']
        pxyz[:, 1] += np.float32(dy_shift)

        # Panel 0 is unpaired vertical X-panel; panels 6 (V) and 7 (H) form a stereo doublet
        vp0 = int(self.bounds['vert_panels'][0])
        hp_dbl = int(self.bounds['horiz_panels'][0])
        vp_dbl = int(self.bounds['vert_panels'][int(self.stations['doublet_stations'][0])])

        bp = np.array([vp0, vp_dbl, hp_dbl], dtype=np.int16)
        bh = np.array([0, 0, 1], dtype=np.int8)
        bi = np.array([15, 22, 11], dtype=np.int16)
        bb = np.zeros((3, 5), dtype=np.float32)
        for k, (p, h, b) in enumerate(zip(bp, bh, bi)):
            if h == 0:
                xc = self.geometry['scint_bar_Xcenter'][b]
                bb[k] = [xc - 5.25, xc + 5.25, -222.0, 198.0, self.geometry['scint_Zpos'][p]]
            else:
                ybar = self.geometry['scint_bar_Ycenter'][b]
                bb[k] = [-123.0, 277.0, ybar - 5.25, ybar + 5.25, self.geometry['scint_Zpos'][p]]

        return {
            'pixel': {'channel_id': pcid, 'xyz_mm': pxyz},
            'bar': {
                'channel_id': np.array([1, 2, 3], dtype=np.int64),
                'layer': bp.copy(),
                'panel': bp,
                'horizontal': bh,
                'bar': bi,
                'bounds_mm': bb,
                'edep_proxy_MeV': np.array([18.0, 45.0, 32.0], dtype=np.float32),
                'pe_proxy': np.array([250, 720, 510], dtype=np.int64),
            },
            'tracker_hypothesis': {
                'station': np.array([0, 1, 2], dtype=np.int8),
                'xy_mm': np.array([[5.0, yc + 8.0], [5.2, yc + 7.4], [5.4, yc + 9.0]], dtype=np.float32),
                'sigma_xy_mm': np.full((3, 2), 0.1, dtype=np.float32),
            },
        }

    def test_static_stations_and_unpaired_panels(self) -> None:
        self.assertEqual(self.stations['n_stations'], 54)
        self.assertEqual(self.stations['n_doublets'], 42)
        unpaired = np.flatnonzero(~self.stations['station_has_y'])
        self.assertEqual(len(unpaired), 12)

    def test_shard0_sidecar_and_patch_trigger(self) -> None:
        ds = V3TrainDataset(
            DATA_ROOT,
            sidecar_root=SIDECAR_ROOT,
            split='train',
            config={'cache_shards': 1, 'load_cluster_sidecar': True},
            item_transform=extract_pf_item,
        )
        try:
            checked = 0
            total_patches = 0
            for idx in range(40):
                it = ds[idx]
                self.assertEqual(it['pix_xyz_mm'].dtype, np.float32)
                self.assertLessEqual(len(it['pix_xyz_mm']), N_MAX_CLUSTERS)
                self.assertEqual(len(it['bar_panel']), len(it['sc_bar_rel']))
                total_patches += len(it['pix_patches'])
                if len(it['pix_first_cid']):
                    pos = np.searchsorted(it['sc_first_cid'], it['pix_first_cid'])
                    safe = np.minimum(pos, len(it['sc_first_cid']) - 1)
                    matched = (pos < len(it['sc_first_cid'])) & (it['sc_first_cid'][safe] == it['pix_first_cid'])
                    self.assertEqual(int(matched.sum()), len(it['pix_first_cid']))
                    checked += 1
                if checked >= 5:
                    break
            self.assertGreaterEqual(checked, 5)
            self.assertGreater(total_patches, 0)
        finally:
            ds.native.close()

    def test_y_mirror_and_stage2_polar_invariance(self) -> None:
        obs = self._synthetic_observation()
        dom = domains.compute_observable_containment(obs, self.geometry, self.bounds)
        item = extract_pf_item(0, obs, 1, dom, self.geometry, self.bounds, {'quantize_xy_mm': None})
        mir = apply_y_mirror_pf_item(
            item,
            y_scint_center=self.bounds['y_scint_center'],
            n_bars=self.bounds['n_bars'],
        )
        self.assertEqual(len(item['pix_patches']), 2)
        yc = self.bounds['y_scint_center']

        # Double mirror returns exact original values
        dbl = apply_y_mirror_pf_item(mir, y_scint_center=yc, n_bars=self.bounds['n_bars'])
        np.testing.assert_allclose(dbl['pix_xyz_mm'], item['pix_xyz_mm'], atol=1e-5)
        np.testing.assert_allclose(dbl['pix_feat'], item['pix_feat'], atol=1e-5)
        np.testing.assert_allclose(dbl['pix_patches'], item['pix_patches'], atol=1e-5)
        np.testing.assert_array_equal(dbl['bar_idx'], item['bar_idx'])

        # Stage-2 polar & 7-dim IP_perp/kink pair features are exact under Y-reflection
        xyz = torch.from_numpy(item['pix_xyz_mm']).unsqueeze(0)
        xyz_m = torch.from_numpy(mir['pix_xyz_mm']).unsqueeze(0)
        mxy = torch.ones((1, len(item['pix_xyz_mm']), 2), dtype=torch.float32)
        v0 = torch.tensor([[1.0, yc + 2.0, -2950.0]], dtype=torch.float32)
        v0_m = torch.tensor([[1.0, yc - 2.0, -2950.0]], dtype=torch.float32)

        p1 = compute_stage2_polar_features(xyz, mxy, v0)
        p2 = compute_stage2_polar_features(xyz_m, mxy, v0_m)
        np.testing.assert_allclose(p1[..., [0, 2, 3, 4, 6, 7]].numpy(), p2[..., [0, 2, 3, 4, 6, 7]].numpy(), atol=1e-5)
        np.testing.assert_allclose(p1[..., [1, 5]].numpy(), -p2[..., [1, 5]].numpy(), atol=1e-5)

        mask = torch.ones((1, len(item['pix_xyz_mm'])), dtype=torch.bool)
        x0 = torch.ones((1, len(item['pix_xyz_mm'])), dtype=torch.float32)
        pf1 = compute_near_vertex_pair_features(xyz, x0, mask, v0)
        pf2 = compute_near_vertex_pair_features(xyz_m, x0, mask, v0_m)
        np.testing.assert_allclose(pf1.numpy(), pf2.numpy(), atol=1e-5)

        # Unmeasured Y axis (measured_xy=[1, 0]) never leaks pseudo y=0 distance
        xonly_xyz = torch.tensor([[[10.0, 999.0, -2500.0]]], dtype=torch.float32)
        xonly_mxy = torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)
        po = compute_stage2_polar_features(xonly_xyz, xonly_mxy, v0)
        self.assertEqual(float(po[0, 0, 1]), 0.0)
        self.assertEqual(float(po[0, 0, 5]), 0.0)
        self.assertEqual(float(po[0, 0, 7]), 0.0)

    def test_quantize_1mm_symmetry_and_model_health(self) -> None:
        obs = self._synthetic_observation()
        dom = domains.compute_observable_containment(obs, self.geometry, self.bounds)
        item_nat = extract_pf_item(0, obs, 0, dom, self.geometry, self.bounds, {'quantize_xy_mm': None})
        item_q1m = extract_pf_item(1, obs, 2, dom, self.geometry, self.bounds, {'quantize_xy_mm': 1.0})
        self.assertGreaterEqual(len(item_nat['pix_xyz_mm']), len(item_q1m['pix_xyz_mm']))
        # Both 5-hit plane-1 and 4-hit plane-3 groups collapse to 1 unique 1x1mm cell each (2 hits total)
        self.assertAlmostEqual(float(item_q1m['c_obs'][13]), float(np.log1p(2.0) / 10.0), places=5)
        self.assertEqual(item_nat['pix_feat'].dtype, np.float32)

        # Plane-stratified selection prevents 40 clusters on plane 0 from crowding out planes 1 and 2
        px = torch.zeros((1, 42, 3), dtype=torch.float32)
        px[0, :40, 2] = -2980.0
        px[0, :40, 0] = torch.linspace(0.0, 2.0, 40)
        px[0, 40, 2] = -2950.0
        px[0, 41, 2] = -2920.0
        pmask = torch.ones((1, 42), dtype=torch.bool)
        ppl = torch.zeros((1, 42), dtype=torch.long)
        ppl[0, 40] = 1
        ppl[0, 41] = 2
        vidx, _ = select_near_vertex_clusters(
            px, pmask, torch.tensor([[0.0, 0.0, -2980.0]]), k_vtx=20, pix_plane=ppl
        )
        self.assertIn(40, vidx[0].tolist())
        self.assertIn(41, vidx[0].tolist())

        batch = collate_pf([item_nat, item_q1m], y_scint_center=self.bounds['y_scint_center'], n_bars=self.bounds['n_bars'])
        for anchored in (True, False):
            model = PinpointFormer(self.geometry, {'task': 'T4', 'dim': 128, 'heads': 8, 'use_vertex_anchor': anchored})
            names = [n for n, _ in model.named_parameters()]
            self.assertFalse(any('linear_readout' in n for n in names))
            out = model(batch, grad_blend=0.3, return_aux=True, return_entropy=True)
            self.assertEqual(out['logits'].shape, (2, 4))
            for ent in out['entropies']:
                self.assertTrue(torch.all(ent >= -1e-6))
                self.assertTrue(torch.all(ent <= 1.0 + 1e-5))
            loss = out['logits'].sum() + 0.01 * out['v_hat'].sum()
            loss.backward()
            groups = pf_branch_groups(model)
            for br, plist in groups.items():
                gnorm = sum(float(p.grad.pow(2).sum()) for _, p in plist if p.grad is not None)
                self.assertGreater(gnorm, 0.0, msg=f'zero grad in branch {br} (anchored={anchored})')

    def test_ast_zero_truth_or_sealed_literals(self) -> None:
        base = Path(__file__).resolve().parent.parent / 'src' / 'analysis' / 'flavour' / 'v3' / 'v2_pinpointformer'
        forbidden = {'truth_sidecar', 'cluster_sidecar', 'initX', 'initY', 'initZ', 'initE', 'Cal-A', 'Cut-B', 'Confirm200k', 'sealed'}
        for fname in ('pixel_tokenizer.py', 'stereo_bars.py', 'attention.py', 'vertex_anchor.py', 'models.py', 'evaluate.py'):
            tree = ast.parse((base / fname).read_text())
            strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
            leaks = [s for s in strings for f in forbidden if f in s]
            self.assertEqual(leaks, [], msg=f'Forbidden literals in {fname}: {leaks}')


if __name__ == '__main__':
    unittest.main()
