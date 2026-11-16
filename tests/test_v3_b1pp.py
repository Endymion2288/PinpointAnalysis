"""Unit tests for V1 ZAlignedStereoNet (B1++): geometry layout, mirror symmetry, modality isolation, and forward shape."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from analysis.flavour.v3.common.domains import compute_observable_containment, detector_bounds
from analysis.flavour.v3.v1_b1pp.models import ZAlignedStereoNet, b1pp_branch_groups
from analysis.flavour.v3.v1_b1pp.representation import (
    apply_y_mirror_to_item,
    collate_b1pp,
    extract_b1pp_item,
    static_plane_layout,
)
from analysis.flavour.v3.v1_b1pp.spectrometer import CAND_Y_FLIP_IDX, TRACKER_Z_MM, extract_spectrometer_kinematics


GEOM_PATH = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s0_s1_2026-11-11/shards/10083/geometry.json')
V1_CODE_DIR = Path('/eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v1_b1pp')


class TestV3B1PP(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.geometry = json.loads(GEOM_PATH.read_text())
        cls.bounds = detector_bounds(cls.geometry)
        cls.layout = static_plane_layout(cls.geometry, cls.bounds)

    def test_static_plane_layout_zero_holes(self) -> None:
        self.assertEqual(self.layout['n_pix'], 15)
        self.assertEqual(self.layout['n_xbar'], 54)
        self.assertEqual(self.layout['n_ybar'], 42)
        self.assertEqual(self.layout['n_seq'], 111)
        self.assertEqual(self.layout['n_bars'], 40)
        z = self.layout['seq_z_mm']
        self.assertTrue(np.all(np.diff(z) >= 0.0))

    def test_spectrometer_y_bending_sign_and_mirror(self) -> None:
        yc = float(self.bounds['y_scint_center'])
        xc = float(self.bounds['x_pix_center'])
        cal_start = (-2800.0, xc + 5.0, yc + 3.2, True)
        # Straight in x, curved in y with s_y = +1.8 mm (Station 1 shifted up by +1.8 mm)
        xy_pos = np.array([
            [xc + 5.1, yc + 4.0],
            [xc + 7.1, yc + 6.8],
            [xc + 9.1, yc + 6.0],
        ], dtype=np.float64)
        st = np.array([0, 1, 2], dtype=np.int64)
        res_pos = extract_spectrometer_kinematics({'xy_mm': xy_pos, 'station': st}, cal_start, self.bounds)
        self.assertTrue(res_pos['cand_mask'][0])
        self.assertFalse(res_pos['cand_mask'][1])
        self.assertGreater(float(res_pos['cand'][0, 7]), 0.0)

        # Mirror around yc and cal_start y
        xy_mir = xy_pos.copy()
        xy_mir[:, 1] = 2.0 * yc - xy_mir[:, 1]
        cal_mir = (-2800.0, xc + 5.0, 2.0 * yc - (yc + 3.2), True)
        res_mir = extract_spectrometer_kinematics({'xy_mm': xy_mir, 'station': st}, cal_mir, self.bounds)
        c_expected = res_pos['cand'].copy()
        c_expected[:, CAND_Y_FLIP_IDX] *= -1.0
        np.testing.assert_allclose(res_mir['cand'], c_expected, atol=1e-5)

    def test_two_station_spectrometer_fallback(self) -> None:
        yc = float(self.bounds['y_scint_center'])
        xc = float(self.bounds['x_pix_center'])
        cal_start = (-2800.0, xc + 5.0, yc + 2.0, True)
        xy_pair = np.array([[xc + 5.1, yc + 3.5], [xc + 7.4, yc + 6.1]], dtype=np.float64)
        st_pair = np.array([0, 1], dtype=np.int64)
        res = extract_spectrometer_kinematics({'xy_mm': xy_pair, 'station': st_pair}, cal_start, self.bounds)
        self.assertTrue(bool(res['cand_mask'][0]))
        self.assertFalse(bool(res['cand_mask'][1]))
        self.assertAlmostEqual(float(res['cand'][0, 0]), 0.5, places=6)

        xy_mir = xy_pair.copy()
        xy_mir[:, 1] = 2.0 * yc - xy_mir[:, 1]
        cal_mir = (-2800.0, xc + 5.0, 2.0 * yc - (yc + 2.0), True)
        res_mir = extract_spectrometer_kinematics({'xy_mm': xy_mir, 'station': st_pair}, cal_mir, self.bounds)
        c_exp = res['cand'].copy()
        c_exp[:, CAND_Y_FLIP_IDX] *= -1.0
        np.testing.assert_allclose(res_mir['cand'], c_exp, atol=1e-5)

    def test_exact_y_mirror_equivariance(self) -> None:
        yc = float(self.bounds['y_scint_center'])
        xc = float(self.bounds['x_pix_center'])
        nx, ny = self.bounds['nx'], self.bounds['ny']
        vp = int(self.bounds['vert_panels'][3])
        hp = int(self.bounds['horiz_panels'][4])

        # Pixel coordinates chosen strictly inside bin interiors
        px = np.array([xc + 2.3, xc + 2.5, xc - 18.7], dtype=np.float64)
        py = np.array([yc + 10.3, yc + 10.35, yc - 22.8], dtype=np.float64)
        pz = np.array([self.bounds['pixel_Zpos'][1], self.bounds['pixel_Zpos'][1], self.bounds['pixel_Zpos'][2]], dtype=np.float64)
        pcid = np.array([1 * nx * ny + 100, 1 * nx * ny + 101, 2 * nx * ny + 200], dtype=np.int64)

        by_edges = self.bounds['scint_bar_Ycenter']
        b_idx = np.array([5, 12], dtype=np.int64)
        obs = {
            'pixel': {'channel_id': pcid, 'xyz_mm': np.column_stack([px, py, pz])},
            'bar': {
                'panel': np.array([vp, hp], dtype=np.int64),
                'horizontal': np.array([0, 1], dtype=np.int64),
                'bar': b_idx,
                'edep_proxy_MeV': np.array([12.5, 48.0], dtype=np.float64),
                'pe_proxy': np.array([150.0, 620.0], dtype=np.float64),
                'bounds_mm': np.array([
                    [xc - 5.0, xc + 5.0, yc - 200.0, yc + 200.0, self.bounds['scint_Zpos'][vp]],
                    [xc - 150.0, xc + 250.0, by_edges[12] - 5.0, by_edges[12] + 5.0, self.bounds['scint_Zpos'][hp]],
                ], dtype=np.float64),
            },
            'tracker_hypothesis': {
                'xy_mm': np.array([[xc + 4.0, yc + 5.1], [xc + 6.0, yc + 8.3], [xc + 8.0, yc + 7.2]], dtype=np.float64),
                'station': np.array([0, 1, 2], dtype=np.int64),
            },
        }

        obs_mir = {
            'pixel': {
                'channel_id': pcid.copy(),
                'xyz_mm': np.column_stack([px, 2.0 * yc - py, pz]),
            },
            'bar': {
                'panel': obs['bar']['panel'].copy(),
                'horizontal': obs['bar']['horizontal'].copy(),
                'bar': np.array([5, 39 - 12], dtype=np.int64),
                'edep_proxy_MeV': obs['bar']['edep_proxy_MeV'].copy(),
                'pe_proxy': obs['bar']['pe_proxy'].copy(),
                'bounds_mm': np.array([
                    [xc - 5.0, xc + 5.0, yc - 200.0, yc + 200.0, self.bounds['scint_Zpos'][vp]],
                    [xc - 150.0, xc + 250.0, 2.0 * yc - (by_edges[12] + 5.0), 2.0 * yc - (by_edges[12] - 5.0), self.bounds['scint_Zpos'][hp]],
                ], dtype=np.float64),
            },
            'tracker_hypothesis': {
                'xy_mm': np.column_stack([obs['tracker_hypothesis']['xy_mm'][:, 0], 2.0 * yc - obs['tracker_hypothesis']['xy_mm'][:, 1]]),
                'station': obs['tracker_hypothesis']['station'].copy(),
            },
        }

        dom0 = compute_observable_containment(obs, self.geometry, self.bounds)
        dom1 = compute_observable_containment(obs_mir, self.geometry, self.bounds)
        item0 = extract_b1pp_item(0, obs, 2, dom0, self.geometry, self.bounds, {'task': 'T4'})
        item0_mir = apply_y_mirror_to_item(item0, n_bars=self.bounds['n_bars'])
        item1 = extract_b1pp_item(0, obs_mir, 2, dom1, self.geometry, self.bounds, {'task': 'T4'})

        self.assertEqual(item0_mir['obs_domain_id'], item1['obs_domain_id'])
        np.testing.assert_allclose(item0_mir['c_obs'], item1['c_obs'], atol=1e-5)
        for k in ('fx_idx', 'fy_idx', 'wx_idx', 'wy_idx', 'bx_idx', 'by_idx'):
            np.testing.assert_array_equal(item0_mir[k], item1[k])
        for k in ('fx_val', 'fy_val', 'wx_val', 'wy_val', 'bx_val', 'by_val'):
            np.testing.assert_allclose(item0_mir[k].astype(np.float32), item1[k].astype(np.float32), atol=1e-3)
        np.testing.assert_allclose(item0_mir['trk_cand'], item1['trk_cand'], atol=1e-4)
        np.testing.assert_allclose(item0_mir['trk_hits'], item1['trk_hits'], atol=1e-4)

    def test_empty_and_model_forward_budget(self) -> None:
        empty_obs = {
            'pixel': {'channel_id': np.empty(0, dtype=np.int64), 'xyz_mm': np.empty((0, 3), dtype=np.float64)},
            'bar': {
                'panel': np.empty(0, dtype=np.int64),
                'horizontal': np.empty(0, dtype=np.int64),
                'bar': np.empty(0, dtype=np.int64),
                'edep_proxy_MeV': np.empty(0, dtype=np.float64),
                'pe_proxy': np.empty(0, dtype=np.float64),
                'bounds_mm': np.empty((0, 5), dtype=np.float64),
            },
            'tracker_hypothesis': {'xy_mm': np.empty((0, 2), dtype=np.float64), 'station': np.empty(0, dtype=np.int64)},
        }
        dom = compute_observable_containment(empty_obs, self.geometry, self.bounds)
        item = extract_b1pp_item(0, empty_obs, 3, dom, self.geometry, self.bounds, {'task': 'T4'})
        batch = collate_b1pp([item, item])

        for task, k in (('T4', 4), ('T3', 3)):
            model = ZAlignedStereoNet(self.layout, {'task': task})
            n_params = sum(p.numel() for p in model.parameters())
            self.assertGreaterEqual(n_params, 2_000_000)
            self.assertLessEqual(n_params, 3_600_000)
            out = model(batch)
            self.assertEqual(tuple(out.shape), (2, k))
            self.assertTrue(torch.isfinite(out).all())
            groups = b1pp_branch_groups(model)
            self.assertEqual(
                set(groups.keys()),
                {'bar_transverse', 'pixel_transverse', 'longitudinal_mixer', 'window_readout', 'spectrometer', 'cobs', 'head'},
            )

    def test_drop_modality_zero_leakage(self) -> None:
        yc = float(self.bounds['y_scint_center'])
        xc = float(self.bounds['x_pix_center'])
        nx, ny = self.bounds['nx'], self.bounds['ny']
        vp = int(self.bounds['vert_panels'][3])
        hp = int(self.bounds['horiz_panels'][4])
        base_bar = {
            'panel': np.array([vp, hp], dtype=np.int64),
            'horizontal': np.array([0, 1], dtype=np.int64),
            'bar': np.array([5, 12], dtype=np.int64),
            'edep_proxy_MeV': np.array([12.5, 48.0], dtype=np.float64),
            'pe_proxy': np.array([150.0, 620.0], dtype=np.float64),
            'bounds_mm': np.array([
                [xc - 5.0, xc + 5.0, yc - 200.0, yc + 200.0, self.bounds['scint_Zpos'][vp]],
                [xc - 150.0, xc + 250.0, yc - 5.0, yc + 5.0, self.bounds['scint_Zpos'][hp]],
            ], dtype=np.float64),
        }
        obs_no_pix = {
            'pixel': {'channel_id': np.empty(0, dtype=np.int64), 'xyz_mm': np.empty((0, 3), dtype=np.float64)},
            'bar': base_bar,
            'tracker_hypothesis': {
                'xy_mm': np.array([[xc + 4.0, yc + 5.1], [xc + 6.0, yc + 8.3], [xc + 8.0, yc + 7.2]], dtype=np.float64),
                'station': np.array([0, 1, 2], dtype=np.int64),
            },
        }
        dom = compute_observable_containment(obs_no_pix, self.geometry, self.bounds)
        item = extract_b1pp_item(0, obs_no_pix, 0, dom, self.geometry, self.bounds, {'task': 'T3', 'drop_modality': 'pixel'})
        self.assertEqual(float(item['c_obs'][0]), 0.0)
        self.assertEqual(float(item['c_obs'][13]), 0.0)
        self.assertNotEqual(item['obs_domain_id'], 1)
        for k in ('fx_idx', 'fy_idx', 'wx_idx', 'wy_idx'):
            self.assertEqual(len(item[k]), 0)

        obs_no_trk = {
            'pixel': {
                'channel_id': np.array([1 * nx * ny + 10], dtype=np.int64),
                'xyz_mm': np.array([[xc + 1.0, yc + 1.0, self.bounds['pixel_Zpos'][1]]], dtype=np.float64),
            },
            'bar': base_bar,
            'tracker_hypothesis': {'xy_mm': np.empty((0, 2), dtype=np.float64), 'station': np.empty(0, dtype=np.int64)},
        }
        dom_t = compute_observable_containment(obs_no_trk, self.geometry, self.bounds)
        item_t = extract_b1pp_item(0, obs_no_trk, 1, dom_t, self.geometry, self.bounds, {'task': 'T3', 'drop_modality': 'tracker'})
        self.assertEqual(float(item_t['c_obs'][2]), 0.0)
        self.assertEqual(float(item_t['c_obs'][3]), 0.0)
        self.assertFalse(bool(item_t['trk_cand_mask'].any()))
        self.assertEqual(len(item_t['trk_hits']), 0)
        # Ensure apply_y_mirror_to_item handles dropped modalities cleanly
        self.assertEqual(len(apply_y_mirror_to_item(item_t)['trk_hits']), 0)

    def test_zero_truth_leakage_ast(self) -> None:
        for py_file in sorted(V1_CODE_DIR.glob('*.py')):
            src = py_file.read_text()
            for forbidden in ('truth_sidecar', 'pixel_clusters', 'uproot', 'event_sidecar', 'cluster_sidecar', 'Cal-A', 'Cut-B', 'Confirm200k'):
                self.assertNotIn(forbidden, src, f'{py_file.name} references forbidden token {forbidden}')
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    self.assertNotIn('truth_sidecar', node.module)
            if py_file.name != 'train.py':
                self.assertNotIn('V3TrainDataset', src)


if __name__ == '__main__':
    unittest.main()
