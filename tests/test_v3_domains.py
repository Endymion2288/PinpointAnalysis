"""Unit tests for W3-01 pure-observation containment domains and spectrum diagnostics."""

import ast
import inspect
import json
from pathlib import Path
import unittest

import numpy as np

from analysis.flavour.v3.common import domains
from analysis.flavour.v3.common.spectrum import compute_energy_match_weights


GEOMETRY_PATH = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s0_s1_2026-11-11/shards/10083/geometry.json')


class TestV3DomainsAndSpectrum(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.geometry = json.loads(GEOMETRY_PATH.read_text())
        cls.bounds = domains.detector_bounds(cls.geometry)

    def test_panel_partition_matches_hardware(self) -> None:
        orient = domains.derive_panel_orientations(self.geometry)
        self.assertEqual(len(orient), 96)
        self.assertEqual(int((orient == 0).sum()), 54)
        self.assertEqual(int((orient == 1).sum()), 42)
        self.assertEqual(len(self.bounds['vert_panels']), 54)
        self.assertEqual(len(self.bounds['horiz_panels']), 42)
        # Check radiation length monotonically increases along pixel planes
        x0s = [domains.cumulative_radiation_length(float(z), self.geometry) for z in self.bounds['pixel_Zpos']]
        self.assertTrue(all(b > a for a, b in zip(x0s[:-1], x0s[1:])))
        self.assertGreater(x0s[-1], 100.0)

    def test_no_truth_access_in_compute_observable_containment(self) -> None:
        src = inspect.getsource(domains.compute_observable_containment)
        tree = ast.parse(src)
        forbidden = {'initX', 'initY', 'initZ', 'initE', 'initPDG', 'fslPDG', 'pixel_centered', 'large_scintillator'}
        found_strings = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertEqual(found_strings & forbidden, set())
        sig = inspect.signature(domains.compute_observable_containment)
        self.assertNotIn('truth', sig.parameters)
        self.assertNotIn('sidecar', sig.parameters)

    def test_empty_and_contained_observations(self) -> None:
        empty_obs = {
            'pixel': {'channel_id': np.empty(0, dtype=np.int64), 'xyz_mm': np.empty((0, 3), dtype=np.float32)},
            'bar': {
                'channel_id': np.empty(0, dtype=np.int64),
                'layer': np.empty(0, dtype=np.int16),
                'panel': np.empty(0, dtype=np.int16),
                'horizontal': np.empty(0, dtype=np.int8),
                'bar': np.empty(0, dtype=np.int16),
                'bounds_mm': np.empty((0, 5), dtype=np.float32),
                'edep_proxy_MeV': np.empty(0, dtype=np.float32),
                'pe_proxy': np.empty(0, dtype=np.int64),
            },
            'tracker_hypothesis': {
                'station': np.empty(0, dtype=np.int8),
                'xy_mm': np.empty((0, 2), dtype=np.float32),
                'sigma_xy_mm': np.empty((0, 2), dtype=np.float32),
            },
        }
        res0 = domains.compute_observable_containment(empty_obs, self.geometry, self.bounds)
        self.assertEqual(res0['obs_domain_id'], 0)
        self.assertEqual(res0['c_obs'].shape, (16,))
        self.assertTrue(np.isfinite(res0['c_obs']).all())
        self.assertFalse(res0['cal_start'][3])

        # Synthetic pixel-contained event around x=0, y=-12, z=pixel_Zpos[2]
        nx, ny = self.bounds['nx'], self.bounds['ny']
        col = nx // 2
        row = ny // 2
        plane = 2
        pid = (plane * ny + row) * nx + col
        px_x = self.geometry['pixel_Xpos'][col]
        px_y = self.geometry['pixel_Ypos'][row]
        px_z = self.geometry['pixel_Zpos'][plane]
        pix_obs = {
            'pixel': {
                'channel_id': np.array([pid, pid + 1, pid + nx], dtype=np.int64),
                'xyz_mm': np.array(
                    [[px_x, px_y, px_z], [px_x + 0.02, px_y, px_z], [px_x, px_y + 0.02, px_z]],
                    dtype=np.float32,
                ),
            },
            'bar': {
                'channel_id': np.array([10], dtype=np.int64),
                'layer': np.array([3], dtype=np.int16),
                'panel': np.array([3], dtype=np.int16),
                'horizontal': np.array([0], dtype=np.int8),
                'bar': np.array([12], dtype=np.int16),
                'bounds_mm': np.array([[-6.75, 3.75, -222.0, 198.0, self.geometry['scint_Zpos'][3]]], dtype=np.float32),
                'edep_proxy_MeV': np.array([25.0], dtype=np.float32),
                'pe_proxy': np.array([500], dtype=np.int64),
            },
            'tracker_hypothesis': empty_obs['tracker_hypothesis'],
        }
        res1 = domains.compute_observable_containment(pix_obs, self.geometry, self.bounds)
        self.assertEqual(res1['obs_domain_id'], 1)
        self.assertTrue(np.isfinite(res1['c_obs']).all())
        self.assertTrue(res1['cal_start'][3])

        # Synthetic upstream veto event -> D_partial_or_edge (3)
        veto_obs = {
            **pix_obs,
            'bar': {
                'channel_id': np.array([0, 10], dtype=np.int64),
                'layer': np.array([0, 3], dtype=np.int16),
                'panel': np.array([0, 3], dtype=np.int16),
                'horizontal': np.array([0, 0], dtype=np.int8),
                'bar': np.array([12, 12], dtype=np.int16),
                'bounds_mm': np.array([
                    [-6.75, 3.75, -222.0, 198.0, self.geometry['scint_Zpos'][0]],
                    [-6.75, 3.75, -222.0, 198.0, self.geometry['scint_Zpos'][3]],
                ], dtype=np.float32),
                'edep_proxy_MeV': np.array([20.0, 25.0], dtype=np.float32),
                'pe_proxy': np.array([400, 500], dtype=np.int64),
            },
        }
        res3 = domains.compute_observable_containment(veto_obs, self.geometry, self.bounds)
        self.assertEqual(res3['obs_domain_id'], 3)
        self.assertEqual(res3['c_obs'][6], 1.0)

    def test_energy_match_reweight(self) -> None:
        rng = np.random.default_rng(42)
        e = np.concatenate([
            rng.uniform(20, 4500, 1000),
            rng.uniform(20, 4500, 1000),
            rng.uniform(20, 2250, 1000),
            rng.uniform(20, 4500, 1000),
        ])
        y = np.repeat(np.arange(4), 1000)
        w, info = compute_energy_match_weights(e, y)
        self.assertTrue(np.all(w[e > 2300.0] == 0.0))
        self.assertTrue(np.all(w[e <= 2300.0] >= 0.0))
        self.assertEqual(len(info['edges_GeV']), 21)


if __name__ == '__main__':
    unittest.main()
