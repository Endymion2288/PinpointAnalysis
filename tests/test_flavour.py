"""Synthetic checks for observation integrity, joins, masks and leakage guards."""

import unittest

import numpy as np
import pandas as pd

from analysis.flavour.inventory import validate_alignment, family_signatures, SIGNATURE, label_status, split_names
from analysis.flavour.readout import pixel_channels, pixel_positions, bar_channels, tracker_measurements
from analysis.flavour.features import event_features
from analysis.flavour.datasets import collate_native


class ReadoutTests(unittest.TestCase):
    def setUp(self):
        self.geometry = {'pixel_Xpos': [-1., 1.], 'pixel_Ypos': [-2., 2.], 'pixel_Zpos': [0., 10.],
                         'scint_bar_Xcenter': [-5., 5.], 'scint_bar_Ycenter': [-5., 5.],
                         'scint_Zpos': [20., 30.], 'layer_is_pixel': [1, 0],
                         'scint_detector_width': 20., 'scint_detector_height': 20.,
                         'scint_bar_width': 8., 'scint_bar_height': 8.}

    def test_pixel_joint_geometry_and_duplicates(self):
        pixel = pixel_channels({'colID': [0, 1, 0], 'rowID': [1, 0, 1], 'layerID': [0, 1, 0]}, self.geometry)
        np.testing.assert_array_equal(pixel['channel_id'], [2, 5])
        np.testing.assert_allclose(pixel['xyz_mm'], [[-1, 2, 0], [1, -2, 10]])
        np.testing.assert_allclose(pixel_positions(pixel['channel_id'], self.geometry), pixel['xyz_mm'])

    def test_invalid_geometry_rejected(self):
        with self.assertRaises(ValueError):
            pixel_channels({'colID': [2], 'rowID': [0], 'layerID': [0]}, self.geometry)

    def test_bar_conservation_orientation_permutation(self):
        record = {'layerID': [1, 1, 1], 'panelID': [0, 0, 1], 'isHorizontal': [0, 0, 1],
                  'rowID': [-1, -1, 1], 'colID': [0, 0, -1], 'edep': [.1, .2, .4], 'nPE': [2, 3, 7]}
        bars, conserved = bar_channels(record, self.geometry)
        self.assertEqual(conserved['output_pe'], 12)
        self.assertAlmostEqual(conserved['output_edep_MeV'], .7)
        self.assertEqual(len(bars['channel_id']), 2)
        np.testing.assert_allclose(bars['bounds_mm'][0], [-9, -1, -10, 10, 20])
        shuffled, _ = bar_channels({k: np.asarray(v)[[2, 0, 1]] for k, v in record.items()}, self.geometry)
        for key in bars:
            np.testing.assert_allclose(bars[key], shuffled[key])

    def test_empty_features_are_finite(self):
        pixel = pixel_channels({k: [] for k in ['colID', 'rowID', 'layerID']}, self.geometry)
        bars, conserved = bar_channels({k: [] for k in ['layerID', 'panelID', 'isHorizontal', 'rowID', 'colID', 'edep', 'nPE']}, self.geometry)
        self.assertTrue(np.isfinite(event_features(pixel, bars)).all())
        self.assertEqual(conserved['output_pe'], 0)

    def test_tracker_truth_free_shuffle_and_empty(self):
        record = {'trackerID': [0, 0, 1], 'x': [.12, .14, .7], 'y': [.22, .24, -.5], 'z': [1, 1, 2]}
        _, measured = tracker_measurements(record, 42)
        _, shuffled = tracker_measurements({k: np.asarray(v)[[2, 1, 0]] for k, v in record.items()}, 42)
        for key in measured:
            np.testing.assert_array_equal(measured[key], shuffled[key])
        _, empty = tracker_measurements({k: [] for k in record}, 42)
        self.assertEqual(empty['xy_mm'].shape, (0, 2))

    def test_collate_offsets_dtypes_and_target_isolation(self):
        empty = {'pixel': {'channel_id': [], 'xyz_mm': []},
                 'bar': {'channel_id': [], 'bounds_mm': [], 'pe_proxy': [], 'edep_proxy_MeV': []},
                 'tracker_hypothesis': {'station': [], 'xy_mm': [], 'sigma_xy_mm': []},
                 'has_response': [False, False, False]}
        active = {'pixel': {'channel_id': [123], 'xyz_mm': [[1, 2, 3]]},
                  'bar': empty['bar'], 'tracker_hypothesis': empty['tracker_hypothesis'],
                  'has_response': [True, False, False]}
        batch = collate_native([{'observations': empty, 'targets': {'label': 3}},
                                {'observations': active, 'targets': {'label': 0}}])
        obs = batch['observations']
        self.assertNotIn('label', obs)
        self.assertNotIn('provenance', obs)
        self.assertEqual(obs['pixel']['offsets'].tolist(), [0, 0, 1])
        self.assertEqual(obs['pixel']['channel_id'].tolist(), [123])
        self.assertEqual(str(obs['pixel']['channel_id'].dtype), 'torch.int64')
        self.assertEqual(obs['bar']['values'].shape, (0, 5))
        self.assertEqual(obs['tracker_hypothesis']['sigma_xy_mm'].shape, (0, 2))
        self.assertEqual(batch['targets']['label'].tolist(), [3, 0])


class IdentityTests(unittest.TestCase):
    def test_split_seed_family_and_fraction(self):
        rng = np.random.default_rng(42)
        hi, lo = rng.integers(0, 2**64-1, size=(2, 10000), dtype='uint64')
        splits = split_names(hi, lo)
        for name, expected in [('train', .7), ('validation', .1), ('calibration', .1), ('test', .1)]:
            self.assertLess(abs(np.mean(splits == name) - expected), .02)
        self.assertTrue(np.any(splits != split_names(hi, lo, 17)))
        self.assertEqual(split_names([hi[0]] * 2, [lo[0]] * 2).tolist(), [splits[0]] * 2)

    def test_empty_pixel_id_repair_requires_nonempty_alignment(self):
        ids = np.array([10, 11, 12])
        hits, counts = np.array([10, 10, 12]), np.array([2, 0, 3])
        self.assertTrue(validate_alignment(ids, hits, counts, 'pixel')['ok'])
        self.assertFalse(validate_alignment(ids, hits, counts, 'bar')['ok'])
        self.assertFalse(validate_alignment(ids, hits, np.array([2, 1, 3]), 'pixel')['ok'])
        self.assertFalse(validate_alignment(ids, hits[:2], counts[:2], 'pixel')['ok'])

    def test_family_excludes_randomized_vertex_and_run(self):
        frame = pd.DataFrame([{k: 1. for k in SIGNATURE}] * 2)
        frame['initZ'], frame['run'] = [-100, 20], [10081, 10084]
        a = family_signatures(frame)
        np.testing.assert_array_equal(a[0], a[1])
        frame.loc[1, 'initE'] = 2.
        self.assertFalse(np.array_equal(*family_signatures(frame)))

    def test_elastic_exception_is_not_silently_cc(self):
        frame = pd.DataFrame({'initPDG': [12, 12, 14], 'fslPDG': [11, 11, 11],
                              'intType': [2, 0, 3], 'scatteringType': [3, 7, 7]})
        self.assertEqual(label_status(10081, frame).tolist(),
                         ['consistent', 'nu_e_elastic_candidate', 'nu_e_elastic_candidate'])


if __name__ == '__main__':
    unittest.main()
