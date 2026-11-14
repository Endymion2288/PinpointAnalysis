"""Check exact-target support and geometric scales with known physical examples."""
import unittest

import numpy as np
import pandas as pd

from analysis.flavour.s5.scope_diagnostics import projected_geometry, target_support_weights


class ScopeDiagnosticsTest(unittest.TestCase):
    def test_targets_are_matched_jointly_and_unsupported_isotope_excluded(self):
        records = []
        for target, counts in [(1000741820, [5, 6, 7, 8]), (1000741840, [8, 7, 6, 5]), (2212, [20, 20, 20, 4])]:
            for label, count in enumerate(counts):
                records.extend([{'label': label, 'tgtPDG': target, 'initE': 500000.,
                                 'initX': 10., 'initY': 10., 'initZ': -2400.}]*count)
        frame = pd.DataFrame(records)
        weight, report = target_support_weights(frame)
        self.assertEqual(report['supported_cells'], 2)
        self.assertEqual(report['retained_events'], 52)
        self.assertTrue(np.all(weight[frame.tgtPDG.eq(2212)] == 0))
        self.assertLessEqual(weight.max(), 1.)
        for target in [1000741820, 1000741840]:
            np.testing.assert_allclose([weight[frame.tgtPDG.eq(target) & frame.label.eq(c)].sum()
                                       for c in range(4)], [5.]*4)

    def test_forward_rays_known_one_pitch_and_acceptance(self):
        geometry = {'pixel_Zpos': [0., 10., 20.], 'pixel_Xpos': [-1., 0., 1.], 'pixel_Ypos': [-1., 0., 1.]}
        result = projected_geometry(np.array([0., 0., 5.]), np.array([0., 0., 100.]),
                                    [[10., 0., 100.], [-10., 0., 100.], [0., 0., -100.], [100., 0., 100.]], geometry)
        self.assertEqual(result['first_pixel_z_mm'], 10.)
        self.assertEqual(result['pixel_interval_fraction'], .5)
        self.assertEqual(result['forward_charged_daughters'], 3)
        self.assertEqual(result['projected_charged_in_pixel'], 2)
        self.assertAlmostEqual(result['maximum_parent_separation_pitch'], .5)
        self.assertAlmostEqual(result['minimum_pair_separation_pitch'], 1.)
        self.assertAlmostEqual(result['minimum_pair_separation_mm'], 1.)
        downstream = projected_geometry(np.array([0., 0., 25.]), np.array([0., 0., 100.]), [[10., 0., 100.]], geometry)
        self.assertIsNone(downstream['first_pixel_z_mm'])
        self.assertIsNone(downstream['maximum_parent_separation_pitch'])


if __name__ == '__main__':
    unittest.main()
