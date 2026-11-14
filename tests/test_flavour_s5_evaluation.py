"""Check paired component errors with a masked event and unequal block weights."""
import unittest

import numpy as np
import pandas as pd

from analysis.flavour.s5.evaluate import auxiliary_quality


class ComponentIntervalTest(unittest.TestCase):
    def test_masked_rows_do_not_enter_paired_block_denominator(self):
        frame = pd.DataFrame({
            'vertex_mm': [[0., 0., 0.], [999., 0., 0.], [420., 0., 0.]],
            'vertex_supervision_valid': [True, False, True],
            'pred_vertex_0': [0., 999., 1.],
            'pred_vertex_1': [0., 0., 0.],
            'pred_vertex_2': [0., 0., 0.],
        })
        weights = np.array([[1., 50., 1.], [3., 100., 0.]])
        result = auxiliary_quality(frame, {'auxiliaries': ['vertex']},
                                   {'vertex': [1., 0., 0.]}, weights)['vertex']
        self.assertEqual(result['events'], 2)
        self.assertEqual(result['component_MSE'][0], .5)
        self.assertEqual(result['constant_train_mean_MSE'][0], 1.)
        interval = result['component_MSE_minus_constant'][0]
        self.assertEqual(interval['estimate'], -.5)
        self.assertEqual(interval['valid_replicates'], 2)
        np.testing.assert_allclose(interval['block_CI95'], [-.9875, -.5125])


if __name__ == '__main__':
    unittest.main()
