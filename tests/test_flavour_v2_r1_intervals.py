import unittest

import numpy as np

from analysis.flavour.v2.r1.interval_review import class_statistics


class CoreIntervalArithmeticTest(unittest.TestCase):
    def test_rebalance_each_draw_and_exclude_unobserved_class(self):
        labels = np.array([0, 0, 1, 1])
        probabilities = np.array([[.8, .2], [.4, .6], [.2, .8], [.6, .4]])
        weights = np.array([[1, 1, 1, 1], [3, 1, 1, 0], [0, 0, 2, 0]])
        result = class_statistics(labels, probabilities, weights)
        expected_loss = ((3*(-np.log(.8))-np.log(.4))/4-np.log(.8))/2
        np.testing.assert_allclose(result[:, 0], [.5, .875, 1.])
        np.testing.assert_allclose(result[1], [.875, expected_loss, .75, 1.])
        self.assertTrue(np.isnan(result[2, 2]))
        np.testing.assert_allclose(result[2, [1, 3]], [-np.log(.8), 1.])


if __name__ == '__main__':
    unittest.main()
