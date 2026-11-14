"""Scientific invariants for calibration and frozen inclusive working points."""
import unittest

import numpy as np

from analysis.flavour.s6.calibration import background_threshold, selection, temperature_fit, transform


class CalibrationTest(unittest.TestCase):
    def test_known_overconfidence_temperature(self):
        y = np.repeat(np.arange(4), 100)
        predicted = y.copy()
        predicted[np.arange(len(y)) % 100 >= 75] = (predicted[np.arange(len(y)) % 100 >= 75]+1) % 4
        p = np.full((len(y), 4), .1/3)
        p[np.arange(len(y)), predicted] = .9
        fitted = temperature_fit(p, y)
        self.assertAlmostEqual(fitted['temperature'], 1.5, places=7)
        self.assertLess(fitted['balanced_NLL_after'], fitted['balanced_NLL_before'])
        np.testing.assert_allclose(transform(p, fitted['temperature']).max(1), .75, atol=1e-8)

    def test_label_shift_constant_prior(self):
        prior = [.1, .6, .001, .299]
        np.testing.assert_allclose(transform(np.full((2, 4), .25), prior=prior), [prior, prior])

    def test_threshold_never_splits_ties(self):
        score = np.array([.9, .8, .8, .5, .85])
        background = np.array([True, True, True, True, False])
        point = background_threshold(score, background, np.ones(5), .5)
        self.assertEqual(point['threshold'], .9)
        self.assertEqual(point['background_passed'], 1)
        point = background_threshold(score, background, np.ones(5), 0.)
        self.assertEqual(point['background_passed'], 0)

    def test_multiclass_reject_and_conflict(self):
        p = np.array([[.25, .2, .2, .35], [.05, .05, .85, .05]])
        np.testing.assert_array_equal(selection(p, [.2, .1, .9, .2]), [1, 4])


if __name__ == '__main__':
    unittest.main()
