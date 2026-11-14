"""Independent statistical and generator-bridge edge cases for final evaluation."""
import unittest

import numpy as np

from analysis.flavour.s6.bootstrap import weights
from analysis.flavour.s6.bootstrap import ci
from analysis.flavour.s6.evaluate import FixedCache, exact_interval, reliability, fixed_block_intervals
from analysis.flavour.s6.gfaser import legacy_record
from analysis.flavour.s3.metrics import MetricCache


class FinalEvaluationTest(unittest.TestCase):
    def test_prior_preserved_in_each_resample(self):
        y = np.array([0, 0, 1, 2, 3, 3])
        prior = np.array([.1, .6, .01, .29])
        w = weights(y, prior, [1, 3, 2, 4, 0, 2])
        np.testing.assert_allclose(np.bincount(y, weights=w), prior)

    def test_frozen_selection_counts_and_exact_zero(self):
        y = np.arange(4)
        p = np.array([[.7, .1, .1, .1], [.1, .7, .1, .1], [.1, .1, .7, .1], [.25]*4])
        point = [{'target_FPR': .01, 'classes': [{'threshold': .5}]*4}]
        cache = FixedCache(y, p, point)
        self.assertEqual(cache.final[0].tolist(), [0, 1, 2, 4])
        self.assertEqual(cache.counts(np.ones(4))[0]['classes'][0]['background_passed'], 0)
        interval = exact_interval(0, 100)
        self.assertEqual(interval[0], 0.)
        self.assertAlmostEqual(interval[1], 1-.025**.01)

    def test_reliability_perfect_uniform_counts(self):
        p = np.eye(4)
        result = reliability(np.arange(4), p, np.ones(4), 20)
        self.assertEqual(result['top_label']['ECE'], 0.)
        self.assertEqual(sum(result['top_label']['counts']), 4)

    def test_conditional_blocks_equal_explicit_event_resamples(self):
        y = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        p = np.array([[.8, .1, .05, .05], [.1, .6, .2, .1], [.1, .7, .1, .1], [.25]*4,
                      [.1, .1, .7, .1], [.6, .1, .2, .1], [.1, .1, .1, .7], [.25]*4])
        blocks = np.array([0, 1, 0, 1, 0, 2, 1, 2])
        multiplicity = np.array([[1, 1, 1], [0, 2, 1], [2, 0, 1], [3, 0, 0]])
        prior = np.array([.15, .35, .01, .49])
        fixed = FixedCache(y, p, [{'target_FPR': .01, 'classes': [{'threshold': .5}]*4}])
        actual = fixed_block_intervals(y, fixed, prior, multiplicity, blocks)
        # Independent reference explicitly duplicates each event of a block.
        samples = []
        for m in multiplicity:
            indices = np.repeat(np.arange(len(y)), m[blocks])
            expanded = FixedCache(y[indices], p[indices], fixed.points)
            samples.append(expanded.vector(weights(y[indices], prior)))
        for i, name in enumerate(fixed.names):
            expected = ci(np.asarray(samples)[:, i])
            self.assertEqual(actual[name]['valid_replicates'], expected['valid_replicates'])
            np.testing.assert_allclose(actual[name]['CI95'], expected['CI95'], rtol=1e-12)

    def test_legacy_ancestry_and_units(self):
        event = {'pdgc': [16, 1000741840, 15, 211], 'status': [0, 0, 1, 1],
            'firstMother': [-1, -1, 0, 1], 'px': [0., 0., 1., -1.], 'py': [0.]*4,
            'pz': [500., 0., 300., 200.], 'E': [500., 170., 301., 201.]}
        record = legacy_record(event, [10., -20., -2200.])
        self.assertEqual((record['intType'], record['neu'], record['fspl']), (2, 16, 15))
        self.assertEqual((record['vx'], record['vy'], record['vz']), (.01, -.02, -2.2))
        self.assertEqual(record['scatteringType'], -1)
        self.assertTrue(np.isnan(record['XSec']))

    def test_overlap_zero_weight_events_do_not_change_statistics(self):
        y = np.tile(np.arange(4), 2)
        p = np.array([[.7, .1, .1, .1], [.1, .6, .2, .1], [.1, .1, .7, .1], [.25]*4,
                      [.1, .7, .1, .1], [.8, .1, .05, .05], [.9, .05, .025, .025], [.05, .05, .05, .85]])
        w = np.array([1, 2, 3, 4, 0, 0, 0, 0.])
        full = MetricCache(y, p).compute(w)[0]
        retained = MetricCache(y[:4], p[:4]).compute(w[:4])[0]
        np.testing.assert_allclose(full, retained, rtol=1e-12, atol=1e-12, equal_nan=True)


if __name__ == '__main__':
    unittest.main()
