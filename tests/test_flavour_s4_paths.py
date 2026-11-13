"""S4 full-validation interventions retain branches and paired metric weights."""
import unittest

import numpy as np
import torch

from analysis.flavour.s3.metrics import MetricCache
from analysis.flavour.s4.functional_probe import contrast_report, permute_context


class S4FunctionalPathTests(unittest.TestCase):
    def test_intervention_preserves_other_branches_and_uses_one_donor(self):
        context = torch.arange(48).reshape(4, 12)
        mapping = np.array([2, 3, 0, 1])
        changed = permute_context(context, mapping, [slice(0, 3), slice(8, 12)])
        torch.testing.assert_close(changed[:, :3], context[mapping, :3])
        torch.testing.assert_close(changed[:, 8:], context[mapping, 8:])
        torch.testing.assert_close(changed[:, 3:8], context[:, 3:8])
        self.assertEqual(context[0, 0].item(), 0)

    def test_paired_interval_averages_mappings_and_keeps_support_weights(self):
        labels = np.tile(np.arange(4), 3)
        probability = np.full((12, 4), .1)
        probability[np.arange(12), labels] = .7
        baseline = MetricCache(labels, probability)
        changed = MetricCache(labels, probability[np.roll(np.arange(12), 1)])
        common = np.array([1., 2., 1., 2.] * 3)
        multiplicities = np.array([[2.] * 4 + [1.] * 4 + [0.] * 4,
                                   [0.] * 4 + [1.] * 4 + [2.] * 4])
        report = contrast_report(baseline, [baseline, changed], np.ones(12), common, multiplicities)
        expected = (changed.nll - baseline.nll) / 2
        self.assertAlmostEqual(report['raw']['NLL']['delta'], expected.mean())
        self.assertAlmostEqual(report['common_support']['NLL']['delta'], np.average(expected, weights=common))
        self.assertEqual(report['raw']['NLL']['per_mapping_deltas'][0], 0.)
        samples = [np.average(expected, weights=common * bw) for bw in multiplicities]
        np.testing.assert_allclose(report['common_support']['NLL']['CI95'], np.quantile(samples, [.025, .975]))
        unchanged = contrast_report(baseline, [baseline], np.ones(12), common, multiplicities)
        self.assertEqual(unchanged['raw']['NLL']['delta'], 0.)
        self.assertEqual(unchanged['probability_change_RMS'], [0.])


if __name__ == '__main__':
    unittest.main()
