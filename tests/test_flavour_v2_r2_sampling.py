"""The order intervention must preserve complete exposure and update budgets."""

import unittest

import numpy as np

from analysis.flavour.v2.r2.sampling import composition, remixed_batches


class SamplingReplayTests(unittest.TestCase):
    def test_remix_preserves_every_event_and_tail_position(self):
        original = [[7, 2, 9], [4], [0, 8, 5], [1, 3, 6]]
        remixed = remixed_batches(original, 20261118)
        self.assertEqual([len(batch) for batch in remixed], [3, 1, 3, 3])
        np.testing.assert_array_equal(np.sort(np.concatenate(remixed)), np.arange(10))
        self.assertEqual(remixed, remixed_batches(original, 20261118))
        self.assertNotEqual(remixed, original)

    def test_composition_uses_actual_counts_including_missing_classes(self):
        labels = np.array([0, 0, 1, 1, 3, 3])
        row = composition([[0, 1], [2, 3], [4, 5]], labels)
        self.assertEqual(row['events'], 6)
        self.assertEqual(row['updates'], 3)
        self.assertEqual(row['class_counts'], [2, 2, 0, 2])
        self.assertEqual(row['terminal_100_batches_class_counts'], [2, 2, 0, 2])
        self.assertIsNone(row['lag1_class_proportions'][2])


if __name__ == '__main__':
    unittest.main()
