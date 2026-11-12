"""Statistical checks against independent sklearn and literal block replicas."""
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score,
                             balanced_accuracy_score, log_loss, confusion_matrix,
                             precision_recall_fscore_support, roc_curve)

from analysis.flavour.s3.metrics import MetricCache, NAMES, bootstrap_weights, interval_report
from analysis.flavour.s3.evaluate import contrast
from analysis.flavour.s3.select import select_candidates, CANDIDATES


class FourClassStatisticsTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(871)
        self.y = np.tile(np.arange(4), 40)
        # Quantization deliberately creates many exact ties and argmax ties.
        score = rng.integers(1, 5, size=(len(self.y), 4)).astype(float)
        self.p = score / score.sum(1, keepdims=True)
        self.w = rng.integers(0, 5, size=len(self.y)).astype(float)

    def test_weighted_ties_match_sklearn(self):
        cache = MetricCache(self.y, self.p)
        for w in [None, self.w, self.w / 7]:
            values, matrix = cache.compute(w)
            result = dict(zip(NAMES, values))
            pred = self.p.argmax(1)
            np.testing.assert_allclose(matrix, confusion_matrix(self.y, pred, sample_weight=w))
            self.assertAlmostEqual(result['NLL'], log_loss(self.y, self.p, sample_weight=w), places=12)
            self.assertAlmostEqual(result['macro_F1'], f1_score(self.y, pred, average='macro', sample_weight=w), places=12)
            self.assertAlmostEqual(result['balanced_accuracy'], balanced_accuracy_score(self.y, pred, sample_weight=w), places=12)
            details = precision_recall_fscore_support(self.y, pred, sample_weight=w, zero_division=0)
            for c in range(4):
                self.assertAlmostEqual(result[f'class_{c}_AP'], average_precision_score(self.y == c, self.p[:, c], sample_weight=w), places=12)
                self.assertAlmostEqual(result[f'class_{c}_AUROC'], roc_auc_score(self.y == c, self.p[:, c], sample_weight=w), places=12)
                for name, expected in zip(['precision', 'recall', 'F1'], details[:3]):
                    self.assertAlmostEqual(result[f'class_{c}_{name}'], expected[c], places=12)
                fpr, tpr, _ = roc_curve(self.y == c, self.p[:, c], sample_weight=w, drop_intermediate=False)
                for acceptance in [.01, .001]:
                    self.assertAlmostEqual(result[f'class_{c}_eff_at_{acceptance:g}_FPR'],
                                           tpr[np.flatnonzero(fpr <= acceptance)[-1]], places=12)
            expected_brier = np.average(np.square(self.p - np.eye(4)[self.y]).sum(1), weights=w)
            self.assertAlmostEqual(result['Brier'], expected_brier, places=12)

    def test_block_weights_equal_literal_resampling(self):
        frame = pd.DataFrame({'run': self.y, 'chunk': np.arange(len(self.y)) // 16,
                              'label': self.y, 'row': np.arange(len(self.y)),
                              'family_hi': np.arange(len(self.y)), 'family_lo': 0})
        # Accepted family connects two different chunks; they must resample together.
        frame.loc[[0, 16], 'family_hi'] = 99999
        weights, summary = bootstrap_weights(frame, replicates=12)
        np.testing.assert_array_equal(weights[:, 0], weights[:, 16])
        self.assertEqual(summary['multi_chunk_blocks'], 1)
        cache = MetricCache(self.y, self.p)
        for w in weights:
            indices = np.repeat(np.arange(len(self.y)), w)
            expected, matrix = MetricCache(self.y[indices], self.p[indices]).compute()
            measured, measured_matrix = cache.compute(w)
            np.testing.assert_allclose(measured, expected, atol=1e-12, equal_nan=True)
            np.testing.assert_allclose(measured_matrix, matrix)

    def test_missing_class_and_shared_seed_interval(self):
        weight = np.where(self.y == 3, 0., 1.)
        report = MetricCache(self.y, self.p).report(weight)
        self.assertIsNone(report['macro_AP'])
        self.assertIsNone(report['class_3_AP'])
        self.assertIsNone(report['class_3_AUROC'])
        cache = MetricCache(self.y, self.p)
        delta = cache.compute()[0] - cache.compute()[0]
        report = interval_report(delta, np.tile(delta, (10, 1)), np.tile(delta, (3, 1)))
        for value in report.values():
            self.assertEqual(value['CI95'], [0., 0.])
            self.assertEqual(value['training_seed_sample_SD'], 0.)

    def test_seed_mean_uses_shared_physical_bootstrap(self):
        rng = np.random.default_rng(832)
        point, samples = {}, {}
        per_seed = [('a17', 'b17'), ('a42', 'b42'), ('a2026', 'b2026')]
        shared = rng.normal(size=(20, len(NAMES)))
        for a, b in per_seed:
            point[a] = {d: np.zeros(len(NAMES)) for d in ['raw', 'common_support']}
            point[b] = {d: np.ones(len(NAMES)) for d in ['raw', 'common_support']}
            samples[a] = {d: shared for d in ['raw', 'common_support']}
            samples[b] = {d: shared + 1 for d in ['raw', 'common_support']}
        names = [n for pair in per_seed for n in pair]
        result = contrast(names, np.tile([-1., 1.], 3) / 3, point, samples, {}, per_seed)
        for domain in result['metrics'].values():
            for value in domain.values():
                np.testing.assert_allclose(value['CI95'], [1., 1.], atol=1e-14)
                self.assertEqual(value['training_seed_sample_SD'], 0.)

    def test_pareto_and_single_frontier_fallback(self):
        rows = [{'task': t, 'macro_AP': .6, 'NLL': 1., 'allocated_gpu_hours': 2.} for t in CANDIDATES]
        rows[0].update(macro_AP=.62, NLL=.99, allocated_gpu_hours=3.)
        rows[1].update(macro_AP=.61, NLL=.98, allocated_gpu_hours=2.)
        self.assertEqual(select_candidates(rows)[0], [0, 1])
        rows[0].update(macro_AP=.62, NLL=.95, allocated_gpu_hours=1.)
        self.assertEqual(select_candidates(rows)[0], [0, 1])
        rows[1].update(macro_AP=.618, allocated_gpu_hours=1.5)
        self.assertEqual(select_candidates(rows)[0], [0, 1])


if __name__ == '__main__':
    unittest.main()
