"""Same convex objective and exact nested-row feature reuse."""
import unittest

import numpy as np
import pandas as pd

from analysis.flavour.s4.feature_bank import select_features
from analysis.flavour.s4.probes import fit_probe


class ReadoutSolverTests(unittest.TestCase):
    def test_correlated_multiclass_solvers_agree_at_measured_stationarity(self):
        rng = np.random.default_rng(42)
        base = rng.normal(size=(240, 8)).astype('float32')
        x = np.concatenate([base, base[:, :4] + 1e-5 * rng.normal(size=(240, 4))], axis=1)
        labels = np.arange(240) % 4
        settings = {'probe_linear_max_iter': 2000, 'probe_linear_tolerance': 1e-8,
            'probe_require_stationarity': True}
        reference, a = fit_probe(x, labels, 'linear', {**settings, 'probe_linear_solver': 'lbfgs'})
        candidate, b = fit_probe(x, labels, 'linear', {**settings, 'probe_linear_solver': 'newton-cg'})
        self.assertTrue(a['passed']); self.assertTrue(b['passed'])
        self.assertLess(abs(a['regularized_mean_objective'] - b['regularized_mean_objective']), 1e-8)
        np.testing.assert_allclose(reference.predict_proba(x), candidate.predict_proba(x), atol=1e-5, rtol=0)
        self.assertLessEqual(b['gradient_infinity_norm'], 1.01e-8)

    def test_solver_iteration_limit_cannot_certify_a_stationary_readout(self):
        rng = np.random.default_rng(4)
        _, audit = fit_probe(rng.normal(size=(80, 12)), np.arange(80) % 4, 'linear',
            {'probe_linear_solver': 'newton-cg', 'probe_linear_max_iter': 1,
             'probe_linear_tolerance': 1e-9, 'probe_require_stationarity': True})
        self.assertFalse(audit['passed']); self.assertFalse(audit['stationarity_passed'])

    def test_feature_selection_keeps_desired_row_order_and_rejects_wrong_labels(self):
        bank = pd.DataFrame({'run': [10081] * 4, 'chunk': [2] * 4,
            'row': [4, 1, 3, 2], 'label': [0, 1, 2, 3]})
        features = {'full': np.arange(12).reshape(4, 3)}
        desired = bank.iloc[[2, 0]].copy()
        np.testing.assert_array_equal(select_features(bank, desired, features)['full'], features['full'][[2, 0]])
        with self.assertRaisesRegex(ValueError, 'exact requested labeled events'):
            select_features(bank, desired.assign(label=3), features)
        with self.assertRaisesRegex(ValueError, 'exact requested labeled events'):
            select_features(bank, desired.assign(row=[9, 10]), features)


if __name__ == '__main__':
    unittest.main()
