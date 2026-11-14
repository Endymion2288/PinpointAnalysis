"""Independently check device objective, Hessian and correlated-feature solve."""
import unittest

import numpy as np
import torch

from analysis.flavour.s4.accelerated_linear import MultinomialObjective, fit
from analysis.flavour.s4.probes import fit_probe


class AcceleratedLinearTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_objective_gradient_and_hessian_match_autograd_with_unpenalized_intercept(self):
        rng = np.random.default_rng(29)
        x = rng.normal(size=(50, 7))
        y = np.arange(50) % 4
        engine = MultinomialObjective(x, y, np.arange(4), device='cpu')
        p = rng.normal(size=32) / 4
        vector = rng.normal(size=32)
        variable = torch.tensor(p, requires_grad=True, dtype=torch.float64)
        values = variable.reshape(4, 8)
        loss = torch.nn.functional.cross_entropy(torch.from_numpy(x) @ values[:, :-1].T + values[:, -1], torch.from_numpy(y))
        loss = loss + values[:, :-1].square().sum() / 100
        gradient, = torch.autograd.grad(loss, variable, create_graph=True)
        hvp, = torch.autograd.grad(gradient @ torch.from_numpy(vector), variable)
        observed, product = engine.gradient_hessian(p)
        self.assertAlmostEqual(engine.function(p), float(loss.detach()), places=13)
        np.testing.assert_allclose(observed, gradient.detach().numpy(), atol=1e-13, rtol=0)
        np.testing.assert_allclose(product(vector), hvp.detach().numpy(), atol=1e-13, rtol=0)

    def test_solver_matches_sklearn_and_limit_does_not_pass(self):
        rng = np.random.default_rng(39)
        base = rng.normal(size=(320, 12)).astype('float32')
        x = np.column_stack((base, base[:, :4] + .001 * rng.normal(size=(320, 4)))).astype('float32')
        labels = np.argmax(base[:, :4] + rng.normal(size=(320, 4)), axis=1)
        candidate, a = fit(x, labels, device='cpu', tolerance=1e-8)
        reference, b = fit_probe(x, labels, 'linear', {'probe_linear_solver': 'newton-cg',
            'probe_linear_tolerance': 1e-8, 'probe_require_stationarity': True})
        self.assertTrue(a['passed']); self.assertTrue(b['passed'])
        self.assertLess(abs(a['regularized_mean_objective'] - b['regularized_mean_objective']), 1e-10)
        np.testing.assert_allclose(candidate.predict_proba(x), reference.predict_proba(x), atol=1e-6, rtol=0)
        _, broken = fit(x, labels, device='cpu', tolerance=1e-9, maximum_iterations=1)
        self.assertFalse(broken['passed'])


if __name__ == '__main__':
    unittest.main()
