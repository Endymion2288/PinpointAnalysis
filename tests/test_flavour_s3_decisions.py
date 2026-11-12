"""Physics denominator and four-class promotion rules."""
import copy
import unittest

import numpy as np

from analysis.flavour.s3.accept import promotion
from analysis.flavour.s3.conditional import operating_points
from analysis.flavour.s3.metrics import NAMES


class S3DecisionTests(unittest.TestCase):
    def test_working_points_keep_ties_and_empty_denominators(self):
        truth = np.array([1, 1, 0, 0, 0, 0], dtype=bool)
        score = np.array([.9, .7, .7, .4, .2, .1])
        out, curves = operating_points(truth, score)
        self.assertEqual(out['FPR_0.01']['signal_selected'], 1)
        self.assertEqual(out['FPR_0.01']['background_selected'], 0)
        self.assertFalse(out['FPR_0.01']['tail_statistics_sufficient'])
        self.assertAlmostEqual(out['purity_0.5']['efficiency'], 1.)
        self.assertAlmostEqual(out['purity_0.8']['efficiency'], .5)
        self.assertTrue(curves)
        empty, _ = operating_points([], [])
        self.assertFalse(empty['defined'])
        conditional, _ = operating_points(truth, score, out['FPR_0.01'])
        self.assertEqual(conditional['at_full_validation_1pct_FPR_threshold']['signal_selected'], 1)

    def example(self):
        raw = {name: {'delta': 0., 'CI95': [-.001, .001], 'training_seed_sample_SD': .001} for name in NAMES}
        raw['macro_AP'].update(delta=.012, CI95=[.002, .02])
        comp = {'name': 'candidate', 'metrics': {'raw': raw, 'common_support': copy.deepcopy(raw)},
                'pairs': [{'a': f'a{s}', 'b': f'b{s}'} for s in [17, 42, 2026]]}
        exps = {name: {'allocated_gpu_hours': 1.} for p in comp['pairs'] for name in p.values()}
        return comp, exps

    def test_tau_only_gain_cannot_promote(self):
        comp, exp = self.example()
        self.assertTrue(promotion(comp, exp)['fully_promoted'])
        comp['metrics']['raw']['macro_AP'].update(delta=.002, CI95=[-.001, .003])
        comp['metrics']['raw']['class_2_AP'].update(delta=.15, CI95=[.10, .20])
        self.assertFalse(promotion(comp, exp)['fully_promoted'])

    def test_class_regression_and_nll_guard(self):
        comp, exp = self.example()
        comp['metrics']['raw']['class_1_AP'].update(delta=-.02, CI95=[-.03, -.01])
        self.assertFalse(promotion(comp, exp)['fully_promoted'])
        comp, exp = self.example()
        comp['metrics']['raw']['NLL'].update(delta=.006, CI95=[-.001, .02])
        self.assertFalse(promotion(comp, exp)['fully_promoted'])

    def test_efficiency_requires_measured_cost_and_quality_interval(self):
        comp, exp = self.example()
        comp['metrics']['raw']['macro_AP'].update(delta=0., CI95=[-.003, .003])
        for pair in comp['pairs']:
            exp[pair['b']]['allocated_gpu_hours'] = .49
        self.assertTrue(promotion(comp, exp)['efficiency_path'])
        comp['metrics']['raw']['class_0_AP']['CI95'][0] = -.02
        self.assertFalse(promotion(comp, exp)['efficiency_path'])
        comp, exp = self.example(); exp['a17']['allocated_gpu_hours'] = None
        self.assertFalse(promotion(comp, exp)['efficiency_path'])


if __name__ == '__main__':
    unittest.main()
