"""Physics denominator and four-class promotion rules."""
import copy
import unittest

import numpy as np

from analysis.flavour.s3.accept import promotion, comparison_inventory, confirmation_contract
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
        exps = {name: {'allocated_gpu_hours': 1., 'GPU': 'NVIDIA H100 NVL'}
                for p in comp['pairs'] for name in p.values()}
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

    def test_mig_device_hours_do_not_imply_whole_gpu_speedup(self):
        comp, exp = self.example()
        comp['metrics']['raw']['macro_AP'].update(delta=0., CI95=[-.003, .003])
        for pair in comp['pairs']:
            exp[pair['b']].update(allocated_gpu_hours=.2, GPU='NVIDIA H100 NVL MIG 3g.47gb')
        decision = promotion(comp, exp)
        self.assertFalse(decision['matched_GPU_type_for_efficiency'])
        self.assertFalse(decision['efficiency_path'])
        comp, exp = self.example(); exp['a17']['allocated_gpu_hours'] = None
        self.assertFalse(promotion(comp, exp)['efficiency_path'])

    def test_exact_contrast_inventory_rejects_partial_or_duplicate_results(self):
        comp, _ = self.example()
        for scope in comp['metrics'].values():
            for value in scope.values():
                value['valid_replicates'] = 400
        comp['replicates'] = 400
        pair = {**comp, 'a': 'a42', 'b': 'b42'}
        evaluation = {'paired_comparisons': [pair], 'seed_mean_comparisons': [comp], 'factorial_interactions': []}
        spec = {'pairs': [{'a': 'a42', 'b': 'b42'}], 'seed_mean_comparisons': [{'name': comp['name'], 'pairs': comp['pairs']}]}
        self.assertTrue(all(comparison_inventory(evaluation, spec).values()))
        partial = copy.deepcopy(evaluation); partial['seed_mean_comparisons'] = []
        self.assertFalse(all(comparison_inventory(partial, spec).values()))
        partial = copy.deepcopy(evaluation); partial['paired_comparisons'].append(pair)
        self.assertFalse(all(comparison_inventory(partial, spec).values()))
        partial = copy.deepcopy(evaluation)
        partial['paired_comparisons'][0]['metrics']['raw']['class_3_AP']['valid_replicates'] = 399
        self.assertFalse(all(comparison_inventory(partial, spec).values()))

    def test_confirmation_requires_both_extra_seeds_for_primary_and_coarse(self):
        screen = {'experiments': [{'name': f'model{t}_s42_e4', 'seed': 42, 'role': 'screen'} for t in range(17)]}
        selection = {'selected_tasks': [0, 5], 'confirmed_tasks': [0, 5, 6]}
        configs = [{**screen['experiments'][t], 'seed': s, 'name': f'model{t}_s{s}_e4', 'role': 's3_frozen_confirmation'}
                   for t in [0, 5, 6] for s in [17, 2026]]
        confirmation = {'screen_selection': selection, 'experiments': configs}
        self.assertTrue(confirmation_contract(screen, confirmation))
        partial = copy.deepcopy(confirmation); partial['experiments'].pop()
        self.assertFalse(confirmation_contract(screen, partial))


if __name__ == '__main__':
    unittest.main()
