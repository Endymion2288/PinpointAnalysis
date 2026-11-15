import unittest

import numpy as np
import pandas as pd
from scipy.special import softmax

from analysis.flavour.v2.r1.metrics import Metrics
from analysis.flavour.v2.r2.evaluate import independent_ranking
from analysis.flavour.v2.r2.accept import learning_gate_matches
from analysis.flavour.v2.r2.readout import crossfit, grouped_folds


class ReadoutDiagnosticTests(unittest.TestCase):
    def test_capacity_control_cannot_reuse_narrow_architecture_fit_gate(self):
        recipe = {'family': 'B1', 'task': 'T4', 'precision': 'BF16_encoder_FP32_head',
            'cnn_width': 16, 'cnn_variant': 'resnet'}
        gate = {'status': 'complete', 'checks': {'fit': True}, 'config': recipe.copy()}
        self.assertTrue(learning_gate_matches(gate, recipe))
        default_recipe = {k: v for k, v in recipe.items() if k!='cnn_variant'}
        self.assertTrue(learning_gate_matches(gate, default_recipe))
        self.assertFalse(learning_gate_matches(gate, {**recipe, 'cnn_width': 32}))
        self.assertFalse(learning_gate_matches(gate, {**recipe, 'task': 'T3'}))

    def test_connected_chunks_stay_in_one_fold_even_with_mixed_labels(self):
        frame = pd.DataFrame({'run': np.repeat(np.arange(30), 4), 'chunk': 0,
            'label': np.tile(np.arange(4), 30), 'family_hi': 0, 'family_lo': np.arange(120)})
        frame.loc[4, 'family_lo'] = frame.loc[0, 'family_lo']
        folds, summary = grouped_folds(frame)
        self.assertEqual(folds[0], folds[4])
        self.assertEqual(len(set(folds[:8])), 1)
        self.assertEqual(summary['resampling_blocks'], 29)
        self.assertEqual(summary['multi_chunk_blocks'], 1)
        np.testing.assert_array_equal(folds, grouped_folds(frame)[0])
        self.assertTrue((np.asarray(summary['fold_class_counts'])>0).all())

    def test_crossfit_bias_recovers_held_out_decisions_without_held_label_leakage(self):
        rng = np.random.default_rng(29)
        labels = np.tile(np.arange(3), 2000)
        logits = rng.normal(size=(len(labels), 3))+1.6*np.eye(3)[labels]
        probability = softmax(logits+[1.2, -1., 0.], axis=1)
        folds = np.arange(len(labels))%5
        contract = {'folds': {'folds': 5}, 'bias_l2': .001, 'bias_bounds': [-3., 3.],
            'temperature_inverse_bounds': [.05, 20.]}
        values, fits = crossfit(probability, labels, folds, contract)
        raw = Metrics(labels, values['raw']).report()
        corrected = Metrics(labels, values['two_global_class_biases']).report()
        self.assertLess(corrected['NLL'], raw['NLL']-.15)
        self.assertGreater(corrected['BA'], raw['BA']+.10)
        self.assertTrue(all(f['bias_health']['success'] for f in fits))
        np.testing.assert_array_equal(values['raw'].argmax(1), values['positive_scalar_temperature'].argmax(1))
        changed = labels.copy()
        changed[folds==0] = (changed[folds==0]+1)%3
        other, _ = crossfit(probability, changed, folds, contract)
        np.testing.assert_array_equal(values['two_global_class_biases'][folds==0],
            other['two_global_class_biases'][folds==0])

    def test_sklearn_ranking_matches_ties_balanced_prior_and_missing_class(self):
        y = np.array([0, 0, 0, 1, 1, 2])
        p = np.array([[.5, .3, .2], [.5, .3, .2], [.3, .4, .3], [.3, .4, .3],
            [.2, .5, .3], [.2, .2, .6]])
        support = np.bincount(y)
        stored = Metrics(y, p).report(len(y)/(3*support[y]))
        for row, expected in zip(independent_ranking(y, p), stored['per_class']):
            for metric in ['AP', 'AUROC']:
                self.assertAlmostEqual(row[metric], expected[metric], places=12)
        self.assertIsNone(independent_ranking(y[y!=2], p[y!=2])[2]['AP'])
        self.assertIsNone(independent_ranking(y[y!=2], p[y!=2])[2]['AUROC'])


if __name__=='__main__':
    unittest.main()
