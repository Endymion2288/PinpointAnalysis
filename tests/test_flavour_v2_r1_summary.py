import unittest

from analysis.flavour.v2.r1.summarize import quality_criteria


T3_RULE = {'delta_BA3_point_min': .02, 'primary_difference_95_lower_min_exclusive': 0.,
           'delta_NLL3_95_upper_max': .01, 'per_class_recall_loss_95_upper_max': .01}
TAU_RULE = {'delta_efficiency_at_1pct_point_min': .01,
            'primary_difference_95_lower_min_exclusive': 0., 'BA3_loss_95_upper_max': .005,
            'delta_NLL4_95_upper_max': .01, 'actual_FPR_required': True}


def effect(point, low, high):
    return {'difference_a_minus_b': point, '95pct': [low, high]}


class QualityCriteriaTests(unittest.TestCase):
    def t3_metrics(self):
        return {'BA3': effect(.03, .02, .04), 'NLL3': effect(-.02, -.03, -.01),
                'recall_e': effect(.08, .06, .10), 'recall_mu': effect(.01, -.005, .02),
                'recall_NC': effect(0., -.01, .01)}

    def test_average_gain_does_not_hide_muon_loss(self):
        metrics = self.t3_metrics()
        self.assertTrue(quality_criteria(metrics, T3_RULE, 'T3')['all_criteria_met'])
        metrics['recall_mu'] = effect(-.06, -.07, -.05)
        report = quality_criteria(metrics, T3_RULE, 'T3')
        self.assertFalse(report['all_criteria_met'])
        self.assertEqual(report['failed'], ['recall_mu_lower_bound'])

    def test_zero_lower_bound_is_not_positive_evidence(self):
        metrics = self.t3_metrics()
        metrics['BA3'] = effect(.02, 0., .04)
        report = quality_criteria(metrics, T3_RULE, 'T3')
        self.assertEqual(report['failed'], ['BA3_positive_lower_bound'])

    def test_tau_gain_requires_common_task_quality_and_actual_fpr(self):
        metrics = {'tau_eff_1pct': effect(.015, .01, .02), 'BA3': effect(-.001, -.004, .002),
                   'NLL4': effect(.002, -.002, .005)}
        valid = quality_criteria(metrics, TAU_RULE, 'T4', actual_fpr_recorded=True)
        self.assertTrue(valid['all_criteria_met'])
        absent = quality_criteria(metrics, TAU_RULE, 'T4')
        self.assertEqual(absent['failed'], ['actual_FPR_recorded_for_both_models'])
        metrics['BA3'] = effect(-.004, -.008, 0.)
        degraded = quality_criteria(metrics, TAU_RULE, 'T4', actual_fpr_recorded=True)
        self.assertEqual(degraded['failed'], ['BA3_lower_bound'])

    def test_unmeasured_interval_cannot_pass(self):
        metrics = self.t3_metrics()
        metrics['BA3'] = effect(None, None, None)
        report = quality_criteria(metrics, T3_RULE, 'T3')
        self.assertFalse(report['all_criteria_met'])
        self.assertEqual(report['failed'], ['BA3_practical_gain', 'BA3_positive_lower_bound'])


if __name__ == '__main__':
    unittest.main()
