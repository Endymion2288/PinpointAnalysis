"""A converged flag cannot replace stationarity or saved-readout replay evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from analysis.flavour.s4.accept_objectives import probe_record


class S4ObjectiveAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directory = self.root / 'experiments/probe'
        self.directory.mkdir(parents=True)
        self.config = {'name': 'probe', 'label_budget': 1000,
            'probe_linear_solver': 'newton-cg', 'probe_linear_tolerance': 1e-6}
        row = {'passed': True, 'stationarity_passed': True, 'stationarity_required': True,
            'finite_coefficients': True, 'convergence_warnings': [], 'solver': 'newton-cg',
            'C': 1., 'fit_events': 1000, 'gradient_infinity_norm': 5e-7}
        self.optimization = {b + '_linear': dict(row) for b in ['full', 'global']}
        self.replay = {'branches': copy.deepcopy(self.optimization), 'fit_events': 1000, 'validation_events': 10000,
            'artifact_prediction_replay': {b: {'events': 10000, 'maximum_probability_difference': 0.,
                'absolute_tolerance': 1e-10, 'passed': True} for b in ['full', 'global']}}

    def record(self, optimization=None, replay=None):
        (self.directory / 'probe_optimization.json').write_text(json.dumps(
            self.optimization if optimization is None else optimization))
        return probe_record(self.root, self.config, self.replay if replay is None else replay)

    def test_complete_measured_evidence_passes(self):
        self.assertTrue(self.record()['passed'])

    def test_claimed_success_does_not_override_bad_gradient_or_changed_objective(self):
        for key, value in [('gradient_infinity_norm', 2e-6), ('gradient_infinity_norm', float('nan')),
                ('C', .1), ('fit_events', 10000), ('convergence_warnings', ['iteration limit'])]:
            row = copy.deepcopy(self.optimization); row['full_linear'][key] = value
            replay = copy.deepcopy(self.replay); replay['branches'] = row
            with self.subTest(key=key, value=value):
                self.assertFalse(self.record(row, replay)['passed'])

    def test_missing_branch_and_unrelated_replay_do_not_pass(self):
        replay = copy.deepcopy(self.replay)
        del replay['artifact_prediction_replay']['global']
        self.assertFalse(self.record(replay=replay)['passed'])
        replay = copy.deepcopy(self.replay)
        replay['branches']['full_linear']['gradient_infinity_norm'] = 1e-8
        self.assertFalse(self.record(replay=replay)['passed'])
        self.assertFalse(self.record(replay={})['passed'])

    def test_saved_prediction_difference_requires_full_10k_and_original_tolerance(self):
        for key, value in [('events', 64), ('absolute_tolerance', .01),
                ('maximum_probability_difference', 1e-6), ('maximum_probability_difference', float('nan'))]:
            replay = copy.deepcopy(self.replay)
            replay['artifact_prediction_replay']['full'][key] = value
            with self.subTest(key=key, value=value):
                self.assertFalse(self.record(replay=replay)['passed'])


if __name__ == '__main__':
    unittest.main()
