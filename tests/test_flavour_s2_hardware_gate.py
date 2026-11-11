"""Prespecified execution selection cannot accept incomplete or invalid runs."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from analysis.flavour.s2.hardware_gate import execution_decision, overall_status, run


class HardwareGateTests(unittest.TestCase):
    def setUp(self):
        self.rule = {'V100_first_epoch_wall_seconds': 100., 'H100_gate_maximum_epoch_wall_seconds': 80.}
        self.activation = {'minimum_applied_over_attempted_steps': .99}
        self.record = {'checks': {'finite_probabilities': True}, 'first_epoch_wall_seconds': 80.,
                       'attempted_steps': 100, 'applied_steps': 99}

    def decide(self, record):
        return execution_decision(self.rule, self.activation, record)

    def test_inclusive_prespecified_boundaries(self):
        self.assertEqual(self.decide(self.record)['status'], 'pass')
        slower = {**self.record, 'first_epoch_wall_seconds': 80.0001}
        self.assertEqual(self.decide(slower)['status'], 'fail')
        skipped = {**self.record, 'applied_steps': 98}
        self.assertEqual(self.decide(skipped)['status'], 'fail')

    def test_speed_cannot_override_invalid_execution(self):
        invalid = {**self.record, 'first_epoch_wall_seconds': 40., 'checks': {'finite_probabilities': False}}
        self.assertEqual(self.decide(invalid)['status'], 'fail')

    def test_both_families_required(self):
        self.assertEqual(overall_status([{'status': 'pass'}, {'status': 'pending'}]), 'pending')
        self.assertEqual(overall_status([{'status': 'fail'}, {'status': 'pending'}]), 'fail')
        self.assertEqual(overall_status([{'status': 'pass'}, {'status': 'pass'}]), 'pass')
        self.assertEqual(overall_status([{'status': 'pass'}]), 'pending')

    def test_classification_quality_does_not_choose_hardware(self):
        a = copy.deepcopy(self.record)
        b = copy.deepcopy(self.record)
        a.update(tau_AP=.1, NLL=3., accuracy=.25)
        b.update(tau_AP=.9, NLL=.1, accuracy=.95)
        self.assertEqual(self.decide(a)['status'], self.decide(b)['status'])

    def test_changed_contract_cannot_reuse_passed_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old_activation = {**self.activation, 'reference_epochs': [{**self.rule, 'task': 0}]}
            activation = {**old_activation, 'minimum_applied_over_attempted_steps': .995}
            config = {'name': 'candidate'}
            contract = {'hardware': {'GPU': 'device'}, 'activation_gate': activation,
                        'experiments': [config]}
            campaign = root / 'campaign.json'
            campaign.write_text(json.dumps(contract))
            directory = root / 'experiments/candidate'
            directory.mkdir(parents=True)
            (directory / 'first_epoch_execution_gate.json').write_text(json.dumps({
                'frozen_rule': activation['reference_epochs'][0], 'frozen_config': config,
                'expected_environment': contract['hardware'], 'frozen_activation_gate': old_activation,
                'record': {'status': 'pass'},
            }))
            with self.assertRaisesRegex(ValueError, 'different frozen contract'):
                run(SimpleNamespace(root=root, campaign=campaign))


if __name__ == '__main__':
    unittest.main()
