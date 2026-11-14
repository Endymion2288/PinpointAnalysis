"""Disconnected frontends must fail even when downstream parameters changed."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from analysis.flavour.s4.parameter_audit import inspect


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for prefix in ['input', 'local', 'local_pool']:
            setattr(self, prefix, torch.nn.Linear(2, 2))


class ParameterAuditTests(unittest.TestCase):
    def inspect_case(self, changed, pretrained=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / 'experiments' / 'fit'
            directory.mkdir(parents=True)
            config = {'seed': 42}
            if pretrained:
                config['pretrained_checkpoint'] = 'exact_parent'
            def build(case):
                model = Encoder()
                if case.get('pretrained_checkpoint'):
                    with torch.no_grad():
                        for parameter in model.parameters():
                            parameter.fill_(3.)
                return model
            torch.manual_seed(42)
            model = build(config)
            with torch.no_grad():
                for prefix in changed:
                    getattr(model, prefix).weight.add_(.01)
            (directory / 'manifest.json').write_text(json.dumps({'status': 'complete', 'config': config}))
            for checkpoint in ['best', 'last']:
                torch.save({'config': config, 'epoch': 3, 'model': model.state_dict()}, directory / (checkpoint + '.pt'))
            before = torch.get_rng_state().clone()
            with patch('analysis.flavour.s4.parameter_audit.build_model', side_effect=build):
                report = inspect(root, 'fit')
            torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
            return report

    def test_pool_updates_cannot_hide_unchanged_frontend(self):
        report = self.inspect_case(['local_pool'])
        self.assertFalse(report['passed'])
        self.assertTrue(report['checks']['last_local_pool_updated'])
        for prefix in ['input', 'local']:
            self.assertEqual(report['checkpoints']['last']['branches'][prefix]['maximum_absolute_change'], 0.)

    def test_all_frontends_update_from_seeded_initialization(self):
        self.assertTrue(self.inspect_case(['input', 'local', 'local_pool'])['passed'])

    def test_parent_values_do_not_count_as_training_updates(self):
        self.assertFalse(self.inspect_case([], pretrained=True)['passed'])
        self.assertTrue(self.inspect_case(['input', 'local', 'local_pool'], pretrained=True)['passed'])


if __name__ == '__main__':
    unittest.main()
