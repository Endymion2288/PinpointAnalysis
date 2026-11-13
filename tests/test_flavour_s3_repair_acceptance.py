"""Scientific acceptance must reject missing health evidence and failed learning."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from analysis.flavour.s3.accept_repair import confirmation_contract, experiment_record
from analysis.flavour.s3.evaluate import factorial_specifications
from analysis.flavour.s3.inspect_repair import summarize_samples
from analysis.flavour.s3.finalize_repair import terminal_accounting_complete
from analysis.flavour.s2.readback import compatible_gpu_name


class RepairAcceptanceTests(unittest.TestCase):
    def test_final_evaluation_waits_for_terminal_reservations(self):
        report = {'missing_history_records': 0, 'allocations': [{'status': 4}, {'status': 3}]}
        self.assertTrue(terminal_accounting_complete(report))
        report['allocations'][0]['status'] = 2
        self.assertFalse(terminal_accounting_complete(report))
        report['allocations'][0]['status'] = 4
        report['allocations'][0]['allocation_complete'] = False
        self.assertFalse(terminal_accounting_complete(report))
        report['allocations'] = []
        self.assertFalse(terminal_accounting_complete(report))
        report.update(missing_history_records=1, allocations=[{'status': 4}])
        self.assertFalse(terminal_accounting_complete(report))

    def test_recorded_BF16_hardware_allowlist_keeps_actual_device(self):
        allowed=['NVIDIA H100 NVL','NVIDIA A100-PCIE-40GB']
        self.assertTrue(compatible_gpu_name('NVIDIA A100-PCIE-40GB',allowed))
        self.assertTrue(compatible_gpu_name('NVIDIA H100 NVL MIG 3g.47gb',allowed))
        self.assertFalse(compatible_gpu_name('Tesla V100-PCIE-32GB',allowed))

    def test_no_legacy_factorial_is_invented_for_revision2(self):
        self.assertEqual(factorial_specifications({'factorial_interactions': []}), [])
        self.assertEqual(len(factorial_specifications({})), 1)

    def test_both_confirmation_seeds_and_identical_scientific_config_are_required(self):
        root = Path(__file__).resolve().parents[1]
        screen = json.loads((root / 'roadmap/configs/s3_repair_screen.json').read_text())
        confirmation = json.loads((root / 'roadmap/configs/s3_repair_confirmation.json').read_text())
        self.assertTrue(confirmation_contract(screen, confirmation))
        changed = copy.deepcopy(confirmation); changed['experiments'].pop()
        self.assertFalse(confirmation_contract(screen, changed))
        changed = copy.deepcopy(confirmation); changed['experiments'][0]['batch_size'] = 4
        self.assertFalse(confirmation_contract(screen, changed))

    def test_finite_healthy_samples_without_an_encoder_signal_do_not_pass(self):
        row = {'step': 1, 'healthy': True, 'informative_nodes': 7,
               'gradient_squared_norms': {'input': 1.}, 'activations': {'head.1': [
                   {'RMS': 1., 'min': -2., 'max': 2., 'zero_fraction': 0.,
                    'fraction_below_minus5': 0., 'all_finite': 1.}]}}
        self.assertTrue(all(summarize_samples([row], 1, 100)['checks'].values()))
        changed = copy.deepcopy(row); changed['gradient_squared_norms']['input'] = 0.
        self.assertFalse(all(summarize_samples([changed], 1, 100)['checks'].values()))
        self.assertFalse(all(summarize_samples([row], 201, 100)['checks'].values()))
        empty = copy.deepcopy(row); empty['informative_nodes'] = 0
        self.assertFalse(all(summarize_samples([empty], 1, 100)['checks'].values()))

    def test_checkpoint_readback_alone_cannot_pass_scientific_acceptance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); path = root / 'experiments/a'; path.mkdir(parents=True)
            config = {'name': 'a'}
            for filename, data in [('manifest.json', {'status': 'complete', 'config': config, 'calibration_test_used': False}),
                                   ('final_checkpoint_checks.json', {'status': 'pass', 'checks': {str(i): True for i in range(14)}})]:
                (path / filename).write_text(json.dumps(data))
            evaluation = {'a': {'allocation_complete': True}}
            record = experiment_record(root, config, evaluation)
            self.assertFalse(record['internal_health_and_effective_learning'])
            (path / 'internal_health_checks.json').write_text(json.dumps(
                {'internal_health_and_learning_passed': True, 'checks': {'learning': False}}))
            self.assertFalse(experiment_record(root, config, evaluation)['internal_health_and_effective_learning'])
