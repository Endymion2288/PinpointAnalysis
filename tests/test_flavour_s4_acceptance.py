"""S4 completion requires learning evidence and the whole frozen inventory."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from analysis.flavour.s4.accept import fit_record, inventory_checks, terminal_allocations, training_health
from analysis.flavour.s4.campaign import host_memory, memory_scope


class S4AcceptanceTests(unittest.TestCase):
    def test_host_cgroup_selects_nearest_finite_allocation_ancestor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'cgroups';leaf = root / 'condor/job.scope'
            leaf.mkdir(parents=True)
            membership = Path(temporary) / 'membership';membership.write_text('0::/condor/job.scope\n')
            (leaf / 'memory.max').write_text('max')
            (leaf.parent / 'memory.max').write_text('16000')
            (leaf.parent / 'memory.current').write_text('12000')
            (leaf.parent / 'memory.peak').write_text('14000')
            scope = memory_scope(root, membership)
            self.assertEqual(scope, leaf.parent)
            self.assertEqual(host_memory(scope), {'limit_bytes': 16000, 'current_bytes': 12000, 'peak_bytes': 14000})
            membership.write_text('0::/host/path/absent/in/container\n')
            (root / 'memory.max').write_text('60000')
            self.assertEqual(memory_scope(root, membership), root)

    def health(self):
        return {'step': 1, 'healthy': True, 'informative_nodes': 7,
            'gradient_squared_norms': {'input': 1.},
            'activations': {'head.2': [{'all_finite': 1., 'zero_fraction': 0.}]}}

    def test_healthy_decision_cannot_replace_scheduled_gradients_and_activations(self):
        row = self.health()
        self.assertTrue(all(training_health([row], 1, 100).values()))
        for change in ['empty_input', 'zero_gradient', 'nonfinite_gradient', 'dead_head', 'nonfinite_activation']:
            broken = copy.deepcopy(row)
            if change == 'empty_input': broken['informative_nodes'] = 0
            if change == 'zero_gradient': broken['gradient_squared_norms']['input'] = 0.
            if change == 'nonfinite_gradient': broken['gradient_squared_norms']['input'] = float('nan')
            if change == 'dead_head': broken['activations']['head.2'][0]['zero_fraction'] = 1.
            if change == 'nonfinite_activation': broken['activations']['head.2'][0]['all_finite'] = 0.
            with self.subTest(change=change):
                self.assertFalse(all(training_health([broken], 1, 100).values()))
        self.assertFalse(all(training_health([row], 101, 100).values()))
        self.assertFalse(all(training_health([], 0, 100).values()))

    def test_checkpoint_alone_does_not_pass_fit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);directory = root / 'experiments/a';directory.mkdir(parents=True)
            config = {'name': 'a', 'objective': 'classification'}
            manifest = {'config': config, 'status': 'complete', 'best_epoch': 1,
                'calibration_test_used': False, 'optimizer_steps': 1}
            (directory / 'manifest.json').write_text(json.dumps(manifest))
            (directory / 'final_checkpoint_checks.json').write_text(json.dumps({'status': 'pass',
                'best_epoch': 1, 'checks': {str(i): True for i in range(14)}}))
            record = fit_record(root, config, {'a': {}})
            self.assertTrue(record['checks']['checkpoint_readback'])
            self.assertFalse(record['passed'])
            (directory / 'training_health.json').write_text(json.dumps({'records': [self.health()]}))
            self.assertTrue(fit_record(root, config, {'a': {}})['passed'])

    def test_active_and_missing_reservations_cannot_pass(self):
        report = {'allocations': [{'status': 'terminal'}, {'status': 'terminal'}]}
        self.assertTrue(terminal_allocations(report))
        report['allocations'][1]['status'] = 'active_or_missing_history'
        self.assertFalse(terminal_allocations(report))
        self.assertFalse(terminal_allocations({'allocations': []}))
        self.assertFalse(terminal_allocations({}))

    def test_inventory_requires_global_probes_selection_and_all_contrasts(self):
        configurations = {'a': {'name': 'a', 'program': 'train'},
            'p': {'name': 'p', 'program': 'probes'},
            'c': {'name': 'c', 'program': 'train', 'supervised_initial_checkpoint': '/best.pt'}}
        specification = {'include_global_frozen': True, 'include_supervised_selection_envelope': True,
            'pairs': [{'a': 'a', 'b': 'p'}], 'seed_mean_comparisons': [{'name': 'mean'}]}
        evaluation = {'experiments': [{'name': n} for n in ['a', 'p', 'p__global', 'c', 'c__selection']],
            'paired_comparisons': specification['pairs'], 'seed_mean_comparisons': specification['seed_mean_comparisons'],
            'requested_complete': True, 'pending': []}
        self.assertTrue(all(inventory_checks(configurations, specification, evaluation).values()))
        for missing in ['p__global', 'c__selection']:
            changed = copy.deepcopy(evaluation)
            changed['experiments'] = [r for r in changed['experiments'] if r['name'] != missing]
            self.assertFalse(inventory_checks(configurations, specification, changed)['requested_matrix_complete'])
        changed = copy.deepcopy(evaluation);changed['paired_comparisons'] = []
        self.assertFalse(inventory_checks(configurations, specification, changed)['all_frozen_pairs_complete'])
        changed = copy.deepcopy(evaluation);changed['seed_mean_comparisons'] = []
        self.assertFalse(inventory_checks(configurations, specification, changed)['all_frozen_seed_means_complete'])


if __name__ == '__main__':
    unittest.main()
