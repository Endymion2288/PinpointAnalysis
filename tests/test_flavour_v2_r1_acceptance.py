import copy
import unittest

from analysis.flavour.v2.r1.accept import DOMAINS, delivery_contract_checks


class DeliveryContractTests(unittest.TestCase):
    def setUp(self):
        self.config = {'models': [], 'pairs': []}
        for seed in [17, 42, 2026]:
            for family in ['B1', 'B4']:
                self.config['models'].append({'name': f'{family}_{seed}', 'family': family,
                    'task': 'T3', 'kind': 'main', 'seed': seed})
            self.config['pairs'].append({'a': f'B1_{seed}', 'b': f'B4_{seed}',
                'domain': 'all', 'reason': 'Scratch architecture comparison'})
        self.expected = copy.deepcopy(self.config)
        self.result = {'models': {e['name']: {domain: {} for domain in DOMAINS}
                                  for e in self.config['models']},
                       'pairs': copy.deepcopy(self.config['pairs']),
                       'mean_seed_pairs': [{'a_family_task_kind': ['B1', 'T3', 'main'],
                           'b_family_task_kind': ['B4', 'T3', 'main'], 'domain': 'all',
                           'reason': 'Scratch architecture comparison', 'seeds': [17, 42, 2026]}]}

    def failed(self):
        return {row['name'] for row in delivery_contract_checks(self.config, self.expected, self.result)
                if not row['passed']}

    def test_complete_matrix_passes_but_internally_consistent_subset_fails(self):
        self.assertEqual(self.failed(), set())
        self.config['models'] = [e for e in self.config['models'] if e['seed'] != 2026]
        self.config['pairs'].pop()
        self.result['pairs'].pop()
        for name in ['B1_2026', 'B4_2026']:
            del self.result['models'][name]
        self.result['mean_seed_pairs'].clear()
        self.assertEqual(self.failed(), {
            'configuration_preserves_frozen_model_and_pair_matrix',
            'evaluation_contains_exactly_all_frozen_models',
            'evaluation_contains_all_frozen_pairs_once',
            'all_frozen_models_have_all_eleven_domains',
            'all_three_seed_pairs_present_once_with_exact_seeds'})

    def test_duplicate_pair_cannot_replace_a_missing_comparison(self):
        self.result['pairs'][1] = copy.deepcopy(self.result['pairs'][0])
        self.assertEqual(len(self.result['pairs']), len(self.expected['pairs']))
        self.assertEqual(self.failed(), {'evaluation_contains_all_frozen_pairs_once'})

    def test_absent_domain_and_wrong_seed_are_detected(self):
        del self.result['models']['B1_42']['tracker_empty']
        self.result['mean_seed_pairs'][0]['seeds'] = [17, 42, 42]
        self.assertEqual(self.failed(), {'all_frozen_models_have_all_eleven_domains',
                                       'all_three_seed_pairs_present_once_with_exact_seeds'})


if __name__ == '__main__':
    unittest.main()
