"""Physics mistakes here would bias every S5 label and visibility result."""
import unittest

import numpy as np

from analysis.flavour.s5.truth import primary_decay, charmed


class DecayContractTest(unittest.TestCase):
    def test_decay_origin_rejects_ionisation_electron_and_other_parent(self):
        rows = {'PID': np.array([7, 7, 7, 8]), 'PDG': np.array([16, 211, 11, 12]),
                'ProdX': np.array([2., 2., 0., 9.]), 'ProdY': np.zeros(4),
                'ProdZ': np.array([-2400., -2400., -2500., -2000.]),
                'Pz': np.ones(4), 'TID': np.arange(4)+100}
        daughters, vertex, valid = primary_decay(rows, [7])
        self.assertTrue(valid)
        self.assertEqual(daughters['PDG'].tolist(), [16, 211])
        np.testing.assert_allclose(vertex, [2, 0, -2400])

    def test_backward_neutrino_has_no_trusted_min_z_origin(self):
        rows = {'PID': np.array([7]), 'PDG': np.array([16]), 'ProdX': np.array([0.]),
                'ProdY': np.array([0.]), 'ProdZ': np.array([-10000.]), 'Pz': np.array([-2.])}
        _, vertex, valid = primary_decay(rows, [7])
        self.assertFalse(valid)
        self.assertIsNone(vertex)

    def test_charm_does_not_depend_on_short_incomplete_pdg_list(self):
        for pdg in [411, -421, 431, 4122, 4332, 443]:
            self.assertTrue(charmed(pdg))
        for pdg in [11, 15, 211, 2212, 1000741860]:
            self.assertFalse(charmed(pdg))


if __name__ == '__main__':
    unittest.main()
