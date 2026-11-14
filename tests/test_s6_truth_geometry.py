"""A 10 micrometre silicon plane must not label a 0.1 mm air gap as silicon."""
import unittest

import pandas as pd

from analysis.flavour.s6.truth_geometry import tau_material_labels


class TruthGeometryTest(unittest.TestCase):
    def test_micrometre_header_conversion(self):
        geometry = {'scint_bar_Xcenter': [0.], 'scint_bar_Ycenter': [0.], 'scint_detector_width': 100.,
            'scint_detector_height': 100., 'silicon_thickness': 10., 'tungsten_Zpos': [-20.],
            'tungsten_thickness': 5., 'pixel_Zpos': [0.], 'scint_Zpos': [-10.], 'scint_thickness': 5.}
        frame = pd.DataFrame({'label': [2, 2], 'tau_decay_valid': [True, True],
            'tau_decay_vertex_mm': [[0., 0., -.1], [0., 0., 0.]],
            'tau_decay_location': ['silicon', 'silicon'], 'tau_decay_contained': [True, True]})
        corrected, report = tau_material_labels(frame, geometry)
        self.assertEqual(corrected.tau_decay_location.tolist(), ['gap', 'silicon'])
        self.assertEqual(report['silicon_thickness_mm_used'], .01)
        self.assertEqual(report['material_labels_changed'], 1)


if __name__ == '__main__':
    unittest.main()
