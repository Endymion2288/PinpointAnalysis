"""Core S5 mask, probability and geometric-reconstruction contracts."""
import unittest

import numpy as np
import torch
import pyarrow as pa

from analysis.flavour.s3.dataset import collate_planes
from analysis.flavour.s5.model import PhysicsModel, losses
from analysis.flavour.s5.objects import reconstruct, vertex_candidate, match_truth


class ModelContractTest(unittest.TestCase):
    def test_bar_relative_path_cannot_observe_unmeasured_coordinate(self):
        torch.set_num_threads(1)
        model = PhysicsModel({'dim': 32, 'latents': 4, 'region_queries': 2, 'empty_response_bypass': True, 'relative_path': True})
        with torch.no_grad():
            model.auxiliary_heads['vertex'][-1].weight.zero_()
            model.auxiliary_heads['vertex'][-1].bias.zero_()
        nodes = np.zeros((3, 18), dtype='float32')
        nodes[:, :3] = [.5, .2, -.7]
        nodes[:, 4] = nodes[:, 13] = nodes[:, 15] = 1
        event = {'nodes': nodes, 'groups': np.zeros(3, int), 'modules': np.zeros(3, int), 'label': 0, 'index': 0}
        captured = []
        handle = model.relative_branch[0].register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach().clone()))
        model.eval()
        with torch.no_grad():
            model(collate_planes([event]))
            nodes[:, 0] = 123.
            model(collate_planes([event]))
        handle.remove()
        torch.testing.assert_close(captured[0], captured[1])
        torch.testing.assert_close(captured[0][0, :3], torch.tensor([0., .2, -.7]))
        torch.testing.assert_close(captured[0][0, 6:9], torch.tensor([0., 1., 1.]))

    def test_masked_auxiliary_does_not_train_missing_targets(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        config = {'dim': 32, 'latents': 4, 'region_queries': 2, 'empty_response_bypass': True,
                  'relative_path': True, 'use_objects': True, 'six_way': True,
                  'auxiliaries': ['vertex', 'semantics', 'charm', 'kinematics']}
        events = []
        for i in range(3):
            nodes = np.zeros((4, 18), dtype='float32')
            nodes[:, :3] = np.random.default_rng(i).normal(size=(4, 3))
            nodes[:, 12] = nodes[:, 15] = 1
            events.append({'nodes': nodes, 'groups': np.zeros(4, int), 'modules': np.zeros(4, int), 'label': i, 'index': i})
        batch = collate_planes(events)
        model = PhysicsModel(config)
        output = model(batch, torch.randn(3, 31))
        q = output['logits'].softmax(-1)
        torch.testing.assert_close(output['probability'][:, 2], q[:, 2:5].sum(-1))
        torch.testing.assert_close(output['probability'].sum(-1), torch.ones(3))
        targets = {'six': torch.tensor([0, 1, 4]), 'vertex': torch.randn(3, 3), 'semantics': torch.ones(3, 3)/3,
                   'charm': torch.zeros(3, 1), 'kinematics': torch.randn(3, 4)}
        for k in config['auxiliaries']:
            targets[k+'_mask'] = torch.zeros(3, dtype=torch.bool)
        total, values, _ = losses(model, output, targets, batch['label'])
        for k in config['auxiliaries']:
            self.assertEqual(float(values[k].detach()), 0.)
        total.backward()
        self.assertGreater(float(model.input[0].weight.grad.norm()), 0.)
        self.assertGreater(float(model.relative_branch[0].weight.grad.norm()), 0.)
        self.assertGreater(float(model.object_branch[0].weight.grad.norm()), 0.)


class GeometryContractTest(unittest.TestCase):
    def test_straight_native_track_and_empty_object_fallback(self):
        geometry = {'pixel_Xpos': np.arange(100)*.02, 'pixel_Ypos': np.arange(100)*.02, 'pixel_Zpos': np.arange(5)*30.}
        xyz = np.array([[1., 1., z] for z in geometry['pixel_Zpos']])
        ids = np.array([(i*100+50)*100+50 for i in range(5)])
        observation = {'pixel': {'channel_id': ids, 'xyz_mm': xyz},
                       'bar': {'bounds_mm': np.empty((0, 5)), 'edep_proxy_MeV': np.array([])},
                       'tracker_hypothesis': {'station': np.array([], int), 'xy_mm': np.empty((0, 2))}}
        result = reconstruct(observation, geometry)
        self.assertEqual(len(result['tracks']), 1)
        self.assertLess(result['tracks'][0]['rms'], 1e-10)
        self.assertEqual(result['object_features'].shape, (31,))
        truth = {'pixel_truth_rows': [list(range(5)), [50]*5, [50]*5, [9]*5], 'tracker_truth_rows': [[]]*5,
                 'family_hi': 1, 'family_lo': 2, 'vertex_mm': [1., 1., 0.], 'tau_decay_vertex_mm': None,
                 'interaction_track_ids': [9], 'tau_direct_track_ids': []}
        row = pa.Table.from_pylist([truth]).to_pandas().iloc[0].to_dict()
        matched = match_truth(result, row, observation, geometry)
        self.assertEqual(matched['pixel_tracks_matched'], 1)
        self.assertEqual(matched['pixel_truth_recovered'], 1)
        observation['pixel'] = {'channel_id': np.array([], int), 'xyz_mm': np.empty((0, 3))}
        result = reconstruct(observation, geometry)
        self.assertEqual(len(result['tracks']), 0)
        self.assertTrue(np.isfinite(result['object_features']).all())

    def test_vertex_back_projection_recovers_known_intersection(self):
        vertex = np.array([2., -3., -2500.])
        tracks = []
        for slope in [np.array([.003, .002]), np.array([-.002, -.004])]:
            tracks.append({'slope': slope, 'xy': vertex[:2]+slope*500, 'z0': -2000., 'first_z': -2400., 'rms': .01})
        result, count = vertex_candidate(tracks)
        self.assertEqual(count, 1)
        np.testing.assert_allclose(result['xyz'], vertex, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
