import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from analysis.flavour.v2.r1.prepare import assign_shards, observation_counts
from analysis.flavour.v2.r1.tasks import conditional_three, probabilities, task_loss, three_labels
from analysis.flavour.v2.r1.data import projection, image_geometry, tracker_features
from analysis.flavour.v2.r1.models import TrackerSet, ProjectionEncoder
from analysis.flavour.v2.r1.metrics import Metrics


class TaskTests(unittest.TestCase):
    def test_three_mapping_and_tau_rejection(self):
        self.assertEqual(three_labels(torch.tensor([0, 1, 2, 3])).tolist(), [0, 1, -1, 2])
        with self.assertRaises(ValueError):
            task_loss(torch.zeros(4, 3), torch.arange(4), 'T3')

    def test_hierarchy_is_four_way_log_likelihood(self):
        torch.manual_seed(42)
        logits = torch.randn(17, 4, dtype=torch.float64, requires_grad=True)
        labels = torch.arange(17) % 4
        p = probabilities(logits, 'T3T')
        torch.testing.assert_close(p.sum(1), torch.ones(17, dtype=torch.float64))
        decomposed = task_loss(logits, labels, 'T3T')
        expected = -p[torch.arange(17), labels].log().mean()
        # BCE targets are float32, with harmless rounding at reduction.
        torch.testing.assert_close(decomposed.double(), expected, rtol=1e-6, atol=1e-7)
        gradient = torch.autograd.grad(decomposed, logits)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue((gradient[labels.eq(2), 1:] == 0).all())
        self.assertTrue((gradient[:, 0] != 0).all())

    def test_unknown_decay_keeps_coarse_tau_supervision(self):
        logits = torch.zeros(3, 7, requires_grad=True)
        loss = task_loss(logits, torch.tensor([2, 2, 1]), 'T4D', torch.tensor([-1, -1, -1]))
        loss.backward()
        self.assertLess(logits.grad[0, 2].item(), 0)
        self.assertTrue((logits.grad[:, 4:] == 0).all())
        torch.testing.assert_close(conditional_three(torch.ones(2, 4)/4), torch.ones(2, 3)/3)

    def test_all_tau_hierarchy_is_finite(self):
        value = torch.zeros(2, 4, requires_grad=True)
        task_loss(value, torch.tensor([2, 2]), 'T3T').backward()
        self.assertTrue(torch.isfinite(value.grad).all())


class PreparationTests(unittest.TestCase):
    def test_whole_source_file_coverage_and_mixed_shards(self):
        frame = pd.DataFrame([dict(run=10081+c, chunk=k, row=r, label=c)
                              for c in range(4) for k in range(9) for r in range(k+1)])
        assigned, stop = assign_shards(frame, 7, 42, target=30)
        self.assertEqual(len(assigned), len(frame))
        self.assertFalse(assigned.duplicated(['run', 'chunk', 'row']).any())
        self.assertTrue((assigned.groupby(['run', 'chunk']).shard.nunique() == 1).all())
        self.assertEqual(assigned.shard.min(), 7)
        self.assertEqual(assigned.shard.max()+1, stop)
        self.assertTrue((assigned.groupby('shard').label.nunique() == 4).all())
        again, _ = assign_shards(frame, 7, 42, target=30)
        pd.testing.assert_frame_equal(assigned, again)

    def test_nested_projection_offsets(self):
        table = pa.Table.from_pylist([
            {'pixel': {'channel_id': [1, 2]}, 'bar': {'channel_id': [3]}, 'tracker_hypothesis': {'station': [1]}},
            {'pixel': {'channel_id': []}, 'bar': {'channel_id': [4, 5]}, 'tracker_hypothesis': {'station': []}}])
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch)/'native.parquet'
            pq.write_table(table, path)
            projected = pq.read_table(path, columns=['pixel.channel_id', 'bar.channel_id', 'tracker_hypothesis'])
        actual = observation_counts(projected)
        np.testing.assert_array_equal(actual['pixel_channels'], [2, 0])
        np.testing.assert_array_equal(actual['bar_channels'], [1, 2])
        np.testing.assert_array_equal(actual['tracker_channels'], [1, 0])


class ObservationTests(unittest.TestCase):
    def test_physical_projection_preserves_signal_and_only_measured_bar_coordinate(self):
        geometry = {'pixel_Xpos': [-2., 2.], 'pixel_Ypos': [-2., 2.], 'pixel_Zpos': [-10., 0.],
            'scint_bar_Xcenter': [-1., 1.], 'scint_bar_Ycenter': [-1., 1.], 'scint_bar_width': 2.,
            'scint_bar_height': 2., 'scint_Zpos': [-11., -1.]}
        observation = {'pixel': {'xyz_mm': np.array([[.2, .4, -10.], [.3, .4, -10.]])},
            'bar': {'bounds_mm': np.array([[-2, 0, -2, 2, -11], [-2, 2, 0, 2, -1]]),
                'horizontal': np.array([0, 1]), 'edep_proxy_MeV': np.array([3., 7.]), 'pe_proxy': np.array([10., 20.])}}
        image = image_geometry(geometry)
        self.assertEqual(image['height'], 12)
        indices, values = projection(observation, geometry, image)
        dense = np.zeros(8*image['height']*image['width'])
        np.add.at(dense, indices, values)
        totals = dense.reshape(8, image['height'], image['width']).sum((1, 2))
        np.testing.assert_array_equal(totals, [2, 1, 3, 10, 2, 1, 7, 20])

    def test_tracker_empty_mask_and_set_permutation(self):
        torch.set_num_threads(1)
        tracker = {'station': np.array([1, 0]), 'xy_mm': np.array([[2., 4.], [3., 5.]]),
                   'sigma_xy_mm': np.array([[.58, .017], [.58, .017]])}
        values, summary = tracker_features(tracker)
        self.assertEqual(values.shape, (2, 5))
        model = TrackerSet(16).eval()
        batch = {'tracker': torch.from_numpy(values)[None], 'tracker_mask': torch.ones(1, 2, dtype=torch.bool)}
        initial = model(batch)
        batch['tracker'] = batch['tracker'].flip(1)
        torch.testing.assert_close(initial, model(batch))
        batch['tracker_mask'].zero_()
        torch.testing.assert_close(model(batch), torch.zeros_like(initial))

    def test_absent_projection_has_finite_bounded_backward(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        model = ProjectionEncoder({'height': 64, 'width': 64}, width=16)
        # One measured view, three missing views, including an empty event.
        batch = {'label': torch.tensor([0, 1]), 'projection_values': torch.tensor([1.]),
            'projection_event': torch.tensor([0]), 'projection_indices': torch.tensor([32*64+32])}
        result = model(batch)
        result[:, 0].sum().backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        self.assertTrue(torch.isfinite(norm))
        self.assertLess(norm.item(), 10000.)


class MetricTests(unittest.TestCase):
    def test_weighted_three_and_four_metrics_against_sklearn_with_ties(self):
        from sklearn.metrics import average_precision_score, balanced_accuracy_score, log_loss, roc_auc_score
        rng = np.random.default_rng(42)
        for k in [3, 4]:
            y = np.arange(120) % k
            p = rng.integers(1, 6, (120, k)).astype(float)
            p /= p.sum(1, keepdims=True)
            weight = rng.integers(1, 4, 120)
            actual = Metrics(y, p).report(weight)
            self.assertAlmostEqual(actual['BA'], balanced_accuracy_score(y, p.argmax(1), sample_weight=weight))
            self.assertAlmostEqual(actual['NLL'], log_loss(y, p, sample_weight=weight))
            ap, auc = [], []
            for c in range(k):
                ap.append(average_precision_score(y == c, p[:, c], sample_weight=weight))
                auc.append(roc_auc_score(y == c, p[:, c], sample_weight=weight))
            self.assertAlmostEqual(actual['macro_AP'], np.mean(ap))
            self.assertAlmostEqual(actual['macro_AUROC'], np.mean(auc))

    def test_ties_cannot_create_fractional_false_positive_acceptance(self):
        report = Metrics(np.array([0, 1, 2]*30), np.ones((90, 3))/3).report()
        self.assertEqual(report['per_class'][0]['eff_FPR_0.01'], 0.)
        self.assertEqual(report['per_class'][0]['eff_purity_0.8'], 0.)


if __name__ == '__main__':
    unittest.main()
