"""Regression checks for S3 gradient disconnection and uncapped regional encoding."""
import unittest

import numpy as np
import torch

from analysis.flavour.s3.dataset import collate_planes, plane_tokens
from analysis.flavour.s3.models_repair import PlaneModuleClassifierV2, segment_pool
from analysis.flavour.s3.models import PlaneModuleClassifier
from analysis.flavour.s3.health import TrainingHealth


class S3RepairTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.geometry = {'pixel_Xpos': np.arange(100) * .02, 'pixel_Ypos': np.arange(100) * .02,
                         'pixel_Zpos': [0., 10.], 'layer_is_pixel': [1, 0, 1]}

    def item(self, n, index=0):
        xy = np.column_stack([np.arange(n) % 100, np.arange(n) // 100])
        obs = {'pixel': {'channel_id': xy[:, 1] * 100 + xy[:, 0],
                         'xyz_mm': np.column_stack([xy * .02, np.zeros(n)])}}
        nodes, groups, modules, _ = plane_tokens(obs, self.geometry, {'modalities': ['pixel'], 'window_pixels': 1})
        return {'nodes': nodes, 'groups': groups, 'modules': modules, 'index': index,
                'label': index % 4, 'region_blocks': 8}

    def model(self, **kwargs):
        torch.manual_seed(17)
        return PlaneModuleClassifierV2({'dim': 16, 'latents': 8, 'region_queries': 2, **kwargs})

    def test_regional_coverage_scales_with_occupancy(self):
        batch = collate_planes([self.item(350), self.item(0, 1)])
        self.assertEqual(int(batch['region_mask'].sum()), len(batch['module_group']))
        self.assertLessEqual(batch['region_mask'].shape[1], 8)
        expected = int(((batch['module_mask'].sum(1) + 7) // 8).sum())
        self.assertEqual(len(batch['region_event']), expected)
        self.assertGreater(len(batch['region_event']), 4)
        ids = batch['region_group'] * 8 + batch['region_position']
        self.assertEqual(len(ids.unique()), len(ids))
        self.assertEqual(batch['block_detector'][-1], 3)

    def test_empty_and_missing_modalities_batch_isolation(self):
        first, empty = self.item(31), self.item(0, 1)
        for compression in ['fixed', 'multiscale']:
            model = self.model(compression=compression).eval()
            mixed = model(collate_planes([first, empty]))
            self.assertTrue(torch.isfinite(mixed).all())
            torch.testing.assert_close(mixed[:1], model(collate_planes([first])), rtol=3e-5, atol=3e-6)
            torch.testing.assert_close(mixed[1:], model(collate_planes([empty])), rtol=3e-5, atol=3e-6)

    def test_large_projection_and_residual_scales_do_not_disconnect_head(self):
        model = self.model().eval()
        # Recreate extreme upstream projection scales without altering data.
        with torch.no_grad():
            model.local_pool[0].weight.mul_(1000)
            model.head[0].weight.mul_(1000)
            model.head[0].bias.fill_(-1000)
        observed = []
        handle = model.head[1].register_forward_hook(lambda m, i, o: observed.append(o.detach()))
        batch = collate_planes([self.item(31), self.item(17, 1)])
        logits = model(batch)
        torch.nn.functional.cross_entropy(logits, batch['label']).backward()
        handle.remove()
        self.assertTrue(torch.isfinite(logits).all())
        self.assertLess(float(observed[0].square().mean().sqrt()), 1.01)
        for module in [model.input, model.local_pool, model.module_read]:
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(float(g.square().sum()) for g in grads), 0)

    def test_direct_readout_reaches_all_blocks_and_empty_segment_is_zero(self):
        x = torch.arange(12.).reshape(6, 2).requires_grad_()
        group = torch.tensor([0, 0, 1, 1, 2, 2])
        pooled, present = segment_pool(x, group, torch.tensor([1, 1, 1, 1, 0, 0]).bool(), 4)
        torch.testing.assert_close(pooled[2:], torch.zeros(2, 4))
        self.assertEqual(present.tolist(), [True, True, False, False])
        pooled.sum().backward()
        self.assertTrue((x.grad[:4].abs().sum(1) > 0).all())
        self.assertEqual(float(x.grad[4:].abs().sum()), 0.)

    def test_bfloat16_backward_and_optimizer_preserve_encoder_gradients(self):
        batch = collate_planes([self.item(42), self.item(19, 1), self.item(0, 2)])
        model = self.model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                logits = model(batch)
                loss = torch.nn.functional.cross_entropy(logits, batch['label'])
            self.assertEqual(logits.dtype, torch.float32)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            self.assertGreater(float(model.input[0].weight.grad.square().sum()), 0.)
            optimizer.step()

    def test_local_compute_chunk_does_not_change_events_or_gradient(self):
        batch = collate_planes([self.item(71), self.item(31, 1)])
        model = self.model().eval()
        outputs, gradients = [], []
        for chunk in [7, 1024]:
            model.local_chunk_blocks = chunk
            model.zero_grad(set_to_none=True)
            logits = model(batch)
            torch.nn.functional.cross_entropy(logits, batch['label']).backward()
            outputs.append(logits.detach())
            gradients.append(model.input[0].weight.grad.detach().clone())
        torch.testing.assert_close(outputs[0], outputs[1], rtol=3e-5, atol=3e-6)
        torch.testing.assert_close(gradients[0], gradients[1], rtol=3e-4, atol=3e-6)

    def test_internal_health_rejects_finite_but_disconnected_legacy_encoder(self):
        model = PlaneModuleClassifier({'dim': 16, 'latents': 8, 'module_queries': 2})
        with torch.no_grad():
            model.head[0].weight.zero_()
            model.head[0].bias.fill_(-1000)
        batch = collate_planes([self.item(17), self.item(13, 1)])
        health = TrainingHealth(model)
        records = []
        for step in range(3):
            model.zero_grad(set_to_none=True); health.start()
            logits = model(batch)
            torch.nn.functional.cross_entropy(logits, batch['label']).backward()
            records.append(health.finish(logits, step + 1))
        health.close()
        self.assertTrue(records[0]['healthy'])
        self.assertFalse(records[2]['healthy'])
        self.assertEqual(records[2]['activations']['head.1'][0]['zero_fraction'], 1.)
        self.assertEqual(records[2]['gradient_squared_norms']['input'], 0.)

    def test_health_probes_and_empty_input_do_not_report_disconnection(self):
        model = self.model()
        health = TrainingHealth(model)
        for step, size in enumerate([0, 0, 0, 31], 1):
            model.zero_grad(set_to_none=True)
            batch = collate_planes([self.item(size)])
            health.start(batch)
            logits = model(batch)
            torch.nn.functional.cross_entropy(logits, batch['label']).backward()
            record = health.finish(logits, step)
            self.assertTrue(record['healthy'])
            self.assertEqual(record['consecutive_exact_zero_encoder_gradient_samples'], 0)
            self.assertEqual(record['informative_nodes'], size)
            self.assertIn('module_read', record['query_diversity'])
            self.assertTrue(all(np.isfinite(v['conditional_entropy_nats'])
                                for v in record['sampled_attention'].values()))
        health.close()


if __name__ == '__main__':
    unittest.main()
