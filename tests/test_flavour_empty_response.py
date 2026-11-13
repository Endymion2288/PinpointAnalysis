"""Empty detector responses must not backpropagate through normalized fake hits."""
import unittest

import numpy as np
import torch

from analysis.flavour.s3.dataset import collate_planes
from analysis.flavour.s3.health import TrainingHealth
from analysis.flavour.s3.models_repair import PlaneModuleClassifierV2, StableAttention
from analysis.flavour.s4.accept import training_health


class EmptyResponseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def event(self, count, index=0):
        nodes = np.zeros((max(1, count), 18), dtype='float32')
        if count:
            nodes[:, 0] = np.linspace(-.2, .3, count)
            nodes[:, 3:6] = [1., 1., np.log(2.)]
            nodes[:, 12] = nodes[:, 15] = 1.
        return {'nodes': nodes, 'groups': np.zeros(len(nodes), dtype='int64'),
            'modules': np.zeros(len(nodes), dtype='int64'), 'label': index % 4, 'index': index}

    def model(self, safe=True):
        torch.manual_seed(17)
        return PlaneModuleClassifierV2({'dim': 16, 'latents': 8, 'region_queries': 2,
            'empty_response_bypass': safe, 'maximum_preclip_gradient_norm': 10000.}).eval()

    def test_safe_routing_preserves_all_nonempty_outputs_and_state_dictionary(self):
        old, new = self.model(False), self.model(True)
        self.assertEqual(set(old.state_dict()), set(new.state_dict()))
        batch = collate_planes([self.event(7), self.event(2, 1)])
        torch.testing.assert_close(old(batch), new(batch), rtol=0, atol=0)

    def test_empty_only_loss_has_no_gradient_to_hardware_encoder(self):
        for dtype in [None, torch.bfloat16]:
            with self.subTest(dtype=dtype):
                model = self.model()
                batch = collate_planes([self.event(0), self.event(0, 1)])
                health = TrainingHealth(model);health.start(batch)
                with torch.autocast('cpu', dtype=torch.bfloat16, enabled=dtype is not None):
                    logits = model(batch)
                    loss = torch.nn.functional.cross_entropy(logits, batch['label'])
                loss.backward()
                record = health.finish(logits, 1);health.close()
                self.assertTrue(record['healthy'])
                self.assertEqual(record['gradient_squared_norms']['input'], 0.)
                self.assertGreater(record['gradient_squared_norms']['head'], 0.)
                for probe in record['sampled_attention'].values():
                    self.assertTrue(np.isfinite(probe['conditional_entropy_nats']))
                torch.testing.assert_close(logits[0], logits[1], rtol=0, atol=0)

    def test_mixed_batch_separates_empty_and_observed_loss_paths(self):
        model = self.model()
        batch = collate_planes([self.event(0), self.event(9, 1)])
        for selected in [0, 1]:
            model.zero_grad(set_to_none=True)
            logits = model(batch)
            torch.nn.functional.cross_entropy(logits[selected:selected+1], batch['label'][selected:selected+1]).backward()
            norm = sum(float(p.grad.square().sum()) for p in model.input.parameters() if p.grad is not None)
            self.assertTrue(np.isfinite(norm))
            self.assertEqual(norm, 0.) if selected == 0 else self.assertGreater(norm, 0.)
        together = model(batch)
        for i, count in enumerate([0, 9]):
            torch.testing.assert_close(together[i:i+1], model(collate_planes([self.event(count, i)])),
                rtol=3e-5, atol=3e-6)

    def test_empty_attention_disables_attention_update_but_preserves_query_route(self):
        torch.manual_seed(42)
        layer = StableAttention(16, empty_source_bypass=True)
        query = torch.randn(2, 3, 16, requires_grad=True)
        source = torch.zeros(2, 4, 16, requires_grad=True)
        result = layer(query, source, torch.zeros(2, 4, dtype=torch.bool))
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(float(source.grad.abs().sum()), 0.)
        self.assertEqual(float(layer.out.bias.grad.abs().sum()), 0.)
        self.assertGreater(float(query.grad.abs().sum()), 0.)

    def test_extreme_finite_gradient_requires_review(self):
        model = self.model();health = TrainingHealth(model)
        batch = collate_planes([self.event(7)])
        health.start(batch);logits = model(batch)
        torch.nn.functional.cross_entropy(logits, batch['label']).backward()
        model.input[0].bias.grad.fill_(1e12)
        row = health.finish(logits, 1);health.close()
        self.assertFalse(row['healthy'])
        self.assertIn('preclip gradient norm exceeds configured review threshold', row['failure_reasons'])
        self.assertFalse(training_health([row], 1, 100)['preclip_gradients_within_review_limit'])


if __name__ == '__main__':
    unittest.main()
