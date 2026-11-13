"""Diagnostics must preserve modality availability and reproduce the fitted head."""
import unittest
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from analysis.flavour.s3.diagnose_repair_paths import (
    branch_slices, contrast, effective_rank, head_logits, presence_permutation, reusable_paths,
)
from analysis.flavour.s3.inspect_repair import reusable_inspection
from analysis.flavour.s3.models_repair import PlaneModuleClassifierV2


class S3PathDiagnosticTests(unittest.TestCase):
    def test_reuse_requires_the_same_complete_checkpoint_and_final_probe(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); directory = root / 'experiments/a'; directory.mkdir(parents=True)
            manifest = {'status': 'complete', 'config': {'name': 'a'}, 'best_epoch': 2, 'best_NLL': 1.2}
            health = {'inspection_version': 1, 'config': manifest['config'], 'best_epoch': 2,
                      'selected_validation_NLL': 1.2, 'internal_health_and_learning_passed': True,
                      'checks': {'connected': True}}
            path = {'diagnostic_version': 1, 'config': manifest['config'], 'best_epoch': 2,
                    'training_status_at_probe_start': 'complete', 'interim_diagnostic': False,
                    'probe_events': 64, 'cached_head_matches_original': True, 'presence_preserved': True}
            def save():
                for filename, data in [('manifest.json', manifest), ('internal_health_checks.json', health),
                                       ('path_diagnostics.json', path)]:
                    (directory / filename).write_text(json.dumps(data))
            save()
            self.assertTrue(reusable_inspection(root, 'a')); self.assertTrue(reusable_paths(root, 'a'))
            path['interim_diagnostic'] = True; save()
            self.assertFalse(reusable_paths(root, 'a'))
            path['interim_diagnostic'] = False; manifest['best_epoch'] = 3; save()
            self.assertFalse(reusable_inspection(root, 'a')); self.assertFalse(reusable_paths(root, 'a'))
            manifest['best_epoch'] = 2; health['checks']['connected'] = False; save()
            self.assertFalse(reusable_inspection(root, 'a'))

    def test_presence_derangement_keeps_singleton_and_modality_availability(self):
        presence = np.array([[1, 0, 0], [1, 1, 0], [1, 0, 0], [0, 0, 1], [1, 1, 0]], dtype=bool)
        permutation = presence_permutation(presence)
        np.testing.assert_array_equal(presence[permutation], presence)
        np.testing.assert_array_equal(np.sort(permutation), np.arange(5))
        np.testing.assert_array_equal(permutation, [2, 4, 0, 3, 1])

    def test_spectrum_distinguishes_identical_and_orthogonal_tokens(self):
        repeated = torch.ones(2, 4, 8)
        orthogonal = torch.eye(4)[None]
        torch.testing.assert_close(effective_rank(repeated), torch.ones(2), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(effective_rank(orthogonal), torch.tensor([4.]))
        self.assertEqual(effective_rank(torch.zeros(1, 4, 8)).item(), 0.)

    def test_cached_context_intervention_detects_disconnected_path(self):
        torch.set_num_threads(1); torch.manual_seed(2026)
        model = PlaneModuleClassifierV2({'dim': 16, 'latents': 8, 'readout_skip': True}).eval()
        slices = branch_slices(model.dim, model.direct)
        width = model.head_input_norm.normalized_shape[0]
        coverage = np.zeros(width, dtype=int)
        for interval in slices.values():
            coverage[interval] += 1
        np.testing.assert_array_equal(coverage, np.ones(width))
        context = torch.randn(8, width)
        logits = head_logits(model, context)
        torch.testing.assert_close(logits, model.head(model.head_input_norm(context)) +
                                   model.linear_readout(model.head_input_norm(context)))
        labels = torch.arange(8) % 4
        self.assertEqual(contrast(logits, logits, labels)['probability_change_RMS'], 0.)
        changed = context.clone(); changed[:, slices['response_summary']] *= -1
        self.assertGreater(contrast(logits, head_logits(model, changed), labels)['probability_change_RMS'], 0.)
        with torch.no_grad():
            model.head[-1].weight.zero_(); model.linear_readout.weight.zero_()
        before, after = head_logits(model, context), head_logits(model, changed)
        self.assertEqual(contrast(before, after, labels)['probability_change_RMS'], 0.)


if __name__ == '__main__':
    unittest.main()
