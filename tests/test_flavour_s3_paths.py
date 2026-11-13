"""Diagnostics must preserve modality availability and reproduce the fitted head."""
import unittest

import numpy as np
import torch

from analysis.flavour.s3.diagnose_repair_paths import (
    branch_slices, contrast, effective_rank, head_logits, presence_permutation,
)
from analysis.flavour.s3.models_repair import PlaneModuleClassifierV2


class S3PathDiagnosticTests(unittest.TestCase):
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
