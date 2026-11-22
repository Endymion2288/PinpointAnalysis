"""Unit tests for Stage 327 Score-Bank DRO Partial-AUC tail fine-tuning."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')

import numpy as np
import torch
from torch import nn

from analysis.flavour.nextgen.seal_guard import (
    SealAuditGuard,
    SealViolationError,
    assert_unsealed_split,
    verify_inference_truth_invariance,
)
from analysis.flavour.nextgen.tail_dro_pauc import (
    ScoreBank,
    TailDROConfig,
    compute_anchored_tail_loss,
    compute_multiclass_ece,
    compute_normalized_pauc,
    dro_pauc_surrogate_loss,
    extract_penultimate_cache,
    finetune_head_on_cached_latents,
    freeze_backbone_for_tail_finetuning,
    predict_from_cached_latents,
    snapshot_backbone_weights,
    tau_log_odds_from_logits7,
    verify_backbone_bit_identity,
)
from analysis.flavour.nextgen.tracklet_ipformer import TrackletIPFormer, collate_tracklet_pf, extract_tracklet_pf_item
from analysis.flavour.v3.common.domains import compute_observable_containment, detector_bounds
from tests.test_nextgen_representation_probes import SyntheticPinpointPhysicsGenerator
from tests.test_nextgen_seal_and_splits import build_minimal_geometry


class _ToyBackboneHeadModel(nn.Module):
    """Minimal backbone + event_head_7 model mimicking NextGenFlavourModel contract."""

    def __init__(self, in_dim: int = 32, hidden_dim: int = 48) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.register_buffer('running_scale', torch.ones(hidden_dim))
        self.event_head_7 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 7),
        )

    def forward_NextGen(self, batch: dict[str, torch.Tensor]):
        h = self.backbone(batch['x'].float()) * self.running_scale
        logits7 = self.event_head_7(h)
        p7 = torch.softmax(logits7.float(), dim=-1)
        probs4 = torch.stack([p7[:, 0], p7[:, 1] + p7[:, 2], p7[:, 3] + p7[:, 4] + p7[:, 5], p7[:, 6]], dim=-1)
        from analysis.flavour.nextgen.protocols import NextGenModelOutput
        return NextGenModelOutput(
            logits7=logits7,
            probs4=probs4,
            charm_logit=torch.zeros(len(logits7), device=logits7.device),
            v_pv_mm=torch.zeros((len(logits7), 3), device=logits7.device),
            scale_pv_mm=torch.ones((len(logits7), 3), device=logits7.device),
        )


class TestTailDROPAUC(unittest.TestCase):
    def test_score_bank_fifo_wraparound_and_finite_grads(self) -> None:
        bank = ScoreBank(capacity=8192)
        self.assertEqual(int(bank.count.item()), 0)

        # Fill past capacity (8192) in chunks to force FIFO wrap-around
        total_pushed = 9000
        all_scores = torch.arange(total_pushed, dtype=torch.float32) * 0.01
        all_tau = (torch.arange(total_pushed) % 4) == 2
        for st in range(0, total_pushed, 500):
            bank.enqueue(all_scores[st:st + 500], all_tau[st:st + 500])

        self.assertEqual(int(bank.count.item()), 8192)
        self.assertEqual(int(bank.ptr.item()), total_pushed % 8192)
        neg_b, pos_b = bank.get_valid()
        self.assertEqual(int(neg_b.numel() + pos_b.numel()), 8192)
        # Earliest 808 entries were overwritten; minimum retained score must be 808 * 0.01
        min_retained = float(torch.cat([neg_b, pos_b]).min().item())
        self.assertAlmostEqual(min_retained, 808 * 0.01, places=4)

        # Verify finite gradients in both FP32 and BF16
        for dt in (torch.float32, torch.bfloat16):
            lg = torch.randn(64, 7, dtype=dt, requires_grad=True)
            lg_ref = lg.detach().clone()
            y4 = torch.randint(0, 4, (64,))
            y4[:16] = 2
            cfg = TailDROConfig(lambda_dro=0.5, lambda_nll=1.0, lambda_kl=0.5, bank_capacity=8192)
            loss, _ = compute_anchored_tail_loss(lg, lg_ref, y4, bank=bank, cfg=cfg)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertIsNotNone(lg.grad)
            self.assertTrue(torch.isfinite(lg.grad).all())

    def test_backbone_freeze_and_bit_identity_plus_cache_equivalence(self) -> None:
        torch.manual_seed(7)
        model = _ToyBackboneHeadModel(in_dim=24, hidden_dim=48)
        snap = snapshot_backbone_weights(model, head_attr='event_head_7')
        head = freeze_backbone_for_tail_finetuning(model, head_attr='event_head_7')

        for name, p in model.named_parameters():
            if name.startswith('event_head_7.'):
                self.assertTrue(p.requires_grad)
            else:
                self.assertFalse(p.requires_grad)

        x = torch.randn(512, 24)
        y = torch.arange(512) % 4
        batches = [{'x': x[i:i + 128], 'label': y[i:i + 128]} for i in range(0, 512, 128)]
        cache = extract_penultimate_cache(model, batches, torch.device('cpu'), verify_atol=1e-7)
        self.assertLessEqual(cache['max_verify_diff'], 1e-7)

        cfg = TailDROConfig(passes=3, lr=5e-4, bank_capacity=256, batch_size=128)
        finetune_head_on_cached_latents(head, cache, cfg, torch.device('cpu'))
        id_res = verify_backbone_bit_identity(model, snap, head_attr='event_head_7')
        self.assertTrue(id_res['passed'])

        # Full model vs cached head must remain exact within 1e-6 after fine-tuning
        pred_cached = predict_from_cached_latents(head, cache['penultimate'], torch.device('cpu'))
        model.eval()
        with torch.no_grad():
            full_logits = model.forward_NextGen({'x': x}).logits7.numpy()
        self.assertLessEqual(float(np.abs(pred_cached['logits7'] - full_logits).max()), 1e-6)

    def test_synthetic_low_fpr_separation_improves_with_bounded_nll(self) -> None:
        """Construct a low-FPR tail-confounder toy dataset where top-1% bg overlaps tau in baseline."""
        rng = np.random.default_rng(2026)
        n_tr, n_va = 4096, 4096

        def _make_split(n: int):
            y = np.arange(n) % 4
            d = 16
            h = rng.normal(0.0, 0.45, size=(n, d)).astype(np.float32)
            # Bulk class signal in coords 0..3
            for c in range(4):
                h[y == c, c] += 2.2
            # Tail discriminator in coord 4: neutral on bulk, separates true tau vs 4% hard bg confounders
            is_tau = y == 2
            is_bg = ~is_tau
            hard_bg = is_bg & (rng.random(n) < 0.04)
            h[hard_bg, 2] += 3.8   # mimics tau in bulk coords -> sits above tau bulk in top-1% tail!
            h[hard_bg, 4] -= 3.0   # negative in coord 4 (confounder signature)
            h[is_tau, 4] += 2.2    # positive in coord 4 (true tau signature)
            return torch.from_numpy(h), torch.from_numpy(y)

        h_tr, y_tr = _make_split(n_tr)
        h_va, y_va = _make_split(n_va)

        # Construct a pretrained head initialized on bulk coords 0..3 only (ignoring tail coord 4)
        head = nn.Sequential(
            nn.LayerNorm(16),
            nn.Linear(16, 48),
            nn.SiLU(),
            nn.Linear(48, 7),
        )
        # Pretrain using masked input (coord 4 zeroed) to mimic a bulk-CE pretrained checkpoint
        ce_opt = torch.optim.AdamW(head.parameters(), lr=3e-3)
        h_tr_bulk = h_tr.clone()
        h_tr_bulk[:, 4] = 0.0
        for _ in range(4):
            for st in range(0, n_tr, 256):
                ce_opt.zero_grad()
                lg7 = head(h_tr_bulk[st:st + 256])
                p4 = torch.stack([
                    torch.softmax(lg7, -1)[:, 0],
                    torch.softmax(lg7, -1)[:, 1:3].sum(-1),
                    torch.softmax(lg7, -1)[:, 3:6].sum(-1),
                    torch.softmax(lg7, -1)[:, 6],
                ], dim=-1)
                loss = torch.nn.functional.nll_loss(torch.log(p4.clamp_min(1e-7)), y_tr[st:st + 256])
                loss.backward()
                ce_opt.step()

        head.eval()
        with torch.no_grad():
            lg_tr_ref = head(h_tr)
        pre_va = predict_from_cached_latents(head, h_va, torch.device('cpu'))
        y_va_np = y_va.numpy()
        pre_nll = float(-np.log(np.clip(pre_va['probs4'][np.arange(n_va), y_va_np], 1e-8, 1.0)).mean())
        pre_pauc = compute_normalized_pauc(pre_va['tau_score'], y_va_np, fpr_max=0.01)
        pre_ece = compute_multiclass_ece(pre_va['probs4'], y_va_np)

        train_cache = {'penultimate': h_tr, 'logits7_ref': lg_tr_ref, 'label': y_tr}
        cfg = TailDROConfig(
            fpr_target=0.01,
            lambda_dro=1.2,
            lambda_nll=1.0,
            lambda_kl=0.25,
            bank_capacity=2048,
            lr=1.5e-3,
            passes=6,
            batch_size=128,
            seed=42,
        )
        finetune_head_on_cached_latents(head, train_cache, cfg, torch.device('cpu'))

        post_va = predict_from_cached_latents(head, h_va, torch.device('cpu'))
        post_nll = float(-np.log(np.clip(post_va['probs4'][np.arange(n_va), y_va_np], 1e-8, 1.0)).mean())
        post_pauc = compute_normalized_pauc(post_va['tau_score'], y_va_np, fpr_max=0.01)
        post_ece = compute_multiclass_ece(post_va['probs4'], y_va_np)

        self.assertGreater(post_pauc, pre_pauc + 0.05)
        self.assertLessEqual(post_nll - pre_nll, 0.02)
        self.assertLessEqual(post_ece, max(0.05, 2.0 * pre_ece))

    def test_seal_guard_and_tracklet_39key_truth_invariance(self) -> None:
        with SealAuditGuard():
            with self.assertRaises(SealViolationError):
                assert_unsealed_split('Cal-A', stage=1)
            with self.assertRaises(SealViolationError):
                assert_unsealed_split('tune6k', stage=1)

        geom = build_minimal_geometry()
        bounds = detector_bounds(geom)
        gen = SyntheticPinpointPhysicsGenerator(geom, seed=11)
        cfg = {
            'dim': 64, 'heads': 4, 'n_rel_pre': 1, 'n_rel_post': 1, 'n_global_layers': 1,
            'pe_only': True, 'bar_threshold_pe': 2.5, 'sipm_npix': 14400.0,
            'apply_sipm_saturation': False, 'invert_sipm_saturation': True,
        }
        items = []
        for i in range(4):
            obs, _ = gen.generate_prompt_vs_displaced_event(displaced=(i % 2 == 0))
            dom = compute_observable_containment(obs, geom, bounds, config=cfg)
            items.append(extract_tracklet_pf_item(i, obs, i % 4, dom, geom, bounds, cfg))

        batch = collate_tracklet_pf(items, y_scint_center=bounds['y_scint_center'], n_bars=bounds['n_bars'])
        model = TrackletIPFormer(geom, cfg)
        snap = snapshot_backbone_weights(model, head_attr='event_head_7')
        head = freeze_backbone_for_tail_finetuning(model, head_attr='event_head_7')
        cache = extract_penultimate_cache(model, [batch], torch.device('cpu'), verify_atol=1e-7)
        self.assertLessEqual(cache['max_verify_diff'], 1e-7)

        finetune_head_on_cached_latents(head, cache, TailDROConfig(passes=2, batch_size=4, bank_capacity=16), torch.device('cpu'))
        verify_backbone_bit_identity(model, snap, head_attr='event_head_7')
        inv = verify_inference_truth_invariance(model, batch, atol=1e-7)
        self.assertTrue(inv['passed'])


if __name__ == '__main__':
    unittest.main()
