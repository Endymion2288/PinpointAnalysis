"""Train-inner isolation, complete coverage and intentional encoder freezing."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from analysis.flavour.s4.low_label import (
    CompletePassSampler, TrainInnerDataset, TransferAdamW, TransferHealth, inner_folds,
)
from analysis.flavour.s4.train import build_model
from analysis.flavour.s4.plan_low_label import RECIPES, resolve


def metadata():
    # Repeated physical families must stay on one side of each inner split.
    return pd.DataFrame({'label': np.repeat(np.arange(4), 36),
        'tau_branch': ['not_tau'] * 72 + ['e', 'mu', 'had'] * 12 + ['not_tau'] * 36,
        'family_hi': np.arange(144, dtype='uint64') // 2,
        'family_lo': np.zeros(144, dtype='uint64')})


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input = torch.nn.Linear(3, 8)
        self.head = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.LayerNorm(8),
                                       torch.nn.SiLU(), torch.nn.Identity(), torch.nn.Linear(8, 4))
        self.linear_readout = torch.nn.Linear(8, 4)
        self.config = {}

    def forward(self, x):
        x = self.input(x)
        return self.head(x) + self.linear_readout(x)


class SamplerData(SimpleNamespace):
    def __len__(self):
        return len(self.indexes)


class LowLabelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def test_inner_folds_preserve_families_and_all_classes(self):
        frame = metadata()
        folds = inner_folds(frame)
        np.testing.assert_array_equal(folds, inner_folds(frame))
        grouped = frame.assign(fold=folds).groupby(['family_hi', 'family_lo']).fold.nunique()
        self.assertTrue(grouped.eq(1).all())
        for fold in range(3):
            self.assertEqual(set(frame.label[folds == fold]), {0, 1, 2, 3})
            self.assertEqual(set(frame.label[folds != fold]), {0, 1, 2, 3})

    def test_inner_dataset_never_requests_original_validation(self):
        calls = []
        def initialize(instance, root, split, config, cache_dir=None):
            calls.append(split)
            instance.metadata = metadata()
            instance.indexes = np.arange(len(instance.metadata))
            instance.workload = np.arange(len(instance.metadata))
        config = {'inner_folds': 3, 'inner_split_seed': 20261114, 'selection_fold': 0}
        with patch('analysis.flavour.s4.low_label.LabelDataset.__init__', initialize):
            train = TrainInnerDataset('/unused', 'train', config)
            hold = TrainInnerDataset('/unused', 'validation', config)
        self.assertEqual(calls, ['train', 'train'])
        self.assertFalse(set(train.indexes) & set(hold.indexes))
        self.assertEqual(set(train.indexes) | set(hold.indexes), set(range(144)))
        self.assertFalse(set(train.metadata.family_hi) & set(hold.metadata.family_hi))

    def test_complete_passes_have_no_repetition_or_missing_events(self):
        dataset = SamplerData(config={'cache_shards': 2, 'bucket_size': 16},
            paths=list(range(9)), native=SimpleNamespace(boundaries=np.arange(0, 100, 10)),
            indexes=np.array([i for i in range(90) if i % 3 != 0]), workload=np.arange(60)[::-1])
        sampler = CompletePassSampler(dataset, 8, 17)
        lengths = []
        for epoch in range(5):
            sampler.epoch = epoch
            batches = list(sampler)
            visits = [i for batch in batches for i in batch]
            self.assertEqual(len(visits), len(dataset))
            self.assertEqual(sorted(visits), list(range(len(dataset))))
            lengths.append(len(batches))
        self.assertEqual(len(set(lengths)), 1)
        with self.assertRaises(ValueError):
            CompletePassSampler(dataset, 8, 17, steps=100)

    def test_freeze_then_unfreeze_and_resume_preserve_optimizer_state(self):
        model = TinyClassifier()
        config = {'complete_pass_steps': 2, 'linear_probe_passes': 1,
            'learning_rate': .01, 'encoder_lr_scale': .1, 'weight_decay': .1}
        optimizer = TransferAdamW(model, config)
        initial = copy.deepcopy(model.state_dict())
        x, y = torch.randn(8, 3), torch.tensor([0, 1, 2, 3] * 2)
        def update(model, optimizer):
            optimizer.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(model(x), y).backward()
            optimizer.step()
        for _ in range(2):
            update(model, optimizer)
        self.assertFalse(model.intentional_encoder_freeze)
        self.assertTrue(torch.equal(initial['input.weight'], model.input.weight))
        self.assertNotIn(model.input.weight, optimizer.state)
        self.assertTrue(torch.equal(initial['head.0.weight'], model.head[0].weight))
        self.assertFalse(torch.equal(initial['linear_readout.weight'], model.linear_readout.weight))
        saved_model, saved_optimizer = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
        resumed = TinyClassifier(); resumed.load_state_dict(saved_model)
        replay = TransferAdamW(resumed, config); replay.load_state_dict(saved_optimizer)
        self.assertFalse(resumed.intentional_encoder_freeze)
        update(model, optimizer); update(resumed, replay)
        self.assertFalse(torch.equal(initial['input.weight'], model.input.weight))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
        roles = {group['role']: group['lr'] for group in optimizer.param_groups}
        self.assertEqual(roles, {'encoder': .001, 'readout': .01})

    def test_mid_LP_resume_keeps_encoder_frozen(self):
        model = TinyClassifier()
        config = {'complete_pass_steps': 3, 'linear_probe_passes': 2}
        optimizer = TransferAdamW(model, config)
        optimizer.zero_grad(set_to_none=True)
        model(torch.randn(8, 3)).sum().backward(); optimizer.step()
        restored = TransferAdamW(TinyClassifier(), config)
        restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        self.assertTrue(restored.model.intentional_encoder_freeze)
        self.assertFalse(restored.model.input.weight.requires_grad)
        self.assertEqual(restored.param_groups[0]['transfer_updates'], 1)

    def test_LP_starts_with_an_exactly_linear_classifier(self):
        config = {'dim': 16, 'latents': 8, 'compile_local': False,
            'empty_response_bypass': True, 'complete_pass_steps': 2, 'linear_probe_passes': 2}
        model = build_model(config)
        TransferAdamW(model, config)
        values = torch.randn(5, model.linear_readout.in_features)
        torch.testing.assert_close(model.head(values), torch.zeros(5, 4), rtol=0, atol=0)
        self.assertTrue(all(name.startswith('linear_readout.') for name, p in model.named_parameters() if p.requires_grad))

    def test_supervised_continuation_cannot_silently_restart_LP(self):
        with self.assertRaisesRegex(ValueError, 'supervised continuation'):
            build_model({'linear_probe_passes': 2, 'supervised_initial_checkpoint': '/unused/best.pt'})

    def test_recipe_resolution_is_budget_local_and_excludes_LP_only_epochs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); configurations = []
            for recipe in RECIPES:
                for kind in ['scratch', 'mae3']:
                    for fold in range(3):
                        name = f'{recipe}_{kind}_{fold}'
                        c = {'name': name, 'label_budget': 1000, 'initialization': kind,
                            'transfer_recipe': recipe, 'selection_fold': fold, 'seed': 42,
                            'batch_size': 16, 'epochs': 12, 'complete_pass_steps': 2,
                            'eligible_selection_first_epoch': 3 if recipe == 'lpft' else 1}
                        configurations.append(c)
                        directory = root / 'experiments' / name; directory.mkdir(parents=True)
                        manifest = {'status': 'complete', 'config': c, 'selection': 'train_inner_NLL',
                            'original_validation_used': False, 'train_events': 666 if fold == 0 else 667,
                            'validation_events': 334 if fold == 0 else 333}
                        history = []
                        for epoch in range(1, 13):
                            loss = 1.5
                            if recipe == 'slow' and epoch == 4:
                                loss = 1.3
                            if recipe == 'lpft' and epoch < 3:
                                loss = .01  # Ineligible, even if much lower.
                            history.append({'epoch': epoch, 'validation': {'NLL': loss},
                                'train_exposures': manifest['train_events'], 'optimizer_steps_cumulative': epoch * 2})
                        (directory / 'manifest.json').write_text(json.dumps(manifest))
                        (directory / 'history.json').write_text(json.dumps(history))
            # A different budget is genuinely unfinished; it must not hold up
            # the independent 1k decision, or enter the 1k objective.
            configurations.append({'name': 'not_started', 'label_budget': 10000})
            campaign = root / 'campaign.json'
            campaign.write_text(json.dumps({'experiments': configurations, 'refit_seeds': [17, 42, 2026],
                'selection_rule': 'shared weighted OOF NLL'}))
            def refit_data(root, split, config):
                return SamplerData(config=config, paths=[Path('unused')], indexes=np.arange(1000),
                    native=SimpleNamespace(boundaries=[0, 1000], close=lambda: None))
            with patch('analysis.flavour.s4.plan_low_label.LabelDataset', side_effect=refit_data):
                report, refit = resolve(root, campaign, budgets=[1000])
            self.assertEqual(report['selections'][0]['selected']['recipe'], 'slow')
            self.assertEqual(report['selections'][0]['selected']['epochs'], 4)
            self.assertFalse(report['original_validation_used'])
            self.assertEqual(len(refit['experiments']), 6)
            self.assertTrue(all(c['select_final_epoch'] and c['epochs'] == 4 for c in refit['experiments']))
            self.assertEqual({c['seed'] for c in refit['experiments']}, {17, 42, 2026})
            self.assertTrue(all(c['complete_pass_steps'] == 63 for c in refit['experiments']))

    def test_health_accepts_explicit_LP_and_still_rejects_disconnection(self):
        model = TinyClassifier()
        TransferAdamW(model, {'complete_pass_steps': 2, 'linear_probe_passes': 5})
        health = TransferHealth(model)
        for step in range(4):
            model.zero_grad(set_to_none=True); health.start()
            logits = model(torch.randn(8, 3)); logits.square().mean().backward()
            report = health.finish(logits, step)
            self.assertTrue(report['healthy'])
            self.assertTrue(report['intentional_encoder_freeze'])
            self.assertNotIn('input', report['gradient_squared_norms'])
        # A zero encoder gradient without the explicit phase remains a defect.
        model.intentional_encoder_freeze = False
        for step in range(3):
            model.zero_grad(set_to_none=True); health.start()
            logits = model(torch.randn(8, 3)); logits.square().mean().backward()
            report = health.finish(logits, step)
        self.assertFalse(report['healthy'])
        health.close()


if __name__ == '__main__':
    unittest.main()
