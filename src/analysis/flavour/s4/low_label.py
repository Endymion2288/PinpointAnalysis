"""Train-only recipe selection, complete passes and staged encoder updates."""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import torch

from analysis.flavour.s2.dataset import SpatialBatchSampler
from analysis.flavour.s3.health import TrainingHealth
from analysis.flavour.s4.dataset import LabelDataset, restrict_dataset


READOUT_PREFIXES = ('head.', 'linear_readout.', 'head_input_norm.', 'branch_norms.', 'summary.')


def inner_folds(metadata, folds=3, seed=20261114):
    """Keep accepted physical families together, within the nested label budget."""
    family = list(zip(metadata.family_hi.tolist(), metadata.family_lo.tolist()))
    groups = np.unique(np.asarray(family, dtype='uint64'), axis=0, return_inverse=True)[1]
    strata = metadata.label.astype(str)
    if 'tau_branch' in metadata:
        strata = strata + ':' + metadata.tau_branch.fillna('unknown').astype(str)
    assignment = np.full(len(metadata), -1, dtype='int16')
    # Randomize whole group IDs, then stratify without shuffling count rows.
    # This keeps group labels aligned in sklearn versions whose SGKF shuffle
    # permutes group-count rows before mapping the original group IDs back.
    groups = np.random.default_rng(seed).permutation(groups.max() + 1)[groups]
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=False)
    for fold, (_, held) in enumerate(splitter.split(np.zeros(len(metadata)), strata, groups)):
        assignment[held] = fold
    if np.any(assignment < 0):
        raise ValueError('train-inner fold assignment incomplete')
    return assignment


class TrainInnerDataset(LabelDataset):
    """Both fitting and selection read only the original train partition."""
    def __init__(self, root, split, config, cache_dir=None):
        if split not in {'train', 'validation'}:
            raise ValueError('train-inner dataset has only fit/selection roles')
        super().__init__(root, 'train', config, cache_dir)
        assignment = inner_folds(self.metadata, config['inner_folds'], config['inner_split_seed'])
        held = assignment == config['selection_fold']
        selected = np.flatnonzero(~held if split == 'train' else held)
        restrict_dataset(self, selected)


class CompletePassSampler(SpatialBatchSampler):
    """Stable shard groups, shuffled order; visit every event exactly once."""
    def __init__(self, dataset, batch_size, seed, shuffle=True, steps=None):
        if steps is not None:
            raise ValueError('complete-pass sampling cannot request repeated/truncated steps')
        super().__init__(dataset, batch_size, seed, shuffle, steps)
        expected = dataset.config.get('complete_pass_steps')
        if shuffle and expected is not None and len(self) != expected:
            raise ValueError('complete-pass length differs from the frozen recipe')

    def _windows(self, cycle=0):
        rng = np.random.default_rng(self.seed + self.epoch + 10000 * cycle)
        shard = np.searchsorted(self.dataset.native.boundaries, self.dataset.indexes, side='right') - 1
        capacity = self.dataset.config.get('cache_shards', 2)
        windows = [np.flatnonzero((shard >= start) & (shard < start + capacity))
                   for start in range(0, len(self.dataset.paths), capacity)]
        order = rng.permutation(len(windows)) if self.shuffle else np.arange(len(windows))
        for i in order:
            index = windows[i]
            if self.shuffle:
                rng.shuffle(index)
            yield index


class TransferAdamW(torch.optim.AdamW):
    """Separate encoder LR and optional LP phase; frozen parameters have no state."""
    def __init__(self, model, config):
        groups = {}
        for name, parameter in model.named_parameters():
            role = 'readout' if name.startswith(READOUT_PREFIXES) else 'encoder'
            decay = parameter.ndim >= 2 and not any(k in name for k in ['embedding', 'queries', 'latents'])
            linear = name.startswith('linear_readout.')
            groups.setdefault((role, decay, linear), []).append(parameter)
        learning_rate = config.get('learning_rate', .0003)
        specifications = [{'params': parameters, 'role': role, 'linear_probe_trainable': linear, 'transfer_updates': 0,
            'lr': learning_rate * (config.get('encoder_lr_scale', 1.) if role == 'encoder' else 1.),
            'weight_decay': config.get('weight_decay', .01) if decay else 0.}
            for (role, decay, linear), parameters in groups.items()]
        super().__init__(specifications, lr=learning_rate, betas=(.9, .95), eps=1e-8)
        self.model = model
        self.freeze_steps = config.get('linear_probe_passes', 0) * config['complete_pass_steps']
        self._set_phase()

    def _set_phase(self):
        updates = self.param_groups[0]['transfer_updates']
        frozen = updates < self.freeze_steps
        if getattr(self.model, 'intentional_encoder_freeze', False) and not frozen:
            self.model.needs_transfer_health = True
        self.model.intentional_encoder_freeze = frozen
        for group in self.param_groups:
            for parameter in group['params']:
                parameter.requires_grad_(not frozen or group['linear_probe_trainable'])

    def step(self, closure=None):
        result = super().step(closure)
        for group in self.param_groups:
            group['transfer_updates'] += 1
        self._set_phase()
        return result

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._set_phase()


class TransferHealth(TrainingHealth):
    """Distinguish an intentional frozen encoder from a disconnected encoder."""
    def start(self, batch=None):
        super().start(batch)
        self.model.needs_transfer_health = False

    def finish(self, logits, step, check=True):
        frozen = getattr(self.model, 'intentional_encoder_freeze', False)
        if frozen:
            self.dead_samples = 0
        report = super().finish(logits, step, check)
        report['intentional_encoder_freeze'] = frozen
        report['gradient_policy'] = 'linear readout only; all feature parameters frozen' if frozen else 'full encoder/readout'
        if frozen:
            self.dead_samples = 0
            report['consecutive_exact_zero_encoder_gradient_samples'] = 0
            readout = report['gradient_squared_norms'].get('linear_readout', 0.)
            encoder_grads = [name for name, parameter in self.model.named_parameters()
                             if not name.startswith('linear_readout.') and parameter.grad is not None]
            if readout <= 0 or encoder_grads:
                report['failure_reasons'].append('LP phase requires nonzero readout gradient and absent encoder gradients')
            report['healthy'] = not report['failure_reasons']
        elif report.get('informative_nodes',0):
            # local_pool gradients cannot certify the compiled input/local
            # encoder: a frozen AOT graph can disconnect just those parameters.
            for name in ['input','local']:
                if hasattr(self.model,name) and report['gradient_squared_norms'].get(name,0)<=0:
                    report['failure_reasons'].append('full fine-tuning has no gradient to '+name)
            report['healthy'] = not report['failure_reasons']
        return report
