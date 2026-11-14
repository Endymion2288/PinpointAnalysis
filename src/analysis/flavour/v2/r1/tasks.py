"""Explicit original labels e=0, mu=1, tau=2, NC=3 and event-normalized losses."""

import torch
from torch.nn import functional as F


NON_TAU = (0, 1, 3)


def output_size(task):
    return {'T3': 3, 'T4': 4, 'T3T': 4, 'T4D': 7}[task]


def three_labels(labels):
    mapping = labels.new_tensor([0, 1, -1, 2])
    return mapping[labels]


def probabilities(logits, task):
    if task == 'T3T':
        t = logits[:, :1].sigmoid()
        r = logits[:, 1:].softmax(-1)*(1-t)
        return torch.cat([r[:, :2], t, r[:, 2:]], -1)
    return logits[:, :output_size(task) if task != 'T4D' else 4].softmax(-1)


def task_loss(logits, labels, task, decay=None, decay_weight=0.1):
    """Every row contributes once; unknown decay never removes the tau CC target."""
    if task == 'T3':
        target = three_labels(labels)
        if (target < 0).any():
            raise ValueError('T3 training must select the frozen non-tau rows')
        return F.cross_entropy(logits, target)
    if task == 'T3T':
        is_tau = labels.eq(2)
        loss = F.binary_cross_entropy_with_logits(logits[:, 0], is_tau.float(), reduction='sum')
        if (~is_tau).any():
            loss = loss+F.cross_entropy(logits[~is_tau, 1:], three_labels(labels[~is_tau]), reduction='sum')
        return loss/len(labels)
    loss = F.cross_entropy(logits[:, :4], labels)
    if task == 'T4D' and decay is not None:
        valid = labels.eq(2) & decay.ge(0)
        if valid.any():
            loss = loss+decay_weight*F.cross_entropy(logits[valid, 4:], decay[valid])
    return loss


def conditional_three(probability):
    selected = probability[:, NON_TAU]
    return selected/selected.sum(-1, keepdim=True).clamp_min(1e-12)
