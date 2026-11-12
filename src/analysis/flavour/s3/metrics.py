"""Four-class weighted statistics with cached, tie-aware score orderings.

Bootstrap weights count physical event blocks, not independent training seeds.
All APs are sklearn's noninterpolated average precision. Brier is the sum
over four classes per event. Undefined one-vs-rest statistics stay NaN.
"""
import numpy as np


CLASSES = ('CC nu_e', 'CC nu_mu', 'CC nu_tau', 'NC')
NAMES = ('macro_AP', 'macro_AUROC', 'macro_F1', 'balanced_accuracy', 'NLL', 'Brier') + tuple(
    f'class_{c}_{name}' for c in range(4)
    for name in ('AP', 'AUROC', 'precision', 'recall', 'F1'))


class MetricCache:
    def __init__(self, labels, probabilities):
        self.y = np.asarray(labels, dtype=np.int64)
        self.p = np.asarray(probabilities, dtype=np.float64)
        if self.y.ndim != 1 or self.p.shape != (len(self.y), 4):
            raise ValueError('expected labels [N] and probabilities [N,4]')
        if not np.isin(self.y, range(4)).all() or not np.isfinite(self.p).all():
            raise ValueError('invalid labels or probabilities')
        if (self.p < 0).any() or (self.p > 1).any() or not np.allclose(self.p.sum(1), 1, atol=1e-6, rtol=0):
            raise ValueError('probabilities must be normalized')
        self.prediction = self.p.argmax(1)
        self.cells = self.y * 4 + self.prediction
        # Preserve the published scores for ranking: even a tiny per-row
        # renormalization can split exact ties. Only log_loss normalizes.
        clipped = np.clip(self.p, np.finfo(float).eps, 1 - np.finfo(float).eps)
        normalized = clipped / clipped.sum(1, keepdims=True)
        self.nll = -np.log(normalized[np.arange(len(self.y)), self.y])
        self.brier = np.square(self.p - np.eye(4)[self.y]).sum(1)
        self.orders = []
        for c in range(4):
            order = np.argsort(-self.p[:, c], kind='stable')
            score = self.p[order, c]
            ends = np.r_[np.flatnonzero(np.diff(score)), len(score) - 1] if len(score) else np.array([], dtype=int)
            self.orders.append((order, ends, self.y[order] == c))

    def compute(self, weight=None):
        weight = np.ones(len(self.y)) if weight is None else np.asarray(weight, dtype=float)
        if weight.shape != self.y.shape or not np.isfinite(weight).all() or (weight < 0).any():
            raise ValueError('weights must be finite, nonnegative and event aligned')
        total = weight.sum()
        matrix = np.bincount(self.cells, weights=weight, minlength=16).reshape(4, 4)
        if total <= 0:
            return np.full(len(NAMES), np.nan), matrix
        support = matrix.sum(1)
        predicted = matrix.sum(0)
        diagonal = matrix.diagonal()
        precision = np.divide(diagonal, predicted, out=np.zeros(4), where=predicted > 0)
        recall = np.divide(diagonal, support, out=np.zeros(4), where=support > 0)
        f1 = np.divide(2 * diagonal, predicted + support, out=np.zeros(4), where=predicted + support > 0)
        per_class = []
        aps, aucs = [], []
        for c, (order, ends, truth) in enumerate(self.orders):
            w = weight[order]
            tp = np.cumsum(w * truth)[ends]
            fp = np.cumsum(w * ~truth)[ends]
            positive, negative = support[c], total - support[c]
            ap, auc = np.nan, np.nan
            if positive > 0:
                incremental = np.diff(np.r_[0., tp])
                precision_curve = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=tp + fp > 0)
                ap = float(np.dot(incremental, precision_curve) / positive)
            if positive > 0 and negative > 0:
                auc = float(np.dot(tp + np.r_[0., tp[:-1]], np.diff(np.r_[0., fp])) / (2 * positive * negative))
            aps.append(ap); aucs.append(auc)
            per_class.extend([ap, auc, precision[c], recall[c], f1[c]])
        values = [np.mean(aps), np.mean(aucs), np.mean(f1), np.mean(recall[support > 0]),
                  np.dot(weight, self.nll) / total, np.dot(weight, self.brier) / total, *per_class]
        return np.asarray(values), matrix

    def report(self, weight=None):
        values, matrix = self.compute(weight)
        w = np.ones(len(self.y)) if weight is None else np.asarray(weight)
        clean = lambda value: float(value) if np.isfinite(value) else None
        return {**{name: clean(v) for name, v in zip(NAMES, values)},
                'events': len(self.y), 'positive_weight_events': int((w > 0).sum()),
                'weight_sum': float(w.sum()), 'class_weight_sums': matrix.sum(1).tolist(),
                'confusion_weighted_counts': matrix.tolist(),
                'Neff': float(w.sum() ** 2 / np.dot(w, w)) if np.dot(w, w) > 0 else 0.}


def bootstrap_weights(frame, replicates=400, seed=2026):
    """Resample accepted connected chunk blocks inside class-signature strata."""
    from analysis.flavour.s2.evaluate import resampling_groups
    groups, summary = resampling_groups(frame)
    rng = np.random.default_rng(seed)
    weights = np.zeros((replicates, len(frame)), dtype=np.uint16)
    for blocks in groups.values():
        multiplicity = np.asarray([np.bincount(rng.integers(len(blocks), size=len(blocks)),
                                               minlength=len(blocks)) for _ in range(replicates)])
        for block, indices in enumerate(blocks):
            weights[:, indices] = multiplicity[:, block, None]
    return weights, summary


def interval_report(observed, samples, per_seed=None):
    """Pointwise physical-event intervals, conditional on the trained seeds."""
    result = {}
    for i, name in enumerate(NAMES):
        finite = np.isfinite(samples[:, i])
        ci = np.quantile(samples[finite, i], [.025, .975]).tolist() if finite.any() else [None, None]
        item = {'delta': float(observed[i]) if np.isfinite(observed[i]) else None,
                'CI95': ci, 'valid_replicates': int(finite.sum())}
        if per_seed is not None:
            item['per_seed_deltas'] = [float(v) if np.isfinite(v) else None for v in per_seed[:, i]]
            item['training_seed_sample_SD'] = float(np.std(per_seed[:, i], ddof=1)) if len(per_seed) > 1 else None
        result[name] = item
    return result
