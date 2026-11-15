"""Tie-aware development metrics for arbitrary three/four-class probabilities."""

import numpy as np


METRICS_REVISION = 'r1-tie-aware-float-boundary-2'


class Metrics:
    def __init__(self, labels, probabilities):
        self.y = np.asarray(labels, dtype='int64')
        self.p = np.asarray(probabilities, dtype='float64')
        self.k = self.p.shape[1]
        if self.p.shape[0] != len(self.y) or not np.isfinite(self.p).all() or not np.allclose(self.p.sum(1), 1., atol=2e-6):
            raise ValueError('invalid probability table')
        self.pred = self.p.argmax(1)
        self.nll = -np.log(np.clip(self.p[np.arange(len(self.y)), self.y], 1e-15, 1.))
        self.orders = []
        for c in range(self.k):
            order = np.argsort(-self.p[:, c], kind='stable')
            ends = np.r_[np.flatnonzero(np.diff(self.p[order, c])), len(order)-1]
            self.orders.append((order, ends, self.y[order] == c))

    def report(self, weight=None):
        w = np.ones(len(self.y)) if weight is None else np.asarray(weight, dtype='float64')
        matrix = np.bincount(self.y*self.k+self.pred, weights=w, minlength=self.k**2).reshape(self.k, self.k)
        support = matrix.sum(1)
        recall = np.divide(matrix.diagonal(), support, out=np.zeros(self.k), where=support>0)
        # cumsum, bincount and sum use different accumulation orders. Permit
        # their O(n*eps) roundoff at an exact FPR/purity boundary so uniform
        # rescaling of weights cannot remove an otherwise admissible tie group.
        relative_roundoff = 4*max(1, len(w))*np.finfo(np.float64).eps
        per_class = []
        for c, (order, ends, truth) in enumerate(self.orders):
            tp = np.cumsum(w[order]*truth)[ends]
            fp = np.cumsum(w[order]*~truth)[ends]
            positive, negative = support[c], w.sum()-support[c]
            precision = np.divide(tp, tp+fp, out=np.zeros_like(tp), where=(tp+fp)>0)
            row = {'recall': float(recall[c]), 'AP': None, 'AUROC': None}
            if positive > 0:
                row['AP'] = float(np.dot(np.diff(np.r_[0., tp]), precision)/positive)
            if positive > 0 and negative > 0:
                row['AUROC'] = float(np.dot(tp+np.r_[0., tp[:-1]], np.diff(np.r_[0., fp]))/(2*positive*negative))
                for fpr in [.01, .001]:
                    allowed = np.flatnonzero(fp <= fpr*negative*(1+relative_roundoff))
                    j = allowed[-1] if len(allowed) else None
                    row[f'eff_FPR_{fpr}'] = float(tp[j]/positive) if j is not None else 0.
                    row[f'actual_FPR_{fpr}'] = float(fp[j]/negative) if j is not None else 0.
                    row[f'threshold_FPR_{fpr}'] = float(self.p[order[ends[j]], c]) if j is not None else None
                    row[f'purity_FPR_{fpr}'] = float(tp[j]/(tp[j]+fp[j])) if j is not None and tp[j]+fp[j]>0 else None
                    row[f'coverage_FPR_{fpr}'] = float((tp[j]+fp[j])/w.sum()) if j is not None else 0.
                for purity in [.8, .9]:
                    eligible = np.flatnonzero((precision >= purity*(1-relative_roundoff)) & (tp+fp > 0))
                    j = eligible[np.argmax(tp[eligible])] if len(eligible) else None
                    row[f'eff_purity_{purity}'] = float(tp[j]/positive) if j is not None else 0.
                    row[f'coverage_purity_{purity}'] = float((tp[j]+fp[j])/w.sum()) if j is not None else 0.
            per_class.append(row)
        return {'BA': float(recall[support>0].mean()), 'NLL': float(np.dot(w, self.nll)/w.sum()),
            'macro_AP': float(np.mean([r['AP'] for r in per_class])) if all(r['AP'] is not None for r in per_class) else None,
            'macro_AUROC': float(np.mean([r['AUROC'] for r in per_class])) if all(r['AUROC'] is not None for r in per_class) else None,
            'per_class': per_class, 'events': len(self.y), 'raw_counts': np.bincount(self.y, minlength=self.k).tolist(),
            'weighted_prior': (support/support.sum()).tolist(),
            'weighted_counts': support.tolist(), 'confusion_counts': matrix.tolist(),
            'Neff': float(w.sum()**2/np.dot(w, w))}


def three_from_four(probabilities):
    value = np.asarray(probabilities)[:, [0, 1, 3]]
    return value/value.sum(1, keepdims=True).clip(1e-15)


def three_targets(labels):
    return np.array([0, 1, -1, 2])[np.asarray(labels, dtype=int)]


def quick_report(labels, probabilities):
    y, p = np.asarray(labels), np.asarray(probabilities)
    keep = y != 2
    p3 = p[keep] if p.shape[1] == 3 else three_from_four(p[keep])
    result = {'T3': Metrics(three_targets(y[keep]), p3).report()}
    if p.shape[1] == 4:
        result['T4'] = Metrics(y, p).report()
    return result
