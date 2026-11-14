"""FP64 Newton-CG with device Hessian products and an independent CPU certificate."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.utils.optimize import _newton_cg
from threadpoolctl import threadpool_limits
import torch

from analysis.flavour.inventory import write_json


class MultinomialObjective:
    """Mean NLL + ||W||²/(2 C n); no intercept regularization, all FP64."""
    def __init__(self, features, labels, classes, C=1., device='cuda'):
        self.x = torch.as_tensor(np.array(features, dtype=np.float64, copy=True), device=device)
        self.y = torch.as_tensor(np.searchsorted(classes, labels), dtype=torch.long, device=device)
        self.n, self.d = self.x.shape
        self.k = len(classes)
        self.ridge = 1. / (C * self.n)
        self.hessian_products = 0

    def values(self, parameters):
        values = torch.as_tensor(parameters.reshape(self.k, self.d + 1), device=self.x.device)
        weights, intercept = values[:, :-1], values[:, -1]
        scores = self.x @ weights.T + intercept
        probability = scores.softmax(1)
        residual = probability.clone()
        residual[torch.arange(self.n, device=self.x.device), self.y] -= 1
        gradient = torch.cat((residual.T @ self.x / self.n + self.ridge * weights,
            residual.mean(0)[:, None]), 1)
        loss = (scores.logsumexp(1) - scores.gather(1, self.y[:, None]).squeeze(1)).mean()
        loss = loss + .5 * self.ridge * weights.square().sum()
        return loss.item(), gradient.cpu().numpy().ravel(), probability

    def function(self, parameters):
        return self.values(parameters)[0]

    def gradient(self, parameters):
        return self.values(parameters)[1]

    def gradient_hessian(self, parameters):
        _, gradient, probability = self.values(parameters)

        def product(vector):
            self.hessian_products += 1
            values = torch.as_tensor(vector.reshape(self.k, self.d + 1), device=self.x.device)
            weights, intercept = values[:, :-1], values[:, -1]
            projected = self.x @ weights.T + intercept
            response = probability * (projected - (probability * projected).sum(1, keepdim=True))
            result = torch.cat((response.T @ self.x / self.n + self.ridge * weights,
                response.mean(0)[:, None]), 1)
            return result.cpu().numpy().ravel()

        return gradient, product


def fit(features, labels, *, device='cuda', tolerance=1e-6, maximum_iterations=200, progress=None):
    from analysis.flavour.s4.probes import linear_stationarity
    started = time.perf_counter()
    scaler = StandardScaler().fit(features)
    transformed = scaler.transform(features).astype('float64')
    classes = np.unique(labels)
    if not np.array_equal(classes, np.arange(4)):
        raise ValueError('four-class label ordering required')
    objective = MultinomialObjective(transformed, labels, classes, device=device)
    rounds = []

    def diagnostic(parameters):
        gradient, hessian = objective.gradient_hessian(parameters)
        row = {'outer_evaluations': len(rounds) + 1,
            'gradient_infinity_norm': float(np.max(np.abs(gradient))),
            'hessian_products': objective.hessian_products, 'seconds': time.perf_counter() - started}
        rounds.append(row)
        if progress:
            write_json(progress, {'phase': 'fitting_device_fp64_newton', **row,
                'utc': datetime.now(timezone.utc).isoformat()})
        return gradient, hessian

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        parameters, iterations = _newton_cg(diagnostic, objective.function, objective.gradient,
            np.zeros(4 * (features.shape[1] + 1), dtype=np.float64), tol=tolerance,
            maxiter=maximum_iterations, maxinner=200)
    messages = [str(row.message) for row in caught]
    model = LogisticRegression(C=1., solver='newton-cg', tol=tolerance, max_iter=maximum_iterations)
    values = parameters.reshape(4, features.shape[1] + 1)
    model.classes_ = classes
    model.coef_, model.intercept_ = values[:, :-1].copy(), values[:, -1].copy()
    model.n_iter_ = np.array([iterations], dtype=np.int32)
    model.n_features_in_ = features.shape[1]
    pipeline = make_pipeline(scaler, FunctionTransformer(np.asarray, kw_args={'dtype': np.float64}), model)
    with threadpool_limits(limits=2):
        certificate = linear_stationarity(pipeline, features, labels)
    from sklearn.metrics import log_loss
    probability = pipeline.predict_proba(features)
    finite = bool(np.isfinite(parameters).all())
    passed = not messages and finite and certificate['gradient_infinity_norm'] <= tolerance + 1e-10
    audit = {'solver': 'newton-cg', 'execution': 'device FP64 analytic gradient and Hessian products',
        'device': str(device), 'iterations': [int(iterations)], 'maximum_iterations': maximum_iterations,
        'tolerance': tolerance, 'convergence_warnings': messages, 'finite_coefficients': finite,
        'passed': passed, 'fit_events': len(labels), 'input_features': features.shape[1],
        'standardization': 'fit on fitted-label events only; original float32 rounding preserved',
        'fit_NLL': float(log_loss(labels, probability, labels=classes)),
        'fit_accuracy': float(np.mean(probability.argmax(1) == labels)),
        'coefficient_L2_norm': float(np.sqrt(np.square(model.coef_).sum() + np.square(model.intercept_).sum())),
        'C': 1., **certificate, 'stationarity_required': True, 'stationarity_passed': passed,
        'hessian_products': objective.hessian_products, 'rounds': rounds,
        'seconds': time.perf_counter() - started,
        'optimizer_implementation': 'sklearn 1.5.2 _newton_cg, maxinner=200, Wolfe line search'}
    del objective
    return pipeline, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--task', type=int, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    config = json.loads(args.campaign.read_text())['experiments'][args.task]
    output = args.root / 'experiments' / config['name']; output.mkdir(parents=True, exist_ok=True)
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('formal acceleration gate requires CUDA')
    torch.set_num_threads(1)
    started = time.perf_counter()
    write_json(output / 'manifest.json', {'status': 'running', 'config': config})
    from analysis.flavour.s4.feature_bank import load
    from analysis.flavour.s4.probes import linear_stationarity
    cases = []
    for reference_name in config['references']:
        directory = args.root / 'experiments' / reference_name
        manifest = json.loads((directory / 'manifest.json').read_text())
        if manifest['status'] != 'complete':
            raise ValueError('completed CPU reference required')
        reference_config = manifest['config']
        (features, metadata), = load(args.root, reference_config, splits=('train',))
        labels = metadata.label.to_numpy()
        for branch, array in features.items():
            candidate, row = fit(array, labels, device=args.device, progress=output / 'progress.json')
            reference = joblib.load(directory / (branch + '_linear_readout.joblib'))
            with threadpool_limits(limits=2):
                cpu = linear_stationarity(reference, array, labels)
                error = float(np.max(np.abs(reference.predict_proba(array) - candidate.predict_proba(array))))
            gap = abs(cpu['regularized_mean_objective'] - row['regularized_mean_objective'])
            passed = row['passed'] and cpu['gradient_infinity_norm'] <= 1e-6 + 1e-10 and gap <= 1e-5 and error <= .005
            cases.append({'reference': reference_name, 'branch': branch, 'candidate': row,
                'CPU_reference': cpu, 'absolute_objective_difference': gap,
                'maximum_fit_probability_difference': error, 'passed': passed})
            print(json.dumps(cases[-1]), flush=True)
            del candidate, reference
        del features, metadata
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'passed': bool(cases) and all(r['passed'] for r in cases),
        'config': config, 'cases': cases, 'seconds': time.perf_counter() - started,
        'GPU': torch.cuda.get_device_name(), 'code_commit': os.environ.get('PINPOINT_CODE_COMMIT'),
        'scope': 'same cached train rows/scaler/C=1/objective; original CPU stationarity threshold; no validation selection'}
    write_json(output / 'solver_probe.json', report)
    write_json(output / 'manifest.json', {'status': 'complete' if report['passed'] else 'optimization_review_required',
        'config': config, 'passed': report['passed'], 'seconds': report['seconds'], 'GPU': report['GPU']})
    if not report['passed']:
        raise RuntimeError('CPU/GPU same-objective acceleration gate failed')


if __name__ == '__main__':
    main()
