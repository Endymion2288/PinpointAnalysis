"""Same-objective Newton-CG regression on frozen original-train representations."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np
import sklearn
import torch

from analysis.flavour.inventory import write_json
from analysis.flavour.s4.dataset import LabelDataset
from analysis.flavour.s4.probes import extract, fit_probe
from analysis.flavour.s4.train import build_model, validate_pretraining


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--task', type=int, required=True)
    args = parser.parse_args(); config = json.loads(args.campaign.read_text())['experiments'][args.task]
    output = args.root / 'experiments' / config['name']; output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'manifest.json', {'status': 'running', 'config': config})
    started = time.perf_counter(); torch.set_num_threads(1)
    rows = []; checks = {}
    for case in config['cases']:
        torch.manual_seed(case['seed']); validate_pretraining(case)
        model = build_model({**case, 'compile_local': False}).eval().requires_grad_(False)
        dataset = LabelDataset(args.root, 'train', case)
        features = extract(model, dataset, {**case, 'extract_branches': ['full', 'global']},
            torch.device('cpu'), output / 'progress.json', case['initialization'])
        labels = dataset.metadata.label.to_numpy(); dataset.native.close()
        for branch, x in features.items():
            reference, a = fit_probe(x, labels, 'linear', {**case, 'probe_linear_solver': 'lbfgs',
                'probe_linear_max_iter': 10000, 'probe_linear_tolerance': 1e-8,
                'probe_require_stationarity': True, 'probe_solver_threads': 2})
            candidate, b = fit_probe(x, labels, 'linear', {**case, 'probe_linear_solver': 'newton-cg',
                'probe_linear_max_iter': 200, 'probe_linear_tolerance': 1e-6,
                'probe_require_stationarity': True, 'probe_solver_threads': 2})
            objective_gap = abs(a['regularized_mean_objective'] - b['regularized_mean_objective'])
            probability_gap = float(np.max(np.abs(reference.predict_proba(x) - candidate.predict_proba(x))))
            passed = a['passed'] and b['passed'] and objective_gap <= 1e-5 and probability_gap <= .005
            name = case['initialization'] + '_' + branch; checks[name] = passed
            rows.append({'name': name, 'events': len(labels), 'features': x.shape[1],
                'LBFGS': a, 'Newton_CG': b, 'absolute_objective_difference': objective_gap,
                'maximum_fit_probability_difference': probability_gap, 'passed': passed})
            print(json.dumps(rows[-1]), flush=True)
        del model, features
    report = {'passed': all(checks.values()), 'checks': checks, 'cases': rows,
        'seconds': time.perf_counter() - started, 'config': config, 'sklearn': sklearn.__version__,
        'scope': 'original train only, same fitted scaler/labels/C=1 and frozen features; '
            'Newton-CG gradient <=1e-6, LBFGS reference <=1e-8, objective gap <=1e-5, probability gap <=0.005',
        'GPU': None, 'utc': datetime.now(timezone.utc).isoformat(),
        'code_commit': os.environ.get('PINPOINT_CODE_COMMIT')}
    write_json(output / 'solver_probe.json', report)
    write_json(output / 'manifest.json', {'status': 'complete' if report['passed'] else 'internal_health_failed',
        'config': config, 'passed': report['passed'], 'seconds': report['seconds'], 'GPU': None})
    if not report['passed']: raise RuntimeError('same-objective readout solver regression failed')


if __name__ == '__main__':
    main()
