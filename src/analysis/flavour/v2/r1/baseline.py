"""B0 HGB on pixel/bar/tracker observation summaries, with actual T3 retraining."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
import joblib

from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.v2.r1.data import SUMMARY_NAMES
from analysis.flavour.v2.r1.metrics import quick_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--task', choices=['T3', 'T4'], required=True)
    parser.add_argument('--learning-rate', type=float, required=True)
    parser.add_argument('--iterations', type=int, default=300)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    output = args.root/'experiments'/args.name
    output.mkdir(parents=True, exist_ok=False)
    table = pd.read_parquet(args.root/'data/b0.parquet')
    catalog = pd.read_parquet(args.root/'data/catalog.parquet')
    train = table[table.split.eq('train') & (table.label.ne(2) if args.task == 'T3' else True)]
    tune = table[table.split.eq('validation')]
    config = {'family': 'B0', 'task': args.task, 'seed': args.seed, 'learning_rate': args.learning_rate,
        'max_iter': args.iterations, 'max_leaf_nodes': 31, 'min_samples_leaf': 30, 'l2_regularization': 1.,
        'early_stopping': False, 'features': SUMMARY_NAMES, 'name': args.name}
    with threadpool_limits(limits=args.threads):
        model = HistGradientBoostingClassifier(learning_rate=args.learning_rate, max_iter=args.iterations,
            max_leaf_nodes=31, min_samples_leaf=30, l2_regularization=1., early_stopping=False, random_state=args.seed)
        tick = time.perf_counter()
        model.fit(train[SUMMARY_NAMES], train.label)
        fit_seconds = time.perf_counter()-tick
        p = model.predict_proba(tune[SUMMARY_NAMES])
        train_p = model.predict_proba(train[SUMMARY_NAMES])
    frame = catalog[catalog.split.eq('validation')].reset_index(drop=True)
    if not np.array_equal(frame[['run', 'chunk', 'row']].to_numpy(), tune[['run', 'chunk', 'row']].to_numpy()):
        raise ValueError('HGB prediction identities differ')
    for c, value in zip(model.classes_, p.T):
        frame[f'p{c}'] = value
    frame.to_parquet(output/'predictions.parquet', index=False, compression=COMPRESSION)
    coverage = train[['run', 'chunk', 'row', 'label']].copy()
    coverage['boosting_iterations'] = args.iterations
    coverage.to_parquet(output/'train_exposures.parquet', index=False, compression=COMPRESSION)
    joblib.dump(model, output/'model.joblib', compress=3)
    write_json(output/'config.json', config)
    write_json(output/'manifest.json', {'status': 'complete', 'config': config,
        'metrics': quick_report(tune.label.to_numpy(), p), 'train_metrics': quick_report(train.label.to_numpy(), train_p),
        'unique_train_events': len(train), 'fit_seconds': fit_seconds, 'wall_seconds': time.perf_counter()-started,
        'device': 'CPU', 'threads': args.threads, 'Cal_A_Cut_B_Confirm_payload_read': False,
        'probability_std': p.std(0).tolist(), 'distinct_probability_vectors': len(np.unique(p, axis=0))})
    print(json.dumps({'name': args.name, 'fit_seconds': fit_seconds, 'status': 'complete'}), flush=True)


if __name__ == '__main__':
    main()
