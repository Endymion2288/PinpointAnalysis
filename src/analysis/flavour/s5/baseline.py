"""Fixed-complexity CPU classifiers for measured moments and reconstructed objects."""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from analysis.flavour.inventory import write_json
from analysis.flavour.s2.train import metrics, save_parquet
from analysis.flavour.s4.dataset import LabelDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--task', type=int, default=0)
    args = parser.parse_args()
    config = json.loads(args.campaign.read_text())['experiments'][args.task]
    train = LabelDataset(args.root, 'train', config)
    validation = LabelDataset(args.root, 'validation', config)
    key = ['run', 'chunk', 'row']
    objects = pd.read_parquet(args.root/'objects.parquet', columns=key+['raw_features', 'object_features'])
    joined = [d.metadata[key].merge(objects, on=key, how='left', sort=False, validate='one_to_one') for d in [train, validation]]
    for variant, columns in [('moments', ['raw_features']), ('object_only', ['object_features']), ('moments_objects', ['raw_features', 'object_features'])]:
        name = f's5_hgb_{variant}_n{config["label_budget"]}_s{config["seed"]}'
        destination = args.root/'experiments'/name
        destination.mkdir(exist_ok=True)
        x, xv = [np.concatenate([np.stack(frame[c]) for c in columns], axis=1) for frame in joined]
        model = HistGradientBoostingClassifier(max_iter=160, max_leaf_nodes=15, learning_rate=.06,
                                              l2_regularization=2., early_stopping=False, random_state=config['seed'])
        started = time.monotonic()
        model.fit(x, train.metadata.label.to_numpy())
        probability = model.predict_proba(xv)
        joblib.dump(model, destination/'model.joblib', compress=3)
        loaded = joblib.load(destination/'model.joblib')
        difference = float(np.max(np.abs(loaded.predict_proba(xv)-probability)))
        if difference != 0:
            raise RuntimeError('object baseline replay differs')
        frame = validation.metadata.copy()
        for c in range(4):
            frame[f'p{c}'] = probability[:, c]
        save_parquet(frame, destination/'validation_predictions.parquet')
        report = metrics(frame.label.to_numpy(), probability)
        write_json(destination/'metrics.json', report)
        write_json(destination/'manifest.json', {'status': 'complete', 'config': {**config, 'name': name, 'variant': variant},
            'algorithm': 'fixed HistGradientBoosting 160 iterations, 15 leaves, no early stopping',
            'train_events': len(train), 'validation_events': len(validation), 'seconds': time.monotonic()-started,
            'maximum_replay_difference': difference, 'input': columns, 'calibration_test_used': False})
        print(json.dumps({'name': name, **report}), flush=True)
    train.native.close()
    validation.native.close()


if __name__ == '__main__':
    main()
