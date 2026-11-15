"""Measure paired uncertainty of the frozen two-task architecture ranking."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.flavour.inventory import write_json
from analysis.flavour.s3.metrics import bootstrap_weights
from analysis.flavour.v2.partitions import KEY, equal_identity_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    ranking = json.loads((root/'architecture_ranking.json').read_text())
    catalog = pd.read_parquet(root/'data/catalog.parquet', filters=[('split', '=', 'validation')])
    catalog = catalog.sort_values(KEY).reset_index(drop=True)
    y = catalog.label.to_numpy()
    weights, design = bootstrap_weights(catalog, 400, 20261116)
    point, samples, rows = {}, {}, {}
    for entry in ranking['ranking']:
        family = entry['family']
        correct = []
        for task in ['t3', 't4']:
            name = f'final_{family.lower()}_{task}_s42'
            manifest = json.loads((root/'experiments'/name/'manifest.json').read_text())
            if manifest['status'] != 'complete':
                raise ValueError('ranking uncertainty needs completed main models')
            frame = pd.read_parquet(root/'experiments'/name/'predictions.parquet').sort_values(KEY).reset_index(drop=True)
            if not equal_identity_rows(frame, catalog) or not np.array_equal(frame.label, y):
                raise ValueError('ranking comparison cohorts differ')
            # Conditional normalization cannot change the three-class argmax.
            scores = frame[['p0', 'p1', 'p3']].to_numpy()
            prediction = np.array([0, 1, 3])[scores.argmax(1)]
            correct.append(prediction == y)
        average = (correct[0].astype(float)+correct[1])/2
        point[family] = float(np.mean([average[y == c].mean() for c in [0, 1, 3]]))
        samples[family] = np.mean([weights[:, y == c]@average[y == c]/weights[:, y == c].sum(1)
                                   for c in [0, 1, 3]], axis=0)
        rows[family] = {'mean_BA3': point[family],
                        'agrees_with_frozen_ranking': abs(point[family]-entry['mean_common_BA3']) < 1e-12}
        if not rows[family]['agrees_with_frozen_ranking']:
            raise ValueError('direct ranking readback differs')
    pairs = []
    families = list(rows)
    for i, a in enumerate(families):
        for b in families[i+1:]:
            values = samples[a]-samples[b]
            pairs.append({'a': a, 'b': b, 'difference_a_minus_b': point[a]-point[b],
                          '95pct': np.quantile(values, [.025, .975]).tolist(),
                          'valid_replicates': len(values)})
    result = {'status': 'complete', 'models': rows, 'pairs': pairs,
              'replicates': 400, 'seed': 20261116, 'bootstrap': design,
              'scope': 'post-hoc pointwise development intervals for frozen mean T3/T4 conditional BA3; fixed fitted seed42; same physical-block draws across all models/tasks; ranking rule and selected families unchanged',
              'Cal_A_Cut_B_Confirm_payload_read': False}
    write_json(root/'evaluation/ranking_review.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
