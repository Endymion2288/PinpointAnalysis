"""Small block multiplicity design, reused across paired model comparisons."""
import json

import numpy as np
import pandas as pd

from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.s2.evaluate import resampling_groups
from analysis.flavour.s6.prepare import KEY


def design(root, frame, replicates=400, seed=2026):
    groups, summary = resampling_groups(frame)
    rng = np.random.default_rng(seed)
    ids = np.empty(len(frame), dtype='int32')
    samples, offset = [], 0
    for blocks in groups.values():
        values = np.asarray([np.bincount(rng.integers(len(blocks), size=len(blocks)), minlength=len(blocks))
            for _ in range(replicates)], dtype='uint16')
        samples.append(values)
        for j, indices in enumerate(blocks):
            ids[indices] = offset+j
        offset += len(blocks)
    keys = frame[KEY].copy()
    keys['bootstrap_block'] = ids
    keys.to_parquet(root/'evaluation/bootstrap_keys.parquet', index=False, compression=COMPRESSION)
    np.save(root/'evaluation/bootstrap_multiplicity.npy', np.concatenate(samples, axis=1))
    write_json(root/'evaluation/bootstrap_design.json', {**summary, 'replicates': replicates, 'seed': seed,
        'weights': 'each replicate normalized within true class to the specified scenario prior'})
    return summary


def load(root, frame):
    keys = pd.read_parquet(root/'evaluation/bootstrap_keys.parquet')
    aligned = frame[KEY].merge(keys, on=KEY, how='left', validate='one_to_one', sort=False)
    if aligned.bootstrap_block.isna().any():
        raise ValueError('bootstrap event keys missing')
    return np.load(root/'evaluation/bootstrap_multiplicity.npy', mmap_mode='r'), aligned.bootstrap_block.to_numpy(dtype=int)


def weights(labels, prior, multiplicity=None):
    y = np.asarray(labels)
    m = np.ones(len(y)) if multiplicity is None else np.asarray(multiplicity, dtype=float)
    count = np.bincount(y, weights=m, minlength=4)
    factor = np.divide(prior, count, out=np.zeros(4), where=count > 0)
    return m*factor[y]


def ci(values):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    return {'CI95': np.quantile(values[finite], [.025, .975]).tolist() if finite.any() else [None, None],
        'valid_replicates': int(finite.sum())}
