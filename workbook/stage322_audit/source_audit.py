"""Recover Clean v2 parents from the production source tags and inspect ROOT targets."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import uproot

VARS = ['evtID', 'tgtPDG', 'initPDG', 'fslPDG', 'initE', 'initX', 'initY', 'initZ',
        'Q2', 'W', 'xBj', 'y', 'intType', 'scatteringType']


def parent_key(tag, row):
    digest = hashlib.blake2b(f'{tag}:{row}'.encode(), digest_size=16).digest()
    pair = np.frombuffer(digest, dtype='<u8') & np.uint64(2**63 - 1)
    return int(pair[1]), int(pair[0])


def read_one(source, tag, selected):
    _, run, chunk = tag.split(':')
    path = source / run / f'{run}_{chunk}.root'
    with uproot.open(path, object_cache=None, array_cache=None) as handle:
        frame = pd.DataFrame(handle['event'].arrays(VARS, library='np'))
    pairs = pd.DataFrame([parent_key(tag, i) for i in range(len(frame))],
                         columns=['family_hi', 'family_lo'], dtype=np.uint64)
    return selected.merge(pd.concat([pairs, frame], axis=1), on=['family_hi', 'family_lo'],
                          how='left', validate='one_to_one')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('clean', type=Path)
    parser.add_argument('source', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    cols = ['run', 'chunk', 'row', 'family_hi', 'family_lo', 'source_tag', 'split',
            'label', 'Ev_GeV', 'geom_weight']
    manifest = pd.read_parquet(args.clean / 'planned_manifest.parquet', columns=cols)
    pinpoint = manifest[manifest.source_tag.str.startswith('pinpoint:')]
    jobs = list(pinpoint.groupby('source_tag'))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        parents = pd.concat(pool.map(lambda group: read_one(args.source, *group), jobs),
                            ignore_index=True)
    assert parents.tgtPDG.notna().all()
    parents['Z'] = (parents.tgtPDG.astype(np.int64) // 10000) % 1000
    parents['is_W'] = parents['Z'] == 74
    others = parents[~parents.is_W]
    keys = ['tgtPDG', 'initPDG', 'fslPDG', 'initE', 'Q2', 'W', 'xBj', 'y', 'intType', 'scatteringType']
    duplicates = parents[parents.duplicated(keys, keep=False)]
    match_g = duplicates.groupby(keys, dropna=False)
    summary = {
        'joined_parents': len(parents),
        'selection_non_W': len(others),
        'independent_source_energy_check_max_abs_GeV': float(np.max(np.abs(parents.Ev_GeV - parents.initE / 1000.))),
        'non_W_distribution_by_target': others.tgtPDG.value_counts().to_dict(),
        'non_W_by_class_split': pd.crosstab(others['split'], others.label).to_dict(),
        'non_W_by_class': others.groupby('label').size().to_dict(),
        'non_W_confirm_fraction': float((others['split'] == 'confirm').sum() /
                                        (manifest['split'] == 'confirm').sum()),
        'cross_split_fingerprint_groups': int((match_g.split.nunique() > 1).sum()),
        'cross_source_fingerprint_groups': int((match_g.source_tag.nunique() > 1).sum()),
        'repeated_fingerprint_events': int(len(duplicates)),
    }
    (args.out / 'source_material.json').write_text(json.dumps(summary, indent=2))
    cols = ['run', 'chunk', 'row', 'tgtPDG', 'Z', 'initPDG', 'is_W']
    parents[cols].to_parquet(args.out / 'source_material_keys.parquet', index=False)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
