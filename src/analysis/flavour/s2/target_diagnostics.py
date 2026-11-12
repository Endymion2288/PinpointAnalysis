"""Describe truth target composition and conditional scores in accepted S2 data."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from analysis.flavour.inventory import write_json
from analysis.flavour.s2.evaluate import tau_metrics


def target_z(pdg):
    """Nuclear PDG 10LZZZAAAI; keep non-nuclear targets explicitly separate."""
    pdg = np.abs(np.asarray(pdg, dtype='int64'))
    return np.where(pdg >= 1_000_000_000, (pdg // 10000) % 1000,
                    np.where(pdg == 2212, 1, 0))


def run(args):
    paths = sorted(p for p in (args.root / 'shards').iterdir()
                   if (p / 'provenance.parquet').exists())
    metadata = pd.concat([pd.read_parquet(path / 'provenance.parquet',
        columns=['run', 'chunk', 'row', 'split', 'label', 'tgtPDG']) for path in paths],
        ignore_index=True)
    metadata['target_Z'] = target_z(metadata.tgtPDG)
    counts = metadata.groupby(['split', 'label', 'tgtPDG', 'target_Z']).size()
    counts = counts.rename('events').reset_index().to_dict('records')
    validation = metadata.loc[metadata.split.eq('validation')]
    results = []
    for path in sorted((args.root / 'experiments').iterdir()):
        manifest_path = path / 'manifest.json'
        if not manifest_path.exists() or json.loads(manifest_path.read_text())['status'] != 'complete':
            continue
        predictions = pd.read_parquet(path / 'validation_predictions.parquet',
                                      columns=['run', 'chunk', 'row', 'label', 'p2'])
        frame = predictions.merge(validation.drop(columns=['split']),
            on=['run', 'chunk', 'row', 'label'], validate='one_to_one', how='left')
        if len(frame) != len(validation) or frame.target_Z.isna().any():
            raise ValueError(f'{path.name}: target diagnostic validation keys differ')
        for z, group in frame.groupby('target_Z', sort=True):
            results.append({'experiment': path.name, 'target_Z': int(z),
                'class_counts': np.bincount(group.label, minlength=4).tolist(),
                **tau_metrics(group)})
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(counts).to_csv(args.output / 'target_counts.csv', index=False)
    write_json(args.output / 'summary.json', {
        'utc': datetime.now(timezone.utc).isoformat(),
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'scope': 'posthoc truth-target composition and conditional validation diagnostics; accepted membership unchanged',
        'model_input': False, 'calibration_test_used': False,
        'train_events': int(metadata.split.eq('train').sum()),
        'validation_events': len(validation),
        'target_counts': counts, 'conditional_metrics': results,
        'grouping': 'nuclear Z from absolute PDG; proton Z=1; other non-nuclear targets Z=0; isotopes retained in count table',
        'common_support_scope': 'main comparisons match fixed coarse energy/vertex bins; target is not an additional matching variable',
        'uncertainty': 'conditional point estimates only; per-target differences and small-tail intervals unmeasured',
    })
    print(json.dumps({'events': len(metadata), 'isotopes_or_PDG_targets': int(metadata.tgtPDG.nunique()),
                      'target_Z_groups': int(metadata.target_Z.nunique()),
                      'model_target_groups': len(results)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
