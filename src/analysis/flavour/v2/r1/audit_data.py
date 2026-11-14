"""Summarize the frozen R1 denominators and observation feature coverage."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.flavour.inventory import write_json
from analysis.flavour.v2.r1.data import SUMMARY_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_parquet(args.root/'data/catalog.parquet')
    b0 = pd.read_parquet(args.root/'data/b0.parquet')
    rows = []
    for split, subset in frame.groupby('split'):
        channels = subset.pixel_channels+subset.bar_channels+subset.tracker_channels
        domains = {'all': np.ones(len(subset), dtype=bool),
            'pixel_centered': subset.pixel_centered.to_numpy(),
            'large_scintillator': subset.large_scintillator.to_numpy(),
            'active': ~subset.all_empty.to_numpy(), 'empty': subset.all_empty.to_numpy(),
            'tracker_active': subset.tracker_channels.gt(0).to_numpy(),
            'tracker_empty': subset.tracker_channels.eq(0).to_numpy()}
        for domain, mask in domains.items():
            current = subset.loc[mask]
            rows.append({'split': split, 'domain': domain, 'events': len(current),
                'original_label_counts': np.bincount(current.label, minlength=4).tolist()})
    features = {}
    for name in SUMMARY_NAMES:
        values = b0[name].to_numpy()
        features[name] = {'finite': bool(np.isfinite(values).all()),
            'range': [float(values.min()), float(values.max())],
            'quantiles_0p01_0p5_0p99': np.quantile(values, [.01,.5,.99]).tolist()}
    source = json.loads((args.root/'preparation.json').read_text())
    write_json(args.root/'data_audit.json', {'status': 'complete', 'domains': rows,
        'summary_feature_count': len(SUMMARY_NAMES), 'features': features,
        'all_summary_features_finite': all(v['finite'] for v in features.values()),
        'tau_branches': frame[frame.label.eq(2)].groupby(['split','tau_branch']).size().rename('events').reset_index().to_dict('records'),
        'observations_compressed_bytes': sum(s['parquet_bytes']['observations.parquet'] for s in source['source_summaries']),
        'PE_input': sum(s['PE_input'] for s in source['source_summaries']),
        'PE_output': sum(s['PE_output'] for s in source['source_summaries']),
        'Cal_A_Cut_B_Confirm_payload_read': False})


if __name__ == '__main__':
    main()
