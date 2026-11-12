"""Count each S3 representation bottleneck on the same train occupancy ranks."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import torch

from analysis.flavour.inventory import write_json
from analysis.flavour.s3.dataset import PlaneDataset, plane_tokens, collate_planes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1); pa.set_cpu_count(1); pa.set_io_thread_count(1)
    configs = json.loads(args.campaign.read_text())['experiments']
    dataset = PlaneDataset(args.root, 'train', configs[0])
    key = ['run', 'chunk', 'row']
    catalog = pd.read_parquet(args.root / 'catalog.parquet',
        columns=key + ['pixel_channels', 'bar_channels', 'tracker_channels'])
    selected = dataset.metadata[key].merge(catalog, on=key, validate='one_to_one', how='left')
    workload = selected[['pixel_channels', 'bar_channels', 'tracker_channels']].sum(axis=1).to_numpy()
    order = np.argsort(workload, kind='stable')
    rows = []
    for rank in [0, (len(order) - 1) // 2, len(order) - 1]:
        index = int(order[rank])
        observations = dataset.native[int(dataset.indexes[index])]['observations']
        for task, config in enumerate(configs):
            nodes, groups, modules, _ = plane_tokens(observations, dataset.geometry, config, seed=index + 42)
            batch = collate_planes([{'nodes': nodes, 'groups': groups, 'modules': modules,
                                     'index': index, 'label': 0}])
            # The dummy label only satisfies the collator; no forward or training.
            count = len(batch['module_id'])
            module_tokens = count * config.get('module_queries', 4)
            manifest = json.loads((args.root / 'experiments' / config['name'] / 'manifest.json').read_text())
            rows.append({'task': task, 'experiment': config['name'],
                **{k: int(selected.iloc[index][k]) for k in key},
                'train_occupancy_rank': rank, 'train_events': len(dataset),
                'raw_all_detector_channels': int(workload[index]),
                'configured_modalities': config['modalities'],
                'represented_active_nodes': int((nodes[:, 15] > 0).sum()),
                'local_blocks_including_empty_placeholder': len(batch['nodes']),
                'physical_module_type_groups_including_empty_placeholder': count,
                'module_query_tokens': module_tokens,
                'global_tokens': config.get('latents', 24) if config['fusion'] == 'latent' else module_tokens,
                'fusion': config['fusion'], 'parameters': manifest['parameters']})
    dataset.native.close()
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.output / 'counts.parquet', index=False)
    write_json(args.output / 'summary.json', {'utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'same min/median/max train occupancy events, all frozen screen configurations; '
                 'CPU transforms only, no forward, fit or calibration/test use; not population quantiles of transformed tokens',
        'occupancy_order': 'sum of accepted pixel/bar/tracker channels; stable ascending order',
        'rows': rows})
    print(json.dumps({'rows': len(rows), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
