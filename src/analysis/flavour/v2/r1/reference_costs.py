"""Keep reused SSL computation and interactive checks separate from batch fits."""

import argparse
import json
from pathlib import Path

from analysis.flavour.inventory import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads((args.root/'campaign.json').read_text())
    reference = json.loads((Path(contract['r0'])/'configs/s6_reference.json').read_text())['models']
    config = next(row['config'] for row in reference if row['name']=='s4_empty_mae3_n100000_s42')
    checkpoint = Path(config['pretrained_checkpoint'])
    source = json.loads((checkpoint.parent/'manifest.json').read_text())
    accounting_path = checkpoint.parents[2]/'campaign_accounting.json'
    campaign = json.loads(accounting_path.read_text())
    historical = {'checkpoint': str(checkpoint), 'source_manifest': str(checkpoint.parent/'manifest.json'),
        'status': source['status'], 'GPU': source['GPU'], 'train_only_unique_events': source['train_events'],
        'inner_reconstruction_probe_events': source['inner_reconstruction_probe_events'],
        'optimizer_steps': source['optimizer_steps'], 'operation_seconds': source['seconds'],
        'operation_device_hours': source['seconds']/3600, 'peak_MiB': source['peak_gpu_mib'],
        'shared_historical_allocation': next(a for a in campaign['allocations'] if a['name']=='pinpoint-s4-core-a'),
        'cost_interpretation': 'operation cost belongs to this reused initialization; shared allocation also ran other experiments and cannot be assigned entirely to this checkpoint'}
    interactive = []
    for path in sorted((args.root/'fit/interactive_a1').glob('B*_T*.json')):
        row = json.loads(path.read_text())
        interactive.append({'path': str(path), 'GPU': row['GPU'], 'seconds': row['seconds'],
                            'scope': 'successful real-fit operation; launcher and data loading excluded'})
    for path in sorted((args.root/'smoke/a1/experiments').glob('*/manifest.json')):
        row = json.loads(path.read_text())
        interactive.append({'path': str(path), 'GPU': row['GPU'], 'seconds': row['wall_seconds'],
                            'scope': 'Train-only runner check; operation wall time'})
    for path in sorted((args.root/'smoke/a1/diagnostics').glob('*/manifest.json')):
        row = json.loads(path.read_text())
        interactive.append({'path': str(path), 'GPU': row['GPU'], 'seconds': row['wall_seconds'],
                            'scope': 'Train-only checkpoint replay; operation wall time'})
    write_json(args.root/'reference_costs.json', {'status': 'complete', 'reused_MAE3': historical,
        'interactive_recorded_operations': interactive,
        'interactive_operation_seconds_lower_bound': sum(r['seconds'] for r in interactive),
        'interactive_launch_setup_and_unrecorded_failed_wall_seconds': None,
        'interactive_accounting_scope': 'measured operation lower bound; no allocation-hour claim for the shared interactive GPU',
        'batch_costs': 'resources.json includes all batch setup, failed attempts and allocated CPU/GPU time',
        'historical_SSL_is_not_new_R1_expenditure': True})


if __name__ == '__main__':
    main()
