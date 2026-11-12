"""S3 backend allocations plus separately labelled conservative T4 bound."""
import argparse
import json
from pathlib import Path

from analysis.flavour.s2.accounting import run
from analysis.flavour.inventory import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    args.app_label = 'pinpoint-s3'
    run(args)
    path = args.root / 'campaign_accounting.json'
    report = json.loads(path.read_text())
    bound = json.loads((args.root / 'interactive_accounting_bound.json').read_text())
    report['interactive_initial_profile'] = bound
    report['organization_GPU_hours_upper_bound'] = report['GPU_hours'] + bound['GPU_hours_upper_bound']
    report['organization_budget_GPU_hours'] = 120
    report['remaining_GPU_hours_lower_bound'] = 120 - report['organization_GPU_hours_upper_bound']
    report['source'] = 'recorded S3 HTCondor ClusterIds and Kubeflow Job UIDs; S2 allocations separately reported'
    write_json(path, report)
    print(json.dumps({key: report[key] for key in ['utc', 'GPU_hours', 'organization_GPU_hours_upper_bound',
        'remaining_GPU_hours_lower_bound', 'missing_history_records']}))


if __name__ == '__main__':
    main()
