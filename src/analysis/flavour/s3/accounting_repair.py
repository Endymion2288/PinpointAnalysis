"""S3 repair allocation plus the frozen prior S3 charge, under one 120-hour budget."""
import json
from pathlib import Path
import sys

from analysis.flavour.inventory import write_json
from analysis.flavour.s3.accounting import main as refresh_repair


def main():
    refresh_repair()
    root = Path(sys.argv[sys.argv.index('--root') + 1])
    report = json.loads((root / 'campaign_accounting.json').read_text())
    prior_root = root.parent / 's3_2026-11-12'
    prior = json.loads((prior_root / 'campaign_accounting.json').read_text())
    report['repair_device_hours_upper_bound'] = report['organization_GPU_hours_upper_bound']
    report['prior_S3_device_hours_upper_bound'] = prior['organization_GPU_hours_upper_bound']
    report['prior_S3_evidence'] = str(prior_root / 'campaign_accounting.json')
    report['organization_GPU_hours_upper_bound'] += prior['organization_GPU_hours_upper_bound']
    report['remaining_GPU_hours_lower_bound'] = 120 - report['organization_GPU_hours_upper_bound']
    report['scope'] = 'S3 revision2 GPU allocations; cumulative organization bound includes legacy S3; S2 excluded'
    write_json(root / 'campaign_accounting.json', report)
    print(json.dumps({k: report[k] for k in ['repair_device_hours_upper_bound',
        'prior_S3_device_hours_upper_bound', 'organization_GPU_hours_upper_bound',
        'remaining_GPU_hours_lower_bound', 'missing_history_records']}))


if __name__ == '__main__':
    main()
