"""S3 backend allocations plus separately labelled conservative T4 bound."""
import argparse
from datetime import datetime
import json
from pathlib import Path

from analysis.flavour.s2.accounting import run
from analysis.flavour.inventory import write_json


def shared_seconds(start, end, intervals):
    """Split a reservation equally among active processes, conserving all time.

    Startup/teardown gaps are divided equally among all participants. These
    are allocation shares, never measured GPU busy time or whole-GPU equivalents.
    """
    points = sorted({start, end, *(max(start, min(end, t))
                      for a, b in intervals.values() for t in [a, b])})
    shares = {name: 0. for name in intervals}
    for a, b in zip(points, points[1:]):
        active = [name for name, (lo, hi) in intervals.items() if lo < b and hi > a]
        recipients = active or list(intervals)
        for name in recipients:
            shares[name] += (b - a) / len(recipients)
    return shares


def allocation_transform(root, allocations):
    directory = root / 'kubeflow'
    replacements = {}
    for path in directory.glob('allocation_evidence_*.json'):
        for row in json.loads(path.read_text())['allocations']:
            replacements[row['job']] = row
    allocations = [replacements.get(row.get('job'), row)
                   if row['seconds'] is None or row['status'] not in [3, 4] else row
                   for row in allocations]
    output = []
    for row in allocations:
        job = row.get('job', '')
        condor_parallel = False
        if 'submission' in row:
            job = json.loads(Path(row['submission']).read_text()).get('parallel_name') or job
            condor_parallel = bool(job)
        record = directory / f'parallel_{job}.json'
        if not record.exists() or row['seconds'] is None:
            output.append(row)
            continue
        processes = json.loads(record.read_text())['processes']
        if not processes or any('finished_at' not in p for p in processes):
            output.append(row)
            continue
        intervals = {p['experiment']: tuple(datetime.fromisoformat(p[k]).timestamp()
                     for k in ['started_at', 'finished_at']) for p in processes}
        if condor_parallel:
            start = min(a for a,b in intervals.values())
            end = max(b for a,b in intervals.values())
        else:
            start = datetime.fromisoformat(row['started_at']).timestamp()
            end = datetime.fromisoformat(row['finished_at']).timestamp() if row.get('finished_at') else start + row['seconds']
        shares = shared_seconds(start, end, intervals)
        for name, seconds in shares.items():
            denominator = end-start if condor_parallel else row['seconds']
            fraction = seconds / denominator if denominator else 1 / len(shares)
            shared = {**row, 'experiment': name, 'seconds': row['seconds']*fraction,
                      'shared_job_total_seconds': row['seconds'],
                      'allocation_fraction': fraction,
                      'reservation_allocation_basis': row.get('allocation_basis'),
                      'allocation_basis': ('remote_wallclock_charged_by_normalized_active_process_interval_shares_including_setup'
                        if condor_parallel else 'equal_share_of_active_process_intervals_with_equal_startup_teardown_shares'),
                      'process_interval_evidence': str(record)}
            for key in ['GPU_hours', 'CPU_core_hours']:
                shared[key] = row[key] * fraction
            if 'GPU_resource_hours' in row:
                shared['GPU_resource_hours'] = {k: v * fraction for k, v in row['GPU_resource_hours'].items()}
            output.append(shared)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    args.app_label = 'pinpoint-s3'
    args.allocation_transform = allocation_transform
    run(args)
    path = args.root / 'campaign_accounting.json'
    report = json.loads(path.read_text())
    bound = json.loads((args.root / 'interactive_accounting_bound.json').read_text())
    report['interactive_initial_profile'] = bound
    report['organization_GPU_hours_upper_bound'] = report['GPU_hours'] + bound['GPU_hours_upper_bound']
    report['organization_budget_GPU_hours'] = 120
    report['remaining_GPU_hours_lower_bound'] = 120 - report['organization_GPU_hours_upper_bound']
    report['bounded_allocation_rows'] = sum(
        row.get('allocation_estimate_kind') == 'conservative_upper_bound' for row in report['allocations'])
    report['cost_interpretation'] = (
        'GPU_hours includes explicitly labelled conservative upper bounds where terminal Pod history '
        'was lost; shared reservations counted once; GPU/MIG device-hours are not throughput equivalents')
    report['source'] = 'recorded S3 HTCondor ClusterIds and Kubeflow Job UIDs; S2 allocations separately reported'
    write_json(path, report)
    print(json.dumps({key: report[key] for key in ['utc', 'GPU_hours', 'organization_GPU_hours_upper_bound',
        'remaining_GPU_hours_lower_bound', 'missing_history_records']}))


if __name__ == '__main__':
    main()
