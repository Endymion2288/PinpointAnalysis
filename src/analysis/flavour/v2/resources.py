"""Close R0 allocations using schedd history and explicit interactive records."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess

from analysis.flavour.inventory import write_json


FIELDS = ['ClusterId', 'ProcId', 'JobStatus', 'ExitCode', 'ExitBySignal', 'RemoteWallClockTime',
          'CommittedTime', 'RequestCpus', 'RequestMemory', 'RequestGPUs', 'MemoryUsage',
          'RemoteUserCpu', 'RemoteSysCpu', 'CumulativeRemoteUserCpu', 'CumulativeRemoteSysCpu',
          'JobStartDate', 'CompletionDate', 'RemoteHost', 'NumJobStarts', 'NumShadowStarts',
          'NumJobCompletions', 'GPUsDeviceName', 'AssignedGPUs']


def terminal(row):
    result = subprocess.run(['condor_history', '-name', row['schedd'], str(row['cluster']),
        '-limit', '1', '-json', '-attributes', ','.join(FIELDS)], capture_output=True, text=True, check=True)
    records = json.loads(result.stdout)
    if len(records) != 1 or records[0]['JobStatus'] not in [3, 4]:
        raise RuntimeError('R0 job is not terminal in history: '+row['name'])
    row['terminal'] = records[0]
    active = subprocess.check_output(['condor_q', '-name', row['schedd'], str(row['cluster']), '-json'], text=True)
    # Some schedd versions return empty stdout rather than [] for no jobs.
    row['absent_from_live_queue'] = not json.loads(active.strip() or '[]')
    if not row['absent_from_live_queue']:
        raise RuntimeError('R0 completed allocation remains live')
    return row


def run(root):
    submissions = []
    rejected = []
    for path in sorted((root/'runtime').glob('*.submission.json')):
        record = json.loads(path.read_text())
        if not record['submitted']:
            rejected.append({'name': record['name'], 'schedd': record['schedd'],
                             'reason': record['stderr'], 'allocated_device_hours': 0})
            continue
        cluster = int(re.search(r'cluster (\d+)', record['stdout']).group(1))
        submissions.append({'name': record['name'], 'schedd': record['schedd'], 'cluster': cluster})
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(terminal, submissions))
    allocation_core_seconds = 0.
    measured_cpu_seconds = 0.
    device_seconds = 0.
    device_hours_by_GPU = {}
    profiles = {'r0_stream_a1': 'profiles/stream_a1.json',
                'r0_stream_compiled_a1': 'profiles/stream_compiled_a1.json'}
    for row in rows:
        ad = row['terminal']
        allocation_core_seconds += ad.get('RemoteWallClockTime', 0.) * ad.get('RequestCpus', 0)
        measured_cpu_seconds += max(ad.get('CumulativeRemoteUserCpu', 0.)+ad.get('CumulativeRemoteSysCpu', 0.),
                                    ad.get('RemoteUserCpu', 0.)+ad.get('RemoteSysCpu', 0.))
        device_seconds += ad.get('RemoteWallClockTime', 0.) * ad.get('RequestGPUs', 0)
        if row['name'] in profiles:
            profile = json.loads((root/profiles[row['name']]).read_text())
            row['observed_GPU'] = {key: profile[key] for key in ['GPU', 'GPU_memory_MiB', 'CUDA_VISIBLE_DEVICES']}
            hours = ad['RemoteWallClockTime'] * ad['RequestGPUs'] / 3600
            device_hours_by_GPU[profile['GPU']] = device_hours_by_GPU.get(profile['GPU'], 0.) + hours
    interactive = []
    for name, file in [('micro_a2', 'profiles/micro_a2.json'), ('fit32_a2', 'fit/fit32_a2.json')]:
        record = json.loads((root/file).read_text())
        interactive.append({'name': name, 'GPU': record['GPU'], 'seconds': record['seconds'],
                            'status': record['status'], 'source': file})
    failed_fit = json.loads((root/'fit/fit32_a1.json').read_text())
    start = datetime.fromisoformat(failed_fit['UTC_started']).timestamp()
    # The traceback ends the failed fit. File mtime gives a timestamped bound,
    # not an independently sampled device allocation interval.
    upper = max(0., (root/'logs/fit32_a1.log').stat().st_mtime-start)+2.
    interactive += [
        {'name': 'fit32_a1', 'GPU': 'Tesla T4', 'seconds_upper_bound': upper,
         'status': 'invalid selection/output; included in cost', 'basis': 'recorded start to terminal traceback log mtime plus 2 s'},
        {'name': 'micro_a1', 'GPU': 'Tesla T4', 'seconds_upper_bound': 240.,
         'status': 'interrupted invalid memory monitor; included in cost',
         'basis': 'observed PID lifetime under 4 min; exact start/end allocation interval unmeasured'}]
    interactive_seconds = sum(row.get('seconds', row.get('seconds_upper_bound', 0.)) for row in interactive)
    result = {'status': 'complete', 'UTC': datetime.now(timezone.utc).isoformat(), 'batch_attempts': rows,
              'rejected_submissions': rejected, 'interactive_attempts': interactive,
              'HTCondor_allocated_CPU_core_hours': allocation_core_seconds/3600,
              'HTCondor_measured_CPU_hours': measured_cpu_seconds/3600,
              'HTCondor_device_hours': device_seconds/3600,
              'HTCondor_device_hours_by_GPU': device_hours_by_GPU,
              'interactive_T4_device_hours_including_failed_upper_bounds': interactive_seconds/3600,
              'total_device_hours_upper_bound': (device_seconds+interactive_seconds)/3600,
              'planning_envelope_device_hours': 2., 'different_GPU_hours_are_not_FLOP_equivalent': True,
              'all_submitted_jobs_terminal_and_absent': all(row['absent_from_live_queue'] for row in rows),
              'interactive_CPU_hours': 'unmeasured; batch CPU measurements and RAM profiles reported separately'}
    write_json(root/'resources.json', result)
    print(json.dumps({key: value for key, value in result.items() if key not in ['batch_attempts', 'interactive_attempts', 'rejected_submissions']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)


if __name__ == '__main__':
    main()
