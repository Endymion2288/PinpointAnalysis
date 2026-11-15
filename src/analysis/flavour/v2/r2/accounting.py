"""Count GPU allocations once and record independent worker concurrency explicitly."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from analysis.flavour.inventory import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--cached-kubeflow',action='store_true')
    args=parser.parse_args()
    command=[sys.executable,'-m','analysis.flavour.v2.r1.accounting','--root',str(args.root)]
    if args.cached_kubeflow:command.append('--cached-kubeflow')
    subprocess.run(command,check=True)
    groups=[json.loads(p.read_text()) for p in (args.root/'groups').glob('*.json')]
    resources=json.loads((args.root/'resources.json').read_text())
    for allocation in resources['allocations']:
        identity=allocation['name'] if allocation['backend']=='kubeflow' else str(allocation['cluster'])+'.0'
        matched=[g for g in groups if g.get('allocation_id')==identity]
        if matched:
            allocation['concurrent_workers']=[g['concurrent_jobs'] for g in matched]
            allocation['worker_configs']=[g['configs'] for g in matched]
        allocation['counting']='one allocation duration regardless of independent worker count'
    interactive=[]
    path=args.root/'audit/parallel_profile_interactive_t4_a1/results.json'
    if path.exists():
        report=json.loads(path.read_text())
        interactive.append({'scope':'interactive Train-only profile phases; launch and parent staging not included',
            'GPU':report['GPU'],'operation_lower_bound_seconds':sum(p['wall_seconds'] for p in report['results'])})
    for attempt in ['interactive_fp32_a1','interactive_fp32_a2']:
        folder=args.root/'smoke'/attempt
        records=[json.loads(p.read_text()) for p in (folder/'experiments').glob('*/manifest.json')]
        if records:
            seconds=sum(r['counters']['train_seconds']+r['counters']['validation_seconds'] for r in records)
            interactive.append({'attempt':attempt,'GPU':records[0]['GPU'],
                'operation_lower_bound_seconds':seconds,
                'scope':'sum distinct model train+validation including saved pre-resume counters; excludes other setup and checkpoint time'})
    resources.update(interactive_operations=interactive,
        interactive_operation_lower_bound_device_hours=sum(r['operation_lower_bound_seconds'] for r in interactive)/3600,
        interactive_total_allocation_seconds='not measured; operation lower bounds reported separately',
        worker_wall_seconds_are_not_summed_as_device_hours=True)
    write_json(args.root/'resources.json',resources)
    print(json.dumps({k:resources[k] for k in ['device_hours','device_hours_by_model_or_partition',
        'active_or_unresolved','interactive_operation_lower_bound_device_hours']}),flush=True)


if __name__=='__main__':main()
