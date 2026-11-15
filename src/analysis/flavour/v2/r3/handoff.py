"""Profile a replacement claim; resume formal writers only after terminal proof."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from analysis.flavour.inventory import write_json


def validate_release(release, allocation, names):
    if release.get('allocation_id')!=allocation or set(release.get('models',[]))!=set(names):
        raise ValueError('replacement release does not identify this allocation and exact model group')
    previous=release.get('previous_allocations',[])
    if not previous or not all(p.get('terminal_confirmed') and p.get('JobStatus') in [3,4,7] for p in previous):
        raise ValueError('replacement writers require confirmed terminal previous allocations')
    covered=[name for p in previous for name in p.get('models',[])]
    if set(covered)!=set(names) or len(covered)!=len(names) or len({p['id'] for p in previous})!=len(previous):
        raise ValueError('terminal proof must cover every previous model writer exactly once')
    if not release.get('previous_writers_terminal') or not release.get('completion_time_decision'):
        raise ValueError('release requires writer termination and a measured completion-time decision')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--configs',type=Path,nargs='+',required=True)
    parser.add_argument('--profile-configs',type=Path,nargs='+',required=True)
    parser.add_argument('--attempt',required=True)
    parser.add_argument('--parallel',type=int,default=3)
    parser.add_argument('--cache-workers',type=int,default=16)
    parser.add_argument('--profile-updates',type=int,default=256)
    parser.add_argument('--release-timeout',type=float,default=600.)
    parser.add_argument('--completed-probes',type=Path,nargs='*',default=[])
    parser.add_argument('--probe-attempt')
    args=parser.parse_args()
    if args.completed_probes and not args.probe_attempt:parser.error('completed probes require a unique probe attempt')
    configs=[json.loads(p.read_text()) for p in args.configs]
    if len({c['name'] for c in configs})!=len(configs):raise ValueError('duplicate formal writer')
    probes=[json.loads(p.read_text()) for p in args.profile_configs]
    keys=['family','task','image','cnn_width','cnn_first_stride','sensor_row_image','dim','bf16',
        'native_relation_adapter','native_relations','native_global_ablation','native_relation_revision',
        'native_pixel_pitch_mm','relation_material_context','relation_plane_policy']
    if len(probes)!=len(configs) or any(any(a.get(k)!=b.get(k) for k in keys) for a,b in zip(configs,probes)):
        raise ValueError('replacement profiles must use each actual formal architecture and precision')
    names=[c['name'] for c in configs]
    profile_attempt='handoff_'+args.attempt
    subprocess.run([sys.executable,'-m','analysis.flavour.v2.r3.profile','--root',str(args.root),
        '--configs',*map(str,args.profile_configs),'--attempt',profile_attempt,
        '--updates',str(args.profile_updates),'--shared-inputs','--cache-workers',str(args.cache_workers),
        '--train-only'],check=True)
    profile_path=args.root/'audit'/('profile_'+profile_attempt)/'results.json'
    profile=json.loads(profile_path.read_text())
    if profile['status']!='complete':raise ValueError('replacement profile incomplete')
    path=args.root/'runtime'/(args.attempt+'.handoff.json')
    release_path=args.root/'runtime'/(args.attempt+'.release.json')
    allocation=os.environ['PINPOINT_JOB_ID']
    record={'status':'ready','UTC':datetime.now(timezone.utc).isoformat(),
        'allocation_id':allocation,'GPU':profile['GPU'],'models':names,'profile':str(profile_path),
        'release_timeout_seconds':args.release_timeout,'Cal_A_Cut_B_Confirm_payload_read':False}
    write_json(path,record);print(json.dumps(record),flush=True)
    probe=None
    shared_path=args.root/'audit'/('shared_allocation_probe_'+args.attempt+'.json')
    shared={'status':'running','allocation_id':allocation,'GPU':profile['GPU'],
        'operation':'completed fixed-weight readability/checkpoint probe during handoff wait and native cache preparation',
        'experiments':list(map(str,args.completed_probes)),'count_as_separate_allocation':False,
        'Cal_A_Cut_B_Confirm_payload_read':False}
    if args.completed_probes:
        write_json(shared_path,shared)
        probe=subprocess.Popen([sys.executable,'-m','analysis.flavour.v2.r3.probes','--root',str(args.root),
            '--experiments',*map(str,args.completed_probes),'--attempt',args.probe_attempt,'--cache-workers','4'])

    def finish_probe():
        if probe is not None:
            code=probe.wait()
            shared.update(status='complete' if code==0 else 'failed',exit_code=code)
            write_json(shared_path,shared)
            if code:raise RuntimeError('completed-model auxiliary probe failed; inspect the saved log')
    started=time.monotonic()
    while not release_path.exists():
        if time.monotonic()-started>=args.release_timeout:
            finish_probe();record['status']='expired_without_release';write_json(path,record);return
        time.sleep(5)
    release=json.loads(release_path.read_text())
    if release.get('action')=='decline':
        if release.get('allocation_id')!=allocation:raise ValueError('decline belongs to a different claim')
        finish_probe();record.update(status='declined',decision=release);write_json(path,record);return
    validate_release(release,allocation,names)
    previous={p['id'] for p in release['previous_allocations']}
    for name in names:
        manifest=json.loads((args.root/'experiments'/name/'manifest.json').read_text())
        if manifest['allocation_id'] not in previous:
            raise ValueError('formal model now belongs to a writer absent from the terminal proof')
    record.update(status='released',release=release);write_json(path,record)
    subprocess.run([sys.executable,'-m','analysis.flavour.v2.r2.group','--root',str(args.root),
        '--name',args.attempt,'--configs',*map(str,args.configs),'--parallel',str(args.parallel),
        '--smoke','--share-r3-inputs','--cache-workers',str(args.cache_workers),
        '--train-module','analysis.flavour.v2.r3.train'],check=True)
    finish_probe()
    record['status']='complete';write_json(path,record)


if __name__=='__main__':main()
