"""Measure end-to-end aggregate throughput for independent jobs on one GPU."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import torch

from analysis.flavour.inventory import write_json
from analysis.flavour.s4.stage_inputs import stage_shards
from analysis.flavour.v2.r2.campaign import config_from_r1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--r1',type=Path,required=True)
    parser.add_argument('--attempt',required=True)
    parser.add_argument('--counts',type=int,nargs='+',default=[1,2])
    parser.add_argument('--updates',type=int,default=256)
    args=parser.parse_args()
    torch.zeros(1,device='cuda:0').sum().item()
    r0=Path(json.loads((args.r1/'campaign.json').read_text())['r0'])
    precision='BF16_encoder_FP32_head' if torch.cuda.get_device_capability()[0]>=8 else 'FP32'
    folder=args.root/'audit'/f'parallel_profile_{args.attempt}'
    folder.mkdir(parents=True,exist_ok=False)
    scratch_parent=os.environ.get('PINPOINT_V2_SCRATCH','/tmp')
    results=[]
    with tempfile.TemporaryDirectory(prefix='r2-profile-',dir=scratch_parent) as scratch:
        paths=sorted(p for p in (r0/'profile_data/shards').iterdir() if (p/'observations.parquet').exists())
        stage=stage_shards(paths,Path(scratch)/'native',workers=3)
        env=dict(os.environ,PINPOINT_R2_SHARED_STAGE=stage['local_shard_root'])
        for phase,count in enumerate(args.counts):
            children=[]
            started=time.perf_counter()
            for i in range(count):
                config=config_from_r1(args.r1,'B1','T4',precision=precision)
                suffix=f'_p{phase}' if args.counts.count(count)>1 else ''
                name=f'profile_b1_{args.attempt}_n{count}{suffix}_w{i}'
                config.update(name=name,data_root=str(r0/'profile_data'),profile=True,
                    profile_updates=args.updates,passes=1)
                path=args.root/'configs'/f'{name}.json'
                write_json(path,config)
                log=(args.root/'logs'/f'{name}.log').open('w')
                child=subprocess.Popen([sys.executable,'-m','analysis.flavour.v2.r2.train',
                    '--root',str(args.root),'--config',str(path)],env=env,stdout=log,stderr=subprocess.STDOUT)
                children.append((name,child,log))
            exits=[]
            for name,child,log in children:
                exits.append(child.wait())
                log.close()
            seconds=time.perf_counter()-started
            records=[]
            for name,_,_ in children:
                path=args.root/'experiments'/name/'manifest.json'
                record=json.loads(path.read_text()) if path.exists() else {}
                records.append({k:record.get(k) for k in ['status','GPU','GPU_peak_MiB','events_per_train_second','counters','memory','wall_seconds']})
            results.append({'phase':phase,'concurrent_jobs':count,'exit_codes':exits,'wall_seconds':seconds,'workers':records,
                'aggregate_operation_events_per_second':sum(v.get('counters',{}).get('training_exposures',0) for v in records)/seconds,
                'sum_training_throughput':sum(v.get('events_per_train_second',0) for v in records)})
            write_json(folder/'results.json',{'status':'running','GPU':torch.cuda.get_device_name(),
                'precision':precision,'results':results,'Cal_A_Cut_B_Confirm_payload_read':False})
            if any(exits):
                raise RuntimeError('concurrent real-input profile failed; inspect child logs before training')
    write_json(folder/'results.json',{'status':'complete','GPU':torch.cuda.get_device_name(),
        'allocation_id':os.environ.get('PINPOINT_JOB_ID'),'precision':precision,'results':results,
        'scope':'Train-only real inputs and forward/backward/updates; no Tune or performance selection',
        'Cal_A_Cut_B_Confirm_payload_read':False})
    print(json.dumps({'attempt':args.attempt,'GPU':torch.cuda.get_device_name(),'results':results}),flush=True)


if __name__=='__main__':
    main()
