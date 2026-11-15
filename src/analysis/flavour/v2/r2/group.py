"""Run independent scaling fits concurrently inside one obtained GPU allocation."""

import argparse
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import time

import torch

from analysis.flavour.inventory import write_json
from analysis.flavour.s4.stage_inputs import stage_shards


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--name',required=True)
    parser.add_argument('--configs',type=Path,nargs='+',required=True)
    parser.add_argument('--parallel',type=int,default=2)
    parser.add_argument('--real-fit',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--share-r3-inputs',action='store_true',
        help='Build exact allocation-local mmap Train/Tune arrays once for native R3 models')
    parser.add_argument('--cache-workers',type=int,default=4)
    parser.add_argument('--post-r3-probe-attempt',
        help='After completed R3 fits, reuse this allocation and its exact Train/Tune mmap arrays for final probes')
    parser.add_argument('--train-module',default='analysis.flavour.v2.r2.train',
        choices=['analysis.flavour.v2.r2.train','analysis.flavour.v2.r3.train'])
    args=parser.parse_args()
    if args.share_r3_inputs and args.train_module!='analysis.flavour.v2.r3.train':
        parser.error('shared R3 transformed inputs require the R3 runner')
    if args.post_r3_probe_attempt and not (args.share_r3_inputs and args.train_module=='analysis.flavour.v2.r3.train'):
        parser.error('post-fit R3 probes require the R3 runner and shared transformed inputs')
    configs=[json.loads(p.read_text()) for p in args.configs]
    if len({c['data_root'] for c in configs})!=1:
        raise ValueError('shared staging group requires one exact observation corpus')
    if not all(Path(c['preflight']).exists() for c in configs):
        raise ValueError('all exact sampler preflights must complete before allocation use')
    torch.zeros(1,device='cuda:0').sum().item()
    if any(c.get('bf16',True) for c in configs) and torch.cuda.get_device_capability()[0]<8:
        raise ValueError('BF16 group requires SM80; an FP32 group needs its own matched D0')
    if args.smoke or args.real_fit:
        r0=json.loads((args.root/'campaign.json').read_text())['r0']
    if args.smoke:
        if args.train_module=='analysis.flavour.v2.r3.train':
            for i,path in enumerate(args.configs):
                command=[sys.executable,'-m','analysis.flavour.v2.r3.smoke',
                    '--root',str(args.root),'--config',str(path),'--attempt',args.name+f'_c{i}']
                if args.share_r3_inputs:command.append('--shared-inputs')
                subprocess.run(command,check=True)
        else:
            command=[sys.executable,'-m','analysis.flavour.v2.r2.smoke','--root',str(args.root),
                '--r1',json.loads((args.root/'campaign.json').read_text())['r1'],
                '--r0',r0,'--attempt',args.name]
            if configs[0]['bf16']:command.append('--bf16')
            subprocess.run(command,check=True)
    if args.real_fit:
        families=sorted({c['family'] for c in configs})
        if len(families)!=1 or len({c.get('cnn_width',16) for c in configs})!=1 or len({c.get('dim',96) for c in configs})!=1:
            raise ValueError('real-fit group requires one family/width/dim recipe')
        command=[sys.executable,'-m','analysis.flavour.v2.r1.fit','--root',str(args.root),
            '--r0',r0,'--attempt',args.name,'--families',*families,
            '--cnn-width',str(configs[0].get('cnn_width',16)),'--dim',str(configs[0].get('dim',96))]
        if configs[0]['bf16']:command.append('--bf16')
        subprocess.run(command,check=True)
    started=time.perf_counter()
    record={'status':'running','GPU':torch.cuda.get_device_name(),'allocation_id':os.environ.get('PINPOINT_JOB_ID'),
        'concurrent_jobs':args.parallel,'configs':list(map(str,args.configs)), 'completed':[],
        'Cal_A_Cut_B_Confirm_payload_read':False}
    record_path=args.root/'groups'/f'{args.name}.json'
    write_json(record_path,record)
    parent=os.environ.get('PINPOINT_V2_SCRATCH','/tmp')
    post_code=0
    with tempfile.TemporaryDirectory(prefix='r2-group-',dir=parent) as scratch:
        paths=sorted(p for p in (Path(configs[0]['data_root'])/'shards').iterdir() if (p/'observations.parquet').exists())
        stage=stage_shards(paths,Path(scratch)/'native',workers=3)
        record['shared_staging']=stage
        env=dict(os.environ,PINPOINT_R2_SHARED_STAGE=stage['local_shard_root'])
        if args.share_r3_inputs:
            from analysis.flavour.v2.r3.data import R3Dataset
            keys=['family','task','native_pixel_pitch_mm','build_relations',
                'relation_plane_policy','image','sensor_row_image','drop_modality']
            signature={k:configs[0].get(k) for k in keys}
            if any({k:c.get(k) for k in keys}!=signature for c in configs):
                raise ValueError('shared R3 arrays require identical observation transforms and candidate graphs')
            os.environ['PINPOINT_NATIVE_SHARD_ROOT']=stage['local_shard_root']
            preparation={}
            for split,var in [('train','PINPOINT_R3_SHARED_TRAIN_INPUTS'),
                    ('validation','PINPOINT_R3_SHARED_TUNE_INPUTS')]:
                common=R3Dataset(Path(configs[0]['data_root']),split,
                    {**configs[0],'cache_training_inputs':False,'cache_storage':'mmap',
                     'cache_workers':args.cache_workers},cache_dir=scratch)
                common.cache_inputs()
                payload=Path(scratch)/(split+'_transformed.pkl')
                with payload.open('wb') as stream:
                    pickle.dump({'root':str(Path(configs[0]['data_root']).resolve()),
                        'input_config':signature,'keys':common.metadata[['run','chunk','row','label']].to_numpy(),
                        'items':common.items,'audit':common.cache_audit},stream,protocol=5)
                preparation[split]=common.cache_audit
                env[var]=str(payload)
                del common
                record['shared_transformed_preparation']=preparation
                write_json(record_path,record)
            del os.environ['PINPOINT_NATIVE_SHARD_ROOT']
        pending=list(zip(args.configs,configs))
        live=[]
        while pending or live:
            while pending and len(live)<args.parallel:
                path,config=pending.pop(0)
                manifest=args.root/'experiments'/config['name']/'manifest.json'
                previous=json.loads(manifest.read_text()) if manifest.exists() else None
                if previous and previous['status']=='complete':
                    record['completed'].append({'name':config['name'],'status':'previously_complete'})
                    continue
                command=[sys.executable,'-m',args.train_module,'--root',str(args.root),'--config',str(path)]
                if previous:command.append('--resume')
                log=(args.root/'logs'/f"{config['name']}_{args.name}.log").open('w')
                process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
                live.append((config['name'],process,log))
            remaining=[]
            for name,process,log in live:
                code=process.poll()
                if code is None:
                    remaining.append((name,process,log))
                else:
                    log.close()
                    record['completed'].append({'name':name,'exit_code':code})
                    if code:
                        # Other active writers keep their progress/checkpoints;
                        # the group stops scheduling new work after a failure.
                        pending=[]
                        record['status']='failed_child'
            live=remaining
            record['active']=[name for name,_,_ in live]
            record['wall_seconds']=time.perf_counter()-started
            write_json(record_path,record)
            if live:time.sleep(5)
        if args.post_r3_probe_attempt and not any(r.get('exit_code',0) for r in record['completed']):
            record['post_r3_probes']={'status':'running','attempt':args.post_r3_probe_attempt,
                'shared_transformed_inputs':True,'count_as_separate_allocation':False}
            write_json(record_path,record)
            post_code=subprocess.run([sys.executable,'-m','analysis.flavour.v2.r3.probes',
                '--root',str(args.root),'--experiments',*[str(args.root/'experiments'/c['name']) for c in configs],
                '--attempt',args.post_r3_probe_attempt,'--cache-workers',str(args.cache_workers)],env=env).returncode
            record['post_r3_probes'].update(status='complete' if post_code==0 else 'failed',exit_code=post_code)
    failed=[r for r in record['completed'] if r.get('exit_code',0)]
    record.update(status='failed' if failed or post_code else 'complete',wall_seconds=time.perf_counter()-started)
    write_json(record_path,record)
    if failed:raise RuntimeError('scaling child failed; inspect its log and saved progress')
    if post_code:raise RuntimeError('post-fit R3 probe failed; completed model endpoints retained')


if __name__=='__main__':
    main()
