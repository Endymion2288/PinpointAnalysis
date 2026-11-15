"""Actual R3 model profiles, independent-model concurrency and full Tune peak."""

import argparse
import gc
import json
import os
import pickle
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import torch

from analysis.flavour.inventory import write_json
from analysis.flavour.v2.partitions import KEY, equal_identity_rows
from analysis.flavour.v2.profile import MemoryMonitor, shutdown_loader
from analysis.flavour.v2.r1.train import predict
from analysis.flavour.v2.r2.train import subset
from analysis.flavour.v2.r3.data import R3Dataset
from analysis.flavour.v2.r3.models import build_model
from analysis.flavour.v2.r3.train import make_loader


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--configs',type=Path,nargs='+',required=True)
    parser.add_argument('--attempt',required=True)
    parser.add_argument('--updates',type=int,default=256)
    parser.add_argument('--fit',action='store_true')
    parser.add_argument('--phase',action='store_true')
    parser.add_argument('--shared-inputs',action='store_true')
    parser.add_argument('--cache-workers',type=int,default=4)
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.zeros(1,device='cuda:0').sum().item()
    campaign=json.loads((args.root/'campaign.json').read_text())
    bf16=torch.cuda.get_device_capability()[0]>=8
    output=args.root/'audit'/f'profile_{args.attempt}'
    output.mkdir(parents=True,exist_ok=False)
    if args.phase:
        subprocess.run([sys.executable,'-m','analysis.flavour.v2.r3.phase',
            '--root',str(args.root),'--attempt',args.attempt],check=True)
    configs=[json.loads(p.read_text()) for p in args.configs]
    if args.fit:
        for i,path in enumerate(args.configs):
            command=[sys.executable,'-m','analysis.flavour.v2.r3.fit','--root',str(args.root),
                '--config',str(path),'--attempt',args.attempt+f'_c{i}','--tasks','T3','T4']
            if not bf16:command.append('--fp32')
            subprocess.run(command,check=True)
    shared_inputs=None;shared_audit=None
    if args.shared_inputs:
        keys=['family','native_pixel_pitch_mm','build_relations','relation_plane_policy']
        signature={key:configs[0].get(key) for key in keys}
        if any({key:c.get(key) for key in keys}!=signature for c in configs):
            raise ValueError('shared profile inputs require identical observation transforms and candidate graphs')
        parent=os.environ.get('PINPOINT_V2_SCRATCH','/tmp')
        temporary=tempfile.TemporaryDirectory(prefix='r3-profile-inputs-',dir=parent)
        common=R3Dataset(Path(campaign['r0'])/'profile_data','train',
            {**configs[0],'cache_training_inputs':True,'cache_workers':args.cache_workers})
        shared_audit=common.cache_audit
        shared_inputs=Path(temporary.name)/'train_inputs.pkl'
        with shared_inputs.open('wb') as stream:
            pickle.dump({'root':str((Path(campaign['r0'])/'profile_data').resolve()),
                'input_config':signature,'keys':common.metadata[['run','chunk','row','label']].to_numpy(),
                'audit':shared_audit,'items':common.items},stream,protocol=5)
        del common;gc.collect()
        write_json(output/'shared_inputs.json',shared_audit)
    results=[]
    phases=[[i] for i in range(len(configs))]+[list(range(len(configs)))]
    for phase,active in enumerate(phases):
        children=[]
        monitor=MemoryMonitor()
        monitor.__enter__()
        started=time.perf_counter()
        for i in active:
            config={**configs[i],'profile':True,'profile_updates':args.updates,'passes':1,
                'bf16':bf16,'precision':'BF16_encoder_FP32_head' if bf16 else 'FP32',
                'data_root':str(Path(campaign['r0'])/'profile_data'),
                'name':f'profile_{args.attempt}_p{phase}_c{i}'}
            path=output/f'p{phase}_c{i}.json'
            write_json(path,config)
            log=(output/f'p{phase}_c{i}.log').open('w')
            env=dict(os.environ)
            if shared_inputs:env['PINPOINT_R3_PROFILE_INPUTS']=str(shared_inputs)
            process=subprocess.Popen([sys.executable,'-m','analysis.flavour.v2.r3.train',
                '--root',str(args.root),'--config',str(path)],env=env,stdout=log,stderr=subprocess.STDOUT)
            children.append((config,process,log))
        for config,process,log in children:
            code=process.wait();log.close()
            if code:raise RuntimeError('R3 real-input profile failed; inspect immutable phase log')
        elapsed=time.perf_counter()-started
        monitor.__exit__(None,None,None)
        workers=[]
        for config,_,_ in children:
            record=json.loads((args.root/'experiments'/config['name']/'manifest.json').read_text())
            workers.append({k:record.get(k) for k in ['config','counters','GPU_peak_MiB','memory','wall_seconds','events_per_train_second','training_cache']})
        results.append({'phase':phase,'concurrent_jobs':len(active),'wall_seconds':elapsed,'workers':workers,
            'aggregate_memory':monitor.peak,
            'aggregate_operation_events_per_second':sum(x['counters']['training_exposures'] for x in workers)/elapsed})
        write_json(output/'results.json',{'status':'running','results':results})
    # The validation rows are the same frozen40k. Observe finite forward passes
    # and memory only; random-initialization scores do not select the model.
    common_validation=None
    validation_preparation=None
    if shared_inputs:
        with MemoryMonitor() as preparation_memory:
            common_validation=R3Dataset(Path(campaign['r1'])/'data','validation',
                {**configs[0],'cache_training_inputs':False,'cache_workers':args.cache_workers})
            common_validation.cache_inputs()
            validation_preparation={**common_validation.cache_audit,'memory':preparation_memory.peak}
        write_json(output/'shared_validation_inputs.json',validation_preparation)
    validation=[]
    for i,original in enumerate(configs):
        config={**original,'bf16':bf16,'cache_training_inputs':False}
        with MemoryMonitor() as memory:
            data=common_validation if common_validation is not None else R3Dataset(Path(campaign['r1'])/'data','validation',config)
            target=pd.read_parquet(args.root/'data/catalog.parquet',filters=[('split','=','validation')]).sort_values(KEY).reset_index(drop=True)
            if not equal_identity_rows(data.metadata.sort_values(KEY).reset_index(drop=True),target):
                raise ValueError('full-validation memory probe changed Tune identities')
            loader,_,_=make_loader(data,config,False)
            torch.manual_seed(config['seed'])
            model=build_model(config).cuda().eval()
            torch.cuda.reset_peak_memory_stats()
            indices,p,logits,seconds=predict(model,loader,config['task'],bf16)
            if len(np.unique(indices))!=40000 or not np.isfinite(logits).all():
                raise ValueError('full-validation forward/coverage probe failed')
            validation.append({'config':original['name'],'events':len(indices),'seconds':seconds,
                'GPU_peak_MiB':torch.cuda.max_memory_allocated()/2**20,'memory':memory.peak,
                'finite':True,'classification_metrics_used_for_selection':False})
            shutdown_loader(loader);data.native.close();del model
            torch.cuda.empty_cache()
    record={'status':'complete','GPU':torch.cuda.get_device_name(),'BF16':bf16,
        'allocation_id':os.environ.get('PINPOINT_JOB_ID'),'results':results,'full_validation':validation,
        'shared_input_preparation':shared_audit,
        'shared_validation_preparation':validation_preparation,
        'Cal_A_Cut_B_Confirm_payload_read':False}
    write_json(output/'results.json',record)
    if shared_inputs:temporary.cleanup()
    print(json.dumps({'status':'complete','profile':str(output),'GPU':record['GPU']}),flush=True)


if __name__=='__main__':
    main()
