"""Freeze a common Train diagnostic set and measure endpoint Train/Tune gaps."""

import argparse
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd
import torch

from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.s4.stage_inputs import stage_shards
from analysis.flavour.v2.partitions import KEY
from analysis.flavour.v2.profile import MemoryMonitor, shutdown_loader
from analysis.flavour.v2.r1.data import R1Dataset
from analysis.flavour.v2.r1.models import build_model
from analysis.flavour.v2.r1.train import predict, save_predictions
from analysis.flavour.v2.r2.evaluate import domains
from analysis.flavour.v2.r1.evaluate import domain_report
from analysis.flavour.v2.r2.train import make_loader, subset


def plan(root):
    folder=root/'gap'
    folder.mkdir(exist_ok=True)
    if (folder/'selection.json').exists():raise FileExistsError('Train gap selection already frozen')
    campaign=json.loads((root/'campaign.json').read_text())
    source=Path(campaign['r1'])/'data'
    catalog=pd.read_parquet(source/'catalog.parquet',filters=[('split','=','train')]).sort_values(KEY).reset_index(drop=True)
    rng=np.random.default_rng(2026111502)
    selected=pd.concat([catalog.iloc[rng.choice(np.flatnonzero(catalog.label.eq(c)),1000,replace=False)] for c in range(4)])
    selected=selected.sort_values(KEY).reset_index(drop=True)
    selected.to_parquet(folder/'train_screen.parquet',index=False,compression=COMPRESSION)
    write_json(folder/'selection.json',{'status':'frozen','events':len(selected),'per_class':1000,'seed':2026111502,
        'source':str(source),'selection':'uniform deterministic sample within label from common nested Train100k',
        'scope':'post-training eval-mode gap diagnostic only; no checkpoint selection or optimizer updates',
        'actual_Tune_used':False,'Cal_A_Cut_B_Confirm_payload_read':False})


def run(root,names,attempt):
    torch.set_num_threads(1)
    torch.zeros(1,device='cuda:0').sum().item()
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    started=time.perf_counter()
    folder=root/'gap'
    spec=json.loads((folder/'selection.json').read_text())
    data_root=Path(spec['source'])
    selected=pd.read_parquet(folder/'train_screen.parquet')
    cases=[]
    parent=os.environ.get('PINPOINT_V2_SCRATCH',os.environ.get('_CONDOR_SCRATCH_DIR','/tmp'))
    with tempfile.TemporaryDirectory(prefix='r2-gap-',dir=parent) as scratch,MemoryMonitor() as memory:
        # Dataset metadata/native boundaries still cover the original corpus;
        # stage all compressed shards once, with bounded decoded caches.
        paths=sorted(p for p in (data_root/'shards').iterdir() if (p/'observations.parquet').exists())
        staging=stage_shards(paths,Path(scratch)/'native')
        os.environ['PINPOINT_NATIVE_SHARD_ROOT']=staging['local_shard_root']
        for name in names:
            experiment=root/'experiments'/name
            manifest=json.loads((experiment/'manifest.json').read_text())
            if manifest['status']!='complete':raise ValueError('gap inference requires completed weights')
            config=manifest['config']
            if config['bf16'] and torch.cuda.get_device_capability()[0]<8:
                raise ValueError('BF16 gap inference requires SM80')
            data=R1Dataset(data_root,'train',config,cache_dir=scratch)
            rows=selected[selected.label.ne(2)] if config['task']=='T3' else selected
            subset(data,rows)
            loader,_,_=make_loader(data,config,False)
            model=build_model(config).cuda()
            for kind,checkpoint in [('endpoint','exposure.pt'),('best_screen','best.pt')]:
                destination=folder/name/kind
                destination.mkdir(parents=True,exist_ok=False)
                state=torch.load(experiment/checkpoint,map_location='cpu',weights_only=False)
                if kind=='best_screen' and manifest['best_screen']['updates']==manifest['counters']['updates']:
                    if not all(torch.equal(value,endpoint_weights[key]) for key,value in state['model'].items()):
                        raise ValueError('same-update endpoint/best weights differ')
                    (destination/'predictions.parquet').symlink_to('../endpoint/predictions.parquet')
                    record={**endpoint_report,'checkpoint':kind,'seconds':0.,'reuses_same_update_endpoint':True}
                    write_json(destination/'report.json',record)
                    cases.append({'name':name,'checkpoint':kind,'events':len(data),'seconds':0.,'reuses_endpoint':True})
                    continue
                model.load_state_dict(state['model'])
                indices,p,logits,seconds=predict(model,loader,config['task'],config['bf16'])
                save_predictions(destination/'predictions.parquet',data,indices,p,logits)
                frame=data.metadata.iloc[indices].reset_index(drop=True)
                for c,values in zip([0,1,3] if p.shape[1]==3 else [0,1,2,3],p.T):frame[f'p{c}']=values
                reports={domain:domain_report(frame,mask) for domain,mask in domains(frame).items()}
                record={'status':'complete','name':name,'checkpoint':kind,'events':len(frame),'seconds':seconds,
                    'GPU':torch.cuda.get_device_name(),'precision':config['precision'],'reports':reports,
                    'scope':'common Train sample in eval mode; compare with same-checkpoint Tune, not dropout rolling loss',
                    'Cal_A_Cut_B_Confirm_payload_read':False}
                write_json(destination/'report.json',record)
                if kind=='endpoint':endpoint_weights,endpoint_report=state['model'],record
                cases.append({'name':name,'checkpoint':kind,'events':len(frame),'seconds':seconds})
            shutdown_loader(loader)
            data.native.close()
            del model,state
            torch.cuda.empty_cache()
        del os.environ['PINPOINT_NATIVE_SHARD_ROOT']
    write_json(folder/f'{attempt}.json',{'status':'complete','cases':cases,'GPU':torch.cuda.get_device_name(),
        'allocation_id':os.environ.get('PINPOINT_JOB_ID'),'wall_seconds':time.perf_counter()-started,
        'staging':staging,'memory':memory.peak,'Cal_A_Cut_B_Confirm_payload_read':False})
    print(json.dumps({'status':'complete','cases':len(cases),'attempt':attempt}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['plan','run'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--names',nargs='+')
    parser.add_argument('--attempt')
    args=parser.parse_args()
    if args.command=='plan':plan(args.root)
    else:run(args.root,args.names,args.attempt)


if __name__=='__main__':main()
