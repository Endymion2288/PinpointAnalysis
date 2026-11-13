"""Fit a geometry-query-only decoder on original train-fit events.

The decoder gets the same positions proposed from visible geometry, but no
event encoder features. This measures the query-position shortcut directly.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pyarrow as pa
import torch
from torch import nn
from torch.utils.data import DataLoader

from analysis.flavour.inventory import write_json
from analysis.flavour.s2.train import save_parquet,save_checkpoint
from analysis.flavour.s4.dataset import PretrainDataset
from analysis.flavour.s4.models import QueryDecoder,objective_loss
from analysis.flavour.s4.pretrain import evaluate


class QueryOnly(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.decoder=QueryDecoder(config.get('dim',96))
        self.constant=nn.Parameter(torch.randn(1,config.get('latents',64),config.get('dim',96))*.02)

    def forward(self,batch):
        latent=self.constant.expand(len(batch['query']),-1,-1)
        return {'reconstruction':self.decoder(batch['query'],latent),'features':{'global_tokens':latent}}

    @staticmethod
    def pool_global(features):
        x=features['global_tokens']
        return torch.cat([x.mean(1),x.amax(1)],-1)


def cache_queries(data,indices):
    keys=['query','query_level','query_near','reconstruction','mask_counts','query_positive_unique']
    return [{key:torch.from_numpy(np.asarray(item[key])) for key in keys}
        for item in (data[int(i)] for i in indices)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--task',type=int,required=True)
    args=parser.parse_args();config=json.loads(args.campaign.read_text())['experiments'][args.task]
    output=args.root/'experiments'/config['name'];output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);pa.set_cpu_count(2);pa.set_io_thread_count(2)
    seed=config.get('seed',42);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    started=time.perf_counter();manifest={'status':'running','config':config,'GPU':None,
        'code_commit':os.environ.get('PINPOINT_CODE_COMMIT') or subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'scope':'fitted geometry/visible-neighborhood query positions only; no event encoder, labels or downstream validation',
        'utc':datetime.now(timezone.utc).isoformat()}
    write_json(output/'manifest.json',manifest)
    train=PretrainDataset(args.root,'train',config);val=PretrainDataset(args.root,'validation',config)
    take=np.sort(rng.choice(len(train),min(config.get('query_baseline_fit_events',2048),len(train)),replace=False))
    fitting=cache_queries(train,take)
    validation=cache_queries(val,range(min(config.get('reconstruction_probe_events',2000),len(val))))
    train.native.close();val.native.close()
    write_json(output/'cache_summary.json',{'fit_events':len(fitting),'inner_probe_events':len(validation),
        'cache_seconds':time.perf_counter()-started,'same_inner_probe_as_full_MAE':len(validation)==2000})
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(fitting,batch_size=config.get('batch_size',32),shuffle=True,generator=generator)
    validation_loader=DataLoader(validation,batch_size=config.get('batch_size',32))
    model=QueryOnly(config);optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01)
    initial,_=evaluate(model,validation_loader,torch.device('cpu'),False,config)
    write_json(output/'initial_reconstruction.json',initial)
    steps=0;history=[]
    while steps<config.get('steps_per_epoch',512):
        model.train()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True);loss,_=objective_loss(model(batch),batch,config)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();steps+=1
            if steps%64==0:
                row={'step':steps,'loss':float(loss.detach()),'seconds':time.perf_counter()-started}
                history.append(row);write_json(output/'progress.json',row);print(json.dumps(row),flush=True)
            if steps>=config.get('steps_per_epoch',512):break
    measured,predictions=evaluate(model,validation_loader,torch.device('cpu'),False,config)
    write_json(output/'reconstruction.json',measured);save_parquet(predictions,output/'query_predictions.parquet')
    save_checkpoint(output/'best.pt',{'model':model.state_dict(),'config':config,'steps':steps})
    write_json(output/'history.json',history)
    manifest.update(status='complete',seconds=time.perf_counter()-started,steps=steps,
        fit_events=len(fitting),inner_probe_events=len(validation),encoder_information='none; shared constant tokens')
    write_json(output/'manifest.json',manifest)


if __name__=='__main__':main()
