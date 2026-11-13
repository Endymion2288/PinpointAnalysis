"""Frozen event representations: train-only branch diagnostics or label curves."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import pyarrow as pa
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader

from analysis.flavour.inventory import write_json
from analysis.flavour.s2.dataset import SpatialBatchSampler
from analysis.flavour.s2.train import metrics,init_worker,save_parquet
from analysis.flavour.s3.dataset import collate_planes
from analysis.flavour.s3.models_repair import PlaneModuleClassifierV2
from analysis.flavour.s4.dataset import LabelDataset
from analysis.flavour.s4.pretrain import feature_statistics
from analysis.flavour.s4.train import build_model


@torch.no_grad()
def extract(model,dataset,config,device):
    sampler=SpatialBatchSampler(dataset,config.get('batch_size',16),42,shuffle=False)
    options={'num_workers':config.get('workers',1),'collate_fn':collate_planes,'pin_memory':device.type=='cuda'}
    if options['num_workers']:options.update(multiprocessing_context='spawn',worker_init_fn=init_worker,prefetch_factor=1)
    loader=DataLoader(dataset,batch_sampler=sampler,**options)
    result={key:[] for key in ['regional','global','local','summary','full']}
    indices=[]
    for batch in loader:
        indices.append(batch.pop('index').numpy());batch.pop('label')
        batch={k:v.to(device) for k,v in batch.items()}
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            features=model.encode(batch)
        branches=features['branches']
        values={'regional':branches[:,:3].flatten(1),'global':branches[:,3],
            'local':branches[:,4:].flatten(1),'summary':features['summary'],'full':features['head_input']}
        for key,value in values.items():result[key].append(value.float().cpu().numpy())
    order=np.argsort(np.concatenate(indices))
    return {key:np.concatenate(values)[order] for key,values in result.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--task',type=int,required=True)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args();config=json.loads(args.campaign.read_text())['experiments'][args.task]
    output=args.root/'experiments'/config['name'];output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();torch.set_num_threads(1);pa.set_cpu_count(1);pa.set_io_thread_count(1)
    torch.manual_seed(config.get('seed',42));np.random.seed(config.get('seed',42))
    model=build_model(config)
    if config.get('supervised_checkpoint'):
        state=torch.load(config['supervised_checkpoint'],map_location='cpu',weights_only=False)
        model.load_state_dict(state['model'],strict=True)
    model.to(args.device).eval().requires_grad_(False)
    train=LabelDataset(args.root,'train',config)
    features=extract(model,train,config,torch.device(args.device))
    train_only=config.get('probe_scope')=='train_only_branches'
    if train_only:
        # Disjoint readout-fit/readout-probe; the S3 supervised encoder has
        # already seen some of these labels. This diagnoses readability, not
        # independent generalization of the upstream S3 representation.
        rng=np.random.default_rng(20261115);perm=rng.permutation(len(train))
        n=min(2000,len(train)//5);fit,hold=perm[n:],perm[:n]
        y=train.metadata.label.to_numpy();yfit,yhold=y[fit],y[hold]
        holdmeta=train.metadata.iloc[hold].reset_index(drop=True)
        xfit={k:v[fit] for k,v in features.items()};xhold={k:v[hold] for k,v in features.items()}
    else:
        val=LabelDataset(args.root,'validation',config)
        xfit=features;xhold=extract(model,val,config,torch.device(args.device))
        yfit=train.metadata.label.to_numpy();yhold=val.metadata.label.to_numpy();holdmeta=val.metadata.copy()
    measured={};statistics={}
    for branch in (list(features) if train_only else ['full']):
        statistics[branch]={'fit':feature_statistics(torch.from_numpy(xfit[branch])),
            'probe':feature_statistics(torch.from_numpy(xhold[branch]))}
        modes=['linear','MLP'] if train_only else ['linear']
        for mode in modes:
            estimator=(LogisticRegression(C=1.,max_iter=500,tol=1e-5,solver='lbfgs') if mode=='linear' else
                MLPClassifier(hidden_layer_sizes=(128,),alpha=.01,batch_size=128,learning_rate_init=.001,
                    max_iter=100,early_stopping=True,validation_fraction=.15,n_iter_no_change=10,random_state=42))
            pipeline=make_pipeline(StandardScaler(),estimator)
            pipeline.fit(xfit[branch],yfit)
            probability=pipeline.predict_proba(xhold[branch])
            measured[branch+'_'+mode]=metrics(yhold,probability)
            prediction=holdmeta.copy()
            for c in range(4):prediction[f'p{c}']=probability[:,c]
            save_parquet(prediction,output/f'{branch}_{mode}_predictions.parquet')
    write_json(output/'metrics.json',measured)
    write_json(output/'feature_statistics.json',statistics)
    write_json(output/'manifest.json',{'utc':datetime.now(timezone.utc).isoformat(),'config':config,'status':'complete',
        'seconds':time.perf_counter()-started,'GPU':torch.cuda.get_device_name() if args.device=='cuda' else None,
        'fit_events':len(yfit),'probe_events':len(yhold),'frozen_encoder':True,
        'scope':'train-only readable-information diagnostic; supervised S3 encoder saw upstream labels' if train_only else 'independent frozen linear-probe label efficiency on complete 10k validation',
        'calibration_test_used':False,'code_commit':os.environ.get('PINPOINT_CODE_COMMIT') or subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()})
    train.native.close()
    if not train_only:val.native.close()
    print(json.dumps({key:{'NLL':value['NLL'],'macro_AP':np.mean([v['PR_AUC'] for v in value['classwise'].values()])} for key,value in measured.items()}),flush=True)


if __name__=='__main__':main()
