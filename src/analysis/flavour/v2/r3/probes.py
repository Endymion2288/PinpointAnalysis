"""Independent Train-fitted readouts of pre-pool, regional and global features.

The native baseline has no separately supervised regional branch. Here region
means the first inter-block message stage, and global means the final event
embedding. Probe quality measures readability, not benefit from retraining.
"""

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis.flavour.inventory import COMPRESSION,write_json
from analysis.flavour.v2.profile import device_batch,shutdown_loader
from analysis.flavour.v2.r1.evaluate import domain_report
from analysis.flavour.v2.r2.evaluate import domains
from analysis.flavour.v2.r2.train import subset
from analysis.flavour.v2.r3.data import R3Dataset
from analysis.flavour.v2.r3.models import build_model
from analysis.flavour.v2.r3.train import make_loader


def pool_blocks(value,event_index,events):
    output=[]
    for event in range(events):
        x=value[event_index==event]
        output.append(torch.cat([x.mean(0),x.amax(0)]))
    return torch.stack(output)


class ReadableFeatures:
    def __init__(self,model):
        self.model=model
        self.values={};self.handles=[]
        backbone=model.backbone
        if getattr(model,'family',None)=='B1':
            self.family='B1'
            modules={'local':backbone.layers[1],'regional':backbone.layers[3],'global':backbone}
        else:
            self.family='native'
            encoder=getattr(backbone,'encoder',backbone)
            modules={'local':encoder.pool,'regional':encoder.global_blocks[0],'global':backbone}
        for name,module in modules.items():
            def capture(_,inputs,output,name=name):
                self.values.setdefault(name,[]).append(output.detach().float())
            self.handles.append(module.register_forward_hook(capture))
        def capture_tracker(_,inputs,output):
            self.values.setdefault('tracker',[]).append(output.detach().float())
        self.handles.append(model.tracker.register_forward_hook(capture_tracker))

    def collect(self,batch):
        events=len(batch['label']);result={}
        for name,values in self.values.items():
            value=torch.cat(values)
            if name in ['global','tracker']:
                result[name]=value
            elif self.family=='B1':
                pooled=torch.cat([value.mean((-1,-2)),value.amax((-1,-2))],-1)
                result[name]=pooled.reshape(events,-1)
            elif name=='local':
                result[name]=pool_blocks(value,batch['event_index'],events)
            else:
                rows=[]
                for event in range(events):
                    x=value[event,batch['event_mask'][event]]
                    rows.append(torch.cat([x.mean(0),x.amax(0)]))
                result[name]=torch.stack(rows)
        self.values={}
        return {name:value.cpu().numpy() for name,value in result.items()}

    def close(self):
        for handle in self.handles:handle.remove()


def extract(model,data,config):
    loader,_,_=make_loader(data,config,False)
    hooks=ReadableFeatures(model)
    arrays={};indices=[]
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            gpu=device_batch(batch,'cuda:0')
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['bf16']):
                model(gpu)
            features=hooks.collect(gpu)
            for name,value in features.items():arrays.setdefault(name,[]).append(value)
            indices.extend(batch['index'].tolist())
    hooks.close();shutdown_loader(loader)
    order=np.argsort(indices)
    if not np.array_equal(np.asarray(indices)[order],np.arange(len(data))):
        raise ValueError('probe features lost or duplicated selected events')
    return {name:np.concatenate(values)[order] for name,values in arrays.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--experiments',type=Path,nargs='+',required=True)
    parser.add_argument('--attempt',required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    campaign=json.loads((args.root/'campaign.json').read_text())
    selected=pd.read_parquet(Path(campaign['r2'])/'gap/train_screen.parquet')
    output=args.root/'probes'/args.attempt;output.mkdir(parents=True,exist_ok=False)
    models={}
    for experiment in args.experiments:
        record=json.loads((experiment/'manifest.json').read_text())
        if record['status']!='complete':raise ValueError('readability requires completed fixed-endpoint weights')
        config={**record['config'],'cache_training_inputs':False}
        if config.get('native_global_ablation'):raise ValueError('probe contract expects intact control/candidate')
        rows=selected[selected.label.ne(2)] if config['task']=='T3' else selected
        with tempfile.TemporaryDirectory(prefix='r3-probes-') as scratch:
            train=R3Dataset(Path(campaign['r1'])/'data','train',config,cache_dir=scratch)
            subset(train,rows)
            tune=R3Dataset(Path(campaign['r1'])/'data','validation',config,cache_dir=scratch)
            model=build_model(config).cuda()
            state=torch.load(experiment/'exposure.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(state['model']);del state
            x=extract(model,train,config);z=extract(model,tune,config)
            y=train.metadata.label.to_numpy()
            reports={}
            for branch in x:
                classifier=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1000,random_state=20261121))
                classifier.fit(x[branch],y)
                p=classifier.predict_proba(z[branch]);frame=tune.metadata.copy()
                for c,value in zip(classifier.classes_,p.T):frame[f'p{c}']=value
                reports[branch]={domain:domain_report(frame,mask) for domain,mask in domains(frame).items()}
                frame.to_parquet(output/(experiment.name+'_'+branch+'_predictions.parquet'),index=False,compression=COMPRESSION)
            models[experiment.name]={'config':config,'reports':reports,'Train_events':len(train),'Tune_events':len(tune),
                'feature_dimensions':{name:value.shape[1] for name,value in x.items()}}
            train.native.close();tune.native.close();del model;torch.cuda.empty_cache()
        write_json(output/'results.json',{'status':'running','models':models})
    write_json(output/'results.json',{'status':'complete','models':models,
        'probe_recipe':'StandardScaler + multinomial logistic C1, max_iter1000; fit only common frozen balanced Train4k; no Tune hyperparameter choice',
        'scope':'readability of frozen features; actual branch use requires training gradients and utility requires retrained ablation',
        'native_regional_definition':'first inter-block geometry-message stage; no independent region branch in B2/B3',
        'Cal_A_Cut_B_Confirm_payload_read':False})


if __name__=='__main__':main()
