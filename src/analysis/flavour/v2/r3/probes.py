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
from analysis.flavour.v2.r1.tasks import probabilities
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
            modules={'local':encoder.pool,'global':backbone}
            if len(encoder.global_blocks):modules['regional']=encoder.global_blocks[0]
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


def extract(model,data,config,reference=None):
    loader,_,_=make_loader(data,config,False)
    hooks=ReadableFeatures(model)
    arrays={};indices=[];scores=[]
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            gpu=device_batch(batch,'cuda:0')
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['bf16']):
                logits=model(gpu).float()
            if reference is not None:scores.append(probabilities(logits,config['task']).cpu().numpy())
            features=hooks.collect(gpu)
            for name,value in features.items():arrays.setdefault(name,[]).append(value)
            indices.extend(batch['index'].tolist())
    hooks.close();shutdown_loader(loader)
    order=np.argsort(indices)
    if not np.array_equal(np.asarray(indices)[order],np.arange(len(data))):
        raise ValueError('probe features lost or duplicated selected events')
    readback=None
    if reference is not None:
        keys=['run','chunk','row']
        columns=['p0','p1','p2','p3'] if config['task']=='T4' else ['p0','p1','p3']
        reference=reference.set_index(keys).loc[pd.MultiIndex.from_frame(data.metadata[keys])]
        values=np.concatenate(scores)[order]
        expected=reference[columns].to_numpy()
        delta=np.abs(values-expected)
        maximum=float(delta.max())
        agreement=float((values.argmax(1)==expected.argmax(1)).mean())
        readback={'status':'complete' if np.isfinite(values).all() and maximum<.01 and agreement>.99 else 'failed',
            'events':len(values),'GPU':torch.cuda.get_device_name(),
            'maximum_probability_abs_difference':maximum,'mean_probability_abs_difference':float(delta.mean()),
            'argmax_agreement':agreement,'minimum_argmax_agreement':.99,
            'tolerance':.01,'scope':'all fixed Tune rows reforwarded from saved exposure weights and freshly prepared accepted inputs; numerical device/batch variation tolerance, not a scientific comparison'}
        if readback['status']!='complete':raise ValueError('checkpoint/readout probability readback failed')
    return {name:np.concatenate(values)[order] for name,values in arrays.items()},readback


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--experiments',type=Path,nargs='+',required=True)
    parser.add_argument('--attempt',required=True)
    parser.add_argument('--cache-workers',type=int,default=4)
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    campaign=json.loads((args.root/'campaign.json').read_text())
    selected=pd.read_parquet(Path(campaign['r2'])/'gap/train_screen.parquet')
    output=args.root/'probes'/args.attempt;output.mkdir(parents=True,exist_ok=False)
    models={}
    cached=None;cached_signature=None
    scratch=tempfile.TemporaryDirectory(prefix='r3-probe-inputs-')
    for experiment in args.experiments:
        record=json.loads((experiment/'manifest.json').read_text())
        if record['status']!='complete':raise ValueError('readability requires completed fixed-endpoint weights')
        config={**record['config'],'cache_training_inputs':False}
        rows=selected[selected.label.ne(2)] if config['task']=='T3' else selected
        # A native control ignores edge tensors, so it can share the exact
        # raw-node inputs and candidate cache with its relation counterpart.
        input_config={**config,'cache_workers':args.cache_workers,'cache_storage':'mmap'}
        if config['family'] in ['B2','B3']:input_config['build_relations']=True
        signature={k:input_config.get(k) for k in ['family','task','image','sensor_row_image',
            'native_pixel_pitch_mm','relation_plane_policy','drop_modality','architecture_revision']}
        if cached_signature!=signature:
            cached=None
            train=R3Dataset(Path(campaign['r1'])/'data','train',input_config,cache_dir=scratch.name)
            subset(train,rows);train.cache_inputs()
            tune=R3Dataset(Path(campaign['r1'])/'data','validation',input_config,cache_dir=scratch.name)
            tune.cache_inputs()
            cached=(train,tune);cached_signature=signature
            write_json(output/'input_cache.json',{'Train':train.cache_audit,'Tune':tune.cache_audit,
                'scope':'allocation-local exact arrays reused for matched frozen-feature probes; not persisted'})
        train,tune=cached
        model=build_model(config).cuda()
        state=torch.load(experiment/'exposure.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(state['model']);del state
        x,_=extract(model,train,config)
        z,readback=extract(model,tune,config,pd.read_parquet(experiment/'predictions.parquet'))
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
            'checkpoint_probability_readback':readback,
            'absent_feature_paths':['regional'] if config.get('native_global_ablation') else [],
            'feature_dimensions':{name:value.shape[1] for name,value in x.items()}}
        del model;torch.cuda.empty_cache()
        write_json(output/'results.json',{'status':'running','models':models})
    write_json(output/'results.json',{'status':'complete','models':models,
        'probe_recipe':'StandardScaler + multinomial logistic C1, max_iter1000; fit only common frozen balanced Train4k; no Tune hyperparameter choice',
        'scope':'readability of frozen features; actual branch use requires training gradients and utility requires retrained ablation',
        'native_regional_definition':'first inter-block geometry-message stage; no independent region branch in B2/B3; absent in the retrained global ablation',
        'Cal_A_Cut_B_Confirm_payload_read':False})
    scratch.cleanup()


if __name__=='__main__':main()
