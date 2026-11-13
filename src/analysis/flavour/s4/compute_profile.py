"""Count training matrix operations on sampled real batches for budget controls.

The proxy includes Linear and attention matmuls, their backwards, and eager
activation-checkpoint recomputation. It excludes normalization, indexing,
elementwise operations and compiler fusion; report actual allocation time too.
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

from analysis.flavour.inventory import write_json
from analysis.flavour.s2.dataset import SpatialBatchSampler
from analysis.flavour.s3.dataset import collate_planes
from analysis.flavour.s3.models_repair import StableAttention
from analysis.flavour.s4.dataset import LabelDataset,PretrainDataset,collate_pretrain
from analysis.flavour.s4.models import PretrainingModel,objective_loss
from analysis.flavour.s4.train import build_model


class MatrixOperations:
    def __init__(self,model):
        self.counts={};self.backward=False;self.handles=[]
        for module in model.modules():
            if isinstance(module,nn.Linear):self.handles.append(module.register_forward_hook(self.linear))
            elif isinstance(module,StableAttention):self.handles.append(module.register_forward_hook(self.attention))

    def add(self,key,value):self.counts[key]=self.counts.get(key,0)+int(value)

    def record(self,kind,output,forward,backward):
        self.add(('recomputed_' if self.backward else '')+'forward_'+kind,forward)
        if output.requires_grad:
            def gradient(value):self.add('backward_'+kind,backward)
            output.register_hook(gradient)

    def linear(self,module,inputs,output):
        forward=2*inputs[0].numel()*module.out_features
        backward=forward*(int(inputs[0].requires_grad)+int(module.weight.requires_grad))
        self.record('dense',output,forward,backward)

    def attention(self,module,inputs,output):
        query,source=inputs[:2]
        forward=4*query.shape[0]*query.shape[1]*source.shape[1]*query.shape[2]
        self.record('attention',output,forward,2*forward)

    def close(self):
        for handle in self.handles:handle.remove()


def shape_operations(config,batch):
    """Eager training matrix-operation proxy from actual ragged batch shapes."""
    ssl='student' in batch
    observed=batch['student'] if ssl else batch
    b=observed['event_mask'].shape[0]
    dim=config.get('dim',96);queries=config.get('region_queries',2)
    latents=config.get('latents',64)
    values={}
    def linear(name,n,input_dim,output_dim,input_grad=True,parameter_grad=True,recompute=False):
        multiplier=1+int(input_grad)+int(parameter_grad)+int(recompute)
        values[name]=values.get(name,0)+2*n*input_dim*output_dim*multiplier
    def attention(name,groups,nq,nk,recompute=False,grad=True):
        linear(name+'_q',groups*nq,dim,dim,grad,grad,recompute)
        linear(name+'_kv',groups*nk,dim,2*dim,grad,grad,recompute)
        linear(name+'_out',groups*nq,dim,dim,grad,grad,recompute)
        linear(name+'_up',groups*nq,dim,4*dim,grad,grad,recompute)
        linear(name+'_down',groups*nq,2*dim,dim,grad,grad,recompute)
        values[name+'_attention']=4*groups*nq*nk*dim*(3 if grad else 1)+4*groups*nq*nk*dim*int(recompute)
    def encoder(prefix,view,grad):
        nb,nt=view['nodes'].shape[:2];nr,nk=view['region_mask'].shape
        event_width=int(torch.bincount(view['region_event'],minlength=b).max())*queries
        linear(prefix+'input',nb*nt,26,dim,False,grad,grad)
        for i in range(2):attention(prefix+f'local{i}',nb,nt,nt,grad,grad)
        linear(prefix+'local_pool',nb,2*dim+8,dim,grad,grad,grad)
        attention(prefix+'regional',nr,queries,nk,False,grad)
        linear(prefix+'position0',nr,1,dim,False,grad)
        linear(prefix+'position1',nr,dim,dim,grad,grad)
        attention(prefix+'latent_read',b,latents,event_width,False,grad)
        for i in range(2):attention(prefix+f'global{i}',b,latents,latents,False,grad)
        linear(prefix+'summary',b,11,dim,False,grad and not ssl)
    encoder('encoder_',observed,True)
    if not ssl:
        head=(15 if config.get('readout_skip',True) else 9)*dim
        linear('head0',b,head,2*dim);linear('head1',b,2*dim,4);linear('linear_route',b,head,4)
    else:
        def decoder(prefix,nq,nk):
            linear(prefix+'position',b*nq,34,dim,False)
            attention(prefix+'read',b,nq,nk)
            linear(prefix+'output0',b*nq,dim,dim);linear(prefix+'output1',b*nq,dim,8)
        decoder('reconstruction_',batch['query'].shape[1],latents)
        if 'rel_query' in batch:
            nk=latents
            if config.get('rel_source')=='global_and_bar_local':
                events=observed['event_index'][observed['block_detector']==1]
                nk+=int(torch.bincount(events,minlength=b).max())
            decoder('rel_',batch['rel_query'].shape[1],nk)
        if config.get('objective')=='distill':
            encoder('teacher_',batch['teacher'],False)
            linear('projector0',b,2*dim,2*dim);linear('projector1',b,2*dim,dim)
            linear('predictor0',b,dim,2*dim);linear('predictor1',b,2*dim,dim)
            linear('teacher_projector0',b,2*dim,2*dim,False,False)
            linear('teacher_projector1',b,2*dim,dim,False,False)
    return {'matrix_operations':sum(values.values()),'components':values}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--task',type=int,required=True)
    args=parser.parse_args();config=json.loads(args.campaign.read_text())['experiments'][args.task]
    output=args.root/'experiments'/config['name'];output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);pa.set_cpu_count(2);pa.set_io_thread_count(2)
    started=time.perf_counter();seed=config.get('seed',42);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    manifest={'status':'running','config':config,'GPU':None,'utc':datetime.now(timezone.utc).isoformat(),
        'code_commit':os.environ.get('PINPOINT_CODE_COMMIT') or subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    write_json(output/'manifest.json',manifest)
    results=[]
    for entry in config['profiles']:
        case={**entry,'compile_local':False};ssl=case.get('objective')!='classification'
        data=(PretrainDataset if ssl else LabelDataset)(args.root,'train',case)
        sampler=SpatialBatchSampler(data,case.get('batch_size',16),seed,steps=case['steps_per_epoch'])
        batches=list(sampler)
        selected=np.sort(rng.choice(len(batches),min(config.get('profile_batches',12),len(batches)),replace=False))
        model=(PretrainingModel(case) if ssl else build_model(case)).train()
        count=MatrixOperations(model);rows=[]
        for index in selected:
            items=[data[int(i)] for i in batches[index]]
            batch=(collate_pretrain if ssl else collate_planes)(items)
            model.zero_grad(set_to_none=True);count.counts={};count.backward=False
            if ssl:loss,_=objective_loss(model(batch),batch,case)
            else:
                labels=batch.pop('label');batch.pop('index')
                loss=torch.nn.functional.cross_entropy(model(batch),labels)
            count.backward=True;loss.backward()
            row={'sampler_batch':int(index),'events':len(items),'operations':sum(count.counts.values()),'components':dict(count.counts)}
            rows.append(row);print(json.dumps({'profile':case['name'],**row}),flush=True)
        count.close();data.native.close()
        values=np.array([r['operations'] for r in rows],dtype=float)
        results.append({'name':case['name'],'config':case,'sampled_batches':rows,
            'mean_operations_per_step':float(values.mean()),'sample_SD':float(values.std(ddof=1)),
            'sample_SE':float(values.std(ddof=1)/np.sqrt(len(values))),
            'reference_training_steps':case['epochs']*case['steps_per_epoch']})
        write_json(output/'matrix_operations.json',{'profiles':results,'scope':__doc__})
    manifest.update(status='complete',seconds=time.perf_counter()-started)
    write_json(output/'manifest.json',manifest)


if __name__=='__main__':main()
