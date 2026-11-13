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
