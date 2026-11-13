"""Freeze supervised continuation steps from the measured MAE matrix budget."""
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
from torch.utils.data import DataLoader

from analysis.flavour.inventory import write_json
from analysis.flavour.s2.dataset import SpatialBatchSampler
from analysis.flavour.s2.train import init_worker
from analysis.flavour.s4.compute_profile import shape_operations
from analysis.flavour.s4.dataset import LabelDataset


def event_shape(event):
    """Exact block/region counts without padded feature or geometry tensors."""
    counts=np.bincount(event['groups'])
    starts=np.r_[0,np.cumsum(counts)[:-1]]
    first=event['nodes'][starts]
    detector=first[:,12:15].argmax(1);detector[first[:,15]==0]=3
    keys=np.column_stack([event['modules'][starts],detector])
    _,inverse=np.unique(keys,axis=0,return_inverse=True)
    module_counts=np.bincount(inverse)
    width=event.get('region_blocks',32)
    return np.array([len(counts),int(((module_counts+width-1)//width).sum()),
        min(int(module_counts.max()),width)],dtype=np.int64)


def compact_batch(shapes):
    """Only the shapes consumed by the declared eager matrix-operation proxy."""
    shapes=np.asarray(shapes,dtype=np.int64)
    return {'nodes':torch.empty((int(shapes[:,0].sum()),32,18),device='meta'),
        'event_mask':torch.empty((len(shapes),int(shapes[:,0].max())),device='meta'),
        'region_mask':torch.empty((int(shapes[:,1].sum()),int(shapes[:,2].max())),device='meta'),
        'region_event':torch.repeat_interleave(torch.arange(len(shapes)),torch.from_numpy(shapes[:,1].copy()))}


class ShapeLabels(LabelDataset):
    def __getitem__(self,index):
        return event_shape(super().__getitem__(index))


def nearest_prefix(costs,target):
    """Choose the common per-epoch prefix nearest the additional SSL budget."""
    totals=np.asarray(costs,dtype=np.int64).sum(0).cumsum()
    choices=[max(1,int(np.searchsorted(totals,target))),min(len(totals),int(np.searchsorted(totals,target))+1)]
    steps=min(choices,key=lambda n:abs(int(totals[n-1])-target))
    return steps,int(totals[steps-1])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--task',type=int,required=True)
    args=parser.parse_args();config=json.loads(args.campaign.read_text())['experiments'][args.task]
    output=args.root/'experiments'/config['name'];output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);pa.set_cpu_count(1);pa.set_io_thread_count(1)
    started=time.perf_counter()
    manifest={'status':'running','config':config,'GPU':None,'utc':datetime.now(timezone.utc).isoformat(),
        'code_commit':os.environ.get('PINPOINT_CODE_COMMIT') or subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    write_json(output/'manifest.json',manifest)
    budgets=json.loads(Path(config['budget_replay']).read_text())['profiles']
    source=next(r for r in budgets if r['name']==config['matched_pretraining'])
    target=int(source['matrix_operations']);experiments=[];audits=[]
    for base in config['supervised_cases']:
        case=dict(base)
        parent=args.root/'experiments'/case['name']
        parent_manifest=json.loads((parent/'manifest.json').read_text())
        if parent_manifest['status']!='complete':raise RuntimeError('Scratch parent must finish first')
        data=ShapeLabels(args.root,'train',case)
        # One shape pass per labeled event; every repeated sampler prefix then
        # costs only integer arithmetic. Keep this small table, not embeddings.
        loader=DataLoader(data,batch_size=128,num_workers=3,multiprocessing_context='spawn',
            worker_init_fn=init_worker,prefetch_factor=1)
        table=torch.cat(list(loader)).numpy()
        np.save(output/f'shapes_n{case["label_budget"]}.npy',table)
        del loader;data.native.close()
        epochs=case['epochs'];offset=epochs
        baseline_operations=0;baseline_exposures=0
        baseline=SpatialBatchSampler(data,case.get('batch_size',16),case['seed'],steps=case['steps_per_epoch'])
        for epoch in range(epochs):
            baseline.epoch=epoch
            for indexes in baseline:
                baseline_operations+=shape_operations(case,compact_batch(table[indexes]))['matrix_operations']
                baseline_exposures+=len(indexes)
        samplers=[SpatialBatchSampler(data,case.get('batch_size',16),case['seed'],steps=1000000) for _ in range(epochs)]
        for epoch,sampler in enumerate(samplers):sampler.epoch=epoch+offset
        iterators=[iter(s) for s in samplers];costs=[[] for _ in samplers];exposures=[[] for _ in samplers]
        cumulative=0;steps=0
        while cumulative<target:
            steps+=1
            for epoch,iterator in enumerate(iterators):
                indexes=next(iterator)
                operations=shape_operations(case,compact_batch(table[indexes]))['matrix_operations']
                costs[epoch].append(operations);exposures[epoch].append(len(indexes));cumulative+=operations
            if steps%500==0:write_json(output/'progress.json',{'label_budget':case['label_budget'],
                'per_epoch_steps':steps,'additional_matrix_operations':cumulative,'target':target,'seconds':time.perf_counter()-started})
        selected,operations=nearest_prefix(costs,target)
        row={'label_budget':case['label_budget'],'pretraining_matrix_operations':target,
            'baseline_supervised_matrix_operations':baseline_operations,
            'additional_supervised_matrix_operations':operations,'relative_matching_error':(operations-target)/target,
            'MAE3_plus_finetuning_matrix_operations':target+baseline_operations,
            'supervised_total_matrix_operations':baseline_operations+operations,
            'baseline_train_exposures':baseline_exposures,
            'epochs':epochs,'steps_per_epoch':selected,'sampler_epoch_offset':offset,
            'additional_train_exposures':sum(sum(x[:selected]) for x in exposures),
            'per_epoch_matrix_operations':[sum(x[:selected]) for x in costs]}
        if abs(row['relative_matching_error'])>.005:raise RuntimeError('cannot match proxy budget within 0.5%')
        name=f's4_supervised_matched_n{case["label_budget"]}_s{case["seed"]}'
        case.update(name=name,initialization='supervised_matched_mae3',supervised_initial_checkpoint=str(parent/'last.pt'),
            epochs=epochs,steps_per_epoch=selected,sampler_epoch_offset=offset,
            dependencies=[base['name']],matched_budget=row,
            continuation_optimizer='fresh optimizer and warmup/cosine; complete last Scratch model including head',
            selection='best continuation validation NLL; original Scratch best reported separately')
        case.pop('pretrained_checkpoint',None)
        experiments.append(case);audits.append(row)
        write_json(output/'matched_budgets.json',{'matches':audits,'scope':source.get('scope',
            'declared eager Linear/attention forward/backward matrix proxy, including eager checkpoint recomputation; excludes elementwise, normalization and compiler fusion')})
        print(json.dumps(row),flush=True)
    write_json(output/'training_campaign.json',{'stage':'S4','calibration_test_used':False,
        'organization_device_hours_cap':60,'experiments':experiments})
    manifest.update(status='complete',seconds=time.perf_counter()-started)
    write_json(output/'manifest.json',manifest)


if __name__=='__main__':main()
