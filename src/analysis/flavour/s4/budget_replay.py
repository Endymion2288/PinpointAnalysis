"""Replay actual ragged training shapes on CPU to freeze matched budget controls."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time

import pyarrow as pa
import torch
from torch.utils.data import DataLoader

from analysis.flavour.inventory import write_json
from analysis.flavour.s2.dataset import SpatialBatchSampler
from analysis.flavour.s2.train import init_worker
from analysis.flavour.s3.dataset import collate_planes,plane_tokens
from analysis.flavour.s4.compute_profile import shape_operations
from analysis.flavour.s4.dataset import PretrainDataset,LabelDataset


class ShapePretrainDataset(PretrainDataset):
    """Identical visible observations and grouping; skip target/query lookups."""
    def __getitem__(self,index):
        event=self.native[int(self.indexes[index])]
        seed=int(self.config.get('seed',42))+index*1009+(int(self.epoch)+self.config.get('mask_epoch_offset',0))*10000019
        visible,_=self.query_geometry.visible(event['observations'],seed,self.config.get('mask_ratio',.75))
        nodes,groups,modules,counts=plane_tokens(visible,self.geometry,self.config,seed)
        return {'nodes':nodes,'groups':groups,'modules':modules,'label':-1,'index':index,
            'raw_counts':counts,'region_blocks':self.config.get('region_blocks',32)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--task',type=int,required=True)
    args=parser.parse_args();config=json.loads(args.campaign.read_text())['experiments'][args.task]
    output=args.root/'experiments'/config['name'];output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);pa.set_cpu_count(1);pa.set_io_thread_count(1)
    started=time.perf_counter();rows=[]
    manifest={'status':'running','config':config,'GPU':None,'utc':datetime.now(timezone.utc).isoformat(),
        'code_commit':os.environ.get('PINPOINT_CODE_COMMIT') or subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    write_json(output/'manifest.json',manifest)
    for case in config['profiles']:
        ssl=case['objective']!='classification'
        if case['objective']=='distill':raise ValueError('EMA needs teacher view accounting; use separate profile')
        data=(ShapePretrainDataset if ssl else LabelDataset)(args.root,'train',case)
        sampler=SpatialBatchSampler(data,case.get('batch_size',16),case.get('seed',42),steps=case['steps_per_epoch'])
        loader=DataLoader(data,batch_sampler=sampler,num_workers=3,persistent_workers=True,
            multiprocessing_context='spawn',worker_init_fn=init_worker,prefetch_factor=1,collate_fn=collate_planes)
        total=0;exposures=0;steps=0;epochs=[]
        for epoch in range(case['epochs']):
            sampler.epoch=epoch
            if ssl:data.epoch.fill_(epoch)
            before=total
            for batch in loader:
                b=len(batch['index']);view=batch
                if ssl:
                    view={'student':batch,'query':torch.empty((b,3*case.get('queries_per_kind',64),22))}
                    if case['objective']=='mae_rel':view['rel_query']=torch.empty((b,128,22))
                total+=shape_operations(case,view)['matrix_operations'];exposures+=b;steps+=1
                if steps%500==0:
                    write_json(output/'progress.json',{'profile':case['name'],'epoch':epoch+1,'steps':steps,
                        'matrix_operations':total,'train_exposures':exposures,'seconds':time.perf_counter()-started})
            epochs.append({'epoch':epoch+1,'steps':steps,'matrix_operations':total-before,
                'cumulative_matrix_operations':total,'cumulative_train_exposures':exposures})
            print(json.dumps({'profile':case['name'],**epochs[-1]}),flush=True)
        data.native.close();del loader
        rows.append({'name':case['name'],'config':case,'matrix_operations':total,'optimizer_steps':steps,
            'training_exposures':exposures,'epochs':epochs,'mean_operations_per_step':total/steps})
        write_json(output/'budgets.json',{'profiles':rows,
            'scope':'exact deterministic sampler/mask/group-shape replay for the declared eager Linear/attention forward/backward proxy; excludes elementwise, normalization, compiler fusion and nontraining time'})
    manifest.update(status='complete',seconds=time.perf_counter()-started)
    write_json(output/'manifest.json',manifest)


if __name__=='__main__':main()
