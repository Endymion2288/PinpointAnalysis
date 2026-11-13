"""Measure exact canonical-input equivalence while skipping unused S2 groups."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import DataLoader, Subset

from analysis.flavour.inventory import write_json
from analysis.flavour.s2.train import init_worker
from analysis.flavour.s3.dataset import collate_planes, plane_tokens
from analysis.flavour.s4.dataset import LabelDataset, QueryGeometry


def compare(observation, geometry, config, repeats):
    seconds={'original':0., 'skip_unused_groups':0.}
    for repeat in range(repeats):
        results={}
        for original in ([True,False] if repeat%2==0 else [False,True]):
            started=time.perf_counter()
            values=plane_tokens(observation,geometry,config,build_input_blocks=original)
            seconds['original' if original else 'skip_unused_groups']+=time.perf_counter()-started
            results[original]=values
        for key,before,after in zip(['nodes','groups','modules','raw_counts'],results[True],results[False]):
            if not np.array_equal(before,after):raise AssertionError('canonical '+key+' differs')
    batches=[]
    for values in [results[True],results[False]]:
        item=dict(zip(['nodes','groups','modules','raw_counts'],values))
        item.update(label=0,index=0,region_blocks=config.get('region_blocks',32))
        batches.append(collate_planes([item]))
    for key in batches[0]:
        if not torch.equal(batches[0][key],batches[1][key]):raise AssertionError('collated '+key+' differs')
    return {'seconds':seconds,'repeats':repeats,'channels':len(values[0]),
        'blocks':len(np.bincount(values[1])),'all_arrays_and_collated_tensors_bitwise_equal':True}


class FrontendComparison(LabelDataset):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.query_geometry=QueryGeometry(self.geometry,pixel_query_patch=16)

    def __getitem__(self,index):
        observation=self.native[int(self.indexes[index])]['observations']
        visible,_=self.query_geometry.visible(observation,42+index*1009,.75)
        return {'index':int(index),'workload':int(self.workload[index]),
            'views':{name:compare(view,self.geometry,self.config,self.config.get('repeats',3))
                for name,view in [('complete',observation),('visible',visible)]}}


def first(items):
    return items[0]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--task',type=int,required=True)
    args=parser.parse_args();config=json.loads(args.campaign.read_text())['experiments'][args.task]
    output=args.root/'experiments'/config['name'];output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();torch.set_num_threads(1);pa.set_cpu_count(1);pa.set_io_thread_count(1)
    manifest={'status':'running','config':config,'GPU':None,'utc':datetime.now(timezone.utc).isoformat(),
        'code_commit':os.environ.get('PINPOINT_CODE_COMMIT') or subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    write_json(output/'manifest.json',manifest)
    data=FrontendComparison(args.root,'train',config)
    indexes=[]
    for label in range(4):
        selected=np.flatnonzero(data.metadata.label.to_numpy()==label)
        ordered=selected[np.argsort(data.workload[selected],kind='stable')]
        indexes.extend(ordered[np.linspace(0,len(ordered)-1,config.get('samples_per_class',8)).astype(int)].tolist())
    loader=DataLoader(Subset(data,indexes),batch_size=1,num_workers=3,collate_fn=first,
        multiprocessing_context='spawn',worker_init_fn=init_worker,prefetch_factor=1)
    rows=list(loader);del loader;data.native.close()
    totals={}
    for view in ['complete','visible']:
        values={key:sum(row['views'][view]['seconds'][key] for row in rows)
            for key in ['original','skip_unused_groups']}
        totals[view]={**values,'transform_speedup':values['original']/values['skip_unused_groups']}
    report={'events':len(rows),'workload_range':[min(row['workload'] for row in rows),max(row['workload'] for row in rows)],
        'all_arrays_and_collated_tensors_bitwise_equal':True,'transform_seconds':totals,'records':rows,
        'scope':'fixed train events at per-class occupancy quantiles, complete and .75 masked views; CPU transform timings exclude native I/O, collation and model; not end-to-end GPU throughput'}
    write_json(output/'equivalence.json',report)
    manifest.update(status='complete',seconds=time.perf_counter()-started);write_json(output/'manifest.json',manifest)
    print(json.dumps({key:value for key,value in report.items() if key!='records'}),flush=True)


if __name__=='__main__':main()
