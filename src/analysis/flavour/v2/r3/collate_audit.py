"""Measure exact R3 collation against the accepted block-scanning implementation."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from analysis.flavour.inventory import write_json
from analysis.flavour.s2.dataset import SpatialBatchSampler
from analysis.flavour.v2.profile import MemoryMonitor
from analysis.flavour.v2.r1.data import collate as original_collate
from analysis.flavour.v2.r3.data import R3Dataset, collate


def reference_collate(events):
    """Pre-optimization R3 padding and edge mapping, for the equivalence audit."""
    batch=original_collate(events)
    if 'relation_edges' not in events[0]:
        return batch
    sources=[];targets=[];block_offset=0
    for event in events:
        groups=event['groups']
        mapping=np.empty(len(groups),dtype='int64')
        blocks=int(groups.max())+1
        for block in range(blocks):
            rows=np.flatnonzero(groups==block)
            mapping[rows]=(block_offset+block)*32+np.arange(len(rows))
        edges=event['relation_edges']
        sources.append(mapping[edges[0]])
        targets.append(mapping[edges[1]])
        block_offset+=blocks
    batch['relation_source']=torch.from_numpy(np.concatenate(sources))
    batch['relation_target']=torch.from_numpy(np.concatenate(targets))
    return batch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--attempt',required=True)
    parser.add_argument('--cache-workers',type=int,default=4)
    parser.add_argument('--cache-storage',choices=['memory','mmap'],default='memory')
    args=parser.parse_args()
    torch.set_num_threads(1)
    campaign=json.loads((args.root/'campaign.json').read_text())
    config={**json.loads(args.config.read_text()),'cache_training_inputs':True,
        'cache_workers':args.cache_workers,'build_relations':True,'cache_storage':args.cache_storage}
    output=args.root/'audit'/f'collate_{args.attempt}.json'
    if output.exists():raise FileExistsError(output)
    with MemoryMonitor() as memory:
        data=R3Dataset(Path(campaign['r0'])/'profile_data','train',config)
        sampler=SpatialBatchSampler(data,config['batch_size'],config['seed'])
        timers={'reference_seconds':0.,'vectorized_seconds':0.}
        events=0;batches=0;max_nodes=0;max_edges=0
        for indexes in sampler:
            items=[data[int(index)] for index in indexes]
            outputs={}
            # Alternate timing order to avoid always giving one version warm arrays.
            order=[('reference_seconds',reference_collate),('vectorized_seconds',collate)]
            for name,method in order[::1 if batches%2==0 else -1]:
                start=time.perf_counter();outputs[name]=method(items)
                timers[name]+=time.perf_counter()-start
            a,b=outputs.values()
            if set(a)!=set(b) or any(not torch.equal(a[k],b[k]) for k in a):
                raise ValueError('vectorized collation changed accepted model tensors')
            max_nodes=max(max_nodes,sum(len(item['nodes']) for item in items))
            max_edges=max(max_edges,sum(item['relation_edges'].shape[1] for item in items))
            events+=len(items);batches+=1
        result={'status':'complete','config':config,'Train_events':events,'batches':batches,
            'all_collated_tensors_bitwise_equal':True,'tensor_names':sorted(a),
            **timers,'CPU_collation_speedup':timers['reference_seconds']/timers['vectorized_seconds'],
            'maximum_batch_nodes':max_nodes,'maximum_batch_edges':max_edges,
            'input_preparation':data.cache_audit,'memory':memory.peak,
            'scope':'same fixed Train10k, same spatial batches; CPU collation only, not GPU throughput',
            'actual_Tune_used':False,'Cal_A_Cut_B_Confirm_payload_read':False}
    write_json(output,result)
    print(json.dumps({k:result[k] for k in ['status','Train_events','batches',*timers,'CPU_collation_speedup']}),flush=True)


if __name__=='__main__':main()
