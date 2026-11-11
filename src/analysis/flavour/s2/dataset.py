"""Observation-only S2 transforms and complete-coverage ragged block batches."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from analysis.flavour.datasets import CachedNativeDataset
from analysis.flavour.s2.representation import token_inputs


class SpatialDataset(Dataset):
    def __init__(self,root,split,config,cache_dir=None):
        self.root=Path(root)
        self.config=config
        plan=json.loads((self.root/'plan.json').read_text())
        self.geometry=json.loads(Path(plan['geometry']).read_text())
        self.paths=sorted(p for p in (self.root/'shards').iterdir() if (p/'observations.parquet').exists())
        self.native=CachedNativeDataset(self.paths,cache_dir=cache_dir,max_shards=config.get('cache_shards',2))
        indexes,metadata=[],[]
        for i,path in enumerate(self.paths):
            frame=pd.read_parquet(path/'provenance.parquet')
            keep=frame.split.eq(split)
            if config.get('train_domain','all')!='all' and split=='train':
                keep &= frame[config['train_domain']]
            index=np.flatnonzero(keep.to_numpy())
            indexes.extend((self.native.boundaries[i]+index).tolist())
            metadata.append(frame.iloc[index])
        self.indexes=np.asarray(indexes,dtype='int64')
        self.metadata=pd.concat(metadata,ignore_index=True)
        self.workload=None
        if config.get('bucket_size',0):
            key=['run','chunk','row']
            counts=pd.read_parquet(self.root/'catalog.parquet',columns=key+['pixel_channels','bar_channels','tracker_channels'])
            joined=self.metadata[key].merge(counts,on=key,how='left',validate='one_to_one',sort=False)
            if joined.pixel_channels.isna().any():
                raise ValueError('batch occupancy catalog lacks selected observations')
            self.workload=joined[['pixel_channels','bar_channels','tracker_channels']].sum(axis=1).to_numpy()

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self,index):
        event=self.native[int(self.indexes[index])]
        nodes,groups,counts=token_inputs(event['observations'],self.geometry,
            pitch_mm=self.config.get('pitch_mm',0.),projection=self.config.get('projection',False),
            reducer=self.config.get('reducer','count'),modalities=self.config.get('modalities',['pixel','bar']),
            origin_shift=self.config.get('origin_shift',0.),shuffle_local=self.config.get('shuffle_local',False),
            seed=int(index)+42)
        return {'nodes':nodes,'groups':groups,'label':event['targets']['label'],'index':index,'raw_counts':counts}


def collate_spatial(events):
    nodes,masks,event_ids,block_ids=[],[],[],[]
    maxblocks=max(int(e['groups'].max())+1 for e in events)
    event_mask=np.zeros((len(events),maxblocks),dtype=bool)
    for i,event in enumerate(events):
        for b in range(int(event['groups'].max())+1):
            block=event['nodes'][event['groups']==b]
            if not 0<len(block)<=32:
                raise ValueError('invalid local block, no truncation permitted')
            padded=np.zeros((32,18),dtype='float32')
            padded[:len(block)]=block
            mask=np.arange(32)<len(block)
            nodes.append(padded)
            masks.append(mask)
            event_ids.append(i)
            block_ids.append(b)
            event_mask[i,b]=True
    return {'nodes':torch.from_numpy(np.stack(nodes)), 'local_mask':torch.from_numpy(np.stack(masks)),
            'event_index':torch.tensor(event_ids),'block_index':torch.tensor(block_ids),
            'event_mask':torch.from_numpy(event_mask),'label':torch.tensor([e['label'] for e in events]),
            'index':torch.tensor([e['index'] for e in events])}


class SpatialBatchSampler:
    """Shuffle indices in mixed-shard windows to avoid random cache thrashing."""

    def __init__(self,dataset,batch_size,seed,shuffle=True,steps=None):
        self.dataset,self.batch_size,self.seed,self.shuffle=dataset,batch_size,seed,shuffle
        self.epoch=0
        self.steps=steps
        self.bucket_size=dataset.config.get('bucket_size',0)
        if self.bucket_size and self.bucket_size%batch_size:
            raise ValueError('bucket size must be divisible by batch size to preserve step counts')

    def __len__(self):
        return self.steps if self.steps is not None else sum((len(ix)+self.batch_size-1)//self.batch_size for ix in self._windows())

    def _windows(self,cycle=0):
        rng=np.random.default_rng(self.seed+self.epoch+10000*cycle)
        shard=np.searchsorted(self.dataset.native.boundaries,self.dataset.indexes,side='right')-1
        order=rng.permutation(len(self.dataset.paths)) if self.shuffle else np.arange(len(self.dataset.paths))
        capacity=self.dataset.config.get('cache_shards',2)
        for start in range(0,len(order),capacity):
            index=np.flatnonzero(np.isin(shard,order[start:start+capacity]))
            if self.shuffle:
                rng.shuffle(index)
            yield index

    def __iter__(self):
        emitted,cycle=0,0
        target=len(self)
        while emitted<target:
            rng=np.random.default_rng(self.seed+self.epoch+10000*cycle+12345)
            for index in self._windows(cycle):
                batches=[]
                width=self.bucket_size or max(1,len(index))
                for start in range(0,len(index),width):
                    group=index[start:start+width]
                    if self.bucket_size:
                        group=group[np.argsort(self.dataset.workload[group],kind='stable')]
                    batches.extend(group[i:i+self.batch_size].tolist() for i in range(0,len(group),self.batch_size))
                if self.shuffle and self.bucket_size:
                    rng.shuffle(batches)
                for batch in batches:
                    if emitted==target:
                        return
                    yield batch
                    emitted+=1
            cycle+=1
