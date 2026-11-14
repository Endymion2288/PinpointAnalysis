"""Timed native inputs and exact packed observation-transform caches on scratch."""

from collections import OrderedDict
import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from analysis.flavour.inventory import write_json
from analysis.flavour.s3.dataset import PlaneDataset, collate_planes, plane_tokens
from analysis.flavour.v2.partitions import row_keys


TIMING_NAMES = ['native_read', 'staging', 'decode', 'tokenization', 'packed_read']


def cache_key(geometry_id, config):
    return {'geometry_id': geometry_id, 'cache_format_revision': 2,
            'readout_revision': config.get('readout_revision', 'accepted_native_2026-11-11'),
            'representation_revision': 'plane_tokens_v1',
            **{key: config.get(key, default) for key, default in [
                ('window_pixels',16),('pitch_mm',0.),('modalities',['pixel','bar']),
                ('signal',True),('local_residual',True),('region_blocks',32)]}}


class TimedPlaneDataset(PlaneDataset):
    def __getitem__(self, index):
        before = self.native.cache_stats()
        started = time.perf_counter()
        event = self.native[int(self.indexes[index])]
        native_seconds = time.perf_counter()-started
        after = self.native.cache_stats()
        started = time.perf_counter()
        nodes, groups, modules, counts = plane_tokens(event['observations'], self.geometry, self.config, seed=index+42)
        transform_seconds = time.perf_counter()-started
        return {'nodes':nodes,'groups':groups,'modules':modules,'raw_counts':counts,
                'label':event['targets']['label'],'index':int(index),
                'region_blocks':self.config.get('region_blocks',32),
                '_timing':np.array([native_seconds,after['stage_seconds']-before['stage_seconds'],
                    after['decode_seconds']-before['decode_seconds'],transform_seconds,0.]),
                '_signals': np.array([np.asarray(event['observations']['bar'][key],dtype='float64').sum()
                                      for key in ['edep_proxy_MeV','pe_proxy']])}


def collate_profile(events):
    started = time.perf_counter()
    batch = collate_planes(events)
    seconds = time.perf_counter()-started
    timing = np.sum([e['_timing'] for e in events],axis=0)
    batch['_profile'] = {'worker_seconds':timing.tolist(),'collate_seconds':seconds,
                         'represented_channels':int(batch['nodes'][...,15].sum()),
                         'padded_channel_slots':batch['local_mask'].numel(),
                         'real_blocks':len(batch['nodes']),
                         'padded_event_blocks':batch['event_mask'].numel()}
    return batch


def signal_readback(item):
    active = item['nodes'][item['nodes'][:,15]>0]
    represented = [int(np.count_nonzero(active[:,12+i])) for i in range(3)]
    if represented != list(item['raw_counts']):
        raise ValueError('native channels lost or duplicated in plane transform')
    bars = active[active[:,13]>0]
    recovered = np.array([np.expm1(bars[:,column].astype('float64')*scale).sum()
                          for column,scale in [(6,10),(7,15)]])
    difference = np.abs(recovered-item['_signals']) / np.maximum(np.abs(item['_signals']),1e-9)
    if np.any(difference>2e-5):
        raise ValueError('plane feature bar signal readback exceeds float32 log tolerance')
    return difference


def build_packed_cache(data, destination, canonical_indexes=None):
    """One shared local writer; all channels remain in contiguous ragged arrays."""
    destination = Path(destination)
    geometry_ids = sorted(set(data.metadata.geometry_id))
    if len(geometry_ids)!=1:
        raise ValueError('one geometry revision per transform cache is required')
    key = cache_key(geometry_ids[0],data.config)
    started=time.perf_counter()
    totals=np.zeros(5); max_signal=np.zeros(2); events=0
    native_readback_events=0
    canonical_indexes=set(range(len(data)) if canonical_indexes is None else canonical_indexes)
    for shard, path in enumerate(data.paths):
        indexes=np.flatnonzero((data.indexes>=data.native.boundaries[shard]) &
                               (data.indexes<data.native.boundaries[shard+1]))
        if len(indexes)!=data.native.boundaries[shard+1]-data.native.boundaries[shard]:
            raise ValueError('R0 cache requires complete profile shards')
        indexes=indexes[np.argsort(data.indexes[indexes])]
        arrays={name:[] for name in ['nodes','groups','modules']}
        offsets=[0]; counts=[]
        for index in indexes:
            item=data[int(index)]
            max_signal=np.maximum(max_signal,signal_readback(item))
            # Compare the unchanged canonical frontend with the original S2
            # grouping enabled. This catches channel/order/mask differences.
            if int(index) in canonical_indexes:
                observation=data.native[int(data.indexes[index])]['observations']
                original=plane_tokens(observation,data.geometry,data.config,seed=int(index)+42,build_input_blocks=True)
                for name,value in zip(['nodes','groups','modules'],original[:3]):
                    if not np.array_equal(item[name],value):
                        raise ValueError('original/cached canonical transform differs: '+name)
                native_readback_events+=1
            for name in arrays:arrays[name].append(item[name])
            offsets.append(offsets[-1]+len(item['nodes']));counts.append(item['raw_counts'])
            totals+=item['_timing'];events+=1
        output=destination/path.name
        output.mkdir(parents=True,exist_ok=True)
        for name,values in arrays.items():np.save(output/(name+'.npy'),np.concatenate(values),allow_pickle=False)
        np.save(output/'offsets.npy',np.asarray(offsets,dtype='int64'),allow_pickle=False)
        np.save(output/'raw_counts.npy',np.asarray(counts,dtype='int64'),allow_pickle=False)
        np.save(output/'event_keys.npy',row_keys(data.metadata.iloc[indexes]),allow_pickle=False)
    result={'cache_key':key,'events':events,'seconds':time.perf_counter()-started,
            'native_source_root':str(data.root.resolve()),
            'worker_seconds':dict(zip(TIMING_NAMES,map(float,totals))),
            'original_canonical_array_readbacks':native_readback_events,
            'channel_coverage_and_response_masks_equal':True,
            'max_relative_bar_signal_error':max_signal.tolist(),
            'bytes':sum(p.stat().st_size for p in destination.rglob('*.npy')),
            'storage':'packed ragged per-event arrays; no labels/truth/provenance in feature arrays'}
    write_json(destination/'metadata.json',result)
    return result


class PackedPlaneDataset(TimedPlaneDataset):
    def __init__(self,*args,packed_root,**kwargs):
        super().__init__(*args,**kwargs)
        self.packed_root=Path(packed_root)
        metadata=json.loads((self.packed_root/'metadata.json').read_text())
        geometry_ids=sorted(set(self.metadata.geometry_id))
        if len(geometry_ids)!=1 or metadata['cache_key']!=cache_key(geometry_ids[0],self.config):
            raise ValueError('geometry/readout/representation cache key mismatch')
        if metadata['native_source_root']!=str(self.root.resolve()):
            raise ValueError('packed cache native source root mismatch')
        for shard,path in enumerate(self.paths):
            selected=np.flatnonzero((self.indexes>=self.native.boundaries[shard]) &
                                    (self.indexes<self.native.boundaries[shard+1]))
            keys=np.load(self.packed_root/path.name/'event_keys.npy',mmap_mode='r',allow_pickle=False)
            if not np.array_equal(keys[self.indexes[selected]-self.native.boundaries[shard]],
                                  row_keys(self.metadata.iloc[selected])):
                raise ValueError('packed cache event identity mismatch')
        self._packed=OrderedDict()

    def __getstate__(self):
        state=self.__dict__.copy()
        state['_packed']=OrderedDict()
        return state

    def __getitem__(self,index):
        started=time.perf_counter()
        native_index=int(self.indexes[index])
        shard=int(np.searchsorted(self.native.boundaries,native_index,side='right')-1)
        local=native_index-self.native.boundaries[shard]
        if shard not in self._packed:
            if len(self._packed)>=self.config.get('cache_shards',2):self._packed.popitem(last=False)
            self._packed[shard]={name:np.load(self.packed_root/self.paths[shard].name/(name+'.npy'),
                mmap_mode='r',allow_pickle=False) for name in ['nodes','groups','modules','offsets','raw_counts']}
        self._packed.move_to_end(shard)
        arrays=self._packed[shard];left,right=arrays['offsets'][local:local+2]
        item={name:arrays[name][left:right] for name in ['nodes','groups','modules']}
        item.update(raw_counts=arrays['raw_counts'][local],label=int(self.metadata.iloc[index].label),
                    index=int(index),region_blocks=self.config.get('region_blocks',32),
                    _timing=np.array([0.,0.,0.,0.,time.perf_counter()-started]),_signals=np.zeros(2))
        return item


def verify_cache(native,packed,batch_size=16):
    """All collated tensors include indices/labels/masks, not just model logits."""
    count=0
    for start in range(0,len(native),batch_size):
        indexes=range(start,min(start+batch_size,len(native)))
        a=collate_planes([native[i] for i in indexes]);b=collate_planes([packed[i] for i in indexes])
        for key in a:
            if not torch.equal(a[key],b[key]):raise ValueError('packed cache tensor differs: '+key)
        count+=len(a['index'])
    return {'events':count,'all_collated_tensors_bitwise_equal':True}


def main():
    parser=argparse.ArgumentParser(description='Verify the final exact-cache format on the fixed micro sample')
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--candidates',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    config=next(row['config'] for row in json.loads(args.candidates.read_text())['models']
                if row['name']=='s4_empty_scratch_n100000_s42')
    with tempfile.TemporaryDirectory(prefix='pinpoint-r0-cache-check-') as temporary:
        scratch=Path(temporary)
        data=TimedPlaneDataset(args.root/'micro_data','train',config,cache_dir=scratch/'native')
        result=build_packed_cache(data,scratch/'packed')
        packed=PackedPlaneDataset(args.root/'micro_data','train',config,packed_root=scratch/'packed')
        result['equivalence']=verify_cache(data,packed)
        result['event_keys_checked']=len(data)
        data.native.close()
    result.update(status='complete',temporary_scratch_removed=not scratch.exists())
    write_json(args.root/'profiles/final_cache_readback.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
