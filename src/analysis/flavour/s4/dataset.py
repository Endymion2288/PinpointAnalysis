"""Visible-only masks and geometry/visible query sampling, independent of targets."""
import json
from collections import OrderedDict
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from analysis.flavour.s3.dataset import PlaneDataset, plane_tokens, collate_planes


def hash_uniform(keys, seed):
    """Stateless geometry field: absent and occupied cells follow the same mask."""
    x = np.asarray(keys, dtype='uint64') + np.uint64(int(seed) % (2**63))
    with np.errstate(over='ignore'):
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
    return ((x ^ (x >> np.uint64(31))) >> np.uint64(11)).astype('float64') / 2**53


class QueryGeometry:
    def __init__(self, geometry, tile_pixels=16, pixel_query_patch=1):
        self.geometry = geometry
        self.nx, self.ny, self.nz = [len(geometry[k]) for k in ['pixel_Xpos','pixel_Ypos','pixel_Zpos']]
        self.x, self.y, self.z = [np.asarray(geometry[k]) for k in ['pixel_Xpos','pixel_Ypos','pixel_Zpos']]
        self.tile = tile_pixels
        self.pixel_query_patch=pixel_query_patch
        self.qx,self.qy=(self.nx+pixel_query_patch-1)//pixel_query_patch,(self.ny+pixel_query_patch-1)//pixel_query_patch
        self.tx, self.ty = (self.nx+tile_pixels-1)//tile_pixels, (self.ny+tile_pixels-1)//tile_pixels
        self.dx, self.dy = np.median(np.diff(self.x)), np.median(np.diff(self.y))
        self.cx, self.cy = int(np.ceil(self.nx*self.dx/4)), int(np.ceil(self.ny*self.dy/4))
        # G4 placement order: one vertical panel in each scint-bearing pixel
        # module; 2*numScintLayers panels in Fortune. Trailing IPT has no bar.
        layer, orientation, panel = [], [], 0
        for module, pixel in enumerate(geometry['layer_is_pixel']):
            if panel == len(geometry['scint_Zpos']):
                break
            width = 1 if pixel else 2*geometry['numScintLayers']
            layer.extend([module]*width)
            orientation.extend([0] if pixel else [j%2 for j in range(width)])
            panel += width
        if panel != len(geometry['scint_Zpos']):
            raise ValueError('panel layout disagrees with accepted geometry')
        self.bar_layer, self.bar_orientation = np.asarray(layer), np.asarray(orientation)
        self.nb = len(geometry['scint_bar_Xcenter'])
        p, b = np.indices((panel, self.nb))
        self.bar_ids = ((self.bar_layer[p]*panel+p)*2*self.nb+self.bar_orientation[p]*self.nb+b).reshape(-1)

    def pixel_parts(self, ids):
        ids = np.asarray(ids, dtype='int64')
        return ids%self.nx, (ids//self.nx)%self.ny, ids//(self.nx*self.ny)

    def mask(self, detector, ids, seed, ratio):
        ids = np.asarray(ids, dtype='int64')
        if detector == 0:
            col,row,plane = self.pixel_parts(ids)
            cells = (plane*self.ty+row//self.tile)*self.tx+col//self.tile
        elif detector == 1:
            # Hardware codec stores orientation between panel and bar.
            panel = ids//(2*self.nb) % len(self.geometry['scint_Zpos'])
            plane = panel
            cells = panel*16+(ids%self.nb)//4
        else:
            # Tracker IDs encode station and fixed 16-mm observation cells.
            plane = ids//1000000
            cells = ids
        result = hash_uniform(cells+detector*10**12, seed) < ratio
        # Ten percent of views add a full detector mask; ten percent add a
        # geometry-selected plane/panel mask. Neither depends on active support.
        mode = seed % 10
        if mode == 0 and detector == (seed//10)%3:
            result = np.ones_like(result, dtype=bool)
        if mode == 1:
            result |= hash_uniform(plane+detector*1000, seed+31) < .2
        return result

    def tracker_ids(self, observation):
        t = observation['tracker_hypothesis']
        xy = np.asarray(t['xy_mm']).reshape(-1,2)
        cells = np.floor(xy/16).astype('int64')+500
        return np.asarray(t['station'])*1000000+cells[:,1]*1000+cells[:,0]

    def visible(self, observation, seed, ratio):
        output = {}
        masks = []
        for detector,name,field in [(0,'pixel','channel_id'),(1,'bar','channel_id'),(2,'tracker_hypothesis',None)]:
            ids = observation[name][field] if field else self.tracker_ids(observation)
            hidden = self.mask(detector,ids,seed,ratio)
            output[name] = {key:np.asarray(values)[~hidden] for key,values in observation[name].items()}
            masks.append(hidden)
        return output, masks

    def queries(self, visible, seed, ratio, per_kind=64):
        """Draw positions before looking up ANY full-event reconstruction target.

        Each kind has 3/4 visible-neighborhood and 1/4 uniform geometry draws;
        if the visible modality is empty, all draws are uniform. Rejection only
        enforces geometry and mask, never occupancy. Duplicates are retained as
        legitimate repeated draws. The objective is this mixture distribution.
        """
        rng = np.random.default_rng(seed+701)
        features, ids_out, levels, distributions = [], [], [], []
        for level in range(3):
            accepted, attempts = [], 0
            while len(accepted)<per_kind:
                n = max(4*per_kind,32)
                near = rng.random(n)<.75
                if level<2:
                    pixel_ids = np.asarray(visible['pixel']['channel_id'],dtype='int64')
                    if len(pixel_ids):
                        centers = pixel_ids[rng.integers(len(pixel_ids),size=n)]
                        col,row,plane = self.pixel_parts(centers)
                    else:
                        near[:] = False
                        col,row,plane = np.zeros(n,dtype='int64'),np.zeros(n,dtype='int64'),np.zeros(n,dtype='int64')
                    if level==0:
                        step = rng.choice([-32,-16,-8,-4,-2,-1,1,2,4,8,16,32],size=(2,n))
                        col,row = col+step[0],row+step[1]
                        plane = plane+rng.choice([-1,0,0,0,1],size=n)
                        col[~near],row[~near],plane[~near] = (rng.integers(limit,size=(~near).sum()) for limit in [self.nx,self.ny,self.nz])
                        valid = (col>=0)&(col<self.nx)&(row>=0)&(row<self.ny)&(plane>=0)&(plane<self.nz)
                        channel_ids = (plane*self.ny+row)*self.nx+col
                        valid &= self.mask(0,channel_ids,seed,ratio)
                        ids = (plane*self.qy+row//self.pixel_query_patch)*self.qx+col//self.pixel_query_patch
                    else:
                        col = np.floor((self.x[np.clip(col,0,self.nx-1)]-self.x[0])/4).astype('int64')
                        row = np.floor((self.y[np.clip(row,0,self.ny-1)]-self.y[0])/4).astype('int64')
                        col += rng.integers(-1,2,n);row += rng.integers(-1,2,n)
                        plane += rng.choice([-1,0,1],size=n)
                        col[~near],row[~near],plane[~near] = (rng.integers(limit,size=(~near).sum()) for limit in [self.cx,self.cy,self.nz])
                        valid = (col>=0)&(col<self.cx)&(row>=0)&(row<self.cy)&(plane>=0)&(plane<self.nz)
                        ids = (plane*self.cy+row)*self.cx+col
                else:
                    bars = visible['bar']
                    if len(bars['channel_id']):
                        choose = rng.integers(len(bars['channel_id']),size=n)
                        panel = np.asarray(bars['panel'])[choose]+rng.integers(-2,3,size=n)
                        bar = np.asarray(bars['bar'])[choose]+rng.integers(-2,3,size=n)
                    else:
                        near[:]=False;panel=np.zeros(n,dtype='int64');bar=np.zeros(n,dtype='int64')
                    panel[~near]=rng.integers(len(self.bar_layer),size=(~near).sum())
                    bar[~near]=rng.integers(self.nb,size=(~near).sum())
                    valid = (panel>=0)&(panel<len(self.bar_layer))&(bar>=0)&(bar<self.nb)
                    safe=np.clip(panel,0,len(self.bar_layer)-1)
                    ids=((self.bar_layer[safe]*len(self.bar_layer)+safe)*2*self.nb+self.bar_orientation[safe]*self.nb+bar)
                    valid &= self.mask(1,ids,seed,ratio)
                accepted.extend((int(i),int(d)) for i,d in zip(ids[valid],near[valid]))
                attempts += 1
                # A mask may leave no hidden bar/pixel cells only for tiny toy
                # geometries; explicit error is preferable to a support-based fallback.
                if attempts>1000:
                    raise RuntimeError('no geometry-only masked query candidates')
            selected = accepted[:per_kind]
            ids = np.asarray([x[0] for x in selected],dtype='int64')
            q=np.zeros((per_kind,22),dtype='float32')
            q[:,18+level]=1
            if level<2:
                if level==0:
                    col,row,plane=ids%self.qx,(ids//self.qx)%self.qy,ids//(self.qx*self.qy)
                    x=self.x[0]+(col*self.pixel_query_patch+(self.pixel_query_patch-1)/2)*self.dx
                    y=self.y[0]+(row*self.pixel_query_patch+(self.pixel_query_patch-1)/2)*self.dy
                    q[:,8:10]=[self.pixel_query_patch*self.dx/420,self.pixel_query_patch*self.dy/420]
                else:
                    col,row,plane=ids%self.cx,(ids//self.cx)%self.cy,ids//(self.cx*self.cy)
                    x,y=self.x[0]+(col+.5)*4,self.y[0]+(row+.5)*4
                    q[:,8:10]=4/420
                q[:,:3]=np.column_stack([x,y,self.z[plane]])/[210,210,3000]
                q[:,3:5]=1;q[:,12]=1
            else:
                panel=ids//(2*self.nb)%len(self.bar_layer);bar=ids%self.nb
                h=ids//self.nb%2
                x=np.where(h,np.mean(self.geometry['scint_bar_Xcenter']),np.asarray(self.geometry['scint_bar_Xcenter'])[bar])
                y=np.where(h,np.asarray(self.geometry['scint_bar_Ycenter'])[bar],np.mean(self.geometry['scint_bar_Ycenter']))
                q[:,:3]=np.column_stack([x,y,np.asarray(self.geometry['scint_Zpos'])[panel]])/[210,210,3000]
                q[:,3]=1-h;q[:,4]=h;q[:,8]=np.where(h,1,self.geometry['scint_bar_width']/420)
                q[:,9]=np.where(h,self.geometry['scint_bar_height']/420,1);q[:,13]=1
            features.append(q);ids_out.extend(ids);levels.extend([level]*per_kind)
            distributions.extend([d for _,d in selected])
        return {'query':np.concatenate(features),'query_id':np.asarray(ids_out),
            'query_level':np.asarray(levels),'query_near':np.asarray(distributions,dtype=bool)}

    def targets(self, full, hidden, queries):
        """Only this method reads hidden support; its outputs go solely to losses."""
        ids,level=queries['query_id'],queries['query_level']
        target=np.zeros((len(ids),4),dtype='float32')
        pixel=np.asarray(full['pixel']['channel_id'],dtype='int64')[hidden[0]]
        col,row,plane=self.pixel_parts(pixel)
        patch=(plane*self.qy+row//self.pixel_query_patch)*self.qx+col//self.pixel_query_patch
        unique,inverse,counts=np.unique(patch,return_inverse=True,return_counts=True)
        mapping={key:i for i,key in enumerate(unique)}
        offsets=np.column_stack([(col%self.pixel_query_patch+.5)/self.pixel_query_patch-.5,
            (row%self.pixel_query_patch+.5)/self.pixel_query_patch-.5])
        moments=np.zeros((len(unique),2))
        np.add.at(moments,inverse,offsets)
        if len(counts):moments/=counts[:,None]
        for j in np.flatnonzero(level==0):
            k=mapping.get(ids[j])
            if k is not None:
                target[j,0]=1;target[j,1]=np.log1p(counts[k])/10;target[j,2:4]=moments[k]
        xcell=np.floor((self.x[col]-self.x[0])/4).astype('int64')
        ycell=np.floor((self.y[row]-self.y[0])/4).astype('int64')
        coarse=(plane*self.cy+ycell)*self.cx+xcell
        unique,counts=np.unique(coarse,return_counts=True)
        count_map=dict(zip(unique,counts))
        counts=np.asarray([count_map.get(i,0) for i in ids[level==1]])
        target[level==1,0]=counts>0;target[level==1,1]=np.log1p(counts)/10
        bar=full['bar'];barids=np.asarray(bar['channel_id'])[hidden[1]]
        mapping={int(i):j for j,i in enumerate(barids)}
        signal=np.column_stack([np.asarray(bar['edep_proxy_MeV'])[hidden[1]],np.asarray(bar['pe_proxy'])[hidden[1]]])
        for j in np.flatnonzero(level==2):
            k=mapping.get(int(ids[j]))
            if k is not None:
                target[j,0]=1;target[j,1]=np.log(2)/10
                target[j,2:4]=np.log1p(signal[k])/[10,15]
        return target


def restrict_dataset(dataset, selected):
    dataset.indexes=dataset.indexes[selected]
    dataset.metadata=dataset.metadata.iloc[selected].reset_index(drop=True)
    if dataset.workload is not None:
        dataset.workload=dataset.workload[selected]


class LabelDataset(PlaneDataset):
    def __init__(self,root,split,config,cache_dir=None):
        super().__init__(root,split,config,cache_dir)
        if split=='train' and config.get('label_budget',100000)<len(self):
            rng=np.random.default_rng(config.get('label_subset_seed',20261113))
            chosen=[]
            # A single seed/order makes every label budget nested. Tau strata
            # receive proportionate quotas, retaining all e/mu/had branches.
            for label in range(4):
                ids=np.flatnonzero(self.metadata.label.to_numpy()==label)
                if label==2 and 'tau_branch' in self.metadata:
                    strata=self.metadata.iloc[ids].tau_branch.to_numpy()
                    ordered=[]
                    groups={key:list(rng.permutation(ids[strata==key])) for key in sorted(set(strata))}
                    used={key:0 for key in groups}
                    for i in range(len(ids)):
                        key=max(groups,key=lambda k:(i+1)*len(groups[k])/len(ids)-used[k])
                        ordered.append(groups[key][used[key]]);used[key]+=1
                    ids=np.asarray(ordered)
                else:
                    ids=rng.permutation(ids)
                chosen.extend(ids[:config['label_budget']//4])
            restrict_dataset(self,np.sort(chosen))
        # Classification inputs have fixed transforms and seeds. SSL uses the
        # separate PretrainDataset, whose masks must never enter this cache.
        self.event_cache_limit = int(os.environ.get('PINPOINT_CLASSIFICATION_CACHE_MIB', '0')) * 2**20
        self._reset_event_cache()

    def _reset_event_cache(self):
        self._event_cache = OrderedDict()
        self._event_cache_bytes = 0
        self._event_cache_pid = os.getpid()
        self._event_cache_hits = self._event_cache_misses = self._event_cache_evictions = 0
        self._native_released_after_complete_cache = False

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_event_cache=OrderedDict(), _event_cache_bytes=0, _event_cache_pid=None,
            _event_cache_hits=0, _event_cache_misses=0, _event_cache_evictions=0,
            _native_released_after_complete_cache=False)
        return state

    def __getitem__(self, index):
        if self.event_cache_limit <= 0 or len(self) > 10000:
            return super().__getitem__(index)
        if self._event_cache_pid != os.getpid():
            self._reset_event_cache()
        index = int(index)
        if index in self._event_cache:
            self._event_cache_hits += 1
            self._event_cache.move_to_end(index)
            return self._event_cache[index][0].copy()
        self._event_cache_misses += 1
        item = super().__getitem__(index)
        size = sum(value.nbytes for value in item.values() if isinstance(value, np.ndarray))
        if size <= self.event_cache_limit:
            while self._event_cache_bytes + size > self.event_cache_limit:
                _, (_, removed) = self._event_cache.popitem(last=False)
                self._event_cache_bytes -= removed
                self._event_cache_evictions += 1
            for value in item.values():
                if isinstance(value, np.ndarray):
                    value.flags.writeable = False
            self._event_cache[index] = (item, size)
            self._event_cache_bytes += size
            if len(self._event_cache) == len(self) and not self._native_released_after_complete_cache:
                self.native.close()
                self._native_released_after_complete_cache = True
        return item.copy()

    def event_cache_stats(self):
        return {'enabled': self.event_cache_limit > 0 and len(self) <= 10000,
            'limit_array_bytes': self.event_cache_limit, 'resident_array_bytes': self._event_cache_bytes,
            'events': len(self._event_cache), 'hits': self._event_cache_hits,
            'misses': self._event_cache_misses, 'evictions': self._event_cache_evictions,
            'native_released_after_complete_cache': self._native_released_after_complete_cache,
            'scope': 'process-owned immutable classification arrays; Python container overhead excluded; no SSL views cached'}


class PretrainDataset(PlaneDataset):
    def __init__(self,root,split,config,cache_dir=None):
        # Both sides of SSL live within the original train partition.
        super().__init__(root,'train',config,cache_dir)
        self.query_geometry=QueryGeometry(self.geometry,pixel_query_patch=config.get('pixel_query_patch',1))
        self.epoch=torch.zeros((),dtype=torch.int64).share_memory_()
        rng=np.random.default_rng(20261114)
        order=rng.permutation(len(self))
        selected=order[2000:] if split=='train' else order[:2000]
        restrict_dataset(self,np.sort(selected))
        self.pretrain_split=split
        self.rel_cache={}

    def __getitem__(self,index):
        native_index=int(self.indexes[index])
        event=self.native[native_index]
        full=event['observations']
        view_epoch=(int(self.epoch)+self.config.get('mask_epoch_offset',0)) if self.pretrain_split=='train' else 0
        seed=int(self.config.get('seed',42))+index*1009+view_epoch*10000019
        visible,hidden=self.query_geometry.visible(full,seed,self.config.get('mask_ratio',.75))
        nodes,groups,modules,counts=plane_tokens(visible,self.geometry,self.config,seed)
        item={'nodes':nodes,'groups':groups,'modules':modules,'label':-1,'index':index,
            'raw_counts':counts,'region_blocks':self.config.get('region_blocks',32)}
        queries=self.query_geometry.queries(visible,seed,self.config.get('mask_ratio',.75),self.config.get('queries_per_kind',64))
        item.update(queries)
        item['reconstruction']=self.query_geometry.targets(full,hidden,queries)
        item['mask_counts']=np.asarray([[len(h),int(h.sum())] for h in hidden])
        # Unique target coverage is a diagnostic only, computed after queries;
        # it is never used to select positions or to build encoder inputs.
        coverage=[]
        for level in range(3):
            take=(queries['query_level']==level)&(item['reconstruction'][:,0]>0)
            coverage.append(len(np.unique(queries['query_id'][take])))
        item['query_positive_unique']=np.asarray(coverage)
        if self.config.get('objective')=='distill':
            n,g,m,c=plane_tokens(full,self.geometry,self.config,seed)
            item['teacher']={'nodes':n,'groups':g,'modules':m,'raw_counts':c,'label':-1,'index':index,
                'region_blocks':self.config.get('region_blocks',32)}
        if self.config.get('objective')=='mae_rel':
            shard=np.searchsorted(self.native.boundaries,native_index,side='right')-1
            if shard not in self.rel_cache:
                # Keep two compact sidecars, independent of the observation LRU.
                if len(self.rel_cache)>=2:self.rel_cache.pop(next(iter(self.rel_cache)))
                table=pq.read_table(self.root/'rel'/f'{shard:03d}.parquet')
                channel=table['channel_id'].combine_chunks()
                self.rel_cache[shard]={'source_row':table['source_row'].to_numpy(),
                    'offsets':channel.offsets.to_numpy(),'channel_id':channel.flatten().to_numpy(),
                    'fractions':table['fractions'].combine_chunks().flatten().flatten().to_numpy().reshape(-1,4),
                    'valid':table['valid'].combine_chunks().flatten().to_numpy(zero_copy_only=False)}
            cache=self.rel_cache[shard]
            source_row=native_index-self.native.boundaries[shard]
            row_index=int(np.searchsorted(cache['source_row'],source_row))
            if row_index>=len(cache['source_row']) or cache['source_row'][row_index]!=source_row:
                raise ValueError('train event lacks its bar contribution sidecar row')
            left,right=cache['offsets'][row_index:row_index+2]
            row={key:cache[key][left:right] for key in ['channel_id','fractions','valid']}
            # Rel queries are visible bars, as in FASERCal's kept-patch targets.
            ids=np.asarray(visible['bar']['channel_id'])
            take=np.arange(len(ids))
            if len(take)>128:
                take=np.random.default_rng(seed+71).choice(take,128,replace=False)
            # Query construction is geometry-only and contains no energy/PDG.
            rel_ids=ids[take]
            rel_q=np.zeros((128,22),dtype='float32');rel_target=np.zeros((128,4),dtype='float32')
            valid=np.zeros(128,dtype=bool)
            bars=visible['bar'];bounds=np.asarray(bars['bounds_mm']).reshape(-1,5)[take]
            h=np.asarray(bars['horizontal'])[take]
            n=len(take)
            rel_q[:n,:3]=np.column_stack([(bounds[:,0]+bounds[:,1])/2,(bounds[:,2]+bounds[:,3])/2,bounds[:,4]])/[210,210,3000]
            rel_q[:n,3]=1-h;rel_q[:n,4]=h
            rel_q[:n,8:10]=np.column_stack([bounds[:,1]-bounds[:,0],bounds[:,3]-bounds[:,2]])/420
            rel_q[:n,13]=1;rel_q[:n,20]=1
            selected=np.searchsorted(row['channel_id'],rel_ids)
            if len(selected) and not np.array_equal(row['channel_id'][selected],rel_ids):
                raise ValueError('visible bar has no matching contribution target')
            rel_target[:n]=row['fractions'][selected];valid[:n]=row['valid'][selected]
            item.update(rel_query=rel_q,rel_target=rel_target,rel_valid=valid)
        return item


def collate_pretrain(items):
    batch={'student':collate_planes(items)}
    batch['student'].pop('label');batch['student'].pop('index')
    for key in ['query','query_id','query_level','query_near','reconstruction','mask_counts']:
        batch[key]=torch.from_numpy(np.stack([item[key] for item in items]))
    if 'query_positive_unique' in items[0]:
        batch['query_positive_unique']=torch.from_numpy(np.stack([item['query_positive_unique'] for item in items]))
    if 'teacher' in items[0]:
        batch['teacher']=collate_planes([i['teacher'] for i in items])
        batch['teacher'].pop('label');batch['teacher'].pop('index')
    if 'rel_query' in items[0]:
        for key in ['rel_query','rel_target','rel_valid']:
            batch[key]=torch.from_numpy(np.stack([i[key] for i in items]))
    return batch
