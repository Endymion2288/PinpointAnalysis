"""Allocation-local transformed inputs and exact-cache evidence."""

import os
import pickle
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from analysis.flavour.v2.profile import init_worker
from analysis.flavour.s2.representation import token_inputs
from analysis.flavour.v2.r1.data import R1Dataset, collate as original_collate
from analysis.flavour.v2.r3.relations import candidate_edges
from analysis.flavour.v2.r3.sensor_rows import sensor_projection
from analysis.flavour.v2.r3.mapped_inputs import write_mapped_items


def keep_events(events):
    return events


class R3Dataset(R1Dataset):
    def __init__(self, root, split, config, cache_dir=None):
        super().__init__(root, split, config, cache_dir)
        self.cache_parent=cache_dir or os.environ.get('PINPOINT_V2_SCRATCH','/tmp')
        self.items = None
        self.cache_positions=None
        shared=(os.environ.get('PINPOINT_R3_SHARED_TUNE_INPUTS') if split=='validation' else
            os.environ.get('PINPOINT_R3_SHARED_TRAIN_INPUTS',os.environ.get('PINPOINT_R3_PROFILE_INPUTS')))
        if shared or (split == 'train' and config.get('cache_training_inputs', False)):
            if shared:
                started=time.perf_counter()
                with Path(shared).open('rb') as stream:payload=pickle.load(stream)
                expected={key:config.get(key) for key in payload['input_config']}
                if (payload['input_config']!=expected or payload['root']!=str(Path(root).resolve())
                        or not np.array_equal(payload['keys'],self.metadata[['run','chunk','row','label']].to_numpy())):
                    raise ValueError('shared profile cache differs from actual model input contract')
                self.items=payload['items'];self.native.close()
                self.cache_positions={int(native):i for i,native in enumerate(self.indexes)}
                self.cache_audit={**payload['audit'],'shared_profile_inputs':True,
                    'shared_original_build_seconds':payload['audit']['seconds'],'seconds':time.perf_counter()-started}
            else:
                self.cache_inputs()

    def __getitem__(self, index):
        if self.items is not None:
            position=index if self.cache_positions is None else self.cache_positions[int(self.indexes[index])]
            return {**self.items[position],'index':index}
        item=super().__getitem__(index)
        if self.config.get('sensor_row_image',False):
            observation=self.native[int(self.indexes[index])]['observations']
            item['projection_indices'],item['projection_values']=sensor_projection(
                observation,self.geometry,self.config['image'])
        if self.config['family'] in ['B2','B3'] and self.config.get('native_pixel_pitch_mm',0.)>0:
            observation=self.native[int(self.indexes[index])]['observations']
            item['nodes'],item['groups'],_=token_inputs(observation,self.geometry,
                pitch_mm=self.config['native_pixel_pitch_mm'],modalities=['pixel','bar'])
        if self.config.get('build_relations',False):
            pitch=self.config.get('native_pixel_pitch_mm',0.)
            widths=(pitch,pitch) if pitch else (.0208,.0228)
            item['relation_edges'],item['relation_candidate_summary']=candidate_edges(item['nodes'],item['groups'],widths,
                plane_policy=self.config.get('relation_plane_policy','all_visible'))
            edges=item['relation_edges'];groups=item['groups'];nodes=item['nodes']
            cross=np.abs(nodes[edges[0],2]-nodes[edges[1],2])>1e-6
            item['relation_candidate_summary'].update(
                real_blocks=len(np.unique(groups[nodes[:,15]>.5])),
                blocks_with_external_context=len(np.unique(groups[edges[0]])),
                blocks_with_cross_plane_context=len(np.unique(groups[edges[0,cross]])))
        return item

    def cache_inputs(self):
        started = time.perf_counter()
        generator = torch.Generator().manual_seed(self.config['seed']+7819)
        loader = DataLoader(self, batch_size=128, shuffle=False, collate_fn=keep_events,
            num_workers=self.config.get('cache_workers', 2), worker_init_fn=init_worker, generator=generator,
            multiprocessing_context='spawn', prefetch_factor=2)
        mapped=self.config.get('cache_storage')=='mmap'
        if mapped:
            directory=Path(tempfile.mkdtemp(prefix='r3-mapped-',dir=self.cache_parent))/'inputs'
            items=write_mapped_items(loader,directory)
        else:
            items = [item for group in loader for item in group]
        if [item['index'] for item in items] != list(range(len(self))):
            raise ValueError('transformed cache changed frozen event coverage')
        for index in np.linspace(0, len(self)-1, min(32, len(self)), dtype=int):
            a, b = collate([self[int(index)]]), collate([items[int(index)]])
            if any(not torch.equal(a[key], b[key]) for key in a if torch.is_tensor(a[key])):
                raise ValueError('transformed cache changed actual model inputs')
        self.cache_audit = {'events':len(items),
            'array_bytes':sum(v.nbytes for item in items for v in item.values() if isinstance(v, np.ndarray)),
            'seconds':time.perf_counter()-started, 'scattered_exact_input_checks':min(32, len(self)),
            'storage':('allocation-local file-backed read-only transformed arrays' if mapped
                else 'allocation-local transformed arrays; no persistent EOS tensor cache')}
        if self.config.get('build_relations',False):
            summaries=[item['relation_candidate_summary'] for item in items]
            totals={name:sum(s[name] for s in summaries) for name in [
                'nodes','nodes_with_any_edge','nodes_with_cross_plane_edge','real_blocks',
                'blocks_with_external_context','blocks_with_cross_plane_context']}
            self.cache_audit['candidate_graph']={'totals':totals,
                'maximum_out_degree':max(s['maximum_out_degree'] for s in summaries),
                'edge_count_quantiles':np.quantile([item['relation_edges'].shape[1] for item in items],[0,.5,.95,.99,1]).tolist(),
                'propagation':'one raw-node message layer followed by complete local-block pooling; block external/cross-plane context measures this actual path'}
        self.items = items
        self.native.close()


def collate(events):
    if events[0]['family'] not in ['B2','B3']:
        return original_collate(events)
    # Each block contains at most 32 nodes. The historical collator scans all
    # event nodes once per block, then repeats those scans for edge remapping.
    # Scatter directly into the same padded slots, preserving within-block
    # order even for callers that do not supply already contiguous groups.
    blocks=np.asarray([int(e['groups'].max())+1 for e in events])
    total=int(blocks.sum())
    nodes=np.zeros((total*32,18),dtype='float32')
    local_mask=np.zeros(total*32,dtype=bool)
    sources=[];targets=[];block_offset=0
    for event,nblocks in zip(events,blocks):
        groups=event['groups']
        counts=np.bincount(groups,minlength=nblocks)
        if np.any((counts<1)|(counts>32)):
            raise ValueError('invalid local block, no truncation permitted')
        order=(np.arange(len(groups)) if np.all(groups[1:]>=groups[:-1])
            else np.argsort(groups,kind='stable'))
        position=np.arange(len(groups))-np.repeat(np.cumsum(counts)-counts,counts)
        slots=(block_offset+groups[order])*32+position
        nodes[slots]=event['nodes'][order]
        local_mask[slots]=True
        if 'relation_edges' in event:
            mapping=np.empty(len(groups),dtype='int64')
            mapping[order]=slots
            edges=event['relation_edges']
            sources.append(mapping[edges[0]])
            targets.append(mapping[edges[1]])
        block_offset+=nblocks
    batch={'nodes':torch.from_numpy(nodes.reshape(total,32,18)),
        'local_mask':torch.from_numpy(local_mask.reshape(total,32)),
        'event_index':torch.from_numpy(np.repeat(np.arange(len(events)),blocks)),
        'block_index':torch.from_numpy(np.concatenate([np.arange(n) for n in blocks])),
        'event_mask':torch.from_numpy(np.arange(int(blocks.max()))[None,:]<blocks[:,None]),
        'label':torch.tensor([e['label'] for e in events]),
        'index':torch.tensor([e['index'] for e in events])}
    width=max(1,max(len(e['tracker']) for e in events))
    tracker=np.zeros((len(events),width,5),dtype='float32')
    tracker_mask=np.zeros((len(events),width),dtype=bool)
    for i,event in enumerate(events):
        tracker[i,:len(event['tracker'])]=event['tracker']
        tracker_mask[i,:len(event['tracker'])]=True
    batch.update(tracker=torch.from_numpy(tracker),tracker_mask=torch.from_numpy(tracker_mask),
        presence=torch.from_numpy(np.stack([e['presence'] for e in events])))
    if sources:
        batch['relation_source']=torch.from_numpy(np.concatenate(sources))
        batch['relation_target']=torch.from_numpy(np.concatenate(targets))
    return batch
