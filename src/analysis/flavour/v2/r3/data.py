"""Allocation-local transformed inputs and exact-cache evidence."""

import os
import pickle
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from analysis.flavour.v2.profile import init_worker
from analysis.flavour.s2.representation import token_inputs
from analysis.flavour.v2.r1.data import R1Dataset, collate as original_collate
from analysis.flavour.v2.r3.relations import candidate_edges


def keep_events(events):
    return events


class R3Dataset(R1Dataset):
    def __init__(self, root, split, config, cache_dir=None):
        super().__init__(root, split, config, cache_dir)
        self.items = None
        if split == 'train' and config.get('cache_training_inputs', False):
            shared=os.environ.get('PINPOINT_R3_PROFILE_INPUTS')
            if shared:
                started=time.perf_counter()
                with Path(shared).open('rb') as stream:payload=pickle.load(stream)
                expected={key:config.get(key) for key in payload['input_config']}
                if (payload['input_config']!=expected or payload['root']!=str(Path(root).resolve())
                        or not np.array_equal(payload['keys'],self.metadata[['run','chunk','row','label']].to_numpy())):
                    raise ValueError('shared profile cache differs from actual model input contract')
                self.items=payload['items'];self.native.close()
                self.cache_audit={**payload['audit'],'shared_profile_inputs':True,
                    'shared_original_build_seconds':payload['audit']['seconds'],'seconds':time.perf_counter()-started}
            else:
                self.cache_inputs()

    def __getitem__(self, index):
        if self.items is not None:
            return self.items[index]
        item=super().__getitem__(index)
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
            'storage':'allocation-local transformed arrays; no persistent EOS tensor cache'}
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
