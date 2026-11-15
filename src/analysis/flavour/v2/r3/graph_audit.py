"""Train-only candidate coverage and recorded-first-contributor edge recall.

Truth is loaded only after the observation-only candidate graph exists. The
pixel trackID is the first contributor retained by PixelSD, not complete hit
ancestry. Recall is therefore conditional on those recorded labels.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile

import awkward as ak
import numpy as np
import pandas as pd
import uproot
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from analysis.flavour.inventory import COMPRESSION,write_json
from analysis.flavour.v2.r3.data import R3Dataset
from analysis.flavour.v2.r3.relations import candidate_edges


def recorded_recall(nodes, edges, tids, hops=3):
    pixel=(nodes[:,12]>.5)&(nodes[:,15]>.5)
    planes=np.unique(nodes[pixel,2])
    neighbors={}
    for a,b in edges.T:
        if pixel[a] and pixel[b]:neighbors.setdefault(int(a),set()).add(int(b))
    # Each source x target-plane is one opportunity, regardless of shower hit
    # multiplicity. A source has up to six nearest visible pixel-plane tests.
    eligible=np.zeros(hops,dtype='int64');found=eligible.copy()
    for pi,z in enumerate(planes):
        source=np.flatnonzero(pixel&(nodes[:,2]==z)&(tids>0))
        for distance in range(1,hops+1):
            for qi in [pi-distance,pi+distance]:
                if not 0<=qi<len(planes):continue
                target=np.flatnonzero(pixel&(nodes[:,2]==planes[qi])&(tids>0))
                tracks={int(t):set(target[tids[target]==t].tolist()) for t in np.unique(tids[target])}
                for a in source:
                    positive=tracks.get(int(tids[a]),set())
                    if positive:
                        eligible[distance-1]+=1
                        found[distance-1]+=bool(positive&neighbors.get(int(a),set()))
    return eligible,found


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--attempt',required=True)
    parser.add_argument('--events',type=int,default=1024)
    parser.add_argument('--compare-plane-policy',action='store_true')
    args=parser.parse_args()
    campaign=json.loads((args.root/'campaign.json').read_text())
    config={**json.loads(args.config.read_text()),'cache_training_inputs':False}
    r0=Path(campaign['r0'])
    output=args.root/'diagnostics'/('graph_'+args.attempt)
    output.mkdir(parents=True,exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='r3-graph-audit-') as scratch:
        data=R3Dataset(r0/'profile_data','train',config,cache_dir=scratch)
        rng=np.random.default_rng(20261120)
        positions=np.concatenate([rng.choice(np.flatnonzero(data.metadata.label.eq(c)),
            args.events//4,replace=False) for c in range(4)])
        selected=data.metadata.iloc[positions].copy()
        selected['position']=positions
        plan=json.loads((Path(campaign['r1'])/'data/plan.json').read_text())
        geometry=data.geometry
        nx,ny=len(geometry['pixel_Xpos']),len(geometry['pixel_Ypos'])
        policies=['all_visible','modality_visible'] if args.compare_plane_policy else [config.get('relation_plane_policy','all_visible')]
        audit=[];opportunities={p:np.zeros(3,dtype='int64') for p in policies}
        success={p:np.zeros(3,dtype='int64') for p in policies}
        for (run,chunk),group in selected.groupby(['run','chunk'],sort=True):
            # Build from measured observations before opening any truth labels.
            examples={int(row.row):data[int(row.position)] for row in group.itertuples()}
            source=Path(plan['input_root'])/str(run)/f'{run}_{chunk:03d}.root'
            with ThreadPoolExecutor(max_workers=2) as executor,uproot.open(source,decompression_executor=executor) as file:
                tree=file['Hits/pixelHits']
                for start in range(0,1000,64):
                    subset=group[group.row.ge(start)&group.row.lt(start+64)]
                    if subset.empty:continue
                    payload=tree.arrays(['layerID','rowID','colID','trackID'],entry_start=start,entry_stop=start+64,library='ak')
                    for row in subset.itertuples():
                        item=examples[int(row.row)];nodes=item['nodes'];edges=item['relation_edges']
                        hit={k:ak.to_numpy(payload[k][int(row.row)-start]) for k in payload.fields}
                        ids=(hit['layerID'].astype('int64')*ny+hit['rowID'])*nx+hit['colID']
                        unique,first=np.unique(ids,return_index=True)
                        xyz=nodes[:,:3]*[210.,210.,3000.]
                        pixel=np.flatnonzero(nodes[:,12]>.5)
                        # Nominal native pixel centers allow exact channel-ID
                        # recovery by nearest known coordinate, no track data.
                        def nearest(values,centers):
                            centers=np.asarray(centers);order=np.argsort(centers)
                            sorted_centers=centers[order];i=np.searchsorted(sorted_centers,values).clip(0,len(centers)-1)
                            j=(i-1).clip(0);return order[np.where(np.abs(sorted_centers[j]-values)<np.abs(sorted_centers[i]-values),j,i)]
                        channel=(nearest(xyz[pixel,2],geometry['pixel_Zpos'])*ny+
                            nearest(xyz[pixel,1],geometry['pixel_Ypos']))*nx+nearest(xyz[pixel,0],geometry['pixel_Xpos'])
                        ix=np.searchsorted(unique,channel)
                        if (ix>=len(unique)).any() or not np.array_equal(unique[ix],channel):
                            raise ValueError('native pixel/recorded-truth channel mismatch')
                        tids=np.full(len(nodes),-1,dtype='int64');tids[pixel]=hit['trackID'][first[ix]]
                        for policy in policies:
                            edges,counts=candidate_edges(nodes,item['groups'],plane_policy=policy)
                            eligible,found=recorded_recall(nodes,edges,tids)
                            opportunities[policy]+=eligible;success[policy]+=found
                            n=len(nodes);valid=nodes[:,15]>.5
                            degree=np.bincount(edges[0],minlength=n)
                            cross=np.abs(nodes[edges[0],2]-nodes[edges[1],2])>1e-6
                            graph=coo_matrix((np.ones(edges.shape[1]),edges),shape=(n,n))
                            components,labels=connected_components(graph,directed=False)
                            sizes=np.bincount(labels[valid],minlength=components)
                            groups=item['groups'];blocks=np.unique(groups[valid])
                            informed=np.unique(groups[edges[0]])
                            cross_informed=np.unique(groups[edges[0,cross]])
                            audit.append({'run':int(run),'chunk':int(chunk),'row':int(row.row),'label':int(row.label),'policy':policy,
                                **counts,'edges':edges.shape[1],
                                'real_blocks':len(blocks),'blocks_with_external_context':len(informed),
                                'blocks_with_cross_plane_context':len(cross_informed),
                                'pixel_nodes':len(pixel),'pixel_nodes_with_edge':int((degree[pixel]>0).sum()),
                                'cross_plane_edges':int(cross.sum()),'largest_component_fraction':float(sizes.max(initial=0)/max(1,valid.sum())),
                                **{f'eligible_pixel_gap{i+1}':int(v) for i,v in enumerate(eligible)},
                                **{f'found_pixel_gap{i+1}':int(v) for i,v in enumerate(found)},
                                'recorded_pixel_rows':len(ids),'duplicate_recorded_channels':len(ids)-len(unique)})
            write_json(output/'progress.json',{'events':len(audit)//len(policies),'expected':len(selected)})
        data.native.close()
    complete=pd.DataFrame(audit);complete.to_parquet(output/'events.parquet',index=False,compression=COMPRESSION)
    variants={}
    for policy,table in complete.groupby('policy'):
        variants[policy]={'events':len(table),'class_counts':table.label.value_counts().sort_index().tolist(),
        'edge_count_quantiles':dict(zip(['p0','p50','p95','p99','max'],np.quantile(table.edges,[0,.5,.95,.99,1]).tolist())),
        'node_weighted_any_edge_coverage':float(table.nodes_with_any_edge.sum()/table.nodes.sum()),
        'node_weighted_cross_plane_coverage':float(table.nodes_with_cross_plane_edge.sum()/table.nodes.sum()),
        'maximum_out_degree':int(table.maximum_out_degree.max()),
        'largest_component_fraction_mean':float(table.largest_component_fraction.mean()),
        'block_weighted_one_prelocal_layer_external_context_coverage':float(table.blocks_with_external_context.sum()/table.real_blocks.sum()),
        'block_weighted_one_prelocal_layer_cross_plane_context_coverage':float(table.blocks_with_cross_plane_context.sum()/table.real_blocks.sum()),
        'propagation_scope':'one raw-node relation layer followed by complete block pooling; graph weak components describe potential repeated propagation and do not claim this one-layer receptive field',
        'recorded_first_contributor_recall':{'eligible_by_visible_pixel_gap':opportunities[policy].tolist(),
            'found_by_visible_pixel_gap':success[policy].tolist(),
            'recall':np.divide(success[policy],opportunities[policy],out=np.zeros(3),where=opportunities[policy]>0).tolist(),
            'definition':'source node x target visible pixel plane opportunity with at least one same recorded positive trackID; succeeds with any candidate link to that recorded track',
            'limitation':'PixelSD keeps first contributor; no complete per-channel ancestry or physical track efficiency is claimed'},
        'duplicate_recorded_pixel_channels':int(table.duplicate_recorded_channels.sum())}
    summary={'status':'complete','variants':variants,
        'scope':'fixed balanced Train-only sample; observation-only graph, truth read subsequently for audit',
        'config':config,'Cal_A_Cut_B_Confirm_payload_read':False}
    write_json(output/'results.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
