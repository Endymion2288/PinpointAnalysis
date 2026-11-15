"""Observation-only bounded relations before native local-block pooling.

Every channel keeps the original raw/local/global path. Additional relations
span different windows on one plane and the next three observed planes on each
side, optionally counted within each modality with adjacent cross-modal links.
A bar constrains only its measured axis; zero-common-axis pairs are left
to the existing global backbone. No origin, momentum or ancestry enters gates.
"""

import numpy as np
from scipy.spatial import cKDTree


def candidate_edges(nodes, groups, pixel_width_xy_mm=(.0208,.0228), same_k=4,
        plane_policy='all_visible'):
    if plane_policy not in ['all_visible','modality_visible']:
        raise ValueError('unknown observed-plane candidate policy')
    xyz=nodes[:,:3].astype('float64')*np.array([210.,210.,3000.])
    measured=nodes[:,3:5]>.5
    width=nodes[:,8:10].astype('float64')*420.
    width[nodes[:,12]>.5]=pixel_width_xy_mm
    valid=nodes[:,15]>.5
    z=np.round(xyz[:,2],3)
    planes=np.unique(z[valid])
    sources=[];targets=[]
    counts={'same_plane':0,'cross_plane':0,'nodes':int(valid.sum()),
        'nodes_with_any_edge':0,'nodes_with_cross_plane_edge':0,'maximum_out_degree':0}
    # Trees are per observed plane and measurement signature. The two bar
    # signatures never create a point constraint in their unmeasured axis.
    buckets={}
    source_groups={}
    plane_sources={}
    for plane in planes:
        source_plane=np.flatnonzero(valid&(z==plane))
        plane_sources[plane]=source_plane
        for signature in [(True,True),(True,False),(False,True)]:
            ix=source_plane[np.all(measured[source_plane]==signature,axis=1)]
            if len(ix):
                buckets[(plane,signature)]=ix
                source_groups[(plane,signature)]=ix
    degrees=np.zeros(len(nodes),dtype='int32')
    cross_degrees=np.zeros(len(nodes),dtype='int32')
    trees={}
    for pi,plane in enumerate(planes):
        source_plane=plane_sources[plane]
        candidates=[(plane,same_k)]
        if plane_policy=='all_visible':
            candidates.extend((planes[pi+offset],2 if abs(offset)==1 else 1)
                for offset in [-1,1,-2,2,-3,3] if 0<=pi+offset<len(planes))
        else:
            is_pixel=bool(nodes[source_plane[0],12]>.5)
            same=np.unique(z[valid&((nodes[:,12]>.5)==is_pixel)])
            index=int(np.searchsorted(same,plane))
            candidates.extend((same[index+offset],2 if abs(offset)==1 else 1)
                for offset in [-1,1,-2,2,-3,3] if 0<=index+offset<len(same))
            other=np.unique(z[valid&((nodes[:,12]>.5)!=is_pixel)])
            before,after=other[other<plane],other[other>plane]
            if len(before):candidates.append((before[-1],1))
            if len(after):candidates.append((after[0],1))
        for target_plane,wanted in candidates:
            same_plane=target_plane==plane
            proposals=36 if same_plane else 8
            for source_signature in [(True,True),(True,False),(False,True)]:
                source=source_groups.get((plane,source_signature))
                if source is None:continue
                candidate_indices=[];candidate_scores=[]
                for target_signature in [(True,True),(True,False),(False,True)]:
                    target=buckets.get((target_plane,target_signature))
                    common=np.asarray(source_signature)&np.asarray(target_signature)
                    if target is None or not common.any():continue
                    axis=np.flatnonzero(common)
                    tree_key=(target_plane,target_signature,tuple(axis))
                    if tree_key not in trees:trees[tree_key]=cKDTree(xyz[target][:,axis])
                    tree=trees[tree_key]
                    k=min(proposals,len(target))
                    _,neighbor=tree.query(xyz[source][:,axis],k=k,workers=1)
                    neighbor=np.asarray(neighbor).reshape(len(source),k)
                    other=target[neighbor]
                    delta=np.abs(xyz[source,None,:2]-xyz[other,:2])
                    gap=np.maximum(0.,delta-(width[source,None,:]+width[other])/2.)
                    dz=abs(float(target_plane-plane))
                    score=np.linalg.norm(gap*common,axis=-1)
                    scale=8. if same_plane else 2.+.5*dz
                    allowed=(score<=scale)&(other!=source[:,None])
                    if same_plane:
                        allowed &= groups[source,None]!=groups[other]
                    # A common-axis relation must still lie within the
                    # physical finite interval on the unmeasured axis.
                    unmeasured=~common
                    allowed &= np.all((gap<=scale)|~unmeasured,axis=-1)
                    score=np.where(allowed,score/scale,np.inf)
                    candidate_indices.append(other);candidate_scores.append(score)
                if not candidate_indices:continue
                candidates=np.concatenate(candidate_indices,axis=1)
                scores=np.concatenate(candidate_scores,axis=1)
                order=np.argsort(scores,axis=1,kind='stable')[:,:wanted]
                selected=np.take_along_axis(candidates,order,axis=1)
                finite=np.isfinite(np.take_along_axis(scores,order,axis=1))
                center=np.broadcast_to(source[:,None],selected.shape)[finite]
                other=selected[finite]
                sources.extend(center.tolist());targets.extend(other.tolist())
                np.add.at(degrees,center,1)
                if not same_plane:
                    np.add.at(cross_degrees,center,1);counts['cross_plane']+=len(center)
                else:counts['same_plane']+=len(center)
    edges=np.asarray([sources,targets],dtype='int32').reshape(2,-1)
    counts.update(nodes_with_any_edge=int((degrees>0).sum()),
        nodes_with_cross_plane_edge=int((cross_degrees>0).sum()),
        maximum_out_degree=int(degrees.max(initial=0)))
    if counts['maximum_out_degree']>same_k+8+(2 if plane_policy=='modality_visible' else 0):
        raise ValueError('candidate graph exceeded bounded degree')
    return edges,counts
