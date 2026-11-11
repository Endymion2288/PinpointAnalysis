"""Loss-controlled pixel representations, retaining all occupied channels.

Inputs derive only from observations. Coarse centers never carry native-cell
residuals. Projected tokens have explicit coordinate validity; no x-y pairing.
"""

import numpy as np


def pixel_view(pixel, geometry, pitch_mm=0., projection=False, reducer='count', origin_shift=0.):
    if reducer not in ['count','or']:
        raise ValueError(reducer)
    xyz=np.asarray(pixel['xyz_mm'],dtype='float64').reshape(-1,3)
    ids=np.asarray(pixel['channel_id'],dtype='int64')
    nx,ny=len(geometry['pixel_Xpos']),len(geometry['pixel_Ypos'])
    native=np.column_stack([ids%nx,(ids//nx)%ny,ids//(nx*ny)])
    lower=np.array([geometry['pixel_Xpos'][0]-.5*np.median(np.diff(geometry['pixel_Xpos'])),
                    geometry['pixel_Ypos'][0]-.5*np.median(np.diff(geometry['pixel_Ypos']))])
    origin=lower+origin_shift*pitch_mm
    cells=(np.floor((xyz[:,:2]-origin)/pitch_mm).astype('int64') if pitch_mm else native[:,:2])
    if not len(ids):
        return {'xyz_mm':np.empty((0,3),dtype='float32'),'measured':np.empty((0,2),dtype='float32'),
                'count':np.empty(0,dtype='float32'),'view':np.empty(0,dtype='int64'),
                'native_to_token':np.empty((0,2 if projection else 1),dtype='int64')}

    def reduce(axis=None):
        key=np.column_stack([native[:,2],cells]) if axis is None else np.column_stack([native[:,2],cells[:,axis]])
        unique,inverse,count=np.unique(key,axis=0,return_inverse=True,return_counts=True)
        out=np.zeros((len(unique),3),dtype='float64')
        out[:,2]=np.asarray(geometry['pixel_Zpos'])[unique[:,0]]
        axes=[0,1] if axis is None else [axis]
        measured=np.zeros((len(unique),2),dtype='float32')
        for column,a in enumerate(axes,1):
            if pitch_mm:
                out[:,a]=origin[a]+(unique[:,column]+.5)*pitch_mm
            else:
                out[:,a]=np.asarray(geometry[['pixel_Xpos','pixel_Ypos'][a]])[unique[:,column]]
            measured[:,a]=1
        return out,measured,(count if reducer=='count' else np.ones(len(count))),inverse

    if not projection:
        positions,measured,count,inverse=reduce()
        view=np.zeros(len(count),dtype='int64')
        mapping=inverse[:,None]
    else:
        a,b=reduce(0),reduce(1)
        positions=np.concatenate([a[0],b[0]])
        measured=np.concatenate([a[1],b[1]])
        count=np.r_[a[2],b[2]]
        view=np.r_[np.ones(len(a[2]),dtype='int64'),np.full(len(b[2]),2,dtype='int64')]
        mapping=np.column_stack([a[3],b[3]+len(a[2])])
    return {'xyz_mm':positions.astype('float32'),'measured':measured,'count':count.astype('float32'),
            'view':view,'native_to_token':mapping}


def token_inputs(observations, geometry, pitch_mm=0., projection=False, reducer='count',
                 modalities=('pixel','bar'), origin_shift=0., shuffle_local=False, seed=0):
    """Return 18-feature nodes and deterministic bounded local neighborhoods.

    Columns: scaled xyz(3), observed x/y(2), log count(1), log edep/PE(2),
    interval widths xy(2), sigma xy(2), detector one-hot(3), response(1), measured local xy residuals in the 4 mm window(2).
    Local blocks group by pixel plane+4mm cell, bar panel, tracker station.
    A block over 32 nodes is split without dropping nodes. The global encoder
    sees every block, so there is no event-level token truncation.
    """
    features,groups,raw_counts,positions=[],[],[],[]
    group_offset=0
    for detector in modalities:
        if detector=='pixel':
            view=pixel_view(observations['pixel'],geometry,pitch_mm,projection,reducer,origin_shift)
            xyz=view['xyz_mm'].astype('float64')
            n=len(xyz)
            feat=np.zeros((n,18),dtype='float32')
            feat[:,3:5]=view['measured']
            feat[:,5]=np.log1p(view['count'])
            feat[:,12]=1
            group_keys=np.column_stack([xyz[:,2],view['view'],np.floor(xyz[:,:2]/4)])
            raw_counts.append(len(observations['pixel']['channel_id']))
        elif detector=='bar':
            bars=observations['bar']
            bounds=np.asarray(bars['bounds_mm'],dtype='float64').reshape(-1,5)
            xyz=np.column_stack([(bounds[:,0]+bounds[:,1])/2,(bounds[:,2]+bounds[:,3])/2,bounds[:,4]])
            n=len(xyz)
            feat=np.zeros((n,18),dtype='float32')
            h=np.asarray(bars['horizontal'])
            feat[:,3],feat[:,4]=1-h,h
            feat[:,6]=np.log1p(bars['edep_proxy_MeV'])/10
            feat[:,7]=np.log1p(bars['pe_proxy'])/15
            feat[:,8]=(bounds[:,1]-bounds[:,0])/420
            feat[:,9]=(bounds[:,3]-bounds[:,2])/420
            feat[:,13]=1
            group_keys=np.asarray(bars['panel'])[:,None]
            raw_counts.append(n)
        elif detector=='tracker':
            tracker=observations['tracker_hypothesis']
            xy=np.asarray(tracker['xy_mm'],dtype='float64').reshape(-1,2)
            station=np.asarray(tracker['station'],dtype='float64')
            # station index is metadata; no ideal crossing z or truth momentum.
            xyz=np.column_stack([xy,station*1000])
            n=len(xyz)
            feat=np.zeros((n,18),dtype='float32')
            feat[:,3:5]=1
            feat[:,10:12]=np.asarray(tracker['sigma_xy_mm']).reshape(-1,2)
            feat[:,14]=1
            group_keys=np.column_stack([station,np.floor(xy/16)])
            raw_counts.append(n)
        else:
            raise ValueError(detector)
        feat[:,:3]=xyz/[210,210,3000]
        feat[:,15]=1
        feat[:,16:18]=((xyz[:,:2]/4-np.floor(xyz[:,:2]/4))-.5)*feat[:,3:5]
        if shuffle_local and n:
            # Observation-only diagnostic: permute x/y among nodes in a plane
            # while retaining each coordinate marginal and per-plane occupancy.
            rng=np.random.default_rng(seed)
            for z in np.unique(xyz[:,2]):
                index=np.flatnonzero(xyz[:,2]==z)
                feat[index,1]=feat[rng.permutation(index),1]
        _,inv=np.unique(group_keys,axis=0,return_inverse=True)
        blocks=[]
        for g in range(inv.max()+1 if len(inv) else 0):
            index=np.flatnonzero(inv==g)
            # Canonical geometric order makes input permutation irrelevant.
            index=index[np.lexsort((feat[index,2],feat[index,1],feat[index,0]))]
            for start in range(0,len(index),32):
                blocks.append(index[start:start+32])
        order=np.concatenate(blocks) if blocks else np.array([],dtype='int64')
        group=np.concatenate([np.full(len(b),group_offset+i,dtype='int64') for i,b in enumerate(blocks)]) if blocks else np.array([],dtype='int64')
        group_offset+=len(blocks)
        features.append(feat[order])
        groups.append(group)
        positions.append(xyz[order])
    if not any(len(f) for f in features):
        # Explicit empty-event token, response bit=0, no invented measurement.
        return np.zeros((1,18),dtype='float32'),np.zeros(1,dtype='int64'),raw_counts
    return np.concatenate(features),np.concatenate(groups),raw_counts
