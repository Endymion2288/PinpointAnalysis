"""Complete-coverage ParticleNet and geometry-adapted ParT S2 baselines.

Both use identical deterministic local blocks (<=32 measured nodes), learned
local encoding, and every pooled block. No per-event token cap. Local pooling
is an explicit shared representation compression, not raw global attention.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def masked_pool(x,mask):
    mean=(x*mask[...,None]).sum(-2)/mask.sum(-1,keepdim=True).clamp_min(1)
    maximum=x.masked_fill(~mask[...,None],-16384.).amax(-2)
    maximum=torch.where(mask.any(-1,keepdim=True),maximum,torch.zeros_like(maximum))
    return torch.cat([mean,maximum],dim=-1)


def response_totals(nodes,mask):
    """Additive information from the CURRENT representation, never raw truth.

    Count projections contain each native pixel in both marginals. Half of the
    sum of measured x/y multiplicities recovers the same total as joint input.
    For OR this deliberately counts occupied represented cells, not the lost
    native multiplicity. The other entries preserve bar/track counts, bar
    signal sums, token count and the two pixel marginal counts.
    """
    values=nodes.float()
    valid=mask*values[...,15]
    pixel=valid*values[...,12]
    bar=valid*values[...,13]
    tracker=valid*values[...,14]
    count=torch.expm1(values[...,5])
    px=(pixel*count*values[...,3]).sum(-1)
    py=(pixel*count*values[...,4]).sum(-1)
    return torch.stack([.5*(px+py),bar.sum(-1),tracker.sum(-1),
        (bar*torch.expm1(values[...,6]*10)).sum(-1),
        (bar*torch.expm1(values[...,7]*15)).sum(-1),valid.sum(-1),px,py],dim=-1)


def scale_totals(totals):
    scales=totals.new_tensor([.1,.1,.1,.1,1/15,.1,.1,.1])
    return torch.log1p(totals)*scales


def physical_distance(geometry):
    xyz=geometry[...,:3]
    delta=xyz[:,:,None,:]-xyz[:,None,:,:]
    common=geometry[:,:,None,3:5]*geometry[:,None,:,3:5]
    # Interval widths are used for bar compatibility; unseen coordinates carry
    # no point-distance constraint. Longitudinal scale prevents same-plane-only
    # neighborhoods from exhausting cross-plane context.
    transverse=(delta[...,:2].square()*common).sum(-1)
    return transverse+delta[...,2].square()*.05


class EdgeBlock(nn.Module):
    def __init__(self,in_dim,out_dim,k=16,same_plane_only=False):
        super().__init__()
        self.k=k
        self.same_plane_only=same_plane_only
        self.edge=nn.Sequential(nn.Linear(in_dim*2,out_dim),nn.LayerNorm(out_dim),nn.GELU(),
                                nn.Linear(out_dim,out_dim),nn.GELU())
        self.skip=nn.Linear(in_dim,out_dim)

    def forward(self,x,mask,distance=None,geometry=None,plane_z=None):
        number=min(self.k,max(1,x.shape[1]-1))
        all_neighbors,all_valid=[],[]
        for start in range(0,x.shape[1],256):
            stop=min(start+256,x.shape[1])
            with torch.no_grad():
                if distance is not None:
                    d=distance[:,start:stop,:].float()
                elif geometry is not None:
                    delta=geometry[:,start:stop,None,:3]-geometry[:,None,:,:3]
                    common=geometry[:,start:stop,None,3:5]*geometry[:,None,:,3:5]
                    d=(delta[...,:2].square()*common).sum(-1)+delta[...,2].square()*.05
                else:
                    d=torch.cdist(x[:,start:stop].float(),x.float()).square()
                d=d.masked_fill(~mask[:,None,:],float('inf'))
                if self.same_plane_only:
                    z=geometry[...,2] if geometry is not None else plane_z
                    same=(z[:,start:stop,None]-z[:,None,:]).abs()<1e-6
                    d=d.masked_fill(~same,float('inf'))
                self_edge=torch.arange(start,stop,device=x.device)[:,None]==torch.arange(x.shape[1],device=x.device)[None,:]
                d=d.masked_fill(self_edge[None],float('inf'))
                if geometry is not None and number>=2 and not self.same_plane_only:
                    same=(geometry[:,start:stop,None,2]-geometry[:,None,:,2]).abs()<1e-6
                    a=number//2
                    local_distance=d.masked_fill(~same,float('inf'))
                    cross_distance=d.masked_fill(same,float('inf'))
                    local=local_distance.topk(a,dim=-1,largest=False).indices
                    cross=cross_distance.topk(number-a,dim=-1,largest=False).indices
                    neighbors=torch.cat([local,cross],dim=-1)
                    finite=torch.cat([local_distance.gather(-1,local),cross_distance.gather(-1,cross)],dim=-1).isfinite()
                else:
                    neighbors=d.topk(number,dim=-1,largest=False).indices
                    finite=d.gather(-1,neighbors).isfinite()
                all_neighbors.append(neighbors)
                all_valid.append(finite)
        neighbors=torch.cat(all_neighbors,dim=1)
        neighbor_finite=torch.cat(all_valid,dim=1)
        batch=torch.arange(len(x),device=x.device)[:,None,None]
        other=x[batch,neighbors]
        center=x[:,:,None,:].expand_as(other)
        edges=self.edge(torch.cat([center,other-center],dim=-1))
        valid=mask[batch,neighbors]&mask[:,:,None]&neighbor_finite
        # A power-of-two sentinel is exactly representable in BF16. A compiled
        # fused max backward must compare against the same rounded value.
        edges=edges.masked_fill(~valid[...,None],-16384.).amax(-2)
        edges=torch.where(valid.any(-1,keepdim=True),edges,torch.zeros_like(edges))
        return (edges+self.skip(x))*mask[...,None]


class PairAttention(nn.Module):
    def __init__(self,dim,heads=4,pair_bias=True,cross_plane=True):
        super().__init__()
        self.heads=heads
        self.cross_plane=cross_plane
        self.norm1,self.norm2=nn.LayerNorm(dim),nn.LayerNorm(dim)
        self.qkv=nn.Linear(dim,3*dim)
        self.out=nn.Linear(dim,dim)
        self.ff=nn.Sequential(nn.Linear(dim,dim*2),nn.GELU(),nn.Linear(dim*2,dim))
        self.pair=nn.Sequential(nn.Linear(8,32),nn.GELU(),nn.Linear(32,heads)) if pair_bias else None

    def attend_tile(self,q,k,v,query_geometry,key_geometry,mask):
        if self.pair is not None:
            delta=query_geometry[:,:,None,:3]-key_geometry[:,None,:,:3]
            common=query_geometry[:,:,None,3:5]*key_geometry[:,None,:,3:5]
            delta=torch.cat([delta[...,:2]*common,delta[...,2:]],dim=-1)
            same_type=(query_geometry[:,:,None,12:15]*key_geometry[:,None,:,12:15]).sum(-1,keepdim=True)
            same_plane=(delta[...,2:].abs()<1e-6).to(delta.dtype)
            transverse=torch.log1p((delta[...,:2].square()*common).sum(-1,keepdim=True)*1e4)
            pair=torch.cat([delta,common,same_type,same_plane,transverse],dim=-1)
            bias=self.pair(pair).permute(0,3,1,2)
            bias=bias.masked_fill(~mask[:,None,None,:],float('-inf'))
        else:
            bias=torch.zeros((q.shape[0],1,1,k.shape[2]),device=q.device,dtype=q.dtype).masked_fill(~mask[:,None,None,:],float('-inf'))
        if not self.cross_plane:
            same=(query_geometry[:,:,None,2]-key_geometry[:,None,:,2]).abs()<1e-6
            bias=bias.masked_fill(~same[:,None],float('-inf'))
        return F.scaled_dot_product_attention(q,k,v,attn_mask=bias)

    def forward(self,x,mask,geometry):
        batch,n,dim=x.shape
        q,k,v=self.qkv(self.norm1(x)).reshape(batch,n,3,self.heads,dim//self.heads).permute(2,0,3,1,4)
        pieces=[]
        for start in range(0,n,256):
            values=(q[:,:,start:start+256],k,v,geometry[:,start:start+256],geometry,mask)
            if self.training and torch.is_grad_enabled() and n>256:
                pieces.append(checkpoint(self.attend_tile,*values,use_reentrant=False))
            else:
                pieces.append(self.attend_tile(*values))
        attention=torch.cat(pieces,dim=2)
        x=x+self.out(attention.transpose(1,2).reshape(batch,n,dim))
        return (x+self.ff(self.norm2(x)))*mask[...,None]


class PointClassifier(nn.Module):
    def __init__(self,architecture='part',dim=64,k=16,pair_bias=True,dynamic=True,cross_plane=True):
        super().__init__()
        self.architecture,self.dynamic=architecture,dynamic
        self.cross_plane=cross_plane
        self.input=nn.Sequential(nn.Linear(18,dim),nn.LayerNorm(dim),nn.GELU())
        if architecture=='part':
            self.local=nn.ModuleList([PairAttention(dim,pair_bias=pair_bias) for _ in range(2)])
            self.global_blocks=nn.ModuleList([PairAttention(dim,pair_bias=pair_bias,cross_plane=cross_plane) for _ in range(4)])
        elif architecture=='particlenet':
            self.local=nn.ModuleList([EdgeBlock(dim,dim,k) for _ in range(3)])
            self.global_blocks=nn.ModuleList([EdgeBlock(dim,dim,k,same_plane_only=not cross_plane) for _ in range(3)])
        else:
            raise ValueError(architecture)
        self.pool=nn.Linear(dim*2+8,dim)
        # Separate detector pooled branches followed by late fusion for both
        # families; geometry/response masks are identical in every ablation.
        self.head=nn.Sequential(nn.Linear(dim*6+8,dim*2),nn.GELU(),nn.Dropout(.1),nn.Linear(dim*2,4))

    def encode_local(self,nodes,local_mask):
        x=self.input(nodes)
        distance=physical_distance(nodes) if self.architecture=='particlenet' else None
        for i,block in enumerate(self.local):
            if self.architecture=='part':
                x=block(x,local_mask,nodes)
            else:
                x=block(x,local_mask,distance if i==0 or not self.dynamic else None)
        summary=scale_totals(response_totals(nodes,local_mask)).to(x.dtype)
        return self.pool(torch.cat([masked_pool(x,local_mask),summary],dim=-1))

    def forward(self,batch):
        nodes,local_mask=batch['nodes'],batch['local_mask']
        parts=[]
        chunk=getattr(self,'local_chunk_blocks',256)
        compiled=getattr(self,'compiled_local',None)
        encode=compiled if compiled is not None and nodes.is_cuda else self.encode_local
        for start in range(0,len(nodes),chunk):
            if self.training and torch.is_grad_enabled():
                encoded=checkpoint(encode,nodes[start:start+chunk],local_mask[start:start+chunk],use_reentrant=False)
            else:
                encoded=encode(nodes[start:start+chunk],local_mask[start:start+chunk])
            parts.append(encoded)
        local=torch.cat(parts)
        # Scatter complete block summaries into the event dimension.
        event_index,block_index=batch['event_index'],batch['block_index']
        totals=response_totals(nodes,local_mask)
        event_totals=totals.new_zeros((batch['event_mask'].shape[0],8))
        event_totals.index_add_(0,event_index,totals)
        shape=(batch['event_mask'].shape[0],batch['event_mask'].shape[1],local.shape[-1])
        event=local.new_zeros(shape)
        event[event_index,block_index]=local
        geometry=nodes.new_zeros((*shape[:2],18))
        mean=(nodes*local_mask[...,None]).sum(1)/local_mask.sum(1,keepdim=True).clamp_min(1)
        geometry[event_index,block_index]=mean
        mask=batch['event_mask']
        # Global pair relations are generated in query tiles to bound memory;
        # every query still sees every real block or the complete KNN candidate set.
        distance=None
        for i,block in enumerate(self.global_blocks):
            if self.architecture=='part':
                if self.training and torch.is_grad_enabled():
                    event=checkpoint(block,event,mask,geometry,use_reentrant=False)
                else:
                    event=block(event,mask,geometry)
            else:
                # First global graph includes both within-plane and across-plane
                # edges; later graphs use learned feature distances.
                event=block(event,mask,distance if i==0 or not self.dynamic else None,
                            geometry if i==0 or not self.dynamic else None,
                            plane_z=geometry[:,:,2] if not self.cross_plane else None)
        pools=[masked_pool(event,mask&(geometry[:,:,12+d]>.5)) for d in range(3)]
        return self.head(torch.cat(pools+[scale_totals(event_totals).to(event.dtype)],dim=-1))
