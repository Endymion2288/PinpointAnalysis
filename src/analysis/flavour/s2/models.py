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
    maximum=x.masked_fill(~mask[...,None],-1e4).amax(-2)
    maximum=torch.where(mask.any(-1,keepdim=True),maximum,torch.zeros_like(maximum))
    return torch.cat([mean,maximum],dim=-1)


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
    def __init__(self,in_dim,out_dim,k=16):
        super().__init__()
        self.k=k
        self.edge=nn.Sequential(nn.Linear(in_dim*2,out_dim),nn.LayerNorm(out_dim),nn.GELU(),
                                nn.Linear(out_dim,out_dim),nn.GELU())
        self.skip=nn.Linear(in_dim,out_dim)

    def forward(self,x,mask,distance=None,geometry=None):
        distance=torch.cdist(x.float(),x.float()).square() if distance is None else distance.float()
        distance=distance.masked_fill(~mask[:,None,:],float('inf'))
        eye=torch.eye(x.shape[1],device=x.device,dtype=torch.bool)[None]
        distance=distance.masked_fill(eye,float('inf'))
        number=min(self.k,max(1,x.shape[1]-1))
        if geometry is not None and number>=2:
            # Explicit quotas prevent the same-plane density from excluding
            # cross-plane context. Longitudinal nearest neighbors prioritize
            # adjacent instrumented planes; physical transverse distances break
            # ties. Mask unavailable neighbors rather than inventing nodes.
            same=(geometry[:,:,None,2]-geometry[:,None,:,2]).abs()<1e-6
            a=number//2
            local=distance.masked_fill(~same,float('inf')).topk(a,dim=-1,largest=False).indices
            cross=distance.masked_fill(same,float('inf')).topk(number-a,dim=-1,largest=False).indices
            neighbors=torch.cat([local,cross],dim=-1)
            neighbor_finite=torch.cat([distance.masked_fill(~same,float('inf')).gather(-1,local),
                                       distance.masked_fill(same,float('inf')).gather(-1,cross)],dim=-1).isfinite()
        else:
            neighbors=distance.topk(number,dim=-1,largest=False).indices
            neighbor_finite=distance.gather(-1,neighbors).isfinite()
        batch=torch.arange(len(x),device=x.device)[:,None,None]
        other=x[batch,neighbors]
        center=x[:,:,None,:].expand_as(other)
        edges=self.edge(torch.cat([center,other-center],dim=-1))
        valid=mask[batch,neighbors]&mask[:,:,None]&neighbor_finite
        edges=edges.masked_fill(~valid[...,None],-1e4).amax(-2)
        edges=torch.where(valid.any(-1,keepdim=True),edges,torch.zeros_like(edges))
        return (edges+self.skip(x))*mask[...,None]


class PairAttention(nn.Module):
    def __init__(self,dim,heads=4,pair_bias=True):
        super().__init__()
        self.heads=heads
        self.norm1,self.norm2=nn.LayerNorm(dim),nn.LayerNorm(dim)
        self.qkv=nn.Linear(dim,3*dim)
        self.out=nn.Linear(dim,dim)
        self.ff=nn.Sequential(nn.Linear(dim,dim*2),nn.GELU(),nn.Linear(dim*2,dim))
        self.pair=nn.Sequential(nn.Linear(8,32),nn.GELU(),nn.Linear(32,heads)) if pair_bias else None

    def forward(self,x,mask,geometry):
        batch,n,dim=x.shape
        q,k,v=self.qkv(self.norm1(x)).reshape(batch,n,3,self.heads,dim//self.heads).permute(2,0,3,1,4)
        if self.pair is not None:
            delta=geometry[:,:,None,:3]-geometry[:,None,:,:3]
            common=geometry[:,:,None,3:5]*geometry[:,None,:,3:5]
            delta=torch.cat([delta[...,:2]*common,delta[...,2:]],dim=-1)
            same_type=(geometry[:,:,None,12:15]*geometry[:,None,:,12:15]).sum(-1,keepdim=True)
            same_plane=(delta[...,2:].abs()<1e-6).to(delta.dtype)
            transverse=torch.log1p((delta[...,:2].square()*common).sum(-1,keepdim=True)*1e4)
            pair=torch.cat([delta,common,same_type,same_plane,transverse],dim=-1)
            bias=self.pair(pair).permute(0,3,1,2)
            bias=bias.masked_fill(~mask[:,None,None,:],float('-inf'))
        else:
            bias=torch.zeros((batch,1,1,n),device=x.device,dtype=x.dtype).masked_fill(~mask[:,None,None,:],float('-inf'))
        attention=F.scaled_dot_product_attention(q,k,v,attn_mask=bias)
        x=x+self.out(attention.transpose(1,2).reshape(batch,n,dim))
        return (x+self.ff(self.norm2(x)))*mask[...,None]


class PointClassifier(nn.Module):
    def __init__(self,architecture='part',dim=64,k=16,pair_bias=True,dynamic=True):
        super().__init__()
        self.architecture,self.dynamic=architecture,dynamic
        self.input=nn.Sequential(nn.Linear(18,dim),nn.LayerNorm(dim),nn.GELU())
        if architecture=='part':
            self.local=nn.ModuleList([PairAttention(dim,pair_bias=pair_bias) for _ in range(2)])
            self.global_blocks=nn.ModuleList([PairAttention(dim,pair_bias=pair_bias) for _ in range(4)])
        elif architecture=='particlenet':
            self.local=nn.ModuleList([EdgeBlock(dim,dim,k) for _ in range(3)])
            self.global_blocks=nn.ModuleList([EdgeBlock(dim,dim,k) for _ in range(3)])
        else:
            raise ValueError(architecture)
        self.pool=nn.Linear(dim*2,dim)
        # Separate detector pooled branches followed by late fusion for both
        # families; geometry/response masks are identical in every ablation.
        self.head=nn.Sequential(nn.Linear(dim*6,dim*2),nn.GELU(),nn.Dropout(.1),nn.Linear(dim*2,4))

    def encode_local(self,nodes,local_mask):
        x=self.input(nodes)
        distance=physical_distance(nodes) if self.architecture=='particlenet' else None
        for i,block in enumerate(self.local):
            if self.architecture=='part':
                x=block(x,local_mask,nodes)
            else:
                x=block(x,local_mask,distance if i==0 or not self.dynamic else None)
        return self.pool(masked_pool(x,local_mask))

    def forward(self,batch):
        nodes,local_mask=batch['nodes'],batch['local_mask']
        parts=[]
        for start in range(0,len(nodes),256):
            if self.training and torch.is_grad_enabled():
                encoded=checkpoint(self.encode_local,nodes[start:start+256],local_mask[start:start+256],use_reentrant=False)
            else:
                encoded=self.encode_local(nodes[start:start+256],local_mask[start:start+256])
            parts.append(encoded)
        local=torch.cat(parts)
        # Scatter complete block summaries into the event dimension.
        event_index,block_index=batch['event_index'],batch['block_index']
        shape=(batch['event_mask'].shape[0],batch['event_mask'].shape[1],local.shape[-1])
        event=local.new_zeros(shape)
        event[event_index,block_index]=local
        geometry=nodes.new_zeros((*shape[:2],18))
        mean=(nodes*local_mask[...,None]).sum(1)/local_mask.sum(1,keepdim=True).clamp_min(1)
        geometry[event_index,block_index]=mean
        mask=batch['event_mask']
        distance=physical_distance(geometry) if self.architecture=='particlenet' else None
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
                            geometry if i==0 or not self.dynamic else None)
        pools=[masked_pool(event,mask&(geometry[:,:,12+d]>.5)) for d in range(3)]
        return self.head(torch.cat(pools,dim=-1))
