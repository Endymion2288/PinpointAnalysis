"""Global reconstruction and EMA objectives on the unchanged S3v2 encoder."""
import copy

import torch
from torch import nn
from torch.nn import functional as F

from analysis.flavour.s2.models import masked_pool
from analysis.flavour.s3.models_repair import PlaneModuleClassifierV2, StableAttention


class QueryDecoder(nn.Module):
    def __init__(self,dim=96):
        super().__init__()
        self.position=nn.Sequential(nn.Linear(22+12,dim),nn.LayerNorm(dim),nn.SiLU())
        self.read=StableAttention(dim)
        self.norm=nn.LayerNorm(dim)
        self.output=nn.Sequential(nn.Linear(dim,dim),nn.LayerNorm(dim),nn.SiLU(),nn.Linear(dim,8))
        self.apply(PlaneModuleClassifierV2.initialize)

    def forward(self,queries,latents,mask=None):
        xyz=queries[...,:3]
        fourier=torch.cat([torch.sin(2*torch.pi*xyz),torch.cos(2*torch.pi*xyz),
            torch.sin(8*torch.pi*xyz),torch.cos(8*torch.pi*xyz)],-1)
        q=self.position(torch.cat([queries,fourier],-1))
        if mask is None:mask=torch.ones(latents.shape[:2],device=latents.device,dtype=torch.bool)
        x=self.norm(self.read(q,latents,mask))
        with torch.autocast(device_type=queries.device.type,enabled=False):
            return self.output(x.float())


class Projection(nn.Module):
    def __init__(self,input_dim,dim=96):
        super().__init__()
        self.network=nn.Sequential(nn.LayerNorm(input_dim),nn.Linear(input_dim,2*dim),nn.SiLU(),nn.Linear(2*dim,dim))
        self.apply(PlaneModuleClassifierV2.initialize)

    def forward(self,x):
        return self.network(x.float())


class PretrainingModel(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.encoder=PlaneModuleClassifierV2(config)
        self.decoder=QueryDecoder(config.get('dim',96))
        dim=config.get('dim',96)
        self.projector=Projection(2*dim,dim)
        self.predictor=Projection(dim,dim)
        # Freeze unused classification parameters without changing the encoder
        # state dictionary or the downstream architecture.
        for name,parameter in self.encoder.named_parameters():
            if name.startswith(('head.','linear_readout.','head_input_norm.','summary.','branch_norms.')):
                parameter.requires_grad_(False)
        self.teacher=None
        self.teacher_projector=None
        if config.get('objective')=='distill':
            # A compiled callable is execution state; teacher runs its own eager
            # local encoder. No truth or downstream labels enter teacher inputs.
            teacher_config={**config,'compile_local':False}
            self.teacher=PlaneModuleClassifierV2(teacher_config)
            self.teacher.load_state_dict(self.encoder.state_dict())
            self.teacher_projector=copy.deepcopy(self.projector)
            self.teacher.requires_grad_(False)
            self.teacher_projector.requires_grad_(False)

    def train(self,mode=True):
        super().train(mode)
        if self.teacher is not None:
            self.teacher.eval();self.teacher_projector.eval()
        return self

    @staticmethod
    def pool_global(features):
        x=features['global_tokens']
        return torch.cat([x.mean(1),x.amax(1)],-1).float()

    @staticmethod
    def pool_teacher_multiscale(features):
        branches=features['branches'].float()
        return (branches[:,:3].mean(1)+branches[:,3]+branches[:,4:].mean(1))/3

    def forward(self,batch):
        features=self.encoder.encode(batch['student'])
        result={'features':features,'reconstruction':self.decoder(batch['query'],features['global_tokens'])}
        if 'rel_query' in batch:
            source=features['global_tokens'];valid=None
            if self.config.get('rel_source')=='global_and_bar_local':
                # Kept bar semantics need their retained detector information.
                # MAE reconstruction still reads ONLY global tokens. Rel can
                # also read every visible bar local block, without truth-based
                # routing, truncation, or changing the downstream classifier.
                selected=batch['student']['block_detector']==1
                local=features['local_tokens'][selected]
                event=batch['student']['event_index'][selected]
                counts=torch.bincount(event,minlength=len(source))
                width=int(counts.max())
                packed=source.new_zeros((len(source),width,source.shape[-1]))
                mask=torch.zeros((len(source),width),device=source.device,dtype=torch.bool)
                slot=torch.arange(len(event),device=source.device)-(counts.cumsum(0)-counts)[event]
                packed[event,slot]=local.to(source.dtype);mask[event,slot]=True
                valid=torch.cat([torch.ones(source.shape[:2],device=source.device,dtype=torch.bool),mask],1)
                source=torch.cat([source,packed],1)
            result['rel']=self.decoder(batch['rel_query'],source,valid)[...,4:8]
        if self.teacher is not None:
            with torch.autocast(device_type=batch['query'].device.type,enabled=False):
                projected=self.projector(self.pool_global(features))
                result['student_projection']=projected
                result['student_prediction']=self.predictor(projected)
            with torch.no_grad():
                teacher_features=self.teacher.encode(batch['teacher'])
                with torch.autocast(device_type=batch['query'].device.type,enabled=False):
                    teacher_input=(self.pool_teacher_multiscale(teacher_features)
                        if self.config.get('teacher_multiscale',False) else self.pool_global(teacher_features))
                    result['teacher_projection']=self.teacher_projector(teacher_input)
        return result

    @torch.no_grad()
    def update_teacher(self,momentum):
        if self.teacher is None:return
        for teacher,student in [(self.teacher,self.encoder),(self.teacher_projector,self.projector)]:
            for target,source in zip(teacher.parameters(),student.parameters()):
                target.mul_(momentum).add_(source,alpha=1-momentum)
            for target,source in zip(teacher.buffers(),student.buffers()):
                target.copy_(source)


def variance_covariance(x,gamma=.2):
    # Small batches are explicit in the protocol; all statistics are over events,
    # never over queries/regions as if those were independent physical samples.
    centered=x-x.mean(0,keepdim=True)
    variance=F.relu(gamma-torch.sqrt(centered.square().mean(0)+1e-4)).mean()
    covariance=centered.T@centered/max(1,len(x)-1)
    off=covariance-torch.diag_embed(torch.diagonal(covariance))
    return variance,off.square().sum()/x.shape[-1]


def objective_loss(output,batch,config):
    predicted=output['reconstruction'].float()
    target=batch['reconstruction'].float()
    levels=batch['query_level']
    parts={}
    losses=[]
    for level in range(3):
        selected=levels==level
        truth=target[selected]
        estimate=predicted[selected]
        # Fixed positive weight is part of the declared mixture objective,
        # rather than a hidden-support based resampling of query positions.
        occupancy=F.binary_cross_entropy_with_logits(estimate[:,0],truth[:,0],
            pos_weight=truth.new_tensor(config.get('occupancy_positive_weight',3.)))
        positive=truth[:,0]>0
        count=F.smooth_l1_loss(estimate[positive,1],truth[positive,1]) if positive.any() else estimate[:,1].sum()*0
        signal=(F.smooth_l1_loss(estimate[positive,2:4],truth[positive,2:4])
            if level==2 and positive.any() else estimate[:,2:4].sum()*0)
        moments=(F.smooth_l1_loss(estimate[positive,2:4],truth[positive,2:4])
            if level==0 and config.get('pixel_query_patch',1)>1 and positive.any() else estimate[:,2:4].sum()*0)
        losses.append(occupancy+count+signal+moments)
        parts[f'level{level}_occupancy']=occupancy
        parts[f'level{level}_count']=count
        parts[f'level{level}_signal']=signal
        parts[f'level{level}_moments']=moments
    reconstruction=torch.stack(losses).mean()
    parts['reconstruction']=reconstruction
    loss=reconstruction
    if config.get('objective')=='mae_rel':
        mask=batch['rel_valid']
        logits=output['rel'].float()
        rel=(-(batch['rel_target'][mask]*logits[mask].log_softmax(-1)).sum(-1).mean()
            if mask.any() else logits.sum()*0)
        parts['bar_rel']=rel
        loss=loss+config.get('rel_weight',.25)*rel
    if config.get('objective')=='distill':
        student=output['student_prediction'].float()
        teacher=output['teacher_projection'].detach().float()
        align=(2-2*(F.normalize(student,dim=-1)*F.normalize(teacher,dim=-1)).sum(-1)).mean()
        variance,covariance=variance_covariance(output['student_projection'],config.get('variance_gamma',.2))
        # Joint reconstruction anchors the teacher/student to measured content;
        # separate variance regularization opposes the constant-vector solution.
        parts.update(distillation=align,variance=variance,covariance=covariance)
        loss=(config.get('distill_reconstruction_weight',.5)*reconstruction+align
            +config.get('variance_weight',5.)*variance+config.get('covariance_weight',.1)*covariance)
    parts['loss']=loss
    return loss,parts
