"""Single-factor local-retention and measurement-relation adapters."""

import torch
from torch import nn

from analysis.flavour.v2.r1.models import build_model as original_model
from analysis.flavour.v2.r1.models import branch_groups as original_groups


RELATION_FEATURES = [
    'signed_dx_over_scale','signed_dy_over_scale','signed_dz_over100mm',
    'signed_slope_x','signed_slope_y','common_measured_x','common_measured_y',
    'source_width_x_over420','source_width_y_over420','target_width_x_over420','target_width_y_over420',
    'log_pair_sigma_x','log_pair_sigma_y','same_plane','source_pixel','target_pixel',
]


def relation_features(source, target, pixel_width_xy_mm):
    delta=(target[:,:3]-source[:,:3])*source.new_tensor([210.,210.,3000.])
    common=source[:,3:5]*target[:,3:5]
    same=delta[:,2:3].abs()<.001
    scale=2.+.5*delta[:,2:3].abs()
    displacement=torch.tanh(delta[:,:2]*common/scale)
    slopes=torch.where(same,torch.zeros_like(delta[:,:2]),
        delta[:,:2]*common/torch.where(same,torch.ones_like(delta[:,2:3]),delta[:,2:3]))
    widths=[]
    for node in [source,target]:
        widths.append(torch.where(node[:,12:13]>.5,node.new_tensor(pixel_width_xy_mm),node[:,8:10]*420.))
    sigma=torch.sqrt((widths[0].square()+widths[1].square())/12.)
    return torch.cat([displacement,torch.tanh(delta[:,2:3]/100.),torch.tanh(slopes/.5),common,
        widths[0]/420.,widths[1]/420.,torch.log1p(sigma)/5.,same.float(),source[:,12:13],target[:,12:13]],dim=-1)


class PrelocalRelations(nn.Module):
    def __init__(self, dim, pixel_width_xy_mm):
        super().__init__()
        self.pixel_width_xy_mm=pixel_width_xy_mm
        self.message=nn.Sequential(nn.Linear(36+len(RELATION_FEATURES),dim),nn.LayerNorm(dim),
            nn.SiLU(),nn.Linear(dim,dim))
        self.norm=nn.LayerNorm(dim)
        self.gate=nn.Parameter(torch.tensor(-2.))

    def forward(self, batch):
        raw=batch['nodes'].reshape(-1,18).float()
        source,target=batch['relation_source'],batch['relation_target']
        context=raw.new_zeros((len(raw),self.norm.normalized_shape[0]))
        if len(source):
            a,b=raw[source],raw[target]
            relation=relation_features(a,b,self.pixel_width_xy_mm)
            difference=b-a
            # Finite bar intervals describe geometry, not a point measurement
            # along the unmeasured axis. Relative coordinates require two
            # measured coordinates; absolute source geometry stays available.
            difference=torch.cat([difference[:,:2]*a[:,3:5]*b[:,3:5],difference[:,2:]],dim=-1)
            messages=self.message(torch.cat([a,difference,relation],dim=-1))
            context.index_add_(0,source,messages.float())
            degree=raw.new_zeros(len(raw))
            degree.index_add_(0,source,torch.ones(len(source),device=raw.device))
            context=self.norm(context/degree[:,None].clamp_min(1))
            context=context*(degree>0)[:,None]*self.gate.sigmoid()
        return context.reshape(*batch['nodes'].shape[:2],-1)


class NativeRelationEncoder(nn.Module):
    def __init__(self, encoder, config):
        super().__init__()
        self.encoder=encoder
        enabled=config.get('native_relations',False)
        if enabled:
            pitch=config.get('native_pixel_pitch_mm',0.)
            widths=(pitch,pitch) if pitch else (.0208,.0228)
            # Shared backbone/head initialization and subsequent dropout RNG
            # are identical across the native control and relation variant.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(config['seed']+6721)
                self.relations=PrelocalRelations(config.get('dim',64),widths)
        else:
            self.relations=None

    def forward(self, batch):
        if self.relations is not None:
            batch={**batch,'prelocal_context':self.relations(batch)}
        return self.encoder(batch)


def build_model(config):
    model = original_model(config)
    if config['family'] == 'B1':
        stride = config.get('cnn_first_stride', 4)
        if stride not in [2, 4] or config.get('cnn_variant', 'resnet') != 'resnet':
            raise ValueError('R3 local-retention control requires ResNet first stride2 or4')
        model.backbone.stem[0].stride = (stride, stride)
    elif config['family'] in ['B2','B3'] and config.get('native_relation_adapter',False):
        model.backbone=NativeRelationEncoder(model.backbone,config)
    return model


def branch_groups(model):
    if getattr(model, 'family', None) != 'B1':
        groups={}
        for group,values in original_groups(model).items():
            for name,parameter in values:
                if name.startswith('backbone.relations.'):
                    part=name.split('.')[2]
                    key='relation_'+part
                else:
                    key=group
                groups.setdefault(key,[]).append((name,parameter))
        return groups
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith('backbone.stem.'):
            group = 'image_stem'
        elif name.startswith('backbone.layers.'):
            group = 'image_stage'+str(int(name.split('.')[2])//2)
        elif name.startswith('backbone.'):
            group = 'image_readout'
        else:
            group = 'tracker' if name.startswith('tracker.') else 'head'
        groups.setdefault(group, []).append((name, parameter))
    return groups
