"""Scratch architecture baselines with the same observation whitelist."""

import torch
from torch import nn

from analysis.flavour.s2.models import PointClassifier, masked_pool
from analysis.flavour.s4.train import build_model as plane_model, validate_pretraining
from analysis.flavour.v2.r1.tasks import output_size


class TrackerSet(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(5, dim), nn.LayerNorm(dim), nn.SiLU(),
                                   nn.Linear(dim, dim), nn.SiLU())
        self.norm = nn.LayerNorm(2*dim)

    def forward(self, batch):
        x = self.input(batch['tracker'])
        active = batch['tracker_mask'].any(1)
        return self.norm(masked_pool(x, batch['tracker_mask']))*active[:, None]


class ResidualBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.GroupNorm(8, cout),
            nn.SiLU(), nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.GroupNorm(8, cout))
        self.skip = nn.Identity() if cin == cout and stride == 1 else nn.Conv2d(cin, cout, 1, stride, bias=False)

    def forward(self, x):
        return torch.nn.functional.silu(self.body(x)+self.skip(x))


class ProjectionEncoder(nn.Module):
    def __init__(self, image, width=16, variant='resnet'):
        super().__init__()
        self.image = image
        if variant == 'resnet':
            self.stem = nn.Sequential(nn.Conv2d(4, width, 7, 4, 3, bias=False), nn.GroupNorm(8, width), nn.SiLU())
            layers = []
            cin = width
            for stage in range(4):
                cout = width*2**stage
                layers.extend([ResidualBlock(cin, cout, 1 if stage == 0 else 2), ResidualBlock(cout, cout)])
                cin = cout
        elif variant == 'base':
            self.stem = nn.Identity()
            layers = []
            cin = 4
            for cout in [24, 48, 72, 96]:
                layers.extend([nn.Conv2d(cin, cout, 3, 2, 1, bias=False),
                               nn.GroupNorm(8, cout), nn.SiLU()])
                cin = cout
        else:
            raise ValueError(f'unknown projection CNN variant: {variant}')
        self.layers = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(4*cin+8)
        self.output_dim = 4*cin+8

    def forward(self, batch):
        b = len(batch['label'])
        h, w = self.image['height'], self.image['width']
        dense = batch['projection_values'].new_zeros(b*8*h*w)
        dense.index_add_(0, batch['projection_event']*(8*h*w)+batch['projection_indices'], batch['projection_values'])
        dense = dense.reshape(b*2, 4, h, w)
        active = dense.flatten(1).any(1)
        totals = torch.log1p(dense.sum((-1, -2))).reshape(b, 8)
        dense = torch.log1p(dense)
        x = self.layers(self.stem(dense))
        pooled = torch.cat([x.mean((-1, -2)), x.amax((-1, -2))], -1)
        # An absent projection carries no convolutional information. Mask it
        # before fusion: repeated normalization of an exactly zero image can
        # otherwise amplify its backward derivative despite finite activations.
        pooled = (pooled*active[:, None]).reshape(b, -1)
        return self.norm(torch.cat([pooled.float(), totals], -1))


class LateFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.family = config['family']
        if self.family == 'B1':
            self.backbone = ProjectionEncoder(config['image'], config.get('cnn_width', 16),
                                              config.get('cnn_variant', 'resnet'))
            size = self.backbone.output_dim
        else:
            dim = config.get('dim', 64)
            self.backbone = PointClassifier('particlenet' if self.family == 'B2' else 'part', dim=dim, dynamic=False)
            self.backbone.head[-1] = nn.Identity()
            self.backbone.local_chunk_blocks = config.get('local_chunk_blocks', 256)
            if config.get('compile_local'):
                self.backbone.compiled_local = torch.compile(self.backbone.encode_local, dynamic=True)
            size = 2*dim
        self.tracker = TrackerSet()
        self.head = nn.Sequential(nn.LayerNorm(size+128+3), nn.Linear(size+128+3, 128),
                                  nn.SiLU(), nn.Dropout(.1), nn.Linear(128, output_size(config['task'])))

    def forward(self, batch):
        features = torch.cat([self.backbone(batch), self.tracker(batch), batch['presence']], -1)
        with torch.autocast(device_type=features.device.type, enabled=False):
            return self.head(features.float())


def build_model(config):
    if config['family'] in ['B1', 'B2', 'B3']:
        return LateFusion(config)
    if config['family'] in ['B4', 'B4-MAE']:
        validate_pretraining(config)
        model = plane_model(config)
        size = output_size(config['task'])
        if size != 4:
            model.head[-1] = nn.Linear(model.head[-1].in_features, size)
            model.linear_readout = nn.Linear(model.linear_readout.in_features, size)
        return model
    raise ValueError(config['family'])


def branch_groups(model):
    """Parameters with separately visible gradients and actual optimizer updates."""
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith('tracker.'):
            branch = 'tracker'
        elif name.startswith('backbone.'):
            branch = 'image' if getattr(model, 'family', None) == 'B1' else 'native'
            if '.local.' in name or '.input.' in name:
                branch += '_local'
            elif '.global_blocks.' in name:
                branch += '_global'
        elif name.startswith(('input.', 'local.', 'local_pool.')):
            branch = 'local'
        elif name.startswith(('module_', 'latent_', 'global_', 'latents', 'position.', 'geometry_')):
            branch = 'regional_global'
        else:
            branch = 'head'
        groups.setdefault(branch, []).append((name, parameter))
    return groups
