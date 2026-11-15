"""Single-factor local-retention and measurement-relation adapters."""

from analysis.flavour.v2.r1.models import build_model as original_model
from analysis.flavour.v2.r1.models import branch_groups as original_groups


def build_model(config):
    model = original_model(config)
    if config['family'] == 'B1':
        stride = config.get('cnn_first_stride', 4)
        if stride not in [2, 4] or config.get('cnn_variant', 'resnet') != 'resnet':
            raise ValueError('R3 local-retention control requires ResNet first stride2 or4')
        model.backbone.stem[0].stride = (stride, stride)
    return model


def branch_groups(model):
    if getattr(model, 'family', None) != 'B1':
        return original_groups(model)
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
