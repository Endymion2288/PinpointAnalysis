"""S5 heads on the repaired, uncapped multiscale native-response encoder."""
import torch
from torch import nn
from torch.nn import functional as F

from analysis.flavour.s3.models_repair import PlaneModuleClassifierV2, segment_pool


class PhysicsModel(PlaneModuleClassifierV2):
    def __init__(self, config):
        super().__init__(config)
        width = self.head[0].in_features
        classes = 6 if config.get('six_way') else 4
        self.head[-1] = nn.Linear(self.head[-1].in_features, classes)
        self.linear_readout = nn.Linear(width, classes)
        self.auxiliary_heads = nn.ModuleDict({
            'vertex': nn.Sequential(nn.Linear(width, 128), nn.LayerNorm(128), nn.SiLU(), nn.Linear(128, 3)),
            'semantics': nn.Linear(width, 3), 'charm': nn.Linear(width, 1), 'kinematics': nn.Linear(width, 4)})
        self.task_log_variances = nn.ParameterDict({k: nn.Parameter(torch.zeros(())) for k in config.get('auxiliaries', [])})
        self.fusion_norm = nn.LayerNorm(width)
        if config.get('use_objects'):
            self.object_branch = nn.Sequential(nn.Linear(31, 128), nn.LayerNorm(128), nn.SiLU(), nn.Linear(128, width), nn.LayerNorm(width))
            self.object_scale = nn.Parameter(torch.full((width,), .1))
        if config.get('relative_path'):
            self.relative_branch = nn.Sequential(nn.Linear(7, 96), nn.LayerNorm(96), nn.SiLU())
            self.relative_pool = nn.Sequential(nn.Linear(192, width), nn.LayerNorm(width))
            self.relative_scale = nn.Parameter(torch.full((width,), .1))
        # Initialize added heads without reinitializing the encoder.
        for module in [self.head[-1], self.linear_readout, self.auxiliary_heads, getattr(self, 'object_branch', nn.Identity()),
                       getattr(self, 'relative_branch', nn.Identity()), getattr(self, 'relative_pool', nn.Identity())]:
            module.apply(self.initialize)

    def forward(self, observation, objects=None):
        encoded = self.encode(observation)
        with torch.autocast(device_type=observation['nodes'].device.type, enabled=False):
            raw = encoded['head_input'].float()
            # The vertex is predicted from measurements, never a truth input.
            vertex = self.auxiliary_heads['vertex'](raw)
            fused = raw
            if self.config.get('relative_path'):
                nodes = observation['nodes'].float()
                valid = observation['local_mask'] & (nodes[..., 15] > 0) & (nodes[..., 14] < .5)
                count = valid.sum(1).clamp_min(1)
                xyz = (nodes[..., :3]*valid[..., None]).sum(1)/count[:, None]
                event = observation['event_index']
                # Tracker z in the raw representation is an ordinal code;
                # only pixel/bar physical positions enter this relative path.
                relative = xyz-vertex[event]
                detector = nodes[:, 0, 12:14]
                values = self.relative_branch(torch.cat([relative, relative.square(), detector[:, :1]], -1))
                pooled, _ = segment_pool(values, event, valid.any(1), len(raw))
                fused = fused+self.relative_scale*self.relative_pool(pooled)
            if self.config.get('use_objects'):
                fused = fused+self.object_scale*self.object_branch(objects.float())
            h = self.fusion_norm(fused)
            logits = self.head(h)+self.linear_readout(h)
            output = {'logits': logits, 'features': h, 'raw_features': raw, 'vertex': vertex}
            output.update({k: head(h) for k, head in self.auxiliary_heads.items() if k != 'vertex'})
            probability = logits.softmax(-1)
            output['probability'] = (torch.stack([probability[:, 0], probability[:, 1], probability[:, 2:5].sum(-1), probability[:, 5]], -1)
                                     if self.config.get('six_way') else probability)
            return output


def masked_mean(values, mask):
    values = values.reshape(len(mask), -1).mean(-1)
    return (values*mask.float()).sum()/mask.sum().clamp_min(1)


def losses(model, output, targets, label):
    if model.config.get('six_way'):
        mask = targets['six'] >= 0
        main = masked_mean(F.cross_entropy(output['logits'], targets['six'].clamp_min(0), reduction='none'), mask)
    else:
        main = F.cross_entropy(output['logits'], label)
    values = {'classification': main}
    counts = {}
    for task in model.config.get('auxiliaries', []):
        mask = targets[task+'_mask']
        counts[task] = int(mask.sum())
        if task in ['vertex', 'kinematics']:
            per_event = F.smooth_l1_loss(output[task], targets[task].float(), reduction='none', beta=.1)
        elif task == 'semantics':
            per_event = -(targets[task].float()*output[task].log_softmax(-1)).sum(-1)
        elif task == 'charm':
            per_event = F.binary_cross_entropy_with_logits(output[task], targets[task].float(), reduction='none')
        else:
            raise ValueError(task)
        values[task] = masked_mean(per_event, mask)
    total = main
    for task, loss in values.items():
        if task == 'classification':
            continue
        weighted = model.config.get('auxiliary_weight', .1)*loss
        if model.config.get('uncertainty_weighting') and counts[task]:
            log_variance = model.task_log_variances[task]
            total = total+torch.exp(-log_variance)*weighted+log_variance
        else:
            total = total+weighted
    return total, values, counts


def build_model(config):
    model = PhysicsModel(config)
    if config.get('pretrained_checkpoint'):
        state = torch.load(config['pretrained_checkpoint'], map_location='cpu', weights_only=False)['model']
        source = {k.removeprefix('encoder.'): v for k, v in state.items() if k.startswith('encoder.')}
        excluded = ('head.', 'linear_readout.', 'head_input_norm.', 'branch_norms.', 'summary.')
        source = {k: v for k, v in source.items() if not k.startswith(excluded)}
        expected = {k for k in PlaneModuleClassifierV2({**config, 'compile_local': False}).state_dict() if not k.startswith(excluded)}
        if set(source) != expected:
            raise ValueError('pretrained repaired encoder incomplete')
        model.load_state_dict(source, strict=False)
    return model
