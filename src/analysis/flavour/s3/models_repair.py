"""S3 revision 2: normalized residuals and an uncapped local readout path.

Peri-LN (ICML 2025), LayerScale (ICCV 2021), QK normalization and
partitioned point-cloud processing motivate individual components. This is
an observation-only detector adaptation, not a reproduction of those models.
Legacy checkpoints keep their original implementation in models.py.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from analysis.flavour.s2.models import PointClassifier, masked_pool, response_totals, scale_totals
from analysis.flavour.s3.models import PlaneModuleClassifier


class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Linear(dim, 4 * dim)
        self.down = nn.Linear(2 * dim, dim)

    def forward(self, x):
        gate, value = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * value)


class StableAttention(nn.Module):
    """Pre/output-normalized sublayers, small learnable residual scales, QK-LN."""
    def __init__(self, dim, heads=4, scale=.1, pair=False, empty_source_bypass=False):
        super().__init__()
        self.heads = heads
        self.empty_source_bypass = empty_source_bypass
        self.qnorm, self.snorm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.q, self.kv, self.out = nn.Linear(dim, dim), nn.Linear(dim, 2 * dim), nn.Linear(dim, dim)
        self.attention_norm, self.ffnorm, self.ffout_norm = [nn.LayerNorm(dim) for _ in range(3)]
        self.ff = SwiGLU(dim)
        self.attention_scale = nn.Parameter(torch.full((dim,), scale))
        self.ff_scale = nn.Parameter(torch.full((dim,), scale))
        self.pair = nn.Sequential(nn.Linear(3, 24), nn.SiLU(), nn.Linear(24, heads)) if pair else None

    def forward(self, x, source, mask, relations=None):
        b, nq, dim = x.shape
        width = dim // self.heads
        q = self.q(self.qnorm(x)).reshape(b, nq, self.heads, width).transpose(1, 2)
        k, v = self.kv(self.snorm(source)).reshape(b, source.shape[1], 2, self.heads, width).permute(2, 0, 3, 1, 4)
        # No trainable Q/K norm scale: initial or drifting projection norms
        # cannot alone make the softmax one-hot.
        q, k = F.layer_norm(q, (width,)), F.layer_norm(k, (width,))
        source_present = mask.any(-1)
        attention_mask = mask
        if self.empty_source_bypass:
            # Give empty rows a numerical key for SDPA, then disconnect their
            # entire attention update. No synthetic key reaches the readout.
            attention_mask = mask.clone()
            attention_mask[:, 0] |= ~source_present
        bias = q.new_zeros((b, 1, 1, source.shape[1]))
        if self.pair is not None and relations is not None:
            bias = self.pair(relations).permute(0, 3, 1, 2).to(q.dtype)
        bias = bias.masked_fill(~attention_mask[:, None, None, :], float('-inf'))
        if getattr(self, 'record_health', False):
            with torch.no_grad(), torch.autocast(device_type=q.device.type, enabled=False):
                # This explicitly labelled probe conditions on at most the
                # first 128 keys; it does not claim full-event entropy.
                keys = min(128, k.shape[2])
                logits = (q[:1, :, :8].float() @ k[:1, :, :keys].float().transpose(-1, -2)) / width**.5
                logits = logits + bias[:1, :, :8, :keys].float()
                observed = mask[:1, :keys].any()
                probability = logits.softmax(-1) if observed else torch.zeros_like(logits)
                entropy = -(probability * probability.clamp_min(1e-30).log()).sum(-1).mean()
                self.health_probe = {'key_count': keys, 'query_count': min(8, nq),
                                     'valid_key_count': int(mask[0, :keys].sum()),
                                     'empty_source': not bool(observed),
                                     'conditional_entropy_nats': float(entropy),
                                     'scope': 'first group, up to eight queries and 128 keys'}
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        attended = attended.transpose(1, 2).reshape(b, nq, dim)
        update = self.attention_norm(self.out(attended))
        if self.empty_source_bypass:
            update = update * source_present[:, None, None]
        x = x + self.attention_scale * update
        return x + self.ff_scale * self.ffout_norm(self.ff(self.ffnorm(x)))


def segment_pool(x, group, valid, segments):
    """FP32 mean/max of every valid token; empty segments have exactly zero values."""
    values = x.float()
    index = group[:, None].expand_as(values)
    count = values.new_zeros(segments).index_add_(0, group, valid.float())
    sums = values.new_zeros((segments, values.shape[1])).index_add_(0, group, values * valid[:, None])
    maximum = values.new_full(sums.shape, float('-inf'))
    maximum.scatter_reduce_(0, index, values.masked_fill(~valid[:, None], float('-inf')),
                            reduce='amax', include_self=True)
    maximum = torch.where(count[:, None] > 0, maximum, torch.zeros_like(maximum))
    return torch.cat([sums / count.clamp_min(1)[:, None], maximum], dim=-1), count > 0


class PlaneModuleClassifierV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.empty_response_bypass = config.get('empty_response_bypass', False)
        self.dim = dim = config.get('dim', 96)
        self.local_chunk_blocks = config.get('local_chunk_blocks', 256)
        self.fusion = 'latent'
        self.multiscale = config.get('compression', 'multiscale') == 'multiscale'
        self.direct = config.get('readout_skip', self.multiscale)
        scale = config.get('residual_scale', .1)
        self.input = nn.Sequential(nn.Linear(26, dim), nn.LayerNorm(dim), nn.SiLU())
        attention_options = {'scale': scale, 'empty_source_bypass': self.empty_response_bypass}
        self.local = nn.ModuleList([StableAttention(dim, **attention_options) for _ in range(2)])
        self.local_pool = nn.Sequential(nn.Linear(2 * dim + 8, dim), nn.LayerNorm(dim))
        queries = config.get('region_queries', 2) if self.multiscale else config.get('module_queries', 4)
        self.module_queries = nn.Parameter(torch.randn(queries, dim) * .02)
        self.module_read = StableAttention(dim, **attention_options)
        self.module_read_norm = nn.LayerNorm(dim)
        self.module_embedding, self.detector_embedding = nn.Embedding(32, dim), nn.Embedding(4, dim)
        self.position = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.geometry_norm = nn.LayerNorm(dim)
        self.geometry_scale = nn.Parameter(torch.full((dim,), scale))
        # Global attention runs over fixed latents; variable regional tokens
        # remain accessible directly and never need quadratic self attention.
        self.latents = nn.Parameter(torch.randn(config.get('latents', 64 if self.multiscale else 24), dim) * .02)
        self.latent_read = StableAttention(dim, **attention_options)
        self.global_blocks = nn.ModuleList([StableAttention(dim, **attention_options) for _ in range(2)])
        self.context = nn.ModuleList([StableAttention(dim, pair=True, **attention_options) for _ in range(2)]) if not self.multiscale else nn.ModuleList()
        self.token_norm = nn.LayerNorm(dim)
        self.global_norm = nn.LayerNorm(dim)
        branches = 7 if self.direct else 4  # three region/module, one global, optional three local
        self.branch_norms = nn.ModuleList([nn.LayerNorm(2 * dim) for _ in range(branches)])
        self.summary = nn.Sequential(nn.Linear(11, dim), nn.LayerNorm(dim), nn.SiLU())
        head_dim = branches * 2 * dim + dim
        # Explicit hidden normalization and a linear route keep the entire
        # classifier from disconnecting its encoder through an activation.
        self.head_input_norm = nn.LayerNorm(head_dim)
        self.head = nn.Sequential(nn.Linear(head_dim, 2 * dim), nn.LayerNorm(2 * dim),
                                  nn.SiLU(), nn.Dropout(.1), nn.Linear(2 * dim, 4))
        self.linear_readout = nn.Linear(head_dim, 4)
        self.apply(self.initialize)
        nn.init.zeros_(self.head[-1].bias)
        nn.init.zeros_(self.linear_readout.bias)
        self.compiled_features = torch.compile(self.encode_features, dynamic=True) if config.get('compile_local') else None

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.trunc_normal_(module.weight, std=.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode_channel_features(self, nodes, mask):
        """Per-channel features before pooling, also available to SSL decoders."""
        xy = nodes[..., 16:18] * (2 * torch.pi)
        fourier = torch.cat([xy.sin(), xy.cos(), (2 * xy).sin(), (2 * xy).cos()], -1)
        fourier = fourier * nodes[..., 3:5].repeat(1, 1, 4)
        x = self.input(torch.cat([nodes, fourier], -1))
        for layer in self.local:
            x = layer(x, x, mask) * mask[..., None]
        return x

    def encode_features(self, nodes, mask):
        x = self.encode_channel_features(nodes, mask)
        summary = scale_totals(response_totals(nodes, mask)).to(x.dtype)
        return torch.cat([masked_pool(x, mask), summary], -1)

    def encode_local(self, nodes, mask):
        features = (self.compiled_features(nodes, mask) if self.compiled_features is not None and nodes.is_cuda
                    else self.encode_features(nodes, mask))
        # The observation hook is outside the compiled encoder. A health
        # sample must never switch the numerical implementation used to fit.
        return self.local_pool(features)

    def encode(self, batch):
        """Observation features shared by classification and S4 objectives."""
        nodes, mask = batch['nodes'], batch['local_mask']
        if self.empty_response_bypass:
            mask = mask & (nodes[..., 15] > 0)
        pieces = []
        for start in range(0, len(nodes), self.local_chunk_blocks):
            values = nodes[start:start + self.local_chunk_blocks], mask[start:start + self.local_chunk_blocks]
            pieces.append(checkpoint(self.encode_local, *values, use_reentrant=False)
                          if self.training and torch.is_grad_enabled() else self.encode_local(*values))
        local = torch.cat(pieces)
        if self.empty_response_bypass:
            local = local * mask.any(-1)[:, None]
        prefix = 'region' if self.multiscale else 'module'
        source_mask = batch[prefix + '_mask']
        packed = local.new_zeros((*source_mask.shape, self.dim))
        packed[batch[prefix + '_group'], batch[prefix + '_position']] = local
        q = self.module_queries[None].expand(len(packed), -1, -1)
        tokens = self.module_read_norm(self.module_read(q, packed, source_mask))
        ids, detector, geo = [batch[prefix + suffix] for suffix in ['_id', '_detector', '_geometry']]
        embedding = self.module_embedding(ids) + self.detector_embedding(detector) + self.position(geo[:, 2:3])
        tokens = tokens + self.geometry_scale * self.geometry_norm(embedding)[:, None]
        event = batch[prefix + '_event']
        b = batch['event_mask'].shape[0]
        counts = torch.bincount(event, minlength=b)
        slot = torch.arange(len(event), device=event.device) - torch.cumsum(counts, 0).sub(counts)[event]
        nq = tokens.shape[1]
        offsets = slot[:, None] * nq + torch.arange(nq, device=event.device)[None]
        width = int(counts.max()) * nq
        features = tokens.new_zeros((b, width, self.dim))
        valid = torch.zeros((b, width), device=event.device, dtype=torch.bool)
        features[event[:, None], offsets] = tokens
        valid[event[:, None], offsets] = True
        z = features.new_zeros((b, width))
        module = torch.zeros((b, width), device=event.device, dtype=torch.long)
        types = torch.full_like(module, 3)
        z[event[:, None], offsets] = geo[:, 2:3]
        module[event[:, None], offsets] = ids[:, None]
        types[event[:, None], offsets] = detector[:, None]
        if self.empty_response_bypass:
            valid = valid & (types < 3)
        if len(self.context):
            relations = torch.stack([z[:, :, None] - z[:, None, :],
                                     (module[:, :, None] == module[:, None, :]).float(),
                                     (types[:, :, None] == types[:, None, :]).float()], -1)
            for layer in self.context:
                features = layer(features, features, valid, relations) * valid[..., None]
        features = self.token_norm(features)
        x = self.latent_read(self.latents[None].expand(b, -1, -1), features, valid)
        all_latents = torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)
        for layer in self.global_blocks:
            x = layer(x, x, all_latents)
        pools = [masked_pool(features, valid & (types == d)) for d in range(3)]
        presence = torch.stack([(valid & (types == d)).any(-1) for d in range(3)], -1)
        normalized_global = self.global_norm(x)
        pools.append(masked_pool(normalized_global, all_latents))
        branch_presence = [presence[:, d] for d in range(3)] + [torch.ones(b, device=x.device, dtype=torch.bool)]
        if self.direct:
            # Every local block reaches classification, including blocks whose
            # event has thousands of regions. No top-k, event cap or resampling.
            block_detector = batch['block_detector']
            index = batch['event_index'] * 3 + block_detector.clamp_max(2)
            direct, _ = segment_pool(local, index, block_detector < 3, b * 3)
            direct = direct.reshape(b, 3, -1)
            pools.extend(direct[:, d] for d in range(3))
            branch_presence.extend(presence[:, d] for d in range(3))
        pools = [norm(pool.float()) * active[:, None] for norm, pool, active in zip(self.branch_norms, pools, branch_presence)]
        totals = nodes.new_zeros((b, 8), dtype=torch.float32)
        totals.index_add_(0, batch['event_index'], response_totals(nodes, mask))
        # Classifier statistics stay FP32 even while the encoders use BF16.
        with torch.autocast(device_type=nodes.device.type, enabled=False):
            summary = self.summary(torch.cat([scale_totals(totals), presence.float()], -1))
            head_input = self.head_input_norm(torch.cat(pools + [summary], -1))
        return {'head_input': head_input, 'branches': torch.stack(pools, 1), 'summary': summary,
                'global_tokens': normalized_global, 'region_tokens': features,
                'region_mask': valid, 'region_detector': types, 'local_tokens': local}

    def forward(self, batch):
        features = self.encode(batch)
        with torch.autocast(device_type=batch['nodes'].device.type, enabled=False):
            head_input = features['head_input']
            return self.head(head_input) + self.linear_readout(head_input)


def build_model(config):
    architecture = config.get('architecture', 'plane_module_v2')
    if architecture == 'plane_module_v2':
        return PlaneModuleClassifierV2(config)
    if architecture == 'plane_module':
        return PlaneModuleClassifier(config)
    if architecture == 'part':
        model = PointClassifier('part', dim=config.get('dim', 64))
        model.local_chunk_blocks = config.get('local_chunk_blocks', 256)
        model.compiled_local = torch.compile(model.encode_local, dynamic=True) if config.get('compile_local') else None
        return model
    raise ValueError(architecture)
