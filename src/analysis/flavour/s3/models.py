"""Complete plane-window encoding, physical module context and late/joint/latent fusion."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from analysis.flavour.s2.models import PairAttention, masked_pool, response_totals, scale_totals


class CrossAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.qnorm, self.knorm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.q, self.kv, self.out = nn.Linear(dim, dim), nn.Linear(dim, 2 * dim), nn.Linear(dim, dim)
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, query, source, mask):
        b, nq, d = query.shape
        q = self.q(self.qnorm(query)).reshape(b, nq, self.heads, d // self.heads).transpose(1, 2)
        k, v = self.kv(self.knorm(source)).reshape(b, source.shape[1], 2, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        bias = torch.zeros((b, 1, 1, source.shape[1]), device=q.device, dtype=q.dtype).masked_fill(~mask[:, None, None], float('-inf'))
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=bias).transpose(1, 2).reshape(b, nq, d)
        x = query + self.out(attended)
        return x + self.ff(x)


class ModuleAttention(nn.Module):
    def __init__(self, dim, pair_geometry=True):
        super().__init__()
        self.heads = 4
        self.norm = nn.LayerNorm(dim)
        self.qkv, self.out = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.pair = nn.Sequential(nn.Linear(3, 24), nn.GELU(), nn.Linear(24, 4)) if pair_geometry else None
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, x, mask, z, module, detector):
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, n, 3, 4, d // 4).permute(2, 0, 3, 1, 4)
        bias = torch.zeros((b, 1, 1, n), device=x.device, dtype=x.dtype)
        if self.pair is not None:
            delta = z[:, :, None] - z[:, None, :]
            relations = torch.stack([delta, (module[:, :, None] == module[:, None, :]).float(),
                                    (detector[:, :, None] == detector[:, None, :]).float()], dim=-1)
            bias = self.pair(relations).permute(0, 3, 1, 2)
        bias = bias.masked_fill(~mask[:, None, None], float('-inf'))
        attention = F.scaled_dot_product_attention(q, k, v, attn_mask=bias).transpose(1, 2).reshape(b, n, d)
        x = x + self.out(attention)
        return (x + self.ff(x)) * mask[..., None]


class PlaneModuleClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.get('dim', 96)
        self.dim = dim
        self.input = nn.Sequential(nn.Linear(26, dim), nn.LayerNorm(dim), nn.GELU())
        self.local = nn.ModuleList([PairAttention(dim, pair_bias=False) for _ in range(2)])
        self.local_pool = nn.Linear(2 * dim + 8, dim)
        self.module_queries = nn.Parameter(torch.randn(config.get('module_queries', 4), dim) * .02)
        self.module_read = CrossAttention(dim)
        if config.get('module_geometry', True):
            self.module_embedding = nn.Embedding(32, dim)
            self.detector_embedding = nn.Embedding(4, dim)
            self.position = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        self.context = nn.ModuleList([ModuleAttention(dim, config.get('pair_geometry', True))
                                      for _ in range(2 if config.get('module_context', True) else 0)])
        self.fusion = config.get('fusion', 'latent')
        if self.fusion == 'latent':
            self.latents = nn.Parameter(torch.randn(config.get('latents', 24), dim) * .02)
            self.latent_read = CrossAttention(dim)
        self.global_blocks = nn.ModuleList([ModuleAttention(dim, self.fusion == 'joint' and config.get('pair_geometry', True))
                                           for _ in range(0 if self.fusion == 'late' else 2)])
        self.head = nn.Sequential(nn.Linear(6 * dim + 8, 2 * dim), nn.GELU(), nn.Dropout(.1), nn.Linear(2 * dim, 4))

    def encode_local(self, nodes, mask):
        xy = nodes[..., 16:18] * (2 * torch.pi)
        if not self.config.get('local_residual', True):
            fourier = nodes.new_zeros((*nodes.shape[:2], 8))
        else:
            fourier = torch.cat([xy.sin(), xy.cos(), (2 * xy).sin(), (2 * xy).cos()], -1)
            fourier = fourier * nodes[..., 3:5].repeat(1, 1, 4)
        x = self.input(torch.cat([nodes, fourier], -1))
        for layer in self.local:
            x = layer(x, mask, nodes)
        summary = scale_totals(response_totals(nodes, mask)).to(x.dtype)
        return self.local_pool(torch.cat([masked_pool(x, mask), summary], -1))

    def forward(self, batch):
        nodes, mask = batch['nodes'], batch['local_mask']
        encoded = []
        for start in range(0, len(nodes), 256):
            values = (nodes[start:start + 256], mask[start:start + 256])
            encoded.append(checkpoint(self.encode_local, *values, use_reentrant=False)
                           if self.training and torch.is_grad_enabled() else self.encode_local(*values))
        local = torch.cat(encoded)
        group = batch['module_group']
        module_mask = batch['module_mask']
        packed = local.new_zeros((*module_mask.shape, self.dim))
        packed[group, batch['module_position']] = local
        queries = self.module_queries[None].expand(len(module_mask), -1, -1)
        module_tokens = self.module_read(queries, packed, module_mask)
        geo = batch['module_geometry']
        if self.config.get('module_geometry', True):
            embedding = self.module_embedding(batch['module_id']) + self.detector_embedding(batch['module_detector']) + self.position(geo[:, 2:3])
            module_tokens = module_tokens + embedding[:, None]
        # Pack fixed per-module query arrays; all physical modules represented.
        event = batch['module_event']
        b = batch['event_mask'].shape[0]
        counts = torch.bincount(event, minlength=b)
        # Module groups are sorted by event in the collator.
        slot = torch.arange(len(event), device=event.device) - torch.cumsum(counts, 0).sub(counts)[event]
        q = module_tokens.shape[1]
        width = int(counts.max()) * q
        features = local.new_zeros((b, width, self.dim))
        valid = torch.zeros((b, width), device=local.device, dtype=torch.bool)
        z = nodes.new_zeros((b, width))
        module = torch.zeros((b, width), device=local.device, dtype=torch.long)
        detector = torch.full((b, width), 3, device=local.device, dtype=torch.long)
        offsets = slot[:, None] * q + torch.arange(q, device=local.device)[None]
        features[event[:, None], offsets] = module_tokens
        valid[event[:, None], offsets] = True
        z[event[:, None], offsets] = geo[:, 2:3]
        module[event[:, None], offsets] = batch['module_id'][:, None]
        detector[event[:, None], offsets] = batch['module_detector'][:, None]
        for layer in self.context:
            features = layer(features, valid, z, module, detector)
        if self.fusion == 'latent':
            x = self.latent_read(self.latents[None].expand(b, -1, -1), features, valid)
            latent_mask = torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)
            zeros = z.new_zeros(x.shape[:2])
            for layer in self.global_blocks:
                x = layer(x, latent_mask, zeros, zeros, zeros)
            # Retain modality-specific module pools alongside global latent context.
            context = masked_pool(x, latent_mask)
            pools = [masked_pool(features, valid & (detector == d)) + context for d in range(3)]
        else:
            if self.fusion == 'joint':
                for layer in self.global_blocks:
                    features = layer(features, valid, z, module, detector)
            pools = [masked_pool(features, valid & (detector == d)) for d in range(3)]
        totals = nodes.new_zeros((b, 8))
        totals.index_add_(0, batch['event_index'], response_totals(nodes, mask))
        return self.head(torch.cat(pools + [scale_totals(totals).to(local.dtype)], -1))
