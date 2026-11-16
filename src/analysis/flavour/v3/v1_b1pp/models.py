"""ZAlignedStereoNet (B1++) zero-hole transverse-preserving stereo architecture (~2.8M params)."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from analysis.flavour.v2.r1.tasks import output_size
from analysis.flavour.v3.common.layers import DropPath, ScaledQKNormAttention
from analysis.flavour.v3.v1_b1pp.representation import FINE_BINS, WIDE_BINS


class TransverseResBlock(nn.Module):
    """2D stereo residual block mixing adjacent z-planes and W without downsampling z."""

    def __init__(self, cin: int, cout: int, stride_w: int = 1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=(3, 3), stride=(1, stride_w), padding=(1, 1), bias=False),
            nn.GroupNorm(8, cout),
            nn.SiLU(),
            nn.Conv2d(cout, cout, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.GroupNorm(8, cout),
        )
        self.skip = (
            nn.Identity()
            if cin == cout and stride_w == 1
            else nn.Conv2d(cin, cout, kernel_size=1, stride=(1, stride_w), bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.body(x) + self.skip(x))


class TransverseLayerEncoder(nn.Module):
    """Encode (B, C_in, N_z, W) via 2D stereo convs -> (plane_tokens, global_2d)."""

    def __init__(self, in_ch: int, channels: tuple[int, ...], stem_k: int, stem_s: int, out_dim: int) -> None:
        super().__init__()
        c0 = channels[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, c0, kernel_size=(3, stem_k), stride=(1, stem_s), padding=(1, stem_k // 2), bias=False),
            nn.GroupNorm(8, c0),
            nn.SiLU(),
        )
        blocks = []
        cin = c0
        for cout in channels[1:]:
            blocks.append(TransverseResBlock(cin, cout, stride_w=2))
            blocks.append(TransverseResBlock(cout, cout, stride_w=1))
            cin = cout
        self.stages = nn.Sequential(*blocks)
        self.last_ch = cin
        self.attn_score = nn.Linear(cin, 1)
        self.proj = nn.Sequential(
            nn.Linear(3 * cin + 2, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor, plane_active: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C_in, N_z, W), plane_active: (B, N_z) bool
        feat2d = self.stages(self.stem(x))  # (B, C, N_z, W')
        act_mask_2d = plane_active[:, None, :, None]
        any_act = plane_active.any(dim=1, keepdim=True).to(feat2d.dtype)
        feat_m = feat2d * act_mask_2d.to(feat2d.dtype)
        denom = plane_active.sum(dim=1, keepdim=True).clamp(min=1).to(feat2d.dtype) * float(feat2d.shape[3])
        g_mean = feat_m.sum(dim=(2, 3)) / denom
        g_max = feat_m.masked_fill(~act_mask_2d, -1e4).amax(dim=(2, 3)) * any_act
        global_2d = torch.cat([g_mean, g_max], dim=-1) * any_act  # (B, 2*C)

        # Coordinate-aware per-plane sequence embedding (B, N_z, W', C)
        feat = feat2d.permute(0, 2, 3, 1)
        mean_p = feat.mean(dim=2)
        max_p = feat.amax(dim=2)
        w = torch.softmax(self.attn_score(feat).float(), dim=2)  # (B, N_z, W', 1) FP32
        attn_p = (w.to(feat.dtype) * feat).sum(dim=2)
        u_grid = torch.linspace(-1.0, 1.0, feat.shape[2], device=feat.device, dtype=torch.float32)[None, None, :, None]
        u_mean = (w * u_grid).sum(dim=2)
        u_std = torch.sqrt(((w * (u_grid - u_mean.unsqueeze(2)).square()).sum(dim=2)).clamp_min(1e-6))
        cat_in = torch.cat([mean_p, max_p, attn_p, u_mean.to(feat.dtype), u_std.to(feat.dtype)], dim=-1)
        pooled = self.proj(cat_in)
        return pooled * plane_active.unsqueeze(-1).to(pooled.dtype), global_2d


class LongitudinalLocalMix(nn.Module):
    """1D depthwise conv (k=5) + GLU along physical z-sorted detector sequence."""

    def __init__(self, dim: int, drop_path: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.pw_up = nn.Linear(dim, 2 * dim)
        self.pw_down = nn.Linear(dim, dim)
        self.gamma = nn.Parameter(torch.full((dim,), 0.5))
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.dwconv(h.transpose(1, 2)).transpose(1, 2)
        a, b = self.pw_up(h).chunk(2, dim=-1)
        h = self.pw_down(a * torch.sigmoid(b))
        return x + self.drop_path(self.gamma * h)


class WindowAttnPool(nn.Module):
    """Mean + Attention pooling over a variable-length slice of sequence embeddings."""

    def __init__(self, dim: int, out_dim: int = 128) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)
        self.proj = nn.Sequential(
            nn.Linear(2 * dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D), mask: (B, L) bool
        any_valid = mask.any(dim=1)
        safe_mask = mask.clone()
        if (~any_valid).any():
            safe_mask[~any_valid, 0] = True
        denom = safe_mask.sum(dim=1, keepdim=True).clamp(min=1).to(x.dtype)
        mean_v = (x * safe_mask.unsqueeze(-1).to(x.dtype)).sum(dim=1) / denom
        logits = self.score(x).squeeze(-1).float().masked_fill(~safe_mask, float('-inf'))
        weights = torch.softmax(logits, dim=-1).to(x.dtype)
        attn_v = (x * weights.unsqueeze(-1)).sum(dim=1)
        out = self.proj(torch.cat([mean_v, attn_v], dim=-1))
        return out * any_valid.unsqueeze(-1).to(out.dtype)


class SpectrometerEncoder(nn.Module):
    """TrackerTrackAttention (14-dim candidates) + TrackerSet (6-dim raw hits) -> out_dim."""

    def __init__(self, out_dim: int = 160) -> None:
        super().__init__()
        self.cand_in = nn.Sequential(
            nn.Linear(14, 96),
            nn.LayerNorm(96),
            nn.SiLU(),
            nn.Linear(96, 96),
            nn.SiLU(),
        )
        self.cand_attn1 = ScaledQKNormAttention(96, heads=4, init_scale=8.0, drop_path=0.05)
        self.cand_attn2 = ScaledQKNormAttention(96, heads=4, init_scale=8.0, drop_path=0.05)

        self.hit_in = nn.Sequential(
            nn.Linear(6, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.hit_score = nn.Linear(64, 1)
        self.fusion = nn.Sequential(
            nn.Linear(192 + 128, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        trk_cand: torch.Tensor,
        trk_cand_mask: torch.Tensor,
        trk_hits: torch.Tensor,
        trk_hit_mask: torch.Tensor,
    ) -> torch.Tensor:
        has_cand = trk_cand_mask.any(dim=1)
        hc = self.cand_in(trk_cand)
        hc = self.cand_attn1(hc, hc, trk_cand_mask)
        hc = self.cand_attn2(hc, hc, trk_cand_mask)
        cm = trk_cand_mask.unsqueeze(-1).to(hc.dtype)
        c_mean = (hc * cm).sum(dim=1) / trk_cand_mask.sum(dim=1, keepdim=True).clamp(min=1).to(hc.dtype)
        c_max = hc.masked_fill(~trk_cand_mask.unsqueeze(-1), -1e4).amax(dim=1)
        cand_pool = torch.cat([c_mean, c_max], dim=-1) * has_cand.unsqueeze(-1).to(hc.dtype)

        has_hit = trk_hit_mask.any(dim=1)
        hh = self.hit_in(trk_hits)
        hm = trk_hit_mask.unsqueeze(-1).to(hh.dtype)
        h_mean = (hh * hm).sum(dim=1) / trk_hit_mask.sum(dim=1, keepdim=True).clamp(min=1).to(hh.dtype)
        safe_hm = trk_hit_mask.clone()
        if (~has_hit).any():
            safe_hm[~has_hit, 0] = True
        h_w = torch.softmax(self.hit_score(hh).squeeze(-1).float().masked_fill(~safe_hm, float('-inf')), dim=-1).to(hh.dtype)
        h_attn = (hh * h_w.unsqueeze(-1)).sum(dim=1)
        hit_pool = torch.cat([h_mean, h_attn], dim=-1) * has_hit.unsqueeze(-1).to(hh.dtype)

        return self.fusion(torch.cat([cand_pool, hit_pool], dim=-1)) * has_hit.unsqueeze(-1).to(hh.dtype)


class ZAlignedStereoNet(nn.Module):
    """Sensor-native zero-hole z-aligned B1++ strong baseline."""

    def __init__(self, layout: dict, config: dict) -> None:
        super().__init__()
        self.layout = layout
        self.config = config
        self.n_seq = int(layout['n_seq'])
        self.n_pix = int(layout['n_pix'])
        self.n_xbar = int(layout['n_xbar'])
        self.n_ybar = int(layout['n_ybar'])
        self.n_bars = int(layout['n_bars'])
        self.channel_dropout = float(config.get('channel_dropout', 0.03))
        dim = int(config.get('dim', 160))
        n_layers = int(config.get('n_long_layers', 4))
        heads = int(config.get('heads', 8))
        drop_path = float(config.get('drop_path', 0.10))

        # Static geometry buffers
        self.register_buffer('plane_geom', torch.from_numpy(layout['plane_geom']))
        self.register_buffer('seq_mod', torch.from_numpy(layout['seq_mod']))
        self.register_buffer('pix_to_seq', torch.from_numpy(layout['pix_to_seq']))
        self.register_buffer('xbar_to_seq', torch.from_numpy(layout['xbar_to_seq']))
        self.register_buffer('ybar_to_seq', torch.from_numpy(layout['ybar_to_seq']))
        self.register_buffer('xbar_geom', torch.stack([
            torch.from_numpy(layout['xbar_z_norm']),
            torch.from_numpy(layout['xbar_x0_norm']),
        ], dim=0))  # (2, 54)
        self.register_buffer('ybar_geom', torch.stack([
            torch.from_numpy(layout['ybar_z_norm']),
            torch.from_numpy(layout['ybar_x0_norm']),
        ], dim=0))  # (2, 42)

        # Pairwise longitudinal distance matrix (L_seq, L_seq, 2) -> [|dz|/500, |dX0|/20]
        z_t = torch.from_numpy(layout['seq_z_mm'])
        x0_t = torch.from_numpy(layout['seq_x0'])
        dz = torch.abs(z_t[:, None] - z_t[None, :]) / 500.0
        dx0 = torch.abs(x0_t[:, None] - x0_t[None, :]) / 20.0
        self.register_buffer('seq_pair_geom', torch.stack([dz, dx0], dim=-1))  # (111, 111, 2)

        # Normalized transverse coordinate channels for colinear 2D track extraction
        self.register_buffer('u_bars', torch.linspace(-1.0, 1.0, self.n_bars))
        self.register_buffer('u_fine', torch.linspace(-1.0, 1.0, FINE_BINS))
        self.register_buffer('u_wide', torch.linspace(-1.0, 1.0, WIDE_BINS))

        # 2D stereo transverse encoders (bar X, bar Y, pixel fine X/Y, pixel wide X/Y)
        self.xbar_enc = TransverseLayerEncoder(6, (32, 48, 64), stem_k=3, stem_s=1, out_dim=dim)
        self.ybar_enc = TransverseLayerEncoder(6, (32, 48, 64), stem_k=3, stem_s=1, out_dim=dim)
        self.pfine_x_enc = TransverseLayerEncoder(5, (16, 24, 32, 48), stem_k=7, stem_s=4, out_dim=64)
        self.pfine_y_enc = TransverseLayerEncoder(5, (16, 24, 32, 48), stem_k=7, stem_s=4, out_dim=64)
        self.pwide_x_enc = TransverseLayerEncoder(5, (16, 24, 32, 48), stem_k=5, stem_s=2, out_dim=64)
        self.pwide_y_enc = TransverseLayerEncoder(5, (16, 24, 32, 48), stem_k=5, stem_s=2, out_dim=64)
        self.pix_fuse = nn.Sequential(
            nn.Linear(4 * 64, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
        )
        stereo_in_dim = 2 * 64 + 2 * 64 + 4 * (2 * 48)
        self.stereo_2d_proj = nn.Sequential(
            nn.Linear(stereo_in_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
        )

        self.geom_emb = nn.Sequential(nn.Linear(4, dim), nn.LayerNorm(dim))
        self.mod_emb = nn.Embedding(3, dim)
        self.active_emb = nn.Linear(1, dim)
        self.pair_bias_mlp = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, heads))

        self.long_blocks = nn.ModuleList([
            nn.ModuleDict({
                'conv': LongitudinalLocalMix(dim, drop_path=drop_path),
                'attn': ScaledQKNormAttention(dim, heads=heads, init_scale=12.0, drop_path=drop_path),
            })
            for _ in range(n_layers)
        ])

        self.win_start = WindowAttnPool(dim, 96)
        self.win_peak = WindowAttnPool(dim, 96)
        self.win_tail = WindowAttnPool(dim, 96)
        self.win_global = WindowAttnPool(dim, 96)

        self.spectrometer = SpectrometerEncoder(out_dim=dim)
        self.cobs_emb = nn.Sequential(
            nn.Dropout(0.20),
            nn.Linear(16, 48),
            nn.LayerNorm(48),
            nn.SiLU(),
        )

        # Head outputs requested task dimension ('T3': 3, 'T4': 4)
        fused_dim = 4 * 96 + 128 + dim + 48
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 256),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(256, output_size(config['task'])),
        )

    def _scatter_bar(
        self,
        b: int,
        n_panels: int,
        geom_2: torch.Tensor,
        ev: torch.Tensor,
        idx: torch.Tensor,
        val: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dense = val.new_zeros((b, 6, n_panels * self.n_bars))
        if len(ev):
            if self.training and self.channel_dropout > 0.0:
                keep = torch.rand(len(ev), device=val.device) >= self.channel_dropout
                ev = ev[keep]
                idx = idx[keep]
                val = val[keep]
            if len(ev):
                dense[ev, 0, idx] = val[:, 0]
                dense[ev, 1, idx] = val[:, 1]
                dense[ev, 4, idx] = 1.0
        dense = dense.view(b, 6, n_panels, self.n_bars)
        dense[:, 2:4, :, :] = geom_2[None, :, :, None].expand(b, -1, -1, self.n_bars)
        dense[:, 5, :, :] = dense[:, 4, :, :] * self.u_bars[None, None, :]
        plane_active = dense[:, 4].any(dim=-1)  # (B, n_panels)
        plane_energy = dense[:, 0].sum(dim=-1)  # (B, n_panels)
        return dense, plane_active, plane_energy

    def _scatter_pixel(
        self,
        b: int,
        n_bins: int,
        u_coord: torch.Tensor,
        ev: torch.Tensor,
        idx: torch.Tensor,
        val: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dense = val.new_zeros((b, 5, self.n_pix * n_bins))
        if len(ev):
            if self.training and self.channel_dropout > 0.0:
                keep = torch.rand(len(ev), device=val.device) >= self.channel_dropout
                ev = ev[keep]
                idx = idx[keep]
                val = val[keep]
            if len(ev):
                dense[ev, 0, idx] = val[:, 0]
                dense[ev, 1, idx] = val[:, 1]
                dense[ev, 2, idx] = val[:, 2]
                dense[ev, 3, idx] = 1.0
        dense = dense.view(b, 5, self.n_pix, n_bins)
        dense[:, 4, :, :] = dense[:, 3, :, :] * u_coord[None, None, :]
        plane_active = dense[:, 3].any(dim=-1)  # (B, 15)
        plane_density = dense[:, 0].sum(dim=-1)  # (B, 15)
        return dense, plane_active, plane_density

    def forward(self, batch: dict) -> torch.Tensor:
        b = len(batch['label'])
        device = batch['c_obs'].device

        xbar, xbar_act, xbar_e = self._scatter_bar(
            b, self.n_xbar, self.xbar_geom, batch['bx_ev'], batch['bx_idx'], batch['bx_val']
        )
        ybar, ybar_act, ybar_e = self._scatter_bar(
            b, self.n_ybar, self.ybar_geom, batch['by_ev'], batch['by_idx'], batch['by_val']
        )
        fx, fx_act, fx_d = self._scatter_pixel(b, FINE_BINS, self.u_fine, batch['fx_ev'], batch['fx_idx'], batch['fx_val'])
        fy, fy_act, fy_d = self._scatter_pixel(b, FINE_BINS, self.u_fine, batch['fy_ev'], batch['fy_idx'], batch['fy_val'])
        wx, wx_act, wx_d = self._scatter_pixel(b, WIDE_BINS, self.u_wide, batch['wx_ev'], batch['wx_idx'], batch['wx_val'])
        wy, wy_act, wy_d = self._scatter_pixel(b, WIDE_BINS, self.u_wide, batch['wy_ev'], batch['wy_idx'], batch['wy_val'])

        h_xbar, g2_xb = self.xbar_enc(xbar, xbar_act)  # (B, 54, D), (B, 160)
        h_ybar, g2_yb = self.ybar_enc(ybar, ybar_act)  # (B, 42, D), (B, 160)

        pix_act = fx_act | fy_act | wx_act | wy_act  # (B, 15)
        h_pfx, g2_pfx = self.pfine_x_enc(fx, fx_act)
        h_pfy, g2_pfy = self.pfine_y_enc(fy, fy_act)
        h_pwx, g2_pwx = self.pwide_x_enc(wx, wx_act)
        h_pwy, g2_pwy = self.pwide_y_enc(wy, wy_act)
        h_pix = self.pix_fuse(torch.cat([h_pfx, h_pfy, h_pwx, h_pwy], dim=-1)) * pix_act.unsqueeze(-1).to(h_xbar.dtype)
        stereo_2d_feat = self.stereo_2d_proj(torch.cat([g2_xb, g2_yb, g2_pfx, g2_pfy, g2_pwx, g2_pwy], dim=-1))

        # Assemble physical z-ordered 111-plane sequence
        seq = h_xbar.new_zeros((b, self.n_seq, h_xbar.shape[-1]))
        seq[:, self.pix_to_seq, :] = h_pix
        seq[:, self.xbar_to_seq, :] = h_xbar
        seq[:, self.ybar_to_seq, :] = h_ybar

        seq_act = torch.zeros((b, self.n_seq), dtype=torch.bool, device=device)
        seq_act[:, self.pix_to_seq] = pix_act
        seq_act[:, self.xbar_to_seq] = xbar_act
        seq_act[:, self.ybar_to_seq] = ybar_act

        seq_energy = h_xbar.new_zeros((b, self.n_seq))
        seq_energy[:, self.pix_to_seq] = 0.25 * (fx_d + fy_d + wx_d + wy_d)
        seq_energy[:, self.xbar_to_seq] = xbar_e
        seq_energy[:, self.ybar_to_seq] = ybar_e

        seq = (
            seq
            + self.geom_emb(self.plane_geom)[None, :, :]
            + self.mod_emb(self.seq_mod)[None, :, :]
            + self.active_emb(seq_act.unsqueeze(-1).to(seq.dtype))
        )

        # Pairwise longitudinal bias (1, H, 111, 111)
        pair_bias = self.pair_bias_mlp(self.seq_pair_geom).permute(2, 0, 1).unsqueeze(0)
        # Keep all 111 sequence tokens valid during self-attention when event is non-empty
        # so conv+attn can propagate across inactive planes; mask completely empty events cleanly.
        any_cal = seq_act.any(dim=1, keepdim=True)  # (B, 1)
        seq_mask = any_cal.expand(b, self.n_seq)
        for blk in self.long_blocks:
            seq = blk['conv'](seq)
            seq = blk['attn'](seq, seq, seq_mask, pair_bias=pair_bias)

        # Pure-observation window readouts
        idx_arange = torch.arange(self.n_seq, device=device)[None, :]
        # First active index (default 0)
        first_scores = torch.where(seq_act, idx_arange, torch.full_like(idx_arange, self.n_seq))
        idx_start = first_scores.min(dim=1, keepdim=True).values.clamp(max=self.n_seq - 1)
        idx_peak = seq_energy.argmax(dim=1, keepdim=True)

        mask_start = (idx_arange >= idx_start) & (idx_arange < idx_start + 8) & any_cal
        mask_peak = (idx_arange >= idx_peak - 4) & (idx_arange <= idx_peak + 4) & any_cal
        mask_tail = (idx_arange >= self.n_seq - 12) & any_cal
        mask_global = any_cal.expand(b, self.n_seq)

        cal_feat = torch.cat([
            self.win_start(seq, mask_start),
            self.win_peak(seq, mask_peak),
            self.win_tail(seq, mask_tail),
            self.win_global(seq, mask_global),
        ], dim=-1)  # (B, 512)

        spec_feat = self.spectrometer(
            batch['trk_cand'], batch['trk_cand_mask'], batch['trk_hits'], batch['trk_hit_mask']
        )
        cobs_feat = self.cobs_emb(batch['c_obs'])

        fused = torch.cat([cal_feat, stereo_2d_feat, spec_feat, cobs_feat], dim=-1)
        with torch.autocast(device_type=fused.device.type, enabled=False):
            return self.head(fused.float())


def b1pp_branch_groups(model: ZAlignedStereoNet) -> dict[str, list[tuple[str, nn.Parameter]]]:
    """Group parameters by physical branch for gradient/update health inspection."""
    groups: dict[str, list[tuple[str, nn.Parameter]]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(('xbar_enc.', 'ybar_enc.')):
            branch = 'bar_transverse'
        elif name.startswith(('pfine_x_enc.', 'pfine_y_enc.', 'pwide_x_enc.', 'pwide_y_enc.', 'pix_fuse.')):
            branch = 'pixel_transverse'
        elif name.startswith(('geom_emb.', 'mod_emb.', 'active_emb.', 'pair_bias_mlp.', 'long_blocks.')):
            branch = 'longitudinal_mixer'
        elif name.startswith(('win_start.', 'win_peak.', 'win_tail.', 'win_global.', 'stereo_2d_proj.')):
            branch = 'window_readout'
        elif name.startswith('spectrometer.'):
            branch = 'spectrometer'
        elif name.startswith('cobs_emb.'):
            branch = 'cobs'
        else:
            branch = 'head'
        groups.setdefault(branch, []).append((name, param))
    return groups
