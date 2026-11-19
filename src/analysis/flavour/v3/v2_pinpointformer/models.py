"""PinpointFormer V2 architecture: micron pixel clusters, stereo bars, vertex anchor, Perceiver."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from analysis.flavour.v2.r1.tasks import output_size
from analysis.flavour.v3.common.domains import cumulative_radiation_length, detector_bounds
from analysis.flavour.v3.common.fast_kernels import MICRO_VTX_FEAT_DIM
from analysis.flavour.v3.common.layers import split_adamw_param_groups
from analysis.flavour.v3.v1_b1pp.models import SpectrometerEncoder
from analysis.flavour.v3.v2_pinpointformer.attention import (
    AttentionStack,
    PerceiverCrossBlock,
    collect_attention_scales,
    collect_effective_attention_sharpness,
)
from analysis.flavour.v3.v2_pinpointformer.pixel_tokenizer import (
    MicroPatchConv2D,
    PIX_FEAT_DIM,
    PIX_PLANE_SUMMARY_DIM,
)
from analysis.flavour.v3.v2_pinpointformer.stereo_bars import StereoBarNexus, static_scint_stations
from analysis.flavour.v3.v2_pinpointformer.vertex_anchor import (
    K_VTX_MAX,
    PAIR_FEAT_DIM,
    VertexAnchorPredictor,
    compute_near_vertex_pair_bias,
    compute_stage2_polar_features,
    select_near_vertex_clusters,
)


class PinpointFormer(nn.Module):
    """Two-stage vertex-anchored transformer over pixel clusters, 3D stereo bars, and spectrometer."""

    def __init__(self, geometry: dict, config: dict) -> None:
        super().__init__()
        self.config = config
        bounds = detector_bounds(geometry)
        stations = static_scint_stations(geometry, bounds)
        dim = int(config.get('dim', 192))
        heads = int(config.get('heads', 8))
        drop_path = float(config.get('drop_path', 0.10))
        n_local_layers = int(config.get('n_local_layers', 4))
        n_global_layers = int(config.get('n_global_layers', 4))
        self.dim = dim
        self.heads = heads
        self.use_vertex_anchor = bool(config.get('use_vertex_anchor', True))
        self.enable_micro_vertex = bool(config.get('enable_micro_vertex', False))
        self.n_queries = int(config.get('n_queries', 16))
        self.k_vtx = int(config.get('k_vtx', K_VTX_MAX))

        pix_x0 = torch.tensor(
            [cumulative_radiation_length(float(z), geometry) for z in bounds['pixel_Zpos']],
            dtype=torch.float32,
        )
        self.register_buffer('pix_plane_x0', pix_x0)
        self.register_buffer(
            'default_center_mm',
            torch.tensor(
                [bounds['x_pix_center'], bounds['y_scint_center'], bounds['z_mid']],
                dtype=torch.float32,
            ),
        )

        # Stage-1 unimodal & stereo encoders
        self.patch_conv = MicroPatchConv2D(out_dim=64)
        self.patch_proj = nn.Sequential(
            nn.Linear(64, dim),
            nn.LayerNorm(dim),
        )
        self.pix_in = nn.Sequential(
            nn.Linear(PIX_FEAT_DIM, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.pix_plane_emb = nn.Sequential(
            nn.Linear(PIX_PLANE_SUMMARY_DIM, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
        )
        self.stereo_bars = StereoBarNexus(stations, dim=dim, heads=heads, drop_path=drop_path)
        self.spectrometer = SpectrometerEncoder(out_dim=dim)

        # Longitudinal z-ordered pixel + calorimeter plane encoder (69 physical planes)
        n_pix_p = len(bounds['pixel_Zpos'])
        n_st = int(stations['n_stations'])
        all_z = np.concatenate([
            np.asarray(bounds['pixel_Zpos'], dtype=np.float64),
            np.asarray(stations['station_z_mm'], dtype=np.float64),
        ])
        all_x0 = np.concatenate([
            pix_x0.numpy().astype(np.float64),
            np.asarray(stations['station_x0'], dtype=np.float64),
        ])
        is_pix = np.r_[np.ones(n_pix_p, dtype=np.float32), np.zeros(n_st, dtype=np.float32)]
        z_order = np.argsort(all_z, kind='mergesort')
        z_geom = np.column_stack([
            (all_z[z_order] - float(bounds['z_mid'])) / 1000.0,
            all_x0[z_order] / 50.0,
            is_pix[z_order],
            1.0 - is_pix[z_order],
        ]).astype(np.float32)
        self.n_st = n_st
        self.register_buffer('z_order_idx', torch.from_numpy(z_order).long())
        self.register_buffer('z_plane_geom', torch.from_numpy(z_geom))
        self.z_pos_proj = nn.Linear(4, dim)
        self.z_profile_encoder = AttentionStack(
            dim=dim,
            n_layers=2,
            heads=heads,
            init_scale=10.0,
            max_scale=50.0,
            drop_path=drop_path,
        )
        self.z_pool_gate = nn.Linear(dim, 1)

        # Stage-1 -> Stage-2 Differentiable Vertex Anchor + Polar Geometry
        if self.use_vertex_anchor:
            self.vertex_anchor = VertexAnchorPredictor(dim=dim, heads=heads)
            self.polar_gate = nn.Parameter(torch.tensor(0.15))
            self.stage2_polar_proj = nn.Sequential(
                nn.Linear(10, dim),
                nn.LayerNorm(dim),
                nn.SiLU(),
            )
            self.pair_bias_mlp = nn.Sequential(
                nn.Linear(PAIR_FEAT_DIM, 32),
                nn.SiLU(),
                nn.Linear(32, heads),
            )
        else:
            self.vertex_anchor = None
            self.polar_gate = None
            self.stage2_polar_proj = None
            self.pair_bias_mlp = None

        self.local_encoder = AttentionStack(
            dim=dim,
            n_layers=n_local_layers,
            heads=heads,
            init_scale=12.0,
            max_scale=50.0,
            drop_path=drop_path,
        )
        self.local_scatter_scale = nn.Parameter(torch.full((dim,), 0.5))
        self.vtx_pool_gate = nn.Linear(dim, 1)

        # Global heterogeneous Perceiver fusion (no linear readout shortcut)
        self.latent_queries = nn.Parameter(torch.randn(1, self.n_queries, dim) * 0.02)
        self.cobs_query = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(16, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
        )
        self.global_blocks = nn.ModuleList([
            PerceiverCrossBlock(
                dim=dim,
                heads=heads,
                init_scale=12.0,
                max_scale=50.0,
                drop_path=drop_path,
            )
            for _ in range(n_global_layers)
        ])

        self.pool_gate = nn.Linear(dim, 1)
        self.global_fusion = nn.Sequential(
            nn.LayerNorm(6 * dim),
            nn.Linear(6 * dim, 3 * dim),
            nn.SiLU(),
        )
        out_dim = int(config.get('num_classes', output_size(config.get('task', 'T4'))))
        if self.enable_micro_vertex:
            self.micro_vtx_token: nn.Module | None = nn.Sequential(
                nn.LayerNorm(MICRO_VTX_FEAT_DIM),
                nn.Linear(MICRO_VTX_FEAT_DIM, dim),
                nn.LayerNorm(dim),
                nn.SiLU(),
            )
            self.micro_vtx_proj: nn.Sequential | None = nn.Sequential(
                nn.LayerNorm(MICRO_VTX_FEAT_DIM),
                nn.Linear(MICRO_VTX_FEAT_DIM, 64),
                nn.SiLU(),
                nn.Linear(64, 3 * dim),
            )
            last_lin = self.micro_vtx_proj[-1]
            if isinstance(last_lin, nn.Linear):
                nn.init.normal_(last_lin.weight, mean=0.0, std=0.02)
                nn.init.zeros_(last_lin.bias)
            # Small-gate initialization (0.02): preserves clean backbone stability while keeping step-1 proj gradients active
            self.micro_vtx_gate: nn.Parameter | None = nn.Parameter(torch.tensor(0.02, dtype=torch.float32))
        else:
            self.micro_vtx_token = None
            self.micro_vtx_proj = None
            self.micro_vtx_gate = None

        self.head = nn.Sequential(
            nn.LayerNorm(3 * dim),
            nn.Linear(3 * dim, 256),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(256, out_dim),
        )

    def build_param_groups(self, base_lr: float, weight_decay: float) -> list[dict]:
        return split_adamw_param_groups(self, base_lr, weight_decay)

    def learned_attention_scales(self) -> dict[str, list[float]]:
        return collect_attention_scales(self)

    def effective_attention_sharpness(self) -> dict[str, list[dict[str, float]]]:
        return collect_effective_attention_sharpness(self)

    def _compute_v_prior(self, batch: dict, b: int, device: torch.device) -> torch.Tensor:
        center = self.default_center_mm[None, :].expand(b, -1).to(device)
        cal_start = batch['cal_start'].float()  # (B, 4): [z, x, y, valid]
        cal_valid = (cal_start[:, 3:4] > 0.5)
        cal_xyz = torch.stack([cal_start[:, 1], cal_start[:, 2], cal_start[:, 0]], dim=-1)
        prior = torch.where(cal_valid, cal_xyz, center)
        if self.enable_micro_vertex and 'avr_pv_mm' in batch:
            avr = batch['avr_pv_mm'].float()  # (B, 4): [x, y, z, valid]
            # Only adopt AVR prior when longitudinally consistent with cal_start (rejects +500 mm downstream IPT)
            dz_cal = avr[:, 2:3] - cal_start[:, 0:1]
            z_consistent = (~cal_valid) | ((dz_cal >= -50.0) & (dz_cal <= 150.0))
            avr_valid = (avr[:, 3:4] > 0.5) & z_consistent
            prior = torch.where(avr_valid, avr[:, :3], prior)
        return prior

    def forward_features(
        self,
        batch: dict,
        grad_blend: float = 1.0,
        return_entropy: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | None]:
        b = int(batch['c_obs'].shape[0])
        device = batch['c_obs'].device

        # 1. Pixel cluster tokens + 32x32 micro-patch morphological features
        pix_feat = batch['pix_feat']
        pix_mask = batch['pix_mask']
        pix_xyz = batch['pix_xyz_mm'].float()
        u_pix = self.pix_in(pix_feat) * pix_mask.unsqueeze(-1).to(pix_feat.dtype)

        if len(batch['pix_patches']):
            patch_emb = self.patch_proj(self.patch_conv(batch['pix_patches']))
            delta_patch = torch.zeros_like(u_pix, dtype=patch_emb.dtype)
            delta_patch.index_put_((batch['patch_ev'], batch['patch_cl']), patch_emb, accumulate=True)
            u_pix = u_pix + delta_patch * pix_mask.unsqueeze(-1).to(u_pix.dtype)

        pix_plane_tok = self.pix_plane_emb(batch['pix_plane_summary'])
        pix_plane_act = batch['pix_plane_summary'][..., 0] > 0.5

        # 2. Stereo calorimeter nexus & raw active bar embeddings
        bar_out = self.stereo_bars(batch)
        cal_tok = bar_out['cal_tokens']
        cal_xyz = bar_out['cal_xyz_mm']
        cal_meas_xy = bar_out['cal_measured_xy']
        cal_mask = bar_out['cal_mask']

        # 3. Magnetic spectrometer kinematics + raw hits
        spec_tok = self.spectrometer(
            batch['trk_cand'], batch['trk_cand_mask'], batch['trk_hits'], batch['trk_hit_mask']
        )

        # 4. Stage-1 Vertex Prediction -> Stage-2 Polar Recentering & IP/kink Pair Bias
        pix_x0 = self.pix_plane_x0[batch['pix_plane']]
        v_prior = self._compute_v_prior(batch, b, device)
        if self.use_vertex_anchor:
            assert (
                self.vertex_anchor is not None
                and self.stage2_polar_proj is not None
                and self.polar_gate is not None
            )
            assert self.pair_bias_mlp is not None
            anchor_src = torch.cat([pix_plane_tok, cal_tok], dim=1)
            anchor_mask = torch.cat([pix_plane_act, cal_mask], dim=1)
            v_hat, log_var_v = self.vertex_anchor(
                anchor_src,
                anchor_mask,
                v_prior,
                pix_tokens=u_pix,
                pix_xyz_mm=pix_xyz,
                pix_mask=pix_mask,
                pix_plane=batch['pix_plane'],
            )
            alpha = float(max(0.0, min(1.0, grad_blend)))
            v_0 = alpha * v_hat + (1.0 - alpha) * v_hat.detach()
            lv_0 = alpha * log_var_v + (1.0 - alpha) * log_var_v.detach()

            pix_meas = pix_mask.unsqueeze(-1).expand(-1, -1, 2).float()
            pix_polar = compute_stage2_polar_features(pix_xyz, pix_meas, v_0, log_var_v=lv_0)
            u_pix = u_pix + (self.polar_gate * self.stage2_polar_proj(
                torch.cat([pix_polar, pix_meas], dim=-1)
            )) * pix_mask.unsqueeze(-1).to(u_pix.dtype)

            cal_polar = compute_stage2_polar_features(cal_xyz, cal_meas_xy, v_0, log_var_v=lv_0)
            cal_tok = cal_tok + (self.polar_gate * self.stage2_polar_proj(
                torch.cat([cal_polar, cal_meas_xy], dim=-1)
            )) * cal_mask.unsqueeze(-1).to(cal_tok.dtype)
        else:
            v_0 = self.default_center_mm[None, :].expand(b, -1).to(device)
            v_hat = v_0
            log_var_v = v_0.new_zeros((b, 3))
            lv_0 = log_var_v

        vtx_idx, vtx_mask = select_near_vertex_clusters(
            pix_xyz, pix_mask, v_0, k_vtx=self.k_vtx, pix_plane=batch['pix_plane']
        )
        vtx_xyz = torch.gather(pix_xyz, 1, vtx_idx.unsqueeze(-1).expand(-1, -1, 3))
        vtx_x0 = torch.gather(pix_x0, 1, vtx_idx)
        u_vtx = torch.gather(u_pix, 1, vtx_idx.unsqueeze(-1).expand(-1, -1, self.dim))

        if self.use_vertex_anchor:
            assert self.pair_bias_mlp is not None
            vtx_pair_bias = compute_near_vertex_pair_bias(
                self.pair_bias_mlp, vtx_xyz, vtx_x0, vtx_mask, v_0, log_var_v=lv_0
            )
        else:
            vtx_pair_bias = None

        entropies: list[torch.Tensor] = []
        if return_entropy:
            u_vtx, local_ents = self.local_encoder(
                u_vtx, vtx_mask, pair_bias=vtx_pair_bias, return_entropy=True
            )
            entropies.extend(local_ents)
        else:
            u_vtx = self.local_encoder(u_vtx, vtx_mask, pair_bias=vtx_pair_bias)

        # Scatter local neighborhood representation back into full cluster tensor
        u_vtx_scaled = (u_vtx * vtx_mask.unsqueeze(-1).to(u_vtx.dtype)) * self.local_scatter_scale
        u_pix = u_pix.scatter_add(
            1,
            vtx_idx.unsqueeze(-1).expand(-1, -1, self.dim),
            u_vtx_scaled.to(u_pix.dtype),
        )

        # 5. Longitudinal 69-plane profile encoder (interleaved pixel + stereo scint stations)
        z_in = torch.cat([pix_plane_tok, cal_tok[:, :self.n_st]], dim=1)[:, self.z_order_idx]
        z_act = torch.cat([pix_plane_act, cal_mask[:, :self.n_st]], dim=1)[:, self.z_order_idx]
        z_in = (z_in + self.z_pos_proj(self.z_plane_geom).unsqueeze(0).to(z_in.dtype)) * z_act.unsqueeze(-1).to(z_in.dtype)
        z_seq = self.z_profile_encoder(z_in, z_act)

        # 6. Heterogeneous Perceiver Fusion over all modalities + c_obs conditional query
        spec_mask = batch['trk_hit_mask'].any(dim=1, keepdim=True)
        if self.enable_micro_vertex and self.micro_vtx_token is not None and 'micro_vtx_feat' in batch:
            mv_feat = batch['micro_vtx_feat'].float()
            mv_tok = self.micro_vtx_token(mv_feat).unsqueeze(1).to(u_pix.dtype)
            mv_mask = (mv_feat[:, 0:1] > 0.5)
            source = torch.cat([u_pix, pix_plane_tok, cal_tok, spec_tok.unsqueeze(1), mv_tok], dim=1)
            source_mask = torch.cat([pix_mask, pix_plane_act, cal_mask, spec_mask, mv_mask], dim=1)
        else:
            source = torch.cat([u_pix, pix_plane_tok, cal_tok, spec_tok.unsqueeze(1)], dim=1)
            source_mask = torch.cat([pix_mask, pix_plane_act, cal_mask, spec_mask], dim=1)

        cobs_q = self.cobs_query(batch['c_obs']).unsqueeze(1)
        latents = torch.cat([self.latent_queries.expand(b, -1, -1), cobs_q], dim=1)
        latent_mask = torch.ones((b, self.n_queries + 1), dtype=torch.bool, device=device)

        for blk in self.global_blocks:
            if return_entropy:
                latents, ent = blk(latents, source, source_mask, latent_mask, return_entropy=True)
                entropies.append(ent)
            else:
                latents = blk(latents, source, source_mask, latent_mask)

        # Equal-footing multi-branch fusion (prevents pixel-token dilution of bar/spec readouts)
        q_lat = latents[:, :self.n_queries]
        q_cobs = latents[:, self.n_queries]
        with torch.autocast(device_type=q_lat.device.type, enabled=False):
            w = torch.softmax(self.pool_gate(q_lat.float()), dim=1)
            pool_attn = (w * q_lat.float()).sum(dim=1)

            z_logits = self.z_pool_gate(z_seq.float()).squeeze(-1).masked_fill(~z_act, -1e4)
            z_w = torch.softmax(z_logits, dim=1) * z_act.float()
            z_w = z_w / z_w.sum(dim=1, keepdim=True).clamp_min(1e-6)
            z_prof_glob = (z_w.unsqueeze(-1) * z_seq.float()).sum(dim=1)

            vtx_logits = self.vtx_pool_gate(u_vtx.float()).squeeze(-1).masked_fill(~vtx_mask, -1e4)
            vtx_w = torch.softmax(vtx_logits, dim=1) * vtx_mask.float()
            vtx_w = vtx_w / vtx_w.sum(dim=1, keepdim=True).clamp_min(1e-6)
            vtx_glob = (vtx_w.unsqueeze(-1) * u_vtx.float()).sum(dim=1)

            fused_global = self.global_fusion(
                torch.cat(
                    [
                        pool_attn,
                        z_prof_glob,
                        bar_out['cal_global'].float(),
                        vtx_glob,
                        spec_tok.float(),
                        q_cobs.float(),
                    ],
                    dim=-1,
                )
            )
            if (
                self.enable_micro_vertex
                and self.micro_vtx_proj is not None
                and self.micro_vtx_gate is not None
                and 'micro_vtx_feat' in batch
            ):
                mv_f = batch['micro_vtx_feat'].float()
                fused_global = fused_global + self.micro_vtx_gate * self.micro_vtx_proj(mv_f)

        return {
            'fused_global': fused_global,
            'cal_global': bar_out['cal_global'],
            'v_hat': v_hat,
            'log_var_v': log_var_v,
            'u_pix': u_pix,
            'u_vtx': u_vtx,
            'pix_xyz_mm': pix_xyz,
            'pix_mask': pix_mask,
            'vtx_idx': vtx_idx,
            'vtx_xyz_mm': vtx_xyz,
            'vtx_mask': vtx_mask,
            'active_bar_emb': bar_out['active_bar_emb'],
            'entropies': entropies,
        }

    def forward(
        self,
        batch: dict,
        grad_blend: float = 1.0,
        return_aux: bool = False,
        return_entropy: bool = False,
    ) -> torch.Tensor | dict:
        aux = self.forward_features(batch, grad_blend=grad_blend, return_entropy=return_entropy)
        with torch.autocast(device_type=aux['fused_global'].device.type, enabled=False):
            logits = self.head(aux['fused_global'].float())
        if not return_aux:
            return logits
        aux['logits'] = logits
        return aux


def pf_branch_groups(model: PinpointFormer) -> dict[str, list[tuple[str, nn.Parameter]]]:
    """Group parameters by physical subsystem for gradient/update health inspection."""
    groups: dict[str, list[tuple[str, nn.Parameter]]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('patch_'):
            branch = 'pixel_patch_conv'
        elif name.startswith(('pix_in.', 'pix_plane_emb.')):
            branch = 'pixel_clusters'
        elif name.startswith('stereo_bars.'):
            branch = 'stereo_bars'
        elif name.startswith('spectrometer.'):
            branch = 'spectrometer'
        elif name.startswith(('vertex_anchor.', 'stage2_polar_proj.', 'pair_bias_mlp.', 'polar_gate')):
            branch = 'vertex_anchor'
        elif name.startswith(('local_encoder.', 'local_scatter_scale', 'vtx_pool_gate.')):
            branch = 'local_vertex_attn'
        elif name.startswith('micro_vtx_'):
            branch = 'micro_vertex_reco'
        elif name.startswith((
            'latent_queries',
            'cobs_query.',
            'global_blocks.',
            'pool_gate.',
            'z_pos_proj.',
            'z_profile_encoder.',
            'z_pool_gate.',
            'global_fusion.',
        )):
            branch = 'global_perceiver'
        else:
            branch = 'head'
        groups.setdefault(branch, []).append((name, param))
    return groups
