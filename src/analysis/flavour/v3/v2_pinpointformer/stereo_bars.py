"""NuGraph2-style stereo scintillator bar nexus eliminating orthogonal ghost crossings."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from analysis.flavour.v3.common.domains import cumulative_radiation_length, detector_bounds
from analysis.flavour.v3.common.layers import ScaledQKNormAttention
from analysis.flavour.v3.v1_b1pp.models import TransverseLayerEncoder


N_MAX_NEXUS = 96
MAX_CROSS_PER_SIDE = 8


def static_scint_stations(geometry: dict, bounds: dict | None = None) -> dict:
    """Group 96 scintillator panels into 54 physical z-sorted stations (42 XY doublets + 12 X-only)."""
    if bounds is None:
        bounds = detector_bounds(geometry)
    sz = np.asarray(bounds['scint_Zpos'], dtype=np.float64)
    vp = np.asarray(bounds['vert_panels'], dtype=np.int64)
    hp = np.asarray(bounds['horiz_panels'], dtype=np.int64)

    # Sort vertical panels strictly by physical z (all 54 stations have a vertical X panel)
    v_order = np.argsort(sz[vp], kind='mergesort')
    vp_sorted = vp[v_order]
    n_st = len(vp_sorted)

    h_order = np.argsort(sz[hp], kind='mergesort')
    hp_sorted = hp[h_order]
    v_z = sz[vp_sorted]

    station_has_y = np.zeros(n_st, dtype=bool)
    station_h_panel = np.full(n_st, -1, dtype=np.int64)
    station_z = v_z.copy()

    for p_h in hp_sorted:
        diffs = np.abs(v_z - sz[p_h])
        s = int(np.argmin(diffs))
        if diffs[s] > 15.0 or station_has_y[s]:
            raise ValueError(f'unexpected horizontal panel pairing failure for panel {int(p_h)}')
        station_has_y[s] = True
        station_h_panel[s] = int(p_h)
        station_z[s] = 0.5 * (v_z[s] + sz[p_h])

    doublet_stations = np.flatnonzero(station_has_y)
    station_to_doublet = np.full(n_st, -1, dtype=np.int64)
    station_to_doublet[doublet_stations] = np.arange(len(doublet_stations), dtype=np.int64)

    panel_to_station = np.full(len(sz), -1, dtype=np.int64)
    panel_to_xslot = np.full(len(sz), -1, dtype=np.int64)
    panel_to_yslot = np.full(len(sz), -1, dtype=np.int64)
    for s in range(n_st):
        pv = int(vp_sorted[s])
        panel_to_station[pv] = s
        panel_to_xslot[pv] = s
        ph = int(station_h_panel[s])
        if ph >= 0:
            panel_to_station[ph] = s
            panel_to_yslot[ph] = int(station_to_doublet[s])

    station_x0 = np.array(
        [cumulative_radiation_length(float(z), geometry) for z in station_z],
        dtype=np.float32,
    )
    z_mid = float(bounds['z_mid'])
    return {
        'n_stations': n_st,
        'n_doublets': len(doublet_stations),
        'n_bars': int(bounds['n_bars']),
        'x_scint_center': float(bounds['x_scint_center']),
        'y_scint_center': float(bounds['y_scint_center']),
        'z_mid': z_mid,
        'bar_x_centers_mm': np.asarray(bounds['scint_bar_Xcenter'], dtype=np.float32),
        'bar_y_centers_mm': np.asarray(bounds['scint_bar_Ycenter'], dtype=np.float32),
        'station_z_mm': station_z.astype(np.float32),
        'station_x0': station_x0,
        'station_has_y': station_has_y,
        'doublet_stations': doublet_stations,
        'panel_to_station': panel_to_station,
        'panel_to_xslot': panel_to_xslot,
        'panel_to_yslot': panel_to_yslot,
    }


class StereoBarNexus(nn.Module):
    """Dual X/Y 2D bar convs + bipartite crossing disambiguation into 3D calorimeter tokens."""

    def __init__(self, stations: dict, dim: int = 192, heads: int = 8, drop_path: float = 0.08) -> None:
        super().__init__()
        self.dim = dim
        self.n_st = int(stations['n_stations'])
        self.n_dbl = int(stations['n_doublets'])
        self.n_bars = int(stations['n_bars'])
        self.x_c = float(stations['x_scint_center'])
        self.y_c = float(stations['y_scint_center'])
        self.z_mid = float(stations['z_mid'])

        self.register_buffer('station_z_mm', torch.from_numpy(stations['station_z_mm']))
        self.register_buffer('station_x0', torch.from_numpy(stations['station_x0']))
        self.register_buffer('station_has_y', torch.from_numpy(stations['station_has_y']))
        self.register_buffer('doublet_stations', torch.from_numpy(stations['doublet_stations']))
        self.register_buffer('panel_to_station', torch.from_numpy(stations['panel_to_station']))
        self.register_buffer('panel_to_xslot', torch.from_numpy(stations['panel_to_xslot']))
        self.register_buffer('panel_to_yslot', torch.from_numpy(stations['panel_to_yslot']))
        self.register_buffer('bar_x_mm', torch.from_numpy(stations['bar_x_centers_mm']))
        self.register_buffer('bar_y_mm', torch.from_numpy(stations['bar_y_centers_mm']))
        self.register_buffer('u_bars', torch.linspace(-1.0, 1.0, self.n_bars))

        z_norm = (self.station_z_mm - self.z_mid) / 1000.0
        x0_norm = self.station_x0 / 50.0
        self.register_buffer('xbar_geom', torch.stack([z_norm, x0_norm], dim=0))  # (2, 54)
        self.register_buffer('ybar_geom', torch.stack([
            z_norm[self.doublet_stations],
            x0_norm[self.doublet_stations],
        ], dim=0))  # (2, 42)

        self.xbar_enc = TransverseLayerEncoder(6, (32, 48, 64), stem_k=3, stem_s=1, out_dim=dim)
        self.ybar_enc = TransverseLayerEncoder(6, (32, 48, 64), stem_k=3, stem_s=1, out_dim=dim)
        self.stereo_2d_proj = nn.Sequential(
            nn.Linear(256, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
        )
        self.station_fuse = nn.Sequential(
            nn.Linear(2 * dim + 5, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
        )
        self.nexus_in = nn.Sequential(
            nn.Linear(7, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
        )
        self.pair_mlp = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, heads))
        self.bipartite_attn = nn.ModuleList([
            ScaledQKNormAttention(dim, heads=heads, init_scale=12.0, drop_path=drop_path)
            for _ in range(2)
        ])

        # Per-active-bar embedding head aligned 1-to-1 with raw observation bars
        self.bar_node_proj = nn.Sequential(
            nn.Linear(dim + 6, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
        )

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        b = len(batch['label'])
        device = batch['c_obs'].device
        bar_ev = batch['bar_ev']
        bar_panel = batch['bar_panel']
        bar_horiz = batch['bar_horiz']
        bar_idx = batch['bar_idx']
        bar_val = batch['bar_val']

        grid_x = torch.zeros((b, 6, self.n_st, self.n_bars), device=device, dtype=torch.float32)
        grid_y = torch.zeros((b, 6, self.n_dbl, self.n_bars), device=device, dtype=torch.float32)

        x_mask_b = (bar_horiz == 0)
        y_mask_b = (bar_horiz == 1)
        if x_mask_b.any():
            ev_x = bar_ev[x_mask_b]
            xs = self.panel_to_xslot[bar_panel[x_mask_b]]
            bx = bar_idx[x_mask_b]
            vx = bar_val[x_mask_b]
            grid_x[ev_x, 0, xs, bx] = vx[:, 0]
            grid_x[ev_x, 1, xs, bx] = vx[:, 1]
            grid_x[ev_x, 4, xs, bx] = 1.0
        if y_mask_b.any():
            ev_y = bar_ev[y_mask_b]
            ys = self.panel_to_yslot[bar_panel[y_mask_b]]
            by = bar_idx[y_mask_b]
            vy = bar_val[y_mask_b]
            grid_y[ev_y, 0, ys, by] = vy[:, 0]
            grid_y[ev_y, 1, ys, by] = vy[:, 1]
            grid_y[ev_y, 4, ys, by] = 1.0

        grid_x[:, 2:4] = self.xbar_geom[None, :, :, None]
        grid_x[:, 5] = grid_x[:, 4] * self.u_bars[None, None, :]
        grid_y[:, 2:4] = self.ybar_geom[None, :, :, None]
        grid_y[:, 5] = grid_y[:, 4] * self.u_bars[None, None, :]

        x_act = grid_x[:, 4].any(dim=-1)  # (B, 54)
        y_act = grid_y[:, 4].any(dim=-1)  # (B, 42)
        x_tok, x_glob = self.xbar_enc(grid_x, x_act)  # (B, 54, D), (B, 128)
        y_tok, y_glob = self.ybar_enc(grid_y, y_act)  # (B, 42, D), (B, 128)
        cal_glob = self.stereo_2d_proj(torch.cat([x_glob, y_glob], dim=-1))

        # Lift Y tokens into 54-station frame
        y_full = torch.zeros_like(x_tok)
        y_act_full = torch.zeros_like(x_act)
        y_ex = grid_y[:, 0]  # (B, 42, 40)
        y_has = grid_y[:, 4]  # (B, 42, 40)
        y_ex_full = torch.zeros((b, self.n_st, self.n_bars), device=device)
        y_has_full = torch.zeros((b, self.n_st, self.n_bars), device=device)
        y_full[:, self.doublet_stations] = y_tok
        y_act_full[:, self.doublet_stations] = y_act
        y_ex_full[:, self.doublet_stations] = y_ex
        y_has_full[:, self.doublet_stations] = y_has

        e_x = grid_x[:, 0].sum(dim=-1)
        e_y = y_ex_full.sum(dim=-1)
        w_x = grid_x[:, 0] + 1e-4 * grid_x[:, 4]
        w_y = y_ex_full + 1e-4 * y_has_full
        x_cent = (w_x * self.bar_x_mm[None, None, :]).sum(dim=-1) / w_x.sum(dim=-1).clamp_min(1e-6)
        y_cent = (w_y * self.bar_y_mm[None, None, :]).sum(dim=-1) / w_y.sum(dim=-1).clamp_min(1e-6)
        x_cent = torch.where(x_act, x_cent, torch.full_like(x_cent, self.x_c))
        y_cent = torch.where(y_act_full, y_cent, torch.full_like(y_cent, self.y_c))

        asym = (e_x - e_y) / (e_x + e_y + 1e-4) * (x_act & y_act_full).float()
        st_aux = torch.stack([
            e_x,
            e_y,
            asym,
            x_act.float(),
            y_act_full.float(),
        ], dim=-1)
        st_tok = self.station_fuse(torch.cat([x_tok, y_full, st_aux], dim=-1))
        st_mask = x_act | y_act_full
        st_tok = st_tok * st_mask.unsqueeze(-1).to(st_tok.dtype)

        st_xyz = torch.stack([
            x_cent,
            y_cent,
            self.station_z_mm[None, :].expand(b, -1),
        ], dim=-1).float()
        st_meas_xy = torch.stack([x_act.float(), y_act_full.float()], dim=-1)

        # Build top-N_MAX_NEXUS stereo (X, Y) crossing nodes across doublet layers (FP32 geometry)
        ex_dbl = grid_x[:, 0, self.doublet_stations]  # (B, 42, 40)
        act_x_dbl = grid_x[:, 4, self.doublet_stations] > 0.5
        ey_dbl = y_ex
        act_y_dbl = y_has > 0.5
        score_x = ex_dbl + 1e-3 * act_x_dbl.float()
        score_y = ey_dbl + 1e-3 * act_y_dbl.float()
        top_ex, top_ix = torch.topk(score_x, k=MAX_CROSS_PER_SIDE, dim=-1)  # (B, 42, 8)
        top_ey, top_iy = torch.topk(score_y, k=MAX_CROSS_PER_SIDE, dim=-1)  # (B, 42, 8)
        top_vx = torch.gather(act_x_dbl, dim=-1, index=top_ix)
        top_vy = torch.gather(act_y_dbl, dim=-1, index=top_iy)

        # Pairwise outer grid inside each doublet (B, 42, 8, 8)
        pair_valid = top_vx[:, :, :, None] & top_vy[:, :, None, :]
        ex_p = top_ex[:, :, :, None].expand(-1, -1, -1, MAX_CROSS_PER_SIDE)
        ey_p = top_ey[:, :, None, :].expand(-1, -1, MAX_CROSS_PER_SIDE, -1)
        pair_score = torch.where(
            pair_valid,
            2.0 * torch.minimum(ex_p, ey_p) + 0.1 * (ex_p + ey_p),
            torch.full_like(ex_p, -1e6),
        ).view(b, -1)  # (B, 42 * 64)

        best_scores, best_flat = torch.topk(pair_score, k=N_MAX_NEXUS, dim=-1)
        nex_mask = best_scores > -1e5
        safe_flat = torch.where(nex_mask, best_flat, torch.zeros_like(best_flat))
        d_idx = safe_flat // (MAX_CROSS_PER_SIDE * MAX_CROSS_PER_SIDE)
        rem = safe_flat % (MAX_CROSS_PER_SIDE * MAX_CROSS_PER_SIDE)
        xi_slot = rem // MAX_CROSS_PER_SIDE
        yi_slot = rem % MAX_CROSS_PER_SIDE

        bx_id = torch.gather(top_ix.view(b, -1), 1, d_idx * MAX_CROSS_PER_SIDE + xi_slot)
        by_id = torch.gather(top_iy.view(b, -1), 1, d_idx * MAX_CROSS_PER_SIDE + yi_slot)
        ex_val = torch.gather(top_ex.view(b, -1), 1, d_idx * MAX_CROSS_PER_SIDE + xi_slot)
        ey_val = torch.gather(top_ey.view(b, -1), 1, d_idx * MAX_CROSS_PER_SIDE + yi_slot)

        st_idx_nex = self.doublet_stations[d_idx]
        nex_x = self.bar_x_mm[bx_id]
        nex_y = self.bar_y_mm[by_id]
        nex_z = self.station_z_mm[st_idx_nex]
        nex_x0 = self.station_x0[st_idx_nex]

        nex_feat = torch.stack([
            (nex_x - self.x_c) / 200.0,
            (nex_y - self.y_c) / 200.0,
            (nex_z - self.z_mid) / 1000.0,
            ex_val,
            ey_val,
            torch.abs(ex_val - ey_val),
            (2.0 * torch.minimum(ex_val, ey_val)) / (ex_val + ey_val + 1e-4),
        ], dim=-1)
        nex_tok = self.nexus_in(nex_feat) * nex_mask.unsqueeze(-1).float()
        nex_xyz = torch.stack([nex_x, nex_y, nex_z], dim=-1).float()
        nex_meas_xy = nex_mask.unsqueeze(-1).expand(-1, -1, 2).float()

        # Combined calorimeter token graph with bipartite longitudinal attention
        cal_tok = torch.cat([st_tok, nex_tok], dim=1)  # (B, 150, D)
        cal_xyz = torch.cat([st_xyz, nex_xyz], dim=1)
        cal_meas_xy = torch.cat([st_meas_xy, nex_meas_xy], dim=1)
        cal_mask = torch.cat([st_mask, nex_mask], dim=1)
        cal_z = cal_xyz[..., 2]
        cal_x0 = torch.cat([self.station_x0[None, :].expand(b, -1), nex_x0], dim=1)

        pair_geom = torch.stack([
            torch.abs(cal_z[:, :, None] - cal_z[:, None, :]) / 500.0,
            torch.abs(cal_x0[:, :, None] - cal_x0[:, None, :]) / 20.0,
        ], dim=-1)
        pair_bias = self.pair_mlp(pair_geom).permute(0, 3, 1, 2)  # (B, H, 150, 150)

        for attn in self.bipartite_attn:
            cal_tok = attn(cal_tok, cal_tok, cal_mask, pair_bias=pair_bias)
        cal_tok = cal_tok * cal_mask.unsqueeze(-1).to(cal_tok.dtype)

        # Gather per-bar embeddings aligned to raw input order (used by V3 dense bar_rel head)
        if len(bar_ev):
            b_st = self.panel_to_station[bar_panel]
            st_gathered = cal_tok[bar_ev, b_st]  # (N_bars_tot, D)
            is_h = bar_horiz.float()
            u_coord = torch.where(
                bar_horiz == 1,
                (self.bar_y_mm[bar_idx] - self.y_c) / 200.0,
                (self.bar_x_mm[bar_idx] - self.x_c) / 200.0,
            )
            b_feat = torch.stack([
                bar_val[:, 0],
                bar_val[:, 1],
                u_coord,
                (self.station_z_mm[b_st] - self.z_mid) / 1000.0,
                is_h,
                1.0 - is_h,
            ], dim=-1)
            active_bar_emb = self.bar_node_proj(torch.cat([st_gathered, b_feat], dim=-1))
        else:
            active_bar_emb = cal_tok.new_empty((0, 128))

        return {
            'cal_tokens': cal_tok,
            'cal_xyz_mm': cal_xyz,
            'cal_x0': cal_x0,
            'cal_measured_xy': cal_meas_xy,
            'cal_mask': cal_mask,
            'cal_global': cal_glob,
            'active_bar_emb': active_bar_emb,
        }
