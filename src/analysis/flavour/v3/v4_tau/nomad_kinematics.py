"""Vectorized on-GPU 10-dim observable NOMAD transverse kinematics (B1++ and PF compatible)."""

from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from analysis.flavour.v3.common.domains import detector_bounds
from analysis.flavour.v3.v1_b1pp.spectrometer import TRACKER_Z_MM
from analysis.flavour.v3.v2_pinpointformer.stereo_bars import static_scint_stations


NOMAD_FEAT_DIM = 10
SAGITTA_CONST_MEV_MM = 100705.0
NOMAD_FEATURE_NAMES = (
    'cos_dphi_l_had',
    'acoplanarity_norm',
    'pt_miss_frac',
    'qt_perp_thrust_log',
    'core_30mm_iso',
    'recoil_energy_frac',
    'log_eigen_ratio',
    'transverse_r90_norm',
    'spec_to_cal_log_ratio',
    'cos_dphi_spec_cal',
)


class NomadTransverseExtractor(nn.Module):
    """Extract Y-mirror-invariant 10-dim transverse NOMAD variables from stereo bars + spectrometer."""

    def __init__(
        self,
        geometry: dict,
        bounds: dict | None = None,
        *,
        phys_calibrated: bool = True,
        pe_per_mip_mev: float = 40.0,
    ) -> None:
        super().__init__()
        self.phys_calibrated = bool(phys_calibrated)
        self.pe_per_mip_mev = float(pe_per_mip_mev)
        if bounds is None:
            bounds = detector_bounds(geometry)
        stations = static_scint_stations(geometry, bounds)

        self.n_st = int(stations['n_stations'])
        self.n_dbl = int(stations['n_doublets'])
        self.n_bars = int(stations['n_bars'])
        self.x_scint_center = float(stations['x_scint_center'])
        self.y_scint_center = float(stations['y_scint_center'])
        self.x_pix_center = float(bounds['x_pix_center'])
        self.z_trk0 = float(TRACKER_Z_MM[0])

        vp = np.asarray(bounds['vert_panels'], dtype=np.int64)
        hp = np.asarray(bounds['horiz_panels'], dtype=np.int64)
        px_map = np.asarray(stations['panel_to_xslot'], dtype=np.int64)
        py_map = np.asarray(stations['panel_to_yslot'], dtype=np.int64)
        b1pp_xslot = px_map[vp]
        b1pp_yslot = py_map[hp]
        if (b1pp_xslot < 0).any() or (b1pp_yslot < 0).any():
            raise ValueError('invalid scintillator station mapping in NomadTransverseExtractor')

        dbl = np.asarray(stations['doublet_stations'], dtype=np.int64)
        st_z = np.asarray(stations['station_z_mm'], dtype=np.float32)

        self.register_buffer('doublet_stations', torch.from_numpy(dbl))
        self.register_buffer('dbl_z_mm', torch.from_numpy(st_z[dbl]))
        self.register_buffer('panel_to_xslot', torch.from_numpy(px_map))
        self.register_buffer('panel_to_yslot', torch.from_numpy(py_map))
        self.register_buffer('b1pp_xslot', torch.from_numpy(b1pp_xslot))
        self.register_buffer('b1pp_yslot', torch.from_numpy(b1pp_yslot))
        self.register_buffer('bar_x_mm', torch.from_numpy(np.asarray(stations['bar_x_centers_mm'], dtype=np.float32)))
        self.register_buffer('bar_y_mm', torch.from_numpy(np.asarray(stations['bar_y_centers_mm'], dtype=np.float32)))

    def _compute_from_grids(
        self,
        e_x: torch.Tensor,
        e_y: torch.Tensor,
        trk_cand: torch.Tensor,
        trk_cand_mask: torch.Tensor,
        obs_domain_id: torch.Tensor,
    ) -> torch.Tensor:
        """Compute purely observable 10-dim NOMAD transverse vector in FP32 from (e_x, e_y, trk_cand)."""
        device = e_x.device
        b = int(e_x.shape[0])
        if b == 0:
            return torch.empty((0, NOMAD_FEAT_DIM), dtype=torch.float32, device=device)

        with torch.amp.autocast(device_type=device.type, enabled=False):
            ex_f = e_x.float()
            ey_f = e_y.float()
            ex_d = ex_f[:, self.doublet_stations, :]  # (B, n_dbl, n_bars)
            ey_d = ey_f  # (B, n_dbl, n_bars)

            ex_st = ex_d.sum(dim=-1)  # (B, n_dbl)
            ey_st = ey_d.sum(dim=-1)  # (B, n_dbl)
            e_st = 0.5 * (ex_st + ey_st)  # (B, n_dbl)

            # Factorized 3D crossing energy weights in MeV
            w3d = (ex_d[:, :, :, None] * ey_d[:, :, None, :]) / e_st.clamp_min(1e-6)[:, :, None, None]
            w_st = w3d.sum(dim=(2, 3))  # (B, n_dbl)
            e_stereo = w_st.sum(dim=-1)  # (B,)
            e_cal_tot = 0.5 * (ex_f.sum(dim=(1, 2)) + ey_f.sum(dim=(1, 2)))  # (B,)
            has_stereo = e_stereo > 1e-4

            # Observable upstream apex across active doublet window [d0, d0 + 1]
            d_grid = torch.arange(self.n_dbl, device=device, dtype=torch.long)[None, :]
            st_act = w_st > 0.5
            no_act = ~st_act.any(dim=1, keepdim=True)
            st_act = torch.where(no_act, w_st > 1e-5, st_act)
            first_scores = torch.where(st_act, d_grid, torch.full_like(d_grid, self.n_dbl))
            d0 = first_scores.min(dim=1).values.clamp(max=max(0, self.n_dbl - 1))  # (B,)

            up_mask = (d_grid >= d0[:, None]) & (d_grid <= d0[:, None] + 1) & (w_st > 1e-6)
            up_empty = ~up_mask.any(dim=1, keepdim=True)
            up_mask = torch.where(up_empty, w_st > 1e-6, up_mask)
            w_up = w3d * up_mask[:, :, None, None].to(w3d.dtype)
            e_up = w_up.sum(dim=(1, 2, 3))  # (B,)

            x_grid = self.bar_x_mm[None, None, :, None]  # (1, 1, n_bars, 1)
            y_grid = self.bar_y_mm[None, None, None, :]  # (1, 1, 1, n_bars)
            z_dbl = self.dbl_z_mm[None, :]  # (1, n_dbl)

            up_valid = e_up > 1e-6
            denom_up = e_up.clamp_min(1e-6)
            x_start = torch.where(
                up_valid,
                (w_up * x_grid).sum(dim=(1, 2, 3)) / denom_up,
                torch.full_like(e_up, self.x_scint_center),
            )
            y_start = torch.where(
                up_valid,
                (w_up * y_grid).sum(dim=(1, 2, 3)) / denom_up,
                torch.full_like(e_up, self.y_scint_center),
            )
            z_start = torch.where(
                up_valid,
                (w_up.sum(dim=(2, 3)) * z_dbl).sum(dim=1) / denom_up,
                torch.full_like(e_up, float(self.dbl_z_mm[0].item())),
            )

            # Pseudo-transverse-momentum vectors relative to observable shower apex
            cal_e_scale = 12.5 if self.phys_calibrated else 1.0
            dx = x_grid - x_start[:, None, None, None]  # (B, 1, n_bars, 1)
            dy = y_grid - y_start[:, None, None, None]  # (B, 1, 1, n_bars)
            dz = (z_dbl - z_start[:, None]).clamp_min(25.0)[:, :, None, None]  # (B, n_dbl, 1, 1)
            inv_r = torch.rsqrt(dx.square() + dy.square() + dz.square())
            qx_vox = (cal_e_scale * w3d) * dx * inv_r
            qy_vox = (cal_e_scale * w3d) * dy * inv_r

            # Spectrometer leading track axis & momentum estimate (purely from trk_cand[:, 0])
            cand0 = trk_cand[:, 0, :].float()
            has_spec = trk_cand_mask[:, 0].bool()
            x_trk0 = self.x_pix_center + 150.0 * cand0[:, 2]
            y_trk0 = self.y_scint_center + 150.0 * cand0[:, 3]
            tx = 0.1 * cand0[:, 4]
            ty = 0.1 * cand0[:, 5]
            is_3st = cand0[:, 0] > 0.75
            has_p_spec = (has_spec & is_3st) if self.phys_calibrated else has_spec
            if self.phys_calibrated:
                # Resolve Audit P1-3: cand0[:, 13] = log1p(|s_y| / 0.05) / 5.0 (exact mm sagitta).
                # First-principles & Geant4 calibrated constant: |s_y| * p = 100705 MeV*mm.
                # 2-station tracks lack a 3rd station to measure bending sagitta -> p_spec = 0.
                sy_abs_mm = 0.05 * torch.expm1((5.0 * cand0[:, 13]).clamp(0.0, 10.0))
                p_3st_mev = (SAGITTA_CONST_MEV_MM / (sy_abs_mm + 0.015)).clamp(min=100.0, max=5.0e6)
                p_spec_mev = torch.where(has_p_spec, p_3st_mev, torch.zeros_like(e_stereo))
                cal_e_scale = 12.5  # W/scintillator sampling inverse 1/f_samp -> total shower MeV
            else:
                p_spec_mev = torch.where(
                    has_spec & is_3st,
                    1000.0 * torch.expm1((5.0 * cand0[:, 8]).clamp(0.0, 10.0)),
                    torch.where(
                        has_spec,
                        5000.0 * cand0[:, 1].clamp_min(0.0),
                        torch.zeros_like(e_stereo),
                    ),
                )
                cal_e_scale = 1.0
            dz_spec = z_dbl - self.z_trk0
            x_ray_spec = x_trk0[:, None] + tx[:, None] * dz_spec
            y_ray_spec = y_trk0[:, None] + ty[:, None] * dz_spec

            # Reflection-symmetric soft-argmax core axis in calorimeter for events without track
            w_smooth = F.avg_pool2d(
                w3d.reshape(b * self.n_dbl, 1, self.n_bars, self.n_bars),
                kernel_size=3,
                stride=1,
                padding=1,
            ).reshape(b, self.n_dbl, self.n_bars, self.n_bars)
            core_win = (d_grid >= d0[:, None] + 1) & (d_grid <= d0[:, None] + 10) & (w_st > 1e-6)
            core_fallback = (d_grid >= d0[:, None]) & (d_grid <= d0[:, None] + 10) & (w_st > 1e-6)
            core_win = torch.where(~core_win.any(dim=1, keepdim=True), core_fallback, core_win)
            w_core_wt = (w_smooth * core_win[:, :, None, None].to(w_smooth.dtype)).clamp_min(0.0)
            # Cube sharpening picks dense EM core with zero tie-breaking asymmetry under Y reflection
            w_core_wt = w_core_wt.pow(3.0)
            w_core_sum = w_core_wt.sum(dim=(1, 2, 3))
            core_valid = w_core_sum > 1e-18
            denom_core = w_core_sum.clamp_min(1e-18)

            x_pk = torch.where(core_valid, (w_core_wt * x_grid).sum(dim=(1, 2, 3)) / denom_core, x_start)
            y_pk = torch.where(core_valid, (w_core_wt * y_grid).sum(dim=(1, 2, 3)) / denom_core, y_start)
            z_pk = torch.where(core_valid, (w_core_wt.sum(dim=(2, 3)) * z_dbl).sum(dim=1) / denom_core, z_start + 50.0)
            dz_pk = (z_pk - z_start).clamp_min(30.0)
            sx_cal = ((x_pk - x_start) / dz_pk).clamp(-2.0, 2.0)
            sy_cal = ((y_pk - y_start) / dz_pk).clamp(-2.0, 2.0)
            dz_cal = (z_dbl - z_start[:, None]).clamp_min(0.0)
            x_ray_cal = x_start[:, None] + sx_cal[:, None] * dz_cal
            y_ray_cal = y_start[:, None] + sy_cal[:, None] * dz_cal

            use_spec = has_spec.unsqueeze(1)
            x_ray = torch.where(use_spec, x_ray_spec, x_ray_cal)  # (B, n_dbl)
            y_ray = torch.where(use_spec, y_ray_spec, y_ray_cal)  # (B, n_dbl)
            r_cut = torch.where(has_spec, torch.full_like(e_stereo, 45.0), torch.full_like(e_stereo, 35.0))

            dr2 = (x_grid - x_ray[:, :, None, None]).square() + (y_grid - y_ray[:, :, None, None]).square()
            m_core = torch.sigmoid((r_cut[:, None, None, None] - torch.sqrt(dr2 + 1e-6)) / 5.0)

            q_core_x = (qx_vox * m_core).sum(dim=(1, 2, 3))
            q_core_y = (qy_vox * m_core).sum(dim=(1, 2, 3))
            q_had_x = (qx_vox * (1.0 - m_core)).sum(dim=(1, 2, 3))
            q_had_y = (qy_vox * (1.0 - m_core)).sum(dim=(1, 2, 3))
            q_cal_x = q_core_x + q_had_x
            q_cal_y = q_core_y + q_had_y

            inv_tan = torch.rsqrt(1.0 + tx.square() + ty.square())
            q_spec_x = p_spec_mev * tx * inv_tan
            q_spec_y = p_spec_mev * ty * inv_tan
            q_lead_x = torch.where(has_p_spec, q_spec_x, q_core_x)
            q_lead_y = torch.where(has_p_spec, q_spec_y, q_core_y)

            q_lead_mag = torch.hypot(q_lead_x, q_lead_y)
            q_had_mag = torch.hypot(q_had_x, q_had_y)
            q_cal_mag = torch.hypot(q_cal_x, q_cal_y)

            pair_ok = (q_lead_mag > 0.05) & (q_had_mag > 0.05)
            cos_raw = (q_lead_x * q_had_x + q_lead_y * q_had_y) / (q_lead_mag * q_had_mag).clamp_min(1e-4)
            cos_dphi_l_had = torch.where(pair_ok, cos_raw.clamp(-1.0, 1.0), torch.zeros_like(cos_raw))
            acoplanarity_norm = torch.where(
                pair_ok,
                (math.pi - torch.acos(cos_raw.clamp(-0.999999, 0.999999))) / math.pi,
                torch.zeros_like(cos_raw),
            )

            pt_sum = torch.hypot(q_lead_x + q_had_x, q_lead_y + q_had_y)
            pt_miss_frac = torch.where(
                has_stereo | has_p_spec,
                pt_sum / (q_lead_mag + q_had_mag + 1.0),
                torch.zeros_like(pt_sum),
            )
            cross_l_h = torch.abs(q_lead_x * q_had_y - q_lead_y * q_had_x)
            thrust_denom = torch.hypot(q_lead_x - q_had_x, q_lead_y - q_had_y) + 1.0
            qt_perp_thrust_log = torch.where(
                pair_ok,
                0.2 * torch.log1p(cross_l_h / thrust_denom),
                torch.zeros_like(cross_l_h),
            )

            e_core = (w3d * m_core).sum(dim=(1, 2, 3))
            e_recoil = (w3d * (1.0 - m_core)).sum(dim=(1, 2, 3))
            core_30mm_iso = torch.where(has_stereo, e_core / e_stereo.clamp_min(1e-4), torch.zeros_like(e_core))
            recoil_energy_frac = torch.where(
                has_stereo | has_p_spec,
                (cal_e_scale * e_recoil) / (cal_e_scale * e_stereo + p_spec_mev).clamp_min(1e-4),
                torch.zeros_like(e_recoil),
            )

            # 2x2 transverse covariance around global energy-weighted transverse centroid
            denom_st = e_stereo.clamp_min(1e-4)
            w_bar_x = w3d.sum(dim=(1, 3))  # (B, n_bars)
            w_bar_y = w3d.sum(dim=(1, 2))  # (B, n_bars)
            bx_c = self.bar_x_mm[None, :]
            by_c = self.bar_y_mm[None, :]
            x_mean = torch.where(has_stereo, (w_bar_x * bx_c).sum(dim=1) / denom_st, torch.full_like(e_stereo, self.x_scint_center))
            y_mean = torch.where(has_stereo, (w_bar_y * by_c).sum(dim=1) / denom_st, torch.full_like(e_stereo, self.y_scint_center))
            dxg = bx_c - x_mean[:, None]
            dyg = by_c - y_mean[:, None]
            cxx = (w_bar_x * dxg.square()).sum(dim=1) / denom_st
            cyy = (w_bar_y * dyg.square()).sum(dim=1) / denom_st
            cxy = (w3d * (x_grid - x_mean[:, None, None, None]) * (y_grid - y_mean[:, None, None, None])).sum(dim=(1, 2, 3)) / denom_st

            tr = cxx + cyy
            disc = torch.sqrt(((cxx - cyy).square() + 4.0 * cxy.square()).clamp_min(1e-8))
            lam_max = (0.5 * (tr + disc)).clamp_min(0.0)
            lam_min = (0.5 * (tr - disc)).clamp_min(0.0)
            log_eigen_ratio = torch.where(
                has_stereo,
                0.25 * torch.log((lam_max + 9.0) / (lam_min + 9.0)),
                torch.zeros_like(lam_max),
            )
            transverse_r90_norm = torch.where(
                has_stereo,
                (torch.sqrt(tr.clamp_min(0.0)) / 150.0).clamp(0.0, 3.0),
                torch.zeros_like(tr),
            )

            spec_to_cal_log_ratio = torch.where(
                has_p_spec,
                0.2 * torch.asinh(torch.log((p_spec_mev + 10.0) / (cal_e_scale * e_cal_tot + 10.0))),
                torch.zeros_like(p_spec_mev),
            )
            t_mag = torch.hypot(tx, ty)
            spec_cal_ok = has_spec & (t_mag > 1e-4) & (q_cal_mag > 0.05)
            cos_spec_cal = torch.where(
                spec_cal_ok,
                ((tx * q_cal_x + ty * q_cal_y) / (t_mag * q_cal_mag).clamp_min(1e-5)).clamp(-1.0, 1.0),
                torch.zeros_like(p_spec_mev),
            )

            feat = torch.stack([
                cos_dphi_l_had,
                acoplanarity_norm,
                pt_miss_frac,
                qt_perp_thrust_log,
                core_30mm_iso,
                recoil_energy_frac,
                log_eigen_ratio,
                transverse_r90_norm,
                spec_to_cal_log_ratio,
                cos_spec_cal,
            ], dim=-1)
            active_dom = (obs_domain_id != 0).unsqueeze(-1).to(feat.dtype)
            return torch.nan_to_num(feat * active_dom, nan=0.0, posinf=0.0, neginf=0.0)

    def from_b1pp(self, batch: dict) -> torch.Tensor:
        """Compute 10-dim NOMAD vector directly from B1++ collated batch."""
        b = int(batch['obs_domain_id'].shape[0])
        device = batch['c_obs'].device
        e_x = torch.zeros(b * self.n_st * self.n_bars, dtype=torch.float32, device=device)
        e_y = torch.zeros(b * self.n_dbl * self.n_bars, dtype=torch.float32, device=device)

        bx_ev = batch['bx_ev']
        if len(bx_ev):
            bx_idx = batch['bx_idx']
            vp = bx_idx // self.n_bars
            bi = bx_idx % self.n_bars
            slot = self.b1pp_xslot[vp] * self.n_bars + bi
            lin = bx_ev * (self.n_st * self.n_bars) + slot
            edep_x = torch.expm1((batch['bx_val'][:, 0].float() * 10.0).clamp(0.0, 12.0))
            e_x.index_add_(0, lin, edep_x)

        by_ev = batch['by_ev']
        if len(by_ev):
            by_idx = batch['by_idx']
            hp = by_idx // self.n_bars
            bi = by_idx % self.n_bars
            slot = self.b1pp_yslot[hp] * self.n_bars + bi
            lin = by_ev * (self.n_dbl * self.n_bars) + slot
            edep_y = torch.expm1((batch['by_val'][:, 0].float() * 10.0).clamp(0.0, 12.0))
            e_y.index_add_(0, lin, edep_y)

        return self._compute_from_grids(
            e_x.view(b, self.n_st, self.n_bars),
            e_y.view(b, self.n_dbl, self.n_bars),
            batch['trk_cand'],
            batch['trk_cand_mask'],
            batch['obs_domain_id'],
        )

    def from_pf(self, batch: dict) -> torch.Tensor:
        """Compute 10-dim NOMAD vector directly from PinpointFormer collated batch."""
        b = int(batch['obs_domain_id'].shape[0])
        device = batch['c_obs'].device
        e_x = torch.zeros(b * self.n_st * self.n_bars, dtype=torch.float32, device=device)
        e_y = torch.zeros(b * self.n_dbl * self.n_bars, dtype=torch.float32, device=device)

        bar_ev = batch['bar_ev']
        if len(bar_ev):
            bp = batch['bar_panel']
            bh = batch['bar_horiz']
            bi = batch['bar_idx']
            edep = torch.expm1((batch['bar_val'][:, 0].float() * 10.0).clamp(0.0, 12.0))

            xm = bh == 0
            if xm.any():
                xslot = self.panel_to_xslot[bp[xm]]
                valid_x = xslot >= 0
                if valid_x.any():
                    lin_x = bar_ev[xm][valid_x] * (self.n_st * self.n_bars) + xslot[valid_x] * self.n_bars + bi[xm][valid_x]
                    e_x.index_add_(0, lin_x, edep[xm][valid_x])

            ym = bh == 1
            if ym.any():
                yslot = self.panel_to_yslot[bp[ym]]
                valid_y = yslot >= 0
                if valid_y.any():
                    lin_y = bar_ev[ym][valid_y] * (self.n_dbl * self.n_bars) + yslot[valid_y] * self.n_bars + bi[ym][valid_y]
                    e_y.index_add_(0, lin_y, edep[ym][valid_y])

        return self._compute_from_grids(
            e_x.view(b, self.n_st, self.n_bars),
            e_y.view(b, self.n_dbl, self.n_bars),
            batch['trk_cand'],
            batch['trk_cand_mask'],
            batch['obs_domain_id'],
        )
