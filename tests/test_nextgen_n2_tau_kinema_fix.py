"""Unit & Gate A/B tests for Experiment N2 (TauKinemaDiffNet first_tgt_p <= 5 gate) and Q13 full 1mm sensor."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F

from analysis.flavour.nextgen.run_q13_full_1mm_sensor import (
    extract_full_1mm_sensor_item,
)
from analysis.flavour.nextgen.run_tau_kinema_diffnet import (
    DEFAULT_CLEAN_ROOT,
    SPLITS_DIR,
    _TauKinemaItemExtractor,
    _base_obs_config,
    collate_tau_kinema,
)
from analysis.flavour.nextgen.seal_guard import (
    SealAuditGuard,
    load_authorized_split_shards,
    verify_inference_truth_invariance,
)
from analysis.flavour.nextgen.tau_kinema_diffnet import (
    TauKinemaDiffNet,
    compute_tau_kinema_loss,
)
from analysis.flavour.nextgen.tracklet_ipformer import extract_tracklet_pf_item
from analysis.flavour.v2.partitions import KEY
from analysis.flavour.v3.common.domains import compute_observable_containment, detector_bounds
from analysis.flavour.v3.v2_pinpointformer.train import move_batch_to_device
from tests.test_nextgen_model_health import (
    _build_32_event_4class_observations,
    run_gate_a_health_check,
    run_gate_b_overfit_32_check,
)
from tests.test_nextgen_seal_and_splits import build_minimal_geometry


def _load_real_or_synthetic_32(device: torch.device) -> tuple[dict, list[dict], dict]:
    cfg_obs = _base_obs_config()
    if (DEFAULT_CLEAN_ROOT / 'geometry.json').is_file() and (SPLITS_DIR / 'internal_train_keys.parquet').is_file():
        geometry = json.loads((DEFAULT_CLEAN_ROOT / 'geometry.json').read_text())
        bounds = detector_bounds(geometry)
        extractor = _TauKinemaItemExtractor(geometry, bounds, cfg_obs)
        tr_keys = pd.read_parquet(SPLITS_DIR / 'internal_train_keys.parquet', columns=KEY + ['label'])
        auth_shards = load_authorized_split_shards(DEFAULT_CLEAN_ROOT / 'data', 'train', stage=1)
        shard0_prov = pd.read_parquet(auth_shards[0] / 'provenance.parquet', columns=KEY + ['label'])
        tr_idx_set = pd.MultiIndex.from_frame(tr_keys[KEY])
        in_tr = pd.MultiIndex.from_frame(shard0_prov[KEY]).isin(tr_idx_set)
        cand_df = shard0_prov.loc[in_tr].reset_index()
        picked_rows: list[int] = []
        for cls_id in range(4):
            rows_c = cand_df.loc[cand_df['label'] == cls_id, 'index'].to_numpy(dtype=int)
            picked_rows.extend(rows_c[:12].tolist())
        obs_list = pq.read_table(auth_shards[0] / 'observations.parquet').to_pylist()
        sc_df = pd.read_parquet(DEFAULT_CLEAN_ROOT / 'truth_sidecar/event_sidecar.parquet')
        sc_idx_map = {
            (int(r), int(c), int(rw)): i
            for i, (r, c, rw) in enumerate(zip(sc_df['run'], sc_df['chunk'], sc_df['row']))
        }
        items: list[dict] = []
        counts_c = [0, 0, 0, 0]
        for max_hits in (1200, 10**9):
            for raw_i in picked_rows:
                row_meta = shard0_prov.iloc[raw_i]
                c4 = int(row_meta['label'])
                if counts_c[c4] >= 8:
                    continue
                ob = obs_list[raw_i]
                if len(ob['pixel']['channel_id']) > max_hits:
                    continue
                dom = compute_observable_containment(ob, geometry, bounds, cfg_obs)
                if dom['obs_domain_id'] == 0 or (len(ob['pixel']['channel_id']) == 0 and len(ob['bar']['panel']) == 0):
                    continue
                it = extractor(raw_i, ob, c4, dom, geometry, bounds, cfg_obs)
                if any(x.get('index') == it.get('index') for x in items):
                    continue
                sc_pos = sc_idx_map[(int(row_meta['run']), int(row_meta['chunk']), int(row_meta['row']))]
                sc_row = sc_df.iloc[sc_pos]
                it['label_7'] = int(sc_row['label_7'])
                it['tau_branch_valid'] = int(sc_row['tau_branch_valid'])
                it['charm_flag'] = int(sc_row['charm_flag'])
                it['charm_valid'] = int(sc_row['charm_valid'])
                it['vertex_xyz_mm'] = np.array(
                    [sc_row['vertex_x_mm'], sc_row['vertex_y_mm'], sc_row['vertex_z_mm']], dtype=np.float32
                )
                it['vertex_valid'] = int(sc_row['vertex_valid'])
                it['vis_kinematics'] = np.array(
                    [sc_row['vis_pz_log1p'], sc_row['vis_pt_log1p'], sc_row['vis_cos_dphi'], sc_row['vis_sin_dphi']],
                    dtype=np.float32,
                )
                it['kin_valid'] = int(sc_row['kin_valid'])
                items.append(it)
                counts_c[c4] += 1
                if sum(counts_c) == 32:
                    break
            if sum(counts_c) == 32:
                break
        assert sum(counts_c) == 32, f'Expected 32 balanced real Internal-Train events, got {counts_c}'
        return geometry, items, bounds

    geometry = build_minimal_geometry()
    bounds = detector_bounds(geometry)
    extractor = _TauKinemaItemExtractor(geometry, bounds, cfg_obs)
    obs32, lbl32 = _build_32_event_4class_observations(geometry, bounds, seed=327)
    items = []
    for i, (ob, c4) in enumerate(zip(obs32, lbl32)):
        dom = compute_observable_containment(ob, geometry, bounds, cfg_obs)
        it = extractor(i, ob, int(c4), dom, geometry, bounds, cfg_obs)
        it['label_7'] = int(c4 if c4 < 2 else (3 if c4 == 2 else 6))
        it['tau_branch_valid'] = 1
        it['charm_flag'] = 0
        it['charm_valid'] = 1
        it['vertex_xyz_mm'] = np.array([0.0, -12.0, bounds['z_mid']], dtype=np.float32)
        it['vertex_valid'] = 1
        it['vis_kinematics'] = np.zeros(4, dtype=np.float32)
        it['kin_valid'] = 1
        items.append(it)
    return geometry, items, bounds


class TestNextGenN2TauKinemaFix(unittest.TestCase):
    def test_interm150_masks_avr_pv_while_front30_retains_it(self) -> None:
        """Explicit assertion: first_tgt_p >= 6 masks out C++ avr_pv while first_tgt_p <= 5 retains it."""
        geometry = build_minimal_geometry()
        bounds = detector_bounds(geometry)
        cfg_obs = _base_obs_config()
        extractor = _TauKinemaItemExtractor(geometry, bounds, cfg_obs)
        obs32, lbl32 = _build_32_event_4class_observations(geometry, bounds, seed=101)
        it0 = extractor(0, obs32[0], int(lbl32[0]), compute_observable_containment(obs32[0], geometry, bounds, cfg_obs), geometry, bounds, cfg_obs)
        it1 = copy.deepcopy(it0)

        # Force event 0 into Front30 (first_tgt_p = 2 <= 5) and event 1 into Interm150 (first_tgt_p = 7 >= 6)
        it0['pix_plane_summary'][:] = 0.0
        it0['pix_plane_summary'][2, 0] = 1.0
        it0['pix_plane_summary'][2, 6] = 0.01
        it0['pix_plane_summary'][2, 7] = 0.01
        it0['cal_start'] = np.array([bounds['pixel_Zpos'][2], 0.0, -12.0, 1.0], dtype=np.float32)
        it0['avr_pv_mm'] = np.array([3.5, -9.5, bounds['pixel_Zpos'][2] - 10.0, 1.0], dtype=np.float32)

        it1['pix_plane_summary'][:] = 0.0
        it1['pix_plane_summary'][7, 0] = 1.0
        it1['pix_plane_summary'][7, 6] = 0.01
        it1['pix_plane_summary'][7, 7] = 0.01
        it1['cal_start'] = np.array([bounds['pixel_Zpos'][7], 0.0, -12.0, 1.0], dtype=np.float32)
        it1['avr_pv_mm'] = np.array([9.5, -4.5, bounds['pixel_Zpos'][7] - 10.0, 1.0], dtype=np.float32)

        batch_a = collate_tau_kinema([it0, it1], y_scint_center=bounds['y_scint_center'], n_bars=bounds['n_bars'])
        it0_mut = copy.deepcopy(it0)
        it1_mut = copy.deepcopy(it1)
        it0_mut['avr_pv_mm'] = np.array([-8.0, -18.0, bounds['pixel_Zpos'][2] + 5.0, 1.0], dtype=np.float32)
        it1_mut['avr_pv_mm'] = np.array([-15.0, -25.0, bounds['pixel_Zpos'][7] + 40.0, 1.0], dtype=np.float32)
        batch_b = collate_tau_kinema([it0_mut, it1_mut], y_scint_center=bounds['y_scint_center'], n_bars=bounds['n_bars'])

        torch.manual_seed(42)
        model = TauKinemaDiffNet(geometry, {'dim': 168, 'heads': 8, 'drop_path': 0.0, 'head_dropout': 0.0}).eval()
        with torch.inference_mode():
            prior_a, _, is_f_a, is_m_a = model._compute_cal_prior(batch_a, 2, torch.device('cpu'))
            prior_b, _, _, _ = model._compute_cal_prior(batch_b, 2, torch.device('cpu'))
            out_a = model.forward_NextGen(batch_a)
            out_b = model.forward_NextGen(batch_b)

        self.assertTrue(bool(is_f_a[0].item()) and not bool(is_m_a[0].item()))
        self.assertTrue(not bool(is_f_a[1].item()) and bool(is_m_a[1].item()))
        # Front30 retains avr_pv_mm
        np.testing.assert_allclose(prior_a[0].numpy(), it0['avr_pv_mm'][:3], atol=1e-5)
        self.assertGreater(float((prior_a[0] - prior_b[0]).abs().max().item()), 1.0)
        self.assertGreater(float((out_a.v_pv_mm[0] - out_b.v_pv_mm[0]).abs().max().item()), 0.1)
        # Interm150 masks out avr_pv_mm completely (bit-exact invariance to avr_pv_mm mutation)
        self.assertEqual(float((prior_a[1] - prior_b[1]).abs().max().item()), 0.0)
        self.assertEqual(float((out_a.v_pv_mm[1] - out_b.v_pv_mm[1]).abs().max().item()), 0.0)
        self.assertEqual(float((out_a.probs4[1] - out_b.probs4[1]).abs().max().item()), 0.0)

    def test_gate_ab_39key_truth_invariance_and_32_real_overfit(self) -> None:
        """Gate A health, 39-key truth invariance == 0.0, and 32/32 real overfit < 0.05."""
        with SealAuditGuard():
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            geometry, items32, bounds = _load_real_or_synthetic_32(device)
            b8 = move_batch_to_device(collate_tau_kinema(items32[:8], bounds['y_scint_center'], bounds['n_bars']), device)
            b32 = move_batch_to_device(collate_tau_kinema(items32, bounds['y_scint_center'], bounds['n_bars']), device)

            torch.manual_seed(42)
            model = TauKinemaDiffNet(geometry, {'dim': 168, 'heads': 8, 'drop_path': 0.0, 'head_dropout': 0.0}).to(device)
            ga = run_gate_a_health_check(
                model,
                b8,
                lambda out, b: compute_tau_kinema_loss(model, out, b)[0],
                expected_logits_shape=(8, 7),
                branch_groups_fn=lambda m: {k: [(f'p{i}', p) for i, p in enumerate(v)] for k, v in m.branch_parameter_groups().items()},
            )
            self.assertTrue(ga['passed'])

            ti = verify_inference_truth_invariance(model, b8, atol=1e-7)
            self.assertEqual(float(ti['max_abs_diff']), 0.0)

            def _gate_b_loss(out: dict, b: dict) -> torch.Tensor:
                l_full, _ = compute_tau_kinema_loss(model, out, b, w_vtx=0.0, w_kin=0.0, w_charm=0.05)
                p4 = out['probs4'].float().clamp_min(1e-7)
                return F.nll_loss(torch.log(p4), b['label'].long()) + 0.002 * l_full

            gb = run_gate_b_overfit_32_check(
                model,
                b32,
                _gate_b_loss,
                steps=100,
                lr=3.0e-3,
                min_acc=1.0,
                max_final_loss=0.05,
                upstream_param_prefixes=('pix_in.', 'stereo_bars.', 'trk_in.'),
            )
            self.assertTrue(gb['passed'])
            self.assertEqual(float(gb['final_acc']), 1.0)
            self.assertLess(float(gb['final_loss']), 0.05)

    def test_q13_full_1mm_sensor_locks_clusters_and_resamples_target_patches(self) -> None:
        """Q13 abl_full_1mm_sensor locks 20um clusters/c_obs/IPT while quantizing target centroids & patches."""
        geometry = build_minimal_geometry()
        bounds = detector_bounds(geometry)
        nx, ny = int(bounds['nx']), int(bounds['ny'])
        # Build a multi-hit target cluster on plane 1 (>3 hits & >0.04mm span to trigger patch) + IPT cluster on plane 12
        cids = [
            1 * ny * nx + 4000 * nx + 6000,
            1 * ny * nx + 4001 * nx + 6001,
            1 * ny * nx + 4002 * nx + 6002,
            1 * ny * nx + 4003 * nx + 6003,
            12 * ny * nx + 4100 * nx + 6100,
            12 * ny * nx + 4101 * nx + 6101,
            12 * ny * nx + 4102 * nx + 6102,
            12 * ny * nx + 4103 * nx + 6103,
        ]
        xyzs = [
            [0.10, -12.30, bounds['pixel_Zpos'][1]],
            [0.12, -12.28, bounds['pixel_Zpos'][1]],
            [0.15, -12.25, bounds['pixel_Zpos'][1]],
            [0.17, -12.23, bounds['pixel_Zpos'][1]],
            [2.20, -10.05, bounds['pixel_Zpos'][12]],
            [2.22, -10.03, bounds['pixel_Zpos'][12]],
            [2.25, -10.00, bounds['pixel_Zpos'][12]],
            [2.27, -9.98, bounds['pixel_Zpos'][12]],
        ]
        obs = {
            'pixel': {'channel_id': np.asarray(cids, dtype=np.int64), 'xyz_mm': np.asarray(xyzs, dtype=np.float32)},
            'bar': {
                'channel_id': np.empty(0, dtype=np.int64), 'layer': np.empty(0, dtype=np.int16),
                'panel': np.empty(0, dtype=np.int16), 'horizontal': np.empty(0, dtype=np.int8),
                'bar': np.empty(0, dtype=np.int16), 'bounds_mm': np.empty((0, 5), dtype=np.float32),
                'pe_proxy': np.empty(0, dtype=np.int64), 'hit_time_ns': np.empty(0, dtype=np.float32),
            },
            'tracker_hypothesis': {
                'station': np.empty(0, dtype=np.int8), 'xy_mm': np.empty((0, 2), dtype=np.float32),
                'sigma_xy_mm': np.empty((0, 2), dtype=np.float32),
            },
        }
        dom = compute_observable_containment(obs, geometry, bounds, {'pe_only': True})
        it_nat = extract_tracklet_pf_item(0, obs, None, dom, geometry, bounds, {'pe_only': True})
        it_q = extract_tracklet_pf_item(0, obs, None, dom, geometry, bounds, {'pe_only': True, 'quantize_cluster_coords_only_mm': 1.00})
        it_full = extract_full_1mm_sensor_item(0, obs, None, dom, geometry, bounds, {'pe_only': True})

        self.assertEqual(len(it_nat['pix_plane']), len(it_full['pix_plane']))
        np.testing.assert_array_equal(it_nat['c_obs'], it_full['c_obs'])
        np.testing.assert_array_equal(it_nat['pix_plane'], it_full['pix_plane'])
        np.testing.assert_array_equal(it_q['pix_xyz_mm'], it_full['pix_xyz_mm'])

        # IPT clusters & patches are bit-for-bit preserved
        ipt_cl = it_full['pix_plane'] >= 12
        np.testing.assert_array_equal(it_nat['pix_xyz_mm'][ipt_cl], it_full['pix_xyz_mm'][ipt_cl])
        np.testing.assert_array_equal(it_nat['pix_feat'][ipt_cl], it_full['pix_feat'][ipt_cl])
        p_ipt = it_full['pix_plane'][it_full['patch_cluster_idx']] >= 12
        self.assertTrue(p_ipt.any())
        np.testing.assert_array_equal(it_nat['pix_patches'][p_ipt], it_full['pix_patches'][p_ipt])

        # Target cluster hits (4 hits within ~0.07 mm) snap to 1-bit 1.00 mm voxel -> n_hits=1, ecc=0, 1.00 mm patch
        p_tgt = ~p_ipt
        self.assertTrue(p_tgt.any())
        tgt_cl = it_full['pix_plane'] < 12
        self.assertEqual(float(it_full['pix_feat'][tgt_cl, 9].max()), 0.0)  # ecc == 0 for single 1mm voxel
        self.assertGreater(float(np.abs(it_q['pix_patches'][p_tgt].astype(np.float32) - it_full['pix_patches'][p_tgt].astype(np.float32)).max()), 1e-2)
        self.assertLess(float(it_full['pix_patches'][p_tgt].sum()), float(it_q['pix_patches'][p_tgt].sum()))  # 1-bit dedup (1 vs 4)


if __name__ == '__main__':
    unittest.main()
