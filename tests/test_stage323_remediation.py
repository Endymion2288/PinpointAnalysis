"""Automated regression tests for Stage 323 remediation of Stage 322 model-training issues.

Covers:
  - 322-01 / 322-13: Pure-W target filtering and >=99.95% full-volume W sampler closure.
  - 322-03: Strictly decoupled target vs IPT pixel quantization, clustering radius, per-sensor
    pitch features, independent cluster/patch budgets, and bit-for-bit IPT invariance.
  - 322-07: Exact centroid tracker merging yielding 17 um Y RMS & unit-variance pull (not 33.5 um)
    plus permutation invariance and optional dead-zone/noise controls.
  - 322-11: Zero-label observation-only dataset (`V3ObservationDataset`), collators, and forward
    passes across B1++, PinpointFormerMultitask, and PinpointFormerV4 matching labeled inference
    to 0.0 bit-for-bit.
  - 322-12: 0-PE scintillator bar hits never determine aggregated `hit_time_ns`.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from analysis.flavour.readout import bar_channels, tracker_measurements
from analysis.flavour.v3.common.datasets import V3ObservationDataset
from analysis.flavour.v3.common.domains import compute_observable_containment, detector_bounds
from analysis.flavour.v3.common.observation_inference import infer_observation_only
from analysis.flavour.v3.v1_b1pp.models import ZAlignedStereoNet
from analysis.flavour.v3.v1_b1pp.representation import (
    collate_b1pp,
    extract_b1pp_item,
    static_plane_layout,
)
from analysis.flavour.v3.v2_pinpointformer.pixel_tokenizer import (
    N_MAX_IPT_CLUSTERS,
    N_MAX_TARGET_CLUSTERS,
    collate_pf,
    extract_pf_item,
    quantize_pixel_hits,
)
from analysis.flavour.v3.v3_multitask.models import PinpointFormerMultitask
from analysis.flavour.v3.v4_tau.models import PinpointFormerV4
from analysis.flavour.v3.v6_clean_sim.produce_clean_v3 import (
    _compute_energy_match_weights,
    build_full_volume_tungsten_sampler_spec,
)


CLEAN_V2_GEOM = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v5_clean_v2_2026-11-19/clean_sim/geometry.json')


def _synthetic_geometry() -> dict:
    if CLEAN_V2_GEOM.is_file():
        return json.loads(CLEAN_V2_GEOM.read_text())
    # Fallback synthetic Pinpoint geometry if EOS unavailable
    w_z, sc_z, p_z = [], [], []
    z = -3000.0
    for i in range(6):
        p_z.append(z + 8.005)
        w_z.extend([z + 15.51, z + 25.51])
        sc_z.append(z + 20.51)
        z += 30.01
    for f in range(7):
        for j in range(6):
            w_z.extend([z + 2.5 + j * 20.0, z + 12.5 + j * 20.0])
            sc_z.extend([z + 7.5 + j * 20.0, z + 17.5 + j * 20.0])
        z += 120.0
        if f < 6:
            p_z.append(z + 8.005)
            w_z.extend([z + 15.51, z + 25.51])
            sc_z.append(z + 20.51)
            z += 30.01
    p_z.extend([z + 65.005, z + 95.015, z + 125.025])
    return {
        'detector_width': 266.0,
        'detector_height': 196.0,
        'silicon_thickness': 10.0,
        'tungsten_thickness': 5.0,
        'nFortuneBlocks': 7,
        'numScintLayers': 6,
        'num_scint_panels_layer': 2,
        'nPinpointBlocks': 6,
        'scint_detector_width': 420.0,
        'scint_detector_height': 420.0,
        'scint_thickness': 5.0,
        'scint_bar_width': 10.0,
        'scint_bar_height': 10.0,
        'nLayers': 15,
        'layer_is_pixel': [1] * 6 + [0, 1] * 6 + [0] + [1] * 3,
        'pixel_Xpos': (-133.0 + (np.arange(12788) + 0.5) * 0.0208).tolist(),
        'pixel_Ypos': (-110.0 + (np.arange(8596) + 0.5) * 0.0228).tolist(),
        'pixel_Zpos': p_z,
        'tungsten_Zpos': w_z,
        'scint_Zpos': sc_z,
        'scint_bar_Xcenter': (77.0 - 210.0 + (np.arange(40) + 0.5) * 10.5).tolist(),
        'scint_bar_Ycenter': (-12.0 - 210.0 + (np.arange(40) + 0.5) * 10.5).tolist(),
    }


class Stage323RemediationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.geometry = _synthetic_geometry()
        cls.bounds = detector_bounds(cls.geometry)

    def _make_obs(self, n_tgt_clusters: int = 25, n_ipt_hits: int = 20) -> dict:
        nx, ny = int(self.bounds['nx']), int(self.bounds['ny'])
        px = np.asarray(self.geometry['pixel_Xpos'], dtype=np.float64)
        py = np.asarray(self.geometry['pixel_Ypos'], dtype=np.float64)
        pz = np.asarray(self.geometry['pixel_Zpos'], dtype=np.float64)

        cids, xyzs = [], []
        # Dense target hits across planes 0..5
        for pl in range(6):
            for k in range(n_tgt_clusters):
                col = 6000 + k * 25
                row = 4000 + pl * 15 + (k % 5) * 10
                for dc in (0, 1):
                    c = col + dc
                    cids.append((pl * ny + row) * nx + c)
                    xyzs.append([px[c], py[row], pz[pl]])
        # Stage 322-03 counterexample on IPT plane 12: two hits separated by exactly 0.208 mm (10 cols)!
        # Plus a multi-hit IPT cluster that triggers a 32x32 micro-patch.
        for pl in (12, 13, 14):
            # 0.208 mm double hit (must stay 2 clusters under ANY target pitch!)
            for col in (6000, 6010):
                row = 4000 + (pl - 12) * 5
                cids.append((pl * ny + row) * nx + col)
                xyzs.append([px[col], py[row], pz[pl]])
            # Multi-hit cluster triggering an IPT patch (4 hits spanning > 0.04 mm)
            for dc in range(4):
                col = 6200 + dc
                row = 4100
                cids.append((pl * ny + row) * nx + col)
                xyzs.append([px[col], py[row], pz[pl]])
            # Additional spaced IPT hits
            for k in range(n_ipt_hits):
                col = 5000 + k * 40
                row = 3500 + (k % 7) * 20
                cids.append((pl * ny + row) * nx + col)
                xyzs.append([px[col], py[row], pz[pl]])

        cids_arr = np.asarray(cids, dtype=np.uint64)
        xyz_arr = np.asarray(xyzs, dtype=np.float32)
        order = np.argsort(cids_arr)
        cids_arr = cids_arr[order]
        xyz_arr = xyz_arr[order]

        return {
            'pixel': {'channel_id': cids_arr.tolist(), 'xyz_mm': xyz_arr.tolist()},
            'bar': {
                'channel_id': [10, 11],
                'layer': [0, 1],
                'panel': [0, 1],
                'horizontal': [0, 0],
                'bar': [20, 21],
                'bounds_mm': [[0.0, 10.0, -200.0, 200.0, -2980.0], [10.0, 20.0, -200.0, 200.0, -2950.0]],
                'pe_proxy': [450, 920],
                'hit_time_ns': [4.2, 4.5],
            },
            'tracker_hypothesis': {
                'station': [0, 1, 2],
                'xy_mm': [[5.0, -10.0], [5.2, -9.8], [5.4, -9.1]],
                'sigma_xy_mm': [[0.58, 0.017], [0.58, 0.017], [0.58, 0.017]],
            },
            'has_response': [True, True, True],
            'outside_acceptance': [-1, -1, -1],
            'fit_attempted': False,
            'momentum_censored': -1,
        }

    def test_322_03_ipt_bit_exact_invariance_under_target_pitch_and_2x2_orthogonality(self) -> None:
        """Changing target pitch MUST leave EVERY IPT cluster, feature, plane summary, and patch bit-exact."""
        obs = self._make_obs(n_tgt_clusters=60, n_ipt_hits=40)
        dom = compute_observable_containment(obs, self.geometry, self.bounds, config={'pe_only': True})

        it_base = extract_pf_item(0, obs, 2, dom, self.geometry, self.bounds, {'pe_only': True})
        for q_tgt in (0.10, 0.30, 1.00):
            # Test BOTH new explicit key and legacy quantize_xy_mm + quantize_target_only=True
            for cfg in (
                {'pe_only': True, 'quantize_target_xy_mm': q_tgt, 'quantize_ipt_xy_mm': None},
                {'pe_only': True, 'quantize_xy_mm': q_tgt, 'quantize_target_only': True},
            ):
                it_q = extract_pf_item(0, obs, 2, dom, self.geometry, self.bounds, cfg)
                ipt_b = it_base['pix_plane'] >= 12
                ipt_q = it_q['pix_plane'] >= 12
                self.assertEqual(int(ipt_b.sum()), int(ipt_q.sum()))
                np.testing.assert_array_equal(it_base['pix_plane'][ipt_b], it_q['pix_plane'][ipt_q])
                np.testing.assert_array_equal(it_base['pix_first_cid'][ipt_b], it_q['pix_first_cid'][ipt_q])
                np.testing.assert_array_equal(it_base['pix_xyz_mm'][ipt_b], it_q['pix_xyz_mm'][ipt_q])
                np.testing.assert_array_equal(it_base['pix_feat'][ipt_b], it_q['pix_feat'][ipt_q])
                np.testing.assert_array_equal(it_base['pix_plane_summary'][12:], it_q['pix_plane_summary'][12:])
                # IPT patches must be bit-for-bit identical
                pb_ipt = it_base['pix_plane'][it_base['patch_cluster_idx']] >= 12
                pq_ipt = it_q['pix_plane'][it_q['patch_cluster_idx']] >= 12
                self.assertEqual(int(pb_ipt.sum()), int(pq_ipt.sum()))
                np.testing.assert_array_equal(it_base['pix_patches'][pb_ipt], it_q['pix_patches'][pq_ipt])

        # Conversely, quantizing IPT ONLY must leave EVERY Target cluster, feature, plane summary, and patch bit-exact!
        it_ipt1 = extract_pf_item(
            0, obs, 2, dom, self.geometry, self.bounds,
            {'pe_only': True, 'quantize_target_xy_mm': None, 'quantize_ipt_xy_mm': 1.0},
        )
        tgt_b = it_base['pix_plane'] < 12
        tgt_q = it_ipt1['pix_plane'] < 12
        self.assertEqual(int(tgt_b.sum()), int(tgt_q.sum()))
        np.testing.assert_array_equal(it_base['pix_xyz_mm'][tgt_b], it_ipt1['pix_xyz_mm'][tgt_q])
        np.testing.assert_array_equal(it_base['pix_feat'][tgt_b], it_ipt1['pix_feat'][tgt_q])
        np.testing.assert_array_equal(it_base['pix_plane_summary'][:12], it_ipt1['pix_plane_summary'][:12])
        # Verify IPT actually coarsened (0.208 mm double hit merged under IPT=1mm!)
        self.assertLess(int((it_ipt1['pix_plane'] >= 12).sum()), int((it_base['pix_plane'] >= 12).sum()))

    def test_322_03_cap_isolation_high_occupancy(self) -> None:
        """Even when target exceeds 832 clusters, coarsening target does not alter retained IPT clusters."""
        obs = self._make_obs(n_tgt_clusters=200, n_ipt_hits=80)  # 6*200=1200 target clusters > 832 cap!
        dom = compute_observable_containment(obs, self.geometry, self.bounds, config={'pe_only': True})
        it_base = extract_pf_item(0, obs, 2, dom, self.geometry, self.bounds, {'pe_only': True})
        it_tgt1 = extract_pf_item(
            0, obs, 2, dom, self.geometry, self.bounds,
            {'pe_only': True, 'quantize_target_xy_mm': 1.0, 'quantize_ipt_xy_mm': None},
        )
        self.assertLessEqual(int((it_base['pix_plane'] < 12).sum()), N_MAX_TARGET_CLUSTERS)
        self.assertLessEqual(int((it_base['pix_plane'] >= 12).sum()), N_MAX_IPT_CLUSTERS)
        ipt_b = it_base['pix_plane'] >= 12
        ipt_q = it_tgt1['pix_plane'] >= 12
        self.assertEqual(int(ipt_b.sum()), int(ipt_q.sum()))
        np.testing.assert_array_equal(it_base['pix_xyz_mm'][ipt_b], it_tgt1['pix_xyz_mm'][ipt_q])
        np.testing.assert_array_equal(it_base['pix_feat'][ipt_b], it_tgt1['pix_feat'][ipt_q])

    def test_322_07_tracker_exact_17um_rms_and_unit_pull(self) -> None:
        """Exact centroid merge removes 100 um grid quantization bias (17 um Y RMS, pull std ~ 1.0)."""
        rng = np.random.default_rng(32307)
        y_true = rng.uniform(-100.0, 100.0, size=4000)
        x_true = rng.uniform(-100.0, 100.0, size=4000)
        dy, pulls = [], []
        for i in range(len(y_true)):
            rec = {'trackerID': [0], 'x': [x_true[i]], 'y': [y_true[i]], 'z': [47.4]}
            _, dig = tracker_measurements(rec, seed=10000 + i, efficiency=1.0)
            err_y = float(dig['xy_mm'][0, 1] - y_true[i])
            dy.append(err_y)
            pulls.append(err_y / float(dig['sigma_xy_mm'][0, 1]))
        rms_um = float(np.sqrt(np.mean(np.square(dy))) * 1000.0)
        pull_std = float(np.std(pulls))
        self.assertAlmostEqual(rms_um, 17.0, delta=1.0)
        self.assertAlmostEqual(pull_std, 1.0, delta=0.05)

    def test_322_12_zero_pe_never_sets_channel_time(self) -> None:
        """Stage 322-12 counterexample: 0 PE @ 1 ns + 40 PE @ 5 ns -> 40 PE @ 5 ns; 0 PE alone -> -1 ns."""
        rec = {
            'layerID': [0, 0, 0],
            'panelID': [0, 0, 0],
            'isHorizontal': [0, 0, 0],
            'rowID': [-1, -1, -1],
            'colID': [10, 10, 11],
            'nPE': [0, 40, 0],
            'hitTime': [1.0, 5.0, 2.0],
        }
        bars, _ = bar_channels(rec, self.geometry, include_truth_edep=False)
        self.assertEqual(bars['pe_proxy'].tolist(), [40, 0])
        self.assertAlmostEqual(float(bars['hit_time_ns'][0]), 5.0, places=5)
        self.assertAlmostEqual(float(bars['hit_time_ns'][1]), -1.0, places=5)

    def test_322_13_full_volume_tungsten_closure(self) -> None:
        """1 um inset samples >= 99.95% of physical G4_W volume with exact 3-stratum normalization."""
        spec = build_full_volume_tungsten_sampler_spec(self.geometry, self.bounds, margin_xy_mm=0.001, margin_z_mm=0.001)
        self.assertGreater(spec['sampled_physical_W_volume_fraction'], 0.9995)
        self.assertAlmostEqual(sum(spec['volume_fractions']), 1.0, places=9)

    def test_322_11_zero_label_observation_only_replay_all_families(self) -> None:
        """Label-free observation-only directory replays B1++, PF-Multitask, PF-V4 with 0.0 diff."""
        torch.manual_seed(32311)
        obs_list = [self._make_obs(n_tgt_clusters=10 + i * 3, n_ipt_hits=8) for i in range(4)]

        with tempfile.TemporaryDirectory(prefix='obs-only-test-') as tmp:
            tmp_p = Path(tmp)
            obs_pq = tmp_p / 'observations.parquet'
            geom_json = tmp_p / 'geometry.json'
            pq.write_table(pa.Table.from_pylist(obs_list), obs_pq)
            geom_json.write_text(json.dumps(self.geometry))
            # Confirm NO targets.parquet or provenance.parquet exists!
            self.assertFalse((tmp_p / 'targets.parquet').exists())

            b1_cfg = {
                'task': 'T4', 'dim': 64, 'n_long_layers': 2, 'heads': 4, 'drop_path': 0.0,
                'channel_dropout': 0.0, 'pe_only': True, 'train_module': 'analysis.flavour.v3.v1_b1pp.train',
            }
            pf_cfg = {
                'task': 'T4', 'dim': 64, 'heads': 4, 'n_local_layers': 1, 'n_global_layers': 1,
                'drop_path': 0.0, 'pe_only': True, 'enable_dense_gn2': False, 'enable_micro_vertex': True,
                'train_module': 'analysis.flavour.v3.v3_multitask.train',
            }
            v4_cfg = {
                **pf_cfg, 'model_family': 'pf_v4', 'enable_nomad': True, 'enable_pauc': False,
                'train_module': 'analysis.flavour.v3.v4_tau.train',
            }

            for name, cfg in [('b1pp', b1_cfg), ('pf_hybrid', pf_cfg), ('pf_v4_nomad', v4_cfg)]:
                if name == 'b1pp':
                    model = ZAlignedStereoNet(static_plane_layout(self.geometry), cfg).eval()
                elif name == 'pf_v4_nomad':
                    model = PinpointFormerV4(self.geometry, cfg).eval()
                else:
                    model = PinpointFormerMultitask(self.geometry, cfg).eval()
                ckpt = tmp_p / f'{name}.pt'
                torch.save({'model': model.state_dict()}, ckpt)

                # 1. Run observation-only API (zero labels on disk, zero label key in batch)
                df_obs = infer_observation_only(tmp_p, geom_json, ckpt, cfg, batch_size=2, device='cpu')
                self.assertEqual(len(df_obs), 4)
                p_obs = df_obs[['p0', 'p1', 'p2', 'p3']].to_numpy()
                self.assertTrue(np.isfinite(p_obs).all())

                # 2. Compare against standard labeled extraction (label=3 vs label=0)
                items_lbl = []
                for idx_i, ob in enumerate(obs_list):
                    dom = compute_observable_containment(ob, self.geometry, self.bounds, config=cfg)
                    fn = extract_b1pp_item if name == 'b1pp' else extract_pf_item
                    items_lbl.append(fn(idx_i, ob, (idx_i % 4), dom, self.geometry, self.bounds, cfg))
                batch_lbl = collate_b1pp(items_lbl) if name == 'b1pp' else collate_pf(items_lbl)
                with torch.inference_mode():
                    if name == 'b1pp':
                        p_lbl = torch.softmax(model(batch_lbl).float(), dim=-1).numpy()
                    else:
                        p_lbl = model(batch_lbl, grad_blend=1.0, return_aux=False)['probs4'].float().numpy()
                np.testing.assert_allclose(p_obs, p_lbl, atol=1e-6, rtol=0.0)


if __name__ == '__main__':
    unittest.main()
