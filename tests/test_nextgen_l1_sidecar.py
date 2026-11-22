"""Unit & data-contract tests for Stage 328 Experiment L1 cluster_sidecar producer."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from analysis.flavour.nextgen.produce_l1_cluster_sidecar import (
    DEFAULT_OUT_ROOT,
    _L1_FORBIDDEN_RE,
    _classify_event_clusters,
    install_l1_stage1_guard,
)
from analysis.flavour.nextgen.seal_guard import SealViolationError, assert_unsealed_path
from analysis.flavour.v3.common.domains import detector_bounds
from analysis.flavour.v3.v2_pinpointformer.pixel_tokenizer import (
    N_MAX_IPT_CLUSTERS,
    N_MAX_TARGET_CLUSTERS,
    _cluster_sensor_groups,
    _compute_plane_neighborhoods,
    _select_group_clusters,
)


class TestNextGenL1Sidecar(unittest.TestCase):
    def test_stage1_seal_guard_blocks_forbidden_paths(self) -> None:
        guard = install_l1_stage1_guard()
        try:
            forbidden = [
                '/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v6_clean_v3_2026-11-19/clean_sim/data/shards/080/observations.parquet',
                '/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v6_clean_v3_2026-11-19/clean_sim/data/shards/099/observations.parquet',
                '/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v6_clean_v3_2026-11-19/clean_sim/data/shards/100/observations.parquet',
                '/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v6_clean_v3_2026-11-19/clean_sim/data/shards/149/observations.parquet',
                '/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v6_clean_v3_2026-11-19/clean_sim/planned_manifest.parquet',
                '/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v6_clean_v3_2026-11-19/clean_sim/validation_meta/tune_truth_meta.parquet',
                '/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v6_clean_v3_2026-11-19/clean_sim/sealed_truth_meta/confirm_truth_meta.parquet',
                '/eos/project-f/fasersim-bonn/public/pinpoint/data/10081/10081_012.root',
                '/eos/project-f/fasersim-bonn/public/pinpoint/data/10081/10081_016.root',
                '/eos/project-f/fasersim-bonn/public/pinpoint/data/10086/10086_000.root',
                '/eos/project-f/fasersim-bonn/public/pinpoint/data/10086/10086_001.root',
                '/eos/project-f/fasersim-bonn/public/pinpoint/data/10086/10086_016.root',
                '/eos/project-f/fasersim-bonn/public/pinpoint/data/10086/10086_029.root',
            ]
            for fp in forbidden:
                with self.assertRaises((SealViolationError, PermissionError), msg=f'Failed to block {fp}'):
                    try:
                        assert_unsealed_path(fp)
                    except SealViolationError:
                        raise
                    if _L1_FORBIDDEN_RE.search(fp):
                        raise SealViolationError(fp)
                    open(fp, 'rb').close()

            allowed = [
                '/eos/project-f/fasersim-bonn/public/pinpoint/data/10081/10081_000.root',
                '/eos/project-f/fasersim-bonn/public/pinpoint/data/10083/10083_011.root',
                '/eos/project-f/fasersim-bonn/public/pinpoint/data/10086/10086_002.root',
                '/eos/project-f/fasersim-bonn/public/pinpoint/data/10086/10086_015.root',
            ]
            for fp in allowed:
                assert_unsealed_path(fp)
                self.assertIsNone(_L1_FORBIDDEN_RE.search(fp), msg=f'False positive block on {fp}')
        finally:
            guard.__exit__(None, None, None)

    def test_synthetic_top3_track_ranking_and_taxonomy(self) -> None:
        nx, ny = 100, 100
        pix_z = np.arange(15, dtype=np.float64) * 30.0 - 3000.0
        # Two adjacent pixels in plane 0 forming one merged cluster:
        # hit 0: e- conversion (tid=50, edep=0.6), hit 1: tau daughter (tid=2, edep=0.4)
        rows = np.array([50, 50], dtype=np.int64)
        cols = np.array([50, 51], dtype=np.int64)
        lays = np.array([0, 0], dtype=np.int64)
        pix_cid = (lays * ny + rows) * nx + cols
        pix_xyz = np.array([[0.0, 0.0, pix_z[0]], [0.0208, 0.0, pix_z[0]]], dtype=np.float64)
        p_raw = {
            'rowID': rows,
            'colID': cols,
            'layerID': lays,
            'pdg': np.array([11, 211], dtype=np.int64),
            'trackID': np.array([50, 2], dtype=np.int64),
            'parentID': np.array([10, 1], dtype=np.int64),
            'edep': np.array([0.6, 0.4], dtype=np.float64),
            'fromPrimaryLepton': np.array([0, 1], dtype=np.int64),
            'fromPrimaryEMShower': np.array([0, 0], dtype=np.int64),
            'fromCharmedHadron': np.array([0, 0], dtype=np.int64),
            'fromTau': np.array([0, 1], dtype=np.int64),
        }
        vtx = np.array([0.0, 0.0, -3005.0], dtype=np.float64)
        sv_xyz = np.array([0.05, 0.02, -3003.0], dtype=np.float64)
        tev = {'PID': np.array([1]), 'TID': np.array([2])}
        res = _classify_event_clusters(
            pix_cid, pix_xyz, p_raw, nx, ny, pix_z, vtx,
            None, tev, None, np.array([1]), np.empty(0, dtype=np.int64),
            sv_xyz, True,
        )
        self.assertEqual(len(res['origin_label']), 1)
        self.assertEqual(int(res['dominant_track_id'][0]), 1)
        self.assertEqual(int(res['second_track_id'][0]), 2)
        self.assertEqual(int(res['third_track_id'][0]), -1)
        self.assertEqual(int(res['origin_label'][0]), 3)            # photon_conv_ee dominant
        self.assertEqual(int(res['secondary_origin_label'][0]), 1)  # tau_decay tie-break (>=0.25 edep)
        self.assertEqual(int(res['sv_group_id'][0]), 1)             # tau SV group
        self.assertEqual(int(res['has_true_sv']), 1)

    def test_shard_000_schema_alignment_and_bounds(self) -> None:
        guard = install_l1_stage1_guard()
        try:
            sc_path = DEFAULT_OUT_ROOT / 'cluster_sidecar/shard_000.parquet'
            obs_path = DEFAULT_OUT_ROOT / 'data/shards/000/observations.parquet'
            prov_path = DEFAULT_OUT_ROOT / 'data/shards/000/provenance.parquet'
            geom_path = DEFAULT_OUT_ROOT / 'geometry.json'
            if not sc_path.is_file():
                self.skipTest(f'Shard 000 sidecar not yet produced at {sc_path}')

            import json
            geom = json.loads(geom_path.read_text())
            bounds = detector_bounds(geom)
            nx_pix, ny_pix = int(bounds['nx']), int(bounds['ny'])

            sc_tbl = pq.read_table(sc_path)
            expected_cols = {
                'run', 'chunk', 'row', 'family_hi', 'family_lo',
                'true_sv_xyz_mm', 'has_true_sv',
                'cl_plane', 'cl_x_mm', 'cl_y_mm', 'origin_label',
                'secondary_origin_label', 'sv_group_id',
                'dominant_track_id', 'second_track_id', 'third_track_id',
                'dominant_edep_frac',
            }
            self.assertTrue(expected_cols.issubset(set(sc_tbl.column_names)))

            sc_rows = sc_tbl.to_pylist()
            obs_rows = pq.read_table(obs_path).to_pylist()
            prov_df = pd.read_parquet(prov_path)
            self.assertEqual(len(sc_rows), 400)
            self.assertEqual(len(obs_rows), 400)
            self.assertEqual(len(prov_df), 400)

            max_dist_mm = 0.0
            for idx in range(400):
                scr = sc_rows[idx]
                ob = obs_rows[idx]
                self.assertEqual(int(scr['run']), int(prov_df['run'].iloc[idx]))
                self.assertEqual(int(scr['chunk']), int(prov_df['chunk'].iloc[idx]))
                self.assertEqual(int(scr['row']), int(prov_df['row'].iloc[idx]))
                self.assertEqual(int(scr['family_hi']), int(prov_df['family_hi'].iloc[idx]))
                self.assertEqual(int(scr['family_lo']), int(prov_df['family_lo'].iloc[idx]))
                self.assertIn(int(scr['has_true_sv']), (0, 1))

                orig = np.asarray(scr['origin_label'], dtype=np.int64)
                sec_orig = np.asarray(scr['secondary_origin_label'], dtype=np.int64)
                sv_g = np.asarray(scr['sv_group_id'], dtype=np.int64)
                frac = np.asarray(scr['dominant_edep_frac'], dtype=np.float64)
                if len(orig) > 0:
                    self.assertTrue(np.all((orig >= 0) & (orig <= 4)))
                    self.assertTrue(np.all((sec_orig >= 0) & (sec_orig <= 4)))
                    self.assertTrue(np.all(sv_g >= -1))
                    self.assertTrue(np.all((frac > 0.0) & (frac <= 1.0001)))

                cid = np.asarray(ob['pixel']['channel_id'], dtype=np.int64)
                xyz = np.asarray(ob['pixel']['xyz_mm'], dtype=np.float64).reshape(-1, 3)
                if len(cid) == 0:
                    self.assertEqual(len(scr['cl_x_mm']), 0)
                    continue
                cl = _cluster_sensor_groups(cid, xyz, nx_pix, ny_pix, 12, 0.06, 0.06)
                ord_c = cl['hit_order']
                off_c = cl['cluster_offsets']
                nh_c = np.diff(off_c).astype(np.float64)
                pl_c = cl['cluster_plane'].astype(np.int64)
                xm_c = np.add.reduceat(xyz[ord_c, 0], off_c[:-1]) / nh_c
                ym_c = np.add.reduceat(xyz[ord_c, 1], off_c[:-1]) / nh_c
                _, _, _, h040, _, cr5, _ = _compute_plane_neighborhoods(pl_c, xm_c, ym_c, nh_c)
                k_tgt = _select_group_clusters(np.flatnonzero(pl_c < 12), pl_c, nh_c, h040, cr5, N_MAX_TARGET_CLUSTERS)
                k_ipt = _select_group_clusters(np.flatnonzero(pl_c >= 12), pl_c, nh_c, h040, cr5, N_MAX_IPT_CLUSTERS)
                k_all = np.concatenate([k_tgt, k_ipt])
                self.assertEqual(len(scr['cl_x_mm']), len(k_all))
                self.assertTrue(np.array_equal(np.asarray(scr['cl_plane'], dtype=np.int64), pl_c[k_all]))
                d = np.hypot(np.asarray(scr['cl_x_mm'], dtype=np.float64) - xm_c[k_all],
                             np.asarray(scr['cl_y_mm'], dtype=np.float64) - ym_c[k_all])
                if len(d):
                    max_dist_mm = max(max_dist_mm, float(d.max()))
            self.assertLess(max_dist_mm, 0.050)
        finally:
            guard.__exit__(None, None, None)


if __name__ == '__main__':
    unittest.main()
