"""Unit and provenance tests for Stage 328 L2 300k Pure-W SSL Corpus production."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow.parquet as pq

from analysis.flavour.nextgen.produce_l2_ssl_corpus import (
    CLEAN_V3_ROOT,
    DEFAULT_L2_ROOT,
    FORBIDDEN_TRUTH_KEYS,
    OBS_CFG,
    SSL_SHARD_KEYS,
    install_l2_seal_guard,
    pack_ssl_csr_zst,
    reconstruct_pix_feat22,
    unpack_ssl_csr_zst,
)
from analysis.flavour.nextgen.dual_regime_modalnet import extract_nonexclusive_spec_candidates
from analysis.flavour.nextgen.seal_guard import SealViolationError
from analysis.flavour.nextgen.tracklet_ipformer import extract_tracklet_pf_item
from analysis.flavour.v3.common.domains import compute_observable_containment, detector_bounds


class TestNextGenL2Corpus(unittest.TestCase):
    """Verify L2 seal provenance, zero truth columns, compact shard size, and 22d reconstruction."""

    def test_l2_seal_guard_blocks_sealed_and_early_genie(self) -> None:
        guard = install_l2_seal_guard()
        try:
            blocked_paths = [
                CLEAN_V3_ROOT / 'planned_manifest.parquet',
                CLEAN_V3_ROOT / 'sealed_truth_meta/cal_a_truth_meta.parquet',
                CLEAN_V3_ROOT / 'data/shards/080/observations.parquet',
                CLEAN_V3_ROOT / 'data/shards/100/observations.parquet',
                Path('/eos/project-f/fasersim-bonn/public/pinpoint/data/10081/10081_000.root'),
                Path('/eos/project-f/fasersim-bonn/public/pinpoint/data/10083/10083_028.root'),
                Path('/eos/project-f/fasersim-bonn/public/pinpoint/data/10086/10086_015.root'),
            ]
            for bp in blocked_paths:
                with self.assertRaises(SealViolationError):
                    open(bp, 'rb')
            guard.violations.clear()
        finally:
            guard.__exit__(None, None, None)

    def test_pack_unpack_zero_truth_and_reconstruct_pix_feat22(self) -> None:
        sh0 = CLEAN_V3_ROOT / 'data/shards/000'
        if not (sh0 / 'observations.parquet').is_file():
            self.skipTest('Clean v3 unsealed shard 000 not accessible in this environment')
        geom = json.loads((CLEAN_V3_ROOT / 'geometry.json').read_text())
        bounds = detector_bounds(geom)
        obs_list = pq.read_table(sh0 / 'observations.parquet').slice(0, 40).to_pylist()

        items = []
        ref_pf22 = []
        for idx, ob in enumerate(obs_list):
            dom = compute_observable_containment(ob, geom, bounds, config=OBS_CFG)
            if int(dom['obs_domain_id']) not in (1, 2):
                continue
            it = extract_tracklet_pf_item(idx, ob, None, dom, geom, bounds, OBS_CFG)
            c8, c8m, st_cnt = extract_nonexclusive_spec_candidates(
                ob['tracker_hypothesis'], dom['cal_start'], bounds
            )
            pf = it['pix_feat']
            ref_pf22.append(pf.copy())
            pf_16 = (
                np.concatenate([pf[:, 3:16], pf[:, 19:22]], axis=1).astype(np.float16)
                if len(pf) > 0
                else np.empty((0, 16), dtype=np.float16)
            )
            keep_bar = np.asarray(it['bar_keep_mask'], dtype=bool)
            bar_pe = (
                np.asarray(ob['bar']['pe_proxy'], dtype=np.int32)[keep_bar]
                if len(keep_bar)
                else np.empty(0, dtype=np.int32)
            )
            items.append({
                'family_hi': np.uint64(idx + 1000000),
                'family_lo': np.uint64(idx * 13 + 7),
                'obs_domain_id': np.int8(dom['obs_domain_id']),
                'c_obs': it['c_obs'].astype(np.float16),
                'cal_start': it['cal_start'].astype(np.float32),
                'avr_pv_mm': it['avr_pv_mm'].astype(np.float32),
                'pix_xyz_mm': it['pix_xyz_mm'].astype(np.float32),
                'pix_feat_16': pf_16,
                'pix_plane': it['pix_plane'].astype(np.int8),
                'pix_plane_summary': it['pix_plane_summary'].astype(np.float32),
                'pix_patches': it['pix_patches'].astype(np.float16),
                'patch_cluster_idx': it['patch_cluster_idx'].astype(np.int16),
                'bar_panel': it['bar_panel'].astype(np.int16),
                'bar_horiz': it['bar_horiz'].astype(np.int8),
                'bar_idx': it['bar_idx'].astype(np.int16),
                'bar_pe': bar_pe,
                'bar_hit_time_ns': it['bar_hit_time_ns'].astype(np.float32),
                'trk_cand': c8.astype(np.float32),
                'trk_cand_mask': c8m.astype(bool),
                'trk_station_counts': st_cnt.astype(np.float32),
                'trk_hits': it['trk_hits'].astype(np.float32),
                'trk_kin': it['trk_kin'].astype(np.float32),
                'trk_morph': it['trk_morph'].astype(np.float16),
                'trk_mask': it['trk_mask'].astype(bool),
            })

        self.assertGreater(len(items), 15)

        # Truth injection must fail immediately
        bad_item = dict(items[0])
        bad_item['label'] = 2
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            with self.assertRaises(AssertionError):
                pack_ssl_csr_zst([bad_item], tdp / 'bad.npz.zst')

            out_p = tdp / 'ok.npz.zst'
            stats = pack_ssl_csr_zst(items, out_p, zstd_level=12)
            self.assertLess(stats['mb_per_2000_events'], 20.0)
            arrs = unpack_ssl_csr_zst(out_p)
            self.assertEqual(tuple(arrs.keys()), SSL_SHARD_KEYS)
            self.assertEqual(len(set(arrs.keys()) & FORBIDDEN_TRUTH_KEYS), 0)
            self.assertTrue(np.all(np.isin(arrs['obs_domain_id'], [1, 2])))

            # Verify pix_feat22 reconstruction fidelity
            pptr = arrs['pix_ptr']
            for k, ref in enumerate(ref_pf22):
                s, e = int(pptr[k]), int(pptr[k + 1])
                rec = reconstruct_pix_feat22(
                    arrs['pix_xyz_mm'][s:e],
                    arrs['pix_plane'][s:e],
                    arrs['pix_feat_16'][s:e],
                    arrs['pix_plane_summary'][k],
                    bounds,
                )
                if len(ref) > 0:
                    max_diff = float(np.max(np.abs(rec - ref)))
                    self.assertLess(max_diff, 5e-3)

    def test_l2_manifest_if_complete(self) -> None:
        man_path = DEFAULT_L2_ROOT / 'l2_corpus_manifest.json'
        if not man_path.is_file():
            self.skipTest('l2_corpus_manifest.json not finalized yet')
        man = json.loads(man_path.read_text())
        self.assertGreaterEqual(int(man['train_shards_summary']['n_contained']), 300000)
        self.assertGreaterEqual(int(man['ssl_val_calib_summary']['n_contained']), 10000)
        self.assertEqual(int(man['truth_columns_present']), 0)
        self.assertEqual(int(man['root_files_on_eos']), 0)
        self.assertEqual(int(man['family_overlap_with_train32k_and_6ood']), 0)
        self.assertEqual(int(man['train_vs_ssl_val_family_overlap']), 0)
        self.assertEqual(int(man['internal_duplicate_families']), 0)
        self.assertEqual(int(man['seal_violations']), 0)
        self.assertGreaterEqual(int(man['min_genie_chunk_index']), 29)
        self.assertLess(float(man['total_eos_gb']), 3.5)
        self.assertLess(float(man['train_shards_summary']['mean_mb_per_2000_events']), 20.0)


if __name__ == '__main__':
    unittest.main()
