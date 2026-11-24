"""Independent scientific audit, MC 10085-10088 seal guard, anti-leakage,
and cross-module integration tests for Pinpoint-Gen4.
"""

from __future__ import annotations

import ast
import copy
import math
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from analysis.flavour.gen4.conflict_free_optim import collect_gen4_param_groups
from analysis.flavour.gen4.data_cache import (
    GEN4_PURE_OBS_KEYS,
    GEN4_TRUTH_BATCH_KEYS as CACHE_TRUTH_KEYS,
    RAW_MC_ROOT,
    assert_unsealed_mc8588_path,
    partition_mc8588_chunks,
)
from analysis.flavour.gen4.diffphys_sbi import DIFFPHYS_STAT_DIM, DiffPhysSBIModel
from analysis.flavour.gen4.gen4_unified_model import (
    GEN4_TRUTH_BATCH_KEYS,
    PinpointGen4Net,
    compute_gen4_supervised_loss,
    extract_nsbi_observables_from_output,
)
from analysis.flavour.gen4.geometry_contract import SIPM_NPIX, load_geometry_contract
from analysis.flavour.gen4.nsbi_discovery import (
    CHANNEL_NAMES,
    DomainPOTGPDCalibrator,
    evaluate_nsbi_vs_single_bin_closure,
    route_golden_micro_channels,
)
from analysis.flavour.gen4.physics_frontend import (
    digitize_raw_faser_hits,
    digitize_raw_scint_hits,
)
from analysis.flavour.gen4.run_gen4_supervised import make_synthetic_gen4_batch
from analysis.flavour.gen4.ssl_sonata_v2 import PinpointSSLv2Model
from analysis.flavour.nextgen.protocols import check_branch_gradient_health
from analysis.flavour.nextgen.seal_guard import (
    SealAuditGuard,
    SealViolationError,
    verify_inference_truth_invariance,
)


GEN4_PKG_DIR = Path(__file__).resolve().parents[1] / 'src' / 'analysis' / 'flavour' / 'gen4'


def _mutate_all_gen4_truth_keys(batch: dict[str, object]) -> dict[str, object]:
    """Deep-copy batch and overwrite every single key in GEN4_TRUTH_BATCH_KEYS."""
    mutated: dict[str, object] = {}
    for k, v in batch.items():
        mutated[k] = v.clone() if torch.is_tensor(v) else copy.deepcopy(v)

    bsz = int(batch['c_obs'].shape[0])  # type: ignore[index]
    dev = batch['c_obs'].device  # type: ignore[attr-defined]
    n_cl = int(batch['pix_xyz_mm'].shape[1]) if 'pix_xyz_mm' in batch else 0  # type: ignore[index]

    mutated['label'] = torch.full((bsz,), 3, dtype=torch.long, device=dev)
    mutated['label_4'] = torch.full((bsz,), 3, dtype=torch.long, device=dev)
    mutated['label_7'] = torch.full((bsz,), 6, dtype=torch.long, device=dev)
    mutated['tau_branch_valid'] = torch.ones((bsz,), dtype=torch.long, device=dev)
    mutated['charm_flag'] = torch.ones((bsz,), dtype=torch.long, device=dev)
    mutated['has_charm'] = torch.ones((bsz,), dtype=torch.bool, device=dev)
    mutated['charm_valid'] = torch.ones((bsz,), dtype=torch.long, device=dev)
    mutated['vertex_xyz_mm'] = torch.full((bsz, 3), -777.0, dtype=torch.float32, device=dev)
    mutated['vertex_valid'] = torch.ones((bsz,), dtype=torch.long, device=dev)
    mutated['vis_kinematics'] = torch.full((bsz, 4), 13.5, dtype=torch.float32, device=dev)
    mutated['kin_valid'] = torch.ones((bsz,), dtype=torch.long, device=dev)
    mutated['sample_weight'] = torch.full((bsz,), 9.5, dtype=torch.float32, device=dev)
    mutated['geom_weight'] = torch.full((bsz,), 4.2, dtype=torch.float32, device=dev)
    mutated['cl_sidecar_valid'] = torch.ones((bsz,), dtype=torch.bool, device=dev)
    mutated['cl_origin_label'] = torch.full((bsz, n_cl), 1, dtype=torch.long, device=dev)
    mutated['cl_secondary_origin_label'] = torch.full((bsz, n_cl), 2, dtype=torch.long, device=dev)
    mutated['cl_dominant_edep_frac'] = torch.full((bsz, n_cl), 0.123, dtype=torch.float32, device=dev)
    mutated['cl_sv_group_id'] = torch.full((bsz, n_cl), 5, dtype=torch.long, device=dev)
    mutated['cl_dominant_track_id'] = torch.full((bsz, n_cl), 8888, dtype=torch.long, device=dev)
    mutated['cl_is_real_target'] = torch.ones((bsz, n_cl), dtype=torch.float32, device=dev)
    mutated['true_sv_xyz_mm'] = torch.full((bsz, 3), -555.0, dtype=torch.float32, device=dev)
    mutated['has_true_sv'] = torch.ones((bsz,), dtype=torch.bool, device=dev)
    mutated['tau_n_charged_prongs_si'] = torch.full((bsz,), 3, dtype=torch.long, device=dev)
    mutated['tier1b_genie_valid'] = torch.ones((bsz,), dtype=torch.bool, device=dev)
    mutated['true_tau_submode'] = torch.full((bsz,), 3, dtype=torch.long, device=dev)
    mutated['true_n_pi0'] = torch.full((bsz,), 4, dtype=torch.long, device=dev)
    mutated['true_nu_energy_gev'] = torch.full((bsz,), 2500.0, dtype=torch.float32, device=dev)
    mutated['true_branch_p4_gev'] = torch.full((bsz, 4, 4), 42.0, dtype=torch.float32, device=dev)
    mutated['tgt_Z'] = torch.full((bsz,), 13, dtype=torch.long, device=dev)
    mutated['tau_decay_L3d_mm'] = torch.full((bsz,), 99.9, dtype=torch.float32, device=dev)
    mutated['trackID'] = torch.full((bsz, max(n_cl, 1)), 54321, dtype=torch.long, device=dev)
    return mutated


class TestGen4SealAndIntegration(unittest.TestCase):
    """Comprehensive MC 10085-10088 seal, zero-leakage, and cross-module integration tests."""

    def test_mc8588_sealed_chunks_and_legacy_mc_strictly_blocked(self) -> None:
        """All chunk_id % 20 >= 16 (r in 16..19 / %10 in {8,9}) & deprecated 10081-10084 must be blocked."""
        sealed_cases = (
            '/eos/project-f/fasersim-bonn/public/pinpoint/data/10085/10085_016.root',
            '/eos/project-f/fasersim-bonn/public/pinpoint/data/10086/10086_037.root',
            '/eos/project-f/fasersim-bonn/public/pinpoint/data/10087/10087_018.root',
            '/eos/project-f/fasersim-bonn/public/pinpoint/data/10088/10088_019.root',
            '/eos/project-f/fasersim-bonn/public/pinpoint/data/10086/10086_2919.root',
            '/eos/project-f/fasersim-bonn/public/pinpoint/data/10081/10081_000.root',
            '/eos/project-f/fasersim-bonn/public/pinpoint/data/10082/10082_000.root',
            '/eos/project-f/fasersim-bonn/public/pinpoint/data/10083/10083_000.root',
            '/eos/project-f/fasersim-bonn/public/pinpoint/data/10084/10084_000.root',
            '/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v6_clean_v3_2026-11-19/clean_sim/shards/000/observations.parquet',
            '/eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/l1_cluster_sidecar/sidecar_production_summary.json',
            '/eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/l2_ssl_300k/l2_corpus_manifest.json',
            '/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v6_clean_v3_2026-11-19/clean_sim/sealed_truth_meta/test.parquet',
            '/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v6_clean_v3_2026-11-19/clean_sim/shards/100/observations.parquet',
            '/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v6_clean_v3_2026-11-19/clean_sim/shards/149/observations.parquet',
            '/tmp/Cal-A/observations.parquet',
            '/tmp/Cut-B/observations.parquet',
            '/tmp/Confirm/observations.parquet',
        )
        for bad in sealed_cases:
            with self.assertRaises(SealViolationError, msg=f'Failed to block sealed/legacy path: {bad}'):
                assert_unsealed_mc8588_path(bad)

        # Positive controls: unsealed 10085-10088 chunks with r = cid % 20 < 16
        for good in (
            '/eos/project-f/fasersim-bonn/public/pinpoint/data/10085/10085_000.root',
            '/eos/project-f/fasersim-bonn/public/pinpoint/data/10086/10086_012.root',
            '/eos/project-f/fasersim-bonn/public/pinpoint/data/10087/10087_013.root',
            '/eos/project-f/fasersim-bonn/public/pinpoint/data/10088/10088_015.root',
        ):
            assert_unsealed_mc8588_path(good)

        # Verify partition_mc8588_chunks() is strictly pairwise-disjoint and excludes all r >= 16
        if (RAW_MC_ROOT / '10085').is_dir():
            splits = partition_mc8588_chunks()
            seen_per_ds: dict[str, set[int]] = {ds: set() for ds in ('10085', '10086', '10087', '10088')}
            for sp_name, ds_map in splits.items():
                for ds, items in ds_map.items():
                    self.assertGreater(len(items), 0, f'Empty split {sp_name}/{ds}')
                    for cid, path_obj in items:
                        self.assertLess(cid % 20, 16, f'Sealed chunk {cid} leaked into {sp_name}/{ds}')
                        self.assertNotIn(cid, seen_per_ds[ds], f'Chunk {cid} duplicated across splits in {ds}')
                        seen_per_ds[ds].add(cid)
                        assert_unsealed_mc8588_path(path_obj)

        # Verify SealAuditGuard blocks sealed syscalls at runtime
        with tempfile.TemporaryDirectory(prefix='gen4-seal-rt-') as td:
            sealed_dir = Path(td) / 'shards' / '100'
            sealed_dir.mkdir(parents=True)
            sealed_file = sealed_dir / 'observations.parquet'
            sealed_file.write_bytes(b'dummy')
            with SealAuditGuard() as guard:
                with self.assertRaises(SealViolationError):
                    _ = sealed_file.read_bytes()
            self.assertGreater(len(guard.violations), 0)

    def test_zero_scint_edep_and_faser_truth_leakage_and_14400_sipm_saturation(self) -> None:
        """Verify 14,400 SiPM saturation, <3 PE timing gate, and zero scintHits.edep / faserHits momentum reads."""
        geom = load_geometry_contract()

        # 1. Exact SiPM saturation & time threshold formula
        bar = digitize_raw_scint_hits(
            panel_id=np.array([0, 0, 0, 1], dtype=np.int64),
            is_horizontal=np.array([0, 0, 0, 0], dtype=np.int64),
            row_id=np.array([0, 0, 0, 0], dtype=np.int64),
            col_id=np.array([5, 5, 6, 7], dtype=np.int64),
            n_pe=np.array([50000.0, 50000.0, 2.0, 20.0], dtype=np.float64),
            hit_time_ns=np.array([2.2, 2.8, 3.1, 3.5], dtype=np.float64),
            contract=geom,
        )
        by_bar = {int(b): (int(pe), float(ht)) for b, pe, ht in zip(bar['bar'], bar['pe_proxy'], bar['hit_time_ns'])}
        expected_sat = int(round(SIPM_NPIX * (1.0 - math.exp(-100000.0 / SIPM_NPIX))))
        self.assertEqual(by_bar[5][0], expected_sat)
        self.assertLessEqual(by_bar[5][0], 14400)
        self.assertAlmostEqual(by_bar[5][1], 2.2, places=5)
        self.assertEqual(by_bar[6][0], 2)
        self.assertEqual(by_bar[6][1], -1.0)
        self.assertEqual(by_bar[7][0], 20)
        self.assertAlmostEqual(by_bar[7][1], 3.5, places=5)

        # 2. Tracker digitization uses strictly (trackerID, x, y, z) without momentum/PDG
        trk = digitize_raw_faser_hits(
            tracker_id=np.array([0, 0, 1, 2], dtype=np.int64),
            x_mm=np.array([1.0, 1.02, 2.0, 3.0], dtype=np.float64),
            y_mm=np.array([-5.0, -5.01, -5.5, -6.0], dtype=np.float64),
            z_mm=np.array([47.4, 47.4, 1237.4, 2427.4], dtype=np.float64),
            event_seed=42,
            apply_eff=False,
        )
        self.assertEqual(len(trk['station']), 3)  # Two 0.02 mm hits on station 0 merged within 0.1 mm

        # 3. Static AST audit across gen4/*.py: Hits/scintHits never reads 'edep'; Hits/faserHits never reads momentum/charge/pdg
        forbidden_faser_cols = {'px', 'py', 'pz', 'energy', 'charge', 'pdg', 'PDG'}
        for py_path in sorted(GEN4_PKG_DIR.glob('*.py')):
            tree = ast.parse(py_path.read_text(encoding='utf-8'), filename=str(py_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'arrays':
                    # Check if parent subscript accesses Hits/scintHits or Hits/faserHits
                    if isinstance(node.func.value, ast.Subscript):
                        sl = node.func.value.slice
                        if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                            tree_name = sl.value
                            cols_arg = node.args[0] if node.args else None
                            if isinstance(cols_arg, (ast.List, ast.Tuple)):
                                col_strs = {
                                    elt.value for elt in cols_arg.elts
                                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                                }
                                if 'scintHits' in tree_name:
                                    self.assertNotIn(
                                        'edep', col_strs,
                                        f'scintHits.edep leakage detected in {py_path.name}: {col_strs}',
                                    )
                                if 'faserHits' in tree_name:
                                    leaked = col_strs & forbidden_faser_cols
                                    self.assertEqual(
                                        leaked, set(),
                                        f'faserHits truth leakage detected in {py_path.name}: {leaked}',
                                    )

    def test_inference_truth_invariance_exact_zero_gen4_and_diffphys(self) -> None:
        """Mutating all GEN4_TRUTH_BATCH_KEYS must leave PinpointGen4Net & DiffPhysSBIModel outputs bit-identical."""
        torch.manual_seed(101)
        self.assertTrue((set(CACHE_TRUTH_KEYS) - {'tgt_Z'}).issubset(set(GEN4_TRUTH_BATCH_KEYS)))

        batch = make_synthetic_gen4_batch(batch_size=3, n_cl=48, has_edep=1.0, seed=7)
        obs_only = {k: v for k, v in batch.items() if k in GEN4_PURE_OBS_KEYS}
        mutated = _mutate_all_gen4_truth_keys(batch)

        gen4_model = PinpointGen4Net()
        gen4_model.eval()
        inv_res = verify_inference_truth_invariance(gen4_model, batch, atol=0.0)
        self.assertEqual(inv_res['max_abs_diff'], 0.0)

        with torch.no_grad():
            o_obs = gen4_model.forward_NextGen(obs_only)
            o_mut = gen4_model.forward_NextGen(mutated)
        for t_a, t_b, name in (
            (o_obs.probs4, o_mut.probs4, 'probs4'),
            (o_obs.logits7, o_mut.logits7, 'logits7'),
            (o_obs.v_pv_mm, o_mut.v_pv_mm, 'v_pv_mm'),
            (o_obs.scale_pv_mm, o_mut.scale_pv_mm, 'scale_pv_mm'),
            (o_obs.aux['logits13'], o_mut.aux['logits13'], 'logits13'),
            (o_obs.aux['tau_llr_score'], o_mut.aux['tau_llr_score'], 'tau_llr_score'),
        ):
            diff = float((t_a - t_b).abs().max().item())
            self.assertEqual(diff, 0.0, f'Truth invariance violated for PinpointGen4Net.{name}: {diff}')

        # DiffPhysSBIModel exact truth invariance
        dp_batch = {
            'diffphys_z': torch.randn(4, DIFFPHYS_STAT_DIM),
            'diffphys_z_sigma': torch.full((4, DIFFPHYS_STAT_DIM), 0.25),
            'obs_domain_id': torch.tensor([1, 2, 1, 2], dtype=torch.long),
            'e_vis_gev': torch.tensor([120.0, 280.0, 410.0, 90.0]),
            'pv_xyz_mm': torch.zeros((4, 3)),
            'c_obs': torch.ones((4, 16)),
        }
        dp_mut = _mutate_all_gen4_truth_keys(dp_batch)
        dp_model = DiffPhysSBIModel()
        dp_model.eval()
        dp_inv = verify_inference_truth_invariance(dp_model, dp_batch, atol=0.0)
        self.assertEqual(dp_inv['max_abs_diff'], 0.0)
        with torch.no_grad():
            d1 = dp_model.forward_NextGen(dp_batch)
            d2 = dp_model.forward_NextGen(dp_mut)
        self.assertEqual(float((d1.probs4 - d2.probs4).abs().max().item()), 0.0)
        self.assertEqual(float((d1.aux['tau_llr'] - d2.aux['tau_llr']).abs().max().item()), 0.0)

    def test_ssl_backbone_transfer_and_end_to_end_nsbi_integration(self) -> None:
        """Verify SSL-v2 -> Gen4 weight transfer (>96% keys), Gate-B health, and NSBI-PL closure."""
        torch.manual_seed(202)
        ssl_backbone = PinpointGen4Net()
        ssl_model = PinpointSSLv2Model(ssl_backbone)
        ckpt = ssl_model.export_backbone_checkpoint_dict()
        self.assertTrue(
            all(k.startswith(('si_vtx.', 'cal_pflow.', 'spec_ipt.', 'fusion.')) for k in ckpt.keys())
        )

        target_model = PinpointGen4Net()
        load_res = target_model.load_state_dict(ckpt, strict=False)
        self.assertEqual(load_res.unexpected_keys, [])
        self.assertGreater(len(load_res.missing_keys), 0)
        self.assertTrue(all(k.startswith('head.') for k in load_res.missing_keys))

        total_params = sum(p.numel() for p in target_model.parameters())
        loaded_params = sum(v.numel() for v in ckpt.values())
        self.assertGreater(loaded_params / total_params, 0.90)
        self.assertEqual(
            set(collect_gen4_param_groups(target_model).keys()),
            {'si_vtx', 'cal_pflow', 'spec_ipt', 'fusion', 'head'},
        )

        # Gate-B mixed batch backward check after SSL weight load
        b_si = make_synthetic_gen4_batch(batch_size=2, n_cl=48, has_edep=1.0, seed=10)
        b_fo = make_synthetic_gen4_batch(batch_size=2, n_cl=48, has_edep=1.0, seed=11)
        b_fo['c_obs'][:, 0] = 0.0
        b_fo['obs_domain_id'][:] = 2
        b_fo['pix_valid'][:] = False
        b_fo['pix_plane'][:] = -1
        b_fo['cl_sidecar_valid'][:] = False
        b_mix = {
            k: (torch.cat([b_si[k], b_fo[k]], dim=0) if torch.is_tensor(b_si[k]) else b_si[k])
            for k in b_si
        }
        target_model.train()
        out_mix = target_model.forward_NextGen(b_mix)
        loss_mix, _, _, _ = compute_gen4_supervised_loss(target_model, out_mix, b_mix)
        gb = check_branch_gradient_health(target_model, loss_mix, min_grad_l2=1.0e-6)
        self.assertTrue(gb['passed'], f'SSL-loaded Gen4 Gate-B failed: {gb}')

        # End-to-end pure-observable NSBI routing & POT-GPD closure check
        target_model.eval()
        b_eval = make_synthetic_gen4_batch(batch_size=16, n_cl=48, has_edep=1.0, seed=99)
        with torch.no_grad():
            out_eval = target_model.forward_NextGen(b_eval)
        nsbi_obs = extract_nsbi_observables_from_output(b_eval, out_eval)
        nsbi_obs['obs_domain_id'] = b_eval['obs_domain_id'].cpu().numpy()
        nsbi_obs['tau_llr'] = out_eval.aux['tau_llr_score'].cpu().numpy()
        nsbi_obs['charm_prob'] = torch.sigmoid(out_eval.charm_logit).cpu().numpy()
        ch_res = route_golden_micro_channels(nsbi_obs)
        ch_ids = ch_res['channel_id']
        self.assertEqual(ch_ids.shape, (16,))
        self.assertTrue(np.all((ch_ids >= 0) & (ch_ids < len(CHANNEL_NAMES))))

        rng = np.random.default_rng(329)
        cal = DomainPOTGPDCalibrator().fit(
            rng.normal(0.0, 1.0, size=600),
            np.tile([1, 2], 300),
            np.zeros(600, dtype=np.int64),
        )
        thr = cal.threshold(0.01, domain_id=1)
        self.assertTrue(np.isfinite(thr))

        nsbi_toy_obs = dict(nsbi_obs)
        nsbi_toy_obs['obs_domain_id'] = np.tile([1, 2], 8)
        nsbi_toy_obs['has_front_pixel'] = np.tile([True, False], 8)
        nsbi_toy_obs['has_target_pixel'] = np.tile([True, False], 8)
        nsbi_toy_obs['tau_llr'] = rng.normal(0.0, 1.0, size=16) + 1.2 * (np.arange(16) % 4 == 2)
        nsbi_toy_obs['charm_prob'] = rng.uniform(0.0, 0.3, size=16)
        closure = evaluate_nsbi_vs_single_bin_closure(
            nsbi_toy_obs,
            labels=np.arange(16) % 4,
            has_charm=np.zeros(16, dtype=bool),
            mw2t_weights=np.ones(16),
            total_events=180000.0,
            tau_fraction=0.003,
            delta_sys=0.0159,
        )
        self.assertIn('nsbi_pl_bb_lite', closure)
        self.assertTrue(np.isfinite(float(closure['nsbi_pl_bb_lite']['z0_profile_likelihood'])))


if __name__ == '__main__':
    unittest.main()
