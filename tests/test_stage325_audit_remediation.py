"""Stage 325 unit tests closing Stage 324 independent scientific audit counterexamples."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from analysis.flavour.v3.common.datasets import V3EvalDataset
from analysis.flavour.v3.common.domains import compute_observable_containment, detector_bounds
from analysis.flavour.v3.common.fast_kernels import get_fast_lib, reconstruct_micron_vertex
from analysis.flavour.v3.v1_b1pp.spectrometer import (
    TRACKER_Z_MM,
    check_magnet12_bore_traversal,
    extract_spectrometer_kinematics,
)
from analysis.flavour.v3.v2_pinpointformer.pixel_tokenizer import extract_pf_item
from analysis.flavour.v3.v3_multitask.losses import (
    FixedWeightEventLoss,
    compute_window_task_denominators,
    focal_charm_bce,
)


CLEAN_V3_GEOM = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v6_clean_v3_2026-11-19/clean_sim/geometry.json')


def _dummy_geom() -> dict:
    if CLEAN_V3_GEOM.is_file():
        return json.loads(CLEAN_V3_GEOM.read_text())
    from tests.test_stage323_remediation import _synthetic_geometry
    return _synthetic_geometry()


class Stage325AuditRemediationTests(unittest.TestCase):
    def test_324_01_sealed_shard_zero_access(self) -> None:
        """With split_shards.json, training/validation must NEVER open or stat sealed shards."""
        with tempfile.TemporaryDirectory(prefix='test-324-01-') as td:
            root = Path(td)
            geom = _dummy_geom()
            (root / 'geometry.json').write_text(json.dumps(geom))
            (root / 'plan.json').write_text(json.dumps({'geometry': str(root / 'geometry.json')}))
            for sname, sp_val in (('000', 'train'), ('001', 'validation'), ('100', 'confirm')):
                sdir = root / 'shards' / sname
                sdir.mkdir(parents=True)
                obs_tbl = pa.Table.from_pylist([{
                    'pixel': {'channel_id': [], 'xyz_mm': []},
                    'bar': {'panel': [], 'bar': [], 'horizontal': [], 'channel_id': [], 'layer': [], 'bounds_mm': [], 'pe_proxy': [], 'hit_time_ns': []},
                    'tracker_hypothesis': {'station': [], 'xy_mm': [], 'sigma_xy_mm': []},
                }])
                tgt_tbl = pa.Table.from_pylist([{'label': 0}])
                pq.write_table(obs_tbl, sdir / 'observations.parquet')
                pq.write_table(tgt_tbl, sdir / 'targets.parquet')
                pd.DataFrame([{
                    'run': 32300, 'chunk': int(sname), 'row': 0,
                    'family_hi': np.uint64(int(sname) + 1), 'family_lo': np.uint64(1),
                    'split': sp_val, 'label': 0,
                }]).to_parquet(sdir / 'provenance.parquet', index=False)

            (root / 'split_shards.json').write_text(json.dumps({
                'train': ['000'],
                'validation': ['001'],
                'confirm': ['100'],
            }))
            sealed_dir = root / 'shards' / '100'
            sealed_hits: list[tuple[str, str]] = []
            sealed_str = str(sealed_dir)

            def _audit(event: str, args: tuple) -> None:
                if event in ('open', 'os.scandir', 'os.listdir', 'os.stat'):
                    for a in args:
                        if isinstance(a, (str, bytes, os.PathLike)):
                            s = os.fsdecode(a)
                            if sealed_str in s:
                                sealed_hits.append((event, s))

            sys.addaudithook(_audit)
            os.chmod(sealed_dir, 0o000)
            try:
                ds_tr = V3EvalDataset(root, split='train')
                ds_va = V3EvalDataset(root, split='validation')
                self.assertEqual(len(ds_tr), 1)
                self.assertEqual(len(ds_va), 1)
                ds_tr.native.close()
                ds_va.native.close()
            finally:
                os.chmod(sealed_dir, 0o755)
            self.assertEqual(sealed_hits, [], f'Sealed shard accessed: {sealed_hits}')

    def test_324_02_exact_grad_accum_equivalence(self) -> None:
        """Full-window denominator accumulation must match single-batch gradients to <1e-5."""
        # 1. Exact Stage 324 4-item focal charm counterexample
        lg_full = torch.tensor([-2.0, 0.0, 0.0, 2.0], requires_grad=True)
        y = torch.tensor([1, 1, 0, 0])
        v = torch.tensor([1, 1, 1, 1])
        loss_full = focal_charm_bce(lg_full, y, v, gamma=1.5, pos_weight=3.0)
        loss_full.backward()
        g_full = lg_full.grad.detach().clone()

        lg_acc = torch.tensor([-2.0, 0.0, 0.0, 2.0], requires_grad=True)
        d_full = float(torch.where(y.float() > 0.5, 3.0, 1.0).sum().item())
        l0 = focal_charm_bce(lg_acc[:2], y[:2], v[:2], gamma=1.5, pos_weight=3.0, denom_override=d_full)
        l1 = focal_charm_bce(lg_acc[2:], y[2:], v[2:], gamma=1.5, pos_weight=3.0, denom_override=d_full)
        (l0 + l1).backward()
        g_acc = lg_acc.grad.detach().clone()
        self.assertLess(float((g_full - g_acc).abs().max().item()), 1e-6)

        # 2. Full 128-event 4-task masked + sample-weighted AdamW equivalence
        rng = torch.Generator().manual_seed(325)
        n = 128
        for w_all in (False, True):
            loss_mod = FixedWeightEventLoss(weight_all_tasks=w_all)
            x_in = torch.randn(n, 16, generator=rng)
            label_7 = torch.randint(0, 7, (n,), generator=rng)
            label_4 = torch.where(label_7 == 0, 0, torch.where(label_7 <= 2, 1, torch.where(label_7 <= 5, 2, 3)))
            # Extreme uneven masks across first/second 64-event halves
            charm_flag = torch.cat([torch.ones(64, dtype=torch.long), torch.zeros(64, dtype=torch.long)])
            charm_valid = torch.cat([torch.ones(50, dtype=torch.long), torch.zeros(14, dtype=torch.long), torch.ones(64, dtype=torch.long)])
            vtx_true = torch.randn(n, 3, generator=rng)
            vtx_valid = torch.cat([torch.ones(64, dtype=torch.long), torch.ones(30, dtype=torch.long), torch.zeros(34, dtype=torch.long)])
            kin_true = torch.randn(n, 4, generator=rng)
            kin_valid = (label_4 != 3).long()
            sw = torch.cat([torch.full((64,), 2.7), torch.full((64,), 0.4)])

            def _make_head():
                torch.manual_seed(1234)
                return torch.nn.Linear(16, 7 + 1 + 3 + 3 + 4)

            m_full = _make_head()
            m_acc = _make_head()
            o_full = torch.optim.AdamW(m_full.parameters(), lr=1e-2)
            o_acc = torch.optim.AdamW(m_acc.parameters(), lr=1e-2)

            full_batch = {
                'label': label_4, 'label_7': label_7, 'charm_flag': charm_flag, 'charm_valid': charm_valid,
                'vertex_xyz_mm': vtx_true, 'vertex_valid': vtx_valid, 'vis_kinematics': kin_true,
                'kin_valid': kin_valid, 'sample_weight': sw,
            }
            out_f = m_full(x_in)
            lf, _ = loss_mod(out_f[:, :7], out_f[:, 7], out_f[:, 8:11], out_f[:, 11:14], out_f[:, 14:18], full_batch)
            o_full.zero_grad()
            lf.backward()
            grad_full = m_full.weight.grad.detach().clone()
            o_full.step()

            mb0 = {k: v[:64] for k, v in full_batch.items()}
            mb1 = {k: v[64:] for k, v in full_batch.items()}
            fd = compute_window_task_denominators([mb0, mb1], pos_weight=3.0, weight_all_tasks=w_all)
            o_acc.zero_grad()
            for sl, mb in ((slice(0, 64), mb0), (slice(64, 128), mb1)):
                oa = m_acc(x_in[sl])
                la, _ = loss_mod(oa[:, :7], oa[:, 7], oa[:, 8:11], oa[:, 11:14], oa[:, 14:18], mb, fixed_denoms=fd)
                la.backward()
            grad_acc = m_acc.weight.grad.detach().clone()
            o_acc.step()

            rel_g = float((grad_full - grad_acc).abs().max().item() / max(float(grad_full.abs().max().item()), 1e-8))
            max_p = float((m_full.weight.detach() - m_acc.weight.detach()).abs().max().item())
            self.assertLess(rel_g, 1e-5, f'w_all={w_all}: rel_g={rel_g}')
            self.assertLess(max_p, 1e-5, f'w_all={w_all}: max_p={max_p}')

    def test_324_06_bore_aware_spectrometer_rejects_shell_muons(self) -> None:
        """On-axis in-bore mu+ keeps valid sagitta; off-axis x=112mm shell track is flagged & momentum-zeroed."""
        bounds = {'x_pix_center': 0.0, 'y_scint_center': -12.0}
        # Track A (on-axis, x=0, y=-10, 100 GeV mu+: s_y = -1.008 mm)
        # Track B (Stage 324 counterexample: x=112 mm outside R=100mm bore, wrong-sign s_y = +0.593 mm)
        obs = {
            'station': [0, 1, 2, 0, 1, 2],
            'xy_mm': [
                [0.0, -10.0], [0.0, -11.008], [0.0, -10.0],
                [112.0, -10.0], [112.0, -9.407], [112.0, -10.0],
            ],
        }
        res = extract_spectrometer_kinematics(obs, (0.0, 0.0, 0.0, False), bounds)
        self.assertTrue(bool(res['cand_mask'][0]))
        self.assertTrue(bool(res['cand_mask'][1]))
        # Slot 0 must be the genuine in-bore track (cand[0,0]==1.0, non-zero sagitta)
        self.assertAlmostEqual(float(res['cand'][0, 0]), 1.0, places=5)
        self.assertGreater(abs(float(res['cand'][0, 6])), 0.1)
        self.assertGreater(float(res['cand'][0, 13]), 0.1)
        # Slot 1 is the shell-scattered x=112mm track: flagged 0.70 with zeroed sagitta/momentum
        self.assertAlmostEqual(float(res['cand'][1, 0]), 0.70, places=5)
        self.assertEqual(float(res['cand'][1, 6]), 0.0)
        self.assertEqual(float(res['cand'][1, 13]), 0.0)

        # Direct traversal checker verification
        ib, mr = check_magnet12_bore_traversal(
            np.array([0.0, 112.0]), np.array([-10.0, -10.0]), np.array([-11.008, -9.407]),
            np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([1.7e-6, -1.0e-6]),
        )
        self.assertTrue(bool(ib[0]))
        self.assertFalse(bool(ib[1]))
        self.assertGreater(float(mr[1]), 110.0)

    def test_324_07_micro_vertex_works_at_both_30mm_and_150mm_spacing(self) -> None:
        """Exact Stage 324-07 counterexample: 3 tracks must reconstruct PV & displacement at BOTH 30mm and 150mm spacing."""
        self.assertIsNotNone(get_fast_lib(), 'C++17 fast kernel library must compile and load')
        for spacing in (30.0, 150.0):
            zs = [spacing * k for k in range(4)]
            # 3 straight tracks originating from PV = (0, 0, -5.0) mm
            slopes = [(0.05, 0.02), (-0.04, 0.03), (0.01, -0.06)]
            def _run(disp_x: float) -> tuple[np.ndarray, bool, np.ndarray]:
                pls, xs, ys, z_arr = [], [], [], []
                for p, zp in enumerate(zs):
                    for ti, (tx, ty) in enumerate(slopes):
                        off_x = disp_x if ti == 2 else 0.0
                        pls.append(p)
                        xs.append(off_x + tx * (zp - (-5.0)))
                        ys.append(ty * (zp - (-5.0)))
                        z_arr.append(zp)
                n = len(pls)
                return reconstruct_micron_vertex(
                    np.array(pls, dtype=np.int16),
                    np.array(xs, dtype=np.float64),
                    np.array(ys, dtype=np.float64),
                    np.array(z_arr, dtype=np.float64),
                    np.full(n, 2.0, dtype=np.float64),
                    np.full(n, 1e-4, dtype=np.float64),
                    4, 0.0208, 0.0228, (0.0, 0.0, 0.0, False), (0.0, 0.0, 0.0),
                )
            pv0, ok0, f0 = _run(0.0)
            pv1, ok1, f1 = _run(0.20)
            self.assertTrue(ok0, f'PV fit failed at spacing={spacing} mm (disp=0)')
            self.assertTrue(ok1, f'PV fit failed at spacing={spacing} mm (disp=0.20)')
            self.assertLess(abs(float(pv0[2]) - (-5.0)), 1.0, f'spacing={spacing} pv0={pv0}')
            self.assertGreater(float(f1[4]), float(f0[4]) + 0.05, f'spacing={spacing}: IP proxy did not respond to 0.20 mm shift')

    def test_324_08_fixed_clustering_coord_only_quantization(self) -> None:
        """quantize_cluster_coords_only_mm must keep exact cluster/token counts, c_obs, and bit-exact IPT."""
        geom = _dummy_geom()
        bounds = detector_bounds(geom)
        nx, ny = int(bounds['nx']), int(bounds['ny'])
        # 2 target hits on plane 0 & 1, plus 2 IPT hits on plane 12 & 13
        cids = [0 * ny * nx + 4000 * nx + 6000, 1 * ny * nx + 4002 * nx + 6005,
                12 * ny * nx + 4100 * nx + 6100, 13 * ny * nx + 4105 * nx + 6108]
        xyzs = [[0.12, -12.34, bounds['pixel_Zpos'][0]], [0.22, -12.30, bounds['pixel_Zpos'][1]],
                [2.20, -10.05, bounds['pixel_Zpos'][12]], [2.36, -9.94, bounds['pixel_Zpos'][13]]]
        obs = {
            'pixel': {'channel_id': cids, 'xyz_mm': xyzs},
            'bar': {'panel': [], 'bar': [], 'horizontal': [], 'channel_id': [], 'layer': [], 'bounds_mm': [], 'pe_proxy': [], 'hit_time_ns': []},
            'tracker_hypothesis': {'station': [], 'xy_mm': [], 'sigma_xy_mm': []},
        }
        dom = compute_observable_containment(obs, geom, bounds, config={'pe_only': True})
        it_nat = extract_pf_item(0, obs, None, dom, geom, bounds, {'pe_only': True})
        it_q = extract_pf_item(0, obs, None, dom, geom, bounds, {'pe_only': True, 'quantize_cluster_coords_only_mm': 1.00})
        self.assertEqual(len(it_nat['pix_plane']), len(it_q['pix_plane']))
        self.assertTrue(np.array_equal(it_nat['c_obs'], it_q['c_obs']))
        self.assertEqual(int(it_nat['obs_domain_id']), int(it_q['obs_domain_id']))
        ipt_m = it_nat['pix_plane'] >= 12
        self.assertTrue(np.array_equal(it_nat['pix_xyz_mm'][ipt_m], it_q['pix_xyz_mm'][ipt_m]))
        self.assertTrue(np.array_equal(it_nat['pix_feat'][ipt_m], it_q['pix_feat'][ipt_m]))
        tgt_m = it_nat['pix_plane'] < 12
        self.assertGreater(float(np.abs(it_nat['pix_xyz_mm'][tgt_m, :2] - it_q['pix_xyz_mm'][tgt_m, :2]).max()), 0.05)


if __name__ == '__main__':
    unittest.main()
