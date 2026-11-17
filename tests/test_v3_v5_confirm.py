"""Unit tests for Phase V5 pre-unsealing freeze guard, zero-retune Cal-A/Cut-B/Confirm flow, and Profile-Likelihood."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.flavour.v3.common.domains import detector_bounds
from analysis.flavour.v3.v5_confirm.freeze import require_v5_freeze
from analysis.flavour.v3.v5_confirm.infer_unsealed import _StressVariantDataset
from analysis.flavour.v3.v5_confirm.significance import (
    apply_frozen_pipeline,
    fit_cal_a_calibrators,
    lock_cut_b_operating_points,
    profile_likelihood_significance,
)


def _synthetic_split(n: int, seed: int, n_fams: int = 80) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 4, size=n, dtype=np.int64)
    obs_dom = rng.choice([0, 1, 2, 3], size=n, p=[0.05, 0.30, 0.45, 0.20]).astype(np.int64)
    fams = rng.integers(0, n_fams, size=n, dtype=np.int64)

    ref = pd.DataFrame({
        'run': 10081 + labels,
        'chunk': np.zeros(n, dtype=np.int64),
        'row': np.arange(n, dtype=np.int64),
        'family_hi': np.zeros(n, dtype=np.int64),
        'family_lo': fams,
        'label': labels,
        'obs_domain_id': obs_dom,
    })

    def _mk_pred(signal_strength: float, noise_scale: float) -> pd.DataFrame:
        logits = rng.normal(0.0, noise_scale, size=(n, 4))
        logits[np.arange(n), labels] += signal_strength
        logits[obs_dom == 0] = 0.0
        ex = np.exp(logits - logits.max(axis=1, keepdims=True))
        p = ex / ex.sum(axis=1, keepdims=True)
        df = ref[['run', 'chunk', 'row']].copy()
        for c in range(4):
            df[f'p_{c}'] = p[:, c]
        df['charm_prob'] = np.where(labels == 1, rng.uniform(0.1, 0.7, size=n), rng.uniform(0.0, 0.3, size=n))
        return df

    return ref, {
        'm_strong': _mk_pred(1.8, 0.7),
        'm_weak': _mk_pred(0.9, 0.8),
    }


class TestV5Confirm(unittest.TestCase):
    def test_v5_modules_zero_direct_sealed_paths(self) -> None:
        v5_dir = Path(__file__).resolve().parents[1] / 'src/analysis/flavour/v3/v5_confirm'
        for p in sorted(v5_dir.glob('*.py')):
            if p.name == 'prepare_unsealed.py':
                continue
            src = p.read_text()
            tree = ast.parse(src, filename=str(p))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    self.assertNotIn('confirm_rows.parquet', node.value, msg=str(p))
                    self.assertNotIn('sidecar_root', node.value, msg=str(p))

    def test_freeze_guard_blocks_unfrozen_or_invalid_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fp = Path(tmp) / 'freeze.json'
            with self.assertRaises(FileNotFoundError):
                require_v5_freeze(fp, verify_git_commit=False, verify_checkpoints=False)

            bad = {
                'status': 'draft',
                'calibration_and_threshold_rules_frozen': False,
                'pre_unsealing_git_commit': None,
                'members': [],
            }
            fp.write_text(json.dumps(bad))
            with self.assertRaises(RuntimeError):
                require_v5_freeze(fp, verify_git_commit=False, verify_checkpoints=False)

            repo = Path(__file__).resolve().parents[1]
            head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
            good = {
                'status': 'frozen',
                'calibration_and_threshold_rules_frozen': True,
                'pre_unsealing_git_commit': head,
                'members': [{'checkpoint_path': str(fp)} for _ in range(15)],
            }
            fp.write_text(json.dumps(good))
            loaded = require_v5_freeze(fp, repo_root=repo, verify_git_commit=True, verify_checkpoints=True)
            self.assertEqual(loaded['pre_unsealing_git_commit'], head)

    def test_cal_a_cut_b_confirm_zero_retune_and_empty_domain(self) -> None:
        ref_cal, mem_cal = _synthetic_split(800, seed=101)
        ref_cut, mem_cut = _synthetic_split(1200, seed=102)
        ref_conf, mem_conf = _synthetic_split(1500, seed=103)

        cal_params = fit_cal_a_calibrators(mem_cal, ref_cal, member_names=['m_strong', 'm_weak'])
        cal_snapshot = json.dumps(cal_params, sort_keys=True)
        cut_locks = lock_cut_b_operating_points(mem_cut, ref_cut, cal_params)
        cut_snapshot = json.dumps(cut_locks, sort_keys=True)

        applied = apply_frozen_pipeline(mem_conf, ref_conf, cal_params, cut_locks)
        # Ensure fitting/locking data structures remain identical (zero Confirm refit)
        self.assertEqual(cal_snapshot, json.dumps(cal_params, sort_keys=True))
        self.assertEqual(cut_snapshot, json.dumps(cut_locks, sort_keys=True))

        # Verify D_empty strictly maps to uniform [0.25]*4 and s_tau == -25.0 and passes == False
        empty = ref_conf['obs_domain_id'].to_numpy() == 0
        self.assertTrue(empty.any())
        self.assertTrue(np.allclose(applied['probs'][empty], 0.25, atol=1e-7))
        self.assertTrue(np.allclose(applied['s_tau'][empty], -25.0, atol=1e-7))
        self.assertFalse(applied['pass_global_1pct'][empty].any())
        self.assertFalse(applied['pass_np_1pct'][empty].any())

    def test_profile_likelihood_monotonicity_under_systematics(self) -> None:
        rng = np.random.default_rng(77)
        n = 6000
        labels = rng.choice([0, 1, 2, 3], size=n, p=[0.28, 0.40, 0.04, 0.28])
        bin_ids = rng.integers(0, 18, size=n, dtype=np.int64)
        # Make top bins of each domain enriched in tau signal
        tau_m = labels == 2
        bin_ids[tau_m] = rng.choice([4, 5, 10, 11, 16, 17], size=int(tau_m.sum()))

        r0 = profile_likelihood_significance(
            bin_ids, labels, sigma_bkg_norm=0.0, sigma_dom_norm=0.0, delta_bkg_sys=np.zeros(18)
        )
        r_mid = profile_likelihood_significance(
            bin_ids, labels, sigma_bkg_norm=0.05, sigma_dom_norm=0.05, delta_bkg_sys=np.full(18, 0.03)
        )
        r_hi = profile_likelihood_significance(
            bin_ids, labels, sigma_bkg_norm=0.15, sigma_dom_norm=0.15, delta_bkg_sys=np.full(18, 0.12)
        )
        self.assertAlmostEqual(r0['z0_profile_likelihood'], r0['z0_stat_only'], places=5)
        self.assertLessEqual(r_mid['z0_profile_likelihood'], r0['z0_stat_only'] + 1e-6)
        self.assertLessEqual(r_hi['z0_profile_likelihood'], r_mid['z0_profile_likelihood'] + 1e-6)
        self.assertGreater(r_hi['z0_profile_likelihood'], 0.0)

    def test_stress_dataset_ideal_tracker_fallback_and_order(self) -> None:
        geometry = {
            'pixel_Xpos': [-10.0, 0.0, 10.0],
            'pixel_Ypos': [-22.0, -12.0, -2.0],
            'pixel_Zpos': [-3000.0, -2990.0],
            'scint_bar_Xcenter': [-2.0, 2.0],
            'scint_bar_Ycenter': [-14.0, -10.0],
            'scint_Zpos': [-2900.0, -2895.0, -2890.0],
            'tungsten_Zpos': [-3000.0, -2950.0],
            'tungsten_thickness': 5.0,
            'scint_thickness': 5.0,
            'silicon_thickness': 50.0,
            'nPinpointBlocks': 1,
            'nFortuneBlocks': 1,
            'numScintLayers': 1,
            'num_scint_panels_layer': 2,
            'scint_detector_width': 100.0,
            'scint_detector_height': 100.0,
            'scint_bar_width': 5.0,
            'scint_bar_height': 5.0,
            'scint_bar_thickness': 2.0,
            'tracker_Zpos': [-2700.0, -2600.0, -2500.0],
        }
        bounds = detector_bounds(geometry)
        hw_map = {'triples_layer_panel_horizontal': [[0, 0, 0], [0, 1, 1]]}
        variants = [
            {'name': 'nominal', 'parameters': {}},
            {'name': 'tracker_degrade', 'parameters': {'tracker_efficiency': 0.5}},
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / 'data_confirm'
            staged_root = root / 'stage_confirm'
            shard_defs = [
                ('0000', [(10001, 0, 0, 0), (10001, 0, 1, 1)]),
                ('0001', [(10002, 0, 0, 2), (10002, 0, 1, 3)]),
            ]
            for sname, rows in shard_defs:
                sdir = data_root / 'shards' / sname
                gdir = staged_root / sname
                sdir.mkdir(parents=True)
                gdir.mkdir(parents=True)
                obs_rows, tgt_rows, prov_rows, ideal_rows = [], [], [], []
                for run_id, ch_id, r_id, lbl in rows:
                    obs_rows.append({
                        'pixel': {'channel_id': [10 + r_id], 'xyz_mm': [[float(r_id), -12.0, -2995.0]]},
                        'bar': {'channel_id': [], 'layer': [], 'panel': [], 'horizontal': [], 'bar': [], 'bounds_mm': [], 'edep_proxy_MeV': [], 'pe_proxy': []},
                        'tracker_hypothesis': {'station': [0], 'xy_mm': [[0.5 + run_id - 10001, -12.0]], 'sigma_xy_mm': [[0.05, 0.05]]},
                        'has_response': [True, False, True],
                        'outside_acceptance': [],
                        'fit_attempted': True,
                        'momentum_censored': 0,
                    })
                    tgt_rows.append({'label': lbl, 'label_status': 'consistent'})
                    prov_rows.append({
                        'run': run_id, 'chunk': ch_id, 'row': r_id,
                        'family_hi': 0, 'family_lo': run_id, 'split': 'confirm', 'label': lbl,
                    })
                    ideal_rows.append({'station': [0], 'xyz_mm': [[0.5 + run_id - 10001, -12.0, -2700.0]]})

                pq.write_table(pa.Table.from_pylist(obs_rows), sdir / 'observations.parquet')
                pq.write_table(pa.Table.from_pylist(tgt_rows), sdir / 'targets.parquet')
                pd.DataFrame(prov_rows).to_parquet(sdir / 'provenance.parquet', index=False)
                pq.write_table(pa.Table.from_pylist(ideal_rows), sdir / 'ideal_tracker.parquet')
                # Stage omits ideal_tracker.parquet (matching stage_shards behavior)
                for fname in ('observations.parquet', 'targets.parquet', 'provenance.parquet'):
                    (gdir / fname).write_bytes((sdir / fname).read_bytes())

            # Request events in cross-shard alternating order: shard 1 -> shard 0 -> shard 1
            stress_manifest = pd.DataFrame({
                'run': [10002, 10001, 10002],
                'chunk': [0, 0, 0],
                'row': [1, 0, 0],
            })
            ds = _StressVariantDataset(
                data_root,
                stress_manifest,
                variants,
                geometry,
                bounds,
                hw_map,
                family='b1pp',
                staged_dir=staged_root,
            )
            self.assertEqual(len(ds), 6)
            self.assertEqual([(m['run'], m['row']) for m in ds.base_meta], [(10002, 1), (10001, 0), (10002, 0)])
            self.assertEqual(ds.base_ideal[0]['xyz_mm'][0][0], 1.5)
            self.assertEqual(ds.base_ideal[1]['xyz_mm'][0][0], 0.5)
            item0 = ds[0]
            self.assertEqual(item0['label'], 3)


if __name__ == '__main__':
    unittest.main()
