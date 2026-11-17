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

from analysis.flavour.v3.v5_confirm.freeze import require_v5_freeze
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


if __name__ == '__main__':
    unittest.main()
