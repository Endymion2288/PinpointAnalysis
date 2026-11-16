"""Static AST and runtime truth-isolation and sidecar alignment tests (W3-02)."""

import ast
import inspect
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from analysis.flavour.datasets import NativeShard
from analysis.flavour.v3.common import calibration, datasets, domains
from analysis.flavour.v3.common.pixel_clusters import cluster_pixel_hits
from analysis.flavour.v3.common.truth_sidecar import (
    CLUSTER_ORIGIN_NAMES,
    LABEL_7_NAMES,
    extract_shard_sidecar,
)


R2_DATA = Path('/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R2/data')
INPUT_ROOT = Path('/eos/project-f/fasersim-bonn/public/pinpoint/data')


class TestV3TruthIsolation(unittest.TestCase):
    def test_eval_dataset_signature_forbids_sidecar(self) -> None:
        sig = inspect.signature(datasets.V3EvalDataset.__init__)
        for param in sig.parameters:
            self.assertNotIn('sidecar', param.lower())
            self.assertNotIn('truth', param.lower())

        # AST check that datasets.py / domains.py / calibration.py never touch sealed or truth coordinate names in eval paths
        eval_src = inspect.getsource(datasets.V3EvalDataset)
        tree = ast.parse(eval_src)
        strings = {
            n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        }
        forbidden = {
            'sidecar', 'truth_sidecar', 'event_sidecar', 'cluster_sidecar',
            'initX', 'initY', 'initZ', 'initE', 'confirm_rows', 'cal_a', 'cut_b',
        }
        self.assertEqual(strings & forbidden, set())

        # Static import check: V3EvalDataset module must not import truth_sidecar
        mod_tree = ast.parse(Path(datasets.__file__).read_text())
        imported = set()
        for node in ast.walk(mod_tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name)
        self.assertFalse(any('truth_sidecar' in m for m in imported))

        # Check calibration AST has no truth vertex leakage
        cal_tree = ast.parse(Path(calibration.__file__).read_text()) if Path(calibration.__file__).exists() else None
        if cal_tree is not None:
            cal_str = {
                n.value for n in ast.walk(cal_tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)
            }
            self.assertFalse(any(x in cal_str for x in ('initX', 'initY', 'initZ', 'confirm_rows')))

    def test_eval_dataset_runtime_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ds = datasets.V3EvalDataset(R2_DATA, split='validation', cache_dir=tmp)
            self.assertEqual(len(ds), 40000)
            self.assertEqual(
                set(ds.metadata.columns),
                {'run', 'chunk', 'row', 'family_hi', 'family_lo', 'split', 'label'},
            )
            item = ds[0]
            self.assertIn('c_obs', item)
            self.assertIn('obs_domain_id', item)
            self.assertNotIn('label_7', item)
            self.assertNotIn('vertex_xyz_mm', item)
            self.assertNotIn('charm_flag', item)
            ds.native.close()

    def test_sidecar_extraction_real_shard_smoke(self) -> None:
        plan = json.loads((R2_DATA / 'plan.json').read_text())
        geometry = json.loads(Path(plan['geometry']).read_text())
        bounds = domains.detector_bounds(geometry)
        shard_path = R2_DATA / 'shards' / '000'
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / 'shard000'
            summary = extract_shard_sidecar(shard_path, INPUT_ROOT, geometry, bounds, out_dir)
            self.assertGreater(summary['events'], 2000)
            ev_df = pd.read_parquet(out_dir / 'event_sidecar.parquet')
            self.assertTrue(np.all(ev_df['energy_nu_GeV'] <= 5500.0))
            self.assertTrue(np.all(ev_df['energy_nu_GeV'] >= 5.0))
            tau_mask = ev_df['label_4'].to_numpy() == 2
            self.assertGreater(ev_df.loc[tau_mask, 'tau_branch_valid'].mean(), 0.95)
            self.assertGreater(ev_df['charm_valid'].mean(), 0.99)
            self.assertGreater(ev_df['vertex_valid'].mean(), 0.95)
            self.assertGreater(ev_df['kin_valid'].mean(), 0.50)

            z = np.load(out_dir / 'dense_sidecar.npz')
            # Verify 1-to-1 cluster_first_channel_id alignment with pixel_clusters
            ns = NativeShard(shard_path, target_columns=('label',))
            prov = pd.read_parquet(shard_path / 'provenance.parquet')
            train_idx = np.flatnonzero((prov['split'] == 'train').to_numpy())
            for local_i in range(min(25, len(train_idx))):
                obs = ns[int(train_idx[local_i])]['observations']
                cinfo = cluster_pixel_hits(
                    obs['pixel']['channel_id'],
                    obs['pixel']['xyz_mm'],
                    bounds['nx'],
                    bounds['ny'],
                )
                s, e = int(z['pix_offsets'][local_i]), int(z['pix_offsets'][local_i + 1])
                self.assertEqual(e - s, cinfo['n_clusters'])
                self.assertTrue(
                    np.array_equal(z['cluster_first_channel_id'][s:e], cinfo['cluster_first_channel_id'])
                )
                if e > s:
                    pix_rel = z['pixel_rel_target'][s:e].astype(np.float32)
                    self.assertTrue(np.allclose(pix_rel.sum(axis=1), 1.0, atol=1e-2))
                    self.assertTrue(np.all((z['cluster_origin'][s:e] >= 0) & (z['cluster_origin'][s:e] < len(CLUSTER_ORIGIN_NAMES))))
                bs, be = int(z['bar_offsets'][local_i]), int(z['bar_offsets'][local_i + 1])
                self.assertEqual(be - bs, len(obs['bar']['channel_id']))
                if be > bs:
                    bar_rel = z['bar_rel_target'][bs:be].astype(np.float32)
                    self.assertTrue(np.allclose(bar_rel.sum(axis=1), 1.0, atol=1e-2))


if __name__ == '__main__':
    unittest.main()
