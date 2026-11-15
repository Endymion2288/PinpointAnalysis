"""R0 family exclusion, blind identities and exact observation cache checks."""

import json
from pathlib import Path
import pickle
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from analysis.flavour.s3.models_repair import build_model
from analysis.flavour.v2.data import PackedPlaneDataset, TimedPlaneDataset, build_packed_cache, verify_cache
from analysis.flavour.v2.metadata import representation_metadata
from analysis.flavour.v2.partitions import audit_partitions, collect_usage, equal_identity_rows, family_index, make_partitions, row_keys
from analysis.flavour.v2.submit import choose_schedd
from analysis.flavour.v2.fit import fit_event_positions


class PartitionTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([{'run': label+1, 'chunk': 0, 'row': row, 'label': label,
                              'family_hi': label+1, 'family_lo': row, 'old_split': 'train',
                              'usable_join': True, 'label_status': 'consistent'}
                             for label in range(4) for row in range(40)])

    def test_historical_response_replica_excludes_whole_family_and_sets_are_nested(self):
        frame = self.frame()
        replica = frame.iloc[[0]].copy()
        replica['row'] = 100
        frame = pd.concat([frame, replica], ignore_index=True)
        masks = np.zeros(len(frame), dtype='uint8')
        masks[-1] = 1
        arguments = (frame, masks, 17, [16, 32], {'confirm': 2, 'cut_b': 2, 'cal_a': 2, 'tune': 2})
        parts, ledger, summary = make_partitions(*arguments)
        self.assertEqual(summary['historically_used_rows'], 2)
        self.assertEqual(summary['historically_used_families'], 1)
        for rows in parts.values():
            self.assertFalse(family_index(rows).isin(family_index(replica)).any())
        self.assertTrue(all(audit_partitions(parts, ledger).values()))
        repeated, _, _ = make_partitions(*arguments)
        for name in parts:
            pd.testing.assert_frame_equal(parts[name], repeated[name])
        broken = {**parts, 'tune': parts['confirm'].copy()}
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            audit_partitions(broken, ledger)
        broken = {**parts, 'tune': parts['tune'].query('label != 3')}
        with self.assertRaisesRegex(ValueError, 'balanced'):
            audit_partitions(broken, ledger)

    def test_usage_preserves_combined_roles_and_rejects_unresolved_or_wrong_family(self):
        frame = self.frame()
        old = pd.concat([frame.iloc[[0]], frame.iloc[[0]]], ignore_index=True)
        old['split'] = ['train', 'validation']
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'manual.parquet'
            old.to_parquet(path)
            usage, _ = collect_usage(frame, [path])
            self.assertEqual(usage[0], 3)
            old['family_lo'] = 999
            old.to_parquet(path)
            with self.assertRaisesRegex(ValueError, 'family mismatch'):
                collect_usage(frame, [path])
            old['run'] = 999
            old.to_parquet(path)
            with self.assertRaisesRegex(ValueError, 'absent from inventory'):
                collect_usage(frame, [path])

    def test_row_packing_rejects_overflow_instead_of_colliding(self):
        frame = self.frame()
        self.assertEqual(len(np.unique(row_keys(frame))), len(frame))
        for name, value in [('run', 2**16), ('chunk', 2**28), ('row', 2**20), ('row', -1)]:
            with self.subTest(name=name, value=value):
                bad = frame.copy()
                bad.loc[0, name] = value
                with self.assertRaises(ValueError):
                    row_keys(bad)

    def test_uint64_family_identity_comparison_keeps_low_bits(self):
        left = self.frame().iloc[:2].copy()
        left['family_hi'] = np.array([2**63+17, 2**63+18], dtype='uint64')
        right = left.copy()
        right.loc[right.index[0], 'family_hi'] = np.uint64(2**63+19)
        self.assertTrue(equal_identity_rows(left, left.iloc[::-1]))
        self.assertFalse(equal_identity_rows(left, right))

    def test_schedd_negative_eos_clause_is_not_treated_as_eos_routing(self):
        snapshot = 'bigbird21.cern.ch 10 0 0\nbigbird20.cern.ch 0 1 0\n'
        replies = [SimpleNamespace(returncode=0, stdout='!regexp("^/eos", Iwd)'),
                   SimpleNamespace(returncode=0, stdout='True'),
                   SimpleNamespace(returncode=0, stdout='NoEos, ValidRequestCpus'),
                   SimpleNamespace(returncode=0, stdout='WRITE ALLOW')]
        with patch('analysis.flavour.v2.submit.subprocess.check_output', return_value=snapshot), \
             patch('analysis.flavour.v2.submit.subprocess.run', side_effect=replies):
            name, _, checks = choose_schedd(Path('/unused'))
        self.assertEqual(name, 'bigbird20.cern.ch')
        self.assertFalse(checks[0]['compatible'])

    def test_schedd_all_eos_policy_rejects_afs_executable_layout(self):
        snapshot = 'bigbird24.cern.ch 10 0 0\nbigbird20.cern.ch 0 1 0\n'
        replies = [SimpleNamespace(returncode=0, stdout='True'),
                   SimpleNamespace(returncode=0, stdout='NoEos, AllEos, AbsPathInEos'),
                   SimpleNamespace(returncode=0, stdout='True'),
                   SimpleNamespace(returncode=0, stdout='NoEos, ValidRequestCpus'),
                   SimpleNamespace(returncode=0, stdout='WRITE ALLOW')]
        with patch('analysis.flavour.v2.submit.subprocess.check_output', return_value=snapshot), \
             patch('analysis.flavour.v2.submit.subprocess.run', side_effect=replies):
            name, _, checks = choose_schedd(Path('/unused'))
        self.assertEqual(name, 'bigbird20.cern.ch')
        self.assertIn('EOS-only', checks[0]['reason'])

    def test_fit_maps_event_keys_instead_of_sample_list_ordinals(self):
        selected = self.frame().iloc[[0, 40, 80, 120]]
        metadata = selected.iloc[[2, 0, 3, 1]].reset_index(drop=True)
        positions = fit_event_positions(selected, metadata)
        self.assertEqual(positions.tolist(), [1, 3, 0, 2])
        broken = metadata.copy()
        broken.loc[0, 'label'] = 0
        with self.assertRaisesRegex(ValueError, 'labels differ'):
            fit_event_positions(selected, broken)


class RepresentationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.geometry = {'pixel_Xpos': list(np.arange(100)*.0208),
                         'pixel_Ypos': list(np.arange(100)*.0228),
                         'pixel_Zpos': [0., 10.], 'layer_is_pixel': [1, 0, 1]}
        self.config = {'modalities': ['pixel', 'bar', 'tracker'], 'window_pixels': 16,
                       'dim': 16, 'latents': 8, 'region_queries': 2, 'region_blocks': 8,
                       'empty_response_bypass': True, 'cache_shards': 1}

    def test_metadata_distinguishes_latent_mixing_from_explicit_geometry(self):
        model = build_model(self.config)
        facts = representation_metadata(model, self.geometry, self.config, plane_windows=True)
        np.testing.assert_allclose(facts['pixel_window_mm'], [.3328, .3648])
        self.assertEqual(facts['explicit_geometry_context_layers'], 0)
        self.assertFalse(facts['explicit_cross_plane_geometry'])
        self.assertEqual(facts['global_latent_attention_layers'], 2)
        self.assertEqual(facts['global_latents'], 8)
        self.assertEqual(facts['queries_per_region_or_module'], 2)
        self.assertIn('tracker_hypothesis', facts['used_observation_fields'])
        self.assertEqual(facts['target_or_provenance_features'], [])
        fixed = build_model({**self.config, 'compression': 'fixed'})
        facts = representation_metadata(fixed, self.geometry, self.config, plane_windows=True)
        self.assertEqual(facts['explicit_geometry_context_layers'], 2)
        self.assertTrue(facts['explicit_cross_plane_geometry'])

    def test_cache_preserves_empty_event_masks_channels_labels_and_detects_damage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            geometry = root/'geometry.json'
            geometry.write_text(json.dumps(self.geometry))
            (root/'plan.json').write_text(json.dumps({'geometry': str(geometry)}))
            shard = root/'shards/000'
            shard.mkdir(parents=True)
            observations = []
            for i in range(5):
                empty = i == 0
                observations.append({
                    'pixel': {'channel_id': [] if empty else [101, 202],
                              'xyz_mm': [] if empty else [[.0208, .0228, 0.], [.0416, .0456, 10.]]},
                    'bar': {'bounds_mm': [] if empty else [[0., 1., 0., 420., 5.]],
                            'horizontal': [] if empty else [False],
                            'edep_proxy_MeV': [] if empty else [3.+i],
                            'pe_proxy': [] if empty else [100+i],
                            'panel': [] if empty else [0], 'layer': [] if empty else [1]},
                    'tracker_hypothesis': {'station': [] if empty else [0],
                                           'xy_mm': [] if empty else [[.1, .2]],
                                           'sigma_xy_mm': [] if empty else [[.01, .01]]}})
            pq.write_table(pa.Table.from_pylist(observations), shard/'observations.parquet')
            pq.write_table(pa.table({'label': [i % 4 for i in range(5)],
                                    'label_status': ['consistent']*5}), shard/'targets.parquet')
            pd.DataFrame({'run': [1]*5, 'chunk': [0]*5, 'row': range(5),
                          'geometry_id': ['test']*5, 'label': [i % 4 for i in range(5)],
                          'split': ['train']*5}).to_parquet(shard/'provenance.parquet')
            native = TimedPlaneDataset(root, 'train', self.config, cache_dir=root/'scratch')
            try:
                result = build_packed_cache(native, root/'packed')
                self.assertEqual(result['original_canonical_array_readbacks'], 5)
                packed = PackedPlaneDataset(root, 'train', self.config, cache_dir=root/'scratch', packed_root=root/'packed')
                self.assertTrue(verify_cache(native, packed)['all_collated_tensors_bitwise_equal'])
                restored = pickle.loads(pickle.dumps(packed))
                self.assertEqual(len(restored._packed), 0)
                with self.assertRaisesRegex(ValueError, 'cache key mismatch'):
                    PackedPlaneDataset(root, 'train', {**self.config, 'window_pixels': 32}, packed_root=root/'packed')
                provenance = pd.read_parquet(shard/'provenance.parquet')
                changed = provenance.copy()
                changed.loc[0, 'row'] = 999
                changed.to_parquet(shard/'provenance.parquet')
                with self.assertRaisesRegex(ValueError, 'event identity mismatch'):
                    PackedPlaneDataset(root, 'train', self.config, packed_root=root/'packed')
                provenance.to_parquet(shard/'provenance.parquet')
                file = root/'packed/000/nodes.npy'
                damaged = np.load(file).copy()
                damaged[1, 15] = 0.
                packed._packed.clear()
                np.save(file, damaged)
                with self.assertRaisesRegex(ValueError, 'tensor differs'):
                    verify_cache(native, packed)
            finally:
                native.native.close()


if __name__ == '__main__':
    unittest.main()
