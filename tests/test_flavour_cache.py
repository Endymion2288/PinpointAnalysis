"""Check actual Parquet staging, LRU eviction and DataLoader process ownership."""

import multiprocessing as mp
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, get_worker_info

from analysis.flavour.datasets import CachedNativeDataset, ShardBatchSampler, collate_native


def cache_collate(events):
    info = get_worker_info()
    return collate_native(events), info.dataset.cache_stats()


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.paths = []
        for shard, count in enumerate([3, 2, 4]):
            path = self.root / f'source{shard}'
            path.mkdir()
            observations = []
            for i in range(count):
                empty = i == 0
                observations.append({
                    'pixel': {'channel_id': [] if empty else [100 * shard + i],
                              'xyz_mm': [] if empty else [[shard, i, .5]]},
                    'bar': {'channel_id': [], 'bounds_mm': [], 'edep_proxy_MeV': [], 'pe_proxy': []},
                    'tracker_hypothesis': {'station': [], 'xy_mm': [], 'sigma_xy_mm': []},
                    'has_response': [not empty, False, False]})
            pq.write_table(pa.Table.from_pylist(observations), path / 'observations.parquet')
            pq.write_table(pa.table({'label': [shard] * count,
                                    'label_status': ['unresolved_metadata'] + ['consistent'] * (count-1),
                                    'initE': [999.] * count}), path / 'targets.parquet')
            (path / 'provenance.parquet').write_text('must not be staged')
            self.paths.append(path)

    def tearDown(self):
        self.temp.cleanup()

    def dataset(self, capacity=1):
        return CachedNativeDataset(self.paths, cache_dir=self.root / 'scratch', max_shards=capacity)

    def test_lru_boundaries_values_and_cleanup(self):
        with self.dataset() as dataset:
            self.assertEqual(len(dataset), 9)
            self.assertEqual(dataset.cache_stats()['misses'], 0)
            self.assertIsNone(dataset.cache_stats()['root'])
            first = dataset[1]
            self.assertEqual(set(first['targets']), {'label', 'label_status'})
            np.testing.assert_array_equal(first['observations']['pixel']['channel_id'], [1])
            np.testing.assert_allclose(first['observations']['pixel']['xyz_mm'], [[0, 1, .5]])
            self.assertEqual(dataset[3]['targets']['label'], 1)
            self.assertEqual(dataset[-1]['targets']['label'], 2)
            self.assertEqual(dataset[0]['observations']['has_response'].tolist(), [False] * 3)
            self.assertEqual(dataset[1]['targets']['label'], 0)
            stats = dataset.cache_stats()
            root = Path(stats['root'])
            self.assertEqual(stats['resident_shards'], 1)
            self.assertEqual(stats['misses'], 4)
            self.assertEqual(stats['hits'], 1)
            self.assertEqual(stats['evictions'], 3)
            self.assertEqual(sorted(p.name for p in (root / '0').iterdir()), list(dataset.FILES))
            self.assertEqual(len(list(root.iterdir())), 1)
            self.assertEqual(stats['resident_bytes'], sum(p.stat().st_size for p in (root / '0').iterdir()))
            for bad in [-10, 9]:
                with self.assertRaises(IndexError):
                    dataset[bad]
            # Earlier event views survive eviction until the current batch releases them.
            np.testing.assert_array_equal(first['observations']['pixel']['channel_id'], [1])
        self.assertFalse(root.exists())
        self.assertEqual(dataset.cache_stats()['resident_bytes'], 0)
        self.assertTrue((self.paths[0] / 'observations.parquet').exists())

    def test_shuffle_coverage_reproducibility_and_class_window(self):
        with self.dataset(3) as dataset:
            sampler = ShardBatchSampler(dataset, 4, window_shards=3, seed=7)
            first = list(sampler)
            self.assertEqual(len(first), len(sampler))
            self.assertEqual(sorted(sum(first, [])), list(range(9)))
            self.assertEqual(first, list(sampler))
            sampler.set_epoch(1)
            self.assertNotEqual(first, list(sampler))
            self.assertEqual(sorted(sum(list(sampler), [])), list(range(9)))
            batch = collate_native([dataset[i] for i in range(9)])
            self.assertEqual(batch['observations']['pixel']['offsets'].tolist(), [0, 0, 1, 2, 2, 3, 3, 4, 5, 6])
            self.assertEqual(batch['observations']['pixel']['channel_id'].tolist(), [1, 2, 101, 201, 202, 203])
            self.assertEqual(batch['targets']['label_status'].count('unresolved_metadata'), 3)
            self.assertNotIn('label_status', batch['observations'])
            with self.assertRaises(ValueError):
                ShardBatchSampler(dataset, 4, window_shards=4)

    def test_spawn_pickle_has_no_parent_cache(self):
        with self.dataset() as dataset:
            dataset[1]
            parent = Path(dataset.cache_stats()['root'])
            restored = pickle.loads(pickle.dumps(dataset))
            self.assertIsNone(restored.cache_stats()['root'])
            with restored:
                restored[1]
                self.assertNotEqual(restored.cache_stats()['root'], str(parent))
            self.assertTrue(parent.exists())

    def test_worker_cache_isolation_and_exit_cleanup(self):
        # fork tests inherited parent ownership; spawn tests serialization.
        for context in ['fork', 'spawn']:
            if context not in mp.get_all_start_methods():
                continue
            with self.subTest(context=context), self.dataset(3) as dataset:
                dataset[1]
                parent_root = Path(dataset.cache_stats()['root'])
                loader = DataLoader(dataset, batch_size=2, num_workers=2, collate_fn=cache_collate,
                                    multiprocessing_context=context)
                roots, labels = set(), []
                for batch, stats in loader:
                    roots.add(stats['root'])
                    labels.extend(batch['targets']['label'].tolist())
                    self.assertLessEqual(stats['resident_shards'], 3)
                self.assertEqual(labels, [0, 0, 0, 1, 1, 2, 2, 2, 2])
                self.assertEqual(len(roots), 2)
                self.assertNotIn(str(parent_root), roots)
                self.assertTrue(parent_root.exists())
                self.assertTrue(all(not Path(p).exists() for p in roots))

    def test_mismatched_rows_and_remote_scratch_rejected(self):
        pq.write_table(pa.table({'label': [0]}), self.paths[0] / 'targets.parquet')
        with self.assertRaisesRegex(ValueError, 'length mismatch'):
            self.dataset()
        with self.assertRaisesRegex(ValueError, 'local scratch'):
            CachedNativeDataset(self.paths, cache_dir='/eos/user/scratch')


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
