"""Shared compressed staging preserves native channels and canonical batches."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from analysis.flavour.s3.dataset import PlaneDataset, collate_planes
from analysis.flavour.s4.stage_inputs import stage_shards


class S4InputStagingTests(unittest.TestCase):
    def test_shared_native_source_preserves_tensors_and_bounded_reader_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'shards').mkdir()
            geometry = {'pixel_Xpos': (np.arange(256)*.02-2.56).tolist(),
                'pixel_Ypos': (np.arange(256)*.02-2.56).tolist(), 'pixel_Zpos': [-20., -10., 0.],
                'layer_is_pixel': [1, 0, 1, 1], 'numScintLayers': 1, 'scint_Zpos': [-21., -18., -16., -11.],
                'scint_bar_Xcenter': [-15., -5., 5., 15.], 'scint_bar_Ycenter': [-15., -5., 5., 15.],
                'scint_detector_width': 40., 'scint_detector_height': 40., 'scint_bar_width': 10., 'scint_bar_height': 10.}
            (root / 'geometry.json').write_text(json.dumps(geometry))
            (root / 'plan.json').write_text(json.dumps({'geometry': str(root / 'geometry.json')}))
            paths = []
            for shard in range(3):
                path = root / 'shards' / str(shard)
                path.mkdir(); paths.append(path)
                observations = []
                for i in range(4):
                    observations.append({'pixel': {'channel_id': [shard*4+i],
                        'xyz_mm': [[geometry['pixel_Xpos'][shard*4+i], geometry['pixel_Ypos'][0], -20.]]},
                        'bar': {'channel_id': [], 'bounds_mm': [], 'layer': [], 'panel': [], 'bar': [],
                            'horizontal': [], 'edep_proxy_MeV': [], 'pe_proxy': []},
                        'tracker_hypothesis': {'station': [], 'xy_mm': [], 'sigma_xy_mm': []},
                        'has_response': [True, False, False]})
                pq.write_table(pa.Table.from_pylist(observations), path / 'observations.parquet')
                pq.write_table(pa.table({'label': list(range(4)), 'label_status': ['consistent']*4}), path / 'targets.parquet')
                pd.DataFrame({'run': [10081+shard]*4, 'chunk': [0]*4, 'row': list(range(4)),
                    'label': list(range(4)), 'split': ['train']*4}).to_parquet(path / 'provenance.parquet')
                (path / 'ideal_truth.parquet').write_text('excluded from observation staging')
            report = stage_shards(paths, root / 'shared', workers=2)
            self.assertEqual(len(report['shards']), 3)
            for path in paths:
                self.assertEqual(sorted(p.name for p in (root / 'shared' / path.name).iterdir()),
                    ['observations.parquet', 'targets.parquet'])
                for name in ['observations.parquet', 'targets.parquet']:
                    self.assertEqual((path / name).read_bytes(), (root / 'shared' / path.name / name).read_bytes())
            config = {'modalities': ['pixel', 'bar', 'tracker'], 'cache_shards': 1}
            with patch.dict(os.environ, {'PINPOINT_NATIVE_SHARD_ROOT': ''}):
                original = PlaneDataset(root, 'train', config, cache_dir=root / 'scratch_a')
            with patch.dict(os.environ, {'PINPOINT_NATIVE_SHARD_ROOT': str(root / 'shared')}):
                staged = PlaneDataset(root, 'train', config, cache_dir=root / 'scratch_b')
            self.assertEqual(original.paths, staged.paths)
            self.assertEqual(staged.native.paths, tuple(root / 'shared' / p.name for p in paths))
            a = collate_planes([original[i] for i in [1, 5, 9, 1]])
            b = collate_planes([staged[i] for i in [1, 5, 9, 1]])
            for key in a:
                self.assertTrue(torch.equal(a[key], b[key]), key)
            self.assertEqual(staged.native.cache_stats()['resident_shards'], 1)
            original.native.close(); staged.native.close()
            self.assertTrue((root / 'shared/0/observations.parquet').exists())
            self.assertTrue((paths[0] / 'observations.parquet').exists())

    def test_shared_staging_requires_local_scratch(self):
        with self.assertRaisesRegex(ValueError, 'local allocation scratch'):
            stage_shards([], Path('/eos/user/example'))


if __name__ == '__main__':
    unittest.main()
