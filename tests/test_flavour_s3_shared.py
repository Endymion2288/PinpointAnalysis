"""A shared GPU allocation must be charged once, including idle gaps."""
import json
from pathlib import Path
import tempfile
import unittest

from analysis.flavour.s3.accounting import allocation_transform, shared_seconds


class SharedAllocationTests(unittest.TestCase):
    def test_overlap_and_idle_gaps_conserve_reservation(self):
        shares = shared_seconds(0, 10, {'a': (2, 6), 'b': (4, 9)})
        self.assertEqual(shares, {'a': 4.5, 'b': 5.5})
        self.assertEqual(sum(shares.values()), 10)

    def test_process_outside_reservation_is_clipped(self):
        self.assertEqual(shared_seconds(0, 10, {'a': (-1, 11), 'b': (2, 4)}),
                         {'a': 9., 'b': 1.})

    def test_split_retains_total_device_cpu_and_resource_hours(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'kubeflow').mkdir()
            (root / 'kubeflow/parallel_j.json').write_text(json.dumps({'processes': [
                {'experiment': 'a', 'started_at': '2026-11-12T00:00:00+00:00',
                 'finished_at': '2026-11-12T01:00:00+00:00'},
                {'experiment': 'b', 'started_at': '2026-11-12T00:30:00+00:00',
                 'finished_at': '2026-11-12T01:00:00+00:00'}]}))
            row = {'job': 'j', 'status': 4, 'seconds': 3600., 'GPU_hours': 1.,
                   'CPU_core_hours': 22., 'GPU_resource_hours': {'mig': 1.},
                   'started_at': '2026-11-12T00:00:00+00:00',
                   'finished_at': '2026-11-12T01:00:00+00:00'}
            rows = allocation_transform(root, [row])
            self.assertEqual([r['GPU_hours'] for r in rows], [.75, .25])
            self.assertEqual(sum(r['CPU_core_hours'] for r in rows), 22.)
            self.assertEqual(sum(r['GPU_resource_hours']['mig'] for r in rows), 1.)
            self.assertEqual(sum(r['seconds'] for r in rows), 3600.)


if __name__ == '__main__':
    unittest.main()
