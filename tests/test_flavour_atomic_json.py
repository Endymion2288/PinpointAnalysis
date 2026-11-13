"""Concurrent completed-model observers cannot truncate or lose each other's JSON writes."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import threading
import unittest

from analysis.flavour.inventory import write_json


class AtomicJsonTests(unittest.TestCase):
    def test_multiple_writers_publish_complete_unicode_records(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'evidence.json'
            barrier = threading.Barrier(8)
            def publish(index):
                payload = {'writer': index, 'values': [index] * 1000, 'description': '完成模型的检查'}
                barrier.wait()
                for _ in range(8):
                    write_json(path, payload)
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(publish, range(8)))
            result = json.loads(path.read_text())
            self.assertEqual(result['values'], [result['writer']] * 1000)
            self.assertEqual(result['description'], '完成模型的检查')
            self.assertEqual(list(Path(temp).iterdir()), [path])


if __name__ == '__main__':
    unittest.main()
