"""Observed EOS sync errors must preserve the last committed checkpoint."""
import errno
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import torch

from analysis.flavour.s2.train import save_checkpoint, save_parquet


class AtomicSyncTests(unittest.TestCase):
    def test_truncated_failed_write_is_fully_replayed(self):
        with tempfile.TemporaryDirectory() as tmp, patch('analysis.flavour.s2.train.time.sleep'):
            path = Path(tmp) / 'last.pt'; state = {'epoch': 2, 'model': torch.arange(10000)}
            calls = 0; original_sync = os.fsync
            def truncate_once(fd):
                nonlocal calls
                calls += 1
                if calls == 1:
                    os.ftruncate(fd, os.fstat(fd).st_size // 2)
                    raise OSError(errno.EPROTO, 'observed truncated EOS sync failure')
                return original_sync(fd)
            with patch('analysis.flavour.s2.train.os.fsync', side_effect=truncate_once):
                save_checkpoint(path, state)
            self.assertEqual(calls, 2)
            torch.testing.assert_close(torch.load(path)['model'], state['model'], rtol=0, atol=0)

    def test_transient_sync_retries_then_publishes_complete_files(self):
        with tempfile.TemporaryDirectory() as tmp, patch('analysis.flavour.s2.train.time.sleep'):
            path = Path(tmp) / 'last.pt'
            with patch('analysis.flavour.s2.train.os.fsync', side_effect=[OSError(errno.EPROTO, 'protocol'), None]) as sync:
                save_checkpoint(path, {'epoch': 2, 'model': torch.arange(7)})
            self.assertEqual(sync.call_count, 2)
            self.assertEqual(torch.load(path)['epoch'], 2)
            self.assertFalse(list(path.parent.glob('.last.pt.*.tmp')))
            parquet = Path(tmp) / 'epoch_predictions.parquet'; data = pd.DataFrame({'p': [.1, .9]})
            with patch('analysis.flavour.s2.train.os.fsync', side_effect=[OSError(errno.EIO, 'I/O'), None]):
                save_parquet(data, parquet)
            pd.testing.assert_frame_equal(pd.read_parquet(parquet), data)

    def test_exhausted_sync_never_replaces_committed_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp, patch('analysis.flavour.s2.train.time.sleep'):
            path = Path(tmp) / 'last.pt'; save_checkpoint(path, {'epoch': 1})
            with patch('analysis.flavour.s2.train.os.fsync', side_effect=OSError(errno.EPROTO, 'protocol')) as sync:
                with self.assertRaises(OSError):
                    save_checkpoint(path, {'epoch': 2})
            self.assertEqual(sync.call_count, 4)
            self.assertEqual(torch.load(path)['epoch'], 1)
            self.assertEqual(len(list(path.parent.glob('.last.pt.*.tmp'))),1)

    def test_permanent_error_is_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('analysis.flavour.s2.train.os.fsync', side_effect=OSError(errno.ENOSPC, 'full')) as sync:
                with self.assertRaises(OSError):
                    save_checkpoint(Path(tmp) / 'last.pt', {'epoch': 1})
            self.assertEqual(sync.call_count, 1)


if __name__ == '__main__':
    unittest.main()
