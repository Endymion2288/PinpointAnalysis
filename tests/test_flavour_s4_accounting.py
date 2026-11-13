"""Allocation history queries must not assume completion follows cluster IDs."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from analysis.flavour.s4.accounting import refresh


class S4AccountingTests(unittest.TestCase):
    def test_terminal_history_recovery_matched_hardware_and_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'logs').mkdir()
            (root / 'logs/submission_one.json').write_text(json.dumps({
                'submitted': True, 'name': 'one', 'stage': 'campaign', 'backend': 'condor',
                'response': '1 job(s) submitted to cluster 123.', 'schedd': 's', 'tasks': [0]}))
            ad = {'ClusterId': 123, 'ProcId': 0, 'JobStatus': 4, 'RemoteWallClockTime': 120.,
                'RequestCpus': 4, 'MATCH_GPUs_DeviceName': 'NVIDIA H100L-1-12C MIG 1g.12gb',
                'AssignedGPUs': 'MIG-example', 'ExitCode': 0}
            calls = []
            def run(command, **kwargs):
                calls.append(command)
                if command[0] == 'condor_history':
                    self.assertNotIn('-since', command)
                    self.assertEqual(command[command.index('-match') + 1], '1')
                return SimpleNamespace(returncode=0, stdout=json.dumps([ad] if command[0]=='condor_history' else []), stderr='')
            with patch('analysis.flavour.s4.accounting.subprocess.run', side_effect=run):
                report = refresh(root)
            self.assertEqual(len(calls), 2)
            self.assertAlmostEqual(report['organization_device_hours'], 120 / 3600)
            self.assertAlmostEqual(report['CPU_allocation_hours'], 480 / 3600)
            row = report['allocations'][0]
            self.assertEqual(row['status'], 'terminal')
            self.assertEqual(row['jobs'][0]['GPU_name'], ad['MATCH_GPUs_DeviceName'])
            with patch('analysis.flavour.s4.accounting.subprocess.run') as command:
                cached = refresh(root)
            command.assert_not_called()
            self.assertTrue(cached['allocations'][0]['cached_terminal_snapshot'])
            self.assertEqual(cached['organization_device_hours'], report['organization_device_hours'])


if __name__ == '__main__':
    unittest.main()
