"""Allocation history queries must not assume completion follows cluster IDs."""
import json
from datetime import datetime, timezone
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from analysis.flavour.s4.accounting import refresh


class S4AccountingTests(unittest.TestCase):
    def test_successful_job_bounds_duration_after_pod_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'logs').mkdir(); (root / 'audit/backends').mkdir(parents=True)
            (root / 'logs/submission_one.json').write_text(json.dumps({
                'submitted': True, 'name': 'one', 'stage': 'campaign', 'backend': 'kubeflow',
                'flavor': 'h100-47gb', 'pod_admission': {'spec': {'containers': [{'name': 's4',
                    'resources': {'requests': {'cpu': '4', 'nvidia.com/mig-3g.47gb': '1'}}}]}}}))
            (root / 'audit/backends/one.pods.json').write_text(json.dumps({'items': [{
                'metadata': {'name': 'one-pod'}, 'status': {'phase': 'Running',
                    'startTime': '2026-11-14T00:00:00Z', 'containerStatuses': [{'name': 's4',
                        'state': {'running': {'startedAt': '2026-11-14T00:00:05Z'}}}]}}]}))
            job = {'metadata': {'name': 'one', 'uid': 'job-uid'}, 'status': {
                'completionTime': '2026-11-14T00:02:00Z', 'succeeded': 1,
                'conditions': [{'type': 'Complete', 'status': 'True'}]}}
            def run(command, **kwargs):
                payload = job if 'job' in command else {'items': []}
                return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr='')
            active_job = {'metadata': job['metadata'], 'status': {'ready': 1}}
            with patch('analysis.flavour.s4.accounting.subprocess.run', side_effect=lambda command, **kwargs:
                    SimpleNamespace(returncode=0, stdout=json.dumps(active_job if 'job' in command else {'items': []}), stderr='')):
                active = refresh(root)
            self.assertEqual(active['allocations'][0]['status'], 'active_or_pending')
            with patch('analysis.flavour.s4.accounting.subprocess.run', side_effect=run):
                report = refresh(root)
            row = report['allocations'][0]
            self.assertEqual(row['status'], 'terminal')
            self.assertEqual(row['allocation_seconds'], 120.)
            self.assertAlmostEqual(report['organization_device_hours'], 120 / 3600)
            self.assertAlmostEqual(report['CPU_allocation_hours'], 480 / 3600)
            self.assertEqual(row['pods'][0]['allocation_estimate_kind'],
                'job_completion_upper_bound_after_pod_cleanup')
            self.assertTrue(row['pods'][0]['termination_inferred_from_successful_job'])
            self.assertEqual(row['job_terminal_evidence']['uid'], 'job-uid')
            # A subsequent missing Pod/Job must preserve the saved endpoint.
            with patch('analysis.flavour.s4.accounting.subprocess.run', return_value=
                    SimpleNamespace(returncode=0, stdout='{"items": []}', stderr='')) as command:
                cached = refresh(root)
            self.assertEqual(command.call_count, 1)
            self.assertEqual(cached['organization_device_hours'], report['organization_device_hours'])

    def test_cpu_diagnostics_do_not_count_gpu_time(self):
        for stage in ['cache_profile', 'parameter_audit', 'solver_probe', 'probe_fit']:
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / 'logs').mkdir()
                (root / 'logs/submission_profile.json').write_text(json.dumps({
                    'submitted': True, 'name': 'profile', 'stage': stage, 'backend': 'condor',
                    'response': '1 job(s) submitted to cluster 123.', 'schedd': 's', 'tasks': [0]}))
                ad = {'ClusterId': 123, 'ProcId': 0, 'JobStatus': 4, 'RemoteWallClockTime': 120.,
                    'RequestCpus': 4, 'RequestGPUs': 0, 'ExitCode': 0}
                with patch('analysis.flavour.s4.accounting.subprocess.run', return_value=
                        SimpleNamespace(returncode=0, stdout=json.dumps([ad]), stderr='')):
                    report = refresh(root)
                self.assertEqual(report['organization_device_hours'], 0.)
                self.assertAlmostEqual(report['CPU_allocation_hours'], 480 / 3600)
                self.assertEqual(report['allocations'][0]['allocation_seconds'], 120.)

    def test_live_queue_covers_all_procs_without_scanning_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'logs').mkdir()
            (root / 'logs/submission_one.json').write_text(json.dumps({
                'submitted': True, 'name': 'one', 'stage': 'campaign', 'backend': 'condor',
                'response': '1 job(s) submitted to cluster 123.', 'schedd': 's', 'tasks': [0]}))
            ad = {'ClusterId': 123, 'ProcId': 0, 'JobStatus': 2, 'RemoteWallClockTime': 0.,
                'JobCurrentStartDate': datetime.now(timezone.utc).timestamp() - 300, 'RequestCpus': 4}
            with patch('analysis.flavour.s4.accounting.subprocess.run', return_value=
                    SimpleNamespace(returncode=0, stdout=json.dumps([ad]), stderr='')) as command:
                report = refresh(root)
            self.assertEqual(command.call_count, 1)
            self.assertEqual(command.call_args.args[0][0], 'condor_q')
            self.assertGreaterEqual(report['organization_device_hours'], 300 / 3600)
            self.assertEqual(report['allocations'][0]['status'], 'active_or_missing_history')

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
