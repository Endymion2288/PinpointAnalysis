"""GPU cost includes scheduled startup and failed image-pull reservations."""

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analysis.flavour.s2.kubeflow import allocations, pod_allocation


class KubeflowAccountingTests(unittest.TestCase):
    def setUp(self):
        self.execution = {'namespace': 'test', 'job': 'j', 'root': '/r',
                          'experiment': 'e', 'stage': 'train'}
        self.pod = {
            'metadata': {'name': 'p', 'uid': 'uid'},
            'spec': {'containers': [{'name': 's2', 'resources': {'requests': {
                'cpu': '4000m', 'nvidia.com/gpu': 1}}}]},
            'status': {'phase': 'Pending', 'conditions': []},
        }
        self.now = datetime(2026, 11, 11, 22, 0, tzinfo=timezone.utc)

    def schedule(self):
        self.pod['status']['conditions'] = [{'type': 'PodScheduled', 'status': 'True',
                                            'lastTransitionTime': '2026-11-11T21:00:00Z'}]

    def test_unscheduled_queue_consumes_no_GPU(self):
        self.assertEqual(pod_allocation(self.pod, self.execution, self.now)['GPU_hours'], 0)

    def test_image_pull_reserves_GPU_from_scheduling(self):
        self.schedule()
        row = pod_allocation(self.pod, self.execution, self.now)
        self.assertEqual(row['GPU_hours'], 1)
        self.assertEqual(row['CPU_core_hours'], 4)
        self.assertTrue(row['running_estimate'])

    def test_success_uses_container_finish_not_query_time(self):
        self.schedule()
        self.pod['status'].update(phase='Succeeded', containerStatuses=[{
            'name': 's2', 'state': {'terminated': {'finishedAt': '2026-11-11T21:30:00Z', 'exitCode': 0}}}])
        row = pod_allocation(self.pod, self.execution, self.now)
        self.assertEqual(row['GPU_hours'], .5)
        self.assertEqual(row['exit_code'], 0)
        self.assertFalse(row['running_estimate'])

    def test_image_pull_deadline_does_not_end_at_old_init_finish(self):
        self.schedule()
        self.pod['status'].update(phase='Failed', initContainerStatuses=[{
            'name': 'injector', 'state': {'terminated': {'finishedAt': '2026-11-11T21:00:02Z'}}}])
        self.pod['job_terminal_time'] = '2026-11-11T21:45:00Z'
        self.assertEqual(pod_allocation(self.pod, self.execution, self.now)['GPU_hours'], .75)

    def test_native_sidecar_and_bootstrap_CPU_reservation(self):
        self.schedule()
        self.pod['spec']['initContainers'] = [
            {'name': 'bootstrap', 'resources': {'requests': {'cpu': '1'}}},
            {'name': 'oauth', 'restartPolicy': 'Always', 'resources': {'requests': {'cpu': '100m'}}},
        ]
        self.assertAlmostEqual(pod_allocation(self.pod, self.execution, self.now)['CPU_core_hours'], 4.1)

    def recover_deleted(self, events):
        execution = {**self.execution, 'uid': 'job-uid', 'backend': 'kubeflow', 'submitted': True}
        job = {'metadata': {'uid': 'job-uid'}, 'status': {'conditions': [{
            'type': 'Failed', 'status': 'True', 'lastTransitionTime': '2026-11-11T21:45:00Z'}]},
            'spec': {'template': {'spec': self.pod['spec']}}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'kubeflow').mkdir()
            (root / 'execution_test.json').write_text(json.dumps(execution))
            cached = {'uid': {'pod': self.pod, 'allocation': pod_allocation(self.pod, execution, self.now),
                              'observed_utc': '2026-11-11T21:30:00+00:00'}}
            (root / 'kubeflow/pod_accounting_snapshots.json').write_text(json.dumps(cached))
            with patch('analysis.flavour.s2.kubeflow.kubectl', side_effect=[
                    json.dumps({'items': []}), json.dumps({'items': [job]}), json.dumps({'items': events})]):
                return allocations(root, self.now)[0]

    def test_deleted_pending_pod_with_expired_create_event(self):
        row = self.recover_deleted([{'reason': 'FailedScheduling', 'involvedObject': {
            'kind': 'Pod', 'name': 'p', 'uid': 'uid'}}])
        self.assertEqual(row['status'], 4)
        self.assertEqual(row['GPU_hours'], 0)
        self.assertEqual(row['finished_at'], '2026-11-11T21:45:00+00:00')

    def test_deleted_scheduled_pod_retains_known_reservation(self):
        self.schedule()
        row = self.recover_deleted([])
        self.assertEqual(row['status'], 4)
        self.assertEqual(row['GPU_hours'], .75)

    def test_deleted_pending_pod_without_events_stays_unknown(self):
        row = self.recover_deleted([])
        self.assertEqual(row['status'], 'missing_terminal_pod_history')
        self.assertIsNone(row['seconds'])


if __name__ == '__main__':
    unittest.main()
