"""Queue handoff must never launch a second writer beside a running task."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analysis.flavour.s3.dispatch import handoff


class QueueHandoffTests(unittest.TestCase):
    def setup_root(self, root):
        args = argparse.Namespace(root=root, name='test-slot')
        directory = root / 'kubeflow/dispatch_test-slot'; directory.mkdir(parents=True)
        (directory / 'ready.json').write_text(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'pod': 'pod'}))
        (root / 'kubeflow/execution_test-slot.json').write_text(json.dumps({'stage': 'dispatch'}))
        (root / 'submission_one.json').write_text(json.dumps({'stage': 'train', 'response': '1 job(s) submitted to cluster 12.',
            'chosen': 'schedd', 'submit': 'arguments = module --campaign /frozen/campaign.json\n',
            'submitted_experiments': [{'task': 5, 'config': {'name': 'primary'}}]}))
        return args, directory

    def mock_commands(self, initial=1, after_hold=5):
        calls = []; queries = 0
        def command(args):
            nonlocal queries
            calls.append(args)
            if args[0] == 'kubectl':
                if 'job' in args:
                    return json.dumps({'status': {'active': 1}})
                return json.dumps({'items': [{'metadata': {'name': 'pod'}, 'status': {'phase': 'Running',
                    'containerStatuses': [{'name': 's3', 'state': {'running': {"startedAt": 'now'}}}]}}]})
            if args[0] == 'condor_q':
                queries += 1
                if queries == 1:
                    return json.dumps([{'ClusterId': 12, 'ProcId': 0, 'JobStatus': initial}])
                if queries == 2:
                    return json.dumps([{'JobStatus': after_hold}])
                return '[]'
            if args[0] == 'condor_hold':
                self.assertIn('JobStatus == 1', args[args.index('-constraint') + 1])
                return 'held idle task'
            if args[0] == 'condor_rm':
                return 'removed held copy'
            raise AssertionError(args)
        return command, calls

    def test_no_ready_slot_preserves_condor_queue(self):
        with tempfile.TemporaryDirectory() as tmp, patch('analysis.flavour.s3.dispatch.command') as command:
            handoff(argparse.Namespace(root=Path(tmp), name='j'))
            command.assert_not_called()

    def test_running_and_concurrently_started_tasks_not_dispatched(self):
        for initial, after in [(2, 5), (1, 2)]:
            with self.subTest(initial=initial), tempfile.TemporaryDirectory() as tmp:
                args, directory = self.setup_root(Path(tmp)); command, calls = self.mock_commands(initial, after)
                with patch('analysis.flavour.s3.dispatch.command', side_effect=command):
                    handoff(args)
                self.assertFalse((directory / 'assignment.json').exists())
                self.assertFalse(any(c[0] == 'condor_rm' for c in calls))

    def test_idle_task_removed_before_assignment_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, directory = self.setup_root(Path(tmp)); command, calls = self.mock_commands()
            with patch('analysis.flavour.s3.dispatch.command', side_effect=command):
                handoff(args)
            assignment = json.loads((directory / 'assignment.json').read_text())
            self.assertEqual(assignment['task'], 5)
            self.assertTrue(any(c[0] == 'condor_rm' for c in calls))
            record = json.loads((args.root / 'kubeflow/execution_test-slot.json').read_text())
            self.assertEqual(record['stage'], 'train')
            self.assertEqual(record['experiment'], 'primary')


if __name__ == '__main__':
    unittest.main()
