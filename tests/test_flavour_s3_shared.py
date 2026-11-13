"""A shared GPU allocation must be charged once, including idle gaps."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analysis.flavour.s3.accounting import allocation_transform, shared_seconds
from analysis.flavour.s3.parallel import main as parallel_main, slots_in_use


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
                   'allocation_estimate_kind': 'conservative_upper_bound',
                   'allocation_basis': 'retained_scheduler_events_to_job_terminal_time_conservative_upper_bound',
                   'pod_phase': None, 'job_terminal_conditions': [{'type': 'Complete', 'status': 'True'}],
                   'CPU_core_hours': 22., 'GPU_resource_hours': {'mig': 1.},
                   'started_at': '2026-11-12T00:00:00+00:00',
                   'finished_at': '2026-11-12T01:00:00+00:00'}
            rows = allocation_transform(root, [row])
            self.assertEqual([r['GPU_hours'] for r in rows], [.75, .25])
            self.assertEqual(sum(r['CPU_core_hours'] for r in rows), 22.)
            self.assertEqual(sum(r['GPU_resource_hours']['mig'] for r in rows), 1.)
            self.assertEqual(sum(r['seconds'] for r in rows), 3600.)
            for shared in rows:
                self.assertEqual(shared['allocation_estimate_kind'], 'conservative_upper_bound')
                self.assertEqual(shared['reservation_allocation_basis'], row['allocation_basis'])
                self.assertIsNone(shared['pod_phase'])
                self.assertEqual(shared['job_terminal_conditions'][0]['type'], 'Complete')

    def test_condor_parallel_group_conserves_remote_wallclock_charge(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / 'kubeflow').mkdir()
            submission = root / 'submission_test.json'
            submission.write_text(json.dumps({'parallel_name': 'group'}))
            (root / 'kubeflow/parallel_group.json').write_text(json.dumps({'processes': [
                {'experiment': 'a', 'started_at': '2026-11-12T00:00:00+00:00', 'finished_at': '2026-11-12T01:00:00+00:00'},
                {'experiment': 'b', 'started_at': '2026-11-12T00:30:00+00:00', 'finished_at': '2026-11-12T01:00:00+00:00'}]}))
            rows = allocation_transform(root, [{'submission': str(submission), 'status': 4,
                'seconds': 7200., 'GPU_hours': 2., 'CPU_core_hours': 24.}])
            self.assertEqual([r['experiment'] for r in rows], ['a', 'b'])
            self.assertEqual(sum(r['seconds'] for r in rows), 7200.)
            self.assertEqual(sum(r['GPU_hours'] for r in rows), 2.)
            self.assertEqual(sum(r['CPU_core_hours'] for r in rows), 24.)

    def test_extra_supervisor_in_same_pod_is_charged_once(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / 'kubeflow').mkdir()
            for name, allocation, start in [('j', 'j', '00:00:00'), ('extra', 'j', '00:30:00')]:
                (root / f'kubeflow/parallel_{name}.json').write_text(json.dumps({
                    'job': name, 'allocation_job': allocation, 'processes': [{
                        'experiment': name, 'started_at': f'2026-11-12T{start}+00:00',
                        'finished_at': '2026-11-12T01:00:00+00:00'}]}))
            row = {'job': 'j', 'status': 4, 'seconds': 3600., 'GPU_hours': 1.,
                   'CPU_core_hours': 22., 'started_at': '2026-11-12T00:00:00+00:00',
                   'finished_at': '2026-11-12T01:00:00+00:00'}
            rows = allocation_transform(root, [row])
            by_name = {r['experiment']: r for r in rows}
            self.assertEqual(set(by_name), {'j', 'extra'})
            self.assertEqual(by_name['j']['GPU_hours'], .75)
            self.assertEqual(by_name['extra']['GPU_hours'], .25)
            self.assertEqual(sum(r['GPU_hours'] for r in rows), 1.)
            self.assertEqual(sum(r['CPU_core_hours'] for r in rows), 22.)
            self.assertTrue(all(len(r['process_interval_evidence']) == 2 for r in rows))


class ParallelLifecycleTests(unittest.TestCase):
    def test_finished_worker_releases_gpu_while_a_peer_keeps_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / 'kubeflow').mkdir()
            campaign = root / 'campaign.json'
            campaign.write_text(json.dumps({'experiments': [{'name': 'peer'}]}))
            (root / 'claims/peer').mkdir(parents=True)
            (root / 'claims/peer/state.json').write_text(json.dumps({'status': 'running', 'job': 'other'}))
            (root / 'experiments/peer').mkdir(parents=True)
            (root / 'experiments/peer/manifest.json').write_text(json.dumps({'status': 'running'}))
            argv = ['parallel', '--root', str(root), '--campaign', str(campaign), '--name', 'done',
                    '--tasks', '0', '--training-module', 'analysis.flavour.s3.train_repair']
            with patch('sys.argv', argv), patch('analysis.flavour.s3.parallel.subprocess.Popen') as launched, \
                    patch('analysis.flavour.s3.parallel.time.sleep', side_effect=AssertionError('idle reservation retained')):
                parallel_main()
            launched.assert_not_called()
            evidence = json.loads((root / 'kubeflow/parallel_done.json').read_text())
            self.assertEqual(evidence['processes'], [])
            self.assertIn('finished_at', evidence)

    def test_backfill_waits_for_a_slot_in_parent_supervisor(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / 'kubeflow').mkdir()
            campaign = root / 'campaign.json'
            campaign.write_text(json.dumps({'experiments': [{'name': 'b'}]}))
            parent = root / 'kubeflow/parallel_j.json'
            previous = {'job': 'j', 'processes': [{'experiment': 'a', 'started_at': '2026-11-12T00:00:00+00:00'}]}
            parent.write_text(json.dumps(previous))
            self.assertEqual(slots_in_use(root / 'kubeflow', 'j'), 1)
            self.assertEqual(slots_in_use(root / 'kubeflow', 'other'), 0)
            class Child:
                pid = 1
                def poll(self):
                    return 0
            def release(_):
                previous['processes'][0]['finished_at'] = '2026-11-12T01:00:00+00:00'
                parent.write_text(json.dumps(previous))
            argv = ['parallel', '--root', str(root), '--campaign', str(campaign), '--name', 'extra',
                    '--allocation-job', 'j', '--tasks', '0', '--concurrency', '1']
            with patch('sys.argv', argv), patch('analysis.flavour.s3.parallel.subprocess.Popen', return_value=Child()) as launch, \
                    patch('analysis.flavour.s3.parallel.time.sleep', side_effect=release) as waited:
                parallel_main()
            self.assertEqual(waited.call_count, 1)
            self.assertEqual(launch.call_count, 1)
            self.assertEqual(slots_in_use(root / 'kubeflow', 'j'), 0)

    def test_repair_backends_claim_each_fit_only_once(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root/'kubeflow').mkdir()
            campaign=root/'campaign.json'
            campaign.write_text(json.dumps({'experiments':[{'name':'a'},{'name':'b'}]}))
            (root/'claims/b').mkdir(parents=True)
            (root/'claims/b/state.json').write_text(json.dumps({'status':'complete','job':'other'}))
            for name in ['a','b']:
                (root/'experiments'/name).mkdir(parents=True)
                (root/'experiments'/name/'manifest.json').write_text(json.dumps({'status':'complete'}))
            class Child:
                pid=1
                def poll(self):
                    return 0
            argv=['parallel','--root',str(root),'--campaign',str(campaign),'--name','j',
                  '--tasks','0','1','--training-module','analysis.flavour.s3.train_repair']
            with patch('sys.argv',argv),patch('analysis.flavour.s3.parallel.subprocess.Popen',return_value=Child()) as launch:
                parallel_main()
            self.assertEqual(launch.call_count,1)
            self.assertEqual(launch.call_args.args[0][-1],'0')
            record=json.loads((root/'kubeflow/parallel_j.json').read_text())
            self.assertEqual([p['experiment'] for p in record['processes']],['a'])
            self.assertEqual(json.loads((root/'claims/b/state.json').read_text())['job'],'other')

    def test_first_child_exit_does_not_end_other_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'kubeflow').mkdir()
            campaign = root / 'campaign.json'
            campaign.write_text(json.dumps({'experiments': [{'name': n} for n in ['a', 'b', 'c']]}))
            polls = {0: [0], 1: [None, None, 0], 2: [None, 0]}
            live = set()
            peak = []

            class Child:
                def __init__(self, task):
                    self.task = task
                    self.pid = task + 10

                def poll(self):
                    result = polls[self.task].pop(0)
                    if result is not None:
                        live.remove(self.task)
                    return result

            def launch(arguments, **kwargs):
                task = int(arguments[-1])
                live.add(task)
                peak.append(len(live))
                return Child(task)

            argv = ['parallel', '--root', str(root), '--campaign', str(campaign),
                    '--name', 'j', '--tasks', '0', '1', '2', '--concurrency', '2']
            with patch('sys.argv', argv), patch('analysis.flavour.s3.parallel.subprocess.Popen', side_effect=launch), \
                    patch('analysis.flavour.s3.parallel.time.sleep'):
                parallel_main()
            evidence = json.loads((root / 'kubeflow/parallel_j.json').read_text())
            self.assertFalse(live)
            self.assertLessEqual(max(peak), 2)
            self.assertEqual([p['exit_code'] for p in evidence['processes']], [0, 0, 0])
            self.assertTrue(all('finished_at' in p for p in evidence['processes']))


if __name__ == '__main__':
    unittest.main()
