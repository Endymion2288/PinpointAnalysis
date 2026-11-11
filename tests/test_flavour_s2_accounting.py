"""Check that partial campaign submissions retain their real experiment IDs."""

import json
from pathlib import Path
import tempfile
import unittest

from analysis.flavour.s2.accounting import accounting_report, submission_jobs


class AccountingTests(unittest.TestCase):
    def test_same_name_across_roots_and_backends_does_not_mix_model_costs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            candidate=root/'h100'
            for directory in [root,candidate]:
                path=directory/'experiments/e/manifest.json'
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({'status':'complete','seconds':20}))
            rows=[
                {'root':str(root),'stage':'train','experiment':'e','cluster':1,'process':0,
                 'seconds':3600,'GPU_hours':1,'status':4},
                {'root':str(candidate),'stage':'train','experiment':'e','cluster':2,'process':0,
                 'seconds':1800,'GPU_hours':.5,'status':4},
                {'root':str(candidate),'stage':'train','experiment':'e','backend':'kubeflow',
                 'namespace':'n','job':'j','seconds':900,'GPU_hours':.25,'status':4},
                {'root':str(root),'stage':'profile','experiment':'profile','seconds':1800,
                 'GPU_hours':.5,'status':4},
            ]
            organization=accounting_report(root,rows,'now')
            child=accounting_report(candidate,[r for r in rows if r['root']==str(candidate)],'now')
            self.assertEqual(organization['GPU_hours'],2.25)
            self.assertEqual(organization['experiments']['e']['allocated_seconds'],3600)
            self.assertEqual(child['GPU_hours'],.75)
            self.assertEqual(child['experiments']['e']['allocated_seconds'],2700)
            self.assertEqual(child['experiments']['e']['untracked_allocation_seconds'],2680)
            self.assertEqual(len(child['experiments']['e']['jobs']),2)
            self.assertTrue(child['experiments']['e']['all_terminal'])

    def test_selected_task_indices_and_named_CPU_queue(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            campaign=root/'campaign.json'
            campaign.write_text(json.dumps({'experiments':[{'name':f'e{i}'} for i in range(8)]}))
            submission=root/'submission_train.json'
            evidence={'stage':'train','chosen':'test-schedd','response':'2 job(s) submitted to cluster 123.',
                'submit':f'arguments = analysis.flavour.s2.train --campaign {campaign} --task $(task)\nqueue task in 0,6\n'}
            submission.write_text(json.dumps(evidence))
            jobs=submission_jobs(submission)
            self.assertEqual([job['experiment'] for job in jobs],['e0','e6'])
            self.assertEqual([job['process'] for job in jobs],[0,1])
            evidence['submitted_experiments']=[{'task':0,'config':{'name':'frozen0'}},
                                               {'task':6,'config':{'name':'frozen6'}}]
            submission.write_text(json.dumps(evidence))
            campaign.write_text(json.dumps({'experiments':[]}))
            self.assertEqual([job['experiment'] for job in submission_jobs(submission)],['frozen0','frozen6'])
            evidence.update(stage='reference',submit='arguments = analysis.flavour.s2.reference --mode $(mode)\nqueue mode in a b c d\n')
            submission.write_text(json.dumps(evidence))
            self.assertEqual(len(submission_jobs(submission)),4)


if __name__=='__main__':
    unittest.main()
