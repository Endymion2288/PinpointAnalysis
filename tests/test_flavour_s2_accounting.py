"""Check that partial campaign submissions retain their real experiment IDs."""

import json
from pathlib import Path
import tempfile
import unittest

from analysis.flavour.s2.accounting import submission_jobs


class AccountingTests(unittest.TestCase):
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
            evidence.update(stage='reference',submit='arguments = analysis.flavour.s2.reference --mode $(mode)\nqueue mode in a b c d\n')
            submission.write_text(json.dumps(evidence))
            self.assertEqual(len(submission_jobs(submission)),4)


if __name__=='__main__':
    unittest.main()
