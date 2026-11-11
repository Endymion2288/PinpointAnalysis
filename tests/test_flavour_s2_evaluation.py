"""Verify paired statistics and explicit common-support diagnostics."""

import unittest
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from analysis.flavour.s2.evaluate import common_support_weights, paired_interval, tau_metrics, factorial_contrast, matched_time, resampling_groups


class EvaluationTests(unittest.TestCase):
    def sample(self):
        rng=np.random.default_rng(42)
        labels=np.repeat(np.arange(4),60)
        return pd.DataFrame({'run':10081+labels,'chunk':np.tile(np.repeat(np.arange(3),20),4),
            'row':np.tile(np.arange(60),4),'label':labels,'p2':rng.random(240),
            'initE':np.full(240,200000.),'initX':np.zeros(240),'initY':np.zeros(240),
            'initZ':np.full(240,-2400.)})

    def test_identity_paired_interval_and_event_mismatch(self):
        frame=self.sample()
        result=paired_interval(frame,frame.iloc[::-1],replicates=30)
        self.assertEqual(result['delta'],0)
        self.assertEqual(result['CI95'],[0,0])
        with self.assertRaises(ValueError):
            paired_interval(frame,frame.iloc[:-1],replicates=2)

    def test_common_support_balances_classes_excludes_empty_support(self):
        frame=self.sample()
        frame=pd.concat([frame,frame[frame.label.eq(0)].assign(row=lambda x:x.row+100,initE=5000000.)],ignore_index=True)
        weights,summary=common_support_weights(frame)
        self.assertEqual(summary['supported_cells'],1)
        self.assertEqual(summary['class_weight_sums'],[60.]*4)
        self.assertTrue(np.all(weights[-60:]==0))
        result=tau_metrics(frame.loc[weights>0],weights[weights>0])
        self.assertEqual(result['tau_prevalence'],.25)
        self.assertEqual(result['Neff'],240.)
        self.assertFalse(result['FPR_0.001']['tail_statistics_sufficient'])

    def test_factorial_same_cells_have_zero_interaction(self):
        frame=self.sample()
        result=factorial_contrast([frame,frame.iloc[::-1],frame,frame],replicates=20)
        self.assertEqual(result['interaction'],0)
        self.assertEqual(result['CI95'],[0,0])

    def test_existing_family_links_join_chunks_for_resampling(self):
        frame=self.sample()
        frame['family_hi']=np.arange(len(frame),dtype='uint64')
        frame['family_lo']=np.zeros(len(frame),dtype='uint64')
        frame.loc[20,'family_hi']=frame.loc[0,'family_hi']
        groups,summary=resampling_groups(frame)
        self.assertEqual(summary['chunks'],12)
        self.assertEqual(summary['resampling_blocks'],11)
        self.assertEqual(summary['multi_chunk_blocks'],1)
        self.assertTrue(any(0 in block and 20 in block for block in groups[(0,)]))

    def test_time_match_excludes_later_better_checkpoint(self):
        frame=self.sample()
        for c in [0,1,3]:
            frame[f'p{c}']=(1-frame.p2)/3
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for name,seconds in [('a',[10,30]),('b',[10,20])]:
                path=root/'experiments'/name
                path.mkdir(parents=True)
                (path/'manifest.json').write_text(json.dumps({'gpu_name':'same GPU'}))
                history=[{'epoch':i+1,'wall_seconds':s,'validation':{'NLL':2-i}} for i,s in enumerate(seconds)]
                (path/'history.json').write_text(json.dumps(history))
                ep=pd.concat([pd.DataFrame({'row':np.arange(len(frame)),'epoch':e,
                       **{f'p{c}':frame[f'p{c}'].to_numpy() for c in range(4)}}) for e in [1,2]])
                ep.to_parquet(path/'epoch_predictions.parquet')
            result=matched_time('a','b',root,{'a':frame,'b':frame})
            self.assertEqual(result['budget_seconds'],20)
            self.assertEqual([r['epoch'] for r in result['runs']],[1,2])
            self.assertEqual(result['tau_AP_B_minus_A'],0)
            (root/'campaign_accounting.json').write_text(json.dumps({'experiments':{
                'a':{'untracked_allocation_seconds':0},'b':{'untracked_allocation_seconds':15}}}))
            result=matched_time('a','b',root,{'a':frame,'b':frame})
            self.assertEqual(result['budget_seconds'],30)
            self.assertEqual([r['epoch'] for r in result['runs']],[2,1])


if __name__=='__main__':
    unittest.main()
