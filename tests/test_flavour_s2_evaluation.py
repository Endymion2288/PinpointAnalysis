"""Verify paired statistics and explicit common-support diagnostics."""

import unittest
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from analysis.flavour.s2.evaluate import common_support_weights, paired_interval, tau_metrics, factorial_contrast, matched_time, resampling_groups, seed_mean_interval, historical_three_class, tau_branch_metrics, input_response_activity, resource_metrics


class EvaluationTests(unittest.TestCase):
    def test_resource_columns_keep_all_attempt_cost_separate_from_program(self):
        manifest={'gpu_hours':1.}
        allocation={'allocation_complete':True,'all_terminal':True,'model_complete':True,
                    'GPU_hours':1.5,'allocated_seconds':5400.,'untracked_allocation_seconds':1800.}
        result=resource_metrics(manifest,allocation)
        self.assertEqual(result['gpu_hours'],1.)
        self.assertEqual(result['program_gpu_hours'],1.)
        self.assertEqual(result['allocated_gpu_hours'],1.5)
        self.assertEqual(result['allocated_seconds'],5400.)
        self.assertEqual(result['allocation_overhead_seconds'],1800.)
        self.assertTrue(result['allocation_complete'])
        for incomplete in [None,{**allocation,'allocation_complete':False},
                           {**allocation,'all_terminal':False},{**allocation,'model_complete':False}]:
            result=resource_metrics(manifest,incomplete)
            self.assertFalse(result['allocation_complete'])
            self.assertIsNone(result['allocated_gpu_hours'])
            self.assertIsNone(result['allocated_seconds'])
            self.assertIsNone(result['allocation_overhead_seconds'])
            self.assertEqual(result['program_gpu_hours'],1.)

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

    def test_seed_mean_keeps_physical_event_count_and_paired_identity(self):
        frame=self.sample()
        result=seed_mean_interval([(frame,frame.iloc[::-1])]*3,replicates=20)
        self.assertEqual(result['events'],len(frame))
        self.assertEqual(result['trained_seed_pairs'],3)
        self.assertEqual(result['delta'],0)
        self.assertEqual(result['CI95'],[0,0])
        self.assertEqual(result['common_support_tau_AP']['CI95'],[0,0])

    def test_seed_mean_averages_metrics_and_reports_training_spread(self):
        frame=self.sample()
        perfect=frame.assign(p2=frame.label.eq(2).astype(float))
        tied=frame.assign(p2=.25)
        result=seed_mean_interval([(tied,perfect),(perfect,tied),(tied,tied)],replicates=20)
        np.testing.assert_allclose(result['per_seed_deltas'],[.75,-.75,0])
        self.assertEqual(result['delta'],0)
        self.assertAlmostEqual(result['delta_seed_std'],.75)
        self.assertEqual(result['CI95'],[0,0])
        self.assertEqual(result['events'],len(frame))
        with self.assertRaises(ValueError):
            seed_mean_interval([(tied,perfect.iloc[:-1])],replicates=2)

    def test_historical_three_class_keeps_distinct_score_definitions(self):
        frame=pd.DataFrame({'label':[0,1,3,2],
            'p0':[.2,.3,.1,1.], 'p1':[.01,.5,.1,0.],
            'p2':[.78,.1,.1,0.], 'p3':[.01,.1,.7,0.]})
        result=historical_three_class(frame)
        self.assertEqual(result['events'],3)
        self.assertEqual(result['excluded_tau_events'],1)
        self.assertEqual(result['confusion_counts'],[[1,0,0],[0,1,0],[0,0,1]])
        self.assertEqual(result['balanced_accuracy'],1)
        self.assertEqual(result['classwise']['0']['raw_four_way_score_ROC_AUC'],.5)
        self.assertEqual(result['classwise']['0']['conditional_score_ROC_AUC'],1)
        self.assertFalse(result['historical_model_reproduced'])
        frame.loc[0,['p0','p1','p3']]=0
        result=historical_three_class(frame)
        self.assertEqual(result['zero_non_tau_score_events'],1)
        self.assertTrue(np.isfinite(result['conditional_NLL']))

    def test_tau_branch_retains_same_background_excludes_other_tau(self):
        frame=pd.DataFrame({'label':[0,1,3,2,2,2],'p2':[.1,.2,.3,.9,.5,.8],
                            'tau_branch':['not_tau']*3+['electron','electron','hadronic']})
        result=tau_branch_metrics(frame,'electron',.8)
        self.assertEqual(result['events'],5)
        self.assertEqual(result['tau_events'],2)
        self.assertEqual(result['background_events'],3)
        self.assertEqual(result['PR_AUC'],1)
        self.assertEqual(result['ROC_AUC'],1)
        self.assertEqual(result['efficiency_at_inclusive_1pct_FPR'],.5)
        missing=tau_branch_metrics(frame,'muon',.8)
        self.assertEqual(missing['tau_events'],0)
        self.assertIsNone(missing['PR_AUC'])

    def test_response_domain_tracks_model_modalities_and_truth_reference(self):
        frame=pd.DataFrame({'pixel_channels':[0,0,1,0], 'bar_channels':[0,0,0,1],
                            'tracker_channels':[0,2,0,0]})
        active,modalities=input_response_activity(frame,{'architecture':'hist_gradient_boosting'})
        self.assertEqual(modalities,['pixel','bar'])
        self.assertEqual(active.tolist(),[False,False,True,True])
        active,_=input_response_activity(frame,{'modalities':['pixel','bar','tracker']})
        self.assertEqual(active.tolist(),[False,True,True,True])
        active,modalities=input_response_activity(frame,{},hardware_model=False)
        self.assertIsNone(active)
        self.assertEqual(modalities,[])

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
