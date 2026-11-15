import unittest
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd

from analysis.flavour.inventory import write_json
from analysis.flavour.s2.dataset import SpatialBatchSampler
from analysis.flavour.v2.r1.interval_review import domain_mask
from analysis.flavour.v2.r1.evaluate import scalar_summary, SCALARS
from analysis.flavour.v2.r2.evaluate import configuration, domains
from analysis.flavour.v2.r2.campaign import extend
from analysis.flavour.v2.r2.finish import advance
from analysis.flavour.v2.r2.schedule import learning_rate, sampler_trace
from analysis.flavour.v2.r2.train import OffsetSampler


class ScalingBudgetTests(unittest.TestCase):
    def data(self):
        sizes = np.array([8, 8, 9, 9, 10])
        boundary = np.r_[0, sizes.cumsum()]
        return SimpleNamespace(paths=list(range(len(sizes))), indexes=np.arange(boundary[-1]),
            native=SimpleNamespace(boundaries=boundary), config={'cache_shards':2, 'bucket_size':0})

    def test_exact_pass_budget_includes_all_tail_rows(self):
        data = self.data()
        sampler = SpatialBatchSampler(data,8,42)
        labels = np.arange(len(data.indexes))%4
        report,sizes,counts = sampler_trace(sampler,labels,12)
        self.assertEqual(report['total_exposures'],len(labels)*12)
        self.assertEqual(report['minimum_row_exposures'],12)
        self.assertEqual(report['maximum_row_exposures'],12)
        self.assertEqual(report['total_updates'],sum(row['updates'] for row in report['per_pass']))
        np.testing.assert_array_equal(counts.sum(0),np.bincount(labels,minlength=4)*12)
        np.testing.assert_array_equal(counts.sum(1),sizes)
        self.assertGreater(len(set(row['updates'] for row in report['per_pass'])),1)

    def test_reparameterized_schedule_uses_complete_prescribed_budget(self):
        self.assertEqual(learning_rate(0,101,.001),.00005)
        self.assertAlmostEqual(learning_rate(5,101,.001),.001)
        self.assertAlmostEqual(learning_rate(100,101,.001),.00005)
        self.assertAlmostEqual(learning_rate(10000,101,.001),.00005)
        self.assertAlmostEqual(learning_rate(50,101,.001),learning_rate(150,301,.001))
        self.assertGreater(learning_rate(100,301,.001),learning_rate(100,101,.001))

    def test_resume_offset_preserves_exact_remaining_batch_order(self):
        sampler = SpatialBatchSampler(self.data(),8,42)
        sampler.epoch=7
        expected=list(sampler)
        resumed=OffsetSampler(sampler)
        resumed.offset=4
        self.assertEqual(list(resumed),expected[4:])
        self.assertEqual(len(resumed),len(expected)-4)

    def test_checkpoint_policies_have_separate_complete_contrasts(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            names=[]
            for design in ['d0','d1','d2']:
                for task in ['t3','t4']:
                    name=f'b1_{design}_{task}_s42'
                    names.append(name)
                    write_json(root/'configs'/f'{name}.json',{'family':'B1','task':task.upper(),
                        'seed':42,'design':design.upper()})
            write_json(root/'campaign.json',{'initial_models':names})
            result=configuration(root)
            self.assertEqual(len(result['models']),12)
            self.assertEqual(len(result['pairs']),72)
            entries={m['name']:m for m in result['models']}
            for pair in result['pairs']:
                self.assertEqual(entries[pair['a']]['kind'],entries[pair['b']]['kind'])
            self.assertEqual(len({tuple(p[k] for k in ['a','b','domain','reason']) for p in result['pairs']}),72)

    def test_independent_occupancy_masks_match_at_boundaries(self):
        frame=pd.DataFrame({'pixel_channels':[0,99,100,999,1000,4999,5000],
            'bar_channels':np.zeros(7),'tracker_channels':np.zeros(7),
            'all_empty':[True]+[False]*6,'pixel_centered':[True]*7,'large_scintillator':[False]*7})
        for name,mask in domains(frame).items():
            if name.startswith('occupancy_'):
                np.testing.assert_array_equal(mask,domain_mask(frame,name))
        self.assertEqual(domains(frame)['occupancy_0_100'].tolist(),[True,True,False,False,False,False,False])

    def test_bootstrap_without_class_support_has_no_invented_working_point(self):
        frame=pd.DataFrame({'label':[0,1,2,3],'p0':[.7,.1,.1,.1],
            'p1':[.1,.7,.1,.1],'p2':[.1,.1,.7,.1],'p3':[.1,.1,.1,.7]})
        compute=scalar_summary(frame)
        value=compute(np.array([1.,0.,0.,1.]))
        self.assertTrue(np.isnan(value[SCALARS.index('recall_mu')]))
        self.assertTrue(np.isnan(value[SCALARS.index('tau_eff_1pct')]))
        self.assertEqual(value[SCALARS.index('BA3')],1.)

    def test_conditional_extension_requires_finished_reference_and_preserves_it(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            names=[]
            for task in ['T3','T4']:
                name=f'b1_d2_{task.lower()}_s42'
                names.append(name)
                write_json(root/'configs'/f'{name}.json',{'name':name,'family':'B1','task':task,'seed':42,
                    'design':'D2','passes':4,'data_root':'frozen300k','pool_events':300000,'tune_screen':'fixed4k',
                    'validation_non_tau_exposures':75000,'initialization':'scratch','bf16':True,'precision':'BF16_encoder_FP32_head'})
            write_json(root/'campaign.json',{'initial_models':names})
            reason=root/'reason.md'
            reason.write_text('Measured reference still improves; extend its prescribed exposure budget.')
            with self.assertRaisesRegex(ValueError,'reference endpoints'):
                extend(root,'D3','D2',reason,passes=12)
            for name in names:write_json(root/'experiments'/name/'manifest.json',{'status':'complete'})
            extend(root,'D3','D2',reason,passes=12)
            result=configuration(root)
            self.assertEqual(len(result['models']),8)
            self.assertEqual(len(result['pairs']),32)
            for name in names:
                old=json.loads((root/'configs'/f'{name}.json').read_text())
                new=json.loads((root/'configs'/f'{name.replace("_d2_","_d3_")}.json').read_text())
                self.assertEqual(old['passes'],4)
                self.assertEqual(new['passes'],12)
                self.assertEqual(new['initialization'],'scratch')
                self.assertEqual(new['data_root'],old['data_root'])
                self.assertEqual(new['precision'],old['precision'])
            write_json(root/'experiments'/names[1]/'manifest.json',{'status':'running'})
            reason.write_text('Completed T3 reference has an observed Train/Tune gap; change only weight decay.')
            extend(root,'D2WD','D2',reason,weight_decay=.1,tasks=['T3'])
            old=json.loads((root/'configs'/f'{names[0]}.json').read_text())
            new=json.loads((root/'configs'/f'{names[0].replace("_d2_","_d2wd_")}.json').read_text())
            self.assertEqual(new['weight_decay'],.1)
            self.assertEqual(new['passes'],old['passes'])
            self.assertEqual(new['data_root'],old['data_root'])
            self.assertNotIn('weight_decay',old)
            self.assertFalse((root/'configs'/f'{names[1].replace("_d2_","_d2wd_")}.json').exists())

    def test_batch_delivery_waits_for_model_dependencies(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            names=[f'b1_{d}_{t}_s42' for d in ['d0','d1'] for t in ['t3','t4']]
            write_json(root/'campaign.json',{'initial_models':names})
            for name in names:
                write_json(root/'configs'/f'{name}.json',{'family':'B1','design':name.split('_')[1].upper()})
            for name in names[:2]:write_json(root/'experiments'/name/'manifest.json',{'status':'complete'})
            calls=[]
            submit=lambda *args,**kwargs:calls.append((args,kwargs))
            self.assertFalse(advance(root,submit))
            self.assertEqual(len(calls),1)
            self.assertTrue(calls[0][1]['gpu'])
            self.assertEqual(calls[0][0][1],'r2_gap_b1_d0_a1')
            write_json(root/'runtime/manual_gap.submission.json',{'submitted':True,
                'command':['analysis.flavour.v2.r2.gap','run','--root',str(root),'--names',names[0]]})
            calls.clear()
            self.assertFalse(advance(root,submit))
            self.assertEqual(calls[0][0][3][-1:],names[1:2])
            for name in names:
                write_json(root/'experiments'/name/'manifest.json',{'status':'complete'})
                for kind in ['endpoint','best_screen']:
                    write_json(root/'gap'/name/kind/'report.json',{'status':'complete'})
            expected={name+'_'+kind:{} for name in names for kind in ['endpoint','best_screen']}
            write_json(root/'evaluation/results.json',{'status':'complete','models':expected})
            for name in ['independent_points','independent_core_intervals']:
                write_json(root/'evaluation'/f'{name}.json',{'status':'complete'})
            calls.clear()
            self.assertFalse(advance(root,submit))
            self.assertEqual([call[0][1] for call in calls],['r2_evaluate_m4_a1','r2_readout_m4_a1'])
            write_json(root/'readout/results.json',{'status':'complete','models':{name:{} for name in names}})
            calls.clear()
            self.assertTrue(advance(root,submit))
            self.assertEqual([call[0][1] for call in calls],['r2_evaluate_m4_a1','r2_readout_m4_a1','r2_figures_m4_a1'])


if __name__=='__main__':
    unittest.main()
