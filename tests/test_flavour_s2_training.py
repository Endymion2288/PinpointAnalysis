"""Check that interrupted optimization resumes the same scientific run."""

import argparse
import json
import shutil
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from analysis.flavour.s2 import train


class TinySpatialDataset:
    def __init__(self,root,split,config):
        self.config=config
        self.metadata=pd.DataFrame({'label':np.arange(4)})
        self.indexes=np.arange(4)
        self.paths=[Path('unused')]
        self.native=SimpleNamespace(boundaries=[0,4],close=lambda:None)

    def __len__(self):
        return 4

    def __getitem__(self,index):
        nodes=np.zeros((2,18),dtype='float32')
        nodes[:,0]=[index*.1,index*.1+.05]
        nodes[:,3:5]=1
        nodes[:,12]=1
        nodes[:,15]=1
        return {'nodes':nodes,'groups':np.zeros(2,dtype='int64'),'label':index,'index':index}


class TrainingTests(unittest.TestCase):
    def test_occupancy_bucketing_preserves_coverage_and_reduces_padding(self):
        dataset=TinySpatialDataset(None,None,{})
        dataset.indexes=np.arange(16)
        dataset.native.boundaries=[0,16]
        dataset.workload=np.arange(1,17)
        normal=list(train.SpatialBatchSampler(dataset,4,42))
        dataset.config={'bucket_size':16}
        bucketed=list(train.SpatialBatchSampler(dataset,4,42))
        self.assertEqual(sorted(sum(bucketed,[])),list(range(16)))
        self.assertEqual(len(normal),len(bucketed))
        cost=lambda batches:sum(max(dataset.workload[b])*len(b) for b in batches)
        self.assertLessEqual(cost(bucketed),cost(normal))

    def test_domain_sampler_repeats_complete_cycles_to_match_step_budget(self):
        dataset=TinySpatialDataset(None,None,{})
        normal=list(train.SpatialBatchSampler(dataset,3,42))
        repeated=list(train.SpatialBatchSampler(dataset,3,42,steps=5))
        self.assertEqual(len(repeated),5)
        self.assertEqual(repeated[:len(normal)],normal)
        self.assertEqual(sorted(sum(repeated[:2],[])),list(range(4)))
        self.assertEqual(sorted(sum(repeated[2:4],[])),list(range(4)))

    def test_resume_matches_uninterrupted_weights_and_predictions(self):
        self.check_resume({})

    def test_step_warmup_resume_matches_uninterrupted_run(self):
        self.check_resume({'schedule':'warmup_cosine_steps','warmup_steps':1})

    def test_fixed_final_epoch_does_not_select_on_validation_NLL(self):
        config={'name':'fixed','architecture':'part','dim':8,'pair_bias':False,
                'epochs':2,'batch_size':4,'workers':0,'seed':42,'select_final_epoch':True}
        good=np.eye(4)*.6+.1
        bad=np.roll(good,1,axis=1)
        with tempfile.TemporaryDirectory() as temporary,patch.object(train,'SpatialDataset',TinySpatialDataset):
            root=Path(temporary);campaign=root/'campaign.json'
            campaign.write_text(json.dumps({'experiments':[config]}))
            with patch.object(train,'evaluate',side_effect=[(np.arange(4),good),(np.arange(4),bad)]):
                train.run(argparse.Namespace(root=root,campaign=campaign,task=0,device='cpu'))
            output=root/'experiments/fixed'
            manifest=json.loads((output/'manifest.json').read_text())
            history=json.loads((output/'history.json').read_text())
            best=torch.load(output/'best.pt',weights_only=False)
            self.assertLess(history[0]['validation']['NLL'],history[1]['validation']['NLL'])
            self.assertEqual(best['epoch'],2)
            self.assertEqual(manifest['best_epoch'],2)
            self.assertEqual(manifest['selection'],'fixed_final_epoch')
            np.testing.assert_array_equal(pd.read_parquet(output/'validation_predictions.parquet')[[f'p{i}' for i in range(4)]],bad)

    def check_resume(self, overrides):
        config={'name':'test','architecture':'part','dim':8,'pair_bias':False,
                'epochs':2,'batch_size':4,'workers':0,'seed':42}
        config.update(overrides)
        with tempfile.TemporaryDirectory() as temporary,patch.object(train,'SpatialDataset',TinySpatialDataset):
            base=Path(temporary)
            campaign=base/'campaign.json'
            campaign.write_text(json.dumps({'experiments':[config]}))
            def args(root):
                return argparse.Namespace(root=base/root,campaign=campaign,task=0,device='cpu')
            train.run(args('full'))
            original=train.metrics
            calls=0
            def interrupt(y,p):
                nonlocal calls
                calls+=1
                if calls==2:
                    raise RuntimeError('simulated preemption')
                return original(y,p)
            with patch.object(train,'metrics',side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError,'simulated preemption'):
                    train.run(args('resumed'))
            full=base/'full/experiments/test'
            resumed=base/'resumed/experiments/test'
            # Reproduce the observed EOS failure: best/predictions are ahead
            # of committed last.pt, and last.tmp is an incomplete archive.
            for filename in ['best.pt','validation_predictions.parquet','epoch_predictions.parquet','metrics.json']:
                shutil.copyfile(full/filename,resumed/filename)
            (resumed/'last.tmp').write_bytes(b'incomplete archive from failed sync')
            train.run(args('resumed'))
            a=torch.load(full/'last.pt',weights_only=False)
            b=torch.load(resumed/'last.pt',weights_only=False)
            self.assertEqual(a['scheduler'],b['scheduler'])
            for key,value in a['model'].items():
                torch.testing.assert_close(value,b['model'][key],rtol=0,atol=0)
            pd.testing.assert_frame_equal(pd.read_parquet(full/'validation_predictions.parquet'),
                                          pd.read_parquet(resumed/'validation_predictions.parquet'))
            pd.testing.assert_frame_equal(pd.read_parquet(full/'epoch_predictions.parquet'),
                                          pd.read_parquet(resumed/'epoch_predictions.parquet'))
            self.assertEqual(b['epoch'],2)
            train.run(args('resumed'))  # completed invocation is idempotent


if __name__=='__main__':
    unittest.main()
