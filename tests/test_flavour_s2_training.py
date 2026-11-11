"""Check that interrupted optimization resumes the same scientific run."""

import argparse
import json
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
    def test_resume_matches_uninterrupted_weights_and_predictions(self):
        config={'name':'test','architecture':'part','dim':8,'pair_bias':False,
                'epochs':2,'batch_size':4,'workers':0,'seed':42}
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
            train.run(args('resumed'))
            full=base/'full/experiments/test'
            resumed=base/'resumed/experiments/test'
            a=torch.load(full/'last.pt',weights_only=False)
            b=torch.load(resumed/'last.pt',weights_only=False)
            for key,value in a['model'].items():
                torch.testing.assert_close(value,b['model'][key],rtol=0,atol=0)
            pd.testing.assert_frame_equal(pd.read_parquet(full/'validation_predictions.parquet'),
                                          pd.read_parquet(resumed/'validation_predictions.parquet'))
            self.assertEqual(b['epoch'],2)
            train.run(args('resumed'))  # completed invocation is idempotent


if __name__=='__main__':
    unittest.main()
