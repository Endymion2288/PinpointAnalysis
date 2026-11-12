"""S3 geometry, compression information, missing detector and batch isolation."""
import unittest
import numpy as np
import torch

from analysis.flavour.s3.dataset import plane_tokens, collate_planes
from analysis.flavour.s3.models import PlaneModuleClassifier
from analysis.flavour.s2.dataset import collate_spatial


class S3Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.geometry = {'pixel_Xpos': np.arange(100)*.02, 'pixel_Ypos': np.arange(100)*.02,
                         'pixel_Zpos': [0., 10.], 'layer_is_pixel': [1, 0, 1]}

    def observation(self, xy):
        xy = np.asarray(xy, dtype='int64').reshape(-1, 2)
        return {'pixel': {'channel_id': xy[:,1]*100+xy[:,0], 'xyz_mm':np.column_stack([xy*.02, np.zeros(len(xy))])}}

    def item(self, obs, index=0, config=None):
        nodes, groups, modules, _ = plane_tokens(obs, self.geometry, config or {'modalities':['pixel']})
        return {'nodes':nodes,'groups':groups,'modules':modules,'index':index,'label':index%4}

    def test_complete_fine_windows_and_permutation(self):
        obs = self.observation([[i%100, i//100] for i in range(350)])
        original = self.item(obs)
        reversed_obs = {'pixel': {k:v[::-1] for k,v in obs['pixel'].items()}}
        other = self.item(reversed_obs)
        for key in ['nodes','groups','modules']:
            np.testing.assert_array_equal(original[key], other[key])
        self.assertEqual(len(original['nodes']),350)
        self.assertLessEqual(np.bincount(original['groups']).max(),32)
        coarse = self.item(obs, config={'modalities':['pixel'],'window_pixels':0})
        self.assertGreater(len(np.unique(original['groups'])),len(np.unique(coarse['groups'])))

    def test_rebin_does_not_recover_native_residuals(self):
        a = self.observation([[10,10],[20,20]])
        b = self.observation([[11,12],[21,23]])
        for window in [0,16,64]:
            cfg={'modalities':['pixel'],'pitch_mm':1.,'window_pixels':window}
            np.testing.assert_array_equal(self.item(a,config=cfg)['nodes'],self.item(b,config=cfg)['nodes'])
        self.assertFalse(np.array_equal(self.item(a)['nodes'],self.item(b)['nodes']))

    def test_modules_are_hardware_layers_and_signal_ablation(self):
        obs=self.observation([[1,1]])
        obs['bar']={'bounds_mm':np.array([[0,2,0,10,5.],[0,10,0,2,5.]]),'horizontal':np.array([0,1]),
                    'panel':np.array([0,0]),'layer':np.array([1,1]),'edep_proxy_MeV':np.array([1.,2.]),'pe_proxy':np.array([10,20])}
        item=self.item(obs,config={'modalities':['pixel','bar'],'signal':False})
        self.assertEqual(set(item['modules']),{0,1})
        self.assertEqual(float(item['nodes'][:,6:8].sum()),0.)
        batch=collate_planes([item])
        self.assertEqual(set(batch['module_id'].tolist()),{0,1})
        # One module/type's bar H/V are encoded without inventing global xy.
        self.assertEqual(float(batch['module_geometry'][:,:2].sum()),0.)

    def test_linear_packing_matches_complete_reference_batches(self):
        items = [self.item(self.observation([[i % 100, i // 100] for i in range(450)])),
                 self.item(self.observation([]), 1)]
        reference = collate_spatial(items)
        packed = collate_planes(items)
        for key in reference:
            torch.testing.assert_close(packed[key], reference[key], rtol=0, atol=0)
        self.assertEqual(int(packed['module_mask'].sum()), len(packed['module_group']))

    def test_empty_and_missing_inputs_all_fusions_gradients_and_batch_independence(self):
        first=self.item(self.observation([[i,i] for i in range(50)]))
        empty=self.item(self.observation([]),1)
        mixed=collate_planes([first,empty])
        single=collate_planes([first])
        for fusion in ['late','joint','latent']:
            torch.manual_seed(17)
            cfg={'dim':16,'fusion':fusion,'latents':6,'module_queries':2}
            model=PlaneModuleClassifier(cfg).eval()
            p=model(mixed)
            self.assertEqual(p.shape,(2,4));self.assertTrue(torch.isfinite(p).all())
            torch.testing.assert_close(p[:1],model(single),rtol=2e-5,atol=2e-6)
            torch.nn.functional.cross_entropy(p,mixed['label']).backward()
            grads=[x.grad for x in model.parameters() if x.grad is not None]
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads),0)

    def test_bfloat16_autocast_module_packing_and_optimizer(self):
        batch=collate_planes([self.item(self.observation([[i,i] for i in range(40)])),
                              self.item(self.observation([]),1)])
        for fusion,geometry in [('late',False),('joint',True),('latent',True)]:
            with self.subTest(fusion=fusion,geometry=geometry):
                model=PlaneModuleClassifier({'dim':16,'fusion':fusion,'module_geometry':geometry,
                                              'latents':6,'module_queries':2})
                optimizer=torch.optim.AdamW(model.parameters(),lr=.0005)
                with torch.autocast('cpu',dtype=torch.bfloat16):
                    p=model(batch);loss=torch.nn.functional.cross_entropy(p.float(),batch['label'])
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(torch.isfinite(x.grad).all() for x in model.parameters() if x.grad is not None))
                optimizer.step()


if __name__ == '__main__':
    unittest.main()
