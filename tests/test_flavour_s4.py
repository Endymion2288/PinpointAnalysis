"""S4 leakage, semantic-target, EMA and encoder-gradient regression checks."""
import copy
import unittest
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from analysis.flavour.s3.dataset import plane_tokens, collate_planes
from analysis.flavour.s3.models_repair import PlaneModuleClassifierV2
from analysis.flavour.s4.dataset import QueryGeometry, collate_pretrain
from analysis.flavour.s4.models import PretrainingModel, objective_loss, variance_covariance
from analysis.flavour.s4.prepare import bar_contributions
from analysis.flavour.s4.pretrain import evaluate,feature_statistics
from analysis.flavour.s4.evaluate import evaluate as evaluate_classification,classwise_operating_points,tau_statistics
from analysis.flavour.s3.metrics import MetricCache
from analysis.flavour.s4.train import build_model as build_downstream
from analysis.flavour.s4.query_baseline import QueryOnly
from analysis.flavour.s4.compute_profile import MatrixOperations,shape_operations
from analysis.flavour.s4.matched_plan import event_shape,compact_batch,nearest_prefix
from analysis.flavour.s4.campaign import host_memory


class S4Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.geometry={'pixel_Xpos':(np.arange(256)*.02-2.56).tolist(),
            'pixel_Ypos':(np.arange(256)*.02-2.56).tolist(),'pixel_Zpos':[-20.,-10.,0.],
            'layer_is_pixel':[1,0,1,1],'numScintLayers':1,'scint_Zpos':[-21.,-18.,-16.,-11.],
            'scint_bar_Xcenter':[-15.,-5.,5.,15.],'scint_bar_Ycenter':[-15.,-5.,5.,15.],
            'scint_detector_width':40.,'scint_detector_height':40.,'scint_bar_width':10.,'scint_bar_height':10.}
        self.qg=QueryGeometry(self.geometry)
        self.config={'dim':16,'latents':8,'region_queries':2,'modalities':['pixel','bar','tracker'],
            'objective':'mae','window_pixels':16,'region_blocks':8,'compile_local':False}

    def observation(self,shift=0):
        ids=(np.arange(150)*13+shift)% (256*256*3)
        c,r,p=self.qg.pixel_parts(ids)
        return {'pixel':{'channel_id':ids,'xyz_mm':np.column_stack([self.qg.x[c],self.qg.y[r],self.qg.z[p]])},
            'bar':{'channel_id':np.array([],dtype='int64'),'bounds_mm':np.empty((0,5)),
                'layer':np.array([],dtype='int64'),'panel':np.array([],dtype='int64'),
                'horizontal':np.array([],dtype='int64'),'bar':np.array([],dtype='int64'),
                'edep_proxy_MeV':np.array([]),'pe_proxy':np.array([],dtype='int64')},
            'tracker_hypothesis':{'station':np.array([],dtype='int64'),'xy_mm':np.empty((0,2)),
                'sigma_xy_mm':np.empty((0,2))}}

    def item(self,shift=0,objective='mae'):
        obs=self.observation(shift);seed=42
        visible,hidden=self.qg.visible(obs,seed,.75)
        n,g,m,c=plane_tokens(visible,self.geometry,self.config)
        result={'nodes':n,'groups':g,'modules':m,'label':-1,'index':shift,'region_blocks':8,'raw_counts':c}
        queries=self.qg.queries(visible,seed,.75,16)
        result.update(queries,reconstruction=self.qg.targets(obs,hidden,queries),mask_counts=np.asarray([[len(h),h.sum()] for h in hidden]))
        if objective=='distill':
            n,g,m,c=plane_tokens(obs,self.geometry,self.config)
            result['teacher']={'nodes':n,'groups':g,'modules':m,'label':-1,'index':shift,'region_blocks':8,'raw_counts':c}
        return result

    def test_hidden_support_changes_neither_encoder_nor_queries(self):
        original=self.observation()
        visible,hidden=self.qg.visible(original,42,.75)
        all_ids=np.arange(256*256*3)
        hidden_ids=all_ids[self.qg.mask(0,all_ids,42,.75)][:100]
        changed=copy.deepcopy(original)
        ids=np.unique(np.r_[original['pixel']['channel_id'][~hidden[0]],hidden_ids])
        c,r,p=self.qg.pixel_parts(ids)
        changed['pixel']={'channel_id':ids,'xyz_mm':np.column_stack([self.qg.x[c],self.qg.y[r],self.qg.z[p]])}
        visible2,_=self.qg.visible(changed,42,.75)
        for name in visible:
            for field in visible[name]:
                np.testing.assert_array_equal(visible[name][field],visible2[name][field])
        for key,value in self.qg.queries(visible,42,.75,16).items():
            np.testing.assert_array_equal(value,self.qg.queries(visible2,42,.75,16)[key])
        a=plane_tokens(visible,self.geometry,self.config)
        b=plane_tokens(visible2,self.geometry,self.config)
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)

    def test_query_negatives_and_uniform_far_coverage_exist(self):
        obs=self.observation();visible,hidden=self.qg.visible(obs,42,.75)
        query=self.qg.queries(visible,42,.75,128)
        targets=self.qg.targets(obs,hidden,query)
        self.assertTrue((targets[:,0]==0).any())
        self.assertTrue((~query['query_near']).any())
        native=query['query_level']==0
        self.assertTrue(self.qg.mask(0,query['query_id'][native],42,.75).all())
        self.assertTrue(np.all(query['query'][:,6:8]==0))
        empty=self.observation();empty['pixel']={k:v[:0] for k,v in empty['pixel'].items()}
        queries=self.qg.queries(empty,42,.75,16)
        self.assertFalse(queries['query_near'].any())

    def test_global_decoder_backpropagates_to_informative_encoder(self):
        model=PretrainingModel(self.config)
        batch=collate_pretrain([self.item(),self.item(19)])
        output=model(batch);loss,parts=objective_loss(output,batch,self.config)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for module in [model.encoder.input,model.encoder.local_pool,model.encoder.latent_read,model.encoder.global_blocks]:
            gradients=[p.grad for p in module.parameters() if p.grad is not None]
            self.assertGreater(sum(float(g.square().sum()) for g in gradients),0)
        self.assertIsNone(model.encoder.head[0].weight.grad)
        self.assertEqual(output['reconstruction'].dtype,torch.float32)

    def test_encoder_feature_api_preserves_classifier_route(self):
        model=PlaneModuleClassifierV2(self.config).eval()
        batch=collate_planes([self.item(),self.item(17)])
        features=model.encode(batch)
        expected=model.head(features['head_input'])+model.linear_readout(features['head_input'])
        torch.testing.assert_close(model(batch),expected,rtol=0,atol=0)
        clone=PlaneModuleClassifierV2(self.config).eval();clone.load_state_dict(model.state_dict(),strict=True)
        torch.testing.assert_close(clone(batch),expected,rtol=0,atol=0)

    def test_downstream_import_keeps_encoder_and_resets_identical_heads(self):
        pretrained=PretrainingModel(self.config)
        with torch.no_grad():pretrained.encoder.input[0].weight.add_(1.)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint=Path(temporary)/'best.pt'
            torch.save({'model':pretrained.state_dict()},checkpoint)
            torch.manual_seed(123);scratch=build_downstream(self.config)
            torch.manual_seed(123);initialized=build_downstream({**self.config,'pretrained_checkpoint':str(checkpoint)})
            torch.testing.assert_close(initialized.input[0].weight,pretrained.encoder.input[0].weight,rtol=0,atol=0)
            for prefix in ['head','linear_readout','head_input_norm','branch_norms','summary']:
                for key,value in getattr(scratch,prefix).state_dict().items():
                    torch.testing.assert_close(getattr(initialized,prefix).state_dict()[key],value,rtol=0,atol=0)
            torch.save({'model':scratch.state_dict()},checkpoint)
            resumed=build_downstream({**self.config,'supervised_initial_checkpoint':str(checkpoint)})
            for key,value in scratch.state_dict().items():
                torch.testing.assert_close(resumed.state_dict()[key],value,rtol=0,atol=0)

    def test_ema_no_teacher_gradients_and_variance_opposes_constant(self):
        config={**self.config,'objective':'distill'}
        model=PretrainingModel(config)
        batch=collate_pretrain([self.item(0,'distill'),self.item(73,'distill')])
        out=model(batch);loss,_=objective_loss(out,batch,config);loss.backward()
        self.assertTrue(all(p.grad is None for p in model.teacher.parameters()))
        before=model.teacher.input[0].weight.detach().clone()
        with torch.no_grad():model.encoder.input[0].weight.add_(1)
        model.update_teacher(.9)
        torch.testing.assert_close(model.teacher.input[0].weight,before+.1)
        const=variance_covariance(torch.ones((8,16)))[0]
        variable=variance_covariance(torch.randn((8,16)))[0]
        self.assertGreater(float(const),float(variable))

    def test_soft_bar_energy_is_aggregated_over_tracks(self):
        record={'layerID':[0,0,0],'panelID':[0,0,0],'isHorizontal':[0,0,0],
            'rowID':[-1,-1,-1],'colID':[1,1,2],'pdg':[11,13,211],'edep':[1.,3.,8.]}
        ids,energy,fractions,valid=bar_contributions(record,self.geometry)
        np.testing.assert_allclose(energy,[4,8]);np.testing.assert_allclose(fractions,[[.25,.75,0,0],[0,0,1,0]])
        self.assertTrue(valid.all());self.assertEqual(len(ids),2)

    def test_partial_Rel_alone_reaches_global_and_local_encoders(self):
        config={**self.config,'objective':'mae_rel'};model=PretrainingModel(config)
        batch=collate_pretrain([self.item(),self.item(19)])
        batch['rel_query']=batch['query'][:,:8]
        batch['rel_valid']=torch.ones((2,8),dtype=torch.bool)
        batch['rel_target']=torch.rand((2,8,4)).softmax(-1)
        output=model(batch)
        loss=-(batch['rel_target']*output['rel'].log_softmax(-1)).sum(-1).mean()
        loss.backward()
        self.assertGreater(float(model.encoder.input[0].weight.grad.square().sum()),0.)
        self.assertGreater(float(model.encoder.latent_read.q.weight.grad.square().sum()),0.)

    def test_kept_bar_route_backpropagates_with_global_features_detached(self):
        items=[]
        for shift in [0,19]:
            obs=self.observation(shift)
            obs['bar']={'channel_id':self.qg.bar_ids[:4],'bounds_mm':np.array([
                [x-5,x+5,-20,20,-21] for x in [-15.,-5.,5.,15.]]),
                'layer':np.zeros(4,dtype=int),'panel':np.zeros(4,dtype=int),
                'horizontal':np.zeros(4,dtype=int),'bar':np.arange(4),
                'edep_proxy_MeV':np.arange(1,5,dtype=float),'pe_proxy':np.arange(1,5)*10}
            n,g,m,c=plane_tokens(obs,self.geometry,self.config)
            item=self.item(shift);item.update(nodes=n,groups=g,modules=m,raw_counts=c);items.append(item)
        batch=collate_pretrain(items);batch['rel_query']=batch['query'][:,:8]
        for source in ['global_and_bar_local','global_and_bar_channels']:
            config={**self.config,'objective':'mae_rel','rel_source':source};model=PretrainingModel(config)
            original=model.encoder.encode
            def detached_global(value):
                result=original(value);result['global_tokens']=result['global_tokens'].detach();return result
            with patch.object(model.encoder,'encode',side_effect=detached_global):
                logits=model(batch)['rel']
            loss=-logits.log_softmax(-1)[...,0].mean();loss.backward()
            self.assertGreater(float(model.encoder.input[0].weight.grad.square().sum()),0.)
            batch['rel_valid']=torch.ones((2,8),dtype=torch.bool)
            batch['rel_target']=torch.rand((2,8,4)).softmax(-1)
            model.zero_grad(set_to_none=True);counter=MatrixOperations(model)
            loss,_=objective_loss(model(batch),batch,config)
            counter.backward=True;loss.backward();counter.close()
            self.assertEqual(shape_operations(config,batch)['matrix_operations'],sum(counter.counts.values()),source)

    def test_checkpoint_restores_ema_optimizer_and_rng(self):
        config={**self.config,'objective':'distill'}
        torch.manual_seed(42)
        model=PretrainingModel(config)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0003)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,4)
        batch=collate_pretrain([self.item(0,'distill'),self.item(73,'distill')])
        def step(network,opt,schedule):
            opt.zero_grad(set_to_none=True)
            loss,_=objective_loss(network(batch),batch,config)
            loss.backward();opt.step();schedule.step();network.update_teacher(.996)
            return loss.detach()
        step(model,optimizer,scheduler)
        buffer=io.BytesIO()
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),'rng':torch.get_rng_state()},buffer)
        expected=step(model,optimizer,scheduler)
        clone=PretrainingModel(config)
        clone_optimizer=torch.optim.AdamW(clone.parameters(),lr=.0003)
        clone_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(clone_optimizer,4)
        buffer.seek(0);state=torch.load(buffer,weights_only=False)
        clone.load_state_dict(state['model']);clone_optimizer.load_state_dict(state['optimizer'])
        clone_scheduler.load_state_dict(state['scheduler']);torch.set_rng_state(state['rng'])
        actual=step(clone,clone_optimizer,clone_scheduler)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        for key,value in model.state_dict().items():
            torch.testing.assert_close(clone.state_dict()[key],value,rtol=0,atol=0)

    def test_complete_reconstruction_evaluation_has_one_row_per_query(self):
        model=PretrainingModel(self.config).eval()
        batch=collate_pretrain([self.item(),self.item(19)])
        report,frame=evaluate(model,[batch],torch.device('cpu'),False,self.config)
        self.assertEqual(len(frame),2*3*16)
        self.assertEqual(set(frame.level),{0,1,2})
        self.assertEqual(set(report['event_permutation_delta_BCE']),{'0','1','2'})
        self.assertTrue(np.isfinite(frame.to_numpy()).all())

    def test_query_only_baseline_cannot_use_event_encoder_information(self):
        model=QueryOnly(self.config).eval()
        batch=collate_pretrain([self.item(),self.item(19)])
        expected=model(batch)['reconstruction']
        batch['student']['nodes'].fill_(123.)
        torch.testing.assert_close(model(batch)['reconstruction'],expected,rtol=0,atol=0)
        report,_=evaluate(model,[batch],torch.device('cpu'),False,self.config)
        self.assertEqual(report['event_permutation_delta_BCE'],{'0':0.,'1':0.,'2':0.})

    def test_patch_target_retains_binary_count_and_subpixel_centroid(self):
        geometry=QueryGeometry(self.geometry,pixel_query_patch=16)
        obs=self.observation();visible,hidden=geometry.visible(obs,42,.75)
        queries=geometry.queries(visible,42,.75,64)
        target=geometry.targets(obs,hidden,queries)
        ids=np.asarray(obs['pixel']['channel_id'])[hidden[0]]
        col,row,plane=geometry.pixel_parts(ids)
        patch=(plane*geometry.qy+row//16)*geometry.qx+col//16
        for j in np.flatnonzero(queries['query_level']==0):
            selected=patch==queries['query_id'][j]
            self.assertEqual(target[j,0],bool(selected.any()))
            self.assertAlmostEqual(float(target[j,1]),float(np.log1p(selected.sum())/10),places=6)
            if selected.any():
                expected=np.column_stack([(col[selected]%16+.5)/16-.5,(row[selected]%16+.5)/16-.5]).mean(0)
                np.testing.assert_allclose(target[j,2:4],expected,rtol=1e-6,atol=1e-6)

    def test_multiscale_ema_target_still_backpropagates_through_global(self):
        config={**self.config,'objective':'distill','teacher_multiscale':True}
        model=PretrainingModel(config)
        batch=collate_pretrain([self.item(0,'distill'),self.item(73,'distill')])
        out=model(batch);loss,_=objective_loss(out,batch,config);loss.backward()
        self.assertGreater(float(model.encoder.latent_read.q.weight.grad.square().sum()),0)
        self.assertTrue(all(p.grad is None for p in model.teacher.parameters()))

    def test_feature_spectrum_detects_rank_one_despite_nonzero_variance(self):
        x=torch.linspace(-1,1,128)[:,None].expand(-1,16)
        measured=feature_statistics(x)
        self.assertGreater(measured['between_event_RMS'],.5)
        self.assertAlmostEqual(measured['centered_effective_rank'],1.,places=5)
        self.assertAlmostEqual(measured['leading_variance_fraction'],1.,places=5)
        self.assertEqual(measured['dimensions_for_90pct_variance'],1)

    def test_tall_probe_covariance_spectrum_matches_direct_SVD(self):
        torch.manual_seed(41)
        x=torch.randn((2048,24))@torch.diag(torch.logspace(0,-3,24))
        measured=feature_statistics(x)
        singular=torch.linalg.svdvals(x-x.mean(0));probability=singular.square()/singular.square().sum()
        expected=float(torch.exp(-(probability*probability.log()).sum()))
        self.assertEqual(measured['spectrum_method'],'centered_FP64_covariance_eigh')
        self.assertAlmostEqual(measured['centered_effective_rank'],expected,places=5)
        np.testing.assert_allclose(measured['first_eight_variance_fractions'],probability[:8],rtol=2e-6,atol=1e-8)

    def test_training_matrix_count_includes_only_required_linear_backwards(self):
        for needs_input_gradient,expected in [(False,96),(True,144)]:
            model=torch.nn.Linear(3,4,bias=False);counter=MatrixOperations(model)
            value=torch.ones((2,3),requires_grad=needs_input_gradient)
            result=model(value).sum();counter.backward=True;result.backward();counter.close()
            self.assertEqual(sum(counter.counts.values()),expected)

    def test_shape_budget_matches_actual_forward_backward_hooks(self):
        for objective in ['mae','mae_rel','distill','classification']:
            config={**self.config,'objective':objective}
            if objective=='classification':
                model=PlaneModuleClassifierV2(config).train();batch=collate_planes([self.item(),self.item(19)])
                counter=MatrixOperations(model);loss=model(batch).square().mean()
            else:
                model=PretrainingModel(config).train()
                batch=collate_pretrain([self.item(0,objective),self.item(19,objective)])
                if objective=='mae_rel':
                    batch['rel_query']=batch['query'][:,:8]
                    batch['rel_valid']=torch.ones((2,8),dtype=torch.bool)
                    batch['rel_target']=torch.rand((2,8,4)).softmax(-1)
                counter=MatrixOperations(model);loss,_=objective_loss(model(batch),batch,config)
            counter.backward=True;loss.backward();counter.close()
            self.assertEqual(shape_operations(config,batch)['matrix_operations'],sum(counter.counts.values()),objective)

    def test_compact_shapes_preserve_real_ragged_batch_cost_and_budget_rounding(self):
        items=[self.item(),self.item(19),self.item(172)]
        items[1]['region_blocks']=1;items[2]['region_blocks']=2
        actual=collate_planes(items)
        compact=compact_batch([event_shape(item) for item in items])
        config={**self.config,'objective':'classification'}
        self.assertEqual(shape_operations(config,actual),shape_operations(config,compact))
        costs=[[10,20,30],[12,25,35]]
        self.assertEqual(nearest_prefix(costs,70),(2,67))
        self.assertEqual(nearest_prefix(costs,110),(3,132))

    def test_allocation_memory_limit_includes_worker_cgroup_and_unlimited_case(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for name,value in [('memory.current','5000'),('memory.max','10000'),('memory.peak','6500')]:
                (root/name).write_text(value+'\n')
            self.assertEqual(host_memory(root),{'current_bytes':5000,'limit_bytes':10000,'peak_bytes':6500})
            (root/'memory.max').write_text('max\n')
            self.assertNotIn('limit_bytes',host_memory(root))

    def test_validation_evaluation_filters_ssl_and_pairs_identical_events(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);n=80;y=np.arange(n)%4
            frame=pd.DataFrame({'run':10081+y,'chunk':np.arange(n)//20,'row':np.arange(n),
                'label':y,'family_hi':np.zeros(n,dtype='uint64'),'family_lo':np.arange(n,dtype='uint64'),
                'initE':np.full(n,200000.),'initX':np.full(n,10.),'initY':np.full(n,10.),'initZ':np.full(n,-2400.),
                'pixel_centered':np.ones(n,dtype=bool),'large_scintillator':np.ones(n,dtype=bool),
                'all_empty':np.zeros(n,dtype=bool),'pixel_channels':np.full(n,100),
                'bar_channels':np.full(n,20),'tracker_channels':np.ones(n,dtype=int),
                'tau_branch':np.full(n,'hadronic')})
            p=np.eye(4)[y]*.6+.1
            for c in range(4):frame[f'p{c}']=p[:,c]
            frame.to_parquet(root/'catalog.parquet',index=False)
            configurations=[]
            for name,program in [('a','train'),('b','probes'),('ssl','pretrain'),('continuation','train')]:
                config={'name':name,'program':program,'label_budget':1000,'modalities':['pixel','bar','tracker']}
                if name=='continuation':config['supervised_initial_checkpoint']=str(root/'experiments/a/best.pt')
                configurations.append(config)
                directory=root/'experiments'/name;directory.mkdir(parents=True)
                (directory/'manifest.json').write_text(json.dumps({'status':'complete','best_NLL':2. if name=='continuation' else 1.}))
                filename='full_linear_predictions.parquet' if program=='probes' else 'validation_predictions.parquet'
                frame.to_parquet(directory/filename,index=False)
                if program=='probes':frame.to_parquet(directory/'global_linear_predictions.parquet',index=False)
            campaign=root/'campaign.json';campaign.write_text(json.dumps({'experiments':configurations}))
            pairs=root/'pairs.json';pairs.write_text(json.dumps({'pairs':[{'a':'a','b':'b'},
                {'a':'b__global','b':'b'}],'include_global_frozen':True,'paired_tau_branches':True,
                'include_supervised_selection_envelope':True,
                'seed_mean_comparisons':[{'name':'identical','pairs':[{'a':'a','b':'b'}]}]}))
            args=SimpleNamespace(root=root,output=root/'evaluation',campaign=[campaign],comparisons=pairs,bootstrap=4,strict=True)
            with patch('analysis.flavour.s4.evaluate.plot_curves'):
                evaluate_classification(args)
            result=json.loads((args.output/'summary.json').read_text())
            self.assertEqual(len(result['experiments']),5)
            self.assertTrue(result['requested_complete'])
            self.assertEqual(result['common_support']['retained_events'],n)
            self.assertEqual(result['paired_comparisons'][0]['metrics']['raw']['macro_AP']['delta'],0.)
            self.assertEqual(result['paired_comparisons'][0]['metrics']['raw']['macro_AP']['CI95'],[0.,0.])
            self.assertEqual(result['paired_comparisons'][1]['metrics']['raw']['macro_AP']['CI95'],[0.,0.])
            for comparison in [result['paired_comparisons'][0],result['seed_mean_comparisons'][0]]:
                self.assertEqual(comparison['tau_branches']['hadronic__common_support']['AP']['CI95'],[0.,0.])
                self.assertEqual(comparison['tau_branches']['electron__raw']['AP']['CI95'],[None,None])
            selected=next(x for x in result['experiments'] if x['name']=='continuation__selection')
            self.assertEqual(selected['config']['selection_envelope']['selected'],'a')

    def test_tau_branch_weighted_statistics_keep_ties_and_exclude_other_branches(self):
        from sklearn.metrics import average_precision_score,roc_auc_score,roc_curve
        labels=np.array([2,2,0,1,3,2,0,1])
        score=np.array([.8,.8,.8,.3,.3,.2,.2,.1])
        probabilities=np.tile((1-score[:,None])/3,(1,4));probabilities[:,2]=score
        weights=np.array([2.,0.,1.,.4,.6,3.,2.,1.])
        result=tau_statistics(MetricCache(labels,probabilities),weights)
        truth=labels==2
        self.assertAlmostEqual(result[0],average_precision_score(truth,score,sample_weight=weights))
        self.assertAlmostEqual(result[1],roc_auc_score(truth,score,sample_weight=weights))
        fpr,tpr,_=roc_curve(truth,score,sample_weight=weights,drop_intermediate=False)
        for measured,acceptance in zip(result[2:],[.01,.001]):
            self.assertEqual(measured,tpr[fpr<=acceptance][-1])
        self.assertTrue(np.isnan(tau_statistics(MetricCache(labels,probabilities),weights*(~truth))).all())

    def test_four_class_operating_points_preserve_ties_and_finite_counts(self):
        frame=pd.DataFrame({'label':np.arange(40)%4,**{f'p{c}':np.full(40,.25) for c in range(4)}})
        result=classwise_operating_points(frame)
        self.assertEqual(len(result),4)
        for points in result.values():
            for point in points.values():
                self.assertEqual(point['signal_selected'],0)
                self.assertEqual(point['background_selected'],0)
                self.assertEqual(point['background_events'],30)
                self.assertGreater(point['background_count_interval_95'][1],0)


if __name__=='__main__':unittest.main()
