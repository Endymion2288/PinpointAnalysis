"""Require the complete realized R3 matrix, scientific evidence and terminal costs."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis.flavour.inventory import write_json
from analysis.flavour.v2.partitions import KEY,equal_identity_rows
from analysis.flavour.v2.r1.accept import DOMAINS
from analysis.flavour.v2.r3.evaluate import configuration
from analysis.flavour.v2.r3.scientific_review import review as scientific_review


def gate_matches(gate,config):
    keys=['family','task','precision','architecture_revision','empty_response_bypass']
    defaults={'cnn_first_stride':4,'native_relation_adapter':False,'native_relations':False,
        'native_global_ablation':False,'native_pixel_pitch_mm':0.,'relation_material_context':False,
        'relation_plane_policy':'all_visible','native_relation_revision':1}
    keys+=['image','cnn_width','cnn_first_stride','sensor_row_image'] if config['family']=='B1' else [
        'dim','native_relation_adapter','native_relations','native_global_ablation',
        'native_pixel_pitch_mm','relation_material_context','relation_plane_policy','native_relation_revision']
    return gate.get('status')=='complete' and all(gate['config'].get(k,defaults.get(k))==config.get(k,defaults.get(k)) for k in keys)


def run(root):
    def read(path):return json.loads(path.read_text())
    contract=read(root/'completion_contract.json');checks=[]
    def check(name,passed):checks.append({'name':name,'passed':bool(passed)})
    check('explicit_R3_completion_contract',contract['stage']=='R3' and contract['status']=='frozen')
    names=contract['formal_models']
    campaign=read(root/'campaign.json')
    expected=campaign.get('formal_models',campaign['initial_models'])
    check('complete_campaign_model_matrix',set(names)==set(expected) and len(names)==len(expected))
    check('unique_formal_model_matrix',len(set(names))==len(names) and len(names)>=5)
    required_evaluations={('evaluation_'+name,'contracts/'+name+'_evaluation.json')
        for name in ['m0','geometry','c1']}
    check('all_frozen_primary_evaluations_required',required_evaluations.issubset(
        {(e['output'],e['contract']) for e in contract['evaluations']}))
    recipes=[read(root/'configs'/f'{name}.json') for name in names]
    check('M0_full_mixing_in_matrix',any(c.get('design')=='M0' for c in recipes))
    check('G1_stride_control_in_matrix',any(c['family']=='B1' and c.get('cnn_first_stride',4)==2 for c in recipes))
    check('G2_within_family_control_in_matrix',any(c['family'] in ['B2','B3'] and c.get('native_relation_adapter') and not c.get('native_relations') for c in recipes))
    check('G2_prelocal_relations_in_matrix',any(c.get('native_relations') and not c.get('native_global_ablation') for c in recipes))
    check('retrained_global_ablation_in_matrix',any(c.get('native_global_ablation') for c in recipes))
    if any(c.get('design')=='C1' for c in recipes):
        check('compact_sensor_row_model_in_matrix',any(c.get('sensor_row_image') for c in recipes))
    gates=[read(p) for p in (root/'fit').glob('**/B*_T*.json')]
    tune=pd.read_parquet(root/'data/catalog.parquet',filters=[('split','=','validation')])
    delivery=[]
    for name in names:
        folder=root/'experiments'/name;manifest=read(folder/'manifest.json');config=read(root/'configs'/f'{name}.json')
        counters=manifest['counters'];preflight=read(Path(config['preflight']))
        check(name+'_complete',manifest['status']=='complete')
        check(name+'_exact_formal_recipe',manifest['config']==config)
        learning_checks={'fit_accuracy','fit_NLL','nonconstant_probability','branch_gradients',
            'branch_updates','finite_branch_activations'}
        check(name+'_real_learning_gate',any(gate_matches(g,config)
            and learning_checks.issubset(g.get('checks',{}))
            and all(g['checks'][key] for key in learning_checks) for g in gates))
        check(name+'_frozen_optimizer_budget',counters['updates']==preflight['total_updates']
            and counters['training_exposures']==preflight['total_exposures']
            and abs(counters['lr_sum']-preflight['lr_sum'])<1e-9)
        coverage=pd.read_parquet(folder/'train_exposures.parquet')
        original=pd.read_parquet(Path(config['data_root'])/'catalog.parquet',filters=[('split','=','train')])
        if config['task']=='T3':original=original[original.label.ne(2)]
        check(name+'_exact_frozen_Train_and_passes',equal_identity_rows(coverage,original)
            and not coverage.duplicated(KEY).any() and coverage.exposures.eq(config['passes']).all())
        check(name+'_actual_class_exposures',np.array_equal(np.bincount(coverage.label,weights=coverage.exposures,minlength=4),counters['exposure_by_class']))
        cache=manifest.get('training_cache',{})
        check(name+'_conserved_cached_inputs',cache.get('scattered_exact_input_checks',0)>=32)
        for kind,file in [('endpoint','predictions.parquet'),('best_screen','best_predictions.parquet')]:
            predictions=pd.read_parquet(folder/file)
            check(name+'_'+kind+'_exact_Tune',equal_identity_rows(predictions,tune) and not predictions.duplicated(KEY).any())
            p=predictions[['p0','p1','p2','p3'] if config['task']=='T4' else ['p0','p1','p3']].to_numpy()
            check(name+'_'+kind+'_valid_nonconstant_outputs',np.isfinite(p).all() and (p>=0).all()
                and (p<=1).all() and np.allclose(p.sum(1),1.,atol=2e-6) and p.std(0).max()>.001
                and len(np.unique(p,axis=0))>=100)
            state=torch.load(folder/('exposure.pt' if kind=='endpoint' else 'best.pt'),map_location='cpu',weights_only=False)
            check(name+'_'+kind+'_finite_checkpoint',all(torch.isfinite(v).all() for v in state['model'].values()))
            check(name+'_'+kind+'_checkpoint_position',state['record']['counters']['updates']==
                (counters['updates'] if kind=='endpoint' else manifest['best_screen']['updates']))
            del state
        history=manifest['history'];best=manifest['best_screen']
        check(name+'_own_task_best_screen',best['criterion']==config['task']+'_NLL'
            and abs(best['NLL']-min(h['metrics'][config['task']]['NLL'] for h in history))<1e-12)
        check(name+'_endpoint_screen',history[-1]['updates']==counters['updates'])
        health=manifest['health'];groups=health[0]['gradient_squared_norms']
        check(name+'_finite_activations',all(v['finite'] for h in health for v in h['activations'].values()))
        check(name+'_actual_training_branch_gradients_and_updates',all(
            any(np.isfinite(h['gradient_squared_norms'][g]) and h['gradient_squared_norms'][g]>0 for h in health)
            and any(h['parameter_update_RMS'][g]>0 for h in health) for g in groups))
        check(name+'_sealed_payloads',not manifest['Cal_A_Cut_B_Confirm_payload_read'])
        delivery.append({'name':name,'parameters':manifest['parameters'],'GPU':manifest['GPU'],
            'precision':manifest['precision'],'updates':counters['updates'],'exposures':counters['training_exposures']})
    controls=contract['matched_optimizer_controls']
    check('all_five_matched_intervention_controls',{
        'global_event_mixing','first_convolution_stride','compact_image_representation',
        'prelocal_native_relations','retrained_global_path_removal'}.issubset({p['factor'] for p in controls}))
    for pair in controls:
        a,b=(Path(pair[key]) for key in ['a','b'])
        ma,mb=(read(p/'manifest.json') for p in [a,b])
        ca,cb=ma['counters'],mb['counters']
        factor=pair['factor']+'_'+a.name+'_vs_'+b.name
        check(factor+'_same_actual_optimizer_budget',all(ca[k]==cb[k] for k in [
            'updates','training_exposures','non_tau_exposures','exposure_by_class'])
            and abs(ca['lr_sum']-cb['lr_sum'])<1e-9)
        check(factor+'_same_common_training_recipe',all(ma['config'].get(k)==mb['config'].get(k)
            for k in ['task','seed','data_root','passes','batch_size','learning_rate','weight_decay',
                'clip_grad_norm','schedule','precision','warmup_fraction','learning_rate_floor_fraction']))
        xa,xb=(pd.read_parquet(p/'train_exposures.parquet').sort_values(KEY).reset_index(drop=True) for p in [a,b])
        check(factor+'_same_exact_labeled_Train_exposures',xa[KEY+['label','exposures']].equals(xb[KEY+['label','exposures']]))
    for evaluation in contract['evaluations']:
        folder=root/evaluation['output'];config=configuration(root,root/evaluation['contract'])
        result=read(folder/'results.json')
        check(evaluation['output']+'_complete_exact_models',result['status']=='complete'
            and set(result['models'])=={e['name'] for e in config['models']})
        check(evaluation['output']+'_all_domains',all(set(r)==DOMAINS for r in result['models'].values()))
        expected={(p['a'],p['b'],p['domain'],p['reason']) for p in config['pairs']}
        actual={(p['a'],p['b'],p['domain'],p['reason']) for p in result['pairs']}
        check(evaluation['output']+'_exact_pairs',expected==actual and len(result['pairs'])==len(expected))
        check(evaluation['output']+'_400_paired_draws',result['replicates']==400)
        for name in ['independent_points.json','independent_core_intervals.json']:
            review=read(folder/name)
            check(evaluation['output']+'_'+name,review['status']=='complete' and not review['failed'])
    for evidence in contract['required_evidence']:
        record=read(root/evidence['path'])
        check(evidence['name'],record['status']==evidence.get('status','complete'))
    probe_models={};occurrences=Counter()
    for path in contract['probe_evidence']:
        result=read(root/path)
        check(path+'_complete',result['status']=='complete')
        for name,model in result['models'].items():
            probe_models[name]=model;occurrences[name]+=1
    check('all_formal_weights_have_independent_full_Tune_readback',set(names).issubset(probe_models))
    check('one_final_probe_per_formal_model',all(occurrences[n]==1 for n in names))
    for name in names:
        probe=probe_models.get(name,{})
        rb=probe.get('checkpoint_probability_readback',{})
        check(name+'_full_Tune_checkpoint_probability_readback',rb.get('status')=='complete'
            and rb.get('events')==len(tune) and rb.get('maximum_probability_abs_difference',np.inf)<.01
            and rb.get('argmax_agreement',0)>.99)
        ablated=read(root/'configs'/f'{name}.json').get('native_global_ablation',False)
        expected_branches={'local','global','tracker'} if ablated else {'local','regional','global','tracker'}
        reports=probe.get('reports',{})
        check(name+'_actual_readability_paths',set(reports)==expected_branches
            and all(set(value)==DOMAINS for value in reports.values())
            and probe.get('Train_events')==4000 and probe.get('Tune_events')==len(tune))
    health=read(root/contract['health_evidence'])
    check('complete_actual_health_model_matrix',health['status']=='complete' and set(health['models'])==set(names))
    for name,model in health['models'].items():
        check(name+'_reviewed_actual_branch_health',model['finite_activations']
            and model['all_branches_received_gradient_and_update'])
    calculated=scientific_review([root/e['output']/'results.json' for e in contract['evaluations']])
    saved=read(root/contract['scientific_review'])
    check('scientific_limits_recomputed_from_all_evaluations',saved==calculated)
    conditional=read(root/'conditional_decisions.json')
    check('all_conditional_extensions_explicitly_decided',conditional['status']=='reviewed'
        and all(conditional.get(k) for k in ['matched_seeds','resolution','capacity_or_shuffled_relations',
            'anchors','late_fusion','branch_competition','compact_quality_recovery']))
    decision=read(root/'decision.json')
    for name in ['sampling','local_retention','native_relations','information_available_used_useful',
            'seed_resolution_anchor_fusion','compact_sensor_rows','selected_models','remaining_limits']:
        check('scientific_'+name,bool(decision.get(name)))
    check('scientific_decision_reviewed',decision['status']=='reviewed')
    resources=read(root/'resources.json')
    check('all_allocations_terminal',not resources['active_or_unresolved'])
    check('count_allocation_once',resources['worker_wall_seconds_are_not_summed_as_device_hours'])
    for figure in contract['figures']:
        check(figure+'_both_formats',all((root/'figures'/f'{figure}.{suffix}').exists() for suffix in ['pdf','png']))
    failures=[c['name'] for c in checks if not c['passed']]
    result={'status':'complete' if not failures else 'failed','checks':checks,'passed':len(checks)-len(failures),
        'failed':failures,'formal_models':delivery,'Cal_A_Cut_B_Confirm_payload_read':False}
    write_json(root/'acceptance.json',result)
    if failures:raise ValueError('R3 acceptance failed: '+', '.join(failures))
    print(json.dumps({'status':'complete','checks':len(checks),'models':len(names)}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    run(parser.parse_args().root)


if __name__=='__main__':main()
