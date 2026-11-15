"""Verify fixed-exposure delivery; require a separate evidence-based scaling decision."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis.flavour.inventory import write_json
from analysis.flavour.v2.partitions import KEY, equal_identity_rows
from analysis.flavour.v2.r1.accept import DOMAINS
from analysis.flavour.v2.r2.evaluate import configuration


def learning_gate_matches(gate, recipe):
    keys = ['family', 'task', 'precision', 'modalities', 'architecture_revision', 'empty_response_bypass']
    keys += ['image', 'cnn_width', 'cnn_variant'] if recipe['family']=='B1' else ['dim', 'compile_local']
    defaults = {'cnn_width': 16, 'cnn_variant': 'resnet', 'dim': 64, 'compile_local': False}
    return (gate['status']=='complete' and all(gate['checks'].values())
        and all(gate['config'].get(key, defaults.get(key))==recipe.get(key, defaults.get(key)) for key in keys))


def run(root):
    checks=[]
    def check(name,value,**detail):checks.append({'name':name,'passed':bool(value),**detail})
    def read(path):return json.loads(path.read_text())
    campaign=read(root/'campaign.json')
    check('initial_six_model_contract',set(campaign['initial_models'])==
        {f'b1_{design}_{task}_s42' for design in ['d0','d1','d2'] for task in ['t3','t4']})
    preparation=read(root/'preparation.json')
    check('frozen_300k_plus_Tune_preparation',preparation['status']=='complete' and preparation['events']==340000
        and preparation['all_event_family_keys_exact'] and preparation['PE_conserved'])
    tune=pd.read_parquet(root/'data/catalog.parquet',filters=[('split','=','validation')])
    check('40k_balanced_Tune',len(tune)==40000 and np.array_equal(tune.label.value_counts().sort_index(),[10000]*4))
    gates=[read(p) for p in (root/'smoke').glob('*/integrity.json')]
    check('BF16_T3_T4_full_runner_recovery_gates',any(g['status']=='complete'
        and {c['task'] for c in g['cases']}=={'T3','T4'}
        and all(c['precision']=='BF16_encoder_FP32_head' and c['maximum_parameter_delta']==0 for c in g['cases']) for g in gates))
    fits=[read(p) for p in (root/'fit').glob('*/B*_T*.json')]
    check('BF16_T3_T4_real_learning_gates',all(any(g['task']==task and g['status']=='complete'
        and g['config']['precision']=='BF16_encoder_FP32_head' and all(g['checks'].values()) for g in fits) for task in ['T3','T4']))
    config=configuration(root)
    evaluation=read(root/'evaluation/results.json')
    expected={e['name'] for e in config['models']}
    check('all_prescribed_checkpoint_views',set(evaluation['models'])==expected)
    check('all_eleven_domains',all(set(evaluation['models'][name])==DOMAINS for name in expected))
    identity=lambda pair:tuple(pair[k] for k in ['a','b','domain','reason'])
    check('all_contrasts_once',sorted(map(identity,evaluation['pairs']))==sorted(map(identity,config['pairs'])))
    check('paired_400_block_intervals',evaluation['status']=='complete' and evaluation['replicates']==400)
    exposure_rows=[]
    for name in campaign['initial_models']+campaign.get('extension_models',[]):
        folder=root/'experiments'/name
        record=read(folder/'manifest.json')
        recipe=read(root/'configs'/f'{name}.json')
        preflight=read(Path(recipe['preflight']))
        counters=record['counters']
        check(name+'_matching_actual_architecture_learning_gate',any(learning_gate_matches(g,recipe) for g in fits))
        check(name+'_complete_exact_recipe',record['status']=='complete' and record['config']==recipe
            and record['precision']==recipe['precision'] and record['config']['bf16'])
        check(name+'_prescribed_budget',counters['updates']==preflight['total_updates']
            and counters['training_exposures']==preflight['total_exposures']
            and counters['non_tau_exposures']==preflight['non_tau_exposures']
            and np.isclose(counters['lr_sum'],preflight['lr_sum'],rtol=1e-10,atol=1e-12))
        catalog=pd.read_parquet(Path(recipe['data_root'])/'catalog.parquet',filters=[('split','=','train')])
        if recipe['task']=='T3':catalog=catalog[catalog.label.ne(2)]
        coverage=pd.read_parquet(folder/'train_exposures.parquet')
        check(name+'_exact_Train_identities_and_passes',equal_identity_rows(coverage,catalog)
            and not coverage.duplicated(KEY).any() and coverage.exposures.eq(recipe['passes']).all())
        check(name+'_actual_exposures',int(coverage.exposures.sum())==counters['training_exposures']
            and np.array_equal(np.bincount(coverage.label,weights=coverage.exposures,minlength=4),counters['exposure_by_class']))
        history=record['history']
        minimum=min(row['metrics'][recipe['task']]['NLL'] for row in history)
        check(name+'_best_is_own_task_screen_NLL',record['best_screen']['criterion']==recipe['task']+'_NLL'
            and abs(record['best_screen']['NLL']-minimum)<1e-12
            and any(row['updates']==record['best_screen']['updates'] and abs(row['metrics'][recipe['task']]['NLL']-minimum)<1e-12 for row in history))
        check(name+'_fixed_screen_windows',len(history)>=counters['non_tau_exposures']//75000
            and history[-1]['updates']==counters['updates'])
        health=record['health']
        check(name+'_finite_branch_activations',all(v['finite'] for h in health for v in h['activations'].values()))
        groups=health[0]['gradient_squared_norms']
        check(name+'_all_branch_gradients_and_updates',all(any(np.isfinite(h['gradient_squared_norms'][g])
            and h['gradient_squared_norms'][g]>0 for h in health)
            and any(h['parameter_update_RMS'][g]>0 for h in health) for g in groups))
        for kind,table,checkpoint in [('endpoint','predictions.parquet','exposure.pt'),('best_screen','best_predictions.parquet','best.pt')]:
            predictions=pd.read_parquet(folder/table)
            check(name+'_'+kind+'_exact_Tune',equal_identity_rows(predictions,tune) and not predictions.duplicated(KEY).any())
            p=predictions[['p0','p1','p2','p3'] if recipe['task']=='T4' else ['p0','p1','p3']].to_numpy()
            check(name+'_'+kind+'_valid_nonconstant_probability',np.isfinite(p).all() and (p>=0).all()
                and (p<=1).all() and np.allclose(p.sum(1),1.,atol=2e-6)
                and len(np.unique(p,axis=0))>=100 and p.std(0).max()>1e-3)
            state=torch.load(folder/checkpoint,map_location='cpu',weights_only=False)
            check(name+'_'+kind+'_finite_checkpoint',all(torch.isfinite(v).all() for v in state['model'].values()))
            check(name+'_'+kind+'_checkpoint_position',state['record']['counters']['updates']==
                (counters['updates'] if kind=='endpoint' else record['best_screen']['updates']))
            gap=read(root/'gap'/name/kind/'report.json')
            check(name+'_'+kind+'_fixed_Train_gap',gap['status']=='complete'
                and gap['events']==(3000 if recipe['task']=='T3' else 4000))
            gap_rows=pd.read_parquet(root/'gap'/name/kind/'predictions.parquet')
            selected=pd.read_parquet(root/'gap/train_screen.parquet')
            if recipe['task']=='T3':selected=selected[selected.label.ne(2)]
            check(name+'_'+kind+'_Train_gap_identities',equal_identity_rows(gap_rows,selected))
        check(name+'_sealed_payloads',not record['Cal_A_Cut_B_Confirm_payload_read'])
        exposure_rows.append({'name':name,'design':recipe['design'],'task':recipe['task'],
            'unique_events':len(coverage),'passes':recipe['passes'],**counters,
            'GPU':record['GPU'],'precision':record['precision'],'allocation_id':record['allocation_id'],
            'wall_seconds':record['wall_seconds']})
    for filename in ['independent_points.json','independent_core_intervals.json']:
        report=read(root/'evaluation'/filename)
        check(filename+'_complete',report['status']=='complete' and not report['failed'])
    check('all_contrasts_independently_reviewed',read(root/'evaluation/independent_core_intervals.json')['pairs']==len(config['pairs']))
    readout=read(root/'readout/results.json')
    check('all_endpoint_grouped_readout_diagnostics',readout['status']=='complete'
        and set(readout['models'])==set(campaign['initial_models']+campaign.get('extension_models',[]))
        and readout['fold_family_integrity'] and readout['fold_chunk_integrity']
        and all(v['temperature_argmax_identical'] for v in readout['models'].values()))
    score_review=read(root/'readout/independent_scores.json')
    check('independent_sklearn_score_review',score_review['status']=='complete' and not score_review['failed'])
    if (root/'sampling/contract.json').exists():
        sampling=read(root/'sampling/acceptance.json')
        check('triggered_final_pass_sampling_control',sampling['status']=='complete'
            and sampling['models']==3 and sampling['pairs']==8
            and sampling['all_frozen_identities'] and sampling['same_prefix_events_batch_sizes_LR'])
        check('sampling_replay_figures',all((root/'figures'/f'r2_sampling_replay.{suffix}').is_file()
            for suffix in ['pdf','png']))
    decision=read(root/'decision.json')
    check('scientific_scaling_and_conditional_expansion_decision',decision['status']=='reviewed'
        and all(decision.get(k) for k in ['data_vs_repetition','convergence','capacity','tau_and_common_task',
            'conditional_extensions','selected_reference','remaining_limits']))
    resources=read(root/'resources.json')
    check('all_allocations_terminal',not resources['active_or_unresolved'])
    check('allocation_based_heterogeneous_accounting',resources['heterogeneous_device_hours_are_not_FLOP_equivalent'])
    for name in ['r2_learning_curves','r2_scaling_endpoints','r2_low_occupancy']:
        check(name+'_figures',all((root/'figures'/f'{name}.{suffix}').is_file() for suffix in ['pdf','png']))
    pd.DataFrame(exposure_rows).to_csv(root/'evaluation/exposure_and_compute.csv',index=False)
    failed=[c['name'] for c in checks if not c['passed']]
    report={'status':'complete' if not failed else 'failed','checks':checks,'passed':len(checks)-len(failed),
        'failed':failed,'models':len(exposure_rows),'checkpoint_views':len(expected),'pairs':len(config['pairs']),
        'scope':'R2 prescribed delivery and explicit scientific decision; global optimality is not established by delivery',
        'Cal_A_Cut_B_Confirm_payload_read':False}
    write_json(root/'acceptance.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='checks'}),flush=True)
    if failed:raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    run(args.root)


if __name__=='__main__':main()
