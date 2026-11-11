"""Paired S2 validation comparisons, conditions, finite tails and uncertainty."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve

from analysis.flavour.inventory import write_json
from analysis.flavour.s2.train import metrics


def tau_metrics(frame,weight=None):
    truth=frame.label.to_numpy()==2
    score=frame.p2.to_numpy()
    if not truth.any() or not (~truth).any():
        return {'events':len(frame),'tau_events':int(truth.sum()),'PR_AUC':None,'ROC_AUC':None}
    out={'events':len(frame),'tau_events':int(truth.sum()),
         'PR_AUC':float(average_precision_score(truth,score,sample_weight=weight)),
         'ROC_AUC':float(roc_auc_score(truth,score,sample_weight=weight)),
         'tau_prevalence':float(np.average(truth,weights=weight))}
    if weight is not None:
        out['Neff']=float(np.sum(weight)**2/np.sum(weight**2))
    fpr,tpr,threshold=roc_curve(truth,score,sample_weight=weight,drop_intermediate=False)
    for acceptance in [.01,.001]:
        valid=np.flatnonzero(fpr<=acceptance)
        i=int(valid[-1])
        cut=threshold[i]
        selected=score>=cut
        background=(~truth)&selected
        signal=truth&selected
        value={'threshold':float(cut) if np.isfinite(cut) else None,
               'tau_efficiency':float(tpr[i]),'background_acceptance':float(fpr[i]),
               'background_selected':int(background.sum()),'signal_selected':int(signal.sum()),
               'tail_statistics_sufficient':int(background.sum())>=25,
               'threshold_scope':'validation_empirical_curve_S2_comparison_not_final_test_operating_point'}
        if weight is not None:
            value['weighted_signal_selected']=float(np.sum(weight[signal]))
            value['weighted_background_selected']=float(np.sum(weight[background]))
        out[f'FPR_{acceptance:g}']=value
    return out


def common_support_weights(frame):
    """Joint energy/vertex overlap on fixed bins, no extrapolation or refitting.

    Reference density per cell = min class count; weights <=1. Each retained
    cell requires at least five events of every class. Explicit benchmark prior,
    not physical flux/exposure weights.
    """
    edges=[np.array([0,100,300,1000,2500,6000])*1000,
           np.array([-1000,-100,0,100,1000]),np.array([-1000,-80,0,80,1000]),
           np.array([-4000,-2600,-2200,-1800,0])]
    values=[frame.initE,frame.initX,frame.initY,frame.initZ]
    bins=np.column_stack([np.digitize(v,e)-1 for v,e in zip(values,edges)])
    in_range=np.all(np.column_stack([(b>=0)&(b<len(e)-1) for b,e in zip(bins.T,edges)]),axis=1)
    cell=np.ravel_multi_index(tuple(np.clip(b,0,len(e)-2) for b,e in zip(bins.T,edges)),
                              tuple(len(e)-1 for e in edges))
    tab=pd.crosstab(cell[in_range],frame.label.to_numpy()[in_range]).reindex(columns=range(4),fill_value=0)
    minimum=tab.min(axis=1)
    valid=minimum>=5
    weight=np.zeros(len(frame))
    for c in range(4):
        per_cell=(minimum[valid]/tab.loc[valid,c]).to_dict()
        flag=frame.label.to_numpy()==c
        weight[flag]=[per_cell.get(i,0.) for i in cell[flag]]
    weight[~in_range]=0
    return weight,{'edges':{key:e.tolist() for key,e in zip(['energy_MeV','x_mm','y_mm','z_mm'],edges)},
        'minimum_events_per_class_per_cell':5,'supported_cells':int(valid.sum()),
        'total_populated_cells':len(tab),'retained_events':int((weight>0).sum()),
        'weight_cap':1.,'class_weight_sums':[float(weight[frame.label.eq(c)].sum()) for c in range(4)],
        'Neff':float(weight.sum()**2/np.sum(weight**2)) if np.any(weight) else 0,
        'scope':'energy_vertex_joint_common_support_diagnostic_no_flux_weights'}


def paired_interval(a,b,replicates=400,seed=2026):
    """Stratified paired chunk bootstrap; seeds never become event replicas."""
    key=['run','chunk','row']
    joined=a[key+['label','p2']].merge(b[key+['label','p2']],on=key,validate='one_to_one',suffixes=('_a','_b'))
    if len(joined)!=len(a) or len(joined)!=len(b) or not joined.label_a.eq(joined.label_b).all():
        raise ValueError('paired comparison has different validation events')
    y=joined.label_a.to_numpy()==2
    pa,pb=joined.p2_a.to_numpy(),joined.p2_b.to_numpy()
    block=joined.groupby(['run','chunk']).indices
    groups={c:[ix for (run,_),ix in block.items() if run==10081+c] for c in range(4)}
    rng=np.random.default_rng(seed)
    differences=[]
    for _ in range(replicates):
        index=np.concatenate([np.concatenate([g[i] for i in rng.integers(0,len(g),len(g))])
                              for g in groups.values()])
        differences.append(average_precision_score(y[index],pb[index])-average_precision_score(y[index],pa[index]))
    low,high=np.quantile(differences,[.025,.975])
    delta=average_precision_score(y,pb)-average_precision_score(y,pa)
    return {'metric':'tau_average_precision_B_minus_A','delta':float(delta),
        'CI95':[float(low),float(high)],'replicates':replicates,'resampling':'paired_class_stratified_chunk_bootstrap',
        'events':len(joined),'minimum_relevant_gain':.01,
        'can_exclude_gain_ge_0.01':bool(high<.01),'positive_gain_supported':bool(low>0),
        'equivalence_claim':False}


def factorial_contrast(frames,replicates=400,seed=2026):
    """Interaction = projection loss at coarse minus loss at native pitch.

    Positive means the combined compression is less damaging than the sum
    of the two individual losses, on the AP scale. Resample identical chunks
    for all four cells; this is distinct from seed-to-seed variation.
    """
    key=['run','chunk','row']
    reference=frames[0].sort_values(key).reset_index(drop=True)
    scores=[]
    for frame in frames:
        frame=frame.sort_values(key).reset_index(drop=True)
        if not frame[key+['label']].equals(reference[key+['label']]):
            raise ValueError('factorial contrast has different validation events')
        scores.append(frame.p2.to_numpy())
    y=reference.label.to_numpy()==2
    coefficients=np.array([1.,-1.,-1.,1.])  # NJ, NP, CJ, CP
    def statistic(index):
        return float(np.dot(coefficients,[average_precision_score(y[index],s[index]) for s in scores]))
    blocks=reference.groupby(['run','chunk']).indices
    groups=[[ix for (run,_),ix in blocks.items() if run==10081+c] for c in range(4)]
    rng=np.random.default_rng(seed)
    samples=[]
    for _ in range(replicates):
        index=np.concatenate([np.concatenate([g[i] for i in rng.integers(0,len(g),len(g))]) for g in groups])
        samples.append(statistic(index))
    return {'interaction':statistic(np.arange(len(y))),'CI95':np.quantile(samples,[.025,.975]).tolist(),
            'definition':'AP_CP - AP_CJ - AP_NP + AP_NJ','replicates':replicates,
            'resampling':'paired_class_stratified_chunk_bootstrap'}


def matched_time(a,b,root,frames):
    paths=[root/'experiments'/name for name in [a,b]]
    manifests=[json.loads((p/'manifest.json').read_text()) for p in paths]
    if manifests[0]['gpu_name']!=manifests[1]['gpu_name']:
        return {'status':'different_GPU_models','GPU_names':[m['gpu_name'] for m in manifests],
                'comparison_measured':False}
    histories=[json.loads((p/'history.json').read_text()) for p in paths]
    budget=min(h[-1]['wall_seconds'] for h in histories)
    selected=[]
    predictions=[]
    for name,path,history in zip([a,b],paths,histories):
        eligible=[r for r in history if r['wall_seconds']<=budget]
        if not eligible:
            return {'status':'no_complete_epoch_within_shared_time','budget_seconds':budget,
                    'comparison_measured':False}
        record=min(eligible,key=lambda r:r['validation']['NLL'])
        ep=pd.read_parquet(path/'epoch_predictions.parquet')
        ep=ep[ep.epoch.eq(record['epoch'])].sort_values('row')
        frame=frames[name].copy()
        if len(ep)!=len(frame) or not np.array_equal(ep.row,np.arange(len(frame))):
            raise ValueError('epoch probabilities do not cover the validation rows')
        frame[[f'p{c}' for c in range(4)]]=ep[[f'p{c}' for c in range(4)]].to_numpy()
        predictions.append(frame)
        selected.append({'experiment':name,'epoch':record['epoch'],'seconds':record['wall_seconds'],
                         'unused_seconds':budget-record['wall_seconds'],'tau_AP':tau_metrics(frame)['PR_AUC']})
    return {'status':'measured_at_epoch_resolution','comparison_measured':True,'budget_seconds':budget,
            'GPU':manifests[0]['gpu_name'],'selection':'minimum_validation_NLL_within_allocated_wall_time',
            'runs':selected,'tau_AP_B_minus_A':selected[1]['tau_AP']-selected[0]['tau_AP']}


def seed_spread(experiments):
    groups={}
    for experiment in experiments:
        config={k:v for k,v in experiment['config'].items() if k not in ['seed','name','role']}
        key=json.dumps(config,sort_keys=True)
        groups.setdefault(key,[]).append(experiment)
    summaries=[]
    for key,group in groups.items():
        seeds=[e['config']['seed'] for e in group]
        if len(set(seeds))!=len(seeds):
            raise ValueError('duplicate seed in a seed-spread group')
        values=[e['tau']['PR_AUC'] for e in group]
        summaries.append({'config':json.loads(key),'seeds':seeds,'tau_AP_mean':float(np.mean(values)),
            'tau_AP_std':float(np.std(values,ddof=1)) if len(values)>1 else None,
            'three_seed_confirmation_complete':set([17,42,2026]).issubset(seeds),
            'interpretation':'training_seed_spread_not_physical_event_uncertainty'})
    return summaries


def error_groups(frame,name):
    """Deterministic examples of every confusion pair and random references."""
    frame=frame.copy()
    probability=frame[[f'p{c}' for c in range(4)]].to_numpy()
    frame['prediction']=probability.argmax(1)
    frame['confidence']=probability.max(1)
    frame['entropy']=-np.sum(probability*np.log(np.maximum(probability,1e-12)),axis=1)
    samples=[]
    for (truth,prediction),group in frame.groupby(['label','prediction']):
        selected=group.nlargest(5,'confidence').copy()
        selected['example_group']=f'true_{truth}_pred_{prediction}_high_confidence'
        samples.append(selected)
    selected=frame.nlargest(12,'entropy').copy()
    selected['example_group']='high_entropy'
    samples.append(selected)
    selected=frame.sample(min(12,len(frame)),random_state=2026).copy()
    selected['example_group']='random_reference'
    samples.append(selected)
    out=pd.concat(samples,ignore_index=True)
    out['experiment']=name
    return out


def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    experiments=[]
    frames={}
    strata=[]
    examples=[]
    factorial={}
    catalog=pd.read_parquet(args.root/'catalog.parquet',columns=['run','chunk','row','all_empty',
                                                               'pixel_channels','bar_channels','tracker_channels'])
    for path in sorted((args.root/'experiments').iterdir()):
        if not (path/'manifest.json').exists():
            continue
        manifest=json.loads((path/'manifest.json').read_text())
        if manifest.get('status')!='complete':
            continue
        frame=pd.read_parquet(path/'validation_predictions.parquet')
        frame=frame.merge(catalog,on=['run','chunk','row'],how='left',validate='one_to_one')
        if frame.all_empty.isna().any():
            raise ValueError('validation predictions absent from S2 catalog')
        frames[path.name]=frame
        examples.append(error_groups(frame,path.name))
        m=metrics(frame.label.to_numpy(),frame[[f'p{c}' for c in range(4)]].to_numpy())
        tau=tau_metrics(frame)
        weight,support=common_support_weights(frame)
        matched=tau_metrics(frame.loc[weight>0],weight[weight>0]) if np.any(weight>0) else {}
        experiments.append({'name':path.name,'config':manifest['config'],'metrics':m,'tau':tau,
                            'matched_tau':matched,'common_support':support,
                            'parameters':manifest['parameters'],'gpu':manifest['gpu_name'],
                            'gpu_hours':manifest.get('gpu_hours'),'peak_gpu_mib':manifest.get('peak_gpu_allocated_mib')})
        masks={'all':np.ones(len(frame),dtype=bool),'pixel_centered':frame.pixel_centered,
               'large_scintillator':frame.large_scintillator}
        if 'all_empty' in frame:
            masks.update(detector_active=~frame.all_empty,all_empty=frame.all_empty)
        for low,high in [(0,100),(100,300),(300,1000),(1000,2500),(2500,6000)]:
            masks[f'energy_{low}_{high}_GeV']=frame.initE.ge(low*1000)&frame.initE.lt(high*1000)
        for low,high in [(-2993,-2600),(-2600,-2200),(-2200,-1800)]:
            masks[f'vertex_z_{low}_{high}_mm']=frame.initZ.ge(low)&frame.initZ.lt(high)
        masks.update(pixel_empty=frame.pixel_channels.eq(0),pixel_1_99=frame.pixel_channels.between(1,99),
                     pixel_100_999=frame.pixel_channels.between(100,999),pixel_ge_1000=frame.pixel_channels.ge(1000),
                     tracker_active=frame.tracker_channels.gt(0),tracker_empty=frame.tracker_channels.eq(0))
        for name,mask in masks.items():
            result=tau_metrics(frame.loc[mask])
            fourclass=metrics(frame.loc[mask,'label'].to_numpy(),frame.loc[mask,[f'p{c}' for c in range(4)]].to_numpy()) if mask.any() else None
            strata.append({'experiment':path.name,'stratum':name,**result,'four_class':fourclass})
        threshold=tau.get('FPR_0.01',{}).get('threshold')
        for branch in ['electron','muon','hadronic','unknown','ambiguous']:
            branch_frame=frame[frame.label.eq(2)&frame.tau_branch.eq(branch)]
            strata.append({'experiment':path.name,'stratum':'tau_'+branch,'events':len(branch_frame),
                'efficiency_at_inclusive_1pct_FPR':float((branch_frame.p2>=threshold).mean()) if len(branch_frame) and threshold is not None else None})
        config=manifest['config']
        if config.get('pitch_mm',0.) in [0.,1.] and config.get('reducer','count')=='count':
            identity={k:v for k,v in config.items() if k not in ['name','role','pitch_mm','projection']}
            key=json.dumps(identity,sort_keys=True)
            factorial.setdefault(key,{})[(config.get('pitch_mm',0.),config.get('projection',False))]=path.name
    comparisons=[]
    if args.comparisons:
        for pair in json.loads(args.comparisons.read_text())['pairs']:
            if pair['a'] not in frames or pair['b'] not in frames:
                raise ValueError(f'missing required completed experiment: {pair}')
            comparisons.append({**pair,**paired_interval(frames[pair['a']],frames[pair['b']],args.bootstrap),
                                'matched_allocated_time':matched_time(pair['a'],pair['b'],args.root,frames)})
    interactions=[]
    for identity,cells in factorial.items():
        order=[(0.,False),(0.,True),(1.,False),(1.,True)]
        if all(cell in cells for cell in order):
            names=[cells[cell] for cell in order]
            interactions.append({'config':json.loads(identity),'experiments_NJ_NP_CJ_CP':names,
                **factorial_contrast([frames[name] for name in names],args.bootstrap)})
    write_json(args.output/'summary.json',{'scope':'S2_validation_selection_no_calibration_or_test',
        'experiments':experiments,'strata':strata,'paired_comparisons':comparisons,
        'seed_spread':seed_spread(experiments),'factorial_interactions':interactions,
        'sample_mixture':'class_balanced_2500_per_class_before_conditional_strata',
        'source_acceptance':'user_attested_existing_frozen_split','physical_yields_measured':False})
    if not experiments:
        raise ValueError('no completed experiments')
    pd.concat(examples,ignore_index=True).to_parquet(args.output/'error_event_groups.parquet',index=False)
    table=[]
    for e in experiments:
        table.append({'experiment':e['name'],'tau_AP':e['tau']['PR_AUC'],'tau_AUC':e['tau']['ROC_AUC'],
            'balanced_accuracy':e['metrics']['balanced_accuracy'],'NLL':e['metrics']['NLL'],
            'tau_eff_at_1pct_FPR':e['tau']['FPR_0.01']['tau_efficiency'],
            'gpu_hours':e['gpu_hours'],'parameters':e['parameters'],'peak_gpu_mib':e['peak_gpu_mib']})
    pd.DataFrame(table).to_csv(args.output/'metrics.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
    for e in experiments:
        f=frames[e['name']]
        precision,recall,_=precision_recall_curve(f.label.eq(2),f.p2)
        ax.plot(recall,precision,label=f"{e['name']} ({e['tau']['PR_AUC']:.3f})")
    ax.set(xlabel='Tau recall',ylabel='Precision (sample priors)',title='S2 validation tau precision-recall',xlim=(0,1),ylim=(0,1))
    ax.legend(fontsize=6,loc='upper right')
    fig.savefig(args.output/'tau_PR.png',dpi=160)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--comparisons',type=Path)
    parser.add_argument('--bootstrap',type=int,default=400)
    run(parser.parse_args())


if __name__=='__main__':
    main()
