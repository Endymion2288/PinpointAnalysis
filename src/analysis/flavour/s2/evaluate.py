"""Paired S2 validation comparisons, conditions, finite tails and uncertainty."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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


def binary_comparison_metrics(y,score):
    fpr,tpr,_=roc_curve(y,score,drop_intermediate=False)
    return {'tau_AP':float(average_precision_score(y,score)),'tau_ROC_AUC':float(roc_auc_score(y,score)),
            **{f'tau_eff_at_{acceptance:g}_FPR':float(tpr[np.flatnonzero(fpr<=acceptance)[-1]])
               for acceptance in [.01,.001]}}


def resampling_groups(frame):
    """Use accepted family links to keep connected chunks in the same block.

    This consumes the existing family annotation; it neither certifies samples
    nor changes their membership. Mixed-class connected blocks stay together.
    """
    chunks=sorted(set(zip(frame.run,frame.chunk)))
    lookup={key:i for i,key in enumerate(chunks)}
    node=np.array([lookup[key] for key in zip(frame.run,frame.chunk)])
    parent=list(range(len(chunks)))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    family_used={'family_hi','family_lo'}.issubset(frame.columns)
    if family_used:
        for index in frame.groupby(['family_hi','family_lo'],sort=False).indices.values():
            unique=np.unique(node[index])
            for other in unique[1:]:
                a,b=root(int(unique[0])),root(int(other))
                parent[max(a,b)]=min(a,b)
    components={}
    for i,n in enumerate(node):
        components.setdefault(root(int(n)),[]).append(i)
    groups={}
    for component,index in sorted(components.items()):
        index=np.asarray(index,dtype='int64')
        stratum=tuple(int(value) for value in sorted(set(frame.label.to_numpy()[index])))
        groups.setdefault(stratum,[]).append(index)
    return groups,{'chunks':len(chunks),'resampling_blocks':len(components),
        'accepted_family_links_used':family_used,
        'multi_chunk_blocks':sum(len(set(node[ix]))>1 for ix in components.values()),
        'blocks_by_class_signature':{str(key):len(value) for key,value in groups.items()}}


def paired_interval(a,b,replicates=400,seed=2026,weight=None):
    """Stratified paired chunk bootstrap; seeds never become event replicas."""
    key=['run','chunk','row']
    joined=a[key+['label','p2']].merge(b[key+['label','p2']],on=key,validate='one_to_one',suffixes=('_a','_b'))
    if len(joined)!=len(a) or len(joined)!=len(b) or not joined.label_a.eq(joined.label_b).all():
        raise ValueError('paired comparison has different validation events')
    y=joined.label_a.to_numpy()==2
    pa,pb=joined.p2_a.to_numpy(),joined.p2_b.to_numpy()
    groups,block_summary=resampling_groups(a.reset_index(drop=True))
    rng=np.random.default_rng(seed)
    differences=[]
    detailed=[]
    weighted=[]
    for _ in range(replicates):
        index=np.concatenate([np.concatenate([g[i] for i in rng.integers(0,len(g),len(g))])
                              for g in groups.values()])
        ma,mb=[binary_comparison_metrics(y[index],p[index]) for p in [pa,pb]]
        differences.append(mb['tau_AP']-ma['tau_AP'])
        detailed.append({key:mb[key]-ma[key] for key in ma})
        if weight is not None:
            weighted.append(average_precision_score(y[index],pb[index],sample_weight=weight[index])-
                            average_precision_score(y[index],pa[index],sample_weight=weight[index]))
    low,high=np.quantile(differences,[.025,.975])
    delta=average_precision_score(y,pb)-average_precision_score(y,pa)
    measured_a,measured_b=[binary_comparison_metrics(y,p) for p in [pa,pb]]
    metrics={key:{'delta':measured_b[key]-measured_a[key],
                  'CI95':np.quantile([row[key] for row in detailed],[.025,.975]).tolist()} for key in measured_a}
    matched={}
    if weight is not None:
        matched={'delta':float(average_precision_score(y,pb,sample_weight=weight)-average_precision_score(y,pa,sample_weight=weight)),
                 'CI95':np.quantile(weighted,[.025,.975]).tolist(),
                 'scope':'conditional_on_fixed_validation_common_support_weights; no flux or exposure inference'}
    return {'metric':'tau_average_precision_B_minus_A','delta':float(delta),
        'CI95':[float(low),float(high)],'replicates':replicates,
        'resampling':'paired_class_stratified_accepted_family_connected_chunk_bootstrap','blocks':block_summary,
        'events':len(joined),'minimum_relevant_gain':.01,
        'can_exclude_gain_ge_0.01':bool(high<.01),'positive_gain_supported':bool(low>0),
        'equivalence_claim':False,'other_paired_metrics':metrics,'common_support_tau_AP':matched}


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
    groups,block_summary=resampling_groups(reference)
    rng=np.random.default_rng(seed)
    samples=[]
    for _ in range(replicates):
        index=np.concatenate([np.concatenate([g[i] for i in rng.integers(0,len(g),len(g))]) for g in groups.values()])
        samples.append(statistic(index))
    return {'interaction':statistic(np.arange(len(y))),'CI95':np.quantile(samples,[.025,.975]).tolist(),
            'definition':'AP_CP - AP_CJ - AP_NP + AP_NJ','replicates':replicates,
            'resampling':'paired_class_stratified_accepted_family_connected_chunk_bootstrap','blocks':block_summary}


def seed_mean_interval(pairs,replicates=400,seed=2026):
    """Bootstrap physical blocks once for every trained seed, not 3x events."""
    key=['run','chunk','row']
    reference=pairs[0][0].sort_values(key).reset_index(drop=True)
    y=reference.label.to_numpy()==2
    scores=[]
    for a,b in pairs:
        aligned=[]
        for frame in [a,b]:
            frame=frame.sort_values(key).reset_index(drop=True)
            if not frame[key+['label']].equals(reference[key+['label']]):
                raise ValueError('seed-mean contrast has different physical validation events')
            aligned.append(frame.p2.to_numpy())
        scores.append(aligned)
    weight,_=common_support_weights(reference)
    def deltas(index,weighted=False):
        options={'sample_weight':weight[index]} if weighted else {}
        return [average_precision_score(y[index],b[index],**options)-average_precision_score(y[index],a[index],**options)
                for a,b in scores]
    groups,block_summary=resampling_groups(reference)
    rng=np.random.default_rng(seed)
    intervals=[]
    weighted=[]
    for _ in range(replicates):
        index=np.concatenate([np.concatenate([g[i] for i in rng.integers(0,len(g),len(g))]) for g in groups.values()])
        intervals.append(float(np.mean(deltas(index))))
        if np.any(weight):
            weighted.append(float(np.mean(deltas(index,True))))
    observed=deltas(np.arange(len(y)))
    low,high=np.quantile(intervals,[.025,.975])
    return {'metric':'mean_seed_tau_AP_B_minus_A','delta':float(np.mean(observed)),
        'CI95':[float(low),float(high)],'per_seed_deltas':observed,
        'delta_seed_std':float(np.std(observed,ddof=1)) if len(observed)>1 else None,
        'events':len(reference),'trained_seed_pairs':len(pairs),'replicates':replicates,'blocks':block_summary,
        'scope':'event interval conditional on trained seeds; training spread separately reported; seeds are not event replicas',
        'minimum_relevant_gain':.01,'can_exclude_gain_ge_0.01':bool(high<.01),'positive_gain_supported':bool(low>0),
        'common_support_tau_AP':{'delta':float(np.mean(deltas(np.arange(len(y)),True))),
            'CI95':np.quantile(weighted,[.025,.975]).tolist(),
            'scope':'conditional on fixed common-support weights and trained seeds'} if weighted else {},
        'equivalence_claim':False}


def matched_time(a,b,root,frames):
    paths=[root/'experiments'/name for name in [a,b]]
    manifests=[json.loads((p/'manifest.json').read_text()) for p in paths]
    if any(m['gpu_name'] is None for m in manifests):
        return {'status':'CPU_reference_not_GPU_time_matched','comparison_measured':False}
    if manifests[0]['gpu_name']!=manifests[1]['gpu_name']:
        return {'status':'different_GPU_models','GPU_names':[m['gpu_name'] for m in manifests],
                'comparison_measured':False}
    for key in ['torch','mixed_precision']:
        if manifests[0].get(key)!=manifests[1].get(key):
            return {'status':'different_software_or_precision','comparison_measured':False}
    histories=[json.loads((p/'history.json').read_text()) for p in paths]
    accounting_file=root/'campaign_accounting.json'
    overheads=[0.,0.]
    if accounting_file.exists():
        allocations=json.loads(accounting_file.read_text())['experiments']
        records=[allocations.get(name) for name in [a,b]]
        if any(record is None or record.get('untracked_allocation_seconds') is None for record in records):
            return {'status':'allocation_history_incomplete','comparison_measured':False}
        overheads=[record['untracked_allocation_seconds'] for record in records]
    budget=min(h[-1]['wall_seconds']+extra for h,extra in zip(histories,overheads))
    selected=[]
    predictions=[]
    for name,path,history,extra in zip([a,b],paths,histories,overheads):
        eligible=[r for r in history if r['wall_seconds']+extra<=budget]
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
        selected.append({'experiment':name,'epoch':record['epoch'],'seconds':record['wall_seconds']+extra,
                         'untracked_allocation_seconds':extra,
                         'unused_seconds':budget-record['wall_seconds']-extra,'tau_AP':tau_metrics(frame)['PR_AUC']})
    return {'status':'measured_at_epoch_resolution','comparison_measured':True,'budget_seconds':budget,
            'GPU':manifests[0]['gpu_name'],'selection':'minimum_validation_NLL_within_allocated_wall_time',
            'allocation_overhead':'all setup/failed/resumed allocation absent from committed epoch history charged before first eligible checkpoint',
            'actual_HTCondor_allocation_used':accounting_file.exists(),
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


def plot_diagnostics(root,output,experiments,frames):
    labels=['CC nu_e','CC nu_mu','CC nu_tau','NC']
    with PdfPages(output/'classification_diagnostics.pdf') as pdf:
        for number,experiment in enumerate(experiments):
            name=experiment['name']
            frame=frames[name]
            fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
            matrix=np.asarray(experiment['metrics']['confusion_counts'])
            normalized=matrix/np.maximum(1,matrix.sum(1,keepdims=True))
            axes[0,0].imshow(normalized,vmin=0,vmax=1,cmap='Blues')
            for i in range(4):
                for j in range(4):
                    axes[0,0].text(j,i,f'{normalized[i,j]:.1%}\n({matrix[i,j]})',ha='center',va='center',fontsize=7)
            axes[0,0].set(xticks=range(4),xticklabels=labels,yticks=range(4),yticklabels=labels,
                          xlabel='Argmax prediction',ylabel='True class',title='Row-normalized confusion and counts')
            for c,label in enumerate(labels):
                truth=frame.label.eq(c)
                if truth.any() and (~truth).any():
                    fpr,tpr,_=roc_curve(truth,frame[f'p{c}'])
                    axes[0,1].plot(fpr,tpr,label=label)
                axes[1,0].hist(frame.loc[truth,'p2'],bins=np.linspace(0,1,41),histtype='step',label=label)
            axes[0,1].plot([0,1],[0,1],color='grey',linestyle=':')
            axes[0,1].set(xlabel='Background acceptance',ylabel='Class efficiency',title='One-vs-rest ROC')
            axes[0,1].legend(fontsize=7)
            axes[1,0].set(xlabel='Uncalibrated tau score',ylabel='Validation events',yscale='log',title='Tau scores by true class')
            axes[1,0].legend(fontsize=7)
            history=json.loads((root/'experiments'/name/'history.json').read_text())
            x=[row.get('epoch',row.get('iteration')) for row in history]
            nll=[row['validation']['NLL'] for row in history]
            ap=[row['validation']['classwise']['2']['PR_AUC'] for row in history]
            axes[1,1].plot(x,nll,color='tab:blue',marker='.',label='NLL')
            other=axes[1,1].twinx()
            other.plot(x,ap,color='tab:orange',marker='.',label='Tau AP')
            axes[1,1].set(xlabel='Epoch' if 'epoch' in history[0] else 'Boosting iteration',
                          ylabel='Validation NLL',title='Learning curves (checkpoint selection: NLL)')
            other.set_ylabel('Tau average precision',color='tab:orange')
            fig.suptitle(f'S2 validation: {name}; sample priors, no calibration or test',fontsize=10)
            pdf.savefig(fig)
            if number==0:
                fig.savefig(output/'first_classification_diagnostic.png',dpi=140)
            plt.close(fig)
    frame=next(iter(frames.values()))
    weight,support=common_support_weights(frame)
    if not np.any(weight):
        return
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for axis,column,scale,unit in zip(axes.flat,['initE','initX','initY','initZ'],[.001,1,1,1],['GeV','mm','mm','mm']):
        values=frame[column].to_numpy()*scale
        if column=='initE':
            positive=values[values>0]
            bins=np.geomspace(max(.1,positive.min()),positive.max(),45)
            axis.set_xscale('log')
        else:
            bins=np.linspace(values.min(),values.max(),45)
        for c,label in enumerate(labels):
            flag=frame.label.eq(c).to_numpy()
            color=f'C{c}'
            axis.hist(values[flag],bins=bins,density=True,histtype='step',color=color,label=label)
            axis.hist(values[flag],bins=bins,weights=weight[flag],density=True,histtype='step',color=color,linestyle='--')
        description={'initE':'True neutrino energy','initX':'True vertex x','initY':'True vertex y','initZ':'True vertex z'}[column]
        axis.set(xlabel=f'{description} [{unit}]',ylabel='Conditional density')
    axes[0,0].legend(fontsize=7)
    fig.suptitle(f'Truth prior diagnostic: solid unweighted, dashed common support; Neff={support["Neff"]:.0f}',fontsize=10)
    fig.savefig(output/'common_support_balance.png',dpi=160)
    fig.savefig(output/'common_support_balance.pdf')
    plt.close(fig)


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
                            'hardware_model':manifest.get('hardware_model',True),
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
    pending=[]
    seed_comparisons=[]
    if args.comparisons:
        comparison_config=json.loads(args.comparisons.read_text())
        for pair in comparison_config['pairs']:
            if pair['a'] not in frames or pair['b'] not in frames:
                if args.allow_incomplete:
                    pending.append(pair)
                    continue
                raise ValueError(f'missing required completed experiment: {pair}')
            weights,_=common_support_weights(frames[pair['a']])
            comparisons.append({**pair,**paired_interval(frames[pair['a']],frames[pair['b']],args.bootstrap,
                                weight=weights if np.any(weights) else None),
                                'matched_allocated_time':matched_time(pair['a'],pair['b'],args.root,frames)})
        for group in comparison_config.get('seed_mean_comparisons',[]):
            missing=sorted({name for pair in group['pairs'] for name in [pair['a'],pair['b']] if name not in frames})
            if missing:
                if args.allow_incomplete:
                    pending.append({'type':'seed_mean','name':group['name'],'missing_experiments':missing})
                    continue
                raise ValueError(f'missing completed experiments for seed mean: {missing}')
            seed_comparisons.append({**group,**seed_mean_interval([(frames[p['a']],frames[p['b']]) for p in group['pairs']],args.bootstrap)})
    interactions=[]
    for identity,cells in factorial.items():
        order=[(0.,False),(0.,True),(1.,False),(1.,True)]
        if all(cell in cells for cell in order):
            names=[cells[cell] for cell in order]
            interactions.append({'config':json.loads(identity),'experiments_NJ_NP_CJ_CP':names,
                **factorial_contrast([frames[name] for name in names],args.bootstrap)})
    write_json(args.output/'summary.json',{'scope':'S2_validation_selection_no_calibration_or_test',
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'python':sys.version,
        'experiments':experiments,'strata':strata,'paired_comparisons':comparisons,
        'seed_spread':seed_spread(experiments),'factorial_interactions':interactions,
        'pending_comparisons':pending,'requested_comparisons_complete':not pending,
        'seed_mean_comparisons':seed_comparisons,
        'sample_mixture':'class_balanced_2500_per_class_before_conditional_strata',
        'source_acceptance':'user_attested_existing_frozen_split','physical_yields_measured':False})
    if not experiments:
        raise ValueError('no completed experiments')
    pd.concat(examples,ignore_index=True).to_parquet(args.output/'error_event_groups.parquet',index=False)
    table=[]
    for e in experiments:
        table.append({'experiment':e['name'],'tau_AP':e['tau']['PR_AUC'],'tau_AUC':e['tau']['ROC_AUC'],
            'hardware_model':e['hardware_model'],
            'balanced_accuracy':e['metrics']['balanced_accuracy'],'NLL':e['metrics']['NLL'],
            'tau_eff_at_1pct_FPR':e['tau']['FPR_0.01']['tau_efficiency'],
            'gpu_hours':e['gpu_hours'],'parameters':e['parameters'],'peak_gpu_mib':e['peak_gpu_mib']})
    pd.DataFrame(table).to_csv(args.output/'metrics.csv',index=False)
    plot_diagnostics(args.root,args.output,experiments,frames)
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
    parser.add_argument('--allow-incomplete',action='store_true',help='Report available comparisons and explicitly list pending pairs')
    run(parser.parse_args())


if __name__=='__main__':
    main()
