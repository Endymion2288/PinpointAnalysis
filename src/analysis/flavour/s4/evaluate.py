"""S4 same-event four-class, tau and matched-resource validation comparisons."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from analysis.flavour.inventory import write_json
from analysis.flavour.s2.evaluate import common_support_weights,tau_metrics,tau_branch_metrics
from analysis.flavour.s3.evaluate import domains,align_frame
from analysis.flavour.s3.metrics import MetricCache,NAMES,bootstrap_weights,interval_report,prediction_diagnostics


TAU_NAMES=('AP','AUROC','eff_at_0.01_FPR','eff_at_0.001_FPR')


def tau_statistics(cache,weight):
    """Tie-aware tau statistics after excluding the other tau decay branches."""
    order,ends,truth=cache.orders[2]
    w=np.asarray(weight,dtype=float)[order]
    tp=np.cumsum(w*truth)[ends];fp=np.cumsum(w*~truth)[ends]
    positive=float(np.sum(w*truth));negative=float(np.sum(w*~truth))
    if not positive or not negative:return np.full(len(TAU_NAMES),np.nan)
    precision=np.divide(tp,tp+fp,out=np.zeros_like(tp),where=tp+fp>0)
    ap=float(np.dot(np.diff(np.r_[0.,tp]),precision)/positive)
    auc=float(np.dot(tp+np.r_[0.,tp[:-1]],np.diff(np.r_[0.,fp]))/(2*positive*negative))
    return np.array([ap,auc,*[float(np.max(tp[fp<=a*negative],initial=0.)/positive) for a in [.01,.001]]])


def tau_interval(observed,samples,per_seed=None):
    result={}
    for i,name in enumerate(TAU_NAMES):
        finite=np.isfinite(samples[:,i]);value=observed[i]
        result[name]={'delta':float(value) if np.isfinite(value) else None,
            'CI95':np.quantile(samples[finite,i],[.025,.975]).tolist() if finite.any() else [None,None],
            'valid_replicates':int(finite.sum())}
        if per_seed is not None:
            result[name].update(per_seed_deltas=[float(v) if np.isfinite(v) else None for v in per_seed[:,i]],
                training_seed_sample_SD=float(np.std(per_seed[:,i],ddof=1)) if len(per_seed)>1 else None)
    return result


def classwise_operating_points(frame):
    """Retain actual selected event counts for all four empirical ROC tails."""
    from scipy.stats import beta
    from analysis.flavour.s3.metrics import CLASSES
    result={}
    for c,name in enumerate(CLASSES):
        binary=pd.DataFrame({'label':np.where(frame.label.to_numpy()==c,2,0),'p2':frame[f'p{c}'].to_numpy()})
        measured=tau_metrics(binary)
        points={}
        for acceptance in [.01,.001]:
            key=f'FPR_{acceptance:g}'
            if key not in measured:continue
            point=dict(measured[key]);ns=int(frame.label.eq(c).sum());nb=len(frame)-ns
            for kind,k,n in [('signal',point['signal_selected'],ns),('background',point['background_selected'],nb)]:
                point[kind+'_count_interval_95']=[float(beta.ppf(.025,k,n-k+1)) if k else 0.,
                    float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]
            point.update(signal_events=ns,background_events=nb,
                threshold_scope='S4 validation empirical ROC; final calibrated operating point not selected',
                count_interval_scope='descriptive Clopper-Pearson counts at validation-selected thresholds; no independent operating-point coverage claim')
            points[key]=point
        result[name]=points
    return result


def evaluate(args):
    started=time.perf_counter();args.output.mkdir(parents=True,exist_ok=True)
    specifications=json.loads(args.comparisons.read_text()) if args.comparisons else {'pairs':[]}
    configs={c['name']:c for path in args.campaign for c in json.loads(path.read_text())['experiments']}
    candidates=[];pending=[]
    for name,config in configs.items():
        program=config.get('program','train' if config.get('objective')=='classification' else 'pretrain')
        if program not in {'train','probes'} or config.get('probe_scope')=='train_only_branches':continue
        directory=args.root/'experiments'/name
        manifest=directory/'manifest.json'
        predictions=directory/('full_linear_predictions.parquet' if program=='probes' else 'validation_predictions.parquet')
        if not manifest.exists() or json.loads(manifest.read_text()).get('status')!='complete' or not predictions.exists():
            pending.append({'name':name,'program':program});continue
        candidates.append((name,config,json.loads(manifest.read_text()),predictions))
        if program=='train' and config.get('supervised_initial_checkpoint') and specifications.get('include_supervised_selection_envelope',False):
            parent=Path(config['supervised_initial_checkpoint']).parent
            parent_manifest=json.loads((parent/'manifest.json').read_text())
            continuation_manifest=json.loads(manifest.read_text())
            use_parent=parent_manifest['best_NLL']<=continuation_manifest['best_NLL']
            selected_directory=parent if use_parent else directory
            selected_manifest=parent_manifest if use_parent else continuation_manifest
            envelope_config={**config,'initialization':config.get('initialization','scratch')+'_selection',
                'selection_envelope':{'parent':parent.name,'continuation':name,'selected':selected_directory.name,
                    'selection':'lowest validation NLL over original Scratch and matched continuation checkpoints',
                    'compute_scope':'charge full parent plus full continuation, regardless of selected checkpoint'}}
            candidates.append((name+'__selection',envelope_config,selected_manifest,
                selected_directory/'validation_predictions.parquet'))
        if program=='probes' and specifications.get('include_global_frozen',False):
            global_path=directory/'global_linear_predictions.parquet'
            if not global_path.exists():pending.append({'name':name+'__global','program':'probes'})
            else:candidates.append((name+'__global',{**config,'probe_representation':'global'},
                json.loads(manifest.read_text()),global_path))
    if not candidates:raise RuntimeError('no completed independent validation predictions')
    key=['run','chunk','row']
    reference=pd.read_parquet(candidates[0][3]).sort_values(key).reset_index(drop=True)
    common,support=common_support_weights(reference)
    weights,blocks=bootstrap_weights(reference,args.bootstrap)
    catalog=pd.read_parquet(args.root/'catalog.parquet')
    columns=key+['pixel_channels','bar_channels','tracker_channels','all_empty']
    occupancy=catalog[columns]
    experiments=[];strata=[];points={};samples={};frames={};tau_points={};tau_samples={}
    tau_domains={}
    if specifications.get('paired_tau_branches',False):
        for branch in ['electron','muon','hadronic']:
            mask=reference.label.ne(2).to_numpy()|reference.tau_branch.eq(branch).to_numpy()
            for domain,w in [('raw',np.ones(len(reference))),('common_support',common)]:
                tau_domains[branch+'__'+domain]=mask*w
    for name,config,manifest,path in candidates:
        frame=align_frame(pd.read_parquet(path),reference)
        if not frame.tau_branch.equals(reference.tau_branch):raise ValueError('different accepted tau branch metadata')
        missing=[c for c in columns if c not in frame]
        if missing:frame=frame.merge(occupancy[key+missing],on=key,how='left',validate='one_to_one')
        frames[name]=frame
        cache=MetricCache(frame.label.to_numpy(),frame[[f'p{c}' for c in range(4)]].to_numpy())
        points[name]={'raw':cache.compute()[0],'common_support':cache.compute(common)[0]}
        samples[name]={domain:np.stack([cache.compute(bw*w)[0] for bw in weights])
            for domain,w in [('raw',np.ones(len(frame))),('common_support',common)]}
        tau_points[name]={domain:tau_statistics(cache,w) for domain,w in tau_domains.items()}
        tau_samples[name]={domain:np.stack([tau_statistics(cache,bw*w) for bw in weights])
            for domain,w in tau_domains.items()}
        tau=tau_metrics(frame)
        experiments.append({'name':name,'config':config,'program':config.get('program','train'),
            'four_class':cache.report(),'common_support_four_class':cache.report(common),
            'tau':tau,'prediction_diagnostics':prediction_diagnostics(cache.p),
            'classwise_operating_points':classwise_operating_points(frame),
            'training_seconds':manifest.get('seconds'),'GPU':manifest.get('gpu_name',manifest.get('GPU')),
            'peak_gpu_mib':manifest.get('peak_gpu_allocated_mib'),'code_commit':manifest.get('code_commit')})
        for domain,mask in domains(frame,config,True).items():
            local=MetricCache(frame.loc[mask,'label'].to_numpy(),frame.loc[mask,[f'p{c}' for c in range(4)]].to_numpy())
            strata.append({'name':name,'stratum':domain,'four_class':local.report(),'tau':tau_metrics(frame.loc[mask])})
        for branch in ['electron','muon','hadronic','unknown','ambiguous']:
            strata.append({'name':name,'stratum':'tau_'+branch,
                'tau':tau_branch_metrics(frame,branch,tau.get('FPR_0.01',{}).get('threshold'))})
        print(name,'paired-validation statistics complete',flush=True)
    comparisons=[]
    for spec in specifications.get('pairs',[]):
        a,b=spec['a'],spec['b']
        if a not in points or b not in points:
            pending.append({'pair':spec});continue
        contrasts={domain:interval_report(points[b][domain]-points[a][domain],samples[b][domain]-samples[a][domain])
            for domain in ['raw','common_support']}
        branch_contrasts={domain:tau_interval(tau_points[b][domain]-tau_points[a][domain],
            tau_samples[b][domain]-tau_samples[a][domain]) for domain in tau_domains}
        comparisons.append({**spec,'metrics':contrasts,'tau_branches':branch_contrasts})
    means=[]
    for spec in specifications.get('seed_mean_comparisons',[]):
        pairs=spec['pairs']
        if any(p[k] not in points for p in pairs for k in ['a','b']):pending.append({'seed_mean':spec});continue
        result={}
        for domain in ['raw','common_support']:
            delta=np.stack([points[p['b']][domain]-points[p['a']][domain] for p in pairs])
            bootstrap=np.stack([samples[p['b']][domain]-samples[p['a']][domain] for p in pairs]).mean(0)
            result[domain]=interval_report(delta.mean(0),bootstrap,delta)
        branch_result={}
        for domain in tau_domains:
            delta=np.stack([tau_points[p['b']][domain]-tau_points[p['a']][domain] for p in pairs])
            bootstrap=np.stack([tau_samples[p['b']][domain]-tau_samples[p['a']][domain] for p in pairs]).mean(0)
            branch_result[domain]=tau_interval(delta.mean(0),bootstrap,delta)
        means.append({**spec,'metrics':result,'tau_branches':branch_result})
    summary={'utc':datetime.now(timezone.utc).isoformat(),'scope':'S4 independent complete 10k validation; calibration/test frozen',
        'experiments':experiments,'strata':strata,'paired_comparisons':comparisons,'seed_mean_comparisons':means,
        'common_support':support,'resampling_blocks':blocks,'bootstrap_replicates':args.bootstrap,
        'pending':pending,'requested_complete':not pending,'seconds':time.perf_counter()-started,
        'finite_MC_scope':'1%/0.1% FPR efficiency and tau branch counts reported; low selected backgrounds retain finite-MC intervals',
        'resource_scope':'per-task elapsed seconds are not allocation sums when sharing GPU; see campaign_accounting and matched_compute separately'}
    summary['interval_scope']='pointwise paired connected-chunk intervals conditional on the trained seeds and fixed common-support weights; seed SD reported separately'
    summary['tau_branch_comparison_scope']='only the stated tau branch versus all non-tau backgrounds; raw and common-support weights use identical physical block multiplicities'
    summary['tau_branch_points']={name:{domain:dict(zip(TAU_NAMES,[float(v) if np.isfinite(v) else None for v in values]))
        for domain,values in item.items()} for name,item in tau_points.items()}
    global_probes=[]
    for name,config,manifest,path in candidates:
        if config.get('program')!='probes' or config.get('probe_representation')=='global':continue
        global_path=path.with_name('global_linear_predictions.parquet')
        if not global_path.exists():continue
        frame=align_frame(pd.read_parquet(global_path),reference)
        cache=MetricCache(frame.label.to_numpy(),frame[[f'p{c}' for c in range(4)]].to_numpy())
        global_probes.append({'name':name,'label_budget':config['label_budget'],
            'initialization':config.get('initialization','scratch'),
            'four_class':cache.report(),'common_support_four_class':cache.report(common),
            'scope':'global-only frozen linear probe; same fitted-label subset and fixed validation'})
    summary['global_only_linear_probes']=global_probes
    write_json(args.output/'summary.json',summary)
    np.savez_compressed(args.output/'bootstrap_statistics.npz',metric_names=np.array(NAMES),
        **{f'{name}__{domain}':x for name,value in samples.items() for domain,x in value.items()})
    plot_curves(args.output,experiments,frames)
    if args.strict and pending:raise RuntimeError('required S4 validation comparisons pending')


def plot_curves(output,experiments,frames):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from sklearn.metrics import roc_curve,precision_recall_curve
    from analysis.flavour.s3.metrics import CLASSES
    groups={}
    for e in experiments:
        c=e['config'];representation=c.get('probe_representation','full')
        identity=(c.get('initialization','scratch'),e['program'],representation)
        groups.setdefault(identity,[]).append(e)
    curve_rows=[]
    sections=list(dict.fromkeys((key[1],key[2]) for key in groups))
    colors={name:f'C{i}' for i,name in enumerate(dict.fromkeys(key[0] for key in groups))}
    labels={'macro_AP':'Macro average precision','NLL':'Negative log-likelihood',
        'class_2_AP':'CC nu_tau average precision','class_2_eff_at_0.01_FPR':'CC nu_tau efficiency at 1% FPR'}
    with PdfPages(output/'label_efficiency.pdf') as pdf:
        for program,representation in sections:
            fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
            for (initialization,kind,route),group in groups.items():
                if (kind,route)!=(program,representation):continue
                budgets=sorted({e['config'].get('label_budget',100000) for e in group})
                for ax,metric in zip(axes.flat,['macro_AP','NLL','class_2_AP','class_2_eff_at_0.01_FPR']):
                    values=[[e['four_class'][metric] for e in group if e['config'].get('label_budget',100000)==budget]
                        for budget in budgets]
                    mean=[float(np.mean(v)) for v in values]
                    spread=[float(np.std(v,ddof=1)) if len(v)>1 else np.nan for v in values]
                    ax.plot(budgets,mean,marker='.',label=initialization,color=colors[initialization])
                    finite=np.isfinite(spread)
                    if finite.any():ax.errorbar(np.asarray(budgets)[finite],np.asarray(mean)[finite],
                        yerr=np.asarray(spread)[finite],fmt='none',capsize=2,color=colors[initialization])
                    curve_rows.extend({'initialization':initialization,'program':program,'label_budget':budget,
                        'representation':representation,'metric':metric,'mean':m,'training_seed_SD':s,'seeds':len(v)}
                        for budget,m,s,v in zip(budgets,mean,spread,values))
                    ax.set(xscale='log',xlabel='Unique labeled training events',ylabel=labels[metric])
            axes[0,0].legend(fontsize=7)
            fig.suptitle(f'S4 {program} / {representation}: fixed validation; seed SD shown for n >= 2')
            pdf.savefig(fig);fig.savefig(output/f'label_efficiency_{program}_{representation}.png',dpi=160)
            if program=='train' or len(sections)==1:fig.savefig(output/'label_efficiency.png',dpi=160)
            plt.close(fig)
    pd.DataFrame(curve_rows).to_csv(output/'label_efficiency.csv',index=False)
    with PdfPages(output/'four_class_diagnostics.pdf') as pdf:
        for e in experiments:
            name=e['name'];frame=frames[name]
            fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
            matrix=np.asarray(e['four_class']['confusion_weighted_counts']);normalized=matrix/np.maximum(1,matrix.sum(1,keepdims=True))
            axes[0].imshow(normalized,vmin=0,vmax=1,cmap='Blues')
            for i in range(4):
                for j in range(4):axes[0].text(j,i,f'{normalized[i,j]:.1%}\n({int(matrix[i,j])})',ha='center',va='center',fontsize=7)
            axes[0].set(xticks=range(4),yticks=range(4),xticklabels=CLASSES,yticklabels=CLASSES,xlabel='Predicted',ylabel='True')
            for c,label in enumerate(CLASSES):
                fpr,tpr,_=roc_curve(frame.label.eq(c),frame[f'p{c}']);precision,recall,_=precision_recall_curve(frame.label.eq(c),frame[f'p{c}'])
                axes[1].plot(fpr,tpr,label=label);axes[2].step(recall,precision,where='post',label=label)
            axes[1].set(xscale='log',xlim=(1e-4,1),xlabel='Background acceptance',ylabel='Class efficiency',title='One-vs-rest ROC')
            axes[2].set(xlabel='Recall',ylabel='Precision',title='One-vs-rest precision-recall');axes[1].legend(fontsize=7)
            fig.suptitle(name+': fixed validation; uncalibrated probabilities',fontsize=10);pdf.savefig(fig);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--campaign',type=Path,nargs='+',required=True)
    parser.add_argument('--comparisons',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bootstrap',type=int,default=400);parser.add_argument('--strict',action='store_true')
    evaluate(parser.parse_args())


if __name__=='__main__':main()
