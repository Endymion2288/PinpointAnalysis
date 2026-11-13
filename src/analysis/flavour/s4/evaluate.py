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


def evaluate(args):
    started=time.perf_counter();args.output.mkdir(parents=True,exist_ok=True)
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
    if not candidates:raise RuntimeError('no completed independent validation predictions')
    key=['run','chunk','row']
    reference=pd.read_parquet(candidates[0][3]).sort_values(key).reset_index(drop=True)
    common,support=common_support_weights(reference)
    weights,blocks=bootstrap_weights(reference,args.bootstrap)
    catalog=pd.read_parquet(args.root/'catalog.parquet')
    columns=key+['pixel_channels','bar_channels','tracker_channels','all_empty']
    occupancy=catalog[columns]
    experiments=[];strata=[];points={};samples={};frames={}
    for name,config,manifest,path in candidates:
        frame=align_frame(pd.read_parquet(path),reference)
        missing=[c for c in columns if c not in frame]
        if missing:frame=frame.merge(occupancy[key+missing],on=key,how='left',validate='one_to_one')
        frames[name]=frame
        cache=MetricCache(frame.label.to_numpy(),frame[[f'p{c}' for c in range(4)]].to_numpy())
        points[name]={'raw':cache.compute()[0],'common_support':cache.compute(common)[0]}
        samples[name]={domain:np.stack([cache.compute(bw*w)[0] for bw in weights])
            for domain,w in [('raw',np.ones(len(frame))),('common_support',common)]}
        tau=tau_metrics(frame)
        experiments.append({'name':name,'config':config,'program':config.get('program','train'),
            'four_class':cache.report(),'common_support_four_class':cache.report(common),
            'tau':tau,'prediction_diagnostics':prediction_diagnostics(cache.p),
            'training_seconds':manifest.get('seconds'),'GPU':manifest.get('gpu_name',manifest.get('GPU')),
            'peak_gpu_mib':manifest.get('peak_gpu_allocated_mib'),'code_commit':manifest.get('code_commit')})
        for domain,mask in domains(frame,config,True).items():
            local=MetricCache(frame.loc[mask,'label'].to_numpy(),frame.loc[mask,[f'p{c}' for c in range(4)]].to_numpy())
            strata.append({'name':name,'stratum':domain,'four_class':local.report(),'tau':tau_metrics(frame.loc[mask])})
        for branch in ['electron','muon','hadronic','unknown','ambiguous']:
            strata.append({'name':name,'stratum':'tau_'+branch,
                'tau':tau_branch_metrics(frame,branch,tau.get('FPR_0.01',{}).get('threshold'))})
        print(name,'paired-validation statistics complete',flush=True)
    specifications=json.loads(args.comparisons.read_text()) if args.comparisons else {'pairs':[]}
    comparisons=[]
    for spec in specifications.get('pairs',[]):
        a,b=spec['a'],spec['b']
        if a not in points or b not in points:
            pending.append({'pair':spec});continue
        contrasts={domain:interval_report(points[b][domain]-points[a][domain],samples[b][domain]-samples[a][domain])
            for domain in ['raw','common_support']}
        comparisons.append({**spec,'metrics':contrasts})
    means=[]
    for spec in specifications.get('seed_mean_comparisons',[]):
        pairs=spec['pairs']
        if any(p[k] not in points for p in pairs for k in ['a','b']):pending.append({'seed_mean':spec});continue
        result={}
        for domain in ['raw','common_support']:
            delta=np.stack([points[p['b']][domain]-points[p['a']][domain] for p in pairs])
            bootstrap=np.stack([samples[p['b']][domain]-samples[p['a']][domain] for p in pairs]).mean(0)
            result[domain]=interval_report(delta.mean(0),bootstrap,delta)
        means.append({**spec,'metrics':result})
    summary={'utc':datetime.now(timezone.utc).isoformat(),'scope':'S4 independent complete 10k validation; calibration/test frozen',
        'experiments':experiments,'strata':strata,'paired_comparisons':comparisons,'seed_mean_comparisons':means,
        'common_support':support,'resampling_blocks':blocks,'bootstrap_replicates':args.bootstrap,
        'pending':pending,'requested_complete':not pending,'seconds':time.perf_counter()-started,
        'finite_MC_scope':'1%/0.1% FPR efficiency and tau branch counts reported; low selected backgrounds retain finite-MC intervals',
        'resource_scope':'per-task elapsed seconds are not allocation sums when sharing GPU; see campaign_accounting and matched_compute separately'}
    global_probes=[]
    for name,config,manifest,path in candidates:
        if config.get('program')!='probes':continue
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
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    groups={}
    for e in experiments:
        c=e['config'];identity=(c.get('initialization','scratch'),e['program'])
        groups.setdefault(identity,[]).append(e)
    curve_rows=[]
    for (initialization,program),group in groups.items():
        budgets=sorted({e['config'].get('label_budget',100000) for e in group})
        for ax,metric in zip(axes.flat,['macro_AP','NLL','class_2_AP','class_2_eff_at_0.01_FPR']):
            values=[[e['four_class'][metric] for e in group if e['config'].get('label_budget',100000)==budget]
                for budget in budgets]
            mean=[float(np.mean(v)) for v in values]
            spread=[float(np.std(v,ddof=1)) if len(v)>1 else 0. for v in values]
            ax.errorbar(budgets,mean,yerr=spread,marker='.',capsize=2,label=initialization+' / '+program)
            curve_rows.extend({'initialization':initialization,'program':program,'label_budget':budget,
                'metric':metric,'mean':m,'training_seed_SD':s,'seeds':len(v)}
                for budget,m,s,v in zip(budgets,mean,spread,values))
            ax.set(xscale='log',xlabel='Unique labeled training events',ylabel=metric)
    axes[0,0].legend(fontsize=7)
    fig.suptitle('S4 label efficiency: fixed validation; bars show training-seed SD')
    fig.savefig(output/'label_efficiency.pdf');fig.savefig(output/'label_efficiency.png',dpi=160);plt.close(fig)
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
