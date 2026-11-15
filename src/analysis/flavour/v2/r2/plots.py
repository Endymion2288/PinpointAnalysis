"""English-only scaling learning curves and paired development contrasts."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS={'D0':'#5f7f95','D1':'#d17b0f','D2':'#288b69','D3':'#9c4f97','D4':'#b74848'}


def save(fig,folder,name):
    fig.tight_layout()
    for suffix in ['pdf','png']:fig.savefig(folder/f'{name}.{suffix}',dpi=160,bbox_inches='tight')
    plt.close(fig)


def readout_figures(root, partial):
    path=root/('readout_partial' if partial else 'readout')/'results.json'
    if not path.exists():return
    result=json.loads(path.read_text())
    names=list(result['models'])
    x=np.arange(len(names))
    fig,axes=plt.subplots(2,3,figsize=(14,7.2))
    variants=[('raw','Raw'),('positive_scalar_temperature','OOF temperature'),
              ('two_global_class_biases','OOF global bias')]
    panels=[('all','BA',100.,'Common BA3 (%)'),('all','NLL',1.,'Common NLL3'),
        ('all','mu_recall',100.,'Common muon recall (%)'),
        ('occupancy_0_100','BA',100.,'Sparse BA3 (%)'),('occupancy_0_100','NLL',1.,'Sparse NLL3'),
        ('occupancy_0_100','mu_recall',100.,'Sparse muon recall (%)')]
    for ax,(domain,metric,scale,title) in zip(axes.flat,panels):
        for i,(variant,label) in enumerate(variants):
            values=[]
            for name in names:
                row=result['models'][name]['domains'][domain][variant]
                values.append((row['per_class'][1]['recall'] if metric=='mu_recall' else row[metric])*scale)
            ax.bar(x+(i-1)*.25,values,width=.25,label=label)
        ax.set_xticks(x,[name.replace('b1_','').replace('_s42','').upper() for name in names],rotation=40)
        ax.set(title=title)
        ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Five grouped folds on Tune; development diagnostic; sparse = occupancy < 100',fontsize=12)
    save(fig,root/'figures','r2_readout_diagnostic_partial' if partial else 'r2_readout_diagnostic')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,metric,scale in [(axes[0],'BA',100.),(axes[1],'NLL',1.)]:
        train,tune=[],[]
        for name in names:
            gap=result['models'][name]['convergence']['Train_Tune_gaps'].get('endpoint',{}).get('T3',{})
            train.append(gap.get(metric,{}).get('Train',np.nan)*scale)
            tune.append(gap.get(metric,{}).get('Tune',np.nan)*scale)
        ax.plot(x,train,'o-',label='Common Train sample (eval mode)')
        ax.plot(x,tune,'s--',label='Complete Tune')
        ax.set_xticks(x,[name.replace('b1_','').replace('_s42','').upper() for name in names],rotation=40)
        ax.set(title='Common three-class '+metric,ylabel='Percent' if metric=='BA' else 'NLL')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Fixed-budget endpoints; Train sample has 1,000 events per class',fontsize=12)
    save(fig,root/'figures','r2_train_tune_gap_partial' if partial else 'r2_train_tune_gap')


def run(root,partial):
    folder=root/'figures'
    folder.mkdir(exist_ok=True)
    campaign=json.loads((root/'campaign.json').read_text())
    records=[]
    for name in campaign['initial_models']+campaign.get('extension_models',[]):
        path=root/'experiments'/name/'manifest.json'
        if path.exists():records.append(json.loads(path.read_text()))
    fig,axes=plt.subplots(2,3,figsize=(14,7.4))
    for record in records:
        config=record['config']
        task,design=config['task'],config['design']
        history=record['history']
        if not history:continue
        x=np.array([h['non_tau_exposures'] for h in history])/1e6
        color=COLORS.get(design,'#555555')
        style='-' if task=='T3' else '--'
        label=f'{design} {task}'
        axes[0,0 if task=='T3' else 1].plot(x,[h['metrics'][task]['NLL'] for h in history],style,
            color=color,label=label,marker='.',markersize=4)
        axes[0,2].plot(x,[100*h['metrics']['T3']['BA'] for h in history],style,color=color,label=label)
        axes[1,0].plot(x,[100*h['metrics']['T3']['per_class'][1]['recall'] for h in history],style,color=color,label=label)
        low=[h.get('occupancy_below_100',{}).get('metrics') for h in history]
        axes[1,1].plot(x,[100*h['T3']['per_class'][1]['recall'] if h else np.nan for h in low],style,
            color=color,label=label)
        if task=='T4':
            axes[1,2].plot(x,[100*h['metrics']['T4']['per_class'][2].get('eff_FPR_0.01',np.nan) for h in history],
                style,color=color,label=label)
    titles=['Three-class NLL','Four-class NLL','Common three-class balanced accuracy',
        'Common muon recall','Muon recall: occupancy < 100','Tau efficiency at 1% FPR']
    for ax,title in zip(axes.flat,titles):
        ax.set(title=title,xlabel='Non-tau training exposures (million)',ylabel='NLL' if 'NLL' in title else 'Percent')
        ax.grid(alpha=.22)
        handles,labels=ax.get_legend_handles_labels()
        if handles:ax.legend(fontsize=8)
    fig.suptitle('Fixed 4k development screen; raw scores; matched seed 42',fontsize=12)
    save(fig,folder,'r2_learning_curves_partial' if partial else 'r2_learning_curves')
    evaluation=root/('evaluation_partial' if partial else 'evaluation')
    if not (evaluation/'results.json').exists():return
    result=json.loads((evaluation/'results.json').read_text())
    table=pd.read_csv(evaluation/'model_domain_metrics.csv')
    primary=table[table.kind.eq('endpoint')&table.domain.eq('all')].sort_values(['task','design'])
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    labels=[f'{r.design} {r.task}' for r in primary.itertuples()]
    axes[0].bar(labels,primary.BA3*100,color=[COLORS.get(d,'#555555') for d in primary.design])
    axes[0].set(title='Fixed-budget endpoints',ylabel='Common BA3 (%)')
    axes[1].bar(labels,primary.NLL3,color=[COLORS.get(d,'#555555') for d in primary.design])
    axes[1].set(title='Fixed-budget endpoints',ylabel='Common NLL3')
    pairs=[p for p in result['pairs'] if p['domain']=='all' and p['a'].endswith('_endpoint')
        and p['reason'] in ['same_unique_extended_budget','same_exposures_more_unique','D1WD_minus_D1']]
    for i,pair in enumerate(pairs):
        value=pair['metrics']['BA3']
        point=value['difference_a_minus_b']*100
        low,high=np.array(value['95pct'])*100
        axes[2].hlines(i,low,high,color='#315b75')
        axes[2].plot(point,i,'o',color='#315b75')
    axes[2].set_yticks(range(len(pairs)),[p['a'].removesuffix('_endpoint').replace('b1_','')+' - '+
        p['b'].removesuffix('_endpoint').replace('b1_','') for p in pairs],fontsize=8)
    axes[2].axvline(0,color='gray',linewidth=.8)
    axes[2].set(title='Paired physical-block 95% intervals',xlabel='Difference in BA3 (percentage points)')
    for ax in axes[:2]:ax.tick_params(axis='x',labelrotation=45)
    save(fig,folder,'r2_scaling_endpoints_partial' if partial else 'r2_scaling_endpoints')
    low=table[table.kind.eq('endpoint')&table.domain.eq('occupancy_0_100')].sort_values(['task','design'])
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for task,marker in [('T3','o'),('T4','s')]:
        rows=low[low.task.eq(task)]
        axes[0,0].plot(rows.design,100*rows['T3_recall_1'],marker=marker,label=task)
        axes[0,1].plot(rows.design,100*rows['BA3'],marker=marker,label=task)
        for ax,metric in [(axes[1,0],'AP'),(axes[1,1],'AUROC')]:
            values=[result['models'][name]['occupancy_0_100']['T3']['per_class'][1][metric] for name in rows.name]
            ax.plot(rows.design,values,marker=marker,label=task)
    axes[0,0].set(title='Low occupancy: muon argmax recall',ylabel='Recall (%)',xlabel='Scaling design')
    axes[0,1].set(title='Low occupancy: common BA3',ylabel='Balanced accuracy (%)',xlabel='Scaling design')
    axes[1,0].set(title='Low occupancy: muon average precision',ylabel='Balanced-prior AP',xlabel='Scaling design')
    axes[1,1].set(title='Low occupancy: muon ranking',ylabel='Balanced-prior AUROC',xlabel='Scaling design')
    axes[1,0].axhline(1/3,color='gray',linestyle=':',label='Chance AP (1/3)')
    axes[1,1].axhline(.5,color='gray',linestyle=':',label='Chance AUROC (0.5)')
    for ax in axes.flat:ax.grid(alpha=.22);ax.legend()
    save(fig,folder,'r2_low_occupancy_partial' if partial else 'r2_low_occupancy')
    readout_figures(root,partial)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--partial',action='store_true')
    args=parser.parse_args()
    run(args.root,args.partial)


if __name__=='__main__':main()
