"""Standalone English S4 pretraining and label-efficiency diagnostics."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def pilot_plots(root,campaign,output):
    configs=json.loads(campaign.read_text())['experiments']
    rows=[]
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for config in configs:
        directory=root/'experiments'/config['name']
        if not (directory/'reconstruction.json').exists():continue
        measured=json.loads((directory/'reconstruction.json').read_text())
        health=json.loads((directory/'training_health.json').read_text())['records']
        label=config['name'].replace('s4p2_','').replace('_s42','')
        steps=[r['step'] for r in health]
        axes[0,0].plot(steps,[r['global_features']['between_event_RMS'] for r in health],label=label)
        axes[0,1].plot(steps,[r['global_features']['centered_effective_rank'] for r in health],label=label)
        for level in range(3):
            q=measured['queries'][f'level{level}_all']
            rows.append({'experiment':config['name'],'label':label,'level':level,'positive_queries':q['positives'],
                'queries':q['queries'],'AP':q['AP'],'prior':q['prevalence'],
                'AP_over_prior':q['AP']/q['prevalence'] if q['AP'] is not None and q['prevalence'] else np.nan,
                'permutation_delta_BCE':measured['event_permutation_delta_BCE'][str(level)]})
    table=pd.DataFrame(rows)
    names=list(dict.fromkeys(table.label))
    for level,title in [(0,'Fine pixel patches'),(1,'4-mm pixel cells'),(2,'Scintillator bars')]:
        selected=table[table.level.eq(level)].set_index('label').reindex(names)
        x=np.arange(len(names))+(level-1)*.25
        axes[1,0].bar(x,selected.AP_over_prior,width=.24,label=title)
        axes[1,1].bar(x,selected.permutation_delta_BCE,width=.24,label=title)
    axes[0,0].set(xlabel='Optimizer step',ylabel='Between-event feature RMS',title='Global event variation')
    axes[0,1].set(xlabel='Optimizer step',ylabel='Centered effective rank',title='Global representation spectrum')
    axes[1,0].set(ylabel='Average precision / query prior',title='Inner train reconstruction probe')
    axes[1,1].set(ylabel='Shuffled minus conditioned BCE',title='Dependence on event content')
    for ax in axes[1]:ax.set_xticks(range(len(names)),names,rotation=25,ha='right',fontsize=7)
    axes[0,0].legend(fontsize=7);axes[1,0].legend(fontsize=7)
    fig.suptitle('S4 reconstruction diagnostics: original train partition only')
    output.mkdir(parents=True,exist_ok=True)
    fig.savefig(output/'pretraining_diagnostics.pdf');fig.savefig(output/'pretraining_diagnostics.png',dpi=160);plt.close(fig)
    table.to_csv(output/'pretraining_measured.csv',index=False)


def functional_plots(report_path, output):
    """Plot fixed-weight interventions using their own FP32 baseline."""
    report = json.loads(report_path.read_text())
    experiments = report['experiments']
    branches = ['global_latent', 'all_regional', 'all_local', 'response_summary', 'all_spatial']
    names = ['Global latents', 'Regional branches', 'Local branches', 'Response summary', 'All spatial branches']
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for metric, label in [('NLL', 'Change in negative log-likelihood'), ('macro_AP', 'Change in macro average precision')]:
        fig, axes = plt.subplots((len(experiments) + 1)//2, 2, figsize=(12, 3.5*((len(experiments)+1)//2)),
            constrained_layout=True, squeeze=False)
        for ax, item in zip(axes.flat, experiments):
            settings = item['checkpoint_config']
            for domain, offset, color in [('raw', -.12, 'C0'), ('common_support', .12, 'C1')]:
                for i, branch in enumerate(branches):
                    point = item['branch_permutation'][branch][domain][metric]
                    lo, hi = point['CI95']
                    ax.hlines(i + offset, lo, hi, color=color, linewidth=1.5)
                    ax.plot(point['delta'], i + offset, '.', color=color,
                        label=domain.replace('_', ' ').title() if i == 0 else None)
                    records.append({'experiment': item['experiment'], 'branch': branch, 'domain': domain,
                        'metric': metric, 'delta': point['delta'], 'CI95_low': lo, 'CI95_high': hi})
            ax.axvline(0, color='0.5', linewidth=.8)
            ax.set(yticks=range(len(names)), yticklabels=names, xlabel=label,
                title=f"{settings.get('initialization', 'scratch')} / {settings['label_budget']:,} labels")
            ax.invert_yaxis()
        for ax in axes.flat[len(experiments):]:
            ax.set_visible(False)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle('Fixed-checkpoint path reliance: complete 10k validation\n'
            f"Changed minus original {report.get('precision', 'CPU FP32')}; mean of three presence-preserving mappings; paired chunk 95% intervals", fontsize=10)
        fig.savefig(output/f'functional_paths_{metric}.pdf')
        fig.savefig(output/f'functional_paths_{metric}.png', dpi=170)
        plt.close(fig)
    pd.DataFrame(records).to_csv(output/'functional_paths.csv', index=False)


def selected_history_index(row, history):
    """Use the submitted checkpoint rule even if another epoch has lower NLL."""
    epoch=row.get('checkpoint_selection',{}).get('epoch')
    if epoch is not None:
        return next(i for i,item in enumerate(history) if item['epoch']==epoch)
    if row['config'].get('select_final_epoch'):
        return len(history)-1
    return int(np.argmin([item['validation']['NLL'] for item in history]))


def classification_plots(root, evaluation_path, output):
    """Display overfitting and the observed gradient/activation ranges."""
    evaluation=json.loads(evaluation_path.read_text())
    models=[row for row in evaluation['experiments'] if row['program']=='train'
        and not row['config'].get('selection_envelope')]
    colors={kind:f'C{i}' for i,kind in enumerate(dict.fromkeys(
        row.get('comparison_group',row['config']['initialization']) for row in models))}
    output.mkdir(parents=True,exist_ok=True)
    with PdfPages(output/'classification_training_curves.pdf') as pdf:
        for budget in sorted({row['config']['label_budget'] for row in models}):
            fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
            seen=set()
            for row in models:
                config=row['config']
                if config['label_budget']!=budget:continue
                history=json.loads((root/'experiments'/row['name']/'history.json').read_text())
                kind=row.get('comparison_group',config['initialization']);seed=config['seed'];color=colors[kind]
                label=kind if kind not in seen else None;seen.add(kind)
                x=[r['optimizer_steps_cumulative'] for r in history]
                series=[[r['train_loss'] for r in history],[r['validation']['NLL'] for r in history],
                    [r['validation']['classwise']['2']['PR_AUC'] for r in history]]
                selected=selected_history_index(row,history)
                for ax,values in zip(axes,series):
                    ax.plot(x,values,color=color,linestyle='-' if seed==42 else '--',
                        alpha=1. if seed==42 else .55,label=label)
                    ax.plot(x[selected],values[selected],'.',color=color)
            for ax,title in zip(axes,['Training cross-entropy','Validation negative log-likelihood','Validation CC nu_tau average precision']):
                ax.set(xlabel='Applied optimizer updates',ylabel=title)
            axes[0].set_yscale('log');axes[1].axhline(np.log(4),color='0.5',linewidth=.8,label='Balanced constant prior')
            axes[0].legend(fontsize=7)
            fig.suptitle(f'{budget:,} labeled events; dots mark submitted checkpoints; dashed lines show seeds 17/2026')
            pdf.savefig(fig);fig.savefig(output/f'training_curves_n{budget}.png',dpi=150);plt.close(fig)
    ranges=[]
    for row in models:
        directory=root/'experiments'/row['name']
        records=json.loads((directory/'training_health.json').read_text())['records']
        gradients=[r['gradient_squared_norms']['input'] for r in records
            if r['informative_nodes']>0 and not r.get('intentional_encoder_freeze',False)]
        silu=[a for r in records for a in r['activations']['head.2']]
        selected_path=directory/'best_checkpoint_health.json'
        selected=json.loads(selected_path.read_text()) if selected_path.exists() else {}
        ranges.append({'name':row['name'],'health_samples':len(records),
            'intentional_LP_samples':sum(r.get('intentional_encoder_freeze',False) for r in records),
            'input_gradient_squared_norm_min':min(gradients),'input_gradient_squared_norm_max':max(gradients),
            'SiLU_RMS_min':min(v['RMS'] for v in silu),'SiLU_RMS_max':max(v['RMS'] for v in silu),
            'SiLU_exact_zero_fraction_max':max(v['zero_fraction'] for v in silu),
            'maximum_preclip_gradient_norm':max(sum(r['gradient_squared_norms'].values())**.5 for r in records),
            'preclip_gradient_review_limit':row['config'].get('maximum_preclip_gradient_norm',10000.),
            'best_FP32_input_gradient_squared_norm':selected.get('health',{}).get('gradient_squared_norms',{}).get('input',np.nan)})
    table=pd.DataFrame(ranges);table.to_csv(output/'classification_health_ranges.csv',index=False)
    fig,axes=plt.subplots(1,4,figsize=(18,max(5,.27*len(table))),constrained_layout=True)
    y=np.arange(len(table))
    colors=np.where(table.maximum_preclip_gradient_norm>table.preclip_gradient_review_limit,'C3','C0')
    axes[0].hlines(y,table.input_gradient_squared_norm_min,table.input_gradient_squared_norm_max,color=colors)
    axes[0].plot(table.best_FP32_input_gradient_squared_norm,y,'x',color='C1',label='Best-checkpoint CPU FP32 probe')
    axes[0].set(xscale='log',xlabel='Input gradient squared norm: sampled range',yticks=y,yticklabels=table.name)
    axes[0].legend(fontsize=7)
    axes[1].hlines(y,table.SiLU_RMS_min,table.SiLU_RMS_max,color='C0')
    axes[1].set(xlabel='Head SiLU RMS: sampled range',yticks=y,yticklabels=[])
    axes[2].plot(table.SiLU_exact_zero_fraction_max,y,'.',color='C0')
    axes[2].set(xlabel='Largest sampled SiLU exact-zero fraction',xlim=(-.02,1.02),yticks=y,yticklabels=[])
    axes[3].scatter(table.maximum_preclip_gradient_norm,y,c=colors,s=14)
    axes[3].axvline(10000.,color='C3',linestyle='--',linewidth=.8,label='Numerical review limit')
    axes[3].set(xscale='log',xlabel='Maximum sampled preclip gradient norm',yticks=y,yticklabels=[])
    axes[3].legend(fontsize=7)
    for ax in axes:ax.invert_yaxis();ax.tick_params(axis='y',labelsize=7)
    fig.suptitle('Internal health: all scheduled BF16 samples and independent selected-checkpoint FP32 probes',fontsize=11)
    fig.savefig(output/'classification_internal_health.pdf');fig.savefig(output/'classification_internal_health.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path);parser.add_argument('--campaign',type=Path)
    parser.add_argument('--functional-report', type=Path)
    parser.add_argument('--evaluation', type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.evaluation and args.root:
        classification_plots(args.root,args.evaluation,args.output)
    elif args.functional_report:
        functional_plots(args.functional_report,args.output)
    elif args.root and args.campaign:
        pilot_plots(args.root,args.campaign,args.output)
    else:
        parser.error('provide --functional-report, or both --root and --campaign')


if __name__=='__main__':main()
