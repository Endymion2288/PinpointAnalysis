"""Standalone English S4 pretraining and label-efficiency diagnostics."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
            'Changed minus original CPU FP32; mean of three presence-preserving mappings; paired chunk 95% intervals', fontsize=10)
        fig.savefig(output/f'functional_paths_{metric}.pdf')
        fig.savefig(output/f'functional_paths_{metric}.png', dpi=170)
        plt.close(fig)
    pd.DataFrame(records).to_csv(output/'functional_paths.csv', index=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path);parser.add_argument('--campaign',type=Path)
    parser.add_argument('--functional-report', type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.functional_report:
        functional_plots(args.functional_report,args.output)
    elif args.root and args.campaign:
        pilot_plots(args.root,args.campaign,args.output)
    else:
        parser.error('provide --functional-report, or both --root and --campaign')


if __name__=='__main__':main()
