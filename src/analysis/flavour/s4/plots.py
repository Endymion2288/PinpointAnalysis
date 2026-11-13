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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();pilot_plots(args.root,args.campaign,args.output)


if __name__=='__main__':main()
