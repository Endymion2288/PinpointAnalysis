"""English R3 geometry, operation-cost and fixed-endpoint figures."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save(fig,folder,name):
    fig.tight_layout()
    for suffix in ['png','pdf']:fig.savefig(folder/(name+'.'+suffix),dpi=160,bbox_inches='tight')
    plt.close(fig)


def diagnostics(root,phase_path,profile_path):
    folder=root/'figures';folder.mkdir(exist_ok=True)
    phase=json.loads(phase_path.read_text())
    fig,axes=plt.subplots(2,2,figsize=(10,8))
    for i,(sample,label) in enumerate([('synthetic','Synthetic points / short lines'),('real_Train','Real Train responses')]):
        record=phase['results'][sample]
        for j,(metric,title) in enumerate([('phase_mean_max_probability_abs_difference','Mean max probability change'),
                ('phase_argmax_agreement','Argmax agreement')]):
            values=np.array([[record[metric][f'{r},{c}'] for c in range(4)] for r in range(4)])
            ax=axes[i,j];im=ax.imshow(values,cmap='viridis' if j else 'magma',vmin=0,vmax=1 if j else None)
            for r in range(4):
                for c in range(4):ax.text(c,r,f'{values[r,c]:.3f}',ha='center',va='center',color='white' if values[r,c]<(0.55 if j else .65*values.max()) else 'black')
            ax.set(title=label+'\n'+title,xlabel='Column phase (cells)',ylabel='Row phase (cells)',xticks=range(4),yticks=range(4))
            fig.colorbar(im,ax=ax,fraction=.046)
    fig.suptitle('Fixed endpoint weights; fixed expanded canvas; all measurements retained\nSampling sensitivity on Train, not a classification-gain estimate',fontsize=11)
    save(fig,folder,'r3_phase_sensitivity')
    profile=json.loads(profile_path.read_text())
    operations=profile['results'];validation=profile['full_validation']
    fig,axes=plt.subplots(1,3,figsize=(12,4.3))
    labels=['Stride 4','Stride 2']
    axes[0].bar(labels,[r['workers'][0]['events_per_train_second'] for r in operations[:2]])
    axes[0].set(title='Training throughput',ylabel='Events / training second')
    axes[1].bar(labels,[r['workers'][0]['GPU_peak_MiB'] for r in operations[:2]])
    axes[1].set(title='Training allocated peak',ylabel='MiB')
    sequential=sum(r['wall_seconds'] for r in operations[:2]);parallel=operations[2]['wall_seconds']
    axes[2].bar(['Sequential pair','Concurrent pair'],[sequential,parallel])
    axes[2].set(title='Two independent 256-update fits',ylabel='Complete operation seconds')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.suptitle(profile['GPU']+'; '+('BF16 encoder / FP32 head' if profile['BF16'] else 'FP32'),fontsize=11)
    save(fig,folder,'r3_stride_cost_profile')
    graphs=sorted((root/'diagnostics').glob('graph_*/results.json'))
    for path in graphs:
        graph=json.loads(path.read_text())
        if 'variants' not in graph:continue
        fig,axes=plt.subplots(1,2,figsize=(10,4.2))
        for policy,item in graph['variants'].items():
            axes[0].plot([1,2,3],np.array(item['recorded_first_contributor_recall']['recall'])*100,'o-',label=policy.replace('_',' '))
            axes[1].plot([50,95,99,100],[item['edge_count_quantiles'][k] for k in ['p50','p95','p99','max']],'o-',label=policy.replace('_',' '))
        axes[0].set(xlabel='Visible pixel-plane separation',ylabel='Recorded-label candidate recall (%)',xticks=[1,2,3],ylim=(0,100))
        axes[1].set(xlabel='Event percentile',ylabel='Candidate edges / event',yscale='symlog')
        for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
        fig.suptitle('Balanced Train audit; first-contributor labels do not encode complete ancestry',fontsize=11)
        save(fig,folder,'r3_candidate_coverage_'+path.parent.name)


def endpoint_figures(root,path):
    folder=root/'figures';folder.mkdir(exist_ok=True)
    result=json.loads(path.read_text());table=pd.read_csv(path.parent/'model_domain_metrics.csv')
    selected=table[table.kind.eq('endpoint')&table.domain.eq('all')]
    fig,axes=plt.subplots(1,3,figsize=(13,4.7))
    for ax,key,scale,title in [(axes[0],'BA3',100.,'Common three-class BA (%)'),
            (axes[1],'NLL3',1.,'Common three-class NLL'),(axes[2],'tau_eff_0.01',100.,'Tau efficiency at 1% FPR (%)')]:
        ax.bar(np.arange(len(selected)),selected[key]*scale)
        ax.set(title=title,xticks=np.arange(len(selected)),xticklabels=[name.replace('_endpoint','') for name in selected.name])
        ax.tick_params(axis='x',labelrotation=70);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Fixed-exposure endpoints on development Tune; raw scores',fontsize=11)
    save(fig,folder,'r3_endpoints_'+path.parent.name)
    rows=[p for p in result['pairs'] if p['domain']=='all' and p['a'].endswith('_endpoint')]
    if rows:
        fig,axes=plt.subplots(1,3,figsize=(13,4.7))
        for ax,key,scale,title in [(axes[0],'BA3',100.,'Common BA difference (pp)'),
                (axes[1],'NLL3',1.,'Common NLL difference'),(axes[2],'tau_eff_1pct',100.,'Tau efficiency difference (pp)')]:
            # Scalar keys follow the shared evaluation contract.
            if key not in rows[0]['metrics']:
                ax.set_visible(False);continue
            for i,row in enumerate(rows):
                item=row['metrics'][key];point=item['difference_a_minus_b'];bounds=item['95pct']
                if point is None or any(v is None for v in bounds):continue
                ax.vlines(i,bounds[0]*scale,bounds[1]*scale)
                ax.plot(i,point*scale,'o')
            ax.axhline(0,color='gray',lw=1)
            ax.set(title=title,xticks=range(len(rows)),xticklabels=[p['reason'] for p in rows]);ax.tick_params(axis='x',labelrotation=70);ax.grid(axis='y',alpha=.2)
        fig.suptitle('Paired physical-block 95% intervals; development comparisons',fontsize=11)
        save(fig,folder,'r3_contrasts_'+path.parent.name)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--phase',type=Path)
    parser.add_argument('--profile',type=Path)
    parser.add_argument('--evaluations',type=Path,nargs='*',default=[])
    args=parser.parse_args()
    if args.phase and args.profile:diagnostics(args.root,args.phase,args.profile)
    for path in args.evaluations:endpoint_figures(args.root,path)


if __name__=='__main__':main()
