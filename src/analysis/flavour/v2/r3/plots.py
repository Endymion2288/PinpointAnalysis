"""English R3 geometry, operation-cost and fixed-endpoint figures."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


def model_label(name):
    name=name.removesuffix('_endpoint').removesuffix('_best_screen')
    labels={'b1_m0_t4_s42':'M0 mixing','b1_g1_stride2_t4_s42':'G1 stride 2',
        'b1_d2_t4_s42':'D2 stride 4','b1_d0_t4_s42':'D0 stride 4',
        'b2_g2_control_t4_s42':'G2 control','b2_g2_relations_t4_s42':'G2 relations',
        'b2_g2_global_ablation_t4_s42':'G2 no global','b1_c1_sensor_rows_t4_s42':'C1 sensor rows'}
    return labels.get(name,name)


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
        ax.set(title=title,xticks=np.arange(len(selected)),xticklabels=[model_label(name) for name in selected.name])
        ax.tick_params(axis='x',labelrotation=35);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Fixed-exposure endpoints on development Tune; raw scores',fontsize=11)
    save(fig,folder,'r3_endpoints_'+path.parent.name)
    rows=[p for p in result['pairs'] if p['domain']=='all' and p['a'].endswith('_endpoint')]
    if rows:
        fig,axes=plt.subplots(1,3,figsize=(14,max(3.8,.65*len(rows)+1.5)))
        for ax,key,scale,title in [(axes[0],'BA3',100.,'Common BA difference (pp)'),
                (axes[1],'NLL3',1.,'Common NLL difference'),(axes[2],'tau_eff_1pct',100.,'Tau efficiency difference (pp)')]:
            # Scalar keys follow the shared evaluation contract.
            if key not in rows[0]['metrics']:
                ax.set_visible(False);continue
            for i,row in enumerate(rows):
                item=row['metrics'][key];point=item['difference_a_minus_b'];bounds=item['95pct']
                if point is None or any(v is None for v in bounds):continue
                ax.hlines(i,bounds[0]*scale,bounds[1]*scale)
                ax.plot(point*scale,i,'o')
            ax.axvline(0,color='gray',lw=1)
            ax.set(title=title,yticks=range(len(rows)),yticklabels=[model_label(p['a'])+'\nminus '+model_label(p['b']) for p in rows])
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(axis='y',labelsize=8);ax.grid(axis='x',alpha=.2)
        fig.suptitle('Paired physical-block 95% intervals; development comparisons',fontsize=11)
        save(fig,folder,'r3_contrasts_'+path.parent.name)

    # Per-class guards expose compensating changes hidden by a common BA.
    for row in rows:
        matching=[p for p in result['pairs'] if p['a']==row['a'] and p['b']==row['b']
            and p['reason']==row['reason'] and p['domain'] in ['all','occupancy_0_100']]
        fig,axes=plt.subplots(1,len(matching),figsize=(5.2*len(matching),3.8),squeeze=False)
        for ax,pair in zip(axes[0],matching):
            for i,key in enumerate(['recall_e','recall_mu','recall_NC']):
                item=pair['metrics'][key];point=item['difference_a_minus_b'];bounds=item['95pct']
                if point is None or any(v is None for v in bounds):continue
                ax.hlines(i,100*bounds[0],100*bounds[1]);ax.plot(100*point,i,'o')
            ax.axvline(0,color='gray',lw=1)
            ax.axvline(-1,color='red',ls='--',lw=1,label='1 pp loss guard')
            ax.set(xlabel='Recall difference (pp)',yticks=range(3),yticklabels=['Electron','Muon','NC'],
                title='All Tune' if pair['domain']=='all' else 'Pixel occupancy 0–100')
            ax.grid(axis='x',alpha=.2)
        fig.suptitle(model_label(row['a'])+' minus '+model_label(row['b'])+
            '\nFixed endpoints; paired block 95% intervals; red dashed: 1 pp recall-loss guard',fontsize=10)
        save(fig,folder,'r3_class_guards_'+row['a'].removesuffix('_endpoint')+'_vs_'+row['b'].removesuffix('_endpoint'))


def matched_cost(root,path):
    (root/'figures').mkdir(exist_ok=True)
    profile=json.loads(path.read_text());phases=profile['results']
    if profile['status']!='complete' or len(phases)!=2 or any(p['concurrent_jobs']!=1 for p in phases):
        raise ValueError('matched compact-image cost figure requires two sequential single-model phases')
    workers=[p['workers'][0] for p in phases]
    fig,axes=plt.subplots(1,3,figsize=(11.5,4))
    labels=['Physical grid\nstride 4','Sensor rows\nstride 2']
    for ax,values,title,unit in [
            (axes[0],[w['events_per_train_second'] for w in workers],'Training throughput','Events / training second'),
            (axes[1],[p['aggregate_operation_events_per_second'] for p in phases],'Including preparation','Events / operation second'),
            (axes[2],[w['GPU_peak_MiB'] for w in workers],'GPU allocated peak','MiB')]:
        bars=ax.bar(labels,values,color=['#4477aa','#228833'])
        ax.bar_label(bars,fmt='%.1f',padding=3);ax.set(title=title,ylabel=unit)
        ax.set_ylim(0,max(values)*1.2);ax.grid(axis='y',alpha=.2)
    fig.suptitle(profile['GPU']+'; same Train10k, 1024 updates, batch 8, BF16\n'
        'Sequential resource probe; classification quality assessed separately',fontsize=10)
    save(fig,root/'figures','r3_compact_matched_cost')


def readability(root,paths):
    models={}
    for path in paths:
        result=json.loads(path.read_text())
        if result['status']!='complete':raise ValueError('feature figures require complete probes')
        models.update(result['models'])
    branches=['local','regional','global','tracker']
    fig,axes=plt.subplots(1,2,figsize=(12,4.4))
    for ax,domain,title in [(axes[0],'all','All Tune'),
            (axes[1],'occupancy_0_100','Pixel occupancy 0–100')]:
        for name,model in models.items():
            values=[100*model['reports'][branch][domain]['T3']['BA']
                if branch in model['reports'] else np.nan for branch in branches]
            ax.plot(range(4),values,'o-',label=model_label(name))
        ax.axhline(100/3,color='gray',ls=':',lw=1)
        ax.set(title=title,xticks=range(4),xticklabels=[b.capitalize() for b in branches],
            ylabel='Common three-class probe BA (%)')
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=8)
    fig.suptitle('Frozen-feature readability; scaler and logistic C1 fitted on Train4k only\n'
        'Global = detector encoder before tracker fusion; ablated regional path is absent; not branch utility',fontsize=10)
    save(fig,root/'figures','r3_feature_readability')


def branch_health(root,paths):
    models={}
    for path in paths:
        result=json.loads(path.read_text())
        if result['status']!='complete':raise ValueError('health figures require completed optimizer-time review')
        models.update(result['models'])
    fig,axes=plt.subplots(len(models),2,figsize=(11.5,2.9*len(models)),squeeze=False)
    for row,(name,model) in enumerate(models.items()):
        branches=model['branches'];positions=np.arange(len(branches))
        for ax,median,low,high,title in [
                (axes[row,0],'gradient_RMS_median','gradient_RMS_min','gradient_RMS_max','Gradient RMS per parameter'),
                (axes[row,1],'representative_parameter_update_RMS_median',
                    'representative_parameter_update_RMS_min','representative_parameter_update_RMS_max',
                    'Representative tensor update RMS')]:
            for i,branch in enumerate(branches):
                if branch[high]>0:
                    minimum=branch.get(low.replace('_min','_positive_min'),branch[low])
                    if minimum is not None and minimum>0:
                        ax.hlines(i,minimum,branch[high],color='#4477aa',alpha=.6)
                if branch[median]>0:ax.plot(branch[median],i,'o',color='#228833',ms=4)
                if branch[low]==0:ax.text(.02,i,'zero steps',transform=ax.get_yaxis_transform(),fontsize=6,va='center')
            ax.set(xscale='log',yticks=positions,yticklabels=[b['branch'] for b in branches],
                title=model_label(name)+' — '+title)
            ax.tick_params(axis='y',labelsize=7);ax.grid(axis='x',alpha=.2)
    fig.suptitle('Actual sampled optimizer steps: positive ranges / overall medians; zero steps marked\n'
        'Positive gradients and updates measure use during fitting; activation and small-fit checks reported separately',fontsize=10)
    save(fig,root/'figures','r3_training_branch_health')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--phase',type=Path)
    parser.add_argument('--profile',type=Path)
    parser.add_argument('--evaluations',type=Path,nargs='*',default=[])
    parser.add_argument('--matched-cost',type=Path)
    parser.add_argument('--probes',type=Path,nargs='*',default=[])
    parser.add_argument('--health',type=Path,nargs='*',default=[])
    args=parser.parse_args()
    if args.phase and args.profile:diagnostics(args.root,args.phase,args.profile)
    for path in args.evaluations:endpoint_figures(args.root,path)
    if args.matched_cost:matched_cost(args.root,args.matched_cost)
    if args.probes:readability(args.root,args.probes)
    if args.health:branch_health(args.root,args.health)


if __name__=='__main__':main()
