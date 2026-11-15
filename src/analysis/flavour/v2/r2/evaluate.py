"""Fixed-endpoint scaling contrasts and separately labeled best-screen results."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.s3.metrics import bootstrap_weights
from analysis.flavour.v2.partitions import KEY, FAMILY, equal_identity_rows
from analysis.flavour.v2.r1.accept import DOMAINS
from analysis.flavour.v2.r1.evaluate import domain_report, scalar_summary, interval
from analysis.flavour.v2.r1.interval_review import review_intervals
from analysis.flavour.v2.r1.metrics import METRICS_REVISION
from analysis.flavour.v2.r1.review import conditional_probabilities, direct_metrics, selected_rates


def domains(frame):
    channels=frame.pixel_channels+frame.bar_channels+frame.tracker_channels
    result={'all':np.ones(len(frame),dtype=bool),'pixel_centered':frame.pixel_centered.to_numpy(dtype=bool),
        'large_scintillator':frame.large_scintillator.to_numpy(dtype=bool),'active':~frame.all_empty.to_numpy(dtype=bool),
        'empty':frame.all_empty.to_numpy(dtype=bool),'tracker_active':frame.tracker_channels.to_numpy()>0,
        'tracker_empty':frame.tracker_channels.to_numpy()==0}
    for left,right in [(0,100),(100,1000),(1000,5000),(5000,np.inf)]:
        result[f'occupancy_{left}_{right}']=channels.ge(left).to_numpy() & channels.lt(right).to_numpy()
    return result


def configuration(root, complete_only=False):
    campaign=json.loads((root/'campaign.json').read_text())
    models=[]
    for name in campaign['initial_models']+campaign.get('extension_models',[]):
        path=root/'experiments'/name/'manifest.json'
        if complete_only and (not path.exists() or json.loads(path.read_text())['status']!='complete'):
            continue
        config=json.loads((root/'configs'/f'{name}.json').read_text())
        for checkpoint,table in [('endpoint','predictions.parquet'),('best_screen','best_predictions.parquet')]:
            models.append({'name':f'{name}_{checkpoint}','experiment':name,'family':config['family'],
                'task':config['task'],'seed':config['seed'],'design':config['design'],'kind':checkpoint,
                'predictions':str(Path('experiments')/name/table)})
    available={m['name'] for m in models}
    pairs=[]
    def add(a,b,reason,kind,domain='all'):
        a,b=f'{a}_{kind}',f'{b}_{kind}'
        if a in available and b in available:pairs.append({'a':a,'b':b,'domain':domain,'reason':reason})
    for kind in ['endpoint','best_screen']:
        for task in ['t3','t4']:
            for a,b,reason in [('d1','d0','same_unique_extended_budget'),('d2','d1','same_exposures_more_unique'),
                               ('d2','d0','combined_data_and_budget')]:
                for domain in ['all','pixel_centered','large_scintillator','occupancy_0_100']:
                    add(f'b1_{a}_{task}_s42',f'b1_{b}_{task}_s42',reason,kind,domain)
        for design in ['d0','d1','d2']:
            for domain in ['all','pixel_centered','large_scintillator','occupancy_0_100']:
                add(f'b1_{design}_t3_s42',f'b1_{design}_t4_s42','T3_minus_conditional_T4',kind,domain)
        for pair in campaign.get('extension_pairs',[]):
            for domain in pair['domains']:add(pair['a'],pair['b'],pair['reason'],kind,domain)
    return {'models':models,'pairs':pairs,'endpoint_primary':True,
        'best_screen_secondary':'minimum own-task NLL on fixed screen; no full-Tune checkpoint selection',
        'seed_scope':'single matched seed42; event bootstrap does not estimate training-seed variability'}


def independent_points(frames,reports):
    checks=[]
    def check(name,passed):checks.append({'name':name,'passed':bool(passed)})
    for name,frame in frames.items():
        original,common=conditional_probabilities(frame)
        y=frame.label.to_numpy()
        check(name+'_valid_probability',np.isfinite(original).all() and (original>=0).all()
            and (original<=1).all() and np.allclose(original.sum(1),1.,atol=2e-6))
        for domain,mask in domains(frame).items():
            report=reports[name][domain]
            check(name+'_'+domain+'_raw_counts',np.array_equal(np.bincount(y[mask],minlength=4),report['raw_counts_original_labels']))
            for task in ['T3','T4']:
                if task=='T4' and original.shape[1]==3:continue
                keep=mask&(y!=2) if task=='T3' else mask
                if not keep.any():continue
                labels=np.searchsorted([0,1,3],y[keep]) if task=='T3' else y[keep]
                p=common[keep] if task=='T3' else original[keep]
                measured,support,balanced=direct_metrics(labels,p)
                row=report[task]
                check(name+'_'+domain+'_'+task+'_BA_NLL',all(abs(measured[k]-row[k])<2e-9 for k in measured))
                ranking = independent_ranking(labels, p)
                for c, values in enumerate(ranking):
                    for metric, value in values.items():
                        expected = row['per_class'][c][metric]
                        check(f'{name}_{domain}_{task}_{c}_{metric}',
                            expected is None if value is None else expected is not None and abs(expected-value)<2e-9)
                for c in range(p.shape[1]):
                    for target in [.01,.001]:
                        item=row['per_class'][c]
                        if support[c]==0 or support.sum()==support[c]:continue
                        _,counts,acceptance,fpr=selected_rates(labels,p,c,item[f'threshold_FPR_{target}'],balanced)
                        check(f'{name}_{domain}_{task}_{c}_{target}_working_point',
                            abs(acceptance[c]-item[f'eff_FPR_{target}'])<2e-9
                            and abs(fpr-item[f'actual_FPR_{target}'])<2e-9 and fpr<=target+2e-9)
    failed=[c['name'] for c in checks if not c['passed']]
    return {'status':'complete' if not failed else 'failed','passed':len(checks)-len(failed),'failed':failed,'checks':checks,
        'scope':'direct per-class arithmetic, selected raw counts and independent sklearn weighted AP/AUROC; no Metrics/scalar_summary calls'}


def independent_ranking(labels, probabilities):
    """Check stored score rankings with sklearn under the declared domain prior."""
    classes = probabilities.shape[1]
    support = np.bincount(labels, minlength=classes)
    weight = len(labels)/(classes*support[labels]) if (support>0).all() else np.ones(len(labels))
    values = []
    for c in range(classes):
        truth = labels==c
        values.append({
            'AP': float(average_precision_score(truth, probabilities[:, c], sample_weight=weight))
                if truth.any() else None,
            'AUROC': float(roc_auc_score(truth, probabilities[:, c], sample_weight=weight))
                if truth.any() and (~truth).any() else None})
    return values


def run(root,replicates,workers,complete_only):
    started=time.perf_counter()
    config=configuration(root,complete_only)
    if not config['models']:raise ValueError('no completed scaling candidates')
    destination=root/('evaluation_partial' if complete_only else 'evaluation')
    destination.mkdir(exist_ok=True)
    write_json(destination/'config.json',config)
    catalog=pd.read_parquet(root/'data/catalog.parquet',filters=[('split','=','validation')]).sort_values(KEY).reset_index(drop=True)
    masks=domains(catalog)
    if set(masks)!=DOMAINS:raise ValueError('domain contract changed')
    frames,reports,rows={},{},[]
    for entry in config['models']:
        record=json.loads((root/'experiments'/entry['experiment']/'manifest.json').read_text())
        if record['status']!='complete':raise ValueError('scaling evaluation requires completed candidates')
        frame=pd.read_parquet(root/entry['predictions']).sort_values(KEY).reset_index(drop=True)
        if not equal_identity_rows(frame,catalog) or not np.array_equal(frame.label,catalog.label):
            raise ValueError('candidate does not cover exactly frozen Tune identities and labels')
        frames[entry['name']]=frame
        reports[entry['name']]={domain:domain_report(frame,mask) for domain,mask in masks.items()}
        for domain,result in reports[entry['name']].items():
            row={**{k:entry[k] for k in ['name','experiment','family','task','seed','design','kind']},'domain':domain,
                'events':result['events'],'non_tau_events':result['non_tau_events'],'GPU':record['GPU'],
                'precision':record['precision'],'updates':record['counters']['updates'],
                'exposures':record['counters']['training_exposures'],'non_tau_exposures':record['counters']['non_tau_exposures']}
            for task in ['T3','T4']:
                if result.get(task):
                    row.update({k+task[-1]:result[task][k] for k in ['BA','NLL','macro_AP','macro_AUROC']})
                    row.update({f'{task}_recall_{c}':v['recall'] for c,v in enumerate(result[task]['per_class'])})
            if result.get('T4'):
                row.update({f'tau_eff_{fpr}':result['T4']['per_class'][2].get(f'eff_FPR_{fpr}') for fpr in [.01,.001]})
            rows.append(row)
    weights,design=bootstrap_weights(catalog,replicates,20261116)
    catalog[KEY+FAMILY+['label']].to_parquet(destination/'bootstrap_keys.parquet',index=False,compression=COMPRESSION)
    needed=sorted({(p[side],p['domain']) for p in config['pairs'] for side in ['a','b']})
    def resample(key):
        name,domain=key
        mask=masks[domain]
        compute=scalar_summary(frames[name].loc[mask].reset_index(drop=True))
        return key,(compute(),np.asarray([compute(w[mask]) for w in weights]))
    with ThreadPoolExecutor(max_workers=workers) as executor:cache=dict(executor.map(resample,needed))
    pairs=[]
    for pair in config['pairs']:
        a,b=cache[(pair['a'],pair['domain'])],cache[(pair['b'],pair['domain'])]
        pairs.append({**pair,'metrics':interval(a[0]-b[0],a[1]-b[1])})
    result={'status':'complete','models':reports,'pairs':pairs,'mean_seed_pairs':[],
        'replicates':replicates,'bootstrap':design,'metrics_revision':METRICS_REVISION,
        'scope':'development Tune; endpoint primary, best-screen secondary; score-selected cuts descriptive',
        'seed_scope':config['seed_scope'],'seconds':time.perf_counter()-started,
        'Cal_A_Cut_B_Confirm_payload_read':False}
    pd.DataFrame(rows).to_csv(destination/'model_domain_metrics.csv',index=False)
    write_json(destination/'results.json',result)
    points=independent_points(frames,reports)
    write_json(destination/'independent_points.json',points)
    core=review_intervals(root,config,result)
    write_json(destination/'independent_core_intervals.json',core)
    if points['failed'] or core['failed']:raise ValueError('independent scaling readback failed')
    print(json.dumps({'status':'complete','models':len(frames),'pairs':len(pairs),
        'point_checks':points['passed'],'core_checks':core['passed'],'output':str(destination)}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--replicates',type=int,default=400)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--complete-only',action='store_true')
    args=parser.parse_args()
    run(args.root,args.replicates,args.workers,args.complete_only)


if __name__=='__main__':main()
