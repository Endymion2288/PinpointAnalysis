"""B0 and explicitly separated sample-prior diagnostics on the S2 subset."""

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import subprocess
import time

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
from sklearn.ensemble import HistGradientBoostingClassifier

from analysis.flavour.features import FEATURE_NAMES
from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.s2.train import metrics


MODES=['b0_all','b0_signal','b0_shape','truth_energy_vertex']


def run(args):
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    started=time.perf_counter()
    features=[]
    provenance=[]
    for path in sorted((args.root/'shards').iterdir()):
        features.append(pd.read_parquet(path/'b0_features.parquet'))
        provenance.append(pd.read_parquet(path/'provenance.parquet'))
    feature=pd.concat(features,ignore_index=True)
    meta=pd.concat(provenance,ignore_index=True)
    if not meta.split.isin(['train','validation']).all():
        raise ValueError('reference training requires S2 train/validation only')
    if args.mode=='b0_all':
        columns=FEATURE_NAMES
    elif args.mode=='b0_signal':
        columns=[c for c in FEATURE_NAMES if c.startswith('log_') or c.endswith('_present')]
    elif args.mode=='b0_shape':
        columns=[c for c in FEATURE_NAMES if '_mean_' in c or '_std_' in c or c in ['pixel_planes','bar_h_fraction']]
    elif args.mode=='truth_energy_vertex':
        columns=['initE','initX','initY','initZ']
        feature=meta
    else:
        raise ValueError(args.mode)
    x=feature[columns].to_numpy(dtype='float32')
    y=meta.label.to_numpy(dtype='int64')
    train=meta.split.eq('train').to_numpy()
    validation=meta.split.eq('validation').to_numpy()
    output=args.root/'experiments'/f'{args.mode}_s{args.seed}'
    output.mkdir(parents=True,exist_ok=True)
    config={'name':output.name,'architecture':'hist_gradient_boosting','seed':args.seed,
            'role':'truth_prior_diagnostic' if args.mode.startswith('truth_') else 'B0_reference',
            'feature_mode':args.mode,'feature_columns':columns,'max_iterations':200,
            'learning_rate':.08,'max_leaf_nodes':15,'min_samples_leaf':40,'l2_regularization':1.}
    manifest={'utc':datetime.now(timezone.utc).isoformat(),'config':config,'status':'running',
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'root':str(args.root),'train_events':int(train.sum()),'validation_events':int(validation.sum()),
        'response_level':'user_accepted_detector_proxy' if args.mode.startswith('b0') else 'truth_prior_diagnostic',
        'hardware_model':args.mode.startswith('b0'),'calibration_test_used':False,
        'selection':'minimum_validation_NLL_at_10_iteration_intervals','parameters':None,
        'gpu_name':None,'gpu_hours':0.,'device':'cpu','load_seconds':time.perf_counter()-started}
    write_json(output/'manifest.json',manifest)
    model=HistGradientBoostingClassifier(max_iter=10,learning_rate=.08,max_leaf_nodes=15,
        min_samples_leaf=40,l2_regularization=1.,early_stopping=False,warm_start=True,random_state=args.seed)
    best=float('inf')
    history=[]
    for iteration in range(10,201,10):
        model.set_params(max_iter=iteration)
        t=time.perf_counter()
        model.fit(x[train],y[train])
        fit_seconds=time.perf_counter()-t
        p=model.predict_proba(x[validation]).astype('float32')
        measured=metrics(y[validation],p)
        history.append({'iteration':iteration,'validation':measured,'fit_increment_seconds':fit_seconds,
                        'wall_seconds':time.perf_counter()-started})
        if measured['NLL']<best:
            best=measured['NLL']
            selected_iteration=iteration
            best_model=copy.deepcopy(model)
            predictions=meta.loc[validation].reset_index(drop=True).copy()
            for c in range(4):
                predictions[f'p{c}']=p[:,c]
            predictions.to_parquet(output/'validation_predictions.parquet',index=False,compression=COMPRESSION)
            write_json(output/'metrics.json',measured)
        write_json(output/'history.json',history)
        print(json.dumps({'mode':args.mode,'iteration':iteration,'NLL':measured['NLL'],
                          'tau_AP':measured['classwise']['2']['PR_AUC']}),flush=True)
    joblib.dump({'model':best_model,'features':columns,'config':config},output/'best.joblib',compress=3)
    manifest.update(status='complete',selected_iteration=selected_iteration,best_NLL=best,
        seconds=time.perf_counter()-started,peak_gpu_allocated_mib=0.,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    write_json(output/'manifest.json',manifest)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--mode',choices=MODES,required=True)
    parser.add_argument('--seed',type=int,default=42)
    run(parser.parse_args())


if __name__=='__main__':
    main()
