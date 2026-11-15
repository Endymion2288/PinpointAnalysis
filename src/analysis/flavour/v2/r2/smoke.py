"""Check the actual runner, endpoint inference and exact mid-pass recovery on Train only."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from analysis.flavour.create_data import lists
from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.v2.fit import fit_event_positions
from analysis.flavour.v2.r1.config import recipe
from analysis.flavour.v2.r1.data import R1Dataset
from analysis.flavour.v2.r2.campaign import config_from_r1, preflight


def run(root, r1, r0, attempt, bf16, tasks=('T3','T4'), simulate_migration=False, workers=2):
    output=root/'smoke'/attempt
    output.mkdir(parents=True,exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='r2-smoke-') as scratch:
        source=R1Dataset(r0/'micro_data','train',recipe(r0,'B1'),cache_dir=scratch)
        selected=pd.read_parquet(r0/'fit/fit32_a2_events.parquet').copy()
        positions=fit_event_positions(selected,source.metadata)
        selected['split']=np.where(selected.groupby('label').cumcount()<2,'validation','train')
        selected['pixel_centered']=True
        selected['large_scintillator']=True
        observed=[]
        for index in positions:
            native=source.native[int(source.indexes[index])]['observations']
            observed.append({**{name:lists(native[name]) for name in ['pixel','bar','tracker_hypothesis']},
                'has_response':np.asarray(native['has_response']).tolist(),'outside_acceptance':[-1]*3,
                'fit_attempted':False,'momentum_censored':-1})
        source.native.close()
    shard=output/'data/shards/000'
    shard.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(observed),shard/'observations.parquet',compression=COMPRESSION)
    pq.write_table(pa.Table.from_pylist([{'label':int(y),'label_status':'consistent'} for y in selected.label]),
        shard/'targets.parquet',compression=COMPRESSION)
    selected.to_parquet(shard/'provenance.parquet',index=False,compression=COMPRESSION)
    selected.to_parquet(output/'data/catalog.parquet',index=False,compression=COMPRESSION)
    selected[selected.split.eq('validation')].to_parquet(output/'data/tune_screen.parquet',index=False,compression=COMPRESSION)
    write_json(output/'data/plan.json',json.loads((r0/'micro_data/plan.json').read_text()))
    cases=[]
    for task in tasks:
        config=config_from_r1(r1,'B1',task,precision='BF16_encoder_FP32_head' if bf16 else 'FP32')
        config.update(data_root=str(output/'data'),passes=3,deterministic_verification=True,workers=workers,
            tune_screen=str(output/'data/tune_screen.parquet'),
            validation_non_tau_exposures=10,preflight=str(output/'preflight'/f'{task}.json'))
        for kind in ['continuous','recovered']:
            config['name']=f'{task.lower()}_{kind}'
            path=output/'configs'/f'{config["name"]}.json'
            write_json(path,config)
            if kind=='continuous':preflight(output,path)
            command=[sys.executable,'-m','analysis.flavour.v2.r2.train','--root',str(output),'--config',str(path)]
            env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8')
            if simulate_migration:env['PINPOINT_JOB_ID']='smoke-simulated-initial-label'
            subprocess.run(command+(['--stop-after-updates','4'] if kind=='recovered' else []),env=env,check=True)
            if kind=='recovered':
                if simulate_migration:env['PINPOINT_JOB_ID']='smoke-simulated-recovery-label'
                subprocess.run(command+['--resume'],env=env,check=True)
        left=output/'experiments'/f'{task.lower()}_continuous'
        right=output/'experiments'/f'{task.lower()}_recovered'
        a=torch.load(left/'exposure.pt',map_location='cpu',weights_only=False)
        b=torch.load(right/'exposure.pt',map_location='cpu',weights_only=False)
        delta=max(float((a['model'][key]-b['model'][key]).abs().max()) for key in a['model'])
        predictions=[]
        for name in ['predictions.parquet','best_predictions.parquet','train_exposures.parquet']:
            x,y=pd.read_parquet(left/name),pd.read_parquet(right/name)
            pd.testing.assert_frame_equal(x,y,check_exact=False,rtol=1e-6,atol=1e-7)
            predictions.append({'file':name,'rows':len(x)})
        if delta>1e-6:raise ValueError(f'mid-pass recovery changed model parameters: {delta}')
        manifests=[json.loads((p/'manifest.json').read_text()) for p in [left,right]]
        for field in ['updates','training_exposures','non_tau_exposures','exposure_by_class','lr_sum']:
            if manifests[0]['counters'][field]!=manifests[1]['counters'][field]:
                raise ValueError(f'recovery changed {field}')
        if not all(m['status']=='complete' and len(m['history'])>1 for m in manifests):
            raise ValueError('screen and completion paths did not run')
        for m in manifests:
            if m['wall_seconds']+1e-6<m['counters']['train_seconds']+m['counters']['validation_seconds']:
                raise ValueError('operation wall excludes recoverable work from an earlier attempt')
        if len(manifests[1]['attempts'])!=2:
            raise ValueError('recovery attempt provenance lost')
        if simulate_migration and (manifests[1]['allocation_id']!='smoke-simulated-recovery-label'
                or manifests[1]['attempts'][0]['allocation_id']!='smoke-simulated-initial-label'):
            raise ValueError('current and historical allocation provenance conflated')
        cases.append({'task':task,'maximum_parameter_delta':delta,'tables':predictions,
            'updates':manifests[0]['counters']['updates'],'screens':len(manifests[0]['history']),
            'resume_at_update':4,'precision':config['precision']})
    report={'status':'complete','cases':cases,'scope':'32 frozen Train rows with synthetic validation; not generalization',
        'simulated_allocation_label_transition':simulate_migration,
        'actual_Tune_used':False,'Cal_A_Cut_B_Confirm_payload_read':False}
    write_json(output/'integrity.json',report)
    print(json.dumps(report),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--r1',type=Path,required=True)
    parser.add_argument('--r0',type=Path,required=True)
    parser.add_argument('--attempt',required=True)
    parser.add_argument('--bf16',action='store_true')
    parser.add_argument('--tasks',choices=['T3','T4'],nargs='+',default=['T3','T4'])
    parser.add_argument('--simulate-migration',action='store_true')
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    run(args.root,args.r1,args.r0,args.attempt,args.bf16,args.tasks,args.simulate_migration,args.workers)


if __name__=='__main__':
    main()
