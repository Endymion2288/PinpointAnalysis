"""Actual R3 cache/global-order runner and deterministic recovery on Train rows."""

import argparse
import json
import os
import pickle
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

from analysis.flavour.inventory import write_json
from analysis.flavour.v2.r3.campaign import preflight
from analysis.flavour.v2.r3.data import R3Dataset


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--attempt',required=True)
    parser.add_argument('--shared-inputs',action='store_true')
    args=parser.parse_args()
    output=args.root/'smoke'/args.attempt
    output.mkdir(parents=True,exist_ok=False)
    campaign=json.loads((args.root/'campaign.json').read_text())
    source=Path(campaign['r2'])/'smoke/d01_k3/data'
    base=json.loads(args.config.read_text())
    base.update(data_root=str(source),passes=3,deterministic_verification=True,
        tune_screen=str(source/'tune_screen.parquet'),validation_non_tau_exposures=10,
        preflight=str(output/'preflight/case.json'))
    if torch.cuda.get_device_capability()[0]<8:
        base.update(bf16=False,precision='FP32')
    env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8')
    preparation={}
    if args.shared_inputs:
        temporary=tempfile.TemporaryDirectory(prefix='r3-shared-smoke-',
            dir=os.environ.get('PINPOINT_V2_SCRATCH','/tmp'))
        keys=['family','task','native_pixel_pitch_mm','build_relations',
            'relation_plane_policy','image','sensor_row_image','drop_modality']
        signature={k:base.get(k) for k in keys}
        for split,var in [('train','PINPOINT_R3_SHARED_TRAIN_INPUTS'),
                ('validation','PINPOINT_R3_SHARED_TUNE_INPUTS')]:
            common=R3Dataset(source,split,{**base,'cache_training_inputs':False,
                'cache_storage':'mmap','cache_workers':2},cache_dir=temporary.name)
            common.cache_inputs()
            payload=Path(temporary.name)/(split+'.pkl')
            with payload.open('wb') as stream:
                pickle.dump({'root':str(source.resolve()),'input_config':signature,
                    'keys':common.metadata[['run','chunk','row','label']].to_numpy(),
                    'items':common.items,'audit':common.cache_audit},stream,protocol=5)
            env[var]=str(payload);preparation[split]=common.cache_audit
            del common
    for kind in ['continuous','recovered']:
        config={**base,'name':kind}
        path=output/'configs'/f'{kind}.json'
        write_json(path,config)
        if kind=='continuous':
            preflight(output,path)
        command=[sys.executable,'-m','analysis.flavour.v2.r3.train','--root',str(output),'--config',str(path)]
        subprocess.run(command+(['--stop-after-updates','4'] if kind=='recovered' else []),env=env,check=True)
        if kind=='recovered':
            subprocess.run(command+['--resume'],env=env,check=True)
    a,b=[torch.load(output/'experiments'/name/'exposure.pt',map_location='cpu',weights_only=False)
        for name in ['continuous','recovered']]
    delta=max(float((a['model'][key]-b['model'][key]).abs().max()) for key in a['model'])
    if delta!=0 or not np.array_equal(a['counts'],b['counts']):
        raise ValueError('R3 mid-pass recovery changed parameters or row exposures')
    for name in ['predictions.parquet','best_predictions.parquet','train_exposures.parquet']:
        x,y=[pd.read_parquet(output/'experiments'/kind/name) for kind in ['continuous','recovered']]
        pd.testing.assert_frame_equal(x,y,check_exact=True)
    source_rows=pd.read_parquet(source/'catalog.parquet')
    train_events=int(source_rows.split.eq('train').sum())
    validation_events=int(source_rows.split.eq('validation').sum())
    result={'status':'complete','maximum_parameter_delta':delta,'same_exact_probability_and_exposure_tables':True,
        'precision':base['precision'],'GPU':torch.cuda.get_device_name(),
        'shared_transformed_inputs':args.shared_inputs,'shared_preparation':preparation,
        'source_Train_rows':len(source_rows),'optimizer_Train_events':train_events,'synthetic_validation_events':validation_events,
        'event_mixing':config.get('event_mixing','spatial'),
        'scope':'frozen source Train rows split into optimizer Train and synthetic validation; tests actual configured cache,sampler,screen,checkpoint and recovery',
        'actual_Tune_used':False,'Cal_A_Cut_B_Confirm_payload_read':False}
    write_json(output/'integrity.json',result)
    print(json.dumps(result),flush=True)
    if args.shared_inputs:temporary.cleanup()


if __name__=='__main__':
    main()
