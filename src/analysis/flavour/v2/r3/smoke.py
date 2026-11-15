"""Actual R3 cache/global-order runner and deterministic recovery on Train rows."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from analysis.flavour.inventory import write_json
from analysis.flavour.v2.r3.campaign import preflight


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--attempt',required=True)
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
    for kind in ['continuous','recovered']:
        config={**base,'name':kind}
        path=output/'configs'/f'{kind}.json'
        write_json(path,config)
        if kind=='continuous':
            preflight(output,path)
        command=[sys.executable,'-m','analysis.flavour.v2.r3.train','--root',str(output),'--config',str(path)]
        env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8')
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
        'source_Train_rows':len(source_rows),'optimizer_Train_events':train_events,'synthetic_validation_events':validation_events,
        'scope':'frozen source Train rows split into optimizer Train and synthetic validation; tests actual cache,global sampler,screen,checkpoint and recovery',
        'actual_Tune_used':False,'Cal_A_Cut_B_Confirm_payload_read':False}
    write_json(output/'integrity.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
