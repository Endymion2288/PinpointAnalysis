"""Reuse frozen R2 grouped folds to diagnose R3 decision shifts, without HPO."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.flavour.inventory import write_json
from analysis.flavour.v2.partitions import KEY,equal_identity_rows
from analysis.flavour.v2.r1.review import conditional_probabilities
from analysis.flavour.v2.r2.evaluate import domains
from analysis.flavour.v2.r2.readout import crossfit,report,convergence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--experiments',type=Path,nargs='+',required=True)
    parser.add_argument('--attempt',required=True)
    args=parser.parse_args()
    campaign=json.loads((args.root/'campaign.json').read_text())
    parent=Path(campaign['r2'])/'readout'
    contract=json.loads((parent/'contract.json').read_text())
    folds=pd.read_parquet(parent/'folds.parquet').sort_values(KEY).reset_index(drop=True)
    output=args.root/'readout'/args.attempt
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'contract.json',{'status':'frozen','UTC':datetime.now(timezone.utc).isoformat(),
        'experiments':list(map(str,args.experiments)),'parent_contract':str(parent/'contract.json'),
        'parent_folds':str(parent/'folds.parquet'),'fixed_recipe':contract,
        'scope':'repeated development Tune diagnostic; reuses identical family-connected five folds; no HPO, no independent or promotion claim',
        'Cal_A_Cut_B_Confirm_payload_read':False})
    models={}
    for experiment in args.experiments:
        manifest=json.loads((experiment/'manifest.json').read_text())
        if manifest['status']!='complete':raise ValueError('readout requires completed fixed endpoints')
        frame=pd.read_parquet(experiment/'predictions.parquet').sort_values(KEY).reset_index(drop=True)
        if not equal_identity_rows(frame,folds) or not np.array_equal(frame.label,folds.label):
            raise ValueError('readout identities/labels differ from frozen common Tune folds')
        keep=frame.label.to_numpy()!=2
        labels=np.searchsorted([0,1,3],frame.label.to_numpy()[keep])
        probability=conditional_probabilities(frame)[1][keep]
        assignment=folds.fold.to_numpy()[keep]
        variants,fits=crossfit(probability,labels,assignment,contract)
        masks={domain:mask[keep] for domain,mask in domains(frame).items()}
        raw=report(labels,probability);original=manifest['metrics']['T3']
        if any(abs(raw[k]-original[k])>1e-10 for k in ['BA','NLL','macro_AP','macro_AUROC']):
            raise ValueError('independent conditional raw metrics differ from completed endpoint')
        models[experiment.name]={'experiment':str(experiment),'fits':fits,
            'convergence':convergence(manifest,args.root),'temperature_argmax_identical':True,
            'domains':{d:{name:report(labels[mask],p[mask]) for name,p in variants.items()}
                for d,mask in masks.items() if mask.any()},
            'fold_metrics':{str(fold):{name:report(labels[assignment==fold],p[assignment==fold])
                for name,p in variants.items()} for fold in range(contract['folds']['folds'])}}
    write_json(output/'results.json',{'status':'complete','models':models,
        'scope':'recoverable common-task decision bias; adds no detector information; postfit scores do not replace frozen raw-model scientific comparisons',
        'development_only':True,'Cal_A_Cut_B_Confirm_payload_read':False})
    print(json.dumps({'status':'complete','models':len(models)}),flush=True)


if __name__=='__main__':main()
