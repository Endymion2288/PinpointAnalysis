"""Formal CPU checkpoint readback and paired validation on completed S4 fits."""
import argparse
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import time

import pandas as pd
import pyarrow as pa
import torch

from analysis.flavour.inventory import write_json
from analysis.flavour.s2.readback import readback
from analysis.flavour.s4.evaluate import evaluate


def checkpoint_configuration(expected, recorded):
    """Resolve only the legacy task-dispatch default; retain strict fit settings."""
    def explicit(config):
        return {**config, 'program':config.get('program',
            'train' if config.get('objective')=='classification' else 'pretrain')}
    if explicit(expected)!=explicit(recorded):
        raise ValueError(expected['name']+': frozen campaign/checkpoint configuration mismatch')
    # readback still checks equality of manifest, last and best configurations.
    return recorded


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--task',type=int,required=True)
    args=parser.parse_args();config=json.loads(args.campaign.read_text())['experiments'][args.task]
    output=args.root/'experiments'/config['name'];output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();torch.set_num_threads(1);pa.set_cpu_count(1);pa.set_io_thread_count(1)
    manifest={'status':'running','config':config,'GPU':None,'utc':datetime.now(timezone.utc).isoformat(),
        'code_commit':os.environ.get('PINPOINT_CODE_COMMIT') or subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    write_json(output/'manifest.json',manifest)
    campaigns=[Path(p) for p in config['campaigns']]
    configurations={c['name']:c for p in campaigns for c in json.loads(p.read_text())['experiments']}
    if config.get('wait_for_completion_seconds',0):
        deadline=time.monotonic()+config['wait_for_completion_seconds']
        while True:
            waiting=[]
            for name,c in configurations.items():
                if c.get('program','train' if c.get('objective')=='classification' else 'pretrain')!='train':continue
                path=args.root/'experiments'/name/'manifest.json'
                status=json.loads(path.read_text()).get('status') if path.exists() else None
                if status in {'failed','internal_health_failed','learning_review_required'}:
                    raise RuntimeError(name+': upstream training failed; evaluation stopped')
                if status!='complete':waiting.append(name)
            if not waiting:break
            if time.monotonic()>deadline:raise TimeoutError('bounded wait for frozen classification matrix expired')
            manifest.update(status='waiting_for_frozen_fits',pending_train=waiting)
            write_json(output/'manifest.json',manifest)
            time.sleep(15)
    catalog=pd.read_parquet(args.root/'catalog.parquet')
    validation=catalog[catalog.split.eq('validation')];verified=[];pending=[];health=[]
    for name,c in configurations.items():
        if c.get('program','train' if c.get('objective')=='classification' else 'pretrain')!='train':continue
        directory=args.root/'experiments'/name;path=directory/'manifest.json'
        if not path.exists() or json.loads(path.read_text()).get('status')!='complete':
            pending.append(name);continue
        m=json.loads(path.read_text())
        frozen=checkpoint_configuration(c,m['config'])
        report=readback(args.root,frozen,{'GPU':m['gpu_name'],'torch':'2.7.1'},validation)
        if config.get('inspect_best_health',False):
            from analysis.flavour.s4.health import inspect
            diagnostic=inspect(args.root,name,catalog)
            if not diagnostic['passed']:
                raise RuntimeError(name+': selected checkpoint FP32 health failed; review model and training')
        rows=json.loads((directory/'training_health.json').read_text())['records']
        if not rows or not all(r['healthy'] for r in rows):raise RuntimeError(name+': internal training health failed')
        if not all(r['gradient_squared_norms'].get('input',0)>0 for r in rows if r.get('informative_nodes',0)>0):
            raise RuntimeError(name+': informative input gradient absent')
        activation={}
        for key in ['head_input_norm','head.0','head.1','head.2']:
            values=[v for r in rows for v in r['activations'].get(key,[])]
            if values:activation[key]={'RMS_range':[min(x['RMS'] for x in values),max(x['RMS'] for x in values)],
                'maximum_fraction_below_minus5':max(x['fraction_below_minus5'] for x in values),
                'maximum_zero_fraction':max(x['zero_fraction'] for x in values)}
        maximum_norm=max(math.sqrt(sum(r['gradient_squared_norms'].values())) for r in rows)
        limit=c.get('maximum_preclip_gradient_norm',10000.)
        health.append({'name':name,'samples':len(rows),'all_healthy':maximum_norm<=limit,'activations':activation,
            'maximum_preclip_gradient_norm':maximum_norm,'gradient_review_limit':limit,
            'gradient_review_pass':maximum_norm<=limit,
            'minimum_input_gradient_squared_norm':min(r['gradient_squared_norms'].get('input',0) for r in rows if r.get('informative_nodes',0)>0),
            'checkpoint_epoch':report['best_epoch']})
        verified.append(name)
    write_json(output/'training_health_summary.json',{'models':health,'samples':sum(r['samples'] for r in health),
        'scope':'sampled activation and gradient evidence including extreme finite gradients; checkpoint readback checked separately'})
    evaluate(SimpleNamespace(root=args.root,campaign=campaigns,comparisons=Path(config['comparisons']),
        output=Path(config['evaluation_output']),bootstrap=config.get('bootstrap',400),strict=config.get('strict',False)))
    if config.get('plot_training_health',False):
        from analysis.flavour.s4.plots import classification_plots
        classification_plots(args.root,Path(config['evaluation_output'])/'summary.json',Path(config['evaluation_output']))
    manifest.update(status='complete',seconds=time.perf_counter()-started,verified_train=verified,pending_train=pending)
    write_json(output/'manifest.json',manifest)


if __name__=='__main__':main()
