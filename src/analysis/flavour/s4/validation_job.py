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


def verify_probe(directory,config,validation):
    """Check numerical solver evidence and the exact frozen prediction catalog."""
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest.get('status')!='complete' or manifest.get('config')!=config:
        raise ValueError(config['name']+': complete frozen probe configuration required')
    optimization=json.loads((directory/'probe_optimization.json').read_text())
    for branch in config.get('probe_branches',['full','global']):
        audit=optimization[branch+'_linear']
        if (not audit.get('passed') or audit.get('convergence_warnings') or
            not audit.get('finite_coefficients') or audit['fit_events']!=config['label_budget']):
            raise ValueError(config['name']+': converged fit-only readout required')
        if config.get('probe_require_stationarity') and (not audit.get('stationarity_passed') or
            audit.get('gradient_infinity_norm',float('inf'))>config['probe_linear_tolerance']+1e-10):
            raise ValueError(config['name']+': measured convex-objective stationarity required')
        frame=pd.read_parquet(directory/(branch+'_linear_predictions.parquet'))
        keys=['run','chunk','row','label']
        if not frame[keys].sort_values(keys).reset_index(drop=True).equals(
                validation[keys].sort_values(keys).reset_index(drop=True)):
            raise ValueError(config['name']+': probe predictions differ from fixed validation catalog')
    return {'name':config['name'],'branches':optimization,'fit_events':manifest['fit_events'],
        'validation_events':manifest['probe_events'],'all_optimizers_converged':True}


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
                if c.get('program','train' if c.get('objective')=='classification' else 'pretrain') not in {'train','probes'}:continue
                path=args.root/'experiments'/name/'manifest.json'
                status=json.loads(path.read_text()).get('status') if path.exists() else None
                if status in {'failed','internal_health_failed','learning_review_required','optimization_review_required'}:
                    raise RuntimeError(name+': upstream fit requires review; evaluation stopped')
                if status!='complete':waiting.append(name)
            if not waiting:break
            if time.monotonic()>deadline:raise TimeoutError('bounded wait for frozen fits expired')
            manifest.update(status='waiting_for_frozen_fits',pending_fits=waiting)
            write_json(output/'manifest.json',manifest)
            time.sleep(15)
    catalog=pd.read_parquet(args.root/'catalog.parquet')
    validation=catalog[catalog.split.eq('validation')];verified=[];pending=[];health=[];probes=[]
    for name,c in configurations.items():
        if c.get('program')=='probes':
            directory=args.root/'experiments'/name;path=directory/'manifest.json'
            if not path.exists() or json.loads(path.read_text()).get('status')!='complete':
                pending.append(name);continue
            probes.append(verify_probe(directory,c,validation));continue
        if c.get('program','train' if c.get('objective')=='classification' else 'pretrain')!='train':continue
        directory=args.root/'experiments'/name;path=directory/'manifest.json'
        if not path.exists() or json.loads(path.read_text()).get('status')!='complete':
            pending.append(name);continue
        m=json.loads(path.read_text())
        frozen=checkpoint_configuration(c,m['config'])
        report=readback(args.root,frozen,{'GPU':m['gpu_name'],'torch':'2.7.1'},validation)
        if c.get('transfer_recipe'):
            from analysis.flavour.s4.parameter_audit import inspect as inspect_parameters
            if not inspect_parameters(args.root,name)['passed']:
                raise RuntimeError(name+': encoder parameter updates absent; review actual training')
        if config.get('inspect_best_health',False):
            from analysis.flavour.s4.health import inspect
            diagnostic=inspect(args.root,name,catalog)
            if not diagnostic['passed']:
                raise RuntimeError(name+': selected checkpoint FP32 health failed; review model and training')
        rows=json.loads((directory/'training_health.json').read_text())['records']
        if not rows or not all(r['healthy'] for r in rows):raise RuntimeError(name+': internal training health failed')
        encoder_rows=[r for r in rows if not r.get('intentional_encoder_freeze',False)]
        frozen_rows=[r for r in rows if r.get('intentional_encoder_freeze',False)]
        for branch in ['input','local']:
            if not all(r['gradient_squared_norms'].get(branch,0)>0 for r in encoder_rows if r.get('informative_nodes',0)>0):
                raise RuntimeError(name+': informative '+branch+' gradient absent')
        if frozen_rows and (not c.get('linear_probe_passes') or not all(
            r['gradient_squared_norms'].get('linear_readout',0)>0 and all(
                value==0 for key,value in r['gradient_squared_norms'].items() if key!='linear_readout') for r in frozen_rows)):
            raise RuntimeError(name+': LP phase gradient policy failed')
        if not encoder_rows:
            raise RuntimeError(name+': no full fine-tuning health evidence')
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
            'minimum_input_gradient_squared_norm':min(r['gradient_squared_norms'].get('input',0) for r in encoder_rows if r.get('informative_nodes',0)>0),
            'minimum_local_gradient_squared_norm':min(r['gradient_squared_norms'].get('local',0) for r in encoder_rows if r.get('informative_nodes',0)>0),
            'parameter_updates_checked':bool(c.get('transfer_recipe')),
            'intentional_LP_samples':len(frozen_rows),
            'checkpoint_epoch':report['best_epoch']})
        verified.append(name)
    write_json(output/'training_health_summary.json',{'models':health,'samples':sum(r['samples'] for r in health),
        'scope':'sampled activation and gradient evidence including extreme finite gradients; checkpoint readback checked separately'})
    write_json(output/'probe_optimization_summary.json',{'models':probes,
        'scope':'fixed-fit-label standardization, solver convergence and exact validation keys; no hyperparameter selection'})
    evaluate(SimpleNamespace(root=args.root,campaign=campaigns,comparisons=Path(config['comparisons']),
        output=Path(config['evaluation_output']),bootstrap=config.get('bootstrap',400),strict=config.get('strict',False)))
    if config.get('plot_training_health',False):
        from analysis.flavour.s4.plots import classification_plots
        classification_plots(args.root,Path(config['evaluation_output'])/'summary.json',Path(config['evaluation_output']))
    manifest.update(status='complete',seconds=time.perf_counter()-started,verified_train=verified,
        pending_train=pending,pending_fits=pending)
    write_json(output/'manifest.json',manifest)


if __name__=='__main__':main()
