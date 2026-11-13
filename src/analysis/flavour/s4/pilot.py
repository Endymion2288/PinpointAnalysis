"""Real-event classification/reconstruction gates, then finite train-corpus pilots."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import torch

from analysis.flavour.inventory import write_json
from analysis.flavour.s3.pilot_repair import optimizer_for
from analysis.flavour.s4.dataset import PretrainDataset,collate_pretrain
from analysis.flavour.s4.models import PretrainingModel,objective_loss
from analysis.flavour.s4.pretrain import to_device,internal_health,run as pretrain_run


def reconstruction_gate(args,config):
    torch.set_num_threads(1);pa.set_cpu_count(1);pa.set_io_thread_count(1)
    torch.manual_seed(config.get('seed',42))
    dataset=PretrainDataset(args.root,'train',config)
    catalog=pd.read_parquet(args.root/'catalog.parquet')
    meta=dataset.metadata[['run','chunk','row']].merge(catalog,on=['run','chunk','row'],how='left',validate='one_to_one')
    active=(meta.pixel_channels+meta.bar_channels).to_numpy()
    rng=np.random.default_rng(42)
    eligible=np.flatnonzero((active>=50)&(active<=3000))
    ids=rng.choice(eligible,32,replace=False)
    items=[dataset[int(i)] for i in ids]
    # Hold masks/queries fixed while fitting 32 real train events. This measures
    # actual optimization capacity; its scores are never downstream validation.
    device=torch.device(args.device)
    model=PretrainingModel(config).to(device);optimizer=optimizer_for(model,config)
    bf16=(device.type=='cuda' and torch.cuda.is_bf16_supported(including_emulation=False)
        and not config.get('fp32_prototype',False))
    output=args.root/'ssl_gate'/config['name'];output.mkdir(parents=True,exist_ok=True)
    records,health=[],[];started=time.perf_counter()
    for step in range(config.get('ssl_gate_steps',256)+1):
        if step%32==0 or step==config.get('ssl_gate_steps',256):
            model.eval();losses=[];positive_counts=[]
            with torch.no_grad():
                for left in range(0,32,8):
                    batch=to_device(collate_pretrain(items[left:left+8]),device)
                    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=bf16):
                        result=model(batch);loss,parts=objective_loss(result,batch,config)
                    losses.append(float(parts['reconstruction']));positive_counts.append(int(batch['reconstruction'][...,0].sum()))
            row={'step':step,'reconstruction_loss':float(np.mean(losses)),'positive_queries':sum(positive_counts),
                'seconds':time.perf_counter()-started}
            records.append(row);write_json(output/'progress.json',{'records':records,'health':health});print(json.dumps(row),flush=True)
        if step==config.get('ssl_gate_steps',256):break
        model.train();take=rng.choice(32,8,replace=False)
        batch=to_device(collate_pretrain([items[int(i)] for i in take]),device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=bf16):
            result=model(batch);loss,parts=objective_loss(result,batch,config)
        loss.backward()
        if step%32==0:
            health.append(internal_health(model,result,batch,step+1))
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();model.update_teacher(.996)
    # Stress the largest original TRAIN event through SSL in both precisions.
    workload=meta[['pixel_channels','bar_channels','tracker_channels']].sum(axis=1).to_numpy()
    maxindex=int(workload.argmax());item=dataset[maxindex]
    stress=[]
    for use_bf16 in ([False,True] if bf16 else [False]):
        batch=to_device(collate_pretrain([item]),device)
        model.train();model.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=use_bf16):
            result=model(batch);loss,_=objective_loss(result,batch,config)
        loss.backward()
        gradients=[p.grad for p in model.parameters() if p.grad is not None]
        stress.append({'raw_channels':int(workload[maxindex]),'visible_nodes':len(item['nodes']),
            'precision':'BF16' if use_bf16 else 'FP32','loss':float(loss.detach()),
            'finite':bool(torch.isfinite(loss)) and all(bool(torch.isfinite(g).all()) for g in gradients),
            'input_gradient_squared_norm':float(model.encoder.input[0].weight.grad.square().sum())})
    checks={'reconstruction_loss_reduction_at_least_10pct':records[-1]['reconstruction_loss']<.9*records[0]['reconstruction_loss'],
        'positive_and_negative_geometry_queries':records[0]['positive_queries']>0 and records[0]['positive_queries']<32*config.get('queries_per_kind',64)*3,
        'all_sampled_internal_health_pass':all(r['healthy'] for r in health),
        'maximum_occupancy_all_precisions_finite':all(r['finite'] and r['input_gradient_squared_norm']>0 for r in stress)}
    report={'config':config,'checks':checks,'passed':all(checks.values()),'records':records,'health':health,'stress':stress,
        'seconds':time.perf_counter()-started,'GPU':torch.cuda.get_device_name() if device.type=='cuda' else None,
        'scope':'32 fixed-mask TRAIN events for optimization gate; maximum train event stress; not generalization'}
    write_json(output/'gate.json',report);dataset.native.close()
    if not report['passed']:raise RuntimeError('SSL real-event optimization/stress gate failed; inspect before training')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--task',type=int,required=True)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args();config=json.loads(args.campaign.read_text())['experiments'][args.task]
    # Submit to both queues while CPU assembly finishes. Allocated waiting time
    # remains part of the backend accounting and is bounded to ten minutes.
    ready_deadline=time.monotonic()+600
    while not (args.root/'data_summary.json').exists():
        if time.monotonic()>ready_deadline:raise TimeoutError('S4 corpus assembly did not finish within allocated readiness window')
        time.sleep(5)
    if not json.loads((args.root/'data_summary.json').read_text())['passed']:
        raise RuntimeError('S4 data assembly failed')
    if config.get('objective')=='backbone_health':
        command=[sys.executable,'-m','analysis.flavour.s3.pilot_repair','--root',str(args.root),
            '--campaign',str(args.campaign),'--task',str(args.task),'--device',args.device]
        subprocess.run(command,check=True)
        command[2]='analysis.flavour.s3.profile'
        subprocess.run(command,check=True)
    else:
        gate_path=args.root/'ssl_gate'/config['name']/'gate.json'
        prior=json.loads(gate_path.read_text()) if gate_path.exists() else None
        if prior is None or not prior.get('passed') or prior['config']!=config:
            reconstruction_gate(args,config)
        pretrain_run(args)


if __name__=='__main__':main()
