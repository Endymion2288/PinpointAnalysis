"""Check whether partial bar Rel uses event content beyond geometry priors."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import DataLoader,Subset

from analysis.flavour.inventory import write_json
from analysis.flavour.s2.train import init_worker
from analysis.flavour.s3.metrics import bootstrap_weights
from analysis.flavour.s4.dataset import PretrainDataset,collate_pretrain
from analysis.flavour.s4.models import PretrainingModel


def cross_entropy(target,probability):
    return -(target*np.log(np.maximum(probability,1e-12))).sum(1)


def inspect(root,name,events=512,replicates=400):
    output=root/'experiments'/name
    config=json.loads((output/'manifest.json').read_text())['config']
    if config['objective']!='mae_rel':raise ValueError('partial bar Rel checkpoint required')
    data=PretrainDataset(root,'validation',{**config,'compile_local':False})
    n=min(events,len(data))
    loader=DataLoader(Subset(data,range(n)),batch_size=16,collate_fn=collate_pretrain,num_workers=2,
        multiprocessing_context='spawn',worker_init_fn=init_worker,prefetch_factor=1)
    model=PretrainingModel({**config,'compile_local':False}).eval()
    state=torch.load(output/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(state['model'],strict=True)
    captured=[]
    handle=model.decoder.register_forward_pre_hook(lambda module,inputs:captured.append(inputs))
    predictions=[];targets=[];shuffled=[];positions=[];event_ids=[];offset=0
    with torch.no_grad():
        for batch in loader:
            captured.clear();result=model(batch)
            query,source,*mask=captured[-1]
            if query is not batch['rel_query']:raise RuntimeError('last decoder call is not Rel')
            changed=model.decoder(query,source.roll(1,0),mask[0].roll(1,0) if mask and mask[0] is not None else None)[...,4:8]
            valid=batch['rel_valid'];count=len(valid)
            predictions.append(result['rel'][valid].softmax(-1).numpy())
            targets.append(batch['rel_target'][valid].numpy());shuffled.append(changed[valid].softmax(-1).numpy())
            positions.append(query[valid][:,:3].numpy())
            ids=torch.arange(offset,offset+count)[:,None].expand_as(valid)[valid]
            event_ids.append(ids.numpy());offset+=count
    handle.remove();captured.clear();del loader
    probability=np.concatenate(predictions);target=np.concatenate(targets)
    changed=np.concatenate(shuffled);position=np.concatenate(positions);event=np.concatenate(event_ids)
    geometry,group=np.unique(np.round(position*np.array([210,210,3000]),3),axis=0,return_inverse=True)
    prior=target.mean(0);geometry_prior=np.zeros((len(geometry),4))
    np.add.at(geometry_prior,group,target);geometry_prior/=np.bincount(group)[:,None]
    losses={'trained':cross_entropy(target,probability),
        'event_permuted':cross_entropy(target,changed),'oracle_constant':cross_entropy(target,np.broadcast_to(prior,target.shape)),
        'oracle_channel_geometry':cross_entropy(target,geometry_prior[group]),
        'target_entropy':cross_entropy(target,target)}
    multiplicities,blocks=bootstrap_weights(data.metadata.iloc[:n].reset_index(drop=True),replicates)
    channel_counts=np.bincount(event,minlength=n)
    values={};samples={}
    for key,loss in losses.items():
        sums=np.bincount(event,weights=loss,minlength=n)
        values[key]=float(loss.mean());samples[key]=(multiplicities@sums)/(multiplicities@channel_counts)
    contrasts={}
    for key in ['event_permuted','oracle_constant','oracle_channel_geometry']:
        difference=samples[key]-samples['trained']
        contrasts[key+'_minus_trained']={'delta':values[key]-values['trained'],
            'CI95':np.quantile(difference,[.025,.975]).tolist()}
    report={'utc':datetime.now(timezone.utc).isoformat(),'name':name,'checkpoint_epoch':state['epoch'],
        'precision':'FP32 CPU, same weights and fixed original train-inner events/queries; no refitting',
        'probe_events':n,'valid_channels':len(target),'hardware_channels':len(geometry),
        'cross_entropy':values,'paired_contrasts':contrasts,'resampling_blocks':blocks,
        'bootstrap_replicates':replicates,'scope':'train-inner semantic diagnostic; no independent classification claim',
        'constant_scope':'oracle constant and per-hardware-channel target means fitted on the same probe; descriptive lower bounds for those prior families',
        'permutation_scope':'cyclic event permutation of the complete decoder source and its mask within each fixed batch, holding Rel queries and targets fixed',
        'interval_scope':'paired accepted-family connected-chunk bootstrap; fixed fitted priors and trained checkpoint'}
    write_json(output/'bar_semantic_diagnostic.json',report)
    data.native.close();print(json.dumps(report),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--task',type=int,required=True)
    args=parser.parse_args();config=json.loads(args.campaign.read_text())['experiments'][args.task]
    output=args.root/'experiments'/config['name'];output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);pa.set_cpu_count(1);pa.set_io_thread_count(1);started=time.perf_counter()
    manifest={'status':'running','config':config,'GPU':None,'utc':datetime.now(timezone.utc).isoformat(),
        'code_commit':os.environ.get('PINPOINT_CODE_COMMIT') or subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    write_json(output/'manifest.json',manifest)
    rows=[inspect(args.root,name,config.get('probe_events',512),config.get('bootstrap',400)) for name in config['checkpoints']]
    write_json(output/'semantics.json',{'experiments':rows})
    manifest.update(status='complete',seconds=time.perf_counter()-started);write_json(output/'manifest.json',manifest)


if __name__=='__main__':main()
