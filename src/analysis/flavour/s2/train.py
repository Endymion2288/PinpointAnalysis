"""Train paired S2 baselines on frozen accepted train/validation subsets only."""

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import resource
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss, roc_auc_score
import torch
from torch.utils.data import DataLoader

from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.s2.dataset import SpatialDataset, SpatialBatchSampler, collate_spatial
from analysis.flavour.s2.models import PointClassifier


def init_worker(worker):
    torch.set_num_threads(1)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)


def save_checkpoint(path,state):
    temporary=path.with_suffix('.tmp')
    torch.save(state,temporary)
    temporary.replace(path)


def metrics(y,p):
    per_class={}
    for c in range(4):
        flag=y==c
        per_class[str(c)]={'events':int(flag.sum()),
            'ROC_AUC':float(roc_auc_score(flag,p[:,c])) if flag.any() and (~flag).any() else None,
            'PR_AUC':float(average_precision_score(flag,p[:,c])) if flag.any() else None}
    return {'events':len(y),'balanced_accuracy':float(balanced_accuracy_score(y,p.argmax(1))),
        'macro_F1':float(f1_score(y,p.argmax(1),labels=[0,1,2,3],average='macro',zero_division=0)),
        'NLL':float(log_loss(y,p,labels=[0,1,2,3])),
        'Brier':float(np.mean(np.sum((p-np.eye(4)[y])**2,axis=1))),
        'confusion_counts':confusion_matrix(y,p.argmax(1),labels=[0,1,2,3]).tolist(),'classwise':per_class}


def evaluate(model,loader,device,amp_dtype):
    model.eval()
    rows,probabilities,labels=[],[],[]
    with torch.no_grad():
        for batch in loader:
            ids=batch.pop('index')
            y=batch.pop('label')
            batch={k:v.to(device) for k,v in batch.items()}
            with torch.autocast(device_type=device.type,dtype=amp_dtype,enabled=device.type=='cuda'):
                p=model(batch).float().softmax(-1)
            probabilities.append(p.cpu().numpy())
            labels.append(y.numpy())
            rows.append(ids.numpy())
    order=np.argsort(np.concatenate(rows))
    return np.concatenate(labels)[order],np.concatenate(probabilities)[order]


def run(args):
    config=json.loads(args.campaign.read_text())['experiments'][args.task]
    experiment=config['name']
    output=args.root/'experiments'/experiment
    output.mkdir(parents=True,exist_ok=True)
    previous=json.loads((output/'manifest.json').read_text()) if (output/'manifest.json').exists() else {}
    if previous.get('status')=='complete':
        if previous['config']!=config:
            raise ValueError('completed experiment name reused for a different configuration')
        print(f'{experiment} already complete; no duplicate training',flush=True)
        return
    seed=config.get('seed',42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    device=torch.device(args.device)
    dtype=torch.bfloat16 if device.type=='cuda' and torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    started=time.perf_counter()
    train=SpatialDataset(args.root,'train',config)
    val=SpatialDataset(args.root,'validation',config)
    if not len(train) or not len(val):
        raise ValueError('train/validation subset empty')
    train_sampler=SpatialBatchSampler(train,config.get('batch_size',4),seed,steps=config.get('steps_per_epoch'))
    val_sampler=SpatialBatchSampler(val,config.get('batch_size',4),seed,shuffle=False)
    workers=config.get('workers',2)
    options={'num_workers':workers,'collate_fn':collate_spatial,'pin_memory':device.type=='cuda',
             'generator':torch.Generator().manual_seed(seed)}
    if workers:
        options.update(multiprocessing_context='spawn',worker_init_fn=init_worker,prefetch_factor=1,persistent_workers=True)
    train_loader=DataLoader(train,batch_sampler=train_sampler,**options)
    val_loader=DataLoader(val,batch_sampler=val_sampler,**options)
    model=PointClassifier(config['architecture'],dim=config.get('dim',64),k=config.get('k',16),
                          pair_bias=config.get('pair_bias',True),dynamic=config.get('dynamic',True),
                          cross_plane=config.get('cross_plane',True)).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.get('learning_rate',.0005),weight_decay=.01)
    epochs=config.get('epochs',12)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=1e-5)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and dtype==torch.float16)
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats()
    counts=np.bincount(train.metadata.label,minlength=4)
    weights=torch.tensor(counts.sum()/(4*counts),dtype=torch.float32,device=device)
    manifest={'utc':datetime.now(timezone.utc).isoformat(),'config':config,'status':'running',
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'campaign':str(args.campaign),'root':str(args.root),'response_level':'user_accepted_detector_proxy',
        'split_status':'accepted_signature_grouped_user_attested','train_events':len(train),'validation_events':len(val),
        'train_class_counts':counts.tolist(),'validation_class_counts':np.bincount(val.metadata.label,minlength=4).tolist(),
        'optimizer_steps_per_epoch':len(train_sampler),'repeated_training_domain':config.get('train_domain','all')!='all',
        'parameters':sum(p.numel() for p in model.parameters()),'device':str(device),
        'gpu_name':torch.cuda.get_device_name() if device.type=='cuda' else None,'torch':torch.__version__,
        'python':sys.version,'numpy':np.__version__,'pyarrow':pa.__version__,
        'mixed_precision':str(dtype) if device.type=='cuda' else None,'selection':'validation_NLL',
        'normalization':'fixed_physical_scales_no_truth','calibration_test_used':False,
        'local_blocks':'same_plane_4mm_then_canonical_chunks32_no_event_truncation',
        'schema_whitelist':'18_observation_features_plus_8_current_representation_counts_and_signal_sums',
        'response_summary_version':2,
        'cross_plane_message_passing':config.get('cross_plane',True),
        'load_seconds':time.perf_counter()-started}
    history=[]
    epoch_predictions=[]
    best=float('inf')
    best_epoch=0
    best_model=None
    best_probabilities=None
    optimizer_steps=0
    amp_skipped_steps=0
    timing=dict(data_wait=0.,transfer=0.,forward_backward=0.,validation=0.)
    start_epoch=0
    prior_seconds=0.
    if (output/'last.pt').exists():
        state=torch.load(output/'last.pt',map_location=device,weights_only=False)
        if state['config']!=config:
            raise ValueError('resume configuration differs from stored checkpoint')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        start_epoch=state['epoch']
        history=state['history']
        best,best_epoch=state['best_NLL'],state['best_epoch']
        best_model=state['best_model']
        best_probabilities=state['best_probabilities']
        timing=state['timing'].copy()
        prior_seconds=state['elapsed_seconds']
        optimizer_steps=state.get('optimizer_steps',start_epoch*len(train_sampler))
        amp_skipped_steps=state.get('amp_skipped_steps',0)
        random.setstate(state['random_state'])
        np.random.set_state(state['numpy_state'])
        torch.set_rng_state(state['torch_state'].cpu())
        if device.type=='cuda':
            torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_states']])
        ep=pd.read_parquet(output/'epoch_predictions.parquet')
        epoch_predictions=[ep[ep.epoch<=start_epoch]]
        # The atomic last checkpoint is authoritative if interruption happened
        # between writing best/predictions and committing the next epoch.
        save_checkpoint(output/'best.pt',{'model':best_model,'config':config,'epoch':best_epoch,'seed':seed})
        predictions=val.metadata.copy()
        for c in range(4):
            predictions[f'p{c}']=best_probabilities[:,c]
        predictions.to_parquet(output/'validation_predictions.parquet',index=False,compression=COMPRESSION)
        write_json(output/'metrics.json',metrics(val.metadata.label.to_numpy(),best_probabilities))
        manifest['resumed_from_epoch']=start_epoch
        manifest['prior_gpu_name']=previous.get('gpu_name')
    write_json(output/'manifest.json',manifest)
    for epoch in range(start_epoch,epochs):
        epoch_start=time.perf_counter()
        train_sampler.epoch=epoch
        model.train()
        loss_total,events=0.,0
        ready=time.perf_counter()
        for step,batch in enumerate(train_loader,1):
            timing['data_wait']+=time.perf_counter()-ready
            batch.pop('index')
            label=batch.pop('label').to(device)
            t=time.perf_counter()
            batch={k:v.to(device,non_blocking=True) for k,v in batch.items()}
            if device.type=='cuda':
                torch.cuda.synchronize()
            timing['transfer']+=time.perf_counter()-t
            optimizer.zero_grad(set_to_none=True)
            t=time.perf_counter()
            with torch.autocast(device_type=device.type,dtype=dtype,enabled=device.type=='cuda'):
                logits=model(batch)
                loss=torch.nn.functional.cross_entropy(logits.float(),label,weight=weights)
            if not torch.isfinite(loss):
                raise ValueError(f'nonfinite loss at epoch {epoch}, all observations retained')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            scale_before=scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale()<scale_before:
                amp_skipped_steps+=1
            else:
                optimizer_steps+=1
            if device.type=='cuda':
                torch.cuda.synchronize()
            timing['forward_backward']+=time.perf_counter()-t
            loss_total+=loss.item()*len(label)
            events+=len(label)
            if step%config.get('progress_every_steps',250)==0:
                progress={'experiment':experiment,'epoch':epoch+1,'step':step,'steps_per_epoch':len(train_sampler),
                    'exposures':events,'train_loss':loss_total/events,
                    'optimizer_steps':optimizer_steps,'amp_skipped_steps':amp_skipped_steps,
                    'events_per_second':events/(time.perf_counter()-epoch_start),
                    'seconds':time.perf_counter()-epoch_start,'timing_cumulative':timing.copy(),
                    'peak_gpu_mib':torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else 0.}
                write_json(output/'progress.json',progress)
                print(json.dumps(progress),flush=True)
            ready=time.perf_counter()
        train_seconds=time.perf_counter()-epoch_start
        t=time.perf_counter()
        y,p=evaluate(model,val_loader,device,dtype)
        timing['validation']+=time.perf_counter()-t
        measured=metrics(y,p)
        ep=pd.DataFrame({'row':np.arange(len(y),dtype='int32'),'epoch':np.full(len(y),epoch+1,dtype='int16')})
        for c in range(4):
            ep[f'p{c}']=p[:,c]
        epoch_predictions.append(ep)
        pd.concat(epoch_predictions,ignore_index=True).to_parquet(output/'epoch_predictions.parquet',
                        index=False,compression=COMPRESSION)
        score=measured['NLL']
        record={'epoch':epoch+1,'train_loss':loss_total/events,'train_seconds':train_seconds,
            'train_events_per_second':events/train_seconds,'validation':measured,
            'wall_seconds':prior_seconds+time.perf_counter()-started,
            'train_exposures':events,'optimizer_steps_cumulative':optimizer_steps,
            'attempted_optimizer_steps_cumulative':(epoch+1)*len(train_sampler),
            'amp_skipped_steps_cumulative':amp_skipped_steps,'timing_cumulative':timing.copy()}
        history.append(record)
        improved=score<best
        if score<best:
            best,best_epoch=score,epoch+1
            best_model=copy.deepcopy(model.state_dict())
            best_probabilities=p.copy()
            predictions=val.metadata.copy()
            for c in range(4):
                predictions[f'p{c}']=p[:,c]
            predictions.to_parquet(output/'validation_predictions.parquet',index=False,compression=COMPRESSION)
            write_json(output/'metrics.json',measured)
        scheduler.step()
        checkpoint={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'config':config,
            'epoch':epoch+1,'seed':seed,'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
            'optimizer_steps':optimizer_steps,'amp_skipped_steps':amp_skipped_steps,
            'history':history,'timing':timing.copy(),'best_NLL':best,'best_epoch':best_epoch,
            'best_model':best_model,'best_probabilities':best_probabilities,
            'elapsed_seconds':prior_seconds+time.perf_counter()-started,
            'random_state':random.getstate(),'numpy_state':np.random.get_state(),
            'torch_state':torch.get_rng_state(),
            'cuda_states':torch.cuda.get_rng_state_all() if device.type=='cuda' else []}
        if improved:
            save_checkpoint(output/'best.pt',{'model':best_model,'config':config,'epoch':best_epoch,'seed':seed})
        save_checkpoint(output/'last.pt',checkpoint)
        write_json(output/'history.json',history)
        print(json.dumps({'experiment':experiment,'epoch':epoch+1,'NLL':score,'tau_AP':measured['classwise']['2']['PR_AUC'],
                          'events_per_second':events/train_seconds}),flush=True)
    elapsed=prior_seconds+time.perf_counter()-started
    manifest.update(status='complete',best_epoch=best_epoch,best_NLL=best,seconds=elapsed,
        gpu_hours=elapsed/3600 if device.type=='cuda' else 0,
        peak_gpu_allocated_mib=max(previous.get('peak_gpu_allocated_mib',0),torch.cuda.max_memory_allocated()/2**20) if device.type=='cuda' else 0,
        parent_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        optimizer_steps=optimizer_steps,amp_skipped_steps=amp_skipped_steps,
        timing_seconds=timing)
    write_json(output/'manifest.json',manifest)
    train.native.close()
    val.native.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--task',type=int,required=True)
    parser.add_argument('--device',default='cuda')
    run(parser.parse_args())


if __name__=='__main__':
    main()
