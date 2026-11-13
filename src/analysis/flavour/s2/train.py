"""Train paired S2 baselines on frozen accepted train/validation subsets only."""

import argparse
import copy
import errno
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import time
import uuid

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


def atomic_output(path, writer):
    """Replay full writes after transient EOS errors, then publish atomically.

    A failed fsync left a truncated Torch archive in S3. Repeating only fsync
    cannot replay lost bytes, so every retry creates a fresh temporary inode.
    """
    transient = {errno.EPROTO, errno.EIO, errno.ETIMEDOUT, errno.ECONNRESET, errno.EAGAIN}
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    for attempt in range(4):
        try:
            if temporary.exists():
                temporary.unlink()
            with temporary.open('wb') as stream:
                writer(stream)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
            return
        except OSError as error:
            if error.errno not in transient or attempt == 3:
                raise
            print(json.dumps({'io_full_write_retry': str(path), 'errno': error.errno,
                              'attempt': attempt + 1}), flush=True)
            time.sleep([.1, .3, 1.][attempt])


def save_checkpoint(path,state):
    atomic_output(path, lambda stream: torch.save(state, stream))


def save_parquet(frame,path):
    """Finish EOS writes before publishing; each replacement gets a new inode."""
    atomic_output(path, lambda stream: frame.to_parquet(stream,index=False,compression=COMPRESSION))


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


def run(args, dataset_class=None, model_factory=None, collate_fn=None, manifest_extra=None,
        health_factory=None, optimizer_factory=None):
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
    dataset_class=dataset_class or SpatialDataset
    train=dataset_class(args.root,'train',config)
    val=dataset_class(args.root,'validation',config)
    if not len(train) or not len(val):
        raise ValueError('train/validation subset empty')
    train_sampler=SpatialBatchSampler(train,config.get('batch_size',4),seed,steps=config.get('steps_per_epoch'))
    val_sampler=SpatialBatchSampler(val,config.get('batch_size',4),seed,shuffle=False)
    workers=config.get('workers',2)
    options={'num_workers':workers,'collate_fn':collate_fn or collate_spatial,'pin_memory':device.type=='cuda',
             'generator':torch.Generator().manual_seed(seed)}
    if workers:
        options.update(multiprocessing_context='spawn',worker_init_fn=init_worker,prefetch_factor=1,persistent_workers=True)
    train_loader=DataLoader(train,batch_sampler=train_sampler,**options)
    val_loader=DataLoader(val,batch_sampler=val_sampler,**options)
    model=(model_factory(config) if model_factory else PointClassifier(config['architecture'],dim=config.get('dim',64),k=config.get('k',16),
                          pair_bias=config.get('pair_bias',True),dynamic=config.get('dynamic',True),
                          cross_plane=config.get('cross_plane',True))).to(device)
    optimizer=(optimizer_factory(model,config) if optimizer_factory else
               torch.optim.AdamW(model.parameters(),lr=config.get('learning_rate',.0005),weight_decay=.01))
    epochs=config.get('epochs',12)
    step_schedule=config.get('schedule')=='warmup_cosine_steps'
    if step_schedule:
        total_steps=epochs*len(train_sampler)
        warmup=config.get('warmup_steps',500)
        minimum=config.get('minimum_learning_rate',1e-5)/config.get('learning_rate',.0005)
        def factor(step):
            if step<warmup:
                return (step+1)/warmup
            fraction=min(1.,(step-warmup)/max(1,total_steps-warmup))
            return minimum+(1-minimum)*.5*(1+math.cos(math.pi*fraction))
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,factor)
    else:
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=1e-5)
    health=health_factory(model) if health_factory else None
    health_records=[]
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and dtype==torch.float16)
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats()
    counts=np.bincount(train.metadata.label,minlength=4)
    weights=torch.tensor(counts.sum()/(4*counts),dtype=torch.float32,device=device)
    manifest={'utc':datetime.now(timezone.utc).isoformat(),'config':config,'status':'running',
        'code_commit':os.environ.get('PINPOINT_CODE_COMMIT') or subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
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
    manifest.update(manifest_extra or {})
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
        health_records=state.get('health_records',[])
        if health is not None and health_records:
            health.dead_samples=health_records[-1]['consecutive_exact_zero_encoder_gradient_samples']
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
        save_parquet(predictions,output/'validation_predictions.parquet')
        write_json(output/'metrics.json',metrics(val.metadata.label.to_numpy(),best_probabilities))
        manifest['resumed_from_epoch']=start_epoch
        manifest['prior_gpu_name']=previous.get('gpu_name')
    write_json(output/'manifest.json',manifest)
    for epoch in range(start_epoch,epochs):
        epoch_start=time.perf_counter()
        train_sampler.epoch=epoch+config.get('sampler_epoch_offset',0)
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
            diagnostic=health is not None and (optimizer_steps%config.get('health_every_steps',100)==0)
            if diagnostic:
                health.start(batch)
            t=time.perf_counter()
            with torch.autocast(device_type=device.type,dtype=dtype,enabled=device.type=='cuda'):
                logits=model(batch)
                loss=torch.nn.functional.cross_entropy(logits.float(),label,weight=weights)
            if not torch.isfinite(loss):
                raise ValueError(f'nonfinite loss at epoch {epoch}, all observations retained')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if diagnostic:
                health_records.append(health.finish(logits,optimizer_steps+1))
                write_json(output/'training_health.json',{'records':health_records})
                if not health_records[-1]['healthy']:
                    manifest.update(status='internal_health_failed',failure=health_records[-1])
                    write_json(output/'manifest.json',manifest)
                    raise RuntimeError('internal training health failed; inspect architecture and optimizer before resuming')
            torch.nn.utils.clip_grad_norm_(model.parameters(),config.get('clip_grad_norm',5.),error_if_nonfinite=True)
            scale_before=scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale()<scale_before:
                amp_skipped_steps+=1
            else:
                optimizer_steps+=1
                if step_schedule:
                    scheduler.step()
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
        if health is not None:
            from analysis.flavour.s3.metrics import prediction_diagnostics
            measured['prediction_diagnostics']=prediction_diagnostics(p)
            # All-empty inputs may be constant; the complete validation set
            # includes many active responses, so a single vector is a defect.
            if measured['prediction_diagnostics']['all_events_same_probability_vector']:
                manifest.update(status='internal_health_failed',failure='constant full-validation prediction')
                write_json(output/'manifest.json',manifest)
                write_json(output/'failed_validation_health.json',measured)
                raise RuntimeError('constant complete validation predictions; architecture/optimization review required')
        ep=pd.DataFrame({'row':np.arange(len(y),dtype='int32'),'epoch':np.full(len(y),epoch+1,dtype='int16')})
        for c in range(4):
            ep[f'p{c}']=p[:,c]
        epoch_predictions.append(ep)
        save_parquet(pd.concat(epoch_predictions,ignore_index=True),output/'epoch_predictions.parquet')
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
            save_parquet(predictions,output/'validation_predictions.parquet')
            write_json(output/'metrics.json',measured)
        if not step_schedule:
            scheduler.step()
        checkpoint={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'config':config,
            'epoch':epoch+1,'seed':seed,'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
            'optimizer_steps':optimizer_steps,'amp_skipped_steps':amp_skipped_steps,
            'history':history,'timing':timing.copy(),'best_NLL':best,'best_epoch':best_epoch,
            'health_records':health_records,
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
    if health is not None:
        health.close()
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
