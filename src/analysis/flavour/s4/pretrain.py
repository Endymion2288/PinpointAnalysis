"""Resumable train-only S4 pretraining with reconstruction and feature readback."""
import argparse
import copy
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
from sklearn.metrics import average_precision_score,roc_auc_score
import torch
from torch.utils.data import DataLoader,Subset

from analysis.flavour.inventory import write_json
from analysis.flavour.s2.dataset import SpatialBatchSampler
from analysis.flavour.s2.train import init_worker,save_checkpoint,save_parquet
from analysis.flavour.s3.pilot_repair import optimizer_for
from analysis.flavour.s4.dataset import PretrainDataset,collate_pretrain
from analysis.flavour.s4.models import PretrainingModel,objective_loss


def to_device(batch,device):
    return {key:to_device(value,device) if isinstance(value,dict) else value.to(device,non_blocking=True)
        for key,value in batch.items()}


def feature_statistics(x):
    x=x.detach().float()
    centered=x-x.mean(0,keepdim=True)
    std=centered.square().mean(0).sqrt()
    singular=torch.linalg.svdvals(centered)
    eigen=singular.square()
    probability=eigen/eigen.sum().clamp_min(1e-30)
    rank=float(torch.exp(-(probability*probability.clamp_min(1e-30).log()).sum())) if eigen.sum()>0 else 0.
    return {'event_count':len(x),'feature_dim':x.shape[-1],
        'RMS':float(x.square().mean().sqrt()),'between_event_RMS':float(std.square().mean().sqrt()),
        'dimension_std_median':float(std.median()),'fraction_dimensions_std_below_1e_minus5':float((std<1e-5).float().mean()),
        'centered_effective_rank':rank,'all_finite':bool(torch.isfinite(x).all())}


def internal_health(model,output,batch,step):
    gradients={}
    for name,parameter in model.named_parameters():
        if parameter.grad is not None:
            group='.'.join(name.split('.')[:2]) if name.startswith('encoder.') else name.split('.')[0]
            gradients.setdefault(group,[]).append(parameter.grad.detach().float().square().sum())
    gradients={key:float(torch.stack(values).sum()) for key,values in gradients.items()}
    local=sum(gradients.get('encoder.'+key,0.) for key in ['input','local','local_pool'])
    all_gradient=sum(gradients.values())
    features=output['features']
    informative=int(((batch['student']['nodes'][...,15]>0)&batch['student']['local_mask']).sum())
    stats=feature_statistics(model.pool_global(features))
    tensors=[output['reconstruction'],features['head_input'],features['global_tokens']]
    finite=all(bool(torch.isfinite(x).all()) for x in tensors) and all(np.isfinite(v) for v in gradients.values())
    reasons=[]
    if not finite:reasons.append('nonfinite activation/gradient')
    if informative and local==0:reasons.append('exact zero informative local encoder gradient')
    if informative and local/max(all_gradient,1e-30)<1e-12:reasons.append('local gradient share below 1e-12')
    record={'step':step,'gradient_squared_norms':gradients,'local_gradient_share':local/max(all_gradient,1e-30),
        'informative_nodes':informative,'global_features':stats,'healthy':not reasons,'failure_reasons':reasons,
        'decoder_output_RMS':float(output['reconstruction'].detach().float().square().mean().sqrt())}
    for key in ['student_projection','teacher_projection']:
        if key in output:record[key]=feature_statistics(output[key])
    return record


@torch.no_grad()
def evaluate(model,loader,device,bf16,config):
    model.eval()
    rows,features,losses,counts,teacher,covered,permutation=[],[],[],[],[],[],[]
    for batch in loader:
        batch=to_device(batch,device)
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=bf16):
            output=model(batch);loss,parts=objective_loss(output,batch,config)
        losses.append({key:float(value) for key,value in parts.items()})
        latents=output['features']['global_tokens']
        # A cyclic event permutation preserves decoder geometry and the latent
        # marginal but breaks event conditioning. Batch-one rows are excluded.
        if len(latents)>1:
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=bf16):
                shuffled=model.decoder(batch['query'],latents.roll(1,0))
            exact=output['reconstruction'][...,0].float()
            changed=shuffled[...,0].float()
            truth=batch['reconstruction'][...,0]
            for level in range(3):
                selected=batch['query_level']==level
                difference=(torch.nn.functional.binary_cross_entropy_with_logits(changed[selected],truth[selected],reduction='none')
                    -torch.nn.functional.binary_cross_entropy_with_logits(exact[selected],truth[selected],reduction='none'))
                permutation.append({'level':level,'queries':int(selected.sum()),'delta_BCE':float(difference.mean())})
        features.append(model.pool_global(output['features']).cpu())
        if 'teacher_projection' in output:teacher.append(output['teacher_projection'].cpu())
        prediction=output['reconstruction'].float().cpu().numpy()
        target=batch['reconstruction'].cpu().numpy()
        level=batch['query_level'].cpu().numpy();near=batch['query_near'].cpu().numpy()
        for j in range(len(prediction)):
            rows.append(np.column_stack([level[j],near[j],target[j],prediction[j,:4]]))
        counts.append(batch['mask_counts'].cpu().numpy())
        if 'query_positive_unique' in batch:covered.append(batch['query_positive_unique'].cpu().numpy())
    values=np.concatenate(rows)
    measured={}
    for level in range(3):
        for kind,flag in [('all',np.ones(len(values),dtype=bool)),('near',values[:,1]==1),('uniform',values[:,1]==0)]:
            selected=values[(values[:,0]==level)&flag]
            y=selected[:,2]
            logits=np.clip(selected[:,6],-50,50)
            p=1/(1+np.exp(-logits))
            has_both=len(np.unique(y))==2
            measured[f'level{level}_{kind}']={'queries':len(y),'positives':int(y.sum()),
                'prevalence':float(y.mean()) if len(y) else None,
                'AP':float(average_precision_score(y,p)) if y.sum() else None,
                'ROC_AUC':float(roc_auc_score(y,p)) if has_both else None,
                'probability_std':float(p.std()) if len(p) else None,
                'predicted_positive_fraction_at_half':float((p>=.5).mean()) if len(p) else None,
                'BCE':float(np.mean(np.logaddexp(0,logits)-y*logits)) if len(y) else None,
                'constant_prevalence_BCE':float(-(y.mean()*np.log(max(y.mean(),1e-12))+(1-y.mean())*np.log(max(1-y.mean(),1e-12)))) if len(y) else None,
                'count_MAE_positive':float(np.abs(selected[y>0,7]-selected[y>0,3]).mean()) if y.sum() else None}
    data=np.concatenate(counts).sum(0)
    report={'loss':{k:float(np.mean([r[k] for r in losses])) for k in losses[0]},'queries':measured,
        'global_features':feature_statistics(torch.cat(features)),
        'mask_counts_total_and_hidden':data.tolist(),
        'empirical_hidden_fraction':[float(a[1]/max(1,a[0])) for a in data],
        'scope':'original train partition inner reconstruction-probe; fixed seeds; mixture query risk, no downstream validation/test'}
    if teacher:report['teacher_features']=feature_statistics(torch.cat(teacher))
    report['event_permutation_delta_BCE']={str(level):float(np.average([r['delta_BCE'] for r in permutation if r['level']==level],
        weights=[r['queries'] for r in permutation if r['level']==level])) for level in range(3)} if permutation else {}
    if covered:
        unique=np.concatenate(covered).sum(0)
        report['positive_unique_queries_by_native_coarse_bar']=unique.tolist()
        report['native_masked_support_coverage_per_view']=float(unique[0]/max(1,data[0,1]))
        report['bar_masked_support_coverage_per_view']=float(unique[2]/max(1,data[1,1]))
    frame=pd.DataFrame(values,columns=['level','near','occupied','log_count','log_energy','log_PE',
        'occupancy_logit','predicted_log_count','predicted_log_energy','predicted_log_PE'])
    return report,frame


def run(args):
    config=json.loads(args.campaign.read_text())['experiments'][args.task]
    root=args.root;output=root/'experiments'/config['name'];output.mkdir(parents=True,exist_ok=True)
    if (output/'manifest.json').exists() and json.loads((output/'manifest.json').read_text()).get('status')=='complete':
        return
    if not json.loads((root/'data_summary.json').read_text())['passed']:
        raise RuntimeError('assembled S4 corpus must pass its new-label join checks')
    torch.set_num_threads(1);pa.set_cpu_count(1);pa.set_io_thread_count(1)
    seed=config.get('seed',42);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    device=torch.device(args.device);bf16=device.type=='cuda' and torch.cuda.is_bf16_supported(including_emulation=False)
    if device.type=='cuda' and not bf16:raise RuntimeError('formal S4 requires native BF16 GPU')
    started=time.perf_counter()
    train=PretrainDataset(root,'train',config);validation=PretrainDataset(root,'validation',config)
    sampler=SpatialBatchSampler(train,config.get('batch_size',16),seed,steps=config.get('steps_per_epoch'))
    val_ids=np.arange(min(config.get('reconstruction_probe_events',2000),len(validation)))
    workers=config.get('workers',1)
    options={'num_workers':workers,'collate_fn':collate_pretrain,'pin_memory':device.type=='cuda',
        'generator':torch.Generator().manual_seed(seed)}
    if workers:options.update(multiprocessing_context='spawn',worker_init_fn=init_worker,prefetch_factor=1,persistent_workers=True)
    train_loader=DataLoader(train,batch_sampler=sampler,**options)
    val_loader=DataLoader(Subset(validation,val_ids),batch_size=config.get('batch_size',16),**options)
    model=PretrainingModel(config).to(device)
    if config.get('initialize_pretraining'):
        state=torch.load(config['initialize_pretraining'],map_location='cpu',weights_only=False)
        model.load_state_dict(state['model'],strict=True)
    optimizer=optimizer_for(model,config)
    total_steps=config.get('epochs',4)*len(sampler);warmup=config.get('warmup_steps',250)
    def factor(step):
        if step<warmup:return (step+1)/max(1,warmup)
        f=min(1.,(step-warmup)/max(1,total_steps-warmup))
        return .03+.97*.5*(1+math.cos(math.pi*f))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,factor)
    history,health=[],[];start_epoch=0;steps=0;prior_seconds=0.;best=float('inf')
    manifest={'utc':datetime.now(timezone.utc).isoformat(),'config':config,'status':'running',
        'train_events':len(train),'inner_reconstruction_probe_events':len(val_ids),'steps_per_epoch':len(sampler),
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'GPU':torch.cuda.get_device_name() if device.type=='cuda' else None,'torch':torch.__version__,
        'precision':'BF16 encoder / FP32 objective' if bf16 else 'FP32',
        'calibration_test_used':False,'downstream_validation_used':False,'proxy':'accepted_existing_values'}
    if (output/'last.pt').exists():
        state=torch.load(output/'last.pt',map_location=device,weights_only=False)
        if state['config']!=config:raise ValueError('resume configuration changed')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        history,health=state['history'],state['health'];start_epoch=state['epoch'];steps=state['steps'];best=state['best']
        prior_seconds=state['seconds'];random.setstate(state['random_state']);np.random.set_state(state['numpy_state'])
        torch.set_rng_state(state['torch_state'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_states']])
        manifest['resumed_from_epoch']=start_epoch
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    write_json(output/'manifest.json',manifest)
    if start_epoch==0:
        baseline,frame=evaluate(model,val_loader,device,bf16,config)
        write_json(output/'initial_reconstruction.json',baseline)
        save_parquet(frame,output/'initial_query_predictions.parquet')
    for epoch in range(start_epoch,config.get('epochs',4)):
        train.epoch.fill_(epoch);sampler.epoch=epoch;model.train()
        epoch_start=time.perf_counter();parts_sum={};seen=0;ready=time.perf_counter();data_wait=0.
        for step,batch in enumerate(train_loader,1):
            data_wait+=time.perf_counter()-ready
            batch=to_device(batch,device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=bf16):
                out=model(batch);loss,parts=objective_loss(out,batch,config)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite S4 objective; review before resuming')
            loss.backward()
            if steps%config.get('health_every_steps',100)==0:
                record=internal_health(model,out,batch,steps+1);health.append(record)
                write_json(output/'training_health.json',{'records':health})
                if not record['healthy']:
                    manifest.update(status='internal_health_failed',failure=record);write_json(output/'manifest.json',manifest)
                    raise RuntimeError('S4 encoder disconnected or numerically unhealthy')
            torch.nn.utils.clip_grad_norm_(model.parameters(),config.get('clip_grad_norm',1.),error_if_nonfinite=True)
            optimizer.step();scheduler.step();steps+=1
            momentum=1-(1-config.get('ema_momentum',.996))*(1+math.cos(math.pi*steps/max(1,total_steps)))/2
            model.update_teacher(momentum)
            batch_size=len(batch['query']);seen+=batch_size
            for key,value in parts.items():parts_sum[key]=parts_sum.get(key,0.)+float(value.detach())*batch_size
            if step%config.get('progress_every_steps',100)==0:
                progress={'epoch':epoch+1,'step':step,'optimizer_steps':steps,'seen':seen,
                    'loss':{k:v/seen for k,v in parts_sum.items()},'events_per_second':seen/(time.perf_counter()-epoch_start),
                    'data_wait_seconds':data_wait,'seconds':prior_seconds+time.perf_counter()-started}
                write_json(output/'progress.json',progress);print(json.dumps(progress),flush=True)
            ready=time.perf_counter()
        train_seconds=time.perf_counter()-epoch_start
        measured,frame=evaluate(model,val_loader,device,bf16,config)
        score=measured['loss']['loss']
        record={'epoch':epoch+1,'optimizer_steps':steps,'train_loss':{k:v/seen for k,v in parts_sum.items()},
            'train_exposures':seen,'train_seconds':train_seconds,'events_per_second':seen/train_seconds,
            'inner_reconstruction_probe':measured,'wall_seconds':prior_seconds+time.perf_counter()-started}
        history.append(record)
        improved=score<best
        best=min(best,score)
        state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
            'config':config,'epoch':epoch+1,'steps':steps,'best':best,'history':history,'health':health,
            'seconds':prior_seconds+time.perf_counter()-started,'random_state':random.getstate(),
            'numpy_state':np.random.get_state(),'torch_state':torch.get_rng_state(),
            'cuda_states':torch.cuda.get_rng_state_all() if device.type=='cuda' else []}
        if improved:
            save_checkpoint(output/'best.pt',{'model':model.state_dict(),'config':config,'epoch':epoch+1})
            save_parquet(frame,output/'query_predictions.parquet');write_json(output/'reconstruction.json',measured)
        save_checkpoint(output/'last.pt',state);write_json(output/'history.json',history)
        print(json.dumps({'epoch':epoch+1,'inner_loss':score,'global_features':measured['global_features']}),flush=True)
    manifest.update(status='complete',optimizer_steps=steps,seconds=prior_seconds+time.perf_counter()-started,
        best_inner_loss=best,peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else 0,
        internal_health_samples=len(health),all_internal_health_pass=all(r['healthy'] for r in health))
    write_json(output/'manifest.json',manifest)
    train.native.close();validation.native.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--task',type=int,required=True)
    parser.add_argument('--device',default='cuda')
    run(parser.parse_args())


if __name__=='__main__':main()
