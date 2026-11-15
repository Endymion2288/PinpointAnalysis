"""R2 fixed-exposure training with update schedules and recoverable checkpoints."""

import argparse
import itertools
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from torch.utils.data import DataLoader

from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.s2.dataset import SpatialBatchSampler
from analysis.flavour.s4.stage_inputs import stage_shards
from analysis.flavour.v2.profile import device_batch, init_worker, shutdown_loader, MemoryMonitor
from analysis.flavour.v2.r1.data import R1Dataset, collate
from analysis.flavour.v2.r1.fit import gradients
from analysis.flavour.v2.r1.health import Activations
from analysis.flavour.v2.r1.metrics import quick_report
from analysis.flavour.v2.r1.models import build_model, branch_groups
from analysis.flavour.v2.r1.tasks import probabilities, task_loss
from analysis.flavour.v2.r1.train import predict, save_predictions
from analysis.flavour.v2.r2.schedule import learning_rate, sampler_trace


class OffsetSampler:
    def __init__(self, sampler):
        self.sampler, self.offset = sampler, 0

    def __iter__(self):
        return itertools.islice(iter(self.sampler), self.offset, None)

    def __len__(self):
        return len(self.sampler)-self.offset


def make_loader(dataset, config, shuffle):
    sampler = SpatialBatchSampler(dataset, config['batch_size'], config['seed'], shuffle=shuffle)
    offset = OffsetSampler(sampler)
    workers = config.get('workers', 2)
    # Loader worker seeding must not consume the model/dropout RNG on resume.
    generator = torch.Generator().manual_seed(config['seed']+7819)
    loader = DataLoader(dataset, batch_sampler=offset, collate_fn=collate,
        num_workers=workers, pin_memory=True, persistent_workers=workers>0,
        worker_init_fn=init_worker, generator=generator,
        **({'multiprocessing_context':'spawn', 'prefetch_factor':2} if workers else {}))
    return loader, sampler, offset


def subset(dataset, selected):
    keys = ['run','chunk','row']
    positions = pd.MultiIndex.from_frame(dataset.metadata[keys]).get_indexer(pd.MultiIndex.from_frame(selected[keys]))
    if (positions<0).any():
        raise ValueError('fixed screen rows absent from Tune observations')
    dataset.indexes = dataset.indexes[positions]
    dataset.metadata = dataset.metadata.iloc[positions].reset_index(drop=True)
    if dataset.workload is not None:
        dataset.workload = dataset.workload[positions]


def atomic_state(path, model, optimizer, config, record, counts, position, best):
    temporary = path.with_suffix('.tmp')
    torch.save({'model':model.state_dict(), 'optimizer':optimizer.state_dict(),
        'config':config, 'record':record, 'counts':counts, 'position':position, 'best':best,
        'torch_rng':torch.get_rng_state(), 'cuda_rng':torch.cuda.get_rng_state(),
        'numpy_rng':np.random.get_state()}, temporary)
    os.replace(temporary, path)


def screen_diagnostics(metadata, indices, probability, logits):
    frame = metadata.iloc[indices]
    labels = frame.label.to_numpy()
    result = {'metrics':quick_report(labels, probability),
        'logit_mean_by_true_class':[], 'probability_mean_by_true_class':[]}
    for c in range(4):
        mask = labels==c
        result['logit_mean_by_true_class'].append(logits[mask].mean(0).tolist() if mask.any() else None)
        result['probability_mean_by_true_class'].append(probability[mask].mean(0).tolist() if mask.any() else None)
    if all(c in frame for c in ['pixel_channels','bar_channels','tracker_channels']):
        low = frame[['pixel_channels','bar_channels','tracker_channels']].sum(axis=1).to_numpy()<100
        result['occupancy_below_100'] = {'raw_counts':np.bincount(labels[low], minlength=4).tolist(),
            'metrics':quick_report(labels[low], probability[low]) if low.any() else None,
            'logit_mean':logits[low].mean(0).tolist() if low.any() else None}
    return result


def run(args):
    started = time.perf_counter()
    config = json.loads(args.config.read_text())
    torch.zeros(1, device='cuda:0').sum().item()
    torch.set_num_threads(config.get('main_threads',1))
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if config.get('deterministic_verification',False):
        torch.use_deterministic_algorithms(True)
    # Fixed FP32 is supported on any compatible card. BF16 changes must be
    # explicit paired recipes; never silently compare a fallback with BF16 D0.
    if config.get('bf16',True) and torch.cuda.get_device_capability()[0]<8:
        raise ValueError('BF16 recipe requires SM80; select an explicit FP32 recipe with matched D0')
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    output = args.root/'experiments'/config['name']
    resume = output/'resume.pt'
    output.mkdir(parents=True, exist_ok=args.resume)
    if args.resume and not resume.exists():
        raise FileNotFoundError('resume requested without a complete saved checkpoint')
    write_json(output/'config.json',config)
    scratch_parent = os.environ.get('PINPOINT_V2_SCRATCH', os.environ.get('_CONDOR_SCRATCH_DIR','/tmp'))
    with tempfile.TemporaryDirectory(prefix='r2-train-',dir=scratch_parent) as scratch, MemoryMonitor() as memory:
        data_root = Path(config['data_root'])
        paths = sorted(p for p in (data_root/'shards').iterdir() if (p/'observations.parquet').exists())
        shared = os.environ.get('PINPOINT_R2_SHARED_STAGE')
        if shared:
            ready = json.loads((Path(shared)/'ready.json').read_text())
            if [p.name for p in paths] != [row['shard'] for row in ready['shards']]:
                raise ValueError('shared staging does not cover the exact dataset shards')
            staging = {'shared':True, **ready}
            os.environ['PINPOINT_NATIVE_SHARD_ROOT'] = shared
        else:
            staging = stage_shards(paths, Path(scratch)/'native', workers=3)
            os.environ['PINPOINT_NATIVE_SHARD_ROOT'] = str(Path(scratch)/'native')
        train = R1Dataset(data_root,'train',config,cache_dir=scratch)
        train_loader, sampler, offset = make_loader(train,config,True)
        profile = config.get('profile',False)
        if profile:
            total_updates = config.get('profile_updates',256)
            trace = None
            small = tune = screen_loader = full_loader = None
        else:
            expected = json.loads(Path(config['preflight']).read_text())
            trace, sizes, class_trace = sampler_trace(sampler,train.metadata.label.to_numpy(),config['passes'])
            if trace['total_updates']!=expected['total_updates'] or trace['total_exposures']!=expected['total_exposures']:
                raise ValueError('runtime sampler differs from frozen preflight')
            total_updates = trace['total_updates']
            tune = R1Dataset(data_root,'validation',config,cache_dir=scratch)
            small = R1Dataset(data_root,'validation',config,cache_dir=scratch)
            subset(small,pd.read_parquet(Path(config['tune_screen'])))
            screen_loader,_,_ = make_loader(small,config,False)
            full_loader,_,_ = make_loader(tune,config,False)
        model = build_model(config).cuda()
        optimizer = torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
        groups = branch_groups(model)
        activations = Activations(model)
        watch = {n:params[0][1] for n,params in groups.items()}
        counts = np.zeros(len(train),dtype='uint16')
        counters = {'updates':0,'training_exposures':0,'non_tau_exposures':0,
            'exposure_by_class':[0]*4,'train_seconds':0.,'validation_seconds':0.,'data_wait_seconds':0.,'lr_sum':0.}
        record = {'status':'running','config':config,'GPU':torch.cuda.get_device_name(),
            'precision':config['precision'],'allocation_id':os.environ.get('PINPOINT_JOB_ID'),
            'capability':list(torch.cuda.get_device_capability()), 'Torch_CUDA_architectures':torch.cuda.get_arch_list(),
            'parameters':sum(p.numel() for p in model.parameters()),'staging':staging,'sampler':trace,
            'history':[],'health':[],'rolling_train':[],'attempts':[],
            'Cal_A_Cut_B_Confirm_payload_read':False}
        start_epoch, start_batch, best = 0,0,float('inf')
        if args.resume:
            state = torch.load(resume,map_location='cpu',weights_only=False)
            if state['config']!=config:
                raise ValueError('resume configuration differs from saved recipe')
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            record, counts = state['record'],state['counts']
            counters = record['counters']
            start_epoch,start_batch = state['position']
            best = state['best']
            torch.set_rng_state(state['torch_rng'])
            torch.cuda.set_rng_state(state['cuda_rng'])
            np.random.set_state(state['numpy_rng'])
        record['status']='running'
        record['attempts'].append({'allocation_id':os.environ.get('PINPOINT_JOB_ID'),
            'GPU':torch.cuda.get_device_name(),'resumed_at_update':counters['updates']})
        record['counters']=counters
        write_json(output/'manifest.json',record)
        torch.cuda.reset_peak_memory_stats()
        interval = config.get('validation_non_tau_exposures',75000)
        next_screen = (counters['non_tau_exposures']//interval+1)*interval
        window=record.get('rolling_window',{})
        rolling_loss, rolling_events = window.get('loss_sum',0.),window.get('events',0)
        rolling_class = np.asarray(window.get('class_counts',[0]*4),dtype='int64')
        rolling_logits = np.asarray(window.get('logit_sums',np.zeros((4,3 if config['task']=='T3' else 4))),dtype='float64')
        trained_from = counters['updates']
        for epoch in range(start_epoch,config.get('passes',1)):
            sampler.epoch = epoch
            offset.offset = start_batch if epoch==start_epoch else 0
            batch_position = offset.offset
            model.train()
            tick = time.perf_counter()
            for batch in train_loader:
                began = tick
                counters['data_wait_seconds'] += time.perf_counter()-tick
                step = counters['updates']
                lr = learning_rate(step,total_updates,config['learning_rate'])
                for group in optimizer.param_groups:
                    group['lr']=lr
                gpu = device_batch(batch,'cuda:0')
                sampled = step in [0,15] or (step+1)%1000==0
                before = {n:p.detach().clone() for n,p in watch.items()} if sampled else {}
                if sampled:
                    activations.start()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config.get('bf16',True)):
                    logits=model(gpu).float()
                    loss=task_loss(logits,gpu['label'],config['task'])
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite training loss')
                loss.backward()
                if sampled:
                    grad=gradients(groups)
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['clip_grad_norm'],error_if_nonfinite=True)
                optimizer.step()
                torch.cuda.synchronize()
                batch_position+=1
                labels=batch['label'].numpy()
                class_count=np.bincount(labels,minlength=4)
                counters['updates']+=1
                counters['training_exposures']+=len(labels)
                counters['non_tau_exposures']+=int((labels!=2).sum())
                counters['exposure_by_class']=(np.asarray(counters['exposure_by_class'])+class_count).tolist()
                counters['lr_sum']+=lr
                np.add.at(counts,batch['index'].numpy(),1)
                rolling_loss+=float(loss.detach())*len(labels)
                rolling_events+=len(labels)
                rolling_class+=class_count
                values=logits.detach().cpu().numpy()
                for c in range(4):
                    rolling_logits[c]+=values[labels==c].sum(0)
                counters['train_seconds']+=time.perf_counter()-began
                if sampled:
                    record['health'].append({'update':counters['updates'],'gradient_squared_norms':grad,
                        'preclip_gradient_norm':float(norm),'activations':activations.finish(),
                        'parameter_update_RMS':{n:float((p-before[n]).detach().float().square().mean().sqrt()) for n,p in watch.items()},
                        'probability_std':probabilities(logits,config['task']).detach().std(0,unbiased=False).cpu().tolist()})
                if counters['updates']%250==0 or counters['updates']==total_updates:
                    roll={'updates':counters['updates'],'exposures':counters['training_exposures'],
                        'class_counts':rolling_class.tolist(),'train_NLL':rolling_loss/rolling_events,
                        'logit_mean_by_true_class':np.divide(rolling_logits,rolling_class[:,None],
                            out=np.zeros_like(rolling_logits),where=rolling_class[:,None]>0).tolist(),'lr':lr}
                    record['rolling_train'].append(roll)
                    print(json.dumps({'name':config['name'],**roll,'train_seconds':counters['train_seconds']}),flush=True)
                    rolling_loss,rolling_events=0.,0
                    rolling_class[:]=0
                    rolling_logits[:]=0
                at_end=counters['updates']==total_updates
                record['rolling_window']={'loss_sum':rolling_loss,'events':rolling_events,
                    'class_counts':rolling_class.tolist(),'logit_sums':rolling_logits.tolist()}
                if not profile and (counters['non_tau_exposures']>=next_screen or at_end):
                    indices,p,raw,seconds=predict(model,screen_loader,config['task'],config.get('bf16',True))
                    counters['validation_seconds']+=seconds
                    row={'pass':epoch+1,'batch_in_pass':batch_position,**counters,
                        'train_NLL':record['rolling_train'][-1]['train_NLL'] if record['rolling_train'] else float(loss.detach()),
                        'lr':lr,**screen_diagnostics(small.metadata,indices,p,raw)}
                    record['history'].append(row)
                    metric=row['metrics'][config['task']]['NLL']
                    if metric<best:
                        best=metric
                        atomic_state(output/'best.pt',model,optimizer,config,record,counts,(epoch,batch_position),best)
                        record['best_screen']={'updates':counters['updates'],'NLL':best,'criterion':config['task']+'_NLL'}
                    next_screen=(counters['non_tau_exposures']//interval+1)*interval
                    atomic_state(resume,model,optimizer,config,record,counts,(epoch,batch_position),best)
                    write_json(output/'manifest.json',record)
                    print(json.dumps({'name':config['name'],'screen':row}),flush=True)
                    model.train()
                if args.stop_after_updates and counters['updates']>=args.stop_after_updates and not at_end:
                    record['status']='checkpointed_for_resume'
                    atomic_state(resume,model,optimizer,config,record,counts,(epoch,batch_position),best)
                    write_json(output/'manifest.json',record)
                    for loader in [train_loader,screen_loader,full_loader]:
                        if loader is not None:shutdown_loader(loader)
                    for dataset in [train,small,tune]:
                        if dataset is not None:dataset.native.close()
                    del os.environ['PINPOINT_NATIVE_SHARD_ROOT']
                    return
                if at_end:
                    break
                tick=time.perf_counter()
            if counters['updates']==total_updates:
                break
            if not profile:
                atomic_state(resume,model,optimizer,config,record,counts,(epoch+1,0),best)
                write_json(output/'manifest.json',record)
        if counters['updates']!=total_updates:
            raise ValueError('training stopped before its prescribed optimizer budget')
        atomic_state(output/'exposure.pt',model,optimizer,config,record,counts,(epoch,batch_position),best)
        if not profile:
            for kind,path in [('endpoint',output/'exposure.pt'),('best_screen',output/'best.pt')]:
                if kind=='best_screen':
                    state=torch.load(path,map_location='cpu',weights_only=False)
                    model.load_state_dict(state['model'])
                indices,p,raw,seconds=predict(model,full_loader,config['task'],config.get('bf16',True))
                save_predictions(output/('predictions.parquet' if kind=='endpoint' else 'best_predictions.parquet'),tune,indices,p,raw)
                record['metrics' if kind=='endpoint' else 'best_metrics']=quick_report(tune.metadata.label.to_numpy()[indices],p)
                counters['validation_seconds']+=seconds
            if not np.all(counts==config['passes']):
                raise ValueError('actual per-event exposures differ from prescribed complete passes')
        coverage=train.metadata[['run','chunk','row','label','family_hi','family_lo']].copy()
        coverage['exposures']=counts
        coverage.to_parquet(output/'train_exposures.parquet',index=False,compression=COMPRESSION)
        for loader in [train_loader,screen_loader,full_loader]:
            if loader is not None:shutdown_loader(loader)
        for dataset in [train,small,tune]:
            if dataset is not None:dataset.native.close()
        record.update(status='complete',counters=counters,unique_train_events=int((counts>0).sum()),
            input_train_events=len(train),wall_seconds=time.perf_counter()-started,
            GPU_peak_MiB=torch.cuda.max_memory_allocated()/2**20,memory=memory.peak,
            events_per_train_second=(counters['training_exposures']/counters['train_seconds']),
            updates_in_this_attempt=counters['updates']-trained_from)
        del os.environ['PINPOINT_NATIVE_SHARD_ROOT']
    write_json(output/'manifest.json',record)
    if resume.exists():resume.unlink()
    print(json.dumps({'name':config['name'],'status':'complete','wall_seconds':record['wall_seconds'],
        'events_per_train_second':record['events_per_train_second']}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--stop-after-updates',type=int,help='save recoverable progress without changing the prescribed schedule')
    run(parser.parse_args())


if __name__=='__main__':
    main()
