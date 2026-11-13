"""Complete-event inference/backward stress measurements on train occupancy ranks."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import torch

from analysis.flavour.s3.dataset import PlaneDataset, collate_planes
from analysis.flavour.s3.models_repair import build_model


def main():
    program_started=time.perf_counter();started_utc=datetime.now(timezone.utc).isoformat()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--tasks','--task',type=int,nargs='+',required=True)
    parser.add_argument('--events',type=int,default=3)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(1);pa.set_cpu_count(1);pa.set_io_thread_count(1)
    torch.manual_seed(42)
    configs=json.loads(args.campaign.read_text())['experiments']
    if args.output is None:
        if len(args.tasks)!=1:
            parser.error('multiple tasks need an explicit --output directory')
        args.output=args.root/'stress'/configs[args.tasks[0]]['name']
    device=torch.device(args.device)
    # FP32 interactive T4 measurements do not stand in for H100/BF16 results.
    bf16=device.type=='cuda' and torch.cuda.is_bf16_supported(including_emulation=False)
    rows=[]; args.output.mkdir(parents=True,exist_ok=True)
    for task in args.tasks:
        config=configs[task]
        dataset=PlaneDataset(args.root,'train',config)
        catalog=pd.read_parquet(args.root/'catalog.parquet')
        order=dataset.metadata[['run','chunk','row']].merge(catalog,on=['run','chunk','row'],validate='one_to_one',how='left')
        workload=order[['pixel_channels','bar_channels','tracker_channels']].sum(axis=1).to_numpy()
        indices=np.argsort(workload,kind='stable')[np.unique(np.linspace(0,len(dataset)-1,args.events,dtype=int))]
        model=build_model(config).to(device)
        for index in indices:
            started=time.perf_counter();item=dataset[int(index)];transform=time.perf_counter()-started
            batch=collate_planes([item]);y=batch.pop('label').to(device);batch.pop('index')
            batch={k:v.to(device) for k,v in batch.items()}
            for training in [False,True]:
                model.train(training);model.zero_grad(set_to_none=True)
                if device.type=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                started=time.perf_counter()
                with torch.set_grad_enabled(training):
                    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=bf16):
                        logits=model(batch);loss=torch.nn.functional.cross_entropy(logits.float(),y)
                    if training:loss.backward()
                if device.type=='cuda':torch.cuda.synchronize()
                finite=bool(torch.isfinite(logits).all()) and all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
                row={'experiment':config['name'],'index':int(index),'raw_channels':int(workload[index]),'represented_nodes':len(item['nodes']),
                     'local_blocks':int(item['groups'].max())+1,'module_groups':len(batch['module_id']),
                     'region_groups':len(batch['region_id']),
                     'encoded_regional_tokens': (len(batch['region_id'])*config.get('region_queries',2)
                        if config.get('compression')=='multiscale' else None),
                     'encoded_fixed_module_tokens': (len(batch['module_id'])*config.get('module_queries',4)
                        if config.get('compression')=='fixed' and config['architecture']!='part' else None),
                     'all_local_direct_readout':config.get('readout_skip',False),
                     'parameters':sum(p.numel() for p in model.parameters()),'mode':'backward' if training else 'inference',
                     'seconds':time.perf_counter()-started,'transform_seconds':transform,'finite':finite,
                     'peak_mib':torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else 0,
                     'input_gradient_squared_norm':float(model.input[0].weight.grad.float().square().sum()) if training else None}
                rows.append(row);print(json.dumps(row),flush=True)
                pd.DataFrame(rows).to_parquet(args.output/'measurements.parquet',index=False)
                if not finite:raise ValueError('nonfinite full-event profile')
        dataset.native.close()
    summary={'utc':datetime.now(timezone.utc).isoformat(),'started_utc':started_utc,
             'program_wall_seconds':time.perf_counter()-program_started,
             'GPU':torch.cuda.get_device_name() if device.type=='cuda' else None,
             'torch':torch.__version__,'precision':'BF16' if bf16 else 'FP32','rows':len(rows),
             'scope':'train occupancy-rank stress events including maximum; first event warmup; no optimizer fitting; no population throughput claim'}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
