"""Measure complete-event S2 resource curves on fixed occupancy strata."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import torch

from analysis.flavour.inventory import COMPRESSION, write_json
from analysis.flavour.s2.dataset import SpatialDataset, collate_spatial
from analysis.flavour.s2.models import PointClassifier


def run(args):
    torch.set_num_threads(1)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    torch.manual_seed(42)
    catalog=pd.read_parquet(args.root/'catalog.parquet')
    train=catalog[catalog.split.eq('train')].copy()
    occupancy=train[['pixel_channels','bar_channels','tracker_channels']].sum(axis=1)
    # Sample ranks, including empty events and the observed upper tail. This is
    # a stress curve, not an unbiased estimate of the event population mean.
    ordered=train.iloc[np.argsort(occupancy.to_numpy(),kind='stable')]
    chosen=ordered.iloc[np.unique(np.linspace(0,len(ordered)-1,args.events,dtype=int))]
    key=['run','chunk','row']
    device=torch.device(args.device)
    dtype=torch.bfloat16 if device.type=='cuda' and torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    rows=[]
    output=args.root/'resource_profile'
    output.mkdir(parents=True,exist_ok=True)
    for pitch,projection in [(0.,False),(0.,True),(.1,False),(.2,False),(1.,False),(1.,True)]:
        dataset=SpatialDataset(args.root,'train',{'pitch_mm':pitch,'projection':projection})
        index=dataset.metadata.reset_index(names='index').merge(chosen[key],on=key,validate='one_to_one')['index']
        for architecture,dim in [('part',48),('particlenet',64)]:
            model=PointClassifier(architecture,dim=dim).to(device)
            parameters=sum(p.numel() for p in model.parameters())
            for i in index:
                t=time.perf_counter()
                item=dataset[int(i)]
                load_transform=time.perf_counter()-t
                t=time.perf_counter()
                batch=collate_spatial([item])
                batch.pop('index')
                y=batch.pop('label').to(device)
                batch={k:v.to(device) for k,v in batch.items()}
                if device.type=='cuda':
                    torch.cuda.synchronize()
                collate_transfer=time.perf_counter()-t
                for mode in ['inference','training']:
                    model.train(mode=='training')
                    model.zero_grad(set_to_none=True)
                    if device.type=='cuda':
                        torch.cuda.reset_peak_memory_stats()
                    t=time.perf_counter()
                    with torch.set_grad_enabled(mode=='training'):
                        with torch.autocast(device_type=device.type,dtype=dtype,enabled=device.type=='cuda'):
                            logits=model(batch)
                            loss=torch.nn.functional.cross_entropy(logits.float(),y)
                        if device.type=='cuda':
                            torch.cuda.synchronize()
                        forward=time.perf_counter()-t
                        t=time.perf_counter()
                        if mode=='training':
                            # FP32 loss / unscaled backward is sufficient for
                            # resource measurement; this job does no fitting.
                            loss.backward()
                        if device.type=='cuda':
                            torch.cuda.synchronize()
                        backward=time.perf_counter()-t if mode=='training' else 0.
                    meta=dataset.metadata.iloc[int(i)]
                    rows.append({**{k:int(meta[k]) for k in key},'architecture':architecture,
                        'dim':dim,'parameters':parameters,'pitch_mm':pitch,'projection':projection,
                        'mode':mode,'nodes':len(item['nodes']),'blocks':int(item['groups'].max())+1,
                        'load_transform_seconds':load_transform,
                        'collate_transfer_seconds':collate_transfer,'forward_seconds':forward,
                        'backward_seconds':backward,'loss_finite':bool(torch.isfinite(loss)),
                        'peak_gpu_mib':torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else 0.})
                pd.DataFrame(rows).to_parquet(output/'measurements.parquet',index=False,compression=COMPRESSION)
                print(json.dumps(rows[-1]),flush=True)
            del model
        dataset.native.close()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    measured=pd.DataFrame(rows)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for (architecture,pitch,projection),group in measured.groupby(['architecture','pitch_mm','projection']):
        label=f'{architecture}, {pitch:g} mm, {"proj" if projection else "joint"}'
        inference=group[group['mode'].eq('inference')]
        training=group[group['mode'].eq('training')]
        axes[0].scatter(inference.blocks,inference.forward_seconds,s=12,label=label)
        axes[1].scatter(training.blocks,training.peak_gpu_mib,s=12,label=label)
    axes[0].set(xlabel='Retained local blocks per event',ylabel='Inference forward time [s]',
                xscale='log',yscale='log',title='Complete-event latency (batch 1)')
    axes[1].set(xlabel='Retained local blocks per event',ylabel='Peak allocated GPU memory [MiB]',
                xscale='log',yscale='log',title='Training memory (batch 1)')
    axes[1].legend(fontsize=5,loc='best')
    fig.savefig(output/'resource_curves.png',dpi=170)
    fig.savefig(output/'resource_curves.pdf')
    plt.close(fig)
    write_json(output/'summary.json',{'events_per_representation':len(chosen),'rows':len(rows),
        'device':str(device),'gpu':torch.cuda.get_device_name() if device.type=='cuda' else None,
        'torch':torch.__version__,'precision':str(dtype),'batch_size':1,'seed':42,
        'sampling':'equally spaced total-channel occupancy ranks including maximum; stress curve, not population mean',
        'warmup':'first event per architecture/representation includes CUDA warmup; identify in individual rows',
        'scope':'resource_measurement_no_training_or_validation_selection'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--events',type=int,default=24)
    parser.add_argument('--device',default='cuda')
    run(parser.parse_args())


if __name__=='__main__':
    main()
