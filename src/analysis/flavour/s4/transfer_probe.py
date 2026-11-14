"""Real CUDA frozen-to-trainable encoder regression with eager comparison."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import time
from types import MethodType

import numpy as np
import pandas as pd
import torch

from analysis.flavour.inventory import write_json
from analysis.flavour.s3.dataset import collate_planes
from analysis.flavour.s4.dataset import LabelDataset
from analysis.flavour.s4.low_label import TransferAdamW,TransferHealth
from analysis.flavour.s4.train import build_model


def legacy_encode_local(self,nodes,mask):
    features=(self.compiled_features(nodes,mask) if self.compiled_features is not None and nodes.is_cuda
        else self.encode_features(nodes,mask))
    return self.local_pool(features)


def inspect(root,config,device):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if device.type!='cuda' or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError('formal FP32/BF16 transfer regression requires native BF16 CUDA')
    output=root/'experiments'/config['name'];output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    write_json(output/'manifest.json',{'status':'running','config':config,'utc':datetime.now(timezone.utc).isoformat()})
    data=LabelDataset(root,'train',config)
    catalog=pd.read_parquet(root/'catalog.parquet')
    counts=data.metadata[['run','chunk','row','label']].merge(catalog[['run','chunk','row',
        'pixel_channels','bar_channels','tracker_channels']],on=['run','chunk','row'],validate='one_to_one')
    occupancy=counts[['pixel_channels','bar_channels','tracker_channels']].sum(axis=1)
    ids=np.concatenate([np.flatnonzero(counts.label.eq(c)&occupancy.between(10,1000))[:2] for c in range(4)])
    if len(ids)!=8:raise ValueError('eight moderate-occupancy train events required')
    items=[data[int(i)] for i in ids];data.native.close()
    batch=collate_planes(items);labels=batch.pop('label').to(device);batch.pop('index')
    batch={k:v.to(device) for k,v in batch.items()}
    reports=[];checks={}
    for bf16 in [False,True]:
        trajectories={}
        for mode in ['legacy_compiled','eager','phase_safe_compiled']:
            torch.compiler.reset()
            torch.manual_seed(config['seed'])
            case={**config,'compile_local':mode!='eager','linear_probe_passes':1,
                'complete_pass_steps':2,'classifier_dropout':0.}
            model=build_model(case).to(device)
            if mode=='legacy_compiled':model.encode_local=MethodType(legacy_encode_local,model)
            optimizer=TransferAdamW(model,case);health=TransferHealth(model)
            initial=model.input[0].weight.detach().clone();records=[];logits_saved=[]
            for step in range(5):
                model.train();optimizer.zero_grad(set_to_none=True);health.start(batch)
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf16):
                    logits=model(batch);loss=torch.nn.functional.cross_entropy(logits,labels)
                loss.backward();record=health.finish(logits,step+1)
                records.append(record);logits_saved.append(logits.detach().cpu())
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                # Include LP validation: it must not prime a frozen compiled graph.
                if step==0:
                    model.eval()
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf16):model(batch)
            trajectory=torch.stack(logits_saved);trajectories[mode]=trajectory
            delta=float((model.input[0].weight-initial).abs().max())
            row={'precision':'BF16' if bf16 else 'FP32','mode':mode,'input_parameter_maximum_change':delta,
                'records':records,'all_sampled_healthy':all(r['healthy'] for r in records)}
            reports.append(row)
            if mode!='legacy_compiled':
                checks[row['precision']+'_'+mode+'_frozen_then_updated']=all(
                    r.get('intentional_encoder_freeze') for r in records[:2]) and all(
                    r['gradient_squared_norms'].get(key,0)>0 for r in records[2:] for key in ['input','local']) and delta>0
                checks[row['precision']+'_'+mode+'_healthy']=row['all_sampled_healthy']
            health.close();del model,optimizer,health;torch.cuda.empty_cache()
        tolerance=.025 if bf16 else 1e-4
        difference=float((trajectories['phase_safe_compiled']-trajectories['eager']).abs().max())
        checks[('BF16' if bf16 else 'FP32')+'_eager_compiled_logits_match']=difference<tolerance
        reports.append({'precision':'BF16' if bf16 else 'FP32','eager_compiled_maximum_logit_error':difference,
            'tolerance':tolerance})
    report={'utc':datetime.now(timezone.utc).isoformat(),'config':config,'passed':all(checks.values()),
        'checks':checks,'experiments':reports,'events':counts.iloc[ids][['run','chunk','row','label']].to_dict('records'),
        'seconds':time.perf_counter()-started,'GPU':torch.cuda.get_device_name(),'code_commit':os.environ.get('PINPOINT_CODE_COMMIT'),
        'scope':'eight original train events, two LP and three full updates, identical initialization and batches; includes LP eval; no validation fitting'}
    write_json(output/'transfer_compilation.json',report)
    write_json(output/'manifest.json',{'status':'complete' if report['passed'] else 'internal_health_failed',
        'config':config,'seconds':report['seconds'],'utc':report['utc'],'all_checks_pass':report['passed']})
    if not report['passed']:raise RuntimeError('frozen/unfrozen compiled encoder regression failed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--task',type=int,required=True);parser.add_argument('--device',default='cuda')
    args=parser.parse_args();config=json.loads(args.campaign.read_text())['experiments'][args.task]
    inspect(args.root,config,torch.device(args.device))


if __name__=='__main__':main()
