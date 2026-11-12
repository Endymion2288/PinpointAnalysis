"""Short real-observation H100/BF16 optimizer checks; not scientific evaluation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import torch

from analysis.flavour.s2.train import metrics
from analysis.flavour.s3.dataset import PlaneDataset, collate_planes
from analysis.flavour.s3.models import PlaneModuleClassifier


def main():
    program_started=time.perf_counter();started_utc=datetime.now(timezone.utc).isoformat()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--tasks',type=int,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1);pa.set_cpu_count(1);pa.set_io_thread_count(1)
    torch.manual_seed(42)
    campaign=json.loads(args.campaign.read_text());assert torch.cuda.get_device_name()==campaign['hardware']['GPU']
    assert torch.__version__==campaign['hardware']['torch'] and torch.cuda.is_bf16_supported(including_emulation=False)
    records=[];args.output.mkdir(exist_ok=True)
    for task in args.tasks:
        config=campaign['experiments'][task]
        train=PlaneDataset(args.root,'train',config);val=PlaneDataset(args.root,'validation',config)
        model=PlaneModuleClassifier(config).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.0005)
        indices=np.concatenate([np.flatnonzero(train.metadata.label.to_numpy()==c)[:4] for c in range(4)])
        losses=[]
        for step in range(4):
            # Every tiny batch contains all four labels; no truth in the inputs.
            use=indices.reshape(4,4)[:,step]
            batch=collate_planes([train[int(i)] for i in use]);batch.pop('index');label=batch.pop('label').cuda()
            batch={k:v.cuda() for k,v in batch.items()};optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=torch.nn.functional.cross_entropy(model(batch).float(),label)
            loss.backward();assert torch.isfinite(loss)
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();losses.append(float(loss))
        model.eval();ids=np.concatenate([np.flatnonzero(val.metadata.label.to_numpy()==c)[:4] for c in range(4)])
        batch=collate_planes([val[int(i)] for i in ids]);batch.pop('index');label=batch.pop('label').numpy();batch={k:v.cuda() for k,v in batch.items()}
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):p=model(batch).float().softmax(-1).cpu().numpy()
        assert np.isfinite(p).all()
        records.append({'experiment':config['name'],'applied_updates':4,'losses':losses,'tiny_validation':metrics(label,p),'parameters':sum(p.numel() for p in model.parameters())})
        (args.output/'smoke.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'started_utc':started_utc,
            'program_wall_seconds':time.perf_counter()-program_started,'GPU':torch.cuda.get_device_name(),'torch':torch.__version__,
            'scope':'execution only; 16 train and 16 validation per config; no model selection','records':records},indent=2)+'\n')
        train.native.close();val.native.close()
        print(config['name'],'optimizer smoke pass',flush=True)


if __name__=='__main__':main()
