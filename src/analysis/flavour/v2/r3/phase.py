"""Observation-conserving convolution sampling-phase diagnosis, Train only."""

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch.nn import functional as F

from analysis.flavour.inventory import write_json
from analysis.flavour.v2.profile import device_batch, MemoryMonitor
from analysis.flavour.v2.r1.data import R1Dataset, collate
from analysis.flavour.v2.r3.models import build_model


def phase_forward(model, batch, phase=None, capture=False):
    encoder=model.backbone
    b=len(batch['label'])
    h,w=encoder.image['height'],encoder.image['width']
    dense=batch['projection_values'].new_zeros(b*8*h*w)
    dense.index_add_(0,batch['projection_event']*(8*h*w)+batch['projection_indices'],batch['projection_values'])
    dense=dense.reshape(b*2,4,h,w)
    active=dense.flatten(1).any(1)
    totals=torch.log1p(dense.sum((-1,-2))).reshape(b,8)
    values=torch.log1p(dense)
    if phase is not None:
        row,column=phase
        values=F.pad(values,(column,3-column,row,3-row))
    summaries={}
    x=encoder.stem(values)
    if capture:
        summaries['stem_RMS']=float(x.float().square().mean().sqrt())
        summaries['stem_max_abs']=float(x.float().abs().max())
    for i,layer in enumerate(encoder.layers):
        x=layer(x)
        if capture and i in [1,3,7]:
            summaries[f'stage{i//2}_RMS']=float(x.float().square().mean().sqrt())
    pooled=torch.cat([x.mean((-1,-2)),x.amax((-1,-2))],-1)
    pooled=(pooled*active[:,None]).reshape(b,-1)
    image=encoder.norm(torch.cat([pooled.float(),totals],-1))
    fused=torch.cat([image,model.tracker(batch),batch['presence']],-1)
    with torch.autocast('cuda',enabled=False):
        logits=model.head(fused.float())
    # Padding inserts zeros into a fixed expanded canvas. Each source channel
    # occurs once, with the same absolute detector coordinate and signal.
    return logits,totals,summaries


def run(root, attempt):
    torch.set_num_threads(1)
    campaign=json.loads((root/'campaign.json').read_text())
    r2=Path(campaign['r2']); r0=Path(campaign['r0'])
    config=json.loads((r2/'configs/b1_d2_t4_s42.json').read_text())
    bf16=torch.cuda.get_device_capability()[0]>=8
    model=build_model(config).cuda().eval()
    state=torch.load(r2/'experiments/b1_d2_t4_s42/exposure.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(state['model'])
    output=root/'diagnostics'/f'phase_{attempt}'
    output.mkdir(parents=True,exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='r3-phase-') as scratch, MemoryMonitor() as memory:
        data=R1Dataset(r0/'micro_data','train',config,cache_dir=scratch)
        items=[data[i] for i in range(len(data))]
        keys=data.metadata[['run','chunk','row','label']].to_dict('records')
        data.native.close()
        h,w=config['image']['height'],config['image']['width']
        synthetic=[]
        for column in range(4):
            index=np.array([(80+column)*w+100+column],dtype='int64')
            synthetic.append({'index':len(synthetic),'label':0,'family':'B1',
                'projection_indices':index,'projection_values':np.ones(1,dtype='float32'),
                'tracker':np.empty((0,5),dtype='float32'),'presence':np.array([1,0,0],dtype='float32')})
        for slope in [-1,0,1]:
            row=np.arange(80,96)
            index=row*w+100+slope*(row-80)
            synthetic.append({**synthetic[0],'index':len(synthetic),
                'projection_indices':index.astype('int64'),'projection_values':np.ones(len(index),dtype='float32')})
        results={}
        exact_reference_max=0.
        for name,events in [('synthetic',synthetic),('real_Train',items)]:
            probabilities,activations,totals_by_phase={},{},{}
            for row in range(4):
                for column in range(4):
                    prediction=[]; moments=[]; totals=[]
                    for start in range(0,len(events),4):
                        gpu=device_batch(collate(events[start:start+4]),'cuda:0')
                        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf16):
                            if row==column==0:
                                actual=model(gpu).float()
                                reconstructed,_,_=phase_forward(model,gpu)
                                exact_reference_max=max(exact_reference_max,float((actual-reconstructed).abs().max()))
                            logits,current,summary=phase_forward(model,gpu,(row,column),capture=True)
                        prediction.extend(logits.float().softmax(-1).cpu().tolist())
                        totals.extend(current.cpu().tolist());moments.append(summary)
                    key=f'{row},{column}'
                    probabilities[key]=np.asarray(prediction)
                    totals_by_phase[key]=np.asarray(totals)
                    activations[key]={k:float(np.mean([m[k] for m in moments])) for k in moments[0]}
            anchor=probabilities['0,0']
            if any(not np.array_equal(v,totals_by_phase['0,0']) for v in totals_by_phase.values()):
                raise ValueError('padding phase changed measured response totals')
            results[name]={'events':len(events),'response_totals_equal':True,
                'phase_mean_max_probability_abs_difference':{key:float(np.abs(v-anchor).max(1).mean()) for key,v in probabilities.items()},
                'phase_argmax_agreement':{key:float((v.argmax(1)==anchor.argmax(1)).mean()) for key,v in probabilities.items()},
                'intermediate_activations':activations}
            np.savez_compressed(output/f'{name}_probabilities.npz',**probabilities)
    if exact_reference_max!=0:
        raise ValueError('phase diagnosis did not reproduce the original forward path exactly')
    record={'status':'complete','GPU':torch.cuda.get_device_name(),'BF16':bf16,
        'reference_forward_max_absolute_difference':exact_reference_max,
        'canvas':'fixed (height+3,width+3); complementary zero padding; no cropping or detector-coordinate changes',
        'results':results,'real_Train_keys':keys,'memory':memory.peak,
        'scope':'fixed-weight sampling sensitivity; neither a training comparison nor a classification gain',
        'Cal_A_Cut_B_Confirm_payload_read':False}
    write_json(output/'results.json',record)
    print(json.dumps({'status':'complete','phase_diagnostic':str(output),'events':len(items)}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--attempt',required=True)
    args=parser.parse_args()
    run(args.root,args.attempt)


if __name__=='__main__':
    main()
