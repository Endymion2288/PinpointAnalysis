"""Freeze the first scaling matrix and measure its exact metadata sampler budgets."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

import numpy as np

from analysis.flavour.inventory import write_json
from analysis.flavour.s2.dataset import SpatialBatchSampler
from analysis.flavour.v2.r1.data import R1Dataset
from analysis.flavour.v2.r2.schedule import learning_rate, sampler_trace


def config_from_r1(r1, family, task, seed=42, precision='BF16_encoder_FP32_head'):
    source = json.loads((r1/'experiments'/f'final_{family.lower()}_{task.lower()}_s17/manifest.json').read_text())['config']
    keys = ['family','task','image','cnn_width','cnn_variant','dim','modalities','train_domain',
        'batch_size','workers','cache_shards','bucket_size','learning_rate','weight_decay','clip_grad_norm',
        'local_chunk_blocks','architecture_revision','empty_response_bypass']
    result = {k:source[k] for k in keys if k in source}
    result.update(seed=seed, initialization='scratch', compile_local=False,
        bf16=precision!='FP32', precision=precision, main_threads=1,
        schedule='prescribed_total_updates_5pct_warmup_cosine_5pct_floor',
        warmup_fraction=.05, learning_rate_floor_fraction=.05)
    return result


def freeze(root, r1):
    if json.loads((r1/'acceptance.json').read_text())['status']!='complete':
        raise ValueError('complete R1 acceptance is required')
    if (root/'campaign.json').exists():
        raise FileExistsError('R2 campaign is already frozen')
    config_names = []
    for design,size,passes in [('D0',100000,4),('D1',100000,12),('D2',300000,4)]:
        for task in ['T3','T4']:
            config = config_from_r1(r1,'B1',task)
            name = f'b1_{design.lower()}_{task.lower()}_s42'
            config.update(name=name,design=design,pool_events=size,passes=passes,
                data_root=str(r1/'data') if size==100000 else str(root/'data'),
                tune_screen=str(r1/'data/tune_screen.parquet'),
                validation_non_tau_exposures=75000,
                preflight=str(root/'preflight'/f'{name}.json'))
            write_json(root/'configs'/f'{name}.json',config)
            config_names.append(name)
    write_json(root/'campaign.json',{'stage':'R2','status':'frozen_before_R2_Tune_access',
        'UTC':datetime.now(timezone.utc).isoformat(),'r1':str(r1),
        'r0':json.loads((r1/'campaign.json').read_text())['r0'],
        'initial_family':'B1','seed':42,'tasks':['T3','T4'],'initial_models':config_names,
        'matrix':{'D0':[100000,4],'D1':[100000,12],'D2':[300000,4]},
        'initialization':'fresh matched seed for all budgets; no continuation from annealed D0',
        'optimizer':'unchanged selected R1 LR,AdamW,weight_decay,batch and sampler within each task',
        'schedule':'each prescribed total optimizer U;5% warmup,cosine,5% floor; wall only accounting/deadline',
        'primary_comparisons':['D1-D0 same unique data extended schedule/exposure',
            'D2-D1 same label exposures more unique data; tails may change updates',
            'T3-T4 same non-tau rows/exposures within each design'],
        'checkpoint_policy':{'primary':'fixed prescribed endpoint','secondary':'minimum own-task NLL on fixed4k screen; no full-Tune checkpoint selection'},
        'screen_every_non_tau_exposures':75000,'full_Tune':'same40k at endpoint and best-screen only',
        'diagnostics':['rolling class proportions,logit means,NLL','screen per-class recall,AP,AUROC and raw probability means',
            'fixed observed occupancy groups','internal activations,gradients,updates','Train/Tune gaps and plateau'],
        'expansion':'after actual D0/D1/D2, inspect convergence,overfit and cost; conditional D3,D4,width32 or native-family control only for a concrete unresolved question',
        'concurrency':'profile1/2 concurrent real-input workers; choose aggregate throughput subject to memory and valid learning; shared compressed staging per allocation',
        'resources':'earliest compatible HTCondor/Kubeflow/full/shared/MIG; heterogeneous cost and concurrent count recorded',
        'planned_initial_stage_hours':18,'bootstrap':{'replicates':400,'unit':'same frozen physical blocks as R1; paired'},
        'Cal_A_Cut_B_Confirm_payload_read':False})


def preflight(root, path):
    config = json.loads(path.read_text())
    destination = Path(config['preflight'])
    if destination.exists():
        raise FileExistsError('sampler preflight is already frozen')
    with tempfile.TemporaryDirectory(prefix='r2-preflight-') as scratch:
        data = R1Dataset(Path(config['data_root']),'train',config,cache_dir=scratch)
        sampler = SpatialBatchSampler(data,config['batch_size'],config['seed'])
        report,sizes,counts = sampler_trace(sampler,data.metadata.label.to_numpy(),config['passes'])
        lr = np.array([learning_rate(i,len(sizes),config['learning_rate']) for i in range(len(sizes))])
        report.update(status='complete',name=config['name'],data_root=config['data_root'],
            lr_sum=float(lr.sum()),lr_first=float(lr[0]),lr_last=float(lr[-1]),lr_max=float(lr.max()),
            actual_precision_recipe=config['precision'],
            Cal_A_Cut_B_Confirm_payload_read=False)
        destination.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(destination.with_suffix('.npz'),batch_size=sizes,class_counts=counts,learning_rate=lr)
        write_json(destination,report)
        data.native.close()
    print(json.dumps({k:v for k,v in report.items() if k not in ['windows','per_pass']}),flush=True)


def extend(root,design,reference_design,reason_file,passes=None,width=None,family=None):
    """Freeze only an explicitly justified follow-up, with matched-data references."""
    campaign=json.loads((root/'campaign.json').read_text())
    reason=reason_file.read_text().strip()
    if not reason:raise ValueError('extension requires a recorded scientific reason')
    configs=[json.loads(p.read_text()) for p in (root/'configs').glob('*.json')]
    references=[c for c in configs if c.get('design')==reference_design]
    if {c['task'] for c in references}!={'T3','T4'} or len(references)!=2:
        raise ValueError('reference design must identify exactly one completed T3/T4 pair')
    for reference in references:
        path=root/'experiments'/reference['name']/'manifest.json'
        if not path.exists() or json.loads(path.read_text())['status']!='complete':
            raise ValueError('freeze extension only after reference endpoints exist')
    new=[]
    for reference in sorted(references,key=lambda c:c['task']):
        target_family=family or reference['family']
        if target_family not in ['B1','B2','B3']:
            raise ValueError('R2 extension adapter currently supports B1/B2/B3')
        config=(config_from_r1(Path(campaign['r1']),target_family,reference['task'])
            if target_family!=reference['family'] else reference.copy())
        name=f'{target_family.lower()}_{design.lower()}_{reference["task"].lower()}_s42'
        path=root/'configs'/f'{name}.json'
        if path.exists():raise FileExistsError('extension configuration already frozen')
        config.update(name=name,design=design,data_root=reference['data_root'],pool_events=reference['pool_events'],
            passes=reference['passes'] if passes is None else passes,tune_screen=reference['tune_screen'],
            validation_non_tau_exposures=reference['validation_non_tau_exposures'],
            preflight=str(root/'preflight'/f'{name}.json'),extension_reference=reference['name'])
        if width is not None:
            if target_family!='B1':raise ValueError('cnn-width applies only to B1')
            config['cnn_width']=width
        new.append((path,config))
    for path,config in new:write_json(path,config)
    campaign.setdefault('extension_models',[]).extend(c['name'] for _,c in new)
    campaign.setdefault('extensions',[]).append({'design':design,'reference_design':reference_design,
        'reason_source':str(reason_file),'reason':reason,'models':[c['name'] for _,c in new],
        'UTC':datetime.now(timezone.utc).isoformat()})
    for _,config in new:
        campaign.setdefault('extension_pairs',[]).append({'a':config['name'],'b':config['extension_reference'],
            'reason':f'{design}_minus_{reference_design}','domains':['all','pixel_centered','large_scintillator','occupancy_0_100']})
    campaign['extension_pairs'].append({'a':new[0][1]['name'],'b':new[1][1]['name'],
        'reason':f'{design}_T3_minus_conditional_T4','domains':['all','pixel_centered','large_scintillator','occupancy_0_100']})
    write_json(root/'campaign.json',campaign)
    print(json.dumps({'status':'frozen_extension','models':[c['name'] for _,c in new]}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['freeze','preflight','extend'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--r1',type=Path)
    parser.add_argument('--configs',type=Path,nargs='*')
    parser.add_argument('--design')
    parser.add_argument('--reference-design')
    parser.add_argument('--reason-file',type=Path)
    parser.add_argument('--passes',type=int)
    parser.add_argument('--cnn-width',type=int)
    parser.add_argument('--family',choices=['B1','B2','B3'])
    args=parser.parse_args()
    if args.command=='freeze':freeze(args.root,args.r1)
    elif args.command=='extend':
        extend(args.root,args.design,args.reference_design,args.reason_file,args.passes,args.cnn_width,args.family)
    else:
        for path in args.configs:preflight(args.root,path)


if __name__=='__main__':
    main()
