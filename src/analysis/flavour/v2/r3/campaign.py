"""Freeze R3 work items and exact exposure/order controls before fitting."""

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
from analysis.flavour.v2.r3.sampling import GlobalBatchSampler
from analysis.flavour.v2.r3.sensor_rows import sensor_row_geometry


def freeze(root, r2):
    accepted = json.loads((r2/'acceptance.json').read_text())
    if accepted['status'] != 'complete':
        raise ValueError('R2 acceptance must be complete')
    if (root/'campaign.json').exists():
        raise FileExistsError('R3 campaign already frozen')
    parent = json.loads((r2/'campaign.json').read_text())
    name = 'b1_m0_t4_s42'
    config = json.loads((r2/'configs/b1_d2_t4_s42.json').read_text())
    config.update(name=name, design='M0', event_mixing='global', order_seed=20261119,
        cache_training_inputs=True, preflight=str(root/'preflight'/f'{name}.json'),
        reference_experiment=str(r2/'experiments/b1_d2_t4_s42'),
        scientific_change='global event reshuffling each pass; original per-update batch sizes and LR')
    write_json(root/'configs'/f'{name}.json', config)
    write_json(root/'campaign.json', {
        'stage':'R3','status':'first_M0_contract_frozen_before_R3_Tune_access',
        'UTC':datetime.now(timezone.utc).isoformat(), 'r2':str(r2),
        'r1':parent['r1'],'r0':parent['r0'],'initial_models':[name],
        'work_items':{
            'M0':'fresh D2 T4, 300k x4, global label-independent order only; same update sizes,U=150117,LR',
            'G0':'observation-conserving padding phase; synthetic points/lines and fixed real Train rows',
            'G1':'B1 first stride4 to2 only; same adopted event mixing on both sides',
            'G2':'within-native-family control: signed,measurement-aware pre-compression sparse relations',
            'G3':'necessary matched seeds17/2026 and native/100um/1mm for the most valuable change',
        },
        'conditional':'only add seeds,resolution,anchor or late-fusion experiments after the preceding physical/learning/cost evidence; each addition frozen with reason',
        'checkpoint_policy':{'primary':'fixed exposure endpoint','secondary':'minimum own-task fixed4k screen NLL'},
        'evaluation':'same Tune40k,all11 domains,400 paired physical-block draws; per-class,sparse rankings,tau working points and cost',
        'required_probes':'local/regional/global readability,actual training gradients,and retrained ablation; distinguish available,used and useful information',
        'promotion':'unchanged v2 2pp BA3 or1pp tau efficiency with the declared NLL/per-class/common-task constraints',
        'resources':'earliest compatible Condor/Kubeflow full/shared/MIG; choose concurrency by measured operation throughput,RAM/VRAM and full-validation peak; count allocation once',
        'Cal_A_Cut_B_Confirm_payload_read':False,
    })
    (root/'data').symlink_to(r2/'data', target_is_directory=True)


def preflight(root, path):
    config = json.loads(path.read_text())
    destination = Path(config['preflight'])
    if destination.exists():
        raise FileExistsError('preflight already frozen')
    with tempfile.TemporaryDirectory(prefix='r3-preflight-') as scratch:
        # No detector payload is read by either metadata-only sampler.
        data = R1Dataset(Path(config['data_root']), 'train', config, cache_dir=scratch)
        baseline = SpatialBatchSampler(data, config['batch_size'], config['seed'])
        sampler = GlobalBatchSampler(data, config['batch_size'], config['seed']) if config.get('event_mixing') == 'global' else baseline
        original, old_sizes, _ = sampler_trace(baseline, data.metadata.label.to_numpy(), config['passes'])
        result, sizes, classes = sampler_trace(sampler, data.metadata.label.to_numpy(), config['passes'])
        if not np.array_equal(old_sizes, sizes):
            raise ValueError('mixing changed per-update batch sizes')
        lr = np.asarray([learning_rate(i,len(sizes),config['learning_rate']) for i in range(len(sizes))])
        result.update(status='complete',name=config['name'],lr_sum=float(lr.sum()),
            same_original_batch_sizes=True,event_mixing=config.get('event_mixing','spatial'),
            original_total_updates=original['total_updates'],actual_precision_recipe=config['precision'],
            Cal_A_Cut_B_Confirm_payload_read=False)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination.with_suffix('.npz'),batch_size=sizes,class_counts=classes,learning_rate=lr)
        write_json(destination,result)
        data.native.close()
    print(json.dumps({k:v for k,v in result.items() if k not in ['windows','per_pass']}),flush=True)


def extend(root, design, mixing, cache_workers, reason):
    campaign=json.loads((root/'campaign.json').read_text())
    m0=json.loads((root/'evaluation_m0/results.json').read_text())
    if m0['status']!='complete':raise ValueError('freeze geometry fits after the M0 endpoint comparison')
    for name in ['independent_points.json','independent_core_intervals.json']:
        review=json.loads((root/'evaluation_m0'/name).read_text())
        if review['status']!='complete' or review['failed']:raise ValueError('M0 independent metric review must pass before geometry freezing')
    r2,r1=Path(campaign['r2']),Path(campaign['r1'])
    if design=='G1':
        base=json.loads((r2/'configs/b1_d2_t4_s42.json').read_text())
        candidates=[{**base,'name':'b1_g1_stride2_t4_s42','design':'G1','cnn_first_stride':2,
            'reference_experiment':str(root/'experiments/b1_m0_t4_s42') if mixing=='global' else str(r2/'experiments/b1_d2_t4_s42'),
            'scientific_change':'first convolution stride4 to2 only; total stride32 to16'}]
    elif design=='G2':
        base=json.loads((root/'configs/probe_b2_relation_t4_s42_c5.json').read_text())
        candidates=[]
        for variant in ['control','relations','global_ablation']:
            name=f'b2_g2_{variant}_t4_s42'
            related=variant!='control'
            candidates.append({**base,'name':name,'design':'G2_'+variant,
                'data_root':str(r1/'data'),'native_relations':related,
                'build_relations':True,'native_global_ablation':variant=='global_ablation',
                'cache_storage':'mmap',
                'candidate_cache_policy':'identical observation-only candidate arrays may be shared; control does not read edges',
                'scientific_change':{'control':'fresh native control for matched data/sampling/exposure/precision',
                    'relations':'observation-only signed measurement/material-aware messages before native local compression',
                    'global_ablation':'retrain relations model without the inter-block stack; preserve prelocal,block/event pooling and tracker'}[variant]})
    elif design=='C1':
        base=json.loads((r2/'configs/b1_d0_t4_s42.json').read_text())
        plan=json.loads((r1/'data/plan.json').read_text())
        geometry=json.loads(Path(plan['geometry']).read_text())
        candidates=[{**base,'name':'b1_c1_sensor_rows_t4_s42','design':'C1',
            'sensor_row_image':True,'image':sensor_row_geometry(geometry),'cnn_first_stride':2,
            'reference_experiment':str(r2/'experiments/b1_d0_t4_s42'),
            'scientific_change':'compact all actual pixel/bar sensor rows; same horizontal1mm bins, views and responses; explicit z/gaps/type channels; stride2; representation screen, not a stride-only contrast'}]
    elif design=='M0T3':
        if mixing!='global':raise ValueError('M0T3 tests the adopted global order')
        base=json.loads((r2/'configs/b1_d2_t3_s42.json').read_text())
        candidates=[{**base,'name':'b1_m0_t3_s42','design':'M0T3',
            'reference_experiment':str(r2/'experiments/b1_d2_t3_s42'),
            'scientific_change':'global full-pass mixing for the actual T3 task'}]
    else:raise ValueError('unknown R3 extension')
    frozen=[]
    for config in candidates:
        config.update(event_mixing=mixing,order_seed=20261119,cache_training_inputs=True,
            cache_workers=cache_workers,freeze_reason=reason,freeze_UTC=datetime.now(timezone.utc).isoformat(),
            preflight=str(root/'preflight'/f"{config['name']}.json"))
        path=root/'configs'/f"{config['name']}.json"
        if path.exists():raise FileExistsError('formal R3 recipe already frozen')
        write_json(path,config)
        frozen.append(config['name'])
    write_json(root/'contracts'/f'{design.lower()}_freeze.json',{
        'status':'frozen','UTC':datetime.now(timezone.utc).isoformat(),'models':frozen,
        'reason':reason,'event_mixing':mixing,'scope':'fixed fresh exposures and original per-update batch sizes; endpoint primary',
        'Cal_A_Cut_B_Confirm_payload_read':False})
    previous=campaign.get('formal_models',campaign['initial_models'])
    campaign['formal_models']=previous+frozen
    campaign.setdefault('extensions',[]).append({'design':design,'models':frozen,'reason':reason})
    write_json(root/'campaign.json',campaign)
    print(json.dumps({'status':'frozen_extension','models':frozen}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['freeze','preflight','G1','G2','C1','M0T3'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--r2',type=Path)
    parser.add_argument('--configs',type=Path,nargs='*')
    parser.add_argument('--mixing',choices=['global','spatial'])
    parser.add_argument('--cache-workers',type=int,default=2)
    parser.add_argument('--reason')
    args=parser.parse_args()
    if args.command=='freeze':
        freeze(args.root,args.r2)
    elif args.command=='preflight':
        for path in args.configs:
            preflight(args.root,path)
    else:
        if not args.reason or not args.mixing:parser.error('extension requires an evidence reason and selected sampling')
        extend(args.root,args.command,args.mixing,args.cache_workers,args.reason)


if __name__=='__main__':
    main()
