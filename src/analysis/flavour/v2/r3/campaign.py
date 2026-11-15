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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['freeze','preflight'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--r2',type=Path)
    parser.add_argument('--configs',type=Path,nargs='*')
    args=parser.parse_args()
    if args.command=='freeze':
        freeze(args.root,args.r2)
    else:
        for path in args.configs:
            preflight(args.root,path)


if __name__=='__main__':
    main()
