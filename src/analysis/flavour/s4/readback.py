"""CPU integrity and measured learning checks for completed SSL fits."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis.flavour.inventory import write_json
from analysis.flavour.s2.readback import finite_tensors


def inspect(root,name):
    output=root/'experiments'/name
    manifest=json.loads((output/'manifest.json').read_text())
    config=manifest['config']
    best=torch.load(output/'best.pt',map_location='cpu',weights_only=False)
    last=torch.load(output/'last.pt',map_location='cpu',weights_only=False)
    history=json.loads((output/'history.json').read_text())
    initial=json.loads((output/'initial_reconstruction.json').read_text())
    measured=json.loads((output/'reconstruction.json').read_text())
    semantic_path=output/'semantic_readback.json'
    semantic=json.loads(semantic_path.read_text()) if semantic_path.exists() else None
    health=json.loads((output/'training_health.json').read_text())['records']
    predictions=pd.read_parquet(output/'query_predictions.parquet')
    selected=min(history,key=lambda row:row['inner_reconstruction_probe']['loss']['loss'])
    actual_steps=config['epochs']*manifest['steps_per_epoch']
    sample_aps={}
    for level in range(3):
        value=measured['queries'][f'level{level}_all']
        sample_aps[str(level)]={'AP':value['AP'],'prevalence':value['prevalence'],'positives':value['positives'],
            'AP_over_prevalence':value['AP']/max(value['prevalence'],1e-30) if value['AP'] is not None else None,
            'initial_AP':initial['queries'][f'level{level}_all']['AP']}
    global_stats=measured['global_features']
    required=['optimizer','scheduler','random_state','numpy_state','torch_state','cuda_states','history','health']
    checks={'complete_epochs_and_steps':manifest['status']=='complete' and last['epoch']==config['epochs']
            and [r['epoch'] for r in history]==list(range(1,config['epochs']+1)) and last['steps']==manifest['optimizer_steps']==actual_steps,
        'frozen_configuration':best['config']==last['config']==config,
        'checkpoint_optimizer_finite':finite_tensors(best['model']) and finite_tensors(last['model']) and finite_tensors(last['optimizer']),
        'resume_fields':all(k in last for k in required),
        'inner_probe_selection':best['epoch']==selected['epoch'] and np.isclose(manifest['best_inner_loss'],selected['inner_reconstruction_probe']['loss']['loss']),
        'query_predictions_finite':bool(np.isfinite(predictions.to_numpy()).all()),
        'positive_negative_candidate_support':all((r['positives']>0 and r['positives']<measured['queries'][f'level{level}_all']['queries'])
            for level,r in enumerate([measured['queries'][f'level{i}_all'] for i in range(3)])),
        'all_internal_health_pass':bool(health) and all(r['healthy'] for r in health),
        'global_event_features_nonconstant':global_stats['all_finite'] and global_stats['between_event_RMS']>1e-4,
        'encoder_informative_gradients':all(r['gradient_squared_norms'].get('encoder.input',0)>0 for r in health if r['informative_nodes']>0)}
    # This is a measured learning gate, separate from finite/readable artifacts.
    # Pixel-native positives may be rare; report their count and do not pretend
    # a few examples provide a precise occupancy-performance estimate.
    informative=[v for k,v in sample_aps.items() if int(k)>0 and v['positives']>=25]
    learning={'coarse_and_bar_AP_above_query_prior':bool(informative) and all(v['AP_over_prevalence']>1.05 for v in informative),
        'coarse_or_bar_AP_improves_initialization':any(v['AP']>max(v['initial_AP'] or 0.,v['prevalence']) for v in informative),
        'inner_reconstruction_loss_improves':measured['loss']['reconstruction']<initial['loss']['reconstruction']}
    if config.get('pixel_query_patch',1)>1:
        fine=sample_aps['0']
        learning['fine_pixel_patch_learns_with_at_least_25_positives']=(fine['positives']>=25
            and fine['AP_over_prevalence'] is not None and fine['AP_over_prevalence']>1.05)
    if config.get('objective')=='mae_rel':
        rel=measured.get('bar_rel',{});before=initial.get('bar_rel',{})
        if semantic and not rel:
            rel=semantic['trained'];before=semantic['initial']
        learning['bar_Rel_cross_entropy_improves_initialization']=(rel.get('valid_channels',0)>0
            and rel.get('cross_entropy',float('inf'))<before.get('cross_entropy',-float('inf')))
        learning['bar_Rel_beats_oracle_constant_probe_prior']=(rel.get('cross_entropy',float('inf'))
            <rel.get('constant_probe_prior_cross_entropy',-float('inf')))
    if config.get('objective')=='distill' and 'student_projection_features' in measured:
        # Exclude gross dimensional collapse before scaling. Passing this
        # diagnostic does not establish useful features: frozen probes and
        # paired downstream fine-tuning remain necessary.
        for key in ['student_projection_features','teacher_features']:
            spectrum=measured[key]
            learning[key+'_not_near_rank_one']=(spectrum['centered_effective_rank']>=2.
                and spectrum['leading_variance_fraction']<.9)
    checks={key:bool(value) for key,value in checks.items()}
    learning={key:bool(value) for key,value in learning.items()}
    report={'utc':datetime.now(timezone.utc).isoformat(),'experiment':name,'checks':checks,
        'integrity_and_internal_health_pass':all(checks.values()),'learning_checks':learning,
        'measured_learning_pass':all(learning.values()),'query_AP':sample_aps,
        'global_feature_statistics':global_stats,'event_permutation_delta_BCE':measured.get('event_permutation_delta_BCE'),
        'bar_rel':measured.get('bar_rel',semantic),
        'scope':'train-only reconstruction probe; downstream benefit remains unmeasured by these checks'}
    write_json(output/'ssl_readback.json',report)
    print(json.dumps(report),flush=True)
    return report


def inspect_semantics(root,name):
    """Read the original Rel pilot checkpoints with the new semantic statistic.

    Keep original BF16 reconstruction selection/results intact. Both semantic
    comparisons use FP32 and the same fixed inner events and geometry queries.
    """
    import pyarrow as pa
    from torch.utils.data import DataLoader,Subset
    from analysis.flavour.s4.dataset import PretrainDataset,collate_pretrain
    from analysis.flavour.s4.models import PretrainingModel
    from analysis.flavour.s4.pretrain import evaluate
    output=root/'experiments'/name
    config=json.loads((output/'manifest.json').read_text())['config']
    if config.get('objective')!='mae_rel':raise ValueError('semantic readback requires partial bar Rel')
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    data=PretrainDataset(root,'validation',config)
    loader=DataLoader(Subset(data,range(min(config.get('reconstruction_probe_events',2000),len(data)))),
        batch_size=config.get('batch_size',16),collate_fn=collate_pretrain,num_workers=0)
    torch.manual_seed(config.get('seed',42));model=PretrainingModel(config).eval()
    initial,_=evaluate(model,loader,torch.device('cpu'),False,config)
    state=torch.load(output/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(state['model'],strict=True)
    trained,_=evaluate(model,loader,torch.device('cpu'),False,config)
    report={'initial':initial['bar_rel'],'trained':trained['bar_rel'],'precision':'FP32 CPU',
        'checkpoint_epoch':state['epoch'],'scope':'same original train-inner probe, no refitting or downstream validation',
        'utc':datetime.now(timezone.utc).isoformat()}
    write_json(output/'semantic_readback.json',report);data.native.close()
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--experiments',nargs='+',required=True)
    parser.add_argument('--semantic-rel',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    if args.semantic_rel:
        for name in args.experiments:inspect_semantics(args.root,name)
    reports=[inspect(args.root,name) for name in args.experiments]
    if not all(r['integrity_and_internal_health_pass'] for r in reports):raise SystemExit('SSL integrity/internal health check failed; diagnose before continuation')


if __name__=='__main__':main()
