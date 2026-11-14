"""Same-backbone S4 full fine-tuning and matched-supervised controls."""
import argparse
import json
from pathlib import Path

import torch

from analysis.flavour.s2.train import run
from analysis.flavour.s3.dataset import collate_planes
from analysis.flavour.s3.health import TrainingHealth
from analysis.flavour.s3.models_repair import PlaneModuleClassifierV2
from analysis.flavour.s3.pilot_repair import optimizer_for
from analysis.flavour.s4.dataset import LabelDataset


def validate_pretraining(config):
    if not config.get('pretrained_checkpoint'):return
    directory=Path(config['pretrained_checkpoint']).parent
    report=json.loads((directory/'manifest.json').read_text())
    if report['status']!='complete' or not report['all_internal_health_pass']:
        raise RuntimeError('pretraining has not completed with internal health')
    readback=json.loads((directory/'ssl_readback.json').read_text())
    if not readback['integrity_and_internal_health_pass'] or not readback['measured_learning_pass']:
        raise RuntimeError('pretraining measured-learning readback required')


def build_model(config):
    if config.get('linear_probe_passes',0) and config.get('supervised_initial_checkpoint'):
        raise ValueError('LP-FT requires fresh classification heads; supervised continuation must not restart LP')
    model=PlaneModuleClassifierV2(config)
    model.head[3].p=config.get('classifier_dropout',.1)
    if config.get('linear_probe_passes',0):
        # During LP, only linear_readout is trainable. The nonlinear path is
        # exactly zero until full fine-tuning, with compatible parameter names.
        torch.nn.init.zeros_(model.head[-1].weight)
        torch.nn.init.zeros_(model.head[-1].bias)
    if config.get('supervised_initial_checkpoint'):
        if config.get('pretrained_checkpoint'):
            raise ValueError('choose one supervised/SSL initialization')
        state=torch.load(config['supervised_initial_checkpoint'],map_location='cpu',weights_only=False)
        model.load_state_dict(state['model'],strict=True)
    if config.get('pretrained_checkpoint'):
        state=torch.load(config['pretrained_checkpoint'],map_location='cpu',weights_only=False)
        source=state['model']
        # The teacher/decoder/projector are never downstream input. Random
        # classifier weights and seed match Scratch, including all branch norms.
        keys={key.removeprefix('encoder.'):value for key,value in source.items() if key.startswith('encoder.')}
        excluded=('head.','linear_readout.','head_input_norm.','branch_norms.','summary.')
        keys={key:value for key,value in keys.items() if not key.startswith(excluded)}
        target=model.state_dict()
        if any(key not in target or value.shape!=target[key].shape for key,value in keys.items()):
            raise ValueError('pretraining/downstream encoder mismatch')
        expected={key for key in target if not key.startswith(excluded)}
        if set(keys)!=expected:raise ValueError('pretraining encoder incomplete')
        target.update(keys);model.load_state_dict(target,strict=True)
    return model


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--task',type=int,required=True)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args();config=json.loads(args.campaign.read_text())['experiments'][args.task]
    pilot_name=config.get('required_pilot','s4_backbone_health_s42')
    pilot=json.loads((args.root/'pilot'/pilot_name/'gate.json').read_text())
    if not pilot['gate_passed']:raise RuntimeError('same-backbone real-event health gate required')
    if config.get('empty_response_bypass') and not pilot['config'].get('empty_response_bypass'):
        raise RuntimeError('empty-response repair requires a matching real-event optimization gate')
    if config.get('required_transfer_probe'):
        probe=json.loads((args.root/'experiments'/config['required_transfer_probe']/'transfer_compilation.json').read_text())
        if not probe['passed']:
            raise RuntimeError('LP-FT requires a passing real CUDA frozen/unfrozen compilation probe')
    validate_pretraining(config)
    extra={'stage':'S4','label_budget':config.get('label_budget'),
            'initialization':config.get('initialization','scratch'),
            'same_backbone':'S3 revision3 multiscale native pixel/bar/tracker' if config.get('empty_response_bypass') else 'S3 revision2 multiscale native pixel/bar/tracker',
            'random_downstream_heads':None if config.get('supervised_initial_checkpoint') else
                'head,linear_readout,head_input_norm,branch_norms,summary',
            'supervised_parent_head_reused':bool(config.get('supervised_initial_checkpoint')),
            'calibration_test_used':False}
    options={}
    if config.get('transfer_recipe'):
        from analysis.flavour.s4.low_label import CompletePassSampler, TrainInnerDataset, TransferAdamW, TransferHealth
        options.update(sampler_factory=CompletePassSampler)
        dataset=TrainInnerDataset if 'selection_fold' in config else LabelDataset
        if 'selection_fold' in config:
            extra.update(selection='train_inner_NLL',
                selection_scope='family-grouped train-inner fold within original nested label budget; no independent validation read',
                original_validation_used=False)
        else:
            extra.update(selection_scope='recipe and training length fixed by train-inner selection')
        health,optimizer=TransferHealth,TransferAdamW
    else:
        dataset,health,optimizer=LabelDataset,TrainingHealth,optimizer_for
    run(args,dataset_class=dataset,model_factory=build_model,collate_fn=collate_planes,
        health_factory=health,optimizer_factory=optimizer,manifest_extra=extra,**options)


if __name__=='__main__':main()
