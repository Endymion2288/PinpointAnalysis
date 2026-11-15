"""Architecture-matched Train-only fits for frozen R3 interventions."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from analysis.flavour.v2.r1.fit import run
from analysis.flavour.v2.r3.data import R3Dataset, collate
from analysis.flavour.v2.r3.models import build_model, branch_groups


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--attempt',required=True)
    parser.add_argument('--fp32',action='store_true')
    parser.add_argument('--tasks',nargs='+',choices=['T3','T4'])
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    campaign=json.loads((args.root/'campaign.json').read_text())
    bf16=config.get('bf16',True) and not args.fp32
    settings=SimpleNamespace(root=args.root,r0=Path(campaign['r0']),attempt=args.attempt,
        families=[config['family']],tasks=args.tasks or [config['task']],compile_local=False,
        cnn_variant=config.get('cnn_variant','resnet'),cnn_width=config.get('cnn_width',16),
        dim=config.get('dim',64),bf16=bf16,batch_size=config['batch_size'],
        learning_rate=.0005,max_steps=512)
    def transform(base):
        # Preserve requested task; the small-sample optimizer and informative
        # rows deliberately differ from the large-sample generalization fit.
        task=base['task']
        result={**base,**config,'task':task,'cache_training_inputs':False,
            'bf16':bf16,'precision':'BF16_encoder_FP32_head' if bf16 else 'FP32'}
        return result
    run(settings,config_transform=transform,model_factory=build_model,
        group_factory=branch_groups,dataset_factory=R3Dataset,collate_fn=collate)


if __name__=='__main__':
    main()
