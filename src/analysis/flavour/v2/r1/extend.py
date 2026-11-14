"""Train a family/task pair on the first available GPU, including compute control."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from analysis.flavour.inventory import write_json
from analysis.flavour.v2.r1.config import recipe


def execute(root, config):
    output = root/'experiments'/config['name']
    if (output/'manifest.json').exists() and json.loads((output/'manifest.json').read_text())['status'] == 'complete':
        return json.loads((output/'manifest.json').read_text())
    path = root/'configs'/f"{config['name']}.json"
    write_json(path, config)
    command = [sys.executable, '-m', 'analysis.flavour.v2.r1.train', '--root', str(root), '--config', str(path)]
    if config.get('compile_local'):
        command.append('--compile-local')
    subprocess.run(command, check=True)
    return json.loads((output/'manifest.json').read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--family', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--drop-modality', choices=['pixel', 'bar', 'tracker'])
    parser.add_argument('--compile-local', action='store_true')
    args = parser.parse_args()
    contract = json.loads((args.root/'campaign.json').read_text())
    selection = json.loads((args.root/'selected_recipes.json').read_text())['selected']
    r0 = Path(contract['r0'])
    family = args.family
    base_family = 'B4' if family == 'B4-MAE' else family
    tasks = ['T3'] if args.drop_modality else (['T4'] if family == 'B4-MAE' else ['T4', 'T3'])
    reference = None
    for task in tasks:
        config = recipe(r0, family, task, seed=args.seed, compile_local=args.compile_local)
        suffix = f'_without_{args.drop_modality}' if args.drop_modality else ''
        name = f'final_{family.lower()}_{task.lower()}_s{args.seed}{suffix}'
        config.update(name=name, data_root=str(args.root/'data'),
            learning_rate=selection[f'{base_family}_{task}']['learning_rate'], passes=4,
            drop_modality=args.drop_modality)
        result = execute(args.root, config)
        if task == 'T4':
            reference = result
        if task == 'T3' and reference is not None:
            matched = dict(config)
            matched.update(name=f'compute_{family.lower()}_t3_s{args.seed}',
                total_compute_seconds=reference['wall_seconds'],
                final_inference_estimate_seconds=reference['full_Tune_inference_seconds'],
                compute_reference=name.replace('_t3_', '_t4_'),
                compute_matching='same GPU allocation, fresh initialization, total operation wall budget including loader, training and validation; realized tolerance reported',
                screen_only=False)
            execute(args.root, matched)
    write_json(args.root/'runtime'/f'extended_{family}_s{args.seed}{suffix}.json', {
        'status': 'complete', 'family': family, 'seed': args.seed, 'drop_modality': args.drop_modality,
        'same_allocation_tasks': tasks, 'Cal_A_Cut_B_Confirm_payload_read': False})


if __name__ == '__main__':
    main()
