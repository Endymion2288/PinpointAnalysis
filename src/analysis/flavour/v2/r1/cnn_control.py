"""Complete the roadmap's plain CNN cost reference with the B1 recipe."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

from analysis.flavour.inventory import write_json
from analysis.flavour.v2.r1.config import recipe
from analysis.flavour.v2.r1.extend import execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads((args.root/'campaign.json').read_text())
    r0 = Path(contract['r0'])
    gate = args.root/'fit/cnn_cost_control/summary.json'
    if not gate.exists():
        command = [sys.executable, '-m', 'analysis.flavour.v2.r1.fit', '--root', str(args.root),
            '--r0', str(r0), '--attempt', 'cnn_cost_control', '--families', 'B1',
            '--tasks', 'T3', 'T4', '--cnn-variant', 'base']
        if torch.cuda.get_device_capability()[0] >= 8:
            command.append('--bf16')
        subprocess.run(command, check=True)
    health = json.loads(gate.read_text())
    if health['status'] != 'complete' or len(health['cases']) != 2:
        raise ValueError('plain CNN must pass both real learning gates')
    selection = json.loads((args.root/'selected_recipes.json').read_text())['selected']
    names = []
    for task in ['T4', 'T3']:
        config = recipe(r0, 'B1', task)
        name = f'control_b1_cnn_{task.lower()}_s42'
        config.update(name=name, data_root=str(args.root/'data'), cnn_variant='base',
            learning_rate=selection[f'B1_{task}']['learning_rate'], passes=4,
            control_role='plain CNN cost reference required by R1 roadmap; B1-selected LR, no separate HPO',
            architecture='four plain 3x3 stride-2 convolutions, 24/48/72/96 channels; shared dual-view encoder and tracker late fusion')
        execute(args.root, config)
        names.append(name)
    write_json(args.root/'runtime/cnn_cost_control.json', {'status': 'complete', 'models': names,
        'scope': 'matched-input/four-pass cost reference, separately labeled; frozen main ranking unchanged',
        'Cal_A_Cut_B_Confirm_payload_read': False})


if __name__ == '__main__':
    main()
