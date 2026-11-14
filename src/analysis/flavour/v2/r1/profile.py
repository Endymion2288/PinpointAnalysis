"""Prepare and run homogeneous-device, Train-only architecture profiles."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from analysis.flavour.inventory import write_json
from analysis.flavour.v2.r1.config import recipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--r0', type=Path, required=True)
    parser.add_argument('--family', required=True)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--compile-local', action='store_true')
    args = parser.parse_args()
    config = recipe(args.r0, args.family, 'T4', compile_local=args.compile_local)
    config.update(name=f'profile_{args.family.lower()}_{args.attempt}', profile=True,
        data_root=str(args.r0/'profile_data'), learning_rate=0., profile_steps=args.steps, passes=2)
    path = args.root/'configs'/f"{config['name']}.json"
    write_json(path, config)
    command = [sys.executable, '-m', 'analysis.flavour.v2.r1.train', '--root', str(args.root), '--config', str(path)]
    if config['compile_local']:
        command.append('--compile-local')
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
