"""Run the frozen small Tune condition and modality-reliance comparisons."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from analysis.flavour.inventory import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--family', required=True)
    parser.add_argument('--compile-local', action='store_true')
    args = parser.parse_args()
    if args.family == 'B0':
        selected = json.loads((args.root/'b0_selected.json').read_text())['selected']
        models = [selected[task]['seed42_model'] for task in ['T3', 'T4']]
    else:
        models = [f'final_{args.family.lower()}_{task.lower()}_s42'
                  for task in (['T4'] if args.family == 'B4-MAE' else ['T3', 'T4'])]
    completed = []
    for model in models:
        for condition, drop in [('nominal', None), ('noise', None), ('combined', None),
                                ('nominal', 'pixel'), ('nominal', 'tracker')]:
            suffix = condition if drop is None else 'remove_'+drop
            name = f'{model}_{suffix}'
            manifest = args.root/'diagnostics'/name/'manifest.json'
            if not (manifest.exists() and json.loads(manifest.read_text())['status'] == 'complete'):
                command = [sys.executable, '-m', 'analysis.flavour.v2.r1.infer', '--root', str(args.root),
                    '--model', model, '--input-root', str(args.root/'stress'/condition), '--name', name]
                if drop is not None:
                    command += ['--drop-modality', drop]
                if args.compile_local:
                    command.append('--compile-local')
                subprocess.run(command, check=True)
            completed.append(name)
    write_json(args.root/'runtime'/f'diagnostics_{args.family}.json', {'status': 'complete',
        'models': models, 'diagnostics': completed,
        'interpretation': 'frozen-weight input removal measures reliance; retrained controls measure incremental value'})


if __name__ == '__main__':
    main()
