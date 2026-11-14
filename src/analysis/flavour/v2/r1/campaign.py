"""Freeze R1 selection rules, screen on one available GPU, and generate extensions."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from analysis.flavour.inventory import write_json
from analysis.flavour.v2.r1.config import recipe


FAMILIES = ['B1', 'B2', 'B3', 'B4']
LEARNING_RATES = {'B1': [.0003, .001], 'B2': [.0001, .0003],
                  'B3': [.0001, .0003], 'B4': [.0003, .001]}


def freeze(args):
    path = args.root/'campaign.json'
    if path.exists():
        raise FileExistsError('R1 campaign contract is immutable')
    profile = {}
    for folder in sorted((args.root/'experiments').glob('profile_*')):
        manifest = folder/'manifest.json'
        if manifest.exists():
            data = json.loads(manifest.read_text())
            if data['status'] == 'complete':
                profile[folder.name] = {key: data.get(key) for key in
                    ['GPU', 'events_per_train_second', 'GPU_peak_MiB', 'parameters', 'counters']}
    if set(data['config']['family'] for folder in sorted((args.root/'experiments').glob('profile_*'))
           if (folder/'manifest.json').exists() for data in [json.loads((folder/'manifest.json').read_text())]
           if data['status'] == 'complete') != set(FAMILIES):
        raise RuntimeError('profile every family successfully before freezing the campaign')
    contract = {'stage': 'R1', 'status': 'frozen_before_first_Tune_score',
        'UTC': datetime.now(timezone.utc).isoformat(), 'r0': str(args.r0),
        'train': 'complete frozen 100k, T3 uses all 75k non-tau rows', 'Tune': 'frozen 40k',
        'screen_Tune': 'frozen family-hash 1k/class; same rows for every trial',
        'families': FAMILIES, 'tasks': ['T3', 'T4'], 'seed_screen': 42,
        'learning_rates': LEARNING_RATES, 'screen_train_seconds_per_trial': args.seconds,
        'HPO': 'four trials/family, same allocated device-minutes; T3/T4 and two LRs within a family share one allocation; heterogeneous devices are not FLOP equivalent',
        'screen_selection': ['maximum common non-tau BA3', 'minimum NLL3 on ties'],
        'architecture_ranking': 'Scratch; arithmetic mean of T3 and T4 conditional BA3, then mean NLL3',
        'extension': {'passes': 4, 'fresh_from_same_initialization': True,
            'schedule': '5% warmup, cosine to 5% initial learning rate; record updates and per-row exposures',
            'compute_control': 'T3 and T4 paired sequentially on one allocation; T3 additional timed control from fresh initialization',
            'minimum_common_exposures': 'four per non-tau Train row'},
        'seeds': [17, 42, 2026], 'seed_expansion': 'two strongest neural families and B4 common reference',
        'B0': {'learning_rates': [.05, .1], 'boosting_iterations': 300, 'seeds': [17, 42, 2026],
            'cost': 'CPU thread and wall seconds separate'},
        'MAE_reference': 'B4-MAE T4, seed42, same supervised recipe and separate historical SSL cost',
        'modality_controls': ['retrain B1 and B4 T3 without tracker at four passes',
            'paired pixel/tracker removal diagnoses reliance; it is not a retrained incremental-value estimate'],
        'domains': ['all', 'pixel_centered', 'large_scintillator', 'active', 'empty', 'tracker_active', 'tracker_empty', 'occupancy'],
        'stress': 'fixed Tune 400/class nominal/noise/combined; accepted S6 amplitudes and paired ideal tracker redigitization',
        'bootstrap': {'replicates': 400, 'unit': 'accepted-family-connected physical source blocks',
            'paired': True, 'training_seed_uncertainty': 'separate per-seed values and sample SD'},
        'budget': {'roadmap_device_hour_envelope': 18,
            'policy': 'priority is early scheduling across GPU models/full-card/MIG; report actual resource by model; update projection from profiles'},
        'profiles': profile, 'Cal_A_Cut_B_Confirm_payload_read': False,
        'collaboration': 'homologous model and explicit geometry cuts; exact original weights/run/cuts unavailable'}
    write_json(path, contract)


def screen(args):
    contract = json.loads((args.root/'campaign.json').read_text())
    root0 = Path(contract['r0'])
    for family in args.families:
        for task in ['T3', 'T4']:
            for lr in contract['learning_rates'][family]:
                name = f'screen_{family.lower()}_{task.lower()}_lr{lr:g}_s42'
                manifest = args.root/'experiments'/name/'manifest.json'
                if manifest.exists() and json.loads(manifest.read_text())['status'] == 'complete':
                    continue
                config = recipe(root0, family, task, compile_local=args.compile_local)
                config.update(name=name, data_root=str(args.root/'data'), learning_rate=lr,
                    screen_only=True, screen_seconds=contract['screen_train_seconds_per_trial'], passes=4)
                path = args.root/'configs'/f'{name}.json'
                write_json(path, config)
                command = [sys.executable, '-m', 'analysis.flavour.v2.r1.train', '--root', str(args.root), '--config', str(path)]
                if config['compile_local']:
                    command.append('--compile-local')
                subprocess.run(command, check=True)
    write_json(args.root/f"screen_complete_{'_'.join(args.families)}.json", {'status': 'complete',
        'GPU_alignment': 'all task/LR trials within each submitted group share one allocation'})


def select(args):
    contract = json.loads((args.root/'campaign.json').read_text())
    selected = {}
    for family in FAMILIES:
        for task in ['T3', 'T4']:
            candidates = []
            for lr in contract['learning_rates'][family]:
                name = f'screen_{family.lower()}_{task.lower()}_lr{lr:g}_s42'
                manifest = json.loads((args.root/'experiments'/name/'manifest.json').read_text())
                if manifest['status'] != 'complete':
                    raise RuntimeError('screen incomplete')
                metric = manifest['metrics']['T3']
                candidates.append((metric['BA'], -metric['NLL'], name, manifest))
            winner = max(candidates, key=lambda value: value[:2])
            selected[f'{family}_{task}'] = {'screen': winner[2], 'metrics': winner[3]['metrics'],
                'learning_rate': winner[3]['config']['learning_rate']}
    write_json(args.root/'selected_recipes.json', {'status': 'complete', 'selected': selected,
        'selection_rule': contract['screen_selection']})
    for family in FAMILIES+['B4-MAE']:
        for task in (['T4'] if family == 'B4-MAE' else ['T3', 'T4']):
            name = f'final_{family.lower()}_{task.lower()}_s42'
            key = f"{'B4' if family == 'B4-MAE' else family}_{task}"
            config = recipe(Path(contract['r0']), family, task, compile_local=True)
            config.update(name=name, data_root=str(args.root/'data'), learning_rate=selected[key]['learning_rate'], passes=4)
            write_json(args.root/'configs'/f'{name}.json', config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['freeze', 'screen', 'select'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--r0', type=Path)
    parser.add_argument('--seconds', type=float, default=300.)
    parser.add_argument('--compile-local', action='store_true')
    parser.add_argument('--families', nargs='+', default=FAMILIES)
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == '__main__':
    main()
