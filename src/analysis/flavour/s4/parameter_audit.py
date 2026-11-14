"""Compare selected and final encoder parameters with their actual initialization."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import torch

from analysis.flavour.inventory import write_json
from analysis.flavour.s4.train import build_model


def inspect(root, name):
    directory = root / 'experiments' / name
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError(name + ': completed fit required for parameter audit')
    config = manifest['config']
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.get('seed', 42))
        initial = build_model({**config, 'compile_local': False})
    parameters = dict(initial.named_parameters())
    rows, checks = {}, {}
    for checkpoint in ['best', 'last']:
        state = torch.load(directory / (checkpoint + '.pt'), map_location='cpu', weights_only=False)
        if state['config'] != config:
            raise ValueError(name + ': checkpoint configuration differs from completed fit')
        rows[checkpoint] = {'epoch': state['epoch'], 'branches': {}}
        for prefix in ['input', 'local', 'local_pool']:
            keys = [key for key in parameters if key.startswith(prefix + '.')]
            if not keys:
                raise ValueError(name + ': expected encoder branch absent: ' + prefix)
            differences = [state['model'][key] - parameters[key].detach() for key in keys]
            maximum = max(float(value.abs().max()) for value in differences)
            finite = all(bool(torch.isfinite(value).all()) for value in differences)
            rows[checkpoint]['branches'][prefix] = {
                'maximum_absolute_change': maximum,
                'L2_change': sum(float(value.double().square().sum()) for value in differences) ** .5,
                'changed_parameter_elements': sum(int(torch.count_nonzero(value)) for value in differences),
                'parameter_elements': sum(value.numel() for value in differences),
                'finite': finite,
            }
            checks[checkpoint + '_' + prefix + '_updated'] = finite and maximum > 0
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'name': name, 'config': config,
        'checkpoints': rows, 'checks': checks, 'passed': all(checks.values()),
        'initialization': {'seed': config.get('seed', 42),
            'pretrained_checkpoint': config.get('pretrained_checkpoint'),
            'supervised_initial_checkpoint': config.get('supervised_initial_checkpoint')},
        'scope': 'CPU parameter differences to seed-matched build_model or exact loaded parent; '
            'compilation disabled only for initialization; complements actual training gradients, '
            'does not establish prediction quality or update of every parameter',
        'code_commit': os.environ.get('PINPOINT_CODE_COMMIT')}
    write_json(directory / 'parameter_update_audit.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--task', type=int, required=True)
    args = parser.parse_args()
    config = json.loads(args.campaign.read_text())['experiments'][args.task]
    output = args.root / 'experiments' / config['name']
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    deadline = time.monotonic() + config.get('wait_for_completion_seconds', 0)
    while True:
        pending = []
        for name in config['targets']:
            path = args.root / 'experiments' / name / 'manifest.json'
            status = json.loads(path.read_text()).get('status') if path.exists() else None
            if status in {'failed', 'internal_health_failed', 'learning_review_required'}:
                raise RuntimeError(name + ': upstream fit requires review')
            if status != 'complete':
                pending.append(name)
        if not pending:
            break
        write_json(output / 'manifest.json', {'status': 'waiting_for_frozen_fits', 'config': config,
            'pending_fits': pending, 'utc': datetime.now(timezone.utc).isoformat()})
        if time.monotonic() >= deadline:
            raise TimeoutError('bounded wait for parameter audit expired')
        time.sleep(15)
    reports = [inspect(args.root, name) for name in config['targets']]
    passed = all(row['passed'] for row in reports)
    write_json(output / 'summary.json', {'models': reports, 'passed': passed})
    write_json(output / 'manifest.json', {'status': 'complete' if passed else 'internal_health_failed',
        'config': config, 'all_checks_pass': passed, 'GPU': None,
        'utc': datetime.now(timezone.utc).isoformat(), 'code_commit': os.environ.get('PINPOINT_CODE_COMMIT')})
    if not passed:
        raise RuntimeError('encoder parameter update audit failed; review actual training')


if __name__ == '__main__':
    main()
