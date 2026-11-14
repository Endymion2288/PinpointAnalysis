"""Fit three nested readouts on CPU after a frozen GPU feature bank is ready."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

from analysis.flavour.inventory import write_json
from analysis.flavour.s4.probes import main as fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--task', type=int, required=True)
    args = parser.parse_args(); config = json.loads(args.campaign.read_text())['experiments'][args.task]
    output = args.root / 'experiments' / config['name']; output.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + config.get('wait_for_completion_seconds', 10800)
    while True:
        source = args.root / 'experiments' / config['feature_source'] / 'manifest.json'
        status = json.loads(source.read_text()).get('status') if source.exists() else None
        if status == 'complete': break
        if status in {'failed', 'internal_health_failed', 'learning_review_required'}:
            raise RuntimeError('frozen feature source failed')
        write_json(output / 'manifest.json', {'status': 'waiting_for_feature_bank', 'config': config,
            'utc': datetime.now(timezone.utc).isoformat()})
        if time.monotonic() >= deadline: raise TimeoutError('bounded wait for frozen features expired')
        time.sleep(15)
    for task in config['tasks']:
        original = sys.argv
        try:
            sys.argv = ['probe_fit', '--root', str(args.root), '--campaign', config['readout_campaign'],
                '--task', str(task), '--device', 'cpu']
            fit()
        finally:
            sys.argv = original
    write_json(output / 'manifest.json', {'status': 'complete', 'config': config, 'GPU': None,
        'utc': datetime.now(timezone.utc).isoformat()})


if __name__ == '__main__':
    main()
