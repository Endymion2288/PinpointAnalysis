"""Profile a possible replacement allocation before releasing an active writer."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from analysis.flavour.inventory import write_json
from analysis.flavour.v2.r1.config import recipe
from analysis.flavour.v2.r1.extend import execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--family', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--minimum-events-per-second', type=float, required=True)
    parser.add_argument('--profile-steps', type=int, default=512)
    parser.add_argument('--release-timeout', type=float, default=900.)
    parser.add_argument('--compile-local', action='store_true')
    args = parser.parse_args()
    root = args.root
    contract = json.loads((root/'campaign.json').read_text())
    config = recipe(Path(contract['r0']), args.family, 'T4', seed=args.seed,
                    compile_local=args.compile_local)
    name = f'profile_{args.family.lower()}_{args.attempt}'
    config.update(name=name, profile=True, data_root=str(root/'data'),
                  learning_rate=0., profile_steps=args.profile_steps, passes=1)
    measured = execute(root, config)
    record_path = root/'runtime'/f'{args.attempt}.handoff.json'
    release_path = root/'runtime'/f'{args.attempt}.release.json'
    rate = measured['events_per_train_second']
    healthy = all(v['finite'] for h in measured['health'] for v in h['activations'].values())
    record = {'status': 'ready' if healthy and rate >= args.minimum_events_per_second else 'rejected',
              'UTC': datetime.now(timezone.utc).isoformat(), 'family': args.family,
              'seed': args.seed, 'allocation_id': os.environ.get('PINPOINT_JOB_ID'),
              'GPU': measured['GPU'], 'profile': name, 'events_per_train_second': rate,
              'minimum_events_per_second': args.minimum_events_per_second,
              'finite_activations': healthy,
              'profile_scope': 'same full Train input and seed, zero LR, first fixed number of batches; no Tune inference',
              'Cal_A_Cut_B_Confirm_payload_read': False}
    write_json(record_path, record)
    print(json.dumps(record), flush=True)
    if record['status'] == 'rejected':
        return
    started = time.monotonic()
    while not release_path.exists():
        if time.monotonic()-started >= args.release_timeout:
            record.update(status='expired_without_release')
            write_json(record_path, record)
            return
        time.sleep(15)
    release = json.loads(release_path.read_text())
    if not (release['allocation_id'] == record['allocation_id']
            and release['previous_writer_terminal']
            and release['family'] == args.family and release['seed'] == args.seed):
        raise ValueError('handoff release does not match this allocation and retired writer')
    for task in ['t4', 't3']:
        path = root/'experiments'/f'final_{args.family.lower()}_{task}_s{args.seed}'
        if path.exists():
            raise ValueError('archive retired incomplete outputs before releasing the new writer')
    record.update(status='released', release=release)
    write_json(record_path, record)
    command = [sys.executable, '-m', 'analysis.flavour.v2.r1.extend', '--root', str(root),
               '--family', args.family, '--seed', str(args.seed)]
    if args.compile_local:
        command.append('--compile-local')
    subprocess.run(command, check=True)
    record.update(status='complete')
    write_json(record_path, record)


if __name__ == '__main__':
    main()
