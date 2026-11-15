"""Submit CPU reviews when frozen R3 endpoints are ready, while GPU probes run."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from analysis.flavour.inventory import write_json


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def endpoint_ready(experiment):
    """The final manifest is published after both complete prediction tables."""
    return read(experiment/'manifest.json').get('status') == 'complete' and all(
        (experiment/name).is_file() and (experiment/name).stat().st_size > 0
        for name in ['predictions.parquet', 'best_predictions.parquet'])


def evaluation_ready(folder):
    records = [read(folder/name) for name in [
        'results.json', 'independent_points.json', 'independent_core_intervals.json']]
    if any(r.get('status') == 'failed' or r.get('failed') for r in records):
        raise ValueError(f'failed evaluation or independent review: {folder}')
    return all(r.get('status') == 'complete' for r in records)


def submit_once(root, name, module, arguments, cpus=2, memory='6GB'):
    record = read(root/'runtime'/f'{name}.submission.json')
    if record:
        if not record.get('submitted'):
            raise ValueError(f'previous submission failed; inspect {name}, use a new attempt for retry')
        return
    subprocess.run([sys.executable, '-m', 'analysis.flavour.v2.submit',
        '--root', str(root), '--stage', 'R3', '--name', name,
        '--cpus', str(cpus), '--memory', memory, '--job-flavour', 'workday',
        module, *arguments], check=True)


def advance(root, submitter=submit_once):
    contract = read(root/'completion_contract.json')
    if contract.get('stage') != 'R3' or contract.get('status') != 'frozen':
        raise ValueError('a frozen R3 completion contract is required')
    evaluations = {}
    for entry in contract['evaluations']:
        output = entry['output']
        path = root/entry['contract']
        missing = [e['id'] for e in read(path)['experiments']
            if not endpoint_ready(Path(e['path']))]
        done = evaluation_ready(root/output)
        evaluations[output] = {'status': 'complete' if done else 'waiting_endpoints' if missing else 'submitted',
            'missing_endpoints': missing}
        if not missing and not done:
            submitter(root, f'r3_{output}_auto_c1', 'analysis.flavour.v2.r3.evaluate',
                ['--root', str(root), '--contract', str(path), '--output', output,
                 '--replicates', '400', '--workers', '8'], cpus=8, memory='12GB')

    experiments = [root/'experiments'/name for name in contract['formal_models']]
    health = root/contract['health_evidence']
    if read(health).get('status') == 'failed':
        raise ValueError('failed formal optimizer-health review')
    health_done = read(health).get('status') == 'complete'
    if all(endpoint_ready(p) for p in experiments) and not health_done:
        submitter(root, 'r3_health_final_auto_c1', 'analysis.flavour.v2.r3.health_review',
            ['--root', str(root), '--experiments', *map(str, experiments), '--attempt', 'final'])

    scientific = root/contract['scientific_review']
    scientific_done = read(scientific).get('status') == 'complete'
    evaluations_done = all(e['status'] == 'complete' for e in evaluations.values())
    if evaluations_done and not scientific_done:
        submitter(root, 'r3_scientific_review_final_auto_c1', 'analysis.flavour.v2.r3.scientific_review',
            ['--evaluations', *[str(root/e['output']/'results.json') for e in contract['evaluations']],
             '--output', str(scientific)])
    done = evaluations_done and health_done and scientific_done
    status = {'status': 'CPU_reviews_complete' if done else 'waiting_CPU_dependencies',
        'contract_revision': contract.get('revision', 1), 'evaluations': evaluations,
        'health_complete': health_done, 'scientific_review_complete': scientific_done,
        'scope': 'metadata polling and CPU batch submission; GPU probes, scientific decisions, '
            'conditional experiments, figures, terminal costs and acceptance still required',
        'Cal_A_Cut_B_Confirm_payload_read': False}
    state_path = root/'runtime/r3_finish_dependencies.json'
    previous = read(state_path)
    if {k: v for k, v in previous.items() if k != 'UTC'} != status:
        write_json(state_path, {**status, 'UTC': datetime.now(timezone.utc).isoformat()})
        print(json.dumps(status), flush=True)
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    (args.root/'runtime').mkdir(exist_ok=True)
    # One observer owns the deterministic submission names; restarts reuse them.
    with (args.root/'runtime/r3_finish.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        while True:
            if advance(args.root) or not args.watch:
                return
            if time.monotonic()-started > 2*86400:
                raise TimeoutError('inspect unfinished R3 CPU review dependencies')
            time.sleep(30)


if __name__ == '__main__':
    main()
