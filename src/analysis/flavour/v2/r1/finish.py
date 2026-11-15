"""Submit the remaining R1 operations only after their measured dependencies finish."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from analysis.flavour.inventory import write_json
from analysis.flavour.v2.r1.campaign import FAMILIES
from analysis.flavour.v2.r1.deliver import evaluation_config, ranking


def complete(root, name):
    path = root/'experiments'/name/'manifest.json'
    return path.exists() and json.loads(path.read_text()).get('status') == 'complete'


def submit(root, name, module, arguments, gpu=False, cpus=4, memory=None, flavour='tomorrow'):
    record = root/'runtime'/f'{name}.submission.json'
    if record.exists() and json.loads(record.read_text()).get('submitted'):
        return
    # 15 GiB covers measured R1 peaks (12,647 MiB) and maps to 5 CPU at CERN.
    # A 16 GiB request maps to 6 CPU / 18 GB, excluding 17 GB MIG hosts.
    memory = memory or ('15GB' if gpu else '16GB')
    command = [sys.executable, '-m', 'analysis.flavour.v2.submit', '--stage', 'R1', '--root', str(root),
        '--name', name, '--cpus', str(cpus), '--memory', memory, '--job-flavour', flavour]
    if gpu:
        command += ['--gpu', '--min-capability', '7.5']
    subprocess.run(command+[module, '--root', str(root), *arguments], check=True)


def advance(root):
    for family in [*FAMILIES, 'B4-MAE']:
        tasks = ['T4'] if family=='B4-MAE' else ['T3', 'T4']
        if all(complete(root, f'final_{family.lower()}_{task.lower()}_s42') for task in tasks):
            submit(root, f'r1_diagnostics_{family.lower()}_a1', 'analysis.flavour.v2.r1.diagnostics',
                ['--family', family, '--compile-local'], gpu=True)
    ranking_path = root/'architecture_ranking.json'
    if not ranking_path.exists() and all(complete(root, f'final_{f.lower()}_{t.lower()}_s42')
                                        for f in FAMILIES for t in ['T3', 'T4']):
        ranking(root)
    if not ranking_path.exists():
        return False
    ranks = json.loads(ranking_path.read_text())
    expected = []
    for family in FAMILIES:
        for seed in ([17,42,2026] if family in ranks['expanded_families'] else [42]):
            if family!='B4' and seed!=42:
                submit(root, f'r1_expand_{family.lower()}_s{seed}_a1', 'analysis.flavour.v2.r1.extend',
                    ['--family', family, '--seed', str(seed), '--compile-local'], gpu=True,
                    flavour='nextweek' if family=='B3' else 'tomorrow')
            expected += [f'final_{family.lower()}_{t.lower()}_s{seed}' for t in ['T3', 'T4']]
            expected.append(f'compute_{family.lower()}_t3_s{seed}')
    expected += ['final_b4-mae_t4_s42', 'final_b4_t3_s42_without_tracker',
                 'final_b1_t3_s42_without_tracker', 'control_b1_cnn_t3_s42', 'control_b1_cnn_t4_s42']
    b1 = root/'experiments/final_b1_t4_s42/manifest.json'
    b1_without = root/'experiments/final_b1_t3_s42_without_tracker/manifest.json'
    if b1.exists() and b1_without.exists():
        full = json.loads(b1.read_text())['config'].get('bf16', True)
        control = json.loads(b1_without.read_text())['config'].get('bf16', True)
        if full!=control:
            expected.append('final_b1_t3_s42_without_tracker_'+('bf16' if full else 'fp32'))
    if not all(complete(root, name) for name in expected):
        return False
    config_path = root/'evaluation_config.json'
    if not config_path.exists():
        evaluation_config(root)
    submit(root, 'r1_evaluate_a1', 'analysis.flavour.v2.r1.evaluate',
        ['--config', str(config_path), '--replicates', '400', '--workers', '8'], cpus=8, memory='16GB')
    config = json.loads(config_path.read_text())
    names = [e['name'] for e in config['models'] if e.get('kind', 'main')=='main' and e['seed']==42]
    paths = [root/'diagnostics'/f'{name}_{condition}'/'manifest.json' for name in names
             for condition in ['nominal', 'noise', 'combined', 'remove_pixel', 'remove_tracker']]
    if not all(p.exists() and json.loads(p.read_text()).get('status')=='complete' for p in paths):
        return False
    submit(root, 'r1_stress_evaluate_a1', 'analysis.flavour.v2.r1.stress_evaluate',
        ['--replicates', '400'], cpus=1, memory='3GB')
    results = [root/'evaluation'/f'{name}.json' for name in ['results', 'stress_results']]
    if not all(p.exists() and json.loads(p.read_text()).get('status')=='complete' for p in results):
        return False
    submit(root, 'r1_independent_full_a1', 'analysis.flavour.v2.r1.review',
        ['--require-full'], cpus=1, memory='3GB', flavour='microcentury')
    submit(root, 'r1_figures_a1', 'analysis.flavour.v2.r1.plots', ['--results'], cpus=2, memory='8GB')
    write_json(root/'runtime/finish_dependencies.json', {'status': 'submitted',
        'scope': 'dependent batch operations submitted; final acceptance, interpretation and backend closure remain'})
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    started = time.monotonic()
    while True:
        if advance(args.root) or args.once:
            return
        if time.monotonic()-started > 7*86400:
            raise TimeoutError('R1 dependencies require inspection')
        time.sleep(30)


if __name__ == '__main__':
    main()
