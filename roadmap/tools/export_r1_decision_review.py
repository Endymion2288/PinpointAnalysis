"""Separate R1 argmax recall from class-score operating-point performance."""

import argparse
import json
from pathlib import Path
from statistics import mean, stdev


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.root/'interim/b1_three_seed'
    evaluation = json.loads((source/'evaluation/results.json').read_text())
    config = json.loads((source/'evaluation_config.json').read_text())
    metric_names = ['recall', 'AP', 'AUROC', 'eff_FPR_0.01', 'eff_FPR_0.001',
                    'eff_purity_0.8', 'eff_purity_0.9']
    groups = []
    for family in ['B0', 'B1', 'B4']:
        for task in ['T3', 'T4']:
            entries = sorted([e for e in config['models'] if e['family']==family and e['task']==task
                              and e.get('kind', 'main')=='main'], key=lambda e: e['seed'])
            if [e['seed'] for e in entries] != [17, 42, 2026]:
                raise ValueError('Expected exactly three main-task seeds per group')
            rows = [evaluation['models'][e['name']]['all']['T3']['per_class'][1] for e in entries]
            groups.append({'family': family, 'task': task, 'seeds': [e['seed'] for e in entries],
                'models': [e['name'] for e in entries], 'common_non_tau_muon': {
                    metric: {'mean': mean(row[metric] for row in rows),
                             'sample_SD': stdev(row[metric] for row in rows),
                             'per_seed': [row[metric] for row in rows]} for metric in metric_names}})
    contrasts = [row for row in evaluation['mean_seed_pairs']
                 if row['domain']=='all' and row['a_family_task_kind'][0]=='B1'
                 and row['b_family_task_kind'][0] in ['B0', 'B4']
                 and row['a_family_task_kind'][2]==row['b_family_task_kind'][2]=='main']
    if len(contrasts)!=4:
        raise ValueError('Expected B1 against both references for both training tasks')
    result = {'status': 'complete_interim', 'source': str(source/'evaluation/results.json'),
        'scope': 'post-hoc development comparison; unchanged R1 ranking and promotion gates; no new fitting',
        'groups': groups, 'contrasts': [{**{k: v for k,v in row.items() if k!='metrics'},
            'metrics': {k: v for k,v in row['metrics'].items() if k=='recall_mu' or k.startswith('mu_eff_')}}
            for row in contrasts],
        'uncertainty': 'paired physical-block pointwise intervals conditional on the three fitted seeds and actual runtime recipes',
        'Cal_A_Cut_B_Confirm_payload_read': False}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'groups': len(groups), 'contrasts': len(contrasts), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
