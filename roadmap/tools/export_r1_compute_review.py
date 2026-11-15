"""Export complete B1/B4 three-seed R1 task and same-allocation controls."""

import argparse
import json
from pathlib import Path
from statistics import mean, stdev


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.root/'interim/b1_three_seed_compute'
    evaluation = json.loads((source/'evaluation/results.json').read_text())
    config = json.loads((source/'evaluation_config.json').read_text())
    review = json.loads((source/'evaluation/independent_review.json').read_text())
    if evaluation['status']!='complete' or review['status']!='complete' or review['failed']:
        raise ValueError('Complete evaluation and independent readback required')
    groups, training = [], []
    for family in ['B1', 'B4']:
        for task, kind in [('T3', 'main'), ('T4', 'main'), ('T3', 'matched_compute')]:
            entries = sorted([e for e in config['models'] if e['family']==family and e['task']==task
                              and e.get('kind', 'main')==kind], key=lambda e: e['seed'])
            if [e['seed'] for e in entries] != [17, 42, 2026]:
                raise ValueError('Expected exactly three seeds per group')
            rows = [evaluation['models'][e['name']]['all']['T3'] for e in entries]
            groups.append({'family': family, 'task': task, 'kind': kind,
                'BA3_mean': mean(row['BA'] for row in rows),
                'BA3_sample_SD': stdev(row['BA'] for row in rows),
                'NLL3_mean': mean(row['NLL'] for row in rows),
                'recall_e_mu_NC_mean': [mean(row['per_class'][c]['recall'] for row in rows) for c in range(3)]})
            if kind!='matched_compute':
                continue
            for entry in entries:
                record = json.loads((args.root/'experiments'/entry['name']/'manifest.json').read_text())
                training.append({'name': entry['name'], 'seed': entry['seed'],
                    **{k: record[k] for k in ['GPU', 'allocation_id', 'wall_seconds', 'compute_alignment',
                                               'counters', 'unique_train_events']},
                    'precision': record['config']['precision'], 'health_samples': len(record['health'])})
    result = {'status': 'complete_interim', 'source': str(source/'evaluation/results.json'),
        'models': len(evaluation['models']), 'pairs': len(evaluation['pairs']),
        'replicates': evaluation['replicates'], 'metrics_revision': evaluation['metrics_revision'],
        'scope': 'complete B1/B4/HGB task and compute development subset; full R1 and Confirm pending',
        'groups': groups, 'training_compute_controls': training,
        'mean_seed_compute_pairs': [row for row in evaluation['mean_seed_pairs']
                                   if row['a_family_task_kind'][2]=='matched_compute'],
        'independent_review': {k: v for k,v in review.items() if k not in ['checks', 'models']},
        'uncertainty': 'pointwise physical-block development intervals conditional on fitted seeds and realized runtimes; seed sample SD is separate',
        'Cal_A_Cut_B_Confirm_payload_read': False}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'groups': len(groups), 'pairs': len(result['mean_seed_compute_pairs']),
                      'output': str(args.output)}))


if __name__ == '__main__':
    main()
