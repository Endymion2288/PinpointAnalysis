"""Export accepted R1 development results and explicit research quality criteria."""

import argparse
import json
from pathlib import Path

from analysis.flavour.inventory import write_json


def quality_criteria(metrics, rule, task, actual_fpr_recorded=False):
    """Check candidate-minus-reference effects; intervals condition on fitted seeds."""
    checks = []

    def criterion(name, value, limit, relation):
        operations = {'>=': lambda a, b: a >= b, '>': lambda a, b: a > b,
                      '<=': lambda a, b: a <= b}
        checks.append({'name': name, 'value': value, 'relation': relation, 'limit': limit,
                       'passed': value is not None and operations[relation](value, limit)})

    def point(name):
        return metrics[name]['difference_a_minus_b']

    def bound(name, side):
        return metrics[name]['95pct'][side]

    if task == 'T3':
        criterion('BA3_practical_gain', point('BA3'), rule['delta_BA3_point_min'], '>=')
        criterion('BA3_positive_lower_bound', bound('BA3', 0),
                  rule['primary_difference_95_lower_min_exclusive'], '>')
        criterion('NLL3_upper_bound', bound('NLL3', 1), rule['delta_NLL3_95_upper_max'], '<=')
        for label in ['e', 'mu', 'NC']:
            criterion('recall_'+label+'_lower_bound', bound('recall_'+label, 0),
                      -rule['per_class_recall_loss_95_upper_max'], '>=')
    elif task == 'T4':
        criterion('tau_1pct_practical_gain', point('tau_eff_1pct'),
                  rule['delta_efficiency_at_1pct_point_min'], '>=')
        criterion('tau_1pct_positive_lower_bound', bound('tau_eff_1pct', 0),
                  rule['primary_difference_95_lower_min_exclusive'], '>')
        criterion('BA3_lower_bound', bound('BA3', 0), -rule['BA3_loss_95_upper_max'], '>=')
        criterion('NLL4_upper_bound', bound('NLL4', 1), rule['delta_NLL4_95_upper_max'], '<=')
        if rule.get('actual_FPR_required'):
            criterion('actual_FPR_recorded_for_both_models', bool(actual_fpr_recorded), True, '>=')
    else:
        raise ValueError(task)
    return {'all_criteria_met': all(row['passed'] for row in checks), 'checks': checks,
            'failed': [row['name'] for row in checks if not row['passed']],
            'scope': 'development research criteria; not independent confirmation or a global optimum claim'}


def compact_domain(report):
    result = {key: report[key] for key in ['events', 'non_tau_events', 'raw_counts_original_labels',
        'tau_forced_assignments_e_mu_NC', 'tau_contamination']}
    for task in ['T3', 'T4']:
        if report.get(task) is None:
            continue
        metrics = report[task]
        result[task] = {key: metrics[key] for key in ['BA', 'NLL', 'macro_AP', 'macro_AUROC', 'Neff']}
        observed = report.get(task+'_observed_mixture', metrics)
        result[task].update(raw_confusion_counts=observed['confusion_counts'],
            benchmark_weighted_confusion_counts=metrics['confusion_counts'],
            benchmark_weighted_prior=metrics['weighted_prior'],
            benchmark_weighted_counts=metrics['weighted_counts'])
        result[task]['per_class'] = [{key: row[key] for key in ['recall', 'AP', 'AUROC',
            'eff_FPR_0.01', 'actual_FPR_0.01', 'threshold_FPR_0.01', 'purity_FPR_0.01',
            'coverage_FPR_0.01', 'eff_FPR_0.001', 'actual_FPR_0.001',
            'threshold_FPR_0.001', 'purity_FPR_0.001', 'coverage_FPR_0.001',
            'eff_purity_0.8', 'coverage_purity_0.8', 'eff_purity_0.9', 'coverage_purity_0.9']}
            for row in metrics['per_class']]
    if 'T4_tau_prior_scenarios' in report:
        result['T4_tau_prior_scenarios'] = report['T4_tau_prior_scenarios']
    return result


def summarize(root, plan):
    def read(relative):
        return json.loads((root/relative).read_text())

    acceptance = read('acceptance.json')
    if acceptance['status'] != 'complete' or acceptance['failed']:
        raise ValueError('Final R1 summary requires completed delivery acceptance')
    evaluation = read('evaluation/results.json')
    stress = read('evaluation/stress_results.json')
    config = read('evaluation_config.json')
    entries = {row['name']: row for row in config['models']}
    def fpr_recorded(names):
        return all(evaluation['models'][name]['all'].get('T4', {}).get('per_class', [{}, {}, {}])[2]
                   .get('actual_FPR_0.01') is not None for name in names)
    training = {}
    for name in entries:
        record = read(f'experiments/{name}/manifest.json')
        training[name] = {key: record.get(key) for key in ['GPU', 'allocation_id', 'parameters',
            'wall_seconds', 'GPU_peak_MiB', 'unique_train_events', 'events_per_train_second',
            'counters', 'compute_alignment', 'capability', 'memory', 'probability_diagnostics',
            'fit_seconds', 'threads']}
        training[name]['GPU'] = record.get('GPU') or record.get('device', 'CPU')
        training[name]['precision'] = record['config'].get('precision', 'CPU')
        training[name]['compile_local'] = record['config'].get('compile_local', False)
        training[name]['learning_curve'] = [{
            **{key: row[key] for key in ['pass', 'updates', 'training_exposures', 'train_NLL', 'lr']},
            'Tune_screen': {task: {key: metrics[key] for key in ['BA', 'NLL']}
                            for task, metrics in row.get('Tune_screen', {}).items()}}
            for row in record.get('history', [])]
        health = record.get('health', [])
        training[name]['health_samples'] = len(health)
    research_criteria = []
    for pair in evaluation['pairs']:
        if pair.get('domain', 'all') != 'all':
            continue
        a, b = entries[pair['a']], entries[pair['b']]
        if not (a.get('kind', 'main') == b.get('kind', 'main') == 'main'
                and a['task'] == b['task'] and a['family'] in ['B1', 'B2', 'B3']
                and b['family'] in ['B0', 'B4']):
            continue
        research_criteria.append({'a': pair['a'], 'b': pair['b'], 'task': a['task'],
            **quality_criteria(pair['metrics'], plan['promotion_gates']['T3' if a['task']=='T3' else 'tau'],
                               a['task'], fpr_recorded([pair['a'], pair['b']]) if a['task']=='T4' else False)})
    for pair in evaluation['mean_seed_pairs']:
        a, b = pair['a_family_task_kind'], pair['b_family_task_kind']
        if not (pair['domain']=='all' and a[1]==b[1] and a[2]==b[2]=='main'
                and a[0] in ['B1', 'B2', 'B3'] and b[0] in ['B0', 'B4']):
            continue
        names = [entry['name'] for entry in config['models']
                 if [entry['family'], entry['task'], entry.get('kind', 'main')] in [a, b]
                 and entry['seed'] in pair['seeds']]
        research_criteria.append({'a_family_task_kind': a, 'b_family_task_kind': b,
            'seeds': pair['seeds'], 'task': a[1],
            **quality_criteria(pair['metrics'], plan['promotion_gates']['T3' if a[1]=='T3' else 'tau'],
                               a[1], len(names)==6 and fpr_recorded(names) if a[1]=='T4' else False)})
    stress_compact = {}
    for name, record in stress['models'].items():
        stress_compact[name] = {key: value for key, value in record.items() if key!='conditions'}
        stress_compact[name]['conditions'] = {condition: {
            'BA3': item['metrics']['T3']['BA'], 'NLL3': item['metrics']['T3']['NLL'],
            'paired_descriptive_differences': item['paired_descriptive_differences'],
            'frozen_points': item['frozen_points']}
            for condition, item in record['conditions'].items()}
    return {'status': 'complete', 'scope': evaluation['scope'], 'source_root': str(root),
        'metrics_revision': evaluation['metrics_revision'], 'ranking': read('architecture_ranking.json'),
        'model_config': config, 'training': training,
        'domains': {name: {domain: compact_domain(report) for domain, report in reports.items()
                          if domain=='all' or (entries[name]['seed']==42 and domain in
                                              ['pixel_centered', 'large_scintillator'])}
                    for name, reports in evaluation['models'].items()},
        'complete_domain_tables': str(root/'evaluation/model_domain_metrics.csv'),
        'pairs': evaluation['pairs'], 'mean_seed_pairs': evaluation['mean_seed_pairs'],
        'seed_spread': evaluation['seed_spread'], 'research_criteria': research_criteria,
        'bootstrap': read('evaluation/bootstrap_design.json'),
        'uncertainty_scope': 'pointwise development physical-block intervals conditional on fitted seeds; sample SD includes actual runtime variation; no independent Confirm or Holm result',
        'stress': {'models': stress_compact, 'replicates': stress['replicates'],
                   'scope': stress['scope'], 'source': str(root/'evaluation/stress_results.json')},
        'resources': {key: value for key, value in read('resources.json').items() if key!='allocations'},
        'complete_allocation_ledger': str(root/'resources.json'),
        'reference_costs': read('reference_costs.json'),
        'acceptance': {key: acceptance[key] for key in ['status', 'passed', 'failed', 'scope']},
        'independent_reviews': {kind: {key: value for key, value in read('evaluation/'+kind+'.json').items()
                                      if key not in ['checks', 'models']}
                                for kind in ['independent_review', 'independent_stress_review', 'independent_core_intervals']},
        'Cal_A_Cut_B_Confirm_payload_read': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--plan', type=Path, default=Path('roadmap/v2/experiment_plan.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root, json.loads(args.plan.read_text()))
    write_json(args.output, result)
    print(json.dumps({'status': result['status'], 'models': len(result['domains']),
                      'criterion_comparisons': len(result['research_criteria']),
                      'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
