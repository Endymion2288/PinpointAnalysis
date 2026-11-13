"""Publish measured revision 2 tables only after the complete scientific gate."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import median

from analysis.flavour.inventory import write_json
from analysis.flavour.s3.plot_repair_paths import spectrum_summary


def short(name):
    return name.removeprefix('s3r_').replace('_e4', '').replace('_', ' ')


def interval(row):
    low, high = row['CI95']
    if row['delta'] is None:
        return 'undefined'
    return f"{row['delta']:+.5f} [{low:+.5f}, {high:+.5f}]" if low is not None else f"{row['delta']:+.5f} [undefined]"


def summarize(root, destination):
    read = lambda path: json.loads(path.read_text())
    acceptance = read(root / 'final_acceptance.json')
    if not acceptance['strict_output_gate_passed']:
        raise ValueError('complete scientific and terminal allocation evidence required')
    evaluation = read(root / 'evaluation_final/summary.json')
    conditional = read(root / 'evaluation_final/conditional/summary.json')
    accounting = read(root / 'campaign_accounting.json')
    experiments, health, paths, exposures = [], {}, {}, {}
    for experiment in evaluation['experiments']:
        name = experiment['name']; directory = root / 'experiments' / name
        inspection = read(directory / 'internal_health_checks.json')
        health[name] = {
            key: inspection[key] for key in ['best_epoch', 'sample_count', 'informative_sample_count',
                'encoder_gradient_squared_norm_min', 'encoder_gradient_squared_norm_median',
                'encoder_gradient_squared_norm_max', 'activation_ranges', 'NLL_minus_constant_prior',
                'internal_health_and_learning_passed']}
        records = read(directory / 'training_health.json')['records']
        health[name]['gradient_path_summary'] = {
            key: {'sampled_squared_norm_min': min(values),
                  'sampled_squared_norm_median': median(values),
                  'sampled_squared_norm_max': max(values),
                  'exact_zero_samples': sum(value == 0 for value in values),
                  'samples': len(values)}
            for key in ['input', 'local', 'local_pool', 'module_read', 'latent_read',
                        'global_blocks', 'head', 'linear_readout']
            if (values := [row['gradient_squared_norms'][key] for row in records
                           if row.get('informative_nodes') != 0 and key in row['gradient_squared_norms']])}
        exposures[name] = read(directory / 'final_checkpoint_checks.json')['training_exposures']
        path = directory / 'path_diagnostics.json'
        if path.exists():
            result = read(path)
            paths[name] = {key: result[key] for key in ['probe_events', 'probe_NLL', 'occupancy_range_nodes',
                'permuted_fraction', 'branch_permutation', 'head_controls', 'branch_between_event_variation_RMS',
                'token_spectrum_scope', 'interpretation']}
            paths[name]['token_spectrum_summary'] = {
                key: spectrum_summary(rows) for key, rows in result['token_spectrum'].items()}
        experiments.append(experiment)
    summary = {
        'status': 'S3_revision2_complete', 'utc': datetime.now(timezone.utc).isoformat(),
        'scope': evaluation['scope'], 'results_root': str(root),
        'legacy_method_conclusions_withdrawn': True, 'strict_output_gate_passed': True,
        'checks': acceptance['checks'], 'counts': {
            'models': len(experiments), 'screen': 8, 'additional_confirmation': 12,
            'training_epochs': 80, 'checkpoint_readback_checks': len(experiments) * 14,
            'internal_health_samples': sum(v['sample_count'] for v in health.values()),
            'fixed_checkpoint_path_diagnostics': len(paths),
            'single_seed_pairs': len(evaluation['paired_comparisons']),
            'three_seed_comparisons': len(evaluation['seed_mean_comparisons']),
            'conditional_domains': len(conditional['paired_domains'])},
        'accepted_samples_and_detector_proxy_unchanged': True,
        'train_events_per_model': 80000, 'validation_events_per_model': 10000,
        'training_exposures': exposures, 'optimizer_steps_per_model': 40064,
        'calibration_test_used': False, 'S4_S6_executed': False,
        'bootstrap_replicates': evaluation['bootstrap_replicates'],
        'resampling_blocks': evaluation['resampling_blocks'], 'common_support': evaluation['common_support'],
        'metric_definitions': evaluation['metric_definitions'], 'experiments': experiments,
        'training_seed_spread': evaluation['training_seed_spread'],
        'paired_comparisons': evaluation['paired_comparisons'],
        'seed_mean_comparisons': evaluation['seed_mean_comparisons'],
        'conditional_three_seed_comparisons': [{
            'domain': row['domain'], 'events': row['events'], 'comparisons': row['seed_mean_comparisons']}
            for row in conditional['paired_domains']],
        'internal_health': health, 'path_diagnostics': paths,
        'promotion_decisions': acceptance['promotion_decisions'],
        'cost': {key: accounting[key] for key in ['repair_device_hours_upper_bound',
            'prior_S3_device_hours_upper_bound', 'organization_GPU_hours_upper_bound',
            'remaining_GPU_hours_lower_bound', 'missing_history_records']},
        'path_diagnostic_scope': '64 occupancy-quantile nonempty validation events, same-presence branch permutation; fixed-checkpoint reliance, not population performance or a retrained ablation',
        'scientific_interpretation': 'see workbook stage87 for the reviewed interpretation; old failed checkpoints do not establish architecture or native-information limitations'}
    write_json(destination, summary)
    lines = [
        '# S3 修复重训的完整实测数据', '',
        '本文件由严格验收后的产物生成数值表，科学解释及图表人工复核记录写入stage87报告。', '',
        f'结果根：`{root}`。accepted 80,000 train / 10,000 validation；calibration/test未使用。', '',
        '## 四分类和训练健康', '',
        '| Model | Best epoch | Macro AP | NLL | Macro F1 | Balanced accuracy | AP e / mu / tau / NC | Health samples |',
        '|---|---:|---:|---:|---:|---:|---|---:|']
    for experiment in experiments:
        name = experiment['name']; values = experiment['four_class']
        ap = ' / '.join(f"{values[f'class_{c}_AP']:.5f}" for c in range(4))
        lines.append(f"| {short(name)} | {health[name]['best_epoch']} | {values['macro_AP']:.5f} | {values['NLL']:.5f} | {values['macro_F1']:.5f} | {values['balanced_accuracy']:.5f} | {ap} | {health[name]['sample_count']} |")
    for title, rows in [('三seed配对', evaluation['seed_mean_comparisons']), ('单seed控制', evaluation['paired_comparisons'])]:
        lines += ['', f'## {title}', '',
                  '差值统一为B−A；区间为400次、128物理块的95%配对percentile interval，条件于已拟合seeds，未合并training seed SD。', '',
                  '| Contrast | Domain | Delta macro AP [95% CI] | Delta NLL [95% CI] |', '|---|---|---:|---:|']
        for row in rows:
            label = row.get('name') or f"{short(row['b'])} minus {short(row['a'])}"
            for scope in ['raw', 'common_support']:
                metrics = row['metrics'][scope]
                lines.append(f"| {label} | {scope} | {interval(metrics['macro_AP'])} | {interval(metrics['NLL'])} |")
    lines += ['', '## 条件域：tracker增量', '',
              '| Domain | Events | Delta macro AP [95% CI] | Delta NLL [95% CI] |', '|---|---:|---:|---:|']
    for domain in conditional['paired_domains']:
        row = next(r for r in domain['seed_mean_comparisons'] if r['name'] == 'tracker_increment_three_seed')
        lines.append(f"| {domain['domain']} | {domain['events']} | {interval(row['metrics']['raw']['macro_AP'])} | {interval(row['metrics']['raw']['NLL'])} |")
    lines += ['', '## 固定checkpoint分支置换诊断', '',
              '下表只比较64-event occupancy探针的NLL变化；无重训，不能作为总体性能增益。', '',
              '| Model | Probe NLL | All regional delta | All local delta | Global latent delta | Summary delta |',
              '|---|---:|---:|---:|---:|---:|']
    for name, path in paths.items():
        values = path['branch_permutation']
        columns = [f"{values[key]['probe_delta_NLL']:+.5f}" if key in values else 'absent'
                   for key in ['all_regional', 'all_local', 'global_latent', 'response_summary']]
        lines.append(f"| {short(name)} | {path['probe_NLL']:.5f} | " + ' | '.join(columns) + ' |')
    lines += ['', '## 资源', '',
              f"修复设备时上界 {summary['cost']['repair_device_hours_upper_bound']:.9f}；累计S3上界 {summary['cost']['organization_GPU_hours_upper_bound']:.9f} / 120，余额下界 {summary['cost']['remaining_GPU_hours_lower_bound']:.9f}。", '',
              '整卡和MIG小时是allocation费用，不视作吞吐等价；跨后端共享分配不授予模型本体效率晋级。', '']
    (root / 'evaluation_final/measured_tables_zh.md').write_text('\n'.join(lines))
    print(json.dumps({'summary': str(destination), 'models': len(experiments), 'table_lines': len(lines)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.root, args.output)


if __name__ == '__main__':
    main()
