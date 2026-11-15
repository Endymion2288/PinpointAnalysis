# R1：紧凑汇总中的事件人数与工作点覆盖率

## 阶段目标

确保最终R1紧凑产物能直接区分实际事件人数和平衡先验下的加权统计，保留固定FPR及purity的效率与覆盖率。验证现有真实结果的导出字段；最终41模型验收仍待训练完成。

## 工作内容

`r1.summarize.compact_domain`增加各T3/T4域的原始confusion人数、benchmark加权confusion、weighted counts和prior；原始人数读取`observed_mixture`，未重加权的域直接使用原报告。固定FPR为1%/0.1%，固定purity为80%/90%，同时保留效率和覆盖率。T3矩阵顺序为e/μ/NC，T4为e/μ/τ/NC。

训练汇总增加HGB实测`fit_seconds`与线程数，设备字段可从CPU manifest的`device`读取。GPU训练继续保留实际allocation、精度、曝光、wall与学习曲线。

增加不等类人数的端到端小例子：原始support为[3,2,1,2]，T4平衡加权support为[2,2,2,2]。检验实际confusion与重加权confusion分别保留；单个τ事件在该benchmark中的覆盖率为25%，与原始1/8人数比例不同。

## 详细数据

真实输入：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R1/interim/b1_three_seed/evaluation/results.json`。仅读取已有Tune统计，不重新推理。

| Item | Measured result |
| --- | ---: |
| Models | 18 |
| Model/domain reports | 198 |
| Task/domain reports | 297 |
| Export checks passed | 9,405 |
| Task/domains with unequal raw and benchmark confusion | 270 |
| R1 + summary unit tests passed | 23 |

每个真实任务域检查raw confusion的行和等于原始support、人数为整数、weighted confusion行和等于weighted support；完整类支持时另检查按原始support独立重加权的矩阵与均匀prior。所有已导出工作点的效率和coverage均在[0,1]内。产物为`R1/audit/compact_domain_export_stage260.json`。

LCG108/Python3.12，`PYTHONPATH=src`，OpenMP/BLAS线程1：

```bash
python -m unittest tests.test_flavour_v2_r1 tests.test_flavour_v2_r1_summary
git diff --check
```

真实读回通过小型Python调用逐个执行`compact_domain(report)`并核算上述行和及权重；没有构造虚假的acceptance运行最终导出。最终`r1_results.json`完整导出尚未测量。

03:46 UTC唯一依赖控制器PID2052979仍存活，5项GPU训练均Running；32/41模型已完成。B2实时约40 events/s，H100整卡的既有同族seed42约96 events/s。资源广告另有兼容较大设备Unclaimed/Idle；广告不代表本用户必然能获配，后续候选先取资源并测吞吐，再按剩余完成时间决定是否替换现有writer。

## 阶段性结论

**事实：**紧凑统计可以保留真实人数与benchmark指标的不同含义，现有18模型的导出检查通过。各几何域类比例不相同，直接把加权confusion写成原始人数会产生错误解释。

**未测量：**剩余9模型、完整133配对与最终验收后的整体导出。当前检查不替代最终R1验收。

## 下一步计划

继续完成R1训练与最终配对统计。候选迁移需要实际更快的Train-only profile、旧writer终态证据和显式release；不使用Tune分数选择设备。全部模型完成后进行full独立复核、图表检查与资源关闭，再运行accept及最终汇总。Cal-A/Cut-B/Confirm继续封存；R2只完善设计。
