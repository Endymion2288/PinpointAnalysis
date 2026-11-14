# S6：完整独立test与generator decay覆盖

## 阶段目标

确认冻结test和generator紧凑truth全覆盖，进入正式统计读回。

## 工作内容

227个独立test推理分块全部完成，仍使用13个冻结checkpoint和同一BF16/compiled/batch64 policy。CPU finalizer合并accepted catalog、truth、四类分数和quality，建立后续全部配对比较共用的block设计。

新增依赖式 `pipeline complete_decay`：全部重放批次齐后严格核对逐条件key覆盖和逐值响应一致，合并decay sidecar，再接入已有冻结分数和物理分支报告。该步骤不改模型、条件、温度或cuts。

## 详细数据

2026-11-14约14:50 UTC：test227/227推理分块，724441事件。转换与quality均227/227；catalog已合并约72MiB、truth约84MiB，最终score合并仍执行中。

Generator220/220个truth批次，共3512事件；τ decay vertex有效440条，对应两套各220个CC τ事件。全部observations、targets、ideal tracker逐值相同。Transport truth364/368批，剩余任务继续原条件执行。

产物：EOS `test_catalog.parquet`、`test_truth.parquet`、`generator/truth_replay/nominal/*/`；正式分类指标、paired区间与完整内部健康汇总尚待读回。

## 阶段性结论

完整test推理覆盖已完成，generator decay记录修复在全部3512事件上保持探测器响应相同。S6还需最终统计、物理分层和资源closure，当前不提前下分类性能结论。

## 下一步计划

执行正式统计pilot后扩展全部60个model/prior组合；完成readout variants、transport和generator decay汇总；读取实际指标和图表后撰写最终报告。
