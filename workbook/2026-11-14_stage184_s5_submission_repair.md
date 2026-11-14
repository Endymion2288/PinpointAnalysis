# S5 提交参数修正及其他类别smoke

## 阶段目标

核对正式批处理实际执行与四类truth mask。

## 工作内容

首次HTCondor cluster `1581590`在`bigbird20.cern.ch`提交160任务；worker实际报`--task: invalid int value: "'0'"`。原因是shell quoting被直接用于HTCondor旧式arguments，宏周围的单引号成为参数内容。提交代码已让`$(task)`保持裸宏，与已有可运行入口一致；失败作业与日志保留成本记录，未生成科学结果。下一轮使用新allocation名称和修正后Git snapshot。

额外存储interaction track IDs和直接τ带电daughter IDs，供对象评价定义primary/τ vertex的可重建分母；不进入对象构造。Primary vertex辅助mask排除真实全空响应。

## 详细数据

LCG108下每类另取16事件：`10081_072`耗时2.803451 s、2个charm；`10082_053`2.446538 s、3个charm；`10084_032`2.910589 s、0个charm。三类的tau mask均无效、分道均`not_tau`、与catalog差异0。加此前τ smoke共64个不同事件。产物为S5根下`truth_smoke_000/040/120.parquet`，τ早期产物为`truth_smoke.parquet`。

## 阶段性结论

四类小样本提取通过。首次正式执行在参数解析阶段失败，因此没有全量truth结论；该结果仅是执行故障证据。修复不改变物理标签定义。

## 下一步计划

用修正后提交入口重跑160任务，合并sidecar；对象候选在独立阶段检查关联效率、fake与vertex resolution。
