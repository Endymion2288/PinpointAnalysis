# S5 全量truth提取与队列兼容修正

## 阶段目标

完成110k事件sidecar提取，并消除已测量的不兼容schedd重复选择。

## 工作内容

V3的160任务均生成Parquet与summary；累计110,000事件，提取任务墙钟时间之和2,300.917987 s（不是allocated CPU-hours）。初步分块统计τ有效decay27,496、charm8,030、与catalog分道差异11；完整分道/mask、差异类型需合并结果后报告。不能将少数差异抹掉或归为hadronic。

`bigbird21`虽然queue较短且WRITE/EOS布局检查通过，实测AFS/CVMFS站点约束存在循环，不能匹配。S5提交入口新增带原因/证据的schedd排除记录，仍在其余兼容候选中按实时idle/held/running选择。该修正只影响调度。此前assemble-v1被此选择再次路由；不把短queue等同于兼容性。

## 详细数据

Truth产物位于S5根`truth_parts/000–159.parquet`及同名summary。V1的160个失败任务历史已保存`runtime/s5-truth-v1.history.txt`，样例ExitCode=2、RemoteWallClockTime=4 s、RequestCpus=4。V2 cluster2206661已移除，未分配计算节点；V3 cluster1581684完成提取。完整allocation成本待历史汇总。

## 阶段性结论

160任务覆盖全部接受事件，11个分道差异将独立诊断。队列兼容性已基于实际match evidence更新。对象和模型学习尚未形成物理结果。

## 下一步计划

正式合并、核对11个分道差异，提交40个对象任务；对象评价完成后执行16事件S5拟合gate。
