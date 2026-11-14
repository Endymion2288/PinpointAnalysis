# S6：完整transport响应、队列恢复与最终图表

## 阶段目标

读回完整实际transport覆盖，推进全test和decay重放，并准备完整统计的英文展示。

## 工作内容

1024个固定原始事件完成nominal、field±5%、W-density±5%；256个τ事件另完成helicity−1/0/+1。三主seed已在Kubeflow原allocation空闲进程中完成全部推理，合并为`predictions/transport_raw.parquet`。Condition统计沿用原冻结温度、prior与cuts。

新增最终图：四prior下τ效率/纯度、energy/vertex/occupancy条件的固定cut区间、真实transport变化、独立generator变化；现有confusion、ROC/PR、reliability、controls、readout、migration图保留，全部英文。

## 详细数据

`predictions/transport_completion.json`实际5888行：五个全类条件各1024，三个τ helicity条件各256；逐条件keys覆盖与去重核对通过。共享transport GPU进程351.800秒、exit0，费用归入原Kubeflow allocation，不增加设备小时。

2026-11-14约14:42 UTC：全部test转换与quality各227/227，test推理182/227。Transport compact truth190/368批、generator4/220批，已完成记录全部响应逐值相等。最终完整truth和test统计尚未完成。

Generator重放首次选bigbird15，排队后 `condor_q -better-analyze` 显示其站点改写Requirements形成 `PreCvmfsRequirements → PreLocalRequirements → PreAfsRequirements → PreCvmfsRequirements` 循环，5326个slot全部拒绝。48个materialized任务全部Idle且无JobStartDate，取消17040232后转到兼容bigbird20（1582756），没有重复运行响应。Runtime保存约束、取消与新提交证据，并排除后续使用该schedd。

Transport统计1582757八项在Idle时设置依赖hold，等待完整test的`bootstrap_design.json`；这是调度等待，未执行条件统计。最终必须release并完成，不以提交替代结果。

## 阶段性结论

实际transport响应和冻结模型分数已经全部覆盖，支持下一步有配对区间的效应评价。最终统计和decay分层仍在执行；S6尚未完成。

## 下一步计划

完整test ready后释放transport统计、执行60模型/prior与paired readout统计；完成全部truth重放、物理分层和资源closure后更新最终报告。
