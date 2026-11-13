# S4 成本即时刷新与全量CPU诊断时限

## 阶段目标

让成本刷新只查询必要的调度记录，并使四checkpoint全量CPU诊断有足够的正式作业时限。

## 工作内容

成本脚本在完整终态cache之外，增加“实时queue + 已确认终态proc”覆盖检查。全部预期proc已在实时queue时，不再扫描不存在的history；只有缺失proc才查询精确cluster history。由此减少活跃作业的20秒history超时，保留缺失终态证据的查询与cache逻辑。

发现功能诊断2200411的`microcentury`实际`MaxRuntime=3600`秒。四个checkpoint逐一对完整10k validation进行CPU FP32 forward，首模型约半小时仍在计算；四模型会超过原时限。已正式修改该运行作业为`workday / MaxRuntime=28800`秒，readback确认JobStatus2，进程连续执行，未重启或重复推理。操作argv和前后状态保存在`audit/functional_probe_runtime_extension.json`。

后续functional_probe提交默认workday。增加每64batch的编码事件数、分析阶段和已完成branch groups原子进度文件。当前运行作业仍使用先前冻结源码，不宣称其具备新增heartbeat。

## 详细数据与验证

LCG108/Python3.12运行`python -m unittest tests.test_flavour_s4_accounting`，2项通过，0.013秒。覆盖活跃queue不扫描history，以及缺失终态的精确history恢复、匹配设备名称、成本求和和terminal cache复用。新增进度输出与提交器的Python编译检查通过。

21:27:48 UTC刷新：7.59250446 device-hours，121.59847071 CPU core-hours，剩余组织device-hour上限52.40749554。MIG和整卡分别记录，不换算FLOPs；当前CPU功能诊断不计为GPU时长。刷新结果包括已失败的V100 prototype与旧OOM allocation。

全量功能诊断结果尚未产生。已有GPU训练继续正常推进，未因本次调度改动中断。

## 阶段性结论

完整queue信息足以计算当前allocation，不必等待history扫描。CPU全量诊断需要独立于短readback作业的时限；本次通过scheduler更新保留已有计算。

## 下一步计划

收取完整功能诊断并核对Head重放与precision drift，完成其图表；继续三seed主比较、Rel/EMA下游和等计算监督控制。最终成本以全部作业终态为准。
