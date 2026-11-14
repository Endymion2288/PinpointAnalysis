# S4 真实FP64加速gate通过与100k读出接续

## 阶段目标

真实检查相同feature/scaler/C=1的设备求解器，再以新名称接续耗时100k读出。原CPU配置、原阈值与历史产物不改写。

## 工作内容与详细数据

Kubeflow `pinpoint-s4-fp64-acceleration-gate`在H100 NVL MIG 3g.47gb完成8组实际对照，总31.740571s。Scratch/MAE3、1k/10k、full/global均通过原目标差≤1e-5、最大概率差≤0.005和双方独立CPU梯度≤1e-6要求。最大概率差0.000152091；每项设备fit加CPU证书耗时0.452–2.518s。此差异为阈值内的数值执行差异，不能称权重或概率逐位相同，也不用于方法性能比较。

来源：结果根`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13/experiments/s4_fp64_acceleration_gate_s42/solver_probe.json`。原CPU100k五组仍在`fitting_full_linear`，未报错；小样本CPU/GPU比较只读原train，不选择validation超参数。

新增[5项100k读出](../roadmap/configs/s4_fp64_readouts.json)、[统一30配置](../roadmap/configs/s4_fp64_objectives_evaluation.json)、[66配对](../roadmap/configs/s4_fp64_objectives_pairs.json)和[正式评价](../roadmap/configs/s4_fp64_objectives_validation.json)。只替换100k数值执行分支为新实验名称；1k/10k既有CPU fits与全部15项分类模型复用。200 outer/200 inner、C=1、fit-only scaler和CPU独立驻点阈值不变，后续仍回放30个磁盘readouts。

## 验证

设备目标公式的独立autograd和相关features测试、既有solver与feature回放测试合计9项执行；实际结果以测试输出记录。真实8项gate已测量通过，100k完整stationarity和统一统计尚未测量。正式任务使用提交后的source与独立name；先保留原CPU运行，待替代产物通过才结束重复计算。

## 阶段性结论与下一步计划

事实：相同目标的FP64设备实现通过真实小样本对照，能显著缩短所测fit耗时；这不改变固定C=1低标签过拟合诊断。下一步完成100k独立CPU证书、磁盘readout回放、完整45预测/66配对，确认实际终态并清理缓存。
