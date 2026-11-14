# S6：冻结生成器对照与独立 test 执行

## 阶段目标

使用冻结概率映射和工作点评价完整 Standard/NUX benchmark，量测四类与τ/charm变化；继续独立test及条件任务。

## 工作内容

220个native generator小批、3512事件完成三主seed推理。逐条件event key/count与固定manifest完全一致，形成三seed raw平均后使用nominal ensemble温度和validation cuts。统计使用400次独立generator source-file block bootstrap，在存在的共同cell内恢复原benchmark占比；两套hard scatter不配对。四个prior场景均执行，charm子集及raw tail计数同时保存。

新增paired digitizer多lane launcher，多个worker共享事件claim，避免重复写入。独立test观测质量由16个CPU worker消费已转换分块；finalizer等待全量概率/质量，不能以部分事件收尾。模型使用模态的活动状态与探测器全部模态活动另列。

## 详细数据

Generator GPU Pod `pinpoint-s6-generator47-gbxld`：2026-11-14 13:50:32Z至13:53:08Z，H10047GB MIG，exit0，156秒；终态已保存。正式输入包含两套各1756事件，τ各220。所有模型内部health通过。

Balanced prior下 NUX−Standard：balanced accuracy −0.031836，95%区间[−0.065620,0.001892]；NLL +0.012881，[−0.018748,0.043537]；固定τ1%背景工作点的τ效率 −0.013636，[−0.058216,0.028932]。各区间400个有效replicates。来源 `evaluation/conditions/generator_comparison.json`。这些区间跨零，没有建立本benchmark下方向确定的差异，也不能证明两个generator等价。

连续能谱仍有coarse-cell内部残差，四类KS约0.043–0.055；charm覆盖较少，详见stage218。对照限定为GENIE3.04 hadronization配置，不是NuWro或全部理论系统误差。

独立test转换HTCondor1582724；质量workers1582728，finalizer1582729；transport inputs1582725。尚无独立test指标或field/material结果。

## 阶段性结论

预定Standard/NUX对照已经完成，统计精度尚不能确定上述主要指标的变化方向。完整test与真实条件范围仍待测量。

## 下一步计划

运行全部test和paired读出/真实transport，对每项合同检查直接产物、事件分母与400次统计。清理临时native cache、保存全部后端终态与allocation成本后写最终报告。
