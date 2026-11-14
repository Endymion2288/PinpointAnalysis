# S6：compact decay truth 实测与物理范围

## 阶段目标

验证记录修复在全部冻结条件下保持响应相同，补齐 τ/charm 与 polarization/secondary implementation 的直接证据。

## 工作内容

Task0的16个CC e事件×5条件、task32的16个CC τ事件×8条件，共13批重放。每批逐值比较observations、targets、ideal tracker，仅保存紧凑truth；原native和模型分数保留。新增 `decay evaluate` 将完整重放truth接到冻结分数上，报告各条件或generator的τ分支、顶点、分支迁移及冻结cut选择计数。

Charm树包含primary charm的所有直接子粒子，含相互作用与电离secondary。其electron/muon出现率明确不称作charm branching fraction；primary charm子集的背景接受数另外列出。保持原G4/PY8DK实现，未将SPINUP=0命名为非极化混合。

## 详细数据

13/13批所有三文件逐值相同，累计761.255 task-seconds。八个τ条件各16/16 decay vertex有效，共128条有效τ真值。Nominal τ分支为hadronic11、muon4、electron1；helicity−1和0与nominal在这16条上相同，helicity+1为hadronic12、muon2、electron2。这个小pilot只验证记录接口，分支差异不作为总体效应结论。

τ pilot每条件有1个primary charm事件，记录3–5条直接子粒子；CC e pilot无primary charm。证据：EOS `audit/compact_truth_pilot.json`、`transport/truth_replay/*/*/summary.json`。批量transport原响应已完成366/368条件批次，其余两个仍由原任务44执行。

HTCondor GPU1582750实时分析：0个立即匹配slot、50个drained后匹配slot。Kubeflow继续nominal，同时用同一Pod剩余两个CPU进程执行transport三seed推理，没有新增GPU申请。费用汇总保留同一allocation，不重复计算共享worker。

## 阶段性结论

全部冻结条件的pilot支持compact truth记录不改变响应。τ/charm记录缺失已定位并修复，完整3512条generator及5888条transport条件truth尚待批量重放覆盖。

## 下一步计划

扩展其余62个transport批次与220个generator批次；合并时逐条件核对keys、响应相等与truth覆盖。完成全部test和paired readout，再运行固定统计与物理子集报告。
