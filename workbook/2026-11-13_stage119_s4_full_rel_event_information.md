# S4：完整Rel超越同probe几何先验

## 阶段目标

用与pilot完全相同的固定512事件，检验完整Rel是否利用几何位置之外的事件信息，同时量化当前MIG工作负载的SM活动。

## 工作内容与详细数据

正式CPU cluster1575200、源码eb4358b完成211.870秒。使用原512个train-inner事件、37,567个有效bar channels、3,676个硬件位置，FP32重算选定checkpoint；与pilot的query/target集合和几何先验一致。400次accepted-family connected-chunk配对bootstrap使用44个blocks。checkpoint固定，几何/常量先验均在同probe拟合。

| 同probe语义CE | 数值 |
| --- | ---: |
| Trained full channel Rel | 0.64275581 |
| Event-permuted decoder source | 0.79791057 |
| Oracle constant | 0.72388333 |
| Oracle per-channel geometry | 0.66291918 |
| Soft-target entropy | 0.19901407 |

| 相对trained的配对差额 | Delta | 95% CI |
| --- | ---: | --- |
| Event-permuted − trained | 0.15515476 | [0.13984213, 0.17272599] |
| Constant − trained | 0.08112752 | [0.07211605, 0.09042362] |
| Per-channel geometry − trained | 0.02016337 | [0.01185518, 0.02956380] |

在同一固定probe上，完整训练后的channel Rel误差低于所拟合的逐硬件位置均值先验，且置换事件source明显退化。原pilot channel Rel的trained CE为0.71790266，未超过几何先验；完整训练的结果支持继续检验下游，不再依据pilot弱结果推测完整方法无效。这里没有给不同checkpoint间差额建立bootstrap区间；上述区间只对应同模型的三项配对控制。

产物为`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13/experiments/s4_rel_full_channel_diagnostic_s42/semantics.json`，精简数据追加于[Rel诊断表](../roadmap/results/s4_rel_semantics.json)。下游ντ辨识能力尚待。

约20:40 UTC通过各自正式Condor作业的只读SSH会话查询VM本地DCGM exporter，每6秒一次，共8次，跨度约42秒。每个VM只返回其分配的MIG profile，保留原始带label的样本于`audit/gpu_telemetry/condor_<cluster>_20261113.json`。

| Allocation | 当时工作负载 | Mean SM_ACTIVE | Range |
| --- | --- | ---: | --- |
| 1574116，24GB MIG | Frozen probe与Rel下游训练交接 | 0.096144 | [0.040270, 0.152018] |
| 1574593，12GB MIG | 完整EMA训练 | 0.231749 | [0, 0.463498] |

`SM_ACTIVE`是SM有至少一个warp的周期比例，不等同于NVML GPU-util百分比；exporter读数可重复，8次poll不能视为8个独立观察。这是短窗口、不同工作负载的资源诊断，不用于比较模型速度或物理效果。两个data worker在另一次进程快照中的平均CPU使用率约90%，与已验证的前端优化方向一致；未测长期SM均值。

## 阶段性结论

完整channel Rel具有超越同probe几何均值先验的事件语义信息证据。该结论与四分类收益分开验证。资源仍受CPU数据准备影响；保留正常任务，优化源码候选继续在HTCondor排队，增加12GB MIG候选以使用实际可获配的小切片。

## 下一步计划

继续1k/10k/100k四分类、frozen/global probes、独立SSL seeds及matched-supervised；获得下游证据后再决定Rel/EMA是否需要额外独立预训练seed确认。汇总最终完整区间、τ子道、common support与成本，当前不扩至1M。
