# R3完整结果与下一步工单

**R3已完成，249项验收通过；本轮没有新的质量晋级模型，继续使用R2的B1 D2真实T3/T4固定终点作为参考。**这回答了当前具体改动的效果，没有证明模型收敛、容量充分或达到最优。完整数值见[r3_results.json](r3_results.json)，执行记录见[stage306](../../workbook/2026-11-15_stage306_r3_final_results.md)。R0–R7总设计、引用和v1归档继续由[总入口](../README.md)管理。

## 1. 哪些设想已经得到检验

全部使用冻结Train/Tune，seed42；每项终点与自身T4 NLL最优4k screen分别评价，主决定使用固定曝光终点。Tune40k原始四类各10k；BA3在真实非τ样本上使用条件三类输出，τ工作点使用完整非τ背景。下表区间是400次paired physical-block bootstrap，条件于本次已拟合权重；不是训练seed总体区间。

| Intervention and reference | Common BA difference, pp [95% interval] | Additional evidence | Decision |
| --- | --- | --- | --- |
| M0 global mixing − D2 spatial | +0.383 [+0.035,+0.735] | μ recall −4.08 pp；稀疏μ −14.816 pp，τ无收益 | 保留原采样 |
| G1 stride 2 − D2 stride 4 | −0.033 [−0.366,+0.353] | e recall −1.43 pp；τ效率无净提升；dense计算增加 | 当前stride2配方不采用 |
| G2 relations − G2 control | −0.023 [−0.308,+0.219] | τ在1% FPR效率 −0.450 pp [−0.795,−0.101] | 当前关系层不晋级 |
| G2 no global − G2 relations | −0.450 [−0.742,−0.140] | NLL3 +0.008938；μ recall −1.370 pp | 保留当前native参照的global stack |
| C1 sensor rows − D0 physical grid | +0.137 [−0.213,+0.504] | e recall −3.42 pp；同设备训练吞吐2.723倍，短operation吞吐1.747倍 | 保留成本研究候选，质量不晋级 |

G2 control、relations、no-global对同100k×4强B1 D0的BA3分别下降0.923、0.947、1.397 pp；τ在1% FPR效率分别为3.14%、2.69%、2.77%，D0为5.11%。这里同时改变表示与容量，不能把跨族差值称为纯几何信息效应。M0/G1使用300k×4；其结果不能与100k×4模型直接解释为架构优劣。

三个native配置不是完整2×2矩阵；移除global stack也减少参数量。结果支持当前重训移除方案降低共同分类质量，不支持“global必然压制局部信息”或“几何信息无用”。当前没有成功组合需要进一步解释增益来源，因此本轮不追加同容量/打乱边控制。

## 2. 已排查的程序因素与剩余限制

修正硅厚度单位后的五项真实小样本拟合全部通过；六个正式模型都有实际优化时的分支梯度、参数更新和finite激活。G2的gate/message/norm各52个健康窗口都有正梯度和更新。故障修复前的结果只保留作诊断，本轮结论来自修复后的完整训练。

四组最终probe覆盖六个正式模型和D2参考。Scaler及multinomial logistic C1仅在冻结balanced Train4k拟合；每个权重重新计算完整Tune40k。除no-global跨A100/H100最大概率差0.003715、argmax一致99.89%外，其余读回最大差均为0。No-global通过预定数值容差，主评价保留A100原始输出。

Native control/relations的local BA3分别47.977%/47.020%，global为48.930%/48.970%；稀疏域local为35.610%/34.433%，global为34.603%/35.094%，tracker为38.315%/38.223%。这些probe衡量当前池化特征和固定线性读出的可读性；不构成原始测量信息上限，也不能单独证明增加某分支有益。No-global重训没有恢复整体质量，当前证据不足以选择branch dropout或gradient modulation。

仍需解决的具体问题：

1. **候选关联质量。**Train1024审计中，modality-visible图的节点cross-plane覆盖约99.01%，但同已记录首贡献trackID的候选recall在visible pixel gap 1/2/3分别仅27.53%/10.15%/17.48%。Coverage与正确关联不同；这些标签不是完整ancestry，数值也不是物理整轨效率。
2. **稀疏事件分离。**多次改动主要重分配e/μ recall，没有恢复BA、AP和AUROC的共同收益。先查观测支持、邻接、方向和接受度，再决定表示改造。
3. **训练与容量。**R2未建立充分收敛，R3只检验固定预算配方。宽度、训练seed、pool抽样、长期优化和更大数据的上限仍未测量。
4. **τ表示与标签。**flat4已经可训练，低误报效率仍低。细分类需要区分直接daughter、经共振展开后的final charged prong和实际重建轨数；增加softmax类别本身没有增益证据。

## 3. 下一项：R4的最小可执行批次

以下是下一阶段设计，尚未启动或测量。先冻结事件、标签定义和评价合同，再读取相应新产物。

| Order | Work and inputs | Required output and decision |
| --- | --- | --- |
| R4.0 | 从现有Train与Tune各固定10k，四类各2.5k；保留原全空、低occupancy、pixel-centered及scintillator域定义。核对truth标签支持、混合贡献与缺失情况 | 输出标签可评价分母、观测覆盖、各类/各域counts及无法判定比例；不把首贡献未匹配当成完整truth的负例 |
| R4.1 | 比较旧nearest-neighbor、实际z spacing+方向/uncertainty的多候选关联与robust fit；bar未测轴保留区间；pixel使用已接受pitch。局部直线和需要场传播的区段分别定义 | candidate recall–edge count、整轨efficiency/fake/duplicate、事件coverage、方向及vertex残差；完整曲线和失败分母，CPU批处理后选择一条路径 |
| R4.2 | 从观测重建对象提取hit数、起点、方向、隔离、质量与不确定度；先训练object-summary HGB。Truth association仅作诊断oracle | 判断对象特征是否提供新分离能力。Oracle收益不作为可部署性能，重建失败事件仍进raw路径 |
| R4.3 | 仅在关联和覆盖支持后，冻结一个raw+object模型与必要raw-only对照；同Train100k×4，沿用同曝光/LR/精度，先真实拟合与资源profile | 对同100k D0作增量配对，再与保留的D2参考分别报告；全域、逐类、稀疏域、τ工作点及完整成本均交付 |

下游晋级继续用[既定门槛](02_tasks_and_evaluation.md#6-统计设计与晋级门槛)：三类ΔBA3至少2 pp、区间支持正效应、NLL与三类recall满足约束；τ在1% FPR效率至少增加1 pp、区间支持正效应且共同三类非劣。没有新证据时，保留精确负结果并改查限制来源。满足晋级或有明确的小收益/成本科学价值后，才冻结匹配seeds17/2026及粒度/机制扩展；不按空闲GPU数量扩矩阵。

## 4. R5–R7如何承接

**类别数：交付真实三类对照，同时保留四类研究。**R2已经实际重训T3；与conditional T4的共同任务差异未建立稳定优势，当前没有依据为了易训练而删掉τ。后续优先比较flat4、τ gate+conditional3及conditional decay辅助；τhad specialist先行，τe/τμ按真实混淆安排。保持τ总训练权重、未知子道coarse监督和完整非τ背景，最终按合并四类/共同三类评价。had1/had3等更多类别在标签与重建支持后再决定。

**R6：只针对已证实缺口选一项。**若R4证实方向或连续性可测而监督不足，比较一项局部监督/topology目标与同总计算的继续监督；若没有对象增益，保留raw B1。C1需要新的逐类质量恢复证据后才能进入成本/质量选择。现有五折bias诊断已不足以恢复e质量，不继续重复相同低维修正。

**R7：独立确认仍单独保留。**先冻结最多两项主比较及最终权重、先验和评价规则，再用Cal-A拟合校准、Cut-B定工作点，最后打开200k Confirm。R3重复使用的Tune区间与局部probe不能替代独立确认。引用依据与方法适用条件见[文献综述](05_literature.md)，尤其是FASERCal监督覆盖、OPERA拓扑、NOMAD末态运动学及局部监督设计。

## 5. 资源与交付位置

R3全部39条allocation终态，18.094167异构device-hours、223.338333 allocated CPU core-hours；按allocation计一次，包含失败、恢复、缓存、probe和CPU辅助。最后Kubeflow Pod已被清理，使用已保存的PodScheduled与live Job Complete时间计11,692s保守上界。GPU utilization未测量，异构/MIG小时不作FLOP等效。

R0已记录上界+R1/R2/R3 batch累计72.363744h，原80h规划名义余量7.636256h，完整交互allocation开销未测量。原R4–R7笛卡尔矩阵不按旧额度直接执行：先做R4 CPU小批次，再按实测收益与吞吐安排一个有依据的GPU比较，保留R7资源。HTCondor与Kubeflow并用，接受不同型号、MIG和平台共享卡；排队、准备、实际吞吐及验证共同决定预计完成时间。

源码和冻结配置在`src/analysis/flavour/v2/r3/`与`configs/r3/`；完整输出在`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R3`。四套主评价、独立复核、最终probe/health、科学与条件决定、23张英文PNG/PDF及完整资源账本均保留。机器汇总为[r3_results.json](r3_results.json)，历史执行和迁移见[r3_execution.json](r3_execution.json)。
