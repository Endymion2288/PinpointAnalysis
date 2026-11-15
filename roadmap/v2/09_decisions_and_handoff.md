# 研究结论与下一批实验决策

**已有改进成立；现有模型尚未达到已验证的最佳效果。R2已完成，保留四类研究和真实三类交付；下一批先控制采样顺序，再检验局部几何信息路径。**本文将资料审查、已经完成的对照和后续决策放在同一入口。详细证据、模型和评价定义分别见[01](01_evidence_review.md)、[03](03_models_and_physics.md)和[02](02_tasks_and_evaluation.md)。

## 1. 对目前结果的判断

| Question | Evidence | Decision |
| --- | --- | --- |
| 是否比旧方法有改进？ | S6独立test上MAE−Scratch的BA增量为+0.3354 pp，区间不跨零；tracker增量明确。新R1同输入对照中B1优于当前B4的总体BA和τ工作点 | 保留这些成果；区分旧独立test与新Tune开发比较 |
| 当前复杂模型是否最好？ | R1三seed的B1 T3 BA3=56.5633%，B4为54.3922%；R2 B1 D2 T3/conditional T4为57.0967%/57.1267%，均未建立收敛或容量上限 | D2两个固定endpoint作为下一批参考；不把已测最强配方称为全局最优 |
| 降成三类是否能解决困难？ | 同allocation等时比较中，B1 T3−T4 conditional的BA3差为+.064 pp，区间跨零；B2为−.313 pp，NLL3均更差 | 三类作为合作组交付；研究主线继续保留四类 |
| τ是否应当放弃？ | B1三seed在开发集1% FPR处的平均τ效率为5.0200%，比B4高2.423 pp；绝对效率仍低，尚无新独立确认 | 保留有实测增益的四类分支，优先研究可见拓扑和低背景工作点 |
| 能否直接增加更多类别？ | 旧flat6在特定10k标签配方下损害τ AP；另一个配方在一个工作点有小幅收益 | 先做条件衰变辅助头；进一步细分须有标签、可见性和合并后任务收益支持 |

上述R1数值是已拟合seed的单模型指标均值，使用相同40k Tune；不是ensemble、原合作组模型的严格复现或新Confirm结果。[stage274](../../workbook/2026-11-15_stage274_r1_final_results.md)给出完整统计。

## 2. 最需要解决的设计问题

优先级最高的是优化顺序、训练饱和与有效信息路径。R2已完成100k/300k与4/12 passes对照，并确认一个固定prefix的末轮顺序效应；更大数据和容量上限仍未建立。复杂编码器先按同plane窗口和canonical region压缩，再通过有限query/latent融合。保留全部输入通道并不等于短轨、窗口边界和跨plane方向关系能够有效影响输出。B1虽然目前较强，1mm投影与累计stride32也可能限制稀疏局部特征，须实际消融验证。

对象重建是另一个已测弱点：旧近邻轨段purity约2.0%–5.4%，primary双轨vertex只有29/806匹配。因此，旧raw+object增量很小不能否定重建路线。下一轮先检查候选关联与整轨质量，再增加对象分类头。方向目标也应先在局部轨迹层验证，当前event-level方向回归尚不能支撑可靠missing-pT。

评价上应同时保留多类决策、单类排序、条件分母和真实先验。较高argmax recall可能来自更多事件被分配到该类；高总体AUROC也不保证稀疏子域排序好。重复使用Tune得到的区间只描述开发结果，最终收益由封存Confirm确认。

### 本次新增的稀疏事件读回

[进展快照](r2_progress_review.json)形成于2026-11-15 10:51:48 UTC，只读取六份R2 manifest中的固定4k screen历史；对每个窗口的低occupancy混淆计数、逐类recall和BA重新核算。AP/AUROC沿用保存记录，没有重新读取事件预测。低occupancy定义始终为原accepted pixel+bar+tracker channels<100，包含空响应；真e/μ/τ/NC分母为285/275/342/309，条件三类的μ占比为275/869=31.6456%。

例如D0 T3在75k→150k→225k→300k非τ曝光时，低occupancy μ recall为7.6364%→48.7273%→10.9091%→8.3636%，对应AUROC为.528130→.565069→.558096→.550168。全域NLL3从.933269降至.879378。D1 T3在300k中间曝光时recall达到87.2727%，AUROC仍为.568411。曲线见`R2/figures/r2_sparse_muon_progress.pdf`。

这支持同时检验训练/决策偏移与局部特征不足，尚不支持把稀疏μ问题全部归为分类阈值，也不支持宣布信息物理上不可分。D1/D2使用更长的完整LR轨迹，其中间点不能替代最终预算比较。快照时D0 T3已完成，其他五模型仍在执行；状态会继续变化，最终以完整40k Tune结果和验收为准。

[stage279](../../workbook/2026-11-15_stage279_r2_d0_readout_diagnostic.md)已补充D0两个完整40k Tune终点及独立sklearn排序读回。T3/conditional T4的BA3为56.6500%/56.5333%；三类化增加μ recall但整体BA差值区间跨零，NLL更差。低occupancy μ AUROC为.545742/.543902。按6,845个physical chunk分五折、每折每类2k事件后，两个全局bias未改善整体或稀疏BA；正温度降低NLL而保持argmax。该固定诊断复用到后续全部endpoint，仍属于Tune开发结果；[紧凑数据](r2_readout_review.json)保留合同、逐折拟合与指标。它支持优先检验局部表示，未排除更一般的条件readout。

## 3. 当前批次：完成R2并作一次有依据的分流

R2完整交付包括B1 seed42的fresh D0=100k×4、D1=100k×12、D2=300k×4及两个D1WD控制，共8模型、16个endpoint/best-screen视图、128配对、11域；另有同prefix末轮的两次重放和8项配对。198项验收通过，全部allocation终态。完整数值、科学决策和成本见[stage284](../../workbook/2026-11-15_stage284_r2_final_results.md)与[结果JSON](r2_results.json)。

T3使用对应四类pool的全部非τ事件；D1/D2在T3上均为900k曝光、T4上均为1.2M曝光。尾批造成小量update差，不能称精确fixed-update。各模型fresh初始化，按完整updates定义warm-up/cosine；endpoint是主比较，own-task minimum screen NLL是次比较，不混合选点机会数。

| Question | Actual evidence | Decision |
| --- | --- | --- |
| 新数据还是重复旧数据？ | T3 D2−D1 BA3 +3.927 pp [3.381,4.439]；D2−D0仅+.447 pp [.078,.815]，且e/NC recall下降 | 当前配方更多unique有益；相对D0未达2 pp实用门槛，不自动扩1M |
| 更强正则化是否解决长训练？ | D1WD−D1的T3/conditional T4 BA3 +1.443/+1.247 pp，gap缩小；两个固定终点仍低于D0 | 单个系数改善成立，未建立最佳正则化或长重复训练优势 |
| 末轮偏移是否只是猜测？ | 固定第11轮prefix，原顺序概率精确复现；重混末轮BA3 +1.277 pp [.810,1.803]、NLL3−.049451 | 顺序效应成立。下一批M0先测fresh D2 T4全程重混，采样与结构收益分别配对；不能将本控制称为fresh12轮结果 |
| 稀疏分离是否已恢复？ | D2 T3稀疏μ AUROC=.5483、T4=.5518；末轮重混稀疏macro AUROC差区间跨零，recall大量再分配 | 稀疏表示仍优先，不能用μ recall上升替代净分离改善 |
| 应否只做三类？ | D2 T3−conditional T4 BA3 −.030 pp [−.393,.341]，T3 NLL更差；μ/NC仍有取舍 | 交付真实T3并保留T4，未建立所有逐类非劣 |
| τ增益是否足够？ | D2 T4在1% FPR效率6.05%；对D0增加.940 pp [.419,1.566] | 正效应成立，点值低于1 pp门槛；保留研究，不宣布独立确认或直接增加flat类别 |
| 是否已经收敛/容量充分？ | 八模型均未达到连续三窗口NLL变化<.001；D1退化、D2仍波动；width16真实32-event拟合通过 | 当前批次完成不等于充分收敛。D3/width32/1M不在本批扩展，容量和规模上限仍开放 |

后续参考固定为`b1_d2_t3_s42/exposure.pt`和`b1_d2_t4_s42/exposure.pt`，共同BA3分别57.0967%/57.1267%，T4 BA4=45.7650%。这指本批已测原始endpoint内的参考，不称全局最优。所有区间条件于已拟合seed42，未估计训练seed与pool抽样变化；Tune是开发资料，Cal-A/Cut-B/Confirm继续封存。

D0/D2两个global bias没有净BA改善；D1WD T3/T4分别恢复.78/1.25 pp。正温度改善部分NLL但保持argmax。这个低维诊断区分置信度、决策偏移与表示限制，不排除更一般readout，也不代替R7校准。前序记录见[stage279](../../workbook/2026-11-15_stage279_r2_d0_readout_diagnostic.md)、[stage282](../../workbook/2026-11-15_stage282_r2_more_unique_data_results.md)和[stage283](../../workbook/2026-11-15_stage283_r2_regularization_and_tail_order_control.md)。

## 4. R3–R7的继续路线

| Stage | Concrete work | Exit criterion |
| --- | --- | --- |
| R3: measured relations | M0先检验D2 T4全程重混，冻结采样配方；B1再检验首层stride4→2及相位；原生族单独检验压缩前跨窗口/跨plane几何边 | 采样与结构分别配对；主任务、逐类/occupancy和成本。最有价值改变补三seed，再做native/100μm/1mm |
| R4: objects and tracker | 增加真实station spacing、方向相容与测量不确定度；多候选关联/robust fit控制，再择一learned edge或instance方案；raw路径覆盖全部事件 | candidate recall、整轨efficiency/fake/duplicate/coverage、vertex与方向残差均报告；raw+object增益包含重建失败事件 |
| R5: tau and labels | 先flat4、tau gate+conditional3、conditional decay辅助；优先τhad specialist，τe/τμ按混淆推进；完整非τ背景及prompt lepton识别一起评估 | 合并后τ工作点改善，共同三类非劣；分支与inclusive效率分别报告；不使用truth路由 |
| R6: targeted learning | 仅在胜出encoder上追加一项连续性/局部监督或topology MAE；同总计算监督为控制；加入已接受幅度的readout训练扰动 | 下游实际收益或数据效率成立；名义与未见扰动pattern同时评价，不以pretext loss下降晋级 |
| R7: independent confirmation | 冻结候选、权重、先验、主比较；Cal-A拟合概率、Cut-B选择阈值，最后打开200k Confirm | 最多两项主要比较、多重比较控制、逐类与τ工作点、系统变化和最终资源完整交付 |

R3的B1降采样与原生几何边是两条独立干预，分别使用本族控制；不把二者同时更换后的差值归为单一机制。G1a的重点是关系发生在压缩之前/窗口之间，以及有符号方向、uncertainty和材料信息，承接旧S2跨层对照。

R5分两条物理路径。OPERA支持在可见母轨/次级顶点上结合1h/3h、电子和μ通道特征，背景必须包含charm、强子再相互作用和大角度μ散射。NOMAD则提供“无法分开τ衰变点时仍利用可测末态运动学”的先例；本项目须先建立方向/动量的resolution–coverage和接受度闭环，才能把它作为实际分类输入。[H7/H8](05_literature.md#h7)说明文献依据。没有可分辨kink不等于没有τ信息。

更多类别的顺序为：四类主头→条件e/μ/had衰变辅助→有充分标签与对象证据时再研究had1/had3。现有`tau_direct_prongs`统计同衰变点的直接子粒子中有多少带电粒子，尚未证明等同于经共振展开后的最终charged prongs或实际重建轨数；不能直接拿它启动七类均衡softmax。未知子道只mask子道loss，继续保留coarse τ监督。新增细分最终仍按合并四类与τ工作点评价。

## 5. 资源和完成标准

资源优先顺序由预计完成时间决定，HTCondor与Kubeflow并用，接受异构GPU、共享设备和MIG。R2首批已实际使用H100 NVL MIG 3g.47gb四进程、H100L MIG 2g.24gb双进程，单卡四进程profile提高了总吞吐。并行度须结合真实输入、RAM、显存和完整validation峰值选择；不以占满进程数代替有效训练吞吐。同卡进程共享一次输入staging，各模型独立optimizer/checkpoint，成本按allocation计一次。

数值配方按对照保持清楚即可，无需等待同型号或整卡。T4的FP32、A100/H100的BF16和V100兼容构建都可按实测使用；若改变精度/软件配方则补必要基线。当前LCG108没有SM70内核，这是该软件构建的限制。运行中的有效训练继续完成；排队副本的迁移保留单writer和实际成本记录。

R1实际batch成本45.708889异构device-hours，已超过原18 h规划；R2一族及必要控制实际7.475833 h、116.65 allocated CPU core-hours，39条allocation全部终态。按R0上界+R1/R2 batch累计54.269577 h，原80 h规划名义余量25.730423 h，交互完整allocation开销仍未测量。后续按实测小批次重排，保留R7确认资源。没有实测收益的复杂分支暂缓，不把项目交付绑定在τ成功或全部候选训练完成上。

统一晋级值沿用[02](02_tasks_and_evaluation.md#6-统计设计与晋级门槛)：三类ΔBA3至少2 pp并满足NLL/逐类约束；τ在1%背景工作点Δ效率至少1 pp、区间支持正效应且共同三类非劣。更小收益如实记录，用成本与科学问题决定是否追加。最终交付至少包含合作组三类对照、保留/暂缓τ的证据、已选模型的完整概率与工作点、新独立确认、可复现配置和资源账本。

旧14份计划保存在[完整v1归档](../archive/2026-11-14_v1/README.md)，历史数值和失败/修复记录保留。当前路线是v2的R0–R7；R0/R1/R2已完成，R3–R7为设计。后续入口为[总路线](../README.md)、[执行工单](04_execution_plan.md)和[工作记录索引](06_workbook_index.md)。
