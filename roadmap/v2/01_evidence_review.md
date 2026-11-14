# 当前结果的独立评估与系统性审查

**现有方法有进步，但最优性没有建立。最有价值的下一步是补齐公平基线和训练饱和研究，并直接建模跨层测量关系。τ 暂时采用独立研究分支，同时把三类任务推进到可与合作组比较的水平。**

审查基准为 `37dd763`，覆盖 236 份工作记录的全量编目与结论摘录、14 份旧路线图、5 份本地 PDF（80 页）、FASERCal 本地 TeX、阶段结果 JSON、当前数据表示/模型/损失/评价代码。重点证据核对范围为 stages 03–07、41–55、65、75、80–87、90、145、163、167、170、177、181–183、189、194、196、199–205、206、219、231–236；其余记录用于追踪执行与故障演变。没有逐项重跑 236 份报告中的实验。完整记录入口见[索引](06_workbook_index.md)。

本轮新增计算只读取已有摘要、六个 revision 3 主模型的 history/manifest 和各 10k validation 预测。复核工具为 [review_current_results.py](../tools/review_current_results.py)，结果为 [current_results_review.json](current_results_review.json)。没有新增训练或重新选择 S6 test 工作点。

## 1. 已成立的改进及其幅度

| Evidence | Measured result | Interpretation |
| --- | --- | --- |
| S6 independent test, 724,441 events | MAE3 ensemble BA 0.435226，CI95 [0.433705, 0.436685]；macro AP 0.475020；macro AUROC 0.705487；NLL 1.201590 | 有效学习，绝对分类能力仍有限 |
| MAE3 − Scratch, same 100k label protocol | ΔBA +0.003354 [0.002508, 0.004166]；ΔNLL −0.007487 [−0.007851, −0.007094] | 预训练的小幅泛化收益成立 |
| Tau at frozen validation 1% background cut | Test efficiency 0.035556 [0.034211, 0.037063]；actual FPR 0.012080；2,331/65,558 τ 通过 | 低背景 τ 能力弱；实际 FPR 应与效率同时读 |
| Rare-tau scenarios at their frozen cuts | τ prior 1%/0.1% 时 purity 2.8099%/0.2867% | 当前选择不适合直接形成高纯稀有 τ 样本 |
| S6 tracker − native control | ΔBA +0.026564 [0.025494, 0.027827] | Tracker 输入收益成立，主要帮助共同三类任务 |
| S6 native − 1mm control | ΔBA +0.000511 [−0.000539, 0.001444]；Δτ efficiency +0.000061 [−0.000750, 0.000885] | 当前编码器下尚未证实原生粒度改善硬分类/τ 工作点 |
| S5 vertex+kinematics, three seeds | ΔNLL −0.001956 [−0.003363, −0.000725]；Δτ AP +0.000547 [−0.001915, 0.003037] | 小幅概率收益，τ 排序增益未建立 |
| S5 six-way, three seeds | Δτ AP −0.004764 [−0.008558, −0.001192] | 该 10k-label flat6 配方不适合作为默认替代 |
| S6 six − direct4, seed42 | Δτ efficiency +0.001464 [0.000366, 0.002556]，BA 区间跨零 | 六类在一个工作点有收益，不能概括为所有细分都失败 |

来源：[S6 完整报告](../../workbook/2026-11-14_stage236_s6_final_results.md)、[S6 JSON](../results/s6_summary.json)、[S5 完整报告](../../workbook/2026-11-14_stage203_s5_final_results.md)。区间是既有权重/校准/阈值条件下的物理块区间；seed spread 在原报告单列。

S2 也保留了更局部的正证据：ParT native 相对 100μm 的 τ AP 增量 +0.008325 [0.002820, 0.013780]；ParticleNet 加 bar 相对 pixel-only 的 τ AP 增量 +0.068908 [0.056633, 0.084432]。这些是各自表示/预算下的贡献，不能与 S6 的 native−1mm 硬分类结果混为同一个问题。见 [stage55](../../workbook/2026-11-12_stage55_s2_final_acceptance.md)。

S4 的低标签收益比 100k 更大：最终 seed42 五初始化矩阵中，MAE3 相对 Scratch 的 macro AP 增量在 1k/10k/100k 分别为 +0.033039/+0.025899/+0.005324。项目目前拥有大规模带类别标签模拟，下一目标应由“少标签表示是否有用”转向“充分使用监督信息后的最高可达性能和稳健性”。低标签研究仍可服务新几何、精细 truth 稀缺或未来实测数据场景。

## 2. 三类表现是否足以解释四类困难

本次使用 S6 预定的六个最终 revision 3 模型已有 validation 概率；同初始化的三 seed 原始概率均值构成 ensemble，未重新拟合温度。

定义：先在评价中去除真 τ，再令 `q_k=p_k/(p_e+p_mu+p_NC)`。这是固定四类模型的**条件三类诊断**。下表保留原域的类混合；BA 为各类 recall 的平均，AUROC 为 one-vs-rest。

| Validation domain | Non-tau counts (e / mu / NC) | MAE3 conditional 3-class BA | Scratch BA | MAE3 AUC (e / mu / NC) |
| --- | --- | ---: | ---: | --- |
| All | 2,500 / 2,500 / 2,500 | 0.546933 | 0.543733 | 0.809711 / 0.723572 / 0.718968 |
| Pixel-centered | 245 / 237 / 227 | 0.744517 | 0.741334 | 0.932970 / 0.890671 / 0.864807 |
| Large scintillator | 1,131 / 1,183 / 1,127 | 0.667814 | 0.667343 | 0.909284 / 0.823739 / 0.802449 |

同一 MAE3 ensemble 在全部四类 validation 上 BA=0.434300；pixel-centered 四类 BA=0.584691。类别数、分母和先验同时变化，不能把两列差解释成三类重训的净收益。

**事实：**去除 τ 后，全域的 e/μ/NC 区分仍有较大提升空间。**判断：**项目的低性能不只由 τ 引入，接受度、输入关系、训练规模和基线设计都需要推进。**未测量：**同预算实际三类重训相对条件三类的增益、与合作组事件完全匹配后的模型差异。

空响应也不能解释全部差距：S6 test 仅 18,852/724,441 个事件完全为空；主模型从全域 BA=0.435226 到 active 域 BA=0.438654，只改变约 0.00343。较高的 pixel-centered 表现提示需要研究几何/可见性，但不能把所有 active 事件的困难归为无响应。R1 应同时给 presence/occupancy 简单基线、模态增量和条件域分母。

## 3. 与合作组及 FASERCal 的可比性

| Source | Relevant evidence | Comparison to implement |
| --- | --- | --- |
| [Mahlstedt 2026-09-16](../../docs/FORTVNE_pixel_vs_scintonly_meeting_260916.pdf), PDF p2–7 | 三类；相同 validation；pixel-centered 加 pixel 后 AUC 0.944/0.842/0.801→0.970/0.906/0.876；large 增量较小 | 统一 fiducial、样本、先验和工作点，再测 pixel 增量 |
| [Wu 2026-10-21](../../docs/FORTVNE_10212026.pdf), p2–4 | 1mm 双投影 CNN/ResNet；40 epochs；M1/M2/M3 输入组合；表中效率/纯度为不同选择点 | 在本批 accepted readout 上实现同构基线，同时保留 tracker hit-set 与 reconstructed-feature 对照 |
| [Mahlstedt 2026-10-07](../../docs/FORTVNE_Update_kinematic_variables_2026-10-07.pdf), p2–9 | pT 与 azimuth 表现不同；精细方向依赖 reconstruction coverage | 分量/角度、质量与覆盖率同时评价 |
| [FASERCal arXiv v2](../../docs/FASERCal/arXiv-2604.07037v2.tar.gz), Methods / Supplement | 31.936M 参数，约 1.226M pooled events，稀疏 3DCal/AHCAL、ECAL、spectrometer；MAE 400 epochs、Rel 100 epochs | 迁移局部结构与监督设计，独立检验 PINPOINT 输入和规模 |
| [FASERCal ICHEP](../../docs/FASERCal/ichep-sam-26-07-30.pdf), 19 pages | 多探测器 MAE→Rel→joint fine-tuning 的整体展示 | 以论文 Methods 确定精确输入/损失，报告用于理解流程 |
| [FASERCal workshop](../../docs/FASERCal/FASERCal_wshopRun4.pdf), 2025-11-14, p19–25 | τhad preliminary 约 11% 分支效率、66% purity；有自身 flux/exposure | 用于设计 τhad specialist；其分支效率和产额不能换成 PINPOINT inclusive 性能 |

当前 pixel-centered 条件三类 AUC 已接近部分旧三类指标，但旧研究样本数、μ/NC 混合和训练输入不同，尚不能宣布超过或落后。FASERCal 的 detector information、样本规模、参数量、训练曝光和评价先验都不同，其高 AUC 不构成把现有代码继续加深即可达到相同效果的证据。

## 4. 已确认的设计不足

### A1. 同输入最强架构排名仍缺失

修复 S3 中，pixel+bar-only 的 fresh ParT 三 seed macro AP=0.45434，高于 multiscale native 的 0.43843；加 tracker 的 multiscale 为 0.46338。后一比较同时改变硬件输入，不能证明分层 backbone 优于 ParT。S2 ParticleNet+tracker 的 BA 点值也说明它值得重新进入同输入竞争。应给 CNN/ResNet、ParticleNet、ParT 和当前 backbone 相同三个模态、相同事件及相同调参额度。

### A2. 关系建模与“保留全部通道”之间有缺口

[dataset.py](../../src/analysis/flavour/s3/dataset.py) 先按同 plane 的 pixel window 分块，每块最多 32 channels；region 按 module 内 canonical block 顺序每 32 blocks 分组。它不是按跨层方向或真实三维邻接构造的区域。

[models_repair.py](../../src/analysis/flavour/s3/models_repair.py) 的 multiscale 设置中 `context` 为空；原始 channel 的两层局部 attention 只在块内进行。之后经过 local mean/max、每 region 两个 learned queries、64 global latents，再进入全局 attention。regional/local skip 使所有块到达分类，但它们也经过 pooling。模型可以通过 latent 间接学习全局关系，尚未具备显式跨层 hit/cluster 相容边、方向残差或材料传播约束。

region 几何 embedding 显式使用 module、detector 和 z；xy 信息主要依赖局部编码保留，没有显式 region centroid/covariance 的相对几何 bias。短轨、相邻窗口边界、跨 plane 连续性和低幅 signed 方向是有针对性的改进对象。上述结构事实不能单独证明某一关系层必定提高性能。

项目此前已有跨层控制：[S2 stage54](../../workbook/2026-11-12_stage54_s2_cross_plane_controls.md) 的 ParT/ParticleNet 在 block compression 后启用跨层 attention/graph，τ AP 增量分别 +0.008031 [−0.000494, +0.016718]、+0.005745 [−0.001693, +0.013147]，主区间均跨零；ParT BA 点值从 0.3928 到 0.4184，相关差值区间未测量。G1 的新问题是能否在当前分层压缩前/跨窗口处引入方向相容、测量不确定度和材料传播关系，而非再次只切换一般 global cross-plane 开关。历史结果继续作为有效控制。

### A3. 规模与优化尚未形成闭环

六个最终 100k-label 模型均为 897,992 参数、4 epochs；全部 best=4，末轮 NLL 变化为 −0.00203 至 −0.00779。当前证据支持继续测饱和曲线，不能直接断言已经过拟合或容量耗尽。

S4 曾完成 matched-compute 长监督控制，特定配方在 1k/10k 过拟合、100k 未建立 MAE 同等收益（[stage163](../../workbook/2026-11-14_stage163_s4_compute_control_results.md)）。这项结果需要保留；它覆盖固定低标签/调度配方，尚未回答更多 unique 带标签事件、独立学习率选择和更长收敛训练的效果。

训练 manifest 记录的 data-wait/总耗时为 31.16%–46.64%，是 loader 计时口径，不能当作 GPU idle 百分比。应复用既有等价缓存/分桶经验，重新 profile 新的较大样本流，先处理重复变换和数据供给。

### A4. 部分运行元数据与实际结构不一致

六个最终 manifest 的 `local_blocks` 仍写 `same_plane_4mm...`，实际配置为 `window_pixels=16`，native pixel window 约 0.3328×0.3648 mm；`cross_plane_message_passing=true` 也不能说明 multiscale 启用了显式跨层几何层。R0 应从实际 transform/model 生成版本化元数据。此次审查保留旧产物原文，并以配置和代码判读实际执行。

### A5. 对象构造尚不能支撑拓扑结论

[objects.py](../../src/analysis/flavour/s5/objects.py) 的 pixel 轨段主要使用相邻可用 plane 的 mutual-nearest xy 关联和至少三点链。S5 实测 segment purity 约 2.0%–5.4%，primary 双轨 vertex 匹配 29/806。该算法对高密度 shower 的组合 fake 严重，且双轨 vertex 指标没有覆盖主要的单 prong kink 问题。raw+objects 收益有限只约束这一候选构造，不约束所有 tracking/object 方法。

### A6. 辅助任务层级和目标可观测性不足

S5 的 semantics/charm/kinematics 多数从 event pooled feature 读出；FASERCal 的 Rel 强调局部 voxel 的 soft semantics。当前 bar Rel 已加入局部 channel 监督，但并不等于完整 pixel/bar/tracker 的 instance/continuity 监督。把 event-level 辅助损失继续叠加，未必能恢复已经压缩掉的局部信息。

visible-primary target 是生成层 proxy，并非接受度内真实 calorimetric observable。方向 MSE 容易收缩到条件均值；S5 Δφ absolute median=1.385622 rad，实际方向能力仍弱。新辅助目标应先验证 readout 可读性、mask 和局部对象层级。

### A7. 选择指标与物理目标有偏差

既有主选点多用整体 NLL，而 τ 在低 FPR 尾部的收益很小；“最小 NLL”“最大 τ AP”“固定 FPR 效率”不是同一目标。0.1% 工作点原 validation 只有 7/7,500 背景通过，阈值估计不稳定。新路线应使用更大的独立 threshold subset，并预定主目标和整体任务非劣界。

### A8. 旧 test 已成为公开研究资料

S6 独立 test 对旧冻结设计仍然有效；其位置/occupancy 诊断已经影响本次架构设计，后续不能重复把同一 test 当新模型最终独立证据。新确认集应从未用于训练、SSL、调参或诊断的旧 train/validation 余量中按 accepted family 清单冻结；可用性需由记录核查，已有生产认证继续有效。

### A9. 读出扰动的影响大于当前方法增益

S6 固定 32k 子样本中，sparse-noise proxy 使 BA 下降 0.017750，combined 条件下降 0.016063；名义 MAE−Scratch 的增量为 0.003354。它们样本和对照不同，不能直接相减作为净收益，但足以把鲁棒性列为关键质量维度。此前已有 augmentation，下一步需定位新增孤立通道对 token 数、pooling 和局部关联的影响；不能简单删除所有稀疏信号，因为 τ 短轨/孤立轻子也可能稀疏。R1/R2 先在新开发事件上保留固定 proxy 压力对照，R6 再扩展鲁棒优化和未见条件确认。

## 5. 故障历史、后续修复与待检验假设

旧 S3 Head 失活、空响应梯度异常、LP-FT 编译断梯度和 frozen solver 未收敛均已有真实修复及重训。当前优化应以 revision 3 的有效学习模型为基线，旧故障结果继续留在诊断历史。

S3 的 global 分支依赖弱，S4 MAE 已有正向纠正证据：[stage170](../../workbook/2026-11-14_stage170_s4_valid_path_results.md) 的 10k MAE global 置换 ΔNLL=+0.060310，frozen global probe 也可读类别。不能沿用“global 从未学习”的旧判断。当前缺少的是对应 100k 主模型的重训删除/替换对照，以及跨层关系是否带来额外可用信息。

| Hypothesis | Existing clue | Experiment that can distinguish it |
| --- | --- | --- |
| Compression loses rare topology | 高 occupancy τ 效率低；native/1mm 工作点接近 | 固定样本/预算，增加边界通信和几何边；训练后做条件配对 |
| Model is not trained to saturation | 六主模型均末轮最优且 NLL 仍降 | 100k 长训练；300k/1M unique-event 曲线，分别固定 updates 与 passes |
| Global and local paths compete | 早期 global 弱；MAE 后使用增强 | 同容量重训 ablation、独立分支 probe 和梯度对齐诊断 |
| Some tau decays lack resolvable topology | S5 873/2,500 无已标 daughter response；低占用/几何强依赖 | 观测 proxy、truth-conditioned 诊断和单粒子/少粒子可分辨性扫描 |
| Spectrum/position shortcuts dominate | 条件支持匹配改变增益；pixel-centered τ 尾效率很低 | 分开匹配 energy/vertex/target；占用作为观测机制另做诊断，不一并当混杂消除 |

当前事实足以优先处理这些问题，尚不足以证明 PINPOINT 的四类信息上限。具体实验与决策见[任务合同](02_tasks_and_evaluation.md)、[模型设计](03_models_and_physics.md)和[执行工单](04_execution_plan.md)。
