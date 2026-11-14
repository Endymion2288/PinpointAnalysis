# S4：可验证的预训练与标签效率

状态更新（2026-11-14）：空响应修复矩阵已完成18项分类重训、完整10k validation及4模型分支诊断，21项技术检查全部通过。235,008次逐步梯度检查、2,916次激活采样、252项checkpoint检查和18次独立best FP32 backward通过；最大preclip norm为188.390472，未重现旧空占位约10¹²的异常。两组同时使用`empty_response_bypass=true`；新配置为[configs/s4_empty_repair.json](configs/s4_empty_repair.json)。MAE3复用此前已读回的3套encoder重新fine-tune，本轮没有重训SSL encoder本身。旧故障结果保留作诊断，不作为新结论。完整结果见[阶段145](../workbook/2026-11-14_stage145_s4_empty_repair_final.md)、[精简数值](results/s4_empty_repair_summary.json)和[分支诊断](results/s4_empty_functional_paths_seed42.json)。输出根为 `/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13`；61个allocation均终止，累计账本24.325865/60 device-hours。本轮不扩1M，优先解决低标签过拟合和τ决策问题。

## 1. 研究问题与固定条件

1. S3 的 global 分支虽有非零梯度，tracker 模型中置换 global 输出的 probe NLL 增量为 −0.001597；预训练能否学到可线性读出的跨探测器表示，并在实际分类中得到利用？
2. tracker 对 νμ、NC 有增益，ντ AP 增量的区间跨零。预训练是否改善 ντ 及其 e/μ/had 子道的混淆，还是主要学习 shower occupancy 和能量先验？
3. 低标签收益能否覆盖新增资源成本？同样预算继续监督或 MAE 是否已能取得相同结果？

沿用已接受的 detector-proxy、geometry 和 family split。既有 80k train、10k validation 原样保留，另从八个新 chunk/类选择 5k train/类，组成 100k train。所有 SSL 输入、重建验证和 probe-fit 均来自 train；独立 validation 只用于下游选择/评价。Calibration/test 不打开。

原始S4合同的共同backbone为S3 revision 2 multiscale native + tracker：Peri-LN、QK-LN、LayerScale、FP32 normalized SiLU Head、全部local blocks直接读出。本次18项分类同时采用revision 3的`empty_response_bypass=true`，由Scratch控制隔离预训练初始化的差别；原冻结配置用于复现历史。导出features的接口与checkpoint参数命名兼容，预训练decoder/teacher在fine-tuning时丢弃；分类heads对所有初始化策略使用相同种子重新初始化。

## 2. 文献依据与适配

| 来源 | 采用的方法 | PINPOINT 的适配 |
| --- | --- | --- |
| [FASERCal, arXiv:2604.07037v2](https://arxiv.org/abs/2604.07037) | MAE、Rel、同下游 Scratch 对照、标签效率 | binary pixel 不预测 analog charge；Rel 仅使用可精确累加的 bar 贡献，明确记为 partial Rel |
| [NOMAE, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Abdelsamad_Multi-Scale_Neighborhood_Occupancy_Masked_Autoencoder_for_Self-Supervised_Learning_in_LiDAR_CVPR_2025_paper.html) | visible-neighborhood occupancy、正负候选、多尺度 | 查询位置独立于 hidden support；额外加入跨层和全几何 uniform 候选，保留孤立短轨的非零覆盖概率 |
| [Sonata, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_Sonata_Self-Supervised_Learning_of_Reliable_Point_Representations_CVPR_2025_paper.html) | self-distillation、geometric shortcut 诊断、linear probe | teacher 仅看完整硬件 observation；不任意旋转或缩放探测器，不将 τ daughter 删除后的视图当作等价完整分类事件 |
| [VICReg, ICLR 2022](https://arxiv.org/abs/2105.04906) | feature variance/covariance 约束和坍缩监测 | 与 EMA 对齐共同使用，检查各分支跨事件方差、有效秩和梯度，不能仅凭 loss 下降验收 |

2026-11-13 已联网获取上述 CVPR 原文、VICReg 原文及 CERN ML training/flavor 文档，来源与时间保存在输出根 `literature/sources.json`。FASERCal 全文与 S3 稳定性文献复用既有缓存。

## 3. 遮蔽、重建与 Rel 合同

遮蔽先作用于 observation，再重新建立 local/region tokens、presence 和全部 summary。遮蔽规则由固定几何 cell ID 和 epoch seed 决定，对空/有响应通道一致；不把 hidden active 坐标、总数、模态总信号或原始 group layout 提交 encoder。先比较 0.5/0.75，增加整 plane/panel 和 detector 遮蔽；不以 truth 选 mask。

Decoder queries 仅由几何和 visible 邻域产生，混合邻近、跨层与 uniform 候选；候选集合先生成，再查询 reconstruction targets。必须报告每种候选的正例比例、loss、PR-AUC、reconstruction coverage，声明优化的是所定义混合查询分布，不能称为均匀全探测器 risk。bar signal 只用接受的能量/PE proxy。负候选不能按 hidden support 筛除。直接依赖 visible geometry 的 trivial baseline 与 encoder-permutation baseline 同时报告。

首轮 512-step pilot 中，256 个 train-inner probe events 的逐 native pixel 查询仅有 2–3/16,384 正例，不能用其 loss 下降支持细结构学习。修正后的目标为 16×16 native-channel patch occupancy、binary channel count、patch 内 native x/y 位置矩，另保留 4-mm occupancy/count；**encoder 输入仍为逐通道原生观测**。64-event 同查询/同 masks 对照的 native/8×8/16×16 正例数为 0/28/83。新的监督尺度需重新通过真实事件拟合与 corpus pilot，不沿用原目标的训练权重。逐 native pixel target 的统计不足作为明确的实现/物理分辨率约束保留。

EMA 单独增加 teacher-multiscale 对照：完整 observation 的 local/regional/global pooled features 等权形成 teacher target，student 仍从 global latent 预测，检验 global-only EMA 是否只重复 occupancy 主轴。与原 global-only teacher 分开运行；不将多个目标改变合并归因。

`ScintSD.cc` 的 `barEnergyMap` 以 channel/track/PDG 为 key 累加；ROOT `Hits/scintHits` 保存相应 `pdg/edep`。额外 sidecar 按硬件 channel 累加 EM(11/22)、muon(13)、hadron、other 的能量分数，正能量有效通道才产生目标。质量检查仅针对新增标签的 channel join、能量一致和 simplex，不重复样本认证。Pixel 首贡献 flags、ghost/hierarchy 不冒充精确 soft truth；本轮不用 pixel Rel。原 global-only Rel pilot 的语义 cross-entropy 为 0.73062，未超过常量 probe prior 的 0.72388，已停止扩量；新 Rel route 允许读取全部 kept bar local blocks，必须先通过独立语义学习 readback（见 stage96）。MAE reconstruction 仍只读取 global latents。模拟贡献只进 loss，不进入 input、mask 或 query 位置。

MAE/EMA仅从observation构造目标；partial bar Rel另使用98k pretrain-fit事件的模拟channel语义，及2k原train内probe的语义评价。Rel衡量事件类别标签效率与额外模拟语义监督，不能称为仅用1k/10k总标注的纯自监督比较。相同continuation steps/views不等同于相同FLOPs，附加decoder和语义目标的计算量另列。

## 4. 分阶段实验与预算

S4 本轮组织上限 **60 device-hours**，包括所有后端的 pilot、编译、失败、训练、probe 和评价；不同设备、MIG 与整卡分别列出，时长不换算成等价 FLOPs。已累计24.325865 device-hours、282.757649 CPU allocation core-hours，包含历史尝试、初始化和保守结束上界。以下为研究工单；当前完成范围为修复后的18项full fine-tuning及完整路径/分类评价，新增路线下的frozen probe、EMA/Rel及compute-matched比较尚未重测。

1. 完成 100k corpus、独立 Rel sidecars；train 内嵌套 98k pretrain-fit/2k reconstruction-probe。监督标签预算为嵌套的 1k/10k/100k，四类平衡，τ 子道按原样本比例分层。
2. 导出 S3 encoder 的 local/regional/global/summary features，在 train 内 fit/probe-validation 上分别拟合 linear/MLP，测可读信息与实际梯度的差别。不能用 64-event 置换取代分类 probe。
3. 真实 32-event 拟合、FP32/BF16、最大 occupancy forward/backward、无泄漏、负候选、EMA/variance、checkpoint/resume 通过后，正式 pilot 比较 mask 0.5/0.75。重建内部验证及分布/coverage 决定主 mask；全零预测不作通过结果。
4. 同 backbone 比较 Scratch / MAE / EMA self-distillation / MAE+barRel；另保留 MAE continuation 和继续监督的 matched-compute 控制。相同 fine-tuning 设置与独立初始化 heads。先 seed 42，关键候选与 Scratch 用 17/42/2026 确认。
5. 每个标签预算同时交付 frozen linear probe 和 full fine-tuning、learning curves、checkpoint 读回、内部健康和总成本表。完整 validation 为 10k，四分类指标、各类 1%/0.1% FPR 效率、τ 子道及 common-support 条件指标齐全。0.1% FPR 的低通过背景数按实测计数报告。
   补充固定 checkpoint 的完整 10k validation 分支置换：三个预先固定的 modality-preserving donor mappings，逐项 global/regional/local/summary，并以同精度 baseline 和 paired chunk bootstrap 比较。用于检验实际拟合路径依赖，结合 frozen readability 解释；不当作重训 ablation 或物理因果重要性。
6. 只有重建健康且下游收益与资源证据支持时才扩至 1M；无收益也完成本轮对照与统计结论，不盲目扩量。1M 是否执行及原因写入最终验收。

## 5. 调度与验收

批量转换只用 HTCondor；选择 schedd 时保存即时 load、EOS policy、WRITE 结果。训练可由 HTCondor/Kubeflow 竞争领取互斥 task claim；使用平台规定的 GPU product/resource flavor，不在 GPU Pod 中自填 CPU/RAM。并发数以真实 CPU/data wait、显存和 GPU utilization 调整。正式长训练不得从 SSH 临时启动或暂停 PID 1 进行迁移。

验收除结果文件完整外，要求 informative encoder 梯度、非饱和 head、非恒定预测、SSL 的 feature 方差和实际拟合证据；若出现明显异常，先定位架构/尺度/采样/优化并修复重训。最终记录所有 allocations 终态和资源成本；阶段报告与代码共同提交。

当前配对冻结于[configs/s4_empty_pairs.json](configs/s4_empty_pairs.json)，全部9项配对和3项三seed均值已完成。MAE3−Scratch的raw结果如下，区间条件于已训练seeds和固定validation；训练seed SD在精简JSON中单列，均值不是ensemble。

| Labels | Macro AP: Scratch → MAE3 | Δmacro AP [CI95] | NLL: Scratch → MAE3 |
| ---: | --- | --- | --- |
| 1,000 | 0.288856 → 0.345980 | +0.057124 [0.052411, 0.062256] | 1.564575 → 1.447382 |
| 10,000 | 0.425891 → 0.440384 | +0.014493 [0.010374, 0.018721] | 1.276154 → 1.252464 |
| 100,000 | 0.469263 → 0.473555 | +0.004292 [0.002020, 0.006590] | 1.214263 → 1.207805 |

1k平均NLL仍高于ln4=1.386294；1k/10k的τ argmax F1差值分别为−0.032139/−0.011317，物理块区间为负，且seed波动较大。三个预算的τ低FPR效率差值区间均跨零。100k raw τ AP差值+0.006024，[0.001484,0.010809]；common-support总体τ AP及macro AP收益未确认。Hadronic τ子道在common-support的AP差值为+0.015837/+0.009678/+0.004792，区间均为正，electron/muon子道未确认改善。排序收益不能替代概率、硬决策和高纯度工作点质量。

新MAE3 seed42在1k/10k完整validation中，global置换的NLL增量为0.055214/0.025673，区间均为正；Head重放误差均为0。Scratch 10k对应效应只有0.00002515，实际很小。Local/regional与summary诊断保留在完整表中。这证明特定checkpoint的拟合路径依赖，不是重训ablation或物理因果重要性。最大train事件36,549个原生节点、14,599个local blocks、962个regional tokens全部参与，另有64个global latents；池化仍是压缩，不称无损。

FASERCal式partial Rel语义学习的前期诊断见[stage119](../workbook/2026-11-13_stage119_s4_full_rel_event_information.md)，与本次分类比较分开。旧矩阵及其解释只保留于标明状态的阶段记录，不复用为新路由的收益证据。Calibration/test继续冻结。

## 6. 由实测结果驱动的后续优化

2026-11-14执行合同已固定，见[阶段146](../workbook/2026-11-14_stage146_s4_train_inner_optimization_contract.md)和[48项train-inner配置](configs/s4_low_label_inner.json)。四种配方覆盖更早checkpoint、较低encoder LR、正则和LP-FT，对Scratch/MAE3及1k/10k各做三折；仅以train-inner OOF NLL共同选择配方和完整遍历次数，再固定末轮重拟合三seeds。该轮实际训练与新validation收益尚未测量。

新18项矩阵确认低标签排序收益，也确认1k过拟合、τ硬决策及低背景工作点仍不足。以下属于尚未执行的下一轮候选。先在原标签预算内固定train-inner选择合同，再对Scratch/MAE3共同优化；本轮不扩至1M。

| 实测问题 | 待比较的改动 | 控制与选择 |
| --- | --- | --- |
| 新1k全部最佳NLL均为第一轮，随后过拟合，MAE3平均NLL仍高于ln4 | 降低重复曝光次数、增加早期checkpoint时间点；分别降低encoder学习率与加强readout正则 | 同设置Scratch/MAE，不只调MAE；记录每个唯一事件的曝光和实际updates |
| 新MAE3的global路径实际影响输出，但低标签概率与决策仍不理想 | 先固定encoder拟合readout，再解冻的LP-FT；与单纯降低encoder学习率分别比较 | readout适配须覆盖相同summary/branch normalization定义；新路由的frozen readability尚未重测 |
| AP与τ argmax F1方向不同 | train-inner选择正则与决策规则，分别检查NLL、Brier、各类confusion和低FPR计数 | 不在现有validation上拟合class bias或温度；S6最终校准继续冻结 |
| 本次只重测MAE3，未确认新路由下其他预训练目标的分类净收益 | 只有主要readout/训练问题解决后，才在同一路由复核EMA、partial Rel、continuation和frozen probes | 模拟语义监督和计算成本单列，保留Scratch及MAE continuation对照 |

下一轮的超参数选择只用原nested-label子集内的固定分层train-inner folds，选择过程使用的标签仍计入同一1k/10k预算；按最终选择重新拟合全部该预算事件。验证集仅作冻结协议的后续评价，不能把多次查看validation描述成新的独立证据。追加方案要记录已观察S4结果后设计，沿用同一validation的区间也条件于这一步设计，不使用calibration/test。

2026-11-13 23:18 UTC重新联网读取[Kumar等，ICLR 2022 LP-FT](https://arxiv.org/abs/2202.10054)及[Wortsman等，CVPR 2022 WiSE-FT](https://openaccess.thecvf.com/content/CVPR2022/html/Wortsman_Robust_Fine-Tuning_of_Zero-Shot_Models_CVPR_2022_paper.html)，源文件与URL记录位于输出根`literature/low_label_transfer_sources.json`。LP-FT给出随机Head与encoder同时更新可能扰动预训练特征的可检验假设；其论文主要研究OOD，当前数据尚未证明这种病因。WiSE-FT依赖可用的zero-shot分类模型，当前没有对应分类Head，暂不直接采用权重插值。前述CVPR 2025 Sonata/NOMAE和FASERCal仍为主要预训练依据。
