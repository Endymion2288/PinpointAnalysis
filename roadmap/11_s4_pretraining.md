# S4：可验证的预训练与标签效率

状态：100k corpus、三个独立seeds的完整MAE3、Scratch三预算×三seeds、MAE3 seed42三预算及完整10k路径诊断完成；19个分类模型和2193次训练监测通过v7统一读回。MAE3三个独立seeds的1k分类配对已完成。独立MAE下游seeds、Rel/EMA与等计算监督控制执行中。训练输出根为 `/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13`。S3 revision 2 的 [最终报告](../workbook/2026-11-13_stage87_s3_repair_final_results.md) 是直接监督依据，旧 S3 故障模型不参与方法比较。

## 1. 研究问题与固定条件

1. S3 的 global 分支虽有非零梯度，tracker 模型中置换 global 输出的 probe NLL 增量为 −0.001597；预训练能否学到可线性读出的跨探测器表示，并在实际分类中得到利用？
2. tracker 对 νμ、NC 有增益，ντ AP 增量的区间跨零。预训练是否改善 ντ 及其 e/μ/had 子道的混淆，还是主要学习 shower occupancy 和能量先验？
3. 低标签收益能否覆盖新增资源成本？同样预算继续监督或 MAE 是否已能取得相同结果？

沿用已接受的 detector-proxy、geometry 和 family split。既有 80k train、10k validation 原样保留，另从八个新 chunk/类选择 5k train/类，组成 100k train。所有 SSL 输入、重建验证和 probe-fit 均来自 train；独立 validation 只用于下游选择/评价。Calibration/test 不打开。

下游共同 backbone 为 S3 revision 2 multiscale native + tracker：Peri-LN、QK-LN、LayerScale、FP32 normalized SiLU Head、全部 local blocks 直接读出。导出 features 的接口保持默认分类数值及 checkpoint 参数命名不变。预训练 decoder/teacher 在 fine-tuning 时丢弃；分类 heads 对所有初始化策略使用相同种子重新初始化。架构额外变更须另加 Scratch 控制。

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

S4 本轮组织上限 **60 device-hours**，包括所有后端的 pilot、编译、失败、训练、probe 和评价；不同设备、MIG 与整卡分别列出，时长不换算成等价 FLOPs。先运行有限 pilot，再冻结正式实验配置；不自动耗尽上限。

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

配对比较冻结于 `configs/s4_pairs.json`；新增语义结果、τ 子道统计和监督预算规划见 [stage106](../workbook/2026-11-13_stage106_s4_paired_semantics_and_budget.md)。完整channel Rel的512-event同查询诊断已超过几何均值先验，见[stage119](../workbook/2026-11-13_stage119_s4_full_rel_event_information.md)及`results/s4_rel_semantics.json`；该结果与分类收益分别评价。

完整路径及v6结果见[stage129](../workbook/2026-11-13_stage129_s4_global_path_and_100k_validation.md)。MAE3 seed42在1k/10k下global置换的NLL增量为0.051899/0.035174，95%区间均为正；local/regional仍有可测作用。100k的macro AP配对增量0.003371，区间跨零；common-support tau AP增量也未确认，不能把低标签收益或global利用直接推广为kinematics-independent tau改进。全部比较仍为固定validation，calibration/test冻结，矩阵未完成。

三seed低标签结果见[stage133](../workbook/2026-11-13_stage133_s4_three_seed_low_label_results.md)。1k时MAE3−Scratch的macro AP均值增量0.057021，物理块CI95=[0.052512,0.061955]；hadronic τ AP在既定common-support诊断中增加0.017168，[0.006759,0.026996]。同时raw τ argmax F1下降0.050129，平均NLL1.450039仍高于ln4，1%FPR效率区间跨零。该结果支持低标签排序收益，过拟合与硬决策问题保留为后续针对性优化依据；不据此直接扩1M。
