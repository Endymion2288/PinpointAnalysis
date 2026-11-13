# S4：可验证的预训练与标签效率

状态：实现与小样本验证中。训练输出根为 `/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13`。S3 revision 2 的 [最终报告](../workbook/2026-11-13_stage87_s3_repair_final_results.md) 是直接监督依据，旧 S3 故障模型不参与方法比较。

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

Decoder queries 仅由几何和 visible 邻域产生，混合邻近、跨层与 uniform 候选；候选集合先生成，再查询 reconstruction targets。必须报告每种候选的正例比例、loss、PR-AUC、reconstruction coverage，声明优化的是所定义混合查询分布，不能称为均匀全探测器 risk。binary pixel 的 coarse occupancy/count 与 native subcell occupancy 分开，bar signal 只用接受的能量/PE proxy。负候选不能按 hidden support 筛除。直接依赖 visible geometry 的 trivial baseline 与 encoder-permutation baseline 同时报告。

`ScintSD.cc` 的 `barEnergyMap` 以 channel/track/PDG 为 key 累加；ROOT `Hits/scintHits` 保存相应 `pdg/edep`。额外 sidecar 按硬件 channel 累加 EM(11/22)、muon(13)、hadron、other 的能量分数，正能量有效通道才产生目标。质量检查仅针对新增标签的 channel join、能量一致和 simplex，不重复样本认证。Pixel 首贡献 flags、ghost/hierarchy 不冒充精确 soft truth；本轮不用 pixel Rel。模拟贡献只进 loss，不进入 input、mask 或 query 位置。

## 4. 分阶段实验与预算

S4 本轮组织上限 **60 device-hours**，包括所有后端的 pilot、编译、失败、训练、probe 和评价；不同设备、MIG 与整卡分别列出，时长不换算成等价 FLOPs。先运行有限 pilot，再冻结正式实验配置；不自动耗尽上限。

1. 完成 100k corpus、独立 Rel sidecars；train 内嵌套 98k pretrain-fit/2k reconstruction-probe。监督标签预算为嵌套的 1k/10k/100k，四类平衡，τ 子道按原样本比例分层。
2. 导出 S3 encoder 的 local/regional/global/summary features，在 train 内 fit/probe-validation 上分别拟合 linear/MLP，测可读信息与实际梯度的差别。不能用 64-event 置换取代分类 probe。
3. 真实 32-event 拟合、FP32/BF16、最大 occupancy forward/backward、无泄漏、负候选、EMA/variance、checkpoint/resume 通过后，正式 pilot 比较 mask 0.5/0.75。重建内部验证及分布/coverage 决定主 mask；全零预测不作通过结果。
4. 同 backbone 比较 Scratch / MAE / EMA self-distillation / MAE+barRel；另保留 MAE continuation 和继续监督的 matched-compute 控制。相同 fine-tuning 设置与独立初始化 heads。先 seed 42，关键候选与 Scratch 用 17/42/2026 确认。
5. 每个标签预算同时交付 frozen linear probe 和 full fine-tuning、learning curves、checkpoint 读回、内部健康和总成本表。完整 validation 为 10k，四分类指标、各类 1%/0.1% FPR 效率、τ 子道及 common-support 条件指标齐全。0.1% FPR 的低通过背景数按实测计数报告。
6. 只有重建健康且下游收益与资源证据支持时才扩至 1M；无收益也完成本轮对照与统计结论，不盲目扩量。1M 是否执行及原因写入最终验收。

## 5. 调度与验收

批量转换只用 HTCondor；选择 schedd 时保存即时 load、EOS policy、WRITE 结果。训练可由 HTCondor/Kubeflow 竞争领取互斥 task claim；使用平台规定的 GPU product/resource flavor，不在 GPU Pod 中自填 CPU/RAM。并发数以真实 CPU/data wait、显存和 GPU utilization 调整。正式长训练不得从 SSH 临时启动或暂停 PID 1 进行迁移。

验收除结果文件完整外，要求 informative encoder 梯度、非饱和 head、非恒定预测、SSL 的 feature 方差和实际拟合证据；若出现明显异常，先定位架构/尺度/采样/优化并修复重训。最终记录所有 allocations 终态和资源成本；阶段报告与代码共同提交。
