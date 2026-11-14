# 文献核对与对 PINPOINT 的采用决定

检索日期：2026-11-14。使用期刊/会议正式页面、arXiv、INSPIRE 文献索引与作者代码；网页/PDF 提取与查询记录位于 `/eos/home-x/xcheng/FASER/Pinpoint/results/roadmap_review_2026-11-14/`。[初次来源记录](source_review.json)保存原27个外部论文/软件链接的题名与访问结果，全部HTTP200（PDG复用已提取PDF）。[追加核对](source_update_stage244.json)记录核心论文再次访问、tracking文献检索及Exa.TrkX补充。本文把资料与实验决定连接起来，论文benchmark数字只说明原任务。

本轮方法重点：FASERCal Methods/Supplement；NuGraph2 context-aware 的输入、辅助头与结果；Object Condensation 编码/损失；OGM 的分支梯度机制；pAUC 的目标、采样与 mini-batch 误差。其余条目核对摘要/官方页面，用于设计筛选。正式采用代码时固定 revision 并补读相应实现。

## 1. 本地资料与 FASERCal

<a id="f1"></a>**F1. S. Alonso-Monsalve et al., Towards foundation-style models for energy-frontier heterogeneous neutrino detectors via self-supervised pretraining.** *Nature Machine Intelligence* 8, 1581–1595 (2026)，online 2026-09-30；[正式论文](https://www.nature.com/articles/s42256-026-01309-6)，DOI `10.1038/s42256-026-01309-6`；[arXiv:2604.07037v2](https://arxiv.org/html/2604.07037v2)，[本地源码](../../docs/FASERCal/arXiv-2604.07037v2.tar.gz)。本次重新核对论文在线全文、Methods 和本地 main.tex。

它使用 sparse 3D convolution patching、module attention 和 Perceiver 融合，Rel 是局部 voxel 的 ghost/hierarchy/particle-type 监督；下游六类、charm、visible/jet momentum、primary vertex 联合训练。pooled sample 为 1,118,058 nominal +108,317 τ-enriched，评价恢复 nominal τ abundance。下游 31,936,341 参数，MAE 400 epochs、Rel continuation 100 epochs；完整预训练约 1,011 GH200 GPU-hours，fine-tuning 约 4.1 H100 GPU-hours。

**采用决定：**保留异构探测器结构和局部监督思想，优先补齐本项目 local→object→event 的关系。当前 100k 语料、0.898M 参数、3-epoch MAE 与它不是同规模复现；现有小收益不能代表该方法的上限。反过来，论文的 3DCal 体素 charge、AHCAL、ECAL 和多 hit-plane spectrometer 信息也不能直接变成 PINPOINT 的可用输入。扩大 SSL 前先建立强监督饱和参照。

**F2. ICHEP 2026, Multi-task masked autoencoder pretraining for TeV neutrino event reconstruction in FASERCal.** [本地 19 页报告](../../docs/FASERCal/ichep-sam-26-07-30.pdf)。用于对照流程与困难通道，精确训练数值以 F1 Methods 为准。

**F3. A. Mascellani, FASERCal, FASER Run 4 Workshop, 2025-11-14.** [本地 37 页报告](../../docs/FASERCal/FASERCal_wshopRun4.pdf)。早期 τhad selection 的分支效率、inclusive efficiency、flux/exposure 与 2026 主论文分别记载。支持先做 τhad specialist，不支持移用其 purity 或 significance 数值。

本项目三份合作组报告：[2026-09-16 pixel 增量](../../docs/FORTVNE_pixel_vs_scintonly_meeting_260916.pdf)、[2026-10-07 运动学](../../docs/FORTVNE_Update_kinematic_variables_2026-10-07.pdf)、[2026-10-21 CNN/ResNet](../../docs/FORTVNE_10212026.pdf)。本次 5 份 PDF 共 80 页均提取成功；当前对照条件和数值见[独立评估第3节](01_evidence_review.md#3-与合作组及-fasercal-的可比性)。

## 2. 点集、关系与计算规模

<a id="p1"></a>**P1. H. Qu, L. Gouskos, ParticleNet: Jet Tagging via Particle Clouds.** *Phys. Rev. D* 101, 056019 (2020)；[arXiv:1902.08570](https://arxiv.org/abs/1902.08570)，[DOI](https://doi.org/10.1103/PhysRevD.101.056019)。

采用点集与 EdgeConv 的局部关系思路，保留已有 detector adaptation 作为独立基线。PINPOINT 固定几何图与 dynamic graph 分开测；输入是 measured hit/cluster，不是已重建四动量粒子。

<a id="p2"></a>**P2. H. Qu, C. Li, S. Qian, Particle Transformer for Jet Tagging.** ICML 2022，PMLR 162:18281–18292；[正式页面](https://proceedings.mlr.press/v162/qu22b.html)，[代码](https://github.com/jet-universe/particle_transformer)。已核对官方摘要和本项目 ParT adaptation 的使用范围。

原论文用 pairwise interaction embedding 改进 attention，并展示大规模训练。这里补齐 tracker late fusion 和同输入对照；relation 使用 detector geometry/uncertainty，不使用 hit 的伪 invariant mass、truth momentum 或 jet ΔR。

<a id="p3"></a>**P3. X. Wu et al., Point Transformer V3: Simpler Faster Stronger.** CVPR 2024；[正式页面](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Point_Transformer_V3_Simpler_Faster_Stronger_CVPR_2024_paper.html)。

其关键启发是高效邻域与规模、扩大有效 receptive field。优先采用稀疏执行和跨窗口交换；serialization 必须尊重离散 plane、非均匀 z 和材料，不把自然三维场景的均匀 voxelization 直接套用。当前实验先用成熟邻域算子，profiler 证明瓶颈后再扩内核。

<a id="p4"></a>**P4. Y. Yue et al., LitePT: Lighter Yet Stronger Point Transformer.** CVPR 2026；[正式页面](https://openaccess.thecvf.com/content/CVPR2026/html/Yue_LitePT_Lighter_Yet_Stronger_Point_Transformer_CVPR_2026_paper.html)，[arXiv:2512.13689](https://arxiv.org/abs/2512.13689)。

高分辨率早期阶段使用便宜的局部编码，压缩后使用 attention，符合 PINPOINT 的资源条件。采用设计原则，不把其 benchmark 的速度/内存提升作为本项目收益；局部方向和 cross-plane propagation 要按 detector 特性设计。

## 3. 中微子重建与实例关系

<a id="h1"></a>**H1. NuGraph2: A Graph Neural Network for Neutrino Physics Event Reconstruction.** 正式题名 *Graph neural network for neutrino physics event reconstruction*，*Phys. Rev. D* 110, 032008 (2024)；[arXiv:2403.11872](https://arxiv.org/abs/2403.11872)，[DOI](https://doi.org/10.1103/PhysRevD.110.032008)。

它在各 readout plane 子图上做 heterogeneous message passing，用 3D context 约束多视图一致性。采用“直接从测量关系构图，再进行高层任务”的原则，适合 bar 一维区间、pixel plane 和 tracker 之间的关系；其 MicroBooNE hit-label efficiency 不等于 PINPOINT event flavour accuracy。

<a id="h2"></a>**H2. NuGraph2 with Context-Aware Inputs: Physics-Inspired Improvements in Semantic Segmentation.** *JHEP* 03 (2026) 149；[arXiv:2509.10684v2 全文](https://arxiv.org/html/2509.10684v2)，[DOI](https://doi.org/10.1007/JHEP03(2026)149)。已读 Methods 2.1–2.3、Discussion 与 Tables 1–2，发表元数据通过 INSPIRE/期刊页核对。

它比较几何/continuity 输入、额外 class-count decoders 和 energy regularization。其 Michel precision 0.60→0.64、recall 0.75→0.78 来自 context-aware features；额外 decoder 和 energy loss 没有超过 baseline。**采用决定：**优先增加隔离度、邻域几何、方向相容和粒子层级表示；event-level 辅助 loss 不能替代缺少的局部结构。这个原任务结果为研究顺序提供依据，不能证明 PINPOINT 同样必然获益。

<a id="h3"></a>**H3. J. Kieseler, Object condensation: one-stage grid-free multi-object reconstruction in physics detectors, graph and image data.** *Eur. Phys. J. C* 80, 886 (2020)；[arXiv:2002.03605](https://arxiv.org/abs/2002.03605)，[DOI](https://doi.org/10.1140/epjc/s10052-020-08461-2)。已抽读对象 assignment、condensation confidence、吸引/排斥与 payload loss。

适合比较 unknown-number object reconstruction 与 edge classification；需要明确定义 truth objects、混合 channel 处理和 noise。原方法每 vertex 指定一个主对象，不能自动解决 PINPOINT 多贡献 bar 的 soft assignment，先在较清楚的 pixel clusters/track segments 上验证。

<a id="h4"></a>**H4. S. R. Qasim et al., Learning representations of irregular particle-detector geometry with distance-weighted graph networks.** *Eur. Phys. J. C* 79, 608 (2019)；[arXiv:1902.07987](https://arxiv.org/abs/1902.07987)，[DOI](https://doi.org/10.1140/epjc/s10052-019-7113-9)。

GravNet/GarNet 为不规则 geometry 的 learned neighborhoods 提供可复用候选。项目已有 GravNet 代码可作接口参考；旧 node-label/数据管线不作为新实验默认输入，需在 accepted observations 和新 manifests 上对照。

相关开放软件：[SPINE, DeepLearnPhysics](https://github.com/DeepLearnPhysics/spine)，全称 *Scalable Particle Imaging with Neural Embeddings*。本次核对 README 的 sparse CNN 局部表示→GNN superstructure 链路，用于参考模块划分；它是 LArTPC 重建框架，不是可直接替换本项目的训练环境或性能证据。

<a id="h5"></a>**H5. Particle Data Group, τ listing, 2025 update.** [PDG PDF](https://pdg.lbl.gov/2025/listings/rpp2025-list-tau.pdf)。质量约 1.777 GeV、寿命约 290 fs，对应 cτ≈87 μm。用于估算 τ decay length 与探测器采样尺度；逐事件可分辨性仍需实际 geometry/response。

<a id="h6"></a>**H6. X. Ju et al. (Exa.TrkX), Performance of a geometric deep learning pipeline for HL-LHC particle tracking.** *Eur. Phys. J. C* 81, 876 (2021)；[arXiv:2103.06995](https://arxiv.org/abs/2103.06995)，[DOI](https://doi.org/10.1140/epjc/s10052-021-09675-8)。追加核对arXiv摘要、INSPIRE期刊元数据及摘要。

其候选构造、筛选和GNN关联分开评价的流程，为R4提供比mutual-nearest-only更合适的参考。PINPOINT首先测物理gate的candidate recall与边数尾部，再用learned edge filtering/GNN，最后给完整track efficiency/fake/duplicate/coverage；不以edge accuracy替代轨迹质量。原TrackML性能和近线性规模结论不直接外推到PINPOINT shower，也不能照用围绕束流碰撞点的origin/方向约束。实现时需补读其图构造和评估定义。

## 4. 优化目标、模态竞争与统计

<a id="m1"></a>**M1. X. Peng et al., Balanced Multimodal Learning via On-the-Fly Gradient Modulation.** CVPR 2022；[正式页面](https://openaccess.thecvf.com/content/CVPR2022/html/Peng_Balanced_Multimodal_Learning_via_On-the-Fly_Gradient_Modulation_CVPR_2022_paper.html)。已抽读 §3 的 modality contribution、discrepancy ratio 和 gradient modulation。

它说明多模态总体改善时，弱模态仍可能欠优化。先做独立 branch probe 和重训 late-fusion 控制，再决定是否采用 modulation；共享 encoder 的 local/global 路径不能机械当作独立 audio/visual 模态，梯度作用范围必须另定义。

<a id="m2"></a>**M2. D. Zhu et al., When AUC meets DRO: Optimizing Partial AUC for Deep Learning with Non-Convex Convergence Guarantee.** ICML 2022，PMLR 162:27548–27573；[正式页面](https://proceedings.mlr.press/v162/zhu22g.html)，[PDF](https://proceedings.mlr.press/v162/zhu22g/zhu22g.pdf)。已抽读 §3–5 的 OPAUC/TPAUC、CVaR/KL-DRO 和 stochastic algorithms。

采用低 FPR 区域的目标设计作为 τ specialist 后置候选。论文明确讨论 naive mini-batch pAUC 的误差问题；当前小 event batch 不能直接提供 0.1% 尾部的稳定估计，需有训练集 score bank/可靠 stochastic estimator。pAUC 优化不等于概率校准，最终仍使用独立 Cal-A/Cut-B。

<a id="m3"></a>**M3. S. Sagawa et al., Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization.** ICLR 2020；[arXiv:1911.08731](https://arxiv.org/abs/1911.08731)。本次核对摘要与方法用途。

只有当物理/响应域表现差异稳定、分组统计足够时，才尝试 group-balanced sampling/GroupDRO。先保留 nominal objective，避免牺牲总体任务去拟合极小最差组；并且不把 ν flavour 的真实能谱差异一律当应消除的 nuisance。

<a id="m4"></a>**M4. C. Guo et al., On Calibration of Modern Neural Networks.** ICML 2017；[正式页面](https://proceedings.mlr.press/v70/guo17a.html)。保留 scalar temperature 的低复杂度基线；新 specialist/多头概率先统一归一化，再在独立 Cal-A 拟合。良好 ECE 不代表有足够 separation，当前 S6 的低 ECE 与低 τ 效率可以同时成立。

<a id="m5"></a>**M5. Z. C. Lipton et al., Detecting and Correcting for Label Shift with Black Box Predictors.** ICML 2018；[正式页面](https://proceedings.mlr.press/v80/lipton18a.html)。区分 label prior shift 与 detector/generator conditional shift；prior 情景不是实际 flux 测量，也不能代替系统效应评价。

## 5. 预训练的候选与优先级

**S1. Multi-Scale Neighborhood Occupancy Masked Autoencoder for Self-Supervised Learning in LiDAR Point Clouds (NOMAE).** CVPR 2025；[正式页面](https://openaccess.thecvf.com/content/CVPR2025/html/Abdelsamad_Multi-Scale_Neighborhood_Occupancy_Masked_Autoencoder_for_Self-Supervised_Learning_in_LiDAR_CVPR_2025_paper.html)。保留多尺度正/负 occupancy query 的思想；孤立低 occupancy 结构可能是 PINPOINT 信号，不能照搬忽略 isolated points 的规则。

**S2. X. Wu et al., Sonata: Self-Supervised Learning of Reliable Point Representations.** CVPR 2025；[正式页面](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_Sonata_Self-Supervised_Learning_of_Reliable_Point_Representations_CVPR_2025_paper.html)。保留防 geometric shortcut 与 frozen probe 的诊断原则。项目 EMA3 已测配方收益较弱，下一轮不默认重做所有 self-distillation 目标。

| Priority | Adopt / compare | Evidence needed in PINPOINT |
| --- | --- | --- |
| Immediate | P1/P2 同输入、三类/四类、监督规模 | 实际重训、收敛曲线、共同非 τ 与 τ 工作点 |
| High | H1/H2 的 measured relation context；P3/P4 局部/全局分工 | 关系单因素 ablation、native/coarse、训练健康与成本 |
| Conditional | H3/H4 对象/实例；M1 模态优化 | 对象质量或弱分支可读但未被使用的证据 |
| Conditional | M2 低 FPR specialist；M3 最差域训练 | 足够背景尾部统计、整体质量约束、域计数 |
| After strong supervision | F1/S1/S2 的定向预训练 | 胜过同总计算监督，或有实测数据效率/跨域优势 |

先做能回答明确物理/训练问题的最小比较。现阶段不以更换 Transformer 名称、增加所有辅助头、直接导入不匹配的 FASERCal 权重或盲目扩大类别数作为主线。
