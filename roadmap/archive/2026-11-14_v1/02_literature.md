# 文献依据与方法迁移

检索核对日期：2026-11-11。优先级依据任务匹配程度、输入可获得性和计算成本；论文在自然场景/jet tagging 上的排序不等于在 PINPOINT 上的排序。

## 1. 本地资料的实际可读性

| Local resource | Inspection result | Use in this roadmap |
| --- | --- | --- |
| `docs/FASERCal/arXiv-2604.07037v2.tar.gz` | 有效 gzip/tar，包含 `main.tex`、`main.bib` 与论文图；已读取 Methods、Results、Supplementary training details | FASERCal 结构、loss、样本、指标和成本的主要可复核来源 |
| `docs/FASERCal/ichep-sam-26-07-30.pdf` | 有效 PDF，19 页；已提取全文 | 对照 2026-07-30 ICHEP 汇报的流程和困难通道 |
| `docs/FASERCal/FASERCal_wshopRun4.pdf` | 更新为有效 PDF，37 页、8,122,725 bytes；全文及关键图表已核对 | 2025-11-14 的完整重建链路、τhad 初步工作点与运动学选择 |
| `docs/FORTVNE_10212026.pdf` | 更新为有效 PDF，7 页、564,085 bytes；全文及选择矩阵已核对 | 1 mm pixel 投影、CNN/ResNet、简化 FASER 特征的条件与限制 |
| `docs/FORTVNE_Update_kinematic_variables_2026-10-07.pdf` | 更新为有效 PDF，9 页、3,546,193 bytes；文字及 raster 运动学图已核对 | 分量级信息、azimuth 泛化、quality selection 与覆盖率 |
| `docs/FORTVNE_pixel_vs_scintonly_meeting_260916.pdf` | 更新为有效 PDF，8 页、618,051 bytes；全文及曲线/counts 已核对 | matched validation 中 pixel 增量的 fiducial 依赖 |

四份更新文件共 61 页；它们的数值、页码、未明确的配置及路线图变更见 [本地汇报核对](05_local_studies.md)。旧研究结果用于确定对照问题；旧程序仍只作为 schema/图形接口参考，没有把其网络结构、truth selection 或 node-label 管线设为本研究前提。上述汇报的指标属于其各自样本，本批数据上的性能尚未重测。

## 2. FASERCal：应重点借鉴的完整链路

<a id="f1"></a>**F1.** S. Alonso-Monsalve et al., *Towards foundation-style models for energy-frontier heterogeneous neutrino detectors via self-supervised pretraining*, **Nature Machine Intelligence 8, 1581–1595 (2026)**，online 2026-09-30。[正式论文](https://doi.org/10.1038/s42256-026-01309-6)。本次在线核对期刊名称、DOI、发表日期及正文；正式版本提供代码和 held-out test 数据链接。

**F2.** 同工作 [arXiv:2604.07037v2](https://arxiv.org/abs/2604.07037v2)，v2 日期 2026-08-12。[本地源码包](../../../docs/FASERCal/arXiv-2604.07037v2.tar.gz)。下面精确数值引用这个本地 v2，避免将早期汇报、arXiv 和正式版的不同表述混为同一测量。

**F3.** S. Alonso-Monsalve et al., *Multi-task masked autoencoder pretraining for TeV neutrino event reconstruction in FASERCal*, ICHEP，2026-07-30。[本地汇报](../../../docs/FASERCal/ichep-sam-26-07-30.pdf)。重点页码为 3（异构探测器）、5–7（MAE、Rel、fine-tuning）、8–9（分类/回归）。

<a id="f4"></a>**F4.** A. Mascellani, *FASERCal*, FASER Run 4 Workshop，**2025-11-14**。[本地汇报](../../../docs/FASERCal/FASERCal_wshopRun4.pdf)。第 12、15–18 页说明 Kalman track fit、三视图 voxel reconstruction、masked/relational pretraining 与多任务 fine-tuning；第 19–25、30–32 页给优化工作点及不同 selection 下的运动学图。这是早期概念设计，样本为约 1.5M unbiased + 68k τ-only，与下述 2026 v2 分开使用。[详细核对](05_local_studies.md)

### 2.1 结构与训练

FASERCal 将 `48×48×200`、约 1 cm 的 3DCal voxels，coarser `18×18×40` AHCAL、`5×5` ECAL 能量矩阵、muon spectrometer 的测量/轨迹信息分别编码。稀疏卷积只处理 active voxels，3DCal 的 `12×12×10` patch 最多产生 320 tokens；AHCAL 的 `6×6×5` patch 最多产生 72 tokens。先做 module 内 attention，再由带 detector-type embedding 的 Perceiver IO bottleneck 融合各模态。主模型 embedding 384、12 heads，下游约 31.94M trainable parameters。

预训练分两步。先以 occupied patches 为基础 mask 75%，用轻量 decoder 预测缺失 occupancy 与 charge，相关辅助探测器也随机遮蔽；MAE 阶段 400 epochs。再训练 100 epochs，按每 batch 0.5 的概率增加 lower-mask（25%）的 relational pass：ghost（二类）、hierarchy（三类）和 EM/muon/hadron（三类），混合像素使用能量贡献归一化的 soft labels。后者包含模拟 truth 监督，不属于完全无标签训练。

fine-tuning 丢弃 decoder，共享 encoder 同时预测 flavour 六类、charm 四类、visible/jet momentum 与 primary vertex。六类为 `νe CC, νμ CC, ντ→e, ντ→μ, ντ→had, NC`；`pℓ` 由 `pvis−pjet` 构造。采用 log magnitudes、方向参数化、uncertainty-weighted multi-task losses。公开代码 [saulam/faserDL](https://github.com/saulam/faserDL) 与其 README 已核对，落地时须固定与论文/weights 一致的 release，再检查真实 tensor 与 target；README 本身不是输入真实性的依据。

### 2.2 论文数据与量化证据

名义样本 1,118,058 events，对应模拟 101 ab⁻¹；额外 τ enrichment 为 108,317 CC ντ events，对应 3,700 ab⁻¹。pool 固定 `85/5/10` 划分，评价时恢复 nominal τ abundance。下面是 v2 的 **one-vs-rest AUROC**，不是 PINPOINT 预期值：

| FASERCal class | Scratch | MAE | MAE+Rel |
| --- | ---: | ---: | ---: |
| νe CC | 0.968 | 0.983 | 0.985 |
| νμ CC | 0.909 | 0.955 | 0.958 |
| NC | 0.885 | 0.943 | 0.947 |
| ντ→had | 0.902 | 0.941 | 0.944 |
| ντ→e | 0.892 | 0.919 | 0.921 |
| ντ→μ | 0.801 | 0.831 | 0.835 |

困难的 leptonic τ 仍与 prompt `e/μ CC` 混淆；不能用 `νe` 的高 AUC 代替 τ 可识别性结论。`τ→e` 最大 `S/√(S+B)` 从 0.38→1.03→1.35，但它依赖该论文 flux、luminosity、选择与样本组成，不移植为 PINPOINT 的 significance 目标。

全量预训练成本为 584 GPU-hours（MAE）+427 GPU-hours（Rel），合计 **1,011 GH200 GPU-hours**；joint fine-tuning 为 **4.1 H100 GPU-hours**。Rel continuation 的新增成本约为 MAE 的 73%，而许多 aggregate AUC 的额外收益仅几千分之一。PINPOINT 应预注册“无 Rel / 有 Rel / 同算力继续 MAE / 同算力继续监督训练”的对照，优先根据 τ 工作点和资源收益决策。

在 matched GENIE/NuWro 上，MAE+Rel flavour macro-AUC 为 0.932/0.928，而 charm 为 0.864/0.721。论文证据支持在该匹配域内 flavour 较稳定，同时明确展示 heavy-flavour 标签的 generator 敏感性；不能解释为“预训练消除了系统效应”。标签效率研究固定验证/测试，100–100k labels、每预算 3 seeds，约 1k labels 的预训练模型达到约 10k labels scratch 的 flavour 水平。

### 2.3 对 PINPOINT 的具体采用与改造

| FASERCal ingredient | PINPOINT decision | Reason and validation |
| --- | --- | --- |
| Sparse local embedding | 采用原则，按 pixel plane 与 bar/panel 分支设计 | 20 μm pixel、bar 一维测量与 1 cm voxel 信息不同；原生位置残差保留 |
| Module-aware attention | 优先采用并消融 | 非均匀 z、材料与接受度是已知结构；与无 module embedding 对照 |
| Perceiver fusion | 首选融合候选，与简单 late fusion 对照 | 防止上千 bar/pixel tokens 的全局 N²；保留小模态信息 |
| MAE reconstruction | 采用 occupancy + 可测信号版本 | binary pixel 不要求重建不存在的 analog charge；重建物理局部邻域而非整个空体积 |
| Soft relational labels | 有 contribution truth 时采用 | 现有 pixel truth flags 只记首次贡献，不能还原能量分数；ghost 也需先有显式重建歧义 |
| Six flavour channels | 作为有监督子类/辅助方案，汇总回四类 | `p(τ)=p(τe)+p(τμ)+p(τhad)`；同时比较直接四类头 |
| Multi-task momentum/vertex | 分步加入，不一次性全部开启 | 标签/可见性/mask 定义先完成，检查负迁移与不可辨识量 |
| Pretrained weights | 方法迁移优先，weights 迁移单列探索 | 输入尺度、geometry、电子学不同；自然场景/3DCal weights 未必合适 |

现有模拟库包含 7,240 个 ROOT 文件，event flavour truth 随文件提供，总事件数尚未测量。因此“少标签”本身不充分证明大规模预训练必要。应同时证明全监督数据充足时仍有收益，或对新 geometry、标定域、少量精细 truth 和未来无标签实测数据具有价值。

## 3. ParticleNet 与 Particle Transformer

<a id="p1"></a>**P1.** H. Qu, L. Gouskos, *ParticleNet: Jet Tagging via Particle Clouds*, **Phys. Rev. D 101, 056019 (2020)**。[论文](https://doi.org/10.1103/PhysRevD.101.056019)、[arXiv:1902.08570](https://arxiv.org/abs/1902.08570)。采用 unordered particle set、EdgeConv 和 dynamic neighborhoods。PINPOINT 使用同层局部几何与跨层相容邻接，测试 fixed/dynamic graph，禁止把 PDG 和 truth momentum 当点特征。

<a id="p2"></a>**P2.** H. Qu, C. Li, S. Qian, *Particle Transformer for Jet Tagging*, **ICML 2022, PMLR 162:18281–18292**。[会议论文](https://proceedings.mlr.press/v162/qu22b.html)、[代码](https://github.com/jet-universe/particle_transformer)。pairwise particle-interaction embedding 加到 attention 是核心可迁移部分；原任务以 100M jets 支持规模化训练。PINPOINT 的 hit 并非已重建 particle，不能给 hit 构造伪四矢量或使用 invariant mass、jet `ΔR/kT` 作为未测输入；采用测量空间 relation 的 ParT adaptation，并在表格中明确其改造版本。

## 4. 近年计算机视觉/机器学习方法筛选

本次在线查看 CVPR 2024/2025、ICCV 2025 的公开论文列表，CVPR 2026 三个 conference day 的列表，以 `point cloud / point transformer / masked / self-supervised / multimodal / sparse attention` 筛选；并核对 NeurIPS 2025 Concerto 的项目页与 arXiv 会议元数据。以下采用会议正式页面、作者项目页和论文摘要/方法描述，不把搜索摘要当作发表依据。

<a id="c1"></a>**C1. Perceiver IO — ICLR 2022.** A. Jaegle et al., *Perceiver IO: A General Architecture for Structured Inputs & Outputs*。[论文](https://arxiv.org/abs/2107.14795)、[会议页面](https://openreview.net/forum?id=fILj7WpI-g)。固定 latent 对 variable-size inputs 做 cross-attention，输入成本随 `N×L` 增长。用于多探测器融合；验证小模态在 pixel/bar 高占用时是否被稀释。

<a id="c2"></a>**C2. PTv3 — CVPR 2024.** X. Wu et al., *Point Transformer V3: Simpler Faster Stronger*。[正式论文](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Point_Transformer_V3_Simpler_Faster_Stronger_CVPR_2024_paper.html)。以 serialization 替代精确 KNN，扩大邻域与处理规模。作为高占用候选；检查各 plane 之间的序列边界，保留 physical residual，不能把均匀体素下采样直接套到薄 silicon planes。

<a id="c3"></a>**C3. Flash3D — CVPR 2025.** L. Chen et al., *Flash3D: Super-scaling Point Transformers through Joint Hardware-Geometry Locality*。[正式论文](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Flash3D_Super-scaling_Point_Transformers_through_Joint_Hardware-Geometry_Locality_CVPR_2025_paper.html)。Perfect Spatial Hashing 对齐局部几何与 GPU tiling/FlashAttention。论文相对 PTv3 报告 2.25× speed、2.4× memory efficiency；这些是其 benchmark 结果。仅在 PINPOINT profiler 证明邻居搜索/内存为瓶颈后尝试自定义 kernels，先用成熟算子建立物理基线。

<a id="c4"></a>**C4. LitePT — CVPR 2026.** Y. Yue et al., *LitePT: Lighter Yet Stronger Point Transformer*。[正式论文](https://openaccess.thecvf.com/content/CVPR2026/html/Yue_LitePT_Lighter_Yet_Stronger_Point_Transformer_CVPR_2026_paper.html)、[arXiv](https://arxiv.org/abs/2512.13689)。高分辨率 early stages 使用 convolution，deep/low-resolution stages 使用 attention，PointROPE 补充位置布局。其报告相对 PTv3 3.6× fewer parameters、约 2× speed/memory improvement。**主线优先参考**局部/全局分工；PINPOINT 用同层 sparse 2D convolution 或 local set/edge encoder，跨层用实际 z/材料关系；PointROPE 仅作位置编码消融，不预设自然场景 isotropic invariance。

<a id="c5"></a>**C5. Sonata — CVPR 2025.** X. Wu et al., *Sonata: Self-Supervised Learning of Reliable Point Representations*。[正式论文](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_Sonata_Self-Supervised_Learning_of_Reliable_Point_Representations_CVPR_2025_paper.html)。指出 sparse point SSL 的 geometric shortcut，通过遮蔽空间信息与 self-distillation 学习可线性探测表示。用于与 MAE 对照：student 只看到部分通道/模态，teacher 看完整可观测读出；用 frozen linear probe 与相同 fine-tuning 比较。PINPOINT 不能通过旋转、删短 track 强行制造错误物理不变性。

<a id="c6"></a>**C6. NOMAE — CVPR 2025.** M. Abdelsamad et al., *Multi-Scale Neighborhood Occupancy Masked Autoencoder for Self-Supervised Learning in LiDAR Point Clouds*。[正式论文](https://openaccess.thecvf.com/content/CVPR2025/html/Abdelsamad_Multi-Scale_Neighborhood_Occupancy_Masked_Autoencoder_for_Self-Supervised_Learning_in_LiDAR_CVPR_2025_paper.html)。在未遮蔽 voxel 邻域预测 occupancy，多尺度 mask，控制 sparse 空域代价与 occupancy leakage。**主线采用其问题定义**：decoder 在固定几何邻域同时看到正/负候选，不能只给 masked active coordinate 后宣称学会发现 occupancy。Methods 3.2 明确忽略远离 visible voxels 的 isolated points；PINPOINT 的孤立短轨可能正是重要信号，需要加上由已知几何抽取的跨层/远区候选和稀疏轨迹诊断，不能直接复制这个忽略规则。

<a id="c7"></a>**C7. Concerto — NeurIPS 2025，arXiv v2 更新 2026-02-28.** Y. Zhang et al., *Concerto: Joint 2D-3D Self-Supervised Learning Emerges Spatial Representations*。[论文](https://arxiv.org/abs/2510.23607)、[作者项目页](https://pointcept.github.io/Concerto/)。结合 3D 内部 self-distillation 和 2D–3D joint embedding，依赖已知对应关系。可迁移为 pixel/bar/tracker 的互补预测，但它们不是 RGB–depth 同视图，须用几何投影和 measurement uncertainty，不能把同一 detector hits 的二维投影误当新增独立硬件信息。

<a id="c8"></a>**C8. Point-PQAE — ICCV 2025.** X. Zhang, S. Zhang, J. Yan, *Towards More Diverse and Challenging Pre-training for Point Cloud Learning: Self-Supervised Cross Reconstruction with Decoupled Views*。[正式论文](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Towards_More_Diverse_and_Challenging_Pre-training_for_Point_Cloud_Learning_ICCV_2025_paper.html)。以 two decoupled views 和 relative position 做 cross-reconstruction；作为跨 module/模态的预训练候选。裁剪可能删除 τ 特征，故优先遮蔽连续 plane/window 而不改变完整事件的分类标签；decoder 重建目标保留完整原事件。

<a id="c9"></a>**C9. PointINS — CVPR 2026.** B. Yang et al., *Towards Foundation Models for 3D Scene Understanding: Instance-Aware Self-Supervised Learning for Point Clouds*。[正式论文](https://openaccess.thecvf.com/content/CVPR2026/html/Yang_Towards_Foundation_Models_for_3D_Scene_Understanding_Instance-Aware_Self-Supervised_Learning_CVPR_2026_paper.html)。引入 offset branch、Offset Distribution Regularization 和 Spatial Clustering Regularization，用 pseudo-instance masks 培养 instance awareness。适合探索 track/shower offset 或 secondary-vertex proposals；密集级联中的“物理粒子”并非自然场景封闭物体，先验证可见轨迹段/簇的定义与错聚率，再启用，不把全部 hit 强行聚为单一粒子。

<a id="c10"></a>**C10. PointCSP — CVPR 2026.** X. Yu et al., *PointCSP: Cross-Sample Semantic Propagation and Stability Preservation in Self-Supervised Point Cloud Learning*。[正式论文](https://openaccess.thecvf.com/content/CVPR2026/html/Yu_PointCSP_Cross-Sample_Semantic_Propagation_and_Stability_Preservation_in_Self-Supervised_Point_CVPR_2026_paper.html)。batch 中序列化多场景，用 state-space 跨样本传递语义，再用 asymmetric semantic-preservation distillation 适配单场景推理。**后置探索**：先验证单事件推理与 batch 顺序/类别组成的独立性，状态不得跨 train/test 或真实事件保留；现阶段复杂度高于预期收益，主线先用事件独立 encoder。

**C11. GGPT — CVPR 2026.** Y. Chen et al., *GGPT: Geometry-Grounded Point Transformer*。[正式论文](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_GGPT_Geometry-Grounded_Point_Transformer_CVPR_2026_paper.html)。核查正文摘要后确认其目标是 RGB sparse-view reconstruction、SfM guidance 和 dense point maps；尽管名称包含 point transformer，任务与当前 detector readout 分类不符，本路线不采用其整体架构。

## 5. 筛选后的优先级与引用边界

| Priority | Methods | Concrete experiment |
| --- | --- | --- |
| P0 | ParticleNet、测量几何版 ParT、简单 late fusion | 原生/粗表示；确定信息收益和可行计算范围 |
| P1 | FASERCal hierarchy/Perceiver + LitePT 原则 | 同层局部编码、module context、小 latent 融合；和 P0 同预算比较 |
| P1 | FASERCal MAE、NOMAE、Sonata | occupancy/signal 重建 vs self-distillation；相同 encoder 与标签预算 |
| P2 | Rel、Concerto、Point-PQAE、PointINS | contribution truth/跨视图关系/对象定义通过质量检查后才扩展 |
| Conditional | PTv3、Flash3D、PointCSP | 对应 profiler、单事件推理与数据依赖门槛满足后执行 |

本次未把 diffusion 场景生成、LLM/VLM 文本接口、全旋转等变性作为主线：它们需要额外计算或不适用于有固定束流、磁场、分层材料和探测器视场的输入。生成模型若后续用于 augmentation，必须保持 generator 家族分组并验证没有虚构可分辨的 τ/vertex 结构。

除 FASERCal 全文外，本次还抽读了 LitePT 与 NOMAE 的 full-paper Methods；其余方法以会议摘要/作者材料支持筛选。PINPOINT 上的量化结论必须来自统一 split 和硬件输入的实验。正式复现实现前，S2–S4 应补读所选方法的 full Methods/代码并固定版本，尤其是 tokenization、mask leakage、label mixture 和 measured-vs-truth 的边界。此处并不声称覆盖 2026 年全部会议或存在已确定的最佳模型。
