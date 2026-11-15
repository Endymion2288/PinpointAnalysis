# 文献核对与对 PINPOINT 的采用决定

检索日期：2026-11-14至15。使用期刊/会议正式页面、arXiv、INSPIRE 文献索引与作者代码；网页/PDF 提取与查询记录位于 `/eos/home-x/xcheng/FASER/Pinpoint/results/roadmap_review_2026-11-14/`。[初次来源记录](source_review.json)保存原27个外部论文/软件链接的题名与访问结果，全部HTTP200（PDG复用已提取PDF）。[追加核对](source_update_stage244.json)记录核心论文再次访问、tracking文献检索及Exa.TrkX补充；[stage278来源](source_update_stage278.json)补充OPERA/NOMAD的τ识别物理路径。本文把资料与实验决定连接起来，论文benchmark数字只说明原任务。

本轮方法重点：FASERCal Methods/Supplement；NuGraph2 context-aware 的输入、辅助头与结果；Object Condensation 编码/损失；OGM 的分支梯度机制；pAUC 的目标、采样与 mini-batch 误差。其余条目核对摘要/官方页面，用于设计筛选。正式采用代码时固定 revision 并补读相应实现。

最终整理时再次访问FASERCal与NuGraph2 context-aware官方全文、pAUC会议页和CERN训练文档，五个来源（另含INFERNO摘要）均HTTP200，访问日期2026-11-15。FASERCal的样本/训练规模与本地`main.tex`一致，NuGraph2的feature extension收益和额外decoder负结果再次核对。访问清单见[source_update_stage290.json](source_update_stage290.json)，网页和提取正文保留在`results/roadmap_review_2026-11-15/`。

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

<a id="p5"></a>**P5. R. Zhang, Making Convolutional Networks Shift-Invariant Again.** ICML 2019，PMLR 97:7324–7334；[正式页面](https://proceedings.mlr.press/v97/zhang19a.html)。2026-11-15追加核对官方摘要与元数据，HTTP200；访问记录见`R1/audit/sparse_projection_sources_stage257.json`。

论文指出stride/pooling的采样相位可能影响CNN输出，并研究降采样前的anti-aliasing。它为stage256稀疏μ损失提供一个可检验的程序/表示假设，不能证明PINPOINT损失已经由aliasing造成。先检查单点/短线与真实Train响应，再比较首层stride和滤波；滤波可能抹去孤立测量，不能直接照用ImageNet方案，也不要求绝对探测器位置具有平移不变性。正式实现前补读方法与算子顺序。

<a id="p6"></a>**P6. B. Graham, L. van der Maaten, Submanifold Sparse Convolutional Networks.** [arXiv:1706.01307](https://arxiv.org/abs/1706.01307)，2017。2026-11-15追加核对arXiv摘要及作者/题名，HTTP200；同上保存来源。

采用在稀疏支持上保留局部结构、减少dense空网格计算的原则。它适合plane内高分辨率残差候选；相隔的plane仍需显式几何边或其他可达路径，不能将submanifold convolution本身当成跨层传播。尚未在本项目实现或测得收益；复用现成兼容算子，先profile，再决定是否值得增加环境依赖。

<a id="p7"></a>**P7. R. Liu et al., An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution.** [arXiv:1807.03247](https://arxiv.org/abs/1807.03247)，2018。2026-11-15核对arXiv作者、题名和摘要，HTTP200；原始摘要及访问记录在`R3/audit/sensor_row_sources_stage292.json`。

论文通过额外坐标通道让卷积访问绝对位置。R3 C1借用这一具体机制，为紧凑sensor-row响应显式提供真实z、前后间距及传感器类型，保留固定探测器边界。它不要求把row ordinal视为物理距离，也不表示坐标通道单独已经改善分类。C1同时改变空行布局、stem输入及stride，整体收益由同响应、同100k×4的正式screen评价；论文的toy/视觉任务倍数不作为PINPOINT效果或速度证据。

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

<a id="h7"></a>**H7. OPERA Collaboration, Final Results of the OPERA Experiment on ντ Appearance in the CNGS Neutrino Beam.** *Phys. Rev. Lett.* 120, 211801 (2018)；[arXiv:1804.04912](https://arxiv.org/abs/1804.04912)，[DOI](https://doi.org/10.1103/PhysRevLett.120.211801)，关联[Erratum](https://doi.org/10.1103/PhysRevLett.121.139901)。2026-11-15通过INSPIRE核对发表记录，下载arXiv PDF并抽读PDF p2–3的Event selection和Analysis strategy；不引用其显著性数值。arXiv摘要/PDF和INSPIRE均HTTP200，PDF444,435bytes。

分析按1h/3h/e/μ衰变形态预选并训练各道BDT，使用拓扑、运动学与prompt lepton识别；背景包括charm、强子再相互作用和大角度μ散射。采用其分道和背景检查思路，PINPOINT的候选、标签覆盖与效率全部重新测量。OPERA的emulsion、能区、顶点采样与PINPOINT不同，不移用其kink cut、分辨率或选择效率。had1/had3细分先区分最终prongs与当前直接daughter计数。

<a id="h8"></a>**H8. NOMAD Collaboration, Final NOMAD results on νμ→ντ and νe→ντ oscillations including a new search for ντ appearance using hadronic τ decays.** *Nucl. Phys. B* 611, 3–39 (2001)；[arXiv:hep-ex/0106102](https://arxiv.org/abs/hep-ex/0106102)，[DOI](https://doi.org/10.1016/S0550-3213(01)00339-X)。2026-11-15通过INSPIRE核对发表记录，下载arXiv PDF并抽读PDF p4的Introduction、p8的likelihood构造及hadronic selection片段；摘要/PDF和INSPIRE均HTTP200，PDF751,604bytes。

论文明确说明空间分辨率不足以分开τ衰变点，依靠可见末态运动学、候选隔离及相关性完成识别。它支持为无可分辨kink的事件保留运动学分支，不能证明PINPOINT已经有相同的能量/动量测量能力。先检验tracker和局部对象的方向/动量resolution–coverage、可见系统与接受度，再与raw分类做配对；当前弱event-level方向回归不满足这一前提。两篇全文、摘录和访问记录保存在`R2/audit/literature_stage278/`。

<a id="m10"></a>**P5的R3追加读取记录（M10旧链接保留）。** 2026-11-15再次读取[arXiv:1904.11486](https://arxiv.org/abs/1904.11486)官方摘要，HTTP200、41415 bytes，记录在`R3/audit/literature_stage286/retrieval.json`；同一论文的正式引用见[P5](#p5)。

论文将降采样引入的相位敏感与表示学习分开讨论，支持R3的保测量padding相位诊断及首层stride单因素控制。PINPOINT的绝对几何与边界仍保留；图像任务的anti-alias收益不能外推为探测器收益。是否增加滤波须另立对照，避免平滑削弱孤立响应。

## 4. 优化目标、模态竞争与统计

<a id="m1"></a>**M1. X. Peng et al., Balanced Multimodal Learning via On-the-Fly Gradient Modulation.** CVPR 2022；[正式页面](https://openaccess.thecvf.com/content/CVPR2022/html/Peng_Balanced_Multimodal_Learning_via_On-the-Fly_Gradient_Modulation_CVPR_2022_paper.html)。已抽读 §3 的 modality contribution、discrepancy ratio 和 gradient modulation。

它说明多模态总体改善时，弱模态仍可能欠优化。先做独立 branch probe 和重训 late-fusion 控制，再决定是否采用 modulation；共享 encoder 的 local/global 路径不能机械当作独立 audio/visual 模态，梯度作用范围必须另定义。

<a id="m2"></a>**M2. D. Zhu et al., When AUC meets DRO: Optimizing Partial AUC for Deep Learning with Non-Convex Convergence Guarantee.** ICML 2022，PMLR 162:27548–27573；[正式页面](https://proceedings.mlr.press/v162/zhu22g.html)，[PDF](https://proceedings.mlr.press/v162/zhu22g/zhu22g.pdf)。已抽读 §3–5 的 OPAUC/TPAUC、CVaR/KL-DRO 和 stochastic algorithms。

采用低 FPR 区域的目标设计作为 τ specialist 后置候选。论文明确讨论 naive mini-batch pAUC 的误差问题；当前小 event batch 不能直接提供 0.1% 尾部的稳定估计，需有训练集 score bank/可靠 stochastic estimator。pAUC 优化不等于概率校准，最终仍使用独立 Cal-A/Cut-B。

<a id="m3"></a>**M3. S. Sagawa et al., Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization.** ICLR 2020；[arXiv:1911.08731](https://arxiv.org/abs/1911.08731)。本次核对摘要与方法用途。

只有当物理/响应域表现差异稳定、分组统计足够时，才尝试 group-balanced sampling/GroupDRO。先保留 nominal objective，避免牺牲总体任务去拟合极小最差组；并且不把 ν flavour 的真实能谱差异一律当应消除的 nuisance。

<a id="m4"></a>**M4. C. Guo et al., On Calibration of Modern Neural Networks.** ICML 2017；[正式页面](https://proceedings.mlr.press/v70/guo17a.html)。保留 scalar temperature 的低复杂度基线；新 specialist/多头概率先统一归一化，再在独立 Cal-A 拟合。良好 ECE 不代表有足够 separation，当前 S6 的低 ECE 与低 τ 效率可以同时成立。

<a id="m5"></a>**M5. Z. C. Lipton et al., Detecting and Correcting for Label Shift with Black Box Predictors.** ICML 2018；[正式页面](https://proceedings.mlr.press/v80/lipton18a.html)。区分 label prior shift 与 detector/generator conditional shift；prior 情景不是实际 flux 测量，也不能代替系统效应评价。

<a id="m6"></a>**M6. M. Li, E. Yumer, D. Ramanan, Budgeted Training: Rethinking Deep Neural Network Training Under Resource Constraints.** ICLR2020；[arXiv:1905.04753v4](https://arxiv.org/abs/1905.04753)，[PDF](https://arxiv.org/pdf/1905.04753)。2026-11-15读取摘要与§3.1–3.2，abstract/PDF均HTTP200，访问记录见`R1/audit/literature_stage264/retrieval.json`。

论文按迭代预算`t/T`定义budget-aware schedule，指出直接截短原schedule与按新预算重新参数化是不同实验，cosine也属于其budget-aware家族。用于R2明确D0/D1/D2的总updates、fresh初始化及LR轨迹；PINPOINT首批仍保留原warm-up/cosine，linear只作另行预定候选，不把其图像任务结果当成本项目性能结论。R1的wall-based控制同时改变曝光与schedule，后续需将这两个因素分别报告。

<a id="m7"></a>**M7. A. K. Menon et al., Long-tail Learning via Logit Adjustment.** ICLR2021；[arXiv:2007.07314v2](https://arxiv.org/abs/2007.07314)，[PDF](https://arxiv.org/pdf/2007.07314)。2026-11-15读取摘要、§3–4.1及式(7)–(9)，abstract/PDF均HTTP200，PDF1,402,421bytes；原文和访问记录位于`R2/audit/literature_stage276/`。

论文将balanced error的Bayes决策写成`argmax(logit_y−log(prior_y))`，并讨论概率校准前提。当前PINPOINT训练pool全局平衡，按全局label frequency作这个平移是共同常数，不能解释或修复低occupancy μ argmax损失。不能因存在mixed-window类比例波动就直接套用long-tail loss。若R2终点仍有排序较好而argmax差的证据，使用按physical family分组的Tune交叉拟合、低维logit-bias诊断，检验decision/readout部分可恢复的量；正scalar temperature另作argmax不变的对照。该诊断不用于宣称物理信息增加，也不替代R7独立Cal-A/Cut-B。

<a id="m8"></a>**M8. I. Loshchilov, F. Hutter, Decoupled Weight Decay Regularization.** ICLR2019；[arXiv:1711.05101](https://arxiv.org/abs/1711.05101)。2026-11-15在线读取摘要，HTTP200、43,412bytes，访问记录及HTML保存在`R2/audit/literature_stage280/`。文章明确区分Adam的L2 penalty和decoupled weight decay，并给出image classification泛化证据。当前已使用AdamW，若完整R2参考表现出Train改善而Tune退化，后续只改变其decoupled衰减强度，保持LR、数据和曝光固定；论文没有替PINPOINT确定最佳系数，单个对照也不构成完整HPO。

<a id="m9"></a>**M9. K. Mishchenko, A. Khaled, P. Richtárik, Random Reshuffling: Simple Analysis with Vast Improvements.** [arXiv:2006.05988](https://arxiv.org/abs/2006.05988)。2026-11-15读取官方摘要，HTTP200、43,602bytes，记录位于`R2/audit/literature_stage283/`。有限样本gradient-descent中reshuffling与顺序采样的差异支持把训练次序作为独立优化因素检查；其理论不直接推出本项目AdamW下的预期收益。stage283实测D1/D1WD末轮e偏移与末段类别组成相符后，使用同一第11轮prefix、同事件/batch大小/updates/LR的原顺序和全池随机排列重放，先检验条件末轮效应，再决定fresh训练控制。

<a id="m10-inferno"></a>**M10. P. de Castro, T. Dorigo, INFERNO: Inference-Aware Neural Optimisation.** [arXiv:1806.04743](https://arxiv.org/abs/1806.04743)。2026-11-15核对官方题名、作者与摘要。该方法针对含nuisance的参数推断优化summary statistics，说明分类性能与最终参数不确定度可以是不同目标。PINPOINT先保留完整score模板和响应矩阵；真实曝光/背景约束明确后再比较过程产额拟合的精度、偏差与coverage。当前没有对应nuisance闭环，不将摘要示例当作本项目效果，也不替代本轮BA/τ工作点评价。实现前须补读其likelihood、近似和优化细节。

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
