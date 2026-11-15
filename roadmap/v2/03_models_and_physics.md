# 模型改进与物理信息利用设计

**R1已建立同输入基线，本轮B1 ResNet最强。下一步先分开检验训练配方与数据规模，再增加真实几何邻接、跨层连续性和高分辨局部路径。对象重建与预训练随后服务已识别的信息缺口。**

B0–B4基线的41模型完整比较与452项验收见[stage274](../../workbook/2026-11-15_stage274_r1_final_results.md)。三类化没有稳定改善共同任务，B1/B2 compute的μ改善伴随e/NC损失，训练收敛未建立。新增关系、对象及专家结构仍为待检验方案；论文编号见[文献表](05_literature.md)。

## 1. 应保留的现有成果

继续使用 accepted 原生 binary pixel、合并后的 bar 能量/PE proxy 和 tracker digitization；保留完整事件、真实空响应、坐标有效性与模态 mask。现有 revision 3 的尺度归一化、LayerScale、QK normalization、FP32 head、empty-response bypass 和真实小样本拟合检查继续复用。

现有模型已经有块内 attention、原生位置残差、多尺度 local/regional skip 和可学习的 global 表示。新实验只修改待回答的关系/压缩/融合因素，不把这些已实现机制重新包装成新增贡献。保留旧模型的 source/config/checkpoint，新增结构使用独立 revision。

## 2. 同输入基线：先证明复杂结构确有必要

| ID | Input representation | Model | Main purpose |
| --- | --- | --- | --- |
| B0 | 观测总量、纵向/横向形状、presence、可测方向/关联质量 | HistGradientBoosting 或小 MLP | 检验接受度、能量和形状能做到多少 |
| B1 | 合作组式 bar 双投影、1mm pixel 双投影、tracker 分支 | CNN/ResNet + late fusion | 对齐三类研究；隔离图像表示与训练协议差异 |
| B2 | 原生 pixel/bar + 独立 tracker set encoder | 当前 ParticleNet adaptation + late fusion | 局部图基线；fixed geometry graph 为默认，dynamic graph 单列 |
| B3 | 原生 pixel/bar + 独立 tracker set encoder | 几何 ParT adaptation + late fusion | 补齐此前 ParT 缺少 tracker 的公平比较 |
| B4 | 原生 pixel/bar/tracker | 现有 multiscale revision 3，Scratch / MAE3 | 共同对照；旧权重只作 warm-start/reference，新样本合同下重新训练 |

所有模型均做 T3/T4；screen 在 100k 四类池及其相同非 τ 子集上执行。B0/B1 的输入来自同一 accepted observations，额外 reconstructed tracker 特征按 R4 的质量合同单列，不能把新增物理重建与 backbone 效应混在一起。

各网络至少有相同调参额度；独立选择合适学习率和正则化。相同默认超参数并不保证公平，强制所有架构都只训四轮也不能回答其实际最佳效果。最终同时给相同计算预算结果和各自收敛结果。

### 2.1 R1发现的稀疏μ问题与后续对照

[stage256](../../workbook/2026-11-15_stage256_r1_sparse_muons_and_nine_model_conditions.md)实测：seed42的B1 T3相对B4，channel数<100的μ recall差为−24.24 pp，CI95[−25.76,−22.41]；该组对全域μ差贡献−6.79 pp。241个全空μ事件的正确差为0，损失来自有响应的稀疏事件。三seed读回现已完成，见下段；occupancy与几何/接受度相关，当前未识别因果机制。

[stage266](../../workbook/2026-11-15_stage266_r1_occupancy_three_seed_readback.md)已完成18模型三seed原始预测读回、1,980项核对：T3低occupancy μ差为−11.46/−24.24/−12.03 pp，均值−15.91 pp，对全域贡献−4.457 pp；T4 conditional同样三个seed均负。B1同结构compute−main T3在该域的μ差均正，平均+16.48 pp，e/NC却分别−10.19/−8.33 pp，不能宣布逐类问题修复。B1/B4 compute各用本族T4预算，跨族不是共同计算控制。点估计与seed SD见[读回结果](r1_occupancy_seeds.json)；没有新增事件或seed总体区间。训练/决策和压缩因素继续分开验证。

B1实际输入是1mm双投影，ResNet累计stride32；它已包含八个channel totals、presence和独立tracker set分支。仅增加计数或presence不能作为新增几何信息的证据。R2先保留结构研究收敛，R3根据三seed结果安排以下最小控制：

| Control | Change and question | Required measurement |
| --- | --- | --- |
| Projection downsampling | 单独将首层stride4改为2，保留通道数及其余层；检验早期压缩是否影响稀疏响应 | 同曝光、实测额外计算/显存；全域逐类与固定occupancy分层；不能把参数相同解释为计算相同 |
| Sampling phase / filtering | 先以单点、短线和真实Train响应检查栅格相位敏感性，再决定是否加入降采样前滤波[P5] | 记录映射与边界，不把探测器整体平移当物理对称性；滤波可能削弱孤立信号，收益待测 |
| Sparse local residual | 在投影融合前保留轻量的高分辨率局部方向/邻接路径，与现有totals skip分开 | 使用全部accepted响应，记录短结构经过聚合的保留率；局部稀疏卷积不能自行连接相隔很远的plane[P6] |

每次只改变一项，先做真实拟合与吞吐验证，再按相同HPO额度比较。`pixel+bar+tracker`原accepted channel数是固定报告分组，扰动/移除后的计数不用于重定义该组。若以后使用occupancy gate或专家混合，路由只能依赖观测，并在Train拟合、Tune选择；全空、低occupancy和全域均保留在分母中。这些是与已冻结R1合同分别配对的结构干预。R2已完成，M0重混因μ质量约束未通过而不采用，G1 stride2已完成，对D2的BA3差−0.0333 pp [−0.3657,+0.3533]，e recall−1.43 pp，当前配方不晋级，见[stage297](../../workbook/2026-11-15_stage297_r3_stride_endpoint_and_resource_handoff.md)；相位敏感和小样本拟合不作为泛化收益。

[stage258](../../workbook/2026-11-15_stage258_r1_ranking_and_training_tradeoffs.md)补充了同结构的有效训练对照：B1 seed42 fresh计算控制在391,633次曝光、wall-based schedule下，低occupancy μ多识别879个，域内recall差+31.38 pp [29.53,33.04]；同组e/NC分别少识别703/367个，全域BA反而下降0.493 pp。全空事件仍全部判NC。结构和投影相同，预算与schedule共同改变，说明训练配方和决策取舍必须与压缩假设一起排查。R2先跟踪逐类概率、recall与AP/ROC；必要时在Train-inner或Tune内按family交叉拟合低维logit-bias/冻结readout诊断，检验固定表示的可读性。该诊断不打开Cal-A，也不重写R1排名。

### 2.2 μ的多类决策与分数辨别能力

[三seed读回](r1_muon_decision_review.json)表明，B1的argmax μ recall损失与one-versus-rest分数的表现不同：

| B1 minus B4 | Mu argmax recall difference | Mu efficiency difference at 1% FPR | Mu efficiency difference at 90% benchmark purity |
| --- | --- | --- | --- |
| T3 training | −2.410 pp [−3.209,−1.690] | +4.850 pp [+4.057,+5.823] | +8.503 pp [+6.486,+10.052] |
| T4 conditional | −1.853 pp [−2.611,−1.063] | +5.383 pp [+4.560,+6.297] | +8.720 pp [+7.431,+10.288] |

T3 μ的1% FPR效率逐seed增益均为正：+5.36/+4.32/+4.87 pp；argmax recall差为+1.37/−6.91/−1.69 pp。B1/B4 T3平均μ AP为.6483/.6000，AUROC为.7591/.7278。这些是同Tune的post-hoc开发读回，区间条件于三个已拟合seed及实际设备/精度，不能当成种子总体区间。

因此R2的决策诊断应在保留原始概率的同时，检查类间offset/决策规则、逐类排序和训练schedule；当前尚不能把μ argmax损失认定为不可逆投影信息缺失。各类fixed-cut选择不构成互斥的三类输出，需另定义多cut冲突/拒判及coverage，不能把单类效率增益写成已经修复argmax。R1预设μ非劣条件仍未通过，不因新增指标改写晋级规则。

R2固定screen的[新增读回](r2_progress_review.json)也显示，低occupancy子域本身的μ AUROC目前约.50–.57，不能把本节全域排序结论外推到该子域。终点评价须同时检查局部排序、预测类比例和e/NC取舍；低维bias仅诊断决策部分，高分辨残差与几何关系负责检验缺少的可用特征。

### 2.3 图像基线的紧凑表示对照

代码复核确认B1对pixel/bar共同使用物理1mm的1322×420网格，两个投影共享CNN权重；合作组报告按pixel plane/panel排列多幅紧凑图像，pixel示例15×250/15×200，strip示例68×40/56×40。二者的空行数量、局部邻接及卷积计算不同，R1并非该表示的严格复现。沿z缩短网格可能减少无响应间隙上的dense计算，是否改善训练或分类尚未测量。

已实现独立紧凑图像候选C1：bar保留真实panel测量轴，pixel保留plane与1mm横向投影；同一accepted响应、信号、边界、totals及tracker set继续保留。111行按真实sensor z排序，真实z、前后层间距与sensor类型经五个静态通道输入，机制参考[CoordConv](05_literature.md#p7)。图像尺寸由accepted geometry生成。256个Train事件、1,490,680条响应的原index和值逐项可逆；32-event T4真实拟合与完整Tune资源/恢复已通过，同100k×4正式分类已完成：相对D0的BA3仅+.1367 pp [−.2134,+.5036]，e recall下降3.42pp，τ无收益，当前配方不晋级。同设备训练吞吐2.723倍、含准备的短operation吞吐1.747倍；保留成本研究候选。五折global-bias修正未恢复逐类质量，见[stage294](../../workbook/2026-11-15_stage294_r3_compact_results_and_matched_cost.md)和[stage296](../../workbook/2026-11-15_stage296_r3_readout_and_shared_final_probes.md)。

先以同Tune40k、同Train100k和4 passes作小规模基线比较，再按实测选一项继续监督；若需要公平等时结论，另在同allocation匹配预算。网格、模态分支、stride与容量分别记录，不能把组合改变后的收益归因于单一因素。其主决策同样包括全域和稀疏逐类、τ工作点及成本。原研究的exact scores/事件列表尚未取得，也不阻止这一同响应的表示研究。

## 3. 原生几何关系：R3 G2及条件扩展

```mermaid
flowchart LR
    P[Native pixel channels] --> L[Plane local encoder]
    L --> X[Window boundary exchange]
    X --> C[Cluster and region features]
    C --> G[Cross-plane sparse graph]
    B[Bar intervals and signals] --> H[Panel encoder]
    T[Tracker hits and covariance] --> S[Station relation encoder]
    G --> F[Geometry aware fusion]
    H --> F
    S --> F
    X --> R[Raw local readout]
    R --> O[Three or four class heads]
    F --> O
    G --> A[Optional association and direction heads]
```

### 3.1 局部表示与窗口边界

正式R3 G2使用B2原生族dim64及本族匹配控制；B4 dim96的后续迁移须单独对照。压缩前消息保留全部raw节点，连接同plane的跨窗口节点及跨plane候选，检验短结构在固定窗口边界处的信息路径。两种低成本实现择一：plane 内 sparse 2D convolution；或基于 measured xy 的 local EdgeConv，邻居数从 8/16 起做 train-only profile。窗口平移只作为边界鲁棒性诊断；不把探测器整体平移、任意旋转视为普适物理不变性。

每个 block/cluster 显式保留 centroid、测量轴 mask、xy covariance/extent、channel count、signal summary、plane/module ID；raw local skip 继续存在。原生 residual 与 cluster moments 同时提供，但 binary pixel 不加入模拟 charge 或 per-track deposit。连接降采样前后的映射，以测量真实局部信号经过压缩的变化。

### 3.2 跨层候选图

已实现的R3 G2原型在全部原生节点上增加一层压缩前消息：同plane跨窗口最多4条，前后同类可见plane及跨传感器候选合计出度最多14。1024个平衡Train事件的同recorded-trackID候选recall，pixel gap1/2/3为27.53%/10.15%/17.48%；该truth只用于审计，不参与构图。PixelSD首贡献者标签不等同完整ancestry，候选recall也不等同整轨效率。低recall要求R4检验更广方向候选、歧义和学习筛选；不能仅因98%以上节点有边就判定关联充分。正式分类增益尚未测量。

新增材料输入为名义轴向slab厚度积分：silicon header单位μm，10换算为0.010 mm，W/scintillator header为mm。它们不是真实轨迹长度、动量或辐射长度。原型修订号3的学习检查与旧revision2分开；X0与散射协方差只有在材料成分和传播模型明确后另行加入。下面保留对象级图和其他候选实现的后续设计。

本族control与relations的首轮比较估计“增加关系分支”的整体效应，额外参数、计算与实际邻域一起改变。若它有值得扩展的收益，再增加一个同容量local MLP或保持degree/plane/modality分布的打乱边重训控制，区分容量与具体几何关联的贡献；打乱只能在事件内部进行，不允许混入其他事件。固定权重时打乱边衡量依赖，不能替代重训控制。有限预算下先完成整体效应，不据此单独宣布几何归纳偏置已被证明。

节点先用 observation-only cluster 或高占用窗口 token，边连接相邻 1–3 个可见 plane，并允许越过无响应 plane。构图依赖已知 geometry、观测 hit/interval 与 uncertainty，不读取 trackID、PDG 或真 vertex。

候选边特征包括：

- `Δx, Δy, Δz`、`Δx/Δz, Δy/Δz`、距离和 measured-axis mask；pixel 斜率保留符号。
- 直线/已知名义场传播后的残差及测量协方差，材料厚度/辐射长度、plane 间距和相对 module。
- Cluster width、局部密度、可见 hit count、signal ratios、隔离度和候选歧义数。

bar 只在其测量轴比较相容性；另一维用区间/边界表示或边缘化距离。把 bar 的未测坐标中心当精确 hit position 会制造方向约束。Tracker 的 station ordinal 保留为 ID，另从 accepted G4 geometry 加入真实 station z 与 covariance；名义 field/material 是已知 detector context，不能用真粒子动量完成候选筛选。

先用固定物理 gate 和有限邻居的 sparse message passing，再考虑 learned edge score。图只表示候选关联，不要求每个节点立即归属于唯一粒子。测量 candidate-edge recall 与边数/事件尾部，控制 pruning 对短轨/高占用的偏差；不能只追求较高 edge precision 而丢掉 rare signal。

S2 已有 block 级跨层 attention/graph 及 plane-only 控制，τ AP 主区间跨零，见 [stage54](../../workbook/2026-11-12_stage54_s2_cross_plane_controls.md)。本轮从其代码复用候选计算，重点改变关系发生的层级、方向/uncertainty 特征和非均匀几何；将这些因素逐项消融，保留旧一般几何距离/attention 为控制。

### 3.3 Region/global 关系

将 region 从“按 canonical 序列每 32 blocks”改为明确的几何邻域，或在原序列 region 之上增加跨 region 边。region token 显式携带 xy/z anchor、extent、plane 和 measurement mask。全局 attention 的 relation bias 使用这些测量几何，先比较：

| Variant | Single change | Question |
| --- | --- | --- |
| G2a | 跨窗口/跨层 sparse edges，保留现有 fusion | 关系是否在进入压缩前增加可用信息？ |
| G2b | 原 backbone + geometry-anchored region/query | query 同质化和几何定位是否限制信息读取？ |
| G2c | 用 late fusion 替代 global latent 路径，保留相同输入 | global 路径是否值得其参数与计算？ |
| G2d | 最佳关系结构的 native / 100μm / 1mm | 新结构能否把原生粒度转为任务收益？ |

G2a/G2b先单因素，只有二者各有价值才组合。这里的G2a–d为设计子项；正式执行ID中的G1专指B1 stride控制，G2专指原生关系族。G2c是重训ablation；置换 feature 只衡量固定权重依赖，不能替代它。Latent 数 64→128/256、queries 2→4 在 plateau 后做小范围扫描，单纯增加数量不视为关系问题的解决。

### 3.4 高占用执行

先用 packed/ragged blocks、occupancy bucket 和等价的 transform cache，减少旧固定 32 槽的 padding 开销。跨层图按候选边线性扩展，避免对三万余 hits 建全事件 N² attention。若需要 token budget，采用 coarse 全局覆盖 + fine 局部残差/候选路径，每个通道继续有聚合路径；不能仅保留最高能或最密集 tokens，因为孤立轻子/短轨可能是关键信号。

压缩质量应在全 occupancy 分位数、低 occupancy 短轨、高 occupancy shower 与空事件上评价，并记录实际丢弃/聚合信息、峰值显存、RAM 和吞吐。调用自定义 C++/CUDA、Flash3D 或 PTv3 serialization 之前，先证明现有邻域/内存算子是主要瓶颈。[P3–P4]

## 4. 优先物理分支：tracker 与 raw/object 双路径

### 4.1 Tracker 的信息增量

现有 tracker 已带来 μ/NC 收益，但主输入的 z 是 station ordinal。先在相同 digitized hits 上增加真实 station spacing、观测关联质量和方向；这些来自 detector 描述，不改变已接受的 resolution/efficiency。

关联从候选 triplet 的宽 gate 开始，分别检查非弯曲 x 与弯曲 y，利用名义磁场传播和 uncertainty。三站测量能定义弯曲代理，但不自动提供有自由度的可靠 momentum fit。若使用 momentum，需输出估计误差、有效区间及 high-momentum censoring/fit-failure mask；在 field/geometry 和少粒子闭环通过后加入分类。保留 hit-set 路径覆盖拟合失败事件。

### 4.2 Pixel tracking 与顶点

先实现多候选方向相容关联与 robust fit/RANSAC/Kalman 类控制，替代 mutual-nearest-only 的早期决策。对 shower 核心、孤立轨、缺层、散射和次级起点分别定义 eligible 对象。再比较 learned edge classification 或 Object Condensation [H3–H4、H6]；观测候选图先固定，truth 只监督边/实例 labels。参照Exa.TrkX分别评价构图、筛选和整轨形成，记录每一步的信号损失；不沿用碰撞点origin约束去排除次级τ/charm轨。

对象评价至少包含：candidate recall、matched track efficiency、fake/duplicate rate、hit purity/completeness、direction residual、primary/secondary vertex residual、质量阈值下 coverage 和全事件成功率。按可见 plane 数、能量、occupancy、材料和粒子类别分层。首次贡献者 truth 的混合/不完整性作为 label-quality mask/soft target 处理，不能宣称为逐 hit 完美 ancestry。

raw/object 融合提供 object tokens（方向、起止位置、covariance、quality、association ambiguity），保留 raw local/global 输出。对象缺失是显式状态，不把无轨事件丢掉。对象统计 HGB 是低成本控制，learned object set encoder 是下一步。

“truth 关联替代重建”的实验仅作研究诊断：若 oracle association 有改善，说明重建是潜在限制；若没有，继续检查表示、任务和统计。它既不是可部署输入，也不是测得的硬件性能上界。正式晋级只接受 observation-only 重建结果。

## 5. τ：分离可见性、衰变拓扑与任务竞争

### 5.1 物理尺度

PDG 的 τ 寿命对应 `cτ≈87 μm`，质量约 `1.777 GeV` [H5]。高能近似平均飞行长度为：

```text
L_tau ≈ (E_tau / 1.777 GeV) * 0.087 mm
E_tau = 100 GeV  ->  L_tau ≈ 4.9 mm
E_tau = 1 TeV    ->  L_tau ≈ 49 mm
```

这是尺度估算，使用的是 τ 能量而非入射 ν 能量。细 transverse pixel pitch 并不保证沿 z 有足够采样点：需要同时考虑实际 plane 间距、材料、母 τ 可见长度、daughter 接受度和散射。S6 primary vertex 的 |Δz| 中位数约 53.6 mm，也提示当前 event-level vertex head 不能直接当精细 kink 重建使用。

R4/R5 分别研究：有可见母轨+daughter 的 kink；没有母轨但 daughter impact parameter/起点可测；多 prong secondary vertex；daughter 没有已标直接响应的事件。S5 的 `fromTau` 首贡献 flags 和 direct-prong matching 有各自定义，“无已标 daughter response”不等于没有任何由 τ 产生的间接测量信息。

### 5.2 τ specialist 的顺序

先训练 `τhad vs non-τ`，因为已有数据下 hadronic branch 贡献大；同时做 `τe vs prompt e`、`τmu vs prompt mu` 的局部判别和全非 τ 背景评价。专属 prompt 类对照可帮助诊断，但最终 FPR 的分母必须包含 e、μ、NC、charm 与全部拟部署背景。

specialist 输入优先使用观测 lepton/track candidate、shower start/width、longitudinal profile、相对方向、隔离度、DCA/impact parameter 与可用的缺失动量 proxy。只有在 visible-system 和接受度模型可验证时才引入 missing-pT；不能拿 generator 真能量、真 vertex 或真 pT balance 当测量。

gate 由观测质量和概率决定，允许 soft routing，使 poor-reconstruction 事件仍通过 raw expert。最终共同四类概率与 specialist 的合并方案在开发集拟合，完整模型重新校准。不要用真衰变 branch、真 decay-in-gap 或真 charm label 路由推理。

### 5.3 采样与损失

保留 coarse 四类总权重；τ 内部 conditional branch 可以分层采样或 inverse-frequency reweight，但需归一化 τ 的总质量，记录目标先验及校准修正。六子类均衡采样会把 τ 的总训练质量从 1/4 改成 1/2，是不同干预，不能与“细分类别”混为一谈。

先比较 CE/BCE 基线和单一 τ branch 权重。低 FPR specialist 若仍受整体 CE 目标限制，再用 pAUC 或 hard-negative replay [M2]：以训练集 prompt/charm/hadronic fake 为难背景，保持随机背景成分和重要性权重。小 batch 内不能稳定估计 0.1% 尾部，需跨 batch train-only buffer 或大批 score bank，按 epoch 更新；不能把 validation 失败事件回流训练。

charm 通过真 labels 做诊断和监督，通过观测 topology 做输入。条件 decay 与 charm/secondary 关联应优先落到 local/object tokens；每次只增加一种辅助，测共享梯度方向和分类负迁移。更多辅助 loss 未通过真实任务收益时，保留相应诊断，不继续堆叠。

### 5.4 可见拓扑与未分辨衰变点分别研究

OPERA最终分析[H7]按1h/3h/e/μ的可见衰变形态使用topology与kinematics，再训练各道BDT；其背景明确包括漏识别primary μ的charm、强子再相互作用和大角度μ散射。这支持本项目先改进候选、prompt lepton识别与对象质量，再研究hadronic子型。上述背景在本项目中的贡献和tag覆盖尚未测量，后续先盘点已有truth支持；未标注事件保留为unknown。

NOMAD[H8]的τ衰变点无法空间分开，采用完整可见末态运动学及其相关性识别τ。因此PINPOINT应保留两条路径：可分辨轨迹/顶点的局部拓扑，及未分辨衰变点的观测运动学。第二条先在tracker方向、动量proxy、可见系统和接受度上验证resolution–coverage；缺少量测能力时不能直接把truth missing-pT加入模型，也不能把无kink当作无信息。两条路径均与raw backbone配对，推理路由只使用观测质量。

进一步拆分had1/had3前，区分tau tree直接charged daughters、经共振展开的最终charged prongs、可见prongs和重建tracks。现有`tau_direct_prongs`只实现第一种计数；其与其他定义的一致性尚未测量。先补label/coverage诊断及条件辅助，只有合并四类或τ工作点改善且共同三类非劣时，才增加正式子类。

## 6. 回归与方向目标的改造

当前 event-level signed px/py 改善弱，应先用冻结 raw/local/graph/object features 的同容量 probe 判别信息可读性，再选择新 head。

- 对 track/cluster direction 预测局部单位向量或 `(sinφ, cosφ)`，同时保留 magnitude 与可见性/不确定度；φ 的周期和低 pT 时不适定单独处理。
- 事件系统用分量标准化的 likelihood 或方向分布（例如 circular/von Mises head）表示不确定性；它可能减轻均值收缩，收益尚需测量。
- 主要验证是分量偏差、角度相关、resolution–coverage、likelihood calibration，以及对分类的实际增量。只优于常数一点的 MSE 不等于可靠重建。
- 依次比较 accepted-response 内的可见 deposit/profile target、可重建局部对象 target、旧 visible-primary proxy；推理输入不含任何 target。

## 7. 预训练与鲁棒性放在明确缺口之后

R2 得到强监督基线后，才在最佳 encoder 上追加：

1. **Topology-aware MAE**：遮蔽连续 plane/window 或整模态的一部分，让模型预测邻域 occupancy/方向连续性和实际可测 signal；负位置从已知 geometry 采样，避免只给 active 坐标泄漏 occupancy。
2. **局部监督**：可靠的 bar soft semantics、track continuity、primary/secondary 关系或 instance labels。Pixel 混合 contribution 不足时使用已验证的弱标签 mask，不硬造完整 soft fractions。
3. **跨模态预测**：用 pixel/bar 预测 tracker 的概率分布/观测摘要；缺失与 out-of-acceptance 分开，不能因本来不在接受度就要求精确命中。

每项与同总计算的继续监督训练比较，分别报告预训练数据量、精细 truth 标注和 continuation 成本。FASERCal 的完整 Rel 和当前 bar-only Rel 属于不同监督覆盖。

readout 采用已接受幅度下的 gain、inefficiency、noise、alignment 和 missing modality augmentation；先保留未参与训练的 pattern 作 held-out 条件。对 generator 变体做独立 hard-scatter 条件匹配，具体统计如旧 S6 保留。若存在稳定的最差域问题，再考虑 group-balanced sampling/GroupDRO [M3]；不强行消除 energy/occupancy 中真实的物理信号。

新模型必须持续保留真实小样本拟合、各分支激活/梯度/参数更新、概率方差及 loss-relative-to-constant 证据。出现已知学习失败时先修复和复训，再作架构/物理结论；这些健康检查服务于模型质量判断。
