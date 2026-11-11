# 模型、训练与实验设计

本方案的独立变量是硬件信息、表示压缩、架构和训练目标。先固定数据与评价合同，再逐项改变；所有数值超参数均为起始建议，最终依据 validation 与资源 profiler 冻结。

## 1. 可测物理线索与任务分解

| Process | Candidate observable evidence | Main confusions and checks |
| --- | --- | --- |
| CC νe | 起始区 EM shower、早期高占用、窄核与长程发展、hadron/EM 空间关系 | π⁰ conversion、charm→e、τ→e；比较 shower 起点与可见 secondary displacement |
| CC νμ | 跨层细长低占用轨迹、穿透性、与 FASER 轨迹的测量关联 | hadron punch-through、secondary μ、charm/τ→μ；报告 FASER acceptance 内外效率 |
| CC ντ | 可见 kink/secondary vertex、1/3-prong 等衰变结构、不同模态的能量与方向不平衡 | 三种 τ decay branch 分别评价；与 charm 相似的 displaced topology；不能使用真实 missing momentum |
| NC | 缺少可辨识 prompt charged lepton、hadronic shower 与 leakage、secondary activity | 低能/出接受度 CC、τ→had、含 charm 的 NC；“没有看见轻子”不等于“没有轻子” |

τ 的真衰变长度可用于理解接受度：`cτ≈87 μm`，`L≈(Eτ/mτ)cτ`，例如 1 TeV τ 约 49 mm。这是物理尺度估算，不是本样本已测分辨率。S0/S5 用 tau truth 统计在各材料/间隙和相邻测量面之间的衰变比例、可见 daughters、projected separation 相对 pitch/误差；再以 detector hits 检查是否存在可分辨证据。对无法看见 kink 的事件，依靠全局拓扑/不平衡的概率信息，并报告条件效率。

## 2. 输入表示：完整目录只构建一次

**Pixel token：**`(x,y,z,plane,module,detector type,validity)` 与 binary response 为主；若确认硬件可测 analog，则增加 `log1p(Q/Q0)`、饱和/threshold mask。保存绝对 detector frame 与可观测 event-centroid-relative frame 两种位置、到边界的距离、局部材料/plane 描述。归一化采用固定物理尺度或训练集统计，不逐事件把 shower 长宽标准化掉；不以真 vertex 居中。

**Bar token：**`(panel z, measured transverse coordinate, orientation, full bar extent, measurement covariance, channel response, valid mask)`；时间只在 channel digitization 和 event-time reference 通过验证后加入。bar 的未测坐标表示为区间/大协方差，不当成 bar 中心处的精确 hit。按 detector type 独立归一化 signal，避免 silicon 和 scintillator 的沉积混成统一 energy unit。

**Tracker token：**先用 digitized station hit set；再比较测量拟合的 track parameters、covariance、fit quality、hit count 和 acceptance flags。轨迹关联图与拟合器只读 observation。无信号、无接受度、坏道和重建失败使用不同 mask；未测的动量值不能用真值替代，也不能用缺省零冒充有效 `q/p`。

所有原生 channel observations 保存在同一 schema 的 shards。 coarse rebin、patch encoder 和 track objects 从该目录派生；不为每个模型复制 ROOT/Parquet，不用一事件一 `.pt/.npz`。新表示保留 channel→token 的映射，以便画原生分辨率 event display 和定位丢失的信息。

### 2.1 原生分辨率不等于全局 N² 注意力

先在**真实 pixel lattice 的每个 sensor plane**做 stride-1 sparse 2D convolution 或 local set/EdgeConv，保留 local residual coordinates；再按物理邻域聚合为多个 tokens。起始候选为 `16×16 / 64×64` 原生 pixels 的 window（约 `0.33×0.36 / 1.33×1.46 mm²`），并自适应细分高占用窗口。低占用 plane、孤立 track-like pixels 和边界信号有单独通路，防止 shower core 独占 token budget。

每个窗口用 learned set encoder 编码完整内部几何，附 multiplicity、centroid、covariance、signal sum 和跨度，之后才压缩。它与“只保存粗 bin 中 hit count”是两个实验处理；所有内部原始点仍可供下一次训练访问。压缩损失通过原生局部结构重建、稀疏双轨分离 toy study、真实事件分类与 per-channel attribution 消融测量。

高占用事件优先用 window attention、serialized neighbors、hit-budget bucketing 和 gradient accumulation，不直接 top-energy 截断。若必须限制 tokens，采用分 plane/区域覆盖的层次采样，并记录保留 hit/charge/track-like structure 的比例；验证 τ 短轨、低能分支、边界事件与截断概率是否相关。`Nmax` 只在测得 p99/p99.9 后冻结，overflow 有完整处理路径。

## 3. 三组架构与推荐主模型

### 3.1 独立基线

**B0：可解释 feature classifier。**从观测构造各 plane occupancy、longitudinal/transverse moments、bar signal profile、penetration、digitized tracker multiplicity、shower compactness、边界距离、简单可重建 track/vertex quality；使用 BDT 或小 MLP。另训练 only-global-signal、only-geometry/acceptance 和 shuffled-local-geometry 的诊断模型，定位网络是否主要依赖能谱或位置。truth 版 kinematics classifier 可单列为信息上限，其结果不计入 hardware baseline。

**B1：ParticleNet adaptation。**分 detector 的 EdgeConv + masked pooling + late fusion。起始 `k=16`（验证 8/16/32）、3 blocks、hidden `64→128→128`。先在 plane 内建几何邻接，另建 adjacent-plane 的方向相容连接，避免裸三维 Euclidean KNN 只选同层；对 bar 用测量相容度，不用虚构精确 xyz。fixed 与 dynamic feature-space graph 单独比较。

**B2：Particle Transformer adaptation。**同一硬件输入集合，4–6 attention blocks、`d=128/192`、4–6 heads、classification token。对完整低占用事件和局部编码后的高占用事件分别验证。`attention(Q,K,V)` 的 logits 增加由测量 relations 计算的 `bᵢⱼ`，与不带 pair bias 的 vanilla set Transformer 对照；名称明确为 geometry-adapted ParT，不声称复现 jet benchmark。

### 3.2 关系变量与物理几何

关系候选包括同层 `Δx,Δy`（以 pitch/测量误差归一）、`Δz`、跨层 slope、bar/HV overlap、已知材料厚度的 `ΔX₀/ΔλI`、detector-pair type、有效时的相对 channel time。测量相容度可用共同坐标框架下的 covariance 加和构造 Mahalanobis 距离；cross-layer slope gate 需要磁场与多重散射容差，不硬编码 truth track。

pair 信息由小 MLP 投影到每个 attention head 的 bias。无相同测量含义的 `Qpixel/Qbar` 不直接作比；缺失 signal/time 分量有 mask。禁止 hit 伪四动量、不变质量、true PDG/charge 或 true primary vertex distance。训练初期不强加 SO(3) invariance：束流、磁场和材料顺序有物理方向。

### 3.3 推荐：局部编码、module context、latent fusion

pixel plane-aware local encoder 与 bar/panel encoder 提供 tokens；按实际 z 排列 module context，加入 learned module/type embedding 和连续位置编码。两层局部/module attention 后，以 `L=32/64` learnable latents cross-attend 各模态，再用 2–4 latent self-attention blocks，输出 event embedding。`d=128/192`、4–6 heads、MLP ratio 2–4 是首轮范围，先争取约 2–10M parameters，实际数量在实现后测量。

对于 `N` local inputs 和 `L` latents，融合 attention 约为 `O(NL+L²)`；还需计入局部构图/卷积、decoder 和 padding，不能只报 attention 理论复杂度。pixel/bar/tracker 先独立压缩，再融合，比较共享 latents、per-detector latent quota 与简单 concatenated pooled embeddings，检查小 tracker 分支是否被高占用 shower 淹没。随机 detector dropout 仅在训练中用于鲁棒性，evaluate 明确区分真实缺失和人为关闭子系统。

主输出先用四类 softmax。第二阶段比较六类 softmax 并汇总 `pτ=pτe+pτμ+pτhad`。也可保留四类主头与只在 τ truth 上计算的 conditional decay head；两种方法分别实验，不将 conditional branch confidence 未经校准地当作额外证据叠加。

### 3.4 可选对象路线

对可见 charged track segments、EM shower 与 secondary-vertex candidates 建立 object queries/offset heads（参考 PointINS 的 instance-aware 表示思想），用 detector hits 和 uncertainty 预测对象，再作 event classification。truth track ID 只能定义监督和评估 matched/fake rate。保存 raw-hit 路径与 residual tokens，让错误对象构造不把事件整体丢弃。只有 object efficiency、purity、vertex resolution 以及四类指标都有报告时，才比较 object-only 与 raw+object fusion。

## 4. 训练目标与辅助物理任务

主损失为 `Lclass=CE(y4,p4)`，训练 class-balanced sampling 与 natural-mixture sampling 是两个预先指定的对照；sample/class weights 不重复补偿。plain CE 为基准，再比较轻微 label smoothing（0–0.02）。标签冲突先按数据合同处理，不用较大 smoothing 隐藏冲突。

优先增加：可见 primary-vertex 预测、local EM/muon/hadron soft semantics、visible signal/momentum 代理、charm presence/decay branch。一次只增加一组，报告 `L=Lclass+Σ λt Lt` 的 λ、有效 label mask 和梯度规模。先固定小 λ（如 0.1 起始），再比较 FASERCal 的 homoscedastic uncertainty weighting；若某个 auxiliary head 只改善自身且损害分类，则降权或移除。

kinematic target 的“visible”定义必须固定：选定真 final-state 粒子集合、是否含 nuclear/remnant、不可见 neutrinos、可进入接受度或全部末态，不能混用 `Eν−Eℓ`、calorimetric signal 和真 particle energy。对 NC 的 primary charged lepton、无可见 vertex 和不可重建 track 使用明确 target mask。回归 `log1p(magnitude)`/vector components、Huber loss 与异方差预测，report median/68%/95% residual，分母接近零时使用绝对 residual。

`pTmiss` 只有在可测 visible momentum 和束流方向模型定义后才作为派生估计；对漏出接受度和核效应分别分层。真实 `pTmiss` 可作辅助 target/诊断，但不作为输入或 event selection。

τ 标签从 primary CC τ 的 direct decay ancestry 定义，比较 `e/μ/had`，不能用“事件内出现电子/μ”来分道；charm 的 leptons 另计。多 charm 的四类排他标签需要先固定 precedence（如 μ→e→had）或改用 multi-label；汇报原始定义与各 generator 一致性，不沿用旧 `trackID==1` 假设或不完整 PDG 列表。

## 5. 预训练：分清自监督与模拟监督

### 5.1 P0：先验证直接监督是否足够

四类 label 已随模拟提供。先绘制总 labelled budget `10³/10⁴/10⁵`（足够时扩展至 `10⁶`）的 learning curve，按固定类别和 τ 子类策略抽取 train partition，验证/校准/测试不变。若 full-data supervised 已饱和而 pretraining 仅增加成本，则优先修正硬件信息和系统效应。

### 5.2 P1：稀疏 masked reconstruction

从 train partition 的 `10⁵` 事件起步，成功后扩到 `10⁶`，不先跑全部库。mask ratio `0.5/0.75`，混合局部 contiguous windows、整 plane、bar panel 和整 detector masking。visible encoder 不能接收被 mask 的 active-hit 列表、精确 hit 坐标、hit count 或由全事件计算的泄漏统计量；若使用完整 detector geometry，它必须对有/无 hit 的 channel 同样已知。

借鉴 NOMAE，在预先由几何/visible 邻域确定的有限区域中预测原生 occupancy（含负候选）与有效信号；对 sampled negatives 明确采样分布/importance weighting。可再预测局部 signal sums、moments 和跨层 continuity，逐项消融。binary pixel 不重建 analog `edep`。避免只对“已知存在的 masked hits”算 occupancy loss，也避免用全空白通道占比制造低 reconstruction loss。

NOMAE 原文不对远离 visible voxels 的孤立点计算 loss；PINPOINT 不能默认这些点无价值。增加与真实 active support 无关、按几何抽取的跨层 windows/远区候选，并分别统计 isolated track-like signals、短轨和 shower core 的重建覆盖率。整 plane/modality masking 时使用 module geometry queries，而不是只用不存在的同层 visible 邻域；不借 truth 选 decoder 的正候选位置。

跟踪 occupancy precision/recall、负候选分布、local structure residual、各模态的 reconstruction loss 与 downstream score。重建清晰图像只作诊断，晋级依据独立分类/效率指标。

### 5.3 P2：自蒸馏与跨模态预测

Sonata 风格 EMA teacher 看完整**硬件 observation**，student 看部分通道/模块；只对可对应的同一事件区域或 event embedding 做 prediction/distillation，检查 feature variance、collapse 和 frozen linear probe。Concerto/Point-PQAE 风格的 cross-view 采用 known geometry + uncertainty 的匹配，H/V bar ambiguity 允许 one-to-many，而非强行对齐。

augmentation 限于有依据的 gain/noise、dead channels、alignment jitter、适度 detector dropout 和随机 mask。所有变体属于同一 generator-event family。任意三维旋转、mirror、uniform rescaling、随机删除全部 τ daughter 后仍强制原类别不变的 contrastive pair，均不作为默认 augmentation。更强几何变换需同时变换条件/field 且验证物理意义。

### 5.4 P3：Relational continuation

现有 pixel 聚合只累加 `edep`，PDG/ancestry flags 取首次创建 pixel 时的贡献者；无法由现存 flags 精确重建 soft mixture。Scintillator per-track edep 可以辅助构造贡献标签，但先检查 source 和 channel 映射。旧标签作为有偏弱标签单列对照，不冒充 FASERCal soft truth。

新模拟仅在必要时增加紧凑的每通道语义能量累加量（EM/muon/hadron、primary/secondary、primary-lepton/charm/τ），保留混合分数/validity，不输出所有 G4 steps。可选 matched/fake spacepoint target 的 ghost 定义必须来自明确重建流程。先做小统计和标注质量检查，再按 HTCondor 扩展。

对同一 encoder 比较 Scratch / MAE / MAE+Rel；另给 MAE 相同追加 GPU-hours、给 Scratch 相同总训练预算，防止把 100 额外 epochs 的效果归给 Rel。仅无下游标签的 reconstruction/distillation 称为 self-supervised；Rel 和 event-label auxiliary training 清楚标为 simulation-supervised。

## 6. 最小实验矩阵与逐级扩大规则

除指定变项外，split、读出版本、训练样本、normalization、checkpoint-selection rule 和 evaluation code 相同。先单种子 42 筛选，再对保留的结论做 `17/42/2026` 三种子；报告筛选预算和最终完整训练。

| ID | Variable | Required runs / comparison | Primary question |
| --- | --- | --- | --- |
| E00 | Data contract | ID/label/unit/channel/empty-event/whitelist checks；truth-only upper bound 单列 | 是否在解正确的硬件问题？ |
| E01 | Information content | binary native pixel；+validated charge；bar；pixel+bar；+digitized FASER；+validated time | 哪些测量真正提供增量？ |
| E02 | Spatial representation | native / 100 μm / 200 μm / 1 mm；相同 analog/binary 合同 | 粗表示损失有多大，发生在哪类拓扑？ |
| E03 | Architecture | BDT/MLP、ParticleNet、vanilla Transformer、ParT、local+latent model | local relation/global context 的作用是什么？ |
| E04 | Fusion | late pooled / joint attention / latent；modality quotas 与 dropout | 是否利用互补模态，是否受某分支压制？ |
| E05 | Pretraining | Scratch / MAE / self-distill / MAE+Rel；同标签、同总 GPU budget | 净收益来自目标还是额外训练量？ |
| E06 | Physics tasks | direct 4-way / six-way sum；vertex / semantics / charm 单独加，再组合 | τ 子道和物理解释能否改善主任务？ |
| E07 | Object branch | raw-only / object-only / raw+object | 可重建对象是否提供超出 raw model 的稳定证据？ |
| E08 | Priors | nominal、class-balanced、energy/vertex matched，only-prior diagnostics | 模型利用的是拓扑、样本先验还是两者？ |
| E09 | Robustness | thresholds/noise/gain/timing/inefficiency/alignment/field/material/generator | 分类收益在何种扰动下稳定或失效？ |
| E10 | Deployment | calibration、batch/order、high occupancy、missing modality、single-event inference | 概率与资源是否满足实际使用？ |

E02 的粗网格按各 plane 的固定 sensor-local 原点定义。binary 读出分别比较 coarse OR（bin 是否有响应）和 coarse count（bin 内 active native pixels 数），后者保留 multiplicity 但丢失内部布局；只有 analog 已验证时才增加同 bin 的 signal sum。所有归约均从同一份 native observation 派生，不能把 binary→analog 的增量归因于空间分辨率。先固定网格原点，另测原点平移的敏感性。

E02 必须至少在 ParticleNet/ParT 两个能处理 point sets 的模型族中核对，不只比较旧 CNN 与新 Transformer。压缩减少 token 的速度优势本身也报告；固定参数/步数和固定总 GPU budget 两种比较都保留。所有 ablations 在 validation 上筛选，最终候选、阈值与校准冻结后统一打开 test。

## 7. 评价：从算法分数到物理效率

### 7.1 数据分母与三种样本权重

主表同时给 `generator source → interaction in declared target/fiducial domain → detector active → selected` 的 cutflow、四类 confusion counts/row-normalized matrix 和 active 条件性能。S0 先核对 generator vertex 是否位于实际目标材料：源文件里的 reaction 与目标探测器内 reaction 不是同一个分母，样本中超出 sensor 横向视场也不自动代表超出更大的 target。目标域内的全空事件须获得先验型输出或明确 abstention，并计入 overall efficiency。truth fiducial domain 是物理样本与接受度的定义，不能作为真实推理时的已知选择；event selection 另用测量量实现。

评价分别使用：class-balanced benchmark；有可靠 flux/cross-section/exposure weights 的 nominal mixture；在共同支持区域做 `Eν`/vertex/target 匹配的 topology diagnostic。matching 权重的边界、截断与 `Neff=(Σw)²/Σw²` 均报告，不能把 extrapolation 区域伪装成匹配。训练 class priors 与部署 priors 不同，若用 prior correction，明确其 label-shift 假设并重新检查条件校准。

### 7.2 必报指标

四类及 τ 子道的 one-vs-rest ROC-AUC、PR-AUC（声明 prevalence）、macro-F1、balanced accuracy、precision/recall；τ 主工作点为 background acceptance `10⁻²,10⁻³` 下的 signal efficiency，`10⁻⁴` 仅在 MC 统计足够时增加。background mixture 在每个表中固定并声明，避免用不同 νμ/NC 比例制造不同 rejection。

报告 calibrated NLL、Brier score、reliability diagram 与 ECE（附 binning/counts）；仅在 calibration partition 拟合 temperature。校准后仍按 energy、vertex、occupancy、containment、ν/anti-ν、τ branch、charm、FASER acceptance 和 reconstruction quality 分层；未覆盖域不报告成“已验证”。

阈值在 validation 上选择，校准在 calibration 上拟合，test 上只计算已冻结设置。阈值或校准会改变 score 时，以完整已冻结流程在 validation 再确定工作点，不能通过 test 扫描最大 FOM。分类 `argmax` 与多类物理 selection thresholds 是两个结果，若多个阈值都通过须预先固定归类/拒判规则。

### 7.3 统计与系统效应

每个模型保留同一测试事件的 probabilities，按 generator-event family 做 paired bootstrap；有 production/chunk 相关时采用相应 block/stratified bootstrap。分别报告三训练种子的 spread 和事件统计区间，不把 seeds 当作独立物理事件。加权事件给 bootstrap/finite-MC 方差与 selected-sample `Neff`；极低背景工作点若有效通过事件少于约 25，标明统计不足并给区间，而不宣称稳定 rejection。

零背景通过不能报告无限 rejection 或发现：未加权独立样本可给 exact binomial/Poisson upper bound（零通过时 95% 上限约 `3/Nbackground`）；加权 MC 须采用与权重分布一致的 finite-MC likelihood/区间，不直接把 `Neff` 填入 exact binomial。样本不足则扩大独立 test MC 或降低工作点。

有物理 exposure 和 nuisance 约束后再计算 `S/√(S+B)`、Asimov `ZA` 和含 background/efficiency nuisance 的 profile-likelihood sensitivity。没有这类权重时，只能给明确声明 prior 的效率/纯度场景扫描，不产出带真实 luminosity 含义的 significance。τ 搜索与四类截面提取的最佳阈值可能不同；后者还需 score/energy migration matrix、acceptance 与 unfolding/forward-folding 验证。

S6 用相同 latent generator events 的 electronics/conditions variants 检查 gain、noise、threshold、time-window/resolution、dead channels、alignment、field、material；同时保留一个条件变化未参与训练的外部测试。generator stress 尽可能固定入射 flavour/energy、vertex、target，比较 GENIE tune/FSI/hadronization 或兼容 TeV 区域的 alternative generator；匹配不代表覆盖所有系统效应。charm/τ decays 单独检查 branching/polarization、secondary production 与实现版本。

## 8. 失败诊断必须产生下一项可检验问题

按完整 confusion pair 分层抽样，包含高置信正确、高置信错误、边界和随机事件，画原生 pixel、bar views、FASER measured tracks、predicted/truth annotations 的独立图层。通过子系统关闭、局部 hit/channel occlusion、saliency sanity checks、shuffle tests 和 response perturbation 来检验因果线索；attention heatmap 本身不作为物理解释。

例如 `τ→μ→νμ CC` 混淆，依次量化 secondary displacement 是否落在可测层间、daughter track 是否进入 FASER、hadronic activity 是否重叠、真实和预测不平衡是否可测；同时检查 ID/label、mask、sampler、normalization 和目标定义。给出每组事件数、权重、误差与图例。少数反常事件进入待查清单；只有在总体统计与可见性证据支持时才调整路线或探测器方案。
