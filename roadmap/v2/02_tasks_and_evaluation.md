# 类别选择、数据划分与评价合同

**先交付实际重训的三类模型，同时保留四类模型和 τ specialist。类别数由同事件、同输入、同计算的比较决定；更多类别只有在保留共同任务质量并改善 τ 工作点时晋级。**

本文效果门槛是下一轮的设计值，样本分配已由 R0 冻结；各阶段实际完成情况由 R0–R7 报告记录。已有 detector-proxy 与样本接受状态继续有效。

## 1. 四种任务的精确定义

| ID | Outputs | Training labels | Required evaluation |
| --- | --- | --- | --- |
| T3 | `e CC, mu CC, NC` | 非 τ 三类；实际重新初始化/训练 | 同一非 τ 样本上的三类指标；单列 τ 混入后的污染 |
| T4 | `e CC, mu CC, tau CC, NC` | 当前四类显式标签映射 | 四类指标、τ 工作点及相同非 τ 样本的条件三类指标 |
| T3T | `tau probability` + `conditional e/mu/NC` | 全事件训练 τ gate；仅非 τ 事件训练条件三类头 | 与 T4 相同四类输出和全部指标；真实三类头独立质量 |
| T4D | T4 + `tau→e/mu/had` conditional head | 四类主监督；仅分支有效的真 τ 监督条件 decay 头 | 合并 τ 指标、各分支、非 τ contamination；不得用真分支路由推理 |

T3T 的概率为：

```text
t = sigmoid(tau_logit)
r = softmax(three_logits)
p4 = [(1-t)*r_e, (1-t)*r_mu, t, (1-t)*r_NC]
L = BCE(t, is_tau) + 1[not_tau] * CE(r, three_label)
```

在统一事件权重和一致归一化下，这是四类对数似然的一种分解；表达能力并不天然超过 flat4。它的研究价值在于分离头、任务调度和针对 τ 的局部信息路径。追加任务权重后优化目标会改变，必须单独校准和对照。

T4D 保留四类 loss，新增 `L_decay` 只在有效 τ labels 上计算。若采用分解输出，令 `p(tau_i)=p(tau)*u_i`，其中 `u=softmax(decay_logits)`，再把三个 τ 概率合并回四类。分支 unknown 仍训练已知的 CC τ 主类别，只 mask 分支 loss。

S5 已测过 flat6，保留其结果作为控制。新条件头与原 flat6 的区别必须分别消融：层级分解、采样权重、损失权重、局部特征输入，不能一次改变后全部归因于“更多类别”。ν/anti-ν、charm、DIS/QE/RES、target isotope 优先作为条件诊断/辅助标签；当前不增加到主物理类别中。

## 2. 三类与四类的公平比较

三类主比较只使用相同的真 e/μ/NC 事件；四类模型先去掉 τ 概率再重新归一化。报告 NLL3、BA3、macro AP3/AUROC3、逐类 ROC/PR、1%/0.1% FPR 效率、固定 80%/90% benchmark purity 的效率与覆盖率。三类 NLL 的无信息参照是 `ln(3)`，四类是 `ln(4)`，两者不直接排名。

训练预算保留两种口径：

1. **共同三类事件与曝光匹配**：T3 使用 T4 的全部非 τ train rows，三类 head 的非 τ exposure 一致；T4 新增的 τ 事件及计算单列。
2. **总计算匹配**：给 T3/T4 相同设备、调参额度与训练计算上限；额外三类 unique events 和曝光记录清楚。这回答相同资源怎样分配更有用。

第一种用于研究任务干扰，第二种用于选实际交付方案。不能让 T3 无限多训后把改善全部解释成去除 τ。

三类模型进入包含 τ 的样本时，输出仍是三类条件分数，不存在物理上正确的 τ→NC 重标注。报告 τ→e/μ/NC 的分配/接受率及由此造成的三类样本污染；部署的 reject 使用观测分数/不确定度规则，不能用真 τ 标记先行剔除。T3 与 T3T/T4 的差异由这个开放输入条件检验。

## 3. 新开发集与独立确认集

旧 S6 test 已公开，作为历史 reference。新模型设计、超参数与阈值只依赖 v2 开发集；最终确认需新的独立事件。

R0 从既有 accepted manifest 和所有历史 train/SSL/validation/probe/manual-event 清单，按 accepted `family_hi/family_lo` 键生成使用记录表。能够用于新确认的事件应未进入任何模型拟合、预训练、选点、方向/对象 probe 或供设计使用的详细事例诊断。同 family 的不同响应、重复采样和再模拟变体进入同一 partition。该步骤核对**研究使用情况**，无需重新认证原生产来源。

优先从旧 train 和未使用的旧 validation 余量中分配；原 test、已拟合 temperature 的旧 calibration 均不复用为新独立确认。曾用于全库 schema/数量盘点、但没有进入模型开发的事件，不因此自动失去可用性。若未使用事件不足，则新生成独立 hard scatters，并复用 accepted response；新确认完成前可以继续完成 R1–R6 的开发实验。

下列四类平衡分配已由 R0 冻结，实际事件与历史隔离检查见 [stage238](../../workbook/2026-11-14_stage238_r0_partitions_and_pipeline.md)。

| Partition | Frozen size | Role | Access rule |
| --- | ---: | --- | --- |
| Train pool | 100k / 300k / 1M nested | 监督、局部监督、train-only SSL | 排除下列所有 family |
| Tune | 40k = 10k/class | HPO、checkpoint 与候选选择 | 重复开发使用，统计按开发结果表述 |
| Cal-A | 40k = 10k/class | 候选冻结后拟合 temperature | 每个已冻结候选拟合一次 |
| Cut-B | 160k = 40k/class | 固定 1%/0.1% 背景阈值 | 与 Cal-A 不交叉，不用于模型排名 |
| Confirm | 200k = 50k/class | 最终独立比较 | 模型、calibrator、prior、cuts、主假设冻结后开放 |

分类 label 预算只计 Train pool 内真正有监督的 unique events；SSL 语料单列。Cut-B 对每个 one-versus-rest 任务有已冻结的 120k 背景事件；0.1% 目标对应约 120 个背景通过，这是期望计数，实际通过数、权重和 Neff 尚未测量。其余 raw 分区人数已核对，Confirm 的类别人数只使用 R0 构造时的平衡检查，不重新读取封存标签。

所有输入模型共用 manifests；所有归一化、簇阈值学习、监督分量尺度、负样本选择均在 Train 内确定。Tune/Cal-A/Cut-B/Confirm 不得与任一候选使用的初始化权重的历史训练或预训练 family 相交；旧 MAE 初始化也要经过这个使用记录核对。对同一个真实事件进行 readout/transport perturbation 时，只改变响应，继续保留其 family 和 partition。

开发期的模型排名使用 Tune 上预定的 ROC/FPR 指标，阈值只作描述性开发统计；需要验证 threshold 稳定性时在 Tune 内交叉拟合。Cal-A/Cut-B 留到候选冻结之后使用，不能一边读取它们的工作点结果一边继续挑模型。最终尾部推断只来自冻结 cuts 和 Confirm。

## 4. 指标层级与工作点

### 主任务

三类交付以 BA3 为主指标，NLL3 和逐类 fixed-FPR 效率为约束；四类交付保留 BA4、NLL4、macro AP4/AUROC4。τ 研究以 **inclusive τ efficiency at frozen 1% background acceptance** 为主要选择目标，同时给实际 FPR、τ purity 和 raw counts。0.1% 为预定次要目标，统计足够后再深入更低尾部。

每个 τ specialist 都须给两种分母：全体 CC τ 的 inclusive efficiency，以及真分支条件效率。分支选择只用分数决定；τ→e/μ/had 的分支概率或重建形态供模型推理使用，truth 仅用于训练和事后分层。专门 τhad 模型的效率不能直接标为 inclusive τ 效率。

每个类的阈值在 Cut-B 选择，完整 ties 不插值成虚构通过事件。Confirm 使用冻结 cut，同时报告实际 FPR；可补充固定 FPR 的描述性 ROC，但不能据此重新挑部署阈值。多 specialist 的 OR/融合必须在开发集定义，再整体选择背景阈值，不能把每个头的 1% FPR 相加后仍称总体 1%。

### 条件域

至少包括 all accepted、active、empty、target envelope、pixel-centered、large scintillator、tracker active/empty、occupancy、energy、vertex xyz、target isotope、ν/anti-ν、τ branch、τ decay/material/首面投影、charm presence 和对象 quality。域选择只在分析中读取 truth；推理 gating 使用可测量的 presence、occupancy、质量与预测 vertex。

主模型在全域训练并保留全部分母。区域专家允许在观测 proxy 定义的域训练和路由；历史 truth fiducial 训练可作诊断，不能在实际数据中用未知真 vertex 路由。center 域表现好不等于全域效率高。

energy/vertex/exact target matching 用来回答固定入射条件下的模型差异；同时保留 nominal 结果。occupancy 与 detector presence 可能是物理信号或接受度机制，不能在匹配中一概抹除后称剩余部分才是有效信息。新增 occupancy 条件控制单独解释。

## 5. 先验、尾部统计与物理目标

平衡 benchmark、production sampling 和稀有 τ 情景沿用明确的四类 prior 向量。AP、purity、NLL 都随混合改变；AUC 的背景组成也会改变。prior 调整依据 label-shift 假设，不能修复 `p(x|y)` 的 detector/generator 变化。

若 τ prior 为 π、效率为 ε、非 τ 总接受率为 f，则：

```text
purity = pi * epsilon / (pi * epsilon + (1-pi) * f)
```

例如 π=0.001、ε=0.05，要达到 purity=0.5，需要 f≈5.0×10⁻⁵；预计通过 100 个背景需约 2M 独立背景事件。这是样本规模估算，不是已测性能。因此本轮先优化 1%/0.1% 工作点；更低背景工作点按实际暴露、背景组成和 MC 支持另行设计。

有真实 flux、cross-section、target exposure 和数据驱动 background constraints 后，才增加含 nuisance 的 profile-likelihood sensitivity 或 discovery significance。当前稀有 prior 是情景分析。可在开发期预先优化多个 FPR 区间，避免只为 balanced AUC 优化。

## 6. 统计设计与晋级门槛

保留 accepted-family 连接的物理块重采样，同一配对使用同一组 resamples。screen 默认 seed42；只将前两名和共同基线扩到 seeds17/42/2026。三 seed 各自差值与 sample SD 单列；事件 bootstrap 不解释为训练种子总体区间。若训练波动和目标增益相当，再增加 seeds，先 profile 和缩小候选。

确认阶段主比较最多两个：最佳三类候选对共同三类基线、最佳 τ 候选对共同 T4 基线。预定主要检验采用 Holm 控制这两个比较的 family-wise error；其余指标和分层给 pointwise 区间并作为解释。400 次 bootstrap 复用开发流程；最终区间使用 2,000 次，先在实际数据上验证数值稳定性。Cal-A/Cut-B 的分组重采样重新拟合温度/阈值，以补充其估计不确定度；另保留条件于冻结 cuts 的性能区间。

建议的研究晋级值如下，须在新 Tune 开放前写入执行合同：

| Decision | Practical effect target | Statistical / quality constraint |
| --- | --- | --- |
| Promote T3 candidate | ΔBA3 ≥ +0.020（2 个百分点） | 主要差值 95% 下界>0；NLL3 增量上界≤+0.010；各类 recall 退化上界≤0.010 |
| Promote tau candidate | 1% 工作点 Δετ ≥ +0.010（1 个百分点） | 主要差值下界>0；BA3 下降上界≤0.005；NLL4 增量上界≤+0.010；实际 FPR 同时交付 |
| Promote native detail | 至少一个预定任务达到上述实际效果，或相同性能显著降低成本 | 相同输入/训练协议；native/coarse 三 seed 配对；全域分母不缩小 |
| Promote object path | 在可复核 fake–efficiency 曲线上改善对象质量，并改善事件任务 | classification 增益须包括重建失败事件，不只评价 matched objects |
| Promote additional subclasses | 改善合并后的四类/τ 目标 | 三类非劣；分支 labels/mixture 明确；新增成本纳入比较 |

这些是资源分配与晋级标准，未达到时保留精确效应和区间。小于实际效果目标的正增益可作为方法结果记录，但不必立即触发更昂贵的模型扩展。某一配方失败后，应先核查学习健康、输入尺度与收敛；有效学习仍无收益时，暂缓该配方并记录其适用范围。

## 7. 三类或四类的最终选择

- R1 先形成可发布给合作组讨论的三类基线表及同输入 raw/image/object 对照，无需等待 τ 分支成功。
- 若 T4/T3T 的共同三类性能达到非劣要求，保留四类概率作为主研究输出；τ 工作点较弱时，将四类结果与三类交付分别说明。
- 若 τ specialist 达到晋级标准并在新确认集成立，四类与分支专家继续投入；若没有达到，三类作为近期交付，τ 保留明确的分支/几何/统计诊断和下一触发条件。
- 更多类别的下一触发条件是局部可分辨性或条件头实测收益，不能仅因 flat4 训练困难就增加类别。3/4/6 类都不会自动使 raw detector 信息增多。
