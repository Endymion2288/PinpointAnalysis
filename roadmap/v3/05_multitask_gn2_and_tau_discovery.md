# 05 多任务 GN2 密集监督与双机制 $\nu_\tau$ 低本底发现（Phase V3 & V4）

**设计核心**：在已经完成物理坐标系重构（`V2 PinpointFormer`）的基础上，把训练期能够触及的高精真值信息通过 **ATLAS `GN2` + FASERCal 同方差多任务损失** 全部转化为驱动微米表征学习的强梯度信号；再针对下游 $\nu_\tau$ 发现物理需求，融合纯闪烁体区运动学与极端低 FPR 尾部优化，推出可解释的似然比与多域模板分析框架。

---

## 1. Phase V3：事件级同方差不确定度加权联合学习（`src/analysis/flavour/v3/v3_multitask/losses.py`）

参照 FASERCal 论文（`arXiv:2604.07037v2` Eq. 8）与 Kendall & Gal 多任务不确定度理论，引入可学习的任务对数方差向量 $u_m = \ln \sigma_m^2$（严格裁剪于 $[-4.0, +4.0]$ 防止某项损失坍缩到零或爆掉梯度）：

$$\mathcal{L}_{\text{event}} = \sum_{m \in \{\text{proc},\, \text{charm},\, \text{vtx},\, \text{kin}\}} \left[ \frac{1}{2}\exp(-u_m)\,\mathcal{L}_m + \frac{1}{2} u_m \right]$$

四个联合监督头的精确定义如下：

### 1.1 7 类物理精细过程分类头 $\mathcal{L}_{\text{proc}}$
将 `T4` 四类扩展为探测器拓扑与电荷物理一致的 **7 个精细过程子类**（`logits` 维度 $B \times 7$）：
`0: e_CC`, `1: mu_minus_CC`, `2: mu_plus_CC`, `3: tau_e`, `4: tau_mu`, `5: tau_had`, `6: NC`。
- **物理动机**：$\tau \to \mu\nu\bar{\nu}$ 与 $\tau \to \text{hadrons}+\nu$ 具有截然相反的下游磁谱仪响应；$\nu_\mu$ 与 $\bar{\nu}_\mu$ 在矢高 $s_y$ 上正负号相反；强行合并到一个类会迫使分类线性超平面互相妥协。
- **未知 $\tau$ 子道无丢失损失**：对于 `label_4 == 2` 但真值直接衰变分支未定（`label_7 == -1`，如 `stage236` 中 40/65,558 例）的事例，绝不丢弃或错误归类，而是严格计算**边际概率对数似然**：
  $$\mathcal{L}_{\text{proc}}(\text{unknown }\tau) = -\ln \left( p_3 + p_4 + p_5 + 10^{-8} \right)$$
- **推理口径严格对齐**：推理输出统一计算精确边际四类概率：
  $$P(e) = p_0,\quad P(\mu) = p_1 + p_2,\quad P(\tau) = p_3 + p_4 + p_5,\quad P(\text{NC}) = p_6$$
  其中在仅非 $\tau$ 事件上的条件三类分布仍保持 $q_k = \frac{P(k)}{1 - P(\tau)}$，与 `R1–R3` 的 `T4 conditional` 评估无缝对齐！

### 1.2 显式 Charm 产生对抗识别头 $\mathcal{L}_{\text{charm}}$
- 输出单一 logit 预测 `charm_flag`（采用温和平衡权重或 `FocalLoss(gamma=1.5)`，仅在 `charm_valid == True` 事件计算）。
- **物理意义**：`CC νμ` 或 `CC νe` 伴随产生的 $D^\pm / D^0 / D_s / \Lambda_c$（寿命 $c\tau \sim 60\text{–}312\,\mu\text{m}$）会产生与 $\tau$ 衰变（$c\tau \approx 87\,\mu\text{m}$）极易混淆的次级顶点；让网络显式建立 Charm 表征是把 `1% / 0.1% FPR` 下 $\nu_\tau$ 纯度推高的先决条件。

### 1.3 三维主顶点回归头 $\mathcal{L}_{\text{vtx}}$
- 与 `V2` 的 Stage-1 顶点锚点共享参数：输入 `vertex_valid == True` 事件上的真值 `vertex_xyz_mm`，损失为按物理轴分辨率归一化（$x/y$ 尺度 `1.0 mm`，$z$ 尺度 `10.0 mm`）的 **异方差高斯对数似然 + SmoothL1**：
  $$\mathcal{L}_{\text{vtx}} = \frac{1}{3}\sum_{a \in \{x,y,z\}} \left[ \frac{1}{2}\exp(-\hat{\ell}_{v,a}) \cdot \text{SmoothL1}\left(\frac{\hat{v}_a - v_a}{s_a}\right) + \frac{1}{2}\hat{\ell}_{v,a} \right]$$

### 1.4 柱坐标可见运动学头 $\mathcal{L}_{\text{kin}}$
- 在 `kin_valid == True` 事例上用 `SmoothL1` 回归 4 个量：`[log1p(pz_vis_GeV), log1p(pt_vis_GeV), cos_dphi_lh, sin_dphi_lh]`，强迫全局融合特征保留轻子-强子反冲平衡关系，直接赋能下文 Phase V4 的纯闪烁体区运动学判别。

---

## 2. 像素簇与通道级 `GN2 + FASERCal` 密集辅助头（`src/analysis/flavour/v3/v3_multitask/dense_heads.py`）

> 所有本节头模块**仅在训练阶段依赖 `truth_sidecar` 计算梯度；在 `V3EvalDataset` 与部署推理期全部作为纯前向物理特征提取器运行，零真值输入。**

### 2.1 全探测器 `Rel` 粒子组分头 $\mathcal{L}_{\text{rel}}$
- 对**每个近顶点/全事件像素簇嵌入** $\mathbf{u}_i$ 以及 **立体量能 Nexus / 闪烁体条节点嵌入** $\mathbf{w}_m$，预测 3 组分能量/击中比例 $(f_{\text{EM}}, f_\mu, f_{\text{had}})$，采用软目标交叉熵 $-\sum_c y_c \ln \hat{f}_c$。
- **纠偏点**：覆盖硅像素层后，网络从前两层硅层像素簇特征就能明确分出“早期成对电子簇射（EM 占比高）”与“纯净单强子/$\mu$ 径迹”。

### 2.2 ATLAS `GN2` 像素簇 5 类物理起源头 $\mathcal{L}_{\text{origin}}$
- 对每个近顶点像素簇（$K_{\text{vtx}} \le 128$）预测 5 类起源分布：
  `0: Primary_lepton`、`1: Primary_hadron`、`2: Tau_daughter`、`3: Charm_daughter`、`4: Secondary_shower`。
- 因为 `Tau_daughter` 与 `Charm_daughter` 簇数量显著少于海量簇射噪点 `Secondary_shower`，使用**平方根反比类频率温和加权** $w_c \propto N_c^{-0.4}$（截断至 `[0.5, 5.0]`），保证每一次训练迭代都把清晰梯度精准送达携带次级折角的那 $1\text{–}4$ 个像素簇 token 上！

### 2.3 同径迹与同次级顶点对偶亲和头 $\mathcal{L}_{\text{compat}}$
- 取近顶点前 $K_{\text{pair}} \le 64$ 个像素簇嵌入 $\mathbf{u}_i$，经线性投影 $\mathbf{e}_i = \text{Norm}(\text{Linear}_{192\to 64}(\mathbf{u}_i))$ 与缩放对称内积得对数几率 $M_{ij} = \tau_{\text{aff}}\,\mathbf{e}_i \cdot \mathbf{e}_j$（不同层 $z_j \ne z_i$ 且两者至少一者属于真实原初/$\tau$/Charm 带电径迹的有效簇对）。
- 监督标签 $Y^{\text{trk}}_{ij} \in \{0, 1\}$ 为两簇拥有相同真值 `dominant_track_id`；另设一轻量分支 $M^{\text{vtx}}_{ij}$ 监督“两簇是否均属于同一非主顶点次级衰变簇（$\tau$ 或 Charm）”。采用对正负样本动态平衡的二元交叉熵。

### 2.4 推理期“隐式次级顶点”自预测摘要回流（Self-Predicted Topological Summary）
**完全不需要手写规则追踪，但能向事件分类头提供与显式顶点同等可解释的物理依据**：
在网络末端分类层前，用模型**自己前向预测出的**簇起源概率分布 $\hat{q}_i(c)$（$c \in \{0,\dots,4\}$）与主顶点估计 $\hat{\mathbf{v}}$ 无真值计算出 8 维可微物理聚合摘要并送入最终 7 类分类头：
1. 近顶点所有簇中 $\hat{q}_i(\text{Tau\_daughter})$ 的最大值与 Top-3 均值（`2` 维）；
2. 近顶点簇中 $\hat{q}_i(\text{Primary\_lepton})$ 的最大值（检测强瞬发 $e/\mu$，直接压低偽 $\tau$ 误报）（`1` 维）；
3. $\hat{q}_i(\text{Charm\_daughter})$ 的最大值与均值（检测并发 Charm 背景）（`2` 维）；
4. 以 $\hat{q}_i(\text{Tau\_daughter})$ 为权重的平均径向距离与平均拟合极角散度（`2` 维）；
5. 由模型自预测高置信径迹簇亲和图 `sigmoid(M_ij) > 0.7` 反推出的高概率次级径迹计数软和 $\sum_{i<j} \sigma(M^{\text{vtx}}_{ij}) / 10.0$（`1` 维）。

多任务总损失：
$$\mathcal{L}_{\text{V3}} = \mathcal{L}_{\text{event}} + 0.35\,\mathcal{L}_{\text{rel}} + 0.50\,\mathcal{L}_{\text{origin}} + 0.30\,\mathcal{L}_{\text{compat}}$$
（以上三项密集系数均已归一至单位均值交叉熵尺度，`W3-08` 与 `W3-09` 分两步消融验收各自净增益）。

---

## 3. Phase V4：双机制 $\nu_\tau$ 发现物理协同与尾部 Partial-AUC 微调

### 3.1 `D_scint_contained` 纯闪烁体区 NOMAD 横向运动学融合（`src/analysis/flavour/v3/v4_tau/nomad_kinematics.py`）
对于落在像素区之外或纯闪烁体区的事例（占 `large_scintillator` 约 79%），次级顶点物理不可见。此时按照 NOMAD 实验（Astier et al., *Phys. Lett. B* 2001）与合作组 [FORTVNE_Update_kinematic_variables_2026-10-07.pdf](../v2/05_literature.md) 思路，从纯粹可观测三维量能 Nexus 节点与磁谱仪先导径迹构造 **10 维无真值平面运动学特征**注入最终融合头：
1. 从量能器三维前段最密集主核心估算主电磁/轻子轴单位横向向量 $\hat{\mathbf{u}}_{T,\ell}$ 与余下强子反冲簇射（剔除最前导或磁谱仪匹配径迹外推廊道后的三维量能能量矩）横向动量方向向量 $\mathbf{Q}_{T,\text{had}}$；
2. 计算横向共面度角偏离 $\cos\Delta\phi_{\ell,\text{had}}$、$|\pi - \Delta\phi_{\ell,\text{had}}|$、横向可见动量不平衡量 $|\mathbf{Q}_{T,\ell} + \mathbf{Q}_{T,\text{had}}| / E_{\text{vis}}$；
3. 先导高能量孤立比（核心 $3\,\text{cm}$ 内能量 / 总可见能量）及横向双主轴宽度比（$\sigma_{\text{major}}/\sigma_{\text{minor}}$，对应 $\tau$ 衰变缺失动量引起的横向平面不对称性）；
4. **谱仪动量 vs 量能器能量比** $\hat{p}_{\mu,\text{spec}} / \max(1.0, E_{\text{cal,MeV}}/1000)$ 与谱仪方向同 $\mathbf{Q}_{T,\text{had}}$ 的反冲夹角：用于抓取 $\tau \to \mu\nu\bar{\nu}$ 两颗逃逸中微子造成的软化特征。

### 3.2 针对 `FPR <= 1% / 0.1%` 工作点的 DRO Partial-AUC 尾部微调（`src/analysis/flavour/v3/v4_tau/pauc_finetune.py`）
标准交叉熵主要受占主导的普通样本支配；当目标是 `1% FPR` 下的 $\tau$ 效率时，微小的 `0.5%` 边缘极端本底足以毁掉工作点。
- **操作方案（不破坏主概率校准）**：
  1. 首先保存多任务充分收敛检查点作为主 `BA3 / BA4 / NLL` 基准。
  2. 基于该权重用较小学习率（主学习率的 `0.15x`）继续微调 **`5 passes`**：保留 `0.5 * L_V3` 锚定整体校准，叠加 Zhu et al. (ICML 2022) **DRO / CVaR One-Way Partial AUC** 可微损失，设评分函数 $s_i = \ln\frac{p_3+p_4+p_5}{p_0+p_1+p_2+p_6+10^{-8}}$，截断窗口 $\beta = 0.01$（即聚焦得分最高的前 `1%` 本底难例）：
     $$\mathcal{L}_{\text{pAUC@}\beta} = \tau_q + \frac{1}{\beta \cdot |B^-| \cdot |B^+|} \sum_{i \in B^+_{\tau}} \sum_{j \in B^-_{\text{non-}\tau}} \max\Big(0,\; s_j - s_i + m_{\text{gap}} - \tau_q\Big)^2$$
     其中 $\tau_q$ 为随批次更新的可微上 $1\%$ 分位数辅助变量（安全边距 $m_{\text{gap}} = 0.6$），且在采样器上维持每批充足的非 $\tau$ 样本量（`batch_size >= 32`，避免小批统计波动）。
  3. 微调后必须在 `Tune`（使用 `Tune` 交叉校准与同能谱切片 `Tune_E_le_2p3TeV`）同时检验：① `1% / 0.1% FPR` 下 $\tau$ 效率净增；② `BA3`、`BA4` 与三类召回率满足非劣约束（$\Delta\text{BA3} \ge -0.5\,\text{pp}$）。

### 3.3 严格设限的条件自监督预训练门禁（`V4.5`，默认暂缓）
吸取 `S4–S6` 仅投入 3 epoch 预训练却耗费巨大机时的教训，v3 明确把无监督掩码预训练（MAE / Mask-Cluster + `Rel`）设定为**后置触发项**——仅当全部满足以下三个条件才开启：
1. 纯监督的 `PinpointFormer-GN2` 已确立明确局部特征能力（在 `Pixel-centered` 域相对 `B0` $\Delta\text{BA4} \ge +4.0\,\text{pp}$ 且微米消融探针通过）；
2. 存在至少 `300k–1M` **位于基准包容区内（排除 `D_empty` 与极端边缘事例）** 的训练流且预算允许至少 `30 epochs` 大批（有效 batch $\ge 256$）预训练；
3. 配备同总墙钟时间（Matched Wall-Time）的强监督对照试验；若在 `100k` 或 `300k` 上无超越同预算监督训练的显著净增，立即止损直接交付纯监督多任务最佳模型进入 `V5 Confirm`。
