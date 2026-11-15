# 01 前序研究取证分析与文献方法映射

**核心结论**：通过逐行代码核查 `src/analysis/flavour/` 与 `roadmap/v2/`、`workbook/` 全部关键记录确认——PINPOINT 数据流水线在 `R0–R3` 没有发生标签泄漏或梯度完全断流，模型学到了基于全局能量矩和下游谱仪存在性的弱信号，但**多处底层几何表征错误、注意力数学核缺陷、混杂全域目标阈值与严重欠收敛叠加在一起，彻底屏蔽了 20.8×22.8 μm 硅像素探测器与正交闪烁体的核心物理分辨率。**

---

## 1. 实测性能基线全景核对（`S6` 及 `R1–R3`）

以下数值全部直接摘自冻结产物 [r1_results.json](../v2/r1_results.json)、[r2_results.json](../v2/r2_results.json)、[r3_results.json](../v2/r3_results.json) 与 [stage236](../../workbook/2026-11-14_stage236_s6_final_results.md)（其中 `R1/R2/R3` 均使用 `Tune40k` 开发集，`All` 含四类各 `10,000` 事例；`Pixel-centered` 为 `[930, 1050, 757, 1028]` 合计 `3,765` 事例、`3,008` 非 $\tau$ 事例；`Large scintillator` 为 `[4424, 4764, 4136, 4678]` 合计 `18,002` 事例、`13,866` 非 $\tau$ 事例）。

| 模型 / 阶段 | 输入与核心特性 | `All` `BA3` | `Pixel-centered` `BA3` | `Large` `BA3` | `All` `BA4` | `Pixel-centered` `BA4` | `All` $\tau$ @1% | `Pixel-centered` $\tau$ @1% |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `R1 B0` (HGB) | 41 维全局标量，**零击中空间坐标** | `54.59%` | `72.77%` | `66.20%` | `43.10%` | `55.92%` | `2.83%` | `4.36%` |
| `R1 B2` (ParticleNet) | 原生通道 kNN 点云 (`dim=64`, `4 passes`) | `55.12%` | `72.97%` | `67.01%` | `43.45%` | `55.75%` | `2.82%` | `2.51%` |
| `R1 B3` (ParT) | 原生通道 Pair-Attention (`4 passes`) | `55.06%` | `72.76%` | `66.81%` | `43.28%` | `55.09%` | `2.70%` | `1.59%` |
| `R1 B4` (Hierarchical) | `PlaneModuleClassifierV2` Scratch | `54.35%` | `72.94%` | `66.16%` | `42.46%` | `55.08%` | `2.69%` | `1.72%` |
| `R1 B4-MAE` | 同 `B4` + S4 预训练微调 | — | — | — | `43.03%` | `55.62%` | `2.41%` | `2.38%` |
| `R1 B1` (`100k×4`, s42) | `1 mm` XZ/YZ ResNet + TrackerSet | `56.62%` | `73.82%` | `68.89%` | `44.62%` | `56.90%` | `5.09%` | `6.87%` |
| `R2 B1-D2` (`300k×4`, s42) | 同 `B1` 扩大唯一训练样本至 300k | `57.10%` | — | — | `45.77%` | — | `6.05%` | — |
| `R3 M0` (`300k×4`, s42) | 同 `B1` 改用全池 Epoch 混洗 | `57.51%` | — | — | `46.03%` | — | `6.03%` | — |
| `R3 G2 control / rel` | 压缩前原生分支控制与关系层 (`100k×4`) | `55.61% / 55.59%` | — | — | `43.44% / 43.47%` | — | `3.14% / 2.69%` | — |
| `S6 MAE3 ensemble` | `B4` 三种子集成（72.4 万独立 Test） | — | — | — | `43.52%` | `56.19%` | `3.56%` | **`0.35%` (全局阈值崩溃)** |

从上表按类别召回率进一步解算（以 `R1 B1 T3 seed=42` 为例）：
- `All`（`30,000` 非 $\tau$）：$[e, \mu, \text{NC}]$ recall = `[63.01%, 49.98%, 56.87%]`，`BA3 = 56.62%`。
- `Large scintillator`（`13,866 / 30,000 = 46.22%`）：$[e, \mu, \text{NC}]$ recall = `[74.71%, 73.87%, 58.10%]`，`BA3 = 68.89%`。
- **其余 `53.78%` 非 `Large scintillator` 事例（`16,134` 例）**：精确反推 $[e, \mu, \text{NC}]$ recall 分别为 **`[53.73%, 28.25%, 55.79%]`，`BA3` 仅 `45.92%`**（`B0` 在同一靶外子集仅 `44.74%`）。

**两条决定性物理诊断证据**：
1. 在包含高分辨像素顶点的 `Pixel-centered` 域，所有细粒度模型（`B2`, `B3`, `B4`, `B4-MAE`）的 `BA4`（`55.08%–55.75%`）和 `1% FPR` $\tau$ 效率（`1.59%–2.51%`）**全部劣于完全不看像素空间坐标的 41 维树模型 `B0`（`55.92%`, `4.36%`）**，证明原生特征链路几乎未提取到亚百微米级拓扑信息。
2. 超过一半的训练集与评价集事例发生在大基准体积之外，其 `μ` 召回甚至跌至 `28.25%`（低于三类随机盲猜 `33.33%`），与中心包容区物理性质完全不在同一分布流形。

---

## 2. 七大代码级与物理级根因详证

### 根因一：`All` 域混训与无条件单一全局阈值诱发跨区域概率反转
- **实测证据**：[stage236](../../workbook/2026-11-14_stage236_s6_final_results.md) 记录 `S6` 独立测试集上 `Target envelope`（`37,457 / 65,558 = 57.1%` $\tau$）在全局 `1% FPR` 阈值 `p(τ) > 0.4467` 下效率为 `6.13%`，而 `BA4` 最高的 `Pixel-centered`（`4,882` 例 $\tau$）仅有 `17` 个通过，效率暴跌 17 倍至 **`0.35%`**。
- **数学机制**：在未包容边缘/下游区，出射粒子携带部分信息逃逸，四类后验概率坍缩到高熵模糊区 $(0.25, 0.25, 0.25, 0.25)$，大量边区或纯强子伪事件带有 $p(\tau) \in [0.25, 0.48]$；相反在 `Pixel-centered` 全包容区，当前模型无法分辨 $\tau$ 衰变的亚毫米折角，$\tau \to e/\mu/\text{had}$ 在模型眼中高度接近干净的瞬发 `CC νe / CC νμ / NC`，于是模型给出极高置信度 $p(e/\mu/\text{NC}) \in [0.80, 0.96]$，反向把 $p(\tau)$ 压到 `< 0.35`。用包含所有区域的单一绝对数值阈值切分，必然把**高信号密度黄金区整片切空**。
- **能谱截断混杂**：[stage04](../../workbook/2026-11-11_stage04_flavour_inventory.md) 记录现有 `10083` ($\nu_\tau$) MC 入射中微子能量上限为 $E_\nu \le 2.3118\,\text{TeV}$，而其余三类均扩展至 $5.0412\,\text{TeV}$，且横向顶点分布不重合。不加控制地以总沉积能量直接分类，存在学入“高总能 $\Rightarrow$ 否定 $\nu_\tau$”模拟伪特征的风险。

### 根因二：`B4 / S3 / S4 / S6` 注意力温度缺失导致的 $O(1/N)$ 弥散死锁
- **代码证据**：[models_repair.py](../../src/analysis/flavour/s3/models_repair.py#L46-L53) 第 49 行：
  ```python
  q, k = F.layer_norm(q, (self.width,)), F.layer_norm(k, (self.width,))
  ```
- **数学证明**：`F.layer_norm` 对末维（`width = dim // heads = 24`）强制标准化且**无仿射尺度或可学习温度参数 $\gamma$**，从而恒有 $\|q\|_2 = \|k\|_2 = \sqrt{\text{width}}$。由 Cauchy-Schwarz 不等式，SDPA 除以 $\sqrt{\text{width}}$ 后的注意力 logits 严格受限于：
  $$z_{ij} = \frac{q_i \cdot k_j}{\sqrt{\text{width}}} = \sqrt{\text{width}} \cdot \cos\angle(q_i, k_j) \in [-\sqrt{24},\, +\sqrt{24}] = [-4.90,\, +4.90]$$
  若一个高能簇射事例有 $N = 1000$ 个有效区域或局部 token，而 $\tau$ 折角关键特征只存在于 $1$ 个局部 token 上，哪怕其余 $999$ 个背景 token 对该查询的余弦相似度处于零均值中性状态（$z_{ij} = 0$），Softmax 能分配给唯一信号 token 的最大注意力权重上限为：
  $$\alpha_{\max} \le \frac{e^{4.90}}{e^{4.90} + 999 \cdot e^{0}} \approx 11.8\%$$
  **超过 $88\%$ 的权重被数学强行摊平给簇射背景噪声**（对应 Veličković et al., ICML 2025 证明的 Softmax 随长度尖锐度退化定理）。
- **物理数据印证**：物理上 $L_{\tau}^{\text{decay}} = \gamma c\tau = \frac{E_\tau}{m_\tau}\cdot 87.03\,\mu\text{m}$，能量越高、像素击中数越多，次级顶点飞行距离越长、理论越易分辨；然而 [stage236](../../workbook/2026-11-14_stage236_s6_final_results.md) 实测显示，`S6` 模型在像素击中 `1–100` 区间 $\tau$ 效率为 `6.25%`，到 `100–1000` 降至 `3.80%`，到 `1000–10000` 暴跌至 **`0.29%`（下降 21.3 倍）**！代码行为与测试集异常完美闭环。

### 根因三：`B4` 多尺度架构丢失横向 $(x,y)$ 上下文与全局标量捷径饿死主干
- **代码证据**：
  1. [dataset.py](../../src/analysis/flavour/s3/dataset.py#L42-L78) 将原生像素按 $16\times 16$ 像素窗分组（物理尺寸 $0.3328\times 0.3648\,\text{mm}$），单块内部超过 32 个像素再行切分；[models_repair.py](../../src/analysis/flavour/s3/models_repair.py#L198) 又将连续 32 个块强行聚合成仅 2 个区域查询 token，并在生成区域位置编码时只取了 `geo[:, 2:3]`（纵向 $z$），**彻底丢弃区域横向坐标 $(x, y)$**；同时第 126 行在 `multiscale=True` 时设定 `self.context = nn.ModuleList()`（**0 层**区域间显式几何上下文消息传递）。
  2. 第 108、120 行使用 `residual_scale = 0.1` 压低注意力分支残差幅度，却在第 249–259 行将 11 维事件总计数标量 `summary` 与所有池化特征拼接后**直连线性分类旁路 `self.linear_readout`**。
- **实测证据**：[stage170](../../workbook/2026-11-14_stage170_s4_valid_path_results.md) 明确测得，10k 从头监督训练模型的 `Global latent` 置换 $\Delta\text{NLL} = 0.000085$，而 `Response summary` 高达 `0.226352`。浅层全局计数直接提供了一条可无痛下降早期训练 loss 的**线性捷径（Shortcut Learning）**，剥夺了深层几何网络在有限 4 passes 内获得有效反传梯度的机会。

### 根因四：`B2` ParticleNet 与 `B3` ParT 的 `63.9:1` 尺度畸变与正交条“零距离盲区”
- **代码证据**：[representation.py](../../src/analysis/flavour/s2/representation.py#L15) 设 `XYZ_SCALES_MM = (210., 210., 3000.)`，而 [s2/models.py](../../src/analysis/flavour/s2/models.py#L48-L56) 与第 77–79 行定义跨通道距离：
  ```python
  common = geometry[:, :, None, 3:5] * geometry[:, None, :, 3:5]
  transverse = (delta[..., :2].square() * common).sum(-1)
  return transverse + delta[..., 2].square() * 0.05
  ```
- **两重数学破坏**：
  1. **纵向尺度过度塌缩**：令横向距离 $\Delta x = 1\,\text{mm}$，其距离贡献为 $(1/210)^2 = 2.268\times 10^{-5}$；要使 $\Delta z$ 贡献等价距离，需满足 $0.05 \cdot (\Delta z / 3000)^2 = 2.268\times 10^{-5}$，解得 $\Delta z = \mathbf{63.9\,\text{mm}}$。整段约 $1.2\,\text{m}$ 长的钨靶在 kNN 度量空间被压缩成横向仅 $18.8\,\text{mm}$ 厚的薄饼，迫使邻域图强行连接不同簇射深度上的无关击中。
  2. **正交闪烁体条距离恒为零**：$X$ 测量面板的 `measured` 标志为 `[1, 0]`，$Y$ 测量面板为 `[0, 1]`，二者按位点乘 `common = [0, 0]`！因此**任意一根 $X$ 条到相邻层任意一根 $Y$ 条之间的横向距离 `transverse` 恒等于 `0`**。kNN 取 Top-$k$ 时，同面板全亮条之间的挑选退化为依赖存储下标顺序的伪随机连接，丧失跨面立体邻域约束。

### 根因五：`B1` 投影网络与 `R3 C1` 表征缺陷及磁谱仪特征缺位
- **代码证据**：[v2/r1/data.py](../../src/analysis/flavour/v2/r1/data.py#L63-L73)、[v2/r1/models.py](../../src/analysis/flavour/v2/r1/models.py#L61-L76) 与 [v2/r3/sensor_rows.py](../../src/analysis/flavour/v2/r3/sensor_rows.py#L28-L41)。
- **三处缺陷与 `FORTVNE_10212026.pdf`（Zhongyi Wu 报告）及 `FORTVNE_pixel_vs_scintonly_meeting_260916.pdf`（Joern Mahlstedt 报告）对比**：
  1. **纵向位置与双视图配对双失**：`projection()` 将特征写入 `1322×420` 网格，卷积核在 XZ/YZ 独立前传后于第 71 行直接执行全图 `x.mean((-1,-2))` 与 `x.amax((-1,-2))` 并各自压成一维向量后才相加拼接。`Conv2d + GlobalPool` 对纵向平移动作近乎完全不变，既丧失反应深度包容位置 $z_v$，又彻底阻止在同一纵向模块 $z_k$ 处对齐 XZ 峰与 YZ 峰。
  2. **`1 mm` 网格的 90% 零空洞污染**：物理闪烁体条宽 $10.5\,\text{mm}$（40 根条覆盖 $420\,\text{mm}$），但代码仅将条能量写入中心 `1 mm` 单列下标 `bx = floor(measured - offset)`。在相邻两根物理紧密贴合的闪烁体条之间存在 **9 列全 0 空画素**，`C1 sensor rows` 虽然压缩了空行（吞吐提升 `2.723` 倍），却仍保留了横轴 `w=420` 稀疏填点，导致 `3×3` 卷积在浅层失去相邻闪烁体条感受野，`e recall` 反而下降 `3.42 pp`（[stage306](../../workbook/2026-11-15_stage306_r3_final_results.md)）。
  3. **欠拟合与缺乏显式谱仪物理重建量**：合作组 `Wu (2026)` 使用无空洞紧凑条阵列（`68×40`, `56×40`）、独立细像素分支（`15×250`, `15×200`）、`~2.9M` 参数、**40 epochs** 训练及 **12 维显式谱仪径迹动量/外推特征**；而 `R1 B1` 参数仅 `0.79M`、仅训练 `4 passes`（`R2` 中 `D0/D1/D2` 全程 `4–12 passes` 均未建立收敛 plateau），`TrackerSet` 仅对原始站坐标做无序池化。此外，[preprocessing_utils.py](../../src/analysis/utils/preprocessing_utils.py#L33-L34) 中曾存在将磁场二次拟合写在 $x$ 而非真实偏转轴 $y$（偶极磁场 $B \parallel \hat{x}$，参见 [objects.py](../../src/analysis/flavour/s5/objects.py#L113)）的历史错位，v3 必须明令杜绝。

### 根因六：FASERCal 范式（`arXiv:2604.07037v2`）为何在 `S4–S6` 水土不服
对照 FASERCal 原论文（Alonso-Monsalve et al., *Nature Machine Intelligence* 8, 1581–1595, 2026）与本仓库 `S4–S6` 实现：

| 维度 | FASERCal 原论文（成功范式） | 本仓库 `S4–S6`（效果仅 `+0.34 pp`） |
| --- | --- | --- |
| **探测器主体形态** | 均质 `3DCal`：$48\times 48\times 200$ 块 $1\,\text{cm}^3$ 三维立方闪烁体，全 3D $(x,y,z)$ + 模拟量电荷 | **极强各向异性混合体**：15 层 `20.8×22.8 μm` **二值**硅像素（每层后夹 $5\,\text{mm} \approx 1.43\,X_0$ 钨板）+ 96 层 **1D 正交** $10.5\,\text{mm}$ 条 + 下游磁谱仪 |
| **模型规模与主干** | `31.936M` 参数（`dim=384`），`SpConv` 3D 稀疏卷积 | `0.898M` 参数（`dim=96`），`1/35` 参数规模且依赖含数学死锁的 `B4` 编码器 |
| **样本域与预训练预算** | 仅选用严格发生于 `3DCal` 内的 `1.226M` 事例；batch `4096`；**400 epochs MAE + 100 epochs Rel**（耗资 `1,011` GH200 小时） | 混入半数靶外/边缘事例；预训练主确认仅 **3 epochs**（[s4_mae_confirmation.json](../configs/s4_mae_confirmation.json#L48)） |
| **`Rel` 语义覆盖范围** | 对**全部 3D 体素**预测 $(f_{\text{EM}}, f_\mu, f_{\text{had}})$ | 仅对粗粒度闪烁体条预测 `bar_rel`，漏掉最核心的高分辨**硅像素层** |
| **下游联合微调目标** | **同方差不确定度加权 4 头联合训练**：6 类味道（拆正反 $\nu_\mu$ 及 $\tau$）+ **4 类 Charm 产生识别** + 柱坐标动量 + **3D 主顶点回归**；配合 `0.75` 层间学习率衰减与 EMA | 下游主模型仅做单任务 4 类分类，缺失显式 Charm 对抗与簇级起源梯度导引 |

### 根因七：为何在 v3 中终止 `v2 R4` 启发式规则重建前置路线
- **物理与实测限制**：PINPOINT 钨靶每层厚度 $5\,\text{mm} \approx 1.43\,X_0$，交错放置的 Fortune 闪烁体模块将中间硅像素层分隔开高达 $\sim 80\text{–}120\,\text{mm}$ 纵向间距。对于典型 $10\text{–}50\,\text{mrad}$ 出射角度的多粒子径迹，仅层间横向偏移即达 $\Delta xy \approx 1\text{–}6\,\text{mm}$，同时电磁簇射产生大量贴近主轴的转换 $e^+e^-$。
- [objects.py](../../src/analysis/flavour/s5/objects.py#L43-L60) 采用固定 `width <= 0.5 mm` 且 `n <= 64` 过滤（直接删光高能电子簇射核），再做无角度外推的相互最近邻配对，结果 `S5` 径迹段纯度仅 `2.0%–5.4%`、双轨顶点匹配 `29/806`；`R3` [relations.py](../../src/analysis/flavour/v2/r3/relations.py) 在像素间隔 $1/2/3$ 层上的同 `trackID` 边召回率同样仅有 `27.53% / 10.15% / 17.48%`（[stage306](../../workbook/2026-11-15_stage306_r3_final_results.md)）。
- **结论**：不先定位主顶点与不靠学习动态消歧，任何纯几何硬规则追迹都无法在高能 TeV 钨靶簇射中兼顾高纯与高覆盖；将整条研究路线卡在等待 `R4` 规则追踪达标，性价比极低，不符合现代高能物理 ML 演进方向。

---

## 3. 前沿文献综述与 v3 精确映射

| 文献 / 出处 | 核心可验证机制 | 在 v3 中的精确落地模块 |
| --- | --- | --- |
| **ATLAS `GN2`** (ATLAS Collaboration, *Nat. Commun.* 2025, `arXiv:2505.19689`) | 淘汰传统 SV1 次级顶点前置重建；在主顶点坐标系下直接输入底层单元，仅**训练期**施加 **Track Origin**（主/重味次级/转化噪声）与 **Pairwise Vertex Compatibility** 密集辅助监督，使本底拒斥提升 $1.8\text{–}3.5\times$ | **Phase V2–V3 核心底座**：`vertex_anchor.py` 实现可微主顶点重中心化 + `losses.py` 对像素簇施加 5 类起源与同径迹/同顶点亲和监督，推理零真值生成次级顶点摘要 |
| **FASERCal** (Alonso-Monsalve et al., *Nat. Mach. Intell.* 8, 1581–1595, 2026, `arXiv:2604.07037v2`) | 同方差不确定度权重统一 6 类味道 + 4 类 Charm + 柱坐标动量 + 3D 顶点；全通道 `Rel` $(f_{\text{EM}}, f_\mu, f_{\text{had}})$ | **Phase V3**：7 类精细味道（拆 $\mu^-/\mu^+$ 与 $\tau_e/\tau_\mu/\tau_{\text{had}}$）+ 显式 Charm 产生头 + 3D 顶点 + 运动学头；并将 `Rel` 从条扩展至像素簇 |
| **`NuGraph2`** (Hewes et al., `arXiv:2403.11872`, 2024) | 针对多平面一维投影读出构造平面节点 $\leftrightarrow$ 3D Nexus 交叉节点异构注意力图消解鬼影，微中子语义分割精度达 `95%+` | **Phase V2**：`stereo_bars.py` 将正交 X-bar / Y-bar 在同模块内建候选 3D 交叉 Nexus 节点并做双部消息传递 |
| **`SPINE`** (Domine et al., *Phys. Rev. D* 2024) | 稀疏卷积提局部像素形态 + 图/Transformer 聚合粒子超图与主顶点回归级联 | **Phase V2**：`pixel_tokenizer.py` 保留全尺度像素簇形态矩 + 对复合大簇采用轻量 2D 局部微补丁卷积 |
| **Learnable QK-Norm & Softmax Sharpness** (Henry et al., *EMNLP* 2020; `ViT-22B`; Veličković et al., *ICML* 2025) | 证明无缩放 QK-Norm/标准 Softmax 随序列长度 $N$ 增大呈必然弥散；必须引入每头可学习缩放 $\gamma_h$ 及 $\log N$ 长度温度因子恢复尖锐度 | **Phase V2**：`ScaledQKNormAttention` 替换原 `StableAttention`，直接根治高占据度 $\tau$ 效率 21 倍塌陷 |
| **NOMAD & 合作组报告** (*Phys. Lett. B* 2001; `FORTVNE_10212026.pdf`; `FORTVNE_Update_kinematic_variables_2026-10-07.pdf`) | 未分辨衰变点 $\nu_\tau$ 依赖出射轻子与强子喷注在横向平面的非共面角与横向动量缺失；磁谱仪提供不可替代的轻子动量/电荷标定 | **Phase V1 & V4**：`spectrometer.py` 纠正并提取 $y$-$z$ 矢高与反向外推残差；`nomad_kinematics.py` 为纯闪烁体区注入 10 维横向动量失配变量 |
| **DRO One-Way Partial AUC** (Zhu et al., *ICML* 2022) | 通过 CVaR / 分位数凸松弛将严格限制 $\text{FPR} \le \beta$（如 `1%`）的非可微部分 AUC 转化为一阶 SGD 可微代理损失 | **Phase V4**：在标准多任务收敛检查点基础上微调 5 passes，专门压低混入最高 1% 本底尾部的伪 $\tau$ 事例 |
