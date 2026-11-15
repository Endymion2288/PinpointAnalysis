# 02 问题重构、可观测域分层与真值侧车规范（Phase V0）

**阶段目标**：在不修改原始探测响应的前提下，消除因靶外非包容事例混杂、$\nu_\tau$ 能谱截断以及全局单阈值导致的“性能压抑与效率反转”，并建立经过零泄漏验证的训练期多任务真值侧车（Truth Sidecar）。

---

## 1. 数据分区继承与零泄漏边界

完全继承 [R0 验收结果](../v2/r0_results.json) 与 [r2/experiment_plan](../v2/experiment_plan.json) 建立的家族无泄漏分区（相同 `family_hi, family_lo` 绝不跨集）：

| 分区名称 | 事件总量（四类平衡） | v3 阶段用途说明 |
| --- | ---: | --- |
| `Train100k` | `100,000` | `V1–V4` 单因子消融、小步长探索、健康与微米探针测试 |
| `Train300k` | `300,000` | `V1 B1++`、`V2 PinpointFormer`、`V3 GN2` 及 `V4` 的主训练基准池 |
| `Train1M` | `1,000,000` | 仅当 `Train300k` 验证增益明确且剩余算力允许时用于终态扩展或 SSL |
| `Tune40k` | `40,000` (`10k` × 4) | **开发期唯一评估集**：5-fold family-connected cross-fitting 拟合开发期分区域校准 |
| `Cal-A40k` | `40,000` | **V5 前全封存**：候选完全冻结后，仅用于拟合各可观测域最终校准温度 $T_d$ 与偏置 $b_d$ |
| `Cut-B160k` | `160,000` | **V5 前全封存**：固定选择各可观测域工作点阈值，绝不参与架构选择 |
| `Confirm200k` | `200,000` (`50k` × 4) | **V5 终审一次性开启**：独立确认报告与出版级物理显著性结果 |

---

## 2. 纯可观测包容度向量 $c_{\text{obs}}$ 与互斥校准域（`src/analysis/flavour/v3/common/domains.py`）

### 设计铁律
**严禁在模型前向推理、样本筛选路由或校准分箱中读取真值顶点 `initX, initY, initZ`。** 真值几何域（`truth_pixel_centered`、`truth_large_scintillator`）仅作评价表的离线物理分层展示；线上路由与分区域校准必须由**纯观测函数**确定：

```python
def compute_observable_containment(observation: dict, geometry: dict) -> dict:
    """仅从 pixel/bar/tracker_hypothesis 提取 16 维包容度特征与互斥可观测域编号。"""
```

### 16 维纯观测包容度特征 $c_{\text{obs}} \in \mathbb{R}^{16}$

| 下标 | 特征名称 | 定义与物理尺度 | 物理意义 |
| --- | --- | --- | --- |
| `0` | `has_pixel` | `float(len(pixel['channel_id']) > 0)` | 是否命中任意硅像素通道 |
| `1` | `has_bar` | `float(len(bar['channel_id']) > 0)` | 是否命中任意闪烁体条 |
| `2` | `has_tracker_any` | `float(len(tracker['station']) > 0)` | 是否击中至少 1 站磁谱仪 |
| `3` | `has_tracker_3station` | `float(len(np.unique(tracker['station'])) == 3)` | 是否贯穿全部 3 站具备动量符号测量能力 |
| `4` | `z_first_active_norm` | `(z_first - z_target_min) / L_target`（空响应置 `1.5`） | 重建起爆深度在靶区内的归一化纵向位置 |
| `5` | `z_last_active_norm` | `(z_last - z_target_min) / L_target` | 最深穿透深度 |
| `6` | `upstream_veto_active` | 最上游前 2 个闪烁体面板或第 0 层像素出现活跃击中 (`0/1`) | **强烈指示上游岩石穿透粒子或进入前已起爆** |
| `7` | `downstream_tail_energy_frac` | 最下游末 6 层闪烁体面板能量 / 总条能量 | 能量向后漏出比例 |
| `8` | `transverse_edge_energy_frac` | 两侧边缘闪烁体条（`bar_idx in {0, 1, 38, 39}`）能量 / 总条能量 | 侧面横向逃逸比例 |
| `9` | `bar_cog_x_norm` | 条能量加权中心 $x_{\text{cog}} / 210.0\,\text{mm}$ | 簇射横向中心 $x$ |
| `10` | `bar_cog_y_norm` | 条能量加权中心 $y_{\text{cog}} / 210.0\,\text{mm}$ | 簇射横向中心 $y$ |
| `11` | `bar_rms_x_norm` | 条能量横向均方根宽度 $\sigma_x / 100.0\,\text{mm}$ | 区分窄电磁簇射 vs 宽强子簇射 |
| `12` | `bar_rms_y_norm` | 条能量横向均方根宽度 $\sigma_y / 100.0\,\text{mm}$ | 同上（$y$ 轴） |
| `13` | `log1p_pixel_hits` | `np.log1p(n_pixel_hits) / 10.0` | 像素占据度尺度 |
| `14` | `log1p_bar_channels` | `np.log1p(n_bar_channels) / 6.0` | 闪烁体活跃条数尺度 |
| `15` | `log1p_bar_edep_norm` | `np.log1p(total_bar_edep_MeV) / 10.0` | 可见沉积总能尺度 |

### 4 个互斥且穷尽的可观测校准域（`obs_domain_id`）

由 `geometry` 精确导出像素有效横向窗口 `[px_min, px_max] × [py_min, py_max]`、像素最后一层位置 $z_{\text{pix,max}}$ 与靶末端 85% 深度阈值 $z_{\text{85}} = z_{\text{target,min}} + 0.85(z_{\text{target,max}} - z_{\text{target,min}})$，按顺序互斥判定：

1. **`obs_domain_id = 0` (`D_empty`)**：`n_pixel_hits == 0 and n_bar_channels == 0 and n_tracker_hits == 0`。
2. **`obs_domain_id = 1` (`D_pix_contained`)**：
   - `has_pixel == 1`；
   - `not upstream_veto_active`；
   - 首个出现像素或闪烁体击中的深度 $z_{\text{first}} \le z_{\text{pix,max}} - 50.0\,\text{mm}$；
   - `transverse_edge_energy_frac < 0.25`，且像素击中（若无足够像素则用条能量重心）横向中位数落于像素边界内缩 $15\,\text{mm}$ 安全盒内；
   - *预期物理属性*：覆盖绝大部分发生于硅像素交错区的可分辨反应顶点，用于驱动基于亚毫米拓扑的 $\nu_\tau$ 遴选。
3. **`obs_domain_id = 2` (`D_scint_contained`)**：
   - 不满足 `D_pix_contained`，且 `has_bar == 1`；
   - `not upstream_veto_active`；
   - `z_first <= z_85` 且 `transverse_edge_energy_frac < 0.30`。
   - *预期物理属性*：主要为发生于纯闪烁体段及像素边缘但横向/纵向上充分包容的事例，适合 NOMAD 平面运动学与全量能器剖面判别。
4. **`obs_domain_id = 3` (`D_partial_or_edge`)**：其余所有非空但存在上游穿透或剧烈边缘逃逸的事例。

> **W3-01 必查统计**：扫描 `Train300k` 与 `Tune40k`，生成 `4 (obs_domain) × 4 (truth_domain)` 混淆转移矩阵（真值分箱按原 [prepare.py](../../src/analysis/flavour/s2/prepare.py#L27-L48) 的 `pixel_centered / large_only / outside_large / empty`），验证 `D_pix_contained` 对真值 `pixel_centered` 的纯度与覆盖率（若需微调边缘阈值 `0.25/0.30` 或内缩距，仅根据 `Train300k` 统计冻结，不触碰 `Tune40k` 调参数）。

---

## 3. $\nu_\tau$ 能谱截断偏差控制（`src/analysis/flavour/v3/common/spectrum.py`）

依据 [stage04](../../workbook/2026-11-11_stage04_flavour_inventory.md) 记录，当前 `10083` $\nu_\tau$ 模拟样本能量范围止于 $E_\nu \le 2.3118\,\text{TeV}$，而其他三类高达 $5.0412\,\text{TeV}$。v3 实施两条强制控制：

1. **双切片报告铁律**：从 `V1` 起，每一份正式评估表在报告标准 `Tune40k` 指标的同时，必须**自动追加两项切片评价**：
   - `Tune_E_le_2p3TeV`（四类共同真值能量支撑 $E_\nu \le 2.30\,\text{TeV}$）：剥离高能端缺 $\tau$ 样本带来的能谱捷径虚高，真实反映拓扑与运动学判别力。
   - `HighE_non_tau_FPR`（在 $E_\nu > 2.30\,\text{TeV}$ 的非 $\tau$ 事例上检查 $p(\tau)$ 尾部误报率及高沉积能量区得分分布斜率）：确保模型不是只学会简单截断高能量事例。
2. **训练期能谱匹配对照开关（`energy_match_reweight`）**：在 `spectrum.py` 中对 `Train` 四类分别统计 20 个对数等宽 $E_\nu$ 分箱密度 $f_c(b)$，默认保持均匀类别采样以保留完整 $5\,\text{TeV}$ 物理三类知识，同时提供消融开关将 $E_\nu \le 2.30\,\text{TeV}$ 内的四类能量权重匹配至最小包络分布，用于定量检查高能端分布外推对 $\nu_\tau$ 阈值稳定的影响。

---

## 4. 训练期密集真值侧车（Truth Sidecar）生成规范（`src/analysis/flavour/v3/common/truth_sidecar.py`）

### 严格安全隔离契约
- 真值侧车以只读格式单独存放在 `/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/truth_sidecar/`（含独立前缀键 `(run, chunk, evtID)`），与观测缓存目录完全分离。
- **代码接口硬契约**：训练封装 `V3TrainDataset(obs_root, sidecar_root, split)` 与评估/推理封装 `V3EvalDataset(obs_root, split)` 为两个独立类；后者构造签名**根本不接收 `sidecar_root` 参数**，从源头保证推理与 `Cal-A / Cut-B / Confirm` 绝不可能触及真值。

### 事件级真值向量表（每事件固定存储，约 88 bytes）

复用并整合 [s5/truth.py](../../src/analysis/flavour/s5/truth.py#L84-L165) 与 [preprocessing_utils.py](../../src/analysis/utils/preprocessing_utils.py#L78-L146) 逻辑：

| 字段 | Dtype & Shape | 提取依据与缺失处理 |
| --- | --- | --- |
| `label_4` | `int8` | 原始 4 类 `0: e_CC, 1: mu_CC, 2: tau_CC, 3: NC` |
| `label_7` | `int8` | 细粒度 7 类：`0: e_CC, 1: mu_minus_CC (l_pdg==13), 2: mu_plus_CC (l_pdg==-13), 3: tau_e, 4: tau_mu, 5: tau_had, 6: NC`；若 `tau_branch=='unknown'` 置 `-1` 并置 `tau_branch_valid=0`，训练期自动对该事例退化为 $\tau$ 三子道边际和监督（不丢样本） |
| `charm_flag` / `charm_valid` | `uint8`, `uint8` | `primaries` 中存在任意含粲强子（PDG $\in \{411,421,431,4122,4112,4212,4222,4132,4232\}$ 及其反粒子）；不依赖后续树是否齐全 |
| `vertex_xyz_mm` / `vertex_valid` | `float32[3]`, `uint8` | `initX, initY, initZ`（mm）；`vertex_valid = isfinite & (not D_empty)` |
| `vis_kinematics` / `kin_valid` | `float32[4]`, `uint8` | `[log1p(pz_vis_GeV), log1p(pt_vis_GeV), cos(dphi_lh), sin(dphi_lh)]`，其中 `dphi_lh` 为主带电轻子（`τ` 通道以直接带电衰变产物向量和近似或标为未定义）与强子系统可见动量矢量和的横向方位角差 |
| `energy_nu_GeV` | `float32` | `initE` 换算 GeV，仅用于生成 `Tune_E_le_2p3TeV` 评估切片与可选重加权 |

### 像素簇与闪烁体条级密集真值标签（仅对 `Train` 分区生成）

为了支撑 Phase V3 的 **ATLAS `GN2` 式密集监督** 与 **全探测器 `Rel`**，对 `Train` 的每个连通像素簇（与 `V2` 的 `pixel_tokenizer.py` 使用**完全一致、无随机性**的八邻域/半径 `0.06 mm` 连通域定义）和每根活跃闪烁体条生成以下稀疏真值侧车（按 ragged 偏移存储在 `.npy` 中）：

1. **通道粒子组分向量 `rel_target` (`float32[N_elem, 3]`)**：
   - 将真值 Geant4 `hit_pdgc` 按物理规则映射为三类能量（条）或像素击中（像素簇）占比：`EM`（$|pdg| \in \{11, 22\}$）、`mu`（$|pdg| == 13$）、`had`（其余强子/离子等，排除未定噪声）；满足 $\sum_c f_c = 1$（缺失记 `rel_valid = 0`）。
2. **像素簇 5 类物理起源标签 `cluster_origin` (`int8[N_pix_clusters]`)**：
   - 先通过 `(run, chunk, evtID)` 核对已缓存颗粒度是否保留逐像素 `trackID / parentID / fromPrimaryLepton / fromCharmedHadron / fromTau`；若已有 parquet 未包含逐像素 `trackID`，由 `truth_sidecar.py` 在 `W3-02` 一次性直接扫描原始 ROOT `Hits/pixelHits` 树与 `trajectories / tau / charm` 树配对写入（仅存几百 KB/分片的整型侧车向量，不重制观测数组）：
     - `0: Primary_lepton`（直接源自初态反应 `trackID==1` 主轻子本身及其轫致/辐射不跨越独立顶点的关联像素；注意在 $\nu_\tau$ 中初态 $\tau$ 本身击中极短几乎不可见，真正标记为类 `2`）
     - `1: Primary_hadron`（源自 `parentID==0` 的初级强子或其直系径迹）
     - `2: Tau_daughter`（直接来自主 $\tau^\pm$ 衰变产物且非远离主衰变点的次级电磁转化噪声）
     - `3: Charm_daughter`（源自初态 Charm 强子衰变产物）
     - `4: Secondary_shower_or_other`（$e^+e^-$ 转换对、$\delta$ 电子、中子反冲次级散射）
3. **簇级真实归属标识 `cluster_ancestry_id` (`int32[N_pix_clusters, 2]`)**：
   - 记录每个像素簇的主导真值径迹号 `dominant_track_id` 与所属衰变顶点族号 `vertex_group_id`（主顶点产生物标 `0`，同一 $\tau$ 衰变顶点标 `1`，同一 Charm 衰变顶点标 `2`，次级孤立相互作用标 `-1`），在 `V3` 训练损失中在线生成任意两簇的 `same_track` 与 `same_decay_vertex` 对偶标签矩阵。

---

## 5. 分区域后验校准与 $\nu_\tau$ 对数似然比评分（`src/analysis/flavour/v3/common/calibration.py`）

为了在理论与工程上根治 `Pixel-centered` 全局阈值崩塌（`0.35%` 效率），定义标准分区域校准与跨区域可比判别分：

1. **分区域温度与偏置校准**：
   - 对非空可观测域 $d \in \{1 (\text{D\_pix}),\, 2 (\text{D\_scint}),\, 3 (\text{D\_edge})\}$，分别拟合独立温度 $T_d > 0$ 与偏置向量 $b_d \in \mathbb{R}^C$：
     $$\hat{\mathbf{p}}_d(x) = \text{softmax}\left( \frac{\mathbf{z}(x)}{T_d} + b_d \right), \qquad T_d > 0$$
   - 其中类别偏置默认加上均匀基准先验对齐，开发期在 `Tune40k` 内使用 **5 折按物理 `family` 分组的交叉拟合（OOF Cross-Fitting）**，严格无泄漏；`V5` 终审期在 `Cal-A40k` 拟合、冻结于 `Cut-B160k`。
2. **标准多域 $\nu_\tau$ 对数似然比评分**：
   $$\hat{p}_d(\tau \mid x) = \sum_{c \in \{\tau_e, \tau_\mu, \tau_{\text{had}}\}} \hat{p}_{d,c}(x)$$
   $$S_\tau(x) = \ln \frac{\hat{p}_d(\tau \mid x)}{\hat{p}_d(e \mid x) + \hat{p}_d(\mu \mid x) + \hat{p}_d(\text{NC} \mid x) + \lambda_c \hat{p}_d(\text{charm} \mid x) + 10^{-8}}$$
   当使用无真值 Charm 头模型时置 $\lambda_c = 0$。由于各域 $d$ 内部分母均为本底概率密度之和，物理意义上接近跨观测域可比的似然比统计量 $\ln \frac{f_d(x \mid \tau)}{f_d(x \mid \text{bkg})}$，**确保即使 `D_pix_contained` 与 `D_partial_or_edge` 具有截然不同的后验熵，同一阈值也能让各区域按真实信噪比贡献 $\tau$ 效率！**
3. **分区域工作点搜索策略**：
   - 在 `Cut-B`（或开发期 `Tune OOF`）上对比两种工作点方案：
     - **策略 I（单调全局 `S_tau` 切分）**：在整个非 $\tau$ 本底上寻找单一阈值 $\eta$ 使加权总 $\text{FPR} \le 1\%$（及 $0.1\%$）；
     - **策略 II（多域最优 Neyman-Pearson 预算分配）**：分别固定每个可观测域的本地 $\text{FPR}_d \in \{0.5\%, 1\%, 2\%\}$，使高纯 `D_pix_contained` 优先吃掉高信噪比信号预算，并在物理统计拟合端输出分域分箱 Asimov 显著性：
       $$Z_0 = \sqrt{2 \sum_{d \in \{1,2,3\}} \sum_{k=1}^{K} \left[ (s_{dk} + b_{dk})\ln\left(1 + \frac{s_{dk}}{b_{dk}}\right) - s_{dk} \right]}$$

> **立即可执行的零成本验证（工单 `W3-03`）**：在 `V1` 开始前，直接读取已保存的 `R2 B1-D2` 与 `S6 MAE3` 原始概率输出，仅应用本节 $d \in \{1, 2, 3\}$ 分域交叉校准 + $S_\tau$ 评分，立即验证在不改网络权重下就能恢复多少 `Pixel-centered` 的 `1% FPR` $\tau$ 效率。
