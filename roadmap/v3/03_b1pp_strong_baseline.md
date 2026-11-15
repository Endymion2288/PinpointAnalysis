# 03 B1++ 传感器原生纵向对齐强基线（Phase V1）

**设计定位**：建立一个**无需复杂动态图、不依赖外部 C++ 编译插件、I/O 极快且彻底修复既有投影与磁谱仪缺陷**的强三维融合基线 `ZAlignedStereoNet (B1++)`。该基线一方面直接对标并超越合作组 `Wu (2026)` 报告的水平，另一方面为后续高精度的 `PinpointFormer` 提供坚实的对比参照与阶段一集成组件。

---

## 1. 无空洞传感器原生二维矩阵（`src/analysis/flavour/v3/v1_b1pp/representation.py`）

彻底废弃将 `10.5 mm` 宽的闪烁体条硬插在 `1322×420` 毫米级画布上造成 9 列零空洞的做法，改为按物理探测单元与几何序号紧凑组织四张张量：

### 1.1 正交闪烁体面板紧凑矩阵（Zero-Hole Scintillator Grids）
- 依据 [readout.py](../../src/analysis/flavour/readout.py#L39-L75) 与 `Pinpoint_G4` 几何约定，每个 Fortune 面板严格包含 40 根条（`bar_idx in 0..39`，条间距 `10.5 mm`），全部 96 个面板严格分为测 $X$ 竖直条面板（`horizontal == 0`，共 $N_{z,x} = 48$ 层）与测 $Y$ 水平条面板（`horizontal == 1`，共 $N_{z,y} = 48$ 层）。
- 分别构建两张零空洞稠密张量（按物理 $z$ 升序排列行）：
  - $X_{\text{bar}} \in \mathbb{R}^{5 \times 48 \times 40}$
  - $Y_{\text{bar}} \in \mathbb{R}^{5 \times 48 \times 40}$
- 5 个特征通道严格物理标准化：
  1. `ch0`: $\ln(1 + E_{\text{edep,MeV}}) / 10.0$
  2. `ch1`: $\ln(1 + N_{\text{PE}}) / 15.0$
  3. `ch2`: $(z_{\text{panel,mm}} - z_{\text{mid}}) / 1000.0$（真实纵向深度中心，打破面板间不等距间距失真）
  4. `ch3`: $\Delta X_0^{\text{cum}} / 50.0$（当前面板到靶前端累计穿过钨+铝+闪烁体辐射长度，精确告知网络各处电磁簇射发展年龄）
  5. `ch4`: `active_mask`（是否有非零物理响应，区分零沉积通道与未布置像素边界）

### 1.2 双分辨率紧凑硅像素投影张量（Dual-Window Compact Pixel Projections）
对 15 个硅像素层（按 $z_{\text{pix}}$ 升序，实际硅面积约 $266\times 196\,\text{mm}^2$），针对每个视角 $v \in \{X, Y\}$ 同时构造细粒度与全幅面两套零偏移对齐矩阵：
- **中心细分辨窗** $P^{\text{fine}}_v \in \mathbb{R}^{4 \times 15 \times 512}$：覆盖中心 $[-128, +128]\,\text{mm}$（步长 `0.50 mm`——保留比旧 `B1` 细一倍的横向双簇剖面，同时覆盖超过 95% 的 `Pixel-centered` 击中）。
- **全宽覆盖窗** $P^{\text{wide}}_v \in \mathbb{R}^{4 \times 15 \times 256}$：覆盖全像素有效宽度 $[-140, +140]\,\text{mm}$（步长 $\approx 1.10\,\text{mm}$），保证外围大角度逃逸粒子不丢失。
- 4 个通道全部严格纯观测计算：
  1. `ch0`: $\ln(1 + N_{\text{pix\_in\_bin}}) / 6.0$（局部电荷密度近似）
  2. `ch1`: 该纵向层-横向 bin 内部最宽连通小簇的横向跨度 $\Delta u_{\max} / 1.0\,\text{mm}$（截断至 `2.0`）
  3. `ch2`: 纵向像素跨度 / 正交非投影轴方向离散度（衡量单层斜穿还是正穿）
  4. `ch3`: `bin_active_mask`

---

## 2. 磁谱仪物理运动学与反向外推特征（`src/analysis/flavour/v3/v1_b1pp/spectrometer.py`）

### ⚠️ 物理坐标系纠偏警示
依据 `Pinpoint_G4` 源码及 [objects.py](../../src/analysis/flavour/s5/objects.py#L18,L113) 注释核实：
- **磁场方向沿着 $\hat{x}$ 轴（$B \parallel \hat{x}$）**，带电粒子洛伦兹力 $\mathbf{F} = q\mathbf{v}\times\mathbf{B}$ 使粒子轨迹在 **$y$-$z$ 平面偏转**！
- 下游 3 个径迹站位于固定纵向坐标 `TRACKER_Z_MM = [47.4, 1237.4, 2427.4]`（两段间距严格相等 $\Delta z = 1190.0\,\text{mm}$），单站分辨率为偏转维 $\sigma_y = 0.017\,\text{mm}$、非偏转维 $\sigma_x = 0.58\,\text{mm}$。
- **开发红线**：严禁沿用旧 [preprocessing_utils.py](../../src/analysis/utils/preprocessing_utils.py#L33-L34) 将二次拟合错误加在 $x$ 上，**所有曲率、矢高与电荷符号均必须在 $y$ 坐标提取！**

### 候选项生成与物理显式特征计算

先实现可测试函数 `extract_spectrometer_kinematics(tracker_obs, cal_start_z_mm, cal_start_cog_xy_mm, geometry) -> dict`：

1. **三站物理三元组重建**（最多保留 `Top-4` 按品质因子排序的不重叠三站候选径迹，另生成事件级标量汇总）：
   - 对每一候选三站坐标点 $(x_0, y_0), (x_1, y_1), (x_2, y_2)$（对应站 $0, 1, 2$）：
     - **偏转面矢高（Bending Sagitta）**：
       $$s_y = y_1 - \frac{y_0 + y_2}{2}, \qquad \sigma_{s_y} = \sqrt{\frac{3}{2}}\,\sigma_y = 0.0208\,\text{mm}$$
       $s_y$ 的正负号直接决定粒子电荷符号 $q = \text{sign}(s_y)$（区分 $\nu_\mu \to \mu^-$ 与 $\bar{\nu}_\mu \to \mu^+$），绝对值 $|s_y|$ 在磁场均匀近似下严格反比于纵向动量 $p_z$。构造数值稳定的动量代理特征 $\hat{p}_{\text{norm}} = \frac{1.0}{(|s_y|/0.5\,\text{mm}) + 0.01}$ 以及显式带符号偏转比 $s_y / \sigma_{s_y}$。
     - **非偏转面直线残差**：
       $$\delta_x = x_1 - \frac{x_0 + x_2}{2}, \qquad \chi^2_x = \left(\frac{\delta_x}{\sqrt{3/2}\,\sigma_x}\right)^2 = \left(\frac{\delta_x}{0.710\,\text{mm}}\right)^2$$
     - **入站位置与入口切线斜率**：
       $$\theta_x = \frac{x_2 - x_0}{2\Delta z}, \qquad \theta_y^{\text{entry}} = \frac{y_1 - y_0}{\Delta z} - \frac{2 s_y}{\Delta z}$$
       （注意：$y(z) = y_0 + \theta_y^{\text{entry}}(z - z_0) + \kappa (z - z_0)^2$ 对等距 3 站成立 $\kappa \Delta z^2 = -2 s_y$，上式精确还原磁铁入口端斜率。）
     - **向靶区簇射起点的反向直线外推匹配**：从量能器观测首活跃层 $z_{\text{start}}$ 及局部前端重心 $(\bar{x}_{\text{start}}, \bar{y}_{\text{start}})$ 出发（零真值！）：
       $$\Delta x_{\text{ext}} = x_0 + \theta_x (z_{\text{start}} - z_0) - \bar{x}_{\text{start}}, \qquad \Delta y_{\text{ext}} = y_0 + \theta_y^{\text{entry}}(z_{\text{start}} - z_0) - \bar{y}_{\text{start}}$$
       并计算归一化反向外推匹配距离 $d_{\text{ext}} = \sqrt{(\Delta x_{\text{ext}}/20)^2 + (\Delta y_{\text{ext}}/20)^2}$。
       - *物理价值*：$\tau \to \mu\nu\bar{\nu}$ 衰变带来的多体运动学反冲，使得其出射 $\mu$ 的动量显著软于瞬发 `CC νμ` 领头 $\mu$（$|s_y|$ 更大），且与强子簇射起点轴线常存在更大横向斜率偏差 $(\Delta x_{\text{ext}}, \Delta y_{\text{ext}})$！
   - 组合成 **14 维候选径迹特征向量**（含质量分数、入站坐标、两轴斜率、带符号 $s_y$、$|s_y|$ 压缩动量代理、$\chi^2_x$、两轴外推残差与反向外推角度差、是否存在标志），送入轻量 `TrackerTrackAttention (dim=128)`；同时保留原始击中集合分支处理仅击中 1–2 站（未形成完整三站候选）的粒子（`B1` 中的 `TrackerSet` 扩展为 6 维输入并加上 $\sigma_x, \sigma_y$ 与到 $\bar{x}_{\text{start}}, \bar{y}_{\text{start}}$ 的相对方向角）。

---

## 3. 纵向 $z$ 对齐双视图立体主干 `ZAlignedStereoNet`（`src/analysis/flavour/v3/v1_b1pp/models.py`）

网络前向路径严禁对未对齐视图做全局 `Avg/MaxPool`，严格按照“**先横向特征提取，再全物理深度序列混合**”四步构造：

1. **横向保序编码器（Transverse-Only Stem + ConvNeXt/ResNet）**：
   - 分别处理四路紧凑矩阵（$X_{\text{bar}}$, $Y_{\text{bar}}$, $P^{\text{fine}}$, $P^{\text{wide}}$）：**仅在最后一个横向坐标轴 $W$ 上实施下采样与池化（`stride = (1, 2)`）**，纵向平面轴长度 $N_z \in \{48, 48, 15, 15\}$ 保持 $1:1$ 不变！
   - 对每一层 $l \in \{0, \dots, N_z - 1\}$，经横轴多头注意力加权池化 + 平均 + 最大池化并映射到共享通道维度 $D = 192$，得到严格携带物理纵向位置的层特征张量：
     - $H_{\text{bar},X} \in \mathbb{R}^{B \times 48 \times 192}$
     - $H_{\text{bar},Y} \in \mathbb{R}^{B \times 48 \times 192}$
     - $H_{\text{pix},X}, H_{\text{pix},Y} \in \mathbb{R}^{B \times 15 \times 192}$（将同一像素平面的 $P^{\text{fine}}$ 与 $P^{\text{wide}}$ 以及 $X/Y$ 双视角于同层拼接投影成 15 层像素嵌入）
2. **统一物理纵向序列拼装（$L_{\text{seq}} = 111$）**：
   - 将上述 $48 + 48 + 15 = 111$ 个纵向平面嵌入按真实物理坐标 $z_{\text{mm}}$ 严格升序交错重排为单一序列 $H \in \mathbb{R}^{B \times 111 \times 192}$（硅像素层、竖直闪烁体层、水平闪烁体层按它们在真实探测器内穿过粒子的先后因果序依次排列！）。
   - 为每个平面注入连续物理位置与材料编码：`MLP([z_norm, log1p(dz_prev_mm), cum_X0_norm]) + ModalityEmbedding({pix, Xbar, Ybar})`。
3. **纵向局部卷积 + 双向全局 Transformer 混合块（4 层）**：
   - 每层顺序执行：① `DepthwiseConv1d(kernel_size=5, padding=2)` + GLU 门控（专门捕获跨 2–3 个相邻正交 X/Y 面板与交错像素层的纵向簇射局部斜率与转化 $e^+e^-$ 起爆突变）；② **Pre-Norm 自注意力**（使用带可学习温度的 QK-Norm，并附加面板间 $|\Delta z|$ 高斯/MLP 相对位置偏置）。
4. **物理分窗纵向读出 + 多分支融合头**：
   - 从输出序列 $H^{(4)} \in \mathbb{R}^{B \times 111 \times 192}$ 中，根据**纯观测**首活跃层位置下标 `idx_start` 提取 4 个具明确物理含义的汇总向量（每组拼接 Mean + AttentionPool）：
     1. **簇射起点窗口** `H[idx_start : idx_start + 8]`：高信噪比区分 `e`（$1\text{–}3\,X_0$ 迅速成倍增殖）vs `hadron / NC`（强子相互作用前较稀疏）vs `μ`（全程单径迹 MIP）；
     2. **主能量峰值邻域**（围绕最大沉积层下标 `idx_peak ± 4`）：捕获电磁/强子簇射核心宽度；
     3. **后端穿透窗口** `H[-12 : ]`：测量强子穿透漏出与尾端最小电离径迹信号；
     4. **全序列双向注意力池化**。
   - 拼接磁谱仪特征向量 $F_{\text{spec}} \in \mathbb{R}^{192}$ 与纯观测包容向量嵌入 `MLP(c_obs)`（总模型参数量控制在 **`~2.8M`**，与合作组 `Wu` 模型量级一致）。

---

## 4. 抑制过拟合的物理精确对称增强与训练超参

为解决 `R2 D1` 延长训练至 `12 passes` 出现的过拟合退化问题，加入不破坏任何模拟物理规律的精确几何增强，并将训练预算恢复至正常收敛水平：

1. **物理严格精确镜像增强**：
   - **左右镜像 ($x \to -x$，训练时概率 `0.5`)**：PINPOINT 偶极磁场沿 $\hat{x}$ 轴指向，整个磁力与正交条布局关于 $x \to -x$ 严格镜面对称！同步反转 $X_{\text{bar}}$ 的 40 根条序号 (`bar_idx -> 39 - bar_idx`)、像素 $X$ 索引、$X$ 方向谱仪坐标 `x -> -x`（仅对物理中心对称或保持中心参考；注意若有微小的 `fDetectorXOffset`，在相对像素/条坐标系下作去偏镜像或验证在几何均值处对称）以及 $c_{\text{obs}}$ 中对应 $x$ 重心符号。所有 4 类/7 类标签严格不变！
   - **上下镜像 ($y \to -y$，训练时概率 `0.5`)**：翻转 $Y_{\text{bar}}$ 的 40 根条序号、像素 $Y$ 索引与谱仪坐标 `y -> -y`（注意需考虑 `Pinpoint_G4` 中 `fDetectorYOffset = -12 mm` 的机械偏移，镜像应关于面板对称中心 $y_{\text{center}}$ 进行：$y' = 2 y_{\text{center}} - y$，保证不越界！）。在磁场不变情况下，$y' - y_{\text{center}}$ 镜像等价于电荷共轭的偏转方向：
     - 训练 `T3 / T4`（不区分正反中微子味道）时，味道标签不变；
     - 训练 `7-way`（显式区分 `mu_minus_CC` 与 `mu_plus_CC`）时，仅当执行 $y$ 镜像且磁场方向物理固定时，$s_y \to -s_y$ 对应电荷符号反转，因此只需同步互换 `mu_minus_CC <-> mu_plus_CC` 真值标签即可保持 100% 物理精确等价！
2. **正则化与优化配置**：
   - 轻量传感器通道随机遮挡 `channel_dropout = 0.03`、主干 `DropPath = 0.10`、`AdamW(lr=5e-4, weight_decay=0.03)`、梯度裁剪 `clip_grad_norm = 1.0`。
   - 训练预算：**短筛 `Train100k × 16 passes`，主训练 `Train300k × 24 passes`**（前 `5%` 线性 warmup + Cosine 退火至 `1e-5`），全程维护 `EMA (decay=0.999)` 并在 `Tune` 上对比 EMA 与最后权重。
   - **同步合作组对齐检查**：用 `B1++` 分别跑 `T3 (仅非τ)` 与 `T4 (四类)`，在 `Large` 与 `Pixel-centered` 域及合作组相似选区上输出 `BA3 / AUROC / 80% & 90% purity 效率`，完成与 [FORTVNE_10212026.pdf](../v2/08_r1_collaboration.md) 的同等级可比对齐。
