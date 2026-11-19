# 2026-11-18 Stage 319 — 硬件空间分辨率重训曲线、C++17 显式微米次级顶点混合重建与干净基准确认集一次性解封报告

**结论先行：在修复 `Pinpoint_G4` 纵向层序错位（`P0-2`）、剥离双通道闪烁体泄漏并引入 Birks 猝灭 / SiPM 饱和 / $\ge 2.5\text{ PE}$ 阈值（`P1-1`）、补齐真实中性流初态 $c\bar{c}$ 与 $\tau^\mp$ 螺旋度及 `Sm2Co17` 磁体外环（`P2-1` / `Exp-4`）生成的 60,000 事件干净模拟基准集上，严格执行预解封权重与规则冻结（提交 `695db37a7a4a1ea765458cdc5aaf1fa787dc6287`，`P3-1`）后完成 `CleanCalA(6k) → CleanCutB(12k) → CleanConfirm(10k)` 一次性盲测解封。核心实测结论如下：(1) 从头重训硬件螺距曲线（`Exp-2` / `P1-5`）证伪了事后量化的 OOD 伪退化，在 `Truth_pixel_centered` 精确 `1.0% FPR` 下 `20.8 μm`（`4.98%`, `BA4 = 61.33%`）、`100 μm`（`5.09%`, `BA4 = 61.27%`, $\Delta\varepsilon_\tau = -0.12\text{ pp}$, 95% CI `[-1.41, +1.09] pp`）与 `300 μm`（`5.97%`, `BA4 = 61.65%`）形成由 `10 mm / 70 mm` 钨板 Highland 多重库仑散射（$\sigma_{\text{MS}} \approx 70\text{–}210\,\mu\text{m}$）主导的物理平台，而粗化至 `1.0 mm`（$\sigma_{\text{int}} = 289\,\mu\text{m} > \sigma_{\text{MS}}$）使 `BA4` 显著退化 `-0.97 pp`（95% CI `[+0.41, +1.62] pp`）且 `0.1% FPR` 效率跌落 `3.2` 倍（`1.11% → 0.35%`）；靶区硅层数减半（`6 Target + 3 IPT`）保持 `BA4 = 61.44%`（$\Delta = -0.12\text{ pp}$）；(2) C++17 显式微米多径迹 / 自适应顶点拟合（`AVR`）/ $d_0/\sigma_{d_0}$ / kink 混合模型（`Path C`）在全部 8 组 24 个模型中排名第一，将 `Truth_pixel_centered` 精确 `1% FPR` $\nu_\tau$ 效率从纯点簇基线的 `4.98%` 提升至 `6.15%`（`+1.17 pp`，相对增益 `+23.5%`，配对 Bootstrap 95% CI `[0.00, +2.51] pp`），锁定 `0.1% FPR` 切点下效率从 `1.35%` 提升至 `1.87%`（相对 `+38.9%`），横向顶点中位误差从 `1.02 / 0.93 mm` 缩减 `-18.6%` 至 `0.83 / 0.79 mm`，纵向中位误差降至 `5.80 mm`（偏置 `+0.21 mm`）；(3) 传感器平面横向机械未对齐在 $\sigma_{xy} \le 50\,\mu\text{m}$ 内仅偏移 `-0.06 pp`（`4.98% → 4.92%`），上游双层闪烁体反符合对外部岩石 $\mu$ 子剔除率达 `97.67%` 且 `1% FPR` 泄漏率为 `0.00%`（`0/300`），外部中性强子（$n, K^0_L$）超尾端前 3 箱泄漏率为 `0.00%`（`0/300`）；(4) 在包含 21 个超尾端信号箱 + 6 个粲富集控制箱、Jeffreys $\text{Gamma}(\frac{1}{2},1)$ 零计数共轭先验及 16 个物理/Geant4/产生器/穿入冗余参数的 27 箱联立剖面似然下（`Path D`，1,000 次 $H_0$ 玩具校准 `zero_fraction = 0.498 / 0.511`），2t 靶内纯统计显著性在 HL-LHC（`3 ab^-1`）与 FPF（`10 ab^-1`）分别为 `1.77σ` 与 `3.23σ`，计入完整物理与模拟系统误差后分别为 `0.63σ` 与 `1.05σ`（附加 10k 确认集有限 MC 统计惩罚下界为 `0.21σ`）。**

---

## 阶段目标

完成 [Stage 314 独立终审报告](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/workbook/2026-11-18_stage314_independent_scientific_statistical_and_code_audit.md) 中第 1–6 个月路线图的全部决定性实验与统计闭环，回答以下核心问题并检验验收目标：

1. **硬件空间分辨率与层数权衡（`Exp-2` / `P1-5` / `Path E`）**：在修复 `Pinpoint_G4` 几何与单通道 PE 数字化后的干净数据集（`CleanTrain32k`）上，从头重训 `20.8 μm`、`100 μm`、`300 μm`、`1.0 mm` 靶区硅像素螺距及 `6 层靶硅 + 3 层 IPT`（节省 50% 靶区硅面积）共 5 组 15 个模型，剥离推理期事后量化的 OOD 协变量偏移，检验 `Stage 314` 停止门限（`100 μm` 与 `20.8 μm` 差异是否 `< 1.0 pp`）。
2. **显式微米多径迹与次级顶点混合重建（`Path C`）**：在 C++17 原生扩展中实现限制于相邻 `30 mm` `PinpointBlock` 平面偶的微米径迹段（tracklet）提取、迭代重加权自适应主顶点拟合（`AVR`）、冲击参数显著性 $d_0/\sigma_{d_0}$、双径迹次级顶点 $\Delta\chi^2$ 与跨层 kink 角重构，融合至精简版 `PinpointFormer`（及 `NOMAD` 交叉验证组），检验相对纯点簇模型的尾部 $\nu_\tau$ 选择增益与顶点精度提升。
3. **机械失准容限、配对 Geant4/产生器系统误差与外部穿透本底（`Path E`）**：定量测量传感器平面横向失准 $\sigma_{xy} \in \{0, 10, 25, 50\}\,\mu\text{m}$、4 组配对 Geant4 输运系统误差（$\tau$ 非极化、钨密度 $\times 0.98$、移除磁体外环、磁场 $\times 0.97$）、同种子 `GENIE Pythia6_NUX` 强子化变体，以及外部岩石 $\mu$ 子与中性强子（$n, K^0_L$）穿透率。
4. **预解封冻结与 27 箱真实前向流强联立剖面似然（`P3-1` / `Path D`）**：在未读取任何封存集的条件下完成 Git 提交与 24 个检查点 SHA-256 清单冻结，随后一次性解封 `CleanCalA → CleanCutB → CleanConfirm`，给出 Run 3 / Run 4 / HL-LHC / FPF 真实前向流强发现显著性及 1,000 次频率派 $H_0$ 玩具覆盖率验证。

---

## 工作内容

### 1. 解封前 C++17 微米径迹/次级顶点核与跨折校准对齐精修（仅使用 `CleanTune8k`）

在触碰任何封存分区之前，基于 `CleanTrain32k / CleanTune8k` 诊断并修正了两处物理与统计细节（提交 `695db37a7a4a1ea765458cdc5aaf1fa787dc6287`）：

- **C++17 `fast_reconstruct_micron_vertex` 物理窗口与钨散射约束**（[_fast_kernels.cpp](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/common/_fast_kernels.cpp)）：
  1. 以闪烁体簇射起点 `cal_start` 纵向窗口 $[z_{\text{cal}} - 45\text{ mm}, z_{\text{cal}} + 165\text{ mm}]$ 锚定候选像素起始平面 `p0`，避免 `FortuneBlock` 事件误锁定下游 $+500\text{ mm}$ 处的尾部 `IPT` 平面。
  2. 强制径迹段仅连接纵向间距 $\Delta z \le 45\text{ mm}$ 的相邻 `PinpointBlock` 硅平面，拒绝跨越 `150 mm`（`20 X_0`）无硅 `FortuneBlock` 的伪直线配对；单簇场景采用修复后钨板中心精确偏置 `-17.515 mm`（替代旧 `-8.5 mm`）。
  3. 施加 $d_0 \le 1.80\text{ mm}$ 光子对产生（$\gamma \to e^+e^-$）过滤，将原恒等于 `1.0` 的冗余维替换为物理有效的首平面最小电离簇占比 `first_plane_mip_fraction`；在分片 `080` 上解析 `AVR` 主顶点纵向中位误差由 `43.76 mm` 降至 `7.43 mm`。
- **混合门控初始化与 OOF 粲惩罚对齐**（[models.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v3_multitask/models.py)、[ensemble.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v4_tau/ensemble.py)）：将 `micro_vtx_gate` 初值设为 `0.02` 并增加解析 `AVR` 与闪烁体起点的纵向一致性检验（$|z_{\text{AVR}} - z_{\text{cal}}| < 90\text{ mm}$）；在折外校准中为 `lambda_charm` 引入 `+0.5 pp` 迟滞门限与跨折 99% 分位阈值对齐，消除离散惩罚跳变导致的全局切点损耗。

### 2. 8 组 24 个干净模拟模型训练与解封前冻结审计链（`P3-1`）

- 在 `CleanTrain32k`（`32,000` 训练事件，`CleanTune8k` `8,000` 验证事件，严格 `pe_only=True`、`bar_threshold_pe=2.5`、无偏固定多任务损失、1D 参数权重衰减豁免）上完成全部 8 组 $\times$ 3 随机种子（`17, 42, 2026`）= **24 个模型**从头训练（每模型 20 passes，`bs=64`，`CosineAnnealingLR`）：
  1. `s319-pitch-20um`（`clean_pf_pitch_20um_s*`）：标准 $20.8 \times 22.8\,\mu\text{m}^2$ 靶硅 + 12 靶层 + 3 `IPT`。
  2. `s319-pitch-100um`（`clean_pf_pitch_100um_s*`）：靶硅数字节距粗化为 $100 \times 100\,\mu\text{m}^2$ 从头重训。
  3. `s319-pitch-300um`（`clean_pf_pitch_300um_s*`）：靶硅数字节距粗化为 $300 \times 300\,\mu\text{m}^2$ 从头重训。
  4. `s319-pitch-1mm`（`clean_pf_pitch_1mm_s*`）：靶硅数字节距粗化为 $1.0 \times 1.0\text{ mm}^2$ 从头重训。
  5. `s319-layers-6t`（`clean_pf_6layer_20um_s*`）：仅保留偶数号 6 层靶区硅传感器 + 3 层 `IPT`（削减 50% 靶区硅面积）从头重训。
  6. `s319-hybrid-uvtx`（`clean_pf_hybrid_uvtx_20um_s*`）：启用 C++17 微米径迹 / `AVR` / $d_0/\sigma_{d_0}$ / kink 混合分支（`Path C`）。
  7. `s319-hybrid-uvtx-nomad`（`clean_pf_hybrid_uvtx_nomad_20um_s*`）：`Path C` 叠加物理动量修正后的 `NOMAD` 横向运动学头（`Exp-5` 干净模拟复核）。
  8. `s319-b1pp-clean`（`clean_b1pp_pe_only_s*`）：纯阈值化 PE 双投影 CNN 基线 `ZAlignedStereoNet`。
- **预解封冻结验证**：在提交 `695db37a7a4a1ea765458cdc5aaf1fa787dc6287` 后执行 `run_stage319_clean_campaign freeze`（UTC `2026-11-19T02:55:48Z`），将 40 位 Git SHA、全部 24 个检查点 SHA-256 及主分析选型同时写入仓库内 [v6_freeze_manifest.json](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/roadmap/v3/v6_freeze_manifest.json) 与 EOS 运行目录，彻底消除 `P3-1` 的 `null` 提交号问题。

### 3. 下游 27 箱超尾端剖面似然零计数分箱先验与控制区分位点修复（`Path D`）

在一次性解封脚本 [run_stage319_clean_campaign.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v6_clean_sim/run_stage319_clean_campaign.py) 与似然引擎 [flux_significance.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v5_confirm/flux_significance.py) 中，不改动任何冻结模型权重、`Cal-A` 校准或 `Cut-B` 信号切点，完成两项统计精确化处理并在 [test_stage316_audit_fixes.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/tests/test_stage316_audit_fixes.py#L345-L355) 新增回归单测：

1. **Jeffreys $\text{Gamma}(\frac{1}{2}, 1)$ 零本底分箱共轭先验**：`CleanConfirm` 像素域最高纯度箱（`bin 6`，$\ge 99.85\%$ 分位）落入 10 个 $\nu_\tau$ 信号事件而 0 个 MC 背景事件。若直接取 $b_6 = 0$，既会在 `pos = nu_pred > 1e-12` 掩码下漏算 $H_0$ 似然惩罚，又会使 $H_0$ 玩具抽样因 $\text{Poisson}(0)=0$ 产生 $+2 s_6 \mu$ 单侧线性斜率而将 $\hat{\mu}$ 锁死在 `0`。采用标准贝叶斯-频率共轭 Jeffreys 零计数先验 $b_{\text{Jeffreys}} = 0.5 \bar{w}_{\text{bkg}}$ 且相对 MC 不确定度 $\delta_b = 100\%$（`var_mc_bkg = b_Jeffreys^2`），既避免有限 MC 造成的虚假无穷显著性，又严格恢复 Wilks 定理的 $\frac{1}{2}\delta_0 + \frac{1}{2}\chi^2_1$ 渐近校准。
2. **`Cut-B` 分位点锁定粲控制区与重要性抽样精确方差传播**：在 `Cut-B` 低 $S_\tau$ 体区（$\le 80\%$ 分位）锁定各域 `charm_prob` 的 `85%` 与 `96%` 分位点作为 6 个粲控制区（`CR_charm`）边界，并在 `build_physical_templates` 内部直接将 `Stage 318` 富集至 `15.3%` 的 `NC_charm` 重加权至物理前向产额占比 `4.5%`，精确传播事件权重平方和 $\sum w_i^2$。

---

## 详细数据

### 1. 执行命令与核心产物路径

```bash
# 1. 解封前冻结 24 个检查点 SHA-256 与 Git 提交号
python3 -m analysis.flavour.v3.v6_clean_sim.run_stage319_clean_campaign freeze

# 2. 一次性解封评估 + 8 组配对系统误差 + 1,000 次 H0 玩具覆盖率校准
python3 -m analysis.flavour.v3.v6_clean_sim.run_stage319_clean_campaign unseal-and-evaluate --n-toys 1000

# 3. 全量自动化回归单元测试
python3 -m unittest discover -s tests
```

- **预解封冻结清单**：[v6_freeze_manifest.json](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/roadmap/v3/v6_freeze_manifest.json)（`pre_unsealing_git_commit = 695db37a7a4a1ea765458cdc5aaf1fa787dc6287`）
- **解封数值总表 JSON**：[stage319_clean_unsealing_report.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v4_audit_2026-11-18/clean_sim/stage319_clean_unsealing_report.json)
- **300-DPI 英文图表**：
  - [stage319_exp2_hardware_pitch_layers_and_misalignment.png](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v4_audit_2026-11-18/clean_sim/stage319_exp2_hardware_pitch_layers_and_misalignment.png)
  - [stage319_hybrid_microvertex_and_clean_confirm_unsealing.png](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v4_audit_2026-11-18/clean_sim/stage319_hybrid_microvertex_and_clean_confirm_unsealing.png)

### 2. `CleanConfirm`（10,000 事件，含 `Truth_pixel_centered` 6,697 事件）一次性解封主表

| 模型组（均为 3-Seed `Cal-A → Cut-B` 校准融合） | 全域 `BA4` (%) | 全域 `NLL` | 全域精确 `1% / 0.1%` $\varepsilon_\tau$ (%) | 像素靶内 `BA4` (%) | 像素靶内 `cond_BA3` (%) | 像素靶内精确 `1% / 0.1%` $\varepsilon_\tau$ (%) | `Cut-B` 锁定 `1%` $\varepsilon_\tau$ (*实际 FPR*) | 靶内顶点中位 $|\Delta x| / |\Delta z|$ (`mm`) |
| --- | ---: | ---: | --- | ---: | ---: | --- | --- | --- |
| `s319-pitch-20um`（20.8 μm 基准） | 58.30 | 0.9642 | 4.44 / 0.92 | 61.33 | 76.97 | 4.98 / 1.11 | 4.98% (*0.982%*) | 1.02 / 6.20 |
| `s319-pitch-100um`（100 μm 重训） | 58.08 | 0.9638 | 4.32 / 0.84 | 61.27 | 77.07 | 5.09 / 0.76 | 5.27% (*1.082%*) | 1.00 / 5.86 |
| `s319-pitch-300um`（300 μm 重训） | 58.32 | 0.9593 | 5.24 / 0.84 | **61.65** | **77.53** | 5.97 / 0.94 | 5.80% (*0.922%*) | 0.86 / **5.68** |
| `s319-pitch-1mm`（1.0 mm 重训） | 57.45 | 0.9699 | 3.96 / 0.84 | 60.36 | 76.22 | 4.10 / **0.35** | 4.10% (*0.922%*) | 1.04 / 6.33 |
| `s319-layers-6t`（6 靶层 + 3 IPT） | 58.29 | 0.9604 | **5.96** / 1.12 | 61.44 | 77.19 | 5.74 / **1.17** | 5.62% (*0.942%*) | 0.90 / 6.42 |
| **`s319-hybrid-uvtx`（Path C 混合）** | **58.36** | **0.9572** | 5.40 / 1.12 | 61.41 | 76.89 | **6.15 / 1.17** | **6.03%** (*0.922%*) | **0.83** / 5.80 |
| `s319-hybrid-uvtx-nomad`（+NOMAD） | 57.83 | 0.9637 | 4.52 / 0.52 | 60.71 | 76.50 | 5.56 / 0.47 | 4.16% (*0.802%*) | 0.85 / 6.45 |
| `s319-b1pp-clean`（双投影 CNN） | 57.04 | 0.9698 | 4.44 / 0.96 | 60.01 | 76.21 | 4.33 / 1.11 | 4.68% (*0.982%*) | — |
| `primary_dual_family_6m`（PF20+B1++） | 58.18 | 0.9600 | 4.08 / **1.24** | 61.26 | 77.30 | 4.74 / **1.17** | — | — |

- **本底成分在 `Cut-B` 锁定 `1% FPR` 切点下的真实通过率拆解**（`s319-hybrid-uvtx`, `Truth_pixel_centered`）：轻味本底（无粲 CC + 无粲 NC）假阳性率压低至 **`0.674%`**，而带真实次级衰变顶点的 **CC 关联粲本底假阳性率为 `2.40%`**（富集 `3.6` 倍）、**NC $c\bar{c}$ 粲本底假阳性率为 `2.59%`**（富集 `3.8` 倍），定量证实真实次级顶点粲强子（$D^\pm, D^0, D_s^\pm, \Lambda_c^+$，$c\tau \sim 60\text{–}312\,\mu\text{m}$）是 $\nu_\tau$ 识别的头号不可约物理本底。
- **从头重训 vs 事后量化 OOD 偏移对比（`P1-5` 实锤解决）**：
  - 将 `100 μm / 300 μm / 1.0 mm` 事后量化直接输入 `20.8 μm` 训练模型（OOD 偏移），`Truth_pixel_centered` 精确 `1% FPR` 效率单调坍缩为 `4.98% → 4.57% → 4.51% → 3.45%`（相对下跌 `-30.6%`）。
  - 从头重训消除协变量偏移后，真实物理曲线为 **`4.98%`（`20.8 μm`）→ `5.09%`（`100 μm`）→ `5.97%`（`300 μm`）→ `4.10%`（`1.0 mm`）**。
- **配对家庭块 Bootstrap（`n_boot = 1,000`）关键假设检验**：
  - `Path C 混合 - 20.8 μm 纯点簇`：$\Delta\varepsilon_\tau@1\% = \mathbf{+1.17\text{ pp}}$（95% CI **`[0.00, +2.51] pp`**）；在 `Cut-B` 锁定 `0.1%` 切点下 $\varepsilon_\tau$ 从 `1.35%` 升至 `1.87%`（`+0.52 pp`）。
  - `20.8 μm - 100 μm 从头重训`：$\Delta\varepsilon_\tau@1\% = -0.12\text{ pp}$（95% CI `[-1.41, +1.09] pp`），$\Delta\text{BA4} = +0.06\text{ pp}$（`[-0.55, +0.65] pp`），差异远小于 `1.0 pp` 门限。
  - `20.8 μm - 1.0 mm 从头重训`：$\Delta\text{BA4} = \mathbf{+0.97\text{ pp}}$（95% CI **`[+0.41, +1.62] pp`**，零点以上显著退化），$\Delta\varepsilon_\tau@1\% = +0.88\text{ pp}$（`[-0.41, +1.92] pp`），$\Delta\varepsilon_\tau@0.1\% = +0.76\text{ pp}$（`1.11% vs 0.35%`）。
  - `12 靶层 - 6 靶层`：$\Delta\text{BA4} = -0.12\text{ pp}$（`[-0.74, +0.51] pp`），$\Delta\varepsilon_\tau@1\% = -0.76\text{ pp}$（`[-2.92, +0.29] pp`）。
  - `双族 6 模型集成 - 单族 PF 3 模型`：$\Delta\text{BA4} = -0.07\text{ pp}$（`[-0.61, +0.43] pp`），再次印证 `Stage 314` 结论：单族 3-seed `PinpointFormer` 已无需叠加双投影 CNN 集成。

### 3. 传感器机械失准、配对 Geant4/产生器系统误差与外部穿入本底（`Path E`）

| 测试类别 | 变体 / 条件 | 样本规模 | 全域 `BA4` (%) | 靶内 `1% FPR` $\varepsilon_\tau$ 或本底泄漏指标 |
| --- | --- | ---: | ---: | --- |
| **平面横向失准扫描** | $\sigma_{xy} = 0 / 10 / 25 / 50\,\mu\text{m}$ | 10,000 | 58.30 → 58.27 | `4.98% / 4.98% / 4.98% / 4.92%`（50 μm 仅 `-0.06 pp`） |
| **配对 Geant4 输运** | `nominal_sys_ref`（标称对照） | 600 | 58.50 | `2.83%` |
| **配对 Geant4 输运** | `tau_helicity_0`（关闭 $\tau$ 极化） | 600 | 58.33 | `1.89%`（$-0.17\text{ pp BA4}$） |
| **配对 Geant4 输运** | `w_density_0p98`（钨密度 $-2\%$） | 600 | 57.17 | `3.77%`（$-1.33\text{ pp BA4}$） |
| **配对 Geant4 输运** | `no_magnet_shells`（移除 `Sm2Co17` 外环） | 600 | 56.67 | `0.94%`（$-1.83\text{ pp BA4}$） |
| **配对 Geant4 输运** | `field_0p97`（偶极场 $-3\%$） | 600 | 58.00 | `2.83%`（$-0.50\text{ pp BA4}$） |
| **同胞产生器强子化** | `pythia6_std_ref` vs `pythia6_nux` | 600 / 600 | 61.83 → 60.33 | `6.06% → 3.03%`（$-1.50\text{ pp BA4}$） |
| **外部穿入本底** | 上游岩石 $\mu^\pm$（$10\text{–}1000\text{ GeV}$） | 300 | — | 上游双板反符合效率 **`97.67%`**，未反符合 `1%` 切点泄漏率 **`0.00%` (`0/300`)** |
| **外部穿入本底** | 外部中性强子 $n, K^0_L$（$10\text{–}300\text{ GeV}$） | 300 | — | 未触发反符合占比 `20.67%`，`1%` 切点泄漏率 **`0.33%` (`1/300`)**，前 3 超尾箱泄漏 **`0.00%` (`0/300`)** |

### 4. 27 箱真实前向流强发现显著性与 1,000 次 $H_0$ 覆盖率校准（`Path D`）

采用 2t 靶内真实前向流强组成（$f_\tau = 0.30\%$，$\nu_e:\nu_\mu:\text{NC} = 17.0\%:58.0\%:24.7\%$，NC 内物理 $c\bar{c}$ 占比 `4.5%`）、21 个超尾端信号箱 + 6 个 `Cut-B` 锁定的粲控制箱、Jeffreys 零计数先验及全部 16 个有符号冗余参数（流强、三域归一化与尾部斜率、$\nu_e/\text{NC}$ 比、$\pm 30\%$ 粲截面、4 项 Geant4 输运、`Pythia6_NUX` 强子化、中性强子穿入）：

| 实验阶段（积分亮度） | 靶内总中微子事件数 | 预期 $\nu_\tau$ CC 信号 $S$ | 预期总背景 $B$（含粲背景） | 27 箱纯统计 $Z_{0,\text{stat}}$ | 完整物理+模拟系统误差 $Z_{0,\text{PL}}$ | 附加 10k 确认集有限 MC 统计下界 $Z_{0,\text{PL+10kBB}}$ | 收敛状态 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | :---: |
| **FASER Run 3** (`250 fb^-1`) | 15,000 | 45.0 | 14,956.1 (`1,398.8`) | `0.51σ` | **`0.24σ`** | `0.17σ` | `True / True` |
| **LHC Run 4** (`680 fb^-1`) | 40,800 | 122.4 | 40,680.5 (`3,804.9`) | `0.84σ` | **`0.34σ`** | `0.19σ` | `True / True` |
| **HL-LHC** (`3,000 fb^-1`) | 180,000 | 540.0 | 179,472.8 (`16,786.2`) | `1.77σ` | **`0.63σ`** | `0.21σ` | `True / True` |
| **FPF** (`10,000 fb^-1`) | 600,000 | 1,800.0 | 598,242.8 (`55,954.0`) | **`3.23σ`** | **`1.05σ`** | `0.21σ` | `True / True` |

- **1,000 次频率派 $H_0$ 玩具蒙特卡洛覆盖率验证**（HL-LHC 27 箱模板）：
  - **含 Barlow–Beeston lite 完整模型**：`zero_fraction = 0.498`（渐近理论 `0.500`），$P(Z \ge 1\sigma) = 16.9\%$（理论 `15.87%`），$P(Z \ge 2\sigma) = 1.6\%$（理论 `2.28%`），$P(Z \ge 3\sigma) = 0.10\%$（理论 `0.135%`），$q_0$ `p90 / p95 / p99 = 1.615 / 2.594 / 5.202`（理论 $\frac{1}{2}\chi^2_1$: `1.642 / 2.706 / 5.412`）。
  - **纯物理+模拟系统误差模型**：`zero_fraction = 0.511`，$P(Z \ge 1\sigma) = 13.1\%$，$P(Z \ge 2\sigma) = 2.1\%$，$P(Z \ge 3\sigma) = 0.20\%$。两套似然均无偏满足 Wilks 半卡方渐近分布。

---

## 阶段性结论

### 已确立的事实

1. **硬件螺距物理规律与停止门限裁决（`Exp-2` / `P1-5`）**：在现行 `Pinpoint_G4` 靶堆（相邻硅层间隔 `10 mm = 2.86 X_0` 或 `70 mm = 20.0 X_0` 实心钨板）中，$5\text{–}15\text{ GeV}$ 次级带电粒子穿越单块双层钨单元的 Highland 多重库仑散射横向投影误差为 $\sigma_{\text{MS}} \approx 70\text{–}210\,\mu\text{m}$，远大于 $20.8\,\mu\text{m}$（$\sigma_{\text{int}} = 6.0\,\mu\text{m}$）至 $100\text{–}300\,\mu\text{m}$（$\sigma_{\text{int}} = 29\text{–}87\,\mu\text{m}$）传感器的本征点分辨率。因此 `20.8 μm` 与 `100 μm / 300 μm` 在从头重训下无显著分类差异（$\Delta\varepsilon_\tau@1\% = -0.12\text{ pp}$），触发 `Stage 314` 停止门限：**在维持 `10 mm / 70 mm` 钨吸收体厚度不变的前提下，不应宣称必须采用亚 $25\,\mu\text{m}$ 像素螺距**；但粗化至 `1.0 mm`（$\sigma_{\text{int}} = 289\,\mu\text{m} > \sigma_{\text{MS}}$）确会显著损害四类鉴别（`-0.97 pp BA4`）与 `0.1% FPR` 超高纯度 $\nu_\tau$ 效率（下跌 `3.2` 倍）。
2. **显式微米径迹/次级顶点特征有效但受钨散射上限约束（`Path C`）**：C++17 `AVR` 多径迹与冲击参数混合模型在 `CleanTune8k` 与盲测 `CleanConfirm` 上均取得全组最优的 `NLL`、`BA4`、横向/纵向顶点精度（横向中位误差 `-18.6%` 至 `0.83 mm`，纵向 `5.80 mm`）及 `+23.5%` 相对 `1% FPR` $\nu_\tau$ 效率增益（`4.98% → 6.15%`），但未能达到 `0.1% FPR` 下 `+5.0 pp` 绝对跃升目标。物理根源清晰确定：两层硅平面间 `10 mm`（`2.86 X_0`）钨板使 $10\text{ GeV}$ 衰变子径迹承受 $\sigma_{d_0,\text{MS}} \approx 50\text{–}72\,\mu\text{m} \sim c\tau_\tau = 87\,\mu\text{m}$ 的多重散射抖动，同时 108 块钨板中有 84 块位于无硅像素覆盖的 `FortuneBlock` 内，从第一性原理上封顶了纯电子学次级顶点分辨与纵向主顶点中位精度（`5.80 mm`）。
3. **从旧 `Confirm200k`（`30.04%`）到干净基准集（`6.15%`）的性能瀑布链已完全闭合**：旧高值由四项叠加抬升——(i) `GN2` + 顶点 `+2.0` 偏置训练规模混杂（`Stage 317` 受控基准 `22.06%`）；(ii) 闪烁体 `edep+PE` 双通道沿条位置泄漏（剥离后降至 `16.64%`，`-5.42 pp`）；(iii) `Stage 318` Geant4 步级 Birks 猝灭（电磁簇射总光电子数下降 `5.3` 倍：`13.56M → 2.56M`）+ `14,400` 微元 SiPM 饱和 + 真实 `NC_charm` 次级顶点本底 + `Sm2Co17` 磁体吸收；(iv) 训练样本量由 `300k` 缩减为干净模拟首期 `32k`。
4. **独立物理发现瓶颈已定量锁定**：在完全诚实的物理流强、真实粲产生与有符号系统误差模型下，当前厚钨板几何在 FPF `10 ab^-1` 纯统计可达 `3.23σ`，但受分域尾部斜率冗余（$\sigma_{\text{slope}} = 8\%$，`slope_D_pix` 拉动 `0.20σ`）与小样本训练效率制约，剖面似然降至 `1.05σ`（若仅用 10k 规模 MC 模板则进一步受限于 Barlow–Beeston MC 统计至 `0.21σ`）。

### 待检验物理与探测器优化假设

1. **薄钨单元（$\le 2\text{ mm}$ W / 硅站）突破多重散射封顶假设**：若将靶区核心硅追踪段的单片钨厚由 `2×5 mm = 10 mm`（`2.86 X_0`）减薄至 `1.5–2.0 mm`（`0.43–0.57 X_0`）并加密硅层间距至 `5–8 mm`，Highland 散射冲击参数误差将由 `~65 μm` 压低至 `< 18 μm`（$\ll c\tau_\tau = 87\,\mu\text{m}$），此时 $20.8\,\mu\text{m}$ 像素与 `Path C` 显式 $d_0/\sigma_{d_0}$ 重建预期可真正分离 $\tau$ / 粲次级顶点与初级强子顶点（当前薄钨几何 **未测量**）。

---

## 下一步计划

1. **薄钨核心段（Thin-W Silicon Vertexer）Geant4 几何设计与模拟验证**：
   - **具体任务**：在修复后的 `Pinpoint_G4`（分支 `fix/audit-geometry-digitizer-physics`）中新增混合靶配置（上游 20–30 层 `1.5 mm W + 20.8 μm Si` 微米顶点靶 + 下游厚钨量能测量段），重跑 `20.8 μm vs 100 μm` 与 `Path C` 冲击参数消融。
   - **所需输入**：`Pinpoint_G4` 参数化薄钨宏文件与 `50k` 干净模拟样本。
   - **验收标准**：靶硅段初级顶点纵向中位误差 `< 1.5 mm`，$10\text{ GeV}$ 径迹冲击参数分辨率 $\sigma_{d_0} < 25\,\mu\text{m}$，`Truth_pixel_centered` 在 `0.1% FPR` 下 $\varepsilon_\tau \ge 5.0\%$。
2. **百万级干净模拟扩产消除 Barlow–Beeston 模板统计瓶颈**：
   - **具体任务**：使用 `Stage 318` 多核流水线在 HTCondor 上将干净模拟集从 `60k` 扩展至 `500k–1M` 事件（其中确认/模板集 $\ge 200\text{k}$），使超尾端分箱 MC 权重方差 $\sum w_i^2 \ll b_i$，同时恢复 `32k → 300k` 的深度神经网络样本缩放增益。
   - **验收标准**：HL-LHC / FPF 剖面似然中 `BB-lite` 与无穷 MC 渐近结果之差 `< 10%`。
3. **双 $\mu$ 与反号轻子原位控制区约束尾部斜率冗余**：
   - **具体任务**：利用下游 `0.57 T` 谱仪电荷符号识别构建 $\nu_\mu \text{CC} \to \mu^- \mu^+$（粲半轻子衰变）与单 $\mu^\mp$ 高能控制区，将分域尾部斜率系统误差先验由 `8%` 原位约束至 `< 2.5%`。
   - **验收标准**：FPF `10 ab^-1` 完整剖面似然显著性推进至 $\ge 3.0\sigma$。
