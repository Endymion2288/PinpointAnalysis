# Stage 309: Roadmap v3 Phase V1 — ZAlignedStereoNet (B1++) 强立体基线、多后端并发训练与 M1 基准评估 (`W3-04`–`W3-05`)

## 阶段目标

完成 `roadmap/v3/` Phase V1（`W3-04`–`W3-05`），在零泄露纯观测量体系（Phase V0）基础上实现传感器原生无空洞纵向对齐架构 `ZAlignedStereoNet (B1++)`，并回答五项核心验收目标：

1. **W3-04（架构实现、物理对称性与 32 事件健康门禁）**：验证 `15 pixel + 54 X_bar + 42 Y_bar = 111` 个物理 $z$ 排序敏感面无空洞编码、基于正确偶极磁场（$\mathbf{B} \parallel \hat{x}$）的 $y$-$z$ 偏转矢高 $s_y$ 与双站回退特征、绕机械中心 $y_{\text{scint\_center}} = -12.0\text{ mm}$ 的精确 `Y` 镜像等变性，以及 `32-event` 真实数据在 100 updates 内 `NLL < 0.05`、`Acc == 100%` 且 7 个物理分支梯度均非零有限。
2. **W3-05 主门禁（Truth `large_scintillator` `BA3 >= 73.0%`）**：全部 6 组 `seed=42` 配置收敛训练（`Train100k x 16 passes` 与 `Train300k x 24 passes`），在未用于超参筛选的独立验证集与全 `Tune40k` 上突破 `R1 B1 (68.89%)`，跨过 `>= 73.0%` 晋级门禁并完成 Collaboration M1 同指标对照（`Wu 2026`、`Mahlstedt 2026`、`R1 B1`、`R2 B1-D2`）。
3. **公共能量支撑一致性**：验证 `Tune_E_le_2p3TeV`（$N=38,874$）与全量 `Tune40k` 指标同步上升，检查 `energy_match_reweight` 对高频尾端 `HighE_gt_2p3TeV_non_tau`（$N=1,123$）误判率的影响。
4. **泛化间隙与收敛轨迹诊断**：量化 `Train/Tune` NLL 泛化间隙与模态消融净收益，识别 `B1++` 的性能边界以指导 Phase V2 `PinpointFormer`。
5. **Kubeflow 与 HTCondor 资源闭环**：协同使用两端异构 GPU 集群（含单 GPU 多进程并发），确保 `resources.json` 清算完结且无残留运行作业。

---

## 工作内容

1. **无空洞传感器原生表示与精确物理对称增强（[representation.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v1_b1pp/representation.py)、[spectrometer.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v1_b1pp/spectrometer.py)）**
   - 废弃旧版将 `10.5 mm` 宽闪烁体条插入 `1322×420` 画布造成稀疏零空洞的做法，改用物理紧凑矩阵：$X_{\text{bar}} \in \mathbb{R}^{6 \times 54 \times 40}$、$Y_{\text{bar}} \in \mathbb{R}^{6 \times 42 \times 40}$（`ch0..5` 为 $\ln(1+E_{\text{edep}})/10$、$\ln(1+N_{\text{PE}})/15$、$z_{\text{norm}}$、累积辐射长度 $\Delta X_0^{\text{cum}}/50$、`active_mask` 及带符号归一化横向坐标 $u_{\text{coord}}$），以及双尺度紧凑硅像素矩阵 $P^{\text{fine}}_v \in \mathbb{R}^{5 \times 15 \times 512}$（$\pm 128\text{ mm}$，步长 `0.50 mm`）与 $P^{\text{wide}}_v \in \mathbb{R}^{5 \times 15 \times 256}$（$\pm 140\text{ mm}$，步长 `1.09 mm`）。
   - 下游磁谱仪（[spectrometer.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v1_b1pp/spectrometer.py)）在物理偏转平面 $y$-$z$ 提取 3-station 三元组矢高 $s_y = y_1 - (y_0 + y_2)/2$、$\sigma_{s_y} = \sqrt{3/2}\,\sigma_y$、入口倾角 $\theta_y^{\text{entry}}$ 及向观测簇射起点 `cal_start` 反推残差 $(\Delta x_{\text{ext}}, \Delta y_{\text{ext}})$ 构成 14 维候选向量；同时实现绕 $y_c = -12.0\text{ mm}$ 机械中心的精确 `Y` 镜像增强（[apply_y_mirror_to_item](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v1_b1pp/representation.py#L252-L277)）。

2. **关键物理结构修复：2D 立体共线卷积与双站磁谱仪回退（[models.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v1_b1pp/models.py)）**
   - 早期初版（归档于 `W3-05/ablation_1d_transverse_v1/`）采用仅沿横轴 $W$ 卷积的 `kernel_size=(1, 3)` 并在逐层进入 `1D z-mixer` 前对 $W$ 全局池化，丢失了跨平面的 $(Z, W)$ 横向位置偏移关联，且未对仅有 2 站径迹击中的事例生成候选特征：`Train100k x 16p T3`（`3,013,398` 参数）最高 `All BA3 = 57.32%`、`best_screen EMA NLL = 0.8302`（`No Tracker` 仅 `52.49%`、`NLL = 0.8919`）。
   - 将 [TransverseResBlock](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v1_b1pp/models.py#L15-L34) 与 `stem` 升级为保持 $N_z$ $1:1$ 不变的 2D `(3, 3)` 立体卷积，向输入网格显式注入横向坐标通道 `u_coord`，在逐层注意力池化中补充加权横向重心与展宽 `(u_mean, u_std)` 及 2D 全局共线读出 `stereo_2d_proj`（参数量 `T4 = 3,124,023`, `T3 = 3,123,766`，处于 `3.6M` 预算内）；并在 `SpectrometerEncoder` 中为仅有 `(0,2)/(0,1)/(1,2)` 两站击中的穿出粒子建立线性斜率与簇射起点外推回退候选特征（权重标记 `0.5`）。升级后同配置 `T3 Full`、`T3 No Pixel` 与 `T3 No Tracker` 的 `best_screen EMA NLL` 立即降至 `0.8101 / 0.8029 / 0.8583`，`All BA3` 提升至 `58.35% / 58.78% / 54.57%`。

3. **多后端异构 GPU 并发训练与评估清算（[train.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v1_b1pp/train.py)、[submit.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/common/submit.py)、[evaluate.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v1_b1pp/evaluate.py)、[accounting.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/common/accounting.py)）**
   - 在 **Kubeflow** 队列使用 `NVIDIA H100 NVL MIG 3g.47gb` 训练全量主力 `b1pp_300k_24p_t4`（`w305-kf-300k-v3`，`bs=256`，单 pass `186.1 s`，总墙钟 `4803.0 s`）；此前亦在 `w305-kf-t4-v2`（`H100 3g.47gb`）与 HTCondor `1583049.0`（`H100L MIG 2g.24gb`）验证单 GPU `--parallel 2` 双进程共享 `/tmp` mmap 缓存运行模式。
   - 在 **HTCondor (`bigbird20.cern.ch`)** 并行提交 5 路 `H100L MIG 1g.12gb`（每作业申请 `4 CPUs / 15 GB` 精确匹配空闲 `1g.12gb` 槽位，避免内存超配排队），完成 `b1pp_100k_16p_t4`（`1583055.0`）、`b1pp_100k_16p_t4_ematched`（`1583056.0`）、`b1pp_100k_16p_t3`（`1583057.0`）、`b1pp_100k_16p_t3_no_pixel`（`1583058.0`）与 `b1pp_100k_16p_t3_no_tracker`（`1583054.0`）；全部 6 项训练步长 1 梯度审计均满足非 dropped 分支正有限梯度与 `drop_modality` 分支精确零梯度（无泄露）。
   - 执行 400 次 `(run, chunk)` 物理块自助法与 5 折 family-connected OOF 分区域校准，输出完整英文图表、M1 对比表及 `resources.json`。

---

## 详细数据

### 1. W3-04 交互 GPU 32 事件过拟合与吞吐健康门禁

命令：`python3 -m analysis.flavour.v3.v1_b1pp.train --smoke-32`（设备：`Tesla T4`，`SM75`，`FP32`）  
报告路径：[w3_04_health_report.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-04/w3_04_health_report.json)

- **`T4`（`3,124,023` 参数）**：第 `8` 次参数更新即通过门禁，`NLL = 0.0442`、`Acc = 1.0000`、`distinct_probs = 31/32`，峰值显存 `625.1 MiB`；7 个物理分支（`bar_transverse`, `pixel_transverse`, `window_readout`, `longitudinal_mixer`, `spectrometer`, `cobs`, `head`）首步梯度范数全正有限（最小 `cobs = 0.0815`，最大 `head = 1.8334`）。
- **`T3`（`3,123,766` 参数）**：第 `11` 次更新通过门禁，`NLL = 0.0451`、`Acc = 1.0000`，峰值显存 `625.4 MiB`；7 分支首步梯度全正有限。
- **`bs=64` 基准吞吐（T4 FP32）**：`280.64 events/s`（`228.05 ms/step`），峰值显存 `2,308.6 MiB`（双进程并发估算显存 `4,617.2 MiB`）。

### 2. W3-05 Collaboration M1 对照基准（Tune40k，`best_screen` 主检查点，`seed=42`）

产物：[evaluation_results.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-05/evaluation_results.json)、[m1_collaboration_table.csv](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-05/m1_collaboration_table.csv)  
（注：每个 `B1++` 运行的主检验权重统一选取 `Tune4k` 筛选集最低 EMA NLL 对应的 `best.pt`；括号内为 400 次 `(run, chunk)` 块自助法 95% 置信区间；另对剔除 `tune_screen` 全部抽样键后的独立留出集 `Tune_non_screen` ($N=32,428$，含 `Large` $N=14,544$) 复核。）

| 模型 | 切片 / 域 | `BA3` (95% CI) | `NLL3` | `BA4` | $\epsilon_\tau @ 1\%$ raw / reg $S_\tau$ |
| --- | --- | ---: | ---: | ---: | ---: |
| `FORTVNE Wu 2026 (ref)` | `Truth_large_scintillator` | `75.0%` | — | — | — |
| `FORTVNE Mahlstedt 2026 (ref)` | `All` / `Truth_large_scintillator` | `56.6%` / `69.0%` | — | — | — |
| `R1 B1 T3 (4p, 100k)` | `All` | `56.62%` | `0.8720` | — | — |
| `R1 B1 T3 (4p, 100k)` | `Truth_pixel_centered` | `73.82%` | `0.6218` | — | — |
| `R1 B1 T3 (4p, 100k)` | `Truth_large_scintillator` | `68.89%` | `0.7061` | — | — |
| `R2 B1-D2 T4 (4p, 300k)` | `All` | `57.13%` | `0.8490` | `45.77%` | `6.05%` / `6.16%` |
| `R2 B1-D2 T4 (4p, 300k)` | `Truth_pixel_centered` | `74.42%` | `0.5995` | `58.66%` | `7.40%` / `10.30%` |
| `R2 B1-D2 T4 (4p, 300k)` | `Truth_large_scintillator` | `70.09%` | `0.6736` | `56.62%` | `10.81%` / `11.48%` |
| **`b1pp_300k_24p_t4` (p14)** | `All` ($N=40,000$) | **`59.95%`** `[59.34, 60.50]` | **`0.7983`** | **`48.24%`** | `6.54%` / `6.30%` |
| **`b1pp_300k_24p_t4` (p14)** | `Obs_D_pix_contained` ($N=3,618$) | **`77.08%`** `[75.48, 78.55]` | `0.5306` | `61.61%` | `9.50%` / **`12.62%`** |
| **`b1pp_300k_24p_t4` (p14)** | `Truth_pixel_centered` ($N=3,765$) | **`79.18%`** `[77.69, 80.49]` | **`0.5017`** | **`63.52%`** | `10.57%` / **`13.61%`** |
| **`b1pp_300k_24p_t4` (p14)** | **`Truth_large_scintillator`** ($N=18,002$) | **`74.11%`** `[73.34, 74.83]` | **`0.5916`** | **`60.26%`** | `12.11%` / `12.28%` |
| **`b1pp_300k_24p_t4` (p14)** | `Tune_non_screen_large` ($N=14,544$) | **`74.05%`** `[73.26, 74.86]` | `0.5968` | `60.08%` | `12.19%` / `12.43%` |
| **`b1pp_300k_24p_t4` (p14)** | `Tune_E_le_2p3TeV` ($N=38,874$) | `59.99%` `[59.33, 60.56]` | `0.7962` | `48.22%` | `6.54%` / `6.30%` |
| `b1pp_300k_24p_t4` (p14) | `HighE_gt_2p3TeV_non_tau` ($N=1,123$) | `60.34%` `[57.48, 63.14]` | `0.8249` | — | FPR: `2.67%` / `2.40%` |
| `b1pp_100k_16p_t4` (p10) | `All` / `Pix` / `Large` | `58.36%` / `76.58%` / `71.83%` | `0.8293` / `0.5559` / `0.6392` | `46.25%` / `60.24%` / `57.57%` | `4.84%`→`4.89%` / `4.23%`→`8.59%` / `8.99%`→`9.60%` |
| `b1pp_100k_16p_t4_ematched` (p12) | `All` / `Pix` / `Large` | `58.12%` / `77.16%` / `71.68%` | `0.8320` / `0.5556` / `0.6419` | `46.28%` / `60.82%` / `57.53%` | `5.24%`→`5.06%` / `5.42%`→`8.06%` / `9.41%`→`9.57%` |
| `b1pp_100k_16p_t3` (p12) | `All` / `Pix` / `Large` | `58.35%` / `77.21%` / `71.65%` | `0.8300` / `0.5461` / `0.6358` | — | — |
| `b1pp_100k_16p_t3_no_pixel` (p14) | `All` / `Pix` / `Large` | `58.78%` / `77.83%` / `72.42%` | `0.8217` / `0.5335` / `0.6242` | — | — |
| `b1pp_100k_16p_t3_no_tracker` (p12) | `All` / `Pix` / `Large` | `54.57%` / `73.46%` / `67.01%` | `0.8734` / `0.6161` / `0.7027` | — | — |

#### 合作组 `AUROC` 与固定纯度效率补充对齐（`Tune40k`, `e / mu / NC`）

- **`b1pp_300k_24p_t4`（`Truth_large_scintillator`）**：`AUROC = 0.9313 / 0.9081 / 0.8643`（对比 `R1 B1 T3`: `0.9106 / 0.8593 / 0.8211`）；`90% purity` 选择效率达 **`55.40% / 59.87% / 29.03%`**（对比 `R1 B1 T3`: `45.82% / 32.05% / 24.43%`，其中 $\nu_\mu$ 效率近翻倍 `+27.82 pp`）；`80% purity` 效率达 `77.78% / 69.67% / 45.81%`。
- **`b1pp_300k_24p_t4`（`Truth_pixel_centered`）**：`AUROC = 0.9437 / 0.9427 / 0.9004`（对比 `R1 B1 T3`: `0.9203 / 0.9089 / 0.8644`）；`90% purity` 效率达 **`61.72% / 77.43% / 41.34%`**（对比 `R1 B1 T3`: `59.35% / 52.76% / 29.09%`）；分区域校准后全域多域 Asimov `Z_0 = 1.191`，在全域 `0.1% FPR` 下 `Pixel-centered` 与 `Large` $\nu_\tau$ 效率分别为 `3.70%`（raw `3.17%`）与 `3.75%`（raw `3.72%`）。

### 3. 三端点对比（`best_screen` vs `ema_final` vs `exposure_final`）与模态消融净收益

| 实验名称 | 硬件与耗时 | 最佳 pass | `best_screen` (`All / Pix / Large BA3`) | `ema_final` (`All / Pix / Large BA3`) | `exposure_final` (`All / Pix / Large BA3`) | 末轮 NLL gap |
| --- | --- | ---: | --- | --- | --- | ---: |
| `b1pp_300k_24p_t4` | `H100 3g.47gb` / `4803.0s` / `7561 MiB` | `14` | `59.95% / 79.18% / 74.11%` | `58.19% / 77.96% / 72.47%` | `58.22% / 78.09% / 72.54%` | `+0.2016` |
| `b1pp_100k_16p_t4` | `H100L 1g.12gb` / `4048.7s` / `3832 MiB` | `10` | `58.36% / 76.58% / 71.83%` | `58.06% / 76.88% / 71.69%` | `57.81% / 76.89% / 71.42%` | `+0.0816` |
| `b1pp_100k_16p_t4_ematched` | `H100L 1g.12gb` / `4025.8s` / `3832 MiB` | `12` | `58.12% / 77.16% / 71.68%` | `57.39% / 76.65% / 71.04%` | `57.15% / 76.64% / 70.70%` | `+0.0738` |
| `b1pp_100k_16p_t3` | `H100L 1g.12gb` / `3262.5s` / `3832 MiB` | `12` | `58.35% / 77.21% / 71.65%` | `57.52% / 76.51% / 70.89%` | `57.18% / 76.25% / 70.55%` | `+0.0991` |
| `b1pp_100k_16p_t3_no_pixel` | `H100L 1g.12gb` / `3218.9s` / `3830 MiB` | `14` | `58.78% / 77.83% / 72.42%` | `58.76% / 78.12% / 72.54%` | `58.88% / 78.16% / 72.62%` | **`+0.0270`** |
| `b1pp_100k_16p_t3_no_tracker` | `H100L 1g.12gb` / `3308.4s` / `2818 MiB` | `12` | `54.57% / 73.46% / 67.01%` | `54.15% / 73.56% / 66.58%` | `54.12% / 73.08% / 66.65%` | `+0.0830` |

- **配对消融差异（`T3 Full - Ablation`, 95% 块自助法 CI）**：
  - **去掉下游磁谱仪（`tracker_drop_delta`）**：全域损失 `+3.79 pp` `[+3.19, +4.36]`、`Large` 损失 `+4.64 pp` `[+3.89, +5.36]`（`Large NC` 召回率从 `66.72%` 跌至 `49.12%`）、`Pixel-centered` 损失 `+3.75 pp` `[+2.23, +5.10]`、`HighE > 2.3 TeV` 损失高达 **`+8.31 pp`** `[+5.43, +11.63]`。
  - **去掉二维像素网格分支（`pixel_drop_delta`）**：在 `Train100k` 规模下，保留高维稠密 4 路二维像素分支在 `All`（`-0.43 pp [-0.79, -0.08]`）与 `Large`（`-0.77 pp [-1.29, -0.38]`）上反而略低于 `No Pixel`（见结论剖析）；在 `HighE > 2.3 TeV` 呈弱正差 `+1.11 pp [-0.80, +2.96]`。

### 4. 三张英文分析图表与资源清算

- Collaboration M1 对照与模态消融：[w3_05_m1_benchmark_and_ablations.pdf](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-05/figures/w3_05_m1_benchmark_and_ablations.pdf)（[PNG](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-05/figures/w3_05_m1_benchmark_and_ablations.png)）
- 分域 $\nu_\tau$ 效率与高能尾段能谱审计：[w3_05_tau_slices_and_spectrum.pdf](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-05/figures/w3_05_tau_slices_and_spectrum.pdf)（[PNG](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-05/figures/w3_05_tau_slices_and_spectrum.png)）
- 验证 NLL 曲线与泛化间隙轨迹：[w3_05_training_dynamics_and_gap.pdf](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-05/figures/w3_05_training_dynamics_and_gap.pdf)（[PNG](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-05/figures/w3_05_training_dynamics_and_gap.png)）
- 资源核算文件：[resources.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/resources.json)（`status = complete`，`active_or_unresolved = []`，总提交 `32` 条）：
  - **HTCondor (`bigbird20.cern.ch`)**：总 CPU 核时 `40.51 core-hours`（含 `W3-02` `4.29`、`V1` `36.23`）、GPU 设备时 `7.63 GPU-hours`（含早期 `1583049.0` 单卡双进程及全部终态作业）。
  - **Kubeflow (`ml.cern.ch`)**：总 GPU 设备时 `3.04 GPU-hours`（其中主力 `w305-kf-300k-v3` 消耗 `1.38 GPU-hours`，历史实验与快照审计见 `resources.json`），两后端合计 `10.67 GPU-hours`。

---

## 阶段性结论

1. **V1 晋级门禁通过，全量关键指标全面超越 R1/R2 强基线（事实）**：`b1pp_300k_24p_t4` 在 `Truth_large_scintillator` 上取得 `BA3 = 74.11%` `[73.34%, 74.83%]`（独立非筛选子集 `74.05%` `[73.26%, 74.86%]`），超过 `>= 73.0%` 门禁线并较 `R1 B1 T3 (68.89%)` 提升 `+5.22 pp`、较 `R2 B1-D2 (70.09%)` 提升 `+4.02 pp`，逼近合作组 `FORTVNE Wu 2026 (~75%)`；同时 `Truth_pixel_centered` 达到 `BA3 = 79.18%`、`BA4 = 63.52%`（高于 `B0 HGB 55.92%` 达 `+7.60 pp`），经 OOF 区域校准后 1% FPR 下 $\epsilon_\tau$ 达到 **`13.61%`**（`Obs_D_pix_contained` 亦达 **`12.62%`**）。
2. **能谱公共支撑一致性成立（事实）**：`Tune_E_le_2p3TeV`（$E_\nu \le 2.3\text{ TeV}$）在 `300k` 模型上的 `BA3 = 59.99%`、`BA4 = 48.22%` 与全量 `Tune40k`（`59.95% / 48.24%`）保持一致；对高能尾部 `HighE_gt_2p3TeV_non_tau`（$N=1,123$），使用 OOF 区域统计量 $S_\tau$ 将 1% 全局切割下的高能误判为 $\tau$ 比例由未校准 `2.67%` 降至 `2.40%`（`100k_ematched` 与 `100k_unweighted` `BA4` 基本相当，分别为 `46.43%` vs `46.29%`）。
3. **2D 立体共线卷积累积与双站磁谱仪显著修复 $\nu_\mu$/NC 鉴别（事实）**：对比 1D 横向池化版与完整 2D `(3,3)` + 双站回退版，`T3 All BA3` 提升 `+1.03 pp`、`No Pixel` 提升 `+1.49 pp`、`No Tracker` 提升 `+2.08 pp`；磁谱仪分支在全模型上对 `Large BA3` 贡献 `+4.64 pp`，尤其使无下游三站全击中的 `NC` vs $\nu_\mu$ 分辨在主模型取得 `70.42%`（`90% purity` 下 $\nu_\mu$ 效率由 `R1 B1` 的 `32.05%` 翻倍至 `59.87%`）。
4. **稀疏二维像素网格末期记忆化是 `B1++` 的主要局限（事实与假设）**：
   - **事实**：去掉 4 路稠密二维像素编码的 `b1pp_100k_16p_t3_no_pixel` 全程 16 passes 后 `gap` 仅 `+0.0270`，且在 `All / Pix / Large` 均反超完整 4 分支 `b1pp_100k_16p_t3`（末轮 `EMA` 更明显：`72.54%` vs `70.89%`）；含二维像素网格的所有模型均在 `pass 10–14` 达到验证最优后出现 `gap` 迅速攀升（`300k_24p_t4` 从 `pass 14` 的 `+0.0475` 升至 `pass 24` 的 `+0.2016`，加之 `EMA(0.999)` 有效平均半衰期约 $693$ 步仅覆盖半个 pass，无法抑制过拟合）。
   - **假设**：硅像素击中仅约 `9.4%` 事例显著集中于像素核心区（在 `Train100k T3` 仅约 `7k` 例），且绝大多数电磁/强子簇射离散散布；将 15 层各 `512/256` 宽稀疏 2D 网格替换为基于 `0.06 mm` 连通域聚类提取的微米原生特征集合与 `PinpointFormer`（Phase V2）可从根源消除空白像素网格噪声记忆化，并将高颗粒度径迹/次级顶点几何释放为净增益。

---

## 下一步计划

推进至 **Phase V2（`W3-06`–`W3-07`，`PinpointFormer` 微米拓扑与可微顶点锚定架构）**：

1. **`W3-06`**：实现 `src/analysis/flavour/v3/v2_pinpointformer/`（`pixel_tokenizer.py` 含 `16` 维微米矩 + `32×32` 大簇微卷积并核对与 `ClusterSidecar` `cluster_id` 一致性、`stereo_bars.py` `NuGraph2` 正交双视图条耦合、`vertex_anchor.py` 两阶段可微顶点极坐标重中心化与近顶点 `IP_perp / dtheta_kink` 偏置、`attention.py` 温度可学习 `ScaledQKNormAttention`、`models.py`），通过 `tests/test_v3_pinpointformer.py` 与 `32-event` 交互 GPU 过拟合闸门。
2. **`W3-07`**：运行单因素消融对比（`PF_V2_ablation_no_anchor` vs `PF_V2_anchored`，`Train100k x 16 passes`, `seed=42`），并交付两项强制物理诊断探针（`src/analysis/flavour/v3/common/probes.py`）：
   - **探针 A（高占据度 $\tau$ 效率曲线与注意力熵）**：消除 `[1000, 10000]` 击中区间效率断崖。
   - **探针 B（微米原生 vs `1.0 mm` 量化）**：在 `Truth_pixel_centered` 达到 `delta_BA4 >= +2.0 pp` 且 `delta_tau_eff_1pct >= +2.0 pp`，并在 `Pixel-centered BA4` 严格超越 `B0 HGB (55.92%)`。
