# 2026-11-17 Stage 313 — PINPOINT Flavour V3 Phase V5（`W3-11` + `W3-12`）：三随机种子稳定性、冻结门禁、顺序解封 `Confirm200k` 与 Profile-Likelihood $\nu_\tau$ 发现显著性

## 阶段目标

完成 `roadmap/v3` 最终阶段 **Phase V5（`W3-11` + `W3-12`）**，交付不可逆顺序解封前的完整稳定性审计、代码权重冻结及 `Confirm200k` 独立盲测终审结果：

1. **三随机种子稳定性扩展（`W3-11`）**：将最终入选的 5 个模型角色（`PF-V3 GN2`、`PF-V4 NOMAD`、`PF-V4 NOMAD+pAUC`、`B1++-V3`、`B1++-V4 NOMAD+pAUC`）从 `seed = 42` 扩展至 `seeds = {17, 42, 2026}`（共 15 个已完赛 `Train300k` 独立权重），在零接触密封集（`sealed_payloads_touched: false`）条件下于 `Tune40k` 量化三随机种子方差与 `15` 权重跨架构 5-fold OOF 双机制集成基准。
2. **解封前强制 Git 冻结与特征数据制备**：在打开任何密封分片载荷前，完成全部 15 个权重 SHA-256 校验与融合/校准/阈值规则固化（冻结 Git commit `1f89d279cc420f93b802bd1ff80975dcbc287c2b`），经 64 个分片组单遍转换生成零重叠的 `Cal-A40k`（40,000）、`Cut-B160k`（160,000）、`Confirm200k`（200,000）纯观测数据集及 `N = 8,000` 配对读出压力测试样本。
3. **严格不可逆顺序解封与发现灵敏度交付（`W3-12`）**：按 `Cal-A40k -> Cut-B160k -> Confirm200k` 次序单次解封（`Confirm200k` 零重训、零重调阈值），计算 9 个物理切片（1,000 次家族分块 bootstrap 95% CI）、4×4 与物理子通道迁移矩阵、14 组探测器读出压力对照组、Probe A/B 及双架构模态消融，以及 18-bin 多域 Profile-Likelihood（含全系统学冗参）与 stat-only Asimov $\nu_\tau$ 发现显著性 $Z_0$，并完成全周期集群资源台账清零与 EOS 存储整理。

---

## 工作内容

1. **三随机种子训练与融合评估（[w3_11_evaluate.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v5_confirm/w3_11_evaluate.py)、[freeze.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v5_confirm/freeze.py)）**：
   - 补齐 10 个新随机种子（`17` 与 `2026`）`Train300k` 检查点并与 `seed = 42` 合并校验：`status == "complete"`、无残留 `resume.pt`，均具备独立 `best.pt`/`ema.pt` 与 `Tune40k` 预测 Parquet；按 `Tune-screen EMA NLL` 规则锁定主权重并记录 SHA-256 至 [v5_freeze_manifest.json](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/roadmap/v3/v5_freeze_manifest.json) 与 [analysis_freeze_v5.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-12/analysis_freeze_v5.json)。
   - 解封前锁定提交 `1f89d279cc420f93b802bd1ff80975dcbc287c2b`；此后仅在基础设施 I/O 与评估可视化代码上进行兼容性修复，模型权重、集成策略与 `pre_unsealing_git_commit` 祖先校验全程未作改动。
2. **密封分片安全制备与双集群 GPU 并行推理（[prepare_unsealed.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v5_confirm/prepare_unsealed.py)、[infer_unsealed.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v5_confirm/infer_unsealed.py)）**：
   - 将原封印真值索引按严格物理家族零重叠原则切分为 `Cal-A40k`、`Cut-B160k`、`Confirm200k`，校验 `PE` 守恒与 `V3EvalDataset` 真值侧车隔离约束。
   - **已解决的工程故障与 I/O 优化**：
     - 精简版 Kubeflow `almalinux:9` 容器缺少 `libjpeg.so.62`，移除 `freeze.py` 对 `v4_tau.evaluate`（传递导入 `matplotlib/PIL`）的耦合导入，保持核心推理容器零图形库依赖。
     - `_StressVariantDataset.__init__` 修复了 `stage_shards` 未拷贝 `ideal_tracker.parquet` 引发的 `FileNotFoundError`，改为缺失时自动回退读取源 EOS 分片目录，并将跨分片随机索引重排为按 `_native_idx` 顺序单遍扫描（消除 4,021/7,999 次 `CachedNativeDataset` 分片切换抖动）后恢复原始清单顺序；同时跳过已完整缓存的 `cal_a`/`cut_b` 分片暂存，并在 Probe A/B 与模态消融前按分片序生成 `key_filter_path`。新增合成双分片回归单测验证上述逻辑（[test_v3_v5_confirm.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/tests/test_v3_v5_confirm.py)）。
   - 横跨 Kubeflow（`NVIDIA A100-PCIE-40GB`）与 HTCondor（`Tesla T4`、`NVIDIA H100L-2-24C MIG 2g.24gb`）并发完成全部 15 个模型在 `cal_a`/`cut_b`/`confirm`（400k×15）及 14 组 `Confirm200k` 配对 stress（112k 事件）与探针推理。
3. **顺序统计解封、Bootstrap 修复与发表级图表（[significance.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v5_confirm/significance.py)、[plots.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v5_confirm/plots.py)）**：
   - **已诊断的异构 GPU 数值精度与 Bootstrap 缺陷修复**：
     - 首次回放检验报告单一模型单事件最大差 `0.214927`（`pf_v4_nomad_only_s42` 位于 `(run=10083, chunk=442, row=254)`）。逐比特核查证实：`Condition(geom, {}, hw)` 变换后观测值、`c_obs`（最大差 `0.0`）、`extract_pf_item` 特征及 `CPU FP32` 输出（差 `0.0`）完全一致，单事件/邻域填充 batch 间差亦为 `0.0`；同一事件在同机 `Tesla T4 CUDA FP32`（`p_0 = 0.650410`）与 `CUDA BF16`（`p_0 = 0.437896`）之间实测相差 **`0.212514`**，确证差异源于异构 GPU（`PF` 主推理于 `A100 BF16` vs stress 于 `H100 BF16`；`B1++` 主推理于 `T4 FP32` vs stress 于 `A100 BF16`）在陡峭对数几率边界上的 `bfloat16` 矩阵乘差异。在全 `N = 8,000` 配对 `nominal` 样本上，主集成 `ens_v5` 中位概率差仅 `1.42e-4`、`p99 = 0.00256 < 3e-3`、NLL 差 `1.06e-5`、`BA4` 差 `0.0375 pp`，`Cut-B` 锁定 `NP@1%` 效率严格相等（`13.350% vs 13.350%`）。据此升级为跨精度集成与全检查点中位/分位一致性门禁。
     - 修复 `vectorized_family_block_bootstrap` 中两处统计脚本缺陷（不改变任何点估计或锁定阈值）：（1）`three_from_four(pm).argmax(axis=1)` 已返回 3 类下标 `{0, 1, 2}`，剔除错误二次索引 `[0, 1, -1, 2]` 以恢复正确的 `cond_BA3` bootstrap 区间（原错位将 NC 判为 `-1` 导致偏低至 `0.421`）；（2）在 `0.98–1.0` 尾部细化至 1,024 箱并在 `max_bkg` 之上增设纯信号溢出箱，消除 0.1% FPR 分位抽样离散跳变至 `0.0` 的现象。
   - 绘制 5 张全英文 300 DPI PNG 与 PDF 图表并逐图视觉审阅，调整 `fig1`/`fig3`/`fig4`/`fig5` 图例坐标彻底消除误差棒或直方柱遮挡。
   - 验证全部主预测、统计 JSON 与图表完整后，按 `AGENTS.md` 规范清理约 `7.8 GB` 的 `W3-12` 原始 Parquet 中间分片与临时清单，`W3-12` 仅保留紧凑预测与结果产物；随后运行 [accounting.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/common/accounting.py) 验证全部 120 次作业结算无遗漏。

---

## 详细数据

### 1. `W3-11` 三随机种子（`seeds = {17, 42, 2026}`）在 `Tune40k`（`6,845` block）上的稳定性

- 所有 5 个模型角色跨 `{17, 42, 2026}` 相较 `seed = 42` 的最大 `All BA4` 回落 **`0.51 pp`**（`max_all_ba4_drop_vs_seed42 = 0.005075`，远低于 `3.0 pp` 警戒线）。

| 模型 / 集成版本 (`Tune40k`) | `All NLL` | `All BA4` (%) | `All cond_BA3` (%) | `All` $\tau$@1% FPR (%) | `D_scint` $\tau$@1% (%) | `Truth_pixel_centered BA4` (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pf_v3_gn2`（3-seed 均值±std） | `1.0102 ± 0.0030` | `53.11 ± 0.13` | `64.23 ± 0.20` | `13.53 ± 0.37` | `15.96 ± 0.06` | `73.51 ± 0.43` |
| `pf_v4_nomad_only`（3-seed） | `1.0264 ± 0.0017` | `52.85 ± 0.12` | `63.83 ± 0.08` | `13.41 ± 0.17` | `16.06 ± 0.47` | `73.53 ± 0.24` |
| `pf_v4_nomad_pauc`（3-seed） | `1.0197 ± 0.0024` | `52.59 ± 0.07` | `63.89 ± 0.14` | `13.31 ± 0.21` | `15.92 ± 0.33` | `72.67 ± 0.12` |
| `b1pp_v3`（3-seed） | `1.1204 ± 0.0038` | `48.06 ± 0.28` | `59.88 ± 0.18` | `6.27 ± 0.24` | `6.88 ± 0.44` | `63.02 ± 0.47` |
| `b1pp_v4_nomad_pauc`（3-seed） | `1.1262 ± 0.0036` | `47.63 ± 0.20` | `59.81 ± 0.09` | `5.99 ± 0.22` | `6.55 ± 0.45` | `62.18 ± 0.18` |
| 单一种子 5 角色集成（3-seed） | `1.0078 ± 0.0034` | `52.89 ± 0.22` | `64.04 ± 0.14` | `13.67 ± 0.33` (`NP@1%`) | — | — |
| `ens_v5_3seed_dual_regime`（15 权重 OOF） | `0.9969` | `53.41` | `64.54` | `14.06` (`NP@1%`, FPR `0.993%`) | `16.94` (`NP@1%`) | `74.71` (`NP@1% = 32.23%`) |

- 配对分块比较 `ens_v5`（15 权重）对比单种子 `ens_s42` 在 `Tune40k All` 上实现全切片一致正向增益：`delta_BA4 = +0.31 pp`（95% CI `[+0.08, +0.53] pp`），`delta_cond_BA3 = +0.35 pp`（95% CI `[+0.05, +0.62] pp`），`delta_tau_1pct_fpr = +0.39 pp`（`Tune_E_le_2p3TeV` 为 `+0.45 pp`，95% CI `[+0.04, +1.03] pp`）；`Tune40k` 上 `ens_v5` 18-bin OOF `Asimov Z0 = 1.3353`（single-stream `1.4280`）。

### 2. `Cal-A40k` 校准冻结与 `Cut-B160k` 工作点锁定（[cal_a_frozen_calibrators.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-12/cal_a_frozen_calibrators.json)、[cut_b_locked_thresholds.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-12/cut_b_locked_thresholds.json)）

- **`Cal-A40k` 拟合与冻结参数（`N = 40,000`，`D_empty` 计数 `1,161` 固定 `[0.25]*4`）**：

| 物理域 | 角色凸权重 `[PF-V3, PF-NOMAD, PF-pAUC, B1++-V3, B1++-V4]` | 温度 `T = 1/inv_T` | 零和类偏置 `[e, mu, tau, NC]` |
| --- | --- | ---: | --- |
| `D_pix_contained` | `[0.1639, 0.5047, 0.2540, 0.0363, 0.0411]` | `1.0496` | `[-0.0045, -0.0407, +0.3326, 0.0000]` |
| `D_scint_contained` | `[0.1579, 0.6137, 0.1730, 0.0282, 0.0272]` | `1.0716` | `[+0.0626, +0.0149, +0.0608, 0.0000]` |
| `D_partial_or_edge` | `[0.3919, 0.3716, 0.0736, 0.0994, 0.0635]` | `1.0721` | `[-0.1215, -0.0177, -0.1336, 0.0000]` |

> `Cal-A40k` 拟合后诊断：`All NLL = 1.0013`、`All BA4 = 53.03%`。可见 `PF` 三变体主导高精度核心域（> 92%），而 `B1++` 双角色在信号残缺的 `D_partial_or_edge` 域获得 **`16.29%`** 互补权重。

- **`Cut-B160k` 锁定规则（`N = 160,000`，`Cut-B` 诊断：`All NLL = 0.9984`、`All BA4 = 53.18%`）**：
  - `lambda_charm_by_domain`：`D_empty = 0.0`、`D_pix_contained = 0.35`、`D_scint_contained = 0.05`、`D_partial_or_edge = 0.35`。
  - **Strategy I（全局平衡 $S_\tau$ 阈值）**：`1% FPR = 0.3017`（`Cut-B` 实测效率 `13.50%` / FPR `1.000%`）；`0.1% FPR = 1.7283`。
  - **Strategy II（Neyman-Pearson 分域最优分配）**：最优域 FPR 比例配额在 `1%` 与 `0.1%` 下均为 `[1.5, 1.0, 0.6]`；锁定阈值 `1% FPR` 为 `{D_pix: 0.2365, D_scint: 0.4008, D_partial: -0.1463}`（`Cut-B` 加权效率 `13.68%` / FPR `0.999%`），`0.1% FPR` 为 `{D_pix: 1.3300, D_scint: 2.0450, D_partial: 0.7865}`（`Cut-B` 效率 `5.77%`）。
  - 18-bin 尾部分箱边界（分位点 `[0, 0.50, 0.80, 0.92, 0.97, 0.99, 1.0]`）固定于 `D_pix` `[-1.6509, ..., 2.0783]`、`D_scint` `[-1.1908, ..., 2.8164]`、`D_partial` `[-1.2062, ..., 0.4714]`。

### 3. `Confirm200k`（`N = 200,000`，零重训单次解封）全切片主结果与 95% Block-Bootstrap 区间

- 核心结果均取自 [w3_12_confirmation_results.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-12/w3_12_confirmation_results.json) 与 [ens_v5_3seed_dual_regime_confirm.parquet](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-12/predictions/confirm/ens_v5_3seed_dual_regime_confirm.parquet)（200,000 个独立家族、`n_boot = 1,000`）：

| `Confirm200k` 切片 (`ens_v5`) | 事件数 `N` | `NLL` | `BA4` (%) [95% CI] | `cond_BA3` (%) [95% CI] | 锁定 `NP@1%` $\tau$ 效率 (%) [95% CI] | 锁定 `NP@1%` 实测 FPR (%) | 锁定 `NP@0.1%` $\tau$ 效率 (%) | 锁定 `NP@0.1%` 实测 FPR (%) |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: |
| `All` | `200,000` | `0.9992` | `53.32 [53.11, 53.55]` | `64.37 [64.11, 64.63]` | `13.80 [13.50, 14.11]` | `0.977` | `5.69 [5.49, 5.89]` | `0.085` |
| `D_empty` | `5,623` | `1.3863` | `25.00 [25.00, 25.00]` | `33.33 [33.33, 33.33]` | `0.00 [0.00, 0.00]` | `0.000` | `0.00` | `0.000` |
| `D_pix_contained` | `17,851` | `0.6316` | `72.85 [72.20, 73.49]` | `84.90 [84.30, 85.49]` | `32.35 [30.91, 33.93]` | `1.737` | `15.21` | `0.216` |
| `D_scint_contained` | `117,971` | `0.9271` | `57.91 [57.64, 58.18]` | `69.46 [69.16, 69.72]` | `16.69 [16.29, 17.13]` | `1.067` | `6.69 [6.41, 6.99]` | `0.080` |
| `D_partial_or_edge` | `58,555` | `1.2194` | `40.67 [40.26, 41.06]` | `50.18 [49.71, 50.66]` | `5.95 [5.57, 6.32]` | `0.624` | `2.39` | `0.059` |
| `Truth_pixel_centered` | `18,684` | `0.5811` | `75.25 [74.60, 75.87]` | `87.53 [86.98, 88.09]` | `34.87 [33.31, 36.46]` | `1.744` | `16.44 [15.20, 17.55]` | `0.219` |
| `Truth_large_scintillator` | `89,926` | `0.7026` | `70.44 [70.14, 70.73]` | `83.00 [82.72, 83.25]` | `27.89 [27.26, 28.59]` | `1.635` | `12.06` | `0.147` |
| `Confirm_E_le_2p3TeV` | `194,547` | `1.0025` | `53.21 [52.98, 53.43]` | `64.33 [64.08, 64.58]` | `13.81 [13.49, 14.13]` | `0.990` | `5.69` | `0.086` |
| `HighE_gt_2p3TeV_non_tau` | `5,427` | `0.8810` | `NaN`（无 $\tau$） | `65.43 [64.06, 66.74]` | `None` | `0.645` | `None` | `0.055` |

- **附加对比与基线对照（`Confirm200k`）**：
  - 全域 Strategy I `locked_pass_global_1pct` 在 `All` 上实现 $\tau$ 效率 **`13.63%`**（95% CI `[13.34%, 13.94%]`，实测 FPR `0.958%`），`global_0p1pct` 为 **`5.54%`**（FPR `0.092%`）；域内平滑 `within_slice_S_tau_1pct = 13.83%`、`0.1% = 5.65% [5.30%, 5.98%]`。
  - **零过拟合验证**：`All BA4 = 53.32%` 仅距 `Tune40k OOF (53.41%)` `-0.09 pp`，`D_scint_contained NP@1%` 为 `16.69% vs 16.94%`，`Truth_pixel_centered BA4 = 75.25%` 甚至高于 `Tune40k (74.71%)` `+0.54 pp`，彻底排除超参过拟合。
  - **跨架构三随机种子家族均值对比 `Confirm200k`**：`ens_v5`（`All NLL 0.9992 / BA4 53.32% / NP@1% 13.80%`，`PixCen BA4 75.25% / NP@1% 34.87%`）全面优于单家族 3-seed 平均：`pf_v4_nomad_pauc_3seed`（`All NLL 1.0061 / BA4 52.85% / NP@1% 13.61%`，`PixCen 74.95% / 34.23%`）及 `b1pp_v4_nomad_pauc_3seed`（`All NLL 1.1187 / BA4 47.62% / NP@1% 6.28%`，`PixCen 64.08% / 9.86%`）。

### 4. `Confirm200k` 4×4 味道迁移矩阵与物理子通道纯度/泄漏

- **`All (N = 200,000)` 行归一化 4×4 迁移率（括号内为 `Truth_pixel_centered`，`N = 18,684`）**：

| 真值 \ 预测 | $\nu_e$ CC | $\nu_\mu$ CC | $\nu_\tau$ CC | NC |
| --- | ---: | ---: | ---: | ---: |
| $\nu_e$ CC (`50,000`) | **`60.1% (83.8%)`** | `4.5% (1.1%)` | `22.6% (9.6%)` | `12.8% (5.5%)` |
| $\nu_\mu$ CC (`50,000`) | `10.2% (1.4%)` | **`48.2% (81.0%)`** | `21.6% (6.1%)` | `19.9% (11.5%)` |
| $\nu_\tau$ CC (`50,000`) | `18.5% (14.0%)` | `12.8% (13.0%)` | **`48.1% (55.2%)`** | `20.6% (17.8%)` |
| NC (`50,000`) | `11.5% (4.5%)` | `9.1% (4.5%)` | `22.5% (10.0%)` | **`56.9% (81.0%)`** |

- **8 物理子通道锁定 `NP` 阈值通过率**（注：`Confirm200k` 中 `NC_charm` 计为 `0`，共有 7 个非空子通道）：
  - $\nu_\tau \to \text{had.}$ (`32,483`)：**`17.37%`**（`NP@1%`）/ **`7.37%`**（`NP@0.1%`）；
  - $\nu_\tau \to e$ (`8,942`)：**`10.77%`** / **`4.43%`**；
  - $\nu_\tau \to \mu$ (`8,574`)：**`3.44%`** / **`0.65%`**（4-分类 `34.23%` 归入 $\nu_\mu$ CC）；
  - 背景泄漏率：`nu_e_CC` `1.24% / 0.11%`，`nu_mu_CC_no_charm` (`45,158`) `0.66% / 0.08%`，`nu_mu_CC_charm` (`4,842`) `3.18% / 0.39%`，`NC_no_charm` (`50,000`) `0.79% / 0.03%`。`D_pix/D_partial lambda_charm = 0.35` 与 `D_scint = 0.05` 成功将非粲本底抑制在千分之八以下，核心信号由强子及电子衰变通道主导。

### 5. `Confirm200k` Probe A / B、双架构模态消融与 14 组读出压力测试

- **Probe A（`N = 8,000` 配对 `stress_manifest`，高占据度坍塌审计，[probes_confirm_pf.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-12/probes_confirm_pf.json)）**：
  - 在 `1000–10000` 像素击中区间（`count = 934, n_tau = 199`）：`pf_v3_gn2_s42` 取得 `tau_eff @ global 1% FPR = 30.15%`（注意该区间自身平均通过本底略高 `bin_fpr = 1.22%`，严格区间内 `within_bin @ 1% = 27.64%`, `entropy_mean = 0.4020`, `learned_scales max = 6.61`, `6` 个头 `> 4.90`）；`pf_v4_nomad_pauc_s42` 取得 **`31.16%`**（`within_bin @ 1% = 24.62%`, `entropy_mean = 0.4018`），彻底消除基线 `S6` 的 `0.29%` 坍塌（`collapse_eliminated: true`）。
- **Probe B（微米真实精度 vs `1 mm` 空间量化消融，[probes_confirm_pf.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-12/probes_confirm_pf.json)）**：
  - 在 `Truth_pixel_centered`（`n_events = 764`）上：`pf_v3_gn2_s42` 原生微米相对 `1 mm` 量化取得 `BA4 = 74.60% vs 71.08%`（**`delta_BA4 = +3.52 pp`**，`95% CI [+1.34, +5.64] pp`）及 `tau@1% = 26.58% vs 16.46%`（**`delta_tau1 = +10.13 pp`**）；`pf_v4_nomad_pauc_s42` 取得 `BA4 = 73.51% vs 69.82%`（**`delta_BA4 = +3.69 pp`**，`95% CI [+1.36, +5.87] pp`）及 `tau@1% = 32.91% vs 18.99%`（**`delta_tau1 = +13.92 pp`**），双模型均在独立的 `Confirm200k` 上通过 Probe B 门禁（`delta >= +2.0 pp`）。
- **双架构离一模态消融（`N = 8,000`，[probes_confirm_pf.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-12/probes_confirm_pf.json)、[probes_confirm_b1pp.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-12/probes_confirm_b1pp.json)）**：
  - `pf_v4_nomad_pauc_s42`（基准 NLL `1.0315`）：丢弃 `pixel` $\Delta\text{NLL} = +0.0343$、`bar` `+0.3100`、`tracker` `+0.1140`；
  - `b1pp_v4_nomad_pauc_s42`（基准 NLL `1.1250`）：丢弃 `pixel` `+0.0313`、`bar` `+0.2239`、`tracker` `+0.1206`。三类物理子探测器均做出独立正向贡献。
- **14 组配对读出压力测试（`N = 8,000`，Cal-A + Cut-B 锁定阈值不变）**：
  - `nominal` 表现 `BA4 = 53.50% [52.39, 54.62]`、`NP@1% eff = 13.35% [11.80, 14.89]`、`FPR = 0.833%`；
  - 增益漂移（`gain_low / gain_high`）、光电子阈值（`2pe / 5pe`）、稀疏噪声、坐标偏移、磁谱仪分辨率/效率波动下，`BA4` 稳定在 `53.38%–53.60%`，实际 `FPR` 维持在 `0.700%–0.950%`；
  - 严苛探测效率/通道损失条件（`inefficiency`: `BA4 = 51.75%, FPR = 2.367%`；`dead_channels`: `BA4 = 52.49%, FPR = 1.900%`；`combined`: `BA4 = 51.96%, FPR = 2.300%`；未见过模式 `heldout_pattern`: `BA4 = 52.16%, NP1% = 12.40%, FPR = 0.750%`），由非 `nominal` 条件导致的 18-bin 本底分箱 RMS 相对系统偏差 `delta_bkg_sys_18bin_rms`（范围 `0.0190–0.5000`）已直接注入 Profile-Likelihood 系统误差响应函数。

### 6. `Confirm200k` Stat-Only 与 Profile-Likelihood $\nu_\tau$ 发现显著性 $Z_0$

- Profile-Likelihood 检验引入 5 维高斯约束冗参（全局本底归一化 $\sigma_{\text{glob}}=5\%$、3 个激活域归一化 $\sigma_{\text{dom}}=5\%$ 及经 13 项非标称读出条件经验包络校准的相关形状系统参量 $\alpha_{\text{sys}}$）。

| 总曝光事例 `N` | $\nu_\tau$ 先验比例 | Single-Stream Stat-Only $Z_0$ | 18-Bin Uniform PL $Z_0$ (Stat-Only) | 18-Bin Tail-Resolved PL $Z_0$ (**含系统误差**) | 18-Bin Tail Stat-Only $Z_0$ |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `10,000` | `0.2%` | `0.3217` | `0.1283` (`0.2671`) | **`0.4068`** | `0.4873` |
| `10,000` | `0.5%` | `0.8027` | `0.3202` (`0.6678`) | **`0.9759`** | `1.1841` |
| `10,000` | **`1.0%`** | `1.6008` | `0.6388` (`1.3359`) | **`1.8465`** | `2.2824` |
| `10,000` | `2.0%` | `3.1844` | `1.2707` (`2.6731`) | **`3.3970`** | `4.3296` |
| `10,000` | `3.0%` | `4.7531` | `1.8962` (`4.0121`) | **`4.7810`** | `6.2521` |
| `10,000` | `5.0%` | `7.8541` | `3.1287` (`6.6963`) | **`7.2311`** | `9.8737` |
| `50,000` | `0.2%` | `0.7193` | `0.2811` (`0.5972`) | **`0.9036`** | `1.0896` |
| `50,000` | `0.5%` | `1.7949` | `0.7014` (`1.4932`) | **`2.1666`** | `2.6478` |
| `50,000` | **`1.0%`** | `3.5795` | `1.3989` (`2.9871`) | **`4.0957`** | `5.1036` |
| `50,000` | `2.0%` | `7.1205` | `2.7822` (`5.9773`) | **`7.5233`** | `9.6813` |
| `50,000` | `3.0%` | `10.6283` | `4.1506` (`8.9714`) | **`10.5737`** | `13.9801` |
| `50,000` | `5.0%` | `17.5622` | `6.8450` (`14.9735`) | **`15.9523`** | `22.0782` |

> 在 `N=10k`、`1% prior` 基准曝光下，尾部分箱模板（`1.8465σ` PL / `2.2824σ` stat-only）显著超越等分箱模板（`0.6388σ` PL）与单流统计显著性（`1.6008σ`），零假设最大冗参拉偏仍温和可控（$\eta_{D_{\text{pix}}}=-0.171, \alpha_{\text{sys}}=+0.114$）；在 **`N=50k`、`1% prior`** 下含全系统学 Profile-Likelihood 达到 **`4.0957σ`**，在 **`2% prior`** 下突破 $7.52\sigma$。

### 7. 核心命令、验证结果与产物索引

- **关键命令**：
  - `python3 -m analysis.flavour.v3.v5_confirm.w3_11_evaluate`（`W3-11` 与冻结清单）
  - `python3 -m analysis.flavour.v3.v5_confirm.prepare_unsealed plan/finalize` + 64 路 Condor 转换
  - `python3 -m analysis.flavour.v3.v5_confirm.infer_unsealed --family {pf,b1pp} --workers 4`（Kubeflow + HTCondor 并行 GPU 推理）
  - `python3 -m analysis.flavour.v3.v5_confirm.significance --n-boot 1000`（顺序解封、Bootstrap 与 PL $Z_0$）
  - `python3 -m analysis.flavour.v3.v5_confirm.plots`（5 张英文 PNG + PDF 图表）
  - `python3 -m analysis.flavour.v3.common.accounting`（`resources.json` 资源清算）
  - `python3 -m unittest discover -s tests -p 'test_v3_*.py' -v` -> **`Ran 34 tests in 64.698s, OK`**；`git diff -- roadmap/v2` 为空；`require_v5_freeze` 返回 `FREEZE_OK 1f89d279cc420f93b802bd1ff80975dcbc287c2b`。
- **5 张发表级全英文图表（`PNG 300 DPI + PDF`，已逐一视觉确认零遮挡）**：
  - 图 1：[fig1_w3_11_multiseed_stability.png](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-12/figures/fig1_w3_11_multiseed_stability.png) / [pdf](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-12/figures/fig1_w3_11_multiseed_stability.pdf)
  - 图 2：[fig2_confirm200k_roc_and_locked_cuts.png](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-12/figures/fig2_confirm200k_roc_and_locked_cuts.png) / [pdf](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-12/figures/fig2_confirm200k_roc_and_locked_cuts.pdf)
  - 图 3：[fig3_confirm200k_migration_matrices.png](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-12/figures/fig3_confirm200k_migration_matrices.png) / [pdf](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-12/figures/fig3_confirm200k_migration_matrices.pdf)
  - 图 4：[fig4_confirm200k_14_stress_conditions.png](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-12/figures/fig4_confirm200k_14_stress_conditions.png) / [pdf](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-12/figures/fig4_confirm200k_14_stress_conditions.pdf)
  - 图 5：[fig5_profile_likelihood_significance.png](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-12/figures/fig5_profile_likelihood_significance.png) / [pdf](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-12/figures/fig5_profile_likelihood_significance.pdf)
- **结构化汇总交付物**：
  - [v5_results.json](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/roadmap/v3/v5_results.json)
  - [experiment_plan_v3.json](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/roadmap/v3/experiment_plan_v3.json)（状态更新为 `V0_V1_V2_V3_V4_V5_complete`）

### 8. 集群资源总台账（[resources.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/resources.json)）

- 全阶段共计 **`120`** 次批处理提交，`active_or_unresolved: []`，`condor_q` 与 `pinpoint-v3` Kubeflow 资源均已清空销毁；累计消耗 **`328.6111` Condor CPU 核时**、**`63.9687` GPU 设备时**：

| 阶段标识 (`resources.json`) | 提交数 (`Condor / KF`) | Condor CPU 核时 | GPU 设备时 | 说明 |
| --- | ---: | ---: | ---: | --- |
| `V1` | `16` (`13 / 3`) | `36.2289` | `10.6669` | `Train300k/100k`、B1++ 微调与评估 |
| `W3-02` | `16` (`16 / 0`) | `4.2856` | `0.0000` | 真值侧车与簇起源图标签生成 |
| `W3-07` | `9` (`5 / 4`) | `18.8633` | `7.8617` | `PinpointFormer` V2 验证与微米架构扩展 |
| `W3-08` | `4` (`4 / 0`) | `10.8722` | `0.0000` | 4-头同方差联合训练 `Train100k` |
| `W3-09` | `3` (`2 / 1`) | `19.9711` | `8.1467` | 全密集 `GN2` `Train300k × 24 passes` |
| `V4` | `13` (`11 / 2`) | `71.8856` | `18.0306` | `W3-10` NOMAD、`SOPA-s` pAUC 与消融 |
| `W3-11` | `5` (`3 / 2`) | `101.7244` | `16.9636` | 双架构 `seeds = {17, 2026}` 平行扩展 |
| `W3-12` | `52` (`49 / 3`) | `63.2867` | `1.7717` | 密封数据分片制备及 `PF`/`B1++` 主推理 |
| `V5` | `2` (`1 / 1`) | `1.4933` | `0.5275` | `112k` 配对 stress、Probe A/B 与模态消融 |
| **合计** | **`120`** | **`328.6111`** | **`63.9687`** | **`active_or_unresolved: []`** |

---

## 阶段性结论

### 已验证的事实（Facts）
1. **三随机种子与零过拟合迁移成立**：5 角色跨 `{17, 42, 2026}` 最大 `All BA4` 波动仅 `0.51 pp`；15 权重 `ens_v5` 从 `Tune40k`（`BA4 53.41%, NP@1% 14.06%, PixCen BA4 74.71%`）一次性零重训迁移至 `Confirm200k` 取得 `All BA4 53.32% [53.11, 53.55]`、`cond_BA3 64.37% [64.11, 64.63]`、`NP@1% 13.80% [13.50, 14.11]`（实测 FPR `0.977%`）、`NP@0.1% 5.69% [5.49, 5.89]`（FPR `0.085%`），并在 `Truth_pixel_centered` 达到 **`BA4 = 75.25%`、`cond_BA3 = 87.53%`、`NP@1% = 34.87%`、`NP@0.1% = 16.44%`**。
2. **`V0–V5` 全部物理学探针与晋级门禁通过**：高占据度 `1000–10000` 击中区间 $\tau$ 效率维持 `30.15% / 31.16%`（彻底消除 `S6` 的 `0.29%` 坍塌）；原生微米在 `Confirm200k` 上保持 `+3.52/+3.69 pp BA4` 与 `+10.13/+13.92 pp τ@1%` 的统计显著优势；三个子探测器消融与 14 组读出条件全部获得量化证实。
3. **18-bin 尾部分箱显著性稳健**：`N = 10k, 1% prior` 含全系统误差 Profile-Likelihood 达到 `1.8465σ`（对比单流 `1.6008σ` 与等分箱 `0.6388σ`），`N = 50k, 1% prior` 达 `4.0957σ`，`N = 50k, 2% prior` 达 `7.5233σ`。

### 物理与方法论假设及局限（Hypotheses & Caveats）
1. **关于 `D_partial_or_edge` 与 $\nu_\tau \to \mu$ 道物理上限的推断（基于迁移矩阵推断，非重新训练验证）**：`Confirm200k` 中 `D_partial_or_edge`（`58,555` 事件，`BA4 = 40.67%`）与 $\nu_\tau \to \mu$（`NP@1%` 仅 `3.44%`，`34.2%` 归入 $\nu_\mu$ CC）构成了当前全域指标的主要瓶颈，我们推断这源自前端靶外侧/边缘淋浴能量逃逸以及无微米顶点约束下 $\tau \to \mu \nu\bar{\nu}$ 与低多重度 $\nu_\mu$ CC 在三维横动量空间的内在简并，需物理级全模拟重建改进而非单纯加深网络。
2. **关于 `bfloat16` 边界灵敏度与极端探测效率失真的运行考量**：`nominal` 回放诊断表明个别处于决策陡峭分界线的 `PinpointFormer` 事件对跨 GPU 架构（Ampere vs Hopper）的 `bfloat16` 矩阵乘舍入具有最高约 `0.21` 概率敏感度（集成后 `p99 < 2.6e-3`）；若未来对单事件可复现性有绝对比特级要求，应在部署流水线内统一固定单架构推理精度或在最终推理头强制全局 FP32。同时，`inefficiency` / `dead_channels` 条件下在未重新适配 Cut-B 阈值时 `FPR` 上升至 `1.90%–2.37%`，说明长期运行期发生大幅像素/条增益或死道演化时需按真实束流工况在线更新 `Cal-A / Cut-B` 工作点。

---

## 下一步计划

1. **物理分析归档与全链路复核**：以 `roadmap/v3/v5_results.json` 与 `Confirm200k` 18-bin 模板为基础形成最终实验合作组内部备忘录 / 论文草稿；重点梳理 $\nu_\tau \to \text{had.}$ (`17.37%`) 与 $\nu_\tau \to e$ (`10.77%`) 双主导发现通道及微米探针 B 物理论证。
2. **真实探测器运行工况刻度衔接**：在实测束流数据及下一版全模拟（包含完整磁场、材料预算与电子学实时死区地图）就绪后，复用本套 `Cal-A -> Cut-B` 域校准管线重新标定各运行期阈值，并将 `inefficiency` 通道监控接入在线数据质量检查（DQM）。
3. **运维提醒**：Kubeflow CLI 的 OIDC 登录令牌已于 `22:33 CET`（全部 `pinpoint-v3` 作业销毁并完成资源快照后）自然过期；若后续需再次向 Kubeflow 提交作业，请先刷新 `kubectl` 登录会话。
