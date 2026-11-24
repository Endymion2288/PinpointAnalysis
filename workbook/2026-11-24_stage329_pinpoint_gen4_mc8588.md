# Stage 329 报告：Pinpoint-Gen4 全栈架构、DiffPhys-SBI、PinpointSSL-v2 与官方 MC 10085-10088 独立盲法审计

## 阶段目标

1. 在严格对齐官方 `Pinpoint_G4` MC `10085-10088`（`scintHits.edep` 与 `faserHits.{px,py,pz,energy,charge,pdg}` 零读取、`14,400 PE` 饱和与 `< 3 PE` 时序门限、`54` 个非均质取样站与 `12` 层硅像素契约）的前提下，构建隔离包 `src/analysis/flavour/gen4/`，验证四项核心物理-机器学习范式在 `Train32k` / `SSL_320k` / `Tune8k`（`Tune_Screen 2,000` + `Tune6k 6,000`）上的定量增益与边界：
   - 物理前端与对称性感知联合重建主干 `PinpointGen4Net`（`BackConeSlabPrior` + `AnisoPTv3` + `CompetitiveSlotNDIVE` + `DeSatSinkhornTomography` + `IntrinsicFMPE` + `UnifiedMuonSpecEncoder`）；
   - 零学习物理前端 + 样条标准化流后验估计器 `DiffPhysSBIModel`（Rossi-Greisen `FISTA` 簇射剥离 + Broken-Line Kalman 弯折统计量 + `140,111` 参数条件 RQ-Spline 流）；
   - 纯无监督点云表示预训练 `PinpointSSL-v2`（跨 `412,796` 事例、`LeJEPA SIGReg` + 能量加权掩码预测）与无梯度冲突微调（`Domain-CGGM` + `MMPareto` + `MCR` + `UParT13`）；
   - 尾部极端事件校准（`POT-GPD`）与 `4` 黄金微通道 + `3` 控制区 `NSBI-PL`（`Barlow-Beeston lite` 剖面似然与 `200` 次 Toy-MC 闭合）。
2. 独立审计员执行零泄漏测试、盲法零读取扫描、全切片及 `1,000` 次家族配对 Bootstrap 复算、300 DPI 全英文总结图表绘制，并对照冻结基线 `Ref-BDT` 与同数据 3-seed 基线 `Ref-L1-GN2 (+KL-DRO)` 如实给出可测性能与已知物理缺陷。

**盲法声明**：本阶段全程严格遵守盲法封存铁律，对 `Cal-A (chunk % 20 == 16..17)`、`Cut-B (chunk % 20 == 18..19)`、`Confirm (shards/125..149)`、`sealed_truth_meta/` 及废弃样本 `10081-10084` / `v6_clean_v3` 执行 **0 次读取**；所有封存指标均标记为 **未测量**。

---

## 工作内容

1. **基础设施、几何契约与数据缓存（`gen4/geometry_contract.py`, `physics_frontend.py`, `data_cache.py`）**
   - 直接从 `Pinpoint_G4` 几何与样本导出 `geometry_mc8588.json`，实施 `chunk_id % 20 >= 16` 与 `shards/100..149` 静态及运行期封存拦截（`SealAuditGuard` + `assert_unsealed_mc8588_path`）。
   - 提取 `36` 维非均质能量沉积特征与暗噪声屏蔽过滤，严格禁用 `scintHits.edep` 与 `faserHits` 真值动量/电荷/PDG 字段；对 `train32k` 的 `2,560` 个 `NC` 背景事例实施在位极化 $\tau\to\mu$（`854/1,403 = 60.87%` 覆盖）手术以支持 `tau_llr` 尾部分类训练。
2. **模型家族训练与融合（HTCondor GPU + Kubeflow GPU）**
   - **`DiffPhys-SBI`**：完成 `2,000` 事例 `FISTA` 纯度审计、Gate-A/B 梯度与真值不变性审计、Stage-I 验证（`seed=42`, `28 epochs`）及 Stage-II `3-seed`（`17, 42, 2026`）训练与 `Tune_Screen` 温度校准（`T = 0.9392`）。
   - **`PinpointGen4Net` 监督基准与 `Ref-L1-GN2 (+KL-DRO)`**：完成 Stage-I 5 组受控因果消融（`Val 4.8k`, `seed=42`, `12 epochs`）、Stage-II `3-seed` 训练（`16 epochs`）以及同数据集 `Ref-L1-GN2 (+KL-DRO)` `3-seed` 对照训练。
   - **`PinpointSSL-v2` 与尾部联合微调**：于 Kubeflow H100 MIG `3g.47gb` 在 `412,796` 事例上预训练 `PinpointSSL-v2`（Ep 1-3，`val loss: 2.8653 -> 2.5361 -> 2.2679`），完成 Stage-I 7 组标签效率（`10% / 25% / 100%`）与 `Domain-CGGM+MMPareto+MCR` 对照，完成 Stage-II `3-seed` 全链路微调、与 `DiffPhys-SBI` 正交概率融合（`Tune_Screen` 搜索得 `alpha_diffphys = 0.33`，`T = 0.9645`）、`Stress400` / `Diag_OOD`、`POT-GPD` 与 `NSBI-PL` 闭合验证。
3. **独立审计、集成测试、绘图与空间清理**
   - 编写并固化 [test\_gen4\_seal\_and\_integration.py](tests/test_gen4_seal_and_integration.py)，核验 `verify_inference_truth_invariance(..., atol=0.0)`、`scintHits.edep` / `faserHits` 真值零读取、`14,400 PE` 饱和及 `SSL-v2` 权重迁移；在 `bigbird15.cern.ch` 提交最终 `unittest` 作业（`17104727.0`）并通过全套 `28/28` 测试。
   - 运行独立审计脚本 `audit_gen4_independent.py`（作业 `17104726.0`）对全部 `6` 组模型重新执行 `Tune6k` 规范得分 `s_tau = log p_tau - log(p_e + p_mu + p_had)` 与 `tau_llr` 复算、`Mahlstedt BA3`、`POT-GPD (150 splits)`、区域 PV Pull、`1,000` 次家族配对 Bootstrap（`seed=20263250`）及静态封存扫描。
   - 编写 [plot\_gen4\_summary.py](src/analysis/flavour/gen4/plot_gen4_summary.py) 生成 4 幅 300 DPI 纯英文总结图并目视复核无标签重叠与坐标截断；确认零在途作业后清理 `47 GB` `cache/ssl_mc8588/*.pkl` 临时缓存及旧空目录 `stage329_gen4`，将 `stage329_gen4_mc8588` 总占用由 `54 GB` 压减至 `6.5 GB`。

---

## 详细数据

### 1. 样本规模、盲法扫描与硬件测试验证

- **数据划分（`foundation_verify_summary.json`）**：`Train32k = 32,000`（`internal_train = 25,600`、`internal_dev_screen = 1,600`、`internal_val_eval = 4,800`；`charm = 2,371`）；`Tune8k = 8,000`（`tune_screen = 2,000`、`Tune6k = 6,000`，其中 `D1_obs = 744`、`D2_obs = 4,498`、`D3_obs = 758`、`Front30 = 343`、`Interm150 = 371`、`Fortune_ScintOnly = 4,127`、`Mahlstedt_PC = 4,500`、`Common_E_50_2312 = 5,714`、`NonW_tgt_Tune6k = 452`、`Tune6k_Charm_nontau = 310`）；`SSL_320k = 412,796`（`D1 = 62,237`、`D2 = 350,559`）+ `ssl_val = 32,599`；`Stress400 = 400 × 7` 项退化规格；`Diag_OOD`：`InDet_CC_Charm = 2,631`、`NonW_Al_Scint = 3,000`、`HighE_gt1p5TeV = 3,000`，而 `InDet_NC_Charm = 0`（官方 `10088 r == 13` 无含粲事例，诚实保留空切片计数 `0`）。
- **封存状态与测试结果**：
  - 静态扫描 `independent_tune6k_audit.json`：`files_scanned = 89`、`n_sealed_hits = 0`、`all_reported_zero = true`（`16/16` 子日志均报告 `seal_violations = 0`）。
  - 推理零真值泄漏：Gate-A `verify_inference_truth_invariance(..., atol=0.0)` 对 `PinpointGen4Net` 与 `DiffPhysSBIModel` 均测得 `max_abs_diff = 0.0`。
  - 模型健康与参数规模：`DiffPhysSBIModel` `140,111` 参数，`5` 分支梯度全部有限非零（`log_lik_scale=0.1174`、`cond_encoder=0.1722`、`flow_blocks=1.3745`、`prior_head=0.5364`、`vtx_calib_head=0.2515`），`N=64` 过拟合 loss `5.2943 -> 1.5349`（`acc=1.000`）；`PinpointGen4Net` `6,340,256` 参数，`5` 参数组梯度全部有限非零，`N=64` 过拟合 loss `11.4433 -> 11.3438`。
  - 物理前端 `FISTA` 纯度审计（`2,000` 事例）：`3-hit` 线段由 `42,009` 条精简至 `14,412` 条，真实 $\tau$/信号线段纯度由 `1.31%` 升至 `2.72%`（`2.08x`），同时保留 $\tau$ 非电子 MIP 簇召回率 `72.18%`、剥离 EM 簇 `67.14%`。
  - 全套单元测试（最终代码改动后于 `bigbird15.cern.ch` 作业 `17104727.0` 复验）：`python -u -m unittest discover -v -s tests -p "test_gen4_*.py"` -> **`Ran 28 tests in 95.824s — OK`**。
  - **封存分割指标（`Cal-A`、`Cut-B`、`Confirm 10k`）**：**未测量**。

### 2. Tune6k（`N=6,000`）全模型独立复算总表

> 下表均由独立审计脚本在 `Tune6k`（`N=6,000`）上按同一物理切片定义复算；各模型分别报告规范得分 $s_\tau^{\text{canon}} = \log p_\tau - \log(p_e + p_\mu + p_{\text{had}})$ 与 UParT 13-submode 贝叶斯对数似然比 `tau_llr` 两套尾部口径。

| 模型家族（3-seed 集成） | `BA4 (%)` | `NLL` | $\tau@1\%$ (`canon` / `llr`, `%`) | $\tau@0.1\%$ (`canon` / `llr`, `%`) | `D1_obs` (`%`) | `Front30` (`%`) | `Interm150` (`%`) | `Fortune Scint` (`%`) | `Mahlstedt BA3` (`%`) | `Front30 PV d_xy / d_z` (`um`) | `W184 Charm AUROC` (`canon`/`llr`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Ref-BDT` | 50.93 | 1.1068 | 4.33 / 4.33 | 0.33 / 0.33 | 50.46 | 51.72 | 45.25 | 51.08 | 62.69 | 未测量 | 0.5312 / 0.5312 |
| `Ref-L1-GN2+DRO` | **55.13** | 1.0658 | **6.40 / 6.53** | **1.33 / 1.53** | 56.54 | 53.97 | 53.46 | **55.01** | 67.29 | **1,265.9 / 7,345.2** | **0.5315** / 0.5310 |
| `DiffPhys-SBI` | 51.30 | 1.0954 | 3.00 / 3.00 | 0.13 / 0.13 | 54.89 | 52.79 | 48.93 | 50.95 | 63.64 | 38,457.9 / 38,105.5 | 0.5279 / 0.5278 |
| `PinpointGen4_SupOnly` | 53.45 | 1.0803 | 3.40 / 2.87 | 0.13 / 0.33 | 57.40 | 55.49 | 53.30 | 52.39 | 66.98 | 6,633.4 / 17,222.2 | 0.5217 / **0.5509** |
| `PinpointGen4_SSL_Tail` | 54.35 | 1.0506 | 1.27 / 1.07 | 0.07 / 0.00 | **58.18** | **55.93** | **55.75** | 53.47 | **69.04** | 3,601.7 / 16,375.2 | 0.4753 / 0.4841 |
| `PinpointGen4_Full_Blend` (`alpha=0.33`) | 54.68 | **1.0394** | 1.53 / 1.60 | 0.00 / 0.00 | 57.73 | 55.55 | 53.75 | 54.01 | 69.00 | 3,601.7 / 16,375.2 | 0.4903 / 0.4966 |

注：单种子均值（± 标准差）方面，`Ref-L1-GN2+DRO` 为 `BA4 = 53.87 ± 0.63%`, `NLL = 1.1374 ± 0.0085`；`PinpointGen4_SupOnly` 为 `BA4 = 52.15 ± 1.60%`, `NLL = 1.0965 ± 0.0279`；`PinpointGen4_SSL_Tail` 为 `BA4 = 53.87 ± 0.30%`, `NLL = 1.0628 ± 0.0044`。所有切片在 `Tune6k` 共同能量窗口 `Common_E_50_2312 (N=5,714)` 上一致（`Full Blend BA4 = 54.82%`, `NLL = 1.0381`），非钨靶切片 `NonW_tgt_Tune6k (N=452)` 上 `Full Blend BA4 = 50.47%`（`SSL_Tail = 51.48%` vs `Ref-L1-GN2+DRO = 50.02%` vs `Ref-BDT = 46.61%`）。

### 3. 1,000 次家族配对 Bootstrap 95% CI（`seed=20263250`, 规范得分 `s_tau`）

| 对比组合（`A vs B`） | `All Tune6k ΔBA4 (pp)` | `All Tune6k ΔNLL` | `All Tune6k Δτ@1% (pp)` | `D1_obs ΔBA4 / ΔNLL` | `Front30 ΔNLL` | `Interm150 ΔBA4` |
| --- | --- | --- | --- | --- | --- | --- |
| `Full_Blend vs Ref-BDT` | **+3.75** [+2.66, +5.00] | **-0.0674** [-0.0771, -0.0578] | **-2.60** [-4.36, -0.65] | **+7.26** / **-0.1527** | **-0.1053** [-0.1510, -0.0538] | **+8.50** [+4.09, +12.60] |
| `Full_Blend vs Ref-L1-GN2+DRO` | -0.45 [-1.70, +0.83] | **-0.0263** [-0.0424, -0.0104] | **-4.87** [-6.68, -2.49] | +1.19 / **-0.1013** | **-0.1003** [-0.1803, -0.0253] | +0.29 [-4.29, +4.51] |
| `Full_Blend vs DiffPhys-SBI` | **+3.38** [+2.38, +4.41] | **-0.0559** [-0.0646, -0.0474] | -1.47 [-2.80, +0.20] | +2.84 / **-0.1210** | **-0.0973** [-0.1391, -0.0586] | **+4.82** [+0.19, +9.38] |
| `Full_Blend vs Gen4_SupOnly` | **+1.23** [+0.33, +2.17] | **-0.0409** [-0.0475, -0.0343] | **-1.87** [-3.61, -0.19] | +0.32 / **-0.0622** | **-0.0562** [-0.0882, -0.0254] | +0.45 [-2.84, +3.89] |
| `Full_Blend vs Gen4_SSL_Tail` | +0.33 [-0.29, +0.95] | **-0.0112** [-0.0151, -0.0074] | +0.40 [-0.52, +1.90] | -0.46 / +0.0027 | -0.0063 [-0.0224, +0.0089] | -2.00 [-4.75, +0.87] |

### 4. Stage-I 因果消融与 `PinpointSSL-v2` 正向迁移（`internal_val_eval N=4,800`, `seed=42`）

| 实验 / 变体 | `BA4 (%)` | `NLL` | `D1_obs (%)` | `tau_llr@1% (%)` | `tau_mu@1% (%)` | `tau_L<30mm@1% (%)` | `Front30 PV xy (um)` | 关键观察 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `gen4_full` | 52.15 | 1.0860 | 56.76 | **2.92** | **2.70** | **2.73** | 4,754.3 | 完整物理与几何分支；`tau` 尾部与 `D1` 均优于减分支 |
| `abl_binary_no_edep` | 52.63 | 1.0877 | 55.53 | 2.17 | 1.80 | 2.02 | 3,440.8 | 去掉 `36` 维 `edep` 特征导致 $\tau@1\%$ 下降 `0.75 pp` |
| `abl_no_sandwich_no_ndive` | 52.63 | **1.0706** | **58.36** | 2.33 | 2.25 | 2.02 | 6,798.0 | 移除 `NDIVE` 使 `Front30` PV 恶化 `+2,043.7 um`，但 `NLL` 更低 |
| `abl_no_m2_no_kink_stub` | 51.85 | 1.0963 | 54.25 | 2.50 | 1.80 | 2.33 | **2,311.2** | 移除 `M2` 与 kink 桩致 `D1_obs -2.51 pp`, `tau_mu -0.90 pp` |
| `abl_no_desat_sinkhorn_fmpe` | 51.88 | 1.0952 | 53.76 | 1.50 | 1.80 | 1.52 | 3,357.2 | 移除去饱和/Sinkhorn/FMPE 致 $\tau@1\%$ 跌至 `1.50%` |
| `scratch_10pct (n=2,560)` | 45.79 | 1.3594 | 52.14 | 1.67 | 2.70 | 1.42 | 9,323.6 | 10% 标签从零训练基准 |
| `ssl_v2_ft_10pct (n=2,560)` | **47.31** | **1.1734** | **52.48** | 1.00 | 1.35 | 1.01 | 17,909.3 | `SSL-v2`: **`BA4 +1.52 pp`, `NLL -0.1861`**（`probs4@1%: 1.92% vs 1.42%`） |
| `scratch_25pct (n=6,400)` | 47.10 | 1.2553 | 52.12 | 1.08 | 2.25 | 1.01 | 3,164.1 | 25% 标签从零训练基准（落后于 `10% SSL`） |
| `ssl_v2_ft_25pct (n=6,400)` | **50.35** | **1.1306** | **54.79** | **1.75** | **4.50** | **1.72** | 3,838.1 | `SSL-v2`: **`BA4 +3.25 pp`, `NLL -0.1247`, `tau_llr +0.67 pp`** |
| `scratch_100pct (n=25,600)` | 52.27 | 1.0771 | 56.09 | 0.83 | 2.25 | 0.81 | 2,786.6 | 100% 从零训练（含无冲突优化） |
| `ssl_v2_ft_100pct (n=25,600)` | **52.46** | **1.0766** | 55.68 | **1.00** | **4.50** | 0.71 | **2,451.1** | `SSL-v2`: **`BA4 +0.19 pp`, `NLL -0.0005`, `tau_mu 2.0x`** |
| `abl_no_conflict_free_adamw` | 52.54 | 1.0784 | 55.19 | 0.92 | 2.25 | 0.71 | 2,257.4 | 启用无冲突优化提升 `tau_probs4@1% (0.75->1.00%)` 与 `tau_mu` |

### 5. `POT-GPD`、`NSBI-PL`、`9/9 PV Pull`、`Stress400` 与 `Diag_OOD` 审计

- **`POT-GPD` 尾部阈值方差对比（`150` 次 split-half, `n_bg_half=2,250`）**：
  - `Gen4 Full Blend`（`tau_llr` / 规范 `s_tau`）在 `FPR = 0.001` 下，阈值标准差分别缩减 **`3.65x`**（`0.1786 -> 0.0489`，方差缩减 **`13.32x`**）和 **`2.50x`**（方差 `6.25x`），实测 held-out FPR 方差分别缩减 **`2.13x`** 与 **`1.92x`**；`DiffPhys-SBI` 在 `FPR = 0.001` 下实测 FPR 方差缩减 **`2.61x`**；`Ref-L1-GN2+DRO` 亦获 **`2.25x`**（`llr`）/ **`2.44x`**（`canon`）FPR 方差缩减。但须注意在 `Tune6k`（`n_bg=2,250`）上极低目标 `FPR = 0.001` 时，`Gen4 Full Blend` 的 GPD 外推平均实测 FPR 偏高至 `0.00165`（`SSL_Tail` 达 `0.00440`），表明极重尾分布外推仍受限于半折本底数。
- **`NSBI-PL` 黄金微通道发现显著度闭合（`180k` 事例, $\tau$ 份额 `0.3%`, $\delta_{\text{sys}}=1.59\%$）**：
  - `Gen4 Full Blend` 路由：`Golden-3Prong-ZeroMu = 194`（$S=25.2, B=4486.3$）、`Golden-Kink-Mu = 216`（$S=30.6, B=5370.1$）、`Golden-Rho-1Prong = 33`（$S=4.0, B=787.2$）、`Fortune-HighMT-NSBI = 769`（$S=116.6, B=17779.8$）；`3` 原位 CR：`CR-mumu = 1,080`、`CR-1mu = 18`、`CR-pi0 = 1,036`。
  - Asimov $Z_0$ 对比单箱 1% FPR：`Gen4 Full Blend` `Z0_stat_only = 1.2352`（vs `0.1903`, `6.49x`）、`Z0_pl_no_bb = 0.2577`（vs `0.1543`, **`1.671x`**）、`Z0_pl_bb_lite = 0.0869`（vs `0.0251`, **`3.461x`**）；`DiffPhys-SBI` 达到更高绝对显著度：`Z0_stat_only = 1.3080`、`Z0_pl_no_bb = 0.3786`（`1.212x`）、`Z0_pl_bb_lite = 0.0967`（`1.856x`）。
  - `200x` Toy-MC 闭合：`Gen4 Full Blend` `H0 zero_fraction = 0.465`（渐近理论 `0.500`）、$p(Z>1\sigma)=0.180$；`DiffPhys-SBI` `H0 zero_fraction = 0.470`、$p(Z>1\sigma)=0.155$（渐近 `0.159`）。
  - **审计发现的未解决缺陷**：`nsbi_unbinned_pl` 在 `Gen4+SSL` 与 `Gen4 Full Blend` 上剖面似然显著度退化为 `0.0`（`DiffPhys-SBI` 为 `0.4261`），源自 `fit_unbinned_nsbi_pl` 的非零 Asimov 梯度实现缺陷，目前以分箱 `4-SR + 3-CR NSBI-PL` 作为可靠主结论。
- **`9/9` Student-$t(\nu=3)$ PV Pull 区域异质性诊断**：
  - 经 `Tune_Screen` 全局 3 轴缩放校正（`[1.764, 2.110, 1.712]`）后，`PinpointGen4_SSL_Tail` 的 `All_valid` `68% |pull|` 为 `[1.06, 0.96, 1.02]`（均落入 `[0.85, 1.15]`），但分区域严重失配：`Front30` 为 `[1.48, 1.02, 2.55]`（`|dz|=16.38 mm`, `d_xy=3.60 mm`），`Interm150` 为 `[0.73, 0.55, 1.56]`（`|dz|=110.18 mm`, `d_xy=7.99 mm`）。根本物理原因为 `Front30` 存在背向散射 hit 触发 `BackConeSlabPrior`（`15/32 = 47%` 事例前置偏移）且全局常数缩放无法兼容不同纵向取样密度区域；`DiffPhys-SBI` 同样存在 `Front30 x=1.35` 与 `Interm150 [0.83, 0.83, 0.78]` 区域偏差。
- **`Stress400` 与 `Diag_OOD`**：
  - `Gen4+SSL+Tail`（3-seed）：`nominal = 53.50%`，`pixel_dead_2pct = 53.50% (0.00 pp)`、`pixel_dead_5pct = 54.00% (+0.50 pp)`、`sipm_sat_cap_0p95 = 53.75% (+0.25 pp)`、`tracker_dead_2pct_noise1 = 54.00% (+0.50 pp)`、`tracker_dead_5pct_noise2 = 52.00% (-1.50 pp)`、**`pixel_dark_1e-5 = 51.25% (-2.25 pp)`**（超 `1 pp` 稳健阈值，亦优于 `DiffPhys-SBI` 的 `48.50% / -5.50 pp`）。`Diag_OOD`：`InDet_CC_Charm BA4 = 53.67%`、`NonW_Al_Scint BA4 = 51.71%`、`HighE_gt1p5TeV BA4 = 48.66%`（均优于 `DiffPhys-SBI` 的 `49.93%` 与 `45.83%`）。

### 6. 图表与产出路径

- 图表路径（全部为 300 DPI 纯英文矢量级 PNG，存放于 `/eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage329_gen4_mc8588/figures/`）：
  - `fig1_gen4_mc8588_overall_tune6k_and_slices.png`：`Tune6k` 6 模型总览 `BA4 / NLL` 及各探测器区域 + `Mahlstedt BA3` 分解；
  - `fig2_gen4_ssl_scaling_and_ablations.png`：`PinpointSSL-v2` 跨 `10% / 25% / 100%` 标签预算正向迁移曲线与 Stage-I 5 组因果消融；
  - `fig3_gen4_submodes_and_nsbi_z0.png`：$\tau$ 衰变子模式与 $L_\tau < 30\text{ mm}$ `1% FPR` 效率对比 + `4-SR + 3-CR NSBI-PL` $Z_0$ 增益；
  - `fig4_gen4_pv_pulls_and_stress.png`：`9/9` Student-$t$ PV Pull 区域分布与 `Stress400` / `Diag_OOD` 对比。
- 审计与验证产物路径：
  - 独立审计汇总：`/eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage329_gen4_mc8588/verification/independent_tune6k_audit.json`
  - 全套单元测试日志：`/eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage329_gen4_mc8588/verification/unittest_gen4_28of28.txt`
  - 绘图与审计摘要：`/eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage329_gen4_mc8588/verification/audit_plot_summary.txt`
- 主要执行命令：
  - `python -u -m unittest discover -v -s tests -p "test_gen4_*.py"`（HTCondor `bigbird15.cern.ch` `17104727.0`）
  - `python -u /afs/cern.ch/user/x/xcheng/condor/Pinpoint/stage329_unittest/audit_gen4_independent.py && python -u -m analysis.flavour.gen4.plot_gen4_summary`（HTCondor `bigbird15.cern.ch` `17104726.0`）

---

## 阶段性结论

### 事实（直接观测）
1. **盲法与零泄漏成立**：全代码与产物静态扫描 `n_sealed_hits = 0`，`16/16` 训练指标文件报告 `seal_violations = 0`，`verify_inference_truth_invariance(..., atol=0.0)` 实测 `max_abs_diff = 0.0`，`28/28` 单元测试在最终代码修改后全部通过。
2. **`PinpointSSL-v2` 逆转了 Stage 328 JEPA 负迁移**：在官方 MC `10085-10088` 上，无监督预训练在 `10% / 25% / 100%` 标签预算下全面超越从零训练（`+1.52 / +3.25 / +0.19 pp BA4`、`-0.1861 / -0.1247 / -0.0005 NLL`），且 `25% SSL-v2`（`50.35%`）显著超过 `25% Scratch`（`47.10%`）。
3. **跨范式融合刷新全域 `NLL`、高精度像素区 `BA4` 与 `Mahlstedt BA3`**：`PinpointGen4_Full_Blend` 相对 `Ref-BDT` 实现 `BA4 +3.75 pp`、`NLL -0.0674`、`D1_obs +7.26 pp`、`Interm150 +8.50 pp`；相对同数据 `Ref-L1-GN2+DRO` 在全局 `NLL` 显著改善 **`-0.0263 [-0.0424, -0.0104]`**（`Front30 NLL -0.1003`、`D1_obs NLL -0.1013`），`Mahlstedt BA3` 达 `69.00%`（vs `67.29%`）。
4. **`Ref-L1-GN2+DRO` 仍在全局 `BA4`、单箱 `1% FPR` $\tau$ 效率与极点分辨率领先**：`Ref-L1-GN2+DRO` 3-seed 集成在 `Tune6k` 上取得 `BA4 = 55.13%`（vs `Full Blend 54.68%`，差值 `-0.45 pp [-1.70, +0.83]` 在统计学上不可显著区分）、$\tau@1\%$ `6.40%`（vs `1.53%`，主要领先于 `Fortune_ScintOnly` 的 `-5.33 pp`）及 `Front30 PV d_xy = 1.27 mm`（vs `3.60 mm`）。
5. **系统性局限已被定位**：(a) `BackConeSlabPrior` 受背向散射触发率高（`Front30` 样本 `15/32 = 47%`），叠加全局 PV 尺度缩放导致 `Front30 z-pull = 2.55`、`Interm150 x/y-pull = 0.73/0.55`；(b) `Gen4` 对 `pixel_dark_1e-5` 暗计数退化敏感（`-2.25 pp`）；(c) `ray`-free `nsbi_unbinned_pl` 在 `Gen4` 分布上存在 `Z0 = 0.0` 优化问题；(d) `bayesian_tau_llr_from_logits13` 在低统计尾部与高暗噪声域的泛化弱于 `Ref-L1-GN2+DRO` 的 KL-DRO 判别边界。

### 假设与待验证推断
1. **假设 A（`Fortune_ScintOnly` 尾部效率差异成因）**：`Gen4_SSL_Tail` 与 `Full_Blend` 在单箱 `1% FPR` 效率（`1.27% / 1.53%`）低于 `PinpointGen4_SupOnly`（`3.40%`）及 `Ref-L1-GN2+DRO`（`6.40%`），而 `BA4` 与 `NLL` 更优，推测与 Stage-III UParT13 先验加权及温度缩放压缩了极少数高纯 $\tau$ 尾部对数几率的动态范围有关；尚待在无能量先验重加权的 `Tune_Screen` 校准下通过可控重训验证。
2. **假设 B（PV 区域失配修复）**：将 `BackConeSlabPrior` 替换为抗背向散射 GNC 初值估计、并按 `D1 / D2 / D3` 观测域分别拟合 Student-$t$ 尺度因子，预计可同时消除 `Front30 z-pull = 2.55` 与 `3.60 mm -> ~1.3 mm` 的横向精度差距；尚待下阶段实验实测。

---

## 下一步计划

1. **修复 PV 区域失配与 `Front30` 背向散射偏差**
   - 输入：`Train32k` 与 `Tune_Screen (N=2,000)`，冻结已训练 `SSL-v2` 主干先验证后处理与微调头；
   - 任务：在 `micro_ndive.py` 引入 `GNC` 鲁棒射线顶点初值并抑制前置背向散射权重，实施按 `obs_dom ∈ {1, 2, 3}` 分域 Student-$t$ 尺度标定；
   - 验收标准：`Front30 PV d_xy <= 1.5 mm` 且 `All_valid`、`Front30`、`Interm150` 全部 `9/9` `p68_abs_pull ∈ [0.85, 1.15]`。
2. **修复 `Gen4` 高纯度 $\tau$ 尾部判别与暗噪声稳健性，以及 `nsbi_unbinned_pl` 梯度**
   - 输入：`Train32k`、`Stress400 (pixel_dark_1e-5)` 与 `nsbi_discovery.py`；
   - 任务：排查 `bayesian_tau_llr_from_logits13` 与 Stage-III `tau` 权重对 `Fortune_ScintOnly` 极低 FPR 尾部的压制并联合 KL-DRO 优化，修复 `fit_unbinned_nsbi_pl` 的 `sr_is_sig_ev` Asimov 梯度；
   - 验收标准：`Tune6k tau@1% >= 6.5%`、`pixel_dark_1e-5 ΔBA4 >= -1.0 pp`、`nsbi_unbinned_pl Z0 > 0`。
3. **封存阶段门禁验证（解封前禁止自动运行）**
   - 在上述两项修正在 `Tune6k` 全部达标后，经人工审批依次解锁 `Cal-A`、最终锁阈并执行 `Cut-B` 与 `Confirm 10k` 单次盲法决胜评估。
