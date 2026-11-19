# Stage 317：`300k×24p` 同预算受控消融、单通道 PE 读出归因、物理动量 `NOMAD` 检验与前向流强 27 箱剖面似然校准

## 结论摘要

严格在不触碰任何封存评估集（`Cal_A_Cut_B_Confirm_payload_read = false`）的前提下，已在 `Train300k × 24 passes`（`batch_size = 128`, `lr = 8e-4`, `ema = 0.9997`）同预算条件下完成 **6 组 × 3 随机种子（`17, 42, 2026`）= 18 个神经网络模型** 的训练与 `Tune40k` 5 折家族分组 OOF 集成 + 1,000 次家族分块配对 Bootstrap 评估，并同步完成 **Exp-6（真实 LHC 前向流强 27 箱超尾 + 粲控制区 + Barlow–Beeston lite 剖面似然及 2,000 次 $H_0$ 频率学覆盖率玩具实验）**。核心定量结论如下：

1. **Exp-1（`P1-2` 稠密监督与曝光扩展解混杂，`300k×24p` 受控对照）**：
   - 在完全相同的 `300k×24p` 训练预算与无偏固定事件损失（`FixedWeightEventLoss`）下，`PF V3 (Event4Head)`（7 类物理衰变道边际化 + 粲辅助头 + 可见运动学）在 `Truth_pixel_centered` 切片内精确 `1.00% FPR` 下的 $\nu_\tau$ 效率达到 **`22.06%`**，较同预算 `PF V2 (Anchored)`（**`18.36%`**）显著提升 **`+3.70 pp`**（95% 配对家族 Bootstrap CI **`[+0.13, +6.80] pp`**，下界严格大于 0；`BA4` 提升 **`+2.30 pp`**，`[+1.26, +3.38] pp`），**满足 Stage 314 复杂度保留门限（`retain_event4head_multitask = true`）**。
   - 继续叠加 `GN2` 像素簇稠密起源/亲和辅助头后，`PF V3 (+GN2 dense)` 的 `Truth_pixel_centered` `BA4` 提升 **`+2.86 pp`**（`69.90% → 72.75%`），但切片内精确 `1.00% FPR` $\nu_\tau$ 效率增量为 **`+2.11 pp`**，其 95% 配对 CI 为 **`[-0.55, +6.01] pp`**（跨越零点），且 `0.10% FPR` 效率增量为 **`-0.92 pp`**（`[-8.63, +11.16] pp`）。依据 Stage 314 预注册停止规则（须同时满足 $\Delta\varepsilon_\tau@1\% \ge +1.0\text{ pp}$ 且 95% CI $> 0$），**裁定在新一代精简主架构中移除 `GN2` 稠密簇辅助头（`retain_dense_gn2_heads = false`）**。
2. **Exp-3（`P1-1` 剥离双通道泄漏与施加真实单通道 PE 阈值/饱和）**：
   - 彻底移除闪烁体 `edep_proxy_MeV` 通道并施加 `>= 2.5 PE` 硬件甄别阈值与 $N_{\text{pix}} = 14,400$ SiPM 微单元饱和后，`PinpointFormer`（`exp3_pf_pe_only_ev4` vs `exp1_pf_v3_event4head`）在 `Truth_pixel_centered` 上的精确 `1.00% FPR` $\nu_\tau$ 效率从 **`22.06%`** 降至 **`16.64%`**（**`-5.42 pp`**，95% CI **`[-8.48, -0.51] pp`**），`All_Tune40k` `BA4` 从 **`50.78%`** 降至 **`49.71%`**（**`-1.07 pp`**，`[-1.42, -0.73] pp`）；而双投影卷积基线 `B1++` 在单通道 PE 下保持不变（`10.04% → 10.96%`，**`+0.92 pp`**，`[-1.05, +3.06] pp`）。这以受控数据证实：`PinpointFormer` 的立体条模块在旧版双通道输入下确实利用了非物理的 $\ln(\text{pe}/\text{edep})$ 沿条未测量位置泄漏（约 `+5.4 pp` 虚高增益）。
   - 单通道 PE 条件下，6 模型跨族集成（`PF_PE_Only + B1++_PE_Only`）相对单一 3-seed `PF_PE_Only` 在 `Truth_pixel_centered` `1.00% FPR` 效率上的净增量为 **`+0.00 pp`**（`16.64% → 16.64%`，`[-1.10, +1.47] pp`），验证单一 `PinpointFormer` 族已涵盖双投影 CNN 信息。
3. **Exp-5（`P1-3` 修正 242 倍谱仪动量定标与移除双站伪动量后的 `NOMAD` 检验）**：
   - 将谱仪矢高常数修正为物理精确值 `100,705.0 MeV·mm` 并将双站欠定轨迹动量归零后，`PF V4 NOMAD (PE-Only)` 相对 `PF V3 (PE-Only)` 在 `Truth_pixel_centered` 上将 `BA4` 提升 **`+1.24 pp`**（`67.59% → 68.83%`，`[+0.39, +2.03] pp`），但精确 `1.00% FPR` $\nu_\tau$ 效率增量仅为 **`+0.40 pp`**（`16.64% → 17.04%`，95% CI **`[-2.96, +3.27] pp`**），低于 `+1.0 pp` 准入门限（**`retain_calibrated_nomad_head = false`**）。
4. **`P2-2` 与 `P2-3` 内部健康验证（无偏初级顶点回归与注意力有效锐度）**：
   - 移除 `vertex_anchor.py` 的 `+2.0` 对数方差偏置并切换至 `FixedWeightEventLoss` 后，`Truth_pixel_centered` 初级顶点纵向中位绝对残差 $\text{med}|\hat{z}_v - z_{\text{true}}|$ 从旧版 `W3-11 GN2` 的 **`9.91 mm`**（3D 中位误差 **`11.37 mm`**）大幅压缩至 **`4.00–4.66 mm`**（`PF V3 Event4Head` 为 **`4.00 mm`**，`PF V3 PE-Only` 为 **`4.30 mm`**，3D 中位误差 **`5.52–5.94 mm`**），误差缩减超过 **55%**。同时揭示了旧版 `Train300k` 几何错位（`P0-2` 靶区硅平面整体偏移 `+10.000 mm`）与顶点损失的耦合效应：在存在 `10 mm` 硬件层序错位的旧样本上强制拉紧真值顶点约束，会使特征骨干为弥合坐标矛盾而付出一部分分类尾效代价（`-5.28 pp`），进一步凸显切换至 Stage 318 洁净几何模拟集的必要性。
   - 豁免 1D 温度/长度标量的 `AdamW` `weight_decay` 后，全部注意力层的头温度基数 `base_scale = exp(log_scale)` 稳定保持在 **`5.35–22.17`**（100% 大于坍缩阈值 `4.90`），序列长度 $N=200$ 处的有效锐度 `eff_sharpness_N200` 达 **`7.75–40.48`**。
5. **Exp-6（`P0-1`、`P1-4`、`Path D` 真实前向流强显著性与 2,000 次 $H_0$ 玩具校准）**：
   - 在真实 LHC 2t 靶基准区前向中微子流强比（$f_\tau = 0.30\%$，$\nu_e:\nu_\mu:\text{NC} = 17.05\%:58.17\%:24.77\%$）及 8 项有符号探测器系统误差 + $\pm 30\%$ 粲产生截面不确定度 + Barlow–Beeston lite 模板统计不确定度下，将粗分 18 箱升级为 **`Cut-B` 锁定的 27 箱（21 个超尾信号箱 + 6 个高 `charm_prob` 粲控制区箱）** 后，预期剖面似然显著性 $Z_{\text{PL}}$ 在 Run 3（`250 fb^-1`, `15k` 基准事件）从 **`0.650σ` 提升至 `1.128σ`（`+73.5%`）**，Run 4（`680 fb^-1`, `45k`）从 **`0.895σ` 提升至 `1.621σ`（`+81.2%`）**，HL-LHC（`3 ab^-1`, `180k`）从 **`1.119σ` 提升至 `2.179σ`（`+94.7%`）**。
   - 2,000 次独立 Poisson + 高斯冗余玩具蒙特卡洛证实 $H_0$ 检验统计量完全无第一类错误膨胀（`q0 = 0` 比例 **`0.581`**；经验超越率 $P(Z_0 \ge 1\sigma) = 9.75\%$、$P(Z_0 \ge 2\sigma) = 0.40\%$、$P(Z_0 \ge 3\sigma) = 0.00\%$，严格保守于渐近 $\frac{1}{2}\chi^2_1$ 曲线）。

---

## 阶段目标

围绕 Stage 314 终审报告确立的第 1–2 周决定性实验清单（`Exp-1`、`Exp-3`、`Exp-5`、`Exp-6`）及 `P0-1`、`P1-1`、`P1-2`、`P1-3`、`P1-4`、`P2-2`、`P2-3` 修复目标，回答以下核心问题并完成验收：

1. 在严格对齐 `300k×24p` 曝光预算后，`Event4Head` 多任务边际化头与 `GN2` 像素簇稠密辅助头各自真实的净 $\nu_\tau$ 鉴别增益是多少？哪些组件通过 `+1.0 pp` 且 95% CI $> 0$ 的准入门限？
2. 剥离非物理 `edep_proxy_MeV` 闪烁体通道并施加真实 PE 阈值与 SiPM 饱和后，三维点簇网络与双投影网络的真实基准性能各变化多少？修正 242 倍谱仪动量定标后的 `NOMAD` 横向运动学头是否达到准入门限？
3. 在真实前向流强先验、有符号模板插值与 Barlow–Beeston lite 统计模型下，引入超尾分箱与粲控制区能否大幅缓解系统误差饱和且保持严格的频率学第 I 类错误控制？

---

## 工作内容

1. **受控 18 模型 GPU 训练矩阵执行（[run_stage317_ablations.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v6_clean_sim/run_stage317_ablations.py)）**：
   - 构建 6 组 3-seed（`17, 42, 2026`）实验配置：`pf_ctrl_v2_anchored`（8.66M 参数）、`pf_ctrl_v3_event4head`（8.66M）、`pf_ctrl_v3_gn2_fixed`（8.98M）、`pf_pe_only_ev4`（8.66M，`pe_only=true`, `bar_threshold_pe=2.5`, `sipm_npix=14400`, `y_mirror=false`）、`pf_pe_only_nomad_phys`（8.80M，物理定标 `NOMAD`）以及 `b1pp_pe_only`（3.12M）。
   - 修复 `extract_pf_item` 在 `pe_only=true` 全条过阈事件下条件省略 `bar_keep_mask` 引发的 `MappedItems` 偏移量错位（见 [pixel_tokenizer.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v2_pinpointformer/pixel_tokenizer.py) 与 [mapped_inputs.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v2/r3/mapped_inputs.py)），以及检查点经本地 `/tmp` 原子同步至 EOS 的防截断机制（[v2_pinpointformer/train.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v2_pinpointformer/train.py)、[v1_b1pp/train.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v1_b1pp/train.py)）。
   - 利用 Kubeflow H100 与 HTCondor GPU 队列单卡多进程共享内存并行完成全部 18 个 `300k×24p` 模型训练。
2. **`Tune40k` 严格盲隔离评估与准入裁决（[evaluate_stage317.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v6_clean_sim/evaluate_stage317.py)）**：
   - 按 `manifest.json` 记录的 4k `tune_screen` NLL 锁定每颗种子的 `best` 或 `ema` 预测文件，在 `Tune40k` 上运行 5 折家族分组凸组合分域校准与 `lambda_charm` 交叉拟合，同时并列输出**切片内精确 1% / 0.1% FPR 效率**与**全局预算 NP 1% FPR 工作点（含实际局域 FPR）**（解决 `P1-4`）。
   - 对 7 对关键模型组执行 `n_boot = 1,000` 次家族分块配对 Bootstrap 检验，输出复杂度保留决策字典供 Stage 319 冻结清单自动读取。
3. **真实前向流强 27 箱剖面似然与玩具 MC 覆盖率审计（[flux_significance.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v5_confirm/flux_significance.py)、[run_exp6_flux_audit.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v5_confirm/run_exp6_flux_audit.py)）**：
   - 构建逐源（$\nu_e\text{ CC}$, $\nu_\mu\text{ light}$, $\text{NC light}$, $\text{CC+NC charm}$）预期产额模板、8 项有符号探测器响应系统误差形变（含非对称分段插值）、分域归一化与斜率冗余、$\pm 30\%$ 粲产生截面先验以及逐箱 Barlow–Beeston lite 二次解析近似 $\hat{\gamma}_b(\mu,\theta)$，执行多起点 `L-BFGS-B` 优化并完成 2,000 次 HL-LHC $H_0$ 玩具标定。

---

## 详细数据

### 1. 执行命令与核心产物路径

```bash
export PYTHONPATH="/eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src:${PYTHONPATH:-}"
python3 -m analysis.flavour.v3.v5_confirm.run_exp6_flux_audit --n-toys 2000 --workers 16
python3 -m analysis.flavour.v3.v6_clean_sim.evaluate_stage317 --n-boot 1000
```

- **Stage 317 评估报告 JSON**：[stage317_ablation_evaluation_report.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v4_audit_2026-11-18/stage317_ablations/stage317_ablation_evaluation_report.json)
- **Stage 317 300-DPI 英文三联图**：[stage317_controlled_ablations_and_vertex_calibration.png](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v4_audit_2026-11-18/stage317_ablations/stage317_controlled_ablations_and_vertex_calibration.png)
- **Exp-6 前向流强显著性报告 JSON**：[exp6_realistic_flux_significance_report.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v4_audit_2026-11-18/exp6_flux_significance/exp6_realistic_flux_significance_report.json)
- **Exp-6 300-DPI 英文前向流强与 $H_0$ 校准图**：[exp6_forward_flux_significance_and_h0_coverage.png](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v4_audit_2026-11-18/exp6_flux_significance/exp6_forward_flux_significance_and_h0_coverage.png)

### 2. `Tune40k` 3-Seed OOF 集成全组指标与初级顶点残差对照表

| 模型组（均为 `300k×24p`，3 seeds） | 参数量 | 闪烁体读出模式 | `All_Tune40k` `BA4` (%) / `NLL` | `All` 精确 `1% FPR` $\varepsilon_\tau$ (%) | `Truth_pix` `BA4` / `cond_BA3` (%) | `Truth_pix` 精确 `1%` / `0.1% FPR` $\varepsilon_\tau$ (%) | `Truth_pix` NP 切点 $\varepsilon_\tau$ (*实际局域 FPR*) | `Truth_pix` $\text{med}|\Delta z|$ / 3D 中位误差 [mm] |
| --- | ---: | --- | --- | ---: | --- | --- | --- | --- |
| `w3_11_pf_v3_gn2_legacy`（旧损失） | 8.98M | 双通道 (`edep+pe`) | `53.24` / `1.0002` | `14.08` | `74.58` / `87.19` | `29.46` / `10.17` | `34.48%` (*`1.99%`*) | `9.91` / `11.37` |
| `exp1_pf_v2_anchored`（无偏顶点） | 8.66M | 双通道 (`edep+pe`) | `49.84` / `1.0740` | `9.79` | `67.60` / `82.13` | `18.36` / `3.17` | `23.51%` (*`1.93%`*) | **`4.66` / `6.26`** |
| `exp1_pf_v3_event4head`（7 类多任务） | 8.66M | 双通道 (`edep+pe`) | `50.78` / `1.0547` | `10.56` | `69.90` / `83.69` | **`22.06` / `6.87`** | `26.82%` (*`1.96%`*) | **`4.00` / `5.52`** |
| `exp1_pf_v3_gn2_fixed`（+GN2 稠密头） | 8.98M | 双通道 (`edep+pe`) | `52.08` / `1.0260` | `11.78` | `72.75` / `85.92` | `24.17` / `5.94` | `28.53%` (*`1.93%`*) | `7.11` / `8.79` |
| `exp3_pf_pe_only_ev4`（真实单通道 PE） | 8.66M | **单通道 PE ($\ge 2.5$)** | `49.71` / `1.0763` | `9.54` | `67.59` / `81.61` | **`16.64` / `5.81`** | `21.27%` (*`1.90%`*) | **`4.30` / `5.94`** |
| `exp5_pf_pe_only_nomad_phys`（物理 NOMAD） | 8.80M | **单通道 PE ($\ge 2.5$)** | `49.94` / `1.0709` | `9.76` | `68.83` / `82.11` | `17.04` / `6.47` | `22.72%` (*`1.93%`*) | **`4.15` / `5.81`** |
| `w3_11_b1pp_dual_legacy` | 3.12M | 双通道 (`edep+pe`) | `48.13` / `1.1147` | `6.39` | `64.49` / `79.20` | `10.04` / `3.04` | `11.36%` (*`1.23%`*) | — |
| `exp3_b1pp_pe_only` | 3.12M | **单通道 PE ($\ge 2.5$)** | `48.11` / `1.1136` | `6.85` | `64.27` / `79.60` | `10.96` / `3.17` | `12.68%` (*`1.40%`*) | — |
| `exp3_dual_family_pe_only_6m` (6 模型) | 11.78M | **单通道 PE ($\ge 2.5$)** | `49.70` / `1.0762` | `9.33` | `67.36` / `81.64` | `16.64` / `6.74` | `21.40%` (*`1.93%`*) | — |

### 3. 1,000 次家族分块配对 Bootstrap 假设检验与准入裁决表

| 实验对比项 (`A minus B`) | `Truth_pix` $\Delta\varepsilon_\tau@1\%$ [pp] (95% CI) | `Truth_pix` $\Delta\varepsilon_\tau@0.1\%$ [pp] (95% CI) | `Truth_pix` $\Delta\text{BA4}$ [pp] (95% CI) | `All_Tune40k` $\Delta\varepsilon_\tau@1\%$ [pp] (95% CI) | Stage 314 门限裁决 |
| --- | --- | --- | --- | --- | --- |
| **Exp-1a**: `V3_Event4Head` − `V2_Anchored` | **`+3.70` `[+0.13, +6.80]`** | `+3.70` `[-4.17, +6.38]` | **`+2.30` `[+1.26, +3.38]`** | **`+0.77` `[+0.30, +1.44]`** | **保留 (`true`)** |
| **Exp-1b**: `V3_GN2_Fixed` − `V3_Event4Head` | `+2.11` `[-0.55, +6.01]` | `-0.92` `[-8.63, +11.16]` | **`+2.86` `[+1.75, +3.99]`** | **`+1.22` `[+0.37, +1.81]`** | **淘汰 (`false`，CI 跨零)** |
| **Exp-3a**: `PF_PE_Only` − `PF_DualChannel` | **`-5.42` `[-8.48, -0.51]`** | `-1.06` `[-5.96, +2.85]` | **`-2.30` `[-3.31, -1.24]`** | **`-1.02` `[-1.84, -0.40]`** | 证实双通道泄漏虚高 |
| **Exp-3b**: `B1PP_PE_Only` − `B1PP_DualChannel` | `+0.92` `[-1.05, +3.06]` | `+0.13` `[-1.63, +2.07]` | `-0.22` `[-1.11, +0.71]` | **`+0.46` `[+0.05, +1.13]`** | 投影 CNN 无沿条泄漏 |
| **Exp-3c**: `DualFamily_6m` − `PF_PE_Only_3m` | `+0.00` `[-1.10, +1.47]` | `+0.92` `[-0.81, +2.82]` | `-0.23` `[-0.51, +0.09]` | `-0.21` `[-0.56, +0.16]` | 单族 3-seed 已饱和 |
| **Exp-5**: `PF_PE_NOMAD_Phys` − `PF_PE_Only` | `+0.40` `[-2.96, +3.27]` | `+0.66` `[-4.57, +3.10]` | **`+1.24` `[+0.39, +2.03]`** | `+0.22` `[-0.35, +0.85]` | **淘汰 (`false`，`< +1.0 pp`)** |

### 4. Exp-6：2t 靶基准区真实前向流强预期发现显著性（$f_\tau = 0.30\%$，8 项有符号系统误差 + 粲截面 + BB-lite）

| 实验阶段与积分亮度 | 基准区总事件 $N_{\text{fid}}$ | 预期 $\nu_\tau$ CC 信号数 $S$ | 粗分 18 箱纯统计 $Z_{\text{stat}}$ | 粗分 18 箱剖面似然 $Z_{\text{PL}}$ | **27 箱（超尾 SR21 + 粲 CR6）纯统计 $Z_{\text{stat}}$** | **27 箱完整剖面似然 $Z_{\text{PL}}$（相对 18 箱增益）** |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| **LHC Run 3** (`250 fb^-1`) | 15,000 | 45.0 | `1.305σ` | `0.650σ` | `1.555σ` | **`1.128σ` (`+73.5%`)** |
| **LHC Run 4** (`680 fb^-1`) | 40,800 | 122.4 | `2.153σ` | `0.874σ` | `2.564σ` | **`1.576σ` (`+80.3%`)** |
| **LHC Run 4 扩展** (`680 fb^-1`, 45k) | 45,000 | 135.0 | `2.261σ` | `0.895σ` | `2.693σ` | **`1.621σ` (`+81.2%`)** |
| **HL-LHC** (`3,000 fb^-1`) | 180,000 | 540.0 | `4.522σ` | `1.119σ` | `5.386σ` | **`2.179σ` (`+94.7%`)** |
| **FPF 规模参考** (`10,000 fb^-1`) | 600,000 | 1,800.0 | `8.256σ` | `1.209σ` | `9.834σ` | **`2.470σ` (`+104.2%`)** |

---

## 阶段性结论

### 已验证事实
1. **多任务与稠密监督的真实因果拆解（解决 `P1-2`）**：旧版报告中 `10.17% → 27.48%` 的跃升主要由 `100k×16p → 300k×24p` 训练曝光扩展以及旧版双通道泄漏贡献；在严格同预算 `300k×24p` 下，7 类物理衰变道边际化 `Event4Head` 提供显著且稳健的 `+3.70 pp` 增益，而 `GN2` 像素簇稠密头的尾部 $\nu_\tau$ 效率增益跨零且引入额外内存/侧车依赖，按预注册规则予以精简移除。
2. **闪烁体单端 PE 读出的物理归因（解决 `P1-1`）**：剥离 `edep_proxy_MeV` 使 `PinpointFormer` 像素区 `1% FPR` $\nu_\tau$ 效率回落 `5.42 pp`，证实双通道比值确实构成了非物理空间位置捷径；此后所有洁净模拟实验均锁定为 `pe_only = true`（`>= 2.5 PE` + `14,400` 微单元饱和）。
3. **真实流强显著性瓶颈与多道超尾似然增益（解决 `P0-1`、`P1-4`、`Path D`）**：在 $f_\tau = 0.30\%$ 真实流强先验与完整系统误差下，粗分 18 箱因最高分箱 $S/B \ll \sigma_{\text{sys}}$ 在 `~1.2σ` 处彻底系统误差饱和；引入 21 箱超尾细分与 6 箱粲富集控制区联立拟合使 HL-LHC 剖面似然显著性翻倍至 `2.18σ`，同时证明要突破 $3\sigma\text{–}5\sigma$ 发现阈值，必须依靠物理层面的高纯微米次级顶点鉴别（`Path C`）提升尾箱信噪比。

### 待验证假设（进入 Stage 319 单次解封检验）
1. 在消除了 `10 mm` 像素层序错位的 Stage 318 洁净数据集上，无偏初级顶点回归与显式 C++17 微米多航迹/AVR 次级顶点混合特征（`Path C`）不仅不会与像素坐标产生几何冲突，反而会将像素区初级顶点纵向中位误差压至 `< 2.0 mm`，并在 `1% / 0.1% FPR` 处显著超越纯点簇基线。

---

## 下一步计划

1. **提交 Stage 317 代码与报告**：保持 Git 工作树完全干净，使 `git rev-parse HEAD` 生成唯一可追溯的预解封提交哈希。
2. **执行 Stage 319 预解封冻结（`P3-1`）**：运行 `run_stage319_clean_campaign.py freeze`，仅依据 `CleanTune8k` 与本阶段保留规则（自动排除未过门限的 `NOMAD` 组，在 `s319-pitch-20um` 与 `s319-hybrid-uvtx` 间选定主模型）生成含 40 位 Git SHA 与 24 个检查点 SHA-256 的 [v6_freeze_manifest.json](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/roadmap/v3/v6_freeze_manifest.json)。
3. **单次解封 `CleanCalA → CleanCutB → CleanConfirm`（`Stage 319`）**：运行 `run_stage319_clean_campaign.py unseal-and-evaluate --n-toys 1000`，交付从头重训的 `20 μm / 100 μm / 300 μm / 1 mm` 螺距与 6 层成本曲线（`Exp-2` / `P1-5`）、机械平面错位容差（`0–50 μm`）、显式微米次级顶点混合增益（`Path C`）、配对 Geant4/强子化系统误差、外部岩石 $\mu$/中性强子穿透本底以及洁净样本前向流强显著性。
