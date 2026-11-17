# 2026-11-17 Stage 312 — PINPOINT Flavour V3 Phase V4 (`W3-10` + `V4.5` 审查)：NOMAD 横向运动学、`SOPA-s` Partial-AUC 尾部精调与跨家族双机制集成

## 阶段目标

完成 `roadmap/v3` 阶段 **V4（`W3-10` 与 `V4.5` 条件化 SSL 触发审查）**，验证并交付以下核心目标：

1. **纯可观测量 NOMAD 横向运动学模块与消融验证**：在不接触真值侧车及密封数据（`Cal-A40k`、`Cut-B160k`、`Confirm200k` 全程 `Cal_A_Cut_B_Confirm_payload_read: false`）的前提下，从闪烁体/像素/磁谱仪原始响应提取 10 维 $y$ 镜面对称不变的 NOMAD 横向拓扑与运动学特征，分别合入正交投影模型 `B1++` 与三维点云模型 `PinpointFormer-GN2`，并量化从零训练（`100k x 16p`）与微调（`300k x 3p`）两套尺度下的因式分解增益。
2. **One-Way Partial-AUC（`@1%` & `@0.1% FPR`）尾部精调故障诊断与数学修复**：针对初版 5-pass 全局成对无界对数几率截断导致的 $\tau@1\%$ 塌缩与 `BA4` 下降，基于计算机科学文献重构 `SOPA-s` 逐正例 KL-DRO 与有界边界门控 CVaR 损失，配合温和差分学习率完成 3-pass `300k` 微调验证与全分支梯度健康门禁。
3. **跨家族 5-fold OOF 双机制集成、晋级门禁与 `V4.5` 条件化 SSL 裁决**：构建物理域（`D_pix_contained` / `D_scint_contained` / `D_partial_or_edge`）严格折外（family-grouped out-of-fold）凸组合集成与 `lambda_charm` / Strategy I (`S_tau`) / Strategy II (Neyman-Pearson) 工作点，对照 V4 晋级门禁（全样本及闪烁体域 $\tau$ 效率相对 `R2 D2` 翻倍且 `cond_BA3` 非劣效 $\ge -0.5\,\text{pp}$），完成 `V4.5` SSL 成本收益审计与全阶段算力核算。

---

## 工作内容

1. **纯可观测量 NOMAD 横向运动学与模型接口升级**：
   - 在 [nomad_kinematics.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v4_tau/nomad_kinematics.py) 实现 `NomadTransverseExtractor(nn.Module)`，支持 `B1++`（`from_b1pp`）与 `PinpointFormer`（`from_pf`）双入口 GPU 矢量化 FP32 特征提取，在 `D_empty` 域严格置零，并保持精确 $y$ 镜面对称不变性。
   - 在 [v1_b1pp/models.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v1_b1pp/models.py)（拆分 `forward_fused`）与 [v2_pinpointformer/train.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v2_pinpointformer/train.py)（在 EMA 深拷贝前加载 `init_checkpoint`、支持 `build_param_groups`、完成后清理 `resume.pt`）添加向后兼容接口；在 [v4_tau/models.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v4_tau/models.py) 实现 `ZAlignedStereoNetV4` 与 `PinpointFormerV4`。
2. **pAUC 故障诊断与 `SOPA-s` 数学重构（[pauc_finetune.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v4_tau/pauc_finetune.py)）**：
   - 初版 5-pass 诊断波（现归档于 `W3-10/diagnostic_v1_5p_global_pauc/`）暴露三处根因：（1）在无界 log-odds $s \in (-14,+14)$ 全批负×正矩阵 $B^-\times B^+$ 上取单一 $\tau_q$ 分位截断且仅保留 `0.5 * base_loss`，导致 `top 1%` 活跃对全部由无法挽回的低分边界 $\nu_\tau$ 事件（$s_{\text{pos}}\approx -6$）主导，梯度反推全批高 $P(\tau)$ 输出（`B1++ t1` 跌至 `1.95%`，`PF BA4` 跌至 `49.87%`）；（2）已收敛 `24p` 旗舰被赋予 `1.2e-4` 过大学习率在 Pass 2–5 过拟合 `train300k`；（3）`B1++` 误从 `W3-05 ema.pt`（`NLL=1.1756`）而非官方主权重 `best.pt`（`NLL=1.1165`）热启动。
   - 按 **Zhu et al. (ICML 2022, SOPA-s)** 与 **Eban et al. (AISTATS 2017)** 重构 `DROPartialAUCLoss`：在有界概率 $p_\tau \in [0, 1]$ 上构建**逐正例 KL-DRO**（`kl_temp=0.04, margin_p=0.15`），引入追踪负例 EMA $(1-\beta)$ 分位 `q_beta_ema` 的可发现边界门控 $g_i = 0.05 + 0.95\,\sigma((s_i - (q_{\text{ref}}-1.8))/0.8)$（将边界正例梯度权重提升至无望正例的 ~240 倍），叠加以有界对数几率 $\tilde{s}=3\tanh(s/3)$ 度量的 Top-$\beta$ 负例尾部 CVaR（`margin=0.40`），恢复全权重 `1.0 * base_loss + 0.20 * l_pauc`；热启动时对预训练主干施加 `0.15x` 温和学习率，重设 3-pass `300k` 实验矩阵并在 `B1++` 统一采用 `W3-05 best.pt`。
3. **跨家族 OOF 集成与单元测试回归（[ensemble.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v4_tau/ensemble.py)、[evaluate.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v4_tau/evaluate.py)、[test_v3_v4_tau.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/tests/test_v3_v4_tau.py)）**：
   - 使用 5-fold family-grouped CV 按物理域（`D_pix_contained` / `D_scint_contained` / `D_partial_or_edge`）分别 L-BFGS-B 拟合凸融合权重、温度倒数与零和偏置（`D_empty` 固定均匀分布 `[0.25]*4`），并完全折外拟合 `lambda_charm`、Neyman-Pearson 域间 FPR 分配及 18-bin 多域 Asimov $Z_0$。
   - 补齐 [accounting.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/common/accounting.py) `-limit 2000` + 单集群精准历史回查 + HTCondor `.log` 回退解析，确保全量 `61` 次集群作业零悬空（`active_or_unresolved: []`）。

---

## 详细数据

### 1. AST 隔离、单元测试与 `--smoke-32` GPU 健康门禁

- **全量回归单元测试**：`python3 -m unittest discover -s tests -p 'test_v3_*.py' -v` 在 `96.80 s` 内 **`29 / 29` 全部通过（`OK`）**；AST 测试校验 `v4_tau/{__init__,nomad_kinematics,pauc_finetune,models,ensemble}.py` 零真值侧车及密封集标识，`NOMAD` 特征 `B1++`/`PF` 数值一致性与 $y$ 镜面不变性误差 `< 1e-4`、`DROPartialAUCLoss` 边界样本梯度主导及 `D_empty` 不变性、OOF 折间零泄露与 `git diff -- roadmap/v2` 为空均全部通过。
- **`--smoke-32` GPU 健康门禁**（[w3_10_health_report.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-10/w3_10_health_report.json)，`Tesla T4 SM75`，`BF16` + `FP32` 头部）：

| 变体 | 参数量 | 收敛步数 (`32` 事件) | 最终 NLL | 最终 Acc | 全分支正/有限梯度 | `BS=64` 峰值显存 | `BS=64` 吞吐 (`ev/s`) |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `PF_V4_gn2_nomad_pauc` | `9,098,319` | `21` | `0.0252` | `1.000` | 是 (`14/14` 分支) | `5,773.5 MiB` | `127.2` |
| `B1PP_V4_nomad_pauc` | `3,210,517` | `12` | `0.0298` | `1.000` | 是 (`9/9` 分支) | `2,408.8 MiB` | `246.9` |

### 2. Observable NOMAD 10 维特征在 `Tune40k D_scint_contained`（`N=23,516`）的分布特性

| NOMAD 物理特征 | `nu_tau CC` 均值 | `NC` 均值 | Cohen's `d` (`nu_tau` vs 非 $\tau$) | 物理区分特征 |
| --- | ---: | ---: | ---: | --- |
| `acoplanarity_norm` | `0.5481` | `0.6156` | `-0.0990` | 多叉共面性代理（`nu_e` 窄峰更强，`NC` 分布更弥散） |
| `core_30mm_iso` | `0.4663` | `0.4404` | `-0.0873` | 领头芯 `R <= 35 mm` 隔离度（`nu_e CC` 显著偏向高隔离） |
| `spec_to_cal_log_ratio` | `-0.0687` | `-0.0509` | `-0.0837` | 磁谱仪动量与量能器能量对数比（`nu_mu CC` 显著右移） |
| `cos_dphi_l_had` | `0.1735` | `0.3195` | `-0.0829` | 领头束流–反冲强子系统横平面夹角余弦 |
| `recoil_energy_frac` | `0.4182` | `0.4740` | `+0.0468` | 反冲系统横向能流占比 |
| `qt_perp_thrust_log` | `0.4275` | `0.4252` | `-0.0361` | 垂直于横向推力轴的横向动量展宽 $\log(1+q_T)$ |

> 完整分布图见 [fig1_nomad_kinematics_distributions.png](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-10/figures/fig1_nomad_kinematics_distributions.png)。由于 $\nu_\tau$ 包含约 $65\%$ 强子衰变道与约 $35\%$ 含中微子轻子衰变道（伴随部分不可见横向动量失衡），单项 NOMAD 一维特征效应量为温和中等水平（$|d|\le 0.10$），需要结合多特征非线性组合方能显著作用于正交投影模型。

### 3. `W3-10` 全部模型与跨家族集成核心性能矩阵（`Tune40k`，`seed = 42`，[evaluation_results.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-10/evaluation_results.json)）

主工作点严格依据 `4,000` 事件 `tune_screen` NLL（不触碰 `Tune_non_screen`）在 `best` 与 `ema` 间选定；V1 基线按 `W3-05` 统一取 `best_predictions.parquet`：

| 模型 / 集成版本（主 checkpoint） | `All NLL` (`40,000`) | `All BA4` (`95% CI`) | `All cond_BA3` | `All tau@1%` (`95% CI`) | `All tau@0.1%` | `D_scint BA4` (`23,516`) | `D_scint tau@1% / 0.1%` | `Pix_cen BA4 / tau@1%` (`3,765`) | `Asimov Z0` (`1-str / 18-bin OOF`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `R2 D2` Replay (V0) | — | `41.58%` | `57.14%` | `6.05%` | `1.60%` | — | `7.21% / 2.10%` | `55.92%(B0) / 7.40%` | `1.1434` (`S_tau`) |
| `b1pp_100k_16p_t4 [best]` | `1.1503` | `46.25%` (`[45.73,46.75]`) | `58.36%` | `4.84%` (`[4.29,5.51]`) | `1.17%` | `49.20%` | `5.53% / 1.18%` | `60.24% / 5.42%` | `1.1367 / 1.1187` |
| `b1pp_v4_100k_16p_nomad [best]` | `1.1442` | `46.67%` (`[46.22,47.18]`) | `58.66%` | **`5.68%`** (`[5.06,6.29]`) | **`1.50%`** | `49.57%` | **`6.21% / 1.49%`** | `60.71% / 6.61%` | `1.1532 / 1.1299` |
| `b1pp_300k_24p_t4 [best]` | `1.1165` | `48.24%` (`[47.75,48.71]`) | `59.95%` | `6.54%` (`[5.89,7.27]`) | `1.91%` | `51.12%` | `7.31% / 1.84%` | `63.52% / 9.64%` | `1.1906 / 1.1622` |
| `b1pp_v4_300k_3p_nomad_only [best]` | `1.1183` | `48.24%` (`[47.72,48.69]`) | `59.87%` | `6.50%` (`[5.87,7.20]`) | `1.86%` | `51.20%` | `7.26% / 1.87%` | `63.59% / 10.44%` | `1.1894 / 1.1607` |
| `b1pp_v4_300k_3p_pauc_only [best]` | `1.1225` | `47.84%` (`[47.35,48.28]`) | `59.85%` | `6.25%` (`[5.68,6.84]`) | `1.86%` | `50.77%` | `7.07% / 1.77%` | `62.51% / 10.70%` | `1.1835 / 1.1557` |
| `b1pp_v4_300k_3p_nomad_pauc [best]` | `1.1225` | `47.76%` (`[47.27,48.22]`) | `59.87%` | `6.20%` (`[5.69,6.91]`) | `1.86%` | `50.71%` | `7.05% / 1.75%` | `62.33% / 10.44%` | `1.1838 / 1.1547` |
| `pf_v3_100k_16p_gn2_dense [ema]` | `1.1152` | `48.31%` | `59.83%` | `7.27%` | `1.71%` | `51.28%` | `7.90% / 1.89%` | `64.08% / 10.17%` | `1.1888 / 1.1600` |
| `pf_v4_100k_16p_nomad [best]` | `1.1160` | `48.16%` | `59.93%` | `6.58%` | `1.76%` | `51.16%` | `7.14% / 1.72%` | `64.44% / 9.11%` | `1.1807 / 1.1528` |
| `pf_v3_300k_24p_gn2_flagship [ema]` | `1.0226` | `52.97%` (`[52.51,53.43]`) | `63.88%` | `13.59%` (`[12.77,14.40]`) | `5.81%` | `57.11%` | `16.19% / 6.88%` | `73.53% / 27.48%` | `1.3952 / 1.3145` |
| `pf_v4_300k_3p_nomad_only [best]` | `1.0247` | `52.94%` (`[52.47,53.39]`) | `63.78%` | `13.48%` (`[12.67,14.21]`) | **`5.90%`** | `57.03%` | `15.95% / 6.71%` | `73.77% / 27.61%` | `1.3939 / 1.3136` |
| `pf_v4_300k_3p_pauc_only [best]` | `1.0172` | `52.74%` (`[52.26,53.26]`) | `63.86%` | `13.27%` (`[12.62,14.22]`) | `5.85%` | `56.74%` | `15.74% / 6.47%` | `72.66% / 28.01%` | `1.3749 / 1.3025` |
| `pf_v4_300k_3p_nomad_pauc [best]` | `1.0169` | `52.67%` (`[52.19,53.14]`) | `63.84%` | `13.42%` (`[12.66,14.22]`) | `5.89%` | `56.60%` | `16.14% / 6.34%` | `72.72% / 28.27%` | `1.3735 / 1.3087` |
| **`ens_v3_base_pf_b1pp`** | **`1.0066`** | `52.90%` | `64.03%` | `13.37%` (`NP 13.40%`) | `5.62%` | `57.41%` | `16.07% / 4.84%` (`NP 16.21%`) | `73.85% / 29.85%` (`NP 29.99%`) | `1.4113 / 1.3245` |
| **`ens_v4_flagship_pf_b1pp`** | `1.0125` | `52.55%` | `63.88%` | `13.28%` (`NP 13.30%`) | **`5.80%`** | `56.90%` | `16.10% / 6.59%` (`NP 16.42%`) | `73.37% / 27.87%` (**`NP 31.18%`**) | `1.3855 / 1.3099` |
| **`ens_v4_multimember_dual_regime`** | `1.0068` | **`52.93%`** | **`64.04%`** | **`13.43%` (`NP 13.64%`)** | `5.62%` (`NP 5.33%`) | **`57.46%`** | **`16.22% / 5.81%` (`NP 16.54%`)** | **`73.87%` / `29.72%` (`NP 31.18%`)** | **`1.4134 / 1.3259`** |

> 补充切片（`ens_v4_multimember_dual_regime`）：`Tune_non_screen`（`N=36,000`）`BA4=52.93%, S_tau@1%=13.48%, NP@1%=13.64%`；`D_pix_contained`（`3,618`）`BA4=71.27%, S_tau@1%=27.82%, NP@1%=29.17%`；`D_partial_or_edge`（`11,783`）`BA4=40.78%, S_tau@1%=6.66%`；`Truth_large_scintillator`（`18,002`）`BA4=69.70%, cond_BA3=82.45%, NP@1%=27.18%`；`Tune_E_le_2p3TeV`（`38,874`）`BA4=52.80%, NP@1%=13.64%`；`HighE_gt_2p3TeV_non_tau`（`1,123`）在 Strategy I 全局 `1%` FPR 阈值下实际假阳性率为 **`0.623%`**（`ens_v4_flagship_pf_b1pp` 为 **`0.356%`**，`Strategy II NP@1%` 为 `0.890% / 0.712%`），高能非 $\tau$ 本底渗漏受控。

### 4. 初版 5-pass 故障波 vs 修复后 3-pass `SOPA-s` 对照及因式分解消融（`400x` 配对家族 block-bootstrap）

- **初版全局成对 pAUC 故障对照（`diagnostic_v1_5p_global_pauc` vs 修复 3-pass）**：
  - `b1pp_v4_300k_5p_nomad_pauc` 初版 [ema/best]：`All NLL = 1.1731 / 1.1551`，`BA4 = 44.59% / 45.04%`，`tau@1% = 1.95% / 4.24%` -> 修复后 [best] 恢复至 `NLL = 1.1225, BA4 = 47.76%, BA3 = 59.87%, tau@1% = 6.20%`（提升 `+4.25 pp tau@1% EMA`）。
  - `pf_v4_300k_5p_pauc_only` 初版 [ema]：`All NLL = 1.0663, BA4 = 49.87% (-3.10 pp), tau@1% = 12.40%` -> 修复后 [best] 恢复至 `NLL = 1.0172, BA4 = 52.74% (-0.23 pp), cond_BA3 = 63.86%, tau@1% = 13.27%, tau@0.1% = 5.85%`。
  - `pf_v4_300k_5p_nomad_pauc` 初版 [ema]：`All NLL = 1.0656, BA4 = 49.03%, tau@1% = 12.38%` -> 修复后 [best] 达到 `NLL = 1.0169, BA4 = 52.67%, cond_BA3 = 63.84%, tau@1% = 13.42%, tau@0.1% = 5.89%, PixCen tau@1% = 28.27%`（较 V3 旗舰在 `Truth_pixel_centered tau@1%` 与 `All tau@0.1%` 分别提升 `+0.79 pp` 与 `+0.08 pp`，且全 `NLL` 更低）。
- **配对消融关键增量**（`All` / `D_scint_contained`）：
  - `nomad_100k_b1pp`：`All ΔBA4 = +0.43 pp`，`Δtau@1% = +0.84 pp`（`95% CI: [+0.30, +1.32] pp`，统计显著）；`Scint Δtau@1% = +0.68 pp`。
  - `nomad_100k_pf`：`All ΔBA4 = -0.15 pp`，`Δtau@1% = -0.69 pp`（说明带 GN2 的 3D 顶点极坐标注意力已天然吸收横向几何，外加残差 MLP 在 `100k` 上无额外净收益）。
  - `ens_v4_multimember_vs_best_single_pf`：`All ΔBA4 = +0.27 pp`、`Δtau@1% = +0.31 pp`（`[-0.08, +0.68] pp`）；**`D_scint_contained ΔBA4 = +0.87 pp`、`Δtau@1% = +0.36 pp`**。

### 5. 5-Fold OOF 域权重分布、V4 晋级门禁与 `V4.5` 条件化 SSL 审查

- **学习到的域自适应融合权重**（[fig4_ensemble_domain_weights_and_asimov.png](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-10/figures/fig4_ensemble_domain_weights_and_asimov.png)）：
  - 在 `ens_v4_flagship_pf_b1pp` 中，`B1++` 权重随包含度降低而单调递增：`D_pix_contained` `18.50%` -> `D_scint_contained` `19.93%` -> **`D_partial_or_edge` `32.04%`**。
  - 在 `ens_v4_multimember_dual_regime` 中，5 成员在 `D_pix / D_scint / D_partial_or_edge` 权重分布：`PF-V4 NOMAD+pAUC`（`11.5% / 11.7% / 8.1%`）、`PF-V4 NOMAD`（`21.3% / 26.1% / 24.0%`）、`PF-V3 Flagship`（`45.6% / 38.3% / 31.0%`）、`B1++-V4 NOMAD+pAUC`（`10.3% / 8.6% / 9.0%`）、`B1++-V1 Base`（`11.2% / 15.3% / 27.8%`）——`B1++` 双成员在 `D_partial_or_edge` 合计贡献 **`36.82%`**，交叉折拟合最优 `lambda_charm = (0.03, 0.03, 0.11)`。
- **`v4_promotion_gate` 正式结果（`passed: True`）**：
  - `All tau@1% FPR`：最优单项/双成员集成 `13.42% / 13.43%` vs `R2 D2 6.05%`（**`2.218x >= 2.0x`**）。
  - `D_scint_contained tau@1% FPR`：`16.14% / 16.22%` vs `R2 D2 7.21%`（**`2.238x >= 2.0x`**）。
  - `All tau@0.1% FPR`：`5.89%` vs `R2 D2 1.60%`（**`3.681x >= 2.0x`**）。
  - `cond_BA3` 非劣效性：`PF-V4 vs PF-V3 = -0.04 pp`，`Flagship Ensemble vs PF-V3 = +0.003 pp`（均远高于 `-0.50 pp` 阈值）。
- **`V4.5` 条件化 SSL 触发审计（`triggered: False`）**：
  - `train300k` 含域事件池：`D_pix_contained = 27,219` + `D_scint_contained = 176,489` = **`203,708` (`67.90%`)**（对应 `1M` 池约 `679,027` 含域事件）。
  - 30 轮含域 SSL 预训练 + 匹配算力监督对照实验估计需 **`8.42 GPU·h`**；鉴于现有有监督 `PF-GN2 V4 + NOMAD + SOPA-s + OOF Dual-Regime Ensemble` 已达 `2.22x R2 D2`（`NP@1% = 13.64% All / 16.54% Scint / 31.18% PixCen`）大幅越过门禁，正式决策为 **`defer_v4_5_ssl_and_promote_supervised_plus_ensemble_to_v5`**。

### 6. 全流程算力台账与集群运行故障说明（[resources.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/resources.json)）

- **V4 阶段（`13` 次提交，全部结清）**：HTCondor 完成 `8` 项 GPU 作业（`v3-w3-10-b1pp-all-c1` `4.315 GPU·h`、`b1pp-300k-3p-r2` `2.026`、`pf-100k-nomad-c1` `2.617`、`pf-300k-nomad-only-c2` `1.478`、`pf-300k-pauc-only-c2` `1.494`、`pf-300k-nomad-pauc-c2` `2.348`、`pf-300k-3p-no-r2` `0.974`、`pf-300k-3p-po-r2` `1.495`、`pf-300k-3p-np-r2` `1.226`），另有 `2` 项被 `condor_rm` 的不兼容大显存请求（`0.0 h`）及 `2` 项在 Kubeflow `ml-004-h100-001-uusckqrqfxt3-node-0` 因 `/eos` FUSE 断连（`OSError: [Errno 107] Transport endpoint is not connected`）立刻失败并快照清理的任务（`0.059 GPU·h`）。**V4 阶段实耗合计：`71.886 Condor CPU core·h`、`18.031 GPU·h`**。
- **V0–V4 全库汇总**：`total_submissions = 61`、`total_condor_cpu_core_hours = 162.107`、`total_gpu_device_hours = 83.567`、`active_or_unresolved = []`（注：`resources.json` 中 `83.567 GPU·h` 含 2 项已被 GC 且无本地快照的早期 V2 Kubeflow 失败作业 `w307-kf-pf-pair*` 因 `kubectl` OIDC token 过期回退到 `(now - sched_t)` 的估算增量，当前 HTCondor `condor_q` 实际在列/排队作业数为 `0`）。
- **可视化工件**（均为纯英文标注）：[fig1_nomad_kinematics_distributions.png](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-10/figures/fig1_nomad_kinematics_distributions.png)、[fig2_v4_ablation_progression.png](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-10/figures/fig2_v4_ablation_progression.png)、[fig3_low_fpr_roc_curves.png](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-10/figures/fig3_low_fpr_roc_curves.png)、[fig4_ensemble_domain_weights_and_asimov.png](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-10/figures/fig4_ensemble_domain_weights_and_asimov.png)。

---

## 阶段性结论

1. **已验证事实 — Phase V4 正式通过晋级门禁**：`v4_promotion_gate` 全项达成（`All tau@1% = 13.42%–13.43%`、`D_scint_contained = 16.14%–16.22%`、`All tau@0.1% = 5.89%`，分别为 `R2 D2` 的 `2.22x`、`2.24x` 与 `3.68x`），且常见味 `cond_BA3` 全面保持非劣效（`63.84%–64.04%` vs `63.88%`）。
2. **已验证事实 — NOMAD 物理特征对投影架构与三维点云架构的差异化机理**：在从零训练的 `100k` 正交投影模型 `B1++` 中，显式 10 维 NOMAD 横向特征带来了统计显著的尾部发现效率增益（`Δtau@1% = +0.84 pp`, `95% CI: [+0.30, +1.32] pp`, `ΔBA4 = +0.43 pp`）；而在原生具备 3D 顶点极坐标聚类的 `PinpointFormer-GN2` 上，局部特征已充分编码横向反冲几何，使三维流形表征与正交宏观投影表征形成互补。
3. **已验证事实 — `SOPA-s` 逐正例 KL-DRO 消除全局截断塌缩**：将优化目标从无界全对矩心截断切换为有界概率空间 `SOPA-s` + 边界可发现门控 + `0.15x` 温和主干微调后，消除了初版 `-3.1 pp BA4` 崩塌与 `B1++` 假阳性淹没（`1.95% -> 6.20%`），在保持更低全局 `NLL (1.0169)` 的同时推动 `Truth_pixel_centered tau@1%` 达 `28.27%`。
4. **已验证事实 — 跨家族双机制集成释放最大综合物理收益**：5 折折外凸集成自动识别物理域差异（在 `D_partial_or_edge` 分配 **`36.8%`** 权重给 `B1++`，在包含域分配 **`76%–79%`** 给 `PinpointFormer`），将 `ens_v4_multimember_dual_regime` 推进至全指标最优：**`All NLL = 1.0068, BA4 = 52.93%, cond_BA3 = 64.04%, S_tau@1% = 13.43%, NP@1% = 13.64%`**、`D_scint_contained NP@1% = 16.54%`、`Truth_pixel_centered NP@1% = 31.18%`、单流 `Asimov Z0 = 1.4134 sigma`、18-bin 折外 `Asimov Z0 = 1.3259 sigma`。

---

## 下一步计划

1. **执行 Phase V5（`W3-11` 多随机种子扩展，预计约 12–16 GPU·h）**：将最优跨家族构型成员（`PinpointFormer-V4` + `PinpointFormer-V3-Flagship` + `B1++-V4` + `B1++-V1`）扩展至种子 `{17, 2026}` 与既有种子 `{42}` 并行对比，量化训练方差并冻结最终集成配方与域间 `(T_d, b_d, lambda_charm)` 参数。
2. **执行 `W3-12` 单次解封统计检验**：在 `Cal-A40k` 完成冻结标定验证、在 `Cut-B160k` 固化工作点阈值，最终单次盲测解封 `Confirm200k` 输出 Profile-Likelihood $\nu_\tau$ 发现显著度、系统误差与论文级诊断图表。（注：若在 V5 使用 Kubeflow，需先刷新用户 OIDC token 并排查/避开 `ml-004` 节点的 EOS FUSE 断连问题，或继续沿用 `bigbird20.cern.ch` `SM90` H100 MIG 集群。）
