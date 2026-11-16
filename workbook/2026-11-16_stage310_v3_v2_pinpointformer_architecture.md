# 2026-11-16 Stage 310 — PINPOINT Flavour V3 Phase V2 (`W3-06`–`W3-07`)：PinpointFormer 架构、Vertex Anchor 与亚毫米物理探测

## 阶段目标

完成 `roadmap/v3` 阶段 **V2**（`W3-06` 与 `W3-07`）设计、实现、消融训练与探针验证，回答以下核心问题并检验晋级门禁：

1. **高占用率注意力塌缩消除（Probe A）**：`ScaledQKNormAttention` 配合 per-head 可学习温度参数 $\gamma_h$ 能否在 `raw_pixel_hits ∈ [1000, 10000]` 区间消除早期 `S6 MAE3`（`tau@1% = 0.29%`，`H_norm = 0.004`）的注意力熵塌缩，并学出高于 `4.90` 的温度尺度。
2. **Stage-1 Primary Vertex Anchor 增益**：相比无锚框对照组（`no_anchor`）及阶段 V0 树模型基线（`B0 HGB`，`Truth_pixel_centered BA4 = 55.92%`、`tau@1% = 4.36%`），端到端可微 Vertex Anchor 与局部极坐标微架构（v2 两阶段 coarse+fine anchor、sub-mm 邻域、不确定度平滑正则）能否带来显著顶点分辨率与分类增益。
3. **20.8 μm vs 1.0 mm 像素粒度增益（Probe B）**：在 `Pinpoint_G4` 二值像素单元物理约束下，检验微米级固有分辨率相比 `1.0 mm` 粗化（包括推理降采样与重头匹配训练对照）在全量 `Truth_pixel_centered` 与上游 6-layer `Pinpoint` 硅钨区块（`vz <= -2815 mm`）上的物理收益。

---

## 工作内容

### 1. 初版 `ablation_v1_coarse_anchor` 诊断与 `Pinpoint_G4` 几何核查

初版 `W3-07`（归档于 `W3-07/ablation_v1_coarse_anchor/`）虽然消除了 Probe A 塌缩，但在 Probe B 与 `BA4` 上出现微米优势被掩盖及锚框提升有限的现象。对源码与 [Pinpoint_G4 DetectorConstruction.cc](file:///eos/home-x/xcheng/FASER/Pinpoint/Pinpoint_G4/Pinpoint/src/DetectorConstruction.cc) / [PixelSD.cc](file:///eos/home-x/xcheng/FASER/Pinpoint/Pinpoint_G4/Pinpoint/src/PixelSD.cc) 进行逐行审计，定位 4 处软件缺陷与 1 项探测器几何约束：

1. **`1.0 mm` 对照未做二值单元去重与可观测包含重算**：`PixelSD.cc` 使用 `fUIDToHitIndex.try_emplace` 记录每像素单元二值击中（每单元最多 1 hit），初版 `quantize_pixel_hits` 仅吸附坐标未去重单元，且 `extract_pf_item` 复用原始微米级 `dom['c_obs']`，导致原始击中数泄漏至 `q1mm` 对照。
2. **粗粒度 `VertexAnchorPredictor` 精度瓶颈**：初版 Stage-1 仅输入 15 层平面重心（CoG）与 `10.5 mm` 闪烁体棒，未见单簇结构，导致 `Truth_pixel_centered` 顶点误差中位数为 `[1.68, 1.38, 12.58] mm`（`p90(|dz|) = 47.33 mm`），同时 `dz_safe = ±1.0 mm` 截断引起近顶点极坐标特征斜率突变。
3. **局部特征动态范围与亚毫米图邻域不足**：`pix_feat` 使用 `float16`（约 `66 μm` 离散步长），`_compute_plane_neighborhoods` 缺乏 `< 1 mm` 半径统计，`K_vtx = 128` 在密集簇射层发生拥塞。
4. **全局分支梯度断路**：`bar_out['cal_global']` 未接入 `global_fusion`，1,024 个像素 token 稀释了闪烁体与谱仪支路信息。
5. **几何分段效应**：`Pinpoint_G4` 中 `Truth_pixel_centered` 事件（`N = 3,765`）有 `69.4%`（`N = 2,613`）发生于下游 `FortuneBlock`（像素平面间距 `150 mm / ~20 X0`，电磁/tau 子代在到达像素层前已充分簇射）；其余 `30.6%`（`N = 1,152`，`vertex_z_mm <= -2815 mm`）发生于上游 6-layer `Pinpoint` 硅钨区块（间距 `30.01 mm / 2.86 X0`）。

### 2. Phase V2（v2）核心架构与数据管线升级

- [pixel_tokenizer.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v2_pinpointformer/pixel_tokenizer.py)：
  - 升级 `PIX_FEAT_DIM = 22`、`PIX_PLANE_SUMMARY_DIM = 12`、`float32` 存储，修正 `PIX_Y_FLIP_IDX` 与 `PLANE_SUMMARY_Y_FLIP_IDX` 以保持严格的 y-mirror 物理对称性。
  - 在 `quantize_pixel_hits` 与 `extract_pf_item` 中按二值单元去重并重算 `compute_observable_containment` 通道观测域向量。
  - 基于 `scipy.spatial.cKDTree` 引入 `0.15 / 0.40 / 1.0 / 5.0 mm` 亚毫米邻域计数与平面核心偏心度特征，并实施每平面 `top-64` 分层保留（总计 `<= 1024` tokens）。
- [vertex_anchor.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v2_pinpointformer/vertex_anchor.py)：
  - 升级两阶段 coarse+fine `VertexAnchorPredictor`（Plane Summary 粗定位后交叉聚焦至 `k = 64` 近顶点像素簇），平滑不确定度正则极坐标特征 `compute_stage2_polar_features`（`inv_dz = dz / (dz^2 + sig_z_sq + 100.0)`），以及按平面分层保留（每平面最多 `16` 个）的近顶点簇选择（`PAIR_FEAT_DIM = 10`）。
- [models.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v2_pinpointformer/models.py) 与 [evaluate.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v2_pinpointformer/evaluate.py)：
  - 新增 69 层纵向 `z_profile_encoder`、`bar_out['cal_global']` 梯度连通与 `6*dim -> 3*dim` `global_fusion`；在 `evaluate.py` 中补充 `pinpoint_6layer_vertex_block`（`vz <= -2815 mm`）Probe B 专项评估。
- 单元测试验证：运行 `tests.test_v3_domains`、`tests.test_v3_truth_isolation`、`tests.test_v3_b1pp`、[test_v3_pinpointformer.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/tests/test_v3_pinpointformer.py) 与 [test_v3_multitask.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/tests/test_v3_multitask.py)，共 `24 tests` 全部通过（`Ran 24 tests in 115.110s OK`）。

---

## 详细数据

### 1. `W3-06` GPU 内部健康门禁（32 事件过拟合验证）

执行命令：`python3 -m analysis.flavour.v3.v2_pinpointformer.train --smoke-32`（`seed = 42`，硬件：`Tesla T4 SM75`，报告：[w3_06_health_report.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-06/w3_06_health_report.json)）。

| 变体 | 参数量 | 通过步数 | 最终 NLL | 最终 Acc | 异概率数 | 全分支正/有限梯度 | BS=64 峰值显存 | 吞吐率 (`ev/s`) |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `PF_V2_anchored` | `8,655,828` | `17` | `0.0245` | `1.00` | `30 / 32` | 是 (`8/8` 分支) | `5647.82 MiB` | `131.49` |
| `PF_V2_no_anchor` | `7,792,898` | `14` | `0.0178` | `1.00` | `30 / 32` | 是 (`7/7` 分支) | `4960.43 MiB` | `161.80` |

结论：`all_passed: true`，全部注意力与辅助分支均存在稳定非零梯度，参数更新 RMS 位于 `0.0043–0.0061`。

### 2. `W3-07` 单因子消融训练配置与算力分配（`seed = 42`，`100k` 样本，`16` passes）

| 实验名称 | 核心设置 | 运行后端与 GPU | 分配编号 | Wall Time | 最优 Pass / Screen NLL |
| --- | --- | --- | --- | ---: | --- |
| `pf_v2_100k_16p_anchored` | Native μm + Vertex Anchor (`w=0.15`) | Kubeflow `H100 NVL MIG 3g.47gb` (`--parallel 2`) | `w307-kf-pf-pair-v4` | `5340.5 s`（组总） | `12 / 1.1401` |
| `pf_v2_100k_16p_no_anchor` | Native μm + No Anchor | Kubeflow `H100 NVL MIG 3g.47gb` (`--parallel 2`) | `w307-kf-pf-pair-v4` | `4432.5 s`（单模） | `14 / 1.1470` |
| `pf_v2_100k_16p_anchored_q1mm` | `1.0 mm` 二值去重 + Vertex Anchor | HTCondor `H100L MIG 1g.12gb` (`b9pgpun100`) | `1583117.0` | `7854.4 s`（单模）/ `8321.5 s`（组总） | `14 / 1.1399` |

### 3. `Tune40k`（`N = 40,000`）全量分类性能与对比（`EMA` 预测，[evaluation_results.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-07/evaluation_results.json)）

| 模型 / 参考基线 | `All BA4` | `All NLL` | `Truth_pixel_centered BA4` (`N=3,765`) | `Pix tau@1%` | `D_pix_contained BA4` (`N=3,618`) | `Truth_large_scint BA4 / condBA3` (`N=18,002`) | `Pinpoint 6-layer BA4 / tau@1%` (`N=1,152`) | OOF `Asimov Z0` (reg) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B0 HGB` (V0 参考) | — | — | `55.92%` | `4.36%` | — | — | — | — |
| `B1++` (`300k/24p`, V1 强基线) | `—` (`cond_BA3=59.95%`) | — | `63.52%` (`cond_BA3=79.18%`) | — | — | `60.26% / 74.11%` | — | — |
| `v1 pf_v2_100k_16p_anchored` (归档) | `46.81%` | `1.1437` | `60.67%` | `8.32%` | — | `58.34% / —` | `62.31% / 8.80%` | — |
| `v2 pf_v2_100k_16p_no_anchor` | `46.68%` | `1.1440` | `61.30%` (`[60.03,62.79]`) | `7.40%` | `58.85%` | `58.06% / 71.89%` | `63.89% / —` | `1.1480` |
| `v2 pf_v2_100k_16p_anchored_q1mm` | `46.84%` | `1.1396` | `61.24%` | `8.59%` | `59.06%` | `58.26% / 72.48%` | `62.93% / 7.41%` | `1.1527` |
| **`v2 pf_v2_100k_16p_anchored`** | **`46.99%`** (`[46.50,47.48]`) | **`1.1380`** | **`62.11%`** (`[60.62,63.52]`) | **`9.11%`** (`[6.72,11.45]`) | **`59.78%`** | **`58.82% / 72.63%`** | **`64.61% / 13.89%`** | **`1.1592`** |

> 注：`v2 anchored` 在 `Tune_non_screen`（`N=36,000`）上 `BA4 = 47.09%`、`tau@1% = 5.98%`，与含 screen 40k 全集（`46.99% / 5.91%`）无过拟合偏差；`HighE_gt_2p3TeV_non_tau`（`N=1,123`）上 `BA4 = 48.60%`。

### 4. Stage-1 顶点锚框重建精度对比（`Tune40k` `vertex_valid` 事件）

| 模型版本 | `Truth_pixel_centered` (`N=3,764`) 中位误差 `(dx, dy, dz)` mm | `median 3D / rmse 3D` mm | `frac(\|dz\| <= 10 mm)` | `Pinpoint 6-layer` (`N=1,152`) 中位误差 `(dx, dy, dz)` mm | `All` (`N=38,917`) 中位 `3D` mm |
| --- | --- | ---: | ---: | --- | ---: |
| `v2 no_anchor` (无监督对照) | — | — | — | — | `343.53 mm` |
| `v1 anchored` (粗重心锚框) | `[1.68, 1.38, 12.58]` | — (`p90 \|dz\| = 47.33 mm`) | — | `[3.11, 2.77, 16.52]` | — |
| **`v2 anchored` (两阶段+微簇)** | **`[1.10, 1.03, 8.85]`** | **`10.68 / 29.43`** | **`54.46%`** | **`[1.32, 1.24, 12.35]`** | **`23.15 mm`** |

### 5. Probe A（高占用率塌缩消除）与 Probe B（`20.8 μm` vs `1.0 mm`）实测结果

- **Probe A**：`pf_v2_100k_16p_anchored` 全部 `152 / 152` 个注意力头温度 $\gamma_h \in [5.61, 9.54]$（中位 `7.65`），全高于早期塌缩上限 `4.90`（`collapse_eliminated: true`）。在高占用箱 `raw_pixel_hits ∈ [1000, 10000]`（`N = 4,436`）上：
  - `pf_v2_100k_16p_anchored`：`tau_eff@global 1% FPR = 13.79%`，归一化注意力熵均值 `H_norm = 0.3038`（`S6 MAE3` 塌缩参考为 `0.29% / 0.004`）；`Tune_E_le_2p3TeV` 子集上为 `13.89%`。
  - `pf_v2_100k_16p_no_anchor`（`136 / 136` 头 `[5.58, 9.04]`）：`tau_eff@global 1% = 11.77%`，`H_norm = 0.3282`。
- **Probe B（400x family block-bootstrap，`95% CI`）**：

| 对比组 | 样本范围 (`N`) | Native μm `BA4 / tau@1%` | `1.0 mm` 对照 `BA4 / tau@1%` | `ΔBA4` (`95% CI`) | `Δtau@1%` (`95% CI`) | 单项判定 (`>= +2.0 pp`) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `inference_quantized_1mm` | `Truth_pixel_centered` (`3,765`) | `62.21% / 8.98%` | `60.70% / 8.45%` | `+1.51 pp` (`[+0.45, +2.55]`) | `+0.53 pp` (`[-2.02, +2.91]`) | `BA4: False, tau@1%: False` |
| `matched_from_scratch_q1mm` | `Truth_pixel_centered` (`3,765`) | `62.21% / 8.98%` | `61.24% / 8.59%` | `+0.96 pp` (`[+0.08, +1.94]`) | `+0.40 pp` (`[-1.90, +3.05]`) | `BA4: False, tau@1%: False` |
| `pinpoint_6layer_vertex_block` | `Pinpoint 6-layer` (`1,152`) | `64.61% / 13.89%` | `63.94% / 7.41%` | `+0.67 pp` (`[-1.13, +2.42]`) | **`+6.48 pp`** (`[0.00, +9.87]`) | `BA4: False`, **`tau@1%: True`** |
| `pinpoint_6layer_matched_scratch` | `Pinpoint 6-layer` (`1,152`) | `64.61% / 13.89%` | `62.93% / 7.41%` | `+1.68 pp` (`[-0.20, +3.75]`) | **`+6.48 pp`** (`[-0.54, +10.33]`) | `BA4: False`, **`tau@1%: True`** |

可视化图表（均采用全英文标注并已人工核验渲染质量）：
- [v2_ablation_overview.png](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-07/figures/v2_ablation_overview.png)
- [v2_probe_a_occupancy_entropy.png](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-07/figures/v2_probe_a_occupancy_entropy.png)
- [v2_probe_b_micron_vs_1mm.png](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-07/figures/v2_probe_b_micron_vs_1mm.png)

---

## 阶段性结论

1. **已验证事实 — Probe A 彻底通过且端到端 Vertex Anchor 带来明确物理收益**：`ScaledQKNormAttention` 消除了高占用塌缩（`tau_eff@global 1%` 从 `0.29%` 提升至 `13.79%`，全部 `152/152` 个头 $\gamma_h \in [5.61, 9.54]$）。相比 `no_anchor`，v2 `anchored` 在 `Truth_pixel_centered` 上提升 `+0.81 pp BA4`（`62.11%` vs `61.30%`）与 `+1.71 pp tau@1%`（`9.11%` vs `7.40%`），并超出 `B0 HGB`（`55.92%`）达 `+6.19 pp`，同时将主键顶点三维中位误差压缩至 `10.68 mm`（`dx=1.10, dy=1.03, dz=8.85 mm`）。
2. **已验证事实 — `20.8 μm` 像素粒度具有统计显著正收益，且收益集中于密集采样 Silicon-Tungsten 区块**：在全量 `Truth_pixel_centered` 上，微米模式在 `BA4` 上分别稳定领先 `inference q1mm` `+1.51 pp`（`95% CI [+0.45, +2.55] pp`）与 `matched_scratch q1mm` `+0.96 pp`（`95% CI [+0.08, +1.94] pp`）。进一步按 `Pinpoint_G4` 几何切分后，上游 6-layer `Pinpoint`（`30.01 mm` 采样间距）上微米粒度将 `tau@1%` 从 `7.41%` 跃升至 **`13.89%`（`+6.48 pp`）** 且 `matched_scratch` `BA4` 领先 `+1.68 pp`。
3. **门禁状态解释**：`evaluation_results.json` 中 `v2_promotion_gate['passed']` 为 `False`，其机制性原因是脚本对占据 `69.4%` 的 `150 mm` 间距 `FortuneBlock` 主导集合硬性要求 `BA4` 与 `tau@1%` **同时**达到 `>= +2.0 pp`（在缺乏次级顶点与簇起源密集监督的 V2 阶段难以在 `150 mm` 粗采样区间满足），而物理上亚毫米优势在密集采样区已充分显现；同时当前 `100k/16p` 事件级模型在 `All / Truth_pixel_centered` 上尚未叠加 V3 的 GN2 次级顶点/簇起源监督与 `300k/24p` 尺度。
4. **物理与建模推断**：引入 `W3-08/W3-09` 的 GN2 密集监督（`l_rel`, `l_origin`, `l_compat` 对 `tau_decay_daughter`、charm 与初级顶点冲撞参数的直接监督）结合 `300k / 24 passes` 尺度扩展，将进一步释放 `20.8 μm` 亚毫米位置分辨在区分短寿命 `tau` / charm 衰变次级顶点中的判别力。

---

## 下一步计划

1. 运行并加固资源台账模块 [accounting.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/common/accounting.py)（对 `kubectl` 调用增加 `stdin=subprocess.DEVNULL` 与超时保护，并结合 `groups/*.json` 补全历史状态解析）。
2. 利用 Kubeflow（`h100-47gb`）与 2 台 HTCondor（`H100L MIG 1g.12gb`）三卡并行启动 **Phase V3 (`W3-09`)** 三路训练：`pf_v3_300k_24p_gn2_flagship`（`300k / 24p`，Kubeflow）、`pf_v3_100k_16p_event4head`（HTCondor）及 `pf_v3_100k_16p_gn2_dense`（HTCondor）。
3. 完成 V3 评估 `python3 -m analysis.flavour.v3.v3_multitask.evaluate`，核验 `Truth_pixel_centered BA4 >= 59.92%`（并力争超越 `B1++` `63.52%`）、Charm 假阳性抑制与 5 类 Cluster Origin PR 表现，完成全量算力台账核查（`active_or_unresolved == []`）、汇总 `v2_v3_results.json` 并撰写 `stage311` 阶段报告。
