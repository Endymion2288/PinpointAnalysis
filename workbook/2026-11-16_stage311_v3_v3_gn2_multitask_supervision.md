# 2026-11-16 Stage 311 — PINPOINT Flavour V3 Phase V3 (`W3-08`–`W3-09`)：PinpointFormer-GN2 多任务密集监督与旗舰验证

## 阶段目标

完成 `roadmap/v3` 阶段 **V3**（`W3-08` 与 `W3-09`）设计、训练、多维度物理评估与算力台账验收，验证以下目标：

1. **事件级 4 头同方差多任务监督（`PF_V3_event4head`）**：引入 7 类精细物理分支（含 $\tau \to e / \mu / \text{hadron}$）及辅助 Charm、初级顶点与可见运动学监督，对比 V2 单一事件头分类收益与顶点重建提升。
2. **ATLAS GN2 风格像素簇起源与轨道兼容密集监督（`PF_V3_gn2_dense` & `Flagship 300k/24p`）**：结合近顶点簇 5 类起源分类（Primary Lepton / Primary Hadron / Tau Daughter / Charm Daughter / Secondary Shower）与全探测器关联边/兼容监督，释放 `20.8 μm` 亚毫米像素分辨在重味与 $\tau$ 次级顶点鉴别中的潜力。
3. **V3 晋级门禁与消融验证**：检验旗舰模型是否在 `Truth_pixel_centered` 上稳定超越 `B0 + 4.0 pp`（`>= 59.92%`）乃至 V1 强基线 `B1++`（`63.52%`）、消除 Probe A 塌缩、通过全量 `Truth_pixel_centered` Probe B，并完成全流程无残留算力台账核对。

---

## 工作内容

1. **只读 `Tune40k` 离线审计索引构建（`W3-08`）**：
   - 执行 [tune_audit.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/common/tune_audit.py) `--finalize-only`，汇总 16 个验证 shard（`040..055`），产出 [tune_event_audit.parquet](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-08/tune40k_offline_audit/tune_event_audit.parquet)、memmap `cluster_audit/` 及 [audit_summary.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-08/tune40k_offline_audit/audit_summary.json)，严格隔离验证集（`Cal_A_Cut_B_Confirm_payload_read: false`），并清理 `491 MB` 中间 `shards/`。
2. **`W3-08` 交互式 GPU 健康门禁与 `W3-09` 三卡并发训练**：
   - 通过 `python3 -m analysis.flavour.v3.v3_multitask.train --smoke-32` 校验 32 事件过拟合、全分支梯度正值有限性与 `BS=64` 显存/吞吐。
   - 在 Kubeflow（`H100 NVL MIG 3g.47gb`）与 2 台 HTCondor（`H100L MIG 1g.12gb`）三卡并发训练 `pf_v3_300k_24p_gn2_flagship`、`pf_v3_100k_16p_event4head` 与 `pf_v3_100k_16p_gn2_dense`（统一种子 `seed = 42`），完结后释放 K8s 资源并清理全部 `resume.pt`。
3. **评估管线兼容修复与多任务评估（`W3-09`）**：
   - 修复 [evaluate.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v3_multitask/evaluate.py) 对 V1 `b1pp_300k_24p_t4` 概率列命名（`p0..p3` vs `p_0..p_3`）及参考指标字段（`slices` / `BA4_raw`）的兼容读取，运行全量评估并视觉核查全部 4 张图表。
4. **算力台账加固与全量回归测试**：
   - 加固 [accounting.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/common/accounting.py)（对 `kubectl` 调用增加 `--request-timeout=5s`、`stdin=subprocess.DEVNULL` 与 `timeout=10`，并关联 `groups/*.json` 回退解析历史作业），核对 [resources.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/resources.json) 全量 `48` 次提交。
   - 汇总结果至 [v2_v3_results.json](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/roadmap/v3/v2_v3_results.json) 并更新 [experiment_plan_v3.json](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/roadmap/v3/experiment_plan_v3.json)。

---

## 详细数据

### 1. `W3-08` `Tune40k` 离线审计与 GPU 健康门禁

- **离线审计摘要**（[audit_summary.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-08/tune40k_offline_audit/audit_summary.json)）：`split_strict = validation_only`，`events = 40,000`，`total_pixel_clusters = 14,835,269`，`charm_positive = 3,121`，`vertex_valid = 38,917`；5 类真实起源分布为：`Primary_lepton = 4,346,310`、`Primary_hadron = 127,964`、`Tau_daughter = 677,025`、`Charm_daughter = 412,840`、`Secondary_shower = 9,271,130`（归档耗时 `6.49 s`）。
- **GPU `--smoke-32` 健康门禁**（[w3_08_health_report.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-08/w3_08_health_report.json)，`Tesla T4 SM75`）：

| 变体 | 参数量 | 通过步数 | 最终 NLL | 最终 Acc | 全分支正/有限梯度 | BS=64 峰值显存 | 吞吐率 (`ev/s`) |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `PF_V3_event4head` | `8,881,440` | `20` | `0.0354` | `1.00` | 是 (`11/11` 分支) | `5653.28 MiB` | `137.55` |
| `PF_V3_gn2_dense` | `8,977,005` | `16` | `0.0368` | `1.00` | 是 (`12/12` 分支) | `5714.83 MiB` | `121.21` |

### 2. `W3-09` 三路模型训练配置与耗时（`seed = 42`）

| 模型实验 | 样本 / Passes | 运行后端与 GPU | 分配编号 | Wall Time (组/批) | 最优 Pass / Screen NLL |
| --- | --- | --- | --- | ---: | --- |
| `pf_v3_100k_16p_event4head` | `100k / 16p` | Condor `H100L MIG 1g.12gb` (`b9pgpun108`) | `1583136.0` | `8772.0 s` (`8784.0 s`) | `12 / 1.1273` |
| `pf_v3_100k_16p_gn2_dense` | `100k / 16p` | Condor `H100L MIG 1g.12gb` (`b9pgpun100`) | `1583137.0` | `9169.8 s` (`9190.0 s`) | `14 / 1.1131` |
| `pf_v3_300k_24p_gn2_flagship` | `300k / 24p` | Kubeflow `H100 NVL MIG 3g.47gb` | `w309-kf-v3-flag` | `11323.9 s` (`11354.0 s`) | `16 / 0.9976` |

### 3. `Tune40k` 全量分类、校准与顶点精度演化（`EMA` 预测，[evaluation_results.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-09/evaluation_results.json)）

| 模型阶段 | `All BA4` (`N=40,000`) | `All NLL` | `Truth_pixel_centered BA4` (`N=3,765`) | `Pix condBA3` | `Pix tau@1%` | `Pix tau@0.1%` | `D_pix_contained BA4` (`N=3,618`) | `Truth_large_scint BA4 / condBA3` (`N=18,002`) | `All Asimov Z0` (reg) | `VtxPix` 中位 `(dx,dy,dz)` mm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B0 HGB` (V0 参考) | — | — | `55.92%` | — | `4.36%` | — | — | — | — | — |
| `B1++ 300k` (V1 强基线) | — | — | `63.52%` | `79.18%` | `10.57%` | — | — | `60.26% / 74.11%` | — | — |
| `V2 PF Anchored 100k` | `46.99%` | `1.1380` | `62.11%` | `77.96%` | `9.11%` | `2.25%` | `59.78%` | `58.82% / 72.63%` | `1.1592` | `[1.10, 1.03, 8.85]` |
| `V3 Event4Head 100k` | `47.77%` | `1.1286` | `62.65%` | `79.14%` | `10.17%` | `4.10%` | `60.41%` | `59.68% / 73.71%` | `1.1737` | `[0.98, 0.92, 6.04]` |
| `V3 GN2 Dense 100k` | `48.31%` | `1.1152` | `64.08%` | `79.97%` | `10.17%` | `2.91%` | `61.65%` | `60.64% / 74.47%` | `1.1886` | `[1.13, 1.05, 12.69]` |
| **`V3 GN2 Flagship 300k`** | **`52.97%`** (`53.26%` Best) | **`1.0226`** | **`73.53%`** (`[72.24,75.04]`) | **`86.43%`** | **`27.48%`** (`[23.40,32.42]`) | **`14.80%`** | **`71.10%`** | **`69.36% / 82.01%`** | **`1.3964`** | **`[0.87, 0.87, 10.21]`** |

> 补充切片（`V3 Flagship EMA`）：`Tune_non_screen`（`N=36,000`）`BA4=52.95%`、`tau@1%=13.60%`；`Tune_E_le_2p3TeV`（`N=38,874`）`BA4=52.79%`、`tau@1%=13.53%`；`HighE_gt_2p3TeV_non_tau`（`N=1,123`）`BA4=60.66%`；`Flagship Best` 检查点在 `Truth_pixel_centered` 上达 `NLL=0.6099`、`BA4=73.17%`、`tau@1%=28.01%`、`tau@0.1%=15.46%`、`cond_BA3=86.64%`。

### 4. Charm 拒绝审计与 `S_tau(lambda_charm)` 扫描

- `Tune40k` 中含 `2,046` 个非 $\tau$ Charm 事件（其中 `Truth_pixel_centered` 含 `192` 个）。`Flagship` 显式 Charm 头在非 $\tau$ 全集上 `AUROC = 0.6900`，在 `Truth_pixel_centered` 上达 **`AUROC = 0.7831`**（较 `100k event4head` `0.7013` 提升 `+0.0818`）。
- 在 `lambda_charm = 0.0` 无惩罚工作点，`Flagship` 因 `tau@1%` 大幅跃升至 `27.87%`（OOF regional），`Truth_pixel_centered` 上 Charm 误判率为 `1.56%`（`3 / 192`，相对 `B1++`/`V2` `0.52% = 1 / 192` 未降低，故 `charm_fpr_reduced_vs_v2 = false`）。利用显式 Charm 惩罚扫描 `lambda_charm`：
  - `lambda_charm = 1.0`：`Charm FPR` 降至 `0.52%`（持平 `B1++`/`V2`），同时 `tau@1%` 保持 **`24.57%`**。
  - `lambda_charm = 1.5 / 2.0`：`Truth_pixel_centered Charm FPR` 降至 **`0.00%`**（`0 / 192`），同时仍保留 **`23.91% / 22.59%`** 的 $\tau$ 效率（较 `B1++` 高 2 倍以上）。

### 5. 近顶点 Cluster 5 类起源分类与 Flagship Probe A / Probe B

- **Cluster Origin（`Truth_pixel_centered`，`441,671` clusters）**：`macro_f1 = 0.4615`、`macro_pr_auc = 0.4892`（全量 `2,139,429` clusters：`0.4478 / 0.4735`）。各类指标为：`Primary_lepton`（`PR-AUC=0.776, F1=0.714`）、`Primary_hadron`（`0.417 / 0.402`）、`Tau_daughter`（`0.266 / 0.299`）、`Charm_daughter`（`0.073 / 0.039`）、`Secondary_shower`（`0.915 / 0.854`）。
- **Flagship Probe A**：`collapse_eliminated = true`，高占用箱 `[1000-10000]`（`N=4,436`）达到 **`tau@global 1% = 34.46%`**（`H_mean = 0.4045`），`152 / 152` 个注意力头 $\gamma_h \in [2.25, 6.28]$（中位 `3.27`，`4` 头 `> 4.90`）。
- **Flagship Probe B 与 `v3_promotion_gate`**：

| 测试范围 (`N`) | Native μm `BA4 / tau@1% / tau@0.1%` | `1.0 mm` 推理对照 | `ΔBA4` (`95% CI`) | `Δtau@1%` (`95% CI`) | `Δtau@0.1%` | 判定 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `Truth_pixel_centered` (`3,765`) | **`73.54% / 27.48% / 14.80%`** | `69.47% / 20.87% / 10.17%` | **`+4.08 pp`** (`[+2.94, +5.19]`) | **`+6.61 pp`** (`[+3.20, +10.31]`) | `+4.62 pp` | **`passes_probe_b: True`** |
| `Pinpoint 6-layer` (`1,152`) | **`74.03% / 31.48% / 21.30%`** | `69.75% / 25.00% / 12.04%` | **`+4.29 pp`** (`[+2.15, +6.46]`) | **`+6.48 pp`** (`[+0.87, +12.92]`) | `+9.26 pp` | **`passes_probe_b: True`** |

`v3_promotion_gate` 正式结果：`passed = True`（`beats_b0_plus_4pp = True`、`probe_a_collapse_eliminated = True`、`probe_b_passes = True`）。

### 6. 全阶段算力台账与回归验证

- **算力台账（[resources.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/resources.json)）**：累计 `48` 次提交（`W3-02: 4.29 cpu·h`、`V1: 36.23 cpu·h / 10.67 gpu·h`、`W3-07: 18.86 cpu·h / 30.63 gpu·h` 含早期排障重跑、`W3-08: 10.87 cpu·h`、`W3-09: 19.97 cpu·h / 8.15 gpu·h`），总计 `90.221 Condor CPU core·h`、`49.441 GPU·h`，`active_or_unresolved: []`。
- **可视化图表**：[v3_progression_overview.png](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-09/figures/v3_progression_overview.png)、[v3_charm_rejection_and_stau.png](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-09/figures/v3_charm_rejection_and_stau.png)、[v3_cluster_origin_confusion_pr.png](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-09/figures/v3_cluster_origin_confusion_pr.png)、[v3_flagship_probes.png](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-09/figures/v3_flagship_probes.png)。

---

## 阶段性结论

1. **已验证事实 — GN2 密集多任务监督 + `300k/24p` 带来跨量级跃升**：`pf_v3_300k_24p_gn2_flagship` 在 `Truth_pixel_centered` 上将 `BA4` 推升至 **`73.53%`**（超过 V0 `B0 HGB` `+17.61 pp`、超过 V1 `B1++ 300k` `+10.01 pp`、超过 V2 `Anchored` `+11.42 pp`），并将 `tau@1% FPR` 提升至 **`27.48%`**（近 3 倍于 `V2 9.11%` 与 `V1 B1++ 10.57%`）；在 `Truth_large_scintillator` 上亦由 V1 `60.26%` 跃升至 **`69.36%` (`+9.10 pp`)**；即使同为 `100k/16p`，加入 GN2 密集监督的 `gn2_dense`（`64.08%`）亦已反超 `300k/24p` `B1++`（`63.52%`）。
2. **已验证事实 — `20.8 μm` 亚毫米精度优势在密集次级顶点监督下全面释放**：全量 `Truth_pixel_centered` Probe B 从 V2 的 `delta_ba4=+1.51 pp / delta_tau=+0.53 pp` 跃升至 **`+4.08 pp BA4`**（`73.54%` vs `69.47%`）与 **`+6.61 pp tau@1%`**（`27.48%` vs `20.87%`），上游 6-layer `Pinpoint` 更达 `74.03% BA4 / 31.48% tau@1% / 21.30% tau@0.1%`（`+9.26 pp tau@0.1%`）。这直接印证：**纯事件级监督不足以驱动表征提取短寿命 $\tau$/Charm 衰变位移拓扑；而 GN2 簇起源与关联兼容性监督使网络真正学会利用微米级亚结构**。
3. **显式 Charm 头提供干净可控的背景抑制度**：尽管 `lambda_charm = 0.0` 下随着 $\tau$ 接收度倍增，`Charm FPR` 升至 `1.56%`，但 `Flagship` Charm 判别头本身具备 `0.7831 AUROC`，在 `S_tau` 中仅引入 `lambda_charm = 1.5` 即可将 `Truth_pixel_centered` Charm 误判清零（`0.00%`），同时保留 `23.91%` 的 `tau@1%`。

---

## 下一步计划

1. 按 `roadmap/v3` 推进 **V4（`W3-10`–`W3-11`）**：双流物理特征融合、`DRO/CVaR` 单边部分 AUC（`@1% FPR`）尾部精调与 `PinpointFormer-GN2` + `B1++` 跨家族集成，以及 `lambda_charm` 联合工作点固化。
2. 完成 V4 冻结后执行 **V5（`W3-12`–`W3-13`）** 3-seed 稳定性扩展（`17, 42, 2026`）与 `Cal-A40k -> Cut-B160k -> Confirm200k` 单次解封统计检验。
