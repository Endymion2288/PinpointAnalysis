# Stage 308: Roadmap v3 Phase V0 — 纯观测域划分、Truth Sidecar 提取与零训练区域校准回放 (`W3-01`–`W3-03`)

## 阶段目标

完成 `roadmap/v3/` Phase V0（`W3-01`–`W3-03`），解决 V2 中阻碍 `Pixel-centered` 性能释放与样本隔离的三项基础工程问题，并建立无泄漏的 V3 基础设施：

1. **W3-01（纯观测含能域与公共能量支撑）**：构建严格基于可观测量（`pixel`、`bar`、`tracker_hypothesis`）的 16 维含能描述子 `c_obs` 与 4 个互斥观测域 `obs_domain_id`（`D_empty=0`, `D_pix_contained=1`, `D_scint_contained=2`, `D_partial_or_edge=3`）；验证 `D_pix_contained` 对 truth `pixel_centered` 的纯度 $\ge 75\%$ 且覆盖率 $\ge 80\%$；冻结公共能量支撑切片（`<= 2300.0 GeV`）与 `energy_match_reweight` 权重。
2. **W3-02（训练集 Truth Sidecar 与零泄漏隔离）**：基于 Geant4 `TrackingAction.cc` 真实轨迹继承关系，为 `Train300k` 提取事件级 `label_7`/`charm`/`vertex`/`kinematics` 与像素簇/闪烁体条粒度的 `truth_sidecar`；通过 AST 与运行时测试保证 `V3EvalDataset` 无法接收或读取 `sidecar_root` 或 `Cal-A40k`/`Cut-B160k`/`Confirm200k` 负载。
3. **W3-03（零训练区域校准回放）**：在不重训现有 `R2 B1-D2 (Tune40k)` 与 `S6 MAE3 (S4 Val10k / S6 Test724k)` 模型的前提下，通过 5-fold family-connected out-of-fold 区域校准 $(T_d, b_d)$ 与区域中性判别统计量 $S_\tau$ 验证是否能够消除全局阈值下 `Pixel-centered` 效率坍塌。

---

## 工作内容

1. **纯观测含能特征与闪烁体面板几何修正（`analysis.flavour.v3.common.domains` & `spectrum`）**
   - 从探测器几何（`geometry.json`）推导物理面板方向与辐射长度 $X_0$ 累积深度：实际闪烁体共 `96` 个面板，为 `54` 块垂直（$X$ 读出）与 `42` 块水平（$Y$ 读出），总计 `111` 个 $z$ 有序敏感平面；统一修正目录能量单位（MeV $\to$ GeV，`nu_tau` 最大能量 `2311.8 GeV`，确立 `2300.0 GeV` 公共能量上限）。
   - 实现 `compute_observable_containment` 提取 16 维 `c_obs`、4 互斥观测域 `obs_domain_id` 以及用于谱仪反推的纯观测簇射起点 `cal_start`。
   - 冻结 `W3-01` 输出与 `energy_match_weight` 20-bin 重加权表；通过 `tests/test_v3_domains.py` 4 项单元测试（0 泄漏 AST 检查与硬件分区一致）。

2. **Cluster/Bar Truth Sidecar 与批量 Extraction（`analysis.flavour.v3.common.pixel_clusters`, `truth_sidecar`, `datasets`, `submit`）**
   - 对每个像素平面独立执行 $r = 0.06\text{ mm}$ `cKDTree` 连通域聚类（`cluster_pixel_hits`），并与 `Pinpoint_G4` 原始 `ROOT` 粒子轨迹祖先关联（映射优先级：`fromTau -> Tau_daughter`，`fromCharmedHadron -> Charm_daughter`，`fromPrimaryLepton | (CC nu_e & fromPrimaryEMShower) -> Primary_lepton`，`parentID == 0 -> Primary_hadron`，其余为 `Secondary_shower_or_other`）。
   - 向 `HTCondor (bigbird20.cern.ch)` 提交 16 个 4-CPU 并行作业（`1583030..1583045`），覆盖全部 `120` 个含训练集分片（跳过仅含验证集的 `040..055` 分片），运行 `truth_sidecar finalize` 汇总为 `event_sidecar.parquet` 与 `cluster_sidecar/*.bin`（零复制 memmap）。
   - 通过 `tests/test_v3_truth_isolation.py` 验证 `V3EvalDataset` 签名与 AST 隔离性、40k 验证集零 sidecar 访问及分片 `000` 端到端一致性。

3. **5-Fold OOF 区域校准回放（`analysis.flavour.v3.common.calibration`）**
   - 在 $(run, chunk)$ `family_hi/lo` 连通 5 折交叉验证框架下按观测域 $d \in \{0,1,2,3\}$ 拟合标量温度与偏置 $(T_d, b_d)$（L-BFGS 优化交叉熵）及物理背景似然比统计量 $S_\tau = \frac{p_d(\tau)}{p_d(\tau) + \sum_{k \neq \tau} \pi_{k|d} p_d(k)}$，并比较原始全局切割、Strategy I 全局 $S_\tau$ 与 OOF Neyman-Pearson 分域预算分配 Strategy II。

---

## 详细数据

### W3-01：观测域混淆矩阵与公共能量支撑

命令：`python3 -m analysis.flavour.v3.common.domains --workers 16`、`python3 -m analysis.flavour.v3.common.spectrum`  
产物目录：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-01/`

| 数据集 | 指标 | 实测值 |
| --- | --- | --- |
| `Train300k` | `D_pix_contained` 对 truth `pixel_centered` 纯度 | **92.46%** |
| `Train300k` | `D_pix_contained` 对 truth `pixel_centered` 覆盖率 | **89.01%** |
| `Train300k` | `D_scint_contained` 对 truth `large_only` 覆盖率 | **95.73%** |
| `Tune40k` | `D_pix_contained` 对 truth `pixel_centered` 纯度 | **92.68%** |
| `Tune40k` | `D_pix_contained` 对 truth `pixel_centered` 覆盖率 | **89.06%** |
| `Tune40k` | `D_scint_contained` 对 truth `large_only` 覆盖率 | **95.65%** |
| `Tune40k` | `Tune_E_le_2p3TeV`（$\le 2300.0\text{ GeV}$）样本数 | **38,874** |
| `Tune40k` | `HighE_gt_2p3TeV_non_tau`（$> 2300.0\text{ GeV}$）样本数 | **1,123** |

所有阈值均大幅超过 V0 验收线（纯度 $\ge 75\%$、覆盖率 $\ge 80\%$）。

### W3-02：Train300k Truth Sidecar 提取统计

命令：`python3 -m analysis.flavour.v3.common.truth_sidecar finalize`（合并 120 个训练分片，耗时 `114.81 s`）  
产物：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/truth_sidecar/{event_sidecar.parquet, cluster_sidecar/*.bin, coverage_summary.json}`

- 事件总数：**300,000**（`label_7`: `e_CC=75,000`, `mu_minus_CC=59,477`, `mu_plus_CC=15,523`, `tau_e=13,357`, `tau_mu=12,830`, `tau_had=48,778`, `NC=75,000`, `unknown_tau=35`）
- 有效率：`tau_branch_valid=99.95%`, `charm_valid=100.0%`（charm positive `7.39%`）, `vertex_valid=97.16%`, `kin_valid=74.96%`
- 簇与条总量：**109,533,511** pixel clusters 与 **209,977,167** active bars；cluster 来源分布：`Primary_lepton 29.12%`、`Primary_hadron 0.86%`、`Tau_daughter 4.55%`、`Charm_daughter 2.64%`、`Secondary_shower_or_other 62.83%`
- 合规标记：`Cal_A_Cut_B_Confirm_payload_read: false`；`tests.test_v3_truth_isolation` 3/3 通过（`73.49 s`）。

### W3-03：5-Fold OOF 区域校准回放结果（1% FPR）

命令：`python3 -m analysis.flavour.v3.common.calibration`  
产物：[replay_results.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-03/replay_results.json)、[w3_03_regional_calibration_replay.pdf](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-03/w3_03_regional_calibration_replay.pdf)

| 模型与样本 | 切片 | `raw_global_cut` $\epsilon_\tau$ | `regional_S_tau` $\epsilon_\tau$ | `regional_NP` $\epsilon_\tau$ | `raw_BA4 -> cal_BA4` |
| --- | --- | --- | --- | --- | --- |
| `R2 B1-D2 T4` (`Tune40k`) | `All` ($N=40,000$) | `6.05%` | `6.16%` | `5.99%` | `0.4577 -> 0.4541` |
| `R2 B1-D2 T4` (`Tune40k`) | `Truth_pixel_centered` ($N=3,765$) | `7.40%` | **`10.30%`** | `8.72%` | `0.5866 -> 0.5907` |
| `R2 B1-D2 T4` (`Tune40k`) | `Obs_D_pix_contained` ($N=3,618$) | `6.51%` | **`9.50%`** | `7.87%` | — |
| `R2 B1-D2 T4` (`Tune40k`) | `Truth_large_scintillator` ($N=18,002$) | `10.81%` | **`11.48%`** | `10.52%` | `0.5662 -> 0.5679` |
| `S6 MAE3` (`S4 Val10k`) | `All` ($N=10,000$) | `3.76%` | `3.40%` | `3.72%` | — |
| `S6 MAE3` (`S4 Val10k`) | `Truth_pixel_centered` ($N=885$) | **`0.00%`** | **`7.39%`** | `3.98%` | — |
| `S6 MAE3` (`S4 Val10k`) | `Obs_D_pix_contained` ($N=858$) | **`0.00%`** | **`7.30%`** | `3.93%` | — |

（`S6 MAE3` 在 `S6Test724k` 使用观测代理域特征时，0.1% FPR 下 `Truth_pixel_centered` 仍为 `0.00%`；而在有完整几何信息的 `S4 Val10k` 上，OOF 区域校准直接将 1% FPR 下 `Truth_pixel_centered` 效率从 `0.00%` 恢复至 `7.39%`。）

---

## 阶段性结论

1. **观测域物理依据成立（事实）**：仅用无泄漏观测特征构成的 `D_pix_contained` 在 `Train300k` / `Tune40k` 上对 `Truth_pixel_centered` 达到 `>92%` 纯度与 `89%` 覆盖率；`54 X_bar + 42 Y_bar` 面板分布与 `2300.0 GeV` 上限修正了一直以来的结构与单位假设偏差。
2. **全局 $\tau$ 坍塌确由未校准跨域置信尺度差异引起（事实）**：`R2 B1-D2` 在不重训的情况下仅靠 OOF 区域 $(T_d, b_d)$ 与 $S_\tau$ 便将 `Pixel-centered` $\nu_\tau$ 选择效率由 `7.40%` 提升至 `10.30%`（+2.90 pp），`S6 MAE3 (Val10k)` 由 `0.00%` 恢复至 `7.39%`。
3. **当前局限与假设**：即便完成区域校准，受限于旧架构丢失精确横纵向局部结构与磁谱仪弯曲符号分辨能力，`R2 B1-D2` 与 `S6 MAE3` 的整体 `BA3 ~ 0.57` 与 $\tau$ 效率仍然偏低，有待 V1 的 `ZAlignedStereoNet (B1++)` 原生立体架构改进。

---

## 下一步计划

1. 执行 Phase V1（`W3-04`–`W3-05`）：完成 `ZAlignedStereoNet (B1++)` 交互式 GPU 健康检验与 32 事件过拟合门控，随后并行调度 `Kubeflow` 与 `HTCondor` GPU 资源训练 6 组 `seed=42` `B1++` 配置，完成 Collaboration M1 对齐评估与消融分析（目标 `Truth_large_scintillator` `BA3 >= 73.0%`）。
