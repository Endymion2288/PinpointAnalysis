# 06 分步执行工单、物理探针与工程纪律（`W3-01`–`W3-12`）

**核心工程准则**：v2 阶段曾在 4-epoch 欠收敛模型上耗费大量人力运行数百条 JSON schema 检验；v3 执行**“前期重物理验证与模型收敛、后期终审才做全矩阵静态锁”**原则，12 张工单严格按依赖顺序推进，每一张工单均给出明确代码路径、命令、预算与进出条件。

---

## 1. 反过度官僚化开发纪律（Anti-Bureaucracy Rules）

1. **探索期单随机种子快速迭代（`W3-01`–`W3-10`）**：探索架构、物理特征与损失设计期间**严格只用 `seed = 42`**；禁止为尚有明显硬伤的架构提前铺开 3-seed × 多任务的庞大矩阵。仅在 `W3-11` 对胜出方案扩补 `{17, 2026}` 验证稳定性。
2. **探索期核心评价收敛至四组主指标**：每次试验只在 `Tune40k` 上检查：① `D_pix_contained`（及真值 `pixel_centered`）的 `BA4 / τ@1%`；② `D_scint_contained`（及真值 `large`）的 `BA3 / τ@1%`；③ `Tune_E_le_2p3TeV` 同能谱切片上的 `τ@1% / AUROC_τ`；④ `All` 域全局指标与分区域校准后加权结果。不做冗长繁杂的跨模型全两两数百对检验，直到 `W3-11/W3-12` 终审。
3. **小样本过拟合闸门不可绕过**：任何新模型在进入百小时级别大规模 GPU 训练前，必须在交互节点通过 **`32-event` 真实数据过拟合检验（64–128 updates 后 NLL < 0.05、准确率 $\ge 98\%$、所有物理分支梯度均非零有限）** 才能提交 Condor/Kubeflow。
4. **充足收敛与独立测试封存**：所有正式单测架构一律训练 `16 passes (100k)` 或 `24–32 passes (300k)` 并监控 `Train/Tune gap`；`Cal-A40k / Cut-B160k / Confirm200k` 严格锁死直至 `W3-12`。

---

## 2. 两大必检物理诊断探针（Mandatory Physics Probes）

在 `V2 (W3-07)` 与 `V3 (W3-09)` 完成时，除常规准确率与 NLL 外，**必须运行并交付以下两项物理探针测试**（代码位置：`src/analysis/flavour/v3/common/probes.py`）：

### 探针 A：高占据度 $\tau$ 效率衰减与注意力归一化熵曲线（治好 S6 数学弥散的唯一金标准）
- **测什么**：按像素击中数分成 `0`、`1–100`、`100–1000`、`1000–10000`、`>10000` 五箱，在 `Tune40k`（及 `E_ν ≤ 2.3 TeV` 子集）统计每一箱的 $\tau$ 识别效率 @ `1% FPR` 与所有 `ScaledQKNormAttention` 层的归一化注意力熵 $H_{\text{norm}} = -\frac{\sum_j \alpha_j \ln \alpha_j}{\ln N_{\text{valid}}}$。
- **验收标准**：`S6 MAE3` 在 `1000–10000` 击中箱暴跌至 `0.29%`（只有 `1–100` 箱的 1/21）。v3 `PinpointFormer` 必须消除高占据度断崖，使 `1000–10000` 区间 $\tau$ 效率达到与 `100–1000` 区间相当或更高水平，同时观察到可学习尺度参数 $\gamma_h = \exp(s_h)$ 自动演化出明显高于 `4.90` 的分化值（典型预期 `10–35`）。

### 探针 B：微米原生输入 vs `1 mm` 人工量化消融（检验硬件价值的唯一金标准）
- **测什么**：保持模型权重或同配方重训输入通道结构完全一致，仅对 `Tune40k` 中 `Pixel-centered` 子集的硅像素输入做一个受控物理破坏——将所有像素真实 `(x, y)` 坐标离散量化到 `1.0 mm × 1.0 mm` 网格中心后重新提取连通簇与形态矩（等于人为抹杀 50 倍像素分辨率）。
- **验收标准**：在 `Pixel-centered` 域计算 `(微米真实输入 − 1mm量化输入)` 的 $\Delta \text{BA4}$ 与 $\Delta \epsilon_\tau@1\%$。只有当微米输入显著超越 `1 mm` 量化输入（目标 $\Delta\text{BA4} \ge +2.0\,\text{pp}$ 且 $\Delta\epsilon_\tau@1\% \ge +2.0\,\text{pp}$）时，才能在物理上确证模型真正解锁了硅像素传感器的核心分辨力！

---

## 3. 12 张有序可执行工单（`W3-01`–`W3-12`）

### Phase V0：问题重构、可观测域与侧车基准

#### 工单 `W3-01`：可观测包容度分层与能谱诊断
- **前置依赖**：无（直接基于 `R0` 冻结 `Train300k / Tune40k` 清单）
- **实现文件**：`src/analysis/flavour/v3/common/domains.py`、`src/analysis/flavour/v3/common/spectrum.py`、`tests/test_v3_domains.py`
- **最小命令行示例**（在 `lxplus-gpu` 交互节点）：
  ```bash
  poetry run python -m analysis.flavour.v3.common.domains \
    --train-manifest /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R0/partitions/train300k.parquet \
    --tune-manifest /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R0/partitions/tune40k.parquet \
    --output /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v3_2026-11-15/W3-01
  ```
- **资源**：纯 CPU，`2–4 cores`，`< 0.5 h`
- **验收门禁**：① 单元测试通过；② 输出 `4×4` 可观测域-真值域混淆矩阵，`D_pix_contained` 覆盖真值 `pixel_centered` $\ge 80\%$ 且纯度 $\ge 75\%$；③ 固化 `Tune_E_le_2p3TeV` 事件键清单。

#### 工单 `W3-02`：训练期事件级与像素簇级 Truth Sidecar 生成
- **前置依赖**：`W3-01`
- **实现文件**：`src/analysis/flavour/v3/common/truth_sidecar.py`、`tests/test_v3_truth_isolation.py`
- **操作动作**：优先读取现有 parquet/ROOT，检查是否已含 `hit_fromCharmedHadron`、`fromTau` 与逐像素 `trackID`；按 [02 第 4 节](02_problem_formulation_and_domains.md#4-训练期密集真值侧车truth-sidecar生成规范srcanalysisflavourv3commontruth_sidecarpy) 与 `pixel_tokenizer` 完全同步的连通域法则为 `Train300k` 一次性生成 `event_sidecar.npy` 与 ragged `cluster_sidecar/`。
- **资源**：HTCondor CPU 并发（按分片 8–16 任务，每任务 4 CPU），约 `1–2 CPU hours` 等效墙钟 `< 20 min`
- **验收门禁**：① `test_v3_truth_isolation.py` 静态 AST 检查 + 运行时检查通过（确认 `V3EvalDataset` 全路径不加载 `sidecar_root`）；② 输出各分支标签实际有效覆盖率表（含 `label_7` 已知率、`charm_valid` 率、像素簇 5 类起源分布统计）。

#### 工单 `W3-03`：既有模型分区域校准零训练回放（立即验证假设）
- **前置依赖**：`W3-01`
- **实现文件**：`src/analysis/flavour/v3/common/calibration.py`
- **操作动作**：读取 `v2` 仓库已存的 `R2 B1-D2 (Tune40k)` 与 `S6 MAE3 (validation)` 原始输出 logits/概率，按照 `D_pix_contained / D_scint_contained / D_partial_or_edge` 运行 5-fold family 交叉拟合温度校准 + $S_\tau$ 对数似然比评分。
- **资源**：单核 CPU，`< 10 min`
- **验收门禁**：产出直接前后对比表（全局单阈值 vs 分区域校准 $S_\tau$ 在 `Pixel-centered` 的 `1% FPR` $\tau$ 效率），用零训练成本在先确认分区域校准能挽救大量 `S6 (0.35%)` 的损失。

---

### Phase V1：`B1++` 传感器原生强基线

#### 工单 `W3-04`：`B1++` 实现与小样本健康检查
- **前置依赖**：`W3-01`
- **实现文件**：`src/analysis/flavour/v3/v1_b1pp/representation.py`、`spectrometer.py`、`models.py`、`train.py`、`tests/test_v3_b1pp.py`
- **核心核验**：编写单测验证正交闪烁体面板填充无零空洞、磁场偏转曲率/矢高严格施加在 $y$-$z$ 平面（验证正负 $s_y$ 镜像对应预期符号反转）以及物理 $x \to -x$、$y \to -y$ 镜像增强保持 `T4/7-way` 标签语义精准。
- **资源**：交互 GPU，`0.3 device-hour`
- **验收门禁**：`32-event` 真实拟合在 100 updates 内 NLL `< 0.05`、准确率 `100%`；所有横向编码、纵向混合、谱仪 `TrackerTrackAttention` 分支梯度全部非零且吞吐实测报告（events/s, 显存峰值）。

#### 工单 `W3-05`：`B1++` 正式训练与合作组 M1 口径对齐
- **前置依赖**：`W3-02`, `W3-04`
- **执行矩阵（均 `seed=42`）**：
  1. `B1pp_100k_16p_T3`（与合作组 `T3` 同任务同非 $\tau$ 口径，对齐 `Wu (2026)` 报告表）；
  2. `B1pp_100k_16p_T4`（四类基线）；
  3. `B1pp_300k_24p_T4`（全量基准池收敛版，全程维护 `EMA 0.999`）。
- **资源**：Kubeflow 或 HTCondor GPU（H100 MIG / A100），预计合计 `~6–8 device-hours`（未测量）
- **验收门禁**：① `large_scintillator` 域 `BA3` 从 `R1 B1 (68.89%)` 提升至 $\ge 73\%$；② `Tune_E_le_2p3TeV` 上与全 `Tune` 一致上升；③ 检查 `Train/Tune NLL gap` 受控，不重现 `D1` 崩坏；④ **同步交付与合作组 [FORTVNE_10212026.pdf](../v2/08_r1_collaboration.md) 的同指标比较表**。

---

### Phase V2：`PinpointFormer` 混合架构突破

#### 工单 `W3-06`：`PinpointFormer` 编码器组装与 32 事例健康过拟合门禁
- **前置依赖**：`W3-01`, `W3-02`
- **实现文件**：`src/analysis/flavour/v3/v2_pinpointformer/pixel_tokenizer.py`、`stereo_bars.py`、`vertex_anchor.py`、`attention.py`、`models.py`、`tests/test_v3_pinpointformer.py`
- **核验重点**：确认连通域聚类、`32×32` 微补丁索引与 `ClusterSidecar` 一一对应（`cluster_id` 顺序逐个核对 0 误差）；检查 `ScaledQKNormAttention` 的梯度在高/低占据度样本上均稳定反传；确认取消 `linear_readout` 旁路后深层注意力分支在早期即获得主导梯度占比。
- **资源**：交互 GPU，`~0.5 device-hour`
- **验收门禁**：`32-event` 真实过拟合 100%；记录 `Train10k` 前向/反传吞吐（确保 `1024` 簇上限下显存 `< 16 GB` 可跑于 T4/1g.12gb 或更高卡型）。

#### 工单 `W3-07`：顶点锚定极坐标单因素消融 + 两大必检物理探针
- **前置依赖**：`W3-06`
- **对比实验（同 `Train100k × 16 passes`, `seed=42`, 基础 `T4 + 轻型顶点锚损失`）**：
  - `PF_V2_ablation_no_anchor`：仅使用全局绝对坐标 + `ScaledQKNormAttention` + `StereoBarNexus`（无 Stage-2 顶点重中心化与冲击参数对偏置）；
  - `PF_V2_anchored`：完整开启两阶段可微主顶点极坐标重中心化 + 近顶点 7 维 `IP_⊥/Δθ_kink` 对偏置。
- **资源**：预计 `~8–10 device-hours`（未测量）
- **必检探针与晋级门禁**：
  1. **对比验证顶点锚定净收益**：`PF_V2_anchored` 在 `Pixel-centered` 上对 `no_anchor` 及 `B0 HGB (55.92%)` 取得显著 `BA4` 净提高；
  2. 运行**探针 A（高占据度 $\tau$ 效率与注意力熵）**：确认 `1000–10000` 击中区间 $\tau$ 效率断崖消失；
  3. 运行**探针 B（微米 vs `1 mm` 人工量化消融）**：确认真实微米簇矩显著超越 `1 mm` 量化版（$\Delta\text{BA4} \ge +2.0\,\text{pp}$）；
  - *失败回退预案*：若探针 B 未通过（微米与 `1 mm` 差距小于 `0.5 pp`），**暂停往下进入 `V3`**，立即核查微补丁感受野、`VertexAnchorPredictor` 精度（以 mm 计的 $\hat{z}_v$ 误差是否过多大于 $\pm 20\,\text{mm}$ 导致极坐标角度污染）或连接半径 `0.06 mm` 合并颗粒度，定位后再往下推进。

---

### Phase V3：`GN2 + FASERCal` 密集多任务全监督

#### 工单 `W3-08`：事件级 4 头同方差多任务联合监督验证（`Train100k × 16 passes`）
- **前置依赖**：`W3-07`
- **实现文件**：`src/analysis/flavour/v3/v3_multitask/losses.py`、`train.py`
- **试验对比**：在 `PF_V2_anchored` 上开启 `7-way + Charm + 3D Vertex + Cylindrical Kinematics` 同方差联合训练，检验分拆 $\tau_\mu / \tau_{\text{had}} / \tau_e$ 与 Charm 显式头的净收益（特别观察高分尾部误判进入 $\tau$ 的 Charm 本底比例是否下降）。
- **资源**：预计 `~6 device-hours`（未测量）

#### 工单 `W3-09`：全密集 `GN2`（簇起源 + 同径迹亲和 + 全探测器 `Rel`）扩展至 `Train300k × 24 passes`
- **前置依赖**：`W3-08`
- **实现文件**：`src/analysis/flavour/v3/v3_multitask/dense_heads.py`
- **操作动作**：完整开启像素簇 5 类物理起源头、同径迹/同次级顶点亲和头、像素与条双端 `Rel` 以及推理期“自预测拓扑摘要回流”，在 `Train300k × 24 passes` 训练最终监督旗舰 **`PinpointFormer-GN2`**（配 EMA）。
- **资源**：预计 `~10–12 device-hours`（未测量）
- **验收门禁**：① 对比 `W3-08`，`Pixel-centered` 的 `1% FPR` $\tau$ 效率进一步明显提升，并产出 `Cluster Origin` 验证混淆矩阵/PR；② `Pixel-centered` 域 `BA4` 相对 `B0 HGB (55.92%)` 净增益 $\ge +4\text{–}6\,\text{pp}$；③ 同步复跑探针 A 与探针 B。

---

### Phase V4 & V5：双机制整合、低 FPR 尾部精调与独立确认

#### 工单 `W3-10`：闪烁体区 NOMAD 运动学融合、跨架构集成与 Partial-AUC 尾部微调
- **前置依赖**：`W3-05`, `W3-09`
- **实现文件**：`src/analysis/flavour/v3/v4_tau/nomad_kinematics.py`、`pauc_finetune.py`、`ensemble.py`
- **操作动作**：
  1. 在 `B1++` 与 `PinpointFormer-GN2` 上均验证加入 10 维无真值 NOMAD 横向运动学特征（作用于 `D_scint_contained`）的增量；
  2. 测试 DRO/CVaR `One-Way Partial AUC (@1% FPR)` 尾部微调 `5 passes`，检验 `1% / 0.1% FPR` 下的 $\tau$ 效率与三类非劣约束；
  3. 构造双架构互补集成：`PinpointFormer-GN2`（主攻微米粒子簇拓扑）+ `B1++`（主攻深层全纵向量能剖面与谱仪强特征），在 `Tune40k` 上使用 5-fold 分域校准验证集成增益。
  4. **条件分支门禁检查**：对照 [05 §3.3](05_multitask_gn2_and_tau_discovery.md#33-严格设限的条件自监督预训练门禁v45默认暂缓) 判断是否触发 `V4.5` 长时 SSL；如未完全满足，正式跳过直接进入 `W3-11`。
- **资源**：预计 `~5–6 device-hours`（未测量）

#### 工单 `W3-11`：优胜架构三随机种子稳定性验证（`seeds = {17, 42, 2026}`）
- **前置依赖**：`W3-10`
- **操作动作**：仅为最终脱颖而出的最多两个架构家族（如 `PinpointFormer-GN2` 与 `B1++`）各自补齐 `seed = 17` 与 `seed = 2026` 训练（与 `seed = 42` 组成三随机种子）。检验不同随机数下跨区域表现波动、并完成 3-seed 平均预测保存。
- **资源**：预计 `~12–16 device-hours`（未测量）

#### 工单 `W3-12`：Phase V5 冻结候选、解封 `Cal-A / Cut-B / Confirm200k` 终审交付
- **前置依赖**：`W3-11` 通过最终评审、所有超参/架构脚本彻底冻结写入 Git commit。
- **实现文件**：`src/analysis/flavour/v3/v5_confirm/run_confirmation.py`、`significance.py`、`plots.py`
- **严格执行次序（不可逆单次解封）**：
  1. **解封 `Cal-A40k`**：拟合并冻结每个可观测域 $d \in \{1, 2, 3\}$ 的校准温度 $(T_d, b_d)$；
  2. **解封 `Cut-B160k`**：锁定两种物理工作点规则——固定加权全域 `1% / 0.1% FPR` 阈值及多域最优对数似然比 $S_\tau$ 分域切分，确定分箱 Profile-Likelihood 模板边界；
  3. **最后一次性解封 `Confirm200k` (`50k × 4`)**：零重训、零重选阈值地单次跑完前向与完整物理统计评估（含 `6,845+` 物理块 bootstrap 95% 区间、`E_ν ≤ 2.3 TeV` 同能谱切片、14 组探测器读出/系统压力对照组、全英文 PNG+PDF 出版图表），计算最终多域 Asimov 显著性 $Z_0$ 与各通道迁移矩阵，撰写 Stage 总结阶段报告并提交 Git。
- **资源**：预计 `~3 GPU/CPU device-hours`（未测量）

---

## 4. CERN 计算调度规范（HTCondor & Kubeflow）

延续 `AGENTS.md` 与 [07 计算资源](../v2/07_resources_and_operations.md) 验证经验：
1. **Schedd 兼容性巡检**：每次 Condor 提交前执行 `condor_status -schedd` 并筛选 `WRITE` 与 EOS 路径支持节点（`stage04` 实测 `bigbird21` 普通队列不支持直写 EOS 需剔除，历史常用稳定节点如 `bigbird20` 视实时负载选择最空闲节点，提交日志须记录选取依据）。
2. **异构 GPU 务实策略**：T4 使用原生 FP32；A100-PCIE-40GB / H100 MIG (`1g.12gb` / `3g.47gb` / `h100-94gb`) 使用 `BF16 encoder + FP32 head & attention logits`；严禁使用过时兼容失败的 V100 分区。
3. **存储体积控制**：EOS 产物目录仅保存最佳/末轮权重、紧凑评估 JSON、英文图表和必需的 `truth_sidecar` 压缩整型索引；单卡本地高速磁盘（`/tmp` 或 Condor scratch）处理高频随机读，不向 `$HOME/condor/` 写入大数据日志。
