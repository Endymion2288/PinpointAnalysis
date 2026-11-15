# Stage307：ML 架构/物理审查与 v3 研究路线图规划

**审查结论：PINPOINT 四类中微子味道分类在现有模拟数据下完全可进一步大幅提升；`S0–S6` 与 `R0–R3` 性能停滞的根本原因已被定位为 `All` 域非包容混杂诱发的全局阈值倒挂、以及 5 处底层几何/注意力算子缺陷和欠拟合，而非探测器硬件分辨率不足。现已冻结全部审查取证并完成可执行、可直接照单开发的 [roadmap v3](../roadmap/v3/README.md)（阶段 `V0–V5`、工单 `W3-01`–`W3-12`），旧总入口原样归档至 [2026-11-15_v2_entry](../roadmap/archive/2026-11-15_v2_entry.md)。**

## 阶段目标

结合前序 306 个工作阶段实测数据、相关源码实现、合作组本地技术报告（`FORTVNE_10212026.pdf`、`FORTVNE_pixel_vs_scintonly_meeting_260916.pdf`、`FORTVNE_Update_kinematic_variables_2026-10-07.pdf`）、FASERCal 论文源码（`arXiv:2604.07037v2`）以及高能物理/计算机科学前沿文献，完成三项目标：
1. 全面裁决现有路线的科学正确性与代码实现缺口；
2. 确定下一阶段（v3）真正具备高信噪比增益的研究优先级；
3. 编写分步落地、涵盖数学模型与工程命令的 `roadmap/v3/` 文档集并保留原始 `roadmap/v2/` 资产零变动。

## 工作内容

### 1. 代码与实测数据深度审计（只读核查）
- **核查源码模块**：
  - `B2/B3` 与表征：[s2/models.py](../src/analysis/flavour/s2/models.py)、[s2/representation.py](../src/analysis/flavour/s2/representation.py)、[s2/prepare.py](../src/analysis/flavour/s2/prepare.py)；
  - `B4/S4/S6` 分层与预训练：[s3/dataset.py](../src/analysis/flavour/s3/dataset.py)、[s3/models_repair.py](../src/analysis/flavour/s3/models_repair.py)、[s4/models.py](../src/analysis/flavour/s4/models.py)、[s4/pretrain.py](../src/analysis/flavour/s4/pretrain.py)；
  - `S5` 对象与真值侧车：[s5/objects.py](../src/analysis/flavour/s5/objects.py)、[s5/truth.py](../src/analysis/flavour/s5/truth.py)；
  - `R1–R3` 模型与前端：[v2/r1/data.py](../src/analysis/flavour/v2/r1/data.py)、[v2/r1/models.py](../src/analysis/flavour/v2/r1/models.py)、[v2/r3/relations.py](../src/analysis/flavour/v2/r3/relations.py)、[v2/r3/sensor_rows.py](../src/analysis/flavour/v2/r3/sensor_rows.py)、[readout.py](../src/analysis/flavour/readout.py)、[utils/geometry.py](../src/analysis/utils/geometry.py)、[utils/preprocessing_utils.py](../src/analysis/utils/preprocessing_utils.py)。
- **核查数据与结果文档**：`R1/R2/R3` 机器汇总（[r1_results.json](../roadmap/v2/r1_results.json)、[r2_results.json](../roadmap/v2/r2_results.json)、[r3_results.json](../roadmap/v2/r3_results.json)）及阶段报告（[stage04](2026-11-11_stage04_flavour_inventory.md)、[stage170](2026-11-14_stage170_s4_valid_path_results.md)、[stage236](2026-11-14_stage236_s6_final_results.md)、[stage274](2026-11-15_stage274_r1_final_results.md)、[stage284](2026-11-15_stage284_r2_final_results.md)、[stage306](2026-11-15_stage306_r3_final_results.md)）。
- **文献与模拟参考对照**：对照外部同目录下本地 FASERCal LaTeX 源码、`Pinpoint_G4` 几何说明及三份 FORTVNE 合作组报告，并结合联网文献核实 ATLAS `GN2` (`arXiv:2505.19689`)、`NuGraph2` (`arXiv:2403.11872`)、`SPINE` (*PRD* 2024)、`QK-Norm` (Henry et al., EMNLP 2020)、Softmax 长度弥散理论 (Veličković et al., ICML 2025)、NOMAD 运动学 (*PLB* 2001) 与 DRO One-Way Partial AUC (Zhu et al., ICML 2022)。

### 2. 仓库文档编制与版本归档
- **归档快照**：完整保存原 `roadmap/README.md` 为 [2026-11-15_v2_entry.md](../roadmap/archive/2026-11-15_v2_entry.md)，更新 [归档索引](../roadmap/archive/README.md)；**严格保持 `roadmap/v2/` 全部文件只读未改**。
- **新建 `roadmap/v3/` 全套 8 个文档**：
  - [v3 总览](../roadmap/v3/README.md)
  - [01 根因取证与文献综述](../roadmap/v3/01_forensic_diagnosis_and_literature.md)
  - [02 问题重构、可观测域与侧车 (V0)](../roadmap/v3/02_problem_formulation_and_domains.md)
  - [03 B1++ 强基线规范 (V1)](../roadmap/v3/03_b1pp_strong_baseline.md)
  - [04 PinpointFormer 架构规范 (V2)](../roadmap/v3/04_pinpointformer_architecture.md)
  - [05 多任务 GN2 与 ντ 发现 (V3–V4)](../roadmap/v3/05_multitask_gn2_and_tau_discovery.md)
  - [06 工单、探针与工程纪律 (`W3-01`–`W3-12`)](../roadmap/v3/06_work_orders_probes_and_operations.md)
  - [experiment_plan_v3.json](../roadmap/v3/experiment_plan_v3.json)
- **更新顶层入口**：重写 [roadmap/README.md](../roadmap/README.md) 为 v3 总入口。

## 详细数据

### 1. 本次定量核对与取证核心数值（摘自已测冻结档案）

> 当前本地终端无测试环境与训练数据。本阶段 **新增模型训练 = 0，GPU/CPU 提交作业 = 0**；新设计模块 `V0–V5` 的训练精度、运行吞吐及计算耗时全部处于 **未测量** 状态。下表所列数值均为对前序已存储结果的精确抽取与数学复算：

| 检查项 | 来源 / 代码位置 | 实测 / 核对结果 |
| --- | --- | --- |
| `Pixel-centered` 细粒度失效 | [r1_results.json](../roadmap/v2/r1_results.json) | `B0 HGB`: `BA4 = 55.92%`, `τ@1% = 4.36%`；`B2/B3/B4/B4-MAE`: `BA4 = 55.75% / 55.09% / 55.08% / 55.62%`，`τ@1% = 2.51% / 1.59% / 1.72% / 2.38%`（无空间坐标树模型反胜全部深层点云/Transformer） |
| `All` 域非包容子集稀释 | [r1_results.json](../roadmap/v2/r1_results.json) (`final_b1_t3_s42`) | `large_scintillator (13,866/30,000 = 46.22%)`: `BA3 = 68.89%`（`[74.71%, 73.87%, 58.10%]`）；由全集 `[63.01%, 49.98%, 56.87%]` 精确减除得余下 **`53.78%` 非包容区** 召回 `[53.73%, 28.25%, 55.79%]`，`BA3 = 45.92%` |
| `S6` 全局单阈值效率反转 | [stage236](2026-11-14_stage236_s6_final_results.md) | `Target envelope` $\tau$ 效率 `6.13%`，`Pixel-centered` (`BA4 = 56.19%`) 跌 17 倍至 **`0.35% (17/4,882)`**；像素击中 `1–100` vs `1000–10000` 效率从 `6.25%` 跌 21 倍至 **`0.29%`** |
| `B4` 注意力 logit 上界 | [models_repair.py](../src/analysis/flavour/s3/models_repair.py#L49) | 无学习尺度 `LayerNorm` Q/K（`width=24`）将点积锁定在 $[-\sqrt{24}, +\sqrt{24}] = [-4.90, +4.90]$；$N=1000$ token 下单个关键信号权重上限 $\le 11.8\%$ |
| `B4` 捷径依赖与空上下文 | [models_repair.py](../src/analysis/flavour/s3/models_repair.py#L126,L198,L250)、[stage170](2026-11-14_stage170_s4_valid_path_results.md) | `multiscale=True` 时 `context` 长度为 `0`、区域位置仅喂入 `geo[:, 2:3]` 丢失 $(x,y)$；10k Scratch `Global latent` 置换 $\Delta\text{NLL} = 0.000085$ vs `Response summary` `0.226352` |
| `B2/B3` 度量畸变与零距离 | [s2/models.py](../src/analysis/flavour/s2/models.py#L48-L56) | `[210, 210, 3000]` 归一化叠加 `0.05*dz^2` 使 `1 mm` 横向距等效 **`63.9 mm`** 纵向距；正交 X/Y 条 `common = [1,0]*[0,1] = [0,0]` 使所有跨面板距离为 `0` |
| $\nu_\tau$ 样本能谱边界 | [stage04](2026-11-11_stage04_flavour_inventory.md) | $\nu_\tau$ (`10083`) $E_\nu \le 2.3118\,\text{TeV}$，其余样本达 $5.0412\,\text{TeV}$ |

### 2. 静态交付验收记录
- JSON 语法与结构校验：`roadmap/v3/experiment_plan_v3.json` 解析通过（`schema_version=3`、`6` 个阶段 `V0–V5`、`12` 张工单 `W3-01`–`W3-12` 及 `2` 项必检物理探针定义完整）；
- 链接与公式闭合核对：覆盖 `11` 个新建/修改 Markdown 文件（含 `roadmap/v3/`、`roadmap/README.md`、`roadmap/archive/` 与本报告）共 `161` 处本地相对链接与锚点，路径、代码行号区间及 `$`/`$$` 公式闭合检查均为 **`0` 错误**；
- 冻结区与空白符检查：`git status --short` 确认 `roadmap/v2/` 目录 0 改动，`git diff --check` 退出码 `0`（无尾随空格等告警）。

## 阶段性结论

### 已确立的事实
1. 现有 `All` 全域评价把 `53.8%` 发生在强截断边缘区的低信息事例与高分辨率包容区混在同一校准分布和单一阈值下，直接导致 `S6` 黄金区 $\tau$ 效率跌至 `0.35%`。
2. 现有深度主干在算子层面分别存在无温度 `QK-Norm` 弥散、`63.9:1` 尺度塌缩与正交条零距离、`1 mm` 网格 `90%` 空洞与过早 `GlobalPool`、`summary` 线性捷径饥饿等明确程序/架构硬伤，叠加 `4 passes` 欠训练与磁谱仪物理特征缺失，致使细粒度模型均未能超越 `B0` 树模型。
3. 硬规则追踪 `s5/objects.py` 与 `R3 G2` 候选关联在 $1.43\,X_0/\text{层}$ 钨靶簇射中纯度与召回严重不足，继续沿 `v2 R4` 将启发式重建作为分类前提无法按期突破。

### 待检验科学假设（由 `V0–V4` 工单逐项证伪或确认）
1. **分区域校准假设（`W3-03`）**：无需重训，仅对既有 `R2 D2` / `S6` 原始概率实施按可观测包容域 $c_{\text{obs}}$ 的分域温度校准 + 对数似然比评分 $S_\tau$，即可显著消除 `Pixel-centered` 全局阈值倒挂。
2. **`B1++` 强基线假设（`W3-04`–`W3-05`）**：使用零空洞紧凑条/像素矩阵、保留纵向 $z$ 序列融合、修正至 $y$-$z$ 偏转面的磁谱仪矢高与反向外推特征、并配合精确镜像增强充分训练，可在 `large_scintillator` 取得明显增益并建立稳固三类交付与合作组可比基准。
3. **`PinpointFormer-GN2` 微米拓扑解锁假设（`W3-06`–`W3-09`）**：全保留 16 维像素簇形态矩 + 正交条 `NuGraph2` 式立体交叉 + 两阶段可微主顶点极坐标重中心化 + 可学习温度 `ScaledQKNormAttention`，结合 `FASERCal` 同方差多任务与 `GN2` 簇起源/亲和密集监督，能够使微米级像素真正在 `Pixel-centered` 上大幅超越 `1 mm` 量化基线并攻克高占据度塌陷与 Charm 误报。

## 下一步计划

回到 `lxplus-gpu` 存在数据与 LCG/Poetry 环境后，按以下精确顺序立即启动执行（详见 [v3 执行工单](../roadmap/v3/06_work_orders_probes_and_operations.md)）：
1. **CPU 首日三步（`W3-01` → `W3-02` → `W3-03`）**：先生成 `c_obs` 分域统计与 $E_\nu \le 2.30\,\text{TeV}$ 对齐切片清单；同步导出零泄漏 `truth_sidecar` 并通过静态/运行时隔离单测；同时立即在已存的 `R2 B1-D2` 和 `S6` 预测文件上回放分区域校准，实证消除 `0.35%` 倒挂。
2. **快速基线突破（`W3-04` → `W3-05`）**：实现并验收 `B1++`（先过 32-event 100% 小样本与 $y$-$z$ 矢高符号单测），在 `Train300k` 取得超越 `R1 B1 (68.89%)` 与对齐合作组的新强基线。
3. **混合架构与密集监督推进（`W3-06` → `W3-10`）**：逐步完成 `PinpointFormer`、高占据度与微米量化两项必检探针、`GN2` 簇起源与同径迹亲和损失，以及闪烁体 NOMAD 运动学 + DRO Partial-AUC 尾部微调。
4. **独立终审（`W3-11` → `W3-12`）**：确认优胜组合扩三随机种子并按不可逆顺序一次性解封 `Cal-A40k → Cut-B160k → Confirm200k` 输出独立确认与发现显著性结果。
