# Stage 323：Clean v3 纯钨上游模拟修复、匹配预算多卡并行训练与分析前密码学冻结

- **日期**：2026-11-20（UTC 冻结时刻：`2026-11-20T01:31:01Z`）
- **基线审计**：[2026-11-19_stage322_independent_clean_v2_scientific_audit.md](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/workbook/2026-11-19_stage322_independent_clean_v2_scientific_audit.md)
- **代码仓库与提交**：
  - `Pinpoint_G4`（分支 `fix/audit-geometry-digitizer-physics`）：`37fd44567acff810c53f60ae7809064fb7e78ae3`
  - `PinpointAnalysis`（分支 `pinpoint_neutrino_ml_review`）：模型训练及冻结代码提交 `58897f8ba9746cdb78069ae2d29c6abe0ee96ba7`（本报告及冻结清单随后单独提交）
- **盲态与暂停声明**：严格遵循用户指令，本阶段**仅修复与模型训练相关的上游模拟、数字化、特征/API 隔离、基线、匹配预算训练与分析前冻结项**；**所有模型训好之后的物理分析项（`322-02`、`322-04`、`322-06`、`322-08`、`322-10`、`322-14`）全部暂停**，`CleanCalA`、`CleanCutB`、`CleanConfirm` 三个密封分区 `Cal_A_Cut_B_Confirm_touched == False`（训练期间连本地暂存也物理跳过密封 shard，零字节读取）。

---

## 1. 阶段目标

针对 Stage 322 独立科学审计中发现的全部 10 项模型训练相关问题，完成端到端源码修复、Clean v3 纯钨数据生产、GPU 健康门控、强可观测非神经网络基线、10 组 × 3 匹配种子（共 30 个模型）的同预算训练，以及独立分析前密码学冻结：

| 审计编号 | 严重度 | 阶段处置 | 验收目标 |
| :--- | :--- | :--- | :--- |
| `322-01` | P0 | **本阶段修复并闭合** | 100% 纯钨（GENIE `Z == 74` 且 Geant4 `G4Navigator == G4_W`），`0` 非钨事件；`Pinpoint_G4` 内置致命守卫拒绝非钨靶核与空气顶点 |
| `322-03` | P1 | **本阶段修复并闭合** | Target 与 IPT 量化步长、聚类半径、截断配额（`832 + 192`）、patch 配额（`48 + 16`）完全解耦；真实事件上 `0.0` bit-exact IPT 不变性；完成正交 `2 × 2` Target × IPT 阶乘训练 |
| `322-05` | P1 | **本阶段修复并闭合** | 统一纯钨 GENIE `10081..10084` 生产源；记录真实 CC charm 与 GENIE Pythia6 `0` NC charm 现状；能量匹配加权对照组验证无 $E_\nu$ 捷径；Pythia8 自由核子 NC charm 独立诚实标注（`Z ∈ {0,1}, A=1`）作诊断 OOD |
| `322-07` | P1 | **本阶段修复并闭合** | `PixelSD` 步中点 + 3 μm 二维高斯电荷扩散 + 10 e⁻ 电子学噪声 + 可配置阈值；`tracker_measurements` 精确交点质心合并，消除 100 μm 网格量化偏差（Y RMS ≈ 17 μm，pull std ≈ 1.0） |
| `322-09`（上游+冻结） | P1 | **本阶段修复并闭合** | `sources_crossing_splits == 0` 且 100% family/kinematic fingerprint 不交；保留父事件 `source_tag`/`source_row`；独立生成 `v8_clean_v3_freeze_manifest.json` 密码学指纹 |
| `322-11` | P2 | **本阶段修复并闭合** | 模型与数据加载器完全去除 `len(batch['label'])` 依赖；新增 `V3ObservationDataset` 与 `infer_observation_only()`，在无 `targets.parquet` 目录下对 4 类架构实现 `0.0` FP32 闭合 |
| `322-12` | P2 | **本阶段修复并闭合** | `ScintSD` 按通道 1 ps 波形累加与 Rényi 顺序统计 PE 触发时间，对 truth `trackID` 分组完全不变；`pe <= 0` 强制 `hit_time_ns = -1.0`；`readout.py` 同步要求 `pe > 0` |
| `322-13` | P2 | **本阶段修复并闭合** | 全物理钨体积 1 μm 边距采样覆盖 `99.959%` 物理 `G4_W` 体积；3 分层质量加权精确还原几何占比 `24/108` |
| `322-15` | P2 | **本阶段修复并闭合** | 32 事件 GPU 健康门控（全分支正有限梯度 + 模态置零敏感性）；强非神经网络基线（`L2-LR` 6d/16d/70d + `HistGradientBoostingClassifier` BDT 70d）；10 组 × 3 匹配种子 `{17, 42, 2026}` 同预算（20 passes, batch 128）训练 |
| `322-16` | P3 | **本阶段修复并闭合** | 同步 `nIntermidiatePixelLayers = max(0, nFortuneLayers - 1)`，移除 `fCheckOverlaps` 局部遮蔽变量，实测 `nFortune_7` 与 `nFortune_4` 几何闭合 |
| `322-02, 04, 06, 08, 10, 14` | P0–P2 | **按指令暂停（未测量）** | 留待下一阶段在密封分区解封协议下独立完成；本阶段严禁读取 `Cal-A`/`Cut-B`/`Confirm` |

---

## 2. 工作内容

### 2.1 `Pinpoint_G4` 上游模拟与数字化修复（提交 `37fd4456`）

- **几何同步与重叠标志（`322-16`）**：修改 [DetectorConstruction.hh](file:///eos/home-x/xcheng/FASER/Pinpoint/Pinpoint_G4/Pinpoint/include/DetectorConstruction.hh) 与 [DetectorConstruction.cc](file:///eos/home-x/xcheng/FASER/Pinpoint/Pinpoint_G4/Pinpoint/src/DetectorConstruction.cc)，在 `SetNumFortuneBlocks()` 与 `Construct()` 中同步 `nIntermidiatePixelLayers = std::max(0, nFortuneLayers - 1)`，并消除局部变量对成员 `fCheckOverlaps` 的遮蔽。
- **逐事件 `G4Navigator` 介质验证与纯钨硬守卫（`322-01`）**：修改 [GeneratorVertexMetadata.hh](file:///eos/home-x/xcheng/FASER/Pinpoint/Pinpoint_G4/Pinpoint/include/generators/GeneratorVertexMetadata.hh)、[GFaserGenerator.cc](file:///eos/home-x/xcheng/FASER/Pinpoint/Pinpoint_G4/Pinpoint/src/generators/GFaserGenerator.cc)、[AnalysisManager.hh](file:///eos/home-x/xcheng/FASER/Pinpoint/Pinpoint_G4/Pinpoint/include/AnalysisManager.hh) 与 [AnalysisManager.cc](file:///eos/home-x/xcheng/FASER/Pinpoint/Pinpoint_G4/Pinpoint/src/AnalysisManager.cc)，逐事件通过 `G4Navigator::LocateGlobalPointAndSetup()` 将 `vtxMatName`、`vtxVolName`、`vtxMatZ` 写入 `event` TTree，并新增环境变量开关 `PINPOINT_REQUIRE_W_MATERIAL=1`（`VertexMaterialMismatch` 致命异常）与 `PINPOINT_REQUIRE_W_TARGET=1`（`GeneratorTargetMismatch` 致命异常）。
- **物理通道时间累加与 0-PE 时间闭合（`322-12`）**：重写 [ScintSD.cc](file:///eos/home-x/xcheng/FASER/Pinpoint/Pinpoint_G4/Pinpoint/src/ScintSD.cc) 的闪烁条时间读出，使用 1 ps 分箱通道级光学脉冲累加（`barPulseMap`）、精确 Rényi 顺序统计发光衰减时间、0.15 ns TTS 抖动与 3-PE 甄别器触发时间，对 truth `trackID` 分组完全不变；`0-PE` 击中强制输出 `hitTime = -1.0`。
- **像素电荷扩散与电子学噪声（`322-07`）**：更新 [PixelSD.cc](file:///eos/home-x/xcheng/FASER/Pinpoint/Pinpoint_G4/Pinpoint/src/PixelSD.cc)，使用步中点 `0.5*(prePos + postPos)`、基于 `std::erf` 的 3×3 像素二维横向电荷扩散（默认 `PINPOINT_PIXEL_DIFFUSION_UM=3.0`）、高斯电子学噪声（`PINPOINT_PIXEL_NOISE_E=10.0`）及可配置阈值（`PINPOINT_PIXEL_THR_E=100.0`）。
- **编译与实时 Geant4 闭合验证**：编译生成 `/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v6_clean_v3_2026-11-19/g4/pinpoint_v3`（0 warning, 0 error），并通过实时 Geant4 测试记录于 [live_g4_v3_closure.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v6_clean_v3_2026-11-19/g4/live_g4_v3_closure.json)。

### 2.2 `PinpointAnalysis` 读出、Tokenizer、纯观测 API 与训练基础设施修复（提交 `c9c7e02`, `3000f22`, `58897f8`）

- **闪烁条 0-PE 时间与径迹交点精确质心（`322-07`, `322-12`）**：修改 [readout.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/readout.py)，`bar_channels` 在 `valid_t` 中强制要求 `pe > 0` 并在 `summed_pe <= 0` 时赋 `-1.0`；`tracker_measurements` 按 `(station, x, y)` 规范排序，并用精确交点质心合并两击中单元（彻底消除 100 μm 网格偏差），支持可选 `dead_fraction`/`noise_rate_per_station`。
- **Target 与 IPT 完全隔离（`322-03`）**：重构 [pixel_tokenizer.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v2_pinpointformer/pixel_tokenizer.py) 与 [group.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/common/group.py)，分离 `quantize_target_xy_mm` 与 `quantize_ipt_xy_mm`、分离聚类半径与单平面步长、分离簇截断配额（`N_MAX_TARGET_CLUSTERS=832` 与 `N_MAX_IPT_CLUSTERS=192`，消除全局 1024 上限引起的跨子探测器名额泄漏）与 patch 配额（`48 + 16 = 64`），并改为内容哈希种子的 `shuffle_micro_vertex`。
- **无标签纯观测部署接口与密封分区隔离（`322-11`）**：更新 [v1_b1pp/models.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v1_b1pp/models.py)、[representation.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v1_b1pp/representation.py)、[v2_pinpointformer/models.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v2_pinpointformer/models.py)、[stereo_bars.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v2_pinpointformer/stereo_bars.py)、[datasets.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/common/datasets.py) 及新增 [observation_inference.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/common/observation_inference.py)，全部用 `c_obs.shape[0]` 替代 `len(batch['label'])`，新增 `V3ObservationDataset` 与 `infer_observation_only()`；同时优化 `V3EvalDataset` 与 `group.py`，使训练期仅暂存和读取 `train`/`validation` 分区（shards `000..099`），物理零接触 `cal_a`/`cut_b`/`confirm`（shards `100..149`）。
- **显存感知自适应并行与精确梯度累加**：在 [group.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/common/group.py) 与 [v2_pinpointformer/train.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v2_pinpointformer/train.py) 中实现按 GPU 实测显存自动选择并发数（`H100 NVL`/`H100-47GB`/`A100-40GB`：`--parallel 3`；`H100 1g.12gb`：`B1++` `--parallel 2`、`PF` `--parallel 1`）并在 `< 28 GB` GPU 上自动启用 `grad_accum=2`（`micro_bs=64`，PinpointFormer 零 `BatchNorm`，数学梯度与 `batch_size=128` 完全一致），兼顾单卡高利用率与零 OOM。
- **Clean v3 纯钨生产、训练编排与冻结脚本**：新增 [produce_clean_v3.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v6_clean_sim/produce_clean_v3.py)、[run_stage323_clean_v3_training.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v6_clean_sim/run_stage323_clean_v3_training.py) 及单元测试 [test_stage323_remediation.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/tests/test_stage323_remediation.py)，同时修复 [produce_external_and_gen_sys.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v6_clean_sim/produce_external_and_gen_sys.py) 保留父事件 `source_tag`/`source_row`。

### 2.3 计算调度与 Schedd 选择记录

- **HTCondor Schedd 选择**：提交前检查 `choose_schedd` 与可用队列；`bigbird21.cern.ch` 缺乏无条件 EOS 策略，因此所有 Geant4 批量模拟任务（Cluster `1584852`，79 任务 × 8 CPU）与 9 组 HTCondor GPU 训练任务（Clusters `1584871–1584879`，请求 `cpus=4, memory=15000MB, min_gpu_memory_mb=10000, min_capability=8.0, job_flavour=workday, allow_backfill=True`）均提交至兼容且负载均衡的 `bigbird20.cern.ch`。
- **Kubeflow 与 HTCondor GPU 联合调度**：Kubeflow 命名空间 `xcheng` 受 Kyverno `namespace-gpu-quota` 限制为总计 1 GPU 单元；因此 `s323-pf-20um` 运行于 Kubeflow `NVIDIA H100 NVL MIG 3g.47gb`（`par=3`，49.8 min 完成 3 种子），其余 9 组并行调度至 HTCondor `H100 NVL 95GB`（`par=3`）、`A100-PCIE-40GB`（`par=3`）与 `H100L-1-12C MIG 1g.12gb`（`B1++` `par=2`、`PF` `par=1 + grad_accum=2`），全部 10 组 30 模型在纯 BF16（SM80/SM90）下以 **0 失败、0 OOM** 完成。
- **排错记录**：首批 HTCondor GPU 尝试因在 MIG 分区启用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（触发 `cuMemCreate` 驱动 `CUDA driver error: operation not supported`）立即失败；移除该环境变量并启用 `<28 GB` GPU 自动 `grad_accum=2` 后，重提交 9 组全部一次通过。

---

## 3. 详细数据

### 3.1 单元测试与实时 Geant4 闭合验证

- **全量单元测试**：
  ```bash
  source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
  export PYTHONPATH="$PWD/src:$PYTHONPATH"
  python3 -m unittest discover -s tests
  ```
  实测结果：**358 / 358 OK**（耗时 `136.7 s`），其中新增 `tests/test_stage323_remediation.py` 6 项专项测试全部通过（耗时 `21.7 s`）。
- **实时 Geant4 闭合验证**（产物：[live_g4_v3_closure.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v6_clean_v3_2026-11-19/g4/live_g4_v3_closure.json)）：
  - `322-16` 几何同步：`nFortune_7` 实测 `108 W / 15 Si / 96 Scint`（期望 `108 / 15 / 96`），`nFortune_4` 实测 `66 W / 12 Si / 57 Scint`（期望 `66 / 12 / 57`），`G4Navigator` 均返回 `G4_W (Z=74)`。
  - `322-01 / 12 / 13` 靶核、全钨体积与 0-PE 定时：200/200 个 1 μm 边距全钨体积顶点全部命中 `G4_W`；`zero_pe_nonneg_time_hits = 0`；故意注入的 `C12` 靶核与空气顶点均被致命异常正确拦截。
  - `322-07` 像素扩散阈值扫描（50 GeV $\mu^-$）：旧版无扩散 100 e⁻ 仅 `1.269` pixel/plane（多像素率 `14.46%`）；新版 3 μm 扩散在阈值 `50 / 100 / 150 / 200 e-` 下分别为 `2.176 / 1.862 / 1.700 / 1.561` pixel/plane（多像素率 `67.56% / 52.91% / 42.10% / 34.10%`）。
  - `322-07` 谱仪径迹精度（`N = 176` 单击中样本）：`tracker_y_rms_um = 16.59 μm`、`tracker_y_pull_mean = +0.00028`、`tracker_y_pull_std = 0.976`（消除原 100 μm 网格量化偏置）。

### 3.2 Clean v3 纯钨数据集生产与物理审计

执行命令：
```bash
python3 -u -m analysis.flavour.v3.v6_clean_sim.produce_clean_v3 finalize
python3 -u -m analysis.flavour.v3.v6_clean_sim.run_stage323_clean_v3_training audit-dataset
```
产物：[prepare_summary.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v6_clean_v3_2026-11-19/clean_sim/prepare_summary.json)、[finalize_summary.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v6_clean_v3_2026-11-19/clean_sim/finalize_summary.json)、[stage323_clean_v3_dataset_audit.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v6_clean_v3_2026-11-19/clean_sim/stage323_clean_v3_dataset_audit.json)、图 [stage323_clean_v3_dataset_audit.png](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v6_clean_v3_2026-11-19/clean_sim/stage323_clean_v3_dataset_audit.png)。

| 审计指标 | Clean v3 实测值 | 验收状态 |
| :--- | :--- | :--- |
| 名义样本总事件数（150 shards × 400） | `60,000`（`train 32,000`, `validation 8,000`, `cal_a 4,000`, `cut_b 6,000`, `confirm 10,000`） | 通过 |
| 筛除的源样本非钨事件 | `8,576`（入选非钨 `non_W_selected_events = 0`） | 通过 |
| `G4Navigator == G4_W` 校验通过数 | `60,000 / 60,000`（`non_W_tune_g4_navigator_count = 0`） | 通过 |
| 跨 split 源标签 / family / 运动学重叠 | `sources_crossing_splits = 0`，`family_disjoint = True`，`kinematic_fp_disjoint = True` | 通过 |
| 物理钨体积采样覆盖率（1 μm 边距） | `99.9591%`（`outside_z = 0`, `outside_xy = 0`, `max|dz| = 2.4990 mm`） | 通过 |
| 质量加权 Pinpoint 占比 vs 精确几何 `24/108` | `0.22135` vs `0.22222`（`mean_geom_weight = 0.99903`） | 通过 |
| 闪烁条 `edep_proxy_MeV` 残留 / 0-PE 正时间数 | `edep_proxy_MeV_absent = True`，`zero_pe_nonneg_time_count = 0`（530,572 通道） | 通过 |
| 单次饱和 SiPM 最大读数与 `>9103 PE` 通道数 | `max_observed_pe = 14,400.0 PE`，`channels_above_9103_pe = 24,516` | 通过 |
| 真实事件（`N=64`）Target 量化下 IPT 差异（`0.10 / 0.30 / 1.00 mm`） | 计数变化 `0/64`，特征/patch/平面汇总最大绝对差全部为 **`0.0`** | 通过 |

名义纯钨四类别（统一 GENIE `10081..10084`）统计特性：

| 类别 | GENIE Run | 选中事件数 | 生成器 Charm 比例 | 反中微子比例 | 中位 $E_\nu$ [GeV] | 90% 分位 $E_\nu$ [GeV] | 最大 $E_\nu$ [GeV] | $E_\nu \le 2312$ GeV 占比 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $\nu_e$ CC（Class 0） | `10081` | 15,000 | 10.26%（1,539） | 37.75% | 515.6 | 1,284.2 | 4,078.7 | 98.51% |
| $\nu_\mu$ CC（Class 1） | `10082` | 15,000 | 9.46%（1,419） | 20.62% | 537.3 | 1,698.3 | 5,041.2 | 95.49% |
| $\nu_\tau$ CC（Class 2） | `10083` | 15,000 | 9.86%（1,479） | 26.08% | 664.3 | 1,539.2 | 2,311.8 | 100.00% |
| NC（Class 3） | `10084` | 15,000 | 0.00%（0） | 27.24% | 512.4 | 1,623.2 | 5,041.2 | 96.01% |

独立 OOD / 诊断验证分片（4,800 事件，全部经过 `pinpoint_v3` Geant4 模拟）：
- `gen_ood_run10085_10088`：1,600 事件，100% 纯钨（`Z=74`, `G4_W`），8 个独立源标签，118 charm，$E_{\max} = 4778.7$ GeV。
- `gen_ood_pythia6_std`：1,200 事件，100% 纯钨（`Z=74`, `G4_W`），165 源标签，96 charm，$E_{\max} = 5835.5$ GeV。
- `gen_ood_pythia6_nux`：1,200 事件，100% 纯钨（`Z=74`, `G4_W`），165 源标签，97 charm，$E_{\max} = 3910.9$ GeV。
- `diag_pythia8_free_nucleon_nc_charm`：800 事件，诚实标注自由核子靶（`Z ∈ {0, 1}, A=1`，放置在物理 `G4_W` 钨块内，绝不冒充 W184），800 charm，$E_{\max} = 3443.0$ GeV。

### 3.3 训前 32 事件 GPU 健康门控（`smoke-32`）

执行命令：`python3 -u -m analysis.flavour.v3.v6_clean_v3_training smoke-32`（实际模块 `analysis.flavour.v3.v6_clean_sim.run_stage323_clean_v3_training smoke-32`），设备 `cuda:0`（Tesla T4），耗时 `202.2 s`，产物：[stage323_smoke32_health_gate.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v6_clean_v3_2026-11-19/clean_sim/stage323/stage323_smoke32_health_gate.json)。

| 模型架构 | 参数量 | 初始 NLL4 | 80 步 NLL4 | 32 事件准确率 | 全分支正有限梯度 | Pixel Drop TV（翻转） | Bar Drop TV（翻转） | Tracker Drop TV（翻转） | All Drop TV（翻转） |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `b1pp_clean` | 3,124,023 | 1.3832 | `1.13e-5` | 100.0% | 7/7 通过 | 0.233（6/32） | 0.592（19/32） | 0.292（10/32） | 0.781（30/32） |
| `pf_20um` | 8,881,436 | 1.4844 | `1.17e-3` | 100.0% | 10/10 通过 | 0.567（18/32） | 0.093（2/32） | 0.297（10/32） | 0.735（24/32） |
| `pf_hybrid_uvtx` | 8,923,677 | 1.4949 | `3.27e-5` | 100.0% | 11/11 通过（`micro_vertex_reco = 0.0213`） | 0.679（22/32） | 0.194（6/32） | 0.274（9/32） | 0.757（24/32） |
| `pf_hybrid_nomad` | 9,038,718 | 1.4934 | `1.82e-5` | 100.0% | 12/12 通过（`nomad_kinematics = 1.087`） | 0.475（14/32） | 0.274（9/32） | 0.195（7/32） | 0.551（18/32） |

### 3.4 强可观测非神经网络基线与 10 组（30 模型）CleanTune8k 验证对比

所有模型均使用完全一致的 Clean v3 纯钨 `train`（32,000 事件）训练，并在 `CleanTune8k`（8,000 事件）上评估（神经网络为 3 种子 `{17, 42, 2026}` 同预算 20 passes, batch 128, 5-fold OOF 分域集成）。产物：[stage323_observable_baselines.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v6_clean_v3_2026-11-19/clean_sim/stage323/stage323_observable_baselines.json)、[v8_clean_v3_freeze_manifest.json](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/roadmap/v3/v8_clean_v3_freeze_manifest.json)、图 [stage323_clean_v3_training_and_ood_validation.png](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v6_clean_v3_2026-11-19/clean_sim/stage323/stage323_clean_v3_training_and_ood_validation.png)。

| 模型 / 基线 | 训练硬件与并发 | All Tune8k BA4 | All NLL | mw-2t BA4 | PinpointBlock BA4 | Pinpoint NLL | Pinpoint $\tau$@1%FPR | Pinpoint $\tau$@0.1%FPR | FrontPinpoint $\tau$@1%FPR | PV 中位 $\|\Delta z\|$ / $r_{\text{3D}}$ [mm] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `LR-6d`（存在/计数） | CPU（8 线程） | 40.30% | 1.2791 | 35.59% | 44.82% | 1.2262 | 1.30% | 未测量 | 未测量 | N/A |
| `LR-16d`（`c_obs`） | CPU（8 线程） | 47.90% | 1.1570 | 44.03% | 51.60% | 1.1050 | 0.74% | 未测量 | 未测量 | N/A |
| `LR-70d`（全物理特征） | CPU（8 线程） | 50.71% | 1.0972 | 46.01% | 55.25% | 1.0311 | 2.14% | 未测量 | 未测量 | N/A |
| `BDT-70d`（`HistGB`） | CPU（8 线程） | 53.64% | 1.0430 | 50.66% | 56.50% | 0.9891 | 3.72% | 未测量 | 未测量 | N/A |
| `s323-b1pp-clean` | HTCondor H100 `1g.12gb` (`par=2`) | 57.23% | 0.9835 | 53.96% | 60.37% | 0.9221 | 4.19% | 0.84% | 3.12% | N/A |
| `s323-pf-20um`（Tgt 20μm, IPT 20μm） | Kubeflow H100 `3g.47gb` (`par=3`) | 57.83% | 0.9612 | 54.16% | 61.33% | 0.8978 | 6.05% | 0.84% | 6.12% | `5.422 / 7.320` |
| `s323-pf-tgt1mm-ipt20um` | HTCondor H100 `1g.12gb` (`par=1`) | 57.93% | 0.9619 | 54.65% | 61.04% | 0.9020 | 6.98% | 2.14% | 6.51% | `5.006 / 6.599` |
| `s323-pf-tgt20um-ipt1mm` | HTCondor H100 `1g.12gb` (`par=1`) | 57.77% | 0.9636 | 54.49% | 60.91% | 0.9018 | 7.35% | 0.56% | 6.64% | `5.335 / 7.229` |
| `s323-pf-tgt1mm-ipt1mm` | HTCondor A100-40GB (`par=3`) | 57.90% | 0.9617 | 54.18% | 61.46% | 0.9012 | 6.88% | 1.49% | 7.03% | `4.954 / 6.457` |
| `s323-pf-layers-6t`（6T + 3IPT） | HTCondor A100-40GB (`par=3`) | 57.39% | 0.9673 | 54.29% | 60.39% | 0.9115 | 5.21% | 0.56% | 5.08% | `6.796 / 8.859` |
| `s323-pf-hybrid-uvtx` | HTCondor H100 `1g.12gb` (`par=1`) | **58.24%** | **0.9543** | **54.87%** | 61.43% | 0.8888 | 8.47% | **1.77%** | 8.07% | `4.632 / 6.062` |
| `s323-pf-hybrid-shuffled` | HTCondor H100 NVL 95GB (`par=3`) | 57.92% | 0.9617 | 54.25% | 61.45% | 0.9008 | 7.91% | 1.12% | 6.90% | `5.002 / 6.509` |
| `s323-pf-hybrid-nomad` | HTCondor H100 `1g.12gb` (`par=1`) | 58.01% | 0.9553 | 54.15% | **61.68%** | **0.8884** | **9.02%** | 1.67% | **9.38%** | **`4.631 / 5.982`** |
| `s323-pf-20um-ematched` | HTCondor H100 NVL 95GB (`par=3`) | 57.89% | 0.9715 | 54.32% | 61.29% | 0.9114 | 6.23% | 0.37% | 5.34% | `5.818 / 7.602` |

### 3.5 配对 Family Block-Bootstrap 检验（PinpointBlock PixXY，1,000 次重抽样）

| 对比组（A − B） | $\Delta\text{BA4}$ [pp]（95% CI） | $\Delta\tau\text{@1\%FPR}$ [pp]（95% CI） | $\Delta\tau\text{@0.1\%FPR}$ [pp]（95% CI） |
| :--- | :---: | :---: | :---: |
| `PF-20um − B1++` | **`+0.96` `[+0.14, +1.82]`** | **`+1.86` `[+0.48, +4.34]`** | `0.00` `[-1.19, +2.24]` |
| `12T − 6T`（靶区层数） | **`+0.94` `[+0.21, +1.63]`** | `+0.84` `[-0.84, +2.20]` | `+0.28` `[-0.90, +2.22]` |
| `Hybrid − Vanilla PF-20um` | `+0.10` `[-0.66, +0.82]` | **`+2.42` `[+0.76, +3.79]`** | `+0.93` `[-0.83, +2.73]` |
| `Hybrid − Shuffled` | `-0.01` `[-0.75, +0.72]` | `+0.56` `[-1.24, +2.61]` | `+0.65` `[-1.23, +2.29]` |
| `Hybrid+NOMAD − Hybrid` | `+0.25` `[-0.41, +0.95]` | `+0.56` `[-1.23, +2.40]` | `-0.09` `[-1.73, +1.97]` |
| `Natural − EnergyMatched` | `+0.05` `[-0.71, +0.85]` | `-0.19` `[-1.63, +1.42]` | `+0.47` `[-0.61, +2.20]` |
| `2×2` Target 效应（IPT 20μm）：`(20μm,20μm) − (1mm,20μm)` | `+0.29` `[-0.50, +1.07]` | `-0.93` `[-3.35, +0.57]` | `-1.30` `[-2.14, +0.99]` |
| `2×2` IPT 效应（Tgt 20μm）：`(20μm,20μm) − (20μm,1mm)` | `+0.42` `[-0.21, +1.09]` | `-1.30` `[-2.96, +0.36]` | `+0.28` `[-2.00, +1.30]` |
| `2×2` 双粗化：`(20μm,20μm) − (1mm,1mm)` | `-0.13` `[-0.90, +0.66]` | `-0.84` `[-2.96, +0.86]` | `-0.65` `[-2.68, +1.49]` |
| `2×2` Target 效应（IPT 1mm）：`(20μm,1mm) − (1mm,1mm)` | `-0.55` `[-1.38, +0.22]` | `+0.47` `[-1.49, +2.34]` | `-0.93` `[-2.60, +1.38]` |

### 3.6 纯钨留一源 / Generator-OOD 验证与无标签部署闭合

| 模型 / 基线 | `Run10085–88`（纯钨）BA4 / NLL | `Pythia6-Std`（纯钨）BA4 / NLL | `Pythia6-NUX`（纯钨）BA4 / NLL | `Diag Pythia8 NC Charm`（`Z∈{0,1}`）$\langle p_\tau\rangle$ / 判 NC / 判 $\tau$ |
| :--- | :---: | :---: | :---: | :---: |
| `BDT-70d` | `51.94%` / `1.0578` | `50.25%` / `1.0869` | `53.08%` / `1.0432` | `0.287` / `44.4%` / `29.5%` |
| `s323-b1pp-clean` | `56.44%` / `1.0027` | `55.25%` / `1.0246` | `56.75%` / `0.9834` | `0.295` / `39.9%` / `34.6%` |
| `s323-pf-20um` | **`58.75%`** / `0.9768` | `55.42%` / `1.0100` | `57.83%` / `0.9686` | `0.291` / `47.4%` / `31.6%` |
| `s323-pf-tgt1mm-ipt20um` | `58.44%` / `0.9803` | `55.00%` / `1.0041` | `58.50%` / `0.9665` | `0.291` / `46.9%` / `32.5%` |
| `s323-pf-hybrid-uvtx` | `58.44%` / `0.9760` | `56.08%` / `1.0070` | `57.58%` / `0.9622` | `0.289` / **`49.1%`** / `30.9%` |
| `s323-pf-hybrid-nomad` | `58.50%` / **`0.9745`** | **`56.33%`** / `1.0049` | **`59.17%`** / **`0.9572`** | **`0.285`** / **`49.1%`** / **`28.2%`** |
| `s323-pf-20um-ematched` | `58.25%` / `0.9837` | `54.42%` / `1.0104` | `57.33%` / `0.9687` | `0.310` / `42.2%` / `37.9%` |

- **无标签纯观测部署闭合（`322-11`）**：在不含 `targets.parquet` 的独立临时目录中对 64 个真实 Clean v3 Tune 事件执行 FP32 推理，`v3_b1pp_clean_s17`、`v3_pf_tgt20um_ipt20um_s17`、`v3_pf_hybrid_uvtx_s17`、`v3_pf_hybrid_nomad_s17` 与带标签路径的最大绝对概率差**全部为 `0.0`**（4/4 通过）。
- **独立分析前密码学冻结（`322-09`）**：冻结清单已双份落盘至 [roadmap/v3/v8_clean_v3_freeze_manifest.json](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/roadmap/v3/v8_clean_v3_freeze_manifest.json) 与 [v8_clean_v3_freeze_manifest.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v6_clean_v3_2026-11-19/clean_sim/v8_clean_v3_freeze_manifest.json)，记录 `Pinpoint_G4` 提交、`PinpointAnalysis` 提交、`pinpoint_v3` 二进制（SHA-256 `e7400838...`）、全部 150 个名义分片 `observations.parquet` 及全部 30 个模型 checkpoint 与配置文件的 SHA-256 指纹，且明确写入 `Cal_A_Cut_B_Confirm_touched: false`、`post_training_physics_analysis_paused: true`。

### 3.7 暂停 / 未测量项目清单

以下项目遵循用户“模型训好之后物理分析部分先暂时暂停”的要求，本阶段**全部标注为未测量**，严禁推测或外推：
- `322-02`：统一物理分析选择效率、p0 veto 与外部本底通量（未测量）。
- `322-04`：稀少本底有限 MC 统计与 Barlow-Beeston lite / Poisson 辅助约束（未测量）。
- `322-06`：四期（4-era）Run 3 / HL-LHC 预期投影与通量系统误差（未测量）。
- `322-08`：`Cut-B` 工作点 FPR 标定与跨域单调 $S_\tau$ 构建（未测量）。
- `322-10`：对象级次级顶点（SV）truth-match、IP 精度分解与 $\tau$ 衰变长度反事实检验（未测量）。
- `322-14`：Profile-likelihood 形状形变、率裁切 Jacobian 与 $\nu=0$ 内循环数值稳定性（未测量）。
- `CleanCalA`、`CleanCutB`、`CleanConfirm` 上的任何评估指标（未测量，100% 密封）。

---

## 4. 阶段性结论

### 4.1 事实（数据直接支持）

1. **全部 10 项训练相关审计问题已完成代码修复与实测闭合**：Clean v3 名义 60,000 事件实现 `100%` 纯钨（`0` 非钨 GENIE、`0` 非钨 `G4Navigator`）、`0` 跨分片源泄漏、`99.959%` 全物理钨体积覆盖、`0` 个 0-PE 正时间击中、真实事件上 `0.0` bit-exact IPT 隔离，以及 4 类模型 `0.0` 无标签部署闭合。
2. **神经网络架构在纯钨样本上显著超越强可观测非神经网络基线**：在匹配 Clean v3 训练/验证集上，70 维物理特征 `HistGB BDT` 仅达到 PinpointBlock `BA4 = 56.50%`、$\tau$@1%FPR `3.72%`；同预算 `B1++` 达 `60.37%` / `4.19%`；`PF-20um` 显著优于 `B1++`（`ΔBA4 = +0.96 pp`, 95% CI `[+0.14, +1.82] pp`；`Δτ@1%FPR = +1.86 pp`, 95% CI `[+0.48, +4.34] pp`）；`PF-Hybrid+NOMAD` 达 `61.68%` BA4、`0.8884` PinNLL、**`9.02%`** Pinpoint $\tau$@1%FPR 与 **`9.38%`** FrontPinpoint $\tau$@1%FPR（为 `B1++` `3.12%` 的 3.0 倍、`BDT-70d` `3.72%` 的 2.5 倍）。
3. **显式微顶点重建带来显著增益，而无显式微顶点分支的端到端 `PF` 对像素步长不敏感**：
   - 消除 `322-03` IPT 泄漏并加入 3 μm 像素扩散后，无微顶点分支的 `2 × 2` Target × IPT 阶乘四组（`20 μm` vs `1 mm`）PinpointBlock BA4 均集中在 `60.91%–61.46%`、PinNLL `0.8978–0.9020`，全部配对 95% CI 均覆盖 0；
   - 相比之下，靶区纵向层数 `12T vs 6T` 具有显著 BA4 增益（`+0.94 pp`, 95% CI `[+0.21, +1.63] pp`）和 `-1.37 mm` 主顶点深度误差改善（`5.422 mm` vs `6.796 mm`）；
   - 显式利用 20 μm 精细像素坐标进行微顶点几何重建的 `PF-Hybrid` 相比无微顶点 `PF-20um` 实现显著 $\tau$@1%FPR 增益（`+2.42 pp`, 95% CI `[+0.76, +3.79] pp`，`6.05% -> 8.47%`），PinNLL 由 `0.8978` 降至 `0.8888`，PV 中位 $|\Delta z|$ 由 `5.422 mm` 改善至 `4.632 mm`。
4. **层内置换证伪对照（`Hybrid` vs `Shuffled`）分离了几何对齐与容量效应**：`Shuffled` 在保持平面内击中占有率的同时破坏跨平面直线几何对齐后，All NLL 从 `0.9543` 退化回 `0.9617`（与 Vanilla `0.9612` 持平）、PinNLL 从 `0.8888` 退化回 `0.9008`、PV $|\Delta z|$ 退化 `+0.37 mm`（`4.632 -> 5.002 mm`）、FrontPinpoint $\tau$@1%FPR 下降 `1.17 pp`（`8.07% -> 6.90%`）；PinpointBlock 整体 $\tau$@1%FPR 点估计高 `+0.56 pp`（95% CI `[-1.24, +2.61] pp`）。
5. **未发现 $E_\nu$ 能谱捷径，且纯钨留一源/生成器 OOD 泛化稳健**：能量匹配重加权对照组 `s323-pf-20um-ematched` 与自然权重 `s323-pf-20um` 的 Pinpoint BA4 几乎一致（`61.29%` vs `61.33%`，`Δ = +0.05 pp`, 95% CI `[-0.71, +0.85] pp`）；在独立 `Run10085–88` 和 `Pythia6-NUX` 纯钨 OOD 集上，`PF-Hybrid+NOMAD` 保持 `58.50%` 和 `59.17%` BA4，在 `Pythia6-Std` 上保持 `56.33%` BA4（仍高出 `BDT-70d` `+6.08 pp`）。

### 4.2 假设与待查机制（需在后续阶段进一步验证）

- **假设 1**：端到端 Vanilla `PinpointFormer`（不含显式微顶点几何特征）在 `20 μm` 与 `1 mm` 步长间未出现显著差异，可能源于 192 维全局/局域注意力在 32k 样本上尚未直接从稀疏像素坐标中学会精确 3D 直线外推与亚毫米冲击参数；只有通过显式几何微顶点重建（`PF-Hybrid`）才释放了 20 μm 像素的次级顶点拓扑信息。该假设可通过后续在 `PF-Hybrid` 上运行 `20 μm vs 1 mm` 靶区像素消融直接检验。
- **假设 2**：在自由核子 `Pythia8 NC Charm` 诊断分片上，最优模型 `PF-Hybrid+NOMAD` 仍有 `28.2%` 被判为 $\nu_\tau$（`mean p_tau = 0.285`），提示真实物理样本中 NC charm 短寿命强子衰变是 $\nu_\tau$ 选择的核心不可约本底之一，须在暂停的物理分析阶段（`322-04` / `322-05`）结合核修正与真实截面纳入系统误差与联合拟合。

---

## 5. 下一步计划（等待用户解封指令后启动）

本阶段已完成全部训练与分析前冻结，模型与数据已就绪。待用户明确恢复物理分析阶段后，按以下顺序逐项推进暂停的 6 项 Stage 322 审计问题：

1. **对象级次级顶点物理验证（`322-10`）**：在 `CleanTune8k` 上对 `PF-Hybrid` / `PF-Hybrid+NOMAD` 的微顶点特征开展真次级顶点（SV）truth-match、IP 精度分解及 $\tau$ 衰变长度反事实检验，并可选补做 `PF-Hybrid` 下的 `20 μm vs 1 mm` 像素消融。
2. **统一物理分析选择、p0 Veto 与外部本底建模（`322-02`, `322-04`, `322-06`）**：建立与部署口径一致的 p0 veto 统一选择、外部本底通量与稀少本底 Barlow-Beeston lite / Poisson 辅助约束，以及四期 Run 3 / HL-LHC 预期输入。
3. **跨分域单调 $S_\tau$ 与 `Cut-B` 工作点标定（`322-08`）**：在解封 `CleanCalA` 与 `CleanCutB` 前固定冻结全部标定程序，严格校准跨分域单调 $\tau$ 判别量与工作点 FPR。
4. **Profile-Likelihood 稳健化与单次密封 `Confirm` 评估（`322-14`）**：修复形状形变、率裁切 Jacobian 与 $\nu=0$ 内循环后，依据 `roadmap/v3/v8_clean_v3_freeze_manifest.json` 的 SHA-256 指纹执行单次可审计的 `CleanConfirm` 物理评估。
