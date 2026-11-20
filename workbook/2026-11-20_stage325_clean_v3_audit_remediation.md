# Stage 325：Stage 324 Clean v3 独立科学审计模型训练与性能相关问题全面修复

- **日期**：2026-11-20
- **PinpointAnalysis 训练锁定 Commit SHA**：`51c3d8966426286de69df8cecacadb314643611f`（全部 27 个模型共享完全一致的代码包 SHA-256：`302dc79d27103752bfd785f59e79925a9c671537b4377a48dfa4c8448694712e`）
- **Pinpoint_G4 Commit SHA**：`48a9e3deadb21cef5ed3fc8c9bc53edcf4a624bf`（分支 `fix/audit-geometry-digitizer-physics`，产物 `pinpoint_v3_1` SHA-256：`629e0efd75d36c34f65f89fd984dc4e7ce6fcb2f7388e7ccfe45b1be72d32726`）
- **Stage 325 结果根目录**：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v7_stage325_2026-11-20`
- **冻结清单**：`roadmap/v3/v9_stage325_freeze_manifest.json`（与 EOS 备份完全一致）
- **盲态状态**：`Cal_A_Cut_B_Confirm_touched = False`，`sealed_shards_accessed_during_freeze = 0`（封存分片 `100..149` 及 `sealed_truth_meta/` 零打开、零 `stat`、零哈希）；模型训练后的物理分析项（`322-02/04/06/08/10/14`）按指示保持**暂停（未测量）**。

---

## 1. 阶段目标

逐项修复 [Stage 324 Clean v3 独立科学审计报告](2026-11-20_stage324_independent_clean_v3_scientific_audit.md) 中全部 15 项与模型训练、特征提取、数据隔离、模拟门禁和模型评估表现相关的问题（`324-01` 至 `324-15`），并在严格零封存访问约束下重新完成单 Commit、等预算、多随机种子（`17, 42, 2026`）的模型训练与独立冻结验证：

| 审计编号 | 级别 | 修复目标与验收标准 |
| --- | --- | --- |
| `324-01` | P1 | 路径级 `split_shards.json` 白名单 + 运行时 `sys.addaudithook`，保证训练与冻结阶段对封存分片 `100..149` 的 `open`/`stat`/`scandir` 次数严格为 `0`。 |
| `324-02` | P1 | 全窗口精确加权分子/分母微批梯度累积（`compute_window_task_denominators`），在真实 128 事件 4-task 掩码加权批量上验证 `bs=128` 与 `64×2` 梯度 L2 相对误差 `< 1e-4`。 |
| `324-03` | P1 | 严格共同支撑 `[50, 2312] GeV`、四类精确等权重和、零权重事件前置过滤（`30,712` 事件）、四项多任务损失全加权（`weight_all_tasks=True`）的能量匹配对照模型 `s325-pf-20um-ematched-v2`，并增加 `(E_nu, nu/nubar)` 匹配验证。 |
| `324-04` | P1 | 使用 `Pythia 8.315 + LHAPDF6:EPPS21nlo_CT18Anlo_W184/0` 核 PDF 与 `260 MeV/c` 核费米动量，经 `pinpoint_v3_1` 纯钨传输生成 `1,200` 条 NC inclusive 与 `1,200` 条 NC charm 钨-184 核 OOD 验证分片（如实标注不含核内 FSI 级联）。 |
| `324-05` | P1 | 训练并加密冻结 `BDT-70d`、`LR-6d/16d/70d`、`Occupancy-10d`、`Energy-only`（保存 `.pkl`、逐事件 `parquet` 预测与 SHA-256）；在严格排除 `tune_screen` 的 `Tune6k` 上以原始 3-seed 平均、配对 Bootstrap 和 KEY 哈希互斥 split-half 锁定 `1% / 0.1% FPR` 工作点（含 Clopper-Pearson 95% CI 与可观测量域 `D1/D2/D3` 分层）。 |
| `324-06` | P1 | 在 `spectrometer.py` 引入 Magnet 1+2 孔径穿越检查（`check_magnet12_bore_traversal`，围绕 `(0, -10) mm` 的 `R_fid = 96 mm` 有效孔径），对穿壳散射三站径迹施加质量罚分、置 `cand[:, 0] = 0.70` 并清零无效矢高/动量；运行 `pinpoint_v3_1` 轴上 vs 穿壳 `100 GeV mu+` 实时闭环验证与 Tune8k 孔径审计。 |
| `324-07` | P1 | 修复 C++17 微顶点内核 `_fast_kernels.cpp`，移除导致中间 Pinpoint 块失效的 `65 mm / 105 mm` 间距截断（支持 `dz_ab <= 185 mm`、`dz_ac <= 340 mm` 及 Highland $\sqrt{\Delta z / 30\text{ mm}}$ 多重散射放缩），输出按 `FrontPinpoint_30mm` 与 `IntermPinpoint_150mm` 分层的对象级 PV/SV 诊断。 |
| `324-08` | P1 | 新增固定聚类 1 mm 纯坐标量化（`quantize_cluster_coords_only_mm=1.00`，保持簇数、token 数、`c_obs` 及 IPT 比特完全一致），同时在 `PF-20um` 和 `PF-Hybrid v2` 上运行配对消融，并将 Stage 323 `6T` 模型严格重评在其自身 `6T` 可观测量域上。 |
| `324-09 / 14` | P2/P3 | 确定性 **FP32** 最终 Tune 预测导出（`eval_fp32=True`），跨卡落盘-纯观测复现闭环检查；确定性零时间戳源码打包（`PINPOINT_CODE_ARCHIVE_SHA256` 跨全部作业比特一致）；记录入口 `UTC_started`、出口 `UTC_finished`、GPU、VRAM、`grad_accum`、PyTorch/CUDA 版本。 |
| `324-10/11/12/13/15` | P2 | `GFaserGenerator.cc` 严格离子 PDG 钨靶门禁（7 组实时 `pinpoint_v3_1` 用例）；`PixelSD.cc` 可选随机暗噪声注入（`PINPOINT_PIXEL_DARK_RATE`）；`pinpoint_v3_1` 实时 $\tau$ V-A 螺旋度极化闭环；正则化对照 `s325-pf-hybrid-v2-reg`（`drop_path=0.18, head_dropout=0.25, weight_decay=0.06`）；7 项探测器退化压力测试；PV Pull 标定审计。 |

---

## 2. 工作内容

### 2.1 `Pinpoint_G4` 物理与数字化器加固（Commit `48a9e3d`）

1. **`Pinpoint/src/generators/GFaserGenerator.cc`（`324-15`）**：在 `PINPOINT_REQUIRE_W_TARGET=1` 门禁下，强制检查离子 PDG 范围 `1000000000 <= fTgtPdg < 1100000000`、从 PDG 解码的 `pdgZ == 74` 与 `fTgtZ == 74` 一致、`pdgA == fTgtA` 且 `180 <= pdgA <= 186`，堵住自由质子 `2212` 伪造 `Z=74` 的绕过漏洞。
2. **`Pinpoint/include/PixelSD.hh` 与 `Pinpoint/src/PixelSD.cc`（`324-10`）**：补充 `#include <unordered_map>` 并新增环境变量 `PINPOINT_PIXEL_DARK_RATE`（默认 `0.0`），按泊松分布每事件每层独立注入随机暗噪声像素击中。
3. **编译构建**：在 `LCG_107/x86_64-el9-gcc11-opt` 环境下零警告零错误编译生成 `/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v7_stage325_2026-11-20/g4/pinpoint_v3_1`（SHA-256：`629e0efd75d36c34f65f89fd984dc4e7ce6fcb2f7388e7ccfe45b1be72d32726`）与 `macros/geom.mac`（SHA-256：`69176c4149c8808f7e14837911d11f60528c9281437759f893481c74e06bf5bf`）。

### 2.2 `PinpointAnalysis` 代码修复与单元测试（Commit `51c3d89`）

1. **零封存路径白名单（`324-01`）**：在 `src/analysis/flavour/v3/common/datasets.py`、`group.py` 和 `produce_clean_v3.py` 中优先读取 `data/split_shards.json`，仅枚举与暂存授权的 `train`（`000..079`）与 `validation`（`080..099`）分片，完全不列出、不 `stat`、不打开封存分片 `100..149`。
2. **精确全窗口梯度累积与全任务样本加权（`324-02`, `324-03`）**：更新 `vertex_anchor.py`、`v3_multitask/losses.py`、`v3_multitask/models.py`、`v3_multitask/train.py` 与 `v2_pinpointformer/train.py`，新增 `compute_window_task_denominators()` 预先计算整个逻辑批量窗口的四任务精确分母，实现任意微批拆分下的精确梯度等价；支持 `weight_all_tasks=True`（将能量匹配权重施加到分类、Charm、Vertex 与 Kinematics 全部 4 项损失）和 `filter_zero_weight_events=True`（训练前剔除零权重事件）。
3. **孔径感知谱仪质控（`324-06`）**：在 `src/analysis/flavour/v3/v1_b1pp/spectrometer.py` 中新增 `check_magnet12_bore_traversal()`，对 Magnet 1 与 Magnet 2 入口/中点/出口逐点校核距束轴 `(0, -10) mm` 的半径是否位于 `R_fid = 96 mm` 磁铁内孔；穿壳三站径迹质量分惩罚 `+25.0`、质量标志 `cand[:, 0]` 降为 `0.70`（低于 `nomad_kinematics.py` 的 `0.75` 有效阈值）且无效矢高/动量清零。
4. **全靶区模块感知 C++17 微顶点内核（`324-07`）**：更新 `src/analysis/flavour/v3/common/_fast_kernels.cpp`，允许相邻靶区像素层最大间距 `dz_ab <= 185 mm`、三点最大跨度 `dz_ac <= 340 mm`，按 Highland 公式 $\sqrt{\Delta z / 30\text{ mm}}$ 放缩多重散射容差，在 150 mm 间距中间 Pinpoint 块放松 `prior_z_prec`，使前部 30 mm 与中间 150.01 mm Pinpoint 块均可构建 3-hit 径迹段并完成 AVR PV 与位移 SV 拟合。
5. **固定聚类 1 mm 纯坐标消融与 SiPM 上限透传（`324-08`, `324-10`）**：在 `pixel_tokenizer.py` 中新增 `quantize_cluster_coords_only_mm`（先以原生 20 μm 完成像素聚类，再仅对靶区像素簇质心 `(x, y)` 做对称栅格量化，严格保持簇数、token 数、`c_obs` 及 IPT 比特不变）；在 `domains.py` 中支持 `sipm_invert_max_occupancy` / `sipm_sat_cap` 透传；在 `submit.py` 中使用确定性 `mtime=0` 源码打包确保跨作业 `PINPOINT_CODE_ARCHIVE_SHA256` 比特一致；在 `v1_b1pp/train.py` 与 `v2_pinpointformer/train.py` 中启用确定性 `eval_fp32=True` 并完整记录入口/出口 UTC 时间戳与软硬件元数据。
6. **回归单元测试**：新增 `tests/test_stage325_audit_remediation.py`（5 项专项单测覆盖 `324-01/02/06/07/08`），提交前全量运行 `python3 -m unittest discover -s tests`，**`363 / 363` 全部通过（`OK`, 133.2 s）**。

### 2.3 硬件调度、单卡并行与 HTCondor Schedd 选择依据

- **HTCondor Schedd 选择**：`choose_schedd()` 检查兼容 AlmaLinux9 与 EOS 写入策略且排队最少的 schedd，选定 `bigbird20.cern.ch`（Kerberos `WRITE ALLOW`）。
- **实时资源探测与单卡并行策略**：提交前探测 HTCondor 未占用空闲整卡池共 89 张（60 × `A100-PCIE-40GB`、25 × `H100 NVL 95GB`、4 × `H200 143GB`，其中 46 张为非抢占正常槽位），单节点均具备 8–16 CPU 与 64–118 GB 内存；Kubeflow 侧 `h100-94gb` 当时无空闲节点，故及时改派 `h100-47gb`（`NVIDIA-H100-NVL MIG 3g.47gb`）实现零排队启动。
- **单卡多进程并行**：全部 9 个训练组（27 个模型）全部采用 **`--parallel 3`**（每卡同时运行 3 个随机种子进程共享同一份本地内存映射特征缓存，`grad_accum=1`，`loader_workers=2` 匹配 6 CPU 核），在 **19–69 分钟** 内零失败完成全部 27 个模型训练（`8` 组运行于 HTCondor `H100 NVL` / `A100-40GB`，`1` 组运行于 Kubeflow `H100 MIG 3g.47gb`）。
- **多核 CPU 加速**：W-184 EPPS21 OOD 模拟采用 HTCondor 2 任务 × 8 CPU 多进程（13 分钟完成 2,400 条 `pinpoint_v3_1` 事件，零 `.root` 中间文件保留）；`frozen-baselines` 与 `freeze-and-validate` 的特征/OOD/退化测试提取改写为 16 核 `fork` 多进程 + 本地 `/tmp` NVMe `.pt` 批缓存（避免 PyTorch 跨进程 fd 句柄上限与 CVMFS `spawn` 重复导入开销），将 `frozen-baselines` 从 >60 分钟加速至 `358.8 s`。

---

## 3. 详细数据

### 3.1 执行命令与环境

```bash
# 运行环境
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export LHAPDF_DATA_PATH="/cvmfs/sft.cern.ch/lcg/external/lhapdfsets/current:$LHAPDF_DATA_PATH"

# 1. 全量单元测试 (363/363 OK)
python3 -m unittest discover -s tests -v

# 2. 数据升级、实时 G4 v3.1 闭环探针与 W-184 EPPS21 OOD HTCondor 提交 (cluster 1584901)
python3 -u -m analysis.flavour.v3.v6_clean_sim.run_stage325_remediation prepare-v3-upgrades

# 3. 本地 Tesla T4 真实模型 324-02 梯度等价、324-06 孔径与 324-07 微顶点对象审计
python3 -u -m analysis.flavour.v3.v6_clean_sim.run_stage325_remediation verify-gpu-gates

# 4. 从同一 Commit 51c3d89 提交全部 9 组 (27 模型): Kubeflow h100-47gb + HTCondor A100/H100
python3 -u -m analysis.flavour.v3.v6_clean_sim.run_stage325_remediation submit-group \
  --group s325-pf-hybrid-v2 --backend kubeflow --kubeflow-flavor h100-47gb --parallel 3
for g in s325-b1pp-clean s325-pf-20um s325-pf-20um-coord1mm s325-pf-hybrid-v2-coord1mm \
         s325-pf-hybrid-v2-shuffled s325-pf-hybrid-v2-nomad s325-pf-20um-ematched-v2 s325-pf-hybrid-v2-reg; do
  python3 -u -m analysis.flavour.v3.v6_clean_sim.run_stage325_remediation submit-group \
    --group "$g" --backend condor --parallel 3 --min-gpu-mem 36000
done

# 5. 非神经网络基线与负对照冻结 (16 核 fork, 358.8 s)
python3 -u -m analysis.flavour.v3.v6_clean_sim.run_stage325_remediation frozen-baselines

# 6. 零封存访问冻结、全量验证、OOD/退化测试与 300-DPI 英文图表渲染 (1,216.4 s)
python3 -u -m analysis.flavour.v3.v6_clean_sim.run_stage325_remediation freeze-and-validate
```

### 3.2 实时 `pinpoint_v3_1` 物理与数字化器门禁闭环（`324-15/10/06/13`）

产物：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v7_stage325_2026-11-20/g4/live_g4_v3_1_closure.json`（耗时 `218.0 s`）：

1. **`324-15` 严格钨靶与材料门禁矩阵（`7 / 7` 全通过）**：`W184_in_W` (`rc=0`)、`W186_in_W` (`rc=0`) 通过；`C12_in_W`、`proton2212_fakeZ74`、`W184_mismatchA186`、`unknown_pdg0_fakeZ74`、`W184_in_Air` 均被 `G4Exception` 致命拦截（`rc=-6`）。
2. **`324-10` 像素随机暗噪声注入验证（20 事例 × 50 GeV $\mu^-$）**：`PINPOINT_PIXEL_DARK_RATE=0.0` 时平均总像素击中 `27.75 / event`；`1e-5` 时增至 `16,514.8 / event`（净增 `+16,487.05 / event`，与 15 层 × $10500^2 \times 10^{-5} = 16,537.5$ 的泊松理论值吻合至 `0.3%`）。
3. **`324-06` 轴上孔内 vs 穿壳 `100 GeV mu+` 谱仪实时闭环（每组 60 事例）**：
   - 轴上孔内 `(x=0, y=-10 mm)`：`60/60` 形成 3 站径迹，**`60/60` 孔内接收（`0` 标记穿壳）**，原始平均矢高 `-1.832 mm`，错误符号 `1/60`（`1.7%` 硬轫致辐射尾）。
   - 穿壳 `(x=112, y=-10 mm)`：`60/60` 形成 3 站径迹，原始矢高均值反向为 `+1.346 mm`（区间 `[-67.75, +34.01] mm`，**`33/60 = 55.0%` 错误电荷符号**）；经 `check_magnet12_bore_traversal` 质控后 **`0/60` 孔内接收、`60/60` 标记穿壳、`0/60` 错误符号进入有效动量支路**。
4. **`324-13` `Py8Decayer` 实时 $\tau^-$ V-A 螺旋度极化闭环（每组 600 条 50 GeV 空中衰变 $\tau^-$）**：

| 螺旋度设定 | 有效衰变事例 | 全可见动量均值 $\langle z_{\text{vis}} \rangle$ | 1-prong $\tau^- \to \pi^- \nu_\tau$ 事例数 | 实测 $\langle z_\pi \rangle$（理论值） | 轻子 1-prong 事例数 | 实测 $\langle z_\ell \rangle$（理论值） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `VA_left` ($h=-1.0$) | 599 | 0.5023 | 61 | **0.3070**（0.3333） | 230 | **0.3849**（0.4000） |
| `unpolarized` ($h=0.0$) | 599 | 0.5458 | 62 | **0.5369**（0.5000） | 220 | **0.3559**（0.3500） |
| `right` ($h=+1.0$) | 599 | 0.5608 | 61 | **0.6812**（0.6667） | 207 | **0.3174**（0.3000） |

### 3.3 真实 128-事件 GPU 梯度等价门禁（`324-02`）、Tune8k 孔径（`324-06`）与分区域微顶点对象审计（`324-07`）

产物：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v7_stage325_2026-11-20/stage325_gpu_gates_and_object_audit.json`：

- **`324-02` 真实 128-事件 FP32 全窗口微批累积等价验证（`bs=128` vs `64×2`，`Dropout/DropPath=0` 保持 `.train()` 模式）**：
  - `PF_Multitask_weightAll`：`loss_full = 11.409535`，`loss_accum = 11.409534`，`loss_abs_diff = 1.43e-6`，`global_grad_l2_rel_diff = 1.47e-6`，`max_param_abs_grad_diff = 2.24e-5`，`adamw_1step_l2_rel_diff = 1.92e-4`（**PASS**）。
  - `PF_V4_NOMAD_procWeight`：`loss_full = 11.295197`，`loss_accum = 11.295196`，`loss_abs_diff = 4.77e-7`，`global_grad_l2_rel_diff = 1.02e-6`，`max_param_abs_grad_diff = 1.34e-5`，`adamw_1step_l2_rel_diff = 3.85e-4`（**PASS**）。
- **`324-06` CleanTune8k 谱仪孔径审计**：总候选 `3,988` 条（含 2-station 候选 `1,427` 条）；在 `2,561` 条 3-station 候选中，孔内有效候选 `1,649` 条、**穿壳候选 `912` 条（占 3-station 候选的 `35.6%`）**；`1,016` 个事例以孔内径迹为主导、`516` 个事例的穿壳主导径迹已被标记并清零无效动量。
- **`324-07` CleanTune8k 分区域微顶点对象级审计（`FrontPinpoint_30mm` vs `IntermPinpoint_150mm`）**：

| 物理区域 | 事例数 | AVR PV 有效率 | 平均径迹段数 | 平均 3-hit 径迹段数 | AVR PV 中位 $|\Delta z|$ [mm] | AVR PV 中位 3D 误差 [mm] | `tau_had` / `CC_charm` / `mu_CC` / `NC` 最大 IP 均值 [μm] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `FrontPinpoint_30mm` | 3,111 | **96.79%** | 24.91 | 17.41 | 13.76 | 18.22 | 1067.6 / 1080.9 / 915.1 / 817.3（`e_CC`: 1213.6） |
| `IntermPinpoint_150mm`（修复前 `0.0%`） | 1,228 | **65.15%** | 11.40 | 4.26 | 41.29 | 43.09 | 449.1 / 434.0 / 331.2 / 222.4（`e_CC`: 332.0） |
| `AllPinpoint_PixXY` | 4,339 | **87.83%** | 21.08 | 13.69 | 22.46 | 24.49 | 891.8 / 894.1 / 750.1 / 651.9（`e_CC`: 964.5） |

- **`324-03` 严格共同支撑等权重表审计**（`train_energy_match_weights_v2_audit.json`）：区间 `[50, 2312] GeV`，四类权重总和精确相等（均为 `7678.000`），有效样本量 `ESS` 分别为 `7370.2 / 7292.5 / 7560.9 / 7351.2`，加权中位能量分别为 `542.7 / 548.8 / 551.9 / 547.6 GeV`，`1,288` 条越界零权重事例在训练前剔除（实际训练 `30,712` 条）。

### 3.4 27 个训练模型健康度与 9 组（+ 6 非神经网络基线）严格排除 `tune_screen` 的 `Tune6k` 性能总表

全部 27 个模型均来自同一提交 `51c3d89` 与同一源码包 SHA-256 `302dc79d27103752bfd785f59e79925a9c671537b4377a48dfa4c8448694712e`，`27/27 complete`、`0 failed`、`eval_precision = fp32`、`grad_accum = 1`、`UTC_started < UTC_finished` 无异常，验证集 NLL 均从初始值平稳收敛至 `0.978–1.007`（无常数预测、无饱和、无发散）。

| 模型 / 对照组（3-seed 原始均值） | Tune6k All BA4 [%] | Tune6k All NLL | Tune6k $(E_\nu,\nu/\bar\nu)$ 匹配 BA4 [%] | Tune6k Pinpoint BA4 [%] (NLL) | Front30 / Interm150 BA4 [%] | 2t 质量加权 BA4 [%] | Split-Half 1% FPR $\tau$ 效率 (TP/746, FP/2187) | Split-Half 1% FPR `D1` 效率 (TP/149, FP/482) | Front30 / Interm150 神经 PV 中位 $|\Delta z|$ [mm] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| `ctrl_energy_only_1d`（冻结核查） | 31.88 | 1.3558 | 未测量 | 未测量 | 未测量 | 32.13 | 1.34% (10, 18) | 1.34% (2, 6) | — |
| `lr_presence_counts_6d` | 40.29 | 1.2776 | 未测量 | 未测量 | 未测量 | 35.59 | 1.74% (13, 18) | 2.01% (3, 5) | — |
| `ctrl_occupancy_only_10d` | 46.00 | 1.1906 | 未测量 | 未测量 | 未测量 | 38.52 | 2.41% (18, 34) | 2.68% (4, 6) | — |
| `lr_c_obs_16d` | 47.78 | 1.1589 | 未测量 | 未测量 | 未测量 | 44.03 | 2.28% (17, 34) | 2.01% (3, 4) | — |
| `lr_full_obs_physics_70d` | 50.36 | 1.1181 | 未测量 | 未测量 | 未测量 | 46.09 | 1.88% (14, 13) | 2.01% (3, 3) | — |
| `bdt_histgb_full_obs_70d` | 53.50 | 1.0451 | 未测量 | 未测量 | 未测量 | 50.35 | 2.95% (22, 16) | 2.68% (4, 2) | — |
| `s325-b1pp-clean`（H100 NVL, 19m） | 56.99 | 0.9883 | 55.85 | 60.52 (0.9215) | 61.59 / 57.65 | 53.71 | 3.89% (29, 30) | 2.68% (4, 4) | — |
| **`s325-pf-20um`**（A100, 56m） | **58.07** | **0.9626** | 56.79 | 61.57 (**0.8956**) | 62.90 / 58.15 | **54.60** | 6.30% (47, 29) | **7.38% (11, 6)** | 5.15 / **5.81** |
| `s325-pf-20um-coord1mm`（H100, 35m） | 57.89 | 0.9648 | 56.67 | 61.48 (0.8978) | 62.59 / 58.56 | 54.08 | 6.03% (45, 28) | 6.04% (9, 2) | 5.11 / 5.99 |
| **`s325-pf-20um-ematched-v2`**（A100, 55m） | **58.10** | 0.9680 | **57.12** | **62.08** (0.9034) | **63.26** / 59.01 | 54.03 | 5.50% (41, 28) | 5.37% (8, 2) | 5.41 / 5.86 |
| `s325-pf-hybrid-v2`（KF H100-47G, 52m） | 57.92 | 0.9638 | 56.73 | 61.54 (0.8968) | 62.47 / 59.06 | 54.20 | 5.90% (44, 23) | 5.37% (8, 5) | **4.24** / 9.92 |
| `s325-pf-hybrid-v2-shuffled`（A100, 58m） | 58.04 | **0.9615** | 56.84 | 61.98 (0.8958) | 62.96 / 59.33 | 54.21 | 6.43% (48, 28) | 4.70% (7, 2) | 4.54 / 8.78 |
| `s325-pf-hybrid-v2-coord1mm`（A100, 57m） | 57.45 | 0.9666 | 56.30 | 61.32 (0.9031) | 61.93 / **59.69** | 53.66 | 6.30% (47, 23) | 4.70% (7, 1) | 5.13 / 9.64 |
| **`s325-pf-hybrid-v2-nomad`**（A100, 69m） | 57.87 | 0.9643 | 56.67 | 61.54 (0.8982) | 62.80 / 58.28 | 54.41 | **7.77% (58, 24)** | 4.03% (6, 1) | **4.13** / 9.39 |
| `s325-pf-hybrid-v2-reg`（H100, 35m） | 57.92 | 0.9673 | 56.57 | 61.75 (0.9007) | 62.71 / 59.24 | 53.94 | 6.30% (47, 22) | 5.37% (8, 4) | **4.09** / 9.94 |

*注*：严格 `0.1% FPR` 互斥 split-half（Half-A 仅允许 `2` 个本底事例，已标注 `low_bg_warning=True`）下，`s325-pf-20um` 为 `2.01% (15/746, FP=3/2187)`、`s325-pf-hybrid-v2-nomad` 为 `1.88% (14/746, FP=4/2187)`、`s325-pf-20um-ematched-v2` 为 `1.88% (14/746, FP=2/2187)`、`s325-b1pp-clean` 为 `0.80% (6/746, FP=0/2187)`。Pinpoint 子区 `1% FPR` 互斥 split-half（`382` 个 $\tau$、`1,190` 个本底）下，`s325-pf-hybrid-v2-nomad` 达最高 **`9.95% (38/382, FP=16/1190)`**，高于 `s325-pf-20um`（`7.85%`）和 `s325-b1pp-clean`（`5.76%`）。

### 3.5 Tune6k 配对 1,000 次 Family-Bootstrap 检验与 `6T` 自身域重评

| 配对比较（严格排除 `tune_screen` 的 `Tune6k` 原始 3-seed 平均） | $\Delta\text{BA4}$ [pp]（95% CI） | $\Delta\tau\text{@1\% FPR}$ [pp]（95% CI） | $\Delta\tau\text{@0.1\% FPR}$ [pp]（95% CI） |
| --- | ---: | ---: | ---: |
| `PF20um − BDT70d`（Tune6k All） | **+4.57** `[+3.51, +5.65]` | **+2.95** `[+0.99, +4.35]` | +0.54 `[-0.20, +1.71]` |
| `PF20um − B1++`（Tune6k All） | **+1.08** `[+0.32, +1.90]` | **+2.14** `[+0.80, +3.57]` | +0.07 `[-0.53, +1.35]` |
| `PF20um − PF20um_Coord1mm`（Tune6k Pinpoint） | +0.09 `[-0.49, +0.69]` | +0.62 `[-1.07, +1.78]` | +0.50 `[-0.66, +1.12]` |
| `HybridV2 − PF20um`（Tune6k All） | -0.15 `[-0.71, +0.38]` | -0.20 `[-1.46, +0.97]` | 0.00 `[-1.10, +0.60]` |
| `HybridV2 − HybridV2_Coord1mm`（Tune6k Pinpoint, `324-08`） | +0.22 `[-0.53, +0.97]` | -1.75 `[-3.14, +0.13]` | +0.50 `[-1.13, +1.22]` |
| `HybridV2 − HybridV2_Shuffled`（Tune6k All） | -0.12 `[-0.66, +0.43]` | -0.47 `[-1.46, +1.38]` | -0.40 `[-0.93, +0.42]` |
| `HybridV2_NOMAD − HybridV2`（Tune6k All） | -0.05 `[-0.62, +0.55]` | **+1.47** `[+0.06, +2.86]` | +0.27 `[-0.20, +1.33]` |
| `HybridV2_Reg − HybridV2`（Tune6k All, `324-11`） | +0.00 `[-0.49, +0.49]` | +0.33 `[-0.80, +1.50]` | +0.40 `[-0.40, +1.09]` |
| `Natural − EMatchV2`（Tune6k 共同支撑, `324-03`） | +0.04 `[-0.56, +0.74]` | +0.81 `[-0.60, +1.64]` | 0.00 `[-0.80, +0.87]` |

- **Stage 323 `6T` 自身可观测量域重评（`324-08`）**：按 `active_pixel_planes=[0,2,4,6,8,10,12,13,14]` 重新计算 `6T` 自身可观测量域分布为 `D_empty=2, D_pix_contained=1,627, D_scint_contained=4,688, D_partial_or_edge=1,683`；其在 `Tune6k` 原始 3-seed 平均上为 `All BA4 = 57.16% (NLL = 0.9663)`，自身 `D_pix_contained BA4 = 60.69% (NLL = 0.8771)`、`D_scint_contained = 56.82% (0.9852)`、`D_partial_or_edge = 54.25% (1.0003)`。

### 3.6 纯钨 Generator-OOD 与钨-184 EPPS21 核 PDF NC Charm 压力测试（`324-04`）

全部 2,400 条新建 W-184 EPPS21 事件均实测满足 `vtx_mat_name == 'G4_W'`、`tgt_Z == 74`、`tgt_pdg == 1000741840`、`E_med = 500.0 GeV`（`gen_ood_w184_epps21_nc_incl` 自然含 21 条 charm = `1.75%`；`gen_ood_w184_epps21_nc_charm` 从 90,902 次尝试中筛得 1,200 条 charm，自然产率 `1.32%`）：

| 模型组（3-seed FP32） | Run10085-88 BA4 [%] (NLL) | Pythia6-std BA4 [%] | Pythia6-NUX BA4 [%] | W184-EPPS21 NC Incl：Pred-NC / Pred-$\tau$ / $\bar p_\tau$ / $p_\tau>0.5$ | W184-EPPS21 NC Charm：Pred-NC / Pred-$\tau$ / $\bar p_\tau$ / $p_\tau>0.5$ | Free-N NC Charm 诊断：Pred-$\tau$ / $\bar p_\tau$ |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `bdt_histgb_full_obs_70d` | 52.81 (1.0616) | 50.92 | 52.92 | 54.1% / 22.3% / 0.254 / — | 51.0% / 25.4% / 0.271 / — | 30.4% / 0.283 |
| `s325-b1pp-clean` | 56.56 (1.0054) | 55.00 | 56.75 | 49.4% / 24.5% / 0.261 / 2.4% | 43.1% / 26.4% / 0.280 / 3.3% | 34.9% / 0.293 |
| `s325-pf-20um` | 58.69 (0.9817) | 55.08 | 57.42 | 55.8% / 22.2% / 0.262 / 4.3% | 50.8% / 25.4% / 0.286 / 4.7% | 31.4% / 0.291 |
| `s325-pf-20um-ematched-v2` | **59.25** (0.9838) | 55.50 | 58.17 | 53.8% / 22.9% / 0.266 / **3.0%** | 48.6% / 27.0% / 0.292 / **2.8%** | 33.2% / 0.296 |
| `s325-pf-hybrid-v2` | 58.69 (0.9787) | 55.17 | 58.67 | 56.9% / 21.7% / 0.256 / 4.2% | **51.9%** / 24.9% / 0.280 / 4.4% | 30.2% / 0.287 |
| `s325-pf-hybrid-v2-reg` | 57.62 (0.9854) | 55.58 | 57.50 | 56.4% / 21.9% / 0.256 / 3.1% | 50.4% / 25.2% / 0.281 / 3.8% | 31.6% / 0.288 |
| **`s325-pf-hybrid-v2-nomad`** | 58.19 (0.9877) | **56.00** | **58.92** | 56.8% / **20.5%** / **0.248** / 3.8% | 51.2% / **21.2%** / **0.268** / 3.2% | 30.9% / 0.286 |

### 3.7 探测器退化压力测试（`324-10/06`）、PV Pull 标定（`324-13`）与 FP32 纯观测复现闭环（`324-09`）

1. **7 项探测器响应退化压力测试（Shard 080，400 条真实 Tune 事例，`s17` FP32 推理）**：

| 模型 (`s17`) | Nominal BA4 / NLL | PixDead 2% BA4 (TV) | PixDead 5% BA4 (TV) | PixDark $10^{-5}$ BA4 (TV) | TrkDead 2%+N1 BA4 (TV) | TrkDead 5%+N2 BA4 (TV) | SiPM Cap 95% BA4 (TV) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `s325-b1pp-clean` | 54.7% / 1.016 | 53.9% (0.002) | 54.2% (0.004) | 54.5% (0.097) | **47.2%** (0.126) | 50.3% (0.133) | 55.2% (0.013) |
| `s325-pf-20um` | 57.8% / 0.987 | 57.3% (0.006) | 57.1% (0.010) | 57.7% (0.083) | **48.9%** (0.140) | 51.1% (0.144) | 57.6% (0.015) |
| `s325-pf-hybrid-v2` | 55.9% / 0.993 | 56.3% (0.008) | 55.8% (0.013) | 56.4% (0.092) | **50.1%** (0.138) | 51.2% (0.142) | 57.4% (0.018) |
| `s325-pf-hybrid-v2-nomad` | 57.2% / 0.985 | 58.0% (0.007) | 57.7% (0.012) | **52.7%** (0.157) | **48.6%** (0.142) | 52.5% (0.152) | 57.2% (0.025) |

2. **PV 异方差 Pull 标定审计（`324-13`，400 条有效 Tune 事例）**：
   - `v325_pf_20um_s17`：`pull_mean = [-0.34, -0.12, +0.29]`，**`pull_std = [8.51, 5.55, 2.92]`**，`p68(|pull|) = [3.56, 3.51, 2.27]`，`median_sigma = [0.57, 0.63, 5.04] mm`。
   - `v325_pf_hybrid_v2_s17`：`pull_mean = [+0.01, -0.62, +0.25]`，**`pull_std = [8.23, 6.98, 2.98]`**，`p68(|pull|) = [4.23, 3.78, 2.45]`，`median_sigma = [0.61, 0.63, 5.42] mm`。
   - `v325_pf_hybrid_v2_nomad_s17`：`pull_mean = [+0.16, -0.33, +0.31]`，**`pull_std = [7.95, 6.68, 3.07]`**，`p68(|pull|) = [3.95, 4.00, 2.49]`，`median_sigma = [0.62, 0.65, 5.41] mm`。
3. **FP32 落盘预测 vs 无标签纯观测复现闭环（`324-09`，64 条 Shard-080 事例，零 `targets.parquet`）**：
   - 全部 4 个代表模型族（`v325_b1pp_clean_s17`, `v325_pf_20um_s17`, `v325_pf_hybrid_v2_s17`, `v325_pf_hybrid_v2_nomad_s17`）**同卡（Tesla T4 IEEE-FP32）最大绝对误差全部为 `0.000e+00`**；
   - **跨卡（训练端 H100/A100 `allow_tf32=True` FP32 落盘 vs T4 SM75 IEEE-FP32 复现）最大绝对误差分别为 `8.90e-5`、`5.52e-5`、`8.34e-5`、`8.64e-5`**（全部 `< 1e-4`，较 Stage 323 BF16 偏差 `~2.5e-2` 降低约 300 倍）。

### 3.8 产出图表与核心路径

- 综合性能、固定聚类坐标消融、分区域 PV 与 OOD/W-184 NC Charm 汇总图（300-DPI 英文）：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v7_stage325_2026-11-20/stage325_remediation_summary.png`
- 7 项探测器响应退化压力测试图（300-DPI 英文）：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v7_stage325_2026-11-20/stage325_detector_stress_tests.png`
- 冻结清单（EOS + 仓库）：`roadmap/v3/v9_stage325_freeze_manifest.json`、`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v7_stage325_2026-11-20/v9_stage325_freeze_manifest.json`
- 非神经网络冻结基线：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v7_stage325_2026-11-20/frozen_baselines/` 与 `stage325_frozen_baselines.json`
- W-184 EPPS21 OOD 分片：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v7_stage325_2026-11-20/ood_shards/gen_ood_w184_epps21_nc_{incl,charm}/`
- **未测量项目（按指令暂停）**：`Cal-A / Cut-B / Confirm` 盲态数据上的任何推理、校准、似然拟合、显著性、谱形系统误差与物理分析项（`322-02/04/06/08/10/14`）均严格为**未测量**。

---

## 4. 阶段性结论

### 4.1 数据直接证实的事实

1. **审计 15 项训练与评估缺陷已全部工程与实验闭环**：零封存访问（`0` 次触碰封存分片 `100..149`）、真实 128-事件微批梯度精确等价（`< 1.5e-6`）、单提交比特一致源码包（`302dc79d27...`）、FP32 落盘与无标签同卡 `0.0` / 跨卡 `< 9e-5` 复现、7/7 严格钨靶门禁、像素暗噪声与 $\tau$ V-A 螺旋度实时闭环、以及 2,400 条纯钨 W-184 EPPS21 NC OOD 验证集均已实测通过。
2. **神经架构对非神经网络基线和 B1++ 的增益在严格 `Tune6k` 上统计显著**：`s325-pf-20um` 超越冻结 `BDT-70d` 达 **`+4.57 pp` BA4**（`[+3.51, +5.65] pp`）与 **`+2.95 pp` @ 1% FPR**（`[+0.99, +4.35] pp`），超越 `s325-b1pp-clean` 达 **`+1.08 pp` BA4**（`[+0.32, +1.90] pp`）与 **`+2.14 pp` @ 1% FPR**（`[+0.80, +3.57] pp`）。
3. **能量谱捷径并非性能主因（`324-03`）**：全四任务严格共同支撑等权训练的 `s325-pf-20um-ematched-v2` 取得 `Tune6k BA4 = 58.10%`（自然分布 `58.07%`）、$(E_\nu, \nu/\bar\nu)$ 匹配 `BA4 = 57.12%`（自然分布 `56.79%`）及 `Run10085-88 OOD BA4 = 59.25%`（9 组最高）。
4. **孔径质控消除了 35.6% 的穿壳谱仪污染，并与 NOMAD 协同改善 $\tau$ 尾部与 W-184 NC Charm 抑制**：实时模拟证实穿壳 100 GeV $\mu^+$ 原始错号率高达 `55.0%`，孔径质控实现 `60/60` 穿壳拦截与 `60/60` 孔内保留；在 CleanTune8k 中标记了 `912/2,561` 条穿壳 3-station 候选。`s325-pf-hybrid-v2-nomad` 相对 `s325-pf-hybrid-v2` 在 `Tune6k` 取得统计显著的 **`+1.47 pp` @ 1% FPR**（`[+0.06, +2.86] pp`），实现全组最高的 `1% FPR` 互斥 split-half $\tau$ 效率（All `7.77%`、Pinpoint `9.95%`）及最低的 W-184 EPPS21 NC Charm 误判 $\tau$ 率（`21.2%`，较 `PF-20um` 的 `25.4%` 降 `4.2 pp`）。
5. **固定聚类 1 mm 纯坐标消融与全靶区微顶点重构揭示了明确的功能边界（`324-07/08`）**：
   - `Hybrid-v2` 在 **前部 30 mm Pinpoint 块** 显著提升神经主顶点分辨率（`|\Delta z|: 5.15 → 4.24 mm`，3D `6.93 → 5.00 mm`，且将其簇坐标量化到 1 mm 后退化回 `5.13 mm / 6.02 mm`）；
   - 但在 **4 分类 `BA4` 与 `1% FPR`** 上，当前 16 维标量微顶点特征并未带来相对 `PF-20um`（`ΔBA4 = -0.15 pp`）或 `Hybrid-v2-shuffled`（`ΔBA4 = -0.12 pp`）的统计显著分类提升；固定聚类 1 mm 坐标量化在 `PF-20um` 和 `Hybrid-v2` 上的分类变化也均在 95% CI 零点内（`+0.09 pp` 与 `+0.22 pp`）；
   - 在 **中间 150 mm Pinpoint 块**，尽管修复后 C++ AVR PV 有效率从 `0.0%` 升至 `65.15%`，但因其 150 mm 间距固有精度较粗（中位 `41.29 mm`），当前架构中 `avr_pv` 硬覆盖 `cal_start` 先验反而使 `Hybrid-v2` 的中间区神经 PV 精度从 `PF-20um` 的 `5.81 mm` 退化至 `9.92 mm`。
6. **新发现的两项具体脆弱性（`324-10`, `324-13`）**：
   - 退化压力测试表明，全部模型对像素坏点（2–5%）与 SiPM 饱和上限（95%）稳健（`TV ≤ 0.025`），但对**谱仪径迹站坏点+噪声击中**（`TrkD2%+N1`）均出现 `-5.7` 至 `-8.9 pp` 的 `BA4` 下降（`TV = 0.126–0.152`），且 `Hybrid-v2+NOMAD` 对像素暗噪声 `1e-5` 较敏感（`BA4: 57.2% → 52.7%`, `TV = 0.157`）。
   - PV 异方差不确定度严重低估重尾残差（`pull_std ≈ [7.9–8.5, 5.6–7.0, 2.9–3.1]`，`p68(|pull|) ≈ [3.5–4.2, 2.3–2.5]`）。

### 4.2 待后续实验验证的假设

1. **假设 A（不确定度加权 PV 先验融合）**：在 `PinpointFormer.forward_features()` 中将硬替换 `v_prior = avr_pv` 改为按间距/径迹段质量门控的软融合（仅在前部 30 mm 密集区启用或与 `cal_start` 做逆方差加权），有望同时保留前部 `4.2 mm` 与中间区 `5.8 mm` 的最优 PV 分辨率。
2. **假设 B（噪声增强训练与 Student-t PV 损失）**：在训练期以小概率注入谱仪随机噪声击中、像素暗噪声，并将 PV 异方差损失改为 Student-t 负对数似然（或在验证集上做分域方差标定），有望同步修复谱仪噪声敏感性与 PV Pull 过窄问题。
3. **假设 C（非标量微顶点表示）**：20 μm 分辨率当前已显著改善前部 PV 定位但尚未转化为 4 分类增益，可能源于 16 维手工标量汇总损失了径迹级冲撞参数与次级顶点拓扑信息，将每条 3-hit 微径迹段作为显式 token 输入注意力层（而非压缩为 16 维标量）是测试 20 μm 物理分辨上限的合理方向。

---

## 5. 下一步计划

1. **模型与前端鲁棒性改进（仍保持 `Cal-A / Cut-B / Confirm` 严格封存）**：
   - 将 `avr_pv` 先验由硬覆盖改为按块间距/不确定度门控融合，验收标准：`FrontPinpoint_30mm` 保持 `≤ 4.3 mm` 同时 `IntermPinpoint_150mm` 恢复至 `≤ 5.9 mm`。
   - 评估谱仪噪声增强训练与暗噪声过滤对 `TrkD2%+N1` / `PixDark 1e-5` 退化鲁棒性的改善，验收标准：噪声下 `BA4` 降幅缩减至 `< 3 pp` 且标称 `Tune6k BA4` 不降。
   - 在验证集对 PV 不确定度实施 Student-t 损失或分域标定，并统一在最终 FP32 评估关闭 `allow_tf32`（跨卡复现误差从 `< 9e-5` 降至 `< 1e-6`），验收标准：三轴 `p68(|pull|) ∈ [0.85, 1.15]`。
2. **恢复暂停的物理分析阶段（待用户确认解冻指令后启动）**：
   - 仅在用户明确解除物理分析暂停、且完成上述模型冻结定版后，按严格盲态协议推进 `322-02/04/06/08/10/14`（冻结选型 → `Cal-A` 校准 → `Cut-B` 锁阈 → `Confirm` 单次解盲与物理系统误差传递）。
