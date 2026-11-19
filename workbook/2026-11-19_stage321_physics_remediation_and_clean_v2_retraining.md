# 2026-11-19 Stage 321：全面修正 Stage 320 探测器物理与统计缺陷、Clean v2 重产重训及探测器设计结论重构前后对照

## 1. 阶段目标

针对 Stage 320 独立物理、机器学习与统计审计（`workbook/2026-11-19_stage320_independent_pinpoint_physics_ml_and_statistical_audit.md`）查出的全部可定位物理、几何、读出、光学、优化与剖面似然缺陷，以 `/eos/home-x/xcheng/FASER/Pinpoint/Pinpoint_G4` 为唯一探测器物理权威，确保模型输入严格仅包含实验可观测量（零 MC 真值），完成端到端重构、重产、重训与一次性确定性 FP32 解封，回答三个核心问题：

1. **物理与数值闭合**：所有 P1/P2/P3 缺陷（108 块交错钨板真实模块抽样、SiPM 单次饱和、剔除 `edep_proxy_MeV`、batch 独立注意力熵、Cal-A 解析梯度与成员置换不变性、Magnet0 解析 $C^1$ 反推与谱仪动量定标、多粲独立 `vg`、真实上游反符合 `PixelSensor[0]`、仅量化 12 层靶硅而保持 3 层 IPT 原生分辨率、连续能谱 8 通道 Pythia8 NC 粲、逐过程全 27 箱质量加权 PL 与独立加性外部本底）是否在单测、活体 Geant4 与 Clean v2 数据集审计中全部闭合？
2. **Stage 320 实验 1 因果证伪**：在真实 24 块 PinpointBlock 钨板与单次 SiPM 饱和下，显式微顶点分支（`s321-pf-hybrid-uvtx`）相对纯基线（`s321-pf-20um`）和**同层打乱跨面 tracklet 关联的证伪对照**（`s321-pf-hybrid-shuffled`）是否能通过 Stage 320 预注册证伪门禁（95% CI 下界 $>0$、点增益 $\ge 1\text{ pp}$、打乱后增益消失）？
3. **重构前后探测器设计结论变化**：修复上述物理缺陷后，Stage 319 的各项硬件设计结论（靶硅像素螺距 $20.8/100/300\ \mu\text{m}/1\text{ mm}$、靶硅层数 $12\text{T}+3\text{IPT}$ vs $6\text{T}+3\text{IPT}$、机械对准公差、Magnet0-NOMAD 谱仪运动学、上游反符合设计及质量加权 2t 灵敏度）哪些维持、哪些被推翻或修正？

---

## 2. 工作内容

### 2.1 全部物理、探测器、ML 与统计修复清单（对应 Stage 320 审计项）

全部代码修复在提交 `9676578b4224d8f9eb9c7624e1a1e118f059cb3b` 与 `357c66fa19b50458a3f691e3c6a9b1d791b1abd8` 完成（位于解封前冻结清单 `v7_freeze_manifest.json` 之前）：

| 编号 | Stage 320 缺陷 | 修复实现与源码位置 |
|---|---|---|
| F1 (P1) | 钨板抽样误将 `0..23` 当作 Pinpoint 靶板，CleanConfirm 像素中心切片中 `3,479/6,697` 实位于 FortuneBlock | `src/analysis/flavour/v3/common/domains.py` (`pinpoint_slab_catalog`) 严格按 `Pinpoint_G4/Pinpoint/src/DetectorConstruction.cc` 导出 108 块 5 mm 钨板精确物理归属：**PinpointBlock 24 块 = `0..11, 24..25, 38..39, 52..53, 66..67, 80..81, 94..95`**（`430×430 mm`，中心 `(+82, -7) mm`），FortuneBlock 84 块为其余索引；`produce_clean_sim.py` 实现三层物理体积分层抽样（40% FrontPinpoint×PixXY、15% IntermPinpoint×PixXY、45% 剩余 24 块外区 + 84 块 Fortune 全区）并记录精确解析权重 `geom_weight`（3D 质量加权后 PinpointBlock = 22.22%，FortuneBlock = 77.78%） |
| F2 (P1) | `ScintSD.cc` 已施加 14,400 微单元 SiPM 饱和，`domains.py` 再次压缩至 9,102.54 PE | `calibrated_bar_channels` 默认 `apply_sipm_saturation=False` + `invert_sipm_saturation=True`（由单次饱和 PE 经可部署反饱和映射计算能量代理，上限钳位防奇异）；`readout.py` 设 `include_truth_edep=False`，观测表彻底剔除 `edep_proxy_MeV`，保留 `hit_time_ns` |
| F3 (P1) | `ScaledQKNormAttention` 熵计算存在 `(B,B,H,Q)` batch 轴广播错误（零像素桶均值 2.2704） | `src/analysis/flavour/v3/common/layers.py` 与 `v2_pinpointformer/attention.py` 将 `denom` 改为 `(B,1,1)`、屏蔽有效 key $<2$ 为 `0.0` 并支持 `query_mask`；单条与混批完全一致 |
| F4 (P1) | `ensemble.py` 校准正则目标（减 `max`）与解析梯度（减均值）不一致、偏置非对称导致成员顺序敏感 | `src/analysis/flavour/v3/v4_tau/ensemble.py` 将正则项统一为均值中心 L2、对称零和类偏置 $b_3 = -(b_0+b_1+b_2)$ 并增加置换等变多起点 L-BFGS-B；解封脚本增加显式反转/轮换成员置换不变性门禁并落盘 `cal_a_frozen_params.json` / `cut_b_locked_params.json` |
| F5 (P1) | 谱仪 `theta_y_entry` 存在符号/3× 系数误差，且缺失 `Magnet0` (`z ∈ [-1565.3, -65.3] mm`) 磁场弯曲反推 | `src/analysis/flavour/v3/v1_b1pp/spectrometer.py` 与 `v4_tau/nomad_kinematics.py` 修正 `theta_y_entry = (y1-y0)/Δ + (60/59)(s_y/Δ)`、精确 `SAGITTA_CONST_MEV_MM = 100820.18`（$B=0.56\text{ T}$）及分段解析 $C^1$ `Magnet0` 反推 |
| F6 (P1) | `quantize_xy_mm` 同时劣化 3 层 IPT 谱仪平面；`train.py` 读默认几何而非 `plan.json` | `pixel_tokenizer.py` 与 `group.py` 新增 `quantize_target_only=True`（仅量化 12 层靶硅 `planes 0..11`，保持 3 层 IPT `planes 12..14` 原生 `20.8×22.8 μm`）；新增 `shuffle_micro_vertex` 实现 Exp-1 证伪对照（平面内绕质心随机旋转簇坐标，保留每面多重度/质心/径向展宽而破坏跨面直线 tracklet 关联）；`train.py` 修复从 `data_root/plan.json` 读取物理几何；`_fast_kernels.cpp` 修正 `dz_back = 17.505 mm` 与中间 PinpointBlock 逻辑 |
| F7 (P1) | 多粲强子共享同一 `vg=2` 组号，无法区分两个独立次级顶点 | `truth_sidecar.py` 与 `dense_heads.py` 改用 `20000 + parent_trackID` 为不同粲强子赋予独立衰变顶点 ID，未解析源标记 `-1` 并在损失中屏蔽 |
| F8 (P1) | Pythia8 NC 粲仅离散能点、3 种通道且写入伪造 `Q2=10, W=5, x=0.05` | `produce_clean_sim.py` 升级为按真实多 TeV 能谱连续抽样的 8 通道（$\nu_e,\bar\nu_e,\nu_\mu,\bar\nu_\mu \times p,n$）Pythia8 NC 含粲生成，记录真实 $(Q^2, W, x, y)$，并计算离线 $\tau$/charm 飞行距离 `L3d_mm` 与 `dz_mm` |
| F9 (P1) | 外部穿入反符合误用位于 `W0` 下游的 `panel <= 1`；27-bin PL 缺失 3D 质量权重、混用 `b_nue_charm`、`tau_helicity_0` 错施用于本底、6 CR 形变填零、外部中性强子无独立加性产额、Gamma(1/2,1) 方差少因子 2 | 真实上游反符合修正为 `PixelSensor[0]`（`z=-2991.995 mm < W0=-2984.49 mm`）命中数 $\ge 1$；`flux_significance.py` 分离 `b_nue_light`/`b_nue_charm`，引入 `event_weights = geom_weight` 与加权 BB-lite $\sum w_i^2$，精确 `2*b_jeff^2` 方差，独立加性 `b_ext`（2% fiducial 产额 + ±50% 归一 nuisance `theta_ext_neutral_hadron`），逐过程信号/本底有符号形变覆盖全部 27 SR+CR 箱（`tau_helicity_0` 仅作用于 $\nu_\tau$ 信号），`significance.py` 修正 uniform 箱独立 RMS 向量与 Cut-B 锁定分箱 |
| F10 (P2/P3) | Cut-B 阈值依赖 `Truth_pixel_centered`；配置 `batch_size=128` 与 `4k/6k` 分区规模记录核对 | 全部 Cut-B 阈值严格在**纯可观测域**（`D_pix_contained` / `D_scint_contained` / `D_partial_or_edge` / `All_Active` / `MassWeighted_2t_Active`）锁定，不接触任何真值顶点切片；所有配置统一显式记录 `batch_size=128` 与 `32k/8k/4k/6k/10k` 分区 |

### 2.2 Clean v2 仿真生产、9 组 27 模型多 GPU 训练与一次性 FP32 解封

1. **Clean v2 Geant4 生产**（输出根目录 `/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v5_clean_v2_2026-11-19/clean_sim`，完整保留 Stage 318/319 历史目录 `v4_audit_2026-11-18/clean_sim` 未改动）：
   - 经 HTCondor 最空闲 schedd `bigbird20.cern.ch`（Cluster `1584832`）提交 80×8 核并行作业，约 21 分钟完成 150 个主分片共 **60,000** 事例（`train 32,000 / validation 8,000 / cal_a 4,000 / cut_b 6,000 / confirm 10,000`），并额外生产 5 组 1,200 事例配对 Geant4 系统误差（`nominal_sys_ref`、`tau_helicity_0`、`w_density_0p98`、`no_magnet_shells`、`field_0p97`）、1,000 事例外部穿入样本（350 像素面内岩石 $\mu$ + 150 外框岩石 $\mu$ + 500 中性强子 $n/K_L^0$）及 2×800 事例匹配生成器强子化对照（`pythia6_std_ref` vs `pythia6_nux`），总计 **68,600** 事例，EOS 零 `.root` 残留。
2. **9 组共 27 个模型并行 GPU 训练**（与 Stage 319 形成 1-to-1 探测器设计可比性，含新增 `300 μm` 靶硅组和 `6T+3IPT` 减层组，每组 3 种子 `17, 42, 2026`，20 passes，`batch_size=128`，单卡并行 `--parallel 2/3` 拉满 GPU 利用率）：
   - Kubeflow `nvidia.com/mig-3g.47gb` (`H100-47GB`, `--parallel 3`)：`s321-pf-hybrid-nomad`、`s321-pf-1mm`、`s321-pf-layers-6t`。
   - HTCondor `A100-40GB` / `H100 NVL 95GB` / `H100-MIG-24GB`（`cpus=4, mem=20GB, allow_backfill=True, job_flavour='longlunch'`）：`s321-pf-20um`（A100 `--parallel 3`）、`s321-pf-100um`（A100 `--parallel 3`）、`s321-pf-300um`（H100-95GB `--parallel 3`）、`s321-pf-hybrid-uvtx`（MIG-24GB `--parallel 2`）、`s321-pf-hybrid-shuffled`（A100 `--parallel 2`）、`s321-b1pp-clean`（MIG-24GB `--parallel 3`）。
   - 全部 27/27 个检查点训练完成且损失全程有限，训练期间 `Cal_A_Cut_B_Confirm_payload_read = False`。
3. **预解封冻结与一次性确定性 FP32 解封**：
   - 仅使用 `CleanTune8k` 选定主 PF 组为 `s321-pf-hybrid-uvtx`，落盘 27 个检查点 SHA-256 与 40 位预解封 Git commit `357c66fa19b50458a3f691e3c6a9b1d791b1abd8` 至 `v7_freeze_manifest.json`（同时归档于仓库 `roadmap/v3/v7_freeze_manifest.json`）。
   - 在本地 `lxplus-gpu` Tesla T4 上执行一次性确定性 **FP32** 解封（`wall_seconds = 5,682.3 s`，零 autocast 漂移），产出全部 9 组及 6-model 双族集成评估、Probe A、OOD 量化曲线、机械对准扫描、5 组 G4 系统误差、生成器系统误差、外部穿入审计、质量加权 2t 27-bin PL 及 1,000 次 $H_0$ toys（含 BB-lite 与纯物理系统误差两套），并保存不可变原始报告 `stage321_clean_v2_unsealing_report.json` 与两张 300-DPI 英文图。
4. **解封后只读物理诊断**：
   - 在不改动任何冻结模型、Cal-A/Cut-B 参数或原始解封报告的前提下，直接从缓存的 FP32 预测运行只读脚本产出 `stage321_post_unsealing_diagnostics.json`，排查出跨观测域混用不同 `lam_by_dom` 对 `tau_flight_length_diagnostics` 和全域 `s_tau` 造成的域间平移效应，并补全纯校准 `p_tau` 配对 bootstrap、能量控制（$E_\nu > 300\text{ GeV}$）飞行距离分层、精确 TP/FP/charm 计数及 `PixelSensor[0]` 逐板（`slab 0..11`）背散射剖面。

---

## 3. 详细数据

### 3.1 软件单测、活体 Geant4 物理闭合与 Clean v2 数据集审计

- **自动化单测**：`source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh && PYTHONPATH="$PWD/src:$PYTHONPATH" python3 -m unittest discover -s tests` → **352 / 352 PASSED**（`142.8 s`），其中新增 `tests/test_stage321_physics_fixes.py` 验证了：校准解析梯度与中心有限差分最大偏差 $<10^{-8}$、全部 $4!=24$ 种成员置换概率最大差 $<10^{-8}$、batch 独立熵在 `[2, 8]` 均匀注意力下返回精确 `[1, 1]`（空/单 key 返回 `0`）、108 块钨板质量加权闭合、单次 SiPM 饱和、Magnet0 反推、仅靶区量化、`PixelSensor[0] >= 1` 反符合及升级版 27-bin PL。
- **活体 Geant4 (`pinpoint_fixed`) 闭合实测**：
  - 15–1,000 GeV 单 $\mu^\pm$ 动量闭合误差：`0.09% – 1.34%`；`Magnet0` 解析 $C^1$ 反推至靶区出口残差 **`0.000 – 0.051 mm`**（对比修复前线性外推在 15 GeV 存在 **`±12.9 mm`** 系统偏移）。
  - 50–1,000 GeV 外入岩石 $\mu$ 在 `PixelSensor[0]` (`z = -2991.995 mm < W0`) 上留下 **1 个像素击中** 的比例为 **30/30 (`100.0%`)**，而旧阈值 `p0_hits >= 10` 为 **0/30 (`0.0%`)**。
- **Clean v2 数据集审计**（`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v5_clean_v2_2026-11-19/clean_sim/stage321_clean_v2_dataset_audit.json`）：
  - `total_events = 60,000`，`family_disjoint_verified = true`。
  - `edep_proxy_MeV_absent_verified = true`；条通道键仅包含 `['bar', 'bounds_mm', 'channel_id', 'hit_time_ns', 'horizontal', 'layer', 'panel', 'pe_proxy']`。
  - 3D 钨板几何包容：`outside_z_count = 0`，`outside_xy_count = 0`，`max_abs_dz_from_slab_center_mm = 2.44996 mm`（精确小于半厚 `2.5 mm`）。
  - 真实模块组成：原始分层 `FrontPinpoint×PixXY = 24,046`、`IntermPinpoint×PixXY = 9,067`、`PinpointBlock 全部 = 37,665`、`FortuneBlock 全部 = 22,335`；施加 `geom_weight` 后质量加权比例为 **PinpointBlock = `22.16%`**（理论精确值 `24/108 = 22.22%`）、**FortuneBlock = `77.84%`**（理论精确值 `77.78%`），`mean_geom_weight = 0.9964`。
  - 单次 SiPM 饱和（分片 080，`539,170` 条通道）：最大观测值 **`14,400.0 PE`**，有 **`24,110` 个通道落在 `(9103, 14400]` PE**（原二次饱和错误下全部被错误压缩至 `<= 9,102.54 PE`），`>= 2.5 PE` 比例 `96.36%`。
  - 连续能谱 Pythia8 NC 粲：`2,400` 事例覆盖 **`276` 个唯一入射能量**（`E_min = 63.1 GeV`，`E_median = 495.2 GeV`，`E_max = 3,358.1 GeV`）。
  - 审计图：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v5_clean_v2_2026-11-19/clean_sim/stage321_clean_v2_physics_audit.png`。

### 3.2 CleanTune8k 冻结前选型与 CleanConfirm v2 确定性 FP32 全组实测

**执行命令**：
```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh >/dev/null 2>&1
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python3 -m analysis.flavour.v3.v6_clean_sim.run_stage321_clean_v2_campaign freeze \
  --clean-root /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v5_clean_v2_2026-11-19/clean_sim
python3 -u -m analysis.flavour.v3.v6_clean_sim.run_stage321_clean_v2_campaign unseal-and-evaluate \
  --clean-root /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v5_clean_v2_2026-11-19/clean_sim --n-toys 1000
```

**CleanConfirm v2（10,000 事例，四类各 2,500）切片与观测域交叉计数**：
- `All`: `10,000`（$\tau=2,500$）= `D_empty 3` + `D_pix 2,311` + `D_scint 5,590` + `D_partial 2,096`
- `PinpointBlock_PixXY_All`（全部真实 24 块 Pinpoint 钨板且在像素 XY 内）: **`5,491`**（$\tau=1,359$，本底 $4,132$）= `D_empty 2` + `D_pix 2,112` + `D_scint 2,097` + `D_partial 1,280`
- `FrontPinpoint_PixXY`（前 12 块 `0..11`）: **`4,021`**（$\tau=972$）= `D_empty 2` + `D_pix 1,821` + `D_scint 1,013` + `D_partial 1,185`
- `IntermPinpoint_PixXY`（6 组交错块 `24-25..94-95`）: **`1,470`**（$\tau=387$）= `D_pix 291` + `D_scint 1,084` + `D_partial 95`
- `FortuneBlock_All`（84 块无近邻硅钨板）: **`3,717`**（$\tau=955$）= `D_empty 1` + `D_pix 197` + `D_scint 2,876` + `D_partial 643`

**Cal-A 置换不变性验证门禁（FP32）**：
- `s321-pf-hybrid-uvtx`（3 种子）：`max_abs_prob_diff = 3.164 × 10⁻¹⁵`，`argmax_flips = 0`，`passed = true`（每域 `grad_inf_norm <= 3.85 × 10⁻⁹`）。
- `primary_dual_family_6m`（3 PF + 3 B1++）：`max_abs_prob_diff = 2.806 × 10⁻⁸`，`argmax_flips = 0`，`passed = true`（每域 `grad_inf_norm <= 2.91 × 10⁻⁸`）。

**表 1：全部 9 个训练组 + 主 6-Model 双族集成在 CleanConfirm v2 (FP32) 上的核心分类、$\tau$@1% FPR 与主顶点精度**

> 注：因部分模型在 Cut-B 上为不同观测域选出不同的 $\lambda_{\text{charm}}$（`s321-pf-hybrid-uvtx` 为 `{D1:0, D2:1.0, D3:0.5}`、`s321-pf-hybrid-shuffled` 为 `{D1:0.5, D2:0, D3:0}`、`s321-pf-1mm` 为 `{D1:1.0, D2:0, D3:0}`、`primary_dual_family_6m` 为 `{D1:0, D2:1.0, D3:0}`，其余 5 组全为 `0.0`），直接把分域 $S_\tau$ 拼在一起在跨观测域切片上排序会引入域间零点平移（$-0.9$）。单观测域 `Obs_D_pix_contained` 及 27-bin PL（分域定边）不受此影响；跨观测域切片在下表同时列出**纯校准 $p_\tau$（跨域同尺度）**与**原始报告 $S_\tau$** 两栏。

| 模型组（3 种子融合） | Tune8k Pin $\tau$@1% | Confirm `PinpointAll` BA4 / condBA3 / NLL | `PinpointAll` $\tau$@1% ($p_\tau$ / $S_\tau$) | `FrontPin` $\tau$@1% ($p_\tau$ / $S_\tau$) | `Obs_D_pix` $\tau$@1% (Cut-B 锁定 TP/FP) | `MW 2t` BA4 / $\tau$@1% ($p_\tau$ / $S_\tau$) | `PinpointAll` 主顶点中位 $\|\Delta x\|/\|\Delta y\|/\|\Delta z\|$ / 3D [mm] |
|---|---:|---:|---:|---:|---:|---:|---:|
| `s321-pf-20um` (12T+3IPT) | 5.48% | 62.53% / 77.60% / 0.8908 | 4.93% / 4.93% | 5.14% / 5.14% | 5.35% (`36/579`, `17/1732=0.98%`) | 53.78% / 5.19% / 5.19% | 0.78 / 0.79 / 4.96 / 6.36 |
| `s321-pf-100um` (仅靶区) | 7.55% | 62.35% / 77.62% / 0.8887 | 6.40% / 6.40% | 5.86% / 5.86% | 6.91% (`38/579`, `16/1732=0.92%`) | 54.60% / 5.78% / 5.78% | 0.80 / 0.80 / 5.23 / 6.64 |
| `s321-pf-300um` (仅靶区) | 6.20% | 62.64% / 77.53% / 0.8903 | 5.15% / 5.15% | 5.04% / 5.04% | 5.70% (`33/579`, `16/1732=0.92%`) | 53.97% / 4.76% / 4.76% | 0.74 / 0.75 / 4.51 / 5.86 |
| `s321-pf-1mm` (仅靶区) | 8.54% | 62.49% / 77.04% / 0.8948 | 6.55% / 6.11% | 5.56% / 5.35% | 7.60% (`49/579`, `19/1732=1.10%`) | 54.14% / 5.53% / 5.83% | 0.81 / 0.85 / 4.74 / 6.10 |
| `s321-pf-layers-6t` (6T+3IPT) | 6.65% | 62.02% / 76.75% / 0.8965 | 6.25% / 6.25% | 6.58% / 6.58% | 6.74% (`39/579`, `19/1732=1.10%`) | 54.58% / 5.88% / 5.88% | **0.91 / 0.92 / 6.92 / 8.47** |
| **`s321-pf-hybrid-uvtx`** (主PF) | 6.92% | 62.33% / 77.74% / 0.8876 | 6.48% / 5.81% | 6.38% / 6.28% | 7.25% (`40/579`, `16/1732=0.92%`) | 54.11% / 5.09% / 5.10% | **0.63 / 0.61 / 4.71 / 5.90** |
| `s321-pf-hybrid-shuffled` (证伪) | 6.20% | 62.08% / 77.67% / 0.8929 | 5.74% / 5.96% | 5.97% / 5.66% | 7.77% (`46/579`, `21/1732=1.21%`) | 54.51% / 4.90% / 4.69% | 0.68 / 0.64 / 4.84 / 5.97 |
| `s321-pf-hybrid-nomad` (M0修正) | 6.74% | 62.21% / 77.60% / 0.8865 | **6.62% / 6.62%** | **7.10% / 7.10%** | 7.60% (`45/579`, `19/1732=1.10%`) | 53.81% / 4.46% / 4.46% | 0.66 / 0.63 / 4.96 / 6.14 |
| `s321-b1pp-clean` (轻基线) | 4.85% | 60.78% / 76.34% / 0.9189 | 3.75% / 3.75% | 3.60% / 3.60% | 3.80% (`32/579`, `26/1732=1.50%`) | 52.66% / 3.84% / 3.84% | — |
| `primary_dual_family_6m` | — | 62.42% / 77.70% / 0.8873 | 6.25% / 5.81% | 6.07% / 5.76% | 7.08% | 54.20% / — / 4.29% | — |

**表 2：关键受控对照的配对 Family Bootstrap 95% 置信区间（CleanConfirm v2，1,000 次重采样）**

| 对照实验 | 切片 | 原始 $S_\tau$ $\Delta\tau$@1% [95% CI] (pp) | 纯校准 $p_\tau$ $\Delta\tau$@1% [95% CI] (pp) | $\Delta\text{BA4}$ [95% CI] (pp) |
|---|---|---:|---:|---:|
| Exp-1：`Hybrid` − `Vanilla 20.8 μm` | `PinpointAll` | `+0.88 [-1.25, +2.26]` | `+1.55 [-0.36, +3.09]` | `-0.20 [-0.82, +0.40]` |
| **Exp-1 证伪**：`Hybrid` − `Hybrid-Shuffled` | `PinpointAll` | **`-0.15 [-2.22, +2.17]`** | **`+0.74 [-1.25, +2.28]`** | `+0.25 [-0.39, +0.82]` |
| **Exp-1 证伪**：`Hybrid` − `Hybrid-Shuffled` | `FrontPinpoint` / `Obs_D_pix` | — | `+0.41 [-1.59, +2.71]` / `-0.35 [-2.72, +2.22]` | `+0.21 [-0.48, +0.98]` / `+0.24 [-0.68, +1.17]` |
| Magnet0-NOMAD：`Hybrid+NOMAD` − `Vanilla 20.8 μm` | `FrontPinpoint` | — | `+1.95 [-0.59, +3.53]` | `+0.34 [-0.52, +1.19]` |
| Magnet0-NOMAD：`Hybrid+NOMAD` − `Hybrid` | `PinpointAll` | `+0.81 [-0.89, +2.26]` | `+0.15` (Front `+0.72 [-2.34, +2.43]`) | `-0.12 [-0.83, +0.60]` |
| Exp-2 仅靶区：`20.8 μm` − `100 μm` | `PinpointAll` | `-1.47 [-2.87, +0.73]` | `-1.47 [-2.87, +0.73]` | `+0.18 [-0.52, +0.83]` |
| Exp-2 仅靶区：`20.8 μm` − `300 μm` | `PinpointAll` | `-0.22 [-1.43, +1.72]` | `-0.22 [-1.43, +1.72]` | `-0.11 [-0.84, +0.55]` |
| Exp-2 仅靶区：`20.8 μm` − `1.0 mm` | `PinpointAll` | `-1.18 [-2.88, +1.06]` | `-1.62 [-3.54, +0.63]` | **`+0.04 [-0.71, +0.77]`** |
| Exp-2 仅靶区：`20.8 μm` − `1.0 mm` | `FrontPinpoint` | `-0.21 [-2.75, +1.69]` | `-0.41 [-3.33, +1.24]` | `-0.06 [-0.96, +0.84]` |
| Exp-2 层数：`12T+3IPT` − `6T+3IPT` | `PinpointAll` | `-1.32 [-2.76, +0.67]` | `-1.32 [-2.76, +0.67]` | `+0.51 [-0.22, +1.15]` |
| Exp-2 层数：`12T+3IPT` − `6T+3IPT` | `FrontPinpoint` | `-1.44 [-3.36, +0.84]` | `-1.44 [-3.36, +0.84]` | `+0.40 [-0.48, +1.31]` |
| 双族集成：`Dual-6m` − `Primary PF` | `PinpointAll` | `0.00 [-1.59, +0.90]` | — | `+0.09 [-0.15, +0.38]` |

- **背景组成精确计数**（`Obs_D_pix_contained` Cut-B 锁定 1% 档，总背景 `1,732` = `92` NC charm + `130` CC charm + `1,510` 轻本底）：
  - `s321-pf-hybrid-uvtx`：`FP = 16/1,732 (0.92%)` = **3 NC charm (`3.26%`) + 1 CC charm (`0.77%`) + 12 light (`0.79%`)**。虽 NC charm 单事例假率高于轻本底约 4.1×，原始通过背景中轻本底仍占 `12/16 = 75%`。

### 3.3 能量控制 $\tau$ 飞行距离诊断、OOD 量化、机械对准与 Probe A

1. **等 1% FPR 纯 $p_\tau$ 的 $\tau$ 飞行距离 $L_{3\text{D}}$ 分层**（`stage321_post_unsealing_diagnostics.json`，修正原始 JSON 跨域 `lam_by_dom` 阈值失配后）：
   - `PinpointBlock_PixXY_All` 中位中微子能量随 $L_{3\text{D}}$ 强烈正相关：`0–1 mm (n=156)` `327.5 GeV`、`1–3 mm (n=184)` `362.6 GeV`、`3–10 mm (n=360)` `522.5 GeV`、`>10 mm (n=659)` **`914.1 GeV`**。
   - **即使控制能量 $E_\nu > 300\text{ GeV}$**，在 `PinpointBlock_PixXY_All`（纯 $p_\tau$ 精确 1% FPR）上：
     - `v2_pf_pitch_20um`：$L_{3\text{D}} < 10\text{ mm}\ (n=452)$ **`3.98%`** vs $L_{3\text{D}} \ge 10\text{ mm}\ (n=613)$ **`8.32%`**；
     - `v2_pf_hybrid_uvtx`：**`4.20%`** vs **`11.26%`**；
     - `v2_pf_hybrid_shuffled`：**`5.09%`** vs **`9.30%`**；
     - `v2_pf_hybrid_nomad`：**`5.53%`** vs **`10.60%`**；
     - **`v2_pf_pitch_1mm_tgtonly`**：**`3.76%`** vs **`12.07%`**（`Obs_D_pix` 上 `>10 mm` 达 `14.55%`）。
   - 该结果清晰表明：当前网络在高 $L_{3\text{D}} \ge 10\text{ mm}$ 区的 2–3× 效率抬升，主要来源于**跨钨板宏观纵向延迟起簇/单 MIP 穿透特征与高能运动学**（在 `1 mm` 靶硅螺距与打乱对照下均完整保留），而非 10 mm 钨板间隙内部的微米级次级顶点拓扑分辨（$L_{3\text{D}} < 10\text{ mm}$ 各模型均在 `3.8%–5.5%` 平台）。
2. **仅靶区 OOD 事后量化 vs 重训曲线**（`PinpointBlock_PixXY_All` 1% FPR）：
   - 重训曲线：`20.8 μm = 4.93%`、`100 μm = 6.40%`、`300 μm = 5.15%`、`1.0 mm = 6.11%`（纯 $p_\tau$ `6.55%`）；
   - 事后 OOD 量化（直接作用于 20.8 μm 权重）：`20.8 μm = 4.93%`、`100 μm = 4.56%`、`300 μm = 4.05%`、`1.0 mm = 5.22%`。
3. **机械平面随机横向错位扫描**（主 PF `s321-pf-hybrid-uvtx`，`PinpointBlock_PixXY_All` 1% FPR）：
   - `0 μm = 5.81%`、`10 μm = 5.81%`、`25 μm = 5.96%`、`50 μm = 5.67%`（$\le 50\ \mu\text{m}$ 平面错位波动在 `±0.15 pp` 内）。
4. **Probe A batch 独立注意力熵验证**（`s321-pf-hybrid-uvtx` seed 42, FP32, 10,000 Confirm 事例）：
   - 所有 5 个像素占据桶的归一化熵严格位于 `[0, 1]`：`0` 击中 `mean = 0.267`（像素层 `layer_00..03 = 0.000` 精确掩码，全局融合层 `0.494–0.562`）、`1–100` `0.445`、`100–1,000` `0.383`、`1,000–10,000` **`0.325`**（全局 1% 档 $\tau$ 效率 **`14.14%`** vs 历史 S6 `0.29%`；桶内精确 1% FPR `6.56%`）、`10,000–100,000` `0.337`。全部 152 个注意力头学习尺度 `min = 7.78, median = 11.52, max = 13.03`（全部 $> 4.90$）。

### 3.4 配对系统误差、真实 `PixelSensor[0]` 反符合与质量加权 2t 剖面似然

1. **5 组 1,200 事例配对 Geant4 输运系统误差**（`primary_dual_family_6m` FP32）：
   - `nominal_sys_ref`：`All BA4 = 58.42%`，`MW 2t BA4 = 50.34%`，`PinpointBlock τ@1% = 5.33%`；
   - `tau_helicity_0`（仅作用于 $\nu_\tau$ 信号模板）：`58.17% / 50.17% / 7.10%`；
   - `w_density_0p98`：`60.00% / 55.21% / 5.92%`；
   - `no_magnet_shells`：`56.00% / 50.69% / 5.92%`；
   - `field_0p97`：`59.92% / 55.42% / 5.33%`。
2. **2×800 事例匹配生成器强子化对照**：
   - `pythia6_std_ref`：`All BA4 = 55.63%`，`MW 2t BA4 = 49.67%`，`PinpointBlock τ@1% = 1.92%`；
   - `pythia6_nux`：`59.00% / 54.47% / 7.69%`（已在 27-bin PL 中纳入逐过程 `gen_pythia6_nux` 形状 nuisance）。
3. **外部穿入审计与 `PixelSensor[0]` 背散射实测**（1,000 事例 + 10,000 Confirm 信号损失）：
   - 像素面内岩石 $\mu$（`350` 例）：`PixelSensor[0] >= 1` 反符合效率 **`100.0% (350/350)`**（95% Clopper-Pearson 下界 `98.95%`），未反符合 $\tau$ 1% 泄漏 `0/350`。
   - 外框（无上游硅覆盖，`71.8%` 钨前端面积）岩石 $\mu$（`150` 例）：上游硅反符合覆盖率 **`0.0%`**，分类器 1% 档泄漏 `0/150`（单侧 95% 上界 `1.98%`）。
   - 外部中性强子（`500` 例 $n/K_L^0$）：未触发 `PixelSensor[0]` 比例 **`84.0% (420/500)`**（修复 Stage 319 误用下游闪烁板得到的 `20.7%` 假象），1% 档通过 `0/500`（95% 上界 `0.60%`），Top-3 超尾部 SR 箱落入 `0/500`。
   - **关键硬件发现 — `PixelSensor[0]` 中微子簇射背散射损失**：要求 `PixelSensor[0] == 0` 对全 2t 质量加权中微子信号损失为 **`1.77%`**，但对 **`FrontPinpoint_PixXY` 信号损失高达 `25.22%`**！逐钨板背散射率呈明显指数衰减：
     - `slab 0`: **`78.9%`**（中位 2 hits）、`slab 1`: **`60.8%`**、`slab 2`: **`45.5%`**、`slab 3`: `32.2%`、`slab 4`: `23.0%`、`slab 5`: `16.7%`、`slab 6`: `13.5%`、`slab 7`: `12.8%`、`slab 8`: `6.6%`、`slab 9`: `5.8%`、`slab 10`: `5.9%`、`slab 11`: `3.2%`；`IntermPinpoint`: `0.07%`；`FortuneAll`: `0.22%`。
4. **质量加权 2t 27-bin（21 SR + 6 Charm-CR）剖面似然与 1,000 次 $H_0$ Toys**：
   - 所有 21 SR 箱均无零 MC 背景（`min_raw_mc_bkg_in_sr21 = 1`，`min_eff_mc_bkg_in_sr21 = 1.0`）；全部优化器收敛（`optimizer_converged = true, bb_optimizer_converged = true`）。

| 阶段 / 积分亮度 | Fiducial 总事例 / $\nu_\tau$ CC / 独立外部中性强子 `b_ext` | 仅统计 $Z_0$ | 全物理+G4/Gen/Ext/Charm 27-bin PL $Z_0$ | +10k-Confirm 加权 BB-lite $Z_0$ |
|---|---:|---:|---:|---:|
| Run 3 (`250 fb⁻¹`) | 15,000 / 45 / 300 | `0.54σ` | **`0.30σ`** | `0.23σ` |
| Run 4 (`680 fb⁻¹`) | 40,800 / 122 / 816 | `0.88σ` | **`0.46σ`** | `0.31σ` |
| HL-LHC (`3 ab⁻¹`) | 180,000 / 540 / 3,600 | `1.85σ` | **`0.87σ`** | `0.42σ` |
| FPF (`10 ab⁻¹`) | 600,000 / 1,800 / 12,000 | `3.38σ` | **`1.48σ`** | `0.47σ` |

- **HL-LHC 1,000 次 $H_0$ Toys 覆盖校验**：
  - 全物理系统误差 PL（`enable_barlow_beeston=False`）：`zero_fraction = 0.531`（渐近 `0.500`）、`P(Z>1σ) = 0.157`（渐近 `0.159`）、`P(Z>2σ) = 0.032`（渐近 `0.023`）、`P(Z>3σ) = 0.003`（渐近 `0.0013`）、`p95(q0) = 2.712`（渐近 `2.706`）。
  - 叠加 10k-Confirm 加权 BB-lite（`enable_barlow_beeston=True`）：`zero_fraction = 0.456`、`P(Z>1σ) = 0.204`、`P(Z>2σ) = 0.040`、`P(Z>3σ) = 0.001`、`p95(q0) = 3.464`（因最稀疏 SR 箱仅 1 个有效 MC 背景，1–2σ 区域相对渐近 $\frac{1}{2}\chi^2_1$ 略偏保守不足，说明在 10k 确认统计下必须扩大稀疏尾箱背景 MC 或以 toys 校准 p 值）。

### 3.5 重构前后（Stage 319 vs Stage 321 Clean v2）探测器设计结论逐项对比

| 探测器设计议题 | Stage 319（存在 P1 缺陷时的旧结论） | Stage 321 Clean v2（全部物理修复 + 确定性 FP32 后实测） | 结论状态 |
|---|---|---|---|
| **1. 靶硅像素螺距**（$20.8/100/300\ \mu\text{m}/1\text{ mm}$） | 声称 $20.8\text{–}300\ \mu\text{m}$ 处于分辨率平台，至 `1.0 mm` 时 `BA4` 显著下跌 `-0.97 pp [+0.41, +1.62]` 且 $\tau$@1% 降至 `4.10%` | 修正 `quantize_target_only=True`（保持 3 层 IPT 原生 $20.8\times22.8\ \mu\text{m}$）并在真实 24 块 PinpointBlock 上评估：`BA4` 从 `20.8 μm` 至 `1.0 mm` **完全平坦**（`62.53% / 62.35% / 62.64% / 62.49%`，$\Delta=+0.04\text{ pp}\ [-0.71, +0.77]$），$\tau$@1% 无可分辨下跌（`4.93% / 6.40% / 5.15% / 6.55%`，$p_\tau$）。**Stage 319 的 `1.0 mm` 性能下跌实为错误劣化 3 层 IPT 谱仪平面所致** | **实质改写（IPT 精度关键，靶硅在当前分类器下至 1 mm 无可分辨 BA4 损失）** |
| **2. 靶硅层数**（$12\text{T}+3\text{IPT}$ vs $6\text{T}+3\text{IPT}$） | 声称减半靶硅至 6 层几乎无损失（$\tau$@1% `5.74% vs 4.98%`，主顶点 $|\Delta z|$ 仅 `6.20 → 6.42 mm` `+3.5%`） | 在真实 24 块 PinpointBlock 钨板上，6 层虽在事件级 $\tau$@1%（`6.25% vs 4.93%`）和 `BA4`（`62.02% vs 62.53%`）无显著损失，但**主顶点纵向分辨率显著劣化 +39%**（`median |Δz|: 4.96 → 6.92 mm`，`p68: 8.05 → 11.33 mm`，`3D: 6.36 → 8.47 mm`），`cond_BA3` 下降 `-0.85 pp`。旧结论系 FortuneBlock 混入稀释所致 | **重要修正（减层对事件级分类影响小，但使单钨板顶点定位精度恶化 ~40%）** |
| **3. 显式微顶点分支与微米级 $\tau$ 位移识别**（Exp-1 证伪） | 声称显式微顶点带来 `+1.17 pp [0.00, +2.51]` $\tau$@1% 增益，归因于捕捉微米级次级位移 | 相对 `20.8 μm` 基线，`Hybrid` 真实改善主顶点横向精度（`|Δx/Δy|: 0.78/0.79 → 0.63/0.61 mm`，`-20%`）并给出正向点差（纯 $p_\tau$ `+1.55 pp [-0.36, +3.09]`）；**但相对打乱跨面 tracklet 关联的 `Hybrid-Shuffled` 证伪对照，`Δτ@1%` 仅为 `+0.74 pp [-1.25, +2.28]`（$p_\tau$）/ `-0.15 pp [-2.22, +2.17]`（$S_\tau$）/ `-0.35 pp`（`Obs_D_pix`），未通过证伪门禁**；能量控制 $L_{3\text{D}}$ 分层证明 `>10 mm` 高效率在 `1 mm` 与 `Shuffled` 中同样存在 | **推翻因果归因（主顶点横向定位增益为真，微米级 $\tau$ 位移识别增益未获证伪对照支持）** |
| **4. 机械横向对准公差**（$0\text{–}50\ \mu\text{m}$） | $0/10/25/50\ \mu\text{m}$ 下 $\tau$@1% 为 `4.98% / 4.98% / 4.98% / 4.92%`（几乎不变） | `0/10/25/50 μm` 下 `PinpointBlock` $\tau$@1% 为 `5.81% / 5.81% / 5.96% / 5.67%`（波动 $\le 0.15\text{ pp}$） | **维持（$\le 50\ \mu\text{m}$ 平面错位对当前事件级选择无显著影响）** |
| **5. 谱仪光学与 NOMAD 横向运动学** | 含 `theta_y_entry` 错误且无 `Magnet0` 反推，加入 NOMAD 使 $\tau$@1% 从 `6.15%` 劣化至 `5.56%`（`-0.59 pp`） | 修正 `theta_y_entry` 与解析 $C^1$ `Magnet0` 反推（活体 G4 残差 `0.000–0.051 mm`）后，**劣化完全逆转**：`s321-pf-hybrid-nomad` 在 `PinpointBlock` 达全场最高 **`6.62%`**、`FrontPinpoint` 达 **`7.10%`**、`Obs_D_pix` `cond_BA3 = 80.54%`（最低 `NLL = 0.8563`），但相对 `Hybrid` 的配对 CI 仍跨零（`[-0.89, +2.26]`），且在质量加权 2t（78% Fortune）上略低（`4.46%`） | **方向逆转（光学修复消除了 NOMAD 在硅靶区的劣化，但在 10k 样本下超额收益尚未达 95% 显著）** |
| **6. 上游带电粒子反符合设计** | 误将 `W0` 下游的 `panel <= 1` 称为上游反符合，声称岩石 $\mu$ 反符合率 `97.7%` 且中性强子仅 `20.7%` 未反符合 | 修正为真实上游 `PixelSensor[0] >= 1`（`z=-2991.995 mm < W0`）：像素面内岩石 $\mu$ 反符合 **`100.0% (350/350)`**，中性强子 **`84.0%`** 未反符合；**同时暴露两项真实硬件短板**：(a) `PixelSensor[0]` 紧贴 `W0` 导致 `FrontPinpoint` 中微子背散射损失 **`25.22%`**（`slab 0` 高达 **`78.9%`**）；(b) 外框 **`71.8%`** 钨前端面积无上游硅覆盖。需增设独立大孔径（$\ge 450\times450\text{ mm}$）前向反符合闪烁体墙并采用径迹指向匹配而非盲零击中阈值 | **重大硬件修正（暴露背散射损失与 71.8% 外框盲区，提出明确新硬件设计需求）** |
| **7. 全 2t 真实探测灵敏度**（27-bin PL） | 无质量加权（高估 Pinpoint 占比至 ~65% 伪切片）、缺陷系统误差模板下：Run 3 `0.24σ`、Run 4 `0.34σ`、HL-LHC `0.63σ` (`BB 0.21σ`)、FPF `1.05σ` (`BB 0.21σ`) | 精确 3D 质量加权（Pinpoint `22.16%` + Fortune `77.84%`）、全 27 箱逐过程有符号形变 + 独立加性 `b_ext` 下：Run 3 **`0.30σ`**、Run 4 **`0.46σ`**、HL-LHC **`0.87σ`** (`BB 0.42σ`)、FPF **`1.48σ`** (`BB 0.47σ`)。依然远低于 $3\sigma$ 发现门槛，全物理系统误差 $H_0$ toys 与渐近分布吻合 | **数值升级但定性维持（全 2t 仅靠现有事件级分类在 HL-LHC/FPF 仍不足 $3\sigma$）** |

### 3.6 产物与英文图表路径

- 预解封冻结清单：
  - `/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v5_clean_v2_2026-11-19/clean_sim/v7_freeze_manifest.json`
  - `/eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/roadmap/v3/v7_freeze_manifest.json`
- Clean v2 数据集审计 JSON 与英文图：
  - `/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v5_clean_v2_2026-11-19/clean_sim/stage321_clean_v2_dataset_audit.json`
  - `/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v5_clean_v2_2026-11-19/clean_sim/stage321_clean_v2_physics_audit.png`
- 不可变一次性 FP32 解封报告、冻结 Cal-A/Cut-B 参数与两张英文图：
  - `/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v5_clean_v2_2026-11-19/clean_sim/stage321_clean_v2_unsealing_report.json`
  - `/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v5_clean_v2_2026-11-19/clean_sim/stage321/cal_a_frozen_params.json`
  - `/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v5_clean_v2_2026-11-19/clean_sim/stage321/cut_b_locked_params.json`
  - `/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v5_clean_v2_2026-11-19/clean_sim/stage321_clean_v2_hardware_and_falsification.png`
  - `/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v5_clean_v2_2026-11-19/clean_sim/stage321_clean_v2_vertex_sys_and_sensitivity.png`
- 解封后只读物理诊断附录：
  - `/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v5_clean_v2_2026-11-19/clean_sim/stage321_post_unsealing_diagnostics.json`

---

## 4. 阶段性结论

### 4.1 事实（由代码、单测、活体 Geant4 及一次性 FP32 解封数据直接证实）

1. **Stage 320 全部可定位物理与软件缺陷已完成闭合**：352/352 单测通过；活体 `pinpoint_fixed` 闭合验证谱仪动量误差 `0.09%–1.34%`、`Magnet0` 反推残差 `0.000–0.051 mm`、岩石 $\mu$ 在 `PixelSensor[0]` 100% 留下 $\ge 1$ 击中；Clean v2 60,000 事例零 `edep_proxy_MeV`、零越界、质量加权 Pinpoint/Fortune = `22.16%/77.84%`、单次 SiPM 饱和达 `14,400 PE`、Pythia8 NC 粲覆盖 276 个连续能量；Cal-A 成员置换不变性达 `2.81 × 10⁻⁸`（6-model）/ `3.16 × 10⁻¹⁵`（3-model）且 0 次 argmax 翻转；Probe A 归一化熵全部严格落在 `[0, 1]`。
2. **Stage 319 关于靶硅 `1.0 mm` 螺距性能下跌的旧结论被证伪**：仅量化 12 层靶硅而保持 3 层 IPT 原生分辨率后，真实 24 块 PinpointBlock 上从 `20.8 μm` 到 `1.0 mm` 的 `BA4` 完全平坦（`62.53% → 62.49%`，$\Delta=+0.04\text{ pp}\ [-0.71, +0.77]$），`τ@1%` 亦无可分辨下降；Stage 319 的 `-0.97 pp` `BA4` 下跌源于同时劣化了 3 层 IPT。
3. **减半靶硅层数（12T → 6T）对事件级分类影响小，但显著劣化纵向顶点定位**：在真实 PinpointBlock 上 `median |Δz|` 由 `4.96 mm` 恶化至 `6.92 mm`（`+39%`），`p68 |Δz|` 由 `8.05 mm` 恶化至 `11.33 mm`（`+41%`），`cond_BA3` 降 `-0.85 pp`。
4. **Exp-1 证伪对照未支持当前架构实现微米级 $\tau$ 位移识别**：`Hybrid` 确将主顶点横向分辨率改善约 `20%`（`0.78/0.79 → 0.63/0.61 mm`），但相对打乱跨面 tracklet 关联的 `Hybrid-Shuffled`，`Δτ@1%` 为 `+0.74 pp [-1.25, +2.28]`（纯 $p_\tau$）/ `-0.15 pp [-2.22, +2.17]`（$S_\tau$）/ `-0.35 pp`（`Obs_D_pix`），未达 95% 显著。能量控制（$E_\nu > 300\text{ GeV}$）分层显示，高效率集中在 $L_{3\text{D}} \ge 10\text{ mm}$ 且在 `1.0 mm` 与 `Shuffled` 模型中同样显著存在。
5. **修复 `Magnet0` 光学消除了 NOMAD 在硅靶区的性能劣化**：`s321-pf-hybrid-nomad` 逆转了 Stage 319 的 `-0.59 pp` 下跌，在 `PinpointBlock`（`6.62%`）、`FrontPinpoint`（`7.10%`）和 `Obs_D_pix`（`cond_BA3 = 80.54%`）均取得 9 组最高点估计（但相对 `Hybrid` 的 95% CI 仍含零）。
6. **`PixelSensor[0]` 单独充当上游反符合存在严重硬件短板**：虽在像素面内反符合 `100.0% (350/350)` 岩石 $\mu$，但直接贴附 `W0` 导致 `FrontPinpoint` 中微子因背散射损失 **`25.22%`**（`slab 0` 达 `78.9%`），且漏掉 **`71.8%`** 无上游硅覆盖的外框钨面积。
7. **全 2t 质量加权 27-bin 剖面似然预期显著性仍不足 $3\sigma$**：在给定 2t 标称产额假设（Run 3/4/HL-LHC/FPF = 15k/40.8k/180k/600k 事例、0.3% $\nu_\tau$ 占比、4.5% NC 粲、2% 独立外部中性强子）下，全物理系统误差 PL $Z_0$ 依次为 `0.30σ / 0.46σ / 0.87σ / 1.48σ`；加入 10k-Confirm 加权 BB-lite 后为 `0.23σ / 0.31σ / 0.42σ / 0.47σ`。

### 4.2 假设与待验证边界（与事实严格区隔）

1. **为什么 `20.8 μm` 靶硅尚未显现相对 `1.0 mm` 的分类优势（物理与算法双假说）**：在 5 mm 钨板（$1.43 X_0$）+ 10 mm 间隙的厚靶构型中，低中动量带电径迹多重库仑散射与电磁簇射展宽可能掩盖了单纯像素点云分类器中的微米信号；**同时**，当前端到端分类器仅在 32k 样本上通过事件级软标签间接学习，并未进行显式逐径迹卡尔曼拟合、电荷关联与独立主/次顶点（PV/SV）三维重建，因此**不能由此得出“20.8 μm 靶硅硬件对 $\tau$ 重建无物理价值”的定论**——该命题只能通过显式对象级径迹/kink/SV 重建与控制钨厚度的材料扫描来判定。
2. **2t 灵敏度的外部先验边界（Stage 320 P0）**：表中的 $Z_0$ 仍条件于固定标称总事例数与 `0.3%` $\nu_\tau$ 占比、`2%` 外部中性强子率假设，以及单一 GENIE + Geant4 (`FTFP_BERT+PY8DK`) 模拟族；跨替代中微子生成器（如 NuWro/FLUKA/Pythia8 DIS）、独立 Geant4 physics list、合作组通量协方差矩阵及测试束标定的外推效应**未测量**。
3. **10k 确认统计与 BB-lite 尾部覆盖**：在 `N_confirm = 10k` 下，配对 `Δτ@1%` 的 95% CI 半宽为 `±1.5–2.5 pp`，无法分辨 `< 1 pp` 级别的硬件等效性；且最稀疏 SR 箱仅 1 个有效 MC 背景，导致加权 BB-lite toys 在 1–2σ 偏离渐近分布。

---

## 5. 下一步计划

| 优先级 | 具体任务 | 所需输入 | 验收标准 |
|---|---|---|---|
| **P0-1** | **向合作组通量组获取真实逐味 $(E_\nu, \theta, x, y)$ 产额权重与通量协方差，并扩充尾部稀有背景 MC（含重要性抽样）** | 合作组官方 Run 3 / Run 4 / HL-LHC / FPF 逐味通量与误差表；独立 HTCondor 稀有背景生成（目标每主导 SR 箱有效背景 $\ge 25$） | 21 SR 箱 `min_eff_mc_bkg >= 25`；BB-lite 与纯物理 PL 的 $Z_0$ 差距 $< 15\%$；1,000 次 BB-lite $H_0$ toys 在 1σ/2σ/3σ 覆盖闭合于渐近带内 |
| **P0-2** | **在 `Pinpoint_G4` 中评估独立大孔径上游反符合闪烁体墙（$\ge 450\times450\text{ mm}$，置于 `PixelSensor[0]` 上游留空气间隙/薄屏蔽）与径迹指向反符合算法** | `Pinpoint_G4` 新增上游反符合几何宏；10k 岩石 $\mu$ + 10k `FrontPinpoint` 中微子 Geant4 模拟 | 全口径（含 71.8% 外框）岩石 $\mu$ 反符合率 $> 99.9\%$，同时将 `FrontPinpoint` 中微子背散射信号损失由 `25.22%` 压低至 `< 2%` |
| **P1-1** | **对象级 $\tau$ 拓扑重建（独立 PV/SV 重建、kink 寻找与 charm/转化抑制）及 2×2 材料-螺距对照** | 固定 Clean v2 同一批母事件，扫描 2 mm vs 5 mm W 厚度 × `20.8 μm` vs `100 μm` vs `1 mm` 靶硅（保持 3 IPT 原生），配合显式多面径迹拟合与次级顶点配对 | 报告真实 PV/SV 匹配效率、$L_{xy}$ pull、kink 效率/假率（分 $\tau$、CC/NC charm、$\gamma$ 转化、强子再作用），明确回答微米级靶硅在对象级重建中的真实物理增益边界 |
| **P1-2** | **跨域统一 $\tau$ 分数规范化（消除不同观测域 $\lambda_{\text{charm}}$ 导致的跨域零点偏移）与独立生成器/物理列表测试** | 固定 Cal-A 校准后，将跨域排序统一改为单一可比较对数似然比或仅在同域内评估 $\lambda_{\text{charm}}$；引入独立 Geant4 physics list（如 `QGSP_BERT`）及替代中微子生成器验证 | 跨域切片的 $S_\tau$ 与 $p_\tau$ 排序严格单调自洽；独立物理列表下的锁定 1% FPR 工作点偏移被定量标定并纳入系统误差包络 |
