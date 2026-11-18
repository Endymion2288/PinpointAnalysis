# Stage 318：洁净 `Pinpoint_G4` 全探测器重模拟、Pythia8 固定靶 `NC_charm` 注入与物理系统误差样本生产

## 结论摘要

基于 Stage 315 修复并验证通过的 Geant4 11.3.0 可执行文件 [pinpoint_fixed](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v4_audit_2026-11-18/g4/pinpoint_fixed)（`Pinpoint_G4` commit `6bc17446364ff4d6e4f345f316435007b6f118b9`），已完成 **60,000 个独立名义中微子事件**（150 个 Parquet 分片，严格按 128-bit 生成器家族哈希隔离划分为 `train` 32,000、`validation` 8,000、`cal_a` 4,000、`cut_b` 6,000、`confirm` 10,000）及 **8 组专项物理系统误差与外部本底分片**（共 4,800 个事件）的全流程生产、合并与临时 ROOT 文件清理。实测物理审计结果如下：

1. **钨靶纵向顶点包含率（验证 `P0-2` 修复）**：60,000 个名义事件主顶点距最近实体 `G4_W` 钨板中心的最大纵向偏移为 **$2.4500\text{ mm} \le 2.5000\text{ mm}$**（半板厚），落入实体钨靶内的比例为 **`60,000 / 60,000 = 100.000%`**（旧模拟样本实测有 `10.96%` 顶点落在错位背板外的空气/闪烁体/硅片间隙中）。
2. **连续多 TeV $\nu_\tau$ CC 能谱与固定靶 `NC_charm` 本底（验证 `P2-1` 修复）**：合并 `Pythia6_Standard` 与 `Pythia6_NUX` 高能池后，共注入 **132 个 $E_\nu > 2.312\text{ TeV}$ 的 $\nu_\tau$ CC 事件**，最高入射能量达 **$3,434.0\text{ GeV}$（$3.43\text{ TeV}$）**，消除了旧样本在 $2.312\text{ TeV}$ 处的人为截断；同时使用 `LCG_107` 内置 Pythia8（8.312）模拟质子/中子固定靶 $c\bar{c}$ 强子化，在 15,000 个 NC 事件中注入 **2,397 个真实含粲强子（$D^\pm, D^0, D_s^\pm, \Lambda_c^+$）的 `NC_charm` 事件**（占 NC 总量的 **`15.98%`**），外加 CC 通道内源产生的 **5,142 个 `CC_charm` 事件**。
3. **Birks 猝灭与 SiPM 微单元饱和数字化响应（验证 `P1-1` 修复）**：对 `validation` 分片 `080` 的 **628,558 个闪烁体通道**实测统计表明，数字化输出严格受限于 SiPM 微单元上限 **`14,400.0 PE`**（$N_{\text{pix}} = 14,400$），大能量沉积区（$E_{\text{dep}} > 5\text{ MeV}$）平均有效光产额为 **`95.51 PE/MeV`**，超过 `2.5 PE` 读出阈值的通道占比为 **`96.49%`**；同时模型输入管线仅暴露饱和后单端 `pe_proxy`，完全屏蔽真值 `edep_proxy_MeV`。
4. **探测器输运、强子化与外部穿透本底配对分片（支撑 `Exp-4` 与 `Path E`）**：完成同一 600 事件确认子集在 5 种 Geant4 探测器配置（名义参考、`PINPOINT_TAU_HELICITY=0` 无极化 $\tau$、`PINPOINT_W_DENSITY_SCALE=0.98` 钨密度 $-2\%$、`PINPOINT_DISABLE_MAGNET_SHELLS=1` 移除 `Sm2Co17` 磁铁壳层、磁场强度 $0.97\times 0.57\text{ T} = 0.5529\text{ T}$）下的配对重模拟，以及顶点逐格匹配的 `pythia6_std_ref` vs `pythia6_nux`（各 600 事件）强子化对比分片与 `external_punchthrough`（300 个上游岩石 $\mu^\pm$ + 300 个上游中性强子 $n / K_L^0$，入射面 $z = -3005\text{ mm}$）外部本底分片。

---

## 阶段目标

围绕独立审查报告 [2026-11-18_stage314_independent_scientific_statistical_and_code_audit.md](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/workbook/2026-11-18_stage314_independent_scientific_statistical_and_code_audit.md) 中的 `Exp-4`、`P0-2`、`P1-1`、`P2-1` 与 `Path E` 系统误差要求，回答以下核心问题并完成验收：

1. 修复 `Pinpoint_G4` 纵向层序与数字化物理模型后，大规模生产的事件样本是否 100% 满足实体钨靶顶点包含、SiPM 微单元饱和及连续多 TeV 能谱覆盖？
2. 如何在不泄漏家族同源事件的前提下，构建包含真实三体/半轻衰变粲强子 `NC_charm`、连续高能 $\nu_\tau$ CC 以及配对探测器/生成器/外部本底变体的洁净基准数据集，并将中间 ROOT 产物清理完毕以满足 EOS 存储规范？

**验收目标**：
- 60,000 个名义事件（150 分片）100% 落入实体 `G4_W` 钨板（$|\Delta z| \le 2.50\text{ mm}$），跨 5 个数据集切分（`train` / `validation` / `cal_a` / `cut_b` / `confirm`）的 `family_disjoint_verified == True`。
- 每个切分均包含 $E_\nu > 2.312\text{ TeV}$ 的 $\nu_\tau$ CC 事件与 ~16% 比例的 Pythia8 固定靶 `NC_charm` 事件。
- 8 组物理系统误差与外部穿透本底分片（各 600 事件）全部就位，`sealed_truth_meta` 与 `train/validation` 训练目录物理隔离，EOS 目录零残留 `.root` 中间文件。

---

## 工作内容

1. **多源初级顶点采集与 Pythia8 固定靶 `NC_charm` 合成（[produce_clean_sim.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v6_clean_sim/produce_clean_sim.py)）**：
   - 从 `/eos/project-f/fasersim-bonn/public/pinpoint/data/10081..10084` 并行读取四类中微子基础初级态，并从 `/eos/experiment/fasernu-data0/MC/Genie_files/genie_output_Pythia6tunes/{Pythia6_Standard,Pythia6_NUX}` 提取全部高能（$E_\nu > 2.312\text{ TeV}$）中微子事件，通过全局 `global_seen_fam` 集合与含生成器调参标签的 63-bit 家族指纹（`_family_pair`）消除跨调参同源碰撞。
   - 编写内联 C++ Pythia8 生成器（`SoftQCD:inelastic = on` + `HardQCD:hardccbar = on`，质子束打钨核子靶并按中微子动量传递洛伦兹变换至实验室系），生成 2,600 个真实含粲强子（PDG $\in \{411, 421, 431, 4122, 4132, 4232, 4332\}$）强子簇，在 Geant4 `FTFP_BERT+PY8DK` 中完成有限寿命微米级次级衰变顶点与弯折/多重带电径迹输运。
   - 直接从 `geometry.json` 导出的 24 块实体 `G4_W` 靶板中心坐标（半厚 $2.50\text{ mm}$，留 $0.05\text{ mm}$ 边界裕量即 $\pm 2.45\text{ mm}$）均匀采样中微子相互作用纵向位置 `vz`，并开启 `PINPOINT_USE_INPUT_Z=1`、`PINPOINT_SKIP_GDML=1`、`PINPOINT_COMPACT_TRUTH=1`。
2. **HTCondor 分布式 Geant4 生产与无损清理**：
   - 通过 HTCondor 提交 8 核并行子块模拟作业（每分片 400 事件拆分为 8 个并发 Geant4 子进程，转换出压缩 Parquet 后立即删除子块 `.root` 文件）。针对并发创建目录触发的 EOS FUSE 元数据缓存竞争，在 `prepare_clean_sim` 中预先创建全部 150 个分片目录，完成 `150/150` 分片与 5 组探测器系统误差分片生产。
   - 执行 `produce_clean_sim finalize`：合并生成 `truth_sidecar/event_sidecar.parquet`（40,000 行，仅含 `train` + `validation`）、`validation_meta/tune_truth_meta.parquet`（8,000 行）以及物理隔离存放于 `sealed_truth_meta/{cal_a,cut_b,confirm}_truth_meta.parquet` 的封存真值表，随后自动清理全部 `parts/` 与 `input_shards/` 临时目录。
3. **外部穿透本底与 GENIE 强子化系统误差样本生产（[produce_external_and_gen_sys.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v6_clean_sim/produce_external_and_gen_sys.py)）**：
   - 在本地 14 核并发运行 `pinpoint_fixed` 生产三组 600 事件专项样本：
     - `external_punchthrough`：300 个上游岩石 $\mu^\pm$（$E \in [50, 2000]\text{ GeV}$，$z = -3005\text{ mm}$ 穿过上游反符合闪烁体）与 300 个上游中性强子（60% 中子 $n$ + 40% $K_L^0$，$E \in [25, 600]\text{ GeV}$，$z = -3005\text{ mm}$ 电中性绕过带电反符合层并在钨靶内发生强子簇射）。
     - `pythia6_std_ref` 与 `pythia6_nux`：按四类中微子及能量区间逐格配对抽取 `Pythia6_Standard` 与 `Pythia6_NUX` 事件，强制赋予完全相同的三维空间顶点 $(v_x, v_y, v_z)$，从而纯粹分离强子化模型差异对 27-bin 似然拟合的系统影响。
4. **物理完整性审计与可视化（[run_stage319_clean_campaign.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/v3/v6_clean_sim/run_stage319_clean_campaign.py) `audit-clean-sim`）**：
   - 核验全部 60,000 事件钨靶包含率、各切分高能 $\nu_\tau$ 与粲强子统计量、数字化光电子饱和曲线，输出结构化审计报告与 300-DPI 英文三联验证图。

---

## 详细数据

### 1. 执行命令与核心产物路径

```bash
export PYTHONPATH="/eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src:${PYTHONPATH:-}"
python3 -m analysis.flavour.v3.v6_clean_sim.produce_clean_sim prepare --cpus 12
python3 -m analysis.flavour.v3.v6_clean_sim.produce_clean_sim submit-workers --shards-per-job 3 --cpus 8
python3 -m analysis.flavour.v3.v6_clean_sim.produce_external_and_gen_sys --cpus 14
python3 -m analysis.flavour.v3.v6_clean_sim.produce_clean_sim finalize
python3 -m analysis.flavour.v3.v6_clean_sim.run_stage319_clean_campaign audit-clean-sim
```

- **数据集根目录**：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v4_audit_2026-11-18/clean_sim`
- **几何真值文件**：[geometry.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v4_audit_2026-11-18/clean_sim/geometry.json)
- **数据集准备与合并摘要**：[prepare_summary.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v4_audit_2026-11-18/clean_sim/prepare_summary.json)、[finalize_summary.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v4_audit_2026-11-18/clean_sim/finalize_summary.json)
- **物理完整性审计 JSON**：[stage318_clean_dataset_audit.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v4_audit_2026-11-18/clean_sim/stage318_clean_dataset_audit.json)
- **300-DPI 英文物理验证图**：[stage318_clean_pinpoint_g4_physics_audit.png](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v4_audit_2026-11-18/clean_sim/stage318_clean_pinpoint_g4_physics_audit.png)

### 2. 名义 60,000 事件切分与高能 / 粲强子物理统计表

| 数据切分 (`split`) | 分片范围 (`shard`) | 总事件数 | 四类均衡数 / 类 | $\nu_\tau$ CC ($E_\nu > 2.312\text{ TeV}$) | $\nu_\tau$ CC 最高能量 [GeV] | `NC_charm` 事件数（占 NC 比例） | `CC_charm` 事件数 | 封存状态 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `train` | `000`–`079` (80) | 32,000 | 8,000 | 71 | 3,434.0 | 1,293 (16.16%) | 2,738 | 开放训练 |
| `validation` (`CleanTune8k`) | `080`–`099` (20) | 8,000 | 2,000 | 14 | 3,306.1 | 333 (16.65%) | 674 | 开放调参 / 选模 |
| `cal_a` (`CleanCalA`) | `100`–`109` (10) | 4,000 | 1,000 | 9 | 3,209.1 | 152 (15.20%) | 346 | 封存（校准专用） |
| `cut_b` (`CleanCutB`) | `110`–`124` (15) | 6,000 | 1,500 | 16 | 2,975.2 | 236 (15.73%) | 525 | 封存（锁阈值专用） |
| `confirm` (`CleanConfirm`) | `125`–`149` (25) | 10,000 | 2,500 | 22 | 2,998.3 | 383 (15.32%) | 859 | 封存（单次解封） |
| **合计** | **`000`–`149` (150)** | **60,000** | **15,000** | **132** | **3,434.0** | **2,397 (15.98%)** | **5,142** | **`family_disjoint_verified = true`** |

### 3. 探测器几何、数字化与配对系统误差实测指标

| 审计项目 | 旧版数据集实测值 | Stage 318 洁净数据集实测值 | 物理与工程意义 |
| --- | --- | --- | --- |
| 钨靶外主顶点比例（$|z_v - z_W| > 2.50\text{ mm}$） | `10.96%` (`6,578 / 60,000`) | **`0.000%` (`0 / 60,000`)** | 消除 12 块背钨板 `-6.010 mm` 错位导致的非钨靶相互作用 |
| 距最近钨板中心最大纵向偏移 $\max |z_v - z_W|$ | `8.46 mm` | **`2.4500 mm`** | 全部严格包络于 $5.00\text{ mm}$ 实心 `G4_W` 板内 |
| 闪烁体单通道最大观测光电子数（分片 `080`，628,558 通道） | 无上限（线性无饱和，最高 $>10^5\text{ PE}$） | **`14,400.0 PE`** | 精确服从 $N_{\text{pix}} = 14,400$ SiPM 微单元指数饱和响应 |
| 高沉积区（$E_{\text{dep}} > 5\text{ MeV}$）平均有效光产额 | 未激活 Birks 猝灭（高估约 $3\times$） | **`95.51 PE/MeV`** | 激活聚苯乙烯 $k_B = 0.126\text{ mm/MeV}$ Birks 定律与光纤衰减 |
| 超出 `2.5 PE` 阈值通道比例 | 零阈值全保留 | **`96.49%`** | 读出链路同步施加 `>= 2.5 PE` 硬件甄别阈值且屏蔽真值 $E_{\text{dep}}$ |
| 配对探测器输运系统误差分片（5 组 × 600 事件） | 无 | **`3,000` 事件（100% 完成）** | 覆盖 $\tau$ 螺旋度极化、钨靶密度 $-2\%$、无磁铁壳层、磁场 $-3\%$ |
| 配对生成器强子化与外部穿透分片（3 组 × 600 事件） | 无 | **`1,800` 事件（100% 完成）** | 覆盖 `Pythia6_Standard` vs `Pythia6_NUX` 与岩石 $\mu^\pm$ / 中性强子 $n, K_L^0$ |
| 洁净数据集上各模型分类效率与发现显著性 | — | **未测量（定于 Stage 319 冻结后单次解封测量）** | 严格遵守 `P3-1` 预解封哈希冻结规程，禁止提前读取封存集 |

---

## 阶段性结论

### 已验证事实
1. [stage318_clean_pinpoint_g4_physics_audit.png](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v4_audit_2026-11-18/clean_sim/stage318_clean_pinpoint_g4_physics_audit.png) 与 [stage318_clean_dataset_audit.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v4_audit_2026-11-18/clean_sim/stage318_clean_dataset_audit.json) 证实，`Pinpoint_G4` 几何与数字化修复在大规模 64,800 事件生产中完全生效：100.00% 名义顶点位于 `G4_W` 实体靶板内部，闪烁体读出严格呈现 Birks 猝灭与 $14,400\text{ PE}$ SiPM 饱和平台，$\nu_\tau$ CC 能谱平滑延伸至 $3.43\text{ TeV}$。
2. 全部 15,000 个名义 NC 事件现已包含 15.98% 真实 Pythia8 固定靶粲强子产生与 Geant4 有限寿命次级衰变，使 $\nu_\tau$ 鉴别器在硅像素区面对真实的 $D^\pm / D^0 / D_s^\pm / \Lambda_c^+$ 寿命折角与半轻子衰变背景。
3. 中间 Geant4 `.root` 文件、临时输入清单 `input_shards/` 与元数据碎片 `parts/` 均已自动清除，EOS 存储结构精简且符合盲法隔离要求。

### 待验证假设（进入 Stage 319 检验）
1. **硬件灵敏度假设（`Exp-2` / `P1-5`）**：在无闪烁体双通道泄漏与真实 Birks/SiPM 响应下，从头重训的 $20\ \mu\text{m}$ 硅像素节距模型及显式微米多航迹次级顶点特征（`Path C`）相较 $100\ \mu\text{m}$ / $300\ \mu\text{m}$ / $1\text{ mm}$ 重训模型与 6 层减半靶区配置，将在硅像素中心切片保持显著的 1% / 0.1% FPR $\tau$ 鉴别优势。
2. **外部穿透本底抑制假设（`Path E`）**：几何对齐后的上游双层闪烁体反符合判据可高效剔除 $z = -3005\text{ mm}$ 入射岩石 $\mu^\pm$，而电中性强子（$n, K_L^0$）在无主带电轻子条件下的高置信度 $\tau$ 尾区泄漏率可通过 `external_punchthrough` 样本定量给出。

---

## 下一步计划

1. **Stage 319 多 GPU 并行从头训练（`CleanTrain32k` × 20 passes，8 组 × 3 随机种子 = 24 个洁净模型）**：
   - 覆盖像素节距重训曲线（`20 μm`、`100 μm`、`300 μm`、`1 mm`）、像素层数减半成本曲线（`6-layer 20 μm`）、显式 C++17 微米航迹/AVR 次级顶点混合架构（`hybrid_uvtx` 与 `hybrid_uvtx_nomad`）及独立投影基线 `clean_b1pp_pe_only`。
2. **Stage 317 受控消融收尾与复杂度准入判定（`Exp-1`、`Exp-3`、`Exp-5`、`Exp-6`）**：
   - 待全部 18 个 `300k×24p` 模型完成后运行 `evaluate_stage317.py`，输出 Stage 317 阶段报告并固化复杂度保留判据。
3. **预解封 Git Commit 与 SHA-256 冻结及单次解封（`P3-1` + `Stage 319`）**：
   - 在完全干净的 Git 工作树上生成含 40 位 `pre_unsealing_git_commit` 与全部模型权重 SHA-256 的 `roadmap/v3/v6_freeze_manifest.json`，随后单次解封 `CleanCalA -> CleanCutB -> CleanConfirm`，输出像素节距曲线、机械平面错位容差（`0/10/25/50 μm`）、配对物理/强子化系统误差包络、外部本底泄漏率与 27-bin 前向通量剖面似然显著性。
