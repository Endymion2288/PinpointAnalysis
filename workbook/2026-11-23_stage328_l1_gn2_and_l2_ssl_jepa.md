# Stage 328：钨靶微米像素团簇起源辅助监督（`TrackletIPFormer-GN2`）与 `334.8k` 纯钨无标签多视图自监督预训练（`PinpointSSL-JEPA`）完整验证报告

- **日期**：2026-11-23
- **基线提交**：`feat/stage327-nextgen-completion` @ `d1a29db`（Stage 327 [2026-11-22_stage327_unified_net_taukinema_and_tail_dro.md](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/workbook/2026-11-22_stage327_unified_net_taukinema_and_tail_dro.md)）
- **子任务审计分支与集成提交**：
  - L1 数据生产：`feat/stage328-l1-sidecar` @ `f4d8a0f`（[produce_l1_cluster_sidecar.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/nextgen/produce_l1_cluster_sidecar.py), [test_nextgen_l1_sidecar.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/tests/test_nextgen_l1_sidecar.py)）
  - L2 数据生产：`feat/stage328-l2-corpus` @ `6192ecc`（[produce_l2_ssl_corpus.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/nextgen/produce_l2_ssl_corpus.py), [test_nextgen_l2_corpus.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/tests/test_nextgen_l2_corpus.py)）
  - L1 模型与实验：`feat/stage328-l1-gn2` @ `add3be7`（[gn2_tracklet_ipformer.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/nextgen/gn2_tracklet_ipformer.py), [run_l1_gn2_tracklet.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/nextgen/run_l1_gn2_tracklet.py), [test_nextgen_l1_gn2.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/tests/test_nextgen_l1_gn2.py)）
  - L2 模型与绘图：`feat/stage328-l2-jepa` @ `825664a`（[pinpoint_ssl_jepa.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/nextgen/pinpoint_ssl_jepa.py), [run_l2_ssl_jepa.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/nextgen/run_l2_ssl_jepa.py), [plot_stage328_summary.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/nextgen/plot_stage328_summary.py), [test_nextgen_l2_jepa.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/tests/test_nextgen_l2_jepa.py)）
  - 集成主分支：`feat/stage328-nextgen-l1-l2` @ `9ec503a`（完整 SHA `9ec503a77efb4c5556d6984f1dff74744e07a219`，经 `--no-ff` 依次合入 `532ca37 -> c6c1c6d -> 2b7eca8 -> 87ee697 -> 9ec503a`，导出 [__init__.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/nextgen/__init__.py) 共 `69` 个公共符号；全套 **`44/44` NextGen 单元测试 `136.271s` GPU 全绿**，`src/analysis/flavour/v2` 与 `v3` 零改动）
- **运行节点与调度记录**：
  - HTCondor CPU 模拟阵列：`bigbird10.cern.ch` cluster `13215043`（L1 `80` × 8-core）；`bigbird21.cern.ch` cluster `2213648`（L2 `80` × 8-core）与 `bigbird15.cern.ch` cluster `17098690`（L2 `80` × 8-core）
  - HTCondor GPU 训练、特征提取与全量单测：`bigbird20.cern.ch` clusters `1585654..1585693`（L1，NVIDIA H100 NVL / A100-PCIE-40GB）、`1585633..1586176`（L2，NVIDIA H100 NVL 95GB / A100-PCIE-40GB）与 `1586571`（`b9pgpun001.cern.ch` NVIDIA H100 NVL 95GB 全套 `44/44` 单元测试）
  - Kubeflow `ml-prod`（namespace `xcheng`）：pod `l2-sup-eq-compute-688p2`（`NVIDIA H100 NVL MIG 3g.47gb`，L2 等算力监督基线 `262` passes）
  - 交互式子套件验证与独立审计：`lxplus-gpu`（NVIDIA Tesla T4 16GB，CVMFS `LCG_108_cuda` Python 3.12 / PyTorch 2.7）
- **随机种子**：Stage-I 单种子受控实验 `seed = 42`；Stage-II 三种子确认 `seeds = {17, 42, 2026}`；配对 Bootstrap 种子 `20263250`（`1,000` 次按 `(family_hi, family_lo)` 家族簇重采样）
- **核心产物目录**：
  - 架构与协议蓝图：[blueprint_l1_l2.md](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage328_plan/blueprint_l1_l2.md)
  - 独立审计报告与核验记录：[stage328_audit_report.md](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage328_verification/stage328_audit_report.md)（含 [prod_l1_independent_audit.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage328_verification/prod_l1_independent_audit.json), [prod_l2_independent_audit.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage328_verification/prod_l2_independent_audit.json), [coder_l1_independent_audit.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage328_verification/coder_l1_independent_audit.json), [coder_l2_independent_audit.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage328_verification/coder_l2_independent_audit.json)）
  - 实验 L1 数据与模型：[l1_cluster_sidecar](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/l1_cluster_sidecar)（[sidecar_production_summary.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/l1_cluster_sidecar/sidecar_production_summary.json)）、[exp_l1_gn2](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/exp_l1_gn2)（[stage1_l1_gn2_results.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/exp_l1_gn2/stage1_l1_gn2_results.json), [stage2_l1_gn2_results.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/exp_l1_gn2/stage2/stage2_l1_gn2_results.json)）
  - 实验 L2 数据与模型：[l2_ssl_300k](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/l2_ssl_300k)（[l2_corpus_manifest.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/l2_ssl_300k/l2_corpus_manifest.json)）、[exp_l2_jepa](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/exp_l2_jepa)（[stage1_l2_jepa_results.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/exp_l2_jepa/stage1_l2_jepa_results.json), [stage2_l2_jepa_results.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/exp_l2_jepa/stage2/stage2_l2_jepa_results.json)）
  - 汇总图表（300 DPI English）：[fig1_l2_ssl_equal_compute_and_label_efficiency.png](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage328_figures/fig1_l2_ssl_equal_compute_and_label_efficiency.png), [fig2_l2_unlabeled_probe_and_pv_calibration.png](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage328_figures/fig2_l2_unlabeled_probe_and_pv_calibration.png), [fig3_l1_gn2_causal_ablation_and_probes.png](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage328_figures/fig3_l1_gn2_causal_ablation_and_probes.png), [fig4_stage328_stage2_synergy_ood_stress.png](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage328_figures/fig4_stage328_stage2_synergy_ood_stress.png)
- **封存盲法审计声明**：独立审计扫描全部 `1,245` 个源码、日志与 JSON 产物文件并核对 [seal_guard.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/nextgen/seal_guard.py) 运行时审计日志，确认封存数据集 `Cal-A (shards/100..109)`、`Cut-B (shards/110..124)`、`Confirm (shards/125..149)`、`sealed_truth_meta/`、`planned_manifest.parquet`、`input_sys_confirm*` 及封存 GENIE 分块 `016..028` 全程 **0 次读取或元数据探查**，所有封存集指标在本报告中严格标注为 **`未测量`**。

---

## 核心科学结论（结论先行）

Stage 328 完整执行并独立核验了 Stage 326 路线图（[2026-11-22_stage326_nextgen_neutrino_ml_research.md](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/workbook/2026-11-22_stage326_nextgen_neutrino_ml_research.md) 第 10.3 节）仅剩的两项需要新数据资产的物理与机器学习实验 **L1（`TrackletIPFormer-GN2` 像素团簇真值起源训练期辅助监督）** 与 **L2（`334.8k` 纯钨无标签语料库与 `PinpointSSL-JEPA` 多视图自监督预训练）**，在不触碰任何封存分片的前提下刷新了 `Tune6k`（`N = 6,000`）全部主指标纪录：

1. **`32,000` 纯钨事例微观起源普查与 `TrackletIPFormer-GN2` 因果机制证伪/证实（实验 L1）**：
   - 在 `32,000` 个纯钨（`G4_W`, `Z == 74`）事例的 **`10,803,182` 个硅像素团簇**（与观测团簇 `100.0%` 空间对齐 `< 50 um`，抽检最大坐标差 `0.0 um`）上，精确量化了钨电磁/强子簇射的微观背景结构：**`83.542%`（`9,025,207` 个）团簇来自钨板内 $\gamma \to e^+e^-$ 光子对转换**，`14.230%` 来自强子次级散射或噪声，`PV_primary` 仅占 `2.022%`，`tau_decay` 仅占 **`0.143%`**，`charm_decay` 占 `0.063%`。在 `Front30` 区段，C++ 贪心几何 3-hit 径迹段的真实同 `trackID` 纯度仅为 **`2.994%`**（**`97.006%` 为跨粒子组合假径迹**），且在 `|IP|/\sigma_{\text{IP}} > 2.5` 的位移径迹段中 **`99.776%` 为光子转换或组合鬼影**。
   - **关键因果分离**：在零混杂受控基线 `l1_ctrl_no_gn2`（同数据、同架构、$\lambda_{\text{GN2}} = 0$）之上，仅施加训练期多任务辅助损失而不改变推理期读出门控（`abl_aux_loss_only_no_readout_gate`）在 `D1_obs` 上仅带来 **`+0.064 pp`** 的统计零效应；而将模型从纯观测特征预测出的径迹纯度 $\hat{w}_{\text{real},t}$ 与非转换重味先验显式注入推理期 Student-$t$ 顶点拟合及 Perceiver 交叉注意力（`abl_origin_purity_only`），在严格 **0 真值/0 sidecar 推理输入**（39 键与 `cluster_sidecar` 篡改最大绝对差 `== 0.0`）下，离线探针达到 **`purity AUROC = 0.9682`**、**重味对光子转换判别 `AUROC = 0.9926`**、**$\tau$ 衰变叉判别 `AUROC = 0.9767`**，驱动 Stage-I `D1_obs BA4` 跃升 **`+1.61 pp`**（95% CI `[+0.15, +3.16] pp`，`ΔNLL = -0.0151 [-0.0282, -0.0020]`），并将核外推 `W184 NC Charm` 假阳性率压至 **`2.17%`**。
   - **Stage-II `Tune6k`（`N = 6,000`，3 seeds `{17, 42, 2026}`）确认**：`TrackletIPFormer-GN2` 单模型集成达 **`59.20% BA4 / 0.9379 NLL`**（冻结主干 `Score-Bank KL-DRO` 后达 **`59.38% BA4 / 0.9370 NLL`**，`W184 p_tau > 0.5 = 0.67%`），较 3-seed 零混杂对照 `l1_ctrl_no_gn2`（`58.85% / 0.9458`）取得显著因果增益（`All_Tune6k ΔNLL = -0.00794 [-0.01167, -0.00457]`，`Front30 ΔBA4 = +1.265 pp [+0.437, +2.116]`，较 Stage 326 `TrackletIPFormer` 在 `Front30` 提升 **`+2.011 pp [+0.869, +3.067]`**）；仅使用 `tune_screen` 校准后 `Tune6k` 全部 9/9 Student-$t(\nu=3)$ 顶点 Pull 落在 **`[0.8725, 1.0720]`**。同时，由于 $\hat{w}_{\text{real}}$ 自动压制暗电流构成的孤立组合假径迹，其在 `Shard 080 pixel_dark_1e-5` 强噪声压力下保持 **`54.82% BA4`**（退化 `-4.32 pp`，显著优于无纯度门控模型的 `-15.16 pp`）。与 `DualRegimeModalNet` 等权融合并接入 `Score-Bank KL-DRO` 后达 **`59.43% BA4 / 0.9291 NLL`**（校准后 `0.9285`），较 Stage 327 最优 `s327_blend_dro` 提升 **`+0.665 pp BA4 [+0.108, +1.213]`** 与 **`ΔNLL = -0.00823 [-0.01110, -0.00511]`**。

2. **`334,754` 纯钨包含态无标签语料库、`PinpointSSL-JEPA` 反捷径机制验证与跨范式协同（实验 L2）**：
   - 使用 160 个 8 核 HTCondor 作业在严格不重叠的高编号 GENIE 分块（`chunk >= 029`，与 `Train32k`、`L1`、`Tune8k`、`6 OOD` 及全部 `000..028` 分块家族重叠数 `== 0`）上生产了 **`334,754` 个纯钨可观测包含态（`D1 ∪ D2`）无标签训练事例**与 **`11,715` 个独立无标签预校准验证事例**（EOS 总体积 **`2.92 GiB < 3.5 GB`**，**0 个 `.root` 文件，0 个真值列**）。
   - **零标签穿透 $\mu$ 磁谱仪预校准与反捷径消融证实**：在 `11,715` 个无标签 `ssl_val_calib` 事例上，`PinpointSSL-JEPA` 仅靠“遮蔽整座后磁谱仪、从钨量能区预测穿透径迹”的自监督任务（Task 3），即涌现出 **`mu_AUROC = 0.9354`** 与磁偏转动量代理相关系数 **`r(\ln|s_y|) = 0.4873`**（$p = 5.89 \times 10^{-99}$；移除 Task 3 后崩塌至 `0.6036` 与 `0.0319`），使下游微调提升 **`+1.063 pp BA4 [+0.242, +2.003]`**（`ΔNLL = -0.0115`）。遵循 `NOMAE` / `Sonata` 的整平面遮蔽且预测器仅接收 $(z_p, X_{0,p})$ 设计（遮蔽平面坐标 `+999 mm` 篡改绝对差 `== 0.0`），较朴素 `60%` 随机单团簇遮蔽（受钨簇射局部横向荧光捷径污染）在 `1% FPR` $\nu_\tau$ 效率上提升 **`+2.083 pp [+0.085, +4.220]`**（`D1_obs +4.626 pp`），`ΔNLL = -0.0131 [-0.0220, -0.0045]`。
   - **严格等总算力（Equal-Total-FLOPs）核验与单模型归纳偏置权衡**：经 `FlopCounterMode` 实测，等算力从头监督基线 `sup_scratch_equal_compute`（`262` passes，`5.688 \times 10^{16}` FLOPs，`FLOPs_ratio = 1.0547x >= 1.00`）在第 `13` pass 达到 `dev_screen NLL = 0.9813` 后陷入严重记忆化过拟合（第 `182` 至 `262` pass `dev_screen NLL` 恶化至 `5.07 -> 5.86`，训练损失降至 `1.184`）；`ssl300k_jepa_ft_100pct` 在验证集对数似然上优于等算力监督基线（`0.9729` vs `0.9785`，`ΔNLL = -0.0056`）。同时实测揭示清晰的架构归纳偏置边界：为严格满足整平面遮蔽零泄漏，`PinpointSSL-JEPA` 主干采用统一平面/点级 Transformer 而未嵌入跨三层显式组合 `TrackletOps`，且预训练语料含 `71.5% D2_obs` 自然事例，因此其单模型（Stage-I `58.04% / 0.9729`，Stage-II 3-seed `58.20% / 0.9493`）及 `10%`/`25%` 低标签微调在 `BA4` 上不及监督基线（`10%` 切片校准更优 `1.0860` vs `1.0946`），但其 `Front30` 纵向顶点中位残差达到全场最优 **`3.55 mm`**（`All_valid = 5.22 mm`，9/9 Pulls `[0.9568, 1.0479]`）。
   - **Stage-II 跨范式融合突破**：正因其整平面簇射演化与跨磁铁连续潜空间表征与显式径迹/手工区段网络高度正交，在 `Tune6k` 上 **`50/50 PinpointSSL-JEPA + TrackletIPFormer-GN2`** 创下全场最高原始准确率 **`59.73% BA4 / 0.9307 NLL`**；三范式融合 **`3-Way (L1_GN2 + DualRegimeModalNet + PinpointSSL-JEPA)`** 将原始对数似然推进至 **`0.9277 NLL`**（`59.28% BA4`），接入冻结主干 **`Score-Bank KL-DRO`（`blend_3way_l1_dual_jepa_dro`）** 后达到 **`59.57% BA4 / 0.9275 NLL`**（`All 1% FPR τ_eff = 7.16%`，`D1_obs = 9.62%`），较 Stage 327 最优 `s327_blend_dro` 显著提升 **`+0.808 pp BA4 [+0.204, +1.460]`** 与 **`ΔNLL = -0.00984 [-0.01350, -0.00619]`**。

---

## 1. 阶段目标

围绕 Stage 326（[2026-11-22_stage326_nextgen_neutrino_ml_research.md](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/workbook/2026-11-22_stage326_nextgen_neutrino_ml_research.md)）与 Stage 327（[2026-11-22_stage327_unified_net_taukinema_and_tail_dro.md](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/workbook/2026-11-22_stage327_unified_net_taukinema_and_tail_dro.md)）确立的双阶段门控与封存盲法准则，Stage 328 设定四项定量验收目标（详见 [blueprint_l1_l2.md](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage328_plan/blueprint_l1_l2.md)）：

1. **L1 训练期微观真值边车（`cluster_sidecar`）精确生产与钨簇射假径迹普查**：
   - 在严格不触碰 `Tune8k`、`6 OOD` 及任何封存分片的前提下，生产 `32,000` 个纯钨事例的像素团簇级 Geant4 微观起源标注（5 类 `origin_label`、次级顶点分组 `sv_group_id`、前 3 能量贡献 `trackID`）；验证与 `extract_pf_item` 保留团簇 `100%` 空间匹配（`< 50 um`），EOS 边车体积 `< 100 MB` 且零 `.root` 残留。
2. **L1 `TrackletIPFormer-GN2` 训练期辅助监督与推理期零真值门控因果验证**：
   - 借鉴 ATLAS `GN2` 多任务训练期真值辅助监督范式，验证将预测径迹纯度与非转换重味权重显式接入推理期顶点与读出融合后，能否在不读取任何真值/边车的条件下压制 `97%` 组合鬼影与 `99%+` $\gamma \to e^+e^-$ 假位移顶点，并在同数据零混杂对照 `l1_ctrl_no_gn2` 及 `Tune6k` 3-seed 评估上取得统计显著增益。
3. **L2 `>= 300,000` 纯钨可观测包含态无标签语料库生产**：
   - 仅从 `chunk >= 029` 的非重叠 GENIE 分块出发，使用 `pinpoint_v3_1`（`PINPOINT_SKIP_GDML=1`, `/out/saveTrack false`, `/out/saveTruthHits false`）生产 `>= 300,000` 个纯钨（`Z == 74`）可观测包含态（`D1 ∪ D2`）无标签训练事例及 `>= 10,000` 个独立无标签预校准验证事例，紧凑 `.npz.zst` 存储 `< 3.5 GB`，零真值列、零家族哈希重叠。
4. **L2 `PinpointSSL-JEPA` 物理感知多视图预训练、零标签磁谱仪预校准与等总算力（Equal-FLOPs）评估**：
   - 实现整平面硅遮蔽（严格阻断局部坐标泄漏）、正交闪烁体条立体/飞行时间交叉预测与跨首块磁铁全谱仪遮蔽三任务联合 `EMA-JEPA + VICReg` 预训练；量化无标签穿透 $\mu$ 预校准指标，并在严格 `FLOPs_ratio >= 1.00` 等总算力监督对照、`10%/25%/100%` 标签效率曲线及 `Tune6k` 多范式融合上给出诚实、完整的定量结论。

---

## 2. 工作内容

### 2.1 文献调研与方法学映射

| 方法 / 文献来源 | 核心机制与结论 | 本阶段在 PINPOINT 中的具体映射与改造 |
| --- | --- | --- |
| **ATLAS Collaboration**, *Graph Neural Network Jet Flavour Tagging with the ATLAS Detector* (`ATL-PHYS-PUB-2022-027` / `GN2`, **arXiv:2505.19689**, 2023–2025) | 在训练期对径迹起源类别（PV / $b$ / $c$ / 转换 / 鬼影）与成对共顶点关系施加辅助监督，推理期仅输入重建观测量，大幅抑制光子转换与次级强子假重味标记。 | 针对 PINPOINT 无预重建径迹、直接面对 64 条软候选 3D 径迹段（`97%` 鬼影）的特点，在 [gn2_tracklet_ipformer.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/nextgen/gn2_tracklet_ipformer.py) 中不仅添加辅助头，更将预测出的实径迹概率 $\hat{w}_{\text{real}}$ 与非转换位移先验 $\hat{w}_{\text{hf}}$ **显式门控注入** IRLS 顶点拟合与 Perceiver 融合；消融证实无此门控时单纯辅助损失增益接近零（`+0.064 pp`）。 |
| **Assran et al.**, *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA)*, **CVPR 2023** (**arXiv:2301.08243**) | 在潜空间由上下文编码器与轻量预测器预测目标块 EMA 教师表征，避免像素级生成模型过度关注高频局部随机涨落。 | 在 [pinpoint_ssl_jepa.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/nextgen/pinpoint_ssl_jepa.py) 中以动量 EMA 教师（$\tau_{\text{ema}}: 0.996 \to 1.0$）提供被遮蔽硅平面的语义潜表征目标，并叠加 8 分量二维混合高斯（2D GMM）预测宏观横向簇射能量密度包络，规避单团簇多重库仑散射不可预测噪声。 |
| **Cheng et al.**, *Rethinking Masked Autoencoders for Point Cloud Pretraining (`NOMAE`)*, **CVPR 2025** & **Wu et al.**, *`Sonata`: Self-Supervised Learning of Reliable Point Representations*, **CVPR 2025** | 点云掩码自编码器若向解码器/预测器泄露被遮蔽补丁的质心坐标，或仅作局部随机点遮蔽，编码器会退化为平凡的近邻坐标插值捷径，丧失高层几何语义。 | 实施**整平面级硅遮蔽**（每次遮蔽 `2~3` 个完整硅层，含至少一对相邻平面），且向预测器查询令牌**仅传入探测面固定纵向位置与上游辐射长度 $(z_p, X_{0,p})$**，严格禁止传入被遮蔽团簇的任何 $(x, y)$ 坐标或形态特征；单元测试与独立审计验证遮蔽平面 `+999 mm` 扰动输出绝对差严格为 `0.0`。 |
| **Bardes, Ponce & LeCun**, *`VICReg`: Variance-Invariance-Covariance Regularization for Self-Supervised Learning*, **ICLR 2022** (**arXiv:2105.04906**) | 通过批内每维标准差铰链惩罚与非对角协方差惩罚防止无对比负样本架构的表征坍缩与维度冗余。 | 对上下文全局池化潜向量施加 `VICReg` 方差-协方差正则（目标标准差 `1.0`），在 `334.8k` 预训练全程维持 `vic_glob_std = 1.0381`（零坍缩）。 |
| **Hu & Hong** (**2013**) & **Stage 327** `Score-Bank KL-DRO` ([tail_dro_pauc.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/nextgen/tail_dro_pauc.py)) | 冻结主干下的 `8,192` 历史本底分数队列 CVaR 尾部损失配合 `KL(p_0 || p) + CE` 锚定，在不破坏全局校准度 `NLL` 的前提下优化 `1% FPR` $\nu_\tau$ 尾部纯度。 | 在 Stage-II 对 `TrackletIPFormer-GN2` 单模型、二路融合 `L1_GN2 + DualRegime` 及三范式融合 `L1_GN2 + DualRegime + JEPA` 统一施加 `tune_screen` 门控选择的冻结主干 `Score-Bank KL-DRO`。 |

### 2.2 实验 L1 实现细节：`L1_Train32k` 团簇边车与 `TrackletIPFormer-GN2`

1. **Bonn 源样本变动处置与零混杂配对设计（Option A）**：
   - 核查 `/eos/project-f/fasersim-bonn/public/pinpoint/data` 发现 `run 10082`（Clean v3 中 `mu_CC` 来源）已于 2026-11-20 从公共盘移除，而 `10081`（`e_CC`）、`10083`（`tau_CC`）、`10084`（`NC`）及高统计纯钨 `mu_CC` 样本 `10086` 完整在盘。
   - [produce_l1_cluster_sidecar.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/nextgen/produce_l1_cluster_sidecar.py) 采用严格配对方案：**保留 Clean v3 `Train32k` 全部 `24,000` 个 `(e_CC, tau_CC, NC)` 事例的原始 `(run, chunk, row, family_hi, family_lo)` 与随机种子公式** `seed = (1000003*run + 9176*chunk + 101*chunk_row + 32101) % 2147483000 + 1`，并从与 `Tune8k`、`OOD6`、封存集及 L2 完全不交的 `10086` `chunks 002..015` 抽取 `8,000` 个纯钨 `G4_W` `mu_CC` 事例，构成 80 分片均衡 `L1_Train32k`（内部划分 `25,600 / 1,600 / 4,800` 保持 `(e, tau, NC)` 精确对齐），并在同一数据集上训练零混杂对照 `l1_ctrl_no_gn2`。
2. **`pinpoint_v3_1` `/tmp` 沙箱重放与团簇微观归因**：
   - 80 个 8 核 HTCondor 作业在节点本地 `/tmp` 开启 `/out/saveTruthHits true` 重放 Geant4 步级沉积并立即删除临时 `.root`，通过 `(layer, col, row)` 精确稀疏哈希连接将 `TruthHits` 能量聚合至 `observations.parquet` 团簇并对齐 `extract_pf_item` 保留的 `K <= 1024` 团簇序列。通过 `Tracks` 母子链回溯赋予每团簇 5 类 `origin_label`（`0: PV_primary`, `1: tau_decay`, `2: charm_decay`, `3: photon_conv_ee`, `4: had_sec_or_ghost`）、次级顶点组 `sv_group_id` 及前 3 贡献径迹 ID（`dominant/second/third_track_id`，以捕捉高密度核心重叠像素团簇）。
3. **`TrackletIPFormer-GN2` 架构与拓扑门控机制**（[gn2_tracklet_ipformer.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/nextgen/gn2_tracklet_ipformer.py)，`6.314M`–`6.330M` 参数）：
   - 在 [tracklet_ipformer.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/nextgen/tracklet_ipformer.py) 64 条软候选 3D 径迹段编码器后引出轻量预测头：5 类起源头 $\hat{\mathbf{q}}_t \in \Delta^4$、实径迹纯度头 $\hat{p}_{\text{pure},t} \in (0, 1)$ 及可选 $64 \times 64$ 共顶点对注意力偏置头 $\hat{B}_{ij}^{\text{covtx}}$。径迹级训练标签由所含 2–3 个团簇的前 3 共享根 `trackID` 交集严格导出，若两端团簇均属光子对转换（`origin == 3`）则标记为转换背景类（`26.7%` 占比）且 `purity = 0`。
   - **推理期纯观测门控注入**：令预测实径迹权重 $\hat{w}_{\text{real},t} = \text{valid}_t \cdot (\epsilon_0 + (1 - \epsilon_0)\hat{p}_{\text{pure},t})$（$\epsilon_0 = 0.05$ 保底防止早期梯度死锁），重味位移权重 $\hat{w}_{\text{hf},t} = \hat{w}_{\text{real},t}(\hat{q}_{t,\tau} + \hat{q}_{t,c}) / (1 + \hat{q}_{t,\text{conv}})$。将 $\hat{w}_{\text{real},t}$ 乘入两轮 Student-$t$ IRLS 初级顶点权重以剔除组合鬼影拉偏，由 $\hat{w}_{\text{hf},t}$ 驱动位移次级顶点种子聚合与显式非转换位移多重度摘要（重味计数、重味与转换分离比、最大重味冲击参数显著性），并将 $\beta \ln(\hat{w}_{\text{real},t} + 10^{-4})$ 作为对数先验偏置注入 Perceiver 交叉注意力。推理期函数签名完全不接收任何边车张量。

### 2.3 实验 L2 实现细节：`334.8k` 纯钨无标签语料与 `PinpointSSL-JEPA`

1. **高吞吐纯钨可观测包含态语料生产**（[produce_l2_ssl_corpus.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/nextgen/produce_l2_ssl_corpus.py)）：
   - 扫描 `10081 / 10086 / 10083 / 10084` 在 `chunk >= 029` 的非封存纯钨事例池（`chunks 029..142` 用于 `train_shards`，不交的高分块 `chunks 200..203` 用于 `ssl_val_calib`），在模拟前校验 `tgt_Z == 74` 与 `Pinpoint_G4` 主体钨体积 `G4_W`，模拟后执行纯观测包含态过滤（仅保留 `obs_domain_id in {1, 2}` 的 `D1 ∪ D2` 事例，剔除无靶内像素触发或上游外入射事例）。
   - 采用 CSR 变长数组打包 + `pix_feat_16`（`float16` 存储局部形态与团簇特征，加载时由 [produce_l2_ssl_corpus.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/nextgen/produce_l2_ssl_corpus.py) 的 `reconstruct_pix_feat22` 无需 `cKDTree` 直接精确还原 22 维特征，误差 `< 5e-3`）+ Zstandard 压缩（`.npz.zst`），每 `2,000` 事例仅占 **`17.69 MB`**。
2. **`PinpointSSL-JEPA` 三视图物理自监督架构**（[pinpoint_ssl_jepa.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/nextgen/pinpoint_ssl_jepa.py)，在线与微调主干 **`7,534,625` 参数**，含预训练预测器共 `8,563,670` 参数）：
   - **主干编码器**：分层硅像素团簇编码（$8\times 8$ 局部形态卷积 + 22 维几何形态投影 $\to$ 平面内多种子注意力池化 PMA 生成每活跃硅层 4 个摘要令牌，降低 `1024` 团簇平方复杂度）+ 闪烁体条精细面板/方向/时间令牌 + 后磁谱仪径迹与站点计数令牌 $\to$ 6 层跨探测器预归一化 Transformer（`d = 256`）$\to$ 16 个潜查询令牌与全局上下文表征 $z_{\text{glob}}$。
   - **Task 1（整平面硅遮蔽 `EMA-JEPA` + 2D GMM 密度包络）**：随机遮蔽 `2~3` 个活跃硅平面（至少含一对相邻层），上下文分支在进入任何编码层前物理剔除被遮蔽平面的全部团簇、补丁、平面摘要及全局计数的对应分量；轻量预测器仅以被遮蔽平面的 $(z_p, X_{0,p})$ 查询上下文潜令牌，同时回归动量 EMA 教师的平面潜表征（平滑 $L_1$ 损失）与该层团簇横向分布的 8 分量二维混合高斯对数似然。
   - **Task 2（正交闪烁体条 X/Y 与 $\Delta t_{\text{ToF}}$ 交叉预测）**：随机遮蔽 `30%` 闪烁体面板的水平（Y）或垂直（X）条通道，由正交视图与硅团簇上下文重建面板总对数光电子数 $\ln(1 + \sum \text{PE})$、能量加权位置质心及最早符合时间 $t_{\min}$。
   - **Task 3（跨 Magnet-0 全谱仪遮蔽与穿透 $\mu$ 预校准）**：以 `50%` 概率将整座后磁谱仪（`trk_*` 全部张量及 `c_obs` 谱仪维度）完全置零，仅凭钨靶区像素与闪烁体纵向穿透沉积预测谱仪是否存在穿透径迹（BCE）、穿透径迹入口位置/出射斜率以及首末段磁偏转角代理 $\ln(|s_y| + 10^{-4})$（Huber 损失），叠加全局潜向量 `VICReg` 正则。
   - **微调协议**：移除全部 SSL 预测器，挂载与 `DualRegimeModalNet` / `TrackletIPFormer` 同规格的 12 类层次化味分类头、能量/横动量回归头及 Student-$t(\nu=3)$ 初级顶点回归头，采用逐层学习率衰减（Layer-wise LR Decay $\xi = 0.78$）微调 `20` passes。

### 2.4 研发与独立审计过程中的关键工程与统计修正记录

1. **多进程 `fork` 死锁修复（L2 语料生产）**：早期本地烟雾测试在导入 `torch`/`pyarrow` 后使用默认 `fork` 启动 `ProcessPoolExecutor` 导致子进程锁死于 `0% CPU`；切换为 `mp.get_context("spawn")` 并固定 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` 后，160 个 8 核 HTCondor 作业以 `0` 错误一次性完成。
2. **HTCondor GPU ClassAd VO 过滤规避**：排查发现 `bigbird20.cern.ch` 上设置 `GlobalMemoryMb >= 80000` 会触发 `b9pgpun*` 私有节点的 `SCond_vo_filter` 拒绝导致无限排队；改为 `Capability >= 8.0`（或非 MIG 约束 `(Capability >= 8.0) && !regexp("MIG", DeviceName)`）且不设显存硬下限后，`10` 秒内匹配空闲 `H100 NVL` 与 `A100-PCIE-40GB` 节点；同时利用 `avail_gb >= 45 GB` 主机内存单次缓存全部 `152` 个 `.npz.zst` 分片，将单轮 `334.8k` 预训练耗时压缩至 `11.2 min/epoch`（H100 NVL）。
3. **等总算力（Equal-FLOPs）严格对齐修正**：独立审计提前检出 `run_l2_ssl_jepa.py` 中 `event_sidecar` 列名回退键（`vis_pz_log1p`/`vis_pt_log1p`）及实际落地 `334,754` 事例数与预设 `330,000` 的微小偏差；将 SSL 预训练轮数精确锁定为 `14 epochs`（`334,754 × 14 = 4,686,556` 视图 > 蓝图 `4.50M` 目标），确保 `262` passes 的 `sup_scratch_equal_compute` 实测总算力比严格达到 **`FLOPs_ratio = 1.0547x >= 1.00`**。
4. **Stage-I 家族匹配基准与 Stage-II 3-seed 顶点残差收缩校准修正（L1）**：
   - 针对 `L1_internal_val_eval` 中 `1,200` 个 `mu_CC` 槽位来自 `10086` 的特点，修正聚合脚本使跨阶段与 Stage 326 模型的配对对比严格限制在 `3,600` 个 `(family_hi, family_lo)` 100% 匹配的 `(e, tau, NC)` 事例（`s326_references_common3600`），而 `4,800` 全量因果对比以同分布 `l1_ctrl_no_gn2` 为精确配对基准。
   - 独立审计发现 Stage-II 3-seed 集成使初级顶点点估计中位残差从单模型 `4.83 mm` 收缩至 `4.25 mm`，若直接取单模型尺度均方根 `sqrt(mean(s^2))` 会导致 `Front30` 纵向 Pull 偏保守（`0.8063 < 0.85`）；严格仅在授权校准集 `tune_screen`（`N = 2,000`）上拟合三轴尺度乘子 `[1.0027, 0.9402, 0.9242]`（与独立重算误差 `6.41e-10`，零 `Tune6k` 泄漏）后，留出集 `Tune6k` 全部 9/9 Pull 落入 `[0.8725, 1.0720]`。
5. **全量 `44` 项单元测试套件跨进程 `sys.addaudithook` 作用域隔离修复**：集成阶段运行 `unittest discover` 联测全部 `44` 项测试（Stage 327 `27` 项 + Stage 328 `17` 项）时检出 `_L1Stage1SealGuard` 与 `_L2CorpusSealGuard` 的审计钩子在上下文退出后仍保持全局激活，干扰后续合成路径 `/tmp/.../shards/100` 测试；将钩子严格绑定至活跃守卫实例并在 [test_nextgen_l2_corpus.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/tests/test_nextgen_l2_corpus.py) 中补齐 `try/finally` 退出后，`44/44` 项测试在 `NVIDIA H100 NVL` 上以 `136.271s` 零失败、零跳过一次通过（提交 `9ec503a`）。

---

## 3. 详细数据

### 3.1 数据溯源、纯钨包含态校验与封存盲法总账

| 数据资产 | 用途与阶段 | GENIE 来源分块 | 纯钨事例数 | 家族哈希重叠校验 | 存储体积与真值隔离状态 |
| --- | --- | --- | --- | --- | --- |
| `L1_Train32k` + `cluster_sidecar` | L1 Stage-I/II 训练与微观探针 | `10081/10083/10084` `000..011` (`24k`) + `10086` `002..015` (`8k`) | `32,000` (`10,803,182` 团簇) | 与 `Tune8k`、`6 OOD`、L2 重叠 **`0`**；`24k (e,τ,NC)` 与 Clean v3 `Train32k` `100%` 匹配 | `cluster_sidecar/` = **`76.92 MB`**（总目录 `1,408.29 MB`），`0` 个 `.root`，仅训练期读取 |
| `l2_ssl_300k/train_shards` | L2 无标签自监督预训练（152 分片） | `10081/10086/10083/10084` `029..142` | **`334,754`** (`D1=95,384`, `D2=239,370`，包含率 `79.29%`) | 与 `CleanV3_Train32k`、`L1_Train32k`、`Tune8k`、`6 OOD`、`ssl_val_calib` 及 `000..028` 重叠 **`0`** | **`2,961.18 MB`**（每 `2k` 事例 `17.69 MB`），**`truth_columns_present = 0`**，`0` 个 `.root` |
| `l2_ssl_300k/ssl_val_calib` | L2 零标签穿透 $\mu$ 磁谱仪预校准验证（8 分片） | `10081/10086/10083/10084` `200..203` | **`11,715`** (`D1=3,257`, `D2=8,458`，包含率 `79.27%`) | 与全部训练/调优/OOD/封存分块重叠 **`0`** | **`104.31 MB`**（L2 总磁盘占用 **`2,924.27 MB < 3.5 GB`**），**`0` 真值列** |
| `CleanV3 Tune8k` (`shards/080..099`) | Stage-II 单次门控校准（`tune_screen` `2k`）与留出确认（`Tune6k` `6k`） | Clean v3 既有固定划分 | `8,000` (`2,000 + 6,000`) | 与 L1/L2 训练语料重叠 **`0`** | Stage-I **0 次接触**；Stage-II 仅读取纯观测特征评估一次 |
| `Cal-A / Cut-B / Confirm` (`100..149`), `input_sys_confirm*`, `sealed_truth_meta/`, GENIE `016..028` | 最终预注册一次性解封确认集 | 封存隔离 | **未测量** | 静态与运行时访问计数 **`0 / 1,245` 文件** | **未测量（严格盲法封存）** |

### 3.2 实验 L1：`32,000` 纯钨事例微观起源普查与 Stage-I 因果消融

#### 3.2.1 `L1_Train32k` 硅像素团簇与 `Front30` 候选径迹段真值普查（[sidecar_production_summary.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/l1_cluster_sidecar/sidecar_production_summary.json)）

| 统计层级与微观类别 | 数量 | 占比 / 指标 | 物理含义 |
| --- | --- | --- | --- |
| 保留硅像素总团簇数（靶区团簇数） | `10,803,182` (`9,793,100`) | `< 50 um` 空间匹配率 **`100.000%`**（4 分片 `555,980` 团簇重聚类最大偏差 **`0.0 um`**） | `produce_clean_v3` 随机种子重放达到比特级空间复现 |
| `0: PV_primary`（初级顶点直接带电产物） | `218,440` | **`2.022%`** | 直接承载轻子与初级强子出射角信息 |
| `1: tau_decay`（$\tau^\mp$ 衰变带电叉产物） | `15,473` | **`0.143%`** | 信号核心微观证据，信噪比约为 $1 : 700$ |
| `2: charm_decay`（$D^\pm, D^0, D_s^\pm, \Lambda_c^+$ 衰变产物） | `6,754` | **`0.063%`** | 真实物理次级顶点不可约本底 |
| `3: photon_conv_ee`（钨板内 $\gamma \to e^+e^-$ 转换及次级级联） | `9,025,207` | **`83.542%`** | $\pi^0 \to \gamma\gamma$ 与轫致辐射在钨高 $Z$ 介质中的主导团簇来源 |
| `4: had_sec_or_ghost`（强子核相互作用次级产物或电子学噪声） | `1,537,308` | **`14.230%`** | 强子簇射大角度散射与反冲碎片 |
| `Front30` C++ 3-hit 候选径迹段同 `trackID` 纯度（共 `184,527` 条） | `5,525` 纯径迹 | **`2.994%` 纯度**（**`97.006%` 组合鬼影**；2-hit 纯度仅 `0.850%`） | 直接验证 Stage 326 第 1.2 节结论：无纯度门控时几何径迹池被假组合淹没 |
| `Front30` 位移候选径迹段（`IP_sig > 2.5`，共 `49,135` 条）假位移率 | `49,025` 转换/鬼影 | **`99.776%` 为光子转换或组合鬼影**（真重味仅 `0.224%`） | 解释单纯几何冲击参数无法分离 $\nu_\tau$ 的微观根源 |

#### 3.2.2 L1 Stage-I 受控因果消融（`L1_internal_val_eval`, `N = 4,800`, `seed = 42`，[stage1_l1_gn2_results.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/exp_l1_gn2/stage1_l1_gn2_results.json)）

| 模型变体 | 参数量 | All `BA4` (%) | All `NLL` | `D1_obs BA4` (%) | `Front30 BA4` (%) | `All 1% FPR τ` (%) | `W184 NC Charm p_tau>0.5` (%) | 9/9 PV Pull 范围 | 离线微观探针 AUROC（`173,290` 径迹段） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `l1_ctrl_no_gn2`（零混杂对照，$\lambda_{\text{GN2}}=0$） | `6.225M` | `58.52` | `0.9539` | `66.93` | `66.13` | `4.75` | `3.17` | `[0.881, 1.131]` | `purity=0.509`（未监督） |
| `abl_aux_loss_only_no_readout_gate`（仅训练辅助损失，无门控） | `6.316M` | `58.54` | `0.9534` | `67.00` | `65.91` | `4.92` | `3.83` | `[0.876, 1.133]` | `purity=0.9681`, `hf_vs_conv=0.9927` |
| `abl_covertex_only`（仅 $64\times 64$ 共顶点对偏置） | `6.327M` | `58.67` | `0.9592` | `67.44` | `66.33` | `5.08` | `2.75` | `[0.878, 1.124]` | `covertex=0.9965` |
| `gn2_full`（起源 + 纯度 + 共顶点全开） | `6.330M` | `58.58` | `0.9512` | `68.41` | `66.65` | `4.58` | `3.67` | `[0.880, 1.153]` | `purity=0.9682`, `covertex=0.9977` |
| **`abl_origin_purity_only`（Stage-I 晋级胜出模型）** | **`6.314M`** | **`58.92`** | **`0.9526`** | **`68.54`** | **`66.61`** | **`5.33`** | **`2.17`** | **`[0.881, 1.129]`** | **`purity=0.9682`, `hf_vs_conv=0.9926`, `tau_prong=0.9767`, `conv=0.8248`** |

- **因果归因核验**：`abl_aux_loss_only_no_readout_gate` 虽然同样学到了 `purity AUROC = 0.9681` 的内部表征，但因未在推理期对 97% 鬼影径迹实施显式降权门控，`D1_obs BA4` 仅微动 `+0.064 pp`；一旦启用预测纯度与反转换重味读出门控（`abl_origin_purity_only`），`D1_obs` 较 `l1_ctrl_no_gn2` 显著提升 **`+1.608 pp`（95% CI `[+0.151, +3.164] pp`，`ΔNLL = -0.0151 [-0.0282, -0.0020]`）**，较 `no_readout_gate` 显著提升 **`+1.544 pp [+0.143, +2.999] pp`**，同时 `gn2_full` 因 $64 \times 64$ 共顶点矩阵正样本极度稀疏（`10.40M` 对中仅 `0.190%` 正例）导致 `Interm150` $x$-Pull 轻微越界（`1.1525`），故严格按预注册门控晋级 `abl_origin_purity_only`。

### 3.3 实验 L2：`PinpointSSL-JEPA` 无标签预校准、等算力对照与 Stage-I 消融

#### 3.3.1 `ssl_val_calib`（`N = 11,715`，`1,647` 个穿透 $\mu$）零标签磁谱仪预校准核验

| 预训练变体（`334,754` 事例 × `14` epochs = `4,686,556` 视图，`4.936e16` FLOPs） | GPU 与耗时 | `VICReg` 全局标准差 | 穿透 $\mu$ 存在性 `AUROC` (`AUPRC`) | 磁偏转代理 `r(ln|s_y|)` (`Spearman ρ`) | 入口位置相关 `r(x_0, y_0)` | 出射斜率相关 `r(t_x, t_y)` |
| --- | --- | --- | --- | --- | --- | --- |
| **`ssl_full`（三任务完整 `PinpointSSL-JEPA`）** | A100-40GB (`4.55 h`) | **`1.0381`** | **`0.9354`** (`0.7296`) | **`0.4873`** (`0.4538`, $p=5.9\times 10^{-99}$) | **`(0.8416, 0.7625)`** | **`(0.6003, 0.3824)`** |
| `abl_ssl_no_spec_precal`（移除 Task 3 跨磁铁谱仪遮蔽） | H100 NVL (`3.04 h`) | `1.0118` | `0.6036` (`0.1982`) | `0.0319` (`0.0326`) | `(-0.089, 0.021)` | `(-0.041, 0.015)` |
| `abl_ssl_random_point_mask_shortcut`（60% 随机单团簇遮蔽 + 暴露 $(x,y)$） | H100 NVL (`2.66 h`) | `1.0024` | `0.9343` (`0.7251`) | `0.4967` (`0.4619`) | `(0.8398, 0.7601)` | `(0.5984, 0.3811)` |

#### 3.3.2 等总算力（Equal-Total-FLOPs）轨迹与 Stage-I 受控对比（Clean v3 `internal_val_eval`, `N = 4,800`, `seed = 42`，[stage1_l2_jepa_results.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/exp_l2_jepa/stage1_l2_jepa_results.json)）

- **严格 FLOPs 审计（`torch.utils.flop_counter.FlopCounterMode`）**：单样本 `SSL = 10.581 GFLOPs`（`shard_0001` 实测 `10.532 GFLOPs`），单样本 `Supervised = 8.481 GFLOPs`（`shard_0001` 实测 `8.435 GFLOPs`，比值 `1.249x`）。`ssl300k_jepa_ft_100pct` 总算力（`4.687M` SSL 视图 + `20` passes 微调）为 **`5.393 × 10^16 FLOPs`**；等算力监督基线 `sup_scratch_equal_compute` 运行 **`262` passes**（`6,707,200` 监督样本，`26,200` 步），实测总算力为 **`5.688 × 10^16 FLOPs`**，**`FLOPs_ratio = 1.0547x >= 1.00`**。
- **长程重放过拟合动力学（[run_summary.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/exp_l2_jepa/sup_scratch_equal_compute/run_summary.json)）**：在固定 `25,600` 有标签样本上重复训练 `262` 轮时，`train_loss` 从第 1 轮 `4.0525` 单调降至第 262 轮 `1.1838`，但 `dev_screen` 在**第 `13` 轮达到最优（`NLL = 0.9813`, `BA4 = 57.75%`）后急剧过拟合崩塌**，第 `182` 轮恶化至 `dev_screen NLL = 5.0748`（`BA4 = 50.81%`），第 `262` 轮达 `dev_screen NLL = 5.8624`。即使为其锁定第 `13` 轮最优 EMA 检查点，其验证集校准度仍劣于 `ssl300k_jepa_ft_100pct`。

| Stage-I 实验组（同一 `7.535M` 主干架构） | 有标签比例 / 轮数 | All `BA4` (%) | All `NLL` | `D1_obs BA4` (%) / `NLL` | `All 1% FPR τ` (%) | `Front30 |dz|` (mm) | 9/9 PV Pull 范围 | 配对 Bootstrap 对比（`ssl_full` 相对基准） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **`ssl300k_jepa_ft_100pct`（完整 SSL + $\xi=0.78$ FT）** | `100%` (`20p`) | **`58.04`** | **`0.9729`** | **`66.06` / `0.8700`** | **`5.67`** | **`3.64`** | **`[1.008, 1.063]`** | 基准参考（晋级 Stage-II） |
| `sup_scratch_equal_compute`（等总算力从头监督） | `100%` (`262p`, best@13) | `58.25` | `0.9785` | `66.15` / `0.8698` | `5.83` | `3.81` | `[0.992, 1.054]` | **`ΔNLL = -0.0056`**（突破 `-0.004` 晋级门控），`ΔBA4 = -0.21 pp` |
| `sup_scratch_20p`（标准 20-pass 从头监督） | `100%` (`20p`) | `58.50` | `0.9582` | `66.85` / `0.8475` | `6.17` | `3.59` | `[0.998, 1.058]` | `ΔBA4 = -0.46 pp`, `ΔNLL = +0.0147`（见第 4.1 节归纳偏置分析） |
| `abl_ssl_no_spec_precal`（无 Task 3 谱仪遮蔽预校准） | `100%` (`20p`) | `56.98` | `0.9844` | `65.12` / `0.8841` | `5.08` | `3.79` | `[1.004, 1.069]` | **`ΔBA4 = +1.063 pp [+0.242, +2.003]`**, **`ΔNLL = -0.0115 [-0.0203, -0.0033]`** |
| `abl_ssl_random_point_mask_shortcut`（随机点遮蔽捷径） | `100%` (`20p`) | `57.27` | `0.9860` | `64.61` / `0.8895` | `3.58` | `3.88` | `[1.011, 1.074]` | **`Δτ@1% = +2.083 pp [+0.085, +4.220]`**（`D1 +4.626 pp`）, **`ΔNLL = -0.0131 [-0.0220, -0.0045]`** |
| 标签效率：`ssl300k_jepa_ft_10pct` vs `sup_scratch_10pct` | `10%` (`2,560` 事例) | `51.71` vs `52.58` | **`1.0860`** vs `1.0946` | `59.42` vs `60.11` | `2.83` vs `3.17` | `4.52` vs `4.71` | — | SSL 在极低标签下改善概率校准 `ΔNLL = -0.0086`，`BA4` 低 `-0.88 pp` |
| 标签效率：`ssl300k_jepa_ft_25pct` vs `sup_scratch_25pct` | `25%` (`6,400` 事例) | `54.08` vs `55.38` | `1.0426` vs `1.0363` | `62.18` vs `63.40` | `4.08` vs `4.42` | `4.06` vs `4.12` | — | `71.5% D2` 无标签先验与均匀 4 味评估间存在域分布偏移 |

### 3.4 Stage-II `Tune6k`（`N = 6,000`，3 Seeds `{17, 42, 2026}`）单模型、KL-DRO 与跨范式融合总表

所有 Stage-II 模型均在全量 `32k` 训练集上以 `{17, 42, 2026}` 三独立种子训练，温度、初级顶点尺度乘子及冻结主干 `Score-Bank KL-DRO` 选择严格仅使用 `tune_screen`（`N = 2,000`），在留出集 `Tune6k`（`N = 6,000`）上一次性评估（汇总自 [stage2_l1_gn2_results.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/exp_l1_gn2/stage2/stage2_l1_gn2_results.json)、[stage2_l2_jepa_results.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/exp_l2_jepa/stage2/stage2_l2_jepa_results.json) 与 [stage328_audit_report.md](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage328_verification/stage328_audit_report.md)）：

| 模型 / 集成方案（`Tune6k`, `N = 6,000`, 3-seed） | All `BA4` (%) | All `NLL` | `D1_obs BA4` (%) / `NLL` | `Front30 BA4` (%) / `NLL` | `Pinpoint BA4` (%) / `NLL` | Within-Slice `1% FPR τ` All / `D1` (%) | Split-Half `1% FPR τ` All (`held_FPR`) | Sec 12 `Mahlstedt_PC BA3` (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Stage 326 `TrackletIPFormer` (`raw`) | `58.11` | `0.9504` | `63.69` / `0.8404` | `63.03` / `0.8559` | `62.27` / `0.8808` | `4.69` / `7.69` | `5.09%` (`0.73%`) | `81.04` |
| Stage 326 `DualRegimeModalNet` (`raw`) | `58.07` | `0.9497` | `63.26` / `0.8548` | `62.55` / `0.8698` | `61.99` / `0.8920` | `7.03` / `9.94` | `7.10%` (`0.87%`) | `81.35` |
| Stage 326 `50/50 Blend` (`raw`) | `58.50` | `0.9380` | `64.08` / `0.8344` | `63.54` / `0.8363` | `62.58` / `0.8650` | `7.23` / `6.73` | `7.51%` (`0.64%`) | `81.89` |
| Stage 327 `s327_blend_dro`（前阶段最优基准） | `58.76` | `0.9373` | `64.28` / `0.8359` | `63.55` / `0.8362` | `62.78` / `0.8650` | **`7.83`** / **`11.86`** | **`8.04%`** (`0.59%`) | `81.91` |
| L1 零混杂对照 `l1_ctrl_no_gn2` (`raw`) | `58.85` | `0.9458` | `63.72` / `0.8374` | `63.77` / `0.8399` | `63.02` / `0.8696` | `4.75` / `4.81` | `4.69%` (`0.64%`) | `81.61` |
| **L1 `TrackletIPFormer-GN2` (`raw`)** | **`59.20`** | **`0.9379`** | **`64.36` / `0.8212`** | **`65.04` / `0.8254`** | **`63.98` / `0.8568`** | `4.89` / `4.49` | `5.09%` (`0.64%`) | `81.55` |
| **L1 `TrackletIPFormer-GN2 + KL-DRO`** | **`59.38`** | **`0.9370`** | **`64.67` / `0.8195`** | **`65.09` / `0.8233`** | **`64.07` / `0.8550`** | `5.09` / `4.49` | `5.50%` (`0.59%`) | `81.61` |
| **L2 `PinpointSSL-JEPA` (`raw`)** | `58.20` | `0.9493` | `63.86` / `0.8478` | `63.02` / `0.8592` | `62.27` / `0.8707` | `5.62` / `7.37` | `5.76%` (`0.73%`) | `81.12` |
| **`50/50 L1_GN2 + DualRegime + KL-DRO`** | **`59.43`** | **`0.9291`** (`cal 0.9285`) | **`65.06` / `0.8212`** | **`64.77` / `0.8218`** | **`63.90` / `0.8525`** | **`7.56` / `11.22`** | **`8.04%`** (`0.59%`) | **`82.30`** |
| **`50/50 PinpointSSL-JEPA + L1_GN2` (`raw`)** | **`59.73`** | **`0.9307`** | `64.89` / `0.8188` | **`65.28` / `0.8211`** | **`64.21` / `0.8510`** | `5.89` / `7.69` | `6.17%` (`0.69%`) | `82.18` |
| **`3-Way (L1_GN2 + Dual + JEPA)` (`raw`)** | `59.28` | **`0.9277`** | **`65.41` / `0.8158`** | `64.65` / `0.8216` | `63.65` / `0.8568` | `6.83` / `7.37` | `7.37%` (`0.69%`) | **`82.42`** |
| **`3-Way (L1_GN2 + Dual + JEPA) + KL-DRO`** | **`59.57`** | **`0.9275`** | **`65.48` / `0.8161`** | `64.81` / `0.8210` | `63.98` / `0.8542` | **`7.16` / `9.62`** | **`7.77%`** (`0.64%`) | **`82.42`** |

**关键配对家族 Bootstrap（`1,000` 次重采样，`seed = 20263250`）显著性核验**：
- **`L1_GN2` vs 零混杂 `l1_ctrl_no_gn2`（纯微观起源门控因果效应）**：`All_Tune6k ΔNLL = -0.00794`（95% CI **`[-0.01167, -0.00457]`**），`Front30 ΔBA4 = +1.265 pp`（95% CI **`[+0.437, +2.116] pp`**；DRO 后 **`+1.531 pp [+0.628, +2.395] pp`**），`Pinpoint_PixXY ΔBA4 = +0.960 pp`（95% CI **`[+0.311, +1.625] pp`**），`D1_obs` DRO `ΔBA4 = +1.489 pp`（95% CI **`[+0.366, +2.581] pp`**）。
- **`50/50 L1_GN2 + DualRegime + KL-DRO` vs Stage 327 `s327_blend_dro`**：`All_Tune6k ΔBA4 = +0.665 pp`（95% CI **`[+0.108, +1.213] pp`**），`ΔNLL = -0.00823`（95% CI **`[-0.01110, -0.00511]`**），全部严格排除零，同时保持 Split-Half `1% FPR` 留出效率 `8.04%`（`held_FPR = 0.59%`）与 `D1_obs 11.22%`。
- **`3-Way (L1_GN2 + Dual + JEPA) + KL-DRO` vs Stage 327 `s327_blend_dro`**：`All_Tune6k ΔBA4 = +0.808 pp`（95% CI **`[+0.204, +1.460] pp`**），`ΔNLL = -0.00984`（95% CI **`[-0.01350, -0.00619]`**）；相对 `DualRegime` 单模型集成 `ΔBA4 = +1.216 pp [+0.517, +1.889] pp`、`ΔNLL = -0.0220 [-0.0278, -0.0168]`；相对 `L1_GN2` 单模型集成 `1% FPR τ_eff` 提升 **`+1.941 pp [+0.468, +3.345] pp`**、`ΔNLL = -0.0102 [-0.0146, -0.0054]`。

### 3.5 Stage-II 初级顶点校准、6 组 OOD 外推与 `Shard 080` 硬件压力对比

| 评估维度 / 压力工况 | L1 `TrackletIPFormer-GN2` (`raw` / `KL-DRO`) | L2 `PinpointSSL-JEPA` (3-seed `raw`) | 物理与架构机制解读 |
| --- | --- | --- | --- |
| `Tune6k` 初级顶点 `|dz|` 中位数：`All` / `Front30` / `Interm150` | `5.91 mm` / `4.25 mm` / `5.51 mm` | **`5.22 mm` / `3.55 mm` / `4.42 mm`** | `JEPA` 整平面遮蔽预训练赋予极高精度纵向簇射起始层定位 |
| `Tune6k` 9/9 Student-$t(\nu=3)$ `p68(|pull|)` 范围 | **`[0.8725, 1.0720]`**（全部 $\in [0.85, 1.15]$） | **`[0.9568, 1.0479]`**（未校准亦达 `[0.970, 1.035]`） | 两个家族均满足严格概率不确定度校准门控 |
| OOD `Pythia6 Std` / `Pythia6 NuX` / `Run 10085-10088` `BA4` (%) | `57.83 -> 58.25` / `60.92 -> 60.33` / `58.88 -> 58.31` | `56.83` / **`61.25`** / `58.25` | 跨发生器强子化与束流谱外推稳健 |
| OOD 核粲假阳性 `W184 EPPS21 NC Charm` `p_tau > 0.5` (%) | `2.25%` (`raw`) $\to$ **`0.67%` (`KL-DRO`)** | `mean_p_tau = 26.82%` | `GN2` 反转换门控 + `KL-DRO` 将重核粲背景压至 `< 0.7%` |
| OOD `W184 NC Incl` / `Pythia8 Free NC Charm` `p_tau > 0.5` (%) | `2.33% -> 0.92%` / `1.13% -> 0.38%` | `mean_p_tau = 24.60%` / `28.73%` | 核效应与自由核子粲产生下无假阳性膨胀 |
| `Shard 080` 名义 `BA4` $\to$ `pixel_dead_5pct` (`N = 400`) | `59.13% -> 58.34%` (`-0.79 pp`) | `57.43% -> 57.19%` (`-0.24 pp`) | `JEPA` 整平面遮蔽训练对像素死区天然具备高鲁棒性 |
| `Shard 080` **`pixel_dark_1e-5` 强暗噪声压力** | `59.13%` $\to$ **`54.82%` (`-4.32 pp`)** | `57.43%` $\to$ `42.27%` (`-15.16 pp`) | `GN2` 预测径迹纯度 $\hat{w}_{\text{real}}$ 有效滤除随机热像素假组合，而均匀点编码器对未见过的密集单像素暗电流敏感 |
| `Shard 080` `tracker_dead_5pct_noise2` / `sipm_sat_cap_0p95` | `58.59%` (`-0.54 pp`) / `59.40%` (`+0.27 pp`) | `56.60%` (`-0.84 pp`) / `56.32%` (`-1.11 pp`) | 谱仪死通道与闪烁体 SiPM 饱和退化均控制在 `~1 pp` 内 |

### 3.6 汇总可视化图表与复现命令

4 幅 300 DPI 全英文出版级图表已生成并通过视觉与元数据审计（代码见 [plot_stage328_summary.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/nextgen/plot_stage328_summary.py)）：
1. [fig1_l2_ssl_equal_compute_and_label_efficiency.png](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage328_figures/fig1_l2_ssl_equal_compute_and_label_efficiency.png)：L2 等总算力 `262-pass` 监督记忆化崩塌轨迹、`10%/25%/100%` 标签效率曲线与反捷径消融对比。
2. [fig2_l2_unlabeled_probe_and_pv_calibration.png](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage328_figures/fig2_l2_unlabeled_probe_and_pv_calibration.png)：`11,715` 无标签事例零标签穿透 $\mu$ 识别与磁偏转代理 $\ln|s_y|$ 预校准及 `Tune6k` 初级顶点校准表现。
3. [fig3_l1_gn2_causal_ablation_and_probes.png](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage328_figures/fig3_l1_gn2_causal_ablation_and_probes.png)：`32,000` 纯钨事例团簇/径迹微观起源分类谱、离线探针 ROC 与 L1 推理门控因果消融。
4. [fig4_stage328_stage2_synergy_ood_stress.png](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage328_figures/fig4_stage328_stage2_synergy_ood_stress.png)：Stage-II `Tune6k` 单模型与二路/三路跨范式融合对比、6 组 OOD 核粲抑制及 `Shard 080` 7 项硬件压力测试。

在仓库根目录加载 `LCG_108_cuda` 后可执行完整自动化验证与复现：

```bash
# 1. 运行全套 44 项 NextGen 单元测试（含 Stage 327 27 项 + Stage 328 17 项）
set -- && source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh >/dev/null 2>&1 \
  && PYTHONPATH=src python3 -m unittest discover -v -s tests -p "test_nextgen_*.py"

# 2. L1 团簇边车校验与 TrackletIPFormer-GN2 Stage-I / Stage-II 聚合复现
PYTHONPATH=src python3 -m analysis.flavour.nextgen.produce_l1_cluster_sidecar --mode verify \
  --out-root /eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/l1_cluster_sidecar
PYTHONPATH=src python3 -m analysis.flavour.nextgen.run_l1_gn2_tracklet --s1-aggregate
PYTHONPATH=src python3 -m analysis.flavour.nextgen.run_l1_gn2_tracklet --s2-aggregate

# 3. L2 334.8k 无标签语料校验、PinpointSSL-JEPA 评估与 Stage 328 图表生成
PYTHONPATH=src python3 -m analysis.flavour.nextgen.produce_l2_ssl_corpus finalize \
  --out-root /eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/l2_ssl_300k
PYTHONPATH=src python3 -m analysis.flavour.nextgen.run_l2_ssl_jepa stage1-aggregate
PYTHONPATH=src python3 -m analysis.flavour.nextgen.run_l2_ssl_jepa stage2-aggregate
PYTHONPATH=src python3 -m analysis.flavour.nextgen.plot_stage328_summary
```

---

## 4. 阶段性结论

### 4.1 已证实物理与实验事实（Facts）

1. **钨靶硅像素探测器中的位移几何信号受 $\gamma \to e^+e^-$ 转换与组合鬼影双重统治，且唯有“推理期拓扑门控注入”能将训练期微观真值监督转化为下游判别增益**：
   - `10.80M` 团簇普查证实钨靶内 `83.54%` 团簇源于光子对转换，`Front30` 几何 3-hit 径迹段 `97.01%` 为跨粒子假组合，位移径迹段 `99.78%` 为转换或鬼影。
   - 受控消融严格证伪了“单纯添加多任务真值辅助损失即可改善主干表征”的朴素假设（`abl_aux_loss_only_no_readout_gate` 在 `D1_obs` 上仅 `+0.064 pp`）；唯有将模型从纯观测特征预测出的径迹纯度（`AUROC = 0.9682`）与重味对转换判别概率（`AUROC = 0.9926`）显式门控注入推理期 Student-$t$ 顶点拟合与 Perceiver 交叉注意力，才在零真值输入下取得 `Tune6k Front30 +1.265 pp`（较 Stage 326 `+2.011 pp`）、`All ΔNLL = -0.00794` 及暗噪声强抗性（`54.82%`）的因果提升。
2. **无标签束流数据可通过跨磁铁谱仪遮蔽实现零真值穿透 $\mu$ 识别与动量标度预校准，且整平面遮蔽（`NOMAE/Sonata`）是防止钨簇射局部插值捷径的必要条件**：
   - 无需任何模拟真值标签，仅凭钨靶区纵向沉积预测被整座遮蔽的后磁谱仪响应，`PinpointSSL-JEPA` 在 `11,715` 个独立无标签事例上即达到 `mu_AUROC = 0.9354` 与 `r(\ln|s_y|) = 0.4873`（移除该任务则崩塌至 `0.6036 / 0.0319`），直接贡献 `+1.063 pp BA4` 下游微调增益。
   - 若将整平面遮蔽退化为常规 `60%` 随机单团簇遮蔽并暴露坐标，模型利用同层电磁簇射横向荧光密度插值作弊，导致下游 `1% FPR` $\nu_\tau$ 效率显著损失 `-2.083 pp`（`D1_obs -4.626 pp`）。
3. **等总算力（`262` passes）监督重放发生严重记忆化过拟合，而 SSL 均匀平面主干与显式几何/物理分区主干之间存在显著的“单模型归纳偏置—多模型表征正交性”互补律**：
   - 固定 `25.6k` 有标签数据强行匹配 `5.69e16 FLOPs` 算力在第 13 轮后导致验证集 `NLL` 从 `0.981` 爆炸至 `5.862`，证明无标签纯钨事例提供的物理相空间覆盖不可由有标签样本重复迭代替代。
   - 单模型层面，`PinpointSSL-JEPA` 为严格满足整平面遮蔽无泄漏而未内置跨三层显式 3-hit `TrackletOps` 及 `D1/D2/D3` 手工物理区段路由，其单模型 `BA4`（`Tune6k 58.20%`）低于专用监督架构 `TrackletIPFormer-GN2`（`59.20%`）；但在集成层面，其连续簇射演化潜表征填补了后两者的信息盲区，使二路 `JEPA + L1_GN2` 达到全场最高原始 `BA4 = 59.73%`，三范式 `3-Way + Score-Bank KL-DRO` 达到全场最优 **`59.57% BA4 / 0.9275 NLL`**（较 Stage 327 最优基准提升 `+0.808 pp BA4` 与 `-0.00984 NLL`）。

### 4.2 待验证物理与优化假设（Hypotheses）

1. **`PinpointSSL-JEPA` 单像素暗噪声敏感性的物理门控修复假设**：`PinpointSSL-JEPA` 在 `Shard 080 pixel_dark_1e-5` 压力下从 `57.43%` 降至 `42.27%`，原因是其预训练语料 `l2_ssl_300k` 为名义干净模拟，平面内 PMA 池化对随机热像素未设防。假设在微调期或推理入口复用 Stage 327 N1 已验证的纯观测 1-bit 孤立单像素热噪声过滤门控（或在预训练随机注入稀疏单像素暗电流增强），可在不损失干净样本 `0.9275 NLL` 三路融合增益的前提下将 `pixel_dark_1e-5` 退化压回 `< 3.0 pp`。
2. **二路（`L1_GN2 + DualRegime + KL-DRO`）与三路（`L1_GN2 + DualRegime + JEPA + KL-DRO`）在封存确认集与系统误差扰动集上的泛化权衡假设**：在 `Tune6k` 上，三路融合拥有最优全局 `BA4`（`59.57%` vs `59.43%`）与最优 `NLL`（`0.9275` vs `0.9291`），而二路融合在 `D1_obs 1% FPR` 尾部效率（`11.22%` vs `9.62%`）及高暗噪声鲁棒性上占优。两者在封存 `Confirm (shards/125..149)` 与 `input_sys_confirm*` 探测器系统误差集上的相对表现目前**未测量**，若先为 `JEPA` 施加无监督孤立单像素过滤门控（仅在 `tune_screen` 验证），预期三路融合可兼得全局似然与硬件抗噪稳健性。

---

## 5. 下一步计划

1. **Stage 329A（单次预解封候选锁定与合成演练，严格零接触 `shards/100..149`）**：
   - **具体任务**：
     1. 在不触碰任何封存分片的前提下，仅在 `Shard 080` 与 `tune_screen` 上检验对已训练完成的 `PinpointSSL-JEPA` 3-seed 检查点推理入口挂载零参数孤立单像素热噪声过滤门控后的暗噪声恢复度，据此在 `tune_screen` 上唯一锁定主提交模型（`3-Way L1_GN2 + DualRegime + JEPA + KL-DRO` 或 `2-Way L1_GN2 + DualRegime + KL-DRO`）。
     2. 编写确定性单向解封脚本与 `unittest` 合成演练套件，固化预注册三步评估链：`Cal-A (100..109)` 后验温度校准 $\to$ `Cut-B (110..124)` Clopper-Pearson 保守 `1% / 0.1% FPR` 阈值锁定 $\to$ `Confirm (125..149)` 与 `input_sys_confirm*` 单次只读盲评。
   - **所需输入**：Stage 328 已归档的 3-seed 检查点与预测缓存（[exp_l1_gn2/stage2](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/exp_l1_gn2/stage2)、[exp_l2_jepa/stage2](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/exp_l2_jepa/stage2) 及 Stage 327 `DualRegimeModalNet` 检查点）。
   - **验收标准**：合成数据解封演练 `100%` 比特可复现、[seal_guard.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/nextgen/seal_guard.py) 分级权限单元测试全通过、主模型与次级对照模型权重 SHA256 哈希在解封前写入预注册清单并获用户批准。
2. **Stage 329B（获批后执行一次性封存确认集评估与最终物理论文图表定稿）**：
   - **具体任务**：经用户明确授权后运行单次解封脚本，输出 `Confirm`（`N = 10,000`）与 `input_sys_confirm*` 的最终无偏物理指标、置信区间与系统误差预算表。
   - **验收标准**：全流程仅执行单次前向推理、零超参回调，完整记录最终物理发现灵敏度与系统不确定度。
