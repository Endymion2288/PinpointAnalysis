# Stage 327 研究报告：统一多模态架构 `PinpointUnifiedNet`（N1）、可微运动学顶点修复与 1 mm 硬件对照（N2 & Q13）及冻结主干 `Score Bank` KL-DRO 尾部优化

- **日期**：2026-11-22
- **基线锚点**：Stage 326 分支 `feat/stage326-nextgen-research` @ `d2987dc`（冻结代码基线 `51c3d8966426286de69df8cecacadb314643611f`；探测器几何与 Geant4 模拟源码 `Pinpoint_G4` @ `48a9e3deadb21cef5ed3fc8c9bc53edcf4a624bf`）
- **Stage 327 工作流分支与提交**：
  - N1 工作流：`feat/stage327-n1-unified` @ `466cd0e4e9303fbf069c02554b0e3831811db76e`
  - N2 & Q13 工作流：`feat/stage327-n2-taukinema` @ `48b955c2eb6e09407b0df147e096b95be6b69863`（前序修复提交 `112e259`）
  - Tail-DRO 与图表工作流：`feat/stage327-tail-dro` @ `1469f9888d6fcad9a243c5da9d94efddf8329963`
  - 集成主分支：`feat/stage327-nextgen-completion` @ `5cc0cee`（代码与导出接口集成；合并提交 `156eb73`、`4becf3f`、`6fc1b0c`）
- **随机种子**：Stage-I 架构实验与消融统一固定 `seed = 42`；Stage-II `Tune6k` 冻结主干尾部确认采用三组独立种子 `{17, 42, 2026}`；配对 family-block bootstrap 统一固定 `seed = 20263250`（`B = 1,000`）
- **计算调度与硬件**：HTCondor 调度节点 `bigbird20.cern.ch`（提交时排队空闲 GPU 负载最优，`--min-capability 8.0`），作业集群 ID `1585572..1585632`，运行于 `NVIDIA H100 NVL`、`NVIDIA H100L-2-24C MIG 2g.24gb`、`NVIDIA H100L-1-12C MIG 1g.12gb` 与 `NVIDIA A100-PCIE-40GB`（交互式单元测试使用 `Tesla T4`）
- **产物与审计目录**：
  - 架构蓝图与修复规范：[stage327_plan](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage327_plan)
  - N1 实验产物：[exp_unified_net](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/exp_unified_net)
  - N2 & Q13 实验产物：[exp_n2_tau_kinema](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/exp_n2_tau_kinema)
  - Tail-DRO 实验产物：[exp_tail_dro](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/exp_tail_dro)
  - 300 DPI 英文汇总图集：[stage327_figures](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage327_figures)
  - 独立交叉审计报告：[stage327_audit_report.md](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage327_verification/stage327_audit_report.md)
- **盲法与封存审计声明**：全程零接触封存数据集 `Cal-A (shards 100..109)`、`Cut-B (shards 110..124)`、`Confirm (shards 125..149)`、`sealed_truth_meta/`、`input_sys_confirm*` 及 GENIE 测试分片 `016..028`（访问计数 = `0`，全部指标标注为**未测量**）；Stage-I 实验 N1、N2 与 Q13 全程零接触 `Tune8k / Tune6k (shards 080..099)`（访问计数 = `0`，其 Stage-II 指标标注为**未测量**）；既有 `src/analysis/flavour/v2/` 与 `src/analysis/flavour/v3/` 修改行数 = `0`。

---

## 核心科学结论（结论先行）

1. **冻结主干 `Score Bank` KL-锚定 CVaR-DRO 尾部微调通过全部预注册门限并在 Stage-II `Tune6k`（`N = 6,000`，3 seeds）独立确认**：在完全保持主干权重 bit-identical（`max_diff = 0.0`）且微调耗时仅 `7.1–21.5 s/seed` 的前提下，锁定配置 `dro_bal_l008_kl15_bank8192` 将解耦 `50/50 Blend` 在 `Tune6k` 上的 `raw All BA4` 从 `58.50%` 提升至 `58.76%`（`+0.27 pp`），`NLL` 从 `0.9380` 改善至 `0.9374`（`-0.0006 <= +0.005` 门限），`Within-Slice 1% FPR` 全样本 $\nu_\tau$ 效率从 `7.23% (108/1494)` 提升至 `7.83% (117/1494)`，前部像素包容域 `D1_obs` 的 `1% FPR` $\nu_\tau$ 效率从 `6.73% (21/312)` 大幅跃升至 `11.86% (37/312)`（`screen_cal` 配对参考校准后达 `12.82% [40/312]`，1,000 次配对 bootstrap `+4.17 pp`，95% CI `[+0.31, +6.33] pp`），同时将核外推样本 `W184 EPPS21 NC Charm` 的假阳性率 `p_tau > 0.5` 从 `1.67%` 压低 **5 倍**至 **`0.33% (4/1200)`**。
2. **实验 N1（`PinpointUnifiedNet`，`7.570M` 参数）完成两轮因果诊断并诚实触发 Section 10.4 预注册回退规则**：Round-1 证明在无在线噪声课程干扰时，单模型 `abl_no_online_spec_curriculum` 在 `internal_val_eval`（`N = 4,800`）达到 `BA4 = 58.60%`、`NLL = 0.9602`，单模型即追平解耦双模型 `50/50 Blend`（`58.54% / 0.9593`），但存在探测器压力脆弱性（`TrkDead 2%+N1 = -5.32 pp`、`PixDark = -6.46 pp`）。Round-2 修复在线噪声注入的斜率缩放漏洞（`noise_hit[:, 3:6]`）、引入多站持续性门控与 `label < 2` 无冲突粲味辅助损失后，彻底消除硬件压力退化（`TrkDead 2%+N1 = -0.07 pp`、`TrkDead 5%+N2 = -0.10 pp`、`PixDark = -1.37 pp`，9/9 Student-t Pulls 落入 `[0.878, 1.099]`），并将 `Split-Half 1% FPR` $\nu_\tau$ 效率推升至 `6.40%`（`D1_obs = 11.49%`）；但端到端联合梯度竞争使干净样本 `All BA4` 回落至 `57.48%`（`NLL = 0.9697`）且 `W184 p_tau > 0.5 = 3.58% > 1.8%`。依据预注册 Section 10.4 规则，N1 终止于 Stage-I，正式锁定解耦 `50/50 TrackletIPFormer + DualRegimeModalNet + Score-Bank KL-DRO` 为下一代主干基线。
3. **实验 N2（`TauKinemaDiffNet` 初级顶点反冲污染修复）通过两步受控实验彻底查明 `19.261 mm` 顶点退化的双重物理根源，将 `Interm150` 纵向顶点中位误差压降至 `1.984 mm`（优于 `6.0 mm` 目标 3 倍）**：第一步 `n2fix` 仅将像素平面先验 `avr_ok` 门控限制于前部 30 mm 钨靶（`first_tgt_p <= 5`），使 `Interm150` 中位 `|dz|` 从 `19.261 mm` 减半至 `10.215 mm`，同时暴露出单硅片钨反照率（tungsten albedo）触发错位与锚点先验盲区；第二步 `n2v2` 引入反散射安全两站簇符合门控与 `_RegimeAwareSymmetricVertexRefiner`，将 `Interm150 |dz|` 压降至 **`1.984 mm`**（`Front30 = 2.136 mm`，9/9 Pulls 位于 `[0.924, 1.096]`），单模型 `All BA4` 从 `56.77%` 跃升 `+1.69 pp` 至 **`58.46%`**（`NLL = 0.9652`），三模型等权融合 `Blend_BDF_n2v2` 达到 `BA4 = 58.83%`、`NLL = 0.9451`。因高精度顶点显著放大了 12 类层次化头部在固定阈值 `0.5` 下的未校准尾部概率（`W184 p_tau > 0.5 = 7.50%`，而在 `1% FPR` 工作点 `p_tau > 0.633` 处仅 `2.17%`）且未设像素暗噪声门控（`PixDark = -12.52 pp`），`n2v2` 严格保留在 Stage-I。
4. **近期原型 Q13（全硬件等效 `1.00 mm` 传感器对照 `abl_full_1mm_sensor`）确立 `20 um` MAPS 像素精度的定量物理价值**：在严格冻结 `clusters` 与 `IPT` 特征（bit-identical）的条件下，同时将 3D 团簇质心 `(x, y)` 与 `32x32` 局部图像块重采样至 `1.00 mm` 体素后，原生 `20 um` 传感器在 `Front30` 薄靶区取得显著对数似然增益 `ΔNLL(native_20um - full_1mm) = -0.0247`（95% CI `[-0.0405, -0.0105]`，跨整个 `Pinpoint_PixXY` 为 `-0.0158 [-0.0280, -0.0030]`），并将 `W184 NC Charm` 假阳性率从 `4.08%` 压低至 `2.33%`；而在 150 mm 厚钨靶区（单层 `52.5 mm W = 15.0 X_0`、层间节距 `20.23 X_0`）多重库仑散射（MCS）抹平亚毫米角分辨率，两者统计无差异（`ΔNLL = +0.0075 [-0.0085, +0.0246]`）。同时证实此前仅量化质心而保留 `20 um` 图像块的 `centroid_only_1mm` 会产生跨模态几何失配伪影（较真实 `full_1mm` 额外损失 `ΔNLL = +0.0110 [+0.0049, +0.0182]`）。

---

## 1. 阶段目标

Stage 327 承接 [2026-11-22_stage326_nextgen_neutrino_ml_research.md](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/workbook/2026-11-22_stage326_nextgen_neutrino_ml_research.md) 第 9–10 节及开放问题 Q13，旨在严格盲法与零接触封存分片的前提下，回答四项定量物理与机器学习问题，并检验预注册验收门限：

1. **实验 N1（`PinpointUnifiedNet` 单模型多模态融合验证）**：构建参数量 `< 8.5M`（目标 `~7.4M`）的统一网络，融合双尺度像素 CNN、软径迹段异面最近点距矩阵 `R_ab`、42 站双激活 Sinkhorn 闪烁体编码器、Magnet-0 4D `IPT-Spec` 联锁以及位移锚定反冲半球特征 `C_hem`。在 Stage-I `internal_val_eval`（`N = 4,800`）检验五项预注册门限：（i）`All BA4 >= 58.3%` 且距 Stage-I 解耦 `50/50 Blend`（`58.54% / 0.9593`）不超过 `0.3 pp BA4` 与 `0.005 NLL`；（ii）`W184 EPPS21 NC Charm` 的 `p_tau > 0.5 <= 1.8%`；（iii）`TrkDead 2%+N1` 压力退化 `< 3.0 pp`；（iv）全部 9 项 Student-t `p68(|pull|)` 位于 `[0.85, 1.15]`。若任一门限未达标，立即执行 Section 10.4 回退规则，锁定解耦 `50/50 Blend`。
2. **实验 N2（`TauKinemaDiffNet` 厚靶初级顶点崩塌修复）**：定位并消除 Stage 326 中 `TauKinemaDiffNet` 在 `Interm150` 区段初级顶点纵向中位误差高达 `19.261 mm` 的物理与程序根因，验收门限为 Stage-I `Interm150` 中位 `|dz| <= 6.0 mm` 且全部 9 项 Student-t Pulls 保持在 `[0.85, 1.15]`，评估其单模型及与 B+D 三路融合的增益与外推安全性。
3. **原型实验 Q13（全传感器等效 `1.00 mm` 像素螺距对照）**：消除 Stage 326 `abl_1mm_quant` 仅量化质心而未降采样 `32x32` 局部图像块的混杂因素，实现质心 `(x, y)` 与局部图像块同步体素化至 `1.00 mm` 的 `abl_full_1mm_sensor`，定量拆解 `20 um` MAPS 在薄靶（`Front30`）与厚靶（`Interm150`）中的真实物理增益。
4. **近期尾部优化（冻结主干 `Score Bank` KL-锚定 Partial-AUC / CVaR-DRO 微调）**：针对 `FPR <= 1%` 极低本底 $\nu_\tau$ 发现工作点小批量梯度稀疏的瓶颈，构建容量 `8,192` 的历史本底分数库与 KL/NLL 双锚定单向 `pAUC(FPR <= 1%)` / CVaR-DRO 目标。在 Stage-I 筛选满足 `ΔNLL <= +0.005` 且无 ECE 倍增的配置后，冻结超参推进至 Stage-II `Tune6k`（`N = 6,000`，3 seeds）完成独立确认。

---

## 2. 工作内容

### 2.1 架构蓝图、数值防护与两轮物理修复规范
在 [blueprint_n1_n2_tail.md](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage327_plan/blueprint_n1_n2_tail.md)、[round2_n1_remediation_spec.md](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage327_plan/round2_n1_remediation_spec.md) 与 [round2_n2_vertex_spec.md](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage327_plan/round2_n2_vertex_spec.md) 中完成精确模块参数预算、张量契约与修复设计：

- **N1 `PinpointUnifiedNet`（[unified_pinpoint_net.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/nextgen/unified_pinpoint_net.py)、[run_unified_pinpoint_net.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/nextgen/run_unified_pinpoint_net.py)）**：
  - **模块组成（Round-2 实测 `7,569,774` 参数，`< 8.5M` 预算）**：包含多尺度像素小块残差编码器、带 FP64 计算后转 FP32 的异面直线最近点距与有符号冲击参数交互矩阵 `R_ab`（`_safe_hypot(dx, dy)` 防止零横动量梯度 `NaN`）、`front_gate` 解耦 30 mm 薄靶像素先验与 150 mm 厚靶闪烁体能量加权先验、42 站双激活（主闪烁体 + Veto/Preshower）Sinkhorn 最优传输与逐事例飞行时间相对中心化（per-event ToF centering）、Magnet-0 内径几何检查的 4D `IPT-Spec` 联锁与 `spec_rel_head`、位移锚定反冲轴半球特征 `C_hem`（抑制 $D \to K/\pi + X$ 同侧粲强子衰变伪装 $\nu_\tau$），以及 `ScaledQKNorm + log(1 + M)` 多重度自适应跨模态注意力。
  - **Round-1 压力故障诊断与 Round-2 三项精确修复**：Round-1 评估发现 `TrkDead 2%+N1` 退化 `-5.32 pp`、`PixDark 1e-5` 退化 `-6.46 pp`。取证定位三项程序与物理机制缺陷并在 Round-2（`466cd0e`）修复：（1）Round-1 在线假三元组噪声注入仅扰动位置 `[:, 0:3]` 而遗留零斜率，导致推理期真实 `tracker_dead_2pct_noise1` 的大角度随机斜率触发 `R_ab` 分布外偏移；Round-2 同步采样斜率 `noise_hit[:, 3:6]` 并施加 4 站纵向持续性门控；（2）在特征预提取 `extract_unified_item`（`unified_pinpoint_net.py:160-185`）中过滤孤立 1-bit 单像素暗噪声团簇（`n_hits == 1`、同平面邻域 `cnt_r040 == 0 & cnt_r1 == 0` 且相邻两层无 `dp in {1, 2}` 紧符合 MIP 伴随击中），与推理期压力分布对齐；（3）将粲味辅助二分类损失掩码收紧至 `label < 2`（仅对 $\nu_e / \nu_\mu$ CC 事例监督 prompt vs charm，避免对 NC $c\bar{c}$ 与 $\nu_\tau$ 产生冲突梯度）。
- **N2 `TauKinemaDiffNet` 初级顶点两步修复（[tau_kinema_diffnet.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/nextgen/tau_kinema_diffnet.py)、[run_tau_kinema_diffnet.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/nextgen/run_tau_kinema_diffnet.py)）**：
  - **Step 1（`n2fix`，提交 `112e259`）**：在 `tau_kinema_diffnet.py` 将像素顶点残差门控从 `first_p >= 0` 收紧为 `avr_ok = (n_tr >= 2) & (first_p >= 0) & (first_tgt_p <= 5)`，保留 Stage 326 Round-2 的 `m_res[NC_cc] = 0` 与 `0.20` 残差缩放。该单行修复将 `Interm150 |dz|` 从 `19.261 mm` 降至 `10.215 mm`，并揭示剩余误差来自两方面：（a）150 mm 钨靶级联反照率（backscatter albedo）在前部 30 mm 像素站产生单平面低电荷击中，使 `22.5%`（`165/734`）的 `Interm150` 事例伪造 `first_p <= 5`；（b）`_DifferentiableAVR` 缺乏对观测先验 `z_prior` 的直接锚定，且闪烁体粗顶点 `z_scint_init` 含上游反散射早触发偏置。
  - **Step 2（`n2v2`，提交 `48b955c`）**：实现反散射安全先验（要求前部两站像素团簇符合 `front_two_station` 才启用薄靶像素先验；厚靶区采用 5% 截尾抗反散射闪烁体前沿加权先验）与 `_RegimeAwareSymmetricVertexRefiner`（以 `(x_prior, y_prior, z_prior)` 为对称零均值中心锚点，利用径迹段投影残差与可学习 Huber 权重精细化 3D 初级顶点），参数量由 `5.658M` 精简至 `5.267M`。
- **原型实验 Q13 全 `1.00 mm` 传感器对照（[run_q13_full_1mm_sensor.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/nextgen/run_q13_full_1mm_sensor.py)）**：
  - 实现 `extract_full_1mm_sensor_item`（`run_q13_full_1mm_sensor.py:178-375`）：锁定原生 `20 um` 连通域团簇列表及全部 `IPT (plane >= 12)` 团簇逐位不变，将靶区（`plane < 12`）团簇内原始像素击中吸附至 **1-bit 去重 `1.00 mm` `rint` 体素晶格**，基于该 `1.00 mm` 体素重算靶区团簇质心、二阶矩、平面摘要与软径迹段，并以 `1.00 mm` 步距重新栅格化 `32x32` 局部图像块（`_rasterize_micro_patches(..., 1.00, 1.00)`），严格保持闪烁体 `clusters` 与磁谱仪 `IPT` 逐位不变（`max_diff = 0.0`），与 `native_20um` 和 `centroid_only_1mm` 在完全相同的 Stage-I 训练设置（`seed = 42`，20 passes）下配对对比。
- **冻结主干 `Score Bank` KL-锚定 DRO 尾部微调（[tail_dro_pauc.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/nextgen/tail_dro_pauc.py)、[run_tail_dro_finetune.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/nextgen/run_tail_dro_finetune.py)）**：
  - **数学形式**：冻结主干全部参数（`requires_grad = False`），仅微调末端分类头参数 $\phi$（`lr = 5e-5`，`weight_decay = 1e-4`，`3 passes`，`batch_size = 128`）。定义对数几率边际分数 $s_\tau(x) = \log p_\tau(x) - \log(1 - p_\tau(x))$，维护容量 $M = 8,192$ 的 FIFO 历史本底分数队列 $\{s_\tau(x_j^-)\}_{j=1}^M$（无梯度停驻），优化混合损失：
    $$\mathcal{L}_{\text{total}}(\phi, \eta) = \lambda_{\text{NLL}} \mathcal{L}_{\text{CE}}(y, p_\phi) + \lambda_{\text{KL}} D_{\text{KL}}\!\left(p_{\phi_0}(\cdot \mid x) \,\|\, p_\phi(\cdot \mid x)\right) + \lambda_{\text{DRO}} \mathcal{L}_{\text{CVaR-pAUC}}(\phi, \eta; \beta = 0.01)$$
    其中 `cvar_balanced` 模式对本底库前 $\beta = 1\%$ 高分假阳性尾部施加平滑对偶阈值 $\eta$ 与 KL-DRO 重加权（温度 $\tau_{\text{DRO}} = 0.05$，软间隔 `margin = 0.5`，分数裁剪 `score_bound = 4.0`），KL 散度锚定项严格约束主分布概率校准漂移。
  - **零失真隐空间缓存加速**：单次前向提取冻结主干隐向量至内存张量后释放主干计算图，实测缓存输出与端到端模型前向 `max_cache_equiv_diff = 0.0`（精确逐位相等），单模型单种子 3-pass 微调在 GPU 上仅耗时 `7.1–21.5 s`。

### 2.2 自动化测试、独立交叉审计与图表生成
- 新增并通过 [test_nextgen_unified_net.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/tests/test_nextgen_unified_net.py)、[test_nextgen_n2_tau_kinema_fix.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/tests/test_nextgen_n2_tau_kinema_fix.py)、[test_nextgen_tail_dro.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/tests/test_nextgen_tail_dro.py)，连同既有 `test_nextgen_*.py` 套件全面验证真值隔离（`truth_invariance max_abs_diff = 0.0`）、主干冻结逐位一致性、封存分片零访问拦截器 `seal_guard.py` 及数值有限性。
- 独立审计器对全部三路代码分支、运行日志、JSON 产物及重算指标完成交叉核验，生成 [stage327_audit_report.md](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage327_verification/stage327_audit_report.md)、[n1_audit.md](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage327_verification/n1_audit.md)、[n2_q13_audit.md](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage327_verification/n2_q13_audit.md) 与 [tail_dro_independent_audit.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage327_verification/tail_dro_independent_audit.json)，确认指标重算误差 `< 1e-6`、零超参漂移、零封存违规。
- 使用 [plot_stage327_summary.py](file:///eos/home-x/xcheng/FASER/Pinpoint/PinpointAnalysis/src/analysis/flavour/nextgen/plot_stage327_summary.py) 生成三张 300 DPI 纯英文出版级总结图：[fig8_tail_dro_stage1_ablations.png](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage327_figures/fig8_tail_dro_stage1_ablations.png)、[fig9_tail_dro_stage2_tune6k.png](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage327_figures/fig9_tail_dro_stage2_tune6k.png)、[fig10_stage327_integrated_summary.png](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage327_figures/fig10_stage327_integrated_summary.png)（CJK 字符扫描计数 = `0`）。

---

## 3. 详细数据

### 3.1 盲法、封存分片与代码边界审计账本

| 审计项目 | 分片 / 路径范围 | Stage-I (N1 / N2 / Q13 / DRO-S1) | Stage-II (DRO-S2) | 审计结论 |
| --- | --- | --- | --- | --- |
| 训练与内部筛选集 `Train / Dev / Val` | `shards 000..079`（Train `25,600`、Dev `1,600`、Val `4,800`） | 已使用 | 仅使用 `Train` 微调与 `Dev` 参考温度 | 合规 |
| Stage-II 确认集 `Tune6k`（来自 `Tune8k`） | `shards 080..099`（`N = 6,000` 均衡 4 味确认集） | 访问计数 = `0`（N1/N2/Q13 **未测量**） | 仅锁定 `dro_bal_l008_kl15_bank8192` 单次评估 | 零超参漂移 |
| 核外推与发生器诊断集（6 OOD shards） | `W184 NC Charm/Incl`、`Run 10085-10088`、`Pythia6`、`Pythia8` | 已评估 | 已评估 | 合规（非封存） |
| 封存校准集 `Cal-A` | `shards 100..109` | 访问计数 = `0`（**未测量**） | 访问计数 = `0`（**未测量**） | 100% 封存完好 |
| 封存阈值锁定集 `Cut-B` | `shards 110..124` | 访问计数 = `0`（**未测量**） | 访问计数 = `0`（**未测量**） | 100% 封存完好 |
| 封存最终确认集 `Confirm` 及系统误差集 | `shards 125..149`、`sealed_truth_meta/`、`input_sys_confirm*`、GENIE `016..028` | 访问计数 = `0`（**未测量**） | 访问计数 = `0`（**未测量**） | 100% 封存完好 |
| 既有生产包边界 | `src/analysis/flavour/v2/`、`src/analysis/flavour/v3/` | `git diff` 行数 = `0` | `git diff` 行数 = `0` | 零侵入 |

### 3.2 实验 N1：`PinpointUnifiedNet` Stage-I 两轮评估与消融对比

数据来源：[stage1_unified_results.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/exp_unified_net/stage1_unified_results.json)（Round-2，`466cd0e`，Condor 集群 `1585597..1585605`、`1585624..1585627`）与 [stage1_unified_results.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/exp_unified_net/round1_archive/stage1_unified_results.json)（Round-1 归档，Condor 集群 `1585572..1585580`、`1585584`、`1585588..1585590`）。所有指标均在 `internal_val_eval`（`N = 4,800`）与 `W184 EPPS21 NC Charm`（`N = 1,200`）实测；`Tune6k` 及封存分片均为**未测量**。

| 模型变体 / 基线 | 参数量 | `All BA4` (%) | `All NLL` | `D1_obs BA4` (%) | `Front30` `\|dz\|` (mm) | `Interm150` `\|dz\|` (mm) | `SH 1% FPR` $\nu_\tau$ 效率 (All / D1, %) | `WS 1% FPR` $\nu_\tau$ 效率 (All / D1, %) | `W184` `p_tau>0.5` (%) | `TrkDead 2%+N1` $\Delta$BA4 (pp) | `PixDark 1e-5` $\Delta$BA4 (pp) | 9/9 Pulls 范围 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Stage 326 `TrackletIPFormer` (B) | 6.23M | 57.81 | 0.9659 | 64.24 | 5.008 | 5.562 | 4.43 / 6.08 | 4.67 / 4.98 | 2.33 | -6.59* | -2.74* | `[0.917, 1.111]` |
| Stage 326 `DualRegimeModalNet` (D) | 6.63M | 57.52 | 0.9829 | 62.56 | — | — | 4.27 / 6.76 | 4.08 / 6.05 | 3.42 | -4.87* | -3.15* | — |
| **Stage 326 解耦 `50/50 Blend` (B+D)** | **12.85M** | **58.54** | **0.9593** | **64.05** | — | — | **5.09 / 6.08** | **5.00 / 6.05** | **2.25** | — | — | — |
| N1 R1 `s1_unified_pinpoint_net` | 7.595M | 58.00 | 0.9662 | 65.12 | 3.991 | 6.384 | 5.42 / 9.46 | 5.08 / 6.76 | 2.67 | -5.32 | -6.46 | `[0.866, 1.149]` |
| N1 R1 `abl_no_online_spec_curriculum` | 7.595M | **58.60** | **0.9602** | 64.86 | 3.810 | 5.931 | 5.58 / 9.46 | 4.67 / 7.12 | 2.58 | -2.97 | -2.79 | `[0.851, 1.075]` |
| N1 R1 `abl_no_recoil_hem` | 7.556M | 58.44 | 0.9644 | 64.58 | 3.528 | 6.062 | 4.93 / 6.08 | 4.75 / 3.91 | 7.08 | -4.00 | -7.13 | `[0.895, 1.085]` |
| N1 R1 `abl_no_tracklet_rab` | 7.595M | 58.23 | 0.9655 | 64.93 | 3.469 | 6.099 | 6.90 / 10.81 | 4.50 / 4.63 | 3.58 | -6.02 | -8.15 | `[0.913, 1.094]` |
| **N1 R2 `s1_unified_pinpoint_net` (最终锁定)** | **7.570M** | **57.48** | **0.9697** | **62.74** | **2.982** | **5.727** | **6.40 / 11.49** | **5.33 / 7.47** | **3.58** | **-0.07** | **-1.37** | **`[0.878, 1.099]`** |
| N1 R2 `abl_no_online_spec_curriculum` | 7.570M | 57.85 | 0.9662 | 63.37 | 2.984 | 5.712 | 4.60 / 9.46 | 4.50 / 6.41 | 2.42 | -1.75 | -3.03 | `[0.870, 1.105]` |
| N1 R2 `abl_no_recoil_hem` | 7.530M | 57.92 | 0.9702 | 63.31 | 2.875 | 4.971 | 5.42 / 8.78 | 5.58 / 6.76 | **5.83** | -1.26 | -2.15 | `[0.894, 1.101]` |
| N1 R2 `abl_no_tracklet_rab` | 7.569M | 57.81 | 0.9666 | 63.12 | 3.027 | 4.684 | 6.73 / 10.81 | 5.92 / 6.76 | 3.25 | -1.06 | -0.43 | `[0.878, 1.111]` |

*注：表中带 `*` 的 B 与 D 单模型压力数值来自 `Shard-080`（`N = 400`）标准参考测试，其余 N1 压力数值均在 `internal_val_eval` 前 400 事例标准压力套件上配对实测。*

**N1 Round-2 预注册门限逐项核验与配对 Bootstrap（`seed = 20263250`，`B = 1,000`）**：
- **门限 1（主分类性能）**：`All BA4 = 57.48% < 58.3%`（未通过）；相对解耦 `50/50 Blend` 的配对差值为 `ΔBA4 = -1.06 pp`（95% CI `[-2.10, -0.15] pp`）、`ΔNLL = +0.0104`（95% CI `[+0.0023, +0.0186]`），超出 `0.3 pp BA4` 与 `0.005 NLL` 容差。相对单模型 `DualRegimeModalNet`，N1 Round-2 显著改善对数似然 `ΔNLL = -0.0132`（95% CI `[-0.0220, -0.0042]`）。
- **门限 2（核外推粲背景抑制）**：`W184 p_tau > 0.5 = 3.58% > 1.8%`（未通过）。但受控消融证明位移锚定反冲半球特征 `C_hem` 发挥关键物理抑制作用：移除 `C_hem` 后（`abl_no_recoil_hem`），Round-1 与 Round-2 的 `W184 p_tau > 0.5` 分别飙升至 **`7.08%`** 与 **`5.83%`**。
- **门限 3（硬件死通道与暗噪声鲁棒性）**：`TrkDead 2%+N1 ΔBA4 = -0.07 pp`（`< 3.0 pp` 门限，**通过**），`TrkDead 5%+N2 ΔBA4 = -0.10 pp`，`PixDead 2% / 5% = -0.02 pp / -0.03 pp`，`PixDark 1e-5 = -1.37 pp`，`SiPM Sat 0.95 = -0.27 pp`。移除在线谱仪噪声课程（`abl_no_online_spec_curriculum`）后，`TrkDead 5%+N2` 恶化 44 倍至 `-4.48 pp`，直接证实斜率一致在线课程的因果有效性。
- **门限 4（异方差顶点不确定度校准）**：`All_valid`、`Front30`、`Interm150` 三区段 $\times$ `(x, y, z)` 全部 9 项 `p68(|pull|)` 为 `[0.958, 1.099, 0.964, 0.926, 0.942, 0.878, 0.977, 0.945, 0.952]`，100% 落入 `[0.85, 1.15]`（**通过**）。
- **处置决定**：严格执行 Section 10.4 预注册回退规则，N1 不进入 Stage-II `Tune6k`，锁定解耦 `50/50 Blend` 进入尾部微调。

### 3.3 实验 N2：`TauKinemaDiffNet` 初级顶点两步修复与多模型融合评估

数据来源：[stage1_n2_evaluation.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/exp_n2_tau_kinema/stage1_n2_evaluation.json)（`n2fix`，Condor 集群 `1585581`）与 [stage1_n2v2_evaluation.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/exp_n2_tau_kinema/stage1_n2v2_evaluation.json)（`n2v2`，Condor 集群 `1585606`）。评估样本为 Stage-I `internal_val_eval`（`N = 4,800`）与 6 组 OOD 分片；`Tune6k` 及封存分片均为**未测量**。

| 模型 / 融合方案 | 参数量 | `Front30` 中位 `\|dz\|` (mm) | `Interm150` 中位 `\|dz\|` (mm) | `All` 中位 3D 顶点误差 (mm) | 9/9 Pulls 范围 | `All BA4` (%) | `All NLL` | `D1_obs BA4` (%) | `WS 1% FPR` $\nu_\tau$ 效率 (%) | `SH 1% FPR` $\nu_\tau$ 效率 (%) | `W184` `p_tau>0.5` (%) |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage 326 `F_r2_d0400dfd`（修复前基线） | 5.658M | 7.333 | 19.261 | 18.910 | `[0.919, 1.125]` | 56.77 | 0.9846 | 63.00 | 3.92 (47/1200) | 3.28 (20/609) | 2.08 |
| Step 1：`F_n2fix`（仅 `first_tgt_p <= 5` 门控） | 5.658M | 6.954 | **10.215** | 16.155 | `[0.918, 1.051]` | 56.60 | 0.9877 | 63.63 | 3.92 (47/1200) | 4.11 (25/609) | **1.58** |
| **Step 2：`F_n2v2`（反散射安全先验 + 对称顶点精修器）** | **5.267M** | **2.136** | **1.984** | **7.335** | **`[0.924, 1.096]`** | **58.46** | **0.9652** | **64.57** | **6.08 (73/1200)** | **5.75 (35/609)** | **7.50** (在 `p>0.633` 为 **2.17%**) |
| 双模型基线 `Blend_BD`（50% B + 50% D） | 12.85M | — | — | — | — | 58.54 | 0.9593 | 64.05 | 5.00 (60/1200) | 5.09 (31/609) | 2.25 |
| 三模型融合 `Blend_BDF_r2`（B + D + `F_r2`） | 18.51M | — | — | — | — | 58.83 | 0.9536 | 64.69 | 4.25 (51/1200) | 3.94 (24/609) | 1.50 |
| 三模型融合 `Blend_BDF_n2fix`（B + D + `F_n2fix`） | 18.51M | — | — | — | — | 58.60 | 0.9558 | 64.95 | 4.83 (58/1200) | 4.43 (27/609) | 1.67 |
| **三模型融合 `Blend_BDF_n2v2`（B + D + `F_n2v2`）** | **18.12M** | — | — | — | — | **58.83** | **0.9451** | **64.69** | **5.92 (71/1200)** | **5.42 (33/609)** | **3.67** |

**N2 关键统计检验（1,000 次配对 family-block bootstrap，`seed = 20263250`）与外推诊断**：
- **单模型跃升（`F_n2v2 - F_r2_d0400dfd`）**：`All ΔBA4 = +1.69 pp`（95% CI `[+0.65, +2.77] pp`），`ΔNLL = -0.0194`（95% CI `[-0.0301, -0.0088]`），`1% FPR Δτ_eff = +2.17 pp`（95% CI `[+0.25, +4.48] pp`），`0.1% FPR Δτ_eff = +1.92 pp`（95% CI `[+0.84, +3.08] pp`），全部四项核心指标的 95% 置信区间严格排除零。
- **三路融合似然增益（`Blend_BDF_n2v2 - Blend_BD`）**：`All ΔNLL = -0.0141`（95% CI `[-0.0179, -0.0105]`），`ΔBA4 = +0.29 pp`（95% CI `[-0.39, +1.00] pp`）。
- **6 OOD 分片表现（`F_n2v2`）**：`Run 10085-10088 BA4 = 57.25% / NLL = 0.9786`（较 `F_n2fix` 的 `56.69% / 0.9967` 显著提升），`Pythia6 NuX BA4 = 58.33% / NLL = 0.9728`，`Pythia6 Std BA4 = 56.17% / NLL = 1.0126`。
- **为何保留于 Stage-I**：`F_n2v2` 将 `1% FPR` 锁定的分类工作点推高至 `p_tau = 0.6330`（在该物理工作点下 `W184 NC Charm` 假阳性率仅为 `2.17%`，而名义未校准 `p_tau > 0.5` 处升至 `7.50%`；平均 `mean_p_tau = 0.2640` 与修复前 `0.2635` 完全持平）；同时 `PixDark 1e-5` 压力下因未加单像素热噪声过滤退化 `-12.52 pp`（`TrkDead 2%+N1 = -5.09 pp`）。因此 `F_n2v2` 作为可微运动学架构里程碑封存于 Stage-I，留待下一阶段接入单像素暗噪声门控与 `Score Bank` KL-DRO 尾部校准。

### 3.4 原型实验 Q13：全硬件等效 `1.00 mm` 像素传感器三向受控对比

数据来源：[q13_full_1mm_comparison.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/exp_n2_tau_kinema/q13_full_1mm_sensor/q13_full_1mm_comparison.json)（Condor 集群 `1585583`，`TrackletIPFormer` 架构，`seed = 42`，`internal_val_eval N = 4,800`）。`Tune6k` 及封存分片均为**未测量**。

| 传感器空间分辨率配置 | `All BA4` (%) | `All NLL` | `Pinpoint_PixXY NLL` (`N=2643`) | `Front30 BA4 / NLL` (`N=1909`) | `Interm150 BA4 / NLL` (`N=734`) | `Front30` `\|dz\|` / 3D (mm) | `Interm150` `\|dz\|` / 3D (mm) | `WS 1% FPR` $\nu_\tau$ 效率 (All / Front30, %) | `W184` `p_tau>0.5` (%) | `Run 10085-88` BA4 / NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| **`native_20um`（原生 `20.8 um` MAPS）** | **57.81** | **0.9659** | **0.9036** | **63.26 / 0.8652** | 55.93 / 1.0035 | 5.008 / 6.553 | 5.562 / 7.541 | **4.67 / 5.68** | **2.33** | **58.25% / 0.9771** |
| `centroid_only_1mm`（Stage 326 仅质心量化） | 57.00 | 0.9848 | 0.9327 | 62.68 / 0.8990 | 54.97 / 1.0205 | — | — | 3.67 / 3.37 | — | — |
| **`full_1mm`（质心 + `32x32` 图像块全 `1.00 mm` 体素）** | **57.83** | **0.9738** | **0.9194** | **62.53 / 0.8899** | **57.03 / 0.9960** | 4.524 / 6.130 | 5.746 / 7.308 | 4.25 / 5.05 | **4.08** | 57.38% / 0.9773 |

**1,000 次配对 family-block bootstrap 差值检验（`seed = 20263250`）**：
- **真实硬件效应 `native_20um - full_1mm`**：
  - `Front30`（30 mm 钨靶，单层约 `1.7 X_0`）：`ΔNLL = -0.0247`（95% CI **`[-0.0405, -0.0105]`**，严格排除零），`ΔBA4 = +0.73 pp [-0.80, +2.27] pp`；
  - `Pinpoint_PixXY`（全像素覆盖区）：`ΔNLL = -0.0158`（95% CI **`[-0.0280, -0.0030]`**，严格排除零）；
  - `Interm150`（150 mm 钨靶，单站节距 `20.23 X_0`）：`ΔNLL = +0.0075`（95% CI `[-0.0085, +0.0246]`，包含零），`ΔBA4 = -1.09 pp [-3.14, +1.03] pp`；
  - `All`（含 `37.4%` 无像素覆盖的 `Fortune_ScintOnly` 事例稀释）：`ΔNLL = -0.0079`（95% CI `[-0.0155, +0.0006]`），`ΔBA4 = -0.02 pp [-0.91, +0.79] pp`；
  - 核外推粲抑制：原生 `20 um` 空间分辨将 `W184 NC Charm` 假阳性率从 `full_1mm` 的 `4.08%` 压降至 **`2.33%`**（相对降幅 **42.9%**）。
- **伪影分离检验 `centroid_only_1mm - full_1mm`**：`All ΔBA4 = -0.83 pp`（95% CI `[-1.65, -0.01] pp`），`All ΔNLL = +0.0110`（95% CI **`[+0.0049, +0.0182]`**），`Interm150 ΔNLL = +0.0245`（95% CI **`[+0.0104, +0.0394]`**）。证实向网络同时馈入 `1 mm` 粗糙质心与未降采样的 `20 um` 精细图像块会诱发内部几何标度矛盾，而全硬件一致的 `full_1mm` 消除了该人为退化。

### 3.5 冻结主干 `Score Bank` KL-DRO 尾部微调：Stage-I 消融与 Stage-II `Tune6k` 确认

#### 3.5.1 Stage-I 候选目标函数筛选与校准保护验证（`internal_dev_screen` 选择，`internal_val_eval N = 4,800` 评估）

数据来源：[stage1_tail_dro_summary.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/exp_tail_dro/stage1/stage1_tail_dro_summary.json) 与 [stage1_unified_tail_dro_summary.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/exp_tail_dro/stage1/stage1_unified_tail_dro_summary.json)，可视化见 [fig8_tail_dro_stage1_ablations.png](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage327_figures/fig8_tail_dro_stage1_ablations.png)。

| 候选微调目标（冻结主干，`3 passes`，`lr=5e-5`） | `Tracklet` (B) `Val BA4 / NLL` | `Tracklet` `ECE_tau` | `Tracklet` `WS 1%` All (%) | `Tracklet` `W184 p>0.5` (%) | `Dual` (D) `Val BA4 / NLL` | `Dual` `ECE_tau` | `Dual` `WS 1%` All (%) | `Dual` `W184 p>0.5` (%) | Sec 10.2 校准门限（`ΔNLL<=0.005` & 无 ECE 倍增） |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| `pretrained_baseline`（未微调原模型） | 57.81% / 0.9659 | 0.0245 | 4.67 | 2.33 | 57.52% / 0.9829 | 0.0143 | 4.08 | 3.42 | 基线参考 |
| `pure_ce_3p`（纯交叉熵继续微调 3 passes） | 58.29% / 0.9746 | 0.0295 | 5.00 | **8.25** | 57.33% / 0.9902 | 0.0275 | 4.08 | **7.50** | **未通过**（Dev `ΔNLL = +0.0067/+0.0060`，W184 恶化 3x） |
| `pure_dro_no_anchor`（无锚定纯 CVaR-DRO） | 56.31% / 1.0294 | **0.0685** | 2.92 | 0.00 | 52.21% / 1.0618 | **0.0861** | 2.67 | 0.00 | **未通过**（`ΔNLL = +0.0635/+0.0789`，ECE 暴增 > 3x） |
| `dro_anchored_nobank`（KL 锚定但无 `Score Bank`） | 57.58% / 0.9692 | 0.0239 | 4.50 | 0.00 | 57.33% / 0.9870 | 0.0274 | 5.00 | 0.00 | **未通过**（小批量 `B=128` 仅 ~0.96 个 1% FPR 本底，Dev `ΔNLL = +0.0053`） |
| `sopa_l005_kl20_bank8192`（SOPA 替代目标） | 57.83% / 0.9704 | **0.0450** | 4.67 | **9.92** | 57.00% / 0.9895 | **0.0484** | 4.67 | **11.83** | **未通过**（Dev `ECE_tau` 膨胀 `2.73x / 2.42x`，触发回退） |
| `dro_bal_l003_kl15_bank8192`（$\lambda_{\text{DRO}}=0.03$） | 57.94% / 0.9663 | 0.0217 | 5.08 | 2.33 | 57.56% / 0.9838 | 0.0164 | 4.17 | 3.08 | 通过（抑制力度偏弱） |
| **`dro_bal_l008_kl15_bank8192`（$\lambda_{\text{DRO}}=0.08$，锁定）** | **57.92% / 0.9667** | **0.0196** | **5.33** | **0.92** | **57.73% / 0.9842** | **0.0191** | **4.42** | **0.67** | **通过并锁定**（Dev 平均 `pAUC(1%)` 最高，`Val ΔNLL <= +0.0013`） |
| `dro_bal_l015_kl15_bank8192`（$\lambda_{\text{DRO}}=0.15$） | 57.81% / 0.9675 | 0.0151 | 4.58 | 0.25 | 57.58% / 0.9851 | 0.0203 | 4.50 | 0.08 | 通过（尾部召回略低于 $\lambda_{\text{DRO}}=0.08$） |

在 Stage-I `internal_val_eval` 上，锁定配置 `dro_bal_l008_kl15_bank8192` 使 `50/50 Blend` 的 `BA4` 从 `58.54%` 升至 **`58.79%`**，`NLL` 保持 `0.9596`（`ΔNLL = +0.0003`），`Within-Slice 1% FPR` 全样本 $\nu_\tau$ 效率从 `5.00%` 升至 **`5.58%`**（`Split-Half 1% FPR` 从 `5.09%` 升至 **`5.42%`**）。

#### 3.5.2 Stage-II `Tune6k`（`N = 6,000`，3 seeds `{17, 42, 2026}`）独立确认结果

数据来源：[stage2_tail_dro_summary.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/exp_tail_dro/stage2/stage2_tail_dro_summary.json) 与独立重算 [tail_dro_independent_audit.json](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage327_verification/tail_dro_independent_audit.json)（Condor 集群 `1585596`、`1585632`，全程零超参修改），可视化见 [fig9_tail_dro_stage2_tune6k.png](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage327_figures/fig9_tail_dro_stage2_tune6k.png) 与 [fig10_stage327_integrated_summary.png](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/stage327_figures/fig10_stage327_integrated_summary.png)。

| 模型（3-seed 集成，`N = 6,000`） | 校准模式 | `All BA4` (%) | `All NLL` | `ECE_top1` / `ECE_tau` | `D1_obs BA4` (%) | `pAUC(1%)` All / `D1_obs` | `WS 1% FPR` All (`N_tau=1494`) | `WS 1% FPR` `D1_obs` (`N_tau=312`) | `WS 1% FPR` `Pinpoint` (`N_tau=801`) | `WS 0.1% FPR` All (`N_tau=1494`) | `SH 1% FPR` `D1_obs` (`N_tau=149`) | `W184 Charm` `p_tau>0.5` (%) |
| --- | --- | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `TrackletIPFormer` (Pretrained) | `raw` | 58.11 | 0.9471 | 0.0168 / 0.0107 | 63.14 | 0.0342 / 0.0361 | 6.43% (96) | 5.13% (16) | 6.37% (51) | 1.41% (21) | 8.05% (12) | 1.58 |
| `TrackletIPFormer + KL-DRO` | `raw` | **58.15** | **0.9468** | 0.0189 / 0.0125 | **63.20** | **0.0410 / 0.0431** | 6.09% (91) | **6.09% (19)** | **7.12% (57)** | **1.94% (29)** | **11.41% (17)** | **0.33** |
| `DualRegimeModalNet` (Pretrained) | `raw` | 58.07 | 0.9497 | 0.0202 / 0.0157 | 62.57 | 0.0473 / 0.0683 | 7.03% (105) | 9.94% (31) | 8.74% (70) | 2.88% (43) | 8.05% (12) | 2.25 |
| `DualRegimeModalNet + KL-DRO` | `raw` | **58.50** | **0.9493** | **0.0105 / 0.0083** | **63.01** | **0.0496 / 0.0831** | **7.23% (108)** | **11.86% (37)** | **9.24% (74)** | 2.74% (41) | **11.41% (17)** | **0.58** |
| **`50/50 Blend` (Pretrained)** | **`raw`** | **58.50** | **0.9380** | **0.0206 / 0.0112** | **64.08** | **0.0448 / 0.0498** | **7.23% (108)** | **6.73% (21)** | **8.49% (68)** | **2.34% (35)** | **8.72% (13)** | **1.67** |
| **`50/50 Blend + KL-DRO`（最终基线）** | **`raw`** | **58.76** | **0.9374** | **0.0145 / 0.0143** | **64.44** | **0.0494 / 0.0737** | **7.83% (117)** | **11.86% (37)** | **9.61% (77)** | **2.74% (41)** | **11.41% (17)** | **0.33** |
| `50/50 Blend` (Pretrained) | `screen_cal` | 59.01 | 0.9392 | 0.0289 / 0.0129 | 64.43 | 0.0473 / 0.0591 | 6.89% (103) | 8.65% (27) | 8.11% (65) | 2.41% (36) | 10.07% (15) | — |
| **`50/50 Blend + KL-DRO`** | **`screen_cal`** | **59.01** | **0.9390** | **0.0260 / 0.0159** | **64.52** | **0.0498 / 0.0804** | 6.36% (95) | **11.86% (37) / 12.82% (40)*** | 8.11% (65) | **2.68% (40)** | **14.09% (21)** | — |

*\*注：`screen_cal` 下复用微调前 `Dev` 温度时 `D1_obs` 为 `11.86% (37/312)`；在 `internal_dev_screen` 上对 DRO 微调后对数几率重新拟合 `T_screen`（独立审计 `tail_dro_independent_audit.json` 口径）时达到 `12.82% (40/312)`，相对校准前基线 `8.65% (27/312)` 净增 `+4.17 pp`（1,000 次配对 bootstrap 95% CI **`[+0.31, +6.33] pp`**，严格排除零）。*

**Stage-II 关键配对 Bootstrap 与 6 OOD 分片稳定性核验**：
- **`DualRegimeModalNet` 单模型 `raw` 增益**：`All_Tune6k ΔBA4 = +0.43 pp`（95% CI **`[+0.09, +0.76] pp`**，严格排除零），`Pinpoint_PixXY ΔBA4 = +0.42 pp`（95% CI **`[+0.04, +0.81] pp`**），`Front30 ΔBA4 = +0.58 pp`（95% CI **`[+0.10, +1.05] pp`**），同时 `ECE_top1` 减半（`0.0202 -> 0.0105`）。
- **`50/50 Blend` 6 组 OOD 外推全面改善或持平**：
  - `W184 EPPS21 NC Charm`（`N = 1,200`）：`p_tau > 0.5` 从 `1.67% (20/1200)` 降至 **`0.33% (4/1200)`**（降低 **5.0 倍**）；
  - `W184 EPPS21 NC Inclusive`（`N = 1,200`）：`p_tau > 0.5` 从 `1.83% (22/1200)` 降至 **`0.17% (2/1200)`**（降低 **11.0 倍**）；
  - `Pythia8 Free-Nucleon NC Charm`（`N = 800`）：`p_tau > 0.5` 从 `2.00% (16/800)` 降至 **`1.00% (8/800)`**（降低 **2.0 倍**）；
  - 4 味均衡外推集：`Run 10085-10088 BA4` 保持 `59.25%`，`NLL` 从 `0.9560` 改善至 `0.9533`；`Pythia6 NuX BA4` 保持 `61.08%`（`NLL = 0.9367`）；`Pythia6 Std BA4` 从 `57.08%` 升至 **`57.42%`**（`NLL` 从 `0.9764` 改善至 `0.9761`）。

### 3.6 关键复现命令与计算效率记录

在仓库根目录执行（需加载 CVMFS `LCG_108_cuda` 环境）：

```bash
# 1. 运行新增及既有 NextGen 自动化测试套件
set -- && source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh >/dev/null 2>&1 \
  && PYTHONPATH=src python3 -m unittest discover -s tests -p "test_nextgen_*.py"

# 2. 实验 N1：Stage-I PinpointUnifiedNet 训练与压力/OOD 完整评估（seed=42）
PYTHONPATH=src python3 -m analysis.flavour.nextgen.run_unified_pinpoint_net \
  --mode train_eval --ablation full --seed 42 \
  --cache-dir /eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/exp_unified_net/cache_r2 \
  --out-dir /eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/exp_unified_net

# 3. 实验 N2 & Q13：TauKinemaDiffNet v2 顶点精修器与全 1.00 mm 传感器受控实验
PYTHONPATH=src python3 -m analysis.flavour.nextgen.run_tau_kinema_diffnet \
  --run-name s1_tau_kinema_diffnet_n2v2 --n2-vtx-mode v2 --seed 42 \
  --out-dir /eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/exp_n2_tau_kinema
PYTHONPATH=src python3 -m analysis.flavour.nextgen.run_q13_full_1mm_sensor \
  --seed 42 --out-dir /eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/exp_n2_tau_kinema/q13_full_1mm_sensor

# 4. 冻结主干 Score-Bank KL-DRO：Stage-I 候选筛选与 Stage-II Tune6k 3-seed 确认及汇总绘图
PYTHONPATH=src python3 -m analysis.flavour.nextgen.run_tail_dro_finetune --stage stage1 \
  --out-dir /eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/exp_tail_dro/stage1
PYTHONPATH=src python3 -m analysis.flavour.nextgen.run_tail_dro_finetune --stage stage2 \
  --out-dir /eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/exp_tail_dro/stage2
PYTHONPATH=src python3 -m analysis.flavour.nextgen.plot_stage327_summary
```

---

## 4. 阶段性结论

### 4.1 实测事实（Facts）

1. **冻结主干 `Score Bank (8,192)` + KL/NLL 锚定 CVaR-DRO 解决了极低假阳性区梯度稀疏与概率失准的固有矛盾**：实测消融证明，无锚定 DRO 会使验证集 `NLL` 崩塌 `+0.064 ~ +0.079` 且 `ECE_tau` 恶化超 `3 倍`；无 `Score Bank` 在 `batch_size = 128` 下无法稳定解析 `1% FPR` 分位点；唯有结合 `8,192` 历史本底队列与 `KL(p_0 || p) + CE` 锚定（`dro_bal_l008_kl15_bank8192`），才能在 `Tune6k` 上同时实现 `NLL` 改善（`0.9380 -> 0.9374`）、`D1_obs 1% FPR` $\nu_\tau$ 效率跃升（`raw 6.73% -> 11.86%`，`screen_cal +4.17 pp [+0.31, +6.33] pp`）以及 `W184 NC Charm` 假阳性率压降 5 倍（`1.67% -> 0.33%`）。
2. **多模态单模型端到端训练与解耦双模型集成存在可量化的“抗噪鲁棒性—干净样本判别率”权衡**：`PinpointUnifiedNet` 在干净数据上（Round-1 `abl_no_online_spec_curriculum` `58.60% / 0.9602`）已证明 `7.57M` 单模型容量足以匹敌 `12.85M` 解耦 `50/50 Blend`（`58.54% / 0.9593`）；加入斜率一致在线假三元组课程与单像素热噪声过滤后，硬件压力退化被压制至 `-0.07 pp`（`TrkDead 2%+N1`），但单主干共享表征必须同时容纳干净径迹几何与强噪声不变性，导致干净样本 `All BA4` 回落至 `57.48%`。解耦 `50/50 Blend` 通过几何主导（`TrackletIPFormer`）与量能/拓扑主导（`DualRegimeModalNet`）两条独立梯度路径的对偶平均，在准确率、对数似然与校准度上仍显著占优。
3. **`TauKinemaDiffNet` 在 150 mm 厚钨靶区的顶点失效由跨区段像素门控泄漏与钨级联反照率共同导致，对称锚定精修器已彻底修复该几何瓶颈**：仅施加 `first_tgt_p <= 5` 门控只能将 `Interm150 |dz|` 降至 `10.215 mm`；进一步剔除单平面钨反散射假触发并引入 `_RegimeAwareSymmetricVertexRefiner` 后，`Interm150 |dz|` 降至 `1.984 mm`（`Front30 = 2.136 mm`），单模型 `BA4` 跃升 `+1.69 pp` 至 `58.46%`，三模型融合 `Blend_BDF_n2v2` 将 Stage-I `NLL` 推进至 `0.9451`。
4. **亚毫米像素分辨率（`20 um` vs `1.00 mm`）的物理收益严格受局部辐射长度调制**：在 `Front30` 薄钨靶区（单层约 `1.7 X_0`），`20 um` MAPS 提供显著对数似然优势（`ΔNLL = -0.0247 [-0.0405, -0.0105]`）与核粲背景抑制（`W184` 假阳性率从 `4.08%` 降至 `2.33%`）；在 `Interm150` 厚钨靶区（层间 `20.23 X_0`），多重库仑散射抹平精细径迹倾角信息，两者无统计显著差异。
5. **位移锚定反冲半球特征 `C_hem` 是抑制核外推粲强子伪装 $\nu_\tau$ 的关键物理不变量**：在 N1 两轮受控消融中，移除 `C_hem` 均导致 `W184 NC Charm` 假阳性率翻倍恶化（Round-1：`2.67% -> 7.08%`；Round-2：`3.58% -> 5.83%`）。

### 4.2 物理与优化机制假设（Hypotheses，待后续验证）

1. **`F_n2v2` 接入 `Score Bank` KL-DRO 与单像素热噪声过滤的协同增益假设**：`F_n2v2` 在名义阈值 `0.5` 处的 `W184 p_tau > 0.5 = 7.50%` 并非物理误判率恶化（其在 `1% FPR` 实际工作点 `p_tau > 0.633` 处假阳性率仅 `2.17%`，且 `mean_p_tau = 0.2640` 未变），而是 12 类层次化头部在高信噪比顶点特征下的尾部对数几率右偏。假设在不触碰主干权重的前提下对 `F_n2v2` 施加与 Stage 327 完全相同的单像素暗噪声预过滤及冻结主干 `Score Bank` KL-DRO 微调，可在保持 `Interm150 |dz| ~ 2.0 mm` 与 `Blend_BDF NLL ~ 0.945` 优势的同时，将其 `W184 p_tau > 0.5` 压回 `< 1.0%` 并将 `PixDark` 退化控制在 `< 2.0 pp`。
2. **封存分片校准传递与系统误差稳健性假设**：假设当前在 `Tune6k` 上锁定的 `50/50 TrackletIPFormer + DualRegimeModalNet + Score-Bank KL-DRO` 模型在后续一次性解封流程（`Cal-A -> Cut-B -> Confirm`）中，结合 `Cal-A` 后验温度校准与 `Cut-B` 分片锁定阈值，能够在真实探测器系统误差扰动集 `input_sys_confirm*` 上保持 `D1_obs` 超过 `10%` 的 `1% FPR` $\nu_\tau$ 发现效率且不出现超出统计涨落的假阳性膨胀。该假设目前在封存集上**未测量**，必须通过严格的预注册单次解封实验检验。

---

## 5. 下一步计划

1. **Stage 328A（Stage-I 受控补充验证：`TauKinemaDiffNet v2` 暗噪声硬化与三路 KL-DRO 评估，零接触封存集）**：
   - **具体任务**：在 `TauKinemaDiffNet v2` 像素团簇入口复用 N1 Round-2 已验证的孤立 1-bit 单像素暗噪声过滤门控（`n_hits == 1`、同平面邻域 `cnt_r040 == 0 & cnt_r1 == 0` 且相邻两层无 `dp in {1, 2}` 紧符合 MIP 伴随击中），并在 `Train/Dev/Val` 上对冻结主干的 `B + D + F_n2v2` 执行 `Score Bank (8,192)` KL-DRO 微调。
   - **所需输入**：`shards 000..079`（`Train / internal_dev_screen / internal_val_eval`）与 6 组非封存 OOD 分片。
   - **验收标准**：Stage-I `Interm150 |dz| <= 2.5 mm`，`PixDark 1e-5` 退化 `< 2.5 pp`，`W184 NC Charm p_tau > 0.5 <= 1.0%`，且三路融合在 `internal_val_eval` 上较二路 `50/50 Blend + KL-DRO` 取得统计显著的 `NLL` 或 `1% FPR` $\nu_\tau$ 效率提升；若未达标，维持当前已通过 Stage-II 确认的二路 `50/50 Blend + KL-DRO` 为唯一冻结候选。
2. **Stage 328B（预注册一次性封存解封前演练与协议冻结）**：
   - **具体任务**：在保持 `shards 100..149` 零读取的前提下，编写单向确定性解封流水线脚本与单元测试（模拟合成数据演练 `Cal-A (100..109)` 温度拟合 $\to$ `Cut-B (110..124)` Clopper-Pearson 阈值锁定 $\to$ `Confirm (125..149)` 与 `input_sys_confirm*` 一次性只读评估），经用户审阅批准后方可执行正式解封。
   - **所需输入**：Stage 327 Stage-II 锁定的 6 个 DRO 微调检查点（`TrackletIPFormer` 与 `DualRegimeModalNet` 各 `{17, 42, 2026}` 三种子，位于 [exp_tail_dro/stage2](file:///eos/home-x/xcheng/FASER/Pinpoint/results/nextgen_research/exp_tail_dro/stage2)）。
   - **验收标准**：合成演练通过 100% 确定性复现检查、`seal_guard` 权限分级验证通过、零超参自由度残留。
