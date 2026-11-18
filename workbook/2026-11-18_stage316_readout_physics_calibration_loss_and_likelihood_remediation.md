# 2026-11-18 Stage 316：分析代码物理读出、谱仪动量、无偏损失、C++17 多径迹次级顶点与似然推断全面修复

## 核心结论

1. **单通道闪烁体读出与零 `edep` 泄露（解决 `P1-1` 分析侧）**：在 `domains.py`、`v1_b1pp/representation.py` 与 `v2_pinpointformer/pixel_tokenizer.py` 中实现 `calibrated_bar_channels(..., pe_only=True)`，完全阻断对未测量 `bar_edep_proxy_MeV` 的读取（单元测试将 `bar_edep_proxy_MeV` 扰动 $10^5$ 倍后 `B1++` 与 `PinpointFormer` 特征张量逐位精确一致，`atol = 0.0`），同步施加 `>= 2.5 PE` 光电子阈值与 `14,400` 微元 SiPM 饱和校正，并在 `pe_only=True` 时默认关闭非物理对称的 `y_mirror` 增广（解决 `P2-4`）。
2. **谱仪矢高动量公式、无偏异方差损失、AdamW 权重衰减解耦与对称消歧全部通过验证（解决 `P1-3`、`P2-2`、`P2-3`、`P2-4`、`P3-1`、`P3-2`）**：
   - `nomad_kinematics.py` 启用第一性原理及 Geant4 标定常数 `SAGITTA_CONST_MEV_MM = 100705.0`（3 站径迹 $p_{\text{spec}} = 100705 / (|s_y| + 0.015)\text{ MeV}$，2 站无矢高径迹严格置零）并配对钨/闪烁体采样因子 `cal_e_scale = 12.5`，修复原先 `204.5x–242x` 的动量标度压缩。
   - `vertex_anchor.py` 与 `v3_multitask/losses.py` 移除异方差指数项内部的 `+2.0` / `+0.5` 正偏置并新增 `FixedWeightEventLoss`，使 `log_var` 驻点精确对应真实残差尺度（`20 um` 残差下收敛至 `-5.28`，修复前被锁死在 `+0.696`）。
   - `layers.py` 及全部 5 类模型（`B1++`、`PF`、`PFMultitask`、`B1++V4`、`PFV4`）将所有 1D 参数、`LayerNorm`、`log_scale`、`len_beta`、门控标量与可学习查询向量的 `weight_decay` 严格设为 `0.0`，并在 `attention.py` / `probes.py` 中补齐考虑 `len_beta` 后的有效注意力锐度汇报。
   - `stereo_bars.py` 引入 `1 - |u_bars|` 对称消歧与 `-1e6` 非激活掩码，双邻条等能并列（`bar 18, 19` 镜像至 `bar 20, 21`）Y 反射跳变由 `10.500 mm` 降至 `0.0000 mm`（`< 1e-4 mm`）。
   - `roadmap/v3/v5_freeze_manifest.json` 补齐解封前提交哈希 `1f89d279cc420f93b802bd1ff80975dcbc287c2b`（`freeze.py` 新增 40 位 SHA 强制校验），`calibration.py` 与 `ensemble.py` 增加 L-BFGS-B 多起点重试与收敛标志，`domains.py` 增加 `validate_finite_observation` 显式拦截 `NaN/Inf`。
3. **C++17 原生加速算子与显式微米多径迹/自适应主次顶点重建（`Path C`）通过真样本与 GPU 门禁**：
   - `_fast_kernels.cpp` 在 120 个真实多平面像素事例上与原 Python 实现逐位精确一致，像素连通域聚类提速 **`6.3x`**（`219.27 ms -> 34.65 ms`），多半径平面邻域统计提速 **`46.2x`**（`5,547.79 ms -> 120.13 ms`）。
   - 同时实现基于 3D 多径迹种子、3 轮 Tukey/Fermi–Dirac 自适应主顶点拟合（AVR）、冲击参数显著性 $d_0/\sigma_{d_0}$、双顶点 $\Delta\chi^2$、衰变长度 $(L_{xy}, L_z)$ 与单叉折角 $\theta_{\text{kink}}$ 的 16 维微米拓扑重建，并接入 `PinpointFormer(enable_micro_vertex=True)`。
   - 本地 `Tesla T4` GPU 32 事例真实样本过拟合与分支梯度健康门禁一次性全部通过：`PF_V2_pe_only_micro_vertex`（第 `21` 步收敛，`final_NLL = 0.0068`，`acc = 1.000`）、`PF_V3_fixed_event_loss_pe_only`（第 `17` 步收敛，`final_NLL = 0.0158`）、`PF_V4_nomad_phys_pe_only`（第 `15` 步收敛，`final_NLL = 0.0263`），`bs=64` 峰值显存仅 `5,650 MiB`（支持 A100-40GB / H100 单卡 `3` 进程并发训练），全部子分支梯度有限且严格正定。

---

## 阶段目标

1. 在不触碰封存集的前提下，一次性修复 Stage 314 审查报告中所有分析、特征工程、模型架构、损失函数、优化器正则化与统计推断代码问题（`P0-1`、`P0-2` 几何感知反符合、`P1-1`、`P1-3`、`P1-4`、`P2-2`、`P2-3`、`P2-4`、`P3-1`、`P3-2`）。
2. 构建 C++17 高性能像素特征提取与显式微米多径迹/次级顶点拓扑重建模块（`Path C`），消除本地与批处理数据预处理瓶颈。
3. 完成全部新旧自动化单元测试（`43` 项）与本地 GPU 32 事例小样本拟合及梯度健康检验，确保修复后的所有模型无常数输出、无梯度消失且具备高吞吐单卡多进程并发能力。

---

## 工作内容

1. **物理单通道读出、几何感知反符合与非有限输入防护（`domains.py`、`datasets.py`、`group.py`、`representation.py`、`pixel_tokenizer.py`）**
   - 实现 `validate_finite_observation`：对像素 `xyz_mm`、闪烁体 `bounds_mm`/`pe_proxy` 及谱仪 `xy_mm`/`sigma_xy_mm` 进行显式有限性校验。
   - 实现 `calibrated_bar_channels`：`pe_only=True` 时仅读取 `bar['pe_proxy']`，施加 `sipm_npix = 14400` 饱和映射与 `bar_threshold_pe = 2.5` 硬件阈值，并在 95% 占空比内做保守逆饱和标定为等效可见能量，完全弃用 `edep_proxy_MeV`。
   - 将上游反符合条件改为动态比较 `scint_Zpos[0] < z_w0` 与 `pixel_Zpos[0] < z_w0`，兼容修复前后两套几何配置；在 `group.py` 中透传所有读出配置键以保证单卡多进程共享缓存精确匹配。
2. **C++17 原生算子与显式微米多径迹/次级顶点混合架构（`_fast_kernels.cpp`、`fast_kernels.py`、`pixel_clusters.py`、`v2_pinpointformer/models.py`）**
   - 使用 C++17 编写连通域像素聚类、单遍平面多半径邻域统计、`32x32` 微补丁栅格化及 3D 多径迹/AVR 主顶点/次级顶点冲击参数与折角重建算子，编译产物缓存于 `/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v4_audit_2026-11-18/native_bin/`，并在 `submit.py` 源码归档中纳入 `*.cpp`。
   - 在 `PinpointFormer` 中新增 `enable_micro_vertex` 模式：以 AVR 主顶点作为 Stage-1 `VertexAnchorPredictor` 的亚毫米先验，同时将 16 维冲击参数/次级顶点/折角向量经 `micro_vtx_token` 送入 Perceiver 跨模态注意力并经 `micro_vtx_proj` 门控残差融合至全局表征。
3. **谱仪动量标定、无偏损失、AdamW 参数分组与镜像对称修复（`nomad_kinematics.py`、`vertex_anchor.py`、`v3_multitask/losses.py`、`layers.py`、`stereo_bars.py`、`attention.py`、`probes.py`）**
   - `NomadTransverseExtractor` 默认启用 `phys_calibrated=True`：采用 `SAGITTA_CONST_MEV_MM = 100705.0` 从 `cand0[:, 13]` 还原 3 站矢高动量，2 站径迹置零，并将闪烁体沉积乘以 `12.5` 还原总簇射能量标度。
   - `heteroscedastic_vertex_loss` 将常数偏置移至指数项外部并采用 `(0.20, 0.20, 2.50) mm` 物理轴尺度；新增无偏 `FixedWeightEventLoss`。
   - 新增 `is_weight_decay_exempt` 与 `split_adamw_param_groups` 并接入 `B1++`、`PF`、`PFMultitask`、`B1++V4`、`PFV4`；新增 `collect_effective_attention_sharpness` 输出序列长度 $N=20, 200$ 下的真实注意力缩放因子。
   - `StereoBarNexus` 采用 `1 - |u_bars|` 对称消歧消除 `torch.topk` 低索引偏置。
4. **前向通量剖面似然、带号系统误差模板变形、粲夸克控制区联合拟合与工作点标注（`flux_significance.py`、`significance.py`、`run_exp6_flux_audit.py`、`calibration.py`、`ensemble.py`、`freeze.py`）**
   - 依据 Trench GENIE $10\text{ ab}^{-1}$ 实测截面产额建立真实前向通量场景（$f_\tau = 0.2090\%$，本底组分 $\nu_e:\nu_\mu:\text{NC} = 11.66\%:64.28\%:24.06\%$），构建带正负号的上下系统误差对数变形向量、分域斜率讨厌参数、$\pm 30\%$ 粲产生率讨厌参数、6-bin 低 $S_\tau$ 高 `charm_prob` 控制区联合拟合（`Path D`）以及解析闭式内层 Barlow–Beeston lite 蒙特卡洛统计不确定度剖面化与 $H_0$ 玩具蒙特卡洛校准检验。
   - 在 `compute_locked_slice_report` 中显式区分 `within_slice_exact_1pct_fpr` 与 `global_budget_variable_local_fpr`（解决 `P1-4`）。

---

## 详细数据

- **执行命令与验证产物**
  - 单元测试命令：`python3 -m unittest tests/test_stage316_audit_fixes.py tests/test_v3_domains.py tests/test_v3_truth_isolation.py tests/test_v3_b1pp.py tests/test_v3_pinpointformer.py tests/test_v3_multitask.py tests/test_v3_v4_tau.py tests/test_v3_v5_confirm.py -v`
  - 单元测试结果：全部 `43/43` 项测试通过（`OK`，其中 `test_stage316_audit_fixes.py` `9/9` 项 `20.80 s` 通过）。
  - GPU 门禁报告路径：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v4_audit_2026-11-18/stage316_smoke/stage316_smoke_summary.json`

- **表 1：C++17 原生算子基准性能与一致性（120 个真实多平面像素事例）**

| 算子模块 | 原 Python 耗时 (`ms`) | C++17 `_fast_kernels` 耗时 (`ms`) | 加速比 | 数值一致性最大偏差 |
| --- | --- | --- | --- | --- |
| `cluster_pixel_hits`（连通域聚类） | `219.27` | `34.65` | `6.3x` | `0`（整数与浮点完全一致） |
| `plane_neighborhoods`（平面多半径统计） | `5,547.79` | `120.13` | `46.2x` | `0.0`（双精度机器精度内一致） |

- **表 2：本地 `Tesla T4` GPU 32 事例小样本拟合与全分支梯度健康检验**

| 模型变体 | 参数量 | 达标更新步数 (`NLL<0.05, acc=1`) | 最终 NLL | 最终准确率 | `bs=64` 峰值显存 (`MiB`) | `bs=64` 吞吐率 (`ev/s`) | 全分支梯度正定有限 & 参数更新 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `PF_V2_pe_only_micro_vertex` | `8,698,069` | `21` | `0.0068` | `1.000` | `5,648.9` | `145.6` | `True / True` |
| `PF_V3_fixed_event_loss_pe_only` | `8,881,436` | `17` | `0.0158` | `1.000` | `5,649.3` | `144.2` | `True / True` |
| `PF_V4_nomad_phys_pe_only` | `8,996,477` | `15` | `0.0263` | `1.000` | `5,650.7` | `139.6` | `True / True` |

---

## 阶段性结论

- **已验证事实**：
  1. 修复后的单通道 `PE` 读出、校准谱仪动量、无偏顶点/多任务损失、零 1D 权重衰减与显式微米多径迹/次级顶点特征均已通过确定性单元测试和真实 GPU 拟合门禁，消除了 Stage 314 指出的全部软件、特征工程与正则化缺陷。
  2. 单进程训练峰值显存仅 `5.65 GiB`，C++17 算子将 CPU 像素特征构建开销压缩了一个数量级以上，完全满足在单张 A100/H100 GPU 上并发运行 `2–3` 个训练进程的吞吐与显存预算。
- **待检验物理假设（将在 Stage 317–319 中通过受控多卡实验验证）**：
  1. 在严格相同训练暴露（`300k×24p`）与 3 随机种子下，剥离调度混杂后 `GN2` 稠密头与校准 `NOMAD` 头是否存在超过 `+1.0 pp` 的真实统计增益（`Exp-1`、`Exp-5`）。
  2. 移除 `edep` 双通道坐标泄露并施加 Birks/SiPM 饱和与 `2.5 PE` 阈值后，`PinpointFormer` 相比投影基线 `B1++` 在像素包含域与闪烁体域的真实性能差值（`Exp-3`）。

---

## 下一步计划

1. **Stage 317（`Exp-1`、`Exp-3`、`Exp-5`、`Exp-6` 受控消融与真实通量显著性评估）**：完成 `Exp-6` 2,000 玩具蒙特卡洛覆盖率与前向通量显著性扫描；同时在 Kubeflow 与 HTCondor 多张 A100/H100 GPU 上以单卡并发模式运行 `Exp-1`（同 `24p` 预算解混杂）、`Exp-3`（单通道 `PE >= 2.5` + SiPM 饱和重训）与 `Exp-5`（`100705/|s_y|` 物理标定 NOMAD 头对照），全部仅使用 `Train300k / Tune40k` 做严格门禁决策。
2. **Stage 318 & Stage 319（`Exp-4` 干净 Geant4 样本批量生产 + `Exp-2` 分辨率重训曲线 + `Path C` 次级顶点混合架构与封存集一次性确认）**：利用 HTCondor 空闲 AlmaLinux9 多核节点并行生产几何对齐、包含 `>2.31 TeV` $\nu_\tau$ CC 与 Pythia8 `NC_charm` 的完整 5 划分干净数据集，随后多卡并发完成从头训练的像素螺距/层数曲线、微米次级顶点混合模型对比及一次性封存集解封确认。
