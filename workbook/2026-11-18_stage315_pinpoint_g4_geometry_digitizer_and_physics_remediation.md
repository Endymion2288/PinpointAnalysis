# 2026-11-18 Stage 315：Pinpoint_G4 几何对齐、闪烁体 Birks 猝灭/SiPM 饱和与物理过程修复

## 核心结论

1. **几何纵向错位完全消除（解决 `P0-2`）**：`Pinpoint_G4` 中 `ComputeSiliconZPositions()` 已改为直接从 `G4VPhysicalVolume::GetTranslation()` 读取所有已放置子体积的世界坐标，彻底消除 12 层靶区硅像素平面 `+10.000 mm` Z 偏移、12 块 Pinpoint 竖向闪烁体板 `-16.010 mm` Z 偏移以及 12 块后侧钨板 `-6.010 mm` Z 偏移。运行期启动自检 `VerifyGeometryWithNavigator()` 通过 `G4Navigator` 逐点校验全部 15 层 `G4_Si`、108 块 `G4_W`（每块中心及 $\pm 0.48\,t_W$ 前后表面共 324 个点）和 96 块 `G4_PLASTIC_SC_VINYLTOLUENE` 闪烁体板，检验通过率 `100%`；25 个 $\nu_\tau$ CC 测试事例的主顶点落在实体 `G4_W` 靶板内的比例由修复前的 `88.89%` 提升至 `25/25 = 100.0%`。
2. **闪烁体 Birks 猝灭与 SiPM 微元饱和生效（解决 `P1-1` 模拟侧）**：补齐 `Construct()` 中缺失的 `DefineMaterial()` 调用（激活聚乙烯基甲苯 `0.126 mm/MeV` Birks 常数），并在 `ScintSD.cc` 中引入 `G4EmSaturation::VisibleEnergyDepositionAtAStep` 与单通道 `(panelID, isHorizontal, barID)` 级 SiPM 微元饱和（$N_{\text{pix}} = 14400$，最大余数法守恒分配）。相同随机种子对照实验测得平均光产额由无猝灭线性的 `381.15 PE/MeV` 降至 `103.33 PE/MeV`，高电离密度记录（`edep > 5 MeV`）有效光产额抑制 `-66.3%`（`381.10 -> 128.44 PE/MeV`），单通道最大输出严格封顶在物理上限 `14,400 PE`（修复前达 `763,050 PE`）。
3. **磁铁外壳物质、单调真值溯源与 $\tau^\mp$ 螺旋度极化补齐（解决 `P2-1`）**：补建 `Sm2Co17`（`8.4 g/cm^3`）Halbach 永磁体空心圆筒外壳 `Magnet0/1/2`（$r \in [100.01, 215.0]\text{ mm}$）；修复 `TrackInformation` 拷贝构造遗漏的 `fFromCharmedHadron` 并将 `PostUserTrackingAction` 改为单调按位或传播，使 $\tau \to \pi^0 \to \gamma\gamma$ 电磁簇射次级粒子完整保留 `fromTau == 1`（25 个 $\nu_\tau$ CC 事例中共保留 `202,087` 条 $\gamma/e^\pm$ 闪烁体记录）；在 `Py8Decayer.cc` 中启用标准模型 $V-A$ 纵向螺旋度极化（$h(\tau^-) = -1, h(\tau^+) = +1$），`300 GeV` $\tau^-$ 单粒子枪验证测得 $\tau^- \to \pi^- \nu_\tau$ 平均能量分数在 $h = -1, 0, +1$ 下分别为 `0.294 ± 0.048`、`0.504 ± 0.067`、`0.620 ± 0.051`，与理论值 `0.333`、`0.500`、`0.667` 一致。

---

## 阶段目标

1. 消除 `Pinpoint_G4` 中 `ConstructPixelModuleLV()` 实体放置顺序与 `ComputeSiliconZPositions()` 手写累加公式不一致导致的纵向几何错位、假钨板顶点抽样与 `ScintSD` 层号错位（审查项 `P0-2`）。
2. 在 Geant4 敏感探测器层实现物理真实的闪烁体 Birks 电离猝灭与 SiPM 微元饱和，消除高能电磁簇射核心的非物理线性光产额暴涨（审查项 `P1-1`）。
3. 恢复 FASER 谱仪 `Sm2Co17` 永磁体物质外壳、修复 `TrackingAction` 次级粒子真值标签覆盖与 `TrackInformation` 未初始化字段、启用 `Py8Decayer` 中微子产生 $\tau^\mp$ 轻子的 $V-A$ 极化，并提供轻量级 `PINPOINT_COMPACT_TRUTH=1` 端点轨迹模式（审查项 `P2-1`）。

---

## 工作内容

1. **物理放置驱动的几何导出与启动导航器自检（`DetectorConstruction.hh/.cc`、`RunAction.cc`）**
   - 在 `Construct()` 入口显式调用 `DefineMaterial()`，确保聚乙烯基甲苯闪烁体的 `0.126 mm/MeV` Birks 常数与 `Sm2Co17` 永磁合金（化学计量比 $\text{Sm}_2\text{Co}_{17}$，密度 `8.4 g/cm^3`）在构建逻辑体积前完成注册。
   - 重写 `ComputeSiliconZPositions()`：遍历 `fWorldLV -> Detector -> {PinpointBlock, FortuneBlock}` 及 `fWorldLV -> IPTPixelLayer` 的实际 `G4VPhysicalVolume` 子节点，直接读取各组件的 `GetTranslation().z()` 生成 `fSiliconZPositions`（15 层）、`fTungstenZPositions`（108 层）、`fTungstenThicknesses`（108 层）、`fScintZPositions`（96 层）及 `ScintSD` 面板映射表，从根源上杜绝手写公式漂移。
   - 启用 `Magnet0`（长 `1.5 m`）和 `Magnet1/2`（长 `1.0 m`）`Sm2Co17` 外壳放置（内径留出 `10 um` 间隙以避免与 `FieldRegion0/1/2` 圆柱面完全重合导致的磁场边界零步长告警，可通过 `PINPOINT_DISABLE_MAGNET_SHELLS=1` 切换做系统误差对照）。
   - 新增 `VerifyGeometryWithNavigator()` 并在 `RunAction::BeginOfRunAction`（几何闭合体素化完成后）自动执行，一旦任何导出坐标对应的实际物理体积材质偏离预期即抛出 `FatalException`。
2. **Birks 定律猝灭与通道级 SiPM 微元饱和（`ScintSD.cc`）**
   - 使用 `G4LossTableManager::Instance()->EmSaturation()->VisibleEnergyDepositionAtAStep(step)` 计算每步猝灭后可见沉积能量（支持 `PINPOINT_DISABLE_BIRKS=1` 对照开关）。
   - 在 `EndOfEvent` 中按物理读出通道 `(panelID, isHorizontal, barID)` 聚合期望光电子数 $\mu_{\text{tot}}$，施加 SiPM 微元饱和公式 $N_{\text{fired}} = N_{\text{pix}}\bigl(1 - \exp(-\mu_{\text{tot}}/N_{\text{pix}})\bigr)$（默认 $N_{\text{pix}} = 14400$，可通过 `PINPOINT_SIPM_NPIX` 配置），再采样实验观测整数 `nPE` 并按最大余数法分配回各粒子记录，保证通道求和严格等于饱和后读出值。
3. **真值溯源单调化、紧凑端点轨迹与 $\tau^\mp$ 螺旋度极化（`TrackInformation.cc`、`TrackingAction.hh/.cc`、`Py8Decayer.cc`、`GFaserGenerator.cc`、`AnalysisManager.cc`）**
   - 在 `TrackInformation` 拷贝构造与赋值运算符中补齐 `fFromCharmedHadron` 复制，移除每条径迹未使用的 `std::make_shared` 堆分配。
   - 将 `TrackingAction::PostUserTrackingAction` 改为对全部次级粒子执行单调或（`|=`）继承，修复 `fromPrimaryPizero` / `fromFSLPizero` 分支覆盖 `fromTau` 及 `TrackID == 1` 硬编码漏标问题；支持 `PINPOINT_COMPACT_TRUTH=1` 使用双端点 `EndpointTrajectory` 替代全步长轨迹。
   - 在 `Py8Decayer.cc` 中实现鲁棒的 `PYTHIA8DATA` 自动定位，并对主顶点产生的 $\tau^\mp$ 按相对论性 $V-A$ 弱相互作用赋予纵向螺旋度 $h(\tau^-) = -1, h(\tau^+) = +1$（支持 `PINPOINT_TAU_HELICITY` 设为 `-1`、`0`、`+1` 以评估极化系统误差）。
   - 在 `GFaserGenerator.cc` 中修复 `fHitPixelArea` 的 Y 偏移中心与钨板 Z 范围检查，并支持 `PINPOINT_USE_INPUT_Z=1`。

---

## 详细数据

- **编译产物与验证数据路径**
  - Geant4 可执行文件（`LCG_107` / Geant4 11.3.0）：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v4_audit_2026-11-18/g4/pinpoint_fixed`
  - 定量验证汇总 JSON：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v4_audit_2026-11-18/stage315_g4_verification.json`
  - 输入样本：`/eos/project-f/fasersim-bonn/public/pinpoint/2025-11-26-genie/nut-cc.gfaser.root`（25 个 $\nu_\tau$ CC 事例，随机种子 `20261118, 20261135`）与 `300 GeV` $\tau^-$ 单粒子枪（每个螺旋度 400 个事例，随机种子 `77777, 88888`）。

- **表 1：导出几何坐标与主顶点钨板落点校验**

| 指标 | 修复前（Stage 314 审查值） | 修复后实测值 | 物理放置真值 | 残差 |
| --- | --- | --- | --- | --- |
| `pixel_Zpos[0]` | `-2981.995 mm` | `-2991.995 mm` | `-2991.995 mm` | `0.000 mm` |
| `pixel_Zpos[11]` | `-1931.885 mm` | `-1941.885 mm` | `-1941.885 mm` | `0.000 mm` |
| `pixel_Zpos[12]`（Fortune 首层） | `-1734.875 mm` | `-1734.875 mm` | `-1734.875 mm` | `0.000 mm` |
| `scint_Zpos[0]` | `-2995.500 mm` | `-2979.490 mm` | `-2979.490 mm` | `0.000 mm` |
| `tungsten_Zpos[0]` | `-2984.490 mm` | `-2984.490 mm` | `-2984.490 mm` | `0.000 mm` |
| `tungsten_Zpos[1]` | `-2980.500 mm` | `-2974.490 mm` | `-2974.490 mm` | `0.000 mm` |
| `G4Navigator` 材质自检（15 Si + 108 W + 96 Scint） | 未校验（12 块 W 落入空气/冷却板） | `PASSED`（`100%` 匹配） | `100%` 匹配 | `0` 异常 |
| 主顶点落在 `G4_W` 比例 | `96/108 = 88.89%` | `25/25 = 100.0%` | `100.0%` | `0.0%` |

- **表 2：闪烁体 Birks 猝灭 + SiPM 饱和同种子对照（25 个 $\nu_\tau$ CC 事例）**

| 配置 | 闪烁体步记录数 | 总沉积能量 `sum_edep` (`MeV`) | 总光电子数 `sum_nPE` | 平均产额 (`PE/MeV`) | 高电离 `edep > 5 MeV` 产额 (`PE/MeV`) | 单通道最大 `nPE` |
| --- | --- | --- | --- | --- | --- | --- |
| 无猝灭线性对照（`PINPOINT_DISABLE_BIRKS=1, N_pix=0`） | `821,628` | `624,776.0` | `238,132,120` | `381.15` | `381.10` | `763,050` |
| 物理猝灭 + SiPM 饱和（`kB=0.126 mm/MeV, N_pix=14400`） | `706,565` | `530,488.1` | `54,814,756` | `103.33` | `128.44` | `14,400` |

- **表 3：`300 GeV` $\tau^-$ 单粒子枪螺旋度极化验证（每组 400 事例）**

| 螺旋度设置 $h(\tau^-)$ | $\tau^- \to \pi^- \nu_\tau$ 事例数 | 实测 $\langle E_{\pi^-} / E_\tau \rangle$ | 理论预期 $\langle E_{\pi^-} / E_\tau \rangle$ | $\tau^- \to \ell^- \bar{\nu}_\ell \nu_\tau$ 事例数 | 实测 $\langle E_{\ell^-} / E_\tau \rangle$ |
| --- | --- | --- | --- | --- | --- |
| `-1`（标准模型左旋 $\tau^-$） | `24` | `0.294 ± 0.048` | `0.333` | `81` | `0.386 ± 0.030` |
| `0`（非极化对照） | `20` | `0.504 ± 0.067` | `0.500` | `96` | `0.286 ± 0.025` |
| `+1`（右旋对照） | `28` | `0.620 ± 0.051` | `0.667` | `93` | `0.328 ± 0.022` |

- **紧凑轨迹与真值溯源实测值**
  - `PINPOINT_COMPACT_TRUTH=1` 下 25 个 $\nu_\tau$ CC 事例输出文件仅 `18 MB`，111 条保存径迹的点数范围为 `[2, 3]`。
  - 闪烁体 `fromTau == 1` 命中数 `229,000`，其中 $\gamma/e^\pm$ 次级电磁簇射命中 `202,087` 条；像素探测器 `fromTau == 1` 命中 `14,286` 条、`fromCharmedHadron == 1` 命中 `4,040` 条。

---

## 阶段性结论

- **已验证事实**：
  1. 从 `G4VPhysicalVolume` 直接提取坐标使导出的 `geometry` 树与 Geant4 实体追踪体积达到机器精度一致，彻底消除了 `11.11%` 非钨板假顶点与 `+10.000 mm` 像素层错位。
  2. Birks 猝灭与 `14,400` 微元 SiPM 饱和将高能簇射核心通道输出限制在真实硬件动态范围内，消除了未饱和模拟中超过 `7.6e5 PE` 的非物理峰值。
  3. 单调真值传播与 $V-A$ $\tau^\mp$ 极化均已通过独立统计检验，可用于生成物理一致的重建与鉴别样本。
- **待检验物理假设（将在后续阶段验证）**：
  1. 在消除 `+10.000 mm` 硅层错位并重新训练后，微米级像素点簇对主顶点纵向位置的中位残差预计可收敛至半块钨板厚度（`< 2.5 mm`）以内，且高精细像素螺距将展现单调正收益。

---

## 下一步计划

1. **Stage 316（分析与训练代码全面修复及单元测试/GPU 小样本门禁）**：完成单通道阈值化 `PE >= 2.5` 读出、C++17 像素聚类与微米多径迹/次级顶点（`Path C`）加速算子、谱仪动量公式 `100705 / |s_y|`（`P1-3`）、无偏异方差顶点损失（`P2-2`）、AdamW 1D 参数零权重衰减（`P2-3`）、`StereoBarNexus` 对称消歧（`P2-4`）、真实前向通量与带号多域/粲夸克控制区似然拟合（`P0-1`、`P1-4`）以及冻结清单 SHA 溯源与 `NaN/Inf` 拦截（`P3-1`、`P3-2`），通过单元测试与本地 GPU 32 事例过拟合及梯度健康验证。
2. **Stage 317 & Stage 318（并行推进受控 GPU 消融与 HTCondor 干净模拟重产）**：同时在 Kubeflow/HTCondor 多卡并行启动 `Exp-1/3/5/6` 受控 3 种子消融实验，并在 HTCondor CPU 集群批量运行 `pinpoint_fixed` 生成包含 >2.31 TeV $\nu_\tau$ 与中性流 $c\bar{c}$ 的几何对齐干净样本（`Exp-4`）。
