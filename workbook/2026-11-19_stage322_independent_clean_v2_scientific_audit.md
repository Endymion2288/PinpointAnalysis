# Stage 322 独立科学审计 · PINPOINT Clean v2 与 Stage 321

## 一、执行科学裁决

**Stage 321 的几何和主要数值可独立复现。纯钨靶物理、靶 pitch 归因和物理发现投影仍有 P0/P1 缺陷。** 审计对象为分析提交 `d65824c`、Geant4 提交 `6bc1744`，及实际 EOS 可执行、生产宏、冻结检查点和预测。

- **最严重风险是产生器核靶与输运材料不一致。** 逐事例回连 51,690 个名义 ROOT 父事例，证实 **3,825 个非钨相互作用被移入钨板**，占 Clean v2 的 6.375%。分别进入 train、Tune、Cal-A、Cut-B、Confirm 的数量为 2,037、535、248、394、611。核内级联和末态不会随顶点重定位转换为钨相互作用。另有 2,400 个计划 NC 粲由自由 p/n Pythia8 强制选样，写 W184 标签。两者直接影响纯钨 τ/粲的科学解释。 [父事例回连][parents]
- **几何闭合。** 108 块钨板中的 Pinpoint/Fortune 为 24/84，靶硅 12 面，IPT 3 面，闪烁 96 面。独立模型与新跑 ROOT、生产 JSON 的位置最大差约 $3\times10^{-12}$ mm。60,000 个计划点均在正确钨板。几何权重恢复了内缩钨体积比例。 [几何闭合][geo] · [钨板坐标表][plates]
- **模型层未发现标签值泄漏。** 五个冻结模型用哑标签、随机标签和随机附属真值的输出完全相同。删掉 `label` 后五个都报错，正式加载器还会读取 `targets.parquet`。实数据推断需要 observation-only 入口、读出与域校准。 [检查点干预][truth]
- **可复算的性能仅适用于本批 MC。** 主 PF 在 5,491 个真值切片事件的 BA4 为 62.33%，conditional BA3 为 77.74%，NLL 为 0.8876。D1 锁定的 1% 档实测 16/1,732 本底与 40/579 信号。锁定的 0.1% 档实测 6/1,732 本底，FPR 为 0.346%。四个曝光的 profile likelihood 数值复现，但外部中性强子和中微子应用了不同 veto，外部率与物理流量也未建立。 [分类][metrics] · [似然][likelihood]
- **硬件因果结论需收缩。** 1 mm 对照实际改变 IPT 聚类及 patch 尺度。6 层对照的主顶点 z 确有约 2 mm 的误差增加。Hybrid 可改善主顶点，微米 τ 位移敏感性尚未获得对象级证据。 [硬件反例和配对比较][counter]

## 二、完整证据基础问题登记表

严重性针对本表所述结论和适用域。VERIFIED 代表观察已独立确立，FALSIFIED 代表声明被反证，PARTIALLY VERIFIED 代表部分闭合，UNVERIFIED 代表证据缺口，NOT APPLICABLE 代表未进入活跃路径。

| ID | 严重性 | 类别 | 发现 | 证据 | 科学影响 | 所需修复 | 验证测试 |
|---|---|---|---|---|---|---|---|
| 322-01 | P0 | 核靶 | **FALSIFIED**。Clean v2 的钨顶点对应正确钨相互作用 | 名义 ROOT 抽取按过程/味筛选，保留母核和四动量后重定位。回连发现 C12 1,865、Al27 1,602、自由质子 150，其他 208。[生产代码](../src/analysis/flavour/v3/v6_clean_sim/produce_clean_sim.py#L317) · [记录转换](../src/analysis/flavour/s6/gfaser.py#L35) · [数据][parents] | 纯钨 MC 性能、τ/粲模板和发现物理解读失效。混合样本数值仍可复核 | 生成与输运在同一核靶。重产、重训与重校准 | 逐事例 GENIE 核靶和 Navigator 材料匹配，独立新 Confirm |
| 322-02 | P0 | 统计选择 | **VERIFIED**。外部中性强子形状用 p0 veto，ν 模板未用同一 veto。外部 μ 无率模板 | 500→420 用于外部形状，硬编码外部率并与 ν 相加。[生产似然路径](../src/analysis/flavour/v3/v6_clean_sim/run_stage321_clean_v2_campaign.py#L1376) · [复算][likelihood] | 预测对应的物理选择不成立。真实发现结论失效 | 所有过程同切，加入可约束的外部 μ 与中性强子率，重建模板 | 逐过程事件选择闭合、控制区率、形状及零箱检验 |
| 322-03 | P1 | 像素 pitch | **FALSIFIED**。靶 pitch 对照保持了 IPT 原始坐标，但 IPT 聚类与 patch 伸缩跟靶 pitch 一起改变 | 全平面共用 $1.45q$ 半径，0.208 mm IPT 双击由 2 簇变 1 簇。1 mm 时 7/32 实例保留的 IPT token 数改变。[代码](../src/analysis/flavour/v3/v2_pinpointformer/pixel_tokenizer.py#L276) · [反例][counter] | 撤回单因素 pitch 归因及 Stage 319 下跌唯一归因。保留当前配置的 BA4 点估计 | 分层聚类、patch 和 tracklet 单位。重训匹配对照，硬件响应另重产 | target×IPT 2×2 与固定 IPT token 逐元素比较 |
| 322-04 | P1 | 稀有本底 | **VERIFIED**。外部结构零及缺少外部有限 MC，Gaussian BB-lite 与 Poisson auxiliary toys 不同，单乘子缩放总预测 | 最稀 SR 有效 MC 为 1。外部尾箱零数以精确上界表达。[权重及 BB-lite](../src/analysis/flavour/v3/v5_confirm/flux_significance.py#L300) · [toys](../src/analysis/flavour/v3/v5_confirm/flux_significance.py#L715) · [独立对照][likelihood] | 不能为真实尾部声明覆盖与发现率 | 分过程有限 MC、合并或补样、同一 auxiliary 生成和拟合，重新评估 | 多 nuisance 真值点的覆盖，$N_{\rm eff}=0–100$ 情景，记录自由与零假设优化状态 |
| 322-05 | P1 | 产生器 | **VERIFIED**。NC 粲用自由 p/n、$Q^2_{\min}=4$ GeV²、强制选 charm 配额并写 W184。CC 与 NC 来源不一致，$\nu_\mu$ charm 超采样 | [弱过程启用与重标](../src/analysis/flavour/v3/v6_clean_sim/produce_clean_sim.py#L461) · [类别抽样](../src/analysis/flavour/v3/v6_clean_sim/produce_clean_sim.py#L624) | 4.5% NC 粲先验与尾部形状缺少核靶、截面和过程中介支持；模型可能利用样本构造 | 统一流量和核靶，重产、重新加权并做留一源实验。偏差出现后重训 | 按生成器、ν/反ν、$E,y,Q^2,W$、D/Ds/Λc 与候选尾部对齐 |
| 322-06 | P1 | 投影输入 | **UNVERIFIED**。四时代均 60 事例/fb、固定 0.3% τ、2% 外部与 50% 外率约束。缺少各时代束线、截面、质量与相关协方差 | [投影入口](../src/analysis/flavour/v3/v6_clean_sim/run_stage321_clean_v2_campaign.py#L1417) | 只成立于指定参数情景 | 用真实曝光、谱形、位置、活时间与选择迁移矩阵重新评估 | 独立产额逐 process、能量、角度、位置闭合 |
| 322-07 | P1 | 数字化 | **PARTIALLY VERIFIED**。一次 PE 饱和和索引正确。真实 tracker、硅电荷及门槛未闭合。0.1 mm 网格加 17 μm 高斯，独立近似下 Y RMS 33.5 μm | `PixelSD.cc:63,79,152`、`FaserSD.cc:32–50`。[tracker 读出](../src/analysis/flavour/readout.py#L113) · [数值][counter] | 模拟像素、位移、谱仪接受度与 real data 成绩不确定 | 校准前向响应和真实条位置。重产响应，按差异重训 | 真实 μ、阈值扫描、双粒子、弯曲残差 pull、噪声及死区 |
| 322-08 | P1 | 操作点 | **FALSIFIED**。将名义 0.1% 当实际背景抑制，或把不同域 $S_\tau$ 直接视作同尺度 | D1 0.346%，D3 0.496%，主 PF 跨域 Pinpoint 1% 排序 D2 为 0 FP。[阈值代码](../src/analysis/flavour/v3/v6_clean_sim/run_stage321_clean_v2_campaign.py#L862) · [计数][metrics] | 尾部性能与实验筛选保障尚未建立 | 固定实际可观测域，物理先验校准，重新评估 | Cal/Cut/Confirm 联合误差、分域可靠性与真实背景 mixture |
| 322-09 | P1 | 证据溯源 | **PARTIALLY VERIFIED**。v7 固定权重 SHA。Confirm 在 Cal/Cut 参数写出前已读入。配置、输入、缓存无完整内容绑定。GENIE 系统样本重新编号 | [读入顺序](../src/analysis/flavour/v3/v6_clean_sim/run_stage321_clean_v2_campaign.py#L944) · [写入顺序](../src/analysis/flavour/v3/v6_clean_sim/run_stage321_clean_v2_campaign.py#L1084) · [系统父源重命名](../src/analysis/flavour/v3/v6_clean_sim/produce_external_and_gen_sys.py#L248) | 可重复计算。软件执行顺序无法证明组织盲化。实际系统父源重用数未核实 | 单独冻结器、输入/可执行/配置指纹、原父 ID。新 Confirm | 只读独立重放、跨产物 source join 与阈值时间线 |
| 322-10 | P1 | 物理解释 | **UNVERIFIED**。Hybrid 具备 PV、SV 和位移 proxy，缺少 secondary truth-match、效率、purity 或残差 | [微顶点输出](../src/analysis/flavour/v3/common/fast_kernels.py#L19) · [反事实算法](../src/analysis/flavour/v3/v2_pinpointformer/pixel_tokenizer.py#L485) · [比较][counter] | 不能主张微米 τ 位移重建或与 charm 因果分离 | 对象级重建和控制衰变反事实。先评估，后按需重训 | 配对固定 $E,y,z$、衰变型与活动，平移衰变位置并量化真匹配 |
| 322-11 | P2 | 部署 | **FALSIFIED**。严格移除全部标签会运行失败。真实与哑标签值无差 | [B1++](../src/analysis/flavour/v3/v1_b1pp/models.py#L348) · [PF](../src/analysis/flavour/v3/v2_pinpointformer/models.py#L242) · [实验][truth] | API 不能直接从无标签源运行。尚无分类数值泄漏证据 | 从观测张量取 batch size。无需因字段变化重训 | 只有 observation 的真实或 mock readout 文件，重放 frozen 模型 |
| 322-12 | P2 | SiPM 时间 | **VERIFIED**。时间按 MC track 分组，零 PE 记录可决定聚合时间。活跃 PF/B1++ 无时间 | `ScintSD.cc:242–264,310–314`。合并 0 PE@1 ns 和 40 PE@5 ns 得 40 PE@1 ns。[条聚合](../src/analysis/flavour/readout.py#L71) · [反例][counter] | 时间若用于 veto/TOF 将有不真实信息。对当前模型分数不适用 | 模拟脉冲、光电子到达、时间门槛抖动和全局时钟，验证后再用 | 通道时间在相同物理脉冲的 truth 分组变化下不变 |
| 322-13 | P2 | 几何质量 | **PARTIALLY VERIFIED**。权重恢复内缩钨体积。采样只覆盖 97.545% W，真实装配和通量分布未建立 | [几何表][plates] · [内缩权重](../src/analysis/flavour/v3/v6_clean_sim/produce_clean_sim.py#L184) | 不能直接当完整靶和真实 fiducial acceptance | 显式表面层、各材料有效质量和束线空间密度 | 全体积与样本边缘 closure |
| 322-14 | P2 | 形状与数值 | **VERIFIED**。系统形状是粗背景合并，0.001 平滑和 ±0.35 裁剪。率的 clip 未计入 Jacobian。内层对 $\nu=0,n>0$ 跳过计数 | [形状](../src/analysis/flavour/v3/v6_clean_sim/run_stage321_clean_v2_campaign.py#L1307) · [内层](../src/analysis/flavour/v3/v5_confirm/flux_significance.py#L343) · [Jacobian](../src/analysis/flavour/v3/v5_confirm/flux_significance.py#L551) | 标称小 pull 下可复算。极端统计模型应纠正 | 逐过程 shape 与数值约束、零箱有效支持 | 有限差分、边界案例与独立 optimizer |
| 322-15 | P2 | 架构 | **PARTIALLY VERIFIED**。有非恒定学习。Dense GN2 在冻结配置关闭，对最新分数贡献 **NOT APPLICABLE**。NOMAD/Hybrid 的增量分类证据弱 | [训练配置](../src/analysis/flavour/v3/v6_clean_sim/run_stage321_clean_v2_campaign.py#L279) · [权重干预][truth] · [模型对照][counter] | 不能用已训轮数或 attention 熵断言因果分支 | 同 budget、matched seeds、物理与轻量 ML 基线 | 分支梯度、32-event 拟合、特征损坏、各域重评估 |
| 322-16 | P3 | 配置 | **VERIFIED**。中间像素数源于初始 `nFortuneLayers-1`，构造函数局部 overlap flag 遮蔽成员。当前宏恰好匹配 | `DetectorConstruction.hh:225–227`、`DetectorConstruction.cc:425–489` | 新 Fortune 宏可能改变实际布置和记录契约 | 显式配置依赖和 placement-derived metadata | 至少两个不同 Fortune 数的 Navigator 和导出闭合 |

## 三、历史缺陷回归审计

### A. 探测器、材料与数值几何

世界空气盒 10×10×20 m。束流为右手系 +z，世界长度 mm。整体栈 $y=-12$ mm，入口 $z=-3000$ mm，栈尾 $-1799.88$ mm。先放六个 PinpointBlock，每个 30.01 mm，顺序为 2 mm Al 墙、6 mm 空气、10 μm Si、5 mm Al 冷却、5 mm W、5 mm 竖条、5 mm W、2 mm Al 墙。七个 FortuneBlock 每个 120 mm，每块六组 W/V/W/H。在前六个 FortuneBlock 后各插一个 PinpointBlock。钨板 $430\times430\times5$ mm³，中心 $(82,-7)$ mm，横向 $[-133,297]\times[-222,208]$ mm²。Pinpoint 索引为 **0–11、24–25、38–39、52–53、66–67、80–81、94–95**。依据 `DetectorConstruction.cc:232–399,425–489` 与生产宏。 [Geant4 宏](../../Pinpoint_G4/Pinpoint/macros/geom.mac) · [108 块位置与 copy 表][plates]

15 个硅面中心 $(0,-12)$ mm，实体 $266\times196$ mm²，像元 $20.8\times22.8$ μm。12,788×8,596 的整数网格在正 X/Y 端分别截去约 9.6/11.2 μm。单 binary 像元位置均匀时，量化 RMS 为 6.00 和 6.58 μm；有效空间分辨率还要计入入射角、电荷共享、噪声、门槛、对准与重建。IPT 三面 $z=(-1734.875,-1704.865,-1674.855)$ mm，栈后 50 mm 空气间隙再加半模块。IPT 配置复用靶 Si LV。三站谱仪另为 $z=(47.4,1237.4,2427.4)$ mm、250×250 mm² 单敏感面模型。几何 JSON 的 `silicon_thickness` 单位 μm。`PixelSD.cc:73–112` touchable local→ROOT IDs→`readout.py:20–36`→Parquet ID/xyz→模型的世界坐标/相对 z。独立解码的 47,144 个实际序列化像素最大误差 0.0001172 mm，在 float32 取整范围。48 个新跑 50 GeV ±μ 样本的 primary p0 位置均小于半 pixel 偏差。 [活体与 Parquet 闭合][geo]

96 个闪烁面含 54 竖和 42 横。每面 40 条，长 420、宽 10、厚 5 mm，pitch 10.5 mm，空气间隙 0.5 mm。竖条沿 Y 读 X，横条沿 X 读 Y。条面中心 $(77,-12)$ mm，W 多出正 X/Y 各 10 mm。新跑 ROOT 的条方向零不一致。`DetectorConstruction.cc:80–96,156–226` 材料为 G4 vinyl toluene，设 Birks $k_B=0.126$ mm/MeV。模型有空气、Al 墙/冷却、FR4 定义、Sm2Co17 壳。FR4 定义无法替代真实支持结构和电子学材料。现有宏中的 `fVetoNuPosition` 是靶入口，独立 VetoNu 硬件路径没有在此次模拟闭合。实际建造 CAD、物料表、支持架与 dead maps 未取得逐参数验证。 [布置源码](../../Pinpoint_G4/Pinpoint/src/DetectorConstruction.cc#L425) · [活体与图表][geo]

磁孔半径 100 mm，中心 $y=-10$ mm。Sm2Co17 外壳从 100.01 到 215 mm、密度 8.4 g/cm³。三个硬边区间 $[-1565.3,-65.3]$、$[137.4,1137.4]$、$[1337.4,2337.4]$ mm。$+B_x=0.57$ T，场积分 0.855、0.570、0.570 T·m。`DetectorConstruction.cc:509–581` 与 `MagneticField.hh:21–46` 给无 fringe、仅孔内场。场孔尺寸、shell 与站面的可见域不同。磁场图和非零 fringe 无现场测量支持。 [独立正负探针][geo]

以 19.3 g/cm³ 计，裸 W 1,927.03 kg，Pinpoint 428.23 kg，Fortune 1,498.80 kg。建模 Al 约 71.3 kg、条约 83 kg，栈约 2.08 t，谱仪和其他支持另计。内部 FORTVNE 材料展示原型和仿真取舍，尚无已验证的满堆安装约束。`large_scintillator` 是全 W XY 和 $z\le z_{\min}+0.85(z_{\max}-z_{\min})$ 的真顶点条件，Confirm 9,037 个，会与全样本 10,000 个口径不同。三层抽样概率 0.40/0.15/0.45，按内缩体积分数反比校正。物理 W 面内所有 60,000 个点最远 z 偏 2.44996 mm，模块标记零差。采样覆盖 97.545% 实体 W，几何有效样本量 30,478，Pinpoint 加权 22.1561% 与 $24/108=22.2222\%$ 相符。该权重不能修正核靶误置、能谱/角谱和检测效率。 [几何抽样](../src/analysis/flavour/v3/v6_clean_sim/produce_clean_sim.py#L184) · [独立测量][geo]

### B. 真值、可观测量与封存

| 量 | 归属及其实际依赖 |
|---|---|
| Pixel row/col/layer，条 channel 与 PE | 实验可观测的模拟替身。ID→几何查表→像素簇、shape、patch、bars。PE→2.5 PE 门槛、逆饱和、MIP 比例，同一 PE 给两个 token 分量 |
| Tracker station、xy 与 σ | 由观测可重建，但现有模拟先取 G4 真穿越点，再合并/效率/加噪。无逐 trackID、PDG、truth momentum forward |
| `c_obs`、`cal_start`、domain | 可观测量重建。计数、范围、几何中心、PE、p0 veto、粗 shower apex。实际名义域逐事例复算，无 10,000 个 Confirm 的记录差；粗 pitch/减层模型内部域可变化 |
| `avr_pv_mm`、tracklet/SV summary、NOMAD | 可观测量重建的候选。簇几何、bar stereo、leading station triple 进入模型，物理正确性待独立对象闭合 |
| 各类 label；真 PV、τ 分支和位移、charm、可见运动学 | MC 监督或后验诊断。训练 sidecar 构成目标。PF 的 7 类 logits 按定义合至 4 类。dataloader 读 label，forward 取其长度 |
| GN2 dense origins/associations | 训练目标至 `compute_losses`。可选前向使用预测的辅助 logits 与 summary，Stage 321 配置关闭 dense 和 cluster sidecar |
| KEY/family/split/seed/source/weight | 模拟元数据，主要用于对齐、审计/生产、bootstrap 与物理评价。打乱对照从 item index 生成随机平面旋转，同一事例如被重新分配数字索引会改变扰动 |
| 真 W 板和像素 XY、true fiducial、τ flight length | 合法 MC 诊断或模板条件。直接用于数据选择会没有实验定义，应补 fiducial migration 和观测切选 |
| Cal-A 和 Cut-B 真标签 | 合法离线监督。Cal-A 按域 balanced NLL，Cut-B 选 $\lambda$ 和阈值。Confirm 标签仅数值评价，软件同时接触 Confirm 的时序不能证明盲化 |

五个 seed 17 检查点核实 SHA 并推断 32 个真实 Parquet 事例。哑标签、随机标签、随机顶点、权重、关联与批次逆序的输出差为 0。单独推断首事件与批次最大差 $1.91\times10^{-6}$，含顶点分量。严格去掉 `label` 的真实失败为 `KeyError`。B1++ 另从 observation 及 KEY 单独读入全套分割和系统样本，只传哑标签。旧模型缓存 SHA/绝对路径存在，输入与预处理 SHA 未绑定，结果文件没有证明实际取用的域校准时间点。 [检查点干预][truth] · [冻结清单](../roadmap/v3/v7_freeze_manifest.json)

60,000 条计划记录无 family 或 event key 重复。80 个名义 ROOT 文件逐事例 join 51,690 个父记录，输入能量差 0，过程/核靶/入射和出射 PDG/能量与 DIS 指纹无精确重复。321/326 源 tag 跨多个 split，92.56% 事件的 source tag 唯一对应单一类别。四类 $E_\nu$ 中位 577.5、626.1、686.1、513.2 GeV，90% 分位 2113.9、2360.5、1584.5、1596.1 GeV。$\nu_\mu$ 粲和 τ 高能优先取样，NC 粲 2,400 全来自单独 Pythia8。同源分配符合 IID 复算，无法测外部新产生器泛化。生成器种子和 ROOT 行级谱系中，名义输入的重复已检查，GFaser 跨库完整四动量指纹和系统样本与 nominal 共父的数量未测量。 [数据与指纹][parents] · [源分布][counter]

### C. 响应与独立公式

- **闪烁与饱和。** 生产配置禁用二次饱和。一次饱和 $N(1-e^{-n/N})$，$N=14400$，若错误再饱和上限约 9102.54 PE。PE-only 移除了 `edep_proxy_MeV`，98% inverse cap 后无法分辨能量排序。分片 080 有 539,170 条，最大 14,400 PE，24,114 条超过数学上的二次饱和上限，2,904 条达到 14,400。分片 125 同样达到 14,400。分片 080 和 125 分别有 3,098、3,141 个零 PE 条仍带非负时间。光学快模型用 $10^4$ photons/MeV、3.8 m 衰减、0.10 transport、0.40 PDE、$c/1.58$ 和 2.1 ns 衰变。Birks 使用 `G4EmSaturation`。57 条活体 0 PE track 仍有正时间，零光电子的反例影响最终条时间。未建完整波形、noise、门槛涨落、双端读出、SiPM 恢复/串扰和真实 calibration。 [G4 观测][geo] · [反例][counter]
- **硅和磁谱仪响应。** `PixelSD.cc` 的步能沉积直接落本地像元，薄硅电荷分享/真实门槛、死图像和 DAQ 未闭合。FASER tracker 从真 pre-step crossing 构建三站拟测量，合并在 0.1 mm 格后加 $(0.58,0.017)$ mm Gaussian 和独立 99% 效率。独立均匀网格近似 Y 为 33.5 μm，单 triplet sagitta 41.0 μm，1 TeV 的一阶动量误差贡献约 41%。多重散射与材料接受度会进一步改变统计。 [流水线](../src/analysis/flavour/readout.py#L113) · [误差计算][counter]
- **磁输运。** 设 $\kappa=0.000299792458\,q B_x/p[\mathrm{GeV}]$ mm⁻¹。用 $y(z)=y_0+y'_0(z-z_0)+\kappa\int_{\rm bore}(z-u)\,du$ 独立积分，两短磁体的二阶差给 $s_y=-590000\kappa$，进第一站斜率 $(y_1-y_0)/1190+(60/59)s_y/1190$。长磁体最前侧的位移修正 $-\kappa L(z-z_c)$，孔内 $+\kappa(z_{\mathrm{exit}}-z)^2/2$，与代码闭合。48 个事例得 47 个三站 primary triplets。中心两探针组 raw crossing 动量相对误差最大 3.49%，偏轴组有一个 77.7% 离群事例及一个未得三站。中心理想闭合与有损输运、散射、匹配和真实分辨的场景应分别评估。 [spectrometer](../src/analysis/flavour/v3/v1_b1pp/spectrometer.py#L15) · [活体公式复核][geo]
- **产生器与 G4 物理。** FTFP_BERT+PY8DK 输运指定的 GFaser status 1 四动量。名义样本含 CC 三 flavour、NC 轻味和 Pythia8 粲。搬移顶点仍保留核介质信息。τ 用 ±1 纵向极化或 0 对照，Py8Decayer 取离散 helicity。相互作用依赖反中微子、核 PDF、阈值、τ full spin density 与分支，未测 $L$ 对 $c\tau\beta\gamma$、预期 branching 和 kinematics。EM/hadronic shower、MCS 由 G4 建模实际执行，特定物理列表的模型系统差、不同材料的束测验证与粒子识别性能未单独建立。 [生产实现](../src/analysis/flavour/v3/v6_clean_sim/produce_clean_sim.py#L415) · [GENIE 方法](https://arxiv.org/abs/0905.2517) · [Pythia8 方法](https://arxiv.org/abs/2203.11601) · [父档][parents]


## 四、最新模型与实验审计

### 模型、训练和数值

Stage 321 的 9 组×3 seeds 在 32k train/8k Tune 上训练。PF 的 nominal budget 为 20 passes、batch 128、BF16 训练、FP32 评价。主 Hybrid seed 17 的 manifest 为 8,923,677 参数、best pass 16。相同 nominal budget 允许不同模型提前在 Tune 选取不同 checkpoint。B1++ 使用 X/Y 多尺度稀疏投影、Z 顺序和谱仪候选。PF 组合像素簇、局部 patch、stereo bars、谱仪和 Perceiver；multitask 有 7 类事件、charm、可见运动学、PV 头。NOMAD 走 PF V4 路径，额外加域门控 10 维特征。GN2 dense 在这些模型关闭，历史多任务成绩不能充当最新 GN2 证据。 [活跃配置](../src/analysis/flavour/v3/v6_clean_sim/run_stage321_clean_v2_campaign.py#L279)

张量的 padded mask、稀疏索引和样本独立性经真实批次反事实。空事例的最终 ensemble 概率 0.25，score −25。这是分析退化约定，对实际触发缺失需另定义。最多 1,024 个簇与 64 个 patch，排序会优先保留高活动区域，per-plane summary 在 cap 之前生成。未按类完整测量丢失的离散 tracklet 数。用独立 NumPy LayerNorm、L2、每 head 温度和有效长度构建非均匀 attention 参考，覆盖空 key 和部分 query mask，最大差 $5.93\times10^{-8}$。另写 balanced NLL 与正则求校准有限差分，目标值差 $2.22\times10^{-16}$，梯度最大差 $4.31\times10^{-11}$，24 种重新拟合排列对 softmax 的最大差 $1.27\times10^{-8}$。后两项支持数值改进。实际非恒定分类、显著高于随机的 BA4 和非零分数方差支持事例分类学习。在 32 个真实事例对主 Hybrid 分别清空像素、条和 tracker，四类概率平均总变差为 0.125、0.314、0.165，对应 7、18、9 个 argmax 改变。五个模型即使清空所有观测仍输出有限值。清空测试同时改变计数和域，在此仅衡量预测依赖。端到端逐分支梯度、独立小样本记忆和极端 tie 时的全排列结果仍未测。 [数值算例][counter] · [模型得分][metrics]

| 对照 | 独立观测及可解释范围 |
|---|---|
| B1++ / 主 PF / 双族 | Stage 321 B1++ 单族 Pinpoint BA4 60.78%，主 PF 62.33%。重跑三 B1++ 权重后双族 Pinpoint BA4 62.42%。报告的双族−PF +0.09 pp 区间 [−0.15,+0.38]，尚未建立额外家族的分类优势。B1++ 保留为轻量 baseline。 |
| Hybrid vs vanilla | 独立重算 PV 3D 中位 5.903 vs 6.366 mm，X/Y/Z 的绝对误差中位改善约 0.144/0.176/0.251 mm。独立 1,000 配对重采样的 z 差区间 [−0.417,−0.085] mm。$p_\tau$ 1% 排序测试的报告增益 +1.55 pp，区间 [−0.36,+3.09]。 |
| Hybrid vs shuffled | 打乱只旋转供微顶点分支处理的各面簇，保留平面计数/质心/径向展宽。主 PF 仍接收正常几何，单簇面不动。位置变化也影响 tracklet 建成率。PV 3D 5.903 vs 5.971 mm，z 差 CI [−0.261,+0.065] mm，τ 低 FPR 区间跨零。 |
| Hybrid+NOMAD vs Hybrid | Pinpoint $p_\tau$ 同秩观察 90 vs 88 TP（共同 41/4,132 FP）。Front 的报告增量 +0.72 pp，CI [−2.34,+2.43]。新旧数据跨 geometry、digitizer、训练和光学的差无法单独归因。 |
| 12T vs 6T | 6T 屏蔽 target 1、3、5、7、9、11，保留其他 6T 和 3 IPT，模拟材料相同。PV z 4.964→6.920 mm，独立配对中位差 95% CI [1.694,2.195] mm。对现有模型支持读出层数影响。 |
| 20.8 μm vs 1 mm | 62.533% vs 62.495% BA4，独立 Δ 95% CI [−0.754,+0.815] pp，覆盖零。IPT 反例使单因素硬件归因失效。无预定义等效界限。 |


用可观测量进行小规模捷径对照。固定均匀选取 6 个训练 shard 的 2,400 例和 4 个 Confirm shard 的 1,600 例，训练 L2 multinomial logistic regression。仅用模态有无、三站是否齐全、像素和条计数，BA4 为 **39.11%**、τ AUC 为 0.578。用全部 16 个 containment summaries 为 **46.22%**、τ AUC 0.596。简单占用和形态量已携带可分类信息。该对照无 raw hits、对象关联或生成器字段，也未将源分区或按谱重匹配。 [观测基线、分片与配置][counter]

历史单测的证据范围也需区分。Stage 321 的 108 板测试使用实现几何目录的输入生成 fixture，无法检查原来的核靶身份。pitch 测试只验证三条 IPT 坐标，无法发现同面两个簇合并。磁光学单测使用给定硬边区间，独立 ROOT 试验补了真实输运与偏轴反例。truth isolation 的 AST 检查主要约束 sidecar import 与路径权限，有标签文件的 fixture 无法暴露去标签前向报错。 [Stage 321 测试](../tests/test_stage321_physics_fixes.py) · [标签隔离测试](../tests/test_v3_truth_isolation.py)

### 物理重建

当前 microvertex 实现从簇和候选 tracklet 形成 PV、2-vertex χ²、最大 impact parameter、kink 和位移 proxy。毫米尺度 PV 改善和 $L_{3D}$ 的分类相关不能给出独立 SV 位置、位移分辨或 τ/charm 条件性能。单 5 mm W 为约 1.43 $X_0$。10 GeV 带电粒子 Highland $\theta_0\approx1.65$ mrad，毫米到厘米外推可形成与微米目标同尺度误差，重建应提供误差传播和归因到正确对象的质量窗。未测 tracklet purity、SV misassociation、$d_0$ pull 或物理 kink efficiency。 [微顶点实现](../src/analysis/flavour/v3/common/fast_kernels.py#L19)

训练后的 NOMAD 代理来自正交条外积和 leading track 候选，交叉时可造 ghost pair，双簇独立响应不建立真实 3D 能量分配。实际缺失 $p_T$、领先轻子与强子 recoil 匹配、尾部 charge sign 和真实磁场图均需独立闭合。 [NOMAD 实现](../src/analysis/flavour/v3/v4_tau/nomad_kinematics.py#L87)

## 五、物理与探测器设计解释

| Stage 321 假设 | 支持与反证 | 最终判断 |
|---|---|---|
| 1. 几何、数字化、统计错误全部消除 | 352/352，活体位置、一次 PE 和 nominal 公式通过。材料误置、IPT、veto 和有限 MC 反例 | **FALSIFIED**。保留通过闭合的局部修复 |
| 2. 推断无 MC 真值泄漏 | 实际检查点对标签值和随机附属字段不敏感。标签形状依赖、可操作阈值与实数据尚需单独验证 | **PARTIALLY VERIFIED**。冻结五模型没有观察到标签值进入计算 |
| 3. 几何和质量采样正确 | 24/84、内缩权重、坐标和计划位置正确。非钨生成器核靶与分层全支持的声明被反证 | **PARTIALLY VERIFIED**。几何权重闭合限于以该内缩密度采样的体积 |
| 4. 至 1 mm 无可区分 BA4 损失 | 独立 Δ 样本区间含零，IPT 聚类同时变化 | **VERIFIED** 对当前处理组合的差异结果。**FALSIFIED** 对靶单因素隔离 |
| 5. 再减硅层恶化纵向顶点 | 12/6 指定读出对照 4.964→6.920 mm，独立配对 CI 不跨零 | **VERIFIED** 对指定读出和 Clean v2。需另测实际硬件拆除和重排 |
| 6. Hybrid 改善 PV，未确立微米 τ 位移 | 横向 PV 降低至 0.63/0.61 mm，z 4.71 mm。与 shuffled 的 τ CI 跨零，且无 SV truth-match | **VERIFIED** 主顶点误差改进。**UNVERIFIED** 真实 τ 位移敏感性 |
| 7. NOMAD 可能有用，增量不确定 | 两磁体及长磁体理想公式闭合，但偏轴有离群。NOMAD 比 Hybrid Pinpoint 多两 TP | **PARTIALLY VERIFIED**。增量优势、真实重建量及可迁移性未建立 |
| 8. p0 veto 有接受度和背散射限制 | 像素投影覆盖前 W 约 28.20%。350/350 内区 μ veto、外区 0/150，420/500 中性强子未触发。名义 τ Front 损失 217/972=22.33%，全靶加权 τ 为 1.737%，似然 fid τ 为 2.075% | **VERIFIED** 对这些 MC。四类合并的 25.22%/约 1.77% 同 Stage 321 定义。真实接受度待测 |
| 9. 2t 似然低于发现门槛 | 数值闭合 0.300–1.480σ，Gaussian-lite 0.228–0.473σ。典型发现标准 5σ，3σ 常为 evidence | **VERIFIED** 参数化计算，不足以验证真实发现灵敏度 |
| 10. MC 与外部输入限制敏感性 | SR 最小有效本底 1，外部 500。仅按统一 veto 会生成新的空箱 | **VERIFIED**。模板、有限 MC、外部率和先验均是底层限制 |

$L_{3D}\ge10$ mm 的分类优势在 1 mm 和打乱模型上保留。只用 $E_\nu>300$ GeV 切选没有匹配 $\tau$ energy、$y$、分支、真实首板与像素活动。结果与宏观迟发簇及单 MIP 穿透解释相容，不能量化其贡献。以三维飞行距离 10 mm 划分板内/跨板也不成立，还取决于 $\Delta z$ 和与实体边界的交点。 [对照数据][counter] · [历史观测](2026-11-19_stage321_physics_remediation_and_clean_v2_retraining.md#33-能量控制-tau-飞行距离诊断ood-量化机械对准与-probe-a)

## 六、统计有效性与敏感性

### 分类、校准与锁定

$BA_4=\tfrac14\sum_cP(\hat c=c\mid c)$。加权版按真类内做几何加权。conditional BA3 先以真标签排除 τ，再从非 τ 三类列 argmax，属于后验诊断。NLL 为 $-\langle\ln p_y\rangle$。各域在相同模拟 Cal-A 均衡任务上分别拟合的 $p_\tau$ 可作诊断比较。真实跨域校准依赖过程混合和 prior，$S_\tau=\ln[p_\tau/(1-p_\tau+\lambda c)]$ 的比较还受域 $\lambda$ 与 charm 输出影响。  [全混淆与指标][metrics]

| 主 PF 观测域 | Cut-B 名义 1% 的 Confirm TP / FP | Cut-B 名义 0.1% 的 Confirm TP / FP |
|---|---|---|
| D1 pixel | 40/579、16/1732 = 0.924% | 23/579、6/1732 = 0.346% |
| D2 scint | 109/1439、58/4151 = 1.397% | 31/1439、7/4151 = 0.169% |
| D3 partial | 19/482、15/1614 = 0.929% | 10/482、8/1614 = 0.496% |

D1 名义 0.1% 阈值的固定阈值 Clopper–Pearson 95% FPR 区间为 [0.127%, 0.752%]，只计 Confirm 抽样。在 D1 1% 档，NC 粲为 3/92，CC 粲 1/130，轻本底 12/1,510。就这批四类等量样本而言，轻本底占 16 个通过事例中的 12 个，物理组成需要过程权重。主 PF 的全 10,000 事例几何加权 BA4 为 54.11%，似然真基准的 9,037 事例为 55.51%。

主 PF 的 Pinpoint 切片 $p_\tau$ 为 88/1359=6.48%，$S_\tau$ 为 79/1359=5.81%，1% 的背景支持仅 41/4,132。跨域单 $S_\tau$ 阈值使 D2 零 FP，D1 2.329% FPR，D3 0.399% FPR。D1 约 17 个 FP 决定 1% 的 Confirm 排位，低 0.1% 只约 1–2 个。Stage 321 同时给“7.25%”诊断秩与“40/579=6.91%”锁定值，分别属于两个估计量。Cut-B 按 quantile 锁定，事后 ROC 用整组并列的保守秩。绝对 FPR 与目标值差异须在正文报告。

Family bootstrap 的单位与独立 family 一致，但每次在同一 Confirm 重新取阈值，只估计模型固定后的配对 ROC 差。没有将 Cal-A、Cut-B、训练 seed、产生器与多个选型的变动纳入置信区间。在外部样本中 0/150 和 0/500 的 95% 单侧上界分别约 1.98% 和 0.60%。后者乘假设的 3,600 个外部事件仍可允许约 22 个存活。 [Bootstrap 实现](../src/analysis/flavour/v3/v6_clean_sim/evaluate_stage317.py#L153) · [分域计数][metrics]

### 独立 27-bin 计算

读取实际 PF parquet，重推 B1++ 三个 seed。用 Cal-A 参数独立合成 log-prob 和 charm proxy，用 Cut-B 分域 $[50,85,95,98.5,99.5,99.85]$ 分位建立 SR，再用 low-S 两箱的 85%/96% charm 分位移入 6 CR。CR 边的独立复算最大差为 0。独立 nominal domain 重算 10,000 Confirm 及独立外部/系统分割全部一致。按每个事件唯一分箱，SR 和 CR 计数互斥；仍共享 generator 权重、MC 文件及 nuisance。 [独立复算][likelihood]

stat-only 为 $q_{0,A}=2\sum_i[(s_i+b_i)\ln(1+s_i/b_i)-s_i]$。分过程 rates 与独立提取的有符号模板构成 $H_0$，Asimov 的 $\mu=1$ 达最优，另写 Poisson deviance 和 Gaussian constraints，再用 SciPy 数值导数优化。Gaussian-lite 的相对方差 $\delta^2=\sum w^2/(\sum w)^2$，优化的 bin multiplier 满足 $\gamma^2+(\delta^2\nu-1)\gamma-\delta^2n=0$。用有效 Poisson auxiliary 比较时 $m=1/\delta^2$、$\gamma=(n+m)/(\nu+m)$。Poisson 列沿用单 bin 近似，只是辅助约束反事实。

| 参数情景 | τ 产额 | stat-only Z | 报告形状与率 nuisance | Gaussian-lite | Poisson auxiliary 对照 |
|---|---:|---:|---:|---:|---:|
| Run 3 250 fb⁻¹ | 45 | 0.5352 | 0.2998 | 0.2284 | 0.2276 |
| Run 4 680 fb⁻¹ | 122.4 | 0.8826 | 0.4574 | 0.3065 | 0.3043 |
| HL-LHC 3,000 fb⁻¹ | 540 | 1.8539 | 0.8681 | 0.4192 | 0.4088 |
| FPF 10,000 fb⁻¹ | 1,800 | 3.3847 | 1.4798 | 0.4734 | 0.4541 |

HL-LHC $b=183007.83$，包括外部中性强子 3,600，独立 $q_0=0.753676$，Gaussian-lite $q_0=0.175699$，与封存标量一致到展示精度。数值导数下独立 FPF Gaussian-lite 优化的收敛标志为 false，目标值与仓库解析梯度一致到展示精度。其余表列的独立优化标志为 true。 [拟合明细][likelihood] · [27-bin 数组][templates]

### 物理输入与统计覆盖

- **同一选择。** Stage 321 仅对外部中性强子应用 p0 veto 后取形状，维持硬编码外部规模，各 ν 分量不 veto。独立把 veto 同时应用各过程后，HL-LHC 信号 540→528.80，bin 18 含约 0.5304 信号、零背景。若将零 MC 当硬零，$H_0$ 排除该箱产生事件，无法报告有限稳健的显著性。以共同背景半事例 $b=15.05$ 续接该箱、加入外部 MC 方差的样例为 $Z\simeq0.354$。它依赖外部规模的选择级别和空箱规则，不能作为物理投影。 [压力测试][likelihood]
- **产额与系统。** 每曝光同样 $60/\mathrm{fb}^{-1}$ 与固定先验。外部 n/KL 的 25–600 GeV、60/40 混合与 70% 像素束是手设。外部 μ、外筒背景、不同源间时空重合和中性级联没有 rate closure。常量 4.5% NC 粲和单一 30% charm nuisance 无由此样本确定的过程建模证据。五个 G4 成组 1,200 事例为配对形状；tau helicity 只作用信号。聚合 ν 背景 morph 按三类归一，未通过六种 light/charm 过程、更细的 fiducial 与能谱迁移传播。Pythia6 generator 对照只粗匹配能量与类，无法区分母态流量和不同 tune 形状。 [模拟外部源](../src/analysis/flavour/v3/v6_clean_sim/produce_external_and_gen_sys.py#L44) · [系统形状](../src/analysis/flavour/v3/v6_clean_sim/run_stage321_clean_v2_campaign.py#L1307)
- **有限 MC 和检验。** 总 bin Gaussian BB-lite 会让 MC 背景相对不确定性缩放含 signal 的总率，信号方差未明确进入拟合，外部空箱无 variance。仓库用 Poisson 生成 auxiliary 后用 Gaussian fit，正负系统变动的率裁剪与 Jacobian 也未一致处理。1,000 toys 仅能以 0.001 粒度估计概率，不能约束 $5\sigma$ 对应 $2.87\times10^{-7}$。Stage 321 HL-LHC BB 的 2σ 尾为 0.040，渐近为 0.02275。独立单箱形如 $n\sim\operatorname{Pois}(10\gamma),m\sim\operatorname{Pois}(\gamma)$，200,000 toys，精确 $n\mid n+m\sim\operatorname{Binom}(n+m,10/11)$。将渐近 1σ/2σ 应用于该样例分别得到 34.16%/0.0595% 拒绝，精确尾为 0.134%/0%。离散观测使阈值显著偏离渐近分布，该算例未给出整个实验的拒绝率。完整 27-bin 多 nuisance 点和 $\mu>0$ 功效仍需独立校验。 [仓库 toys](../src/analysis/flavour/v3/v5_confirm/flux_significance.py#L715) · [独立 toys][likelihood]


## 七、科学新颖性与研究方向

最有潜力的贡献是为混合钨靶、超薄像素、stereo scint 与磁谱仪建立同一套可测量的重建基准，能将事件分类变化追溯到明确的对象级信息与真实硬件改动。现阶段最可信的物理研究方向是对 PV 纵向、背散射 veto 和低粒子动量下 tracklet 可辨识度做闭合。有效 MC、生成物理和真实响应优先于事件网络扩容。

- FASER 的真实碰撞中微子数据给出束线通量与实数据链参照。FPF 报告对新几何和不同曝光提供外部情景，输入应重新映射到 PINPOINT 的材料、空间分布与效率。 [FASER 中微子观测](https://arxiv.org/abs/2303.14185) · [FPF 综述](https://arxiv.org/abs/2203.05090)
- ATLAS GN2 的 track origin 与 pairwise vertex supervision 是合理技术方向。PINPOINT 的间歇取样与钨中的 MCS 决定了不同的轨迹可辨识度。与可传播协方差的 analytic tracklet/vertex fitter 对比，先要求对象闭合，再训练更多密集头。 [ATLAS GN2](https://arxiv.org/abs/2505.19689)
- NOMAD 的 $p_T$ 和隔离变量可建立 τ/普通 CC 鉴别的可证伪假设。对 PINPOINT 的研究先测 recoil 能标、lepton matching、ghost pair 与 resolution，再比较简单可解释 kinematic likelihood 和额外网络分支。 [NOMAD 文献检索](https://inspirehep.net/literature?q=collaboration%3ANOMAD%20and%20title%3Aoscillations)
- 稀疏物理尾部适合过程特有的 Poisson auxiliary 或加权 MC 约束。侧带与控制区的权重、系统变体与生成谱应能由审计目录中的独立模板逐步闭合。 [Barlow–Beeston](https://doi.org/10.1016/0010-4655(93)90005-W) · [Cowan 等](https://arxiv.org/abs/1007.1727)

## 八、优先级修复与实验计划

资源为初步量级。小样本先在现成 lxplus GPU/CPU 环境闭合，大样本模拟通过 HTCondor 并按实际 CPU 请求匹配执行线程。GPU 训练可使用 Kubeflow。所有新阶段使用独立数据和冻结路径。

| 优先级与问题 | 缺口与最小实现 | 数据、基线和对照 | 指标、统计比较与资源 | 证伪 / 验收 / 可得结论 |
|---|---|---|---|---|
| 必须 1。产生器与 Geant4 是同一核靶吗 | 保留 A/Z、源行和生成权重，Navigator 逐事例读顶点材料。统一 ν/反ν、CC/NC kinematic 范围 | 错配 3,825 个父样本、2k W 检查样本、独立 W GENIE 各味、同谱 NC charm/light。若模拟 Al/C，保留实际材料作为独立过程 | 非匹配计数、$E,y,Q^2$、末态及尾部。先 6–12 CPU，正式 50k–300k、80×8 Condor CPU 量级 | 任一错配证伪。零错配和有效产额/能谱闭合后，才建立物理一致的 W 基准 |
| 必须 2。真实选择后的本底尾部能约束吗 | 所有过程共用 observable cuts。记录外部归一的选择级别、独立控制区、auxiliary、移除概率与时间/轨迹 veto | 外部 rock μ、n/KL、级联和 beam exposure 控制，W 一致 ν 样本。对照无 veto、p0 veto、时空 matching veto | 用每关键过程每关键箱有效 MC 25 起步，0.1% 档若需 100 个 FP 至少约 $10^5$ 有效背景/域。10⁵–10⁶ CPU toys，考虑计算方差更低的尾部抽样 | 不同选择或控制区 rates 不闭合即证伪。尾部区间与 nuisance 覆盖达设计阈值再声明参数化灵敏度 |
| 必须 3。靶 pitch 是否独立于 IPT | 各传感器独立定义响应、聚类和 patch。分开验证 target 与 IPT 的坐标、聚类、候选与 model inputs | 同 W-only 事例共 seed，target×IPT 两因子对照。6T、12T 与物理拆面另设 | token 逐元素、PV/SV 与 BA4、ROC、seed 和资源对等。先 1k 样本和 4–12 GPU 小时 smoke | 任一 IPT 表征随 target-only config 改变即证伪隔离。以预先定义 ±0.5 pp BA4 等效为示例，并为 τ 尾部和 PV/SV 定独立 tolerances |
| 必须 4。是否有可定位的真实位移信息 | 统一 G4 母事例中改变 τ decay 点并重输运。真 PV/SV/track 对上重建，明确接受、配对效率和假分裂 | 固定 $E_\tau,y,z$、τ mode 的各距离和 charm；零微拓扑、全统计配平和物理原位 tracklet 基线 | SV/track efficiency、purity、residual、$d_0$ pull、kink、事件分数。先 1k 母样本×5 位移，6–12 核多任务 | 在短距离无正确 SV 匹配或位移扫描无响应即证伪。稳定优于匹配的 occupancy/energy 基线后才支持微拓扑因果解释 |
| 必须 5。推断能否跨产生器和读出部署 | Observation-only 入口、明确观测域、配置/input fingerprints、Cal/Cut/Confirm 独立运行 | 留一源验证、source-disjoint 核靶一致 MC 与控制数据。冻结 nuisance、阈值切选和算子版本 | 实测 FPR，ECE/NLL、先验、选择与跨域不确定性。1 CPU 小时 + 1 GPU 小时 smoke，正式按尾部规模 | 无标签目录推断数值闭合，并以事先规定界限检验外源可靠性，建立部署基线 |
| 可选 6。NOMAD、GN2 或 ensemble 是否有独立收益 | same architecture 加/删单 feature、同样训练预算，记录梯度和 32-event 小样本拟合 | B1++、简单 LR/BDT、matched seeds、加噪、打乱和可观测量 physics proxy | matched seed 测试，尾部、对象级误差、延迟。3–5 seeds 每组约 12–40 GPU·h | 持续没有可重复增量则降级。实测对象与尾部同时改善后保留复杂度 |

## 九、发表准备度

| 目标 | 裁决 | 可声明范围 |
|---|---|---|
| 内部物理合作组备忘录 | 具备 | 可发表仿真几何与 ROOT 闭合、P0/P1 证据、原始核靶计数与报告条件中的准确数值 |
| 仪器或重建论文 | 缺关键硬件证据 | 具体 MC 读出消融与 PV 误差及几何可写。硬件 pitch 方案、微米 SV 及实际 veto 效率须经独立真实响应 |
| ML 方法论文 | 需要有归因力的实验 | 可证明原型可训练、可复算。需要公平架构对照、强受观测基线、额外域泛化与对象证据 |
| 敏感性或发现论文 | 尚未准备 | 目前能写明确参数情景的条件数值。核靶、束流、控制区、实数据校准及覆盖完成前，不能写作真实发现预期 |

## 十、最终裁决

1. **仍有 P0/P1。** 产生器核靶错置与本底选择不相干各自阻断相应的物理声明，靶 pitch 同时改变 IPT 阻断硬件归因。
2. **历史问题部分纠正。** 24/84、硅坐标、单次 PE 响应、理想磁输运和校准数学确已改善，核靶、时间、部署与稀疏尾部仍有缺陷。
3. **可相信条件数值。** Confirm 的混合 MC 分类和给定假设的 Asimov 数值可复算。真实物理结论必须按上表重新建立。
4. **模型可作为观测型算法原型。** 哑标签加观测输入能运行。可靠部署需要实际响应与校准、可观测事件构建和域路由。
5. **有分类和 PV 信息，物理因果来源未拆分。** 能谱、空间、占用、生成器和真实核相互作用都能独立改变输出。微米 τ/粲表征未得到对象证据。
6. **独立确立的改进。** 几何和像元映射、硬边公式、PE-only 一次饱和、calibration 数值梯度，以及指定 6T 的 PV z 损失、Hybrid 的部分 PV 改善。
7. **须撤回或缩弱的声明。** 全部修复、完整物理发现投影、单因素 pitch 对照、实验固定 0.1% 假率、跨域 $S_\tau$ 同尺度、已证实微米次级顶点重建。
8. **值得继续。** 科学收益来自统一核靶、可测对象和外部 veto/本底，而后才是事件级模型优化。
9. **前三项实验。** 重建核靶一致的 W 和可控 NC 粲集。用统一选择及外部控制区重建似然。做正交 target/IPT 对照和匹配 τ 位移的对象级反事实。
10. **不同做法。** 先冻结产生器核靶、实读出、对象重建、统计尾部四套独立验收，再以实验设计为单位冻结训练和评价。保留反例、失败及未测结果，与正面结果同一可追溯标准。

### 复现、产物与测量边界

数值产物写入独立审计 EOS 目录，脚本在 `workbook/stage322_audit/`。模拟工作树的宏与实际 production 宏字节一致。使用分析仓库 `d65824c` 与模拟仓库 `6bc1744` 的源码。模拟执行用 LCG_107，分析用 LCG_108_cuda 的 Python 3.12 和 Tesla T4。数据回连和推断分片默认 6 workers，四个独立 G4 探针并行运行，BLAS/OMP 各用 2 线程。审计 RNG 322，G4 探针四个 seed 为 322100、322111、322180、322191，另一路 seed 加 17。unittest 352/352，204.261 s。

PF 全分割主要用冻结的 FP32 parquet，另验证五个真实 checkpoint；B1++ 的三名成员重算各分割与系统样本。运行 ROOT 探针、抽查实际 Clean v2 parquet，并对 60k 计划记录闭合。源码与冻结可执行的几何行为闭合，未重建二进制构建谱系。完整 Clean v2 生产 ROOT 已在生产阶段清理，无法逐事件直接回读这批 ROOT。未测实际 data-taking 信号、完整控制区、稀有 MC 的真实交互概率、最终 CAD，以及端到端训练复现。保持原检查点、参数和封存结果。已存在的未跟踪 docs 与 Stage 320 审计文件不纳入本审计提交。

~~~bash
audit=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/stage322_independent_audit_2026-11-19
clean=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v5_clean_v2_2026-11-19/clean_sim
mkdir -p "$audit"
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:$PYTHONPATH" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
python3 -m unittest discover -s tests > "$audit/unittest.log" 2>&1
bash workbook/stage322_audit/run_g4.sh "$audit"
python3 workbook/stage322_audit/geometry.py "$clean" "$audit"
python3 workbook/stage322_audit/source_audit.py "$clean" /eos/project-f/fasersim-bonn/public/pinpoint/data "$audit"
python3 workbook/stage322_audit/infer.py "$clean" "$audit" truth
python3 workbook/stage322_audit/infer.py "$clean" "$audit" b1
python3 workbook/stage322_audit/counterexamples.py "$clean" "$audit"
python3 workbook/stage322_audit/recalculate.py "$clean" "$audit"
~~~

[geo]: /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/stage322_independent_audit_2026-11-19/geometry_closure.json
[plates]: /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/stage322_independent_audit_2026-11-19/placement_table.csv
[parents]: /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/stage322_independent_audit_2026-11-19/source_material.json
[truth]: /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/stage322_independent_audit_2026-11-19/truth_interventions.json
[metrics]: /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/stage322_independent_audit_2026-11-19/classification.json
[counter]: /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/stage322_independent_audit_2026-11-19/counterexamples.json
[likelihood]: /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/stage322_independent_audit_2026-11-19/independent_likelihood.json
[templates]: /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/stage322_independent_audit_2026-11-19/template_hllhc.npz
