# 更新的本地汇报：结果核对与路线图调整

核对日期：2026-11-11。四份更新后的 PDF 共 61 页，已提取全文，并渲染核对文字层未覆盖的关键图表。以下页码均为从 1 开始的 **PDF 页码**，可能与幻灯片页脚相差 1。数值为汇报中的既有结果，本次未重跑这些分析。

本轮证据支持继续研究原生 pixel，同时把**接受度、位置关联、工作点与重建覆盖率**提升为必要对照。既有研究的三类结果不构成当前四类、尤其是 CC ντ 的性能依据。

## 1. 资料索引

| ID | Source | Date | Pages | Size (bytes) | Main evidence |
| --- | --- | --- | ---: | ---: | --- |
| L1 | [Pixel + scintillator vs scintillator only performance study](../../../docs/FORTVNE_pixel_vs_scintonly_meeting_260916.pdf)，Joern Mahlstedt | 2026-09-16 | 8 | 618,051 | 同验证事件、两种 fiducial 的 pixel 增量 |
| L2 | [Update kinematic variables](../../../docs/FORTVNE_Update_kinematic_variables_2026-10-07.pdf)，Joern Mahlstedt | 2026-10-07 | 9 | 3,546,193 | 分量级运动学、泛化与重建覆盖问题 |
| L3 | [PID on the New Sample](../../../docs/FORTVNE_10212026.pdf)，Zhongyi Wu | 2026-10-21 | 7 | 564,085 | 1 mm 投影 CNN/ResNet 与简化 FASER 重建 |
| F4 | [FASERCal, FASER Run 4 Workshop](../../../docs/FASERCal/FASERCal_wshopRun4.pdf)，Anna Mascellani | **2025-11-14** | 37 | 8,122,725 | 全重建→预训练→多任务分类/回归；τ 与低背景工作点 |

这些文件在上一阶段曾是登录页，现已全部替换为有效 PDF。本文件及当前路线图使用更新内容，stage02 报告保留当时的检查记录。2025 workshop 的配置、样本与初步结果分别记录，不与 2026 arXiv v2/Nature Machine Intelligence 数据混合。

## 2. L1：pixel 的收益强烈依赖所研究的体积

### 2.1 设置与数值

第 2 页声明：相同 Geant4 samples、相同 validation events，三分类 `νe CC / νμ CC / NC`；关闭 pixel branch 后从头训练 Scint-only 模型。pixel 的具体空间压缩、模型版本与生产 run IDs 未在该汇报完整列出，不能把 “Scint + Pixel” 自动解释为原生 pixel 输入。

两种体积定义为：pixel-centered 使用 pixel 横向边界内缩 2 cm、vertices 在最前 50 cm；large/scintillator volume 使用 Scintillator 横向边界内缩 1.5 cm、下游端内缩 20 cm。实际坐标原点、target 区域和代码选择仍需复核后映射至当前 ROOT geometry。

第 3 页的 one-vs-rest validation ROC-AUC：

| Fiducial | Inputs | νe CC | νμ CC | NC |
| --- | --- | ---: | ---: | ---: |
| Pixel-centered | Scint + Pixel | 0.970 | 0.906 | 0.876 |
| Pixel-centered | Scint only | 0.944 | 0.842 | 0.801 |
| Large | Scint + Pixel | 0.923 | 0.774 | 0.719 |
| Large | Scint only | 0.925 | 0.747 | 0.698 |

中心体积 AUC 增量为 `+0.026/+0.064/+0.075`。large 的 νe 两者接近；表中舍入值之差为 −0.002，旁注写 −0.003，保留这一精度差异，不将千分位差值作为精确结论。large 的 νμ/NC 增量为 +0.027/+0.021。

第 4 页在 90% purity 的示意工作点：νe efficiency 约 50%→70%，νμ 约 51%→80%，NC 在该高纯度工作点约不变。这是约 **20/29 个百分点**，不是相对百分比；第 6 页另写 νμ +28%，与图中舍入值略有差异。第 5 页 large 在 80% purity：νμ 约 52%→65%，νe/NC 约不变。不能由某个工作点 NC 近似不变推断其整条曲线或 AUC 没有收益。

第 7 页 confusion counts 的逐行总数在两种输入之间一致：pixel-centered 的 `νe/νμ/NC=3,368/18,821/7,234`，合计 29,423；large 为 `20,130/107,826/40,551`，合计 168,507。这与 matched validation 的表述一致，但没有原始事件键就不能独立验证 identity 或 split。large 的 NC argmax recall 在加 pixel 时为约 0.45、Scint-only 为约 0.68，而 NC AUC 提高；这说明排序能力和多类 argmax 决策应分别报告，不能只用一个 confusion diagonal 选模型。

### 2.2 对新路线的影响

已有证据支持“pixel 在中心区域有额外分类信息”；其在全接受度内、当前四类及 τ 子道的增益尚未确定。S0 固定全目标域、pixel-centered 与 large 条件定义；S2 对同一批模型在这些区域交叉评价，不先依 true vertex 筛掉困难事件。另做训练域控制，分清测试接受度变化与训练谱形变化的影响。

所有 pixel 消融须同时给 native response、位置/occupancy、能量与 FASER acceptance 分层，以及固定背景接受率和固定 purity 的工作点。purity 表需写明类别混合/权重。最终主结果仍包含目标域内无 pixel 响应的事件。

## 3. L3：未见 pixel 增益的结论对应 1 mm 双投影

### 3.1 实际输入与训练

第 2–3 页给出 `80/10/10` train/validation/test，40 epochs、cosine learning rate、相同 class weights：

- Scintillator 为两个 `panel×bar` 图像：`68×40`、`56×40`。
- Pixel 为两个 `plane×1 mm bin` 图像：`15×250`、`15×200`。这同时压缩横向分辨率并丢失同 plane 内 x–y hit 配对关系。
- FASER 为简化 track reconstruction 得到的 12 个特征；M1=strip+FASER，M2=strip+pixel+FASER，M3=strip+pixel。
- Base CNN 每视图 4 个 convolution layers，M1 约 0.2M parameters；ResNet 为 stem+3 stages×2 blocks，M1 约 1.5M、加 pixel 约 2.9M。

pixel 图像的 250/200 个 1 mm bins 与本次 ROOT 的 `266×196 mm²` 活性尺寸不同。需要追溯它对应旧几何、固定图像范围还是裁剪，并检查 detector/world 坐标及保留 hit 比例；仅凭 shape 不能认定现有数据被裁剪。

第 4 页报告的 efficiency/purity（均为汇报工作点）：

| Network | Inputs | νe eff. / purity | νμ eff. / purity | NC eff. / purity |
| --- | --- | --- | --- | --- |
| Base CNN | M1 | 0.518 / 0.953 | 0.379 / 0.965 | 0.241 / 0.863 |
| Base CNN | M2 | 0.478 / 0.946 | 0.387 / 0.960 | 0.234 / 0.845 |
| Base CNN | M3 | 0.506 / 0.956 | 0.891 / 0.782 | 0.216 / 0.878 |
| ResNet | M1 | 0.571 / 0.979 | 0.432 / 0.977 | 0.256 / 0.953 |
| ResNet | M2 | 0.519 / 0.978 | 0.431 / 0.971 | 0.259 / 0.927 |
| ResNet | M3 | 0.555 / 0.976 | 0.890 / 0.813 | 0.247 / 0.932 |

这些数字支持汇报所说的 ResNet 改善及此输入下 pixel 未见明确增益。M2 相对 M3 的 νμ purity 增加，但 efficiency 降低，必须比较完整曲线或同工作点指标，才能量化 FASER 的净贡献。第 5–6 页的 selection-efficiency matrix 行和小于 1；例如 Base M1 的 νe 行为 `0.518+0.003+0.004=0.525`，有未进入三个 selection 的事件，不是完整 argmax confusion matrix。精确阈值、是否选择互斥、fiducial/触发分母、类权重值及新样本 run IDs 尚需原分析配置补齐。

### 3.2 FASER 简化响应的可用参考与边界

第 2 页列出 `σx=0.58 mm、σy=17 μm`，99% hit efficiency、无 noise hits；0.1 mm 内 steps 合并，中间 hit 与 x 方向直线相差小于 3 mm；`|s|<25 mm` 对应其估算 `p≥4 GeV`，动量 capped at 2.4 TeV，straight back-projection。引用为 SCT design 文献 arXiv:1811.10243 与 FASER tracking arXiv:2207.11427。

这些是旧研究的可复核起始假设，尚不是当前生产几何/field/conditions 已验证的 digitizer。需核对坐标与 bending plane、step 合并是否依赖 truth、轨迹组合与 fake rate、sagitta 分辨率和 momentum cap 造成的偏差。无 track、拟合失败、出接受度和 capped momentum 分别编码；优先拟合带协方差的 `q/p`，不能把 cap 当作真实测量。FASERCal 2025 的铁芯谱仪也不能作为 PINPOINT 下游 FASER 的条件替代。

### 3.3 与 L1 的统一解释及新增对照

两份报告的 fiducial、生产样本、网络、是否加入 FASER、输入压缩和评价工作点并未统一；不能据此断言互相矛盾，也不能将差异单独归因于 rebin。S2 用固定其他模态的 `2×2` 因子对照：

| Pixel geometry | Joint pixel locations | Separate xz / yz projections |
| --- | --- | --- |
| Native pitch | 原生 `(x,y,z)` / plane-aware sets | native-pitch 两视图边缘分布，不传 x–y 配对 |
| 1 mm transverse grid | 保留粗 cell 的联合 x–y 位置 | 与 L3 相近的 1 mm 双投影 |

先明确每视图的 OR/count/signal reducer，并固定可比较的通道统计；分别测量空间粒度损失、投影关联损失及其交互。native 投影可用 sparse sets，避免展开巨幅 dense images。保留 100/200 μm 中间网格作尺度曲线。表中每项使用相同事件、读出、模态、split 与评价；旧 CNN 数字只作历史参照。

## 4. L2：运动学问题更指向局部分量和覆盖率

第 2–7 页分别展示 large/small fiducial 下 νe、νμ charged-lepton `|p|、pT、pz、θ` 与 DIS `xBj/y/Q²/W²` 的 residual/response；标题带 “Score WP”。图像为嵌入 raster，已人工查看。small/large 的精确定义、score cut 和完整事件分母未在该文件给出，不能默认与 L1 两体积完全相同。相对动量 residual 与角度 residual 的有效事件数也应分别记录。

第 8 页总结了已尝试方向：替代 target/loss 改善 pT magnitude、但 azimuth 泛化不足；frozen-feature/pooling probes 没有找回足够横向信息；PID/HID reweighting 提高 muon-hit efficiency 但牺牲 purity/global performance；hit/hybrid reconstruction 对部分高纯事件方向精确、覆盖有限；overfit 与 CC-only fine-tuning 能学训练目标但 validation 仍是瓶颈。它支持优先检验 pixel/component 表示，而不能单由训练集拟合就证明硬件上全样本可辨识。

第 9 页建议 vertex-centred coordinates、component directions 和 quality selections。新路线将其具体化为：

1. 仅用**重建/预测 vertex**居中，附 uncertainty/validity，保留 absolute detector frame 与 raw-hit fallback；true vertex 只作诊断上限。
2. 在全局 pooling 前保留 track/shower/component direction；比较 local+global、raw+objects，先报告关联 efficiency/purity/fake rate。kinematics 为可选辅助任务，其晋级还需改善分类或解释困难通道。
3. 分开报告 `pT magnitude、px/py、wrapped Δφ、Δθ`，不能用 polar θ 的窄 residual 代替 azimuth φ 可用性。低 pT 时 azimuth 不稳定，使用目标有效性/不确定度与绝对分量 residual。
4. 每张 resolution/response 图配 all-event→reconstructable→quality-selected 的计数、权重与覆盖率；扫描 quality threshold 的 resolution–coverage 曲线，分别检查能量/位置/类别。不能用少量高纯事件的精度代替全样本能力。

## 5. F4：全链路与 τ 工作点的早期证据

### 5.1 应借鉴的重建步骤

第 8–16 页展示的是 2025 概念设计：10 个 3DCal modules、1 cm scintillator cubes、ECAL、AHCAL 与约 1.5 T magnetised iron muon spectrometer；WLS fibres 三方向投影先重建 3D voxels，spectrometer 用 GenFit/Kalman 重建 tracks，之后才送 ML。第 12 页的 11 tracking stations/4 SciFi planes 及预期 momentum resolution 属于这一谱仪。第 13 页还展示增加 tungsten target mass 会损失 tracking，支持在研究中保留材料/接受度依赖。

第 17–18 页采用 masked reconstruction + relational semantics 预训练，再联合 flavour/charm classification 与 `pvis/pjet` regression。样本为约 1.5M unbiased + 68k τ-only，split 85/5/10，和 2026 v2 的样本数不同。其核心启发是“真实读出→带歧义和误差的重建→共享表示”，不是把模拟 truth tracks 或理想三维坐标直接作为硬件输入。PINPOINT bar 只有两向不完整信息，不具有该三视图 cubic readout 的完整性。

### 5.2 工作点及分母

第 19–21 页给出 on-axis、780 fb⁻¹、其 detector/flux/selection 下的优化工作点：

| Selection | Threshold | Efficiency | Purity | Weighted signal / background |
| --- | --- | ---: | ---: | --- |
| νe CC | p(νe)>0.90 | 67% | 99% | 约 1001 / 9 |
| νμ CC | p(νμ)>0.96 | 84% | 约 100% | 约 6144 / 5 |
| NC | p(NC)>0.94 | 58% | 99% | 约 1711 / 13 |
| ντ→had，preliminary | p(τhad)>0.95 | **11% of τhad** | 66% | 约 1.9 / 1 |

τhad 行的分母是该表 17.7 个 hadronic τ 期望事件，`1.9/17.7≈11%`；对全部 τ 的分母 `17.7+4.4+4.6=26.7`，由表中舍入数得到效率约 **7.1%**。两者不能混用。汇报称 background acceptance 约 `8×10⁻⁵`；这是强烈依赖 flux/工作点与有限 MC 的初步结果，不能直接设为 PINPOINT 已有 rejection 或 expected yields。

第 21 页明确指出 τ→e/μ 与 prompt e/μ 背景难分，仅靠 missing pT 不足；因此优先用 τhad 建立有统计支撑的诊断，同时保留 leptonic τ 分支和最终 inclusive 四类目标。charm 结果在第 33–34 页另给，不能由 flavour 的高纯度推导 charm 背景已解决。

### 5.3 运动学定义与选择条件

第 22–24 页构造 `pℓ=pvis−pjet`、`pTmiss=|(pvis)T|`，并将 `|pvis|` 作为 Evis proxy。`|Σpᵢ|`、`ΣEᵢ` 与 incident Eν 是不同量；只在相应前向、质量、不可见粒子与 containment 近似下接近。新研究保存独立 target 定义，不把这些等式当作普适能量守恒条件。

第 25、30 页的 resolution 图要求 class-specific score cuts，而第 22–24 页的总体分布没有 tagging selection。第 32 页 pTmiss residual 使用 `(true−reco)/true`，与常用 `(reco−true)/true` 符号相反；注释又明确 Gaussian fit 用所有事件，橙色另画 P>0.9 子集。新报告统一 residual 符号，写明 fit/support/selection，报告非高斯尾部和 coverage。第 31 页 argmax efficiency 随能量变化，与第 19–21 页优化阈值的 efficiency 也分开引用。

## 6. 落入研究阶段的具体变更

| Stage | Added requirement | Deliverable / gate |
| --- | --- | --- |
| S0 | 追溯 L1–L3 run IDs、fiducial/score/weight 定义；核对 L3 tracker 简化假设 | 历史与当前样本的适用边界、版本表；缺失项保留待查状态 |
| S1 | bar/pixel/tracker response 与 reconstruction coverage 分开记录 | 可复核的 no-response、out-of-acceptance、failed-fit、capped-value masks |
| S2 | pixel 的粒度×联合/投影 2×2 对照，加 fiducial 分层 | 区分空间分辨率、位置关联和接受度的增益；保留三类历史参照，主评价四类 |
| S3 | FASER 增量按固定 purity/background acceptance 比较 | 完整 efficiency–purity/ROC 曲线、类混合、track acceptance 与资源 |
| S5 | 预测 vertex/局部分量与 raw fallback；θ/φ 分开；τhad 与 τleptonic 分道 | resolution–coverage、各 τ 分母、charm 混淆与全样本性能 |
| S6 | 固定 residual、selection、finite-MC 和 flux 定义 | 不将 selected precision、AUC 或一个阈值等同于最终物理灵敏度 |

当前数据的 pixel ID/label/production 问题仍按 S0 处理。新资料提高了研究问题的针对性，没有替代现有数据合同验证，也没有给出新四分类的测量性能。
