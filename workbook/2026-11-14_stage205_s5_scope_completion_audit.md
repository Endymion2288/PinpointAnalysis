# S5 原始范围审查、靶核匹配与τ几何结果

S5的原始执行要求现已逐项对应到实际产物。本次补齐exact-target联合匹配和全部validation τ的几何诊断，完成独立score-key读回，并修正执行入口中的“S5未执行”状态。[完整证据映射](../roadmap/results/s5_scope_completion_audit.json)与[阶段203完整矩阵](2026-11-14_stage203_s5_final_results.md)共同构成本阶段交付。

## 阶段目标

按照`roadmap/04_execution.md`第6节、`03_models_and_experiments.md`相关物理诊断和`12_s5_physics.md`检查完成性。区分研究已交付、方法未获得净收益、尚未测量的重建量及S6范围。沿用用户接受的samples、split和detector-proxy。

## 工作内容

原29结果的common-support只包含生成能量与真实顶点，未包含原工单中的target。本次加入exact `tgtPDG`，与原E/x/y/z bins联合匹配；不合并不同W同位素，不对支持外赋外推权重。对已保存29份概率重新评价33项同事件比较、400次accepted physical-block bootstrap，没有重新训练或改变模型选择。

补充读取32份τ ROOT中的primaries和τ直接子粒子字段。沿用forward-neutrino产生点定义，将带电daughter及parent方向直线投影到首个下游pixel面中心。记录层间位置、材料/间隙、入接受度数、parent/daughter位移和最小daughter-pair距离，按实际x/y pitch归一化，再连接存储response和既有分类分数。该计算没有模拟散射、磁场、shower及pattern recognition，不作为测得的kink或轨迹分辨率。

## 原要求与证据

下文`ROOT`为`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s5_2026-11-14`；精确路径与字段记录在JSON映射中。

| 原要求 | 直接证据与实测 | 状态 |
| --- | --- | --- |
| 四类与六子类求和 | 最终矩阵26神经网络、3 HGB；direct4/six/vertex+kinematics各17/42/2026三seed，八完整passes；six概率求和验证 | 完成；完整域six的τ AP下降，共同域区间跨零 |
| τ→e/μ/had、位置、prongs、可见性、charm和containment | `evaluation/final/summary.json`的sample strata、operating points和domain block intervals；τ分母444/395/1660，unknown1 | 完成；子道与inclusive分母分列 |
| 逐项辅助任务、标签定义/masks、组合和负迁移 | vertex/semantics/charm/kinematics各单项，combined/uncertainty；truth sidecar、manifest masks、health中的representation-gradient cosine | 完成；NLL的小幅收益与AP结果分别表述 |
| 对象先matching，再分类 | `objects_summary.json`、object coverage表、3 HGB及raw+objects；cluster/segment/station fake与vertex效率/resolution | 完成；当前算法fake严重，没有否定对象路线 |
| 可见/不可见及正确/错误/随机raw response | `native_event_examples.pdf`和`native_examples.csv`，固定random seed及各组分母 | 完成 |
| 能谱/真实vertex/target匹配 | 原E/vertex支持与新增`evaluation/scope/summary.json` exact-target支持，29模型/33配对/400 bootstrap | 已补齐；共同域缩小到3,179事件 |
| score与观测量联系 | true-class内occupancy/track/cluster/leakage-proxy相关性及英文图；新增投影域score分布 | 完成；相关性未当作因果解释 |
| 程序与物理来源分开审查 | 独立truth连接、白名单、未知mask、坐标mask修复、健康/readback；响应可见性、接受度、对象fake和原生图 | 完成；保留占用/位置与细拓扑的待查混淆 |
| predicted vertex-relative与absolute/raw | 同10维输入、同容量五项坐标训练及paired controls，raw路径保留 | 完成；收益限于单seed/对应任务 |
| px/py/pT、θ/φ分别评价 | 逆变换、绝对/相对residual与wrapped角；两项新head重训、真实fit、10k train/validation ridge诊断 | 完成；φ仍弱 |
| quality/score cut的resolution–coverage及能量/位置 | `kinematics.coverage`、strata和`object_resolution_coverage`及三份图 | 完成；quantiles没有称为校准预测区间 |
| 衰变材料/间隙、层间位置和pitch尺度 | 新增2,500事件几何Parquet、分道分母、投影CDF、存储response交叉表 | 已补齐；直线几何潜力与实际重建分开 |
| charm假τ、辅助/objects净贡献 | 最终summary的charm/charm-branch fake计数、全部33 paired comparisons和三seed表 | 完成；负结果保留 |
| 困难通道有计数、区间和物理解释 | 阶段203的visible/invisible及轻子分道；本报告新增pixel投影域0/655和控制 | 完成；未声称硬件信息上界 |
| 异常输出审查、修复重训和真实学习 | 1,291次健康采样、88,900更新、26 checkpoint复算、均衡六类24/24 fit；回归修复及方向读出 | 完成；旧S3失活结果未纳入结论 |
| 后端、成本、清理、报告和commit | `campaign_accounting.json`、终态ads、cleanup、不可变源码和提交记录；25条记录全部终止 | 完成；13主图+1补充图 |

原S3工单提到的拟合q/p/covariance没有被本报告写成已测量。固定输入的tracker z是station ordinal，当前S5合同交付三站position compatibility及fake/coverage。真实站点距离、field transport、轨迹参数与covariance需要独立重建合同；当前对象关联不能替代它们。对象学习式instance decoder是原路线图的可选扩展，本轮实现并比较的是固定观测对象。

## 详细数据

### Exact-target共同支持

每个联合格须有四类各至少5事件；权重为该格最小class count除本类count，上限1。保留82格、3,179事件，排除6,821；每类weight sum562，Neff=2,886.875109。仅W-182/183/184/186有共同支持，各同位素的每类weight sum分别138/15/197/212，对应保留744/89/1,150/1,196事件。C、Al等在当前细联合格下无支持，不将本结果外推给这些靶核。

三seed平均差值为variant减相同seed八pass direct4，使用同一组400次物理block重采样。下表仅为新增exact-target域。

| Variant | Metric | Mean delta | Block CI95 |
| --- | --- | ---: | --- |
| vertex+kinematics | NLL | −0.002353405 | [−0.004963051, −0.000034102] |
| vertex+kinematics | Macro AP | +0.001411675 | [−0.000986359, +0.003865881] |
| vertex+kinematics | Tau AP | +0.004098818 | [−0.000176555, +0.008597456] |
| six | NLL | +0.000555893 | [−0.003314712, +0.004522493] |
| six | Macro AP | +0.001367354 | [−0.002480608, +0.005162275] |
| six | Tau AP | +0.002857958 | [−0.004431715, +0.009591377] |

Vertex+kinematics在该域仍有小幅NLL改善，AP区间均跨零。Six的完整域τ AP差−0.004764有负区间，而较小共同域没有确认正/负差；支持域、能量/顶点与target权重同时变化，不能将差异归因为靶核本身。这些区间条件于已训练seeds、固定匹配权重和重复使用的validation，训练seed spread在完整JSON另列。

### τ的几何尺度、响应与score复核

2,500个τ中，2,486个衰变位于首/末pixel面中心之间；2,498个有下游pixel面。材料/位置分类为outside transverse1,046、tungsten695、scintillator578、silicon101、gap77、downstream2、upstream1。这里沿用既有geometry位置和厚度，不重新标定硬件。

| Quantity | Events | Measured result |
| --- | ---: | --- |
| 首下游pixel面内有forward charged daughter投影 | 655 / 2500 | 26.20%；其中653有存储pixel τ响应 |
| parent/daughter投影均在pixel内，可定义位移 | 651 | 位移median0.297461 mm；max-axis/pitch median12.3222 |
| 上述位移大于一个pitch | 601 / 651 | 全部τ分母下24.04% |
| 至少两个charged daughters投影在pixel内 | 100 / 2500 | 均来自hadronic分道 |
| 最近pair间距大于一个pitch | 95 / 100 | 全部τ分母下3.80% |
| decay至首下游pixel距离 | 2498 | median73.2332 mm，68% quantile102.3718 mm |
| 任意存储pixel τ响应 | 1200 / 2500 | 48.00%；与首面直线投影为不同分母 |

Pixel pitch为x20.8 μm、y22.8 μm；`pitch/sqrt(12)`的6.0044/6.5818 μm只作均匀单cell量化尺度。投影超过pitch不保证上游parent有可拟合hit，也不保证daughter轨迹在散射/shower后可关联，更不构成single-prong kink效率。

主模型`confirm8_vertex_kinematics/seed42`沿用1%背景阈值0.501878798，7,500背景通过75、全τ通过66。在首面daughter投影域为0/655，Clopper–Pearson描述计数CI95为[0,0.005616]；没有把zero-count block bootstrap的[0,0]当成真实效率已被精确测为零。该域τ score范围0.025541–0.496528，median0.233069；不是常数输出或浮点阈值相等导致的零计数。

重新从完整validation预测与几何Parquet独立按run/chunk/row连接2,500事件，最大score差0。Direct4 seed42控制同域也是0/655（全τ57）；HGB moments同域4/655（全τ66）。结果保存`geometry_crosscheck.json`。τ→μ有首面投影的103事件通过0，描述计数上限3.518%；较小分母不能给出强排除。

投影域/非投影域的入射能量median分别679.73/667.19 GeV，pixel channels median625/4，平均initX−10.293/−117.426 mm。直接证据支持明显的响应占用和横向位置关联；在不受额外控制的两组之间，无法分离样本接受度/背景差异与当前模型对细拓扑的提取能力。此现象作为明确待查混淆保留，未用它推出硬件没有分类信息。内部健康、真实小样本拟合及回归修复依据仍见阶段203。

### 辅助标签与新增成本

同一冻结10k监督事件，三个初始化seed的有效label masks均为vertex9,756、bar semantics9,492、charm10,000、kinematics9,971。它们是同一批事件的多种额外模拟监督，不能把有效mask相加当独立事件数。全部110k truth sidecar是为标签/评价提取，训练只使用所选10k；没有新增模拟或重新编译G4。标签提取和对象处理成本已包含在本阶段CPU allocation中。

补充CPU job1582683在bigbird20运行，提交前按idle/held/running及EOS/WRITE兼容性选择；终态JobStatus4、ExitCode0、4 CPUs、实际allocation89 s，新增0.098888889 core-hours。任务内最终stdout记录60.403120 s，summary在画图前记录55.427713 s；allocation与计算区间分列。

现在25条后端记录全部终止，总remote GPU allocation仍2.673888889 device-hours，CPU allocation为61.355000 core-hours；四次交互T4 fit232.280006 s另列。原EOS清理释放1,134,394,836 bytes不变；本次额外清理2份AFS脚本1,500 bytes，连同此前38份合计40份、33,756 bytes。`~/condor/Pinpoint/flavour_s5/`已空，精确脚本与源码快照保存在EOS。保留26 best checkpoints、3 HGB、完整预测、13主图及1补充图。

### 验证与复现

相关12项unittest全部通过（4.327 s），包括新增target/geometry数值测试；显式限制单文件解压线程后的真实ROOT reader smoke也通过。独立score-key replay为0；补充英文图已实际查看并修正横轴过密刻度。最终验收增加原范围诊断、匹配域类别/target权重相等和几何readback检查；完整matrix gate及资源closure均通过。完成性映射含18项要求、57条去重直接证据路径，均实际存在。

完整补充计算使用冻结源码aa0b331；后续代码增加保存产物的独立读回、限制单文件解压线程、图刻度和最终验收/记账/清理支持。概率、原始几何结果和模型选择没有随文档更新而改变。已终止allocation使用保存的真实终态证据，后续CPU记账无需重新请求已清理的Kubeflow资源。

最终6份相关Markdown的79个本地链接全部存在；两份更新/新增结果JSON解析与完成状态核对通过，14份PDF齐全，`git diff --check`通过。

```bash
# LCG108_cuda环境，PYTHONPATH指向仓库src。
# 完整计算通过4-CPU HTCondor运行；源码快照aa0b331。
python -m analysis.flavour.s5.scope_diagnostics --root /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s5_2026-11-14 \
  --workers 4 --bootstrap 400
# 只复核紧凑既有产物并重画图，无ROOT批量读取或新推理。
python -m analysis.flavour.s5.scope_diagnostics --root /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s5_2026-11-14 --readback-only
python -m analysis.flavour.s5.finalize --root /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s5_2026-11-14 \
  --campaign roadmap/configs/s5_final_matrix.json --require-closure
```

## 阶段性结论

S5的研究交付已完成。Vertex+kinematics有小幅NLL改善，尚未确认AP或τ高纯度效率净收益；six-way未建立晋级依据。Native pixel几何允许一部分τ具有超过pitch的投影位移，但当前神经网络的高纯度τ选择没有接受首面投影域，需继续分离占用/位置和细拓扑提取。对象fake与方位角重建不足有明确数据依据；它们均未被写成硬件信息上界。

## 下一步计划

S6冻结候选、概率校准、阈值和selection后，使用保留的独立test，评价条件域及预定detector-proxy扰动。真实硬件标定、flux产额和generator系统包络未测量，沿用用户接受值推进。若开展新的vector/probabilistic head、物理场下的q/p拟合或更强实例关联，应单列合同和同预算控制，保留本轮负结果。
