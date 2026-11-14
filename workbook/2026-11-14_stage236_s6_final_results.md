# S6：冻结独立测试、条件效应与最终研究结果

修复后的MAE3三seed ensemble在724,441个独立test事件上的balanced accuracy为**43.5226%**，95%物理块区间为**[43.3705%,43.6685%]**。相对同预算Scratch ensemble，增量为**0.3354个百分点**，区间[0.2508,0.4166]。冻结validation 1%背景阈值下，τ效率为**3.5556%**，test实际weighted FPR为**1.2080%**。本轮支持预训练的小幅收益；当前低背景τ选择能力仍不足，稀有τ先验下纯度低。旧故障S3结果仅保留为诊断历史。

## 阶段目标

完成[执行路线图第7节](../roadmap/04_execution.md#7-s6系统效应与最终物理结果)和[S6冻结合同](../roadmap/13_s6_final_evaluation.md)：概率校准、全部accepted独立test、预定方法对照、物理分层、读出与真实transport扰动、可用generator对照、τ/charm实现证据、统计区间、英文图和资源关闭。样本、split和detector-proxy沿用用户接受状态，来源认证与真实硬件标定不再作为前置工作。

## 工作内容与冻结边界

输出根目录记为 `R=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s6_2026-11-14`。原始输入仍为 `/eos/project-f/fasersim-bonn/public/pinpoint/data` 的10081/10082/10083/10084；类别顺序为 `CC νe / CC νμ / CC ντ / NC`，包含相应反中微子。真实transport基于匹配的Geant4实现，在R下隔离构建，未修改原始模拟目录。

| Partition | CC νe | CC νμ | CC ντ | NC | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Calibration | 87,281 | 414,559 | 64,908 | 155,524 | 722,272 |
| Independent test | 86,905 | 416,305 | 65,558 | 155,673 | 724,441 |
| Empty response in test | 2,297 | 9,994 | 2,303 | 4,258 | 18,852 |

所有accepted test事件均保留，包括空响应。主方法为100k标签预算MAE3的seeds17/42/2026三份raw probability均值，再拟合一个ensemble temperature。Scratch三seed ensemble为同预算pretraining控制。另有S3修复版native/1mm/tracker、S5 direct4/vertex+kinematics/six及channel-Rel5的预定seed42控制；总计13个checkpoint、15个单模型或ensemble、四个prior，共60项结果。跨阶段控制的预算不同，不把其差异解释成总体架构排名。

主模型采用此前重新训练验证的revision 3多尺度编码器：dim96、64个latents、32个region blocks×2 queries、readout skip、残差scale0.1、Head与融合尺度规范化，并使用empty-response bypass。完整架构与训练论文依据在[S3修复](../roadmap/10_s3_repair.md)及[S4](../roadmap/11_s4_pretraining.md)。本阶段评价这些已修复checkpoint，内部采样另见下文；没有沿用36,547→124→24的故障模型结果。

Calibration仅拟合单参数temperature；原有10k validation按固定规则选择每类1%和0.1%背景阈值。**2026-11-14T13:50:23.738039Z**写入`analysis_freeze.json`后才开放test。模型、prior、温度、阈值、条件、subset与统计定义均未用test重选。失败分块按相同冻结campaign恢复。校准与label-shift解释依据Guo等ICML2017、Lipton等ICML2018；引用及适用范围见[S6方法依据](../roadmap/13_s6_final_evaluation.md#方法与计算依据)。

事件统计采用7,240个accepted-family连接的source-chunk物理块，seed2026、400次block bootstrap。配对比较复用同一组重采样；generator两个样本的hard scatter独立，采用独立source-file block。区间为pointwise percentile 95%，条件于既有checkpoint、calibrator、阈值及训练种子；三seed spread另列，不当作完整训练不确定度。Raw整数尾计数另给Clopper–Pearson区间，Neff不替代raw分母。

## 详细数据：主结果与概率校准

Balanced prior为四类各0.25；主表按此固定混合加权，test Neff为449,611.3174。以下指标均来自全部724,441个事件。

| Metric | Value | 95% block interval |
| --- | ---: | --- |
| Balanced accuracy | 0.435225878 | [0.433704723, 0.436685364] |
| Macro F1 | 0.436624877 | [0.435149510, 0.438084679] |
| Macro AP | 0.475020253 | [0.473628550, 0.476387304] |
| Macro AUROC | 0.705487288 | [0.704491399, 0.706571189] |
| NLL | 1.201590409 | [1.199946985, 1.203085549] |
| Brier | 0.652385531 | [0.651538412, 0.653213914] |

| Class | AP | AUROC | Argmax recall | Argmax F1 |
| --- | ---: | ---: | ---: | ---: |
| CC νe | 0.609545 | 0.790668 | 0.520235 | 0.531582 |
| CC νμ | 0.480791 | 0.705575 | 0.411381 | 0.430271 |
| CC ντ | 0.346679 | 0.630584 | 0.376400 | 0.359441 |
| NC | 0.463066 | 0.695123 | 0.432888 | 0.425206 |

Raw confusion counts如下，行是真值、列是预测；加权precision与下表的raw类别混合不同。

| True / predicted | CC νe | CC νμ | CC ντ | NC |
| --- | ---: | ---: | ---: | ---: |
| CC νe | 45,211 | 8,877 | 20,237 | 12,580 |
| CC νμ | 51,083 | 171,260 | 99,944 | 94,018 |
| CC ντ | 12,132 | 13,498 | 24,676 | 15,252 |
| NC | 20,130 | 30,010 | 38,144 | 67,389 |

主temperature为1.007143202，未触及预定[0.05,20]边界，拟合导数约−5.55×10⁻¹²。Calibration balanced NLL从1.202473559降至1.202467551。Test top-label ECE（20个固定等宽bins）从0.007038878降至0.007006804，变化很小；bin counts、逐类reliability及条件域结果保留在各`result.json`。这一名义结果不保证shift条件或低统计尾部的校准。

完整tie-aware ROC/PR保存在每个模型/prior的Parquet；曲线为描述性test结果，未从test曲线挑新工作点。英文图：[confusion](/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s6_2026-11-14/evaluation/figures/test_confusion.pdf)、[ROC/PR](/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s6_2026-11-14/evaluation/figures/test_roc_pr.pdf)、[reliability](/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s6_2026-11-14/evaluation/figures/test_reliability.pdf)。

## 详细数据：固定选择与prior情景

| Validation target background acceptance | Frozen τ threshold | Test τ efficiency | 95% block interval | Actual weighted test FPR | Weighted τ purity |
| --- | ---: | ---: | --- | ---: | ---: |
| 1% | 0.4467281811 | 0.035556301 | [0.034210717, 0.037062582] | 0.012080097 | 0.495236618 |
| 0.1% | 0.5169396211 | 0.003111748 | [0.002709101, 0.003539863] | 0.000945358 | 0.523174483 |

1%行对应τ通过2,331/65,558、背景通过8,787/658,883；背景中e/μ/NC分别731/5,948/2,108。0.1%行对应τ204/65,558、背景712/658,883，背景分解49/494/169。Raw FPR与weighted FPR分开保存；上表使用balanced背景混合。Validation 0.1%阈值只有7/7,500个背景通过，已标低尾统计；当前block区间条件于这个冻结阈值，不覆盖阈值估计不确定度。Test所有60配置的总背景通过数均≥25，未为此补造稀有背景样本。

带reject的四类selection采用各自阈值；多类同时通过时按冻结`log(p_c/threshold_c)`决策。主balanced 1%选择weighted reject率0.862568479，raw reject623,063，多阈值冲突0。Raw 4×5选择矩阵、all/target/active/selected cutflow和每类exact区间见主`result.json`。实际test FPR偏离validation目标时保留原cuts。

| Prior scenario | Assumed τ prior | τ efficiency at frozen 1% cut | Actual weighted FPR | τ purity |
| --- | ---: | ---: | ---: | ---: |
| Balanced | 0.25 | 0.03555630 | 0.01208010 | 0.49523662 |
| Production sampling | 0.08986642 | 0.02908875 | 0.01019143 | 0.21986338 |
| Tau 1 percent | 0.01 | 0.02770066 | 0.00967788 | 0.02809937 |
| Tau 0.1 percent | 0.001 | 0.02722780 | 0.00947856 | 0.00286720 |

各scenario的prior修正和validation阈值均在test前冻结。Production sampling为原生成采样混合[0.12084229,0.57396521,0.08986642,0.21532608]，并非物理flux；另两个稀有τ情景完整prior为[0.1,0.6,0.01,0.29]和[0.1,0.6,0.001,0.299]。稀有τ情景下当前工作点纯度很低。[Prior图](/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s6_2026-11-14/evaluation/figures/tau_prior_scenarios.pdf)同时给两个冻结工作点的区间。

## 详细数据：预训练、粒度与辅助任务

同预算Scratch ensemble的BA0.431872256、NLL1.209077236、τ效率@冻结1%为0.029241283。MAE−Scratch的paired差为：

| Metric | Difference | 95% paired interval |
| --- | ---: | --- |
| Balanced accuracy | +0.003353622 | [+0.002507538, +0.004166165] |
| NLL | −0.007486826 | [−0.007850644, −0.007094435] |
| τ efficiency at frozen 1% cut | +0.006315019 | [+0.005285655, +0.007292114] |

MAE三单seed BA为0.433756942/0.433243285/0.431334083，样本SD0.001276658；Scratch为0.430554579/0.430853778/0.429050845，SD0.000966204。Ensemble的事件区间与这些seed spread分别保留。本轮test支持在100k标签预算下的小幅收益，不把它外推为已证明全量训练或所有generator上的改进。

以下为预定seed42阶段控制，均使用各模型自己的冻结validation工作点；每格为差值和95%paired区间。

| Comparison | Δ balanced accuracy | Δ NLL | Δ τ efficiency at frozen 1% cut |
| --- | --- | --- | --- |
| Native − 1 mm | +0.000511 [−0.000539,+0.001444] | −0.002820 [−0.003188,−0.002464] | +0.000061 [−0.000750,+0.000885] |
| Tracker − native | +0.026564 [+0.025494,+0.027827] | −0.021126 [−0.021715,−0.020556] | +0.001083 [+0.000031,+0.002138] |
| Vertex+kinematics − direct4 | +0.000167 [−0.000633,+0.000958] | −0.001035 [−0.001355,−0.000676] | −0.001388 [−0.002238,−0.000410] |
| Six − direct4 | −0.000623 [−0.001652,+0.000405] | −0.002422 [−0.002846,−0.001976] | +0.001464 [+0.000366,+0.002556] |

原生细粒度对BA和τ工作点的增量当前未确认；其NLL、macro AP等有较小改善，不据此判定原生信息无用。Tracker在对应S3预算下提供明确总体收益。Vertex+kinematics的NLL改善伴随τ固定阈值效率下降；six在该test工作点有收益，但BA区间跨零。S5 validation的τ AP结果与本表的test工作点是不同指标和分母，不追溯改写S5选择。[模型图](/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s6_2026-11-14/evaluation/figures/test_controls.pdf)只展示预定不同预算控制，完整差值在`evaluation/paired_comparisons.json`。

## 详细数据：物理分层与共同支持

| Test stratum | τ events | Balanced accuracy | τ efficiency at frozen 1% cut |
| --- | ---: | ---: | ---: |
| All | 65,558 | 0.435226 | 0.035556 |
| Target envelope | 37,457 | 0.518085 | 0.061297 |
| Pixel centered | 4,882 | 0.561929 | 0.003482 |
| Active response | 63,255 | 0.438654 | 0.036851 |
| Empty response | 2,303 | 0.250000 | 0 |

Pixel-centered域τ通过17/4,882，95%block效率区间[0.002021,0.005257]。这与其较高的argmax BA同时成立，说明总体分类与低背景τ选择需分别评价。Target envelope是冻结几何包络诊断，不等于精确W形状survey，也不作为模型输入。

τ→e/had/μ分别11,538/42,678/11,302个，效率为0.020194/0.043816/0.020085；另40个unknown保留，效率1/40，95%block区间[0,0.071566]。Pixel占用0、1–100、100–1000、1000–10000、10000–50000对应τ数16,685/27,867/14,879/6,121/6，效率0.000300/0.062547/0.037973/0.002941/0。最后一bin统计很少，bootstrap只有399个有效τ分母，不把零效率解释成物理定律。

Incoming energy<100、100–300、300–1000、1000–3000 GeV的τ效率为0/0.003598/0.039807/0.053370；第一bin0/2,205保留exact计数上限，≥3000 GeV没有τ，指标undefined并在图中标“No events”。[物理条件图](/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s6_2026-11-14/evaluation/figures/tau_physical_conditions.pdf)给block区间；sign、x/y/z、charm、tracker接受度及可重建quality等共63个非空预定strata保留在各`result.json`。Containment只测读出边缘/末层活动代理，真实能量泄漏比例未测量。

联合energy/x/y/z/exact target PDG的固定common-support诊断保留925/2,360个populated cells、686,693个事件，排除37,748个，Neff411,249.4532；不是物理flux重加权。主模型BA0.418782808、Scratch0.416704496，主NLL1.218058886；九个预定模型及其paired区间保存在`evaluation/exact_target_overlap.json`。完整accepted test主结果不由该子域替代。

**事实**：当前τ工作点效率随占用和位置显著变化，tracker有总体分类收益。**待检验假设**：高占用事件的细拓扑尚未充分进入最终判别，可能涉及局部编码/压缩、优化和接受度。条件关联未分离这些原因，后续不据此直接在已公开test上调模型。

## 详细数据：读出proxy与真实transport

读出对照为每类8,000、合计32,000个固定test事件，14条件共448,000个响应。包含nominal、gain、threshold、sparse noise、inefficiency、dead channels、coordinate bias、tracker resolution/efficiency、组合及未参与训练augmentation的heldout pattern。主ensemble在同一子样本nominal下BA0.436750、τ效率0.036500、实际FPR0.0127917。所有条件复用名义校准与cuts，以下区间为相对该nominal的400次paired物理块区间。

| Readout condition | Δ balanced accuracy | Δ τ efficiency at frozen 1% cut |
| --- | --- | --- |
| Combined | −0.0160625 [−0.0198792,−0.0118468] | −0.0080000 [−0.0103767,−0.0059647] |
| Sparse noise | −0.0177500 [−0.0211378,−0.0141489] | −0.0007500 [−0.0023800,+0.0010175] |
| Gain low | −0.0011250 [−0.0026440,+0.0004716] | −0.0055000 [−0.0074356,−0.0035731] |
| Gain high | −0.0005938 [−0.0020179,+0.0006216] | +0.0033750 [+0.0018931,+0.0048435] |
| Heldout pattern | −0.0020938 [−0.0041356,−0.0003024] | +0.0025000 [+0.0010156,+0.0037788] |
| Tracker resolution high | −0.0001875 [−0.0009189,+0.0004700] | −0.0008750 [−0.0016212,−0.0003718] |

Gain high时实际FPR0.0144167、纯度0.479699；heldout pattern时FPR0.0137917、纯度0.485226。效率增加伴随更多背景，不能只据效率称方法改善。Combined的实际FPR0.0098333、纯度0.491379。完整14条件、四prior、逐类指标、migration、reliability及三seed差在`evaluation/conditions/variants/`；[读出图](/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s6_2026-11-14/evaluation/figures/readout_conditions.pdf)展示τ效率差。幅度是用户接受的detector-proxy压力条件，并非测得的硬件systematic；未重新标定或检验其认证。

真实Geant4 transport使用1,024个固定hard scatter/vertex、四类各256；nominal/field0.95/field1.05/W-density0.95/W-density1.05各1,024个响应，加τ helicity −1/0/+1各256个，共5,888。Nominal BA0.439453、τ效率0.039063、FPR0.015625。以下主τ效率差均未确定方向：

| Transport condition | Δ τ efficiency | 95% paired interval |
| --- | ---: | --- |
| W density 0.95 | +0.003906 | [−0.019557,+0.026035] |
| W density 1.05 | +0.011719 | [−0.016529,+0.042158] |
| Field 0.95 | −0.003906 | [−0.028000,+0.018387] |
| Field 1.05 | +0.019531 | [−0.004549,+0.044903] |
| Tau helicity −1 | +0.015625 | [−0.011910,+0.043178] |
| Tau helicity 0 | 0 | [0,0] |
| Tau helicity +1 | 0 | [−0.018607,+0.022928] |

Helicity+1的BA差+0.016602，CI[+0.000980,+0.032529]；τ工作点并未相应改善。Helicity0响应与nominal相同，Pythia SPINUP=0不称为“unpolarized mixture”。部分FPR/纯度压力差有非零区间，完整结果随`evaluation/conditions/transport/`交付；不能因τ效率区间跨零就称所有系统等价。[Transport图](/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s6_2026-11-14/evaluation/figures/transport_conditions.pdf)给BA/τ效率差。固定hard scatter与vertex仍允许不同条件下G4随机流分叉，有限transport stochastic差异包含在本比较内。

## 详细数据：generator及τ/charm实现证据

找到并使用以下可读兼容输入：

```text
/eos/experiment/fasernu-data0/MC/Genie_files/genie_output_Pythia6tunes/
  Pythia6_Standard|Pythia6_NUX/
  EPOSLHC_pi|EPOSLHC_k|Powheg_pythia_charm_central/0..19/
```

配置为GENIE3.04、G18_02a_00_000系列的Pythia6 Standard/NUX hadronization对照。每套e512、μ512、τ220、NC512，共3,512个独立hard scatters；按class、flux source、signed incoming PDG、W isotope及固定energy bins匹配，在共同benchmark vertex支持下transport。两个generator样本不按event index配对。NuWro未找到；用户建议的papawan私人路径返回权限拒绝，未绕过访问控制。该对照不代表跨所有generator或FSI的完整包络。

| NUX − Standard, balanced prior | Difference | 95% independent source-file interval |
| --- | ---: | --- |
| Balanced accuracy | −0.03183594 | [−0.06562003,+0.00189188] |
| NLL | +0.01288127 | [−0.01874759,+0.04353697] |
| τ efficiency at frozen 1% cut | −0.01363636 | [−0.05821590,+0.02893250] |

这三个总体指标未确定差的方向，不能称等价。更细的pointwise结果中，τ argmax recall差−0.100000、CI[−0.205250,−0.009260]，NC AUROC差−0.041399、CI[−0.076612,−0.005786]；属于多项描述性对照，不是经过多重检验校正的独立发现。[Generator图](/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s6_2026-11-14/evaluation/figures/generator_comparison.pdf)和`generator_comparison.json`保留四prior、flavour/charm、低统计尾与undefined purity。

最初G4设置关闭trajectory storage，而`FillPythiaTree`依赖trajectory container，造成重模拟τ/charm tree空。已在隔离`R/g4_truth/`构建中，仅记录primary τ/charm及direct children的初始position/momentum，`AppendStep`不积累trajectory。未改变physics、seed、input、cuts；全3,512+5,888个响应的observations、targets和ideal tracker均逐值相同，原分数不需重算。证据为`{generator,transport}/truth_completion.json`及compact `decay_truth.parquet`。

Generator τ440/440顶点有效；Standard had/e/μ=135/47/38，NUX=150/30/40。Primary charm Standard125、NUX111。Transport τvertex有效2,046/2,048；run10083/chunk165/row950在nominal与helicity0中vertex validity false，hadronic direct-child分类仍保留。这两行未剔除、未称未衰变，具体vertex原因待查。Charm direct children包含interaction/ionization次级，因此e/μ出现率不是charm衰变分支比。FTFP_BERT+PY8DK、Pythia8.313与原secondary implementation不变；polarization代理、τ branch及固定selection计数分别记录在两个`*_decay.json`。

## 详细数据：内部健康、数值重放与迁移

Calibration/test各227个分块，对13个checkpoint逐块首batch采集内部激活，共5,902个model×block检查；全部记录healthy、sampled finite。MAE三seed合并两个split的`head_input_norm` RMS范围0.894109–0.907030，Head首Linear RMS3.307461–4.365007，随后normalization RMS0.960736–1.014098，activation RMS0.156809–0.394917；这些Head层采样零比例均0。此证据覆盖每块首batch，不冒充每个事件的全层遍历；每事件输出分数另已完整读回。未出现旧S3的常数输出或Head整体归零，健康检查不是单看finite checkpoint。

32,000事件nominal数值重放，三单seed最大概率差0.001973271/0.000617206/0.001115561，均低于冻结0.003容差。主ensemble四prior下argmax变化数0、两个τcut membership变化数0；BA、τ效率、FPR差0，NLL差约1.56–2.09×10⁻⁷。单seed计数差另保存。`audit/inference_health_readback.json`与`audit/replay_numerical_effects.json`给完整范围及实测计数。

MAE/Scratch各四prior的score-energy和class-energy-prediction矩阵齐全。S5 vertex+kinematics辅助能量输出全部finite/physical；722,221个可用truth pairs，矩阵范围外0。Vertex绝对残差中位数x/y/z=11.7913/13.7955/53.6016 mm，90%分位49.3156/73.0486/198.3113 mm。能量目标是S5 visible-primary proxy，不是incoming ν能量或实测calorimetric energy；forward-folding/unfolding未验证。[迁移图](/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s6_2026-11-14/evaluation/figures/energy_score_response.pdf)标明GeV、概率、row fraction与无τ的最高能区，矩阵和有效性见`evaluation/migration/`。

## 资源、失败恢复与产物

最终后端记账包含1,284条HTCondor实际ads、8份Kubeflow allocation，其中两次失败也计入。未materialize且已取消的factory行保留取消证据，不赋予虚构runtime；全部queue/history查询成功、无未解释缺行。9个Kubeflow提交中整卡94GB请求未调度即取消，allocation为0。

| Backend quantity | Final measured allocation |
| --- | ---: |
| HTCondor CPU allocation | 220.648333 core-hours |
| HTCondor actual user+system CPU | 93.384444 core-hours |
| HTCondor GPU allocation | 0.984444 device-hours |
| Kubeflow H100 47GB MIG allocation | 1.531389 instance-hours, includes upper bound |
| Kubeflow requested CPU allocation | 33.690556 core-hours, includes upper bound |

一个MIG instance小时不换称整张H100小时。资源统计属于S6准备/推理/对照/统计，不含此前S2–S5训练成本。GPU utilization fraction在MIG/RBAC接口下未测量，货币费用未测量；不从allocation小时猜利用率。

Calibration首个47GB Pod失败exit1，PodScheduled至主容器结束16秒；此前其终态以Kubernetes List保存，被记账读取器遗漏，已修复展开并按UID去重，最终成本包含该行。首test Pod的额外exec返回137，随后Job为BackoffLimitExceeded、Pod被GC，主容器精确termination与kill原因未保留，不能断言OOM。按实际PodScheduled至Job终态2026-11-14T14:49:55Z计2,960秒保守上界；六个claim在确认Pod消失/Job失败后才回收，两lane×一loader恢复完成。HTCondor1582744误用仅有数值policy的配置，在加载模型前失败；1582750以正确campaign恢复，未产生另一套科学分数。两个site Requirements循环不兼容的schedd实时排除，取消未运行factory后迁移到bigbird20。

最终画图HTCondor1582784 exit0、运行34秒、初始排队141秒。全部S6 Condor queue为空；Kubeflow `app=pinpoint-s6`的Pod/Job为空，已结束Job删除前的终态保留。`evaluation/resource_accounting.json`、`evaluation/backend_closure.json`及`runtime/`保留查询、失败上界、取消和恢复证据。

最终缓存清理净释放**28,572,275,170 bytes（28.572 GB）**，删除46个过时AFS提交目录。2,701份任务summary归档为6,420,824 bytes的JSONL.gz，日志归档10,692,399 bytes、2,375个members，均先读回再删除原目录；实测见`evaluation/cleanup.json`。完整raw/逐模型校准分数、merged catalog/truth/quality/decay、全部metrics/curves/bootstrap、10组英文PDF+PNG、冻结配置/source snapshots/G4可执行保留在R。Git只保存代码、合同、中文报告与[精简数值](../roadmap/results/s6_summary.json)、[60项CSV](../roadmap/results/s6_metrics.csv)、[完成证据映射](../roadmap/results/s6_scope_completion_audit.json)。完整`evaluation/final_summary.json`约7.8 MB，留EOS。

## 执行与验证

本轮统计/校准/truth geometry的11项单元验证通过；命令为：

```bash
which() { command -v "$@"; }
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest tests.test_s6_evaluation tests.test_s6_calibration tests.test_s6_truth_geometry
```

大批量转换、推理、重模拟和统计均通过HTCondor/Kubeflow；各任务完整命令、资源、source archive/commit、schedd选择和终态在`runtime/*.submission.json`及对应证据中。最终轻量汇总/清理命令：

```bash
R=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s6_2026-11-14
python -m analysis.flavour.s6.accounting collect --root "$R"
python -m analysis.flavour.s6.report --root "$R"
python -m analysis.flavour.s6.audit --root "$R"
python -m analysis.flavour.s6.cleanup --root "$R"
python -m analysis.flavour.s6.cleanup --root "$R" --apply
python -m analysis.flavour.s6.audit --root "$R"
```

清理只在全部消费者完成及后端终止后执行；已经完成的清理不重复运行。新增List终态解析按真实失败Pod验证16秒归入且只计一次；图表最后读回见`audit/final_figure_readback.json`。此外逐项人工对照04第7节/S6合同：空事件与cutflow、60项结果/完整曲线、400次区间/seed spread、物理条件/共同支持、14读出/8 transport、独立generator、τ/charm边界、迁移、10组图、资源与清理均有直接产物。真实timing/window、硬件systematic幅度、NuWro、flux/exposure、产额及significance标为未测量，不以文件存在替代这些测量。

## 阶段性结论

1. 修复架构与空响应处理后的模型完成真正独立、预先冻结的test评价。MAE相对同预算Scratch的BA/NLL/τ工作点有小幅正收益，绝对τ低背景效率仍低。
2. 对应控制预算下tracker增益明确；原生粒度的τ工作点收益未建立，不能推断原生读出不包含额外信息。辅助任务体现指标取舍，没有全指标一致改善。
3. Combined/noise等proxy影响达到可测水平；真实transport和Standard–NUX对照的统计仍有限，整体τ工作点差未确定方向。已测压力变化不等于真实硬件或跨全部generator的系统误差。
4. 低τ先验下当前选择纯度不足；未知物理flux/exposure时，不能将balanced benchmark换称实验物理产额或灵敏度。

## 下一步计划

- 在train/validation建立高occupancy细拓扑的独立工单：对照局部邻域编码、可变token预算和保留局部readout的多尺度路径，固定训练算力与标签，逐层监测梯度、尺度和小样本拟合。先检验压缩/优化原因，再决定扩模型或扩大训练语料。
- 分离τ衰变位置、可见长度、pixel首面投影、occupancy与tracker接受度；比较同物理支持的τ branch/charm背景，使用可测对象与方向残差，不把truth顶点作为输入。验收需同时改善固定背景工作点、概率质量和覆盖率。
- 新模型选择只用新的train/validation合同；最终确认使用新独立事件，保留本轮test为已公开参考。以后获得可读NuWro/其他GENIE配置、timing或hardware measurements时，再补对应输入与不确定度测量；不重做已接受样本认证。
