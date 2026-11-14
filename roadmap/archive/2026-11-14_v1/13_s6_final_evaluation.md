# S6：校准、条件变体与冻结独立test

本阶段目标来自`04_execution.md`第7节及`03_models_and_experiments.md`第7节。样本、split和detector-proxy基值沿用用户接受状态；不重新认证样本或要求真实硬件标定。研究完成需同时交付独立test、概率校准、条件效应、generator对照、物理选择与资源证据，不能以单一绿色检查替代。

## 输入、候选与冻结顺序

使用原inventory中全部usable/consistent calibration和test，不重新分split。Calibration722,272事件，test724,441事件；准确各类数在`results/flavour/s6_2026-11-14/plan.json`。原有10k validation用于候选/工作点规则，calibration只拟合概率转换，test只执行已冻结流程。额外变体与原事件保持家族对应。

`configs/s6_candidates.json`在任何calibration/test模型推理前固定13个checkpoint：100k-label MAE3与Scratch各三seed；S5 direct4/vertex+kinematics/six的seed42；S3 native/1mm/tracker三项seed42；额外channel-Rel5 seed42。主方法预定为100k MAE3三seed raw probability均值，Scratch均值为pretraining控制。单seed辅助及跨阶段不同预算分别报告，不解释成同预算总体架构比较。

先准备calibration、核对checkpoint真实回放和推理接口，再拟合单参数temperature。Ensemble先平均raw四类probability，再拟合一个ensemble temperature。校准目标为四类各0.25的benchmark prior，用calibration class权重实现；生产sample mixture不冒充物理flux。拟合temperature范围[0.05,20]预先固定，报告端点与fit数值状态。

校准与prior变换冻结后，在既有validation选择每类1%和0.1%背景接受率的阈值；四类argmax与带reject的多类selection分开。多阈值同时通过时选择`log(p_c/threshold_c)`最大者；全部未通过则reject。最终`analysis_freeze.json`保存checkpoint身份、概率转换、priors、阈值、条件变体、事件范围和评价定义，然后才开放test response/inference。失败分块可按相同冻结配置恢复，不借恢复重选方法或阈值。

## 条件效应与generator范围

Nominal和digitizer变体按相同事件运行，保留单项、组合与未参与训练augmentation的条件域。优先覆盖gain、threshold、noise、inefficiency/dead channels、alignment及tracker分辨率；未知幅度明确标为detector-proxy压力范围，不能称作测得的真实硬件systematic。Time/window当前无可测输入，报告输入不支持项；field/material需实际transport/resimulation，不能用平移hit冒充。可复用匹配现成G4构建；需要增加宏接口或输出时才在results下隔离构建。

Generator对照需先读schema、能区/target/flavour覆盖，寻找GENIE tune/FSI/hadronization或可用NuWro。按incoming flavour、energy、vertex与target共同支持匹配，flavour和charm分别报告；generator不同事件不能假装paired transport。τ/charm branching、polarization和secondary implementation的已测与缺失项分别列出。条件包络与MC统计区间分列，幅度没有硬件约束时只报告压力变化。

## 必须交付的评价

完整test保存每事件四类raw/calibrated分数，输出confusion counts/row normalization、完整ROC/PR与明确prevalence、macro/class AP/AUROC/F1/BA、NLL/Brier、固定阈值效率/纯度、reject率及score/energy migration。主表有all-event、目标域、active、selected cutflow，空响应计入总体。报告energy、vertex、occupancy、containment、ν/anti-ν、τ branch、charm、tracker acceptance与可重建quality分层。

Reliability图/ECE明确bins/counts及有限统计；temperature校准没有自动保证条件域校准。400次physical-family/chunk block bootstrap给事件区间，三训练seed spread另报；weighted结果报告Neff和权重分布。低FPR背景通过数不足25时标出；零计数报告描述上限，不以Neff代替exact binomial分母。Test曲线用于描述，不从test重选工作点。

没有可靠exposure/flux时，以固定priors扫描效率/纯度，物理luminosity产额和significance标未测量。Prior correction明确label-shift假设，不在test拟合未知physical mixture。S2/S3/S4/S5的原生分辨率、模态与pretraining贡献由同split对照表追溯；最终test中的预定controls另给。

## 方法与计算依据

- Guo et al., ICML2017，temperature scaling：[原文](https://proceedings.mlr.press/v70/guo17a.html)。只在calibration拟合，参数少，适合冻结test。
- Lipton et al., ICML2018，label shift：[原文](https://arxiv.org/abs/1802.03916)。Prior correction仅在条件响应稳定的假设下解释，本轮scenario不估计真实flux。
- Kumar et al., NeurIPS2019，Verified Uncertainty Calibration：[原文](https://arxiv.org/abs/1909.10155)。报告bin counts和有限样本误差，不以较小ECE自动证明全面校准。
- Ovadia et al., NeurIPS2019，dataset shift uncertainty：[原文](https://arxiv.org/abs/1906.02530)。Nominal校准与shift下可靠性分别评价，seed ensemble的价值由数据测量。
- FASERCal的matched GENIE/NuWro flavour/charm差异沿用`02_literature.md`，不能直接移植其数值。

网络页面和标题记录在EOS `literature/sources.json`。CERN flavor按[官方文档](https://ml.docs.cern.ch/)申请，H10047GB MIG与其他可用卡分别计时；复用已记录终态避免凭据过期阻塞记账。HTCondor提交前实时选兼容schedd；AFS仅放小脚本，数据、图、日志和模型均写EOS。批处理开始前做小样本验证，按CPU预算限制解压和BLAS线程；最终删除临时native/ideal caches，保留概率、紧凑统计及复现输入定义。

## 完成结果与证据

S6已完成；[最终中文报告](../../../workbook/2026-11-14_stage236_s6_final_results.md)给样本、冻结顺序、完整主表、四prior、工作点原始计数、paired控制、物理条件、失败恢复与成本。[精简JSON](../../results/s6_summary.json)、[60项模型/prior CSV](../../results/s6_metrics.csv)和[原要求映射](../../results/s6_scope_completion_audit.json)保存在Git；大数据与图留在`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s6_2026-11-14`。

2026-11-14T13:50:23.738039Z冻结后开放全部724,441 test；calibration722,272，15个probability mappings、60项结果、400次物理块bootstrap齐全。主MAE3 ensemble的balanced accuracy0.435225878、CI95[0.433704723,0.436685364]，相对同预算Scratch的差+0.003353622、CI[+0.002507538,+0.004166165]。冻结validation 1%背景阈值下τ效率0.035556301，test实际weighted FPR0.012080097，balanced τ纯度0.495236618；τprior降至1%/0.1%时纯度仅0.028099373/0.002867199。Test未重选温度、priors、cuts、subset或模型，旧故障S3不参与结论。

直接科学交付包括14个paired readout条件（32,000×14响应）、8个实际G4 transport条件（5,888响应）、GENIE3.04/Pythia6 Standard–NUX的3,512独立hard scatters、9模型共同energy/vertex/exact-target支持、63个非空物理strata及score/energy migration。τ/charm compact truth重放的9,400个响应逐值相同；13个checkpoint在calibration/test各227块的首batch内部采样均healthy/finite。Nominal数值重放对主ensemble的argmax与两个τcuts均无计数变化。

方法层面的结果为：同预算MAE有小幅收益，tracker总体收益明确，原生粒度的BA/τ工作点收益当前未确认；高occupancy与pixel-centered域的低τ效率需在下一轮train/validation分离原因。Combined/noise/gain等proxy可明显改变性能；transport与generator整体τ效率差统计区间较宽，不能称等价或泛化已充分证明。

资源按实际终态、失败和初始化计入：HTCondor220.648333 CPU allocation core-hours、0.984444 GPU device-hours；Kubeflow47GB MIG为1.531389 instance-hours（含缺失Pod终态的一项保守上界），requested CPU33.690556 core-hours。全部queue/Pod/Job已关闭，28.572 GB临时产物清理、46个旧AFS提交目录删除；2,701份任务summary及日志已归档读回。GPU utilization fraction因MIG/RBAC接口不可测，标为未测量。

真实硬件systematic幅度、timing/window、NuWro、physical flux/exposure、luminosity产额、significance及unfolding仍为未测量。用户接受的proxy基值继续使用；这些输入边界不触发重复认证。后续新模型需新的选择合同与独立确认数据，本轮test保持为公开冻结结果。
