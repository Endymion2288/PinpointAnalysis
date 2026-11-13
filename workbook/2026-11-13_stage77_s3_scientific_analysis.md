# S3科学分析与最终结论

S3已完成。17项screen和8项新增seed确认共25个科学模型全部完成四轮训练、每项80,064 applied optimizer updates、零AMP skips，350/350项CPU checkpoint/optimizer/prediction读回通过。完整四分类评价及19项strict输出门槛通过，8项候选晋级决定均为false。推荐保留S2 ParT native作为reference backbone，同时保留1 mm控制；当前证据没有确立native优于1 mm。Calibration/test继续冻结，S4–S6未执行。

## 阶段目标

检验plane-aware局部编码、module context与late/joint/latent fusion能否改善`CC nu e / CC nu mu / CC nu tau / NC`四分类，并检验局部窗口是否改变native与1 mm表示差异。预定判据是完整四分类质量和资源合同；研究完成允许得到继续简单基线的结论。“shared 4 mm compression抹平native优势”作为待检验假设保留，本轮未得到因果确认。

## 工作内容

沿用用户接受的80,000 train、10,000 validation事件及原split、geometry、detector-proxy。Run10081/10082/10083/10084依次对应四类，validation每类2,500。硬件observation白名单构成模型输入，truth只用于标签及诊断分层；proxy使用现有参数。

实现独立的`analysis.flavour.s3`：同plane的16/64-pitch或4 mm窗口处理完整canonical blocks；每个module/type的4/8个learned queries读取局部块；显式module/type/连续z和pair relations形成context；late、joint和24-token latent fusion共享module读出接口。设计参考FASERCal的module attention和Perceiver结构原则；本实现没有执行其预训练，也没有复现其1 cm voxels及31.9M参数模型。ParT/ParticleNet为只读S2基线。参考文献与结构边界见[执行合同](../roadmap/09_s3_operations.md)。

17项screen覆盖窗口/表示2×2、fusion、module/pair geometry、pixel/bar双向模态、tracker、bar signal和module queries，固定seed42、四轮、batch4。按冻结规则选中task14（tracker候选）及task0（4 mm late），主native task5与1 mm task6也确认，各增加seed17/2026，合计8项。所有seed均计入，以full-validation NLL选checkpoint。四种结构分别形成三个训练seed的比较。

## 详细数据

### 完整四分类与统计范围

最终评价包含41个模型、22项单seed配对、9项三seed比较、1项2×2交互，pending=0。七个预定条件域比较全部完成，37个S3/ParT/ParticleNet模型在完整域加七个子域产生1,184条逐类工作点记录。主评价和条件配对均同时保留raw/common-support的34项指标：macro AP/AUROC/F1、balanced accuracy、NLL/Brier，四类AP/AUROC/precision/recall/F1及1%/0.1% FPR效率。完整混淆矩阵、能量/顶点/occupancy分层与tau衰变分支保存在原始评价中。

400次bootstrap共享128个accepted-family/chunk blocks（每类32个）。区间为给定训练模型的pointwise配对物理事件区间，未作多重比较修正，不包含训练随机性；三seed sample SD另列，不作为置信区间。三seed比较平均各seed的指标/配对差值，不构造概率ensemble。固定S2 energy/vertex common support含8,559个正权重事件，Neff=7,959.413647，target Z未匹配；用于粗相空间控制，不解释为flux重加权或generator独立验证。

以下所有均值基于seeds17/42/2026；除AP外其余列为均值，完整sample SD见[版本化数值表](../roadmap/results/s3_summary.json)。

| Model | Macro AP mean ± sample SD | Macro F1 mean | Balanced accuracy mean | NLL mean |
|---|---:|---:|---:|---:|
| ParT native | 0.456496 ± 0.005208 | 0.410335 | 0.413533 | 1.227834 |
| ParT 1 mm | 0.456559 ± 0.012097 | 0.413238 | 0.416567 | 1.227951 |
| ParticleNet native | 0.431786 ± 0.003102 | 0.386228 | 0.394800 | 1.243669 |
| ParticleNet 1 mm | 0.435660 ± 0.006379 | 0.394025 | 0.400000 | 1.242295 |
| 4 mm late | 0.350287 ± 0.008475 | 0.331080 | 0.342067 | 1.334393 |
| Fine/context/latent native | 0.252090 ± 0.003620 | 0.117562 | 0.249400 | 1.385946 |
| Fine/context/latent 1 mm | 0.252982 ± 0.005166 | 0.100000 | 0.250000 | 1.385976 |
| Fine/context/latent + tracker | 0.370623 ± 0.002252 | 0.361487 | 0.363833 | 1.320335 |

四种S3结构均未超过ParT或ParticleNet。正式配对结果如下，方向统一为候选减基线，NLL差值为正表示变差：

| Candidate − baseline | Macro AP delta [95% CI] | NLL delta [95% CI] |
|---|---:|---:|
| 4 mm late − ParT native | -0.106209 [-0.115541, -0.098287] | +0.106559 [+0.097738, +0.116623] |
| 4 mm late − ParticleNet native | -0.081499 [-0.089936, -0.073548] | +0.090724 [+0.082547, +0.100175] |
| Fine/context/latent native − ParT native | -0.204406 [-0.213984, -0.195871] | +0.158112 [+0.147317, +0.169775] |
| Fine/context/latent native − ParticleNet native | -0.179696 [-0.189245, -0.171513] | +0.142277 [+0.132780, +0.151864] |
| Fine/context/latent 1 mm − ParT native | -0.203514 [-0.213317, -0.195061] | +0.158142 [+0.147368, +0.169795] |
| Fine/context/latent 1 mm − ParticleNet native | -0.178803 [-0.188176, -0.170962] | +0.142307 [+0.132711, +0.151949] |
| Fine/context/latent + tracker − ParT native | -0.085873 [-0.094278, -0.077817] | +0.092501 [+0.083788, +0.102169] |
| Fine/context/latent + tracker − ParticleNet native | -0.061163 [-0.070176, -0.052587] | +0.076666 [+0.068154, +0.086011] |

Tracker候选相对ParT的common-support macro AP差为−0.075652，95% CI [−0.084450, −0.066237]，粗相空间控制后仍为负。所有8项确认比较均未满足质量晋级或资源晋级规则；整卡/MIG subtype不相同的成本不能作为吞吐效率等价比较。

### 逐类取舍

下表先对每个seed独立argmax计算precision/recall/F1，再取三个seed均值；AP同样为逐seed均值。

| Model | Class | AP | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| ParT native | e | 0.603107 | 0.564839 | 0.495067 | 0.527273 |
| ParT native | mu | 0.450864 | 0.466709 | 0.333067 | 0.380801 |
| ParT native | tau | 0.354573 | 0.315802 | 0.553600 | 0.398916 |
| ParT native | NC | 0.417441 | 0.438547 | 0.272400 | 0.334352 |
| 4 mm late | e | 0.457055 | 0.456214 | 0.380800 | 0.412630 |
| 4 mm late | mu | 0.341811 | 0.315241 | 0.551200 | 0.400317 |
| 4 mm late | tau | 0.286026 | 0.292451 | 0.208800 | 0.242830 |
| 4 mm late | NC | 0.316254 | 0.329249 | 0.227467 | 0.268541 |
| Fine/context/latent + tracker | e | 0.463373 | 0.428905 | 0.410133 | 0.416344 |
| Fine/context/latent + tracker | mu | 0.372400 | 0.367512 | 0.398133 | 0.380915 |
| Fine/context/latent + tracker | tau | 0.291391 | 0.301918 | 0.256000 | 0.276573 |
| Fine/context/latent + tracker | NC | 0.355326 | 0.356653 | 0.391067 | 0.372115 |

Tracker相对ParT的四类AP差均小于−0.01，四个95%事件区间均在零以下；NC argmax recall/F1却提高，NC F1差+0.037763，CI [+0.016925,+0.054399]。Muon F1均值接近、recall上升且precision下降。4 mm late的mu recall/F1也较高。这说明argmax决策边界发生取舍，不能表述成所有逐类指标均下降，更不能把argmax收益等同于相同purity下的收益。

### 全部screen对照

以下为seed42的best-NLL checkpoint，对相同10,000事件评价，含空响应。

| Frozen experiment | Macro AP | Macro F1 | Balanced accuracy | NLL |
|---|---:|---:|---:|---:|
| s3_a4_context_latent_1mm_s42_e4 | 0.321517 | 0.314126 | 0.328900 | 1.356914 |
| s3_a4_context_latent_s42_e4 | 0.261014 | 0.167641 | 0.267500 | 1.384043 |
| s3_a4_late_s42_e4 | 0.358741 | 0.334573 | 0.346000 | 1.330101 |
| s3_f16_context_joint_s42_e4 | 0.332317 | 0.317753 | 0.330700 | 1.352187 |
| s3_f16_context_latent_1mm_s42_e4 | 0.250000 | 0.100000 | 0.250000 | 1.386298 |
| s3_f16_context_latent_bar_s42_e4 | 0.251354 | 0.125677 | 0.253900 | 1.386008 |
| s3_f16_context_latent_nogeometry_s42_e4 | 0.250000 | 0.100000 | 0.250000 | 1.386298 |
| s3_f16_context_latent_nopair_s42_e4 | 0.250000 | 0.100000 | 0.250000 | 1.386298 |
| s3_f16_context_latent_nosignal_s42_e4 | 0.251846 | 0.132109 | 0.253500 | 1.385919 |
| s3_f16_context_latent_pixel_s42_e4 | 0.250000 | 0.100000 | 0.250000 | 1.386298 |
| s3_f16_context_latent_q8_s42_e4 | 0.280445 | 0.240556 | 0.286500 | 1.375602 |
| s3_f16_context_latent_s42_e4 | 0.250000 | 0.100000 | 0.250000 | 1.386298 |
| s3_f16_context_latent_tracker_s42_e4 | 0.368196 | 0.357890 | 0.359600 | 1.324486 |
| s3_f16_joint_s42_e4 | 0.346750 | 0.327627 | 0.337000 | 1.340082 |
| s3_f16_late_s42_e4 | 0.338682 | 0.317650 | 0.335000 | 1.342968 |
| s3_f16_latent_s42_e4 | 0.248288 | 0.100000 | 0.250000 | 1.386325 |
| s3_f64_context_latent_s42_e4 | 0.284258 | 0.248549 | 0.285900 | 1.374839 |
4 mm late→16-pitch late的macro AP差−0.020059，CI [−0.025400,−0.015138]；fine joint→context joint为−0.014433，[−0.017993,−0.011131]；fine context joint→context latent为−0.082317，[−0.089567,−0.076321]。这些比较限定在当前实现、容量与四轮配方，不推及架构类别的一般优劣。

### 分辨率、压缩边界与优化诊断

主native与1 mm的三seed macro AP差−0.000892，CI [−0.001297,−0.000465]，逐seed差为[−0.002677,0,0]；NLL差−0.000030，CI [−0.000189,+0.000111]。两组seed42/2026在完整validation均只有1组概率向量，seed17分别有438/485组，属于弱学习。两组seed17均由NLL选中epoch1；native该轮AP=0.256270，最后一轮AP=0.253596，不能用最后一轮替代正式结果；1 mm选中轮AP=0.258947。1 mm seed17虽有485组分数，argmax仍全部预测NC。

Screen的fine/native与fine/1 mm均为常数AP0.25；4 mm窗口native/1 mm AP分别0.261014083/0.321516517。交互`(fine_native−fine_1mm)−(4mm_native−4mm_1mm)`为+0.060502，CI [+0.054428,+0.067035]。它包含fine单元的优化失败，不能据此认定细窗口恢复了native信息，或4 mm压缩抹平了native优势。

在同8个validation事件上用CPU FP32/eval完成一次反向、无optimizer更新。以下是新增seed的状态诊断，全部参数finite：

| State | Epoch | Distinct vectors / 8 | Head GELU exact-zero fraction | Input gradient squared norm | Module-read gradient squared norm |
|---|---:|---:|---:|---:|---:|
| Native seed17 best | 1 | 6 | 0.989583313 | 1.106993e-10 | 8.247433e-5 |
| Native seed17 last | 4 | 2 | 0.998697937 | 1.052449e-10 | 2.480705e-6 |
| Native seed2026 best/last | 4 | 1 | 1.000000000 | 8.942770e-38 | 9.340110e-32 |
| 1 mm seed17 best | 1 | 2 | 0.998697937 | 1.128925e-7 | 4.892828e-5 |
| 1 mm seed17 last | 4 | 2 | 0.999348938 | 4.140766e-19 | 1.077761e-13 |
| 1 mm seed2026 best/last | 4 | 1 | 1.000000000 | 0 | 0 |

Native seed2026仍有极小非零梯度，与seed42此前测得的严格零梯度区分。Seed17的局部诊断不支持完全切断。Late与tracker的新增seed在同8事件各有8组分数、上游梯度非零，仍不以梯度范数大小排名模型。证据在`diagnostic_confirmation_{primary,coarse}_summary.json`及late/tracker逐state JSON；历史机制见[阶段65](2026-11-12_stage65_s3_prediction_collapse_diagnostic.md)。CPU FP32同样出现弱/零梯度，BF16量化不能独自解释这一现象。四轮完成、参数finite和零AMP skips证明程序执行，不能证明有效学习。

Token census在同一最高occupancy train事件测得：36,547个pixel+bar native nodes经4 mm/16-pitch窗口形成2,641/14,597个local blocks，之后都压成124个module tokens、再成24个latent tokens。8 queries增至248 module tokens，global仍24。三个预定occupancy ranks×17配置共51条计数见[阶段75](2026-11-13_stage75_s3_token_bottlenecks.md)。这些计数定位了压缩边界；信息损失对分类的独立贡献仍未测量。

### 模态与条件域

Tracker相对主模型的seed42 AP差+0.118196，CI [+0.109862,+0.127903]。两者参数均749,868，主模型已塌缩；该差值同时包含观测信息和优化轨迹变化。Late参数373,060，ParT151,228；tracker候选相对late还同时改变窗口/context/fusion，不能当作单独tracker消融。塌缩模型间的geometry/bar零差也不证明这些信息无用。

七个预定域的三seed配对如下：

| Domain | Events | Tracker candidate − ParT native macro AP [95% CI] |
|---|---:|---:|
| pixel_centered | 885 | -0.102804 [-0.125521, -0.075881] |
| large_scintillator | 4494 | -0.119003 [-0.129816, -0.105696] |
| detector_active | 9718 | -0.088423 [-0.096898, -0.080073] |
| all_empty | 282 | +0.000000 [+0.000000, +0.000000] |
| pixel_ge_1000 | 1201 | -0.071196 [-0.090151, -0.049649] |
| tracker_active | 5630 | -0.080316 [-0.090314, -0.069257] |
| tracker_empty | 4370 | -0.096414 [-0.107379, -0.085067] |

有响应的六个域均未显示tracker候选优于ParT；全空域AP零差来自缺少响应，不证明硬件信息等效。各域保留全部事件及自身class priors，域间AP不能直接作为同事件消融；例如高占用域e prior=43.7968%，全空域tau prior=31.9149%。详细逐类分母见[阶段78](2026-11-13_stage78_s3_domain_denominators.md)。

Tracker输入是用户接受的hit/station proxy，station index/3不是毫米飞行距离；现有输入没有time或拟合q/p。本阶段未测量真实轨迹拟合性能及硬件标定。

### 相同工作点、尾部与tau分支

以下为完整validation、seed42的经验工作点，各类信号2,500、背景7,500；计数为实际选中事件。

| Model, seed42 | Class | 1% FPR efficiency (signal/background) | 0.1% FPR efficiency (signal/background) | Purity ≥ 0.5 efficiency (signal/background) |
|---|---|---:|---:|---:|
| ParT native | e | 0.1832 (458/75) | 0.0476 (119/7) | 0.5752 (1438/1438) |
| ParT native | mu | 0.0984 (246/75) | 0.0196 (49/7) | 0.3172 (793/793) |
| ParT native | tau | 0.0372 (93/75) | 0.0068 (17/7) | 0.0688 (172/172) |
| ParT native | NC | 0.0704 (176/75) | 0.0228 (57/7) | 0.2584 (646/646) |
| Tracker candidate | e | 0.1064 (266/75) | 0.0348 (87/7) | 0.3216 (804/804) |
| Tracker candidate | mu | 0.0408 (102/75) | 0.0076 (19/7) | 0.1392 (348/348) |
| Tracker candidate | tau | 0.0176 (44/75) | 0.0004 (1/7) | Unreachable |
| Tracker candidate | NC | 0.0364 (91/75) | 0.0064 (16/7) | 0.0640 (160/160) |

1%工作点每项实际75个背景；三seed tracker−ParT效率差[e,mu,tau,NC]=[−0.073467,−0.053467,−0.020133,−0.039467]，四个95%事件区间均为负。0.1%点实际背景7个，所有域该点都低于预定25背景充分性标记；pixel-centered/高占用/全空子域最多允许0背景。因此本轮不据此宣称尾部精度已稳定。工作点在validation上经验扫描，未冻结部署阈值；purity仅对应benchmark prior。完整purity≥0.8/0.9、阈值、子域计数及全域阈值迁移在conditional summary，曲线见40页PDF。

Validation tau分支为electron445、muon395、hadronic1660，unknown/ambiguous均0。各分支背景均用全部7,500个非tau事件，排除其他tau分支；分支PR受不同prior影响。ParT seed42在inclusive 1%背景阈值的分支信号计数为7/4/82，合计93，分支效率分别1.5730%/1.0127%/4.9398%。这些分支补充诊断与完整四类评价并列，不单独授予backbone晋级。

### 执行、资源及产物

训练通过HTCondor/Kubeflow执行。确认批次以一个`nvidia.com/mig-3g.47gb` reservation运行四个独立进程并自动补入后续任务，8项子进程exit0。Job `pinpoint-s3-confirm-mig47-parallel-v1`在2026-11-13 02:49:33 UTC完成，Complete condition时间02:49:34 UTC；当前S3 Pod数量0。共享reservation按活跃进程等分，启动/退出空档等分，组织成本只累计一次；显存占用不替代GPU compute utilization。

S3 accounting为31.057126064 device-hours，其中HTCondor15.903611111、Kubeflow15.153514953；Kubeflow含MIG6.884444444及已删除整卡尝试的8.269070509小时生命周期保守上界。另加初期interactive T4保守上界0.181142613，组织上界**31.238268678 device-hours**，预算120，余额下界88.761731322。S2的92.73055556另列，不计入S3。整卡与MIG小时分别记录，不视作吞吐等价。

确认reservation按Scheduled事件至Job Complete condition的14,259秒计保守上界，即3.960833333 MIG device-hours；终态Pod已被清理，Pod phase/exit_code未观测，不能写成Pod Failed。共享成本派生8条分摊行，仍只计一份reservation。审计已补正旧缓存的推定终态和上界标签，成本数值不变；18条上界行对应7个Job记录（含2个零GPU小时记录），不代表18次训练失败。无缺失计费历史记录。精确GPU busy time及初次T4 context分配时长未测量。

结果根`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_2026-11-12`保留模型、图表和证据；仓库版本化[最终数值表](../roadmap/results/s3_summary.json)及配置，不纳入大型产物。主要产物：

- `evaluation_final/summary.json`：41模型、所有配对/交互/分层与seed spread。
- `evaluation_final/conditional/summary.json`和`sampled_curves.parquet`：工作点、阈值、分母和条件配对。
- `four_class_diagnostics.pdf` 41页、`four_class_resource_tradeoff.pdf` 4页；资源图按GPU subtype分开。
- `conditional/four_class_domain_curves.pdf` 40页。
- `event_displays/{primary,tracker}/native_error_groups.pdf`实际分别8/20页（目录使用完整模型名）；按存在的confusion pairs及四个固定random references选图。主模型只预测NC，不能要求出现不存在的预测组。
- `final_acceptance.json`、`campaign_accounting.json`与`review/`：严格验收、资源和人工复核证据。

已人工检查基线、常数模型、tracker诊断页，条件曲线首/末页，两个native display首/末页及MIG资源页；英文坐标、图例、物理单位可读。常数预测PR改为阶梯显示，recall>0处precision=0.25；原线性连接会造成视觉误导，AP数值未受影响。Native图中的high-confidence是组内分数排序标签；主模型约0.25的分数没有绝对高置信含义。事件示例用于形态诊断，不作为总体物理结论的统计证据。

### 复现与验证

正式最终流程由`analysis.flavour.s3.finalize`完成。收尾只刷新accounting、strict gate、版本化摘要及已有曲线的绘制，没有重复训练或bootstrap。环境与验收命令见[阶段80最终审计](2026-11-13_stage80_s3_final_acceptance.md)。19项strict检查同时覆盖冻结配置、350项读回、22/9/1完整配对/交互、raw/common-support各34项与400有效replicates、七域、终态资源和预算。

## 阶段性结论

事实：当前冻结实现与四轮预算下，plane-aware fine window、module context和latent fusion未取得达到晋级标准的完整四分类收益。最好的确认S3候选tracker的AP为0.370623±0.002252，低于ParT0.456496±0.005208；四类AP和1% FPR效率的配对事件区间支持这一差距，NC argmax F1收益同时保留。主native/1 mm多seed出现弱学习或常数输出，分辨率结论受优化失败干扰。

未决假设：modality/context拼接尺度与head大负输入可能导致GELU饱和，module/global固定query压缩可能丢失信息。现有诊断支持继续检验这些机制，但尚未分离其因果贡献；不据此认定native无用、1 mm等效或4 mm已抹平优势。

方法决定：继续使用S2 ParT native reference及1 mm控制，S3研究完成、新模型不晋级。主模型更多参数未带来质量收益，混合GPU设备的资源分摊也不满足效率晋级证据。

## 下一步计划

本轮止于S3。后续若继续研究，先独立冻结优化稳定性实验，再启动预训练阶段：固定相同输入/seed/预算，逐项对照head normalization、残差缩放、GELU/SiLU及学习率尺度，保存激活/梯度、unique-score与NLL轨迹；成功标准是多个seed持续有效学习并相对原配方改善完整四分类。优化稳定后再控制容量与query预算比较4 mm/16-pitch及native/1 mm，检验module/global压缩信息损失；新增tracker信息也在稳定模型内同参数对照。具体预算和晋级标准在新实验启动前冻结。本轮未提交这些训练，S4–S6未执行。
