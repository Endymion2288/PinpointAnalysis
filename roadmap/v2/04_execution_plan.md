# 下一轮执行路线、实验矩阵与工单

**R0、R1已完成，R2首批六模型正在执行。**R1完整交付41模型、133项配对、21项三seed平均配对、11域和55条压力路径；452项验收全部通过，独立nominal/core-interval/stress检查分别通过2,557/2,512/5,666项。最终数值、统计口径及成本见[stage274](../../workbook/2026-11-15_stage274_r1_final_results.md)和[r1_results.json](r1_results.json)。Cal-A/Cut-B/Confirm继续封存。

本轮B1 ResNet最强：T3平均BA3为56.5633%，T4 conditional为56.3044%，τ效率在1% FPR下为5.0200%。相对B4的τ效率增益2.423 pp [1.941,2.977]满足预定τ研究门槛；T3相对HGB的1.973333 pp低于2 pp实用门槛，相对B4则未满足μ recall非劣。完整交付不等于全部科学门槛通过。

三类化没有稳定改善共同任务：B1/B2等时compute T3−T4 conditional的BA3分别为+.064 pp [−.179,+.302]和−.313 pp [−.514,−.113]，NLL3均更差，μ改善伴随e/NC损失。所有seed42主网络四pass末轮NLL仍下降。下一步先以B1的fresh D0/D1/D2分开检查训练配方、unique数据与重复曝光；原生B2作为问题驱动的第二族，R3再做表示单因素。

R1已记录终态batch成本45.708889异构device-hours、307.181111 CPU core-hours，超过18 h规划；所有allocation终态，历史SSL和交互下界单列。后续按实际成本分批选最小矩阵，不直接照搬原80 h总规划的余下60 h实验。资源继续以尽快开始、兼容即可为原则，接受不同GPU、共享设备和MIG。

本文件与[experiment_plan.json](experiment_plan.json)记录研究合同及状态。W01–W06/W08已完成，W07为R3待实现；R1入口为`analysis.flavour.v2.r1`，规格JSON不能直接传给旧CLI。原始审查见stage237，R0见stage238，R1中期过程见stages239–273，最终状态以stage274为准；不修改历史报告形成时的结果。

## 1. 分阶段依赖与交付

| Stage | Estimated window | Main experiment | Required deliverable / gate |
| --- | --- | --- | --- |
| R0（完成） | Completed 2026-11-14 | 使用记录、v2 manifests、loader/transform/profile、实际架构元数据 | 73项最终读回与81项测试通过；Confirm保持封存 |
| R1（完成） | Completed 2026-11-15 | B0–B4 T3/T4、三seed扩展、同计算及合作组域 | 41模型、133配对、21平均配对、55压力路径；452项验收通过 |
| R2 | Week 2–3 | unique data / passes / capacity 小矩阵 | 明确仍在改善、过拟合或资源受限；选择收敛参考配方 |
| R3 | Week 3–5 | G1a/G1b/G1c，最佳结构的 native/coarse | 几何关系/压缩单因素收益；三 seed 前两名确认；梯度及内部健康 |
| R4 | Week 4–6 | Tracking/vertex/association，raw/object 融合 | 对象 fake–efficiency–coverage 与全事件分类净增益 |
| R5 | Week 5–7 | T4 / T3T / T4D，τ specialists / charm negatives | 固定 FPR 的 inclusive/branch 结果、三类非劣及四类投入决策 |
| R6 | Week 6–8 | 定向预训练、同总计算监督控制、条件鲁棒训练 | 胜过强监督的实际效果或数据效率/鲁棒性优势 |
| R7 | Week 8–10 | 候选冻结、Cal-A、Cut-B、新 Confirm 与系统对照 | 新独立确认、统计/物理结果、资源与可复现交付 |

周数为单人研究排期估计，尚未测量；数据准备与训练可按依赖分批推进。R1 的三类交付单独完成，R4/R5 无需阻塞合作组三类比较。没有收益的结构在确认学习有效后记录其负结果，不必把全部候选扩成完整大矩阵。

## 2. R0：使后续比较具体可执行

### R0.1 使用记录与 manifests

已复用accepted inventory、原train/validation子集、SSL/Rel、probe、方向与对象训练、calibration/test manifests，追溯并排除1,560,122个历史使用family。Train100k/300k/1M、Tune40k、Cal-A40k、Cut-B160k、Confirm200k已按[数据合同](02_tasks_and_evaluation.md)平衡冻结。新Confirm只公开family身份；row/label封存，R0后续profile与最终读回未打开其payload。未使用family足以满足分配，不需要新生成样本才能开展R1。

产物为 `manifests/usage_ledger.parquet`、`partitions.json`、每个 split 的 compact row/family manifest 与 counts。原始 detector responses 按需转换，避免把 7.24M 事件全部展开后再决定训练规模。具体 source-family 认证继续沿用用户接受状态。

### R0.2 实际模型元数据

从实际 transform/config/model 生成：plane window 的 pixel 数与 mm 尺寸、blocks/region 规则、显式 cross-plane layers、latent/query 数、是否包含 tracker、参数量和输入列。修正未来 manifests 的过时 `4mm`/`cross_plane_message_passing` 文本，不改写旧科学结果。

R0实际导出：897,992参数，16×16 pixel窗口为0.3328×0.3648 mm；2个region queries、64个latents、2个global latent attention层、0个显式geometry context层。latent可以混合不同plane信息，当前结构没有显式跨plane几何关系层。完整白名单位于`representation_metadata.json`。

### R0.3 数据效率

取 Train 内低/中/高 occupancy 的固定 256 个真实事件和一份 10k 流水样本；前者做交互 microbenchmark，后者通过 HTCondor 做完整 loader profile。复用已验证的局部缓存/等价前端方案，分别测 EOS staging、decode、tokenization、collate、transfer、compute、validation。

默认比较 2/4 loader workers、bounded scratch cache、packed/bucketed batch 和观测 transform cache；CPU 线程与作业申请一致。减少同一 ROOT/Parquet 多次解码和每 epoch 重复排序；cache key 含 geometry/readout/representation revision。缓存是临时执行资产，最终仅保留能恢复的配置与小型核对结果。

进入训练前，新表示与原观察量在事件键、总响应、mask 和通道映射上通过同事件检查。旧窗口和新边界消息表示的差异是模型实验因素，不要求它们 logits 相同。性能门槛以可重复 profile 为准；建议把记录的 data-wait/总时间降到 20% 以下，达不到时指出具体瓶颈，仍可在剩余预算内推进必要训练。

R0已在10k真实Train事件上完成native/packed全部collated tensors逐位等价、256-event原frontend数组核对及GPU数值读回；32-event真实拟合在64updates后NLL=0.000418、accuracy=100%，分支梯度和参数更新通过。该R0拟合检验优化路径；后续R1泛化比较已完成，见stage274。

同一A100/eager下，native2/4 workers热态吞吐69.551/70.301 events/s，aggregate RSS6.515/9.820 GiB；分桶后即便使用packed也未超过native。另一H100/compiled下，native2与packed+bucket2为173.892/188.144 events/s，steady wait20.880%/10.801%。后者是cache与sampler的组合比较；两张卡的数据不构成同设备编译收益测量。

R1短训练默认native、2 workers、共享local staging、bounded 2-shard decoded LRU；compiled复用S3修复环境。packed热态约快8.20%，但10k构建需53.964 s，每pass节省4.356 s，约13个热态pass才摊平。本估计尚未测量独立重启的总成本；同表示多训练复用时再选择cache。局部32-slot利用率仅8.83%，inference的steady wait为22.89%或35.84%，后续单独优化local执行、collate与持久validation loader。各新架构仍须先做真实小样本拟合与计时。

## 3. R1：最小强基线矩阵

| Experiment group | Task | Input | Initial comparison |
| --- | --- | --- | --- |
| B0 | T3 / T4 | 同一 accepted observations 的 summary/profile | HGB 为主，小 MLP 可替代 |
| B1 | T3 / T4 | bar/pixel 双投影 + tracker set branch | 合作组式 ResNet；CNN 作为成本参照 |
| B2 | T3 / T4 | native pixel/bar + tracker set branch | Fixed-graph ParticleNet |
| B3 | T3 / T4 | native pixel/bar + tracker set branch | Geometry ParT |
| B4 | T3 / T4 | native pixel/bar/tracker | Revision 3 Scratch |
| B4-MAE reference | T4 | 与 B4 相同 | 既有 MAE3 初始化，单列 SSL 成本 |

神经网络架构排名首先比较Scratch；B4-MAE是初始化控制。初始seed42，使用相同100k四类Train pool及其非τ子集、同一Tune。每族使用相同LR候选数与screen规则，在各族已获allocation内比较；不同设备下不构成跨族FLOP等效HPO。先两个学习率候选，再延长最优候选；AdamW、weight decay、有效batch和schedule均记录。

2026-11-14执行更新：按用户要求优先使用可立即启动的不同GPU型号、整卡与MIG，Condor与Kubeflow并用，不再为统一型号等待。相同族的任务/学习率与同计算控制尽量在同一allocation执行；共同曝光比较仍使用完整相同非τ事件。跨设备小时不是FLOP等效量，按型号/分区与实际精度分别报告，配置和权重不按设备混成一个虚假的同计算结论。

训练阶段不以固定四轮作为所有模型的终点。短 screen 给每族相同设备上限和 validation 频率；延长训练到 R2 的收敛合同。HGB 的 boosting 预算按 CPU 时间和复杂度单列。

只对最强两个神经网络与共同基线扩 seeds17/42/2026。合作组配置缺少 run/完整 score/权重时，先在本批数据实现描述清楚的同构基线；相同事件列表和 exact cuts 可用后补做严格复现，不阻塞自身模型改进。报告“同构对照”与“原研究复现”的完成状态。

R1 交付包含全域/pixel-centered/large 三类表、三类训练与四类条件三类配对、τ 污染、calibration 前 raw score 分布和资源 Pareto；原生/图像的贡献与是否有 tracker 分开评价。Tune 内固定小型 noise/combined 开发对照随名义结果一起给出，沿用已接受的 proxy 幅度；它们用于研究已知弱点，R6/R7 另保留未见 pattern。不得以简单去除孤立 hits 的方式牺牲短轨而不报告其效率。

最终accept从冻结排名重建完整矩阵，精确要求当前41模型、133配对、21项三seed平均配对及每模型11域；验证身份、配对唯一性及三个seed名单，不能以内部一致的中期子集代替完整交付。实际24模型子集负对照和26项相关tests通过，见[stage267](../../workbook/2026-11-15_stage267_r1_completion_matrix_acceptance.md)。最终完整accept已实际通过452项；独立核心区间复核已增加到七项BA/NLL/recall，11套英文PDF/PNG已目视，全部allocation终态。

## 4. R2：用小矩阵确定训练规模

执行状态（2026-11-15）：300k Train+40k Tune身份及PE守恒核对通过，D0/D1/D2共六份精确sampler preflight已冻结。BF16三/四类真实拟合均100%，中途恢复与连续训练参数差值为0。Kubeflow H100 MIG 3g.47gb最多四并行执行D0/D1，HTCondor H100 MIG 2g.24gb双并行执行D2；完整批次科学结果尚未产出。配置、实算U和后续验收见[r2_execution.json](r2_execution.json)、[configs/r2](configs/r2/campaign.json)及[stage275](../../workbook/2026-11-15_stage275_r2_execution_and_parallel_gates.md)。[最新只读快照](r2_progress_review.json)与[stage278](../../workbook/2026-11-15_stage278_research_synthesis_and_r2_progress.md)补充已完成模型和低occupancy排序，原六份配置不变。

根据R1最终四pass与三seed结果，首批优先B1一族；B2为原生第二族候选，先profile并按明确问题决定是否扩展。B4保留R1 D0及逐类共同参考；是否追加其长训练由前两族的实测问题和费用决定，不自动运行三族全矩阵。B3末轮仍明显改善，若B2的规模收益有限且资源支持，再考虑其长训练控制。优先在seed42做：

表中N是冻结四类pool的事件数；T4实际监督N个unique labels，T3使用其中3N/4个非τlabels。D1/D2的总曝光在T4均为1.2M，在T3均为0.9M，标签预算按真正训练的行数报告。

| ID | Frozen pool events N | Passes | Contrast |
| --- | ---: | ---: | --- |
| D0 | 100k | 4 | fresh、实际精度与新update schedule匹配；R1另作历史参照 |
| D1 | 100k | 12 | 同数据更长曝光；重新定义完整 schedule，不从已退火到最低 LR 的尾部机械续跑 |
| D2 | 300k | 4 | 与 D1 约相同总曝光，更多 unique events |
| D3 | 300k | 12 | D1/D2 仍改善且预算支持时执行 |
| D4 | 1M | 4 起 | 300k 有实际收益后，仅对最强配方扩展 |

固定 updates 的比较对齐 sampler、batch、warm-up/update schedule，分别保留 actual exposures；固定 passes 的比较回答增加数据后训练成本与质量。两种比较各自说明改变因素，不能只报一条随 N 同时改变曝光的曲线。

收敛参考：每隔固定 label exposures 评价 Tune，连续三个 evaluation windows 的 NLL 变化小于 0.001 且主指标没有实际改善时，进入 plateau 检查；记录 train/Tune gap。这个规则是停止计算的工程启发式，不是物理信息饱和证明。出现过拟合先调整正则化/augmentation，在相同 HPO 额度下比较。

先完成首批固定预算终点，不因中间recall波动提前选择配方。D1−D0区分当前schedule下的更长训练，D2−D1区分同曝光下更多unique事件；若D1/D2仍改善才考虑D3，若更多unique明显有益再为1M单独计时，若Train/Tune均停滞再用width32检查容量。排序较好而argmax差时做family交叉拟合bias诊断；低occupancy排序本身也弱时优先局部路径。[09 §3](09_decisions_and_handoff.md#3-当前批次完成r2并作一次有依据的分流)给出完整分流表。endpoint是主比较，best-screen是次比较，不混合两种选点策略或将screen次数不同造成的选择效应称为数据规模增益。

D0完整Tune和readout诊断见[stage279](../../workbook/2026-11-15_stage279_r2_d0_readout_diagnostic.md)。原始评价之外，正式R2交付还要求所有endpoint复用同一五折、两个global bias与正温度对照；完整AP/AUROC经独立sklearn核对。容量扩展必须通过实际width/family匹配的真实拟合检查。合同与当前D0数值在[r2_readout_review.json](r2_readout_review.json)，不把其中两个模型的局部完成状态作为完整R2完成。

容量对照按实际模型选择一个加宽版本：B1为793,674参数，可先将width16增为32；B2为199,882参数，可先将dim64增为128。实际参数、吞吐和显存由实现/profile生成，不把不同族都称为0.9M，也不直接把所有族扩到3–4M。只有学习曲线显示容量限制且收益足够，才继续扩大。先选择数据/训练，再扩大SSL。

R1的B1 seed42使用T4/FP32，新增seed使用H100 MIG/BF16；seed波动含实际执行差异。R2沿用尽早启动的设备政策，每个配对固定数值配方；若运行时fallback改变精度，补相应D0控制或单列配方，不能把旧异精度D0直接解释为纯数据/曝光效应。无需等待同一GPU型号。

B1计算控制说明增加曝光与改变wall-based schedule可以显著改变逐类取舍。D1/D2使用明确的update/exposure schedule，逐窗口保留NLL、AP/ROC、概率分布和e/μ/NC recall；低occupancy组使用原accepted计数定义。必要的低维readout/logit-bias诊断在Train-inner或Tune内按family交叉拟合，Cal-A/Cut-B/Confirm继续封存。根据[M7](05_literature.md#m7)，全局平衡pool的frequency logit adjustment仅加共同常数；若仍有排序与argmax分歧，先检验交叉拟合bias可恢复量，不直接引入long-tail loss，正scalar temperature保留argmax不变对照。

首批设计具体为：同一族/任务固定输入、初始化方式、loss、batch、sampler、selected LR、AdamW和正则化；D0/D1/D2从fresh初始化，按各自预定总updates定义5% warm-up及cosine、保留5% LR floor，wall仅用于成本/deadline。D1扩展完整schedule，不能机械续接D0的低LR尾部。该预算归一化思路参考[M6](05_literature.md#m6)，不据此预设linear优于本项目cosine。

现有sampler保留各mixed-shard window的尾批，D1/D2同曝光不保证精确同updates。提交前从冻结清单实算U，保存batch-size分布、LR轨迹及`sum(lr)`；首批D2−D1明确称同曝光比较。精确fixed-update控制需要另列实际停止步数/曝光，不能同时宣称固定passes、固定曝光及精确同updates。每75k非τ Train曝光评价同一4k Tune screen，终点评价完整40k Tune并保留固定预算checkpoint；best-screen checkpoint单列。新D0匹配实际数值配方。设计见[stage264](../../workbook/2026-11-15_stage264_r2_schedule_and_data_design.md)；已实算300k的T3/T4 U为112630/150117，冻结记录见[r2_execution.json](r2_execution.json)，完整运行成本仍待终态统计。

[stage271](../../workbook/2026-11-15_stage271_r1_sampling_metadata_review.md)的24种metadata重放不支持大段单类别训练：single-class batch均少于.3%。Two-shard窗口仍有类比例自相关，少数缺类窗口每轮最多覆盖T3的6.24%或T4的7.923%事件；分类影响未测量。R2 preflight同时保存窗口/滚动类频率及末段组成，检查head/logit偏移是否对应局部先验。只有出现模型证据，才追加一个同事件、曝光、batch-size序列、updates与LR的重混控制，先profile其I/O；不因metadata差异自动扩矩阵或重写R1结论。

按R1实测wall线性估算，B1在1g MIG完成D1+D2的T3/T4四项约需10.087 device-hours；B2在H100 NVL约13.406 h，两族直接全跑约23.493 h，尚未含新D0、300k转换、失败及额外profile。它们来自不同设备，属于排期估算，不能当FLOP等效成本；300k的实际I/O和训练wall尚未测量。原R2的18 h规划应采用分批执行：首批冻结最有实际收益的B1一族，完成D1/D2并补数值配方匹配的D0；第二族只在首批结果能回答一个尚未解决的问题且实际剩余预算支持时追加相应控制。保持现有可运行分区，不为达到估算吞吐等待整卡；实际设备较慢时缩小下一批矩阵并重新计时。

## 5. R3–R6：按证据逐步增加结构

### R3：几何关系与压缩

以R2配方固定训练和输入，从B1的stride4→2及相位诊断开始，先测额外计算与稀疏信号保留。原生路线优先G1a（压缩前跨窗口/跨层边），使用本族共同baseline；其后按结果决定G1b（显式几何anchors）和G1c（重训去除latent、late fusion）。不同族的结构干预分别配对，不能将跨族输入表示/训练变化归于单一关系层。G1a/G1b均有独立收益时再组合。只对最佳两者做三seed，再仅对最佳结构进行native/100μm/1mm三点配对；不一次启动完整矩阵。

必须给 local/region/global 的独立可读 probe、真实训练梯度及重训 ablation，分别回答“有信息”“实际使用”“增加该分支是否有益”。多分支仍存在竞争时才启动 branch dropout/gradient modulation。随机 mask 的缺模态条件要符合实际推理，不用模拟 truth 决定 dropout。

### R4：重建对象

候选图与 robust fit 先在 Train/Tune 的 10k 子样本上评价完整 fake–efficiency 曲线，比较旧 nearest-neighbor baseline。再选择一种 learned association/instance 方法。只有候选边 recall 和对象覆盖支持后续任务，才扩展全样本；未达标时保留 raw 分类，继续定位拥挤、散射、缺层或 label-quality 原因。

分类最小矩阵为 raw-only、object-summary HGB、raw+object tokens、训练诊断 oracle-association；oracle 只用于问题定位，不参与正式模型排名。Tracking、vertex 和 momentum 的准确目标分别定义，fit 失败事件保留。

### R5：τ 与类别数

使用最佳共同 backbone，先比较 flat T4、T3T、T4D；条件头 λ 从 `{0, 0.1, 0.3}` 的 train-inner 小扫描选择。τhad specialist 优先，τe/τmu 随局部可见性和混淆证据推进。对 specialist 做共同三类非劣和全非 τ 的背景评价；hard-negative replay 和 pAUC 最多各一项独立对照。

可见衰变拓扑与未分辨衰变点的运动学分开设计：[H7/H8](05_literature.md#h7)分别提供OPERA和NOMAD依据。前者同时检查charm、强子再相互作用、大角度μ散射及prompt lepton漏识别；后者先完成可测方向/动量的resolution–coverage。had1/had3细分须先核对直接daughter、最终charged prong与重建track标签定义，保留条件辅助及unknown mask，不直接由当前直接prong计数启动七类softmax。

量化每种拓扑的 eligible/observed/reconstructed/selected 分母，单列 charm 背景；计算几何/占用共同支持后的 paired 效应。当前 S6 的高 occupancy 诊断只用作提出假设，新开发集验证后才据此改变采样/模型。

R5 阶段报告明确选择：近期三类交付、四类保留并继续、或条件六类继续；按[任务合同](02_tasks_and_evaluation.md)的实际效果与非劣界决定，不以训练完成或某个 AUC 点值决定。

### R6：定向预训练与条件鲁棒性

只在胜出的 encoder 上做 topology MAE、局部语义/continuity supervision 和同总计算监督 continuation。先选择一个目标；没有下游实际收益时，不扩成所有目标组合。比较 train-only SSL corpus 的 100k 与较大规模，成本与 label availability 清楚区分。

Detector augmentation 使用已接受幅度；名义和 held-out pattern 同时评价。Standard/NUX 与已有 transport 接口复用，增加统计量前由目标效应和所需区间宽度决定。NuWro/真实 calibration 等新输入可取得后补充，现有研究继续推进。

## 6. R7：最终确认

锁定最佳 T3 与最佳 T4/τ 候选及各自共同基线，冻结 architecture、权重、calibrator、prior、cuts、主比较、条件与 bootstrap 定义，然后开放 Confirm。先拟合 Cal-A 温度，再用 Cut-B 选阈值；确认集不用于模型或 cut selection。

评价完整 raw counts/weighted metrics、四/三类 confusion、ROC/PR、fixed-cut efficiency/FPR/purity、reliability、分支/接受度/occupancy、对象 coverage、detector/generator 条件与资源。对两个主要假设控制多重比较，其他分层注明描述性。S6 旧 test 仍作为历史 reference，不改变其冻结结论。

若最终确认未达到晋级值，报告精确差值、区间、训练波动和资源；实际模型选择服从共同任务质量和已测收益。对每个阶段先写中文 workbook，再提交代码、配置、小型表格和报告。长期 checkpoint、英文 PDF/PNG、完整预测与日志留 results。

## 7. 资源与运行规则

原首轮80 device-hours分配如下，保留作规划与实测偏差的参照。R0、R1已有终态成本，R2已进入执行，其终态成本待统计；R3–R7尚未启动；表中的剩余矩阵需要按实际结果分批重排：

| Stage | Planned device-hours ceiling |
| --- | ---: |
| R0 microbench / fit gates (complete; actual upper bound 1.085 h) | 2 |
| R1 baselines / selection (complete; actual batch 45.709 h) | 18 |
| R2 scaling | 18 |
| R3 geometry | 20 |
| R4 objects | 8 |
| R5 tau | 8 |
| R6 targeted pretraining / robustness | 4 |
| R7 inference / confirmation | 2 |
| Total | 80 |

R1实际batch成本45.708889 h，CPU307.181111 core-hours，122条allocation均终态；已计全部失败、探针和迁移。交互操作下界另有.325533 h，复用历史SSL .635321 h单列。按既有R0上界与R1 batch口径累计46.793744 h，原80 h规划的名义余量33.206256 h，还未计R1交互完整开销；不能再默认执行后续原计划合计60 h。首批R2只选一族D0/D1/D2，后续阶段按实测收益和剩余成本重排。1M 扩展或大规模新模拟超出首轮范围时，先由已测吞吐生成独立资源计划。不同 MIG/整卡型号分别计 allocation；模型计算效率尽量在相同设备和并发条件下比较，不把 MIG 小时当整卡等效 FLOPs。

R0实际：A100 allocation 0.746667 h，H100 allocation 0.162500 h，交互T4含失败尝试上界0.175688 h，合计上界 **1.084855 device-hours**；HTCondor allocated CPU **7.686667 core-hours**、实际进程CPU **1.544444 h**。八条已提交作业均终止，一次站点拒绝未产生allocation。2 h规划额度内已包含失败与编译；GPU小时按型号分别记录，不作FLOP等效相加。

以本轮reference热态吞吐线性外推，100k×4-pass的训练计算约需A100/eager/native2 **1.60 h**，或H100/compiled/native2 **0.64 h**；这是预算估计，不是100k训练实测，不含首次编译、读入、验证与新架构变化。H100/packed+bucket约0.59 h，再加按事件数线性估算的cache构建约0.15 h。这是R0形成时的预算外推；R1最终batch实测45.708889 h已替代它作为成本依据，下一批使用对应架构的真实吞吐和验证开销。

大样本转换、统计和重模拟经 HTCondor；长训练并用 HTCondor 与 Kubeflow，优先能尽快启动的兼容GPU、共享设备和MIG。交互 GPU 用于真实小样本拟合和压力测试。当前固定CVMFS构建最低SM75，低于SM80采用FP32/eager；不以整卡或同型号作为统一准入条件。同计算任务在已获得的同一allocation内配对，跨设备报告实际型号/分区/精度和曝光，不以等待同卡保证比较。操作和恢复见[07](07_resources_and_operations.md)。

提交前查询所有可用 schedd 的 running/idle/held、任务兼容性与写入策略，选择兼容且负载最低者并记录依据。每任务一个 writer，迁移时核对实际进程/终态；复用已有 claims、resume 和资源记账。

输入：`/eos/project-f/fasersim-bonn/public/pinpoint/data`。运行输出：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/`。提交文件：`~/condor/Pinpoint/flavour_v2/<stage>/`。CPU/thread、loader 数和 RAM 在实际 profile 后匹配申请，禁止把大量重复 JSON、per-event 小文件或 xAOD 中间产物作为长期输出。EOS 只保留最终模型、必要预测、小型诊断和可复现配置。

## 8. 首批实现工单与验收

| Work item | Code/config | Input | Acceptance | Status |
| --- | --- | --- | --- | --- |
| W01 | `analysis.flavour.v2.partitions/usage_closure` | 历史 accepted manifests | 使用清单可追溯；Confirm 与历史使用 family 无交叉 | Complete |
| W02 | `analysis.flavour.v2.metadata` | 实际 transform/model | windows、explicit edges、modalities 与真实实现一致 | Complete |
| W03 | `analysis.flavour.v2.data/profile/fit/accept/resources/plots` | 固定 Train events | 分段计时、等价输入/输出、真实学习、RAM/GPU memory与终态 | Complete |
| W04 | `analysis.flavour.v2.r1.tasks/metrics/evaluate` | 同一 label mapping | T3/T4完整评价、raw counts/污染与133配对；T3T/T4D概率/loss已测试，实际训练留R5 | Complete R1 |
| W05 | `analysis.flavour.v2.r1.models/data/fit` | 同一 observed hits | adapters、八项真实学习门槛及全部41模型健康/完整输出验收通过 | Complete R1 |
| W06 | `analysis.flavour.v2.r1.campaign/extend/train` | 冻结 Train/Tune | 16项screen、四pass排名、必要三seed和10项compute完成；Confirm封存 | Complete R1 |
| W07 | geometry sparse graph prototype | plane/cluster measurements | 候选边 recall、边数尾部、memory；不读取 truth 构图 | Pending R3 |
| W08 | [三类合作组对照报告](08_r1_collaboration.md) | R1 实测输出 | 18行域表与完整结果核对；统一prior/工作点、三seed/compute及原研究复现缺项明确 | Complete R1 comparison |

W01–W03的完整执行命令及产物在[stage238](../../workbook/2026-11-14_stage238_r0_partitions_and_pipeline.md)。W04–W06的实现、真实拟合与screen分别见stages239–243；完整R1与W08验收见stage274。独立审查的历史复核命令为：

```bash
which() { command -v "$@"; }
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python roadmap/tools/review_current_results.py \
  --output /eos/home-x/xcheng/FASER/Pinpoint/results/roadmap_review_2026-11-14/current_results_review.json
```

它只读取历史摘要、validation scores和training histories。R1模型命令与提交见stages239–243及[资源执行说明](07_resources_and_operations.md)；最终模型排名与预算配对已测量；训练收敛仍待R2，不由短screen或四pass排名推断。
