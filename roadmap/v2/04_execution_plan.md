# 下一轮执行路线、实验矩阵与工单

**R0已完成，R1正在执行。**完整Train100k/Tune40k已通过身份、标签、PE与计数验收；四个神经网络的八项真实学习及集成检查、B0三seed、全部16项screen已完成。B1–B4四pass训练、同计算控制及必要的seed重复在Condor/Kubeflow执行。冻结合同见[配置](configs/r1_campaign.json)，screen实测见[stage243](../../workbook/2026-11-14_stage243_r1_all_screens_and_full_training.md)。完整排名、paired比较和stress仍待完成。

首组B4 seed42完整T3/T4与无tracker重训的400次paired结果见[stage246](../../workbook/2026-11-14_stage246_r1_b4_paired_task_and_tracker_results.md)：tracker全域ΔBA3=+3.90 pp，CI95[3.43,4.41]；T3−T4 conditional为−0.33 pp，[−0.69,0.004]。共同BA接近但逐类有取舍，B4尚未显示对HGB的全面优势。普通CNN成本参照与完整四类图表已补齐实现/依赖，实际训练见[stage245](../../workbook/2026-11-14_stage245_r1_cost_control_and_delivery_dependencies.md)。上述中期结果不改变主排名规则。

本文件与 [experiment_plan.json](experiment_plan.json) 记录研究合同及执行状态。W01–W03由`analysis.flavour.v2`实现并完成HTCondor验证，见[stage238](../../workbook/2026-11-14_stage238_r0_partitions_and_pipeline.md)与[r0_results.json](r0_results.json)；W04–W06已开始实现与验证，W08待实际完整结果。R1入口位于`analysis.flavour.v2.r1`，本JSON不能直接交给旧训练CLI。已有accepted samples、response参数和路径工具继续复用。阶段编号使用R0–R7，与已完成的旧S0–S6区分。

25/55条条件路径的实际配对及工作点修复见[stage250](../../workbook/2026-11-15_stage250_r1_condition_interactions_and_operating_point_repair.md)：MAE相对Scratch的noise退化多3.00 pp，CI95[−5.20,−0.92]；相关中期400次统计已修正浮点边界，主要BA/NLL结论不变。该描述性条件分析不修改冻结的架构排名。

六个T4模型的中期比较见[stage252](../../workbook/2026-11-15_stage252_r1_t4_architecture_contrasts_and_independent_review.md)：B1相对B4的BA3/τ工作点提高1.82/2.40 pp，但全域μ recall低5.30 pp；283项独立概率/人数/曝光读回通过。CNN成本参照、B1精度对齐的无tracker控制已完成。完整排名、必要seed扩展及最终R1交付仍待完成。

## 1. 分阶段依赖与交付

| Stage | Estimated window | Main experiment | Required deliverable / gate |
| --- | --- | --- | --- |
| R0（完成） | Completed 2026-11-14 | 使用记录、v2 manifests、loader/transform/profile、实际架构元数据 | 73项最终读回与81项测试通过；Confirm保持封存 |
| R1 | Week 1–2 | B0–B4 的 T3/T4 同输入 screen；合作组表示与域对齐 | 共同三类性能、τ contamination、同计算/曝光表；前两名确定 |
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

R0已在10k真实Train事件上完成native/packed全部collated tensors逐位等价、256-event原frontend数组核对及GPU数值读回；32-event真实拟合在64updates后NLL=0.000418、accuracy=100%，分支梯度和参数更新通过。该拟合检验优化路径，三类/四类泛化效果仍未测量。

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

神经网络架构排名首先比较 Scratch；B4-MAE 是初始化控制，不能把它对 Scratch B3 的差异直接称为架构优势。初始 seed42，使用相同 100k 四类 Train pool 及其非 τ 子集、同一 Tune。每族相同 HPO 计算额度，先两个学习率候选、短预算筛查后延长最优候选；AdamW、weight decay、有效 batch 和 schedule 均记录。

2026-11-14执行更新：按用户要求优先使用可立即启动的不同GPU型号、整卡与MIG，Condor与Kubeflow并用，不再为统一型号等待。相同族的任务/学习率与同计算控制尽量在同一allocation执行；共同曝光比较仍使用完整相同非τ事件。跨设备小时不是FLOP等效量，按型号/分区与实际精度分别报告，配置和权重不按设备混成一个虚假的同计算结论。

训练阶段不以固定四轮作为所有模型的终点。短 screen 给每族相同设备上限和 validation 频率；延长训练到 R2 的收敛合同。HGB 的 boosting 预算按 CPU 时间和复杂度单列。

只对最强两个神经网络与共同基线扩 seeds17/42/2026。合作组配置缺少 run/完整 score/权重时，先在本批数据实现描述清楚的同构基线；相同事件列表和 exact cuts 可用后补做严格复现，不阻塞自身模型改进。报告“同构对照”与“原研究复现”的完成状态。

R1 交付包含全域/pixel-centered/large 三类表、三类训练与四类条件三类配对、τ 污染、calibration 前 raw score 分布和资源 Pareto；原生/图像的贡献与是否有 tracker 分开评价。Tune 内固定小型 noise/combined 开发对照随名义结果一起给出，沿用已接受的 proxy 幅度；它们用于研究已知弱点，R6/R7 另保留未见 pattern。不得以简单去除孤立 hits 的方式牺牲短轨而不报告其效率。

## 4. R2：用小矩阵确定训练规模

先选 R1 最强原生模型与 B4 共同参考，若两者相同则只训练一族。优先在 seed42 做：

| ID | Unique labels | Passes | Contrast |
| --- | ---: | ---: | --- |
| D0 | 100k | 4 | R1 可复用的基准 |
| D1 | 100k | 12 | 同数据更长曝光；重新定义完整 schedule，不从已退火到最低 LR 的尾部机械续跑 |
| D2 | 300k | 4 | 与 D1 约相同总曝光，更多 unique events |
| D3 | 300k | 12 | D1/D2 仍改善且预算支持时执行 |
| D4 | 1M | 4 起 | 300k 有实际收益后，仅对最强配方扩展 |

固定 updates 的比较对齐 sampler、batch、warm-up/update schedule，分别保留 actual exposures；固定 passes 的比较回答增加数据后训练成本与质量。两种比较各自说明改变因素，不能只报一条随 N 同时改变曝光的曲线。

收敛参考：每隔固定 label exposures 评价 Tune，连续三个 evaluation windows 的 NLL 变化小于 0.001 且主指标没有实际改善时，进入 plateau 检查；记录 train/Tune gap。这个规则是停止计算的工程启发式，不是物理信息饱和证明。出现过拟合先调整正则化/augmentation，在相同 HPO 额度下比较。

当前约 0.9M 参数模型与约 3–4M 的单一加宽版本作容量对照；准确参数数由实现生成。只有学习曲线显示容量限制且收益足够，才考虑 10M 量级。先选择数据/训练，再扩大 SSL，避免继续围绕 1k-label 的复杂 probe 争取小收益。

## 5. R3–R6：按证据逐步增加结构

### R3：几何关系与压缩

以 R2 配方固定训练和输入，先执行 G1a（跨窗口/跨层边）、G1b（显式几何 anchors）、G1c（重训去除 latent、late fusion）与共同 baseline，seed42。若 G1a/G1b 有独立收益，再组合；否则保留更简单模型。对最佳两者做三 seed，随后仅对最佳结构进行 native/100μm/1mm 三点配对。

必须给 local/region/global 的独立可读 probe、真实训练梯度及重训 ablation，分别回答“有信息”“实际使用”“增加该分支是否有益”。多分支仍存在竞争时才启动 branch dropout/gradient modulation。随机 mask 的缺模态条件要符合实际推理，不用模拟 truth 决定 dropout。

### R4：重建对象

候选图与 robust fit 先在 Train/Tune 的 10k 子样本上评价完整 fake–efficiency 曲线，比较旧 nearest-neighbor baseline。再选择一种 learned association/instance 方法。只有候选边 recall 和对象覆盖支持后续任务，才扩展全样本；未达标时保留 raw 分类，继续定位拥挤、散射、缺层或 label-quality 原因。

分类最小矩阵为 raw-only、object-summary HGB、raw+object tokens、训练诊断 oracle-association；oracle 只用于问题定位，不参与正式模型排名。Tracking、vertex 和 momentum 的准确目标分别定义，fit 失败事件保留。

### R5：τ 与类别数

使用最佳共同 backbone，先比较 flat T4、T3T、T4D；条件头 λ 从 `{0, 0.1, 0.3}` 的 train-inner 小扫描选择。τhad specialist 优先，τe/τmu 随局部可见性和混淆证据推进。对 specialist 做共同三类非劣和全非 τ 的背景评价；hard-negative replay 和 pAUC 最多各一项独立对照。

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

建议首轮 80 device-hours 分配如下，是规划值，不是已经申请或消耗的成本：

| Stage | Planned device-hours ceiling |
| --- | ---: |
| R0 microbench / fit gates | 2 |
| R1 baselines / selection | 18 |
| R2 scaling | 18 |
| R3 geometry | 20 |
| R4 objects | 8 |
| R5 tau | 8 |
| R6 targeted pretraining / robustness | 4 |
| R7 inference / confirmation | 2 |
| Total | 80 |

预算包括编译、失败和无效尝试；阶段没有效果时将剩余额度用于下一条独立假设或提前收尾。1M 扩展或大规模新模拟超出首轮范围时，先由已测吞吐生成独立资源计划。不同 MIG/整卡型号分别计 allocation；模型计算效率尽量在相同设备和并发条件下比较，不把 MIG 小时当整卡等效 FLOPs。

R0实际：A100 allocation 0.746667 h，H100 allocation 0.162500 h，交互T4含失败尝试上界0.175688 h，合计上界 **1.084855 device-hours**；HTCondor allocated CPU **7.686667 core-hours**、实际进程CPU **1.544444 h**。八条已提交作业均终止，一次站点拒绝未产生allocation。2 h规划额度内已包含失败与编译；GPU小时按型号分别记录，不作FLOP等效相加。

以本轮reference热态吞吐线性外推，100k×4-pass的训练计算约需A100/eager/native2 **1.60 h**，或H100/compiled/native2 **0.64 h**；这是预算估计，不是100k训练实测，不含首次编译、读入、验证与新架构变化。H100/packed+bucket约0.59 h，再加按事件数线性估算的cache构建约0.15 h。R1的18 h上限需据各族实测速率分批分配，不执行全部候选的完整长训练矩阵。

大样本转换、统计和重模拟经 HTCondor；长训练并用 HTCondor 与 Kubeflow，优先能尽快启动的兼容GPU、共享设备和MIG。交互 GPU 用于真实小样本拟合和压力测试。当前固定CVMFS构建最低SM75，低于SM80采用FP32/eager；不以整卡或同型号作为统一准入条件。同计算任务在已获得的同一allocation内配对，跨设备报告实际型号/分区/精度和曝光，不以等待同卡保证比较。操作和恢复见[07](07_resources_and_operations.md)。

提交前查询所有可用 schedd 的 running/idle/held、任务兼容性与写入策略，选择兼容且负载最低者并记录依据。每任务一个 writer，迁移时核对实际进程/终态；复用已有 claims、resume 和资源记账。

输入：`/eos/project-f/fasersim-bonn/public/pinpoint/data`。运行输出：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/`。提交文件：`~/condor/Pinpoint/flavour_v2/<stage>/`。CPU/thread、loader 数和 RAM 在实际 profile 后匹配申请，禁止把大量重复 JSON、per-event 小文件或 xAOD 中间产物作为长期输出。EOS 只保留最终模型、必要预测、小型诊断和可复现配置。

## 8. 首批实现工单与验收

| Work item | Code/config | Input | Acceptance | Status |
| --- | --- | --- | --- | --- |
| W01 | `analysis.flavour.v2.partitions/usage_closure` | 历史 accepted manifests | 使用清单可追溯；Confirm 与历史使用 family 无交叉 | Complete |
| W02 | `analysis.flavour.v2.metadata` | 实际 transform/model | windows、explicit edges、modalities 与真实实现一致 | Complete |
| W03 | `analysis.flavour.v2.data/profile/fit/accept/resources/plots` | 固定 Train events | 分段计时、等价输入/输出、真实学习、RAM/GPU memory与终态 | Complete |
| W04 | `analysis.flavour.v2.r1.tasks/metrics/evaluate` | 同一 label mapping | T3/T4/T3T/T4D概率与loss已实现并测试；完整实际评价待交付 | Implemented; R1 evaluation pending |
| W05 | `analysis.flavour.v2.r1.models/data/fit` | 同一 observed hits | adapters和八项真实T3/T4学习门槛通过；各族完整泛化待测 | Learning gates passed |
| W06 | `analysis.flavour.v2.r1.campaign/extend/train` | 冻结 Train/Tune | 16项screen完成；四pass、同计算和raw counts在执行；不访问Confirm | In progress R1 |
| W07 | geometry sparse graph prototype | plane/cluster measurements | 候选边 recall、边数尾部、memory；不读取 truth 构图 | Pending R3 |
| W08 | 三类合作组对照报告 | R1 实测输出 | 统一域/先验/工作点和 τ contamination；未对齐项具体列出 | Pending R1 |

W01–W03的完整执行命令及产物在[stage238](../../workbook/2026-11-14_stage238_r0_partitions_and_pipeline.md)。W04–W06的实现、真实拟合与screen分别见stages239–243；W08在最终完整比较后交付。独立审查的历史复核命令为：

```bash
which() { command -v "$@"; }
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python roadmap/tools/review_current_results.py \
  --output /eos/home-x/xcheng/FASER/Pinpoint/results/roadmap_review_2026-11-14/current_results_review.json
```

它只读取历史摘要、validation scores和training histories。R1模型命令与提交见stages239–243及[资源执行说明](07_resources_and_operations.md)；短screen完成不表示完整模型排名或收敛已测量。
