# S3：Plane-aware 局部编码、Module context 与四分类融合

**S3 revision 2已完成20模型修复重训及完整validation评价，见[最终报告](../workbook/2026-11-13_stage87_s3_repair_final_results.md)和[修复合同](10_s3_repair.md)。本文件的旧批次结果保留为故障诊断，原方法结论撤回。**

本文件保存原 S3 实验合同及执行记录，原方法晋级结论已撤回。Revision 2沿用S2接受的80k train/10k validation、原生通道、geometry和detector-proxy，已完成8个screen与12个确认训练；新结果根为`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_repair_2026-11-13`。Multiscale+tracker通过本轮质量门槛，pixel+bar-only继续以fresh ParT为基线；global弱利用及ντ未见明确增益见最终报告。原结果根`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_2026-11-12`只作故障诊断。本文记录S3结束时的calibration/test冻结状态；后续S4–S6已完成，见[S6最终结果](13_s6_final_evaluation.md)。

## 问题与实验合同

S2没有隔离“4 mm compression抹平native优势”的因果；S3把它作为待检验假设。先做同编码器native/1 mm×4 mm/16-pixel-window的2×2控制，再比较64-pixel窗口、每module 4/8个readout queries。局部窗口后还有module与global bottleneck；原生支持完整保留不等于编码可逆，报告每层nodes/blocks/module tokens/latents与质量、成本。

主任务是`CC nu e / CC nu mu / CC nu tau / NC`四分类。预先固定主评价为macro one-vs-rest AP与四类NLL，并联合看macro F1、balanced accuracy、macro AUROC、Brier、各类AP/AUROC/precision/recall/F1和混淆矩阵。τ AP、τ分支及固定背景接受率效率继续保留；不能用τ单项决定backbone。使用full-validation NLL选择checkpoint。

训练均4 epochs、batch4、20,016 attempted steps/epoch、80,064 updates、seed42，AdamW与S2相同。环境为H100 NVL/LCG108/Torch2.7.1/BF16；按用户后续资源指令，允许H100 NVL 3g.47gb MIG执行冻结科学任务，逐模型记录实际GPU名称及共享方式。T4仅用于早期验证。整卡与MIG device-hours分列，不能当同等吞吐量用于硬件无关的加速结论。dim96，参数实测随消融记录；相对S2容量不同，S2跨架构比较同时呈现实际GPU allocation与参数，S3内部对照用于拆分处理变化。

`configs/s3_screen.json`冻结首轮17项，`configs/s3_pairs.json`冻结22主配对。输入modalities、feature masks与窗口来自配置，不修改accepted proxy值。科学输出未完成时显式列pending，不能按已完成的有利子集提前验收。

| Tasks | Treatment | Question |
| --- | --- | --- |
| 0–4 | 4 mm/16-pixel late；16-pixel joint/latent；context+joint | 局部细化、融合和context各自作用 |
| 5 | 16-pixel/module-context/latent，native | 预定主模型，pixel+bar |
| 6–9 | 主模型1 mm；4 mm/native与1 mm；64-pixel/native | 表示粒度与局部压缩的2×2及窗口控制 |
| 10–11 | 关闭module embedding+pair relations；仅关闭pair | 结构分组之外的已知几何贡献 |
| 12–14 | pixel-only、bar-only、pixel+bar+accepted tracker | 双向模态增量及tracker四分类收益/覆盖 |
| 15 | binary bar support，关闭已有proxy信号 | 可用bar signal的增量；pixel始终binary |
| 16 | 每module 8 queries | 后续module bottleneck控制 |

## 实现与信息边界

`analysis.flavour.s3`独立实现。Pixel采用16/64个真实pitch的同层二维窗口（约0.3328×0.3648/1.3312×1.4592 mm）；4 mm是独立控制。每块至多32 nodes，超限canonical分块全部处理。局部坐标残差及Fourier features从当前表示计算；1 mm使用冻结的`reducer=count`，保留bin内native hit数，不保留bin内细坐标残差，也不外挂当前表示以外的raw统计。这一对照不是binary OR。16-pitch窗口小于1 mm coarse中心间距，coarse表示中该窗口多数为单node；因此2×2交互描述这组具体表示与编码，不能推成所有native信息的普遍因果结论。两层局部set attention后作learned pooling。

Pixel module为geometry中`layer_is_pixel`真项的有序位置，bar使用已有hardware `layer`，tracker使用station ID；这些是硬件结构。每module/type的4个learned queries读取该组所有局部块。Module context加入module/type embedding、连续z及Δz/same-module/same-type关系；global pair不把bar未知横坐标当精确点。Geometry-off仍保留同一物理分组，只关闭显式embedding/relations，不能解释为去除全部几何信息。

关系坐标的实际定义：pixel/bar为硬件z[mm]/3000；继承S2的accepted tracker proxy把station index编码为station/3。后者是station元数据坐标，跨tracker的Δz输入不能解释成毫米飞行距离。当前17项输入与配置保持冻结，tracker增量按该accepted hit/station表示报告；nominal tracker-plane物理z的替换属于另一个对照，不在当前结果中冒称已实现。

Late、joint、latent均读取相同module tokens；joint是在这些tokens上的全连接attention，latent是24 learned queries的cross-attention与两层latent self-attention。Late仅pooling。所有融合保留三模态独立pool与当前表示的计数/信号和。Module context开启/关闭可能改变参数和算力，实测记录，不作纯参数无关因果声明。

当前输入没有可用channel time或拟合q/p；time关闭由输入白名单直接证明，绝不训练两个相同输入却称timing消融。Charge消融只针对已有bar edep/PE proxy。Tracker仅为用户接受的测量hit proxy，分别报告all-event、active、tracker-active/inactive、输入全空、接受度与固定背景/相同purity曲线。拟合轨迹的q/p、covariance、fake率和真实重建覆盖未测量，不把hit-proxy结果写成track reconstruction的表现；对象/拟合重建仍属于后续S5分支。

## 筛选、确认与统计

首轮全部完成后在结构候选tasks 0–9、14、16中作四分类质量/资源Pareto筛选。Task14含tracker，模态贡献另与task5同事件配对。选择规则先冻结：第一个为macro AP最高的Pareto候选；第二个为其余Pareto候选中NLL最低者；若同点或只有一个Pareto点，用其余候选中四类macro AP损失≤0.005且allocation最低者，若无此候选则用其余最低NLL者。并列按NLL、allocation、task ID顺序。所有类的退化及区间须单列解释，不能因总体均值改善隐去某一类。

两个候选加seed17/2026，与原seed42组成三seed确认。预定主模型task5及其1 mm counterpart task6也保留三seed分辨率比较；已被候选覆盖的run去重。最多新增8项确认，保存独立冻结confirmation配置及选择依据，不能重写screen合同。若候选无收益，仍完成解释性控制，推荐继续简单基线。

物理事件区间使用S2相同的128 accepted-family/chunk blocks、每类32，400次paired bootstrap；四类指标逐项报告pointwise区间。三seed使用同一事件块重采样后平均每seed差，另列训练sample SD。固定粗E/vertex支持沿用S2的8,559事件/Neff=7,959.41365，target Z/同位素未匹配；raw与common-support四分类均评价。1%/0.1%工作点、各条件域、τ分支和空响应保留原分母。各类AP不同prior不得直接当纯拓扑比较。

方法晋级落实用户的四分类要求：质量路径为三seed平均macro AP、macro F1或balanced accuracy至少增加0.01且对应配对95%事件区间支持正收益，NLL增加的区间上界不超过0.01；各类AP下降超过0.01的证据须逐类说明，存在此退化时不作全面晋级。资源路径为至少50%实际训练成本下降且质量处于预定近等效界限（macro AP下降不超过0.005、各类AP下降不超过0.01、NLL增加不超过0.01）。两条路径均同时报告训练spread和模型条件。原roadmap的τ AP+0.01/工作点相对+10%保留为额外物理收益指标，不能单独替代四分类验收。若未达标，保持简单基线并报告解释性结果；S3研究完成不要求新模型必胜。

## 资源与调度

S3组织上限120 GPUh（screen/probe预留70，确认及最终诊断预留50），包含失败、恢复、旧尝试与所有后端，不自动耗尽。S2的92.73055556 GPUh另列。先做CPU/少量T4完整事件验证，再做H100/BF16最大占用与真实optimizer smoke；正式大训练全部通过HTCondor或Kubeflow。

按照用户最新要求，Kubeflow和HTCondor哪边实际获配就用哪边，减少等待，不固定后端优先级。Kubeflow使用CERN `h100-94gb` flavor：`NVIDIA-H100-NVL`与`nvidia.com/gpu:1`，CPU/RAM由平台注入46/127G，先server Pod dry-run。Pod真实Scheduled/Running才算获得GPU；Pending观察记录实际scheduler事件，不推断一定更快。`analysis.flavour.s3.backend --dispatch`可在Condor队列继续时申请一个Kubeflow slot；`dispatch watch`在真实CUDA/BF16 ready后，用schedd端`JobStatus == 1`条件hold并移除一个idle副本，再发布唯一冻结任务assignment。已Running的任务保持原地。worker准备好后等待assignment至多300 s，所有占用计入预算。登录已正常，单GPU配额中不创建重复writer。

当前dispatch Job在近3小时Pending后，于2026-11-12 17:27 UTC把包含排队时间的总时限从6小时延长至12小时，保留同一Job UID及完整Pod template，维持原队列位置。GPU flavor、四轮训练与120 GPUh组织预算未变。后台watcher的进程时限独立于Job；到期后确认旧进程退出再续接，避免多个handoff watcher竞争同一slot。

HTCondor入口`analysis.flavour.s3.condor`每次检查schedd排队、EOS兼容性、WRITE和GPU pool。两后端任务分配在提交记录中明确；只迁移真实idle且已确认无writer的原任务，先终止旧副本再提交。训练复用S2已验证的atomic checkpoint/resume loop，通过显式dataset/model/collator hooks接入，S2默认行为保持。

## 来源与本次补读

FASERCal本地arXiv v2的architecture Methods再次核对了module-level self-attention→Perceiver IO、detector-type embedding和31,936,341参数；本实现采用其结构原则，不复制1 cm voxel或31.9M模型规模。原文与引用见[02_literature.md](02_literature.md)。

2026-11-12联网补读[Perceiver IO全文](https://arxiv.org/html/2107.14795)第3节及复杂度、[LitePT全文](https://arxiv.org/html/2512.13689v1)Methods（early local processing、low-resolution attention与PointROPE）、[PTv3正式页](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Point_Transformer_V3_Simpler_Faster_Stronger_CVPR_2024_paper.html)，并重新核对[CERN flavors](https://ml.docs.cern.ch/#available-flavors)、[training](https://ml.docs.cern.ch/ml/training/)。HTML与FASERCal Methods提取保存在results的`literature/`。S3用成熟PyTorch算子实现局部/全局分工；未实现或声称复现LitePT的sparse convolution/custom kernel。

## 完成门槛

阶段69已完成seed42的四单元2×2及配对事件统计，详见[完整报告](../workbook/2026-11-12_stage69_s3_factorial_results.md)。Fine/native和fine/1 mm最终均为常数预测；4 mm/native和4 mm/1 mm的macro AP分别0.261014083/0.321516517。4 mm窗口的native−1 mm为−0.060502435，95%事件区间[−0.067034736, −0.054427614]；交互为+0.060502435，[+0.054427614, +0.067034736]。该交互包含fine模型的优化失败，不能据此确立“4 mm抹平native优势”。四个best checkpoint的同8事件FP32诊断显示fine两项head GELU全零、上游分类梯度为零；4 mm/native的GELU零比例99.09%，4 mm/1 mm为64.84%。这些诊断不改变正式训练与输入。

最终17/17项screen与8/8项确认完成，共25模型、350/350项CPU读回通过；每模型四轮、80,064 applied updates、零AMP skips。完整评价为41模型、22单seed配对、9三seed比较及1交互，pending=0；条件评价37模型、七个子域、1,184条逐类工作点记录，pending=0。19项strict输出门槛全部通过，八项质量/效率晋级决定均false。Tracker候选与4 mm late的三seed macro AP均值分别0.370622738/0.350286756，ParT native为0.456496213；tracker−ParT差−0.085873475，事件CI [−0.094277671,−0.077817008]。保留ParT native reference及1 mm控制，不能从当前塌缩模型认定native等效或压缩抹平优势。

共享GPU执行以`analysis.flavour.s3.backend --tasks <task IDs> --concurrency 4 --flavor h100-47gb`启动，独立冻结配置、seed、checkpoint和EOS日志；父进程等待所有训练退出，`parallel`记录子进程区间。确认Job `pinpoint-s3-confirm-mig47-parallel-v1`已Complete，8子进程exit0，当前无S3 Pod。Accounting按活跃时间等分reservation及启动/退出空档，总成本只计一次。确认reservation上界14,259秒=3.960833333 MIG device-hours，端点依据Scheduled事件与Job Complete condition；没有观测到清理后的Pod phase/exit code，缓存已改为null并标注推定终态。已删除整卡尝试仍以8.269070509小时生命周期上界计费。

最终S3组织上界31.238268678 device-hours（含interactive T4上界0.181142613），预算120，S2另列。18条显式上界分摊记录对应7个Job、含两个零小时记录，缺失历史0；整卡/MIG不是吞吐等价设备，不据混合成本授予效率晋级。详见[最终科学报告](../workbook/2026-11-13_stage77_s3_scientific_analysis.md)、[最终验收审计](../workbook/2026-11-13_stage80_s3_final_acceptance.md)和[版本化摘要](results/s3_summary.json)。英文诊断/资源/条件曲线PDF分别41/4/40页，主模型/tracker的native事件图分别8/20页，已人工复核。

执行入口：`analysis.flavour.s3.evaluate`对raw/common support同时输出所有四类配对区间与2×2交互，所有seed共享物理block multiplicities；`analysis.flavour.s3.select`仅在完整screen/CPU读回/终态计费通过后按上述规则写独立confirmation campaign和三seed配对。`analysis.flavour.s3.accounting`累计两后端，并单列初次interactive T4保守计费上界。阶段57记录早期只读执行验证；最终验收以全部confirmation产物为准。

`analysis.flavour.s3.monitor --watch`每60秒观察训练产物，只对新发布complete的模型执行14项CPU读回；scheduler liveness另查。最终`analysis.flavour.s3.accept --comparisons <frozen-confirmation-pairs.json>`精确比对所有冻结配对、确认矩阵及raw/common support的34项四类统计和400个有效replicates，不能只凭数量判定通过。

条件域gate还逐项核对七个预定域、全部S3与ParT/ParticleNet模型、适用的单seed/三seed冻结配对，以及raw/common-support的34项统计和bootstrap记录；物理上无定义的统计须明确记录null及有效replicate数。条件域曲线PDF与Parquet同时检查。细节见[阶段76](../workbook/2026-11-13_stage76_s3_conditional_gate_audit.md)。

以上退出条件已满足：冻结25模型、350项读回、完整四分类配对/交互/三seed、模态和资源图、条件域/尾部计数、native事件图、终态资源证据和明确上界、中文科学报告及gate decision均已完成。最终strict输出门槛exit0，图示人工复核完成。S3研究完成，新模型不晋级；S4–S6未执行。
