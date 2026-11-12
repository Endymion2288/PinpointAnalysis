# S3：Plane-aware 局部编码、Module context 与四分类融合

S3进行中。继承S2接受的80k train/10k validation、原生通道、geometry和detector-proxy；calibration/test冻结。不重复生成或认证样本。正式根目录为`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_2026-11-12`，其中输入通过symlink引用S2，输出独立保存。S2结果只读复用。

## 问题与实验合同

S2没有隔离“4 mm compression抹平native优势”的因果；S3把它作为待检验假设。先做同编码器native/1 mm×4 mm/16-pixel-window的2×2控制，再比较64-pixel窗口、每module 4/8个readout queries。局部窗口后还有module与global bottleneck；原生支持完整保留不等于编码可逆，报告每层nodes/blocks/module tokens/latents与质量、成本。

主任务是`CC nu e / CC nu mu / CC nu tau / NC`四分类。预先固定主评价为macro one-vs-rest AP与四类NLL，并联合看macro F1、balanced accuracy、macro AUROC、Brier、各类AP/AUROC/precision/recall/F1和混淆矩阵。τ AP、τ分支及固定背景接受率效率继续保留；不能用τ单项决定backbone。使用full-validation NLL选择checkpoint。

训练均4 epochs、batch4、20,016 attempted steps/epoch、80,064 updates、seed42，AdamW与S2相同。正式环境为整卡H100 NVL/LCG108/Torch2.7.1/BF16；不能将T4或MIG probe混入主比较。dim96，参数实测随消融记录；相对S2容量不同，S2跨架构比较同时呈现实际GPU allocation与参数，S3内部对照用于拆分处理变化。

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

`analysis.flavour.s3`独立实现。Pixel采用16/64个真实pitch的同层二维窗口（约0.3328×0.3648/1.3312×1.4592 mm）；4 mm是独立控制。每块至多32 nodes，超限canonical分块全部处理。局部坐标残差及Fourier features从当前表示计算；1 mm处理不携带native residual或额外原生multiplicity。两层局部set attention后作learned pooling。

Pixel module为geometry中`layer_is_pixel`真项的有序位置，bar使用已有hardware `layer`，tracker使用station ID；这些是硬件结构。每module/type的4个learned queries读取该组所有局部块。Module context加入module/type embedding、连续z及Δz/same-module/same-type关系；global pair不把bar未知横坐标当精确点。Geometry-off仍保留同一物理分组，只关闭显式embedding/relations，不能解释为去除全部几何信息。

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

HTCondor入口`analysis.flavour.s3.condor`每次检查schedd排队、EOS兼容性、WRITE和GPU pool。两后端任务分配在提交记录中明确；只迁移真实idle且已确认无writer的原任务，先终止旧副本再提交。训练复用S2已验证的atomic checkpoint/resume loop，通过显式dataset/model/collator hooks接入，S2默认行为保持。

## 来源与本次补读

FASERCal本地arXiv v2的architecture Methods再次核对了module-level self-attention→Perceiver IO、detector-type embedding和31,936,341参数；本实现采用其结构原则，不复制1 cm voxel或31.9M模型规模。原文与引用见[02_literature.md](02_literature.md)。

2026-11-12联网补读[Perceiver IO全文](https://arxiv.org/html/2107.14795)第3节及复杂度、[LitePT全文](https://arxiv.org/html/2512.13689v1)Methods（early local processing、low-resolution attention与PointROPE）、[PTv3正式页](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Point_Transformer_V3_Simpler_Faster_Stronger_CVPR_2024_paper.html)，并重新核对[CERN flavors](https://ml.docs.cern.ch/#available-flavors)、[training](https://ml.docs.cern.ch/ml/training/)。HTML与FASERCal Methods提取保存在results的`literature/`。S3用成熟PyTorch算子实现局部/全局分工；未实现或声称复现LitePT的sparse convolution/custom kernel。

## 完成门槛

执行入口：`analysis.flavour.s3.evaluate`对raw/common support同时输出所有四类配对区间与2×2交互，所有seed共享物理block multiplicities；`analysis.flavour.s3.select`仅在完整screen/CPU读回/终态计费通过后按上述规则写独立confirmation campaign和三seed配对。`analysis.flavour.s3.accounting`累计两后端，并单列初次interactive T4保守计费上界。阶段57已在16个只读S2模型上完成统计执行验证；40项S3实验/配对/交互当时仍pending，不作S3验收。

`analysis.flavour.s3.monitor --watch`每60秒观察训练产物，只对新发布complete的模型执行14项CPU读回；scheduler liveness另查。最终`analysis.flavour.s3.accept --comparisons <frozen-confirmation-pairs.json>`精确比对所有冻结配对、确认矩阵及raw/common support的34项四类统计和400个有效replicates，不能只凭数量判定通过。

17首轮和选择后去重的确认项全部真实终态、四轮/80,064 updates/零AMP skips；CPU checkpoint/optimizer/predictions读回通过；冻结数据与配置一致。完成四分类所有配对与两种窗口×表示交互、三seed、模态/资源Pareto、接受度/空响应/高占用/错误组与英文图示、真实终态成本、中文科学报告及gate decision。最终无pending、strict输出门槛exit0并提交后，才能标记S3完成。S4–S6未执行。
