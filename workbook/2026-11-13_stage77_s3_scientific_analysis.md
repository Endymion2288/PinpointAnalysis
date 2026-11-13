# S3科学分析：screen完成，确认批次进行中

本文为最终报告的进行中版本。17项screen已完成，八项确认训练尚未全部完成；三seed方法晋级、最终条件域比较和组织成本尚未测得，S3未最终验收。以下表格仅使用已经发布的完整结果。

## 阶段目标

检验plane-aware局部编码、module context与late/joint/latent fusion能否改善`CC nu e / CC nu mu / CC nu tau / NC`四分类，并回答native与1 mm表示的差异是否随局部窗口改变。预定判据是完整四分类质量和资源合同；研究允许得到继续简单基线的结论。用户提出的“shared 4 mm compression抹平native优势”是本阶段要检验的假设。

## 工作内容

沿用用户接受的80,000 train、10,000 validation事件及原有split、geometry、detector-proxy。Run10081/10082/10083/10084依次对应四类，validation每类2,500事件。Calibration/test始终冻结，未重新认证样本或标定proxy。输入为硬件observation白名单，truth只用于监督标签与诊断分层。

实现与旧研究程序独立的`analysis.flavour.s3`：同plane的16/64-pitch或4 mm窗口，完整处理canonical blocks；每个module/type的4/8个learned queries读取全部局部块；显式module/type/连续z和pair relations形成context；late、joint和24-token latent fusion共享module读出接口。设计参考FASERCal的module attention与Perceiver结构原则，未声称复现其预训练、1 cm voxels或31.9M参数模型。ParT/ParticleNet作为只读S2基线。文献与结构边界见[执行合同](../roadmap/09_s3_operations.md)。

首轮17项覆盖窗口/表示2×2、late/joint/latent、module/pair geometry、pixel/bar双向模态、accepted tracker、bar signal和module queries。所有任务冻结四轮、batch4、80,064 applied optimizer updates、seed42；首轮全部完成，AMP skips为0，17×14=238项CPU读回通过。以full-validation NLL选择checkpoint，不按某一类AP选轮次。

首轮完成后按冻结规则选择task14（含tracker的fine/context/latent）和task0（4 mm late）；预定主模型task5及1 mm对应task6也确认，合计四种结构各增加seed17/2026，共八项。它们与screen seed42组成三seed比较；不挑选有利seed替换失败实验。

## 详细数据

### 完整screen四分类结果

下表为seed42、各自最佳NLL checkpoint，正向指标越大越好，NLL越小越好。所有模型均评价相同10,000个validation事件，包括空响应。

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

核心候选及基线的逐类AP如下，类别顺序固定为[e, mu, tau, NC]：

| Model | AP e | AP mu | AP tau | AP NC |
|---|---:|---:|---:|---:|
| part_native_joint_s42_e4_v2 | 0.604781 | 0.460651 | 0.358584 | 0.425130 |
| pn_native_joint_s42_e4_v2 | 0.598822 | 0.414593 | 0.355105 | 0.369575 |
| s3_a4_late_s42_e4 | 0.467282 | 0.360682 | 0.287407 | 0.319593 |
| s3_f16_context_latent_s42_e4 | 0.250000 | 0.250000 | 0.250000 | 0.250000 |
| s3_f16_context_latent_tracker_s42_e4 | 0.452587 | 0.376319 | 0.290370 | 0.353509 |

上述是screen的单seed结果。已完成的S2基线三seed汇总供后续确认比较使用：

| Baseline | Macro AP mean +/- sample SD | NLL mean +/- sample SD | Mean AP [e, mu, tau, NC] |
|---|---:|---:|---|
| part, 1 mm | 0.456559 +/- 0.012097 | 1.227951 +/- 0.013303 | 0.604141, 0.450335, 0.353783, 0.417978 |
| part, native | 0.456496 +/- 0.005208 | 1.227834 +/- 0.006563 | 0.603107, 0.450864, 0.354573, 0.417441 |
| particlenet, 1 mm | 0.435660 +/- 0.006379 | 1.242295 +/- 0.006693 | 0.597374, 0.415034, 0.352515, 0.377719 |
| particlenet, native | 0.431786 +/- 0.003102 | 1.243669 +/- 0.004225 | 0.596544, 0.411017, 0.352376, 0.367206 |

### 配对统计与条件化范围

首轮22/22个冻结配对与1个2×2交互已计算，pending为0。每个比较同时输出raw/common-support的34项统计，包括macro AP/AUROC/F1、balanced accuracy、NLL/Brier及四类AP/AUROC/precision/recall/F1、1%与0.1%背景接受率效率。英文诊断图包含逐类ROC/PR、混淆矩阵和学习曲线。

400次bootstrap共享128个accepted-family/chunk blocks、每类32个。区间是条件于已训练模型的pointwise物理事件区间，未作多重比较修正，不包含训练随机性；三seed的sample SD将单独报告。固定S2 energy/vertex common-support保留8,559个正权重事件，Neff=7,959.413646535，target Z未匹配。它是固定粗相空间诊断，不是物理flux重加权或完整generator独立检验。

Seed42的4 mm late→16-pitch late，macro AP差为−0.020058880，95%事件区间[−0.025400125, −0.015137736]；fine joint→context joint为−0.014432757，[−0.017992908, −0.011131464]；fine context joint→context latent为−0.082316853，[−0.089566596, −0.076320669]。这些是当前实现、参数容量和冻结训练配方的比较，不能推广为架构类别的一般优劣。

### 分辨率与优化机制

Fine/context/latent的native和1 mm模型均对全部10,000事件给出单一概率向量，macro AP均0.25。4 mm窗口的native/1 mm macro AP分别0.261014083/0.321516517。4 mm窗口native−1 mm为−0.060502435，[−0.067034736, −0.054427614]；交互`(fine_native−fine_1mm)−(4mm_native−4mm_1mm)`为+0.060502435，[+0.054427614, +0.067034736]。

该交互包含fine单元的优化失败。对四单元best checkpoint的同8事件CPU FP32/eval诊断中，fine两项head GELU均全零，分类损失到编码器的梯度为零；4 mm/native的GELU零比例99.09%，4 mm/1 mm为64.84%。FP32也存在这一机制，不能只归因于BF16概率量化。全部finite、零AMP skips和四轮完成不等于模型保持有效学习。原始逐层与梯度证据见[阶段65](2026-11-12_stage65_s3_prediction_collapse_diagnostic.md)与[阶段69](2026-11-12_stage69_s3_factorial_results.md)。

CPU token census进一步定位了结构压缩边界。在同一最高occupancy train事件中，pixel+bar的36,547个native nodes经过4 mm/16-pitch窗口分别形成2,641/14,597个local blocks，随后均变为124个module query tokens；latent均再变为24 tokens。8 queries配置把module tokens增至248，global仍24。该计数取三个预定occupancy ranks、17配置，共51条，证明具体结构与转换行为，不能代替分类统计。详见[阶段75](2026-11-13_stage75_s3_token_bottlenecks.md)。

### 模态、接受度与空响应

Tracker候选相对主模型的seed42 macro AP差为+0.118196335，[+0.109862158, +0.127902741]。对照模型已塌缩，增加tracker同时改变了输入、参数与优化状态，因而不能把这个差值全部归因于tracker测量的独立信息。类似地，塌缩模型之间的geometry/bar零差不证明这些信息没有用。

Validation的tracker-active域有5,630事件，tracker-empty有4,370，所有探测器响应全空有282；pixel-centered域885，large-scintillator域4,494，pixel channels>=1,000域1,201。各域保持全部事件及本域class priors。Seed42 tracker候选在pixel-centered/large-scintillator域macro AP为0.503675/0.427976，tracker-active/empty为0.335919/0.370482；域间AP差同时受到prior与样本难度影响，不能作为同一事件的tracker消融证据。

当前tracker为用户接受的hit/station proxy，关系输入中的station index/3不是毫米飞行距离。当前输入没有time或拟合q/p，不进行相同输入的伪timing消融。真实轨迹拟合性能和硬件标定不属于本轮已测量内容，也不是继续使用现有proxy的前置条件。

最终条件域配对、固定背景/相同purity曲线及尾部计数等待确认训练后统一计算。1%/0.1%阈值为validation经验工作点，不是部署阈值；低背景计数必须与效率一起解释。Purity只针对本benchmark prior，不推算真实产额。

### 执行、成本与产物

所有正式训练通过HTCondor或Kubeflow执行。确认批次使用一个`nvidia.com/mig-3g.47gb` reservation、四个独立训练进程；supervisor等待所有子进程退出并自动补入队列。每项具有独立seed/config/checkpoint，未重复提交writer。显存占用与MIG device-hours均不替代GPU compute utilization。

共享reservation按活跃进程时间等分，启动/退出空档等分，组织成本只计一次。缺失终态Pod历史的旧attempt使用明确标注的生命周期保守上界。整卡与MIG成本分列，不能用混合device-hours直接授予模型效率晋级。组织上限120 device-hours，S2成本另列；最终S3终态成本尚未测得。

结果根为`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_2026-11-12`。已完成产物为`screen_figures/four_class_diagnostics.pdf`（33页）、`four_class_resource_tradeoff.pdf`（3页）、`screen_metrics.csv`、`token_census/`及版本化`roadmap/results/s3_screen_summary.json`。最终`evaluation_final/`将由自动流程统一更新，现阶段不能把其中screen汇总误认为完整confirmation结果。

当前已启动的自动执行入口如下；它等待训练输出后依次执行readback、Job终态、accounting、四分类bootstrap、条件域诊断、native事件图和strict gate。这里记录复现命令，本报告整理时未重复启动该进程：

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
S3_ROOT=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_2026-11-12
python -m analysis.flavour.s3.finalize --root "$S3_ROOT" \
  --campaign roadmap/configs/s3_screen.json \
  --confirmation roadmap/configs/s3_confirmation.json \
  --comparisons roadmap/configs/s3_confirmation_pairs.json \
  --job pinpoint-s3-confirm-mig47-parallel-v1
```

已实际验证：screen的238项CPU读回；阶段75的18项model/metrics/decision tests；阶段76新增条件域完整性检查后的8项decision tests。条件域gate逐项比对七个预定域、全部模型、冻结配对、34项raw/common-support统计与400次bootstrap记录，而非只读取成功标记。最终完整科学输出的strict gate与图示人工复核仍待执行。

## 阶段性结论

事实：seed42的细窗口、module context和latent在当前实现与四轮预算下没有给出一致的四分类收益；主fine/native和fine/1 mm均已塌缩。当前证据没有确立shared 4 mm压缩抹平native优势，也不支持native与1 mm输入物理等效。仍存在明确的module/global压缩边界，但其信息损失贡献尚不能与优化失败分离。

待检验假设：未归一化的modality/context拼接及当前优化设置可能促使head进入大负值区。Head normalization、残差缩放或激活干预需要独立冻结版本、稳定优化诊断和同架构对照；本轮不静默替换配方。三seed确认将测量当前配方的波动与失败稳定性，尚未得出最终方法晋级决定。

## 下一步计划

完成八项confirmation与CPU读回，联合报告三seed均值、sample SD、四类配对区间、条件域/尾部统计、native错误事件图和终态成本。逐项审阅strict gate覆盖及英文图示，更新本报告和roadmap完成状态，提交版本化最终汇总与推荐backbone。S4–S6不执行。
