# S4低标签训练优化与train-inner选择合同

## 阶段目标

针对[阶段145](2026-11-14_stage145_s4_empty_repair_final.md)测得的1k过拟合、τ硬决策退化及global路径利用差异，在原1k/10k标签预算内优化下游训练。本文冻结48项train-inner对照、选择规则和随后12项三seed重拟合的生成方式；科学结果尚未测量。100k修复矩阵保留作参考，本轮不扩1M。

## 工作内容

旧seed42的1k Scratch/MAE3首个checkpoint前均累计7,104次事件输入、512次更新，分别得到validation NLL 1.535145/1.399620。第二轮NLL升至2.227242/1.617415，训练loss继续下降，支持过拟合判断。10k首轮也已重复使用约19,782次事件输入。此轮将每个epoch改为标签子集的一次完整遍历，保留所有事件及原生节点，在更早的训练位置评价。

模型沿用revision 3、`empty_response_bypass=true`及已通过的真实事件gate，Scratch重新初始化，MAE3复用三套已验证encoder。输入、响应代理、family split和嵌套标签集合沿用已接受定义。新的训练配方选择仅读取原train，在各标签预算内部按物理family分三折，并按四类及τ衰变道分层；不使用独立10k validation选择配方或轮数。

### 论文依据与可检验改动

[Kumar等，ICLR 2022](https://arxiv.org/abs/2202.10054)说明随机Head与encoder共同更新可能改变已学习特征，提出LP-FT；本轮重新联网读取其摘要，缓存为输出根`literature/lpft_iclr2022_revisited.html`。论文主要研究OOD，其机制尚未在本项目得到病因证明，因此作为受控候选。[Sonata，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_Sonata_Self-Supervised_Learning_of_Reliable_Point_Representations_CVPR_2025_paper.html)强调frozen linear probe与多尺度表示检查，支持分别测量可读信息和full fine-tuning的实际利用，不能从整体AP推断各分支均工作。

四种配方对Scratch/MAE3全部运行，readout学习率均为3×10⁻⁴：

| Recipe | Encoder LR scale | Matrix weight decay | Head dropout | Frozen linear phase |
| --- | ---: | ---: | ---: | ---: |
| early | 1.0 | 0.01 | 0.1 | 0 |
| slow | 0.1 | 0.01 | 0.1 | 0 |
| regularized | 0.1 | 0.1 | 0.3 | 0 |
| lpft | 0.1 | 0.1 | 0.3 | 2 complete passes |

Bias、normalization、embedding、queries和latents不作weight decay。`early→slow`仅改变encoder学习率；后两项是完整训练配方比较，不把合并正则改动或LP初始化差异解释为单因素效应。

LP阶段只训练`linear_readout`，全部特征变换、summary、branch norms及非线性Head冻结；非线性Head末层置零，使该阶段输出严格为固定features的线性分类。两次完整遍历后解冻，encoder使用0.1倍学习率，尚未更新的encoder optimizer state从空状态开始。LP期间要求线性readout非零梯度、其余参数无梯度；解冻的首个更新强制记录完整健康检查，之后继续要求真实input梯度及非失活Head。只有第3轮起的LP-FT checkpoint进入候选选择，前两轮仅诊断。

不通过class bias、温度或validation工作点调参修补τ结果。后续比较仍报告NLL、Brier、macro AP/F1、各类confusion、τ e/μ/had及common-support结果，低FPR处保留实际通过计数。

### 固定选择规则

矩阵为4配方×2初始化×2预算×3折，训练seed固定42，每项12次完整遍历。每个预算对每个配方及合格epoch，先按各折留出事件数加权计算Scratch和MAE3的OOF NLL，再对两种初始化等权平均；取最小者，精确同分时依次选择更少updates和声明的配方顺序。两组最终共同采用所选配方及完整遍历次数，避免只调MAE。各折用于选择的标签仍计入原预算。

随后用完整原预算重拟合seeds 17/42/2026，共12项。只提交固定末轮checkpoint，不按独立validation NLL重新选择训练长度；validation曲线可保留作诊断。本次优化在已观察阶段145结果后设计，后续同validation评价仍为条件评价，calibration/test继续冻结。

## 详细数据

配置为[configs/s4_low_label_inner.json](../roadmap/configs/s4_low_label_inner.json)。实际划分如下，计数顺序均为CC νe/CC νμ/CC ντ/NC：

| Labels | Fold | Fit / selection events | Selection class counts | Tau e / mu / had |
| ---: | ---: | --- | --- | --- |
| 1,000 | 0 | 666 / 334 | 83 / 84 / 84 / 83 | 14 / 15 / 55 |
| 1,000 | 1 | 667 / 333 | 84 / 83 / 82 / 84 | 14 / 14 / 54 |
| 1,000 | 2 | 667 / 333 | 83 / 83 / 84 / 83 | 15 / 15 / 54 |
| 10,000 | 0 | 6,666 / 3,334 | 833 / 834 / 834 / 833 | 144 / 146 / 544 |
| 10,000 | 1 | 6,667 / 3,333 | 833 / 833 / 833 / 834 | 144 / 145 / 544 |
| 10,000 | 2 | 6,667 / 3,333 | 834 / 833 / 833 / 833 | 144 / 145 / 544 |

1k三折每次完整遍历分别为52/52/49次更新；10k均为425次。48项共计划137,088次更新，计划总事件输入2,112,000次；这里是配置计算值，尚不是已完成计数。GPU耗时、独立validation改善及τ工作点变化均未测量。仍计入原S4 60 device-hours上限，已完成历史账本24.325865。

新增`CompletePassSampler`固定shard分组、随机化组及组内次序，使每个epoch精确访问每个所选事件一次，并兼顾本地缓存和occupancy bucketing。分层时先随机化完整family ID再使用不打乱group-count对应关系的划分，实际各类计数见上表。

已完成41项相关测试，包括family隔离、train-inner两个角色只请求原train、完整遍历覆盖、LP精确线性输出、冻结/解冻与中途resume、冻结阶段梯度策略，以及固定末轮在validation NLL变差时仍提交末轮。既有S4语义目标、预训练初始化、读回及完整评价测试同时通过。`git diff --check`通过。

可复现命令（LCG108 CUDA环境并设置仓库src为PYTHONPATH）：

```bash
python -m unittest tests.test_flavour_s4_low_label tests.test_flavour_s2_training tests.test_flavour_s4
python -m analysis.flavour.s4.plan_low_label plan \
  --root /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13 \
  --source roadmap/configs/s4_empty_repair.json \
  --output roadmap/configs/s4_low_label_inner.json
```

## 阶段性结论

事实：原1k设置在首次checkpoint前已有约7次事件曝光，新采样、分层及LP训练状态实现已通过针对性测试，48项合同已固定。

假设：更早checkpoint、减少曝光、降低encoder LR和LP-FT可能改善概率质量或保留预训练信息，尚未测量净收益。即使排序改善，也不能替代ντ硬决策、common-support和低背景效率的证据。

## 下一步计划

提交本阶段代码与配置后，由Kubeflow/HTCondor竞争领取48项train-inner任务；沿用已测的47GB MIG六进程、24GB MIG两进程、12GB MIG单进程并发范围，按实际内存、吞吐和终态调整。全部inner训练和健康证据齐全后运行`plan_low_label resolve`，冻结12项重拟合，完成独立validation配对比较；之后按新路由补齐frozen probes及其他预训练目标和matched-compute控制。选择和训练失败时先检查具体架构或优化原因，不用低AP单独否定硬件信息。
