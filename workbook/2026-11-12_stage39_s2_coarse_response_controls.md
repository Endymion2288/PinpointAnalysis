# 阶段 39：S2 粗格 multiplicity 与网格原点控制

ParT 1mm OR 控制已完整结束，14项输出读回全部通过。count−OR 的 τ ΔAP=−0.00003083，pointwise 95%事件区间=[−0.00529256,+0.00474363]；固定common-support为+0.00253247，区间=[−0.00249439,+0.00724162]。当前seed42、模型和四轮预算下，未观察到保留multiplicity的正收益，区间上界低于预定+0.01。此结果不等于multiplicity没有分类信息，也不构成跨种子的等效结论。S2其余控制和确认继续执行。

## 阶段目标

回答同一1mm粗格内保留active native pixel数量，相对只保留occupied cell，是否改善四分类和τ识别；给出输出完整性、事件区间、工作点和实际成本。

## 工作内容

`1573782.10`在`bigbird20.cern.ch`正常退出，JobStatus=4、ExitCode=0、NumJobStarts=1，原节点`b9pgpun003.cern.ch`。CPU读回检查冻结配置、四轮checkpoint/optimizer、NLL选择、10k最佳预测与40k逐轮预测；14项全部通过。沿用用户接受的sample/split/detector-proxy。

OR与既有count模型使用相同1mm粗格中心、occupied-cell集合、pixel+bar输入和模型结构。OR将每个occupied cell的pixel count设为1；局部特征和全局加性汇总均仅使用当前OR表示，没有恢复native total count。该比较同时移除格内multiplicity及其全事件总和，不能解释为单独的局部布局或空间粒度效应。

资源核算和部分评价均正常退出。评价commit=`c93be6e`：17 results（4 references+13完整科学模型）、457行分域指标、51个历史三类诊断、19项配对完成、26项pending、2组四格交互。配对图第2页及OR的四分类诊断页已实际查看。

## 详细数据

ParT dim48、151,228 parameters；H100 NVL/LCG108/Python3.12.11/Torch2.7.1/BF16，seed42、batch4、4 epochs、20,016 attempted steps/epoch。独立train80k/validation10k，各类20k/2,500；四轮exposures=80,040/80,044/80,040/80,040，合计320,164。80,064次更新全部应用，0 AMP skips；NLL选择epoch4。仍使用同平面4mm分块、canonical chunks≤32 nodes的局部learned compression，随后处理全部全局块。

| Epoch | OR NLL | Tau AP | Cumulative program wall [s] |
| --- | ---: | ---: | ---: |
| 1 | 1.29901877 | 0.28931086 | 1,546.78850 |
| 2 | 1.27470661 | 0.33093780 | 3,052.97763 |
| 3 | 1.25867373 | 0.35401651 | 4,797.53294 |
| 4 | 1.23353902 | 0.35505940 | 6,315.90998 |

| Metric | Count | OR |
| --- | ---: | ---: |
| Tau AP | 0.35502857 | 0.35505940 |
| Fixed common-support tau AP | 0.32086106 | 0.31832859 |
| Tau ROC AUC | 0.63183757 | 0.63114600 |
| NLL | 1.24293540 | 1.23353902 |
| Balanced accuracy | 0.40330 | 0.41440 |
| Macro F1 | 0.39880034 | 0.41593826 |
| Brier | 0.67581568 | 0.66976770 |
| CC nu_e AP | 0.59928946 | 0.59565740 |
| CC nu_mu AP | 0.41981120 | 0.44136400 |
| NC AP | 0.39628130 | 0.40420295 |
| All-attempt allocation [GPUh] | 2.02166667 | 1.75694444 |

四类AP均以每类25%的benchmark prior为条件。OR的argmax recall为e/μ/τ/NC=0.4664/0.3244/0.5308/0.3360。OR的NLL、balanced accuracy、macro F1和Brier点估计优于count，电子AP点估计降低；这些非τ差值的配对CI未测量，不据此确定总体优劣。

沿用400次、128个accepted-family/chunk blocks（每类32）的配对bootstrap，方向为count−OR。raw τ ΔAP=−0.00003083，CI=[−0.00529256,+0.00474363]；固定common support为8,559 events、Neff=7,959.41365，ΔAP=+0.00253247，CI=[−0.00249439,+0.00724162]。两个pointwise事件区间均跨0、上界均低于+0.01；common-support区间以固定E/vertex权重为条件。区间不包含跨训练种子的波动，也没有多重比较同时覆盖保证。

### 工作点、分支与域

Validation empirical 1% FPR下，OR阈值=0.42987147，count阈值=0.44188008；二者均通过88/2,500个τ和75/7,500个背景，τ效率=0.0352。效率差=0，CI=[−0.00787757,+0.00784652]；相同通过数不意味着通过同一批事件。0.1% FPR下OR通过12个τ/7个背景、count通过10个τ/7个背景，效率差count−OR=−0.0008，CI=[−0.00545785,+0.00242097]，尾部统计不足。最终工作点仍留待S6。

OR在inclusive 1%阈值下，τ electron/muon/hadronic通过10/5/73个，分母为445/395/1,660；相应分支AP=0.08051427/0.06342823/0.29138062。分支AP使用各分支τ与同一7,500个non-τ背景。分支间或分支差值CI未测量。

OR的pixel-centered域885 events/176 τ、AP=0.31675295；large-scintillator域4,494/1,053、AP=0.38456060。pixel+bar空输入440/148、AP=0.33636364，tracker-only response域158/58、AP=0.36708861，后两域AUC=0.5、AP等于域prior。域内count−OR配对CI未测量，训练域控制仍待结果。

### 成本、产物与复现

OR程序wall=6,316.84693 s，实际allocation=6,325 s=1.75694444 GPUh，额外8.15307 s，peak allocated memory=1,545.35400 MiB。共同allocation预算6,324.06305 s内，OR选epoch4，count选epoch3（余847.83373 s），count−OR τ ΔAP=−0.00094933；这是整轮分辨率点估计，CI未测量。OR和count具有相同occupied-cell token数量，测得wall差异不能直接解释为OR在结构上减少计算量。

正式根目录：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11/h100/`。`part_or_count_comparison.json`保存训练、14项读回、两模型、配对、分域和核算快照；`part_or_count_paired_review.png`保存本次配对图第2页，`evaluation_final/part_1mm_or_diagnostic_review.png`保存OR四分类诊断。日志为`readback_part_1mm_or.log`、`accounting_part_1mm_or.log`和`evaluation_part_1mm_or.log`。

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
S2_RESULTS=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11
python -m analysis.flavour.s2.readback --root "$S2_RESULTS/h100" \
  --campaign "$PWD/roadmap/configs/s2_h100_v2.json" --experiments part_1mm_or_s42_e4_v2
python -m analysis.flavour.s2.accounting --root "$S2_RESULTS"
python -m analysis.flavour.s2.evaluate --root "$S2_RESULTS/h100" \
  --output "$S2_RESULTS/h100/evaluation_final" \
  --comparisons "$PWD/roadmap/configs/s2_complete_pairs_v2.json" --allow-incomplete
```

06:56:24 UTC组织累计62.68373474 GPUh，H100视图26.44540141 GPUh，missing history records=0。07:02:13 UTC正式13 complete、6 running、16 idle、0 held；原队列已启动ParT no-pair。Kubeflow v7仍Pending、无node，科学任务未迁移。

## 阶段性结论

事实：在当前ParT/seed42/四轮预算下，保留1mm格内multiplicity没有得到τ AP或1% FPR效率的正事件区间支持；τ AP的raw/common-support区间均可排除至少+0.01的收益。OR保留occupied-cell形态和bar信息，仍达到与count接近的τ表现。

待检验解释：占用形态和bar响应可能已吸收大部分当前模型可利用的计数相关信息；模型优化也可能使额外信息没有转化为收益。当前实验不能区分这些解释，不把有限模型的结果当作探测器信息量定理。ParticleNet OR、网格原点、x–y关联、模态/域/跨层和三种子仍按冻结合同继续。

## 下一步计划

完成ParT平移及后续控制，逐项读回并更新配对；完成剩余22项科学配置及41项单seed配对、4项三种子均值。实际结果、资源和物理解释齐备后执行无`--allow-incomplete`的严格评价，再验收S2。calibration/test保留至S6，不进入S3–S6。

## 补充：ParT 半格平移完成

`1573782.11`正常结束，JobStatus=4、ExitCode=0、NumJobStarts=1，原节点`b9pgpun001.cern.ch`；07:06:07 UTC的14项读回全部pass。原点在两个横向坐标各平移0.5 mm，仍用1mm count表示；它改变cell分配和粗格中心，后续4mm局部分块也从改变后的中心计算，因而是整个当前表示对原点的敏感性控制。训练/验证集合、seed42、20,016 steps/epoch、四轮、151,228 parameters、H100/BF16均与主模型一致，80,064 applied、0 skips、320,164 exposures，NLL选epoch4。

| Epoch | Shift NLL | Tau AP | Cumulative program wall [s] |
| --- | ---: | ---: | ---: |
| 1 | 1.33265860 | 0.29312442 | 1,453.82434 |
| 2 | 1.28265792 | 0.32966793 | 2,933.12753 |
| 3 | 1.25045882 | 0.35366070 | 4,372.96805 |
| 4 | 1.21677041 | 0.35566398 | 5,815.90250 |

Shift的τ AP=0.35566398、固定common-support AP=0.32141340、τ AUC=0.63382659、balanced accuracy=0.42230、macro F1=0.42380322、Brier=0.65983713。四类e/μ/τ/NC AP=0.60649926/0.47177288/0.35566398/0.42743996，argmax recall=0.4852/0.3136/0.5440/0.3464。NLL和这些总体指标点估计优于reference-origin count，但非τ差值CI未测量，不能仅以τ稳定推断所有分类输出稳定。

400次、同128 blocks的pointwise配对区间，方向为reference origin−shift：

| Metric | Difference [95% event CI] |
| --- | --- |
| Raw tau AP | −0.00063541 [−0.00587022,+0.00528142] |
| Fixed common-support tau AP | −0.00055234 [−0.00760615,+0.00632436] |
| Tau ROC AUC | −0.00198901 [−0.00642941,+0.00356352] |
| Tau efficiency at 1% FPR | +0.00200000 [−0.00553201,+0.00880512] |

Raw/common-support τ AP区间均跨0且两端位于±0.01内。这约束了当前两次已训练模型的事件差异，不证明跨种子/其他网格原点稳定，也不构成总体等效检验。1% FPR下shift阈值=0.47906652，通过83个τ/75个背景；reference通过88/75。Shift的0.1% FPR通过15个τ/7个背景，尾部统计不足。τ electron/muon/hadronic在inclusive 1%下通过9/4/70个；分支及域内差值CI未测量。Pixel-centered域AP=0.33175445，large-scintillator域AP=0.38435797，空输入与tracker-only域仍只有prior型输出。

Shift程序wall=5,816.62953 s，实际allocation=5,824 s=1.61777778 GPUh，额外7.37047 s，peak=1,545.10254 MiB。同allocation预算5,823.27298 s内shift选epoch4，reference选epoch3（余347.04366 s），reference−shift ΔAP=−0.00155390；仅整轮点估计、无CI。

评价commit=`5f44e21`，正常退出：18 results（4 references+14科学模型）、484行分域、54个历史三类诊断、20 paired/25 pending、2 factorial。`part_grid_origin_comparison.json`保存本次训练/读回/配对/分域/核算快照；`part_grid_origin_paired_review.png`及`evaluation_final/part_1mm_shift_diagnostic_review.png`已实际查看。复现命令同上，读回experiment改为`part_1mm_shift_s42_e4_v2`；日志为`readback_part_1mm_shift.log`、`accounting_part_1mm_shift.log`、`evaluation_part_1mm_shift.log`。

07:07:06 UTC组织累计63.75366560 GPUh，H100视图27.51533227 GPUh，missing=0；正式14 complete、6 running、15 idle，ParticleNet fixed-graph已接续启动。Kubeflow v7仍Pending、无node。接下来完成余下21项科学配置；ParT两项粗格控制未提供τ正收益支持，局部关联/跨层/模态/训练域/多种子仍按冻结合同继续，最终仍需无`--allow-incomplete`的严格评价。

## 补充：ParticleNet OR 完成，两族 multiplicity 对照齐备

`1573782.13`正常结束，JobStatus=4、ExitCode=0、NumJobStarts=1，原节点`b9pgpun001.cern.ch`。07:20:47 UTC全部14项读回pass；ParticleNet dim64、161,156 parameters、pixel+bar、seed42、4 epochs、20,016 steps/epoch、H100/BF16，NLL选epoch4；80,064 applied、0 skips、320,164 exposures。OR信息处理与上文相同，全局汇总未恢复丢失的native count。

| Epoch | PN OR NLL | Tau AP | Cumulative program wall [s] |
| --- | ---: | ---: | ---: |
| 1 | 1.31932070 | 0.29077937 | 1,136.90129 |
| 2 | 1.26677756 | 0.33138767 | 2,253.02940 |
| 3 | 1.25774181 | 0.34698988 | 3,369.29729 |
| 4 | 1.24245326 | 0.35593261 | 4,506.13683 |

PN OR的τ AP=0.35593261、固定common-support AP=0.32412235、τ AUC=0.63141312、NLL=1.24245326、balanced accuracy=0.40220、macro F1=0.39945714、Brier=0.67542784。e/μ/τ/NC AP=0.59660299/0.41364948/0.35593261/0.37165539，argmax recall=0.5004/0.2948/0.5328/0.2808。OR的NLL和balanced accuracy点估计也优于PN count（1.24970777/0.39770），非τ配对CI未测量。

| PN count−OR metric | Difference [95% pointwise event CI] |
| --- | --- |
| Raw tau AP | −0.00289658 [−0.00956869,+0.00254101] |
| Fixed common-support tau AP | −0.00094003 [−0.00773349,+0.00441296] |
| Tau ROC AUC | −0.00303173 [−0.00937309,+0.00259286] |
| Tau efficiency at 1% FPR | −0.00040000 [−0.00898382,+0.00784358] |

相同400次、128 blocks、固定权重合同。PN与ParT的count−OR raw/common-support τ AP区间都跨0、上界都低于+0.01；两族均未显示count的正事件区间支持。两次模型族对照不是额外训练种子，不将一致的方向解释为已测跨种子稳定性。PN的raw区间下界距−0.01较近，400次bootstrap的边界位置不作为精确排除OR收益的依据。

PN OR在validation empirical 1% FPR下阈值=0.43944392，通过90个τ/75个背景；count通过89/75。0.1% FPR下OR通过15个τ/7个背景、count通过9/7，count−OR效率差=−0.0024、CI=[−0.00477869,+0.00200918]，尾部不足。OR的τ electron/muon/hadronic在inclusive 1%下通过7/6/77个，分支AP=0.08260038/0.06292380/0.29134561；分支差值CI未测量。pixel-centered AP=0.31133000，large-scintillator AP=0.39005589；空输入和tracker-only域仍仅有prior型输出。

程序wall=4,507.09343 s，allocation=4,515 s=1.25416667 GPUh，额外7.90657 s，peak=1,552.50732 MiB。同allocation预算4,514.04339 s内OR选epoch4、count选epoch3（余617.95981 s），count−OR τ ΔAP=−0.00718265；整轮点估计、CI未测量。OR/count没有改变occupied-cell token数量，不能把wall差异视为结构复杂度降低。

评价commit=`cd5f5c2`，正常退出：19 results（4 references+15科学模型）、511行分域、57个历史三类诊断、21 paired/24 pending、2 factorial。`pn_or_count_comparison.json`保存完整训练/读回/配对/分域/核算快照，`pn_or_count_paired_review.png`和`evaluation_final/pn_1mm_or_diagnostic_review.png`已查看。配对图新增第3页的单行图在默认密集刻度下文字拥挤，`s2.plots`将横轴主刻度限制为约5个；用已保存统计量重绘并实际查看，未重拟合或改变CI。复现使用上文命令、experiment改为`pn_1mm_or_s42_e4_v2`；日志为`readback_pn_1mm_or.log`、`accounting_pn_1mm_or.log`、`evaluation_pn_1mm_or.log`及`plots_coarse_controls.log`。

07:21:48 UTC累计65.22408182 GPUh，H100视图28.98574848 GPUh，missing=0。正式15 complete、6 running、14 idle，pixel-only控制已接续启动；Kubeflow v7仍Pending、无node。接下来完成ParticleNet平移、两族关联/图/跨层控制、模态/域控制与额外种子，共20项科学配置。S2整体尚未验收，calibration/test保持冻结。
