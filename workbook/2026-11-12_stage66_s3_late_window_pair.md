# 阶段66：S3 late窗口配对与恢复checkpoint实证

## 阶段目标

用首个完成的冻结窗口配对检验fine-window在late fusion中的表现，报告四类差异和事件区间。记录第三个完成模型及实际恢复checkpoint。完整screen和三seed确认仍待完成。

## 工作内容

4 mm late、16-pitch late和16-pitch context-joint均完成四轮，各自80,064次optimizer更新、零AMP skips，14项CPU读回全部通过，共42项。只对完成的late窗口对执行冻结的400次paired bootstrap，共用128 accepted-family/chunk blocks、每类32个，保留原始score ties。Common-support权重沿用固定E/vertex定义。

失败任务`1573835.3`的唯一恢复作业`1573839.0`已在H100 NVL实际启动，manifest明确`resumed_from_epoch=1`。重新发布的正式epoch2 checkpoint已CPU读回，随后进入epoch3。未改模型、seed、样本或训练合同。

## 详细数据

结果根：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_2026-11-12`。

### 三个已完成模型，seed42

| Model | Macro AP | NLL | AP e | AP mu | AP tau | AP NC | Actual GPUh |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 mm late | 0.358741 | 1.330101 | 0.467282 | 0.360682 | 0.287407 | 0.319593 | 0.987500 |
| 16-pitch late | 0.338682 | 1.342968 | 0.432262 | 0.349015 | 0.282193 | 0.291258 | 1.230833 |
| 16-pitch context-joint | 0.332317 | 1.352187 | 0.412374 | 0.322616 | 0.285572 | 0.308705 | 1.519722 |

三者均以full-validation NLL选第四轮。真实Condor终态分别为`1573835.0/.1/.4`、JobStatus4/ExitCode0，allocation为3,555/4,431/5,471 s。10,000 validation事件的不同概率向量数分别9,561/9,559/9,412；这些完成模型未表现为所有事件同一概率向量。统计见`screen_progress_three.json`与每个模型的`final_checkpoint_checks.json`。

### 冻结配对：16-pitch late减4 mm late

区间为给定这两个已训练seed42模型的pointwise 95%物理事件区间，不包含训练seed波动。

| Metric | Raw delta | Raw CI95 | Common-support delta | Common-support CI95 |
| --- | ---: | --- | ---: | --- |
| Macro AP | −0.020059 | [−0.025400, −0.015138] | −0.019518 | [−0.025504, −0.013869] |
| NLL | +0.012867 | [+0.010110, +0.016279] | +0.012857 | [+0.009775, +0.016289] |
| Macro F1 | −0.016923 | [−0.024711, −0.009615] | −0.015644 | [−0.023975, −0.007521] |
| Balanced accuracy | −0.011000 | [−0.018229, −0.003865] | −0.010433 | [−0.018229, −0.002889] |
| AP e | −0.035020 | [−0.046312, −0.024768] | −0.029668 | [−0.041352, −0.019481] |
| AP mu | −0.011667 | [−0.018869, −0.005171] | −0.014116 | [−0.022565, −0.004875] |
| AP tau | −0.005214 | [−0.010356, +0.000036] | −0.004360 | [−0.010036, +0.001503] |
| AP NC | −0.028335 | [−0.040586, −0.015385] | −0.029928 | [−0.044884, −0.015185] |

所有34项指标及400个有效replicates保存在`paired_late_progress.json`，不是只计算表中八项。Raw每类2,500事件；common support为8,559个正权重事件、Neff7,959.413646535195，target Z未匹配。

本配对比较同一native输入上的两种局部窗口，并未比较native与1 mm。不能把它改写成native通道信息无用，或已证实4 mm抹平native优势。Context-joint与joint的冻结配对须等joint完成后另算，不能借上表的跨配置点值代替。

### 实际恢复与资源

`recovery_task3_epoch2_published.json`：正式checkpoint为epoch2、40,032 updates、0 AMP skips；模型与optimizer张量全部finite，`resumed_from_epoch=1`。这证明真实恢复已越过原失败位置。此前损坏的epoch2临时文件未被当作有效checkpoint使用。原失败2,363 s和重做计算全部保留计费。

2026-11-12 15:59:46 UTC账本：已分配8.654567526 GPUh，加入首次交互profile保守上界为8.835710139 GPUh，missing history0。该快照早于context-joint终态，属于进行中成本，不能当最终S3成本。Kubeflow Job仍Pending、没有节点；Condor在实际获得GPU的后端继续执行。没有根据观察超时重提任务或改变H100型号。

## 阶段性结论

事实：在当前seed42、late fusion、四轮和完整validation下，细化局部窗口使四分类macro AP、F1、balanced accuracy降低、NLL升高；macro AP的raw和common-support事件区间均在零以下。e、mu、NC的AP区间也在零以下；tau区间包含零。完整四分类指标揭示的差异比只看tau更明确。

未测量：该窗口配对的三seed训练波动，及窗口效果随其它优化设置的变化。待完成的主native/1 mm与2×2比较仍可能受已定位的分类头塌缩影响，必须结合预测与梯度诊断解释。

## 下一步计划

继续剩余14项screen的完成、恢复监视和CPU读回。全部17项真实终态后进行strict评价，再按预定Pareto规则冻结确认配置；不能根据这三个完成模型提前选胜者。最终完成四类、三seed、模态、条件域和资源比较，以及错误显示、中文科学报告和strict验收；S4–S6继续冻结。
