# 阶段60：S3首完整epoch与双后端实际获配调度

## 阶段目标

确认正式训练完成首个full-validation epoch并报告完整四类状态，持续按实际获配速度推进全部screen。S3尚未完成。

## 工作内容

17项screen已经真实提交。5项目前在HTCondor运行，其他任务继续排队。Kubeflow `pinpoint-s3-anyqueue-slot-v1`已通过server Pod admission，申请整卡H100 NVL、46 CPU、127G RAM，写作时仍Pending约6分钟。`dispatch watch`进程已启动，每15秒检查真实runtime readiness；只在Kubeflow CUDA/BF16通过后交接Condor idle任务。此前probe与固定主模型Kubeflow尝试均以主动缩短deadline结束，终态证据和计费保留，不创建双writer。

为了先获得预定主模型及关键对照，将本批Condor主task5的JobPrio设为20、1 mm task6设为10、tracker task14设为5。Running任务保持原地。实际成功命令是`condor_prio -n bigbird20.cern.ch -p <value> <job>`；`-name`在该客户端不被识别，首次尝试未改变优先级，随后用正确语法执行并查询确认。具体记录`priority_main_controls.json`。

同步roadmap总览与执行表，移除“S3未启动”和仅凭τ工作点晋级的旧状态；四类合同与S3独立120 GPUh上限保持阶段56以来的冻结值。现有bar proxy按用户接受值使用，time和拟合q/p在当前S3输入中不存在。

## 详细数据

输入为冻结80,000 train、10,000 validation，每类validation 2,500。结果根：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_2026-11-12`。

`s3_a4_late_s42_e4`首轮训练829.709871 s、80,040 exposures、96.467455 events/s；验证71.959266 s，累计程序wall911.525927 s。完成20,016/20,016 attempted/applied optimizer updates，AMP skips=0。四轮训练仍在继续。

| First-epoch metric | Value |
| --- | ---: |
| Four-class macro AP | 0.2940118071 |
| Four-class macro AUROC | 0.5531329267 |
| Macro F1 | 0.2320735214 |
| Balanced accuracy | 0.2825000000 |
| NLL (float64 evaluation) | 1.3770788120 |
| Four-class Brier sum | 0.7454048692 |

| Class | AP | AUROC | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CC nu e | 0.319723 | 0.561579 | 0.313151 | 0.1924 | 0.238355 |
| CC nu mu | 0.305866 | 0.569472 | 0.275092 | 0.6896 | 0.393293 |
| CC nu tau | 0.274521 | 0.532667 | 0.284178 | 0.2220 | 0.249270 |
| NC | 0.275937 | 0.548813 | 0.266393 | 0.0260 | 0.047376 |

混淆矩阵行是真实e/mu/tau/NC，列为argmax预测同序：

```text
 481 1531 418 70
 282 1724 443 51
 298 1589 555 58
 475 1423 537 65
```

μ预测占62.67%，NC recall为2.6%。这是该控制模型第1/4轮的弱分类状态，不是全screen或最终主模型结果。未据此删改配置或停止分支。

数据来自`experiments/s3_a4_late_s42_e4/history.json`和首轮`epoch_predictions.parquet`，使用S3 MetricCache独立计算，完整值含各类1%/0.1%工作点保存于`first_epoch_four_class.json`。NLL相对history差约3.2e-11，来自float64计算，未改变checkpoint选择。读取全validation的步骤只做预测结果评价，不重新认证split或proxy。

14:35:00 UTC计费快照：实际后端GPU分配0.9015824045 GPUh，叠加初次T4 profile保守上界后组织上界1.0827250179 GPUh，missing history=0，剩余预算下界118.9172749821 GPUh。这是运行中快照，不是终态成本；S2成本独立保留。

## 阶段性结论

事实：首个正式完整epoch训练/验证/epoch预测写回正常，20,016 updates全部应用。当前该控制模型偏向μ预测，NC识别很弱。程序未出现AMP skips或非有限数值。

假设：这种偏向可能随后续训练改变，或反映a4 late表征/优化的限制；尚无证据区分。用全部4轮、局部窗口/融合控制和三seed确认回答，不以首轮表现归因于原生硬件信息无用。

## 下一步计划

继续全部17项至真实终态；两后端按实际获配交接idle任务。完成14项CPU读回和完整四类screen/bootstrap/2×2交互后选两个候选，独立冻结确认配置并追加三seed。最终完成条件域、错误显示、资源Pareto、逐类晋级判断和严格输出门槛。calibration/test仍冻结，S4–S6未执行。
