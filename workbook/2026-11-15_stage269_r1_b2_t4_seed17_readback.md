# R1：B2 seed17 四类完成及独立读回

## 阶段目标

核对新完成的B2 seed17四类任务是否使用正确事件、完整四pass曝光并实现有效学习。继续现有两项B2训练，保持冻结排名和完整交付合同。

## 工作内容

`final_b2_t4_s17`在H100 NVL、allocation1582928.0完成。独立读取40k Tune预测及100k Train曝光，使用sklearn重算BA、NLL、recall与raw confusion；从52项health记录核对五个分支。该allocation已继续执行真实T3训练，随后执行同allocation等时控制。

审计产物为`R1/audit/b2_t4_s17_stage269.json`，**24项检查通过、0项失败**。本次没有增加推理或bootstrap，没有读取Cal-A/Cut-B/Confirm payload。

## 详细数据

| Item | Measured value |
| --- | ---: |
| Actual precision | BF16 encoder / FP32 head |
| Operation wall | 4,983.576084 s |
| Training time | 4,601.953503 s |
| Validation time, including full inference | 369.841211 s |
| Full Tune inference | 241.689980 s |
| Data wait | 76.722344 s |
| Updates | 50,046 |
| Unique Train events | 100,000 |
| Train exposures | 400,000; each event exactly 4 |
| Exposures e / mu / tau / NC | 100,000 / 100,000 / 100,000 / 100,000 |
| BA3, conditional on non-tau | 55.170% |
| NLL3 | .8952693855414487 |
| Recall3 e / mu / NC | 52.48% / 53.38% / 59.65% |
| BA4 | 43.295% |
| NLL4 | 1.2169480848244445 |
| Tau efficiency at 1% / 0.1% FPR | 3.15% / .51% |

BA与逐类recall独立重算一致，NLL最大绝对差1.34e−15。sklearn对存储的float32概率转换为float64后提示归一化精度；原始概率均finite、位于[0,1]且总和通过2e−6容差，独立NLL与记录一致。Tune身份、标签、几何域和occupancy metadata与冻结catalog一致；Train身份与逐类曝光也完全一致。

`native_local/native_global/native/tracker/head`五分支均有finite正梯度及正参数更新，activation均finite；预测有38,918个不同概率向量，四类概率std约.2360/.1692/.0927/.1644。没有常数预测或上游零梯度证据。

本seed的B1/B4 T4 conditional BA3分别为56.2633%/54.2533%；新B2点值位于两者之间。配对区间尚未测量，不能据此改变冻结的seed42四pass排名，亦不能用单seed点差替代最终三seed结论。

05:38 UTC已完成**36/41**模型。1582928的T3 tail已有2,500次updates和19,974次曝光；1582929的T4已进入第三pass，32,750次updates及261,769次曝光。两项任务均实际优化中。仍缺B2 seed17的T3/compute，以及seed2026的T4/T3/compute。

## 验证与产物

使用LCG108、单线程BLAS，读取`catalog.parquet`、`predictions.parquet`、`train_exposures.parquet`与`manifest.json`；身份检查复用`KEY/equal_identity_rows`，指标独立使用sklearn的`balanced_accuracy_score/log_loss/recall_score/confusion_matrix`。原始检查的数值与判定保存在审计JSON中。作业进度读取命令：

```bash
condor_q -name bigbird20.cern.ch -constraint 'Owner == "xcheng"' \
  -af ClusterId JobStatus NumJobStarts RemoteHost
condor_tail -name bigbird20.cern.ch -maxbytes 1700 1582928
condor_tail -name bigbird20.cern.ch -maxbytes 1700 1582929
git diff --check
```

`R1`为`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R1`。本阶段只更新实测记录与路线状态，已有26项R1 tests不因文档更新重复执行。

## 阶段性结论

**事实：**新完成B2四类模型的身份、曝光、指标及分支健康均通过独立读回；其conditional BA3为55.170%。R1尚未完成全部矩阵、最终统计及accept，不以本次读回替代最终验收。

## 下一步计划

保留两个正在运行的H100 allocation完成剩余五模型。唯一finish控制器继续等待完整41模型后执行400次配对统计、完整独立复核与英文图表；所有allocation终态后更新成本并执行增强验收。R2仅保留设计，不启动训练。
