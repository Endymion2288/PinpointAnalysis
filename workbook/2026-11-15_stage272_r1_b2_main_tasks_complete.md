# R1：B2 三类主任务全部完成

## 阶段目标

核对B2最后两个真实T3模型，补齐三个seed的T3/T4四pass主任务。其后仅剩两个同allocation等时控制，完整统计仍按冻结合同执行。

## 工作内容

`final_b2_t3_s17/s2026`均完成。对两个新模型独立读取40k Tune预测及75k Train曝光，核对身份、标签、metadata、概率、四pass曝光、BA/NLL、逐类recall、raw confusion与五分支health，**40项通过、0项失败**。审计为`R1/audit/b2_t3_seeds_stage272.json`；NLL在float64归一化后用sklearn重算，与记录最大差1.69e−9。

## 详细数据

两者均为H100 NVL、eager BF16 encoder/FP32 head，分别allocation1582928.0/1582929.0；初始化fresh，selected T3 LR=.0001。每个非τ Train事件精确曝光4次，各e/μ/NC总曝光100k，τ为0。Tune仍保留10k τ用于污染评价。

| Model | BA3 (%) | NLL3 recorded | Recall e / mu / NC (%) | Updates | Operation wall (s) | Full Tune inference (s) |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| B2 T3 seed17 | 55.0867 | .9006997992 | 55.27 / 48.30 / 61.69 | 37,547 | 3,974.528058 | 242.180603 |
| B2 T3 seed42, existing | 55.1200 | .9009733511 | 56.90 / 50.30 / 58.16 | 37,550 | 3,554.083728 | 191.808719 |
| B2 T3 seed2026 | 55.0000 | .8982249127 | 55.30 / 50.50 / 59.20 | 37,544 | 3,312.764873 | 172.749496 |

新seed17/2026训练时间分别3,579.835445/3,047.264253 s，validation合计370.457051/259.946994 s，data wait73.620027/44.727650 s。各39项health samples，`native_local/native_global/native/tracker/head`均有finite正梯度及正参数更新，activation finite，预测均有38,918个不同向量。

B2真实T3三个seed平均BA3为55.0689%，T4 conditional为55.3911%，点差**−.3222 pp**；平均NLL3分别.8999660210/.8925342520，点差+.0074317690。逐类T3−T4 conditional recall点差e/μ/NC为−1.6300/−.8800/+1.5433 pp。这里T3/T4分别使用各自在screen选择的LR=.0001/.0003，不能把任务比较解释成只改变类别数的机制消融。区间尚未测量；等待统一400次统计。

07:00 UTC R1完成**39/41**模型。seed17 compute已约98k次曝光，seed2026 compute已500次updates、3,995次曝光；两者都实际优化。目标分别为本allocation fresh T4 wall **4,983.576084/4,183.187681 s**，以±5%作为最终预算要求，实际偏差尚未测量。

## 验证与产物

读回采用LCG108、单线程BLAS、`KEY/equal_identity_rows`及sklearn的`balanced_accuracy_score/log_loss/recall_score/confusion_matrix`。仅访问Train/Tune和已生成的模型记录；Cal-A/Cut-B/Confirm payload保持封存。完整原始审计保留每项判定与原始confusion。

```bash
condor_q -name bigbird20.cern.ch -constraint 'Owner == "xcheng"' \
  -af ClusterId JobStatus NumJobStarts RemoteHost
condor_tail -name bigbird20.cern.ch -maxbytes 16000 1582928
condor_tail -name bigbird20.cern.ch -maxbytes 16000 1582929
git diff --check
```

`R1_ROOT=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R1`。已有26项tests对应代码未改动，本阶段不重复执行。

## 阶段性结论

**事实：**所有R1四pass主任务、必要seed扩展和输入控制均已完成；新B2 T3身份、曝光及真实学习读回通过。当前B2四类条件三类平均点值优于真实T3，但存在逐类取舍与不同selected LR，不能声称类别数本身产生全部差异。

## 下一步计划

完成仅剩的两个compute模型后，唯一finish控制器执行完整41模型、133配对、21项mean-seed比较及400次physical-block统计。核对全部英文图，关闭allocation账本并运行增强accept；通过后导出最终结果，更新合作组交付及下一步路线。不启动R2训练。
