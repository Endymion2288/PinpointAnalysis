# R1：B3 等时控制完成及独立学习读回

## 阶段目标

完成最后一项B3等时控制，核对同allocation预算、真实预测、训练曝光和所有分支健康。其结果补齐B3交付，不改变冻结的四pass排名；完整R1仍等待B2两个扩展seed。

## 工作内容

`compute_b3_t3_s42`已实际完成，作业1582880终态JobStatus4、Exit0。对同allocation内B3 T3 main、T4 main和compute T3三个模型，独立读取Tune预测及Train曝光，重新用sklearn计算共同BA3、NLL3和逐类recall，并核对五个分支的梯度及参数更新。

完整检查位于`R1/audit/b3_complete_stage268.json`；[紧凑结果](../roadmap/v2/r1_b3_compute_control.json)保留51项检查结果、真实计数及点估计差值。没有重新训练或新推理，也未运行新的bootstrap。

## 详细数据

三个模型均在A100、allocation1582880.0、BF16 encoder/FP32 head上运行；T4的三类指标先在非τ上条件化，三者共用相同30k非τ Tune事件。

| Model | BA3 (%) | NLL3 recorded | Recall (%) e / mu / NC | Unique Train rows | Train exposures | Operation wall (s) |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| B3 T3 four passes | 55.060 | .8973010022 | 59.68 / 48.21 / 57.29 | 75,000 | 300,000 | 7,660.921688 |
| B3 T4 four passes | 55.290 | .8930369420 | 60.80 / 48.14 / 56.93 | 100,000 | 400,000 | 9,510.193230 |
| B3 T3 matched compute | 55.370 | .8898837578 | 56.84 / 55.47 / 53.80 | 75,000 | 383,610 | 9,495.205675 |

Compute目标为同allocation fresh T4的9,510.193230 s，实际相对误差**−.1575946%**，通过±5%预算要求。最终Tune推理耗时377.506873 s，预算预留395.961716 s。Compute执行48,014次updates，每个Train事件曝光5–6次；main T3/T4分别37,550/50,051 updates，逐事件均精确曝光4次。

Compute−main T3的BA3点差**+.310 pp**、NLL3 −.0074172443，e/μ/NC recall分别−2.84/+7.26/−3.49 pp；compute−T4 conditional为BA3 +.080 pp、NLL3 −.0031531842，逐类−3.96/+7.33/−3.13 pp。区间尚未测量，将由最终统一400次配对统计给出；这些点差不代表训练seed总体或已确认的优势。预算及schedule一起改变，不能单独归因于重复曝光。

三个模型的health samples分别39/52/50；native_local、native_global、native、tracker、head均有finite正梯度及正参数更新，activations finite。Compute预测有38,918个不同概率向量，三类std约.2671/.2141/.1991。不是常数预测或只靠checkpoint完成的学习失败结果。

独立BA3及recall与manifest一致；NLL3最大绝对差**8.04e−10**，来自独立归一化/实现数值差异。Train/Tune身份、标签、概率有限且归一化、曝光总数、同allocation、目标预算、四pass精确曝光及终态检查共**51项通过、0项失败**。

1582880完整allocation为**26,802 s**、4 CPU，包含本allocation fresh T4、T3与compute；不能只计最后compute的wall。05:19:23 UTC记账为**40.163263 allocation device-hours**、278.908813 allocated CPU core-hours，剩两项B2 GPU作业，无Idle候选，成本尚未关闭。异构device-hours不是FLOP等效。

## 验证命令与产物

```bash
condor_history -name bigbird20.cern.ch 1582880 -limit 1 -json \
  -attributes ClusterId,JobStatus,ExitCode,RemoteWallClockTime,NumJobStarts,RequestCpus
python -m analysis.flavour.v2.r1.accounting --root "$R1_ROOT" --cached-kubeflow
git diff --check
```

原始读回使用现有`KEY/equal_identity_rows`及sklearn的`balanced_accuracy_score/log_loss/recall_score`，并从`train_exposures.parquet`重新求和。JSON与报告点差经独立算术核对。相关验收代码的26 tests已在stage267通过，本阶段只增加实测报告与紧凑结果，没有无理由重跑测试。

`R1`为`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R1`。所有B3权重与预测留在对应`experiments/`目录；Cal-A/Cut-B/Confirm payload保持封存。

## 阶段性结论

**事实：**B3等时控制完整完成、预算对齐且真实学习有效；比四pass三类有小幅BA点增益及更低NLL，同时仍有逐类取舍。R1现为**35/41**模型完成，B3已无待训练交付。

**未完成：**两个B2扩展seed各自T4、T3与compute共六个模型；完整41模型/133配对/21项平均配对、最终英文图表、独立复核、终态资源及accept。B3点估计没有改变原四pass排名和扩展选择。

## 下一步计划

继续使用当前两个已获配H100推进B2；全部模型完成后由唯一finish控制器执行完整评价及复核。按stage267增强的完整合同验收，实际全部通过后导出最终结果、更新合作组对照与路线状态并提交最终报告。R2不启动训练。
