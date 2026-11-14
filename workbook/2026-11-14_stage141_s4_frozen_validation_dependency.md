# 冻结重训矩阵的自动评价依赖

## 阶段目标

在训练持续执行时准备完整评价任务，使18项模型齐备后立即读回和评价，并在上游数值故障时停止。

## 工作内容与详细数据

`s4_empty_validation.json`继续固定18项模型、9项配对、3项三seed均值比较、400次128块bootstrap、完整10k validation、独立best FP32 backward及英文训练/健康图。新增最多7,200 s等待完整矩阵，15 s读取一次轻量manifest，不反复读取checkpoint；等待时manifest明确为`waiting_for_frozen_fits`。发现上游failed/internal_health_failed/learning_review_required即退出，不用部分模型通过strict评价。等待与CPU评价均不申请GPU，长等待使用workday队列，CPU allocation按真实时长计费。

截至本报告，10个完整训练并行，已完成超过31,524次更新的范围检查和639次计划内部采样，最大已记录preclip norm为170.440308。10k seed17/42前两轮出现部分NLL回升，仍按原固定最低validation NLL选择，不改变配方。以上是中期监测，最终18模型、工作点及三seed区间未测量。

## 阶段性结论

评价已定义为冻结训练的下游依赖，能够区分等待、上游失败与完整技术证据。数值健康与泛化质量分别记录，不因内部正常而跳过概率质量问题。

## 下一步计划

通过HTCondor提交单CPU等待/评价任务，全部模型完成后自动计算完整对照；回收GPU和刷新终态成本后再运行专用`accept_empty`。最终结果齐备后更新roadmap与中文阶段报告。
