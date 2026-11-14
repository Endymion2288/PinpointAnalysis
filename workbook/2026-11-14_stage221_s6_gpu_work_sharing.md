# S6：test/paired 变体共享常驻 GPU

## 阶段目标

减少ROOT转换波次之间的GPU空档，保持冻结模型/precision/batch/阈值不变。

## 工作内容

Nominal inference优先领取已准备分块；没有可领取的nominal任务时，检查固定32k paired subset中已齐备的8-shard任务，并使用常驻的三个primary checkpoint执行14种读出变体。共享同一原子claim，nominal恢复可领取任务后优先继续。模型不重新载入、不改变BF16 compiled/batch64规则；仍无任务时短等待并释放GPU。变体nominal在全量完成时与正式test分数作独立readback。

## 详细数据

第一块test response已经转换；尚无正式test指标。Generator3512事件/220分块推理和400次独立block对照已完成，见stage220。新调度的吞吐及nominal replay差异尚未测量；原14条件真实输入及tokenization检查已通过，校准阶段相同GPU执行策略有13/13 checkpoint回放证据。

## 阶段性结论

共享work queue实现能够在CPU准备空档安排必要的paired实验，实际资源收益需正式作业计时。不额外增加模型/工作点选择。

## 下一步计划

同时使用可运行的HTCondor GPU与Kubeflow47GB MIG，完成全部nominal与paired分数，验证health/coverage/replay，再运行最终统计。
