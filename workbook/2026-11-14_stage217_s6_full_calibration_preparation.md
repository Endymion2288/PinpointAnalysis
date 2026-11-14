# S6：完整校准准备完成

## 阶段目标

保持全部 accepted calibration 事件，完成最后一个转换分块；确认 transport/generator 最终合并不能漏掉条件或事件。

## 工作内容

最后 task129 的重试已完成，227个calibration分块全部准备完毕。补充 shifted prediction 合并的逐条件 key/count 核对。τ helicity只重transport原始CC τ事件；评价时以不变的nominal非τ事件补齐固定背景分母，不把只有τ信号的子集当完整四类样本。

## 详细数据

输入每块的raw bar记录量由metadata检查：task129为35,682,669，其他分块中位数36,258,184；该块没有明显更高的记录量，不能把长时间归因为异常占据度。重试实际运行中CPU正常，原作业根因仍未定位。准确转换时间/内存/字节写入 `shards/calibration_0129/summary.json`，全量合并时汇总。

前226块的13模型推理已完成并内部health通过；task129尚待补推理，所以temperature、阈值和最终test尚未测量。Generator已有114/220个nominal小批完成；精确最终计数按completion读取。

## 阶段性结论

全部calibration输入准备完成，没有删除慢事件或改变split。下一步可以补齐正式分数并完成冻结。

## 下一步计划

补推task129；全量calibration合并及15个温度拟合后检查数值、保存freeze，立即开放既有test campaign。
