# S4：逐 batch 形状的计算预算重放

## 阶段目标

为 matched-compute supervised continuation 提供稳定预算，避免长尾 occupancy 使少量随机 batch 的成本估计失真。

## 工作内容与详细数据

12-batch CPU profile 的 MAE 每 step 运算量均值为 8.8885943×10^10，sample SE 为4.5759391×10^10；Rel对应为3.9141013×10^10，SE为1.4679429×10^10。该不确定性过大，不能据此宣称精确等计算量。

新增 `shape_operations`，从实际 ragged batch 形状计算同一个明确定义的 surrogate：Linear 和 attention 的 forward、所需 backward、eager local checkpoint recomputation。使用实际 local block数、region padding、query数、event region最大长度以及 kept bar local source长度；考虑输入是否需要gradient与frozen summary/teacher。没有将elementwise、normalization、编译优化和GPU型号等价包含进该量。

真实 forward/backward hooks 与shape计数在合成变长事件的MAE、EMA与classification三种实际模型上逐项总量相等，`test_shape_budget_matches_actual_forward_backward_hooks`通过。既有17项S4回归检查也通过。

新增正式CPU `budget_replay.py`，使用与训练完全相同的sampler、epoch seed、mask、native observation和group构建，跳过无关target查找和神经网络数值计算，累计完整MAE3及三个Scratch标签预算的训练矩阵运算proxy。3个dataloader workers+1主进程匹配4CPU请求。配置为`s4_budget_replay.json`。其数值目前未测量；完成后才冻结matched-budget继续监督的额外steps。该replay是训练成本分析，不重新认证输入样本或proxy。

## 阶段性结论

少量batch的成本样本不足以支持稳定budget matching。当前增加确定性全sampler重放；最终同时报告这个明确定义的surrogate与实际allocation时长，避免把共享GPU上各task elapsed直接相加。

## 下一步计划

执行CPU重放；按完整预训练预算冻结supervised continuation，完成剩余分类和frozen probes，汇总配对多seed结果及总资源成本。
