# S4：Rel 重建预算计数验证

## 阶段目标与工作内容

补充Rel双decoder路径的计算预算检查，清理shape计数中未使用的临时量。没有修改输入、目标、模型输出或正式运行配置。

## 验证

`test_shape_budget_matches_actual_forward_backward_hooks`现在同时覆盖MAE、MAE+Rel、EMA和classification。实测计数与实际forward/backward hooks总量相同，检查通过。完整sampler CPU budget replay已提交bigbird21 cluster2198174；尚未用抽样估算设置matched-compute结论。

## 阶段性结论与下一步

Rel新增decoder路径已纳入明确的矩阵运算proxy。继续等待完整重放与下游训练，按实测结果决定保留策略。
