# S4 最佳checkpoint的独立FP32 backward

## 阶段目标

在全部计划BF16训练监测之外，独立验证最终选中分类checkpoint的输入梯度、Head激活及事件间预测变化。

## 工作内容与详细数据

新增`s4.health.inspect`。每个完成模型按固定catalog顺序选取每类2个、原生占用1–3,000的非空validation事件，CPU FP32、eval模式、关闭compile，严格载入完整best checkpoint并做一次cross-entropy backward。记录实际event keys、输入梯度平方范数、Head各层激活和预测变化，不更新权重或选择阈值。配置、best epoch与selected NLL完全匹配的通过结果可以复用。

`s4.validation_job`在`inspect_best_health=true`时调用该检查，失败即保留诊断并终止评价。最终汇总要求每个训练模型同时具备训练期间健康记录、14项checkpoint读回和本独立best-checkpoint证据。未对frozen linear probes强行施加encoder梯度要求，其encoder按定义冻结。

6项测试通过，包含真实模型的正常FP32 backward，以及主动将分类两条logit输出权重置零后的反例：反例输入梯度为零、预测恒定，检查按预期失败。另验证仅checkpoint和训练健康标记仍不能代替best检查。正式样本的全矩阵FP32检查尚未测量，下一次CPU validation将执行，不在本报告写作已通过。

## 阶段性结论

验收能够区分“训练期间曾有健康采样”与“最终所选模型仍有连通路径”，故障反例已验证。该检查说明数值与梯度连通性，不直接证明每个分支均有独立物理信息或良好泛化。

## 下一步计划

剩余分类模型完成后开启独立best检查与统一评价；只对新完成或证据不匹配的模型重做该诊断。保留1k过拟合与τ argmax退化的事实，并以完整对照矩阵决定后续优化。
