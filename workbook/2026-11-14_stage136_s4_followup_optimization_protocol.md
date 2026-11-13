# S4 结果驱动的下一轮优化协议

## 阶段目标

将已观察的低标签过拟合、global表示可读与τ决策退化转成可区分的训练优化问题。

## 工作内容与详细数据

在`roadmap/11_s4_pretraining.md`新增后续扩展协议，分别比较重复曝光/早期checkpoint、encoder学习率、readout正则及LP-FT。每个候选保留相同设置的Scratch对照，避免只优化预训练模型。超参数选择只使用原nested标签子集内的分层train-inner folds，标签仍计入该预算；不在现有validation拟合class bias、温度或正则参数。当前S4主矩阵保持冻结，扩展尚未执行，实测数值为“未测量”。

联网读取ICLR 2022《Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution》和CVPR 2022《Robust Fine-Tuning of Zero-Shot Models》原始来源页，URL、HTTP状态及原文存于输出根`literature/low_label_transfer_sources.json`。LP-FT提供可检验优化方案，但论文的OOD问题不能直接当作本任务过拟合病因；WiSE-FT所需的zero-shot分类Head当前不存在，未机械采用。此前已检索的FASERCal、Sonata/NOMAE CVPR2025仍负责主要预训练设计。

## 阶段性结论

当前证据支持优先处理训练曝光与readout优化，而不是继续压缩输入或仅凭τ AP增加宣称全指标改善。新方案的有效性尚未测量，下一轮需单独冻结并注明利用了当前S4观察。

## 下一步计划

先完成当前S4矩阵与100k独立seeds，依据最终质量/成本决定是否执行上述小规模扩展。继续保持calibration/test冻结和真实内部健康检查。
