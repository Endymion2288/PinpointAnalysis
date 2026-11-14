# S4 修复后等计算量监督对照的完整结果

## 阶段目标

检验增加监督训练计算是否能复现 MAE3 收益，同时检查低标签长训练恶化的程序与学习原因。

## 工作内容

三个 seed42 continuation 均完成，使用阶段145修复后的 Scratch parent（含原Head），没有冻结/解冻阶段。正式 CPU 评价 cluster1578799 已完成，九个真实训练模型通过readback和独立FP32，含三个监督selection envelope共12组预测，全部覆盖原10k validation；12项配对均完成400次128物理块bootstrap。预算及矩阵操作定义见阶段153，完整统计位于输出根 `evaluation/empty_matched/summary.json`，[精简实测](../roadmap/results/s4_empty_matched_summary.json)保存结果、历史和健康范围。

## 详细数据

| Labels | Scratch macro AP / NLL | MAE3 macro AP / NLL | Extra supervised macro AP / NLL |
| --- | --- | --- | --- |
| 1k | 0.296908 / 1.535145 | 0.359192 / 1.399620 | 0.278105 / 2.947359 |
| 10k | 0.427734 / 1.277739 | 0.443739 / 1.253412 | 0.407188 / 1.304645 |
| 100k | 0.468453 / 1.214940 | 0.473777 / 1.207316 | 0.468609 / 1.215594 |

额外监督相对MAE3的macro AP差及条件性95%物理块区间，依次为 −0.081087 [−0.089387, −0.073402]、−0.036551 [−0.043889, −0.028954]、−0.005168 [−0.008222, −0.001843]。所有预算的监督envelope均选择原Scratch parent；其成本仍包括全部continuation。100k额外监督相对原Scratch的macro AP变化仅+0.000157 [−0.001620, +0.002253]，NLL变化+0.000654 [−0.001355, +0.002524]。

1k长训练的training CE从第1轮0.542303降到第8轮0.044685，validation NLL却从2.947359升到6.431765；已按validation-NLL选择最早的第1轮，仍比parent恶化。各类原标签数均250，该训练CE不含类别先验不平衡。读回的10k预测有9,719个不同概率向量，class probability std约0.277–0.356，最大概率接近0.99995；不是常量预测。101项训练采样均finite/healthy，Head SiLU RMS为0.349540–0.852977、精确零比例最大0，独立FP32的input梯度正常。逐步preclip梯度检查6,984次均通过，最大norm152.605804；晚期部分训练批次input梯度平方范数极小至1.69e-12，与训练CE接近零同时出现，保留作优化饱和记录。

这些证据支持长时间重复1k训练产生严重泛化间隙和过度置信；没有重现此前全前端冻结故障或Head全零。尚未测量该checkpoint在train上固定eval-mode NLL、logit temperature拟合、系统变化的影响，因此不将训练CE当作精确train-eval NLL。正在重训的1k train-inner LP-FT使用8次完整pass而非近10万次重复曝光，是本轮针对低标签过拟合的既定优化。

低FPR必须结合实际计数：三个continuation在τ的1% FPR下分别保留28/38/77个τ及75个背景；0.1% FPR下分别4/5/14个τ及7个背景。后者背景统计不足，不据此声称稳定尾部改进。Common-support保留8,559事件，四类权重和均1,586，Neff=7,959.414，完整结果保留raw/common-support与τ子道，不能代替真实通量加权。

英文图表位于输出根 `evaluation/empty_matched/training_curves_n1000.png`、`training_curves_n10000.png`、`training_curves_n100000.png`、`classification_internal_health.pdf`及`label_efficiency.png`，不纳入Git。

另以新增参数差分检查核对六个小checkpoint，见[实测](../roadmap/results/s4_parameter_audit_regression.json)：两个旧故障1k LP refit的best/last input/local改变均严格为0；两个正常10k early refit为正。新LP inner第12轮Scratch的input/local最大改变为0.002280809/0.003081903，MAE3为0.002477829/0.003383529。Scratch该折通用best位于第2轮，故其前端按LP设计未变；OOF resolver仅使用第3轮及以后的history，最终refit固定第8轮，不使用该通用best。不能把内折第2轮的有意冻结误写为修复失败。

## 阶段性结论

事实：在这一固定监督continuation配方和seed42下，匹配所声明MAE3额外矩阵操作的长训练没有复现MAE3的收益；1k/10k出现明显过拟合，100k改善区间跨零。该结果不覆盖所有可能的监督正则化或学习率方案，也不构成跨硬件实测FLOPs等效结论。当前validation已多次用于研究，区间条件于该权重与当前选择，不能替代训练seed重复或最终test。

## 下一步计划

继续修复后LP-FT三seed与10k新OOF选择，完成EMA/MAE5/Rel及frozen readouts。汇总所有有效分支后，分别判断概率质量、类别判别和τ尾部，并累计实际allocation成本。
