# S4 修复后计算预算实测与配对合同

## 阶段目标

实际生成修复后监督continuation预算，冻结同事件比较，并继续核查frozen readout的概率误差来源。

## 工作内容

正式CPU计划1578771使用源码`e398254`完成，25.253704秒。复用同一nested-label subset及分组的既有shape表，重新计算固定监督prefix；三个修复Scratch parent均已complete。生成[三项训练配置](../roadmap/configs/s4_empty_supervised_matched.json)、[预算表](../roadmap/results/s4_empty_matched_budgets.json)、[评价模型](../roadmap/configs/s4_empty_matched_evaluation.json)、[12项配对](../roadmap/configs/s4_empty_matched_pairs.json)和[CPU评价任务](../roadmap/configs/s4_empty_matched_validation.json)。

评价包含六项seed42阶段145控制、三个新continuation及三个由parent/continuation最低validation NLL构成的selection envelopes。Envelope无论选中哪个checkpoint均计入全部parent及continuation计算。两种监督结果分别与MAE3比较；固定单seed的物理块区间不能替代训练seed重复。

修正训练manifest的一处描述：监督continuation确实加载完整parent及Head，不能写成随机重置Head；模型加载逻辑原本正确，已有回归检查逐参数验证。CPU等待逻辑现在同时等待train/probes，遇到`optimization_review_required`停止，不把未收敛probe的诊断预测纳入评价。

## 详细数据

新增训练的声明矩阵预算目标均为1,804,323,309,279,552，范围沿用阶段151的eager Linear/attention proxy，实际allocation另计。

| Labels | Epochs | Steps/epoch | New updates | New exposures | Relative matching error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 8 | 873 | 6,984 | 97,065 | +0.030651% |
| 10,000 | 8 | 742 | 5,936 | 93,613 | +0.069562% |
| 100,000 | 4 | 1,424 | 5,696 | 90,984 | +0.008342% |

三项均通过0.5%合同。原baseline曝光分别为57,043/158,245/399,696，全部计入对应总成本；新的分类收益尚未测量。

收敛修复probes已实际开始。1k full/global的L-BFGS迭代分别为Scratch 580/323、MAE3 732/414，无ConvergenceWarning、系数finite。其full/global validation NLL分别为3.676257/1.779222和3.247376/1.579259；求解收敛没有消除概率误差，不能据此断言表示无信息。当前full为1,440维、global为192维，仅使用1k fit标签。旧任务未保存fit NLL，真实过拟合程度尚未由fit/validation误差差额测量。

为后续定位，readout现在保存fit NLL、fit accuracy、参数范数和held-out NLL，并保存带Scaler及分类器的小型joblib最终模型；不保存大规模事件features。本次新字段尚未在正式任务上测量，不填推测值。固定C=1保持不变。

实际测试：等待未收敛probe立即停止、完整监督Head加载、legacy配置严格读回三项共0.378秒通过；真实probe求解器及fit-only标准化测试0.016秒通过。`git diff --check`通过。正式原probes保持冻结，不改写其历史manifest或预测。

03:03:50 UTC，1k/10k train-inner全部48项已complete；后续按既定规则解析10k共同配方。1k三seed重拟合当时完成4/6，其余2项仍运行，完整配对结果尚未测量。EMA/MAE5/channel Rel首批已正式提交：compact cluster2204965（bigbird21）及12GB cluster1578796（bigbird20），共享任务claims。AFS定期清理检查仅针对账本terminal且EOS有完全相同脚本副本的目录，本次没有待删终态目录，active/queued提交保留。

## 阶段性结论

事实：修复后的三个监督计算预算已生成且匹配误差均低于0.07%；1k线性probe的数值收敛问题已解决，但概率质量仍差。新的计算对照分类结果和fit/validation差额尚未测量。

待检验假设：1k的高维readout可能需要更强正则；需要train-inner选择与fit误差证据，不能直接在validation拟合C或把该问题归咎于encoder。

## 下一步计划

正式提交三项continuation和冻结CPU评价。解析已完成的10k OOF表，冻结六项固定末轮重拟合及同设置的三个SSL目标下游。保存inner各allocation终态并将Kubeflow唯一GPU单元接续到新任务。
