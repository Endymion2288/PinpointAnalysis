# S4 完整矩阵的证据汇总与验收入口

## 阶段目标

把完整矩阵、模型内部健康、有效预训练和资源终态纳入同一可复核入口，推进三个独立MAE seeds的统一评价。

## 工作内容与详细数据

新增`analysis.flavour.s4.accept`，按冻结campaign及comparison文件构造预期清单。本轮要求33个full fine-tuning模型、30个frozen full/global probes、3个监督继续训练selection envelopes，共96组validation预测、90个单配对和9个seed均值配对。额外从`configs/s4_core_wave.json`纳入最初MAE3 seed42的预训练定义。

验收分别读取14项checkpoint checks、全部计划健康采样的step序列、informative input梯度、有限激活和SiLU非整层全零证据。Frozen probes核对fit/validation计数、full/global特征有限性及各自预测。完整SSL逐项检查integrity与measured-learning，不能以checkpoint存在替代学习证据。完整10k路径诊断要求原batch形状重放、presence保持和四个指定模型。

评价清单同时要求raw/common-support及τ e/μ/had配对、四分类1%/0.1% FPR实际MC计数、400次/128物理块bootstrap。全部allocation须有终态，组织成本不得超过60 device-hours；模型指标是否晋级另行依据实际区间解释。低标签NLL高于常数先验或不超过其他模型，会作为科学结果和训练设置的局限记录，不伪装成改善。

7项单元测试通过，包括新验收的4项和成本的3项。测试验证：健康标记不能代替非零梯度和完整采样；仅checkpoint不能通过；缺少global probe、selection envelope、配对或资源终态时不能通过。真实progress_v6上的只读核对成功执行；7个完整预训练的学习读回均通过，部分下游和allocation终态尚缺，汇总正确保持未完成。

MAE3的1k seeds17/2026已完成，best validation NLL为1.43857161/1.50928232。v7正式CPU读回提交为1578086（bigbird20，冻结9b2b78c），22:43:56 UTC开始运行，使用既定400次bootstrap及完整10k validation。v7配对结果尚未完成，不据此先报三seed平均收益。

另外提交缓存EMA worker2204028（bigbird21，冻结9b2b78c，24GB及以上兼容pool，2并发），由提交入口实时选择最低idle的兼容schedd。22:47的better-analyze显示0个立即愿运行槽位、50个drained后可匹配槽位；未更改运行中的worker。AFS仅放小型wrapper/submit，实际日志与数据均在EOS。

## 阶段性结论

预训练学习和完整路径的既有证据能够读回；S4总体尚未完成。当前有明确的缺项清单，不能由部分seed、部分标签预算或任务退出码替代完整结果。

## 下一步计划

收取v7的三seed低标签区间和新增Rel100k/matched/frozen结果，继续EMA、MAE5和Rel下游。矩阵完成后用strict validation及新验收入口汇总，检查英文图表并记录资源终态，再决定是否扩1M。
