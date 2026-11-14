# S6：完整readout覆盖与数值重放效应

## 阶段目标

核对全部readout事件和原nominal分数的一致性，分开数值重放残差与detector条件效应。

## 工作内容

29个任务合并，逐variant核对32000个原冻结test事件和keys。原test与重放保持同一BF16/compiled/batch64 policy，事件batch组成和并发backend不同；各模型四类概率最大差与固定0.003容差比较，不调整容差。

新增replay_metrics，用三seed与ensemble的冻结calibration/prior/cuts，量化同32000事件重放−原nominal的BA、NLL、τ效率及cut membership差；给小幅条件效应提供数值参照，不将其混入物理systematic区间。

## 详细数据

`predictions/variants_completion.json`：32000事件、14条件；三seed最大概率差分别0.00197327137、0.00061720610、0.00111556053，均低于0.003。全coverage通过，累计448000条条件响应。逐项paired统计和数值重放指标尚待读回。

模型/prior60/60，common-support9/9及对比已齐；全部实际transport与generator decay报告已完成。剩余是14条件统计、最后英文图、资源终态与清理。

## 阶段性结论

完整readout推理已完成并通过原冻结容差。数值残差对低统计cut的影响仍需直接列出，不能仅凭最大概率差推断所有选择计数不变。

## 下一步计划

运行14条件paired评价与replay_metrics，更新冻结最终结果表；读取所有图，核对原合同后关闭后端、压缩缓存并提交最终报告。
