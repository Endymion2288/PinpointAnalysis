# S6：真实transport条件效应与紧凑产物收尾

## 阶段目标

读回八个真实transport条件的配对效应，保留decay validity边界，并准备完成后清理中间文件。

## 工作内容

固定subset每类256，共1024事件；五个全类条件与三个τ helicity条件合计5888响应，全部以原主ensemble映射与cuts评价。Helicity条件复用不变的nominal非τ背景。400次配对block重采样；seed spread另报。

新增cleanup实现：只有分析合同齐备、完整backend queries、无active jobs、独立backend closure后才执行；先合并压缩task summaries并读回，再归档日志、删除native caches/重复分块和过时AFS提交目录。全量分数、merged truth、统计、bootstrap、图、source/config/G4可执行保留。尚未执行清理。

## 详细数据

Balanced nominal BA0.439453125，冻结1%背景cut的τ效率0.0390625。相对nominal：

| G4条件 | ΔBA | paired 95% CI | Δτ效率@1% | paired 95% CI |
| --- | ---: | --- | ---: | --- |
| density0.95 | −0.005859 | [−0.036825,0.022815] | 0.003906 | [−0.019557,0.026035] |
| density1.05 | 0.026367 | [−0.002174,0.051744] | 0.011719 | [−0.016529,0.042158] |
| field0.95 | −0.000977 | [−0.026427,0.026987] | −0.003906 | [−0.028000,0.018387] |
| field1.05 | −0.013672 | [−0.041087,0.012611] | 0.019531 | [−0.004549,0.044903] |
| helicity−1 | 0.009766 | [−0.006286,0.026004] | 0.015625 | [−0.011910,0.043178] |
| helicity0 | 0 | [0,0] | 0 | [0,0] |
| helicity+1 | 0.016602 | [0.000980,0.032529] | 0 | [−0.018607,0.022928] |

这是冻结proxy条件的pointwise区间，没有测得硬件误差幅度或真实polarization混合。不能把跨零区间解释成条件等价。Helicity+1的BA变化在当前pointwise区间内为正，但低背景τ效率未确定变化方向。

全部5888条compact truth响应逐值相同。τ有效decay vertex2046/2048条；同一事件run10083/chunk165/row950在nominal与helicity0的vertex validity为false，保留hadronic直接子粒子分类与validity。少量未定vertex不终止分析，不解释为无衰变；原因待查。

证据：EOS `evaluation/conditions/transport/*.json`、`transport/truth_completion.json`、`transport/decay_truth.parquet`。约15:02 UTC，模型/prior22/60项、common-support9/9项完成，readout20/29批仍执行中。

## 阶段性结论

真实field/density/polarization代理已完成冻结响应与统计。当前样本对大多数低背景τ效率变化的方向仍未确定；这些结论来自实际G4而非hit坐标伪变换。

## 下一步计划

合并common-support对照，完成全部test/prior、readout和migration；渲染英文图、核对最终物理表，再获取终态和清理产物。
