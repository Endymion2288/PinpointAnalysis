# S6：完整test统计与readout尾批恢复

## 阶段目标

读回主ensemble和同预算控制，保存后端失败证据，完成剩余readout条件。

## 工作内容

全部60项（15个模型/ensemble×4prior）完成，400个配对replicate逐项保留。主方法未按test重选。图表实际读回后调整τ效率显示范围，补各bin原始τ计数，避免0–1坐标压平低效率区间。

Kubeflow test Pod在全nominal写入之后、readout阶段终止，Job Failed/BackoffLimitExceeded；额外exec返回137，原Pod已GC。保留最后Running快照、Job终态、Event和六个claim owner，再只回收该已消失Pod的未完成任务2/3/4/5/8/9。用两lane×一loader的独立variants worker在Kubeflow与HTCondor恢复；所有完成分数和原冻结定义保留。

## 详细数据

完整test724441事件，主MAE三seed ensemble balanced：BA0.435225878 [0.433704723,0.436685364]，NLL1.201590409 [1.199946985,1.203085549]，macro AP0.475020253、macro AUROC0.705487288。冻结1%背景cut的τ效率0.035556301 [0.034210717,0.037062582]；0.1%cut为0.003111748 [0.002709101,0.003539862]。

同预算Scratch ensemble BA0.431872256、NLL1.209077236、τ效率@1%0.029241283。MAE−Scratch的paired差：BA0.003353622 [0.002507538,0.004166165]；NLL−0.007486826 [−0.007850644,−0.007094435]；τ效率0.006315019 [0.005285655,0.007292114]。

τ先验1%时，原冻结1%背景cut的τ效率0.027700662、纯度0.028099373；τ先验0.1%时效率0.027227798、纯度0.002867199。上述是固定prior scenario，不是真实物理flux。

失败Job终态2026-11-14T14:49:55Z；没有保留主容器精确termination/kill reason，单独exit137不能证明OOM。资源按实际PodScheduled到Job终态计保守上界，明确标记Pod endpoint未直接观测，包含这次失败。新watch使用无限request-timeout捕获尾批Pod终态。

## 阶段性结论

MAE相对同预算Scratch的改善有配对统计支持，但低背景τ效率仍低，稀有τprior下纯度很低；当前方法尚不足以作高纯τ选择。后端进程终止没有改变已经完成的test覆盖或冻结分析。Readout尾批和资源收尾尚未完成。

## 下一步计划

完成六个尾批，执行全32000事件×14条件nominal replay核对与paired统计；完成全部英文图、最终物理审查、backend关闭与compact cleanup。
