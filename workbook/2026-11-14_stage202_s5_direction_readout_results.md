# S5 完整训练误差与收敛方向读出

## 阶段目标

区分修复MLP的横向预测不足与encoder没有方向信息，记录实际train/validation证据。

## 工作内容

诊断复用Kubeflow当前47GB MIG，原六并发分配中出现空位后10:42:05 UTC启动，不增加allocation。读取最佳MLP的完整10k train与10k validation，得到1,440维fusion features；实际提取与FP64 ridge求解203.807529 s。完整分量readback最大差2.384186e-7，与既有selected-checkpoint预测一致。

Ridge使用train family folds选择逐分量正则化，两个分量均选10.0。最终系数在全部有效train标签重拟合。保留`regression_diagnostic/ridge_readout.npz`、完整validation预测与summary，没有保存大型feature bank。它是固定已训练encoder上的读出修复，不是重新训练encoder或独立encoder CV。

## 详细数据

| Readout / split | Valid events | px MSE | py MSE | px bias | py bias |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train constant / train | 9,971 | 0.028285827 | 0.027677612 | 0 | 0 |
| MLP / train | 9,971 | 0.027708222 | 0.027236771 | 0.005981473 | 0.007926359 |
| Ridge / train | 9,971 | 0.027844218 | 0.027311729 | ≈0 | ≈0 |
| Train constant / validation | 9,973 | 0.028805437 | 0.028573225 | — | — |
| MLP / validation | 9,973 | 0.028464897 | 0.028672704 | 0.002115391 | 0.010963514 |
| Ridge / validation | 9,973 | 0.028505082 | 0.028475051 | −0.003956925 | 0.002822821 |

单位为既定asinh/7 encoded target，bias为prediction−target。常数validation bias未单独测量。

Ridge px减constant MSE为−0.000300356，400次128-block CI95[−0.000375901,−0.000213775]；py为−0.000098174，CI[−0.000178426,−0.000018241]。相对改善约1.04%/0.34%。Validation相关为0.101877/0.061390，std0.015889/0.013275，远低于target0.169665/0.169016。

证据根：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s5_2026-11-14/regression_diagnostic/`。通过前一阶段10项单元测试与本次完整真实数据复算；本阶段没有修改模型训练计算。新增最终三seed比较图和train/readout图，以及小型JSON导出入口`analysis.flavour.s5.summarize`；finalize要求上述完整方向审查存在。

最终图人工检查发现横向误差图的细小小数刻度重叠，已改为百分比和最多4个主要刻度，并从同一summary单独重画；数值和模型不变。

## 阶段性结论

py并非整条通道失活：train上MLP优于constant，固定encoder也保留了少量可泛化的signed py信息。正则化读出减轻了MLP py偏置，获得小幅验证改善；没有测到强方向重建能力。低预测幅度、弱相关及整体角度误差仍须保留。Conditional-mean回归本身也可能对不确定方向产生收缩，但当前没有训练完整方向概率模型，不将这一假设当成已确认病因。

不据此宣布探测器没有横向信息；也不把ridge读出的小幅MSE收益称为已解决方位角重建。下一种vector/probabilistic head需要新合同、真实拟合和同预算验证。

## 下一步计划

完成剩余six-way三个seed的八轮确认，统一29模型配对评价并保存真实后端终态。最终报告把分类收益、逐分量回归改善及仍弱的方向重建分别表述，再合并冗余产物并提交。
