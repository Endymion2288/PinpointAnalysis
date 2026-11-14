# S6：最终分析冻结与独立 test 开放

## 阶段目标

完成全部校准与固定工作点，冻结后开放既有独立test，不用test选择方法或阈值。

## 工作内容

全部722,272 calibration事件、13 checkpoints与2 ensembles完成scalar temperature拟合。每个mapping用既有完整10,000 validation及指定prior确定四类1%/0.1%背景工作点，保留完整tie并采用inclusive >=。Nominal calibration/threshold和全部conditions/statistics/evaluation定义写入EOS `analysis_freeze.json`。

## 详细数据

冻结时间2026-11-14 13:50:23.738Z，之前未读取test响应。15个T均在[0.05,20]内部，范围1.007143至1.163542，NLL全部下降。主MAE ensemble T=1.007143，balanced NLL=1.202468；Scratch ensemble T=1.012531，NLL=1.209935。所有mapping的validation事件数10,000，全部各类阈值的weighted FPR不超过目标。完整温度表与主阈值见 `roadmap/results/s6_calibration_freeze.json`。

独立test为724,441事件，CC e86,905、CC μ416,305、CC τ65,558、NC155,673。已按原227分块提交转换，HTCondor1582724；CPU2/6000MiB、max_materialize64，实时队列选择bigbird20。Transport输入准备1582725；Generator3512响应的三主seed推理申请Kubeflow47GB MIG，`pinpoint-s6-generator47`。提交不是完成，test分类与条件影响尚未测量。

## 阶段性结论

Calibration和validation工作点已经冻结。正式test及条件评价可以按相同流程执行，后续失败恢复不改变候选、数据或cuts。

## 下一步计划

转换/推理/观测质量并行；完成32k paired读出、1024事件真实field/material/helicity、Standard/NUX对照、400次block intervals、score/energy response和资源记账，然后逐项审查S6合同。
