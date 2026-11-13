# S4：首项配对分类结果与前端等价验证

## 阶段目标

完成统一checkpoint验收和首项同事件配对统计，明确低标签增益的物理范围；检验删除重复前端分组是否保持输入完全一致，以减少GPU等待。

## 工作内容

CPU v3（cluster2198902，commit434efc3）已完成，实际80.659秒。4个完成的分类模型全部通过14项checkpoint合同，统一检查best/last、optimizer、resume状态、每epoch概率、完整validation keys和NLL选择。内部434次训练监测均通过，有有效节点的监测批次都保留正的输入梯度。另收集4个随机frozen probe的full/global结果，共12组固定10k validation预测，400次accepted-family connected-chunk配对bootstrap。完整矩阵仍有pending，本报告是进度结果。

前端`plane_tokens`此前调用`token_inputs`构造S2的4mm分组及排序，再丢弃groups并构造S3的plane/window分组。新增`build_blocks=False`仅跳过前一组未使用的分组/排序；S2默认路径保留，S3/S4继续以原规则建立canonical native channels、local blocks和regions。新回归比较pixel/bar/tracker、重合tracker坐标不同误差、输入置换、空事件、masked observations、不同window和signal/residual开关的全部输出，要求逐元素完全相等。

新增正式CPU诊断`frontend_profile`将在每类8个occupancy分位点（含两端）的原train事件上比较完整/.75 masked视图，检查输出数组及全部collated tensors位级相等，交替执行原/优化路径各3次。只测transform耗时，不冒充端到端训练吞吐量。

## 详细数据

| 初始化 | 标签数 | Macro AP | ντ AP | NLL | ντ效率@1% FPR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Scratch seed42 | 1,000 | 0.29720868 | 0.25953440 | 1.51618843 | 0.0092 |
| Scratch seed42 | 10,000 | 0.42710017 | 0.32129605 | 1.27706020 | 0.0260 |
| Scratch seed42 | 100,000 | 0.46976869 | 0.35133806 | 1.21380233 | 0.0300 |
| MAE3 seed42 fine-tuning | 1,000 | 0.36179163 | 0.28432114 | 1.40226192 | 0.0116 |

1k标签MAE3减Scratch的配对差额：

| 指标 | Raw差额 [95% CI] | Energy/vertex common-support差额 [95% CI] |
| --- | --- | --- |
| Macro AP | 0.06458296 [0.05791726, 0.07131453] | 0.05987638 [0.05206827, 0.06752863] |
| NLL | −0.11392652 [−0.12667733, −0.10258087] | −0.10829194 [−0.12306518, −0.09451014] |
| ντ AP | 0.02478674 [0.01404154, 0.03596568] | 0.01015166 [−0.00119118, 0.02224383] |
| ντ效率@1% FPR | 0.00240000 [−0.00464488, 0.00828932] | 0.00157698 [−0.00540699, 0.00690413] |
| ντ效率@0.1% FPR | −0.00040000 [−0.00203021, 0.00081800] | −0.00099912 [−0.00292303, 0.00077559] |

τ子道分别与全部非τ背景比较：

| 子道AP差额 | Raw [95% CI] | Common support [95% CI] |
| --- | --- | --- |
| τ→e | 0.00274487 [−0.00633536, 0.01145574] | −0.00212945 [−0.01288251, 0.00627659] |
| τ→μ | 0.00333151 [−0.00259544, 0.01054865] | 0.00163804 [−0.00450331, 0.00889055] |
| τ→had | 0.02588401 [0.01503268, 0.03963505] | 0.01247660 [0.00086970, 0.02639851] |

Common support保留8,559事件，103个supported cells，每类权重总和1,586，Neff=7,959.41。使用固定energy/vertex联合分格，非flux weights。以上为固定seed/权重条件下的逐项区间；训练seed变异、matched-compute增益和多重比较后结论未完成。

4个分类模型的`head_input_norm`跨监测RMS为0.90986210–0.99999428，`head.2`（FP32 SiLU）精确零比例均为0。1k MAE模型最小有效输入梯度平方范数为1.78353×10⁻⁶；1k Scratch为1.50993×10⁻⁸，发生于接近完全拟合的训练后期。其余10k/100k Scratch最小值为0.12030884/0.01837640。Head第一Linear的RMS可随训练增大，后续独立归一化与SiLU仍保持非零输出，不能仅据Linear幅值判定死Head。

结果根：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13/evaluation/progress_v3`。人工查看`label_efficiency_train_full.png`、`label_efficiency_probes_full.png`、`label_efficiency_probes_global.png`，标签与数值对应；单seed不画SD误差棒，两个seed的随机frozen结果显示seed SD。

12GB MIG候选提交成功：`s4-compute-small-a`，bigbird20 cluster1574593，配置/源代码与资源选择证据保存在`logs/submission_s4-compute-small-a.json`。共享原6项compute-recovery任务，获配及训练尚待。

## 验证与阶段性结论

修复模型在首项1k标签比较中有四分类AP/NLL改善，但ντ raw AP的改善在common support下减小且区间跨零，τ→e/μ与严格误识率端的增益尚未得到支持。τ→had的AP差额是需要独立seed和等预算继续检验的具体方向。低标签后期过拟合与死Head分别依据训练/validation曲线、激活和梯度判断。

LCG108 CUDA/Python3.12执行`python -m unittest tests.test_flavour_spatial tests.test_flavour_s3 tests.test_flavour_s3_repair tests.test_flavour_s4`，51项通过（13.648秒）；`frontend_profile.py`、提交器和成本脚本的Python编译检查通过。首次验证命令误写了不存在的`tests.test_flavour_s2`模块，该次其余42项实际通过；以上修正命令给出完整51项通过结果。真实事件测量尚未完成，暂不声称吞吐改善。

12GB MIG候选已于20:16:30 UTC在b9pgpun106.cern.ch启动完整EMA3；compact候选当时仍Idle。首次使用小切片确实提前推进了同一任务，内部训练/学习结果仍按原合同验收。

## 下一步计划

完成前端真实事件验证，再决定是否将相同输入的优化源码用于后续未启动训练。继续完整独立seed与MAE/EMA/Rel下游、matched-supervised、full/global frozen对照。所有模型完成后运行strict统计、资源终态和总成本验收。
