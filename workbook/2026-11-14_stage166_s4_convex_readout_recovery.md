# S4 100k frozen readout 求解失败与同目标修复

## 阶段目标

解决固定C=1读出未收敛问题，保持表示与正则化目标的含义，并释放CPU求解期间的GPU allocation。

## 工作内容

两项100k `s4_converged_frozen_{scratch|mae3}_n100000_s42` 已结束，full分支均触及5,000次L-BFGS迭代上限，status为`optimization_review_required`，不能进入完整readability结论。global分支分别在266/1,789次迭代收敛。原CPU评价2204995因上游需审查而exit1，真实history确认，未得出完整readability结果。

采用CPU float64 Newton-CG求解相同目标：

`mean multinomial NLL + ||W||²/(2 C n)`，C=1，intercept不正则化。

保留原fit-only StandardScaler输出，包括float32舍入，再转float64。保存实际正则化目标值、系数与intercept梯度无穷范数；新结果必须无ConvergenceWarning，且实测梯度不超过1e-6（数值余量1e-10）。这改变数值求解方法，不按validation调C或feature。新full/global读出均使用Newton-CG，max_iter=200，BLAS两线程。

先运行[正式CPU solver gate](../roadmap/configs/s4_newton_solver_gate.json)：Scratch/MAE3各1k原train事件，full/global共四项，在相同features和C=1上比较L-BFGS参考与Newton-CG。预先要求参考梯度<=1e-8、Newton-CG<=1e-6、正则化目标差<=1e-5、fit概率最大差<=0.005，双方均无收敛告警。gate通过前不启动大规模恢复。

后续[五项GPU feature bank](../roadmap/configs/s4_newton_feature_banks.json)分别对五种初始化提取一次100k train和10k validation的full/global表示；[15项读出](../roadmap/configs/s4_newton_readouts.json)由[五组CPU任务](../roadmap/configs/s4_newton_cpu_groups.json)拟合1k/10k/100k嵌套标签，按精确事件键选择行、各自在本预算内拟合scaler。同一encoder的三个预算共用冻结表示，GPU阶段完全不拟合readout。暂存数组总量约3.59GB，完整CPU读回与评价后删除数组，保留小型模型、预测和provenance。单独缓存也允许后续只重算CPU优化，避免反复提取原始观测。

## 详细数据与文献

full输入维数1,440，global为192；此前500次L-BFGS已失败，本轮5,000次仍不足，不能用完成迭代或finite系数替代收敛。当前环境scikit-learn为1.5.2。2026-11-14联网读取[匹配1.5文档](https://scikit-learn.org/1.5/modules/linear_model.html#logistic-regression)、[1.5.2 Newton-CG实现](https://github.com/scikit-learn/scikit-learn/blob/1.5.2/sklearn/utils/optimize.py)及[JMLR 2008: Trust Region Newton Method for Large-Scale Logistic Regression](https://www.jmlr.org/papers/v9/lin08b.html)。TRON提供大规模logistic利用曲率与Hessian-vector乘积的依据，本实现采用sklearn的line-search Newton-CG。源码确认其终止条件为梯度无穷范数，另外独立重算同一目标的梯度。来源、字节数和读取UTC保存于输出根`literature/frozen_solver_sources.json`。

九项针对性CPU测试实际0.443秒通过，覆盖两种求解器在相关features上的同目标解、迭代不足拒绝、nested-row顺序与错误标签拒绝、收敛和catalog读回，以及四种CPU stage不计GPU时长。三个新增执行模块通过语法编译。随后完整相关58项测试实际14.903秒全部通过。真实solver gate、100k新驻点和新readout分类指标尚未测量。

冻结[完整目标评价](../roadmap/configs/s4_transfer4_objectives_evaluation.json)、[66项配对](../roadmap/configs/s4_transfer4_objectives_pairs.json)及[CPU任务](../roadmap/configs/s4_transfer4_objectives_validation.json)，包括15组seed42 fine-tuning和30组frozen full/global预测。全部以原10k validation及400次物理块bootstrap评价，Rel额外模拟监督和fine-tuning与frozen整体协议差别写入scope。

另外[修复后完整inner健康汇总](../roadmap/results/s4_transfer4_inner_health.json)记录48 fits、137,088次更新、2,112,000次曝光、1,732次健康采样（含148次有意LP采样），全部informative FT的input/local梯度为正，最大采样preclip norm为223.817445。与阶段154旧汇总相比，该表已替换12项故障LP结果。

本轮没有修改正在执行的旧campaign。其结果按原配置保存；未收敛者诊断用途，新的恢复名称独立。所有后端成本继续累计同一60 device-hours；04:09:39 UTC已用31.503981，剩余28.496019，CPU357.624231 core-hours。

## 阶段性结论

事实：已确认100k full readout的数值优化未完成；不能据此判断表示无信息。新方案通过梯度证据核查同一凸目标，并将GPU提取与CPU求解分开。真实新求解结果仍待测。

## 下一步计划

完成CPU真实gate，通过后正式提交五项GPU提取与五组CPU求解，完整45组分类/读出评价使用驻点合格的新结果。若实际梯度或同目标对照未通过，继续检查求解及条件数，不以新迭代上限直接验收。
