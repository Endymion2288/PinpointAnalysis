# S4 frozen probe求解器收敛与固定末轮图示

## 阶段目标

修复首批frozen probes的数值求解未收敛问题，避免将求解故障解释为表示或探测器信息无效；确保新LP-FT的训练图标记真正提交的固定末轮。

## 工作内容

已运行的`pinpoint-s4-empty-frozen`保持原冻结合同。其1k Scratch/MAE3日志均出现`lbfgs failed to converge`及500次迭代上限，因此这批输出保留为诊断。另生成[六项converged probes](../roadmap/configs/s4_converged_frozen_probes.json)，继续使用相同输入、标签子集、seed42、StandardScaler、C=1、L-BFGS及tol=1e-5，只将max_iter设为5000。

新实现保存每个branch/mode的迭代数、上限、tolerance、ConvergenceWarning、系数finite状态和fit样本数。未收敛时写`optimization_review_required`，预测保留为诊断，任务不能作为complete进入评价。Scaler仅在实际fit标签事件上估计，未使用validation选择C或数值参数。

继续检查训练图发现其默认用validation最小NLL标点，且颜色仍按initialization合并。已改为按评价合同的comparison_group分色，标记manifest记录的提交epoch；固定末轮即使NLL更高也标记末轮。内部梯度范围排除明确的LP冻结采样，并单列LP采样数，避免将有意的零梯度画在对数轴或当作断梯度。

## 详细数据

首批未收敛1k full/global NLL分别为Scratch 3.674914/1.779222、MAE3 3.251221/1.579259，均为无量纲诊断输出。这些值未通过收敛要求，不作表示质量或最终分类结论。尚未测量max_iter=5000的新结果；即使求解收敛，也可能保留真实的低标签过拟合，届时应按数据报告。

两个新增针对性测试实际通过：一项在真实四分类拟合中强制max_iter=1，确认记录未收敛；增加数值上限后收敛，Scaler均值与fit数据一致，预测finite。另一项确认早期NLL较低时固定末轮图仍标记末轮，并优先采用显式checkpoint epoch。先前完整30项S4测试已通过，`git diff --check`通过。

阶段149的CPU评价提交为`bigbird20.cern.ch` cluster1578766。为使首次正式评价使用本次修正的绘图源码，已撤回该提交，live history确认JobStatus=3、RemoteWallClockTime=52秒；没有GPU分配。原配置不变，后续用新提交名冻结新源码重提。该撤回及52 core-seconds仍计入原账本。

02:48:17 UTC账本实际累计26.002459/60 device-hours，剩余33.997541；CPU allocations累计304.873232 core-hours。Kubeflow六进程、Condor inner compact两进程与新获配的12GB单进程均在推进10k inner；1k refits及原frozen任务也正在正式运行。并发进度不是GPU利用率百分比，当前百分比未测量。

## 阶段性结论

事实：首批probe求解未收敛，已保留诊断输出并冻结新的数值修复合同。深度模型的训练健康与Logistic Regression数值收敛是两个独立检查。

待检验假设：增加求解上限能解决这次L-BFGS停止原因；不预设它能解决1k的概率过拟合。新probe结果及优化深度模型的完整validation收益尚未测量。

## 下一步计划

正式提交新六项probes和采用更新源码的1k CPU评价。10k inner完成后按固定OOF规则独立选择，保留原validation条件评价范围。新probe若仍未收敛，先审查求解器及特征条件数，再决定后续数值方案。
