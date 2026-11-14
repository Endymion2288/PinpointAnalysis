# S6：正式统计pilot与overlap并行

## 阶段目标

以完整独立test验证统计耗时、内存、条件区间及输出，再扩展预定60个模型/prior组合。

## 工作内容

首项是预定顺序中的MAE seed17、balanced prior；严格使用冻结温度和validation cuts。完成400次物理block区间、63个条件分层的固定cut区间、完整tie-aware ROC/PR、raw counts与cutflow。

Exact-target common support拆为九个模型任务，读所需分数列并仅在原固定权重>0的事件上计算；原support bins、最小计数、类prior与block multiplicity不变。独立测试加入零权重但高分误分类事件，确认删除这些事件后加权统计不变。

## 详细数据

Pilot724441事件：BA0.4337569421、NLL1.2030261018，统计211.936秒。HTCondor1582752 exit0，ResidentSetSize2500000KiB（2441.4MiB），实际User+Sys CPU213秒；RemoteWallClockTime1611秒包含等待完整test。其余model/prior尚未测量，不以单seed结果代替最终主ensemble。

该单seed在冻结balanced 1%背景cut的τ效率0.03300894、实际weighted FPR0.01121106；0.1%背景cut的τ效率0.00306599、实际FPR0.00082621。工作点仍来自validation，test FPR不强制重新压到目标。全部63个非空stratum带固定工作点block区间。

`python -m unittest tests.test_s6_evaluation tests.test_s6_calibration tests.test_s6_truth_geometry`，11项通过。八个真实transport条件统计已完成，每项4.5–9.2秒；完整generator truth3512行、τ440条均valid，全部响应相同。

## 阶段性结论

统计实现已经完整真实test pilot验证，可以并行扩展。当前单seed总体分类与低背景τ效率有限；最终结论须使用预定ensemble、控制组和全部物理分层。

## 下一步计划

提交其余59项，九个固定overlap模型并行执行；完成29个readout推理任务、14条件统计与transport truth尾部，然后汇总物理结论与图表。
