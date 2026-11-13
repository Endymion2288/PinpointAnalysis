# S4：四类别尾部实际计数

## 阶段目标

将每类1%和0.1%FPR效率与实际通过事件数一起报告。

## 工作内容与验证

S4评价新增`classwise_operating_points`，逐类保存经验ROC的threshold、signal/background selected、各自总事件数，以及描述性的Clopper–Pearson计数区间。区间针对validation上已选阈值的计数，不作为独立测试工作点覆盖率。原paired block bootstrap继续提供相同物理事件上的方法差异区间；calibration/test保持冻结。

`test_four_class_operating_points_preserve_ties_and_finite_counts`通过：四类常量概率保留原始ties，严格背景接受度下均选择0个signal/background，背景计数区间上限仍为正。此前17项S4回归与shape-count对照已通过。

CPU shape budget replay已具备配置。首次提交在存在未提交评价代码时被source-freezing检查阻止，未创建allocation；本commit后重新提交。当前主训练与queued campaigns不受影响。

## 阶段性结论与下一步

评价包含四类尾部实际计数和明确定义的统计范围。继续完成CPU成本重放、Rel/EMA修复和下游实验矩阵；全部模型完成后再运行最终400-replicate配对比较。
