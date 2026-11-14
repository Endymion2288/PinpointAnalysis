# S5 完整十五项筛查与确认冻结

## 阶段目标

依据预定规则选择三seed确认候选，固定最终评价矩阵。

## 工作内容

十五项screen全部完成并通过internal health、selected checkpoint replay与实际encoder update。恢复Job的Succeeded Pod已由watch保存真实终态。`select`正式运行，输出`selection.json`。确认三种结构为direct4、vertex+kinematics、six-way；8完整passes，各17/42/2026，均重新从相应MAE3 encoder初始化。两项kinematics head修复是额外4-pass探索对照，不纳入此前候选排序。

## 详细数据

按完整screen NLL排序前五：vertex+kinematics1.236937037、six1.237478999、semantics1.237785572、relative1.238425776、combined1.238886391。Direct4为1.240712302。

坐标同容量absolute/relative：仅vertex路径NLL1.242916981/1.238425776，macro AP0.446455060/0.451145707；加kinematics路径NLL1.241389413/1.241505426，macro AP0.446537324/0.447914066。相对坐标不是各任务一致改善，配对区间由完整评价给出。

HTCondor控制组1581763已在10:01:18 UTC开始，GPU为H100L-2-24C MIG2g.24gb。Direct4 seeds17/42已认领，Kubeflow将跳过这些名字，继续其余未认领配置。原MIG47恢复Pod已Succeeded；资源上界与终态汇总继续保留。

`s5_final_matrix.json`包含15项screen、2项component repair、9项confirmation，共26个神经网络；evaluator另加入3个HGB，共29结果。所有评价仍是完整相同10k validation，calibration/test未打开。

## 阶段性结论

预定主指标选择vertex+kinematics作为数据选择候选；其旧横向回归配方仍保留“未学好”的诊断，分类收益不等同于方向重建成功。six-way保留独立FASERCal假设。当前没有新结构完成三seed晋级。

## 下一步计划

尝试whole H10094GB flavor，以平台实际CPU/RAM和已测每fit主存为依据安排最多十并发；若资源排队则转回已成功的47GB MIG六并发。两后端通过claims分工，完成后运行统一400次block评价。
