# S5 完整评价入口与组件回归合同

## 阶段目标

在观察S5模型validation结果前固定困难通道、成对区间和运动学评价。

## 工作内容

新增`evaluate.py/plots.py`：同10k事件对齐、raw/common-support、400次物理块bootstrap、三seed均值与spread分列；direct4对各辅助任务、vertex对relative、相同vertex+kinematics任务对relative均有预定配对。六类输出核对τ求和并单列六分类confusion/AP。

τ分道、衰变材料/位置、可见性、prong、charm与containment给分母与1%/0.1% FPR计数、Clopper–Pearson描述区间；CC e/μ和NC各自报告charm假τ分母。模型分数与可测occupancy/对象量的相关性在真实类别内计算。对象vertex同时报告all/reconstructable/quality coverage和物理块区间。

回归target为已声明primary-system proxy。px/py用绝对GeV residual，pT在true pT>0.1 GeV用相对residual；θ difference与wrapped Δφ使用rad。报告50/68/95%绝对误差分位、分类confidence选择的coverage以及能量/位置分层；这不是calibrated predictive uncertainty。真实vertex/p3只作target与诊断。

Standalone英文PDF/PNG包括τ efficiency–purity、分道visibility、score–observable、object及component resolution–coverage；原生响应示例含可见/不可见τ的正确/错误组、charm背景及固定随机参考，并记录总体组计数。Truth与predicted markers为独立图层。

## 详细数据

运动学synthetic检查用4类各1个事件，把预测设置为精确target，包含px=1、py=2、pz=100 GeV、E=110 GeV及vertex=(1,2,−2500) mm；用于验证逆变换、误差符号和wrapped角。正式数据指标尚未测量。对象40分块任务在bigbird20 cluster1581721运行，已有部分完成；未将完成checkpoint当作学习验收。

## 阶段性结论

评价入口和固定合同已实现；S5分类/回归泛化和任务净收益尚未测量。下一步合并objects，先检查fake/coverage，再启动真实16事件拟合和13项screen。

## 下一步计划

Objects完成标记后启动拟合gate，通过后并发正式训练。每项保存128事件replay、实际encoder更新和完整validation概率。
