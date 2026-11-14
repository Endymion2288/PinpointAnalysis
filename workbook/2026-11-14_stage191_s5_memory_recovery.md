# S5 主存并发修复与有限标签诊断

## 阶段目标

在现有GPU allocation上推进完整矩阵，同时解释τ分道差异与观测containment的定义。

## 工作内容

Kubeflow namespace GPU quota为1，第二份47GB allocation在server dry-run时被拒绝，没有创建Pod。随后在既有Pod复用新版源码运行五项coordinate，合计九个训练。该Pod于09:36 UTC失败；最后一次cgroup读取主存55,499,161,600 bytes、peak55,867,568,128 bytes，配额63G。九并发配置低估了训练/验证两套persistent DataLoader的主存开销。Pod在请求终态时已被清理，Job保存`Failed/BackoffLimitExceeded`；没有取得container exit code或OOMKilled字段，故OOM只作为与主存证据相符的假设。

恢复配置`configs/s5_recovery.json`将未完成的十一项从相同pretrained encoder重新初始化，以新名称运行；四个已完成神经网络及三个HGB结果保留。训练仍使用一名worker，validation在主进程读取，避免每个fit另驻留一份Python/torch/decoded cache。数据、batch size、seed、完整passes、优化器及架构保持原配置。新allocation限制六并发；中断模型不能进入科学比较。

评价按passes/label budget分cohort，4-pass screen只与同预算direct4比较，后续较长确认不覆盖其control。修复object coverage评价的label连接。

## 详细数据

CPU观测诊断cluster1581749，bigbird20；exit0，wall46 s、4 CPUs。10,000 validation事件中9,517具有有效bar signal，1,399满足预定containment proxy：已测bar轴边缘能量≤1%、末端100 mm bar能量≤5%、存在pixel时其边缘1 mm占用≤1%。483个bar无有效signal事件保留invalid mask。产物`measurement_diagnostics.parquet/json`，未使用calibration/test。

逐个检查全部11个τ新旧分道差异，仅一个属于validation，其余10个属于train。四例含Pz≤0且stored-min-z不在neutrino anchor的直接daughter；七例为初始Pz>0的charged daughter stored-min-z不在anchor。它们均保留unknown分道，不从旧catalog强行填回。七例的具体后续散射轨迹未测量；可确认的是源码保存轨迹z极值，不能将这些位置全部解释为出生点。证据与全部PDG/Pz/距离在`truth_disagreements.json`；复现入口`analysis.flavour.s5.truth_audit`。

三个HGB均完整回放、最大差0。Moments / object-only / moments+objects的validation NLL分别1.253048883 / 1.338692566 / 1.247475827，balanced accuracy分别0.4062 / 0.3412 / 0.4093。当前只描述点估计，配对区间等待统一评价。

## 阶段性结论

九并发的主存配置失败，不能用GPU显存较空推断主存也充足；本次失败与中断均保留并计费。已完成模型具有独立selected-checkpoint回放和实际参数更新，不受随后Pod失败影响。观测leakage proxy有明确分母；它不是generator层面的真实shower containment。

## 下一步计划

以六并发、validation worker0执行恢复矩阵，持续读取主存和内部健康；完整screen后选择关键候选进行17/42/2026共同预算确认，完成400次block bootstrap与物理解释。
