# S4 同目标驻点回归通过与五路特征提取

## 阶段目标

确认数值求解器修复保留相同目标与精度，并正式恢复完整预训练目标比较。

## 工作内容与详细数据

CPU cluster2205010完成真实原train的Scratch/MAE3各1,000事件。四项均按阶段166预先阈值通过，实际耗时1,160.881秒，未使用GPU或calibration/test。full/global维数为1,440/192，scaler各自只fit所用原train标签，C=1。实测如下：

| Initialization / branch | Reference gradient infinity norm | Newton-CG gradient infinity norm | Objective difference | Maximum fit probability difference |
| --- | ---: | ---: | ---: | ---: |
| Scratch full | 9.86831e-10 | 3.87538e-8 | 5.46707e-12 | 2.66855e-6 |
| Scratch global | 3.33178e-9 | 9.81539e-7 | 7.19615e-10 | 4.93342e-5 |
| MAE3 full | 1.33853e-9 | 6.53254e-8 | 1.60520e-11 | 4.88406e-6 |
| MAE3 global | 4.87946e-9 | 9.11534e-7 | 4.80493e-10 | 3.86933e-5 |

全部参考梯度≤1e-8、Newton≤1e-6、目标差≤1e-5、概率差≤0.005且没有ConvergenceWarning。参考continuation中三项仍在零数值变化时由objective规则停止，但独立实际梯度已满足预定1e-8要求；没有凭optimizer成功状态代替验收。完整实际终止原因保留于[数值表](../roadmap/results/s4_newton_solver_gate_v2.json)。

05:10:37 UTC Kubeflow H100 NVL MIG3g.47gb正式获配，五种encoder在同一allocation中并发提取，每种100k train+10k validation。Pod为`pinpoint-s4-newton-features-mig47-v2-5gnfd`，自动注入22CPU/63GB，代码dc5758e。Condor备用2205011当时仍idle，确认五项claims均属Kubeflow后正式移除，没有重复writer。05:11:58 UTC每路累计2,784至2,800事件、约46 events/s，包含启动和编译；这个初始吞吐不作为稳定性能测量。MIG的`nvidia-smi`利用率返回N/A，未编造百分比。

五组CPU读出2205012–2205016及完整评价2205017已提交bigbird21。每次提交实时选择兼容EOS、WRITE允许且idle最少schedd，依据保存在EOS`logs/submission_*.json`。正式[提交摘要](../roadmap/results/s4_stationary_v2_execution.json)与完整源`audit/stationary_v2_execution.json`均保存。

当前100k训练仍保持冻结四轮，EMA3/MAE5进行第四轮、Rel5第三轮；最终分类指标未测量。30个CPU full/global读出的100k驻点和磁盘概率回放仍待实际完成。

## 阶段性结论

事实：同一真实表示与正则化目标上，Newton-CG与满足严格梯度的L-BFGS参考一致，读出恢复已通过数值回归。该结果解决数值算法验收，不代表100k表示质量或物理分类收益。

## 下一步计划

完成五路features、15预算读出及45预测/66配对统计，检查各预算的实际驻点和30次磁盘回放。训练与GPU提取完成后保存真实终态并释放资源；完整验收后清理约3.59GB暂存数组与小型gate缓存。
