# S4 参数更新证据与修复优先调度

## 阶段目标

用实际参数差分验证断梯度影响，并优先完成修复gate及LP重训，恢复两个标签预算的有效选择。

## 工作内容

按相同seed重新构造Scratch或导入同一SSL encoder，与实际`last.pt`逐组比较参数最大绝对差，未拟合事件或重新预测。结果保存于[参数差分表](../roadmap/results/s4_transfer_parameter_change.json)，补充训练健康记录：

| Model | Epoch | Input max change | Local max change | Local pool max change |
| --- | ---: | ---: | ---: | ---: |
| Faulty LP Scratch 1k, seed17 | 8 | 0 | 0 | 0.001519 |
| Faulty LP MAE3 1k, seed42 | 8 | 0 | 0 | 0.001793 |
| Non-LP Scratch 10k, seed17 | 6 | 0.101191 | 0.062986 | 0.072453 |
| Non-LP MAE3 10k, seed42 | 6 | 0.030745 | 0.057506 | 0.043335 |

该证据确认故障LP的前端参数实际未更新。Non-LP更新不能提前认可包含故障LP竞争者的旧共同选择；10k当前六项候选训练虽已全部complete，仍待新LP候选完成后重选。

完整相关测试`python -m unittest tests.test_flavour_s4 tests.test_flavour_s4_low_label tests.test_flavour_s4_paths`实际48项通过，14.520秒，涵盖下游加载、实际路由、形状预算、严格梯度、OOF选择及probe验收。真实CUDA编译gate尚未执行，不能以这48项代替GPU结果。

检查被停止的两个LP allocation后，发现它们结束前已领取三个100k non-LP任务。live Condor history分别确认2204965/1578796为JobStatus3；只对其三个未完成non-LP claims做归档，原checkpoint/config保留，冻结[三项恢复任务](../roadmap/configs/s4_repaired_objectives_recovery.json)。这样避免stale running claim让新队列跳过未完成工作，不处理任何仍在运行的owner。

## 详细数据

03:29 UTC三个100k恢复case中，一个manifest处于running，另两个尚未写manifest；原owners均已移除，归档证据为EOS `audit/repaired_objective_claim_recovery.json`。恢复使用相同科学配置和任务名，按现有checkpoint机制正式resume。

两个gate候选1578805/1578806排队超过十分钟。将其资源缩为2 CPUs、8000 MiB、3600秒上限后，`better-analyze`实际报告1个槽位愿意运行、49个需drain，仍未获配；没有绕过机器requirements。同期Kubeflow六项10k训练全部完成，主要开始可重算的frozen特征提取。

冻结[13项修复优先dispatch](../roadmap/configs/s4_transfer_recovery_dispatch.json)：一个真实CUDA gate及12项新LP inner，任务配置和名称均与既有提交相同，共享claims。计划保存原Kubeflow状态与日志后结束当前allocation，将其唯一GPU单元接续此dispatch；未完成提取的claims只在确认旧Pod终止后归档，稍后正式恢复。已完成的10k训练和probes不重算，结束/重启时间全部计入原账本。

## 阶段性结论

事实：1k故障LP前端参数确实未更新；新严格检查拒绝其进入结论。48项CPU测试通过。真实CUDA gate仍未测量。

待检验假设：修复编译时序能恢复真实更新并保持eager一致性；其优先级高于继续提取当前可重算的frozen features。

## 下一步计划

保存和终止当前Kubeflowallocation，按唯一GPU配额正式接续13任务。读取真实gate结果并检查所有解冻首步的input/local梯度；通过后完成12项重训及两预算共同重选，再恢复未完成的100k任务和frozen提取。
