# S4 真实CUDA复现与固定权重数值对照

## 阶段目标

真实复现LP编译断梯度，核查修复后的梯度/参数更新，并用相同权重将编译误差与BF16训练轨迹差异分开测量。

## 工作内容

Kubeflow `pinpoint-s4-transfer-recovery-main`在H100 NVL MIG 3g.47gb执行八个固定原train事件的回归，源码`6f54c44`，84.326220秒。旧编译、eager和修复编译各做两次LP更新、一次LP eval及三次解冻更新，分别用FP32/BF16。旧路径在两种精度均复现input/local梯度缺失及参数零更新；修复路径和eager在解冻后均有正梯度及实际更新。

整项gate仍为**failed**：BF16独立优化轨迹的最大logit差0.025334，超过预设0.025。12项新LP候选没有启动。失败Job和Pod终态快照已保存到EOS后删除，未将其改写为通过；[精简回归实测](../roadmap/results/s4_transfer_compile_gate_v3.json)保留全部checks。

## 详细数据

| Precision/path | Input maximum parameter change | Post-LP input gradients | Result |
| --- | ---: | --- | --- |
| FP32 legacy compiled | 0 | Absent | Fault reproduced |
| FP32 eager | 0.0000903867 | Positive on all 3 steps | Healthy |
| FP32 repaired compiled | 0.0000903867 | Positive on all 3 steps | Healthy |
| BF16 legacy compiled | 0 | Absent | Fault reproduced |
| BF16 eager | 0.0000904128 | Positive on all 3 steps | Healthy |
| BF16 repaired compiled | 0.0000904389 | Positive on all 3 steps | Healthy |

FP32独立轨迹最大logit误差为0.00000105798，低于0.0001；BF16为0.025334，高于0.025。修复BF16的三次input梯度平方范数为98.8625/62.0038/43.2322，local为53.7376/31.9739/24.0230，均通过健康检查。数值修复已恢复训练边，但当前gate整体失败，尚未认可重训方案。

独立BF16轨迹在每步舍入后再分别经过Adam更新，累积差异同时包含精度和优化状态差别，不能单独定位编译forward误差。为此固定[新gate](../roadmap/configs/s4_transfer_compile_gate_v4.json)：在相同最终权重、相同八事件上，用eager FP32作参考，测compiled FP32、eager BF16、compiled BF16，并记录概率最大差。预先固定FP32最大logit误差<1e-4；compiled BF16相对FP32的RMS误差不得超过eager BF16对应误差的2倍加1e-4。这个数值回归范围只适用于所测八事件，不是全样本精度误差定理或物理系统误差界。原BF16独立轨迹差继续作为诊断保留，不重新提高其0.025阈值来追认旧gate。

2026-11-14联网阅读与实际软件版本一致的[PyTorch 2.7.1 numerical accuracy](https://github.com/pytorch/pytorch/blob/v2.7.1/docs/source/notes/numerical_accuracy.rst)：浮点重排、不同实现与reduced precision均可能改变结果，按同输入/同权重检查能定位执行路径差异。源文件及时间记于EOS `literature/transfer_compile_sources.json`。LP-FT及autograd/compiler来源见阶段156。

冻结[新inner合同](../roadmap/configs/s4_low_label_inner_v4.json)和[13项dispatch](../roadmap/configs/s4_transfer_recovery_dispatch_v4.json)，36项non-LP继续逐字复用，12项新名LP依赖新gate；旧gate及blocked候选保留历史。新数值对照尚未执行，未测量其结论。代码通过语法编译检查；48项相关CPU测试此前已通过，真实GPU回归是此次新验证的主体。

原Kubeflow main-b已在六项10k训练完成后于03:32:19 UTC确认Pod不存在，旧运行快照及allocation结束上界保留，五项未完成probe claims仅在确认owner消失后归档。恢复范围为三个100k non-LP及五项probe，冻结于[八项恢复配置](../roadmap/configs/s4_repaired_readout_recovery.json)。GPU调度、编译失败及重算继续累计原60 device-hours。

## 阶段性结论

事实：真实CUDA复现旧断梯度，修复恢复两种精度的input/local更新；FP32轨迹一致。BF16旧gate未通过，独立轨迹差异的来源仍需相同权重对照，不直接作编译正确或错误结论。

待测：新固定权重FP32/BF16对照、修复后的12项完整LP训练与两预算重选。若相同权重对照仍异常，继续审查精度/编译实现，不放行长训练。

## 下一步计划

用正式Kubeflow Job运行新gate，通过后启动六并发LP重训。其余正常non-LP及frozen恢复继续使用Condor。新OOF选择、固定轮数refits、完整物理评价与总成本齐全前，S4保持进行中。
