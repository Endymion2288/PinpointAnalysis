# S4 将真实参数更新纳入评价

## 阶段目标

避免独立 FP32 backward 或下游梯度掩盖实际训练前端零更新，将已发现的故障转为可重复的参数差分检查。

## 工作内容

新增 `analysis.flavour.s4.parameter_audit`，在 CPU 上从同种子 `build_model` 重建实际初始化；有 SSL 或监督 parent 时加载同一个 parent。分别比较 best/last checkpoint 的 input、local、local_pool 参数，保存最大差、L2 差、改变的元素数及有限性；每个分支都必须有真实改变。初始化阶段关闭 compile，不改变参数或载入规则，并恢复调用者 RNG。

后续使用新源码的 train-inner refit 评价自动调用参数检查，另外分别检查真实训练中的 input/local 梯度；旧源码已正式运行的评价保持冻结，新增[九模型CPU参数检查](../roadmap/configs/s4_transfer4_1k_parameter_audit.json)补充本轮1k任务。比较相同已接受模型参数，不涉及 samples、split 或硬件标定。

## 详细数据

实际运行四项针对性测试，0.155秒全部通过：local_pool改变而input/local未变会失败；所有前端真实改变会通过；从parent加载的值不能误计为下游训练更新；独立validation评价拒绝input有梯度而local为零的记录。参数检查也验证调用者CPU随机状态不变。命令：

```bash
python -m unittest tests.test_flavour_s4_parameter_audit \
  tests.test_flavour_s4.S4Tests.test_legacy_classification_configs_execute_and_receive_checkpoint_readback
```

新增CPU提交stage使用1核，日志写入EOS，小型wrapper写入AFS，按冻结任务等待上游完成。真实九模型参数差尚未测量。

1k正式训练cluster1578817（bigbird20）和CPU评价1578818已提交，CPU评价正在等待训练。正常readout恢复另有12GB MIG候选1578819，与两个compact候选共享claims；未在未获配节点上启动训练。03:49:08 UTC账本为30.386366/60 device-hours、344.887803 CPU allocation core-hours，全部历史失败和调度开销继续累计。

## 阶段性结论

事实：评价已具备独立的前端梯度与真实参数变化检查，已知断梯度故障会被拒绝。参数改变本身不能证明预测有效，也不能证明每个参数均学到物理结构，仍需拟合、完整validation与物理分块证据。

## 下一步计划

正式提交参数检查，等待九项修复重训完成后给出best/last实测。10k折内候选继续完成12 pass后重选，新的评价同时要求上述参数和训练梯度证据。
