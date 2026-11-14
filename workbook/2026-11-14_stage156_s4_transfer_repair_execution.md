# S4 编译冻结修复的正式验证与重训依赖

## 阶段目标

让LP修复先通过真实FP32/BF16 CUDA回归，再启动12项新LP训练；保留原始故障输出和其资源成本，避免重复调度历史任务。

## 工作内容

真实gate已正式提交：`pinpoint-s4-transfer-gate-compact`为bigbird20 cluster1578805、`pinpoint-s4-transfer-gate-small`为1578806，两者共享同一gate任务claim，单进程竞争可用native-BF16 GPU。03:17 UTC均Idle，已将自己这两个作业的JobPrio设为100，优先完成修复验证。它们没有临时通过SSH执行训练。

12项新LP inner均显式依赖`s4_transfer_compile_gate_v3_s42`，训练入口另外读取`transfer_compilation.json`并要求passed。新合同复用36项完整non-LP控制，新LP有独立任务名及执行版本。原先1k评价1578768以缺input梯度失败；10k评价1578800已撤回，因为其共同候选选择包含故障LP。正在运行的10k early模型保持原冻结训练配置，结果仅为待确认的候选，不提前认可旧选择。

## 详细数据

2026-11-14 03:17 UTC联网重新获取并阅读[ICLR 2022 LP-FT](https://arxiv.org/abs/2202.10054)、与实际PyTorch版本匹配的[2.7.1 autograd说明](https://github.com/pytorch/pytorch/blob/v2.7.1/docs/source/notes/autograd.rst)及[compiler troubleshooting](https://github.com/pytorch/pytorch/blob/v2.7.1/docs/source/torch.compiler_troubleshooting.rst)。autograd说明冻结参数不会进入forward反向图，compile说明生成图依赖guards及首次trace假设；这要求实际检验参数状态转换，不能据此直接推断当前实现一定重编译。当前bug由实际梯度缺失建立证据，文档用于修复设计。稳定版HTML只返回版本跳转stub，改取v2.7.1官方源文档，缓存及来源记录为EOS `literature/transfer_compile_sources.json`。

阶段155的十项CPU测试实际运行5.547秒，全部通过。本阶段新增的是依赖及调度，不新增模型测试；真实CUDA修复效果尚未测量。gate保留旧路径作为诊断对照，修复路径必须同时有真实input/local梯度、参数变化和与eager一致的logits。

03:17:24 UTC累计账本28.500986/60 device-hours，剩余31.499014；CPU累计325.936942 core-hours，包含故障与撤回。资源账本不会因重训重新计零。10k当前early重拟合采样的input/local梯度均为正；这不代替待重训LP候选及重新选择。

## 阶段性结论

事实：修复代码、严格梯度规则与新任务依赖已经提交，真实GPU回归已排队。1k/10k旧OOF选择和受影响的LP分类结果继续保持失效状态。

待检验假设：新编译时序可恢复解冻梯度并达到eager一致性；只有gate和真实重训共同确认后，才能重新建立LP方案证据。

## 下一步计划

正式提交依赖gate的12项重训，由可先获配的后端运行。读取gate终态，失败时继续审查编译/优化状态；通过后检查每个新LP任务解冻首步及后续input/local梯度，完成两个预算的共同OOF重选与新refits。
