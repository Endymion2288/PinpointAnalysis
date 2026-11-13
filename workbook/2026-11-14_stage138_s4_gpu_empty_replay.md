# H100 BF16空响应重放与逐更新梯度证据

## 阶段目标

验证CPU定位的空占位缺陷是否出现在实际H100 BF16路径，并记录新训练每个update的裁剪前数值范围。

## 工作内容与详细数据

通过已分配的Kubeflow Pod `pinpoint-s4-recovery-c-9pxgf`执行短诊断，未启动持久训练进程。读取原train首批16事件，旧/新路由都从相同seed初始化。命令为`analysis.flavour.s4.gradient_replay --device cuda --bf16`，完整输出位于S4输出根`audit/initial_gradient_decomposition_cuda_bf16.json`。首次诊断因单独设置PYTHONPATH覆盖LCG依赖路径而未导入NumPy；恢复LCG路径后成功，没有修改训练worker环境。

| Original fit | Old BF16 preclip norm | Repaired BF16 preclip norm | Old / repaired input gradient squared norm |
| --- | ---: | ---: | ---: |
| Scratch 10k seed17 | 6.4641942×10¹² | 156.099991 | 4.17857855×10²⁵ / 54.824191 |
| Scratch 1k seed2026 | 1.8329892×10¹² | 59.731205 | 3.35984706×10²⁴ / 163.877248 |

两项修复后空事件损失到input的梯度均为0；有响应事件仍有梯度，norm约29.199与40.19。后者两次CUDA计算有0.003以内差别，未将GPU reduction写成逐位一致。以上为eval模式、未启用compile的诊断；正式compiled训练仍需真实拟合gate。

训练器现在把每一步已经计算的裁剪前norm汇总为`gradient_health`，保存检查次数、最大norm和最小clip系数至progress、last checkpoint和final manifest；恢复时接续这些计数。修复模型验收同时要求检查次数等于实际optimizer updates且最大norm在配置审查界限内。缺失证据、少查update或有限但过大数值均不能通过。这是原计划采样之外的全update检查，不扩大逐步日志。

原21项相关检查通过；新增验收反例覆盖没有全update证据、计数不足、norm=10¹²和正常记录。正式新模型指标未测量。

### 调度与成本

Kubeflow重新登录后Job/Pod API可用；metrics API返回RBAC Forbidden，`nvidia-smi`在MIG容器内无法读取显存与利用率。因此本阶段GPU利用率未测量。第二个47GB MIG请求被Kyverno `namespace-gpu-quota`拒绝，原因是已占用namespace允许的1 GPU unit；未创建Pod、未产生allocation。

新gate已进入HTCondor `1578483`（bigbird20，compact BF16候选）。该入口记录所有schedd负载、EOS兼容策略和WRITE授权，按idle/held/running最少者选择；其真实获配和gate结果留待后续报告。先前小卡候选1578171在检查时已离队，未将其误写成手动移除成功。

23:45:51 UTC资源账本为17.800496827/60 device-hours、217.931683697 CPU allocation core-hours，剩余42.199503173 device-hours。包含全部初始化、失败与并发allocation，MIG分型保留，不换算等效FLOPs。

## 阶段性结论

实际H100 BF16也复现同一空占位梯度缺陷，新的mask/attention路由能消除该虚假input梯度。全update数值范围已成为新模型验收所需证据，不能由最终best checkpoint的健康标记替代。

## 下一步计划

在先获配的HTCondor或Kubeflow资源完成真实32-event拟合和最大占用压力测试，随后执行冻结的18模型成对重新训练。所有实际分配继续计入原60 device-hour预算，旧任务完成后及时释放资源并清理已归档的AFS提交文件。
