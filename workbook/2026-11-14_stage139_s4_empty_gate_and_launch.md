# 空响应修复的真实拟合gate与完整矩阵启动

## 阶段目标

以真实train事件验证修复后的可学习性和完整占用覆盖，再启动冻结的18模型成对重训，建立该矩阵专用的技术验收。

## 工作内容与详细数据

HTCondor `1578483`（bigbird20，实际`b9pgpun204`，H100 2g.24gb MIG）使用提交版本e8a1daf运行新gate。32个原train事件每类8个，按既有20–3,000 pixel+bar占用选择。768 updates后accuracy=1.0、NLL=1.899898×10⁻⁷、32个不同预测向量，13次计划内部健康检查通过。拟合用时111.109176 s，peak allocated GPU=738.715820 MiB。配置、事件keys和全过程位于S4输出根`pilot/s4_empty_backbone_health_s42/gate.json`。

随后同一batch作业以新模型初始化执行空/中位/最大原train占用的独立压力测试：

| Native channels | Mode | Finite | Peak allocated GPU [MiB] | Input weight gradient squared norm |
| ---: | --- | --- | ---: | ---: |
| 0 | Inference | true | 96.481934 | 未测量 |
| 0 | Backward | true | 131.027832 | 0 |
| 556 | Inference | true | 138.130859 | 未测量 |
| 556 | Backward | true | 113.116699 | 1730.412842 |
| 36,549 | Inference | true | 242.161133 | 未测量 |
| 36,549 | Backward | true | 762.557617 | 966.608704 |

该train最大事件36,549个原生节点全部进入表示，没有事件截断。压力测试用时83.738422 s，6行均完成，精度BF16、Torch2.7.1，产物为`stress/s4_empty_backbone_health_s42/measurements.parquet`和`summary.json`。23:55:44 UTC正式pilot child以exit0结束。

完整重训通过以下三项候选竞争共享claims，所有任务为715ed4a冻结源码；每个实际writer只领取一个实验名：

| Allocation candidate | Schedd / cluster | Pool | Concurrent fits |
| --- | --- | --- | ---: |
| s4-empty-compact-a | bigbird20 / 1578687 | native BF16 ≥20GB including 24GB MIG | 2 |
| s4-empty-compact-b | bigbird20 / 1578688 | same | 2 |
| s4-empty-small-a | bigbird20 / 1578689 | H100 12GB MIG | 1 |

提交入口每次重新检查所有schedd负载、EOS策略和WRITE授权，选择idle/held/running最少的兼容者，具体快照位于`logs/submission_s4-empty-*.json`。任务按100k、10k、1k的顺序领取以优先启动较长的训练。正式分类结果尚未测量，不将排队或启动写成完成。

新增`analysis.flavour.s4.accept_empty`单独检查这18项矩阵：配置/源码产物、14项checkpoint读回、独立best FP32 backward、初始32次及每100步的健康采样、所有update的范数检查计数、三套预训练来源readback、完整10k四分类和400次128块bootstrap、三seed对照、终态allocation及原60 device-hour预算。验收分别报告技术执行和NLL相对constant prior的实际数值；不能把技术通过写成预训练改善。该入口尚未在未完成矩阵上声称通过。

21项相关测试及新增的全update证据测试已通过（验收测试文件6项）。23:55前后清理6个已终态AFS提交文件共5,648 bytes，每个原文与EOS提交记录逐字一致；审计位于`audit/afs_cleanup_20261113T235535.json`，运行中的提交文件保留。

## 阶段性结论

新路由通过真实事件记忆与完整最大占用的BF16 backward；空事件不向硬件input投影传播梯度，真实响应仍能有效学习。S4方法收益仍等待18项实际重训与统一评价。

## 下一步计划

旧Kubeflow allocation结束后保存Pod终态、释放GPU配额，再以同一共享claims运行新矩阵；依据现有单fit显存与host内存观测，先采用6并发并监视host/显存范围。完成后统一评价及终态成本统计。原S4有问题的Scratch和相应方法收益不作为正式结论。
