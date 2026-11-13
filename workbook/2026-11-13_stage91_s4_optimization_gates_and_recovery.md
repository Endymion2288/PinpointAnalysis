# S4 真实事件优化检查与评价修复

## 阶段目标

检验新目标能否拟合真实事件，并完成 corpus 训练前的完整评价路径。发现程序错误时修复重跑，不将未完成训练作为方法结论。

## 工作内容

已完成的真实事件分类/重建检查保留；重建优化以同一 32-event train 集和固定 masks/queries 测量。随后 corpus 初始评价表使用 `prediction[j,:4]`，错误地截取 query 轴；应为 `prediction[j,:,:4]`。三项 corpus pilot 在首个优化 step 前因该 shape 错误退出，尚未产生 corpus 训练结果。补充完整 `evaluate` 路径测试后重启。

MAE+barRel 的旧 loader 将大 sidecar 转 Python lists，32-event 随机读取多个 shard 过慢。改为 Arrow 连续 buffers/offsets 与 `searchsorted`，目标数值不变。旧 Job 在已完成检查和错误日志保存后终止，失败/未完成 claims 归档再重新领取。所有时间纳入资源统计。

后续 batch 使用已提交代码的独立小型 snapshot 和冻结 campaign，`PYTHONPATH` 指向 snapshot；manifest 记录实际 snapshot commit。这样可以继续开发诊断工具而不改变正在运行的代码。

## 详细数据

共同 backbone 的 32-event 分类拟合：768 steps，accuracy **1.0**、NLL **2.4087248675641604e−5**、32 个独立预测向量，用时 **103.41126708 s**；全部内部健康检查通过。

| SSL target | 256-step reconstruction loss | Positive queries / 32 events | 用时 s | 最大 occupancy FP32/BF16 |
| --- | --- | --- | --- | --- |
| MAE 50% | 0.2660980559885502 | 733 | 84.99079775 | 通过 |
| MAE 75% | 0.2913586273789406 | 714 | 84.75896170 | 通过 |
| EMA + reconstruction/variance | 0.335360586643219 | 714 | 90.17047190 | 通过 |

最大 occupancy 原始通道数 **36,549**（含 tracker）。50% mask 保留 18,575 nodes，75% mask 保留 9,097 nodes；两精度 loss/梯度 finite，input projection 梯度平方范数均非零。此处是优化/压力检查，不是重建泛化或下游物理性能。

优化后的 Rel loader 两个真实 train 事件 smoke 用时 **2.651844006 s**，有效 bar targets 分别 110、49，soft fraction simplex 和误差均为 0。

8 项 S4 测试通过，包括新增完整评价 query-row 数量和 permutation 统计检查：

```bash
python -m unittest discover -s tests -p 'test_flavour_s4.py'
```

旧 GPU Job 已删除，Pod 确认不存在；终态和保守 allocation end 见 `logs/pinpoint-s4-pilot-a_allocation_end.json`。成本统计对尚未发布的 Condor history 保留缺失状态和已缓存证据，不把 history timeout 记为零成本。

## 阶段性结论

共同 backbone 具有真实小样本分类拟合能力，三种已完成 SSL 目标能降低固定 train 查询的重建损失并向 encoder 传播梯度。Corpus 重建收益及四分类收益尚未测量。评价轴错误和 Rel 读取低效属于程序问题，修复后继续原实验。

## 下一步计划

复用配置完全相同的已通过 32-event gates，重跑四项 corpus pilots；为 barRel 完成同等 gate。分析 native/coarse/bar 的正例数、AP/先验比、跨事件方差及置换影响后，冻结正式预训练和下游对照。
