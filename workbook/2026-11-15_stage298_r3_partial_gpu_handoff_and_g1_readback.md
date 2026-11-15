# R3：按完成时间迁移两个模型，保留消融原卡

## 阶段目标

用已取得设备的实测数据选择并行方案，减少 G2 剩余时间；完成 G1 的全 Tune 权重重算。此阶段没有提前决定 G2 的科学效果，R3 仍未完成。

## 工作内容

在 Kubeflow 3g.47gb 上，先测三个真实 G2 架构各自单进程和三并行；整体迁移没有速度优势，因此结束该尝试。完成同 allocation 的必要 G1 probe 后，重新取得 3g.47gb，实测 control/relations 双并行。消融模型继续使用原 Condor 1g MIG。

两个拟迁移模型均保存 100k 曝光 checkpoint 后，读取实际 `resume.pt`，检查保存预算和旧 allocation ID。确认旧 Condor 从队列消失、history status3 且 event log 有 eviction/abort 后，才原子写入与新 claim 对应的 release。全程没有两个存活 writer 写同一正式模型。

新增训练组的附加只读 probe：完成两个模型后，复用 allocation-local Train/Tune mmap，依次检查这两个模型及独立消融模型；必要时等待后者正式完成。该操作只读已完成权重，不训练第三个副本，三个 native 模型仍各自只有一个正式 writer。

## 详细数据

根目录：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R3/`。

资源 profile 使用同一 Train10k 池、每个模型 256 updates、batch8、BF16 encoder/FP32 head、真实 revision3 架构。每项只曝光约 2,048 条记录，不是完整训练 10k 事件。

| 已获得的 3g.47gb 方案 | 单模型 training events/s | Concurrent training events/s | Phase operation events/s |
| --- | --- | --- | ---: |
| K3 三个模型各自单进程 | 63.605 / 61.927 / 72.941 | — | 54.653 / 54.115 / 61.825 |
| K3 三并行 | — | 22.609 / 22.095 / 23.774 | 62.901 |
| K4 control/relations 各自单进程 | 63.541 / 62.063 | — | 54.930 / 54.270 |
| K4 双并行 | — | 32.398 / 31.426 | 58.119 |

这里 phase operation 包含 worker 启动及该 phase 的开销，不包含 parent 一次共享输入准备；K4 共享准备另测 86.300060 s。不能把这组短 profile 当成完整正式训练吞吐，也不能将单模型速度乘并发数当作实际并行速度。

17:48:54 UTC 迁移决定使用旧任务近期约 2,000 updates 的速率：control 22.2922、relations 22.0921、ablation 25.6969 events/s。后备卡双并行明确快于前两个旧 writer；消融继续旧卡，避免把全部工作挤在一个分区。

| 完成时间估计项目 | 数值 |
| --- | ---: |
| 保持旧分配的剩余关键路径 | 15,156.990 s |
| 新分配的剩余关键路径 | 12,643.525 s |
| 候选吞吐降低10%的敏感性估计 | 13,827.450 s |
| 新建140k Train/Tune输入缓存估计 | 1,208.201 s |
| 新卡恢复验证时间余量 | 180 s |
| 另外加入的时间余量 | 600 s |
| 预计提前完成 | 2,513.465 s（41.891 min） |
| 吞吐降低10%后预计提前 | 1,329.540 s（22.159 min） |

估计使用 Train10k profile 线性外推输入准备；剩余 validation 按已测首个4k screen推算三个screen及两个40k完整推理，并计入消融原卡的关键路径。它包含当时未保存工作损失，仍是预测；没有运行旧分配剩余反事实，不能称为实测节省。

旧 control/relations 为 1583008.0 / 1583009.0。两者终态分配时间分别 7,011 / 7,010 s，全部保留成本。17:49:43 UTC event log 确认 eviction/abort，之后 release 新 `pinpoint-r3-g2-k4`。消融 1583010.0 未移除。

保存点均为 12,515 updates、100,000 次曝光。旧 control 最后滚动日志为 13,000 updates/103,880 次曝光，relations 为 12,750/101,880；至少 5,760 次已记录、尚未保存的曝光会重放，末个滚动点之后的实际量未测量。最终有效模型轨迹仍按冻结的 400k 曝光执行，重复计算另由 allocation 成本完整覆盖，不将遗失的更新计入有效终点。

G1 probe 已完成，共同 Train4k 只拟合 scaler+logistic C1，全部 Tune40k 评价：

| 权重 | Local BA3 | Regional BA3 | Global BA3 | Tracker BA3 |
| --- | ---: | ---: | ---: | ---: |
| G1 stride2 | 45.2333% | 46.2400% | 50.3500% | 41.3000% |
| R2 D2 stride4（既有probe） | 44.9833% | 48.2267% | 50.2800% | 41.5267% |

G1 全部 Tune40k 保存概率重算最大/平均差0、argmax agreement1.0。global 是 tracker 融合前的 detector encoder；层级 probe 使用固定聚合及有限 Train4k 线性分类器，不代表信息上限或该层的重训效用。G1 的正式不晋级决定保持 stage297 的原始模型配对结果。

主要产物为 `audit/profile_handoff_g2_k{3,4}/results.json`、`audit/native_migration_decision_k{3,4}.json`、`audit/g2_k4_writer_handoff.json`、`runtime/g2_k4.release.json`、`probes/g1_c1/results.json`。新 claim 的训练及附加 native probes 尚未完成，最终账本也未形成。

验证：43 项 R1/R2/R3 单元测试通过，三个修改模块 `py_compile` 通过。新卡 control/relations 两项真实共享 mmap 恢复均已完成：参数最大差0、概率/曝光表逐位一致；各自使用源 Train32中的optimizer Train24和synthetic validation8，未使用真实Tune。证据为 `smoke/g2_k4_c{0,1}/integrity.json`。正式共享缓存现已开始准备，尚未恢复大样本优化。

17:53 UTC 资源快照为 12.358424 异构 device-hours（包含 whole-H100 尝试98s的生命周期上界），仅 `r3_g2_ablation_c9` 和 `pinpoint-r3-g2-k4` 两条 allocation 活动；这是运行中快照，不是最终账本。

## 阶段性结论

事实：对当前 G2，三并行没有建立整体迁移优势；两个模型迁移、一个保留原卡的组合有正的完成时间预测。G1 权重/完整输入读回一致，分层可读性证据齐备，原 stride 干预仍未满足科学收益门槛。

假设：候选 profile 在剩余完整数据上的吞吐可以维持，缓存准备接近线性估计。后续以实际速率和完整终态替代估计；它们不能提前支持 G2 几何关系的物理收益。

## 下一步计划

核对共享 Train100k/Tune40k 缓存守恒及继续训练速率。正式终点齐备后运行完整 geometry 配对及独立复核、全部 native probes 和 health；决定是否触发三seed/粒度/机制控制，再完成科学决定、英文图、终态成本与全部 R3 验收。
