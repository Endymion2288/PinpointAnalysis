# 阶段 14：空间矩阵正式启动与预算实测

两族 native-joint 已在固定 V100 上开始 80k/10k 正式训练。前 250-step 实测覆盖 1,000 train exposures，ParT 为 11.311 events/s，ParticleNet 为 16.488 events/s；后者 500 steps 提高至 21.920 events/s，存在首次 loader/CUDA 开销。依据该吞吐继续提交其余十个已冻结空间配置。

后续修正：1573741/1573743 的 v1 任务已在首个 epoch 完成前停止，原因和新配置见[阶段 15](2026-11-11_stage15_s2_summary_correction.md)。本页仅保留当时的真实运行记录，未产生科学空间比较结论。

## 阶段目标

将阶段 13 的 12-run 空间矩阵具体运行，并让实际 loader/optimizer 成本约束初轮资源预算。

## 工作内容

1573741.0/1 分别执行配置 index0（ParT native joint）与 index6（ParticleNet native joint），模型版本 `6a0ab0d`，均为 Tesla V100-PCIE-32GB、LCG_107。此后继续 index1–5/7–11，不重提仍运行的两个配置。每次 submission 记录当前 schedd 负载/兼容性和 ClusterId。

新增 [roadmap/07_s2_operations.md](../roadmap/archive/2026-11-14_v1/07_s2_operations.md)，统一列出实际产物、运行/恢复命令、软件环境和科学验收未完成项；同步 roadmap 入口，避免将 S2 误写为尚未实现或已全部完成。

## 详细数据

| Family / checkpoint | Exposures | Train elapsed [s] | Events/s | Data wait [s] | Forward+backward [s] | Peak GPU [MiB] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ParT step250 | 1,000 | 88.410 | 11.311 | 45.203 | 42.906 | 807.04 |
| ParticleNet step250 | 1,000 | 60.648 | 16.488 | 26.572 | 33.907 | 763.52 |
| ParticleNet step500 | 2,000 | 91.240 | 21.920 | 28.516 | 62.250 | 896.88 |

这些是第一 epoch 的早期吞吐；包含首次 worker/cache 开销，也只覆盖当前 shard window。用 ParT step250 与 ParticleNet step500 保守外推十二配置的 train 部分约 71.5 GPU-hours，未含 validation/I/O 尾部，粗化/projection 的成本也尚未单独实测。因而先继续已定义的空间矩阵，并监测后续完整 epoch 的实际资源；不是已消费时长或排期承诺。

截至本报告没有完成任何科学空间 epoch，validation NLL/AP 与主效应：未测量。train loss 早期约 1.39–1.41，不作为分类结论。B0 及 prior 诊断、GPU profile 的实际结果仍见阶段 12–13。

新增训练计数和 sampler 改动的 3 项测试在 LCG_107 再次通过（4.859 s），包含 occupancy coverage/padding、domain repetition 和完整恢复权重一致性。

## 阶段性结论

事实：正式尺寸模型、实际 80k 目录、多 worker loader 和优化器已共同工作，早期显存低于 1 GiB allocated。初轮吞吐支持继续冻结矩阵；完整 epochs/validation 仍待作业结果。

## 下一步计划

监测十二配置的完成状态、完整 epoch 吞吐与 attempted/applied steps，处理实际失败后仅恢复相应任务。完成配对空间比较，再安排 OR/shift、modal、pair/graph/shuffle、训练域和三种子；仅在科学清单全部有结果后验收 S2。
