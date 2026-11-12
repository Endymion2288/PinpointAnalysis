# 阶段68：4 mm coarse对照完成与保留Kubeflow排队位置

## 阶段目标

读回完成的4 mm／1 mm控制，继续补齐2×2矩阵；维护两后端排队，使长期Pending的Kubeflow Job能保留位置并留出实际训练时间。

## 工作内容

`s3_a4_context_latent_1mm_s42_e4`完成四轮、80,064 updates、零AMP skips，14项CPU读回全部通过。Full-validation NLL选第四轮。此配置仍使用1 mm `reducer=count`，局部窗口4 mm；不是native响应，也不是binary OR。

Condor释放的H100立即运行task9的64-pitch控制。Kubeflow仍Pending且没有node；保留现有Job，修改唯一的Job总时限，未创建新副本、未迁移Running任务。时限从21,600 s改为43,200 s，因Job时限包含排队时间。实际训练仍为冻结四轮，GPU组织预算仍为120 GPUh。

## 详细数据

结果根：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_2026-11-12`。

| Measurement | Actual value |
| --- | --- |
| Model | 4 mm context-latent, 1 mm pixel representation, seed42 |
| Parameters | 749,868 |
| Macro AP | 0.321516517 |
| NLL from training history | 1.356914366 |
| Macro F1 | 0.314125560 |
| Balanced accuracy | 0.328900000 |
| AP [e, mu, tau, NC] | [0.395244044, 0.307243947, 0.288937201, 0.294640877] |
| Validation sample | 10,000 events, 2,500 per class |
| Allocation | 3,150 s = 0.875000000 GPUh |
| Terminal Job | `1573835.7`, bigbird20, JobStatus4/ExitCode0 |
| Completed screen/readback | 8/17 models, 112 CPU checks |

原始证据为模型`manifest.json`、`history.json`、`final_checkpoint_checks.json`及Condor event log。4 mm native对照尚在第四轮，2×2尚未完整，未在本阶段选择确认候选。

Kubeflow更新实际时间2026-11-12 17:27:16 UTC；Job `pinpoint-s3-anyqueue-slot-v1`，UID `0faac6e9-b702-4994-ad6b-7a09f91d3540`。更新前后UID和完整`spec.template`逐项相等，只有Job的`activeDeadlineSeconds`修改。GPU仍为`NVIDIA-H100-NVL`＋`nvidia.com/gpu:1`，没有改CPU/RAM或输入。证据`kubeflow/deadline_extension_pinpoint-s3-anyqueue-slot-v1.json`。

Pod注入的`oauth2-sidecar`为原生`initContainer`且`restartPolicy=Always`，主容器为`s3`；使用平台原生sidecar生命周期。此次读取仅包含容器名和restart policy，未保存认证环境或secret内容。

此前最近账本为16:52:59 UTC：11.585495156 GPUh，加首次交互profile保守上界后11.766637769 GPUh，missing0；恢复task3的两次attempt已合并为5,843 s且均终态。该账本早于本次4 mm coarse模型终态，仍为进行中快照。

同期tracker首三轮macro AP为0.307410/0.319131/0.346416，NLL为1.369851/1.360902/1.348154；仍训练第四轮，不用中间结果晋级。

## 阶段性结论

事实：coarse 4 mm控制训练与读回完成，具有高于平衡类prior0.25的四分类AP点值。它与fine／1 mm常数预测模型的差异须在完整冻结配对中报告，并结合阶段65的优化诊断解释，不能直接当作native信息损失的因果证据。

Kubeflow延长时限保留了原队列位置；它尚未获得GPU，因此没有将Pending时间记作GPU allocation。下一项实际获配仍由Condor运行，符合先获配先使用的调度要求。

## 下一步计划

继续剩余9项screen的训练与读回，完成全部四个2×2单元、完整四分类与成本统计后再冻结确认。后台watcher原进程有独立时限：只在确认旧进程退出后续接，不能启动多个handoff watcher。全部screen和确认结束后终结未使用的GPU slot，读回所有后端终态成本，并执行最终strict gate。
