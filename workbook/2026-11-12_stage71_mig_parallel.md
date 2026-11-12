# 阶段目标

利用已经获配的 Kubeflow H100 NVL 3g.47gb MIG 分区，提高 S3 screen 的吞吐，同时保持每个实验独立、可恢复并避免与 HTCondor 重复写出。

# 工作内容

Kubeflow namespace 的 GPU quota 为 1 个 GPU unit，因此没有再创建第二个 Job。当前 Job `pinpoint-s3-anyqueue-mig47-v3` 的 Pod 获得 `nvidia.com/mig-3g.47gb:1`，并已通过 runtime handoff 启动 task 11。单个 S3 训练进程的峰值显存很低，故在同一 Pod、同一 MIG 分区内以独立 Python 进程并发启动 task 12、13、15、16。每个进程使用冻结 campaign、独立 experiment 输出目录和独立 EOS 日志。

为避免重复训练，先在 `bigbird20.cern.ch` 对 cluster `1573917` 的 proc 1、2、3、4 执行 `condor_hold`，随后执行 `condor_rm`，确认不再保留对应任务，再启动 Pod 内的手动进程。操作记录为：

`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_2026-11-12/kubeflow/manual_parallel_launch_2026-11-12.json`

# 详细数据

运行检查命令：

```bash
kubectl get pod -n xcheng -l job-name=pinpoint-s3-anyqueue-mig47-v3 -o wide
kubectl exec -n xcheng <pod> -c s3 -- nvidia-smi -L
```

实测 GPU 为 `NVIDIA H100 NVL`，暴露一个 `MIG 3g.47gb` device。启动后各训练进程的 `/proc/<pid>/status` 读数如下，单位为 KiB：

| task | experiment | VmRSS |
|---:|---|---:|
| 11 | `s3_f16_context_latent_nopair_s42_e4` | 2,070,692 |
| 12 | `s3_f16_context_latent_pixel_s42_e4` | 1,745,816 |
| 13 | `s3_f16_context_latent_bar_s42_e4` | 1,599,644 |
| 15 | `s3_f16_context_latent_nosignal_s42_e4` | 1,773,772 |
| 16 | `s3_f16_context_latent_q8_s42_e4` | 1,791,136 |

此时 task 11/12/13/15/16 均存在训练进程。启动约 3 分钟后的进度文件读数为：task 11 为 epoch 1 step 11250，task 12 为 step 2000，task 13 为 step 3750，task 15 和 16 均为 step 1750；各进程均报告有限 loss、无 AMP skip，并写入 EOS。

日志路径为：

`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_2026-11-12/kubeflow/manual_parallel_task<task>_*.log`

# 阶段性结论

1. 在当前 quota 约束下，可以通过一个已获配 MIG Pod 内的进程级并发利用 GPU，避免额外 Job 长时间 Pending。
2. 表中 5 个训练父进程的 VmRSS 合计约 8.56 GiB，属于主机内存，不能与 GPU 分区容量比较。CUDA 实测峰值由各模型 manifest 独立报告。
3. 进程级并发会增加数据读取竞争，最终吞吐以每个 experiment 的 `progress.json` 和完成时间为准；不能用单进程吞吐直接外推总加速比。

# 下一步计划

后续结果见阶段72。5 项均完成并通过 CPU readback；更新 accounting，避免将共享 GPU 的各进程时间重复相加。期间曾暂停多个 DataLoader/resource-tracker，导致共同停顿，恢复后训练继续。最后一个 stopped resource-tracker 在模型均发布完成后仍阻止 PID 1 退出，恢复它后 Job 正常 Complete。这些占用全部计入成本。后续使用父进程管理并等待所有子训练退出，不再通过暂停进程维持容器。
