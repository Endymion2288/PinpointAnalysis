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
2. 5 个训练进程的内存总量约 8.98 GiB，明显低于 47GB 分区；截至记录时所有进程都正常运行。
3. 进程级并发会增加数据读取竞争，最终吞吐以每个 experiment 的 `progress.json` 和完成时间为准；不能用单进程吞吐直接外推总加速比。

# 下一步计划

继续监控 5 个进程，等待各自 `manifest.json` 进入 `complete`，随后运行 CPU readback、统一四分类评价和 accounting。若某一进程先完成，保留其结果并在同一 Pod 中补充尚未完成的 screen task；不再创建第二个 Kubeflow Job。
