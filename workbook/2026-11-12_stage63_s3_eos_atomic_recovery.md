# 阶段63：S3实际EOS写入失败与原子恢复

## 阶段目标

恢复无module-context的f16 latent训练，并针对实际出现的EOS临时文件截断实施完整写入重试。模型、样本、seed和80,064更新合同保持冻结。

## 工作内容

Condor `1573835.3`在第二轮结束保存`last.pt`时真实失败：`os.fsync`返回`OSError: [Errno 71] Protocol error`。原子发布前的`last.tmp`不完整，不能用其恢复第二轮optimizer状态。正式`last.pt`仍完整保留第一轮，既有resume协议以它为唯一权威，回滚领先的best/predictions并重做第二轮。

将共享I/O helper改为整个临时文件的有限重写：遇到EPROTO/EIO/ETIMEDOUT/ECONNRESET/EAGAIN时关闭失败句柄、创建新的临时inode、重新序列化全部内容、flush/fsync成功后才原子替换。最多4次，间隔0.1/0.3/1 s。永久错误或重试耗尽仍失败，不忽略错误，不把临时文件当成功产物。不能只重试同一句柄的fsync，因为本次已实际观察到缺失尾部的Torch archive。

此修改用于checkpoint与Parquet原子写入，训练算子、optimizer、scheduler、RNG和配置未变。已启动进程不热改；后续排队/恢复进程加载新helper。失败尝试和重做计算都计入组织GPU预算。

## 详细数据

结果根：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_2026-11-12`。

| Evidence | Actual value |
| --- | --- |
| Failed experiment | `s3_f16_latent_s42_e4` |
| Source Condor Job | `1573835.3`, bigbird20 |
| Authoritative terminal state | JobStatus4, ExitCode1, NumJobStarts1 |
| Actual allocation | 2,363 s = 0.6563888889 GPUh |
| Published `last.pt` | epoch1, 20,016 updates, 0 AMP skips, 9,991,975 bytes, all tensors finite |
| Failed `last.tmp` | 9,202,816 bytes; Torch archive缺少central directory，无法读取 |
| Ahead `best.pt` | epoch2，finite |
| Ahead epoch table | epoch1/epoch2各10,000行 |
| Recovery start | committed epoch1；从epoch2重做 |

终态通过实时`condor_q`空结果及`condor_history`明确JobStatus4/ExitCode1共同确认，不根据观察超时推断。输入输出检查为CPU只读Torch/Parquet读回，不重新认证样本或调整proxy。证据`recovery_task3_io.json`、`logs/screen_condor_v1.1573835.3.err`及统一event log；失败前training loss和20,016 steps/epoch未出现AMP skips。

7项I/O与恢复检查真实通过，27.232 s，`unit_atomic_replay_recovery.log`。覆盖临时archive实际截断后的全部字节重写、瞬时错误重试、永久错误不重试、重试耗尽保留旧正式checkpoint，以及领先best/predictions/epoch表与损坏last.tmp的恢复。恢复后模型权重、best predictions和完整epoch predictions与不中断训练逐项完全一致。

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest tests.test_flavour_atomic_sync tests.test_flavour_s2_training
```

恢复提交沿用`analysis.flavour.s3.condor train --tasks 3`，提交前再次查询schedd负载和EOS/WRITE兼容性，保留唯一恢复writer。原失败Job已终态。正式seed42/4 epochs/20,016 steps每轮保持原值。

同期训练进度：a4 late第三轮已完成，macro AP0.340190、NLL1.350208，正训练第四轮；预定主模型首轮macro AP0.248448、NLL1.389397，已训练第二轮。上述仍为中间结果。Kubeflow slot仍Pending，交接watcher继续等待真实GPU ready。

待运行task6的实际`condor_q -better-analyze`结果保存在`queue_analysis_task6.txt`：5,320 slots/4,114 machines中当前0个匹配且愿意运行，12个释放资源后可匹配。没有根据队列观察替换冻结GPU型号或减少样本。

## 阶段性结论

事实：本次退出是已定位的EOS协议/写入问题。第二轮临时checkpoint无法恢复；第一轮正式checkpoint完整，既有原子resume能处理领先的best/predictions。有限完整重写已通过故障注入验证。

假设：存储错误的底层服务端触发原因未测量；不从这一作业推断分类结构失败或硬件信息无用。模型收益仍由完整screen与三seed确认判断。

## 下一步计划

### 真实恢复完成，2026-11-12 16:33 UTC补记

恢复作业`1573839.0`实际从epoch1启动，重新发布的epoch2/40,032 updates checkpoint已独立读回finite，随后完成四轮。真实history为JobStatus4、ExitCode0、allocation3,480 s。原失败2,363 s与恢复3,480 s合计5,843 s = 1.623055556 GPUh，全部计入该实验，不用只含有效轮次的程序时间替代实际allocation。

最终`last.pt`为epoch4、80,064 updates、零AMP skips，320,164 training exposures；14项CPU读回全部通过。Full-validation NLL选择epoch2，NLL1.386325198、macro AP0.248288286；第四轮最新macro AP0.249834541不是选中的模型指标。Manifest有效轮次时间合计4,614.870396 s，与真实allocation差1,228.129604 s，包含失败重做和setup等开销。

证据为`recovery_task3_epoch2_published.json`、最终`manifest.json`、`final_checkpoint_checks.json`及恢复event log；读回UTC16:33:05。真实恢复结果证明训练已越过原失败点，不能据此推断EOS服务端触发原因。此次分类表现接近prior，训练完成与分类质量分开评价。

继续全部17项，不删除失败分支或缩减预算合同；将失败/重做成本合并进最终accounting。完成全部screen后再冻结确认、评价完整四分类和2×2交互，并执行最终strict gate。
