# S4：12GB MIG候选与资源利用

## 阶段目标

为相同冻结实验增加可用GPU队列入口，减少等待，记录主机内存与数据准备的实际限制。

## 工作内容与详细数据

2026-11-13约20:10 UTC，cluster2198680的`condor_q -better-analyze`显示0个愿意立即运行的slot、50个在drained条件下可匹配的slot。不能将后者视为现成空闲资源。当前pool包括whole H100/H200/A100及24GB H100 MIG；即时库存另有7个12GB H100 MIG，均Busy，因此新增候选不承诺立刻获配。

现有完整监督训练的峰值allocated tensor显存为807.88–978.10MiB，MAE3为818.44–832.05MiB，batch64 EMA pilot为1201.70MiB。它们是PyTorch tensor allocator测量，不是设备总显存。新增`native-bf16-small`仅匹配平台公布的`NVIDIA H100L-1-12C MIG 1g.12gb`，申请4CPU/12GB主机内存、至少10,000MiB设备显存，每allocation仅一个训练程序。worker、样本、steps、优化器和模型完全复用原配置，继续共享claims；只要其他后端先领取就跳过，不重复训练。当前LCG CUDA build不支持V100/sm70，不能以改成FP32作为可运行保证。

Kubeflow恢复wave为4并发，20:11左右cgroup内存为28,422,443,008bytes，峰值37,596,946,432bytes，limit62,999,998,464bytes；此前10并发的host OOM已计入成本。`nvidia-smi`对MIG设备的utilization字段返回N/A，pod metrics API对当前账户不可读，因此本轮没有新的可靠GPU利用率百分比，不以显存占用代替利用率。

实际训练日志显示数据准备存在等待：Scratch1k seed17第一epoch step500累计data wait139.79787s、forward/backward30.26247s、transfer0.90727s，总训练172.21622s；MAE3→10k seed42的epoch5约84.03events/s。数据等待同时包含worker处理、stage/decode和调度，未进一步分解，不能全部归因于EOS带宽。当前继续使用已冻结sampler及bounded shard cache，保证比较条件一致；独立候选allocation可推进其他任务。

20:08:56 UTC的成本快照为3.28264513 device-hours、74.54728806 requested-CPU core-hours，上限60 device-hours。包括先前编译、故障、空worker及Kubeflow训练container申请的CPU；不同MIG/整卡时长单列，不转换为等价硬件FLOPs。

## 验证与阶段性结论

新增的是硬件匹配分支与并发限制，不改变训练数值路径。LCG108环境的`python -m analysis.flavour.s4.submit --help`及`python -m py_compile src/analysis/flavour/s4/submit.py`均通过。作业须在实际获配后确认设备和首批训练，尚未运行的任务不记为已完成。

CPU进度评价v3已按commit434efc3提交cluster2198902（bigbird21），补上旧配置默认program的严格一致性检查。完成后检查完整结果和图表。

## 下一步计划

继续比较两个后端的即时终态与claims；先获配者推进相同6项预训练/等预算任务。所有正式模型继续执行内部健康与学习读回；等独立seed、full/frozen下游完成后汇总物理结论。
