# S6：BF16回放通过与calibration转换扩展

## 阶段目标

定位stage207 FP32回放差异，固定可靠推理模式；用完整转换pilot的实际吞吐和内存安排全calibration转换。

## 工作内容

在Kubeflow按CERN flavor申请H100 MIG 3g.47gb，同时向HTCondor提交GPU候选。Kubeflow先启动，取消尚处于idle的Condor1582694，避免重复占卡。正式回放使用原BF16与compiled encoder，严格载入13个checkpoint；64个既有validation事件按每类16个occupancy分位点选择，含最高occupancy尾部。

新增共享权重常驻的推理worker及allocation内多process lanes；通过EOS原子目录claim领取已完成CPU转换的任务，每任务只有一个writer。任务完成后发布概率Parquet及summary，失败不发布完成标记。没有可领取任务时最多等待60秒，释放空闲GPU。待测batch大小的profile只读取calibration，不触碰test。

## 详细数据

回放GPU为NVIDIA H100 NVL MIG 3g.47gb，PyTorch2.7.1，BF16+compiled，batch16、workers4。64事件、13模型用47.111 s，其中首次编译主要为28.169 s和9.548 s；GPU peak336.69 MiB，父进程peak RSS2410.39 MiB。三个representation覆盖110105/88489/109621节点。

所有13模型最大概率差均小于0.003；最大为MAE seed2026的0.000734627，其余最大不超过0.000308306。所有首批内部health记录通过。证据为`results/flavour/s6_2026-11-14/audit/h100_compiled_replay64.json`。Pod `pinpoint-s6-bf16-replay47-c8k47`主container于12:24:25 UTC exit0，Pod终态Succeeded，已保存`runtime/pinpoint-s6-bf16-replay47.terminal.json`。正式推理采用BF16+compiled，T4 FP32不进入正式分数。

完整CPU pilot任务0转换3175个calibration事件，用680.140 s，peak RSS2448.289 MiB，输出77143157 bytes。事件键数值与原manifest逐项完全一致，无重复、无缺失；原manifest键为int32、生成表为int64，修正finalize使用值比较，避免将等值整型判为不同事件。truth也是3175行。证据为`audit/conversion_pilot_coverage.json`及`shards/calibration_0000/summary.json`。

其余226个calibration任务提交为cluster2207854，schedd bigbird21；实时队列选择记录于`runtime/s6_calibration_conversion_bulk.submission.json`。每任务2CPU/8000MiB、max_materialize64；pilot显示实际CPU主要由Python逐事件处理消耗，解压线程与较低CPU申请匹配。AFS仅保留小脚本，所有输出/log位于EOS。Test仍未转换或推理。

## 阶段性结论

stage207回放差异在恢复原BF16+compiled模式后消失至固定容差内，支持执行精度/模式是其原因；未修改checkpoint或模型结构。完整输入转换可在既定资源内完成，已扩大到全部calibration。

## 下一步计划

测量batch16/64/128的吞吐、GPU内存及概率差，固定推理policy；并行处理calibration并拟合temperature。继续构建generator输入桥接与真实paired transport接口，待校准、阈值和变体合同全部冻结后才开放test。
