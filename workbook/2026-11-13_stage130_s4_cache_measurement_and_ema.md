# S4 固定输入缓存实测与完整EMA读回

## 阶段目标

验证固定分类输入缓存的真实数值与性能，完成EMA内部学习审查，推进剩余下游矩阵。

## 工作内容与详细数据

HTCondor CPU profile1577642（bigbird20，冻结bf833cb）完成原nested 1k标签子集的两次完整遍历。原始/缓存reader交替调用，全部canonical collated tensors及metadata逐位一致。

| Reader | First pass [s] | Second pass [s] |
| --- | ---: | ---: |
| Original native decode/transform | 32.56347 | 32.69101 |
| Immutable classification cache | 33.00652 | 2.13331 |

第二次CPU读取/变换/拼接比值15.324倍。缓存实际为93,819,352bytes（约89.473MiB）、1000个完整事件，1000次miss及随后1000次hit，0次淘汰；全部事件进入缓存后native reader已释放。首次pass耗时相当，证据支持后续重复访问省去native decode/token transform。这里未测模型或GPU端到端吞吐，不能将15.324倍用于训练速度或成本换算。精简产物为`roadmap/results/s4_classification_cache_profile.json`。

profile在schedd中排队约11分钟，165个CPU槽位的better-analyze忽略用户优先级时愿运行，但同一schedd同期出现批量CPU队列；只把本4CPU短诊断JobPrio调到10，未抢占运行任务。动作与readback在`audit/cache_profile_queue_priority.json`。作业随后正常获配，未更改输入、内存、CPU或运行时限。

EMA3完整3×1536 updates完成，程序7034.394秒；10项integrity/internal-health及6项measured-learning全部通过。2k原train-inner probe的fine pixel patch/4mm/bar AP为0.22010417/0.47919746/0.72953945，均超过各自query prior；event-permutation ΔBCE为0.03480455/0.18735404/0.40341889。global跨事件RMS0.963479、effective rank8.089592、leading variance fraction0.237841；teacher及student projection秩为12.062730/13.320962，均不接近rank1。分类收益尚未测量。完整读回精简存于`roadmap/results/s4_ema_readback.json`。

三个matched-compute继续监督均已完成。1k从原best checkpoint开始，继续训练首epoch NLL2.91909，末epoch6.37729，train loss降至0.04485；70个health samples均通过，Head RMS最大0.83528，有效输入梯度保留。100k继续训练best NLL1.21415050。选择仍按预先冻结的原始-or-continuation NLL，计入全部计算成本。新checkpoint统一读回及配对统计待下一次validation。

22:12 UTC组织成本10.686243device-hours/60，CPU allocation152.764254core-hours；没有将并发程序elapsed相加为allocation时长。22:13清理12个已终态AFS提交文件，共10966bytes；逐项核对terminal accounting与EOS保存的原始脚本一致，运行及排队脚本保留。

## 阶段性结论

分类输入缓存对真实1k子集保持数值及事件顺序，显著缩短后续CPU遍历；端到端收益待新worker。EMA完成了非坍缩且有事件条件信息的预训练，不能仅凭重建结果判断分类收益。低标签继续监督的退化具有过拟合证据，未观察到旧Head断连。

## 下一步计划

冻结缓存实现与已验证输入合同，让新worker领取MAE5、Rel/EMA及剩余frozen任务；再测真实data wait、training elapsed和内存。积累新增分类模型后做正式v7与最终strict评价，继续保持calibration/test冻结。
