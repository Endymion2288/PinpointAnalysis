# S4：主机内存故障恢复与预算重放结果

## 阶段目标

恢复正式训练，限制并发的主机内存成本；保留已完成结果和epoch checkpoints，故障不进入科学比较。

## 实测故障与处理

Kubeflow `pinpoint-s4-main-b-sc6hw` 的训练container于2026-11-13 19:19:08 UTC以exit137、`OOMKilled`终止。该Pod资源为22CPU/63GB host RAM、H100 47GB MIG；10并发对host RAM的要求过高。不能把GPU显存余量当作主机内存余量。两项Rel pilots已完成并通过读回；其余10项当时活动任务被中断，保存的5项epoch checkpoint保留，未保存完整epoch的任务需重跑。

`audit/backends/`保留Pod/Job终态，`logs/main_b_oom_recovery.json`记录每项checkpoint和释放的claim；只有该已终止allocation的running claims被移入`audit/abandoned_claims/`。新的 recovery wave 合并原主矩阵、独立MAE seeds和per-channel Rel/random frozen控制，配置不改名、不改训练设置；运行并发降为4。Scheduler新增cgroup memory.current/max/peak记录，已有任务活动时主机内存超过75%则推迟新任务；这个启动限制与并发上限共同使用。

大尺寸CPU frozen-probe spectrum改用centered FP64 covariance eigendecomposition，避免对100k×feature_dim执行完整tall SVD；不抽样、不改变事件集合或训练梯度。小batch内部健康继续用原SVD。回归直接比较两种谱算法。

## 详细数据

完整MAE3 shape replay：18,432optimizer steps、294,497实际train exposures，matrix-operation proxy **1,804,323,309,279,552**；每epoch分别601,371,884,865,600 / 600,834,554,005,632 / 602,116,870,408,320。代理包含声明的Linear/attention前后向与eager checkpoint recomputation，不代表全部硬件FLOPs。

其余重复监督shape reads由一次性精确event-shape table替代，原CPU cluster2198174在MAE profile完整落盘后移除；记录`logs/budget_replay_cached_replacement.json`。新的正式matched-plan cluster1574117已在`b9g34p8702`运行，4CPU；将冻结原Scratch完整计算量及额外监督steps，数值尚未完成。

## 验证

包含FP64 spectrum与SVD比对的22项S4测试通过（45.333秒）。加入cgroup计量的有限/无限额度检查后，23项全部通过（48.791秒）。恢复后的吞吐、host peak和所有allocation成本将在后续读回报告。

## 阶段性结论与下一步

此次为并发资源配置问题，不提供分类或架构效果证据。以4并发恢复epoch checkpoints，监测实际host memory；完成独立seeds、等预算监督、完整validation后给出S4结论。
