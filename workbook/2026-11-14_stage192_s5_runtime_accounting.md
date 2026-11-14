# S5 恢复运行与资源证据保存

## 阶段目标

确认降低DataLoader主存开销后的并发训练能推进，并保存容易被平台清理的终态与完整成本。

## 工作内容

恢复Job为`pinpoint-s5-recovery-a`，47GB H100 NVL MIG、CPU22/RAM63G，由平台注入。十一项新名称配置依次运行、最多六并发。原HTCondor GPU候补1581724仍idle且指向旧快照，已在保存queue广告后移除。新增`watch.py`以Kubernetes watch持久化Pod终态；`accounting.py`汇总每份Condor历史与每个Kubeflow allocation，复用supervisor不重复计费，admission失败计零。

## 详细数据

恢复Pod启动后09:41:30 UTC主存29,287,374,848 bytes，peak29,472,509,952；首轮validation之后09:45:42主存44,385,075,200、peak44,877,750,272。两次cgroup记录均无OOM、无oom_kill。六项均已写出epoch1：charm/kinematics/combined/uncertainty/objects/vertex-kinematics validation NLL分别1.318133/1.311335/1.319317/1.307385/1.294078/1.311328。尚未完成，不能作为最终比较。

09:45成本快照：远程device-hours0.501616314（含失败首allocation的Job终态时间上界），CPU allocation27.346670017 core-hours。两次真实T4拟合实测106.023407 s另列；interactive allocation边界未记录，不能将其写成精确分配小时。60 device-hour预算尚有充足空间；MIG与整卡不做吞吐等价换算。详细记录`campaign_accounting.json`与`runtime/backends/`。

GPU利用率未测量：MIG nvidia-smi返回N/A，metrics API返回Forbidden；训练显存约0.93 GB/fit只用于判断显存空间，不代表硬件利用率。并发依据主存、CPU和实际完整训练耗时共同判断。

## 阶段性结论

六并发在首轮validation后保持约44.4 GB主存，低于平台63G配置；仍需完成后续全部passes。新资源统计包括失败、未调度和零allocation尝试，保留无法取到精确Pod终态的证据边界。

## 下一步计划

先对四项已完成神经网络和三个HGB运行统一评价入口的诊断子矩阵，检查辅助误差、分母与图表；最终结论必须使用全部十五项screen及三seed确认，不能由已完成子集替代。
