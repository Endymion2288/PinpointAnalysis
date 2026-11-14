# S4 分阶段读出恢复的正式执行与存储

## 阶段目标

落实GPU/CPU读出分阶段执行与结果等待，减少无效GPU占用，同时保存可审查的调度证据。

## 工作内容与详细数据

新CPU solver gate已正式运行于bigbird21 cluster2205005，只读取原train1k×两种初始化。04:36:50 UTC第一组1,000事件提取完成，CPU实测约2.124 events/s，进入求解；完整gate尚未通过。

GPU feature候选1578832已正式hold，实际仍未获配，不在等待gate时占用GPU。旧readout allocation2205000已正式移除、history为JobStatus3；Kubeflow旧`pinpoint-s4-transfer4-refits-readouts`删除前保存Job/Pod和active任务，04:32:05 UTC确认Pod不存在的时刻记录为allocation结束保守上界，没有伪造成功终态。对应legacy故障记录和claims均保留。

CPU五组已正式提交：Scratch2205006、MAE32205007、EMA31578834、MAE51578835、channel Rel52205008；完整45预测/66配对评价2205009等待所有有效上游。schedd由每次实时队列、EOS兼容和WRITE选择，精简提交记录见[执行证据](../roadmap/results/s4_stationary_readout_execution.json)，冻结脚本和完整调度依据写入EOS。

已有两组100k正常fine-tune allocation2204998/1578819继续运行，维持原冻结训练。Kubeflow正式两并发检查低标签分支依赖，没有在SSH中临时启动训练。04:34:54 UTC成本33.186903/60 device-hours、374.403684 CPU core-hours，包含失败、停止allocation及初始化。

清理AFS中25个终态提交目录，共55,853 bytes；只有与EOS `submission_*.json`中run.sh/job.sub完全一致且无额外文件的目录才清理。active、queued、held目录保留。清理证据在EOS`audit/afs_terminal_cleanup_20261114_0435.json`。

新增临时feature读取集成测试，覆盖数组/metadata读回、desired事件顺序和encoder维数变更拒绝；连同三项求解/键选择测试均通过。不改变运行中的冻结任务。

## 阶段性结论

事实：正式执行链路和终态/成本记录已落实，GPU新恢复等待真实CPU gate；完整readout数值和学习结论仍待测。CPU等待、GPU提取、模型训练成本分别记录，原60 device-hours不重置。

## 下一步计划

solver gate通过后release Condor GPU候选，并在Kubeflow分支检查完成后提交同一feature campaign竞争claims；同一encoder只写一次临时feature bank。15项CPU readout收敛及完整评价验收后删除暂存数组，保留小型模型和结果。
