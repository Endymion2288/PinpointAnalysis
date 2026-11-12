# 阶段目标

在首轮17项完整screen基础上冻结候选，并使用同一个Kubeflow MIG分区并发完成三seed确认的新增8项训练。

# 工作内容

冻结Pareto选择给出候选task14（tracker context latent）和task0（4 mm late fusion），连同预定task5/6分辨率对照组成确认矩阵。每个结构新增seed17/2026，保持4 epochs、batch4、80,064 updates、accepted输入及其它超参数。配置已提交为`1a14d3e`，无数据重认证或calibration/test使用。

Job `pinpoint-s3-confirm-mig47-parallel-v1`申请`nvidia.com/mig-3g.47gb:1`，平台注入22 CPU/63G RAM，已Running于`ml-004-h100-001-uusckqrqfxt3-node-0`。最多并发4个独立训练，首先运行task0/1/6/7，退出后自动填补task2/3/4/5。没有并行创建第二个GPU Job，也没有重复提交HTCondor科学任务。

新增finalize入口，自动等待8项readback与Job Complete后运行完整评价。每个阶段状态写EOS；发生错误时记录实际失败阶段和错误，停止后续步骤，不提前报告验收通过。将来若Pod先被GC，保留的terminal Job起止时间可提供明确标记的reservation保守上界，不将未知Pod历史当零成本。

# 详细数据

约22:56 UTC进度：4个训练均为epoch1；4 mm late的seed17/2026各3750 step，tracker的seed17为2250、seed2026为2500 step，AMP skips均为0。三seed最终指标尚未测量。

40项相关单元测试通过，包含共享成本守恒、首个子进程退出后继续等待其余训练、Pod GC的上界及缺失历史处理、整卡/MIG效率结论隔离、四分类统计与晋级规则。`git diff --check`和Python编译检查通过。

8项训练产物和每个子进程日志在：

`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_2026-11-12/kubeflow/`

其下`parallel_pinpoint-s3-confirm-mig47-parallel-v1.json`记录进程起止与exit code；`confirmation_finalize.log`和根目录`confirmation_finalize_state.json`记录后续评价。最终图表和统计写`evaluation_final/`，最终门槛写`final_acceptance.json`。这些路径的存在不代表所有阶段已完成，应读取实际status。

HTCondor cluster1573917已无队列任务，其AFS `job.sub`和`run.sh`已复制到EOS `submission_archives/`，随后从AFS清理；证据为`afs_submission_cleanup_final_screen.json`。大日志及模型一直保存在EOS。

# 阶段性结论

四进程并发的确认训练已实际启动，不需要继续等待4个独立GPU allocation。父进程等待全部子训练，解决先前手动后台进程的容器生命周期问题。当前没有并发后的NVML利用率百分比，不报告未测量的硬件利用率或加速倍数。

# 下一步计划

确认8项训练及readback均通过；等待finalize产生三seed统计、条件域、事件图示和strict gate。随后据实际数据撰写最终中文科学报告并提交。S3尚未完成，S4–S6未执行。
