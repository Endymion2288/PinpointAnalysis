# S5 双后端确认与flavor选择

## 阶段目标

让可用资源立即执行未认领的任务，完成三seed分类确认及运动学修复。

## 工作内容

HTCondor1581763已使用H100 MIG2g.24gb执行direct4 seeds17/42。Kubeflow whole H10094的server admission配置为CPU46/RAM127G，GPU本身准入通过，但调度事件连续报告GPU/CPU/memory不足；Pod未分配节点、无startTime。保存Job/Pod/events后于10:12:41 UTC删除，无device/CPU allocation。

随后使用已验证的H10047GB MIG重新提交`pinpoint-s5-confirmation-b`，10:13:27开始。六并发为两项4-pass component-head repair、direct4 seed2026、vertex+kinematics三seed。后续空位执行尚未被Condor认领的six-way任务；共同claims保证单个experiment只有一个writer。

资源统计增加未调度取消的明确零allocation分支，watch在Pod删除时也停止并保存最后快照，防止无效监听遗留。整个训练继续使用CPU1/OMP1的每fit设置、train worker1与validation worker0。运行输出均在EOS；AFS只保存小型提交脚本。

## 详细数据

当前科学矩阵为26个神经网络+3个HGB，十五项screen已完成；确认与两项repair刚开始，尚未测得最终验证结果。whole H100没有用于任何fit，不计入可比较的训练速度。原MIG47恢复allocation已Succeeded，完整terminal Pod证据保存在`runtime/backends/`。

## 阶段性结论

实际调度选择基于可用资源与终态证据；未等待整卡队列。两个后端均在推进训练，旧未学习的横向回归只作诊断对照。

## 下一步计划

完整screen的CPU评价正在执行；完成component repair后先检查四分量误差和符号，再读取三seed分类净贡献，最后统一收尾与去重产物。
