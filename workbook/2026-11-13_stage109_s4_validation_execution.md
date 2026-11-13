# S4：恢复运行、条件统计和正式CPU评价

## 阶段目标

在主机内存可控的正式allocation中恢复完整矩阵；为checkpoint/数值与四分类、τ子道比较提供统一CPU评价作业。

## 工作内容与详细数据

`pinpoint-s4-recovery-c-9pxgf` 于19:29:39 UTC启动，4并发，真实运行MAE独立seed17/2026、batch64 EMA pilot和MAE3 1k恢复。19:33–19:37 cgroup current约30–33GiB，peak33.27GiB，limit58.67GiB；未再出现OOM。此为主机RAM，GPU MIG利用率接口返回N/A，不能据此报告GPU百分比。实际SSL吞吐约160–180 events/s（seed17）、140–165 events/s（seed2026），EMA batch64约50–58 events/s；任务与I/O不同，不作跨方法速度优劣结论。

独立SSL seeds实际claim后，按JobStatus==1移除Condor重复备份1574110/2198175；仍保留与主矩阵不同的channel Rel备用任务1574116。20个终态且队列为空的S4 AFS小提交文件已清理，共15,758 bytes，内容与EOS submission archive完全一致；活动脚本保留。

19:34 accounting为 **2.34661171 device-hours、58.89788752 CPU core-hours**；这是一份运行快照，包含OOM和V100失败allocation，不是最终成本。MIG/whole不换算等价FLOPs。

Matched supervised planning已测得1k额外8×873steps、误差0.0306506%；10k额外8×742steps、误差0.0695624%。原监督完整预算分别1.0631279172×10^15与3.0623671179×10^15 matrix operations，新增MAE目标1.8043233093×10^15。100k table仍由正式4CPU Condor处理，未完成值不推测。

新增 `validation_job.py`：CPU检查所有已完成train的14项checkpoint合同和内部health，汇总Head激活/梯度，再执行固定validation 400次配对bootstrap。提交时冻结campaigns/comparisons依赖文件，不读取后续修改的配置。完整矩阵尚未完成，progress评价明确记录pending；最终验收使用strict完整评价。绘图按fine-tune/frozen-full/frozen-global分页。

Condor host RAM申请改为8,000MB/并发（至少12,000MB）；2并发16,000MB兼容24GB H100 MIG槽位的6CPU/18GB host quota。现有queued channel Rel request已同步修正并记录。

## 验证与阶段性结论

恢复代码的23项S4检查通过；CPU评价将实际验证所有可用产物，完整结论待矩阵完成。1k MAE3在前五epoch的validation NLL从1.4023升到4.1371，同时train loss下降，当前健康正常；这是低标签过拟合证据，保留epoch1 NLL选择，不把后期过拟合作为死Head故障。

## 下一步计划

完成全样本预算规划和继续监督训练、独立seeds、Rel/EMA读回；完成400次paired statistics、谱/τ子道图表、allocation终态及S4结论。
