# S4：Rel事件条件性、几何先验与配对区间

## 阶段目标

检验bar Rel超越常量的微小CE改善是否来自事件信息，区分几何位置先验与对事件响应的利用。

## 工作内容

新增正式CPU `rel_diagnostic.py`，在同一512个训练内probe事件、固定queries/targets和各checkpoint上，以FP32重新执行网络。对旧global-only、kept bar local blocks、pooling前kept bar channels三条路线，使用decoder pre-hook读取实际source tokens和mask；cyclic事件置换整个source与mask，保持query和target不变。这个诊断不改变权重、输入合同或活动GPU任务。

同时构建同probe的oracle constant和每硬件channel几何oracle（按query的物理x/y/z识别channel），报告两类先验与训练模型的CE差、目标entropy和事件置换CE差。几何oracle在同probe拟合，是该先验族的描述性下界，不当独立预测器。400次accepted-family connected-chunk bootstrap使用每事件有效bar count与loss sum，避免把同一event数百channel当独立样本。

## 详细数据与验证

24项S4测试全部通过（47.095秒），新增诊断完整forward/source-hook/目标几何/paired bootstrap路径。固定512个probe的类组成实测为CCνe129、CCνμ118、CCντ129、NC136；完整2000为515/497/481/507。该记录只解释已有内部评价覆盖，保持原accepted数据与split，不改变probe成员。

目前实际三checkpoint语义诊断尚未运行；不得用测试fixture的随机CE作物理结果。对应正式配置`configs/s4_rel_diagnostic.json`。

正式CPU完整validation的第一份进度结果已完成三档Scratch seed42，400replicates，macro AP0.29720868/0.42710017/0.46976869；后续模型与对照尚pending，不能称S4已完成。输出为`evaluation/progress_v1`。

## 阶段性结论与下一步

常量CE优势、几何oracle和event permutation将分别回答不同问题。只有实际诊断和下游分类对照完成后才能解释Rel改善；活动EMA/独立MAE继续按已冻结协议训练。
