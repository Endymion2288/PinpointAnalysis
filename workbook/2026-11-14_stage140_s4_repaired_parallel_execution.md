# 修复矩阵的双平台并发执行与终态回收

## 阶段目标

确认数值修复进入真实完整训练，保留每一步证据，尽快利用可获配GPU完成成对矩阵。

## 工作内容与详细数据

完整模型使用冻结源码715ed4a。HTCondor 1578687和1578688分别获配H100 24GB MIG，以2并发运行100k下seed17及seed42的Scratch/MAE3；12GB候选1578689仍排队。Kubeflow旧作业`pinpoint-s4-recovery-c`全部原任务已完成，Pod实际为Succeeded，main container退出码0，00:04:35 UTC结束。保存完整Pod终态至`audit/backends/pinpoint-s4-recovery-c.pods.json`后删除旧Job，其释放记录位于`audit/released_pinpoint-s4-recovery-c.json`。

随后新Job `pinpoint-s4-empty-main`以CERN flavor `h100-47gb`获配，Pod为`pinpoint-s4-empty-main-s9998`，同一H100 47GB MIG内6并发。此时领取100k seed2026成对及10k seed17/42成对任务。与两个Condor allocation共10个训练并行，没有重复writer。Pod起步cgroup用量约25GB，随后约31GB；limit为63,000,000,000 bytes。最终峰值尚未测量，MIG GPU利用率接口不可用，未填入推测数值。

在100k seed17实际各500个updates时，Scratch/MAE3的最大裁剪前norm分别为153.924347 / 48.508270；此后已记录更新均未超过审查界限。全部更新计数在progress中持续增加，正式完整epoch和最佳validation指标此时尚未测量。

`python -m unittest discover -s tests -p 'test_flavour*.py'`在LCG108环境下169项测试全部通过，用时21.005 s；包括既有resume与checkpoint恢复、数值故障、空路径隔离、健康证据及统计合同。图表已人工查看，原两个异常Scratch在梯度范围与最大norm图中显式标红。

00:10:29 UTC累计19.375185145/60 device-hours，剩余40.624814855 device-hours；CPU allocation233.275013522 core-hours。所有GPU分型分别保留，含旧失败/初始化/新gate/新allocation，不将并发模型时长相加作为硬件费用。

专用验收现在要求实际train catalog的最大占用事件出现在BF16压力测试中，非空原生节点数与represented nodes一致，空/最大占用分别有inference和backward；同时保留原用户已接受的样本和detector-proxy，不重复认证。

## 阶段性结论

新路由在真实完整训练中已消除先前初始化出现的10¹²量级范数，两个GPU平台的10并发任务运行正常。尚未以未完成epoch、技术测试或pilot拟合代替独立validation结果。

## 下一步计划

完成全部18模型；每个完成模型执行独立best FP32 backward及14项读回，随后统一9项配对、3项三seed均值比较、四分类/τ条件指标与英文图。最终回收allocation、刷新费用并以`accept_empty`检查技术完成情况；AP/NLL是否改善由实际数据另作判断。
