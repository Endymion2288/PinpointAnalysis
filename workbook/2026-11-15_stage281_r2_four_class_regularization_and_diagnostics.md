# Stage281：R2四类正则化控制、三窗口口径与泛化图表

## 阶段目标

在D1 T4完整参考与Train gap形成后，补上相同的weight-decay控制，回答正则化是否改变三类/四类取舍与τ工作点。该控制已在Kubeflow运行，首批D2和D1WD T3仍在HTCondor执行；本阶段不声称完整R2完成。

## 工作内容

stage280只启动T3控制时，T4完整终点尚未形成。随后D1 T4的共同4k Train NLL4=1.012396882、BA4=.5455，而40k Tune NLL4=1.337161873、BA4=.4237，支持相同过拟合分流。τ在1% FPR下，D0 endpoint/D1 endpoint/D1 best-screen效率分别为.0511/.0438/.0534。长期重复曝光没有形成稳定τ收益。

据此冻结[D1WD T4理由](../roadmap/v2/configs/r2/d1wd_t4_reason.md)和[配置](../roadmap/v2/configs/r2/b1_d1wd_t4_s42.json)。只改变AdamW weight decay .01→.1，与已在运行的T3控制采用同一系数；没有额外选择系数。D1WD T4−D1 T4是单因素控制，与D0/D2的差值用于综合选择；补充D1WD T3−conditional T4。初始72项配对完整保留，正式矩阵成为8模型、16视图、128项配对、11个域。

`readout.convergence`新增最后四个screen点的三次连续NLL变化。最后三个点只有两次变化，不能据此满足“连续三窗口变化小于.001”；保留原三点摘要并明确每项含义。工程plateau仍须结合BA、Train/Tune gap和完整预定预算，不称其为物理信息饱和。

英文图增加`r2_train_tune_gap`；`r2_low_occupancy`扩为μ recall、BA3、μ AP和AUROC四面板；endpoint配对图包含D1WD−D1。实际产物在完整模型、Train gap、评价和readout就绪后由watcher提交生成，当前未把新增正式图称为已完成。

## 详细数据

D1WD T4精确preflight：150,153 updates，1,200,000训练曝光，900,000非τ曝光，每行12次，100,000 unique events，sum(LR)=23.595463349912084；与D1 T4完全相同。preflight作业`1582980`在兼容且负载最低的`bigbird20.cern.ch`执行，已完成。

Kubeflow job `pinpoint-r2-d1wd-t4-k4`、Pod `pinpoint-r2-d1wd-t4-k4-kzw7h`，启动约7秒后即Running；使用已释放的H100 MIG3g.47gb服务资源，单独一个模型进程。它与原四进程D0/D1作业不同：后者已于11:51:14 UTC正常完成，资源释放。D1WD T3仍在Condor `1582978`，H100L MIG2g.24gb；D2 T3/T4继续在`1582961`的另一块MIG2g.24gb双并行。

D1WD T3在首个75k screen时记录11次健康检查，image/tracker/head在全部11次记录中均有非零梯度和参数更新，所有激活有限。该模型实际在学习；其后续泛化效果仍待固定终点，不用学习健康代替性能比较。

资源记账在11:55 UTC前后的只读快照为4.429100异构device-hours，仍有D2和D1WD T3两个active allocation，当时T4尚未启动，不能作为最终总成本。分型号/分区单列，单卡双/四进程不重复相加。完整统计以结束后的`resources.json`为准。

软件补查：现成`LCG_107_cuda/x86_64-el9-gcc11-opt`可导入Torch2.5.1/CUDA12.5，Python3.11.9，编译架构字符串为`sm_50 sm_60 sm_61 sm_70 sm_75 sm_80 sm_86 compute_86`。当前v2模型在V100上的forward/backward未测量；该元数据与历史S2执行相容，不把它作为本轮新模型运行证据。结果保存为`R2/audit/lcg107_cuda_build.json`，没有改变当前LCG108/BF16训练。

验证使用LCG108、单线程BLAS/OMP及`PYTHONPATH=$PWD/src`：

```bash
python -m unittest tests.test_flavour_v2_r0 tests.test_flavour_v2_r2 tests.test_flavour_v2_r2_readout
```

22项测试通过；新增反例包含`.1`大幅变化后接两个小变化，不能被最后三点的近似稳定误判为三个稳定窗口。正式matrix函数读回为16视图、128配对。当前代码与配置保存于仓库，训练、图表与模型输出均留results。

## 阶段性结论

事实：T4也出现Train改善而Tune退化，原始τ工作点没有随更长重复训练稳定上升。四类正则化控制已按相同因素、固定系数启动，三类/四类均有匹配参考。

假设：增强weight decay能否减少过拟合、是否保留τ排序和改善共同三类取舍，均等待实际终点；更多unique数据的最终收益也尚未形成。

## 下一步计划

完成D2及D1WD T3/T4，统一执行共同Train gap、400次完整paired区间、独立指标/区间读回、五折readout和正式图表。最终根据同输入配方的收益、收敛/过拟合及成本决定是否还存在必须在R2澄清的数据或容量问题；形成科学决策、终态资源和验收后更新路线状态。
