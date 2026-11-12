# 阶段62：S3 tracker关系坐标的实际定义

## 阶段目标

明确S3 module relations中tracker坐标的实际信息层级，保持当前输入和科学合同冻结。S3仍在训练中。

## 工作内容

沿代码路径`analysis.flavour.s2.representation.token_inputs`→`s3.dataset.plane_tokens`→`collate_planes`→`ModuleAttention`核对关系坐标语义。Pixel和bar使用硬件z[mm]/3000；accepted tracker proxy在S2已将station index乘1000构造元数据坐标，归一化后为station/3，S3沿用这一表示。它没有提供实际crossing z或真实飞行距离。

文档明确跨tracker的Δz是station元数据关系，不能标成测得的毫米距离；以后需测试nominal tracker-plane物理z时，应建立独立对照。新增manifest描述与代码注释；没有修改任何输入数组、proxy值、模型算子、配置、seed或训练流程。前面已启动模型的manifest不追写，本报告和roadmap为整个S3提供统一语义说明。尚未启动的tracker任务将记录更明确的manifest字段。

## 详细数据

证据为当前代码：`src/analysis/flavour/s2/representation.py`中tracker `xyz=(xy, station*1000)`及`xyz/[210,210,3000]`；`s3.dataset`保留该坐标，并用`n_layers+station`建立module ID；`s3.models.ModuleAttention`在当前z字段上形成差值。这里是程序定义核对，不重新认证samples，也不调整accepted proxy。

新增`module_z_coordinate_definition`为`pixel_bar_physical_z_mm_div3000_tracker_station_index_div3_not_physical_distance`。实际nominal tracker-plane z、飞行时间、拟合q/p：当前S3均未测量或输入。

结果根：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s3_2026-11-12`。本次重新查询确认6项Condor科学作业Running，Kubeflow slot Pending且无node；队列/完成输出watcher均由真实进程查询确认存活。尚无完整4-epoch模型。

新增首轮记录：无module-context的f16 latent完成epoch1，NLL1.386621、macro AP0.251924；a4 late与f16 late仍完成1轮，后续轮次进行中。这些是首轮点估计，不用于Pareto选择或S3收益结论。完整值保存在各实验`history.json`及`monitor_state.json`。

本次只改描述与注释，无模型行为变化，不重复GPU压力测试或统计测试。先前27项执行/统计/后端、9项四类工作点/决策、3项交接及6项精确验收检查的相关结果分别保留。

## 阶段性结论

事实：S3的tracker分支使用accepted hit/station表示，关系坐标是元数据。当前分类是否利用该信息、是否改善各类性能，尚无完整screen证据。

假设：真实nominal站点距离可能改变关系学习，需要未来独立对照，不能从当前结果推断其收益。该语义核对不解释不含tracker的模型首轮表现。

## 下一步计划

保持17项训练与原判据，使用完成输出watcher执行CPU读回。全部screen和终态计费完整后，再四类评价、2×2交互、冻结三seed确认。保留主模型与1 mm对照、条件域/资源/错误组，最终严格验收后才能完成S3。
