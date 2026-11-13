# S4：真实事件前端等价与CPU耗时

## 阶段目标

确认删除未使用的S2分组不会改变S3/S4训练输入，量化其CPU transform收益，并用于后续未启动的冻结任务。

## 工作内容与详细数据

正式CPU作业2199204，commit29b9cf8，在4CPU/12GB申请下完成65.660秒。原train按每类occupancy的8个分位点选择32个事件，包括空事件和各类最大占用，合计占用范围0–36,549通道。每事件比较完整observation及固定`.75`遮蔽后的visible observation；queries/targets不参与此输入等价检查。原路径与跳过未用分组路径交替运行，各3次。

全部`nodes/groups/modules/raw_counts`数组及全部collated tensors逐位相等。没有改变native通道、window、module/region归属、sampler、随机种子、训练steps或梯度路径。先前的51项S2/S3/S4检查全部通过；本轮真实事件补充了最大占用和实际多模态响应。该变化不增加模型或前端表示的自由参数，不需要另定义科学baseline。

| CPU transform视图 | 原路径各调用耗时之和 (s) | 跳过重复分组 (s) | 测得加速比 |
| --- | ---: | ---: | ---: |
| Complete observation | 7.95049141 | 4.49209245 | 1.76988597 |
| Visible after masking | 2.54129903 | 1.59089778 | 1.59739932 |

这些是CPU transform调用的墙钟耗时之和，排除native文件读取、collation和模型；3个worker同时运行，不能将上述和当成作业总耗时或GPU端到端加速比。数据stage/decode仍有开销，后续训练按实际events/s检查收益。

产物：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13/experiments/s4_frontend_equivalence_s42/equivalence.json`，含全部32个事件的逐视图数据；[精简表](../roadmap/results/s4_frontend_equivalence.json)纳入仓库。

补充：MAE3 seed42→10k标签已完成，best epoch3，macro AP0.44504900、τ AP0.33378495、NLL1.25460715。其checkpoint独立读回和配对区间由CPU进度v4执行，此处尚不作完整验收。12GB H100 MIG上的EMA已推进至至少200steps/12,732训练exposures，3次内部监测通过；正在运行，尚无完整预训练结论。

此前失败的Kubeflow `pinpoint-s4-main-b`在确认终态并保存Pod/Job原始JSON后已删除；OOM allocation仍保留于总成本。当前恢复Pod继续运行。另修正readback报告的源代码字段：正式batch使用`PINPOINT_CODE_COMMIT`，同时记录训练manifest的commit，避免仓库HEAD在作业运行时变化而误标评价版本；14项合同和训练数据不变。

## 阶段性结论

真实输入逐位等价，CPU前端的重复工作可去除。继续运行现有正常任务；新获配worker使用已验证的提交源码，竞争相同冻结任务claims。没有重启或重做已完成的科学实验。

## 验证与下一步计划

运行readback脚本的Python编译检查；提交CPU v4并用优化源码补充一个正式GPU候选wave，优先推进独立MAE seeds的低标签下游。完整Rel完成后补做相同512-event的事件置换/几何诊断。最终保留严格完整矩阵、配对四分类/τ子道、matched-compute和allocation终态验收。
