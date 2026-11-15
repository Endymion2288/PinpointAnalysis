# R1：合作组三类对照与独立人数读回

## 阶段目标

将R1与合作组原文档的对应关系整理为可读交付，明确相同任务、实际输入和训练差异；进一步从已有预测概率独立读回人数、weighted confusion与固定purity指标。R1最终同计算交付继续等待剩余训练。

## 工作内容

新增[合作组对照页](../roadmap/v2/08_r1_collaboration.md)，直接读取已有seed42投影控制统计，列出all、pixel-centered与large三个域、六项模型，共18行BA3/AUROC/90% benchmark purity效率。逐项核对Wu PDF p2–4和Mahlstedt PDF p2–7中的图像、网络、tracker、样本与工作点定义。

R1独立读回增加原始confusion、按原始support重加权的confusion，并从raw probabilities重算固定FPR的coverage/purity。固定80%/90% purity使用另一实现`sklearn.metrics.precision_recall_curve`，按完整score ties选择；同recall时保留更高threshold。该检查不调用R1的Metrics实现。

## 详细数据

对照页18行共126个数值均从`R1/interim/projection_controls/evaluation/results.json`逐项核对四舍五入结果；本地链接及语法检查通过。R1与summary的23项测试通过。新增完整概率读回经HTCondor1582927执行，Exit0，allocation 29 s；1,185项检查通过。weighted confusion最大差为5.057×10⁻¹⁰加权事件，coverage/purity最大差为4.258×10⁻¹⁴；纯整数raw confusion逐项一致。完整结果为`R1/interim/b1_three_seed/evaluation/independent_review.json`。

B1 T3的pixel-centered BA3为73.817%，large为68.891%，全域为56.620%。对应AUROC e/μ/NC分别为.9203/.9089/.8644、.9106/.8593/.8211和.8151/.7594/.7339。各域数据和先验在对照页明确列出，不把域差直接当成模型改进。

R1投影为两个1322×420、每个4通道的物理XZ/YZ图，B1 T3为793,545参数、4 passes；Wu原图像分别为strip 68×40/56×40、pixel 15×250/15×200，ResNet带pixel约2.9M参数、40 epochs，tracker为12维重建量。R1的tracker set与这些重建量不同，因此当前交付属于同类方法对照，严格原研究复现尚未完成。缺少的事件身份、完整配置、scores/cuts与tracker特征定义已逐项列出。

03:57:27 UTC资源为35.253826 allocation device-hours，CPU分配255.187254 core-hours；5项GPU训练继续运行，两个候选probe为Idle且尚无allocation。异构小时继续分设备/分区记录，历史MAE SSL另列，原18 h R1规划已超过。

为缩短B2剩余执行时间，新增两个仅Train profile的候选：1582925/1582926，准入为当前兼容Capability与30,000 MiB显存，不限定型号。门槛65 events/s，相对于2g当前约40 events/s，fresh全程仍可能节省数小时；是否实际迁移取决于获配、真实吞吐、旧进度与终态证据。该数值是资源启发式，未改变模型/排名合同。

04:00 UTC 1582925已获配，尚未产生profile结果；1582926仍Idle，现有2g两个writer继续训练。可见Unclaimed广告并不代表本用户立即可用：Condor分析在早期快照没有找到愿意接收的匹配slot，多个较大卡广告的`AvailableGPUs`为空。

04:02 UTC 1582925在H100 NVL得到64.632576 events/s、activation finite，因门槛65被拒绝，Exit0，allocation 130 s。04:03:49 UTC旧2g已经过91,905曝光、实测40.251754 events/s；按三项操作约1.1M曝光、fresh额外600 s余量估计，继续旧任务需6.957 h，fresh在该64.63速率约4.894 h，可节省2.063 h。该推算不是最终实测wall。因此不再用接近65的固定启发式拒绝有明显完成时间收益的候选：新attempt只用1 event/s作为可执行性下限，release前另按当前余量要求至少节省1 h。证据为`R1/audit/b2_remaining_time_stage261.json`；原attempt保持原门槛和拒绝记录。旧writer未因此提前移除。

seed2026的1582926同样正常Exit0，allocation 112 s，profile为61.119355 events/s并因65门槛拒绝。新候选1582928/1582929分别对应seeds17/2026，采用余量判定；当前仍在排队，实际新profile及迁移尚未测量。两次已拒绝探针共242 allocation seconds全部记账，不作为模型学习结果。

运行环境为LCG108、Python3.12、OpenMP/BLAS线程1；大型读回通过HTCondor执行：

```bash
python -m analysis.flavour.v2.r1.review --root "$R1_ROOT/interim/b1_three_seed"
python -m unittest tests.test_flavour_v2_r1 tests.test_flavour_v2_r1_summary
python -m analysis.flavour.v2.r1.accounting --root "$R1_ROOT" --cached-kubeflow
git diff --check
```

## 阶段性结论

**事实：**R1已能提供与合作组相同三类问题的、定义明确的条件域结果；当前表不能量化相对原研究的模型增益。新的独立读回覆盖最终compact产物中人数及工作点的含义。

**未测量：**候选迁移后的全程实际提速、完整41模型133配对、最终资源和验收。Cal-A/Cut-B/Confirm继续封存，R2未启动。

## 下一步计划

完成新增读回并记录结果；候选profile若显示足够剩余时间收益，再按旧writer终态→归档→release迁移。完成R1全部模型、full独立复核和图表核对，关闭资源后运行accept及最终汇总，并更新W08完整状态。
