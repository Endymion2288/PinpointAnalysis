# R1：B2 四类三 seed 齐备

## 阶段目标

完成B2最后一个四类seed的独立读回，区分已测四类结果与仍在执行的三类及等时控制，更新真实进度和资源账本。

## 工作内容

`final_b2_t4_s2026`在H100 NVL、allocation1582929.0完成，随后同allocation的T3已实际开始优化。独立读取其Tune预测与Train曝光，重算BA/NLL、逐类recall及raw confusion，并检查五个分支的梯度、更新和activation。**24项通过、0项失败**，证据为`R1/audit/b2_t4_s2026_stage270.json`。NLL读回前在float64中归一化存储概率，与记录的最大绝对差为3.69e−10。

## 详细数据

三个B2 T4均在H100 NVL、eager BF16 encoder/FP32 head、100k unique与400k曝光下完成，表中为开发Tune点估计。

| Seed | Conditional BA3 (%) | NLL3 | BA4 (%) | NLL4 | Tau efficiency at 1% / .1% FPR (%) |
| --- | ---: | ---: | ---: | ---: | --- |
| 17 | 55.170 | .8952693855 | 43.2950 | 1.2169480848 | 3.15 / .51 |
| 42 | 55.670 | .8917917944 | 43.4475 | 1.2151854297 | 2.82 / .38 |
| 2026 | 55.3333 | .8905415760 | 43.3550 | 1.2128326226 | 3.01 / .43 |
| Realized-seed mean | 55.3911 | .8925342520 | 43.3658 | 1.2149887124 | 2.9933 / .4400 |

新seed2026操作wall为**4,183.187681 s**，训练3,918.537014 s，validation合计259.414502 s，其中full Tune inference172.958800 s；data wait47.701162 s，updates50,044。每个Train事件精确曝光4次，各类100k次；40k Tune身份、标签和几何/occupancy metadata与冻结catalog一致。

新模型T3 recall e/μ/NC为58.73/50.51/56.76%；T4 recall e/μ/τ/NC为51.42/44.03/31.34/46.63%。52个health samples中`native_local/native_global/native/tracker/head`均有finite正梯度及正参数更新；所有activation finite，38,918个不同概率向量。真实学习检查通过。

06:05 UTC seed2026的后续T3已达到750次updates、5,994次曝光。B2 seed17 T3继续运行；R1已完成**37/41**模型，剩余两seed的T3及compute共四项。此时完整133配对、21项mean-seed比较和最终accept尚未执行；表中三seed均值没有新增区间，不修改四pass冻结排名或研究晋级结论。

06:05:43 UTC重新查询终态与现有allocation后，R1累计**41.709302 allocation device-hours**、**286.639009 allocated CPU core-hours**。仅两个B2 allocation active，无Idle候选；所有旧尝试、迁移及探针继续包含在成本内。这是截至该时点成本，异构小时不是FLOP等效，最终成本待两个allocation终态。

## 验证命令与产物

```bash
condor_tail -name bigbird20.cern.ch -maxbytes 800 1582929
python -m analysis.flavour.v2.r1.accounting --root "$R1_ROOT" --cached-kubeflow
git diff --check
```

独立读回使用LCG108与单线程BLAS、现有`KEY/equal_identity_rows`及sklearn。原始预测、曝光、health及成本在`R1`下；`R1_ROOT=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R1`。Cal-A/Cut-B/Confirm payload保持封存。本阶段没有模型或评价代码改动，不重复已通过的26 tests。

## 阶段性结论

**事实：**B2四类三个seed齐备，新完成模型有效学习并通过身份、曝光、指标及健康读回。Conditional BA3均值55.3911%、τ1%效率均值2.9933%是已实现seed的点估计；完整区间和相对基线质量判定仍待统一统计。

## 下一步计划

两个现有allocation继续完成T3及同allocation等时控制。全部41模型齐备后执行统一400次配对统计、完整独立读回、图表目视检查和终态记账；增强accept全部通过后再导出最终R1结果及更新W04/W05/W06/W08。不启动R2训练。
