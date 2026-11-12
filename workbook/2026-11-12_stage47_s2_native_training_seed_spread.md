# 阶段47：ParT native 三次训练的重复性

## 阶段目标

回答固定输入、训练合同和物理事件后，ParT native 的结果随训练 seed 如何变化。验收本组17/42/2026三个模型的终态、四轮历史、checkpoint与逐事件概率，分开记录训练波动与事件统计不确定度。原生相对1 mm、两种架构的四组三seed配对仍按既定合同等待完整结果。

## 工作内容

新增seed17/2026均实际exit0，分别为`1573782.23/24`；与既有seed42一起通过各14项CPU读回。三者使用80,000 train、10,000 validation，各类20,000/2,500，H100 NVL、LCG108、BF16、151,228参数、batch4、每轮20,016 steps、四轮。输入为pixel+bar；共同同平面4 mm、每块最多32 nodes的learned compression处理全部全局块，没有裁掉全事件的尾部块。这一模型不等同于未经压缩的全事件ParT。

新增英文图入口在`analysis.flavour.s2.plots`展示逐seed AP、均值与样本标准差，实际查看PNG以及两项新增模型的四分类混淆、ROC、分数和学习曲线页。该绘图只读取已有评价，不再次拟合或bootstrap。

## 详细数据

### 同事件的逐次训练结果

| Seed | Best epoch by full NLL | Tau AP | Common-support tau AP | Four-class BA | Macro F1 | NLL | Tau argmax recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 4 | 0.35583088 | 0.31687645 | 0.4125 | 0.41212061 | 1.23418634 | 0.4448 |
| 42 | 4 | 0.35858391 | 0.32383619 | 0.4184 | 0.41844267 | 1.22107879 | 0.5488 |
| 2026 | 4 | 0.34930306 | 0.31647300 | 0.4097 | 0.40044304 | 1.22823669 | 0.6672 |

Raw AP均值=0.35457262、sample SD=0.00476665、极差=0.00928085；common-support均值=0.31906188、sample SD=0.00413959。Common support固定8,559 events、Neff=7,959.41365，仅控制既有粗E/vertex cells，不控制target Z/同位素。每个模型评估相同10,000个物理事件，不能把三次预测视为30,000个独立事件。没有做ensemble，也没有挑选最佳seed代表三次均值。

标准差只描述这三次已训练模型的AP波动，不是事件CI，也不是原生−1 mm或ParT−ParticleNet差值的三seed区间。当前四组预定三seed均值比较均未完成，不能用本组unpaired spread替代paired差值。

| Seed | AP: CC nu e | AP: CC nu mu | AP: CC nu tau | AP: NC |
| --- | ---: | ---: | ---: | ---: |
| 17 | 0.60049952 | 0.43840408 | 0.35583088 | 0.41405143 |
| 42 | 0.60478117 | 0.46065084 | 0.35858391 | 0.42513003 |
| 2026 | 0.60404125 | 0.45353721 | 0.34930306 | 0.41314115 |

三次argmax τ recall从44.48%到66.72%，而AP极差不到0.01。模型对τ的排序和四类决策边界不是同一指标；尚未校准的分数尺度/类别偏置可能参与这一差异，成因未测量。诊断页显示seed2026将更多νμ和NC判为τ，不能仅由τ recall较高判断其更好。

### 工作点、训练曝光和资源

| Seed | Tau selected at 1% FPR | Background selected | Threshold | Tau selected at 0.1% FPR | Background selected |
| --- | ---: | ---: | ---: | ---: | ---: |
| 17 | 82 / 2,500 | 75 / 7,500 | 0.44347152 | 10 | 7 |
| 42 | 93 / 2,500 | 75 / 7,500 | 0.43040636 | 17 | 7 |
| 2026 | 87 / 2,500 | 75 / 7,500 | 0.49984741 | 9 | 7 |

这些是各自validation经验曲线工作点；0.1%背景接受率的尾部统计不足，不能形成稳定优劣结论。Calibration和test保持冻结至S6。

| Seed | Exposures by epoch | Total exposures | Allocation [s] | Allocation [GPUh] | Program wall [s] | Peak allocated [MiB] |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 17 | 80040 / 80036 / 80036 / 80040 | 320152 | 6075 | 1.68750000 | 6066.62958 | 1583.52490 |
| 42 | 80040 / 80044 / 80040 / 80040 | 320164 | 6993 | 1.94250000 | 6903.39536 | 1589.18994 |
| 2026 | 80044 / 80048 / 80040 / 80040 | 320172 | 6082 | 1.68944444 | 6074.30122 | 1584.63574 |

每项80,064 applied updates、0 AMP skips；exposures是实际处理的事件次数，unique train仍各80,000。Seed42含早期硬件首轮门槛执行，不把wall差异归因于seed本身；成本全部计入组织预算。10:15:56 UTC评价前核算：组织81.76424820 GPUh，H100视图45.52591487 GPUh，missing history records=0。

正式产物根目录：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11/h100/`。`part_native_seed_repeatability.json`保存逐模型metrics/manifest/history/readback、spread和真实terminal；`evaluation_final/training_seed_spread.png/.pdf`及`comparison_figures.json`保存英文图与数值；两个`part_native_s*_diagnostic_review.png`为实际查看页。终态为`part_native_s17_terminal.json`、`part_native_s2026_terminal.json`，读回日志为`readback_part_native_s17.log`和`readback_part_native_s2026.log`。

本批评价commit=`be4b319`，accounting/evaluate/target/inventory顺序进程真实exit0：30 results=4 references+26科学配置、808 strata、90历史三类表、30 pairs、15 pending、2 factorial、0 seed means。Target诊断210 model×Z组。Inventory为26 output_complete、3 running、6 not_started，strict gate=false。后续训练终态与这一固定评价快照分开记录。

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
S2_RESULTS=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11
python -m analysis.flavour.s2.plots --evaluation "$S2_RESULTS/h100/evaluation_final"
```

新增绘图已实际运行，均值/SD与summary一致，PNG和两模型分类页已查看；语法与`git diff --check`通过。53项既有回归在阶段44后的日志` s2_regression_suite_055a8f0.log`中通过；本次绘图未改核心计算，未重跑完整套件。

## 阶段性结论

事实：首个固定配置三seed完成，AP样本SD约0.00477；argmax决策和经验阈值存在更明显变化。尚未测量原生/粗表示与两架构的完整三seed差值，不能宣称三seed科学比较已确认。

## 下一步计划

完成center训练域、额外coarse/ParticleNet seeds和plane-only控制，保持相同物理事件的配对评价。35项终态与读回、41单seed配对和4三seed均值齐备后，运行不带`--allow-incomplete`的最终评价与`--require-complete`输出门槛，再形成S2 gate decision。S3–S6不开始。
