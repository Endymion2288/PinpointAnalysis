# S4 三个独立MAE seeds的低标签结果

## 阶段目标

确认MAE3在1k标签下的独立seed收益，同时检查τ排序、硬分类、低背景工作点与概率质量是否一致改善。

## 工作内容

v7 CPU作业1578086（bigbird20，源码9b2b78c）完成，程序319.608828秒。19个训练模型各14项checkpoint检查通过；2,193次训练健康采样均通过，SiLU RMS范围0.088457–0.835281，精确全零比例最大0。最小informative input梯度平方范数为7.9403×10⁻¹²；该极值包含继续训练后高度拟合的低标签任务，不以非零本身宣称良好泛化。

评价覆盖48组预测、28个单配对和1个完整三seed配对，仍有101个预定项目未齐。所有结果来自相同10k validation，每类2,500。使用400次、128个accepted family/chunk物理块的配对bootstrap；区间条件于已拟合seeds与validation选择，training seed sample SD另列。Calibration/test未使用。

精简数据为`roadmap/results/s4_progress_validation_v7.json`，完整统计、各条件域与英文图表位于输出根`evaluation/progress_v7/`。新增`s4.summarize`直接从完整统计导出数值，不重新选择或重算区间；统一导出完整四类实际工作点计数，合并重复scope文字。已人工查看训练标签效率图。后续绘图显式把横轴限制于实际标签预算，减少log轴的空白。

## 详细数据

### 1k标签，独立预训练与下游seeds17/42/2026

| Model | Macro AP mean ± sample SD | τ AP mean ± sample SD | NLL mean ± sample SD |
| --- | ---: | ---: | ---: |
| Scratch | 0.290584 ± 0.005769 | 0.261182 ± 0.002218 | 1.540932 ± 0.022012 |
| MAE3 | 0.347605 ± 0.014037 | 0.288147 ± 0.003366 | 1.450039 ± 0.054424 |

三个MAE3的macro AP增量依次为0.060637、0.064583、0.045842；τ AP增量为0.025763、0.024787、0.030345。三seed平均是指标平均，不是ensemble。

| MAE3 − Scratch | Raw delta [CI95] | Common-support delta [CI95] |
| --- | ---: | ---: |
| Macro AP | +0.057021 [0.052512, 0.061955] | +0.051424 [0.045730, 0.056903] |
| τ AP | +0.026965 [0.019238, 0.035548] | +0.011427 [0.002564, 0.020248] |
| NLL | −0.090893 [−0.099831, −0.082318] | −0.085518 [−0.097095, −0.075055] |
| Macro F1 | +0.045551 [0.039521, 0.052339] | +0.038089 [0.030321, 0.046118] |
| τ argmax F1 | −0.050129 [−0.063767, −0.036140] | −0.063305 [−0.079627, −0.046767] |
| τ efficiency at 1% FPR | +0.002933 [−0.001348, 0.007343] | +0.001459 [−0.003338, 0.005188] |

τ argmax F1的三seed增量为−0.087409、+0.057830、−0.120808，sample SD=0.094975；seed依赖明显。MAE3的均值NLL仍高于均衡常数先验ln4=1.386294。原始1k训练的最佳NLL均在第一轮，随后train loss继续下降而validation NLL变差；当前证据指向低标签过拟合和概率/决策不稳定，未观察到旧Head整层失活或上游断连。AP提升不等于τ硬分类与概率质量已经充分解决。

### τ子道与尾部计数

| τ branch AP delta | Raw [CI95] | Common-support [CI95] |
| --- | ---: | ---: |
| Electron | +0.003204 [−0.004323, 0.010568] | −0.001908 [−0.009184, 0.004803] |
| Muon | −0.000197 [−0.005770, 0.005803] | −0.003703 [−0.011610, 0.002060] |
| Hadronic | +0.031014 [0.021458, 0.040893] | +0.017168 [0.006759, 0.026996] |

在既定粗E/vertex common-support诊断中，hadronic排序收益仍被区间支持；未匹配全部kinematics、target和拓扑变量，不能解释为完全排除了这些差异。Leptonic τ收益未确认。

1% FPR的τ signal通过计数，Scratch→MAE3分别为seed17的32→43、seed42的23→29、seed2026的31→36；每模型2,500 signal与7,500 background，background均75。0.1% FPR分别为9→6、3→2、6→4，background均7；没有支持稀有工作点改善。各点保留Clopper–Pearson计数区间，其阈值来自validation，不是独立最终工作点覆盖声明。

### 新完成控制

Channel Rel100k seed42实测macro AP=0.475273、τ AP=0.357799、NLL=1.203310。MAE5同parent continuation对照尚未统一评价，当前不给Rel独立收益或晋级判断。

Matched监督100k相对原Scratch：Δmacro AP=0.000465，CI95=[−0.001422,0.002620]；ΔNLL=0.000348，[−0.001659,0.002331]。三个标签预算按冻结的原始-or-continuation NLL selection均保留原Scratch，费用仍包含完整继续训练。1k继续监督的best NLL=2.919094，不作为有效模型改进。

验收与统一评价的5项针对性测试通过；数值统计未因绘图和精简导出改变。调用命令：

```bash
python -m analysis.flavour.s4.summarize \
  --evaluation /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13/evaluation/progress_v7/summary.json \
  --health /eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s4_2026-11-13/experiments/s4_validation_progress_v7/training_health_summary.json \
  --output roadmap/results/s4_progress_validation_v7.json
```

## 阶段性结论

完整三seed证据支持MAE3在1k标签下提高总体及hadronic τ排序。τ argmax F1退化、NLL仍高于常数先验、低FPR区间跨零均须同时保留，不能宣称τ识别全面改善。旧Head故障已与本轮过拟合现象区分。

## 下一步计划

完成10k/100k独立MAE、MAE5、local/channel Rel和EMA的fine-tuning及frozen矩阵，再做strict评价。低标签优化应优先处理过多重复曝光、encoder/readout学习率与readout正则，并以train-inner选择和同配方Scratch控制验证；不在已观察的validation上静默替换冻结主矩阵。根据完整收益与成本决定是否扩1M。
