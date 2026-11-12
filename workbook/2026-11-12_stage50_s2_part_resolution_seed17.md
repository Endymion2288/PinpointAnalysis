# 阶段50：ParT 分辨率比较的 seed17 配对

ParT seed17 的 native−1 mm 全域τ AP差为+0.00440110，pointwise 95%事件区间为[−0.00207130,+0.01145529]；固定粗E/vertex支持后差为−0.00138996，区间[−0.00768450,+0.00504632]。本seed既未确认正收益，也不能排除预定+0.01的raw AP收益。Seed42的结果方向相近，但不以两seed平均替代预定三seed确认。

## 阶段目标

读回完成的`part_1mm_joint_s17_e4_v2`，在相同validation物理事件上与native seed17配对，量化AP、工作点、四分类和真实成本。三seed均值继续等待seed2026，calibration/test保持冻结。

## 工作内容

`bigbird20.cern.ch`的`1573782.25`真实JobStatus=4、ExitCode=0、NumJobStarts=1，节点`slot1_2@b9pgpun501.cern.ch`，四轮完成、14项CPU读回全部pass。使用冻结80k train/10k validation、各类20k/2,500、seed17、batch4、每轮20,016 attempted steps；80,064 applied updates、0 AMP skips。H100 NVL、LCG108、Torch2.7.1、Python3.12.11、BF16，按full-validation NLL选epoch4。

输入为pixel+bar，ParT dim48、151,228参数；native与1 mm共同使用同平面4 mm/每块最多32 nodes的learned compression并处理全部全局块。1 mm使用当前表示的count及signal summaries，不回填native细节；本对照评价的是当前压缩模型的表示收益。

## 详细数据

### 同seed主指标和工作点

| Metric | 1 mm | Native |
| --- | ---: | ---: |
| Tau AP | 0.35142978 | 0.35583088 |
| Fixed-support tau AP | 0.31826641 | 0.31687645 |
| Tau ROC AUC | 0.63069056 | 0.63461651 |
| Balanced accuracy | 0.4237 | 0.4125 |
| Macro F1 | 0.42196263 | 0.41212061 |
| NLL | 1.22338293 | 1.23418634 |
| Brier | 0.66280388 | 0.66898133 |
| Argmax tau recall | 0.4700 | 0.4448 |
| Tau selected at 1% FPR | 74 / 2,500 | 82 / 2,500 |
| Background selected | 75 / 7,500 | 75 / 7,500 |
| Validation threshold | 0.44041902 | 0.44347152 |
| Tau selected at 0.1% FPR | 8 | 10 |
| Background selected at 0.1% FPR | 7 | 7 |

1 mm四类AP依次为0.60413524/0.46579126/0.35142978/0.43085711；native为0.60049952/0.43840408/0.35583088/0.41405143。Native的τ排序略好，1 mm的综合四分类点估计较好；非τ及综合指标差值CI未测量。

400次配对bootstrap，128个accepted-family/chunk blocks、每类32。固定支持包含8,559 events，Neff=7,959.41365，仅匹配既有粗E/vertex cells，未匹配target Z或同位素。事件区间条件于当前拟合模型及固定权重，与训练seed spread分开。

| Native minus 1 mm | Difference [95% CI] |
| --- | --- |
| Raw tau AP | +0.00440110 [−0.00207130,+0.01145529] |
| Fixed-support tau AP | −0.00138996 [−0.00768450,+0.00504632] |
| Tau ROC AUC | +0.00392595 [−0.00145142,+0.00917055] |
| Tau efficiency at 1% FPR | +0.00320000 [−0.00566270,+0.01129947] |
| Tau efficiency at 0.1% FPR | +0.00080000 [−0.00282462,+0.00601083] |

所有这些区间跨0；0.1%工作点只有7背景，尾部统计不足。Inclusive 1%阈值下，1 mm的τ→e/μ/had通过8/3/63个，native为10/4/68个，分母分别445/395/1,660。分支差值CI未测量。

条件区域点估计：中心885 events/176τ，1 mm/native AP=0.36207764/0.33472758；large 4,494 events/1,053τ为0.38140820/0.38330180。当前未计算这两项分辨率条件配对CI，不由点估计宣称区域优劣，也不把跨区域AP绝对值直接比较。

### 训练历史与成本

1 mm四轮实际exposures为80,040/80,036/80,036/80,040，总320,152，与native seed17相同。Unique train各80,000，没有把重复曝光计为新增物理事件。

| Resource | 1 mm | Native |
| --- | ---: | ---: |
| Backend allocation [s] | 8,536 | 6,075 |
| Allocation [GPUh] | 2.37111111 | 1.68750000 |
| Program wall [s] | 8,517.75985 | 6,066.62958 |
| Allocation outside program [s] | 18.24015 | 8.37042 |
| Peak allocated memory [MiB] | 1,541.58447 | 1,583.52490 |

1 mm allocation高40.51%，显存峰值低2.65%。这两次训练位于不同节点和运行时段，不能将耗时差归因于粗表示本身。共同allocation预算6,074.21591 s下，按各自可完成整轮的最低NLL选点，1 mm可用epoch2、AP=0.31745051，native为epoch4、AP=0.35583088；差值+0.03838038仅为整轮分辨率的时间比较点估计，没有CI，亦不隔离硬件竞争、I/O和表示原因。所有四轮成本均入账。

11:15:10 UTC组织成本85.50462006 GPUh，missing history records=0。当前主评价commit=`5e93412`，32 results=4 references+28科学配置、862 strata、96历史三类表、32 paired/13 pending、2 factorial、0 seed means。

正式产物位于`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11/h100/`。`part_1mm_s17_terminal.json`保留真实终态，`readback_part_1mm_s17.log`保留读回；`part_resolution_seed17/paired_comparisons_first_page.png`及同目录`paired_comparisons.pdf`实际展示seed42和17的raw/common区间及共同allocation点估计；`evaluation_final/part_1mm_s17_diagnostic_review.png`为实际查看的第5页分类诊断。两张PNG已查看，图中区间和主表一致。没有改变核心计算或训练合同。

Accounting/evaluate/target/inventory顺序进程真实exit0。Target刷新至224 model×Z组，输出清单28 complete、4 running、3 not_started，strict gate=false。`part_resolution_seed17_comparison.json`保存主表、历史、配对、分域、读回和成本；`stage50_execution_snapshot.json`记录11:26 UTC实际队列、Kubeflow flavor和上述评价计数。Kubeflow v9仍Pending、无node。日志为`accounting_part_1mm_s17.log`、`evaluation_part_1mm_s17.log`、`target_part_1mm_s17.log`及`inventory_part_1mm_s17.log`。本阶段复用计算入口，验证结果与主summary一致，未重跑已通过的完整回归套件。

```bash
python -m analysis.flavour.s2.accounting --root "$S2_RESULTS"
python -m analysis.flavour.s2.evaluate --root "$S2_RESULTS/h100" \
  --output "$S2_RESULTS/h100/evaluation_final" \
  --comparisons "$PWD/roadmap/configs/s2_complete_pairs_v2.json" --allow-incomplete
python -m analysis.flavour.s2.target_diagnostics --root "$S2_RESULTS/h100" \
  --output "$S2_RESULTS/h100/target_diagnostics"
python -m analysis.flavour.s2.output_inventory --root "$S2_RESULTS/h100" \
  --campaign "$PWD/roadmap/configs/s2_h100_v2.json"
```

## 阶段性结论

事实：新增同seed配对在raw和固定支持上都未确认native的正收益，raw区间仍包含+0.01。四分类综合指标与τ排序的点估计方向不同，不能用一个指标替代全部分类性能。当前还没有完整三seed分辨率结论。

## 下一步计划

完成既有seed2026、ParticleNet确认和plane-only作业。35项科学输出、41单seed配对、4三seed均值齐备后运行无allow-incomplete的严格评价、target刷新和require-complete输出门槛，再给出最终S2科学报告及gate decision。
