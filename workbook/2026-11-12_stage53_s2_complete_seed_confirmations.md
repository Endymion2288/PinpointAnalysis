# 阶段53：四组预定三seed比较全部完成

PN native−1 mm三seed平均τ AP差为−0.00013880，pointwise 95%事件区间=[−0.00454523,+0.00421196]；1 mm下ParT−PN平均差为+0.00126795，区间=[−0.00485443,+0.00824882]。至此四组预定均值全部完成，raw/common区间均跨0、上界均低于预定+0.01；这一判断条件于当前三次拟合及固定权重，不是训练seed总体区间或一般信息等效结论。

## 阶段目标

补齐PN 1 mm seed17/2026的主评价，完成PN分辨率均值和粗表示架构均值；统一整理四分类描述统计、训练spread、工作点及成本。跨层控制仍待四轮终态，S2未整体验收。

## 工作内容

新增`pn_1mm_joint_s17_e4_v2`与`pn_1mm_joint_s2026_e4_v2`对应`1573782.29/30`，均真实JobStatus=4、ExitCode=0、NumJobStarts=1。分别在`slot1_1@b9pgpun300.cern.ch`和`slot1_4@b9pgpun003.cern.ch`完成四轮；各14项CPU读回pass，full-validation NLL均选epoch4，80,064 applied updates、0 AMP skips。

沿用80k train/10k validation、每类20k/2,500、pixel+bar、batch4、每轮20,016 attempted steps，H100 NVL/LCG108/Torch2.7.1/Python3.12.11/BF16。共同同平面4 mm/每块最多32 nodes的learned compression处理全部全局块；PN dim64/161,156参数，ParT dim48/151,228参数。12个三seed模型全部通过14项读回，未重新选seed、改预算或换checkpoint准则。

## 详细数据

### 完整四组均值

| Prespecified B−A contrast | Raw mean tau AP difference [95% event CI] | Fixed E/vertex support difference [95% CI] | Raw training difference sample SD |
| --- | --- | --- | ---: |
| ParT native−1 mm | +0.00079004 [−0.00282494,+0.00445758] | −0.00201089 [−0.00526091,+0.00111996] | 0.00553825 |
| PN native−1 mm | −0.00013880 [−0.00454523,+0.00421196] | −0.00134447 [−0.00522708,+0.00286726] | 0.00519741 |
| ParT−PN, native | +0.00219679 [−0.00401226,+0.00795043] | −0.00244385 [−0.00832807,+0.00308100] | 0.00518276 |
| ParT−PN, 1 mm | +0.00126795 [−0.00485443,+0.00824882] | −0.00177744 [−0.00654021,+0.00456009] | 0.00480552 |

每次对同一10,000物理事件的128个accepted-family/chunk blocks（每类32）共享重采样，计算三seed AP差值后平均，400次bootstrap；并非ensemble或30,000独立事件。训练差值SD另列，不用事件区间替代未来seed不确定度。固定粗E/vertex支持为8,559 events、Neff=7,959.41365，未匹配target Z/同位素，未使用flux或exposure。

### 新增比较的逐seed结果

| Seed | PN 1 mm AP | PN native−1 mm [95% CI] | ParT−PN at 1 mm [95% CI] |
| --- | ---: | --- | --- |
| 17 | 0.35528849 | −0.00607559 [−0.01271788,+0.00147840] | −0.00385872 [−0.01341433,+0.00560998] |
| 42 | 0.35303603 | +0.00206870 [−0.00485307,+0.00996035] | +0.00199254 [−0.00584165,+0.01104786] |
| 2026 | 0.34921934 | +0.00359049 [−0.00200623,+0.00854150] | +0.00567004 [−0.00328837,+0.01440331] |

逐seed方向不一致，均值结果不能表示每个seed同向。PN 1 mm raw AP均值=0.35251462、sample SD=0.00306799；固定支持均值=0.32285020、SD=0.00042297。三个seed不足以判断表示或架构的seed方差总体大小。

### 四分类描述统计和工作点

| Three-seed mean | ParT native | ParT 1 mm | PN native | PN 1 mm |
| --- | ---: | ---: | ---: | ---: |
| AP: CC nu e | 0.60310732 | 0.60414105 | 0.59654370 | 0.59737415 |
| AP: CC nu mu | 0.45086405 | 0.45033550 | 0.41101658 | 0.41503406 |
| AP: CC nu tau | 0.35457262 | 0.35378258 | 0.35237583 | 0.35251462 |
| AP: NC | 0.41744087 | 0.41797796 | 0.36720646 | 0.37771866 |
| Balanced accuracy | 0.41353333 | 0.41656667 | 0.39480000 | 0.40000000 |
| BA training sample SD | 0.00444110 | 0.01150014 | 0.00301164 | 0.00347707 |

`three_seed_four_class_summary.json`从各自full-NLL所选history读取这12项结果，保存每类AP/argmax recall、BA/macroF1/NLL/Brier的逐seed值、均值和sample SD。综合和非τ差值CI未测量，不把这些描述均值当作已确认配对发现。

| PN 1 mm seed | BA | Macro F1 | NLL | Tau argmax recall | Tau at inclusive 1% FPR | Background |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 0.4040 | 0.40061396 | 1.23669653 | 0.4204 | 72 / 2,500 | 75 / 7,500 |
| 42 | 0.3977 | 0.39400330 | 1.24970777 | 0.5268 | 89 / 2,500 | 75 / 7,500 |
| 2026 | 0.3983 | 0.38745682 | 1.24047940 | 0.5512 | 69 / 2,500 | 75 / 7,500 |

17/2026阈值=0.43802711/0.49218926；对应PN native分别62/81个τ，native−1 mm效率差=−0.00400、CI=[−0.01340974,+0.00830403]和+0.00480、CI=[−0.00659706,+0.01440896]。1 mm下ParT−PN效率差为+0.00080/[−0.00781065,+0.00897176]及+0.00520/[−0.00632075,+0.01389556]。这些新增工作点区间均跨0，工作点三seed平均CI未测量。

新增PN 1 mm17/2026在0.1%均选14τ/7背景，尾部统计不足。完整τ→e/μ/had、中心/large、能量/vertex及空/active条件表保存在主summary。各分支有自身分母/prior，不能把分支AP直接当作inclusive AP。Calibration/test保持冻结。

### 曝光与实际成本

| PN 1 mm seed | Epoch exposures | Allocation [s] | Program wall [s] | Peak allocated [MiB] |
| --- | --- | ---: | ---: | ---: |
| 17 | 80040 / 80036 / 80036 / 80040 | 5348 | 5335.00685 | 1552.54199 |
| 42 | 80040 / 80044 / 80040 / 80040 | 5171 | 5145.06316 | 1552.50732 |
| 2026 | 80044 / 80048 / 80040 / 80040 | 4560 | 4552.55094 | 1550.78711 |

总exposures为320,152/320,164/320,172，与对应native seed一致。PN 1 mm总allocation=15,079 s=4.18861111 GPUh，PN native=15,797 s=4.38805556 GPUh；ParT 1 mm为21,716 s=6.03222222 GPUh。节点、运行时段和早期门槛不同，成本差不隔离架构或表示原因。

新增PN分辨率共同allocation点估计：17的1 mm/native可选epoch3/4、ΔAP=−0.00045045；2026为epoch4/3、ΔAP=−0.00160274。粗表示架构比较17的PN/ParT可选epoch4/2、ParT−PN ΔAP=−0.03783799；2026为epoch4/3、ΔAP=−0.00100335。均只在整轮分辨率按预算内最低NLL选点，没有CI，不替换主四轮步数匹配结论。

12:52:40 UTC组织累计91.86595936 GPUh，missing history records=0。Kubeflow v9已主动结束、实际0 GPUh；12:42 UTC释放配额后同一备用训练Pod的server dry-run通过，自动注入46 CPU/127G RAM/1整卡H100，仍未提交真实训练，GPU runtime未测量。证据与命令规则更新于`roadmap/08_kubeflow.md`。

### 验证与产物

结果根目录`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11/h100/`。`complete_seed_confirmation_comparison.json`保存全部12模型metrics/manifest/history/readback、相关单seed配对、全部4均值、四分类描述统计、终态及成本；`stage53_execution_snapshot.json`记录13:05 UTC两项plane-only实际running。新增终态/读回为`pn_1mm_s*_terminal.json`和`readback_pn_1mm_s*.log`。

`complete_seed_confirmations/`中的逐seed配对、四组均值、四个配置training spread三张PNG，以及两张`evaluation_final/pn_1mm_s*_diagnostic_review.png`均已实际查看；后者为分类PDF的19/20页。图中事件区间和训练SD明确区分，数值与主表一致。

Accounting/evaluate/target/inventory顺序进程真实exit0，评价commit=`4aa30ef`。37 results=4 references+33科学配置、997 strata、111历史三类表、39单seed配对、4均值、2 pending、2 factorial；target同步259 model×Z组。Inventory确认33 output_complete、2 running、strict gate=false。日志统一后缀`complete_seed_confirmations.log`。本阶段复用已验证的核心计算，新增描述统计仅读取已有history，未重跑完整回归套件。

```bash
python -m analysis.flavour.s2.readback --root "$S2_RESULTS/h100" \
  --campaign "$PWD/roadmap/configs/s2_h100_v2.json" \
  --experiments pn_1mm_joint_s17_e4_v2 pn_1mm_joint_s2026_e4_v2
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

事实：四组预定三seed比较全部完成；当前共享压缩、四轮预算下，τ AP平均差较小，raw/common事件区间均跨0并排除该已拟合条件下+0.01收益。四分类描述统计显示架构间的νμ/NC差异比τ更明显，成本和指标须一起看。

解释范围：这些结果约束当前模型，不确定原始硬件细节对更强编码器/辅助任务的可用上限；不把未确认收益改写为信息无用。训练spread、物理事件CI、条件支持和真实硬件分辨能力分别表述。

## 下一步计划

完成两项plane-only四轮终态和14项读回。所有35项齐备后直接运行无allow-incomplete的严格评价、最终target和require-complete门槛；补齐最后两项配对、成本和科学总报告，再作S2 gate decision。
