# 阶段52：原生输入下两架构的三seed比较

ParT−ParticleNet的native三seed平均τ AP差为+0.00219679，pointwise 95%事件区间=[−0.00401226,+0.00795043]；固定E/vertex支持后的平均差为−0.00244385，区间=[−0.00832807,+0.00308100]。在当前三个已拟合模型的条件下，上界排除预定+0.01平均收益；不能由此宣布两架构一般等效。四分类的描述统计显示ParT在三seed平均BA、NLL和各类AP上更好，相关综合指标差值CI未测量。

## 阶段目标

读回ParticleNet native新增seed17/2026，完成第二组预定三seed均值，保留训练波动、各类和τ工作点；记录Kubeflow在资源排队结束后的真实终态。

## 工作内容

`pn_native_joint_s17_e4_v2`与`pn_native_joint_s2026_e4_v2`对应`1573782.27/28`，真实JobStatus=4、ExitCode=0、各NumJobStarts=1，分别在`slot1_2@b9pgpun003.cern.ch`、`slot1_3@b9pgpun001.cern.ch`结束。两者均四轮完成，14项CPU读回pass，按full-validation NLL选epoch4；各80,064 applied updates、0 AMP skips。

相同80k train/10k validation、每类20k/2,500、pixel+bar、seed17/42/2026、batch4、每轮20,016 attempted steps，H100 NVL/LCG108/Torch2.7.1/Python3.12.11/BF16。共用同平面4 mm/每块最多32 nodes的learned compression并处理全部全局块。ParT dim48/151,228参数，ParticleNet dim64/161,156参数；参数相差约6.6%，这不是参数严格相等的单机制消融。

## 详细数据

### 主配对与训练波动

| Seed | ParticleNet tau AP | ParT tau AP | ParT−PN raw [95% CI] | Fixed-support difference [95% CI] |
| --- | ---: | ---: | --- | --- |
| 17 | 0.34921291 | 0.35583088 | +0.00661797 [−0.00158945,+0.01451046] | −0.00285973 [−0.01071550,+0.00419042] |
| 42 | 0.35510474 | 0.35858391 | +0.00347918 [−0.00368918,+0.01035692] | +0.00180789 [−0.00453733,+0.00875252] |
| 2026 | 0.35280983 | 0.34930306 | −0.00350677 [−0.01113083,+0.00414656] | −0.00627972 [−0.01437286,+0.00189351] |
| Mean of paired differences | — | — | +0.00219679 [−0.00401226,+0.00795043] | −0.00244385 [−0.00832807,+0.00308100] |

400次共享物理事件块bootstrap，128 accepted-family/chunk blocks、每类32；对同一物理事件重采样后取三seed AP差值平均。10,000个物理事件没有扩成30,000个独立事件；这不是ensemble，也没有重选seed。区间条件于当前已拟合seed，未量化未来随机训练总体。Raw差值sample SD=0.00518276，common差值sample SD=0.00405981，另列为训练spread。

ParticleNet native raw AP均值=0.35237583、sample SD=0.00296979，common均值=0.32150573、SD=0.00157471；ParT对应0.35457262/0.00476665、0.31906188/0.00413959。固定粗E/vertex支持为8,559 events、Neff=7,959.41365，未匹配target Z或同位素。Raw/common平均差值符号不同，区间均跨0；不把common支持诊断视为去除全部物理先验。

### 四分类与工作点

| ParticleNet seed | BA | Macro F1 | NLL | Tau argmax recall | Tau selected at 1% FPR |
| --- | ---: | ---: | ---: | ---: | ---: |
| 17 | 0.3915 | 0.38698453 | 1.24774134 | 0.4540 | 62 / 2,500 |
| 42 | 0.3974 | 0.39204149 | 1.23930563 | 0.5664 | 83 / 2,500 |
| 2026 | 0.3955 | 0.37965804 | 1.24395910 | 0.4700 | 81 / 2,500 |

1%工作点均为75/7,500背景，threshold依seed为0.45215777/0.45402154/0.45583045；ParT分别通过82/93/87个τ。新增17/2026的效率差值为+0.00800/[−0.00681172,+0.01802223]及+0.00240/[−0.00877140,+0.01021993]，都跨0；效率三seed平均CI未测量。0.1%每项背景仅7个，尾部统计不足。所有阈值来自validation经验曲线，calibration/test冻结。

| Three-seed arithmetic mean | ParticleNet native | ParT native |
| --- | ---: | ---: |
| AP: CC nu e | 0.59654370 | 0.60310732 |
| AP: CC nu mu | 0.41101658 | 0.45086405 |
| AP: CC nu tau | 0.35237583 | 0.35457262 |
| AP: NC | 0.36720646 | 0.41744087 |
| Balanced accuracy | 0.39480000 | 0.41353333 |
| NLL | 1.24366869 | 1.22783394 |

这些均值描述当前拟合；非τ和综合指标均值差CI未测量。新增PN seed2026的νμ argmax recall只有425/2,500=17%，但其νμ AP为0.41389131，说明argmax类别偏置和排序质量须分别检查。分支、能量、vertex及空间域条件表保存在主summary与独立快照；不按单张混淆图推断物理原因。

### 曝光与实际成本

| PN seed | Epoch exposures | Allocation [s] | Program wall [s] | Peak allocated [MiB] |
| --- | --- | ---: | ---: | ---: |
| 17 | 80040 / 80036 / 80036 / 80040 | 4766 | 4758.19138 | 1592.49902 |
| 42 | 80040 / 80044 / 80040 / 80040 | 6310 | 6293.25696 | 1593.10254 |
| 2026 | 80044 / 80048 / 80040 / 80040 | 4721 | 4711.66460 | 1587.94580 |

曝光总数分别320,152/320,164/320,172，unique train仍各80,000。PN三次总allocation=15,797 s=4.38805556 GPUh；ParT native为19,150 s=5.31944444 GPUh。节点、时段和早期门槛经历不同，不把全部耗时差归因于架构。

共同allocation预算下，seed17 ParT可用epoch3、PN epoch4，ΔAP=+0.00450540；seed2026同样3/4，ΔAP=−0.00925409。均为按各自预算内NLL选点的整轮分辨率点估计，没有CI。主三seed比较保持四轮步数合同。

12:11:00 UTC组织89.75616158 GPUh，missing history records=0。Kubeflow v9在所有剩余科学任务获得Condor GPU后，于12:03 UTC主动缩短deadline结束，始终无node；不是自然六小时排队到期。终态和scheduler事件核算0 GPUh，GPU runtime未测量。H10094GB flavor已实际通过Pod准入，task33/34备用配置未提交；task34 dry-run因当时probe占用单GPU请求配额而被拒绝，没有创建新Pod。

### 验证与产物

正式根目录`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11/h100/`。`native_architecture_three_seed_comparison.json`保存六模型metrics/manifest/history/readback、三项单seed与均值配对、条件表、终态和成本；`stage52_execution_snapshot.json`记录12:21 UTC实际3个科学任务running。终态为`pn_native_s17_terminal.json`和`pn_native_s2026_terminal.json`，读回日志同名`readback_pn_native_s*.log`。

`native_architecture_three_seeds/`中的逐seed和均值配对PNG、更新的`evaluation_final/training_seed_spread.png`、两张`pn_native_s*_diagnostic_review.png`均已实际查看。诊断页为主PDF的27/28页，图与主表一致。没有新增模型或改变核心评价入口，未重跑已通过的完整回归套件。

本批accounting/evaluate/target/inventory顺序进程真实exit0，评价commit=`5023d64`。35 results=4 references+31科学配置、943 strata、105历史三类表、35单seed+2均值、8 pending、2 factorial。Inventory有32个training complete、3 running，但只有31项output_complete：PN 1 mm seed17在本批评价扫描后结束，已另通过14项读回，等待下批主评价及终态核算。Target扫描已纳入该模型，故为36个结果×7 Z=252组，不冒充与本批35-result主表完全同步。日志后缀为`native_architecture_seeds.log`，严格gate=false。

```bash
python -m analysis.flavour.s2.readback --root "$S2_RESULTS/h100" \
  --campaign "$PWD/roadmap/configs/s2_h100_v2.json" \
  --experiments pn_native_joint_s17_e4_v2 pn_native_joint_s2026_e4_v2
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

事实：第二组预定三seed均值完成，当前native两架构的τ AP平均差小，区间跨0；固定已训练模型的事件区间排除+0.01平均收益。ParT的四分类描述统计更好而成本更高，不能仅用τ单指标判断所有反应类别。

解释范围：比较两套共同表示下的具体实现，不隔离单一attention/graph机制；训练spread不等同于事件CI。当前还未完成粗表示架构均值、PN分辨率均值和跨层控制，S2尚未验收。

## 下一步计划

将已完成PN 1 mm seed17与即将结束seed2026合并评价，完成余下两组均值。等待既有两个plane-only四轮终态；35科学输出、41单seed+4均值全部齐备后执行严格无allow-incomplete评价、最终target与require-complete门槛，完成中文科学总报告和gate decision。
