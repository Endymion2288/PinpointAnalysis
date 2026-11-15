# R3：全部晋级指标的独立区间复核

## 阶段目标

τ晋级条件直接使用1%背景工作点的配对区间；原独立core-interval实现只覆盖BA3/NLL3/三类recall/BA4/NLL4七项。补齐τ的1%和0.1%工作点区间数值验证，保持事件、draws、评价矩阵和科学门槛。

## 工作内容

在`r1.interval_review`独立实现按原始类别的完整score-group直方图。每个draw分别累计原始physical-block multiplicity，除以该类support；τ效率使用τ的实际加权分母，FPR使用有support的非τ类别接受率均值。整个ties同时进入或退出，没有插值。仅分批物化16个draws，不调用主评价的Metrics、scalar_summary或排序缓存。

独立复核由七项扩为九项；已有配对点值、percentile区间和实际三seed均值/SD路径一起覆盖。`r3.scientific_review`要求这些复核包含全部晋级指标，且pairs、replicates与seed平均范围匹配当前完整评价。最终四套矩阵要求和所有质量门槛不变。

只对R3已完成的M0/C1补做CPU独立复核。G1的历史单独评价保留形成时结果，最终完整geometry矩阵会自动包含九项新复核；未来补充no-global对强B1的评价也使用新实现。旧R1/R2历史报告不因此重写为九项审查。

## 详细数据

根目录：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R3`。

旧M0/C1独立core报告均为116项通过，原始七指标报告已在覆盖前分别保存至`audit/evaluation_{m0,c1}_independent_core_before_tau.json`。模型权重、预测和主评价结果保持原始内容；本次只增加计算复核。

测试手工构造四类各1000事件：三种非τ背景有相同的.1/.8/.9分数组，τ有.95/.85/.75/.05组；精确预期1%/0.1%效率为35%/15%。各类整体重新乘2/3/5/7后相同，验证类别先验保持；将一个.8背景事件权重翻倍，使整组超过1%配额后，效率变为15%/15%，验证不拆ties。前50个τ加倍后预期400/1050、200/1050；无τ或无背景权重时均返回未定义。

46项相关tests通过，用时6.123s；随后新增的“缺τ指标或只有20次draw的独立报告不能通过最终scientific review”单项测试通过，用时0.002s。合计47项不同tests已验证；当前训练代码没有变化。

实际提交前两次均检查schedd队列、WRITE和EOS策略，选择bigbird20；每项2CPU/6GB：

| Review | Condor job | Submission UTC | Live state at 18:35 UTC |
| --- | --- | --- | --- |
| M0 full400-draw independent review | 1583022.0 | 18:32:52 | Running, slot1_25@b9p14p9047.cern.ch |
| C1 full400-draw independent review | 1583023.0 | 18:32:54 | Running, slot1_1@b9g23p2657.cern.ch |

执行形式：

```bash
python -m analysis.flavour.v2.submit --root "$R3_ROOT" --stage R3 \
  --name r3_m0_tau_interval_c1 --cpus 2 --memory 6GB --job-flavour workday \
  analysis.flavour.v2.r1.interval_review --root "$R3_ROOT" \
  --config "$R3_ROOT/evaluation_m0/config.json" \
  --evaluation "$R3_ROOT/evaluation_m0/results.json" \
  --output "$R3_ROOT/evaluation_m0/independent_core_intervals.json"
```

C1使用对应`evaluation_c1`路径和独立attempt。两任务均已终态JobStatus4/Exit0，完整allocation分别46/47s，实际RSS269/293MB；2CPU合计0.051667 allocated core-hours，没有新增GPU allocation。没有在工作节点执行40k×400的大样本统计。

完整九项复核实际结果：

| Evaluation | Model/domain combinations | Paired contrasts | Checks passed | Added tau checks | Maximum tau point error | Maximum tau interval error | Review seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M0 | 16 | 8 | 148 | 32 | 1.80411e−16 | 3.55185e−16 | 35.128538 |
| C1 | 16 | 8 | 148 | 32 | 1.94289e−16 | 1.85182e−16 | 34.277116 |

两项原116条检查记录逐字段完全相同，新增64条τ点值/区间检查全部通过。汇总见`audit/independent_tau_intervals_stage302.json`。两个已有评价再次应用未改变的完整晋级规则，四项all-domain endpoint/best对比全部完成，primary eligible列表为空，见`audit/scientific_review_m0_c1_tau_c1.json`。该文件只覆盖M0/C1，不代替最终四套评价的scientific review。

## 阶段性结论

事实：原独立区间验证缺少τ两项指标；新增算术实现和manual ties/先验/权重测试已通过，真实M0/C1批复核均完成，新增τ检查与主评价在浮点精度内一致。冻结指标及晋级值没有变化，M0/C1不晋级的决定保持。

待测：三个G2正式终点、最终native probes/health及完整四矩阵科学决定。没有新的模型性能结论，R3未完成。

## 下一步计划

等待三个G2完成，并要求完整四矩阵在全部晋级指标上独立复核，再执行最终科学/条件决定、英文图、终态账本与accept。自动观察者的CPU提交会截取当前源码，geometry和补充强基线比较因此使用九项复核；训练中的GPU代码与其冻结预算保持。
