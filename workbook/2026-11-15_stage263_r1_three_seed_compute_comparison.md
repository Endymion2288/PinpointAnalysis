# R1：B1 三 seed 等时预算比较

## 阶段目标

完成B1/B4三seed的共同事件、共同曝光与同allocation等时预算比较，回答增加三类重复曝光是否稳定改善分类，以及四类训练是否明显损害共同三类任务。完整R1仍在执行，本阶段不启动R2，不访问Cal-A/Cut-B/Confirm payload。

## 工作内容

收齐B1的三个T3 matched-compute模型，在冻结的`R1/interim/b1_three_seed_compute/evaluation_config.json`上完成24模型、56配对、12项mean-seed配对。使用400次配对物理块重采样，统计seed为20261116。全部比较使用同一30k非τ Tune事件；T4先在非τ类别上条件化。T3 main与T4 main中共同非τ事件均曝光4次；compute T3使用相同75k unique非τ事件，按本allocation fresh T4的实测wall延长训练，并采用wall-based schedule。

增加`roadmap/tools/export_r1_compute_review.py`，仅在评价与独立读回均complete后导出[紧凑结果](../roadmap/v2/r1_b1_three_seed_compute.json)。增加英文四面板`r1_matched_compute`图：三类BA均值与seed SD、compute−T4的BA/NLL配对区间、B1逐类recall取舍；完整R1绘图和最终验收同时要求该图。

## 详细数据

### 三 seed 指标

每行是seeds17/42/2026的算术均值；SD为三个实际训练结果的sample SD，包含设备与精度差异。Recall顺序为e/μ/NC。

| Model | Mean BA3 (%) | BA sample SD (pp) | Mean NLL3 | Mean recall (%) e / mu / NC |
| --- | ---: | ---: | ---: | --- |
| B1 T3 main | 56.56333 | .24007 | .870377479 | 60.9400 / 51.0633 / 57.6867 |
| B1 T4 conditional | 56.30444 | .17206 | .867105285 | 58.8600 / 51.0833 / 58.9700 |
| B1 T3 matched compute | 56.36889 | .21001 | .875548876 | 58.2333 / 57.6667 / 53.2067 |
| B4 T3 main | 54.39222 | .06744 | .902022214 | 54.0467 / 53.4733 / 55.6567 |
| B4 T4 conditional | 54.48667 | .21497 | .904369367 | 55.8167 / 52.9367 / 54.7067 |
| B4 T3 matched compute | 54.67444 | .04857 | .899336940 | 56.7467 / 54.1633 / 53.1133 |

| Paired contrast | Delta BA3 (pp), 95% interval | Delta NLL3, 95% interval |
| --- | --- | --- |
| B1 compute T3 minus T4 | +.06444 [−.17861,+.30226] | +.008443592 [+.006379003,+.010497085] |
| B1 compute T3 minus main T3 | −.19444 [−.42774,+.03423] | +.005171397 [+.003805558,+.006575054] |
| B4 compute T3 minus T4 | +.18778 [−.07190,+.41289] | −.005032427 [−.006562434,−.003441161] |
| B4 compute T3 minus main T3 | +.28222 [+.08821,+.46817] | −.002685275 [−.003670893,−.001616101] |

B1 compute−T4的逐类recall差为e **−.62667 pp [−.97914,−.22460]**、μ **+6.58333 pp [6.13923,7.01115]**、NC **−5.76333 pp [−6.15654,−5.34584]**。μ1% FPR效率增加.47333 pp [.05069,.96959]，μ90% benchmark purity效率增加.73000 pp [.01652,1.49759]。三seed的BA差为+.23667/−.36667/+.32333 pp，SD为.37586 pp；μrecall差为+5.21/+14.32/+.22 pp，SD为7.14962 pp。

T4−compute T3的BA区间下界为−.30226 pp，满足原总体−.5 pp非劣界；逐类μ/NC取舍仍明显。以反方向NLL差判断时，T4更好。B1 compute−main T3则为e −2.70667、μ +6.60333、NC −4.48000 pp，不能将增加三类曝光等同于普遍改善。

以上是开发Tune上的pointwise物理块区间，条件于已经拟合的三个seed和实际runtime；没有估计训练seed总体区间，也没有使用独立Confirm。预算与学习率schedule一起改变，尚不能把差异单独归因于曝光。

### 实际预算与学习健康

| Seed | Device / allocation | Actual / target wall (s) | Relative error (%) | Training exposures |
| --- | --- | ---: | ---: | ---: |
| 17 | H100 1g MIG / 1582904.0 | 3405.297805 / 3405.089740 | .0061104 | 394684 |
| 42 | T4 / 1582883.0 | 5199.130752 / 5193.243869 | .1133566 | 391633 |
| 2026 | H100 1g MIG / 1582905.0 | 3430.896837 / 3428.810744 | .0608401 | 396968 |

三个模型均与本allocation fresh T4配对，使用相同75k unique非τ事件，各有51个health samples。`R1/audit/b1_matched_compute_stage263.json`的24项检查全部通过：activations finite、概率非恒定、image/tracker/head有正梯度与参数更新、同allocation及wall偏差小于5%。该比较控制本allocation的操作wall，不是跨设备FLOP等效。

新增compute作业1582904/1582905已Exit0，allocation分别9505/9617 s，包含各自本allocation的fresh T4与其他必要操作。统计作业1582930 Exit0，allocation226 s、实际统计213.289972 s；独立复核1582931 Exit0，allocation32 s。

04:31:42 UTC累计资源为**37.747854 allocation device-hours**、267.606282 allocated CPU core-hours，已超过原R1 18 h规划。完整终态成本尚未关闭；全部旧尝试、失败与候选profile均保留记账。当前34/41模型complete，B2两seed的三项操作及B3 compute继续运行，另有一项B2资源候选排队。

## 验证与产物

24模型独立概率、人数、曝光与工作点读回共**1,431项全部通过**。BA最大误差1.08e−14，NLL 5.11e−15，工作点1.11e−14；weighted confusion最大误差5.06e−10加权事件，coverage/purity 4.26e−14。紧凑compute导出另有32项数值独立核对通过。

```bash
python -m analysis.flavour.v2.r1.evaluate --root "$R1_ROOT/interim/b1_three_seed_compute" \
  --config "$R1_ROOT/interim/b1_three_seed_compute/evaluation_config.json" --replicates 400 --workers 8
python -m analysis.flavour.v2.r1.review --root "$R1_ROOT/interim/b1_three_seed_compute"
python roadmap/tools/export_r1_compute_review.py --root "$R1_ROOT" \
  --output roadmap/v2/r1_b1_three_seed_compute.json
python -m analysis.flavour.v2.r1.plots --root "$R1_ROOT/interim/b1_three_seed_compute" --compute-only
git diff --check
```

评价与复核通过HTCondor执行。R1及summary相关23个tests通过；导出脚本语法通过。三seed图已实际生成并目视核对，seed42 fallback也在既有中期结果上运行成功，没有重做与本改动无关的训练。

完整结果与独立复核分别位于`R1/interim/b1_three_seed_compute/evaluation/results.json`、`independent_review.json`。英文图位于`R1/interim/b1_three_seed_compute/figures/r1_matched_compute.{pdf,png}`。其中`R1`为`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R1`。

## 阶段性结论

**事实：**B1在共同曝光下T3的BA3比T4高.259 pp；同allocation等时预算下该差缩小至.064 pp且区间跨零，NLL更差。B4从增加重复曝光获得小幅开发收益，B1没有表现出相同趋势。四类总体非劣及τ相对增益支持保留四类研究主线，同时完成真实三类交付；当前τ绝对效率和稀有prior情景仍不足以支持高纯物理分析。

**待检验：**新的unique事件、明确update/exposure schedule、互斥决策与结构容量分别能带来多少改进。当前结果未建立收敛或全局最优，不能将一个族的曝光响应推广至所有架构。

## 下一步计划

收齐B2扩展seed及B3 compute，完整41模型/133配对、55压力路径、图表与独立复核实际通过后关闭资源并执行最终accept。R2继续只设计：完整R1后冻结首族，先以数值配方匹配的D0/D1/D2区分更多unique与更多重复曝光，再按已测成本决定追加对照。
