# 阶段259：R1三seed任务证据与按吞吐调度

**B1三seed支持保留四类主线。**相对B4，τ在开发1% FPR工作点的平均效率提高2.423 pp [1.941,2.977]；实际三类重训相对四类条件三类只提高0.259 pp BA3，并有e/NC取舍。B1 T3相对HGB的平均BA3增益为1.973 pp，未达到预设2 pp实用门槛。B2两项seed已在更快的2g MIG上从fresh初始化继续，完整R1尚未验收。

## 阶段目标

量化冻结排名的事件不确定度；核对B1完整三seed是否支持单seed结论；按实际剩余完成时间改进资源使用，同时保持单writer和同allocation计算控制。

## 工作内容

新增`r1.ranking_review`，直接读取同一40k Tune的八项主模型概率，独立计算T3/T4条件三类BA的平均，并在所有族、任务之间复用相同400次物理块draw。四项点值均与冻结排名一致，排序和扩seed选择不变。

将已经complete的B1、B4、HGB共18个三seed主模型组成中期评估，保留原最终合同中的40项配对，未完成的计算控制和B2不进入该次统计。HTCondor1582923用8 CPU完成400次统计，allocation72 s、实际统计60.52 s。独立读回1582924核对概率、身份、计数、曝光和工作点，573项全部通过，最大工作点差1.113e−14。

`plots --stress-only`可在nominal完整交付前绘制已完成的55条压力路径，并补充T4图例；B1三seed图复用可选择family/interim目录的绘图工具。两份英文PNG已目视核对，PDF保存在同目录。

资源侧新增`r1.handoff`：在新的实际allocation中用完整Train输入、相同seed、LR=0的512 batches测吞吐，不做Tune推理；达到门槛后等待显式release。release绑定新allocation ID，要求旧writer已终止且未完成输出已归档，再启动fresh T4/T3/compute。profile用于资源选择，不作为模型学习证据。已有真实学习门槛与主训练健康检查继续适用。

## 详细数据

### 冻结排名的配对区间

seed42、固定已拟合权重，400次accepted-family连接物理块draw、seed20261116、6,845块；下列为T3/T4 conditional BA3平均差的pointwise开发区间，不含重新训练波动。

| Comparison | Mean BA3 difference | 95% interval |
| --- | ---: | --- |
| B1 − B2 | +1.1617 pp | [+0.8348,+1.4426] pp |
| B1 − B3 | +1.3817 pp | [+1.0703,+1.6938] pp |
| B1 − B4 | +2.0433 pp | [+1.6644,+2.4224] pp |
| B2 − B3 | +0.2200 pp | [−0.0468,+0.4944] pp |
| B2 − B4 | +0.8817 pp | [+0.5220,+1.2548] pp |
| B3 − B4 | +0.6617 pp | [+0.2941,+1.0327] pp |

实测见[r1_ranking_uncertainty.json](../roadmap/v2/r1_ranking_uncertainty.json)。B1的当前开发领先有区间支持；B2/B3的差仍包含零。四pass和有限LR扫描不代表全局最优。

### 三seed主任务

Train pool100k，每类25k；T3训练75k非τ、T4训练100k全类，各四pass。Tune40k，每类10k。seeds17/42/2026；下表SD是已实现seed与运行配方的sample SD。B1 seed42使用T4/FP32，另两seed使用H100 1g MIG/BF16 encoder+FP32 head，不能把SD解释为纯RNG变化。

| Model | Mean BA3 | Sample SD BA3 | Mean NLL3 |
| --- | ---: | ---: | ---: |
| B1 T3 | 0.565633 | 0.002401 | 0.870377 |
| B1 T4 conditional | 0.563044 | 0.001721 | 0.867105 |
| B4 T3 | 0.543922 | 0.000674 | 0.902022 |
| B4 T4 conditional | 0.544867 | 0.002150 | 0.904369 |
| HGB T3 | 0.545900 | 0 | 0.897449 |
| HGB T4 conditional | 0.545300 | 0 | 0.897246 |

HGB在本数据和配方下三个seed实际输出一致，不提供额外随机训练变异。

以下区间是在同一物理块draw内先算各seed差，再平均；条件于这三组已拟合权重，训练波动单列。

| Comparison | Metric | Point difference | 95% interval |
| --- | --- | ---: | --- |
| B1 T3 − B1 T4 conditional | BA3 | +0.2589 pp | [+0.0418,+0.4790] pp |
| Same | NLL3 | +0.003272 | [+0.001661,+0.004801] |
| Same | e recall | +2.080 pp | [+1.751,+2.459] pp |
| Same | mu recall | −0.020 pp | [−0.393,+0.327] pp |
| Same | NC recall | −1.283 pp | [−1.636,−0.966] pp |
| B1 T3 − B4 T3 | BA3 | +2.1711 pp | [+1.8078,+2.5198] pp |
| Same | mu recall | −2.410 pp | [−3.209,−1.690] pp |
| B1 T3 − HGB T3 | BA3 | +1.9733 pp | [+1.5132,+2.3782] pp |
| B1 T4 − B4 T4 | BA3 | +1.8178 pp | [+1.4576,+2.1782] pp |
| Same | tau efficiency at 1% FPR | +2.4233 pp | [+1.9408,+2.9774] pp |
| Same | tau efficiency at 0.1% FPR | +0.8367 pp | [+0.5031,+1.1502] pp |
| Same | NLL4 | −0.037649 | [−0.040768,−0.034205] |
| B1 T4 − HGB T4 | tau efficiency at 1% FPR | +2.1900 pp | [+1.6640,+2.8273] pp |

B1 T4在三个seed的τ1%效率为5.13%、5.09%、4.84%，平均5.02%；实际开发FPR均为1%。这些cut在同一个开发Tune上选取，未开放Cal-A/Cut-B/Confirm。平衡benchmark的τ purity约62%，在πτ=0.001且非τ等比例的固定响应情景约0.5%；后者是prior情景，不是实测flux。

预设研究条件按原值逐项检查：B1 T3对B4因μ非劣未满足，对HGB因1.9733 pp小于2 pp未满足；B1 T4对B4/HGB均满足本轮τ条件。τ条件约束共同BA而不包含逐类μ非劣；B1 T4对B4的μ平均差仍为−1.853 pp [−2.611,−1.063]。满足τ条件不代表逐类全面改善。

T4相对T3的BA3区间下界为−0.4790 pp，在当前开发区间内满足−0.5 pp总体非劣界；e recall仍低2.080 pp。此结果支持同时交付真实T3与四类概率，继续保留τ研究，而非宣称两任务完全相同。

完整逐seed值、mean-seed配对、先验情景和逐项条件见[r1_b1_three_seed.json](../roadmap/v2/r1_b1_three_seed.json)。中期完整结果在`R1/interim/b1_three_seed/evaluation/`；英文图为`figures/r1_reference_seeds.{pdf,png}`。

### 压力路径的新增读回

全部55路径已在stage258独立核算。本阶段绘制完整英文图`R1/figures/r1_stress.{pdf,png}`。B3 T3/T4移除tracker的BA3差分别为−6.583 pp [−8.773,−4.239]、−6.417 pp [−8.664,−4.030]；移除pixel分别为+0.500 pp [−1.352,+2.164]、+0.167 pp [−1.159,+1.588]。这些是固定权重移除诊断，不能替代重训增量实验。

B3从训练A100到诊断1g MIG的nominal读回最大概率差为0.004043/0.002643；T3 argmax改变0个，T4改变1个，T4 τ的1%/0.1% cut选择改变0/1个。条件差均减去同一诊断路径的nominal，避免把跨设备/形状数值差当成条件效应。

### B2按实际剩余完成时间迁移

1g MIG上的B2主训练约22 events/s；同seed、同完整Train前512 batches的2g profile为37.4113（seed17）和34.6244（seed2026）events/s，activation均finite。初始40 events/s启发式拒绝了两个35–37的probe；按实际余量投影，seed17保持1g约13.07 h训练，fresh2g约8.23 h，预计节省4.84 h。这是1100k近似总曝光与线性吞吐估算，不含未来实际验证/启动；最终wall尚未测量。因此将新的资源探针门槛改为32，未使用模型分数选择设备。

| Seed | Old cluster / terminal allocation | New cluster / device | Handoff evidence |
| --- | --- | --- | --- |
| 17 | 1582914, removed, 2616 s | 1582921, H100 2g.24gb MIG | `audit/b2_s17_handoff_stage259.json` |
| 2026 | 1582915, removed, 2929 s | 1582922, H100 2g.24gb MIG | `audit/b2_s2026_handoff_stage259.json` |

两个旧job均有eviction/abort日志和history JobStatus=3，未完成T4目录移到`failed_attempts/`后才写release；新T4 manifest实际allocation分别为1582921.0/1582922.0。T3和fresh compute将在各自同一新allocation连续执行。旧运行费用和四个探针attempt全部计入资源。B1仍在1g执行compute，不为统一型号迁移。

默认GPU准入仍为10,000 MiB、当前LCG108兼容Capability，不限定型号或整卡。20,000 MiB只用于这组寻找更快替代资源的probe。历史S2已有LCG107的V100兼容环境；已修正文档，不再把它写成未知的未来条件。当前R1没有切换到LCG107，其当前模型性能与兼容性未测量。

03:31:39 UTC资源快照为33.102248 allocation device-hours，仍有5项GPU训练；分区小时继续分别记录。完整成本未关闭，原18 h规划已经超过。Cal-A/Cut-B/Confirm继续封存，R2未启动。

## 验证命令与产物

LCG108/Python3.12、`PYTHONPATH=src`、OpenMP/BLAS线程1：

```bash
python -m analysis.flavour.v2.r1.ranking_review --root "$R1_ROOT"
python -m analysis.flavour.v2.r1.evaluate --root "$R1_ROOT/interim/b1_three_seed" \
  --config "$R1_ROOT/interim/b1_three_seed/evaluation_config.json" --replicates 400 --workers 8
python -m analysis.flavour.v2.r1.review --root "$R1_ROOT/interim/b1_three_seed"
python -m analysis.flavour.v2.r1.plots --root "$R1_ROOT" --stress-only
python roadmap/tools/plot_r1_reference_seeds.py --root "$R1_ROOT" \
  --family B1 --interim b1_three_seed
python -m unittest tests.test_flavour_v2_r0
python -m unittest tests.test_flavour_v2_r1 tests.test_flavour_v2_r1_summary
git diff --check
```

前三项经HTCondor执行，图表读取小型结果本地生成。8项R0与21项R1/汇总测试、修改模块语法检查通过；573项真实独立读回通过。所有输出根目录为`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R1`。补充R2表的标签口径：100k/300k/1M表示四类pool，T3实际unique监督标签为75k/225k/750k，不能把未训练的τ标签计入T3预算。

## 阶段性结论

**事实：**B1三seed对HGB/B4的τ开发收益稳定为正，整体共同三类损失较小，但逐类仍有代价。T3平均增益相对HGB略低于预设实际效果值，对B4则存在μ退化。B2/B3排序未显示明确间隔，二者均未建立训练饱和。按实际吞吐进行迁移可以选择更合适的MIG，而无需等待同型号或整卡。

**待检验假设：**更长训练、更多unique事件与合适决策是否能同时改善B1的逐类取舍，需R2实际对照；早期stride/稀疏压缩的因果作用需R3单因素验证。当前事件区间不覆盖训练种子总体，也不作为独立Confirm结论。

## 下一步计划

完成5项在运行allocation的必要操作，补齐41模型、133配对的nominal结果；完成full独立核算、全部英文图目视核对及终态资源记账后运行accept。仅在验收complete后导出`r1_results.json`，形成R1最终报告和R2冻结配方。唯一控制器仍为PID2052979，不另建第二个控制器。
