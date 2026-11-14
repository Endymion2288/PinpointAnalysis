# R1：完整数据验收与正式基线启动

**阶段目标**

完成共同 Train100k/Tune40k 的转换验收，启动正式 B0–B4 比较，并验证配对统计与图表的完整入口。R1 的四轮训练、三seed、stress和最终bootstrap尚未完成。

产物根目录：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R1`。

**工作内容与详细数据**

56/56分片已转换并通过CPU Condor finalization（1582865）。共140,000事件，Train每类25,000、Tune每类10,000；物理键与family逐行吻合冻结manifest，labels与targets吻合，无重复物理事件。输入/输出总PE均为 **606,862,092,543**；四类Parquet合计 **2,635.024 MiB**。各分片转换耗时之和34,820.391秒，属于并行worker耗时累计，不能当作单一job wall time。最终证据为`preparation.json`。

实际几何域分母：

| Split | All | Pixel-centered | Large scintillator | Empty all modalities | Tracker active |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 100,000 | 9,421 | 44,960 | 2,846 | 56,083 |
| Tune | 40,000 | 3,765 | 18,002 | 1,083 | 22,450 |

Tune pixel-centered四类人数为930/1050/757/1028；large为4424/4764/4136/4678，顺序e/μ/τ/NC。条件域并非类别平衡，报告必须保留raw counts，并在比较AP/NLL/purity时显式统一benchmark prior。31项B0观测summary全部finite；完整范围、分位数、τ分支与域分母由CPU审计`data_audit.json`给出。

正式提交记录：HGB campaign1582866、固定stress准备1582867、B1/B2/B3 screen1582868/1582869/1582870，均在现场兼容且WRITE可用的最空闲schedd `bigbird20.cern.ch`；B4 screen在Kubeflow `pinpoint-r1-screen-b4-k11`，实际H100 MIG3g.47gb。Condor GPU任务无exact型号要求；根据实测RSS调低等待任务的CPU/RAM请求，并允许capability≥7卡以FP32运行。修改前后请求与原因保存在`audit/queue_resize_a1.json`和`flexible_precision_scheduling_a1.json`。

HGB初始实际结果（均为全Tune40k上的共同30k非τ；未附统计区间）：

| Training task | LR | Train unique | Train BA3 | Tune BA3 | Tune NLL3 | CPU fit seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T3 | 0.05 | 75,000 | 0.657573 | 0.545900 | 0.897449 | 11.3253 |
| T4 conditional T3 | 0.05 | 100,000 | 0.646147 | 0.545300 | 0.897246 | 17.5807 |

这是同数据summary基线，尚不能据此判断三类优于四类；两LR的选择使用冻结的Tune1k/class子集，不使用全Tune另行选点。神经族仍在screen，不给出完整架构排名。

统计与诊断入口补充如下：

- 400次paired bootstrap继续使用accepted-family-connected物理source blocks；每次重采样重新规范到均匀类别先验，避免块大小造成类别混合变化。保存共同BA/NLL/AP/AUROC、recall、各类1%/0.1% FPR与80%/90% purity效率差值；几何域配对另列。
- main、matched-compute与without-tracker使用不同记录类型，训练seed均值只纳入main。每族两任务四pass完成后按冻结规则排名，再扩前两族与B4至三seed。
- fixed small Tune的noise/combined同时给描述性ROC与**冻结全nominal Tune阈值**下的效率、实际FPR变化。paired去pixel/tracker仅解释reliance；B1/B4无tracker真实重训才估计增量价值。combined从ideal tracker crossings同RNG重数字化。
- 同计算控制记录精确allocation ID、目标/实际operation wall time、逐行exposures、updates与schedule。GPU型号/MIG分区记账，不将混合device-hours换算为FLOP等价量。
- 实现完整领域表、ROC/PR、raw scores、资源与学习曲线、stress英文图，以及R1交付验收。验收通过不代表已经达到全局最优或科学晋级门槛。

**验证结果**

30项相关单元测试通过，包含固定先验的配对bootstrap、相同预测的精确零差、以及扰动后保留nominal阈值时FPR升高的检查。额外Train-only图表集成使用之前8个模拟validation事件，完整跑通8模型、3项几何域配对、400次bootstrap、五种诊断条件和全部英文图；人为复用相同预测因此stress差值为零。`smoke/graphics_a1/integrity.json`标记`actual_Tune_used=false`，该结果不用于科学比较。实际查看了输出图，资源图调整为独立图例，避免标签重叠。

```bash
python -m analysis.flavour.v2.r1.b0campaign --root R1_OUT --threads 4
python -m analysis.flavour.v2.r1.campaign screen --root R1_OUT --families B4 --compile-local
python -m analysis.flavour.v2.r1.campaign select --root R1_OUT --families B4
python -m analysis.flavour.v2.r1.extend --root R1_OUT --family B4 --seed 42 --compile-local
python -m analysis.flavour.v2.r1.evaluate --root R1_OUT --config R1_OUT/evaluation_config.json
python -m analysis.flavour.v2.r1.stress_evaluate --root R1_OUT
python -m unittest tests.test_flavour_v2_r1 tests.test_flavour_spatial tests.test_flavour_s3_repair
```

**阶段性结论**

完整共同输入已通过验收，summary baseline已实际训练并产生非恒定预测。正式比较不再受输入准备阻塞。Train/Tune的明显差距提示HGB存在泛化损失；是否来自容量、先验/域差异或训练样本分布，仍需后续配对分析。当前没有证据宣布任何神经族最优、τ不可识别或模态无效。

**下一步计划**

逐族完成screen后立即衔接四pass与同allocation计算控制；所有Scratch族完成后确定前两名并扩seed，运行MAE及无tracker控制。随后真实执行完整统计、stress、资源汇总与交付验收，并在报告中给出三类近期交付和四类后续投入的证据。
