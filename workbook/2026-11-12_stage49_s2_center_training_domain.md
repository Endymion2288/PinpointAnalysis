# 阶段49：中心训练域、覆盖与条件选点

按冻结的全域validation NLL选点，ParticleNet full-domain训练相对center-only训练的全域τ AP增量为+0.05449639，pointwise 95%事件区间=[+0.04020247,+0.06874907]；固定粗E/vertex支持后为+0.03210627，区间=[+0.01756088,+0.04732816]。全域和large区域的1%背景接受率下τ效率差值也有正区间。中心区域本身的AP差值区间跨0，不能据全域结论宣称full训练在每个区域均优胜。

## 阶段目标

完成center-only训练控制，补齐full/center/large训练域的交叉评价。分开检查unique训练数据、重复曝光、类别权重、区域内学习过程和全域选点；给出统一条件下可以支持的结论。

## 工作内容

`pn_native_centertrain_s42_e4_v2`对应`bigbird20.cern.ch`的`1573782.21`，JobStatus=4、ExitCode=0、NumJobStarts=1，实际H100节点`b9pgpun003.cern.ch`。四轮完整执行，14项CPU checkpoint/optimizer/逐轮概率读回全部pass；按full-validation NLL选epoch2。复用既有`conditional_pairs`和`domain_curves`，未新增模型或改变选点规则。

输入仍为native pixel+bar、ParticleNet dim64/161,156参数、同平面4 mm/每块最多32 nodes的共同局部learned compression，处理全部全局块。Seed42、4 epochs、batch4、20,016 attempted steps/epoch、H100 NVL/LCG108/Python3.12.11/Torch2.7.1/BF16。Validation保持全10,000 events、每类2,500；calibration/test未使用。

中心域仍为既有pixel边界内缩20 mm、目标前500 mm，large为scintillator横向边界内缩15 mm、下游内缩200 mm。这是现有geometry flag，不声明与历史报告fiducial完全相同。

## 详细数据

### 训练数据与预算

| Train domain | Unique events | e / mu / tau / NC | Four-epoch exposures | Exposures / unique event |
| --- | ---: | --- | ---: | ---: |
| Full | 80,000 | 20,000 / 20,000 / 20,000 / 20,000 | 320,164 | 4.00205 |
| Center only | 7,429 | 1,903 / 2,099 / 1,423 / 2,004 | 319,233 | 42.97119 |
| Large only | 35,936 | 8,864 / 9,479 / 8,341 / 9,252 | 320,039 | 8.90580 |

中心各轮exposures=79,817/79,811/79,810/79,795；80,064 applied updates、0 AMP skips。Loss按各自训练域计数使用inverse-class-frequency权重，中心四类权重=0.97595901/0.88482611/1.30516514/0.92677146，full均为1。不能将差异简单归因于未加权的类别比例；固定更新步数也未隔离unique样本量、覆盖、重复次数和权重。

| Unique train domain | Mean native pixel+bar channels | Median | p90 | p99 |
| --- | ---: | ---: | ---: | ---: |
| Full | 1088.69814 | 550 | 2822 | 6669.01 |
| Center only | 3088.16274 | 2592 | 5607 | 12093.20 |
| Large only | 1953.32716 | 1543 | 3826.5 | 8109.60 |

这张表从既有catalog读取unique train事件，未重新抽样。中心平均通道数为full的2.83656倍；它是原生读出占用，不是compressed token数、逐batch实际曝光占用或GPU耗时分解。

### 冻结checkpoint的全域结果

| Metric | Center-only, epoch2 | Full, epoch4 |
| --- | ---: | ---: |
| Tau AP | 0.30060835 | 0.35510474 |
| Fixed E/vertex support tau AP | 0.28992203 | 0.32202830 |
| Tau ROC AUC | 0.55289477 | 0.63150771 |
| NLL | 1.89692973 | 1.23930563 |
| Balanced accuracy | 0.3773 | 0.3974 |
| Macro F1 | 0.37440741 | 0.39204149 |
| Brier | 0.87103286 | 0.67434646 |

| Class | Center-only AP | Full AP |
| --- | ---: | ---: |
| CC nu_e | 0.53574037 | 0.59882218 |
| CC nu_mu | 0.34384143 | 0.41459288 |
| CC nu_tau | 0.30060835 | 0.35510474 |
| NC | 0.29212502 | 0.36957463 |

所有AP点估计提高，不代表所有argmax recall提高：NC recall为center-only 0.4420、full 0.2844，τ recall分别0.1976/0.5664。非τ、综合指标及recall差值CI未测量。

400次、128个accepted-family/chunk blocks（每类32）的配对bootstrap；固定common support为8,559 events、Neff=7,959.41365，保持既有粗E/vertex权重，未控制target Z/同位素。区间为pointwise，条件于当前训练模型与固定权重。

| Full minus center-only | Difference [95% CI] |
| --- | --- |
| Raw tau AP | +0.05449639 [+0.04020247,+0.06874907] |
| Fixed-support tau AP | +0.03210627 [+0.01756088,+0.04732816] |
| Tau ROC AUC | +0.07861293 [+0.06236274,+0.09454607] |
| Tau efficiency at 1% FPR | +0.01800000 [+0.00576943,+0.02511136] |
| Tau efficiency at 0.1% FPR | −0.00160000 [−0.00492485,+0.00523576] |

Raw/common AP区间均支持至少+0.01的当前模型收益。全域1%阈值下center-only选38τ/75背景，full选83/75；中心模型阈值0.41309738。0.1%分别7/7和3/7，尾部背景统计不足。中心模型的electron/muon/hadronic τ在inclusive 1%阈值下通过2/2/34个，共同分母445/395/1,660；full为7/4/72，分支差值CI未测量。

### 条件域配对与逐轮选点

| Domain | Center-only AP | Full AP | Full−center AP [95% CI] |
| --- | ---: | ---: | --- |
| Pixel-centered: 885 events / 176 tau | 0.33893120 | 0.31582532 | −0.02310588 [−0.07607109,+0.03040053] |
| Large: 4494 events / 1053 tau | 0.33751861 | 0.38642610 | +0.04890749 [+0.02692902,+0.07352384] |

中心域AP区间较宽，不能确认优劣，也不能排除对任一方有实质收益。其1%阈值下center-only/full分别通过7/1个τ、各7背景，尾部统计不足。Large域分别21/50个τ、各34背景，full−center效率差=+0.02754036，CI=[+0.01018262,+0.04255754]；AUC差=+0.02574052，CI=[+0.01277125,+0.04078930]。条件域使用各自经验阈值和观测prior，未重新匹配E/vertex，不把跨域AP绝对值作配对。

| Center-only epoch | Full NLL | Center NLL | Large NLL | Full tau AP | Center tau AP |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.93434090 | 1.11418915 | 1.32005878 | 0.28695782 | 0.33573690 |
| 2 | 1.89692973 | 1.05824922 | 1.19777005 | 0.30060835 | 0.33893120 |
| 3 | 2.34644926 | 0.99970189 | 1.25736956 | 0.30492740 | 0.34312602 |
| 4 | 2.94781724 | 1.13733903 | 1.68057926 | 0.28865826 | 0.37063087 |

中心域自身NLL最低在epoch3，全域和large域在epoch2；posthoc own-domain epoch3的中心AP=0.34312602，较full的0.31582532高0.02730070，仅为点估计，无CI，不替代主比较。Epoch4中心AP更高但NLL变差，说明AP排序与概率log-loss衡量不同方面，未按AP重新选点。

利用NLL的逐事件可加性，中心域外9,115 events的NLL=(10000×full NLL−885×center NLL)/9115，四轮为2.01397165/1.97835948/2.47720861/3.12360146；没有对AP作此线性分解。三种训练域的完整条件曲线均保存，full各区域NLL都在epoch4最低，large-only自身域在epoch4最低但主选epoch1，参见阶段46。

### 资源、产物与验证

Center-only程序wall=9,432.52941 s，allocation=9,440 s=2.62222222 GPUh，额外7.47059 s；peak allocated memory=967.49316 MiB。Full为6,310 s、1,593.10254 MiB。中心模型实际分配耗时高49.60%、峰值显存较低，不能将较少unique事件直接解释为固定步数下更便宜。更高占用与重复训练是需要考虑的因素，节点/I/O和局部压缩成本的因果分解未测量。

共同allocation预算6,309.67797 s时，center可用的NLL最优checkpoint仍是epoch2，full为epoch4，ΔAP仍+0.05449639；这是整轮分辨率点估计，无独立CI。四轮成本全部计入，不按所选早期checkpoint截去已执行训练。10:45:54 UTC组织83.26137263 GPUh，H100视图47.02303930 GPUh，missing history records=0。

正式产物根目录`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/s2_2026-11-11/h100/`。`center_training_comparison.json`保存主/条件配对、三域学习曲线、manifest/history/readback/成本及真实terminal；`conditional_center_training.*`、`domain_learning_curves_complete.*`为已实际查看的英文图。`domain_training_design.json`记录三域样本、权重和曝光，`domain_train_occupancy.json`记录unique train占用，`domain_outside_center_nll.json`记录可加NLL补域。日志为`readback_pn_native_centertrain.log`、`accounting_center_training.log`、`evaluation_center_training.log`、`conditional_center_training.log`和`domain_learning_curves_complete.log`。

主评价commit=`ac374f0`，本批31 results=4 references+27科学配置，835 strata、93历史三类表、31 paired/14 pending、2 factorial、0 seed means。核算/评价/target/inventory顺序进程真实exit0，target刷新至217 model×Z组，inventory为27 output_complete、5 running、3 not_started，strict gate=false。中心模型四分类诊断页及独立配对图已实际查看；运行快照为`stage49_execution_snapshot.json`。严格科学验收仍未完成。三模型共12轮full NLL/AP重算与history一致，主checkpoint条件指标与独立条件分析一致；本阶段复用现有计算入口，没有改核心代码、没有重跑已通过的完整回归套件。

```bash
python -m analysis.flavour.s2.readback --root "$S2_RESULTS/h100" \
  --campaign "$PWD/roadmap/configs/s2_h100_v2.json" --experiments pn_native_centertrain_s42_e4_v2
python -m analysis.flavour.s2.accounting --root "$S2_RESULTS"
python -m analysis.flavour.s2.evaluate --root "$S2_RESULTS/h100" \
  --output "$S2_RESULTS/h100/evaluation_final" \
  --comparisons "$PWD/roadmap/configs/s2_complete_pairs_v2.json" --allow-incomplete
python -m analysis.flavour.s2.conditional_pairs --root "$S2_RESULTS/h100" \
  --a pn_native_centertrain_s42_e4_v2 --b pn_native_joint_s42_e4_v2 \
  --output "$S2_RESULTS/h100/conditional_center_training.json"
python -m analysis.flavour.s2.domain_curves --root "$S2_RESULTS/h100" \
  --experiments pn_native_joint_s42_e4_v2 pn_native_centertrain_s42_e4_v2 pn_native_largetrain_s42_e4_v2 \
  --output "$S2_RESULTS/h100/domain_learning_curves_complete.json"
```

## 阶段性结论

事实：当前完整目标域与冻结选点下，full训练优于center-only的τ AP和1%工作点；large区域也支持这一方向，中心区域的差值尚未确定。三域控制已量化训练覆盖与选点对结果的影响。

解释范围：该控制同时改变unique样本量、重复曝光、覆盖和loss权重，不隔离单一空间原因；两份历史三类/投影研究与当前四类/表示的差异也未全部消除。仍保留full作为完整目标域参照，记录中心条件域的不确定结论。

## 下一步计划

继续既有coarse/ParticleNet seeds及plane-only作业，补齐四组三seed差值与跨层效应。所有35项终态、14项读回、41单seed配对和4三seed均值齐备后执行无allow-incomplete的严格评价、完整输出门槛和最终S2报告；不开始S3–S6。
