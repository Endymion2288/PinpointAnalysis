# 阶段246：R1首组完整B4三类/四类与tracker配对结果

**Tracker对共同三类识别有明确增益；目前三类重训与四类条件三类的整体BA接近，但逐类取舍不同。B4尚未显示对summary HGB的全面优势。**本阶段只完成seed42 B4及HGB的中期比较，R1完整架构排名、三seed和同计算控制继续运行。

## 阶段目标

使用已完成的真实四pass模型，先回答tracker是否增加信息，以及移除τ训练是否改善共同三类。以物理块配对统计替代单点比较，为后续完整R1及R2提供可核对证据，不改变冻结的架构排名或seed扩展规则。

## 工作内容

读取B4 T3、T4、T3 without tracker和对应HGB的全部Tune分数。对全域、pixel-centered、large scintillator分别计算统一三类先验的指标、τ污染和400次配对物理块区间。计算在HTCondor1582887执行，8CPU/8 workers，实际统计wall48.234 s，正常终态。每个bootstrap使用相同权重作用于所有模型。

完整产物：`/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R1/interim/b4_seed42/evaluation/`。小型读回为[r1_interim_b4.json](../roadmap/v2/r1_interim_b4.json)。英文图表由`roadmap/tools/plot_r1_interim_b4.py`生成，位于上述中期目录的`figures/r1_interim_b4.pdf/png`。

## 详细数据

### 样本、输入与统计

Train T4为100,000 unique events、400,000 exposures；T3为其中全部75,000非τ事件、300,000 exposures。所有模型共用40,000 Tune事件，四类各10,000；共同三类分母为30,000。Pixel-centered分母3,765，原标签e/μ/τ/NC人数为930/1,050/757/1,028，共同三类3,008。Large分母18,002，人数为4,424/4,764/4,136/4,678，共同三类13,866。

几何域按冻结的实际sensor/bar坐标定义：pixel-centered为pixel边缘内缩20 mm、target起点后首500 mm；large为scintillator边缘内缩15 mm、排除target末端200 mm。真vertex仅用于事后域评价。各域NLL/AP/AUC使用统一三类先验，原人数单列；这些cuts尚未证明与合作组历史研究完全一致。

Bootstrap使用6,845个run/chunk物理块并按class signature分层，accepted-family连接已应用，本Tune中multi-chunk连接块为0。区间条件于本次训练的seed42和权重；训练随机性尚未由本表量化。Cal-A/Cut-B/Confirm未打开。

### 共同三类实测

| Model | All BA3 | All NLL3 | Pixel-centered BA3 | Large BA3 |
| --- | ---: | ---: | ---: | ---: |
| B4 T3 | 0.543500 | 0.901241 | 0.729443 | 0.661576 |
| B4 T4, conditional T3 | 0.546767 | 0.902870 | 0.729932 | 0.665936 |
| B4 T3 without tracker | 0.504533 | 0.934803 | 0.665271 | 0.608607 |
| HGB T3 | 0.545900 | 0.897449 | 0.727671 | 0.661994 |
| HGB T4, conditional T3 | 0.545300 | 0.897246 | 0.724828 | 0.660160 |

B4 T3/T4在同一H100 NVL MIG3g.47gb allocation顺序执行，wall分别1,604.216/2,024.190 s；无tracker在H100 NVL执行，wall2,122.267 s。上述成本包含各自操作的加载、训练与验证；去tracker的wall不作跨设备计算效率结论。HGB为CPU方法，CPU线程/boosting预算单列。

以下差值和区间均以**百分点**表达；BA越大越好。

| Contrast | All ΔBA3 [95% interval] | Pixel-centered | Large |
| --- | --- | --- | --- |
| B4 T3 − B4 T4 conditional | −0.327 [−0.689, +0.004] | −0.049 [−0.837, +0.699] | −0.436 [−0.873, −0.022] |
| B4 T3 − retrained without tracker | +3.897 [+3.431, +4.407] | +6.417 [+4.979, +7.720] | +5.297 [+4.595, +5.993] |
| B4 T3 − HGB T3 | −0.240 [−0.675, +0.267] | +0.177 [−0.868, +1.354] | −0.042 [−0.538, +0.573] |
| B4 T4 − HGB T4 conditional | +0.147 [−0.260, +0.660] | +0.510 [−0.635, +1.618] | +0.578 [−0.036, +1.168] |

Tracker重训对照的全域ΔNLL3=−0.033562，CI95 [−0.036840,−0.030540]。本结果支持当前输入下tracker的增量价值；冻结权重后直接删除输入的诊断仍须单列为reliance。

B4 T3−T4的全域ΔNLL3=−0.001629，CI95 [−0.003614,+0.000350]。但argmax recall分别变化：e −4.27 pp [−4.86,−3.75]；μ +4.65 pp [+4.02,+5.26]；NC −1.36 pp [−2.01,−0.73]。同Tune描述性1%非τ背景工作点下，μ效率提升1.09 pp [+0.28,+1.76]、NC提升0.61 pp [+0.06,+1.39]，e提升0.04 pp [−0.83,+0.90]。因此BA接近不表示逐类行为等价。

B4 T3−HGB T3的全域ΔNLL3=+0.003792 [0.000293,0.006745]。B4 T4−HGB T4的ΔNLL4=+0.008504 [0.005900,0.010870]，描述性1% FPR下Δτ效率=−0.001400 [−0.006366,+0.003264]。原始概率的对数损失目前更支持保留HGB为强参照，不能把复杂编码器视为已经胜出。

### τ进入三类输出时的行为

B4 T3对10,000个真τ强制argmax分配到e/μ/NC为2,426/3,982/3,592；T4去除τ概率并重归一化后为2,923/3,500/3,577。此处没有将τ重标为NC。

T3在三个分别选取的1%非τ背景cut下，τ接受率为2.24%/3.40%/2.32%。四类均匀benchmark中，相应选中样本的τ占比为9.18%/18.07%/16.45%；若假设τ先验0.001、其余三类均匀，保持条件响应不变，占比分别为0.0303%/0.0662%/0.0591%。这些是先验情景，尚无真实flux归一化；完整阈值、四类raw selected counts和0.1%工作点在结果JSON中。

### 复现与验证

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R1_ROOT=/eos/home-x/xcheng/FASER/Pinpoint/results/flavour/v2_2026-11-14/R1
# 统计通过HTCondor执行；下列是实际payload入口。
python -m analysis.flavour.v2.r1.evaluate --root "$R1_ROOT/interim/b4_seed42" \
  --config "$R1_ROOT/interim/b4_seed42/config.json" --replicates 400 --workers 8
python roadmap/tools/plot_r1_interim_b4.py --root "$R1_ROOT"
```

`evaluate`逐个检查Tune事件/family身份及标签一致性后输出5个模型、12项配对，bootstrap metadata及原分数保持可追溯。没有据中期结果改变主排名、学习率或实验矩阵。

## 阶段性结论

**事实：**本配方、四次共同非τ曝光、seed42下，tracker提高全域及两个几何域的BA3。三类重训没有显示整体BA提高，同时改变逐类recall和固定FPR效率。Summary HGB与B4的BA相近，HGB原始NLL更低。

**研究判断：**τ并非当前共同三类效果的唯一限制，简单减少类别不能替代表示和训练研究。三类与四类继续并行，待三seed和同总计算控制后决定实际交付配方。Tracker保留为后续强基线输入，后续R3/R4再研究几何关联及重建对象的额外价值。

**未测量：**完整B1–B4排名、三seed任务差异、同总计算T3结果、MAE净收益、55条stress及最终独立确认。当前区间不作为全局最优、模型等价或最终部署工作点结论。

## 下一步计划

继续完成冻结矩阵，按完整seed42排名扩展前两名及B4。使用全部三seed、计算控制、无tracker/CNN/MAE和压力条件形成R1完整报告；R2依据实际胜出配方进行100k×12与300k×4比较，检验训练和数据规模是否带来额外收益。
