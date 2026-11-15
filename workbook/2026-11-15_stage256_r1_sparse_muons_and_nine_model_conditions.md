# 阶段256：稀疏μ损失定位与九模型条件比较

**B1相对B4的全域μ recall损失主要集中在低occupancy：channel数<100的2,801个μ事件中，B1少识别679个，recall差−24.24个百分点，配对CI95[−25.76,−22.41] pp。**该区间对全域μ recall贡献−6.79 pp；1,000–4,999区间则贡献+1.53 pp。同一模型在不同可见响应区间的得失方向不同，为下一轮稀疏事件控制提供了具体目标。

## 阶段目标

对stage254发现的μ代价作可加分解及物理块区间检验；完成新加入B1/B2后的45条条件路径统计和独立核算。补充本轮四pass结果的解释，不更改冻结排名。

## 工作内容

新增`r1.occupancy_review`，使用相同Tune40k、同一组400个物理块draw，直接从保存的预测计算逐类正确差和共同分母贡献。每个draw检验四个occupancy区间的贡献之和等于全域差。保存模型各自正确人数及paired disagreement人数，以核对错误位置。

对HGB、B1、B2、B4的T3/T4和B4-MAE T4共9模型，完成5条件×9模型=45路径、16项模型配对，保留名义cut和重新选FPR的两种结果；执行新独立核算，生成两张英文PDF/PNG。

## 详细数据

### 稀疏μ的同事件定位

Train/模型与stage254完全相同，都是seed42、四pass。Occupancy定义为原accepted输入的`pixel_channels+bar_channels+tracker_channels`，在任何移除或扰动之前计算。三类比较只使用真e/μ/NC；τ保留在原始人数中，不重标为NC。

下表为B1 T3−B4 T3。区间使用400次accepted-family连接的物理块重采样、seed20261116；40k Tune、6,845块，与全域比较采用相同draw。

| Accepted channels | μ events | B1 correct | B4 correct | B1 only / B4 only correct | Within-domain Δrecall, pp [CI95] | Contribution to all μ Δrecall, pp [CI95] |
| --- | ---: | ---: | ---: | --- | --- | --- |
| 0–99 | 2801 | 326 | 1005 | 67 / 746 | −24.241 [−25.762,−22.411] | −6.790 [−7.298,−6.252] |
| 100–999 | 3653 | 2246 | 2421 | 303 / 478 | −4.791 [−6.258,−3.348] | −1.750 [−2.264,−1.214] |
| 1000–4999 | 3336 | 2308 | 2155 | 357 / 204 | +4.586 [+3.107,+5.960] | +1.530 [+1.030,+1.999] |
| ≥5000 | 210 | 118 | 108 | 22 / 12 | +4.762 [−0.944,+10.483] | +0.100 [−0.020,+0.220] |

四行正确人数差合计−691，对全域μ recall贡献−6.91 pp。每个draw的可加恒等式最大误差2.78×10⁻¹⁷；逐类全域点估计与先前完整评估一致。全部stratum/class有400个有效draw。0–99区间包含241个全空μ事件，两模型对这些事件均预测NC、正确差0；已定位的差额来自有响应的稀疏事件。

同一区间还存在明显类别取舍，完整e/μ/NC差值及区间见[r1_occupancy_localization.json](../roadmap/v2/r1_occupancy_localization.json)。Occupancy与几何、入射条件、接受度相关；本分解定位实际损失，不识别单独的因果机制。它是开发Tune上的事后描述性分层，事件区间条件于当前训练模型。

**待检验假设：**投影下采样/池化、稀疏轨迹结构的表达和共享分类头的决策边界，可能影响这些事件。下一轮应保留全域和逐类验收，检验稀疏表示、geometry/native分支或观测条件专家，避免只优化fiducial域平均BA。具体原因尚未测量。

### 九模型压力测试

固定Tune1,600事件、每类400，1,356个物理块、0个跨chunk块。400个共享draw，seed20261117。Noise/combined沿用已经接受的proxy幅度；这是开发条件，不将幅度解释为实测探测器系统误差界。其定义保存在[r1_stress_nine_models.json](../roadmap/v2/r1_stress_nine_models.json)。

下表为相对同模型nominal的共同BA3变化，单位百分点。移除项固定模型权重，是依赖性诊断；重训增量价值已在stage254另测。

| Model | Noise ΔBA3 [CI95] | Combined ΔBA3 [CI95] | Remove pixel ΔBA3 [CI95] | Remove tracker ΔBA3 [CI95] |
| --- | --- | --- | --- | --- |
| B1 T3 | −0.167 [−0.847,+0.577] | −0.750 [−1.992,+0.545] | −3.000 [−4.567,−1.295] | −5.417 [−7.560,−3.065] |
| B1 T4 conditional | +0.167 [−0.253,+0.666] | −0.833 [−2.094,+0.271] | −2.667 [−4.197,−1.129] | −5.583 [−8.053,−3.254] |
| B2 T3 | −0.417 [−1.827,+0.995] | −0.417 [−1.743,+0.750] | −1.917 [−3.329,−0.382] | −6.667 [−8.876,−4.141] |
| B2 T4 conditional | −0.417 [−1.418,+0.670] | −0.333 [−1.353,+0.727] | −2.333 [−3.562,−1.037] | −8.083 [−10.410,−5.591] |

B2−B1的combined退化差为T3 +0.333 pp [−1.546,+1.928]、T4 +0.500 pp [−1.096,+1.996]，当前区间包含零。B1 T3−B4 T3在combined下的BA差仍为+2.833 pp [+0.438,+5.153]，但这是1,600事件子集的描述性结果，不替代40k全域排名。

MAE−Scratch的noise/combined退化差仍为−3.000 pp [−5.196,−0.916]、−2.917 pp [−5.152,−0.709]，与stage250一致。预训练参照在这组已知条件下的脆弱性继续成立；其训练及推理硬件、数值读回差异继续保留。

### 固定τ cut与重选FPR

使用完整40k nominal Tune的原cut。小子集的nominal实际FPR可以偏离1%，下表保留原始通过人数（各有400信号/1200背景）。

| Model / condition | Selected τ | Selected non-τ | τ efficiency | Actual FPR |
| --- | ---: | ---: | ---: | ---: |
| B1 nominal | 19 | 17 | 4.75% | 1.4167% |
| B1 noise | 19 | 18 | 4.75% | 1.5000% |
| B1 combined | 18 | 16 | 4.50% | 1.3333% |
| B2 nominal | 10 | 18 | 2.50% | 1.5000% |
| B2 noise | 7 | 11 | 1.75% | 0.9167% |
| B2 combined | 3 | 8 | 0.75% | 0.6667% |

B2 combined的固定cut效率差−1.75 pp [−3.164,−0.518]，实际FPR也下降−0.833 pp [−1.419,−0.332]；重新选相同FPR的效率差为+0.25 pp [−1.244,+1.535]。这说明固定score阈值的漂移与相同背景接受率下的判别损失需要分别评价。

### 数值读回、验证与资源

B1 nominal相对完整Tune保存概率的最大绝对差为T3 0.0006344、T4 0.0004765；B2分别0.0038429/0.0037330。新增四模型在1,600事件上的完整argmax及条件三类argmax改变均为0；T4的1%/0.1% τ cut通过变化均为0。B1主训练在T4/FP32、诊断在H100 1g MIG/FP32；B2主训练与诊断均H100/BF16 encoder+FP32 head。允许TF32及实际batch形状/执行差异均保留，读回点差不解释为扰动效应。

4,580项独立核算通过，最大绝对差1.443×10⁻¹⁵，覆盖45路径的点估计、固定cut人数和16项配对条件差值。统计1582907使用1 CPU/3GB、allocation88 s；审计1582908同资源、13 s；occupancy定位1582909同资源、39 s。三项均终态4、ExitCode0。

Occupancy脚本直接计算正确差，验证每个draw贡献可加、逐类全域点估计、pair disagreement计数和400个有效draw；绘图语法及紧凑JSON读回通过。图表`R1/figures/r1_occupancy_localization.{pdf,png}`与`R1/interim/stress_nine_models/figures/r1_interim_stress.{pdf,png}`已目视核查。

```bash
python -m analysis.flavour.v2.r1.occupancy_review --root "$R1_ROOT"
python -m analysis.flavour.v2.r1.stress_evaluate \
  --root "$R1_ROOT/interim/stress_nine_models" --replicates 400
python -m analysis.flavour.v2.r1.review \
  --root "$R1_ROOT/interim/stress_nine_models" --stress-only
python roadmap/tools/plot_r1_occupancy_localization.py --root "$R1_ROOT"
python roadmap/tools/plot_r1_interim_stress.py \
  --root "$R1_ROOT" --interim stress_nine_models
```

使用LCG108/Python3.12、PYTHONPATH=`src`、BLAS/OpenMP线程1。完整统计在results；仓库仅保留代码、小型结果和报告。Cal-A/Cut-B/Confirm未读取。

01:52:33 UTC的累计GPU allocation时间为24.575057 h，仍在运行；按A100 10.509356 h、H100 7.018889 h、H100 1g MIG 0.549078 h、H100 3g MIG 2.685833 h、T4 3.652734 h、早期失败V100 0.063333 h及未取得CUDA型号的历史h100-47gb分区0.095833 h分别记录。异构时间不作为等效FLOPs；最终成本尚未测量，历史MAE预训练另计。

## 阶段性结论

**事实：**投影模型的μ代价具有清楚的occupancy结构；其高occupancy优势不能补偿低occupancy损失。45条条件路径已完成并经独立点估计核算，B1/B2之间的noise/combined鲁棒性排序尚未建立。固定τ阈值可随条件漂移，即使重选FPR后的判别能力没有明确变差。

**研究判断：**R2应检查训练规模和收敛；后续几何/融合实验需单列有响应的低occupancy样本，保留全域逐类质量和固定cut表现。当前结果支持定位下一问题，尚不足以把损失归因于某一具体结构或改变冻结的架构排名。

## 下一步计划

完成B3 T3后确定另一强族并扩必要seed；完成同allocation计算控制和余下10条条件路径。执行完整名义/条件统计、55路径独立审计、终态资源记账和R1验收，再交付最终报告及下一阶段执行建议。
