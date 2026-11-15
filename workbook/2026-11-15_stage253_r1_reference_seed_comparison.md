# 阶段253：B4三seed共同任务、HGB参照与训练取舍

**B4三个实际seed下，T3−T4条件三类的平均BA差为−0.094个百分点，配对CI95为[−0.334,+0.086] pp。**T3的NLL较低，平均差−0.002347，[−0.003613,−0.001109]。平均BA接近，但逐类recall的任务差值随seed有明显变化。当前没有建立“去掉τ训练能普遍提高三类效果”的结论。

## 阶段目标

使用新完成的B4 seed17，与既有seed42/2026一起检验共同任务差异及HGB参照；区分固定训练模型的事件区间与训练/执行变化。

## 工作内容

对B4与B0的T3/T4、seeds17/42/2026共12个已训练模型计算全Tune结果。三项T3−T4及六项同seed HGB−B4比较，共9项配对、400个共享物理块draw。由各seed的指标差值取均值，保留逐seed差值和sample SD；没有平均概率，也没有构造ensemble。

完成12模型的独立概率/人数/曝光读回，并生成英文PDF/PNG，直接并列展示实际seed差异与条件于这三个已训练模型的事件区间。

## 详细数据

### 数据、曝光和运行条件

Train为同一100k四类pool；T3使用全部75k非τ。B4 T3每个非τ事件4次、总300k曝光，T4每事件4次、总400k曝光。Tune40k、四类各10k，共同非τ30k，6,845个accepted-family连接的物理块、0个跨chunk块。400次按class signature分层的共同重采样，seed20261116。

seed17在A100、seed42在H100 MIG3g.47gb、seed2026在H100整卡；各seed内部的T3/T4同allocation、均为BF16 encoder/FP32 head。sample SD包含实际seed和设备执行差异；本矩阵不能单独识别纯RNG总体方差。事件区间条件于这三个已训练模型，不包含重新训练的不确定度，也没有对seed总体作推断。

### B4逐seed与平均结果

| Seed | T3 BA3 | T3 NLL3 | T4 conditional BA3 | T4 NLL3 | T3−T4 BA3, pp |
| --- | ---: | ---: | ---: | ---: | ---: |
| 17 | 0.544700 | 0.902666 | 0.542533 | 0.907383 | +0.216667 |
| 42 | 0.543500 | 0.901241 | 0.546767 | 0.902870 | −0.326667 |
| 2026 | 0.543567 | 0.902159 | 0.545300 | 0.902855 | −0.173333 |
| Mean | 0.543922 | 0.902022 | 0.544867 | 0.904369 | −0.094444 |

BA3的sample SD为T3 0.067440 pp、T4 0.214968 pp；配对BA差的SD为0.280126 pp。三个seed各自T3 NLL均低于T4，差值−0.004716/−0.001629/−0.000696，但这是同曝光比较，不是同计算结论；seed17的fresh计算控制仍在运行。

三项T4的BA4为0.421625/0.424575/0.423450，均值0.423217；NLL4均值1.225322。τ在1% FPR处的效率为2.40%/2.69%/2.70%，均值2.596667%。这些是每个模型分别选开发工作点的指标均值，不是ensemble性能或独立确认效率。

### 逐类取舍及区间的含义

下表为T3−T4条件三类recall，单位均为百分点。

| Metric | Seed17 | Seed42 | Seed2026 | Mean | Conditional event CI95 | Sample SD |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Electron recall | +2.52 | −4.27 | −3.56 | −1.7700 | [−2.0884,−1.4682] | 3.7322 |
| Muon recall | −1.73 | +4.65 | −1.31 | +0.5367 | [+0.1403,+0.8840] | 3.5684 |
| NC recall | −0.14 | −1.36 | +4.35 | +0.9500 | [+0.5543,+1.3339] | 3.0070 |

三项recall差值在seed间都改变符号。不能将较窄的mean-seed事件区间当作“重新训练仍会得到相同逐类取舍”的证据。当前平均BA较稳定，不能替代逐类稳定性检查。

### HGB仍是强共同参照

HGB三次实际拟合的BA3/NLL3在本配方下相同：T3为0.545900/0.897449，T4条件三类为0.545300/0.897246。`HistGradientBoostingClassifier`配置固定300 iterations、关闭early stopping；三个seed在这些指标上没有产生差异，不将零SD解释为一般模型训练没有不确定度。

| Mean contrast | ΔBA3, pp [CI95] | ΔNLL3 [CI95] |
| --- | --- | --- |
| HGB T3 − B4 T3 | +0.1978 [−0.2712,+0.5816] | −0.004573 [−0.007271,−0.001103] |
| HGB T4 − B4 T4 | +0.0433 [−0.4247,+0.3936] | −0.007123 [−0.009932,−0.004163] |

HGB−B4四类BA差为+0.7808 pp，[+0.4111,+1.1641]，NLL4差−0.010358，[−0.012608,−0.007744]；τ在1% FPR处效率差+0.2333 pp，[−0.2525,+0.6989]。这些是固定三seed的开发比较，不是确认阶段检验。B4尚未体现相对观测summary的整体优势；这不能推出原生信息无效或四pass已收敛。

### 独立验证、图表与产物

381项独立检查通过。BA/NLL/工作点最大绝对误差分别为1.077×10⁻¹⁴、5.107×10⁻¹⁵、1.113×10⁻¹⁴，prior purity误差0；事件、family、label、条件域metadata、逐类曝光与原计数器一致。

完整结果：`R1/interim/reference_seeds/evaluation/results.json`；独立审计同目录`independent_review.json`。紧凑结果为[r1_reference_seeds.json](../roadmap/v2/r1_reference_seeds.json)。英文图`R1/interim/reference_seeds/figures/r1_reference_seeds.{pdf,png}`已目视核查。

```bash
python -m analysis.flavour.v2.r1.evaluate \
  --root "$R1_ROOT/interim/reference_seeds" \
  --config "$R1_ROOT/interim/reference_seeds/config.json" \
  --replicates 400 --workers 8
python -m analysis.flavour.v2.r1.review \
  --root "$R1_ROOT/interim/reference_seeds" \
  --config "$R1_ROOT/interim/reference_seeds/config.json"
python roadmap/tools/plot_r1_reference_seeds.py --root "$R1_ROOT"
```

使用LCG108/Python3.12、PYTHONPATH=`src`，BLAS/OpenMP线程1。统计通过Condor1582899执行，8 CPU/16GB、进程57.435 s、allocation67 s；独立审计1582900，1 CPU/3GB、allocation17 s；均终态4、ExitCode0。

### 其余R1进度与资源

B1 seed42 T3完整四pass和40k推理已完成，BA3=0.566200、NLL3=0.871967，逐类recall为0.6301/0.4998/0.5687；该项的完整配对、三seed和同计算结论待交付。B2十条条件诊断1582895已完成，allocation188 s、ExitCode0，总条件覆盖35/55。B1诊断1582901已提交，由现有控制器根据真实完成状态触发。

01:07:30 UTC累计组织性device-hours为21.526759，仍有主计算控制、B3 T3和后续依赖；最终成本未测量。资源按实际型号/分区分别记录，异构小时不作FLOP等效。Kubeflow凭据仍未恢复，当前无active Kubeflow allocation，Condor继续执行。Cal-A/Cut-B/Confirm未读取。

## 阶段性结论

**事实：**B4三seed均值下，三类重训没有显示明确的BA优势，NLL略好；逐类差值的seed变化远大于平均BA差。HGB保持相近的共同BA和较好的NLL。

**研究判断：**类别数不应根据一个seed或平均BA单独决定。继续保留四类作为研究输出，同时完成实际三类交付及其τ污染评价。R2需回答训练是否充分；几何/融合改进还要同时观察逐类质量与稳定性。

## 下一步计划

继续完成B3完整T3，严格按冻结规则确定前两族并扩seed；保留B4共同参考。补齐全部同allocation计算控制和余下20条条件路径，执行最终统计、完整独立审计、英文图表、资源终态与R1验收。当前不启动R2训练。
