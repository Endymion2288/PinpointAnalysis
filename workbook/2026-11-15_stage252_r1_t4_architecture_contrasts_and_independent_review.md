# 阶段252：完整四类架构对比、逐类取舍与独立读回

**seed42下，B1 ResNet相对B4 Scratch的共同三类BA提高1.82个百分点，400次配对CI95为[1.34,2.25] pp；τ在1% FPR处的效率提高2.40个百分点，[1.81,3.02] pp。**同时，μ recall降低5.30个百分点，[−6.10,−4.41] pp。平均性能提高与逐类退化同时存在；完整T3/T4架构排名和多seed矩阵尚未完成。

## 阶段目标

在已完成的全Tune四类模型上比较图像、native与summary方法，定位总体指标背后的取舍，独立核对概率表、曝光、条件域及新增τ先验情景。

## 工作内容

完成B1/B2/B3/B4/B4-MAE/B0六个T4模型的共同评价。主对照为B1相对其余五模型，all/pixel-centered/large各一组，共15项配对、每项400次物理块bootstrap。它们是完整排名前的描述性中期比较，架构选择仍使用冻结的Scratch T3/T4共同BA均值。

新增`analysis.flavour.v2.r1.review`独立从概率数组和人数计算：事件/label/family与域metadata、逐类均值BA/NLL、每个cut的实际通过人数/FPR/效率、不同prior的purity、Train覆盖与逐类曝光。该路径不调用正式`Metrics`、`domain_report`或`quick_report`计算指标。完整模式另核对排名和55条条件路径，已接入依赖控制器自动提交和最终验收；allocation终态仍由统计任务结束后的`accounting/accept`检查。

最终统计增加逐模型GPU、精度和allocation记录。三seed的sample SD将明确为实际种子与运行配方共同变化，不能从混用FP32/BF16和设备的三次训练单独识别RNG方差。新增`accounting --cached-kubeflow`，在当前凭据过期且所有历史Pod已有终态证据时直接读缓存，避免重复认证提示；非终态或缺失证据继续保留unresolved。

## 详细数据

### 样本与实际训练

全Tune40k、四类各10k、共同非τ30k。pixel-centered为3,765，原标签e/μ/τ/NC人数930/1,050/757/1,028；large为18,002，4,424/4,764/4,136/4,678。域内指标统一三/四类benchmark prior，保留原始混合结果。接受的family连接后有6,845个物理块，0个跨chunk块；按class signature分层、400次共同draw、seed=20261116。

五项神经网络均使用完整100k Train、每事件4次，四类曝光各100k。B1实际Tesla T4/FP32；B2 H100 NVL/BF16；B3 A100/BF16；B4 H100 MIG3g.47gb/BF16；MAE A100/BF16。B0为CPU HGB、100k监督样本；MAE历史SSL成本另列。所有区间条件于本seed、已选配方和实际精度，是开发描述性区间，不是最终确认或训练seed总体区间。

| T4 model | Common BA3 | NLL3 | BA4 | NLL4 | τ efficiency at 1% FPR |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1 ResNet | 0.564933 | 0.866249 | 0.446225 | 1.187792 | 0.0509 |
| B2 fixed-graph ParticleNet | 0.556700 | 0.891792 | 0.434475 | 1.215185 | 0.0282 |
| B3 geometry ParT | 0.552900 | 0.893037 | 0.432775 | 1.214732 | 0.0270 |
| B4 Scratch | 0.546767 | 0.902870 | 0.424575 | 1.223468 | 0.0269 |
| B4-MAE | 0.547200 | 0.890817 | 0.430325 | 1.212520 | 0.0241 |
| B0 HGB | 0.545300 | 0.897246 | 0.431025 | 1.214964 | 0.0283 |

### B1相对共同基线的实际配对

BA和效率差值均为百分点，NLL为原单位。CI为pointwise 95%，这些中期多项比较未进行确认阶段Holm检验。

| Reference | ΔBA3 [CI95], pp | ΔNLL3 [CI95] | Δτ efficiency at 1% FPR [CI95], pp |
| --- | --- | --- | --- |
| B2 | +0.823 [0.482,1.183] | −0.02554 [−0.02962,−0.02188] | +2.270 [1.743,2.952] |
| B3 | +1.203 [0.844,1.567] | −0.02679 [−0.03091,−0.02326] | +2.390 [1.848,3.016] |
| B4 | +1.817 [1.342,2.247] | −0.03662 [−0.04043,−0.03244] | +2.400 [1.813,3.021] |
| B4-MAE | +1.773 [1.350,2.197] | −0.02457 [−0.02817,−0.02132] | +2.680 [2.130,3.158] |
| HGB | +1.963 [1.483,2.432] | −0.03100 [−0.03470,−0.02687] | +2.260 [1.630,2.930] |

相对B4，B1的BA4为+2.165 pp，[1.726,2.555]；NLL4为−0.03568，[−0.03896,−0.03234]。本seed的τ候选效果值得继续验证，但它不是已通过独立确认的物理结果。相对HGB的共同三类BA点差1.963 pp尚未达到冻结的2 pp实际效果目标，且μ recall有退化，不能宣布三类晋级。

### μ取舍的域定位

共同三类argmax下，B1/B4全域μ recall为46.94%/52.24%，对应4,694/5,224个正确μ事件。e recall差+4.33 pp，[3.526,5.069]；NC +6.42 pp，[5.719,7.159]；μ −5.30 pp，[−6.101,−4.410]。

| Domain | True μ | ResNet recall | Scratch recall | ResNet−Scratch |
| --- | ---: | ---: | ---: | ---: |
| Large scintillator | 4,764 | 70.6969% | 65.9740% | +4.7229 pp |
| Outside large scintillator | 5,236 | 25.3247% | 39.7441% | −14.4194 pp |
| Tracker active | 6,692 | 58.9809% | 70.2481% | −11.2672 pp |
| Tracker empty | 3,308 | 22.5816% | 15.8102% | +6.7715 pp |
| Occupancy <100 channels | 2,801 | 10.7819% | 27.5973% | −16.8154 pp |

large域的μ差值有配对CI95[+3.594,+5.857] pp；其他细分在这里是读回点估计，尚未测量其专门区间。outside-large由all−large的原始支持/正确人数相减，B1/B4为1,326/2,081个正确μ，共同分母5,236。上述域重叠，不能把各行当独立来源相加。pixel-centered的共同BA3分别74.1848%/72.9932%；large为69.0375%/66.5936%。

**假设：**图像下采样和late-fusion对低占用、外围或tracker主导事件的使用，与B4存在互补。这不是已经证明的网络内部因果解释。当前结果具体支持检查这些域的分支表示和训练目标；不支持把native整体弃用，也不支持通过切除困难域提高全域分数。

### 独立数值与曝光读回

先在已有五模型controls上完成90项检查，再在六个T4模型上完成283项，均通过。后者跨模型、各域BA最大绝对误差9.215×10⁻¹⁵，NLL 4.885×10⁻¹⁵，工作点1.099×10⁻¹⁴，新增prior purity误差0。逐类曝光与训练计数器一致，预测metadata与冻结Tune一致。

已生成并目视核查两组英文PDF/PNG：`R1/interim/architectures_seed42/figures/r1_interim_architectures`给出三项配对区间；`r1_interim_muon_domains`给出带μ分母的域取舍。紧凑结果为[r1_interim_architectures.json](../roadmap/v2/r1_interim_architectures.json)；完整预测、区间和独立检查留在results。

### 执行与资源

```bash
python -m analysis.flavour.v2.r1.evaluate \
  --root "$R1_ROOT/interim/architectures_seed42" \
  --config "$R1_ROOT/interim/architectures_seed42/config.json" \
  --replicates 400 --workers 8
python -m analysis.flavour.v2.r1.review \
  --root "$R1_ROOT/interim/architectures_seed42" \
  --config "$R1_ROOT/interim/architectures_seed42/config.json"
python roadmap/tools/plot_r1_interim_architectures.py --root "$R1_ROOT"
python -m analysis.flavour.v2.r1.accounting --root "$R1_ROOT" --cached-kubeflow
```

环境沿用LCG108/Python3.12，PYTHONPATH=`src`，BLAS/OpenMP线程为1。大样本统计均为Condor：1582897用8 CPU/16GB，进程78.146 s、allocation87 s；独立controls审计1582896用1 CPU/3GB、allocation24 s；六T4审计1582898同为1 CPU/3GB、14 s，均终态4、ExitCode0。

00:37:24 UTC累计组织性device-hours为19.465937，仍有主训练和诊断运行/排队；A100 8.003246、H100 6.218012、T4 2.399679、已知H100 MIG2.685833、仅知flavor的历史终态上界0.095833、失败V100 0.063333。异构小时不作FLOP等效，历史SSL另列。原18 h规划已超过；必要实验继续完成，最终成本尚未测量。

单个依赖控制器已重启加载stage251的配对配置；它继续等待B1/B3 T3等完整任务，再按冻结规则排名和扩seed。Kubeflow凭据仍需恢复，当前没有未关闭的Kubeflow allocation；Condor保持推进。Cal-A/Cut-B/Confirm未读取。

## 阶段性结论

**事实：**本seed下，ResNet在共同BA、NLL及τ工作点上均优于当前B4，但μ全域召回率有明确取舍，且这种取舍随域改变。native方法没有在这组四pass同输入比较中形成全面优势。独立人数/概率/曝光读回支持当前数值。

**研究判断：**下一步选择应同时考虑平均任务、逐类要求和全域接受度。R2仍需研究是否收敛；R3/R4应针对已观察到的几何/可见性差异设计单因素检验。平均BA接近或低于图像对照，不能单独说明原生细粒度信息无效。

## 下一步计划

完成剩余主T3、fresh同allocation计算控制和排名后必要的三seed扩展；补齐剩余30条stress路径。随后执行完整400次配对统计、`review --require-full`、英文图表、终态资源审查和最终R1报告。当前不启动R2训练。
